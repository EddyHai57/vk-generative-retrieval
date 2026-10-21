"""Bounded P engineering tools; synthetic fixtures, no server jobs."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_p_full_backend import native, own_job_fixture


def test_whitelist_and_pin_harvesting(tmp_path):
    m = importlib.import_module('analysis.p_full_engineering')
    value = m.input_roles()
    from analysis.ur001_science_entry import A_ROOT, admit_inputs
    assert len(value['train']) == 25 and len(value['metadata']) == 3
    assert value['validation'] == str(A_ROOT / 'subsamples/ur0.01/validation/week_25.parquet')
    inputs = tmp_path / 'inputs.json'; inputs.write_text(json.dumps(value))
    assert admit_inputs(str(inputs)) == m.backend.file_sha(inputs)
    run = tmp_path / 'producer'; (run / 'logs').mkdir(parents=True)
    manifest = tmp_path / 'manifest.json'
    source = {'commit': 'a' * 40, 'files': {'example.py': 'b' * 64}}
    runtime = {'runtime': {'python': '3.11.16'}, 'lock_sha256': 'c' * 64}
    manifest.write_text(json.dumps({'source': source, 'runtime': runtime}))
    receipt = {'verified': True, 'manifest_sha256': m.backend.file_sha(manifest)}
    for name, content in [('run.json', '{}'), ('terminal.json', '{}'), ('logs/stdout.log', 'progress\n' + json.dumps(receipt) + '\n')]:
        (run / name).write_text(content)
    pin = m.harvest_pin(manifest, run, m.backend.file_sha(manifest))
    assert set(pin) == {'manifest_sha256', 'source', 'runtime', 'job_sha256', 'receipt'}
    assert pin['receipt'] == receipt and pin['source'] == source and pin['runtime'] == runtime
    assert pin['job_sha256'] == {name: m.backend.file_sha(run / name) for name in ('run.json', 'terminal.json', 'logs/stdout.log')}
    with pytest.raises(ValueError, match='pinned'):
        m.harvest_pin(manifest, run, '0' * 64)


def test_measured_template_fields_cannot_drift():
    m = importlib.import_module('analysis.p_full_engineering')
    value = {'source': None}; m.measure(value, 'source', {'commit': 'a'})
    assert value['source'] == {'commit': 'a'}
    m.measure(value, 'source', {'commit': 'a'})
    with pytest.raises(ValueError, match='template source'):
        m.measure(value, 'source', {'commit': 'b'})


def test_capture_compares_native_steps_and_scoring(native):
    m = importlib.import_module('analysis.p_full_engineering')
    data = native.fr.synthetic_data()
    config = {**native.default_config(), 'batch_size': 8}
    first = m.capture(native, data, config, 'cpu', lambda: None)
    second = m.capture(native, data, config, 'cpu', lambda: None)
    assert first == second and len(first['steps']) == 3
    # Independent first-step control calls the untouched native function.
    import random
    from analysis.tensor_cache_parity import freeze
    model = native._model(data, config, 'cpu'); optimizer = native.make_optimizer(model, config)
    rows, _ = native.fr._view(data, config['history_len'])
    training = importlib.import_module(native.__package__ + '.retrieval_training')
    negatives = importlib.import_module(native.__package__ + '.retrieval_negatives')
    cursor = training.EpochCursor(len(rows), config['seed']); stats = training.SamplerStats()
    batch = [rows[i] for i in cursor.next(config['batch_size'])]
    record = native.fr.train_step(model, optimizer, batch, data, stats, cursor.epoch, config,
             random.Random(f"p-b1-negative-v1:{config['seed']}"), None, negatives.target_frequency(rows))
    assert first['steps'][0]['record'] == freeze(record)
    assert first['steps'][0]['parameters'] == freeze(model.state_dict())
    assert first['steps'][0]['optimizer'] == freeze(optimizer.state_dict())
    assert set(first['steps'][0]) == {'tensors', 'record', 'parameters', 'optimizer', 'cursor', 'negative_rng'}
    assert set(first['evaluation']) == {'users', 'items', 'norms', 'scores', 'original_evaluator'}
    changed = {**data, 'rows': [{**r, 'side': [[a, 1 if b == 255 else b + 1, c] for a, b, c in r['side']]} for r in data['rows']]}
    assert m.capture(native, changed, config, 'cpu', lambda: None) != first


def test_cli_dry_and_invalid():
    for cmd, status in [('dry', 0), ('invalid', 2)]:
        got = subprocess.run([sys.executable, '-B', '-m', 'analysis.p_full_engineering', cmd], capture_output=True, text=True)
        assert got.returncode == status
        if not status:
            assert json.loads(got.stdout)['training_submitted'] is False


@pytest.mark.parametrize('for_parity', [False, True])
def test_prepare_closes_measured_originals_without_submit(tmp_path, monkeypatch, for_parity):
    m = importlib.import_module('analysis.p_full_engineering')
    import analysis.ur001_derive_cli as derive
    import analysis.new_instance_prepare as environment
    from types import ModuleType
    dag = ModuleType('analysis.producer_dag_binding'); seen = []
    dag.verify_producer = lambda *args: seen.append('producer')
    monkeypatch.setitem(sys.modules, dag.__name__, dag)
    source = {'commit': 'a' * 40, 'files': {'example.py': 'b' * 64}}
    runtime = {'runtime': {'python': '3.11.16'}, 'lock_sha256': 'c' * 64}
    monkeypatch.setattr(derive, 'source_identity', lambda: source)
    monkeypatch.setattr(derive, 'verify_dag', lambda *a, **kw: seen.append('dag'))
    monkeypatch.setattr(environment, 'runtime', lambda *a: runtime)
    config = {'seed': 42}
    monkeypatch.setattr(m, 'frozen_config', lambda *a: config)
    metadata = []
    for name in ('users', 'items', 'embeddings'):
        path = tmp_path / name; path.write_bytes(b'metadata'); metadata.append(str(path))
    roles = {'schema': 'ur001-science-inputs-v1', 'train': [], 'validation': '', 'metadata': metadata}
    monkeypatch.setattr(m, 'input_roles', lambda: roles)
    monkeypatch.setattr(m.backend, 'admit_inputs', lambda path: m.backend.file_sha(path))
    lock = tmp_path / 'lock'; lock.write_bytes(b'lock')
    stage2 = tmp_path / 'stage2'; stage2.mkdir()
    manifest = stage2 / 'UR001-manifest.json'; manifest.write_text(json.dumps({'source': source, 'runtime': runtime}))
    run = tmp_path / 'run'; (run / 'logs').mkdir(parents=True)
    for name in ('run.json', 'terminal.json', 'logs/stdout.log'):
        (run / name).write_text('{}\n')
    value = {'schema': m.backend.SCHEMA, 'predecessor': None, 'native_root': str(tmp_path / 'native'),
             'stage2': str(stage2), 'stage2_run': str(run), 'ops_root': str(tmp_path / 'ops'),
             'output_root': str(tmp_path / 'science-output'), 'run_ids': {'preflight': 'pf', 'train': 'train', 'verify': 'verify'},
             'config': None, 'source': None, 'runtime': None, 'items_metadata': None, 'embedding': None,
             'admission_sha256': None, 'lock': {'path': str(lock), 'sha256': m.backend.file_sha(lock)},
             'producer_pin': {'manifest_sha256': m.backend.file_sha(manifest)}}
    template = tmp_path / 'template.json'; template.write_text(json.dumps(value))
    def plan(stage, inputs, request, digest):
        assert stage == 'preflight'
        loaded = json.loads(request.read_bytes())
        assert loaded['source'] == source and loaded['runtime'] == runtime and loaded['config'] == config
        assert loaded['producer_pin']['job_sha256']['run.json'] == m.backend.file_sha(run / 'run.json')
        assert loaded['admission_sha256'] == m.backend.file_sha(inputs)
        assert loaded['items_metadata']['path'] == metadata[1]
        seen.append('plan'); return {'status': 'planned_dry', 'request_sha256': digest, 'stage': 'preflight',
                                    'expected_job': {'argv': ['original']}, 'launcher_argv': ['--', 'original']}
    monkeypatch.setattr(m.backend, 'plan', plan)
    output = tmp_path / 'prepared'
    receipt = m.prepare(template, m.backend.file_sha(template), tmp_path / 'config', 'd' * 64, output, for_assembly_parity=for_parity)
    assert seen == ['producer', 'dag', 'plan']
    planned = json.loads((output / 'plan.json').read_bytes())
    if for_parity:
        argv = m.parity_argv(output / 'inputs.json', output / 'request.json', receipt['request_sha256'])
        assert planned['expected_job']['argv'] == argv and planned['launcher_argv'] == ['--', *argv]
        assert planned['stage'] == 'assembly-parity'
    else:
        assert planned['stage'] == 'preflight'
    assert receipt['training_submitted'] is False and receipt['real_preflight'] is False
    assert receipt['request_sha256'] == m.backend.file_sha(output / 'request.json')
    assert receipt['plan_sha256'] == m.backend.file_sha(output / 'plan.json')
    with pytest.raises(ValueError, match='new output'):
        m.prepare(template, m.backend.file_sha(template), tmp_path / 'config', 'd' * 64, output)


def test_reference_assembly_reads_independent_original_files(tmp_path, native):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from analysis.ur001_derivation import derive
    from analysis.tensor_cache_parity import freeze
    m = importlib.import_module('analysis.p_full_engineering')
    parts = [(0, [{'user_id': u, 'item_id': 1 + (u + k) % 20,
                   'like': True, 'share': False, 'bookmark': False, 'open_comments': False,
                   'click_on_author': False, 'dislike': False, 'timespent': 3}
                  for k in range(5) for u in range(12)])]
    rows = []
    result = derive(parts, (25, [{'user_id': u, 'item_id': u + 1, 'like': True} for u in range(12)]), rows.append)
    catalog = result['catalog']; support = result['supported']; queries = result['validation']
    features = np.zeros((len(catalog) + 1, 64), dtype=np.float32)
    adapted = m.backend.adapt_rows(native, rows, queries, catalog, support, parts, {i: 5 for i in catalog}, features)
    stage2 = tmp_path / 'stage2'; stage2.mkdir()
    for name, collection in [('train', rows), ('validation', queries)]:
        (stage2 / f'UR001-{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in collection))
    for name, collection in [('catalog', catalog), ('supported', support)]:
        (stage2 / f'UR001-{name}.json').write_text(json.dumps(collection))
    raw = tmp_path / 'week.parquet'; pq.write_table(pa.Table.from_pylist(parts[0][1]), raw)
    items = tmp_path / 'items.parquet'; pq.write_table(pa.table({'item_id': catalog, 'duration': [5] * len(catalog)}), items)
    embeddings = tmp_path / 'embeddings.npz'; np.savez(embeddings, item_id=np.array(catalog, dtype=np.uint32), embedding=features[1:].astype(np.float16))
    inputs = tmp_path / 'inputs.json'; inputs.write_text(json.dumps({'train': [str(raw)]}))
    value = {'stage2': str(stage2), 'items_metadata': {'path': str(items), 'sha256': m.backend.file_sha(items)},
             'embedding': {'path': str(embeddings), 'sha256': m.backend.file_sha(embeddings)}}
    reference = m.reference_assembly(native, value, inputs, len(rows), len(queries))
    assert freeze({**adapted, 'support': sorted(adapted['support'])}) == freeze({**reference, 'support': sorted(reference['support'])})
    # Independent source bytes changed: original assembly must detect the
    # difference against the previously assembled adapter data.
    changed = [{**r, 'timespent': 4} for r in parts[0][1]]
    pq.write_table(pa.Table.from_pylist(changed), raw)
    changed_reference = m.reference_assembly(native, value, inputs, len(rows), len(queries))
    assert changed_reference['rows'] != adapted['rows']


def test_frozen_config_requires_original_git_bytes(tmp_path, native, monkeypatch):
    m = importlib.import_module('analysis.p_full_engineering')
    root = tmp_path / 'source'; config = root / 'configs/seqm1/p-seqm1-mean-s42.json'
    config.parent.mkdir(parents=True); config.write_text(json.dumps(native.default_config()))
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'frozen'], check=True)
    monkeypatch.setattr(m.backend, 'native_for', lambda *args: native)
    assert m.frozen_config(root, config, m.backend.file_sha(config)) == native.default_config()
    config.write_text(json.dumps(native.default_config(), indent=2))
    with pytest.raises(ValueError, match='config bytes'):
        m.frozen_config(root, config, m.backend.file_sha(config))


@pytest.mark.parametrize('tamper', [None, 'seed', 'unknown', 'actual_env', 'input_bytes'])
def test_engineering_job_binding(own_job_fixture, monkeypatch, tamper):
    m = importlib.import_module('analysis.p_full_engineering')
    backend, value, records, request, digest, _ = own_job_fixture
    expected = backend.job_contract(value, records, request, digest, 'preflight')
    expected['argv'] = m.parity_argv(records[1]['path'], request, digest)
    run = Path(value['ops_root']) / 'runs' / value['run_ids']['preflight']; run.mkdir()
    for k, v in expected['env'].items(): monkeypatch.setenv(k, v)
    if tamper == 'seed': expected['seed'] += 1
    if tamper == 'unknown': expected['unbound'] = True
    if tamper == 'actual_env': monkeypatch.setenv('OMP_NUM_THREADS', '2')
    if tamper == 'input_bytes': records[-1]['bytes'] += 1
    (run / 'run.json').write_text(json.dumps(expected))
    if tamper is None:
        assert m.check_engineering_job(value, records, request, digest) == backend.file_sha(run / 'run.json')
    else:
        with pytest.raises(ValueError):
            m.check_engineering_job(value, records, request, digest)


@pytest.mark.parametrize('fault', [None, 'assembly', 'score', 'job_end'])
def test_assembly_parity_orchestration_preserves_failures(own_job_fixture, native, monkeypatch, fault):
    import copy
    m = importlib.import_module('analysis.p_full_engineering')
    backend, value, records, request, digest, _ = own_job_fixture
    config = {**native.default_config(), 'batch_size': 8}
    value.update(config=config, output_root=str(request.parent / 'output'), disk_cap_bytes=1024**3)
    data = native.fr.synthetic_data(); data['queries'] = (data['queries'] * 2)[:32]
    reference = copy.deepcopy({**data, 'rows': data['rows'][:24]})
    if fault == 'assembly': reference['rows'][0]['target'] += 1
    run = Path(value['ops_root']) / 'runs' / value['run_ids']['preflight']; run.mkdir()
    expected = backend.job_contract(value, records, request, digest, 'preflight')
    expected['argv'] = m.parity_argv(records[1]['path'], request, digest)
    for k, v in expected['env'].items(): monkeypatch.setenv(k, v)
    (run / 'run.json').write_text(json.dumps(expected))
    monkeypatch.setattr(backend, 'read_transport', lambda *a: value)
    monkeypatch.setattr(backend, 'stage_inputs', lambda *a: records)
    monkeypatch.setattr(backend, 'read_context', lambda *a: (value, native, data, {}))
    monkeypatch.setattr(m, 'reference_assembly', lambda *a: reference)
    original_context = native.b1.execution_context
    monkeypatch.setattr(native.b1, 'execution_context', lambda device: None if device == 'cuda' else original_context(device))
    monkeypatch.setattr(native.b1, 'resource_guard', lambda *a: lambda: None)
    original_capture = m.capture; calls = []
    def cpu_capture(module, selected, cfg, device, guard):
        assert device == 'cuda'; calls.append('capture')
        result = original_capture(module, selected, cfg, 'cpu', guard)
        if fault == 'score' and len(calls) == 2: result['evaluation']['scores'] = 'changed'
        return result
    monkeypatch.setattr(m, 'capture', cpu_capture)
    def close(*args):
        if fault == 'job_end': (run / 'run.json').write_text(json.dumps({**expected, 'created_at': 'new bytes'}))
    monkeypatch.setattr(backend, 'close_context', close)
    if fault:
        with pytest.raises(ValueError, match={'assembly': 'input bytes', 'score': 'scoring', 'job_end': 'job changed'}[fault]):
            m.assembly_parity(Path(records[1]['path']), request, digest)
    else:
        result = m.assembly_parity(Path(records[1]['path']), request, digest)
        assert result['verified'] is True and result['steps'] == 3
        assert result['original_formal_loader_verified'] is False and result['scientific_verified'] is False
        assert result['training_rows'] == 24 and result['validation_queries'] == 32
    assert len(calls) == (0 if fault == 'assembly' else 2)


@pytest.mark.parametrize('for_parity', [False, True, 'helper'])
def test_helper_real_launcher_chain(tmp_path, for_parity):
    """Real helper/plan/launcher in the existing complete deployment fixture."""
    import io
    import os
    import tarfile
    from test_p_full_backend import ROOT, test_cli_plan_inputs_feed_real_launcher_prepare
    root = tmp_path / 'chain'; root.mkdir()
    test_cli_plan_inputs_feed_real_launcher_prepare(root)
    deployment = root / 'deployed'
    # Keep relocated admission/inventory/pins from that real-chain fixture;
    # materialize only missing tracked dependencies from this fixed commit.
    raw = subprocess.check_output(['git', '-C', os.environ.get('P_FULL_TEST_GIT_ROOT', str(ROOT)), 'archive',
                                   'HEAD', 'analysis', 'src', 'configs'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            if not member.isfile(): continue
            target = deployment / member.name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
    helper = deployment / 'analysis/p_full_engineering.py'
    helper.write_bytes((ROOT / 'analysis/p_full_engineering.py').read_bytes())
    dag = deployment / 'analysis/producer_dag_binding.py'
    if not dag.exists():
        dag.write_bytes(subprocess.check_output(['git', '-C', os.environ.get('P_FULL_TEST_GIT_ROOT', str(ROOT)),
                        'show', 'b1628fa239cf5e6715f9515dd57158c5bf214a07:analysis/producer_dag_binding.py']))
    subprocess.run(['git', '-C', str(deployment), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(deployment), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'engineering fixture dependencies'], check=True)
    # Child process uses real source_identity, frozen_config byte checks,
    # verify_producer, prepare, backend.plan and frozen launcher.prepare.
    # Only host runtime/native-validation providers and synthetic derivation
    # payload verifier are boundaries; the fixture has no real scientific data.
    script = r'''import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from analysis import p_full_engineering as e,ur001_derive_cli as d,new_instance_prepare as n
root=Path(sys.argv[1]);parity=sys.argv[2]=='True';helperpf=sys.argv[2]=='helper'
sha=e.backend.file_sha
value=json.loads((root/'request.json').read_bytes()); native=Path(value['native_root'])
raw=__import__('subprocess').check_output(['git','-C',sys.argv[3],'show','0ca61bc:configs/seqm1/p-seqm1-mean-s42.json'])
config=native/'configs/seqm1/p-seqm1-mean-s42.json';config.parent.mkdir(parents=True);config.write_bytes(raw)
import subprocess
subprocess.run(['git','init','-q',str(native)],check=True)
subprocess.run(['git','-C',str(native),'add','.'],check=True)
subprocess.run(['git','-C',str(native),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','native fixture'],check=True)
e.backend.native_for=lambda *a: SimpleNamespace(validate_config=lambda value,**k:value)
runtime={'runtime':{'python':'3.11.16'},'lock_sha256':value['lock']['sha256']}
n.runtime=lambda *a: runtime
d.verify_dag=lambda *a,**k: {'fixture_payload_boundary':True}
source={'commit':'d'*40,'files':{'producer.py':'e'*64}}
manifest=Path(value['stage2'])/'UR001-manifest.json';mv=json.loads(manifest.read_bytes());mv.update(source=source,runtime=runtime)
manifest.write_text(json.dumps(mv))
run=Path(value['stage2_run']);job={key:None for key in ('schema','run_id','kind','cwd','argv','seed','git','config','inputs','env','resources','ops_source','overlay','pinned_files','stop_mode')}
job.update(schema='vkgr-run-v1',run_id=run.name,git={'commit':source['commit'],'dirty':False,'dirty_files':[]})
(run/'run.json').write_text(json.dumps(job))
terminal={key:None for key in ('schema','run_id','status','exit_code','signal','wall_cap_exceeded','started_at','ended_at','wall_seconds')}
terminal.update(schema='vkgr-terminal-v1',run_id=run.name,status='succeeded',exit_code=0,wall_cap_exceeded=False)
(run/'terminal.json').write_text(json.dumps(terminal))
(run/'logs/stdout.log').write_text(json.dumps({'verified':True,'manifest_sha256':sha(manifest)})+'\n')
sys.path.insert(0,str(root/'ops-source/ops'))
from vkgr_ops import launcher
ops=root/'ops-source'
subprocess.run(['git','init','-q',str(ops)],check=True)
subprocess.run(['git','-C',str(ops),'add','.'],check=True)
subprocess.run(['git','-C',str(ops),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','original OPS bytes'],check=True)
for contract in value['job_contract'].values():contract['ops_source']=launcher._ops_source()
value.update(source=None,runtime=None,config=None,admission_sha256=None,producer_pin={'manifest_sha256':sha(manifest)})
template=root/'helper-template.json';template.write_text(json.dumps(value))
output=root/('parity-plan' if parity else 'pf-plan')
receipt=e.prepare(template,sha(template),config,sha(config),output,for_assembly_parity=parity,for_helper_preflight=helperpf)
plan=json.loads((output/'plan.json').read_bytes())
sys.path.insert(0,str(root/'ops-source/ops'))
from vkgr_ops import launcher
args=launcher.build_parser().parse_args(plan['launcher_argv']);spec=launcher.spec_from_args(args);metadata=launcher.prepare(spec)
expected=plan['expected_job']
for key in ('git','config','inputs','cwd','overlay','pinned_files'):
    assert metadata[key]==expected[key],key
for key in ('argv','seed','stop_mode','run_id','kind'):assert spec[key]==expected[key],key
assert {**{k:spec[k] for k in launcher.RESOURCE_ARGS},'wall_seconds':spec['wall_hours']*3600.0}==expected['resources']
assert launcher._ops_source()==expected['ops_source']
assert metadata['env']=={k:v for k,v in expected['env'].items() if k not in ('VKGR_RUN_ID','VKGR_PROGRESS_PATH')}
dry=subprocess.run([sys.executable,'-B','-m','analysis.p_full_engineering','dry'],capture_output=True,text=True)
assert dry.returncode==0 and json.loads(dry.stdout)['training_submitted'] is False
assert not Path(value['ops_root']).exists()
print(json.dumps({'status':'helper_launcher_metadata_dry_verified','for_parity':parity,'input_count':len(metadata['inputs']),'receipt':receipt}))
'''
    got = subprocess.run([sys.executable, '-B', '-c', script, str(root), str(for_parity),
                          os.environ.get('P_FULL_TEST_GIT_ROOT', str(ROOT))], cwd=deployment,
                          env={**os.environ, 'PYTHONPATH': str(deployment / 'src')}, capture_output=True, text=True)
    assert got.returncode == 0, got.stdout + got.stderr
    receipt = json.loads(got.stdout)
    assert receipt['status'] == 'helper_launcher_metadata_dry_verified' and receipt['input_count'] > 30


def test_engineering_job_hash_is_of_validated_raw_bytes(own_job_fixture, monkeypatch):
    m = importlib.import_module('analysis.p_full_engineering')
    backend, value, records, request, digest, _ = own_job_fixture
    expected = backend.job_contract(value, records, request, digest, 'preflight')
    expected['argv'] = m.parity_argv(records[1]['path'], request, digest)
    run = Path(value['ops_root']) / 'runs' / value['run_ids']['preflight']; run.mkdir()
    job = run / 'run.json'; job.write_text(json.dumps(expected))
    validated_sha = backend.file_sha(job)
    for k, v in expected['env'].items(): monkeypatch.setenv(k, v)
    original_json = backend.strict_json
    def change_after_read(raw):
        parsed = original_json(raw)
        if parsed.get('run_id') == expected['run_id']:
            job.write_text(json.dumps({**expected, 'created_at': 'changed after validated read'}))
        return parsed
    monkeypatch.setattr(backend, 'strict_json', change_after_read)
    assert m.check_engineering_job(value, records, request, digest) == validated_sha
    assert backend.file_sha(job) != validated_sha


def test_stage_progress_is_stderr_json_only(capsys):
    from analysis.p_full_training_backend import stage_progress
    stage_progress('fixture', week='week_00', step=1)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert json.loads(captured.err) == {'stage': 'fixture', 'week': 'week_00', 'step': 1}


def test_helper_preflight_never_runs_steps(own_job_fixture, native, monkeypatch):
    import copy
    m = importlib.import_module('analysis.p_full_engineering')
    backend, value, records, request, digest, _ = own_job_fixture
    value.update(config={**native.default_config(), 'batch_size': 8}, output_root=str(request.parent / 'output'), disk_cap_bytes=1024**3)
    data = native.fr.synthetic_data(); data['queries'] = (data['queries'] * 2)[:32]
    reference = copy.deepcopy({**data, 'rows': data['rows'][:24]})
    expected = backend.job_contract(value, records, request, digest, 'preflight')
    expected['argv'] = m.helper_argv('preflight', records[1]['path'], request, digest)
    run = Path(value['ops_root']) / 'runs' / value['run_ids']['preflight']; run.mkdir()
    (run / 'run.json').write_text(json.dumps(expected))
    for k, v in expected['env'].items(): monkeypatch.setenv(k, v)
    monkeypatch.setattr(backend, 'read_transport', lambda *a: value)
    monkeypatch.setattr(backend, 'stage_inputs', lambda *a: records)
    monkeypatch.setattr(backend, 'read_context', lambda *a: (value, native, data, {}))
    monkeypatch.setattr(backend, 'close_context', lambda *a: None)
    monkeypatch.setattr(native.b1, 'execution_context', lambda *a: None)
    monkeypatch.setattr(native.b1, 'resource_guard', lambda *a: lambda: None)
    monkeypatch.setattr(m, 'reference_assembly', lambda *a: reference)
    def forbidden(*args): raise AssertionError('read-only helper preflight ran optimizer steps')
    monkeypatch.setattr(m, 'capture', forbidden)
    result = m.assembly_parity(Path(records[1]['path']), request, digest, read_only=True)
    assert result['status'] == 'preflight_verified' and result['steps'] == 0
    assert result['model_created'] is False and result['scientific_verified'] is False


def test_helper_preflight_cli_rejects_holdout_before_request(tmp_path):
    import os
    m = importlib.import_module('analysis.p_full_engineering')
    from analysis.ur001_science_entry import FORBIDDEN_ROOTS
    roles = m.input_roles(); roles['validation'] = str(FORBIDDEN_ROOTS[0] / 'forbidden.parquet')
    inputs = tmp_path / 'inputs.json'; inputs.write_text(json.dumps(roles))
    got = subprocess.run([sys.executable, '-B', '-m', 'analysis.p_full_engineering', 'preflight',
                          '--inputs', str(inputs), '--request', str(tmp_path / 'missing-request'), '--request-sha256', '0' * 64],
                         env={**os.environ, 'PYTHONPATH': 'src'}, capture_output=True, text=True)
    assert got.returncode == 2
    result = json.loads(got.stdout)
    assert result['status'] == 'HOLDOUT_INPUT_REJECTED' and result['data_opened'] is False


def test_helper_preflight_cli_dispatch_is_read_only(tmp_path, monkeypatch, capsys):
    m = importlib.import_module('analysis.p_full_engineering'); seen = []
    def readonly(inputs, request, digest, *, read_only=False):
        assert read_only is True
        seen.append((inputs, request, digest))
        return {'status': 'preflight_verified', 'steps': 0, 'model_created': False}
    monkeypatch.setattr(m, 'assembly_parity', readonly)
    inputs = tmp_path / 'inputs.json'; request = tmp_path / 'request.json'
    assert m.main(['preflight', '--inputs', str(inputs), '--request', str(request), '--request-sha256', 'a' * 64]) == 0
    assert seen == [(inputs, request, 'a' * 64)]
    assert json.loads(capsys.readouterr().out) == {'status': 'preflight_verified', 'steps': 0, 'model_created': False}
