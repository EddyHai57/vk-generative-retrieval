"""Original-producer subset rebuild; no server data or scientific jobs."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_p_full_backend import native, own_job_fixture, actual_dag_module


def fixture_inputs(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    parts = [(0, [{'user_id': u, 'item_id': 1 + (u + k) % 20,
                   'like': k % 3 != 1, 'share': False, 'bookmark': False,
                   'open_comments': False, 'click_on_author': False,
                   'dislike': False, 'timespent': 3}
                  for k in range(7) for u in range(40)])]
    validation = (25, [{'user_id': u, 'item_id': 1 + u % 20, 'like': True} for u in range(40)])
    data = tmp_path / 'raw'; (data / 'train').mkdir(parents=True); (data / 'validation').mkdir()
    raw = data / 'train/week_00.parquet'; pq.write_table(pa.Table.from_pylist(parts[0][1]), raw)
    val = data / 'validation/week_25.parquet'; pq.write_table(pa.Table.from_pylist(validation[1]), val)
    items = tmp_path / 'items.parquet'; pq.write_table(pa.table({'item_id': list(range(1, 21)), 'duration': [5] * 20}), items)
    emb = tmp_path / 'embeddings.npz'; np.savez(emb, item_id=np.arange(1, 21, dtype=np.uint32), embedding=np.arange(1280, dtype=np.float16).reshape(20, 64))
    return {'train': [str(raw)], 'validation': str(val)}, items, emb, parts, validation


def test_original_producer_loader_and_adapter_are_equal(tmp_path, native):
    m = importlib.import_module('analysis.p_subset_reference_rebuild')
    from analysis.ur001_derivation import derive
    from analysis.tensor_cache_parity import freeze
    roles, items, emb, parts, validation = fixture_inputs(tmp_path)
    output = tmp_path / 'reference'
    manifest = m.rebuild_reference(native, roles, items, emb, output, {'consumer': {'source_fingerprint': 'source'}}, lambda: None)
    assert manifest['original_parameterized_loader_verified'] is True
    assert manifest['original_legacy_wrapper_used'] is False
    data, binding = m.load_reference(native, output, manifest)
    rows = []; derived = derive(parts, validation, rows.append)
    import numpy as np
    rd = importlib.import_module(native.__package__ + '.retrieval_data')
    with np.load(emb) as values:
        features = rd.content_join(derived['catalog'], values['item_id'], values['embedding'])
    adapted = m.backend.adapt_rows(native, rows, derived['validation'], derived['catalog'], derived['supported'], parts, {i: 5 for i in derived['catalog']}, features)
    assert m.canonical_identity(data) == m.canonical_identity(adapted)
    assert binding['side_manifest_sha256'] == manifest['side_sha256']
    assert manifest['old_pins']['prepared_equal'] is False
    config = {**native.default_config(), 'batch_size': 8}
    assert m.helper.capture(native, data, config, 'cpu', lambda: None) == m.helper.capture(native, adapted, config, 'cpu', lambda: None)
    with pytest.raises(FileExistsError):
        m.rebuild_reference(native, roles, items, emb, output, {'consumer': {'source_fingerprint': 'source'}}, lambda: None)
    (output / 'side/side-train-l100.json').write_text('[]')
    with pytest.raises(ValueError):
        m.load_reference(native, output, manifest)


def test_module_cli_dry_invalid():
    for command, expected in [('dry', 0), ('invalid', 2)]:
        run = subprocess.run([sys.executable, '-B', '-m', 'analysis.p_subset_reference_rebuild', command], capture_output=True, text=True)
        assert run.returncode == expected
        if expected == 0:
            assert json.loads(run.stdout)['training_submitted'] is False


def test_reference_rejects_external_sidecar_even_when_artifact_rehashed(tmp_path, native):
    m = importlib.import_module('analysis.p_subset_reference_rebuild')
    roles, items, emb, _, _ = fixture_inputs(tmp_path)
    out = tmp_path / 'reference'
    value = m.rebuild_reference(native, roles, items, emb, out, {'consumer': {'source_fingerprint': 'source'}}, lambda: None)
    spec = json.loads((out / 'inputs.json').read_bytes()); spec['side_manifest'] = str(tmp_path / 'external-side.json')
    (out / 'inputs.json').write_text(json.dumps(spec))
    value['artifacts']['inputs.json'] = m.backend.file_sha(out / 'inputs.json')
    with pytest.raises(ValueError, match='reference.*paths'):
        m.load_reference(native, out, value)


def test_holdout_cli_rejected_before_missing_request(tmp_path):
    m = importlib.import_module('analysis.p_subset_reference_rebuild')
    roles = m.roles_for(tmp_path / 'raw', tmp_path / 'metadata')
    roles['validation'] = str(tmp_path / 'test/week_26.parquet')
    path = tmp_path / 'inputs.json'; path.write_text(json.dumps(roles))
    run = subprocess.run([sys.executable, '-B', '-m', 'analysis.p_subset_reference_rebuild', 'preflight',
        '--inputs', str(path), '--request', str(tmp_path / 'missing'), '--request-sha256', '0' * 64], capture_output=True, text=True)
    assert run.returncode == 2
    receipt = json.loads(run.stdout)
    assert 'holdout' in receipt['error'].lower() or 'forbidden' in receipt['error'].lower()


@pytest.mark.parametrize('fault', ['', 'job_end', 'input_end'])
def test_preflight_is_readonly_and_rechecks_job(tmp_path, monkeypatch, actual_dag_module, own_job_fixture, fault):
    from types import SimpleNamespace
    m = importlib.import_module('analysis.p_subset_reference_rebuild')
    backend, value, records, request, digest, _ = own_job_fixture
    value.update(output_root=str(tmp_path / 'out'), disk_cap_bytes=1024**3)
    roles = {}
    expected = m.job_contract(value, records, request, digest, 'preflight')
    run = Path(value['ops_root']) / 'runs' / expected['run_id']; run.mkdir()
    job = run / 'run.json'; job.write_text(json.dumps(expected))
    for k, v in expected['env'].items(): monkeypatch.setenv(k, v)
    monkeypatch.setattr(m, 'read_transport', lambda *a: (value, roles))
    monkeypatch.setattr(m, 'stage_inputs', lambda *a: records)
    native = SimpleNamespace(b1=SimpleNamespace(resource_guard=lambda *a: lambda: None))
    calls = []
    def context(*a):
        calls.append(1)
        if len(calls) == 2:
            if fault == 'job_end': job.write_text(json.dumps({**expected, 'created_at': 'drift'}))
            if fault == 'input_end': request.write_text('{"drift":true}')
        return native
    monkeypatch.setattr(m, 'context', context)
    def forbidden(*a, **kw): raise AssertionError('readonly PF entered scientific path')
    monkeypatch.setattr(m, 'rebuild_reference', forbidden)
    monkeypatch.setattr(m.helper, 'capture', forbidden)
    if fault:
        with pytest.raises(ValueError, match='job changed|pinned file'):
            m.execute('preflight', records[1]['path'], request, digest)
    else:
        result = m.execute('preflight', records[1]['path'], request, digest)
        assert result['verified'] is True and result['read_only'] is True
        assert result['payload_written'] is False and result['engineering'] is False
    assert len(calls) == 2


def test_formal_plan_real_launcher_chain(tmp_path):
    """Synthetic inventory relocation only; real -m plan and original launcher."""
    import io
    import os
    import tarfile
    from test_p_full_backend import ROOT, test_cli_plan_inputs_feed_real_launcher_prepare
    root = tmp_path / 'chain'; root.mkdir()
    test_cli_plan_inputs_feed_real_launcher_prepare(root)
    deployed = root / 'deployed'
    raw = subprocess.check_output(['git', '-C', os.environ.get('P_FULL_TEST_GIT_ROOT', str(ROOT)), 'archive', 'HEAD', 'analysis', 'src', 'configs'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            if not member.isfile(): continue
            path = deployed / member.name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(archive.extractfile(member).read())
    (deployed / 'analysis/p_subset_reference_rebuild.py').write_bytes((ROOT / 'analysis/p_subset_reference_rebuild.py').read_bytes())
    script = r'''import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);deployed=root/'deployed'
from analysis import p_subset_reference_rebuild as m
sha=m.backend.file_sha
value=json.loads((root/'request.json').read_bytes());old=json.loads((root/'inputs.json').read_bytes())
data=Path(old['train'][0]).parent.parent
roles=m.roles_for(data,Path(old['metadata'][0]).parent)
inputs=root/'subset-inputs.json';inputs.write_text(json.dumps(roles))
# Inventory substitution is confined to this synthetic clean deployment.
inv=json.loads((deployed/'configs/ur001-preparation/hf-files.json').read_bytes())
prefix='subsamples/ur0.01'
subset={**inv,'files':[r for r in inv['files'] if r['path'].startswith(prefix+'/')]}
fixed=deployed/'configs/a40-subset/hf-files.json';fixed.write_text(json.dumps(subset))
module=deployed/'analysis/ur001_derive_cli.py'
text=module.read_text().replace('d7df6268431bb0d0bcd77d56809bf8920914b4b8858f58f0dac409c7df81ff49',sha(fixed)).replace("'subsamples/ur0.01_ir0.01'","'subsamples/ur0.01'")
module.write_text(text)
stage=Path(value['stage2']);physical=stage/'UR001-input-manifest.json'
p=json.loads(physical.read_bytes())
p['files']=[{**r,'path':r['path'].removeprefix(prefix+'/')} for r in p['files']]
assert [r['path'] for r in p['files']]==[*(f'train/week_{w:02d}.parquet' for w in range(25)),'validation/week_25.parquet']
p['data_root']=str(data);physical.write_text(json.dumps(p))
manifest=stage/'UR001-manifest.json';mv=json.loads(manifest.read_bytes());mv['upstream']['UR001-input-manifest.json']=sha(physical);manifest.write_text(json.dumps(mv))
config=root/'frozen-config.json';config.write_text(json.dumps(value['config']))
value={k:v for k,v in value.items() if k not in ('predecessor','items_metadata','embedding')}
value.update(schema=m.SCHEMA,config_file={'path':str(config),'sha256':sha(config)},reference=None,admission_sha256=sha(inputs))
value['producer_pin']['manifest_sha256']=sha(manifest)
subprocess.run(['git','-C',str(deployed),'add','.'],check=True)
subprocess.run(['git','-C',str(deployed),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','subset transport fixture'],check=True)
value['source']={'commit':subprocess.check_output(['git','-C',str(deployed),'rev-parse','HEAD'],text=True).strip()}
sys.path.insert(0,str(root/'ops-source/ops'));from vkgr_ops import launcher
for stage in ('preflight','rebuild'):
 request=root/('subset-'+stage+'.json');request.write_text(json.dumps(value))
 cmd=[sys.executable,'-B','-m','analysis.p_subset_reference_rebuild','plan','--stage',stage,'--inputs',str(inputs),'--request',str(request),'--request-sha256',sha(request)]
 run=subprocess.run(cmd,cwd=deployed,env={**os.environ,'PYTHONPATH':str(deployed/'src')},capture_output=True,text=True)
 assert run.returncode==0,run.stdout+run.stderr
 plan=json.loads(run.stdout);spec=launcher.spec_from_args(launcher.build_parser().parse_args(plan['launcher_argv']));metadata=launcher.prepare(spec)
 expected=plan['expected_job']
 for key in ('git','config','inputs','cwd','overlay','pinned_files'):assert metadata[key]==expected[key],key
 for key in ('argv','seed','stop_mode','run_id','kind'):assert spec[key]==expected[key],key
 assert metadata['env']=={k:v for k,v in expected['env'].items() if k not in ('VKGR_RUN_ID','VKGR_PROGRESS_PATH')}
 assert {**{k:spec[k] for k in launcher.RESOURCE_ARGS},'wall_seconds':spec['wall_hours']*3600.0}==expected['resources']
 assert any(r['path'].endswith('/docs/p3c_validation.md') for r in plan['inputs'])
 assert len(plan['inputs'])>30
 # Reprepared metadata must expose a changed official physical byte.
 target=Path(roles['validation']);before=target.read_bytes();target.write_bytes(before+b'drift')
 assert launcher.prepare(spec)['inputs']!=expected['inputs']
 target.write_bytes(before)
 dry=subprocess.run([sys.executable,'-B','-m','analysis.p_subset_reference_rebuild','dry'],capture_output=True,text=True)
 assert dry.returncode==0 and json.loads(dry.stdout)['training_submitted'] is False
assert not Path(value['ops_root']).exists()
print(json.dumps({'verified':True,'formal_stages':2,'input_count':len(plan['inputs']),'submit':False}))
'''
    got = subprocess.run([sys.executable, '-B', '-c', script, str(root)], cwd=deployed,
             env={**os.environ, 'PYTHONPATH': str(deployed / 'src')}, capture_output=True, text=True)
    assert got.returncode == 0, got.stdout + got.stderr
    assert json.loads(got.stdout)['verified'] is True


def test_identity_sorts_only_mappings_and_preserves_tensor_and_row_bytes():
    import torch
    m = importlib.import_module('analysis.p_subset_reference_rebuild')
    value = {'rows': [{'side': [1, 2], 'history': [3, 4]}], 'content': torch.tensor([0.0, -0.0])}
    reordered = {'content': value['content'], 'rows': [{'history': [3, 4], 'side': [1, 2]}]}
    assert m.canonical_identity(value) == m.canonical_identity(reordered)
    for changed in ({**value, 'content': torch.tensor([0.0, 0.0])},
                    {**value, 'rows': [{'side': [2, 1], 'history': [3, 4]}]}):
        assert m.canonical_identity(value) != m.canonical_identity(changed)


@pytest.mark.parametrize('command', ['preflight', 'rebuild', 'compare'])
def test_formal_cli_dispatch(tmp_path, monkeypatch, capsys, command):
    m = importlib.import_module('analysis.p_subset_reference_rebuild')
    def execute(stage, inputs, request, digest):
        assert stage == command and inputs == 'inputs' and request == 'request' and digest == 'a' * 64
        return {'verified': True, 'stage': stage}
    monkeypatch.setattr(m, 'execute', execute)
    assert m.main([command, '--inputs', 'inputs', '--request', 'request', '--request-sha256', 'a' * 64]) == 0
    assert json.loads(capsys.readouterr().out)['stage'] == command


@pytest.mark.parametrize('key,changed', [('seed', 2024), ('argv', ['wrong']), ('resources', {}), ('env', {})])
def test_own_job_contract_rejects_drift(own_job_fixture, monkeypatch, key, changed):
    m = importlib.import_module('analysis.p_subset_reference_rebuild')
    _, value, records, request, digest, _ = own_job_fixture
    expected = m.job_contract(value, records, request, digest, 'preflight')
    run = Path(value['ops_root']) / 'runs' / expected['run_id']; run.mkdir()
    (run / 'run.json').write_text(json.dumps({**expected, key: changed}))
    for k, v in expected['env'].items(): monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match='own complete job'):
        m.check_job(value, records, request, digest, 'preflight')


@pytest.mark.parametrize('fault', ['', 'score', 'producer'])
def test_compare_executes_original_and_adapter_paths(tmp_path, native, own_job_fixture, actual_dag_module, monkeypatch, fault):
    m = importlib.import_module('analysis.p_subset_reference_rebuild')
    from analysis.ur001_derivation import derive
    roles, items, emb, parts, validation = fixture_inputs(tmp_path)
    roles['metadata'] = [str(tmp_path / 'unused-users'), str(items), str(emb)]
    output = tmp_path / 'reference'
    manifest = m.rebuild_reference(native, roles, items, emb, output, {'consumer': {'source_fingerprint': 'source'}}, lambda: None)
    stage2 = tmp_path / 'comparison-stage2'; stage2.mkdir()
    rows = []; result = derive(parts, validation, rows.append)
    for name, values in [('train', rows), ('validation', result['validation'])]:
        (stage2 / ('UR001-' + name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in values))
    for name, values in [('catalog', result['catalog']), ('supported', result['supported'])]:
        (stage2 / ('UR001-' + name + '.json')).write_text(json.dumps(values))
    # The compact adapter consumes the real derived files and their exact pins,
    # rather than the former materialized reference_assembly fixture path.
    artifacts = {path.name: {'sha256': m.backend.file_sha(path), 'size': path.stat().st_size}
                 for path in stage2.iterdir()}
    (stage2 / 'UR001-manifest.json').write_text(json.dumps({'counts': result['counts'], 'artifacts': artifacts}))
    _, value, records, request, digest, _ = own_job_fixture
    inputs = Path(records[1]['path']); inputs.write_text(json.dumps(roles))
    records[1].update(sha256=m.backend.file_sha(inputs), bytes=inputs.stat().st_size)
    value.update(output_root=str(output), stage2=str(stage2), disk_cap_bytes=1024**3,
                 config={**native.default_config(), 'batch_size': 8}, config_file={'sha256': 'f' * 64},
                 producer_pin={'fixed': True}, reference={'manifest': str(output / 'reference-manifest.json'), 'run_dir': str(tmp_path / 'reference-job'), 'pin': {}})
    manifest.update(schema='p-subset-reference-manifest-v1', config_sha256='f' * 64, subset_producer={'fixed': fault != 'producer'})
    monkeypatch.setattr(actual_dag_module, 'verify_producer', lambda *a: {'manifest': manifest})
    monkeypatch.setattr(m, 'read_transport', lambda *a: (value, roles))
    monkeypatch.setattr(m, 'stage_inputs', lambda *a: records)
    monkeypatch.setattr(m, 'context', lambda *a: native)
    monkeypatch.setattr(native.b1, 'source_identity', lambda: {'source_sha': 'source'})
    monkeypatch.setattr(native.b1, 'resource_guard', lambda *a: lambda: None)
    monkeypatch.setattr(native.b1, 'execution_context', lambda *a: None)
    expected = m.job_contract(value, records, request, digest, 'compare')
    run = Path(value['ops_root']) / 'runs' / expected['run_id']; run.mkdir(exist_ok=True)
    (run / 'run.json').write_text(json.dumps(expected))
    for k, v in expected['env'].items(): monkeypatch.setenv(k, v)
    original_capture = m.helper.capture; calls = []
    def cpu_capture(module, data, config, device, guard):
        assert device == 'cuda'; calls.append(1)
        result = original_capture(module, data, config, 'cpu', guard)
        if fault == 'score' and len(calls) == 2: result['evaluation']['scores'] = 'drift'
        return result
    monkeypatch.setattr(m.helper, 'capture', cpu_capture)
    if fault:
        with pytest.raises(ValueError, match='source identity|steps/evaluation'):
            m.execute('compare', inputs, request, digest)
    else:
        receipt = m.execute('compare', inputs, request, digest)
        assert receipt['verified'] is True and receipt['bitwise_equal'] is True
        assert receipt['steps'] == 3 and receipt['queries'] == 32 and receipt['checkpoint_written'] is False
    assert len(calls) == (0 if fault == 'producer' else 2)
