"""P UR001 adapter and native scientific call-path regression, synthetic only."""
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp('p-full-native')
    raw = subprocess.check_output(['git', '-C', os.environ.get('P_FULL_TEST_GIT_ROOT', str(ROOT)), 'archive', '0ca61bc', 'src', 'configs/seqm1', '.python-version', 'pyproject.toml', 'uv.lock'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(root, filter='data')
    from analysis.throughput_step_harness import _package
    return _package('p_full_test_native', root)('retrieval_seqm1_run')


def test_dry_and_invalid_cli():
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src') + os.pathsep + str(ROOT)}
    for command, expected in [('dry', 0), ('illegal', 2)]:
        got = subprocess.run([sys.executable, '-B', '-m', 'analysis.p_full_training_backend', command],
                             cwd=ROOT, env=env, capture_output=True, text=True)
        assert got.returncode == expected
        if not expected:
            assert json.loads(got.stdout)['training'] is False


def test_holdout_rejected_before_any_loader(tmp_path):
    from analysis import p_full_training_backend as m
    from analysis.ur001_science_entry import A_ROOT, FORBIDDEN_ROOTS, SCHEMA
    request = tmp_path / 'inputs.json'
    value = {'schema': SCHEMA,
             'train': [str(A_ROOT / f'subsamples/ur0.01/train/week_{w:02}.parquet') for w in range(25)],
             'validation': str(FORBIDDEN_ROOTS[2] / 'week_26.parquet'),
             'metadata': [str(A_ROOT / 'metadata' / n) for n in
                          ('users_metadata.parquet', 'items_metadata.parquet', 'item_embeddings.npz')]}
    request.write_text(json.dumps(value))
    got = subprocess.run([sys.executable, '-B', '-m', 'analysis.p_full_training_backend',
                          'preflight', '--inputs', str(request), '--request', str(tmp_path / 'absent'),
                          '--request-sha256', '0' * 64], cwd=ROOT,
                         env={**os.environ, 'PYTHONPATH': str(ROOT / 'src') + os.pathsep + str(ROOT)},
                         capture_output=True, text=True)
    assert got.returncode == 2
    assert json.loads(got.stdout)['status'] == 'HOLDOUT_INPUT_REJECTED'
    assert json.loads(got.stdout)['data_opened'] is False


def test_adapter_and_native_steps_equal(native):
    import numpy as np
    import torch
    from analysis import p_full_training_backend as m
    from analysis.tensor_cache_parity import freeze
    from analysis.ur001_derivation import derive
    torch.set_num_threads(1)
    parts = [(0, [{'user_id': u, 'item_id': 1 + (u + k) % 20,
                   'like': True, 'share': False, 'bookmark': False, 'open_comments': False,
                   'click_on_author': False, 'dislike': False, 'timespent': 3}
                  for k in range(5) for u in range(12)])]
    validation = (25, [{'user_id': u, 'item_id': u + 1, 'like': True} for u in range(12)])
    train = []
    result = derive(parts, validation, train.append)
    catalog = result['catalog']; support = result['supported']; queries = result['validation']
    features = np.zeros((len(catalog) + 1, 64), dtype=np.float32)
    durations = {i: 5 for i in catalog}
    adapted = m.adapt_rows(native, train, queries, catalog, support, parts, durations, features, synthetic=True)
    hs = importlib.import_module(native.__package__ + '.history_side')
    side = hs.build_side(parts, {'train': train, 'validation': queries}, durations)
    old = {**adapted, 'rows': [{**r, 'side': s} for r, s in zip(train, side['train'])],
           'queries': [{**r, 'side': s} for r, s in zip(queries, side['validation'])]}
    assert freeze({**adapted, 'support': sorted(adapted['support'])}) == freeze({**old, 'support': sorted(old['support'])})
    config = {**native.default_config(), 'batch_size': 12, 'max_epochs': 1, 'min_epochs': 0}
    def capture(data):
        import random
        training = importlib.import_module(native.__package__ + '.retrieval_training')
        negatives = importlib.import_module(native.__package__ + '.retrieval_negatives')
        model = native._model(data, config, 'cpu'); opt = native.make_optimizer(model, config)
        rows, view_queries = native.fr._view(data, 50)
        cursor = training.EpochCursor(len(rows), config['seed']); stats = training.SamplerStats()
        rng = random.Random(f"p-b1-negative-v1:{config['seed']}"); frequency = negatives.target_frequency(rows)
        trace = []
        for _ in range(3):
            batch = [rows[i] for i in cursor.next(config['batch_size'])]
            tensors = native.fr._tensors2(batch, data['idmap'], max(len(r['history']) for r in batch), config['feature'])
            record = native.fr.train_step(model, opt, batch, data, stats, cursor.epoch, config, rng, None, frequency)
            trace.append(freeze([tensors, record, model.state_dict(), opt.state_dict(), cursor.state_dict(), rng.getstate()]))
        users, items, norms = native.fr.export_vectors_side2(model, data['catalog'], view_queries, data['idmap'], config['batch_size'], 50, config['feature'])
        trace.append(freeze([users, items, norms, users.astype('float64') @ items.astype('float64').T,
                            native.evaluate_model(model, data, view_queries, config)]))
        return trace
    assert capture(adapted) == capture(old)


def test_scale_guard_rejects_wrong_count(native, tmp_path):
    from analysis import p_full_training_backend as m
    data = native.fr.synthetic_data()
    with pytest.raises(ValueError, match='count'):
        m.run_session(native, tmp_path / 'out', data, native.default_config(), expected_rows=len(data['rows']) + 1)
    assert not (tmp_path / 'out').exists()


def test_loop_matches_native_and_verify(native, tmp_path):
    from analysis import p_full_training_backend as m
    from analysis.tensor_cache_parity import freeze
    data = native.fr.synthetic_data()
    config = {**native.default_config(), 'max_epochs': 1, 'min_epochs': 0, 'batch_size': 32}
    old = native.run_session(tmp_path / 'old', data, config)
    new = m.run_session(native, tmp_path / 'new', data, config, expected_rows=len(data['rows']))
    # Runtime and physical checkpoint filenames differ; scientific records do not.
    for key in ('records', 'selection', 'stats', 'data_diagnostics', 'ablation_g0'):
        assert freeze(old[key]) == freeze(new[key])
    for left, right in zip(old['evaluations'], new['evaluations']):
        assert freeze(left) == freeze(right)
    assert native.verify_run(tmp_path / 'new/manifest.json', new['manifest_sha256'], data, config)['status'] == 'verified'


def test_real_launcher_deployment_metadata_dry(tmp_path):
    """Actual frozen launcher.prepare against a clean deployed Git tree; no submit."""
    deployment = tmp_path / 'deployment'; deployment.mkdir()
    ops = tmp_path / 'ops-source'; ops.mkdir()
    for target, args in ((deployment, ['HEAD', 'analysis/ur001_science_entry.py']),
                         (ops, ['5840e6b', 'ops'])):
        with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git', 'archive', *args]))) as archive:
            archive.extractall(target, filter='data')
    (deployment / 'analysis/p_full_training_backend.py').write_bytes((ROOT / 'analysis/p_full_training_backend.py').read_bytes())
    # The separately deployed historical native tree retains its p3c data override.
    native = tmp_path / 'native'; (native / 'docs').mkdir(parents=True)
    (native / 'docs/p3c_validation.md').write_bytes(b'synthetic data-channel fixture\n')
    subprocess.run(['git', 'init', '-q', str(deployment)], check=True)
    subprocess.run(['git', '-C', str(deployment), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(deployment), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'synthetic deployment fixture'], check=True)
    request = tmp_path / 'request.json'; request.write_text('{}')
    worker = [sys.executable, '-B', '-m', 'analysis.p_full_training_backend', 'dry']
    code = '''import json,sys
from vkgr_ops import launcher
root,cwd,request,worker=json.loads(sys.argv[1])
a=launcher.build_parser().parse_args(['--root',root,'--run-id','p-full-backend-fixture','--kind','ops',
    '--cwd',cwd,'--config',request,'--input',request,'--cpu-cores','1','--ram-mib','1024',
    '--vram-mib','0','--disk-gib','0','--wall-hours','1','--',*worker])
spec=launcher.spec_from_args(a); metadata=launcher.prepare(spec)
assert metadata['git']['dirty'] is False and len(metadata['inputs']) == 1
print(json.dumps({'status':'prepared_dry','metadata':metadata}))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code,
                             json.dumps([str(tmp_path / 'ops-root'), str(deployment), str(request), worker])],
                            env={**os.environ, 'PYTHONPATH': str(ops / 'ops')}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'prepared_dry'
    dry = subprocess.run(worker, cwd=deployment, env={**os.environ, 'PYTHONPATH': str(deployment)},
                         capture_output=True, text=True)
    assert dry.returncode == 0 and json.loads(dry.stdout)['status'] == 'dry'
    assert not (tmp_path / 'ops-root/runs').exists()


@pytest.mark.parametrize('change', ['future', 'misaligned', 'support', 'nonfinite', 'padding'])
def test_adapter_rejects_input_drift(native, change):
    import copy
    import numpy as np
    from analysis import p_full_training_backend as m
    row = {'example_id': 0, 'user_id': 1, 'history': [1], 'history_positions': [[0, 0]],
           'target': 2, 'target_position': [1, 0], 'past_likes': [1]}
    query = {'user_id': 1, 'history': [1], 'history_positions': [[0, 0]],
             'target': 2, 'target_position': [25, 0], 'oov': False}
    supported = [1, 2]; features = np.zeros((3, 64), dtype=np.float32)
    if change == 'future':
        row['history_positions'] = [[0, 1]]; row['target_position'] = [0, 1]
    if change == 'misaligned': row['history_positions'] = []
    if change == 'support': supported = [2, 1]
    if change == 'nonfinite': features[1, 0] = np.nan
    if change == 'padding': features[0, 0] = 1
    event = {'item_id': 1, 'like': True, 'share': False, 'bookmark': False,
             'open_comments': False, 'click_on_author': False, 'dislike': False, 'timespent': 3}
    with pytest.raises(ValueError):
        m.adapt_rows(native, [row], [query], [1, 2], supported, [(0, [event, event])], {1: 5, 2: 5}, features)


def test_training_loop_semantics_are_frozen(native):
    """Pin the complete post-admission orchestration AST, including stop/selection/RNG."""
    import ast
    import inspect
    from analysis import p_full_training_backend as m
    old = ast.parse(inspect.getsource(native.run_session)).body[0]
    new = ast.parse(inspect.getsource(m.run_session)).body[0]
    # The copied body after its one real-runtime admission check is bytecode-
    # independent but AST-identical; no precision/sampler/selection edits hide here.
    def tail(function):
        begin = next(i for i, node in enumerate(function.body)
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and ast.unparse(node.value.func) == 'b1.execution_context')
        return [ast.dump(node, include_attributes=False) for node in function.body[begin:]]
    assert tail(old) == tail(new)


@pytest.fixture
def own_job_fixture(tmp_path):
    """Transport-only files and complete fixed job, no scientific computation."""
    from analysis import p_full_training_backend as m
    import hashlib
    stage2 = tmp_path / 'stage2'; stage2.mkdir()
    native_root = tmp_path / 'native'; native_root.mkdir()
    ops = tmp_path / 'ops'; run = ops / 'runs/p-full-train-fixture'; run.mkdir(parents=True)
    def record(path, content=b'{}'):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        return {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    records = [record(tmp_path / 'inputs.json'), record(tmp_path / 'lock'),
               record(stage2 / 'UR001-manifest.json'), record(tmp_path / 'metadata.npz')]
    request = tmp_path / 'request.json'; request.write_text('{}')
    digest = m.file_sha(request)
    records.insert(0, {'path': str(request), 'sha256': digest, 'bytes': request.stat().st_size})
    env = {'PYTHONPATH': str(m.ROOT / 'src'), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    transport = {'env': env, 'resources': {'cpu_cores': 1, 'ram_mib': 98304, 'vram_mib': 32768,
                   'disk_gib': 4, 'wall_seconds': 172800.0},
                 'ops_source': {'commit': 'a' * 40, 'sha256': 'b' * 64},
                 'overlay': None, 'pinned_files': {}, 'stop_mode': 'signal'}
    value = {'config': {'seed': 42}, 'source': {'commit': 'c' * 40}, 'ops_root': str(ops),
             'run_ids': {'preflight': 'p-full-pf-fixture', 'train': 'p-full-train-fixture', 'verify': 'p-full-verify-fixture'},
             'job_contract': {s: transport for s in ('preflight', 'train', 'verify')}}
    return m, value, records, request, digest, run


@pytest.mark.parametrize('change', ['seed', 'schema', 'kind', 'dirty_files', 'extra_input', 'missing_input',
                                   'duplicate_input', 'input_order', 'input_bytes', 'input_sha', 'env', 'resources',
                                   'ops_source', 'overlay', 'pinned_files', 'stop_mode', 'boolean_seed',
                                   'actual_env', 'extra_field', 'duplicate_json'])
def test_complete_own_job_rejects_drift(own_job_fixture, monkeypatch, change):
    import copy
    m, value, records, request, digest, run_dir = own_job_fixture
    expected = m.job_contract(value, records, request, digest, 'train')
    actual = copy.deepcopy(expected)
    for key, val in expected['env'].items(): monkeypatch.setenv(key, val)
    if change == 'seed': actual['seed'] = 2024
    if change == 'schema': actual['schema'] = 'other'
    if change == 'kind': actual['kind'] = 'unregistered'
    if change == 'dirty_files': actual['git']['dirty_files'] = ['untracked.py']
    if change == 'extra_input': actual['inputs'].append(copy.deepcopy(records[-1]))
    if change == 'missing_input': actual['inputs'].pop()
    if change == 'duplicate_input': actual['inputs'][1] = copy.deepcopy(actual['inputs'][0])
    if change == 'input_order': actual['inputs'][1], actual['inputs'][2] = actual['inputs'][2], actual['inputs'][1]
    if change == 'input_bytes': actual['inputs'][0]['bytes'] += 1
    if change == 'input_sha': actual['inputs'][0]['sha256'] = '0' * 64
    if change == 'env': actual['env']['OMP_NUM_THREADS'] = '2'
    if change == 'resources': actual['resources']['vram_mib'] = 0
    if change == 'ops_source': actual['ops_source']['commit'] = 'd' * 40
    if change == 'overlay': actual['overlay'] = {'extra': 'unapproved'}
    if change == 'pinned_files': actual['pinned_files'] = {'extra.py': '0' * 64}
    if change == 'stop_mode': actual['stop_mode'] = 'none'
    if change == 'boolean_seed': actual['seed'] = False
    if change == 'actual_env': monkeypatch.setenv('OMP_NUM_THREADS', '2')
    if change == 'extra_field': actual['unregistered'] = False
    text = json.dumps(actual)
    if change == 'duplicate_json': text = text[:-1] + ',"seed":42}'
    (run_dir / 'run.json').write_text(text)
    with pytest.raises(ValueError, match='job|environment|duplicate'):
        m.check_job(value, records, request, digest, 'train')


def test_complete_own_job_accepts_exact_and_records_sha(own_job_fixture, monkeypatch):
    m, value, records, request, digest, run_dir = own_job_fixture
    expected = m.job_contract(value, records, request, digest, 'train')
    for key, val in expected['env'].items(): monkeypatch.setenv(key, val)
    expected['created_at'] = 'synthetic'; expected['admission'] = {'status': 'admitted'}
    (run_dir / 'run.json').write_text(json.dumps(expected))
    original_sha = m.check_job(value, records, request, digest, 'train')
    assert original_sha == m.file_sha(run_dir / 'run.json')
    (run_dir / 'run.json').write_text(json.dumps({**expected, 'created_at': 'changed'}))
    assert m.check_job(value, records, request, digest, 'train') != original_sha


@pytest.mark.parametrize('bad_field,bad_value', [('seed', 2024), ('schema', 'other'), ('kind', 'unregistered')])
def test_bad_job_rejected_before_context_or_science(own_job_fixture, monkeypatch, bad_field, bad_value):
    m, value, records, request, digest, run_dir = own_job_fixture
    expected = m.job_contract(value, records, request, digest, 'train')
    for key, val in expected['env'].items(): monkeypatch.setenv(key, val)
    (run_dir / 'run.json').write_text(json.dumps({**expected, bad_field: bad_value}))
    monkeypatch.setattr(m, 'read_transport', lambda *args: value)
    monkeypatch.setattr(m, 'stage_inputs', lambda *args: records)
    def forbidden(*args, **kwargs): raise AssertionError('scientific context entered')
    monkeypatch.setattr(m, 'read_context', forbidden)
    with pytest.raises(ValueError, match='job'):
        m.execute('train', Path(records[1]['path']), request, digest)


def test_cli_plan_inputs_feed_real_launcher_prepare(tmp_path):
    """Actual backend plan and frozen launcher; only fixture admission roots/pins relocate."""
    import hashlib
    deployment = tmp_path / 'deployed'; deployment.mkdir()
    ops_source = tmp_path / 'ops-source'; ops_source.mkdir()
    files = ['analysis/ur001_science_entry.py', 'analysis/ur001_prepare.py', 'analysis/ur001_derive_cli.py',
             'analysis/profile_anchor_steps.py', 'configs/ur001-preparation/hf-files.json']
    for target, args in ((deployment, ['HEAD', *files]), (ops_source, ['5840e6b', 'ops'])):
        raw = subprocess.check_output(['git', '-C', os.environ.get('P_FULL_TEST_GIT_ROOT', str(ROOT)), 'archive', *args])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive: archive.extractall(target, filter='data')
    (deployment / 'analysis/p_full_training_backend.py').write_bytes((ROOT / 'analysis/p_full_training_backend.py').read_bytes())
    data = tmp_path / 'data'; data.mkdir()
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, sort_keys=True).encode() if not isinstance(value, bytes) else value
        path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw)}
    # Relocation belongs only to this clean synthetic deployment fixture.
    admission = deployment / 'analysis/ur001_science_entry.py'
    admission.write_text(admission.read_text().replace('/workspace/vkgr-ur001-data-20261020-v1', str(data)))
    inventory_path = deployment / 'configs/ur001-preparation/hf-files.json'
    inventory = json.loads(inventory_path.read_bytes())
    for rec in inventory['files']:
        fixed = put(data / rec['path'], b'synthetic payload\n')
        rec.update(size=fixed['size'], sha256=fixed['sha256'])
    inventory_path.write_text(json.dumps(inventory))
    derive = deployment / 'analysis/ur001_derive_cli.py'
    derive.write_text(derive.read_text().replace('19cd71f02aff4fce93349907dbc5fcdbf006afc679ee899d9c98965123270fa0', hashlib.sha256(inventory_path.read_bytes()).hexdigest()))
    native = tmp_path / 'native'
    native_file = put(native / 'src/vkgr/__init__.py', b'')
    p3c = put(native / 'docs/p3c_validation.md', b'synthetic data-channel fixture\n')
    pins = {'schema': 'anchor-profile-pins-v1', 'g': {}, 'p': {
        'source_files': {'src/vkgr/__init__.py': native_file['sha256']}, 'p3c_data_sha256': p3c['sha256']}}
    pins_file = put(deployment / 'configs/anchor-profile/source-pins.json', pins)
    profile = deployment / 'analysis/profile_anchor_steps.py'
    profile.write_text(profile.read_text().replace('1227ff9a3b6dbe7cb6f6e2b05be154a12e3a6d224bf980d4df33380b05e4747b', pins_file['sha256']))
    stage2 = tmp_path / 'stage2'
    physical = put(stage2 / 'UR001-input-manifest.json', {
        'profile': 'full', 'inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        'data_root': str(data / 'subsamples/ur0.01'),
        'files': [{**r, 'path': r['path'].removeprefix('subsamples/ur0.01/')} for r in inventory['files'][:26]]})
    artifacts = {}
    for name in ('train.jsonl', 'validation.jsonl', 'catalog.json', 'supported.json', 'S0.json', 'counts.json'):
        rec = put(stage2 / ('UR001-' + name), b'{}\n')
        artifacts['UR001-' + name] = {'sha256': rec['sha256'], 'size': rec['size']}
    manifest = put(stage2 / 'UR001-manifest.json', {'upstream': {'UR001-input-manifest.json': physical['sha256']}, 'artifacts': artifacts})
    producer = tmp_path / 'producer'; producer_pins = {}
    for name in ('run.json', 'terminal.json', 'logs/stdout.log'):
        producer_pins[name] = put(producer / name, {})['sha256']
    inputs = tmp_path / 'inputs.json'
    admitted = {'schema': 'ur001-science-inputs-v1', 'train': [str(data / r['path']) for r in inventory['files'][:25]],
                'validation': str(data / inventory['files'][25]['path']),
                'metadata': [str(data / 'metadata' / n) for n in ('users_metadata.parquet', 'items_metadata.parquet', 'item_embeddings.npz')]}
    inp = put(inputs, admitted); lock = put(tmp_path / 'lock', b'synthetic runtime lock')
    subprocess.run(['git', 'init', '-q', str(deployment)], check=True)
    subprocess.run(['git', '-C', str(deployment), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(deployment), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'transport fixture'], check=True)
    source = subprocess.check_output(['git', '-C', str(deployment), 'rev-parse', 'HEAD'], text=True).strip()
    fixed = {r['path']: r for r in inventory['files']}
    env = {'PYTHONPATH': str(deployment / 'src'), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    contract = {'env': env, 'resources': {'cpu_cores': 1, 'ram_mib': 98304, 'vram_mib': 32768, 'disk_gib': 4, 'wall_seconds': 7200.0},
                'ops_source': {'commit': '5840e6b686ec9651314a9e28a1a2c0befcc05158', 'sha256': 'b' * 64},
                'overlay': None, 'pinned_files': {}, 'stop_mode': 'signal'}
    value = {'schema': 'p-full-backend-request-v1', 'source': {'commit': source}, 'runtime': {},
             'lock': {k: lock[k] for k in ('path', 'sha256')}, 'native_root': str(native), 'stage2': str(stage2),
             'stage2_run': str(producer), 'producer_pin': {'manifest_sha256': manifest['sha256'], 'job_sha256': producer_pins},
             'items_metadata': {'path': str(data / 'metadata/items_metadata.parquet'), 'sha256': fixed['metadata/items_metadata.parquet']['sha256']},
             'embedding': {'path': str(data / 'metadata/item_embeddings.npz'), 'sha256': fixed['metadata/item_embeddings.npz']['sha256']},
             'config': {'seed': 42}, 'ops_root': str(tmp_path / 'ops-root'),
             'run_ids': {'preflight': 'p-full-pf-fixture', 'train': 'p-full-train-fixture', 'verify': 'p-full-verify-fixture'},
             'output_root': str(tmp_path / 'output'), 'admission_sha256': inp['sha256'], 'predecessor': None,
             'disk_cap_bytes': 1024**3, 'job_contract': {stage: contract for stage in ('preflight', 'train', 'verify')}}
    request = tmp_path / 'request.json'; req = put(request, value)
    cli = [sys.executable, '-B', '-m', 'analysis.p_full_training_backend', 'plan', '--stage', 'preflight',
           '--inputs', str(inputs), '--request', str(request), '--request-sha256', req['sha256']]
    planned = subprocess.run(cli, cwd=deployment, env={**os.environ, 'PYTHONPATH': str(deployment)}, capture_output=True, text=True)
    assert planned.returncode == 0, planned.stdout + planned.stderr
    plan = json.loads(planned.stdout)
    assert plan['status'] == 'planned_dry' and len(plan['inputs']) > 30
    assert len({r['path'] for r in plan['inputs']}) == len(plan['inputs'])
    assert any(r['path'] == str(native / 'docs/p3c_validation.md') for r in plan['inputs'])
    script = '''import json,sys
from vkgr_ops import launcher
plan=json.loads(sys.argv[1]);args=launcher.build_parser().parse_args(plan['launcher_argv'])
metadata=launcher.prepare(launcher.spec_from_args(args))
for key in ('git','config','inputs','cwd'):
    assert metadata[key] == plan['expected_job'][key], key
assert metadata['env'] == {k:v for k,v in plan['expected_job']['env'].items() if k not in ('VKGR_RUN_ID','VKGR_PROGRESS_PATH')}
print(json.dumps({'status':'prepared_dry','inputs':len(metadata['inputs'])}))
'''
    prepared = subprocess.run([sys.executable, '-B', '-c', script, planned.stdout],
                              env={**os.environ, 'PYTHONPATH': str(ops_source / 'ops')}, capture_output=True, text=True)
    assert prepared.returncode == 0, prepared.stderr
    assert json.loads(prepared.stdout)['inputs'] == len(plan['inputs'])
    assert not Path(value['ops_root']).exists()


@pytest.mark.parametrize('tamper', ['run_bytes', 'request_bytes'])
def test_pf_rechecks_closure_and_run_sha_before_receipt(own_job_fixture, monkeypatch, tamper):
    from types import SimpleNamespace
    m, value, records, request, digest, _ = own_job_fixture
    value.update(output_root=str(request.parent / 'output'), disk_cap_bytes=1024**3, runtime={},
                 native_root=str(request.parent / 'native'), stage2=str(request.parent / 'stage2'))
    run_dir = Path(value['ops_root']) / 'runs' / value['run_ids']['preflight']; run_dir.mkdir()
    expected = m.job_contract(value, records, request, digest, 'preflight')
    for key, val in expected['env'].items(): monkeypatch.setenv(key, val)
    run_file = run_dir / 'run.json'; run_file.write_text(json.dumps(expected))
    monkeypatch.setattr(m, 'read_transport', lambda *args: value)
    monkeypatch.setattr(m, 'stage_inputs', lambda *args: records)
    native = SimpleNamespace(b1=SimpleNamespace(execution_context=lambda *a: None,
                             resource_guard=lambda *a: lambda: None))
    # This test isolates receipt closure; facade science has separate parity tests.
    from analysis import p_full_parallel_eval
    native.parallel_eval_identity = {}
    monkeypatch.setattr(p_full_parallel_eval, 'bind', lambda *args, **kwargs: native)
    monkeypatch.setattr(m, 'read_context', lambda *args: (value, native, {'rows': [], 'queries': []}, {}))
    monkeypatch.setattr(m, 'check_predecessor', lambda *args: None)
    def close(*args):
        if tamper == 'run_bytes': run_file.write_text(json.dumps({**expected, 'created_at': 'changed'}))
        else: request.write_text('{"changed": true}')
    monkeypatch.setattr(m, 'close_context', close)
    def forbidden(*args): raise AssertionError('receipt emitted before end binding')
    monkeypatch.setattr(m, 'write_stage_receipt', forbidden)
    with pytest.raises(ValueError, match='job|pinned'):
        m.execute('preflight', Path(records[1]['path']), request, digest)


@pytest.fixture
def actual_dag_module(monkeypatch):
    """Actual shared helper, never a stub; fallback supports the isolated P branch."""
    try:
        return importlib.import_module('analysis.producer_dag_binding')
    except ModuleNotFoundError as exc:
        if exc.name != 'analysis.producer_dag_binding':
            raise
    # The parent integrates this dependency separately. Its reviewed source is
    # already in this repository's Git objects; no private directory is needed.
    from types import ModuleType
    raw = subprocess.check_output(['git', '-C', os.environ.get('P_FULL_TEST_GIT_ROOT', str(ROOT)),
                                   'show', 'b1628fa239cf5e6715f9515dd57158c5bf214a07:analysis/producer_dag_binding.py'])
    module = ModuleType('analysis.producer_dag_binding')
    module.__file__ = 'producer_dag_binding.py@b1628fa'
    exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize('drift', [None, 'source_commit', 'source_ledger', 'runtime_dependency', 'runtime_lock'])
def test_read_context_checks_actual_consumer_identity_before_producer(tmp_path, monkeypatch, actual_dag_module, drift):
    import copy
    import hashlib
    from analysis import p_full_training_backend as m
    from analysis import new_instance_prepare, ur001_derive_cli, ur001_prepare
    data_root = tmp_path / 'data'; metadata_root = data_root / 'metadata'; metadata_root.mkdir(parents=True)
    def record(path):
        path.write_bytes(b'synthetic metadata')
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    items = record(metadata_root / 'items_metadata.parquet')
    embedding = record(metadata_root / 'item_embeddings.npz')
    lock = record(tmp_path / 'lock')
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps({'metadata': [str(metadata_root / 'users_metadata.parquet'), items['path'], embedding['path']]}))
    source = {'commit': 'a' * 40, 'files': {'src/vkgr/example.py': 'b' * 64}}
    runtime = {'runtime': {'python': '3.11.16', 'numpy': '1.26.4', 'torch': '2.11.0+cu128'},
               'lock_sha256': lock['sha256'], 'cuda_smoke_executed': False}
    actual_source = copy.deepcopy(source); actual_runtime = copy.deepcopy(runtime)
    if drift == 'source_commit': actual_source['commit'] = 'c' * 40
    if drift == 'source_ledger': actual_source['files']['src/vkgr/example.py'] = 'c' * 64
    if drift == 'runtime_dependency': actual_runtime['runtime']['numpy'] = '2.0.0'
    if drift == 'runtime_lock': actual_runtime['lock_sha256'] = 'c' * 64
    value = {'items_metadata': items, 'embedding': embedding, 'lock': lock, 'source': source,
             'runtime': runtime, 'stage2': str(tmp_path / 'stage2'), 'stage2_run': str(tmp_path / 'producer'), 'producer_pin': {}}
    monkeypatch.setattr(m, 'read_transport', lambda *args: value)
    official = [{'path': str(Path(r['path']).relative_to(data_root)), 'sha256': r['sha256'],
                 'size': Path(r['path']).stat().st_size} for r in (items, embedding)]
    monkeypatch.setattr(ur001_prepare, 'load_file_manifest', lambda: {'files': official})
    monkeypatch.setattr(ur001_derive_cli, 'source_identity', lambda: actual_source)
    monkeypatch.setattr(new_instance_prepare, 'runtime', lambda *args: actual_runtime)
    seen = []
    class ProducerBoundaryReached(Exception): pass
    def boundary(*args, **kwargs):
        seen.append('producer'); raise ProducerBoundaryReached()
    # Keep actual_dag_module.check_consumer itself intact. Only the next,
    # expensive producer stage is a boundary marker for positive reachability.
    monkeypatch.setattr(actual_dag_module, 'verify_producer', boundary)
    if drift is None:
        with pytest.raises(ProducerBoundaryReached):
            m.read_context(inputs, tmp_path / 'request.json', '0' * 64)
        assert seen == ['producer']
    else:
        with pytest.raises(ValueError, match='consumer exact source/runtime'):
            m.read_context(inputs, tmp_path / 'request.json', '0' * 64)
        assert seen == []


@pytest.mark.parametrize('bad_manifest_pin', [False, True])
def test_formal_verify_forwards_to_original_native_and_propagates_failure(own_job_fixture, native, monkeypatch, bad_manifest_pin):
    """Exercise execute('verify'), with the actual frozen verifier on CPU data."""
    m, value, records, request, digest, _ = own_job_fixture
    output = request.parent / 'native-output'
    data = native.fr.synthetic_data()
    config = {**native.default_config(), 'max_epochs': 1, 'min_epochs': 0, 'batch_size': 32}
    binding = {'fixed_scientific_inputs': 'synthetic verifier fixture', 'parallel_evaluation': {}}
    # Keep this forwarding spy isolated from the facade code-identity guard.
    from analysis import p_full_parallel_eval
    monkeypatch.setattr(native, 'parallel_eval_identity', {}, raising=False)
    monkeypatch.setattr(p_full_parallel_eval, 'bind', lambda *args, **kwargs: native)
    trained = native.run_session(output, data, config, 'cpu', inputs_binding=binding)
    manifest = output / 'manifest.json'
    records.append({'path': str(manifest), 'sha256': trained['manifest_sha256'], 'bytes': manifest.stat().st_size})
    value.update(output_root=str(output), disk_cap_bytes=1024**3, config=config, runtime={},
                 native_root=str(request.parent / 'native'), stage2=str(request.parent / 'stage2'))
    run_dir = Path(value['ops_root']) / 'runs' / value['run_ids']['verify']; run_dir.mkdir()
    expected = m.job_contract(value, records, request, digest, 'verify')
    for key, member in expected['env'].items(): monkeypatch.setenv(key, member)
    (run_dir / 'run.json').write_text(json.dumps(expected))
    monkeypatch.setattr(m, 'read_transport', lambda *args: value)
    monkeypatch.setattr(m, 'stage_inputs', lambda *args: records)
    monkeypatch.setattr(m, 'read_context', lambda *args: (value, native, data, binding))
    verify_digest = '0' * 64 if bad_manifest_pin else trained['manifest_sha256']
    monkeypatch.setattr(m, 'check_predecessor', lambda *args: {'native_manifest_sha256': verify_digest})
    closure = []
    monkeypatch.setattr(m, 'close_context', lambda *args: closure.append('closed'))
    # No real CUDA or live-server resource probe is performed in this CPU test.
    original_execution_context = native.b1.execution_context
    monkeypatch.setattr(native.b1, 'execution_context', lambda device: None if device == 'cuda' else original_execution_context(device))
    guards = []
    def guard(): guards.append('checked')
    monkeypatch.setattr(native.b1, 'resource_guard', lambda *args: guard)
    original_verify = native.verify_run; calls = []
    def observe(*args, **kwargs):
        assert args[0] == manifest and args[1] == verify_digest
        assert args[2] is data and args[3] is config and args[4] == 'cuda'
        assert kwargs['check'] is guard and kwargs['inputs_binding'] == binding
        calls.append('original-native-verify')
        # Check the formal CUDA argument above, then run the original frozen
        # arithmetic/verifier on its existing synthetic CPU-supported path.
        cpu_args = (*args[:4], 'cpu', *args[5:])
        result = original_verify(*cpu_args, **kwargs)
        return {**result, 'test_evidence': 'original-native-CPU-verifier'}
    monkeypatch.setattr(native, 'verify_run', observe)
    receipt_path = m.stage_receipt_path(output, 'verify')
    if bad_manifest_pin:
        with pytest.raises(ValueError, match='external hash mismatch'):
            m.execute('verify', Path(records[1]['path']), request, digest)
        assert not receipt_path.exists() and closure == []
    else:
        receipt = m.execute('verify', Path(records[1]['path']), request, digest)
        assert receipt['native_verification']['test_evidence'] == 'original-native-CPU-verifier'
        assert receipt['native_verification']['status'] == 'verified'
        assert receipt['scientific_verified'] is True and receipt['verified'] is True
        assert json.loads(receipt_path.read_bytes()) == receipt
        assert closure == ['closed']
    assert calls == ['original-native-verify']
    if not bad_manifest_pin:
        assert guards
