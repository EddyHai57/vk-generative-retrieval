"""UR001 input adapter for the frozen M1-mean scientific functions.

No submit command. Formal stages require the pinned request and launcher identity;
training additionally requires an explicit approved-science switch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

from analysis.ur001_science_entry import InputRejected, admit_inputs, checked_path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'p-full-backend-request-v1'


def stage_progress(stage, **details):
    print(json.dumps({'stage': stage, **details}, sort_keys=True), file=sys.stderr, flush=True)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def record_path(record):
    if type(record) is not dict or set(record) != {'path', 'sha256'}:
        raise ValueError('exact file record required')
    path = checked_path(record['path'])
    if file_sha(path) != record['sha256']:
        raise ValueError('pinned file bytes differ')
    return path


def native_for(root):
    # Frozen source and the sole registered data-channel override are checked
    # before importing its modules. No scientific source is copied or patched.
    from analysis.profile_anchor_steps import _source_identity, _validate_source_ledger
    from analysis.throughput_step_harness import _package
    _source_identity(root, 'p')
    _validate_source_ledger(root, 'p')
    return _package('p_full_native', root)('retrieval_seqm1_run')


def adapt_rows(native, rows, queries, catalog, supported, partitions, durations, features, *, synthetic=False):
    import numpy as np
    import torch
    from vkgr.retrieval_data import catalog_map
    idmap = catalog_map(catalog)
    if supported != sorted(set(supported)) or not set(supported) <= set(catalog):
        raise ValueError('train support/catalog differs')
    if (features.dtype != np.float32 or features.shape != (len(catalog) + 1, 64)
            or not np.isfinite(features).all() or np.any(features[0])):
        raise ValueError('original content shape/dtype/PAD differs')
    for train, collection in ((True, rows), (False, queries)):
        for ordinal, row in enumerate(collection):
            history = row['history']; positions = row['history_positions']; target = row['target_position']
            if (not 1 <= len(history) <= 50 or len(history) != len(positions)
                    or any(i not in idmap for i in history)
                    or any(type(p) is not list or len(p) != 2 or any(type(v) is not int or v < 0 for v in p)
                           or p[0] > 24 for p in positions)
                    or any(a >= b for a, b in zip(positions, positions[1:]))
                    or type(target) is not list or len(target) != 2
                    or any(type(v) is not int or v < 0 for v in target) or positions[-1] >= target
                    or (train and (target[0] > 24 or row.get('example_id') != ordinal
                                   or row['target'] not in supported))
                    or (not train and (target[0] != 25 or row.get('oov') is not (row['target'] not in idmap)))):
                raise ValueError('UR001 history/target identity differs')
    side_module = importlib.import_module(native.__package__ + '.history_side')
    side = side_module.build_side(partitions, {'train': rows, 'validation': queries}, durations)
    return {'rows': [{**r, 'side': s} for r, s in zip(rows, side['train'])],
            'queries': [{**r, 'side': s} for r, s in zip(queries, side['validation'])],
            'catalog': catalog, 'idmap': idmap, 'content': torch.from_numpy(features),
            'support': set(supported), 'supported_indices': [idmap[i] for i in supported],
            'synthetic': synthetic}


def run_session(native, output, data, config, device='cpu', check=None, inputs_binding=None, *, expected_rows):
    # The copied orchestration below preserves the original sequence of calls.
    # The original subset count is replaced by the already verified UR001 count.
    if type(expected_rows) is not int or expected_rows <= 0 or len(data['rows']) != expected_rows:
        raise ValueError('frozen producer train count differs')
    b1, fr, f1 = native.b1, native.fr, native.f1
    validate_config, _model = native.validate_config, native._model
    make_optimizer, binding_for4 = native.make_optimizer, native.binding_for4
    evaluate_model, ablation_g0 = native.evaluate_model, native.ablation_g0
    CHECKPOINT_SCHEMA, MANIFEST_SCHEMA, KIND = native.CHECKPOINT_SCHEMA, native.MANIFEST_SCHEMA, native.KIND
    import torch
    EarlyStop = importlib.import_module(native.__package__ + '.retrieval_hp').EarlyStop
    training = importlib.import_module(native.__package__ + '.retrieval_training')
    EpochCursor, SamplerStats = training.EpochCursor, training.SamplerStats
    target_frequency = importlib.import_module(native.__package__ + '.retrieval_negatives').target_frequency
    synthetic = bool(data.get('synthetic'))
    validate_config(config, synthetic)
    if not synthetic and (device != 'cuda' or inputs_binding is None or len(data['rows']) != expected_rows):
        raise ValueError('unapproved real runtime')
    b1.execution_context(device)
    rows, queries = fr._view(data, config['history_len'])
    seed = config['seed']; model = _model(data, config, device)
    optimizer = make_optimizer(model, config)
    cursor = EpochCursor(len(rows), seed); stats = SamplerStats(); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    negative_rng = random.Random(f'p-b1-negative-v1:{seed}')
    frequency = target_frequency(rows) if config['logq'] else None
    pool = sorted(data['support']) if config['extra_uniform'] else None
    binding = binding_for4(data, config, rows, queries, model, device, inputs_binding)
    view = {**data, 'rows': rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    step = 0; records = []; evaluations = []; best = None; last = None; train_seconds = 0.; eval_seconds = 0.
    diagnostics = {**b1.data_diagnostics(rows, queries, data['support']), 'side_visibility': f1.side_visibility(queries, data['support'])}
    b1._write_progress('train', 0, config['max_epochs'])
    while not stop.stopped:
        if check: check()
        begin = time.monotonic()
        indices = cursor.next(config['batch_size'])
        record = fr.train_step(model, optimizer, [rows[i] for i in indices], view, stats, cursor.epoch, config, negative_rng, pool, frequency)
        step += 1; records.append({**record, 'step': step, 'epoch': cursor.epoch + 1, 'indices_sha256': b1._digest(indices)})
        train_seconds += time.monotonic() - begin
        if cursor.offset != cursor.size:
            continue
        begin = time.monotonic(); epoch = cursor.epoch + 1
        b1._write_progress('evaluate', epoch, config['max_epochs'])
        result, epoch_diag = evaluate_model(model, data, queries, config, check)
        primary = result['metrics']['all']; improved = stop.observe(epoch, primary['recall@50'], primary['ndcg@50'])
        predictions_name = f'predictions-epoch-{epoch:03d}.json'
        predictions_sha = b1._write_json_atomic(output / predictions_name, result['predictions'])
        evaluations.append({'epoch': epoch, 'result': {k: v for k, v in result.items() if k != 'predictions'}, 'predictions_ref': predictions_name,
                            'predictions_sha256': predictions_sha, 'diagnostics': epoch_diag})
        eval_seconds += time.monotonic() - begin
        if check: check()
        state = {'schema': CHECKPOINT_SCHEMA, 'binding_sha256': b1._digest(binding), 'step': step, 'epoch': epoch, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'cursor': cursor.state_dict(), 'negative_rng': negative_rng.getstate(),
                 'stats': b1._stats_value(stats), 'selection': stop.state(), **b1._rng_state()}
        last = {'path': 'last.pt', 'sha256': b1._torch_save_atomic(output / 'last.pt', state), 'step': step, 'epoch': epoch}
        if improved:
            best = {'path': 'best.pt', 'sha256': b1._torch_save_atomic(output / 'best.pt', state), 'step': step, 'epoch': epoch, 'metric': list(stop.best_metric)}
        value = {'schema': MANIFEST_SCHEMA, 'kind': KIND, 'status': 'running', 'config': config,
                 'binding': binding, 'steps': step, 'records': records, 'evaluations': evaluations, 'selection': stop.state(),
                 'stats': b1._stats_value(stats), 'best': best, 'last': last, 'data_diagnostics': diagnostics,
                 'training_seconds': train_seconds, 'evaluation_seconds': eval_seconds,
                 'hit_max_epochs': stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience, 'ablation_g0': None}
        if not stop.stopped:
            b1._write_json_atomic(output / 'manifest.json', value)
            b1._write_progress('train', epoch, config['max_epochs'])
    # finished: best-checkpoint g -> 0 ablation, then the single 'complete' manifest write
    ablation = None
    if model.anchored:
        model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=True)['model'])
        best_epoch = stop.best_epoch
        ablation = ablation_g0(model, data, queries, config, best_epoch, evaluations[best_epoch - 1]['result']['metrics']['all']['recall@50'], check)
    value = {**value, 'status': 'complete', 'ablation_g0': ablation}
    sha = b1._write_json_atomic(output / 'manifest.json', value)
    b1._write_progress('complete', epoch, config['max_epochs'])
    return {**value, 'manifest_sha256': sha}



def strict_json(raw):
    def pairs(entries):
        value = {}
        for key, member in entries:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = member
        return value
    def reject(value):
        raise ValueError('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def invocation(stage, inputs, request, request_sha):
    argv = [sys.executable, '-B', '-m', 'analysis.p_full_training_backend', stage,
            '--inputs', str(inputs), '--request', str(request), '--request-sha256', request_sha]
    if stage != 'preflight':
        argv.append('--execute-approved-science')
    return argv


def read_transport(inputs, request, request_sha):
    # Path/holdout admission always precedes request, manifest, payload or model reads.
    admission_sha = admit_inputs(str(inputs))
    value = strict_json(record_path({'path': str(request), 'sha256': request_sha}).read_bytes())
    fields = {'schema', 'source', 'runtime', 'lock', 'native_root', 'stage2', 'stage2_run',
              'producer_pin', 'items_metadata', 'embedding', 'config', 'ops_root', 'run_ids', 'output_root',
              'admission_sha256', 'predecessor', 'disk_cap_bytes', 'job_contract'}
    if type(value) is not dict or set(value) != fields or value['schema'] != SCHEMA:
        raise ValueError('closed P backend request schema required')
    if value['admission_sha256'] != admission_sha:
        raise ValueError('input admission bytes differ')
    for name in ('native_root', 'stage2', 'stage2_run', 'ops_root', 'output_root'):
        checked_path(value[name])
    for name in ('lock', 'items_metadata', 'embedding'):
        checked_path(value[name]['path'])
    admitted = strict_json(Path(inputs).read_bytes())
    if value['items_metadata']['path'] != admitted['metadata'][1] or value['embedding']['path'] != admitted['metadata'][2]:
        raise ValueError('metadata path differs from admitted whitelist')
    if (type(value['run_ids']) is not dict or set(value['run_ids']) != {'preflight', 'train', 'verify'}
            or len(set(value['run_ids'].values())) != 3
            or any(type(v) is not str or re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,127}', v) is None
                   for v in value['run_ids'].values())):
        raise ValueError('three unique fixed stage job IDs required')
    if type(value['job_contract']) is not dict or set(value['job_contract']) != set(value['run_ids']):
        raise ValueError('closed per-stage job contracts required')
    if value['predecessor'] is not None:
        for key in ('manifest', 'run_dir'):
            checked_path(value['predecessor'][key])
    return value



def physical_records(physical, admitted):
    """The producer stores paths relative to the one fixed full-data prefix."""
    from analysis.ur001_derive_cli import records_for
    from analysis.ur001_science_entry import A_ROOT
    fixed, inventory_sha, prefix = records_for('full')
    base = A_ROOT / prefix
    if (physical.get('profile') != 'full' or physical.get('inventory_sha256') != inventory_sha
            or physical.get('data_root') != str(base)):
        raise ValueError('fixed full physical data root/profile/inventory differs')
    records = physical.get('files')
    if type(records) is not list or len(records) != 26:
        raise ValueError('exact train/validation physical inventory required')
    if [{key: record[key] for key in ('path', 'size', 'sha256')} for record in records] != fixed:
        raise ValueError('fixed physical record inventory differs')
    paths = [str(base / record['path']) for record in records]
    if paths != [*admitted['train'], admitted['validation']]:
        raise ValueError('physical inventory differs from admitted paths')
    return list(zip(records, paths))

def stage_inputs(value, inputs, request, request_sha, stage):
    """Exact ordered launcher input set, derived only from authenticated manifests.

    This parses metadata and stats files; raw data/checkpoint bytes are hashed by
    launcher.prepare and check_job, before any scientific callback is entered.
    """
    result = []; seen = {}
    def add(path, digest, size=None):
        path = checked_path(str(path))
        if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('input pin required')
        if size is None:
            size = path.stat().st_size
        if type(size) is not int or size < 0:
            raise ValueError('input byte count required')
        record = {'path': str(path), 'sha256': digest, 'bytes': size}
        if str(path) in seen:
            if encoded(seen[str(path)]) != encoded(record):
                raise ValueError('conflicting input role pins')
            return
        result.append(record); seen[str(path)] = record
    def metadata(path, digest):
        path = record_path({'path': str(path), 'sha256': digest})
        add(path, digest)
        return strict_json(path.read_bytes())
    def producer(run_dir, pin):
        if set(pin['job_sha256']) != {'run.json', 'terminal.json', 'logs/stdout.log'}:
            raise ValueError('closed producer job snapshot pins required')
        for name in ('run.json', 'terminal.json', 'logs/stdout.log'):
            add(Path(run_dir) / name, pin['job_sha256'][name])
    add(request, request_sha); add(inputs, value['admission_sha256'])
    add(value['lock']['path'], value['lock']['sha256'])
    root = Path(value['stage2'])
    manifest = metadata(root / 'UR001-manifest.json', value['producer_pin']['manifest_sha256'])
    if set(manifest['upstream']) != {'UR001-input-manifest.json'}:
        raise ValueError('closed UR001 upstream manifest required')
    physical = metadata(root / 'UR001-input-manifest.json', manifest['upstream']['UR001-input-manifest.json'])
    names = {'UR001-train.jsonl', 'UR001-validation.jsonl', 'UR001-catalog.json',
             'UR001-supported.json', 'UR001-S0.json', 'UR001-counts.json'}
    if set(manifest['artifacts']) != names:
        raise ValueError('closed UR001 artifact set required')
    for name in sorted(names):
        record = manifest['artifacts'][name]; add(root / name, record['sha256'], record['size'])
    producer(value['stage2_run'], value['producer_pin'])
    admitted = strict_json(Path(inputs).read_bytes())
    from analysis.ur001_science_entry import A_ROOT
    for record, path in physical_records(physical, admitted):
        add(path, record['sha256'], record['size'])
    from analysis.ur001_prepare import load_file_manifest
    official = {r['path']: r for r in load_file_manifest()['files']}
    for path in admitted['metadata']:
        rel = Path(path).relative_to(A_ROOT).as_posix(); record = official[rel]
        add(path, record['sha256'], record['size'])
    for name in ('items_metadata', 'embedding'):
        if value[name]['sha256'] != seen[value[name]['path']]['sha256']:
            raise ValueError('metadata request pin differs from official inventory')
    from analysis.profile_anchor_steps import _source_pins
    pins = _source_pins()['p']; native = Path(value['native_root'])
    for name, digest in sorted(pins['source_files'].items()):
        add(native / name, digest)
    add(native / 'docs/p3c_validation.md', pins['p3c_data_sha256'])
    predecessor = value['predecessor']
    if stage == 'preflight':
        if predecessor is not None:
            raise ValueError('preflight predecessor must be null')
    else:
        if type(predecessor) is not dict or set(predecessor) != {'manifest', 'run_dir', 'pin'}:
            raise ValueError('fixed predecessor input closure required')
        prior = 'preflight' if stage == 'train' else 'train'
        if (Path(predecessor['manifest']) != stage_receipt_path(Path(value['output_root']), prior)
                or Path(predecessor['run_dir']) != Path(value['ops_root']) / 'runs' / value['run_ids'][prior]):
            raise ValueError('fixed predecessor input paths differ')
        receipt = metadata(predecessor['manifest'], predecessor['pin']['manifest_sha256'])
        producer(predecessor['run_dir'], predecessor['pin'])
        if stage == 'verify':
            output = Path(value['output_root'])
            trained = metadata(output / 'manifest.json', receipt['native_manifest_sha256'])
            for name in ('best', 'last'):
                record = trained[name]
                if record['path'] != name + '.pt':
                    raise ValueError('fixed native checkpoint policy required')
                add(output / record['path'], record['sha256'])
            for ordinal, record in enumerate(trained['evaluations'], 1):
                if record['predictions_ref'] != f'predictions-epoch-{ordinal:03d}.json':
                    raise ValueError('fixed native predictions input policy required')
                add(output / record['predictions_ref'], record['predictions_sha256'])
    return result


def job_contract(value, records, request, request_sha, stage):
    contract = value['job_contract'][stage]
    fields = {'env', 'resources', 'ops_source', 'overlay', 'pinned_files', 'stop_mode'}
    if type(contract) is not dict or set(contract) != fields:
        raise ValueError('closed job transport contract required')
    numerical_env = {'PYTHONPATH': str(ROOT / 'src'), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    if encoded(contract['env']) != encoded(numerical_env):
        raise ValueError('frozen job numerical environment differs')
    resources = contract['resources']
    if (type(resources) is not dict or set(resources) != {'cpu_cores', 'ram_mib', 'vram_mib', 'disk_gib', 'wall_seconds'}
            or any(type(resources[k]) is not int or resources[k] < 0 for k in ('cpu_cores', 'ram_mib', 'vram_mib', 'disk_gib'))
            or resources['cpu_cores'] < 1 or resources['ram_mib'] < 1
            or type(resources['wall_seconds']) is not float or not 0 < resources['wall_seconds'] < float('inf')
            or contract['overlay'] is not None or encoded(contract['pinned_files']) != b'{}'
            or contract['stop_mode'] != 'signal'):
        raise ValueError('closed job resources/overlay/pins/stop contract required')
    ops = contract['ops_source']
    if (type(ops) is not dict or set(ops) != {'commit', 'sha256'}
            or type(ops['commit']) is not str or re.fullmatch('[0-9a-f]{40}', ops['commit']) is None
            or type(ops['sha256']) is not str or re.fullmatch('[0-9a-f]{64}', ops['sha256']) is None):
        raise ValueError('fixed job OPS source required')
    if type(value['config']['seed']) is not int:
        raise ValueError('integer scientific seed required for job')
    if (not records or records[0]['path'] != str(request)
            or len({r['path'] for r in records}) != len(records)):
        raise ValueError('unique complete ordered job inputs required')
    run_id = value['run_ids'][stage]; run_dir = Path(value['ops_root']) / 'runs' / run_id
    return {'schema': 'vkgr-run-v1', 'run_id': run_id, 'kind': 'ops', 'cwd': str(ROOT),
            'argv': invocation(stage, records[1]['path'], request, request_sha), 'seed': value['config']['seed'],
            'git': {'commit': value['source']['commit'], 'dirty': False, 'dirty_files': []},
            'config': {'path': str(request), 'sha256': request_sha}, 'inputs': records,
            **{key: contract[key] for key in fields - {'env'}},
            'env': {**numerical_env, 'VKGR_RUN_ID': run_id, 'VKGR_PROGRESS_PATH': str(run_dir / 'progress.json')}}


def check_job(value, records, request, request_sha, stage):
    expected = job_contract(value, records, request, request_sha, stage)
    path = checked_path(str(Path(value['ops_root']) / 'runs' / expected['run_id'] / 'run.json'))
    raw = path.read_bytes(); actual = strict_json(raw)
    if (type(actual) is not dict or set(actual) - set(expected) - {'created_at', 'admission'}
            or any(encoded(actual.get(key)) != encoded(member) for key, member in expected.items())):
        raise ValueError('own complete job contract differs')
    if any(os.environ.get(key) != member for key, member in expected['env'].items()):
        raise ValueError('actual job environment differs')
    stage_progress('raw_binding_begin')
    for record in records:
        path = record_path({'path': record['path'], 'sha256': record['sha256']})
        if path.stat().st_size != record['bytes']:
            raise ValueError('own job input byte count differs')
    stage_progress('raw_binding_complete')
    return hashlib.sha256(raw).hexdigest()


def plan(stage, inputs, request, request_sha):
    value = read_transport(inputs, request, request_sha)
    records = stage_inputs(value, inputs, request, request_sha, stage)
    expected = job_contract(value, records, request, request_sha, stage)
    env = value['job_contract'][stage]['env']; resources = expected['resources']
    launcher = ['--root', value['ops_root'], '--run-id', expected['run_id'], '--kind', expected['kind'],
                '--cwd', str(ROOT), '--config', str(request), '--seed', str(expected['seed'])]
    for key, member in env.items():
        launcher += ['--env', key + '=' + member]
    for key in ('cpu_cores', 'ram_mib', 'vram_mib', 'disk_gib'):
        launcher += ['--' + key.replace('_', '-'), str(resources[key])]
    launcher += ['--wall-hours', str(resources['wall_seconds'] / 3600), '--stop-mode', expected['stop_mode']]
    for record in records:
        launcher += ['--input', record['path']]
    launcher += ['--', *expected['argv']]
    return {'status': 'planned_dry', 'stage': stage, 'training': False, 'payload_parsed': False,
            'submitted': False, 'request_sha256': request_sha, 'expected_job': expected,
            'launcher_argv': launcher, 'inputs': records}


def read_context(inputs, request, request_sha):
    """The complete own-job check in execute precedes this scientific context."""
    stage_progress('context_begin')
    value = read_transport(inputs, request, request_sha)
    metadata = record_path(value['items_metadata']); embedding = record_path(value['embedding'])
    admitted = strict_json(Path(inputs).read_bytes())
    from analysis.ur001_prepare import load_file_manifest
    official = {r['path']: r for r in load_file_manifest()['files']}
    for record in (value['items_metadata'], value['embedding']):
        rel = Path(record['path']).relative_to(Path(admitted['metadata'][0]).parent.parent).as_posix()
        fixed = official[rel]
        if record['sha256'] != fixed['sha256'] or Path(record['path']).stat().st_size != fixed['size']:
            raise ValueError('official metadata inventory differs')
    from analysis.ur001_derive_cli import source_identity, verify_dag
    from analysis.new_instance_prepare import runtime
    from analysis.producer_dag_binding import verify_producer, check_consumer
    source = source_identity(); env = runtime(record_path(value['lock']))
    check_consumer(source, value['source'], env, value['runtime'])
    root = Path(value['stage2'])
    stage_progress('producer_binding_begin')
    proof = verify_producer(root / 'UR001-manifest.json', Path(value['stage2_run']), value['producer_pin'])
    verification = verify_dag(root, require_profile='full')
    stage_progress('producer_binding_complete')
    manifest = json.loads((root / 'UR001-manifest.json').read_bytes())
    physical = json.loads((root / 'UR001-input-manifest.json').read_bytes())
    records = physical_records(physical, admitted)
    stage_progress('raw_binding_begin')
    for record, physical_path in records:
        path = record_path({'path': physical_path, 'sha256': record['sha256']})
        if path.stat().st_size != record['size']:
            raise ValueError('raw byte count differs')
    stage_progress('raw_binding_complete')
    native = native_for(Path(value['native_root']))
    config = native.validate_config(value['config'], synthetic=False)
    if config['arm'] != 'mean':
        raise ValueError('only frozen M1-mean backend supported')
    expected = {**native.default_config(), 'seed': config['seed']}
    if config != expected:
        raise ValueError('frozen M1-mean configuration differs')
    if type(value['disk_cap_bytes']) is not int or value['disk_cap_bytes'] <= 0:
        raise ValueError('positive frozen run disk cap required')
    from analysis import p_full_compact_inputs as compact
    guard = compact.make_guard(native, value)
    cache = compact.cache_root(value['output_root']) / os.environ['VKGR_RUN_ID']
    guard()
    import numpy as np
    from vkgr.retrieval_data import content_join
    stage_progress('content_join_begin')
    with np.load(embedding, allow_pickle=False) as features:
        catalog = json.loads((root / 'UR001-catalog.json').read_bytes())
        content = content_join(catalog, features['item_id'], features['embedding'])
    stage_progress('content_join_complete')
    stage_progress('jsonl_load_begin')
    rows, queries = compact.load_rows(root, cache, manifest, guard)
    stage_progress('jsonl_load_complete')
    supported = json.loads((root / 'UR001-supported.json').read_bytes())
    counts = manifest['counts']
    if len(rows) != counts['train_examples'] or len(queries) != counts['eligible']:
        raise ValueError('producer rows/queries count differs')
    side_module = importlib.import_module(native.__package__ + '.history_side')
    durations = side_module.load_durations(metadata, catalog)
    import pyarrow.parquet as pq
    columns = ['user_id', 'item_id', *side_module.FEEDBACK_BITS, 'timespent']
    def events(path):
        stage_progress('history_side_week_begin', week=path.stem)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=columns, use_threads=False):
            guard()
            yield from batch.to_pylist()
        stage_progress('history_side_week_complete', week=path.stem)
    partitions = ((w, events(Path(path))) for w, path in enumerate(admitted['train']))
    stage_progress('history_side_begin')
    data = compact.adapt_rows(native, rows, queries, catalog, supported, partitions, durations, content,
                              root=cache / 'side', guard=guard)
    stage_progress('history_side_complete')
    binding = {'ur001': verification, 'producer': proof,
               'consumer_source': source, 'consumer_runtime': env, 'test_gate': 'closed_hash_only'}
    # Recheck immutable closure after loading, before exposing data to science.
    verify_dag(root, require_profile='full')
    verify_producer(root / 'UR001-manifest.json', Path(value['stage2_run']), value['producer_pin'])
    if source_identity() != source or runtime(record_path(value['lock'])) != env:
        raise ValueError('consumer source/runtime changed during input read')
    guard()
    stage_progress('context_complete')
    native = compact.facade(native, guard)
    return value, native, data, binding


def close_context(value, binding):
    stage_progress('closure_begin')
    from analysis.ur001_derive_cli import source_identity, verify_dag
    from analysis.new_instance_prepare import runtime
    from analysis.producer_dag_binding import verify_producer
    root = Path(value['stage2'])
    if source_identity() != value['source'] or runtime(record_path(value['lock'])) != value['runtime']:
        raise ValueError('consumer source/runtime changed during science')
    if verify_dag(root, require_profile='full') != binding['ur001']:
        raise ValueError('UR001 closure changed during science')
    if verify_producer(root / 'UR001-manifest.json', Path(value['stage2_run']), value['producer_pin']) != binding['producer']:
        raise ValueError('producer closure changed during science')
    record_path(value['items_metadata']); record_path(value['embedding'])
    native_for(Path(value['native_root']))
    stage_progress('closure_complete')


def stage_receipt_path(output, stage):
    return output.parent / (output.name + '-backend-receipts') / (stage + '.json')


def write_stage_receipt(output, stage, receipt):
    # Transport receipt only. The native manifest and checkpoint bytes are untouched.
    path = stage_receipt_path(output, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)


def check_predecessor(stage, value, binding):
    from analysis.producer_dag_binding import verify_producer
    record = value['predecessor']
    if stage == 'preflight':
        if record is not None:
            raise ValueError('preflight may not consume an undeclared predecessor')
        return None
    prior = 'preflight' if stage == 'train' else 'train'
    if type(record) is not dict or set(record) != {'manifest', 'run_dir', 'pin'}:
        raise ValueError('fixed predecessor closure required')
    expected_path = stage_receipt_path(Path(value['output_root']), prior)
    if checked_path(record['manifest']) != expected_path:
        raise ValueError('fixed predecessor receipt path differs')
    expected_run = Path(value['ops_root']) / 'runs' / value['run_ids'][prior]
    if checked_path(record['run_dir']) != expected_run:
        raise ValueError('fixed predecessor job differs')
    proof = verify_producer(expected_path, expected_run, record['pin'])
    receipt = proof['manifest']
    if (receipt.get('schema') != 'p-full-backend-receipt-v1' or receipt.get('stage') != prior
            or receipt.get('binding') != binding or receipt.get('engineering') is not False
            or receipt.get('test_read') is not False or receipt.get('scientific_verified') is not False
            or receipt.get('training') is not (prior == 'train')
            or receipt.get('status') != ('complete' if prior == 'train' else 'verified')):
        raise ValueError('fixed predecessor semantics differ')
    if prior == 'train' and file_sha(Path(value['output_root']) / 'manifest.json') != receipt.get('native_manifest_sha256'):
        raise ValueError('native trained manifest differs from producer receipt')
    return receipt


def execute(stage, inputs, request, request_sha):
    transport = read_transport(inputs, request, request_sha)
    records = stage_inputs(transport, inputs, request, request_sha, stage)
    run_sha = check_job(transport, records, request, request_sha, stage)
    value, native, data, binding = read_context(inputs, request, request_sha)
    from analysis.p_full_parallel_eval import bind
    native = bind(native.compact_original, Path(value['output_root']).parent / (Path(value['output_root']).name + '-parallel-eval'),
                  output_root=Path(value['output_root']), disk_cap_bytes=value['disk_cap_bytes'])
    from analysis import p_full_compact_inputs as compact
    native = compact.facade(native, compact.make_guard(native, value))
    binding = {**binding, 'parallel_evaluation': native.parallel_eval_identity}
    if encoded(value) != encoded(transport):
        raise ValueError('request changed before scientific context')
    predecessor = check_predecessor(stage, value, binding)
    run_id = value['run_ids'][stage]
    native.b1.execution_context('cuda')
    output = checked_path(value['output_root'])
    for protected in (ROOT, Path(value['native_root']), Path(value['stage2'])):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('exclusive output separate from source/input required')
    guard = native.b1.resource_guard(output, value['disk_cap_bytes'])
    guard()
    if stage == 'preflight':
        receipt = {'schema': 'p-full-backend-receipt-v1', 'status': 'verified', 'verified': True,
                'stage': stage, 'scientific_verified': False, 'source': value['source'], 'runtime': value['runtime'],
                'engineering': False, 'training': False, 'payload_written': False,
                'engineering_cache_written': True,
                'test_read': False, 'data_opened': True, 'run_id': run_id,
                'request_sha256': request_sha, 'binding': binding,
                'training_rows': len(data['rows']), 'validation_queries': len(data['queries']),
                'assembly_workers': 0, 'native_scientific_functions_unchanged': True}
        close_context(value, binding)
        if check_job(value, records, request, request_sha, stage) != run_sha:
            raise ValueError('own job changed during preflight')
        receipt['run_manifest_sha256'] = run_sha
        write_stage_receipt(output, stage, receipt)
        return receipt
    config = value['config']
    if stage == 'train':
        result = run_session(native, output, data, config, 'cuda', check=guard, inputs_binding=binding,
                             expected_rows=binding['ur001']['counts']['train_examples'])
        details = {'status': result['status'], 'verified': False, 'training': True,
                   'manifest_sha256': result['manifest_sha256']}
    else:
        manifest = output / 'manifest.json'
        result = native.verify_run(manifest, predecessor['native_manifest_sha256'], data, config, 'cuda', check=guard, inputs_binding=binding)
        details = {'status': result['status'], 'verified': result['status'] == 'verified', 'training': False,
                   'native_verification': result, 'manifest_sha256': file_sha(manifest)}
    close_context(value, binding)
    check_predecessor(stage, value, binding)
    if check_job(value, records, request, request_sha, stage) != run_sha:
        raise ValueError('own job changed during science')
    guard()
    receipt = {'schema': 'p-full-backend-receipt-v1', 'engineering': False, 'test_read': False,
               'stage': stage, 'run_id': run_id, 'run_manifest_sha256': run_sha, 'request_sha256': request_sha, 'binding': binding,
               'source': value['source'], 'runtime': value['runtime'], **details,
               'native_manifest_sha256': details['manifest_sha256'],
               'scientific_verified': stage == 'verify', 'verified': True}
    write_stage_receipt(output, stage, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('dry')
    for name in ('preflight', 'train', 'verify', 'plan'):
        command = sub.add_parser(name)
        command.add_argument('--inputs', type=Path, required=True)
        command.add_argument('--request', type=Path, required=True)
        command.add_argument('--request-sha256', required=True)
        if name == 'plan':
            command.add_argument('--stage', choices=('preflight', 'train', 'verify'), required=True)
        elif name != 'preflight':
            command.add_argument('--execute-approved-science', action='store_true', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'dry':
            result = {'status': 'dry', 'engineering': False, 'training': False, 'test_read': False,
                      'assembly_workers': 0, 'real_preflight': False}
        elif args.command == 'plan':
            result = plan(args.stage, args.inputs, args.request, args.request_sha256)
        else:
            result = execute(args.command, args.inputs, args.request, args.request_sha256)
    except (InputRejected, ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        stage_progress('entry_failed', error_type=type(exc).__name__, message=str(exc))
        result = {'status': 'HOLDOUT_INPUT_REJECTED' if isinstance(exc, InputRejected) and exc.holdout else 'REJECTED',
                  'data_opened': False if isinstance(exc, InputRejected) else 'UNKNOWN',
                  'training': args.command == 'train', 'verified': False, 'error_type': type(exc).__name__}
        print(json.dumps(result, sort_keys=True)); return 2
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
