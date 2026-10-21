"""Mechanical P PF preparation and bounded original-assembly/adapter comparison.

No scheduler, submit, run_session, checkpoint or scientific acceptance command.
The assembly comparison does not restore the original pinned subset loader.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from analysis import p_full_training_backend as backend


def input_roles():
    from analysis.ur001_science_entry import A_ROOT, SCHEMA
    return {'schema': SCHEMA, 'train': [str(A_ROOT / f'subsamples/ur0.01/train/week_{w:02}.parquet') for w in range(25)],
            'validation': str(A_ROOT / 'subsamples/ur0.01/validation/week_25.parquet'),
            'metadata': [str(A_ROOT / 'metadata' / n) for n in ('users_metadata.parquet', 'items_metadata.parquet', 'item_embeddings.npz')]}


def harvest_pin(manifest, run, digest):
    value = backend.strict_json(backend.record_path({'path': str(manifest), 'sha256': digest}).read_bytes())
    run = Path(run)
    files = ('run.json', 'terminal.json', 'logs/stdout.log')
    paths = {name: backend.checked_path(str(run / name)) for name in files}
    lines = [line for line in paths['logs/stdout.log'].read_bytes().splitlines() if line.strip()]
    if not lines:
        raise ValueError('original producer stdout required')
    return {'manifest_sha256': digest, 'source': value['source'], 'runtime': value['runtime'],
            'job_sha256': {name: backend.file_sha(path) for name, path in paths.items()},
            'receipt': backend.strict_json(lines[-1])}


def measure(value, key, actual):
    if value.get(key) is not None and backend.encoded(value[key]) != backend.encoded(actual):
        raise ValueError('template ' + key + ' differs from measured original')
    value[key] = actual


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(backend.encoded(value) + b'\n')


def frozen_config(native_root, config_path, digest):
    config = backend.strict_json(backend.record_path({'path': str(config_path), 'sha256': digest}).read_bytes())
    native = backend.native_for(native_root)
    native.validate_config(config, synthetic=False)
    if config['arm'] != 'mean':
        raise ValueError('original frozen mean config required')
    rel = 'configs/seqm1/p-seqm1-mean-s' + str(config['seed']) + '.json'
    raw = subprocess.check_output(['git', '-C', str(native_root), 'show', 'HEAD:' + rel])
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('config bytes differ from original frozen mean config')
    return config


def prepare(template, digest, config_path, config_sha, output, *, for_assembly_parity=False, for_helper_preflight=False):
    backend.stage_progress('prepare_begin')
    if for_assembly_parity and for_helper_preflight:
        raise ValueError('one helper plan stage required')
    from analysis.ur001_derive_cli import source_identity, verify_dag
    from analysis.new_instance_prepare import runtime
    from analysis.producer_dag_binding import verify_producer
    value = backend.strict_json(backend.record_path({'path': str(template), 'sha256': digest}).read_bytes())
    if value.get('schema') != backend.SCHEMA or value.get('predecessor') is not None:
        raise ValueError('preflight request template with null predecessor required')
    output = backend.checked_path(str(output))
    if output.exists() or output == backend.ROOT or backend.ROOT in output.parents:
        raise ValueError('new output outside clean consumer source required')
    for name in ('native_root', 'stage2', 'stage2_run', 'ops_root', 'output_root'):
        backend.checked_path(value[name])
    backend.stage_progress('original_evidence_begin')
    config = frozen_config(Path(value['native_root']), config_path, config_sha)
    measure(value, 'config', config)
    measure(value, 'source', source_identity())
    measure(value, 'runtime', runtime(backend.record_path(value['lock'])))
    roles = input_roles()
    for key, path in [('items_metadata', roles['metadata'][1]), ('embedding', roles['metadata'][2])]:
        measure(value, key, {'path': path, 'sha256': backend.file_sha(backend.checked_path(path))})
    manifest = Path(value['stage2']) / 'UR001-manifest.json'
    old_pin = value['producer_pin']; manifest_sha = old_pin['manifest_sha256']
    pin = harvest_pin(manifest, Path(value['stage2_run']), manifest_sha)
    if set(old_pin) != {'manifest_sha256'} and backend.encoded(old_pin) != backend.encoded(pin):
        raise ValueError('template producer pin differs from original')
    value['producer_pin'] = pin
    verify_producer(manifest, Path(value['stage2_run']), pin)
    verify_dag(Path(value['stage2']), require_profile='full')
    backend.stage_progress('original_evidence_complete')
    # Own run IDs must be unused. No scheduler method is imported or invoked.
    for run_id in value['run_ids'].values():
        if (Path(value['ops_root']) / 'runs' / run_id).exists():
            raise ValueError('requested job already exists')
        queue = Path(value['ops_root']) / 'queue'
        if any(path.stem.endswith('-' + run_id) for path in queue.glob('*/*.json')):
            raise ValueError('requested job already queued')
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs.json'; request = output / 'request.json'
    write_new(inputs, roles)
    measure(value, 'admission_sha256', backend.admit_inputs(str(inputs)))
    write_new(request, value)
    request_sha = backend.file_sha(request)
    backend.stage_progress('plan_begin')
    planned = backend.plan('preflight', inputs, request, request_sha)
    if for_assembly_parity or for_helper_preflight:
        command = 'assembly-parity' if for_assembly_parity else 'preflight'
        argv = helper_argv(command, inputs, request, request_sha)
        planned['expected_job']['argv'] = argv
        separator = planned['launcher_argv'].index('--')
        planned['launcher_argv'] = planned['launcher_argv'][:separator + 1] + argv
        planned['stage'] = command
    write_new(output / 'plan.json', planned)
    backend.stage_progress('prepare_complete')
    return {'schema': 'p-full-engineering-receipt-v1', 'status': 'prepared_dry', 'training_submitted': False,
            'payload_written': False, 'real_preflight': False, 'template_sha256': digest,
            'config_sha256': config_sha, 'inputs_sha256': backend.file_sha(inputs),
            'request_sha256': request_sha, 'plan_sha256': backend.file_sha(output / 'plan.json')}


def capture(native, data, config, device, guard):
    from analysis.tensor_cache_parity import freeze, observe_p_inputs
    training = importlib.import_module(native.__package__ + '.retrieval_training')
    negatives = importlib.import_module(native.__package__ + '.retrieval_negatives')
    model = native._model(data, config, device); optimizer = native.make_optimizer(model, config)
    rows, queries = native.fr._view(data, config['history_len'])
    cursor = training.EpochCursor(len(rows), config['seed']); stats = training.SamplerStats()
    rng = random.Random(f"p-b1-negative-v1:{config['seed']}")
    frequency = negatives.target_frequency(rows) if config['logq'] else None
    pool = sorted(data['support']) if config['extra_uniform'] else None
    result = {'steps': []}
    for _ in range(3):
        guard()
        batch = [rows[i] for i in cursor.next(config['batch_size'])]
        trace = {}
        step = observe_p_inputs(native.fr.train_step, trace)
        record = step(model, optimizer, batch, data, stats, cursor.epoch, config, rng, pool, frequency)
        tensors = native.fr._tensors2(batch, data['idmap'], max(len(r['history']) for r in batch), config['feature'])
        backend.stage_progress('step_complete', step=_ + 1)
        result['steps'].append({'tensors': freeze([tensors, trace]), 'record': freeze(record),
                                'parameters': freeze(model.state_dict()), 'optimizer': freeze(optimizer.state_dict()),
                                'cursor': freeze(cursor.state_dict()), 'negative_rng': freeze(rng.getstate())})
    guard()
    backend.stage_progress('evaluation_begin')
    users, items, norms = native.fr.export_vectors_side2(model, data['catalog'], queries, data['idmap'],
                            config['batch_size'], config['history_len'], config['feature'], check=guard)
    result['evaluation'] = dict(users=freeze(users), items=freeze(items), norms=freeze(norms),
        scores=freeze(users.astype('float64') @ items.astype('float64').T),
        original_evaluator=freeze(native.evaluate_model(model, data, queries, config, check=guard)))
    backend.stage_progress('evaluation_complete')
    return result


def reference_assembly(native, value, inputs, rows_count, query_count, guard=lambda: None):
    """Independent original assembly functions, explicitly not b1.real_inputs."""
    import itertools
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    root = Path(value['stage2'])
    def read_prefix(name, count):
        with (root / name).open() as stream:
            return [json.loads(line) for line in itertools.islice(stream, count)]
    rows = read_prefix('UR001-train.jsonl', rows_count)
    queries = read_prefix('UR001-validation.jsonl', query_count)
    catalog = json.loads((root / 'UR001-catalog.json').read_bytes())
    supported = json.loads((root / 'UR001-supported.json').read_bytes())
    rd = importlib.import_module(native.__package__ + '.retrieval_data')
    side = importlib.import_module(native.__package__ + '.history_side')
    durations = side.load_durations(backend.record_path(value['items_metadata']), catalog)
    roles = backend.strict_json(Path(inputs).read_bytes())
    def events(path):
        backend.stage_progress('reference_history_week_begin', week=Path(path).stem)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,
                columns=['user_id', 'item_id', *side.FEEDBACK_BITS, 'timespent'], use_threads=False):
            guard()
            yield from batch.to_pylist()
        backend.stage_progress('reference_history_week_complete', week=Path(path).stem)
    parts = ((week, events(path)) for week, path in enumerate(roles['train']))
    loaded = side.build_side(parts, {'train': rows, 'validation': queries}, durations)
    with np.load(backend.record_path(value['embedding']), allow_pickle=False) as features:
        content = torch.from_numpy(rd.content_join(catalog, features['item_id'], features['embedding']))
    idmap = rd.catalog_map(catalog)
    return {'rows': [{**r, 'side': s} for r, s in zip(rows, loaded['train'])],
            'queries': [{**q, 'side': s} for q, s in zip(queries, loaded['validation'])],
            'catalog': catalog, 'idmap': idmap, 'content': content, 'support': set(supported),
            'supported_indices': [idmap[i] for i in supported], 'synthetic': False}


def helper_argv(command, inputs, request, digest):
    return [sys.executable, '-B', '-m', 'analysis.p_full_engineering', command,
            '--inputs', str(inputs), '--request', str(request), '--request-sha256', digest]


def parity_argv(inputs, request, digest):
    return helper_argv('assembly-parity', inputs, request, digest)


def check_engineering_job(value, records, request, digest, command='assembly-parity'):
    expected = backend.job_contract(value, records, request, digest, 'preflight')
    expected['argv'] = helper_argv(command, records[1]['path'], request, digest)
    job = Path(value['ops_root']) / 'runs' / expected['run_id'] / 'run.json'
    raw = backend.checked_path(str(job)).read_bytes()
    actual = backend.strict_json(raw)
    if (set(actual) - set(expected) - {'created_at', 'admission'}
            or any(backend.encoded(actual.get(k)) != backend.encoded(v) for k, v in expected.items())
            or any(os.environ.get(k) != v for k, v in expected['env'].items())):
        raise ValueError('engineering own complete job contract differs')
    for rec in records:
        path = backend.record_path({'path': rec['path'], 'sha256': rec['sha256']})
        if path.stat().st_size != rec['bytes']:
            raise ValueError('engineering input bytes differ')
    return hashlib.sha256(raw).hexdigest()


def assembly_parity(inputs, request, digest, *, read_only=False):
    from analysis.tensor_cache_parity import freeze
    backend.stage_progress('helper_preflight_begin' if read_only else 'assembly_parity_begin')
    value = backend.read_transport(inputs, request, digest)
    records = backend.stage_inputs(value, inputs, request, digest, 'preflight')
    def check_job():
        return check_engineering_job(value, records, request, digest, 'preflight' if read_only else 'assembly-parity')
    job_sha = check_job()
    value, native, data, binding = backend.read_context(inputs, request, digest)
    native.b1.execution_context('cuda')
    config = value['config']; count = config['batch_size'] * 3
    if len(data['rows']) < count or len(data['queries']) < 32:
        raise ValueError('fixed engineering prefix unavailable')
    adapted = {**data, 'rows': data['rows'][:count], 'queries': data['queries'][:32]}
    guard = native.b1.resource_guard(Path(value['output_root']), value['disk_cap_bytes'])
    guard()
    backend.stage_progress('original_assembly_begin')
    reference = reference_assembly(native, value, inputs, count, 32, guard)
    backend.stage_progress('original_assembly_complete')
    def identity(d):
        return freeze({**d, 'support': sorted(d['support'])})
    if identity(adapted) != identity(reference):
        raise ValueError('original assembly/new adapter input bytes differ')
    if not read_only:
        backend.stage_progress('reference_steps_begin')
        left = capture(native, reference, config, 'cuda', guard)
        backend.stage_progress('adapter_steps_begin')
        right = capture(native, adapted, config, 'cuda', guard)
        if left != right:
            raise ValueError('original assembly/new adapter step or scoring differs')
    backend.close_context(value, binding)
    if check_job() != job_sha:
        raise ValueError('engineering job changed')
    backend.stage_progress('helper_preflight_complete' if read_only else 'assembly_parity_complete')
    return {'schema': 'p-full-engineering-receipt-v1', 'status': 'preflight_verified' if read_only else 'assembly_parity_verified',
            'verified': True, 'engineering': True, 'scientific_verified': False,
            'original_formal_loader_verified': False, 'training_submitted': False, 'checkpoint_written': False,
            'test_read': False, 'steps': 0 if read_only else 3, 'model_created': not read_only, 'training_rows': count, 'validation_queries': 32,
            'run_manifest_sha256': job_sha, 'request_sha256': digest, 'binding': binding,
            'comparison_sha256': hashlib.sha256(backend.encoded(identity(reference) if read_only else left)).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True); sub.add_parser('dry')
    p = sub.add_parser('prepare')
    for name in ('template', 'config', 'output'): p.add_argument('--' + name, type=Path, required=True)
    for name in ('template-sha256', 'config-sha256'): p.add_argument('--' + name, required=True)
    stages = p.add_mutually_exclusive_group()
    stages.add_argument('--for-assembly-parity', action='store_true')
    stages.add_argument('--for-helper-preflight', action='store_true')
    for command in ('assembly-parity', 'preflight'):
        p = sub.add_parser(command)
        for name in ('inputs', 'request'): p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'dry':
            result = {'training_submitted': False, 'real_preflight': False, 'original_formal_loader_verified': False}
        elif args.command == 'prepare':
            result = prepare(args.template, args.template_sha256, args.config, args.config_sha256, args.output, for_assembly_parity=args.for_assembly_parity, for_helper_preflight=args.for_helper_preflight)
        else:
            result = assembly_parity(args.inputs, args.request, args.request_sha256, read_only=args.command == 'preflight')
    except backend.InputRejected as exc:
        print(json.dumps({'status': 'HOLDOUT_INPUT_REJECTED' if exc.holdout else 'REJECTED',
                          'verified': False, 'data_opened': False, 'training_submitted': False})); return 2
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(json.dumps({'status': 'REJECTED', 'verified': False, 'error_type': type(exc).__name__})); return 2
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
