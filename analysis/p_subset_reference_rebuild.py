"""Fresh subset reference through original producers and parameterized loader.

No legacy pin patch, checkpoint, scheduler or scientific acceptance decision.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys

from analysis import p_full_training_backend as backend
from analysis import p_full_engineering as helper

SCHEMA = 'p-subset-reference-request-v1'
STAGES = {'preflight': 'preflight', 'rebuild': 'train', 'compare': 'verify'}
OLD_PREPARED = '7f3ccd70c02157542e94f41f39e62988fbd650b4be8976455ff20cfe1e4394d2'
OLD_INPUTS = 'a12b98003aa2fbb58df83bc70bb448e426d01c2a4cc2e6357bf1f95778f23e38'
REFERENCE_FILES = {'inputs.json', 'queries/validation-queries.json',
    *('prepared/' + n for n in ('catalog.json', 'content.npy', 'examples.jsonl', 'idmap.json', 'support.json', 'prepare-manifest.json')),
    *('history/' + n for n in ('train-l100.json', 'train-l50.json', 'train-l20.json', 'validation-l100.json', 'validation-l50.json', 'validation-l20.json', 'train-slices.json', 'length-stats.json', 'manifest.json')),
    *('side/' + n for n in ('side-train-l100.json', 'side-validation-l100.json', 'manifest.json'))}


def canonical_identity(data):
    from analysis.tensor_cache_parity import freeze
    def ordered(value):
        if isinstance(value, dict):
            return {k: ordered(value[k]) for k in sorted(value)}
        if isinstance(value, list):
            return [ordered(v) for v in value]
        if isinstance(value, set):
            return sorted(value)
        return value
    return freeze(ordered(data))


def original(native, name):
    return importlib.import_module(native.__package__ + '.' + name)


def partitions(roles, guard):
    import pyarrow.parquet as pq
    result = []
    for week, path in enumerate(roles['train']):
        guard(); backend.stage_progress('subset_original_week', week=week)
        rows = []
        columns = ['user_id', 'item_id', 'like', 'share', 'bookmark', 'open_comments', 'click_on_author', 'dislike', 'timespent']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=columns, use_threads=False):
            guard(); rows.extend(batch.to_pylist())
        result.append((week, rows))
    return result


def load_reference(native, output, manifest):
    output = Path(output)
    if set(manifest['artifacts']) != REFERENCE_FILES:
        raise ValueError('closed reference artifact set required')
    for name, digest in manifest['artifacts'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('relative reference artifact required')
        backend.record_path({'path': str(output / name), 'sha256': digest})
    spec = backend.strict_json((output / 'inputs.json').read_bytes())
    native.fr.validate_inputs(spec, 'f3_f4_recency')
    expected = {'prepared': str(output / 'prepared'), 'holdout': str(output / 'queries'),
                'history_manifest': str(output / 'history/manifest.json'), 'history_sha256': manifest['history_sha256'],
                'side_manifest': str(output / 'side/manifest.json'), 'side_sha256': manifest['side_sha256'],
                'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None}
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError('reference fixed sidecar paths/pins differ')
    experiment = original(native, 'retrieval_experiment')
    data = experiment.load_training_inputs(output / 'prepared', output / 'queries',
                manifest['prepared_sha256'], manifest['original_source_fingerprint'], manifest['query_sha256'])
    data['synthetic'] = False
    data, binding = native.fr.apply_feat02(data, spec, 50)
    # Loader provenance fields are not part of model input identity.
    return {k: v for k, v in data.items() if k not in ('prepared', 'prepared_sha256', 'query_sha256')}, binding


def rebuild_reference(native, roles, items, embedding, output, binding, guard):
    import numpy as np
    from analysis.ur001_derivation import parquet_rows
    rd, audit = original(native, 'retrieval_data'), original(native, 'data_audit')
    history, side = original(native, 'history_v2'), original(native, 'history_side')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    parts = partitions(roles, guard)
    guard()
    validation = (25, list(parquet_rows(Path(roles['validation']), 1)))
    catalog = sorted({r['item_id'] for _, rows in parts for r in rows})
    supported = sorted({r['item_id'] for _, rows in parts for r in rows if r['like']})
    examples = list(rd.prefix_examples(parts))
    queries = audit.build_queries(parts, validation, audit.approved_policy(), set(catalog))['queries']
    with np.load(embedding, allow_pickle=False) as values:
        features = rd.content_join(catalog, values['item_id'], values['embedding'])
    rd.write_prepared(output / 'prepared', catalog, features, examples, binding, supported)
    (output / 'queries').mkdir()
    with (output / 'queries/validation-queries.json').open('xb') as stream:
        stream.write(audit.canonical_bytes(queries))
    guard()
    h = history.write_prepared(output / 'history', parts, examples, queries, catalog, binding)
    s = side.write_side(output / 'side', output / 'history/manifest.json', h['manifest_sha256'],
                        parts, side.load_durations(items, catalog), binding)
    spec = {'schema': 'p-feat02-inputs-v1', 'prepared': str(output / 'prepared'),
            'holdout': str(output / 'queries'), 'raw_root': str(Path(roles['train'][0]).parent.parent),
            'history_manifest': str(output / 'history/manifest.json'), 'history_sha256': h['manifest_sha256'],
            'disk_cap_bytes': 64 * 1024**3, 'test_gate': 'closed_hash_only',
            'side_manifest': str(output / 'side/manifest.json'), 'side_sha256': s['manifest_sha256'],
            'items_metadata_sha256': backend.file_sha(items),
            'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None}
    helper.write_new(output / 'inputs.json', spec)
    manifest = {'prepared_sha256': backend.file_sha(output / 'prepared/prepare-manifest.json'),
                'query_sha256': backend.file_sha(output / 'queries/validation-queries.json'),
                'history_sha256': h['manifest_sha256'], 'side_sha256': s['manifest_sha256'],
                'original_source_fingerprint': binding['consumer']['source_fingerprint'],
                'original_parameterized_loader_verified': True, 'original_legacy_wrapper_used': False,
                'artifacts': {p.relative_to(output).as_posix(): backend.file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    manifest['old_pins'] = {'prepared_sha256': OLD_PREPARED, 'inputs_sha256': OLD_INPUTS,
                            'prepared_equal': manifest['prepared_sha256'] == OLD_PREPARED,
                            'inputs_equal': backend.file_sha(output / 'inputs.json') == OLD_INPUTS,
                            'policy': 'report_only'}
    load_reference(native, output, manifest)
    guard()
    return manifest


def roles_for(data, metadata_root):
    data, metadata_root = backend.checked_path(str(data)), backend.checked_path(str(metadata_root))
    return {'schema': 'p-subset-reference-inputs-v1',
            'train': [str(data / f'train/week_{w:02d}.parquet') for w in range(25)],
            'validation': str(data / 'validation/week_25.parquet'),
            'metadata': [str(metadata_root / name) for name in ('users_metadata.parquet', 'items_metadata.parquet', 'item_embeddings.npz')]}


def read_transport(inputs, request, digest):
    # Reject all alternate/test paths before request, producer or payload reads.
    roles = backend.strict_json(backend.checked_path(str(inputs)).read_bytes())
    if type(roles) is not dict or set(roles) != {'schema', 'train', 'validation', 'metadata'}:
        raise ValueError('closed subset input roles required')
    from analysis.ur001_derive_cli import secure
    for path in [*roles['train'], roles['validation'], *roles['metadata']]:
        secure(backend.checked_path(path))
    if len(roles['train']) != 25 or len(roles['metadata']) != 3:
        raise ValueError('fixed subset train/validation/metadata roles required')
    data = Path(roles['train'][0]).parent.parent
    if roles != roles_for(data, Path(roles['metadata'][0]).parent):
        raise ValueError('fixed subset train/validation/metadata paths required')
    value = backend.strict_json(backend.record_path({'path': str(request), 'sha256': digest}).read_bytes())
    fields = {'schema', 'source', 'runtime', 'lock', 'native_root', 'stage2', 'stage2_run', 'producer_pin',
              'config', 'config_file', 'ops_root', 'run_ids', 'output_root', 'admission_sha256', 'reference',
              'disk_cap_bytes', 'job_contract'}
    if type(value) is not dict or set(value) != fields or value['schema'] != SCHEMA:
        raise ValueError('closed subset reference request required')
    if value['admission_sha256'] != backend.file_sha(inputs):
        raise ValueError('subset input admission bytes differ')
    for name in ('native_root', 'stage2', 'stage2_run', 'ops_root', 'output_root'):
        backend.checked_path(value[name])
    if (type(value['run_ids']) is not dict or set(value['run_ids']) != set(STAGES.values())
            or len(set(value['run_ids'].values())) != 3
            or any(type(v) is not str or re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,127}', v) is None for v in value['run_ids'].values())
            or set(value['job_contract']) != set(STAGES.values())):
        raise ValueError('three unique fixed jobs/contracts required')
    return value, roles


def stage_inputs(value, roles, inputs, request, digest, stage):
    from analysis.ur001_derive_cli import records_for
    from analysis.ur001_prepare import load_file_manifest
    from analysis.profile_anchor_steps import _source_pins
    records = []; seen = set()
    def add(path, sha, size=None):
        path = backend.checked_path(str(path))
        if str(path) in seen:
            raise ValueError('duplicate subset input role')
        seen.add(str(path)); records.append({'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size if size is None else size})
    def metadata(path, sha):
        path = backend.record_path({'path': str(path), 'sha256': sha}); add(path, sha)
        return backend.strict_json(path.read_bytes())
    def producer(run, pin):
        if set(pin['job_sha256']) != {'run.json', 'terminal.json', 'logs/stdout.log'}:
            raise ValueError('closed producer job pins required')
        for name in ('run.json', 'terminal.json', 'logs/stdout.log'):
            add(Path(run) / name, pin['job_sha256'][name])
    add(request, digest); add(inputs, value['admission_sha256'])
    add(value['lock']['path'], value['lock']['sha256'])
    add(value['config_file']['path'], value['config_file']['sha256'])
    raw, _, _ = records_for('subset')
    root = Path(roles['train'][0]).parent.parent
    expected = [str(Path(p).relative_to(root)) for p in [*roles['train'], roles['validation']]]
    if expected != [r['path'] for r in raw]:
        raise ValueError('fixed subset physical roles differ')
    for row in raw:
        add(root / row['path'], row['sha256'], row['size'])
    official = {r['path']: r for r in load_file_manifest()['files']}
    for path in roles['metadata']:
        row = official['metadata/' + Path(path).name]; add(path, row['sha256'], row['size'])
    stage2 = Path(value['stage2'])
    manifest = metadata(stage2 / 'UR001-manifest.json', value['producer_pin']['manifest_sha256'])
    if set(manifest['upstream']) != {'UR001-input-manifest.json'}:
        raise ValueError('closed subset upstream required')
    physical = metadata(stage2 / 'UR001-input-manifest.json', manifest['upstream']['UR001-input-manifest.json'])
    if (physical.get('data_root') != str(root)
            or [{k: r[k] for k in ('path', 'size', 'sha256')} for r in physical['files']] != raw):
        raise ValueError('subset producer physical inventory differs')
    names = {'UR001-train.jsonl', 'UR001-validation.jsonl', 'UR001-catalog.json', 'UR001-supported.json', 'UR001-S0.json', 'UR001-counts.json'}
    if set(manifest['artifacts']) != names:
        raise ValueError('closed subset artifact set required')
    for name in sorted(names):
        row = manifest['artifacts'][name]; add(stage2 / name, row['sha256'], row['size'])
    producer(value['stage2_run'], value['producer_pin'])
    pins = _source_pins()['p']; native = Path(value['native_root'])
    for name, sha in sorted(pins['source_files'].items()):
        add(native / name, sha)
    add(native / 'docs/p3c_validation.md', pins['p3c_data_sha256'])
    reference = value['reference']
    if stage != 'compare':
        if reference is not None:
            raise ValueError('reference must be null before comparison')
    else:
        if type(reference) is not dict or set(reference) != {'manifest', 'run_dir', 'pin'}:
            raise ValueError('closed reference producer required')
        if (Path(reference['manifest']) != Path(value['output_root']) / 'reference-manifest.json'
                or Path(reference['run_dir']) != Path(value['ops_root']) / 'runs' / value['run_ids']['train']):
            raise ValueError('fixed reference producer paths differ')
        ref = metadata(reference['manifest'], reference['pin']['manifest_sha256'])
        if set(ref['artifacts']) != REFERENCE_FILES:
            raise ValueError('closed reference artifact set required')
        for name, sha in sorted(ref['artifacts'].items()):
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('relative reference artifact required')
            add(Path(value['output_root']) / name, sha)
        producer(reference['run_dir'], reference['pin'])
    return records


def invocation(stage, inputs, request, digest):
    return [sys.executable, '-B', '-m', 'analysis.p_subset_reference_rebuild', stage,
            '--inputs', str(inputs), '--request', str(request), '--request-sha256', digest]


def job_contract(value, records, request, digest, stage):
    job = backend.job_contract(value, records, request, digest, STAGES[stage])
    job['argv'] = invocation(stage, records[1]['path'], request, digest)
    return job


def check_job(value, records, request, digest, stage):
    expected = job_contract(value, records, request, digest, stage)
    job = Path(value['ops_root']) / 'runs' / expected['run_id'] / 'run.json'
    raw = backend.checked_path(str(job)).read_bytes(); actual = backend.strict_json(raw)
    if (type(actual) is not dict or set(actual) - set(expected) - {'created_at', 'admission'}
            or any(backend.encoded(actual.get(k)) != backend.encoded(v) for k, v in expected.items())
            or any(os.environ.get(k) != v for k, v in expected['env'].items())):
        raise ValueError('subset own complete job contract differs')
    for record in records:
        path = backend.record_path({'path': record['path'], 'sha256': record['sha256']})
        if path.stat().st_size != record['bytes']:
            raise ValueError('subset input byte count differs')
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def plan(stage, inputs, request, digest):
    value, roles = read_transport(inputs, request, digest)
    records = stage_inputs(value, roles, inputs, request, digest, stage)
    job = job_contract(value, records, request, digest, stage)
    resources = job['resources']
    argv = ['--root', value['ops_root'], '--run-id', job['run_id'], '--kind', 'ops',
            '--cwd', str(backend.ROOT), '--config', str(request), '--seed', str(job['seed'])]
    for key, member in value['job_contract'][STAGES[stage]]['env'].items():
        argv += ['--env', key + '=' + member]
    for key in ('cpu_cores', 'ram_mib', 'vram_mib', 'disk_gib'):
        argv += ['--' + key.replace('_', '-'), str(resources[key])]
    argv += ['--wall-hours', str(resources['wall_seconds'] / 3600), '--stop-mode', job['stop_mode']]
    for record in records:
        argv += ['--input', record['path']]
    return {'status': 'planned_dry', 'stage': stage, 'submitted': False, 'training_submitted': False,
            'inputs': records, 'expected_job': job, 'launcher_argv': [*argv, '--', *job['argv']]}


def context(value):
    from analysis.ur001_derive_cli import source_identity, verify_dag
    from analysis.new_instance_prepare import runtime
    from analysis.producer_dag_binding import check_consumer, verify_producer
    check_consumer(source_identity(), value['source'], runtime(backend.record_path(value['lock'])), value['runtime'])
    verify_producer(Path(value['stage2']) / 'UR001-manifest.json', Path(value['stage2_run']), value['producer_pin'])
    verify_dag(Path(value['stage2']), require_profile='subset')
    config = helper.frozen_config(Path(value['native_root']), value['config_file']['path'], value['config_file']['sha256'])
    if backend.encoded(config) != backend.encoded(value['config']):
        raise ValueError('subset frozen config differs')
    if type(value['disk_cap_bytes']) is not int or not 0 < value['disk_cap_bytes'] <= 64 * 1024**3:
        raise ValueError('original reference disk cap required')
    return backend.native_for(Path(value['native_root']))


def execute(stage, inputs, request, digest):
    value, roles = read_transport(inputs, request, digest)
    from analysis.producer_dag_binding import verify_producer
    records = stage_inputs(value, roles, inputs, request, digest, stage)
    job_sha = check_job(value, records, request, digest, stage)
    native = context(value)
    output = Path(value['output_root'])
    guard = native.b1.resource_guard(output, value['disk_cap_bytes']); guard()
    if stage == 'preflight':
        # No model/optimizer/train_step; same module, context and raw binding.
        result = {'verified': True, 'read_only': True, 'engineering': False, 'payload_written': False}
    elif stage == 'rebuild':
        identity = native.b1.source_identity()
        binding = {'consumer': {'source_fingerprint': identity['source_sha']}, 'original_source': identity,
                    'subset_producer': value['producer_pin'], 'request_sha256': digest}
        result = rebuild_reference(native, roles, Path(roles['metadata'][1]), Path(roles['metadata'][2]), output, binding, guard)
        result.update({'schema': 'p-subset-reference-manifest-v1', 'source': value['source'], 'runtime': value['runtime'],
                       'request_sha256': digest, 'own_job_sha256': job_sha,
                       'subset_producer': value['producer_pin'], 'config_sha256': value['config_file']['sha256']})
    else:
        ref = value['reference']
        proof = verify_producer(Path(ref['manifest']), Path(ref['run_dir']), ref['pin'])
        if (proof['manifest'].get('schema') != 'p-subset-reference-manifest-v1'
                or proof['manifest']['original_source_fingerprint'] != native.b1.source_identity()['source_sha']
                or backend.encoded(proof['manifest'].get('subset_producer')) != backend.encoded(value['producer_pin'])
                or proof['manifest'].get('config_sha256') != value['config_file']['sha256']):
            raise ValueError('original reference source identity differs')
        reference, _ = load_reference(native, output, proof['manifest'])
        count = len(reference['rows'])
        if count < value['config']['batch_size'] * 3 or len(reference['queries']) < 32:
            raise ValueError('fixed subset comparison prefix unavailable')
        # Keep the complete subset train population: native EpochCursor and
        # target_frequency must see the same rows as the original subset run.
        reference = {**reference, 'queries': reference['queries'][:32]}
        adapted = helper.reference_assembly(native, {**value,
                'items_metadata': {'path': roles['metadata'][1], 'sha256': backend.file_sha(roles['metadata'][1])},
                'embedding': {'path': roles['metadata'][2], 'sha256': backend.file_sha(roles['metadata'][2])}},
                inputs, count, 32, guard)
        # UR001 adapter is a distinct path from original prepared+history+side.
        parts = partitions(roles, guard)
        side = original(native, 'history_side')
        adapted = backend.adapt_rows(native, [{k: v for k, v in r.items() if k != 'side'} for r in adapted['rows']],
                [{k: v for k, v in r.items() if k != 'side'} for r in adapted['queries']], adapted['catalog'],
                sorted(adapted['support']), parts, side.load_durations(Path(roles['metadata'][1]), adapted['catalog']),
                adapted['content'].numpy())
        if canonical_identity(reference) != canonical_identity(adapted):
            raise ValueError('original loader and UR001 adapter assembly differ')
        native.b1.execution_context('cuda')
        first = helper.capture(native, reference, value['config'], 'cuda', guard)
        second = helper.capture(native, adapted, value['config'], 'cuda', guard)
        if first != second:
            raise ValueError('original loader and adapter steps/evaluation differ')
        result = {'verified': True, 'bitwise_equal': True, 'steps': 3, 'queries': 32, 'training_rows': count,
                  'original_parameterized_loader_verified': True, 'original_legacy_wrapper_used': False,
                  'comparison_sha256': __import__('hashlib').sha256(backend.encoded(first)).hexdigest(),
                  'old_pins': proof['manifest']['old_pins']}
        verify_producer(Path(ref['manifest']), Path(ref['run_dir']), ref['pin'])
    context(value)
    if check_job(value, records, request, digest, stage) != job_sha:
        raise ValueError('subset own job changed during execution')
    if stage == 'rebuild':
        helper.write_new(output / 'reference-manifest.json', result)
        result = {'verified': True, 'manifest_sha256': backend.file_sha(output / 'reference-manifest.json')}
    return {**result, 'stage': stage, 'training_submitted': False, 'checkpoint_written': False,
            'request_sha256': digest, 'own_job_sha256': job_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('dry')
    for command in ('plan', *STAGES):
        p = sub.add_parser(command)
        if command == 'plan':
            p.add_argument('--stage', choices=STAGES, required=True)
        p.add_argument('--inputs', required=True); p.add_argument('--request', required=True)
        p.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'dry':
            result = {'status': 'dry', 'training_submitted': False, 'checkpoint_written': False}
        elif args.command == 'plan':
            result = plan(args.stage, args.inputs, args.request, args.request_sha256)
        else:
            result = execute(args.command, args.inputs, args.request, args.request_sha256)
        print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({'status': 'ALERT', 'error': str(exc)}, sort_keys=True)); return 2


if __name__ == '__main__':
    raise SystemExit(main())
