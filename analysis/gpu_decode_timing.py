"""Fixed-source adapter of the original 512-query/1,2,4,8-worker timing method.

No scientific training, holdout or arbitrary checkpoint loader. Formal stages
require a SHA-pinned request, successful original synthetic producer and OPS job.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SHA256 = '1c5b9ce35d685dea416e67b0c89936ffbd67f66e8fae32f162a9e2f1c2a4065f'
CSR_SHA256 = 'f07b563fae327a372662741dd898538aaea53d6203a326025c08bd4f9f9ad123'
SCHEMA = 'gpu-decode-timing-request-v1'
RUN_IDS = {'preflight': 'g-gpu-decode-preflight-v1', 'run': 'g-gpu-decode-timing-v1'}


def digest(path):
    from analysis.g_decode_preflight_retry import digest as original
    return original(absolute(path))


def absolute(path):
    from analysis.g_decode_preflight_retry import absolute as original
    p = original(path)
    if any('holdout' in x.lower() or x.lower() in ('test', 'week_26', 'week26') for x in p.parts):
        raise ValueError('test/holdout path forbidden')
    return p


def engine():
    from analysis import g_decode_timing as original
    if Path(original.__file__).resolve() != ROOT / 'analysis/g_decode_timing.py':
        raise ValueError('original timing import is outside deployment tree')
    if digest(original.__file__) != ENGINE_SHA256 or digest(ROOT / 'src/vkgr/gpu_trie_decode.py') != CSR_SHA256:
        raise ValueError('frozen timing/CSR source differs')
    original.references()
    return original


def check_run(run, expected):
    required = {'run_id', 'kind', 'git', 'cwd', 'argv', 'config', 'inputs'}
    if set(expected) != required or expected['git'].get('dirty') is not False:
        raise ValueError('explicit complete producer expectation required')
    if any(run.get(k) != expected[k] for k in required - {'inputs'}):
        raise ValueError('producer source/argv/config binding differs')
    def pins(records):
        result = []
        for record in records:
            if not {'path', 'sha256'} <= set(record) <= {'path', 'sha256', 'bytes'}:
                raise ValueError('fixed input metadata fields required')
            if 'bytes' in record and (record['bytes'] != absolute(record['path']).stat().st_size or digest(record['path']) != record['sha256']):
                raise ValueError('launcher input size/hash metadata differs')
            result.append({'path': record['path'], 'sha256': record['sha256']})
        if len({x['path'] for x in result}) != len(result):
            raise ValueError('duplicate producer inputs')
        return result
    if pins(run.get('inputs', [])) != pins(expected['inputs']):
        raise ValueError('producer input binding differs')


def accepted_producer(base, expected, payload):
    """Exact metadata, input-set, successful terminal and original stdout binding."""
    base = absolute(base)
    run = json.loads(absolute(base / 'run.json').read_text())
    term = json.loads(absolute(base / 'terminal.json').read_text())
    check_run(run, expected)
    if term.get('run_id') != expected['run_id'] or term.get('status') != 'succeeded' or term.get('exit_code') != 0 or term.get('signal') is not None:
        raise ValueError('successful original producer terminal required')
    raw = absolute(base / 'logs/stdout.log').read_bytes()
    if json.loads(raw) != payload:
        raise ValueError('producer stdout/artifact differs')
    return hashlib.sha256(raw).hexdigest()


def verify_query(left, a, right, b):
    from analysis.gpu_trie_decode_probe import trace_digest
    if len(a) != 4 or len(b) != 4 or trace_digest(a) != trace_digest(b) or left != right:
        raise ValueError('ALERT per-query original/CSR beam bytes differ')
    return trace_digest(a)


def merge_parity_peak(pre, parity):
    original = engine()
    for record in (pre, parity):
        original.require_gpu_peak(record['per_worker_peak_mib'], record['vram_samples'])
    peak = max(pre['per_worker_peak_mib'], parity['per_worker_peak_mib'])
    if peak > 8192:
        raise MemoryError('first conservative VRAM declaration exceeded')
    pre['per_worker_peak_mib'] = peak
    pre['vram_samples'] += parity['vram_samples']


def real_subset(request):
    # Old checkpoint is gone; a new authorized subset producer has not yet
    # emitted an accepted bundle. Do not invent a native-loader/data contract.
    raise ValueError('no authorized real subset bundle is available; required: authorization, native source ledger plus pinned delta, runtime/lock, successful producer job/config/argv/input pins, training manifest, checkpoint hash/binding, catalog/SID/history/validation pins and engineering=False native loader')


def read_request(path, pin, *, formal=True):
    path = absolute(path)
    if digest(path) != pin:
        raise ValueError('request SHA differs')
    value = json.loads(path.read_text())
    fields = {'schema', 'source_commit', 'lock', 'input_root', 'input_manifest_sha256', 'generation', 'ops_root', 'output_root'}
    if type(value) is not dict or set(value) != fields or value['schema'] != SCHEMA:
        raise ValueError('fixed synthetic request fields required')
    for name in ('input_root', 'ops_root', 'output_root'):
        absolute(value[name])
    out = Path(value['output_root']); inp = Path(value['input_root'])
    if out == inp or inp in out.parents or out in inp.parents or out == ROOT or ROOT in out.parents or out in ROOT.parents:
        raise ValueError('fresh output separate from source and original synthetic root required')
    if type(value['lock']) is not dict or set(value['lock']) != {'path', 'sha256'}:
        raise ValueError('fixed lock pin fields required')
    lock = absolute(value['lock']['path'])
    if digest(lock) != value['lock']['sha256']:
        raise ValueError('lock pin differs')
    if formal:
        from analysis.g_decode_preflight_retry import source_ledger
        source_ledger(ROOT, value['source_commit'])
        engine().environment(lock)
    return value


def invocation(stage, request, pin):
    return [sys.executable, '-B', '-m', 'analysis.gpu_decode_timing', stage,
            '--request', str(request), '--request-sha256', pin, '--execute-on-new-instance']


def stage_inputs(value, request, pin, stage):
    original = engine()
    pins = [{'path': str(request), 'sha256': pin}, value['lock'],
            {'path': str(original.PINS), 'sha256': original.PINS_SHA},
            {'path': str(Path(value['input_root']) / 'inputs.json'), 'sha256': value['input_manifest_sha256']}]
    if stage == 'run':
        p = Path(value['output_root']) / 'preflight.json'
        pins.append({'path': str(p), 'sha256': digest(p)})
    return pins


def expected_job(value, request, pin, stage):
    from analysis.a40_cost import YAML, YAML_SHA
    return {'run_id': RUN_IDS[stage], 'kind': 'ops',
            'git': {'commit': value['source_commit'], 'dirty': False, 'dirty_files': []},
            'cwd': str(ROOT), 'argv': invocation(stage, request, pin),
            'config': {'path': str(YAML), 'sha256': YAML_SHA},
            'inputs': stage_inputs(value, request, pin, stage)}


def synthetic_inputs(value, request, pin):
    import numpy as np
    original = engine()
    root = Path(value['input_root']); path = root / 'inputs.json'
    if digest(path) != value['input_manifest_sha256']:
        raise ValueError('original input manifest pin differs')
    doc = json.loads(path.read_text())
    fixed = {'schema': 'UR001_SYNTHETIC_G_DECODE_TIMING_V1', 'status': 'synthetic_generated',
             'items': original.N, 'branching': original.K, 'depth': 3, 'queries': 512, 'history': 50, 'seed': 42,
             'source': value['generation']['git']['commit'], 'script_sha256': ENGINE_SHA256,
             'source_pins_sha256': original.PINS_SHA, 'lock_sha256': value['lock']['sha256'],
             'synthetic_input': True, 'science_training': False, 'science_data_read': False, 'engineering': False}
    if any(doc.get(k) != v or type(doc.get(k)) is not type(v) for k, v in fixed.items()):
        raise ValueError('original synthetic contract differs')
    generation = value['generation']
    expected_pins = [{'path': value['lock']['path'], 'sha256': value['lock']['sha256']},
                     {'path': str(Path(generation['cwd']) / 'configs/g-decode-timing/source-pins.json'), 'sha256': original.PINS_SHA}]
    argv = [sys.executable, '-B', '-m', 'analysis.g_decode_timing', 'generate', '--root', str(root),
            '--lock', value['lock']['path'], '--ops-root', value['ops_root'], '--execute-on-new-instance']
    if generation['run_id'] != 'g-decode-timing-generate-v1' or generation['argv'] != argv or generation['inputs'] != expected_pins:
        raise ValueError('fixed original generator closure differs')
    from analysis.a40_cost import YAML, YAML_SHA
    if generation['config'] != {'path': str(Path(generation['cwd']) / YAML.relative_to(ROOT)), 'sha256': YAML_SHA}:
        raise ValueError('original generator config differs')
    producer = accepted_producer(Path(value['ops_root']) / 'runs' / generation['run_id'], generation, doc)
    names = ('ids.npy', 'sids.npy', 'queries.npy')
    if [r['name'] for r in doc['files']] != list(names):
        raise ValueError('synthetic file set differs')
    for record in doc['files']:
        if digest(root / record['name']) != record['sha256']:
            raise ValueError('synthetic array bytes differ')
    arrays = [np.load(root / name, mmap_mode='r', allow_pickle=False) for name in names]
    for array, shape in zip(arrays, [(original.N,), (original.N, 4), (512, 50)]):
        if array.dtype != np.dtype('u4') or array.shape != shape:
            raise ValueError('synthetic array schema differs')
    if np.any(arrays[2] < 1) or np.any(arrays[2] > original.N):
        raise ValueError('synthetic history item outside catalog')
    env = original.environment(Path(value['lock']['path']))
    env.update(schema='GPU_CSR_SYNTHETIC_G_DECODE_TIMING_V1', request_sha256=pin,
               adapter_sha256=digest(__file__), session_source_sha256=digest(ROOT / 'src/vkgr/gpu_decode_session.py'))
    return env, [{'history': [int(x) for x in row]} for row in arrays[2]], producer


def timing_function(inputs, predecessor, write_new):
    """Same original code object; dependencies enumerated, no global monkeypatch."""
    original = engine()
    from vkgr.gpu_decode_session import SessionFactory
    namespace = {**vars(original), 'inputs': inputs, 'predecessor': predecessor,
                 'write_new': write_new, 'SessionFactory': SessionFactory}
    return FunctionType(original.run.__code__, namespace, original.run.__name__)


def execute(stage, request, pin):
    original = engine(); value = read_request(request, pin)
    expected = expected_job(value, request, pin, stage)
    run_dir = Path(value['ops_root']) / 'runs' / expected['run_id']
    if os.environ.get('VKGR_RUN_ID') != expected['run_id']:
        raise ValueError('current OPS job identity required')
    job = json.loads((run_dir / 'run.json').read_text())
    check_run(job, expected)
    out = Path(value['output_root']); lock = Path(value['lock']['path']); ops = Path(value['ops_root'])
    env, queries, producer = synthetic_inputs(value, request, pin)
    from vkgr.gpu_decode_session import SessionFactory
    from analysis.gpu_trie_decode_probe import original_decode, trace_digest
    from vkgr.gpu_trie_decode import decode_history_csr
    from vkgr.sid_depth import SIDTokensLD
    factory = lambda ignored: SessionFactory(value['input_root'])
    def inputs(*ignored):
        return env, queries, producer
    def predecessor(ignored_stage, ignored_root, ignored_lock, ignored_ops, payload):
        if ignored_stage != 'preflight':
            raise ValueError('only exact new preflight predecessor accepted')
        return accepted_producer(ops / 'runs' / RUN_IDS['preflight'], expected_job(value, request, pin, 'preflight'), payload)
    def write(path, payload):
        payload['decoder'] = 'gpu_layer_csr_exact_batch1'
        payload['trace_manifest_sha256'] = digest(out / 'beam-parity.json')
        if path.name == 'preflight.json':
            parity = json.loads((out / 'beam-parity.json').read_text())
            merge_parity_peak(payload, parity)
            if payload['result_sha256'] != parity['result_sha256']:
                raise ValueError('ALERT timed preflight differs from trace baseline')
        if 'limitations' in payload:
            payload['limitations'] = [x for x in payload['limitations'] if not x.startswith('trie is the original')]
            payload['limitations'].append('CSR identity hashing copies tensors outside query timer; wall includes these guards')
        original.write_new(path, payload)
    if stage == 'preflight':
        if out.exists():
            raise FileExistsError('fresh GPU preflight directory required')
        out.mkdir()
        with (out / 'inputs.json').open('xb') as stream:
            stream.write((Path(value['input_root']) / 'inputs.json').read_bytes())
        session = factory(None)(); tokens = SIDTokensLD(session.codec, 50)
        from analysis.a40_cost import selected
        from vkgr.throughput_config import guard
        config = selected()
        rows = []; baseline = []
        for ordinal, query in enumerate(queries):
            guard(config, out)
            a = []; b = []
            left = original_decode(session.model, tokens, query['history'], trace=a)
            right = decode_history_csr(session.model, tokens, query['history'], trie=session.trie, trace=b)
            verify_query(left, a, right, b)
            rows.append({'ordinal': ordinal, 'trace_sha256': trace_digest(a)})
            baseline.append({'result': {'ranked': left, 'trace_sha256': trace_digest(a)}})
            print(json.dumps({'stage': 'beam_parity', 'completed': ordinal + 1, 'total': 512}), file=sys.stderr, flush=True)
        parity_peak = original._SAMPLER.sample()
        original.require_gpu_peak(parity_peak, original._SAMPLER.samples)
        if parity_peak > 8192:
            raise MemoryError('first conservative VRAM declaration exceeded')
        original.write_new(out / 'beam-parity.json', {'queries': rows, 'result_sha256': original.verify_results(baseline, baseline),
                            'inputs_manifest_sha256': value['input_manifest_sha256'], 'request_sha256': pin,
                            'source': value['source_commit'], 'synthetic': True, 'real_subset': False,
                            'per_worker_peak_mib': parity_peak, 'vram_samples': original._SAMPLER.samples})
        del session, tokens
        namespace = {**vars(original), 'inputs': inputs, 'write_new': write, 'SessionFactory': factory}
        result = FunctionType(original.preflight.__code__, namespace)(out, lock, ops)
        parity = json.loads((out / 'beam-parity.json').read_text())
        if result['result_sha256'] != parity['result_sha256']:
            raise ValueError('ALERT timed preflight differs from trace baseline')
        return result
    pre = json.loads((out / 'preflight.json').read_text())
    if pre.get('trace_manifest_sha256') != digest(out / 'beam-parity.json'):
        raise ValueError('preflight beam-parity binding differs')
    run = timing_function(inputs, predecessor, write)
    # Replace only the root bound into the factory, keeping original function code.
    run.__globals__['SessionFactory'] = factory
    return run(out, lock, ops)


def prepare(stage, request, pin, ops_code):
    from analysis.a40_ops import load_ops
    from analysis.a40_cost import YAML
    _, launcher, _, _ = load_ops(ops_code)
    value = read_request(request, pin, formal=False)
    from analysis.g_decode_preflight_retry import source_ledger
    source_ledger(ROOT, value['source_commit'])
    root = Path(value['output_root']); ops = Path(value['ops_root'])
    if (ops / 'runs' / RUN_IDS[stage]).exists():
        raise FileExistsError('run already exists')
    for state in ('pending', 'claimed', 'rejected'):
        if list((ops / 'queue' / state).glob('*-' + RUN_IDS[stage] + '.json')):
            raise FileExistsError('run already queued')
    vram = 8192
    if stage == 'run':
        pre = json.loads((root / 'preflight.json').read_text())
        accepted_producer(ops / 'runs' / RUN_IDS['preflight'], expected_job(value, request, pin, 'preflight'), pre)
        engine().require_gpu_peak(pre['per_worker_peak_mib'], pre['vram_samples'])
        vram = math.ceil(9 * pre['per_worker_peak_mib'] * 1.15 / 256) * 256
    raw = ['--root', str(ops), '--run-id', RUN_IDS[stage], '--kind', 'ops', '--cwd', str(ROOT), '--config', str(YAML),
           '--seed', '42', '--env', 'PYTHONPATH=' + str(ROOT / 'src'), '--env', 'OMP_NUM_THREADS=1',
           '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'MKL_NUM_THREADS=1', '--env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
           '--cpu-cores', '9', '--ram-mib', '98304', '--vram-mib', str(vram), '--disk-gib', '2', '--wall-hours', '8' if stage == 'preflight' else '48']
    for record in stage_inputs(value, request, pin, stage):
        raw += ['--input', record['path']]
    spec = launcher.spec_from_args(launcher.build_parser().parse_args([*raw, '--', *invocation(stage, request, pin)]))
    metadata = launcher.prepare(spec)
    check_run({**metadata, **{k: spec[k] for k in ('run_id', 'kind', 'cwd', 'argv')}}, expected_job(value, request, pin, stage))
    return {'status': 'prepared_dry', 'spec': spec, 'metadata': metadata,
            'vram_declaration_scope': 'not_measured conservative first limit' if stage == 'preflight' else 'measured worker peak x9 x1.15 ceil256', 'no_submit': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('dry')
    real = sub.add_parser('real-subset')
    real.add_argument('--request', type=Path, required=True)
    for command in ('preflight', 'run', 'prepare'):
        item = sub.add_parser(command)
        item.add_argument('--request', type=Path, required=True)
        item.add_argument('--request-sha256', required=True)
        if command == 'prepare':
            item.add_argument('--stage', choices=('preflight', 'run'), required=True)
            item.add_argument('--ops-code', type=Path, required=True)
        else:
            item.add_argument('--execute-on-new-instance', action='store_true', required=True)
    args = parser.parse_args(argv)
    if args.command == 'dry':
        result = {'status': 'dry', 'queries': 512, 'workers': [1, 2, 4, 8], 'beam': 100, 'steps': 4, 'batch': 1,
                  'science_training': False, 'timing_code_sha256': ENGINE_SHA256,
                  'real_subset': 'FAIL_CLOSED: no available authorized bundle'}
    elif args.command == 'real-subset':
        result = real_subset(args.request)
    elif args.command == 'prepare':
        result = prepare(args.stage, args.request, args.request_sha256, args.ops_code)
    else:
        result = execute(args.command, args.request, args.request_sha256)
    print(json.dumps(result, sort_keys=True))
    original = engine()
    if original._SAMPLER is not None:
        original._SAMPLER.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
