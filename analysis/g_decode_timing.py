"""Bounded synthetic A1 decode timing; no training, dataset or checkpoint reads."""
import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import threading
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / 'configs/g-decode-timing/source-pins.json'
PINS_SHA = 'd4cea5a7cf732581a2cd1b0ee1792cea90521fe7ac9771f191466e9b8b9ad0df'
N, K, DEPTH, QUERIES, HISTORY, SEED = 12_012_895, 229, 3, 512, 50, 42
WORKERS = (1, 2, 4, 8)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def references():
    if digest(PINS) != PINS_SHA:
        raise ValueError('reference ledger differs')
    pins = json.loads(PINS.read_text())
    for name, expected in pins['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('original source differs: ' + name)
    return pins


def fresh_path(path):
    from analysis.a40_cost import safe
    path = safe(path)
    if any('holdout' in p.lower() or p.lower() == 'test' or 'week_26' in p.lower() for p in path.parts):
        raise ValueError('real evaluation data forbidden')
    return path


def write_new(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, sort_keys=True)


def balanced_sids(n, k, depth):
    """Assign increasing synthetic IDs by the original recursive capacity rule."""
    import numpy as np
    if type(n) is not int or n < 1 or type(k) is not int or k < 2 or depth != 3:
        raise ValueError('positive n, branching >=2 and depth3 required')
    out = np.zeros((n, depth + 1), dtype=np.uint32)

    def visit(first, count, level):
        children = min(k, count)
        sizes = np.full(children, count // children, dtype=np.int64)
        sizes[:count % children] += 1
        digits = np.repeat(np.arange(children, dtype=np.uint32), sizes)
        out[first:first + count, level] = digits
        if level == depth - 1:
            starts = np.r_[0, np.cumsum(sizes[:-1])]
            out[first:first + count, depth] = np.arange(count) - np.repeat(starts, sizes)
        else:
            cursor = first
            for size in sizes:
                visit(cursor, int(size), level + 1)
                cursor += int(size)
    visit(0, n, 0)
    return out


def verify_results(original, candidate):
    if len(original) != len(candidate):
        raise ValueError('query count differs')
    hashes = []
    for a, b in zip(original, candidate):
        left = json.dumps(a['result'], sort_keys=True, separators=(',', ':')).encode()
        right = json.dumps(b['result'], sort_keys=True, separators=(',', ':')).encode()
        if left != right:
            raise ValueError('per-query ranked result/audit differs')
        hashes.append(hashlib.sha256(left).hexdigest())
    return hashlib.sha256(''.join(hashes).encode()).hexdigest()


def process_vram():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=10)
    for line in raw.splitlines():
        pid, value = [x.strip() for x in line.split(',')]
        if int(pid) == os.getpid():
            return int(value)
    raise ValueError('own context-inclusive GPU memory unavailable')


def require_gpu_peak(peak, samples):
    if (type(peak) not in (int, float) or not math.isfinite(peak) or peak <= 0 or
            type(samples) is not int or samples <= 0):
        raise ValueError('positive measured context-inclusive GPU peak and samples required')


class ContextSampler:
    """Sample only this process, including its CUDA context, for its lifetime."""
    def __init__(self):
        self.peak = 0
        self.samples = 0
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def sample(self):
        value = process_vram()
        with self.lock:
            self.peak = max(self.peak, value)
            self.samples += 1
        return self.peak

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except (ValueError, OSError, subprocess.SubprocessError):
                pass  # CUDA may not yet have created this process's context.
            self.stop.wait(.1)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=11)


_SAMPLER = None


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def timed_decode(model, codec, history, *, beam):
    import torch
    from vkgr.sid_depth import SIDTokensLD
    from vkgr.sid_depth_beam import decode_history_d
    tokens = SIDTokensLD(codec, HISTORY)
    cuda = next(model.parameters()).device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize()
    started = time.perf_counter()
    value = decode_history_d(model, tokens, history, beam=beam)
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {'result': value, 'seconds': elapsed,
            'vram_mib': _SAMPLER.sample() if cuda else 0,
            'vram_samples': _SAMPLER.samples if cuda else 0,
            'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved() if cuda else 0}


@dataclass(frozen=True)
class SessionFactory:
    root: str
    device: str = 'cuda'

    def __call__(self):
        import numpy as np
        import torch
        from analysis import g_decode_timing as canonical
        from vkgr.sid import SIDCatalog
        from vkgr.sid_depth import ValidatedCodecD
        from vkgr.generative_arch import make_model_arch
        from vkgr.throughput_decode import TorchSession
        pins = references()
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        ids = np.load(Path(self.root) / 'ids.npy', mmap_mode='r', allow_pickle=False)
        sids = np.load(Path(self.root) / 'sids.npy', mmap_mode='r', allow_pickle=False)
        from vkgr.throughput_config import rss_bytes
        rss_before = rss_bytes(); started = time.perf_counter()
        catalog = SIDCatalog(ids, sids, K, DEPTH, [])
        trie_measurement = {'build_seconds': time.perf_counter() - started,
                            'rss_before_bytes': rss_before, 'rss_after_bytes': rss_bytes(),
                            'process_peak_rss_after_bytes': peak_rss(),
                            'index_storage_bytes': int(catalog.keys.nbytes + catalog.key_items.nbytes),
                            'index': 'original SIDCatalog sorted-key prefix index'}
        codec = ValidatedCodecD(catalog, DEPTH)
        if self.device.startswith('cuda'):
            if canonical._SAMPLER is not None:
                canonical._SAMPLER.close()
            canonical._SAMPLER = canonical.ContextSampler()
        model = make_model_arch(codec, pins['a1_config'], device=self.device)
        session = TorchSession(model, codec, canonical.timed_decode, beam=100)
        session.trie_measurement = trie_measurement
        return session


def invocation(stage, root, lock, ops_root):
    return [sys.executable, '-B', '-m', 'analysis.g_decode_timing', stage,
            '--root', str(root), '--lock', str(lock), '--ops-root', str(ops_root),
            '--execute-on-new-instance']


def environment(lock):
    from analysis.a40_cost import runtime, source_sha, selected
    runtime(lock); references()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('deterministic CUDA workspace required')
    if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=all'], text=True).strip():
        raise ValueError('clean fixed deployment required')
    selected()
    return {'schema': 'UR001_SYNTHETIC_G_DECODE_TIMING_V1', 'source': source_sha(), 'lock_sha256': digest(lock), 'source_pins_sha256': PINS_SHA,
            'script_sha256': digest(__file__), 'engineering': False,
            'synthetic_input': True, 'science_training': False, 'science_data_read': False}


def predecessor(stage, root, lock, ops_root, payload):
    from analysis.a40_cost import source_sha, YAML, YAML_SHA
    run_id = 'g-decode-timing-' + stage + '-v1'
    base = fresh_path(ops_root) / 'runs' / run_id
    run = json.loads((base / 'run.json').read_text())
    terminal = json.loads((base / 'terminal.json').read_text())
    raw = (base / 'logs/stdout.log').read_bytes()
    if (terminal.get('status') != 'succeeded' or terminal.get('exit_code') != 0 or
            terminal.get('run_id') != run_id or run.get('run_id') != run_id or
            run.get('kind') != 'ops' or run.get('git', {}).get('commit') != source_sha() or
            run.get('git', {}).get('dirty') is not False or run.get('cwd') != str(ROOT) or
            run.get('argv') != invocation(stage, root, lock, ops_root) or
            run.get('config') != {'path': str(YAML), 'sha256': YAML_SHA} or
            json.loads(raw) != payload):
        raise ValueError('original producer/terminal/stdout binding differs')
    expected = {str(lock): digest(lock), str(PINS): PINS_SHA}
    if stage == 'preflight':
        expected[str(root / 'inputs.json')] = digest(root / 'inputs.json')
    if {p['path']: p['sha256'] for p in run['inputs']} != expected:
        raise ValueError('producer input binding differs')
    return hashlib.sha256(raw).hexdigest()


def generate(root, lock, ops_root):
    import numpy as np
    from vkgr.throughput_config import guard
    from analysis.a40_cost import selected
    env = environment(lock); root = fresh_path(root)
    if root.exists():
        raise FileExistsError('new synthetic directory required')
    guard(selected(), root.parent, projected_memory=N * 48, projected_disk=N * 24)
    root.mkdir()
    rng = np.random.default_rng(SEED)
    for name, value in [('ids.npy', np.arange(1, N + 1, dtype=np.uint32)),
                        ('sids.npy', balanced_sids(N, K, DEPTH)),
                        ('queries.npy', rng.integers(1, N + 1, size=(QUERIES, HISTORY), dtype=np.uint32))]:
        with (root / name).open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
    value = {**env, 'status': 'synthetic_generated', 'items': N, 'branching': K, 'depth': DEPTH,
             'queries': QUERIES, 'history': HISTORY, 'seed': SEED,
             'files': [{'name': name, 'sha256': digest(root / name)}
                       for name in ('ids.npy', 'sids.npy', 'queries.npy')]}
    write_new(root / 'inputs.json', value)
    return value


def inputs(root, lock, ops_root):
    import numpy as np
    env = environment(lock); root = fresh_path(root)
    doc = json.loads((root / 'inputs.json').read_text())
    if any(doc.get(k) != v for k, v in env.items()) or (doc.get('items'), doc.get('branching'), doc.get('queries'), doc.get('history')) != (N, K, QUERIES, HISTORY):
        raise ValueError('fixed synthetic input identity differs')
    if [p['name'] for p in doc['files']] != ['ids.npy', 'sids.npy', 'queries.npy']:
        raise ValueError('synthetic file list differs')
    for record in doc['files']:
        if digest(fresh_path(root / record['name'])) != record['sha256']:
            raise ValueError('synthetic bytes differ')
    producer = predecessor('generate', root, lock, ops_root, doc)
    queries = np.load(root / 'queries.npy', mmap_mode='r', allow_pickle=False)
    ids = np.load(root / 'ids.npy', mmap_mode='r', allow_pickle=False)
    sids = np.load(root / 'sids.npy', mmap_mode='r', allow_pickle=False)
    if (ids.shape != (N,) or ids.dtype != np.dtype('u4') or sids.shape != (N, 4) or
            sids.dtype != np.dtype('u4') or queries.shape != (QUERIES, HISTORY) or queries.dtype != np.dtype('u4')):
        raise ValueError('synthetic array schema differs')
    return env, [{'history': [int(v) for v in row]} for row in queries], producer


def preflight(root, lock, ops_root):
    from vkgr.throughput_config import rss_bytes
    env, queries, producer = inputs(root, lock, ops_root)
    start = time.perf_counter(); before = rss_bytes()
    session = SessionFactory(str(root))()
    build_seconds = time.perf_counter() - start
    trie_rss = rss_bytes()
    results = [session.decode(query) for query in queries]
    measured_peak = max(result['vram_mib'] for result in results)
    require_gpu_peak(measured_peak, results[-1]['vram_samples'])
    if measured_peak > 8192:
        raise MemoryError('first conservative VRAM declaration exceeded')
    value = {**env, 'status': 'ready', 'inputs_manifest_sha256': digest(root / 'inputs.json'),
             'generation_stdout_sha256': producer, 'session_sha256': session.identity(),
             'trie_codec_model_build_seconds': build_seconds, 'trie_measurement': session.trie_measurement, 'rss_before_bytes': before,
             'rss_after_build_bytes': trie_rss, 'peak_scope': 'own process 0.1s background plus end-of-query samples including CUDA context',
             'per_worker_peak_mib': measured_peak, 'vram_samples': results[-1]['vram_samples'],
             'torch_peak_reserved_bytes': max(result['torch_peak_reserved_bytes'] for result in results),
             'result_sha256': verify_results(results, results), 'measured_queries': QUERIES, 'no_recall_metrics': True}
    write_new(root / 'preflight.json', value)
    return value


def run(root, lock, ops_root):
    from analysis.a40_cost import selected
    from vkgr.throughput_config import guard, rss_bytes
    from vkgr.throughput_decode import decode_queries, InitializerBudget, VRAMReceipt
    env, queries, producer = inputs(root, lock, ops_root)
    pre = json.loads((root / 'preflight.json').read_text())
    pre_stdout = predecessor('preflight', root, lock, ops_root, pre)
    if pre.get('status') != 'ready' or pre.get('generation_stdout_sha256') != producer or pre.get('inputs_manifest_sha256') != digest(root / 'inputs.json') or pre.get('measured_queries') != QUERIES:
        raise ValueError('preflight input binding differs')
    if (root / 'timing.json').exists():
        raise FileExistsError('timing receipt exists')
    config = selected(); factory = SessionFactory(str(root)); session = factory()
    if session.identity() != pre['session_sha256']:
        raise ValueError('original model/codec identity differs')
    budget = InitializerBudget.for_session(session, estimated_init_bytes=max(session.storage_bytes(), 4 * 1024**3),
                                            budget_bytes_per_worker=8 * 1024**3)
    measured = []; baseline = None; peak = pre['per_worker_peak_mib']
    for workers in WORKERS:
        guard(config, root, projected_memory=workers * budget.budget_bytes_per_worker)
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10)
        free = int(raw.strip())
        start = time.perf_counter()
        values = decode_queries(queries, session.decode, replace(config, decode_workers=workers),
                                session_factory=factory, state_sha=session.identity(), device='cuda',
                                vram_receipt=VRAMReceipt(peak, free, True, True), initializer_budget=budget)
        wall = time.perf_counter() - start
        if baseline is None:
            baseline = values
        result_sha = verify_results(baseline, values)
        if result_sha != pre['result_sha256']:
            raise ValueError('timing results differ from original preflight baseline')
        peak = max(peak, max(v['vram_mib'] for v in values))
        seconds = [v['seconds'] for v in values]
        measured.append({'workers': workers, 'query_seconds': seconds, 'wall_seconds_including_guards_init_ipc': wall,
                         'mean_query_seconds': sum(seconds) / len(seconds),
                         'epoch_cost_formula_seconds': 26827 * sum(seconds) / len(seconds) / workers,
                         'context_inclusive_per_worker_peak_mib': max(v['vram_mib'] for v in values),
                         'process_tree_rss_after_bytes': rss_bytes(include_children=True),
                         'context_inclusive_samples': sum(v['vram_samples'] for v in values),
                         'result_sha256': result_sha, 'bitwise_equal_to_workers1': True})
    value = {**env, 'status': 'synthetic_decode_timing_verified', 'no_recall_metrics': True,
             'items': N, 'branching': K, 'depth': DEPTH, 'queries': QUERIES, 'history': HISTORY,
             'batch': 1, 'beam': 100, 'seed': SEED, 'workers': measured,
             'preflight_stdout_sha256': pre_stdout, 'inputs_manifest_sha256': digest(root / 'inputs.json'),
             'preflight': pre, 'vram_peak_scope': 'own process 0.1s background plus end-of-query sampled memory includes CUDA context; transient unsampled peaks UNKNOWN',
             'limitations': ['synthetic histories differ from real distribution',
                             'trie is the original sorted-key prefix index and original lazy allowed-next cache',
                             'parallel wall includes state guards, initialization and IPC; query time excludes those',
                             'RSS uses process high-water marks and post-build snapshots; concurrent tree peak is separately guarded']}
    write_new(root / 'timing.json', value)
    return value


def job(args):
    from analysis.a40_ops import load_ops
    from analysis.a40_cost import YAML, YAML_SHA
    _, launcher, scheduler, _ = load_ops(args.ops_code)
    root = fresh_path(args.root); ops_root = fresh_path(args.ops_root)
    stage = args.stage; run_id = 'g-decode-timing-' + stage + '-v1'
    vram = 0 if stage == 'generate' else 8192
    if stage == 'run':
        pre = json.loads((root / 'preflight.json').read_text())
        predecessor('preflight', root, args.lock, ops_root, pre)
        vram = math.ceil(9 * pre['per_worker_peak_mib'] * 1.15 / 256) * 256
    raw = ['--root', str(ops_root), '--run-id', run_id, '--kind', 'ops', '--cwd', str(ROOT),
           '--config', str(YAML), '--input', str(args.lock), '--input', str(PINS),
           '--env', 'PYTHONPATH=' + str(ROOT / 'src'), '--env', 'OMP_NUM_THREADS=1',
           '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'MKL_NUM_THREADS=1',
           '--env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', '--cpu-cores', '9',
           '--ram-mib', '98304', '--vram-mib', str(vram), '--disk-gib', '2',
           '--wall-hours', '48' if stage == 'run' else '2']
    if stage != 'generate':
        raw += ['--input', str(root / 'inputs.json')]
    if stage == 'run':
        raw += ['--input', str(root / 'preflight.json')]
    argv = invocation(stage, root, args.lock, ops_root)
    spec = launcher.spec_from_args(launcher.build_parser().parse_args([*raw, '--', *argv]))
    metadata = launcher.prepare(spec)
    value = {'status': 'prepared_dry', 'spec': spec, 'metadata': metadata,
             'vram_declaration_scope': 'not_measured conservative first limit' if stage == 'preflight' else 'measured worker peak x9 x1.15 ceil256' if stage == 'run' else 'cpu_only'}
    if args.command == 'prepare':
        return value
    if (ops_root / 'runs' / run_id).exists():
        raise FileExistsError('original run exists')
    for state in ('pending', 'claimed', 'rejected'):
        if list((ops_root / 'queue' / state).glob('*-' + run_id + '.json')):
            raise FileExistsError('run already queued')
    path = scheduler.submit(launcher.resolve_root(str(ops_root)), spec)
    return {**value, 'status': 'written', 'queue_file': path.name}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('dry')
    for name in ('generate', 'preflight', 'run', 'prepare', 'submit'):
        item = sub.add_parser(name)
        for field in ('root', 'lock', 'ops-root'):
            item.add_argument('--' + field, type=Path, required=True)
        if name in ('prepare', 'submit'):
            item.add_argument('--ops-code', required=True)
            item.add_argument('--stage', choices=('generate', 'preflight', 'run'), required=True)
        else:
            item.add_argument('--execute-on-new-instance', action='store_true', required=True)
    args = parser.parse_args(argv)
    if args.command == 'dry':
        value = {'items': N, 'branching': K, 'depth': DEPTH, 'queries': QUERIES,
                 'history': HISTORY, 'workers': list(WORKERS), 'science_training': False}
    elif args.command in ('prepare', 'submit'):
        value = job(args)
    else:
        value = globals()[args.command](args.root, args.lock, args.ops_root)
    print(json.dumps(value, sort_keys=True))
    from analysis import g_decode_timing as canonical
    if canonical._SAMPLER is not None:
        canonical._SAMPLER.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
