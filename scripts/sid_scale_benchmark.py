"""Synthetic-only, bounded CPU cost probe. Streaming is NOT balanced SID fitting."""
import os

THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
for _key in THREAD_ENV:
    os.environ[_key] = '1'

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

MIB = 1024 ** 2
SIZES = (1250000, 12470000)
METHODS = ('bhkm', 'rqvae', 'streaming')


def _positive(*values):
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError('positive integer sizes required')


def synthetic_blocks(n, block=2048, seed=42):
    _positive(n, block)
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    rng = np.random.default_rng(seed)
    for start in range(0, n, block):
        rows = rng.standard_normal((min(block, n-start), 64), dtype=np.float32)
        rows /= np.linalg.norm(rows, axis=1, keepdims=True)
        yield rows


def allocation_estimate(n, k=256, method='bhkm'):
    """Known live array floor, deliberately NOT a prediction of peak RSS."""
    _positive(n, k)
    if method not in METHODS:
        raise ValueError('unknown method')
    edges = n * k
    # _assign: float64 distances, int64 row/child/order, ids[row] uint32.
    # Root rows/input are at least two float32 copies; _distance temporaries are not counted.
    if method == 'bhkm':
        parts = {'root_edge_arrays': edges * 36, 'input_and_root_rows': n * 64 * 4 * 2}
    elif method == 'rqvae':
        # Worker materializes blocks via list + concatenate: both live at concatenate.
        # CPU torch.from_numpy shares X, so do NOT count another full tensor copy.
        parts = {'materialization_list_and_concat': n * 64 * 4 * 2}
    else:
        parts = {'block_and_distances': 2048 * (64*4 + k*8), 'centers': k*64*16}
    return {'method': method, 'edge_count': edges if method == 'bhkm' else None,
            'known_array_bytes': sum(parts.values()), 'components': parts,
            'measured': False, 'interpretation': 'known array floor; not peak RSS'}


def suffix_summary(codes):
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != 3 or not len(codes):
        raise ValueError('nonempty semantic three-column codes required')
    _, counts = np.unique(codes, axis=0, return_counts=True)
    return {'occupied_leaves': len(counts), 'max_leaf': int(counts.max()),
            'suffix_vocab': int(counts.max()), 'items_needing_suffix': int(counts[counts>1].sum()),
            'excess_collisions': int((counts-1).sum())}


def streaming_fit(n, k=256, block=2048, seed=42):
    """One-pass online Euclidean root k-means; no balancing, no full SID."""
    _positive(n, k, block)
    if n < k:
        raise ValueError('more centers than rows')
    counts = np.zeros(k, np.int64)
    centers = np.empty((k, 64), np.float64)
    initialized = 0
    assigned = 0
    for rows in synthetic_blocks(n, block, seed):
        take = min(k-initialized, len(rows))
        if take:
            centers[initialized:initialized+take] = rows[:take]
            counts[initialized:initialized+take] = 1
            initialized += take
            assigned += take
            rows = rows[take:]
        if not len(rows):
            continue
        data = rows.astype(np.float64)
        distances = (data*data).sum(1)[:, None] - 2*(data@centers.T) + (centers*centers).sum(1)
        labels = distances.argmin(1)
        batch_counts = np.bincount(labels, minlength=k)
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, data)
        updated = counts + batch_counts
        centers = (centers*counts[:, None]+sums)/updated[:, None]
        counts = updated
        assigned += len(rows)
    return {'assigned_items': assigned, 'cluster_counts': counts.tolist(),
            'finite': bool(np.isfinite(centers).all()), 'balanced': False,
            'passes': 1, 'suffix': 'UNKNOWN', 'scope': 'root prototype only; not production SID'}


def _numpy_threads():
    """Pin/query NumPy's bundled OpenBLAS without adding a runtime dependency.

    Unknown BLAS is fail-closed: the benchmark must not silently use many cores.
    """
    import ctypes
    candidates = list(Path(np.__file__).parent.glob('.dylibs/*openblas*'))
    candidates += list(Path(np.__file__).parent.parent.glob('numpy.libs/*openblas*'))
    reported = []
    for path in candidates:
        library = ctypes.CDLL(str(path))
        for suffix in ('64_', '_64_', ''):
            try:
                setter = getattr(library, 'openblas_set_num_threads'+suffix)
                getter = getattr(library, 'openblas_get_num_threads'+suffix)
            except AttributeError:
                continue
            setter.argtypes = [ctypes.c_int]
            getter.restype = ctypes.c_int
            setter(1)
            reported.append(getter())
            break
    if not reported or any(value != 1 for value in reported):
        raise RuntimeError('cannot verify NumPy BLAS single-threading')
    return reported


def _worker(method, n, k, block, seed, wall_seconds):
    import resource
    import torch
    from contextlib import nullcontext
    blas_threads = _numpy_threads()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with nullcontext():
        metadata = {'pid': os.getpid(), 'torch_threads': torch.get_num_threads(),
                    'torch_interop_threads': torch.get_num_interop_threads(),
                    'blas_threads': blas_threads,
                    'thread_env': {key: os.environ[key] for key in THREAD_ENV}}
        started = time.monotonic()
        if method == 'streaming':
            result = {'status': 'COMPLETE', **streaming_fit(n, k, block, seed)}
        else:
            from vkgr import sid, sid_rqvae
            values = np.concatenate(list(synthetic_blocks(n, block, seed)))
            if method == 'bhkm':
                fitted = sid.fit(np.arange(n, dtype=np.uint32), values, branching=k, depth=3, seed=seed)
                result = {'status': 'COMPLETE', 'suffix': suffix_summary(fitted.sids[:, :3])}
            else:
                fitted = sid_rqvae.fit(values, K=k, seed=seed,
                    max_steps=sid_rqvae.PREREG['max_steps'], device='cpu', wall_cap_seconds=wall_seconds)
                status = fitted['status']
                result = {'status': 'COMPLETE' if status == 'complete' else
                          ('CENSORED_WALL' if status in ('fail_wall', 'fail_wall_projected') else 'FAILED'),
                          'fit_status': status, 'completed_steps': fitted['step'],
                          'requested_steps': sid_rqvae.PREREG['max_steps'], 'suffix': 'UNKNOWN'}
                if status == 'complete':
                    result['suffix'] = suffix_summary(sid_rqvae.export_codes(fitted['state'], values))
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result.update(metadata, worker_wall_seconds=time.monotonic()-started,
                      worker_peak_rss_bytes=int(peak if sys.platform == 'darwin' else peak*1024))
        return result


def _rss_bytes(pid):
    value = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True, check=False)
    if value.returncode == 0 and value.stdout.strip():
        return int(value.stdout.strip()) * 1024
    return 0


def run_method(method, n, k=256, block=2048, seed=42, wall_seconds=60, rss_mib=4096):
    _positive(n, k, block)
    if method not in METHODS or n < k or not 0 < wall_seconds <= 60 or not 0 < rss_mib <= 4096:
        raise ValueError('invalid method or capped resource request')
    estimate = allocation_estimate(n, k, method)
    base = {'method': method, 'n': n, 'k': k, 'seed': seed, 'estimate': estimate,
            'suffix': 'UNKNOWN', 'wall_cap_seconds': wall_seconds, 'rss_cap_bytes': int(rss_mib*MIB)}
    if method != 'streaming' and estimate['known_array_bytes'] > rss_mib*MIB:
        return {**base, 'status': 'SKIPPED_RESOURCE', 'observed_peak_rss_bytes': None, 'wall_seconds': None}
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', method, '--n', str(n),
               '--k', str(k), '--block', str(block), '--seed', str(seed), '--wall-seconds', str(wall_seconds)]
    env = {**os.environ, **{key: '1' for key in THREAD_ENV}}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]/'src')
    start = time.monotonic()
    peak = 0
    status = None
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        while process.poll() is None:
            elapsed = time.monotonic()-start
            if elapsed >= wall_seconds:
                status = 'CENSORED_WALL'
                break
            peak = max(peak, _rss_bytes(process.pid))
            if peak > rss_mib*MIB:
                status = 'CENSORED_RSS'
                break
            time.sleep(min(.05, max(0, wall_seconds-(time.monotonic()-start))))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        out, err = process.communicate()
    base.update(wall_seconds=time.monotonic()-start, observed_peak_rss_bytes=peak,
                rss_measurement='parent sampled RSS; worker ru_maxrss when available')
    if status:
        return {**base, 'status': status}
    if process.returncode:
        return {**base, 'status': 'FAILED', 'returncode': process.returncode,
                'error': err[-2000:]}
    result = json.loads(out)
    if result['worker_peak_rss_bytes'] > rss_mib*MIB:
        result.update(status='CENSORED_RSS', suffix='UNKNOWN')
    if base['wall_seconds'] > wall_seconds:
        result.update(status='CENSORED_WALL', suffix='UNKNOWN')
    return {**base, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--worker', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--n', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--k', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--block', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--wall-seconds', type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 60:
        parser.error('wall cap must be in (0,60]')
    if args.worker:
        defaults = {'n': SIZES[0], 'k': 256, 'block': 2048, 'seed': 42}
        worker_args = {key: value if (value := getattr(args, key)) is not None else default
                       for key, default in defaults.items()}
        print(json.dumps(_worker(args.worker, **worker_args, wall_seconds=args.wall_seconds)))
        return
    if any(getattr(args, key) is not None for key in ('n', 'k', 'block', 'seed')):
        parser.error('--n, --k, --block and --seed are worker-only options')
    # Reserve the fresh destination before doing work; never overwrite old outputs.
    if args.output_dir:
        args.output_dir.mkdir(parents=False, exist_ok=False)
    report = {'schema': 'synthetic-sid-scale-v1', 'synthetic_only': True, 'seed': 42,
              'dimension': 64, 'threads': 1, 'results': [run_method(method, n, wall_seconds=args.wall_seconds) for n in SIZES for method in METHODS]}
    serialized = json.dumps(report, sort_keys=True)
    if args.output_dir:
        with (args.output_dir/'results.json').open('x') as target:
            target.write(serialized+'\n')
    print(serialized)


if __name__ == '__main__':
    main()
