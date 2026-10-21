"""Frozen CPU SGNS bounded synthetic cost; never exports vectors or health metrics."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / 'configs/sgns-throughput/source-pins.json'
PINS_SHA = '6caa6d1095aad3d25e1a3c9b70b380782067936ef8ef0476e55050a536ef823e'
N = 12_012_895
BATCHES = 2000
TOTAL_POSITIVES = 18_787_015_620
VERIFY_POSITIVE_RATIO = TOTAL_POSITIVES / 173_851_080
A_REPORT_SHA = '64f7eec73ac04c92b4b1a283e6e6223a8a3b9a2f88e20328b0a0c67e19ced158'


def references():
    from analysis.g_decode_timing import digest
    if digest(PINS) != PINS_SHA:
        raise ValueError('SGNS source ledger differs')
    value = json.loads(PINS.read_text())
    for name, sha in value['files'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('original SGNS source differs: ' + name)
    return value


def synthetic_histories(histogram, catalog_size, seed):
    import numpy as np
    if type(catalog_size) is not int or catalog_size < 3:
        raise ValueError('catalog >=3 required')
    rng = np.random.default_rng(seed)
    histories = {}
    for key, count in sorted(histogram.items(), key=lambda p: int(p[0])):
        length = int(key)
        if length < 1 or type(count) is not int or count < 1:
            raise ValueError('positive histogram lengths and integer counts required')
        for _ in range(count):
            histories[len(histories) + 1] = rng.integers(1, catalog_size + 1, size=length, dtype=np.uint32)
    return histories


def extrapolate(positive_pairs, seconds):
    if (type(positive_pairs) is not int or positive_pairs <= 0 or
            type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('positive finite duration and pair count required')
    throughput = positive_pairs / seconds
    fit = TOTAL_POSITIVES / throughput
    verify = 64.40261220932007 * VERIFY_POSITIVE_RATIO
    return {'positive_pairs_per_second': throughput, 'fit_seconds': fit,
            'verify_seconds': verify, 'total_seconds': fit + verify,
            'within_96_hours': fit + verify <= 96 * 3600,
            'verify_positive_ratio': VERIFY_POSITIVE_RATIO,
            'formula': '18787015620 / measured_positive_pairs_per_second; verify=64.40261220932007 * 18787015620 / 173851080'}


class _BatchBoundary(Exception):
    pass


def bounded_fit(catalog, histories, content, *, config=None, engineering=False, limit_batches=BATCHES):
    import torch
    from vkgr import sid_coll_sgns as sg
    config = config or sg.SGNSConfig()
    config.validate(engineering)
    if type(limit_batches) is not int or limit_batches < 1:
        raise ValueError('positive batch limit required')
    if not engineering and limit_batches != BATCHES:
        raise ValueError('formal batch limit is frozen')
    original_pairs, original_step = sg.iter_positive_pairs, torch.optim.SGD.step
    traversals = completed = 0
    started = None
    duration = None
    setup_started = time.perf_counter()

    def pairs(*args, **kwargs):
        nonlocal traversals, started
        traversals += 1
        if traversals == 2:
            started = time.perf_counter()
        return original_pairs(*args, **kwargs)

    def step(optimizer, *args, **kwargs):
        nonlocal completed, duration
        value = original_step(optimizer, *args, **kwargs)
        completed += 1
        if completed == limit_batches:
            if started is None or traversals != 2:
                raise ValueError('original SGNS training traversal differs')
            duration = time.perf_counter() - started
            raise _BatchBoundary()
        return value

    try:
        sg.iter_positive_pairs = pairs
        torch.optim.SGD.step = step
        try:
            sg.fit_sgns(catalog, histories, content, device='cpu', engineering=engineering, config=config)
        except _BatchBoundary:
            pass
        else:
            raise ValueError('original fit returned before bounded limit')
    finally:
        sg.iter_positive_pairs = original_pairs
        torch.optim.SGD.step = original_step
    if completed != limit_batches or duration is None or duration <= 0:
        raise ValueError('bounded batch measurement incomplete')
    return {'completed_batches': completed, 'measured_positive_pairs': completed * config.batch_size,
            'train_seconds': duration, 'setup_and_prescan_seconds': started - setup_started,
            'wall_seconds': time.perf_counter() - setup_started,
            'config': asdict(config), 'device': 'cpu', 'no_vectors_exported': True,
            'health_check_run': False, 'boundary': 'after original SGD.step before final-batch bookkeeping'}


def load_histogram(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != A_REPORT_SHA:
        raise ValueError('A aggregate identity differs')
    train = json.loads(raw)['train']
    histogram = train['exposure_length']['histogram']
    if (train['items'] != N or train['users'] != 99_956 or train['rows'] != 376_040_151 or
            sum(histogram.values()) != train['users'] or
            sum(int(k) * count for k, count in histogram.items()) != train['rows']):
        raise ValueError('A histogram conserved counts differ')
    return histogram


def execute(args):
    from analysis.g_decode_timing import fresh_path, environment, digest, write_new, peak_rss
    root = fresh_path(args.root)
    if root.exists():
        raise FileExistsError('new SGNS synthetic output directory required')
    pins = references()
    histogram = load_histogram(args.aggregate)
    env = environment(args.lock)
    from vkgr.throughput_config import guard, rss_bytes
    from analysis.a40_cost import selected
    from vkgr.sid_coll_sgns import SGNSConfig
    import numpy as np
    config = SGNSConfig()
    if pins['device'] != 'cpu' or pins['config'] != asdict(config):
        raise ValueError('original C1 device/config differs')
    guard(selected(), root.parent, projected_memory=64 * 1024**3, projected_disk=1024**2)
    root.mkdir()
    build_started = time.perf_counter()
    histories = synthetic_histories(histogram, N, config.seed)
    catalog = np.arange(1, N + 1, dtype=np.uint32)
    content = np.zeros((N, config.dim), dtype=np.float32)
    content[:, 0] = 1  # Synthetic nonzero fallback; not an exported representation.
    generation_seconds = time.perf_counter() - build_started
    result = bounded_fit(catalog, histories, content, config=config, engineering=False)
    value = {**env, 'schema': 'UR001_SYNTHETIC_SGNS_THROUGHPUT_V1',
             'status': 'bounded_original_sgns_timing_measured', 'seed': config.seed,
             'sgns_source_pins_sha256': digest(PINS), 'script_sha256': digest(__file__),
             'a_report_sha256': digest(args.aggregate), 'c1_spec_sha256': pins['c1_spec_sha256'],
             'c1_receipt_sha256': pins['c1_receipt_sha256'], 'device': 'cpu',
             'catalog': N, 'users': len(histories), 'synthetic_generation_seconds': generation_seconds,
             'measurement': result, 'extrapolation': extrapolate(result['measured_positive_pairs'], result['train_seconds']),
             'process_peak_rss_bytes': peak_rss(), 'process_rss_after_bytes': rss_bytes(),
             'context_inclusive_gpu_peak': 'NOT_APPLICABLE_CPU', 'no_recall_metrics': True,
             'limitations': ['synthetic item frequency/order differs from real train; histogram exact',
                            'full original validation/pair prescan and initialization measured separately',
                            'throughput excludes setup, includes sampling and original SGD; final bookkeeping after boundary omitted',
                            'input_verify extrapolated by frozen original full/subset positive-pair ratio; not remeasured']}
    write_new(root / 'timing.json', value)
    return value


def invocation(args):
    return [sys.executable, '-B', '-m', 'analysis.sgns_throughput', 'run',
            '--root', str(args.root), '--aggregate', str(args.aggregate), '--lock', str(args.lock),
            '--execute-on-new-instance']


def job(args):
    from analysis.a40_ops import load_ops
    from analysis.a40_cost import YAML
    from analysis.g_decode_timing import fresh_path
    references(); load_histogram(args.aggregate)
    root = fresh_path(args.root)
    if root.exists():
        raise FileExistsError('new output directory required')
    _, launcher, scheduler, _ = load_ops(args.ops_code)
    ops_root = fresh_path(args.ops_root)
    run_id = 'sgns-throughput-2000-v1'
    if (ops_root / 'runs' / run_id).exists():
        raise FileExistsError('original run exists')
    for state in ('pending', 'claimed', 'rejected'):
        if list((ops_root / 'queue' / state).glob('*-' + run_id + '.json')):
            raise FileExistsError('original queued identity exists')
    raw = ['--root', str(ops_root), '--run-id', run_id, '--kind', 'ops', '--cwd', str(ROOT),
           '--config', str(YAML), '--seed', '42', '--input', str(args.lock),
           '--input', str(PINS), '--input', str(args.aggregate),
           '--env', 'PYTHONPATH=' + str(ROOT / 'src'), '--env', 'OMP_NUM_THREADS=1',
           '--env', 'OPENBLAS_NUM_THREADS=1', '--env', 'MKL_NUM_THREADS=1',
           '--env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8', '--cpu-cores', '2',
           '--ram-mib', '98304', '--vram-mib', '0', '--disk-gib', '1', '--wall-hours', '48']
    spec = launcher.spec_from_args(launcher.build_parser().parse_args([*raw, '--', *invocation(args)]))
    metadata = launcher.prepare(spec)
    if args.command == 'prepare':
        return {'status': 'prepared_dry', 'spec': spec, 'metadata': metadata}
    path = scheduler.submit(launcher.resolve_root(str(ops_root)), spec)
    return {'status': 'written', 'queue_file': path.name, 'spec': spec, 'metadata': metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('dry')
    for command in ('run', 'prepare', 'submit'):
        item = sub.add_parser(command)
        for flag in ('root', 'aggregate', 'lock'):
            item.add_argument('--' + flag, type=Path, required=True)
        if command == 'run':
            item.add_argument('--execute-on-new-instance', action='store_true', required=True)
        else:
            item.add_argument('--ops-root', type=Path, required=True)
            item.add_argument('--ops-code', required=True)
    args = parser.parse_args(argv)
    if args.command == 'dry':
        print(json.dumps({'device': 'cpu', 'batches': BATCHES, 'items': N,
                          'science_training': False, 'vectors_exported': False}))
    else:
        print(json.dumps(execute(args) if args.command == 'run' else job(args), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
