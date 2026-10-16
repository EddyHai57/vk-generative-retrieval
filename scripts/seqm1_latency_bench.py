"""User tower ONLY: empirical validation lengths, synthetic feature values.

No checkpoint, embedding or holdout is read. Input preparation, H2D transfer,
feature store, item tower, ANN and network are outside the timed scope.
The inference that synthetic values represent gather cost remains unverified.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from vkgr.retrieval_php01 import make_php01_model
from vkgr.retrieval_seqm1 import ARM_ORDER, make_seqm1_model
from vkgr.training import seed_all

CATALOG = 120416
SUPPORTED = 21496
WIDTH = 50


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--arms', nargs='+', choices=ARM_ORDER, default=['mean'])
    parser.add_argument('--lengths-json', type=Path, required=True)
    parser.add_argument('--lengths-sha', required=True)
    parser.add_argument('--catalog', type=int, default=CATALOG)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--fixture-mode', action='store_true', help='non-formal tiny local tests only')
    parser.add_argument('--expected-count', type=int, default=2520)
    return parser


def validate_args(args):
    if min(args.batch, args.iters, args.expected_count) < 1 or args.warmup < 0 or args.catalog < 2:
        raise ValueError('invalid positive size / warmup')
    if not args.fixture_mode and (args.batch, args.iters, args.warmup, args.catalog, args.expected_count, args.seed, args.arms) != (512, 1000, 50, CATALOG, 2520, 7, ['mean']):
        raise ValueError('formal run requires batch512, timed1000, warmup50, catalog120416, source2520, seed7, mean only')


def histogram(lengths):
    return {str(n): count for n, count in sorted(Counter(lengths).items())}


def load_lengths(path, expected_sha, expected_count=2520):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError('validation JSON sha256 mismatch')
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError('validation query count mismatch')
    lengths = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('history'), list):
            raise ValueError('expected validation rows with history lists')
        n = len(row['history'])  # Do not inspect history values or target fields.
        if not 1 <= n <= WIDTH:
            raise ValueError('history length must be in 1..50')
        lengths.append(n)
    return {'sha256': actual, 'count': len(lengths), 'histogram': histogram(lengths),
            'lengths_sha256': hashlib.sha256(json.dumps(lengths, separators=(',', ':')).encode()).hexdigest(),
            'lengths': lengths}


def length_schedule(counts, size, seed):
    """Largest-remainder allocation + seeded shuffle; no uniform approximation."""
    total = sum(counts.values())
    allocated = {int(n): size * count // total for n, count in counts.items()}
    order = sorted(counts, key=lambda n: (-(size * counts[n] % total), int(n)))
    for n in order[:size - sum(allocated.values())]:
        allocated[int(n)] += 1
    result = [n for n in sorted(allocated) for _ in range(allocated[n])]
    random.Random(seed).shuffle(result)
    return result


def synthetic_batch(generator, lengths, catalog, width=WIDTH):
    rows = len(lengths)
    histories = torch.zeros(rows, width, dtype=torch.long)
    side = torch.zeros(rows, width, 6, dtype=torch.long)
    for j, n in enumerate(lengths):
        histories[j, :n] = torch.randint(1, catalog, (n,), generator=generator)
        side[j, :n, 0] = torch.randint(0, 64, (n,), generator=generator)
        side[j, :n, 1] = torch.randint(1, 256, (n,), generator=generator)
        side[j, :n, 2] = torch.randint(1, 256, (n,), generator=generator)
        side[j, :n, 3] = torch.sort(torch.randint(0, 25, (n,), generator=generator)).values
    return histories, side


def summary(milliseconds):
    ordered = sorted(milliseconds)
    def pct(q):
        return ordered[math.ceil(q * len(ordered)) - 1]
    return {'mean': statistics.fmean(ordered), 'p50': pct(.50), 'p95': pct(.95), 'p99': pct(.99), 'n': len(ordered)}


def run_batches(model, batches, device, catalog, seed, timed=True, online=False):
    """Keep only one batch on device. Every model deterministically replays inputs."""
    generator = torch.Generator().manual_seed(seed)
    digest = hashlib.sha256()
    times = []
    with torch.no_grad():
        for lengths in batches:
            inputs = synthetic_batch(generator, lengths, catalog, max(lengths) if online else WIDTH)
            for tensor in inputs:
                digest.update(tensor.numpy().tobytes())
            histories, side = (tensor.to(device) for tensor in inputs)
            if device == 'cuda':
                torch.cuda.synchronize()  # includes completed input transfer, before timer
            begin = time.perf_counter()
            model.users(histories, side)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begin) * 1000.
            if timed:
                times.append(elapsed / len(lengths))
    return times, digest.hexdigest()


def measure(name, model, device, catalog, batch, iters, warmup, source, seed=7):
    model.eval()
    report = {'parameters': sum(p.numel() for p in model.parameters())}
    for mode, size in [('offline', batch), ('online', 1)]:
        lengths = length_schedule(source['histogram'], iters * size, seed)
        warm = length_schedule(source['histogram'], warmup * size, seed + 1)
        split = lambda values: (values[i:i + size] for i in range(0, len(values), size))
        run_batches(model, split(warm), device, catalog, seed + 2, timed=False, online=mode == 'online')
        times, digest = run_batches(model, split(lengths), device, catalog, seed, online=mode == 'online')
        report[mode + '_ms_per_query'] = {**summary(times), 'raw_ms_per_query': times,
            'timed_calls': iters, 'warmup_calls': warmup, 'batch': size, 'timed_queries': len(lengths),
            'timed_length_histogram': histogram(lengths), 'warmup_length_histogram': histogram(warm),
            'input_sha256': digest, 'padding_width': WIDTH if mode == 'offline' else 'actual history length'}
    return report


def export_pass(model, lengths, device, catalog, batch, seed=7):
    batches = [lengths[i:i + batch] for i in range(0, len(lengths), batch)]
    times, digest = run_batches(model, batches, device, catalog, seed)
    return {'mean_ms_per_query': sum(ms * len(rows) for ms, rows in zip(times, batches)) / len(lengths),
            'calls': len(batches), 'queries': len(lengths), 'batch_sizes': [len(rows) for rows in batches],
            'length_histogram': histogram(lengths), 'input_sha256': digest,
            'scope': 'separate single validation-length export pass; synthetic values; width50; no preparation/transfer'}


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    source = load_lengths(args.lengths_json, args.lengths_sha, args.expected_count)
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('cuda requested but not available')
    supported = list(range(1, min(SUPPORTED, args.catalog - 1) + 1))
    content = torch.zeros(args.catalog, 64)
    report = {}
    for name in ['anchor'] + args.arms:
        seed_all(42)
        model = (make_php01_model(content, supported, WIDTH, 64) if name == 'anchor' else make_seqm1_model(content, supported, WIDTH, name)).to(args.device)
        report[name] = measure(name, model, args.device, args.catalog, args.batch, args.iters, args.warmup, source, args.seed)
        report[name]['validation_export_pass'] = export_pass(model, source['lengths'], args.device, args.catalog, args.batch, args.seed)
        del model
    base = report['anchor']
    ratio = {name: {mode + '_mean': row[mode + '_ms_per_query']['mean'] / base[mode + '_ms_per_query']['mean'] for mode in ('offline', 'online')} for name, row in report.items() if name != 'anchor'}
    print(json.dumps({'schema': 'seqm1-latency-v2', 'formal': not args.fixture_mode,
        'device': args.device, 'device_name': torch.cuda.get_device_name() if args.device == 'cuda' else 'cpu',
        'threads': torch.get_num_threads(), 'catalog': args.catalog, 'batch': args.batch, 'iters': args.iters,
        'warmup': args.warmup, 'seed': args.seed, 'model_init_seed': 42, 'torch': torch.__version__,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'length_source': {k: v for k, v in source.items() if k != 'lengths'},
        'models': report, 'ratio_vs_anchor': ratio, 'quantiles': 'nearest rank ceil(q*n)',
        'scope': 'model.users() only; real validation length distribution, synthetic values; not service latency; no checkpoint, feature store, item tower, ANN or network; full recompute, no KV cache'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
