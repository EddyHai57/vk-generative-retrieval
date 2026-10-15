"""SEQ-M1 user-tower latency: anchor (F4-T010) vs SASRec-v3 arms, same script, same machine, same synthetic input distribution.
Measures ``model.users(histories, side)`` ONLY (histories + side tensors already gathered): no feature store, no item tower, no ANN, no network.
No data file is read, no path argument, nothing is written. Prints one JSON line.

offline  batch export: ``--batch`` queries of width 50 per call, reported as ms per query (design 3.11, batch 512 by default)
online   single query (batch 1), history length drawn per call: ``FULL_SHARE`` of calls have 50 events, the rest uniform 1..49
         (the validation share of n=50 is 62.7% per the M0 diagnosis; the split of the remaining 37.3% is an APPROXIMATION here)
Values are random in the registered ranges; user-tower cost does not depend on the values (gather position differences are ignored, an inference).
CPU runs use one thread. New events change P/G/O of every position, so the online number is a full recompute of <= 50 tokens (no KV cache).
Usage: PYTHONPATH=src python scripts/seqm1_latency_bench.py --device cpu|cuda [--arms mean last attn bi pure] [--iters 1000]
"""
import argparse
import json
import statistics
import time

import torch

from vkgr.retrieval_php01 import make_php01_model
from vkgr.retrieval_seqm1 import ARM_ORDER, make_seqm1_model
from vkgr.training import seed_all

CATALOG = 120416
SUPPORTED = 21496
WIDTH = 50
FULL_SHARE = .627


def build_parser():
    parser = argparse.ArgumentParser(description='SEQ-M1 user-tower latency (synthetic shapes only)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--arms', nargs='+', choices=ARM_ORDER, default=['mean'])
    parser.add_argument('--catalog', type=int, default=CATALOG)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--iters', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=50)
    return parser


def synthetic_batch(generator, lengths, catalog):
    rows = len(lengths)
    histories = torch.zeros(rows, WIDTH, dtype=torch.long); side = torch.zeros(rows, WIDTH, 6, dtype=torch.long)
    for j, n in enumerate(lengths):
        histories[j, :n] = torch.randint(1, catalog, (n,), generator=generator)
        side[j, :n, 0] = torch.randint(0, 64, (n,), generator=generator)
        side[j, :n, 1] = torch.randint(1, 256, (n,), generator=generator)
        side[j, :n, 2] = torch.randint(1, 256, (n,), generator=generator)
        side[j, :n, 3] = torch.sort(torch.randint(0, 25, (n,), generator=generator)).values
    return histories, side


def online_length(generator):
    if float(torch.rand(1, generator=generator)) < FULL_SHARE:
        return WIDTH
    return int(torch.randint(1, WIDTH, (1,), generator=generator))


def summary(milliseconds):
    ordered = sorted(milliseconds)

    def pct(q):
        return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
    return {'mean': statistics.fmean(ordered), 'p50': pct(.50), 'p95': pct(.95), 'p99': pct(.99), 'n': len(ordered)}


def time_calls(model, inputs, device, warmup):
    """Milliseconds per call; warm-up calls are run (cycling the inputs) and discarded."""
    out = []
    with torch.no_grad():
        for k in range(warmup):
            model.users(*inputs[k % len(inputs)])
        for histories, side in inputs:
            if device == 'cuda':
                torch.cuda.synchronize()
            begin = time.perf_counter()
            model.users(histories, side)
            if device == 'cuda':
                torch.cuda.synchronize()
            out.append((time.perf_counter() - begin) * 1000.)
    return out


def measure(name, model, device, catalog, batch, iters, warmup):
    model.eval()
    generator = torch.Generator().manual_seed(7)
    offline_inputs = [tuple(t.to(device) for t in synthetic_batch(generator, [WIDTH] * batch, catalog)) for _ in range(max(3, min(iters, 20)))]
    online_inputs = [tuple(t.to(device) for t in synthetic_batch(generator, [online_length(generator)], catalog)) for _ in range(iters)]
    offline = [ms / batch for ms in time_calls(model, offline_inputs, device, warmup)]
    online = time_calls(model, online_inputs, device, warmup)
    return {'offline_ms_per_query': summary(offline), 'online_ms_per_query': summary(online),
            'parameters': sum(p.numel() for p in model.parameters())}


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.device == 'cpu':
        torch.set_num_threads(1)
    elif not torch.cuda.is_available():
        raise SystemExit('cuda requested but not available')
    supported = list(range(1, min(SUPPORTED, args.catalog - 1) + 1)); content = torch.zeros(args.catalog, 64)
    models = {}
    seed_all(42); models['anchor'] = make_php01_model(content, supported, WIDTH, 64).to(args.device)
    for arm in args.arms:
        seed_all(42); models[arm] = make_seqm1_model(content, supported, WIDTH, arm).to(args.device)
    report = {name: measure(name, model, args.device, args.catalog, args.batch, args.iters, args.warmup) for name, model in models.items()}
    base = report['anchor']
    ratio = {name: {'offline_mean': row['offline_ms_per_query']['mean'] / base['offline_ms_per_query']['mean'],
                    'online_mean': row['online_ms_per_query']['mean'] / base['online_ms_per_query']['mean']} for name, row in report.items() if name != 'anchor'}
    print(json.dumps({'device': args.device, 'threads': torch.get_num_threads(), 'catalog': args.catalog, 'batch': args.batch, 'iters': args.iters, 'warmup': args.warmup,
                      'torch': torch.__version__, 'models': report, 'ratio_vs_anchor': ratio, 'scope': 'model.users() only; synthetic values; not a service latency'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
