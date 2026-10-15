"""P-HP-01 H4 memory smoke: full-size synthetic shapes only (no data file is read, no path argument, nothing written).

For every dim: seed 42 -> make_php01_model -> AdamW -> ``--steps`` in-batch logQ (alpha 1, T 0.15) steps with grad clip ->
full-catalog item export + 2520 user export + normalize_export + one 32-query blockwise top-50.
CUDA: peak allocated / reserved and the mem_get_info delta (includes the CUDA context), plus the declared-VRAM rule
(<= 3072 MiB measured -> 3072, else ceil(measured * 1.2 / 256) * 256). CPU: analytic parameter + optimizer bytes and
ru_maxrss only, vram = UNKNOWN. Prints one JSON line.
Usage: PYTHONPATH=src python scripts/php01_vram_smoke.py --device cuda [--dims 64 128]
"""
import argparse
import json
import math
import resource

import numpy as np
import torch

from vkgr.baselines import blockwise_topk
from vkgr.retrieval_experiment import normalize_export
from vkgr.retrieval_negatives import masked_ce_logq
from vkgr.retrieval_php01 import make_php01_model
from vkgr.retrieval_php01_run import make_optimizer
from vkgr.training import seed_all

CATALOG = 120415
SUPPORTED = 21496
BATCH = 512
WIDTH = 50
QUERIES = 2520
BASE_VRAM_MIB = 3072


def declared_vram_mib(measured_mib):
    return BASE_VRAM_MIB if measured_mib <= BASE_VRAM_MIB else math.ceil(measured_mib * 1.2 / 256) * 256


def build_parser():
    parser = argparse.ArgumentParser(description='P-HP-01 H4 VRAM smoke (synthetic shapes only)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--dims', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--catalog', type=int, default=CATALOG)
    parser.add_argument('--steps', type=int, default=3)
    return parser


def synthetic_batch(generator, rows, supported, catalog):
    """Full-width histories with side values in the registered ranges and non-decreasing weeks 0..24."""
    histories = torch.randint(1, catalog, (rows, WIDTH), generator=generator)
    side = torch.zeros(rows, WIDTH, 6, dtype=torch.long)
    side[..., 0] = torch.randint(0, 64, (rows, WIDTH), generator=generator)
    side[..., 1] = torch.randint(1, 256, (rows, WIDTH), generator=generator)
    side[..., 2] = torch.randint(1, 256, (rows, WIDTH), generator=generator)
    side[..., 3] = torch.sort(torch.randint(0, 25, (rows, WIDTH), generator=generator), dim=1).values
    return histories, side


def measure(dim, device, catalog, steps):
    supported_count = min(SUPPORTED, catalog - 1)
    if device == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
    seed_all(42)
    content = torch.zeros(catalog, 64)
    model = make_php01_model(content, list(range(1, supported_count + 1)), WIDTH, dim).to(device)
    optimizer = make_optimizer(model, {'learning_rate': 0.001, 'weight_decay': 0.0001})
    generator = torch.Generator().manual_seed(7); finite = True; free_min = None
    for _ in range(steps):
        histories, side = synthetic_batch(generator, BATCH, supported_count, catalog)
        targets = torch.randperm(supported_count, generator=generator)[:BATCH] + 1
        log_q = torch.full((len(targets),), math.log(1 / supported_count))
        mask = torch.ones(len(targets), len(targets), dtype=torch.bool)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss, _ = masked_ce_logq(model.users(histories.to(device), side.to(device)), model.items(targets.to(device)),
                                 torch.arange(len(targets), device=device), mask.to(device), 0.15, log_q)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        finite = finite and bool(torch.isfinite(loss.detach()))
        if device == 'cuda':
            free_min = torch.cuda.mem_get_info()[0] if free_min is None else min(free_min, torch.cuda.mem_get_info()[0])
    model.eval(); item_rows = []; user_rows = []
    with torch.no_grad():
        for start in range(1, catalog, BATCH):
            item_rows.append(model.items(torch.arange(start, min(start + BATCH, catalog), device=device)).cpu().numpy())
        for start in range(0, QUERIES, BATCH):
            histories, side = synthetic_batch(generator, min(BATCH, QUERIES - start), supported_count, catalog)
            user_rows.append(model.users(histories.to(device), side.to(device)).cpu().numpy())
    items = normalize_export(np.concatenate(item_rows)); users = normalize_export(np.concatenate(user_rows))
    blockwise_topk(users[:32], items, np.arange(1, catalog), 50)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    row = {'dim': dim, 'param_bytes': param_bytes, 'optimizer_state_bytes': 2 * param_bytes, 'finite_loss': finite,
           'ru_maxrss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device == 'cuda':
        torch.cuda.synchronize(); free_min = min(free_min if free_min is not None else free_before, torch.cuda.mem_get_info()[0])
        used_mib = (free_before - free_min) / 2 ** 20
        allocated = torch.cuda.max_memory_allocated() / 2 ** 20; reserved = torch.cuda.max_memory_reserved() / 2 ** 20
        measured = max(used_mib, reserved)
        row.update(vram={'max_allocated_mib': allocated, 'max_reserved_mib': reserved, 'mem_get_info_delta_mib': used_mib},
                   declared_vram_mib=declared_vram_mib(measured))
    else:
        row['vram'] = 'UNKNOWN'
    del model, optimizer
    return row


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available')
    rows = [measure(dim, args.device, args.catalog, args.steps) for dim in args.dims]
    print(json.dumps({'task': 'P-HP-01', 'device': args.device, 'catalog': args.catalog, 'steps': args.steps, 'batch': BATCH,
                      'queries': QUERIES, 'base_vram_mib': BASE_VRAM_MIB, 'dims': rows}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
