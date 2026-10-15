"""FEAT-04b memory smoke: full-size synthetic shapes only (no data file is read, no path argument, nothing written).
For every variant (C f4_item_popularity, D f4_item_quality, E f4_recency_behavior): seed 42 -> make_feat04b_model -> AdamW -> ``--steps`` in-batch logQ
(alpha 1, T 0.10) steps with grad clip, the candidates taken from ``model.candidate_items`` -> full-catalog candidate export + 2520 user export + normalize_export
+ one 32-query blockwise top-50. C/D get random synthetic bucket ids over the full catalog (row 0 = PAD = 0).
CUDA: peak allocated / reserved and the mem_get_info delta (includes the CUDA context), plus the declared-VRAM rule
declared = max(1024, ceil(measured * 1.15 / 256) * 256) with measured = max(mem_get_info delta, max_reserved).
CPU: parameter + optimizer bytes and ru_maxrss only, vram = UNKNOWN. Prints one JSON line.
Usage: PYTHONPATH=src python scripts/feat04b_vram_smoke.py --device cuda [--variants f4_item_popularity f4_item_quality f4_recency_behavior]
"""
import argparse
import json
import math
import resource

import numpy as np
import torch

from vkgr import retrieval_feat04b as m4
from vkgr.baselines import blockwise_topk
from vkgr.retrieval_experiment import normalize_export
from vkgr.retrieval_negatives import masked_ce_logq
from vkgr.training import seed_all

CATALOG = 120415
SUPPORTED = 21496
BATCH = 512
WIDTH = 50
QUERIES = 2520
TEMPERATURE = 0.10
MIN_VRAM_MIB = 1024
HEADROOM = 1.15
ROUND_MIB = 256
VARIANTS = ('f4_item_popularity', 'f4_item_quality', 'f4_recency_behavior')


def declared_vram_mib(measured_mib):
    """measured x 1.15, rounded up to a multiple of 256 MiB, never below the 1024 MiB floor of a P job."""
    return max(MIN_VRAM_MIB, math.ceil(measured_mib * HEADROOM / ROUND_MIB) * ROUND_MIB)


def measured_mib(used_mib, reserved_mib):
    """The conservative reading: the larger of the mem_get_info delta (includes the CUDA context) and the allocator's peak reserved."""
    return max(used_mib, reserved_mib)


def build_parser():
    parser = argparse.ArgumentParser(description='FEAT-04b VRAM smoke (synthetic shapes only)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--catalog', type=int, default=CATALOG)
    parser.add_argument('--steps', type=int, default=3)
    return parser


def synthetic_batch(generator, rows):
    """Full-width histories with side values in the registered ranges and non-decreasing weeks 0..24."""
    histories = torch.randint(1, 100, (rows, WIDTH), generator=generator)
    side = torch.zeros(rows, WIDTH, 6, dtype=torch.long)
    side[..., 0] = torch.randint(0, 64, (rows, WIDTH), generator=generator)
    side[..., 1] = torch.randint(1, 256, (rows, WIDTH), generator=generator)
    side[..., 2] = torch.randint(1, 256, (rows, WIDTH), generator=generator)
    side[..., 3] = torch.sort(torch.randint(0, 25, (rows, WIDTH), generator=generator), dim=1).values
    return histories, side


def synthetic_bucket_ids(generator, catalog, variant):
    """Random bucket ids over the whole index space for the variant's own two buffers (PAD row 0 stays 0); None for E."""
    if variant not in m4.STAT_VARIANTS:
        return None
    ids = {}
    for _, buffer, edges in m4.STAT_LAYOUT[variant]:
        vector = torch.randint(1, len(getattr(m4, edges)) + 2, (catalog,), generator=generator)
        vector[0] = 0
        ids[buffer] = vector
    return ids


def measure(variant, device, catalog, steps):
    supported_count = min(SUPPORTED, catalog - 1)
    if device == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
    generator = torch.Generator().manual_seed(7)
    seed_all(42)
    content = torch.zeros(catalog, 64)
    model = m4.make_feat04b_model(variant, content, list(range(1, supported_count + 1)), WIDTH, 42, synthetic_bucket_ids(generator, catalog, variant)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    finite = True; free_min = None
    for _ in range(steps):
        histories, side = synthetic_batch(generator, BATCH)
        targets = torch.randperm(supported_count, generator=generator)[:BATCH] + 1
        log_q = torch.full((len(targets),), math.log(1 / supported_count))
        mask = torch.ones(len(targets), len(targets), dtype=torch.bool)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss, _ = masked_ce_logq(model.users(histories.to(device), side.to(device)), model.candidate_items(targets.to(device)),
                                 torch.arange(len(targets), device=device), mask.to(device), TEMPERATURE, log_q)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        finite = finite and bool(torch.isfinite(loss.detach()))
        if device == 'cuda':
            free_min = torch.cuda.mem_get_info()[0] if free_min is None else min(free_min, torch.cuda.mem_get_info()[0])
    model.eval(); item_rows = []; user_rows = []
    with torch.no_grad():
        for start in range(1, catalog, BATCH):
            item_rows.append(model.candidate_items(torch.arange(start, min(start + BATCH, catalog), device=device)).cpu().numpy())
        for start in range(0, QUERIES, BATCH):
            histories, side = synthetic_batch(generator, min(BATCH, QUERIES - start))
            user_rows.append(model.users(histories.to(device), side.to(device)).cpu().numpy())
    items = normalize_export(np.concatenate(item_rows)); users = normalize_export(np.concatenate(user_rows))
    blockwise_topk(users[:32], items, np.arange(1, catalog), 50)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    row = {'variant': variant, 'param_bytes': param_bytes, 'optimizer_state_bytes': 2 * param_bytes, 'finite_loss': finite,
           'ru_maxrss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device == 'cuda':
        torch.cuda.synchronize(); free_min = min(free_min if free_min is not None else free_before, torch.cuda.mem_get_info()[0])
        used_mib = (free_before - free_min) / 2 ** 20
        allocated = torch.cuda.max_memory_allocated() / 2 ** 20; reserved = torch.cuda.max_memory_reserved() / 2 ** 20
        row.update(vram={'max_allocated_mib': allocated, 'max_reserved_mib': reserved, 'mem_get_info_delta_mib': used_mib},
                   measured_mib=measured_mib(used_mib, reserved), declared_vram_mib=declared_vram_mib(measured_mib(used_mib, reserved)))
    else:
        row['vram'] = 'UNKNOWN'
    del model, optimizer
    return row


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available')
    rows = [measure(variant, args.device, args.catalog, args.steps) for variant in args.variants]
    print(json.dumps({'task': 'P-FEAT04B', 'device': args.device, 'catalog': args.catalog, 'steps': args.steps, 'batch': BATCH, 'queries': QUERIES,
                      'temperature': TEMPERATURE, 'min_vram_mib': MIN_VRAM_MIB, 'variants': rows}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
