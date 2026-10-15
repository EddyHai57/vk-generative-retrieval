"""FEAT-04a memory smoke: full-size synthetic shapes only (no data file is read, no path argument, nothing written).
For every variant: seed 42 -> make_feat04a_model -> AdamW -> ``--steps`` in-batch logQ (alpha 1, T 0.10) steps with grad clip ->
full-catalog item export + 2520 user export + normalize_export + one 32-query blockwise top-50.
CUDA: peak allocated / reserved and the mem_get_info delta (includes the CUDA context), plus the declared-VRAM rule of the card
(measured x 1.15 rounded up to 256 MiB, never below the P-job floor 1024 MiB). CPU: parameter + optimizer bytes and ru_maxrss only, vram = UNKNOWN.
Prints one JSON line.
Usage: PYTHONPATH=src python scripts/feat04a_vram_smoke.py --device cuda [--features f4_recency_pool f4_hist_author_content]
"""
import argparse
import json
import math
import resource

import numpy as np
import torch

from vkgr import retrieval_feat04a as m4
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
FLOOR_MIB = 1024
AUTHOR_ROWS = 11135


def declared_vram_mib(measured_mib):
    return max(FLOOR_MIB, math.ceil(measured_mib * 1.15 / 256) * 256)


def build_parser():
    parser = argparse.ArgumentParser(description='FEAT-04a VRAM smoke (synthetic shapes only)')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--features', nargs='+', default=[m4.A_FEATURE], choices=list(m4.FEATURES4A))
    parser.add_argument('--catalog', type=int, default=CATALOG)
    parser.add_argument('--steps', type=int, default=3)
    return parser


def synthetic_batch(generator, rows, catalog):
    """Full-width histories with side values in the registered ranges and non-decreasing weeks 0..24."""
    histories = torch.randint(1, catalog, (rows, WIDTH), generator=generator)
    side = torch.zeros(rows, WIDTH, 6, dtype=torch.long)
    side[..., 0] = torch.randint(0, 64, (rows, WIDTH), generator=generator)
    side[..., 1] = torch.randint(1, 256, (rows, WIDTH), generator=generator)
    side[..., 2] = torch.randint(1, 256, (rows, WIDTH), generator=generator)
    side[..., 3] = torch.sort(torch.randint(0, 25, (rows, WIDTH), generator=generator), dim=1).values
    return histories, side


def build_model(feature, catalog, supported_count):
    kwargs = {}
    content = torch.zeros(catalog, 64)
    if feature == getattr(m4, 'B_FEATURE', None):
        g = torch.Generator().manual_seed(3)
        index = torch.randint(0, AUTHOR_ROWS, (catalog,), generator=g); index[0] = 0
        kwargs = {'author_index': index, 'vocab_size': AUTHOR_ROWS - 2}
        content = torch.nn.functional.normalize(torch.randn(catalog, 64, generator=g), dim=1); content[0] = 0.
    return m4.make_feat04a_model(feature, content, list(range(1, supported_count + 1)), WIDTH, 42, **kwargs)


def measure(feature, device, catalog, steps):
    supported_count = min(SUPPORTED, catalog - 1)
    if device == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
    seed_all(42)
    model = build_model(feature, catalog, supported_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(7); finite = True; free_min = None
    for _ in range(steps):
        histories, side = synthetic_batch(generator, BATCH, catalog)
        targets = torch.randperm(supported_count, generator=generator)[:BATCH] + 1
        log_q = torch.full((len(targets),), math.log(1 / supported_count))
        mask = torch.ones(len(targets), len(targets), dtype=torch.bool)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss, _ = masked_ce_logq(model.users(histories.to(device), side.to(device)), model.items(targets.to(device)),
                                 torch.arange(len(targets), device=device), mask.to(device), TEMPERATURE, log_q)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        finite = finite and bool(torch.isfinite(loss.detach()))
        if device == 'cuda':
            free_min = torch.cuda.mem_get_info()[0] if free_min is None else min(free_min, torch.cuda.mem_get_info()[0])
    model.eval(); item_rows = []; user_rows = []
    with torch.no_grad():
        for start in range(1, catalog, BATCH):
            item_rows.append(model.items(torch.arange(start, min(start + BATCH, catalog), device=device)).cpu().numpy())
        for start in range(0, QUERIES, BATCH):
            histories, side = synthetic_batch(generator, min(BATCH, QUERIES - start), catalog)
            user_rows.append(model.users(histories.to(device), side.to(device)).cpu().numpy())
    items = normalize_export(np.concatenate(item_rows)); users = normalize_export(np.concatenate(user_rows))
    blockwise_topk(users[:32], items, np.arange(1, catalog), 50)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    row = {'feature': feature, 'param_bytes': param_bytes, 'optimizer_state_bytes': 2 * param_bytes, 'finite_loss': finite,
           'ru_maxrss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device == 'cuda':
        torch.cuda.synchronize(); free_min = min(free_min if free_min is not None else free_before, torch.cuda.mem_get_info()[0])
        used_mib = (free_before - free_min) / 2 ** 20
        allocated = torch.cuda.max_memory_allocated() / 2 ** 20; reserved = torch.cuda.max_memory_reserved() / 2 ** 20
        measured = max(used_mib, reserved)
        row.update(vram={'max_allocated_mib': allocated, 'max_reserved_mib': reserved, 'mem_get_info_delta_mib': used_mib}, declared_vram_mib=declared_vram_mib(measured))
    else:
        row['vram'] = 'UNKNOWN'
    del model, optimizer
    return row


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available')
    rows = [measure(feature, args.device, args.catalog, args.steps) for feature in args.features]
    print(json.dumps({'task': 'FEAT-04A', 'device': args.device, 'catalog': args.catalog, 'steps': args.steps, 'batch': BATCH, 'queries': QUERIES,
                      'floor_vram_mib': FLOOR_MIB, 'features': rows}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
