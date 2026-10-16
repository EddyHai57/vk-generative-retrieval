"""Full-size synthetic stack probe. Own-PID GPU samples include context;
allocator peaks plus sampled nonallocator overhead give a conservative total.
No global free-memory delta is used. CPU output is explicitly VRAM UNKNOWN.
"""
import argparse
import json
import math
import resource
import os

import numpy as np
import torch

from vkgr import retrieval_feat04combo as m4
from vkgr import retrieval_seq_combo_stack as stack
from vkgr.baselines import blockwise_topk
from vkgr.retrieval_experiment import normalize_export
from vkgr.retrieval_negatives import masked_ce_logq
from vkgr.training import seed_all
from vkgr.seq_combo_stack_memory import own_gpu_mib, declared_vram_mib, measured_mib

CATALOG = 120415
SUPPORTED = 21496
BATCH = 512
WIDTH = 50
QUERIES = 2520
TEMPERATURE = 0.10
MIN_VRAM_MIB = 1024
HEADROOM = 1.15
ROUND_MIB = 256
VARIANTS = ('f4_combo_c_d_e',)


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
        vector = torch.randint(1, len(getattr(m4.old, edges)) + 2, (catalog,), generator=generator)
        vector[0] = 0
        ids[buffer] = vector
    return ids


def measure(variant, device, catalog, steps):
    supported_count = min(SUPPORTED, catalog)
    if device == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    samples=[]
    def observe(stage):
        if device=='cuda':
            torch.cuda.synchronize()
            samples.append({'stage':stage,'pid':os.getpid(),'own_process_mib':own_gpu_mib(),
                            'current_reserved_mib':torch.cuda.memory_reserved()/2**20,
                            'max_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                            'max_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
    observe('baseline')
    generator = torch.Generator().manual_seed(7)
    seed_all(42)
    content = torch.zeros(catalog + 1, 64)
    model = stack.make_model(content, list(range(1, supported_count + 1)), WIDTH, synthetic_bucket_ids(generator, catalog + 1, variant)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    finite = True
    for step in range(steps):
        histories, side = synthetic_batch(generator, BATCH)
        targets = torch.randperm(supported_count, generator=generator)[:BATCH] + 1
        log_q = torch.full((len(targets),), math.log(1 / supported_count))
        mask = torch.ones(len(targets), len(targets), dtype=torch.bool)
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss, _ = masked_ce_logq(model.users(histories.to(device), side.to(device)), model.candidate_items(targets.to(device)),
                                 torch.arange(len(targets), device=device), mask.to(device), TEMPERATURE, log_q)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        finite = finite and bool(torch.isfinite(loss.detach()))
        observe('train-'+str(step+1))
    model.eval(); item_rows = []; user_rows = []
    with torch.no_grad():
        for start in range(1, catalog + 1, BATCH):
            item_rows.append(model.candidate_items(torch.arange(start, min(start + BATCH, catalog + 1), device=device)).cpu().numpy())
        observe('catalog-export')
        for start in range(0, QUERIES, BATCH):
            histories, side = synthetic_batch(generator, min(BATCH, QUERIES - start))
            user_rows.append(model.users(histories.to(device), side.to(device)).cpu().numpy())
    observe('user-export')
    items = normalize_export(np.concatenate(item_rows)); users = normalize_export(np.concatenate(user_rows))
    blockwise_topk(users[:32], items, np.arange(1, catalog + 1), 50)
    observe('top50')
    if len(items)!=catalog or len(users)!=QUERIES:raise ValueError('actual export population differs from declared shape')
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    row = {'variant': variant, 'param_bytes': param_bytes, 'optimizer_state_bytes': 2 * param_bytes, 'finite_loss': finite,
           'ru_maxrss': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'catalog_items_exported':len(items),'users_exported':len(users),'history_width':WIDTH}
    if device == 'cuda':
        measured=measured_mib(samples)
        row.update(vram={'method':'own_pid_nvidia_smi+torch_peaks','samples':samples},
                   measured_mib=measured,declared_vram_mib=declared_vram_mib(measured))
    else:
        row['vram'] = 'UNKNOWN'
    del model, optimizer
    return row


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available')
    rows = [measure(variant, args.device, args.catalog, args.steps) for variant in args.variants]
    print(json.dumps({'task': 'P-SEQ-COMBO-STACK', 'device': args.device, 'catalog': args.catalog, 'steps': args.steps, 'batch': BATCH, 'queries': QUERIES, 'history': WIDTH,
                      'temperature': TEMPERATURE, 'min_vram_mib': MIN_VRAM_MIB, 'variants': rows}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
