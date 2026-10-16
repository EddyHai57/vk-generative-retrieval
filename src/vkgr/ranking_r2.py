"""R2 isolated objective and train-only mining engineering primitives.

No training entry point or checkpoint loading. The frozen A1/T3 modules stay unchanged.
"""
import copy
import math
import random
import time
import torch
from torch.nn import functional as F


def sequence_logprob(logits, labels):
    """Sum full-vocabulary next-SID log probabilities over all identifier positions."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.long or not torch.isfinite(logits).all():
        raise ValueError('sequence log probability tensor schema')
    if (labels < 0).any() or (labels >= logits.shape[-1]).any():
        raise ValueError('label outside vocabulary')
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum(-1)


def dpo_with_ce(chosen, rejected, ref_chosen, ref_rejected, positive_ce):
    """Frozen engineering coefficients: beta=.1, positive next-SID CE weight=1."""
    values = (chosen, rejected, ref_chosen, ref_rejected)
    if chosen.ndim != 1 or not chosen.numel() or any(x.shape != chosen.shape or not torch.isfinite(x).all() for x in values):
        raise ValueError('finite matching sequence log probability vectors required')
    if positive_ce.ndim != 0 or not torch.isfinite(positive_ce):
        raise ValueError('finite scalar positive CE required')
    margin = (chosen - rejected) - (ref_chosen.detach() - ref_rejected.detach())
    return -F.logsigmoid(.1 * margin).mean() + positive_ce


def frozen_reference(model):
    """Independent reference snapshot, deterministic eval mode, without gradients."""
    reference = copy.deepcopy(model)
    reference.eval()
    reference.requires_grad_(False)
    return reference


def select_negative(ranked_items, row):
    """First beam item outside target and complete target-prefix past_likes.

    Observed exposure history is NOT a like list and must not be used as the ban set.
    No eligible candidate returns None; it is never replaced by a random item.
    """
    if type(row.get('target')) is not int or type(row.get('past_likes')) is not list or any(type(i) is not int for i in row['past_likes']):
        raise ValueError('explicit target and full past_likes required')
    if any(type(i) is not int for i in ranked_items):
        raise ValueError('integer ranked items required')
    banned = set(row['past_likes']) | {row['target']}
    return next((item for item in ranked_items if item not in banned), None)


def sample_training_indices(population):
    """Exactly 20,000 unique train indices using a local Python Random(0)."""
    if type(population) is not int or population < 20000:
        raise ValueError('at least 20000 training examples required; no sample-size fallback')
    return random.Random(0).sample(range(population), 20000)


def benchmark(model, tokens, rows, *, estimate_count=20000):
    """Decode only supplied training rows. No weights, predictions or examples are persisted."""
    from .sid_depth_beam import decode_histories_batched_d
    if not rows or estimate_count != 20000:
        raise ValueError('nonempty bounded train microbatch and fixed 20000 estimate required')
    device = next(model.parameters()).device
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    model.eval()
    with torch.inference_mode():
        # Separate one-query warmup; timed queries remain the exact selected prefix.
        decode_histories_batched_d(model, tokens, [rows[0]['history']], 10, 1)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        results = decode_histories_batched_d(model, tokens, [row['history'] for row in rows], 10, 1)
        eligible = sum(select_negative(items, row) is not None for row, (items, _) in zip(rows, results))
        sync()
        elapsed = time.perf_counter() - start
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('invalid measured wall time')
    estimate = elapsed / len(rows) * estimate_count
    return {'schema': 'r2-mining-engineering-v1', 'model_weights': 'random_initialization_engineering_only',
            'claim_boundary': 'random weights and sampled prefix; not new SFT best throughput or final mining admission',
            'beam': 10, 'decode_batch_size': 1, 'queries': len(rows), 'eligible_pairs': eligible,
            'elapsed_seconds': elapsed, 'queries_per_second': len(rows) / elapsed,
            'estimate_queries': estimate_count, 'linear_estimate_seconds': estimate,
            'over_four_hours': estimate > 4 * 3600,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}
