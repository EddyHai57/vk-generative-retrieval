"""Frozen S1 SGNS. Callers supply the sealed full train event sequences only.

No dataset reader or checkpoint access is provided here. Histories must already be
uniquely ordered within user; their canonical digest binds that exact input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .sid_tok_audit import feature_digest
from .sid_v2 import catalog_digest


@dataclass(frozen=True)
class SGNSConfig:
    dim: int = 64
    window: int = 5
    epochs: int = 5
    negatives: int = 5
    batch_size: int = 4096
    seed: int = 42
    initial_lr: float = 0.025
    final_lr_ratio: float = 1e-4

    def validate(self, engineering: bool) -> None:
        for key in ('dim', 'window', 'epochs', 'negatives', 'batch_size'):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be nonnegative integer')
        if not np.isfinite(self.initial_lr) or self.initial_lr <= 0:
            raise ValueError('invalid learning rate')
        if not 0 < self.final_lr_ratio <= 1:
            raise ValueError('invalid final learning rate ratio')
        if not engineering and self != SGNSConfig():
            raise ValueError('scientific configuration is frozen; overrides require engineering=True')


def iter_positive_pairs(histories: Mapping[int, Sequence[int]], window: int):
    """Yield directed pairs in frozen user/center/context order, keeping repeats."""
    for user in sorted(histories):
        events = histories[user]
        for center, item in enumerate(events):
            for context in range(max(0, center-window), min(len(events), center+window+1)):
                if events[context] != item:
                    yield int(item), int(events[context])


def learning_rate(processed: int, total: int, initial: float = .025,
                  final_ratio: float = 1e-4) -> float:
    if total <= 0 or not 0 <= processed <= total:
        raise ValueError('invalid positive-pair progress')
    return initial * (1 - (1-final_ratio) * processed / total)


def sgns_loss(center: torch.Tensor, positive: torch.Tensor,
              negative: torch.Tensor) -> torch.Tensor:
    """SUM over pairs, and SUM over each pair's negatives (never a mean)."""
    pos = (center * positive).sum(dim=-1)
    neg = (center.unsqueeze(1) * negative).sum(dim=-1)
    return -F.logsigmoid(pos).sum() - F.logsigmoid(-neg).sum()


def _sample_from_cdf(centers, positives, cdf, count, rng):
    out = np.empty((len(centers), count), dtype=np.int64)
    # Per-pair rejection consumes a continuous RNG stream, independent of batching.
    for row, (center, positive) in enumerate(zip(centers, positives)):
        filled = 0
        while filled < count:
            candidates = np.searchsorted(cdf, rng.random(count-filled), side='right')
            accepted = candidates[(candidates != center) & (candidates != positive)]
            out[row, filled:filled+len(accepted)] = accepted
            filled += len(accepted)
    return out


def sample_negatives(centers, positives, probabilities, count, rng):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError('invalid negative distribution')
    total_mass = probabilities.sum()
    if not np.isclose(total_mass, 1):
        raise ValueError('negative distribution must sum to one')
    centers = np.asarray(centers, dtype=np.int64)
    positives = np.asarray(positives, dtype=np.int64)
    if centers.shape != positives.shape or centers.ndim != 1:
        raise ValueError('negative pair shapes mismatch')
    if (total_mass-probabilities[centers]-probabilities[positives] <= 0).any():
        raise ValueError('negative distribution has no eligible item')
    cdf = probabilities.cumsum(); cdf[-1] = 1.
    return _sample_from_cdf(centers, positives, cdf, count, rng)


def _array_digest(array) -> str:
    array = np.ascontiguousarray(array)
    header = json.dumps({'dtype': array.dtype.str, 'shape': list(array.shape)}, sort_keys=True).encode()
    return hashlib.sha256(header + b'\n' + array.tobytes()).hexdigest()


def _sequence_digest(histories) -> str:
    digest = hashlib.sha256()
    for user in sorted(histories):
        digest.update(json.dumps([int(user), [int(x) for x in histories[user]]], separators=(',', ':')).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def health_check(vectors, catalog, histories, *, require_pass=True):
    """S0 adjacent different-item cosine versus equal-size Dprime random pairs."""
    rows = {int(item): row for row, item in enumerate(catalog)}
    rng = random.Random(0)
    adjacent_sum = random_sum = 0.
    count = 0
    for user in sorted(histories):
        events = histories[user]
        for left, right in zip(events, events[1:]):
            if left == right:
                continue
            adjacent_sum += float(np.dot(vectors[rows[int(left)]], vectors[rows[int(right)]]))
            a, b = rng.sample(range(len(catalog)), 2)
            random_sum += float(np.dot(vectors[a], vectors[b]))
            count += 1
    if count == 0:
        raise ValueError('health check has no adjacent different-item pairs')
    adjacent, baseline = adjacent_sum/count, random_sum/count
    passed = bool(np.isfinite([adjacent, baseline]).all() and adjacent > baseline)
    audit = {'adjacent_pairs': count, 'cooccurrence_mean_cosine': adjacent,
             'random_mean_cosine': baseline, 'difference': adjacent-baseline,
             'random_method': 'Python Random(0).sample(range(N),2)', 'passed': passed}
    if require_pass and not passed:
        raise ValueError('SGNS health check failed: ' + json.dumps(audit, sort_keys=True))
    return audit


def fit_sgns(catalog, histories, content, *, device='cpu', engineering=False, config=None):
    """Return (float32 catalog-ordered vectors, audit); no implicit writes.

    LR is evaluated at the number of positives already processed before each SGD
    batch. The terminal scheduler state after the last batch equals the floor.
    CUDA requests use deterministic algorithms and fail if a kernel cannot honor
    them. CPU is the default and needs no CUDA-only determinism assumptions.
    """
    config = config or SGNSConfig()
    if type(engineering) is not bool:
        raise ValueError('engineering must be explicit boolean')
    config.validate(engineering)
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    if device == 'cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise ValueError('CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG before initialization')
    catalog = np.asarray(catalog)
    if catalog.ndim != 1 or catalog.dtype.kind not in 'iu' or len(catalog) < 3:
        raise ValueError('catalog must contain at least three integer item IDs')
    if (catalog <= 0).any() or not np.all(catalog[1:] > catalog[:-1]):
        raise ValueError('catalog IDs must be unique positive ascending; PAD/OOV forbidden')
    content = np.asarray(content, dtype=np.float32)
    if content.shape != (len(catalog), config.dim) or not np.isfinite(content).all():
        raise ValueError('content missing, shape mismatch or nonfinite')
    content_norm = np.linalg.norm(content.astype(np.float64), axis=1)
    members = set(map(int, catalog))
    for user, events in histories.items():
        if not isinstance(user, (int, np.integer)) or isinstance(user, bool):
            raise ValueError('users must have integer IDs for ascending order')
        for item in events:
            if not isinstance(item, (int, np.integer)) or isinstance(item, bool) or int(item) not in members:
                raise ValueError('history contains PAD/OOV or noninteger item')
    rows = {int(item): row for row, item in enumerate(catalog)}
    frequency = np.zeros(len(catalog), dtype=np.int64)
    for events in histories.values():
        for item in events:
            frequency[rows[int(item)]] += 1
    participation = np.zeros(len(catalog), dtype=np.int64)
    pairs_per_epoch = 0
    for center, positive in iter_positive_pairs(histories, config.window):
        participation[rows[center]] += 1
        participation[rows[positive]] += 1
        pairs_per_epoch += 1
    if ((participation == 0) & (content_norm == 0)).any():
        raise ValueError('zero-norm content for no-positive fallback')
    if not pairs_per_epoch:
        raise ValueError('no positive pairs; health check undefined')
    weights = frequency.astype(np.float64) ** .75
    if np.count_nonzero(weights) < 3:
        raise ValueError('negative distribution has no eligible item')
    probabilities = weights / weights.sum()
    cdf = probabilities.cumsum(); cdf[-1] = 1.
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator(device='cpu').manual_seed(config.seed)
    initial = torch.empty((len(catalog), config.dim), dtype=torch.float32)
    initial.uniform_(-.5/config.dim, .5/config.dim, generator=generator)
    initialization_digest = _array_digest(initial.numpy())
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        center_table = torch.nn.Embedding.from_pretrained(initial.to(device), freeze=False, sparse=True)
        output_table = torch.nn.Embedding.from_pretrained(torch.zeros_like(initial).to(device), freeze=False, sparse=True)
        optimizer = torch.optim.SGD([center_table.weight, output_table.weight], lr=config.initial_lr,
                                    momentum=0, weight_decay=0)
        total = pairs_per_epoch * config.epochs
        processed = steps = 0
        negative_digest = hashlib.sha256()
        epoch_audits = []
        first_lr = last_lr = None
        for epoch in range(config.epochs):
            batch = []
            epoch_loss = 0.
            def train_batch(batch):
                nonlocal processed, steps, epoch_loss, first_lr, last_lr
                pair_array = np.asarray(batch, dtype=np.int64)
                negatives = _sample_from_cdf(pair_array[:,0], pair_array[:,1], cdf, config.negatives, rng)
                negative_digest.update(negatives.astype('<i8', copy=False).tobytes())
                lr = learning_rate(processed, total, config.initial_lr, config.final_lr_ratio)
                if first_lr is None:
                    first_lr = lr
                last_lr = lr
                optimizer.param_groups[0]['lr'] = lr
                idx = torch.from_numpy(pair_array).to(device)
                neg = torch.from_numpy(negatives).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = sgns_loss(center_table(idx[:,0]), output_table(idx[:,1]), output_table(neg))
                if not torch.isfinite(loss):
                    raise ValueError('SGNS nonfinite loss; STOP')
                loss.backward()
                for table in (center_table, output_table):
                    if not torch.isfinite(table.weight.grad.coalesce().values()).all():
                        raise ValueError('SGNS nonfinite gradient; STOP')
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                processed += len(batch); steps += 1
                optimizer.param_groups[0]['lr'] = learning_rate(processed, total, config.initial_lr, config.final_lr_ratio)
            for center, positive in iter_positive_pairs(histories, config.window):
                batch.append((rows[center], rows[positive]))
                if len(batch) == config.batch_size:
                    train_batch(batch); batch = []
            if batch:
                train_batch(batch)
            if not torch.isfinite(center_table.weight).all() or not torch.isfinite(output_table.weight).all():
                raise ValueError('SGNS nonfinite table; STOP')
            epoch_audits.append({'epoch': epoch+1, 'processed_positive_pairs': processed,
                                 'summed_loss': epoch_loss, 'steps': steps})
        vectors = center_table.weight.detach().cpu().numpy().copy()
        output_digest = _array_digest(output_table.weight.detach().cpu().numpy())
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn)
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    fallback = participation == 0
    if not np.isfinite(vectors).all() or (norms[~fallback] == 0).any():
        raise ValueError('SGNS nonfinite or positive-participating zero center; STOP')
    vectors[~fallback] = (vectors[~fallback].astype(np.float64)/norms[~fallback,None]).astype(np.float32)
    vectors[fallback] = (content[fallback].astype(np.float64)/content_norm[fallback,None]).astype(np.float32)
    health = health_check(vectors, catalog, histories)
    audit = {'schema': 'vkgr.sid_coll_sgns.v1', 'config': asdict(config), 'engineering': engineering,
             'device': device, 'torch_version': str(torch.__version__), 'threads': 1,
             'deterministic_algorithms': True, 'epochs_completed': config.epochs,
             'training_contract': {'loss_reduction': 'sum', 'optimizer': 'SGD',
                                   'momentum': 0, 'weight_decay': 0, 'gradient_clipping': None,
                                   'frequency_power': .75, 'subsampling': False,
                                   'distance_weighting': False, 'watch_ratio_weighting': False,
                                   'early_stopping': False, 'export_table': 'center_only',
                                   'initial_input_bound': .5/config.dim,
                                   'positive_order': 'user ascending; center ascending; context oldest to newest',
                                   'negative_rng': 'numpy.default_rng(seed), continuous across epochs'},
             'positive_pairs_per_epoch': pairs_per_epoch, 'processed_positive_pairs': processed,
             'steps': steps, 'epoch_audits': epoch_audits,
             'learning_rates': {'initial': first_lr, 'last_batch': last_lr,
                               'final': learning_rate(processed,total,config.initial_lr,config.final_lr_ratio),
                               'evaluation': 'before batch at processed positive count; final is terminal scheduler state'},
             'frequency_sha256': _array_digest(frequency),
             'catalog_sha256': catalog_digest(catalog),
             'raw_train_sequence_sha256': _sequence_digest(histories),
             'train_sequence_sha256': _sequence_digest(histories),
             'content_feature_sha256': feature_digest(content),
             'content_sha256': feature_digest(content), 'vectors_sha256': feature_digest(vectors),
             'input_initialization_sha256': initialization_digest,
             'output_initialization': 'zeros',
             'output_table_sha256': output_digest, 'negative_trace_sha256': negative_digest.hexdigest(),
             'positive_participation': participation.tolist(),
             'positive_participation_sha256': _array_digest(participation),
             'fallback_mask': fallback.tolist(), 'fallback_mask_sha256': _array_digest(fallback),
             'fallback_count': int(fallback.sum()), 'fallback_proportion': float(fallback.mean()),
             'representation': 'collaborative centers; no-positive items use content fallback in a different space',
             'health': health}
    return vectors, audit


def save_sgns_output(out_dir, vectors, audit):
    """Write only to a brand-new directory; partial failures remain evidence."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    with (out_dir/'vectors.npy').open('xb') as handle:
        np.save(handle, vectors, allow_pickle=False)
    with (out_dir/'audit.json').open('x', encoding='utf-8') as handle:
        json.dump(audit, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
