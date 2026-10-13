"""P-NEG-01 primitives: logQ correction (alpha=1.0) and mixed in-batch + supported-uniform negatives.

Correction uses log of the expected number of times a column is sampled per batch
(sampled-softmax convention): in-batch target column j -> B * count_j / N_train_targets,
uniform column -> M / |pool after removing batch targets|. For in-batch only, the constant
log B cancels inside softmax, i.e. it equals plain log(count_j / N).
``ITEM_TARGET_FREQ_TRAIN_V1`` counts only the train target of each training example (no
validation/test, never a model input). ``SUPPORT_POOL_V1`` = train-defined supported items.
"""
import math

import torch

from .retrieval_training import masked_ce

LOGQ_ALPHA = 1.0


def target_frequency(rows):
    counts = {}
    for row in rows:
        target = row['target']
        if type(target) is not int:
            raise ValueError('invalid target')
        counts[target] = counts.get(target, 0) + 1
    if not counts:
        raise ValueError('empty target frequency')
    return {'name': 'ITEM_TARGET_FREQ_TRAIN_V1', 'counts': counts, 'total': sum(counts.values())}


def logq_batch(column_items, frequency):
    values = []
    for item in column_items:
        count = frequency['counts'].get(item, 0)
        if count <= 0:
            raise ValueError('in-batch column without train target frequency')
        values.append(math.log(count / frequency['total']))
    return torch.tensor(values, dtype=torch.float32)


def logq_mixed(column_items, uniform_items, frequency, *, batch_size, pool_size, extra_count):
    if type(batch_size) is not int or batch_size < 1 or type(pool_size) is not int or pool_size < 1 or type(extra_count) is not int or not 1 <= extra_count <= pool_size:
        raise ValueError('invalid mixed sampling sizes')
    values = []
    for item in column_items:
        if item in uniform_items:
            values.append(math.log(extra_count / pool_size))
        else:
            count = frequency['counts'].get(item, 0)
            if count <= 0:
                raise ValueError('in-batch column without train target frequency')
            values.append(math.log(batch_size * count / frequency['total']))
    return torch.tensor(values, dtype=torch.float32)


def masked_ce_logq(users, items, labels, mask, temperature, log_q, alpha=LOGQ_ALPHA):
    """``log_q=None`` delegates to the frozen ``masked_ce`` (bit-identical)."""
    if log_q is None:
        return masked_ce(users, items, labels, mask, temperature)
    if not math.isfinite(temperature) or temperature <= 0 or users.ndim != 2 or items.ndim != 2 or users.shape[1] != items.shape[1] or mask.shape != (len(users), len(items)) or labels.shape != (len(users),):
        raise ValueError('invalid loss input')
    if not isinstance(log_q, torch.Tensor) or log_q.shape != (len(items),) or not torch.isfinite(log_q).all() or alpha != LOGQ_ALPHA:
        raise ValueError('invalid logQ correction')
    if not torch.isfinite(users).all() or not torch.isfinite(items).all():
        raise ValueError('nonfinite vectors')
    if not mask[torch.arange(len(users)), labels].all():
        raise ValueError('masked target')
    valid = mask.sum(1) > 1; skips = int((~valid).sum())
    if not valid.any():
        return None, skips
    scores = torch.nn.functional.normalize(users.float(), dim=1) @ torch.nn.functional.normalize(items.float(), dim=1).T / temperature
    scores = scores - alpha * log_q.to(scores.device)[None, :]
    loss = torch.nn.functional.cross_entropy(scores.masked_fill(~mask, -torch.inf)[valid], labels[valid])
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss')
    return loss, skips


def supported_uniform_columns(rows, pool, idmap, rng, extra_count=128):
    """In-batch targets + ``extra_count`` distinct uniform draws from supported items not in the batch."""
    if not rows or pool != sorted(set(pool)) or any(idmap.get(i, 0) <= 0 for i in pool):
        raise ValueError('invalid supported pool')
    targets = [r['target'] for r in rows]
    if any(t not in idmap or idmap[t] <= 0 for t in targets):
        raise ValueError('invalid target')
    base = set(targets); candidates = [i for i in pool if i not in base]
    if len(candidates) < extra_count:
        raise ValueError('insufficient supported negative candidates')
    negatives = rng.sample(candidates, extra_count)
    columns = sorted(base | set(negatives)); lookup = {i: j for j, i in enumerate(columns)}
    masks = []
    for row in rows:
        past = set(row['past_likes'])
        masks.append([i == row['target'] or i not in past for i in columns])
    return (torch.tensor([idmap[i] for i in columns], dtype=torch.long), torch.tensor([lookup[i] for i in targets], dtype=torch.long),
            torch.tensor(masks, dtype=torch.bool), negatives, len(candidates))
