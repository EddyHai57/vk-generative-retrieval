"""Round24 loss adapters; existing baseline loss remains unchanged.

Corrected-logQ reduction adapted from NonameUntitled/logq,
train_in_batch_logq_new.py, commit f48221cad05612a2558966288e77020e61f3ff01.
Copyright the upstream contributors. Licensed under Apache-2.0, see
licenses/logq-Apache-2.0.txt. Changes: stable logsumexp, weighted rows,
full-known-positive conditional proposal, and caller-provided masking.
Paper: Khrylchenko et al., RecSys 2025, doi:10.1145/3705328.3748033.
MNS is independently implemented from Yang et al. 2020, Eq.5.
"""
import numpy as np
import torch
from torch.nn import functional as F


def negative_mask(item_ids, positive_sets, target_ids):
    ids = np.asarray(item_ids)
    return np.stack([
        ~np.isin(ids, np.fromiter(positives, dtype=ids.dtype)) & (ids != target)
        for positives, target in zip(positive_sets, target_ids)
    ])


def proposal_values(item_ids, positive_sets, frequency_counts):
    total = sum(frequency_counts.values())
    q = np.array([frequency_counts.get(int(item), 0) / total for item in item_ids])
    mass = np.array([sum(frequency_counts.get(int(item), 0) for item in positives) / total
                     for positives in positive_sets])
    return q, mass


def corrected_logq_rows(positive_scores, negative_scores, conditional_q, mask):
    """Upstream stop-gradient weighted loss, in log space (one value per row)."""
    corrected = (negative_scores - conditional_q.log()).masked_fill(~mask, -torch.inf)
    log_partition = torch.logsumexp(corrected, dim=-1)
    log_mean_partition = log_partition - mask.sum(-1).to(torch.float64).log()
    probability = torch.sigmoid(positive_scores - log_mean_partition)
    multiplier = (1.0 - probability).detach()
    return multiplier * (-positive_scores + log_partition)


def contrast_loss(users, items, q, weights, temperature, mask, mode,
                  *, excluded_mass=None, catalog_size=None):
    scores = (F.normalize(users.float(), dim=-1)
              @ F.normalize(items.float(), dim=-1).T / temperature).to(torch.float64)
    allowed = torch.as_tensor(mask, dtype=torch.bool, device=scores.device)
    valid = allowed.any(-1)
    skipped = int((~valid).sum())
    if not valid.any():
        return scores.sum() * 0, skipped
    # Column i is the deterministic positive for query i, before any correction.
    labels = torch.arange(len(users), device=scores.device)
    positive = scores[labels, labels]
    proposal = torch.as_tensor(q, dtype=torch.float64, device=scores.device)
    if mode == 'corrected-logq':
        mass = torch.as_tensor(excluded_mass, dtype=torch.float64, device=scores.device)
        conditional = proposal[None, :] / (1 - mass[valid, None])
        losses = corrected_logq_rows(positive[valid], scores[valid], conditional,
                                     allowed[valid])
    elif mode == 'mixed-uniform1024':
        batch = len(users)
        uniform = len(items) - batch
        mixed = batch / len(items) * proposal + uniform / len(items) / catalog_size
        logits = scores - mixed.log()[None, :]
        logits[labels, labels] = positive
        allowed = allowed.clone()
        allowed[labels, labels] = True
        losses = F.cross_entropy(logits.masked_fill(~allowed, -torch.inf)[valid],
                                 labels[valid], reduction='none')
    else:
        raise ValueError(mode)
    selected = weights.to(scores.device, dtype=torch.float64)[valid]
    return (losses * selected).sum() / selected.sum(), skipped
