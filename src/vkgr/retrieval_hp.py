"""Batch1 P model factory and budget: INIT_EMB_N002 switch, L100 id_mean, BUDGET_E100_MIN5_P3.

Frozen modules (retrieval_models / retrieval_sasrec / retrieval_experiment) are not modified.
``init='legacy'`` builds exactly the frozen modules (value-equal under the same seed).
"""
import math

import torch

from .retrieval_models import TwoTower
from .retrieval_sasrec import SASRecIDTwoTower

INITS = ('legacy', 'small')
ARCHITECTURES = ('id_mean', 'sasrec')


class EarlyStop:
    """Patience counting starts after ``min_epochs``; best is chosen over all epochs.

    Ties count as bad epochs (strict ``>`` on (recall@50, ndcg@50)), as in the frozen rule.
    ``EarlyStop(30, 0, 3)`` is decision-equal to ``retrieval_experiment.ValidationSelection``.
    """

    def __init__(self, max_epochs, min_epochs, patience):
        if any(type(v) is not int for v in (max_epochs, min_epochs, patience)) or max_epochs < 1 or min_epochs < 0 or patience < 1 or min_epochs > max_epochs:
            raise ValueError('invalid early-stop budget')
        self.max_epochs = max_epochs; self.min_epochs = min_epochs; self.patience = patience
        self.epoch = 0; self.best_epoch = None; self.best_metric = None; self.bad_epochs = 0

    @property
    def stopped(self):
        return self.bad_epochs >= self.patience or self.epoch >= self.max_epochs

    def observe(self, epoch, recall, ndcg):
        if self.stopped or type(epoch) is not int or epoch != self.epoch + 1 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in (recall, ndcg)):
            raise ValueError('invalid validation budget/metrics')
        self.epoch = epoch; metric = (recall, ndcg)
        improved = self.best_metric is None or metric > self.best_metric
        if improved:
            self.best_metric = metric; self.best_epoch = epoch; self.bad_epochs = 0
        elif epoch > self.min_epochs:
            self.bad_epochs += 1
        return improved

    def state(self):
        return {'epoch': self.epoch, 'best_epoch': self.best_epoch, 'best_metric': list(self.best_metric) if self.best_metric is not None else None,
                'bad_epochs': self.bad_epochs, 'max_epochs': self.max_epochs, 'min_epochs': self.min_epochs, 'patience': self.patience}


def init_embeddings(module, std=0.02):
    """N(0, std) for every nn.Embedding, then zero the padding row. Other layers untouched."""
    with torch.no_grad():
        for _, child in module.named_modules():
            if isinstance(child, torch.nn.Embedding):
                child.weight.normal_(0.0, std)
                if child.padding_idx is not None:
                    child.weight[child.padding_idx].zero_()
    return embedding_report(module)


def embedding_report(module):
    report = {}
    for name, child in module.named_modules():
        if isinstance(child, torch.nn.Embedding):
            weight = child.weight.detach()
            rows = torch.cat([weight[:child.padding_idx], weight[child.padding_idx + 1:]]) if child.padding_idx is not None else weight
            report[name] = {'rows': int(weight.shape[0]), 'std': float(rows.float().std()),
                            'pad_row_zero': bool(torch.all(weight[child.padding_idx] == 0)) if child.padding_idx is not None else None}
    return report


class IDMeanL(TwoTower):
    """Frozen ``TwoTower('id_mean')`` with the history width bound raised to ``max_len``."""

    def __init__(self, content, supported_indices, max_len):
        super().__init__('id_mean', content, supported_indices)
        object.__setattr__(self, 'max_len', max_len)

    def users(self, histories):
        if histories.ndim != 2 or not 1 <= histories.shape[1] <= self.max_len or torch.any((histories != 0).sum(1) == 0):
            raise ValueError('invalid/empty history')
        valid = histories != 0; encoded = self.items(histories)
        return encoded.sum(1) / valid.sum(1, keepdim=True)


def make_model(architecture, init, content, supported_indices, max_len):
    if architecture not in ARCHITECTURES or init not in INITS or max_len not in (20, 50, 100):
        raise ValueError('unregistered model switch')
    if architecture == 'id_mean':
        model = TwoTower('id_mean', content, supported_indices) if max_len <= 50 else IDMeanL(content, supported_indices, max_len)
    else:
        if max_len > 50:
            raise ValueError('frozen SASRec position table is 50')
        model = SASRecIDTwoTower(content, supported_indices)
    report = init_embeddings(model) if init == 'small' else embedding_report(model)
    object.__setattr__(model, 'init_report', {'init': init, 'std': 0.02 if init == 'small' else None, 'embeddings': report})
    return model
