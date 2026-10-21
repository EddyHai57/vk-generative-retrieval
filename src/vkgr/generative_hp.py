"""Batch1 G model factory + budget: INIT_EMB_N002 and BUDGET_E100_MIN5_P3 for the TIGER-style model.

Frozen G2 files (tiger.py, generative_training.py) are not modified. For history_len<=50
the model is exactly ``generative_training.make_model`` (value-equal under the same seed);
``init='small'`` then re-initialises only nn.Embedding tables to N(0, 0.02) with PAD=0.
The forward formula (token embedding x sqrt(d_model) + position) is unchanged (card risk 10).
"""
import math

import torch
from torch import nn

from .tiger import TigerModel

CORE_KEYS = ('d_model', 'heads', 'encoder_layers', 'decoder_layers', 'ff', 'dropout', 'precision', 'learning_rate', 'weight_decay', 'gradient_clip_norm', 'effective_batch', 'seed')


class EarlyStop:
    """Same decision rule and shared case table as the P line (retrieval_hp.EarlyStop)."""

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


def embedding_report(module):
    report = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.Embedding):
            weight = child.weight.detach()
            rows = torch.cat([weight[:child.padding_idx], weight[child.padding_idx + 1:]]) if child.padding_idx is not None else weight
            report[name] = {'rows': int(weight.shape[0]), 'std': float(rows.float().std()),
                            'pad_row_zero': bool(torch.all(weight[child.padding_idx] == 0)) if child.padding_idx is not None else None}
    return report


def init_embeddings(module, std=0.02):
    with torch.no_grad():
        for _, child in module.named_modules():
            if isinstance(child, nn.Embedding):
                child.weight.normal_(0.0, std)
                if child.padding_idx is not None:
                    child.weight[child.padding_idx].zero_()
    return embedding_report(module)


class TigerModelL(TigerModel):
    """TigerModel with a longer position table (L=100 -> 400 SID tokens)."""

    def __init__(self, vocab_size, d_model, nhead, encoder_layers, decoder_layers, ff, max_tokens):
        super().__init__(vocab_size, d_model, nhead, encoder_layers, decoder_layers, ff)
        self.max_tokens = max_tokens
        self.position = nn.Embedding(max_tokens, d_model)

    def embed(self, tokens):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.max_tokens:
            raise ValueError('invalid token sequence')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.embedding(tokens) * math.sqrt(self.d_model) + self.position(positions)[None]


def core_config(config):
    """The frozen G2 core view (used by generative_training and the sharded workers)."""
    return {k: config[k] for k in CORE_KEYS}


def make_model_b1(codec, config, *, device):
    from . import generative_training as gt
    if config['init'] not in ('legacy', 'small') or config['history_len'] not in (20, 50, 100):
        raise ValueError('unregistered G model switch')
    core = gt._core(core_config(config))
    if config['history_len'] <= 50:
        model = gt.make_model(codec, core, device=device)
    else:
        model = TigerModelL(codec.vocab_size, core['d_model'], core['heads'], 1, 1, core['ff'], 4 * config['history_len']).to(device=device, dtype=torch.float32)
    report = init_embeddings(model) if config['init'] == 'small' else embedding_report(model)
    model.init_report = {'init': config['init'], 'std': 0.02 if config['init'] == 'small' else None, 'embeddings': report}
    return model
