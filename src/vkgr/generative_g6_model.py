"""G6-PREP: the A1 model with a position table of (depth + 1) x history_len rows, and the depth-generic training step.

``tiger_arch.TigerArchModel`` fixes the absolute position table at 200 rows (50 items x 4 tokens); a depth-4 history of 50
items is 250 tokens. ``TigerArchModelD`` builds the table at the requested size in the same construction order, so with
``max_tokens == 200`` it is value-identical to the frozen class under the same seed (tests pin that). The frozen
``generative_training.full_vocab_ce`` / ``train_effective_batch`` accept exactly four positions and a depth-3 codec; the
copies here take ``depth + 1`` positions and the window tokenizer, and are otherwise line for line the same step.
"""
import math

import torch
from torch import nn

from . import generative_arch as ga
from . import generative_hp as ghp
from . import generative_training as gt
from .generative_data_v2 import sid_targets_l
from .sid_depth import ValidatedCodecD
from .tiger_arch import POSITIONS, TigerArchModel, _Core


class TigerArchModelD(TigerArchModel):
    """``TigerArchModel`` with ``max_tokens`` position rows; the module order and the RNG consumption are unchanged."""

    def __init__(self, vocab_size, d_model=128, heads=4, head_dim=32, encoder_layers=1, decoder_layers=1, ff=512, dropout=0.0, max_tokens=POSITIONS):
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive int')
        nn.Module.__init__(self)
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = nn.Embedding(max_tokens, d_model)
        self.transformer = _Core(d_model, heads, head_dim, encoder_layers, decoder_layers, ff, dropout)
        self.output = nn.Linear(d_model, vocab_size)
        self.spec = {'d_model': d_model, 'heads': heads, 'head_dim': head_dim, 'inner_dim': heads * head_dim, 'encoder_layers': encoder_layers,
                     'decoder_layers': decoder_layers, 'ff': ff, 'dropout': dropout, 'vocab': vocab_size, 'positions': max_tokens}

    def embed(self, tokens):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.max_tokens:
            raise ValueError('invalid token sequence')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.embedding(tokens) * math.sqrt(self.d_model) + self.position(positions)[None]


def make_model_arch_d(codec, config, *, device):
    """``generative_arch.make_model_arch`` for a depth-validated codec: position rows = (depth + 1) x history_len."""
    if not isinstance(codec, ValidatedCodecD):
        raise ValueError('depth-validated codec required')
    if config['init'] not in ('legacy', 'small') or config['history_len'] != 50:
        raise ValueError('unregistered G-ARCH model switch')
    if any(type(config[k]) is not int for k in ga.SHAPE_KEYS[:-1]) or type(config['dropout']) is not float:
        raise ValueError('typed architecture config required')
    max_tokens = (codec.depth + 1) * config['history_len']
    model = TigerArchModelD(codec.vocab_size, *(config[k] for k in ga.SHAPE_KEYS), max_tokens=max_tokens).to(device=device, dtype=torch.float32)
    report = ghp.init_embeddings(model) if config['init'] == 'small' else ghp.embedding_report(model)
    model.init_report = {'init': config['init'], 'std': 0.02 if config['init'] == 'small' else None, 'embeddings': report, 'arch': ga.arch_report(model)}
    return model


def full_vocab_ce_d(logits, labels, positions):
    """``generative_training.full_vocab_ce`` for ``positions`` (= depth + 1) teacher-forced positions."""
    if type(positions) is not int or logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape or logits.shape[0] < 1 \
            or logits.shape[1] != positions or logits.shape[2] < 3 or labels.dtype != torch.long or not logits.is_floating_point():
        raise ValueError('depth+1 position CE tensor schema')
    if not torch.isfinite(logits).all() or (labels < 0).any() or (labels >= logits.shape[2]).any():
        raise ValueError('nonfinite logits or invalid label index')
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())


def train_batch_d(model, optimizer, rows, codec, tokens, config, device):
    """One effective-batch step (microbatch accumulation, clip, AdamW) with the window tokenizer ``tokens``."""
    microbatch = config['microbatch']
    if type(microbatch) is not int or microbatch < 1 or not rows or len(rows) > 512:
        raise ValueError('nonempty bounded effective batch required')
    positions = codec.depth + 1
    model.train()
    optimizer.zero_grad(set_to_none=True)
    n = len(rows)
    total = 0.
    sizes = []
    try:
        for start in range(0, n, microbatch):
            batch = rows[start:start + microbatch]
            source, decoder, labels = (x.to(device) for x in sid_targets_l(batch, tokens))
            loss = full_vocab_ce_d(model(source, decoder), labels, positions)
            weighted = loss * (len(batch) / n)
            if not torch.isfinite(weighted):
                raise ValueError('nonfinite loss')
            weighted.backward()
            total += float(loss.detach()) * len(batch) / n
            sizes.append(len(batch))
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        except RuntimeError as exc:
            raise ValueError('nonfinite gradient') from exc
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('nonfinite updated parameters')
        return gt.StepResult(total, n, tuple(sizes))
    finally:
        optimizer.zero_grad(set_to_none=True)
