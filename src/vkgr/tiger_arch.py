"""GARCH-01: TIGER paper-scale encoder-decoder with independent attention head dimensions (explicit opt-in model).

``nn.MultiheadAttention`` ties the attention width to ``d_model`` (``d_model % heads == 0``). The TIGER paper uses
6 heads x 64 dims (inner_dim 384) at d_model 128, which the stock module cannot express. ``IndependentHeadAttention``
projects d_model -> 3 x inner_dim and back (``inner_dim = heads * head_dim``) and mirrors the operator order of
``F.multi_head_attention_forward(need_weights=False)``, so at ``inner_dim == d_model`` it is bit-identical to the stock
module in train mode and with the stock fast path disabled (tolerance only against the fused fast path, see the tests).

The default path (``tiger.TigerModel`` / ``generative_training``) is not touched: this module is reachable only through
``generative_arch`` / ``generative_b1_arch``. Positional encoding stays the anchor's absolute learned table (the paper
uses T5 relative position bias): a documented deviation.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

# Module-level aliases so tests can mutate this module's operators without touching torch.nn.functional (the stock model uses it too).
sdpa = F.scaled_dot_product_attention
linear = F.linear
relu = F.relu

POSITIONS = 200  # absolute position table rows, as in the anchor TigerModel


def expected_param_count(vocab, d_model, heads, head_dim, encoder_layers, decoder_layers, ff, positions=POSITIONS):
    """Closed form of the parameter count (calibrated against the stock model where inner_dim == d_model)."""
    inner = heads * head_dim
    attention = 3 * inner * d_model + 3 * inner + inner * d_model + d_model
    feed_forward = d_model * ff + ff + ff * d_model + d_model
    norm = 2 * d_model
    transformer = encoder_layers * (attention + 2 * norm + feed_forward) + decoder_layers * (2 * attention + 3 * norm + feed_forward) + 2 * norm
    return transformer + vocab * d_model + positions * d_model + d_model * vocab + vocab


class IndependentHeadAttention(nn.Module):
    """Attention with ``inner_dim = heads * head_dim`` independent of ``d_model``; batch-first (B, T, d_model) in and out."""

    def __init__(self, d_model, heads, head_dim, dropout):
        super().__init__()
        self.heads, self.head_dim, self.inner, self.dropout_p = heads, head_dim, heads * head_dim, dropout
        self.in_proj_weight = nn.Parameter(torch.empty((3 * self.inner, d_model)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * self.inner))
        self.out_proj = nn.Linear(self.inner, d_model)
        # same order as nn.MultiheadAttention: out_proj default init first, then xavier weight and zero biases
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def _project(self, query_t, kv_t, self_attention):
        """q, k, v as (L, B, inner): the packed in-projection of F.multi_head_attention_forward."""
        E = self.inner
        if self_attention:
            proj = linear(query_t, self.in_proj_weight, self.in_proj_bias)
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        w_q, w_kv = self.in_proj_weight.split([E, 2 * E])
        b_q, b_kv = self.in_proj_bias.split([E, 2 * E])
        kv = linear(kv_t, w_kv, b_kv).unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        return linear(query_t, w_q, b_q), kv[0], kv[1]

    def forward(self, query, kv, key_padding_mask=None, is_causal=False):
        B, Tq, _ = query.shape
        S = kv.shape[1]
        H, Dh = self.heads, self.head_dim
        query_t = query.transpose(1, 0)  # the (L, B, E) layout that batch_first F.multi_head_attention_forward works in
        self_attention = kv is query
        q, k, v = self._project(query_t, query_t if self_attention else kv.transpose(1, 0), self_attention)
        q = q.view(Tq, B * H, Dh).transpose(0, 1).view(B, H, Tq, Dh)
        k = k.view(S, B * H, Dh).transpose(0, 1).view(B, H, S, Dh)
        v = v.view(S, B * H, Dh).transpose(0, 1).view(B, H, S, Dh)
        mask = None
        if key_padding_mask is not None:
            if is_causal:
                raise ValueError('padding mask together with the causal hint is not supported')
            mask = torch.zeros_like(key_padding_mask, dtype=q.dtype).masked_fill(key_padding_mask, float('-inf'))
            mask = mask.view(B, 1, 1, S).expand(-1, H, -1, -1).reshape(B * H, 1, S).view(B, H, -1, S)
        out = sdpa(q, k, v, mask, self.dropout_p if self.training else 0.0, is_causal)
        out = out.permute(2, 0, 1, 3).contiguous().view(B * Tq, self.inner)
        out = linear(out, self.out_proj.weight, self.out_proj.bias)
        return out.view(Tq, B, -1).transpose(1, 0)


class ArchEncoderLayer(nn.Module):
    """Post-LN, ReLU layer; submodule names and registration order follow nn.TransformerEncoderLayer."""

    def __init__(self, d_model, heads, head_dim, ff, dropout):
        super().__init__()
        self.self_attn = IndependentHeadAttention(d_model, heads, head_dim, dropout)
        self.linear1 = nn.Linear(d_model, ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, key_padding_mask)))
        return self.norm2(x + self.dropout2(self.linear2(self.dropout(relu(self.linear1(x))))))


class ArchDecoderLayer(nn.Module):
    """Post-LN, ReLU layer; submodule names and registration order follow nn.TransformerDecoderLayer."""

    def __init__(self, d_model, heads, head_dim, ff, dropout):
        super().__init__()
        self.self_attn = IndependentHeadAttention(d_model, heads, head_dim, dropout)
        self.multihead_attn = IndependentHeadAttention(d_model, heads, head_dim, dropout)
        self.linear1 = nn.Linear(d_model, ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, memory_key_padding_mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, None, True)))  # causal; no target padding exists
        x = self.norm2(x + self.dropout2(self.multihead_attn(x, memory, memory_key_padding_mask)))
        return self.norm3(x + self.dropout3(self.linear2(self.dropout(relu(self.linear1(x))))))


def _clones(layer, count):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(count)])


def _reset(module):
    """nn.Transformer._reset_parameters: xavier-uniform every matrix in registration order."""
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class _Stack(nn.Module):
    def __init__(self, layer, count, norm):
        super().__init__()
        self.layers = _clones(layer, count)
        self.norm = norm


class _Core(nn.Module):
    """Plays the role of ``nn.Transformer``: children ``encoder`` / ``decoder``, each with ``layers`` and ``norm``."""

    def __init__(self, d_model, heads, head_dim, encoder_layers, decoder_layers, ff, dropout):
        super().__init__()
        self.encoder = _Stack(ArchEncoderLayer(d_model, heads, head_dim, ff, dropout), encoder_layers, nn.LayerNorm(d_model, eps=1e-5))
        self.decoder = _Stack(ArchDecoderLayer(d_model, heads, head_dim, ff, dropout), decoder_layers, nn.LayerNorm(d_model, eps=1e-5))
        _reset(self)


class TigerArchModel(nn.Module):
    """Same interface, child names, parameter order and state_dict keys as ``tiger.TigerModel``, with independent head dims.

    At anchor shape (1+1 layers, 4 heads x 32, d_model 128, dropout 0) it is a drop-in: same-seed initialisation and
    ``load_state_dict`` of anchor checkpoints work unchanged.
    """

    def __init__(self, vocab_size, d_model=128, heads=4, head_dim=32, encoder_layers=1, decoder_layers=1, ff=512, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position = nn.Embedding(POSITIONS, d_model)
        self.transformer = _Core(d_model, heads, head_dim, encoder_layers, decoder_layers, ff, dropout)
        self.output = nn.Linear(d_model, vocab_size)
        self.spec = {'d_model': d_model, 'heads': heads, 'head_dim': head_dim, 'inner_dim': heads * head_dim, 'encoder_layers': encoder_layers,
                     'decoder_layers': decoder_layers, 'ff': ff, 'dropout': dropout, 'vocab': vocab_size, 'positions': POSITIONS}

    def embed(self, tokens):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= POSITIONS:
            raise ValueError('invalid token sequence')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.embedding(tokens) * math.sqrt(self.d_model) + self.position(positions)[None]

    def encode(self, source):
        padding = source.eq(0)
        if padding.all(dim=1).any():
            raise ValueError('empty source')
        x = self.embed(source)
        for layer in self.transformer.encoder.layers:
            x = layer(x, padding)
        return self.transformer.encoder.norm(x), padding

    def decode(self, memory, padding, decoder):
        x = self.embed(decoder)
        for layer in self.transformer.decoder.layers:
            x = layer(x, memory, padding)
        return self.output(self.transformer.decoder.norm(x))

    def forward(self, source, decoder):
        memory, padding = self.encode(source)
        return self.decode(memory, padding, decoder)
