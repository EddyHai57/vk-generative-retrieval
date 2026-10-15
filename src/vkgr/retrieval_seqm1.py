"""SEQ-M1 P model: SASRec-v3 = the F4-T010 anchor plus a gated 1-layer pre-LN Transformer branch (new file; frozen modules untouched).

Anchored arms (mean / last / attn / bi):  u = m + g * r
  m  anchor user vector (``F4IDMeanD.users``: mean of the per-event F3+F4 vectors), computed by the anchor code itself
  r  readout of the branch (causal or bidirectional Transformer over x_k = e_k + O[back index + 1]), g a scalar gate starting at exactly 0
  => at step 0 the user vector equals the anchor bit for bit.
Diagnostic arm ``pure``:  u = z_n (last valid position of the branch), no m, no g (the S1 reading under aligned inputs and training).
Draw order (pre-registered): seed -> item embedding -> T -> R -> small init -> P -> G -> small init (exactly make_php01_model) ->
O -> Transformer layer -> LN_f (weight 0.1) -> [attn: W_a, w=0] -> g=0. ``init_embeddings`` runs once, before the branch exists.
The module-level helpers (causal_mask, pool_*, combine, order_ids ...) are the seams the mutation tests patch.
"""
import hashlib
import json
from typing import NamedTuple, Optional

import torch

from .retrieval_feat02 import POSITION_EDGES, WEEKGAP_EDGES, position_bucket, week_gap, weekgap_bucket
from .retrieval_features import RATIO_BUCKETS, TIMESPENT_BUCKETS, ratio_bucket, timespent_bucket
from .retrieval_hp import embedding_report, init_embeddings
from .retrieval_php01 import F4IDMeanD, _draw_tables, _small_init_d, init_std, make_table

DIM = 64
HEADS = 4
FFN = 256
DROPOUT = .1
ORDER_ROWS = 51                       # back index 0..49 -> rows 1..50, row 0 = PAD
GATE_INIT = 0.                        # pre-registered: exactly zero (the zero-step equivalence depends on it)
FINAL_NORM_INIT = .1                  # pre-registered scale constant (design D2); not a tuning knob
# arm -> (mask, readout, anchored)
ARMS = {'mean': ('causal', 'mean', True), 'last': ('causal', 'last', True), 'attn': ('causal', 'attn', True),
        'bi': ('bi', 'mean', True), 'pure': ('causal', 'last', False)}
ARM_ORDER = tuple(ARMS)


class Parts(NamedTuple):
    e: torch.Tensor                   # [B, W, 64] anchor per-event vectors (0 at PAD)
    valid: torch.Tensor               # [B, W] bool
    z: torch.Tensor                   # [B, W, 64] branch outputs after LN_f (0 at PAD)
    r: torch.Tensor                   # [B, 64] readout
    m: Optional[torch.Tensor]         # [B, 64] anchor user vector (None for ``pure`` unless with_anchor=True)
    alpha: Optional[torch.Tensor]     # [B, W] attention weights (attn arm only)


# ---------------------------------------------------------------- seams (small pure functions; mutation tests patch these)

def make_layer():
    """One pre-LN encoder layer: 4 heads, d=64, FFN 256, ReLU, dropout 0.1, default PyTorch initialisation."""
    return torch.nn.TransformerEncoderLayer(DIM, HEADS, dim_feedforward=FFN, dropout=DROPOUT, activation='relu', batch_first=True, norm_first=True)


def causal_mask(n, device):
    """Boolean [n, n] attention mask, True = may not attend: position j sees keys k <= j only."""
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


def key_padding(valid):
    return ~valid


def assert_left_aligned(valid):
    n = valid.sum(1, keepdim=True)
    if not torch.equal(valid, torch.arange(valid.shape[1], device=valid.device).unsqueeze(0) < n):
        raise ValueError('valid history slots must be left-aligned')


def order_ids(valid):
    """Row of O per event: back index + 1 (the last event is 1), PAD = 0. Counted from the most recent event, never from the oldest."""
    rank = valid.long().cumsum(1) - 1
    back = valid.long().sum(1, keepdim=True) - 1 - rank
    return (back + 1) * valid


def branch_input(model, e, valid):
    return e + model.order_table(order_ids(valid))


def mask_pad(z, valid):
    """``where`` (not a multiplication): a NaN/Inf produced at a PAD position must not reach the readout."""
    return torch.where(valid.unsqueeze(-1), z, torch.zeros_like(z))


def pool_mean(z, valid):
    return z.sum(1) / valid.sum(1, keepdim=True)


def pool_last(z, valid):
    return z[torch.arange(len(z), device=z.device), valid.sum(1) - 1]


def attention_weights(scores, valid):
    return scores.masked_fill(~valid, float('-inf')).softmax(1)


def combine(m, g, r):
    return m + g * r


def _after_build(model):
    """Hook after the model is fully built (no-op). Nothing may re-draw parameters here."""


# ---------------------------------------------------------------- model

class SeqM1IDMean(F4IDMeanD):
    def __init__(self, content, supported_indices, max_len, arm):
        if type(arm) is not str or arm not in ARMS:
            raise ValueError('unregistered SEQ-M1 arm')
        super().__init__(content, supported_indices, max_len, DIM)
        mask, readout, anchored = ARMS[arm]
        object.__setattr__(self, 'arm', arm); object.__setattr__(self, 'mask_kind', mask)
        object.__setattr__(self, 'readout', readout); object.__setattr__(self, 'anchored', anchored)

    def add_branch(self):
        """O, the Transformer layer, LN_f, the attention pooling and the gate, created in the pre-registered draw order."""
        self.order_table = make_table(ORDER_ROWS, DIM)
        _small_init_d(self, ('order_table',), init_std(DIM))
        self.layer = make_layer()
        self.final_norm = torch.nn.LayerNorm(DIM)
        with torch.no_grad():
            self.final_norm.weight.fill_(FINAL_NORM_INIT)
        if self.readout == 'attn':
            self.attn_proj = torch.nn.Linear(DIM, DIM)
            self.attn_w = torch.nn.Parameter(torch.zeros(DIM))
        if self.anchored:
            self.gate = torch.nn.Parameter(torch.tensor(GATE_INIT))

    # -- anchor path
    def anchor_vector(self, histories, side):
        """m: the anchor's own users() (not recomputed here, so the g=0 equivalence is bit for bit)."""
        return super().users(histories, side)

    def event_vectors(self, histories, side, valid):
        """e_k = Emb*sup + T + R + P[back index bucket] + G[week-gap bucket], same operations and order as the anchor's users()."""
        encoded = self.items(histories)
        t = self.timespent_bucket(timespent_bucket(side[..., 1], valid))
        r = self.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
        encoded = encoded + t + r
        gap = week_gap(side[..., 3], valid)
        return encoded + self.position_table(position_bucket(valid)) + self.weekgap_table(weekgap_bucket(gap, valid))

    # -- branch path
    def _branch(self, histories, side, valid):
        e = self.event_vectors(histories, side, valid)
        x = branch_input(self, e, valid)
        mask = causal_mask(x.shape[1], x.device) if self.mask_kind == 'causal' else None
        f = self.layer(x, src_mask=mask, src_key_padding_mask=key_padding(valid))
        z = mask_pad(self.final_norm(f), valid)
        if not bool(torch.isfinite(z[valid]).all()):
            raise ValueError('nonfinite user encoding')
        return e, z

    def _readout(self, z, valid):
        if self.readout == 'mean':
            return pool_mean(z, valid), None
        if self.readout == 'last':
            return pool_last(z, valid), None
        scores = (torch.tanh(self.attn_proj(z)) * self.attn_w).sum(-1)
        alpha = attention_weights(scores, valid)
        return (alpha.unsqueeze(-1) * z).sum(1), alpha

    def parts(self, histories, side, with_anchor=False):
        valid = self._check(histories, side); assert_left_aligned(valid)
        e, z = self._branch(histories, side, valid)
        r, alpha = self._readout(z, valid)
        m = self.anchor_vector(histories, side) if self.anchored or with_anchor else None
        return Parts(e, valid, z, r, m, alpha)

    def users(self, histories, side=None):
        valid = self._check(histories, side); assert_left_aligned(valid)
        _, z = self._branch(histories, side, valid)
        r, _ = self._readout(z, valid)
        if not self.anchored:
            return r
        return combine(self.anchor_vector(histories, side), self.gate, r)

    def diagnose(self, histories, side):
        """Per-query diagnostics (1-D tensors) for the manifest; no_grad, call in eval mode."""
        with torch.no_grad():
            p = self.parts(histories, side, with_anchor=True)
            index = torch.arange(len(histories), device=histories.device)
            last_event = p.e[index, p.valid.sum(1) - 1]
            if self.anchored:
                branch = self.gate * p.r
                u = combine(p.m, self.gate, p.r)
                ratio = branch.norm(dim=1) / p.m.norm(dim=1).clamp_min(1e-12)
            else:
                u, ratio = p.r, None
            cosine = torch.nn.functional.cosine_similarity
            out = {'cos_u_m': cosine(u, p.m, dim=1, eps=1e-12), 'cos_u_last_event': cosine(u, last_event, dim=1, eps=1e-12)}
            if ratio is not None:
                out['branch_ratio'] = ratio
            if p.alpha is not None:
                n = p.valid.sum(1).float()
                entropy = -(p.alpha * torch.log(p.alpha.clamp_min(1e-30))).sum(1)
                out['attn_entropy_ratio'] = torch.where(n > 1, entropy / torch.log(n.clamp_min(2.)), torch.ones_like(n))
                out['attn_last_weight'] = p.alpha[index, p.valid.sum(1) - 1]
            return out


def make_seqm1_model(content, supported_indices, max_len, arm):
    """Anchor draws exactly as ``make_php01_model(dim=64)`` (init_embeddings runs ONCE, before the branch), then the branch draws."""
    if type(max_len) is not int or max_len != 50:
        raise ValueError('SEQ-M1 is HIST_V2_OBS_L50 only')
    model = SeqM1IDMean(content, supported_indices, max_len, arm)
    std = init_std(DIM)
    init_embeddings(model, std)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _draw_tables(model, std)
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    object.__setattr__(model, 'init_report', {'init': 'small', 'std': std, 'embeddings': embedding_report(model)})
    model.add_branch()
    _after_build(model)
    return model


# ---------------------------------------------------------------- reports

def _numel(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


def parameter_report_seqm1(model):
    """numel per group (module types are NOT counted: the Transformer's Linear layers are not tables)."""
    value = {'item_embedding': model.embedding.weight.numel(),
             'anchor_tables': _numel(model.timespent_bucket, model.ratio_bucket, model.position_table, model.weekgap_table),
             'order_table': _numel(model.order_table), 'layer': _numel(model.layer), 'final_norm': _numel(model.final_norm)}
    if model.readout == 'attn':
        value['attn_pool'] = _numel(model.attn_proj) + model.attn_w.numel()
    if model.anchored:
        value['gate'] = model.gate.numel()
    value['total'] = sum(value.values())
    return value


def expected_param_counts_seqm1(arm, catalog_rows):
    """Pure formula for ``parameter_report_seqm1`` (design 3.4)."""
    if type(arm) is not str or arm not in ARMS:
        raise ValueError('unregistered SEQ-M1 arm')
    layer = 2 * 2 * DIM + (3 * DIM * DIM + 3 * DIM) + (DIM * DIM + DIM) + (DIM * FFN + FFN) + (FFN * DIM + DIM)
    value = {'item_embedding': catalog_rows * DIM,
             'anchor_tables': (TIMESPENT_BUCKETS + 1 + RATIO_BUCKETS + 1 + len(POSITION_EDGES) + 2 + len(WEEKGAP_EDGES) + 2) * DIM,
             'order_table': ORDER_ROWS * DIM, 'layer': layer, 'final_norm': 2 * DIM}
    mask, readout, anchored = ARMS[arm]
    if readout == 'attn':
        value['attn_pool'] = DIM * DIM + DIM + DIM
    if anchored:
        value['gate'] = 1
    value['total'] = sum(value.values())
    return value


def branch_spec(model):
    """Structure read off the INSTANTIATED modules (so editing a live layer changes the digest)."""
    layer = model.layer
    if not isinstance(layer, torch.nn.TransformerEncoderLayer):
        raise ValueError('branch must be one TransformerEncoderLayer')
    return {'arm': model.arm, 'mask': model.mask_kind, 'readout': model.readout, 'anchored': model.anchored, 'num_layers': 1,
            'd_model': layer.self_attn.embed_dim, 'heads': layer.self_attn.num_heads, 'ffn': layer.linear1.out_features,
            'dropout': [layer.dropout.p, layer.dropout1.p, layer.dropout2.p, layer.self_attn.dropout],
            'activation': getattr(layer.activation, '__name__', type(layer.activation).__name__), 'norm_first': layer.norm_first,
            'ln_eps': [layer.norm1.eps, layer.norm2.eps, model.final_norm.eps],
            'final_norm_weight_values': sorted(set(model.final_norm.weight.tolist())), 'final_norm_bias_values': sorted(set(model.final_norm.bias.tolist())),
            'gate_init': float(model.gate.detach()) if model.anchored else None, 'order_rows': model.order_table.num_embeddings,
            'attn_dim': model.attn_proj.out_features if model.readout == 'attn' else None,
            'attn_w_zero': bool((model.attn_w == 0).all()) if model.readout == 'attn' else None}


def branch_spec_sha256(model):
    return hashlib.sha256(json.dumps(branch_spec(model), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
