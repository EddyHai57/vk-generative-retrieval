"""FEAT-03 P user-tower variants on the F3 anchor (new version; FEAT-02 modules are used through the module, never modified).

``variant`` switch (one per run; item tower unchanged; every variant is F3 plus one more variable):
  f4_control       equivalence fixture for FEAT-02 F4 (tests only, never submitted): F3 + P[position] + G[week gap]
  f3_pos_only      Q1a: F3 + P[position-from-last bucket] only (diagnostic, no quota)
  f3_gap_only      Q1b: F3 + G[week-gap bucket] only (diagnostic, no quota)
  f4_recency_pool  Q2: F4 event vectors, pooled with 13 learnable recency logits (7 position + 6 week-gap, init exactly 0) as a normalised weighted mean
  f3_user_activity Q3 (not wired yet)
Every variant constructs the model with the FEAT-02 feature string ``f3_f4_recency`` (K3), so the item, T, R, P, G tables are drawn
in the F4 order from the same seed; the table(s) a variant does not use are dropped AFTER initialisation. The variant name lives in
``model.variant``. Seams (K2/K4b) are module-level functions resolved at call time; FEAT-02 helpers are called as ``f2.<name>``.
"""
import torch

from . import retrieval_feat02 as f2
from .retrieval_features import RATIO_BUCKETS, TIMESPENT_BUCKETS, ratio_bucket, timespent_bucket
from .retrieval_hp import embedding_report, init_embeddings
from .training import seed_all

FEATURES3 = ('f4_control', 'f3_pos_only', 'f3_gap_only', 'f4_recency_pool', 'f3_user_activity')
BASE_FEATURE = 'f3_f4_recency'
IMPLEMENTED = ('f4_control', 'f3_pos_only', 'f3_gap_only', 'f4_recency_pool')       # Q3 raises ValueError in users() until its FEAT
UNUSED_TABLES = {'f3_pos_only': ('weekgap_table',), 'f3_gap_only': ('position_table',)}
INERT_ROWS3 = dict(f2.INERT_ROWS)                                 # own copy: f2.INERT_ROWS is never mutated
RECENCY_PARAMS = {'recency_logit_pos': len(f2.POSITION_EDGES) + 1, 'recency_logit_gap': len(f2.WEEKGAP_EDGES) + 1}      # Q2 scalars: (7,) and (6,)


class Feat03IDMean(f2.Feat02IDMean):
    def __init__(self, variant, content, supported_indices, max_len):
        if variant not in FEATURES3:
            raise ValueError('unregistered FEAT-03 variant')
        super().__init__(BASE_FEATURE, content, supported_indices, max_len)       # self.feature stays 'f3_f4_recency' for all variants
        object.__setattr__(self, 'variant', variant)

    def users(self, histories, side=None):
        if self.variant not in IMPLEMENTED:
            raise ValueError('variant not implemented yet')
        valid = self._check(histories, side)
        if self.variant == 'f4_recency_pool':
            pos_ids = f2.position_bucket(valid)
            gap_ids = f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid)
            logits = recency_logits(self.recency_logit_pos, self.recency_logit_gap, pos_ids, gap_ids, valid)
            return weighted_mean(event_vectors(self, histories, side, valid), pooled_weights(logits, self._pool_mask(histories, valid)))
        return event_vectors(self, histories, side, valid).sum(1) / valid.sum(1, keepdim=True)

    def _pool_mask(self, histories, valid):
        """Identity seam: every valid event, supported or not, takes part in the pooling (unsupported events are never dropped)."""
        return valid


def event_vectors(model, histories, side, valid):
    """Per-event vectors (B, W, 64); one left-to-right addition at a time, in the Feat02IDMean.users order (f4_control must be bit-equal)."""
    x = model.items(histories)
    x = x + model.timespent_bucket(timespent_bucket(side[..., 1], valid))
    x = x + model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
    if model.variant == 'f3_pos_only':
        return x + model.position_table(f2.position_bucket(valid))
    if model.variant == 'f3_gap_only':
        return x + model.weekgap_table(f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid))
    x = x + model.position_table(f2.position_bucket(valid))              # f4_control: P first, then G (never P+G first)
    return x + model.weekgap_table(f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid))


def init_recency_logits(model):
    """Q2: the 13 scalar logits as nn.Parameter, exactly 0 (so the start equals F4 bit for bit). No random draw."""
    model.recency_logit_pos = torch.nn.Parameter(torch.zeros(RECENCY_PARAMS['recency_logit_pos']))
    model.recency_logit_gap = torch.nn.Parameter(torch.zeros(RECENCY_PARAMS['recency_logit_gap']))


def recency_logits(logit_pos, logit_gap, pos_ids, gap_ids, valid):
    """(B, W) finite logits u[pb - 1] + v[gb - 1]; PAD (id 0) is clamped to index 0 and masked later by ``pooled_weights``."""
    return logit_pos[(pos_ids - 1).clamp_min(0)] + logit_gap[(gap_ids - 1).clamp_min(0)]


def pooled_weights(logits, valid):
    """(B, W) weights exp(l - max valid l), PAD exactly 0. Double where: the value fed to exp at a PAD slot is m - m = 0, so exp cannot overflow
    and the backward pass has no 0 * inf. ``m`` is detached (the normalised result is shift invariant)."""
    m = logits.masked_fill(~valid, float('-inf')).amax(1, keepdim=True).detach()
    safe = torch.where(valid, logits, m)
    return torch.where(valid, torch.exp(safe - m), torch.zeros_like(safe))


def weighted_mean(x, w):
    """(B, 64) = sum_k w_k x_k / sum_k w_k (same reduction as the uniform mean when every valid w is 1.0)."""
    return (x * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True)


def draw_variant_tables(model):
    """Construct P then G (each construction consumes a default draw), then N(0, 0.02) in that order and clear the PAD rows."""
    f2._small_init(model, model.add_variant_tables())


def drop_unused_tables(model, variant):
    """Drop the tables a variant does not use (after initialisation, so the random stream matches F4) and recompute init_report."""
    for name in UNUSED_TABLES.get(variant, ()):
        if hasattr(model, name):
            delattr(model, name)
    object.__setattr__(model, 'init_report', {'init': 'small', 'std': 0.02, 'embeddings': embedding_report(model)})


def make_feat03_model(variant, content, supported_indices, max_len, seed):
    """Always init=small. Draw order (K3): item, T, R, small-init T,R, P, G, small-init P,G; then unused tables are dropped."""
    if variant not in FEATURES3:
        raise ValueError('unregistered FEAT-03 variant')
    if max_len not in (20, 50, 100):
        raise ValueError('unregistered history length')
    seed_all(seed)
    model = Feat03IDMean(variant, content, supported_indices, max_len)
    init_embeddings(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    f2._small_init(model, model.add_anchor_tables())
    draw_variant_tables(model)
    drop_unused_tables(model, variant)
    if variant == 'f4_recency_pool':
        init_recency_logits(model)
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    return model


def _report3(rows, inert, scalars=None):
    by_table = {name: {'total': n * f2.EMBEDDING_DIM, 'trainable': (n - inert.get(name, 1)) * f2.EMBEDDING_DIM} for name, n in rows.items()}
    by_table.update({name: {'total': n, 'trainable': n} for name, n in (scalars or {}).items()})      # Q2 nn.Parameter scalars: no PAD row
    return {'total': sum(v['total'] for v in by_table.values()), 'trainable_effective': sum(v['trainable'] for v in by_table.values()), 'by_table': by_table}


def parameter_report3(model):
    """Parameters added on top of the item embedding ('trainable' = total minus rows fixed at zero)."""
    rows = {name: module.weight.shape[0] for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)) and name != 'embedding'}
    scalars = {name: getattr(model, name).numel() for name in RECENCY_PARAMS if hasattr(model, name)}
    return _report3(rows, INERT_ROWS3, scalars)


def expected_param_counts3(variant):
    """Pure formula for ``parameter_report3`` (no model needed)."""
    if variant not in IMPLEMENTED:
        raise ValueError('parameter counts not defined for this variant yet')
    rows = {'timespent_bucket': TIMESPENT_BUCKETS + 1, 'ratio_bucket': RATIO_BUCKETS + 1}
    if variant in ('f4_control', 'f3_pos_only', 'f4_recency_pool'):
        rows['position_table'] = len(f2.POSITION_EDGES) + 2
    if variant in ('f4_control', 'f3_gap_only', 'f4_recency_pool'):
        rows['weekgap_table'] = len(f2.WEEKGAP_EDGES) + 2
    return _report3(rows, INERT_ROWS3, RECENCY_PARAMS if variant == 'f4_recency_pool' else None)
