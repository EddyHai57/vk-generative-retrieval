"""FEAT-03 P user-tower variants on the F3 anchor (new version; FEAT-02 modules are used through the module, never modified).

``variant`` switch (one per run; item tower unchanged; every variant is F3 plus one more variable):
  f4_control       equivalence fixture for FEAT-02 F4 (tests only, never submitted): F3 + P[position] + G[week gap]
  f3_pos_only      Q1a: F3 + P[position-from-last bucket] only (diagnostic, no quota)
  f3_gap_only      Q1b: F3 + G[week-gap bucket] only (diagnostic, no quota)
  f4_recency_pool  Q2 (not wired yet)
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
IMPLEMENTED = ('f4_control', 'f3_pos_only', 'f3_gap_only')       # Q2/Q3 raise ValueError in users() until their FEATs
UNUSED_TABLES = {'f3_pos_only': ('weekgap_table',), 'f3_gap_only': ('position_table',)}
INERT_ROWS3 = dict(f2.INERT_ROWS)                                 # own copy: f2.INERT_ROWS is never mutated


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
        return event_vectors(self, histories, side, valid).sum(1) / valid.sum(1, keepdim=True)


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
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    return model


def _report3(rows, inert):
    by_table = {name: {'total': n * f2.EMBEDDING_DIM, 'trainable': (n - inert.get(name, 1)) * f2.EMBEDDING_DIM} for name, n in rows.items()}
    return {'total': sum(v['total'] for v in by_table.values()), 'trainable_effective': sum(v['trainable'] for v in by_table.values()), 'by_table': by_table}


def parameter_report3(model):
    """Parameters added on top of the item embedding ('trainable' = total minus rows fixed at zero)."""
    rows = {name: module.weight.shape[0] for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)) and name != 'embedding'}
    return _report3(rows, INERT_ROWS3)


def expected_param_counts3(variant):
    """Pure formula for ``parameter_report3`` (no model needed)."""
    if variant not in IMPLEMENTED:
        raise ValueError('parameter counts not defined for this variant yet')
    rows = {'timespent_bucket': TIMESPENT_BUCKETS + 1, 'ratio_bucket': RATIO_BUCKETS + 1}
    if variant in ('f4_control', 'f3_pos_only'):
        rows['position_table'] = len(f2.POSITION_EDGES) + 2
    if variant in ('f4_control', 'f3_gap_only'):
        rows['weekgap_table'] = len(f2.WEEKGAP_EDGES) + 2
    return _report3(rows, INERT_ROWS3)
