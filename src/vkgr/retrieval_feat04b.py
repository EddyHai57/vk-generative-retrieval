"""FEAT-04b P user-tower / candidate-side variants on the F4-T010 anchor (new file; FEAT-02/FEAT-03 modules are used through the module, never modified).

``variant`` switch (one per run; the item tower is the anchor's unless a later variant says otherwise):
  f4_control          equivalence fixture for the anchor F4 (id_mean + F3 + F4), tests only, never a real run
  f4_recency_pool     chain A = Q2-T010: F4 event vectors pooled with 13 learnable recency logits (init exactly 0); delegated to the FEAT-03 Q2 functions
  f4_item_popularity  C (candidate-side popularity/freshness tables)   -- registered, implemented in a later feature
  f4_item_quality     D (candidate-side watch-ratio/like-rate tables)   -- registered, implemented in a later feature
  f4_recency_behavior E (A plus six behaviour-bit logits)               -- registered, implemented in a later feature
Seams are resolved at call time through the modules (``f3.<name>``, ``f2.<name>``): the mutation tests patch the module attribute, so this file never
writes ``from .retrieval_feat03 import x`` (K2). ``Feat04bIDMean`` bypasses ``Feat03IDMean.__init__`` (it only accepts FEATURES3) and calls the
FEAT-02 constructor with the FEAT-03 base feature string, exactly what ``Feat03IDMean`` does after its own check.
"""
import hashlib
import json

import torch

from . import retrieval_feat02 as f2
from . import retrieval_feat03 as f3
from .history_side import FEEDBACK_BITS
from .retrieval_features import ratio_bucket, timespent_bucket
from .retrieval_hp import init_embeddings
from .training import seed_all

CONTROL = 'f4_control'
A_FEATURE = 'f4_recency_pool'
FEATURES4B = ('f4_control', 'f4_recency_pool', 'f4_item_popularity', 'f4_item_quality', 'f4_recency_behavior')
IMPLEMENTED = (CONTROL, A_FEATURE)
REAL_FEATURES4B = ('f4_item_popularity', 'f4_item_quality', 'f4_recency_behavior')       # the variants this runner may submit as real runs (later features)
BEHAVIOR_BITS = FEEDBACK_BITS                                          # bit b of side[..., 0] is BEHAVIOR_BITS[b] (history_side.FEEDBACK_BITS order)


class Feat04bIDMean(f3.Feat03IDMean):
    def __init__(self, variant, content, supported_indices, max_len):
        if variant not in FEATURES4B:
            raise ValueError('unregistered FEAT-04b variant')
        f2.Feat02IDMean.__init__(self, f3.BASE_FEATURE, content, supported_indices, max_len)      # bypass Feat03IDMean.__init__ (FEATURES3 only)
        object.__setattr__(self, 'variant', variant)

    def history_items(self, indices):
        """Item vectors the user tower reads for history events (default = items; the C/D seam keeps this path untouched)."""
        return self.items(indices)

    def candidate_items(self, indices):
        """Item vectors used as candidates in train_step and in the evaluation export (default = items; C/D override this one only)."""
        return self.items(indices)

    def users(self, histories, side=None):
        if self.variant not in IMPLEMENTED:
            raise ValueError('variant not implemented yet')
        valid = self._check(histories, side)
        x = event_vectors4(self, histories, side, valid)
        if self.variant == A_FEATURE:
            pos_ids = f2.position_bucket(valid)
            gap_ids = f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid)
            logits = f3.recency_logits(self.recency_logit_pos, self.recency_logit_gap, pos_ids, gap_ids, valid)
            return f3.weighted_mean(x, f3.pooled_weights(logits, self._pool_mask(histories, valid)))
        return x.sum(1) / valid.sum(1, keepdim=True)


def event_vectors4(model, histories, side, valid):
    """Per-event vectors (B, W, 64); one left-to-right addition at a time, in the Feat02IDMean.users order (f4_control must be bit-equal):
    history_items, +T, +R, +P, +G (P first, then G; never P+G first). Every variant takes x_k from here."""
    x = model.history_items(histories)
    x = x + model.timespent_bucket(timespent_bucket(side[..., 1], valid))
    x = x + model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
    x = x + model.position_table(f2.position_bucket(valid))
    return x + model.weekgap_table(f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid))


def make_feat04b_model(variant, content, supported_indices, max_len, seed, stat_buckets=None):
    """Always init=small. Draw order (the FEAT-03 one): item, T, R, small-init T,R, P, G, small-init P,G; chain A then the 13 logits (exactly 0, no draw).

    Only f4_control and f4_recency_pool exist at this stage; ``stat_buckets`` (C/D) must be None for them.
    """
    if variant not in FEATURES4B:
        raise ValueError('unregistered FEAT-04b variant')
    if variant not in IMPLEMENTED:
        raise ValueError('variant not implemented yet')
    if max_len not in (20, 50, 100):
        raise ValueError('unregistered history length')
    if stat_buckets is not None:
        raise ValueError('stat_buckets belong to the C/D variants only')
    seed_all(seed)
    model = Feat04bIDMean(variant, content, supported_indices, max_len)
    init_embeddings(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    f2._small_init(model, model.add_anchor_tables())
    f3.draw_variant_tables(model)
    f3.drop_unused_tables(model, variant)                              # no table is dropped for these variants; it recomputes init_report
    if variant == A_FEATURE:
        f3.init_recency_logits(model)
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    return model


def parameter_report4b(model):
    """Parameters added on top of the item embedding (FEAT-03 convention; PAD rows are not trainable)."""
    return f3.parameter_report3(model)


def expected_param_counts4b(variant):
    """Pure formula for ``parameter_report4b``."""
    if variant not in FEATURES4B:
        raise ValueError('unregistered FEAT-04b variant')
    if variant not in IMPLEMENTED:
        raise ValueError('parameter counts not defined for this variant yet')
    return f3.expected_param_counts3(variant)


def bucket_spec_sha256(variant):
    """Per-variant digest of every constant the variant relies on (stored in that variant's run binding only).

    f4_control / chain A: {FEAT-03 spec digest, logit sizes [7, 6, 6], FEATURES4B} plus the variant name; C/D/E add their own constants in later features.
    """
    if variant not in FEATURES4B:
        raise ValueError('unregistered FEAT-04b variant')
    if variant not in IMPLEMENTED:
        raise ValueError('bucket spec not defined for this variant yet')
    spec = {'feat03_bucket_spec_sha256': f3.bucket_spec3_sha256(), 'features4b': list(FEATURES4B), 'variant': variant,
            'logit_sizes': [f3.RECENCY_PARAMS['recency_logit_pos'], f3.RECENCY_PARAMS['recency_logit_gap'], len(BEHAVIOR_BITS)]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
