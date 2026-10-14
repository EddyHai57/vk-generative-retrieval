"""FEAT-02 P user-tower variants on the F3 anchor (new version; FEAT-01 modules are imported, never modified).

``feature`` switch (exactly one per run; item tower unchanged; every variant is F3 plus one more variable):
  f3             equivalence control for the FEAT-01 F3 model (tests only, never submitted)
  f3_masked      F3 whose two bucket embeddings are added on *supported* history events only (not counted against the quota)
  f3_f4_recency  F3 + P[position-from-last bucket] + G[week-gap bucket], both anchored on the LAST history event
Side tensor per history slot: [feedback_mask, timespent, duration, week, place, platform]; PAD slots are all zero.
A variant reads only its own columns; the others cannot change its output.
Bucket edges are train-only audit constants (scripts/feat02_audit_constants.py, pre-registered grids + 2% merge rule).
"""
import hashlib
import json

import torch

from .retrieval_features import RATIO_BUCKETS, TIMESPENT_BUCKETS, TIMESPENT_EDGES, ratio_bucket, timespent_bucket
from .retrieval_hp import embedding_report, init_embeddings
from .retrieval_models import TwoTower

FEATURES = ('f3', 'f3_masked', 'f3_f4_recency')
SIDE_WIDTH = 6
EMBEDDING_DIM = 64
MAX_WEEK = 24                                  # history events come from train weeks 0..24; validation queries have target week 25
POSITION_EDGES = (1, 2, 4, 8, 16, 32)          # back index 0 | 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-49 (0 = most recent event)
WEEKGAP_EDGES = (1, 2, 3, 5, 9)                # week(last event) - week: 0 | 1 | 2 | 3-4 | 5-8 | 9-24 (17+ merged: 1.94% < 2%)


def bucket_spec_sha256():
    """Digest of every hard-coded bucket constant; stored in the run binding so a changed constant breaks verify."""
    spec = {'timespent_edges': list(TIMESPENT_EDGES), 'ratio_buckets': RATIO_BUCKETS, 'position_edges': list(POSITION_EDGES),
            'weekgap_edges': list(WEEKGAP_EDGES), 'max_week': MAX_WEEK}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def supported_mask(supported, histories, valid):
    """History events whose item has a trained id embedding (train-defined ``supported`` buffer)."""
    return valid & supported[histories]


def position_bucket(valid):
    """Bucket id (1..7, 0 = PAD) of the back index: 0 for the last valid event, counting backwards."""
    rank = valid.long().cumsum(1) - 1
    back = valid.long().sum(1, keepdim=True) - 1 - rank
    edges = torch.tensor(POSITION_EDGES, dtype=back.dtype, device=back.device)
    return (torch.bucketize(back.clamp_min(0).contiguous(), edges, right=True) + 1) * valid


def week_gap(week, valid):
    """Week of the LAST valid history event minus the week of each event (weeks are non-decreasing on valid slots)."""
    last = (week * valid).amax(1, keepdim=True)
    return (last - week) * valid


def weekgap_bucket(gap, valid):
    """Bucket id (1..6, 0 = PAD) of a week gap."""
    edges = torch.tensor(WEEKGAP_EDGES, dtype=gap.dtype, device=gap.device)
    return (torch.bucketize(gap.contiguous(), edges, right=True) + 1) * valid


class Feat02IDMean(TwoTower):
    def __init__(self, feature, content, supported_indices, max_len):
        super().__init__('id_mean', content, supported_indices)
        if feature not in FEATURES:
            raise ValueError('unregistered FEAT-02 feature')
        object.__setattr__(self, 'feature', feature); object.__setattr__(self, 'max_len', max_len)

    def add_anchor_tables(self):
        """The two F3 tables, created in the FEAT-01 order so their draws equal FeatureIDMean('f3_watch_buckets')."""
        self.timespent_bucket = torch.nn.Embedding(TIMESPENT_BUCKETS + 1, EMBEDDING_DIM, padding_idx=0)
        self.ratio_bucket = torch.nn.Embedding(RATIO_BUCKETS + 1, EMBEDDING_DIM, padding_idx=0)
        return ('timespent_bucket', 'ratio_bucket')

    def add_variant_tables(self):
        """Tables of the one extra variable, created after the anchor tables are initialised."""
        if self.feature == 'f3_f4_recency':
            self.position_table = torch.nn.Embedding(len(POSITION_EDGES) + 2, EMBEDDING_DIM, padding_idx=0)
            self.weekgap_table = torch.nn.Embedding(len(WEEKGAP_EDGES) + 2, EMBEDDING_DIM, padding_idx=0)
            return ('position_table', 'weekgap_table')
        return ()

    def _check(self, histories, side):
        if histories.ndim != 2 or not 1 <= histories.shape[1] <= self.max_len or torch.any((histories != 0).sum(1) == 0):
            raise ValueError('invalid/empty history')
        if side is None or side.dtype != torch.long or side.shape != (*histories.shape, SIDE_WIDTH):
            raise ValueError('history side required')
        valid = histories != 0
        if torch.any(side[~valid] != 0):
            raise ValueError('PAD slot carries side values')
        mask, ts, dur = side[..., 0][valid], side[..., 1][valid], side[..., 2][valid]
        if torch.any((mask < 0) | (mask >= 1 << 6)) or torch.any((ts < 1) | (ts > 255)) or torch.any((dur < 1) | (dur > 255)):
            raise ValueError('side value outside registered range')
        if self.feature == 'f3_f4_recency':
            week = side[..., 3]
            if torch.any((week[valid] < 0) | (week[valid] > MAX_WEEK)):
                raise ValueError('history week outside train weeks 0..24')
            if torch.any(valid & (week != torch.cummax(week.masked_fill(~valid, -1), 1).values)):
                raise ValueError('history weeks must be non-decreasing')
        return valid

    def users(self, histories, side=None):
        valid = self._check(histories, side); encoded = self.items(histories)
        t = self.timespent_bucket(timespent_bucket(side[..., 1], valid))
        r = self.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
        if self.feature == 'f3_masked':
            keep = supported_mask(self.supported, histories, valid).unsqueeze(-1).to(encoded.dtype)
            encoded = encoded + t * keep + r * keep
        else:
            encoded = encoded + t + r
        if self.feature == 'f3_f4_recency':
            gap = week_gap(side[..., 3], valid)
            encoded = encoded + self.position_table(position_bucket(valid)) + self.weekgap_table(weekgap_bucket(gap, valid))
        return encoded.sum(1) / valid.sum(1, keepdim=True)


def _small_init(model, names):
    for name in names:
        child = model.get_submodule(name)
        with torch.no_grad():
            child.weight.normal_(0.0, 0.02)
            child.weight[child.padding_idx].zero_()


def make_feat02_model(feature, content, supported_indices, max_len):
    """Always INIT_EMB_N002. Draw order: item embedding, then T and R (as FEAT-01 F3), then the variant's own tables."""
    if max_len not in (20, 50, 100):
        raise ValueError('unregistered history length')
    model = Feat02IDMean(feature, content, supported_indices, max_len)
    init_embeddings(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _small_init(model, model.add_anchor_tables())
    _small_init(model, model.add_variant_tables())
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    object.__setattr__(model, 'init_report', {'init': 'small', 'std': 0.02, 'embeddings': embedding_report(model)})
    return model


def _report(rows):
    by_table = {name: {'total': n * EMBEDDING_DIM, 'trainable': (n - 1) * EMBEDDING_DIM} for name, n in rows.items()}
    return {'total': sum(v['total'] for v in by_table.values()), 'trainable_effective': sum(v['trainable'] for v in by_table.values()), 'by_table': by_table}


def parameter_report(model):
    """Parameters added on top of the item embedding. 'trainable' = total minus the rows fixed at zero (PAD).
    F3 convention: 896 total / 768 trainable (T + R); a variant's own tables are the by_table entries after those two."""
    rows = {name: module.weight.shape[0] for name, module in model.named_modules()
            if isinstance(module, torch.nn.Embedding) and name != 'embedding'}
    return _report(rows)


def expected_param_counts(feature):
    """Pure formula for ``parameter_report`` (no model needed)."""
    if feature not in FEATURES:
        raise ValueError('unregistered FEAT-02 feature')
    rows = {'timespent_bucket': TIMESPENT_BUCKETS + 1, 'ratio_bucket': RATIO_BUCKETS + 1}
    if feature == 'f3_f4_recency':
        rows.update(position_table=len(POSITION_EDGES) + 2, weekgap_table=len(WEEKGAP_EDGES) + 2)
    return _report(rows)
