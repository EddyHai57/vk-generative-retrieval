"""FEAT-01 P user-tower features over HIST_V2_SIDE_V1 (history-event side fields).

``feature`` switch (exactly one per run; item tower unchanged):
  none              plain id_mean (equal to the frozen small-init id_mean; equivalence control)
  f1_ratio_pool     F1: weighted mean, w = clip(timespent/duration, 0.1, 2.0); zero new parameters
  f2_feedback       F2: event vector = item emb + sum of 6 feedback embeddings (multi-hot)
  f3_watch_buckets  F3: event vector = item emb + timespent-bucket emb + watch-ratio-bucket emb
  c0_unsupported_const  split control (not counted): one shared learned vector added to every *unsupported*
                    history event only; supported events and all side values are ignored. Measures the
                    "unsupported history becomes visible" part of F2/F3 without any feedback/watch information.
Side tensor per history slot: [feedback_mask, timespent, duration]; PAD slots are all zero.
Bucket edges are frozen in docs/superpowers/specs/2026-10-14-feat01-card.md (train-only audit).
"""
import torch

from .retrieval_hp import embedding_report, init_embeddings
from .retrieval_models import TwoTower

FEATURES = ('none', 'f1_ratio_pool', 'f2_feedback', 'f3_watch_buckets', 'c0_unsupported_const')
FEEDBACK_DIM = 6
RATIO_WEIGHT_MIN, RATIO_WEIGHT_MAX = 0.1, 2.0
TIMESPENT_EDGES = (4, 10, 31, 61, 135, 255)  # buckets 1-3 | 4-9 | 10-30 | 31-60 | 61-134 | 135-254 | 255
TIMESPENT_BUCKETS = len(TIMESPENT_EDGES) + 1   # ids 1..7, 0 = PAD
RATIO_BUCKETS = 5                              # ids 1..5: <0.1 | [0.1,0.5) | [0.5,1) | =1 | >1, 0 = PAD


def ratio_weight(timespent, duration, valid):
    ratio = timespent.float() / duration.clamp_min(1).float()
    return ratio.clamp(RATIO_WEIGHT_MIN, RATIO_WEIGHT_MAX) * valid


def timespent_bucket(timespent, valid):
    edges = torch.tensor(TIMESPENT_EDGES, dtype=timespent.dtype, device=timespent.device)
    return (torch.bucketize(timespent.contiguous(), edges, right=True) + 1) * valid


def ratio_bucket(timespent, duration, valid):
    """Integer comparisons only (no float boundary drift)."""
    bucket = torch.full_like(timespent, 5)
    bucket = torch.where(timespent == duration, torch.full_like(bucket, 4), bucket)
    bucket = torch.where(timespent < duration, torch.full_like(bucket, 3), bucket)
    bucket = torch.where(2 * timespent < duration, torch.full_like(bucket, 2), bucket)
    bucket = torch.where(10 * timespent < duration, torch.full_like(bucket, 1), bucket)
    return bucket * valid


class FeatureIDMean(TwoTower):
    def __init__(self, feature, content, supported_indices, max_len):
        super().__init__('id_mean', content, supported_indices)
        if feature not in FEATURES:
            raise ValueError('unregistered FEAT-01 feature')
        object.__setattr__(self, 'feature', feature); object.__setattr__(self, 'max_len', max_len)

    def add_feature_tables(self):
        """Created after the item embedding is initialised so the item table matches the anchor bit for bit."""
        feature = self.feature
        if feature == 'f2_feedback':
            self.feedback = torch.nn.Embedding(FEEDBACK_DIM, 64)
        if feature == 'f3_watch_buckets':
            self.timespent_bucket = torch.nn.Embedding(TIMESPENT_BUCKETS + 1, 64, padding_idx=0)
            self.ratio_bucket = torch.nn.Embedding(RATIO_BUCKETS + 1, 64, padding_idx=0)
        if feature == 'c0_unsupported_const':
            self.unsupported_const = torch.nn.Embedding(1, 64)

    def _check(self, histories, side):
        if histories.ndim != 2 or not 1 <= histories.shape[1] <= self.max_len or torch.any((histories != 0).sum(1) == 0):
            raise ValueError('invalid/empty history')
        if side is None or side.dtype != torch.long or side.shape != (*histories.shape, 3):
            raise ValueError('history side required')
        valid = histories != 0
        if torch.any(side[~valid] != 0):
            raise ValueError('PAD slot carries side values')
        mask, ts, dur = side[..., 0][valid], side[..., 1][valid], side[..., 2][valid]
        if torch.any((mask < 0) | (mask >= 1 << FEEDBACK_DIM)) or torch.any((ts < 1) | (ts > 255)) or torch.any((dur < 1) | (dur > 255)):
            raise ValueError('side value outside registered range')
        return valid

    def users(self, histories, side=None):
        if self.feature == 'none' and side is None:
            side = torch.zeros(*histories.shape, 3, dtype=torch.long, device=histories.device)
            side[..., 1:] = (histories != 0).long().unsqueeze(-1)
        valid = self._check(histories, side); encoded = self.items(histories)
        if self.feature == 'f1_ratio_pool':
            weights = ratio_weight(side[..., 1], side[..., 2], valid)
            return (encoded * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True)
        if self.feature == 'f2_feedback':
            bits = torch.stack([(side[..., 0] >> b) & 1 for b in range(FEEDBACK_DIM)], -1).to(encoded.dtype) * valid.unsqueeze(-1)
            encoded = encoded + bits @ self.feedback.weight
        if self.feature == 'f3_watch_buckets':
            encoded = encoded + self.timespent_bucket(timespent_bucket(side[..., 1], valid)) + self.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
        if self.feature == 'c0_unsupported_const':
            unsupported = (valid & ~self.supported[histories]).unsqueeze(-1).to(encoded.dtype)
            encoded = encoded + unsupported * self.unsupported_const.weight[0]
        return encoded.sum(1) / valid.sum(1, keepdim=True)


def make_feature_model(feature, content, supported_indices, max_len):
    """Always INIT_EMB_N002; item embedding is drawn first, so it equals the anchor's under the same seed."""
    if max_len not in (20, 50, 100):
        raise ValueError('unregistered history length')
    model = FeatureIDMean(feature, content, supported_indices, max_len)
    init_embeddings(model)                       # identical draw sequence to retrieval_hp.make_model('id_mean','small')
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.add_feature_tables()
    for name, child in model.named_modules():
        if isinstance(child, torch.nn.Embedding) and name != 'embedding':
            with torch.no_grad():
                child.weight.normal_(0.0, 0.02)
                if child.padding_idx is not None:
                    child.weight[child.padding_idx].zero_()
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    report = embedding_report(model)
    object.__setattr__(model, 'init_report', {'init': 'small', 'std': 0.02, 'embeddings': report})
    return model
