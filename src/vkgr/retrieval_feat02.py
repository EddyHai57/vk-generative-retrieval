"""FEAT-02 P user-tower variants on the F3 anchor (new version; FEAT-01 modules are imported, never modified).

``feature`` switch (exactly one per run; item tower unchanged; every variant is F3 plus one more variable):
  f3             equivalence control for the FEAT-01 F3 model (tests only, never submitted)
  f3_masked      F3 whose two bucket embeddings are added on *supported* history events only (not counted against the quota)
  f3_f4_recency  F3 + P[position-from-last bucket] + G[week-gap bucket], both anchored on the LAST history event
  f3_f5_scene    F3 + Place[place id] + Platform[platform id] of each history event's own row (agent excluded)
  f3_f6_author_content  F3 + shared Author embedding + learned Linear(64,64) projection of the official catalog content vector,
                 used in BOTH the history encoding and the item tower: v(i) = emb(i)*supported(i) + Author[author(i)] + W c(i),
                 so an unsupported item (id part masked to zero) still gets Author + W c instead of a zero vector
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

FEATURES = ('f3', 'f3_masked', 'f3_f4_recency', 'f3_f5_scene', 'f3_f6_author_content')
F6_FEATURE = 'f3_f6_author_content'
SIDE_WIDTH = 6
EMBEDDING_DIM = 64
MAX_WEEK = 24                                  # history events come from train weeks 0..24; validation queries have target week 25
POSITION_EDGES = (1, 2, 4, 8, 16, 32)          # back index 0 | 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-49 (0 = most recent event)
WEEKGAP_EDGES = (1, 2, 3, 5, 9)                # week(last event) - week: 0 | 1 | 2 | 3-4 | 5-8 | 9-24 (17+ merged: 1.94% < 2%)
PLACE_VALUES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)   # train weeks 00-24, ascending
PLATFORM_VALUES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
AUTHOR_VOCAB_SIZE = 11133          # authors of SUPPORTED catalog items (train-only audit); author table has AUTHOR_VOCAB_SIZE + 2 rows
AUTHOR_VOCAB_SHA256 = '327ee976f2c8fff4e40267546bc2dd16818ad4d59e97da34106bb515dd1626bc'   # sha256 of the compact ascending JSON list
# PAD + OOV rows contribute exactly zero; every other table has only PAD; the content projection has no inert row
INERT_ROWS = {'place_table': 2, 'platform_table': 2, 'author_table': 2, 'content_projection': 0}


def bucket_spec_sha256():
    """Digest of every hard-coded bucket constant; stored in the run binding so a changed constant breaks verify."""
    spec = {'timespent_edges': list(TIMESPENT_EDGES), 'ratio_buckets': RATIO_BUCKETS, 'position_edges': list(POSITION_EDGES),
            'weekgap_edges': list(WEEKGAP_EDGES), 'max_week': MAX_WEEK,
            'place_values': list(PLACE_VALUES), 'platform_values': list(PLATFORM_VALUES),
            'author_vocab_size': AUTHOR_VOCAB_SIZE, 'author_vocab_sha256': AUTHOR_VOCAB_SHA256}
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


def scene_columns(side):
    """Raw (place, platform) columns of the 6-column side tensor."""
    return side[..., 4], side[..., 5]


def category_ids(raw, table, valid):
    """Id of a raw uint8 value: 0 = PAD slot, 1 = value not in the train-defined ``table`` (OOV), else table index + 2."""
    lookup = torch.ones(256, dtype=torch.long, device=raw.device)
    lookup[torch.tensor(table, dtype=torch.long, device=raw.device)] = torch.arange(len(table), dtype=torch.long, device=raw.device) + 2
    return lookup[raw.clamp(0, 255)] * valid


def project_content(projection, content):
    """The learned Linear(64, 64, bias=False) applied to the prepared official catalog embedding rows (F6)."""
    return projection(content)


class Feat02IDMean(TwoTower):
    def __init__(self, feature, content, supported_indices, max_len, author_index=None, vocab_size=None):
        super().__init__('id_mean', content, supported_indices)
        if feature not in FEATURES:
            raise ValueError('unregistered FEAT-02 feature')
        object.__setattr__(self, 'feature', feature); object.__setattr__(self, 'max_len', max_len)
        if feature != F6_FEATURE:
            if author_index is not None or vocab_size is not None:
                raise ValueError('author_index/vocab_size belong to the F6 variant only')
            return
        vocab_size = AUTHOR_VOCAB_SIZE if vocab_size is None else vocab_size
        if type(vocab_size) is not int or vocab_size < 0:
            raise ValueError('invalid author vocabulary size')
        if not isinstance(author_index, torch.Tensor) or author_index.dtype != torch.long or author_index.shape != (len(content),) \
                or int(author_index[0]) != 0 or int(author_index.min()) < 0 or int(author_index.max()) > vocab_size + 1:
            raise ValueError('author_index must be a long vector over the catalog index space with PAD at row 0 and rows in [0, V+1]')
        object.__setattr__(self, 'vocab_size', vocab_size)
        self.register_buffer('content', content.detach().clone())          # the prepared official catalog embedding, immutable on load
        self.register_buffer('author_index', author_index.detach().clone())

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
        if self.feature == 'f3_f5_scene':
            self.place_table = torch.nn.Embedding(len(PLACE_VALUES) + 2, EMBEDDING_DIM, padding_idx=0)
            self.platform_table = torch.nn.Embedding(len(PLATFORM_VALUES) + 2, EMBEDDING_DIM, padding_idx=0)
            return ('place_table', 'platform_table')
        if self.feature == F6_FEATURE:
            self.author_table = torch.nn.Embedding(self.vocab_size + 2, EMBEDDING_DIM, padding_idx=0)
            self.content_projection = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
            return ('author_table',)          # the Linear weight is drawn right after the embeddings in make_feat02_model
        return ()

    def items(self, indices):
        """Item tower. F6: emb*supported + Author[author_index] + W c, so an unsupported item is Author + W c (or W c with an OOV author)."""
        encoded = super().items(indices)
        if self.feature != F6_FEATURE:
            return encoded
        rows = self.author_index[indices]
        author = self.author_table(rows) * (rows >= 2).unsqueeze(-1)
        return encoded + author + project_content(self.content_projection, self.content[indices])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if hasattr(self, 'author_index'):
            expected = self.author_index; actual = state_dict.get('author_index')
            if not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype or not torch.equal(actual.cpu(), expected.cpu()):
                raise ValueError('checkpoint immutable author_index mismatch')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

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
        if self.feature == 'f3_f5_scene':
            for column in scene_columns(side):
                if torch.any((column[valid] < 0) | (column[valid] > 255)):
                    raise ValueError('place/platform outside uint8 [0,255]')
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
        if self.feature == 'f3_f5_scene':
            place, platform = scene_columns(side)
            place_ids, platform_ids = category_ids(place, PLACE_VALUES, valid), category_ids(platform, PLATFORM_VALUES, valid)
            encoded = encoded + self.place_table(place_ids) * (place_ids >= 2).unsqueeze(-1) + self.platform_table(platform_ids) * (platform_ids >= 2).unsqueeze(-1)
        return encoded.sum(1) / valid.sum(1, keepdim=True)


def _small_init(model, names):
    for name in names:
        child = model.get_submodule(name)
        with torch.no_grad():
            child.weight.normal_(0.0, 0.02)
            child.weight[child.padding_idx].zero_()


def make_feat02_model(feature, content, supported_indices, max_len, author_index=None, vocab_size=None):
    """Always INIT_EMB_N002. Draw order: item embedding, then T and R (as FEAT-01 F3), then the variant's own tables
    (F6: author embedding, then the N(0, 0.02) content projection weight; vocab_size defaults to AUTHOR_VOCAB_SIZE)."""
    if max_len not in (20, 50, 100):
        raise ValueError('unregistered history length')
    model = Feat02IDMean(feature, content, supported_indices, max_len, author_index, vocab_size)
    init_embeddings(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _small_init(model, model.add_anchor_tables())
    _small_init(model, model.add_variant_tables())
    if feature == F6_FEATURE:
        with torch.no_grad():
            model.content_projection.weight.normal_(0.0, 0.02)       # not the default kaiming draw: init=small keeps projections at embedding scale
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    object.__setattr__(model, 'init_report', {'init': 'small', 'std': 0.02, 'embeddings': embedding_report(model)})
    return model


def _report(rows):
    by_table = {name: {'total': n * EMBEDDING_DIM, 'trainable': (n - INERT_ROWS.get(name, 1)) * EMBEDDING_DIM} for name, n in rows.items()}
    return {'total': sum(v['total'] for v in by_table.values()), 'trainable_effective': sum(v['trainable'] for v in by_table.values()), 'by_table': by_table}


def parameter_report(model):
    """Parameters added on top of the item embedding. 'trainable' = total minus the rows fixed at zero (PAD; F5 tables also OOV).
    F3 convention: 896 total / 768 trainable (T + R); a variant's own tables are the by_table entries after those two."""
    rows = {name: module.weight.shape[0] for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)) and name != 'embedding'}
    return _report(rows)


def expected_param_counts(feature, vocab_size=None):
    """Pure formula for ``parameter_report`` (no model needed); F6 uses AUTHOR_VOCAB_SIZE unless ``vocab_size`` is given."""
    if feature not in FEATURES:
        raise ValueError('unregistered FEAT-02 feature')
    rows = {'timespent_bucket': TIMESPENT_BUCKETS + 1, 'ratio_bucket': RATIO_BUCKETS + 1}
    if feature == F6_FEATURE:
        rows.update(author_table=(AUTHOR_VOCAB_SIZE if vocab_size is None else vocab_size) + 2, content_projection=EMBEDDING_DIM)
    if feature == 'f3_f4_recency':
        rows.update(position_table=len(POSITION_EDGES) + 2, weekgap_table=len(WEEKGAP_EDGES) + 2)
    if feature == 'f3_f5_scene':
        rows.update(place_table=len(PLACE_VALUES) + 2, platform_table=len(PLATFORM_VALUES) + 2)
    return _report(rows)
