"""FEAT-04a P user-tower variants on the F4-T010 anchor (new file; FEAT-02/FEAT-03 modules are used through the module, never modified).

``variant`` switch (one per run; the item tower is the anchor's in every variant):
  f4_control              equivalence fixture for the anchor F4 (id_mean + F3 + F4), tests only, never a real run
  f4_recency_pool         A = Q2-T010: F4 event vectors pooled with 13 learnable recency logits (init exactly 0); delegated to FEAT-03 Q2
  f4_hist_author_content  B = F6-hist: every HISTORY event vector gets Author[author(item)] (OOV row contributes exactly 0) + W c(item);
                          the item tower (candidate side) is untouched, so unsupported candidates stay exactly 0
All seams are resolved at call time through the module (``f3.<name>``, ``f2.<name>``): the mutation tests patch the module attribute,
so this file never writes ``from .retrieval_feat03 import x`` (K2).
"""
import hashlib
import json

import torch

from . import retrieval_feat02 as f2
from . import retrieval_feat03 as f3
from . import retrieval_features as rf

CONTROL = 'f4_control'
A_FEATURE = 'f4_recency_pool'
B_FEATURE = 'f4_hist_author_content'
FEATURES4A = (CONTROL, A_FEATURE, B_FEATURE)
REAL_FEATURES4A = (A_FEATURE, B_FEATURE)


class Feat04aIDMean(f3.Feat03IDMean):
    """B model. ``Feat03IDMean.__init__`` only accepts the FEAT-03 variants and ``Feat02IDMean`` refuses author data for a non-F6 feature,
    so the FEAT-02 constructor is called with the F4 feature string and the immutable author/content buffers are registered here."""

    def __init__(self, content, supported_indices, max_len, author_index, vocab_size):
        f2.Feat02IDMean.__init__(self, f3.BASE_FEATURE, content, supported_indices, max_len)
        if type(vocab_size) is not int or vocab_size < 0:
            raise ValueError('invalid author vocabulary size')
        if not isinstance(author_index, torch.Tensor) or author_index.dtype != torch.long or author_index.shape != (len(content),) \
                or int(author_index[0]) != 0 or int(author_index.min()) < 0 or int(author_index.max()) > vocab_size + 1:
            raise ValueError('author_index must be a long vector over the catalog index space with PAD at row 0 and rows in [0, V+1]')
        object.__setattr__(self, 'variant', B_FEATURE); object.__setattr__(self, 'vocab_size', vocab_size)
        self.register_buffer('content', content.detach().clone())          # the prepared official catalog embedding, immutable on load
        self.register_buffer('author_index', author_index.detach().clone())

    def users(self, histories, side=None):
        """Uniform mean of the B event vectors (no recency weights). Does not depend on ``Feat03IDMean.users`` (its event vectors are not redirectable)."""
        valid = self._check(histories, side)
        return event_vectors4a(self, histories, side, valid).sum(1) / valid.sum(1, keepdim=True)


def history_author_content(model, histories):
    """(B, W, 64) = Author[a] * (a >= 2) + W c of each HISTORY item (PAD slots exactly 0). Reads only the history item ids and the static,
    train-defined ``author_index`` / ``content`` buffers; rows 0 (PAD) and 1 (OOV) contribute exactly zero and receive no gradient."""
    rows = model.author_index[histories]
    author = model.author_table(rows) * (rows >= 2).unsqueeze(-1)
    out = author + f2.project_content(model.content_projection, model.content[histories])
    return out * (histories != 0).unsqueeze(-1)


def event_vectors4a(model, histories, side, valid):
    """B per-event vectors (B, W, 64); one left-to-right addition at a time: (((((e + hist) + T) + R) + P) + G)."""
    x = model.items(histories)
    x = x + history_author_content(model, histories)
    x = x + model.timespent_bucket(rf.timespent_bucket(side[..., 1], valid))
    x = x + model.ratio_bucket(rf.ratio_bucket(side[..., 1], side[..., 2], valid))
    x = x + model.position_table(f2.position_bucket(valid))                 # P first, then G (never P + G first)
    return x + model.weekgap_table(f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid))


def init_b_tables(model):
    """Construct Author (one default draw), construct W (one default draw), then N(0, 0.02) for Author (PAD row cleared) and for W (not Kaiming)."""
    model.author_table = torch.nn.Embedding(model.vocab_size + 2, f2.EMBEDDING_DIM, padding_idx=0)
    model.content_projection = torch.nn.Linear(f2.EMBEDDING_DIM, f2.EMBEDDING_DIM, bias=False)
    with torch.no_grad():
        model.author_table.weight.normal_(0.0, 0.02)
        model.author_table.weight[0].zero_()
        model.content_projection.weight.normal_(0.0, 0.02)


def make_feat04a_model(variant, content, supported_indices, max_len, seed, author_index=None, vocab_size=None):
    """f4_control / f4_recency_pool delegate to the FEAT-03 constructor (draw order item, T, R, P, G).
    B repeats that order exactly, and only AFTER all anchor draws constructs Author and W, so the anchor tables are bit-equal to the control's."""
    from .training import seed_all
    if variant not in FEATURES4A:
        raise ValueError('unregistered FEAT-04a variant')
    if variant != B_FEATURE:
        if author_index is not None or vocab_size is not None:
            raise ValueError('author_index/vocab_size belong to the author variant only')
        return f3.make_feat03_model(variant, content, supported_indices, max_len, seed)
    if author_index is None:
        raise ValueError('the author variant needs author_index')
    if max_len not in (20, 50, 100):
        raise ValueError('unregistered history length')
    seed_all(seed)
    model = Feat04aIDMean(content, supported_indices, max_len, author_index, f2.AUTHOR_VOCAB_SIZE if vocab_size is None else vocab_size)
    f2.init_embeddings(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    f2._small_init(model, model.add_anchor_tables())
    f3.draw_variant_tables(model)
    init_b_tables(model)
    if any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if n in before):
        raise ValueError('feature tables touched anchor parameters')
    object.__setattr__(model, 'init_report', {'init': 'small', 'std': 0.02, 'embeddings': f2.embedding_report(model)})
    return model


def parameter_report4a(model):
    """Parameters added on top of the item embedding (FEAT-03 convention; PAD/OOV rows are not trainable)."""
    return f3.parameter_report3(model)


def expected_param_counts4a(variant, vocab_size=None):
    """Pure formula for ``parameter_report4a``; B uses AUTHOR_VOCAB_SIZE unless ``vocab_size`` is given."""
    if variant not in FEATURES4A:
        raise ValueError('unregistered FEAT-04a variant')
    if variant != B_FEATURE:
        return f3.expected_param_counts3(variant)
    rows = {'timespent_bucket': rf.TIMESPENT_BUCKETS + 1, 'ratio_bucket': rf.RATIO_BUCKETS + 1,
            'position_table': len(f2.POSITION_EDGES) + 2, 'weekgap_table': len(f2.WEEKGAP_EDGES) + 2,
            'author_table': (f2.AUTHOR_VOCAB_SIZE if vocab_size is None else vocab_size) + 2, 'content_projection': f2.EMBEDDING_DIM}
    return f3._report3(rows, f3.INERT_ROWS3)


def bucket_spec4a_sha256(variant):
    """Digest of every constant the variant relies on (A: the FEAT-03 spec and the logit sizes; B: the author vocabulary pins). Stored in the run binding."""
    if variant not in FEATURES4A:
        raise ValueError('unregistered FEAT-04a variant')
    spec = {'feat03_bucket_spec_sha256': f3.bucket_spec3_sha256(), 'features4a': list(FEATURES4A), 'variant': variant}
    if variant == A_FEATURE:
        spec['recency_logit_sizes'] = [f3.RECENCY_PARAMS['recency_logit_pos'], f3.RECENCY_PARAMS['recency_logit_gap']]
    if variant == B_FEATURE:
        spec['author_vocab_size'] = f2.AUTHOR_VOCAB_SIZE
        spec['author_vocab_sha256'] = f2.AUTHOR_VOCAB_SHA256
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
