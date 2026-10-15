"""FEAT-04a P user-tower variants on the F4-T010 anchor (new file; FEAT-02/FEAT-03 modules are used through the module, never modified).

``variant`` switch (one per run; the item tower is the anchor's in every variant):
  f4_control              equivalence fixture for the anchor F4 (id_mean + F3 + F4), tests only, never a real run
  f4_recency_pool         A = Q2-T010: F4 event vectors pooled with 13 learnable recency logits (init exactly 0); delegated to FEAT-03 Q2
All seams are resolved at call time through the module (``f3.<name>``, ``f2.<name>``): the mutation tests patch the module attribute,
so this file never writes ``from .retrieval_feat03 import x`` (K2).
"""
import hashlib
import json

from . import retrieval_feat02 as f2
from . import retrieval_feat03 as f3

CONTROL = 'f4_control'
A_FEATURE = 'f4_recency_pool'
FEATURES4A = (CONTROL, A_FEATURE)
REAL_FEATURES4A = (A_FEATURE,)


def make_feat04a_model(variant, content, supported_indices, max_len, seed, author_index=None, vocab_size=None):
    """f4_control / f4_recency_pool delegate to the FEAT-03 constructor, so the draw order (item, T, R, P, G) is the anchor's."""
    if variant not in FEATURES4A:
        raise ValueError('unregistered FEAT-04a variant')
    if author_index is not None or vocab_size is not None:
        raise ValueError('author_index/vocab_size belong to the author variant only')
    return f3.make_feat03_model(variant, content, supported_indices, max_len, seed)


def parameter_report4a(model):
    """Parameters added on top of the item embedding (FEAT-03 convention; PAD/OOV rows are not trainable)."""
    return f3.parameter_report3(model)


def expected_param_counts4a(variant):
    """Pure formula for ``parameter_report4a``."""
    if variant not in FEATURES4A:
        raise ValueError('unregistered FEAT-04a variant')
    return f3.expected_param_counts3(variant)


def bucket_spec4a_sha256(variant):
    """Digest of every constant the variant relies on (A: the FEAT-03 spec and the logit sizes). Stored in the run binding."""
    if variant not in FEATURES4A:
        raise ValueError('unregistered FEAT-04a variant')
    spec = {'feat03_bucket_spec_sha256': f3.bucket_spec3_sha256(), 'features4a': list(FEATURES4A), 'variant': variant}
    if variant == A_FEATURE:
        spec['recency_logit_sizes'] = [f3.RECENCY_PARAMS['recency_logit_pos'], f3.RECENCY_PARAMS['recency_logit_gap']]
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
