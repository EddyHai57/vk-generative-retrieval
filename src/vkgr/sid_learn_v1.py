"""S2 learnable SID (LETTER-style offline tokenizer): torch-free contract pieces.

Original code written from the paper description only. The LETTER and ETEGRec repositories have an UNKNOWN
licence, so no source, configuration or weight of either is copied or consulted line by line.

This module holds only what the frozen card (decision_log 2026-10-18, S2 freeze) fixes without ambiguity:
the new schema / version identity, the frozen fit configuration, the fixed step schedule, the operator-defined
C1' content-fallback set with its 251 / 120415 gate, the NaN/Inf STOP rule and the H5a / H5b identity of the
CONTENT_V1 + C1' input family. Losses, fixed PCA and the fit loop live elsewhere.
Pure numpy: no file access, no label, query or split data.
"""
import hashlib
import math
import re

import numpy as np

from .data_audit import canonical_bytes
from .sid_tok_audit import CONTENT_V1_FEATURE_SHA256, feature_digest

SCHEMA = 'sidmap-learn-v1'
PROTOCOL = 'generative-like-sid-learn-v1'
TASK_ID = 'SID-LEARN-S2'
SID_VERSION = 'LEARN_LETTER_B256D3_S42'
CONFIG_SCHEMA = 'sid-learn-fit-v1'

# C1' accepted input (card section 7). File hashes are pinned for formal runs.
C1_VECTORS_FILE_SHA256 = 'bf840eb0ecb350a4044bbe20a45e4ed82d6402f54d364aad8f17f5f554d582c7'
C1_INPUT_MANIFEST_SHA256 = '725fb211a342c518770623be81d490feaaec38a285c954b5b6f51f50a388e9a4'

CATALOG_ITEMS = 120415
FALLBACK_EXPECTED = 251
PAIR_WINDOW = 5
L2_SOURCE_COMMIT = '80f7eed179981c65094c628744099a181a81d907'  # gtok01r/g, the repaired RQ-VAE L2 version

_HEX64 = re.compile(r'[0-9a-f]{64}')


def _bad(message):
    raise ValueError(message)


def _hex64(value, what):
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        _bad(f'{what}: lowercase sha256 hex required')
    return value


def catalog_digest(ids):
    """Same definition as sid_v2.catalog_digest (sha256 of the canonical JSON id list)."""
    return hashlib.sha256(canonical_bytes([int(x) for x in np.asarray(ids).tolist()])).hexdigest()


# --------------------------------------------------------------------------- NaN / Inf
def require_finite(name, array):
    """NaN / Inf are never imputed: any occurrence is a STOP."""
    if not np.isfinite(np.asarray(array)).all():
        _bad(f'STOP: {name} holds NaN or Inf')
    return array


# --------------------------------------------------------------------------- frozen configuration
def expected_config():
    """The ruled S2 recipe; no validation selection, scan, or optimizer retuning."""
    return {
        'schema': CONFIG_SCHEMA, 'task_id': TASK_ID, 'sid_version': SID_VERSION,
        'method': 'LETTER-style offline tokenizer (original implementation from the paper)',
        'reconstruction_input': {'name': 'CONTENT_V1', 'dim': 64},
        'alignment_target': {'name':'C1_PRIME','source_dim':64,'dim':32,
                             'projection':'fixed_pca_train_observed','seed':42,'quantized_directly':False},
        'semantic_layers': 3, 'codebook_size': 256, 'suffix': 'deterministic',
        'architecture': {'content_dim':64,'encoder_hidden':[512,256,128],'latent_dim':32,'levels':3},
        'l2_source_commit': L2_SOURCE_COMMIT, 'beta_inherited': 0.25, 'loss_base':'L2_total',
        'alignment_latent':'straight_through_quantized','alignment_direction':'latent_to_target',
        'alignment_normalization':'both_l2',
        'alpha': 0.1, 'temperature': 0.07, 'gamma': 0.01,
        'diversity': {'probability':'softmax_negative_squared_distance','tau':1.0,'distance_dimension':32,
                      'q_axis':'batch','normalize_log_k':True,'layer_reduction':'mean','eps':1e-12,
                      'mask_fallback':False,'ema':False,'detach_encoder':False,'detach_codebook':False,
                      'residual_source':'forward_hard_chain'},
        'weights_are_priors_not_tuned': True,
        'seed': 42, 'batch_size': 1024, 'max_steps': 300000, 'early_stopping': False, 'select_by_validation': False,
        'optimizer': {'name':'adagrad','lr':0.1,'initial_accumulator_value':0.1,'eps':1e-7},
        'reset': {'enabled':True,'reset_check_every':100,'dead_window':500,'reset_until':270000,'rng_reset_offset':7919},
        'wall_cap_seconds': 14400, 'usage_diag_steps': [2000, 'final'],
        'fallback': {'window': PAIR_WINDOW, 'expected': FALLBACK_EXPECTED, 'denominator': CATALOG_ITEMS,
                     'masked_in_align_only': True,'exclude_align_negatives':True},
    }


def _first_difference(left, right, path='config'):
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                return f'{path}.{key}'
            found = _first_difference(left[key], right[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f'{path}[{index}]')
            if found:
                return found
        return None
    return None if left == right else path


def validate_config(value):
    """Strict whitelist: key set, types and values must equal the approved configuration exactly."""
    found = _first_difference(value, expected_config())
    if found:
        _bad(f'unapproved fit config at {found}')
    return value


def steps_per_epoch(n, batch_size):
    """A trailing partial batch counts as a step (same epoch handling as the L2 fit loop)."""
    if type(n) is not int or type(batch_size) is not int or n < 1 or batch_size < 1:
        _bad('n and batch_size must be positive ints')
    return -(-n // batch_size)


def total_steps(n, batch_size, epochs):
    if type(epochs) is not int or epochs < 1:
        _bad('epochs must be a positive int')
    return steps_per_epoch(n, batch_size) * epochs


# --------------------------------------------------------------------------- C1' content-fallback set
def recompute_fallback_mask(catalog, histories, window=PAIR_WINDOW):
    """Items of `catalog` with no legal C1' co-occurrence pair; bool [len(catalog)].

    Same rule as sid_coll_sgns.iter_positive_pairs: inside one user, events at index distance <= window with a
    different item form a pair; users are never connected. `histories` maps user -> ordered item ids of the
    0-24 week training period. Every event must be a catalog item. An item that never appears is a fallback
    item too (participation 0), exactly as fit_sgns counts it.
    """
    if type(window) is not int or window < 1:
        _bad('window must be a positive int')
    catalog = np.asarray(catalog)
    if catalog.ndim != 1 or catalog.dtype.kind not in 'iu' or (len(catalog) > 1 and not np.all(catalog[1:] > catalog[:-1])):
        _bad('catalog must be a strictly ascending integer vector')
    participates = np.zeros(len(catalog), dtype=bool)
    for user in sorted(histories):
        events = histories[user]
        for item in events:
            if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                _bad('history holds a non-integer item')
        if not len(events):
            continue
        values = np.array([int(x) for x in events], dtype=np.int64)
        rows = np.searchsorted(catalog, values)
        if (rows >= len(catalog)).any() or (catalog[np.minimum(rows, len(catalog) - 1)] != values).any():
            _bad('history holds an item outside the catalog')
        for distance in range(1, min(window, len(values) - 1) + 1):
            different = values[:-distance] != values[distance:]
            participates[rows[:-distance][different]] = True
            participates[rows[distance:][different]] = True
    return ~participates


def check_fallback_gate(mask):
    """Gate: the recomputed count must equal the registered 251 of 120415. Anything else is a STOP."""
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 1:
        _bad('STOP: fallback mask must be a bool vector')
    if len(mask) != CATALOG_ITEMS or int(mask.sum()) != FALLBACK_EXPECTED:
        _bad(f'STOP: fallback count {int(mask.sum())} of {len(mask)} differs from '
             f'{FALLBACK_EXPECTED} of {CATALOG_ITEMS}')
    return int(mask.sum())


# --------------------------------------------------------------------------- input family identity (H5a / H5b)
IDENTITY_KEYS = ('content_feature_sha256', 'c1_vectors_array_sha256', 'c1_vectors_file_sha256',
                 'c1_input_manifest_sha256', 'catalog_sha256')


def check_formal_pins(content_digest, c1_vectors_file_sha256, c1_input_manifest_sha256):
    """Formal runs: CONTENT_V1 array digest and the two C1' file hashes must be exactly the registered ones."""
    if content_digest != CONTENT_V1_FEATURE_SHA256:
        _bad('formal run needs the CONTENT_V1 array digest')
    if c1_vectors_file_sha256 != C1_VECTORS_FILE_SHA256:
        _bad("formal run needs the pinned C1' vectors file hash")
    if c1_input_manifest_sha256 != C1_INPUT_MANIFEST_SHA256:
        _bad("formal run needs the pinned C1' input manifest hash")


def input_family_identity(ids, content, c1_vectors, c1_manifest, *, engineering, c1_vectors_file_sha256,
                          c1_input_manifest_sha256):
    """Validate the CONTENT_V1 + C1' input family and return its identity dict (five digests)."""
    if type(engineering) is not bool:
        _bad('engineering must be an explicit bool')
    if not isinstance(ids, np.ndarray) or ids.dtype != np.uint32 or ids.ndim != 1 or len(ids) < 2 \
            or np.any(ids[1:] <= ids[:-1]):
        _bad('strict ascending uint32 catalog required')
    if not engineering and len(ids) != CATALOG_ITEMS:
        _bad('formal catalog must have 120415 items')
    for name, array in (('content', content), ('c1_vectors', c1_vectors)):
        if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.shape != (len(ids), 64):
            _bad(f'{name} must be float32 [N, 64] aligned with the catalog')
        require_finite(name, array)
    if not np.allclose(np.linalg.norm(c1_vectors.astype(np.float64), axis=1), 1, atol=2e-6, rtol=0):
        _bad("C1' rows must be unit vectors")
    _hex64(c1_vectors_file_sha256, 'c1_vectors_file_sha256')
    _hex64(c1_input_manifest_sha256, 'c1_input_manifest_sha256')
    content_digest, c1_digest, cat_digest = feature_digest(content), feature_digest(c1_vectors), catalog_digest(ids)
    if type(c1_manifest) is not dict:
        _bad("C1' input manifest must be a dict")
    expected = {'schema': 'coll-input-v1', 'status': 'accepted', 'variant': 'C1', 'dimension': 64,
                'engineering': engineering, 'catalog_sha256': cat_digest, 'vectors_sha256': c1_digest,
                'content_feature_sha256': content_digest}
    for key, want in expected.items():
        got = c1_manifest.get(key)
        if type(got) is not type(want) or got != want:
            _bad(f"C1' input manifest field {key} does not match the supplied arrays")
    if not engineering:
        check_formal_pins(content_digest, c1_vectors_file_sha256, c1_input_manifest_sha256)
    return {'content_feature_sha256': content_digest, 'c1_vectors_array_sha256': c1_digest,
            'c1_vectors_file_sha256': c1_vectors_file_sha256, 'c1_input_manifest_sha256': c1_input_manifest_sha256,
            'catalog_sha256': cat_digest}


def family_digest(identity):
    """sha256 over the canonical JSON of the five identity digests (key order independent)."""
    if type(identity) is not dict or set(identity) != set(IDENTITY_KEYS):
        _bad('input family identity must hold exactly the five registered digests')
    for key in IDENTITY_KEYS:
        _hex64(identity[key], key)
    return hashlib.sha256(canonical_bytes({key: identity[key] for key in IDENTITY_KEYS})).hexdigest()


def h5_checks(identity, stored_family_sha256, *, engineering):
    """H5a: content source digest. H5b: the combined CONTENT_V1 + C1' identity equals the bound one."""
    expected = identity['content_feature_sha256'] if engineering else CONTENT_V1_FEATURE_SHA256
    return [{'id': 'H5a', 'ok': identity['content_feature_sha256'] == expected,
             'actual': identity['content_feature_sha256'], 'expected': expected},
            {'id': 'H5b', 'ok': family_digest(identity) == stored_family_sha256,
             'actual': family_digest(identity), 'expected': stored_family_sha256}]
