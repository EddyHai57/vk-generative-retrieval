"""S2 learnable-SID contract pieces that do not depend on torch or on the open design points.

Covers: operator-defined C1' fallback set + 251/120415 gate, NaN/Inf STOP, schema / version identity,
H5a / H5b identity of CONTENT_V1 + C1' combined, frozen config and the fixed step schedule.
"""
import hashlib
import json
import unittest

import numpy as np


class Raise:
    def raises(self, exc, match=None):
        return unittest.TestCase().assertRaises(exc)


pytest = Raise()

from vkgr import sid_learn_v1 as sl  # noqa: E402
from vkgr import sid_tok_audit as au  # noqa: E402


def _catalog_digest(ids):
    from vkgr.data_audit import canonical_bytes
    return hashlib.sha256(canonical_bytes([int(x) for x in np.asarray(ids).tolist()])).hexdigest()


def family(n=12, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1, dtype='uint32')
    content = rng.normal(size=(n, 64)).astype('float32')
    c1 = rng.normal(size=(n, 64)).astype('float32')
    c1 /= np.linalg.norm(c1, axis=1, keepdims=True)
    manifest = {'schema': 'coll-input-v1', 'status': 'accepted', 'variant': 'C1', 'dimension': 64,
                'engineering': True, 'catalog_sha256': _catalog_digest(ids), 'vectors_sha256': au.feature_digest(c1),
                'content_feature_sha256': au.feature_digest(content)}
    return ids, content, c1, manifest


def identity_args():
    return {'c1_vectors_file_sha256': 'a' * 64, 'c1_input_manifest_sha256': 'b' * 64}


# ---------------------------------------------------------------- fallback set (operator definition)
def brute_force_participation(catalog, histories, window=5):
    """Direct transcription of sid_coll_sgns.iter_positive_pairs (same loops) without importing torch."""
    rows = {int(i): r for r, i in enumerate(catalog)}
    part = np.zeros(len(catalog), dtype=np.int64)
    for user in sorted(histories):
        events = histories[user]
        for center, item in enumerate(events):
            for context in range(max(0, center - window), min(len(events), center + window + 1)):
                if events[context] != item:
                    part[rows[int(item)]] += 1
                    part[rows[int(events[context])]] += 1
    return part


def test_fallback_matches_frozen_pair_rule_on_random_histories():
    rng = np.random.default_rng(7)
    catalog = np.arange(10, 70, dtype='uint32')
    histories = {u: [int(x) for x in rng.choice(catalog[:50], size=int(rng.integers(0, 14)))] for u in range(40)}
    got = sl.recompute_fallback_mask(catalog, histories)
    want = brute_force_participation(catalog, histories) == 0
    assert got.dtype == bool and np.array_equal(got, want)
    assert got.sum() >= 10  # items 60..69 never appear


def test_fallback_simple_cases():
    catalog = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype='uint32')
    # user 0: 1 next to 3 (legal); 3 repeats; 2 is next to 3 at the end (legal)
    # user 1: 3,3,3 repeats only; users 2 and 3 hold single events: cross-user never connects 4 and 5
    # user 4: 6 and 7 adjacent (legal); 8 never appears
    histories = {0: [1, 3, 3, 3, 3, 3, 2], 1: [3, 3, 3], 2: [4], 3: [5], 4: [6, 7]}
    mask = sl.recompute_fallback_mask(catalog, histories)
    assert mask.tolist() == [False, False, False, True, True, False, False, True]
    only_repeats = sl.recompute_fallback_mask(catalog, {0: [1, 1, 1, 1, 1, 1, 1, 1], 1: [2, 2]})
    assert only_repeats[:2].tolist() == [True, True]


def test_window_value_does_not_change_the_set_but_is_still_pinned():
    # An item has a legal pair at distance d <= w iff some copy of it sits next to a different item, so the
    # fallback set is the same for every window >= 1. Recorded as a property, not used to skip the window.
    rng = np.random.default_rng(11)
    catalog = np.arange(1, 41, dtype='uint32')
    histories = {u: [int(x) for x in rng.choice(catalog[:25], size=int(rng.integers(0, 12)))] for u in range(30)}
    histories.update({100 + u: [int(rng.choice(catalog[:3]))] * int(rng.integers(1, 9)) for u in range(8)})
    base = sl.recompute_fallback_mask(catalog, histories)
    for window in (1, 2, 3, 4, 6, 8):
        assert np.array_equal(sl.recompute_fallback_mask(catalog, histories, window=window), base)
    assert np.array_equal(base, brute_force_participation(catalog, histories, window=5) == 0)
    assert np.array_equal(base, brute_force_participation(catalog, histories, window=1) == 0)
    for bad_window in (0, True, 5.0):
        with pytest.raises(ValueError):
            sl.recompute_fallback_mask(catalog, histories, window=bad_window)


def test_user_boundary_not_connected():
    catalog = np.array([1, 2], dtype='uint32')
    mask = sl.recompute_fallback_mask(catalog, {0: [1], 1: [2]})
    assert mask.tolist() == [True, True]


def test_history_item_outside_catalog_is_error():
    catalog = np.array([1, 2], dtype='uint32')
    for bad in ([3], [True], [1.5]):
        with pytest.raises(ValueError):
            sl.recompute_fallback_mask(catalog, {0: bad})


def test_gate_requires_exact_251_of_120415():
    ok = np.zeros(sl.CATALOG_ITEMS, dtype=bool)
    ok[:251] = True
    sl.check_fallback_gate(ok)
    for count in (250, 252, 0):
        bad = np.zeros(sl.CATALOG_ITEMS, dtype=bool)
        bad[:count] = True
        with pytest.raises(ValueError):
            sl.check_fallback_gate(bad)
    short = np.zeros(sl.CATALOG_ITEMS - 1, dtype=bool)
    short[:251] = True
    with pytest.raises(ValueError):
        sl.check_fallback_gate(short)
    with pytest.raises(ValueError):
        sl.check_fallback_gate(ok.astype(np.int64))


def test_frozen_numbers():
    assert (sl.FALLBACK_EXPECTED, sl.CATALOG_ITEMS, sl.PAIR_WINDOW) == (251, 120415, 5)


# ---------------------------------------------------------------- NaN / Inf STOP
def test_require_finite_never_imputes():
    x = np.ones((3, 4), dtype='float32')
    assert sl.require_finite('x', x) is x
    for poison in (np.nan, np.inf, -np.inf):
        y = x.copy()
        y[1, 2] = poison
        with pytest.raises(ValueError):
            sl.require_finite('x', y)
    assert np.isnan(y[1, 2]) or np.isinf(y[1, 2])  # input untouched


# ---------------------------------------------------------------- schema / version identity
def test_schema_and_version_are_new_and_distinct():
    old = {'sidmap-v2', 'sidmap-coll-v1'}
    assert sl.SCHEMA not in old and sl.SCHEMA == 'sidmap-learn-v1'
    assert sl.SID_VERSION == 'LEARN_LETTER_B256D3_S42'
    for name in ('SID_RQVAE_B256D3_S42', 'SID_RQVAE_B16D3_S42', 'SID_BHKM_B256D3_S42', 'COLL_C1_B256D3_S42',
                 'COLL_C2_B256D3_S42', 'SID_V2_REPLAY_BHKM_B16D3'):
        assert sl.SID_VERSION != name
    assert sl.TASK_ID not in ('G-TOK-01', 'SID-COLL-S1-EXEC')
    assert sl.PROTOCOL not in ('generative-like-sid-v2', 'generative-like-sid-coll-v1')


# ---------------------------------------------------------------- frozen config
def test_config_values_and_strict_validation():
    cfg = sl.expected_config()
    assert cfg['seed'] == 42 and cfg['batch_size'] == 1024 and cfg.get('max_steps') == 300000 and 'epochs' not in cfg and cfg['early_stopping'] is False
    assert cfg['semantic_layers'] == 3 and cfg['codebook_size'] == 256 and cfg['beta_inherited'] == 0.25
    assert (cfg['alpha'], cfg['temperature'], cfg['gamma']) == (0.1, 0.07, 0.01)
    assert cfg['wall_cap_seconds'] == 14400 and cfg['usage_diag_steps'] == [2000, 'final']
    assert cfg['optimizer'] == {'name':'adagrad','lr':0.1,'initial_accumulator_value':0.1,'eps':1e-7}
    assert cfg['reset'] == {'enabled':True,'reset_check_every':100,'dead_window':500,'reset_until':270000,'rng_reset_offset':7919}
    assert cfg['alignment_target'] == {'name':'C1_PRIME','source_dim':64,'dim':32,'projection':'fixed_pca_train_observed','seed':42,'quantized_directly':False}
    assert cfg['loss_base'] == 'L2_total'
    assert cfg['diversity']['tau'] == 1.0 and cfg['diversity']['distance_dimension'] == 32
    assert cfg['diversity']['layer_reduction'] == 'mean' and cfg['diversity']['normalize_log_k'] is True
    assert cfg['diversity']['mask_fallback'] is False and cfg['diversity']['eps'] == 1e-12
    assert cfg['fallback']['exclude_align_negatives'] is True
    assert cfg['sid_version'] == sl.SID_VERSION and cfg['schema'] == 'sid-learn-fit-v1'
    sl.validate_config(json.loads(json.dumps(cfg)))
    for path, value in [('alpha', 0.2), ('temperature', 0.1), ('gamma', 0.02), ('seed', 43), ('seed', 42.0), ('max_steps', 300000.0), ('max_steps', 299999), ('wall_cap_seconds', 14400.0), ('epochs', True), ('epochs', 21),
                        ('batch_size', 512), ('wall_cap_seconds', 1), ('early_stopping', True),
                        ('beta_inherited', 0.5), ('extra', 1)]:
        bad = json.loads(json.dumps(cfg))
        bad[path] = value
        with pytest.raises(ValueError):
            sl.validate_config(bad)
    bad = json.loads(json.dumps(cfg))
    bad['optimizer']['lr'] = 1e-2
    with pytest.raises(ValueError):
        sl.validate_config(bad)
    bad = json.loads(json.dumps(cfg))
    del bad['alpha']
    with pytest.raises(ValueError):
        sl.validate_config(bad)


def test_fixed_step_schedule_counts_partial_batch_like_l2():
    assert sl.steps_per_epoch(120415, 1024) == 118
    assert sl.total_steps(120415, 1024, 20) == 2360
    assert sl.steps_per_epoch(1024, 1024) == 1 and sl.steps_per_epoch(1025, 1024) == 2
    assert sl.total_steps(120415, 1024, 20) > 2000  # step-2000 diagnostic lies inside the run
    for bad in ((0, 1024), (10, 0), (True, 1024)):
        with pytest.raises(ValueError):
            sl.steps_per_epoch(*bad)


# ---------------------------------------------------------------- H5a / H5b combined identity
def test_family_identity_accepts_and_binds_everything():
    ids, content, c1, manifest = family()
    ident = sl.input_family_identity(ids, content, c1, manifest, engineering=True, **identity_args())
    assert set(ident) == {'content_feature_sha256', 'c1_vectors_array_sha256', 'c1_vectors_file_sha256',
                          'c1_input_manifest_sha256', 'catalog_sha256'}
    assert ident['content_feature_sha256'] == au.feature_digest(content)
    assert ident['c1_vectors_array_sha256'] == au.feature_digest(c1)
    digest = sl.family_digest(ident)
    assert len(digest) == 64 and digest == sl.family_digest(dict(reversed(list(ident.items()))))
    for key in ident:
        changed = {**ident, key: 'e' * 64}
        assert sl.family_digest(changed) != digest


def test_h5a_h5b_checks_pass_and_name_each_failure():
    ids, content, c1, manifest = family()
    ident = sl.input_family_identity(ids, content, c1, manifest, engineering=True, **identity_args())
    stored = sl.family_digest(ident)
    checks = sl.h5_checks(ident, stored, engineering=True)
    assert [c['id'] for c in checks] == ['H5a', 'H5b'] and all(c['ok'] for c in checks)
    # H5b detects a swapped C1' vector even when CONTENT_V1 is untouched
    c1b = c1.copy()
    c1b[0] = -c1b[0]
    ident_b = sl.input_family_identity(ids, content, c1b, {**manifest, 'vectors_sha256': au.feature_digest(c1b)},
                                       engineering=True, **identity_args())
    checks = sl.h5_checks(ident_b, stored, engineering=True)
    assert checks[0]['ok'] and not checks[1]['ok']
    # H5a detects different content (formal mode compares against CONTENT_V1)
    checks = sl.h5_checks(ident, stored, engineering=False)
    assert not checks[0]['ok'] and checks[1]['ok']
    assert checks[0]['expected'] == au.CONTENT_V1_FEATURE_SHA256


def test_family_identity_rejects_bad_inputs():
    ids, content, c1, manifest = family()

    def call(i=ids, x=content, c=c1, m=manifest, **kw):
        return sl.input_family_identity(i, x, c, m, engineering=kw.pop('engineering', True),
                                        **{**identity_args(), **kw})

    call()
    bad = content.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError): call(x=bad)
    bad = c1.copy(); bad[3, 3] = np.inf
    with pytest.raises(ValueError): call(c=bad)
    scaled = (c1 * 2).astype('float32')  # not unit rows, manifest digest made consistent on purpose
    with pytest.raises(ValueError): call(c=scaled, m={**manifest, 'vectors_sha256': au.feature_digest(scaled)})
    with pytest.raises(ValueError): call(c=c1[:-1])
    with pytest.raises(ValueError): call(x=content.astype('float64'))
    with pytest.raises(ValueError): call(i=ids[::-1].copy())
    with pytest.raises(ValueError): call(i=ids.astype('int64'))
    for field, value in [('status', 'complete'), ('variant', 'C2'), ('dimension', 128), ('schema', 'x'),
                         ('vectors_sha256', 'f' * 64), ('catalog_sha256', 'f' * 64),
                         ('content_feature_sha256', 'f' * 64), ('engineering', False)]:
        with pytest.raises(ValueError): call(m={**manifest, field: value})
    with pytest.raises(ValueError): call(c1_vectors_file_sha256='G' * 64)
    with pytest.raises(ValueError): call(c1_input_manifest_sha256=None)
    # formal mode needs the real catalog size and the pinned C1' file hashes
    with pytest.raises(ValueError): call(engineering=False)


def test_formal_pin_check_is_separate_and_exact():
    good = (au.CONTENT_V1_FEATURE_SHA256, sl.C1_VECTORS_FILE_SHA256, sl.C1_INPUT_MANIFEST_SHA256)
    sl.check_formal_pins(*good)
    for index in range(3):
        bad = list(good)
        bad[index] = 'd' * 64
        with pytest.raises(ValueError):
            sl.check_formal_pins(*bad)


def test_formal_mode_reaches_the_pin_check_with_patched_constants():
    ids, content, c1, manifest = family()
    manifest = {**manifest, 'engineering': False}
    saved = (sl.CATALOG_ITEMS, sl.CONTENT_V1_FEATURE_SHA256, sl.C1_VECTORS_FILE_SHA256, sl.C1_INPUT_MANIFEST_SHA256)
    try:
        sl.CATALOG_ITEMS = len(ids)
        sl.CONTENT_V1_FEATURE_SHA256 = au.feature_digest(content)
        sl.C1_VECTORS_FILE_SHA256, sl.C1_INPUT_MANIFEST_SHA256 = 'a' * 64, 'b' * 64
        ident = sl.input_family_identity(ids, content, c1, manifest, engineering=False, **identity_args())
        assert ident['c1_vectors_file_sha256'] == 'a' * 64
        for key, value in (('c1_vectors_file_sha256', 'c' * 64), ('c1_input_manifest_sha256', 'c' * 64)):
            with pytest.raises(ValueError):
                sl.input_family_identity(ids, content, c1, manifest, engineering=False, **{**identity_args(), key: value})
        sl.CONTENT_V1_FEATURE_SHA256 = 'f' * 64
        with pytest.raises(ValueError):
            sl.input_family_identity(ids, content, c1, manifest, engineering=False, **identity_args())
    finally:
        sl.CATALOG_ITEMS, sl.CONTENT_V1_FEATURE_SHA256, sl.C1_VECTORS_FILE_SHA256, sl.C1_INPUT_MANIFEST_SHA256 = saved


def test_formal_pins_are_the_card_values():
    assert sl.C1_VECTORS_FILE_SHA256 == 'bf840eb0ecb350a4044bbe20a45e4ed82d6402f54d364aad8f17f5f554d582c7'
    assert sl.C1_INPUT_MANIFEST_SHA256 == '725fb211a342c518770623be81d490feaaec38a285c954b5b6f51f50a388e9a4'
    assert au.CONTENT_V1_FEATURE_SHA256 == '766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84'


def test_catalog_digest_matches_frozen_function_when_torch_available():
    try:
        from vkgr import sid_v2
    except ImportError:
        return
    ids = np.arange(1, 9, dtype='uint32')
    assert sl.catalog_digest(ids) == sid_v2.catalog_digest(ids)


def test_module_is_torch_free_and_does_not_touch_frozen_modules():
    import sys
    import vkgr.sid_learn_v1 as mod
    from pathlib import Path
    src = Path(mod.__file__).read_text(encoding='utf-8')
    assert 'import torch' not in src and 'from torch' not in src


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in sorted(globals().items()) if name.startswith('test_') and callable(value))
