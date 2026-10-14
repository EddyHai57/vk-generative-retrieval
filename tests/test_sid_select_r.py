"""GTOK-01R FEAT-002: diagnostic criteria (I1/I2/I4), probe/decision schemas, select_arm, preflight, override.
Pure synthetic records, no file access, no hard-coded temp paths.
Expected values are hand-computed from the pre-registered design text (gtok01r-design.md sections 2 and 4, as revised).
"""
import copy
import hashlib
import json
import unittest
from unittest import mock
import numpy as np
from vkgr import sid_rqvae_r as rr
from vkgr import sid_tok_audit as au

KS = {'T1': 256, 'T2': 16}
HEALTHY = {'T1': [230, 230, 230], 'T2': [14, 14, 14]}
MIN_OK = {'T1': 205, 'T2': 13}
JUST_BELOW = {'T1': 204, 'T2': 12}
LOG_STEPS = list(range(1000, 30001, 1000))
USAGE_STEPS = list(range(2000, 30001, 2000))
P0_LOSS = {'T1': (0.8199, 1.0100, 0.8115), 'T2': (0.8754, 0.8116, 0.8115)}
CONFIG_SHA = 'c' * 64
CATALOG_SHA = 'd' * 64
CONTENT_FILE_SHA = 'e' * 64
CONTENT_ARRAY_SHA = 'f' * 64
COMMIT = 'a' * 40
VARIANTS = ('T1', 'T2')
ARMS = ('p0', 'l1', 'l2', 'l3', 'l4')
L_ARMS = ('l1', 'l2', 'l3', 'l4')


def make_probe(variant, arm, *, status='complete', steps=None, usage=None, usage_at=None, final_usage=None, r2=0.3,
               norms=(1.0, 0.8, 0.5, 0.3), ms=10.0, loss_head=None, recon_val=0.5, diagnostics=None,
               config_sha=CONFIG_SHA, commit=COMMIT):
    """A schema-valid probe-result. Defaults: p0 reproduces the old collapse, every L arm is fully healthy."""
    K = KS[variant]
    if usage is None:
        usage = [1, 1, 2] if arm == 'p0' else HEALTHY[variant]
    if steps is None:
        steps = 30000 if status == 'complete' else 12000
    loss_vals = [0.9] * 30
    if arm == 'p0':
        loss_vals[:3] = list(P0_LOSS[variant])
    if loss_head is not None:
        loss_vals[:len(loss_head)] = list(loss_head)
    usage_at = usage_at or {}
    elapsed = ms * steps / 1000 if status == 'complete' else 3001.0
    final = None
    if status == 'complete':
        final = {'canonical_usage': list(final_usage if final_usage is not None else usage), 'r2': r2,
                 'residual_norms': list(norms)}
    return {
        'schema': 'gtok01r-probe-v1', 'variant': variant, 'arm': arm, 'K': K,
        'config_sha256': config_sha, 'code_commit': commit, 'content_file_sha256': CONTENT_FILE_SHA,
        'content_array_sha256': CONTENT_ARRAY_SHA, 'catalog_sha256': CATALOG_SHA,
        'status': status, 'steps': steps, 'elapsed': elapsed, 'ms_per_step': elapsed * 1000 / steps,
        'loss_log': [[s, v] for s, v in zip(LOG_STEPS, loss_vals) if s <= steps],
        'recon_log': [[s, recon_val] for s in LOG_STEPS if s <= steps],
        'usage_log': [[s, list(usage_at.get(s, usage))] for s in USAGE_STEPS if s <= steps],
        'diagnostics': diagnostics if diagnostics is not None else [],
        'final': final,
    }


def set_arm(spec, arm, fn=None, **kw):
    """Same defect on both K. fn(variant) -> kwargs overrides kw."""
    for variant in VARIANTS:
        spec[(variant, arm)] = fn(variant) if fn else dict(kw)
    return spec


def make_set(spec=None):
    spec = spec or {}
    return [make_probe(v, arm, **spec.get((v, arm), {})) for v in VARIANTS for arm in ARMS]


def select(spec=None, **kw):
    return rr.select_arm(make_set(spec), **kw)


def below_a1(variant):
    return {'usage': [JUST_BELOW[variant], 230 if variant == 'T1' else 14, 230 if variant == 'T1' else 14]}


def r2_zero(variant):
    return {'r2': 0.0}


def r2_low(variant):
    return {'r2': 0.01}


def criteria(result, arm, variant):
    return result['per_arm_per_K'][arm][variant]


def rec(step, z=0.5, dead=0.0, sub=(50, 50, 50)):
    return {'step': step, 'z_std_mean': z, 'dead_relu': [dead] * 6, 'usage_sub': list(sub)}


def i1_probe(diag, usage=None, usage_at=None, variant='T1'):
    return make_probe(variant, 'p0', usage=usage, usage_at=usage_at, diagnostics=diag)


class Constants(unittest.TestCase):
    def test_selection_constants_are_the_preregistered_values(self):
        self.assertEqual(rr.PREREG_R_SELECT, {
            'variant_K': {'T1': 256, 'T2': 16},
            'p0_loss_windows': {'T1': (0.8199, 1.0100, 0.8115), 'T2': (0.8754, 0.8116, 0.8115)},
            'p0_loss_tol': 1e-3, 'p0_usage_ref': (1, 1, 2), 'p0_usage_slack': 1, 'p0_final_usage_max': 3,
            'usage_log_steps': tuple(range(2000, 30001, 2000)),
            'a2_from_step': 20000, 'a2_to_step': 30000,
            'd_points': (20000, 22000, 24000, 26000, 28000, 30000),
            'e_r2_min': 0.05,
            'i1_z_std_min': 1e-3, 'i1_dead_frac': 0.99, 'i1_max_step': 3000, 'i1_usage_max': 3,
            'i2_step': 30000, 'i2_rel_tol': 1e-3,
            'i4_norm_range': (0.99, 1.01),
            'old_fit_wall_seconds': {'T1': 4354, 'T2': 4302}, 'preflight_limit_seconds': 6000,
        })

    def test_flat_prereg_r_of_the_core_module_is_not_extended(self):
        self.assertNotIn('d_points', rr.PREREG_R)
        self.assertEqual(rr.PREREG_R['probe_steps'], 30000)

    def test_gate_thresholds_are_reused_not_rewritten(self):
        self.assertEqual(au.USAGE_NUM_DEN, (4, 5))
        self.assertTrue(au.usage_ok(205, 256) and not au.usage_ok(204, 256))
        self.assertTrue(rr.layers_ok([205, 205, 205], 256) and not rr.layers_ok([205, 204, 205], 256))
        self.assertTrue(rr.layers_ok([13, 13, 13], 16) and not rr.layers_ok([13, 12, 13], 16))
        # equality case of the 4/5 rule (K=5: 4*5 == 5*4) must pass; K=256/16 never hit equality
        self.assertTrue(rr.layers_ok([4, 4, 4], 5))
        self.assertFalse(rr.layers_ok([3, 4, 4], 5))


class SstAndI4(unittest.TestCase):
    def test_sst_is_sum_over_features_mean_over_rows(self):
        x = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=np.float32)
        self.assertEqual(rr.sst_of(x), 2.0)

    def test_sst_matches_reconstruction_report_definition(self):
        rng = np.random.RandomState(0)
        x = rng.normal(size=(50, 64)).astype(np.float32)
        data = x.astype(np.float64)
        self.assertAlmostEqual(rr.sst_of(x), float(((data - data.mean(0)) ** 2).sum()) / 50, places=12)

    def test_i4_range_and_zero_rows(self):
        unit = np.eye(64, dtype=np.float32)
        self.assertEqual(rr.classify_i4(unit, 0), 'SUPPORTED')
        self.assertEqual(rr.classify_i4(unit, 1), 'REFUTED')  # zero-norm count must equal the catalog value
        with_zero = np.concatenate([unit, np.zeros((1, 64), np.float32)])  # mean norm 64/65 = 0.9846 < 0.99
        self.assertEqual(rr.classify_i4(with_zero, 1), 'REFUTED')
        hundred = np.concatenate([np.tile(unit, (2, 1))[:100], np.zeros((1, 64), np.float32)])  # mean 100/101
        self.assertEqual(rr.classify_i4(hundred, 1), 'SUPPORTED')
        self.assertEqual(rr.classify_i4(hundred, 0), 'REFUTED')
        self.assertEqual(rr.classify_i4(hundred, 2), 'REFUTED')
        self.assertEqual(rr.classify_i4(unit * np.float32(1.02), 0), 'REFUTED')
        self.assertEqual(rr.classify_i4(unit * np.float32(1.005), 0), 'SUPPORTED')

    def test_i4_range_ends_are_inclusive(self):
        unit = np.eye(8, dtype=np.float32)  # mean norm exactly 1.0
        for lo_hi in ((1.0, 1.01), (0.99, 1.0)):
            with mock.patch.dict(rr.PREREG_R_SELECT, {'i4_norm_range': lo_hi}):
                self.assertEqual(rr.classify_i4(unit, 0), 'SUPPORTED')
        with mock.patch.dict(rr.PREREG_R_SELECT, {'i4_norm_range': (1.0000001, 1.01)}):
            self.assertEqual(rr.classify_i4(unit, 0), 'REFUTED')
        with mock.patch.dict(rr.PREREG_R_SELECT, {'i4_norm_range': (0.99, 0.9999999)}):
            self.assertEqual(rr.classify_i4(unit, 0), 'REFUTED')


class I2(unittest.TestCase):
    def probe(self, recon_at_30000, total_at_30000=None, recon_at_29000=0.0):
        p = make_probe('T1', 'p0')
        p['recon_log'][-2][1] = recon_at_29000
        p['recon_log'][-1][1] = recon_at_30000
        if total_at_30000 is not None:
            p['loss_log'][-1][1] = total_at_30000
        return p

    def test_close_to_sst_supported_far_refuted(self):
        self.assertEqual(rr.classify_i2(self.probe(100.05), 100.0), 'SUPPORTED')
        self.assertEqual(rr.classify_i2(self.probe(99.95), 100.0), 'SUPPORTED')
        self.assertEqual(rr.classify_i2(self.probe(100.2), 100.0), 'REFUTED')
        self.assertEqual(rr.classify_i2(self.probe(99.8), 100.0), 'REFUTED')

    def test_boundary_is_strict(self):
        with mock.patch.dict(rr.PREREG_R_SELECT, {'i2_rel_tol': 0.25}):
            self.assertEqual(rr.classify_i2(self.probe(5.0), 4.0), 'REFUTED')  # |5-4|/4 == 0.25 exactly
            self.assertEqual(rr.classify_i2(self.probe(4.99), 4.0), 'SUPPORTED')
            self.assertEqual(rr.classify_i2(self.probe(3.0), 4.0), 'REFUTED')  # |3-4|/4 == 0.25 exactly
            self.assertEqual(rr.classify_i2(self.probe(3.01), 4.0), 'SUPPORTED')

    def test_reads_the_recon_log_entry_of_step_30000_not_total_and_not_29000(self):
        near_at_30000_far_at_29000 = self.probe(100.0, total_at_30000=500.0, recon_at_29000=900.0)
        self.assertEqual(rr.classify_i2(near_at_30000_far_at_29000, 100.0), 'SUPPORTED')
        far_at_30000_near_at_29000 = self.probe(500.0, total_at_30000=100.0, recon_at_29000=100.0)
        self.assertEqual(rr.classify_i2(far_at_30000_near_at_29000, 100.0), 'REFUTED')

    def test_missing_step_30000_entry_is_an_error_not_a_third_state(self):
        p = self.probe(100.0)
        p['recon_log'] = p['recon_log'][:-1]
        with self.assertRaises(ValueError):
            rr.classify_i2(p, 100.0)
        with self.assertRaises(ValueError):
            rr.classify_i2(self.probe(100.0), 0.0)


class I1(unittest.TestCase):
    COLLAPSED = [1, 1, 1]

    def test_supported_when_a_holds_early_and_usage_stays_collapsed(self):
        p = i1_probe([rec(0), rec(100, z=5e-4), rec(3000)], usage=self.COLLAPSED)
        self.assertEqual(rr.classify_i1(p), 'SUPPORTED')
        p = i1_probe([rec(0), rec(100, dead=1.0)], usage=self.COLLAPSED)
        self.assertEqual(rr.classify_i1(p), 'SUPPORTED')

    def test_a_boundaries_z_std_strictly_below_and_dead_fraction_inclusive(self):
        at_z = i1_probe([rec(100, z=1e-3)], usage=self.COLLAPSED)
        self.assertEqual(rr.classify_i1(at_z), 'REFUTED')  # 1e-3 itself does not satisfy "< 1e-3"
        below_z = i1_probe([rec(100, z=9.99e-4)], usage=self.COLLAPSED)
        self.assertEqual(rr.classify_i1(below_z), 'SUPPORTED')
        at_dead = i1_probe([rec(100, dead=0.99)], usage=self.COLLAPSED)
        self.assertEqual(rr.classify_i1(at_dead), 'SUPPORTED')  # 0.99 itself satisfies ">= 0.99"
        under_dead = i1_probe([rec(100, dead=0.9899)], usage=self.COLLAPSED)
        self.assertEqual(rr.classify_i1(under_dead), 'REFUTED')

    def test_one_dead_relu_layer_is_enough(self):
        r = rec(100)
        r['dead_relu'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.995]
        self.assertEqual(rr.classify_i1(i1_probe([r], usage=self.COLLAPSED)), 'SUPPORTED')

    def test_a_only_counts_up_to_step_3000(self):
        late = i1_probe([rec(0), rec(3000), rec(4000, z=0.0)], usage=self.COLLAPSED)
        self.assertEqual(rr.classify_i1(late), 'REFUTED')  # formal definition: no record <= 3000 satisfies (a)
        at_3000 = i1_probe([rec(0), rec(3000, z=0.0)], usage=self.COLLAPSED)
        self.assertEqual(rr.classify_i1(at_3000), 'SUPPORTED')

    def test_a_after_3000_with_usage_above_3_at_2000_is_inconclusive(self):
        p = i1_probe([rec(0), rec(1000), rec(4000, z=0.0)], usage=self.COLLAPSED, usage_at={2000: [50, 50, 50]})
        self.assertEqual(rr.classify_i1(p), 'INCONCLUSIVE')

    def test_a_holds_but_usage_recovers_late_is_inconclusive(self):
        p = i1_probe([rec(0), rec(100, z=0.0)], usage=self.COLLAPSED, usage_at={22000: [40, 40, 40]})
        self.assertEqual(rr.classify_i1(p), 'INCONCLUSIVE')

    def test_collapsed_full_needs_every_point_and_every_layer_at_most_3(self):
        diag = [rec(0), rec(100, z=0.0)]
        self.assertEqual(rr.classify_i1(i1_probe(diag, usage=[3, 3, 3])), 'SUPPORTED')
        self.assertEqual(rr.classify_i1(i1_probe(diag, usage=[3, 3, 3], usage_at={30000: [3, 3, 4]})), 'INCONCLUSIVE')
        self.assertEqual(rr.classify_i1(i1_probe(diag, usage=[3, 3, 3], usage_at={2000: [4, 3, 3]})), 'INCONCLUSIVE')
        self.assertEqual(rr.classify_i1(i1_probe(diag, usage=[3, 3, 3], usage_at={16000: [3, 4, 3]})), 'INCONCLUSIVE')

    def test_refuted_needs_usage_at_2000_at_most_3_on_every_layer(self):
        diag = [rec(0), rec(100), rec(3000)]
        self.assertEqual(rr.classify_i1(i1_probe(diag, usage=[3, 3, 3])), 'REFUTED')
        self.assertEqual(rr.classify_i1(i1_probe(diag, usage=[3, 3, 3], usage_at={2000: [3, 3, 4]})), 'INCONCLUSIVE')

    def test_p0_that_does_not_collapse_is_inconclusive(self):
        p = i1_probe([rec(0), rec(100), rec(3000)], usage=[50, 50, 50])
        self.assertEqual(rr.classify_i1(p), 'INCONCLUSIVE')

    def test_a_requires_some_record_up_to_3000_for_refuted(self):
        self.assertEqual(rr.classify_i1(i1_probe([rec(4000)], usage=self.COLLAPSED)), 'INCONCLUSIVE')
        self.assertEqual(rr.classify_i1(i1_probe([], usage=self.COLLAPSED)), 'INCONCLUSIVE')

    def test_usage_sub_never_influences_i1(self):
        for diag_a, usage, expected in (
                ([rec(0, sub=(1, 1, 1)), rec(100, z=0.0, sub=(1, 1, 1))], self.COLLAPSED, 'SUPPORTED'),
                ([rec(0, sub=(1, 1, 1)), rec(100, sub=(1, 1, 1))], self.COLLAPSED, 'REFUTED'),
                ([rec(0, sub=(1, 1, 1)), rec(100, z=0.0, sub=(1, 1, 1))], [50, 50, 50], 'INCONCLUSIVE'),
                ([rec(0, sub=(1, 1, 1)), rec(100, sub=(1, 1, 1))], [50, 50, 50], 'INCONCLUSIVE')):
            diag_b = copy.deepcopy(diag_a)
            for r in diag_b:
                r['usage_sub'] = [200, 200, 200]
            self.assertEqual(rr.classify_i1(i1_probe(diag_a, usage=usage)), expected)
            self.assertEqual(rr.classify_i1(i1_probe(diag_b, usage=usage)), expected)


class Schemas(unittest.TestCase):
    def probe(self, **kw):
        return make_probe('T1', 'l1', **kw)

    def test_valid_probes_pass_for_every_status_and_arm(self):
        for variant in VARIANTS:
            for arm in ARMS:
                rr.validate_probe(make_probe(variant, arm))
        for status in ('fail_nonfinite', 'fail_wall', 'fail_wall_projected'):
            rr.validate_probe(self.probe(status=status))
            rr.validate_probe(self.probe(status=status, steps=1000))

    def test_json_round_trip_still_valid(self):
        p = json.loads(json.dumps(self.probe()))
        rr.validate_probe(p)

    def test_rejects_missing_extra_and_wrong_schema(self):
        for key in list(self.probe()):
            p = self.probe()
            del p[key]
            with self.assertRaises(ValueError, msg=key):
                rr.validate_probe(p)
        p = self.probe()
        p['extra'] = 1
        with self.assertRaises(ValueError):
            rr.validate_probe(p)
        for bad in ('gtok01r-probe-v0', 'gtok01-probe-v1', None, 1):
            p = self.probe()
            p['schema'] = bad
            with self.assertRaises(ValueError, msg=str(bad)):
                rr.validate_probe(p)
        with self.assertRaises(ValueError):
            rr.validate_probe([1, 2])

    def test_rejects_uppercase_or_unknown_arm_and_variant(self):
        for arm in ('L1', 'P0', 'l5', '', None, 1):
            p = self.probe()
            p['arm'] = arm
            with self.assertRaises(ValueError, msg=str(arm)):
                rr.validate_probe(p)
        for variant in ('t1', 'T3', None):
            p = self.probe()
            p['variant'] = variant
            with self.assertRaises(ValueError, msg=str(variant)):
                rr.validate_probe(p)

    def test_k_must_match_the_variant_and_be_a_real_int(self):
        for K in (16, 255, 256.0, True, None):
            p = self.probe()
            p['K'] = K
            with self.assertRaises(ValueError, msg=str(K)):
                rr.validate_probe(p)

    def test_rejects_bad_digests_and_commit(self):
        for key, bad in (('config_sha256', 'C' * 64), ('config_sha256', 'c' * 63), ('catalog_sha256', 5),
                         ('content_file_sha256', 'g' * 64), ('content_array_sha256', ''),
                         ('code_commit', 'a' * 64), ('code_commit', 'A' * 40), ('code_commit', None)):
            p = self.probe()
            p[key] = bad
            with self.assertRaises(ValueError, msg=key):
                rr.validate_probe(p)

    def test_rejects_bad_status_and_final_nullness(self):
        p = self.probe()
        p['status'] = 'running'
        with self.assertRaises(ValueError):
            rr.validate_probe(p)
        p = self.probe()
        p['final'] = None
        with self.assertRaises(ValueError):
            rr.validate_probe(p)  # complete without final
        p = self.probe(status='fail_wall')
        p['final'] = {'canonical_usage': [1, 1, 1], 'r2': 0.1, 'residual_norms': [1, 0.5, 0.2, 0.1]}
        with self.assertRaises(ValueError):
            rr.validate_probe(p)  # non-complete with final

    def test_rejects_malformed_final(self):
        for bad in ({'canonical_usage': [1, 1], 'r2': 0.1, 'residual_norms': [1, 0.5, 0.2, 0.1]},
                    {'canonical_usage': [1, 1, 1.5], 'r2': 0.1, 'residual_norms': [1, 0.5, 0.2, 0.1]},
                    {'canonical_usage': [1, 1, 1], 'r2': float('nan'), 'residual_norms': [1, 0.5, 0.2, 0.1]},
                    {'canonical_usage': [1, 1, 1], 'r2': 0.1, 'residual_norms': [1, 0.5, 0.2]},
                    {'canonical_usage': [1, 1, 1], 'r2': 0.1, 'residual_norms': [1, 0.5, 0.2, 0.1], 'x': 1},
                    {'canonical_usage': [1, 1, 300], 'r2': 0.1, 'residual_norms': [1, 0.5, 0.2, 0.1]},
                    {'canonical_usage': [1, 1, 1], 'r2': True, 'residual_norms': [1, 0.5, 0.2, 0.1]}):
            p = self.probe()
            p['final'] = bad
            with self.assertRaises(ValueError, msg=str(bad)):
                rr.validate_probe(p)

    def test_rejects_wrong_steps_numbers_and_inconsistent_ms_per_step(self):
        for key, bad in (('steps', 29999), ('steps', True), ('steps', 30000.0), ('elapsed', float('nan')),
                         ('elapsed', 'x'), ('elapsed', -1.0), ('ms_per_step', 1.0), ('ms_per_step', None)):
            p = self.probe()
            p[key] = bad
            with self.assertRaises(ValueError, msg=f'{key}={bad}'):
                rr.validate_probe(p)

    def test_complete_requires_the_full_log_grids(self):
        for key in ('loss_log', 'recon_log', 'usage_log'):
            p = self.probe()
            p[key] = p[key][:-1]
            with self.assertRaises(ValueError, msg=key):
                rr.validate_probe(p)
        p = self.probe()
        p['usage_log'][3][0] = 9999
        with self.assertRaises(ValueError):
            rr.validate_probe(p)

    def test_log_entry_shapes(self):
        p = self.probe()
        p['usage_log'][0][1] = [1, 2]
        with self.assertRaises(ValueError):
            rr.validate_probe(p)
        p = self.probe()
        p['loss_log'][0] = [1000]
        with self.assertRaises(ValueError):
            rr.validate_probe(p)
        p = self.probe()
        p['loss_log'][0][1] = float('inf')
        with self.assertRaises(ValueError):
            rr.validate_probe(p)
        p = self.probe()
        p['diagnostics'] = 'none'
        with self.assertRaises(ValueError):
            rr.validate_probe(p)
        p = self.probe()
        p['diagnostics'] = [{'step': 1}]
        with self.assertRaises(ValueError):
            rr.validate_probe(p)

    def test_non_complete_probe_may_carry_a_log_prefix_only(self):
        p = self.probe(status='fail_wall', steps=5000)
        self.assertEqual([s for s, _ in p['usage_log']], [2000, 4000])
        rr.validate_probe(p)
        p['ms_per_step'] = None
        with self.assertRaises(ValueError):
            rr.validate_probe(p)  # steps > 0 requires a number


class ProbeSet(unittest.TestCase):
    def test_complete_set_is_accepted_in_any_order(self):
        probes = make_set()
        rr.validate_probe_set(probes)
        rr.validate_probe_set(list(reversed(probes)), config_sha256=CONFIG_SHA)

    def test_wrong_count_missing_and_duplicate_pairs_rejected(self):
        probes = make_set()
        for bad in (probes[:-1], probes + [probes[0]], probes[:-1] + [copy.deepcopy(probes[0])], []):
            with self.assertRaises(ValueError):
                rr.validate_probe_set(bad)
        swapped = probes[:-1] + [make_probe('T1', 'l4')]
        with self.assertRaises(ValueError):
            rr.validate_probe_set(swapped)

    def test_config_sha_and_commit_must_agree(self):
        probes = make_set()
        probes[4]['config_sha256'] = '9' * 64
        with self.assertRaises(ValueError):
            rr.validate_probe_set(probes)
        probes = make_set()
        probes[7]['code_commit'] = 'b' * 40
        with self.assertRaises(ValueError):
            rr.validate_probe_set(probes)
        with self.assertRaises(ValueError):
            rr.validate_probe_set(make_set(), config_sha256='9' * 64)

    def test_content_and_catalog_digests_must_agree(self):
        for key in ('content_file_sha256', 'content_array_sha256', 'catalog_sha256'):
            probes = make_set()
            probes[3][key] = '1' * 64
            with self.assertRaises(ValueError, msg=key):
                rr.validate_probe_set(probes)

    def test_invalid_member_is_rejected(self):
        probes = make_set()
        probes[2]['arm'] = 'L2'
        with self.assertRaises(ValueError):
            rr.validate_probe_set(probes)

    def test_select_arm_rejects_an_invalid_set(self):
        with self.assertRaises(ValueError):
            rr.select_arm(make_set()[:-1])


class SelectArm(unittest.TestCase):
    def test_all_healthy_picks_l1_with_no_preflight_surprise(self):
        r = select()
        self.assertEqual(r['decision'], 'l1')
        self.assertEqual(r['escalate_listed_arms'], [])
        self.assertIsNone(r['override'])
        self.assertEqual(sorted(r['per_arm_per_K']), list(L_ARMS))

    def test_fixed_order_first_qualified_wins(self):
        for failing, expected in ((['l1'], 'l2'), (['l1', 'l2'], 'l3'), (['l1', 'l2', 'l3'], 'l4')):
            spec = {}
            for arm in failing:
                set_arm(spec, arm, fn=below_a1)
            self.assertEqual(select(spec)['decision'], expected, failing)
        spec = {}
        for arm in ('l1', 'l2'):
            set_arm(spec, arm, fn=below_a1)
        set_arm(spec, 'l4', usage=None)
        self.assertEqual(select(spec)['decision'], 'l3')

    def test_first_qualified_wins_even_if_a_later_arm_has_better_r2(self):
        spec = {}
        set_arm(spec, 'l1', r2=0.06)
        set_arm(spec, 'l2', r2=0.95)
        set_arm(spec, 'l3', r2=0.99)
        self.assertEqual(select(spec)['decision'], 'l1')

    def test_both_k_must_pass_for_every_criterion(self):
        defects = {
            'a1': below_a1, 'b': r2_zero, 'e': r2_low,
            'c': lambda v: {'norms': (1.0, 0.8, 0.8, 0.3)},
            'd': lambda v: {'usage_at': {24000: [JUST_BELOW[v], HEALTHY[v][1], HEALTHY[v][2]]}},
        }
        for name, fn in defects.items():
            for bad_variant in VARIANTS:
                spec = {}
                set_arm(spec, 'l2', usage=None)
                spec[(bad_variant, 'l1')] = fn(bad_variant)
                self.assertEqual(select(spec)['decision'], 'l2', f'{name} fails only on {bad_variant}')

    def test_a1_boundary_each_layer_both_k(self):
        for variant in VARIANTS:
            for layer in range(3):
                ok = list(HEALTHY[variant])
                ok[layer] = MIN_OK[variant]
                bad = list(HEALTHY[variant])
                bad[layer] = JUST_BELOW[variant]
                r_ok = select({(variant, 'l1'): {'usage': ok}})
                r_bad = select({(variant, 'l1'): {'usage': bad}})
                self.assertEqual(r_ok['decision'], 'l1', (variant, layer))
                self.assertTrue(criteria(r_ok, 'l1', variant)['a1'])
                self.assertNotEqual(r_bad['decision'], 'l1', (variant, layer))
                self.assertFalse(criteria(r_bad, 'l1', variant)['a1'])

    def test_a1_reads_the_final_canonical_usage_not_the_training_log(self):
        low = HEALTHY['T1'][:]
        low[1] = 204
        r = select({('T1', 'l1'): {'final_usage': low}})  # log stays healthy, final canonical is below
        self.assertFalse(criteria(r, 'l1', 'T1')['a1'])
        self.assertNotEqual(r['decision'], 'l1')
        r = select({('T1', 'l1'): {'usage_at': {30000: [204, 230, 230]}}})  # only the last log point is below
        self.assertTrue(criteria(r, 'l1', 'T1')['a1'])
        self.assertFalse(criteria(r, 'l1', 'T1')['d'])

    def test_b_r2_exactly_zero_fails_positive_passes(self):
        r = select({('T1', 'l1'): {'r2': 0.0}})
        self.assertFalse(criteria(r, 'l1', 'T1')['b'])
        r = select({('T1', 'l1'): {'r2': 1e-9}})
        self.assertTrue(criteria(r, 'l1', 'T1')['b'])
        r = select({('T1', 'l1'): {'r2': -0.2}})
        self.assertFalse(criteria(r, 'l1', 'T1')['b'])

    def test_e_r2_exactly_005_passes_just_below_fails(self):
        r = select({('T1', 'l1'): {'r2': 0.05}, ('T2', 'l1'): {'r2': 0.05}})
        self.assertTrue(criteria(r, 'l1', 'T1')['e'])
        self.assertEqual(r['decision'], 'l1')
        r = select({('T1', 'l1'): {'r2': 0.0499999}})
        self.assertFalse(criteria(r, 'l1', 'T1')['e'])
        self.assertTrue(criteria(r, 'l1', 'T1')['b'])
        self.assertNotEqual(r['decision'], 'l1')

    def test_c_equal_or_increasing_residual_norms_fail(self):
        for norms in ((1.0, 0.8, 0.8, 0.3), (1.0, 0.8, 0.5, 0.5), (1.0, 1.0, 0.5, 0.3), (1.0, 0.8, 0.5, 0.6)):
            r = select({('T2', 'l1'): {'norms': norms}})
            self.assertFalse(criteria(r, 'l1', 'T2')['c'], norms)
            self.assertNotEqual(r['decision'], 'l1')
        r = select({('T2', 'l1'): {'norms': (1.0, 0.8, 0.5, 0.4999)}})
        self.assertTrue(criteria(r, 'l1', 'T2')['c'])
        self.assertEqual(r['decision'], 'l1')

    def test_d_every_one_of_the_six_points_matters(self):
        for variant in VARIANTS:
            for step in (20000, 22000, 24000, 26000, 28000, 30000):
                low = [JUST_BELOW[variant]] + HEALTHY[variant][1:]
                r = select({(variant, 'l1'): {'usage_at': {step: low}}})
                self.assertFalse(criteria(r, 'l1', variant)['d'], (variant, step))
                self.assertNotEqual(r['decision'], 'l1')
                ok_edge = [MIN_OK[variant]] + HEALTHY[variant][1:]
                r = select({(variant, 'l1'): {'usage_at': {step: ok_edge}}})
                self.assertTrue(criteria(r, 'l1', variant)['d'], (variant, step))
                self.assertEqual(r['decision'], 'l1')

    def test_d_points_before_20000_are_ignored(self):
        for step in (2000, 10000, 16000, 18000):
            low = [JUST_BELOW['T1']] + HEALTHY['T1'][1:]
            r = select({('T1', 'l1'): {'usage_at': {step: low}}})
            self.assertTrue(criteria(r, 'l1', 'T1')['d'], step)
            self.assertEqual(r['decision'], 'l1')
        all_early_low = {s: [1, 1, 1] for s in range(2000, 19000, 2000)}
        r = select({('T1', 'l1'): {'usage_at': all_early_low}, ('T2', 'l1'): {'usage_at': {
            s: [1, 1, 1] for s in range(2000, 19000, 2000)}}})
        self.assertEqual(r['decision'], 'l1')

    def test_d_checks_every_layer_of_the_point(self):
        for layer in range(3):
            low = list(HEALTHY['T1'])
            low[layer] = 204
            r = select({('T1', 'l1'): {'usage_at': {26000: low}}})
            self.assertFalse(criteria(r, 'l1', 'T1')['d'], layer)

    def test_a2_requires_strict_rise_and_more_than_p0_final_usage(self):
        def layer2(v20, v30):  # layers 0 and 1 pass a1; layer 2 (K=256) fails a1 with final usage 2
            return {('T1', 'l2'): {'final_usage': [230, 230, 2], 'usage': [230, 230, 2],
                                   'usage_at': {20000: [230, 230, v20], 30000: [230, 230, v30]}}}
        for v20, v30, expected in ((2, 2, False), (2, 3, False), (2, 4, True), (3, 3, False), (3, 4, True),
                                   (4, 4, False), (5, 4, False), (5, 6, True), (100, 101, True), (100, 100, False),
                                   (100, 99, False)):
            r = select(layer2(v20, v30))
            self.assertEqual(criteria(r, 'l2', 'T1')['a2'], expected, (v20, v30))

    def test_a2_needs_every_failing_layer_to_rise(self):
        def two_layers(at30000):
            return {('T1', 'l2'): {'final_usage': [100, 100, 230], 'usage': [100, 100, 230],
                                   'usage_at': {20000: [100, 100, 230], 30000: at30000}}}
        self.assertFalse(criteria(select(two_layers([101, 100, 230])), 'l2', 'T1')['a2'])
        self.assertFalse(criteria(select(two_layers([100, 101, 230])), 'l2', 'T1')['a2'])
        self.assertTrue(criteria(select(two_layers([101, 101, 230])), 'l2', 'T1')['a2'])

    def test_a2_does_not_constrain_layers_that_already_pass_a1(self):
        # layer 0 fails a1 and rises; layers 1 and 2 pass a1 and stay flat
        spec = {('T1', 'l2'): {'final_usage': [100, 230, 230], 'usage': [100, 230, 230],
                               'usage_at': {20000: [100, 230, 230], 30000: [150, 230, 230]}}}
        self.assertTrue(criteria(select(spec), 'l2', 'T1')['a2'])
        # a1 satisfied on all layers: a2 is vacuously true even when nothing rises
        self.assertTrue(criteria(select(), 'l1', 'T1')['a2'])

    def test_a2_is_not_a_qualification_condition(self):
        # canonical final usage fails a1 on layer 0, the training log is healthy everywhere (d holds) and layer 0
        # rises from 20000 to 30000 (a2 holds): every other criterion passes, yet the arm must not qualify
        spec = {('T1', 'l1'): {'final_usage': [100, 230, 230], 'usage': [230, 230, 230],
                               'usage_at': {20000: [205, 230, 230], 30000: [230, 230, 230]}}}
        r = select(spec)
        c = criteria(r, 'l1', 'T1')
        self.assertEqual((c['a1'], c['a2'], c['b'], c['c'], c['d'], c['e']), (False, True, True, True, True, True))
        self.assertEqual(r['decision'], 'l2')

    def test_a2_only_decides_escalate_versus_stop(self):
        def climbing(v):
            return {'final_usage': [100, 230, 230], 'usage': [100, 230, 230],
                    'usage_at': {20000: [100, 230, 230], 30000: [120, 230, 230]}} if v == 'T1' else {}

        def flat(v):
            return {'final_usage': [100, 230, 230], 'usage': [100, 230, 230]} if v == 'T1' else {}
        for fn, expected in ((climbing, ('ESCALATE', ['l1'])), (flat, ('STOP', []))):
            spec = {}
            set_arm(spec, 'l1', fn=fn)
            for arm in ('l2', 'l3', 'l4'):
                set_arm(spec, arm, fn=below_a1)
            r = select(spec)
            self.assertEqual((r['decision'], r['escalate_listed_arms']), expected)

    def test_p0_loss_window_tolerance(self):
        for variant in VARIANTS:
            ref = P0_LOSS[variant]
            for window in range(3):
                head = list(ref)
                head[window] = ref[window] + 0.0009
                r = select({(variant, 'p0'): {'loss_head': head}})
                self.assertEqual(r['decision'], 'l1', (variant, window, 'inside'))
                head[window] = ref[window] - 0.0011
                r = select({(variant, 'p0'): {'loss_head': head}})
                self.assertEqual(r['decision'], 'INVALID_CONTROL', (variant, window, 'outside'))

    def test_p0_usage_log_point_must_stay_within_old_record_plus_slack(self):
        for point in USAGE_STEPS:
            r = select({('T1', 'p0'): {'usage_at': {point: [2, 2, 3]}}})
            self.assertEqual(r['decision'], 'l1', point)
            for bad in ([3, 1, 2], [1, 3, 2], [1, 1, 4]):
                r = select({('T2', 'p0'): {'usage_at': {point: bad}}})
                self.assertEqual(r['decision'], 'INVALID_CONTROL', (point, bad))

    def test_p0_final_usage_at_most_3_per_layer(self):
        r = select({('T1', 'p0'): {'final_usage': [3, 3, 3]}})
        self.assertEqual(r['decision'], 'l1')
        for bad in ([4, 1, 2], [1, 4, 2], [1, 1, 4]):
            r = select({('T2', 'p0'): {'final_usage': bad}})
            self.assertEqual(r['decision'], 'INVALID_CONTROL', bad)

    def test_p0_non_complete_on_either_k_is_invalid_control_even_if_the_prefix_matches(self):
        for variant in VARIANTS:
            for status in ('fail_nonfinite', 'fail_wall', 'fail_wall_projected'):
                r = select({(variant, 'p0'): {'status': status}})
                self.assertEqual(r['decision'], 'INVALID_CONTROL', (variant, status))
                self.assertEqual(r['escalate_listed_arms'], [])
                self.assertIsNone(r['preflight'])

    def test_invalid_control_beats_a_qualified_arm(self):
        r = select({('T1', 'p0'): {'status': 'fail_wall'}})
        self.assertEqual(r['decision'], 'INVALID_CONTROL')
        self.assertTrue(all(criteria(r, 'l1', v)['a1'] for v in VARIANTS))

    def test_non_complete_l_arm_is_unqualified_with_null_criteria(self):
        for status in ('fail_nonfinite', 'fail_wall'):
            for variant in VARIANTS:
                r = select({(variant, 'l1'): {'status': status}})
                self.assertEqual(r['decision'], 'l2', (variant, status))
                c = criteria(r, 'l1', variant)
                self.assertEqual(c, {'status': status, 'a1': None, 'a2': None, 'b': None, 'c': None, 'd': None,
                                     'e': None})
                self.assertEqual(criteria(r, 'l1', 'T2' if variant == 'T1' else 'T1')['status'], 'complete')

    def test_non_complete_l_arm_is_never_an_escalate_candidate(self):
        # l1 would be an escalate candidate on T1, but is non-complete on T2 -> not listed, nothing else -> STOP
        spec = {('T1', 'l1'): r2_low('T1'), ('T2', 'l1'): {'status': 'fail_wall'}}
        for arm in ('l2', 'l3', 'l4'):
            set_arm(spec, arm, fn=below_a1)
        r = select(spec)
        self.assertEqual((r['decision'], r['escalate_listed_arms']), ('STOP', []))
        # with l2 a genuine candidate, only l2 is listed
        spec = {('T1', 'l1'): {'status': 'fail_nonfinite'}, ('T2', 'l1'): {'status': 'fail_nonfinite'}}
        set_arm(spec, 'l2', fn=r2_low)
        set_arm(spec, 'l3', fn=below_a1)
        set_arm(spec, 'l4', fn=below_a1)
        r = select(spec)
        self.assertEqual((r['decision'], r['escalate_listed_arms']), ('ESCALATE', ['l2']))

    def test_escalate_when_only_a1_with_a2_d_or_e_fail(self):
        # "l1 collapses, l2 fails only e"
        spec = {}
        set_arm(spec, 'l1', fn=r2_zero)
        set_arm(spec, 'l2', fn=r2_low)
        set_arm(spec, 'l3', fn=below_a1)
        set_arm(spec, 'l4', fn=below_a1)
        r = select(spec)
        self.assertEqual((r['decision'], r['escalate_listed_arms']), ('ESCALATE', ['l2']))
        # only d fails
        spec = {}
        for arm in L_ARMS:
            set_arm(spec, arm, fn=lambda v: {'usage_at': {22000: [JUST_BELOW[v], HEALTHY[v][1], HEALTHY[v][2]]}})
        r = select(spec)
        self.assertEqual((r['decision'], r['escalate_listed_arms']), ('ESCALATE', list(L_ARMS)))

    def test_escalate_for_an_arm_still_climbing_at_30000(self):
        def climbing(v):
            K = KS[v]
            final = [int(K * 0.7)] * 3
            return {'usage': final, 'final_usage': final,
                    'usage_at': {20000: [int(K * 0.6)] * 3, 30000: [int(K * 0.7)] * 3}}
        spec = {}
        set_arm(spec, 'l1', fn=r2_zero)
        set_arm(spec, 'l2', fn=climbing)
        set_arm(spec, 'l3', fn=r2_zero)
        set_arm(spec, 'l4', fn=r2_zero)
        r = select(spec)
        self.assertEqual((r['decision'], r['escalate_listed_arms']), ('ESCALATE', ['l2']))
        self.assertTrue(criteria(r, 'l2', 'T1')['a2'] and not criteria(r, 'l2', 'T1')['a1'])

    def test_slow_arm_with_a_layer_neither_at_80_percent_nor_rising_is_stop(self):
        def slow(v):
            K = KS[v]
            low = [int(K * 0.7)] * 3
            return {'usage': low, 'final_usage': low, 'usage_at': {20000: [int(K * 0.7), int(K * 0.7), int(K * 0.8)],
                                                                    30000: [int(K * 0.7)] * 3}}
        spec = {}
        for arm in L_ARMS:
            set_arm(spec, arm, fn=slow)
        r = select(spec)
        self.assertEqual((r['decision'], r['escalate_listed_arms']), ('STOP', []))

    def test_b_or_c_failure_without_any_other_candidate_is_stop(self):
        for fn in (r2_zero, lambda v: {'norms': (1.0, 0.8, 0.8, 0.3)}):
            spec = {}
            for arm in L_ARMS:
                set_arm(spec, arm, fn=fn)
            r = select(spec)
            self.assertEqual((r['decision'], r['escalate_listed_arms']), ('STOP', []))

    def test_b_or_c_failure_removes_an_arm_from_the_candidates(self):
        spec = {}
        set_arm(spec, 'l1', fn=lambda v: {'r2': 0.0})
        set_arm(spec, 'l2', fn=lambda v: {'norms': (1.0, 0.8, 0.8, 0.3), 'r2': 0.01})
        set_arm(spec, 'l3', fn=r2_low)
        set_arm(spec, 'l4', fn=below_a1)
        r = select(spec)
        self.assertEqual((r['decision'], r['escalate_listed_arms']), ('ESCALATE', ['l3']))

    def test_candidate_list_keeps_the_fixed_order(self):
        spec = {}
        set_arm(spec, 'l1', fn=r2_zero)
        set_arm(spec, 'l2', fn=r2_low)
        set_arm(spec, 'l3', fn=r2_low)
        set_arm(spec, 'l4', fn=r2_low)
        r = select(spec)
        self.assertEqual(r['escalate_listed_arms'], ['l2', 'l3', 'l4'])

    def test_a_candidate_needs_both_k(self):
        spec = {('T1', 'l1'): r2_low('T1'), ('T2', 'l1'): below_a1('T2')}
        for arm in ('l2', 'l3', 'l4'):
            set_arm(spec, arm, fn=r2_zero)
        r = select(spec)  # l1 on T2: a1 fails, a2 false (flat) -> not a candidate
        self.assertEqual((r['decision'], r['escalate_listed_arms']), ('STOP', []))

    def test_per_arm_per_k_reports_every_criterion(self):
        r = select()
        for arm in L_ARMS:
            for variant in VARIANTS:
                self.assertEqual(criteria(r, arm, variant),
                                 {'status': 'complete', 'a1': True, 'a2': True, 'b': True, 'c': True, 'd': True,
                                  'e': True})

    def test_pure_function_does_not_mutate_inputs_and_is_repeatable(self):
        probes = make_set({('T1', 'l1'): r2_low('T1')})
        snapshot = copy.deepcopy(probes)
        first = rr.select_arm(probes)
        second = rr.select_arm(copy.deepcopy(probes))
        self.assertEqual(probes, snapshot)
        self.assertEqual(first, second)
        self.assertEqual(rr.select_arm(json.loads(json.dumps(probes))), first)

    def test_p0_control_details_are_reported(self):
        r = select()
        self.assertEqual(sorted(r['p0_control']), ['T1', 'T2'])
        self.assertTrue(all(c['valid'] for c in r['p0_control'].values()))
        r = select({('T2', 'p0'): {'final_usage': [1, 1, 4]}})
        self.assertTrue(r['p0_control']['T1']['valid'])
        self.assertFalse(r['p0_control']['T2']['valid'])
        self.assertFalse(r['p0_control']['T2']['final'])


class Preflight(unittest.TestCase):
    def patched(self, t1=1000.0, t2=100.0):
        return mock.patch.dict(rr.PREREG_R_SELECT, {'old_fit_wall_seconds': {'T1': t1, 'T2': t2}})

    def spec(self, l1_ms, p0_ms=1.0):
        spec = {}
        set_arm(spec, 'p0', ms=p0_ms)
        set_arm(spec, 'l1', ms=l1_ms)
        return spec

    def test_ratio_and_projected_seconds(self):
        r = select(set_arm(set_arm({}, 'p0', ms=4.0), 'l1', ms=6.0))
        pre = r['preflight']
        self.assertEqual(pre['T1'], {'overhead_ratio': 1.5, 'projected_fit_seconds': 4354 * 1.5})
        self.assertEqual(pre['T2'], {'overhead_ratio': 1.5, 'projected_fit_seconds': 4302 * 1.5})
        self.assertTrue(pre['exceeds'])  # 6531 s > 6000 s
        same = select(set_arm(set_arm({}, 'p0', ms=4.0), 'l1', ms=4.0))['preflight']
        self.assertEqual(same['T1'], {'overhead_ratio': 1.0, 'projected_fit_seconds': 4354.0})
        self.assertFalse(same['exceeds'])

    def test_uses_the_selected_arm_and_the_same_k_p0(self):
        spec = {('T1', 'p0'): {'ms': 2.0}, ('T2', 'p0'): {'ms': 5.0}, ('T1', 'l1'): {'ms': 3.0},
                ('T2', 'l1'): {'ms': 20.0}, ('T1', 'l2'): {'ms': 99.0}, ('T2', 'l2'): {'ms': 99.0}}
        pre = select(spec)['preflight']
        self.assertEqual(pre['T1']['overhead_ratio'], 1.5)
        self.assertEqual(pre['T2']['overhead_ratio'], 4.0)

    def test_boundary_6000_does_not_exceed_6000_01_does(self):
        with self.patched():
            at = select(self.spec(6.0))['preflight']
            self.assertEqual(at['T1']['projected_fit_seconds'], 6000.0)
            self.assertFalse(at['exceeds'])
            over = select(self.spec(6.00001))['preflight']
            self.assertGreater(over['T1']['projected_fit_seconds'], 6000.0)
            self.assertTrue(over['exceeds'])

    def test_exceeds_on_either_k(self):
        with self.patched(t1=100.0, t2=1000.0):
            spec = self.spec(6.0)
            self.assertFalse(select(spec)['preflight']['exceeds'])
        with self.patched(t1=100.0, t2=1000.0):
            spec = {('T1', 'p0'): {'ms': 1.0}, ('T2', 'p0'): {'ms': 1.0}, ('T1', 'l1'): {'ms': 1.0},
                    ('T2', 'l1'): {'ms': 6.00001}}
            self.assertTrue(select(spec)['preflight']['exceeds'])

    def test_exceeds_never_changes_the_decision(self):
        cheap = select(self.spec(1.0))
        costly = select(self.spec(1000.0))
        self.assertTrue(costly['preflight']['exceeds'])
        self.assertFalse(cheap['preflight']['exceeds'])
        self.assertEqual(cheap['decision'], 'l1')
        self.assertEqual(costly['decision'], 'l1')
        self.assertEqual(cheap['per_arm_per_K'], costly['per_arm_per_K'])

    def test_preflight_is_null_for_invalid_control_and_stop(self):
        invalid = select({('T1', 'p0'): {'status': 'fail_wall'}})
        stop = {}
        for arm in L_ARMS:
            set_arm(stop, arm, fn=below_a1)
        self.assertEqual(invalid['decision'], 'INVALID_CONTROL')
        self.assertEqual(select(stop)['decision'], 'STOP')
        for res in (invalid, select(stop)):
            self.assertIsNone(res['preflight'])

    def test_preflight_is_null_for_escalate(self):
        spec = {}
        for arm in L_ARMS:
            set_arm(spec, arm, fn=r2_low)
        r = select(spec)
        self.assertEqual(r['decision'], 'ESCALATE')
        self.assertIsNone(r['preflight'])


class Override(unittest.TestCase):
    def escalate_spec(self):
        spec = {}
        set_arm(spec, 'l1', fn=r2_zero)
        set_arm(spec, 'l2', fn=r2_low)
        set_arm(spec, 'l3', fn=r2_low)
        set_arm(spec, 'l4', fn=below_a1)
        return spec

    def test_listed_arm_with_source_is_accepted_and_records_everything(self):
        mech = select(self.escalate_spec())
        self.assertEqual((mech['decision'], mech['escalate_listed_arms']), ('ESCALATE', ['l2', 'l3']))
        r = select(self.escalate_spec(), override_arm='l3', override_source='2026-10-16 09:00 decision entry')
        self.assertEqual(r['decision'], 'l3')
        self.assertEqual(r['override'], {'arm': 'l3', 'source': '2026-10-16 09:00 decision entry',
                                         'escalate_listed_arms': ['l2', 'l3']})
        self.assertEqual(r['escalate_listed_arms'], mech['escalate_listed_arms'])
        self.assertEqual(r['per_arm_per_K'], mech['per_arm_per_K'])
        self.assertIsNotNone(r['preflight'])
        self.assertEqual(set(r['preflight']), {'T1', 'T2', 'exceeds'})

    def test_every_listed_arm_can_be_released(self):
        for arm in ('l2', 'l3'):
            self.assertEqual(select(self.escalate_spec(), override_arm=arm, override_source='s')['decision'], arm)

    def test_arm_outside_the_candidates_is_refused(self):
        for arm in ('l1', 'l4', 'p0', 'L2', 'STOP', ''):
            with self.assertRaises(ValueError, msg=arm):
                select(self.escalate_spec(), override_arm=arm, override_source='s')

    def test_both_options_are_required_and_source_non_empty(self):
        with self.assertRaises(ValueError):
            select(self.escalate_spec(), override_arm='l2')
        with self.assertRaises(ValueError):
            select(self.escalate_spec(), override_source='s')
        for source in ('', '   ', None, 5):
            with self.assertRaises(ValueError, msg=repr(source)):
                select(self.escalate_spec(), override_arm='l2', override_source=source)

    def test_refused_when_the_recomputation_is_not_escalate(self):
        stop = {}
        for arm in L_ARMS:
            set_arm(stop, arm, fn=below_a1)
        with self.assertRaises(ValueError):
            select(stop, override_arm='l2', override_source='s')
        with self.assertRaises(ValueError):  # mechanical result is a qualified arm
            select(None, override_arm='l2', override_source='s')
        with self.assertRaises(ValueError):
            select({('T1', 'p0'): {'status': 'fail_wall'}}, override_arm='l1', override_source='s')

    def test_override_uses_a_fresh_recomputation_not_trusted_input(self):
        probes = make_set(self.escalate_spec())
        probes_changed = copy.deepcopy(probes)
        for p in probes_changed:
            if p['arm'] == 'l2':
                p['final']['r2'] = 0.5  # now l2 qualifies -> mechanical result l2, not ESCALATE
        with self.assertRaises(ValueError):
            rr.select_arm(probes_changed, override_arm='l3', override_source='s')


def sha_of(probe):
    return hashlib.sha256(json.dumps(probe, sort_keys=True).encode()).hexdigest()


class DecisionSchema(unittest.TestCase):
    def build(self, spec=None, **kw):
        probes = make_set(spec)
        return rr.build_decision(probes, [sha_of(p) for p in probes], config_sha256=CONFIG_SHA, **kw)

    def test_built_decisions_validate_for_every_outcome(self):
        stop = {}
        for arm in L_ARMS:
            set_arm(stop, arm, fn=below_a1)
        esc = {}
        for spec, expected in ((None, 'l1'), (stop, 'STOP'), ({('T1', 'p0'): {'status': 'fail_wall'}}, 'INVALID_CONTROL')):
            d = self.build(spec)
            self.assertEqual(d['decision'], expected)
            rr.validate_decision(d)
            rr.validate_decision(json.loads(json.dumps(d)))
        for arm in L_ARMS:
            set_arm(esc, arm, fn=r2_low)
        d = self.build(esc)
        self.assertEqual(d['decision'], 'ESCALATE')
        rr.validate_decision(d)
        d = self.build(esc, override_arm='l2', override_source='entry')
        self.assertEqual(d['decision'], 'l2')
        rr.validate_decision(d)

    def test_fixed_fields_and_probe_references(self):
        d = self.build()
        self.assertEqual(set(d), {'schema', 'config_sha256', 'catalog_sha256', 'content_file_sha256',
                                  'content_array_sha256', 'probe_results', 'per_arm_per_K', 'decision',
                                  'escalate_listed_arms', 'override', 'preflight'})
        self.assertEqual(d['schema'], 'gtok01r-decision-v1')
        self.assertEqual((d['config_sha256'], d['catalog_sha256'], d['content_file_sha256'],
                          d['content_array_sha256']), (CONFIG_SHA, CATALOG_SHA, CONTENT_FILE_SHA, CONTENT_ARRAY_SHA))
        pairs = [(e['variant'], e['arm']) for e in d['probe_results']]
        self.assertEqual(sorted(pairs), sorted((v, a) for v in VARIANTS for a in ARMS))
        self.assertEqual(len(set(pairs)), 10)
        for e in d['probe_results']:
            self.assertEqual(set(e), {'variant', 'arm', 'sha256', 'code_commit'})
            self.assertEqual(e['code_commit'], COMMIT)
        self.assertNotIn('p0_control', d)

    def test_build_decision_refuses_inconsistent_inputs(self):
        probes = make_set()
        shas = [sha_of(p) for p in probes]
        with self.assertRaises(ValueError):
            rr.build_decision(probes, shas[:-1], config_sha256=CONFIG_SHA)
        with self.assertRaises(ValueError):
            rr.build_decision(probes, shas[:-1] + ['XYZ'], config_sha256=CONFIG_SHA)
        with self.assertRaises(ValueError):
            rr.build_decision(probes, shas, config_sha256='9' * 64)
        with self.assertRaises(ValueError):
            rr.build_decision(probes[:-1], shas[:-1], config_sha256=CONFIG_SHA)

    def mutate(self, fn):
        d = self.build()
        fn(d)
        with self.assertRaises(ValueError):
            rr.validate_decision(d)

    def test_rejects_missing_extra_and_type_errors(self):
        for key in list(self.build()):
            self.mutate(lambda d, key=key: d.pop(key))
        self.mutate(lambda d: d.update(extra=1))
        self.mutate(lambda d: d.update(schema='gtok01r-decision-v0'))
        self.mutate(lambda d: d.update(config_sha256='C' * 64))
        self.mutate(lambda d: d.update(catalog_sha256=None))
        self.mutate(lambda d: d.update(decision='L1'))
        self.mutate(lambda d: d.update(decision='go'))
        self.mutate(lambda d: d.update(escalate_listed_arms='l2'))
        self.mutate(lambda d: d.update(override=1))
        self.mutate(lambda d: d.update(preflight=None))  # decision l1 requires preflight
        self.mutate(lambda d: d.update(preflight={'T1': {}, 'T2': {}, 'exceeds': False}))
        self.mutate(lambda d: d['preflight'].update(exceeds='no'))
        self.mutate(lambda d: d['preflight']['T1'].update(overhead_ratio='x'))
        self.mutate(lambda d: d.update(per_arm_per_K=[]))
        self.mutate(lambda d: d['per_arm_per_K'].pop('l3'))
        self.mutate(lambda d: d['per_arm_per_K'].update(L1=d['per_arm_per_K']['l1']))
        self.mutate(lambda d: d['per_arm_per_K']['l1'].pop('T2'))
        self.mutate(lambda d: d['per_arm_per_K']['l1']['T1'].update(a1='yes'))
        self.mutate(lambda d: d['per_arm_per_K']['l1']['T1'].update(zz=True))
        self.mutate(lambda d: d['per_arm_per_K']['l1']['T1'].update(status='fail_wall'))  # criteria must be null then
        self.mutate(lambda d: d['per_arm_per_K']['l1']['T1'].update(status='weird'))

    def test_rejects_probe_reference_errors(self):
        self.mutate(lambda d: d['probe_results'].pop())
        self.mutate(lambda d: d['probe_results'].append(dict(d['probe_results'][0])))
        self.mutate(lambda d: d['probe_results'][0].update(arm='L1'))
        self.mutate(lambda d: d['probe_results'][0].update(sha256='zz'))
        self.mutate(lambda d: d['probe_results'][0].update(code_commit='b' * 40))
        self.mutate(lambda d: d['probe_results'][1].update(variant=d['probe_results'][0]['variant'],
                                                         arm=d['probe_results'][0]['arm']))
        self.mutate(lambda d: d['probe_results'][0].update(extra=1))

    def test_override_and_listed_arm_consistency(self):
        esc = {}
        for arm in L_ARMS:
            set_arm(esc, arm, fn=r2_low)
        released = lambda: self.build(esc, override_arm='l2', override_source='entry')

        def check(fn):
            d = released()
            fn(d)
            with self.assertRaises(ValueError):
                rr.validate_decision(d)
        check(lambda d: d['override'].update(arm='l3'))
        check(lambda d: d['override'].update(source=''))
        check(lambda d: d['override'].pop('source'))
        check(lambda d: d['override'].update(escalate_listed_arms=['l2']))
        check(lambda d: d.update(escalate_listed_arms=[]))
        check(lambda d: (d['escalate_listed_arms'].remove('l2'),
                         d['override'].update(escalate_listed_arms=list(d['escalate_listed_arms']))))
        d = self.build(esc)  # ESCALATE without listed arms is invalid
        d['escalate_listed_arms'] = []
        with self.assertRaises(ValueError):
            rr.validate_decision(d)
        d = self.build()  # qualified arm must not list candidates
        d['escalate_listed_arms'] = ['l2']
        with self.assertRaises(ValueError):
            rr.validate_decision(d)

    def test_preflight_must_be_null_for_non_l_decisions(self):
        stop = {}
        for arm in L_ARMS:
            set_arm(stop, arm, fn=below_a1)
        d = self.build(stop)
        self.assertEqual(d['decision'], 'STOP')
        self.assertIsNone(d['preflight'])
        d['preflight'] = self.build()['preflight']
        with self.assertRaises(ValueError):
            rr.validate_decision(d)


if __name__ == '__main__':
    unittest.main()
