"""GTOK-02: mutation suite. Every mutant must be killed by an ASSERTION FAILURE of the named tests.

``assert_killed`` runs the named tests under a patch and requires ``result.failures`` to be non-empty. A mutant that only makes the tests
crash (``errors`` without ``failures``) does not count as killed, and a patch aimed at a name that does not exist raises instead of
silently passing; both are checked at the bottom of this file.

Inline expressions cannot be patched by name, so ``src`` rebuilds one function from its own source text with a single literal replacement
and the patch swaps the rebuilt function in. All shared fixtures are built once in ``setUpClass``, before any mutant is active.
"""
import inspect
import io
import textwrap
import unittest
from unittest import mock

import numpy as np

from tests import test_gtok02_formal_path as formal
from tests import test_gtok02_old_entries as old
from tests import test_gtok02_prereg as prereg
from tests import test_gtok02_select_curve as sc
from tests import test_sid_tok2_run as t2
from tests import test_sid_v2s as t1
from vkgr import generative_tok2_run as tok2
from vkgr import sid_tok_audit as au
from vkgr import sid_tok2_run as run
from vkgr import sid_v2s as s2

V2S = 'tests.test_sid_v2s.'
RUN = 'tests.test_sid_tok2_run.'
FORMAL = 'tests.test_gtok02_formal_path.'
SC = 'tests.test_gtok02_select_curve.'
OLD = 'tests.test_gtok02_old_entries.'


def run_ids(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ids)
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


def assert_killed(case, patch_target, replacement, test_ids):
    """patch_target: 'package.module.name' or (object, 'name'). Passes only when an assertion of the named tests fails."""
    patcher = mock.patch.object(*patch_target, replacement) if isinstance(patch_target, tuple) else mock.patch(patch_target, replacement)
    with patcher:
        result = run_ids(test_ids)
    case.assertGreater(result.testsRun, 0)
    case.assertFalse(result.wasSuccessful(), f'mutant survived: {patch_target}')
    case.assertTrue(result.failures, f'mutant only crashed the tests (errors without a failed assertion): {patch_target}')
    return result


def src(function, old, new, *more):
    """The function rebuilt from its source with literal replacements (pairs: old, new, old, new, ...; every anchor must exist)."""
    text = textwrap.dedent(inspect.getsource(function))
    pairs = [(old, new)] + list(zip(more[::2], more[1::2]))
    for anchor, replacement in pairs:
        if anchor not in text:
            raise AssertionError(f'mutation anchor not found in {function.__name__}: {anchor!r}')
        text = text.replace(anchor, replacement, 1)
    namespace = dict(function.__globals__)
    exec(compile(text, f'<mutant {function.__name__}>', 'exec'), namespace)
    return namespace[function.__name__]


GATE_KILLERS = [RUN + 'GateTests.test_each_gate_triggers_on_its_own_violation']
KILLERS = {
    'registry': [V2S + 'RegistryTests.test_six_labels_with_expected_spec_in_both_contexts',
                 FORMAL + 'RegistrationTests.test_every_new_label_is_registered_in_both_tables_with_its_spec'],
    'spec_engineering': [V2S + 'RegistryTests.test_replay_label_is_engineering_only'],
    'pad': [V2S + 'DepthTwoTests.test_pad_depth2_puts_zero_in_third_and_rank_in_fourth',
            V2S + 'DepthTwoTests.test_derive_keeps_the_parent_partition_and_numbers_by_item_id'],
    'pad_check': [V2S + 'FailClosedTests.test_depth2_specific_rejections', V2S + 'WriteRefusalTests.test_inconsistent_inputs_are_refused_before_creating_anything'],
    'b1_basis': [RUN + 'GateTests.test_depth2_balance_uses_the_three_unpadded_columns'],
    'depth_check': [V2S + 'FailClosedTests.test_manifest_key_set_and_values'],
    'derive': [V2S + 'DepthTwoTests.test_derive_keeps_the_parent_partition_and_numbers_by_item_id'],
    'derive_sorted': [V2S + 'DepthTwoTests.test_derive_rejects_unsorted_or_duplicate_ids'],
    'select': [SC + 'SelectTests.test_design_example', SC + 'SelectTests.test_near_boundary_is_ten_hits_inclusive',
               SC + 'SelectTests.test_ties_use_ndcg_then_enumeration_order', SC + 'SelectTests.test_rule_e_does_not_fire_when_b256d2_is_held_or_unverified',
               SC + 'SelectTests.test_k_can_be_zero'],
    'cer': [SC + 'CollisionRateTests.test_known_points', SC + 'CollisionRateTests.test_cer_is_not_computed_from_the_largest_leaf'],
    'validate': [FORMAL + 'CommittedConfigTests.test_the_formal_validation_refuses_everything_else'],
    'expected': [RUN + 'CheckExpectedTests.test_any_single_integer_off_by_one_is_named', RUN + 'CheckExpectedTests.test_there_is_no_tolerance'],
    'exit4': [RUN + 'CliTests.test_expected_mismatch_exits_4_and_does_not_touch_the_bundle'],
    'expected_for': [RUN + 'VerifyTests.test_expected_comparison_is_skipped_in_engineering_and_reported_in_formal_hook'],
    'rederive': [RUN + 'VerifyTests.test_depth2_rederivation_is_bit_exact'],
    'replay': [RUN + 'VerifyTests.test_replay_fit_catches_a_mapping_that_is_not_the_fit_output'],
    'binding': [RUN + 'VerifyTests.test_wrong_config_sha_or_tokenizer_record_is_rejected'],
    'q1': [SC + 'Q1Tests.test_all_36_combinations', SC + 'Q1Tests.test_priorities_are_not_swapped', SC + 'Q1Tests.test_rows_are_reported'],
    'state': [SC + 'StateTests.test_state_needs_both_seeds_beyond_the_threshold', SC + 'StateTests.test_tentative_state_uses_the_same_dead_zone'],
    'old': [OLD + 'OldEntryPins.test_every_pinned_file_is_byte_identical_to_the_base'],
}


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for label in t1.NEW_LABELS:
            t1.sweep_world(label)
        t1.collision_world()
        for variant in t2.VARIANTS:
            t2.fitted(variant)
        t2.synthetic_inputs()

    def test_unmutated_selection_is_green(self):
        ids = sorted({name for names in KILLERS.values() for name in names} | set(GATE_KILLERS))
        result = run_ids(ids)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertGreaterEqual(result.testsRun, len(ids))

    # --- 1 registry -------------------------------------------------------------------------------------------------------------
    def test_M01_registry_misses_one_label(self):
        for label in s2.SID_VERSIONS:
            with self.subTest(label):
                reduced = tuple(x for x in s2.SID_VERSIONS if x != label)
                assert_killed(self, 'vkgr.sid_v2s.SID_VERSIONS', reduced, KILLERS['registry'])

    def test_M02_spec_ignores_the_engineering_flag(self):
        original = s2._spec
        assert_killed(self, 'vkgr.sid_v2s._spec', lambda label, engineering: original(label, True), KILLERS['spec_engineering'])

    # --- 3, 4 padding -----------------------------------------------------------------------------------------------------------
    def test_M03_pad_goes_to_the_last_position(self):
        def wrong(raw3):
            out = np.zeros((len(raw3), 4), dtype=np.uint32)
            out[:, 0], out[:, 1], out[:, 2] = raw3[:, 0], raw3[:, 1], raw3[:, 2]
            return out

        assert_killed(self, 'vkgr.sid_v2s.pad_depth2', wrong, KILLERS['pad'])

    def test_M03b_pad_digit_is_not_zero(self):
        def wrong(raw3):
            out = np.zeros((len(raw3), 4), dtype=np.uint32)
            out[:, 0], out[:, 1], out[:, 2], out[:, 3] = raw3[:, 0], raw3[:, 1], 1, raw3[:, 2]
            return out

        assert_killed(self, 'vkgr.sid_v2s.pad_depth2', wrong, KILLERS['pad'])

    def test_M04_loader_accepts_a_non_zero_pad_column(self):
        assert_killed(self, 'vkgr.sid_v2s._check_padding', lambda sids: None, KILLERS['pad_check'])

    # --- 5 balance basis --------------------------------------------------------------------------------------------------------
    def test_M05_depth2_balance_is_computed_on_the_padded_columns(self):
        assert_killed(self, 'vkgr.sid_v2s.unpad_depth2', lambda sids: sids, KILLERS['b1_basis'])

    # --- 6 gates ----------------------------------------------------------------------------------------------------------------
    def test_M06_gates_sweep_skips_one_gate(self):
        original = run.gates_sweep
        for gate in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'B1'):
            def skipping(audit, *, expected_content_sha256=au.CONTENT_V1_FEATURE_SHA256, _gate=gate):
                result = original(audit, expected_content_sha256=expected_content_sha256)
                checks = [{**c, 'ok': True} if c['id'] == _gate else c for c in result['checks']]
                return {'passed': all(c['ok'] for c in checks), 'checks': checks}

            with self.subTest(gate):
                assert_killed(self, 'vkgr.sid_tok2_run.gates_sweep', skipping, GATE_KILLERS)

    # --- 7 depth field ----------------------------------------------------------------------------------------------------------
    def test_M07_loader_ignores_the_manifest_depth(self):
        mutant = src(s2._load_bundle, " or type(manifest['depth']) is not int or manifest['depth'] != depth", '')
        assert_killed(self, 'vkgr.sid_v2s._load_bundle', mutant, KILLERS['depth_check'])

    # --- 8 selection ------------------------------------------------------------------------------------------------------------
    def test_M08_selection_rule_mutants(self):
        original = sc.sel.select
        for name, old_text, new_text in (
                ('near is exclusive', "r['hits'] >= best - NEAR_HITS", "r['hits'] > best - NEAR_HITS"),
                ('near is 11', "r['hits'] >= best - NEAR_HITS", "r['hits'] >= best - 11"),
                ('top1 instead of top2', 'ranked[:2]', 'ranked[:1]'),
                ('ties: reverse enumeration', "-r['ndcg50'], ORDER.index(r['label'])", "-r['ndcg50'], -ORDER.index(r['label'])"),
                ('ties: no ndcg', "-r['ndcg50'], ", ''),
                ('no rule E', "rule_e = any(r['label'] == 'B256D2' for r in ranked)", 'rule_e = False'),
                ('rule E without verification', "rule_e = any(r['label'] == 'B256D2' for r in ranked)", "rule_e = 'B256D2' in results"),
                ('old configs get seeds', 'if label in NEW]', 'if True]')):
            with self.subTest(name):
                assert_killed(self, (sc.sel, 'select'), src(original, old_text, new_text), KILLERS['select'])

    # --- 9 CER ------------------------------------------------------------------------------------------------------------------
    def test_M09_cer_uses_the_largest_leaf(self):
        original = sc.curve.collision_rates
        mutant = src(original, "needing = audit['items_needing_suffix']", "needing = audit['leaves']['max']")
        assert_killed(self, (sc.curve, 'collision_rates'), mutant, KILLERS['cer'])

    def test_M09b_cer_without_the_cross_check_uses_the_largest_leaf(self):
        mutant = src(sc.curve.collision_rates, "needing = audit['items_needing_suffix']", "needing = audit['leaves']['max']",
                     "if not math.isclose(cer, 1 - audit['leaves']['count'] / items, rel_tol=0, abs_tol=1e-12):", 'if False:')
        assert_killed(self, (sc.curve, 'collision_rates'), mutant, KILLERS['cer'])

    def test_M09c_cer_cross_check_is_removed_and_the_inputs_disagree(self):
        mutant = src(sc.curve.collision_rates, "if not math.isclose(cer, 1 - audit['leaves']['count'] / items, rel_tol=0, abs_tol=1e-12):", 'if False:')
        assert_killed(self, (sc.curve, 'collision_rates'), mutant, [SC + 'CollisionRateTests.test_cer_has_two_equal_definitions_and_a_mismatch_is_an_error'])

    # --- 12, 13 validate_config -------------------------------------------------------------------------------------------------
    def test_M12_formal_validation_accepts_engineering_labels(self):
        mutant = src(tok2.validate_config, 'allowed = sid_v2s.ENGINEERING_SID_VERSIONS if engineering else sid_v2s.SID_VERSIONS',
                     'allowed = sid_v2s.ENGINEERING_SID_VERSIONS')
        assert_killed(self, 'vkgr.generative_tok2_run.validate_config', mutant, KILLERS['validate'])

    def test_M12b_formal_validation_accepts_the_old_t3_label(self):
        mutant = src(tok2.validate_config, 'allowed = sid_v2s.ENGINEERING_SID_VERSIONS if engineering else sid_v2s.SID_VERSIONS',
                     "allowed = (sid_v2s.ENGINEERING_SID_VERSIONS if engineering else sid_v2s.SID_VERSIONS) + ('SID_BHKM_B256D3_S42',)")
        assert_killed(self, 'vkgr.generative_tok2_run.validate_config', mutant, KILLERS['validate'])

    def test_M13_validate_config_does_not_check_task_id(self):
        mutant = src(tok2.validate_config, "'schema', 'task_id', 'heads'", "'schema', 'heads'")
        assert_killed(self, 'vkgr.generative_tok2_run.validate_config', mutant, KILLERS['validate'])

    def test_M13b_validate_config_does_not_check_schema(self):
        mutant = src(tok2.validate_config, "('schema', 'task_id', 'heads'", "('task_id', 'heads'")
        assert_killed(self, 'vkgr.generative_tok2_run.validate_config', mutant, KILLERS['validate'])

    # --- 17 derive --------------------------------------------------------------------------------------------------------------
    def test_M17_rank_inside_a_group_is_descending(self):
        mutant = src(s2.derive_depth2_from_parent, 'np.lexsort((np.arange(len(ids)), key))', 'np.lexsort((-np.arange(len(ids)), key))')
        assert_killed(self, 'vkgr.sid_v2s.derive_depth2_from_parent', mutant, KILLERS['derive'])

    def test_M17b_derive_accepts_unsorted_ids(self):
        mutant = src(s2.derive_depth2_from_parent, ' or np.any(ids[1:] <= ids[:-1])', '')
        assert_killed(self, 'vkgr.sid_v2s.derive_depth2_from_parent', mutant, KILLERS['derive_sorted'])

    # --- 18 expected / exit 4 ---------------------------------------------------------------------------------------------------
    def test_M18_expected_comparison_misses_the_last_position(self):
        mutant = src(run.check_expected, 'for i in range(4)]', 'for i in range(3)]')
        assert_killed(self, 'vkgr.sid_tok2_run.check_expected', mutant, KILLERS['expected'])

    def test_M18b_expected_comparison_has_a_tolerance(self):
        mutant = src(run.check_expected, 'if want != got]', 'if abs(want - got) > 1]')
        assert_killed(self, 'vkgr.sid_tok2_run.check_expected', mutant, KILLERS['expected'])

    def test_M18c_expected_mismatch_exits_1_instead_of_4(self):
        assert_killed(self, 'vkgr.sid_tok2_run.EXIT_EXPECTED', 1, KILLERS['exit4'])

    def test_M18d_expected_is_never_enforced(self):
        assert_killed(self, 'vkgr.sid_tok2_run._expected_for', lambda config, name, engineering: None, KILLERS['expected_for'])

    def test_M18e_depth2_rederivation_is_skipped(self):
        assert_killed(self, 'vkgr.sid_tok2_run._rederive', lambda *args, **kwargs: None, KILLERS['rederive'])

    def test_M18f_replay_fit_is_skipped(self):
        assert_killed(self, 'vkgr.sid_tok_run._replay_fit', lambda *args, **kwargs: None, KILLERS['replay'])

    def test_M18g_config_binding_is_skipped(self):
        assert_killed(self, 'vkgr.sid_tok2_run._check_config_binding', lambda *args, **kwargs: None, KILLERS['binding'])

    # --- 19 Q1 table and states -------------------------------------------------------------------------------------------------
    def test_M19_q1_rows_one_and_two_swapped(self):
        # row 2 (a tentative rise) wins over row 1 (a confirmed rise) when both appear
        mutant = src(sc.curve.judge_q1, "if 'UP' in pair:", "if 'UP' in pair and 'T_UP' not in pair:")
        assert_killed(self, (sc.curve, 'judge_q1'), mutant, KILLERS['q1'])

    def test_M19a_q1_rows_three_and_four_swapped(self):
        # row 4 would win over row 3 for (DOWN, DOWN). Swapping rows 2 and 3 is NOT a mutant: their conditions exclude each other.
        mutant = src(sc.curve.judge_q1, "elif pair == ('DOWN', 'DOWN'):", 'elif False:')
        assert_killed(self, (sc.curve, 'judge_q1'), mutant, KILLERS['q1'])

    def test_M19b_q1_peak_row_fires_on_a_single_down(self):
        mutant = src(sc.curve.judge_q1, "elif pair == ('DOWN', 'DOWN'):", "elif 'DOWN' in pair:")
        assert_killed(self, (sc.curve, 'judge_q1'), mutant, KILLERS['q1'])

    def test_M19c_q1_ignores_an_up_in_the_second_position(self):
        mutant = src(sc.curve.judge_q1, "if 'UP' in pair:", "if pair[0] == 'UP':")
        assert_killed(self, (sc.curve, 'judge_q1'), mutant, KILLERS['q1'])

    def test_M19d_state_threshold_is_exclusive(self):
        mutant = src(sc.curve.state, 'if delta_2024 >= THRESHOLD_HITS and delta_2025 >= THRESHOLD_HITS:',
                     'if delta_2024 > THRESHOLD_HITS and delta_2025 > THRESHOLD_HITS:')
        assert_killed(self, (sc.curve, 'state'), mutant, KILLERS['state'])

    def test_M19e_tentative_state_uses_a_five_hit_threshold(self):
        mutant = src(sc.curve.tentative_state, 'if delta_42 >= THRESHOLD_HITS:', 'if delta_42 >= 5:')
        assert_killed(self, (sc.curve, 'tentative_state'), mutant, KILLERS['state'])

    # --- old entries ------------------------------------------------------------------------------------------------------------
    def test_M_old_a_byte_added_to_a_pinned_file_is_detected(self):
        assert_killed(self, (old, 'read_bytes'), lambda path: old.Path(path).read_bytes() + b'0', KILLERS['old'])

    def test_M_old_a_pin_in_the_table_is_changed(self):
        tampered = dict(old.PINS)
        tampered[sorted(tampered)[3]] = '0' * 40
        assert_killed(self, (old, 'PINS'), tampered, KILLERS['old'])

    def test_M_old_a_pinned_file_is_deleted(self):
        def missing(path):
            if str(path).endswith('src/vkgr/sid_tok_run.py'):
                return b''
            return old.Path(path).read_bytes()

        assert_killed(self, (old, 'read_bytes'), missing, KILLERS['old'])


class HarnessOfTheHarness(unittest.TestCase):
    """assert_killed itself: a survivor, a crash and a mistyped target must all make the suite fail."""

    def test_a_surviving_mutant_fails_the_suite(self):
        with self.assertRaises(AssertionError):
            assert_killed(self, 'vkgr.sid_v2s.pad_depth2', s2.pad_depth2, KILLERS['pad'])

    def test_a_crash_without_a_failed_assertion_does_not_count_as_killed(self):
        def crash(raw3):
            raise RuntimeError('boom')

        with self.assertRaises(AssertionError):
            assert_killed(self, 'vkgr.sid_v2s.pad_depth2', crash, [V2S + 'DepthTwoTests.test_pad_depth2_puts_zero_in_third_and_rank_in_fourth'])

    def test_a_mistyped_target_fails_the_suite_instead_of_passing(self):
        with self.assertRaises(AttributeError):
            assert_killed(self, 'vkgr.sid_v2s.pad_depht2', lambda raw3: raw3, KILLERS['pad'])
        with self.assertRaises(AttributeError):
            assert_killed(self, (s2, 'pad_depht2'), lambda raw3: raw3, KILLERS['pad'])

    def test_a_source_anchor_that_does_not_exist_fails_loudly(self):
        with self.assertRaises(AssertionError):
            src(run.check_expected, 'this text is not in the function', 'x')

    def test_the_prereg_checker_is_part_of_the_mutation_story(self):
        self.assertTrue(callable(prereg.check_consistency))
        self.assertTrue(formal.committed())


if __name__ == '__main__':
    unittest.main()
