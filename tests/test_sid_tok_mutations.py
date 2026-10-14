"""GTOK-01 FEAT-002: mutants A1..A10 of the SIDMAP_V2 contract and the tokenizer audit must all be killed."""
import io
import unittest
from unittest import mock

import numpy as np

from tests import test_sid_v2 as v2

V = 'tests.test_sid_v2.'
A = 'tests.test_sid_tok_audit.'


def run_ids(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ids)
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


def assert_killed(case, patch_target, replacement, test_ids):
    with mock.patch(patch_target, replacement):
        result = run_ids(test_ids)
    case.assertFalse(result.wasSuccessful(), f'mutant survived: {patch_target}')
    return result


def noop(*args, **kwargs):
    return None


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the shared worlds before any patch is active so that no mutant can poison the fixtures.
        for name in ('bhkm', 'rqvae', 'formal'):
            v2.world(name)

    def test_unmutated_selection_is_green(self):
        ids = [V + 'FailClosed.test_failed_gates_bundle_is_rejected', V + 'FailClosed.test_wrong_content_digest',
               V + 'FailClosed.test_wrong_catalog', V + 'FailClosed.test_weights_container_change_with_identical_arrays',
               V + 'FailClosed.test_audit_json_mismatch', V + 'BuildCodec.test_rejects_noncanonical_suffix_bad_digits_and_bad_shapes',
               A + 'HandComputed.test_collisions_are_counted_on_the_first_three_codes',
               A + 'HandComputed.test_explained_variance_hand_computed', A + 'Gates.test_R1_usage_boundary_is_integer_exact',
               A + 'BalancedChildren.test_spread_two_is_a_violation_unless_tolerance_two_is_explicit']
        result = run_ids(ids)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertEqual(result.testsRun, len(ids))

    def test_A1_load_bundle_skips_gates_passed(self):
        assert_killed(self, 'vkgr.sid_v2._check_gates_passed', noop, [V + 'FailClosed.test_failed_gates_bundle_is_rejected'])

    def test_A2_load_bundle_skips_content_digest(self):
        assert_killed(self, 'vkgr.sid_v2._check_content', noop, [V + 'FailClosed.test_wrong_content_digest'])

    def test_A3_load_bundle_skips_catalog_equality(self):
        assert_killed(self, 'vkgr.sid_v2._check_catalog', noop, [V + 'FailClosed.test_wrong_catalog'])

    def test_A4_load_bundle_skips_weights_sha(self):
        from vkgr import sid_v2
        original = sid_v2._verify_artifact
        assert_killed(self, 'vkgr.sid_v2._verify_artifact',
                      lambda directory, name, digest: None if name == 'weights.npz' else original(directory, name, digest),
                      [V + 'FailClosed.test_weights_container_change_with_identical_arrays'])

    def test_A5_load_bundle_skips_audit_recomputation(self):
        assert_killed(self, 'vkgr.sid_v2._check_audit', noop,
                      [V + 'FailClosed.test_audit_json_mismatch', V + 'FailClosed.test_rqvae_audit_rq_section_is_recomputed'])

    def test_A6_collisions_counted_on_four_codes(self):
        def four_code_counts(sids):
            return np.unique(np.asarray(sids, dtype=np.int64), axis=0, return_counts=True)[1]
        assert_killed(self, 'vkgr.sid_tok_audit._leaf_counts', four_code_counts,
                      [A + 'HandComputed.test_collisions_are_counted_on_the_first_three_codes',
                       A + 'HandComputed.test_leaf_statistics'])

    def test_A7_explained_variance_with_the_wrong_denominator(self):
        assert_killed(self, 'vkgr.sid_tok_audit._total_ss', lambda x: float((np.asarray(x, dtype=np.float64) ** 2).sum()),
                      [A + 'HandComputed.test_explained_variance_hand_computed'])

    def test_A8_usage_compared_with_strict_greater(self):
        assert_killed(self, 'vkgr.sid_tok_audit.usage_ok', lambda used, K: used * 5 > K * 4,
                      [A + 'Gates.test_R1_usage_boundary_is_integer_exact'])

    def test_A9_codec_built_without_check_codec(self):
        assert_killed(self, 'vkgr.generative_data._check_codec', lambda codec, prepared=None: codec.catalog,
                      [V + 'BuildCodec.test_rejects_noncanonical_suffix_bad_digits_and_bad_shapes',
                       V + 'BuildCodec.test_catalog_equality_is_part_of_the_codec_check'])

    def test_A10_balanced_children_tolerance_two(self):
        assert_killed(self, 'vkgr.sid_tok_audit.BALANCE_TOLERANCE', 2,
                      [A + 'BalancedChildren.test_spread_two_is_a_violation_unless_tolerance_two_is_explicit',
                       A + 'BalancedChildren.test_hand_example_is_unbalanced_at_the_root'])


if __name__ == '__main__':
    unittest.main()
