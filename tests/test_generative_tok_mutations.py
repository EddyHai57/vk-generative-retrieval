"""GTOK-01 FEAT-004: mutants D1..D7 of the G training entry must all be killed (run inside the frozen-base harness)."""
import dataclasses
import io
import unittest
from unittest import mock

import test_generative_tok as T

P = 'test_generative_tok.'
RUN = 'vkgr.generative_tok_run.'


def run_ids(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(P + name) for name in ids)
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
        # Build every shared bundle and run before any patch is active so that no mutant can poison the fixtures.
        f = T.fx()
        for name in ('legacy', 'unbalanced', 'twin'):
            f.bundle(name)
        for name in ('b1', 'legacy', 'unbalanced', 'unbalanced-batched'):
            f.cached(name)

    def test_unmutated_selection_is_green(self):
        ids = ['SidTakesEffectTests.test_context_uses_the_bundle_codec', 'SidTakesEffectTests.test_binding_function_carries_the_bundle_manifest_sha',
               'FailClosedTests.test_engineering_flag_mismatch_is_rejected', 'FailClosedTests.test_fresh_manifest_has_the_tok_schema_and_kind',
               'ConfigTests.test_registry_rejections', 'FailClosedTests.test_other_bundle_or_wrong_sha_is_rejected',
               'FailClosedTests.test_same_mapping_but_another_bundle_manifest_is_a_binding_mismatch',
               'FailClosedTests.test_best_checkpoint_one_extra_byte_is_rejected',
               'FailClosedTests.test_best_checkpoint_hash_is_checked_even_when_the_file_still_loads']
        result = run_ids(ids)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertEqual(result.testsRun, len(ids))

    def test_D1_context_ignores_the_new_codec(self):
        from vkgr import generative_tok_data as data
        original = data.build_tok_context
        def ignore_codec(context, loaded, length, sid):
            return original(context, loaded, length, dataclasses.replace(sid, codec=context.codec))
        assert_killed(self, RUN + 'build_tok_context', ignore_codec,
                      ['SidTakesEffectTests.test_context_uses_the_bundle_codec', 'UnbalancedTests.test_unbalanced_sid_trains_and_verifies'])

    def test_D2_binding_without_the_bundle_sha(self):
        from vkgr import generative_tok_run as tok
        original = tok._binding
        def without_sha(*args, **kwargs):
            value = original(*args, **kwargs); value['sid'] = {k: v for k, v in value['sid'].items() if k != 'manifest_sha256'}
            return value
        assert_killed(self, RUN + '_binding', without_sha, ['SidTakesEffectTests.test_binding_function_carries_the_bundle_manifest_sha'])

    def test_D3_loader_ignores_the_engineering_flag(self):
        from vkgr import generative_tok_run as tok
        original = tok._load_sid
        def always_engineering(*args):
            args = list(args); args[5] = True
            return original(*args)
        assert_killed(self, RUN + '_load_sid', always_engineering, ['FailClosedTests.test_engineering_flag_mismatch_is_rejected'])

    def test_D4_manifest_kind_written_as_g_b1_train(self):
        assert_killed(self, RUN + 'MANIFEST_KIND', 'g_b1_train', ['FailClosedTests.test_fresh_manifest_has_the_tok_schema_and_kind'])

    def test_D5_validate_config_accepts_the_old_sid_label(self):
        from vkgr import generative_tok_run as tok
        original = tok.validate_config
        old = ('SID_BHKM_B16D3_S42', 'SID_RANDPERM42_OF_BHKM_B16D3')
        def lenient(value, engineering=False):
            if type(value) is dict and value.get('sid') in old:
                return original({**value, 'sid': tok.sid_v2.SID_VERSIONS[0]}, engineering)
            return original(value, engineering)
        assert_killed(self, RUN + 'validate_config', lenient, ['ConfigTests.test_registry_rejections'])

    def test_D6_verify_does_not_compare_the_binding(self):
        assert_killed(self, RUN + '_check_binding', noop,
                      ['FailClosedTests.test_other_bundle_or_wrong_sha_is_rejected',
                       'FailClosedTests.test_same_mapping_but_another_bundle_manifest_is_a_binding_mismatch',
                       'FailClosedTests.test_source_drift_is_a_binding_mismatch'])

    def test_D7_verify_does_not_check_the_best_checkpoint_hash(self):
        assert_killed(self, RUN + '_check_best_hash', noop,
                      ['FailClosedTests.test_best_checkpoint_one_extra_byte_is_rejected',
                       'FailClosedTests.test_best_checkpoint_hash_is_checked_even_when_the_file_still_loads'])


if __name__ == '__main__':
    unittest.main()
