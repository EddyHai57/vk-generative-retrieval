"""GTOK-02: mutation suite for the G training entry (runs inside the frozen-base harness, next to tests/test_generative_tok2.py).

Every mutant must be killed by an ASSERTION FAILURE of the named tests (a crash without a failed assertion does not count, and a patch
aimed at a missing name raises). Shared fixtures (the engineering world and the cached runs) are built in setUpClass, before any mutant.
"""
import inspect
import io
import textwrap
import unittest
from unittest import mock

import test_generative_tok2 as t
from vkgr import generative_tok2_run as tok2

T = 'test_generative_tok2.'


def run_ids(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ids)
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


def assert_killed(case, patch_target, replacement, test_ids):
    patcher = mock.patch.object(*patch_target, replacement) if isinstance(patch_target, tuple) else mock.patch(patch_target, replacement)
    with patcher:
        result = run_ids(test_ids)
    case.assertGreater(result.testsRun, 0)
    case.assertFalse(result.wasSuccessful(), f'mutant survived: {patch_target}')
    case.assertTrue(result.failures, f'mutant only crashed the tests (errors without a failed assertion): {patch_target}')
    return result


def src(function, old, new, *more):
    text = textwrap.dedent(inspect.getsource(function))
    for anchor, replacement in [(old, new)] + list(zip(more[::2], more[1::2])):
        if anchor not in text:
            raise AssertionError(f'mutation anchor not found in {function.__name__}: {anchor!r}')
        text = text.replace(anchor, replacement, 1)
    namespace = dict(function.__globals__)
    exec(compile(text, f'<mutant {function.__name__}>', 'exec'), namespace)
    return namespace[function.__name__]


KILLERS = {
    'binding': [T + 'SidTakesEffectTests.test_binding_function_carries_the_bundle_manifest_sha_and_depth'],
    'kind': [T + 'FailClosedTests.test_fresh_manifest_has_the_tok2_schema_and_kind'],
    'task_id': [T + 'ConfigTests.test_registry_rejections'],
    'sid_equals_bundle': [T + 'SidTakesEffectTests.test_config_sid_must_equal_the_bundle_version'],
    'last': [T + 'FailClosedTests.test_last_checkpoint_hash_is_checked'],
    'selection': [T + 'FailClosedTests.test_selection_is_replayed_from_the_published_epochs'],
    'binding_check': [T + 'FailClosedTests.test_other_bundle_or_wrong_sha_is_rejected'],
    'best': [T + 'FailClosedTests.test_best_checkpoint_one_extra_byte_is_rejected'],
    'members': [T + 'SidTakesEffectTests.test_members_extend_the_gtok01_members'],
}


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = t.fx()
        for name in ('b1', 'legacy', 'unbalanced', *t.NEW_LABELS):
            f.cached(name)
        for name in ('legacy', 'unbalanced', 'twin', *t.NEW_LABELS):
            f.bundle(name)

    def test_unmutated_selection_is_green(self):
        ids = sorted({name for names in KILLERS.values() for name in names})
        result = run_ids(ids)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)

    def test_M10_binding_forgets_the_bundle_sha(self):
        mutant = src(tok2._binding, "'manifest_sha256': info['manifest_sha256'], ", '')
        assert_killed(self, (tok2, '_binding'), mutant, KILLERS['binding'])

    def test_M10b_binding_forgets_the_depth(self):
        mutant = src(tok2._binding, "'depth': info['depth'],", '')
        assert_killed(self, (tok2, '_binding'), mutant, KILLERS['binding'])

    def test_M11_manifest_kind_is_the_gtok01_kind(self):
        assert_killed(self, (tok2, 'MANIFEST_KIND'), 'g_tok_train', KILLERS['kind'])

    def test_M11b_manifest_schema_is_the_gtok01_schema(self):
        assert_killed(self, (tok2, 'MANIFEST_SCHEMA'), 'g-tok-train-manifest-v1', KILLERS['kind'])

    def test_M12_runner_validation_does_not_check_the_task_id(self):
        mutant = src(tok2.validate_config, "'schema', 'task_id', 'heads'", "'schema', 'heads'")
        assert_killed(self, (tok2, 'validate_config'), mutant, KILLERS['task_id'])

    def test_M12b_runner_task_id_is_the_gtok01_one(self):
        assert_killed(self, (tok2, 'TASK_ID'), 'G-TOK-01', KILLERS['task_id'])

    def test_M12c_config_sid_is_not_compared_with_the_bundle(self):
        mutant = src(tok2._contexts, "if sid.info['sid_version'] != config['sid']:", 'if False:')
        assert_killed(self, (tok2, '_contexts'), mutant, KILLERS['sid_equals_bundle'])

    def test_M13_last_checkpoint_hash_is_not_checked(self):
        mutant = src(tok2.verify, "file_sha256(run_dir / 'last.pt') != value['last'].get('sha256')", 'False')
        assert_killed(self, (tok2, 'verify'), mutant, KILLERS['last'])

    def test_M14_selection_replay_is_not_compared(self):
        mutant = src(tok2.verify, "stop.state() != value['selection']", 'False')
        assert_killed(self, (tok2, 'verify'), mutant, KILLERS['selection'])

    def test_M15_binding_check_is_skipped(self):
        assert_killed(self, (tok2, '_check_binding'), lambda recorded, fresh: None, KILLERS['binding_check'])

    def test_M16_best_checkpoint_hash_is_not_checked(self):
        assert_killed(self, (tok2, '_check_best_hash'), lambda run_dir, best: None, KILLERS['best'])

    def test_M17_source_members_drop_the_new_files(self):
        assert_killed(self, (tok2, 'TOK2_MEMBERS'), tok2.TOK_MEMBERS, KILLERS['members'])


class HarnessOfTheHarness(unittest.TestCase):
    def test_a_surviving_mutant_fails_the_suite(self):
        with self.assertRaises(AssertionError):
            assert_killed(self, (tok2, '_check_binding'), tok2._check_binding, KILLERS['binding_check'])

    def test_a_crash_without_a_failed_assertion_does_not_count_as_killed(self):
        def crash(recorded, fresh):
            raise RuntimeError('boom')

        with self.assertRaises(AssertionError):
            assert_killed(self, (tok2, '_check_binding'), crash, [T + 'FailClosedTests.test_source_drift_is_a_binding_mismatch'])

    def test_a_mistyped_target_fails_the_suite_instead_of_passing(self):
        with self.assertRaises(AttributeError):
            assert_killed(self, (tok2, '_check_bindnig'), lambda recorded, fresh: None, KILLERS['binding_check'])


if __name__ == '__main__':
    unittest.main()
