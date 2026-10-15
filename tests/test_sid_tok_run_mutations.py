"""GTOK-01 FEAT-003: mutants C1..C6 of the tokenizer CLI must all be killed."""
import copy
import io
import unittest
from unittest import mock

from tests import test_sid_tok_run as t

T = 'tests.test_sid_tok_run.'


def run_ids(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ids)
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


def assert_killed(case, patch_target, replacement, test_ids):
    with mock.patch(patch_target, replacement):
        result = run_ids(test_ids)
    case.assertFalse(result.wasSuccessful(), f'mutant survived: {patch_target}')
    return result


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the shared bundles under the fixture patches before any mutant is active.
        for patch in (mock.patch.dict(t.rq.PREREG, lr=0.1), mock.patch.object(t.rq, 'fit', t.refreshing_fit)):
            patch.start()
            cls.addClassCleanup(patch.stop)
        t.built()

    def test_unmutated_selection_is_green(self):
        ids = [T + 'Failures.test_wrong_content_catalog_and_config_sha_are_rejected',
               T + 'Failures.test_usage_below_80_percent_writes_failure_json_and_no_manifest',
               T + 'Failures.test_wall_cap_result_exposes_no_model', T + 'Failures.test_t3_calls_frozen_fit_unchanged_and_drops_nodes',
               T + 'Pipeline.test_compare_has_the_t0_shaped_baseline_row_and_correct_labels',
               T + 'Frozen.test_named_frozen_fields_and_type_confusion',
               T + 'Failures.test_config_with_changed_max_steps_is_rejected_by_the_cli']
        result = run_ids(ids)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertEqual(result.testsRun, len(ids))

    def test_C1_input_sha_check_skipped(self):
        assert_killed(self, 'vkgr.sid_tok_run._check_file_sha', lambda path, expected: t.Path(path),
                      [T + 'Failures.test_wrong_content_catalog_and_config_sha_are_rejected'])

    def test_C2_gate_failure_still_writes_a_manifest(self):
        original = t.run.write_failure_record

        def also_manifest(output, record):
            original(output, record)
            (t.Path(output) / 'run_manifest.json').write_text('{}')

        assert_killed(self, 'vkgr.sid_tok_run.write_failure_record', also_manifest,
                      [T + 'Failures.test_usage_below_80_percent_writes_failure_json_and_no_manifest',
                       T + 'Failures.test_wall_cap_with_injected_clock_writes_failure_json_and_no_manifest',
                       T + 'Failures.test_nan_writes_failure_json_and_no_manifest'])

    def test_C3_wall_timeout_returns_a_partial_state(self):
        def leaky(base, raw):
            return {**base, 'status': raw['status'], 'reasons': ['fit:' + raw['status']], 'state': {'K': base['K'], 'state_dict': {}}}

        assert_killed(self, 'vkgr.sid_tok_run._failed_result', leaky, [T + 'Failures.test_wall_cap_result_exposes_no_model'])

    def test_C4_compare_labels_t1_as_t0(self):
        original = t.run._label

        def mislabel(version):
            return 'T0' if version == 'SID_RQVAE_B256D3_S42' else original(version)

        assert_killed(self, 'vkgr.sid_tok_run._label', mislabel,
                      [T + 'Pipeline.test_compare_has_the_t0_shaped_baseline_row_and_correct_labels'])

    def test_C5_validate_config_lets_max_steps_through(self):
        original = t.run.validate_config

        def lenient(value):
            value = copy.deepcopy(value)
            if isinstance(value, dict) and isinstance(value.get('rqvae'), dict):
                value['rqvae']['max_steps'] = t.rq.PREREG['max_steps']
            return original(value)

        assert_killed(self, 'vkgr.sid_tok_run.validate_config', lenient,
                      [T + 'Frozen.test_named_frozen_fields_and_type_confusion', T + 'Frozen.test_every_field_change_is_rejected',
                       T + 'Failures.test_config_with_changed_max_steps_is_rejected_by_the_cli'])

    def test_C6_t3_keeps_nodes(self):
        assert_killed(self, 'vkgr.sid_tok_run._drop_nodes', lambda model: model,
                      [T + 'Failures.test_t3_calls_frozen_fit_unchanged_and_drops_nodes'])


if __name__ == '__main__':
    unittest.main()
