"""G6-PREP step 6b mutants M46..M58 of sid_tok_run_d (depth/levels reaching the fits, gate handling, baseline checks): all killed."""
import unittest

from tests.mutation_util import mutated, run_ids
from vkgr import sid_tok_run_d as run

T = 'tests.test_sid_tok_run_d.'
ALL = [T + 'Config', T + 'Pipeline', T + 'Refusals']


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import test_sid_tok_run_d
        test_sid_tok_run_d.built()  # build the shared bundles before any patch is active

    def test_unmutated_selection_is_green(self):
        result = run_ids(ALL)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertGreaterEqual(result.testsRun, 30)

    def kill(self, old, new, ids=None, count=1):
        with mutated(run, old, new, count):
            result = run_ids(ids or ALL)
        self.assertFalse(result.wasSuccessful(), f'mutant survived: {old!r} -> {new!r}')

    def test_M46_levels_do_not_reach_the_rqvae_fit(self):
        self.kill("params['levels'] = variant['levels']", "params['levels'] = 3", [T + 'Pipeline'])

    def test_M47_bhkm_fit_ignores_the_configured_depth(self):
        self.kill("depth=spec['depth'], seed=spec['seed']", 'depth=3, seed=spec[\'seed\']', [T + 'Pipeline'])

    def test_M48_gate_failure_is_published_as_success(self):
        self.kill("    if result['ok']:\n        manifest = write_success_bundle_d", "    if True:\n        manifest = write_success_bundle_d", [T + 'Refusals.test_gate_failure_writes_failure_json_and_no_manifest'])

    def test_M49_b2_dropped_from_the_fit_gates(self):
        self.kill("expected_content_sha256=expected_content_sha256, prereg=prereg)", 'expected_content_sha256=expected_content_sha256)', [T + 'Refusals.test_a_fresh_bhkm_fit_records_B1_and_B2_in_its_gates'])

    def test_M50_replay_baseline_equality_not_enforced(self):
        self.kill("        if not np.array_equal(stored.sids, sids):\n            raise ValueError('baseline mapping differs from the bundle mapping')\n",
                  '        pass\n', [T + 'Pipeline.test_replay_that_differs_from_the_baseline_is_refused'])

    def test_M51_prefix_agreement_uses_one_digit(self):
        self.kill('ad.prefix_agreement(stored.ids, stored.sids, ids, sids, base_depth)', 'ad.prefix_agreement(stored.ids, stored.sids, ids, sids, 1)',
                  [T + 'Pipeline.test_depth_four_reports_the_prefix_agreement_with_the_depth_three_baseline'])

    def test_M52_engineering_overrides_allowed_in_formal_mode(self):
        self.kill('    if overrides and not args.engineering:', '    if False:', [T + 'Refusals.test_engineering_only_options_need_engineering'])

    def test_M53_overrides_allowed_for_bhkm(self):
        self.kill("    if overrides and kind != 'rqvae':", '    if False:', [T + 'Refusals.test_overrides_only_apply_to_rqvae_and_engineering_rqvae_needs_max_steps'])

    def test_M54_output_directory_may_pre_exist(self):
        self.kill('output = run._new_output(args.output)', 'output = Path(args.output)', [T + 'Refusals.test_an_existing_output_directory_is_refused'])

    def test_M55_config_whitelist_is_not_applied(self):
        self.kill("    found = run._first_difference(value, expected_config())\n    if found:", '    found = None\n    if found:', [T + 'Config', T + 'Refusals'])

    def test_M56_config_binding_not_checked_on_verify(self):
        self.kill("    if loaded.manifest['fit_config_sha256'] != config_sha:", '    if False:', [T + 'Refusals.test_verify_refuses_a_config_with_the_same_values_but_different_bytes'])

    def test_M57_replay_fit_accepts_any_mapping(self):
        self.kill("    if not (np.array_equal(again['ids'], stored.ids) and np.array_equal(again['sids'], stored.sids)) or again['summary'] != config['summary']:",
                  '    if False:', [T + 'Refusals.test_replay_fit_refuses_a_mapping_the_frozen_fit_does_not_give'])

    def test_M58_failure_record_keeps_the_depth_three_schema(self):
        self.kill("record.update({'schema': FAILURE_SCHEMA, 'task_id': TASK_ID, 'depth': result['depth']})", "record.update({'depth': result['depth']})",
                  [T + 'Refusals.test_gate_failure_writes_failure_json_and_no_manifest'])

    def test_M59_frozen_levels_changed_in_the_expected_config(self):
        self.kill("rqvae['levels'] = 4", "rqvae['levels'] = 3", [T + 'Config'])


if __name__ == '__main__':
    unittest.main()
