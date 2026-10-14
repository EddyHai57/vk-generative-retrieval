"""GTOK-01R FEAT-003: mutants C1-C11 (plus extras X*) of the tokenizer CLI must all be killed.

A mutant is a source edit of src/vkgr/sid_tok_run_r.py executed inside the real module namespace and restored
afterwards (same harness as tests/test_sid_select_r_mutations.py). It counts as killed only when the selected tests
report at least one assertion failure and no unrelated error. Several tests assert the specific error message so that
a mutant cannot be killed (or hidden) by an unrelated ValueError further down the pipeline.
"""
import contextlib
import importlib
import io
import unittest
from pathlib import Path

from tests import test_sid_tok_run_r as t

P = 'tests.test_sid_tok_run_r.'
SELECT = P + 'SelectOutcomes.'
FITNEG = P + 'FitNegatives.'
VERNEG = P + 'VerifyNegatives.'
PROBE = P + 'Probe.'
CONFIG = P + 'Config.'
SIZE = P + 'TrainLogSize.'
SIZE_CLI = P + 'SizeThroughCli.'
PIPE = P + 'Pipeline.'
ARGS = P + 'ArgumentErrors.'


def run_ids(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ids)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    result.transcript = stream.getvalue()
    return result


@contextlib.contextmanager
def mutated(case, edits):
    """Apply the (old, new) source edits inside the live module namespace; restore it afterwards."""
    module = importlib.import_module('vkgr.sid_tok_run_r')
    path = Path(module.__file__)
    source = path.read_text()
    for old, new in edits:
        case.assertEqual(source.count(old), 1, 'mutation anchor must occur exactly once: ' + repr(old))
        source = source.replace(old, new, 1)
    saved = dict(module.__dict__)
    arms_before = dict(t.rr.PREREG_R['arms'])  # a mutant may leave a foreign dict in the live arm table
    try:
        exec(compile(source, str(path), 'exec'), module.__dict__)
        # the import-time snapshot must stay the one taken without the class-level test patches
        module.__dict__['_FROZEN_R'] = saved['_FROZEN_R']
        yield module
    finally:
        module.__dict__.clear()
        module.__dict__.update(saved)
        t.rr.PREREG_R['arms'].clear()
        t.rr.PREREG_R['arms'].update(arms_before)


def assert_killed(case, edits, test_ids):
    with mutated(case, edits):
        result = run_ids(test_ids)
    case.assertFalse(result.wasSuccessful(), 'mutant survived: ' + repr(edits))
    case.assertTrue(result.failures, 'mutant killed by errors only, not by an assertion:\n' + result.transcript[-1500:])
    case.assertEqual(result.errors, [], 'mutant raised unrelated errors:\n' + result.transcript[-1500:])


def one(old, new):
    return [(old, new)]


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the shared bundles under the fixture patches before any mutant is active.
        t.build_under_patches()

    def test_unmutated_selection_is_green(self):
        ids = [FITNEG + 'test_wrong_decision_sha_and_missing_file', SIZE + 'test_limit_boundary',
               SELECT + 'test_stop_writes_the_decision_and_exits_3', VERNEG + 'test_untampered_copy_still_verifies']
        result = run_ids(ids)
        self.assertTrue(result.wasSuccessful(), result.transcript)
        self.assertEqual(result.testsRun, len(ids))


# -------------------------------------------------------------------------------------------------- test groups
BAD_SHAS = [FITNEG + 'test_wrong_decision_sha_and_missing_file',
            FITNEG + 'test_wrong_input_and_config_shas_and_changed_config_are_refused',
            SELECT + 'test_validation_failures_exit_1_and_write_nothing',
            PROBE + 'test_probe_argument_and_input_errors_exit_1_before_any_output',
            CONFIG + 'test_load_config_rejects_wrong_sha_and_duplicate_keys']
CONFIG_TESTS = [CONFIG + 'test_every_field_change_is_rejected', CONFIG + 'test_named_frozen_fields_and_type_confusion']
FAILURE_FIT = [FITNEG + 'test_gate_failure_writes_failure_json_with_diagnostics_and_no_manifest',
               FITNEG + 'test_wall_cap_writes_failure_json_and_no_manifest',
               FITNEG + 'test_nonfinite_training_writes_failure_json_and_no_manifest']
NO_MODEL = [FITNEG + 'test_wall_cap_and_gate_failed_results_expose_no_model']
TAMPER_FIT = [VERNEG + 'test_fit_params_effective_prereg_and_overrides_are_bound_to_the_config']
INPUTS_FIT = [FITNEG + 'test_decision_for_other_inputs_is_refused', VERNEG + 'test_decision_for_other_inputs_is_refused']
FROZEN_LIVE = [CONFIG + 'test_formal_commands_refuse_live_constants_other_than_the_preregistered_ones']


class RequiredMutants(unittest.TestCase):
    """C1-C11 of the plan (section 7)."""

    @classmethod
    def setUpClass(cls):
        t.build_under_patches()

    def test_C1_input_sha_check_skipped(self):
        assert_killed(self, one('return run._check_file_sha(path, digest)', 'return gd._no_symlink(path)'), BAD_SHAS)

    def test_C2_fit_accepts_stop_escalate_and_invalid_control_decisions(self):
        assert_killed(self, one("if decision['decision'] not in rr.PREREG_R['arm_order']:", 'if False:'),
                      [FITNEG + 'test_decision_that_is_not_an_l_arm_is_refused',
                       VERNEG + 'test_stop_decision_cannot_verify_a_bundle'])

    def test_C3_gate_failure_still_writes_a_manifest(self):
        assert_killed(self, one('    run.write_failure_record(output, record)\n',
                                "    run.write_failure_record(output, record)\n"
                                "    (output / sid_v2.MANIFEST_NAME).write_text('{}')\n"), FAILURE_FIT)

    def test_C4_verify_does_not_compare_the_arm_with_the_decision(self):
        assert_killed(self, one("if tok.get('arm') != arm or (decision['override'] is not None and "
                                "decision['override']['arm'] != tok.get('arm')):", 'if False:'),
                      [VERNEG + 'test_arm_differing_from_the_decision_is_refused',
                       VERNEG + 'test_decision_with_another_arm_is_refused_for_this_bundle',
                       PIPE + 'test_fit_records_the_arm_of_the_decision_not_a_fixed_one'])

    def test_C5_verify_does_not_compare_arm_hparams(self):
        assert_killed(self, one("if tok.get('arm_hparams') != _expected_arm_hparams(decision['decision'], engineering):",
                                'if False:'), [VERNEG + 'test_tampered_arm_hparams_are_refused'])

    def test_C6_select_validation_failure_still_leaves_a_decision(self):
        assert_killed(self, one('    return run._new_output(path)',
                                "    out = run._new_output(path)\n    out.mkdir()\n"
                                "    (out / 'decision.json').write_text('{}')\n    return out"),
                      [SELECT + 'test_validation_failures_exit_1_and_write_nothing',
                       SELECT + 'test_override_is_refused_unless_the_recomputation_is_escalate_and_the_arm_is_listed',
                       SELECT + 'test_probes_with_different_commits_are_refused'])

    def test_C7_fit_does_not_require_a_decision(self):
        assert_killed(self, one("fit.add_argument('--decision', required=True)", "fit.add_argument('--decision', required=False)"),
                      [FITNEG + 'test_fit_without_decision_is_an_argument_error'])

    def test_C7b_verify_does_not_require_a_decision(self):
        assert_killed(self, one("verify.add_argument('--decision', required=True)",
                                "verify.add_argument('--decision', required=False)"),
                      [VERNEG + 'test_verify_without_decision_is_an_argument_error'])

    def test_C8_validate_config_lets_a_frozen_constant_through(self):
        paths = [('rqvae', 'max_steps'), ('rqvae', 'wall_cap_seconds'), ('rqvae', 'lr'), ('prereg_r', 'arms', 'l2', 'lr'),
                 ('prereg_r', 'reset', 'dead_window'), ('prereg_r', 'probe_wall_cap_seconds'), ('prereg_r', 'probe_steps'),
                 ('prereg_r', 'arms', 'l4', 'betas'), ('select_r', 'e_r2_min'), ('select_r', 'd_points'),
                 ('select_r', 'p0_final_usage_max'), ('select_r', 'preflight_limit_seconds'),
                 ('select_r', 'a2_from_step'), ('variants', 'T1', 'codebook_size'), ('recipe',)]
        helper = ("def _lenient(value):\n    value = copy.deepcopy(value)\n    expected = expected_config_r()\n"
                  "    try:\n        node, ref = value, expected\n        for part in PATH[:-1]:\n            node, ref = node[part], ref[part]\n"
                  "        node[PATH[-1]] = ref[PATH[-1]]\n    except (KeyError, TypeError):\n        pass\n    return value\n\n\ndef validate_config_r(value):")
        for path in paths:
            with self.subTest(path=path):
                edits = [('def validate_config_r(value):', f'PATH = {path!r}\n\n\n' + helper),
                         ('found = run._first_difference(value, expected_config_r())',
                          'found = run._first_difference(_lenient(value), expected_config_r())')]
                assert_killed(self, edits, CONFIG_TESTS)

    def test_C9_wall_timeout_returns_a_partial_state(self):
        assert_killed(self, one('result = run._failed_result(base, raw)',
                                "result = {**run._failed_result(base, raw), 'state': {'K': K, 'state_dict': {}}}"), NO_MODEL)

    def test_C9b_gate_failed_result_hands_back_the_model(self):
        assert_killed(self, one("return {**base, 'ok': False, 'status': 'gate_failed', 'audit': audit, 'gates': gates, 'detail': detail}",
                                "return {**base, 'ok': False, 'status': 'gate_failed', 'audit': audit, 'gates': gates, "
                                "'detail': detail, 'weights': state}"), NO_MODEL)

    def test_C10_decision_with_a_missing_override_field_is_accepted(self):
        assert_killed(self, one('    rr.validate_decision(decision)\n',
                                "    decision.setdefault('override', None)\n    rr.validate_decision(decision)\n"),
                      [VERNEG + 'test_decision_missing_the_override_field_is_refused',
                       FITNEG + 'test_decision_with_missing_extra_or_mistyped_fields_is_refused'])

    def test_C11_train_log_size_assertion_removed(self):
        assert_killed(self, one('    if size >= TRAIN_LOG_MAX_BYTES:', '    if False:'),
                      [SIZE + 'test_assertion_fires_when_the_limit_is_exceeded', SIZE + 'test_limit_boundary',
                       SIZE_CLI + 'test_oversized_train_log_aborts_the_publication'])


class ExtraMutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t.build_under_patches()

    def kill(self, old, new, tests):
        assert_killed(self, one(old, new), tests)

    def test_X1_probe_enables_the_projection_stop(self):
        self.kill('clock=clock, project=False, diagnostics=True,', 'clock=clock, project=True, diagnostics=True,',
                  [PROBE + 'test_probe_calls_fit_r_with_the_probe_arguments_only_from_the_config'])

    def test_X2_probe_switches_diagnostics_off(self):
        self.kill('clock=clock, project=False, diagnostics=True,', 'clock=clock, project=False, diagnostics=False,',
                  [PROBE + 'test_probe_calls_fit_r_with_the_probe_arguments_only_from_the_config',
                   PROBE + 'test_probe_writes_a_valid_result_bound_to_the_inputs'])

    def test_X3_probe_uses_the_fit_wall_cap(self):
        self.kill("config['prereg_r']['probe_wall_cap_seconds'])", "config['rqvae']['wall_cap_seconds'])",
                  [CONFIG + 'test_probe_and_fit_params_come_from_their_own_sections',
                   PROBE + 'test_probe_calls_fit_r_with_the_probe_arguments_only_from_the_config'])

    def test_X4_probe_uses_the_fit_length(self):
        self.kill("config['prereg_r']['probe_steps'])", "config['rqvae']['max_steps'])",
                  [CONFIG + 'test_probe_and_fit_params_come_from_their_own_sections'])

    def test_X5_invalid_control_exits_3(self):
        self.kill("return EXIT_ERROR if chosen == 'INVALID_CONTROL' else EXIT_GATE", 'return EXIT_GATE',
                  [SELECT + 'test_invalid_control_writes_the_decision_but_exits_1'])

    def test_X5b_stop_exits_1(self):
        self.kill("return EXIT_ERROR if chosen == 'INVALID_CONTROL' else EXIT_GATE", 'return EXIT_ERROR',
                  [SELECT + 'test_stop_writes_the_decision_and_exits_3', SELECT + 'test_escalate_writes_the_decision_and_exits_3'])

    def test_X6_every_decision_exits_0(self):
        self.kill("if chosen in rr.PREREG_R['arm_order']:", 'if True:',
                  [SELECT + 'test_stop_writes_the_decision_and_exits_3', SELECT + 'test_escalate_writes_the_decision_and_exits_3',
                   SELECT + 'test_invalid_control_writes_the_decision_but_exits_1'])

    def test_X7_select_skips_the_config_check(self):
        self.kill('    load_config_r(args.config, args.config_sha)\n    probes =', '    probes =',
                  [SELECT + 'test_select_refuses_a_config_that_is_not_the_approved_one'])

    def test_X8_select_does_not_pass_the_probe_file_hashes(self):
        self.kill('rr.build_decision(probes, list(args.probe_sha),', "rr.build_decision(probes, ['0' * 64] * len(probes),",
                  [SELECT + 'test_decision_references_the_probe_files_by_their_own_sha256'])

    def test_X9_select_formal_constant_check_removed(self):
        self.kill('    output = _new_output(args.output)\n    if not args.engineering:\n        require_frozen_constants()\n    load_config_r',
                  '    output = _new_output(args.output)\n    load_config_r', FROZEN_LIVE)

    def test_X10_decision_input_check_removed(self):
        self.kill("    if decision['content_file_sha256'] != content_sha or decision['content_array_sha256'] != inputs['content_sha256'] \\\n"
                  "            or decision['catalog_sha256'] != inputs['catalog_sha256']:", '    if False:', INPUTS_FIT)

    def test_X10b_decision_content_file_leg_removed(self):
        self.kill("decision['content_file_sha256'] != content_sha or ", '', INPUTS_FIT)

    def test_X10c_decision_content_array_leg_removed(self):
        self.kill("decision['content_array_sha256'] != inputs['content_sha256'] \\\n            or ", '', INPUTS_FIT)

    def test_X10d_decision_catalog_leg_removed(self):
        self.kill(" \\\n            or decision['catalog_sha256'] != inputs['catalog_sha256']:", ':', INPUTS_FIT)

    def test_X11_variant_K_check_removed(self):
        self.kill("if rr.PREREG_R_SELECT['variant_K'][variant] != K:", 'if False:',
                  [FITNEG + 'test_k_mismatch_between_variant_and_decision_is_refused'])

    def test_X11b_variant_coverage_check_removed(self):
        self.kill("if decision['per_arm_per_K'][decision['decision']][variant]['status'] != 'complete':", 'if False:',
                  [FITNEG + 'test_k_mismatch_between_variant_and_decision_is_refused'])

    def test_X12_decision_config_binding_removed(self):
        self.kill("if decision['config_sha256'] != config_sha:", 'if False:',
                  [FITNEG + 'test_decision_for_another_config_is_refused',
                   VERNEG + 'test_decision_must_belong_to_this_config_and_sha'])

    def test_X13_recipe_check_removed(self):
        self.kill("if tok.get('recipe') != RECIPE:", 'if False:', [VERNEG + 'test_missing_or_wrong_recipe_is_refused'])

    def test_X14_bundle_decision_sha_check_removed(self):
        self.kill("if tok.get('decision_sha256') != decision_sha:", 'if False:',
                  [VERNEG + 'test_decision_sha_recorded_in_the_bundle_must_match'])

    def test_X15_manifest_config_binding_removed(self):
        self.kill("if loaded.manifest['fit_config_sha256'] != config_sha:", 'if False:',
                  [VERNEG + 'test_bundle_fitted_with_another_config_file_is_refused'])

    def test_X16_fit_binding_legs_removed(self):
        for leg in ("tok['fit'] == _jsonable(fit_params(config, overrides))", "tok['effective_prereg'] == _jsonable(rq.PREREG)",
                    "set(overrides) <= set(OVERRIDE_OPTIONS_R)", "tok['variant'] == run._label(loaded.info['sid_version'])",
                    "tok['schema'] == CONFIG_SCHEMA_R", "set(tok) == TOKENIZER_CONFIG_KEYS"):
            with self.subTest(leg=leg):
                assert_killed(self, one(leg, 'True'), TAMPER_FIT + [VERNEG + 'test_variant_label_in_the_bundle_must_match_its_sid_version',
                                                                   VERNEG + 'test_missing_extra_key_in_tokenizer_config_is_refused'])

    def test_X17_prefix_replay_never_fails(self):
        self.kill("            != log['prefix_digest']:",
                  "            != _replay_prefix_digest_r(tok, loaded.info['branching'], content, device, loaded.manifest['engineering']):",
                  [PIPE + 'test_replay_prefix_reproduces_the_digest_and_detects_a_difference'])

    def test_X18_replay_runs_with_diagnostics_on(self):
        self.kill("probe_step=params['probe_step'], diagnostics=False)", "probe_step=params['probe_step'], diagnostics=True)",
                  [PIPE + 'test_replay_runs_fit_r_for_the_recorded_arm_with_the_recorded_hparams'])

    def test_X19_replay_ignores_the_recorded_arm_hparams(self):
        self.kill("with run._prereg_context(tok['effective_prereg'], engineering), _arm_context(arm, tok['arm_hparams']):",
                  "with run._prereg_context(tok['effective_prereg'], engineering):",
                  [PIPE + 'test_replay_helper_installs_the_recorded_arm_hparams_and_restores_the_table'])

    def test_X19b_replay_does_not_restore_the_arm_table(self):
        self.kill('        rr.PREREG_R[\'arms\'][arm] = saved\n', '        pass\n',
                  [PIPE + 'test_replay_helper_installs_the_recorded_arm_hparams_and_restores_the_table'])

    def test_X20_bundle_records_a_fixed_arm_table(self):
        self.kill("'arm_hparams': _jsonable(rr.PREREG_R['arms'][arm])}", "'arm_hparams': _jsonable(rr.PREREG_R['arms']['l1'])}",
                  [PIPE + 'test_fit_records_the_arm_of_the_decision_not_a_fixed_one'])

    def test_X21_bundle_records_a_fixed_arm(self):
        self.kill("'zero_norm_rows': int((np.linalg.norm(content, axis=1) == 0).sum()), 'recipe': RECIPE, 'arm': arm,",
                  "'zero_norm_rows': int((np.linalg.norm(content, axis=1) == 0).sum()), 'recipe': RECIPE, 'arm': 'l1',",
                  [PIPE + 'test_fit_records_the_arm_of_the_decision_not_a_fixed_one'])

    def test_X22_producer_lists_no_new_file(self):
        self.kill("PRODUCER_FILES_R = ('sid_tok_run_r.py', 'sid_rqvae_r.py', 'sid_tok_run.py',", "PRODUCER_FILES_R = ('sid_tok_run.py',",
                  [PIPE + 'test_producer_lists_the_new_source_files'])

    def test_X23_probe_keeps_non_finite_log_entries(self):
        self.kill('        if not _finite(entry):\n            break', '        if False:\n            break',
                  [PROBE + 'test_probe_nonfinite_run_is_recorded_as_a_valid_result'])

    def test_X24_override_gate_removed(self):
        self.kill('if given and not args.engineering:', 'if False:',
                  [PROBE + 'test_probe_argument_and_input_errors_exit_1_before_any_output',
                   FITNEG + 'test_variant_and_engineering_rules'])

    def test_X25_explicit_max_steps_not_required(self):
        self.kill("if args.engineering and 'max_steps' not in given:", 'if False:',
                  [PROBE + 'test_probe_argument_and_input_errors_exit_1_before_any_output',
                   FITNEG + 'test_variant_and_engineering_rules'])

    def test_X26_commit_format_not_checked(self):
        self.kill('if not _HEX40.fullmatch(args.code_commit):', 'if False:',
                  [PROBE + 'test_probe_argument_and_input_errors_exit_1_before_any_output'])

    def test_X27_failure_record_names_a_fixed_arm(self):
        self.kill("'arm': result['detail']['arm'],", "'arm': 'l2',",
                  [FITNEG + 'test_gate_failure_writes_failure_json_with_diagnostics_and_no_manifest'])

    def test_X28_frozen_constant_check_legs_removed(self):
        for leg in ("_jsonable(rr.PREREG_R) != _FROZEN_R['prereg_r']", "_jsonable(rr.PREREG_R_SELECT) != _FROZEN_R['select_r']",
                    "_jsonable(rq.PREREG) != run._FROZEN_PREREG"):
            with self.subTest(leg=leg):
                assert_killed(self, one(leg, 'False'), FROZEN_LIVE)

    def test_X29_verify_formal_constant_check_removed(self):
        self.kill('    config = load_config_r(args.config, args.config_sha)\n    if not args.engineering:\n        require_frozen_constants()\n    inputs',
                  '    config = load_config_r(args.config, args.config_sha)\n    inputs',
                  [VERNEG + 'test_formal_verify_refuses_changed_live_constants'])

    def test_X30_probe_result_is_not_validated_before_writing(self):
        self.kill('    rr.validate_probe(record)\n', '', [PROBE + 'test_probe_refuses_an_unusable_result_and_writes_nothing'])

    def test_X31_oversized_check_skipped_in_publication(self):
        self.kill('    assert_train_log_size(result[\'train_log\'])\n', '',
                  [SIZE_CLI + 'test_oversized_train_log_aborts_the_publication'])

    def test_X32_verify_requires_no_l_arm(self):
        self.kill("    require_arm(decision)\n    run._say(verify_bundle_r", "    run._say(verify_bundle_r",
                  [VERNEG + 'test_stop_decision_cannot_verify_a_bundle'])

    def test_X33_decision_not_bound_to_inputs_in_verify(self):
        self.kill('    check_decision_inputs(decision, inputs, content_sha)\n    loaded =', '    loaded =',
                  [VERNEG + 'test_decision_for_other_inputs_is_refused'])


if __name__ == '__main__':
    unittest.main()
