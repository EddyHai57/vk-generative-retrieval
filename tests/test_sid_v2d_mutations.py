"""G6-PREP step 6a mutants M31..M45 of sid_v2d (depth key, schema isolation, version registry, gates, weights, audit): all killed."""
import unittest

from tests.mutation_util import mutated, run_ids
from vkgr import sid_v2d

T = 'tests.test_sid_v2d.'
ALL = [T + 'RoundTrip', T + 'Versions', T + 'FailClosed']


class Mutants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import test_sid_v2d
        for name in ('d4', 'd3', 'rqvae'):  # build the shared worlds before any patch is active
            test_sid_v2d.world(name)

    def test_unmutated_selection_is_green(self):
        result = run_ids(ALL)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertGreaterEqual(result.testsRun, 32)

    def kill(self, old, new, ids=None, count=1):
        with mutated(sid_v2d, old, new, count):
            result = run_ids(ids or ALL)
        self.assertFalse(result.wasSuccessful(), f'mutant survived: {old!r} -> {new!r}')

    def test_M31_depth_value_not_compared(self):
        self.kill(" or type(manifest['depth']) is not int or manifest['depth'] != depth", '', [T + 'FailClosed.test_depth_value_mismatch'])

    def test_M32_depth_key_not_required(self):
        self.kill('MANIFEST_KEYS = frozenset(sid_v2.MANIFEST_KEYS | {\'depth\'})', 'MANIFEST_KEYS = frozenset(sid_v2.MANIFEST_KEYS)',
                  [T + 'RoundTrip', T + 'FailClosed.test_depth_key_is_required_and_must_match_the_version'])

    def test_M33_loader_also_accepts_the_depth_three_schema(self):
        self.kill('        if manifest[key] != value:\n            raise ValueError(f\'manifest {key} mismatch\')',
                  '        if manifest[key] != value and not (key == \'schema\' and manifest[key] == \'sidmap-v2\'):\n            raise ValueError(f\'manifest {key} mismatch\')',
                  [T + 'FailClosed.test_schema_and_protocol_and_task'])

    def test_M34_gates_passed_flag_not_checked(self):
        self.kill("sid_v2._check_gates_passed(manifest['gates'])", 'pass', [T + 'FailClosed.test_a_bundle_whose_own_gates_honestly_failed_is_refused'])

    def test_M35_catalog_identity_not_checked(self):
        self.kill('sid_v2._check_catalog(manifest, catalog_ids, ids)', 'pass', [T + 'FailClosed.test_wrong_content_digest_and_wrong_catalog'])

    def test_M36_weights_not_checked_against_the_mapping(self):
        self.kill('        _check_weights_d(state, ids, sids, content)', '        pass',
                  [T + 'FailClosed.test_weights_that_disagree_with_the_mapping_are_refused_even_when_the_audit_matches_them'])

    def test_M37_audit_not_recomputed(self):
        self.kill("    _check_audit_d(manifest, gd._json(directory / 'audit.json'), kind, branching, depth, ids, sids, content, state, engineering)",
                  '    pass', [T + 'FailClosed.test_audit_json_must_equal_the_recomputation', T + 'FailClosed.test_rqvae_audit_rq_section_is_recomputed'])

    def test_M38_gate_B2_dropped_from_the_recomputation(self):
        self.kill('required_steps=required, prereg=prereg_for(manifest[\'sid_version\'], len(ids), engineering))', 'required_steps=required, prereg=None)',
                  [T + 'RoundTrip.test_depth_four_bhkm'])

    def test_M39_formal_prereg_is_derived_instead_of_frozen(self):
        self.kill("if not engineering and sid_version == 'SID_BHKM_B16D4_S42':", 'if False:', [T + 'Versions.test_prereg_for'])

    def test_M40_codec_built_for_a_fixed_depth(self):
        self.kill('sd.ValidatedCodecD(SIDCatalog(ids, sids, branching, depth, []), depth)', 'sd.ValidatedCodecD(SIDCatalog(ids, sids, branching, depth, []), 3)',
                  [T + 'RoundTrip.test_depth_four_bhkm'])

    def test_M41_tokenizer_depth_not_compared(self):
        self.kill("(tokenizer['sid_version'], tokenizer['kind'], tokenizer['branching'], tokenizer['depth']) \\\n            != (manifest['sid_version'], kind, branching, depth)",
                  "(tokenizer['sid_version'], tokenizer['kind'], tokenizer['branching']) \\\n            != (manifest['sid_version'], kind, branching)",
                  [T + 'FailClosed.test_tokenizer_json_must_describe_the_manifest'])

    def test_M42_replay_label_accepted_in_formal_context(self):
        self.kill('allowed = ENGINEERING_SID_VERSIONS if engineering else SID_VERSIONS', 'allowed = ENGINEERING_SID_VERSIONS',
                  [T + 'Versions.test_replay_label_is_engineering_only_and_unknown_labels_are_refused'])

    def test_M43_weights_container_with_a_missing_codebook_accepted(self):
        self.kill("    if set(arrays) - {'K'} != set(expected):\n        raise ValueError('weights tensor names mismatch')", '',
                  [T + 'FailClosed.test_rqvae_weights_with_an_extra_tensor_are_refused'])

    def test_M44_weights_built_with_three_levels(self):
        self.kill('expected = rd.RQVAED(branching, depth).state_dict()', 'expected = rd.RQVAED(branching, 3).state_dict()', [T + 'RoundTrip.test_rqvae_depth_four'])

    def test_M45_writer_skips_the_new_directory_rule(self):
        self.kill("    if output.exists() or not output.parent.is_dir():\n        raise ValueError('output must be a new directory inside an existing one')\n    output.mkdir()",
                  '    output.mkdir(exist_ok=True)', [T + 'RoundTrip.test_output_must_be_a_new_directory'])


if __name__ == '__main__':
    unittest.main()
