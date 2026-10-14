"""G6-PREP step 2 mutants M1..M10 of sid_depth (depth parameter, suffix allocation, codec check): all must be killed."""
import unittest

from tests.mutation_util import assert_killed, run_ids
from vkgr import sid_depth as sd

T = 'tests.test_sid_depth.'
ALL = [T + 'DepthThreeMatchesFrozenCheck', T + 'DepthFourCheck', T + 'TokenClasses', T + 'ValidatedCodec', T + 'AssignSuffix']


class Mutants(unittest.TestCase):
    def test_unmutated_selection_is_green(self):
        result = run_ids(ALL)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertGreaterEqual(result.testsRun, 23)

    def kill(self, old, new, ids=None):
        assert_killed(self, sd, old, new, ids or ALL)

    def test_M1_maximum_depth_off_by_one(self):
        self.kill('MAX_DEPTH = 8', 'MAX_DEPTH = 7', [T + 'DepthFourCheck.test_depth_argument_range'])

    def test_M2_lexsort_drops_a_semantic_column(self):
        self.kill('tuple(model.sids[:, j] for j in reversed(range(depth)))', 'tuple(model.sids[:, j] for j in reversed(range(depth - 1)))')

    def test_M3_suffix_starts_at_one(self):
        self.kill('out[order, depth] = rank', 'out[order, depth] = rank + 1', [T + 'AssignSuffix'])

    def test_M4_bases_check_misses_a_digit(self):
        self.kill('[model.branching] * depth +', '[model.branching] * (depth - 1) +')

    def test_M5_token_offsets_shifted(self):
        self.kill('        total = 2\n', '        total = 3\n', [T + 'TokenClasses'])

    def test_M6_validated_codec_ignores_the_depth_argument(self):
        self.kill('check_codec_v2(self, depth)', 'check_codec_v2(self, 3)', [T + 'ValidatedCodec'])

    def test_M7_canonical_suffix_groups_by_three_digits_only(self):
        self.kill('codes[1:, :depth] != codes[:-1, :depth]', 'codes[1:, :3] != codes[:-1, :3]', [T + 'DepthFourCheck', T + 'ValidatedCodec'])

    def test_M8_minimum_depth_off_by_one(self):
        self.kill('MIN_DEPTH = 2', 'MIN_DEPTH = 1', [T + 'DepthFourCheck.test_depth_argument_range'])

    def test_M9_catalog_depth_attribute_not_compared(self):
        self.kill(' or model.depth != depth', '', [T + 'DepthThreeMatchesFrozenCheck', T + 'DepthFourCheck'])

    def test_M10_window_check_removed(self):
        self.kill(' or max_len != WINDOW', ' or max_len < 0', [T + 'TokenClasses.test_windowed_tokens_depth_four_and_registered_window_only'])


if __name__ == '__main__':
    unittest.main()
