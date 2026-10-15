"""G6-PREP step 3 mutants M11..M17 of sid_depth_beam (position count, last-position offset, tie-breaks): all must be killed."""
import unittest

from tests.mutation_util import assert_killed, run_ids
from vkgr import sid_depth_beam as sdb

T = 'tests.test_sid_depth_beam.'
ALL = [T + 'FrozenDepthThree', T + 'DepthFour', T + 'TieBreaks']


class Mutants(unittest.TestCase):
    def test_unmutated_selection_is_green(self):
        result = run_ids(ALL)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertGreaterEqual(result.testsRun, 12)

    def kill(self, old, new, ids=None, count=1):
        with self.subTest(old=old):
            from tests import mutation_util
            with mutation_util.mutated(sdb, old, new, count):
                result = run_ids(ids or ALL)
            self.assertFalse(result.wasSuccessful(), f'mutant survived: {old!r} -> {new!r}')

    def test_M11_serial_beam_loops_over_four_positions(self):
        self.kill('for position in range(catalog.depth + 1):\n        logits = score', 'for position in range(4):\n        logits = score', [T + 'DepthFour'])

    def test_M12_group_decoder_loops_over_four_positions(self):
        self.kill('for position in range(catalog.depth + 1):\n        counts', 'for position in range(4):\n        counts', [T + 'DepthFour'])

    def test_M13_last_position_has_no_offset(self):
        self.kill('total + float(logprob[row, codec.offsets[position] + digit])',
                  'total + float(logprob[row, digit if position == catalog.depth else codec.offsets[position] + digit])', [T + 'DepthFour', T + 'FrozenDepthThree'])

    def test_M14_sort_without_path_tie_break(self):
        self.kill('expanded.sort(key=lambda x: (-x[1], x[0]))', 'expanded.sort(key=lambda x: -x[1])', [T + 'TieBreaks.test_equal_scores_are_ordered_by_path_not_by_generation_order'])

    def test_M15_final_ranking_without_item_tie_break(self):
        self.kill('ranked.sort(key=lambda x: (-x[1], x[0]))\n    audit', 'ranked.sort(key=lambda x: -x[1])\n    audit',
                  [T + 'TieBreaks.test_final_ranking_breaks_ties_by_item_id_not_by_path'])

    def test_M16_group_decoder_ignores_path_in_lexsort(self):
        self.kill('order = np.lexsort(tuple(expanded[:, j] for j in reversed(range(position + 1))) + (-score,))[:beam]',
                  'order = np.argsort(-score, kind="stable")[:beam]', [T + 'TieBreaks.test_group_decoder_breaks_the_same_tie_the_same_way'])

    def test_M17_beam_not_truncated(self):
        self.kill('active = expanded[:beam]', 'active = expanded', [T + 'TieBreaks', T + 'DepthFour.test_every_returned_item_is_a_catalog_item_and_unique'])


if __name__ == '__main__':
    unittest.main()
