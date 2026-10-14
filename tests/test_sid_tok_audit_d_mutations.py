"""G6-PREP step 4 mutants M18..M27 of sid_tok_audit_d (levels covered by R1/B1, vocabulary, frozen block): all must be killed."""
import unittest

from tests.mutation_util import mutated, run_ids
from vkgr import sid_tok_audit_d as ad

A = 'tests.test_sid_tok_audit_d.'
P = 'tests.test_g6_bhkm_prereg.'
ALL = [A + 'DepthThreeEqualsFrozenAudit', A + 'HandComputedDepthFour', A + 'RealFitMatchesTheCapacityRule', A + 'BalanceAtEveryLevel',
       A + 'DepthFourGates', A + 'VocabularyAndAgreement', P + 'Capacity', P + 'FrozenBlock']


class Mutants(unittest.TestCase):
    def test_unmutated_selection_is_green(self):
        result = run_ids(ALL)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertGreaterEqual(result.testsRun, 27)

    def kill(self, old, new, ids=None, count=1):
        with mutated(ad, old, new, count):
            result = run_ids(ids or ALL)
        self.assertFalse(result.wasSuccessful(), f'mutant survived: {old!r} -> {new!r}')

    def test_M18_R1_looks_at_three_levels_only(self):
        self.kill("used = [audit['positions'][d]['distinct'] for d in range(depth)]", "used = [audit['positions'][d]['distinct'] for d in range(3)]",
                  [A + 'DepthFourGates'])

    def test_M19_B1_stops_one_level_early(self):
        self.kill('    for level in range(depth):\n        order, starts', '    for level in range(depth - 1):\n        order, starts',
                  [A + 'BalanceAtEveryLevel', A + 'DepthThreeEqualsFrozenAudit'])

    def test_M20_vocabulary_formula_uses_three_digits(self):
        self.kill("'vocab_size': int(2 + depth * branching + suffix_vocab)", "'vocab_size': int(2 + 3 * branching + suffix_vocab)",
                  [A + 'VocabularyAndAgreement', A + 'HandComputedDepthFour'])

    def test_M21_frozen_block_leaf_count_is_the_depth_three_one(self):
        self.kill("'leaves.count': 65536,", "'leaves.count': 4096,", [P + 'FrozenBlock'])

    def test_M22_frozen_block_suffix_vocabulary_is_the_depth_three_one(self):
        self.kill("'suffix_vocabulary': 2,", "'suffix_vocabulary': 30,", [P + 'FrozenBlock'])

    def test_M23_explained_variance_stops_at_three(self):
        self.kill('for d in range(1, depth + 1)}\n\n\ndef balanced', 'for d in range(1, 4)}\n\n\ndef balanced', [A + 'HandComputedDepthFour'])

    def test_M24_prefix_groups_stop_at_three(self):
        self.kill("for d in range(1, depth + 1)}\n    return {", "for d in range(1, 4)}\n    return {", [A + 'HandComputedDepthFour'])

    def test_M25_R4_accepts_the_wrong_number_of_norms(self):
        self.kill('len(norms) == depth + 1 and ', '', [A + 'DepthFourGates.test_R4_needs_five_strictly_decreasing_norms'])

    def test_M26_B2_ignores_mismatches(self):
        self.kill('if type(actual) is not type(expected) or actual != expected:', 'if False:', [A + 'DepthFourGates.test_bhkm_B1_and_B2'])

    def test_M27_capacity_rule_drops_the_remainder(self):
        self.kill('following[quotient + 1] += count * remainder', 'following[quotient] += count * remainder', [P + 'Capacity', A + 'RealFitMatchesTheCapacityRule'])

    def test_M28_suffix_canonical_ignores_the_last_semantic_digit(self):
        self.kill('tuple(sids[:, j] for j in reversed(range(depth)))', 'tuple(sids[:, j] for j in reversed(range(depth - 1)))',
                  [A + 'HandComputedDepthFour', A + 'DepthThreeEqualsFrozenAudit'])


if __name__ == '__main__':
    unittest.main()
