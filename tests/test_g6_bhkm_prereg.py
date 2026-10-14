"""G6-PREP step 4: the pre-registered shape of SID_BHKM_B16D4_S42, re-derived from the capacity rule alone (no data read)."""
import unittest
from collections import Counter

from vkgr import sid_tok_audit_d as ad

N = 120415


def split_levels(n, branching, depth):
    """Independent plain-list restatement of the capacity rule: sizes of all nodes at every level."""
    levels = [[n]]
    for _ in range(depth):
        following = []
        for size in levels[-1]:
            k = min(branching, size)
            quotient, remainder = divmod(size, k)
            following += [quotient + 1] * remainder + [quotient] * (k - remainder)
        levels.append(following)
    return levels


class Capacity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = split_levels(N, 16, 4)

    def test_first_levels(self):
        self.assertEqual(Counter(self.levels[1]), {7526: 15, 7525: 1})
        self.assertEqual(sorted(Counter(self.levels[2])), [470, 471])
        self.assertEqual(dict(Counter(self.levels[3])), {29: 2465, 30: 1631})  # the depth-3 anchor leaves
        self.assertEqual(len(self.levels[3]), 4096)

    def test_every_depth_three_node_is_wide_enough_to_fill_sixteen_children(self):
        self.assertGreaterEqual(min(self.levels[3]), 16)
        self.assertEqual([min(16, size) for size in self.levels[3]].count(16), 4096)

    def test_depth_four_leaves(self):
        leaves = Counter(self.levels[4])
        self.assertEqual(len(self.levels[4]), 65536)
        self.assertEqual(dict(leaves), {1: 10657, 2: 54879})
        self.assertEqual(sum(self.levels[4]), N)
        self.assertEqual(sum(s * c for s, c in leaves.items() if s > 1), 109758)

    def test_capacity_table_agrees_with_the_plain_restatement(self):
        table = ad.capacity_table(N, 16, 4)
        self.assertEqual(table['leaf_sizes'], dict(sorted(Counter(self.levels[4]).items())))
        self.assertEqual((table['leaves'], table['distinct'], table['suffix_vocabulary'], table['vocab_size']), (65536, [16, 16, 16, 16], 2, 68))
        self.assertEqual((table['semantic_collision_groups'], table['semantic_collision_items'], table['singleton_leaves'], table['items_needing_suffix']),
                         (54879, 109758, 10657, 54879))


class FrozenBlock(unittest.TestCase):
    def test_block_equals_what_the_capacity_rule_gives(self):
        t = ad.capacity_table(N, 16, 4)
        expected = {
            'items': N, 'full_sid_collisions': 0, 'leaves.count': t['leaves'], 'leaves.max': max(t['leaf_sizes']),
            'leaves.singletons': t['singleton_leaves'],
            'positions.0.distinct': t['distinct'][0], 'positions.1.distinct': t['distinct'][1],
            'positions.2.distinct': t['distinct'][2], 'positions.3.distinct': t['distinct'][3],
            'positions.4.distinct': t['suffix_vocabulary'],
            'prefix_groups.1': 16, 'prefix_groups.2': 256, 'prefix_groups.3': 4096, 'prefix_groups.4': t['leaves'],
            'suffix_vocabulary': t['suffix_vocabulary'], 'vocab_size': t['vocab_size'],
            'semantic_collision_groups': t['semantic_collision_groups'], 'semantic_collision_items': t['semantic_collision_items'],
            'items_needing_suffix': t['items_needing_suffix']}
        self.assertEqual(ad.PREREG_BHKM_B16D4, expected)

    def test_the_numbers_of_the_design_sheet(self):
        block = ad.PREREG_BHKM_B16D4
        self.assertEqual((block['leaves.count'], block['suffix_vocabulary'], block['vocab_size']), (65536, 2, 68))
        self.assertEqual((block['semantic_collision_groups'], block['semantic_collision_items'], block['leaves.singletons']), (54879, 109758, 10657))
        self.assertEqual(block['vocab_size'], 2 + 4 * 16 + block['suffix_vocabulary'])

    def test_depth_three_anchor_is_not_the_frozen_block(self):
        anchor = ad.capacity_table(N, 16, 3)
        self.assertNotEqual(anchor['leaves'], ad.PREREG_BHKM_B16D4['leaves.count'])
        self.assertEqual((anchor['suffix_vocabulary'], anchor['vocab_size']), (30, 80))


if __name__ == '__main__':
    unittest.main()
