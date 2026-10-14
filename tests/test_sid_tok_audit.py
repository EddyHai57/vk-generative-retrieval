"""GTOK-01 FEAT-002: tokenizer audit report items, hard gates and comparison helpers."""
import copy
import json
import math
import unittest

import numpy as np

from vkgr import sid_tok_audit as au


def balanced_sids(n, b):
    """Canonical balanced hierarchical mapping for item rows 0..n-1 (children get n//k or n//k+1 items)."""
    sids = np.zeros((n, 4), dtype=np.uint32)

    def split(lo, hi, depth):
        size = hi - lo
        if depth == 3:
            sids[lo:hi, 3] = np.arange(size)
            return
        k = min(b, size)
        base, extra = divmod(size, k)
        position = lo
        for child in range(k):
            width = base + (1 if child < extra else 0)
            sids[position:position + width, depth] = child
            split(position, position + width, depth + 1)
            position += width

    split(0, n, 0)
    return sids


def content_from_first_column(values):
    content = np.zeros((len(values), 64), dtype=np.float32)
    content[:, 0] = values
    return content


# Hand-built example (b=2, N=10): leaves (000)=3 items, (001)=1, (010)=2, (100)=1, (111)=3.
HAND_IDS = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.uint32)
HAND_SIDS = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1],
                      [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 2]], dtype=np.uint32)
HAND_X = [0, 2, 4, 10, 20, 22, 36, 40, 42, 44]  # global mean 22, SST = 2760


def hand_audit(kind='bhkm'):
    return au.audit_mapping(HAND_IDS, HAND_SIDS, content_from_first_column(HAND_X), branching=2, kind=kind)


def entropy_perplexity(counts):
    p = [c / sum(counts) for c in counts]
    return math.exp(-sum(x * math.log(x) for x in p))


class HandComputed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = hand_audit()

    def test_legality_and_structure_flags(self):
        a = self.a
        self.assertEqual((a['items'], a['full_sid_collisions'], a['roundtrip_items'], a['reachable_items'],
                          a['valid_prefix_deadends']), (10, 0, 10, 10, 0))
        self.assertEqual((a['ids_ascending'], a['digits_in_range'], a['suffix_canonical'], a['all_finite']),
                         (True, True, True, True))
        self.assertEqual((a['kind'], a['branching']), ('bhkm', 2))

    def test_positions_distinct_utilization_perplexity(self):
        expected = [(2, 2, [6, 4]), (2, 2, [5, 5]), (2, 2, [6, 4]), (3, 3, [5, 3, 2])]  # base, distinct, counts
        for position, (base, distinct, counts) in zip(self.a['positions'], expected):
            self.assertEqual((position['base'], position['distinct']), (base, distinct))
            self.assertEqual(position['utilization'], 1.0)
            self.assertAlmostEqual(position['perplexity'], entropy_perplexity(counts), places=12)

    def test_leaf_statistics(self):
        leaves = self.a['leaves']  # leaf sizes sorted: 1 1 2 3 3
        self.assertEqual((leaves['count'], leaves['max'], leaves['median'], leaves['p95'], leaves['p99']), (5, 3, 2.0, 3, 3))
        self.assertEqual((leaves['mean'], leaves['singletons'], leaves['distinct_sizes']), (2.0, 2, 3))
        self.assertEqual(leaves['histogram_head'], [[1, 2], [2, 1], [3, 2]])
        self.assertEqual(self.a['suffix_vocabulary'], 3)

    def test_collisions_are_counted_on_the_first_three_codes(self):
        a = self.a
        self.assertEqual((a['semantic_collision_groups'], a['semantic_collision_items'], a['items_needing_suffix']), (3, 8, 5))

    def test_prefix_groups_and_vocabulary(self):
        self.assertEqual(self.a['prefix_groups'], {'1': 2, '2': 4, '3': 5})
        self.assertEqual(self.a['vocab_size'], 2 + 3 * 2 + 3)

    def test_explained_variance_hand_computed(self):
        # SST 2760; SSE by first digit 1435/3, by first two digits 66, by leaf 18 (see derivation in the plan notes)
        ev = self.a['explained_variance']
        self.assertAlmostEqual(ev['1'], 1 - (1435 / 3) / 2760, places=12)
        self.assertAlmostEqual(ev['2'], 1 - 66 / 2760, places=12)
        self.assertAlmostEqual(ev['3'], 1 - 18 / 2760, places=12)

    def test_audit_is_json_stable(self):
        again = json.loads(json.dumps(self.a, allow_nan=False))
        self.assertTrue(au.audits_equal(self.a, again))
        self.assertEqual(again, self.a)  # only lists, str keys and python scalars

    def test_degenerate_content_and_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            au.audit_mapping(HAND_IDS, HAND_SIDS, np.ones((10, 64), dtype=np.float32), branching=2, kind='bhkm')
        with self.assertRaises(ValueError):
            au.audit_mapping(HAND_IDS, HAND_SIDS, content_from_first_column(HAND_X), branching=2, kind='other')
        with self.assertRaises(ValueError):
            au.audit_mapping(HAND_IDS, HAND_SIDS[:, :3], content_from_first_column(HAND_X), branching=2, kind='bhkm')
        with self.assertRaises(ValueError):
            au.audit_mapping(HAND_IDS, HAND_SIDS, content_from_first_column(HAND_X)[:9], branching=2, kind='bhkm')


class T0Shape(unittest.TestCase):
    """Rebuild the P5 facts directly: N=120,415, 16-ary, 4096 leaves, 1631 with 30 items and 2465 with 29."""

    @classmethod
    def setUpClass(cls):
        n = 120415
        cls.ids = (np.arange(n) * 3 + 7).astype(np.uint32)
        cls.sids = balanced_sids(n, 16)
        cls.content = np.random.RandomState(0).randn(n, 64).astype(np.float32)
        cls.a = au.audit_mapping(cls.ids, cls.sids, cls.content, branching=16, kind='bhkm')

    def test_p5_leaf_facts(self):
        leaves = self.a['leaves']
        self.assertEqual((leaves['count'], leaves['max'], leaves['median'], leaves['p95'], leaves['p99']),
                         (4096, 30, 29.0, 30, 30))
        self.assertEqual(leaves['histogram_head'], [[29, 2465], [30, 1631]])
        self.assertEqual((self.a['suffix_vocabulary'], self.a['vocab_size']), (30, 80))
        self.assertEqual(self.a['prefix_groups'], {'1': 16, '2': 256, '3': 4096})
        self.assertEqual((self.a['semantic_collision_groups'], self.a['semantic_collision_items'],
                          self.a['items_needing_suffix']), (4096, 120415, 120415 - 4096))

    def test_legality_balance_and_gates(self):
        a = self.a
        self.assertEqual((a['full_sid_collisions'], a['roundtrip_items'], a['reachable_items'], a['valid_prefix_deadends']),
                         (0, 120415, 120415, 0))
        self.assertTrue(a['balanced_children']['balanced'])
        result = au.gates(a, 'bhkm', expected_content_sha256=a['content_sha256'])
        self.assertTrue(result['passed'], result)


def good_rq(steps=100):
    return {'status': 'complete', 'steps': steps, 'max_steps': steps, 'elapsed': 5.0, 'wall_cap_seconds': 7200,
            'reconstruction': {'mse': 0.2, 'sst': 1.0, 'r2': 0.8, 'residual_norms': [1.0, 0.6, 0.3, 0.1]}}


class Gates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 4096  # 16 x 16 x 16 leaves of one item: balanced, every digit value used
        cls.ids = np.arange(n, dtype=np.uint32) * 2 + 1
        cls.sids = balanced_sids(n, 16)
        cls.content = np.random.RandomState(1).randn(n, 64).astype(np.float32)
        cls.base = au.audit_mapping(cls.ids, cls.sids, cls.content, branching=16, kind='bhkm')
        cls.digest = cls.base['content_sha256']

    def bhkm(self, audit):
        return au.gates(audit, 'bhkm', expected_content_sha256=self.digest)

    def rqvae(self, audit, rq=None, **kw):
        audit = copy.deepcopy(audit)
        audit['kind'] = 'rqvae'
        return au.gates(audit, 'rqvae', good_rq(5) if rq is None else rq, expected_content_sha256=self.digest,
                        required_steps=5, **kw)

    def failed(self, result):
        return {c['id'] for c in result['checks'] if not c['ok']}

    def test_clean_audits_pass_with_the_right_check_groups(self):
        result = self.bhkm(self.base)
        self.assertTrue(result['passed'])
        self.assertEqual([c['id'] for c in result['checks']], ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'B1'])
        result = self.rqvae(self.base)
        self.assertTrue(result['passed'], result)
        self.assertEqual([c['id'] for c in result['checks']], ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'R1', 'R2', 'R3', 'R4'])
        with self.assertRaises(ValueError):
            au.gates(self.base, 'other')

    def test_gates_are_json_stable(self):
        result = self.bhkm(self.base)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_H1_H2_full_sid_collision(self):
        sids = self.sids.copy()
        sids[1] = sids[0]
        a = au.audit_mapping(self.ids, sids, self.content, branching=16, kind='bhkm')
        failed = self.failed(self.bhkm(a))
        self.assertEqual(a['full_sid_collisions'], 1)
        self.assertTrue({'H1', 'H2', 'H4'} <= failed)
        self.assertNotIn('H3', failed)

    def test_H3_deadend(self):
        a = copy.deepcopy(self.base)
        a['valid_prefix_deadends'] = 1
        self.assertEqual(self.failed(self.bhkm(a)), {'H3'})

    def test_H4_noncanonical_suffix_range_and_order(self):
        sids = self.sids.copy()
        sids[0, 3] = 5  # leaf of one item must have suffix 0
        a = au.audit_mapping(self.ids, sids, self.content, branching=16, kind='bhkm')
        self.assertEqual(self.failed(self.bhkm(a)), {'H4'})
        sids = self.sids.copy()
        sids[0, 0] = 16  # digit outside the 16-ary range: legality cannot be established, so it fails closed
        a = au.audit_mapping(self.ids, sids, self.content, branching=16, kind='bhkm')
        self.assertFalse(a['digits_in_range'])
        self.assertTrue({'H1', 'H2', 'H3', 'H4'} <= self.failed(self.bhkm(a)))
        ids = self.ids.copy()
        ids[[0, 1]] = ids[[1, 0]]
        a = au.audit_mapping(ids, self.sids, self.content, branching=16, kind='bhkm')
        self.assertFalse(a['ids_ascending'])
        self.assertIn('H4', self.failed(self.bhkm(a)))

    def test_H5_content_digest(self):
        result = au.gates(self.base, 'bhkm', expected_content_sha256='0' * 64)
        self.assertEqual(self.failed(result), {'H5'})
        self.assertEqual(au.CONTENT_V1_FEATURE_SHA256, '766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84')
        self.assertEqual(self.failed(au.gates(self.base, 'bhkm')), {'H5'})  # default pin is CONTENT_V1

    def test_H6_nonfinite_numbers(self):
        content = self.content.copy()
        content[3, 7] = np.nan
        a = au.audit_mapping(self.ids, self.sids, content, branching=16, kind='bhkm')
        self.assertFalse(a['all_finite'])
        self.assertIsNone(a['explained_variance'])
        self.assertEqual(self.failed(au.gates(a, 'bhkm', expected_content_sha256=a['content_sha256'])), {'H6'})
        rq = good_rq(5)
        rq['reconstruction']['mse'] = float('inf')
        self.assertIn('H6', self.failed(self.rqvae(self.base, rq)))

    def test_B1_unbalanced_tree_only_for_bhkm(self):
        sids = self.sids.copy()
        sids[:, 0] = np.where(np.arange(len(sids)) < 3000, 0, 1)  # two children hold 3000 / 1096 items
        a = au.audit_mapping(self.ids, sids, self.content, branching=16, kind='bhkm')
        self.assertIn('B1', self.failed(self.bhkm(a)))
        self.assertNotIn('B1', [c['id'] for c in self.rqvae(a)['checks']])

    def test_R1_usage_boundary_is_integer_exact(self):
        for K, low, high in ((256, 204, 205), (16, 12, 13)):
            for position in range(3):
                for used, ok in ((low, False), (high, True)):
                    a = copy.deepcopy(self.base)
                    a['branching'] = K
                    for d in range(3):
                        a['positions'][d]['distinct'] = K
                    a['positions'][position]['distinct'] = used
                    with self.subTest(K=K, position=position, used=used):
                        result = self.rqvae(a)
                        self.assertEqual('R1' not in self.failed(result), ok)
        a = copy.deepcopy(self.base)  # exactly 80 percent passes (>=, not >): 8 of 10
        a['branching'] = 10
        for d in range(3):
            a['positions'][d]['distinct'] = 8
        self.assertNotIn('R1', self.failed(self.rqvae(a)))
        a['positions'][1]['distinct'] = 7
        self.assertIn('R1', self.failed(self.rqvae(a)))

    def test_R2_training_length_status_and_wall(self):
        for change in ({'status': 'fail_wall'}, {'steps': 4}, {'max_steps': 6}, {'elapsed': 7201.0}):
            rq = {**good_rq(5), **change}
            with self.subTest(change=change):
                self.assertEqual(self.failed(self.rqvae(self.base, rq)), {'R2'})
        self.assertEqual(self.failed(self.rqvae(self.base, {**good_rq(5), 'elapsed': 7200.0})), set())
        self.assertIn('R2', self.failed(au.gates(self.base, 'rqvae', None, expected_content_sha256=self.digest)))
        self.assertEqual(au.REQUIRED_STEPS, 300000)

    def test_R3_reconstruction_r2_positive(self):
        for mse, ok in ((0.999999, True), (1.0, False), (1.5, False)):
            rq = good_rq(5)
            rq['reconstruction'].update(mse=mse, sst=1.0)
            self.assertEqual('R3' not in self.failed(self.rqvae(self.base, rq)), ok)

    def test_R4_residual_norms_strictly_decrease(self):
        for norms, ok in (([1.0, 0.6, 0.3, 0.1], True), ([1.0, 0.6, 0.6, 0.1], False), ([1.0, 0.6, 0.7, 0.1], False)):
            rq = good_rq(5)
            rq['reconstruction']['residual_norms'] = norms
            self.assertEqual('R4' not in self.failed(self.rqvae(self.base, rq)), ok)


class BalancedChildren(unittest.TestCase):
    def test_balanced_trees(self):
        for n, b in ((8, 2), (7, 2), (1, 2), (100, 4), (4096, 16)):
            with self.subTest(n=n, b=b):
                found = au.balanced_children(balanced_sids(n, b), b)
                self.assertTrue(found['balanced'], found)
                self.assertLessEqual(found['max_spread'], 1)

    def test_spread_two_is_a_violation_unless_tolerance_two_is_explicit(self):
        # root children hold 4 / 2 items (spread 2); every deeper node is balanced
        sids = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint32)
        found = au.balanced_children(sids, 2)
        self.assertFalse(found['balanced'])
        self.assertEqual(found['max_spread'], 2)
        self.assertTrue(au.balanced_children(sids, 2, tolerance=2)['balanced'])

    def test_missing_child_is_a_violation(self):
        sids = balanced_sids(8, 2)
        sids[:4, 1] = 0  # node with 4 items must have min(2, 4) = 2 children
        self.assertFalse(au.balanced_children(sids, 2)['balanced'])
        self.assertEqual(au.BALANCE_TOLERANCE, 1)

    def test_hand_example_is_unbalanced_at_the_root(self):
        found = au.balanced_children(HAND_SIDS, 2)
        self.assertFalse(found['balanced'])
        self.assertEqual(found['max_spread'], 2)


class AuditsEqual(unittest.TestCase):
    def test_float_tolerance_but_exact_everything_else(self):
        self.assertTrue(au.audits_equal({'x': 1.0, 'y': [1, 2]}, {'x': 1.0 + 1e-12, 'y': [1, 2]}))
        self.assertFalse(au.audits_equal({'x': 1.0}, {'x': 1.0 + 1e-6}))
        self.assertTrue(au.audits_equal(0.0, 0.0))
        self.assertFalse(au.audits_equal(float('nan'), float('nan')))
        self.assertFalse(au.audits_equal(1, 2))
        self.assertFalse(au.audits_equal(10 ** 12, 10 ** 12 + 1))  # integers never use the tolerance
        self.assertFalse(au.audits_equal(1, 1.0))
        self.assertFalse(au.audits_equal(True, 1))
        self.assertFalse(au.audits_equal('a', 'b'))
        self.assertTrue(au.audits_equal(None, None))
        self.assertFalse(au.audits_equal(None, 0))
        self.assertFalse(au.audits_equal([1, 2], [1, 2, 3]))
        self.assertFalse(au.audits_equal({'a': 1}, {'a': 1, 'b': 2}))
        self.assertTrue(au.audits_equal({'a': [{'b': 1.0}]}, {'a': [{'b': 1.0 + 1e-13}]}))
        self.assertTrue(au.audits_equal(1.0, 1.0 + 1e-6, rel=1e-3))


class CompareAndRender(unittest.TestCase):
    def test_compare_two_audits_and_markdown(self):
        other = au.audit_mapping(HAND_IDS, HAND_SIDS, content_from_first_column(HAND_X), branching=2, kind='rqvae')
        table = au.compare({'T0': hand_audit(), 'T1': other})
        self.assertEqual(table['labels'], ['T0', 'T1'])
        rows = {row['item']: row['values'] for row in table['rows']}
        self.assertEqual(rows['leaves.count'], {'T0': 5, 'T1': 5})
        self.assertEqual(rows['suffix_vocabulary'], {'T0': 3, 'T1': 3})
        text = au.render_markdown(table)
        lines = text.splitlines()
        self.assertEqual(lines[0], '| item | T0 | T1 |')
        self.assertEqual(lines[1], '|---|---|---|')
        self.assertIn('| leaves.count | 5 | 5 |', lines)
        self.assertTrue(text.endswith('\n'))
        with self.assertRaises(KeyError):  # a label that is not in the table is an error, not a blank cell
            au.compare({'T0': {}})


if __name__ == '__main__':
    unittest.main()
