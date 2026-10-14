"""G6-PREP step 4: depth-generic tokenizer audit and gates. Depth 3 must reproduce the frozen audit exactly."""
import copy
import unittest

import numpy as np

from tests import g6_fixtures as fx
from vkgr import sid as bhkm
from vkgr import sid_depth as sd
from vkgr import sid_tok_audit as au
from vkgr import sid_tok_audit_d as ad
from vkgr.data_audit import canonical_bytes

CONTENT_SHA = au.CONTENT_V1_FEATURE_SHA256


def content_for(n, seed=0, dim=64):
    return np.random.RandomState(seed).normal(size=(n, dim)).astype(np.float32)


balanced_codes, mapping_from = fx.balanced_codes, fx.mapping_from


def audit_d(codes, depth, branching, kind='bhkm', content=None, seed=0):
    ids, sids = mapping_from(codes, seed)
    content = content_for(len(ids), seed) if content is None else content
    return ad.audit_mapping_d(ids, sids, content, branching=branching, kind=kind, depth=depth)


class DepthThreeEqualsFrozenAudit(unittest.TestCase):
    def mappings(self):
        for seed in range(3):
            ids, sids = fx.synth(250, 4, 3, seed)
            yield ids, sids, content_for(250, seed)
        ids, sids = mapping_from(balanced_codes(300, 5, 3))
        yield ids, sids, content_for(300, 9)

    def test_audit_is_byte_identical(self):
        for ids, sids, content in self.mappings():
            for kind in ('bhkm', 'rqvae'):
                b = 5 if len(ids) == 300 else 4
                with self.subTest(kind=kind, n=len(ids)):
                    old = au.audit_mapping(ids, sids, content, branching=b, kind=kind)
                    new = ad.audit_mapping_d(ids, sids, content, branching=b, kind=kind, depth=3)
                    self.assertEqual(canonical_bytes(new), canonical_bytes(old))
                    self.assertTrue(au.audits_equal(new, old))

    def test_balanced_children_matches(self):
        for ids, sids, _ in self.mappings():
            b = 5 if len(ids) == 300 else 4
            self.assertEqual(ad.balanced_children_d(sids, b, 3), au.balanced_children(sids, b))
            self.assertEqual(ad.balanced_children_d(sids, b, 3, tolerance=2), au.balanced_children(sids, b, tolerance=2))

    def test_gates_match_for_both_kinds(self):
        ids, sids, content = next(iter(self.mappings()))
        old = au.audit_mapping(ids, sids, content, branching=4, kind='rqvae')
        new = ad.audit_mapping_d(ids, sids, content, branching=4, kind='rqvae', depth=3)
        rq = {'status': 'complete', 'steps': 7, 'max_steps': 7, 'elapsed': 1.0, 'wall_cap_seconds': 5.0,
              'reconstruction': {'mse': 0.5, 'sst': 1.0, 'r2': 0.5, 'residual_norms': [4.0, 3.0, 2.0, 1.0]}}
        self.assertEqual(ad.gates_d(new, 'rqvae', rq, depth=3, expected_content_sha256=new['content_sha256'], required_steps=7),
                         au.gates(old, 'rqvae', rq, expected_content_sha256=old['content_sha256'], required_steps=7))
        old_b = au.audit_mapping(ids, sids, content, branching=4, kind='bhkm')
        new_b = ad.audit_mapping_d(ids, sids, content, branching=4, kind='bhkm', depth=3)
        self.assertEqual(ad.gates_d(new_b, 'bhkm', None, depth=3), au.gates(old_b, 'bhkm', None))


class HandComputedDepthFour(unittest.TestCase):
    IDS = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint32)
    SIDS = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=np.uint32)
    CONTENT = np.array([[0, 0], [2, 0], [8, 0], [10, 0], [20, 0], [24, 0]], dtype=np.float32)

    def setUp(self):
        self.audit = ad.audit_mapping_d(self.IDS, self.SIDS, self.CONTENT, branching=2, kind='bhkm', depth=4)

    def test_counts(self):
        a = self.audit
        self.assertEqual((a['items'], a['suffix_vocabulary'], a['vocab_size']), (6, 2, 2 + 4 * 2 + 2))
        self.assertEqual([p['distinct'] for p in a['positions']], [2, 2, 2, 2, 2])
        self.assertEqual([p['base'] for p in a['positions']], [2, 2, 2, 2, 2])
        self.assertEqual(a['prefix_groups'], {'1': 2, '2': 3, '3': 3, '4': 4})
        leaves = a['leaves']
        self.assertEqual((leaves['count'], leaves['max'], leaves['singletons'], leaves['median']), (4, 2, 2, 1.5))
        self.assertEqual((a['semantic_collision_groups'], a['semantic_collision_items'], a['items_needing_suffix']), (2, 4, 2))
        self.assertEqual((a['full_sid_collisions'], a['roundtrip_items'], a['reachable_items'], a['valid_prefix_deadends']), (0, 6, 6, 0))
        self.assertTrue(a['suffix_canonical'] and a['ids_ascending'] and a['digits_in_range'])
        self.assertEqual(sorted(a['explained_variance']), ['1', '2', '3', '4'])
        self.assertAlmostEqual(a['positions'][0]['perplexity'], 2.0, places=12)

    def test_explained_variance_against_a_plain_reference(self):
        x = self.CONTENT.astype(np.float64)
        sst = ((x - x.mean(0)) ** 2).sum()
        for depth in (1, 2, 3, 4):
            groups = {}
            for row, sid in enumerate(self.SIDS):
                groups.setdefault(tuple(sid[:depth]), []).append(row)
            sse = sum(((x[r] - x[r].mean(0)) ** 2).sum() for r in groups.values())
            self.assertAlmostEqual(self.audit['explained_variance'][str(depth)], 1 - sse / sst, places=12)

    def test_unbalanced_tree_is_reported(self):
        self.assertFalse(self.audit['balanced_children']['balanced'])


class RealFitMatchesTheCapacityRule(unittest.TestCase):
    """The frozen ``sid.fit`` at depth 4: sizes follow n//k with the remainder to the first children (design section 3)."""

    def fitted(self, n, branching, depth):
        rng = np.random.RandomState(n)
        features = rng.normal(size=(n, 64)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        ids = np.sort(rng.choice(10 ** 6, n, replace=False))
        model = bhkm.fit(ids, features, branching=branching, depth=depth, seed=42, max_iter=3)
        return model, features

    def test_depth_four_fit_shape_equals_the_capacity_table(self):
        for n, b in ((200, 3), (700, 4), (1500, 16)):
            with self.subTest(n=n, b=b):
                model, features = self.fitted(n, b, 4)
                audit = ad.audit_mapping_d(model.ids, model.sids, features, branching=b, kind='bhkm', depth=4)
                table = ad.capacity_table(n, b, 4)
                self.assertEqual([p['distinct'] for p in audit['positions'][:4]], table['distinct'])
                self.assertEqual(audit['leaves']['count'], table['leaves'])
                self.assertEqual(audit['suffix_vocabulary'], table['suffix_vocabulary'])
                self.assertEqual(audit['vocab_size'], table['vocab_size'])
                self.assertEqual((audit['semantic_collision_groups'], audit['semantic_collision_items'], audit['leaves']['singletons']),
                                 (table['semantic_collision_groups'], table['semantic_collision_items'], table['singleton_leaves']))
                self.assertTrue(audit['balanced_children']['balanced'])
                self.assertEqual(audit['full_sid_collisions'], 0)

    def test_the_balanced_codes_helper_agrees_with_the_capacity_table(self):
        for n, b, d in ((200, 3, 4), (1500, 16, 4), (400, 4, 3)):
            ids, sids = mapping_from(balanced_codes(n, b, d))
            table = ad.capacity_table(n, b, d)
            audit = ad.audit_mapping_d(ids, sids, content_for(n), branching=b, kind='bhkm', depth=d)
            self.assertEqual(audit['leaves']['count'], table['leaves'])
            self.assertEqual(audit['suffix_vocabulary'], table['suffix_vocabulary'])
            self.assertTrue(audit['balanced_children']['balanced'])

    def test_capacity_table_by_hand(self):
        # 12 items, b=2, depth 2: 6/6, then 3/3 each; leaf sizes 3 -> suffix vocabulary 3
        table = ad.capacity_table(12, 2, 2)
        self.assertEqual((table['leaves'], table['distinct'], table['suffix_vocabulary'], table['vocab_size']), (4, [2, 2], 3, 2 + 4 + 3))
        # 10 items, b=4, depth 2: first level 3,3,2,2; second level splits each into singletons (3 -> 1,1,1; 2 -> 1,1)
        table = ad.capacity_table(10, 4, 2)
        self.assertEqual((table['leaves'], table['distinct'], table['suffix_vocabulary'], table['singleton_leaves']), (10, [4, 3], 1, 10))
        self.assertEqual((table['semantic_collision_groups'], table['semantic_collision_items'], table['vocab_size']), (0, 0, 2 + 2 * 4 + 1))
        self.assertEqual(table['leaf_sizes'], {1: 10})
        # the design-sheet case: N = 120,415, 16-ary, depth 3 (anchor) has 4,096 leaves of 29 or 30 items
        anchor = ad.capacity_table(120415, 16, 3)
        self.assertEqual((anchor['leaves'], anchor['leaf_sizes'], anchor['suffix_vocabulary'], anchor['vocab_size']), (4096, {29: 2465, 30: 1631}, 30, 80))


class BalanceAtEveryLevel(unittest.TestCase):
    def test_a_violation_in_the_last_semantic_level_is_found(self):
        codes = balanced_codes(12, 2, 4)
        _, sids = mapping_from(codes)
        self.assertTrue(ad.balanced_children_d(sids, 2, 4)['balanced'])
        bad = codes.copy()
        node = np.flatnonzero((bad[:, :3] == bad[0, :3]).all(axis=1))
        self.assertGreaterEqual(len(node), 2)
        bad[node, 3] = 0  # every item of one depth-3 node now shares the last digit: one child instead of min(b, size)
        _, bad_sids = mapping_from(bad)
        found = ad.balanced_children_d(bad_sids, 2, 4)
        self.assertFalse(found['balanced'])
        self.assertEqual(found['violations'], 1)
        for level in range(3):  # the same defect planted at each shallower level is found too
            planted = codes.copy()
            members = np.flatnonzero((planted[:, :level] == planted[0, :level]).all(axis=1))
            planted[members, level] = 0
            _, planted_sids = mapping_from(planted)
            self.assertFalse(ad.balanced_children_d(planted_sids, 2, 4)['balanced'], level)

    def test_a_node_with_a_missing_child_is_a_violation(self):
        codes = balanced_codes(30, 3, 2)
        codes[codes[:, 0] == 2, 1] = 0  # one node keeps a single child instead of min(b, size) = 3
        _, sids = mapping_from(codes)
        found = ad.balanced_children_d(sids, 3, 2)
        self.assertEqual((found['balanced'], found['violations'], found['nodes']), (False, 1, 1 + 3))

    def test_sibling_spread_two_needs_tolerance_two(self):
        codes = balanced_codes(18, 3, 2)  # 6/6/6, each split 2/2/2
        moved = np.flatnonzero(codes[:, 0] == 2)[0]
        codes[moved, 0] = 0  # root siblings become 7 / 6 / 5; every second-level node keeps spread <= 1
        _, sids = mapping_from(codes)
        strict = ad.balanced_children_d(sids, 3, 2)
        self.assertEqual((strict['balanced'], strict['violations'], strict['max_spread']), (False, 1, 2))
        self.assertTrue(ad.balanced_children_d(sids, 3, 2, tolerance=2)['balanced'])


class DepthFourGates(unittest.TestCase):
    def audit(self, kind='rqvae'):
        codes = balanced_codes(300, 4, 4)
        return audit_d(codes, 4, 4, kind)

    def rq(self, norms=(5.0, 4.0, 3.0, 2.0, 1.0)):
        return {'status': 'complete', 'steps': 9, 'max_steps': 9, 'elapsed': 1.0, 'wall_cap_seconds': 2.0,
                'reconstruction': {'mse': 0.5, 'sst': 1.0, 'r2': 0.5, 'residual_norms': list(norms)}}

    def check(self, result, check_id):
        return next(c for c in result['checks'] if c['id'] == check_id)

    def gates(self, audit, rq=None, kind='rqvae', **extra):
        return ad.gates_d(audit, kind, rq, depth=4, expected_content_sha256=audit['content_sha256'], required_steps=9, **extra)

    def test_rqvae_gates_pass_on_a_clean_depth_four_audit(self):
        audit = self.audit()
        audit['positions'] = [dict(p, distinct=p['base']) for p in audit['positions']]
        result = self.gates(audit, self.rq())
        self.assertTrue(result['passed'], result)
        self.assertEqual([c['id'] for c in result['checks']], ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'R1', 'R2', 'R3', 'R4'])

    def test_R1_covers_all_four_levels_with_integer_boundaries(self):
        audit = self.audit()
        for K, ok_used, bad_used in ((256, 205, 204), (16, 13, 12)):
            audit['branching'] = K
            for level in range(4):
                with self.subTest(K=K, level=level):
                    good = [K] * 4
                    good[level] = ok_used
                    audit['positions'] = [dict(p, distinct=v) for p, v in zip(audit['positions'], good + [1])]
                    self.assertTrue(self.check(self.gates(audit, self.rq()), 'R1')['ok'])
                    worse = [K] * 4
                    worse[level] = bad_used
                    audit['positions'] = [dict(p, distinct=v) for p, v in zip(audit['positions'], worse + [1])]
                    r1 = self.check(self.gates(audit, self.rq()), 'R1')
                    self.assertFalse(r1['ok'])
                    self.assertEqual(r1['value'], worse)

    def test_R4_needs_five_strictly_decreasing_norms(self):
        audit = self.audit()
        for norms in ((5, 4, 3, 2, 1), (9.5, 4, 3, 2, 0.1)):
            self.assertTrue(self.check(self.gates(audit, self.rq(norms)), 'R4')['ok'])
        for norms in ((5, 4, 3, 2, 2), (5, 4, 3, 3.5, 1), (5, 4, 3, 2), (1, 2, 3, 4, 5)):
            with self.subTest(norms=norms):
                self.assertFalse(self.check(self.gates(audit, self.rq(norms)), 'R4')['ok'])

    def test_bhkm_B1_and_B2(self):
        audit = self.audit('bhkm')
        result = self.gates(audit, None, 'bhkm')
        self.assertEqual([c['id'] for c in result['checks']][-1], 'B1')
        self.assertTrue(self.check(result, 'B1')['ok'])
        table = ad.capacity_table(300, 4, 4)
        prereg = {'items': 300, 'leaves.count': table['leaves'], 'suffix_vocabulary': table['suffix_vocabulary'], 'vocab_size': table['vocab_size'],
                  'positions.3.distinct': 4}
        result = self.gates(audit, None, 'bhkm', prereg=prereg)
        self.assertEqual([c['id'] for c in result['checks']][-2:], ['B1', 'B2'])
        self.assertTrue(self.check(result, 'B2')['ok'])
        wrong = dict(prereg, **{'leaves.count': table['leaves'] + 1})
        b2 = self.check(self.gates(audit, None, 'bhkm', prereg=wrong), 'B2')
        self.assertFalse(b2['ok'])
        self.assertEqual([m['key'] for m in b2['value']], ['leaves.count'])

    def test_B2_is_for_bhkm_only(self):
        self.assertRaises(ValueError, self.gates, self.audit(), self.rq(), 'rqvae', prereg={'items': 300})

    def test_hard_gates_fail_closed(self):
        audit = self.audit('bhkm')
        broken = copy.deepcopy(audit)
        broken['full_sid_collisions'] = 1
        self.assertFalse(self.check(self.gates(broken, None, 'bhkm'), 'H1')['ok'])
        broken = copy.deepcopy(audit)
        broken['content_sha256'] = '0' * 64
        self.assertFalse(self.check(ad.gates_d(broken, 'bhkm', None, depth=4), 'H5')['ok'])


class VocabularyAndAgreement(unittest.TestCase):
    def test_vocabulary_formula_uses_the_depth(self):
        for depth, b in ((3, 4), (4, 4), (4, 3)):
            audit = audit_d(balanced_codes(200, b, depth), depth, b)
            self.assertEqual(audit['vocab_size'], 2 + depth * b + audit['suffix_vocabulary'])

    def test_prefix_agreement(self):
        ids = np.arange(6, dtype=np.uint32)
        old = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.uint32)
        new = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 2, 0, 0, 0], [1, 0, 0, 3, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=np.uint32)
        got = ad.prefix_agreement(ids, old, ids, new, 3)
        self.assertEqual(got, {'items': 6, 'equal_prefix_items': 4, 'rate': 4 / 6, 'prefix_digits': 3})
        self.assertEqual(ad.prefix_agreement(ids, old, ids, new, 1)['equal_prefix_items'], 6)
        for bad in ((ids, old, ids[::-1].copy(), new, 3), (ids, old, ids, new, 0), (ids, old, ids, new, 4), (ids, old[:, :2], ids, new, 3)):
            self.assertRaises(ValueError, ad.prefix_agreement, *bad)


if __name__ == '__main__':
    unittest.main()
