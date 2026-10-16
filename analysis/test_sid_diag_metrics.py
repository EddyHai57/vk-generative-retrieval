"""Synthetic hand oracles for frozen S0 math; no real data."""
import importlib
import math
import unittest


class S0MathTests(unittest.TestCase):
    def setUp(self):
        try:
            self.m = importlib.import_module('analysis.sid_diag_metrics')
        except ModuleNotFoundError:
            self.fail('S0 math module has not been implemented')
        self.codes = {'a': (0, 0, 0), 'b': (0, 0, 1),
                      'c': (0, 1, 0), 'd': (1, 0, 0)}

    def test_structure_hand_oracle(self):
        r = self.m.structure_metrics(self.codes, 2, 3)
        self.assertEqual(r['n_items'], 4)
        self.assertEqual([x['distinct_codes'] for x in r['layers']], [2, 2, 2])
        self.assertEqual([x['nodes'] for x in r['layers']], [2, 3, 4])
        self.assertEqual([x['avg_children'] for x in r['layers']], [2, 1.5, 4/3])
        self.assertEqual([x['avg_items_per_node'] for x in r['layers']], [2, 4/3, 1])
        self.assertAlmostEqual(r['layers'][0]['normalized_entropy'], -(0.75*math.log2(.75)+.25*math.log2(.25)))
        self.assertEqual(r['effective_layers'], 3)
        self.assertEqual(r['leaves']['count'], 4)
        self.assertEqual(r['leaves']['avg_items'], 1)

    def test_structure_collisions_and_suffix(self):
        r = self.m.structure_metrics({'a': (0,0,9), 'b': (0,0,9), 'c': (0,0,8)}, 2, 2)
        self.assertEqual(r['leaves']['count'], 1)
        self.assertEqual(r['leaves']['collision_items'], 2)
        self.assertEqual(r['full_code_leaves']['count'], 2)
        self.assertEqual(r['full_code_leaves']['max_items'], 2)
        self.assertEqual(r['full_code_leaves']['collision_items'], 1)
        self.assertEqual(r['full_code_leaves']['collision_pairs'], 1)
        self.assertEqual(r['full_code_leaves']['collided_leaves'], 1)
        self.assertEqual(r['effective_layers'], 0)

    def test_empty_structure(self):
        r = self.m.structure_metrics({}, 2, 3)
        self.assertIsNone(r['leaves']['avg_items'])
        self.assertEqual(r['leaves']['max_items'], 0)
        self.assertTrue(all(x['avg_children'] is None for x in r['layers']))
        self.assertEqual(r['effective_layers'], 0)

    def test_invalid_structure(self):
        for codes, b, d in [({'a': (0,)},2,2), ({'a': (2,)},2,1), ({'a':(0,),'b':(0,1)},2,1), ({},1,1), ({},2,0)]:
            with self.assertRaises(ValueError): self.m.structure_metrics(codes,b,d)

    def test_prefix_item_full_sid_and_oov(self):
        codes = dict(self.codes, e=(0,0,0))
        r = self.m.prediction_metrics(['a','c','x'], [['e','b','a'],['a','b'],['d']], codes)
        self.assertEqual(r['n_oov'], 1)
        self.assertEqual(r['n_in_catalog'], 2)
        self.assertEqual(r['prefix_by_k']['1']['top1_hits'], 2)
        self.assertEqual(r['prefix_by_k']['2']['top1_hits'], 1)
        self.assertEqual(r['prefix_by_k']['3']['top1_hits'], 1)
        self.assertEqual(r['exact_item']['top1_hits'], 0)
        self.assertEqual(r['exact_item']['top50_hits'], 1)
        self.assertEqual(r['full_sid']['top1_hits'], 1)
        self.assertEqual(r['full_sid']['top1_rate'], .5)

    def test_stable_dedup_before_cutoff(self):
        r = self.m.prediction_metrics(['a'], [['b']*60+['a']], self.codes)
        self.assertEqual(r['exact_item']['top50_hits'], 1)
        self.assertEqual(r['exact_item']['top1_hits'], 0)

    def test_archived_ties_order_not_resorted(self):
        r = self.m.prediction_metrics(['a'], [['b','a']], self.codes)
        self.assertEqual(r['exact_item']['top1_hits'], 0)
        self.assertEqual(r['exact_item']['top50_hits'], 1)

    def test_empty_predictions_and_padded_d2(self):
        codes={'a':(0,0,0,1),'b':(0,0,0,2)}
        r = self.m.prediction_metrics(['a','x'], [['b'],[]], codes)
        self.assertEqual(r['exact_item']['top50_rate'], 0)
        self.assertEqual(r['prefix_by_k']['3']['top50_rate'], 1)
        self.assertEqual(r['full_sid']['top50_rate'], 0)
        r = self.m.prediction_metrics(['x'], [[]], self.codes)
        self.assertIsNone(r['exact_item']['top1_rate'])
        r = self.m.prediction_metrics([], [], self.codes)
        self.assertEqual(r['n_total'],0)

    def test_prediction_validation(self):
        for targets,preds in [(['a'],[]), (['a'],[['unknown']])]:
            with self.assertRaises(ValueError): self.m.prediction_metrics(targets,preds,self.codes)
        with self.assertRaises(ValueError): self.m.prediction_metrics([],[],{'a':(0,0)})

    def test_popularity_g4_boundaries_oov_first(self):
        targets=['o','a','b','c','d','e','f']
        codes={x:(i,0,0) for i,x in enumerate(targets[1:])}
        r=self.m.popularity_metrics(targets, [[x] if x!='o' else [] for x in targets], {'o':999,'a':0,'b':1,'c':2,'d':3,'e':9,'f':10},2,9,codes)
        self.assertEqual({k:v['n'] for k,v in r['bins'].items()}, {'oov':1,'unsupported':1,'low':2,'mid':2,'high':1})
        self.assertEqual(r['bins']['oov']['hits'], 0)
        self.assertEqual(r['supplemental_longtail_unsupported']['hits'],3)
        self.assertEqual(r['bins']['unsupported']['hits'],1)

    def test_popularity_missing_is_unsupported_empty_rate(self):
        r=self.m.popularity_metrics(['a'],[[]],{},0,0,self.codes)
        self.assertEqual(r['bins']['unsupported']['n'],1)
        self.assertIsNone(r['bins']['high']['rate'])
        with self.assertRaises(ValueError): self.m.popularity_metrics([],[],{},4,3,self.codes)
        with self.assertRaises(ValueError): self.m.popularity_metrics(['a'],[[]],{'a':-1},0,1,self.codes)

    def test_cooccurrence_manual_seed0_ordered_including_self(self):
        # random.Random(0).randrange(2) produces 1,1,0,1 for two pairs.
        codes={'a':(0,0,0),'b':(1,0,0)}
        r=self.m.cooccurrence_metrics(codes,[['a','a','b','b','a']])
        self.assertEqual(r['n_adjacent_events'],4)
        self.assertEqual(r['n_same_item_excluded'],2)
        self.assertEqual(r['n_pairs'],2)
        for x in r['prefix_by_k'].values():
            self.assertEqual(x['cooccurrence_matches'],0)
            self.assertEqual(x['random_matches'],1)
            self.assertEqual(x['random_rate'],.5)
            self.assertEqual(x['lift'],0)
        self.assertEqual(r,self.m.cooccurrence_metrics(codes,[['a','a','b','b','a']]))

    def test_cooccurrence_zero_random_and_zero_events(self):
        # seed0 random draws d,d; choose k3 mismatch observed but random is self.
        codes={i:(i,0,0) for i in range(8)}
        # seed 1 first random pair (2,1) differs -> unknown lift.
        r=self.m.cooccurrence_metrics(codes,[[0,1]],seed=1)
        self.assertEqual(r['prefix_by_k']['1']['random_matches'],0)
        self.assertIsNone(r['prefix_by_k']['1']['lift'])
        r=self.m.cooccurrence_metrics(self.codes,[['a'],[],['b','b']])
        self.assertEqual(r['n_pairs'],0)
        self.assertIsNone(r['prefix_by_k']['1']['lift'])
        self.assertIsNone(r['prefix_by_k']['1']['cooccurrence_rate'])
        self.assertEqual(self.m.cooccurrence_metrics({},[])['n_pairs'],0)

    def test_cooccurrence_unknown_fails_without_bridging(self):
        with self.assertRaises(ValueError): self.m.cooccurrence_metrics(self.codes,[['a','x','b']])

    def test_decomposition_specified_seeds_only_boundary(self):
        c={42:{'high':1000,'mid':0,'low':0,'unsupported':0},2024:{'high':3,'mid':1,'low':0,'unsupported':0},2025:{'high':0,'mid':2,'low':0,'unsupported':0}}
        z={s:{k:0 for k in v} for s,v in c.items()}
        r=self.m.decompose_gain(c,z,[2024,2025])
        self.assertEqual(r['total_gain'],6)
        self.assertEqual(r['head_share'],.5)
        self.assertEqual(r['rule_branch'],'head_share_ge_0_5')
        self.assertEqual(r['seeds'],[2024,2025])
        c[2025]['mid']=3
        self.assertEqual(self.m.decompose_gain(c,z,[2024,2025])['rule_branch'],'head_share_lt_0_5')

    def test_decomposition_nonpositive_missing_duplicate(self):
        a={1:{'high':0,'mid':0,'low':0,'unsupported':0}}
        z={1:{'high':0,'mid':1,'low':0,'unsupported':0}}
        r=self.m.decompose_gain(a,z,[1])
        self.assertIsNone(r['head_share'])
        self.assertEqual(r['rule_branch'],'no_positive_gain')
        for seeds in ([],[1,1],[2]):
            with self.assertRaises(ValueError): self.m.decompose_gain(a,z,seeds)

if __name__=='__main__': unittest.main()
