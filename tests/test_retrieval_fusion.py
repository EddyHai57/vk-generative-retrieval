import unittest
from fractions import Fraction
from itertools import product
from vkgr.retrieval_fusion import weight_candidates, weighted_rrf, redistribute_quota, select_weights

class FusionTests(unittest.TestCase):
    def test_grid_has_one_representative_per_ratio(self):
        candidates = weight_candidates()
        self.assertEqual(len(candidates), 65)
        self.assertEqual(candidates, sorted(candidates))
        ratios = set()
        for weights in candidates:
            first = next(w for w in weights if w)
            ratios.add(tuple(Fraction(w, first) for w in weights))
        self.assertEqual(len(ratios), 65)
        for weights in product(range(3), repeat=4):
            if any(weights):
                first = next(w for w in weights if w)
                self.assertIn(tuple(Fraction(w, first) for w in weights), ratios)
        self.assertNotIn((2, 2, 2, 2), candidates)
        self.assertIn((1, 1, 1, 1), candidates)

    def test_rrf_hand_oracle_zero_route_tie_and_limit(self):
        # item 9: 1/61 + 2/62, item 3: 1/62 + 2/61
        self.assertEqual(weighted_rrf([[9,3],[3,9],[1],[2]], (1,2,0,0)), [3,9])
        self.assertEqual(weighted_rrf([[9,3],[3,9],[],[]], (1,1,0,0)), [3,9])
        self.assertEqual(weighted_rrf([[9,3],[],[],[]], (1,0,0,0), final_k=1), [9])
        self.assertEqual(weighted_rrf([[],[],[1],[]], (1,0,0,0)), [])

    def test_invalid_or_duplicate_ranks_rejected(self):
        for ranks, weights in (([[1,1],[],[],[]],(1,0,0,0)), ([[True],[],[],[]],(1,0,0,0)), ([[1],[],[],[]],(0,0,0,0)), ([[1],[],[],[]],(-1,1,0,0)), ([[1],[],[],[]],(1,1,0))):
            with self.assertRaises(ValueError): weighted_rrf(ranks, weights)
        with self.assertRaises(ValueError): weighted_rrf([[],[],[],[]],(1,0,0,0), final_k=0)

    def test_quota_original_order_remainder_and_invalid_active(self):
        self.assertEqual(redistribute_quota([0,2,3]), {0:134,2:133,3:133})
        self.assertEqual(redistribute_quota([3,0,2]), {0:134,2:133,3:133})
        for active in ([],[1,1],[4]):
            with self.assertRaises(ValueError): redistribute_quota(active)

    def test_selection_metric_then_sparsity_then_lex(self):
        rows=[{'weights':(1,1,0,0),'recall@50':.2,'ndcg@50':.1}, {'weights':(0,1,0,0),'recall@50':.2,'ndcg@50':.1}, {'weights':(1,0,0,0),'recall@50':.2,'ndcg@50':.1}]
        self.assertEqual(select_weights(rows),(0,1,0,0))
        rows.append({'weights':(1,1,1,1),'recall@50':.3,'ndcg@50':.01})
        self.assertEqual(select_weights(rows),(1,1,1,1))
        with self.assertRaises(ValueError):select_weights([{'weights':(1,0,0,0),'recall@50':float('nan'),'ndcg@50':.1}])

    def test_mathematical_tie_across_different_rank_sum_orders(self):
        routes=[]
        for first,second in ((1,2),(7,1),(2,7)):
            row=list(range(1000,1007));row[first-1]=1;row[second-1]=2;routes.append(row)
        result=weighted_rrf([*routes,[]],(1,1,1,0))
        self.assertLess(result.index(1),result.index(2))
