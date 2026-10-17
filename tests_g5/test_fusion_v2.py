"""Hand-computed tests for the G5 fusion core (synthetic only)."""
import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from vkgr_g5 import fusion_v2 as f  # noqa: E402


def lst(start, n=100):
    return list(range(start, start + n))


class RrfTests(unittest.TestCase):
    def test_quota(self):
        self.assertEqual([f.quota(n) for n in (1, 2, 3, 4)], [50, 100, 66, 50])

    def test_hand_computed_two_route_order_and_exact_scores(self):
        p = lst(1); g = [3, 1, 200]                      # G short list: no padding
        got = f.fuse_query([p, g], ['P', 'G'])
        # item1: 1/61+1/62 ; item3: 1/63+1/61 ; item2: 1/62 ; item200: 1/63 ; item4: 1/64
        self.assertEqual(got['final'][:5], [1, 3, 2, 200, 4])
        self.assertEqual(got['filled'], [100, 3]); self.assertEqual(got['pool_size'], 101)
        self.assertEqual(got['quota'], 100)
        self.assertEqual(Fraction(1, 61) + Fraction(1, 62) > Fraction(1, 63) + Fraction(1, 61), True)

    def test_tie_goes_to_smaller_item_id(self):
        # 5 and 9 each receive 1/61 + 1/62 exactly; the smaller id comes first
        got = f.fuse_query([[5, 9] + lst(1000, 98), [9, 5] + lst(2000, 98)], ['P', 'P'])
        self.assertEqual(got['final'][:2], [5, 9])

    def test_three_routes_use_quota_66_and_never_exceed_198(self):
        got = f.fuse_query([lst(1), lst(1000), lst(5000)], ['P', 'G', 'POP'])
        self.assertEqual(got['quota'], 66); self.assertEqual(got['filled'], [66, 66, 66])
        self.assertEqual(got['pool_size'], 198); self.assertEqual(len(got['final']), 50)
        self.assertTrue(all(len(e) == 66 for e in got['entered']))

    def test_single_route_is_its_own_top50_and_g_single_can_be_short(self):
        self.assertEqual(f.fuse_query([lst(1)], ['P'])['final'], lst(1, 50))
        self.assertEqual(f.fuse_query([[7, 8, 9]], ['G'])['final'], [7, 8, 9])

    def test_short_g_is_not_padded_and_slots_not_given_to_p(self):
        got = f.fuse_query([lst(1), [500]], ['P', 'G'])
        self.assertEqual(got['filled'], [100, 1]); self.assertEqual(got['pool_size'], 101)       # P still exactly its quota

    def test_p_or_pop_short_of_quota_fails_the_input_gate(self):
        for names in (['P', 'G'], ['POP', 'G']):
            with self.assertRaisesRegex(ValueError, 'input gate'):
                f.fuse_query([lst(1, 99), lst(500)], names)
        with self.assertRaisesRegex(ValueError, 'input gate'):
            f.fuse_query([lst(1, 65), lst(500), lst(900)], ['P', 'G', 'POP'])

    def test_duplicates_inside_one_route_are_fatal_even_beyond_the_quota(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            f.fuse_query([lst(1, 100) + [1], lst(500)], ['P', 'G'])

    def test_cross_route_duplicates_are_merged_not_fatal(self):
        got = f.fuse_query([lst(1), lst(1)], ['P', 'P'])
        self.assertEqual(got['pool_size'], 100); self.assertEqual(got['final'], lst(1, 50))


class ExactnessTests(unittest.TestCase):
    def test_scores_are_exact_fractions_with_k60_and_rank_from_one(self):
        scores = f.rrf_scores([[10, 20], [20, 30]])
        self.assertEqual(scores, {10: Fraction(1, 61), 20: Fraction(1, 62) + Fraction(1, 61), 30: Fraction(1, 62)})
        self.assertTrue(all(isinstance(v, Fraction) for v in scores.values()))

    def test_float_rounding_cannot_break_an_exact_tie(self):
        # ranks (1,2,7) summed in route order vs the same ranks permuted: exactly equal as Fractions, different as floats
        def routes(x, y):
            a = [x] + [y if r == 7 else 10000 + r for r in range(2, 67)]; a.insert(0, a.pop(0))
            a = [x if r == 1 else y if r == 7 else 10000 + r for r in range(1, 67)]
            b = [y if r == 1 else x if r == 2 else 20000 + r for r in range(1, 67)]
            c = [y if r == 2 else x if r == 7 else 30000 + r for r in range(1, 67)]
            return [a, b, c]
        for x, y in ((5, 9), (9, 5)):
            got = f.fuse_query(routes(x, y), ['P', 'G', 'POP'])
            self.assertEqual(got['final'][:2], [5, 9])

    def test_condition_needs_both_net_hits_and_significance(self):
        self.assertTrue(f.condition_met(14, 2)); self.assertFalse(f.condition_met(20, 10))       # +10 but p ~ 0.099
        self.assertFalse(f.condition_met(5, 0)); self.assertFalse(f.condition_met(8, 3))          # +5 below the integer bar

    def test_net_gain_of_exactly_six_can_never_be_significant(self):
        # smallest possible p for net +6 is b=6,c=0 -> 2/64 = 0.03125 >= 0.025, so the >=6 bar is implied by p < 0.025 (it is not the binding one)
        self.assertGreater(f.mcnemar_exact(6, 0), 0.025); self.assertLess(f.mcnemar_exact(7, 0), 0.025)
        self.assertFalse(f.condition_met(6, 0)); self.assertTrue(f.condition_met(7, 0))

    def test_nearest_rank_ceil_not_floor(self):
        self.assertEqual(f.nearest_rank(list(range(1, 8)), Fraction(975, 1000)), 7)


class StatsTests(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertEqual(f.mcnemar_exact(5, 0), 0.0625); self.assertEqual(f.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(f.mcnemar_exact(10, 2), 2 * 79 / 4096); self.assertEqual(f.mcnemar_exact(3, 3), 1.0)
        self.assertEqual(f.mcnemar_exact(2, 10), f.mcnemar_exact(10, 2))

    def test_nearest_rank_quantiles(self):
        values = list(range(1, 10001))
        self.assertEqual((f.nearest_rank(values, Fraction(25, 1000)), f.nearest_rank(values, Fraction(975, 1000))), (250, 9750))
        self.assertEqual(f.nearest_rank([4, 5, 6], Fraction(25, 1000)), 4)

    def test_bootstrap_is_deterministic_shared_and_pinned(self):
        d = {'a': [1, 0, -1, 1, 0, 1, 1, 0], 'b': [1, 0, -1, 1, 0, 1, 1, 0], 'c': [1] * 8}
        one = f.bootstrap_intervals(d, 500, 20261014); two = f.bootstrap_intervals(d, 500, 20261014)
        self.assertEqual(one, two); self.assertEqual(one['a'], one['b']); self.assertEqual(one['c'], [8, 8])
        import random
        rng = random.Random(20261014); idx = [rng.randrange(8) for _ in range(8)]
        self.assertEqual(sum(d['a'][i] for i in idx) in range(-8, 9), True)

    def test_bootstrap_first_resample_matches_manual_stream(self):
        d = {'a': [1, 0, 0, 1, 0, 1]}
        import random
        rng = random.Random(20261014); manual = []
        for _ in range(7):
            manual.append(sum(d['a'][rng.randrange(6)] for _ in range(6)))
        lo = f.nearest_rank(sorted(manual), Fraction(25, 1000)); hi = f.nearest_rank(sorted(manual), Fraction(975, 1000))
        self.assertEqual(f.bootstrap_intervals(d, 7, 20261014)['a'], [lo, hi])

    def test_localization_rule(self):
        q = [{'oov': False, 'support': 'unsupported', 'popularity_group': 'unsupported0'} for _ in range(12)] + \
            [{'oov': False, 'support': 'supported', 'popularity_group': 'positive_gt_p90'} for _ in range(12)]
        short = ['full'] * 24
        def vec(in_set, out_set):
            c1 = [0] * 24; c5 = [1] * in_set + [0] * (12 - in_set) + [1] * out_set + [0] * (12 - out_set)
            return {'C1': c1, 'C5': c5}
        self.assertTrue(f.localization(vec(6, 6), 'C5', q, short)['localized'])          # 6 of 12 = exactly half
        self.assertFalse(f.localization(vec(6, 7), 'C5', q, short)['localized'])         # 6 of 13 < half
        self.assertFalse(f.localization(vec(5, 0), 'C5', q, short)['localized'])         # below 6 hits
        self.assertFalse(f.localization(vec(0, 0), 'C5', q, short)['localized'])         # total <= 0
        neg = {'C1': [0] * 12 + [1] * 8 + [0] * 4, 'C5': [1] * 6 + [0] * 18}                 # +6 in set but -8 outside: total -2
        self.assertFalse(f.localization(neg, 'C5', q, short)['localized'])
        oov = [{**x, 'oov': True} for x in q]
        self.assertEqual(f.localization(vec(6, 6), 'C5', oov, short)['low_support_queries'], 0)   # OOV never in the set
        g = [{'oov': False, 'support': 'supported', 'popularity_group': 'positive_gt_p90'}] * 12
        self.assertEqual(f.localization({'C1': [0] * 12, 'C5': [1] * 7 + [0] * 5}, 'C5', g, ['short'] * 7 + ['full'] * 5)['net_in_set'], 7)

    def test_history_buckets(self):
        self.assertEqual([f.history_bucket(n) for n in (1, 10, 11, 25, 26, 49, 50)], ['1-10', '1-10', '11-25', '11-25', '26-49', '26-49', '50'])
        with self.assertRaises(ValueError):
            f.history_bucket(51)


def make_world(n=40, catalog=400):
    queries, p, g = [], [], []
    for i in range(n):
        target = (i * 7) % catalog
        queries.append({'target': target if i % 10 else catalog + 5, 'oov': not bool(i % 10), 'support': 'oov' if not i % 10 else ('supported' if i % 3 else 'unsupported'),
                        'popularity_group': 'oov' if not i % 10 else f.ITEM_GROUPS[i % 4], 'user_group': 'gt20', 'history_len': 1 + i % 50})
        p.append([(target + j * 3) % catalog for j in range(100)] if i % 2 else [(target + 1 + j) % catalog for j in range(100)])
        g.append([(target + 7 * j) % catalog for j in range(100 if i % 4 else 40)])
    pop = list(range(300, 400))
    return queries, p, g, pop


class RunTests(unittest.TestCase):
    def test_run_matches_direct_recomputation_and_pairs(self):
        queries, p, g, pop = make_world()
        routes = {('P', 42): p, ('P', 2024): p[::-1], ('G', 42): g, ('POP', None): pop}
        out = f.run(routes, queries, 400, seeds=(42,))
        c1 = sum(1 for q, r in zip(queries, p) if q['target'] in r[:50])
        self.assertEqual(out['rows']['seed42:C1']['slices']['all']['hits'], c1)
        self.assertEqual(out['rows']['seed42:C2']['slices']['oov']['hits'], 0)
        pair = out['paired']['H1']; self.assertEqual(pair['delta_hits'], pair['b'] - pair['c'])
        c5 = out['rows']['seed42:C5']['slices']['all']; self.assertEqual(pair['delta_hits'], c5['hits'] - c1)
        self.assertEqual(out['rows']['seed42:C5']['slices']['all']['vs_C1']['net'], pair['delta_hits'])
        self.assertLessEqual(max(s['n'] for k, s in out['rows']['seed42:C6']['slices'].items() if k == 'all'), 40)
        for cell in out['rows']['seed42:C5']['slices'].values():
            self.assertEqual(sum(cell['attribution'].values()), cell['hits'])
        self.assertEqual(out['mechanical_conditions']['alpha_per_family'], 0.025)
        self.assertEqual(out['paired']['D1']['mechanical_condition'], 'descriptive_only')

    def test_missing_seed_inputs_are_not_run_not_substituted(self):
        queries, p, g, pop = make_world()
        out = f.run({('P', 42): p, ('G', 42): g, ('POP', None): pop, ('P', 2024): p}, queries, 400, seeds=(42, 2024))
        self.assertEqual(out['rows'].get('seed2024:C5'), None)

    def test_query_metadata_validation(self):
        queries, p, g, pop = make_world()
        bad = [dict(q) for q in queries]; bad[1]['support'] = 'oov'
        with self.assertRaisesRegex(ValueError, 'support label'):
            f.run({('P', 42): p, ('G', 42): g, ('POP', None): pop, ('P', 2024): p}, bad, 400, seeds=(42,))
        with self.assertRaisesRegex(ValueError, 'query count'):
            f.run({('P', 42): p[:-1], ('G', 42): g, ('POP', None): pop, ('P', 2024): p}, queries, 400, seeds=(42,))

    def test_ndcg_uses_log2_rank_plus_one(self):
        res = [{'final': [9, 7, 5]}]
        m = f.row_metrics(res, [{'target': 5}])
        self.assertAlmostEqual(m['N@50'], 1 / math.log2(4)); self.assertEqual(m['hits'], 1)


if __name__ == '__main__':
    unittest.main()
