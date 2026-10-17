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

    def test_bootstrap_stream_matches_manual_loop_over_the_whole_distribution(self):
        import random
        d = {'a': [(i * 7) % 3 - 1 for i in range(40)], 'b': [(i * 5) % 3 - 1 for i in range(40)]}
        rng = random.Random(20261014); sums = {'a': [], 'b': []}
        for _ in range(300):
            index = [rng.randrange(40) for _ in range(40)]
            for k in ('a', 'b'):
                sums[k].append(sum(d[k][i] for i in index))
        want = {k: [f.nearest_rank(sorted(v), Fraction(25, 1000)), f.nearest_rank(sorted(v), Fraction(975, 1000))] for k, v in sums.items()}
        self.assertEqual(f.bootstrap_intervals(d, 300, 20261014), want)

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


def hand_world():
    """5 queries; P = items 0..99 for every query; Pop = 300..399; G lists chosen so every report cell is hand-computable."""
    P = list(range(100)); POP = list(range(300, 400))
    targets = [0, 500, 77, 40, 20]
    G = [[0], [7, 500], [77], list(range(900, 1000)), [20] + list(range(1000, 1098))]
    meta = [('supported', 'positive_gt_p90', 10), ('unsupported', 'unsupported0', 3), ('supported', 'positive_gt_p90', 50),
            ('supported', 'positive_median_to_p90', 30), ('unsupported', 'positive_le_median', 20)]
    queries = [{'target': t, 'oov': False, 'support': m[0], 'popularity_group': m[1], 'user_group': 'gt20', 'history_len': m[2]} for t, m in zip(targets, meta)]
    return queries, [list(P) for _ in targets], G, POP


class HandComputedReportTests(unittest.TestCase):
    def setUp(self):
        queries, P, G, POP = hand_world()
        P24 = [list(range(100, 200)) for _ in queries]; P24[0] = [0] + list(range(101, 200))              # seed-2024 P differs
        G24 = [list(range(2000, 2100)) for _ in queries]                                                    # all full
        self.queries = queries
        self.routes = {('P', 42): P, ('G', 42): G, ('POP', None): POP, ('P', 2024): P24, ('G', 2024): G24}
        self.out = f.run(self.routes, queries, 2000, seeds=(42, 2024))

    def cell(self, row, slice_name='all'):
        return self.out['rows'][row]['slices'][slice_name]

    def test_hits_and_new_lost_against_c1_and_c4(self):
        self.assertEqual([self.cell('seed42:C1')['hits'], self.cell('seed42:C4')['hits'], self.cell('seed42:C5')['hits']], [3, 2, 4])
        self.assertEqual(self.cell('seed42:C5')['vs_C1'], {'new': 2, 'lost': 1, 'net': 1})      # new q1,q2 ; lost q3
        self.assertEqual(self.cell('seed42:C5')['vs_C4'], {'new': 2, 'lost': 0, 'net': 2})
        self.assertEqual(self.cell('seed42:C4')['vs_C1'], {'new': 0, 'lost': 1, 'net': -1})
        self.assertEqual(self.cell('seed42:C4')['vs_C4'], {'new': 0, 'lost': 0, 'net': 0})

    def test_route_attribution_and_g_independent_hits(self):
        self.assertEqual(self.cell('seed42:C5')['attribution'], {'P42∩G42': 3, 'G42': 1})
        self.assertEqual(self.cell('seed42:C5')['g_independent_hits'], 1)                          # only q1 (target 500 not in P)
        self.assertIsNone(self.cell('seed42:C4')['g_independent_hits'])
        self.assertEqual(self.cell('seed42:C4')['attribution'], {'P42∩Pop': 0} if False else self.cell('seed42:C4')['attribution'])
        self.assertEqual(sum(self.cell('seed42:C4')['attribution'].values()), 2)

    def test_overlap_denominators_and_summaries(self):
        o = self.out['rows']['seed42:C5']['overlap']['P42|G42']
        self.assertAlmostEqual(o['over_quota']['mean'], 0.008); self.assertAlmostEqual(o['over_quota']['median'], 0.01)
        self.assertAlmostEqual(o['over_second_route_size']['mean'], (2.5 + 1 / 99) / 5); self.assertEqual(o['over_second_route_size']['median'], 0.5)
        self.assertEqual((o['over_second_route_size']['p10'], o['over_second_route_size']['p90']), (0.0, 1.0))

    def test_pool_fill_coverage_and_g_short_slices(self):
        c5 = self.out['rows']['seed42:C5']
        self.assertEqual(c5['filled']['G42']['mean'], (1 + 2 + 1 + 100 + 99) / 5); self.assertEqual(c5['filled']['P42']['mean'], 100)
        self.assertEqual(c5['coverage']['pool_distinct_items'], 299); self.assertAlmostEqual(c5['coverage']['pool_coverage'], 299 / 2000)
        self.assertEqual(c5['g_short_queries'], 4)
        self.assertEqual((self.cell('seed42:C5', 'g_short:short')['n'], self.cell('seed42:C5', 'g_short:full')['n']), (4, 1))   # 99 is short, 100 is full
        self.assertEqual(self.cell('seed42:C5', 'g_short:full')['vs_C1'], {'new': 0, 'lost': 1, 'net': -1})                     # q3 only
        self.assertEqual(c5['pool_size']['mean'], (100 + 101 + 100 + 200 + 198) / 5)

    def test_final_distinct_items_and_final_coverage_hand_computed(self):
        # C1: P top50 for every query -> items 0..49 ; C2: union of G top-50 lists ; C4: P ranks 1..25 + Pop ranks 1..25 ; C5: see comments
        # C5 finals: q0 {0..49}; q1 {0..48,500}; q2 {0..48,77}; q3 {0..24,900..924}; q4 {0..25,1000..1023}  -> 50+1+1+25+24 = 101 distinct
        for row, distinct in (('C1', 50), ('C2', 104), ('C4', 50), ('C5', 101)):
            cov = self.out['rows'][f'seed42:{row}']['coverage']
            self.assertEqual(cov['final_distinct_items'], distinct, row); self.assertAlmostEqual(cov['final_coverage'], distinct / 2000, msg=row)

    def test_run_level_localization_is_hand_computed_and_uses_seed_42_g_lengths(self):
        # low-support set = {q0,q1,q2,q4}: q1,q4 by support/popularity, q0,q2,q4 by short G list (seed-42 lengths 1,2,1,100,99); q3 is full and well supported
        # net vs C1 inside the set: q1 +1, q2 +1 -> 2 ; total net = 4 - 3 = 1 ; C6 has the same hit set as C5 here
        loc = self.out['localization']
        self.assertEqual(list(loc), ['C5', 'C6'])
        want = {'low_support_queries': 4, 'net_in_set': 2, 'net_total': 1, 'localized': False}
        self.assertEqual(loc['C5'], want); self.assertEqual(loc['C6'], want)

    def test_not_run_text_names_the_reason(self):
        out = f.run({k: v for k, v in self.routes.items() if k != ('P', 2024)}, self.queries, 2000, seeds=(42, 2024))
        self.assertIn('P2024', out['paired']); self.assertIn('seed 42', out['paired'])
        out = f.run(self.routes, self.queries, 2000, seeds=(2024,))
        self.assertIn('seed 42 not requested', out['paired']); self.assertEqual(out['paired'], out['localization']); self.assertEqual(out['paired'], out['mechanical_conditions'])
        self.assertEqual(sorted(out['rows']), ['seed2024:C1', 'seed2024:C4', 'seed2024:C5'])

    def test_oov_hits_are_surfaced_as_an_anomaly_flag(self):
        queries = [dict(q) for q in self.queries]; queries[0].update(oov=True, support='oov', popularity_group='oov')
        out = f.run(self.routes, queries, 2000, seeds=(42,))
        self.assertEqual((out['rows']['seed42:C5']['oov_hits'], out['rows']['seed42:C5']['oov_nonzero_hits_anomaly']), (1, True))   # target 0 is a candidate: upstream bug

    def test_manifest_pins_script_hash_and_carries_report_notes(self):
        import hashlib
        from pathlib import Path
        b = self.out['bootstrap']
        self.assertEqual(b['fusion_v2_script_sha256'], hashlib.sha256(Path(f.__file__).read_bytes()).hexdigest())
        self.assertEqual((b['resamples'], b['seed'], b['comparable_with_numpy_default_rng_0_batches']), (10000, 20261014, False))
        notes = ' '.join(self.out['report_notes'])
        for needle in ('default_rng(0)', 'necessary condition', 'not sufficient', 'alpha = 0.025', 'p < 0.05', 'second decoding cross-check', 'D-1', '0.2%', '623 OOV', 'right-censoring', 'No online A/B'):
            self.assertIn(needle, notes)
        self.assertNotIn('at least 7 hits is required', notes)

    def test_summarize_even_length_median(self):
        self.assertEqual(f.summarize([4, 1, 3, 2])['median'], 2.5); self.assertEqual(f.summarize([1, 2, 3])['median'], 2)

    def test_slice_membership_and_history_buckets(self):
        self.assertEqual(self.cell('seed42:C5', 'support:unsupported')['n'], 2); self.assertEqual(self.cell('seed42:C5', 'history:50')['n'], 1)
        self.assertEqual(self.cell('seed42:C5', 'history:1-10')['n'], 2); self.assertEqual(self.cell('seed42:C5', 'oov')['hits'] if 'oov' in self.out['rows']['seed42:C5']['slices'] else 0, 0)
        self.assertEqual(self.out['rows']['seed42:C5']['oov_hits'], None); self.assertFalse(self.out['rows']['seed42:C5']['oov_nonzero_hits_anomaly'])

    def test_reproduction_rows_use_same_seed_routes_and_g_short_of_that_seed(self):
        self.assertEqual(sorted(k for k in self.out['rows'] if k.startswith('seed2024:')), ['seed2024:C1', 'seed2024:C4', 'seed2024:C5'])
        self.assertEqual(self.cell('seed2024:C1')['hits'], 1)                          # only q0's target 0 is in the seed-2024 P list
        self.assertNotIn('g_short:short', self.out['rows']['seed2024:C5']['slices'])    # G24 is all full: label comes from seed 2024's own G
        self.assertEqual(self.out['rows']['seed2024:C5']['routes'], ['P2024', 'G2024'])
        self.assertEqual(sorted(k for k in self.out['rows'] if k.startswith('seed42:')), ['seed42:C1', 'seed42:C2', 'seed42:C3', 'seed42:C4', 'seed42:C5', 'seed42:C6', 'seed42:C7'])

    def test_missing_inputs_are_reported_as_not_run_explicitly(self):
        self.assertIsInstance(self.out['paired'], dict); self.assertEqual(self.out['not_run'], {})
        out = f.run({k: v for k, v in self.routes.items() if k != ('G', 2024)}, self.queries, 2000, seeds=(42, 2024))
        self.assertIn('G2024', out['not_run']['seed2024']); self.assertNotIn('seed2024:C1', out['rows'])
        self.assertIsInstance(out['paired'], dict)                                      # the main table does not depend on seed 2024's G
        out = f.run({k: v for k, v in self.routes.items() if k != ('P', 2024)}, self.queries, 2000, seeds=(42,))
        self.assertIn('P2024', out['not_run']['seed42']); self.assertEqual(out['rows'], {})
        for key in ('paired', 'mechanical_conditions', 'localization'):
            self.assertTrue(out[key].startswith('NOT_RUN(main table (seed 42) unavailable'), key)


class PairedTableTests(unittest.TestCase):
    def vectors(self, c5, c4, c7, c1=None, c6=None, n=40):
        def v(ones):
            return [1 if i in ones else 0 for i in range(n)]
        return {'C1': v(set(c1 or ())), 'C4': v(set(c4)), 'C5': v(set(c5)), 'C6': v(set(c6 or c5)), 'C7': v(set(c7))}

    def test_h2_is_intersection_union_with_correct_references(self):
        ten = range(10)
        only_a = f.paired_table(self.vectors(ten, c4=(), c7=range(7)), 200)[1]            # vs C4: +10 (p~0.002) ; vs C7: +3
        self.assertEqual((only_a['H1_condition_met'], only_a['H2_condition_met']), (True, False))
        only_b = f.paired_table(self.vectors(ten, c4=range(7), c7=()), 200)[1]
        self.assertEqual(only_b['H2_condition_met'], False)
        both = f.paired_table(self.vectors(ten, c4=(), c7=()), 200)[1]
        self.assertTrue(both['H2_condition_met'])
        pairs, _ = f.paired_table(self.vectors(ten, c4=(), c7=range(7), c1=range(5)), 200)
        self.assertEqual((pairs['H2b']['row'], pairs['H2b']['reference']), ('C5', 'C7')); self.assertEqual((pairs['H1']['b'], pairs['H1']['c']), (5, 0))
        self.assertEqual(pairs['D1']['mechanical_condition'], 'descriptive_only')

    def test_run_passes_frozen_bootstrap_defaults(self):
        from unittest import mock
        queries, P, G, POP = hand_world(); calls = []
        real = f.bootstrap_intervals
        def spy(diffs, n_resamples=None, seed=None):
            calls.append((n_resamples, seed)); return real(diffs, 50, seed)
        with mock.patch.object(f, 'bootstrap_intervals', spy):
            f.run({('P', 42): P, ('G', 42): G, ('POP', None): POP, ('P', 2024): P}, queries, 2000, seeds=(42,))
        self.assertEqual(calls, [(10000, 20261014)])


class LocalizationIndependenceTests(unittest.TestCase):
    def test_each_condition_alone_puts_the_query_in_the_set(self):
        base = {'oov': False, 'support': 'supported', 'popularity_group': 'positive_gt_p90'}
        qs = [{**base, 'support': 'unsupported'}, {**base, 'popularity_group': 'unsupported0'}, {**base, 'popularity_group': 'positive_le_median'}, dict(base), dict(base)]
        short = ['full', 'full', 'full', 'short', 'full']
        got = f.localization({'C1': [0] * 5, 'C5': [1] * 5}, 'C5', qs, short)
        self.assertEqual((got['low_support_queries'], got['net_in_set'], got['net_total']), (4, 4, 5))   # the last query meets none of the conditions


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
