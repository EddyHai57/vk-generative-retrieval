"""GTOK-02: seed-selection rule and collision-curve helpers (pure Python / numpy; no torch, no data files).

Everything here is a literal oracle: hits are integers out of 2520 validation queries, audits are hand-built dicts or tiny synthetic
mappings counted by brute force.
"""
import importlib.util
import itertools
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sel = load('gtok02_select')
curve = load('gtok02_curve')
N_QUERIES = 2520


def entry(hits, ndcg=0.03, verified=True):
    return {'recall50': hits / N_QUERIES, 'ndcg50': ndcg, 'verified': verified}


def design_example():
    """gtok02-design 3.7: B32 228, B64 241, B128 247, B512 262, B1024 250, B256D2 240, T3 254, B16 190."""
    hits = {'B32': 228, 'B64': 241, 'B128': 247, 'B512': 262, 'B1024': 250, 'B256D2': 240, 'B256D3': 254, 'B16': 190}
    return {label: entry(value) for label, value in hits.items()}


class SelectTests(unittest.TestCase):
    def test_design_example(self):
        out = sel.select(design_example())
        self.assertEqual(out['top2'], ['B512', 'B256D3'])
        self.assertEqual(out['near'], ['B256D3', 'B512'])  # canonical enumeration order
        self.assertEqual(out['selected'], ['B256D3', 'B512'])
        self.assertEqual(out['supplement'], ['B512', 'B256D2'])
        self.assertTrue(out['rule_e'])
        self.assertEqual((out['k'], out['trainings']), (2, 4))
        self.assertEqual(out['seeds'], [2024, 2025])
        self.assertEqual(out['ranking'][0], {'label': 'B512', 'hits': 262, 'ndcg50': 0.03})
        self.assertEqual(out['max_hits'], 262)

    def test_order_is_the_enumeration_of_the_eight_points(self):
        self.assertEqual(sel.ORDER, ('B16', 'B32', 'B64', 'B128', 'B256D3', 'B512', 'B1024', 'B256D2'))
        self.assertEqual(sel.NEW, ('B32', 'B64', 'B128', 'B512', 'B1024', 'B256D2'))

    def test_near_boundary_is_ten_hits_inclusive(self):
        results = design_example()
        results['B1024'] = entry(252)  # exactly 10 below 262
        self.assertIn('B1024', sel.select(results)['near'])
        results['B1024'] = entry(251)  # 11 below
        self.assertNotIn('B1024', sel.select(results)['near'])

    def test_near_new_configs_get_seeds_and_old_ones_do_not(self):
        results = design_example()
        results['B1024'] = entry(260)
        results['B16'] = entry(262)
        out = sel.select(results)
        self.assertIn('B1024', out['supplement'])
        self.assertNotIn('B16', out['supplement'])
        self.assertNotIn('B256D3', out['supplement'])

    def test_ties_use_ndcg_then_enumeration_order(self):
        results = {label: entry(200) for label in sel.ORDER}
        results['B64'] = entry(300, ndcg=0.02)
        results['B128'] = entry(300, ndcg=0.05)
        results['B512'] = entry(300, ndcg=0.05)
        out = sel.select(results)
        self.assertEqual([r['label'] for r in out['ranking'][:3]], ['B128', 'B512', 'B64'])
        self.assertEqual(out['top2'], ['B128', 'B512'])

    def test_unverified_runs_are_excluded_with_a_reason(self):
        results = design_example()
        results['B512'] = entry(999, verified=False)
        del results['B1024']
        out = sel.select(results)
        self.assertNotIn('B512', [r['label'] for r in out['ranking']])
        self.assertEqual(out['excluded'], {'B512': 'not verified', 'B1024': 'missing'})
        self.assertEqual(out['top2'], ['B256D3', 'B128'])

    def test_hits_must_be_integral(self):
        results = design_example()
        results['B32'] = {'recall50': 228.4 / N_QUERIES, 'ndcg50': 0.03, 'verified': True}
        with self.assertRaises(ValueError):
            sel.select(results)
        self.assertEqual(sel.hits_of(228 / N_QUERIES), 228)
        self.assertEqual(sel.hits_of(0.0), 0)

    def test_rule_e_does_not_fire_when_b256d2_is_held_or_unverified(self):
        held = design_example()
        del held['B256D2']
        out = sel.select(held)
        self.assertFalse(out['rule_e'])
        self.assertEqual(out['supplement'], ['B512'])
        unverified = design_example()
        unverified['B256D2'] = entry(240, verified=False)
        self.assertFalse(sel.select(unverified)['rule_e'])

    def test_k_can_be_zero(self):
        results = {'B16': entry(300), 'B256D3': entry(299), 'B32': entry(200), 'B64': entry(210), 'B128': entry(220), 'B512': entry(230),
                   'B1024': entry(240)}
        out = sel.select(results)
        self.assertEqual((out['k'], out['trainings'], out['supplement']), (0, 0, []))
        self.assertFalse(out['rule_e'])

    def test_rule_e_adds_b256d2_only_once(self):
        results = design_example()
        results['B256D2'] = entry(262)
        out = sel.select(results)
        self.assertEqual(out['supplement'].count('B256D2'), 1)
        self.assertEqual(out['k'], len(out['supplement']))

    def test_fewer_than_two_verified_points_is_reported_not_hidden(self):
        out = sel.select({'B32': entry(228)})
        self.assertEqual(out['top2'], ['B32'])
        self.assertIn('fewer than two', out['note'])

    def test_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            sel.select({'B999': entry(10)})

    def test_proposal_is_not_an_enqueue(self):
        out = sel.select(design_example())
        self.assertEqual(out['status'], 'proposal')
        json.dumps(out)  # serialisable


def audit(items, count, maximum, needing, collision_items, distinct, branching=256):
    return {'items': items, 'branching': branching, 'items_needing_suffix': needing, 'semantic_collision_items': collision_items,
            'leaves': {'count': count, 'max': maximum}, 'positions': [{'distinct': d} for d in distinct]}


KNOWN = {
    'T0': audit(120415, 4096, 30, 116319, 120415, [16, 16, 16, 30], 16),
    'T3': audit(120415, 120415, 1, 0, 0, [256, 256, 2, 1]),
    'B32': audit(120415, 32768, 4, 87647, 120415, [32, 32, 32, 4], 32),
    'B256D2': audit(120415, 65536, 2, 54879, 2 * 54879, [256, 256, 1, 2]),
}


class CollisionRateTests(unittest.TestCase):
    def rates(self, value):
        try:
            return curve.collision_rates(value)
        except ValueError as exc:  # an internal consistency error on a consistent audit is a failed expectation, not a crash
            self.fail(f'collision_rates rejected a consistent audit: {exc}')

    def test_known_points(self):
        t0 = self.rates(KNOWN['T0'])
        self.assertEqual((t0['CER'], t0['LCR']), (116319 / 120415, 1.0))
        self.assertEqual(self.rates(KNOWN['T3'])['CER'], 0.0)
        self.assertEqual(self.rates(KNOWN['T3'])['LCR'], 0.0)
        b32 = self.rates(KNOWN['B32'])
        self.assertEqual((b32['CER'], b32['LCR']), (87647 / 120415, 1.0))
        d2 = self.rates(KNOWN['B256D2'])
        self.assertAlmostEqual(d2['CER'], 0.4557, places=4)
        self.assertAlmostEqual(d2['LCR'], 0.9115, places=4)
        self.assertEqual((d2['leaves_max'], d2['positions_distinct']), (2, [256, 256, 1, 2]))
        self.assertAlmostEqual(d2['mean_leaf_size'], 120415 / 65536)

    def test_cer_has_two_equal_definitions_and_a_mismatch_is_an_error(self):
        for name, value in KNOWN.items():
            self.assertAlmostEqual(curve.collision_rates(value)['CER'], 1 - value['leaves']['count'] / value['items'], places=12, msg=name)
        broken = json.loads(json.dumps(KNOWN['T0']))
        broken['leaves']['count'] += 1
        with self.assertRaises(ValueError):
            curve.collision_rates(broken)

    def test_cer_is_not_computed_from_the_largest_leaf(self):
        a = audit(100, 50, 2, 50, 100, [4, 4, 4, 2])
        b = audit(100, 50, 9, 50, 100, [4, 4, 4, 9])
        self.assertEqual(self.rates(a)['CER'], self.rates(b)['CER'])

    def test_against_brute_force_counts_on_a_real_audit(self):
        from vkgr import sid_tok_audit as au
        rng = np.random.RandomState(4)
        n = 60
        ids = (np.arange(n) * 2 + 1).astype(np.uint32)
        codes = rng.randint(0, 3, size=(n, 3))
        sids = np.zeros((n, 4), dtype=np.uint32)
        sids[:, :3] = codes
        seen = Counter()
        for row in range(n):
            key = tuple(codes[row])
            sids[row, 3] = seen[key]
            seen[key] += 1
        content = rng.randn(n, 64).astype(np.float32)
        mapping = au.audit_mapping(ids, sids, content, branching=3, kind='bhkm')
        sizes = Counter(tuple(codes[row]) for row in range(n))
        needing = sum(size - 1 for size in sizes.values())
        in_collision = sum(size for size in sizes.values() if size > 1)
        rates = curve.collision_rates(mapping)
        self.assertEqual(rates['CER'], needing / n)
        self.assertEqual(rates['LCR'], in_collision / n)
        self.assertEqual(rates['leaves_max'], max(sizes.values()))


class StateTests(unittest.TestCase):
    def test_state_needs_both_seeds_beyond_the_threshold(self):
        cases = {(6, 6): 'UP', (5, 6): 'FLAT', (6, 5): 'FLAT', (10, 30): 'UP', (-6, -6): 'DOWN', (-6, -5): 'FLAT', (6, -6): 'FLAT', (0, 0): 'FLAT',
                 (-20, 20): 'FLAT'}
        for deltas, expected in cases.items():
            with self.subTest(deltas):
                self.assertEqual(curve.state(*deltas), expected)

    def test_tentative_state_uses_the_same_dead_zone(self):
        self.assertEqual([curve.tentative_state(d) for d in (6, 5, 0, -5, -6)], ['T_UP', 'T_FLAT', 'T_FLAT', 'T_FLAT', 'T_DOWN'])

    def test_threshold_is_six_hits(self):
        self.assertEqual(curve.THRESHOLD_HITS, 6)
        self.assertEqual(curve.T3_HITS, {42: 254, 2024: 274, 2025: 275})

    def test_classify_against_t3(self):
        self.assertEqual(curve.classify({42: 262, 2024: 281, 2025: 282}), 'UP')
        self.assertEqual(curve.classify({42: 240, 2024: 265, 2025: 268}), 'DOWN')
        self.assertEqual(curve.classify({42: 262, 2024: 281, 2025: 270}), 'FLAT')
        self.assertEqual(curve.classify({42: 262, 2024: 281}), 'T_UP')  # a seed is missing: seed 42 decides, tentatively
        self.assertEqual(curve.classify({42: 240}), 'T_DOWN')
        self.assertEqual(curve.classify({2024: 300, 2025: 300}), 'UP')  # seed 42 is never part of the decision
        self.assertEqual(curve.classify({2024: 300}), 'MISSING')

    def test_seed42_does_not_enter_the_definite_state(self):
        self.assertEqual(curve.classify({42: 100, 2024: 281, 2025: 282}), 'UP')
        self.assertEqual(curve.classify({42: 400, 2024: 270, 2025: 270}), 'FLAT')


GRID = {  # rows: B512 state, columns: B1024 state; order UP, DOWN, FLAT, T_UP, T_DOWN, T_FLAT
    'UP': 'RRRRRR', 'DOWN': 'RPNUTT', 'FLAT': 'RNNUTT', 'T_UP': 'RUUUUU', 'T_DOWN': 'RTTUTT', 'T_FLAT': 'RTTUTT'}
STATES = ('UP', 'DOWN', 'FLAT', 'T_UP', 'T_DOWN', 'T_FLAT')
CODES = {'R': 'RISING', 'U': 'UNDECIDED', 'P': 'PEAK_AT_256', 'N': 'NOT_RISING', 'T': 'TENTATIVE_NOT_RISING'}


class Q1Tests(unittest.TestCase):
    def test_all_36_combinations(self):
        for row, a in enumerate(STATES):
            for column, b in enumerate(STATES):
                expected = CODES[GRID[a][column]]
                with self.subTest(b512=a, b1024=b):
                    self.assertEqual(curve.judge_q1(a, b)['verdict'], expected)

    def test_priorities_are_not_swapped(self):
        self.assertEqual(curve.judge_q1('UP', 'T_UP')['verdict'], 'RISING')  # row 1 before row 2
        self.assertEqual(curve.judge_q1('DOWN', 'T_UP')['verdict'], 'UNDECIDED')  # row 2 before row 3
        self.assertEqual(curve.judge_q1('DOWN', 'DOWN')['verdict'], 'PEAK_AT_256')  # row 3 before row 4
        self.assertEqual(curve.judge_q1('FLAT', 'DOWN')['verdict'], 'NOT_RISING')  # row 4, not the peak row
        self.assertEqual(curve.judge_q1('DOWN', 'T_DOWN')['verdict'], 'TENTATIVE_NOT_RISING')  # row 5

    def test_rows_are_reported(self):
        self.assertEqual([curve.judge_q1(a, b)['row'] for a, b in (('UP', 'FLAT'), ('T_UP', 'FLAT'), ('DOWN', 'DOWN'), ('FLAT', 'FLAT'), ('T_FLAT', 'FLAT'))],
                         [1, 2, 3, 4, 5])

    def test_peak_subrule_needs_both_seeds_of_both_points(self):
        base = {2024: 270, 2025: 270}
        up = curve.judge_q1('UP', 'UP', {2024: 280, 2025: 281}, {2024: 288, 2025: 290})
        self.assertEqual(up['peak'], 'B1024')
        down = curve.judge_q1('UP', 'UP', {2024: 290, 2025: 291}, {2024: 280, 2025: 281})
        self.assertEqual(down['peak'], 'B512')
        flat = curve.judge_q1('UP', 'UP', {2024: 280, 2025: 281}, {2024: 282, 2025: 279})
        self.assertEqual(flat['peak'], 'B512_B1024_INDISTINGUISHABLE')
        self.assertEqual(curve.judge_q1('UP', 'UP', base, {2024: 280})['peak'], 'UNDECIDED')
        self.assertEqual(curve.judge_q1('UP', 'UP')['peak'], 'UNDECIDED')
        self.assertIsNone(curve.judge_q1('FLAT', 'FLAT', base, base)['peak'])

    def test_the_text_never_claims_more_than_the_rule(self):
        for a, b in itertools.product(STATES, repeat=2):
            text = curve.judge_q1(a, b)['text']
            for forbidden in ('significant', 'causes', 'caused', 'proves'):
                self.assertNotIn(forbidden, text)


class SpearmanAndOutputTests(unittest.TestCase):
    def test_spearman(self):
        self.assertAlmostEqual(curve.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(curve.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
        self.assertAlmostEqual(curve.spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)
        self.assertIsNone(curve.spearman([1, 1, 1], [1, 2, 3]))

    def points(self):
        hits = {'B16': {42: 190, 2024: 181, 2025: 208}, 'B256D3': {42: 254, 2024: 274, 2025: 275}, 'B32': {42: 228},
                'B512': {42: 262, 2024: 281, 2025: 282}}
        audits = {'B16': KNOWN['T0'], 'B256D3': KNOWN['T3'], 'B32': KNOWN['B32'], 'B512': audit(120415, 120415, 1, 0, 0, [512, 236, 1, 1], 512)}
        return {label: {'audit': audits[label], 'hits': hits[label], 'status': 'ok'} for label in hits}

    def test_rows_skip_missing_points_and_never_interpolate(self):
        points = self.points()
        points['B1024'] = {'audit': None, 'hits': {}, 'status': 'FIT_FAILED'}
        rows = curve.rows(points)
        self.assertEqual([r['label'] for r in rows], ['B16', 'B32', 'B256D3', 'B512', 'B1024'])
        failed = rows[-1]
        self.assertEqual((failed['status'], failed['CER'], failed['hits_s42']), ('FIT_FAILED', None, None))
        by = {r['label']: r for r in rows}
        self.assertEqual(by['B32']['hits_s2024'], None)
        self.assertEqual(by['B256D3']['hits_s2025'], 275)
        self.assertEqual(by['B32']['CER'], 87647 / 120415)

    def test_write_outputs_csv_markdown_and_optional_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = curve.write_outputs(self.points(), Path(tmp) / 'curve')
            self.assertTrue({'curve.csv', 'curve.md'} <= {p.name for p in written})
            text = (Path(tmp) / 'curve' / 'curve.csv').read_text().splitlines()
            self.assertEqual(text[0].split(',')[:3], ['label', 'branching', 'depth'])
            self.assertEqual(len(text), 1 + 4)
            self.assertIn('| B512 |', (Path(tmp) / 'curve' / 'curve.md').read_text())
            with self.assertRaises(FileExistsError):
                curve.write_outputs(self.points(), Path(tmp) / 'curve')
            if not curve.has_matplotlib():
                self.assertEqual(sorted(p.name for p in written), ['curve.csv', 'curve.md'])

    def test_sources_do_not_touch_the_test_split(self):
        for name in ('gtok02_select', 'gtok02_curve'):
            text = (ROOT / 'scripts' / f'{name}.py').read_text().lower()
            for forbidden in ('week_26', 'week-26', 'week26', "'test'", '"test"', 'test_gate', '.parquet', 'torch'):
                self.assertNotIn(forbidden, text, (name, forbidden))


if __name__ == '__main__':
    unittest.main()
