"""FEAT-001 (FEAT-02 P line): constants audit script, pure functions + holdout guard."""
import importlib.util
import inspect
import unittest
from fractions import Fraction
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'feat02_audit_constants.py'


def load_script():
    spec = importlib.util.spec_from_file_location('feat02_audit_constants', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spread(totals, edges, n):
    """Per-position counts whose bucket totals equal `totals` (all mass on the first position of a bucket)."""
    starts = [0, *edges]
    counts = [0] * n
    for start, total in zip(starts, totals):
        counts[start] = total
    return counts


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script()

    def test_position_and_weekgap_histograms(self):
        # history order is oldest -> newest; back-index 0 = most recent event
        a = [[0, 3], [2, 1], [2, 5]]   # back-index of events: 2,1,0 ; week gaps vs last (week 2): 2,0,0
        b = [[1, 0], [4, 2]]           # back-index 1,0 ; gaps vs last (week 4): 3,0
        c = [[24, 9]]                  # back-index 0 ; gap 0
        pos = self.m.position_hist([a, b, c])
        gap = self.m.weekgap_hist([a, b, c])
        self.assertEqual(len(pos), 50)
        self.assertEqual(len(gap), 25)
        self.assertEqual(pos[:4], [3, 2, 1, 0])
        self.assertEqual(sum(pos), 6)
        self.assertEqual(gap[:5], [4, 0, 1, 1, 0])  # gap 0: a twice, b once, c once
        self.assertEqual(sum(gap), 6)
        # a 50-long history fills back-index 49 exactly once
        long_history = [[0, i] for i in range(50)]
        self.assertEqual(self.m.position_hist([long_history])[49], 1)
        with self.assertRaises(ValueError):
            self.m.position_hist([[[0, i] for i in range(51)]])

    def test_derive_edges_merge_rule_is_mechanical(self):
        d = self.m.derive_edges
        # nothing to merge: shares 2% exactly are NOT below the 2% threshold
        pos_edges = (1, 2, 4, 8, 16, 32)
        counts = [1, 1, 1, 1, 1, 1, 1, 1] + [1] * 42  # 50 positions, one event each
        r = d(counts, pos_edges, Fraction(1, 50))
        self.assertEqual(r['edges'], pos_edges)
        self.assertEqual(r['counts'], [1, 1, 2, 4, 8, 16, 18])
        self.assertEqual(r['steps'], [])
        # smallest bucket merges into the smaller neighbour (b0=10 < b2=40) -> edge 1 disappears
        edges = (1, 3, 5)
        r = d(spread([10, 1, 40, 49], edges, 6), edges, Fraction(1, 50))
        self.assertEqual(r['edges'], (3, 5))
        self.assertEqual(r['counts'], [11, 40, 49])
        # tie between neighbours -> lower neighbour (b0), not (1, 5)
        r = d(spread([30, 1, 30, 39], edges, 6), edges, Fraction(1, 50))
        self.assertEqual(r['edges'], (3, 5))
        self.assertEqual(r['counts'], [31, 30, 39])
        # cascade: [1,1,1,97] -> tie of smallest takes the lower index -> [2,1,97] -> [3,97]
        r = d(spread([1, 1, 1, 97], edges, 6), edges, Fraction(1, 50))
        self.assertEqual(r['edges'], (5,))
        self.assertEqual(r['counts'], [3, 97])
        self.assertEqual(len(r['steps']), 2)
        # last bucket too small -> merges left
        r = d([49, 50, 1], (2,), Fraction(1, 50))
        self.assertEqual(r['edges'], ())
        self.assertEqual(r['counts'], [100])
        # week-gap grid works on 25 bins with the same rule
        gap_edges = (1, 2, 3, 5, 9, 17)
        r = d([100] * 25, gap_edges, Fraction(1, 50))
        self.assertEqual(r['edges'], gap_edges)
        self.assertEqual(r['counts'], [100, 100, 100, 200, 400, 800, 800])
        with self.assertRaises(ValueError):
            d([0, 0, 0], (1,), Fraction(1, 50))

    def test_category_table_sorted_unique_with_counts(self):
        t = self.m.category_table([[3, 1, 3], [2, 1], []])
        self.assertEqual(t, {'values': [1, 2, 3], 'counts': [2, 1, 2]})
        import numpy as np
        t = self.m.category_table([np.array([5, 5, 0], dtype=np.uint8), np.array([7], dtype=np.uint8)])
        self.assertEqual(t, {'values': [0, 5, 7], 'counts': [1, 2, 1]})
        self.assertTrue(all(type(v) is int for v in t['values'] + t['counts']))

    def test_author_vocab_from_supported_items_only(self):
        catalog = [10, 11, 12, 13, 14]
        support = {10, 11, 12}
        authors = {10: 7, 11: 7, 12: 5, 13: 5, 14: 99}
        base = self.m.author_vocab(catalog, support, authors)
        self.assertEqual(base['vocab'], [5, 7])
        self.assertEqual(base['vocab_size'], 2)
        self.assertEqual(base['unsupported_items'], 2)
        self.assertEqual(base['unsupported_with_vocab_author_fraction'], 0.5)
        self.assertEqual(base['supported_with_vocab_author_fraction'], 1.0)
        # unsupported item's author changed -> vocab and sha unchanged
        moved = self.m.author_vocab(catalog, support, {**authors, 14: 1234})
        self.assertEqual(moved['vocab'], base['vocab'])
        self.assertEqual(moved['vocab_sha256'], base['vocab_sha256'])
        # supported item's author changed -> vocab and sha change
        changed = self.m.author_vocab(catalog, support, {**authors, 12: 6})
        self.assertEqual(changed['vocab'], [6, 7])
        self.assertNotEqual(changed['vocab_sha256'], base['vocab_sha256'])
        # sha = sha256 of the compact ascending JSON list
        import hashlib
        self.assertEqual(base['vocab_sha256'], hashlib.sha256(b'[5,7]').hexdigest())
        with self.assertRaises(ValueError):
            self.m.author_vocab(catalog, support, {10: 7})  # catalog item without metadata

    def test_audit_rejects_non_train_weeks(self):
        with self.assertRaises(ValueError):
            self.m.weekgap_hist([[[24, 0], [25, 0]]])
        with self.assertRaises(ValueError):
            self.m.position_hist([[[25, 0]]])
        with self.assertRaises(ValueError):
            self.m.weekgap_hist([[[-1, 0]]])

    def test_audit_never_references_holdout(self):
        source = SCRIPT.read_text().lower()
        self.assertNotIn('holdout', source)
        self.assertNotIn('validation-queries', source)
        # the path fragments are assembled at run time so this file itself stays clean under the public-repo path scan
        self.assertNotIn('/work' + 'space', source)
        self.assertNotIn('/us' + 'ers/', source)
        self.assertEqual(set(inspect.signature(self.m.main).parameters), {'argv'})
        parser_args = self.m.build_parser().parse_args(['--inputs-spec', 'a', '--src', 'b'])
        self.assertEqual(set(vars(parser_args)), {'inputs_spec', 'src'})


if __name__ == '__main__':
    unittest.main()
