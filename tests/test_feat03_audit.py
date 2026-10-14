"""FEAT-03 FEAT-001: Q3 train-only bucket audit (pure functions, synthetic end-to-end, holdout guard)."""
import builtins
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'feat03_audit_constants.py'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat01 import prepared_world  # noqa: E402


def load_script():
    spec = importlib.util.spec_from_file_location('feat03_audit_constants', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canon_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def spread(totals, edges, n):
    """Per-value counts whose bucket totals equal `totals` (all mass on the first value of a bucket)."""
    counts = [0] * n
    for start, total in zip([0, *edges], totals):
        counts[start] = total
    return counts


class AuditPureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script()

    def test_registered_grids_and_rule(self):
        self.assertEqual(self.m.LEN_GRID, (4, 8, 16, 32, 50))
        self.assertEqual(self.m.ACT_GRID, (2, 3, 5, 9, 17))
        self.assertEqual(self.m.MIN_SHARE, Fraction(1, 50))
        self.assertEqual(self.m.MAX_LEN, 50)

    def test_length_and_activity_histograms(self):
        histories = [[1, 2, 3], [4], [5] * 50]
        counts = self.m.length_hist(histories)
        self.assertEqual(len(counts), 51)
        self.assertEqual((counts[0], counts[1], counts[3], counts[50]), (0, 1, 1, 1))
        self.assertEqual(sum(counts), 3)
        with self.assertRaises(ValueError):
            self.m.length_hist([[]])                              # an example always has >= 1 history event
        with self.assertRaises(ValueError):
            self.m.length_hist([[1] * 51])

    def test_activity_counts_distinct_weeks_not_span(self):
        # weeks 0,0,5 -> 2 distinct weeks (a span would say 5); weeks 0,3,7 -> 3 distinct (span 7)
        a = [[0, 1], [0, 2], [5, 0]]
        b = [[0, 0], [3, 0], [7, 0]]
        c = [[24, 4]]
        self.assertEqual(self.m.distinct_weeks(a), 2)
        self.assertEqual(self.m.distinct_weeks(b), 3)
        self.assertEqual(self.m.distinct_weeks(c), 1)
        counts = self.m.activity_hist([a, b, c])
        self.assertEqual(len(counts), 26)
        self.assertEqual(counts[:4], [0, 1, 1, 1])
        self.assertEqual(sum(counts), 3)
        # a translation of every week leaves the count unchanged
        self.assertEqual(self.m.distinct_weeks([[w + 10, r] for w, r in b]), 3)
        for bad in ([[25, 0]], [[-1, 0]], [[True, 0]], []):
            with self.assertRaises(ValueError):
                self.m.distinct_weeks(bad)

    def test_merge_rule_is_mechanical(self):
        d = self.m.merge_small_buckets
        edges = (1, 3, 5)
        # nothing to merge: a bucket at exactly 2% is NOT below the threshold
        counts = [1] * 50
        r = d(counts, (1, 2, 4, 8, 16, 32), Fraction(1, 50))
        self.assertEqual(r['edges'], (1, 2, 4, 8, 16, 32))
        self.assertEqual(r['counts'], [1, 1, 2, 4, 8, 16, 18])
        self.assertEqual(r['steps'], [])
        # smallest bucket merges into the smaller neighbour (b0=10 < b2=40) -> edge 1 disappears
        r = d(spread([10, 1, 40, 49], edges, 6), edges, Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((3, 5), [11, 40, 49]))
        # tie between neighbours -> the lower one
        r = d(spread([30, 1, 30, 39], edges, 6), edges, Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((3, 5), [31, 30, 39]))
        # cascade, smallest-tie takes the lower index: [1,1,1,97] -> [2,1,97] -> [3,97]
        r = d(spread([1, 1, 1, 97], edges, 6), edges, Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((5,), [3, 97]))
        self.assertEqual(len(r['steps']), 2)
        # the last bucket too small merges left; the first one merges right
        r = d([49, 50, 1], (2,), Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((), [100]))
        r = d([1, 50, 49], (1,), Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((), [100]))
        with self.assertRaises(ValueError):
            d([0, 0, 0], (1,), Fraction(1, 50))

    def test_bucket_shares_and_table_hash(self):
        counts = spread([10, 30, 60], (1, 3), 6)
        table = self.m.bucket_table(counts, (1, 3))
        self.assertEqual(table['edges'], [1, 3])
        self.assertEqual(table['counts'], [10, 30, 60])
        self.assertEqual(table['shares'], [0.1, 0.3, 0.6])
        self.assertEqual(self.m.table_sha256(table), canon_sha(table))
        other = self.m.bucket_table(spread([11, 29, 60], (1, 3), 6), (1, 3))
        self.assertNotEqual(self.m.table_sha256(other), self.m.table_sha256(table))

    def test_holdout_guard_rejects_forbidden_fragments(self):
        guard = self.m.guard_path
        guard('/a/h2/manifest.json')
        guard('/a/h2/train-l50.json')
        guard('/a/b/latest/x.json')                                # 'latest' is not the test partition
        for bad in ('/a/b/' + 'validation' + '-queries/x.json', '/a/b/' + 'validation' + '-queries.json',
                    '/a/b/' + 'te' + 'st/x.json', '/a/b/week_' + '26.parquet', '/a/b/week_' + '25.parquet'):
            with self.assertRaises(ValueError, msg=bad):
                guard(bad)

    def test_script_source_is_clean_and_cli_is_minimal(self):
        source = SCRIPT.read_text()
        lowered = source.lower()
        self.assertNotIn('holdout', lowered)
        self.assertNotIn('validation' + '-queries', lowered)       # fragments are assembled at run time in the script
        self.assertNotIn('/work' + 'space', lowered)
        self.assertNotIn('/us' + 'ers/', lowered)
        self.assertEqual(set(inspect.signature(self.m.main).parameters), {'argv'})
        args = self.m.build_parser().parse_args(['--inputs-spec', 'a', '--src', 'b'])
        self.assertEqual(set(vars(args)), {'inputs_spec', 'src'})
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                self.m.build_parser().parse_args(['--inputs-spec', 'a', '--src', 'b', '--out', 'c'])


class AuditEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script()

    def run_main(self, tmp, opened=None):
        _, _, _, _, _, h2, _ = prepared_world(tmp)
        spec = Path(tmp) / 'spec.json'
        spec.write_text(json.dumps({'history_manifest': str(Path(tmp) / 'h2' / 'manifest.json'), 'history_sha256': h2['manifest_sha256']}))
        out = io.StringIO()
        real_open = builtins.open

        def spy(file, *args, **kwargs):
            if opened is not None:
                opened.append(str(file))
            return real_open(file, *args, **kwargs)

        with mock.patch('builtins.open', spy), contextlib.redirect_stdout(out):
            code = self.m.main(['--inputs-spec', str(spec), '--src', str(ROOT / 'src')])
        return code, out.getvalue(), h2

    def test_synthetic_world_matches_independent_count(self):
        from vkgr import history_v2
        with tempfile.TemporaryDirectory() as tmp:
            opened = []
            code, text, h2 = self.run_main(tmp, opened)
            loaded = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 50)
            self.assertEqual(code, 0)
            self.assertEqual(text.count('\n'), 1)                  # one JSON document, one line
            out = json.loads(text)
            self.assertEqual(json.dumps(out, sort_keys=True, separators=(',', ':')) + '\n', text)
            self.assertNotIn(tmp, text)                            # no paths in stdout
            self.assertNotIn('/', text)
            rows = loaded['train']
            n = [len(r['history'][-50:]) for r in rows]
            a = [len({p[0] for p in r['history_positions'][-50:]}) for r in rows]
            self.assertEqual(out['examples'], len(rows))
            self.assertEqual(out['n_equals_50'], {'count': sum(x == 50 for x in n), 'share': sum(x == 50 for x in n) / len(n)})
            for key, values, size, grid in (('len', n, 51, (4, 8, 16, 32, 50)), ('act', a, 26, (2, 3, 5, 9, 17))):
                counts = [values.count(v) for v in range(size)]
                expected = self.m.merge_small_buckets(counts, grid)
                doc = out[key]
                self.assertEqual(doc['candidate_edges'], list(grid))
                self.assertEqual(doc['final_edges'], list(expected['edges']))
                self.assertEqual(doc['final_counts'], expected['counts'])
                self.assertEqual(sum(doc['final_counts']), len(rows))
                self.assertEqual(doc['final_shares'], [c / len(rows) for c in expected['counts']])
                self.assertEqual(doc['merge_steps'], expected['steps'])
            self.assertEqual(out['merge_count'], len(out['len']['merge_steps']) + len(out['act']['merge_steps']))
            self.assertEqual(out['table_sha256']['len'], canon_sha({'edges': out['len']['final_edges'], 'counts': out['len']['final_counts'], 'shares': out['len']['final_shares']}))
            self.assertEqual(out['table_sha256']['act'], canon_sha({'edges': out['act']['final_edges'], 'counts': out['act']['final_counts'], 'shares': out['act']['final_shares']}))
            self.assertEqual(out['schema'], 'feat03-audit-v1')
            self.assertTrue(out['train_only'])
            # holdout guard: no validation-queries / test partition path was opened
            forbidden = ('validation' + '-queries', '/te' + 'st/', 'week_' + '26')
            self.assertTrue(opened)
            for path in opened:
                for fragment in forbidden:
                    self.assertNotIn(fragment, path)

    def test_main_refuses_a_holdout_path_in_the_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / 'spec.json'
            spec.write_text(json.dumps({'history_manifest': str(Path(tmp) / ('validation' + '-queries') / 'manifest.json'), 'history_sha256': '0' * 64}))
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()) as out:
                self.m.main(['--inputs-spec', str(spec), '--src', str(ROOT / 'src')])
            self.assertEqual(out.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
