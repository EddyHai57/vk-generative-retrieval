"""FEAT-04b FEAT-002: C/D train-only statistics audit (pure functions, synthetic end-to-end against an independent brute force, holdout guard) and the
audit launcher (a copy of the FEAT-03 launcher with the business script name replaced)."""
import builtins
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import random
import re
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'feat04b_audit_constants.py'
LAUNCHER = ROOT / 'scripts' / 'feat04b_audit_launch.py'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_history_v2 import synthetic_world  # noqa: E402

AC16_FIELDS = ('sum_likes', 'n_train_examples', 'sum_likes_minus_n_train_examples', 'likes_vs_target_freq', 'timespent', 'duration', 'stats_self_inclusion',
               'p_like_report_threshold', 'p_like_exceeds_threshold', 'logq_redundancy', 'grids', 'like_p0', 'stats_sha256')
SELF_FIELDS = ('examples', 'p_expo', 'p_ratio', 'p_like', 'p_first', 'p_first_is_own_row', 'share_n_le_3', 'share_n_eq_1')


def load_script():
    spec = importlib.util.spec_from_file_location('feat04b_audit_constants', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canon_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# ---------------------------------------------------------------- the same four bucket formulas, written independently (design section 3.4)

def b_expo(n, edges):
    return 1 + sum(n >= e for e in edges)


def b_first(week, edges):
    return 1 + sum(week >= e for e in edges)


def b_ratio(total, n, edges):
    return 1 + sum(total >= e * n for e in edges)


def b_like(likes, n, edges, k, p0):
    return 1 + sum(1000 * likes + k * p0 >= e * (n + k) for e in edges)


class AuditPureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script()

    def test_registered_constants(self):
        m = self.m
        self.assertEqual(m.EXPO_GRID, (2, 4, 8, 16, 32, 64, 128, 256, 512))
        self.assertEqual(m.FIRST_GRID, (1, 3, 6, 10, 15, 20, 23))
        self.assertEqual(m.RATIO_GRID, (200, 350, 500, 650, 800, 1000, 1250))
        self.assertEqual(m.LIKE_GRID, (5, 10, 20, 40, 80, 160))
        self.assertEqual((m.LIKE_K, m.MIN_SHARE), (10, Fraction(1, 50)))
        self.assertEqual(m.P_LIKE_REPORT_THRESHOLD, 0.30)
        self.assertIs(type(m.P_LIKE_REPORT_THRESHOLD), float)

    def test_bucket_formulas_match_the_independent_reference(self):
        m = self.m
        rng = random.Random(1)
        for _ in range(300):
            n = rng.randint(1, 700); week = rng.randint(0, 24); likes = rng.randint(0, n); total = rng.randint(0, 2000 * n); p0 = rng.randint(0, 400)
            self.assertEqual(m.expo_bucket(n, m.EXPO_GRID), b_expo(n, m.EXPO_GRID))
            self.assertEqual(m.first_bucket(week, m.FIRST_GRID), b_first(week, m.FIRST_GRID))
            self.assertEqual(m.ratio_bucket(total, n, m.RATIO_GRID), b_ratio(total, n, m.RATIO_GRID))
            self.assertEqual(m.like_bucket(likes, n, m.LIKE_GRID, m.LIKE_K, p0), b_like(likes, n, m.LIKE_GRID, m.LIKE_K, p0))
        self.assertEqual(m.expo_bucket(1, m.EXPO_GRID), 1)
        self.assertEqual(m.expo_bucket(2, m.EXPO_GRID), 2)                 # an edge is the first value of its bucket
        self.assertEqual(m.ratio_bucket(1000, 5, (200,)), 2)               # S >= e*n with e = 200: 1000 >= 200 * 5
        self.assertEqual(m.ratio_bucket(999, 5, (200,)), 1)

    def test_like_p0_is_a_half_up_integer(self):
        self.assertEqual(self.m.like_p0(30, 1000), 30)
        self.assertEqual(self.m.like_p0(1, 4000), 0)                       # 0.25 permille -> 0
        self.assertEqual(self.m.like_p0(1, 2000), 1)                       # 0.5 -> half up
        self.assertEqual(self.m.like_p0(3, 2000), 2)                       # 1.5 -> 2
        with self.assertRaises(ValueError):
            self.m.like_p0(1, 0)

    def test_merge_rule_is_mechanical(self):
        d = self.m.merge_small_buckets
        edges = (1, 3, 5)
        # nothing to merge: a bucket at exactly 2% is NOT below the threshold
        r = d([1, 1, 2, 4, 8, 16, 18], (1, 2, 4, 8, 16, 32), Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts'], r['steps']), ((1, 2, 4, 8, 16, 32), [1, 1, 2, 4, 8, 16, 18], []))
        # smallest bucket merges into the smaller neighbour (b0=10 < b2=40) -> edge 1 disappears
        r = d([10, 1, 40, 49], edges, Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((3, 5), [11, 40, 49]))
        # tie between neighbours -> the lower one
        r = d([30, 1, 30, 39], edges, Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((3, 5), [31, 30, 39]))
        # cascade, smallest-tie takes the lower index: [1,1,1,97] -> [2,1,97] -> [3,97]
        r = d([1, 1, 1, 97], edges, Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((5,), [3, 97]))
        self.assertEqual(len(r['steps']), 2)
        self.assertEqual(r['steps'][0], {'merged_bucket': 0, 'into_bucket': 1, 'removed_edge': 1})
        # the last bucket too small merges left; the first one merges right
        r = d([49, 50, 1], (2, 9), Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((2,), [49, 51]))
        r = d([1, 50, 49], (1, 9), Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((9,), [51, 49]))
        # a single bucket stays
        r = d([100], (), Fraction(1, 50))
        self.assertEqual((r['edges'], r['counts']), ((), [100]))
        for bad in ([0, 0, 0], []):
            with self.assertRaises(ValueError):
                d(bad, (1, 2)[:max(len(bad) - 1, 0)], Fraction(1, 50))
        with self.assertRaises(ValueError):
            d([1, 2, 3], (1,), Fraction(1, 50))                            # edges must be one fewer than the buckets

    def test_script_source_is_clean_and_cli_is_minimal(self):
        source = SCRIPT.read_text()
        lowered = source.lower()
        self.assertNotIn('holdout', lowered)
        self.assertNotIn('validation' + '-queries', lowered)
        self.assertNotIn('/work' + 'space', lowered)
        self.assertNotIn('/us' + 'ers/', lowered)
        self.assertEqual(set(inspect.signature(self.m.main).parameters), {'argv'})
        args = self.m.build_parser().parse_args(['--inputs-spec', 'a', '--src', 'b'])
        self.assertEqual(set(vars(args)), {'inputs_spec', 'src'})
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                self.m.build_parser().parse_args(['--inputs-spec', 'a', '--src', 'b', '--out', 'c'])

    def test_holdout_guard_rejects_forbidden_fragments(self):
        guard = self.m.guard_path
        guard('/a/prepared/prepare-manifest.json')
        guard('/a/raw/subsamples/x/train/week_00.parquet')
        guard('/a/raw/subsamples/x/train/week_24.parquet')
        for bad in ('/a/b/' + 'validation' + '-queries/x.json', '/a/b/' + 'validation' + '-queries.json', '/a/b/' + 'te' + 'st/x.json',
                    '/a/b/week_' + '26.parquet', '/a/b/week_' + '25.parquet'):
            with self.assertRaises(ValueError, msg=bad):
                guard(bad)


# ---------------------------------------------------------------- synthetic end to end

def write_world(tmp, seed=3):
    """A pinned synthetic prepared dir + 25 raw train weeks + items_metadata.parquet. Returns (spec_path, partitions, examples, catalog, durations, support, prepared_sha)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vkgr.data_audit import file_sha256
    from vkgr.retrieval_data import prefix_examples, write_prepared
    tmp = Path(tmp)
    partitions, _, _, catalog = synthetic_world(seed, users=6, weeks=25, rows_per_week=40, items=40)
    rng = random.Random(seed + 100)
    durations = {i: rng.choice([5, 9, 25, 60, 180]) for i in catalog}
    for _, rows in partitions:
        for r in rows:
            for name in ('share', 'bookmark', 'open_comments', 'click_on_author', 'dislike'):
                r[name] = rng.random() < 0.2
            r.update(place=rng.randrange(8), platform=rng.randrange(4), agent=rng.randrange(3), timespent=rng.choice([0, 1, 2, 5, 9, 12, 40, 70, 150, 255]))
    examples = list(prefix_examples(partitions))
    support = sorted({r['item_id'] for _, rows in partitions for r in rows if r['like']})
    records = []
    for week, rows in partitions:
        role = f'subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet'
        path = tmp / 'raw' / role; path.parent.mkdir(parents=True, exist_ok=True)
        types = {'user_id': pa.uint32(), 'item_id': pa.uint32(), 'place': pa.uint8(), 'platform': pa.uint8(), 'agent': pa.uint8(), 'timespent': pa.uint8()}
        columns = {k: pa.array([r[k] for r in rows], t) for k, t in types.items()}
        columns.update({k: pa.array([r[k] for r in rows], pa.bool_()) for k in ('like', 'dislike', 'share', 'bookmark', 'click_on_author', 'open_comments')})
        pq.write_table(pa.table(columns), str(path))
        records.append({'role': role, 'sha256': file_sha256(path), 'size_bytes': path.stat().st_size})
    (tmp / 'raw' / 'metadata').mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({'item_id': pa.array(catalog, pa.uint32()), 'author_id': pa.array([1] * len(catalog), pa.uint32()),
                             'duration': pa.array([durations[i] for i in catalog], pa.uint8())}), str(tmp / 'raw' / 'metadata' / 'items_metadata.parquet'))
    write_prepared(tmp / 'prepared', catalog, np.zeros((len(catalog) + 1, 64), dtype=np.float32), examples,
                   {'input_records': records, 'consumer': {'source_fingerprint': 'fixture'}}, support)
    spec = {'schema': 'p-feat02-inputs-v1', 'prepared': str(tmp / 'prepared'), 'holdout': str(tmp / 'no-such-holdout'), 'raw_root': str(tmp / 'raw'),
            'items_metadata_sha256': file_sha256(tmp / 'raw' / 'metadata' / 'items_metadata.parquet')}
    (tmp / 'spec.json').write_text(json.dumps(spec))
    return tmp / 'spec.json', partitions, examples, catalog, durations, support, file_sha256(tmp / 'prepared' / 'prepare-manifest.json')


def brute_force(partitions, examples, catalog, durations, support, out):
    """Everything the audit reports, recomputed row by row without any item_stats code (and with the audit's own final edges / LIKE_P0)."""
    rows = {i: [] for i in catalog}
    for week, events in partitions:
        for ordinal, r in enumerate(events):
            rows[r['item_id']].append((week, ordinal, int(r['like']), min(2000, 1000 * r['timespent'] // durations[r['item_id']]), r['timespent']))
    stat = {i: (len(v), min(x[0] for x in v), sum(x[2] for x in v), sum(x[3] for x in v)) for i, v in rows.items()}
    grids = out['grids']; k = out['like_k']; p0 = out['like_p0']

    def buckets(n, first, likes, total):
        return {'expo': b_expo(n, grids['expo']['final_edges']), 'first': b_first(first, grids['first']['final_edges']),
                'ratio': b_ratio(total, n, grids['ratio']['final_edges']), 'like': b_like(likes, n, grids['like']['final_edges'], k, p0)}
    changed = dict.fromkeys(('expo', 'ratio', 'like', 'first', 'own'), 0); small = single = 0
    for e in examples:
        item, position = e['target'], tuple(e['target_position'])
        n, first, likes, total = stat[item]
        rest = [x for x in rows[item] if (x[0], x[1]) != position]
        self_row = next(x for x in rows[item] if (x[0], x[1]) == position)
        n2 = len(rest); likes2 = sum(x[2] for x in rest); total2 = sum(x[3] for x in rest)
        before = buckets(n, first, likes, total)
        changed['expo'] += b_expo(n2, grids['expo']['final_edges']) != before['expo']
        changed['like'] += b_like(likes2, n2, grids['like']['final_edges'], k, p0) != before['like']
        changed['ratio'] += n2 == 0 or b_ratio(total2, n2, grids['ratio']['final_edges']) != before['ratio']
        changed['first'] += n2 == 0 or b_first(min(x[0] for x in rest), grids['first']['final_edges']) != before['first']
        changed['own'] += (self_row[0], self_row[1]) == min((x[0], x[1]) for x in rows[item])
        small += n <= 3; single += n == 1
    return stat, changed, small, single


class AuditEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script()

    def run_main(self, tmp, opened=None, **world_kwargs):
        from vkgr import retrieval_ablation_run as ablation
        spec, partitions, examples, catalog, durations, support, prepared_sha = write_world(tmp, **world_kwargs)
        out = io.StringIO()
        real_open = builtins.open

        def spy(file, *args, **kwargs):
            if opened is not None:
                opened.append(str(file))
            return real_open(file, *args, **kwargs)

        real_path_open = Path.open

        def path_spy(self, *args, **kwargs):
            if opened is not None:
                opened.append(str(self))
            return real_path_open(self, *args, **kwargs)

        import pyarrow.parquet as pq
        real_file, real_table = pq.ParquetFile, pq.read_table

        def file_spy(path, *args, **kwargs):
            if opened is not None:
                opened.append(str(path))
            return real_file(path, *args, **kwargs)

        def table_spy(path, *args, **kwargs):
            if opened is not None:
                opened.append(str(path))
            return real_table(path, *args, **kwargs)

        with mock.patch.object(ablation, 'PREPARED_SHA', prepared_sha), mock.patch('builtins.open', spy), mock.patch.object(Path, 'open', path_spy), \
                mock.patch.object(pq, 'ParquetFile', file_spy), mock.patch.object(pq, 'read_table', table_spy), contextlib.redirect_stdout(out):
            code = self.m.main(['--inputs-spec', str(spec), '--src', str(ROOT / 'src')])
        return code, out.getvalue(), (partitions, examples, catalog, durations, support, prepared_sha)

    def test_synthetic_world_matches_an_independent_brute_force(self):
        from vkgr import item_stats
        with tempfile.TemporaryDirectory() as tmp:
            opened = []
            code, text, (partitions, examples, catalog, durations, support, prepared_sha) = self.run_main(tmp, opened)
            self.assertEqual(code, 0)
            self.assertEqual(text.count('\n'), 1)                                   # one JSON document, one line
            out = json.loads(text)
            self.assertEqual(json.dumps(out, sort_keys=True, separators=(',', ':')) + '\n', text)
            self.assertNotIn(tmp, text)                                             # no paths in stdout
            self.assertNotIn('/', text)
            for field in AC16_FIELDS:
                self.assertIn(field, out, field)
            self.assertEqual(out['schema'], 'feat04b-audit-v1')
            self.assertTrue(out['train_only'])
            stat, changed, small, single = brute_force(partitions, examples, catalog, durations, support, out)
            # --- statistics
            expected_stats = {'n': [stat[i][0] for i in catalog], 'first_week': [stat[i][1] for i in catalog], 'likes': [stat[i][2] for i in catalog],
                              'ratio_pm_sum': [stat[i][3] for i in catalog]}
            self.assertEqual(out['stats_sha256'], item_stats.stats_digest(expected_stats))
            self.assertEqual((out['catalog_items'], out['supported_items'], out['rows_total']), (len(catalog), len(support), sum(expected_stats['n'])))
            # --- AC16: likes vs training targets
            target_freq = {}
            for e in examples:
                target_freq[e['target']] = target_freq.get(e['target'], 0) + 1
            self.assertEqual(out['n_train_examples'], len(examples))
            self.assertEqual(out['sum_likes'], sum(expected_stats['likes']))
            self.assertEqual(out['sum_likes_minus_n_train_examples'], sum(expected_stats['likes']) - len(examples))
            equal = sum(stat[i][2] == target_freq.get(i, 0) for i in catalog)
            self.assertEqual(out['likes_vs_target_freq'], {'items_equal': equal, 'share_items_equal': equal / len(catalog),
                                                           'sum_abs_diff': sum(abs(stat[i][2] - target_freq.get(i, 0)) for i in catalog)})
            # --- AC16: raw ranges
            events = [r for _, rows in partitions for r in rows]
            zero = sum(r['timespent'] == 0 for r in events)
            self.assertGreater(zero, 0, 'the fixture must contain timespent == 0 rows')
            self.assertEqual(out['timespent'], {'rows': len(events), 'zero_rows': zero, 'zero_share': zero / len(events),
                                                'min': min(r['timespent'] for r in events), 'max': max(r['timespent'] for r in events)})
            self.assertEqual(out['duration'], {'min': min(durations.values()), 'max': max(durations.values())})
            # --- LIKE_P0 and the grids
            total_likes, total_n = sum(expected_stats['likes']), sum(expected_stats['n'])
            self.assertEqual(out['like_p0'], (2000 * total_likes + total_n) // (2 * total_n))
            self.assertEqual((out['like_k'], out['ratio_cap_pm']), (10, 2000))
            targets = [stat[e['target']] for e in examples]
            for name, grid, bucket in (('expo', self.m.EXPO_GRID, lambda s, edges: b_expo(s[0], edges)), ('first', self.m.FIRST_GRID, lambda s, edges: b_first(s[1], edges)),
                                       ('ratio', self.m.RATIO_GRID, lambda s, edges: b_ratio(s[3], s[0], edges)),
                                       ('like', self.m.LIKE_GRID, lambda s, edges: b_like(s[2], s[0], edges, 10, out['like_p0']))):
                doc = out['grids'][name]
                self.assertEqual(doc['candidate_edges'], list(grid))
                counts = [sum(bucket(s, grid) == b for s in targets) for b in range(1, len(grid) + 2)]
                self.assertEqual(doc['candidate_counts'], counts)
                self.assertEqual(doc['candidate_shares'], [c / len(examples) for c in counts])
                merged = self.m.merge_small_buckets(counts, grid, Fraction(1, 50))
                self.assertEqual(doc['final_edges'], list(merged['edges']))
                self.assertEqual(doc['final_counts'], merged['counts'])
                self.assertEqual(doc['merge_steps'], merged['steps'])
                final = [sum(bucket(s, doc['final_edges']) == b for s in targets) for b in range(1, len(doc['final_edges']) + 2)]
                self.assertEqual(doc['final_counts'], final, 'merged totals equal a recount with the final edges')
                self.assertEqual(sum(doc['final_counts']), len(examples))
                self.assertEqual(doc['final_shares'], [c / len(examples) for c in final])
                self.assertTrue(len(doc['final_edges']) < len(grid) or not doc['merge_steps'])
            self.assertTrue(any(out['grids'][name]['merge_steps'] for name in out['grids']), 'the fixture must exercise the merge rule')
            # --- stats_self_inclusion against a from-scratch recomputation
            size = len(examples); si = out['stats_self_inclusion']
            self.assertEqual(si['examples'], size)
            for key, count in (('p_expo', changed['expo']), ('p_ratio', changed['ratio']), ('p_like', changed['like']), ('p_first', changed['first']),
                               ('p_first_is_own_row', changed['own']), ('share_n_le_3', small), ('share_n_eq_1', single)):
                self.assertEqual(si[key], count / size, key)
            self.assertEqual(out['p_like_report_threshold'], 0.30)
            self.assertEqual(out['p_like_exceeds_threshold'], si['p_like'] > 0.30)
            # --- logq redundancy is present and well-formed
            lq = out['logq_redundancy']
            self.assertEqual(lq['supported_items'], len(support))
            for key in ('expo_spearman', 'like_spearman'):
                self.assertTrue(lq[key] is None or -1.0 <= lq[key] <= 1.0)
            self.assertEqual(out['inputs'], {'items_metadata_sha256': json.loads((Path(tmp) / 'spec.json').read_text())['items_metadata_sha256'], 'prepared_sha256': prepared_sha})
            # --- the holdout guard: nothing outside the prepared dir, the 25 train weeks and the metadata file was opened
            forbidden = ('validation' + '-queries', '/te' + 'st/', 'week_' + '25', 'week_' + '26', 'no-such-holdout')
            self.assertTrue(opened)
            self.assertEqual(sorted({Path(p).name for p in opened if Path(p).name.startswith('week_')}), [f'week_{w:02d}.parquet' for w in range(25)])
            self.assertTrue(any(p.endswith('items_metadata.parquet') for p in opened))
            for path in opened:
                for fragment in forbidden:
                    self.assertNotIn(fragment, path)

    def test_the_audit_does_not_depend_on_anything_outside_train(self):
        """Two runs on worlds that differ only in a holdout directory (absent vs present with garbage) print the same bytes."""
        texts = []
        for garbage in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                if garbage:
                    (Path(tmp) / 'no-such-holdout').mkdir()
                    (Path(tmp) / 'no-such-holdout' / ('validation' + '-queries.json')).write_text('not json')
                code, text, _ = self.run_main(tmp)
                self.assertEqual(code, 0)
                texts.append(text)
        self.assertEqual(texts[0], texts[1])

    def test_main_refuses_a_holdout_path_in_the_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / 'spec.json'
            spec.write_text(json.dumps({'prepared': str(Path(tmp) / ('validation' + '-queries') / 'x'), 'raw_root': str(Path(tmp) / 'raw'), 'items_metadata_sha256': '0' * 64}))
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()) as out:
                self.m.main(['--inputs-spec', str(spec), '--src', str(ROOT / 'src')])
            self.assertEqual(out.getvalue(), '')

    def test_main_fails_closed_on_a_wrong_prepared_pin_and_prints_nothing(self):
        from vkgr import retrieval_ablation_run as ablation
        with tempfile.TemporaryDirectory() as tmp:
            spec, *_ = write_world(tmp)
            out = io.StringIO()
            with mock.patch.object(ablation, 'PREPARED_SHA', '0' * 64), contextlib.redirect_stdout(out):
                with self.assertRaises(ValueError):
                    self.m.main(['--inputs-spec', str(spec), '--src', str(ROOT / 'src')])
            self.assertEqual(out.getvalue(), '')

    def test_main_fails_closed_on_a_stats_defect(self):
        """A defect the statistics builder refuses (a catalog item with no train row) stops the audit before anything is printed."""
        from vkgr import retrieval_ablation_run as ablation
        with tempfile.TemporaryDirectory() as tmp:
            spec, partitions, examples, catalog, durations, support, prepared_sha = write_world(tmp)
            out = io.StringIO()
            with mock.patch.object(ablation, 'PREPARED_SHA', prepared_sha), contextlib.redirect_stdout(out):
                from vkgr import item_stats
                with mock.patch.object(item_stats, 'build_item_stats', side_effect=ValueError('catalog item without any train row (n == 0)')):
                    with self.assertRaises(ValueError):
                        self.m.main(['--inputs-spec', str(spec), '--src', str(ROOT / 'src')])
            self.assertEqual(out.getvalue(), '')


# ---------------------------------------------------------------- launcher (copy of the FEAT-03 launcher tests, names replaced)

GIT_QUIET = ['-c', 'user.name=guard', '-c', 'user.email=guard@example.invalid', '-c', 'commit.gpgsign=false']
BUSINESS = (
    "import json, sys\n"
    "sys.stdout.buffer.write(json.dumps({'argv': sys.argv[1:], 'word': '\\u00e9'}, ensure_ascii=False).encode() + b'\\n\\x00tail-without-newline')\n"
)


def git(repo, *args, check=True):
    return subprocess.run(['git', *GIT_QUIET, '-C', str(repo), *args], check=check, capture_output=True, text=True)


class LauncherTests(unittest.TestCase):
    def make_repo(self, tmp):
        repo = Path(tmp) / 'repo'
        for rel, text in (('scripts/feat04b_audit_constants.py', BUSINESS), ('src/vkgr/mod.py', 'X = 1\n'), ('docs/note.md', 'one\n')):
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(text)
        git(repo, 'init', '-q')
        git(repo, 'add', '-A')
        git(repo, 'commit', '-q', '-m', 'base')
        return repo

    def params(self, repo, **override):
        p = {'worktree': str(repo), 'commit': git(repo, 'rev-parse', 'HEAD').stdout.strip(),
             'tree': git(repo, 'rev-parse', 'HEAD^{tree}').stdout.strip(),
             'script_sha256': hashlib.sha256((repo / 'scripts/feat04b_audit_constants.py').read_bytes()).hexdigest(),
             'inputs_spec': 'spec-path', 'src': 'src-path'}
        p.update(override)
        return p

    def send(self, params):
        lines = LAUNCHER.read_text().split('\n')
        text = '\n'.join(['P = ' + json.dumps(params), *lines[1:]])
        return subprocess.run([sys.executable, '-B', '-'], input=text.encode(), capture_output=True)

    def test_committed_launcher_has_a_single_placeholder_line(self):
        text = LAUNCHER.read_text()
        self.assertEqual(text.split('\n')[0], 'P = None')
        self.assertEqual(len(re.findall(r'(?m)^\s*P\s*=', text)), 1)
        lowered = text.lower()
        for fragment in ('/work' + 'space', '/us' + 'ers/', 'hello' + 'bike', 'validation' + '-queries', 'feat03'):
            self.assertNotIn(fragment, lowered)
        self.assertIn('feat04b_audit_constants.py', text)

    def test_all_good_passes_stdout_through_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self.make_repo(tmp)
            params = self.params(repo)
            result = self.send(params)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = json.dumps({'argv': ['--inputs-spec', 'spec-path', '--src', 'src-path'], 'word': '\u00e9'}, ensure_ascii=False).encode() + b'\n\x00tail-without-newline'
            self.assertEqual(result.stdout, expected)
            receipt = json.loads(result.stderr.decode().strip().splitlines()[-1])
            self.assertEqual(receipt, {'commit': params['commit'], 'tree': params['tree'], 'script_sha256': params['script_sha256']})
            # a modified tracked file outside src/ and scripts/ does not matter
            (repo / 'docs/note.md').write_text('changed\n')
            self.assertEqual(self.send(params).returncode, 0)

    def assert_refused(self, params):
        result = self.send(params)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')

    def test_commit_tree_and_script_mismatches_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self.make_repo(tmp)
            self.assert_refused(self.params(repo, commit='0' * 40))
            self.assert_refused(self.params(repo, tree='1' * 40))
            self.assert_refused(self.params(repo, script_sha256='2' * 64))
            self.assert_refused({'worktree': str(repo)})                            # incomplete parameters
            self.assertEqual(self.send(self.params(repo)).returncode, 0)

    def test_tracked_change_under_src_or_scripts_is_refused(self):
        for rel in ('src/vkgr/mod.py', 'scripts/feat04b_audit_constants.py'):
            with self.subTest(rel), tempfile.TemporaryDirectory() as tmp:
                repo = self.make_repo(tmp)
                params = self.params(repo)
                if rel.startswith('scripts'):
                    params['script_sha256'] = hashlib.sha256(((repo / rel).read_text() + '# edit\n').encode()).hexdigest()
                (repo / rel).write_text((repo / rel).read_text() + '# edit\n')
                self.assert_refused(params)

    def test_untracked_file_under_src_or_scripts_is_refused(self):
        for rel in ('src/vkgr/shadow.py', 'scripts/extra.py'):
            with self.subTest(rel), tempfile.TemporaryDirectory() as tmp:
                repo = self.make_repo(tmp)
                params = self.params(repo)
                (repo / rel).write_text('Y = 2\n')
                self.assert_refused(params)
                (repo / '.gitignore').write_text('')
                # an ignored byte-code cache is not an untracked source file
                (repo / rel).unlink()
                (repo / 'src/vkgr/__pycache__').mkdir()
                (repo / 'src/vkgr/__pycache__/mod.pyc').write_bytes(b'x')
                (repo / '.gitignore').write_text('__pycache__/\n')
                git(repo, 'add', '.gitignore')
                git(repo, 'commit', '-q', '-m', 'ignore caches')
                self.assertEqual(self.send(self.params(repo)).returncode, 0)

    def test_launcher_writes_nothing_into_the_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self.make_repo(tmp)
            before = git(repo, 'status', '--porcelain').stdout
            self.assertEqual(self.send(self.params(repo)).returncode, 0)
            self.assertEqual(git(repo, 'status', '--porcelain').stdout, before)

    def test_launcher_is_a_name_replaced_copy_of_the_feat03_launcher(self):
        old = (ROOT / 'scripts' / 'feat03_audit_launch.py').read_text()
        new = LAUNCHER.read_text()
        self.assertEqual(new, old.replace('feat03_audit_constants.py', 'feat04b_audit_constants.py').replace('FEAT-03 audit launcher', 'FEAT-04b audit launcher'))


if __name__ == '__main__':
    unittest.main()
