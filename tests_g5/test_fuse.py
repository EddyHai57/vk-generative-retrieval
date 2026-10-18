"""Synthetic tests for the fuse command (small hand-computed world; module constants are patched to the toy registry)."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from vkgr_g5 import fuse as fz  # noqa: E402

P_SHA = {42: 'a' * 64, 2024: 'b' * 64, 2025: 'c' * 64}
G_SHA = {42: 'd' * 64, 2024: 'e' * 64, 2025: 'f' * 64}
N = 6
TARGETS = [0, 500, 77, 40, 20, 9999]                                   # the last query is OOV (its target is outside every candidate list)
G42 = [[0], [7, 500], [77], list(range(900, 1000)), [20] + list(range(1000, 1098)), [1, 2, 3]]
HISTORY = [[1, 2], list(range(12)), list(range(50)), list(range(30)), [8, 9, 10], [11]]      # lengths 2, 12, 50, 30, 3, 1
META = [('supported', 'positive_gt_p90'), ('unsupported', 'unsupported0'), ('supported', 'positive_gt_p90'), ('supported', 'positive_gt_median_le_p90'),
        ('unsupported', 'positive_le_median'), ('oov', 'oov')]
OOV = [False] * 5 + [True]
CATALOG_DIGEST = 'cd' * 32


def digest(value):
    return hashlib.sha256(value).hexdigest()


class World:
    def __init__(self, root):
        self.root = Path(root)
        self.p = {42: [list(range(100)) for _ in range(N)], 2024: [[0] + list(range(101, 200)) if i == 0 else list(range(100, 200)) for i in range(N)], 2025: [list(range(200, 300)) for _ in range(N)]}
        self.g = {42: G42, 2024: [list(range(2000, 2100)) for _ in range(N)], 2025: [list(range(3000, 3100)) for _ in range(N)]}
        self.pop = [40] + list(range(300, 399))                      # seed 42's Pop contains query 3's target
        self.hits = {('P', s): sum(1 for t, r in zip(TARGETS, self.p[s]) if t in r[:50]) for s in self.p}
        self.hits.update({('G', s): sum(1 for t, r in zip(TARGETS, self.g[s]) if t in r[:50]) for s in self.g})
        self.write_all()

    def put(self, rel, value):
        path = self.root / rel; path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, sort_keys=True).encode(); path.write_bytes(data)
        return str(path), digest(data)

    def write_all(self):
        self.pexp, self.gasm = {}, {}
        for s in self.p:
            _, top_sha = self.put(f'p{s}/top100.json', self.p[s])
            record = {'schema': 'p-seqm1-export100-v1', 'status': 'complete', 'depth': 100, 'queries': N, 'seed': s, 'train_manifest_sha256': P_SHA[s], 'top100_sha256': top_sha,
                      'gate': {'first50_equal': True, 'mismatched_queries': 0, 'queries': N, 'compared_depth': 50}}
            self.pexp[s] = self.put(f'p{s}/export-manifest.json', record)
        rows_template = [{'ordinal': i, 'user_id': 100 + i, 'target': TARGETS[i], 'oov': OOV[i], 'history': HISTORY[i], 'support': META[i][0], 'popularity_group': META[i][1], 'user_group': 'gt20'} for i in range(N)]
        for s in self.g:
            rows = [{**r, 'items': self.g[s][i]} for i, r in enumerate(rows_template)]
            asm = {'schema': 'g5-g-assembly-v1', 'split': 'validation', 'run_manifest_sha256': G_SHA[s], 'seed': s, 'query_count': N, 'catalog_size': 2000, 'catalog_digest': CATALOG_DIGEST, 'rows': rows,
                   'checks': {'archive_replay_equal': True, 'slice_consistency_equal': True, 'metric_equals_best': True}, 'pop': {'ranking': self.pop, 'hits_at_50': 1},
                   'slice_statistics_manifest_sha256': 's' * 64, 'thresholds': [1, 3], 'item_groups': list(fz.fv.ITEM_GROUPS), 'user_groups': list(fz.fv.USER_GROUPS)}
            self.gasm[s] = self.put(f'g{s}/g-assembly.json', asm)
        meta = {'schema': 'g5-p-query-meta-v1', 'split': 'validation', 'query_count': N, 'catalog_size': 2000, 'catalog_digest': CATALOG_DIGEST, 'history_len': 50, 'manifests_checked': 3,
                'rows': [{'ordinal': i, 'user_id': 100 + i, 'target': TARGETS[i], 'oov': OOV[i], 'history': HISTORY[i], 'support_p': META[i][0]} for i in range(N)]}
        self.meta = self.put('meta/p-query-meta.json', meta)

    def edit(self, key, mutate):
        """Rewrite one pinned file with ``mutate(value)`` and refresh its pin (and, for top100, the pin inside the P export manifest)."""
        kind, seed = key
        if kind == 'meta':
            value = json.loads(Path(self.meta[0]).read_text()); mutate(value); self.meta = self.put('meta/p-query-meta.json', value)
        elif kind == 'g':
            value = json.loads(Path(self.gasm[seed][0]).read_text()); mutate(value); self.gasm[seed] = self.put(f'g{seed}/g-assembly.json', value)
        elif kind == 'pexp':
            value = json.loads(Path(self.pexp[seed][0]).read_text()); mutate(value); self.pexp[seed] = self.put(f'p{seed}/export-manifest.json', value)
        elif kind == 'top':
            lists = json.loads((Path(self.pexp[seed][0]).with_name('top100.json')).read_text()); mutate(lists)
            _, sha = self.put(f'p{seed}/top100.json', lists)
            self.edit(('pexp', seed), lambda rec: rec.__setitem__('top100_sha256', sha))

    def argv(self, out, *extra):
        argv = []
        for s in (42, 2024, 2025):
            argv += ['--p-export', f'{s}={self.pexp[s][0]}@{self.pexp[s][1]}', '--g-assembly', f'{s}={self.gasm[s][0]}@{self.gasm[s][1]}']
        return [*argv, '--p-query-meta', f'{self.meta[0]}@{self.meta[1]}', '--output-dir', str(out), '--split', 'validation', *extra]


@contextlib.contextmanager
def toy(world):
    with mock.patch.object(fz, 'P_MANIFEST_SHA', P_SHA), mock.patch.object(fz, 'G_MANIFEST_SHA', G_SHA), mock.patch.object(fz, 'QUERIES', N), \
            mock.patch.object(fz, 'EXPECT_HITS', world.hits):
        yield


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch('sys.stdout', out), mock.patch('sys.stderr', err):
        code = fz.main(argv)
    return code, out.getvalue(), err.getvalue()


class FuseCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.w = World(self.tmp.name); self.out = Path(self.tmp.name) / 'result'

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run_writes_results_with_hand_computed_cells_and_provenance(self):
        with toy(self.w):
            code, stdout, stderr = run_main(self.w.argv(self.out))
        self.assertEqual(code, 0, stderr)
        result = json.loads((self.out / 'fusion-results.json').read_text()); rows = result['fusion']['rows']
        self.assertEqual([rows[f'seed42:C{i}']['slices']['all']['hits'] for i in (1, 5)], [3, 4])                    # same hand-computed world as the core tests (Pop differs here)
        self.assertEqual(rows['seed42:C3']['slices']['all']['hits'], 1)                                           # only seed 42's Pop list holds target 40 in its top 50
        self.assertEqual([rows['seed42:C5']['slices'][f'history:{b}']['n'] for b in ('1-10', '11-25', '26-49', '50')], [3, 1, 1, 1])        # lengths 2, 12, 30, 50 and 3, 1
        self.assertEqual(rows['seed42:C5']['pool_size']['mean'], (100 + 101 + 100 + 200 + 198 + 100) / 6)          # G lists enter uncut (up to 100), P uncut
        self.assertEqual(rows['seed42:C5']['slices']['oov']['n'], 1); self.assertEqual(rows['seed42:C5']['oov_hits'], 0); self.assertFalse(rows['seed42:C5']['oov_nonzero_hits_anomaly'])
        self.assertEqual(rows['seed42:C5']['slices']['support:oov']['n'], 1)
        self.assertEqual(result['fusion']['localization']['C5'], {'low_support_queries': 4, 'net_in_set': 2, 'net_total': 1, 'localized': False})
        self.assertEqual(result['checks']['single_route_hits']['P42'], self.w.hits[('P', 42)])
        self.assertEqual(result['provenance']['fusion_v2_script_sha256'], hashlib.sha256(Path(fz.fv.__file__).read_bytes()).hexdigest())
        self.assertEqual(len(result['provenance']['fuse_script_sha256']), 64); self.assertEqual(result['inputs']['p_query_meta'], self.w.meta[1])
        self.assertTrue(result['generated_at_utc'].endswith('+00:00'))
        self.assertEqual(json.loads(stdout)['results_sha256'], hashlib.sha256((self.out / 'fusion-results.json').read_bytes()).hexdigest())
        md = (self.out / 'fusion-results.md').read_text()
        for needle in ('no verdict', 'seed42:C5', 'default_rng(0)', 'necessary condition'):
            self.assertIn(needle, md)
        self.assertNotIn('verdict:', md.lower()); self.assertNotIn('adopt', md.lower())

    def test_g_candidates_enter_uncut_and_are_checked_beyond_rank_50(self):
        with toy(self.w):
            code, _, err = run_main(self.w.argv(self.out)); self.assertEqual(code, 0, err)
            base = json.loads((self.out / 'fusion-results.json').read_text())['fusion']['rows']['seed42:C5']['pool_size']['mean']
        self.assertEqual(base, (100 + 101 + 100 + 200 + 198 + 100) / 6)                                          # truncating G to 50 would give a smaller pool

    def test_check_only_does_every_input_gate_and_runs_the_fusion_in_memory(self):
        with toy(self.w):
            code, stdout, _ = run_main(self.w.argv(self.out, '--check-only'))
        self.assertEqual(code, 0); got = json.loads(stdout)
        self.assertEqual((got['status'], got['written'], got['queries']), ('preflight_ok', False, N)); self.assertFalse(self.out.exists())
        self.assertGreater(got['fusion_serialized_bytes'], 0); self.assertEqual(got['rows'], 13); self.assertEqual(got['oov_nonzero_hits_anomaly_rows'], [])

    def test_non_finite_numbers_are_refused_in_preflight_and_output_and_preflight_size_is_the_real_one(self):
        real = fz.fv.run; seen = {}
        def spy(*a):
            seen['result'] = result = real(*a); return result
        with toy(self.w), mock.patch.object(fz.fv, 'run', spy):
            code, stdout, _ = run_main(self.w.argv(self.out, '--check-only'))
        self.assertEqual(code, 0); self.assertEqual(json.loads(stdout)['fusion_serialized_bytes'], len(json.dumps(seen['result'], sort_keys=True)))
        def nan_run(*a):
            result = real(*a); result['rows']['seed42:C5']['slices']['all']['R@50'] = float('nan'); return result
        for extra in (('--check-only',), ()):
            with toy(self.w), mock.patch.object(fz.fv, 'run', nan_run):
                code, stdout, err = run_main(self.w.argv(self.out, *extra))
            self.assertEqual(code, 2); self.assertIn('ValueError', err); self.assertFalse(self.out.exists()); self.assertEqual(stdout, '')

    def fails(self, edits, message, check_only=False):
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            for key, mutate in edits:
                w.edit(key, mutate)
            with toy(w):
                code, _, err = run_main(w.argv(Path(tmp) / 'r', *(['--check-only'] if check_only else [])))
            self.assertEqual(code, 2, message); self.assertIn(message, err); self.assertFalse((Path(tmp) / 'r').exists())

    def test_missing_catalog_digest_and_group_vocabularies_are_gated(self):
        self.fails([(('g', 2025), lambda v: v.pop('catalog_digest'))], 'catalog size or digest differs')
        self.fails([(('g', 2024), lambda v: v.__setitem__('item_groups', list(reversed(v['item_groups']))))], 'disagree on slice statistics or the Pop list (seed 2024)')
        self.fails([(('g', 2025), lambda v: v.__setitem__('user_groups', list(reversed(v['user_groups']))))], 'disagree on slice statistics or the Pop list (seed 2025)')

    def test_oov_anomaly_rows_are_reported_when_a_route_hits_an_oov_query(self):
        self.w.p[2025][5] = [TARGETS[5]] + list(range(200, 299))                                 # a P route now holds the OOV query's target in its top 50
        self.w.hits[('P', 2025)] = sum(1 for t, r in zip(TARGETS, self.w.p[2025]) if t in r[:50]); self.w.write_all()
        with toy(self.w):
            code, stdout, err = run_main(self.w.argv(self.out, '--check-only'))
        self.assertEqual(code, 0, err); rows = json.loads(stdout)['oov_nonzero_hits_anomaly_rows']
        self.assertTrue(rows and all(isinstance(r, str) for r in rows)); self.assertTrue(any('seed2025' in r or '2025' in r for r in rows))

    def test_alignment_and_ordinals_fail_closed(self):
        self.fails([(('g', 2024), lambda v: v['rows'][2].__setitem__('history', [4, 5]))], 'query alignment failed at ordinal 2')
        self.fails([(('meta', None), lambda v: v['rows'][1].__setitem__('target', 501))], 'query alignment failed at ordinal 1')
        self.fails([(('g', 42), lambda v: v['rows'][5].__setitem__('oov', False))], 'query alignment failed at ordinal 5')
        self.fails([(('meta', None), lambda v: v['rows'][3].__setitem__('ordinal', 9))], 'query alignment failed at ordinal 3')
        self.fails([(('g', 2025), lambda v: v['rows'][4].__setitem__('ordinal', 9))], 'query alignment failed at ordinal 4')
        self.fails([(('g', 42), lambda v: v['rows'].pop())], 'query count differs between P metadata and G assembly')

    def test_p_export_manifest_and_candidate_file_gates(self):
        message = 'P export manifest not acceptable'
        for field, value in (('status', 'running'), ('schema', 'other'), ('queries', N - 1), ('seed', 7), ('depth', 99)):
            self.fails([(('pexp', 2024), lambda v, f=field, x=value: v.__setitem__(f, x))], message)
        self.fails([(('pexp', 42), lambda v: v['gate'].__setitem__('first50_equal', False))], message)
        self.fails([(('pexp', 42), lambda v: v['gate'].__setitem__('mismatched_queries', 1))], message)
        self.fails([(('pexp', 2025), lambda v: v.__setitem__('train_manifest_sha256', '0' * 64))], 'not bound to the accepted P train manifest')
        self.fails([(('pexp', 42), lambda v: v.__setitem__('top100_sha256', '9' * 64))], 'input hash mismatch: top100.json')
        self.fails([(('top', 2024), lambda v: v.__setitem__(3, list(range(99))))], 'wrong shape')
        self.fails([(('top', 42), lambda v: v.__setitem__(1, list(range(500, 600))))], 'single-route hits differ from the accepted runs')

    def test_g_assembly_and_metadata_gates(self):
        message = 'not acceptable or not bound to the accepted G run'
        for field, value in (('schema', 'other'), ('split', 'train'), ('run_manifest_sha256', '1' * 64), ('seed', 42), ('query_count', N + 1)):
            self.fails([(('g', 2024), lambda v, f=field, x=value: v.__setitem__(f, x))], message)
        self.fails([(('g', 2025), lambda v: v['checks'].__setitem__('slice_consistency_equal', False))], 'G assembly not acceptable')
        for field, value in (('schema', 'other'), ('split', 'train'), ('query_count', N + 1), ('manifests_checked', 1), ('history_len', 40)):
            self.fails([(('meta', None), lambda v, f=field, x=value: v.__setitem__(f, x))], 'P query metadata not acceptable')
        self.fails([(('g', 42), lambda v: v['rows'][0].__setitem__('items', [999]))], 'single-route hits differ from the accepted runs')
        self.fails([(('g', 2025), lambda v: v.__setitem__('catalog_size', 1999))], 'catalog size or digest differs')
        self.fails([(('g', 2025), lambda v: v.__setitem__('catalog_digest', '0' * 64))], 'catalog size or digest differs')

    def test_g_assemblies_must_agree_on_statistics_pop_and_labels(self):
        self.fails([(('g', 2024), lambda v: v.__setitem__('slice_statistics_manifest_sha256', 'x' * 64))], 'disagree on slice statistics or the Pop list (seed 2024)')
        self.fails([(('g', 2025), lambda v: v.__setitem__('thresholds', [2, 3]))], 'disagree on slice statistics or the Pop list (seed 2025)')
        self.fails([(('g', 2025), lambda v: v['pop'].__setitem__('ranking', list(reversed(v['pop']['ranking']))))], 'disagree on slice statistics or the Pop list (seed 2025)')
        self.fails([(('g', 2024), lambda v: v['rows'][1].__setitem__('support', 'supported'))], 'disagree on slice labels (seed 2024)')
        self.fails([(('g', 2024), lambda v: v['rows'][1].__setitem__('popularity_group', 'positive_gt_p90'))], 'disagree on slice labels (seed 2024)')
        self.fails([(('g', 2025), lambda v: v['rows'][4].__setitem__('user_group', '1'))], 'disagree on slice labels (seed 2025)')

    def test_every_pinned_file_is_hash_checked(self):
        for name in ('meta', 'g', 'pexp', 'top'):
            with tempfile.TemporaryDirectory() as tmp:
                w = World(tmp)
                path = {'meta': Path(w.meta[0]), 'g': Path(w.gasm[2025][0]), 'pexp': Path(w.pexp[2024][0]), 'top': Path(w.pexp[42][0]).with_name('top100.json')}[name]
                path.write_bytes(path.read_bytes() + b' ')
                with toy(w):
                    code, _, err = run_main(w.argv(Path(tmp) / 'r'))
                self.assertEqual(code, 2, name); self.assertIn('input hash mismatch', err, name)

    def test_symlinks_and_non_regular_files_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp); real = Path(w.gasm[42][0]); link = real.with_name('link.json'); link.symlink_to(real)
            argv = w.argv(Path(tmp) / 'r'); argv[argv.index('--g-assembly') + 1] = f'42={link}@{w.gasm[42][1]}'
            with toy(w):
                code, _, err = run_main(argv)
            self.assertEqual(code, 2); self.assertIn('input hash mismatch', err)
            argv = w.argv(Path(tmp) / 'r2'); argv[argv.index('--g-assembly') + 1] = f'42={real.parent}@{w.gasm[42][1]}'
            with toy(w):
                code, _, err = run_main(argv)
            self.assertEqual(code, 2); self.assertIn('input hash mismatch', err)

    def test_arguments_split_seeds_and_output_dir(self):
        with toy(self.w):
            argv = self.w.argv(self.out); idx = argv.index('--g-assembly'); del argv[idx:idx + 2]
            code, _, err = run_main(argv); self.assertEqual(code, 2); self.assertIn('all three seeds', err)
            argv = self.w.argv(self.out); argv[argv.index('--p-query-meta') + 1] = 'nohash'
            self.assertEqual(run_main(argv)[0], 2)
            with self.assertRaises(SystemExit):
                fz.main(self.w.argv(self.out)[:-1] + ['test'])
            self.out.mkdir()
            code, _, err = run_main(self.w.argv(self.out)); self.assertEqual(code, 2); self.assertIn('must not exist', err)

    def test_held_out_component_rule_applies_to_every_path(self):
        with toy(self.w):
            for name in ('week_26', 'test', 'a-test-b', 'x.week-26.y'):
                code, _, err = run_main(self.w.argv(Path(self.tmp.name) / name / 'r')); self.assertEqual(code, 2, name); self.assertIn('held-out', err)
            self.assertEqual(run_main(self.w.argv(Path(self.tmp.name) / 'latest-results'))[0], 0)       # 'latest' contains 'test' but is not a component match
        for which in ('meta', 'g', 'p'):
            with tempfile.TemporaryDirectory() as tmp:
                w = World(Path(tmp) / 'week_26')
                with toy(w):
                    code, _, err = run_main(w.argv(Path(tmp) / 'out'))
                self.assertEqual(code, 2, which); self.assertIn('held-out', err)
        for which in ('meta', 'g', 'p'):                                # one path at a time: only that input lives below a held-out-named directory
            with tempfile.TemporaryDirectory() as tmp:
                w = World(tmp); bad = Path(tmp) / 'test'; bad.mkdir()
                argv = w.argv(Path(tmp) / 'out')
                if which == 'meta':
                    target = bad / 'meta.json'; target.write_bytes(Path(w.meta[0]).read_bytes()); argv[argv.index('--p-query-meta') + 1] = f'{target}@{w.meta[1]}'
                elif which == 'g':
                    target = bad / 'g.json'; target.write_bytes(Path(w.gasm[42][0]).read_bytes()); argv[argv.index('--g-assembly') + 1] = f'42={target}@{w.gasm[42][1]}'
                else:
                    target = bad / 'export-manifest.json'; target.write_bytes(Path(w.pexp[42][0]).read_bytes()); argv[argv.index('--p-export') + 1] = f'42={target}@{w.pexp[42][1]}'
                with toy(w):
                    code, _, err = run_main(argv)
                self.assertEqual(code, 2, which); self.assertIn('held-out', err, which)

    def test_p_vs_g_support_label_disagreement_is_counted_not_fatal(self):
        self.w.edit(('meta', None), lambda v: v['rows'][0].__setitem__('support_p', 'unsupported'))
        with toy(self.w):
            code, stdout, stderr = run_main(self.w.argv(self.out, '--check-only'))
        self.assertEqual(code, 0, stderr); self.assertEqual(json.loads(stdout)['checks']['support_label_mismatch_p_vs_g'], 1)


class UnitTests(unittest.TestCase):
    def test_pair_and_seeded_parsing(self):
        self.assertEqual(fz.pair('/a/b@' + 'a' * 64), ('/a/b', 'a' * 64))
        for bad in ('x', '@' + 'a' * 64, '/a@short'):
            with self.assertRaises(ValueError):
                fz.pair(bad)
        good = [f'{s}=/p{s}@' + 'a' * 64 for s in (42, 2024, 2025)]
        self.assertEqual(sorted(fz.seeded(good)), [42, 2024, 2025])
        for bad in (good[:2], good + [good[0]], ['7=/p@' + 'a' * 64] + good[1:]):
            with self.assertRaises(ValueError):
                fz.seeded(bad)

    def test_registered_constants_match_the_accepted_tables(self):
        self.assertEqual(fz.EXPECT_HITS, {('P', 42): 466, ('P', 2024): 481, ('P', 2025): 471, ('G', 42): 388, ('G', 2024): 376, ('G', 2025): 384}); self.assertEqual(fz.QUERIES, 2520)
        self.assertEqual(fz.P_MANIFEST_SHA[42][:8], 'bdb0e39d'); self.assertEqual(fz.G_MANIFEST_SHA[2025][:8], '86767e4d')


if __name__ == '__main__':
    unittest.main()
