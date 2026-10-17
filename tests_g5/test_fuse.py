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
TARGETS = [0, 500, 77, 40, 20]
G42 = [[0], [7, 500], [77], list(range(900, 1000)), [20] + list(range(1000, 1098))]
HISTORY = [[1, 2], list(range(12)), list(range(50)), list(range(30)), [8, 9, 10]]      # lengths 2, 12, 50, 30, 3
META = [('supported', 'positive_gt_p90', 10), ('unsupported', 'unsupported0', 3), ('supported', 'positive_gt_p90', 50), ('supported', 'positive_gt_median_le_p90', 30), ('unsupported', 'positive_le_median', 20)]


def digest(value):
    return hashlib.sha256(value).hexdigest()


class World:
    def __init__(self, root):
        self.root = Path(root); n = 5
        self.p = {42: [list(range(100)) for _ in range(n)], 2024: [[0] + list(range(101, 200)) if i == 0 else list(range(100, 200)) for i in range(n)], 2025: [list(range(200, 300)) for _ in range(n)]}
        self.g = {42: G42, 2024: [list(range(2000, 2100)) for _ in range(n)], 2025: [list(range(3000, 3100)) for _ in range(n)]}
        self.pop = [40] + list(range(300, 399))                      # seed 42's Pop contains query 3's target
        self.pop_other = list(reversed(self.pop))                    # the other seeds' assemblies carry a different (reversed) Pop list
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
            top_path, top_sha = self.put(f'p{s}/top100.json', self.p[s])
            record = {'schema': 'p-seqm1-export100-v1', 'status': 'complete', 'depth': 100, 'queries': 5, 'seed': s, 'train_manifest_sha256': P_SHA[s], 'top100_sha256': top_sha,
                      'gate': {'first50_equal': True, 'mismatched_queries': 0, 'queries': 5, 'compared_depth': 50}}
            self.pexp[s] = self.put(f'p{s}/export-manifest.json', record)
        rows_template = [{'ordinal': i, 'user_id': 100 + i, 'target': TARGETS[i], 'oov': False, 'history': HISTORY[i], 'support': META[i][0], 'popularity_group': META[i][1], 'user_group': 'gt20'} for i in range(5)]
        for s in self.g:
            rows = [{**r, 'items': self.g[s][i]} for i, r in enumerate(rows_template)]
            asm = {'schema': 'g5-g-assembly-v1', 'split': 'validation', 'run_manifest_sha256': G_SHA[s], 'seed': s, 'query_count': 5, 'catalog_size': 2000, 'rows': rows,
                   'checks': {'archive_replay_equal': True, 'slice_consistency_equal': True, 'metric_equals_best': True}, 'pop': {'ranking': self.pop if s == 42 else self.pop_other, 'hits_at_50': 0}, 'slice_statistics_manifest_sha256': 's' * 64}
            self.gasm[s] = self.put(f'g{s}/g-assembly.json', asm)
        meta = {'schema': 'g5-p-query-meta-v1', 'split': 'validation', 'query_count': 5, 'catalog_size': 2000,
                'rows': [{'ordinal': i, 'user_id': 100 + i, 'target': TARGETS[i], 'oov': False, 'history': HISTORY[i], 'support_p': META[i][0]} for i in range(5)]}
        self.meta = self.put('meta/p-query-meta.json', meta)

    def argv(self, out, *extra):
        argv = []
        for s in (42, 2024, 2025):
            argv += ['--p-export', f'{s}={self.pexp[s][0]}@{self.pexp[s][1]}', '--g-assembly', f'{s}={self.gasm[s][0]}@{self.gasm[s][1]}']
        return [*argv, '--p-query-meta', f'{self.meta[0]}@{self.meta[1]}', '--output-dir', str(out), '--split', 'validation', *extra]


@contextlib.contextmanager
def toy(world):
    with mock.patch.object(fz, 'P_MANIFEST_SHA', P_SHA), mock.patch.object(fz, 'G_MANIFEST_SHA', G_SHA), mock.patch.object(fz, 'QUERIES', 5), \
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
        result = json.loads((self.out / 'fusion-results.json').read_text())
        rows = result['fusion']['rows']
        self.assertEqual([rows[f'seed42:C{i}']['slices']['all']['hits'] for i in (1, 5)], [3, 4])                    # same hand-computed world as the core tests (Pop differs here)
        c3 = rows['seed42:C3']['slices']['all']['hits']; self.assertEqual(c3, 1)                      # only seed 42's Pop list holds target 40 in its top 50
        self.assertEqual([rows['seed42:C5']['slices'][f'history:{b}']['n'] for b in ('1-10', '11-25', '26-49', '50')], [2, 1, 1, 1])
        self.assertEqual(result['fusion']['localization']['C5']['net_total'], rows['seed42:C5']['slices']['all']['hits'] - rows['seed42:C1']['slices']['all']['hits'])
        self.assertEqual(result['checks']['single_route_hits']['P42'], self.w.hits[('P', 42)])
        self.assertEqual(result['provenance']['fusion_v2_script_sha256'], hashlib.sha256(Path(fz.fv.__file__).read_bytes()).hexdigest())
        self.assertEqual(len(result['provenance']['fuse_script_sha256']), 64); self.assertEqual(result['inputs']['p_query_meta'], self.w.meta[1])
        self.assertEqual(json.loads(stdout)['results_sha256'], hashlib.sha256((self.out / 'fusion-results.json').read_bytes()).hexdigest())
        md = (self.out / 'fusion-results.md').read_text()
        for needle in ('no verdict', 'seed42:C5', 'default_rng(0)', 'necessary condition'):
            self.assertIn(needle, md)
        self.assertNotIn('verdict:', md.lower()); self.assertNotIn('adopt', md.lower())          # mechanical report only

    def test_check_only_does_every_input_gate_but_writes_nothing(self):
        with toy(self.w):
            code, stdout, _ = run_main(self.w.argv(self.out, '--check-only'))
        self.assertEqual(code, 0); got = json.loads(stdout); self.assertEqual((got['status'], got['written'], got['queries']), ('preflight_ok', False, 5)); self.assertFalse(self.out.exists())

    def test_misalignment_wrong_hits_and_bindings_fail_closed(self):
        def fails(mutator, message):
            with tempfile.TemporaryDirectory() as tmp:
                w = World(tmp); mutator(w); w.write_all_after = None
                with toy(w):
                    code, _, err = run_main(w.argv(Path(tmp) / 'r'))
                self.assertEqual(code, 2, message); self.assertIn(message, err); self.assertFalse((Path(tmp) / 'r').exists())
        def history(w):
            asm = json.loads(Path(w.gasm[2024][0]).read_text()); asm['rows'][2]['history'] = [4, 5]; w.gasm[2024] = w.put('g2024/g-assembly.json', asm)
        fails(history, 'query alignment failed at ordinal 2')
        def target(w):
            meta = json.loads(Path(w.meta[0]).read_text()); meta['rows'][1]['target'] = 501; w.meta = w.put('meta/p-query-meta.json', meta)
        fails(target, 'query alignment failed at ordinal 1')
        def wrong_hits(w):
            w.p[42][1] = list(range(500, 600)); w.write_all_after = True
            top_path, top_sha = w.put('p42/top100.json', w.p[42]); rec = json.loads(Path(w.pexp[42][0]).read_text()); rec['top100_sha256'] = top_sha; w.pexp[42] = w.put('p42/export-manifest.json', rec)
        fails(wrong_hits, 'single-route hits differ from the accepted runs')
        def unbound_p(w):
            rec = json.loads(Path(w.pexp[2025][0]).read_text()); rec['train_manifest_sha256'] = '0' * 64; w.pexp[2025] = w.put('p2025/export-manifest.json', rec)
        fails(unbound_p, 'not bound to the accepted P train manifest')
        def gate_false(w):
            rec = json.loads(Path(w.pexp[42][0]).read_text()); rec['gate']['first50_equal'] = False; w.pexp[42] = w.put('p42/export-manifest.json', rec)
        fails(gate_false, 'P export manifest not acceptable')
        def unbound_g(w):
            asm = json.loads(Path(w.gasm[42][0]).read_text()); asm['run_manifest_sha256'] = '1' * 64; w.gasm[42] = w.put('g42/g-assembly.json', asm)
        fails(unbound_g, 'not acceptable or not bound to the accepted G run')
        def failed_check(w):
            asm = json.loads(Path(w.gasm[2025][0]).read_text()); asm['checks']['slice_consistency_equal'] = False; w.gasm[2025] = w.put('g2025/g-assembly.json', asm)
        fails(failed_check, 'G assembly not acceptable')
        def mismatched_count(w):
            rec = json.loads(Path(w.pexp[42][0]).read_text()); rec['gate']['mismatched_queries'] = 1; w.pexp[42] = w.put('p42/export-manifest.json', rec)
        fails(mismatched_count, 'P export manifest not acceptable')
        def shallow(w):
            rec = json.loads(Path(w.pexp[2024][0]).read_text()); rec['depth'] = 99; w.pexp[2024] = w.put('p2024/export-manifest.json', rec)
        fails(shallow, 'P export manifest not acceptable')
        def wrong_seed_field(w):
            asm = json.loads(Path(w.gasm[2024][0]).read_text()); asm['seed'] = 42; w.gasm[2024] = w.put('g2024/g-assembly.json', asm)
        fails(wrong_seed_field, 'not acceptable or not bound to the accepted G run')
        def g_hits(w):
            asm = json.loads(Path(w.gasm[42][0]).read_text()); asm['rows'][0]['items'] = [999]; w.gasm[42] = w.put('g42/g-assembly.json', asm)
        fails(g_hits, 'single-route hits differ from the accepted runs')
        def catalog(w):
            asm = json.loads(Path(w.gasm[2025][0]).read_text()); asm['catalog_size'] = 1999; w.gasm[2025] = w.put('g2025/g-assembly.json', asm)
        fails(catalog, 'catalog size differs')
        def short_list(w):
            w.p[2024][3] = list(range(99)); top = w.put('p2024/top100.json', w.p[2024]); rec = json.loads(Path(w.pexp[2024][0]).read_text()); rec['top100_sha256'] = top[1]; w.pexp[2024] = w.put('p2024/export-manifest.json', rec)
        fails(short_list, 'wrong shape')

    def test_sha_pins_and_arguments_are_enforced(self):
        with toy(self.w):
            path = Path(self.w.gasm[42][0]); path.write_bytes(path.read_bytes() + b' ')
            code, _, err = run_main(self.w.argv(self.out)); self.assertEqual(code, 2); self.assertIn('input hash mismatch', err)
        with toy(self.w):
            argv = self.w.argv(self.out); idx = argv.index('--g-assembly'); del argv[idx:idx + 2]
            code, _, err = run_main(argv); self.assertEqual(code, 2); self.assertIn('all three seeds', err)
            argv = self.w.argv(self.out); argv[argv.index('--p-query-meta') + 1] = 'nohash'
            self.assertEqual(run_main(argv)[0], 2)
            with self.assertRaises(SystemExit):
                fz.main(self.w.argv(self.out)[:-1] + ['test'])

    def test_output_dir_and_held_out_paths_are_refused(self):
        with toy(self.w):
            self.out.mkdir()
            code, _, err = run_main(self.w.argv(self.out)); self.assertEqual(code, 2); self.assertIn('must not exist', err)
            for name in ('week_26', 'test', 'a-test-b', 'x.week-26.y'):
                bad = Path(self.tmp.name) / name / 'r'
                code, _, err = run_main(self.w.argv(bad)); self.assertEqual(code, 2, name); self.assertIn('held-out', err)
            self.assertEqual(run_main(self.w.argv(Path(self.tmp.name) / 'latest-results'))[0], 0)       # 'latest' contains 'test' but is not a component match

    def test_p_vs_g_support_label_disagreement_is_counted_not_fatal(self):
        meta = json.loads(Path(self.w.meta[0]).read_text()); meta['rows'][0]['support_p'] = 'unsupported'; self.w.meta = self.w.put('meta/p-query-meta.json', meta)
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
