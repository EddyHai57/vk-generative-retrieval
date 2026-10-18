"""Tests for the G-side assembly. Pure tests always run; the integration tests need the G code base (VKGR_G5_G_ROOT) and must run in
their own interpreter (the P and G code bases both provide a package named ``vkgr``)."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
G_ROOT = os.environ.get('VKGR_G5_G_ROOT')
sys.path.insert(0, str(ROOT / 'src'))
from vkgr_g5 import assemble_g as ag  # noqa: E402

ITEM_GROUPS = ('unsupported0', 'positive_le_median', 'positive_gt_median_le_p90', 'positive_gt_p90')
USER_GROUPS = ('1', '2-5', '6-20', 'gt20')
REAL_TMP = os.path.realpath(tempfile.gettempdir())      # resolved: the assembly resolves its paths, so a symlinked temp root would defeat the nesting checks


def fake_stats():
    return SimpleNamespace(item_counts={1: 0, 2: 3, 3: 5, 4: 20, 5: 3}, user_counts={10: 1, 11: 5, 12: 6, 13: 20, 14: 21}, thresholds=(3, 10), manifest_sha256='s' * 64)


def fake_queries():
    base = lambda u, t, o=False, h=3: {'user_id': u, 'target': t, 'oov': o, 'history': list(range(h))}
    return [base(10, 1), base(11, 2), base(12, 3), base(13, 4), base(14, 5, h=50), base(10, 999, o=True, h=1)]


class LabelTests(unittest.TestCase):
    def test_labels_follow_the_aggregate_items_rule_exactly(self):
        got = ag.labels(fake_stats(), fake_queries(), ITEM_GROUPS, USER_GROUPS)
        self.assertEqual(got['popularity'], ['unsupported0', 'positive_le_median', 'positive_gt_median_le_p90', 'positive_gt_p90', 'positive_le_median', 'oov'])
        self.assertEqual(got['support'], ['unsupported', 'supported', 'supported', 'supported', 'supported', 'oov'])
        self.assertEqual(got['user_history'], ['1', '2-5', '6-20', '6-20', 'gt20', '1'])
        self.assertEqual(got['truncated_history'], ['3', '3', '3', '3', '50', '1'])

    def test_history_gate(self):
        bad = fake_queries(); bad[0]['history'] = []
        with self.assertRaisesRegex(ValueError, 'bounded eligible history'):
            ag.labels(fake_stats(), bad, ITEM_GROUPS, USER_GROUPS)
        bad[0]['history'] = list(range(51))
        with self.assertRaises(ValueError):
            ag.labels(fake_stats(), bad, ITEM_GROUPS, USER_GROUPS)

    def test_pop_ranking_is_count_desc_then_item_asc_and_needs_depth(self):
        stats = SimpleNamespace(item_counts={7: 5, 3: 5, 9: 1, 1: 0, 2: 9}); self.assertEqual(ag.pop_ranking(stats, [1, 2, 3, 7, 9], k=4), [2, 3, 7, 9])
        with self.assertRaisesRegex(ValueError, 'Pop depth'):
            ag.pop_ranking(stats, [1, 2, 3], k=4)


def fake_metrics(preds, queries, catalog, ks):
    return {'all': {'n': len(preds), 'hits': sum(1 for p, q in zip(preds, queries) if q['target'] in p)}}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.queries = fake_queries(); self.items = [[1, 2], [2], [3], [4, 9], [5], [1]]; self.catalog = [1, 2, 3, 4, 5, 9]
        self.assigned = ag.labels(fake_stats(), self.queries, ITEM_GROUPS, USER_GROUPS)
        spec = (('popularity', ITEM_GROUPS + ('oov',)), ('user_history', USER_GROUPS), ('support', ('supported', 'unsupported', 'oov')), ('truncated_history', tuple(str(i) for i in range(1, 51))))
        self.aggregate = {s: ag.regroup_metrics(self.items, self.queries, self.catalog, self.assigned[s], names, fake_metrics) for s, names in spec}
        self.cb = lambda v: json.dumps(v, sort_keys=True).encode()

    def gate(self, aggregate=None, items=None):
        return ag.consistency_gate(items or self.items, self.queries, self.catalog, self.assigned, aggregate or self.aggregate, ITEM_GROUPS, USER_GROUPS, fake_metrics, self.cb)

    def test_consistent_aggregate_passes(self):
        self.assertTrue(self.gate())

    def test_any_section_mismatch_fails_closed(self):
        for section in self.aggregate:
            bad = json.loads(json.dumps(self.aggregate)); name = next(iter(bad[section])); bad[section][name]['n'] += 1
            with self.assertRaisesRegex(ValueError, section):
                self.gate(bad)
        with self.assertRaises(ValueError):
            self.gate(items=[[2, 1], [2], [3], [4, 9], [1], [1]])      # a candidate change that alters a slice hit count

    def test_empty_slice_uses_the_empty_metrics_copy(self):
        got = ag.regroup_metrics(self.items, self.queries, self.catalog, self.assigned['popularity'], ('nonexistent',), fake_metrics)
        self.assertEqual(got, {'nonexistent': {'n': 0, 'hits': 0}})


CB = lambda v: json.dumps(v, sort_keys=True).encode()
EV = SimpleNamespace(ITEM_GROUPS=ITEM_GROUPS, USER_GROUPS=USER_GROUPS)


def core_world(stats, queries, items, catalog, sha='x'):
    """A fake run whose archive aggregate is exactly what the real aggregation rule would produce (built with the module's own regroup)."""
    assigned = ag.labels(stats, queries, ITEM_GROUPS, USER_GROUPS)
    spec = (('popularity', ITEM_GROUPS + ('oov',)), ('user_history', USER_GROUPS), ('support', ('supported', 'unsupported', 'oov')), ('truncated_history', tuple(str(i) for i in range(1, 51))))
    aggregate = {s: ag.regroup_metrics(items, queries, catalog, assigned[s], names, fake_metrics) for s, names in spec}
    aggregate['metrics'] = {'all_eligible': {'recall@50': 0.5, 'ndcg@50': 0.25}}
    record = {'epoch': 1, 'evaluation': {'kind': 'query_evaluation', 'manifest_sha256': 'm' * 64, 'computation_sha256': 'c' * 64}, 'checkpoint_sha256': sha, 'metrics': aggregate}
    value = {'best': {'epoch': 1, 'metric': [0.5, 0.25], 'sha256': sha}, 'epochs': [record]}
    rows = [{'ordinal': i, 'items': list(it)} for i, it in enumerate(items)]
    reader = lambda kind, run, rec, ctx, toks: (aggregate, 'c' * 64, rows)
    context = SimpleNamespace(queries=queries, catalog=catalog, slice_definition=stats)
    return value, context, reader


def run_core(value, context, reader, **kw):
    return ag.assemble_core(value, '/unused', context, None, reader, EV, fake_metrics, CB, **kw)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.stats = fake_stats(); self.queries = fake_queries(); self.catalog = [1, 2, 3, 4, 5]
        self.items = [[1, 2], [2], [3], [4, 5], [5], [1]]

    def world(self, **kw):
        return core_world(self.stats, self.queries, self.items, self.catalog, **kw)

    def test_rows_carry_query_fields_and_labels_and_catalog_digest(self):
        value, context, reader = self.world()
        rec = run_core(value, context, reader, pop_k=3, digest=lambda c: 'd%d' % len(c))
        self.assertEqual(rec['catalog_digest'], 'd5'); self.assertEqual(rec['catalog_size'], 5); self.assertEqual(rec['query_count'], 6)
        for row, q, it in zip(rec['rows'], self.queries, self.items):
            self.assertEqual((row['user_id'], row['target'], row['oov'], row['history'], row['items']), (q['user_id'], q['target'], q['oov'], q['history'], it))
        self.assertEqual([r['support'] for r in rec['rows']], ['unsupported', 'supported', 'supported', 'supported', 'supported', 'oov'])
        self.assertEqual([r['popularity_group'] for r in rec['rows']][5], 'oov')
        self.assertEqual([r['user_group'] for r in rec['rows']], ['1', '2-5', '6-20', '6-20', 'gt20', '1'])
        self.assertEqual(len(rec['rows'][4]['history']), 50)
        self.assertIsNone(run_core(*self.world(), pop_k=3)['catalog_digest'])

    def test_g_short_queries_boundary_is_100(self):
        self.items = [list(range(99)), list(range(100)), list(range(101)), [1], [1], [1]]
        value, context, reader = self.world()
        self.assertEqual(run_core(value, context, reader, pop_k=3)['g_short_queries'], 4)      # 99, and the three 1-item rows; 100 and 101 are not short

    def test_pop_hits_at_50_counts_positions_50_not_51(self):
        stats = SimpleNamespace(item_counts={i: 100 - i for i in range(1, 61)}, user_counts={10: 1, 11: 1}, thresholds=(3, 10), manifest_sha256='s' * 64)
        catalog = list(range(1, 61)); base = lambda u, t: {'user_id': u, 'target': t, 'oov': False, 'history': [1]}
        queries = [base(10, 50), base(11, 51), base(10, 1)]; items = [[1], [1], [1]]
        value, context, reader = core_world(stats, queries, items, catalog)
        rec = run_core(value, context, reader, pop_k=60)
        self.assertEqual(rec['pop']['ranking'][:3], [1, 2, 3]); self.assertEqual(rec['pop']['hits_at_50'], 2)      # targets 50 and 1 hit; 51 does not

    def test_best_hash_epoch_kind_and_order_gates(self):
        import copy
        value, context, reader = self.world()
        for mutate, pattern in (
                (lambda v: v['best'].__setitem__('sha256', 'y'), 'best checkpoint hash'),
                (lambda v: v['best'].__setitem__('epoch', True), 'best epoch outside'),
                (lambda v: v['best'].__setitem__('epoch', 0), 'best epoch outside'),
                (lambda v: v['best'].__setitem__('epoch', 2), 'best epoch outside'),
                (lambda v: v['epochs'][0].__setitem__('epoch', 2), 'query_evaluation archive'),
                (lambda v: v['epochs'][0]['evaluation'].__setitem__('kind', 'l100_query_evaluation'), 'query_evaluation archive'),
                (lambda v: v['best'].__setitem__('metric', [0.5, 0.26]), 'best metric'),
                (lambda v: v['epochs'][0]['evaluation'].__setitem__('computation_sha256', 'd' * 64), 'archive replay')):
            bad = copy.deepcopy(value); mutate(bad)
            with self.assertRaisesRegex(ValueError, pattern):
                run_core(bad, context, reader, pop_k=3)
        run_core(value, context, reader, pop_k=3)
        short = lambda *a: (reader(*a)[0], 'c' * 64, reader(*a)[2][:-1])
        with self.assertRaisesRegex(ValueError, 'ordinal order'):
            run_core(value, context, short, pop_k=3)
        swapped = lambda *a: (reader(*a)[0], 'c' * 64, list(reversed(reader(*a)[2])))
        with self.assertRaisesRegex(ValueError, 'ordinal order'):
            run_core(value, context, swapped, pop_k=3)


class CliTests(unittest.TestCase):
    def test_argument_and_path_gates(self):
        base = ['--prepared', 'p', '--prepared-sha', 'a', '--history-manifest', 'h', '--history-sha', 'b', '--sid-manifest', 's', '--sid-sha', 'c', '--content', 'c', '--content-sha', 'd',
                '--input-manifest', 'i', '--input-sha', 'e', '--vectors', 'v', '--vectors-sha', 'f', '--manifest', 'm', '--manifest-sha', 'g', '--scratch', '/tmp/g5-asm-scratch-x']
        with self.assertRaises(SystemExit):
            ag.main(['--line-root', '/x', *base, '--output', 'o', '--split', 'test'])
        err = io.StringIO()
        with mock.patch('sys.stderr', err):
            code = ag.main(['--line-root', '/nonexistent-root', *base, '--output', '/tmp/g5-asm-out-x.json', '--split', 'validation'])
        self.assertEqual(code, 2)


@unittest.skipUnless(G_ROOT and (Path(G_ROOT) / 'src/vkgr/coll_sid_a1_run.py').is_file(), 'VKGR_G5_G_ROOT (G code base) required')
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(G_ROOT) / 'src'))
        import vkgr
        if Path(G_ROOT).resolve() not in Path(vkgr.__file__).resolve().parents:
            raise unittest.SkipTest('vkgr already imported from another code base; run this module alone')

    def build_world(self, tmp):
        import numpy as np
        import torch
        from types import MappingProxyType
        from vkgr import coll_sid_a1_run as g, generative_experiment as ge, sid_coll_run as fit, sid_coll_v1 as coll, sid_tok_audit as au, sid_v2
        from vkgr.data_audit import canonical_bytes, file_sha256
        from vkgr.generative_data_v2 import ListExamples
        from vkgr.generative_evaluation import SliceStatistics
        from vkgr.generative_foundation import prepare_fixture
        torch.set_num_threads(1)
        fixture = json.loads((Path(G_ROOT) / 'tests/fixtures/g2_synthetic.json').read_text()); prepared = prepare_fixture(fixture)
        ids = np.array(fixture['catalog'], dtype='uint32'); n = len(ids)
        content = np.random.default_rng(42).normal(size=(n, 64)).astype('float32'); content /= np.linalg.norm(content, axis=1, keepdims=True)
        manifest = {'schema': 'coll-input-v1', 'status': 'accepted', 'variant': 'C1', 'dimension': 64, 'engineering': True, 'catalog_sha256': sid_v2.catalog_digest(ids),
                    'vectors_sha256': au.feature_digest(content), 'content_feature_sha256': au.feature_digest(content), 'train_sequence_sha256': 'a' * 64, 'script_sha256': 'b' * 64,
                    'commit': 'c' * 40, 'checkpoint_sha256': None, 'source_files': {'src/vkgr/sid_coll_sgns.py': 'b' * 64}}
        fcfg = fit.expected_config('C1'); result = fit.fit_mapping(ids, content, manifest, fcfg, engineering=True); out = tmp / 'bundle'
        coll.write_bundle(out, ids=ids, sids=result['sids'], vectors=content, input_manifest=manifest, config=fcfg, config_sha='e' * 64, engineering=True, producer={'src/vkgr/sid_coll_run.py': 'b' * 64})
        np.save(tmp / 'content.npy', np.vstack((np.zeros((1, 64), dtype='float32'), content))); np.save(tmp / 'vectors.npy', content); (tmp / 'input.json').write_bytes(canonical_bytes(manifest))
        loadargs = (out / 'run_manifest.json', file_sha256(out / 'run_manifest.json'), tmp / 'content.npy', file_sha256(tmp / 'content.npy'), fixture['catalog'], True, tmp / 'input.json',
                    file_sha256(tmp / 'input.json'), tmp / 'vectors.npy', file_sha256(tmp / 'vectors.npy'))
        loaded = g._as_depth3(g.coll_base._load_sid(*loadargs))
        counts = {i: 0 for i in fixture['catalog']}; users = {}
        for part in fixture['train_partitions']:
            for row in part['rows']:
                if row['like']:
                    counts[row['item_id']] += 1; users[row['user_id']] = users.get(row['user_id'], 0) + 1
        stats = SliceStatistics((1, 3), MappingProxyType(counts), MappingProxyType(users), 'f' * 64, 'f' * 64, 'f' * 64)
        binding = {'queries_sha256': ge._digest(prepared['queries']), 'catalog_sha256': ge._digest(fixture['catalog']), 'synthetic': True, 'engineering': True, 'b1': {'history_version': 'HIST_V2_OBS_L50'}}
        context = ge.RunContext(ge._CONTEXT_KEY, examples=ListExamples(prepared['examples']), codec=loaded.codec, queries=prepared['queries'], binding=binding, stats=stats).check()
        cfg = dict(g.default_config(), d_model=8, heads=1, head_dim=8, encoder_layers=1, decoder_layers=1, ff=32, dropout=0.0, microbatch=2, max_epochs=1, min_epochs=1, seed=2024)
        g.validate_config(cfg, True)
        args = (Path(G_ROOT), tmp / 'unused-prepared', 'a' * 64, tmp / 'unused-hist', 'b' * 64, *loadargs[:4])
        with mock.patch.object(g, '_contexts', return_value=(context, True, loaded)), mock.patch.dict(os.environ, {'VKGR_DATA_ROOT': str(tmp)}):
            value = g.run(*args, cfg, tmp / 'train', temporary=tmp / 'scratch', decode={'mode': 'batched', 'batch_size': 1})
        return g, context, value, tmp / 'train'

    def test_assemble_core_on_a_real_synthetic_run_and_tamper_gates(self):
        from vkgr import generative_b1_run as b1, generative_evaluation as ev, sid_depth as sd
        from vkgr.baselines import metrics
        from vkgr.data_audit import canonical_bytes
        with tempfile.TemporaryDirectory(dir=REAL_TMP) as tmp:
            tmp = Path(tmp); g, context, value, run_dir = self.build_world(tmp)
            tokens = sd.SIDTokensLD(context.codec, value['config']['history_len'])
            record = ag.assemble_core(value, run_dir, context, tokens, b1._read_evaluation, ev, metrics, canonical_bytes, pop_k=len(context.catalog)//2)
            self.assertEqual(record['schema'], ag.SCHEMA); self.assertEqual(record['query_count'], len(context.queries))
            self.assertEqual([r['ordinal'] for r in record['rows']], list(range(len(context.queries))))
            best = value['best']['epoch']; shard = sorted((run_dir / f'evaluation-{best:03d}').glob('shard-*.json'))[0]
            archived = json.loads(shard.read_text())['rows']
            for row, arch in zip(record['rows'], archived):
                self.assertEqual(row['items'], arch['items'])
            self.assertEqual(record['pop']['ranking'], ag.pop_ranking(context.slice_definition, context.catalog, len(context.catalog) // 2)); self.assertEqual(len(record['pop']['ranking']), len(context.catalog) // 2)
            self.assertTrue(all(r['support'] in ('supported', 'unsupported', 'oov') and r['popularity_group'] in ev.ITEM_GROUPS + ('oov',) and r['user_group'] in ev.USER_GROUPS for r in record['rows']))
            import copy; tampered = copy.deepcopy(value); tampered['epochs'][best - 1]['metrics']['coverage'] += 1.0
            with self.assertRaises(ValueError):
                ag.assemble_core(tampered, run_dir, context, tokens, b1._read_evaluation, ev, metrics, canonical_bytes, pop_k=len(context.catalog)//2)
            shard.write_bytes(shard.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                ag.assemble_core(value, run_dir, context, tokens, b1._read_evaluation, ev, metrics, canonical_bytes, pop_k=len(context.catalog)//2)

    def test_wiring_gates_metric_kind_and_slice_consistency_each_fail_alone(self):
        import copy
        from vkgr import generative_b1_run as b1, generative_evaluation as ev, sid_depth as sd
        from vkgr.baselines import metrics
        from vkgr.data_audit import canonical_bytes
        with tempfile.TemporaryDirectory(dir=REAL_TMP) as tmp:
            tmp = Path(tmp); g, context, value, run_dir = self.build_world(tmp)
            tokens = sd.SIDTokensLD(context.codec, value['config']['history_len']); k = len(context.catalog) // 2; best = value['best']['epoch']
            call = lambda v, reader=b1._read_evaluation: ag.assemble_core(v, run_dir, context, tokens, reader, ev, metrics, canonical_bytes, pop_k=k)
            call(value)                                                                          # baseline passes
            bad = copy.deepcopy(value); bad['best']['metric'] = [bad['best']['metric'][0] + 0.25, bad['best']['metric'][1]]
            with self.assertRaisesRegex(ValueError, 'best metric does not match'):
                call(bad)
            bad = copy.deepcopy(value); bad['epochs'][best - 1]['evaluation']['kind'] = 'l100_query_evaluation'
            with self.assertRaisesRegex(ValueError, 'query_evaluation archive'):
                call(bad)
            # an aggregate whose slice section disagrees with the rows but which the recorded metrics also carry: only the consistency gate can catch it
            def lying_reader(kind, run, record, ctx, toks):
                aggregate, computation, rows = b1._read_evaluation(kind, run, record, ctx, toks)
                aggregate = copy.deepcopy(aggregate); first = next(iter(aggregate['popularity'])); aggregate['popularity'][first]['recall@50'] += 0.5
                return aggregate, computation, rows
            bad = copy.deepcopy(value)
            bad['epochs'][best - 1]['metrics'] = lying_reader(bad['epochs'][best - 1]['evaluation']['kind'], run_dir, bad['epochs'][best - 1], context, tokens)[0]
            with self.assertRaisesRegex(ValueError, 'slice consistency gate failed: popularity'):
                call(bad, lying_reader)

    def test_no_checkpoint_is_opened_by_the_assembly(self):
        import torch
        from vkgr import generative_b1_run as b1, generative_evaluation as ev, sid_depth as sd
        from vkgr.baselines import metrics
        from vkgr.data_audit import canonical_bytes
        with tempfile.TemporaryDirectory(dir=REAL_TMP) as tmp:
            tmp = Path(tmp); g, context, value, run_dir = self.build_world(tmp)
            tokens = sd.SIDTokensLD(context.codec, value['config']['history_len'])
            with mock.patch.object(torch, 'load', side_effect=AssertionError('checkpoint read')):
                ag.assemble_core(value, run_dir, context, tokens, b1._read_evaluation, ev, metrics, canonical_bytes, pop_k=len(context.catalog)//2)

    def main_args(self, tmp, g, run_dir, sha, output='asm-out/g.json', scratch='asm-scratch', extra=()):
        base = []
        for flag in ('prepared', 'prepared-sha', 'history-manifest', 'history-sha', 'sid-manifest', 'sid-sha', 'content', 'content-sha', 'input-manifest', 'input-sha', 'vectors', 'vectors-sha'):
            base += ['--' + flag, 'x']
        return ['--line-root', G_ROOT, *base, '--manifest', str(run_dir / 'manifest.json'), '--manifest-sha', sha, '--output', str(tmp / output), '--scratch', str(tmp / scratch), '--split', 'validation', *extra]

    def patched_main(self, argv, context, engineering=False):
        real = ag.assemble_core; n = len(context.catalog) // 2
        import vkgr.coll_sid_a1_run as cr
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cr, '_contexts', return_value=(context, engineering, None)), mock.patch.object(ag, 'assemble_core', side_effect=lambda *a, **k: real(*a, pop_k=n, **k)), \
                mock.patch('sys.stdout', out), mock.patch('sys.stderr', err):
            code = ag.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_main_writes_one_new_file_matches_stdout_sha_and_check_only_writes_nothing(self):
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory(dir=REAL_TMP) as tmp:
            tmp = Path(tmp); g, context, value, run_dir = self.build_world(tmp); sha = file_sha256(run_dir / 'manifest.json')
            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, extra=('--check-only',)), context)
            self.assertEqual(code, 0, err); summary = json.loads(out); self.assertEqual((summary['status'], summary['written']), ('preflight_ok', False))
            self.assertFalse((tmp / 'asm-out').exists())
            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha), context)
            self.assertEqual(code, 0, err); summary = json.loads(out); target = tmp / 'asm-out/g.json'
            self.assertEqual(summary['status'], 'complete'); self.assertEqual(summary['output_sha256'], file_sha256(target)); self.assertEqual([p.name for p in (tmp / 'asm-out').iterdir()], ['g.json'])
            record = json.loads(target.read_text()); self.assertEqual(record['schema'], 'g5-g-assembly-v1')
            import vkgr.coll_sid_a1_run as cr; self.assertEqual(record['catalog_digest'], cr._digest(list(context.catalog)))
            self.assertEqual((record['seed'], record['run_manifest_sha256'], record['run_id']), (2024, sha, tmp.name)); self.assertEqual(value['config']['seed'], 2024); self.assertEqual(record['script_sha256'], ag.script_sha256())
            self.assertEqual(summary['catalog_digest'], record['catalog_digest']); self.assertEqual(Path(summary['vkgr_file']).parent, Path(G_ROOT).resolve() / 'src' / 'vkgr')
            self.assertIs(summary['cuda_initialized'], False); self.assertGreater(summary['max_rss_mib'], 0); self.assertEqual(summary['script_sha256'], ag.script_sha256())
            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha), context)          # the output now exists: never overwrite
            self.assertEqual(code, 2); self.assertIn('new paths', err)

    def test_main_refuses_engineering_context_binding_and_source_tamper_and_bad_paths(self):
        import copy
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory(dir=REAL_TMP) as tmp:
            tmp = Path(tmp); g, context, value, run_dir = self.build_world(tmp); sha = file_sha256(run_dir / 'manifest.json')
            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, extra=('--check-only',)), context, engineering=True)
            self.assertEqual(code, 2); self.assertIn('formal', err)
            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, 'f' * 64, extra=('--check-only',)), context)
            self.assertEqual(code, 2)                                                                   # manifest sha pin mismatch
            for argv, text in ((self.main_args(tmp, g, run_dir, sha, output='asm-out/g.json', scratch='asm-out', extra=('--check-only',)), 'distinct'),
                               (self.main_args(tmp, g, run_dir, sha, output='run-inside/g.json', scratch='asm-scratch', extra=('--check-only',)), None)):
                code, out, err = self.patched_main(argv, context)
                self.assertEqual(code, 0 if text is None else 2, err)
                if text:
                    self.assertIn(text, err)
            argv = self.main_args(tmp, g, run_dir, sha, extra=('--check-only',)); argv[argv.index('--output') + 1] = str(Path(G_ROOT) / 'x' / 'g.json')
            code, out, err = self.patched_main(argv, context); self.assertEqual(code, 2); self.assertIn('new paths', err)
            existing = tmp / 'asm-scratch'; existing.mkdir()
            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, extra=('--check-only',)), context); self.assertEqual(code, 2)
            class Proxy:                                                                              # RunContext is sealed; delegate everything but the binding
                def __init__(self, inner): self._inner = inner; self.binding = dict(inner.binding, extra='x')
                def __getattr__(self, name): return getattr(self._inner, name)
            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, scratch='asm-scratch2', extra=('--check-only',)), Proxy(context))
            self.assertEqual(code, 2); self.assertIn('context binding differs from the run binding', err)
            import vkgr.coll_sid_a1_run as cr
            fresh = cr._source(Path(G_ROOT)); bad_files = dict(fresh['files']); bad_files[next(iter(bad_files))] = '0' * 64
            with mock.patch.object(cr, '_source', return_value=dict(fresh, files=bad_files)):
                code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, scratch='asm-scratch3', extra=('--check-only',)), context)
            self.assertEqual(code, 2); self.assertIn('deployment source differs', err)
            with mock.patch.object(cr, '_source', return_value=dict(fresh, source_sha256='0' * 64)):
                code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, scratch='asm-scratch5', extra=('--check-only',)), context)
            self.assertEqual(code, 2); self.assertIn('deployment source differs', err)
            (tmp / 'asm-scratch4.partial').write_text('x')                                          # a leftover partial next to a new path is refused too
            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, scratch='asm-scratch4', extra=('--check-only',)), context)
            self.assertEqual(code, 2); self.assertIn('new paths', err)
            (tmp / 'asm-out-p').mkdir(); (tmp / 'asm-out-p' / 'g.json.partial').write_text('x')
            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, output='asm-out-p/g.json', scratch='asm-scratch6', extra=('--check-only',)), context)
            self.assertEqual(code, 2); self.assertIn('new paths', err)

            code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, output='same/g.json', scratch='same/g.json', extra=('--check-only',)), context)
            self.assertEqual(code, 2); self.assertIn('distinct', err)                                        # out == scratch
            import subprocess, vkgr.coll_sid_a1_run as cr2
            with mock.patch.object(cr2, '_source', side_effect=subprocess.CalledProcessError(1, 'git')):
                code, out, err = self.patched_main(self.main_args(tmp, g, run_dir, sha, scratch='asm-scratch7', extra=('--check-only',)), context)
            self.assertEqual(code, 2); self.assertIn('CalledProcessError', err)


    def test_main_rejects_a_line_root_that_is_not_the_imported_vkgr(self):
        with tempfile.TemporaryDirectory(dir=REAL_TMP) as tmp:
            tmp = Path(tmp); argv = self.main_args(tmp, None, tmp, 'a' * 64, extra=('--check-only',)); argv[1] = str(tmp)
            err = io.StringIO()
            with mock.patch('sys.stderr', err):
                self.assertEqual(ag.main(argv), 2)
            self.assertIn('not imported from --line-root', err.getvalue())
            argv[1] = str(Path(G_ROOT).parent); err = io.StringIO()                                    # an ancestor of the real root is not the root either
            with mock.patch('sys.stderr', err):
                self.assertEqual(ag.main(argv), 2)
            self.assertIn('not imported from --line-root', err.getvalue())


if __name__ == '__main__':
    unittest.main()
