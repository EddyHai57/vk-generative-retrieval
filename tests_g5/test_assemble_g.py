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
        cfg = dict(g.default_config(), d_model=8, heads=1, head_dim=8, encoder_layers=1, decoder_layers=1, ff=32, dropout=0.0, microbatch=2, max_epochs=1, min_epochs=1)
        g.validate_config(cfg, True)
        args = (Path(G_ROOT), tmp / 'unused-prepared', 'a' * 64, tmp / 'unused-hist', 'b' * 64, *loadargs[:4])
        with mock.patch.object(g, '_contexts', return_value=(context, True, loaded)), mock.patch.dict(os.environ, {'VKGR_DATA_ROOT': str(tmp)}):
            value = g.run(*args, cfg, tmp / 'train', temporary=tmp / 'scratch', decode={'mode': 'batched', 'batch_size': 1})
        return g, context, value, tmp / 'train'

    def test_assemble_core_on_a_real_synthetic_run_and_tamper_gates(self):
        from vkgr import generative_b1_run as b1, generative_evaluation as ev, sid_depth as sd
        from vkgr.baselines import metrics
        from vkgr.data_audit import canonical_bytes
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
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
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
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
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            tmp = Path(tmp); g, context, value, run_dir = self.build_world(tmp)
            tokens = sd.SIDTokensLD(context.codec, value['config']['history_len'])
            with mock.patch.object(torch, 'load', side_effect=AssertionError('checkpoint read')):
                ag.assemble_core(value, run_dir, context, tokens, b1._read_evaluation, ev, metrics, canonical_bytes, pop_k=len(context.catalog)//2)


if __name__ == '__main__':
    unittest.main()
