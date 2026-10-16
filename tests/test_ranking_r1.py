"""Synthetic oracles: temperature one is bit-identical to the A1/T3 path."""
import copy
import json
from pathlib import Path
import unittest
import torch
from vkgr import ranking_r1 as r
from vkgr import generative_g6_model as old
from vkgr import sid_depth_beam as beam
from vkgr.sid_depth import SIDTokensLD
from tests.test_generative_g6_unit import codec_of, g6_config


class Temperature(unittest.TestCase):
    def setUp(self):
        self.ids, self.codec = codec_of(3, n=40)
        self.config = g6_config(sid='SID_V2_REPLAY_BHKM_B16D3', dropout=0.0, temperature=1.0)
        torch.manual_seed(7)
        self.base = old.make_model_arch_d(self.codec, self.config, device='cpu').eval()
        torch.manual_seed(7)
        self.model = r.make_model(self.codec, self.config, device='cpu').eval()
        self.tokens = SIDTokensLD(self.codec, 50)

    def test_tau_one_weights_logits_loss_gradient_and_update_identical(self):
        source = torch.tensor([self.tokens.history([int(self.ids[0])])])
        decoder = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[2, 3, 4, 5]])
        a, b = self.base(source, decoder), self.model(source, decoder)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(self.base.state_dict().keys(), self.model.state_dict().keys())
        x, y = old.full_vocab_ce_d(a, labels, 4), r.temperature_ce(b, labels, 1.)
        self.assertTrue(torch.equal(x, y))
        x.backward(); y.backward()
        for p, q in zip(self.base.parameters(), self.model.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))
        oa = torch.optim.AdamW(self.base.parameters(), lr=.0003)
        ob = torch.optim.AdamW(self.model.parameters(), lr=.0003)
        oa.step(); ob.step()
        for p, q in zip(self.base.parameters(), self.model.parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_tau_one_serial_and_batched_decode_identical(self):
        histories = [[int(x)] for x in self.ids[:3]]
        for history in histories:
            self.assertEqual(beam.decode_history_d(self.base, self.tokens, history, 100), beam.decode_history_d(self.model, self.tokens, history, 100))
        self.assertEqual(beam.decode_histories_batched_d(self.base, self.tokens, histories, 100, 1), beam.decode_histories_batched_d(self.model, self.tokens, histories, 100, 1))

    def test_tau_point_eight_scales_forward_and_decode_once(self):
        config = dict(self.config, temperature=.8)
        torch.manual_seed(7)
        model = r.make_model(self.codec, config, device='cpu').eval()
        source = torch.tensor([self.tokens.history([int(self.ids[0])])])
        decoder = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(source, decoder), self.base(source, decoder) / .8))
            memory, padding = self.base.encode(source)
            self.assertTrue(torch.equal(model.decode(memory, padding, decoder), self.base.decode(memory, padding, decoder) / .8))

    def test_invalid_temperature_stops(self):
        for tau in (0., -1., float('nan'), .7, True):
            with self.assertRaises(ValueError):
                r.make_model(self.codec, dict(self.config, temperature=tau), device='cpu')


class Config(unittest.TestCase):
    def test_three_configs_only_change_schema_task_temperature_and_seed(self):
        from vkgr import ranking_r1_run as run
        base = json.loads(Path('configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json').read_text())
        for seed in (42, 2024, 2025):
            value = json.loads(Path(f'configs/ranking_r1/g-ranking-r1-t08-s{seed}.json').read_text())
            run.validate_config(value, False)
            self.assertEqual({k: v for k, v in value.items() if k not in ('schema', 'task_id', 'temperature', 'seed')}, {k: v for k, v in base.items() if k not in ('schema', 'task_id', 'seed')})
            self.assertEqual(value['temperature'], .8)
            self.assertEqual(value['seed'], seed)
            with self.assertRaises(ValueError):
                run.validate_config(dict(value, temperature=1.), False)
            with self.assertRaises(ValueError):
                run.validate_config(dict(value, sid='SID_BHKM_B1024D3_S42'), False)

    def test_source_binding_includes_every_new_runtime_module(self):
        from vkgr import ranking_r1_run as run
        for name in ('ranking_r1', 'ranking_r1_run'):
            self.assertIn(f'src/vkgr/{name}.py', run.G6_MEMBERS)

    def test_old_manifest_cannot_enter_r1_verify(self):
        import tempfile
        from vkgr import ranking_r1_run as run
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / 'manifest.json'
            path.write_text(json.dumps({'schema': 'g-g6-train-manifest-v1'}))
            with self.assertRaises(ValueError):
                run._read_run(path, file_sha256(path))


class RunnerSynthetic(unittest.TestCase):
    def test_train_and_verify_new_checkpoint_and_reject_temperature_tamper(self):
        import tempfile
        from types import MappingProxyType, SimpleNamespace
        from unittest.mock import patch
        from vkgr import generative_experiment as ge, generative_evaluation as ev
        from vkgr.generative_data_v2 import ListExamples
        from vkgr import ranking_r1_run as run
        ids, codec = codec_of(3, n=8)
        ids = [int(i) for i in ids]
        rows = [{'history': [ids[0]], 'target': ids[1]}, {'history': [ids[1]], 'target': ids[2]}]
        queries = [{'user_id': 1, 'history': [ids[0]], 'history_positions': [[0, 0]], 'target': ids[1], 'target_position': [25, 0], 'oov': False}]
        stats = ev.SliceStatistics((1, 1), MappingProxyType({i: 1 for i in ids}), MappingProxyType({1: 2}), 'a'*64, 'b'*64, 'c'*64)
        context = ge.RunContext(ge._CONTEXT_KEY, examples=ListExamples(rows), codec=codec, queries=queries,
            binding={'queries_sha256': ge._digest(queries), 'catalog_sha256': ge._digest(ids), 'synthetic': True, 'engineering': True, 'b1': {}}, stats=stats).check()
        info = {'sid_version': 'SID_V2_REPLAY_BHKM_B16D3', 'manifest_sha256': 'd'*64, 'branching': codec.catalog.branching,
                'depth': 3, 'suffix_vocabulary': codec.catalog.bases[-1], 'vocab_size': codec.vocab_size, 'kind': 'bhkm', 'engineering': True}
        sid = SimpleNamespace(info=info, manifest={'content_feature_sha256': 'e'*64})
        config = g6_config(sid=info['sid_version'], temperature=.8, schema=run.SCHEMA, task_id=run.TASK_ID)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp, patch.object(run, '_contexts', return_value=(context, True, sid)):
            common = (root, Path('synthetic'), '1'*64, Path('synthetic'), '2'*64, Path('synthetic'), '3'*64, Path('synthetic'), '4'*64)
            out = Path(temp)/'train'
            result = run.run(*common, config, out, temporary=Path(temp)/'scratch', decode={'mode': 'batched', 'batch_size': 1})
            self.assertEqual(result['status'], 'complete')
            verified = run.verify(*common, out/'manifest.json', result['manifest_sha256'], Path(temp)/'verify', temporary=Path(temp)/'verify-scratch', decode={'mode': 'batched', 'batch_size': 1})
            self.assertEqual(verified['status'], 'verified')
            bad = dict(config, temperature=.7)
            with self.assertRaises(ValueError):
                run.validate_config(bad, True)
            self.assertEqual(sorted(p.name for p in out.glob('*.pt')), ['best.pt', 'last.pt'])

    def test_dropout_training_step_tau_one_is_bit_identical(self):
        from vkgr import generative_arch as ga
        ids, codec = codec_of(3, n=10)
        config = g6_config(sid='SID_V2_REPLAY_BHKM_B16D3', temperature=1.)
        torch.manual_seed(42)
        base = old.make_model_arch_d(codec, config, device='cpu')
        torch.manual_seed(42)
        model = r.make_model(codec, config, device='cpu')
        rows = [{'history': [int(ids[i])], 'target': int(ids[i+1])} for i in range(3)]
        tokens = SIDTokensLD(codec, 50)
        oa, ob = ga.make_optimizer_arch(base, config), ga.make_optimizer_arch(model, config)
        torch.manual_seed(99)
        a = old.train_batch_d(base, oa, rows, codec, tokens, config, 'cpu')
        torch.manual_seed(99)
        b = old.train_batch_d(model, ob, rows, codec, tokens, config, 'cpu')
        self.assertEqual(a, b)
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)


class DecodeGate(unittest.TestCase):
    def test_runner_requires_frozen_batched_one_and_single_worker(self):
        from vkgr import ranking_r1_run as run
        for decode, workers in ((None, 1), ({'mode': 'batched', 'batch_size': 2}, 1), ({'mode': 'batched', 'batch_size': 1}, 2)):
            with self.assertRaises(ValueError):
                run._gate(decode, workers)
        run._gate({'mode': 'batched', 'batch_size': 1}, 1)


class SeparateDataRoot(unittest.TestCase):
    def test_explicit_existing_data_root_and_source_root_remain_separate(self):
        import os
        import tempfile
        from unittest.mock import patch
        from vkgr import ranking_r1_data as data
        from vkgr import ranking_r1_run as run
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            accepted = Path(temp).resolve()
            with patch.dict(os.environ, {'VKGR_DATA_ROOT': str(accepted)}):
                self.assertEqual(data.data_root(source), accepted)
                files = run._source(source)['files']
                self.assertIn('src/vkgr/ranking_r1_data.py', files)
                self.assertEqual(files['src/vkgr/ranking_r1.py'], run.file_sha256(source/'src/vkgr/ranking_r1.py'))
            with patch.dict(os.environ, {'VKGR_DATA_ROOT': ''}):
                with self.assertRaises(ValueError):
                    data.data_root(source)
            with patch.dict(os.environ, {'VKGR_DATA_ROOT': str(accepted/'missing')}):
                with self.assertRaises(ValueError):
                    data.data_root(source)


class DataLoaderRouting(unittest.TestCase):
    def test_g3_loader_receives_data_root_without_mutating_old_module(self):
        import os
        import tempfile
        from unittest.mock import patch
        from vkgr import ranking_r1_run as run, generative_real as gr
        source = Path(__file__).resolve().parents[1]
        original_loader = gr.load_context
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            accepted = Path(temp).resolve()
            with patch.dict(os.environ, {'VKGR_DATA_ROOT': str(accepted)}), patch.object(run.gr, 'load_context', side_effect=RuntimeError('synthetic routing sentinel')) as loader:
                with self.assertRaisesRegex(RuntimeError, 'routing sentinel'):
                    run._contexts(source, Path('p'), 'a'*64, Path('h'), 'b'*64, Path('s'), 'c'*64, Path('c'), 'd'*64,
                                  run.default_config(), accepted/'new-scratch')
                self.assertEqual(loader.call_args.args[0], accepted)
            self.assertIs(gr.load_context, original_loader)
