"""G6-PREP step 7a: config registry, depth-aware model/position table and the depth-generic training step (no engineering world).

Run with the other G tests inside the frozen-base harness:  python -B -m unittest discover -s tests -p 'test_generative_g6*.py'
"""
import json
import unittest
from pathlib import Path

import torch

from tests import g6_fixtures as f
from vkgr import sid_depth as sd
from vkgr.sid import SIDCatalog

ROOT = Path(__file__).resolve().parents[1]
REPLAY = 'SID_V2D_REPLAY_BHKM_B16D3'
A1_CONFIG = ROOT / 'configs' / 'garch01' / 'g-garch01-a1-lr3e-4-s42.json'
G6_CONFIG = ROOT / 'configs' / 'g6prep' / 'g-g6-a1-bhkm-b16d4-s42.json'


def g6_config(sid=REPLAY, **overrides):
    """Engineering-size independent-head config (3 heads x 16 on d_model 64, 2+2 layers) on the G6 schema."""
    from vkgr import generative_arch as ga
    value = ga.default_config()
    value.update({'d_model': 64, 'heads': 3, 'head_dim': 16, 'encoder_layers': 2, 'decoder_layers': 2, 'ff': 128, 'dropout': 0.1,
                  'microbatch': 2, 'max_epochs': 3, 'min_epochs': 1, 'learning_rate': .001})
    value.update({'schema': 'g-g6-run-v1', 'task_id': 'G-HP-01-G6', 'sid': sid})
    value.update(overrides)
    return value


def codec_of(depth, branching=3, n=120, seed=1):
    ids, sids = f.synth(n, branching, depth, seed)
    return ids, sd.ValidatedCodecD(SIDCatalog(ids, sids, branching, depth, []), depth)


class ConfigTests(unittest.TestCase):
    def test_committed_config_differs_from_a1_only_in_schema_task_and_sid(self):
        a1, g6 = json.loads(A1_CONFIG.read_text()), json.loads(G6_CONFIG.read_text())
        self.assertEqual(set(a1), set(g6))
        self.assertEqual({k for k in a1 if a1[k] != g6[k]}, {'schema', 'task_id', 'sid'})
        self.assertEqual((g6['schema'], g6['task_id'], g6['sid']), ('g-g6-run-v1', 'G-HP-01-G6', 'SID_BHKM_B16D4_S42'))

    def test_formal_validation_accepts_the_committed_config_and_the_rqvae_label(self):
        from vkgr import generative_g6_run as g6
        value = json.loads(G6_CONFIG.read_text())
        self.assertEqual(g6.validate_config(value), value)
        self.assertEqual(g6.default_config(), value)
        g6.validate_config(dict(value, sid='SID_RQVAE_B256D4_S42'))
        for microbatch in (512, 256, 128):
            g6.validate_config(dict(value, microbatch=microbatch))
        for seed in (42, 2024, 2025):
            g6.validate_config(dict(value, seed=seed))

    def test_formal_validation_rejects_everything_else(self):
        from vkgr import generative_g6_run as g6
        value = json.loads(G6_CONFIG.read_text())
        bad = {'schema': 'g-b1-arch-run-v1', 'task_id': 'G-ARCH-01', 'sid': 'SID_BHKM_B16D3_S42', 'history_len': 100, 'beam': 200,
               'learning_rate': .003, 'seed': 7, 'microbatch': 64, 'max_epochs': 30, 'min_epochs': 1, 'patience': 5, 'dropout': 0.0,
               'heads': 4, 'head_dim': 32, 'encoder_layers': 1, 'ff': 512, 'weight_decay': 0.0, 'test_gate': 'open', 'precision': 'bf16',
               'init': 'legacy', 'history_version': 'HIST_V1', 'effective_batch': 256, 'gradient_clip_norm': 2.0, 'd_model': 64}
        for key, replacement in bad.items():
            with self.subTest(key=key):
                self.assertRaises(ValueError, g6.validate_config, dict(value, **{key: replacement}))
        for label in (REPLAY, 'SID_RQVAE_B256D3_S42', 'SID_RQVAE_B16D4_S42'):
            with self.subTest(label=label):
                self.assertRaises(ValueError, g6.validate_config, dict(value, sid=label))
        for broken in ({k: v for k, v in value.items() if k != 'sid'}, dict(value, extra=1), dict(value, seed=42.0), dict(value, dropout=1)):
            self.assertRaises(ValueError, g6.validate_config, broken)
        self.assertRaises(ValueError, g6.validate_config, [])

    def test_engineering_validation_is_bounded_and_accepts_the_replay_label(self):
        from vkgr import generative_g6_run as g6
        g6.validate_config(g6_config(), engineering=True)
        for overrides in ({'d_model': 256}, {'heads': 9}, {'head_dim': 24}, {'ff': 4096}, {'dropout': .9}, {'microbatch': 600}, {'max_epochs': 200},
                          {'sid': 'SID_BHKM_B16D3_S42'}, {'history_len': 100}, {'beam': 50}):
            with self.subTest(overrides=overrides):
                self.assertRaises(ValueError, g6.validate_config, g6_config(**overrides), True)
        self.assertRaises(ValueError, g6.validate_config, g6_config(), False)  # the replay label is not a formal label


class ModelTests(unittest.TestCase):
    def test_depth_three_model_equals_the_frozen_arch_model(self):
        from vkgr import generative_g6_model as gm
        from vkgr.tiger_arch import TigerArchModel
        for shape in ((128, 4, 32, 1, 1, 512, 0.0), (64, 3, 16, 2, 2, 128, 0.1)):
            torch.manual_seed(5)
            old = TigerArchModel(80, *shape)
            torch.manual_seed(5)
            new = gm.TigerArchModelD(80, *shape, max_tokens=200)
            self.assertEqual(list(old.state_dict()), list(new.state_dict()))
            for key, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
            self.assertEqual(old.spec, new.spec)

    def test_position_table_follows_the_token_count(self):
        from vkgr import generative_g6_model as gm
        model = gm.TigerArchModelD(40, 64, 3, 16, 1, 1, 64, 0.0, max_tokens=250)
        self.assertEqual(tuple(model.position.weight.shape), (250, 64))
        self.assertEqual(model.spec['positions'], 250)
        self.assertEqual(tuple(model.embed(torch.ones((2, 250), dtype=torch.long)).shape), (2, 250, 64))
        self.assertRaises(ValueError, model.embed, torch.ones((2, 251), dtype=torch.long))
        self.assertRaises(ValueError, model.embed, torch.ones((2, 0), dtype=torch.long))
        self.assertRaises(ValueError, model.embed, torch.ones(5, dtype=torch.long))
        small = gm.TigerArchModelD(40, 64, 3, 16, 1, 1, 64, 0.0, max_tokens=200)
        self.assertRaises(ValueError, small.embed, torch.ones((1, 250), dtype=torch.long))

    def test_factory_sets_the_table_from_depth_and_history_length(self):
        from vkgr import generative_g6_model as gm
        for depth, rows in ((3, 200), (4, 250), (5, 300)):
            _, codec = codec_of(depth)
            model = gm.make_model_arch_d(codec, g6_config(), device='cpu')
            self.assertEqual(tuple(model.position.weight.shape), (rows, 64), depth)
            report = model.init_report['arch']
            self.assertEqual((report['spec']['positions'], report['spec']['vocab']), (rows, codec.vocab_size))
            self.assertEqual(report['params_total'], report['params_expected'])
            self.assertEqual(model.init_report['init'], 'small')

    def test_depth_three_factory_equals_the_frozen_arch_factory(self):
        from vkgr import generative_arch as ga
        from vkgr import generative_g6_model as gm
        _, codec = codec_of(3)
        torch.manual_seed(9)
        old = ga.make_model_arch(codec, dict(ga.default_config(), d_model=64, heads=3, head_dim=16, encoder_layers=2, decoder_layers=2, ff=128, dropout=0.1), device='cpu')
        torch.manual_seed(9)
        new = gm.make_model_arch_d(codec, g6_config(), device='cpu')
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        self.assertEqual(old.init_report, new.init_report)

    def test_factory_refuses_inconsistent_configs(self):
        from vkgr import generative_g6_model as gm
        _, codec = codec_of(4)
        for overrides in ({'init': 'xavier'}, {'history_len': 100}, {'heads': 3.0}, {'dropout': 0}):
            with self.subTest(overrides=overrides):
                self.assertRaises(ValueError, gm.make_model_arch_d, codec, g6_config(**overrides), device='cpu')
        self.assertRaises(ValueError, gm.make_model_arch_d, f.plain_codec(120, 3, 4, 1), g6_config(), device='cpu')  # needs a depth-validated codec


class TrainingStepTests(unittest.TestCase):
    @staticmethod
    def rows(ids, count=8, width=4):
        return [{'history': [int(x) for x in ids[(i * 3) % 90:(i * 3) % 90 + width]], 'target': int(ids[(i * 7 + 3) % len(ids)])} for i in range(count)]

    def test_loss_accepts_depth_plus_one_positions_only(self):
        from vkgr import generative_g6_model as gm
        logits, labels = torch.randn(3, 5, 20), torch.randint(0, 20, (3, 5))
        self.assertAlmostEqual(float(gm.full_vocab_ce_d(logits, labels, 5)),
                               float(torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())), places=6)
        for bad in ((logits, labels, 4), (logits[:, :4], labels[:, :4], 5), (logits, labels.float(), 5), (logits[0], labels, 5)):
            self.assertRaises(ValueError, gm.full_vocab_ce_d, *bad)
        nan = logits.clone()
        nan[0, 0, 0] = float('nan')
        self.assertRaises(ValueError, gm.full_vocab_ce_d, nan, labels, 5)
        self.assertRaises(ValueError, gm.full_vocab_ce_d, logits, torch.full_like(labels, 99), 5)

    def test_depth_three_step_equals_the_frozen_effective_batch_step(self):
        from vkgr import generative_arch as ga
        from vkgr import generative_g6_model as gm
        from vkgr import generative_training as gt
        config = g6_config()
        results = []
        for frozen in (True, False):
            ids, codec = codec_of(3, seed=4)
            torch.manual_seed(3)
            model = gm.make_model_arch_d(codec, config, device='cpu')
            optimizer = ga.make_optimizer_arch(model, config)
            rows = self.rows(ids)
            if frozen:
                step = gt.train_effective_batch(model, optimizer, rows, codec, microbatch=3, device='cpu')
            else:
                step = gm.train_batch_d(model, optimizer, rows, codec, sd.SIDTokensLD(codec, 50), dict(config, microbatch=3), 'cpu')
            results.append((step, dict(model.state_dict()), optimizer.state_dict()))
        self.assertEqual(results[0][0], results[1][0])
        for key, value in results[0][1].items():
            self.assertTrue(torch.equal(value, results[1][1][key]), key)
        for index, state in results[0][2]['state'].items():
            for name in ('exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(state[name], results[1][2]['state'][index][name]))

    def test_depth_four_steps_train_with_five_positions(self):
        from vkgr import generative_arch as ga
        from vkgr import generative_g6_model as gm
        ids, codec = codec_of(4, seed=2)
        config = g6_config(dropout=0.0, learning_rate=.001)
        torch.manual_seed(1)
        model = gm.make_model_arch_d(codec, config, device='cpu')
        optimizer = ga.make_optimizer_arch(model, config)
        tokens = sd.SIDTokensLD(codec, 50)
        rows = self.rows(ids, 16)
        losses = [gm.train_batch_d(model, optimizer, rows, codec, tokens, dict(config, microbatch=4), 'cpu') for _ in range(12)]
        self.assertTrue(all(step.examples == 16 and step.microbatch_sizes == (4, 4, 4, 4) for step in losses))
        self.assertLess(losses[-1].loss_mean, losses[0].loss_mean)
        self.assertEqual(tuple(model(*tokens.batch([rows[0]['history']], [rows[0]['target']])[:2]).shape), (1, 5, codec.vocab_size))

    def test_step_refuses_bad_batches_and_leaves_no_gradients(self):
        from vkgr import generative_arch as ga
        from vkgr import generative_g6_model as gm
        ids, codec = codec_of(4)
        config = g6_config()
        model = gm.make_model_arch_d(codec, config, device='cpu')
        optimizer = ga.make_optimizer_arch(model, config)
        tokens = sd.SIDTokensLD(codec, 50)
        for rows in ([], [{'history': [], 'target': int(ids[0])}], [{'history': [int(ids[0])] * 51, 'target': int(ids[0])}],
                     [{'history': [int(ids[0])], 'target': 'x'}], [{'history': [int(ids[0])], 'target': 10 ** 9}]):
            self.assertRaises(ValueError, gm.train_batch_d, model, optimizer, rows, codec, tokens, config, 'cpu')
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertRaises(ValueError, gm.train_batch_d, model, optimizer, self.rows(ids, 2), codec, tokens, dict(config, microbatch=0), 'cpu')


if __name__ == '__main__':
    unittest.main()
