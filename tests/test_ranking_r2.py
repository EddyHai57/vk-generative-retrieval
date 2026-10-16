"""Synthetic-only oracles for the ranking objective and mining engineering path."""
import math
import random
import unittest
import torch
from vkgr import ranking_r2 as r


class Objective(unittest.TestCase):
    def test_dpo_hand_example_and_reference_detached(self):
        pc = torch.tensor([math.log(.6)], dtype=torch.float64, requires_grad=True)
        pr = torch.tensor([math.log(.2)], dtype=torch.float64, requires_grad=True)
        rc = torch.tensor([math.log(.4)], dtype=torch.float64, requires_grad=True)
        rr = torch.tensor([math.log(.2)], dtype=torch.float64, requires_grad=True)
        ce = torch.tensor(2., dtype=torch.float64, requires_grad=True)
        loss = r.dpo_with_ce(pc, pr, rc, rr, ce)
        expected = math.log1p(math.exp(-.1 * math.log(1.5))) + 2
        self.assertAlmostEqual(loss.item(), expected, places=14)
        loss.backward()
        self.assertIsNone(rc.grad)
        self.assertIsNone(rr.grad)
        self.assertGreater(pr.grad.item(), 0)
        self.assertLess(pc.grad.item(), 0)
        self.assertEqual(ce.grad.item(), 1)

    def test_sequence_logprobs_are_summed_not_averaged(self):
        logits = torch.zeros(1, 4, 3, dtype=torch.float64)
        self.assertAlmostEqual(r.sequence_logprob(logits, torch.tensor([[0, 1, 2, 0]])).item(), -4 * math.log(3))

    def test_freeze_reference_copies_and_disables_dropout_and_grad(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.9))
        ref = r.frozen_reference(model)
        self.assertFalse(ref.training)
        self.assertTrue(all(not p.requires_grad for p in ref.parameters()))
        with torch.no_grad():
            model[0].weight.add_(1)
        self.assertFalse(torch.equal(model[0].weight, ref[0].weight))

    def test_nonfinite_and_mismatched_dpo_rejected(self):
        for pc in (torch.tensor([float('nan')]), torch.tensor([0., 1.])):
            with self.assertRaises(ValueError):
                r.dpo_with_ce(pc, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.tensor(0.))


class Mining(unittest.TestCase):
    def test_complete_past_likes_not_observed_history(self):
        row = {'target': 3, 'history': [8], 'past_likes': [1, 2]}
        self.assertEqual(r.select_negative([3, 1, 2, 8, 9], row), 8)
        self.assertIsNone(r.select_negative([3, 1, 2], row))
        with self.assertRaises(ValueError):
            r.select_negative([9], {'target': 3, 'history': [1]})

    def test_deterministic_seed_zero_without_replacement(self):
        a = r.sample_training_indices(20001)
        self.assertEqual(a, random.Random(0).sample(range(20001), 20000))
        self.assertEqual(len(set(a)), 20000)
        with self.assertRaises(ValueError):
            r.sample_training_indices(19999)

    def test_benchmark_uses_only_training_histories_and_fixed_decode(self):
        from tests.test_sid_depth_beam import world
        from vkgr.sid_depth import SIDTokensLD
        ids, codec, model = world(3, n=30)
        rows = [{'example_id': i, 'history': [int(ids[0])], 'target': int(ids[1]), 'past_likes': [int(ids[0])]} for i in range(2)]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        out = r.benchmark(model, SIDTokensLD(codec, 50), rows, estimate_count=20000)
        self.assertEqual(out['beam'], 10)
        self.assertEqual(out['decode_batch_size'], 1)
        self.assertEqual(out['queries'], 2)
        self.assertEqual(out['model_weights'], 'random_initialization_engineering_only')
        self.assertTrue(out['elapsed_seconds'] > 0)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()


class TrainOnlyInputs(unittest.TestCase):
    def test_pinned_train_loading_never_opens_validation_and_keeps_full_likes(self):
        import json
        import hashlib
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from vkgr import ranking_r2_microbench as mb
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            def put(name, obj, jsonl=False):
                data = ((json.dumps(obj) + '\n') if jsonl else json.dumps(obj)).encode()
                (root / name).write_bytes(data)
                return hashlib.sha256(data).hexdigest()
            row = {'example_id': 0, 'user_id': 4, 'history': [1], 'history_positions': [[0, 0]], 'target': 2, 'target_position': [0, 2], 'past_likes': [1]}
            artifacts = {'catalog.json': put('catalog.json', [1, 2, 3]), 'examples.jsonl': put('examples.jsonl', row, True)}
            psha = put('prepare-manifest.json', {'schema_version': 1, 'task_id': 'P4B', 'kind': 'prepare', 'status': 'success', 'examples': 1, 'catalog_items': 3, 'artifacts': artifacts})
            hsha = put('history.json', {'kind': 'history_v2_prepare', 'history_version': 'HIST_V2_OBS', 'lengths': [100, 50, 20], 'test_gate': 'closed_hash_only', 'training_examples': 1,
                'artifacts': {'train-l50.json': put('train-l50.json', [{'example_id': 0, 'history': [1, 3], 'history_positions': [[0, 0], [0, 1]], 'target': 2, 'target_position': [0, 2]}]), 'validation-l50.json': 'must-not-open'}})
            opened = []
            original = Path.open
            def checked(path, *args, **kwargs):
                opened.append(path.name)
                self.assertNotIn('validation', path.name)
                self.assertNotIn('test', path.name)
                return original(path, *args, **kwargs)
            with patch.object(Path, 'open', checked):
                rows, catalog = mb.load_training(root / 'prepare-manifest.json', psha, root / 'history.json', hsha)
            self.assertEqual(rows[0]['past_likes'], [1])
            self.assertEqual(rows[0]['history'], [1, 3])
            self.assertEqual(catalog, [1, 2, 3])
            self.assertIn('examples.jsonl', opened)
            (root / 'train-l50.json').write_text('[]')
            with self.assertRaises(ValueError):
                mb.load_training(root / 'prepare-manifest.json', psha, root / 'history.json', hsha)


class Cli(unittest.TestCase):
    def test_help_has_no_checkpoint_or_holdout_options(self):
        from vkgr import ranking_r2_microbench as mb
        from contextlib import redirect_stdout
        import io
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            mb.main(['--help'])
        self.assertEqual(stopped.exception.code, 0)
        text = output.getvalue()
        self.assertNotIn('--checkpoint', text)
        self.assertNotIn('--validation', text)
        self.assertNotIn('--test', text)
        self.assertIn('--queries', text)

    def test_timing_receipt_binds_current_execution_commit(self):
        from vkgr import ranking_r2_microbench as mb
        self.assertRegex(mb.execution_commit(), '^[0-9a-f]{40}$')


class RuntimeBinding(unittest.TestCase):
    def test_cli_configures_formal_runtime_before_model_and_records_it(self):
        import json
        import io
        from pathlib import Path
        from types import SimpleNamespace
        from contextlib import redirect_stdout, ExitStack
        from unittest.mock import patch
        from vkgr import ranking_r2_microbench as mb, generative_g6_run as g6, generative_g6_model as gm
        events = []
        config = json.loads(Path('configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json').read_text())
        profile = {'deterministic_algorithms': True, 'allow_tf32': False, 'device': 'cuda:0'}
        from vkgr.sid_v2 import build_codec
        import numpy as np
        codec = build_codec(np.array([11,12], dtype=np.uint32), np.array([[0,0,0,0],[1,0,0,0]],dtype=np.uint32),256)
        sid = SimpleNamespace(info={'sid_version': config['sid']}, codec=codec)
        model = SimpleNamespace(spec={})
        def runtime(engineering):
            self.assertFalse(engineering)
            events.append('runtime')
            return profile
        def factory(*args, **kwargs):
            self.assertEqual(events, ['runtime'])
            events.append('model')
            return model
        with ExitStack() as stack:
            for target, kwargs in [
                ('vkgr.ranking_r2_microbench._read', {'return_value': config}),
                ('vkgr.ranking_r2_microbench.load_training', {'return_value': ([{}] * 20000, [])}),
                ('vkgr.generative_g6_run._runtime', {'side_effect': runtime}),
                ('vkgr.generative_g6_run._load_sid_v2', {'return_value': sid}),
                ('vkgr.generative_g6_model.make_model_arch_d', {'side_effect': factory}),
                ('vkgr.sid_depth.SIDTokensLD', {'return_value': object()}),
                ('vkgr.ranking_r2.benchmark', {'return_value': {}}),
                ('torch.cuda.is_available', {'return_value': True}),
                ('torch.cuda.get_device_name', {'return_value': 'synthetic'}),
                ('torch.cuda.manual_seed_all', {}), ('torch.manual_seed', {}),
                ('torch.set_num_threads', {}), ('torch.set_num_interop_threads', {})]:
                stack.enter_context(patch(target, **kwargs))
            output = io.StringIO()
            with redirect_stdout(output):
                mb.main([x for n in ('prepared', 'history', 'sid', 'content') for x in ('--'+n, 'synthetic', '--'+n+'-sha256', '0'*64)] + ['--config', 'synthetic'])
        self.assertEqual(events, ['runtime', 'model'])
        self.assertEqual(json.loads(output.getvalue())['runtime'], profile)
