"""Batch1 G switches on the frozen-base engineering fixture (run inside the INFRA-01 6.1 harness, OMP=1)."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests' / 'fixtures' / 'batch1_early_stop_cases.json'


def tmpdir():
    return tempfile.TemporaryDirectory(dir='/private/tmp')


def engineering_world(tmp):
    """Engineering context + a HIST_V2 manifest produced by the shared P/G producer."""
    from vkgr import generative_real as gr
    from vkgr import history_v2
    context, receipt = gr.prepare_engineering(ROOT, Path(tmp) / 'prepared')
    examples = context.examples.rows(list(range(len(context.examples))))
    likes = {6: (1, 10), 7: (1, 20), 8: (1, 10), 9: (2, 50), 10: (3, 10), 11: (3, 30), 12: (4, 20), 13: (4, 40), 14: (5, 50), 15: (6, 40), 16: (7, 30)}
    week0 = [{'user_id': 1 if k % 2 else 3, 'item_id': (10, 20, 30, 40, 50, 60)[k % 6], 'like': False} for k in range(6)]
    week0 += [{'user_id': u, 'item_id': i, 'like': True} for _, (u, i) in sorted(likes.items())]
    later = [(w, [{'user_id': (1, 2, 3, 4, 5, 6, 7)[k % 7], 'item_id': (10, 20, 30, 40, 50, 60)[(k * 5 + w) % 6], 'like': False} for k in range(200)]) for w in (1, 2)]
    partitions = [(0, week0)] + later
    hist = history_v2.write_prepared(Path(tmp) / 'hist', partitions, examples, context.queries, list(context.catalog), {'synthetic': True})
    return context, receipt, Path(tmp) / 'hist' / 'manifest.json', hist['manifest_sha256']


def config(**overrides):
    from vkgr.generative_b1_run import default_config
    value = default_config()
    value.update({'microbatch': 2, 'max_epochs': 3, 'min_epochs': 1, 'd_model': 64, 'ff': 256, 'learning_rate': .001})
    value.update(overrides)
    return value


class EarlyStopAndInitTests(unittest.TestCase):
    def test_shared_case_table(self):
        from vkgr.generative_hp import EarlyStop
        for case in json.loads(FIXTURE.read_text())['cases']:
            stop = EarlyStop(case['max_epochs'], case['min_epochs'], case['patience'])
            for epoch, (recall, ndcg) in enumerate(case['metrics'], 1):
                self.assertFalse(stop.stopped, case['name'])
                stop.observe(epoch, recall, ndcg)
            self.assertTrue(stop.stopped)
            self.assertEqual((stop.epoch, stop.best_epoch), (case['stop_epoch'], case['best_epoch']), case['name'])

    def _codec(self, tmp):
        from vkgr import generative_real as gr
        context, _ = gr.prepare_engineering(ROOT, Path(tmp) / 'p')
        return context.codec

    def test_legacy_l50_is_value_equal_to_frozen_factory(self):
        from vkgr import generative_training as gt
        from vkgr.generative_hp import make_model_b1, core_config
        with tmpdir() as tmp:
            codec = self._codec(tmp)
            cfg = config(init='legacy')
            torch.manual_seed(42); expected = gt.make_model(codec, core_config(cfg), device='cpu').state_dict()
            torch.manual_seed(42); actual = make_model_b1(codec, cfg, device='cpu').state_dict()
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_small_init_embeddings_only_and_sqrt_d_forward_kept(self):
        from vkgr import generative_training as gt
        from vkgr.generative_hp import make_model_b1, core_config
        with tmpdir() as tmp:
            codec = self._codec(tmp)
            torch.manual_seed(42); legacy = make_model_b1(codec, config(init='legacy'), device='cpu')
            torch.manual_seed(42); small = make_model_b1(codec, config(init='small'), device='cpu')
            for name in ('embedding', 'position'):
                weight = getattr(small, name).weight.detach()
                rows = weight[1:] if name == 'embedding' else weight
                self.assertLess(abs(float(rows.std()) - 0.02), 0.006, name)
            self.assertTrue(torch.equal(small.embedding.weight[0], torch.zeros(64)))
            for key, value in legacy.state_dict().items():
                if not key.startswith(('embedding.', 'position.')):
                    self.assertTrue(torch.equal(value, small.state_dict()[key]), key)
            tokens = torch.tensor([[2, 3, 0]])
            expected = small.embedding(tokens) * 8.0 + small.position(torch.arange(3))[None]
            self.assertTrue(torch.allclose(small.embed(tokens), expected))
            self.assertEqual(small.init_report['init'], 'small')
            torch.manual_seed(42); frozen = gt.make_model(codec, core_config(config()), device='cpu')
            frozen.load_state_dict(small.state_dict())  # gs workers rebuild with the frozen factory

    def test_l100_tokens_and_position_table(self):
        from vkgr.generative_data_v2 import SIDTokensL
        from vkgr.generative_hp import make_model_b1
        with tmpdir() as tmp:
            codec = self._codec(tmp)
            tokens50 = SIDTokensL(codec, 50); tokens100 = SIDTokensL(codec, 100)
            history = [10, 20, 30, 40, 50, 60] * 20
            self.assertEqual(tokens50.history(history), codec.history(history))
            self.assertEqual(len(tokens100.history(history)), 400)
            model = make_model_b1(codec, config(history_len=100), device='cpu')
            self.assertEqual(model.position.weight.shape[0], 400)
            source = torch.tensor([tokens100.history(history)])
            memory, _ = model.encode(source)
            self.assertEqual(memory.shape[1], 400)


class ContextTests(unittest.TestCase):
    def test_v2_context_keeps_identity_and_replaces_history(self):
        from vkgr import history_v2
        from vkgr.generative_data_v2 import build_v2_context
        with tmpdir() as tmp:
            context, _, manifest, sha = engineering_world(tmp)
            loaded = history_v2.load_prepared(manifest, sha, 50)
            v2 = build_v2_context(context, loaded, 50, 'SID_BHKM_B16D3_S42')
            self.assertEqual([q['target'] for q in v2.queries], [q['target'] for q in context.queries])
            self.assertEqual([q['oov'] for q in v2.queries], [q['oov'] for q in context.queries])
            self.assertGreater(max(len(q['history']) for q in v2.queries), 3)
            self.assertEqual(max(len(q['history']) for q in v2.queries), 50)
            originals = context.examples.rows(list(range(len(context.examples))))
            rows = v2.examples.rows(list(range(len(v2.examples))))
            self.assertEqual([r['target'] for r in rows], [r['target'] for r in originals])
            self.assertGreater(sum(len(r['history']) for r in rows), sum(len(r['history']) for r in originals))
            self.assertEqual(v2.binding['b1']['history_version'], 'HIST_V2_OBS_L50')
            self.assertNotEqual(v2.binding['queries_sha256'], context.binding['queries_sha256'])
            bad = {**loaded, 'validation': [{**loaded['validation'][0], 'target': 41}] + loaded['validation'][1:]}
            with self.assertRaises(ValueError):
                build_v2_context(context, bad, 50, 'SID_BHKM_B16D3_S42')

    def test_random_sid_permutation_preserves_leaf_occupancy(self):
        import numpy as np
        from vkgr.generative_diagnostics import permuted_sid_codec
        from vkgr import generative_data as gd
        with tmpdir() as tmp:
            context, _, _, _ = engineering_world(tmp)
            codec = context.codec
            perm = permuted_sid_codec(codec, 42)
            again = permuted_sid_codec(codec, 42)
            gd._check_codec(perm)
            self.assertTrue(np.array_equal(perm.catalog.sids, again.catalog.sids))
            self.assertTrue(np.array_equal(perm.catalog.ids, codec.catalog.ids))
            self.assertEqual(perm.catalog.bases, codec.catalog.bases)
            self.assertEqual(perm.vocab_size, codec.vocab_size)
            self.assertEqual(perm.catalog.audit()['leaf_occupancy_histogram'], codec.catalog.audit()['leaf_occupancy_histogram'])
            self.assertFalse(np.array_equal(perm.catalog.sids, codec.catalog.sids))
            for item in map(int, codec.catalog.ids):
                self.assertEqual(perm.catalog.decode(perm.catalog.encode(item)), item)
            self.assertFalse(np.array_equal(permuted_sid_codec(codec, 43).catalog.sids, perm.catalog.sids))


class RunTests(unittest.TestCase):
    def _run(self, tmp, **overrides):
        from vkgr import generative_b1_run as b1
        context, receipt, manifest, sha = engineering_world(tmp)
        cfg = config(**overrides)
        value = b1.run(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, cfg, Path(tmp) / 'run', temporary=Path(tmp) / 'scratch')
        return b1, receipt, manifest, sha, cfg, value

    def test_l50_run_best_last_and_verify(self):
        with tmpdir() as tmp:
            b1, receipt, manifest, sha, cfg, value = self._run(tmp)
            run = Path(tmp) / 'run'
            self.assertEqual(value['status'], 'complete')
            self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
            self.assertEqual(len(value['epochs']), 3)
            self.assertEqual(value['binding']['budget'], 'BUDGET_E100_MIN5_P3')
            self.assertIn('init_report', value['binding'])
            report = b1.verify(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, run / 'manifest.json', value['manifest_sha256'], Path(tmp) / 'verify', temporary=Path(tmp) / 'scratch-v')
            self.assertEqual(report['replayed_epochs'], [value['best']['epoch']])
            with self.assertRaises(ValueError):
                b1.verify(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, run / 'manifest.json', '0' * 64, Path(tmp) / 'verify2', temporary=Path(tmp) / 'scratch-v2')
            original = (run / 'best.pt').read_bytes(); (run / 'best.pt').write_bytes(original + b'0')
            with self.assertRaises(ValueError):
                b1.verify(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, run / 'manifest.json', value['manifest_sha256'], Path(tmp) / 'verify3', temporary=Path(tmp) / 'scratch-v3')

    def test_forged_selection_rejected(self):
        from vkgr.data_audit import file_sha256
        with tmpdir() as tmp:
            b1, receipt, manifest, sha, cfg, value = self._run(tmp)
            run = Path(tmp) / 'run'
            forged_value = json.loads((run / 'manifest.json').read_text())
            forged_value['epochs'][0]['metrics']['metrics']['all_eligible']['recall@50'] = 0.999
            forged = run / 'forged.json'; forged.write_text(json.dumps(forged_value))
            with self.assertRaises(ValueError):
                b1.verify(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, forged, file_sha256(forged), Path(tmp) / 'verify', temporary=Path(tmp) / 'scratch-v')

    def test_sharded_eval_matches_serial(self):
        import os
        from unittest import mock
        with tmpdir() as tmp:
            b1, receipt, manifest, sha, cfg, serial = self._run(tmp)
            with mock.patch.dict(os.environ, {'OMP_NUM_THREADS': '1'}):
                value = b1.run(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, cfg, Path(tmp) / 'run-w2', temporary=Path(tmp) / 'scratch2', eval_workers=2)
            self.assertEqual([e['evaluation']['computation_sha256'] for e in serial['epochs']], [e['evaluation']['computation_sha256'] for e in value['epochs']])

    def test_l100_and_random_sid_runs_verify(self):
        for overrides in ({'task_id': 'HIST-LEN', 'history_len': 100}, {'task_id': 'G4', 'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'}):
            with self.subTest(**overrides), tmpdir() as tmp:
                b1, receipt, manifest, sha, cfg, value = self._run(tmp, **overrides)
                self.assertEqual(value['status'], 'complete')
                b1.verify(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, Path(tmp) / 'run' / 'manifest.json', value['manifest_sha256'], Path(tmp) / 'verify', temporary=Path(tmp) / 'scratch-v')

    def test_beam_diagnostic_decode_only(self):
        from vkgr.generative_diagnostics import read_diagnostic
        with tmpdir() as tmp:
            b1, receipt, manifest, sha, cfg, value = self._run(tmp)
            for beam in (50, 200):
                result = b1.beam_diagnostic(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, Path(tmp) / 'run' / 'manifest.json', value['manifest_sha256'], beam, Path(tmp) / f'diag-{beam}', temporary=Path(tmp) / f'scratch-{beam}')
                diag = read_diagnostic(result['manifest_path'], result['manifest_sha256'])
                self.assertEqual(diag['kind'], 'beam_diagnostic')
                self.assertEqual(diag['beam'], beam)
                self.assertEqual(diag['query_count'], 7)
                self.assertIn('prefix_hit_rate', diag['summary'])
                self.assertIn('rescued_vs_beam100', diag['summary'])
                self.assertEqual(diag['optimizer_steps'], 0)
            with self.assertRaises(ValueError):
                b1.beam_diagnostic(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, Path(tmp) / 'run' / 'manifest.json', value['manifest_sha256'], 100, Path(tmp) / 'diag-100', temporary=Path(tmp) / 's100')

    def test_committed_batch1_configs_are_registered(self):
        from vkgr import generative_data as gd
        from vkgr.generative_b1_run import validate_config
        paths = sorted((ROOT / 'configs' / 'batch1').glob('g-*.json'))
        self.assertEqual(len(paths), 8)
        for path in paths:
            validate_config(gd._json(path))

    def test_cli_rejects_bad_worker_env_before_output(self):
        import contextlib
        import io
        import os
        from unittest import mock
        from vkgr import generative_b1_run as b1
        with tmpdir() as tmp, mock.patch.dict(os.environ, {'VKGR_EVAL_WORKERS': '0'}), contextlib.redirect_stderr(io.StringIO()):
            code = b1.main(['train', '--prepared', 'x', '--prepared-sha', '0' * 64, '--history-manifest', 'y', '--history-sha', '0' * 64,
                            '--config', 'z', '--config-sha', '0' * 64, '--output', str(Path(tmp) / 'out'), '--scratch', str(Path(tmp) / 's')])
            self.assertEqual(code, 1)
            self.assertFalse((Path(tmp) / 'out').exists())

    def test_config_registry(self):
        from vkgr.generative_b1_run import validate_config, default_config
        validate_config(default_config())
        for bad in ({'init': 'legacy'}, {'max_epochs': 30}, {'history_len': 30}, {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'}, {'beam': 200},
                    {'learning_rate': .01}, {'dropout': .1}, {'patience': 4}, {'x': 1}):
            with self.assertRaises(ValueError, msg=bad):
                validate_config({**default_config(), **bad})
        validate_config({**default_config(), 'task_id': 'G4', 'sid': 'SID_RANDPERM42_OF_BHKM_B16D3', 'seed': 2025})


if __name__ == '__main__':
    unittest.main()
