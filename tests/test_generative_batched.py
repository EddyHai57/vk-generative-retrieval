"""INFRA-02: batched beam decode must equal the frozen per-query decode; synthetic CPU fixtures only."""
import contextlib
import importlib.util
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from vkgr import generative_evaluation as ev
from vkgr import generative_experiment as ge
from vkgr import generative_training as gt
from vkgr import generative_sharded as gs
from vkgr.data_audit import file_sha256
from vkgr.tiger import decode_history
try:
    from vkgr import generative_batched as gb
except ImportError:
    gb = None
from test_generative_training import config

ROOT = Path(__file__).resolve().parents[1]
PIN = '40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d'
CK = 'a' * 64


def tmpdir():
    return tempfile.TemporaryDirectory(dir='/private/tmp')


def synthetic(base):
    fixture = ROOT / 'tests/fixtures/g2_synthetic.json'
    return ge.prepare_synthetic(ROOT, ROOT / 'artifacts/generative-contract-manifest.json', PIN, fixture, file_sha256(fixture), base / 'prepared')


def tiny(context, seed=42, scale=None):
    torch.manual_seed(seed); model = gt.make_model(context.codec, config(), device='cpu')
    if scale is not None:
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(scale)
    return model


def histories(context, count, max_len, seed):
    rng = random.Random(seed); items = list(context.catalog)
    return [[rng.choice(items) for _ in range(rng.randint(1, max_len))] for _ in range(count)]

class EnvTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gb, 'generative_batched missing')

    def test_decode_env_parsing(self):
        self.assertIsNone(gb.decode_mode_from_env({}))
        self.assertIsNone(gb.decode_mode_from_env({'VKGR_DECODE': ''}))
        self.assertIsNone(gb.decode_mode_from_env({'VKGR_DECODE': 'serial'}))
        self.assertEqual(gb.decode_mode_from_env({'VKGR_DECODE': 'batched'}), {'mode': 'batched', 'batch_size': gb.DEFAULT_BATCH})
        self.assertEqual(gb.decode_mode_from_env({'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '8'}), {'mode': 'batched', 'batch_size': 8})
        self.assertEqual(gb.decode_mode_from_env({'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '256'})['batch_size'], 256)
        bad = ({'VKGR_DECODE': 'gpu'}, {'VKGR_DECODE': 'Batched'}, {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '0'},
               {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '08'}, {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '257'},
               {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': 'x'}, {'VKGR_DECODE_BATCH': '8'}, {'VKGR_DECODE': 'serial', 'VKGR_DECODE_BATCH': '8'})
        for env in bad:
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, 'decode env'):
                gb.decode_mode_from_env(env)

    def test_b1_cli_rejects_bad_decode_env_before_output(self):
        from vkgr import generative_b1_run as b1
        argv = lambda tmp: ['train', '--prepared', 'x', '--prepared-sha', '0' * 64, '--history-manifest', 'y', '--history-sha', '0' * 64,
                            '--config', 'z', '--config-sha', '0' * 64, '--output', str(Path(tmp) / 'out'), '--scratch', str(Path(tmp) / 's')]
        for env in ({'VKGR_DECODE': 'gpu'}, {'VKGR_DECODE': 'batched', 'VKGR_EVAL_WORKERS': '2'}):
            with self.subTest(env=env), tmpdir() as tmp, patch.dict(os.environ, env), patch.object(gs, 'cpu_quota_cores', return_value=None):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    code = b1.main(argv(tmp))
                self.assertEqual(code, 1); self.assertIn('decode env', err.getvalue())
                self.assertFalse((Path(tmp) / 'out').exists())

class DecodeEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gb, 'generative_batched missing')

    def _check(self, model, tokens, hs, beam, serial):
        expected = [serial(h) for h in hs]
        for size in (1, 2, 3, 7, 64):
            with self.subTest(beam=beam, batch=size):
                got = gb.decode_histories_batched(model, tokens, hs, beam, size)
                self.assertEqual(got, expected)  # items (order, uniqueness, count) and full audit dict

    def test_random_histories_many_models_and_beams(self):
        with tmpdir() as d:
            context = synthetic(Path(d))
            hs = list(q['history'] for q in context.queries) + histories(context, 60, 50, 7)
            for seed, scale in ((42, None), (1, 4.), (2, .01)):
                model = tiny(context, seed, scale)
                for beam in (1, 2, 3, 100):
                    self._check(model, context.codec, hs, beam, lambda h: decode_history(model, context.codec, h, beam))

    def test_exact_ties_keep_path_tie_break(self):
        with tmpdir() as d:
            context = synthetic(Path(d)); model = tiny(context)
            with torch.no_grad():
                model.output.weight.zero_(); model.output.bias.zero_()
            hs = histories(context, 20, 50, 3)
            for beam in (1, 2, 100):
                self._check(model, context.codec, hs, beam, lambda h: decode_history(model, context.codec, h, beam))

    def test_l100_window_tokens(self):
        from vkgr.generative_data_v2 import SIDTokensL
        from vkgr.generative_diagnostics import decode_history_l
        from vkgr.generative_hp import make_model_b1
        from test_generative_b1 import config as b1_config
        with tmpdir() as d:
            context = synthetic(Path(d)); tokens = SIDTokensL(context.codec, 100)
            torch.manual_seed(5); model = make_model_b1(context.codec, b1_config(history_len=100, task_id='HIST-LEN'), device='cpu')
            hs = histories(context, 30, 100, 11)
            for beam in (3, 100):
                self._check(model, tokens, hs, beam, lambda h: decode_history_l(model, tokens, h, beam))

    def test_batch_size_validation(self):
        with tmpdir() as d:
            context = synthetic(Path(d)); model = tiny(context)
            for bad in (0, 257, 1.5, True):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    gb.decode_histories_batched(model, context.codec, [context.queries[0]['history']], 100, bad)

def large_codec(items=3000, branching=16, seed=0):
    """Synthetic real-scale SID catalog (beam=100 truncates at every position)."""
    import numpy as np
    from vkgr import generative_data as gd
    from vkgr.sid import SIDCatalog
    rng = np.random.RandomState(seed); ids = np.arange(10, 10 + items, dtype=np.uint32)
    prefixes = rng.randint(0, branching, size=(items, 3)).astype(np.uint32)
    order = np.lexsort((ids, prefixes[:, 2], prefixes[:, 1], prefixes[:, 0])); codes = prefixes[order]
    starts = np.r_[True, np.any(codes[1:] != codes[:-1], axis=1)]; rows = np.arange(items)
    sids = np.zeros((items, 4), dtype=np.uint32); sids[:, :3] = prefixes
    sids[order, 3] = (rows - np.maximum.accumulate(np.where(starts, rows, 0))).astype(np.uint32)
    return gd._ValidatedCodec(SIDCatalog(ids, sids, branching, 3, None))


class LargeCatalogTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gb, 'generative_batched missing')

    def test_real_scale_catalog_beam100_equal(self):
        codec = large_codec(); items = [int(i) for i in codec.catalog.ids]; rng = random.Random(9)
        hs = [[rng.choice(items) for _ in range(rng.choice((1, 2, 5, 50, 50, 50)))] for _ in range(40)]
        for seed in (0, 1):
            torch.manual_seed(seed); model = gt.make_model(codec, config(), device='cpu')
            expected = [decode_history(model, codec, h, 100) for h in hs]
            self.assertTrue(all(len(items) == 100 for items, _ in expected))
            for size in (1, 8, 40):
                with self.subTest(seed=seed, batch=size):
                    self.assertEqual(gb.decode_histories_batched(model, codec, hs, 100, size), expected)


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gb, 'generative_batched missing')

    def test_projection_equal_to_serial(self):
        with tmpdir() as d:
            base = Path(d); context = synthetic(base); model = tiny(context)
            for shard in (1, 3, 64):
                ref = ev.evaluate_queries(model, context, output=base / f's-{shard}', checkpoint_sha256=CK, shard_size=shard)
                for size in (1, 2, 64):
                    with self.subTest(shard=shard, batch=size):
                        calls = []
                        got = gb.evaluate_queries_batched(model, context, output=base / f'b-{shard}-{size}', checkpoint_sha256=CK, shard_size=shard,
                                                          batch_size=size, budget_check=lambda: calls.append(1))
                        self.assertEqual(gs.evaluation_projection(base / f's-{shard}'), gs.evaluation_projection(base / f'b-{shard}-{size}'))
                        self.assertEqual(got['manifest']['computation_sha256'], ref['manifest']['computation_sha256'])
                        self.assertEqual(got['observations']['execution']['decode'], 'batched')
                        self.assertEqual(got['observations']['execution']['batch_size'], size)
                        self.assertGreaterEqual(len(calls), 2)

    def test_cross_resume_and_stop(self):
        with tmpdir() as d:
            base = Path(d); context = synthetic(base); model = tiny(context)
            full = ev.evaluate_queries(model, context, output=base / 'full', checkpoint_sha256=CK, shard_size=3)['manifest']['computation_sha256']
            part = ev.evaluate_queries(model, context, output=base / 'p1', checkpoint_sha256=CK, shard_size=3, stop_after_shards=1)
            self.assertFalse(part['manifest']['complete'])
            done = gb.evaluate_queries_batched(model, context, output=base / 'r1', checkpoint_sha256=CK, shard_size=3, batch_size=4,
                                               resume_manifest=part['manifest_path'], resume_sha256=part['manifest_sha256'])
            self.assertEqual(done['manifest']['computation_sha256'], full)
            part = gb.evaluate_queries_batched(model, context, output=base / 'p2', checkpoint_sha256=CK, shard_size=3, batch_size=4, stop_after_shards=1)
            self.assertEqual(part['manifest']['completed_queries'], 3)
            done = ev.evaluate_queries(model, context, output=base / 'r2', checkpoint_sha256=CK, shard_size=3,
                                       resume_manifest=part['manifest_path'], resume_sha256=part['manifest_sha256'])
            self.assertEqual(done['manifest']['computation_sha256'], full)
            stopped = gb.evaluate_queries_batched(model, context, output=base / 'p3', checkpoint_sha256=CK, shard_size=3, batch_size=4, should_stop=lambda: True)
            self.assertEqual(stopped['manifest']['completed_queries'], 3)

    def test_decode_gate_failure_publishes_no_manifest(self):
        with tmpdir() as d:
            base = Path(d); context = synthetic(base); model = tiny(context)
            with patch.object(gb, 'decode_histories_batched', side_effect=lambda m, t, h, b, s: [([10 ** 9], {'invalid': 0, 'deadend': 0, 'collision': 0, 'max_active': 1})] * len(h)):
                with self.assertRaisesRegex(ValueError, 'decoded candidate/audit gate'):
                    gb.evaluate_queries_batched(model, context, output=base / 'bad', checkpoint_sha256=CK, batch_size=4)
            self.assertFalse((base / 'bad' / 'manifest.json').exists())

class B1IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gb, 'generative_batched missing')

    def _pair(self, tmp, **overrides):
        from vkgr import generative_b1_run as b1
        from test_generative_b1 import config as b1_config, engineering_world
        _, receipt, manifest, sha = engineering_world(tmp)
        cfg = b1_config(**overrides); common = (ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha)
        serial = b1.run(*common, cfg, Path(tmp) / 'run', temporary=Path(tmp) / 's1')
        batched = b1.run(*common, cfg, Path(tmp) / 'run-b', temporary=Path(tmp) / 's2', decode={'mode': 'batched', 'batch_size': 3})
        return b1, common, serial, batched

    def test_b1_batched_run_matches_serial_and_cross_verifies(self):
        for overrides in ({}, {'task_id': 'HIST-LEN', 'history_len': 100}):
            with self.subTest(**overrides), tmpdir() as tmp:
                b1, common, serial, batched = self._pair(tmp, **overrides)
                self.assertEqual([e['evaluation']['computation_sha256'] for e in serial['epochs']], [e['evaluation']['computation_sha256'] for e in batched['epochs']])
                self.assertEqual(serial['selection'], batched['selection'])
                self.assertTrue(all(e['observations']['execution']['decode'] == 'batched' for e in batched['epochs']))
                self.assertTrue(all('execution' not in e['observations'] for e in serial['epochs']))
                report = b1.verify(*common, Path(tmp) / 'run' / 'manifest.json', serial['manifest_sha256'], Path(tmp) / 'verify', temporary=Path(tmp) / 's3',
                                   decode={'mode': 'batched', 'batch_size': 2})
                self.assertEqual(report['status'], 'verified')

    def test_b1_rejects_batched_with_sharded_workers(self):
        from vkgr import generative_b1_run as b1
        with self.assertRaisesRegex(ValueError, 'decode env'):
            b1._evaluate(None, None, None, {'history_len': 50}, {}, Path('/nonexistent'), CK, None, 2, decode={'mode': 'batched', 'batch_size': 2})


class ParityScriptTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gb, 'generative_batched missing')

    def test_parity_script_reports_only_boolean_wall_and_vram(self):
        from vkgr import generative_b1_run as b1
        from test_generative_b1 import config as b1_config, engineering_world
        spec = importlib.util.spec_from_file_location('g_decode_parity', ROOT / 'scripts' / 'g_decode_parity.py')
        script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
        with tmpdir() as tmp:
            _, receipt, manifest, sha = engineering_world(tmp)
            value = b1.run(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, b1_config(), Path(tmp) / 'run', temporary=Path(tmp) / 's1')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = script.main(['--prepared', str(receipt['manifest_path']), '--prepared-sha', receipt['manifest_sha256'], '--history-manifest', str(manifest),
                                    '--history-sha', sha, '--manifest', str(Path(tmp) / 'run' / 'manifest.json'), '--manifest-sha', value['manifest_sha256'],
                                    '--scratch', str(Path(tmp) / 'scratch'), '--batch-size', '3'], root=ROOT)
            self.assertEqual(code, 0)
            report = json.loads(out.getvalue())
            self.assertEqual(set(report), {'equal', 'queries', 'batch_size', 'serial', 'batched'})
            self.assertIs(report['equal'], True); self.assertEqual(report['queries'], 7); self.assertEqual(report['batch_size'], 3)
            for side in ('serial', 'batched'):
                self.assertEqual(set(report[side]), {'wall_seconds', 'peak_vram_bytes'})
                self.assertIsNone(report[side]['peak_vram_bytes'])


if __name__ == '__main__':
    unittest.main()
