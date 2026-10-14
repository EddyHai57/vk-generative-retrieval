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
        from vkgr import generative_b1_batched as b1
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


class _NearTieModel(torch.nn.Module):
    """Stub decoder (log_softmax patched to identity): first digit 0/1 both score 1.0; under first digit 1 the
    second step adds 2**-24, under 0 it adds 0. float64 totals rank first-digit-1 paths strictly above (1+2**-24 > 1);
    float32 totals round 1+2**-24 to 1.0 and the (-score, item) tie-break would rank by item id instead."""
    def __init__(self, offsets, vocab):
        super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(1)); self.offsets = offsets; self.vocab = vocab

    def encode(self, source):
        return torch.zeros(source.shape[0], source.shape[1], 1), torch.zeros(source.shape, dtype=torch.bool)

    def decode(self, memory, padding, decoder):
        rows, width = decoder.shape; out = torch.zeros(rows, width, self.vocab, dtype=torch.float32)
        if width == 1:
            out[:, -1, :] = 1.
        elif width == 2:
            out[(decoder[:, 1] - self.offsets[0]) == 1, -1, :] = 2. ** -24
        return out


class Float64AccumulationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gb, 'generative_batched missing')

    def test_ulp_near_tie_needs_float64_totals(self):
        codec = large_codec(items=16, branching=2, seed=3); catalog = codec.catalog
        first = {int(i): int(s[0]) for i, s in zip(catalog.ids, catalog.sids)}
        self.assertEqual(set(first.values()), {0, 1})
        float64_order = sorted(first, key=lambda i: (-first[i], i))
        self.assertNotEqual(float64_order, sorted(first))  # float32 (all totals 1.0) would give plain id order
        model = _NearTieModel(codec.offsets, codec.vocab_size); hs = [[10], [11, 12], [13]]
        with patch.object(torch, 'log_softmax', lambda x, dim=-1: x):
            serial = [decode_history(model, codec, h, 100) for h in hs]
            self.assertTrue(all(items == float64_order for items, _ in serial))
            for size in (1, 3):
                with self.subTest(batch=size):
                    self.assertEqual(gb.decode_histories_batched(model, codec, hs, 100, size), serial)


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
        from vkgr import generative_b1_batched as b1
        from test_generative_b1 import config as b1_config, engineering_world
        _, receipt, manifest, sha = engineering_world(tmp)
        cfg = b1_config(**overrides); common = (ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha)
        serial = b1.run(*common, cfg, Path(tmp) / 'run', temporary=Path(tmp) / 's1')
        batched = b1.run(*common, cfg, Path(tmp) / 'run-b', temporary=Path(tmp) / 's2', decode={'mode': 'batched', 'batch_size': 3})
        return b1, common, serial, batched

    def test_default_runner_source_identity_is_6f3901d(self):
        """MEDIUM-4: the default runner keeps the 6f3901d member set and bytes; only the opt-in entry binds the batched code."""
        import subprocess
        from vkgr import generative_b1_run as b1
        from vkgr import generative_b1_batched as bb
        names = ('history_v2', 'generative_data_v2', 'generative_hp', 'generative_diagnostics', 'generative_b1_run',
                 'generative_sharded', 'generative_real', 'generative_training', 'generative_evaluation', 'generative_experiment', 'generative_data', 'tiger', 'sid')
        self.assertEqual(b1.B1_MEMBERS, tuple(f'src/vkgr/{n}.py' for n in names))
        try:
            blob = subprocess.run(['git', '-C', str(ROOT), 'show', '6f3901d:src/vkgr/generative_b1_run.py'], capture_output=True, check=True).stdout
        except (subprocess.CalledProcessError, OSError):
            self.skipTest('6f3901d not in this checkout')
        self.assertEqual((ROOT / 'src/vkgr/generative_b1_run.py').read_bytes(), blob)
        self.assertEqual(bb.MEMBERS, b1.B1_MEMBERS + ('src/vkgr/generative_batched.py', 'src/vkgr/generative_b1_batched.py'))

    def test_entry_points_do_not_cross_verify_and_restore_globals(self):
        from vkgr import generative_b1_run as b1
        from vkgr import generative_b1_batched as bb
        from test_generative_b1 import config as b1_config, engineering_world
        members, evaluate = b1.B1_MEMBERS, b1._evaluate
        with tmpdir() as tmp:
            _, receipt, manifest, sha = engineering_world(tmp); common = (ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha)
            plain = b1.run(*common, b1_config(), Path(tmp) / 'plain', temporary=Path(tmp) / 's1')
            wrapped = bb.run(*common, b1_config(), Path(tmp) / 'wrapped', temporary=Path(tmp) / 's2')
            self.assertIs(b1.B1_MEMBERS, members); self.assertIs(b1._evaluate, evaluate)
            self.assertNotIn('src/vkgr/generative_batched.py', plain['binding']['source']['files'])
            self.assertIn('src/vkgr/generative_batched.py', wrapped['binding']['source']['files'])
            self.assertEqual([e['evaluation']['computation_sha256'] for e in plain['epochs']], [e['evaluation']['computation_sha256'] for e in wrapped['epochs']])
            with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                bb.verify(*common, Path(tmp) / 'plain' / 'manifest.json', plain['manifest_sha256'], Path(tmp) / 'v1', temporary=Path(tmp) / 's3')
            with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                b1.verify(*common, Path(tmp) / 'wrapped' / 'manifest.json', wrapped['manifest_sha256'], Path(tmp) / 'v2', temporary=Path(tmp) / 's4')
            self.assertIs(b1.B1_MEMBERS, members); self.assertIs(b1._evaluate, evaluate)

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

    def test_l100_batched_runs_guard_between_decode_blocks(self):
        from vkgr import generative_b1_run as b1
        from vkgr.generative_data_v2 import SIDTokensL
        from vkgr.generative_diagnostics import evaluate_l100
        from vkgr.generative_hp import make_model_b1
        from test_generative_b1 import config as b1_config, engineering_world
        with tmpdir() as tmp:
            _, receipt, manifest, sha = engineering_world(tmp); cfg = b1_config(task_id='HIST-LEN', history_len=100)
            context, _ = b1._contexts(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, cfg, Path(tmp) / 's')
            tokens = SIDTokensL(context.codec, 100); torch.manual_seed(4); model = make_model_b1(context.codec, cfg, device='cpu')
            ref = evaluate_l100(model, context, tokens, output=Path(tmp) / 'serial', checkpoint_sha256=CK)
            n = len(context.queries); self.assertGreater(n, 2)
            events = []; real = gb.decode_histories_batched
            def spy(m, t, h, b, s):
                events.append(('decode', len(h))); return real(m, t, h, b, s)
            with patch.object(gb, '_BLOCK_FACTOR', 1), patch.object(gb, 'decode_histories_batched', side_effect=spy):
                got = gb.evaluate_l100_batched(model, context, tokens, output=Path(tmp) / 'batched', checkpoint_sha256=CK, batch_size=2,
                                               guard=lambda: events.append(('guard',)))
            decodes = [e for e in events if e[0] == 'decode']
            self.assertEqual(len(decodes), -(-n // 2)); self.assertEqual(sum(e[1] for e in decodes), n)
            for i, e in enumerate(events):  # every decode block is bracketed by guard calls
                if e[0] == 'decode':
                    self.assertEqual(events[i - 1], ('guard',)); self.assertEqual(events[i + 1], ('guard',))
            self.assertEqual(got['manifest']['computation_sha256'], ref['manifest']['computation_sha256'])
            self.assertEqual(got['manifest']['observations']['execution']['block_queries'], 2)

    def test_b1_rejects_batched_with_sharded_workers(self):
        from vkgr import generative_b1_batched as bb
        with self.assertRaisesRegex(ValueError, 'decode env'):
            bb._evaluator({'mode': 'batched', 'batch_size': 2})(None, None, None, {'history_len': 50}, {}, Path('/nonexistent'), CK, None, 2)


class ParityScriptTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gb, 'generative_batched missing')

    def _world(self, tmp, entry='plain', **overrides):
        from vkgr import generative_b1_run as b1
        from vkgr import generative_b1_batched as bb
        from test_generative_b1 import config as b1_config, engineering_world
        spec = importlib.util.spec_from_file_location('g_decode_parity', ROOT / 'scripts' / 'g_decode_parity.py')
        script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
        _, receipt, manifest, sha = engineering_world(tmp)
        runner = b1 if entry == 'plain' else bb
        value = runner.run(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, b1_config(**overrides), Path(tmp) / 'run', temporary=Path(tmp) / 's1')
        def call(checkpoint, scratch):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = script.main(['--prepared', str(receipt['manifest_path']), '--prepared-sha', receipt['manifest_sha256'], '--history-manifest', str(manifest),
                                    '--history-sha', sha, '--manifest', str(Path(tmp) / 'run' / 'manifest.json'), '--manifest-sha', value['manifest_sha256'],
                                    '--scratch', str(Path(tmp) / scratch), '--batch-size', '3', '--checkpoint', checkpoint], root=ROOT)
            return code, out.getvalue(), err.getvalue()
        return value, call

    def _report(self, code, out, err, checkpoint):
        self.assertEqual(code, 0, err)
        report = json.loads(out)
        self.assertEqual(set(report), {'equal', 'queries', 'batch_size', 'checkpoint', 'epoch', 'serial', 'batched'})
        self.assertEqual(report['queries'], 7); self.assertEqual(report['batch_size'], 3); self.assertEqual(report['checkpoint'], checkpoint)
        for side in ('serial', 'batched'):
            self.assertEqual(set(report[side]), {'wall_seconds', 'peak_vram_bytes'})
            self.assertIsNone(report[side]['peak_vram_bytes'])
        return report

    def test_parity_script_best_and_last_report_only_boolean_wall_and_vram(self):
        for entry, overrides in (('plain', {}), ('batched-entry', {}), ('plain', {'task_id': 'HIST-LEN', 'history_len': 100})):
            with self.subTest(entry=entry, **overrides), tmpdir() as tmp:
                value, call = self._world(tmp, entry, **overrides)
                for checkpoint, epoch in (('best', value['best']['epoch']), ('last', value['last']['epoch'])):
                    report = self._report(*call(checkpoint, f'scratch-{checkpoint}'), checkpoint)
                    self.assertIs(report['equal'], True); self.assertEqual(report['epoch'], epoch)

    def test_parity_script_loads_the_requested_checkpoint(self):
        from vkgr import generative_evaluation as gev
        with tmpdir() as tmp:
            value, call = self._world(tmp)
            records = {r['epoch']: r for r in value['epochs']}
            for checkpoint in ('best', 'last'):
                seen = []; real = gb.decode_histories_batched
                def spy(m, t, h, b, s):
                    seen.append(gev.model_digest(m)); return real(m, t, h, b, s)
                with self.subTest(checkpoint=checkpoint), patch.object(gb, 'decode_histories_batched', side_effect=spy):
                    self._report(*call(checkpoint, f'scratch-{checkpoint}'), checkpoint)
                    self.assertEqual(seen, [records[value[checkpoint]['epoch']]['model_sha256']])
            (Path(tmp) / 'run' / 'last.pt').write_bytes(b'tampered')
            code, out, err = call('last', 'scratch-tampered')
            self.assertEqual(code, 1); self.assertEqual(out, ''); self.assertIn('last checkpoint', err)

    def test_parity_script_reports_false_on_any_difference(self):
        """MEDIUM-1: a batched result that differs from serial in order, membership or audit must give equal=false (exit 0)."""
        mutations = {'order': lambda items, audit: (items[1:2] + items[:1] + items[2:], audit),
                     'drop': lambda items, audit: (items[:-1], audit),
                     'audit': lambda items, audit: (items, {**audit, 'max_active': audit['max_active'] + 1})}
        with tmpdir() as tmp:
            _, call = self._world(tmp); real = gb.decode_histories_batched
            for name, mutate in mutations.items():
                def fake(m, t, h, b, s, mutate=mutate):
                    out = real(m, t, h, b, s); out[-1] = mutate(*out[-1]); return out
                with self.subTest(mutation=name), patch.object(gb, 'decode_histories_batched', side_effect=fake):
                    report = self._report(*call('best', f'scratch-{name}'), 'best')
                    self.assertIs(report['equal'], False)

    def test_parity_diag_reports_only_counts_and_magnitudes(self):
        from test_generative_b1 import config as b1_config, engineering_world
        from vkgr import generative_b1_batched as bb
        spec = importlib.util.spec_from_file_location('g_decode_parity_diag', ROOT / 'scripts' / 'g_decode_parity_diag.py')
        diag = importlib.util.module_from_spec(spec); spec.loader.exec_module(diag)
        with tmpdir() as tmp:
            _, receipt, manifest, sha = engineering_world(tmp)
            value = bb.run(ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha, b1_config(), Path(tmp) / 'run', temporary=Path(tmp) / 's1')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = diag.main(['--prepared', str(receipt['manifest_path']), '--prepared-sha', receipt['manifest_sha256'], '--history-manifest', str(manifest),
                                  '--history-sha', sha, '--manifest', str(Path(tmp) / 'run' / 'manifest.json'), '--manifest-sha', value['manifest_sha256'],
                                  '--scratch', str(Path(tmp) / 'scratch'), '--batch-size', '3', '--checkpoint', 'last'], root=ROOT)
            self.assertEqual(code, 0)
            report = json.loads(out.getvalue())
            self.assertEqual(set(report), {'checkpoint', 'queries', 'batch_size', 'variants', 'wall_seconds', 'logit_probe'})
            self.assertEqual(set(report['variants']), {'serial_repeat', 'batched_b1', 'serial_copy_memory', 'batched_b3'})
            self.assertTrue(all(v['differing_queries'] == 0 for v in report['variants'].values()))
            self.assertNotIn('items', out.getvalue()); self.assertNotIn('recall', out.getvalue())

    def test_parity_script_requires_checkpoint_choice(self):
        with tmpdir() as tmp:
            _, call = self._world(tmp)
            for bad in ('', 'epoch-1', 'BEST'):
                with self.subTest(bad=bad), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    call(bad, f'scratch-bad-{bad}')


if __name__ == '__main__':
    unittest.main()
