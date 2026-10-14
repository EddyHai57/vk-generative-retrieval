"""G6-PREP step 7b: A1 training entry on the engineering world (depth-3 replay equals the arch runner; depth 4 trains, verifies, fails closed).

Runs inside the frozen-base harness (the engineering G contract needs git base 0376d9a), OMP_NUM_THREADS=1:
    python -B -m unittest discover -s tests -p 'test_generative_g6*.py'
The engineering world has 6 items. The registry pins the labels to branching 16, which the 6-item mappings cannot satisfy, so the tests
patch ``sid_v2d.VERSION_SPEC`` / ``prereg_for`` and the balance gate (test scope only, for writing and loading a bundle alike);
production code never does that.
"""
import atexit
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from test_generative_b1 import ROOT, engineering_world
from test_generative_g6_unit import g6_config

REPLAY = 'SID_V2D_REPLAY_BHKM_B16D3'
D4 = 'SID_BHKM_B16D4_S42'
# 6 engineering items (ids 10..60): leaf sizes 3 / 2 / 1, canonical suffixes by ascending item id, 4 semantic digits, branching 4.
UNBALANCED4 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 2, 2, 2, 0]], dtype=np.uint32)


def arch_config(**overrides):
    """The same engineering structure on the GARCH-01 schema (the frozen arch runner is the depth-3 reference)."""
    from vkgr import generative_arch as ga
    value = g6_config(**overrides)
    value.update({'schema': ga.SCHEMA, 'task_id': ga.TASK_ID, 'sid': 'SID_BHKM_B16D3_S42'})
    return value


def _lenient_balance(sids, branching, depth, tolerance=None):
    return {'balanced': True, 'nodes': 0, 'violations': 0, 'max_spread': 0}


@contextlib.contextmanager
def toy_spec(label, branching, depth):
    from vkgr import sid_tok_audit_d as ad
    from vkgr import sid_v2d
    with mock.patch.dict(sid_v2d.VERSION_SPEC, {label: ('bhkm', branching, depth)}), mock.patch.object(ad, 'balanced_children_d', _lenient_balance), \
            mock.patch.object(sid_v2d, 'prereg_for', lambda *args, **kwargs: None):
        yield


class Bundle:
    def __init__(self, directory, label, branching, depth):
        from vkgr.data_audit import file_sha256
        self.directory, self.label, self.branching, self.depth = Path(directory), label, branching, depth
        self.manifest = self.directory / 'run_manifest.json'
        self.sha = file_sha256(self.manifest)


class Fixture:
    """One engineering world plus bundles/runs, built lazily and cached for the whole module."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        atexit.register(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        (self.tmp / 'world').mkdir()
        self.context, self.receipt, self.hist_manifest, self.hist_sha = engineering_world(self.tmp / 'world')
        self.ids = np.asarray(self.context.codec.catalog.ids, dtype=np.uint32)
        self.content = np.random.RandomState(7).randn(len(self.ids), 64).astype(np.float32)
        from vkgr import sid_tok_audit as au
        from vkgr.data_audit import file_sha256
        self.digest = au.feature_digest(self.content)
        self.content_path = self.tmp / 'content.npy'
        np.save(self.content_path, np.vstack([np.zeros((1, 64), dtype=np.float32), self.content]))
        self.content_sha = file_sha256(self.content_path)
        self._bundles, self._runs, self._counter = {}, {}, 0

    def fresh(self, stem):
        self._counter += 1
        return self.tmp / f'{stem}-{self._counter}'

    def bundle(self, name):
        if name not in self._bundles:
            from vkgr import sid_tok_audit_d as ad
            from vkgr import sid_v2d
            codec = self.context.codec
            if name == 'legacy3':
                label, sids, branching, depth, record = REPLAY, np.asarray(codec.catalog.sids, dtype=np.uint32), int(codec.catalog.branching), 3, 'legacy3'
            elif name in ('unbalanced4', 'twin4'):  # twin4: the same mapping under another tokenizer record => another manifest sha
                label, sids, branching, depth, record = D4, UNBALANCED4, 4, 4, name
            else:
                raise KeyError(name)
            with toy_spec(label, branching, depth):
                mapping = ad.audit_mapping_d(self.ids, sids, self.content, branching=branching, kind='bhkm', depth=depth)
                gates = ad.gates_d(mapping, 'bhkm', None, depth=depth, expected_content_sha256=self.digest)
                if not gates['passed']:
                    raise AssertionError(f'fixture mapping fails its own gates: {gates}')
                sid_v2d.write_bundle_d(self.tmp / f'bundle-{name}', sid_version=label, engineering=True, ids=self.ids, sids=sids,
                                       content_feature_sha256=self.digest, tokenizer={'variant': record}, audit={'mapping': mapping, 'rq': None},
                                       gates=gates, fit_config_sha256='c' * 64, producer={'src/vkgr/synthetic.py': 'a' * 64})
            self._bundles[name] = Bundle(self.tmp / f'bundle-{name}', label, branching, depth)
        return self._bundles[name]

    def common(self):
        return (ROOT, self.receipt['manifest_path'], self.receipt['manifest_sha256'], self.hist_manifest, self.hist_sha)

    def run_g6(self, bundle, stem='run', cfg=None, decode=None, **kwargs):
        from vkgr import generative_g6_run as g6
        out = self.fresh(stem)
        with toy_spec(bundle.label, bundle.branching, bundle.depth):
            return g6.run(*self.common(), bundle.manifest, bundle.sha, self.content_path, self.content_sha, cfg or g6_config(bundle.label), out,
                          temporary=self.fresh(stem + '-scratch'), decode=decode, expected_content_digest=self.digest, **kwargs), out

    def verify_g6(self, bundle, manifest_path, manifest_sha, decode=None, **kwargs):
        from vkgr import generative_g6_run as g6
        with toy_spec(bundle.label, bundle.branching, bundle.depth):
            return g6.verify(*self.common(), bundle.manifest, bundle.sha, self.content_path, self.content_sha, manifest_path, manifest_sha,
                             self.fresh('verify'), temporary=self.fresh('verify-scratch'), decode=decode, expected_content_digest=self.digest, **kwargs)

    def cached(self, name):
        """name -> (manifest value, run directory); each is trained once per module."""
        if name not in self._runs:
            if name == 'arch':
                from vkgr import generative_b1_arch as ba
                out = self.fresh('arch')
                self._runs[name] = (ba.run(*self.common(), arch_config(), out, temporary=self.fresh('arch-scratch')), out)
            elif name == 'legacy3':
                self._runs[name] = self.run_g6(self.bundle('legacy3'), 'legacy3')
            elif name == 'd4':
                self._runs[name] = self.run_g6(self.bundle('unbalanced4'), 'd4')
            elif name == 'd4-batched':
                from vkgr import generative_batched as gb
                with mock.patch.dict(os.environ, {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '1'}):
                    decode = gb.decode_mode_from_env()
                self._runs[name] = self.run_g6(self.bundle('unbalanced4'), 'd4-batched', decode=decode)
            else:
                raise KeyError(name)
        return self._runs[name]

    def copy_run(self, name):
        value, out = self.cached(name)
        target = self.fresh('copy')
        shutil.copytree(out, target)
        return value, target


_FIXTURE = []


def fx():
    if not _FIXTURE:
        _FIXTURE.append(Fixture())
    return _FIXTURE[0]


def per_epoch(value):
    return [(e['model_sha256'], e['evaluation']['computation_sha256'], e['metrics']) for e in value['epochs']], value['losses']


class DepthThreeEquivalence(unittest.TestCase):
    def test_replay_bundle_reproduces_the_arch_runner_per_epoch(self):
        f = fx()
        arch, _ = f.cached('arch')
        g6, _ = f.cached('legacy3')
        self.assertEqual(len(g6['epochs']), 3)
        self.assertEqual(per_epoch(g6), per_epoch(arch))
        self.assertEqual((g6['best']['epoch'], g6['best']['metric']), (arch['best']['epoch'], arch['best']['metric']))
        self.assertEqual(g6['selection'], arch['selection'])
        self.assertEqual(g6['binding']['context_b1']['sid_sha256'], arch['binding']['context_b1']['sid_sha256'])
        self.assertEqual(g6['binding']['init_report']['arch'], arch['binding']['init_report']['arch'])
        self.assertEqual(g6['binding']['init_report']['arch']['spec']['positions'], 200)
        self.assertEqual(g6['binding']['sid']['depth'], 3)

    def test_depth_three_run_verifies(self):
        f = fx()
        value, out = f.cached('legacy3')
        report = f.verify_g6(f.bundle('legacy3'), out / 'manifest.json', value['manifest_sha256'])
        self.assertEqual((report['status'], report['replayed_epochs'], report['epochs']), ('verified', [value['best']['epoch']], 3))

    def test_runner_restores_the_frozen_decoder_names(self):
        from vkgr import generative_batched as gb
        from vkgr import tiger
        before = (tiger.decode_history, gb.decode_histories_batched)
        fx().cached('legacy3')
        self.assertEqual((tiger.decode_history, gb.decode_histories_batched), before)
        f = fx()
        with self.assertRaises(ValueError):
            f.run_g6(f.bundle('legacy3'), 'bad', g6_config(REPLAY, history_len=100))
        self.assertEqual((tiger.decode_history, gb.decode_histories_batched), before)


class DepthFour(unittest.TestCase):
    def test_depth_four_trains_with_a_250_row_position_table_and_five_positions(self):
        f = fx()
        value, out = f.cached('d4')
        self.assertEqual((value['status'], len(value['epochs'])), ('complete', 3))
        self.assertEqual(sorted(p.name for p in out.glob('*.pt')), ['best.pt', 'last.pt'])
        sid = value['binding']['sid']
        self.assertEqual((sid['depth'], sid['branching'], sid['suffix_vocabulary'], sid['vocab_size'], sid['sid_version']), (4, 4, 3, 2 + 4 * 4 + 3, D4))
        self.assertIs(sid['engineering'], True)
        self.assertEqual(value['binding']['init_report']['arch']['spec']['positions'], 250)
        self.assertEqual(value['kind'], 'g_g6_train')
        self.assertEqual(value['schema'], 'g-g6-train-manifest-v1')
        self.assertEqual(value['binding']['task'], 'G-HP-01-G6')
        state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
        self.assertEqual(tuple(state['model']['position.weight'].shape), (250, 64))
        self.assertEqual(state['schema'], 'g-g6-checkpoint-v1')
        self.assertTrue(all(np.isfinite(row['loss_mean']) for row in value['losses']))
        for epoch in value['epochs']:
            self.assertEqual(epoch['metrics']['candidate_counts']['max'], 6)  # the catalog has 6 items: the 5-step beam returns them all

    def test_depth_four_run_verifies_serially(self):
        f = fx()
        value, out = f.cached('d4')
        report = f.verify_g6(f.bundle('unbalanced4'), out / 'manifest.json', value['manifest_sha256'])
        self.assertEqual(report['status'], 'verified')

    def test_batched_decode_with_batch_one_equals_the_serial_run_and_serial_verify_passes(self):
        f = fx()
        serial, _ = f.cached('d4')
        batched, out = f.cached('d4-batched')
        self.assertEqual([e['model_sha256'] for e in serial['epochs']], [e['model_sha256'] for e in batched['epochs']])
        self.assertEqual([e['evaluation']['computation_sha256'] for e in serial['epochs']], [e['evaluation']['computation_sha256'] for e in batched['epochs']])
        self.assertEqual(serial['selection'], batched['selection'])
        self.assertTrue(all(e['observations']['execution']['decode'] == 'batched' for e in batched['epochs']))
        self.assertTrue(all('execution' not in e['observations'] for e in serial['epochs']))
        self.assertEqual(f.verify_g6(f.bundle('unbalanced4'), out / 'manifest.json', batched['manifest_sha256'])['status'], 'verified')

    def test_the_frozen_depth_three_decoder_cannot_serve_a_depth_four_catalog(self):
        from vkgr import generative_evaluation as ev
        f = fx()
        value, out = f.cached('d4')
        state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
        self.assertIn('model', state)
        # the swap is what makes depth four evaluable: with the frozen names the 4-step beam cannot decode a 5-digit SID
        from vkgr import generative_g6_model as gm
        from vkgr import sid_v2d
        from vkgr import tiger
        bundle = f.bundle('unbalanced4')
        with toy_spec(bundle.label, bundle.branching, bundle.depth):
            loaded = sid_v2d.load_bundle_d(bundle.manifest, bundle.sha, catalog_ids=f.ids, content=f.content, expected_content_digest=f.digest, engineering=True)
        model = gm.make_model_arch_d(loaded.codec, g6_config(D4), device='cpu')
        model.load_state_dict(state['model'])
        history = [int(f.ids[0])]
        self.assertRaises(ValueError, tiger.decode_history, model, loaded.codec, history, 20)
        self.assertEqual(len(ev.model_digest(model)), 64)


class FreshRuns(unittest.TestCase):
    """One-epoch runs trained inside the test (never cached), so that they exercise the code as it is when the test executes."""

    def one_epoch(self, **extra):
        return g6_config(D4, max_epochs=1, min_epochs=1, **extra)

    def test_depth_four_run_records_its_identity_and_verifies(self):
        f = fx()
        bundle = f.bundle('unbalanced4')
        value, out = f.run_g6(bundle, 'fresh4', self.one_epoch())
        self.assertEqual((value['schema'], value['kind'], value['status']), ('g-g6-train-manifest-v1', 'g_g6_train', 'complete'))
        self.assertEqual(value['binding']['task'], 'G-HP-01-G6')
        sid = value['binding']['sid']
        self.assertEqual((sid['depth'], sid['vocab_size'], sid['sid_version']), (4, 21, D4))
        self.assertEqual(value['binding']['init_report']['arch']['spec']['positions'], 250)
        self.assertIn('src/vkgr/generative_g6_run.py', value['binding']['source']['files'])
        state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
        self.assertEqual((state['schema'], tuple(state['model']['position.weight'].shape)), ('g-g6-checkpoint-v1', (250, 64)))
        self.assertTrue(all(np.isfinite(row['loss_mean']) for row in value['losses']))
        self.assertEqual(f.verify_g6(bundle, out / 'manifest.json', value['manifest_sha256'])['status'], 'verified')

    def test_depth_three_run_equals_the_arch_runner(self):
        from vkgr import generative_b1_arch as ba
        f = fx()
        arch = ba.run(*f.common(), arch_config(max_epochs=1, min_epochs=1), f.fresh('fresh-arch'), temporary=f.fresh('fresh-arch-scratch'))
        replay, _ = f.run_g6(f.bundle('legacy3'), 'fresh3', g6_config(REPLAY, max_epochs=1, min_epochs=1))
        self.assertEqual(per_epoch(replay), per_epoch(arch))

    def test_depth_decode_swaps_and_restores_the_two_frozen_names(self):
        from vkgr import generative_batched as gb
        from vkgr import generative_g6_run as g6
        from vkgr import sid_depth_beam as sdb
        from vkgr import tiger
        before = (tiger.decode_history, gb.decode_histories_batched)
        with g6.depth_decode():
            self.assertIs(tiger.decode_history, sdb.decode_history_d)
            self.assertIs(gb.decode_histories_batched, sdb.decode_histories_batched_d)
        self.assertEqual((tiger.decode_history, gb.decode_histories_batched), before)
        with self.assertRaises(RuntimeError), g6.depth_decode():
            raise RuntimeError('boom')
        self.assertEqual((tiger.decode_history, gb.decode_histories_batched), before)

    def test_gates_are_checked_by_the_runner_itself(self):
        from vkgr import generative_g6_run as g6
        for decode, workers in ((None, 2), ({'mode': 'batched', 'batch_size': 2}, 1), ({'mode': 'serial', 'batch_size': 1}, 1)):
            self.assertRaises(ValueError, g6._gate, decode, workers)
        g6._gate(None, 1)
        g6._gate({'mode': 'batched', 'batch_size': 1}, 1)


class FailClosed(unittest.TestCase):
    def test_runs_of_other_entries_do_not_cross_verify(self):
        from vkgr import generative_b1_arch as ba
        f = fx()
        arch, arch_out = f.cached('arch')
        g6, g6_out = f.cached('legacy3')
        with self.assertRaises(ValueError):
            ba.verify(*f.common(), g6_out / 'manifest.json', g6['manifest_sha256'], f.fresh('x1'), temporary=f.fresh('s1'))
        with self.assertRaises(ValueError):
            f.verify_g6(f.bundle('legacy3'), arch_out / 'manifest.json', arch['manifest_sha256'])

    def test_a_different_or_shallower_bundle_is_refused_at_verify(self):
        f = fx()
        value, out = f.cached('d4')
        with self.assertRaises(ValueError):
            f.verify_g6(f.bundle('twin4'), out / 'manifest.json', value['manifest_sha256'])
        with self.assertRaises(ValueError):
            f.verify_g6(f.bundle('legacy3'), out / 'manifest.json', value['manifest_sha256'])

    def test_tampered_manifest_wrong_pin_and_tampered_checkpoint(self):
        from vkgr.data_audit import file_sha256
        f = fx()
        value, run = f.copy_run('d4')
        bundle = f.bundle('unbalanced4')
        with self.assertRaises(ValueError):
            f.verify_g6(bundle, run / 'manifest.json', '0' * 64)
        forged_value = json.loads((run / 'manifest.json').read_text())
        forged_value['epochs'][0]['metrics']['metrics']['all_eligible']['recall@50'] = 0.999
        forged = run / 'forged.json'
        forged.write_text(json.dumps(forged_value))
        with self.assertRaises(ValueError):
            f.verify_g6(bundle, forged, file_sha256(forged))
        (run / 'best.pt').write_bytes((run / 'best.pt').read_bytes() + b'0')
        with self.assertRaises(ValueError):
            f.verify_g6(bundle, run / 'manifest.json', value['manifest_sha256'])

    def test_a_forged_binding_in_the_manifest_is_refused(self):
        from vkgr.data_audit import file_sha256
        f = fx()
        value, run = f.copy_run('d4')
        forged_value = json.loads((run / 'manifest.json').read_text())
        forged_value['binding']['sid']['depth'] = 3  # the checkpoint still carries the true binding; only the manifest lies
        forged = run / 'forged-binding.json'
        forged.write_text(json.dumps(forged_value))
        with self.assertRaisesRegex(ValueError, 'binding'):
            f.verify_g6(f.bundle('unbalanced4'), forged, file_sha256(forged))

    def test_gates_refuse_before_any_output(self):
        f = fx()
        bundle = f.bundle('unbalanced4')
        for label, kwargs in (('workers 2', {'eval_workers': 2}), ('batch 2', {'decode': {'mode': 'batched', 'batch_size': 2}})):
            out = f.fresh('gate')
            with self.subTest(label), toy_spec(bundle.label, bundle.branching, bundle.depth), self.assertRaises(ValueError):
                from vkgr import generative_g6_run as g6
                g6.run(*f.common(), bundle.manifest, bundle.sha, f.content_path, f.content_sha, g6_config(D4), out, temporary=f.fresh('gs'),
                       expected_content_digest=f.digest, **kwargs)
            self.assertFalse(out.exists())

    def test_config_and_bundle_must_agree(self):
        f = fx()
        for label, bundle, overrides in (('replay label on a depth-4 bundle', f.bundle('unbalanced4'), {'sid': REPLAY}),
                                         ('depth-4 label on a depth-3 bundle', f.bundle('legacy3'), {'sid': D4}),
                                         ('history 100', f.bundle('unbalanced4'), {'history_len': 100}), ('beam 200', f.bundle('unbalanced4'), {'beam': 200}),
                                         ('arch schema', f.bundle('unbalanced4'), {'schema': 'g-b1-arch-run-v1'})):
            out_before = set(f.tmp.iterdir())
            with self.subTest(label), self.assertRaises(ValueError):
                f.run_g6(bundle, 'cfg', g6_config(**{'sid': bundle.label, **overrides}))
            self.assertEqual([p for p in set(f.tmp.iterdir()) - out_before if re.fullmatch(r'cfg-\d+', p.name)], [])  # no output directory

    def test_cli_refuses_bad_environment_before_writing(self):
        from vkgr import generative_g6_run as g6
        f = fx()
        bundle = f.bundle('unbalanced4')
        cfg = f.tmp / 'cfg.json'
        cfg.write_text(json.dumps(g6_config(D4), indent=1, sort_keys=True) + '\n')
        from vkgr.data_audit import file_sha256
        for env in ({'VKGR_EVAL_WORKERS': '2'}, {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '2'}, {'VKGR_DECODE_BATCH': '1'}):
            out = f.fresh('cli')
            argv = ['train', '--prepared', str(f.receipt['manifest_path']), '--prepared-sha', f.receipt['manifest_sha256'], '--history-manifest', str(f.hist_manifest),
                    '--history-sha', f.hist_sha, '--sid-manifest', str(bundle.manifest), '--sid-sha', bundle.sha, '--content', str(f.content_path),
                    '--content-sha', f.content_sha, '--config', str(cfg), '--config-sha', file_sha256(cfg), '--output', str(out), '--scratch', str(f.fresh('cs'))]
            clean = {k: v for k, v in os.environ.items() if not k.startswith('VKGR_')}
            clean.update(env)
            stderr = io.StringIO()
            with self.subTest(env), mock.patch.dict(os.environ, clean, clear=True), contextlib.redirect_stderr(stderr):
                self.assertEqual(g6.main(argv), 1)
            self.assertIn('G6 gate failed', stderr.getvalue())
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
