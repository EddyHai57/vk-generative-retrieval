"""GTOK-01 FEAT-004: G training entry for SIDMAP_V2 tokenizers.

Runs inside the frozen-base harness (the engineering G contract needs git base 0376d9a), OMP_NUM_THREADS=1:
    python -B -m unittest discover -s tests -p 'test_generative_*.py'
The engineering world has 6 items with a branching-3 mapping, while the registry pins SID_V2_REPLAY_BHKM_B16D3 to
branching 16 for bundles written by the tokenizer CLI. The tests therefore patch ``sid_v2.VERSION_SPEC`` (test scope
only) to the branching of the mapping under test; production code never does that.
"""
import atexit
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from test_generative_b1 import ROOT, config as b1_config, engineering_world

REPLAY = 'SID_V2_REPLAY_BHKM_B16D3'
# 6 engineering items (ids 10..60), branching 4, leaf sizes 3 / 2 / 1; canonical suffixes by ascending item id.
UNBALANCED = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [1, 1, 1, 0], [1, 1, 1, 1], [1, 2, 2, 0]], dtype=np.uint32)
ANCHOR = 'configs/batch1/g-hist-v2-l50-s42.json'
COMMITTED = {'g-tok-t1-rqvae-b256-s42.json': 'SID_RQVAE_B256D3_S42', 'g-tok-t2-rqvae-b16-s42.json': 'SID_RQVAE_B16D3_S42',
             'g-tok-t3-bhkm-b256-s42.json': 'SID_BHKM_B256D3_S42'}
TOK_EXTRA = ('src/vkgr/generative_batched.py', 'src/vkgr/generative_b1_batched.py', 'src/vkgr/sid_v2.py', 'src/vkgr/sid_rqvae.py',
             'src/vkgr/sid_tok_audit.py', 'src/vkgr/generative_tok_data.py', 'src/vkgr/generative_tok_run.py')


def tok_config(sid=REPLAY, **overrides):
    value = b1_config()
    value.update({'schema': 'g-tok-run-v1', 'task_id': 'G-TOK-01', 'sid': sid})
    value.update(overrides)
    return value


def _lenient_balance(sids, branching, tolerance=None):
    return {'balanced': True, 'nodes': 0, 'violations': 0, 'max_spread': 0}


@contextlib.contextmanager
def replay_spec(branching):
    """Test-scope only: the registry pins the REPLAY label to branching 16 and gate B1 demands a balanced k-means tree,
    which the 6-item engineering mapping cannot satisfy. Both are relaxed identically for writing and loading a bundle
    (B1 itself is covered by tests/test_sid_tok_audit.py); nothing in src is changed."""
    from vkgr import sid_tok_audit as au
    from vkgr import sid_v2
    with mock.patch.dict(sid_v2.VERSION_SPEC, {REPLAY: ('bhkm', branching)}), mock.patch.object(au, 'balanced_children', _lenient_balance):
        yield


class Bundle:
    def __init__(self, directory, branching, ids, sids):
        self.directory, self.branching, self.ids, self.sids = Path(directory), branching, ids, sids
        self.manifest = self.directory / 'run_manifest.json'
        from vkgr.data_audit import file_sha256
        self.sha = file_sha256(self.manifest)


class Fixture:
    """One engineering world plus bundles/runs, built lazily and cached for the whole module."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory(); atexit.register(self._tmp.cleanup)
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
            from test_sid_v2 import World
            codec = self.context.codec
            if name == 'legacy':
                sids, branching = np.asarray(codec.catalog.sids, dtype=np.uint32), int(codec.catalog.branching)
            elif name == 'unbalanced':
                sids, branching = UNBALANCED, 4
            elif name == 'twin':  # the legacy mapping again, but another tokenizer record => another manifest sha
                sids, branching = np.asarray(codec.catalog.sids, dtype=np.uint32), int(codec.catalog.branching)
            else:
                raise KeyError(name)
            with replay_spec(branching):
                world = World(REPLAY, 'bhkm', branching, True, self.ids, sids, self.content)
                self.assert_gates(world)
                world.write(self.tmp / f'bundle-{name}', **({'tokenizer': {'variant': 'twin'}} if name == 'twin' else {}))
            self._bundles[name] = Bundle(self.tmp / f'bundle-{name}', branching, self.ids, sids)
        return self._bundles[name]

    @staticmethod
    def assert_gates(world):
        if not world.gates['passed']:
            raise AssertionError(f'fixture mapping fails its own gates: {world.gates}')

    def run_tok(self, bundle, stem='run', cfg=None, decode=None, **kwargs):
        from vkgr import generative_tok_run as tok
        out = self.fresh(stem)
        with replay_spec(bundle.branching):
            return tok.run(ROOT, self.receipt['manifest_path'], self.receipt['manifest_sha256'], self.hist_manifest, self.hist_sha,
                           bundle.manifest, bundle.sha, self.content_path, self.content_sha, cfg or tok_config(), out,
                           temporary=self.fresh(stem + '-scratch'), decode=decode, expected_content_digest=self.digest, **kwargs), out

    def verify_tok(self, bundle, manifest_path, manifest_sha, decode=None, **kwargs):
        from vkgr import generative_tok_run as tok
        args = dict(sid_manifest=bundle.manifest, sid_sha=bundle.sha, content_path=self.content_path, content_sha=self.content_sha)
        args.update(kwargs)
        with replay_spec(bundle.branching):
            return tok.verify(ROOT, self.receipt['manifest_path'], self.receipt['manifest_sha256'], self.hist_manifest, self.hist_sha,
                              args['sid_manifest'], args['sid_sha'], args['content_path'], args['content_sha'], manifest_path, manifest_sha,
                              self.fresh('verify'), temporary=self.fresh('verify-scratch'), decode=decode, expected_content_digest=self.digest)

    def cached(self, name):
        """name -> (manifest value, run directory); each is trained once per module."""
        if name not in self._runs:
            if name == 'b1':
                from vkgr import generative_b1_run as b1
                out = self.fresh('b1')
                value = b1.run(ROOT, self.receipt['manifest_path'], self.receipt['manifest_sha256'], self.hist_manifest, self.hist_sha,
                               b1_config(), out, temporary=self.fresh('b1-scratch'))
                self._runs[name] = (value, out)
            elif name == 'legacy':
                self._runs[name] = self.run_tok(self.bundle('legacy'), 'legacy')
            elif name == 'unbalanced':
                self._runs[name] = self.run_tok(self.bundle('unbalanced'), 'unbalanced')
            elif name == 'unbalanced-batched':
                from vkgr import generative_batched as gb
                with mock.patch.dict(os.environ, {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '1'}):
                    decode = gb.decode_mode_from_env()
                self._runs[name] = self.run_tok(self.bundle('unbalanced'), 'unbalanced-batched', decode=decode)
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


class EquivalenceTests(unittest.TestCase):
    def test_replay_bundle_reproduces_the_default_runner_per_epoch(self):
        f = fx()
        b1_value, _ = f.cached('b1'); tok_value, _ = f.cached('legacy')
        self.assertEqual(len(tok_value['epochs']), 3)
        self.assertEqual(per_epoch(tok_value), per_epoch(b1_value))
        # checkpoint bytes differ (the checkpoint embeds the run binding); the selection outcome must not
        self.assertEqual((tok_value['best']['epoch'], tok_value['best']['metric']), (b1_value['best']['epoch'], b1_value['best']['metric']))
        self.assertEqual(tok_value['selection'], b1_value['selection'])
        self.assertEqual(tok_value['binding']['sid']['vocab_size'], f.context.codec.vocab_size)
        self.assertEqual(tok_value['binding']['context_b1']['sid_sha256'], b1_value['binding']['context_b1']['sid_sha256'])

    def test_legacy_replay_run_verifies(self):
        f = fx()
        value, out = f.cached('legacy')
        report = f.verify_tok(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])
        self.assertEqual((report['status'], report['replayed_epochs'], report['epochs']), ('verified', [value['best']['epoch']], 3))


class UnbalancedTests(unittest.TestCase):
    def test_unbalanced_sid_trains_and_verifies(self):
        f = fx(); bundle = f.bundle('unbalanced')
        value, out = f.cached('unbalanced')
        self.assertEqual(value['status'], 'complete'); self.assertEqual(len(value['epochs']), 3)
        self.assertEqual(sorted(p.name for p in out.glob('*.pt')), ['best.pt', 'last.pt'])
        sid = value['binding']['sid']
        self.assertEqual((sid['branching'], sid['suffix_vocabulary'], sid['vocab_size']), (4, 3, 2 + 3 * 4 + 3))
        self.assertEqual(sid['sid_version'], REPLAY); self.assertIs(sid['engineering'], True)
        self.assertEqual(value['kind'], 'g_tok_train'); self.assertEqual(value['schema'], 'g-tok-train-manifest-v1')
        report = f.verify_tok(bundle, out / 'manifest.json', value['manifest_sha256'])
        self.assertEqual(report['status'], 'verified')

    def test_batched_decode_batch1_equals_serial_per_epoch(self):
        f = fx()
        serial, _ = f.cached('unbalanced'); batched, out = f.cached('unbalanced-batched')
        self.assertEqual([e['evaluation']['computation_sha256'] for e in serial['epochs']],
                         [e['evaluation']['computation_sha256'] for e in batched['epochs']])
        self.assertEqual(per_epoch(serial), per_epoch(batched))
        self.assertEqual(batched['decode'], {'mode': 'batched', 'batch_size': 1}); self.assertIsNone(serial['decode'])
        # a batched-trained run is verified serially (VKGR_DECODE unset) as the server step S7 does
        report = f.verify_tok(f.bundle('unbalanced'), out / 'manifest.json', batched['manifest_sha256'])
        self.assertEqual(report['status'], 'verified')


class SidTakesEffectTests(unittest.TestCase):
    def test_different_bundle_changes_binding_vocab_and_model_digest(self):
        f = fx()
        legacy, _ = f.cached('legacy'); unbalanced, _ = f.cached('unbalanced')
        self.assertNotEqual(legacy['binding']['context_b1']['sid_sha256'], unbalanced['binding']['context_b1']['sid_sha256'])
        self.assertNotEqual(legacy['binding']['sid']['vocab_size'], unbalanced['binding']['sid']['vocab_size'])
        self.assertNotEqual(legacy['binding']['sid']['manifest_sha256'], unbalanced['binding']['sid']['manifest_sha256'])
        self.assertNotEqual(legacy['binding']['context_binding_sha256'], unbalanced['binding']['context_binding_sha256'])
        self.assertNotEqual(legacy['epochs'][0]['model_sha256'], unbalanced['epochs'][0]['model_sha256'])

    def test_context_uses_the_bundle_codec(self):
        from vkgr import generative_tok_run as tok
        f = fx(); bundle = f.bundle('unbalanced'); config = tok_config()
        with replay_spec(bundle.branching):
            context, engineering, loaded = tok._contexts(ROOT, f.receipt['manifest_path'], f.receipt['manifest_sha256'], f.hist_manifest, f.hist_sha,
                                                         bundle.manifest, bundle.sha, f.content_path, f.content_sha, config, f.fresh('ctx'),
                                                         expected_content_digest=f.digest)
        self.assertTrue(engineering)
        self.assertTrue(np.array_equal(context.codec.catalog.sids, UNBALANCED))
        self.assertEqual(context.codec.vocab_size, 17)
        self.assertEqual(context.binding['b1']['sid_version'], REPLAY)
        self.assertNotEqual(context.binding['b1']['sid_sha256'], context.binding['b1']['base_sid_sha256'])
        self.assertEqual(context.binding['tok']['sid_manifest_sha256'], bundle.sha)
        self.assertEqual(loaded.info['manifest_sha256'], bundle.sha)

    def test_binding_records_bundle_sha_content_and_every_source_member(self):
        from vkgr import generative_tok_run as tok
        from vkgr.data_audit import file_sha256
        f = fx(); bundle = f.bundle('legacy')
        value, _ = f.cached('legacy')
        self.assertEqual(value['binding']['task'], 'G-TOK-01')
        self.assertEqual(value['binding']['sid']['manifest_sha256'], file_sha256(bundle.manifest))
        self.assertEqual(value['binding']['sid']['content_feature_sha256'], f.digest)
        files = value['binding']['source']['files']
        self.assertEqual(sorted(files), sorted(tok.TOK_MEMBERS))
        for name, digest in files.items():
            self.assertEqual(digest, file_sha256(ROOT / name), name)
        self.assertEqual(value['binding']['config']['schema'], 'g-tok-run-v1')

    def test_binding_function_carries_the_bundle_manifest_sha(self):
        from vkgr import generative_tok_run as tok
        from vkgr.generative_hp import make_model_b1
        f = fx(); bundle = f.bundle('unbalanced'); config = tok_config()
        with replay_spec(bundle.branching):
            context, engineering, loaded = tok._contexts(ROOT, f.receipt['manifest_path'], f.receipt['manifest_sha256'], f.hist_manifest, f.hist_sha,
                                                         bundle.manifest, bundle.sha, f.content_path, f.content_sha, config, f.fresh('ctx'),
                                                         expected_content_digest=f.digest)
        model = make_model_b1(context.codec, config, device='cpu')
        binding = tok._binding(ROOT, context, config, tok._runtime(engineering), model, f.receipt['manifest_sha256'], f.hist_sha, loaded)
        self.assertEqual(binding['sid'].get('manifest_sha256'), bundle.sha)
        self.assertEqual(set(binding['sid']), {'sid_version', 'manifest_sha256', 'content_feature_sha256', 'branching', 'suffix_vocabulary',
                                               'vocab_size', 'kind', 'engineering'})

    def test_tok_members_extend_the_b1_members(self):
        from vkgr import generative_b1_run as b1
        from vkgr import generative_tok_run as tok
        self.assertEqual(tok.TOK_MEMBERS, b1.B1_MEMBERS + TOK_EXTRA)
        for name in tok.TOK_MEMBERS:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_config_sid_must_equal_the_bundle_version(self):
        f = fx(); bundle = f.bundle('legacy')
        with self.assertRaises(ValueError):
            f.run_tok(bundle, 'wrong-sid', cfg=tok_config(sid='SID_BHKM_B256D3_S42'))


class ConfigTests(unittest.TestCase):
    def anchor(self):
        return json.loads((ROOT / ANCHOR).read_text())

    def test_default_config_is_the_anchor_with_three_changes(self):
        from vkgr import generative_tok_run as tok
        default = tok.default_config(); anchor = self.anchor()
        self.assertEqual({k for k in default if default[k] != anchor.get(k)}, {'schema', 'task_id', 'sid'})
        self.assertEqual((default['schema'], default['task_id']), ('g-tok-run-v1', 'G-TOK-01'))
        self.assertEqual(set(default), set(anchor))

    def test_committed_configs_differ_from_the_anchor_only_in_schema_task_and_sid(self):
        from vkgr import generative_b1_run as b1
        from vkgr import generative_tok_run as tok
        anchor = self.anchor()
        paths = sorted((ROOT / 'configs' / 'gtok01').glob('g-tok-t*.json'))
        self.assertEqual([p.name for p in paths], sorted(COMMITTED))
        for path in paths:
            with self.subTest(path.name):
                text = path.read_text(); value = json.loads(text)
                self.assertEqual(text, json.dumps(value, indent=1, sort_keys=True) + '\n')
                self.assertEqual(value, {**anchor, 'schema': 'g-tok-run-v1', 'task_id': 'G-TOK-01', 'sid': COMMITTED[path.name]})
                self.assertEqual({k for k in value if value[k] != anchor[k]}, {'schema', 'task_id', 'sid'})
                tok.validate_config(value)
                with self.assertRaises(ValueError):
                    b1.validate_config(value)
        with self.assertRaises(ValueError):
            tok.validate_config(anchor)

    def test_registry_rejections(self):
        from vkgr import generative_tok_run as tok
        tok.validate_config(tok.default_config())
        for sid in COMMITTED.values():
            tok.validate_config({**tok.default_config(), 'sid': sid})
        bad_cases = ({'history_len': 20}, {'history_len': 100}, {'beam': 50}, {'beam': 200}, {'sid': 'SID_BHKM_B16D3_S42'},
                     {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'}, {'sid': REPLAY}, {'task_id': 'HIST-V2'}, {'schema': 'g-b1-run-v1'},
                     {'init': 'legacy'}, {'max_epochs': 30}, {'learning_rate': .01}, {'patience': 4}, {'dropout': .1}, {'d_model': 64, 'ff': 256},
                     {'seed': 7}, {'microbatch': 64}, {'extra': 1})
        for bad in bad_cases:
            with self.subTest(bad), self.assertRaises(ValueError):
                tok.validate_config({**tok.default_config(), **bad})
        missing = tok.default_config(); del missing['beam']
        with self.assertRaises(ValueError):
            tok.validate_config(missing)
        for seed in (2024, 2025):
            tok.validate_config({**tok.default_config(), 'seed': seed})

    def test_replay_label_only_in_engineering_and_engineering_bounds(self):
        from vkgr import generative_tok_run as tok
        engineering = tok_config()
        tok.validate_config(engineering, True)
        with self.assertRaises(ValueError):
            tok.validate_config(engineering, False)
        for bad in ({'history_len': 100}, {'beam': 200}, {'sid': 'SID_BHKM_B16D3_S42'}, {'schema': 'g-b1-run-v1'}, {'d_model': 32}):
            with self.subTest(bad), self.assertRaises(ValueError):
                tok.validate_config({**engineering, **bad}, True)


class FailClosedTests(unittest.TestCase):
    def test_other_bundle_or_wrong_sha_is_rejected(self):
        f = fx()
        value, out = f.copy_run('legacy')
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            f.verify_tok(f.bundle('unbalanced'), out / 'manifest.json', value['manifest_sha256'])
        with self.assertRaises(ValueError):
            f.verify_tok(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'], sid_sha='0' * 64)

    def test_same_mapping_but_another_bundle_manifest_is_a_binding_mismatch(self):
        f = fx(); legacy, twin = f.bundle('legacy'), f.bundle('twin')
        self.assertTrue(np.array_equal(legacy.sids, twin.sids)); self.assertNotEqual(legacy.sha, twin.sha)
        value, out = f.copy_run('legacy')
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            f.verify_tok(twin, out / 'manifest.json', value['manifest_sha256'])

    def test_source_drift_is_a_binding_mismatch(self):
        from vkgr import generative_tok_run as tok
        f = fx(); value, out = f.copy_run('legacy'); real = tok._source
        def drifted(root):
            result = real(root); files = dict(result['files']); files['src/vkgr/sid_v2.py'] = '0' * 64
            return {**result, 'files': files, 'source_sha256': '1' * 64}
        with mock.patch.object(tok, '_source', drifted), self.assertRaisesRegex(ValueError, 'binding mismatch'):
            f.verify_tok(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])
        self.assertEqual(f.verify_tok(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])['status'], 'verified')

    def test_tampered_manifest_metrics_are_rejected(self):
        from vkgr.data_audit import file_sha256
        f = fx(); value, out = f.copy_run('legacy')
        forged_value = json.loads((out / 'manifest.json').read_text())
        forged_value['epochs'][0]['metrics']['metrics']['all_eligible']['recall@50'] = 0.999
        forged = out / 'forged.json'; forged.write_text(json.dumps(forged_value))
        with self.assertRaises(ValueError):
            f.verify_tok(f.bundle('legacy'), forged, file_sha256(forged))

    def test_best_checkpoint_one_extra_byte_is_rejected(self):
        f = fx(); value, out = f.copy_run('legacy')
        (out / 'best.pt').write_bytes((out / 'best.pt').read_bytes() + b'0')
        with self.assertRaisesRegex(ValueError, 'best checkpoint hash mismatch'):
            f.verify_tok(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])

    def test_best_checkpoint_hash_is_checked_even_when_the_file_still_loads(self):
        f = fx(); value, out = f.copy_run('legacy')
        state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True); state['note'] = 1
        torch.save(state, out / 'best.pt')
        with self.assertRaisesRegex(ValueError, 'best checkpoint hash mismatch'):
            f.verify_tok(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])

    def test_engineering_flag_mismatch_is_rejected(self):
        from vkgr import generative_tok_run as tok
        from vkgr import sid_v2
        from vkgr.data_audit import file_sha256
        f = fx(); bundle = f.bundle('legacy')
        directory = f.fresh('flipped'); shutil.copytree(bundle.directory, directory)
        manifest = json.loads((directory / 'run_manifest.json').read_text()); manifest['engineering'] = False
        (directory / 'run_manifest.json').write_bytes(sid_v2.canonical_bytes(manifest))
        flipped = Bundle(directory, bundle.branching, bundle.ids, bundle.sids)
        with self.assertRaises(ValueError):
            f.run_tok(flipped, 'flipped-run')
        # formal context (engineering=False) refuses an engineering bundle and a non-CONTENT_V1 digest
        catalog = tuple(int(x) for x in f.ids)
        with replay_spec(bundle.branching):
            with self.assertRaises(ValueError):
                tok._load_sid(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog, False, sid_v2.CONTENT_V1_FEATURE_SHA256)
            with self.assertRaises(ValueError):
                tok._load_sid(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog, False, f.digest)
            loaded = tok._load_sid(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog, True, f.digest)
        self.assertEqual(loaded.info['manifest_sha256'], file_sha256(bundle.manifest))

    def test_content_file_is_pinned_and_checked(self):
        from vkgr import generative_tok_run as tok
        from vkgr.data_audit import file_sha256
        f = fx(); bundle = f.bundle('legacy'); catalog = tuple(int(x) for x in f.ids)
        bad_rows = np.vstack([np.ones((1, 64), dtype=np.float32), f.content]); short = np.vstack([np.zeros((1, 64), dtype=np.float32), f.content[:-1]])
        cases = {}
        for name, array in (('pad-row-not-zero', bad_rows), ('row-count', short), ('float64', np.vstack([np.zeros((1, 64)), f.content]))):
            path = f.fresh(name).with_suffix('.npy'); np.save(path, array); cases[name] = (path, file_sha256(path))
        cases['sha'] = (f.content_path, '0' * 64)
        with replay_spec(bundle.branching):
            for name, (path, sha) in cases.items():
                with self.subTest(name), self.assertRaises(ValueError):
                    tok._load_sid(bundle.manifest, bundle.sha, path, sha, catalog, True, f.digest)

    def test_b1_and_tok_manifests_refuse_each_other(self):
        from vkgr import generative_b1_run as b1
        f = fx()
        tok_value, tok_out = f.cached('legacy'); b1_value, b1_out = f.cached('b1')
        with self.assertRaises(ValueError):
            b1.verify(ROOT, f.receipt['manifest_path'], f.receipt['manifest_sha256'], f.hist_manifest, f.hist_sha, tok_out / 'manifest.json',
                      tok_value['manifest_sha256'], f.fresh('x-verify'), temporary=f.fresh('x-scratch'))
        with self.assertRaises(ValueError):
            f.verify_tok(f.bundle('legacy'), b1_out / 'manifest.json', b1_value['manifest_sha256'])

    def test_fresh_manifest_has_the_tok_schema_and_kind(self):
        from vkgr import generative_b1_run as b1
        f = fx()
        value, out = f.run_tok(f.bundle('legacy'), 'one-epoch', cfg=tok_config(max_epochs=1, min_epochs=1))
        self.assertEqual((value['schema'], value['kind']), ('g-tok-train-manifest-v1', 'g_tok_train'))
        on_disk = json.loads((out / 'manifest.json').read_text())
        self.assertEqual((on_disk['schema'], on_disk['kind']), ('g-tok-train-manifest-v1', 'g_tok_train'))
        self.assertEqual(torch.load(out / 'best.pt', map_location='cpu', weights_only=True)['schema'], 'g-tok-checkpoint-v1')
        with self.assertRaises(ValueError):
            b1._read_run(out / 'manifest.json', value['manifest_sha256'])


class MainTests(unittest.TestCase):
    def argv(self, command, f, bundle, out, **drop):
        args = {'--prepared': str(f.receipt['manifest_path']), '--prepared-sha': f.receipt['manifest_sha256'], '--history-manifest': str(f.hist_manifest),
                '--history-sha': f.hist_sha, '--sid-manifest': str(bundle.manifest), '--sid-sha': bundle.sha, '--content': str(f.content_path),
                '--content-sha': f.content_sha, '--output': str(out), '--scratch': str(out) + '-scratch'}
        if command == 'train':
            config = f.fresh('cfg').with_suffix('.json'); config.write_text(json.dumps(tok_config(), sort_keys=True))
            from vkgr.data_audit import file_sha256
            args.update({'--config': str(config), '--config-sha': file_sha256(config)})
        else:
            args.update({'--manifest': '/nonexistent/manifest.json', '--manifest-sha': '0' * 64})
        for key in drop.get('drop', ()):
            args.pop(key)
        return [command] + [part for item in args.items() for part in item]

    def call(self, argv, env):
        from vkgr import generative_tok_run as tok
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, env), contextlib.redirect_stderr(stderr):
            return tok.main(argv), stderr.getvalue()

    def test_invalid_environment_exits_1_before_any_output(self):
        f = fx(); bundle = f.bundle('legacy')
        for env in ({'VKGR_EVAL_WORKERS': '0'}, {'VKGR_EVAL_WORKERS': 'x'}, {'VKGR_DECODE': 'fast'}, {'VKGR_DECODE_BATCH': '4'},
                    {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '0'}, {'VKGR_DECODE': 'batched', 'VKGR_EVAL_WORKERS': '2'}):
            for command in ('train', 'verify'):
                with self.subTest(env=env, command=command):
                    out = f.fresh('main-out')
                    clean = {k: v for k, v in os.environ.items() if not k.startswith('VKGR_')}
                    with mock.patch.dict(os.environ, clean, clear=True):
                        code, err = self.call(self.argv(command, f, bundle, out), env)
                    self.assertEqual(code, 1); self.assertIn('G-TOK gate failed', err)
                    self.assertFalse(out.exists()); self.assertFalse(Path(str(out) + '-scratch').exists())

    def test_missing_sid_manifest_is_an_argument_error_before_any_output(self):
        f = fx(); bundle = f.bundle('legacy')
        for drop in ('--sid-manifest', '--sid-sha', '--content', '--content-sha'):
            with self.subTest(drop):
                out = f.fresh('main-out')
                with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
                    self.call(self.argv('train', f, bundle, out, drop=(drop,)), {})
                self.assertEqual(caught.exception.code, 2); self.assertFalse(out.exists())

    def test_engineering_world_cannot_pass_the_formal_content_pin_through_the_cli(self):
        f = fx(); bundle = f.bundle('legacy'); out = f.fresh('main-out')
        with replay_spec(bundle.branching):
            code, err = self.call(self.argv('train', f, bundle, out), {})
        self.assertEqual(code, 1); self.assertIn('G-TOK gate failed', err); self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
