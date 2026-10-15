"""GTOK-02: G training entry for SIDMAP_V2S tokenizers.

Runs inside the frozen-base harness (the engineering G contract needs git base 0376d9a), OMP_NUM_THREADS=1:
    python -B -m unittest discover -s tests -p 'test_generative_*.py'
A green harness run does not prove the formal path: the engineering world only has six items and synthetic content. The formal path
is covered by tests/test_gtok02_formal_path.py and, on real inputs, by the server preflight.
The engineering world has 6 items. The six new labels get a mapping from the real frozen ``sid.fit`` at the label's own branching
(no registry patch). Only the legacy-equivalence and unbalanced bundles use the engineering-only replay label, whose registry entry is
patched in test scope (branching of the mapping under test; relaxed B1) exactly as tests/test_generative_tok.py does; src is never patched.
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

REPLAY = 'SID_V2S_REPLAY_BHKM_B16D3'
UNBALANCED = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [1, 1, 1, 0], [1, 1, 1, 1], [1, 2, 2, 0]], dtype=np.uint32)
ANCHOR = 'configs/batch1/g-hist-v2-l50-s42.json'
NEW_LABELS = ('SID_BHKM_B32D3_S42', 'SID_BHKM_B64D3_S42', 'SID_BHKM_B128D3_S42', 'SID_BHKM_B512D3_S42', 'SID_BHKM_B1024D3_S42',
              'SID_BHKM_B256D2_S42')
PARENT_SHA = 'd' * 64
PRODUCER = {'src/vkgr/sid_v2s.py': 'a' * 64}
CONFIG_SHA = 'c' * 64


def tok2_config(sid=REPLAY, **overrides):
    value = b1_config()
    value.update({'schema': 'g-tok2-run-v1', 'task_id': 'G-TOK-02', 'sid': sid})
    value.update(overrides)
    return value


def _lenient_balance(sids, branching, tolerance=None):
    return {'balanced': True, 'nodes': 0, 'violations': 0, 'max_spread': 0}


@contextlib.contextmanager
def replay_spec(branching):
    """Test scope only: the registry pins the replay label to branching 16 and B1 demands a balanced tree, which the 6-item world
    cannot satisfy. Both are relaxed identically for writing and loading a replay bundle; nothing in src is changed."""
    from vkgr import sid_tok_audit as au
    from vkgr import sid_v2s as s2
    with mock.patch.dict(s2.VERSION_SPEC, {REPLAY: ('bhkm', branching, 3)}), mock.patch.object(au, 'balanced_children', _lenient_balance):
        yield


class Bundle:
    def __init__(self, directory, branching, ids, sids, label):
        from vkgr.data_audit import file_sha256
        self.directory, self.branching, self.ids, self.sids, self.label = Path(directory), branching, ids, sids, label
        self.manifest = self.directory / 'run_manifest.json'
        self.sha = file_sha256(self.manifest)
        self.replay = label == REPLAY


class Fixture:
    """One engineering world plus bundles/runs, built lazily and cached for the whole module."""

    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        atexit.register(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name).resolve()
        (self.tmp / 'world').mkdir()
        self.context, self.receipt, self.hist_manifest, self.hist_sha = engineering_world(self.tmp / 'world')
        self.ids = np.asarray(self.context.codec.catalog.ids, dtype=np.uint32)
        raw = np.random.RandomState(7).randn(len(self.ids), 64)
        self.content = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)  # sid.fit wants unit rows
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

    def _write(self, directory, label, sids, tamper_label=None):
        from vkgr import sid_tok_audit as au
        from vkgr import sid_v2s as s2
        _, branching, depth = s2.VERSION_SPEC[label]
        parent_sha = PARENT_SHA if depth == 2 else None
        tokenizer = {'fit': {'branching': branching, 'depth': depth}}
        if depth == 2:
            tokenizer['parent_manifest_sha256'] = parent_sha
        audit = {'mapping': s2.sweep_audit(self.ids, sids, self.content, branching=branching, depth=depth),
                 'sweep': s2.sweep_section(label, True, parent_sha)}
        gates = au.gates(audit['mapping'], 'bhkm', None, expected_content_sha256=self.digest)
        if not gates['passed']:
            raise AssertionError(f'fixture mapping fails its own gates: {gates}')
        s2.write_bundle(directory, sid_version=label, engineering=True, ids=self.ids, sids=sids, content_feature_sha256=self.digest,
                        tokenizer=tokenizer if tamper_label is None else {**tokenizer, 'variant': tamper_label}, audit=audit,
                        gates=gates, fit_config_sha256=CONFIG_SHA, producer=PRODUCER)
        return branching

    def bundle(self, name):
        if name not in self._bundles:
            from vkgr import sid as bhkm
            from vkgr import sid_v2s as s2
            codec = self.context.codec
            directory = self.tmp / f'bundle-{name}'
            if name in ('legacy', 'twin'):  # the legacy mapping; "twin" differs only in its tokenizer record => another manifest sha
                sids, label = np.asarray(codec.catalog.sids, dtype=np.uint32), REPLAY
                branching = int(codec.catalog.branching)
                with replay_spec(branching):
                    self._write(directory, label, sids, tamper_label='twin' if name == 'twin' else None)
            elif name == 'unbalanced':
                sids, label, branching = UNBALANCED, REPLAY, 4
                with replay_spec(branching):
                    self._write(directory, label, sids)
            elif name in NEW_LABELS:
                label = name
                _, branching, depth = s2.VERSION_SPEC[label]
                if depth == 3:
                    sids = bhkm.fit(self.ids, self.content, branching=branching, depth=3, seed=42, max_iter=10).sids
                else:
                    parent = bhkm.fit(self.ids, self.content, branching=256, depth=3, seed=42, max_iter=10)
                    sids, _ = s2.derive_depth2_from_parent(self.ids, parent.sids)
                self._write(directory, label, sids)
            else:
                raise KeyError(name)
            self._bundles[name] = Bundle(directory, branching, self.ids, sids, label)
        return self._bundles[name]

    def spec(self, bundle):
        return replay_spec(bundle.branching) if bundle.replay else contextlib.nullcontext()

    def run_tok2(self, bundle, stem='run', cfg=None, decode=None, **kwargs):
        from vkgr import generative_tok2_run as tok2
        out = self.fresh(stem)
        with self.spec(bundle):
            return tok2.run(ROOT, self.receipt['manifest_path'], self.receipt['manifest_sha256'], self.hist_manifest, self.hist_sha,
                            bundle.manifest, bundle.sha, self.content_path, self.content_sha, cfg or tok2_config(bundle.label), out,
                            temporary=self.fresh(stem + '-scratch'), decode=decode, expected_content_digest=self.digest, **kwargs), out

    def verify_tok2(self, bundle, manifest_path, manifest_sha, decode=None, **kwargs):
        from vkgr import generative_tok2_run as tok2
        args = dict(sid_manifest=bundle.manifest, sid_sha=bundle.sha, content_path=self.content_path, content_sha=self.content_sha)
        args.update(kwargs)
        with self.spec(bundle):
            return tok2.verify(ROOT, self.receipt['manifest_path'], self.receipt['manifest_sha256'], self.hist_manifest, self.hist_sha,
                               args['sid_manifest'], args['sid_sha'], args['content_path'], args['content_sha'], manifest_path, manifest_sha,
                               self.fresh('verify'), temporary=self.fresh('verify-scratch'), decode=decode,
                               expected_content_digest=self.digest)

    def cached(self, name):
        """name -> (manifest value, run directory); each is trained once per module."""
        if name not in self._runs:
            if name == 'b1':
                from vkgr import generative_b1_run as b1
                out = self.fresh('b1')
                value = b1.run(ROOT, self.receipt['manifest_path'], self.receipt['manifest_sha256'], self.hist_manifest, self.hist_sha,
                               b1_config(), out, temporary=self.fresh('b1-scratch'))
                self._runs[name] = (value, out)
            elif name in ('legacy', 'unbalanced'):
                self._runs[name] = self.run_tok2(self.bundle(name), name, cfg=tok2_config(REPLAY))
            elif name == 'unbalanced-batched':
                from vkgr import generative_batched as gb
                with mock.patch.dict(os.environ, {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '1'}):
                    decode = gb.decode_mode_from_env()
                self._runs[name] = self.run_tok2(self.bundle('unbalanced'), 'unbalanced-batched', decode=decode)
            elif name in NEW_LABELS:
                self._runs[name] = self.run_tok2(self.bundle(name), name)
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
        b1_value, _ = f.cached('b1')
        tok_value, _ = f.cached('legacy')
        self.assertEqual(len(tok_value['epochs']), 3)
        self.assertEqual(per_epoch(tok_value), per_epoch(b1_value))
        self.assertEqual((tok_value['best']['epoch'], tok_value['best']['metric']), (b1_value['best']['epoch'], b1_value['best']['metric']))
        self.assertEqual(tok_value['selection'], b1_value['selection'])
        self.assertEqual(tok_value['binding']['sid']['vocab_size'], f.context.codec.vocab_size)
        self.assertEqual(tok_value['binding']['context_b1']['sid_sha256'], b1_value['binding']['context_b1']['sid_sha256'])

    def test_legacy_replay_run_verifies(self):
        f = fx()
        value, out = f.cached('legacy')
        report = f.verify_tok2(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])
        self.assertEqual((report['status'], report['replayed_epochs'], report['epochs']), ('verified', [value['best']['epoch']], 3))


class UnbalancedTests(unittest.TestCase):
    def test_unbalanced_sid_trains_and_verifies(self):
        f = fx()
        bundle = f.bundle('unbalanced')
        value, out = f.cached('unbalanced')
        self.assertEqual((value['status'], len(value['epochs'])), ('complete', 3))
        self.assertEqual(sorted(p.name for p in out.glob('*.pt')), ['best.pt', 'last.pt'])
        sid = value['binding']['sid']
        self.assertEqual((sid['branching'], sid['suffix_vocabulary'], sid['vocab_size'], sid['depth']), (4, 3, 2 + 3 * 4 + 3, 3))
        self.assertEqual((sid['sid_version'], sid['engineering']), (REPLAY, True))
        self.assertEqual((value['kind'], value['schema']), ('g_tok2_train', 'g-tok2-train-manifest-v1'))
        self.assertEqual(f.verify_tok2(bundle, out / 'manifest.json', value['manifest_sha256'])['status'], 'verified')

    def test_batched_decode_batch1_equals_serial_per_epoch(self):
        f = fx()
        serial, _ = f.cached('unbalanced')
        batched, out = f.cached('unbalanced-batched')
        self.assertEqual([e['evaluation']['computation_sha256'] for e in serial['epochs']],
                         [e['evaluation']['computation_sha256'] for e in batched['epochs']])
        self.assertEqual(per_epoch(serial), per_epoch(batched))
        self.assertEqual(batched['decode'], {'mode': 'batched', 'batch_size': 1})
        self.assertIsNone(serial['decode'])
        self.assertEqual(f.verify_tok2(f.bundle('unbalanced'), out / 'manifest.json', batched['manifest_sha256'])['status'], 'verified')


class NewLabelTests(unittest.TestCase):
    def test_each_new_label_trains_three_epochs_and_verifies(self):
        from vkgr import sid_v2s as s2
        f = fx()
        for label in NEW_LABELS:
            with self.subTest(label):
                bundle = f.bundle(label)
                value, out = f.cached(label)
                _, branching, depth = s2.VERSION_SPEC[label]
                sid = value['binding']['sid']
                self.assertEqual((value['status'], len(value['epochs'])), ('complete', 3))
                self.assertEqual((sid['sid_version'], sid['branching'], sid['depth'], sid['engineering'], sid['kind']),
                                 (label, branching, depth, True, 'bhkm'))
                self.assertEqual(sid['vocab_size'], 2 + 3 * branching + sid['suffix_vocabulary'])
                self.assertEqual(sid['manifest_sha256'], bundle.sha)
                self.assertEqual(value['config']['sid'], label)
                self.assertEqual(f.verify_tok2(bundle, out / 'manifest.json', value['manifest_sha256'])['status'], 'verified')

    def test_depth2_runs_on_the_padded_four_position_framework(self):
        f = fx()
        bundle = f.bundle('SID_BHKM_B256D2_S42')
        value, _ = f.cached('SID_BHKM_B256D2_S42')
        self.assertEqual(int(bundle.sids[:, 2].max()), 0)
        self.assertEqual(value['binding']['sid']['depth'], 2)
        self.assertEqual(value['binding']['sid']['vocab_size'], 2 + 3 * 256 + value['binding']['sid']['suffix_vocabulary'])

    def test_a_run_of_one_label_does_not_verify_against_another_labels_bundle(self):
        f = fx()
        value, out = f.copy_run('SID_BHKM_B32D3_S42')
        with self.assertRaisesRegex(ValueError, 'binding mismatch|differs from the bundle'):
            f.verify_tok2(f.bundle('SID_BHKM_B64D3_S42'), out / 'manifest.json', value['manifest_sha256'])


class SidTakesEffectTests(unittest.TestCase):
    def test_different_bundle_changes_binding_vocab_and_model_digest(self):
        f = fx()
        legacy, _ = f.cached('legacy')
        unbalanced, _ = f.cached('unbalanced')
        self.assertNotEqual(legacy['binding']['context_b1']['sid_sha256'], unbalanced['binding']['context_b1']['sid_sha256'])
        self.assertNotEqual(legacy['binding']['sid']['vocab_size'], unbalanced['binding']['sid']['vocab_size'])
        self.assertNotEqual(legacy['binding']['sid']['manifest_sha256'], unbalanced['binding']['sid']['manifest_sha256'])
        self.assertNotEqual(legacy['epochs'][0]['model_sha256'], unbalanced['epochs'][0]['model_sha256'])

    def test_context_uses_the_bundle_codec(self):
        from vkgr import generative_tok2_run as tok2
        f = fx()
        bundle = f.bundle('unbalanced')
        with replay_spec(bundle.branching):
            context, engineering, loaded = tok2._contexts(ROOT, f.receipt['manifest_path'], f.receipt['manifest_sha256'], f.hist_manifest,
                                                          f.hist_sha, bundle.manifest, bundle.sha, f.content_path, f.content_sha,
                                                          tok2_config(), f.fresh('ctx'), expected_content_digest=f.digest)
        self.assertTrue(engineering)
        self.assertTrue(np.array_equal(context.codec.catalog.sids, UNBALANCED))
        self.assertEqual(context.codec.vocab_size, 17)
        self.assertEqual(context.binding['tok']['sid_manifest_sha256'], bundle.sha)
        self.assertEqual(loaded.info['manifest_sha256'], bundle.sha)

    def test_binding_records_bundle_sha_depth_content_and_every_source_member(self):
        from vkgr import generative_tok2_run as tok2
        from vkgr.data_audit import file_sha256
        f = fx()
        bundle = f.bundle('legacy')
        value, _ = f.cached('legacy')
        self.assertEqual(value['binding']['task'], 'G-TOK-02')
        self.assertEqual(value['binding']['sid']['manifest_sha256'], file_sha256(bundle.manifest))
        self.assertEqual(value['binding']['sid']['content_feature_sha256'], f.digest)
        self.assertEqual(set(value['binding']['sid']), {'sid_version', 'manifest_sha256', 'depth', 'content_feature_sha256', 'branching',
                                                        'suffix_vocabulary', 'vocab_size', 'kind', 'engineering'})
        files = value['binding']['source']['files']
        self.assertEqual(sorted(files), sorted(tok2.TOK2_MEMBERS))
        for name, digest in files.items():
            self.assertEqual(digest, file_sha256(ROOT / name), name)
        self.assertEqual(value['binding']['config']['schema'], 'g-tok2-run-v1')

    def test_binding_function_carries_the_bundle_manifest_sha_and_depth(self):
        from vkgr import generative_tok2_run as tok2
        from vkgr.generative_hp import make_model_b1
        f = fx()
        bundle = f.bundle('SID_BHKM_B256D2_S42')
        config = tok2_config('SID_BHKM_B256D2_S42')
        context, engineering, loaded = tok2._contexts(ROOT, f.receipt['manifest_path'], f.receipt['manifest_sha256'], f.hist_manifest, f.hist_sha,
                                                      bundle.manifest, bundle.sha, f.content_path, f.content_sha, config, f.fresh('ctx'),
                                                      expected_content_digest=f.digest)
        model = make_model_b1(context.codec, config, device='cpu')
        binding = tok2._binding(ROOT, context, config, tok2._runtime(engineering), model, f.receipt['manifest_sha256'], f.hist_sha, loaded)
        self.assertEqual(binding['sid'].get('manifest_sha256'), bundle.sha)
        self.assertEqual(binding['sid'].get('depth'), 2)
        self.assertEqual(binding['task'], 'G-TOK-02')

    def test_members_extend_the_gtok01_members(self):
        from vkgr import generative_tok_run as tok
        from vkgr import generative_tok2_run as tok2
        self.assertEqual(tok2.TOK2_MEMBERS, tok.TOK_MEMBERS + ('src/vkgr/sid_v2s.py', 'src/vkgr/generative_tok2_run.py'))
        for name in tok2.TOK2_MEMBERS:
            self.assertTrue((ROOT / name).is_file(), name)

    def test_config_sid_must_equal_the_bundle_version(self):
        f = fx()
        with self.assertRaises(ValueError):
            f.run_tok2(f.bundle('legacy'), 'wrong-sid', cfg=tok2_config('SID_BHKM_B32D3_S42'))


class ConfigTests(unittest.TestCase):
    def anchor(self):
        return json.loads((ROOT / ANCHOR).read_text())

    def test_default_config_is_the_anchor_with_three_changes(self):
        from vkgr import generative_tok2_run as tok2
        default, anchor = tok2.default_config(), self.anchor()
        self.assertEqual({k for k in default if default[k] != anchor.get(k)}, {'schema', 'task_id', 'sid'})
        self.assertEqual((default['schema'], default['task_id']), ('g-tok2-run-v1', 'G-TOK-02'))
        self.assertEqual(set(default), set(anchor))

    def test_registry_rejections(self):
        from vkgr import generative_tok2_run as tok2
        from vkgr import sid_v2s as s2
        tok2.validate_config(tok2.default_config())
        for sid in s2.SID_VERSIONS:
            tok2.validate_config({**tok2.default_config(), 'sid': sid})
        bad_cases = ({'history_len': 20}, {'beam': 50}, {'sid': 'SID_BHKM_B256D3_S42'}, {'sid': 'SID_RQVAE_B256D3_S42'},
                     {'sid': 'SID_BHKM_B16D3_S42'}, {'sid': REPLAY}, {'task_id': 'G-TOK-01'}, {'schema': 'g-tok-run-v1'},
                     {'schema': 'g-b1-run-v1'}, {'init': 'legacy'}, {'max_epochs': 30}, {'learning_rate': .01}, {'patience': 4},
                     {'dropout': .1}, {'d_model': 64, 'ff': 256}, {'seed': 7}, {'microbatch': 64}, {'extra': 1})
        for bad in bad_cases:
            with self.subTest(bad), self.assertRaises(ValueError):
                tok2.validate_config({**tok2.default_config(), **bad})
        for seed in (2024, 2025):
            tok2.validate_config({**tok2.default_config(), 'seed': seed})

    def test_replay_label_only_in_engineering(self):
        from vkgr import generative_tok2_run as tok2
        engineering = tok2_config()
        tok2.validate_config(engineering, True)
        with self.assertRaises(ValueError):
            tok2.validate_config(engineering, False)
        tok2.validate_config(tok2_config('SID_BHKM_B256D2_S42'), True)
        for bad in ({'history_len': 100}, {'beam': 200}, {'sid': 'SID_BHKM_B256D3_S42'}, {'task_id': 'G-TOK-01'}, {'d_model': 32}):
            with self.subTest(bad), self.assertRaises(ValueError):
                tok2.validate_config({**engineering, **bad}, True)


class FailClosedTests(unittest.TestCase):
    def test_other_bundle_or_wrong_sha_is_rejected(self):
        f = fx()
        value, out = f.copy_run('legacy')
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            f.verify_tok2(f.bundle('unbalanced'), out / 'manifest.json', value['manifest_sha256'])
        with self.assertRaises(ValueError):
            f.verify_tok2(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'], sid_sha='0' * 64)

    def test_same_mapping_but_another_bundle_manifest_is_a_binding_mismatch(self):
        f = fx()
        legacy, twin = f.bundle('legacy'), f.bundle('twin')
        self.assertTrue(np.array_equal(legacy.sids, twin.sids))
        self.assertNotEqual(legacy.sha, twin.sha)
        value, out = f.copy_run('legacy')
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            f.verify_tok2(twin, out / 'manifest.json', value['manifest_sha256'])

    def test_source_drift_is_a_binding_mismatch(self):
        from vkgr import generative_tok2_run as tok2
        f = fx()
        value, out = f.copy_run('legacy')
        real = tok2._source

        def drifted(root):
            result = real(root)
            files = dict(result['files'])
            files['src/vkgr/sid_v2s.py'] = '0' * 64
            return {**result, 'files': files, 'source_sha256': '1' * 64}

        with mock.patch.object(tok2, '_source', drifted), self.assertRaisesRegex(ValueError, 'binding mismatch'):
            f.verify_tok2(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])
        self.assertEqual(f.verify_tok2(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])['status'], 'verified')

    def test_tampered_manifest_metrics_are_rejected(self):
        from vkgr.data_audit import file_sha256
        f = fx()
        value, out = f.copy_run('legacy')
        forged_value = json.loads((out / 'manifest.json').read_text())
        forged_value['epochs'][0]['metrics']['metrics']['all_eligible']['recall@50'] = 0.999
        forged = out / 'forged.json'
        forged.write_text(json.dumps(forged_value))
        with self.assertRaises(ValueError):
            f.verify_tok2(f.bundle('legacy'), forged, file_sha256(forged))

    def test_best_checkpoint_one_extra_byte_is_rejected(self):
        f = fx()
        value, out = f.copy_run('legacy')
        (out / 'best.pt').write_bytes((out / 'best.pt').read_bytes() + b'0')
        with self.assertRaisesRegex(ValueError, 'best checkpoint hash mismatch'):
            f.verify_tok2(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])

    def test_last_checkpoint_hash_is_checked(self):
        f = fx()
        value, out = f.copy_run('legacy')
        (out / 'last.pt').write_bytes((out / 'last.pt').read_bytes() + b'0')
        with self.assertRaisesRegex(ValueError, 'last checkpoint mismatch'):
            f.verify_tok2(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])

    def test_selection_is_replayed_from_the_published_epochs(self):
        from vkgr.data_audit import file_sha256
        f = fx()
        value, out = f.copy_run('legacy')
        forged_value = json.loads((out / 'manifest.json').read_text())
        forged_value['selection']['best_epoch'] = forged_value['selection']['best_epoch'] % 3 + 1
        forged = out / 'forged-selection.json'
        forged.write_text(json.dumps(forged_value))
        with self.assertRaisesRegex(ValueError, 'selection replay mismatch'):
            f.verify_tok2(f.bundle('legacy'), forged, file_sha256(forged))

    def test_best_checkpoint_hash_is_checked_even_when_the_file_still_loads(self):
        f = fx()
        value, out = f.copy_run('legacy')
        state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
        state['note'] = 1
        torch.save(state, out / 'best.pt')
        with self.assertRaisesRegex(ValueError, 'best checkpoint hash mismatch'):
            f.verify_tok2(f.bundle('legacy'), out / 'manifest.json', value['manifest_sha256'])

    def test_engineering_flag_mismatch_is_rejected(self):
        from vkgr import generative_tok2_run as tok2
        from vkgr import sid_v2s as s2
        from vkgr.data_audit import canonical_bytes, file_sha256
        f = fx()
        bundle = f.bundle('legacy')
        directory = f.fresh('flipped')
        shutil.copytree(bundle.directory, directory)
        manifest = json.loads((directory / 'run_manifest.json').read_text())
        manifest['engineering'] = False
        (directory / 'run_manifest.json').write_bytes(canonical_bytes(manifest))
        flipped = Bundle(directory, bundle.branching, bundle.ids, bundle.sids, REPLAY)
        with self.assertRaises(ValueError):
            f.run_tok2(flipped, 'flipped-run')
        catalog = tuple(int(x) for x in f.ids)
        with replay_spec(bundle.branching):
            with self.assertRaises(ValueError):  # formal context refuses an engineering bundle and a non-CONTENT_V1 digest
                tok2._load_sid(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog, False, s2.CONTENT_V1_FEATURE_SHA256)
            with self.assertRaises(ValueError):
                tok2._load_sid(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog, False, f.digest)
            loaded = tok2._load_sid(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog, True, f.digest)
        self.assertEqual(loaded.info['manifest_sha256'], file_sha256(bundle.manifest))

    def test_new_labels_are_accepted_by_the_formal_loader_gate_but_not_with_engineering_content(self):
        from vkgr import generative_tok2_run as tok2
        f = fx()
        bundle = f.bundle('SID_BHKM_B32D3_S42')
        catalog = tuple(int(x) for x in f.ids)
        with self.assertRaises(ValueError):  # engineering bundle in a formal context: refused
            tok2._load_sid(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog, False)

    def test_content_file_is_pinned_and_checked(self):
        from vkgr import generative_tok2_run as tok2
        from vkgr.data_audit import file_sha256
        f = fx()
        bundle = f.bundle('legacy')
        catalog = tuple(int(x) for x in f.ids)
        bad_rows = np.vstack([np.ones((1, 64), dtype=np.float32), f.content])
        short = np.vstack([np.zeros((1, 64), dtype=np.float32), f.content[:-1]])
        cases = {}
        for name, array in (('pad-row-not-zero', bad_rows), ('row-count', short), ('float64', np.vstack([np.zeros((1, 64)), f.content]))):
            path = f.fresh(name).with_suffix('.npy')
            np.save(path, array)
            cases[name] = (path, file_sha256(path))
        cases['sha'] = (f.content_path, '0' * 64)
        with replay_spec(bundle.branching):
            for name, (path, sha) in cases.items():
                with self.subTest(name), self.assertRaises(ValueError):
                    tok2._load_sid(bundle.manifest, bundle.sha, path, sha, catalog, True, f.digest)

    def test_b1_tok_and_tok2_manifests_refuse_each_other(self):
        from vkgr import generative_b1_run as b1
        from vkgr import generative_tok_run as tok
        import test_generative_tok as old_tests
        f = fx()
        tok2_value, tok2_out = f.cached('legacy')
        b1_value, b1_out = f.cached('b1')
        old = old_tests.fx()
        tok_value, tok_out = old.cached('legacy')
        with self.assertRaises(ValueError):
            b1.verify(ROOT, f.receipt['manifest_path'], f.receipt['manifest_sha256'], f.hist_manifest, f.hist_sha, tok2_out / 'manifest.json',
                      tok2_value['manifest_sha256'], f.fresh('x-verify'), temporary=f.fresh('x-scratch'))
        with self.assertRaises(ValueError):
            f.verify_tok2(f.bundle('legacy'), b1_out / 'manifest.json', b1_value['manifest_sha256'])
        with self.assertRaises(ValueError):
            f.verify_tok2(f.bundle('legacy'), tok_out / 'manifest.json', tok_value['manifest_sha256'])
        with self.assertRaises(ValueError):  # tok.verify (GTOK-01) given a GTOK-02 manifest
            with old_tests.replay_spec(old.bundle('legacy').branching):
                tok.verify(ROOT, old.receipt['manifest_path'], old.receipt['manifest_sha256'], old.hist_manifest, old.hist_sha,
                           old.bundle('legacy').manifest, old.bundle('legacy').sha, old.content_path, old.content_sha,
                           tok2_out / 'manifest.json', tok2_value['manifest_sha256'], old.fresh('y-verify'), temporary=old.fresh('y-scratch'),
                           expected_content_digest=old.digest)

    def test_fresh_manifest_has_the_tok2_schema_and_kind(self):
        from vkgr import generative_b1_run as b1
        from vkgr import generative_tok_run as tok
        f = fx()
        value, out = f.run_tok2(f.bundle('legacy'), 'one-epoch', cfg=tok2_config(max_epochs=1, min_epochs=1))
        self.assertEqual((value['schema'], value['kind']), ('g-tok2-train-manifest-v1', 'g_tok2_train'))
        on_disk = json.loads((out / 'manifest.json').read_text())
        self.assertEqual((on_disk['schema'], on_disk['kind']), ('g-tok2-train-manifest-v1', 'g_tok2_train'))
        self.assertEqual(torch.load(out / 'best.pt', map_location='cpu', weights_only=True)['schema'], 'g-tok2-checkpoint-v1')
        for refuse in (b1._read_run, tok._read_run):
            with self.assertRaises(ValueError):
                refuse(out / 'manifest.json', value['manifest_sha256'])


class MainTests(unittest.TestCase):
    def argv(self, command, f, bundle, out, **drop):
        args = {'--prepared': str(f.receipt['manifest_path']), '--prepared-sha': f.receipt['manifest_sha256'], '--history-manifest': str(f.hist_manifest),
                '--history-sha': f.hist_sha, '--sid-manifest': str(bundle.manifest), '--sid-sha': bundle.sha, '--content': str(f.content_path),
                '--content-sha': f.content_sha, '--output': str(out), '--scratch': str(out) + '-scratch'}
        if command == 'train':
            from vkgr.data_audit import file_sha256
            config = f.fresh('cfg').with_suffix('.json')
            config.write_text(json.dumps(tok2_config(), sort_keys=True))
            args.update({'--config': str(config), '--config-sha': file_sha256(config)})
        else:
            args.update({'--manifest': '/nonexistent/manifest.json', '--manifest-sha': '0' * 64})
        for key in drop.get('drop', ()):
            args.pop(key)
        return [command] + [part for item in args.items() for part in item]

    def call(self, argv, env):
        from vkgr import generative_tok2_run as tok2
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, env), contextlib.redirect_stderr(stderr):
            return tok2.main(argv), stderr.getvalue()

    def test_invalid_environment_exits_1_before_any_output(self):
        f = fx()
        bundle = f.bundle('legacy')
        for env in ({'VKGR_EVAL_WORKERS': '0'}, {'VKGR_EVAL_WORKERS': 'x'}, {'VKGR_DECODE': 'fast'}, {'VKGR_DECODE_BATCH': '4'},
                    {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '0'}, {'VKGR_DECODE': 'batched', 'VKGR_EVAL_WORKERS': '2'},
                    {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '2'}):
            for command in ('train', 'verify'):
                with self.subTest(env=env, command=command):
                    out = f.fresh('main-out')
                    clean = {k: v for k, v in os.environ.items() if not k.startswith('VKGR_')}
                    with mock.patch.dict(os.environ, clean, clear=True):
                        code, err = self.call(self.argv(command, f, bundle, out), env)
                    self.assertEqual(code, 1)
                    self.assertIn('G-TOK2 gate failed', err)
                    self.assertFalse(out.exists())
                    self.assertFalse(Path(str(out) + '-scratch').exists())

    def test_missing_sid_manifest_is_an_argument_error_before_any_output(self):
        f = fx()
        bundle = f.bundle('legacy')
        for drop in ('--sid-manifest', '--sid-sha', '--content', '--content-sha'):
            with self.subTest(drop):
                out = f.fresh('main-out')
                with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(io.StringIO()):
                    self.call(self.argv('train', f, bundle, out, drop=(drop,)), {})
                self.assertEqual(caught.exception.code, 2)
                self.assertFalse(out.exists())

    def test_engineering_world_cannot_pass_the_formal_content_pin_through_the_cli(self):
        f = fx()
        bundle = f.bundle('legacy')
        out = f.fresh('main-out')
        with replay_spec(bundle.branching):
            code, err = self.call(self.argv('train', f, bundle, out), {})
        self.assertEqual(code, 1)
        self.assertIn('G-TOK2 gate failed', err)
        self.assertFalse(out.exists())

    def test_help_exits_zero_for_both_subcommands(self):
        from vkgr import generative_tok2_run as tok2
        for command in ('train', 'verify'):
            with self.subTest(command), self.assertRaises(SystemExit) as caught, contextlib.redirect_stdout(io.StringIO()):
                tok2.main([command, '--help'])
            self.assertEqual(caught.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
