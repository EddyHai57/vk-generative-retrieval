"""GTOK-01B: the G6 runner's explicitly registered depth-3 SIDMAP_V2 path (T3 = SID_BHKM_B256D3_S42, loaded with sid_v2.load_bundle).

Runs inside the frozen-base harness (the engineering G contract needs git base 0376d9a), OMP_NUM_THREADS=1:
    python -B -m unittest discover -s tests -p 'test_generative_g6*.py'
``ConfigV2Tests`` is the field-by-field assertion of configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json against the A1 s42 config and can be run
alone (from the repo root: ``python -B -m unittest discover -s tests -p 'test_generative_g6_v2.py' -k ConfigV2Tests``).
The cached engineering world, the arch run and the SIDMAP_V2D legacy3 run are shared with ``test_generative_g6``. The registry pins the V2
labels to branching 16 / 256 and gate B1 demands a balanced k-means tree, which the 6-item mapping cannot satisfy, so the tests patch
``sid_v2.VERSION_SPEC`` and the balance gate (test scope only, for writing and loading a bundle alike); production code never does that.
"""
import contextlib
import io
import json
import os
import re
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import test_generative_g6 as world
from test_generative_b1 import ROOT
from test_generative_g6 import arch_config, per_epoch
from test_generative_g6_unit import g6_config

T3 = 'SID_BHKM_B256D3_S42'
V2_REPLAY = 'SID_V2_REPLAY_BHKM_B16D3'
V2D_REPLAY = world.REPLAY
A1_CONFIG = ROOT / 'configs' / 'garch01' / 'g-garch01-a1-lr3e-4-s42.json'
G6_CONFIG = ROOT / 'configs' / 'g6prep' / 'g-g6-a1-bhkm-b16d4-s42.json'
NEW_CONFIG = ROOT / 'configs' / 'gtok01b' / 'g-gtok-a1-t3-b256d3-s42.json'
OLD_CONFIG = ROOT / 'configs' / 'gtok01a' / 'g-gtok-a1-t3-b256d3-s42.json'


def _lenient_balance(sids, branching, tolerance=None):
    return {'balanced': True, 'nodes': 0, 'violations': 0, 'max_spread': 0}


@contextlib.contextmanager
def v2_spec(label, branching):
    from vkgr import sid_tok_audit as au
    from vkgr import sid_v2
    with mock.patch.dict(sid_v2.VERSION_SPEC, {label: ('bhkm', branching)}), mock.patch.object(au, 'balanced_children', _lenient_balance):
        yield


class V2Fixture:
    """SIDMAP_V2 bundles of the shared engineering world's own depth-3 mapping, plus cached V2 runs."""

    def __init__(self):
        self._bundles, self._runs = {}, {}

    def bundle(self, name):
        if name not in self._bundles:
            from test_sid_v2 import World
            f = world.fx()
            codec = f.context.codec
            sids, branching = np.asarray(codec.catalog.sids, dtype=np.uint32), int(codec.catalog.branching)
            label, engineering = {'replay': (V2_REPLAY, True), 'formal-t3': (T3, False)}[name]
            directory = f.fresh(f'bundle-v2-{name}')
            with v2_spec(label, branching):
                bundle_world = World(label, 'bhkm', branching, engineering, f.ids, sids, f.content)
                if not bundle_world.gates['passed']:
                    raise AssertionError(f'fixture mapping fails its own gates: {bundle_world.gates}')
                bundle_world.write(directory)
            self._bundles[name] = world.Bundle(directory, label, branching, 3)
        return self._bundles[name]

    def run_v2(self, bundle, stem='v2run', cfg=None, decode=None, **kwargs):
        from vkgr import generative_g6_run as g6
        f = world.fx()
        out = f.fresh(stem)
        with v2_spec(bundle.label, bundle.branching):
            return g6.run(*f.common(), bundle.manifest, bundle.sha, f.content_path, f.content_sha, cfg or g6_config(bundle.label), out,
                          temporary=f.fresh(stem + '-scratch'), decode=decode, expected_content_digest=f.digest, **kwargs), out

    def verify_v2(self, bundle, manifest_path, manifest_sha, decode=None):
        from vkgr import generative_g6_run as g6
        f = world.fx()
        with v2_spec(bundle.label, bundle.branching):
            return g6.verify(*f.common(), bundle.manifest, bundle.sha, f.content_path, f.content_sha, manifest_path, manifest_sha,
                             f.fresh('verify-v2'), temporary=f.fresh('verify-v2-scratch'), decode=decode, expected_content_digest=f.digest)

    def cached(self, name):
        if name not in self._runs:
            if name == 'v2':
                self._runs[name] = self.run_v2(self.bundle('replay'), 'v2')
            else:
                raise KeyError(name)
        return self._runs[name]


_V2 = []


def v2():
    if not _V2:
        _V2.append(V2Fixture())
    return _V2[0]


def catalog():
    return tuple(int(x) for x in world.fx().ids)


class ConfigV2Tests(unittest.TestCase):
    """B3: the committed T3 config is the A1 s42 config with exactly schema / task_id / sid replaced, and the formal G6 runner accepts it."""

    def setUp(self):
        self.a1, self.new = json.loads(A1_CONFIG.read_text()), json.loads(NEW_CONFIG.read_text())

    def test_key_sets_are_equal(self):
        self.assertEqual(set(self.new), set(self.a1))

    def test_exactly_schema_task_and_sid_differ(self):
        self.assertEqual({k for k in self.a1 if self.a1[k] != self.new[k]}, {'schema', 'task_id', 'sid'})
        self.assertEqual((self.new['schema'], self.new['task_id'], self.new['sid']), ('g-g6-run-v1', 'G-HP-01-G6', T3))

    def test_every_other_field_has_the_same_value_and_type(self):
        for key in sorted(set(self.a1) - {'schema', 'task_id', 'sid'}):
            with self.subTest(key=key):
                self.assertEqual(self.new[key], self.a1[key])
                self.assertIs(type(self.new[key]), type(self.a1[key]))

    def test_differs_from_the_g6prep_d4_config_only_in_sid(self):
        g6prep = json.loads(G6_CONFIG.read_text())
        self.assertEqual(set(g6prep), set(self.new))
        self.assertEqual({k for k in g6prep if g6prep[k] != self.new[k]}, {'sid'})

    def test_file_bytes_are_the_canonical_serialisation(self):
        self.assertEqual(NEW_CONFIG.read_text(), json.dumps(self.new, indent=1, sort_keys=True) + '\n')

    def test_formal_validation_accepts_the_committed_config(self):
        from vkgr import generative_g6_run as g6
        self.assertEqual(g6.validate_config(self.new, engineering=False), self.new)

    def test_the_gtok01a_config_with_the_wrong_task_id_is_refused(self):
        from vkgr import generative_g6_run as g6
        old = json.loads(OLD_CONFIG.read_text())
        self.assertEqual(old['task_id'], 'G-TOK-01A')
        with self.assertRaisesRegex(ValueError, 'frozen G6 field changed'):
            g6.validate_config(old, engineering=False)

    def test_only_t3_is_registered_for_the_formal_v2_path(self):
        from vkgr import generative_g6_run as g6
        from vkgr import sid_v2
        self.assertEqual(g6.V2_SID_VERSIONS, (T3,))
        self.assertIn(T3, sid_v2.SID_VERSIONS)
        for label in (V2_REPLAY, 'SID_RQVAE_B256D3_S42', 'SID_RQVAE_B16D3_S42'):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, 'unregistered SIDMAP_V2D label'):
                g6.validate_config(dict(self.new, sid=label), engineering=False)
        for label in (V2_REPLAY, T3):
            with self.subTest(label=label):
                g6.validate_config(g6_config(label), engineering=True)


class DispatchTests(unittest.TestCase):
    class Sentinel(Exception):
        pass

    def test_registered_v2_labels_per_context(self):
        from vkgr import generative_g6_run as g6
        self.assertEqual(g6._v2_labels(False), (T3,))
        self.assertEqual(g6._v2_labels(True), (T3, V2_REPLAY))

    def test_formal_load_calls_the_v2_loader_formally_with_the_content_v1_pin(self):
        from vkgr import generative_g6_run as g6
        from vkgr import sid_v2
        from vkgr import sid_v2d
        f, bundle = world.fx(), v2().bundle('replay')
        with mock.patch.object(sid_v2, 'load_bundle', side_effect=self.Sentinel) as load, self.assertRaises(self.Sentinel):
            g6._load_sid_v2(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog(), False, sid_v2d.CONTENT_V1_FEATURE_SHA256)
        (manifest, sha), kwargs = load.call_args
        self.assertEqual((manifest, sha), (bundle.manifest, bundle.sha))
        self.assertIs(kwargs['engineering'], False)
        self.assertEqual(kwargs['expected_content_digest'], sid_v2d.CONTENT_V1_FEATURE_SHA256)
        np.testing.assert_array_equal(kwargs['catalog_ids'], f.ids)
        np.testing.assert_array_equal(kwargs['content'], f.content)

    def test_formal_load_refuses_a_non_content_v1_digest_before_loading(self):
        from vkgr import generative_g6_run as g6
        from vkgr import sid_v2
        f, bundle = world.fx(), v2().bundle('replay')
        with mock.patch.object(sid_v2, 'load_bundle', side_effect=self.Sentinel) as load:
            with self.assertRaisesRegex(ValueError, 'formal context requires the CONTENT_V1 digest'):
                g6._load_sid_v2(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog(), False, f.digest)
        load.assert_not_called()

    def test_v2_codec_becomes_a_depth_three_codec_with_the_same_mapping(self):
        from vkgr import generative_g6_run as g6
        from vkgr import sid_depth as sd
        from vkgr import sid_v2
        f, bundle = world.fx(), v2().bundle('replay')
        with v2_spec(bundle.label, bundle.branching):
            loaded = sid_v2.load_bundle(bundle.manifest, bundle.sha, catalog_ids=f.ids, content=f.content, expected_content_digest=f.digest,
                                        engineering=True)
        adapted = g6._as_depth3(loaded)
        self.assertIsInstance(adapted.codec, sd.ValidatedCodecD)
        self.assertEqual(adapted.codec.depth, 3)
        old, new = loaded.codec, adapted.codec
        np.testing.assert_array_equal(np.asarray(new.catalog.ids), np.asarray(old.catalog.ids))
        np.testing.assert_array_equal(np.asarray(new.catalog.sids), np.asarray(old.catalog.sids))
        self.assertEqual((new.vocab_size, new.catalog.suffix_vocabulary), (old.vocab_size, old.catalog.suffix_vocabulary))
        self.assertEqual(list(new.offsets), list(old.offsets))
        self.assertEqual(adapted.info, {**loaded.info, 'depth': 3})
        self.assertIs(adapted.manifest, loaded.manifest)


class FormalSidPath(unittest.TestCase):
    """B4/B5 on the formal SID path (engineering=False, no mock of sid_v2.load_bundle): a T3-labelled formal SIDMAP_V2 bundle of the
    synthetic world goes through the runner's loader and the depth-3 adaptation into the G6 context. Test scope additionally pins the
    formal content digest to the synthetic content's digest (the real CONTENT_V1 rows exist only on the server)."""

    def load_formal(self, bundle):
        from vkgr import generative_g6_run as g6
        from vkgr import sid_v2d
        f = world.fx()
        with v2_spec(T3, bundle.branching), mock.patch.object(sid_v2d, 'CONTENT_V1_FEATURE_SHA256', f.digest):
            return g6._load_sid_v2(bundle.manifest, bundle.sha, f.content_path, f.content_sha, catalog(), False, f.digest)

    def test_formal_t3_bundle_is_accepted_and_builds_a_depth_three_g6_context(self):
        from vkgr import generative_g6_run as g6
        from vkgr import history_v2
        from vkgr import sid_depth as sd
        f = world.fx()
        sid = self.load_formal(v2().bundle('formal-t3'))
        self.assertIsInstance(sid.codec, sd.ValidatedCodecD)
        self.assertEqual((sid.info['sid_version'], sid.info['depth'], sid.info['engineering'], sid.info['kind']), (T3, 3, False, 'bhkm'))
        loaded = history_v2.load_prepared(f.hist_manifest, f.hist_sha, 50)
        context = g6.build_g6_context(f.context, loaded, g6.HISTORY_LENGTH, sid)
        self.assertEqual((context.binding['g6']['sid_version'], context.binding['g6']['depth'], context.binding['g6']['engineering']), (T3, 3, False))
        self.assertEqual(context.codec.vocab_size, f.context.codec.vocab_size)

    def test_formal_load_refuses_an_engineering_bundle(self):
        with self.assertRaises(ValueError):
            self.load_formal(v2().bundle('replay'))


class V2DepthThreeEquivalence(unittest.TestCase):
    def test_v2_replay_bundle_reproduces_the_arch_runner_per_epoch(self):
        arch, _ = world.fx().cached('arch')
        value, _ = v2().cached('v2')
        self.assertEqual(len(value['epochs']), 3)
        self.assertEqual(per_epoch(value), per_epoch(arch))
        self.assertEqual((value['best']['epoch'], value['best']['metric']), (arch['best']['epoch'], arch['best']['metric']))
        self.assertEqual(value['selection'], arch['selection'])
        self.assertEqual(value['binding']['context_b1']['sid_sha256'], arch['binding']['context_b1']['sid_sha256'])
        self.assertEqual(value['binding']['init_report']['arch'], arch['binding']['init_report']['arch'])
        self.assertEqual(value['binding']['init_report']['arch']['spec']['positions'], 200)
        self.assertEqual((value['binding']['sid']['depth'], value['binding']['sid']['sid_version']), (3, V2_REPLAY))
        self.assertEqual(value['binding']['task'], 'G-HP-01-G6')

    def test_v2_and_v2d_bundles_of_the_same_mapping_train_identically(self):
        legacy3, _ = world.fx().cached('legacy3')
        value, _ = v2().cached('v2')
        for key in ('model_sha256',):
            self.assertEqual([e[key] for e in value['epochs']], [e[key] for e in legacy3['epochs']])
        self.assertEqual([e['evaluation']['computation_sha256'] for e in value['epochs']],
                         [e['evaluation']['computation_sha256'] for e in legacy3['epochs']])
        self.assertNotEqual(value['binding']['sid']['manifest_sha256'], legacy3['binding']['sid']['manifest_sha256'])

    def test_fresh_one_epoch_v2_run_equals_the_arch_runner(self):
        from vkgr import generative_b1_arch as ba
        f = world.fx()
        arch = ba.run(*f.common(), arch_config(max_epochs=1, min_epochs=1), f.fresh('fresh-arch-v2'), temporary=f.fresh('fresh-arch-v2-scratch'))
        value, _ = v2().run_v2(v2().bundle('replay'), 'fresh-v2', g6_config(V2_REPLAY, max_epochs=1, min_epochs=1))
        self.assertEqual(per_epoch(value), per_epoch(arch))


class V2FailClosed(unittest.TestCase):
    def test_v2_run_verifies(self):
        value, out = v2().cached('v2')
        report = v2().verify_v2(v2().bundle('replay'), out / 'manifest.json', value['manifest_sha256'])
        self.assertEqual((report['status'], report['replayed_epochs'], report['epochs']), ('verified', [value['best']['epoch']], 3))

    def test_runs_do_not_verify_against_the_other_bundle_format(self):
        f = world.fx()
        value, out = v2().cached('v2')
        with self.assertRaises(ValueError):
            f.verify_g6(f.bundle('legacy3'), out / 'manifest.json', value['manifest_sha256'])
        legacy3, legacy3_out = f.cached('legacy3')
        with self.assertRaises(ValueError):
            v2().verify_v2(v2().bundle('replay'), legacy3_out / 'manifest.json', legacy3['manifest_sha256'])

    def test_label_and_bundle_format_must_agree_before_any_output(self):
        f = world.fx()
        for name, run, pattern in (('V2D bundle, V2 label', lambda: f.run_g6(f.bundle('legacy3'), 'cfgv2', g6_config(V2_REPLAY)), 'manifest key set mismatch'),
                                   ('V2 bundle, V2D label', lambda: v2().run_v2(v2().bundle('replay'), 'cfgv2', g6_config(V2D_REPLAY)), 'manifest key set mismatch')):
            before = set(f.tmp.iterdir())
            with self.subTest(name), self.assertRaisesRegex(ValueError, pattern):
                run()
            self.assertEqual([p for p in set(f.tmp.iterdir()) - before if re.fullmatch(r'cfgv2-\d+', p.name)], [])

    def test_runner_restores_the_frozen_decoder_names(self):
        from vkgr import generative_batched as gb
        from vkgr import tiger
        before = (tiger.decode_history, gb.decode_histories_batched)
        v2().cached('v2')
        self.assertEqual((tiger.decode_history, gb.decode_histories_batched), before)


class V2Cli(unittest.TestCase):
    """Local end-to-end through ``main``: exactly the 12 train flags, then verify with --manifest/--manifest-sha. The engineering world's
    content is not the CONTENT_V1 rows, so (test scope only) the runner's default content digest is pointed at the synthetic digest."""

    TRAIN_FLAGS = ('--prepared', '--prepared-sha', '--history-manifest', '--history-sha', '--sid-manifest', '--sid-sha', '--content', '--content-sha',
                   '--output', '--scratch', '--config', '--config-sha')

    def argv(self, command, bundle, out, tail):
        f = world.fx()
        args = {'--prepared': str(f.receipt['manifest_path']), '--prepared-sha': f.receipt['manifest_sha256'], '--history-manifest': str(f.hist_manifest),
                '--history-sha': f.hist_sha, '--sid-manifest': str(bundle.manifest), '--sid-sha': bundle.sha, '--content': str(f.content_path),
                '--content-sha': f.content_sha, '--output': str(out), '--scratch': str(f.fresh('cli-scratch'))}
        args.update(tail)
        return [command] + [part for item in args.items() for part in item]

    def call(self, argv, env, bundle):
        from vkgr import generative_g6_run as g6
        f = world.fx()
        clean = {k: v for k, v in os.environ.items() if not k.startswith('VKGR_')}
        clean.update(env)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, clean, clear=True), v2_spec(bundle.label, bundle.branching), \
                mock.patch.object(g6.run, '__kwdefaults__', {**g6.run.__kwdefaults__, 'expected_content_digest': f.digest}), \
                mock.patch.object(g6.verify, '__kwdefaults__', {**g6.verify.__kwdefaults__, 'expected_content_digest': f.digest}), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = g6.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def train(self, env):
        from vkgr.data_audit import file_sha256
        f, bundle = world.fx(), v2().bundle('replay')
        config = f.fresh('cli-cfg').with_suffix('.json')
        config.write_text(json.dumps(g6_config(V2_REPLAY), indent=1, sort_keys=True) + '\n')
        out = f.fresh('cli-out')
        argv = self.argv('train', bundle, out, {'--config': str(config), '--config-sha': file_sha256(config)})
        self.assertEqual(tuple(a for a in argv if a.startswith('--')), self.TRAIN_FLAGS)
        code, stdout, stderr = self.call(argv, env, bundle)
        self.assertEqual(code, 0, stderr)
        report = json.loads(stdout.strip().splitlines()[-1])
        self.assertEqual(report['status'], 'complete')
        return report, out, json.loads((out / 'manifest.json').read_text())

    def test_train_then_verify_through_the_cli(self):
        report, out, value = self.train({})
        self.assertEqual(len(value['epochs']), 3)
        self.assertEqual(value['binding']['sid']['sid_version'], V2_REPLAY)
        self.assertEqual(value['binding']['init_report']['arch']['spec']['positions'], 200)
        cached, _ = v2().cached('v2')
        self.assertEqual([e['model_sha256'] for e in value['epochs']], [e['model_sha256'] for e in cached['epochs']])
        bundle = v2().bundle('replay')
        code, stdout, stderr = self.call(self.argv('verify', bundle, world.fx().fresh('cli-verify'),
                                                   {'--manifest': str(out / 'manifest.json'), '--manifest-sha': report['manifest_sha256']}), {}, bundle)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout.strip().splitlines()[-1])['status'], 'verified')

    def test_batched_decode_with_batch_one_equals_the_serial_cli_run(self):
        _, _, serial = self.train({})
        _, _, batched = self.train({'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '1'})
        self.assertEqual([e['model_sha256'] for e in batched['epochs']], [e['model_sha256'] for e in serial['epochs']])
        self.assertEqual([e['evaluation']['computation_sha256'] for e in batched['epochs']],
                         [e['evaluation']['computation_sha256'] for e in serial['epochs']])
        self.assertTrue(all(e['observations']['execution']['decode'] == 'batched' for e in batched['epochs']))


if __name__ == '__main__':
    unittest.main()
