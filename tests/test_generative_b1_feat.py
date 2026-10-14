"""GFEAT-01 opt-in entry (vkgr.generative_b1_feat) on the frozen-base engineering fixture.

Run inside the INFRA-01 6.1 harness with OMP_NUM_THREADS=1 and ``discover -s tests`` (bare cross-test imports).
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_generative_b1 import config as b1_config
from test_generative_feat import side_world

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = '2c8882b4084a41d2862332a794477ba1d812fae3'
B1 = {'mode': 'batched', 'batch_size': 1}
FEATURES = {'g-feat01-f2-feedback-s42.json': 'f2_feedback', 'g-feat01-f3-watch-buckets-s42.json': 'f3_watch_buckets', 'g-feat01-none-ctrl-s42.json': 'none'}
# sha256 of every default-path file at the INFRA-02b base (2c8882b): this task adds files only, none of these may change.
DEFAULT_PATH = {
    'generative_b1_run': 'd161aec55f74339129d556991532ec49ce9e520edc6425c7179398c4b5055af5',
    'generative_b1_batched': 'd9f0ffba92fe57cc3dcd137b7083bdc92459561aa03d2bcacb4f5ebdb015bbb6',
    'generative_batched': '3918502e374bcedcf0873d2d9e419bc690f42d7ae991f81f9db793fb688b2564',
    'generative_evaluation': 'a70adb8ef10b118c8310cb033d0a70abc57352557c78a0e1fe5986e1372ccd20',
    'generative_training': '3de7c934df486dbeaa29afb3d60cd0c0b6078a365fdb3338d29fc4f4b2382bfb',
    'generative_data': 'ea4357989c3a520e7c7dbc1dfce2bbeff192f3d89e4e5f0703cfd27bf8e0c7c7',
    'generative_data_v2': 'd90f4b4338faefb6a9c72649e965a8e7480c7566e2b60ec683b9a4282b9b8157',
    'generative_hp': 'cecf05758a3903d4be5f66f7f5ae627ca05b30869091432a9e62e57a73e86140',
    'generative_sharded': '993ac3879c9c51c8a090656e7eb25cad5185d49ac0726208bfae4d94b02efed1',
    'generative_real': '79ba4fe5770438cf3a107c7064829f04b6d64de020b7b85e0a40a727c578fa9a',
    'generative_experiment': 'b0a94b26151c5fab13b0bd35bf0b3a0ff8ed8a3173f6b5aa08d410a83296b846',
    'generative_diagnostics': 'b72cd0bd974ded67f6afcae88f4db5a14ad5a6fb46368fbc9a5745b864926db4',
    'tiger': '09ab37d79da8e46dabf522d8d73d6dbd88b5b77985375e459e4977896a18c352',
    'sid': '9602580f907cc2e8cfa3684aec7828df2bdfc87b59e71b6fd3865a9a19de91a3',
    'history_v2': '1c8d8718ab7527524df9d47dfbb518ee7f5b419ca34dfbb8971aca7efc31f06e',
    'retrieval_models': 'a482cfde4ec0b6bf5525fa51d3bf226d8c10a4dc4aac785c1440721ee68f30f2',
}
ANCHOR_CONFIG_SHA = 'c196528f0f9d2f8af6974e34768396ee6fa3aac7cdc56d052851dc081c4144eb'
REPLACED = ('B1_MEMBERS', 'validate_config', '_contexts', 'make_model_b1', 'SIDTokensL', 'train_batch', '_evaluate', '_binding')


def tmpdir():
    return tempfile.TemporaryDirectory(dir='/private/tmp')


def entry_config(feature, **overrides):
    value = b1_config()
    value.update(schema='g-feat01-run-v1', task_id='GFEAT-01', feature=feature, side_version='HIST_V2_SIDE_V1')
    value.update(overrides)
    return value


def rewrite_side(world, tmp, name, mutate):
    """A self-consistent (re-hashed, correctly bound) copy of the sidecar after ``mutate(side)``."""
    from vkgr.history_v2 import _write_json
    source = Path(world.side_manifest).parent
    manifest = json.loads((source / 'manifest.json').read_text())
    side = {split: json.loads((source / f'side-{split}-l100.json').read_text()) for split in ('train', 'validation')}
    mutate(side)
    out = Path(tmp) / name; out.mkdir()
    manifest['artifacts'] = {f'side-{split}-l100.json': _write_json(out / f'side-{split}-l100.json', side[split]) for split in ('train', 'validation')}
    return out / 'manifest.json', _write_json(out / 'manifest.json', manifest)


class ConfigTests(unittest.TestCase):
    def test_config_registry(self):
        from vkgr import generative_b1_feat as feat
        feat.validate_config_feat(feat.default_config())
        for feature in ('none', 'f2_feedback', 'f3_watch_buckets'):
            feat.validate_config_feat(feat.default_config(feature))
        for seed in (42, 2024, 2025):
            feat.validate_config_feat({**feat.default_config('f2_feedback'), 'seed': seed})
        feat.validate_config_feat(entry_config('f3_watch_buckets'), engineering=True)
        bad = ({'feature': 'f1_ratio_pool'}, {'feature': 'c0_unsupported_const'}, {'feature': None}, {'feature': ['none']}, {'history_len': 20}, {'history_len': 100},
               {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'}, {'init': 'legacy'}, {'max_epochs': 30}, {'x': 1}, {'side_version': 'HIST_V2_SIDE_V2'}, {'side_version': None},
               {'schema': 'g-b1-run-v1'}, {'task_id': 'HIST-V2'}, {'task_id': 'G4'}, {'beam': 200}, {'learning_rate': .01}, {'seed': 7}, {'patience': 4})
        for patch_ in bad:
            with self.subTest(patch_=patch_), self.assertRaises(ValueError):
                feat.validate_config_feat({**feat.default_config('f2_feedback'), **patch_})
        for key in ('feature', 'side_version', 'beam'):
            missing = {k: v for k, v in feat.default_config('f2_feedback').items() if k != key}
            with self.subTest(missing=key), self.assertRaises(ValueError):
                feat.validate_config_feat(missing)
        with self.assertRaises(ValueError):
            feat.validate_config_feat([])

    def test_committed_gfeat01_configs_are_registered(self):
        from vkgr import generative_b1_feat as feat
        paths = sorted((ROOT / 'configs' / 'gfeat01').glob('*.json'))
        self.assertEqual([p.name for p in paths], sorted(FEATURES))
        anchor_path = ROOT / 'configs' / 'batch1' / 'g-hist-v2-l50-s42.json'
        self.assertEqual(hashlib.sha256(anchor_path.read_bytes()).hexdigest(), ANCHOR_CONFIG_SHA)
        anchor = json.loads(anchor_path.read_text())
        for path in paths:
            with self.subTest(config=path.name):
                value = json.loads(path.read_text())
                feat.validate_config_feat(value)
                self.assertEqual(value['feature'], FEATURES[path.name])
                self.assertEqual({k for k in set(value) | set(anchor) if value.get(k) != anchor.get(k)}, {'schema', 'task_id', 'feature', 'side_version'})
                self.assertEqual((value['schema'], value['task_id'], value['side_version'], value['seed']), ('g-feat01-run-v1', 'GFEAT-01', 'HIST_V2_SIDE_V1', 42))
                # generated, not hand-typed: same serialisation as the anchor file
                self.assertEqual(path.read_text(), json.dumps(value, indent=1, sort_keys=True) + '\n')


class DefaultPathTests(unittest.TestCase):
    def test_default_path_files_unchanged_since_2c8882b(self):
        """The new entry is purely additive: every file of the default / INFRA-02 path is byte-identical to the base."""
        from vkgr import generative_b1_batched as bb
        from vkgr import generative_b1_run as b1
        for name, expected in DEFAULT_PATH.items():
            with self.subTest(file=name):
                data = (ROOT / 'src' / 'vkgr' / f'{name}.py').read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
                try:
                    blob = subprocess.run(['git', '-C', str(ROOT), 'show', f'{BASE_COMMIT}:src/vkgr/{name}.py'], capture_output=True, check=True).stdout
                except (subprocess.CalledProcessError, OSError):
                    continue  # base commit not in this checkout: the pinned sha above is the evidence
                self.assertEqual(data, blob)
        names = ('history_v2', 'generative_data_v2', 'generative_hp', 'generative_diagnostics', 'generative_b1_run', 'generative_sharded', 'generative_real',
                 'generative_training', 'generative_evaluation', 'generative_experiment', 'generative_data', 'tiger', 'sid')
        self.assertEqual(b1.B1_MEMBERS, tuple(f'src/vkgr/{n}.py' for n in names))
        self.assertEqual(bb.MEMBERS, b1.B1_MEMBERS + ('src/vkgr/generative_batched.py', 'src/vkgr/generative_b1_batched.py'))
        anchor = ROOT / 'configs' / 'batch1' / 'g-hist-v2-l50-s42.json'
        self.assertEqual(hashlib.sha256(anchor.read_bytes()).hexdigest(), ANCHOR_CONFIG_SHA)

    def test_feature_entry_members_cover_vendored_and_new_modules(self):
        from vkgr import generative_b1_batched as bb
        from vkgr import generative_b1_feat as feat
        added = tuple(f'src/vkgr/{n}.py' for n in ('history_side', 'retrieval_features', 'retrieval_hp', 'retrieval_sasrec', 'retrieval_models',
                                                   'generative_feat', 'generative_feat_eval', 'generative_b1_feat'))
        self.assertEqual(feat.FEAT_MEMBERS, bb.MEMBERS + added)
        for member in feat.FEAT_MEMBERS:
            self.assertTrue((ROOT / member).is_file(), member)
        self.assertEqual(len(set(feat.FEAT_MEMBERS)), len(feat.FEAT_MEMBERS))


class EntryRunTests(unittest.TestCase):
    """End to end on the engineering fixture. One world per class; every run writes to its own output/scratch."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tmpdir(); cls.addClassCleanup(cls._tmp.cleanup)
        cls.tmp = Path(cls._tmp.name); cls.world = side_world(cls.tmp)
        w = cls.world
        cls.common = (ROOT, w.receipt['manifest_path'], w.receipt['manifest_sha256'], w.hist_manifest, w.hist_sha)
        cls.side = (w.side_manifest, w.side_sha)
        cls.runs = {}

    def run_entry(self, name, feature, *, decode=None, side=True, **overrides):
        """feat.run into a fresh directory; cached by name so every test reads the same run."""
        from vkgr import generative_b1_feat as feat
        if name not in self.runs:
            self.runs[name] = feat.run(*self.common, entry_config(feature, **overrides), self.tmp / name, temporary=self.tmp / f'{name}-scratch',
                                       decode=decode, side=self.side if side and feature != 'none' else None)
        return self.runs[name]

    def verify_entry(self, name, value, tag, *, side=True, manifest=None, sha=None, decode=None):
        from vkgr import generative_b1_feat as feat
        return feat.verify(*self.common, manifest or self.tmp / name / 'manifest.json', sha or value['manifest_sha256'], self.tmp / f'{name}-verify-{tag}',
                           temporary=self.tmp / f'{name}-vscratch-{tag}', side=self.side if side else None, decode=decode)

    @staticmethod
    def projection(run):
        return ([(e['epoch'], e['model_sha256'], e['evaluation']['computation_sha256'], json.dumps(e['metrics'], sort_keys=True)) for e in run['epochs']],
                run['losses'], run['selection'], run['best']['epoch'], run['best']['metric'])

    def test_none_control_matches_default_runner_and_batched_entry(self):
        """feature=none through the new entry reproduces the default runner and the INFRA-02 entry: model_sha256, computation_sha256,
        losses and selection, epoch by epoch (serial and batch-1 decode)."""
        from vkgr import generative_b1_batched as bb
        from vkgr import generative_b1_run as b1
        reference = b1.run(*self.common, b1_config(), self.tmp / 'ref-default', temporary=self.tmp / 'ref-default-scratch')
        batched = bb.run(*self.common, b1_config(), self.tmp / 'ref-bb', temporary=self.tmp / 'ref-bb-scratch', decode=B1)
        none_serial = self.run_entry('none-serial', 'none'); none_b1 = self.run_entry('none-b1', 'none', decode=B1)
        self.assertEqual(len(reference['epochs']), 3)
        for name, run in (('bb b=1', batched), ('none serial', none_serial), ('none b=1', none_b1)):
            with self.subTest(run=name):
                self.assertEqual(self.projection(run), self.projection(reference))
        files = none_serial['binding']['source']['files']
        self.assertIn('src/vkgr/generative_feat.py', files); self.assertIn('src/vkgr/generative_b1_feat.py', files); self.assertIn('src/vkgr/history_side.py', files)
        self.assertNotIn('src/vkgr/generative_feat.py', reference['binding']['source']['files'])
        self.assertEqual(none_serial['binding']['feat01'], {'feature': 'none', 'side_version': None})
        self.assertEqual(none_serial['binding']['context_b1'], reference['binding']['context_b1'])
        self.assertNotIn('feat01', json.loads((self.tmp / 'none-serial' / 'evaluation-001' / 'manifest.json').read_text())['binding']['context'])  # context untouched

    def test_none_takes_no_side_arguments(self):
        from vkgr import generative_b1_feat as feat
        with self.assertRaises(ValueError):
            feat.run(*self.common, entry_config('none'), self.tmp / 'none-with-side', temporary=self.tmp / 'none-with-side-scratch', side=self.side)
        self.assertFalse((self.tmp / 'none-with-side').exists() or (self.tmp / 'none-with-side-scratch').exists())
        for feature in ('f2_feedback', 'f3_watch_buckets'):
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                feat.run(*self.common, entry_config(feature), self.tmp / f'{feature}-no-side', temporary=self.tmp / f'{feature}-no-side-scratch')
            self.assertFalse((self.tmp / f'{feature}-no-side').exists() or (self.tmp / f'{feature}-no-side-scratch').exists())

    def test_f2_f3_train_then_verify_roundtrip(self):
        none = self.run_entry('none-serial', 'none')
        for feature in ('f2_feedback', 'f3_watch_buckets'):
            with self.subTest(feature=feature):
                value = self.run_entry(feature, feature, decode=B1)
                run = self.tmp / feature
                self.assertEqual(value['status'], 'complete'); self.assertEqual(len(value['epochs']), 3)
                self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
                self.assertEqual(value['config']['feature'], feature)
                self.assertEqual(value['binding']['feat01'], {'feature': feature, 'side_version': 'HIST_V2_SIDE_V1'})
                feat01 = json.loads((run / 'evaluation-001' / 'manifest.json').read_text())['binding']['context']['feat01']
                self.assertEqual(set(feat01), {'side_version', 'side_manifest_sha256', 'history_manifest_sha256', 'train_side_sha256', 'validation_side_sha256'})
                self.assertEqual((feat01['side_manifest_sha256'], feat01['history_manifest_sha256']), (self.world.side_sha, self.world.hist_sha))
                self.assertNotEqual([e['model_sha256'] for e in value['epochs']], [e['model_sha256'] for e in none['epochs']])
                report = self.verify_entry(feature, value, 'ok')  # verify decodes serially, training evaluated with the batch-1 decoder
                self.assertEqual((report['status'], report['replayed_epochs'], report['epochs']), ('verified', [value['best']['epoch']], 3))
                # verify may also use the batch-1 decoder
                self.assertEqual(self.verify_entry(feature, value, 'b1', decode=B1)['status'], 'verified')

    def test_batched_decode_recorded_in_observations(self):
        for name, feature, decode in (('none-serial', 'none', None), ('none-b1', 'none', B1), ('f2_feedback', 'f2_feedback', B1), ('f3-serial', 'f3_watch_buckets', None)):
            value = self.run_entry(name, feature, decode=decode)
            for e in value['epochs']:
                with self.subTest(run=name, epoch=e['epoch']):
                    if decode is None:
                        self.assertNotIn('execution', e['observations'])
                    else:
                        self.assertEqual((e['observations']['execution']['decode'], e['observations']['execution']['batch_size']), ('batched', 1))
        # same weights path => serial and batch-1 f3 evaluation agree with each other (the bit-equality that verify relies on)
        self.assertEqual([e['evaluation']['computation_sha256'] for e in self.run_entry('f3-serial', 'f3_watch_buckets')['epochs']],
                         [e['evaluation']['computation_sha256'] for e in self.run_entry('f3_watch_buckets', 'f3_watch_buckets', decode=B1)['epochs']])

    def test_verify_rejects_tampering(self):
        from vkgr.data_audit import file_sha256
        value = self.run_entry('f2_feedback', 'f2_feedback', decode=B1)
        run = self.tmp / 'f2_feedback'; manifest = json.loads((run / 'manifest.json').read_text())
        # a different (valid, re-hashed, correctly bound) side file: the context binding no longer matches the trained run
        path, sha = rewrite_side(self.world, self.tmp, 'other-side', lambda s: s['validation'][0][-1].__setitem__(0, (s['validation'][0][-1][0] + 1) % 64))
        from vkgr import generative_b1_feat as feat
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            feat.verify(*self.common, run / 'manifest.json', value['manifest_sha256'], self.tmp / 'v-other', temporary=self.tmp / 'v-other-s', side=(path, sha))
        # side arguments are mandatory for f2/f3
        with self.assertRaises(ValueError):
            self.verify_entry('f2_feedback', value, 'noside', side=False)
        with self.assertRaises(ValueError):  # wrong pins
            self.verify_entry('f2_feedback', value, 'badsha', sha='0' * 64)
        def forged(tag, mutate):
            tampered = json.loads(json.dumps(manifest)); mutate(tampered)
            target = run / f'forged-{tag}.json'; target.write_text(json.dumps(tampered))
            return target, file_sha256(target)
        cases = {'feature switched to none': lambda m: m['config'].__setitem__('feature', 'none'),
                 'forged metric': lambda m: m['epochs'][0]['metrics']['metrics']['all_eligible'].__setitem__('recall@50', 0.999),
                 'forged feat01 binding': lambda m: m['binding']['feat01'].__setitem__('feature', 'f3_watch_buckets'),
                 'forged computation sha': lambda m: m['epochs'][0]['evaluation'].__setitem__('computation_sha256', '0' * 64)}
        for tag, mutate in cases.items():
            target, pin = forged(tag.replace(' ', '-'), mutate)
            with self.subTest(case=tag), self.assertRaises(ValueError):
                self.verify_entry('f2_feedback', value, 'f-' + tag.replace(' ', '-'), manifest=target, sha=pin)
        # checkpoint tampering last (it changes the run directory)
        copy_dir = self.tmp / 'f2-tampered'
        import shutil
        shutil.copytree(run, copy_dir)
        (copy_dir / 'best.pt').write_bytes((copy_dir / 'best.pt').read_bytes() + b'0')
        with self.assertRaises(ValueError):
            feat.verify(*self.common, copy_dir / 'manifest.json', value['manifest_sha256'], self.tmp / 'v-ckpt', temporary=self.tmp / 'v-ckpt-s', side=self.side)

    def test_entries_do_not_cross_verify_and_restore_globals(self):
        from vkgr import generative_b1_batched as bb
        from vkgr import generative_b1_feat as feat
        from vkgr import generative_b1_run as b1
        before = {name: getattr(b1, name) for name in REPLACED}
        same = lambda: all(getattr(b1, name) is value for name, value in before.items())
        value = self.run_entry('f3_watch_buckets', 'f3_watch_buckets', decode=B1)
        self.assertTrue(same())
        with self.assertRaises(ValueError):  # exception path restores too
            feat.run(*self.common, entry_config('f2_feedback', history_len=20), self.tmp / 'x-fail', temporary=self.tmp / 'x-fail-s', side=self.side)
        self.assertTrue(same())
        manifest = self.tmp / 'f3_watch_buckets' / 'manifest.json'
        for name, call in (('b1.verify', b1.verify), ('bb.verify', bb.verify)):
            with self.subTest(verifier=name), self.assertRaises(ValueError):
                call(*self.common, manifest, value['manifest_sha256'], self.tmp / f'v-{name}', temporary=self.tmp / f'v-{name}-s')
            self.assertTrue(same())
        plain = b1.run(*self.common, b1_config(), self.tmp / 'plain-default', temporary=self.tmp / 'plain-default-s')
        with self.assertRaises(ValueError):
            feat.verify(*self.common, self.tmp / 'plain-default' / 'manifest.json', plain['manifest_sha256'], self.tmp / 'v-plain', temporary=self.tmp / 'v-plain-s')
        self.assertTrue(same())
        self.assertEqual(b1.verify(*self.common, self.tmp / 'plain-default' / 'manifest.json', plain['manifest_sha256'], self.tmp / 'v-plain-ok',
                                   temporary=self.tmp / 'v-plain-ok-s')['status'], 'verified')  # the default verifier still works after the entry ran

    def test_unregistered_variants_rejected_in_feature_entry(self):
        from vkgr import generative_b1_feat as feat
        cases = {'L20': {'history_len': 20}, 'L100': {'history_len': 100, 'task_id': 'GFEAT-01'}, 'random SID': {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'},
                 'task_id': {'task_id': 'HIST-V2'}, 'schema': {'schema': 'g-b1-run-v1'}, 'init legacy': {'init': 'legacy'}}
        for name, patch_ in cases.items():
            out = self.tmp / f'rej-{name.replace(" ", "-")}'
            with self.subTest(case=name), self.assertRaises(ValueError):
                feat.run(*self.common, entry_config('f2_feedback', **patch_), out, temporary=self.tmp / f'{out.name}-s', side=self.side)
            self.assertFalse(out.exists())  # nothing is published for a rejected variant
        for decode in ({'mode': 'batched', 'batch_size': 2}, {'mode': 'batched', 'batch_size': 64}, {'mode': 'serial', 'batch_size': 1}):
            out = self.tmp / f'rej-decode-{decode["batch_size"]}-{decode["mode"]}'
            with self.subTest(decode=decode), self.assertRaises(ValueError):
                feat.run(*self.common, entry_config('none'), out, temporary=self.tmp / f'{out.name}-s', decode=decode)
            self.assertFalse(out.exists())


class CliTests(unittest.TestCase):
    def call(self, tmp, argv_extra=(), env=None, config=None, side_args=()):
        from vkgr import generative_b1_feat as feat
        from vkgr import generative_sharded as gs
        tmp = Path(tmp)
        path = tmp / 'cfg.json'
        path.write_text(json.dumps(config if config is not None else entry_config('none'), sort_keys=True))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        argv = ['train', '--prepared', 'x', '--prepared-sha', '0' * 64, '--history-manifest', 'y', '--history-sha', '0' * 64, '--config', str(path), '--config-sha', sha,
                '--output', str(tmp / 'out'), '--scratch', str(tmp / 's'), *side_args, *argv_extra]
        err = io.StringIO()
        with patch.dict(os.environ, env or {}), patch.object(gs, 'cpu_quota_cores', return_value=None), contextlib.redirect_stderr(err):
            code = feat.main(argv)
        return code, err.getvalue()

    def test_cli_gates_before_output(self):
        side = ('--side-manifest', 'z', '--side-sha', '0' * 64)
        cases = {'batch 2': dict(env={'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '2'}),
                 'batch 64': dict(env={'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '64'}),
                 'batch invalid': dict(env={'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': 'x'}),
                 'decode gpu': dict(env={'VKGR_DECODE': 'gpu'}),
                 'workers 2': dict(env={'VKGR_EVAL_WORKERS': '2'}),
                 'workers 2 + batch 1': dict(env={'VKGR_EVAL_WORKERS': '2', 'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '1'}),
                 'workers invalid': dict(env={'VKGR_EVAL_WORKERS': '0'}),
                 'f2 without side': dict(config=entry_config('f2_feedback')),
                 'f3 without side': dict(config=entry_config('f3_watch_buckets')),
                 'none with side': dict(side_args=side),
                 'side manifest without sha': dict(config=entry_config('f2_feedback'), side_args=('--side-manifest', 'z')),
                 'side sha without manifest': dict(config=entry_config('f2_feedback'), side_args=('--side-sha', '0' * 64)),
                 'unregistered feature': dict(config=entry_config('f1_ratio_pool'), side_args=side)}
        for name, kwargs in cases.items():
            with self.subTest(case=name), tmpdir() as tmp:
                code, err = self.call(tmp, **kwargs)
                self.assertEqual(code, 1, err)
                self.assertIn('G-FEAT01 gate failed', err)
                self.assertFalse((Path(tmp) / 'out').exists() or (Path(tmp) / 's').exists())

    def test_cli_help_lists_side_arguments(self):
        from vkgr import generative_b1_feat as feat
        out = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(out):
            feat.main(['train', '--help'])
        text = out.getvalue()
        for flag in ('--side-manifest', '--side-sha', '--config', '--history-manifest'):
            self.assertIn(flag, text)


if __name__ == '__main__':
    unittest.main()
