"""G6-PREP step 6b: depth-4 tokenizer fit/verify CLI, the frozen G6 fit config and its pre-registration.

All data is synthetic and runs on CPU with --engineering. The pre-registered Adagrad lr 0.4 collapses the codebooks on toy
data, so the RQ-VAE pipeline tests train with lr 0.1 followed by a k-means codebook refresh (same recipe as the GTOK-01
tests); nothing in src changes the constants. The config tests and the formal-path refusals run unpatched.
"""
import atexit
import contextlib
import copy
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tests import test_sid_tok_run as t1
from vkgr import sid as bhkm
from vkgr import sid_depth as sd
from vkgr import sid_rqvae as rq
from vkgr import sid_rqvae_d as rd
from vkgr import sid_tok_audit as au
from vkgr import sid_tok_audit_d as ad
from vkgr import sid_tok_run_d as run
from vkgr import sid_v2d
from vkgr.data_audit import canonical_bytes, file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'g6prep' / 'g-g6-tok.json'
SRC_PATH = ROOT / 'src' / 'vkgr' / 'sid_tok_run_d.py'
_REAL_FIT_D = rd.fit_d
_COUNTER = [0]


def blob_inputs(root, n=2560, blobs=256, noise=0.05, tag='in'):
    rng = np.random.RandomState(5)
    centers = rng.randn(blobs, 64)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    x = centers[np.repeat(np.arange(blobs), n // blobs)] + rng.randn(n, 64) * noise
    return t1.write_inputs(root, (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), tag)


def write_config(root, mutate=None):
    value = run.expected_config()
    if mutate:
        mutate(value)
    _COUNTER[0] += 1
    path = Path(root) / f'config-{_COUNTER[0]}.json'
    path.write_bytes(canonical_bytes(value))
    return {'path': path, 'sha': file_sha256(path)}


def fit_argv(variant, files, config, output, *extra, engineering=True):
    return t1.fit_argv(variant, files, config, output, *extra, engineering=engineering)


def verify_argv(manifest, files, config, *extra, engineering=True):
    return t1.verify_argv(manifest, files, config, *extra, engineering=engineering)


def cli(argv, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run.main(argv, **kwargs)
    return code, out.getvalue(), err.getvalue()


def refreshing_fit_d(content, **kwargs):
    """Real depth-generic fit, then the k-means codebook refresh (the weights written are the ones used)."""
    out = _REAL_FIT_D(content, **kwargs)
    if out['status'] == 'complete':
        model = rd.RQVAED(kwargs['K'], kwargs['levels'])
        model.load_state_dict(out['state']['state_dict'])
        rq.init_codebooks(model, torch.from_numpy(content), kwargs['seed'], 10)
        out['state'] = {'K': kwargs['K'], 'state_dict': {k: v.detach().clone() for k, v in model.state_dict().items()}}
    return out


RQ_EXTRA = ('--max-steps', '1000', '--batch-size', '256', '--probe-from', '100', '--probe-step', '200')


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name).resolve()
        cls.config = write_config(cls.root)
        cls.files = blob_inputs(cls.root)

    def out(self, name='out'):
        _COUNTER[0] += 1
        return self.root / f'{name}-{_COUNTER[0]}'


_BUILT = {}


def built():
    """Engineering bundles for B16D4, the depth-3 replay and RQ256D4, built once per process."""
    if not _BUILT:
        tmp = tempfile.TemporaryDirectory()
        atexit.register(tmp.cleanup)
        root = Path(tmp.name).resolve()
        config, files = write_config(root), blob_inputs(root)
        outs = {}
        with mock.patch.dict(rq.PREREG, lr=0.1), mock.patch.object(rd, 'fit_d', refreshing_fit_d):
            for variant, extra in (('B16D4', ()), ('B16D3_REPLAY', ()), ('RQ256D4', RQ_EXTRA)):
                outs[variant] = root / f'bundle-{variant}'
                code, _, err = cli(fit_argv(variant, files, config, outs[variant], *extra))
                assert code == 0, (variant, code, err)
        base = bhkm.fit(files['ids'], files['items'], branching=16, depth=3, seed=42, max_iter=10)
        np.savez(root / 'baseline.npz', ids=base.ids, sids=base.sids)
        _BUILT.update(root=root, config=config, files=files, outs=outs, baseline=root / 'baseline.npz',
                      baseline_sha=file_sha256(root / 'baseline.npz'), baseline_model=base)
    return _BUILT


class Config(unittest.TestCase):
    def test_committed_config_is_the_preregistration_in_code(self):
        loaded = run.load_config(CONFIG_PATH, file_sha256(CONFIG_PATH))
        self.assertEqual(loaded, run.expected_config())
        self.assertEqual(CONFIG_PATH.read_bytes(), canonical_bytes(run.expected_config()))

    def test_frozen_values(self):
        cfg = run.expected_config()
        self.assertEqual((cfg['schema'], cfg['task_id']), ('g-g6-tok-fit-v1', 'G-HP-01-G6'))
        r = cfg['rqvae']
        self.assertEqual((r['levels'], r['max_steps'], r['wall_cap_seconds'], r['batch_size'], r['lr'], r['beta'], r['seed']),
                         (4, 300000, 7200, 1024, 0.4, 0.25, 42))
        self.assertEqual(cfg['variants']['B16D4'], {'sid_version': 'SID_BHKM_B16D4_S42', 'kind': 'bhkm', 'branching': 16, 'depth': 4,
                                                    'max_iter': 10, 'seed': 42, 'prereg': ad.PREREG_BHKM_B16D4})
        self.assertEqual(cfg['variants']['B16D3_REPLAY'], {'sid_version': 'SID_V2D_REPLAY_BHKM_B16D3', 'kind': 'bhkm', 'branching': 16,
                                                           'depth': 3, 'max_iter': 10, 'seed': 42})
        self.assertEqual(cfg['variants']['RQ256D4'], {'sid_version': 'SID_RQVAE_B256D4_S42', 'kind': 'rqvae', 'codebook_size': 256, 'levels': 4})
        self.assertEqual(set(cfg['variants']), set(run.VARIANTS))
        for name, spec in cfg['variants'].items():
            self.assertEqual(run.VARIANT_LABELS[spec['sid_version']], name)
            self.assertEqual(sid_v2d.VERSION_SPEC[spec['sid_version']][0], spec['kind'])

    def test_every_frozen_value_is_pinned(self):
        def leaves(value, path=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from leaves(item, path + (key,))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    yield from leaves(item, path + (index,))
            else:
                yield path, value

        def set_path(value, path, new):
            for key in path[:-1]:
                value = value[key]
            value[path[-1]] = new
        count = 0
        for path, leaf in leaves(run.expected_config()):
            changed = copy.deepcopy(run.expected_config())
            set_path(changed, path, (leaf + 1) if isinstance(leaf, (int, float)) and not isinstance(leaf, bool) else (not leaf if isinstance(leaf, bool) else str(leaf) + 'x'))
            with self.subTest(path=path):
                self.assertRaises(ValueError, run.validate_config, changed)
            count += 1
        self.assertGreater(count, 60)
        extra = run.expected_config()
        extra['rqvae']['new'] = 1
        self.assertRaises(ValueError, run.validate_config, extra)
        missing = run.expected_config()
        del missing['variants']['RQ256D4']
        self.assertRaises(ValueError, run.validate_config, missing)

    def test_prereg_block_in_the_config_equals_the_capacity_rule(self):
        block = run.expected_config()['variants']['B16D4']['prereg']
        self.assertEqual(block, ad.prereg_block(120415, 16, 4))

    def test_no_local_paths_or_company_names_in_the_committed_files(self):
        markers = ('/Us' + 'ers/', '/work' + 'space', 'Hello' + 'Bike', '/pri' + 'vate/', '/ho' + 'me/')
        for path in (CONFIG_PATH, SRC_PATH, ROOT / 'src' / 'vkgr' / 'sid_v2d.py'):
            text = path.read_text(encoding='utf-8')
            for marker in markers:
                self.assertNotIn(marker, text, (path.name, marker))
            self.assertIsNone(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text), path.name)


class Pipeline(unittest.TestCase):
    def test_bhkm_depth_four_bundle(self):
        b = built()
        manifest = json.loads((b['outs']['B16D4'] / 'run_manifest.json').read_text())
        self.assertEqual((manifest['sid_version'], manifest['depth'], manifest['branching'], manifest['engineering']), ('SID_BHKM_B16D4_S42', 4, 16, True))
        self.assertEqual(manifest['gates']['passed'], True)
        self.assertEqual([c['id'] for c in manifest['gates']['checks']][-2:], ['B1', 'B2'])
        code, out, err = cli(verify_argv(b['outs']['B16D4'], b['files'], b['config']))
        self.assertEqual((code, err), (0, ''), err)
        self.assertEqual(json.loads(out)['checks'], ['load_bundle', 'config_binding'])

    def test_depth_four_mapping_is_the_frozen_fit_at_depth_four(self):
        b = built()
        again = bhkm.fit(b['files']['ids'], b['files']['items'], branching=16, depth=4, seed=42, max_iter=10)
        loaded = sid_v2d.load_bundle_d(b['outs']['B16D4'] / 'run_manifest.json', file_sha256(b['outs']['B16D4'] / 'run_manifest.json'),
                                      catalog_ids=b['files']['ids'], content=b['files']['items'], expected_content_digest=b['files']['digest'], engineering=True)
        self.assertTrue(np.array_equal(loaded.codec.catalog.sids, again.sids))
        self.assertEqual(loaded.codec.catalog.nodes, [])
        self.assertEqual(loaded.codec.depth, 4)

    def test_replay_fit_is_reproduced(self):
        b = built()
        code, out, err = cli(verify_argv(b['outs']['B16D4'], b['files'], b['config'], '--replay-fit'))
        self.assertEqual(code, 0, err)
        self.assertIn('replay_fit', json.loads(out)['checks'])

    def test_depth_three_replay_equals_the_frozen_fit_and_the_baseline(self):
        b = built()
        manifest = json.loads((b['outs']['B16D3_REPLAY'] / 'run_manifest.json').read_text())
        self.assertEqual((manifest['sid_version'], manifest['depth']), ('SID_V2D_REPLAY_BHKM_B16D3', 3))
        base = b['baseline_model']
        stored = np.load(b['outs']['B16D3_REPLAY'] / 'mapping.npz')
        self.assertTrue(np.array_equal(stored['ids'], base.ids) and np.array_equal(stored['sids'], base.sids))
        code, out, err = cli(verify_argv(b['outs']['B16D3_REPLAY'], b['files'], b['config'], '--baseline-mapping', str(b['baseline']), '--baseline-sha', b['baseline_sha']))
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['baseline'], {'mode': 'exact', 'equal': True})

    def test_replay_that_differs_from_the_baseline_is_refused(self):
        b = built()
        wrong = b['root'] / 'wrong-baseline.npz'
        sids = b['baseline_model'].sids.copy()
        sids[0, 0] = (sids[0, 0] + 1) % 16
        np.savez(wrong, ids=b['baseline_model'].ids, sids=sids)
        code, _, err = cli(verify_argv(b['outs']['B16D3_REPLAY'], b['files'], b['config'], '--baseline-mapping', str(wrong), '--baseline-sha', file_sha256(wrong)))
        self.assertEqual(code, 1)
        self.assertIn('baseline', err)

    def test_depth_four_reports_the_prefix_agreement_with_the_depth_three_baseline(self):
        b = built()
        code, out, err = cli(verify_argv(b['outs']['B16D4'], b['files'], b['config'], '--baseline-mapping', str(b['baseline']), '--baseline-sha', b['baseline_sha']))
        self.assertEqual(code, 0, err)
        report = json.loads(out)['baseline']
        mapping = np.load(b['outs']['B16D4'] / 'mapping.npz')['sids']
        equal = int(np.all(mapping[:, :3] == b['baseline_model'].sids[:, :3], axis=1).sum())
        self.assertEqual(report, {'mode': 'prefix_agreement', 'items': len(mapping), 'equal_prefix_items': equal, 'rate': equal / len(mapping), 'prefix_digits': 3})

    def test_rqvae_depth_four_bundle(self):
        b = built()
        manifest = json.loads((b['outs']['RQ256D4'] / 'run_manifest.json').read_text())
        self.assertEqual((manifest['sid_version'], manifest['depth'], manifest['kind']), ('SID_RQVAE_B256D4_S42', 4, 'rqvae'))
        tokenizer = json.loads((b['outs']['RQ256D4'] / 'tokenizer.json').read_text())
        self.assertEqual(tokenizer['config']['fit']['levels'], 4)
        self.assertEqual(tokenizer['config']['fit']['max_steps'], 1000)
        with mock.patch.dict(rq.PREREG, lr=0.1), mock.patch.object(rd, 'fit_d', refreshing_fit_d):
            code, out, err = cli(verify_argv(b['outs']['RQ256D4'], b['files'], b['config']))
        self.assertEqual(code, 0, err)

    def test_rqvae_prefix_replay(self):
        b = built()
        with mock.patch.dict(rq.PREREG, lr=0.1):
            code, out, err = cli(verify_argv(b['outs']['RQ256D4'], b['files'], b['config'], '--replay-prefix', '--device', 'cpu'))
        self.assertEqual(code, 0, err)
        self.assertIn('replay_prefix', json.loads(out)['checks'])

    def test_replay_flags_are_kind_specific(self):
        b = built()
        code, _, err = cli(verify_argv(b['outs']['B16D4'], b['files'], b['config'], '--replay-prefix'))
        self.assertEqual(code, 1)
        code, _, err = cli(verify_argv(b['outs']['RQ256D4'], b['files'], b['config'], '--replay-fit'))
        self.assertEqual(code, 1)


class Refusals(Base):
    def test_gate_failure_writes_failure_json_and_no_manifest(self):
        out = self.out()
        with mock.patch.dict(rq.PREREG, lr=0.1), mock.patch.object(rd, 'fit_d', refreshing_fit_d):
            code, stdout, stderr = cli(fit_argv('RQ256D4', self.files, self.config, out, '--max-steps', '100', '--batch-size', '256', '--probe-from', '100', '--probe-step', '200'))
        self.assertEqual(code, 3, stderr)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['failure.json'])
        record = json.loads((out / 'failure.json').read_text())
        self.assertEqual((record['schema'], record['task_id'], record['status']), ('g-g6-failure-v1', 'G-HP-01-G6', 'gate_failed'))
        self.assertTrue(set(record['reasons']) & {'R1', 'R3'})
        self.assertEqual(record['depth'], 4)
        self.assertNotIn('weights', json.dumps(record))

    def test_wall_cap_failure_returns_no_model(self):
        out = self.out()
        with mock.patch.dict(rq.PREREG, lr=0.1):
            code, stdout, stderr = cli(fit_argv('RQ256D4', self.files, self.config, out, '--max-steps', '300', '--batch-size', '256', '--probe-from', '100', '--probe-step', '200',
                                                '--wall-cap-seconds', '1'), clock=t1.Ticker(10000.0))
        self.assertEqual(code, 3, stderr)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['failure.json'])
        self.assertEqual(json.loads((out / 'failure.json').read_text())['reasons'], ['fit:fail_wall'])

    def test_formal_mode_refuses_the_toy_content(self):
        out = self.out()
        code, _, err = cli(fit_argv('B16D4', self.files, self.config, out, engineering=False))
        self.assertEqual(code, 1)
        self.assertIn('CONTENT_V1', err)
        self.assertFalse(out.exists())

    def test_engineering_only_options_need_engineering(self):
        for extra in (('--max-steps', '10'), ('--batch-size', '8'), ('--wall-cap-seconds', '5'), ('--probe-from', '5'), ('--probe-step', '10')):
            out = self.out()
            code, _, err = cli(fit_argv('RQ256D4', self.files, self.config, out, *extra, engineering=False))
            self.assertEqual(code, 1, extra)
            self.assertIn('engineering-only', err)
            self.assertFalse(out.exists())

    def test_overrides_only_apply_to_rqvae_and_engineering_rqvae_needs_max_steps(self):
        out = self.out()
        self.assertEqual(cli(fit_argv('B16D4', self.files, self.config, out, '--max-steps', '10'))[0], 1)
        self.assertEqual(cli(fit_argv('RQ256D4', self.files, self.config, out))[0], 1)
        self.assertFalse(out.exists())

    def test_formal_rqvae_needs_cuda(self):
        code, _, err = cli(fit_argv('RQ256D4', self.files, self.config, self.out(), '--device', 'cpu', engineering=False))
        self.assertEqual(code, 1)

    def test_unknown_variant_and_old_variants(self):
        for variant in ('T1', 'T3', 'B16D3', 'RQ16D4', ''):
            out = self.out()
            self.assertEqual(cli(fit_argv(variant, self.files, self.config, out))[0], 1, variant)
            self.assertFalse(out.exists())

    def test_input_pins_are_checked_before_anything_is_written(self):
        out = self.out()
        bad = dict(self.files, content_sha='0' * 64)
        self.assertEqual(cli(fit_argv('B16D4', bad, self.config, out))[0], 1)
        bad = dict(self.files, catalog_sha='0' * 64)
        self.assertEqual(cli(fit_argv('B16D4', bad, self.config, out))[0], 1)
        self.assertEqual(cli(fit_argv('B16D4', self.files, dict(self.config, sha='0' * 64), out))[0], 1)
        self.assertFalse(out.exists())

    def test_an_existing_output_directory_is_refused(self):
        out = self.out()
        out.mkdir()
        with mock.patch.object(bhkm, 'fit', side_effect=AssertionError('the fit must not start')):
            self.assertEqual(cli(fit_argv('B16D4', self.files, self.config, out))[0], 1)
        self.assertEqual(list(out.iterdir()), [])

    def test_a_config_with_a_changed_constant_is_refused(self):
        for mutate in (lambda v: v['rqvae'].__setitem__('max_steps', 10), lambda v: v['rqvae'].__setitem__('levels', 3),
                       lambda v: v['variants']['B16D4'].__setitem__('depth', 3), lambda v: v.__setitem__('task_id', 'G-TOK-01')):
            out = self.out()
            self.assertEqual(cli(fit_argv('B16D4', self.files, write_config(self.root, mutate), out))[0], 1)
            self.assertFalse(out.exists())

    def test_the_depth_three_config_of_the_previous_pipeline_is_not_accepted(self):
        out = self.out()
        old = {'path': t1.CONFIG_PATH, 'sha': file_sha256(t1.CONFIG_PATH)}
        self.assertEqual(cli(fit_argv('B16D4', self.files, old, out))[0], 1)

    def test_verify_refuses_a_bundle_fitted_with_another_config(self):
        b = built()
        other = write_config(b['root'], lambda v: None)
        # same content, different file name is fine: the sha is of the bytes, so identical bytes verify
        self.assertEqual(cli(verify_argv(b['outs']['B16D4'], b['files'], other))[0], 0)
        changed = write_config(b['root'], lambda v: v['variants']['B16D3_REPLAY'].__setitem__('max_iter', 11))
        self.assertEqual(cli(verify_argv(b['outs']['B16D4'], b['files'], changed))[0], 1)  # not an approved config

    def test_verify_refuses_a_config_with_the_same_values_but_different_bytes(self):
        b = built()
        pretty = b['root'] / 'config-pretty.json'
        pretty.write_text(json.dumps(run.expected_config(), indent=2, sort_keys=True))
        other = {'path': pretty, 'sha': file_sha256(pretty)}
        self.assertEqual(run.load_config(pretty, other['sha']), run.expected_config())  # valid config, different sha
        code, _, err = cli(verify_argv(b['outs']['B16D4'], b['files'], other))
        self.assertEqual(code, 1)
        self.assertIn('different config file', err)

    def test_replay_fit_refuses_a_mapping_the_frozen_fit_does_not_give(self):
        b = built()
        spec = dict(run.expected_config()['variants']['B16D4'], seed=7)
        other = run._fit_frozen_kmeans(b['files']['ids'], b['files']['items'], spec)
        self.assertFalse(np.array_equal(other['sids'], np.load(b['outs']['B16D4'] / 'mapping.npz')['sids']))
        with mock.patch.object(run, '_fit_frozen_kmeans', return_value=other):
            code, _, err = cli(verify_argv(b['outs']['B16D4'], b['files'], b['config'], '--replay-fit'))
        self.assertEqual(code, 1)
        self.assertIn('fit replay', err)

    def test_a_fresh_bhkm_fit_records_B1_and_B2_in_its_gates(self):
        out = self.out()
        code, stdout, stderr = cli(fit_argv('B16D4', self.files, self.config, out))
        self.assertEqual(code, 0, stderr)
        ids = [c['id'] for c in json.loads((out / 'run_manifest.json').read_text())['gates']['checks']]
        self.assertEqual(ids, ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'B1', 'B2'])

    def test_verify_refuses_the_formal_flag_on_an_engineering_bundle(self):
        b = built()
        self.assertEqual(cli(verify_argv(b['outs']['B16D4'], b['files'], b['config'], engineering=False))[0], 1)


if __name__ == '__main__':
    unittest.main()
