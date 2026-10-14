"""GTOK-01R FEAT-003: probe / select / fit / verify CLI, expected_config_r and configs/gtok01r/g-tok-01r.json.

Synthetic data, CPU, --engineering, tiny step counts. The pre-registered probe grid (30,000 steps, logs every 1,000)
is far too long for a unit test, so the pipeline classes patch the live grid constants of sid_rqvae_r to a 300-step
version (class Tiny); the committed config and the selection outcomes are tested unpatched. Toy data collapses the
pre-registered Adagrad lr 0.4, so the pipeline classes also refresh the codebooks after the real fit_r (the same
fixture as tests/test_sid_tok_run.py); nothing in src changes.
"""
import atexit
import contextlib
import copy
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

from tests import test_sid_select_r as sel
from tests import test_sid_tok_run as old
from vkgr import generative_data as gd
from vkgr import sid_rqvae as rq
from vkgr import sid_rqvae_r as rr
from vkgr import sid_tok_audit as au
from vkgr import sid_tok_run as oldrun
from vkgr import sid_tok_run_r as rrun
from vkgr import sid_v2
from vkgr.data_audit import canonical_bytes, file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'gtok01r' / 'g-tok-01r.json'
SRC_PATH = ROOT / 'src' / 'vkgr' / 'sid_tok_run_r.py'
OLD_CONFIG_PATH = ROOT / 'configs' / 'gtok01' / 'g-tok-01.json'
COMMIT = 'a' * 40
LOCAL_MARKERS = ('/Us' + 'ers/', '/work' + 'space', 'Hello' + 'Bike', '/pri' + 'vate/', '/ho' + 'me/')
_REAL_FIT_R = rr.fit_r
_COUNTER = [0]
TINY_STEPS = 300
TINY_ARGS = ['--max-steps', '300', '--log-every', '100', '--eval-every', '100', '--probe-from', '100',
             '--probe-step', '200']
BUNDLE_FILES = {'mapping.npz', 'tokenizer.json', 'weights.npz', 'audit.json', 'train-log.json', 'run_manifest.json'}
L_ARMS = ('l1', 'l2', 'l3', 'l4')


def tiny_patches():
    """Live grid constants of a 300-step probe (three 100-step loss windows, usage every 100 steps), arm l1 at lr 0.025."""
    return [mock.patch.dict(rr.PREREG_R, probe_steps=TINY_STEPS),
            mock.patch.dict(rr.PREREG_R_SELECT, usage_log_steps=(100, 200, 300), a2_from_step=200, a2_to_step=300,
                            d_points=(100, 200, 300), i2_step=300),
            mock.patch.dict(rq.PREREG, log_every=100),
            # toy data collapses the pre-registered lr 0.4 and 300 steps are too few for lr 0.1 on K=256 (gate R3)
            mock.patch.dict(rr.PREREG_R['arms']['l1'], lr=0.025)]


def refreshing_fit_r(content, **kwargs):
    """Real fit_r, then the k-means codebook refresh of tests/test_sid_tok_run.py (the weights written are the ones used)."""
    out = _REAL_FIT_R(content, **kwargs)
    if out['status'] == 'complete':
        model = rq.RQVAE(kwargs['K'])
        model.load_state_dict(out['state']['state_dict'])
        rq.init_codebooks(model, torch.from_numpy(content), kwargs['seed'], 10)
        out['state'] = {'K': kwargs['K'], 'state_dict': {k: v.detach().clone() for k, v in model.state_dict().items()}}
    return out


def start_patches(cls):
    for patch in tiny_patches() + [mock.patch.object(rr, 'fit_r', refreshing_fit_r)]:
        patch.start()
        cls.addClassCleanup(patch.stop)


def write_config_r(root, mutate=None):
    value = rrun.expected_config_r()
    if mutate:
        mutate(value)
    _COUNTER[0] += 1
    path = Path(root) / f'config-r-{_COUNTER[0]}.json'
    path.write_bytes(canonical_bytes(value))
    return {'path': path, 'sha': file_sha256(path)}


def pin(argv, flag, path, sha):
    return argv + [f'--{flag}', str(path), f'--{flag}-sha', sha]


def common_argv(files, config):
    argv = pin([], 'config', config['path'], config['sha'])
    argv = pin(argv, 'content', files['content'], files['content_sha'])
    return pin(argv, 'catalog', files['catalog'], files['catalog_sha'])


def probe_argv(arm, variant, files, config, output, *extra, engineering=True, commit=COMMIT):
    argv = ['probe', '--arm', arm, '--variant', variant, '--code-commit', commit, '--output', str(output)]
    argv += common_argv(files, config)
    if engineering:
        argv.append('--engineering')
    return argv + list(extra)


def select_argv(probes, files, config, output, *extra):
    argv = ['select', '--output', str(output)] + common_argv_no_inputs(config)
    for path, sha in probes:
        argv += ['--probe', str(path), '--probe-sha', sha]
    return argv + list(extra)


def common_argv_no_inputs(config):
    return pin([], 'config', config['path'], config['sha'])


def fit_argv(variant, decision, files, config, output, *extra, engineering=True):
    argv = ['fit', '--variant', variant, '--decision', str(decision['path']), '--decision-sha', decision['sha'],
            '--output', str(output)] + common_argv(files, config)
    if engineering:
        argv.append('--engineering')
    return argv + list(extra)


def verify_argv(bundle, decision, files, config, *extra, engineering=True):
    manifest = Path(bundle) / 'run_manifest.json' if Path(bundle).is_dir() else Path(bundle)
    argv = ['verify', '--bundle', str(manifest), '--sha', file_sha256(manifest), '--decision', str(decision['path']),
            '--decision-sha', decision['sha']] + common_argv(files, config)
    if engineering:
        argv.append('--engineering')
    return argv + list(extra)


def cli(argv, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = rrun.main(argv, **kwargs)
    return code, out.getvalue(), err.getvalue()


def input_digests(files):
    return {'content_file_sha256': files['content_sha'], 'content_array_sha256': files['digest'],
            'catalog_sha256': sid_v2.catalog_digest(files['ids'])}


def tiny_probe(variant, arm, files, config_sha, *, status='complete', usage=None, r2=0.3):
    """Schema-valid probe-result for the live (patched) tiny grid, bound to the given input files."""
    steps = rr.PREREG_R['probe_steps']
    log_steps = list(range(rq.PREREG['log_every'], steps + 1, rq.PREREG['log_every']))
    usage_steps = list(rr.PREREG_R_SELECT['usage_log_steps'])
    usage = usage or ([1, 1, 2] if arm == 'p0' else sel.HEALTHY[variant])
    loss = list(sel.P0_LOSS[variant]) if arm == 'p0' else [0.9] * 3
    loss += [0.9] * (len(log_steps) - len(loss))
    elapsed = 5.0 if status == 'complete' else 3001.0
    n = steps if status == 'complete' else 100
    return {'schema': 'gtok01r-probe-v1', 'variant': variant, 'arm': arm, 'K': sel.KS[variant],
            'config_sha256': config_sha, 'code_commit': COMMIT, **input_digests(files), 'status': status, 'steps': n,
            'elapsed': elapsed, 'ms_per_step': elapsed * 1000 / n,
            'loss_log': [[s, v] for s, v in zip(log_steps, loss) if s <= n],
            'recon_log': [[s, 0.5] for s in log_steps if s <= n],
            'usage_log': [[s, list(usage)] for s in usage_steps if s <= n], 'diagnostics': [],
            'final': {'canonical_usage': list(usage), 'r2': r2, 'residual_norms': [1.0, 0.8, 0.5, 0.3]}
            if status == 'complete' else None}


def _next():
    _COUNTER[0] += 1
    return _COUNTER[0]


def write_json(root, name, value):
    _COUNTER[0] += 1
    path = Path(root) / f'{_COUNTER[0]}-{name}'
    path.write_bytes(canonical_bytes(value))
    return path, file_sha256(path)


def make_decision(root, files, config, probes_fn=None, *extra, expect=0):
    """Ten tiny probes -> the select CLI -> {'path', 'sha', 'value'} of the decision it wrote."""
    probes_fn = probes_fn or (lambda v, a: tiny_probe(v, a, files, config['sha']))
    written = [write_json(root, f'probe-{v}-{a}.json', probes_fn(v, a)) for v in sel.VARIANTS for a in sel.ARMS]
    _COUNTER[0] += 1
    output = Path(root) / f'decision-{_COUNTER[0]}'
    code, out, err = cli(select_argv(written, files, config, output, *extra, '--engineering'))
    assert code == expect, (code, out, err)
    path = output / 'decision.json'
    return {'path': path, 'sha': file_sha256(path), 'value': json.loads(path.read_text()), 'output': output,
            'stdout': out}


def rebundle(src, dst, edit_tokenizer=None, edit_manifest=None):
    """Copy a bundle and rewrite its tokenizer.json config (and manifest) with consistent hashes."""
    shutil.copytree(src, dst)
    if edit_tokenizer:
        path = Path(dst) / 'tokenizer.json'
        value = json.loads(path.read_text())
        edit_tokenizer(value['config'])
        path.write_bytes(canonical_bytes(value))
    manifest_path = Path(dst) / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts']['tokenizer.json'] = file_sha256(Path(dst) / 'tokenizer.json')
    if edit_manifest:
        edit_manifest(manifest)
    manifest_path.write_bytes(canonical_bytes(manifest))
    return Path(dst)


_BUILT = {}


def built():
    """Engineering bundles for T1 and T2 fitted through select -> fit, built once per process (call under patches)."""
    if not _BUILT:
        tmp = tempfile.TemporaryDirectory()
        atexit.register(tmp.cleanup)
        root = Path(tmp.name).resolve()
        config, files = write_config_r(root), old.write_inputs(root, old.blob_items())
        decision = make_decision(root, files, config)
        outs = {}
        for variant in ('T1', 'T2'):
            outs[variant] = root / f'bundle-{variant}'
            code, _, err = cli(fit_argv(variant, decision, files, config, outs[variant], *TINY_ARGS))
            assert code == 0, (variant, code, err)
        _BUILT.update(root=root, config=config, files=files, decision=decision, outs=outs)
    return _BUILT


def build_under_patches():
    """built() with the tiny constants active only while it runs (mutation classes need the unpatched constants)."""
    patches = tiny_patches() + [mock.patch.object(rr, 'fit_r', refreshing_fit_r)]
    for patch in patches:
        patch.start()
    try:
        return built()
    finally:
        for patch in reversed(patches):
            patch.stop()


class Tiny(unittest.TestCase):
    """300-step grid constants and the codebook refresh; per-class temp dir, config and inputs."""

    @classmethod
    def setUpClass(cls):
        start_patches(cls)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name).resolve()
        cls.config = write_config_r(cls.root)
        cls.files = old.write_inputs(cls.root, old.blob_items())

    def out(self, name='out'):
        _COUNTER[0] += 1
        return self.root / f'{name}-{_COUNTER[0]}'

    def decision(self, files=None, config=None, probes_fn=None, *extra, expect=0):
        return make_decision(self.root, files or self.files, config or self.config, probes_fn, *extra, expect=expect)

    def fit(self, variant, decision, *extra, files=None, config=None, engineering=True, **kwargs):
        output = self.out()
        code, stdout, stderr = cli(fit_argv(variant, decision, files or self.files, config or self.config, output,
                                            *extra, engineering=engineering), **kwargs)
        return output, code, stdout, stderr


def leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaf_paths(item, prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaf_paths(item, prefix + (index,))
    else:
        yield prefix


def container(value, path):
    for part in path:
        value = value[part]
    return value


class Config(unittest.TestCase):
    """Committed config, PREREG_R mirror, frozen-field rejection, formal-path refusals (no patches)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name).resolve()
        cls.files = old.write_inputs(cls.root, old.blob_items(2048))
        cls.committed = {'path': CONFIG_PATH, 'sha': file_sha256(CONFIG_PATH)}

    def test_committed_config_equals_expected_config_r(self):
        loaded = rrun.load_config_r(CONFIG_PATH, file_sha256(CONFIG_PATH))
        self.assertEqual(loaded, rrun.expected_config_r())
        self.assertEqual(loaded['schema'], 'g-tok-01r-fit-v1')
        self.assertEqual(loaded['task_id'], 'G-TOK-01')
        self.assertEqual(loaded['recipe'], 'gtok01r')
        self.assertEqual(sorted(loaded['variants']), ['T1', 'T2'])
        self.assertEqual(loaded['variants']['T1'], {'sid_version': 'SID_RQVAE_B256D3_S42', 'kind': 'rqvae',
                                                    'codebook_size': 256})
        self.assertEqual(loaded['variants']['T2'], {'sid_version': 'SID_RQVAE_B16D3_S42', 'kind': 'rqvae',
                                                    'codebook_size': 16})

    def test_prereg_r_equals_the_json_field_by_field(self):
        loaded = json.loads(CONFIG_PATH.read_text())

        def as_json(value):
            if isinstance(value, dict):
                return {k: as_json(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [as_json(v) for v in value]
            return value

        self.assertEqual(loaded['prereg_r'], as_json(rr.PREREG_R))
        self.assertEqual(loaded['select_r'], as_json(rr.PREREG_R_SELECT))
        paths = list(leaf_paths(loaded['prereg_r']))
        self.assertGreater(len(paths), 60)
        for path in paths:
            self.assertEqual(container(loaded['prereg_r'], path), container(as_json(rr.PREREG_R), path), path)
            self.assertIs(type(container(loaded['prereg_r'], path)), type(container(as_json(rr.PREREG_R), path)), path)
        for path in leaf_paths(loaded['select_r']):
            self.assertIs(type(container(loaded['select_r'], path)), type(container(as_json(rr.PREREG_R_SELECT), path)), path)
        for key, value in rq.PREREG.items():
            expected = list(value) if isinstance(value, tuple) else value
            self.assertEqual(loaded['rqvae'][key], expected, key)
        self.assertEqual(loaded['rqvae']['usage_diag_every'], rr.PREREG_R['usage_diag_every'])
        self.assertEqual(loaded['content'], {'feature_array_sha256': au.CONTENT_V1_FEATURE_SHA256, 'dim': 64})

    def test_every_field_change_is_rejected(self):
        base = json.loads(CONFIG_PATH.read_text())
        paths = list(leaf_paths(base))
        self.assertGreaterEqual(len(paths), 120)
        for path in paths:
            with self.subTest(change=path):
                broken = copy.deepcopy(base)
                parent = container(broken, path[:-1])
                old_value = parent[path[-1]]
                if isinstance(old_value, str):
                    parent[path[-1]] = old_value + 'x'
                elif isinstance(old_value, bool):
                    parent[path[-1]] = not old_value
                else:
                    parent[path[-1]] = old_value * 2 + 1
                with self.assertRaises(ValueError):
                    rrun.validate_config_r(broken)

    def test_every_field_removal_and_extra_field_is_rejected(self):
        base = json.loads(CONFIG_PATH.read_text())
        for path in leaf_paths(base):
            if isinstance(path[-1], int):
                continue
            with self.subTest(remove=path):
                broken = copy.deepcopy(base)
                del container(broken, path[:-1])[path[-1]]
                with self.assertRaises(ValueError):
                    rrun.validate_config_r(broken)
        for where in ((), ('rqvae',), ('content',), ('variants',), ('variants', 'T1'), ('prereg_r',), ('prereg_r', 'arms'),
                      ('prereg_r', 'arms', 'l1'), ('select_r',)):
            with self.subTest(extra=where):
                broken = copy.deepcopy(base)
                container(broken, where)['extra'] = 1
                with self.assertRaises(ValueError):
                    rrun.validate_config_r(broken)

    def test_named_frozen_fields_and_type_confusion(self):
        base = json.loads(CONFIG_PATH.read_text())
        edits = (('rqvae', 'max_steps', 299999), ('rqvae', 'wall_cap_seconds', 7201), ('rqvae', 'usage_gate_num_den', [4, 6]),
                 ('rqvae', 'seed', 43), ('prereg_r', 'probe_steps', 29999), ('prereg_r', 'probe_wall_cap_seconds', 3001),
                 ('select_r', 'e_r2_min', 0.04), ('select_r', 'p0_final_usage_max', 4), ('select_r', 'preflight_limit_seconds', 6001),
                 ('select_r', 'a2_from_step', 18000), ('select_r', 'd_points', [18000, 20000, 22000, 24000, 26000, 28000, 30000]))
        for section, name, replacement in edits:
            with self.subTest(field=name):
                broken = copy.deepcopy(base)
                broken[section][name] = replacement
                with self.assertRaises(ValueError):
                    rrun.validate_config_r(broken)
        for path, replacement in ((('rqvae', 'max_steps'), 300000.0), (('rqvae', 'lr'), 1), (('rqvae', 'seed'), True),
                                  (('prereg_r', 'arms', 'l2', 'lr'), '0.1'), (('prereg_r', 'arms', 'l1', 'reset'), 1),
                                  (('select_r', 'e_r2_min'), 1)):
            with self.subTest(type=path):
                broken = copy.deepcopy(base)
                container(broken, path[:-1])[path[-1]] = replacement
                with self.assertRaises(ValueError):
                    rrun.validate_config_r(broken)
        broken = copy.deepcopy(base)
        broken['variants']['T3'] = {'sid_version': 'SID_BHKM_B256D3_S42', 'kind': 'bhkm'}
        with self.assertRaises(ValueError):
            rrun.validate_config_r(broken)
        with self.assertRaises(ValueError):
            rrun.validate_config_r([base])
        with self.assertRaises(ValueError):
            rrun.validate_config_r(oldrun.expected_config())

    def test_probe_and_fit_params_come_from_their_own_sections(self):
        config = rrun.expected_config_r()
        shared = {'seed': 42, 'batch_size': 1024, 'check_every': 100, 'probe_from': 500, 'probe_step': 2000}
        self.assertEqual(rrun.probe_params(config, {}), {**shared, 'max_steps': 30000, 'wall_cap_seconds': 3000,
                                                         'log_every': 1000, 'eval_every': 2000})
        self.assertEqual(rrun.fit_params(config, {}), {**shared, 'max_steps': 300000, 'wall_cap_seconds': 7200,
                                                       'log_every': 1000, 'eval_every': 2000})
        given = {'max_steps': 300, 'wall_cap_seconds': 5.0, 'log_every': 100, 'eval_every': 50, 'batch_size': 64,
                 'probe_from': 100, 'probe_step': 200}
        for function in (rrun.probe_params, rrun.fit_params):
            self.assertEqual(function(config, given), {**given, 'seed': 42, 'check_every': 100})

    def test_load_config_rejects_wrong_sha_and_duplicate_keys(self):
        with self.assertRaises(ValueError):
            rrun.load_config_r(CONFIG_PATH, '0' * 64)
        duplicate = self.root / 'dup.json'
        text = CONFIG_PATH.read_text()
        duplicate.write_text(text.replace('"task_id": "G-TOK-01",', '"task_id": "G-TOK-01", "task_id": "G-TOK-01",', 1))
        self.assertNotEqual(duplicate.read_text(), text)
        with self.assertRaises(ValueError):
            rrun.load_config_r(duplicate, file_sha256(duplicate))

    def test_formal_path_refuses_cpu_and_other_content_before_writing(self):
        decision = {'path': self.root / 'none.json', 'sha': '0' * 64}
        for name, argv in (
                ('probe-cpu', probe_argv('p0', 'T2', self.files, self.committed, self.root / 'o1', '--device', 'cpu',
                                         engineering=False)),
                ('fit-cpu', fit_argv('T2', decision, self.files, self.committed, self.root / 'o2', '--device', 'cpu',
                                     engineering=False))):
            with self.subTest(case=name):
                code, _, err = cli(argv)
                self.assertEqual(code, 1)
                self.assertTrue(err.startswith('ERROR'), err)
        self.assertNotEqual(self.files['digest'], au.CONTENT_V1_FEATURE_SHA256)
        output = self.root / 'o3'
        code, _, err = cli(probe_argv('p0', 'T2', self.files, self.committed, output, '--device', 'cuda', engineering=False))
        self.assertEqual(code, 1)
        self.assertIn('CONTENT_V1', err)
        self.assertFalse(output.exists())
        for name in ('o1', 'o2'):
            self.assertFalse((self.root / name).exists())

    def test_formal_commands_refuse_live_constants_other_than_the_preregistered_ones(self):
        output = self.root / 'live'
        with mock.patch.dict(rr.PREREG_R, probe_steps=299):
            with self.assertRaises(ValueError):
                rrun.require_frozen_constants()
        with mock.patch.dict(rr.PREREG_R_SELECT, e_r2_min=0.01):
            with self.assertRaises(ValueError):
                rrun.require_frozen_constants()
        with mock.patch.dict(rq.PREREG, lr=0.1):
            with self.assertRaises(ValueError):
                rrun.require_frozen_constants()
        rrun.require_frozen_constants()
        p0 = [(self.root / 'x.json', '0' * 64)] * 10
        with mock.patch.dict(rr.PREREG_R_SELECT, e_r2_min=0.01):
            code, _, err = cli(select_argv(p0, self.files, self.committed, output))
        self.assertEqual(code, 1)
        self.assertIn('pre-registered', err)
        self.assertFalse(output.exists())


class Probe(Tiny):
    def probe(self, arm, variant, *extra, files=None, config=None, engineering=True, **kwargs):
        output = self.out('probe')
        code, out, err = cli(probe_argv(arm, variant, files or self.files, config or self.config, output,
                                        *(list(extra) or TINY_ARGS), engineering=engineering), **kwargs)
        return output, code, out, err

    def test_probe_writes_a_valid_result_bound_to_the_inputs(self):
        for arm, variant in (('p0', 'T2'), ('l4', 'T1'), ('l1', 'T2')):
            with self.subTest(arm=arm, variant=variant):
                output, code, out, err = self.probe(arm, variant)
                self.assertEqual(code, 0, err)
                self.assertEqual([p.name for p in output.iterdir()], ['probe-result.json'])
                path = output / 'probe-result.json'
                value = json.loads(path.read_text())
                rr.validate_probe(value)
                self.assertEqual(json.loads(out)['probe_sha256'], file_sha256(path))
                self.assertEqual((value['arm'], value['variant'], value['K']), (arm, variant, sel.KS[variant]))
                self.assertEqual(value['status'], 'complete')
                self.assertEqual(value['steps'], TINY_STEPS)
                self.assertEqual(value['config_sha256'], self.config['sha'])
                self.assertEqual(value['code_commit'], COMMIT)
                self.assertEqual({k: value[k] for k in input_digests(self.files)}, input_digests(self.files))
                self.assertEqual([s for s, _ in value['usage_log']], [100, 200, 300])
                self.assertEqual([s for s, _ in value['loss_log']], [100, 200, 300])
                self.assertGreater(len(value['diagnostics']), 10)
                self.assertEqual(value['diagnostics'][0]['step'], 0)
                self.assertTrue(math_close(value['ms_per_step'], value['elapsed'] * 1000 / value['steps']))
                final = value['final']
                self.assertEqual(len(final['canonical_usage']), 3)
                self.assertEqual(len(final['residual_norms']), 4)

    def test_probe_final_is_the_canonical_export_of_the_trained_state(self):
        states = []

        def spy(content, **kwargs):
            out = refreshing_fit_r(content, **kwargs)
            states.append(out['state'])
            return out

        with mock.patch.object(rr, 'fit_r', spy):
            output, code, _, err = self.probe('l1', 'T2')
        self.assertEqual(code, 0, err)
        value = json.loads((output / 'probe-result.json').read_text())
        report = rq.reconstruction_report(states[0], self.files['items'])
        used = rq.codebook_usage(rq.export_codes(states[0], self.files['items']), 16)
        self.assertEqual(value['final']['canonical_usage'], used)
        self.assertAlmostEqual(value['final']['r2'], report['r2'], places=12)
        self.assertEqual(value['final']['residual_norms'], report['residual_norms'])

    def test_probe_calls_fit_r_with_the_probe_arguments_only_from_the_config(self):
        calls = []

        def spy(content, **kwargs):
            calls.append(kwargs)
            return refreshing_fit_r(content, **kwargs)

        with mock.patch.object(rr, 'fit_r', spy):
            output, code, _, err = self.probe('l3', 'T2')
        self.assertEqual(code, 0, err)
        self.assertEqual(len(calls), 1)
        kwargs = calls[0]
        self.assertEqual((kwargs['arm'], kwargs['K'], kwargs['max_steps']), ('l3', 16, TINY_STEPS))
        self.assertIs(kwargs['project'], False)
        self.assertIs(kwargs['diagnostics'], True)
        self.assertEqual(kwargs['wall_cap_seconds'], rrun.expected_config_r()['prereg_r']['probe_wall_cap_seconds'])
        self.assertEqual(kwargs['wall_cap_seconds'], 3000)
        self.assertEqual((kwargs['eval_every'], kwargs['log_every'], kwargs['seed'], kwargs['batch_size']), (100, 100, 42, 1024))

    def test_probe_wall_cap_is_recorded_with_status_and_no_final(self):
        output, code, _, err = self.probe('l2', 'T2', *TINY_ARGS, clock=old.Ticker())
        self.assertEqual(code, 0, err)
        value = json.loads((output / 'probe-result.json').read_text())
        rr.validate_probe(value)
        self.assertEqual(value['status'], 'fail_wall')
        self.assertIsNone(value['final'])
        self.assertLess(value['steps'], TINY_STEPS)
        self.assertNotIn('state', value)

    def test_probe_nonfinite_run_is_recorded_as_a_valid_result(self):
        files = old.write_inputs(self.root, (self.files['items'] * np.float32(1e20)).astype(np.float32), tag='huge')
        output, code, _, err = self.probe('l1', 'T2', files=files)
        self.assertEqual(code, 0, err)
        text = (output / 'probe-result.json').read_text()
        self.assertNotIn('NaN', text)
        self.assertNotIn('Infinity', text)
        value = json.loads(text)
        rr.validate_probe(value)
        self.assertEqual(value['status'], 'fail_nonfinite')
        self.assertIsNone(value['final'])

    def test_probe_argument_and_input_errors_exit_1_before_any_output(self):
        output = self.out('probe-bad')
        base = probe_argv('l1', 'T2', self.files, self.config, output, *TINY_ARGS)

        def replace(flag, value, argv=None):
            argv = list(argv or base)
            argv[argv.index(flag) + 1] = value
            return argv

        wrong = '0' * 64
        cases = {
            'upper-case-arm': replace('--arm', 'L1'), 'p0-upper': replace('--arm', 'P0'), 'unknown-arm': replace('--arm', 'l5'),
            'variant-t3': replace('--variant', 'T3'), 'unknown-variant': replace('--variant', 'T9'),
            'no-commit': [a for i, a in enumerate(base) if a not in ('--code-commit', COMMIT)],
            'short-commit': replace('--code-commit', 'abc123'), 'upper-commit': replace('--code-commit', 'A' * 40),
            'engineering-needs-max-steps': probe_argv('l1', 'T2', self.files, self.config, output),
            'overrides-need-engineering': probe_argv('l1', 'T2', self.files, self.config, output, '--max-steps', '300',
                                                     engineering=False),
            'wrong-content-sha': replace('--content-sha', wrong), 'wrong-catalog-sha': replace('--catalog-sha', wrong),
            'wrong-config-sha': replace('--config-sha', wrong), 'unknown-flag': base + ['--bogus'],
            'missing-parent': replace('--output', str(self.root / 'no-such-parent' / 'out')),
        }
        messages = {'overrides-need-engineering': 'engineering-only', 'engineering-needs-max-steps': 'explicit --max-steps',
                    'short-commit': 'code-commit', 'upper-commit': 'code-commit', 'no-commit': 'code-commit'}
        for name, argv in cases.items():
            with self.subTest(case=name):
                code, out, err = cli(argv, clock=old.Ticker())  # a wrongly accepted run stops at its first wall check
                self.assertEqual(code, 1, err)
                self.assertEqual(out, '')
                self.assertTrue(err.startswith('ERROR'), err)
                self.assertIn(messages.get(name, ''), err)
        self.assertFalse(output.exists())

    def test_probe_refuses_an_unusable_result_and_writes_nothing(self):
        def broken(content, **kwargs):
            out = refreshing_fit_r(content, **kwargs)
            out['ms_per_step'] = out['ms_per_step'] * 2
            return out

        with mock.patch.object(rr, 'fit_r', broken):
            output, code, out, err = self.probe('l1', 'T2')
        self.assertEqual(code, 1, err)
        self.assertIn('ms_per_step', err)
        self.assertEqual(out, '')
        self.assertFalse(output.exists())

    def test_probe_refuses_an_existing_output_directory_and_a_changed_config(self):
        output = self.out('exists')
        output.mkdir()
        code, _, err = cli(probe_argv('l1', 'T2', self.files, self.config, output, *TINY_ARGS))
        self.assertEqual(code, 1, err)
        self.assertEqual(list(output.iterdir()), [])
        config = write_config_r(self.root, lambda v: v['prereg_r'].__setitem__('probe_wall_cap_seconds', 3001))
        output, code, _, err = self.probe('l1', 'T2', config=config)
        self.assertEqual(code, 1, err)
        self.assertFalse(output.exists())


def math_close(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b))


class SelectOutcomes(unittest.TestCase):
    """The four select outcomes through the CLI, on the real (unpatched) 30,000-step grid; no training."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name).resolve()
        cls.config = {'path': CONFIG_PATH, 'sha': file_sha256(CONFIG_PATH)}
        cls.files = {}

    def probes(self, spec=None, config_sha=None, drop=None, duplicate=None, mutate=None):
        spec = spec or {}
        written = []
        for variant in sel.VARIANTS:
            for arm in sel.ARMS:
                if (variant, arm) == drop:
                    continue
                value = sel.make_probe(variant, arm, config_sha=config_sha or self.config['sha'], **spec.get((variant, arm), {}))
                if mutate:
                    mutate(value)
                written.append(write_json(self.root, f'p-{variant}-{arm}.json', value))
        if duplicate:
            written.append(written[duplicate])
        return written

    def run_select(self, written, *extra):
        _COUNTER[0] += 1
        output = self.root / f'sel-{_COUNTER[0]}'
        code, out, err = cli(select_argv(written, self.files, self.config, output, *extra))
        return output, code, out, err

    def decision_of(self, output):
        return json.loads((output / 'decision.json').read_text())

    def test_healthy_l1_is_selected_with_exit_0(self):
        output, code, out, err = self.run_select(self.probes())
        self.assertEqual(code, 0, err)
        value = self.decision_of(output)
        rr.validate_decision(value)
        self.assertEqual(value['decision'], 'l1')
        self.assertIsNone(value['override'])
        self.assertEqual(value['config_sha256'], self.config['sha'])
        report = json.loads(out)
        self.assertEqual(report['decision'], 'l1')
        self.assertEqual(report['decision_sha256'], file_sha256(output / 'decision.json'))
        self.assertIs(report['preflight_exceeds'], False)
        self.assertEqual([p.name for p in output.iterdir()], ['decision.json'])

    def test_decision_references_the_probe_files_by_their_own_sha256(self):
        written = self.probes()
        output, code, _, err = self.run_select(written)
        self.assertEqual(code, 0, err)
        refs = {(e['variant'], e['arm']): e['sha256'] for e in self.decision_of(output)['probe_results']}
        for path, sha in written:
            value = json.loads(Path(path).read_text())
            self.assertEqual(refs[(value['variant'], value['arm'])], sha)

    def test_select_refuses_a_config_that_is_not_the_approved_one(self):
        other = write_config_r(self.root, lambda v: v['select_r'].__setitem__('e_r2_min', 0.04))
        written = self.probes(config_sha=other['sha'])
        output = self.root / 'sel-other-config'
        code, out, err = cli(select_argv(written, self.files, other, output))
        self.assertEqual(code, 1, err)
        self.assertIn('unapproved fit config', err)
        self.assertFalse(output.exists())

    def test_first_qualified_arm_in_fixed_order(self):
        spec = sel.set_arm({}, 'l1', usage=[1, 1, 2])
        output, code, _, err = self.run_select(self.probes(spec))
        self.assertEqual(code, 0, err)
        self.assertEqual(self.decision_of(output)['decision'], 'l2')

    def test_stop_writes_the_decision_and_exits_3(self):
        spec = {}
        for arm in L_ARMS:
            sel.set_arm(spec, arm, usage=[1, 1, 2])
        output, code, out, err = self.run_select(self.probes(spec))
        self.assertEqual(code, 3, err)
        value = self.decision_of(output)
        rr.validate_decision(value)
        self.assertEqual((value['decision'], value['escalate_listed_arms'], value['preflight']), ('STOP', [], None))
        self.assertEqual(json.loads(out)['decision'], 'STOP')

    def escalate_spec(self):
        spec = {}
        for arm in ('l1', 'l3', 'l4'):
            sel.set_arm(spec, arm, usage=[1, 1, 2])
        return sel.set_arm(spec, 'l2', sel.r2_low)

    def test_escalate_writes_the_decision_and_exits_3(self):
        output, code, _, err = self.run_select(self.probes(self.escalate_spec()))
        self.assertEqual(code, 3, err)
        value = self.decision_of(output)
        self.assertEqual((value['decision'], value['escalate_listed_arms']), ('ESCALATE', ['l2']))
        self.assertIsNone(value['override'])

    def test_invalid_control_writes_the_decision_but_exits_1(self):
        spec = sel.set_arm({}, 'p0', status='fail_wall')
        output, code, out, err = self.run_select(self.probes(spec))
        self.assertEqual(code, 1, err)
        value = self.decision_of(output)
        rr.validate_decision(value)
        self.assertEqual((value['decision'], value['preflight']), ('INVALID_CONTROL', None))
        self.assertEqual(json.loads(out)['decision'], 'INVALID_CONTROL')

    def test_override_releases_a_listed_arm_and_records_the_source(self):
        source = '2026-10-16 09:00 release l2'
        output, code, _, err = self.run_select(self.probes(self.escalate_spec()), '--override-arm', 'l2',
                                               '--override-source', source)
        self.assertEqual(code, 0, err)
        value = self.decision_of(output)
        rr.validate_decision(value)
        self.assertEqual(value['decision'], 'l2')
        self.assertEqual(value['override'], {'arm': 'l2', 'source': source, 'escalate_listed_arms': ['l2']})
        self.assertEqual(value['escalate_listed_arms'], ['l2'])

    def test_override_is_refused_unless_the_recomputation_is_escalate_and_the_arm_is_listed(self):
        cases = {
            'healthy-set': (self.probes(), ['--override-arm', 'l2', '--override-source', 's']),
            'arm-not-listed': (self.probes(self.escalate_spec()), ['--override-arm', 'l3', '--override-source', 's']),
            'no-source': (self.probes(self.escalate_spec()), ['--override-arm', 'l2']),
            'no-arm': (self.probes(self.escalate_spec()), ['--override-source', 's']),
            'blank-source': (self.probes(self.escalate_spec()), ['--override-arm', 'l2', '--override-source', ' ']),
            'upper-arm': (self.probes(self.escalate_spec()), ['--override-arm', 'L2', '--override-source', 's']),
            'p0-arm': (self.probes(self.escalate_spec()), ['--override-arm', 'p0', '--override-source', 's']),
        }
        for name, (written, extra) in cases.items():
            with self.subTest(case=name):
                output, code, out, err = self.run_select(written, *extra)
                self.assertEqual(code, 1, err)
                self.assertFalse(output.exists())
                self.assertEqual(out, '')

    def test_validation_failures_exit_1_and_write_nothing(self):
        wrong = '0' * 64
        good = self.probes()
        cases = {
            'nine-probes': self.probes(drop=('T2', 'l4')),
            'duplicate-probe': self.probes(drop=('T2', 'l4'), duplicate=0),
            'eleven-probes': good + [good[0]],
            'wrong-probe-sha': [(good[0][0], wrong)] + good[1:],
            'probe-for-another-config': self.probes(config_sha='b' * 64),
            'schema-mismatch': self.probes(mutate=lambda v: v.__setitem__('schema', 'gtok01r-probe-v0')),
            'extra-field': self.probes(mutate=lambda v: v.__setitem__('extra', 1)),
            'upper-case-arm': self.probes(mutate=lambda v: v.__setitem__('arm', v['arm'].upper())),
        }
        for name, written in cases.items():
            with self.subTest(case=name):
                output, code, out, err = self.run_select(written)
                self.assertEqual(code, 1, err)
                self.assertTrue(err.startswith('ERROR'), err)
                self.assertFalse(output.exists())
                self.assertEqual(out, '')

    def test_probes_with_different_commits_are_refused(self):
        counter = [0]

        def other_commit(value):
            counter[0] += 1
            if counter[0] == 3:
                value['code_commit'] = 'b' * 40

        output, code, _, err = self.run_select(self.probes(mutate=other_commit))
        self.assertEqual(code, 1, err)
        self.assertFalse(output.exists())

    def test_select_needs_every_probe_flag_pair_and_a_new_output(self):
        good = self.probes()
        output = self.root / 'sel-pair'
        argv = select_argv(good, self.files, self.config, output)
        index = argv.index('--probe-sha')
        code, _, err = cli(argv[:index] + argv[index + 2:])
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())
        existing = self.root / 'sel-existing'
        existing.mkdir()
        code, _, err = cli(select_argv(good, self.files, self.config, existing))
        self.assertEqual(code, 1, err)
        self.assertEqual(list(existing.iterdir()), [])
        code, _, err = cli(['select', '--output', str(output)])
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())

    def test_preflight_is_reported_and_never_changes_the_decision(self):
        def slow(variant):
            return {'ms': 10.0 * (2.0 if variant == 'T1' else 1.0)}

        spec = sel.set_arm({}, 'l1', slow)
        sel.set_arm(spec, 'p0', lambda v: {'ms': 10.0})
        output, code, out, err = self.run_select(self.probes(spec))
        self.assertEqual(code, 0, err)
        value = self.decision_of(output)
        self.assertEqual(value['decision'], 'l1')
        self.assertIs(value['preflight']['exceeds'], True)
        self.assertIs(json.loads(out)['preflight_exceeds'], True)


class FitNegatives(Tiny):
    """fit refuses before training; failures after training leave failure.json and never a manifest."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.good = make_decision(cls.root, cls.files, cls.config)

    def refused(self, variant, decision, *extra, message=None, **kwargs):
        output, code, out, err = self.fit(variant, decision, *extra, **kwargs)
        self.assertEqual(code, 1, err)
        self.assertEqual(out, '')
        self.assertFalse(output.exists())
        if message is not None:
            self.assertIn(message, err)
        return err

    def with_decision(self, mutate):
        value = copy.deepcopy(self.good['value'])
        mutate(value)
        path, sha = write_json(self.root, 'decision-edit.json', value)
        return {'path': path, 'sha': sha}

    def test_fit_without_decision_is_an_argument_error(self):
        output = self.out()
        argv = fit_argv('T2', self.good, self.files, self.config, output, *TINY_ARGS)
        index = argv.index('--decision')
        code, out, err = cli(argv[:index] + argv[index + 2:])  # --decision-sha stays: only the path is missing
        self.assertEqual(code, 1)
        self.assertTrue(err.rstrip().endswith('required: --decision'), err)
        self.assertFalse(output.exists())

    def test_wrong_decision_sha_and_missing_file(self):
        bad = {'path': self.good['path'], 'sha': '0' * 64}
        self.refused('T2', bad, *TINY_ARGS)
        self.refused('T2', {'path': self.root / 'missing.json', 'sha': self.good['sha']}, *TINY_ARGS)

    def test_decision_that_is_not_an_l_arm_is_refused(self):
        stop = make_decision(self.root, self.files, self.config, lambda v, a: tiny_probe(v, a, self.files, self.config['sha'],
                                                                                          usage=[1, 1, 2] if a != 'p0' else None),
                             expect=3)
        self.assertEqual(stop['value']['decision'], 'STOP')

        def escalate_probe(v, a):
            if a == 'l2':
                return tiny_probe(v, a, self.files, self.config['sha'], r2=0.01)
            return tiny_probe(v, a, self.files, self.config['sha'], usage=[1, 1, 2] if a != 'p0' else None)

        escalate = make_decision(self.root, self.files, self.config, escalate_probe, expect=3)
        self.assertEqual(escalate['value']['decision'], 'ESCALATE')
        invalid = make_decision(self.root, self.files, self.config,
                                lambda v, a: tiny_probe(v, a, self.files, self.config['sha'],
                                                        status='fail_wall' if a == 'p0' else 'complete'), expect=1)
        self.assertEqual(invalid['value']['decision'], 'INVALID_CONTROL')
        for name, decision in (('STOP', stop), ('ESCALATE', escalate), ('INVALID_CONTROL', invalid)):
            with self.subTest(decision=name):
                self.refused('T2', decision, *TINY_ARGS, message='not an l1..l4 arm')

    def test_decision_for_another_config_is_refused(self):
        other = write_config_r(self.root, lambda v: None)
        path = Path(other['path'])
        path.write_bytes(path.read_bytes() + b'\n')
        other['sha'] = file_sha256(path)
        decision = make_decision(self.root, self.files, other)
        self.refused('T2', decision, *TINY_ARGS, message='different config')

    def test_decision_for_other_inputs_is_refused(self):
        other_files = old.write_inputs(self.root, old.blob_items(2048, seed=5), tag='other')
        decision = make_decision(self.root, other_files, self.config)
        self.refused('T2', decision, *TINY_ARGS, message='does not match the inputs')
        for key in ('content_file_sha256', 'content_array_sha256', 'catalog_sha256'):
            with self.subTest(field=key):
                self.refused('T2', self.with_decision(lambda v: v.__setitem__(key, '9' * 64)), *TINY_ARGS,
                             message='does not match the inputs')

    def test_decision_with_missing_extra_or_mistyped_fields_is_refused(self):
        for field in rr.DECISION_KEYS:
            with self.subTest(missing=field):
                self.refused('T2', self.with_decision(lambda v: v.pop(field)), *TINY_ARGS, message='decision fields differ')
        self.refused('T2', self.with_decision(lambda v: v.__setitem__('extra', 1)), *TINY_ARGS, message='decision fields differ')
        self.refused('T2', self.with_decision(lambda v: v.__setitem__('schema', 'gtok01r-decision-v0')), *TINY_ARGS)
        self.refused('T2', self.with_decision(lambda v: v.__setitem__('decision', 'l5')), *TINY_ARGS)

    def test_k_mismatch_between_variant_and_decision_is_refused(self):
        with mock.patch.dict(rr.PREREG_R_SELECT, variant_K={'T1': 256, 'T2': 32}):
            self.refused('T2', self.good, *TINY_ARGS, message='variant K')
        entry = lambda v: v['per_arm_per_K']['l1']['T2'].update(status='fail_wall', a1=None, a2=None, b=None, c=None,
                                                                 d=None, e=None)
        self.refused('T2', self.with_decision(entry), *TINY_ARGS, message='does not cover')

    def test_wrong_input_and_config_shas_and_changed_config_are_refused(self):
        wrong = '0' * 64
        self.refused('T2', self.good, *TINY_ARGS, files={**self.files, 'content_sha': wrong})
        self.refused('T2', self.good, *TINY_ARGS, files={**self.files, 'catalog_sha': wrong})
        self.refused('T2', self.good, *TINY_ARGS, config={**self.config, 'sha': wrong})
        changed = write_config_r(self.root, lambda v: v['rqvae'].__setitem__('max_steps', 299999))
        self.refused('T2', self.good, *TINY_ARGS, config=changed)

    def test_variant_and_engineering_rules(self):
        self.refused('T3', self.good, *TINY_ARGS)
        self.refused('T9', self.good, *TINY_ARGS)
        self.refused('T2', self.good, message='explicit --max-steps', clock=old.Ticker())  # no --max-steps
        self.refused('T2', self.good, '--max-steps', '300', engineering=False, message='engineering-only',
                     clock=old.Ticker())
        self.refused('T2', self.good, '--max-steps', '0')

    def test_existing_output_directory_is_refused(self):
        output = self.out('exists')
        output.mkdir()
        code, _, err = cli(fit_argv('T2', self.good, self.files, self.config, output, *TINY_ARGS))
        self.assertEqual(code, 1, err)
        self.assertEqual(list(output.iterdir()), [])

    def collapsed(self, tag):
        directions = old.blob_items(3, blobs=3, noise=0.0, seed=3)
        files = old.write_inputs(self.root, directions[np.arange(2048) % 3], tag=tag)
        return files, make_decision(self.root, files, self.config)

    def test_gate_failure_writes_failure_json_with_diagnostics_and_no_manifest(self):
        files, decision = self.collapsed('three')
        output, code, out, err = self.fit('T2', decision, *TINY_ARGS, files=files)
        self.assertEqual(code, 3, err)
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['failure.json'])
        failure = json.loads((output / 'failure.json').read_text())
        self.assertEqual(failure['status'], 'gate_failed')
        self.assertIn('R1', failure['reasons'])
        self.assertIs(failure['gates']['passed'], False)
        self.assertEqual((failure['schema'], failure['recipe'], failure['arm']), ('g-tok-01r-failure-v1', 'gtok01r', 'l1'))
        self.assertEqual(failure['decision_sha256'], decision['sha'])
        self.assertTrue(failure['engineering'])
        self.assertNotIn('state', failure)
        detail = failure['detail']
        self.assertGreater(len(detail['diagnostics']), 10)
        self.assertIn('reset_log', detail)
        self.assertIn('recon_log', detail)
        self.assertEqual(json.loads(out)['status'], 'gate_failed')

    def test_wall_cap_writes_failure_json_and_no_manifest(self):
        output, code, _, err = self.fit('T2', self.good, *TINY_ARGS, clock=old.Ticker())
        self.assertEqual(code, 3, err)
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['failure.json'])
        failure = json.loads((output / 'failure.json').read_text())
        self.assertEqual((failure['status'], failure['reasons']), ('fail_wall', ['fit:fail_wall']))
        self.assertIsNone(failure['audit'])
        self.assertEqual(failure['arm'], 'l1')

    def test_wall_cap_and_gate_failed_results_expose_no_model(self):
        config = rrun.expected_config_r()
        digest = self.files['digest']
        result = rrun.fit_variant_r(config, 'T2', self.good['value'], self.good['sha'], self.files['ids'],
                                    self.files['items'], device='cpu', expected_content_sha256=digest, engineering=True,
                                    overrides={'max_steps': 300, 'log_every': 100, 'eval_every': 100, 'probe_from': 100,
                                               'probe_step': 200}, clock=old.Ticker())
        self.assertEqual(result['status'], 'fail_wall')
        for key in ('state', 'sids', 'codec', 'weights', 'ids', 'tokenizer', 'train_log'):
            self.assertNotIn(key, result)
        files, decision = self.collapsed('three-direct')
        result = rrun.fit_variant_r(config, 'T2', decision['value'], decision['sha'], files['ids'], files['items'],
                                    device='cpu', expected_content_sha256=files['digest'], engineering=True,
                                    overrides={'max_steps': 300, 'log_every': 100, 'eval_every': 100, 'probe_from': 100,
                                               'probe_step': 200}, clock=old.Ticker(0.001))
        self.assertEqual(result['status'], 'gate_failed')
        for key in ('state', 'sids', 'codec', 'weights', 'ids', 'tokenizer', 'train_log'):
            self.assertNotIn(key, result)

    def test_nonfinite_training_writes_failure_json_and_no_manifest(self):
        files = old.write_inputs(self.root, (self.files['items'] * np.float32(1e20)).astype(np.float32), tag='huge-fit')
        decision = make_decision(self.root, files, self.config)
        output, code, _, err = self.fit('T2', decision, *TINY_ARGS, files=files)
        self.assertEqual(code, 3, err)
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['failure.json'])
        text = (output / 'failure.json').read_text()
        self.assertEqual(json.loads(text)['status'], 'fail_nonfinite')
        self.assertNotIn('NaN', text)


def synthetic_train_log(diag=320, resets=8100, loss=300, usage=150):
    """Realistic-length train-log (300,000-step run): 320 diagnostic records, a dead-code reset at every check."""
    record = {'step': 123456, 'recon': 0.123456789012345, 'codebook': 0.0123456789012345, 'commit': 0.00123456789012345,
              'z_std_mean': 0.123456789012345, 'dead_relu': [0.123456789012345] * 6, 'usage_sub': [255, 255, 255],
              'perplexity': [200.123456789012, 200.123456789012, 200.123456789012],
              'nearest_dist': [0.123456789012345] * 3,
              'grad_norm': {'encoder': 0.123456789012345, 'decoder': 0.123456789012345, 'codebooks': 0.123456789012345},
              'rel_update': {'encoder': 0.00123456789012345, 'decoder': 0.00123456789012345,
                             'codebooks': 0.00123456789012345}}
    return {'K': 256, 'seed': 42, 'init_digest': 'a' * 64, 'prefix_digest': 'b' * 64,
            'loss_log': [[1000 * i, 0.123456789012345] for i in range(loss)],
            'usage_log': [[2000 * i, [255, 255, 255]] for i in range(usage)], 'arm': 'l4',
            'recon_log': [[1000 * i, 0.123456789012345] for i in range(loss)],
            'diagnostics': [dict(record, step=i) for i in range(diag)],
            'reset_log': [{'step': 100 * (i // 3 + 5), 'layer': i % 3, 'n_reset': 17} for i in range(resets)],
            'ms_per_step': 14.123456789, 'observations': {'device': 'cuda', 'wall_seconds': 4321.123456}}


class TrainLogSize(unittest.TestCase):
    def test_realistic_length_train_log_stays_under_one_megabyte(self):
        log = synthetic_train_log()
        size = len(canonical_bytes(log))
        self.assertLess(size, 1_000_000)
        self.assertGreater(size, 400_000)  # the fixture really is long: not a vacuous pass
        self.assertEqual(rrun.TRAIN_LOG_MAX_BYTES, 1_000_000)
        rrun.assert_train_log_size(log)

    def test_assertion_fires_when_the_limit_is_exceeded(self):
        log = synthetic_train_log(resets=40000)
        self.assertGreater(len(canonical_bytes(log)), 1_000_000)
        with self.assertRaises(ValueError) as caught:
            rrun.assert_train_log_size(log)
        self.assertIn('train-log.json exceeds', str(caught.exception))

    def test_limit_boundary(self):
        base = {'pad': ''}
        overhead = len(canonical_bytes(base))
        rrun.assert_train_log_size({'pad': 'x' * (1_000_000 - overhead - 1)})
        with self.assertRaises(ValueError):
            rrun.assert_train_log_size({'pad': 'x' * (1_000_000 - overhead)})


class SizeThroughCli(Tiny):
    def test_oversized_train_log_aborts_the_publication(self):
        decision = make_decision(self.root, self.files, self.config)

        def bloated(content, **kwargs):
            out = refreshing_fit_r(content, **kwargs)
            out['reset_log'] = [{'step': 1, 'layer': 0, 'n_reset': 1}] * 40000
            return out

        with mock.patch.object(rr, 'fit_r', bloated):
            output, code, out, err = self.fit_with(decision)
        self.assertEqual(code, 1, err)
        self.assertIn('train-log.json exceeds', err)
        self.assertFalse(output.exists())

    def fit_with(self, decision):
        output = self.out()
        code, out, err = cli(fit_argv('T2', decision, self.files, self.config, output, *TINY_ARGS))
        return output, code, out, err


class Pipeline(Tiny):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.b = built()

    def verify(self, variant, *extra, **kwargs):
        return cli(verify_argv(self.b['outs'][variant], self.b['decision'], self.b['files'], self.b['config'], *extra,
                               **kwargs))

    def load(self, variant, **kwargs):
        files, manifest = self.b['files'], self.b['outs'][variant] / 'run_manifest.json'
        return sid_v2.load_bundle(manifest, file_sha256(manifest), catalog_ids=files['ids'], content=files['items'],
                                  expected_content_digest=files['digest'], engineering=True)

    def test_select_to_fit_selected_l1_for_both_variants(self):
        value = self.b['decision']['value']
        self.assertEqual(value['decision'], 'l1')
        for variant in ('T1', 'T2'):
            tokenizer = json.loads((self.b['outs'][variant] / 'tokenizer.json').read_text())['config']
            self.assertEqual(tokenizer['arm'], 'l1')
            self.assertEqual(tokenizer['decision_sha256'], self.b['decision']['sha'])

    def test_bundle_file_set_is_the_unchanged_whitelist(self):
        for variant in ('T1', 'T2'):
            self.assertEqual(set(os.listdir(self.b['outs'][variant])), BUNDLE_FILES)

    def test_old_loader_accepts_and_new_verify_passes(self):
        for variant, version in (('T1', 'SID_RQVAE_B256D3_S42'), ('T2', 'SID_RQVAE_B16D3_S42')):
            with self.subTest(variant=variant):
                loaded = self.load(variant)
                self.assertEqual(loaded.info['sid_version'], version)
                self.assertTrue(loaded.manifest['gates']['passed'])
                self.assertEqual(loaded.manifest['fit_config_sha256'], self.b['config']['sha'])
                code, out, err = self.verify(variant)
                self.assertEqual(code, 0, err)
                report = json.loads(out)
                self.assertEqual((report['sid_version'], report['arm']), (version, 'l1'))
                self.assertIn('arm_hparams', report['checks'])

    def test_tokenizer_config_keeps_old_fields_and_adds_the_four_new_keys(self):
        for variant in ('T1', 'T2'):
            with self.subTest(variant=variant):
                tokenizer = json.loads((self.b['outs'][variant] / 'tokenizer.json').read_text())['config']
                self.assertEqual(set(tokenizer), {'variant', 'schema', 'fit', 'effective_prereg', 'engineering_overrides',
                                                  'zero_norm_rows', 'recipe', 'arm', 'decision_sha256', 'arm_hparams'})
                self.assertEqual(tokenizer['recipe'], 'gtok01r')
                self.assertEqual(tokenizer['schema'], 'g-tok-01r-fit-v1')
                arm = json.loads(json.dumps(rr.PREREG_R['arms']['l1']))
                self.assertEqual(tokenizer['arm_hparams'], arm)
                self.assertEqual(tokenizer['arm_hparams']['rng_reset_offset'], 7919)
                self.assertEqual(tokenizer['effective_prereg'], json.loads(json.dumps(rq.PREREG)))
                self.assertEqual(tokenizer['fit']['max_steps'], TINY_STEPS)
                self.assertEqual(tokenizer['engineering_overrides']['max_steps'], TINY_STEPS)

    def test_train_log_holds_diagnostics_reset_log_and_stays_small(self):
        for variant in ('T1', 'T2'):
            with self.subTest(variant=variant):
                path = self.b['outs'][variant] / 'train-log.json'
                log = json.loads(path.read_text())
                self.assertLess(path.stat().st_size, 1_000_000)
                self.assertEqual(log['arm'], 'l1')
                self.assertEqual(log['diagnostics'][0]['step'], 0)
                self.assertGreater(len(log['diagnostics']), 10)
                self.assertIsInstance(log['reset_log'], list)
                self.assertEqual([s for s, _ in log['recon_log']], [100, 200, 300])
                self.assertGreater(log['ms_per_step'], 0)
                for key in ('K', 'seed', 'init_digest', 'prefix_digest', 'loss_log', 'usage_log', 'usage_diagnostic',
                            'observations'):
                    self.assertIn(key, log)

    def test_producer_lists_the_new_source_files(self):
        producer = self.load('T2').manifest['producer']
        self.assertEqual(producer, rrun._producer_r())
        self.assertEqual(set(producer), {f'src/vkgr/{n}' for n in (
            'sid_tok_run_r.py', 'sid_rqvae_r.py', 'sid_tok_run.py', 'sid_rqvae.py', 'sid_v2.py', 'sid_tok_audit.py', 'sid.py')})
        for name in ('sid_tok_run_r.py', 'sid_rqvae_r.py', 'sid_rqvae.py', 'sid_v2.py', 'sid_tok_audit.py'):
            path = ROOT / 'src' / 'vkgr' / name
            self.assertEqual(producer[f'src/vkgr/{name}'], file_sha256(path), name)

    def test_engineering_bundles_are_marked_and_the_formal_loader_refuses_them(self):
        files = self.b['files']
        for variant in ('T1', 'T2'):
            manifest = self.b['outs'][variant] / 'run_manifest.json'
            self.assertIs(json.loads(manifest.read_text())['engineering'], True)
            with self.assertRaises(ValueError):
                sid_v2.load_bundle(manifest, file_sha256(manifest), catalog_ids=files['ids'], content=files['items'],
                                   expected_content_digest=files['digest'], engineering=False)
            code, _, err = self.verify(variant, engineering=False)
            self.assertEqual(code, 1, err)

    def test_replay_prefix_reproduces_the_digest_and_detects_a_difference(self):
        for variant in ('T1', 'T2'):
            stored = json.loads((self.b['outs'][variant] / 'train-log.json').read_text())['prefix_digest']
            self.assertRegex(stored, r'^[0-9a-f]{64}$')
            code, out, err = self.verify(variant, '--replay-prefix')
            self.assertEqual(code, 0, err)
            self.assertTrue(json.loads(out)['replay_prefix'])
        with mock.patch.object(rrun, '_replay_prefix_digest_r', lambda *a, **k: 'f' * 64):
            code, _, err = self.verify('T2', '--replay-prefix')
        self.assertEqual(code, 1)
        self.assertIn('prefix', err)

    def test_replay_runs_fit_r_for_the_recorded_arm_with_the_recorded_hparams(self):
        calls = []

        def spy(content, **kwargs):
            calls.append((kwargs['arm'], dict(rr.PREREG_R['arms'][kwargs['arm']]), kwargs['max_steps'], kwargs['diagnostics']))
            return _REAL_FIT_R(content, **kwargs)

        with mock.patch.object(rr, 'fit_r', spy):
            code, _, err = self.verify('T2', '--replay-prefix')
        self.assertEqual(code, 0, err)
        self.assertEqual(len(calls), 1)
        arm, table, max_steps, diagnostics = calls[0]
        recorded = json.loads((self.b['outs']['T2'] / 'tokenizer.json').read_text())['config']['arm_hparams']
        self.assertEqual((arm, max_steps, diagnostics), ('l1', 200, False))
        self.assertEqual(json.loads(json.dumps(table)), recorded)

    def test_replay_helper_installs_the_recorded_arm_hparams_and_restores_the_table(self):
        tok = json.loads((self.b['outs']['T2'] / 'tokenizer.json').read_text())['config']
        tok['arm_hparams']['lr'] = 0.3
        seen = []

        def spy(content, **kwargs):
            seen.append(rr.PREREG_R['arms'][kwargs['arm']]['lr'])
            return _REAL_FIT_R(content, **kwargs)

        live = rr.PREREG_R['arms']['l1']['lr']
        self.assertNotEqual(live, 0.3)
        with mock.patch.object(rr, 'fit_r', spy):
            rrun._replay_prefix_digest_r(tok, 16, self.b['files']['items'], 'cpu', True)
        self.assertEqual(seen, [0.3])
        self.assertEqual(rr.PREREG_R['arms']['l1']['lr'], live)

    def test_fit_records_the_arm_of_the_decision_not_a_fixed_one(self):
        files, config = self.b['files'], self.b['config']
        decision = make_decision(self.b['root'], files, config,
                                 lambda v, a: tiny_probe(v, a, files, config['sha'], usage=[1, 1, 2] if a == 'l1' else None))
        self.assertEqual(decision['value']['decision'], 'l2')
        output = self.b['root'] / f'bundle-l2-{_next()}'
        code, _, err = cli(fit_argv('T2', decision, files, config, output, *TINY_ARGS))
        self.assertEqual(code, 0, err)
        tokenizer = json.loads((output / 'tokenizer.json').read_text())['config']
        self.assertEqual(tokenizer['arm'], 'l2')
        self.assertEqual(tokenizer['arm_hparams'], json.loads(json.dumps(rr.PREREG_R['arms']['l2'])))
        self.assertEqual(json.loads((output / 'train-log.json').read_text())['arm'], 'l2')
        code, _, err = cli(verify_argv(output, decision, files, config, '--replay-prefix'))
        self.assertEqual(code, 0, err)
        code, _, err = cli(verify_argv(output, self.b['decision'], files, config))
        self.assertEqual(code, 1)
        self.assertIn('arm differs from the decision', err)

    def test_old_sid_tok_run_verify_refuses_the_new_bundle(self):
        files = self.b['files']
        manifest = self.b['outs']['T2'] / 'run_manifest.json'
        old_config = {'path': OLD_CONFIG_PATH, 'sha': file_sha256(OLD_CONFIG_PATH)}
        for label, config in (('old-config', old_config), ('new-config', self.b['config'])):
            with self.subTest(config=label):
                argv = old.verify_argv(manifest, files, config)
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = oldrun.main(argv)
                self.assertEqual(code, 1, err.getvalue())
                self.assertTrue(err.getvalue().startswith('ERROR'))
        with self.assertRaises(ValueError):
            oldrun.load_config(self.b['config']['path'], self.b['config']['sha'])

    def test_dead_code_resets_are_logged_and_replayable(self):
        small = {'dead_window': 100, 'reset_check_every': 100, 'reset_until': 300}
        with mock.patch.dict(rr.PREREG_R['arms']['l1'], small):
            decision = make_decision(self.root, self.files, self.config)
            output, code, _, err = self.fit('T1', decision, *TINY_ARGS)
            self.assertEqual(code, 0, err)
            log = json.loads((output / 'train-log.json').read_text())
            self.assertGreater(len(log['reset_log']), 0)
            for entry in log['reset_log']:
                self.assertEqual(set(entry), {'step', 'layer', 'n_reset'})
                self.assertIn(entry['layer'], (0, 1, 2))
            tokenizer = json.loads((output / 'tokenizer.json').read_text())['config']
            self.assertEqual(tokenizer['arm_hparams']['dead_window'], 100)
            code, out, err = cli(verify_argv(output, decision, self.files, self.config, '--replay-prefix'))
            self.assertEqual(code, 0, err)


class VerifyNegatives(Tiny):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.b = built()

    def verify_bundle(self, bundle, *, decision=None, **kwargs):
        return cli(verify_argv(bundle, decision or self.b['decision'], self.b['files'], self.b['config'], **kwargs))

    def tamper(self, edit, name='t'):
        _COUNTER[0] += 1
        return rebundle(self.b['outs']['T2'], self.b['root'] / f'{name}-{_COUNTER[0]}', edit)

    def refused(self, bundle, message, *, decision=None, extra=()):
        code, out, err = cli(verify_argv(bundle, decision or self.b['decision'], self.b['files'], self.b['config'], *extra))
        self.assertEqual(code, 1, err)
        self.assertIn(message, err)
        self.assertEqual(out, '')

    def test_formal_verify_refuses_changed_live_constants(self):
        # this class runs under the 300-step constants, so a formal (non-engineering) verify must stop right there
        code, out, err = cli(verify_argv(self.b['outs']['T2'], self.b['decision'], self.b['files'], self.b['config'],
                                         engineering=False))
        self.assertEqual(code, 1)
        self.assertIn('pre-registered', err)
        self.assertEqual(out, '')

    def test_untampered_copy_still_verifies(self):
        copy_ = self.tamper(lambda c: None)
        code, _, err = self.verify_bundle(copy_)
        self.assertEqual(code, 0, err)

    def test_verify_without_decision_is_an_argument_error(self):
        manifest = self.b['outs']['T2'] / 'run_manifest.json'
        argv = ['verify', '--bundle', str(manifest), '--sha', file_sha256(manifest), '--decision-sha',
                self.b['decision']['sha']] + common_argv(self.b['files'], self.b['config'])
        argv.append('--engineering')
        code, out, err = cli(argv)
        self.assertEqual(code, 1)
        self.assertTrue(err.rstrip().endswith('required: --decision'), err)
        self.assertEqual(out, '')

    def test_arm_differing_from_the_decision_is_refused(self):
        self.refused(self.tamper(lambda c: c.__setitem__('arm', 'l2')), 'arm differs from the decision')
        self.refused(self.tamper(lambda c: c.__setitem__('arm', 'p0')), 'arm differs from the decision')

    def test_tampered_arm_hparams_are_refused(self):
        self.refused(self.tamper(lambda c: c['arm_hparams'].__setitem__('lr', 0.1)), 'arm_hparams')
        self.refused(self.tamper(lambda c: c['arm_hparams'].__setitem__('dead_window', 499)), 'arm_hparams')
        self.refused(self.tamper(lambda c: c['arm_hparams'].pop('eps')), 'arm_hparams')
        self.refused(self.tamper(lambda c: c['arm_hparams'].__setitem__('extra', 1)), 'arm_hparams')
        self.refused(self.tamper(lambda c: c.__setitem__('arm_hparams', json.loads(json.dumps(rr.PREREG_R['arms']['l2'])))),
                     'arm_hparams')

    def test_missing_or_wrong_recipe_is_refused(self):
        self.refused(self.tamper(lambda c: c.pop('recipe')), 'recipe')
        self.refused(self.tamper(lambda c: c.__setitem__('recipe', 'gtok01')), 'recipe')

    def test_missing_extra_key_in_tokenizer_config_is_refused(self):
        for key in ('arm', 'arm_hparams', 'decision_sha256', 'fit', 'effective_prereg'):
            with self.subTest(missing=key):
                code, _, err = self.verify_bundle(self.tamper(lambda c: c.pop(key)))
                self.assertEqual(code, 1, err)
        code, _, err = self.verify_bundle(self.tamper(lambda c: c.__setitem__('extra', 1)))
        self.assertEqual(code, 1, err)

    def test_decision_sha_recorded_in_the_bundle_must_match(self):
        self.refused(self.tamper(lambda c: c.__setitem__('decision_sha256', '1' * 64)), 'decision_sha256')

    def test_decision_must_belong_to_this_config_and_sha(self):
        wrong = {'path': self.b['decision']['path'], 'sha': '0' * 64}
        code, out, err = self.verify_bundle(self.b['outs']['T2'], decision=wrong)
        self.assertEqual(code, 1, err)
        other = write_config_r(self.b['root'], None)
        path = Path(other['path'])
        path.write_bytes(path.read_bytes() + b'\n')
        other['sha'] = file_sha256(path)
        decision = make_decision(self.b['root'], self.b['files'], other)
        self.refused(self.b['outs']['T2'], 'different config', decision=decision)

    def test_decision_with_another_arm_is_refused_for_this_bundle(self):
        def probes(v, a):
            if a == 'l1':
                return tiny_probe(v, a, self.b['files'], self.b['config']['sha'], usage=[1, 1, 2])
            return tiny_probe(v, a, self.b['files'], self.b['config']['sha'])

        decision = make_decision(self.b['root'], self.b['files'], self.b['config'], probes)
        self.assertEqual(decision['value']['decision'], 'l2')
        self.refused(self.b['outs']['T2'], 'arm differs from the decision', decision=decision)

    def test_decision_missing_the_override_field_is_refused(self):
        value = copy.deepcopy(self.b['decision']['value'])
        del value['override']
        path, sha = write_json(self.b['root'], 'decision-no-override.json', value)
        self.refused(self.b['outs']['T2'], 'decision fields differ', decision={'path': path, 'sha': sha})

    def test_stop_decision_cannot_verify_a_bundle(self):
        stop = make_decision(self.b['root'], self.b['files'], self.b['config'],
                             lambda v, a: tiny_probe(v, a, self.b['files'], self.b['config']['sha'],
                                                     usage=[1, 1, 2] if a != 'p0' else None), expect=3)
        self.refused(self.b['outs']['T2'], 'not an l1..l4 arm', decision=stop)

    def test_decision_for_other_inputs_is_refused(self):
        other_files = old.write_inputs(self.b['root'], old.blob_items(2048, seed=9), tag='other-verify')
        decision = make_decision(self.b['root'], other_files, self.b['config'])
        self.refused(self.b['outs']['T2'], 'does not match the inputs', decision=decision)

    def test_bundle_fitted_with_another_config_file_is_refused(self):
        def other_sha(manifest):
            manifest['fit_config_sha256'] = '2' * 64

        bundle = rebundle(self.b['outs']['T2'], self.b['root'] / f'other-config-sha-{_next()}', None, other_sha)
        self.refused(bundle, 'different config')

    def test_fit_params_effective_prereg_and_overrides_are_bound_to_the_config(self):
        self.refused(self.tamper(lambda c: c['fit'].__setitem__('max_steps', 301)), 'config')
        self.refused(self.tamper(lambda c: c['fit'].__setitem__('seed', 7)), 'config')
        self.refused(self.tamper(lambda c: c.__setitem__('schema', 'g-tok-01-fit-v1')), 'config')
        self.refused(self.tamper(lambda c: c['engineering_overrides'].__setitem__('bogus', 1)), 'config')
        self.refused(self.tamper(lambda c: c['effective_prereg'].__setitem__('beta', 0.5)), 'config')

    def test_variant_label_in_the_bundle_must_match_its_sid_version(self):
        self.refused(self.tamper(lambda c: c.__setitem__('variant', 'T1')), 'config')


class ArgumentErrors(Tiny):
    def test_argument_errors_exit_1_before_any_output_exists(self):
        decision = make_decision(self.root, self.files, self.config)
        output = self.out('args')
        base = fit_argv('T2', decision, self.files, self.config, output, *TINY_ARGS)
        cases = {'no-command': [], 'bare-fit': ['fit'], 'bare-probe': ['probe'], 'bare-select': ['select'],
                 'bare-verify': ['verify'], 'unknown-command': ['compare'], 'unknown-flag': base + ['--bogus'],
                 'abbreviation': [a.replace('--decision-sha', '--decision-s') for a in base],
                 'max-steps-text': base + ['--max-steps', 'many']}
        for name, argv in cases.items():
            with self.subTest(case=name):
                code, out, err = cli(argv)
                self.assertEqual(code, 1, err)
                self.assertEqual(out, '')
                self.assertTrue(err.startswith('ERROR'), err)
        self.assertFalse(output.exists())


class Static(unittest.TestCase):
    def test_module_opens_no_label_query_or_split_data_and_reuses_helpers_by_import(self):
        source = SRC_PATH.read_text()
        for word in ('validation-queries', 'examples.jsonl', 'support.json', 'history', 'target') + LOCAL_MARKERS[:2]:
            self.assertNotIn(word, source, word)
        self.assertRegex(source, r'rr\.fit_r\(')
        self.assertRegex(source, r'rr\.build_decision\(')
        self.assertRegex(source, r'sid_v2\.write_bundle\(')
        self.assertNotIn('def fit(', source)

    def test_new_file_has_no_local_paths_secrets_or_address_literals(self):
        text = SRC_PATH.read_text() + CONFIG_PATH.read_text()
        for word in LOCAL_MARKERS:
            self.assertNotIn(word, text, word)
        self.assertIsNone(re.search(r'\b\d{1,3}(\.\d{1,3}){3}\b', text))
        self.assertLess(SRC_PATH.stat().st_size, 1_000_000)
        self.assertLess(CONFIG_PATH.stat().st_size, 1_000_000)

    def test_frozen_files_are_not_imported_for_editing(self):
        from tests import test_sid_rqvae_r as base
        for path, digest in base.FROZEN_SHA256.items():
            self.assertEqual(file_sha256(ROOT / path), digest, path)


if __name__ == '__main__':
    unittest.main()
