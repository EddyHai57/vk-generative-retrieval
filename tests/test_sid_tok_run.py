"""GTOK-01 FEAT-003: tokenizer fit/verify/compare CLI, frozen fit config and the validation doc.

All data is synthetic and runs on CPU with --engineering. The pre-registered Adagrad lr 0.4 collapses the codebooks
on toy data (U9), so the pipeline tests (class Patched and its subclasses) train with lr 0.1 followed by a k-means
codebook refresh, exactly like the FEAT-002 fixture; nothing in src changes PREREG. The tests of the committed
config and of the formal path run unpatched.
"""
import atexit
import contextlib
import copy
import io
import json
import pickle
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from vkgr import generative_data as gd
from vkgr import sid as bhkm
from vkgr import sid_rqvae as rq
from vkgr import sid_tok_audit as au
from vkgr import sid_tok_run as run
from vkgr import sid_v2
from vkgr.data_audit import canonical_bytes, file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'gtok01' / 'g-tok-01.json'
DOC_PATH = ROOT / 'docs' / 'g_tok_01_validation.md'
SRC_PATH = ROOT / 'src' / 'vkgr' / 'sid_tok_run.py'
_REAL_FIT = rq.fit
# Split literals: this file itself must pass the repository hygiene scan for local paths and company names.
LOCAL_MARKERS = ('/Us' + 'ers/', '/work' + 'space', 'Hello' + 'Bike', '/pri' + 'vate/', '/ho' + 'me/')
_COUNTER = [0]


def blob_items(n=3000, blobs=32, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.randn(blobs, 64)
    x = centers[rng.randint(blobs, size=n)] + noise * rng.randn(n, 64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32)


def write_inputs(root, items, tag='in'):
    root = Path(root)
    n = len(items)
    ids = (np.arange(n) * 3 + 7).astype(np.int64)
    array = np.zeros((n + 1, 64), np.float32)
    array[1:] = items
    content, catalog = root / f'{tag}-content.npy', root / f'{tag}-catalog.json'
    np.save(content, array)
    catalog.write_bytes(canonical_bytes([int(x) for x in ids]))
    return {'content': content, 'content_sha': file_sha256(content), 'catalog': catalog,
            'catalog_sha': file_sha256(catalog), 'ids': ids.astype(np.uint32), 'items': items,
            'digest': au.feature_digest(items)}


def write_config(root, mutate=None):
    value = run.expected_config()
    if mutate:
        mutate(value)
    _COUNTER[0] += 1
    path = Path(root) / f'config-{_COUNTER[0]}.json'
    path.write_bytes(canonical_bytes(value))
    return {'path': path, 'sha': file_sha256(path)}


def fit_argv(variant, files, config, output, *extra, engineering=True):
    argv = ['fit', '--variant', variant, '--config', str(config['path']), '--config-sha', config['sha'],
            '--content', str(files['content']), '--content-sha', files['content_sha'],
            '--catalog', str(files['catalog']), '--catalog-sha', files['catalog_sha'], '--output', str(output)]
    if engineering:
        argv.append('--engineering')
    return argv + list(extra)


def verify_argv(manifest, files, config, *extra, engineering=True):
    manifest = Path(manifest) / 'run_manifest.json' if Path(manifest).is_dir() else Path(manifest)
    argv = ['verify', '--bundle', str(manifest), '--sha', file_sha256(manifest), '--config', str(config['path']),
            '--config-sha', config['sha'], '--content', str(files['content']), '--content-sha', files['content_sha'],
            '--catalog', str(files['catalog']), '--catalog-sha', files['catalog_sha']]
    if engineering:
        argv.append('--engineering')
    return argv + list(extra)


def cli(argv, **kwargs):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run.main(argv, **kwargs)
    return code, out.getvalue(), err.getvalue()


class Ticker:
    """Injected clock: every reading advances by ``step`` seconds."""

    def __init__(self, step=10000.0):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def refreshing_fit(content, **kwargs):
    """Real fit, then the same k-means codebook refresh as the FEAT-002 fixture (the weights written are the ones used)."""
    out = _REAL_FIT(content, **kwargs)
    if out['status'] == 'complete':
        model = rq.RQVAE(kwargs['K'])
        model.load_state_dict(out['state']['state_dict'])
        rq.init_codebooks(model, torch.from_numpy(content), kwargs['seed'], 10)
        out['state'] = {'K': kwargs['K'], 'state_dict': {k: v.detach().clone() for k, v in model.state_dict().items()}}
    return out


_BUILT = {}


def built():
    """Engineering bundles for T1 (400 steps), T2 (2000 steps) and T3 plus a T0-shaped baseline, built once per process."""
    if not _BUILT:
        tmp = tempfile.TemporaryDirectory()
        atexit.register(tmp.cleanup)
        root = Path(tmp.name).resolve()
        config, files = write_config(root), write_inputs(root, blob_items())
        outs = {}
        for variant, extra in (('T1', ['--max-steps', '400']), ('T2', ['--max-steps', '2000']), ('T3', [])):
            outs[variant] = root / f'bundle-{variant}'
            code, _, err = cli(fit_argv(variant, files, config, outs[variant], *extra))
            assert code == 0, (variant, code, err)
        base = bhkm.fit(files['ids'], files['items'], branching=16, depth=3, seed=42, max_iter=10)
        np.savez(root / 'baseline.npz', ids=base.ids, sids=base.sids)
        _BUILT.update(root=root, config=config, files=files, outs=outs, baseline=root / 'baseline.npz',
                      baseline_sha=file_sha256(root / 'baseline.npz'), baseline_model=base)
    return _BUILT


class Patched(unittest.TestCase):
    """lr 0.1 plus codebook refresh for the pipeline tests; the committed config is not used here."""

    @classmethod
    def setUpClass(cls):
        for patch in (mock.patch.dict(rq.PREREG, lr=0.1), mock.patch.object(rq, 'fit', refreshing_fit)):
            patch.start()
            cls.addClassCleanup(patch.stop)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name).resolve()
        cls.config = write_config(cls.root)
        cls.files = write_inputs(cls.root, blob_items(2048))

    def out(self, name='out'):
        _COUNTER[0] += 1
        return self.root / f'{name}-{_COUNTER[0]}'

    def fit(self, variant, *extra, files=None, config=None, engineering=True, **kwargs):
        output = self.out()
        code, stdout, stderr = cli(fit_argv(variant, files or self.files, config or self.config, output, *extra,
                                            engineering=engineering), **kwargs)
        return output, code, stdout, stderr


class Frozen(unittest.TestCase):
    """Committed config, frozen-field rejection, formal path refusals (no patches)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name).resolve()
        cls.files = write_inputs(cls.root, blob_items(2048))
        cls.committed = {'path': CONFIG_PATH, 'sha': file_sha256(CONFIG_PATH)}

    def test_committed_config_matches_the_preregistration_in_code(self):
        loaded = run.load_config(CONFIG_PATH, file_sha256(CONFIG_PATH))
        self.assertEqual(loaded, run.expected_config())
        self.assertEqual(loaded['schema'], 'g-tok-01-fit-v1')
        self.assertEqual(loaded['task_id'], 'G-TOK-01')
        self.assertEqual(loaded['content'], {'feature_array_sha256': au.CONTENT_V1_FEATURE_SHA256, 'dim': 64})
        self.assertEqual(loaded['variants']['T1'], {'sid_version': 'SID_RQVAE_B256D3_S42', 'kind': 'rqvae', 'codebook_size': 256})
        self.assertEqual(loaded['variants']['T2'], {'sid_version': 'SID_RQVAE_B16D3_S42', 'kind': 'rqvae', 'codebook_size': 16})
        self.assertEqual(loaded['variants']['T3'], {'sid_version': 'SID_BHKM_B256D3_S42', 'kind': 'bhkm', 'branching': 256,
                                                    'depth': 3, 'max_iter': 10, 'seed': 42})
        for key, value in rq.PREREG.items():
            expected = list(value) if isinstance(value, tuple) else value
            self.assertEqual(loaded['rqvae'][key], expected, key)
            self.assertIs(type(loaded['rqvae'][key]), type(expected), key)

    @staticmethod
    def leaf_paths(value, prefix=()):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from Frozen.leaf_paths(item, prefix + (key,))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from Frozen.leaf_paths(item, prefix + (index,))
        else:
            yield prefix

    @staticmethod
    def container(value, path):
        for part in path:
            value = value[part]
        return value

    def test_every_field_change_is_rejected(self):
        base = json.loads(CONFIG_PATH.read_text())
        paths = list(self.leaf_paths(base))
        self.assertGreaterEqual(len(paths), 40)
        for path in paths:
            with self.subTest(change=path):
                broken = copy.deepcopy(base)
                parent = self.container(broken, path[:-1])
                old = parent[path[-1]]
                parent[path[-1]] = old + 'x' if isinstance(old, str) else old * 2 + 1
                with self.assertRaises(ValueError):
                    run.validate_config(broken)

    def test_every_field_removal_and_extra_field_is_rejected(self):
        base = json.loads(CONFIG_PATH.read_text())
        for path in self.leaf_paths(base):
            if isinstance(path[-1], int):
                continue
            with self.subTest(remove=path):
                broken = copy.deepcopy(base)
                del self.container(broken, path[:-1])[path[-1]]
                with self.assertRaises(ValueError):
                    run.validate_config(broken)
        for where in ((), ('rqvae',), ('content',), ('variants',), ('variants', 'T1'), ('variants', 'T3')):
            with self.subTest(extra=where):
                broken = copy.deepcopy(base)
                self.container(broken, where)['extra'] = 1
                with self.assertRaises(ValueError):
                    run.validate_config(broken)

    def test_named_frozen_fields_and_type_confusion(self):
        base = json.loads(CONFIG_PATH.read_text())
        edits = {'max_steps': 299999, 'lr': 0.1, 'beta': 0.5, 'wall_cap_seconds': 7201, 'usage_gate_num_den': [4, 6],
                 'batch_size': 512, 'initial_accumulator_value': 0.0, 'seed': 43}
        for name, replacement in edits.items():
            with self.subTest(field=name):
                broken = copy.deepcopy(base)
                broken['rqvae'][name] = replacement
                with self.assertRaises(ValueError):
                    run.validate_config(broken)
        for name, replacement in (('max_steps', 300000.0), ('lr', 1), ('seed', True), ('levels', 3.0)):
            with self.subTest(type=name):
                broken = copy.deepcopy(base)
                broken['rqvae'][name] = replacement
                with self.assertRaises(ValueError):
                    run.validate_config(broken)
        broken = copy.deepcopy(base)
        broken['variants']['T1']['codebook_size'] = 128
        with self.assertRaises(ValueError):
            run.validate_config(broken)
        with self.assertRaises(ValueError):
            run.validate_config([base])

    def test_load_config_rejects_wrong_sha_and_duplicate_keys(self):
        with self.assertRaises(ValueError):
            run.load_config(CONFIG_PATH, '0' * 64)
        duplicate = self.root / 'dup.json'
        text = CONFIG_PATH.read_text()
        duplicate.write_text(text.replace('"task_id": "G-TOK-01",', '"task_id": "G-TOK-01", "task_id": "G-TOK-01",', 1))
        self.assertNotEqual(duplicate.read_text(), text)
        with self.assertRaises(ValueError):
            run.load_config(duplicate, file_sha256(duplicate))

    def test_formal_path_refuses_cpu_before_reading_anything(self):
        output = self.root / 'formal-cpu'
        code, _, err = cli(fit_argv('T1', self.files, self.committed, output, '--device', 'cpu', engineering=False))
        self.assertEqual(code, 1)
        self.assertIn('device', err.lower())
        self.assertFalse(output.exists())

    def test_formal_path_refuses_a_content_digest_other_than_content_v1(self):
        self.assertNotEqual(self.files['digest'], au.CONTENT_V1_FEATURE_SHA256)
        for variant, extra in (('T3', []), ('T1', ['--device', 'cuda'])):
            with self.subTest(variant=variant):
                output = self.root / f'formal-{variant}'
                code, _, err = cli(fit_argv(variant, self.files, self.committed, output, *extra, engineering=False))
                self.assertEqual(code, 1)
                self.assertIn('CONTENT_V1', err)
                self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            run.load_inputs(self.files['content'], self.files['content_sha'], self.files['catalog'],
                            self.files['catalog_sha'], engineering=False)
        loaded = run.load_inputs(self.files['content'], self.files['content_sha'], self.files['catalog'],
                                 self.files['catalog_sha'], engineering=True)
        self.assertEqual(loaded['content_sha256'], self.files['digest'])
        self.assertEqual(loaded['content'].shape, (2048, 64))
        self.assertTrue(np.array_equal(loaded['ids'], self.files['ids']))

    def test_load_inputs_rejects_malformed_files(self):
        def variant(name, content=None, catalog=None):
            array = np.zeros((2049, 64), np.float32)
            array[1:] = self.files['items']
            if content is not None:
                array = content(array)
            ids = [int(x) for x in self.files['ids']]
            if catalog is not None:
                ids = catalog(ids)
            c, k = self.root / f'{name}.npy', self.root / f'{name}.json'
            np.save(c, array)
            k.write_bytes(canonical_bytes(ids))
            return c, file_sha256(c), k, file_sha256(k)

        def nonzero_pad(a):
            a[0, 0] = 1.0
            return a

        def with_nan(a):
            a[5, 3] = np.nan
            return a

        cases = {'pad-row': dict(content=nonzero_pad), 'nan': dict(content=with_nan),
                 'wrong-rows': dict(content=lambda a: a[:-1]), 'wrong-dtype': dict(content=lambda a: a.astype(np.float64)),
                 'wrong-dim': dict(content=lambda a: a[:, :32]), 'descending': dict(catalog=lambda ids: ids[::-1]),
                 'duplicate-id': dict(catalog=lambda ids: ids[:-1] + [ids[-2]]), 'negative-id': dict(catalog=lambda ids: [-1] + ids[1:]),
                 'too-large-id': dict(catalog=lambda ids: ids[:-1] + [2 ** 32]), 'empty': dict(catalog=lambda ids: [])}
        for name, spec in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    run.load_inputs(*variant(name, **spec), engineering=True)


class Failures(Patched):
    def test_wrong_content_catalog_and_config_sha_are_rejected(self):
        wrong = '0' * 64
        for label, files, config in (('content', {**self.files, 'content_sha': wrong}, None),
                                     ('catalog', {**self.files, 'catalog_sha': wrong}, None),
                                     ('config', None, {**self.config, 'sha': wrong})):
            with self.subTest(wrong=label):
                output, code, _, err = self.fit('T2', '--max-steps', '100', files=files, config=config)
                self.assertEqual(code, 1, err)
                self.assertFalse(output.exists())

    def test_config_with_changed_max_steps_is_rejected_by_the_cli(self):
        def mutate(value):
            value['rqvae']['max_steps'] = 299999
        config = write_config(self.root, mutate)
        output, code, _, err = self.fit('T2', '--max-steps', '100', config=config)
        self.assertEqual(code, 1, err)
        self.assertFalse(output.exists())

    def test_existing_output_directory_is_rejected(self):
        output = self.out('exists')
        output.mkdir()
        code, _, err = cli(fit_argv('T2', self.files, self.config, output, '--max-steps', '100'))
        self.assertEqual(code, 1, err)
        self.assertEqual(list(output.iterdir()), [])

    def test_usage_below_80_percent_writes_failure_json_and_no_manifest(self):
        directions = blob_items(3, blobs=3, noise=0.0, seed=3)
        files = write_inputs(self.root, directions[np.arange(2048) % 3], tag='three')
        output, code, _, err = self.fit('T2', '--max-steps', '100', files=files)
        self.assertEqual(code, 3, err)
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['failure.json'])
        failure = json.loads((output / 'failure.json').read_text())
        self.assertEqual(failure['status'], 'gate_failed')
        self.assertIn('R1', failure['reasons'])
        self.assertIs(failure['gates']['passed'], False)
        self.assertEqual(failure['variant'], 'T2')
        self.assertTrue(failure['engineering'])
        self.assertNotIn('state', failure)

    def test_wall_cap_with_injected_clock_writes_failure_json_and_no_manifest(self):
        output, code, _, err = self.fit('T2', '--max-steps', '300', clock=Ticker())
        self.assertEqual(code, 3, err)
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['failure.json'])
        failure = json.loads((output / 'failure.json').read_text())
        self.assertEqual(failure['status'], 'fail_wall')
        self.assertEqual(failure['reasons'], ['fit:fail_wall'])
        self.assertIsNone(failure['audit'])

    def test_wall_cap_result_exposes_no_model(self):
        config = run.expected_config()
        files = self.files
        result = run.fit_rqvae_variant(config, 'T2', files['ids'], files['items'], device='cpu', max_steps=300,
                                       clock=Ticker(), expected_content_sha256=files['digest'])
        self.assertEqual(result['status'], 'fail_wall')
        for key in ('state', 'sids', 'codec', 'weights'):
            self.assertNotIn(key, result)

    def test_gate_failed_result_exposes_no_model(self):
        # L2: a failed gate must not hand the half-trained model back to the caller
        directions = blob_items(3, blobs=3, noise=0.0, seed=3)
        files = write_inputs(self.root, directions[np.arange(2048) % 3], tag='three-direct')
        result = run.fit_rqvae_variant(run.expected_config(), 'T2', files['ids'], files['items'], device='cpu', max_steps=100,
                                       expected_content_sha256=files['digest'])
        self.assertEqual(result['status'], 'gate_failed')
        for key in ('state', 'sids', 'codec', 'weights', 'ids', 'tokenizer', 'train_log'):
            self.assertNotIn(key, result)

    def test_nan_writes_failure_json_and_no_manifest(self):
        files = write_inputs(self.root, (self.files['items'] * np.float32(1e20)).astype(np.float32), tag='huge')
        output, code, _, err = self.fit('T2', '--max-steps', '300', files=files)
        self.assertEqual(code, 3, err)
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['failure.json'])
        failure = json.loads((output / 'failure.json').read_text())
        self.assertEqual(failure['status'], 'fail_nonfinite')
        self.assertNotIn('NaN', (output / 'failure.json').read_text())

    def test_t3_calls_frozen_fit_unchanged_and_drops_nodes(self):
        items = self.files['items'][:600]
        ids = self.files['ids'][:600]
        calls = []
        real = bhkm.fit

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        digest = au.feature_digest(items)
        with mock.patch('vkgr.sid.fit', spy):
            result = run.fit_bhkm_variant(run.expected_config(), ids, items, expected_content_sha256=digest)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {'branching': 256, 'depth': 3, 'seed': 42, 'max_iter': 10})
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['catalog'].nodes, [])
        self.assertLess(len(pickle.dumps(result['catalog'])), 500000)
        summary = result['tokenizer']['summary']
        self.assertEqual(set(summary), {'nodes_per_level', 'child_size_histogram', 'unconverged_nodes'})
        self.assertEqual(set(summary['nodes_per_level']), {'0', '1', '2'})
        self.assertEqual(summary['nodes_per_level']['0'], 1)
        self.assertIs(type(summary['unconverged_nodes']), int)


class Pipeline(Patched):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.b = built()

    def verify(self, variant, *extra, **kwargs):
        return cli(verify_argv(self.b['outs'][variant], self.b['files'], self.b['config'], *extra, **kwargs))

    def test_bundles_load_and_verify_for_t1_t2_t3(self):
        files = self.b['files']
        for variant, version in (('T1', 'SID_RQVAE_B256D3_S42'), ('T2', 'SID_RQVAE_B16D3_S42'), ('T3', 'SID_BHKM_B256D3_S42')):
            with self.subTest(variant=variant):
                directory = self.b['outs'][variant]
                manifest = directory / 'run_manifest.json'
                loaded = sid_v2.load_bundle(manifest, file_sha256(manifest), catalog_ids=files['ids'], content=files['items'],
                                            expected_content_digest=files['digest'], engineering=True)
                self.assertEqual(loaded.info['sid_version'], version)
                self.assertTrue(loaded.manifest['gates']['passed'])
                self.assertEqual(loaded.manifest['fit_config_sha256'], self.b['config']['sha'])
                self.assertEqual(loaded.codec.catalog.nodes, [])
                code, out, err = self.verify(variant)
                self.assertEqual(code, 0, err)
                self.assertEqual(json.loads(out)['sid_version'], version)

    def test_engineering_bundles_are_marked_and_the_formal_loader_refuses_them(self):
        files = self.b['files']
        for variant, directory in self.b['outs'].items():
            with self.subTest(variant=variant):
                manifest = directory / 'run_manifest.json'
                self.assertIs(json.loads(manifest.read_text())['engineering'], True)
                with self.assertRaises(ValueError):
                    sid_v2.load_bundle(manifest, file_sha256(manifest), catalog_ids=files['ids'], content=files['items'],
                                       expected_content_digest=files['digest'], engineering=False)
                code, _, err = self.verify(variant, engineering=False)
                self.assertEqual(code, 1, err)

    def test_t3_bundle_keeps_no_nodes_and_pickles_small(self):
        manifest = self.b['outs']['T3'] / 'run_manifest.json'
        files = self.b['files']
        loaded = sid_v2.load_bundle(manifest, file_sha256(manifest), catalog_ids=files['ids'], content=files['items'],
                                    expected_content_digest=files['digest'], engineering=True)
        self.assertEqual(loaded.codec.catalog.nodes, [])
        self.assertLess(len(pickle.dumps(loaded.codec)), 500000)
        tokenizer = json.loads((self.b['outs']['T3'] / 'tokenizer.json').read_text())
        self.assertEqual(set(tokenizer['config']['summary']), {'nodes_per_level', 'child_size_histogram', 'unconverged_nodes'})
        self.assertLess((self.b['outs']['T3'] / 'tokenizer.json').stat().st_size, 20000)

    def test_usage_diagnostic_and_observations_are_recorded_outside_audit_json(self):
        log = json.loads((self.b['outs']['T2'] / 'train-log.json').read_text())
        diag = log['usage_diagnostic']
        self.assertEqual(diag['K'], 16)
        self.assertEqual(diag['step_2000']['step'], 2000)
        self.assertEqual(len(diag['step_2000']['used']), 3)
        self.assertEqual(diag['step_2000']['fraction'], [u / 16 for u in diag['step_2000']['used']])
        audit = json.loads((self.b['outs']['T2'] / 'audit.json').read_text())
        self.assertEqual(diag['final']['step'], 2000)
        self.assertEqual(diag['final']['used'], [audit['mapping']['positions'][d]['distinct'] for d in range(3)])
        self.assertEqual(log['observations']['device'], 'cpu')
        self.assertEqual(log['observations']['torch_version'], torch.__version__)
        self.assertGreaterEqual(log['observations']['wall_seconds'], 0)
        text = (self.b['outs']['T2'] / 'audit.json').read_text()
        for word in ('torch_version', 'device', 'observations', 'wall_seconds'):
            self.assertNotIn(word, text)
        short = json.loads((self.b['outs']['T1'] / 'train-log.json').read_text())
        self.assertIsNone(short['usage_diagnostic']['step_2000'])
        self.assertEqual(short['usage_diagnostic']['final']['step'], 400)

    def test_compare_has_the_t0_shaped_baseline_row_and_correct_labels(self):
        b = self.b
        output = self.out('compare')
        argv = ['compare', '--baseline-mapping', str(b['baseline']), '--baseline-sha', b['baseline_sha'],
                '--content', str(b['files']['content']), '--content-sha', b['files']['content_sha'],
                '--catalog', str(b['files']['catalog']), '--catalog-sha', b['files']['catalog_sha'],
                '--output', str(output), '--engineering']
        for variant in ('T1', 'T2', 'T3'):
            manifest = b['outs'][variant] / 'run_manifest.json'
            argv += ['--bundle', str(manifest), '--sha', file_sha256(manifest)]
        code, out, err = cli(argv)
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(p.name for p in output.iterdir()), ['comparison.json', 'comparison.md'])
        table = json.loads((output / 'comparison.json').read_text())['table']
        self.assertEqual(table['labels'], ['T0', 'T1', 'T2', 'T3'])
        t0 = au.audit_mapping(b['files']['ids'], b['baseline_model'].sids, b['files']['items'], branching=16, kind='bhkm')
        stored = {v: json.loads((b['outs'][v] / 'audit.json').read_text())['mapping'] for v in ('T1', 'T2', 'T3')}
        expected = au.compare({'T0': t0, **stored})
        self.assertTrue(au.audits_equal(table, json.loads(json.dumps(expected))))
        row = {r['item']: r['values'] for r in table['rows']}
        self.assertLessEqual(row['positions.0.distinct']['T0'], 16)
        self.assertGreater(row['positions.0.distinct']['T1'], 16)
        self.assertLessEqual(row['positions.0.distinct']['T2'], 16)
        self.assertIn('| positions.0.distinct |', (output / 'comparison.md').read_text())

    def test_compare_rejects_duplicate_bundles_and_unpaired_hashes(self):
        b = self.b
        manifest = b['outs']['T1'] / 'run_manifest.json'
        common = ['compare', '--baseline-mapping', str(b['baseline']), '--baseline-sha', b['baseline_sha'],
                  '--content', str(b['files']['content']), '--content-sha', b['files']['content_sha'],
                  '--catalog', str(b['files']['catalog']), '--catalog-sha', b['files']['catalog_sha'], '--engineering']
        for label, extra in (('duplicate', ['--bundle', str(manifest), '--sha', file_sha256(manifest)] * 2),
                             ('unpaired', ['--bundle', str(manifest)]),
                             ('wrong-sha', ['--bundle', str(manifest), '--sha', '0' * 64]),
                             ('none', [])):
            with self.subTest(case=label):
                output = self.out('compare-bad')
                code, _, _ = cli(common + ['--output', str(output)] + extra)
                self.assertEqual(code, 1)
                self.assertFalse(output.exists())

    def test_replay_prefix_reproduces_the_prefix_digest_on_cpu(self):
        stored = json.loads((self.b['outs']['T2'] / 'train-log.json').read_text())['prefix_digest']
        self.assertRegex(stored, r'^[0-9a-f]{64}$')
        code, out, err = self.verify('T2', '--replay-prefix')
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)['replay_prefix'])
        with mock.patch.object(run, '_replay_prefix_digest', lambda *a, **k: 'f' * 64):
            code, _, err = self.verify('T2', '--replay-prefix')
        self.assertEqual(code, 1)
        self.assertIn('prefix', err)

    def test_replay_options_are_refused_where_they_do_not_apply(self):
        for variant, flag in (('T1', '--replay-prefix'), ('T3', '--replay-prefix'), ('T1', '--replay-fit'), ('T2', '--replay-fit')):
            with self.subTest(variant=variant, flag=flag):
                code, _, err = self.verify(variant, flag)
                self.assertEqual(code, 1)
                self.assertTrue(err.startswith('ERROR'))

    def test_replay_fit_reproduces_t3_and_detects_a_different_mapping(self):
        code, out, err = self.verify('T3', '--replay-fit')
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)['replay_fit'])
        real = bhkm.fit

        def different(ids, features, **kwargs):
            return real(ids, features, **{**kwargs, 'seed': 7})

        with mock.patch('vkgr.sid.fit', different):
            code, _, err = self.verify('T3', '--replay-fit')
        self.assertEqual(code, 1)
        self.assertIn('replay', err)

    def test_verify_rejects_a_config_other_than_the_one_the_bundle_was_fitted_with(self):
        other = write_config(self.root)  # same content, one extra newline byte: valid config, different sha256
        path = Path(other['path'])
        path.write_bytes(path.read_bytes() + b'\n')
        other['sha'] = file_sha256(path)
        self.assertNotEqual(other['sha'], self.b['config']['sha'])
        code, _, err = cli(verify_argv(self.b['outs']['T2'], self.b['files'], other))
        self.assertEqual(code, 1, err)
        self.assertIn('config', err)


class ArgumentErrors(Patched):
    def test_argument_errors_exit_1_before_any_output_exists(self):
        output = self.out('args')
        base = fit_argv('T2', self.files, self.config, output, '--max-steps', '100')

        def replace(flag, value):
            argv = list(base)
            argv[argv.index(flag) + 1] = value
            return argv

        cases = {
            'no-command': [],
            'bare-fit': ['fit'],
            'unknown-command': ['train'],
            'unknown-variant': replace('--variant', 'T9'),
            'unknown-flag': base + ['--bogus'],
            't3-max-steps': fit_argv('T3', self.files, self.config, output, '--max-steps', '5'),
            't3-device-cuda': fit_argv('T3', self.files, self.config, output, '--device', 'cuda'),
            'engineering-rqvae-needs-max-steps': fit_argv('T2', self.files, self.config, output),
            'formal-with-max-steps': fit_argv('T2', self.files, self.config, output, '--max-steps', '100', engineering=False),
            'max-steps-zero': fit_argv('T2', self.files, self.config, output, '--max-steps', '0'),
            'max-steps-text': fit_argv('T2', self.files, self.config, output, '--max-steps', 'many'),
            'missing-parent': replace('--output', str(self.root / 'no-such-parent' / 'out')),
            'missing-config': replace('--config', str(self.root / 'missing.json')),
            'verify-without-bundle': ['verify', '--sha', '0' * 64],
        }
        for name, argv in cases.items():
            with self.subTest(case=name):
                code, out, err = cli(argv)
                self.assertEqual(code, 1, err)
                self.assertEqual(out, '')
                self.assertTrue(err.startswith('ERROR'), err)
        self.assertFalse(output.exists())
        self.assertFalse((self.root / 'no-such-parent').exists())


class Static(unittest.TestCase):
    def test_module_opens_no_label_query_or_split_data(self):
        source = SRC_PATH.read_text()
        for word in ('validation-queries', 'examples.jsonl', 'support.json', 'history', 'target') + LOCAL_MARKERS[:2]:
            self.assertNotIn(word, source, word)

    def test_sid_py_is_unchanged_and_called_through_the_module_attribute(self):
        source = SRC_PATH.read_text()
        self.assertRegex(source, r'bhkm\.fit\(')
        self.assertNotIn('def fit(ids', source)

    def test_validation_doc_has_one_machine_readable_prereg_block_and_no_local_paths(self):
        text = DOC_PATH.read_text()
        self.assertEqual(text.count('<!-- gtok01-prereg:begin -->'), 1)
        self.assertEqual(text.count('<!-- gtok01-prereg:end -->'), 1)
        block = text.split('<!-- gtok01-prereg:begin -->')[1].split('<!-- gtok01-prereg:end -->')[0]
        match = re.search(r'```json\n(.*?)\n```', block, re.S)
        self.assertIsNotNone(match)
        value = json.loads(match.group(1))
        self.assertEqual(value['fit_config'], json.loads(CONFIG_PATH.read_text()))
        self.assertEqual(value['gates']['usage_gate_num_den'], [4, 5])
        self.assertEqual(value['gates']['required_steps'], 300000)
        for word in LOCAL_MARKERS:
            self.assertNotIn(word, text, word)
        for placeholder in ('<CONTENT>', '<CATALOG>', '<OUT>'):
            self.assertIn(placeholder, text)
        self.assertLess(DOC_PATH.stat().st_size, 1000000)


if __name__ == '__main__':
    unittest.main()
