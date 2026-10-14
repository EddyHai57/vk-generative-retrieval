"""GTOK-01 FEAT-002: SIDMAP_V2 bundle contract (write/load round trip, fail-closed negatives, isolation)."""
import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vkgr import generative_data as gd
from vkgr import sid_rqvae as rq
from vkgr import sid_run
from vkgr import sid_tok_audit as au
from vkgr import sid_v2
from vkgr.data_audit import canonical_bytes, file_sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_KEYS = {'schema', 'protocol', 'task_id', 'status', 'sid_version', 'kind', 'engineering', 'branching',
                 'catalog_items', 'suffix_vocabulary', 'vocab_size', 'content_feature_sha256', 'catalog_sha256',
                 'artifacts', 'fit_config_sha256', 'producer', 'gates'}
PRODUCER = {'src/vkgr/sid_v2.py': 'a' * 64, 'src/vkgr/sid_tok_audit.py': 'b' * 64}
CONFIG_SHA = 'c' * 64
TOKENIZER = {'variant': 'synthetic', 'beta': 0.25}


def balanced_sids(n, b):
    sids = np.zeros((n, 4), dtype=np.uint32)

    def split(lo, hi, depth):
        size = hi - lo
        if depth == 3:
            sids[lo:hi, 3] = np.arange(size)
            return
        k = min(b, size)
        base, extra = divmod(size, k)
        position = lo
        for child in range(k):
            width = base + (1 if child < extra else 0)
            sids[position:position + width, depth] = child
            split(position, position + width, depth + 1)
            position += width

    split(0, n, 0)
    return sids


class World:
    """Everything needed to write one bundle (kept in memory; bundles are written into temp dirs)."""

    def __init__(self, version, kind, branching, engineering, ids, sids, content, weights=None, train_log=None, rq_info=None,
                 required_steps=au.REQUIRED_STEPS):
        self.version, self.kind, self.branching, self.engineering = version, kind, branching, engineering
        self.ids, self.sids, self.content, self.weights, self.train_log = ids, sids, content, weights, train_log
        self.digest = au.feature_digest(content)
        mapping = au.audit_mapping(ids, sids, content, branching=branching, kind=kind)
        self.audit = {'mapping': mapping, 'rq': rq_info}
        self.gates = au.gates(mapping, kind, rq_info, expected_content_sha256=self.digest, required_steps=required_steps)

    def write(self, output, **override):
        args = dict(sid_version=self.version, engineering=self.engineering, ids=self.ids, sids=self.sids,
                    content_feature_sha256=self.digest, tokenizer=TOKENIZER, audit=self.audit, gates=self.gates,
                    fit_config_sha256=CONFIG_SHA, producer=PRODUCER, weights=self.weights, train_log=self.train_log)
        args.update(override)
        return sid_v2.write_bundle(output, **args)

    def load(self, directory, sha=None, **override):
        directory = Path(directory)
        args = dict(catalog_ids=self.ids, content=self.content, expected_content_digest=self.digest,
                    engineering=self.engineering)
        args.update(override)
        return sid_v2.load_bundle(directory / 'run_manifest.json', sha or file_sha256(directory / 'run_manifest.json'), **args)


def same_leaf_pair(sids):
    """Two row indices that share their first three codes."""
    _, inverse, counts = np.unique(sids[:, :3], axis=0, return_inverse=True, return_counts=True)
    leaf = int(np.flatnonzero(counts >= 2)[0])
    rows = np.flatnonzero(inverse.ravel() == leaf)
    return int(rows[0]), int(rows[1])


def bhkm_world(version='SID_V2_REPLAY_BHKM_B16D3', branching=16, engineering=True, n=6000):
    ids = (np.arange(n) * 3 + 5).astype(np.uint32)
    content = np.random.RandomState(7).randn(n, 64).astype(np.float32)
    return World(version, 'bhkm', branching, engineering, ids, balanced_sids(n, branching), content)


def rqvae_world(seed=42):
    """Toy RQ-VAE: 16 unit-norm blobs, K=16. The pre-registered lr 0.4 collapses on this toy data (U9), so the
    fixture trains with lr 0.1 and then refreshes the three codebooks with k-means on all rows; the mapping is
    always exported from exactly the weights that are written, so weights and mapping agree by construction."""
    import torch
    rng = np.random.RandomState(5)
    centers = rng.randn(16, 64)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    content = centers[np.repeat(np.arange(16), 100)] + rng.randn(1600, 64) * 0.05
    content = (content / np.linalg.norm(content, axis=1, keepdims=True)).astype(np.float32)
    steps = 1000
    with mock.patch.dict(rq.PREREG, lr=0.1):
        out = rq.fit(content, K=16, seed=seed, max_steps=steps, batch_size=256, wall_cap_seconds=1e9, probe_from=100, probe_step=200)
    assert out['status'] == 'complete', out['status']
    model = rq.RQVAE(16)
    model.load_state_dict(out['state']['state_dict'])
    rq.init_codebooks(model, torch.from_numpy(content), seed, 10)
    state = {'K': 16, 'state_dict': {k: v.detach().clone() for k, v in model.state_dict().items()}}
    ids = (np.arange(1600) * 2 + 3).astype(np.uint32)
    sids = rq.assign_fourth_code(ids, rq.export_codes(state, content))
    info = {'status': 'complete', 'steps': steps, 'max_steps': steps, 'elapsed': float(out['elapsed']),
            'wall_cap_seconds': 7200, 'required_steps': steps, 'reconstruction': rq.reconstruction_report(state, content)}
    log = {'loss_log': out['loss_log'], 'usage_log': out['usage_log'], 'init_digest': out['init_digest'],
           'prefix_digest': out['prefix_digest']}
    return World('SID_RQVAE_B16D3_S42', 'rqvae', 16, True, ids, sids, content, weights=state, train_log=log,
                 rq_info=info, required_steps=steps)


_CACHE = {}


def world(name):
    if name not in _CACHE:
        _CACHE[name] = {'bhkm': bhkm_world, 'rqvae': rqvae_world,
                        'formal': lambda: bhkm_world('SID_BHKM_B256D3_S42', 256, False)}[name]()
    return _CACHE[name]


def rewrite_manifest(directory, mutate):
    path = Path(directory) / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_bytes(canonical_bytes(manifest))
    return file_sha256(path)


def tamper(directory, name, data):
    """Replace one artifact and re-sign the manifest so that only the content of the artifact is wrong."""
    directory = Path(directory)
    (directory / name).write_bytes(data)

    def update(manifest):
        manifest['artifacts'][name] = file_sha256(directory / name)

    return rewrite_manifest(directory, update)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def bundle(self, name, copy_name=None):
        """Write (or copy) a fresh valid bundle and return its directory."""
        w = world(name)
        target = self.root / (copy_name or name)
        if not target.exists():
            w.write(target)
        return target

    def copy_of(self, source, name='copy'):
        target = self.root / name
        shutil.copytree(source, target)
        return target


class Constants(unittest.TestCase):
    def test_registry_and_constants(self):
        self.assertEqual((sid_v2.SCHEMA, sid_v2.PROTOCOL, sid_v2.TASK_ID), ('sidmap-v2', 'generative-like-sid-v2', 'G-TOK-01'))
        self.assertEqual(sid_v2.SID_VERSIONS, ('SID_RQVAE_B256D3_S42', 'SID_RQVAE_B16D3_S42', 'SID_BHKM_B256D3_S42'))
        self.assertEqual(sid_v2.ENGINEERING_SID_VERSIONS, sid_v2.SID_VERSIONS + ('SID_V2_REPLAY_BHKM_B16D3',))
        self.assertEqual(sid_v2.CONTENT_V1_FEATURE_SHA256, '766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84')
        self.assertEqual(set(sid_v2.MANIFEST_KEYS), MANIFEST_KEYS)
        for old in ('SID_BHKM_B16D3_S42', 'SID_RANDPERM42_OF_BHKM_B16D3'):
            self.assertNotIn(old, sid_v2.ENGINEERING_SID_VERSIONS)

    def test_canonical_sids_sorts_by_id_and_assigns_canonical_suffix(self):
        ids = np.array([30, 10, 20, 40], dtype=np.uint32)
        codes = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]])
        out_ids, sids = sid_v2.canonical_sids(ids, codes)
        self.assertEqual(out_ids.tolist(), [10, 20, 30, 40])
        self.assertEqual(sids.tolist(), [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
        self.assertEqual((out_ids.dtype, sids.dtype), (np.uint32, np.uint32))

    def test_catalog_digest_matches_the_context_function(self):
        import hashlib
        ids = np.array([3, 5, 9], dtype=np.uint32)
        self.assertEqual(sid_v2.catalog_digest(ids), hashlib.sha256(canonical_bytes([3, 5, 9])).hexdigest())


class BuildCodec(unittest.TestCase):
    def test_accepts_canonical_mapping(self):
        w = world('bhkm')
        codec = sid_v2.build_codec(w.ids, w.sids, 16)
        self.assertIsInstance(codec, gd._ValidatedCodec)
        codec.check_identity()
        self.assertEqual(codec.catalog.nodes, [])
        self.assertEqual(codec.vocab_size, 2 + 3 * 16 + codec.catalog.suffix_vocabulary)

    def test_rejects_noncanonical_suffix_bad_digits_and_bad_shapes(self):
        w = world('bhkm')
        first, second = same_leaf_pair(w.sids)
        swapped = w.sids.copy()
        swapped[[first, second], 3] = swapped[[second, first], 3]
        high = w.sids.copy()
        high[0, 0] = 16
        for name, ids, sids, b in (('swapped suffix', w.ids, swapped, 16), ('digit >= branching', w.ids, high, 16),
                                   ('branching too small', w.ids, w.sids, 8), ('int64 ids', w.ids.astype(np.int64), w.sids, 16),
                                   ('wrong width', w.ids, w.sids[:, :3], 16), ('unsorted ids', w.ids[::-1].copy(), w.sids, 16)):
            with self.subTest(name), self.assertRaises(ValueError):
                sid_v2.build_codec(ids, sids, b)

    def test_catalog_equality_is_part_of_the_codec_check(self):
        w = world('bhkm')
        with self.assertRaises(ValueError):
            sid_v2.build_codec(w.ids, w.sids, 16, catalog_ids=w.ids + 1)
        sid_v2.build_codec(w.ids, w.sids, 16, catalog_ids=w.ids)


class RoundTrip(Base):
    def test_bhkm_engineering_round_trip_and_manifest_shape(self):
        w = world('bhkm')
        manifest = w.write(self.root / 'a')
        self.assertEqual(set(manifest), MANIFEST_KEYS)
        self.assertEqual(sorted(os.listdir(self.root / 'a')), ['audit.json', 'mapping.npz', 'run_manifest.json', 'tokenizer.json'])
        loaded = w.load(self.root / 'a')
        self.assertIsInstance(loaded, sid_v2.LoadedSID)
        self.assertEqual(loaded.manifest, manifest)
        self.assertEqual(loaded.manifest['status'], 'complete')
        self.assertEqual(loaded.manifest['gates']['passed'], True)
        self.assertEqual(loaded.info['sid_version'], 'SID_V2_REPLAY_BHKM_B16D3')
        self.assertEqual(loaded.info['manifest_sha256'], file_sha256(self.root / 'a' / 'run_manifest.json'))
        self.assertEqual((loaded.info['branching'], loaded.info['suffix_vocabulary'], loaded.info['vocab_size']),
                         (16, loaded.codec.catalog.suffix_vocabulary, loaded.codec.vocab_size))
        self.assertTrue(np.array_equal(loaded.codec.catalog.sids, w.sids))
        loaded.codec.check_identity()
        self.assertEqual(manifest['catalog_sha256'], sid_v2.catalog_digest(w.ids))
        self.assertEqual(manifest['content_feature_sha256'], w.digest)

    def test_rqvae_round_trip_has_weights_and_train_log(self):
        w = world('rqvae')
        self.assertTrue(w.gates['passed'], w.gates)
        manifest = w.write(self.root / 'r')
        self.assertEqual(sorted(os.listdir(self.root / 'r')),
                         ['audit.json', 'mapping.npz', 'run_manifest.json', 'tokenizer.json', 'train-log.json', 'weights.npz'])
        self.assertEqual(set(manifest['artifacts']), {'mapping.npz', 'tokenizer.json', 'weights.npz', 'audit.json', 'train-log.json'})
        loaded = w.load(self.root / 'r')
        self.assertEqual(loaded.info['sid_version'], 'SID_RQVAE_B16D3_S42')

    def test_formal_bundle_round_trip_and_default_pin(self):
        w = world('formal')
        w.write(self.root / 'f')
        loaded = w.load(self.root / 'f', engineering=False)
        self.assertEqual(loaded.info['sid_version'], 'SID_BHKM_B256D3_S42')
        with self.assertRaises(ValueError):  # the default content pin is CONTENT_V1, not this synthetic content
            sid_v2.load_bundle(self.root / 'f' / 'run_manifest.json', file_sha256(self.root / 'f' / 'run_manifest.json'),
                               catalog_ids=w.ids, content=w.content)

    def test_bundle_bytes_are_deterministic(self):
        for name in ('bhkm', 'rqvae'):
            w = world(name)
            first = w.write(self.root / (name + '1'))
            second = w.write(self.root / (name + '2'))
            self.assertEqual(first, second)
            for artifact in os.listdir(self.root / (name + '1')):
                self.assertEqual(file_sha256(self.root / (name + '1') / artifact), file_sha256(self.root / (name + '2') / artifact))

    def test_write_requires_a_new_directory_and_publishes_the_manifest_last(self):
        w = world('rqvae')
        w.write(self.root / 'x')
        with self.assertRaises(ValueError):
            w.write(self.root / 'x')
        newest = max(os.stat(self.root / 'x' / n).st_mtime_ns for n in os.listdir(self.root / 'x') if n != 'run_manifest.json')
        self.assertGreaterEqual(os.stat(self.root / 'x' / 'run_manifest.json').st_mtime_ns, newest)
        with mock.patch('os.replace', side_effect=OSError('boom')):
            with self.assertRaises(OSError):
                w.write(self.root / 'y')
        self.assertFalse((self.root / 'y' / 'run_manifest.json').exists())

    def test_write_refuses_inconsistent_inputs_before_creating_anything(self):
        w = world('bhkm')
        bad_audit = copy.deepcopy(w.audit)
        bad_audit['mapping']['items'] += 1
        first, second = same_leaf_pair(w.sids)
        swapped = w.sids.copy()
        swapped[[first, second], 3] = swapped[[second, first], 3]
        cases = {
            'unregistered version': dict(sid_version='SID_FOO'),
            'REPLAY version needs engineering': dict(engineering=False),
            'gates failed': dict(gates={**w.gates, 'passed': False}),
            'audit of another mapping': dict(audit=bad_audit),
            'digest not the audited one': dict(content_feature_sha256='e' * 64),
            'rqvae pieces on a bhkm bundle': dict(weights={'K': 16, 'state_dict': {}}),
            'bad producer': dict(producer={'x': 'nothex'}),
            'bad config sha': dict(fit_config_sha256='zz'),
            'noncanonical suffix': dict(sids=swapped),
        }
        for name, override in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                w.write(self.root / ('bad-' + str(abs(hash(name)))), **override)
        self.assertEqual(os.listdir(self.root), [])


class FailClosed(Base):
    def reject(self, w, directory, sha=None, **kw):
        with self.assertRaises(ValueError):
            w.load(directory, sha, **kw)

    def test_valid_copy_loads(self):
        for name in ('bhkm', 'rqvae'):
            w = world(name)
            w.load(self.copy_of(self.bundle(name), name + '-copy'))

    def test_tampered_artifact_bytes_fail_the_pinned_sha(self):
        for name, artifact in (('bhkm', 'mapping.npz'), ('bhkm', 'audit.json'), ('bhkm', 'tokenizer.json'),
                               ('rqvae', 'weights.npz'), ('rqvae', 'train-log.json')):
            with self.subTest(name=name, artifact=artifact):
                directory = self.copy_of(self.bundle(name), f'{name}-{artifact}')
                data = bytearray((directory / artifact).read_bytes())
                data[len(data) // 2] ^= 1
                (directory / artifact).write_bytes(bytes(data))
                self.reject(world(name), directory)

    def test_tampered_mapping_even_when_resigned(self):
        w = world('bhkm')
        directory = self.copy_of(self.bundle('bhkm'), 'resigned')
        sids = w.sids.copy()
        sids[0, 3] = 5  # a leaf of one item must carry suffix 0
        import io
        buffer = io.BytesIO()
        np.savez(buffer, ids=w.ids, sids=sids)
        sha = tamper(directory, 'mapping.npz', buffer.getvalue())
        self.reject(w, directory, sha)

    def test_garbage_mapping_and_extra_arrays_even_when_resigned(self):
        w = world('bhkm')
        import io
        buffer = io.BytesIO()
        np.savez(buffer, ids=w.ids, sids=w.sids, extra=w.ids)
        for data in (b'not an npz', buffer.getvalue()):
            directory = self.copy_of(self.bundle('bhkm'), f'g{len(data)}')
            self.reject(w, directory, tamper(directory, 'mapping.npz', data))

    def test_wrong_catalog(self):  # A3: only the catalog identity is wrong
        w = world('bhkm')
        directory = self.copy_of(self.bundle('bhkm'), 'cat')
        other = w.ids.copy()
        other[10] += 1
        self.reject(w, directory, catalog_ids=other)
        self.reject(w, directory, catalog_ids=w.ids[:-1])

    def test_wrong_content_digest(self):  # A2: the caller pins another content digest, everything else is right
        w = world('bhkm')
        directory = self.copy_of(self.bundle('bhkm'), 'dig')
        self.reject(w, directory, expected_content_digest='f' * 64)
        self.reject(w, directory, expected_content_digest=sid_v2.CONTENT_V1_FEATURE_SHA256)

    def test_changed_content_array(self):
        w = world('bhkm')
        directory = self.copy_of(self.bundle('bhkm'), 'content')
        content = w.content.copy()
        content[0, 0] += 1.0
        self.reject(w, directory, content=content)
        self.reject(w, directory, content=w.content[:-1])

    def test_missing_extra_artifacts_and_extra_manifest_key(self):
        w = world('bhkm')
        directory = self.copy_of(self.bundle('bhkm'), 'extra-file')
        (directory / 'extra.txt').write_text('x')
        self.reject(w, directory)
        directory = self.copy_of(self.bundle('bhkm'), 'listed-missing')
        sha = rewrite_manifest(directory, lambda m: m['artifacts'].update({'ghost.bin': '0' * 64}))
        self.reject(w, directory, sha)
        directory = self.copy_of(self.bundle('bhkm'), 'unlisted')
        sha = rewrite_manifest(directory, lambda m: m['artifacts'].pop('audit.json'))
        self.reject(w, directory, sha)
        directory = self.copy_of(self.bundle('bhkm'), 'extra-key')
        sha = rewrite_manifest(directory, lambda m: m.update({'note': 'x'}))
        self.reject(w, directory, sha)
        directory = self.copy_of(self.bundle('bhkm'), 'missing-key')
        sha = rewrite_manifest(directory, lambda m: m.pop('producer'))
        self.reject(w, directory, sha)
        directory = self.copy_of(self.bundle('rqvae'), 'rq-lacks-weights')
        sha = rewrite_manifest(directory, lambda m: m['artifacts'].pop('weights.npz'))
        self.reject(world('rqvae'), directory, sha)

    def test_manifest_identity_fields(self):
        w = world('bhkm')
        for field, value in (('schema', 'sidmap-v1'), ('protocol', 'other'), ('task_id', 'G-TOK-02'), ('status', 'failed'),
                             ('kind', 'rqvae'), ('catalog_items', 299), ('suffix_vocabulary', 99), ('vocab_size', 1),
                             ('catalog_sha256', '0' * 64), ('content_feature_sha256', '0' * 64), ('engineering', 'yes')):
            with self.subTest(field=field):
                directory = self.copy_of(self.bundle('bhkm'), 'id-' + field)
                self.reject(w, directory, rewrite_manifest(directory, lambda m: m.update({field: value})))

    def test_failed_gates_bundle_is_rejected(self):  # A1: gates honestly report failure, everything else is consistent
        n = 6000
        sids = balanced_sids(n, 16)
        sids[:, 0] = np.where(np.arange(n) < 5000, 0, 1)  # unbalanced tree: B1 fails
        sids = rq.assign_fourth_code((np.arange(n) * 3 + 5).astype(np.uint32), sids[:, :3].astype(np.int64))
        w = World('SID_V2_REPLAY_BHKM_B16D3', 'bhkm', 16, True, (np.arange(n) * 3 + 5).astype(np.uint32), sids,
                  np.random.RandomState(7).randn(n, 64).astype(np.float32))
        self.assertFalse(w.gates['passed'])
        with self.assertRaises(ValueError):
            w.write(self.root / 'refused')
        w.write(self.root / 'forced', require_passed=False)
        self.reject(w, self.root / 'forced')

    def test_engineering_flag_must_match_both_ways(self):
        w = world('bhkm')
        directory = self.bundle('bhkm')
        self.reject(w, directory, engineering=False)  # engineering bundle into a formal context
        formal = world('formal')
        directory = self.bundle('formal')
        self.reject(formal, directory, engineering=True)  # formal bundle into an engineering context
        directory = self.copy_of(self.bundle('formal'), 'formal-flip')
        self.reject(formal, directory, rewrite_manifest(directory, lambda m: m.update({'engineering': True})), engineering=False)
        directory = self.copy_of(self.bundle('bhkm'), 'replay-flag-off')
        self.reject(w, directory, rewrite_manifest(directory, lambda m: m.update({'engineering': False})), engineering=False)

    def test_unregistered_version_and_branching_mismatch(self):
        w = world('bhkm')
        for version in ('SID_FOO', 'SID_BHKM_B16D3_S42', 'SID_RANDPERM42_OF_BHKM_B16D3', 'SID_BHKM_B256D3_S42'):
            directory = self.copy_of(self.bundle('bhkm'), 'v-' + version)
            with self.subTest(version=version):
                self.reject(w, directory, rewrite_manifest(directory, lambda m: m.update({'sid_version': version})))
        directory = self.copy_of(self.bundle('bhkm'), 'branching')
        self.reject(w, directory, rewrite_manifest(directory, lambda m: m.update({'branching': 256})))

    def test_symlinked_artifact_or_manifest(self):
        w = world('bhkm')
        directory = self.copy_of(self.bundle('bhkm'), 'symlink')
        shutil.copyfile(directory / 'mapping.npz', self.root / 'elsewhere.npz')
        os.symlink(self.root / 'elsewhere.npz', directory / 'mapping.link')
        os.replace(directory / 'mapping.link', directory / 'mapping.npz')  # sha of the target is still right
        self.assertTrue((directory / 'mapping.npz').is_symlink())
        self.reject(w, directory)
        directory = self.bundle('bhkm', 'real')
        os.symlink(directory, self.root / 'dirlink')
        self.reject(w, self.root / 'dirlink')
        os.symlink(directory / 'run_manifest.json', self.root / 'run_manifest.json')
        with self.assertRaises(ValueError):
            sid_v2.load_bundle(self.root / 'run_manifest.json', file_sha256(directory / 'run_manifest.json'),
                               catalog_ids=w.ids, content=w.content, expected_content_digest=w.digest, engineering=True)

    def test_manifest_name_and_sha_pin(self):
        w = world('bhkm')
        directory = self.bundle('bhkm')
        shutil.copyfile(directory / 'run_manifest.json', directory / 'other.json')
        with self.assertRaises(ValueError):
            sid_v2.load_bundle(directory / 'other.json', file_sha256(directory / 'other.json'), catalog_ids=w.ids,
                               content=w.content, expected_content_digest=w.digest, engineering=True)
        self.reject(w, directory, '0' * 64)
        self.reject(w, directory, 'not-a-sha')

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        w = world('bhkm')
        directory = self.copy_of(self.bundle('bhkm'), 'dup')
        text = (directory / 'run_manifest.json').read_text()
        (directory / 'run_manifest.json').write_text(text.replace('{', '{"schema":"sidmap-v2",', 1))
        self.reject(w, directory)
        directory = self.copy_of(self.bundle('bhkm'), 'nan')
        (directory / 'run_manifest.json').write_text(text.replace('"branching":16', '"branching":NaN'))
        self.reject(w, directory)

    def test_audit_json_mismatch(self):  # A5: audit.json lies about a number, signature and gates stay consistent
        w = world('bhkm')
        directory = self.copy_of(self.bundle('bhkm'), 'audit')
        audit = copy.deepcopy(w.audit)
        audit['mapping']['positions'][0]['perplexity'] += 0.5
        self.reject(w, directory, tamper(directory, 'audit.json', canonical_bytes(audit)))
        directory = self.copy_of(self.bundle('bhkm'), 'audit-int')
        audit = copy.deepcopy(w.audit)
        audit['mapping']['leaves']['max'] += 1
        self.reject(w, directory, tamper(directory, 'audit.json', canonical_bytes(audit)))
        directory = self.copy_of(self.bundle('bhkm'), 'gates')
        self.reject(w, directory, rewrite_manifest(directory, lambda m: m['gates']['checks'][0].update({'value': 5})))

    def test_audit_json_float_tolerance_is_relative_1e9(self):
        w = world('bhkm')
        directory = self.copy_of(self.bundle('bhkm'), 'tol')
        audit = copy.deepcopy(w.audit)
        audit['mapping']['positions'][0]['perplexity'] *= 1 + 1e-12
        w.load(directory, tamper(directory, 'audit.json', canonical_bytes(audit)))

    def test_rqvae_audit_rq_section_is_recomputed(self):
        w = world('rqvae')
        directory = self.copy_of(self.bundle('rqvae'), 'rq-audit')
        audit = copy.deepcopy(w.audit)
        audit['rq']['reconstruction']['mse'] *= 0.5  # claims a better fit than the weights give
        self.reject(w, directory, tamper(directory, 'audit.json', canonical_bytes(audit)))

    def test_weights_container_change_with_identical_arrays(self):  # A4: same arrays, different bytes, only the sha differs
        w = world('rqvae')
        directory = self.copy_of(self.bundle('rqvae'), 'wsha')
        state = w.weights
        arrays = {'K': np.array(state['K'], dtype=np.int64), **{k: v.numpy() for k, v in state['state_dict'].items()}}
        path = directory / 'weights.npz'
        before = file_sha256(path)
        np.savez(path, **arrays)  # numpy's own container: same arrays, different bytes (timestamps, compression)
        self.assertNotEqual(file_sha256(path), before)
        self.reject(w, directory)

    def test_weights_that_disagree_with_the_mapping(self):
        w = world('rqvae')
        other = rq.fit(w.content, K=16, seed=7, max_steps=300, batch_size=256, wall_cap_seconds=1e9, probe_from=100, probe_step=200)['state']
        self.assertEqual(other['K'], 16)
        self.assertFalse(np.array_equal(rq.export_codes(other, w.content), rq.export_codes(w.weights, w.content)))
        directory = self.copy_of(self.bundle('rqvae'), 'wrong-weights')
        arrays = {'K': np.array(16, dtype=np.int64), **{k: v.numpy() for k, v in other['state_dict'].items()}}
        import io
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        self.reject(w, directory, tamper(directory, 'weights.npz', buffer.getvalue()))

    def test_malformed_weights(self):
        w = world('rqvae')
        import io
        state = w.weights
        base = {'K': np.array(16, dtype=np.int64), **{k: v.numpy() for k, v in state['state_dict'].items()}}
        variants = {'wrong K': {**base, 'K': np.array(8, dtype=np.int64)},
                    'missing tensor': {k: v for k, v in base.items() if k != 'codebooks.2'},
                    'extra tensor': {**base, 'zzz': np.zeros(1, dtype=np.float32)},
                    'float64 tensor': {**base, 'codebooks.0': base['codebooks.0'].astype(np.float64)},
                    'wrong shape': {**base, 'codebooks.0': base['codebooks.0'][:8]}}
        for name, arrays in variants.items():
            with self.subTest(name):
                directory = self.copy_of(self.bundle('rqvae'), 'mw-' + name.replace(' ', '-'))
                buffer = io.BytesIO()
                np.savez(buffer, **arrays)
                self.reject(w, directory, tamper(directory, 'weights.npz', buffer.getvalue()))

    def test_rqvae_formal_loader_pins_the_preregistered_training_length(self):
        w = world('rqvae')
        directory = self.bundle('rqvae')
        self.reject(w, directory, engineering=False)  # engineering manifest into a formal context (flag mismatch)
        formal = World('SID_RQVAE_B16D3_S42', 'rqvae', 16, False, w.ids, w.sids, w.content, weights=w.weights,
                       train_log=w.train_log, rq_info=w.audit['rq'], required_steps=1000)
        self.assertTrue(formal.gates['passed'], formal.gates)
        formal.write(self.root / 'formal-rq')
        self.reject(formal, self.root / 'formal-rq')  # 1000 steps are not the pre-registered 300,000


class Isolation(Base):
    def test_p5_manifest_is_rejected_by_load_bundle(self):
        w = world('bhkm')
        p5 = ROOT / 'artifacts' / 'p5-sid-manifest.json'
        with self.assertRaises(ValueError):
            sid_v2.load_bundle(p5, file_sha256(p5), catalog_ids=w.ids, content=w.content, expected_content_digest=w.digest)
        directory = self.root / 'p5'
        directory.mkdir()
        shutil.copyfile(p5, directory / 'run_manifest.json')
        with self.assertRaises(ValueError):
            sid_v2.load_bundle(directory / 'run_manifest.json', file_sha256(directory / 'run_manifest.json'),
                               catalog_ids=w.ids, content=w.content, expected_content_digest=w.digest)

    def test_v2_manifest_is_rejected_by_the_frozen_p5_loader(self):
        directory = self.bundle('formal')
        with self.assertRaises((ValueError, KeyError, TypeError)):
            sid_run.load_static(directory / 'run_manifest.json', file_sha256(directory / 'run_manifest.json'))


if __name__ == '__main__':
    unittest.main()
