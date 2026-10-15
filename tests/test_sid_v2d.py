"""G6-PREP step 6a: SIDMAP_V2D bundle contract (depth-aware sibling of SIDMAP_V2): round trips and fail-closed negatives."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tests import g6_fixtures as fx
from vkgr import sid_depth as sd
from vkgr import sid_rqvae as rq
from vkgr import sid_rqvae_d as rd
from vkgr import sid_tok_audit as au
from vkgr import sid_tok_audit_d as ad
from vkgr import sid_v2
from vkgr import sid_v2d
from vkgr.data_audit import canonical_bytes, file_sha256

PRODUCER = {'src/vkgr/sid_v2d.py': 'a' * 64, 'src/vkgr/sid_tok_audit_d.py': 'b' * 64}
CONFIG_SHA = 'c' * 64
TOKENIZER = {'variant': 'synthetic'}
V2D_KEYS = set(sid_v2.MANIFEST_KEYS) | {'depth'}


class World:
    """Everything needed to write one SIDMAP_V2D bundle (in memory; bundles are written into temporary directories)."""

    def __init__(self, version, engineering, ids, sids, content, weights=None, train_log=None, rq_info=None, required_steps=au.REQUIRED_STEPS):
        self.version, self.engineering = version, engineering
        self.kind, self.branching, self.depth = sid_v2d.VERSION_SPEC[version]
        self.ids, self.sids, self.content, self.weights, self.train_log = ids, sids, content, weights, train_log
        self.digest = au.feature_digest(content)
        mapping = ad.audit_mapping_d(ids, sids, content, branching=self.branching, kind=self.kind, depth=self.depth)
        self.audit = {'mapping': mapping, 'rq': rq_info}
        prereg = sid_v2d.prereg_for(version, len(ids), engineering)
        self.gates = ad.gates_d(mapping, self.kind, rq_info, depth=self.depth, expected_content_sha256=self.digest,
                                required_steps=required_steps, prereg=prereg)

    def write(self, output, **override):
        args = dict(sid_version=self.version, engineering=self.engineering, ids=self.ids, sids=self.sids, content_feature_sha256=self.digest,
                    tokenizer=TOKENIZER, audit=self.audit, gates=self.gates, fit_config_sha256=CONFIG_SHA, producer=PRODUCER,
                    weights=self.weights, train_log=self.train_log)
        args.update(override)
        return sid_v2d.write_bundle_d(output, **args)

    def load(self, directory, sha=None, **override):
        directory = Path(directory)
        args = dict(catalog_ids=self.ids, content=self.content, expected_content_digest=self.digest, engineering=self.engineering)
        args.update(override)
        return sid_v2d.load_bundle_d(directory / 'run_manifest.json', sha or file_sha256(directory / 'run_manifest.json'), **args)


def bhkm_world(version, n=3000):
    kind, branching, depth = sid_v2d.VERSION_SPEC[version]
    ids = (np.arange(n) * 3 + 5).astype(np.uint32)
    sids = sd.assign_suffix(ids, fx.balanced_codes(n, branching, depth))
    content = np.random.RandomState(7).randn(n, 64).astype(np.float32)
    return World(version, True, ids, sids, content)


def rqvae_world(seed=42):
    """Toy 4-level RQ-VAE: 256 unit-norm blobs of 10 points. The pre-registered lr 0.4 collapses on toy data, so the fixture trains with
    lr 0.1 and refreshes the four codebooks with k-means on all rows; the mapping is exported from exactly the weights that are written."""
    rng = np.random.RandomState(5)
    centers = rng.randn(256, 64)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    content = centers[np.repeat(np.arange(256), 10)] + rng.randn(2560, 64) * 0.05
    content = (content / np.linalg.norm(content, axis=1, keepdims=True)).astype(np.float32)
    steps = 1000
    with mock.patch.dict(rq.PREREG, lr=0.1):
        out = rd.fit_d(content, K=256, levels=4, seed=seed, max_steps=steps, batch_size=256, wall_cap_seconds=1e9, probe_from=100, probe_step=200)
    assert out['status'] == 'complete', out['status']
    model = rd.RQVAED(256, 4)
    model.load_state_dict(out['state']['state_dict'])
    rq.init_codebooks(model, torch.from_numpy(content), seed, 10)
    state = {'K': 256, 'state_dict': {k: v.detach().clone() for k, v in model.state_dict().items()}}
    ids = (np.arange(2560) * 2 + 3).astype(np.uint32)
    sids = sd.assign_suffix(ids, rd.export_codes_d(state, content))
    info = {'status': 'complete', 'steps': steps, 'max_steps': steps, 'elapsed': float(out['elapsed']), 'wall_cap_seconds': 7200,
            'required_steps': steps, 'reconstruction': rd.reconstruction_report_d(state, content)}
    log = {'loss_log': out['loss_log'], 'usage_log': out['usage_log'], 'init_digest': out['init_digest'], 'prefix_digest': out['prefix_digest']}
    return World('SID_RQVAE_B256D4_S42', True, ids, sids, content, weights=state, train_log=log, rq_info=info, required_steps=steps)


_CACHE = {}


def world(name):
    if name not in _CACHE:
        _CACHE[name] = {'d4': lambda: bhkm_world('SID_BHKM_B16D4_S42'), 'd3': lambda: bhkm_world('SID_V2D_REPLAY_BHKM_B16D3'),
                        'rqvae': rqvae_world}[name]()
    return _CACHE[name]


def rewrite_manifest(directory, mutate):
    path = Path(directory) / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_bytes(canonical_bytes(manifest))
    return file_sha256(path)


def replace_artifact(directory, name, data):
    """Swap one artifact and re-sign the manifest, so that only the content of that artifact is wrong."""
    directory = Path(directory)
    (directory / name).write_bytes(data)
    return rewrite_manifest(directory, lambda m: m['artifacts'].__setitem__(name, file_sha256(directory / name)))


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def bundle(self, name='d4', **override):
        w = world(name)
        w.write(self.root / 'b', **override)
        return w, self.root / 'b'


class RoundTrip(Fixture):
    def test_depth_four_bhkm(self):
        w, d = self.bundle('d4')
        manifest = json.loads((d / 'run_manifest.json').read_text())
        self.assertEqual(set(manifest), V2D_KEYS)
        self.assertEqual((manifest['schema'], manifest['protocol'], manifest['task_id'], manifest['depth']),
                         ('sidmap-v2d', 'generative-like-sid-v2d', 'G-HP-01-G6', 4))
        loaded = w.load(d)
        self.assertIsInstance(loaded.codec, sd.ValidatedCodecD)
        self.assertEqual(loaded.codec.depth, 4)
        self.assertEqual(loaded.info['depth'], 4)
        self.assertEqual(loaded.info['vocab_size'], manifest['vocab_size'])
        self.assertEqual(loaded.codec.vocab_size, 2 + 4 * 16 + manifest['suffix_vocabulary'])
        self.assertTrue(np.array_equal(loaded.codec.catalog.sids, w.sids))
        self.assertIn('B2', [c['id'] for c in manifest['gates']['checks']])

    def test_depth_three_replay_matches_the_depth_three_bundle_contract(self):
        w, d = self.bundle('d3')
        loaded = w.load(d)
        mapping = au.audit_mapping(w.ids, w.sids, w.content, branching=16, kind='bhkm')
        gates = au.gates(mapping, 'bhkm', None, expected_content_sha256=w.digest)
        sid_v2.write_bundle(self.root / 'v2', sid_version='SID_V2_REPLAY_BHKM_B16D3', engineering=True, ids=w.ids, sids=w.sids,
                            content_feature_sha256=w.digest, tokenizer=TOKENIZER, audit={'mapping': mapping, 'rq': None}, gates=gates,
                            fit_config_sha256=CONFIG_SHA, producer=PRODUCER)
        old = sid_v2.load_bundle(self.root / 'v2' / 'run_manifest.json', file_sha256(self.root / 'v2' / 'run_manifest.json'),
                                 catalog_ids=w.ids, content=w.content, expected_content_digest=w.digest, engineering=True)
        for name in ('ids', 'sids', 'keys', 'key_items'):
            self.assertTrue(np.array_equal(getattr(loaded.codec.catalog, name), getattr(old.codec.catalog, name)), name)
        self.assertEqual((loaded.codec.offsets, loaded.codec.vocab_size), (old.codec.offsets, old.codec.vocab_size))
        self.assertEqual(loaded.info['vocab_size'], old.info['vocab_size'])

    def test_rqvae_depth_four(self):
        w, d = self.bundle('rqvae')
        loaded = w.load(d)
        self.assertEqual((loaded.info['kind'], loaded.info['depth'], loaded.info['branching']), ('rqvae', 4, 256))
        self.assertEqual(sorted(p.name for p in d.iterdir()), sorted(['run_manifest.json', 'mapping.npz', 'tokenizer.json', 'weights.npz', 'audit.json', 'train-log.json']))

    def test_bundles_are_deterministic(self):
        w = world('d4')
        a = w.write(self.root / 'a')
        b = w.write(self.root / 'b')
        self.assertEqual(canonical_bytes(a), canonical_bytes(b))
        self.assertEqual(file_sha256(self.root / 'a' / 'mapping.npz'), file_sha256(self.root / 'b' / 'mapping.npz'))

    def test_output_must_be_a_new_directory(self):
        w = world('d4')
        w.write(self.root / 'x')
        self.assertRaises(ValueError, w.write, self.root / 'x')

    def test_failed_gates_are_not_published(self):
        w = world('d4')
        bad = dict(w.gates, passed=False)
        self.assertRaises(ValueError, w.write, self.root / 'x', gates=bad)


class Versions(unittest.TestCase):
    def test_registry(self):
        self.assertEqual(sid_v2d.SID_VERSIONS, ('SID_BHKM_B16D4_S42', 'SID_RQVAE_B256D4_S42'))
        self.assertEqual(sid_v2d.ENGINEERING_SID_VERSIONS, sid_v2d.SID_VERSIONS + ('SID_V2D_REPLAY_BHKM_B16D3',))
        self.assertEqual(sid_v2d.VERSION_SPEC, {'SID_BHKM_B16D4_S42': ('bhkm', 16, 4), 'SID_RQVAE_B256D4_S42': ('rqvae', 256, 4),
                                                'SID_V2D_REPLAY_BHKM_B16D3': ('bhkm', 16, 3)})

    def test_replay_label_is_engineering_only_and_unknown_labels_are_refused(self):
        self.assertEqual(sid_v2d._spec_d('SID_BHKM_B16D4_S42', False), ('bhkm', 16, 4))
        self.assertRaises(ValueError, sid_v2d._spec_d, 'SID_V2D_REPLAY_BHKM_B16D3', False)
        self.assertEqual(sid_v2d._spec_d('SID_V2D_REPLAY_BHKM_B16D3', True), ('bhkm', 16, 3))
        for label in ('SID_BHKM_B16D3_S42', 'SID_RQVAE_B256D3_S42', 'SID_RQVAE_B16D4_S42', '', None):
            for engineering in (True, False):
                with self.subTest(label=label, engineering=engineering):
                    self.assertRaises(ValueError, sid_v2d._spec_d, label, engineering)
        self.assertRaises(ValueError, sid_v2d._spec_d, 'SID_BHKM_B16D4_S42', 1)

    def test_prereg_for(self):
        self.assertEqual(sid_v2d.prereg_for('SID_BHKM_B16D4_S42', 120415, False), ad.PREREG_BHKM_B16D4)
        self.assertEqual(sid_v2d.prereg_for('SID_BHKM_B16D4_S42', 3000, True), ad.prereg_block(3000, 16, 4))
        self.assertEqual(sid_v2d.prereg_for('SID_V2D_REPLAY_BHKM_B16D3', 3000, True), ad.prereg_block(3000, 16, 3))
        self.assertIsNone(sid_v2d.prereg_for('SID_RQVAE_B256D4_S42', 3000, True))
        # a formal bundle of any other size still gets the frozen block, so it cannot pass B2
        self.assertEqual(sid_v2d.prereg_for('SID_BHKM_B16D4_S42', 3000, False), ad.PREREG_BHKM_B16D4)


class FailClosed(Fixture):
    def reload(self, w, d, sha=None, **override):
        return w.load(d, sha, **override)

    def test_wrong_manifest_sha(self):
        w, d = self.bundle()
        self.assertRaises(ValueError, self.reload, w, d, '0' * 64)

    def test_engineering_flag_mismatch(self):
        w, d = self.bundle()
        self.assertRaises(ValueError, self.reload, w, d, None, engineering=False)

    def test_depth_key_is_required_and_must_match_the_version(self):
        w, d = self.bundle()
        sha = rewrite_manifest(d, lambda m: m.pop('depth'))
        self.assertRaisesRegex(ValueError, 'key set', self.reload, w, d, sha)

    def test_depth_value_mismatch(self):
        w, d = self.bundle()
        sha = rewrite_manifest(d, lambda m: m.__setitem__('depth', 3))
        self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_extra_manifest_key(self):
        w, d = self.bundle()
        sha = rewrite_manifest(d, lambda m: m.__setitem__('extra', 1))
        self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_schema_and_protocol_and_task(self):
        w = world('d4')
        for index, (key, value) in enumerate((('schema', 'sidmap-v2'), ('protocol', 'generative-like-sid-v2'), ('task_id', 'G-TOK-01'), ('status', 'running'))):
            with self.subTest(key=key):
                d = self.root / f'case{index}'
                w.write(d)
                sha = rewrite_manifest(d, lambda m: m.__setitem__(key, value))
                self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_unregistered_sid_version(self):
        w, d = self.bundle()
        sha = rewrite_manifest(d, lambda m: m.__setitem__('sid_version', 'SID_BHKM_B16D3_S42'))
        self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_the_depth_three_contract_and_this_one_refuse_each_other(self):
        w, d = self.bundle('d3')
        # a SIDMAP_V2 loader refuses a V2D bundle
        self.assertRaises(ValueError, sid_v2.load_bundle, d / 'run_manifest.json', file_sha256(d / 'run_manifest.json'),
                          catalog_ids=w.ids, content=w.content, expected_content_digest=w.digest, engineering=True)
        # a V2D loader refuses a SIDMAP_V2 bundle
        mapping = au.audit_mapping(w.ids, w.sids, w.content, branching=16, kind='bhkm')
        gates = au.gates(mapping, 'bhkm', None, expected_content_sha256=w.digest)
        out = self.root / 'v2'
        sid_v2.write_bundle(out, sid_version='SID_V2_REPLAY_BHKM_B16D3', engineering=True, ids=w.ids, sids=w.sids, content_feature_sha256=w.digest,
                            tokenizer=TOKENIZER, audit={'mapping': mapping, 'rq': None}, gates=gates, fit_config_sha256=CONFIG_SHA, producer=PRODUCER)
        self.assertRaises(ValueError, sid_v2d.load_bundle_d, out / 'run_manifest.json', file_sha256(out / 'run_manifest.json'),
                          catalog_ids=w.ids, content=w.content, expected_content_digest=w.digest, engineering=True)

    def test_manifest_that_claims_failed_gates(self):
        w, d = self.bundle()
        sha = rewrite_manifest(d, lambda m: m['gates'].__setitem__('passed', False))
        self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_manifest_gates_must_equal_the_recomputed_gates(self):
        w, d = self.bundle()
        sha = rewrite_manifest(d, lambda m: m['gates']['checks'].pop())  # drop B2
        self.assertRaisesRegex(ValueError, 'gates', self.reload, w, d, sha)

    def test_wrong_content_digest_and_wrong_catalog(self):
        w, d = self.bundle()
        self.assertRaisesRegex(ValueError, 'content', self.reload, w, d, None, expected_content_digest='0' * 64)
        other = w.content.copy()
        other[0, 0] += 1
        self.assertRaises(ValueError, self.reload, w, d, None, content=other)
        self.assertRaisesRegex(ValueError, 'catalog', self.reload, w, d, None, catalog_ids=w.ids[:-1])
        self.assertRaisesRegex(ValueError, 'catalog', self.reload, w, d, None, catalog_ids=w.ids + 1)

    def test_artifact_set_and_bytes(self):
        w, d = self.bundle()
        (d / 'stray.txt').write_text('x')
        self.assertRaisesRegex(ValueError, 'artifact', self.reload, w, d)
        (d / 'stray.txt').unlink()
        data = (d / 'mapping.npz').read_bytes()
        (d / 'mapping.npz').write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        self.assertRaises(ValueError, self.reload, w, d)

    def test_noncanonical_mapping_with_a_valid_signature_is_refused(self):
        w, d = self.bundle()
        sids = w.sids.copy()
        sids[0, 4] = 1  # a leaf with a single item must carry suffix 0
        sid_v2._write_npz(self.root / 'tamper.npz', {'ids': w.ids, 'sids': sids})
        sha = replace_artifact(d, 'mapping.npz', (self.root / 'tamper.npz').read_bytes())
        self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_depth_three_mapping_in_a_depth_four_bundle_is_refused(self):
        w, d = self.bundle()
        three = world('d3')
        sid_v2._write_npz(self.root / 'three.npz', {'ids': three.ids[:len(w.ids)], 'sids': three.sids[:len(w.ids)]})
        sha = replace_artifact(d, 'mapping.npz', (self.root / 'three.npz').read_bytes())
        self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_audit_json_must_equal_the_recomputation(self):
        w, d = self.bundle()
        audit = json.loads((d / 'audit.json').read_text())
        audit['mapping']['leaves']['count'] += 1
        sha = replace_artifact(d, 'audit.json', canonical_bytes(audit))
        self.assertRaisesRegex(ValueError, 'audit', self.reload, w, d, sha)

    def test_tokenizer_json_must_describe_the_manifest(self):
        w, d = self.bundle()
        tokenizer = json.loads((d / 'tokenizer.json').read_text())
        tokenizer['depth'] = 3
        sha = replace_artifact(d, 'tokenizer.json', canonical_bytes(tokenizer))
        self.assertRaisesRegex(ValueError, 'tokenizer', self.reload, w, d, sha)

    def test_rqvae_weights_must_reproduce_the_mapping(self):
        w, d = self.bundle('rqvae')
        other = rd.fit_d(w.content, K=256, levels=4, seed=1, max_steps=20, batch_size=256, wall_cap_seconds=1e9, probe_from=10, probe_step=20,
                         check_every=10, log_every=10, eval_every=10)['state']
        arrays = sid_v2d._weights_arrays_d(other, 256, 4)
        sid_v2._write_npz(self.root / 'w.npz', arrays)
        sha = replace_artifact(d, 'weights.npz', (self.root / 'w.npz').read_bytes())
        self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_rqvae_weights_with_three_codebooks_are_refused(self):
        w, d = self.bundle('rqvae')
        arrays = sid_v2d._weights_arrays_d(w.weights, 256, 4)
        arrays = {k: v for k, v in arrays.items() if k != 'codebooks.3'}
        sid_v2._write_npz(self.root / 'w.npz', arrays)
        sha = replace_artifact(d, 'weights.npz', (self.root / 'w.npz').read_bytes())
        self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_rqvae_weights_with_an_extra_tensor_are_refused(self):
        w, d = self.bundle('rqvae')
        arrays = sid_v2d._weights_arrays_d(w.weights, 256, 4)
        arrays['codebooks.4'] = arrays['codebooks.3'].copy()  # a fifth codebook that nothing reads
        sid_v2._write_npz(self.root / 'w.npz', arrays)
        sha = replace_artifact(d, 'weights.npz', (self.root / 'w.npz').read_bytes())
        self.assertRaisesRegex(ValueError, 'tensor names', self.reload, w, d, sha)

    def test_rqvae_audit_rq_section_is_recomputed(self):
        w, d = self.bundle('rqvae')
        audit = json.loads((d / 'audit.json').read_text())
        audit['rq']['reconstruction']['mse'] *= 2
        sha = replace_artifact(d, 'audit.json', canonical_bytes(audit))
        self.assertRaises(ValueError, self.reload, w, d, sha)

    def test_a_bundle_whose_own_gates_honestly_failed_is_refused(self):
        # unbalanced random codes: B1/B2 fail in the manifest AND in the recomputation, so only the passed-flag check can refuse it
        n = 400
        ids = (np.arange(n) * 3 + 5).astype(np.uint32)
        codes = np.random.RandomState(3).randint(0, 16, size=(n, 4))
        w = World('SID_BHKM_B16D4_S42', True, ids, sd.assign_suffix(ids, codes), np.random.RandomState(7).randn(n, 64).astype(np.float32))
        self.assertFalse(w.gates['passed'])
        w.write(self.root / 'honest', require_passed=False)
        self.assertRaisesRegex(ValueError, 'gates did not pass', w.load, self.root / 'honest')

    def test_weights_that_disagree_with_the_mapping_are_refused_even_when_the_audit_matches_them(self):
        a = world('rqvae')
        b = rqvae_world(seed=43)
        self.assertTrue(b.gates['passed'])
        self.assertFalse(all(torch.equal(a.weights['state_dict'][k], b.weights['state_dict'][k]) for k in a.weights['state_dict']))
        mixed = World('SID_RQVAE_B256D4_S42', True, a.ids, a.sids, a.content, weights=b.weights, train_log=b.train_log,
                      rq_info=b.audit['rq'], required_steps=1000)
        self.assertTrue(mixed.gates['passed'], mixed.gates)
        mixed.write(self.root / 'mixed')
        self.assertRaisesRegex(ValueError, 'weights do not reproduce', mixed.load, self.root / 'mixed')

    def test_writer_rejects_inconsistent_inputs(self):
        w = world('d4')
        for override in ({'weights': w.weights or {'K': 16}}, {'train_log': {}}, {'sids': w.sids[:, :4]}, {'fit_config_sha256': 'x'},
                         {'content_feature_sha256': '1' * 64}, {'producer': {}}, {'tokenizer': []}):
            with self.subTest(override=list(override)):
                self.assertRaises(ValueError, w.write, self.root / f'bad{len(list(self.root.iterdir()))}', **override)


if __name__ == '__main__':
    unittest.main()
