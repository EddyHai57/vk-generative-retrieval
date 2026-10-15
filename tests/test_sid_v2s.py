"""GTOK-02: SIDMAP_V2S contract (six resolution-sweep tokenizers): registry, depth-2 padding, bundle round trip, fail-closed loader."""
import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_sid_v2 import World as OldWorld, balanced_sids
from vkgr import sid as bhkm
from vkgr import sid_tok_audit as au
from vkgr import sid_v2
from vkgr import sid_v2s as s2
from vkgr.data_audit import canonical_bytes, file_sha256

NEW_LABELS = {
    'SID_BHKM_B32D3_S42': ('bhkm', 32, 3), 'SID_BHKM_B64D3_S42': ('bhkm', 64, 3), 'SID_BHKM_B128D3_S42': ('bhkm', 128, 3),
    'SID_BHKM_B512D3_S42': ('bhkm', 512, 3), 'SID_BHKM_B1024D3_S42': ('bhkm', 1024, 3), 'SID_BHKM_B256D2_S42': ('bhkm', 256, 2)}
REPLAY = 'SID_V2S_REPLAY_BHKM_B16D3'

# Ten items, hand-computed. The parent (c0, c1) groups are (0,0): ids 3,5,9,31; (0,1): ids 8,21; (1,0): ids 12,20,30; (2,2): id 40.
IDS = np.array([3, 5, 8, 9, 12, 20, 21, 30, 31, 40], dtype=np.uint32)
PARENT = np.array([[0, 0, 7, 0], [0, 0, 1, 0], [0, 1, 2, 0], [0, 0, 5, 0], [1, 0, 3, 0],
                   [1, 0, 3, 1], [0, 1, 9, 1], [1, 0, 4, 2], [0, 0, 6, 1], [2, 2, 0, 0]], dtype=np.uint32)
RANK = np.array([0, 1, 0, 2, 0, 1, 1, 2, 3, 0], dtype=np.uint32)


class RegistryTests(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual((s2.SCHEMA, s2.PROTOCOL, s2.TASK_ID), ('sidmap-v2s', 'generative-like-sid-v2s', 'G-TOK-02'))
        self.assertEqual(s2.CONTENT_V1_FEATURE_SHA256, sid_v2.CONTENT_V1_FEATURE_SHA256)

    def test_six_labels_with_expected_spec_in_both_contexts(self):
        self.assertEqual(set(s2.SID_VERSIONS), set(NEW_LABELS))
        self.assertEqual(len(s2.SID_VERSIONS), 6)
        for label, spec in NEW_LABELS.items():
            for engineering in (False, True):
                with self.subTest(label=label, engineering=engineering):
                    self.assertEqual(s2._spec(label, engineering), spec)

    def test_no_label_is_registered_only_for_engineering(self):
        self.assertEqual(set(s2.ENGINEERING_SID_VERSIONS) - set(s2.SID_VERSIONS), {REPLAY})
        self.assertTrue(set(s2.SID_VERSIONS) <= set(s2.ENGINEERING_SID_VERSIONS))
        self.assertEqual(s2.VERSION_SPEC[REPLAY], ('bhkm', 16, 3))
        self.assertEqual(set(s2.VERSION_SPEC), set(s2.ENGINEERING_SID_VERSIONS))

    def test_replay_label_is_engineering_only(self):
        self.assertEqual(s2._spec(REPLAY, True), ('bhkm', 16, 3))
        with self.assertRaises(ValueError):
            s2._spec(REPLAY, False)

    def test_unregistered_and_old_labels_are_rejected_in_both_contexts(self):
        old = list(sid_v2.ENGINEERING_SID_VERSIONS) + ['SID_BHKM_B16D3_S42', 'nope', '', None, 7]
        for label in old:
            for engineering in (False, True):
                with self.subTest(label=label, engineering=engineering), self.assertRaisesRegex(ValueError, 'unregistered sid_version'):
                    s2._spec(label, engineering)

    def test_engineering_flag_must_be_a_bool(self):
        for flag in (None, 0, 1, 'x'):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                s2._spec('SID_BHKM_B32D3_S42', flag)


class DepthTwoTests(unittest.TestCase):
    def test_pad_depth2_puts_zero_in_third_and_rank_in_fourth(self):
        raw3 = np.array([[1, 2, 0], [1, 2, 1], [3, 0, 0]], dtype=np.uint32)
        padded = s2.pad_depth2(raw3)
        self.assertEqual((padded.dtype, padded.shape), (np.dtype('uint32'), (3, 4)))
        self.assertEqual(padded.tolist(), [[1, 2, 0, 0], [1, 2, 0, 1], [3, 0, 0, 0]])

    def test_unpad_depth2_is_the_inverse(self):
        raw3 = np.array([[1, 2, 0], [1, 2, 1], [3, 0, 0]], dtype=np.uint32)
        self.assertEqual(s2.unpad_depth2(s2.pad_depth2(raw3)).tolist(), raw3.tolist())
        self.assertEqual(s2.unpad_depth2(s2.pad_depth2(raw3)).dtype, np.dtype('uint32'))

    def test_derive_keeps_the_parent_partition_and_numbers_by_item_id(self):
        sids4, raw3 = s2.derive_depth2_from_parent(IDS, PARENT)
        self.assertEqual((sids4.dtype, sids4.shape, raw3.dtype, raw3.shape), (np.dtype('uint32'), (10, 4), np.dtype('uint32'), (10, 3)))
        self.assertTrue(np.array_equal(sids4[:, :2], PARENT[:, :2]))
        self.assertTrue(np.array_equal(raw3[:, :2], PARENT[:, :2]))
        self.assertEqual(sids4[:, 2].tolist(), [0] * 10)
        self.assertEqual(sids4[:, 3].tolist(), RANK.tolist())
        self.assertEqual(raw3[:, 2].tolist(), RANK.tolist())
        self.assertTrue(np.array_equal(sids4, s2.pad_depth2(raw3)))

    def test_derive_leaf_statistics_match_hand_computation(self):
        sids4, _ = s2.derive_depth2_from_parent(IDS, PARENT)
        _, counts = np.unique(sids4[:, :3], axis=0, return_counts=True)
        self.assertEqual((len(counts), int(counts.max()), len(IDS) - len(counts)), (4, 4, 6))

    def test_derive_rejects_unsorted_or_duplicate_ids(self):
        for ids in (IDS[::-1].copy(), np.r_[IDS[:3], IDS[2:-1]].astype(np.uint32)):
            with self.subTest(ids=ids.tolist()), self.assertRaises(ValueError):
                s2.derive_depth2_from_parent(ids, PARENT)

    def test_derive_rejects_bad_shapes_and_dtypes(self):
        with self.assertRaises(ValueError):
            s2.derive_depth2_from_parent(IDS, PARENT[:-1])
        with self.assertRaises(ValueError):
            s2.derive_depth2_from_parent(IDS, PARENT[:, :3])
        with self.assertRaises(ValueError):
            s2.derive_depth2_from_parent(IDS, PARENT.astype(np.int64))

    def test_derive_does_not_modify_its_inputs(self):
        ids, parent = IDS.copy(), PARENT.copy()
        s2.derive_depth2_from_parent(ids, parent)
        self.assertTrue(np.array_equal(ids, IDS) and np.array_equal(parent, PARENT))


# ---------------------------------------------------------------------------------------------------------------------
# bundle layer (item 4)
# ---------------------------------------------------------------------------------------------------------------------
PRODUCER = {'src/vkgr/sid_v2s.py': 'a' * 64, 'src/vkgr/sid_v2.py': 'b' * 64}
CONFIG_SHA = 'c' * 64
PARENT_SHA = 'd' * 64
SMALL_N = 1500
LABELS_D3 = [label for label, spec in NEW_LABELS.items() if spec[2] == 3]
D2_LABEL = 'SID_BHKM_B256D2_S42'


def unit_rows(n, seed=7):
    x = np.random.RandomState(seed).randn(n, 64)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class SweepWorld:
    """Everything needed to write one SIDMAP_V2S bundle, kept in memory. Formal path (engineering=False) by default."""

    def __init__(self, label, ids, sids, content, engineering=False):
        self.label, self.ids, self.sids, self.content, self.engineering = label, ids, sids, content, engineering
        _, self.branching, self.depth = s2.VERSION_SPEC[label]
        self.digest = au.feature_digest(content)
        self.parent_sha = PARENT_SHA if self.depth == 2 else None
        self.tokenizer = {'fit': {'branching': self.branching, 'depth': self.depth}}
        if self.depth == 2:
            self.tokenizer['parent_manifest_sha256'] = self.parent_sha
        self.audit = {'mapping': s2.sweep_audit(ids, sids, content, branching=self.branching, depth=self.depth),
                      'sweep': s2.sweep_section(label, engineering, self.parent_sha)}
        self.gates = au.gates(self.audit['mapping'], 'bhkm', None, expected_content_sha256=self.digest)

    def write(self, output, **override):
        args = dict(sid_version=self.label, engineering=self.engineering, ids=self.ids, sids=self.sids,
                    content_feature_sha256=self.digest, tokenizer=self.tokenizer, audit=self.audit, gates=self.gates,
                    fit_config_sha256=CONFIG_SHA, producer=PRODUCER)
        args.update(override)
        return s2.write_bundle(output, **args)

    def load(self, directory, sha=None, **override):
        directory = Path(directory)
        args = dict(catalog_ids=self.ids, content=self.content, expected_content_digest=self.digest, engineering=self.engineering)
        args.update(override)
        return s2.load_bundle(directory / 'run_manifest.json', sha or file_sha256(directory / 'run_manifest.json'), **args)


_WORLDS = {}


def small_ids(n=SMALL_N):
    return (np.arange(n) * 3 + 5).astype(np.uint32)


def sweep_world(label):
    """Real frozen sid.fit on a small synthetic catalog (depth 2 derives from a real 256-ary depth-3 parent)."""
    if label not in _WORLDS:
        ids, content = small_ids(), unit_rows(SMALL_N)
        _, branching, depth = s2.VERSION_SPEC[label]
        if depth == 3:
            sids = bhkm.fit(ids, content, branching=branching, depth=3, seed=42, max_iter=10).sids
        else:
            parent = bhkm.fit(ids, content, branching=256, depth=3, seed=42, max_iter=10)
            sids, _ = s2.derive_depth2_from_parent(ids, parent.sids)
        _WORLDS[label] = SweepWorld(label, ids, sids, content)
    return _WORLDS[label]


def collision_mapping():
    """Smallest B1-clean 256-ary depth-2 mapping with leaf collisions: 256 first-level nodes of 257 items, one
    second-level node of two items per first-level node. Returns (ids, parent sids, content)."""
    n = 256 * 257
    ids = (np.arange(n) * 2 + 5).astype(np.uint32)
    j = np.arange(n) % 257
    parent = np.zeros((n, 4), dtype=np.uint32)
    parent[:, 0] = np.arange(n) // 257
    parent[:, 1] = np.where(j < 2, 0, j - 1)
    parent[:, 2] = np.where(j == 1, 1, 0)
    return ids, parent, unit_rows(n, 3)


def collision_world():
    if 'collision' not in _WORLDS:
        ids, parent, content = collision_mapping()
        sids, _ = s2.derive_depth2_from_parent(ids, parent)
        _WORLDS['collision'] = SweepWorld(D2_LABEL, ids, sids, content)
    return _WORLDS['collision']


def rewrite_manifest(directory, mutate):
    path = Path(directory) / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_bytes(canonical_bytes(manifest))
    return file_sha256(path)


def tamper(directory, name, data):
    """Replace one artifact and re-sign the manifest, so that only the artifact content is wrong."""
    directory = Path(directory)
    (directory / name).write_bytes(data)
    return rewrite_manifest(directory, lambda m: m['artifacts'].__setitem__(name, file_sha256(directory / name)))


def tamper_json(directory, name, mutate):
    value = json.loads((Path(directory) / name).read_text())
    mutate(value)
    return tamper(directory, name, canonical_bytes(value))


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.count = 0

    def bundle(self, label):
        self.count += 1
        target = self.root / f'bundle-{self.count}'
        sweep_world(label).write(target)
        return target


class SweepAuditTests(unittest.TestCase):
    def test_sweep_section_per_depth(self):
        self.assertEqual(s2.sweep_section('SID_BHKM_B32D3_S42', False, None),
                         {'depth': 3, 'pad_digit_zero': False, 'balanced_basis': 'full', 'parent_manifest_sha256': None})
        self.assertEqual(s2.sweep_section(D2_LABEL, False, PARENT_SHA),
                         {'depth': 2, 'pad_digit_zero': True, 'balanced_basis': 'raw3', 'parent_manifest_sha256': PARENT_SHA})

    def test_depth2_balance_is_judged_before_padding(self):
        ids, parent, content = collision_mapping()
        sids4, raw3 = s2.derive_depth2_from_parent(ids, parent)
        self.assertEqual(au.balanced_children(raw3, 256), {'balanced': True, 'nodes': 65793, 'violations': 0, 'max_spread': 1})
        padded = au.balanced_children(sids4, 256)
        self.assertEqual((padded['balanced'], padded['violations']), (False, 256))
        audit = collision_world().audit['mapping']
        self.assertEqual(audit['balanced_children'], au.balanced_children(raw3, 256))
        self.assertTrue(collision_world().gates['passed'], collision_world().gates)
        self.assertEqual((audit['leaves']['count'], audit['leaves']['max'], audit['items_needing_suffix'], audit['vocab_size']),
                         (65536, 2, 256, 772))

    def test_depth3_balance_is_unchanged_full_mapping(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        self.assertEqual(w.audit['mapping']['balanced_children'], au.balanced_children(w.sids, 32))
        self.assertEqual(w.audit['mapping'], au.audit_mapping(w.ids, w.sids, w.content, branching=32, kind='bhkm'))


class RoundTripTests(Base):
    def test_every_label_round_trips_on_the_formal_path(self):
        for label in NEW_LABELS:
            with self.subTest(label):
                w = sweep_world(label)
                self.assertTrue(w.gates['passed'], w.gates)
                manifest = w.write(self.root / label)
                self.assertEqual(set(manifest), set(s2.MANIFEST_KEYS))
                self.assertEqual(s2.MANIFEST_KEYS, sid_v2.MANIFEST_KEYS | {'depth'})
                self.assertEqual(sorted(os.listdir(self.root / label)), ['audit.json', 'mapping.npz', 'run_manifest.json', 'tokenizer.json'])
                loaded = w.load(self.root / label, engineering=False)
                self.assertIsInstance(loaded, sid_v2.LoadedSID)
                self.assertEqual((manifest['schema'], manifest['protocol'], manifest['task_id'], manifest['depth']),
                                 ('sidmap-v2s', 'generative-like-sid-v2s', 'G-TOK-02', w.depth))
                self.assertIs(manifest['engineering'], False)
                self.assertEqual((loaded.info['sid_version'], loaded.info['depth'], loaded.info['engineering']), (label, w.depth, False))
                self.assertEqual(loaded.info['branching'], w.branching)
                self.assertTrue(np.array_equal(loaded.codec.catalog.sids, w.sids))
                loaded.codec.check_identity()
                self.assertEqual(loaded.codec.vocab_size, manifest['vocab_size'])
                self.assertEqual(manifest['vocab_size'], 2 + 3 * w.branching + manifest['suffix_vocabulary'])

    def test_depth3_third_digit_may_be_all_zero(self):
        for label in ('SID_BHKM_B512D3_S42', 'SID_BHKM_B1024D3_S42'):
            with self.subTest(label):
                w = sweep_world(label)
                self.assertEqual(int(w.sids[:, 2].max()), 0)  # the tree has run out of items: the third digit is the within-cluster number 0
                self.bundle(label)  # writes and passes

    def test_depth2_padded_mapping_passes_the_frozen_codec_check(self):
        w = sweep_world(D2_LABEL)
        self.assertEqual(int(w.sids[:, 2].max()), 0)
        codec = s2.sid_v2.build_codec(w.ids, w.sids, 256)
        self.assertEqual(codec.catalog.bases[:3], [256, 256, 256])

    def test_collision_bundle_round_trip(self):
        w = collision_world()
        w.write(self.root / 'c')
        loaded = w.load(self.root / 'c')
        self.assertEqual((loaded.info['vocab_size'], loaded.info['suffix_vocabulary']), (772, 2))
        self.assertEqual(loaded.manifest['depth'], 2)

    def test_bundle_bytes_are_deterministic(self):
        w = sweep_world('SID_BHKM_B64D3_S42')
        first, second = w.write(self.root / 'a'), w.write(self.root / 'b')
        self.assertEqual(first, second)
        for name in os.listdir(self.root / 'a'):
            self.assertEqual(file_sha256(self.root / 'a' / name), file_sha256(self.root / 'b' / name))

    def test_write_needs_a_new_directory_and_publishes_the_manifest_last(self):
        from unittest import mock
        w = sweep_world('SID_BHKM_B32D3_S42')
        w.write(self.root / 'x')
        with self.assertRaises(ValueError):
            w.write(self.root / 'x')
        with mock.patch('os.replace', side_effect=OSError('boom')), self.assertRaises(OSError):
            w.write(self.root / 'y')
        self.assertFalse((self.root / 'y' / 'run_manifest.json').exists())


class WriteRefusalTests(Base):
    def test_inconsistent_inputs_are_refused_before_creating_anything(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        d2 = sweep_world(D2_LABEL)
        bad_audit = copy.deepcopy(w.audit)
        bad_audit['mapping']['items'] += 1
        bad_sweep = copy.deepcopy(w.audit)
        bad_sweep['sweep']['depth'] = 2
        extra = copy.deepcopy(w.audit)
        extra['rq'] = None
        padded_nonzero = d2.sids.copy()
        padded_nonzero[:, 2] = 1
        wrong_parent = copy.deepcopy(d2.tokenizer)
        wrong_parent['parent_manifest_sha256'] = 'e' * 64
        cases = {
            'unregistered version': (w, dict(sid_version='SID_FOO')),
            'replay label needs engineering': (w, dict(sid_version=REPLAY)),
            'old label': (w, dict(sid_version='SID_BHKM_B256D3_S42')),
            'gates failed': (w, dict(gates={**w.gates, 'passed': False})),
            'audit of another mapping': (w, dict(audit=bad_audit)),
            'sweep depth wrong': (w, dict(audit=bad_sweep)),
            'rq section present': (w, dict(audit=extra)),
            'digest is not the audited one': (w, dict(content_feature_sha256='e' * 64)),
            'bad producer': (w, dict(producer={'x': 'nothex'})),
            'bad config sha': (w, dict(fit_config_sha256='zz')),
            'depth-2 pad column non-zero': (d2, dict(sids=padded_nonzero)),
            'depth-2 parent sha differs from tokenizer': (d2, dict(tokenizer=wrong_parent)),
            'depth-3 tokenizer carries a parent': (w, dict(tokenizer={**w.tokenizer, 'parent_manifest_sha256': PARENT_SHA})),
        }
        for name, (world, override) in cases.items():
            target = self.root / name.replace(' ', '-')
            with self.subTest(name), self.assertRaises(ValueError):
                world.write(target, **override)
            self.assertFalse(target.exists(), name)


class FailClosedTests(Base):
    def test_wrong_pins_and_inputs(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        directory = self.bundle(w.label)
        with self.assertRaises(ValueError):
            w.load(directory, sha='0' * 64)
        with self.assertRaises(ValueError):
            w.load(directory, catalog_ids=w.ids + 1)
        with self.assertRaises(ValueError):
            w.load(directory, expected_content_digest='f' * 64)
        with self.assertRaises(ValueError):
            w.load(directory, content=unit_rows(SMALL_N, seed=8))
        with self.assertRaises(ValueError):  # formal default pin is CONTENT_V1, not this synthetic content
            s2.load_bundle(directory / 'run_manifest.json', file_sha256(directory / 'run_manifest.json'), catalog_ids=w.ids, content=w.content)

    def test_one_byte_of_any_artifact_is_detected(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        for name in ('mapping.npz', 'tokenizer.json', 'audit.json'):
            directory = self.bundle(w.label)
            data = (directory / name).read_bytes()
            (directory / name).write_bytes(data + b'0')
            with self.subTest(name), self.assertRaises(ValueError):
                w.load(directory)

    def test_artifact_set_must_be_exact(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        extra = self.bundle(w.label)
        (extra / 'weights.npz').write_bytes(b'x')
        missing = self.bundle(w.label)
        (missing / 'audit.json').unlink()
        for directory in (extra, missing):
            with self.assertRaises(ValueError):
                w.load(directory)

    def test_symlinked_artifact_is_refused(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        directory = self.bundle(w.label)
        real = self.root / 'real-audit.json'
        shutil.copy(directory / 'audit.json', real)
        (directory / 'audit.json').unlink()
        (directory / 'audit.json').symlink_to(real)
        with self.assertRaises(ValueError):
            w.load(directory)

    def test_manifest_key_set_and_values(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        mutations = {
            'extra key': lambda m: m.__setitem__('extra', 1),
            'missing depth': lambda m: m.pop('depth'),
            'missing gates': lambda m: m.pop('gates'),
            'gates not passed': lambda m: m['gates'].__setitem__('passed', False),
            'gates differ from recomputation': lambda m: m['gates']['checks'].pop(),
            'schema': lambda m: m.__setitem__('schema', 'sidmap-v2'),
            'protocol': lambda m: m.__setitem__('protocol', 'generative-like-sid-v2'),
            'task id': lambda m: m.__setitem__('task_id', 'G-TOK-01'),
            'status': lambda m: m.__setitem__('status', 'running'),
            'unregistered label': lambda m: m.__setitem__('sid_version', 'SID_FOO'),
            'other registered label': lambda m: m.__setitem__('sid_version', 'SID_BHKM_B64D3_S42'),
            'depth differs from registry': lambda m: m.__setitem__('depth', 2),
            'depth is a string': lambda m: m.__setitem__('depth', '3'),
            'engineering flag flipped': lambda m: m.__setitem__('engineering', True),
            'branching': lambda m: m.__setitem__('branching', 64),
            'vocab size': lambda m: m.__setitem__('vocab_size', m['vocab_size'] + 1),
            'catalog sha': lambda m: m.__setitem__('catalog_sha256', '0' * 64),
        }
        for name, mutate in mutations.items():
            directory = self.bundle(w.label)
            sha = rewrite_manifest(directory, mutate)
            with self.subTest(name), self.assertRaises(ValueError):
                w.load(directory, sha=sha)

    def test_engineering_flag_must_match_the_context(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        directory = self.bundle(w.label)
        with self.assertRaises(ValueError):
            w.load(directory, engineering=True)
        replay = SweepWorld(REPLAY, small_ids(), bhkm.fit(small_ids(), unit_rows(SMALL_N), branching=16, depth=3, seed=42, max_iter=10).sids,
                            unit_rows(SMALL_N), engineering=True)
        replay.write(self.root / 'replay')
        self.assertEqual(replay.load(self.root / 'replay', engineering=True).info['sid_version'], REPLAY)
        with self.assertRaises(ValueError):
            replay.load(self.root / 'replay', engineering=False)

    def test_noncanonical_suffix_and_digit_range(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        bad_suffix, high_digit = w.sids.copy(), w.sids.copy()
        bad_suffix[0, 3] = 1
        high_digit[0, 0] = 32
        for name, sids in (('suffix', bad_suffix), ('digit', high_digit)):
            directory = self.bundle(w.label)
            data = self.root / f'{name}.npz'
            s2.sid_v2._write_npz(data, {'ids': w.ids, 'sids': sids})
            sha = tamper(directory, 'mapping.npz', data.read_bytes())
            with self.subTest(name), self.assertRaises(ValueError):
                w.load(directory, sha=sha)

    def test_audit_json_is_recomputed(self):
        w = sweep_world('SID_BHKM_B32D3_S42')
        mutations = {
            'mapping.items': lambda a: a['mapping'].__setitem__('items', a['mapping']['items'] + 1),
            'mapping.leaves.max': lambda a: a['mapping']['leaves'].__setitem__('max', 99),
            'mapping.balanced': lambda a: a['mapping']['balanced_children'].__setitem__('violations', 1),
            'extra top-level key': lambda a: a.__setitem__('rq', None),
            'missing sweep': lambda a: a.pop('sweep'),
            'sweep.depth': lambda a: a['sweep'].__setitem__('depth', 2),
            'sweep.pad_digit_zero': lambda a: a['sweep'].__setitem__('pad_digit_zero', True),
            'sweep.balanced_basis': lambda a: a['sweep'].__setitem__('balanced_basis', 'raw3'),
            'sweep.parent': lambda a: a['sweep'].__setitem__('parent_manifest_sha256', PARENT_SHA),
            'sweep extra key': lambda a: a['sweep'].__setitem__('note', 1),
        }
        for name, mutate in mutations.items():
            directory = self.bundle(w.label)
            sha = tamper_json(directory, 'audit.json', mutate)
            with self.subTest(name), self.assertRaises(ValueError):
                w.load(directory, sha=sha)

    def test_depth2_specific_rejections(self):
        w = sweep_world(D2_LABEL)
        nonzero = w.sids.copy()
        nonzero[:, 2] = 1  # still a canonical, in-range mapping: only the padding rule can reject it
        directory = self.bundle(D2_LABEL)
        data = self.root / 'nonzero.npz'
        s2.sid_v2._write_npz(data, {'ids': w.ids, 'sids': nonzero})
        sha = tamper(directory, 'mapping.npz', data.read_bytes())
        with self.assertRaisesRegex(ValueError, 'pad'):
            w.load(directory, sha=sha)
        mutations = {
            'sweep parent missing': lambda a: a['sweep'].__setitem__('parent_manifest_sha256', None),
            'sweep parent not hex': lambda a: a['sweep'].__setitem__('parent_manifest_sha256', 'Z' * 64),
            'sweep parent differs from tokenizer': lambda a: a['sweep'].__setitem__('parent_manifest_sha256', 'e' * 64),
            'sweep depth 3': lambda a: a['sweep'].__setitem__('depth', 3),
            'sweep not padded': lambda a: a['sweep'].__setitem__('pad_digit_zero', False),
            'basis full': lambda a: a['sweep'].__setitem__('balanced_basis', 'full'),
        }
        for name, mutate in mutations.items():
            directory = self.bundle(D2_LABEL)
            sha = tamper_json(directory, 'audit.json', mutate)
            with self.subTest(name), self.assertRaises(ValueError):
                w.load(directory, sha=sha)
        directory = self.bundle(D2_LABEL)
        sha = tamper_json(directory, 'tokenizer.json', lambda t: t['config'].__setitem__('parent_manifest_sha256', 'e' * 64))
        with self.assertRaises(ValueError):
            w.load(directory, sha=sha)

    def test_depth2_balance_is_recomputed_on_the_unpadded_columns(self):
        w = collision_world()
        directory = self.root / 'c'
        w.write(directory)
        loaded_gates = json.loads((directory / 'run_manifest.json').read_text())['gates']
        self.assertTrue(loaded_gates['passed'])
        # an audit computed on the padded four columns says "not balanced": the loader must not accept that story
        wrong = au.audit_mapping(w.ids, w.sids, w.content, branching=256, kind='bhkm')
        self.assertFalse(wrong['balanced_children']['balanced'])
        sha = tamper_json(directory, 'audit.json', lambda a: a.__setitem__('mapping', json.loads(canonical_bytes(wrong))))
        with self.assertRaises(ValueError):
            w.load(directory, sha=sha)


class IsolationTests(Base):
    def test_old_loader_rejects_every_new_bundle(self):
        for label in NEW_LABELS:
            directory = self.bundle(label)
            w = sweep_world(label)
            for engineering in (False, True):
                with self.subTest(label=label, engineering=engineering), self.assertRaises(ValueError):
                    sid_v2.load_bundle(directory / 'run_manifest.json', file_sha256(directory / 'run_manifest.json'), catalog_ids=w.ids,
                                       content=w.content, expected_content_digest=w.digest, engineering=engineering)

    def test_new_loader_rejects_an_old_t3_shaped_bundle(self):
        world = OldWorld('SID_BHKM_B256D3_S42', 'bhkm', 256, False, small_ids(), balanced_sids(SMALL_N, 256), unit_rows(SMALL_N))
        world.write(self.root / 'old')
        self.assertEqual(world.load(self.root / 'old').info['sid_version'], 'SID_BHKM_B256D3_S42')  # the old loader accepts it
        for engineering in (False, True):
            with self.subTest(engineering=engineering), self.assertRaises(ValueError):
                s2.load_bundle(self.root / 'old' / 'run_manifest.json', file_sha256(self.root / 'old' / 'run_manifest.json'),
                               catalog_ids=world.ids, content=world.content, expected_content_digest=world.digest, engineering=engineering)

    def test_old_registry_is_untouched_by_importing_the_new_module(self):
        self.assertEqual(sid_v2.SID_VERSIONS, ('SID_RQVAE_B256D3_S42', 'SID_RQVAE_B16D3_S42', 'SID_BHKM_B256D3_S42'))
        self.assertFalse(set(s2.SID_VERSIONS) & set(sid_v2.ENGINEERING_SID_VERSIONS))


if __name__ == '__main__':
    unittest.main()
