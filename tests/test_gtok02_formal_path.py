"""GTOK-02: the six new SID versions are accepted on the real path (engineering=False).

GTOK-01A failed because a new tokenizer was registered for engineering contexts only, so the formal loader refused it on real data.
Two layers are checked here without a harness (the third layer is the server preflight on real inputs):
  L1  registration consistency: every label is in the formal and the engineering table; all 18 committed training configs pass
      ``generative_tok2_run.validate_config(cfg, engineering=False)``; the runner's TASK_ID and the configs' task_id come from one value.
  L2  synthetic formal bundles: fit -> audit -> gates -> write_bundle(engineering=False) -> load_bundle(engineering=False) -> codec ->
      validate_config(False) -> the runner's own ``_load_sid`` on the formal branch. The content pin is the one test knob: the digest of
      the synthetic content, passed as ``expected_content_digest`` (and, for ``_load_sid`` only, as the module constant it compares to).
"""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests.test_sid_v2s import NEW_LABELS, SMALL_N, small_ids, unit_rows
from vkgr import generative_tok_run as tok
from vkgr import generative_tok2_run as tok2
from vkgr import sid_tok2_run as run
from vkgr import sid_v2
from vkgr import sid_v2s as s2
from vkgr import sid_tok_audit as au
from vkgr.data_audit import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'configs' / 'gtok02'
TEMPLATE = ROOT / 'configs' / 'gtok01' / 'g-tok-t3-bhkm-b256-s42.json'
STEMS = {'SID_BHKM_B32D3_S42': 'b32d3', 'SID_BHKM_B64D3_S42': 'b64d3', 'SID_BHKM_B128D3_S42': 'b128d3', 'SID_BHKM_B512D3_S42': 'b512d3',
         'SID_BHKM_B1024D3_S42': 'b1024d3', 'SID_BHKM_B256D2_S42': 'b256d2'}
SEEDS = (42, 2024, 2025)
VARIANT_OF = {label: name for name, label in ((n, v['sid_version']) for n, v in run.expected_config()['variants'].items())}
T3_SHA = 'a7af78a3fc04f0f0d174b983987c34d7efa754edd88a05d1350cdadd06fd47b8'


def committed():
    return {path.name: json.loads(path.read_text()) for path in sorted(CONFIG_DIR.glob('g-tok2-bhkm-*.json'))}


class RegistrationTests(unittest.TestCase):
    def test_every_new_label_is_registered_in_both_tables_with_its_spec(self):
        self.assertEqual(set(NEW_LABELS), set(STEMS))
        for label, spec in NEW_LABELS.items():
            with self.subTest(label):
                self.assertIn(label, s2.SID_VERSIONS)
                self.assertIn(label, s2.ENGINEERING_SID_VERSIONS)
                self.assertEqual(s2._spec(label, False), spec)
                self.assertEqual(s2._spec(label, True), spec)

    def test_no_new_label_is_engineering_only(self):
        self.assertEqual(set(s2.ENGINEERING_SID_VERSIONS) - set(s2.SID_VERSIONS), {'SID_V2S_REPLAY_BHKM_B16D3'})
        self.assertEqual(set(s2.SID_VERSIONS), set(NEW_LABELS))

    def test_old_registry_is_unchanged(self):
        self.assertEqual(sid_v2.SID_VERSIONS, ('SID_RQVAE_B256D3_S42', 'SID_RQVAE_B16D3_S42', 'SID_BHKM_B256D3_S42'))
        self.assertTrue(set(s2.SID_VERSIONS).isdisjoint(sid_v2.ENGINEERING_SID_VERSIONS))
        self.assertTrue(set(sid_v2.SID_VERSIONS).isdisjoint(s2.ENGINEERING_SID_VERSIONS))

    def test_runner_and_configs_share_one_task_id(self):
        self.assertEqual((tok2.TASK_ID, tok2.SCHEMA, s2.TASK_ID, run.TASK_ID), ('G-TOK-02', 'g-tok2-run-v1', 'G-TOK-02', 'G-TOK-02'))
        for name, cfg in committed().items():
            with self.subTest(name):
                self.assertEqual((cfg['task_id'], cfg['schema']), (tok2.TASK_ID, tok2.SCHEMA))


class CommittedConfigTests(unittest.TestCase):
    def test_eighteen_configs_plus_the_fit_config(self):
        self.assertEqual(sorted(p.name for p in CONFIG_DIR.iterdir()),
                         sorted(['g-tok-02.json'] + [f'g-tok2-bhkm-{stem}-s{seed}.json' for stem in STEMS.values() for seed in SEEDS]))
        self.assertEqual(len(committed()), 18)

    def test_every_config_passes_the_formal_validation(self):
        for name, cfg in committed().items():
            with self.subTest(name):
                self.assertIs(tok2.validate_config(cfg, engineering=False), cfg)
                self.assertIs(tok2.validate_config(cfg), cfg)

    def test_every_config_differs_from_the_t3_template_in_four_keys_only(self):
        template = json.loads(TEMPLATE.read_text())
        for name, cfg in committed().items():
            with self.subTest(name):
                self.assertEqual({k for k in set(cfg) | set(template) if cfg.get(k) != template.get(k)} - {'seed'},
                                 {'schema', 'task_id', 'sid'})
                self.assertEqual(set(cfg), set(template))
                self.assertEqual(cfg['schema'], 'g-tok2-run-v1')
                self.assertEqual(cfg['task_id'], 'G-TOK-02')

    def test_file_names_match_their_sid_and_seed_and_cover_the_grid(self):
        seen = set()
        for name, cfg in committed().items():
            match = re.fullmatch(r'g-tok2-bhkm-(b\d+d\d)-s(\d+)\.json', name)
            self.assertIsNotNone(match, name)
            self.assertEqual((STEMS[cfg['sid']], cfg['seed']), (match.group(1), int(match.group(2))), name)
            seen.add((cfg['sid'], cfg['seed']))
        self.assertEqual(seen, {(label, seed) for label in STEMS for seed in SEEDS})

    def test_files_are_canonical_json(self):
        for path in sorted(CONFIG_DIR.glob('*.json')):
            with self.subTest(path.name):
                text = path.read_text()
                self.assertEqual(text, json.dumps(json.loads(text), sort_keys=True, indent=1) + '\n')

    def test_the_gtok01_runners_refuse_the_new_configs(self):
        from vkgr import generative_b1_run as b1
        for name, cfg in committed().items():
            with self.subTest(name), self.assertRaises(ValueError):
                tok.validate_config(cfg)
            with self.assertRaises(ValueError):
                b1.validate_config(cfg)

    def test_the_formal_validation_refuses_everything_else(self):
        cfg = committed()['g-tok2-bhkm-b32d3-s42.json']
        bad_cases = {'T3 label': {'sid': 'SID_BHKM_B256D3_S42'}, 'T1 label': {'sid': 'SID_RQVAE_B256D3_S42'}, 'T2 label': {'sid': 'SID_RQVAE_B16D3_S42'},
                     'replay label': {'sid': 'SID_V2S_REPLAY_BHKM_B16D3'}, 'gtok01 replay label': {'sid': 'SID_V2_REPLAY_BHKM_B16D3'},
                     'gtok01 task': {'task_id': 'G-TOK-01'}, 'gtok01 schema': {'schema': 'g-tok-run-v1'}, 'seed': {'seed': 7},
                     'd_model': {'d_model': 64, 'ff': 256}, 'microbatch': {'microbatch': 64}, 'lr': {'learning_rate': 0.001},
                     'epochs': {'max_epochs': 10}, 'extra key': {'extra': 1}, 'test gate': {'test_gate': 'open'}}
        for name, change in bad_cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                tok2.validate_config({**cfg, **change}, engineering=False)


def formal_world(variant):
    """Fit one variant with the real frozen code on a small synthetic catalog and return what a formal write/load needs."""
    ids, content = small_ids(), unit_rows(SMALL_N)
    digest = au.feature_digest(content)
    config = run.expected_config()
    parent = None
    if variant == 'S256D2':
        from vkgr import sid as bhkm
        model = bhkm.fit(ids, content, branching=256, depth=3, seed=42, max_iter=10)
        parent = {'ids': model.ids, 'sids': model.sids, 'manifest_sha256': T3_SHA}
    result = run.fit_sweep_variant(config, variant, ids, content, expected_content_sha256=digest, engineering=False, parent=parent)
    return ids, content, digest, result


class SyntheticFormalBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name).resolve()

    def test_each_label_goes_through_the_whole_formal_chain(self):
        for variant, label in ((VARIANT_OF[label], label) for label in STEMS):
            with self.subTest(variant):
                ids, content, digest, result = formal_world(variant)
                self.assertTrue(result['ok'], result.get('gates'))
                self.assertIs(result['engineering'], False)
                output = self.root / f'formal-{variant}'
                manifest = run.write_success_bundle(output, result, config_sha='c' * 64)
                self.assertIs(manifest['engineering'], False)
                path = output / 'run_manifest.json'
                loaded = s2.load_bundle(path, file_sha256(path), catalog_ids=ids, content=content, expected_content_digest=digest, engineering=False)
                self.assertEqual((loaded.info['sid_version'], loaded.info['engineering']), (label, False))
                again = sid_v2.build_codec(loaded.codec.catalog.ids, loaded.codec.catalog.sids, loaded.info['branching'], catalog_ids=ids)
                self.assertEqual(again.vocab_size, loaded.codec.vocab_size)
                cfg = committed()[f'g-tok2-bhkm-{STEMS[label]}-s42.json']
                self.assertEqual(cfg['sid'], loaded.info['sid_version'])
                tok2.validate_config(cfg, engineering=False)
                with self.assertRaises(ValueError):  # the same bundle is not accepted as an engineering bundle
                    s2.load_bundle(path, file_sha256(path), catalog_ids=ids, content=content, expected_content_digest=digest, engineering=True)

    def test_the_runners_own_formal_loader_accepts_each_label(self):
        """tok2._load_sid on the engineering=False branch. Its content pin is the module constant; patching that one name to the
        synthetic digest is the test-only knob (the formal path itself is not changed)."""
        for label in STEMS:
            variant = VARIANT_OF[label]
            with self.subTest(variant):
                ids, content, digest, result = formal_world(variant)
                output = self.root / f'runner-{variant}'
                run.write_success_bundle(output, result, config_sha='c' * 64)
                array = np.vstack([np.zeros((1, 64), dtype=np.float32), content])
                content_path = self.root / f'content-{variant}.npy'
                np.save(content_path, array)
                path = output / 'run_manifest.json'
                catalog = tuple(int(x) for x in ids)
                with mock.patch.object(s2, 'CONTENT_V1_FEATURE_SHA256', digest):
                    loaded = tok2._load_sid(path, file_sha256(path), content_path, file_sha256(content_path), catalog, False, digest)
                    cfg = committed()[f'g-tok2-bhkm-{STEMS[label]}-s2025.json']
                    self.assertEqual(loaded.info['sid_version'], cfg['sid'])
                    self.assertIs(loaded.info['engineering'], False)
                    self.assertEqual(loaded.codec.vocab_size, loaded.manifest['vocab_size'])
                with self.assertRaises(ValueError):  # real pin: the synthetic content is not CONTENT_V1
                    tok2._load_sid(path, file_sha256(path), content_path, file_sha256(content_path), catalog, False, digest)

    def test_depth2_vocabulary_has_a_constant_third_position(self):
        _, _, _, result = formal_world('S256D2')
        padded = result['sids']
        self.assertEqual(int(padded[:, 2].max()), 0)
        codec = sid_v2.build_codec(result['ids'], padded, 256)
        self.assertEqual(codec.vocab_size, 2 + 3 * 256 + codec.catalog.suffix_vocabulary)


if __name__ == '__main__':
    unittest.main()
