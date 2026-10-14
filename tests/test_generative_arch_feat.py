"""GFEAT-02: GFEAT-01 side features (F2 feedback, F3 watch buckets) on the GARCH-01 paper-scale model (TigerArchModel).

server-safe: synthetic codec only (no G1 engineering contract). Run in the frozen-base harness with OMP_NUM_THREADS=1 and
``discover -s tests``. "Bitwise" = torch.equal / digest equality, never allclose.
"""
import hashlib
import json
import unittest
from pathlib import Path

import arch_feat_checks as fc
from test_generative_feat import VENDORED, git_blob

ROOT = Path(__file__).resolve().parents[1]
GFEAT01_COMMIT = 'e4d1118a4fedb34ad137e21b010e7ccec5de4d3e'
A1 = ROOT / 'configs' / 'garch01' / 'g-garch01-a1-lr3e-4-s42.json'
A1_SHA = 'df899c33f9b6a935cf81ab83d9406cc49a6582529e8297560356d4c1eac2dbd7'  # GARCH-01 report 10.2 (the config A1 trained with)
CONFIGS = {'g-gfeat02-none-ctrl-s42.json': 'none', 'g-gfeat02-f2-feedback-s42.json': 'f2_feedback', 'g-gfeat02-f3-watch-buckets-s42.json': 'f3_watch_buckets'}
# sha256 at the GFEAT-02 base (1ffbe27 = GARCH-01 5983cf6 + GFEAT-01 e4d1118): GFEAT-02 adds files only, these stay byte-identical.
COMPOSED = {
    'tiger_arch': '696ef8e183f6ea63e35c8f45b19bd25f79483c0c8d85c3e99a9ba817e34af41b',
    'generative_arch': 'cd661f909ed16b770dba2194f568e44e77d53c506a091b1bb7b3d3988b48ac01',
    'generative_b1_arch': '3d8874bdc19aa334f55666510b5736cd3776aa4862174c20a4441fb5d1c3c0e0',
    'generative_feat': 'dd44adfd8b98981fce477aa72ee517279edc5078c104896818b7a1db5bba0e29',
    'generative_feat_eval': 'c1a2a78e3dff1235298e29485c49301282ce36972c37fa330fb43f11533c98d9',
    'generative_b1_feat': '98a89c0f159b40254c6230f313984748d62ed46a587d84f24397547f18797c40',
}
# GFEAT-01 pins of the two shared definition files (its tests/test_generative_feat.py VENDORED, same bytes as P d70a993)
GFEAT01_SHARED = {'history_side': '8faf06bb4dead2a83311bec1edfc84aaea1959fbbbde88871a1697de29cdea73',
                  'retrieval_features': 'acc60664458a439823daa99edf1b9e44348ab6ca8783dd284a595404856878fa'}
BASE_COMMIT = '1ffbe2722e3b658beb5f18a246137f05604db343'


class ModelTests(unittest.TestCase):
    """server-safe."""

    @classmethod
    def setUpClass(cls):
        cls.codec = fc.codec()

    def test_a_none_is_the_garch01_model(self):
        fc.check_none_is_plain_arch(self.codec)

    def test_base_parameters_keep_the_garch01_random_stream(self):
        fc.check_base_params_equal_plain(self.codec)

    def test_side_tables_follow_base_init_and_keep_the_global_stream(self):
        fc.check_side_tables_follow_base_init(self.codec)

    def test_side_table_shapes_counts_and_pad_rows(self):
        import torch
        base = sum(p.numel() for p in fc.plain(self.codec).parameters()); d = fc.SHAPE['d_model']
        f2, f3 = fc.build(self.codec, 'f2_feedback'), fc.build(self.codec, 'f3_watch_buckets')
        self.assertEqual(sum(p.numel() for p in f2.parameters()) - base, 6 * d)
        self.assertEqual(sum(p.numel() for p in f3.parameters()) - base, (8 + 6) * d)
        for table in (f3.timespent_bucket, f3.ratio_bucket):
            self.assertTrue(torch.equal(table.weight[0], torch.zeros(d)))
        self.assertEqual(f2.init_report['side'], {'feature': 'f2_feedback', 'params_side': 6 * d, 'params_side_trainable': 6 * d, 'params_total': base + 6 * d})
        self.assertEqual(f3.init_report['side'], {'feature': 'f3_watch_buckets', 'params_side': 14 * d, 'params_side_trainable': 12 * d, 'params_total': base + 14 * d})

    def test_injection_token_space_before_sqrt_d_broadcast_to_four_tokens(self):
        for feature in fc.SIDE_TABLES:
            with self.subTest(feature=feature):
                fc.check_injection(self.codec, feature)
        fc.check_injection_before_scale(self.codec)

    def test_side_vector_reads_the_registered_columns(self):
        for feature in fc.SIDE_TABLES:
            with self.subTest(feature=feature):
                fc.check_side_vector_columns(self.codec, feature)

    def test_encoder_consumes_injected_input_decoder_untouched(self):
        for feature in fc.SIDE_TABLES:
            with self.subTest(feature=feature):
                fc.check_encode_uses_injected_input(self.codec, feature)
                fc.check_decoder_untouched(self.codec, feature)

    def test_b_zero_side_equals_none_eval_and_training_steps(self):
        for feature in fc.SIDE_TABLES:
            with self.subTest(feature=feature):
                fc.check_zero_side_equals_none_eval(self.codec, feature)
                fc.check_zero_side_equals_none_train(self.codec, feature)

    def test_side_required_and_bad_side_rejected(self):
        import torch
        for feature in fc.SIDE_TABLES:
            model = fc.build(self.codec, feature); source, side = fc.batch(self.codec)
            for name, bad in (('None', None), ('float', side.float()), ('wrong items', side[:, :-1])):
                with self.subTest(feature=feature, case=name), self.assertRaises(ValueError):
                    model.encode(source, bad)
            pad = side.clone(); pad[1, 2] = torch.tensor([1, 5, 9])
            with self.assertRaises(ValueError):
                model.encode(source, pad)


class DefinitionTests(unittest.TestCase):
    """server-safe. (c) feature definitions = GFEAT-01's files byte for byte; bucket edges are the train-defined card constants."""

    def test_c_shared_definitions_sha_equal_gfeat01(self):
        self.assertEqual({k: VENDORED[k] for k in GFEAT01_SHARED}, GFEAT01_SHARED)  # GFEAT-01's own pins unchanged
        fc.check_shared_sources(ROOT, GFEAT01_SHARED)
        for name in fc.SHARED_FILES:
            with self.subTest(file=name):
                blob = git_blob(GFEAT01_COMMIT, f'src/vkgr/{name}.py')
                if blob is not None:  # the GFEAT-01 commit itself, when this checkout has it
                    self.assertEqual(hashlib.sha256(blob).hexdigest(), VENDORED[name])

    def test_bucket_edges_are_the_frozen_train_constants(self):
        fc.check_bucket_definitions()

    def test_injection_code_is_gfeat01_code(self):
        from vkgr import generative_arch_feat as gaf
        from vkgr import generative_feat as gf
        for name in ('add_side_tables', '_valid', 'side_vector', 'embed_source'):
            self.assertIs(getattr(gaf.TigerArchModelSide, name), getattr(gf.TigerModelSide, name), name)
        self.assertIs(gaf.gf, gf)


class RegistryTests(unittest.TestCase):
    def test_registry(self):
        fc.check_registry()

    def test_committed_configs_are_a1_plus_the_feature_switch(self):
        from vkgr import generative_arch_feat as gaf
        self.assertEqual(hashlib.sha256(A1.read_bytes()).hexdigest(), A1_SHA)
        a1 = json.loads(A1.read_text())
        paths = sorted((ROOT / 'configs' / 'gfeat02').glob('*.json'))
        self.assertEqual([p.name for p in paths], sorted(CONFIGS))
        for path in paths:
            with self.subTest(config=path.name):
                value = json.loads(path.read_text())
                gaf.validate_config(value)
                self.assertEqual(value['feature'], CONFIGS[path.name])
                self.assertEqual({k for k in set(value) | set(a1) if value.get(k) != a1.get(k)}, {'schema', 'task_id', 'feature', 'side_version'})
                self.assertEqual((value['schema'], value['task_id'], value['side_version'], value['seed']), ('g-arch-feat-run-v1', 'GFEAT-02', 'HIST_V2_SIDE_V1', 42))
                self.assertEqual(path.read_text(), json.dumps(value, indent=1, sort_keys=True) + '\n')


class ComposedFilesUntouchedTests(unittest.TestCase):
    def test_garch01_and_gfeat01_files_byte_identical_to_base(self):
        """GFEAT-02 only adds files: the six composed modules equal the GFEAT-02 base commit."""
        for name, expected in COMPOSED.items():
            with self.subTest(file=name):
                data = (ROOT / 'src' / 'vkgr' / f'{name}.py').read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
                blob = git_blob(BASE_COMMIT, f'src/vkgr/{name}.py')
                if blob is not None:
                    self.assertEqual(data, blob)

if __name__ == '__main__':
    unittest.main()
