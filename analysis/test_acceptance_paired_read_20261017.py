"""Synthetic checks; no server paths, checkpoints, or real predictions read."""
import importlib.util
import pathlib
import tempfile
import types
import unittest

SOURCE = pathlib.Path(__file__).with_name('acceptance_paired_read_20261017.py')


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SOURCE.exists(), 'missing acceptance reader')
        spec = importlib.util.spec_from_file_location('acceptance_reader', SOURCE)
        self.r = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.r)

    def test_exact_pool_is_binomial_of_summed_discordants(self):
        pairs = {str(s): {'only_new': 1, 'only_control': 0, 'delta_hits': 1,
                         'mcnemar_exact_two_sided': 1.0} for s in (42, 2024, 2025)}
        value = self.r.pool_summary(pairs)
        self.assertEqual(value['only_new'], 3)
        self.assertEqual(value['only_control'], 0)
        self.assertEqual(value['p_pool'], 0.25)
        self.assertFalse(value['independent_observations'])

    def test_pool_rejects_missing_seed_and_inconsistent_counts(self):
        pairs = {str(s): {'only_new': 1, 'only_control': 0, 'delta_hits': 1,
                         'mcnemar_exact_two_sided': 1.0} for s in (42, 2024, 2025)}
        pairs['42']['delta_hits'] = 2
        with self.assertRaises(ValueError):
            self.r.pool_summary(pairs)
        with self.assertRaises(ValueError):
            self.r.pool_summary({})

    def test_registered_three_comparisons_include_m1_and_no_e(self):
        self.assertEqual(self.r.COMPARISONS, (
            ('combo_vs_f4_registered', 'p-feat04combo-cde', 'p-php01-h1-t010'),
            ('combo_vs_m1_descriptive', 'p-feat04combo-cde', 'p-seqm1-mean'),
            ('m2_vs_m1_registered', 'p-seqm2-k4', 'p-seqm1-mean')))

    def test_pinned_loader_rejects_modified_source_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            p = pathlib.Path(folder) / 'source.py'
            p.write_text('raise RuntimeError("must not execute")')
            with self.assertRaisesRegex(ValueError, 'SHA'):
                self.r.pinned_module(p, '0' * 64, {})

    def test_manifest_identity_requires_real_closed_validation_and_seed(self):
        m = {'config': {'seed': 42}, 'binding': {'synthetic': False,
             'test_gate': 'closed_hash_only'}, 'selection': {'best_metric': [0.1, 0.02]}}
        self.r.check_manifest('p-seqm1-mean-s42', m)
        for changed in ({**m, 'config': {'seed': 2024}},
                        {**m, 'binding': {'synthetic': True, 'test_gate': 'closed_hash_only'}},
                        {**m, 'selection': {'best_metric': [float('nan'), 0.02]}}):
            with self.assertRaises(ValueError):
                self.r.check_manifest('p-seqm1-mean-s42', changed)

    def test_exact_known_values_and_invalid_counts(self):
        self.assertEqual(self.r.mcnemar(0, 0), 1.0)
        self.assertEqual(self.r.mcnemar(3, 0), .25)
        self.assertAlmostEqual(self.r.mcnemar(118, 73), .0013902601073185607)
        for a, b in ((-1, 2), (True, 2), (1.5, 2)):
            with self.assertRaises(ValueError):
                self.r.mcnemar(a, b)

    def test_archive_rejects_foreign_catalog_and_support(self):
        # Synthetic adapter models the already-pinned upstream archive boundary;
        # this tests additional consumer-side catalog/support/slice checks.
        metric = lambda count: {'count': count, 'recall@50': 0.0, 'ndcg@50': 0.0}
        manifest = {'config': {'seed': 42}, 'binding': {'synthetic': False,
                    'test_gate': 'closed_hash_only', 'support_sha256': 'support'},
                    'selection': {'best_metric': [0.0, 0.0], 'best_epoch': 1},
                    'evaluations': [{'epoch': 1, 'predictions_ref': 'predictions-epoch-001.json',
                    'predictions_sha256': 'predictions', 'result': {'metrics': {
                    'all': metric(2520), 'oov': metric(623), 'train_observed': metric(1897)}}}]}
        result = {'hits': [0] * 2520, 'ndcg': [0.0] * 2520}
        upstream = types.SimpleNamespace(ROOT=pathlib.Path('/synthetic'),
                   load_archive=lambda *args: result,
                   read=lambda path, pin: (manifest if path.name == 'manifest.json' else [[1]], pin))
        state = {'manifest_sha256': 'manifest'}
        self.r.accepted_archive(upstream, 'p-seqm1-mean-s42', state, [], {}, {1}, 'support')
        with self.assertRaisesRegex(ValueError, 'support'):
            self.r.accepted_archive(upstream, 'p-seqm1-mean-s42', state, [], {}, {1}, 'other')
        with self.assertRaisesRegex(ValueError, 'catalog'):
            self.r.accepted_archive(upstream, 'p-seqm1-mean-s42', state, [], {}, {2}, 'support')
        manifest['evaluations'][0]['result']['metrics']['oov']['recall@50'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'slice'):
            self.r.accepted_archive(upstream, 'p-seqm1-mean-s42', state, [], {}, {1}, 'support')


if __name__ == '__main__':
    unittest.main()

class ArchivePinTest(unittest.TestCase):
 def test_other_config_with_valid_own_verify_is_rejected(self):
  from analysis.acceptance_paired_read_20261017 import check_manifest_pin, MANIFEST_PINS
  rid='p-seqm2-k4-s42'
  check_manifest_pin(rid,MANIFEST_PINS[rid])
  with self.assertRaises(ValueError):check_manifest_pin(rid,'f'*64)
  with self.assertRaises(ValueError):check_manifest_pin('p-seqm2-k4-s99','f'*64)

class RuntimePathsTest(unittest.TestCase):
 def test_deployment_paths_are_private_parameters(self):
  from analysis import acceptance_paired_read_20261017 as r
  with self.assertRaises(ValueError):r.configure_paths({'phase2':'relative','combo_ops':'/workspace/public-fixture'})
  r.configure_paths({'phase2':'/workspace/fixture/reader.py','combo_ops':'/workspace/fixture-ops'})
  self.assertEqual(str(r.PHASE2),'/workspace/fixture/reader.py')
