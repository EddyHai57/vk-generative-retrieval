import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('reader', Path(__file__).with_name('phase2_validation_read.py'))
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


class MethodChecks(unittest.TestCase):
    def test_unverified_run_does_not_publish_selection(self):
        import json
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'runs/p-fixture-s42';(run/'out').mkdir(parents=True)
            (run/'terminal.json').write_text(json.dumps({'status':'succeeded','exit_code':0}))
            (run/'out/manifest.json').write_text(json.dumps({'status':'complete','selection':{'best_metric':[.9,.8]}}))
            with patch.object(reader,'ROOT',root),patch.object(reader,'verifier',return_value={'verified':False}):
                self.assertNotIn('selection',reader.inspect_run('p-fixture-s42'))

    def test_behavior_beta_requires_verified_unique_best_epoch(self):
        beta={'like':.4,'dislike':-.2,'share':.1,'bookmark':0.,'click_on_author':-.05,'open_comments':.03}
        manifest={'selection':{'best_epoch':2},'best':{'epoch':2},'evaluations':[
            {'epoch':2,'diagnostics':{'behavior_beta':beta}},
            {'epoch':3,'diagnostics':{'behavior_beta':{k:99. for k in beta}}}]}
        self.assertEqual(reader.best_behavior_beta(manifest,True),beta)
        with self.assertRaises(ValueError):reader.best_behavior_beta(manifest,False)
        manifest['evaluations'].append(manifest['evaluations'][0])
        with self.assertRaises(ValueError):reader.best_behavior_beta(manifest,True)

    def test_behavior_beta_rejects_nonfinite_or_missing_bit(self):
        beta={'like':.4,'dislike':-.2,'share':.1,'bookmark':0.,'click_on_author':-.05,'open_comments':float('nan')}
        m={'selection':{'best_epoch':1},'best':{'epoch':1},'evaluations':[{'epoch':1,'diagnostics':{'behavior_beta':beta}}]}
        with self.assertRaises(ValueError):reader.best_behavior_beta(m,True)
        del beta['open_comments']
        with self.assertRaises(ValueError):reader.best_behavior_beta(m,True)

    def test_exit_zero_is_not_verified(self):
        self.assertFalse(reader.verified_records([{'status': 'complete', 'exit_code': 0}], 'a' * 64))
        self.assertFalse(reader.verified_records([{'status': 'unverified'}], 'a' * 64))
        self.assertFalse(reader.verified_records([{'status': 'verified'}], 'a' * 64))
        self.assertTrue(reader.verified_records([{'status': 'verified'}], 'a' * 64, job_bound=True))
        self.assertTrue(reader.verified_records([{'status': 'verified', 'manifest_sha256': 'a' * 64}], 'a' * 64))
        self.assertFalse(reader.verified_records([{'status': 'verified', 'manifest_sha256': 'b' * 64}], 'a' * 64))

    def test_oracle_partition_and_structural_oov(self):
        x = reader.overlap([1, 1, 0, 0], [1, 0, 1, 0], [False, False, False, True])
        self.assertEqual((x['all']['both'], x['all']['union'], x['all']['only_p'], x['all']['only_g']), (1, 3, 1, 1))
        self.assertEqual(x['all']['jaccard'], 1 / 3)
        self.assertEqual(x['oov']['union'], 0)
        self.assertEqual(x['non_oov']['oracle_recall'], 1)
        with self.assertRaises(ValueError):
            reader.overlap([0, 1], [0, 0], [False, True])

    def test_exact_mcnemar_known_oracle(self):
        self.assertEqual(reader.mcnemar(0, 0), 1)
        self.assertEqual(reader.mcnemar(0, 3), .25)
        self.assertEqual(reader.mcnemar(3, 3), 1)

    def test_query_identity_is_full_and_ordered(self):
        q = [{'user_id': 9, 'target': 4, 'history': [3], 'history_positions': [[0, 1]], 'target_position': [25, 0], 'oov': False}]
        row = {'ordinal': 0, 'query_sha256': reader.digest(q[0]), 'items': [4]}
        reader.check_g_rows([row], q)
        row['query_sha256'] = reader.digest({**q[0], 'history': [5]})
        with self.assertRaises(ValueError): reader.check_g_rows([row], q)

    def test_rules_keep_adoption_stricter_than_effective(self):
        self.assertEqual(reader.classify([8, 9, 1])['classification'], 'effective')
        self.assertFalse(reader.classify([8, 9, 1])['temperature_adopt_condition'])
        self.assertTrue(reader.classify([6, 6, 6])['temperature_adopt_condition'])
        self.assertEqual(reader.classify([-6, -6, -6])['classification'], 'harmful')
        self.assertEqual(reader.classify([-20, 0, 40], feature=True)['classification'], 'UNCLASSIFIED_PENDING_KIRO')

    def test_serialization_conventions_and_fixed_archive_role(self):
        import hashlib
        self.assertEqual(reader.plain_digest([1, 2]), hashlib.sha256(b'[1,2]').hexdigest())
        self.assertEqual(reader.digest([1, 2]), hashlib.sha256(b'[1,2]\n').hexdigest())
        with self.assertRaises(ValueError):
            reader.archive_role('other.json', 'predictions-epoch-001.json')


if __name__ == '__main__': unittest.main()
