import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from lsvd.regression import compare_users


class RegressionTests(unittest.TestCase):
    def test_exact_counts_hits_and_zero_targets(self):
        users = np.array([2, 9, 11])
        counts = np.array([2, 0, 4])
        hits = np.array([1, 0, 1])
        old_scores = np.array([.5, 0, .25])
        result = compare_users(users, counts, hits, users, counts, hits, old_scores, 200)
        self.assertEqual(result['status'], '一致')
        self.assertEqual(result['new_macro'], .375)
        self.assertEqual(result['target_users'], 2)
        self.assertEqual(result['changed_scores'], 0)

    def test_reordered_users_align_by_id(self):
        result = compare_users(np.array([2, 9]), np.array([2, 4]), np.array([1, 1]),
                               np.array([9, 2]), np.array([4, 2]), np.array([1, 1]),
                               np.array([.25, .5]), 200)
        self.assertEqual(result['status'], '一致')

    def test_same_macro_with_changed_users_is_mismatch(self):
        users = np.array([2, 9])
        counts = np.array([2, 2])
        result = compare_users(users, counts, np.array([1, 0]), users, counts,
                               np.array([0, 1]), np.array([0, .5]), 200)
        self.assertEqual(result['new_macro'], result['old_macro'])
        self.assertEqual(result['status'], '不一致')
        self.assertEqual(result['changed_hits'], 2)

    def test_caps_denominator_and_rejects_user_mismatch(self):
        users = np.array([2])
        result = compare_users(users, np.array([300]), np.array([2]), users,
                               np.array([300]), None, np.array([.01]), 200)
        self.assertEqual(result['new_macro'], .01)
        self.assertEqual(result['status'], '目标数和逐用户分数一致，旧命中数未保存')
        with self.assertRaises(ValueError):
            compare_users(users, np.array([1]), np.array([1]), np.array([3]),
                          np.array([1]), None, np.array([1.]), 200)

    def test_changed_counts_with_same_capped_scores(self):
        users = np.array([2, 9])
        hits = np.array([1, 1])
        result = compare_users(users, np.array([300, 400]), hits, users,
                               np.array([400, 300]), hits, np.array([.005, .005]), 200)
        self.assertEqual(result['status'], '不一致')
        self.assertEqual(result['changed_targets'], 2)

    def test_changed_old_hits_with_same_saved_scores(self):
        users = np.array([2, 9])
        counts = np.array([2, 2])
        result = compare_users(users, counts, np.array([1, 0]), users, counts,
                               np.array([0, 1]), np.array([.5, 0]), 200)
        self.assertEqual(result['status'], '不一致')
        self.assertEqual(result['changed_hits'], 2)

    def test_same_macro_without_saved_hits_checks_individual_scores(self):
        users = np.array([2, 9])
        counts = np.array([2, 2])
        result = compare_users(users, counts, np.array([1, 0]), users, counts,
                               None, np.array([0, .5]), 200)
        self.assertEqual(result['new_macro'], result['old_macro'])
        self.assertEqual(result['status'], '不一致')
        self.assertEqual(result['changed_scores'], 2)

    def test_duplicate_user_ids_are_rejected(self):
        users = np.array([2, 2])
        with self.assertRaises(ValueError):
            compare_users(users, np.array([1, 1]), np.array([1, 1]), users,
                          np.array([1, 1]), None, np.array([1., 1.]), 200)

    def test_content_cli_preserves_group_then_k_axes(self):
        with tempfile.TemporaryDirectory(prefix='lsvd-content-regression-') as directory:
            root = Path(directory)
            current = root / 'current'
            legacy = root / 'legacy'
            current.mkdir()
            legacy.mkdir()
            users = np.array([2, 9])
            counts = np.array([[300, 400], [200, 250], [100, 150]])
            hits = np.arange(18).reshape(3, 3, 2)
            scores = hits / np.minimum(counts[:, None, :], np.array([50, 200, 1000])[None, :, None])
            np.savez(current / 'statistics.npz', user_ids=users, counts=counts,
                     hits=hits, ks=np.array([50, 200, 1000]))
            specs = [dict(slice_id=index, ruler='main', place_scope='all',
                          seen_scope=scope, breakdown='all')
                     for index, scope in enumerate(['all', 'seen', 'unseen'])]
            (current / 'slice_specs.json').write_text(json.dumps(specs))
            np.save(legacy / 'metrics_nprobe64.npy', scores.transpose(2, 0, 1))
            np.save(legacy / 'target_counts_nprobe64.npy', counts.T)
            np.save(root / 'users.npy', users)
            command = [sys.executable, '-B', '-m', 'lsvd.regression',
                       '--current', str(current), '--legacy', str(legacy),
                       '--legacy-users', str(root / 'users.npy'),
                       '--output', str(root / 'comparison.csv')]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            import csv
            with (root / 'comparison.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row['changed_scores'] == '0' for row in rows))
            self.assertTrue(all(row['changed_targets'] == '0' for row in rows))
            invalid = subprocess.run([sys.executable, '-B', '-m', 'lsvd.regression',
                                      '--unknown'], capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()
