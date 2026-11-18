import unittest
import numpy as np

from lsvd.two_tower_batching import remaining_user_batches
from lsvd.two_tower import TowerConfig
from lsvd.tests.test_two_tower_round19 import store_fixture


class RemainingUserBatchTests(unittest.TestCase):
    def test_priority_fills_feasible_epoch_without_losing_targets(self):
        chunks = [np.arange(0, 17), np.arange(17, 30), np.arange(30, 40)]
        batches = list(remaining_user_batches(chunks, np.array([0, 1, 2]),
                                              np.array([0, 1, 2]), 4, 2))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(40))
        self.assertEqual(len(batches), 10)
        self.assertTrue(all(len(batch) == 4 for batch in batches))
        owners = np.repeat(np.arange(3), [17, 13, 10])
        for batch in batches:
            self.assertLessEqual(np.bincount(owners[batch]).max(), 2)
        self.assertTrue(np.all(owners[batches[0]] == np.array([0, 0, 1, 1])))

    def test_store_sampler_is_deterministic_and_preserves_weighted_targets(self):
        store = store_fixture()
        store.weights[::3] = 0.25
        config = TowerConfig(packed_window=True, max_user_targets=8,
                             batch_size=128, balance_user_batches=True)
        first = list(store.packed_batches(config, np.random.default_rng(42)))
        second = list(store.packed_batches(config, np.random.default_rng(42)))
        np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(len(store.samples)))
        self.assertEqual(len(first), len(second))
        for batch, repeated in zip(first, second):
            np.testing.assert_array_equal(batch, repeated)
            users = store.user_rows[store.samples[batch]]
            self.assertLessEqual(np.unique(users, return_counts=True)[1].max(), 8)
        self.assertEqual(store.weights[np.concatenate(first)].sum(), store.weights.sum())


if __name__ == '__main__':
    unittest.main()
