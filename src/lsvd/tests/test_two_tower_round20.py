import unittest

import numpy as np
import torch

from lsvd.two_tower import TowerConfig, UserTower
from lsvd.two_tower_data import EventStore
from lsvd.tests.test_two_tower_round19 import store_fixture


class BatchDiagnosisTests(unittest.TestCase):
    def test_cap_one_tail_keeps_round19_shuffle_order(self):
        store = store_fixture()
        config = TowerConfig(packed_window=True, max_user_targets=1, batch_size=128)
        actual = list(store.packed_batches(config, np.random.default_rng(42)))
        # With both users fitting, each round takes the next occurrence of
        # each user, preserving its position in the original shuffled queue.
        order = np.random.default_rng(42).permutation(len(store.samples))
        occurrences = np.zeros(len(store.user_ids), dtype=np.int64)
        expected = []
        for index in order:
            user = store.user_rows[store.samples[index]]
            round_index = int(occurrences[user])
            if round_index == len(expected):
                expected.append([])
            expected[round_index].append(index)
            occurrences[user] += 1
        self.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)

    def test_each_cap_keeps_natural_targets_once_and_limits_users(self):
        store = store_fixture()
        for cap in (64, 8, 1):
            with self.subTest(cap=cap):
                config = TowerConfig(packed_window=True, max_user_targets=cap, batch_size=128)
                batches = list(store.packed_batches(config, np.random.default_rng(42)))
                np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(len(store.samples)))
                for batch in batches:
                    users = store.user_rows[store.samples[batch]]
                    self.assertLessEqual(np.unique(users, return_counts=True)[1].max(), cap)
                weights = np.concatenate([store.weights[batch] for batch in batches])
                self.assertEqual(float(weights.sum()), float(store.weights.sum()))

    def test_shared_window_mean_support_is399_and_strictly_past(self):
        torch.manual_seed(42)
        config = TowerConfig(packed_window=True, causal_readout='window_mean', dropout=0.0)
        columns = dict(store_fixture().columns)
        columns['week'] = np.full(460, 2, dtype=np.uint8)
        store = EventStore(columns, np.array([7]), np.array([0, 460]))
        batch, _, _, _, _ = store.train_batch(np.array([400, 410]), config, 'cpu')
        self.assertEqual(int(batch['source_rows'][0]), 1)
        model = UserTower(config, 'causal').eval()
        with torch.no_grad():
            model.gate.fill_(1)
        items = torch.randn(len(batch['items']), 64)
        before = model(items, batch, batch['valid'])
        changed = items.clone()
        changed[:int(batch['readouts'][1]) - 399, 0] += 100
        torch.testing.assert_close(before[1], model(changed, batch, batch['valid'])[1], atol=2e-6, rtol=2e-6)
        changed = items.clone()
        changed[int(batch['readouts'][0]):, 0] += 100
        torch.testing.assert_close(before[0], model(changed, batch, batch['valid'])[0], atol=2e-6, rtol=2e-6)
        changed = items.clone()
        changed[:100, 0] += 100
        self.assertGreater(float((before[0] - model(changed, batch, batch['valid'])[0]).abs().max()), 1e-5)

    def test_one_history_event_has_same_two_readouts(self):
        torch.manual_seed(42)
        config = TowerConfig(packed_window=True, causal_readout='window_mean', dropout=0)
        store = store_fixture()
        batch, _, _, _, _ = store.train_batch(np.array([1]), config, 'cpu')
        mean = UserTower(config, 'causal').eval()
        last = UserTower(TowerConfig(packed_window=True, dropout=0), 'causal').eval()
        last.load_state_dict(mean.state_dict())
        with torch.no_grad():
            mean.gate.fill_(1)
            last.gate.fill_(1)
        items = torch.randn(len(batch['items']), 64)
        torch.testing.assert_close(mean(items, batch, batch['valid']), last(items, batch, batch['valid']))


if __name__ == '__main__':
    unittest.main()
