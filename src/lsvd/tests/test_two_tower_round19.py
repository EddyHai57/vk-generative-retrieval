import unittest

import numpy as np
import torch

from lsvd.two_tower import TowerConfig, UserTower
from lsvd.two_tower_data import EventStore


def store_fixture():
    size = 460
    columns = dict(item=np.arange(size, dtype=np.uint32),
                   week=np.tile(np.repeat([1, 2], 115), 2).astype(np.uint8),
                   place=np.zeros(size, dtype=np.uint8),
                   main=np.ones(size, dtype=bool), secondary=np.ones(size, dtype=bool),
                   feedback=np.ones(size, dtype=np.uint16), ratio=np.ones(size, dtype=np.float32))
    return EventStore(columns, np.array([7, 9]), np.array([0, 230, 460]))


class PackedWindowTests(unittest.TestCase):
    def test_strict_history_window_week_sharing_and_reverse_positions(self):
        config = TowerConfig(packed_window=True)
        store = store_fixture()
        batch, targets, _, _, _ = store.train_batch(np.array([200, 210, 229]), config, 'cpu')
        self.assertEqual(len(batch['group_offsets']), 2)
        for index, target in enumerate(targets):
            end = int(batch['readouts'][index])
            start = max(int(batch['group_offsets'][0]), end - 200)
            history = batch['items'][start:end].numpy()
            np.testing.assert_array_equal(history, np.arange(target - 200, target))
        weeks = store.columns['week'][batch['source_rows']]
        np.testing.assert_array_equal(batch['gap'].numpy(), 2 - weeks + 1)
        self.assertEqual(UserTower.relative_distances(torch.tensor([210]), torch.tensor([10, 209])).tolist(), [[199, 0]])

    def test_batches_keep_all_natural_targets_and_cap_each_user(self):
        store = store_fixture()
        config = TowerConfig(packed_window=True, batch_size=128)
        batches = list(store.packed_batches(config, np.random.default_rng(42)))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(len(store.samples)))
        for indices in batches:
            users = store.user_rows[store.samples[indices]]
            self.assertLessEqual(np.unique(users, return_counts=True)[1].max(), 64)
        self.assertTrue(any(len(np.unique(store.user_rows[store.samples[x]])) > 1 for x in batches))

    def test_future_and_outside_200_cannot_change_readout(self):
        torch.manual_seed(42)
        config = TowerConfig(packed_window=True, dropout=0.0)
        store = store_fixture()
        batch, _, _, _, _ = store.train_batch(np.array([200, 210]), config, 'cpu')
        model = UserTower(config, 'causal').eval()
        with torch.no_grad():
            model.gate.fill_(1.0)
        items = torch.randn(len(batch['items']), 64)
        before = model(items, batch, batch['valid'])
        changed = items.clone()
        first_end, second_end = map(int, batch['readouts'])
        changed[first_end:] += 100
        torch.testing.assert_close(before[0], model(changed, batch, batch['valid'])[0])
        changed = items.clone()
        changed[:second_end - 200] += 100
        torch.testing.assert_close(before[1], model(changed, batch, batch['valid'])[1], atol=2e-6, rtol=2e-6)
        before.sum().backward()
        self.assertIsNotNone(model.position.weight.grad)

    def test_empty_history_and_mean_use_same_packing(self):
        store = store_fixture()
        config = TowerConfig(packed_window=True, dropout=0.0)
        batch, _, _, _, _ = store.train_batch(np.array([0, 20]), config, 'cpu')
        items = torch.randn(len(batch['items']), 64)
        model = UserTower(config, 'mean').eval()
        output = model(items, batch, batch['valid'])
        torch.testing.assert_close(output[0], torch.zeros(64))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
