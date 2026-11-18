"""Small behavioral regression tests for the new retrieval entry point."""

import math
import unittest
import numpy as np
import torch
from lsvd.two_tower import (
    TowerConfig,
    ItemTower,
    UserTower,
    corrected_loss,
    rank_partition,
    rank_buckets,
)

from lsvd.two_tower_data import EventStore


def features(count=5):
    return {
        "content": torch.randn(count, 64),
        "id_index": torch.arange(count),
        "author_index": torch.tensor([0, 1, 2, 1, 0][:count]),
        "duration_bucket": torch.ones(count, dtype=torch.long),
        "rank_bucket": torch.arange(count),
    }


class LossTests(unittest.TestCase):
    def test_logq_weighted_matches_manual_and_positive_mask(self):
        users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        items = users.clone()
        ids = torch.tensor([10, 20])
        weights = torch.tensor([1.0, 0.25])
        loss, skips = corrected_loss(
            users, items, ids, [{10}, {20}], torch.tensor([0.75, 0.25]), weights, 0.1
        )
        scores = users @ items.T / 0.1 - torch.tensor([math.log(0.75), math.log(0.25)])
        manual = (
            torch.nn.functional.cross_entropy(scores, torch.arange(2), reduction="none")
            * weights
        ).sum() / weights.sum()
        self.assertAlmostEqual(float(loss), float(manual), places=6)
        self.assertEqual(skips, 0)
        masked, skips = corrected_loss(
            users,
            items,
            ids,
            [{10, 20}, {20}],
            torch.tensor([0.75, 0.25]),
            weights,
            0.1,
        )
        self.assertEqual(skips, 1)
        self.assertAlmostEqual(
            float(masked),
            float(torch.nn.functional.cross_entropy(scores[1:], torch.tensor([1]))),
            places=6,
        )

    def test_duplicate_targets_are_masked_and_no_negative_skips(self):
        x = torch.eye(2)
        loss, skips = corrected_loss(
            x,
            x,
            torch.tensor([9, 9]),
            [{9}, {9}],
            torch.ones(2) / 2,
            torch.ones(2),
            0.1,
        )
        self.assertEqual(skips, 2)
        self.assertEqual(float(loss), 0.0)


class ModelTests(unittest.TestCase):
    def test_id_dropout_and_cold_path_keep_content_author_duration(self):
        config = TowerConfig(id_dropout=1.0, use_rank=True)
        model = ItemTower(config, 5, 3)
        model.train()
        data = features()
        normal = model(data)
        cold = model(data, cold=True)
        torch.testing.assert_close(normal, cold)
        model.eval()
        before = model(data, cold=True)
        with torch.no_grad():
            model.ids.weight.add_(100)
            model.ranks.weight[1:].add_(100)
        torch.testing.assert_close(before, model(data, cold=True))
        self.assertTrue(torch.isfinite(before).all())
        torch.testing.assert_close(before.norm(dim=-1), torch.ones(5))

    def test_causal_prefix_and_padding_empty_history(self):
        config = TowerConfig(transformer_layers=1, dropout=0.0)
        model = UserTower(config, "causal")
        model.eval()
        x = torch.randn(2, 4, 64)
        valid = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
        event = {
            "feedback": torch.zeros(2, 4, 9),
            "ratio": torch.ones(2, 4, dtype=torch.long),
            "place": torch.ones(2, 4, dtype=torch.long),
            "gap": torch.ones(2, 4, dtype=torch.long),
        }
        hidden = model.encode(x, event, valid)
        changed = x.clone()
        changed[:, 2:, 0] += 500
        torch.testing.assert_close(
            hidden[:, :2], model.encode(changed, event, valid)[:, :2]
        )
        out = model(x, event, valid)
        self.assertTrue(torch.isfinite(out).all())
        torch.testing.assert_close(out[1], torch.zeros(64))
        changed = x.clone()
        changed[:, 3:] += 500
        torch.testing.assert_close(out, model(changed, event, valid))

    def test_dcn_changes_item_vectors_and_has_gradients(self):
        model = ItemTower(TowerConfig(cross_layers=2), 5, 3)
        out = model(features())
        out[:, 0].sum().backward()
        self.assertIsNotNone(model.cross[0].weight.grad)


class DataTests(unittest.TestCase):
    def test_history_strict_prefix_and_weights(self):
        columns = dict(
            item=np.arange(5, dtype=np.uint32),
            week=np.zeros(5, dtype=np.uint8),
            place=np.zeros(5, dtype=np.uint8),
            main=np.array([0, 0, 0, 1, 0], dtype=bool),
            secondary=np.array([0, 0, 1, 0, 0], dtype=bool),
            feedback=np.zeros(5, dtype=np.uint16),
            ratio=np.ones(5, dtype=np.float32),
        )
        store = EventStore(columns, np.array([1]), np.array([0, 5]))
        config = TowerConfig(max_history=2)
        batch, targets, _, _, weights = store.train_batch(
            np.array([0, 1]), config, "cpu"
        )
        np.testing.assert_array_equal(targets, [2, 3])
        np.testing.assert_array_equal(weights, [0.25, 1.0])
        for row, target in enumerate(targets):
            history = batch["items"][row].numpy()[batch["valid"][row].numpy()].tolist()
            self.assertLessEqual(len(history), 2)
            self.assertNotIn(int(target), history)
            self.assertTrue(all(item < int(target) for item in history))

    def test_rank_partition_boundary_and_train_only_buckets(self):
        rank = np.arange(10)
        counts = np.array([1, 2, 0, 0, 1, 0, 0, 0, 0, 0])
        boundary, groups = rank_partition(rank, counts)
        self.assertEqual(boundary, 5)
        np.testing.assert_array_equal(groups, [0, 0, 1, 1, 0, 2, 2, 2, 2, 2])
        buckets = rank_buckets(rank, boundary)
        self.assertTrue(np.all(buckets[5:] == 0))
        self.assertTrue(np.all(buckets[:5] > 0))

    def test_packing_uses_nine_bits_and_unknown_new_item(self):
        columns = dict(
            item=np.array([0]),
            week=np.array([0]),
            place=np.array([2]),
            main=np.array([False]),
            secondary=np.array([False]),
            feedback=np.array([511]),
            ratio=np.array([1.0]),
        )
        store = EventStore(columns, np.array([1]), np.array([0, 1]))
        batch = store.history_batch(np.array([1]), np.array([0]), [24], 200, "cpu")
        self.assertEqual(batch["feedback"].shape, (1, 1, 9))
        self.assertEqual(float(batch["feedback"].sum()), 9.0)
        self.assertEqual(int(batch["items"][0, 0]), 0)
        self.assertEqual(int(batch["valid"][0, 0]), 1)


if __name__ == "__main__":
    unittest.main()
