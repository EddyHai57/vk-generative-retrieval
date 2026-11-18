import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from lsvd.two_tower import TowerConfig, TwoTower
from lsvd.two_tower_data import synthetic, encode_batch, EventStore


class PipelineTests(unittest.TestCase):
    def test_synthetic_store_is_strict_prefix_and_cold_items_retrievable(self):
        config = TowerConfig(dropout=0.0)
        catalog, store, data, _ = synthetic(config, user_count=32, item_count=1200)
        sample_indices = np.array([0, 1, 20, 51])
        batch, targets, _, _, _ = store.train_batch(sample_indices, config, "cpu")
        for row, target in enumerate(targets):
            self.assertNotIn(
                int(target), batch["items"][row][batch["valid"][row]].tolist()
            )
        model = TwoTower(catalog, config)
        model.eval()
        users = encode_batch(
            model, catalog, store.query_batch(data.user_ids[:2], config, "cpu"), "cpu"
        )
        cold = model.item(catalog.features(np.array([801, 809])))
        self.assertTrue(torch.isfinite(users @ cold.T).all())
        self.assertTrue((catalog.id_index[catalog.rows([801, 809])] == 0).all())

    def test_cli_synthetic_train_retrieval_shared_evaluator(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "trial"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lsvd.two_tower_run",
                    "synthetic",
                    "--output",
                    str(output),
                    "--epochs",
                    "1",
                    "--users",
                    "32",
                    "--max-train-samples",
                    "128",
                    "--replicates",
                    "2",
                    "--structures",
                    "mlp_mean",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["models"]), 1)
            self.assertTrue((output / "mlp_mean/evaluation/all/metrics.csv").exists())
            self.assertTrue(
                (output / "mlp_mean/cold-evaluation/group_diagnostics.json").exists()
            )
            self.assertGreater(summary["models"][0]["training_samples_per_second"], 0)
            self.assertGreater(summary["models"][0]["main"]["hits"], 0)
            self.assertEqual(len(summary["models"][0]["groups"]), 13)
            again = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lsvd.two_tower_run",
                    "synthetic",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(again.returncode, 0)

    def test_other_place_positives_are_masked_but_not_sampled(self):
        columns = dict(
            item=np.array([10, 20, 30]),
            week=np.array([0, 1, 2]),
            place=np.array([0, 2, 1]),
            main=np.array([True, True, False]),
            secondary=np.array([False, False, True]),
            feedback=np.array([1, 1, 2]),
            ratio=np.ones(3),
        )
        store = EventStore(columns, np.array([7]), np.array([0, 3]))
        self.assertEqual(store.samples.tolist(), [0, 2])
        self.assertEqual(store.positives[7], {10, 20, 30})
        self.assertEqual(store.frequency_ids.tolist(), [10, 30])

    def test_invalid_command_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "lsvd.two_tower_run", "invalid"], capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)


class FullScopeTests(unittest.TestCase):
    def test_full_training_scope_keeps_users_without_validation(self):
        from lsvd.two_tower_data import select_trial_users

        users = select_trial_users(
            np.array([1, 2]), np.array([1, 3]), 2, 7, all_training=True
        )
        np.testing.assert_array_equal(users, [1, 2, 3])
        sampled = select_trial_users(
            np.array([1, 2]), np.array([1, 3]), 1, 7, all_training=False
        )
        self.assertEqual(len(sampled), 1)
        self.assertNotIn(3, sampled)


class InitializationTests(unittest.TestCase):
    def test_legacy_small_embedding_initialization_keeps_padding_zero(self):
        catalog, _, _, _ = synthetic(TowerConfig(), user_count=8)
        model = TwoTower(catalog, TowerConfig(), user_structure="causal")
        self.assertLess(float(model.item.ids.weight.std().detach()), 0.03)
        self.assertGreater(float(model.item.ids.weight.std().detach()), 0.01)
        torch.testing.assert_close(model.item.ids.weight[0], torch.zeros(64))


class ConfigurationTests(unittest.TestCase):
    def test_rank_and_training_initial_values_are_cli_switches(self):
        result = subprocess.run(
            [sys.executable, "-m", "lsvd.two_tower_run", "synthetic", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in (
            "--use-rank",
            "--id-dropout",
            "--temperature",
            "--patience",
            "--seed",
        ):
            self.assertIn(flag, result.stdout)


class LegacyResidualTests(unittest.TestCase):
    def test_zero_gate_equals_mean_with_shared_initialization(self):
        config = TowerConfig(dropout=0.0)
        torch.manual_seed(18)
        mean = __import__("lsvd.two_tower", fromlist=["UserTower"]).UserTower(
            config, "mean"
        )
        torch.manual_seed(18)
        causal = __import__("lsvd.two_tower", fromlist=["UserTower"]).UserTower(
            config, "causal"
        )
        mean.eval()
        causal.eval()
        x = torch.randn(2, 4, 64)
        valid = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        event = dict(
            feedback=torch.zeros(2, 4, 9),
            ratio=torch.ones(2, 4, dtype=torch.long),
            place=torch.ones(2, 4, dtype=torch.long),
            gap=torch.ones(2, 4, dtype=torch.long),
        )
        torch.testing.assert_close(
            mean(x, event, valid), causal(x, event, valid), rtol=0, atol=0
        )
        with torch.no_grad():
            causal.gate.fill_(1)
        events = causal.events(x, event, valid)
        average = events.sum(1) / valid.sum(1, keepdim=True)
        hidden = causal.residual_norm(causal.encode(x, event, valid)) * valid.unsqueeze(
            -1
        )
        expected = torch.nn.functional.normalize(
            average + hidden.sum(1) / valid.sum(1, keepdim=True), dim=-1
        )
        torch.testing.assert_close(causal(x, event, valid), expected)

    def test_shared_item_initialization_across_user_structures(self):
        config = TowerConfig()
        catalog, _, _, _ = synthetic(config, user_count=8)
        torch.manual_seed(42)
        mean = TwoTower(catalog, config, "mean")
        torch.manual_seed(42)
        causal = TwoTower(catalog, config, "causal")
        for name, value in mean.item.state_dict().items():
            torch.testing.assert_close(
                value, causal.item.state_dict()[name], rtol=0, atol=0
            )


class ContentCacheTests(unittest.TestCase):
    def test_training_content_cache_is_exact_and_full_catalog_stays_available(self):
        catalog, store, data, _ = synthetic(TowerConfig(), user_count=8)
        ids = np.array([[1, 2, 3], [100, 200, 801]])
        original = catalog.features(ids)["content"].clone()
        with tempfile.TemporaryDirectory() as folder:
            catalog.cache_training_content(
                np.r_[store.columns["item"], [0, 1, 9, 17]], Path(folder) / "content"
            )
            torch.testing.assert_close(
                catalog.features(ids)["content"], original, rtol=0, atol=0
            )
            expected = torch.as_tensor(catalog.columns["content"][[1, 9, 17]].copy())
            catalog.columns["content"][[1, 9, 17]] = 999
            cached = catalog.features(np.array([1, 9, 17]))["content"]
            torch.testing.assert_close(cached, expected, rtol=0, atol=0)
            self.assertTrue((Path(folder) / "content/values.npy").exists())


class ThreadRestoreTests(unittest.TestCase):
    def test_retrieval_restores_shared_threads_after_output_error(self):
        from lsvd.two_tower_retrieval import retrieve

        config = TowerConfig()
        catalog, store, data, _ = synthetic(config, user_count=8)
        model = TwoTower(catalog, config)
        torch.set_num_threads(4)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                retrieve(
                    model, catalog, store, data, "cpu", Path(folder), exact_users=0
                )
        self.assertEqual(torch.get_num_threads(), 4)


class PlaceEncodingTests(unittest.TestCase):
    def test_unsigned_place_codes_stay_distinct(self):
        columns = dict(
            item=np.array([1, 2]),
            week=np.array([0, 0]),
            place=np.array([254, 255], dtype=np.uint8),
            main=np.array([False, False]),
            secondary=np.array([False, False]),
            feedback=np.zeros(2, dtype=np.uint16),
            ratio=np.ones(2),
        )
        store = EventStore(columns, np.array([7]), np.array([0, 2]))
        batch = store.history_batch(np.array([2]), np.array([0]), [24], 200, "cpu")
        self.assertEqual(batch["place"].tolist(), [[255, 256]])


class ExtendedGroupingTests(unittest.TestCase):
    def test_three_groups_save_all_rulers_and_cutoffs(self):
        from lsvd.two_tower_retrieval import diagnostic_evaluations

        config = TowerConfig()
        catalog, store, data, _ = synthetic(config, user_count=8)
        predictions = {
            int(u): [int(data.targets["item_id"][data.targets["offsets"][i]])]
            for i, u in enumerate(data.user_ids)
        }
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "evaluation"
            _, groups = diagnostic_evaluations(
                catalog,
                data,
                predictions,
                "group-test",
                output,
                2,
                diagnostic_ks=(50, 200, 1000),
                save_diagnostics=True,
            )
            self.assertTrue(all(row["k"] == 200 for row in groups))
            files = list((output / "groups").glob("*/metrics.json"))
            self.assertEqual(len(files), 13)
            rows = json.loads(files[0].read_text())
            self.assertEqual({row["k"] for row in rows}, {50, 200, 1000})
            self.assertEqual(
                {row["ruler"] for row in rows}, {"main", "secondary", "like"}
            )


class PostEvaluationAdapterTests(unittest.TestCase):
    def test_popularity_adapter_preserves_queries_and_full_history_filter(self):
        from lsvd.two_tower_postevaluate import local_popularity_predictions

        catalog, store, data, _ = synthetic(TowerConfig(), user_count=8)
        predictions = local_popularity_predictions(data)
        self.assertEqual(predictions.shape, (8, 1000))
        for row, user in enumerate(data.user_ids):
            self.assertFalse(
                set(predictions[row].tolist()) & set(data.seen_items(user).tolist())
            )
        from lsvd.baselines import BaselineData, global_popularity

        reference = BaselineData(
            dict(user_id=data.user_ids),
            data.items,
            data.history_user_ids,
            data.history_offsets,
            data.history_items,
        )
        np.testing.assert_array_equal(predictions, global_popularity(reference))


class DryRunTests(unittest.TestCase):
    def test_forward_only_entry_has_receipt_and_no_training_outputs(self):
        from lsvd.two_tower_run import main

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "dry"
            self.assertEqual(
                main(
                    ["synthetic", "--output", str(output), "--users", "8", "--dry-run"]
                ),
                0,
            )
            record = json.loads((output / "dry_run.json").read_text())
            self.assertEqual(record["optimizer_steps"], 0)
            self.assertEqual(len(record["models"]), 3)
            self.assertTrue(all(model["loss_finite"] for model in record["models"]))
            self.assertFalse((output / "summary.json").exists())
            self.assertFalse(any(output.rglob("*.pt")))


class MemoryUnitTests(unittest.TestCase):
    def test_peak_rss_is_bytes_on_mac_and_linux(self):
        from lsvd.two_tower_run import peak_rss_bytes
        from types import SimpleNamespace
        from unittest.mock import patch

        with patch(
            "lsvd.two_tower_run.resource.getrusage",
            return_value=SimpleNamespace(ru_maxrss=123),
        ):
            with patch("lsvd.two_tower_run.sys.platform", "darwin"):
                self.assertEqual(peak_rss_bytes(), 123)
            with patch("lsvd.two_tower_run.sys.platform", "linux"):
                self.assertEqual(peak_rss_bytes(), 123 * 1024)
