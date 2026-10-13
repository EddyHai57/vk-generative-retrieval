"""P4H-03A finite typed science API tests; no operation ownership."""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import unittest

import torch

try:
    from vkgr import retrieval_sasrec_experiment as experiment
except ImportError:
    experiment = None

from vkgr import retrieval_sasrec_plan as plan
from vkgr.retrieval_sasrec import SASRecIDTwoTower


class ModulePresenceTests(unittest.TestCase):
    def test_typed_science_module_exists(self):
        self.assertIsNotNone(experiment, "P4H-03A typed science module is missing")


@unittest.skipIf(experiment is None, "P4H-03A not implemented")
class TypedBoundaryTests(unittest.TestCase):
    def test_exact_typed_api_has_no_cli_manifest_path_or_process_owner(self):
        self.assertEqual(experiment.SCIENCE_API, (
            "pilot_capacity", "verify_pilot", "train_slot", "resume_slot", "verify_slot",
            "select_route", "verify_completion", "summarize", "infer_route", "verify_infer_route",
        ))
        source = Path(experiment.__file__).read_text()
        tree = ast.parse(source)
        imported = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
        self.assertTrue(imported.isdisjoint({"argparse", "os", "pathlib", "subprocess", "sys", "vkgr.retrieval_sasrec_operation"}))
        self.assertNotIn("operation_manifest", source)
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body))
        for name in experiment.SCIENCE_API:
            self.assertTrue(callable(getattr(experiment, name)))
            self.assertTrue(all("manifest" not in parameter for parameter in inspect.signature(getattr(experiment, name)).parameters))

    def test_train_only_pilot_has_no_quality_or_validation_semantics(self):
        events = []
        sink = lambda kind, value: events.append((kind, value))
        measurements = {
            arm: {
                "history_source": arm,
                "optimizer_steps": 30,
                "measurements": {name: {"seconds": 1.0, "bytes": 2} for name in experiment.PILOT_MEASUREMENTS},
            }
            for arm in ("like", "observed")
        }
        queries = [{"query_id": index, "history": [1, 2]} for index in range(3)]
        receipt = experiment.pilot_capacity(measurements, queries, sink=sink)
        self.assertEqual(receipt["validation_target_read"], False)
        self.assertEqual(receipt["quality_metrics_computed"], False)
        self.assertNotIn("selection", receipt)
        self.assertEqual(experiment.verify_pilot(receipt)["status"], "pilot_completed+verified")
        self.assertEqual(events[0][0], "pilot")
        for changed in (
            [{"query_id": 1, "history": [1], "target": 2}],
            [{"query_id": 1, "history": [1], "quality_metric": 0.2}],
        ):
            with self.assertRaises(ValueError):
                experiment.pilot_capacity(measurements, changed, sink=sink)


@unittest.skipIf(experiment is None, "P4H-03A not implemented")
class FiniteTrainingAndVerificationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.content = torch.zeros(5, 64)
        self.idmap = {1: 1, 2: 2, 3: 3, 4: 4}
        self.rows = [
            {"history": [1, 2], "target": 3, "past_likes": [1, 2]},
            {"history": [2, 1], "target": 4, "past_likes": [1, 2]},
        ]
        self.batches = [{"epoch": epoch, "epoch_end": True, "rows": copy.deepcopy(self.rows)} for epoch in range(1, 31)]
        self.slot = copy.deepcopy(plan.FORMAL_SLOTS[1])

    def make(self):
        torch.manual_seed(42)
        model = SASRecIDTwoTower(self.content, [1, 2, 3, 4])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.slot["learning_rate"], weight_decay=1e-4)
        return model, optimizer

    @staticmethod
    def evaluate(model, epoch):
        return {
            "epoch": epoch,
            "metrics": {"all": {"recall@50": epoch / 100, "ndcg@50": epoch / 200}},
            "prediction_sha256": hashlib.sha256(f"epoch:{epoch}".encode()).hexdigest(),
        }

    def test_continuous_and_7_20_30_resume_are_exact(self):
        full_model, full_optimizer = self.make();torch.manual_seed(99)
        full = experiment.train_slot(full_model, full_optimizer, self.batches, self.idmap, self.slot, evaluate=self.evaluate, sink=lambda *_: None)
        first_model, first_optimizer = self.make();torch.manual_seed(99)
        first = experiment.train_slot(first_model, first_optimizer, self.batches, self.idmap, self.slot, evaluate=self.evaluate, sink=lambda *_: None, stop_after=7)
        second_model, second_optimizer = self.make()
        second = experiment.resume_slot(first, second_model, second_optimizer, self.batches, self.idmap, self.slot, evaluate=self.evaluate, sink=lambda *_: None, stop_after=20)
        final_model, final_optimizer = self.make()
        final = experiment.resume_slot(second, final_model, final_optimizer, self.batches, self.idmap, self.slot, evaluate=self.evaluate, sink=lambda *_: None)
        self.assertEqual(full["public"], final["public"])
        for name, value in full_model.state_dict().items():
            torch.testing.assert_close(value, final_model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(set(final["public"]["checkpoint_roles"]), {"best", "last"})
        self.assertLessEqual(len(first["public"]["checkpoint_roles"]), 3)
        self.assertIn("preemption", first["public"]["checkpoint_roles"])

    def test_multiple_batches_per_epoch_only_evaluate_at_boundary_and_patience_stops(self):
        batches = []
        for epoch in range(1, 7):
            batches.extend([
                {"epoch": epoch, "epoch_end": False, "rows": copy.deepcopy(self.rows)},
                {"epoch": epoch, "epoch_end": True, "rows": copy.deepcopy(self.rows)},
            ])
        calls = []
        def constant(model, epoch):
            calls.append(epoch)
            return {"epoch": epoch, "metrics": {"all": {"recall@50": 0.2, "ndcg@50": 0.1}}, "prediction_sha256": hashlib.sha256(f"fixed:{epoch}".encode()).hexdigest()}
        model, optimizer = self.make();torch.manual_seed(99)
        try:
            state = experiment.train_slot(model, optimizer, batches, self.idmap, self.slot, evaluate=constant, sink=lambda *_: None)
        except ValueError as error:
            self.fail(f"multi-batch epoch contract rejected: {error}")
        self.assertEqual(calls, [1, 2, 3, 4])
        self.assertEqual(state["public"]["steps"], 8)
        self.assertEqual(state["public"]["completed_epochs"], 4)
        self.assertEqual(state["public"]["stop_reason"], "patience")
        self.assertEqual(state["public"]["status"], "completed+unverified")

    def test_verifier_replays_draws_and_calls_fresh_best_once_without_optimizer(self):
        model, optimizer = self.make();torch.manual_seed(99)
        state = experiment.train_slot(model, optimizer, self.batches, self.idmap, self.slot, evaluate=self.evaluate, sink=lambda *_: None)
        calls = []
        def fresh(best_state, slot):
            calls.append((best_state, slot))
            return self.evaluate(model, 30)
        verified = experiment.verify_slot(state, self.batches, self.idmap, fresh_export_exact=fresh, sink=lambda *_: None)
        self.assertEqual(verified["status"], "completed+verified")
        self.assertEqual(len(calls), 1)
        self.assertEqual(verified["optimizer_steps_during_verify"], 0)
        damaged = copy.deepcopy(state);damaged["public"]["step_records"][0]["draw_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            experiment.verify_slot(damaged, self.batches, self.idmap, fresh_export_exact=fresh, sink=lambda *_: None)

    def test_operation_sink_emits_only_successful_training_actions_with_in_memory_identity(self):
        model, optimizer = self.make();torch.manual_seed(99)
        events = []
        state = experiment.train_slot(
            model, optimizer, self.batches[:1], self.idmap, self.slot,
            evaluate=self.evaluate, sink=lambda *_: None, operation_sink=lambda kind, payload: events.append((kind, payload)),
        )
        self.assertEqual([kind for kind, _ in events], ["optimizer_step", "epoch_validation", "checkpoint"])
        optimizer_event, validation_event, checkpoint_event = [payload for _, payload in events]
        self.assertEqual(optimizer_event["slot_id"], self.slot["slot_id"])
        self.assertEqual((optimizer_event["epoch"], optimizer_event["step"]), (1, 1))
        self.assertEqual(validation_event["prediction_sha256"], self.evaluate(model, 1)["prediction_sha256"])
        self.assertEqual(checkpoint_event["checkpoint_roles"], state["public"]["checkpoint_roles"])
        self.assertEqual(checkpoint_event["artifact_identity"], "in_memory_sha256_only")
        self.assertNotIn("path", checkpoint_event)

        failed_events = []
        bad_model, bad_optimizer = self.make();torch.manual_seed(99)
        def fail_evaluate(*_):
            raise ValueError("evaluation failed")
        with self.assertRaisesRegex(ValueError, "evaluation failed"):
            experiment.train_slot(
                bad_model, bad_optimizer, self.batches[:1], self.idmap, self.slot,
                evaluate=fail_evaluate, sink=lambda *_: None,
                operation_sink=lambda kind, payload: failed_events.append((kind, payload)),
            )
        self.assertEqual([kind for kind, _ in failed_events], ["optimizer_step"])

    def test_operation_sink_emits_fresh_export_exact_only_after_success(self):
        model, optimizer = self.make();torch.manual_seed(99)
        state = experiment.train_slot(model, optimizer, self.batches, self.idmap, self.slot, evaluate=self.evaluate, sink=lambda *_: None)
        events = []
        experiment.verify_slot(
            state, self.batches, self.idmap,
            fresh_export_exact=lambda *_: self.evaluate(model, 30), sink=lambda *_: None,
            operation_sink=lambda kind, payload: events.append((kind, payload)),
        )
        self.assertEqual([kind for kind, _ in events], ["fresh_export_exact"])
        self.assertEqual(events[0][1]["checkpoint_sha256"], state["public"]["checkpoint_roles"]["best"])
        self.assertEqual(events[0][1]["artifact_identity"], "in_memory_sha256_only")


@unittest.skipIf(experiment is None, "P4H-03A not implemented")
class SelectionCompletionAndSummaryTests(unittest.TestCase):
    METRIC_KEYS = ("recall@10", "recall@20", "recall@50", "ndcg@10", "ndcg@20", "ndcg@50")

    def records(self, failed=None):
        values = {}
        for slot in plan.FORMAL_SLOTS:
            base = 0.30 if slot["history_source"] == "like" else 0.40
            if slot["seed"] == 2024:base -= 0.01
            if slot["seed"] == 2025:base -= 0.02
            metrics = {key: base for key in self.METRIC_KEYS}
            is_failed = slot["slot_id"] == failed
            record = {
                "slot_id": slot["slot_id"],
                "status": "failed+spent" if is_failed else "completed+verified",
                "config": copy.deepcopy(slot),
                "metrics": {"all": metrics},
                "checkpoint_sha256": hashlib.sha256(slot["slot_id"].encode()).hexdigest(),
                "spent": {"steps": 1, "wall_seconds": 1.0, "new_bytes": 1},
            }
            if is_failed:
                record["failure_reason"] = "synthetic_resource_cap"
            values[slot["slot_id"]] = record
        return list(values.values())

    def controls(self):
        return {
            "like": {42: {key: 0.10 for key in self.METRIC_KEYS}, 2024: {key: 0.10 for key in self.METRIC_KEYS}, 2025: {key: 0.10 for key in self.METRIC_KEYS}},
            "observed": {42: {key: 0.20 for key in self.METRIC_KEYS}, 2024: {key: 0.20 for key in self.METRIC_KEYS}, 2025: {key: 0.20 for key in self.METRIC_KEYS}},
        }

    def test_completion_preserves_failed_spent_and_never_replaces_slot(self):
        records = self.records(failed="P4H-S03")
        result = experiment.verify_completion(records, sink=lambda *_: None)
        self.assertEqual(result["status"], "incomplete/negative")
        self.assertFalse(result["eligibility"])
        self.assertEqual(result["route_status"], "NOT_ELIGIBLE")
        self.assertEqual(result["failed_slots"], ["P4H-S03"])
        self.assertEqual(result["terminal_by_slot"]["P4H-S03"], {
            "status": "failed+spent", "spent": True, "failure_reason": "synthetic_resource_cap",
        })
        self.assertEqual(result["terminal_by_slot"]["P4H-S02"], {
            "status": "completed+verified", "spent": False, "failure_reason": None,
        })
        self.assertFalse(result["refund_allowed"])
        with self.assertRaises(ValueError):experiment.verify_completion(records[:-1], sink=lambda *_: None)
        with self.assertRaises(ValueError):experiment.verify_completion(records + [copy.deepcopy(records[0])], sink=lambda *_: None)

    def test_route_selection_uses_seed42_and_eligibility_uses_each_confirmation_seed(self):
        records = self.records()
        selected = experiment.select_route(records, self.controls(), sink=lambda *_: None)
        self.assertEqual(selected["history_source"], "observed")
        self.assertEqual(selected["slot_id"], "P4H-S06")
        self.assertTrue(selected["selection_conditioned"])
        self.assertTrue(selected["eligible"])
        controls = self.controls();controls["observed"][2025] = {key: 0.50 for key in self.METRIC_KEYS}
        rejected = experiment.select_route(records, controls, sink=lambda *_: None)
        self.assertFalse(rejected["eligible"])
        self.assertEqual(rejected["confirmation"], {2024: True, 2025: False})

    def test_route_selection_uses_all_eligible_not_slice_quality(self):
        records = self.records()
        for record in records:
            record["metrics"]["in_catalog"] = {key: (0.9 if record["slot_id"] == "P4H-S02" else 0.1) for key in self.METRIC_KEYS}
            record["metrics"]["oov"] = {key: 0.0 for key in self.METRIC_KEYS}
        try:
            selected = experiment.select_route(records, self.controls(), sink=lambda *_: None)
        except ValueError as error:
            self.fail(f"all-eligible selection rejected valid slice metrics: {error}")
        self.assertEqual(selected["slot_id"], "P4H-S06")
        self.assertEqual(selected["selection_scope"], "all_eligible")

    def test_five_factorial_estimands_and_primary_slice_members_are_hand_calculated(self):
        records = self.records()
        controls = self.controls()
        members = {name: "a" * 64 for name in ("like_mean", "observed_mean", "like_sasrec", "observed_sasrec")}
        slices = {name: {"members": dict(members)} for name in ("train_like_count_1", "train_like_count_2_5", "train_like_count_6_20", "train_like_count_gt20")}
        summary = experiment.summarize(records, controls, slices, sink=lambda *_: None)
        seed42 = summary["estimands"][42]["recall@50"]
        self.assertAlmostEqual(seed42["like_encoder_effect"], 0.20)
        self.assertAlmostEqual(seed42["observed_encoder_effect"], 0.20)
        self.assertAlmostEqual(seed42["mean_history_source_effect"], 0.10)
        self.assertAlmostEqual(seed42["sasrec_history_source_effect"], 0.10)
        self.assertAlmostEqual(seed42["interaction"], 0.0)
        self.assertTrue(summary["selection_conditioned"])
        failed_records = self.records(failed="P4H-C04")
        negative = experiment.summarize(failed_records, controls, slices, sink=lambda *_: None)
        self.assertEqual(negative["route_status"], "NOT_ELIGIBLE")
        self.assertFalse(negative["infer_allowed"])
        self.assertEqual(negative["negative_ledger"]["P4H-C04"], {
            "status": "failed+spent", "spent": True, "failure_reason": "synthetic_resource_cap",
        })
        self.assertEqual(set(negative["missing_estimand_cells"]), {42, 2024, 2025})
        for seed, metrics in negative["missing_estimand_cells"].items():
            self.assertEqual(set(metrics), set(self.METRIC_KEYS))
            for metric, cells in metrics.items():
                self.assertEqual(set(cells), {
                    "like_encoder_effect", "observed_encoder_effect", "mean_history_source_effect",
                    "sasrec_history_source_effect", "interaction",
                })
                self.assertTrue(all(reason == "undefined_due_to_failed_spent:P4H-C04" for reason in cells.values()))
        changed = copy.deepcopy(slices);changed["train_like_count_1"]["members"]["observed_sasrec"] = "b" * 64
        with self.assertRaises(ValueError):experiment.summarize(records, controls, changed, sink=lambda *_: None)


@unittest.skipIf(experiment is None, "P4H-03A not implemented")
class StrictRouteTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.catalog = list(range(100, 180))
        self.idmap = {item: index + 1 for index, item in enumerate(self.catalog)}
        content = torch.zeros(len(self.catalog) + 1, 64)
        torch.manual_seed(3);self.model = SASRecIDTwoTower(content, list(range(1, len(content)))).eval()
        with torch.no_grad():
            for parameter in self.model.parameters():parameter.zero_()
        self.queries = [
            {"query_id": 1, "user_id": 7, "history": [100], "history_positions": [0]},
            {"query_id": 2, "user_id": 8, "history": [], "history_positions": []},
        ]
        self.popularity = list(reversed(self.catalog))

    def test_strict_top80_real_ids_tie_fallback_and_shard_resume(self):
        events = []
        partial = experiment.infer_route(
            self.model, self.catalog, self.idmap, self.queries, self.popularity,
            shard_size=1, max_shards=1, sink=lambda *_: None,
            operation_sink=lambda kind, payload: events.append((kind, payload)),
        )
        self.assertEqual(partial["status"], "partial")
        self.assertEqual([kind for kind, _ in events], ["route_shard"])
        self.assertEqual(events[0][1]["shard"], 0)
        self.assertEqual(events[0][1]["range"], [0, 1])
        self.assertEqual(events[0][1]["record_count"], 1)
        self.assertEqual(events[0][1]["records_sha256"], hashlib.sha256(experiment._canonical(partial["shards"][0]["records"])).hexdigest())
        self.assertEqual(events[0][1]["artifact_identity"], "in_memory_sha256_only")
        self.assertNotIn("path", events[0][1])
        changed_queries = copy.deepcopy(self.queries);changed_queries[0]["history"] = [101]
        with self.assertRaises(ValueError):
            experiment.infer_route(self.model, self.catalog, self.idmap, changed_queries, self.popularity, shard_size=1, resume=partial, sink=lambda *_: None)
        with self.assertRaises(ValueError):
            experiment.infer_route(self.model, self.catalog, self.idmap, self.queries, list(reversed(self.popularity)), shard_size=1, resume=partial, sink=lambda *_: None)
        complete = experiment.infer_route(self.model, self.catalog, self.idmap, self.queries, self.popularity, shard_size=1, resume=partial, sink=lambda *_: None)
        verified = experiment.verify_infer_route(complete, self.queries, self.catalog, sink=lambda *_: None)
        self.assertEqual(verified["status"], "P4H_ROUTE_VALIDATION_READY")
        records = [record for shard in complete["shards"] for record in shard["records"]]
        self.assertEqual(records[0]["candidates"], self.catalog)
        self.assertEqual(records[1]["candidates"], self.popularity)
        self.assertTrue(records[1]["fallback"])
        self.assertTrue(all(len(record["candidates"]) == len(set(record["candidates"])) == 80 for record in records))

    def test_catalog79_unknown_history_target_fields_and_short_candidates_fail(self):
        with self.assertRaises(ValueError):
            experiment.infer_route(self.model, self.catalog[:-1], {item: index + 1 for index, item in enumerate(self.catalog[:-1])}, self.queries, self.popularity[:-1], sink=lambda *_: None)
        changed = copy.deepcopy(self.queries);changed[0]["history"] = [999]
        with self.assertRaises(ValueError):
            experiment.infer_route(self.model, self.catalog, self.idmap, changed, self.popularity, sink=lambda *_: None)
        changed = copy.deepcopy(self.queries);changed[0]["target"] = 101
        with self.assertRaises(ValueError):
            experiment.infer_route(self.model, self.catalog, self.idmap, changed, self.popularity, sink=lambda *_: None)
        complete = experiment.infer_route(self.model, self.catalog, self.idmap, self.queries, self.popularity, sink=lambda *_: None)
        complete["shards"][0]["records"][0]["candidates"].pop()
        with self.assertRaises(ValueError):
            experiment.verify_infer_route(complete, self.queries, self.catalog, sink=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
