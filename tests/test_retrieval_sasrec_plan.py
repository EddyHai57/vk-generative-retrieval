import copy
import hashlib
import json
from pathlib import Path
import unittest

try:
    from vkgr import retrieval_sasrec_plan as plan
except ImportError:
    plan = None

ROOT = Path(__file__).resolve().parents[1]


class ModulePresenceTests(unittest.TestCase):
    def test_p4h_plan_module_exists(self):
        self.assertIsNotNone(plan, "P4H-01 plan validator is missing")


@unittest.skipIf(plan is None, "P4H-01 plan validator not implemented")
class FrozenPlanTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/p4h-sasrec-v2.json").read_bytes())

    def test_exact_task_slot_decision_and_cutoff_registry(self):
        value = plan.validate_config(copy.deepcopy(self.config))
        self.assertEqual(value["task_ids"], [
            "P4H-01", "P4H-02", "P4H-03A", "P4H-03B", "P4H-04",
            "P4H-05", "P4H-06", "P4H-07", "P4H-08",
        ])
        self.assertEqual(len(value["formal_slots"]), 12)
        self.assertEqual([row["slot_id"] for row in value["formal_slots"]], [
            "P4H-S01", "P4H-S02", "P4H-S03", "P4H-S04",
            "P4H-S05", "P4H-S06", "P4H-S07", "P4H-S08",
            "P4H-C01", "P4H-C02", "P4H-C03", "P4H-C04",
        ])
        self.assertEqual(value["remote_decisions"], {
            "D-P4H-REMOTE-01": "BLOCKED",
            "D-P4H-REMOTE-02": "BLOCKED",
            "D-P4H-G5-01": "BLOCKED_DEFAULT_ORIGINAL_P4D_TT",
        })
        self.assertEqual(value["common_test"], "CLOSED")
        self.assertEqual(value["g4_cutoff"], {
            "trigger": "G4_FREEZE",
            "hard_inputs": ["G4-freeze-receipt"],
            "optional_inputs": ["P4H-route-receipt"],
            "default": "ORIGINAL_P4D_TT",
            "wait_for_p4h": False,
        })
        self.assertEqual(value["route"]["top_k"], 80)
        self.assertEqual(value["route"]["g5_total_candidates"], 400)
        self.assertEqual(value["route"]["g5_route_quota"], 80)

    def test_slot_matrix_is_exact_2x2_with_matched_confirmation_only(self):
        slots = self.config["formal_slots"]
        seed42 = slots[:8]
        expected = [
            (history, 42, learning_rate, temperature)
            for history in ("like", "observed")
            for learning_rate, temperature in ((0.001, 0.07), (0.001, 0.15), (0.0003, 0.07), (0.0003, 0.15))
        ]
        self.assertEqual([(row["history_source"], row["seed"], row["learning_rate"], row["temperature"]) for row in seed42], expected)
        self.assertEqual(
            [(row["history_source"], row["seed"], row["learning_rate"], row["temperature"]) for row in slots[8:]],
            [("like", 2024, 0.001, 0.15), ("like", 2025, 0.001, 0.15), ("observed", 2024, 0.001, 0.15), ("observed", 2025, 0.001, 0.15)],
        )
        self.assertEqual({row["encoder"] for row in slots}, {"id_only_sasrec_style"})
        self.assertEqual(self.config["mean_controls"], [
            {"history_source": "like", "source": "P4D-id_mean", "mode": "read_only"},
            {"history_source": "observed", "source": "P4E-watch_history-id_mean", "mode": "read_only"},
        ])

    def test_config_drift_rejects_without_silent_search_or_test_open(self):
        mutations = []
        value = copy.deepcopy(self.config);value["formal_slots"].append(copy.deepcopy(value["formal_slots"][-1]));mutations.append(value)
        value = copy.deepcopy(self.config);value["formal_slots"][8]["learning_rate"] = 0.0003;mutations.append(value)
        value = copy.deepcopy(self.config);value["best_seed_selection"] = True;mutations.append(value)
        value = copy.deepcopy(self.config);value["common_test"] = "OPEN";mutations.append(value)
        value = copy.deepcopy(self.config);value["remote_decisions"]["D-P4H-REMOTE-01"] = "APPROVED";mutations.append(value)
        value = copy.deepcopy(self.config);value["proposed_thresholds"]["throughput_floor"]["status"] = "APPROVED";mutations.append(value)
        value = copy.deepcopy(self.config);value["g4_cutoff"]["wait_for_p4h"] = True;mutations.append(value)
        value = copy.deepcopy(self.config);value["accepted_inputs"]["p4f_target_review"] = "0" * 64;mutations.append(value)
        value = copy.deepcopy(self.config);value["public_boundary"]["server_paths_allowed"] = True;mutations.append(value)
        for mutation in mutations:
            with self.assertRaises(ValueError):
                plan.validate_config(mutation)


@unittest.skipIf(plan is None, "P4H-01 plan validator not implemented")
class PilotRolePolicyTests(unittest.TestCase):
    def fixture(self):
        return plan.synthetic_pilot_manifest()

    def test_prepared_train_target_and_watch_bijection_pass(self):
        manifest, payloads = self.fixture()
        touched = []
        plan.validate_pilot_manifest(manifest, before_path=lambda role: touched.append(role))
        result = plan.validate_pilot_payloads(payloads)
        self.assertTrue(result["prepared_watch_bijection"])
        self.assertEqual(result["validation_target_read"], False)
        self.assertEqual(result["quality_metrics_computed"], False)
        self.assertEqual(len(touched), len(manifest["roles"]))

    def test_exact_target_free_schema_id_passes_its_own_token(self):
        manifest, payloads = self.fixture()
        role = next(row for row in manifest["roles"] if row["role"] == "train-prefix-target-free-sidecar")
        self.assertEqual(role["schema_id"], "p4h-train-prefix-target-free-v1")
        self.assertEqual(role["schema_fields"], ["query_id", "history"])
        plan.validate_pilot_manifest(manifest)
        self.assertTrue(plan.validate_pilot_payloads(payloads)["sidecar_target_free"])

    def test_validation_or_holdout_lineage_stops_before_path(self):
        for split in ("validation", "holdout"):
            manifest, _ = self.fixture()
            role = next(row for row in manifest["roles"] if row["role"] == "train-prefix-target-free-sidecar")
            role["lineage"]["split"] = split
            role["lineage"]["source_partitions"] = [split + "25"]
            touched = []
            with self.assertRaisesRegex(ValueError, "STOP_BEFORE_PATH"):
                plan.validate_pilot_manifest(manifest, before_path=lambda role: touched.append(role))
            self.assertEqual(touched, [])

    def test_sidecar_forbidden_field_nested_or_value_semantics_stops_before_path(self):
        mutations = (
            lambda role: role["schema_fields"].append("target"),
            lambda role: role["lineage"]["semantic_tags"].append("quality_metric"),
            lambda role: role["lineage"].update(derived_from="validation_target_removed"),
        )
        for mutate in mutations:
            manifest, _ = self.fixture()
            role = next(row for row in manifest["roles"] if row["role"] == "train-prefix-target-free-sidecar")
            mutate(role);touched=[]
            with self.assertRaisesRegex(ValueError, "STOP_BEFORE_PATH"):
                plan.validate_pilot_manifest(manifest, before_path=lambda role: touched.append(role))
            self.assertEqual(touched, [])


@unittest.skipIf(plan is None, "P4H-01 plan validator not implemented")
class VerifierRoleSourceTests(unittest.TestCase):
    def test_three_verifier_role_tables_are_exact_and_nonoverlapping(self):
        expected_counts = {"pilot-verify": 21, "formal-verify": 27, "infer-route-verify": 25}
        for stage, count in expected_counts.items():
            fixture = plan.synthetic_verifier_manifest(stage)
            result = plan.validate_verifier_manifest(fixture["manifest"], fixture["sealed"])
            self.assertEqual(result["manifest_role_count"], count)
            groups = result["groups"]
            self.assertTrue(set(groups["common"]).isdisjoint(groups["sealed_wrapper"]))
            self.assertTrue(set(groups["input_hashes"]).isdisjoint(groups["artifact_hashes"]))
            self.assertNotIn(groups["sealed_wrapper"][0], groups["input_hashes"])
            self.assertNotIn(groups["sealed_wrapper"][0], groups["artifact_hashes"])
            self.assertNotIn(groups["sealed_wrapper"][0], result["complete_artifact_set_roles"])

    def test_missing_extra_or_cross_source_role_stops_before_nested_paths(self):
        for stage in ("pilot-verify", "formal-verify", "infer-route-verify"):
            fixture = plan.synthetic_verifier_manifest(stage)
            cases = []
            missing = copy.deepcopy(fixture);missing["sealed"]["artifact_hashes"].pop();cases.append(missing)
            extra = copy.deepcopy(fixture);extra["sealed"]["input_hashes"].append({"role":"extra-role","sha256":"a"*64});cases.append(extra)
            crossed = copy.deepcopy(fixture);crossed["sealed"]["input_hashes"][0]["role"] = crossed["sealed"]["artifact_hashes"][0]["role"];cases.append(crossed)
            for case in cases:
                touched=[]
                with self.assertRaisesRegex(ValueError, "STOP_BEFORE_PATH"):
                    plan.validate_verifier_manifest(case["manifest"], case["sealed"], before_nested_path=lambda role:touched.append(role))
                self.assertEqual(touched, [])

    def test_verifier_exact_schema_id_and_fields_reject_drift_before_nested_paths(self):
        for stage in ("pilot-verify", "formal-verify", "infer-route-verify"):
            fixture = plan.synthetic_verifier_manifest(stage)
            self.assertTrue(all("schema_fields" in row for row in fixture["manifest"]["roles"]))
            plan.validate_verifier_manifest(fixture["manifest"], fixture["sealed"])
            for key, value in (("schema_id", "wrong-schema"), ("schema_fields", ["extra"])):
                changed = copy.deepcopy(fixture)
                changed["manifest"]["roles"][-1][key] = value
                touched = []
                with self.assertRaisesRegex(ValueError, "STOP_BEFORE_PATH"):
                    plan.validate_verifier_manifest(changed["manifest"], changed["sealed"], before_nested_path=lambda role: touched.append(role))
                self.assertEqual(touched, [])
            changed = copy.deepcopy(fixture)
            changed["sealed"]["predecessor_stage"] = "wrong-stage"
            touched = []
            with self.assertRaisesRegex(ValueError, "STOP_BEFORE_PATH"):
                plan.validate_verifier_manifest(changed["manifest"], changed["sealed"], before_nested_path=lambda role: touched.append(role))
            self.assertEqual(touched, [])


@unittest.skipIf(plan is None, "P4H-01 plan validator not implemented")
class LifecycleAndFeedSchemaTests(unittest.TestCase):
    def test_mutating_and_verifier_operations_require_distinct_roots_and_identities(self):
        pairs = (("pilot", "pilot-verify"), ("formal-train", "formal-verify"), ("formal-resume", "formal-verify"), ("infer-route", "infer-route-verify"))
        for mutating, verifier in pairs:
            left = plan.synthetic_operation(mutating, "left")
            right = plan.synthetic_operation(verifier, "right")
            self.assertTrue(plan.validate_operation_pair(left, right)["independent_operations"])
            for field in ("operation_id", "exclusive_output_root", "exclusive_control_root", "controller_identity", "admission_sha256", "g_receipt_sha256", "worker_pgid"):
                changed = copy.deepcopy(right);changed[field] = left[field]
                with self.assertRaises(ValueError):
                    plan.validate_operation_pair(left, changed)

    def test_command_stage_map_has_exact_eleven_commands_and_local_remote_boundary(self):
        mapping = plan.COMMAND_STAGES
        self.assertEqual(list(mapping), ["prepare-controller", "pilot", "pilot-verify", "formal-train", "formal-resume", "formal-verify", "select", "completion-verify", "summarize", "infer-route", "infer-route-verify"])
        self.assertEqual({name for name, row in mapping.items() if row["runtime"] == "local"}, {"select", "completion-verify", "summarize"})
        self.assertEqual(mapping["prepare-controller"]["stages"], ["pilot", "pilot-verify", "formal-train", "formal-resume", "formal-verify", "infer-route", "infer-route-verify"])
        self.assertTrue(all(row["business_args"] == ["--operation-manifest", "--operation-manifest-sha256"] for row in mapping.values()))

    def test_g_feed_descriptor_schema_and_claim_boundary_are_exact(self):
        descriptor = plan.synthetic_feed_descriptor()
        self.assertTrue(plan.validate_feed_descriptor(descriptor)["schema_only"])
        self.assertEqual(descriptor["feed_mode"], "0700")
        self.assertEqual(descriptor["publisher_lock"]["creation"], "O_CREAT|O_EXCL")
        self.assertEqual(descriptor["receipt_schema_id"], "p4h-g-feed-receipt-v2")
        self.assertEqual(descriptor["allowed_statuses"], ["OPEN", "CLOSED", "OWNER_LOST"])
        self.assertEqual(plan.FEED_CLAIM_BOUNDARY, "accidental_duplicate_stale_wrong_job_only_not_malicious_same_uid")
        changed = copy.deepcopy(descriptor);changed["feed_mode"] = "0755"
        with self.assertRaises(ValueError):plan.validate_feed_descriptor(changed)


@unittest.skipIf(plan is None, "P4H-01 plan validator not implemented")
class OwnedArtifactManifestTests(unittest.TestCase):
    def test_public_manifest_binds_plan_source_test_protocol_and_config_without_self_hash(self):
        manifest_path = ROOT / "artifacts/p4h-sasrec-plan-manifest-v2.json"
        manifest = json.loads(manifest_path.read_bytes())
        self.assertEqual(manifest["task_id"], "P4H-01")
        self.assertEqual(manifest["status"], "LOCAL_FROZEN_REMOTE_BLOCKED")
        expected = {
            "src/vkgr/retrieval_sasrec_plan.py",
            "tests/test_retrieval_sasrec_plan.py",
            "docs/p4h_sasrec_protocol_v2.md",
            "configs/p4h-sasrec-v2.json",
        }
        self.assertEqual(set(manifest["members"]), expected)
        for name, record in manifest["members"].items():
            payload = (ROOT / name).read_bytes()
            self.assertEqual(record, {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        self.assertEqual(manifest["manifest_self_hash_policy"], "external_after_freeze_no_self_reference")
        self.assertEqual(manifest["remote_decisions"], {
            "D-P4H-REMOTE-01": "BLOCKED",
            "D-P4H-REMOTE-02": "BLOCKED",
            "D-P4H-G5-01": "BLOCKED_DEFAULT_ORIGINAL_P4D_TT",
        })


if __name__ == "__main__":
    unittest.main()
