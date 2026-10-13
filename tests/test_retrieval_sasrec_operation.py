"""P4H-03B local operation/controller/feed synthetic tests."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

try:
    from vkgr import retrieval_sasrec_operation as operation
except ImportError:
    operation = None

from vkgr import retrieval_sasrec_plan as plan


class ModulePresenceTests(unittest.TestCase):
    def test_operation_module_exists(self):
        self.assertIsNotNone(operation, "P4H-03B operation module is missing")


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class CommandAndManifestTests(unittest.TestCase):
    def test_exact_eleven_command_parser_and_one_way_science_dispatch(self):
        parser = operation.build_parser()
        self.assertEqual(list(operation.COMMAND_HANDLERS), list(plan.COMMAND_STAGES))
        for command in plan.COMMAND_STAGES:
            parsed = parser.parse_args([command, "--operation-manifest", "/server-only/op.json", "--operation-manifest-sha256", "a" * 64])
            self.assertEqual(parsed.command, command)
            self.assertEqual(set(vars(parsed)), {"command", "operation_manifest", "operation_manifest_sha256"})
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([command, "--operation-manifest", "/op", "--operation-manifest-sha256", "a" * 64, "--seed", "42"])
        with patch.object(operation.science, "verify_completion", return_value={"status": "complete"}) as call:
            self.assertEqual(operation.dispatch_science("completion-verify", [], sink=lambda *_: None), {"status": "complete"})
            call.assert_called_once()

    def test_invalid_pilot_role_or_lineage_stops_before_any_path_probe(self):
        value = operation.synthetic_operation_manifest("pilot")
        touched = []
        result = operation.validate_static_manifest(value, path_probe=lambda role, path: touched.append((role, path)))
        self.assertEqual(result["stage"], "pilot")
        self.assertGreater(len(touched), 0)
        for mutate in (
            lambda role: role.update(schema_id="wrong"),
            lambda role: role["lineage"].update(split="validation", source_partitions=["validation25"]),
        ):
            changed = copy.deepcopy(value)
            role = next(row for row in changed["roles"] if row["role"] == "train-prefix-target-free-sidecar")
            mutate(role);touched=[]
            with self.assertRaisesRegex(ValueError, "STOP_BEFORE_PATH"):
                operation.validate_static_manifest(changed, path_probe=lambda role, path: touched.append((role, path)))
            self.assertEqual(touched, [])

    def test_relative_roots_hash_drift_and_remote_decision_fail_closed(self):
        value = operation.synthetic_operation_manifest("pilot")
        for mutate in (
            lambda row: row.update(code_root="relative"),
            lambda row: row.update(exclusive_output_root="relative"),
            lambda row: row.update(source_sha256="0" * 64),
            lambda row: row["remote_decisions"].update({"D-P4H-REMOTE-01": "APPROVED"}),
        ):
            changed=copy.deepcopy(value);mutate(changed)
            with self.assertRaises(ValueError):operation.validate_static_manifest(changed)


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class ControllerBootstrapTests(unittest.TestCase):
    def test_prepare_has_no_worker_then_admission_binds_start_and_fresh_g(self):
        value = operation.synthetic_operation_manifest("pilot")
        identity = operation.current_process_identity()
        with patch.object(operation.subprocess, "Popen", side_effect=AssertionError("worker before admission")):
            start = operation.prepare_controller(value, identity, sink=lambda *_: None)
        self.assertEqual(start["state"], "AWAITING_ADMISSION")
        self.assertFalse(start["worker_spawned"])
        self.assertEqual(start["controller_pgid"], identity["pgid"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with operation.synthetic_live_feed(root / "feed", now=time.time()) as publisher:
                controller = operation.LiveFeedController(publisher.descriptor_path, publisher.descriptor_sha256, root / "accepted.json")
                observed = controller.observe(now=time.time())
                admission = operation.synthetic_admission(start, observed["receipt_sha256"], issued_at=time.time())
                accepted = operation.admit_controller(start, admission, controller, now=time.time(), sink=lambda *_: None)
                self.assertEqual(accepted["state"], "ADMITTED")
                self.assertEqual(accepted["controller_start_receipt_sha256"], operation.digest(start))
                stale = copy.deepcopy(admission);stale["expires_at"] = time.time() - 1
                with self.assertRaises(ValueError):operation.admit_controller(start, stale, controller, now=time.time(), sink=lambda *_: None)

    def test_start_or_admission_identity_drift_never_spawns(self):
        value=operation.synthetic_operation_manifest("pilot");start=operation.prepare_controller(value,operation.current_process_identity(),sink=lambda *_:None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            with operation.synthetic_live_feed(root/"feed",now=time.time()) as publisher:
                controller=operation.LiveFeedController(publisher.descriptor_path,publisher.descriptor_sha256,root/"accepted.json");observed=controller.observe(now=time.time());admission=operation.synthetic_admission(start,observed["receipt_sha256"],issued_at=time.time())
                for changed in (
                    {**start,"controller_start_ticks":"drift"},
                    {**admission,"operation_id":"other"},
                    {**admission,"g_receipt_sha256":"0"*64},
                ):
                    with patch.object(operation,"spawn_synthetic_worker_tree",side_effect=AssertionError("must not spawn")):
                        with self.assertRaises(ValueError):operation.admit_controller(start if "expires_at" in changed else changed,changed if "expires_at" in changed else admission,controller,now=time.time(),sink=lambda *_:None)


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class ProcessTreeSafeStopTests(unittest.TestCase):
    def test_controller_outside_worker_group_closes_child_grandchild_and_preserves_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();stop=root/"stop.json";sentinel=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])
            handle=None
            try:
                handle=operation.spawn_synthetic_worker_tree(stop)
                self.assertNotEqual(handle["worker_identity"]["pgid"],os.getpgrp())
                self.assertNotEqual(handle["worker_identity"]["pgid"],sentinel.pid)
                self.assertTrue(handle["grandchild_identity"])
                result=operation.stop_worker_tree(handle,stop,expected_identity=handle["worker_identity"],seal_deadline=3.0,signal_grace=0.5)
                self.assertEqual(result["state"],"CLOSED")
                self.assertTrue(result["descendant_closure_verified"])
                self.assertTrue(result["single_preemption_checkpoint"])
                self.assertIsNone(sentinel.poll())
            finally:
                if handle is not None and handle["process"].poll() is None:
                    operation.stop_worker_tree(handle,stop,expected_identity=handle["worker_identity"],seal_deadline=1.0,signal_grace=0.2)
                sentinel.terminate();sentinel.wait(timeout=3)

    def test_birth_drift_fails_unsafe_without_signal_then_correct_identity_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop=Path(tmp).resolve()/"stop.json";handle=operation.spawn_synthetic_worker_tree(stop)
            try:
                wrong={**handle["worker_identity"],"start_ticks":str(handle["worker_identity"]["start_ticks"])+"-wrong"}
                result=operation.stop_worker_tree(handle,stop,expected_identity=wrong,seal_deadline=0.2,signal_grace=0.1)
                self.assertEqual(result["state"],"FAILED_UNSAFE")
                self.assertEqual(result["signals_sent"],0)
                self.assertIsNone(handle["process"].poll())
            finally:
                closed=operation.stop_worker_tree(handle,stop,expected_identity=handle["worker_identity"],seal_deadline=2.0,signal_grace=0.2)
                self.assertEqual(closed["state"],"CLOSED")


class FeedFixtureMixin:
    def descriptor(self, root=None):
        value=plan.synthetic_feed_descriptor()
        if root is not None:
            stat=root.stat();value["feed_root"]=str(root);value["feed_root_identity"]={"device":stat.st_dev,"inode":stat.st_ino};value["feed_owner_uid"]=os.getuid()
        value["publisher_process"]=operation.current_publisher_identity() if operation is not None else value["publisher_process"]
        return value


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class GFeedDirectoryOwnershipTests(FeedFixtureMixin,unittest.TestCase):
    def test_owner_mode_root_identity_and_symlink_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/"feed";root.mkdir(mode=0o700);descriptor=self.descriptor(root)
            self.assertTrue(operation.validate_feed_directory(descriptor)["owner_mode_identity_verified"])
            root.chmod(0o755)
            with self.assertRaises(ValueError):operation.validate_feed_directory(descriptor)
            root.chmod(0o700)
            alias=root.parent/"alias";alias.symlink_to(root,target_is_directory=True);changed={**descriptor,"feed_root":str(alias)}
            with self.assertRaises(ValueError):operation.validate_feed_directory(changed)


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class GFeedExclusivePublisherLockTests(FeedFixtureMixin,unittest.TestCase):
    def test_exclusive_lock_refuses_second_publisher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()/"feed";root.mkdir(mode=0o700)
            fd,record=operation.acquire_synthetic_publisher_lock(root)
            try:
                self.assertEqual(record["creation"],"O_CREAT|O_EXCL")
                with self.assertRaises(FileExistsError):operation.acquire_synthetic_publisher_lock(root)
            finally:os.close(fd)

    def test_live_controller_rejects_lock_file_without_holder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            publisher=operation.synthetic_live_feed(root/"feed",now=time.time())
            controller=operation.LiveFeedController(publisher.descriptor_path,publisher.descriptor_sha256,root/"state.json")
            publisher.close()
            with self.assertRaisesRegex(ValueError,"lock holder"):
                controller.observe(now=time.time())


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class GFeedPublisherBirthExecutableTests(FeedFixtureMixin,unittest.TestCase):
    def test_live_birth_and_executable_match_and_drift_reject(self):
        descriptor=self.descriptor();observed=operation.current_publisher_identity()
        self.assertTrue(operation.validate_publisher_identity(descriptor,observed)["publisher_birth_and_executable_verified"])
        for key in ("start_ticks","executable_sha256"):
            changed=copy.deepcopy(observed);changed[key]="0"*64 if key.endswith("sha256") else "wrong"
            with self.assertRaises(ValueError):operation.validate_publisher_identity(descriptor,changed)


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class GFeedWriterInventoryTests(FeedFixtureMixin,unittest.TestCase):
    def test_exact_one_expected_writer_and_unknown_writer_reject(self):
        descriptor=self.descriptor();publisher=descriptor["publisher_process"]
        inventory=operation.synthetic_writer_inventory(descriptor,[publisher])
        self.assertTrue(operation.validate_writer_inventory(descriptor,inventory)["no_other_writer_inventory_verified"])
        changed=copy.deepcopy(inventory);changed["writable_fd_processes"].append({**publisher,"pid":publisher["pid"]+1,"fd_modes":["w"]});changed["unknown_writer_count"]=1
        with self.assertRaises(ValueError):operation.validate_writer_inventory(descriptor,changed)


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class GFeedReceiptChainTests(FeedFixtureMixin,unittest.TestCase):
    def test_normal_atomic_head_old_or_new_and_terminal_statuses(self):
        snapshot=operation.synthetic_feed_snapshot(now=100.0,count=3)
        self.assertEqual(operation.validate_feed_snapshot(snapshot,now=100.5)["action"],"CONTINUE")
        old=copy.deepcopy(snapshot);old["head"]={"seq":1,"receipt_name":snapshot["receipts"][1]["name"],"receipt_sha256":snapshot["receipts"][1]["sha256"]}
        self.assertEqual(operation.validate_feed_snapshot(old,now=100.5)["head_seq"],1)
        closed=copy.deepcopy(snapshot);closed["receipts"][-1]["value"]["status"]="CLOSED";operation.rehash_snapshot(closed)
        self.assertEqual(operation.validate_feed_snapshot(closed,now=100.5)["action"],"STOP_REQUESTED")
        lost=copy.deepcopy(snapshot);lost["receipts"][-1]["value"]["status"]="OWNER_LOST";operation.rehash_snapshot(lost)
        self.assertEqual(operation.validate_feed_snapshot(lost,now=100.5)["action"],"STOP_REQUESTED")

    def test_missing_rollback_gap_fork_wrong_identity_stale_and_closed_append_reject(self):
        base=operation.synthetic_feed_snapshot(now=100.0,count=3)
        cases=[]
        changed=copy.deepcopy(base);changed["receipts"].pop(1);cases.append(changed)
        changed=copy.deepcopy(base);changed["head"]["seq"]=0;cases.append(changed)
        changed=copy.deepcopy(base);changed["receipts"][2]["value"]["seq"]=4;cases.append(changed)
        changed=copy.deepcopy(base);changed["receipts"].append(copy.deepcopy(changed["receipts"][2]));cases.append(changed)
        changed=copy.deepcopy(base);changed["receipts"][1]["value"]["g_job_identity"]["job_id_hash"]="0"*64;cases.append(changed)
        changed=copy.deepcopy(base);changed["receipts"][1]["value"]["publisher_process"]["start_ticks"]="wrong";cases.append(changed)
        for changed in cases:
            with self.assertRaises(ValueError):operation.validate_feed_snapshot(changed,now=100.5)
        with self.assertRaises(ValueError):operation.validate_feed_snapshot(base,now=200.0)
        closed=copy.deepcopy(base);closed["receipts"][1]["value"]["status"]="CLOSED";operation.rehash_snapshot(closed)
        with self.assertRaises(ValueError):operation.validate_feed_snapshot(closed,now=100.5)


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class ExecutableOwnerAndLifecycleTests(FeedFixtureMixin, unittest.TestCase):
    def pilot_inputs(self):
        measurements = {
            arm: {"history_source": arm, "optimizer_steps": 1, "measurements": {name: {"seconds": 0.0, "bytes": 0} for name in operation.science.PILOT_MEASUREMENTS}}
            for arm in ("like", "observed")
        }
        return measurements, [{"query_id": 0, "history": [1]}]

    def test_module_help_and_all_eleven_commands_map_to_real_callables(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "vkgr.retrieval_sasrec_operation", "--help"],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout.lower())
        self.assertEqual(set(operation.COMMAND_DISPATCH), set(plan.COMMAND_STAGES))
        self.assertTrue(all(callable(handler) for handler in operation.COMMAND_DISPATCH.values()))
        self.assertEqual(len({id(handler) for handler in operation.COMMAND_DISPATCH.values()}), 11)

    def test_cli_select_calls_handler_and_writes_hash_bound_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            value = operation.synthetic_operation_manifest("select", workspace=root)
            path = root / "select.json"
            sha = operation.write_immutable_json(path, value)
            expected = {"eligible": False, "route_status": "NOT_ELIGIBLE", "failed_slots": ["P4H-S01"]}
            with patch.object(operation.science, "select_route", return_value=expected) as handler:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(operation.main(["select", "--operation-manifest", str(path), "--operation-manifest-sha256", sha]), 0)
            handler.assert_called_once()
            receipt = json.loads(stdout.getvalue())
            artifact = Path(receipt["artifact_path"])
            self.assertEqual(artifact.read_bytes(), operation.canonical(expected))
            self.assertEqual(receipt["artifact_sha256"], operation._file_sha256(artifact))
            self.assertFalse(Path(value["exclusive_control_root"]).exists())

    def test_cli_reads_sha_bound_typed_input_and_calls_real_science_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            records = [{"slot_id": slot["slot_id"], "config": slot, "status": "failed+spent", "spent": {"wall_seconds": 1}, "failure_reason": "synthetic"} for slot in plan.FORMAL_SLOTS]
            input_path = root / "records.json"
            input_sha = operation.write_immutable_json(input_path, records)
            value = operation.synthetic_operation_manifest(
                "completion-verify",
                workspace=root,
                typed_inputs={"args": [{"path": str(input_path), "sha256": input_sha}], "kwargs": {}},
            )
            manifest_path = root / "completion.json"
            manifest_sha = operation.write_immutable_json(manifest_path, value)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(operation.main(["completion-verify", "--operation-manifest", str(manifest_path), "--operation-manifest-sha256", manifest_sha]), 0)
            receipt = json.loads(stdout.getvalue())
            result = json.loads(Path(receipt["artifact_path"]).read_bytes())
            self.assertEqual(result["registered_identities"], 12)
            self.assertEqual(result["route_status"], "NOT_ELIGIBLE")
            self.assertEqual(receipt["artifact_sha256"], operation._file_sha256(receipt["artifact_path"]))

    def test_verifier_manifest_requires_real_closed_predecessor_and_cross_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with self.assertRaisesRegex(ValueError, "closed predecessor"):
                operation.synthetic_operation_manifest("pilot-verify", workspace=root)
            with self.assertRaisesRegex(ValueError, "closed predecessor"):
                operation.synthetic_operation_manifest("pilot-verify", workspace=root, predecessor=plan.synthetic_verifier_manifest("pilot-verify")["sealed"])
            mutating = operation.synthetic_operation_manifest("pilot", workspace=root)
            result = operation.execute_typed_operation(mutating, *self.pilot_inputs(), sink=lambda *_: None)
            verifier = operation.synthetic_operation_manifest("pilot-verify", workspace=root, predecessor=result["sealed"])
            changed = copy.deepcopy(verifier)
            changed["verifier_contract"]["predecessor_operation_manifest_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                operation.execute_typed_operation(changed, result["typed_result"])
            opened = copy.deepcopy(result["sealed"])
            opened["state"] = "OPEN"
            with self.assertRaisesRegex(ValueError, "closed predecessor"):
                operation.synthetic_operation_manifest("pilot-verify", workspace=root, predecessor=opened)

    def test_local_aggregates_write_only_local_result_without_remote_evidence(self):
        for command in ("select", "completion-verify", "summarize"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                value = operation.synthetic_operation_manifest(command, workspace=Path(tmp).resolve())
                expected = {"command": command, "local": True}
                with patch.object(operation.science, operation.COMMAND_HANDLERS[command].__name__, return_value=expected):
                    result = operation.execute_typed_operation(value)
                self.assertEqual(result["typed_result"], expected)
                self.assertEqual(Path(result["artifact_path"]).read_bytes(), operation.canonical(expected))
                for field in ("controller_identity", "admission_sha256", "g_receipt_sha256", "worker_pgid", "exclusive_control_root"):
                    self.assertNotIn(field, result)
                self.assertFalse(Path(value["exclusive_control_root"]).exists())

    def test_remote_mutating_and_verifier_execute_live_lifecycle_and_sealed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mutating = operation.synthetic_operation_manifest("pilot", workspace=root)
            with patch.object(operation.LiveFeedController, "observe", autospec=True, wraps=operation.LiveFeedController.observe) as observe:
                mutating_result = operation.execute_typed_operation(mutating, *self.pilot_inputs(), sink=lambda *_: None)
            self.assertGreaterEqual(observe.call_count, 3)
            self.assertEqual(mutating_result["state"], "CLOSED")
            self.assertEqual(mutating_result["terminal_status"], "completed+unverified")
            self.assertFalse(operation._group_alive(mutating_result["worker_pgid"]))
            for row in mutating_result["sealed"]["artifact_hashes"]:
                artifact = Path(row["path"])
                self.assertTrue(artifact.is_file())
                self.assertEqual(row["size"], artifact.stat().st_size)
                self.assertEqual(row["sha256"], operation._file_sha256(artifact))
            opened_artifact = next(row for row in mutating_result["sealed"]["artifact_hashes"] if row["role"] == "pilot-opened-artifact-ledger")
            opened_value = json.loads(Path(opened_artifact["path"]).read_bytes())
            self.assertEqual(opened_value["opened_roles"], [])
            self.assertEqual(opened_value["opened_sha256"], {})
            verifier = operation.synthetic_operation_manifest("pilot-verify", workspace=root, predecessor=mutating_result["sealed"])
            verified = operation.execute_typed_operation(verifier, mutating_result["typed_result"])
            completed = operation.complete_verified(verifier, verified["verifier_sealed_path"], verified["verifier_sealed_sha256"])
            self.assertEqual(completed["status"], "completed+verified")
            self.assertNotEqual(mutating_result["worker_pgid"], verified["worker_pgid"])
            self.assertTrue(verified["predecessor_read_only"])

    def test_hash_consistent_wrong_artifact_is_rejected_before_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mutating = operation.synthetic_operation_manifest("pilot", workspace=root)
            result = operation.execute_typed_operation(mutating, *self.pilot_inputs(), sink=lambda *_: None)
            sealed = copy.deepcopy(result["sealed"])
            row = next(item for item in sealed["artifact_hashes"] if item["role"] == "pilot-capacity-measurement-index")
            artifact = Path(row["path"])
            artifact.unlink()
            wrong = {"schema_version": 1, "operation_id": mutating["operation_id"], "role": row["role"], "result": result["typed_result"]}
            row["sha256"] = operation.write_immutable_json(artifact, wrong)
            row["size"] = artifact.stat().st_size
            sealed["complete_artifact_set_sha256"] = hashlib.sha256(plan.canonical_bytes(sealed["artifact_hashes"])).hexdigest()
            wrapper = Path(result["exclusive_output_root"]) / "sealed-predecessor.json"
            wrapper.unlink()
            operation.write_immutable_json(wrapper, sealed)
            with self.assertRaises(ValueError):
                operation.synthetic_operation_manifest("pilot-verify", workspace=root / "verify", predecessor=sealed)

    def test_resigned_wrapper_cannot_hide_tampered_operation_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mutating = operation.synthetic_operation_manifest("pilot", workspace=root)
            result = operation.execute_typed_operation(mutating, *self.pilot_inputs(), sink=lambda *_: None)
            sealed = copy.deepcopy(result["sealed"])
            sealed["admission_sha256"] = "0" * 64
            wrapper = Path(result["exclusive_output_root"]) / "sealed-predecessor.json"
            wrapper.unlink()
            operation.write_immutable_json(wrapper, sealed)
            with self.assertRaises(ValueError):
                operation.synthetic_operation_manifest("pilot-verify", workspace=root / "verify", predecessor=sealed)

    def test_handwritten_closed_summary_cannot_complete_without_verifier_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mutating = operation.synthetic_operation_manifest("pilot", workspace=root)
            result = operation.execute_typed_operation(mutating, *self.pilot_inputs(), sink=lambda *_: None)
            verifier = operation.synthetic_operation_manifest("pilot-verify", workspace=root / "verify", predecessor=result["sealed"])
            fake = {"state": "CLOSED", "terminal_status": "completed+verified", "operation_manifest_sha256": operation.digest(verifier)}
            with self.assertRaises((TypeError, ValueError)):
                operation.complete_verified(verifier, fake)

    def test_each_pilot_artifact_rejects_schema_identity_and_semantic_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mutating = operation.synthetic_operation_manifest("pilot", workspace=root)
            result = operation.execute_typed_operation(mutating, *self.pilot_inputs(), sink=lambda *_: None)
            original_sealed = copy.deepcopy(result["sealed"])
            wrapper = Path(result["sealed_path"])
            semantic_mutators = {
                "pilot-capacity-measurement-index": lambda payload: payload.update(units={"seconds": "bytes", "bytes": "seconds"}),
                "pilot-resource-ledger": lambda payload: payload.update(spent={"wall_seconds": 999, "new_bytes": 0}),
                "pilot-stage-event-ledger": lambda payload: payload.update(events=[]),
                "pilot-opened-artifact-ledger": lambda payload: payload.update(opened_roles=["never-opened"], opened_sha256={"never-opened": "0" * 64}),
                "pilot-artifact-index": lambda payload: payload.update(artifacts=[], complete_artifact_set_sha256=operation.digest([])),
            }
            for role in semantic_mutators:
                for mode in ("schema", "identity", "semantic"):
                    with self.subTest(role=role, mode=mode):
                        sealed = copy.deepcopy(original_sealed)
                        row = next(item for item in sealed["artifact_hashes"] if item["role"] == role)
                        path = Path(row["path"])
                        payload = json.loads(path.read_bytes())
                        path.unlink()
                        if mode == "schema":
                            payload["extra"] = True
                        elif mode == "identity":
                            payload["operation_id"] = "wrong-operation"
                        else:
                            semantic_mutators[role](payload)
                        row["sha256"] = operation.write_immutable_json(path, payload)
                        row["size"] = path.stat().st_size
                        sealed["complete_artifact_set_sha256"] = operation._artifact_set_sha(sealed["artifact_hashes"])
                        wrapper.unlink()
                        operation.write_immutable_json(wrapper, sealed)
                        with self.assertRaises(ValueError):
                            operation.synthetic_operation_manifest("pilot-verify", workspace=root / f"verify-{role}-{mode}", predecessor=sealed)
                        path.unlink()
                        original_row = next(item for item in original_sealed["artifact_hashes"] if item["role"] == role)
                        # Restore exact immutable bytes from the constructor, not the tampered wrapper.
                        if role == "pilot-artifact-index":
                            others = [item for item in original_sealed["artifact_hashes"] if item["role"] != role]
                            restored = {"schema_version": 1, "operation_id": mutating["operation_id"], "artifacts": others, "complete_artifact_set_sha256": operation._artifact_set_sha(others)}
                        else:
                            restored = operation._artifact_payload(role, mutating, result["typed_result"], spent=original_sealed["spent"], events=json.loads((Path(result["exclusive_output_root"]) / "stage-events.json").read_bytes())["events"], opened=json.loads((Path(result["exclusive_output_root"]) / "opened-artifacts.json").read_bytes())["opened"])
                        restored_sha = operation.write_immutable_json(path, restored)
                        self.assertEqual(restored_sha, original_row["sha256"])
                        wrapper.unlink()
                        operation.write_immutable_json(wrapper, original_sealed)

    def test_formal_and_route_artifacts_have_exact_schemas_and_reject_tamper(self):
        pin = "a" * 64
        cases = {
            "formal-train": {
                "verifier": "formal-verify",
                "typed": {"public": {"checkpoint_roles": {"best": pin, "last": pin}, "evaluations": [{"prediction_sha256": pin}], "selection": {"best": 1}, "step_records": [{"draw_sha256": pin}]}},
                "events": ["controller_start", "admission", "optimizer_step", "epoch_validation", "checkpoint", "stop", "seal"],
            },
            "infer-route": {
                "verifier": "infer-route-verify",
                "typed": {"query_ids": [1], "shards": [{"index": 0, "records": [{"query_id": 1, "candidates": list(range(80)), "fallback": False}]}]},
                "events": ["controller_start", "admission", "route_shard", "stop", "seal"],
            },
        }
        semantic_fields = {
            "formal-checkpoint-index": ("best", "bad"),
            "formal-epoch-metric-index": ("selection_replay_sha256", "bad"),
            "formal-prediction-hash-index": ("epoch_hashes", ["bad"]),
            "formal-draw-mask-provenance": ("aggregate_sha256", "bad"),
            "formal-resource-ledger": ("spent", {"wall_seconds": 9, "new_bytes": 0}),
            "formal-stage-event-ledger": ("events", []),
            "formal-opened-artifact-ledger": ("opened_roles", []),
            "formal-artifact-index": ("artifacts", []),
            "route-shard-index": ("query_coverage_sha256", "bad"),
            "route-candidate-hash-index": ("aggregate_sha256", "bad"),
            "route-resource-ledger": ("spent", {"wall_seconds": 9, "new_bytes": 0}),
            "route-stage-event-ledger": ("events", []),
            "route-opened-artifact-ledger": ("opened_roles", []),
            "route-artifact-index": ("artifacts", []),
        }
        for stage, case in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp).resolve()
                value = {"stage": stage, "operation_id": f"{stage}-001", "exclusive_output_root": str(output), "caps": {"wall_seconds": 5, "new_bytes": 100, "seal_seconds": 2, "signal_grace_seconds": 1}}
                spent = {"wall_seconds": 1, "new_bytes": 0}
                events = [{"sequence": index, "kind": kind} for index, kind in enumerate(case["events"])]
                opened = [{"role": "input", "path": "/synthetic/input.json", "sha256": pin, "size": 1, "schema_id": "synthetic-v1"}]
                rows = operation._write_artifact_set(value, case["typed"], spent=spent, events=events, opened=opened)
                sealed = {"predecessor_operation_id": value["operation_id"], "predecessor_stage": stage, "spent": spent, "artifact_hashes": rows}
                evidence = {"events": events, "opened": opened, "caps_spent": {"caps": value["caps"], "spent": spent}}
                for row in rows:
                    role = row["role"]
                    payload = json.loads(Path(row["path"]).read_bytes())
                    operation._validate_artifact_payload(role, payload, sealed, evidence)
                    for mode in ("schema", "identity", "semantic"):
                        with self.subTest(stage=stage, role=role, mode=mode):
                            changed = copy.deepcopy(payload)
                            if mode == "schema":
                                changed["extra"] = True
                            elif mode == "identity":
                                changed["operation_id"] = "wrong"
                            else:
                                field, replacement = semantic_fields[role]
                                changed[field] = replacement
                                if role.endswith("opened-artifact-ledger"):
                                    changed["opened_sha256"] = {}
                                if role.endswith("artifact-index"):
                                    changed["complete_artifact_set_sha256"] = operation.digest([])
                            with self.assertRaises(ValueError):
                                operation._validate_artifact_payload(role, changed, sealed, evidence)

    def test_tampered_admission_final_g_worker_and_spent_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mutating = operation.synthetic_operation_manifest("pilot", workspace=root)
            result = operation.execute_typed_operation(mutating, *self.pilot_inputs(), sink=lambda *_: None)
            wrapper = Path(result["sealed_path"])
            cases = {
                "admission_sha256": "0" * 64,
                "final_g_receipt_sha256": "1" * 64,
                "worker_birth_inventory_sha256": "2" * 64,
                "spent": {"wall_seconds": 999, "new_bytes": 0},
            }
            for field, replacement in cases.items():
                with self.subTest(field=field):
                    sealed = copy.deepcopy(result["sealed"])
                    sealed[field] = replacement
                    wrapper.unlink()
                    operation.write_immutable_json(wrapper, sealed)
                    with self.assertRaises(ValueError):
                        operation.synthetic_operation_manifest("pilot-verify", workspace=root / f"verify-{field}", predecessor=sealed)
                    wrapper.unlink()
                    operation.write_immutable_json(wrapper, result["sealed"])

    def test_complete_verified_rejects_cross_operation_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mutating = operation.synthetic_operation_manifest("pilot", workspace=root)
            result = operation.execute_typed_operation(mutating, *self.pilot_inputs(), sink=lambda *_: None)
            verifier = operation.synthetic_operation_manifest("pilot-verify", workspace=root / "verify", predecessor=result["sealed"])
            verified = operation.execute_typed_operation(verifier, result["typed_result"])
            other = copy.deepcopy(verifier)
            other["operation_id"] = "other-verifier-operation"
            with self.assertRaises(ValueError):
                operation.complete_verified(other, verified["verifier_sealed_path"], verified["verifier_sealed_sha256"])

    def test_v6_nested_artifacts_use_disk_bound_private_records_and_no_public_queries(self):
        pin = "a" * 64
        cases = (
            (
                "formal-train",
                {"public": {"checkpoint_roles": {"best": pin, "last": pin}, "evaluations": [{"epoch": 2, "step": 7, "prediction_sha256": pin}], "selection": {"best": 1}, "step_records": [{"draw_sha256": pin}]}},
                "formal-checkpoint-index",
            ),
            (
                "infer-route",
                {"query_ids": [1], "shards": [{"index": 0, "records": [{"query_id": 1, "candidates": list(range(80)), "fallback": False}]}]},
                "route-shard-index",
            ),
        )
        for stage, typed, role in cases:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp).resolve()
                value = {"stage": stage, "operation_id": f"{stage}-v6", "exclusive_output_root": str(output), "caps": {"wall_seconds": 5, "new_bytes": 100, "seal_seconds": 2, "signal_grace_seconds": 1}}
                spent = {"wall_seconds": 1, "new_bytes": 0}
                events = [{"sequence": 0, "kind": "optimizer" if stage == "formal-train" else "infer_shard"}]
                rows = operation._write_artifact_set(value, typed, spent=spent, events=events, opened=[])
                row = next(item for item in rows if item["role"] == role)
                payload = json.loads(Path(row["path"]).read_bytes())
                nested = [item for item in (payload.get("best"), payload.get("last"), payload.get("preemption")) if item is not None] if stage == "formal-train" else payload["shards"]
                self.assertTrue(nested)
                for item in nested:
                    self.assertEqual(set(item), {"path", "sha256", "size", "checkpoint_role", "source_checkpoint_sha256", "epoch", "step", "operation_id"} if stage == "formal-train" else {"shard", "range", "path", "sha256", "size", "count"})
                    private = Path(item["path"])
                    self.assertTrue(private.is_file())
                    self.assertEqual(operation._file_sha256(private), item["sha256"])
                    self.assertEqual(private.stat().st_size, item["size"])
                self.assertNotIn("records", json.dumps(payload))

    def test_v6_cli_open_audit_and_runtime_transition_events_are_not_fixed_empty_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            measurements, sidecar = self.pilot_inputs()
            value = operation.synthetic_operation_manifest("pilot", workspace=root)
            audit = []
            science_payloads = [measurements, sidecar]
            for index, role in enumerate(value["roles"]):
                path = root / f"accepted-role-{index}.json"
                sha = operation.write_immutable_json(path, science_payloads[index] if index < 2 else {})
                role.update(path=str(path), sha256=sha)
                audit.append({"role": role["role"], "path": str(path), "sha256": sha, "schema_id": role["schema_id"]})
            value["typed_inputs"] = {"audit": audit, "args": [{"role": value["roles"][0]["role"]}, {"role": value["roles"][1]["role"]}], "kwargs": {}}
            manifest_path = root / "manifest.json"
            manifest_sha = operation.write_immutable_json(manifest_path, value)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(operation.main(["pilot", "--operation-manifest", str(manifest_path), "--operation-manifest-sha256", manifest_sha]), 0)
            result = json.loads(stdout.getvalue())
            output = Path(result["exclusive_output_root"])
            opened = json.loads((output / "opened-artifacts.json").read_bytes())
            expected_opened = {row["role"]: row["sha256"] for row in audit}
            self.assertEqual(opened["opened_roles"], sorted(expected_opened))
            self.assertEqual(opened["opened_sha256"], expected_opened)
            self.assertEqual([row["role"] for row in opened["opened"]], [row["role"] for row in audit])
            self.assertTrue(all(set(row) == {"role", "path", "sha256", "size", "schema_id"} for row in opened["opened"]))
            events = json.loads((output / "stage-events.json").read_bytes())["events"]
            self.assertEqual([row["sequence"] for row in events], list(range(len(events))))
            self.assertTrue(all("observed_at" in row and "state" in row for row in events))

    def test_v6_terminal_map_rejects_partial_or_ordinarily_resigned_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            value = operation.synthetic_operation_manifest("pilot", workspace=root)
            result = operation.execute_typed_operation(value, *self.pilot_inputs(), sink=lambda *_: None)
            output = Path(result["exclusive_output_root"])
            terminal = json.loads((output / "terminal.json").read_bytes())
            expected_roles = {"controller-start", "admission", "g-observations", "final-g-observation", "worker-birth-inventory", "worker-descendant-closure", "caps-spent", "stage-events", "opened-artifacts", "artifacts", "terminal-preimage", "closure-preimage"}
            self.assertEqual(set(terminal["complete_evidence_sha256"]), expected_roles)
            admission_path = output / "admission.json"
            admission = json.loads(admission_path.read_bytes())
            admission_path.unlink(); admission["expires_at"] = 0
            new_sha = operation.write_immutable_json(admission_path, admission)
            sealed = copy.deepcopy(result["sealed"]); sealed["admission_sha256"] = new_sha
            wrapper = Path(result["sealed_path"]); wrapper.unlink(); operation.write_immutable_json(wrapper, sealed)
            with self.assertRaises(ValueError):
                operation.synthetic_operation_manifest("pilot-verify", workspace=root / "verify", predecessor=sealed)

    def test_v6_verifier_has_complete_evidence_map_and_missing_member_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mutating = operation.synthetic_operation_manifest("pilot", workspace=root)
            result = operation.execute_typed_operation(mutating, *self.pilot_inputs(), sink=lambda *_: None)
            verifier = operation.synthetic_operation_manifest("pilot-verify", workspace=root / "verify", predecessor=result["sealed"])
            verified = operation.execute_typed_operation(verifier, result["typed_result"])
            wrapper = json.loads(Path(verified["verifier_sealed_path"]).read_bytes())
            self.assertEqual(set(wrapper["complete_evidence_sha256"]), {"controller-start", "admission", "g-observations", "final-g-observation", "worker-birth-inventory", "worker-descendant-closure", "caps-spent", "stage-events", "opened-artifacts", "artifacts", "terminal-preimage", "closure-preimage"})
            missing = Path(verified["exclusive_output_root"]) / "g-observations.json"
            missing.rename(missing.with_suffix(".missing"))
            with self.assertRaises(ValueError):
                operation.complete_verified(verifier, verified["verifier_sealed_path"], verified["verifier_sealed_sha256"])

    def test_v7_checkpoint_private_payload_rejects_hash_consistent_nested_tamper(self):
        pin = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            value = {"stage": "formal-train", "operation_id": "formal-v7", "exclusive_output_root": str(Path(tmp).resolve()), "caps": {"wall_seconds": 5, "new_bytes": 100, "seal_seconds": 2, "signal_grace_seconds": 1}}
            typed = {"public": {"checkpoint_roles": {"best": pin}, "evaluations": [{"epoch": 2, "step": 7, "prediction_sha256": pin}], "selection": {"best": 1}, "step_records": [{"draw_sha256": pin}]}}
            rows = operation._write_artifact_set(value, typed, spent={"wall_seconds": 1, "new_bytes": 0}, events=[], opened=[])
            payload = json.loads(Path(next(row for row in rows if row["role"] == "formal-checkpoint-index")["path"]).read_bytes())
            item = payload["best"]
            operation._validate_checkpoint_private(item, "best", value["operation_id"])
            private_path = Path(item["path"])
            original = json.loads(private_path.read_bytes())
            mutations = {
                "schema_version": 2,
                "operation_id": "wrong",
                "checkpoint_role": "last",
                "source_checkpoint_sha256": "b" * 64,
                "epoch": 3,
                "step": 8,
            }
            for field, replacement in mutations.items():
                with self.subTest(field=field):
                    changed = copy.deepcopy(original);changed[field] = replacement
                    private_path.unlink();item["sha256"] = operation.write_immutable_json(private_path, changed);item["size"] = private_path.stat().st_size
                    with self.assertRaises(ValueError):
                        operation._validate_checkpoint_private(item, "best", value["operation_id"])
            if private_path.exists():private_path.unlink()
            item["sha256"] = operation.write_immutable_json(private_path, original);item["size"] = private_path.stat().st_size

    def test_v7_route_private_payload_rejects_range_count_order_top80_and_catalog_tamper(self):
        catalog = list(range(100, 181))
        records = [
            {"query_id": 7, "candidates": catalog[:80], "fallback": False},
            {"query_id": 9, "candidates": list(reversed(catalog[:80])), "fallback": True},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            value = {"stage": "infer-route", "operation_id": "route-v7", "exclusive_output_root": str(Path(tmp).resolve()), "caps": {"wall_seconds": 5, "new_bytes": 100, "seal_seconds": 2, "signal_grace_seconds": 1}}
            typed = {"query_ids": [7, 9], "shards": [{"index": 0, "identity_sha256": "a" * 64, "start": 0, "end": 2, "records": records}]}
            rows = operation._write_artifact_set(value, typed, spent={"wall_seconds": 1, "new_bytes": 0}, events=[], opened=[])
            payload = json.loads(Path(next(row for row in rows if row["role"] == "route-shard-index")["path"]).read_bytes())
            item = payload["shards"][0]
            operation._validate_route_private(item, value["operation_id"], 0, [7, 9], set(catalog))
            private_path = Path(item["path"]);original = json.loads(private_path.read_bytes())
            mutators = {
                "schema": lambda row: row.update(schema_version=2),
                "operation": lambda row: row.update(operation_id="wrong"),
                "shard": lambda row: row.update(shard=1),
                "range": lambda row: row.update(start=1),
                "count": lambda row: row["records"].pop(),
                "order": lambda row: row["records"].reverse(),
                "top80": lambda row: row["records"][0]["candidates"].pop(),
                "duplicate": lambda row: row["records"][0]["candidates"].__setitem__(1, row["records"][0]["candidates"][0]),
                "catalog": lambda row: row["records"][0]["candidates"].__setitem__(0, 999),
                "record_schema": lambda row: row["records"][0].update(score=1.0),
            }
            for name, mutate in mutators.items():
                with self.subTest(name=name):
                    changed=copy.deepcopy(original);mutate(changed)
                    private_path.unlink();item["sha256"]=operation.write_immutable_json(private_path,changed);item["size"]=private_path.stat().st_size
                    with self.assertRaises(ValueError):operation._validate_route_private(item,value["operation_id"],0,[7,9],set(catalog))
            if private_path.exists():private_path.unlink()
            item["sha256"]=operation.write_immutable_json(private_path,original);item["size"]=private_path.stat().st_size

    def test_v7_opened_roles_are_exact_bijection_with_accepted_inputs(self):
        roles = [
            {"role": "catalog", "path": "/inputs/catalog.json", "sha256": "a" * 64, "schema_id": "catalog-v1"},
            {"role": "queries", "path": "/inputs/queries.json", "sha256": "b" * 64, "schema_id": "queries-v1"},
        ]
        opened = [
            {**row, "size": index + 10}
            for index, row in enumerate(roles)
        ]
        operation._validate_opened_role_bijection(roles, opened)
        for changed in (
            opened[:-1], opened + [copy.deepcopy(opened[0])], opened + [{"role": "extra", "path": "/x", "sha256": "c" * 64, "schema_id": "x", "size": 1}],
            [{**opened[0], "role": "queries"}, opened[1]], [{**opened[0], "path": "/wrong"}, opened[1]],
        ):
            with self.assertRaises(ValueError):operation._validate_opened_role_bijection(roles, changed)

    def test_v7_operation_sink_rejects_missing_fake_and_cross_operation_events(self):
        operation_id = "pilot-v7-events"
        base = [
            {"sequence": 0, "kind": "controller_start", "state": "AWAITING_ADMISSION", "observed_at": 1.0, "operation_id": operation_id},
            {"sequence": 1, "kind": "admission", "state": "ADMITTED", "observed_at": 2.0, "operation_id": operation_id},
            {"sequence": 2, "kind": "pilot_summary", "state": "SUCCEEDED", "observed_at": 3.0, "operation_id": operation_id, "payload": {"query_count": 2, "query_identity_sha256": "a" * 64, "artifact_identity": "in_memory_sha256_only"}},
            {"sequence": 3, "kind": "stop", "state": "STOP_REQUESTED", "observed_at": 4.0, "operation_id": operation_id},
            {"sequence": 4, "kind": "seal", "state": "CLOSED", "observed_at": 5.0, "operation_id": operation_id},
        ]
        operation._validate_science_event_audit("pilot", operation_id, base)
        cases = []
        missing = copy.deepcopy(base);missing.pop(2);cases.append(missing)
        fake = copy.deepcopy(base);fake[2]["kind"] = "checkpoint";cases.append(fake)
        cross = copy.deepcopy(base);cross[2]["operation_id"] = "other-operation";cases.append(cross)
        duplicate = copy.deepcopy(base);duplicate.insert(3, copy.deepcopy(duplicate[2]));cases.append(duplicate)
        for changed in cases:
            with self.assertRaises(ValueError):operation._validate_science_event_audit("pilot", operation_id, changed)

    def test_operational_integrity_claim_explicitly_excludes_same_uid_total_rewrite(self):
        boundary = operation.OPERATIONAL_INTEGRITY_THREAT_MODEL
        self.assertEqual(set(boundary["protects"]), {"accidental-operation", "wrong-job", "duplicate-writer", "stale", "rollback", "fork", "partial-write", "ordinary-tamper"})
        self.assertEqual(boundary["out_of_scope"], "OUT_OF_SCOPE_REQUIRES_EXTERNAL_TRUST_ROOT:same-UID-malicious-self-consistent-total-rewrite")

    def test_cli_failure_is_nonzero_and_does_not_write_success_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            value = operation.synthetic_operation_manifest("select", workspace=root)
            path = root / "select.json"
            sha = operation.write_immutable_json(path, value)
            with patch.object(operation.science, "select_route", side_effect=ValueError("synthetic failure")):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertNotEqual(operation.main(["select", "--operation-manifest", str(path), "--operation-manifest-sha256", sha]), 0)
            self.assertFalse((Path(value["exclusive_output_root"]) / "result.json").exists())


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class RouteArtifactClosureV8Tests(unittest.TestCase):
    def closure(self, root):
        operation_id = "route-v8-closure"
        records = [
            [{"query_id": 7, "candidates": list(range(100, 180)), "fallback": False}],
            [{"query_id": 9, "candidates": list(range(200, 280)), "fallback": True}],
        ]
        typed = {
            "query_ids": [7, 9],
            "shards": [
                {"index": ordinal, "start": ordinal, "end": ordinal + 1, "records": shard_records}
                for ordinal, shard_records in enumerate(records)
            ],
        }
        value = {
            "stage": "infer-route",
            "operation_id": operation_id,
            "exclusive_output_root": str(root),
            "caps": {"wall_seconds": 5, "new_bytes": 100, "seal_seconds": 2, "signal_grace_seconds": 1},
        }
        private_entries = operation._route_shard_entries(value, typed["shards"])
        events = []
        for ordinal, (entry, shard_records) in enumerate(zip(private_entries, records)):
            events.append({
                "sequence": ordinal + 2,
                "kind": "route_shard",
                "state": "SUCCEEDED",
                "observed_at": float(ordinal + 3),
                "operation_id": operation_id,
                "payload": {
                    "shard": ordinal,
                    "range": [ordinal, ordinal + 1],
                    "record_count": 1,
                    "records_sha256": operation.digest(shard_records),
                    "route_identity_sha256": "a" * 64,
                    "artifact_identity": "private_path_sha256_size",
                    **entry,
                },
            })
        spent = {"wall_seconds": 1, "new_bytes": 0}
        rows = operation._write_artifact_set(value, typed, spent=spent, events=events, opened=[])
        sealed = {
            "predecessor_operation_id": operation_id,
            "predecessor_stage": "infer-route",
            "spent": spent,
            "artifact_hashes": rows,
        }
        payloads = {row["role"]: json.loads(Path(row["path"]).read_bytes()) for row in rows}
        evidence = {"events": events, "opened": [], "caps_spent": {"caps": value["caps"], "spent": spent}}
        return rows, payloads, sealed, evidence

    def resign_public_artifact(self, rows, payloads, role):
        row = next(item for item in rows if item["role"] == role)
        path = Path(row["path"])
        path.unlink()
        row["sha256"] = operation.write_immutable_json(path, payloads[role])
        row["size"] = path.stat().st_size
        artifact_role = "route-artifact-index"
        artifact_row = next(item for item in rows if item["role"] == artifact_role)
        others = [item for item in rows if item["role"] != artifact_role]
        payloads[artifact_role] = {
            "schema_version": 1,
            "operation_id": artifact_row["role"].replace("-artifact-index", "-v8-closure"),
            "artifacts": copy.deepcopy(others),
            "complete_artifact_set_sha256": operation._artifact_set_sha(others),
        }
        artifact_path = Path(artifact_row["path"])
        artifact_path.unlink()
        artifact_row["sha256"] = operation.write_immutable_json(artifact_path, payloads[artifact_role])
        artifact_row["size"] = artifact_path.stat().st_size

    def test_happy_path_has_one_bidirectional_route_authority_per_ordinal(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, payloads, sealed, evidence = self.closure(Path(tmp).resolve())
            operation._validate_route_artifact_closure(payloads, sealed, evidence)

    def test_reversed_private_candidates_with_self_consistent_outer_hashes_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, payloads, sealed, evidence = self.closure(Path(tmp).resolve())
            shard = payloads["route-shard-index"]["shards"][0]
            private_path = Path(shard["path"])
            private = json.loads(private_path.read_bytes())
            private["records"][0]["candidates"].reverse()
            private_path.unlink()
            shard["sha256"] = operation.write_immutable_json(private_path, private)
            shard["size"] = private_path.stat().st_size
            self.resign_public_artifact(rows, payloads, "route-shard-index")
            for row in rows:
                operation._validate_artifact_payload(row["role"], payloads[row["role"]], sealed, evidence)
            with self.assertRaises(ValueError):
                operation._validate_route_artifact_closure(payloads, sealed, evidence)

    def test_event_range_count_and_path_drift_reject(self):
        for field, replacement in (("range", [1, 2]), ("count", 2), ("path", "/wrong/private.json")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                _, payloads, sealed, evidence = self.closure(Path(tmp).resolve())
                evidence["events"][0]["payload"][field] = replacement
                with self.assertRaises(ValueError):
                    operation._validate_route_artifact_closure(payloads, sealed, evidence)

    def test_candidate_hash_swap_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, payloads, sealed, evidence = self.closure(Path(tmp).resolve())
            hashes = payloads["route-candidate-hash-index"]["per_shard_hashes"]
            hashes.reverse()
            payloads["route-candidate-hash-index"]["aggregate_sha256"] = operation.digest(hashes)
            with self.assertRaises(ValueError):
                operation._validate_route_artifact_closure(payloads, sealed, evidence)

    def test_missing_duplicate_or_out_of_order_ordinal_rejects(self):
        mutators = (
            lambda payloads, evidence: evidence["events"].pop(),
            lambda payloads, evidence: evidence["events"].append(copy.deepcopy(evidence["events"][0])),
            lambda payloads, evidence: payloads["route-shard-index"]["shards"].pop(),
            lambda payloads, evidence: payloads["route-shard-index"]["shards"].__setitem__(1, copy.deepcopy(payloads["route-shard-index"]["shards"][0])),
            lambda payloads, evidence: payloads["route-candidate-hash-index"]["per_shard_hashes"].pop(),
            lambda payloads, evidence: payloads["route-candidate-hash-index"]["per_shard_hashes"].append(payloads["route-candidate-hash-index"]["per_shard_hashes"][0]),
            lambda payloads, evidence: evidence["events"].reverse(),
        )
        for index, mutate in enumerate(mutators):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as tmp:
                _, payloads, sealed, evidence = self.closure(Path(tmp).resolve())
                mutate(payloads, evidence)
                with self.assertRaises(ValueError):
                    operation._validate_route_artifact_closure(payloads, sealed, evidence)

    def test_cross_operation_route_event_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, payloads, sealed, evidence = self.closure(Path(tmp).resolve())
            evidence["events"][0]["operation_id"] = "other-operation"
            with self.assertRaises(ValueError):
                operation._validate_route_artifact_closure(payloads, sealed, evidence)


@unittest.skipIf(operation is None, "P4H-03B not implemented")
class LiveGFeedAdmissionTests(FeedFixtureMixin, unittest.TestCase):
    def test_admission_reads_descriptor_directory_lock_process_inventory_and_persists_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with operation.synthetic_live_feed(root / "feed", now=time.time()) as publisher:
                state_path = root / "control" / "accepted-head.json"
                controller = operation.LiveFeedController(publisher.descriptor_path, publisher.descriptor_sha256, state_path)
                first = controller.observe(now=time.time())
                self.assertEqual(first["head_seq"], 0)
                publisher.publish(now=time.time())
                second = controller.observe(now=time.time())
                self.assertEqual(second["head_seq"], 1)
                self.assertTrue(state_path.is_file())
                publisher.set_head(0)
                with self.assertRaisesRegex(ValueError, "rollback"):
                    controller.observe(now=time.time())

    def test_admission_rejects_unknown_writer_and_wrong_live_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with operation.synthetic_live_feed(root / "feed", now=time.time()) as publisher:
                controller = operation.LiveFeedController(publisher.descriptor_path, publisher.descriptor_sha256, root / "state.json")
                writer = subprocess.Popen([sys.executable, "-c", "import pathlib,sys,time; f=pathlib.Path(sys.argv[1]).open('r+b'); time.sleep(30)", str(publisher.head_path)])
                try:
                    time.sleep(0.1)
                    with self.assertRaisesRegex(ValueError, "unknown writer"):
                        controller.observe(now=time.time())
                finally:
                    writer.terminate();writer.wait(timeout=3)
                descriptor = json.loads(publisher.descriptor_path.read_text())
                descriptor["publisher_process"]["start_ticks"] = "wrong"
                changed = root / "wrong-descriptor.json"
                changed_sha = operation.write_immutable_json(changed, descriptor)
                with self.assertRaises(ValueError):
                    operation.LiveFeedController(changed, changed_sha, root / "other-state.json").observe(now=time.time())

    def test_admission_and_atomic_stage_boundaries_use_live_observation_not_snapshot_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with operation.synthetic_live_feed(root / "feed", now=time.time()) as publisher:
                value = operation.synthetic_operation_manifest("pilot", workspace=root, feed_descriptor_path=publisher.descriptor_path, feed_descriptor_sha256=publisher.descriptor_sha256)
                start = operation.prepare_controller(value, operation.current_process_identity(), sink=lambda *_: None)
                controller = operation.LiveFeedController(publisher.descriptor_path, publisher.descriptor_sha256, root / "accepted.json")
                observed = controller.observe(now=time.time())
                admission = operation.synthetic_admission(start, observed["receipt_sha256"], issued_at=time.time())
                accepted = operation.admit_controller(start, admission, controller, now=time.time(), sink=lambda *_: None)
                calls = []
                operation.run_atomic_stage(controller, "synthetic-stage", lambda: calls.append("body"), now=lambda: time.time())
                self.assertEqual(calls, ["body"])
                with self.assertRaises((TypeError, ValueError)):
                    operation.admit_controller(start, admission, operation.synthetic_feed_snapshot(time.time()), now=time.time(), sink=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
