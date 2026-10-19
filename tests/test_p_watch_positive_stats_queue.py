import unittest
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p_watch_positive_stats_queue import (RUN_ID, inventory, validate_job, verify_fresh_deployment, file_sha256,
                                          closed_scan_inputs)


class WatchPositiveStatsQueueTests(unittest.TestCase):
    def paths(self):
        base = Path.cwd() / "synthetic-queue"
        root, ops, cwd = base / "scheduler", base / "vkgr-p-watch-positive-ops-v2", base / "vkgr-p-watch-positive-v2"
        return root, ops, cwd, ops / "request.json", base / "scheduler-code", base / "python"

    def make_job(self, root=None, ops=None, cwd=None, config=None):
        defaults = self.paths()
        root = Path(root) if root else defaults[0]
        ops = Path(ops) if ops else defaults[1]
        cwd = Path(cwd) if cwd else defaults[2]
        config = Path(config) if config else ops / "request.json"
        scheduler_code, python = defaults[4:]
        return {
            "run_id": RUN_ID, "schema": "vkgr-queue-job-v1", "kind": "ops",
            "cwd": str(cwd), "config": str(config),
            "argv": [str(python), "-B", str(cwd / "scripts/p_watch_positive_stats.py"), "--request", str(config)],
            "cpu_cores": 1, "ram_mib": 8192, "disk_gib": 1, "vram_mib": 0,
            "wall_hours": 1, "no_admission": False, "seed": 0,
            "inputs": [str(config), str(ops / "train-audit.json"), str(ops / "p-inputs.json")],
            "pinned_files": [], "pinned_manifest": str(ops / "pins.json"),
        }

    def test_fixed_job_schema_accepts_cpu_only_stats_request(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root, ops, cwd, config, scheduler_code, python = self.paths()
            ops = Path(temp) / "vkgr-p-watch-positive-ops-v2"; ops.mkdir()
            job = self.make_job(str(root), str(ops), str(cwd), str(ops / "request.json")); Path(job["config"]).write_text("{}")
            source = {"root": str(root), "ops": str(root / "g-ops"), "scheduler_code": str(scheduler_code)}
            with patch("p_watch_positive_stats_queue.expected_scan_inputs", return_value=job["inputs"]):
                self.assertIs(validate_job(job, source, str(ops), str(python)), job)

    def test_job_cannot_change_run_id_or_request_script(self):
        root, ops, cwd, config, scheduler_code, python = self.paths()
        source = {"root": str(root), "ops": str(root / "g-ops"), "scheduler_code": str(scheduler_code)}
        job = self.make_job(); job["run_id"] = "p-watch-positive-stats-retry"
        with self.assertRaises(ValueError):
            validate_job(job, source, str(ops), str(python))

    def test_job_rejects_reused_g_ops_namespace_and_undeclared_extra_input(self):
        root, ops, cwd, config, scheduler_code, python = self.paths()
        source = {"root": str(root), "ops": str(root / "g-ops"), "scheduler_code": str(scheduler_code)}
        job = self.make_job()
        old_ops = Path(source["ops"]); old_config = str(old_ops / "request.json")
        job["config"] = old_config; job["argv"][2] = str(cwd / "scripts/p_watch_positive_stats.py"); job["argv"][4] = old_config
        job["inputs"] = [old_config, str(old_ops / "train-audit.json"), str(old_ops / "p-inputs.json")]
        job["pinned_manifest"] = str(old_ops / "pins.json")
        with self.assertRaises(ValueError):
            validate_job(job, source, str(source["ops"]), str(python))
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            fresh_ops = Path(temp) / "vkgr-p-watch-positive-ops-v2"; fresh_ops.mkdir()
            job = self.make_job(str(root), str(fresh_ops), str(cwd), str(fresh_ops / "request.json"))
            Path(job["config"]).write_text("{}")
            job["inputs"].append(str(Path.cwd() / "synthetic-queue/undeclared-extra"))
            with patch("p_watch_positive_stats_queue.expected_scan_inputs", return_value=job["inputs"][:-1]):
                with self.assertRaises(ValueError):
                    validate_job(job, source, str(fresh_ops), str(python))
        job = self.make_job(); job["argv"][2] = str(cwd / "scripts/other.py")
        with self.assertRaises(ValueError):
            validate_job(job, source, str(ops), str(python))

    def test_scanner_input_closure_is_exact_train_week_zero_to_twenty_four(self):
        root = Path.cwd() / "synthetic-inputs"
        audit_path = root / "train-audit.json"
        config = root / "ops/request.json"
        request = {"inputs_metadata": str(root / "p-inputs.json"), "train_audit": str(audit_path)}
        audit = {"train_files": [{"path": f"subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet", "sha256": "a" * 64}
                                  for week in range(25)]}
        values = {"raw_root": str(root / "raw")}
        inputs = closed_scan_inputs(str(config), request, values, audit, root / "catalog.json")
        self.assertEqual(inputs, sorted(inputs))
        self.assertEqual(len(inputs), 30)
        self.assertIn(str(root / "raw/subsamples/ur0.01_ir0.01/train/week_00.parquet"), inputs)
        self.assertIn(str(root / "raw/subsamples/ur0.01_ir0.01/train/week_24.parquet"), inputs)
        self.assertNotIn(str(root / "raw/subsamples/ur0.01_ir0.01/train/week_25.parquet"), inputs)
        self.assertNotIn(str(root / "raw/subsamples/ur0.01_ir0.01/validation/week_25.parquet"), inputs)

    def test_inventory_reports_fixed_queue_without_creating_state(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "queue-root"; ops = Path(temp) / "ops"
            for name in ("pending", "claimed", "rejected"):
                (root / "queue" / name).mkdir(parents=True)
            ops.mkdir()
            value = inventory(root, ops, self.make_job(str(root), str(ops), str(Path.cwd() / "synthetic-checkout"), str(ops / "request.json")))
            self.assertFalse(value["run_exists"])
            self.assertFalse(value["intent_exists"])
            self.assertEqual(value["queue"], {"pending": [], "claimed": [], "rejected": [], "reasons": {}})

    def make_deployment(self, root):
        root = Path(root); deploy = root / "vkgr-p-watch-positive-v2"; ops = root / "vkgr-p-watch-positive-ops-v2"
        files = ("src/vkgr/p_watch_positive.py", "scripts/p_watch_positive_stats.py", "src/vkgr/retrieval_data.py",
                 "scripts/p3_label_eda.py", "configs/p_watch_positive/p-watch-positive-train-only.json")
        hashes = {}
        for relative in files:
            path = deploy / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(relative)
            hashes[relative] = file_sha256(path)
        subprocess.run(["git", "init", "-q", str(deploy)], check=True)
        subprocess.run(["git", "-C", str(deploy), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(deploy), "config", "user.name", "Test"], check=True)
        subprocess.run(["git", "-C", str(deploy), "add", *files], check=True)
        subprocess.run(["git", "-C", str(deploy), "commit", "-qm", "fixture"], check=True)
        commit = subprocess.run(["git", "-C", str(deploy), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        ops.mkdir()
        pinned_path = ops / "pins.json"
        pinned_path.write_text(json.dumps({"schema": "vkgr-pinned-v1", "base_commit": commit, "target_commit": commit,
                                           "files": hashes, "overrides": {}}))
        record = {"commit": commit, "files": hashes, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "data_overlay": False}
        (ops / "deployment.json").write_text(json.dumps(record))
        config = {"source_pins": hashes}
        job = {"cwd": str(deploy), "pinned_manifest": str(pinned_path)}
        return deploy, ops, job, config, record

    def test_fresh_deployment_binds_git_commit_and_complete_file_map(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            deploy, ops, job, config, record = self.make_deployment(temp)
            result = verify_fresh_deployment(deploy, ops, job, config)
            self.assertEqual(result["commit"], record["commit"])
            path = deploy / "src/vkgr/p_watch_positive.py"
            path.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "file hash"):
                verify_fresh_deployment(deploy, ops, job, config)

    def test_deployment_record_must_live_in_ops_and_forged_deploy_copy_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            deploy, ops, job, config, record = self.make_deployment(temp)
            (deploy / "deployment.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "deployment record"):
                verify_fresh_deployment(deploy, ops, job, config)


if __name__ == "__main__":
    unittest.main()
