import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p_watch_positive_deploy as deploy_module
from p_watch_positive_deploy import deploy, inventory


class WatchPositiveDeployLayoutTests(unittest.TestCase):
    def test_real_helper_deploy_preserves_git_bytes_and_overlays_only_fixed_p3c_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source"; bare = root / "remote.git"
            deploy_path = root / "vkgr-p-watch-positive-train-v4"
            ops = root / "vkgr-p-watch-positive-train-ops-v4"
            scheduler = root / "scheduler"
            for state in ("pending", "claimed", "rejected"):
                (scheduler / "queue" / state).mkdir(parents=True)
            files = (
                "src/vkgr/p_watch_positive.py", "scripts/p_watch_positive_stats.py", "src/vkgr/retrieval_data.py",
                "scripts/p3_label_eda.py", "configs/p_watch_positive/p-watch-positive-train-only.json",
                "docs/p3c_validation.md",
            )
            contract_source = root / "private-p3c.md"
            contract_source.write_bytes(b"synthetic p3c contract fixture\n")
            sanitized_sha = hashlib.sha256(b"synthetic sanitized git fixture\n").hexdigest()
            contract_sha = hashlib.sha256(contract_source.read_bytes()).hexdigest()
            deploy_module.P3C = {"path": "docs/p3c_validation.md", "git_sha256": sanitized_sha,
                                 "contract_sha256": contract_sha, "source_path": str(contract_source)}
            source.mkdir(); subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
            pins = {}
            for relative in files:
                path = source / relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic sanitized git fixture\n" if relative == "docs/p3c_validation.md" else relative.encode())
                pins[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            subprocess.run(["git", "-C", str(source), "add", *files], check=True)
            subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
            subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
            subprocess.run(["git", "-C", str(source), "branch", "-M", "codex/statsfixture"], check=True)
            subprocess.run(["git", "-C", str(source), "remote", "add", "origin", str(bare)], check=True)
            subprocess.run(["git", "-C", str(source), "push", "-q", "origin", "codex/statsfixture"], check=True)
            commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            approval_payload = {"schema": "p-watch-positive-stats-request-v1", "inputs_metadata": str(root / "inputs.json"),
                                "train_audit": str(root / "audit.json"),
                                "anchor_support_sha256": "b080240542ae85166186fb621f7b074f9728f8b0e7d01d2c69ae0d0bff6b53a2",
                                "quantile_method": "nearest_rank", "source_pins": pins}
            pinned = {"schema": "vkgr-pinned-v1", "base_commit": commit, "target_commit": commit, "files": pins, "overrides": {}}
            request = {"root": str(scheduler), "source": str(source), "destination": str(deploy_path), "ops": str(ops),
                       "branch": "codex/statsfixture", "commit": commit, "files": pins,
                       "run_ids": ["p-watch-positive-stats-v1"],
                       "p3c_data_channel": {**deploy_module.P3C,
                           "ops_saved_path": str(ops / "p3c-validation-git-sanitized-v1.md")},
                       "new_ops_documents": {"request.json": approval_payload, "pins.json": pinned}}
            before = inventory(request)
            self.assertTrue(before["clear"])
            dirty_destination = root / "vkgr-p-watch-positive-train-v4-dirty"
            dirty_ops = root / "vkgr-p-watch-positive-train-ops-v4-dirty"
            dirty_request = {**request, "destination": str(dirty_destination), "ops": str(dirty_ops),
                "p3c_data_channel": {**request["p3c_data_channel"],
                    "ops_saved_path": str(dirty_ops / "p3c-validation-git-sanitized-v1.md")}}
            dirty_before = inventory(dirty_request)
            real_run = subprocess.run
            def add_untracked_after_worktree(args, *call_args, **call_kwargs):
                result = real_run(args, *call_args, **call_kwargs)
                if args[:4] == ["git", "-C", str(source), "worktree"] and args[4] == "add":
                    (dirty_destination / "unexpected.txt").write_text("synthetic extra dirty path")
                return result
            with patch.object(deploy_module.subprocess, "run", side_effect=add_untracked_after_worktree):
                with self.assertRaisesRegex(ValueError, "deployment is dirty"):
                    deploy(dirty_request, dirty_before["request_sha256"])
            result = deploy(request, before["request_sha256"])
            self.assertEqual(result["status"], "deployed")
            self.assertTrue((ops / "deployment.json").is_file())
            self.assertFalse((deploy_path / "deployment.json").exists())
            self.assertEqual(hashlib.sha256((deploy_path / "docs/p3c_validation.md").read_bytes()).hexdigest(), contract_sha)
            self.assertEqual(hashlib.sha256((ops / "p3c-validation-git-sanitized-v1.md").read_bytes()).hexdigest(), sanitized_sha)
            status = subprocess.run(["git", "-C", str(deploy_path), "status", "--porcelain", "--untracked-files=all"],
                                    check=True, capture_output=True, text=True).stdout
            self.assertEqual(status, " M docs/p3c_validation.md\n")
            record = json.loads((ops / "deployment.json").read_text())
            self.assertEqual(record["commit"], commit)
            self.assertEqual(record["files"], pins)
            self.assertTrue(record["data_overlay"])
            self.assertEqual(record["p3c_data_channel"]["copy_sha256"], {
                "git_bytes_before_save": sanitized_sha, "ops_saved_after_copy": sanitized_sha,
                "source_before_copy": contract_sha, "source_after_copy": contract_sha,
                "deployment_after_copy": contract_sha})

    def test_p3c_mapping_rejects_wrong_path_and_fixed_hashes(self):
        request = {"ops": "/tmp/vkgr-p-watch-positive-train-ops-v4"}
        channel = {**deploy_module.P3C,
                   "ops_saved_path": request["ops"] + "/p3c-validation-git-sanitized-v1.md"}
        deploy_module._p3c({**request, "p3c_data_channel": channel})
        for field, value in (("path", "docs/other.md"), ("git_sha256", "0" * 64),
                             ("contract_sha256", "1" * 64), ("source_path", "/tmp/other.md")):
            with self.assertRaises(ValueError):
                deploy_module._p3c({**request, "p3c_data_channel": {**channel, field: value}})


if __name__ == "__main__":
    unittest.main()
