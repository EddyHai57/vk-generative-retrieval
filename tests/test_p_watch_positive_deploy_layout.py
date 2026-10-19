import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p_watch_positive_deploy import deploy, inventory
from p_watch_positive_stats_queue import verify_fresh_deployment


class WatchPositiveDeployLayoutTests(unittest.TestCase):
    def test_real_helper_deploy_keeps_record_in_ops_and_checkout_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source"; bare = root / "remote.git"
            deploy_path = root / "vkgr-p-watch-positive-v2"
            ops = root / "vkgr-p-watch-positive-ops-v2"
            scheduler = root / "scheduler"
            for state in ("pending", "claimed", "rejected"):
                (scheduler / "queue" / state).mkdir(parents=True)
            files = (
                "src/vkgr/p_watch_positive.py", "scripts/p_watch_positive_stats.py", "src/vkgr/retrieval_data.py",
                "scripts/p3_label_eda.py", "configs/p_watch_positive/p-watch-positive-train-only.json",
            )
            source.mkdir(); subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
            pins = {}
            for relative in files:
                path = source / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(relative)
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
                       "new_ops_documents": {"request.json": approval_payload, "pins.json": pinned}}
            before = inventory(request)
            self.assertTrue(before["clear"])
            result = deploy(request, before["request_sha256"])
            self.assertEqual(result["status"], "deployed")
            self.assertTrue((ops / "deployment.json").is_file())
            self.assertFalse((deploy_path / "deployment.json").exists())
            status = subprocess.run(["git", "-C", str(deploy_path), "status", "--porcelain", "--untracked-files=all"],
                                    check=True, capture_output=True, text=True).stdout
            self.assertEqual(status, "")
            record = json.loads((ops / "deployment.json").read_text())
            self.assertEqual(record["commit"], commit)
            self.assertEqual(record["files"], pins)
            verified = verify_fresh_deployment(deploy_path, ops,
                                               {"cwd": str(deploy_path), "pinned_manifest": str(ops / "pins.json")},
                                               approval_payload)
            self.assertEqual(verified["commit"], commit)


if __name__ == "__main__":
    unittest.main()
