from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from vkgr.config import load_config
from vkgr.runtime import load_fixture, compute_result

ROOT = Path(__file__).resolve().parents[1]

class ManifestTests(unittest.TestCase):
    def api(self):
        try:
            from vkgr import manifest
        except ImportError:
            self.fail("versioned manifest interface is missing")
        return manifest

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.result_path = self.directory / "result.json"
        result = compute_result(load_fixture(ROOT / "tests/fixtures/p1_synthetic.json"), 42)
        self.result_path.write_text(json.dumps(result, sort_keys=True) + "\n")

    def make(self):
        return self.api().build_manifest(ROOT / "configs/smoke.yaml", self.result_path,
            datetime.now(timezone.utc).isoformat(), 0.01)

    def test_manifest_tracks_versioned_inputs_and_actual_source(self):
        value = self.make()
        self.assertEqual(set(value), {
            "schema_version", "task_id", "status", "git_commit", "git_dirty", "source_fingerprint",
            "config_path", "config_hash", "effective_config", "fixture_path", "fixture_hash", "seed",
            "result_path", "result_hash", "python_version", "dependency_versions", "started_at_utc",
            "elapsed_seconds", "claim_boundary"})
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["task_id"], "P1")
        self.assertEqual(value["status"], "success")
        self.assertEqual(value["config_path"], "configs/smoke.yaml")
        self.assertEqual(value["seed"], 42)
        self.assertEqual(value["effective_config"], load_config(ROOT / "configs/smoke.yaml").effective_config)
        self.assertEqual(value["result_hash"], hashlib.sha256(self.result_path.read_bytes()).hexdigest())
        self.assertRegex(value["source_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertRegex(value["git_commit"], r"^[0-9a-f]{40}$")
        self.assertIs(type(value["git_dirty"]), bool)
        self.assertEqual(value["dependency_versions"], {"PyYAML": "6.0.2"})
        self.assertEqual(value["claim_boundary"], "synthetic scaffold execution; model quality not evaluated")

    def test_writes_and_reloads_manifest(self):
        api = self.api()
        self.assertTrue(hasattr(api, "write_manifest"), "manifest writer missing")
        self.assertTrue(hasattr(api, "verify_manifest"), "manifest verifier missing")
        path = self.directory / "run_manifest.json"
        value = self.make()
        api.write_manifest(path, value)
        self.assertEqual(json.loads(path.read_text()), value)
        self.assertIsNone(api.verify_manifest(path))

    def test_rejects_invalid_manifest_structure_and_types(self):
        value = self.make()
        path = self.directory / "run_manifest.json"
        cases = [("schema_version", True), ("schema_version", 2), ("task_id", "P2"),
                 ("status", "failed"), ("seed", True), ("seed", -1),
                 ("git_commit", "pending"), ("git_dirty", 1),
                 ("source_fingerprint", "unknown"), ("config_hash", "abc"),
                 ("python_version", "3.9.6"), ("dependency_versions", {"PyYAML": "0"}),
                 ("started_at_utc", "yesterday"), ("started_at_utc", "2026-01-01T00:00:00+10:00"),
                 ("elapsed_seconds", True), ("elapsed_seconds", -1), ("elapsed_seconds", float("nan")),
                 ("claim_boundary", "model evaluated")]
        for key, bad in cases:
            damaged = dict(value)
            damaged[key] = bad
            path.write_text(json.dumps(damaged))
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                self.api().verify_manifest(path)
        for damaged in ([], {}, {**value, "extra": 1}, {k: v for k, v in value.items() if k != "fixture_hash"}):
            path.write_text(json.dumps(damaged))
            with self.subTest(structure=type(damaged)), self.assertRaises(ValueError):
                self.api().verify_manifest(path)

    def test_rejects_unsafe_manifest_paths(self):
        value = self.make()
        path = self.directory / "run_manifest.json"
        for field in ("config_path", "fixture_path", "result_path"):
            for bad in ("/tmp/x", "../x", "outputs/../x", "https://example.test/x", "./configs/smoke.yaml", "a//b", None):
                damaged = dict(value)
                damaged[field] = bad
                path.write_text(json.dumps(damaged))
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    self.api().verify_manifest(path)

    def test_rejects_hash_tamper_and_missing_inputs(self):
        value = self.make()
        path = self.directory / "run_manifest.json"
        for key in ("config_hash", "fixture_hash", "result_hash", "source_fingerprint"):
            damaged = dict(value)
            damaged[key] = "0" * 64
            path.write_text(json.dumps(damaged))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api().verify_manifest(path)
        damaged = dict(value)
        damaged["result_path"] = self.directory.relative_to(ROOT).as_posix() + "/missing.json"
        path.write_text(json.dumps(damaged))
        with self.assertRaises(ValueError):
            self.api().verify_manifest(path)
        path.write_text(json.dumps(value))
        self.result_path.write_text("{}")
        with self.assertRaises(ValueError):
            self.api().verify_manifest(path)

    def test_rejects_config_seed_and_result_inconsistency(self):
        value = self.make()
        path = self.directory / "run_manifest.json"
        for key, bad in (("seed", 43), ("effective_config", {}), ("fixture_path", "configs/smoke.yaml")):
            damaged = dict(value)
            damaged[key] = bad
            if key == "fixture_path":
                damaged["fixture_hash"] = value["config_hash"]
            path.write_text(json.dumps(damaged))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api().verify_manifest(path)
        result = json.loads(self.result_path.read_text())
        result["weighted_sum"] = 999
        self.result_path.write_text(json.dumps(result))
        value["result_hash"] = hashlib.sha256(self.result_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.api().verify_manifest(path)

    def test_rejects_manifest_result_from_another_run_directory(self):
        value = self.make()
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs") as other:
            path = Path(other) / "run_manifest.json"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                self.api().verify_manifest(path)

    def test_rejects_duplicate_json_keys(self):
        value = self.make()
        path = self.directory / "run_manifest.json"
        path.write_text(json.dumps(value)[:-1] + ', "seed": 42}')
        with self.assertRaises(ValueError):
            self.api().verify_manifest(path)

    def test_rejects_nonexistent_git_commit(self):
        value = self.make()
        value["git_commit"] = "0" * 40
        path = self.directory / "run_manifest.json"
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.api().verify_manifest(path)
