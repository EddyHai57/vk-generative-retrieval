from pathlib import Path
import unittest
import tempfile
import copy
import yaml
from vkgr.config import load_config

class ConfigTests(unittest.TestCase):
    def test_loads_approved_smoke(self):
        try:
            from vkgr.config import load_config
        except ImportError:
            self.fail("P1 load_config interface is missing")
        config = load_config(Path("configs/smoke.yaml"))
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.fixture_path, "tests/fixtures/p1_synthetic.json")
        self.assertFalse(config.effective_config["resources"]["network_required"])

    def reject(self, raw):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(raw)
            try:
                load_config(path)
            except Exception as error:
                self.assertIsInstance(error, ValueError, "invalid config must fail with ValueError")
            else:
                self.fail("invalid config was accepted")

    def test_rejects_unknown_or_missing_fields(self):
        good = yaml.safe_load(Path("configs/smoke.yaml").read_text())
        for section in (None, "data", "resources", "run", "claims"):
            for mutation in ("extra", "missing"):
                value = copy.deepcopy(good)
                target = value if section is None else value[section]
                if mutation == "extra":
                    target["unknown"] = "unexpected"
                else:
                    target.pop(next(iter(target)))
                with self.subTest(section=section, mutation=mutation):
                    self.reject(yaml.safe_dump(value))

    def test_rejects_duplicate_yaml_keys(self):
        good = Path("configs/smoke.yaml").read_text()
        self.reject(good + "task_id: P1\n")
        self.reject(good.replace("  seed: 42", "  seed: 1\n  seed: 42"))

    def test_rejects_wrong_types_and_boundary_values(self):
        good = yaml.safe_load(Path("configs/smoke.yaml").read_text())
        cases = [((), "schema_version", True), ((), "schema_version", 2),
                 ((), "task_id", "P2"), (("resources",), "network_required", 0),
                 (("resources",), "network_required", True),
                 (("resources",), "gpu_required", True),
                 (("resources",), "execution", "cuda"),
                 (("run",), "seed", True), (("run",), "seed", -1),
                 (("run",), "seed", 2**32), (("run",), "seed", "42"),
                 (("run",), "seed", 42.0), (("run",), "model", "tiger"),
                 (("claims",), "allowed", "model_quality")]
        for trail, key, bad in cases:
            value = copy.deepcopy(good)
            target = value
            for part in trail:
                target = target[part]
            target[key] = bad
            with self.subTest(trail=trail, key=key, bad=bad):
                self.reject(yaml.safe_dump(value))

    def test_rejects_unapproved_fixture_paths(self):
        good = yaml.safe_load(Path("configs/smoke.yaml").read_text())
        for bad in ("/tmp/fixture.json", "../fixture.json", "https://example.test/x",
                    "tests/fixtures/../p1_synthetic.json", "tests//fixtures/p1_synthetic.json",
                    "tests/fixtures/other.json", None, 42):
            value = copy.deepcopy(good)
            value["data"]["fixture_path"] = bad
            with self.subTest(path=bad):
                self.reject(yaml.safe_dump(value))

    def test_malformed_yaml_is_explicit_value_error(self):
        for raw in ("run: [", "!!python/object:os.system {}", "? [unhashable]\n: value", "&cycle {data: *cycle}"):
            with self.subTest(raw=raw):
                self.reject(raw)
