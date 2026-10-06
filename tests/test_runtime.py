from pathlib import Path
import json
import unittest
import tempfile

class RuntimeTests(unittest.TestCase):
    def api(self):
        try:
            from vkgr import runtime
        except ImportError:
            self.fail("P1 synthetic runtime interface is missing")
        return runtime

    def test_loads_hand_authored_fixture(self):
        fixture = self.api().load_fixture(Path("tests/fixtures/p1_synthetic.json"))
        self.assertEqual(fixture["values"], [3, 7, 11, 19])
        self.assertEqual(fixture["origin"], "hand_authored_synthetic_not_vk_schema")

    def test_rejects_non_synthetic_or_invalid_fixture(self):
        cases = [None, {}, {"schema_version": True, "origin": "hand_authored_synthetic_not_vk_schema", "values": [1]},
                 {"schema_version": 1, "origin": "vk", "values": [1]},
                 {"schema_version": 1, "origin": "hand_authored_synthetic_not_vk_schema", "values": []},
                 {"schema_version": 1, "origin": "hand_authored_synthetic_not_vk_schema", "values": [True]},
                 {"schema_version": 1, "origin": "hand_authored_synthetic_not_vk_schema", "values": [-1]},
                 {"schema_version": 1, "origin": "hand_authored_synthetic_not_vk_schema", "values": [1001]},
                 {"schema_version": 1, "origin": "hand_authored_synthetic_not_vk_schema", "values": [1] * 17}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            for case in cases:
                path.write_text(json.dumps(case))
                with self.subTest(case=case), self.assertRaises(ValueError):
                    self.api().load_fixture(path)

    def test_seed_drives_reproducible_computation(self):
        runtime = self.api()
        self.assertTrue(hasattr(runtime, "compute_result"), "seeded computation is missing")
        fixture = runtime.load_fixture(Path("tests/fixtures/p1_synthetic.json"))
        result = runtime.compute_result(fixture, 42)
        self.assertEqual(result["weighted_sum"], 144)
        self.assertEqual(result["weights"], [2, 1, 5, 4])
        self.assertEqual(runtime.compute_result(fixture, 42), result)
        self.assertNotEqual(runtime.compute_result(fixture, 43)["weights"], result["weights"])

    def test_run_smoke_creates_verifiable_reproducible_result(self):
        from vkgr.manifest import verify_manifest
        runtime = self.api()
        self.assertTrue(hasattr(runtime, "run_smoke"), "single-call smoke orchestration missing")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=root / "outputs") as tmp:
            first = runtime.run_smoke(root / "configs/smoke.yaml", Path(tmp) / "one")
            second = runtime.run_smoke(root / "configs/smoke.yaml", Path(tmp) / "two")
            self.assertEqual(first.name, "run_manifest.json")
            self.assertIsNone(verify_manifest(first))
            self.assertIsNone(verify_manifest(second))
            self.assertEqual((first.parent / "result.json").read_bytes(), (second.parent / "result.json").read_bytes())
            self.assertEqual(json.loads(first.read_text())["result_hash"], json.loads(second.read_text())["result_hash"])

    def test_refuses_existing_output_without_changing_it(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=root / "outputs") as tmp:
            output = Path(tmp) / "existing"
            output.mkdir()
            marker = output / "result.json"
            marker.write_bytes(b"precious existing output")
            with self.assertRaises((ValueError, FileExistsError)):
                self.api().run_smoke(root / "configs/smoke.yaml", output)
            self.assertEqual(marker.read_bytes(), b"precious existing output")
            self.assertEqual([p.name for p in output.iterdir()], ["result.json"])

    def test_rejects_duplicate_fixture_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.json"
            path.write_text('{"schema_version": 1, "origin": "hand_authored_synthetic_not_vk_schema", "values": [1], "values": [1]}')
            with self.assertRaises(ValueError):
                self.api().load_fixture(path)
