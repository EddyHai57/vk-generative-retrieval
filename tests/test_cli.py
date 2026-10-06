from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class CliTests(unittest.TestCase):
    def call(self, *args):
        return subprocess.run([sys.executable, "-B", "-m", "vkgr", *args],
                              capture_output=True, text=True, cwd=ROOT)

    def test_smoke_and_verify_cli(self):
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs") as tmp:
            output = Path(tmp) / "run"
            run = self.call("smoke", "--config", "configs/smoke.yaml", "--output-dir", str(output))
            self.assertEqual(run.returncode, 0, run.stderr)
            manifest = output / "run_manifest.json"
            self.assertTrue(manifest.is_file())
            verify = self.call("verify", "--manifest", str(manifest))
            self.assertEqual(verify.returncode, 0, verify.stderr)

    def test_logs_success_to_stderr_without_personal_environment(self):
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs") as tmp:
            run = self.call("smoke", "--config", "configs/smoke.yaml", "--output-dir", str(Path(tmp) / "run"))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("INFO", run.stderr)
            self.assertIn("smoke completed", run.stderr)
            self.assertEqual(run.stdout, "")
            self.assertNotIn(str(ROOT), run.stderr)
            self.assertNotIn("/Users/", run.stderr)

    def test_invalid_input_exits_two_with_safe_diagnostic(self):
        cases = [("smoke", "--config", "/private/tmp/p1-missing-private-path.yaml", "--output-dir", "outputs/p1-unreached"),
                 ("verify", "--manifest", "/private/tmp/p1-missing-private-path.json")]
        for args in cases:
            with self.subTest(command=args[0]):
                run = self.call(*args)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn("ERROR", run.stderr)
                self.assertNotIn("Traceback", run.stderr)
                self.assertNotIn("private-path", run.stderr)
                self.assertNotIn("/Users/", run.stderr)
                self.assertEqual(run.stdout, "")

    def test_cli_refuses_existing_run_and_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs") as tmp:
            output = Path(tmp) / "run"
            first = self.call("smoke", "--config", "configs/smoke.yaml", "--output-dir", str(output))
            self.assertEqual(first.returncode, 0, first.stderr)
            files = {p.name: p.read_bytes() for p in output.iterdir()}
            second = self.call("smoke", "--config", "configs/smoke.yaml", "--output-dir", str(output))
            self.assertEqual(second.returncode, 2)
            self.assertIn("already exists", second.stderr)
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, files)

    def test_argument_errors_are_nonzero_and_main_returns_int(self):
        from vkgr.__main__ import main
        self.assertEqual(self.call("smoke").returncode, 2)
        self.assertEqual(self.call("unknown").returncode, 2)
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs") as tmp:
            self.assertEqual(main(["smoke", "--config", "configs/smoke.yaml", "--output-dir", str(Path(tmp) / "run")]), 0)
