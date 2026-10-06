from pathlib import Path
import tempfile
import unittest
import subprocess
from unittest.mock import patch
from vkgr.runtime import run_smoke

ROOT = Path(__file__).resolve().parents[1]

class BoundaryTests(unittest.TestCase):
    def test_rejects_output_outside_p1_area_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "unapproved"
            with self.assertRaises(ValueError):
                run_smoke(ROOT / "configs/smoke.yaml", output)
            self.assertFalse(output.exists())
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs") as tmp:
            output = Path(tmp) / ".." / Path(tmp).name / "noncanonical"
            with self.assertRaises(ValueError):
                run_smoke(ROOT / "configs/smoke.yaml", output)
            self.assertFalse(output.exists())

    def test_git_failure_stops_before_output_is_created(self):
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs") as tmp:
            output = Path(tmp) / "run"
            with patch("vkgr.manifest.subprocess.run", side_effect=subprocess.CalledProcessError(1, ["git"])):
                with self.assertRaises(subprocess.SubprocessError):
                    run_smoke(ROOT / "configs/smoke.yaml", output)
            self.assertFalse(output.exists())

    def test_symlink_input_and_output_are_rejected_without_writing(self):
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs") as tmp:
            directory = Path(tmp)
            config_link = directory / "config.yaml"
            config_link.symlink_to(ROOT / "configs/smoke.yaml")
            with self.assertRaises(ValueError):
                run_smoke(config_link, directory / "run")
            self.assertFalse((directory / "run").exists())
            with tempfile.TemporaryDirectory() as external:
                link = directory / "linked"
                link.symlink_to(external, target_is_directory=True)
                with self.assertRaises(ValueError):
                    run_smoke(ROOT / "configs/smoke.yaml", link / "run")
                self.assertFalse((Path(external) / "run").exists())

    def test_smoke_and_verify_work_with_network_blocked_and_only_git_subprocesses(self):
        import socket
        import random
        from vkgr.manifest import verify_manifest
        original_run = subprocess.run
        def local_git_only(args, *positional, **kwargs):
            if args[0] != "git":
                raise AssertionError("only local Git subprocesses are allowed")
            return original_run(args, *positional, **kwargs)
        before = random.getstate()
        with tempfile.TemporaryDirectory(prefix="p1-test-", dir=ROOT / "outputs") as tmp:
            with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")), \
                 patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")), \
                 patch("subprocess.run", side_effect=local_git_only):
                path = run_smoke(ROOT / "configs/smoke.yaml", Path(tmp) / "run")
                self.assertIsNone(verify_manifest(path))
        self.assertEqual(random.getstate(), before)

    def test_fixture_symlink_is_rejected_before_reading_or_git(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "repo"
            (root / "configs").mkdir(parents=True)
            (root / "tests/fixtures").mkdir(parents=True)
            shutil.copyfile(ROOT / "configs/smoke.yaml", root / "configs/smoke.yaml")
            external = Path(temporary) / "external.json"
            shutil.copyfile(ROOT / "tests/fixtures/p1_synthetic.json", external)
            (root / "tests/fixtures/p1_synthetic.json").symlink_to(external)
            output = root / "outputs/p1-run"
            with patch("vkgr.manifest.ROOT", root), \
                 patch("vkgr.manifest.git_context", side_effect=RuntimeError("unsafe fixture reached Git")):
                try:
                    run_smoke(root / "configs/smoke.yaml", output)
                except Exception as error:
                    self.assertIsInstance(error, ValueError, "fixture symlink must be rejected at path boundary")
                else:
                    self.fail("unsafe fixture was accepted")
            self.assertFalse(output.exists())
