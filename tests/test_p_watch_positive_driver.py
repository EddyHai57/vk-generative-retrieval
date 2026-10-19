import json
import sys

from scripts import p_watch_positive_driver as driver
from vkgr import p_watch_positive_ops as ops


def test_train_stdout_must_bind_the_fixed_run_id_and_manifest(tmp_path):
    p = tmp_path / "stdout"
    p.write_text(json.dumps({"status": "complete", "run_id": ops.RUN_IDS[42], "manifest_sha256": "a" * 64}) + "\n")
    assert driver.train_receipt(p, ops.RUN_IDS[42])["manifest_sha256"] == "a" * 64
    assert not ops.parse_verified_stdout(p.read_text(), ops.RUN_IDS[42], "a" * 64)


def test_verify_job_binds_new_verify_identity_and_train_manifest():
    train = {"run_id": ops.RUN_IDS[42], "seed": 42, "argv": ["python", "-m", "vkgr.p_watch_positive_run", "train", "--output", "/runs/train/out"]}
    result = driver.verify_job(train, "b" * 64, "/runs/train/out/manifest.json")
    assert result["run_id"] == ops.RUN_IDS[42] + "-verify"
    assert result["argv"][-1] == "b" * 64


def test_verify_job_rebinds_scheduler_identity_and_adds_pinned_manifest_input():
    train = {"run_id": ops.RUN_IDS[42], "seed": 42, "inputs": ["data"],
             "env": ["VKGR_RUN_ID=" + ops.RUN_IDS[42]],
             "argv": ["python", "-m", "vkgr.p_watch_positive_run", "train", "--output", "/runs/train/out"]}
    result = driver.verify_job(train, "c" * 64, "/runs/train/out/manifest.json")
    assert all(value.partition("=")[0] != "VKGR_RUN_ID" for value in result["env"])
    assert "/runs/train/out/manifest.json" in result["inputs"]


def test_driver_accepts_only_real_vkgr_succeeded_terminal_record(tmp_path, monkeypatch):
    import subprocess
    package = tmp_path / "ops" / "vkgr_ops"
    package.mkdir(parents=True)
    for name in ("__init__.py", "common.py", "admission.py", "launcher.py", "scheduler.py", "status.py", "supervisor.py"):
        raw = subprocess.check_output(["git", "show", "infra/04-sched-forecast:ops/vkgr_ops/" + name])
        (package / name).write_bytes(raw)
    monkeypatch.syspath_prepend(str(package.parent))
    from vkgr_ops.supervisor import terminal_record
    accepted = terminal_record(ops.RUN_IDS[42], 1.0, 0, None, False)
    assert driver.terminal_success(accepted, ops.RUN_IDS[42])
    failed = terminal_record(ops.RUN_IDS[42], 1.0, 2, None, False)
    assert not driver.terminal_success(failed, ops.RUN_IDS[42])


def test_own_session_check_requires_pid1_and_session_leadership():
    assert driver.validate_own_session(500, 1, 500)
    try:
        driver.validate_own_session(500, 25, 500)
    except ValueError:
        pass
    else:
        raise AssertionError("child process must not masquerade as detached driver")


def test_finite_driver_payload_enforces_three_runs_900_second_poll_and_48h_cap():
    entries = [{"run_id": ops.RUN_IDS[seed], "train_job": {"run_id": ops.RUN_IDS[seed]}}
               for seed in ops.SEEDS]
    payload = {"schema": "p-watch-positive-driver-v1", "root": "/ops", "ops": "/private/ops",
               "request_sha256": "a" * 64, "entries": entries, "plan_sha256": ops.digest(entries),
               "deadline_seconds": 172800, "poll_seconds": 900}
    assert driver.validate_driver_payload(payload) is payload
    try:
        driver.validate_driver_payload({**payload, "deadline_seconds": 172801})
    except ValueError:
        pass
    else:
        raise AssertionError("driver deadline must not exceed 48 hours")
    try:
        driver.validate_driver_payload({**payload, "poll_seconds": 899})
    except ValueError:
        pass
    else:
        raise AssertionError("driver poll must not be shorter than 900 seconds")


def _driver_fixture(tmp_path, monkeypatch, *, complete_train_stdout=True, manifest_exists=True):
    import os
    import subprocess
    from pathlib import Path
    scheduler_ops = tmp_path / "scheduler" / "ops"
    package = scheduler_ops / "vkgr_ops"
    package.mkdir(parents=True)
    for name in ops.SCHEDULER_SOURCE_FILES:
        raw = subprocess.check_output(["git", "show", "infra/04-sched-forecast:ops/vkgr_ops/" + name])
        (package / name).write_bytes(raw)
    for name in list(sys.modules):
        if name == "vkgr_ops" or name.startswith("vkgr_ops."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(scheduler_ops))
    from vkgr_ops.supervisor import terminal_record

    root = tmp_path / "queue-root"
    ops_dir = tmp_path / "private-ops"
    ops_dir.mkdir()
    run_id = ops.RUN_IDS[42]
    run_dir = root / "runs" / run_id
    (run_dir / "out").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}")
    pinned = tmp_path / "pinned.json"
    pinned.write_text(json.dumps({"files": {}}))
    source = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    job = {"schema": "vkgr-queue-job-v1", "run_id": run_id, "kind": "real", "cwd": str(source),
           "config": str(config), "seed": 42, "inputs": [], "pinned_manifest": str(pinned),
           "pinned_files": [], "env": [], "cpu_cores": 1, "ram_mib": 8192, "vram_mib": 1024,
           "disk_gib": 1, "wall_hours": 1, "stop_mode": "signal", "no_admission": False,
           "argv": ["python", "-B", "-m", "vkgr.p_watch_positive_run", "train", "--output", str(run_dir / "out")]}
    (root / "queue" / "claimed").mkdir(parents=True)
    (root / "queue" / "claimed" / ("000-" + run_id + ".json")).write_text(json.dumps(job))
    manifest_sha = "a" * 64
    if manifest_exists:
        (run_dir / "out" / "manifest.json").write_text("manifest bytes")
        manifest_sha = ops.sha(run_dir / "out" / "manifest.json")
    if complete_train_stdout:
        (run_dir / "logs" / "stdout.log").write_text(json.dumps({"status": "complete", "run_id": run_id,
            "manifest_sha256": manifest_sha}) + "\n")
    else:
        (run_dir / "logs" / "stdout.log").write_text("no complete runner receipt\n")
    (run_dir / "run.json").write_text(json.dumps({"schema": "vkgr-run-v1", "run_id": run_id,
        "kind": "real", "argv": job["argv"], "config": {"path": str(config), "sha256": ops.sha(config)},
        "git": {"commit": commit, "dirty": False}, "pinned_files": {},
        "resources": {"cpu_cores": 1, "ram_mib": 8192, "vram_mib": 1024, "disk_gib": 1, "wall_seconds": 3600.0},
        "env": {"VKGR_RUN_ID": run_id, "VKGR_PROGRESS_PATH": str(run_dir / "progress.json")}, "inputs": []}))
    (run_dir / "terminal.json").write_text(json.dumps(terminal_record(run_id, 1.0, 0, None, False)))
    entries = [{"run_id": ops.RUN_IDS[seed], "train_job": job if seed == 42 else {"run_id": ops.RUN_IDS[seed]},
                "scheduler_code": str(tmp_path / "scheduler")} for seed in ops.SEEDS]
    payload = {"schema": "p-watch-positive-driver-v1", "root": str(root), "ops": str(ops_dir),
        "request_sha256": "b" * 64, "entries": entries, "plan_sha256": ops.digest(entries),
        "deadline_seconds": 1800, "poll_seconds": 900}
    monkeypatch.setattr(driver, "wait_for_reparented_session", lambda: True)
    return payload, root, ops_dir, run_id


def test_driver_writes_alert_when_train_stdout_is_missing(tmp_path, monkeypatch):
    payload, root, ops_dir, run_id = _driver_fixture(tmp_path, monkeypatch)
    (root / "runs" / run_id / "logs" / "stdout.log").unlink()
    summary = driver.driver(payload)
    assert summary["status"] == "ALERT"
    assert summary["failed_run_id"] == run_id
    assert "stdout" in summary["detail"].lower()
    assert json.loads((ops_dir / "driver-terminal.json").read_text()) == summary


def test_driver_writes_alert_when_train_stdout_has_no_unique_complete_receipt(tmp_path, monkeypatch):
    payload, root, ops_dir, run_id = _driver_fixture(tmp_path, monkeypatch, complete_train_stdout=False)
    summary = driver.driver(payload)
    assert summary["status"] == "ALERT"
    assert summary["failed_run_id"] == run_id
    assert "receipt" in summary["detail"].lower()
    assert (ops_dir / "driver-terminal.json").is_file()


def test_driver_writes_alert_when_successful_train_manifest_is_missing(tmp_path, monkeypatch):
    payload, root, ops_dir, run_id = _driver_fixture(tmp_path, monkeypatch, manifest_exists=False)
    (root / "runs" / run_id / "logs" / "stdout.log").write_text(json.dumps({"status": "complete",
        "run_id": run_id, "manifest_sha256": "a" * 64}) + "\n")
    summary = driver.driver(payload)
    assert summary["status"] == "ALERT"
    assert summary["failed_run_id"] == run_id
    assert "manifest" in summary["detail"].lower()
    assert (ops_dir / "driver-terminal.json").is_file()


def test_driver_writes_alert_with_detail_when_train_manifest_sha_is_mismatched(tmp_path, monkeypatch):
    payload, root, ops_dir, run_id = _driver_fixture(tmp_path, monkeypatch)
    (root / "runs" / run_id / "out" / "manifest.json").write_text("corrupted after stdout receipt")
    summary = driver.driver(payload)
    assert summary["status"] == "ALERT"
    assert summary["failed_run_id"] == run_id
    assert "manifest SHA differs" in summary["detail"]
    assert (ops_dir / "driver-terminal.json").is_file()


def test_driver_writes_alert_when_two_complete_train_receipts_are_present(tmp_path, monkeypatch):
    payload, root, ops_dir, run_id = _driver_fixture(tmp_path, monkeypatch)
    stdout = root / "runs" / run_id / "logs" / "stdout.log"
    stdout.write_text(stdout.read_text() + stdout.read_text())
    summary = driver.driver(payload)
    assert summary["status"] == "ALERT"
    assert summary["failed_run_id"] == run_id
    assert "one run-bound complete manifest receipt" in summary["detail"]
    assert (ops_dir / "driver-terminal.json").is_file()


def test_driver_writes_alert_when_verify_scheduler_source_validation_fails(tmp_path, monkeypatch):
    payload, root, ops_dir, run_id = _driver_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(ops, "validate_scheduler_source", lambda path: (_ for _ in ()).throw(ValueError("source pin failed")))
    summary = driver.driver(payload)
    assert summary["status"] == "ALERT"
    assert summary["failed_run_id"] == run_id + "-verify"
    assert "source pin failed" in summary["detail"]
    assert (ops_dir / "driver-terminal.json").is_file()


def test_driver_writes_alert_when_verify_launcher_prepare_fails(tmp_path, monkeypatch):
    payload, root, ops_dir, run_id = _driver_fixture(tmp_path, monkeypatch)
    import vkgr_ops.launcher as launcher
    monkeypatch.setattr(launcher, "prepare", lambda job: (_ for _ in ()).throw(ValueError("prepare failed")))
    summary = driver.driver(payload)
    assert summary["status"] == "ALERT"
    assert summary["failed_run_id"] == run_id + "-verify"
    assert "prepare failed" in summary["detail"]
    assert (ops_dir / "driver-terminal.json").is_file()


def test_driver_writes_alert_when_verify_scheduler_submit_fails(tmp_path, monkeypatch):
    payload, root, ops_dir, run_id = _driver_fixture(tmp_path, monkeypatch)
    import vkgr_ops.launcher as launcher
    import vkgr_ops.scheduler as scheduler
    calls = []
    monkeypatch.setattr(launcher, "prepare", lambda job: {})
    def fail_once(root_arg, job):
        calls.append(job["run_id"])
        raise OSError("submit failed")
    monkeypatch.setattr(scheduler, "submit", fail_once)
    summary = driver.driver(payload)
    assert summary["status"] == "ALERT"
    assert summary["failed_run_id"] == run_id + "-verify"
    assert "submit failed" in summary["detail"]
    assert calls == [run_id + "-verify"]
    assert (ops_dir / "driver-terminal.json").is_file()
