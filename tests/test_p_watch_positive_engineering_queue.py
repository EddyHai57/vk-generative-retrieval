import pytest
import json
import sys
from pathlib import Path

from scripts import p_watch_positive_engineering_queue as queue
from vkgr import p_watch_positive_ops as ops


def test_inventory_distinguishes_written_from_claimed(tmp_path):
    job = {"run_id": ops.RUN_IDS[42]}
    state = queue.inventory(tmp_path, tmp_path / "ops", job)
    assert state["queue"] == {"pending": [], "claimed": [], "rejected": [], "reasons": {}}
    assert state["run_exists"] is False


def test_queue_state_surfaces_rejected_reason_without_retries(tmp_path):
    rejected = tmp_path / "queue" / "rejected"
    rejected.mkdir(parents=True)
    (rejected / "job-p-watch-positive-s42.json").write_text("{}")
    (rejected / "job-p-watch-positive-s42.reason").write_text("denied")
    state = queue.queue_state(tmp_path, ops.RUN_IDS[42])
    assert state["rejected"] == ["job-p-watch-positive-s42.json"]
    assert state["reasons"] == {"job-p-watch-positive-s42.reason": "denied"}


def test_scheduler_source_accepts_real_ops_kinds_and_injected_env_only(tmp_path, monkeypatch):
    import subprocess
    import sys
    from pathlib import Path
    package = tmp_path / "ops" / "vkgr_ops"
    package.mkdir(parents=True)
    for name in ("__init__.py", "common.py", "admission.py", "launcher.py", "scheduler.py", "status.py", "supervisor.py"):
        raw = subprocess.check_output(["git", "show", "infra/04-sched-forecast:ops/vkgr_ops/" + name])
        (package / name).write_bytes(raw)
    monkeypatch.syspath_prepend(str(tmp_path / "ops"))
    assert ops.validate_scheduler_source(tmp_path / "ops")["commit"] == ops.SCHEDULER_SOURCE_COMMIT
    import vkgr_ops.launcher as launcher
    spec = {"run_id": ops.RUN_IDS[42], "kind": "real", "cwd": str(tmp_path), "config": None,
            "seed": 42, "inputs": [], "pinned_manifest": None, "pinned_files": [], "env": [],
            "cpu_cores": 1, "ram_mib": 1024, "vram_mib": 2048, "disk_gib": 1,
            "wall_hours": 1, "stop_mode": "signal", "no_admission": False, "argv": ["python", "-V"]}
    assert launcher._validate_static(spec) == {}
    with pytest.raises(launcher.LaunchError):
        launcher._validate_static({**spec, "kind": "train"})
    with pytest.raises(launcher.LaunchError):
        launcher._validate_static({**spec, "env": ["VKGR_RUN_ID=forged"]})


def test_fixed_stage_builder_emits_ops_kind_and_launcher_injected_identity(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    package = tmp_path / "ops-code" / "ops" / "vkgr_ops"
    package.mkdir(parents=True)
    for name in ops.SCHEDULER_SOURCE_FILES:
        raw = subprocess.check_output(["git", "show", "infra/04-sched-forecast:ops/vkgr_ops/" + name])
        (package / name).write_bytes(raw)
    monkeypatch.syspath_prepend(str(package.parent))
    from vkgr_ops import launcher

    deployment = tmp_path / "vkgr-p-watch-positive-train-v4"
    deployment.mkdir()
    ops_dir = tmp_path / "vkgr-p-watch-positive-train-ops-v4"
    ops_dir.mkdir()
    (deployment / "src" / "vkgr").mkdir(parents=True)
    (deployment / "src" / "vkgr" / "p_watch_positive_ops.py").write_text("# pinned worker source\n")
    for name in ("inputs.json", "stats.json", "approval.json", "control-42.json", "control-2024.json", "control-2025.json"):
        (ops_dir / name).write_text("{}")
    for seed in ops.SEEDS:
        (deployment / f"config-{seed}.json").write_text("{}")
    subprocess.run(["git", "init", "-q", str(deployment)], check=True)
    subprocess.run(["git", "-C", str(deployment), "add", "."], check=True)
    subprocess.run(["git", "-C", str(deployment), "-c", "user.name=Codex Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(deployment), "rev-parse", "HEAD"], text=True).strip()
    request_path = ops_dir / "worker-request.json"
    request = {"stage": "cuda-preview", "stage_run_id": ops.STAGE_RUN_IDS["cuda-preview"],
               "inputs": str(ops_dir / "inputs.json"), "inputs_sha256": "a" * 64,
               "stats_report": str(ops_dir / "stats.json"), "stats_sha256": "b" * 64,
               "approval": str(ops_dir / "approval.json"), "approval_sha256": "c" * 64,
               "controls": {str(seed): {"path": str(ops_dir / f"control-{seed}.json"), "sha256": ops.CONTROL_MANIFEST_SHA256[seed]} for seed in ops.SEEDS},
               "source_contract": {"deployed_files": {"src/vkgr/p_watch_positive_ops.py": ops.sha(deployment / "src/vkgr/p_watch_positive_ops.py")},
                   "p3c_data_channel": {**ops.P3C_DATA_CHANNEL,
                       "ops_saved_path": str(ops_dir / "p3c-validation-git-sanitized-v1.md")}},
               "configs": {str(seed): {"path": str(deployment / f"config-{seed}.json"), "sha256": "d" * 64} for seed in ops.SEEDS},
               "conservative_vram_mib": 39168,
               "output": str(ops_dir / "preflight-unused")}
    request_path.write_text(json.dumps(request))
    pinned = ops_dir / "pinned.json"
    pinned.write_text(json.dumps({"schema": "vkgr-pinned-v1", "base_commit": commit, "target_commit": commit,
        "files": {"src/vkgr/p_watch_positive_ops.py": ops.sha(deployment / "src/vkgr/p_watch_positive_ops.py")}, "overrides": {}}))
    (ops_dir / "deployment.json").write_text(json.dumps({"data_overlay": True, "p3c_data_channel": {
        **request["source_contract"]["p3c_data_channel"], "copy_sha256": {
            "git_bytes_before_save": ops.P3C_DATA_CHANNEL["git_sha256"],
            "ops_saved_after_copy": ops.P3C_DATA_CHANNEL["git_sha256"],
            "source_before_copy": ops.P3C_DATA_CHANNEL["contract_sha256"],
            "source_after_copy": ops.P3C_DATA_CHANNEL["contract_sha256"],
            "deployment_after_copy": ops.P3C_DATA_CHANNEL["contract_sha256"]}}}))
    preflight_id = ops.STAGE_RUN_IDS["preflight"]
    preflight_req_path = ops_dir / "preflight-request.json"
    preflight_req = {**request, "stage": "preflight", "stage_run_id": preflight_id}
    preflight_req.pop("conservative_vram_mib")
    preflight_req_path.write_text(json.dumps(preflight_req))
    old_preflight_req_path = ops_dir / "old-preflight-request.json"
    old_preflight_req_path.write_text(json.dumps({**preflight_req,
        "stage_run_id": "p-watch-positive-preflight-v2"}))
    stage_root = tmp_path / "scheduler-root"
    preflight_run = stage_root / "runs" / preflight_id
    preflight_run.mkdir(parents=True)
    (stage_root / "queue" / "claimed").mkdir(parents=True)
    stdout_path = preflight_run / "logs" / "stdout.log"
    stdout_path.parent.mkdir()
    stdout_path.write_text(json.dumps({"schema": "p-watch-positive-engineering-preflight-v1",
        "stage_run_id": preflight_id, "status": "preflight_ok", "engineering": False,
        "synthetic": False, "validation_metrics_read": False}) + "\n")
    preflight_argv = ["/usr/bin/python3", "-B", "-m", "vkgr.p_watch_positive_run", "preflight",
                      "--request", str(preflight_req_path), "--request-sha", ops.sha(preflight_req_path)]
    pf_inputs = [str(preflight_req_path), request["inputs"], request["stats_report"], request["approval"],
                 *[ref["path"] for ref in request["controls"].values()],
                 *[ref["path"] for ref in request["configs"].values()]]
    claim = {"schema": "vkgr-queue-job-v1", "run_id": preflight_id, "kind": "ops", "argv": preflight_argv,
             "inputs": pf_inputs,
             "cpu_cores": 1, "ram_mib": 8192, "vram_mib": 0, "disk_gib": 1,
             "wall_hours": 4, "stop_mode": "signal", "no_admission": False}
    (stage_root / "queue" / "claimed" / ("000-" + preflight_id + ".json")).write_text(json.dumps(claim))
    (preflight_run / "run.json").write_text(json.dumps({"schema": "vkgr-run-v1", "run_id": preflight_id,
        "kind": "ops", "argv": preflight_argv, "env": {"VKGR_RUN_ID": preflight_id},
        "resources": {"cpu_cores": 1, "ram_mib": 8192, "vram_mib": 0, "disk_gib": 1, "wall_seconds": 14400.0},
        "stop_mode": "signal",
        "inputs": [{"path": str(Path(path).resolve()), "sha256": ops.sha(path), "bytes": Path(path).stat().st_size}
                   for path in pf_inputs]}))
    from vkgr_ops.supervisor import terminal_record
    (preflight_run / "terminal.json").write_text(json.dumps(terminal_record(preflight_id, 1.0, 0, None, False)))
    proof = {"run_id": preflight_id, "path": str(stdout_path), "sha256": ops.sha(stdout_path)}
    queue_req = {"stage": "cuda-preview", "stage_request": str(request_path),
                 "stage_request_sha256": ops.sha(request_path), "deployment": str(deployment),
                 "ops": str(ops_dir), "python": "/usr/bin/python3",
                 "root": str(stage_root), "inputs": request["inputs"],
                 "inputs_sha256": request["inputs_sha256"], "stats_report": request["stats_report"],
                 "stats_sha256": request["stats_sha256"], "approval": request["approval"],
                 "approval_sha256": request["approval_sha256"],
                     "controls": request["controls"], "source_contract": request["source_contract"],
                     "pinned_manifest": str(pinned),
                 "conservative_vram_mib": 39168, "preflight_proof": proof}
    with pytest.raises(ValueError, match="stage request/job fixed identity mismatch"):
        queue._stage_job({**queue_req, "stage": "preflight",
            "stage_request": str(old_preflight_req_path),
            "stage_request_sha256": ops.sha(old_preflight_req_path)})
    preflight_job = queue._stage_job({**queue_req, "stage": "preflight",
        "stage_request": str(preflight_req_path), "stage_request_sha256": ops.sha(preflight_req_path)})
    assert preflight_job["argv"][:5] == ["/usr/bin/python3", "-B", "-m", "vkgr.p_watch_positive_run", "preflight"]
    job = queue._stage_job(queue_req)
    assert job["kind"] == "ops"
    assert job["argv"][:5] == ["/usr/bin/python3", "-B", "-m", "vkgr.p_watch_positive_ops", "cuda-preview"]
    assert all(value.partition("=")[0] not in ("VKGR_RUN_ID", "VKGR_PROGRESS_PATH") for value in job["env"])
    prepared = launcher.prepare(job)
    assert prepared["env"]["PYTHONPATH"] == str(deployment / "src") + __import__("os").pathsep + str(deployment / "scripts")
    from vkgr_ops import scheduler
    pending = scheduler.submit(tmp_path / "scheduler-root", job)
    record = json.loads(pending.read_text())
    assert record["kind"] == "ops" and record["run_id"] == ops.STAGE_RUN_IDS["cuda-preview"]

    # The same accepted stage receipt must reject a module substitution even when claim and run agree.
    claim_path = next((stage_root / "queue" / "claimed").glob("*-" + preflight_id + ".json"))
    claim = json.loads(claim_path.read_text())
    run_record_path = preflight_run / "run.json"
    run_record = json.loads(run_record_path.read_text())
    old_module_argv = ["/usr/bin/python3", "-B", "-m", "vkgr.p_watch_positive_ops", "preflight",
                       "--request", str(preflight_req_path), "--request-sha", ops.sha(preflight_req_path)]
    claim["argv"] = old_module_argv
    run_record["argv"] = old_module_argv
    claim_path.write_text(json.dumps(claim))
    run_record_path.write_text(json.dumps(run_record))
    with pytest.raises(ValueError, match="module binding"):
        queue._read_successful_stage(queue_req, proof, "preflight", "preflight_ok")


def test_training_namespace_is_distinct_from_the_accepted_stats_v2_deployment():
    request = {"deployment": "/workspace/vkgr-p-watch-positive-train-v4",
               "ops": "/workspace/vkgr-p-watch-positive-train-ops-v4"}
    assert queue.validate_namespaces(request) is True
    for generation in ("v1", "v2", "v3"):
        with pytest.raises(ValueError, match="fresh training namespace"):
            queue.validate_namespaces({
                "deployment": f"/workspace/vkgr-p-watch-positive-train-{generation}",
                "ops": f"/workspace/vkgr-p-watch-positive-train-ops-{generation}"})
    with pytest.raises(ValueError, match="fresh training namespace"):
        queue.validate_namespaces({"deployment": "/workspace/vkgr-p-watch-positive-v2",
                                   "ops": "/workspace/vkgr-p-watch-positive-ops-v2"})


def test_common_accepts_current_python_symlink_but_rejects_other_interpreters_and_input_symlinks(tmp_path, monkeypatch):
    from vkgr import p_watch_positive_run as run
    base = tmp_path / "request"; base.mkdir()
    deployment = base / "vkgr-p-watch-positive-train-v4"; deployment.mkdir()
    ops_dir = base / "vkgr-p-watch-positive-train-ops-v4"; ops_dir.mkdir()
    scheduler = base / "scheduler"; (scheduler / "ops").mkdir(parents=True)
    root = base / "runs"; root.mkdir()
    python = base / ".venv" / "bin" / "python"; python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve(strict=True))
    inputs = base / "inputs.json"; inputs.write_text("{}")
    stats = base / "stats.json"; stats.write_text("{}")
    approval = base / "approval.json"; approval.write_text("{}")
    req = {"action": "inventory", "mode": "stage", "root": str(root), "ops": str(ops_dir),
        "deployment": str(deployment), "scheduler_code": str(scheduler), "python": str(python),
        "stats_report": str(stats), "stats_sha256": "a" * 64, "approval": str(approval),
        "approval_sha256": "c" * 64, "inputs": str(inputs), "inputs_sha256": "a" * 64,
        "controls": {}, "source_contract": {"training_files": {}, "p3c_data_channel": {
            **ops.P3C_DATA_CHANNEL,
            "ops_saved_path": str(ops_dir / "p3c-validation-git-sanitized-v1.md")}}}
    monkeypatch.setattr(ops, "sha", lambda _: "a" * 64)
    monkeypatch.setattr(ops, "read_control_manifests", lambda _: {"training_files": {}, "manifests": {}})
    monkeypatch.setattr(ops, "validate_source_contract", lambda _: {})
    monkeypatch.setattr(ops, "validate_scheduler_source", lambda _: {})
    monkeypatch.setattr(run, "validate_stats_report", lambda *_: {"engineering": False, "synthetic": False})
    monkeypatch.setattr(run, "validate_training_approval", lambda *_args: {})
    assert queue._common(req)[2]["engineering"] is False
    wrong_python = base / "other-python"; wrong_python.symlink_to(Path("/bin/sh").resolve(strict=True))
    with pytest.raises(ValueError, match="current Python executable"):
        queue._common({**req, "python": str(wrong_python)})
    input_link = base / "inputs-link.json"; input_link.symlink_to(inputs)
    with pytest.raises(ValueError, match="non-symlink path required: inputs"):
        queue._common({**req, "inputs": str(input_link)})
