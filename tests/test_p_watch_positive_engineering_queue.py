import pytest
import hashlib
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

from scripts import p_watch_positive_engineering_queue as queue
from scripts import p_watch_positive_deploy as deploy_helper
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


def test_formal_p3c_pin_value_is_the_frozen_contract_sha():
    assert ops.P3C_DATA_CHANNEL["path"] == "docs/p3c_validation.md"
    assert ops.P3C_DATA_CHANNEL["contract_sha256"] == "c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f"


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

    deployment = tmp_path / "vkgr-p-watch-positive-train-v6"
    deployment.mkdir()
    ops_dir = tmp_path / "vkgr-p-watch-positive-train-ops-v6"
    ops_dir.mkdir()
    (deployment / "src" / "vkgr").mkdir(parents=True)
    (deployment / "src" / "vkgr" / "p_watch_positive_ops.py").write_text("# pinned worker source\n")
    sanitized_fixture = b"synthetic sanitized Git p3c fixture\n"
    contract_fixture = b"synthetic data-channel p3c fixture\n"
    p3c_git_sha = hashlib.sha256(sanitized_fixture).hexdigest()
    p3c_contract_sha = hashlib.sha256(contract_fixture).hexdigest()
    channel_constants = {"path": "docs/p3c_validation.md", "git_sha256": p3c_git_sha,
        "contract_sha256": p3c_contract_sha, "source_path": str(tmp_path / "private-p3c.fixture.md")}
    monkeypatch.setattr(ops, "P3C_DATA_CHANNEL", channel_constants)
    p3c_path = deployment / "docs/p3c_validation.md"
    p3c_path.parent.mkdir(parents=True)
    p3c_path.write_bytes(sanitized_fixture)
    for name in ("inputs.json", "stats.json", "approval.json", "control-42.json", "control-2024.json", "control-2025.json"):
        (ops_dir / name).write_text("{}")
    for seed in ops.SEEDS:
        (deployment / f"config-{seed}.json").write_text("{}")
    subprocess.run(["git", "init", "-q", str(deployment)], check=True)
    subprocess.run(["git", "-C", str(deployment), "add", "."], check=True)
    subprocess.run(["git", "-C", str(deployment), "-c", "user.name=Codex Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(deployment), "rev-parse", "HEAD"], text=True).strip()
    p3c_path.write_bytes(contract_fixture)
    request_path = ops_dir / "worker-request.json"
    request = {"stage": "cuda-preview", "stage_run_id": ops.STAGE_RUN_IDS["cuda-preview"],
               "inputs": str(ops_dir / "inputs.json"), "inputs_sha256": "a" * 64,
               "stats_report": str(ops_dir / "stats.json"), "stats_sha256": "b" * 64,
               "approval": str(ops_dir / "approval.json"), "approval_sha256": "c" * 64,
               "controls": {str(seed): {"path": str(ops_dir / f"control-{seed}.json"), "sha256": ops.CONTROL_MANIFEST_SHA256[seed]} for seed in ops.SEEDS},
               "source_contract": {"deployed_files": {"src/vkgr/p_watch_positive_ops.py": ops.sha(deployment / "src/vkgr/p_watch_positive_ops.py")},
                   "p3c_data_channel": {**channel_constants,
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
            "git_bytes_before_save": channel_constants["git_sha256"],
            "ops_saved_after_copy": channel_constants["git_sha256"],
            "source_before_copy": channel_constants["contract_sha256"],
            "source_after_copy": channel_constants["contract_sha256"],
            "deployment_after_copy": channel_constants["contract_sha256"]}}}))
    preflight_id = ops.STAGE_RUN_IDS["preflight"]
    preflight_req_path = ops_dir / "preflight-request.json"
    preflight_req = {**request, "stage": "preflight", "stage_run_id": preflight_id}
    preflight_req.pop("conservative_vram_mib")
    preflight_req_path.write_text(json.dumps(preflight_req))
    old_preflight_req_path = ops_dir / "old-preflight-request.json"
    old_preflight_req_path.write_text(json.dumps({**preflight_req,
        "stage_run_id": "p-watch-positive-preflight-v3"}))
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
    assert job["pinned_files"] == ["docs/p3c_validation.md=" + p3c_contract_sha]
    assert job["argv"][:5] == ["/usr/bin/python3", "-B", "-m", "vkgr.p_watch_positive_ops", "cuda-preview"]
    stage_env = dict(item.split("=", 1) for item in job["env"])
    assert stage_env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert stage_env["OMP_NUM_THREADS"] == "1"
    assert all(value.partition("=")[0] not in ("VKGR_RUN_ID", "VKGR_PROGRESS_PATH") for value in job["env"])
    prepared = launcher.prepare(job)
    assert prepared["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert prepared["env"]["OMP_NUM_THREADS"] == "1"
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
    request = {"deployment": "/workspace/vkgr-p-watch-positive-train-v6",
               "ops": "/workspace/vkgr-p-watch-positive-train-ops-v6"}
    assert queue.validate_namespaces(request) is True
    for generation in ("v1", "v2", "v3", "v4", "v5"):
        with pytest.raises(ValueError, match="fresh training namespace"):
            queue.validate_namespaces({
                "deployment": f"/workspace/vkgr-p-watch-positive-train-{generation}",
                "ops": f"/workspace/vkgr-p-watch-positive-train-ops-{generation}"})
    with pytest.raises(ValueError, match="fresh training namespace"):
        queue.validate_namespaces({"deployment": "/workspace/vkgr-p-watch-positive-v2",
                                   "ops": "/workspace/vkgr-p-watch-positive-ops-v2"})


def _real_deploy_metadata_dry_fixture(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    worktree_listing = subprocess.check_output(["git", "-C", str(repo), "worktree", "list", "--porcelain"], text=True)
    worktrees = [Path(line.removeprefix("worktree ")) for line in worktree_listing.splitlines()
        if line.startswith("worktree ")]
    infra = None
    for path in worktrees:
        launcher_path = path / "ops/vkgr_ops/launcher.py"
        if not launcher_path.is_file():
            continue
        commit_result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True)
        if commit_result.returncode != 0 or commit_result.stdout.strip() != "41b4a4db2f4c84f16ee13df82775c60915a0e4d3":
            continue
        try:
            source = ops.validate_scheduler_source(path / "ops")
        except ValueError:
            continue
        if source["files"]["launcher.py"] == "e6ba787a961eb76dfb241ce6ce37981fda94b5a3eea22729b4c4c730c476387f":
            infra = path
            break
    assert infra is not None, "frozen infra04 scheduler worktree not found by provenance"
    assert ops.validate_scheduler_source(infra / "ops")["commit"] == ops.SCHEDULER_SOURCE_COMMIT

    source = tmp_path / "source"
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(repo), str(source)], check=True)
    branch = "codex/pwatch-fixture"
    subprocess.run(["git", "-C", str(source), "checkout", "-qb", branch], check=True)
    subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
    subprocess.run(["git", "-C", str(source), "remote", "set-url", "origin", str(bare)], check=True)
    subprocess.run(["git", "-C", str(source), "push", "-q", "-u", "origin", branch], check=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()

    tracked = subprocess.check_output(["git", "-C", str(source), "ls-tree", "-r", "--name-only",
        ops.M1_SOURCE_COMMIT, "--", ".python-version", "pyproject.toml", "uv.lock", "src/vkgr"], text=True).splitlines()
    training_files = {name: hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(source), "show", ops.M1_SOURCE_COMMIT + ":" + name])).hexdigest()
        for name in tracked if name in (".python-version", "pyproject.toml", "uv.lock")
        or (name.startswith("src/vkgr/") and "/" not in name[len("src/vkgr/"):] and name.endswith(".py"))}
    assert len(training_files) == 51
    delta = {name: ops.sha(source / name) for name in ops.DELTA_PATHS}
    deployed_files = {**training_files, **delta}
    tree_files = {name: ops.sha(source / name) for name in ops.TREE_DELTA_PATHS}

    python = str(Path(sys.executable).absolute())
    runtime_json = subprocess.check_output([python, "-B", "-c",
        "import importlib.metadata,json,platform;print(json.dumps({'python':platform.python_version(),'exact_dependencies':{n:importlib.metadata.version(n) for n in ('numpy','torch','pyarrow','PyYAML')}},sort_keys=True))"], text=True)
    runtime = json.loads(runtime_json)
    runtime["lock_sha256"] = ops.sha(source / "uv.lock")
    controls_dir = tmp_path / "runs"
    controls_dir.mkdir()
    control_refs = {}
    control_hashes = {}
    for seed in ops.SEEDS:
        manifest = {"schema": "p-seqm1-train-manifest-v1", "kind": "p_seqm1_train", "status": "complete",
            "config": {"seed": seed, "arm": "mean"}, "run_id": f"p-seqm1-mean-s{seed}",
            "selection": {"epoch": ops.CONTROL_STOP_EPOCH[seed], "max_epochs": 100},
            "binding": {"source": {"files": training_files, "python": runtime["python"],
                "dependencies": runtime["exact_dependencies"], "lock_sha": runtime["lock_sha256"],
                "git_commit": ops.M1_SOURCE_COMMIT, "git_dirty": True}}}
        raw = json.dumps(manifest, sort_keys=True).encode()
        path = controls_dir / f"p-seqm1-mean-s{seed}" / "out" / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        control_hashes[seed] = hashlib.sha256(raw).hexdigest()
        control_refs[str(seed)] = {"path": str(path), "sha256": control_hashes[seed]}
    monkeypatch.setattr(ops, "CONTROL_MANIFEST_SHA256", control_hashes)

    contract_path = tmp_path / "synthetic-p3c-contract.md"
    contract_bytes = b"synthetic fixture contract bytes; no real contract content\n"
    contract_path.write_bytes(contract_bytes)
    git_p3c_sha = hashlib.sha256(subprocess.check_output(
        ["git", "-C", str(source), "show", commit + ":docs/p3c_validation.md"])).hexdigest()
    channel_constants = {"path": "docs/p3c_validation.md", "git_sha256": git_p3c_sha,
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(), "source_path": str(contract_path)}
    monkeypatch.setattr(ops, "P3C_DATA_CHANNEL", channel_constants)
    monkeypatch.setattr(deploy_helper, "P3C", channel_constants)

    deployment = tmp_path / "vkgr-p-watch-positive-train-v6"
    ops_dir = tmp_path / "vkgr-p-watch-positive-train-ops-v6"
    run_root = tmp_path / "scheduler-root"
    for state in ("pending", "claimed", "rejected"):
        (run_root / "queue" / state).mkdir(parents=True)
    (run_root / "runs").mkdir()
    run_ids = [ops.STAGE_RUN_IDS["preflight"], ops.STAGE_RUN_IDS["cuda-preview"],
        *ops.RUN_IDS.values(), *(rid + "-verify" for rid in ops.RUN_IDS.values())]
    deployment_request = {"root": str(run_root), "source": str(source), "destination": str(deployment),
        "ops": str(ops_dir), "branch": branch, "commit": commit, "files": deployed_files,
        "run_ids": run_ids, "p3c_data_channel": {**channel_constants,
            "ops_saved_path": str(ops_dir / "p3c-validation-git-sanitized-v1.md")},
        "new_ops_documents": {"pinned-manifest.json": {"schema": "vkgr-pinned-v1",
            "base_commit": commit, "target_commit": commit, "files": deployed_files, "overrides": {}}}}
    inventory = deploy_helper.inventory(deployment_request)
    assert inventory["clear"] is True
    deployment_receipt = deploy_helper.deploy(deployment_request, inventory["request_sha256"])
    assert deployment_receipt["status"] == "deployed"

    inputs_path = ops_dir / "inputs.json"; inputs_path.write_text("{}")
    report_path = ops_dir / "stats-report.json"
    report = {"schema": "p-watch-positive-train-only-report-v1", "run_id": "p-watch-positive-stats-v1",
        "engineering": False, "synthetic": False, "N_like_train_examples": 1, "N_like": 1,
        "N_aux": 1, "N_aux_candidates": 1, "empty_history_aux_count": 0,
        "empty_history_compatibility_note": "synthetic metadata fixture only", "stop_reasons": [],
        "gate": {"week_ge_25_in_auxiliary": False, "N_aux_positive": True,
            "invalid_ratio_fraction_at_or_below_1_percent": True, "empty_history_aux_count": 0},
        "inputs": {"train_weeks": "00-24", "quantile_method": "nearest_rank",
            "supported_items_sha256": "b080240542ae85166186fb621f7b074f9728f8b0e7d01d2c69ae0d0bff6b53a2"}}
    report_path.write_text(json.dumps(report, sort_keys=True))
    report_sha = ops.sha(report_path)
    evidence_path = ops_dir / "decision-evidence.fixture.json"; evidence_path.write_text("{}")
    approval_path = ops_dir / "approval.fixture.json"
    approval = {"schema": "p-watch-positive-training-approval-v1", "status": "approved", "approved_by": "Kiro",
        "stats_run_id": "p-watch-positive-stats-v1", "stats_stdout_sha256": report_sha,
        "authorized_run_ids": list(ops.RUN_IDS.values()), "seeds": list(ops.SEEDS),
        "decision_evidence": {"path": str(evidence_path), "sha256": ops.sha(evidence_path)},
        "card_sha256": "a" * 64}
    approval_path.write_text(json.dumps(approval, sort_keys=True))
    approval_sha = ops.sha(approval_path)
    channel = {**channel_constants, "ops_saved_path": str(ops_dir / "p3c-validation-git-sanitized-v1.md")}
    source_contract = {"root": str(deployment), "training_files": training_files,
        "deployed_files": deployed_files, "pinned_delta": delta, "tree_files": tree_files,
        "runtime": runtime, "accepted_runtime": runtime, "python": python,
        "p3c_data_channel": channel}
    configs = {str(seed): {"path": str(deployment / f"configs/p_watch_positive/p-watch-positive-s{seed}.json"),
        "sha256": ops.sha(deployment / f"configs/p_watch_positive/p-watch-positive-s{seed}.json")} for seed in ops.SEEDS}
    stage_request = {"stage": "preflight", "stage_run_id": ops.STAGE_RUN_IDS["preflight"],
        "inputs": str(inputs_path), "inputs_sha256": ops.sha(inputs_path),
        "stats_report": str(report_path), "stats_sha256": report_sha,
        "approval": str(approval_path), "approval_sha256": approval_sha,
        "controls": control_refs, "source_contract": source_contract, "configs": configs,
        "output": str(ops_dir / "preflight-output.fixture.json")}
    stage_path = ops_dir / "preflight-request.fixture.json"
    stage_path.write_text(json.dumps(stage_request, sort_keys=True))
    request = {"action": "dry-run", "mode": "stage", "root": str(run_root), "ops": str(ops_dir),
        "deployment": str(deployment), "scheduler_code": str(infra), "python": python,
        "stats_report": str(report_path), "stats_sha256": report_sha,
        "approval": str(approval_path), "approval_sha256": approval_sha,
        "inputs": str(inputs_path), "inputs_sha256": ops.sha(inputs_path), "controls": control_refs,
        "source_contract": source_contract, "stage": "preflight", "stage_request": str(stage_path),
        "stage_request_sha256": ops.sha(stage_path), "pinned_manifest": str(ops_dir / "pinned-manifest.json")}
    return request, deployment, ops_dir, source_contract, deployment_receipt, commit, infra


def test_real_deploy_metadata_and_dry_use_real_launcher_with_only_p3c_dirty(tmp_path, monkeypatch):
    request, deployment, ops_dir, source_contract, receipt, commit, infra = _real_deploy_metadata_dry_fixture(tmp_path, monkeypatch)
    assert receipt["p3c_data_channel"]["copy_sha256"]["deployment_after_copy"] == ops.P3C_DATA_CHANNEL["contract_sha256"]
    result = queue.execute(request)
    assert result["status"] == "dry_ok"
    req, jobs, *_ = queue.validate_request(request)
    assert len(jobs) == 1
    assert jobs[0]["pinned_files"] == [
        "docs/p3c_validation.md=" + ops.P3C_DATA_CHANNEL["contract_sha256"]]
    stage_env = dict(item.split("=", 1) for item in jobs[0]["env"])
    assert stage_env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert stage_env["OMP_NUM_THREADS"] == "1"
    for name in tuple(sys.modules):
        if name == "vkgr_ops" or name.startswith("vkgr_ops."):
            del sys.modules[name]
    launcher, _scheduler = queue._scheduler(req)
    prepared = launcher.prepare(jobs[0])
    assert prepared["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert prepared["env"]["OMP_NUM_THREADS"] == "1"
    assert prepared["git"]["dirty_files"] == [{"path": "docs/p3c_validation.md",
        "sha256": ops.P3C_DATA_CHANNEL["contract_sha256"]}]
    import torch
    from vkgr.retrieval_ablation_run import execution_context
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with monkeypatch.context() as guard_env:
        guard_env.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        with pytest.raises(ValueError, match="bound execution device unavailable or nondeterministic"):
            execution_context("cuda")
    (deployment / "unexpected-dirty.fixture").write_text("one additional unpinned path\n")
    with pytest.raises(launcher.LaunchError, match="dirty path not pinned"):
        launcher.prepare(jobs[0])
    rejected_extra = "dirty path not pinned for kind=ops: unexpected-dirty.fixture"
    receipt_dir = Path(tempfile.gettempdir()) / "vkgr-p-watch-positive-p3c-launcher-integration"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    integration_receipt = {"schema": "p-watch-positive-p3c-launcher-integration-v1",
        "fixtures": {"stats_report": "synthetic metadata fixture; validators executed unchanged",
            "approval": "synthetic metadata fixture; validators executed unchanged",
            "m1_control_manifests": "synthetic manifests with runtime/source ledgers from fixed M1 Git commit",
            "p3c_contract_bytes": "synthetic fixture bytes; no contract text stored",
            "p3c_git_bytes": "original tracked Git blob, SHA verified; no private contract bytes",
            "fixture_constants": ["CONTROL_MANIFEST_SHA256", "P3C_DATA_CHANNEL", "deploy_helper.P3C"],
            "production_science_or_gate_constants_changed": False},
        "deployment": {"commit": commit, "path": str(deployment), "ops_path": str(ops_dir),
            "status": receipt["status"], "p3c_data_channel": receipt["p3c_data_channel"]},
        "source_contract_p3c_data_channel": source_contract["p3c_data_channel"],
        "launcher": {"worktree": str(infra), "commit": "41b4a4db2f4c84f16ee13df82775c60915a0e4d3",
            "path": str(Path(launcher.__file__).resolve()),
            "sha256": hashlib.sha256(Path(launcher.__file__).read_bytes()).hexdigest()},
        "job": {"run_id": jobs[0]["run_id"], "kind": jobs[0]["kind"],
            "pinned_files": jobs[0]["pinned_files"], "env": prepared["env"],
            "dirty_files": prepared["git"]["dirty_files"]},
        "dry": result, "extra_dirty_negative": {"status": "rejected_by_real_launcher.prepare",
            "error": rejected_extra},
        "metadata_validation": "queue.execute -> validate_request -> _common -> _stage_job; no mocks",
        "real_launcher_calls": ["queue.execute dry-run", "launcher.prepare direct pinned-path proof",
            "launcher.prepare direct additional-dirty rejection"]}
    receipt_path = receipt_dir / "receipt.json"
    receipt_path.write_text(json.dumps(integration_receipt, sort_keys=True, indent=2) + "\n")
    assert receipt_path.is_file()


def test_common_accepts_current_python_symlink_but_rejects_other_interpreters_and_input_symlinks(tmp_path, monkeypatch):
    from vkgr import p_watch_positive_run as run
    base = tmp_path / "request"; base.mkdir()
    deployment = base / "vkgr-p-watch-positive-train-v6"; deployment.mkdir()
    ops_dir = base / "vkgr-p-watch-positive-train-ops-v6"; ops_dir.mkdir()
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
