import pytest
import hashlib
import json
import runpy
import sys
import subprocess
import os
from pathlib import Path

from vkgr import p_watch_positive_ops as ops


def test_fixed_training_run_ids_are_exactly_the_registered_three():
    assert ops.RUN_IDS == {42: "p-watch-positive-s42", 2024: "p-watch-positive-s2024", 2025: "p-watch-positive-s2025"}
    with pytest.raises(ValueError):
        ops.run_id_for_seed(7)


def test_python_path_allows_only_symlink_resolving_to_current_interpreter(tmp_path):
    current = Path(sys.executable).resolve(strict=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(current)
    assert ops.validate_python_executable(str(venv_python)) == current
    wrong = tmp_path / "other-bin-python"
    wrong.symlink_to(Path("/bin/sh").resolve(strict=True))
    with pytest.raises(ValueError, match="current Python executable"):
        ops.validate_python_executable(str(wrong))


def test_stage_request_accepts_current_python_symlink_but_keeps_input_paths_nonsymlink(tmp_path, monkeypatch):
    from vkgr import p_watch_positive_run as run
    from vkgr import retrieval_ablation_run as ablation
    base = tmp_path / "stage"
    deployment = base / "vkgr-p-watch-positive-train-v3"
    ops_dir = base / "vkgr-p-watch-positive-train-ops-v3"
    deployment.mkdir(parents=True); ops_dir.mkdir()
    input_file = base / "inputs.json"; input_file.write_text("{}")
    stats = base / "stats.json"; stats.write_text("{}")
    approval = base / "approval.json"; approval.write_text("{}")
    python = base / ".venv" / "bin" / "python"; python.parent.mkdir(parents=True)
    python.symlink_to(Path(sys.executable).resolve(strict=True))
    configs = {str(seed): {"path": str(base / f"config-{seed}.json"), "sha256": "d" * 64}
               for seed in ops.SEEDS}
    config_values = {ref["path"]: {**run.default_config(), "seed": seed}
                     for seed, ref in ((seed, configs[str(seed)]) for seed in ops.SEEDS)}
    req = {"stage": "preflight", "stage_run_id": ops.STAGE_RUN_IDS["preflight"],
        "deployment": str(deployment), "ops": str(ops_dir), "python": str(python),
        "inputs": str(input_file), "inputs_sha256": "a" * 64, "configs": configs,
        "stats_report": str(stats), "stats_sha256": "b" * 64, "approval": str(approval),
        "approval_sha256": "c" * 64, "controls": {}, "source_contract": {},
        "output": str(ops_dir / "unused.json")}
    monkeypatch.setenv("VKGR_RUN_ID", ops.STAGE_RUN_IDS["preflight"])
    monkeypatch.setattr(ops, "read_control_manifests", lambda _: {"training_files": {}, "manifests": {}})
    monkeypatch.setattr(ops, "validate_source_contract", lambda _: {})
    monkeypatch.setattr(ops, "sha", lambda _: "a" * 64)
    monkeypatch.setattr(run, "validate_stats_report", lambda *_: {"engineering": False, "synthetic": False})
    monkeypatch.setattr(run, "validate_training_approval", lambda *_args: {})
    monkeypatch.setattr(ablation, "read_pinned", lambda path, _sha: config_values[path])
    controls, report, loaded, reference = ops._load_stage_request(req, "preflight")
    assert controls["training_files"] == {} and report["engineering"] is False
    assert set(loaded) == set(ops.SEEDS) and reference["seed"] == 42
    monkeypatch.setattr(ops, "_real_stage_context",
                        lambda *_: (_ for _ in ()).throw(RuntimeError("data-loader sentinel reached")))
    with pytest.raises(RuntimeError, match="data-loader sentinel reached"):
        ops.formal_preflight(req)
    old_id_req = {**req, "stage_run_id": "p-watch-positive-preflight-v1"}
    monkeypatch.setenv("VKGR_RUN_ID", old_id_req["stage_run_id"])
    with pytest.raises(ValueError, match="scheduler-injected fixed run_id"):
        ops._load_stage_request(old_id_req, "preflight")
    monkeypatch.setenv("VKGR_RUN_ID", ops.STAGE_RUN_IDS["preflight"])
    input_link = base / "inputs-link.json"; input_link.symlink_to(input_file)
    with pytest.raises(ValueError, match="non-symlink stage path required: inputs"):
        ops._load_stage_request({**req, "inputs": str(input_link)}, "preflight")


def test_real_frozen_seqm1_validator_still_rejects_schema_task_and_hyperparameter_changes():
    from vkgr import retrieval_seqm1_run as m1
    valid = m1.default_config()
    assert m1.validate_config(valid, synthetic=True) is valid
    for changed in ({**valid, "schema": "p-seq-m1-run-v1"},
                    {**valid, "task_id": "P-SEQ-OTHER"},
                    {**valid, "learning_rate": valid["learning_rate"] * 2}):
        with pytest.raises(ValueError):
            m1.validate_config(changed, synthetic=True)


def test_preflight_id_rotates_without_changing_unused_cuda_or_train_ids():
    assert ops.STAGE_RUN_IDS == {
        "preflight": "p-watch-positive-preflight-v4",
        "cuda-preview": "p-watch-positive-cuda-preview-v2",
    }
    assert ops.STAGE_RUN_IDS["preflight"] != "p-watch-positive-preflight-v3"
    assert ops.STAGE_RUN_IDS["cuda-preview"] != "p-watch-positive-cuda-preview-v1"


def test_cuda_execution_context_guard_rejects_missing_workspace_before_hardware_use(monkeypatch):
    import torch
    from vkgr.retrieval_ablation_run import execution_context
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with monkeypatch.context() as env:
        env.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        with pytest.raises(ValueError, match="bound execution device unavailable or nondeterministic"):
            execution_context("cuda")


def test_p3c_channel_requires_fixed_mapping_saved_git_bytes_and_copy_receipt(tmp_path, monkeypatch):
    git_bytes = b"synthetic sanitized Git document\n"
    contract_bytes = b"synthetic private contract document\n"
    git_sha = hashlib.sha256(git_bytes).hexdigest()
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    ops_dir = tmp_path / "vkgr-p-watch-positive-train-ops-v4"
    ops_dir.mkdir()
    source = tmp_path / "private-contract.md"; source.write_bytes(contract_bytes)
    saved = ops_dir / "p3c-validation-git-sanitized-v1.md"; saved.write_bytes(git_bytes)
    channel_constants = {"path": "docs/p3c_validation.md", "git_sha256": git_sha,
        "contract_sha256": contract_sha, "source_path": str(source)}
    monkeypatch.setattr(ops, "P3C_DATA_CHANNEL", channel_constants)
    channel = {**channel_constants, "ops_saved_path": str(saved)}
    copy_sha = {"git_bytes_before_save": git_sha, "ops_saved_after_copy": git_sha,
        "source_before_copy": contract_sha, "source_after_copy": contract_sha,
        "deployment_after_copy": contract_sha}
    receipt_path = ops_dir / "deployment.json"
    receipt_path.write_text(json.dumps({"data_overlay": True,
        "p3c_data_channel": {**channel, "copy_sha256": copy_sha}}))
    assert ops.validate_p3c_data_channel(channel) == channel
    receipt_path.write_text(json.dumps({"data_overlay": True,
        "p3c_data_channel": {**channel, "path": "docs/other.md", "copy_sha256": copy_sha}}))
    with pytest.raises(ValueError, match="deployment receipt differs"):
        ops.validate_p3c_data_channel(channel)
    receipt_path.write_text(json.dumps({"data_overlay": True,
        "p3c_data_channel": {**channel, "copy_sha256": copy_sha}}))
    for field, value in (("path", "docs/not-p3c.md"), ("git_sha256", "0" * 64),
                         ("contract_sha256", "1" * 64), ("source_path", str(tmp_path / "wrong.md"))):
        with pytest.raises(ValueError):
            ops.validate_p3c_data_channel({**channel, field: value})
    source.unlink()
    with pytest.raises(ValueError, match="contract source missing or changed"):
        ops.validate_p3c_data_channel(channel)
    source.write_bytes(contract_bytes + b"changed")
    with pytest.raises(ValueError, match="contract source missing or changed"):
        ops.validate_p3c_data_channel(channel)
    source.write_bytes(contract_bytes)
    saved.unlink()
    with pytest.raises(ValueError, match="saved Git-original"):
        ops.validate_p3c_data_channel(channel)
    saved.write_bytes(b"changed synthetic Git document\n")
    with pytest.raises(ValueError, match="saved Git-original"):
        ops.validate_p3c_data_channel(channel)
    assert ops.RUN_IDS == {42: "p-watch-positive-s42", 2024: "p-watch-positive-s2024",
                          2025: "p-watch-positive-s2025"}


def test_control_manifest_binding_pins_sha_and_early_stop():
    assert ops.CONTROL_MANIFEST_SHA256 == {
        42: "bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e",
        2024: "8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8",
        2025: "783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb",
    }
    manifests = {seed: {"sha256": ops.CONTROL_MANIFEST_SHA256[seed], "selection": {"epoch": stop, "max_epochs": 100}}
                 for seed, stop in ((42, 15), (2024, 10), (2025, 14))}
    assert ops.validate_control_manifests(manifests) is True
    manifests[42]["selection"]["epoch"] = 100
    with pytest.raises(ValueError, match="max_epochs"):
        ops.validate_control_manifests(manifests)


def test_context_inclusive_vram_declaration_rounds_peak_times_margin_up_to_256_mib():
    assert ops.declared_vram_mib(1000) == 1280
    assert ops.declared_vram_mib(1024) == 1280


def test_context_inclusive_measurement_adds_allocator_peak_and_process_context_overhead():
    import os
    memory = {"pid": os.getpid(), "context_process_mib": 500, "context_reserved_mib": 250,
              "samples": [{"stage": stage, "own_process_mib": 1000, "reserved_mib": 700,
                           "max_reserved_mib": 800, "max_allocated_mib": 750}
                          for stage in ("gradient", "decode", "reload")]}
    assert ops.measured_vram(memory) == 1100
    assert ops.declared_vram_mib(ops.measured_vram(memory)) == 1280


def test_verify_stdout_requires_verified_train_and_manifest_binding():
    line = '{"status":"verified","run_id":"p-watch-positive-s42","manifest_sha256":"' + "a" * 64 + '"}\n'
    assert ops.parse_verified_stdout(line, "p-watch-positive-s42", "a" * 64)
    assert not ops.parse_verified_stdout(line.replace('"status":"verified"', '"status":"complete"'), "p-watch-positive-s42", "a" * 64)


def test_pinned_delta_is_closed_over_the_six_authorized_paths(tmp_path):
    training = {"baseline.py": "0" * 64}
    delta = {name: "1" * 64 for name in ops.DELTA_PATHS}
    deployed = {**training, **delta}
    for name, pin in deployed.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"baseline" if name == "baseline.py" else b"new")
        if name == "baseline.py": training[name] = ops.sha(path)
        else: delta[name] = ops.sha(path)
    deployed = {**training, **delta}
    assert ops.validate_pinned_delta(tmp_path, training, deployed, delta)
    with pytest.raises(ValueError, match="exact six-path"):
        ops.validate_pinned_delta(tmp_path, training, deployed, {**delta, "extra.py": "2" * 64})


def test_m1_source_ledger_and_global_tree_have_separate_closed_delta_sets():
    assert set(ops.DELTA_PATHS) == {
        "src/vkgr/p_watch_positive.py", "src/vkgr/p_watch_positive_epoch.py",
        "src/vkgr/p_watch_positive_nonempty.py", "src/vkgr/p_watch_positive_rows.py",
        "src/vkgr/p_watch_positive_run.py",
        "src/vkgr/p_watch_positive_ops.py",
    }
    diff = subprocess.check_output(["git", "diff", "--name-status", "--no-renames",
        ops.M1_SOURCE_COMMIT, "HEAD"], text=True)
    rows = [line.split("\t", 1) for line in diff.splitlines()]
    assert all(len(row) == 2 and row[0] == "A" for row in rows)
    assert {row[1] for row in rows} == set(ops.TREE_DELTA_PATHS)
    root = Path(__file__).resolve().parents[1]
    current = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tree_pins = {name: ops.sha(root / name) for name in ops.TREE_DELTA_PATHS}
    assert ops.validate_deployed_git_tree(root, current, tree_pins)


def test_real_module_entry_defines_cuda_helpers_before_invoking_main():
    required = {"own_gpu_mib", "measured_vram", "inspect_cuda_context", "sample_cuda_memory",
                "initial_vram_declaration", "build_plan"}
    observed = set()

    class StopAtMain(Exception):
        pass

    def trace(frame, event, arg):
        if event == "call" and frame.f_code.co_name == "main" and frame.f_globals.get("__name__") == "__main__":
            observed.update(frame.f_globals)
            raise StopAtMain()
        return trace

    prior = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(StopAtMain):
            runpy.run_path(str(Path(ops.__file__).resolve()), run_name="__main__")
    finally:
        sys.settrace(prior)
    assert required <= observed


def test_actual_module_help_and_invalid_command_have_real_exit_statuses():
    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    env = {**__import__("os").environ, "PYTHONPATH": str(root / "src") + ":" + str(root / "scripts"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    ops_help = subprocess.run([python, "-B", "-m", "vkgr.p_watch_positive_ops", "--help"],
                              cwd=root, env=env, capture_output=True, text=True)
    assert ops_help.returncode == 0 and "preflight" in ops_help.stdout and "cuda-preview" in ops_help.stdout
    runner_help = subprocess.run([python, "-B", "-m", "vkgr.p_watch_positive_run", "--help"],
                                 cwd=root, env=env, capture_output=True, text=True)
    assert runner_help.returncode == 0 and "train" in runner_help.stdout and "verify" in runner_help.stdout
    invalid = subprocess.run([python, "-B", "-m", "vkgr.p_watch_positive_ops", "invalid"],
                             cwd=root, env=env, capture_output=True, text=True)
    assert invalid.returncode != 0


def test_historically_dirty_m1_controls_are_accepted_only_when_all_51_blobs_match(tmp_path):
    import copy
    import hashlib
    import subprocess
    root = Path(__file__).resolve().parents[1]
    worktrees = subprocess.check_output(["git", "-C", str(root), "worktree", "list", "--porcelain"], text=True)
    candidates = [Path(line.removeprefix("worktree ")) for line in worktrees.splitlines() if line.startswith("worktree ")]
    main_repo = next((path for path in candidates if (path / "cache/pgr01-p-recovery-v1/main05/p-watch-original-control-dirty-byte-audit-20261019.json").is_file()), None)
    assert main_repo is not None
    audit = json.loads((main_repo / "cache/pgr01-p-recovery-v1/main05/p-watch-original-control-dirty-byte-audit-20261019.json").read_text())
    assert audit["status"] == "old51_all_exact_git_blobs_three_seeds"
    assert all(row == {"git_dirty": True, "mismatched_old_git_blobs": [], "seed": seed, "source_files": 51}
               for row, seed in zip(audit["records"], ops.SEEDS))
    refs = {}
    original_manifest_bytes = {}
    for seed in ops.SEEDS:
        original = main_repo / f"cache/pgr01-p-recovery-v1/main05/p-watch-training-control-s{seed}-manifest-20261019.json"
        original_manifest_bytes[seed] = original.read_bytes()
        manifest = tmp_path / "runs" / f"p-seqm1-mean-s{seed}" / "out" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(original_manifest_bytes[seed])
        refs[seed] = {"path": str(manifest), "sha256": ops.CONTROL_MANIFEST_SHA256[seed]}
    controls = ops.read_control_manifests(refs)
    assert len(controls["training_files"]) == 51
    assert all(controls["manifests"][seed]["git_dirty"] is True for seed in ops.SEEDS)
    assert ops.validate_original_source_identity(root, controls, controls["training_files"]) is True
    bad_hash_refs = {**refs, 2024: {**refs[2024], "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="fixed M1 control path/SHA"):
        ops.read_control_manifests(bad_hash_refs)
    wrong_run = tmp_path / "runs" / "p-seqm1-mean-s2024-copy" / "out" / "manifest.json"
    wrong_run.parent.mkdir(parents=True)
    wrong_run.write_bytes(original_manifest_bytes[2024])
    wrong_run_refs = {**refs, 2024: {**refs[2024], "path": str(wrong_run)}}
    with pytest.raises(ValueError, match="M1 control manifest file/SHA"):
        ops.read_control_manifests(wrong_run_refs)
    wrong_seed = tmp_path / "runs" / "p-seqm1-mean-s2024" / "out" / "manifest.json"
    wrong_seed.write_bytes(original_manifest_bytes[2025])
    wrong_seed_refs = {**refs, 2024: {**refs[2024], "path": str(wrong_seed)}}
    with pytest.raises(ValueError, match="M1 control manifest file/SHA"):
        ops.read_control_manifests(wrong_seed_refs)
    for name, expected in controls["training_files"].items():
        blob = subprocess.check_output(["git", "-C", str(root), "show", ops.M1_SOURCE_COMMIT + ":" + name])
        assert hashlib.sha256(blob).hexdigest() == expected
    wrong_controls = copy.deepcopy(controls)
    wrong_original = wrong_controls["training_files"]
    first_source = next(iter(wrong_original))
    wrong_original[first_source] = "0" * 64
    with pytest.raises(ValueError, match="historically dirty M1 source bytes differ"):
        ops.validate_original_source_identity(root, wrong_controls, wrong_original)
    delta = {name: ops.sha(root / name) for name in ops.DELTA_PATHS}
    contract = {"root": str(root), "training_files": controls["training_files"],
        "deployed_files": {**controls["training_files"], **delta}, "pinned_delta": delta,
        "tree_files": {name: ops.sha(root / name) for name in ops.TREE_DELTA_PATHS},
        "runtime": controls["runtime"], "accepted_runtime": controls["runtime"],
        "p3c_data_channel": {},
        "python": sys.executable, "accepted_controls": controls}
    assert ops.validate_runtime(controls["runtime"], controls["runtime"])
    with pytest.raises(ValueError, match="closed p3c data-channel mapping"):
        ops.validate_source_contract(contract)
    for field in ("path", "git_sha256", "contract_sha256", "source_path"):
        invalid = dict(ops.P3C_DATA_CHANNEL)
        invalid[field] = "wrong"
        with pytest.raises(ValueError, match="closed p3c data-channel mapping"):
            ops.validate_source_contract({**contract, "p3c_data_channel": invalid})
    changed_old_source = copy.deepcopy(contract)
    old_path = next(iter(changed_old_source["training_files"]))
    changed_old_source["training_files"][old_path] = "0" * 64
    changed_old_source["deployed_files"][old_path] = "0" * 64
    changed_old_source["accepted_controls"]["training_files"][old_path] = "0" * 64
    with pytest.raises(ValueError):
        ops.validate_original_source_identity(root, controls, changed_old_source["training_files"])
    changed_runtime = copy.deepcopy(controls["runtime"])
    changed_runtime["lock_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Python/dependencies/lock"):
        ops.validate_runtime(controls["runtime"], changed_runtime)
