import pytest
import json
import runpy
import sys
import subprocess
from pathlib import Path

from vkgr import p_watch_positive_ops as ops


def test_fixed_training_run_ids_are_exactly_the_registered_three():
    assert ops.RUN_IDS == {42: "p-watch-positive-s42", 2024: "p-watch-positive-s2024", 2025: "p-watch-positive-s2025"}
    with pytest.raises(ValueError):
        ops.run_id_for_seed(7)


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
        "python": sys.executable, "accepted_controls": controls}
    assert ops.validate_runtime(controls["runtime"], controls["runtime"])
    with pytest.raises(ValueError, match="actual Python/dependencies/lock"):
        ops.validate_source_contract(contract)
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
