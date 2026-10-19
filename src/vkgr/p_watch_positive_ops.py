"""Finite P watch-positive engineering gates and queue-plan primitives."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import argparse
import random

RUN_IDS = {42: "p-watch-positive-s42", 2024: "p-watch-positive-s2024", 2025: "p-watch-positive-s2025"}
SEEDS = (42, 2024, 2025)
CONTROL_MANIFEST_SHA256 = {
    42: "bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e",
    2024: "8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8",
    2025: "783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb",
}
CONTROL_STOP_EPOCH = {42: 15, 2024: 10, 2025: 14}
M1_SOURCE_COMMIT = "0ca61bc35ad68bd79b86103b54e470445cbab07f"
STAGE_RUN_IDS = {
    "preflight": "p-watch-positive-preflight-v1",
    "cuda-preview": "p-watch-positive-cuda-preview-v1",
}
SCHEDULER_SOURCE_COMMIT = "41b4a4db2f4c84f16ee13df82775c60915a0e4d3"
SCHEDULER_SOURCE_FILES = {
    "__init__.py": "55a281f0f01a9532753d819789cc49d0ba47ad9e4cd52e437463d5f64af648a4",
    "admission.py": "3c0bfa4b218897327902077f7fee6fd75ddee90eedf767881cbef4cec0e8af93",
    "common.py": "a17995c5ed15efb4a928f5049d298d7509996480a8110ea7bb1a1f76231b3c5f",
    "launcher.py": "e6ba787a961eb76dfb241ce6ce37981fda94b5a3eea22729b4c4c730c476387f",
    "scheduler.py": "3c7e8fd8ddf865c6170aa226b709658bbf20048edb7b4c5992117ba4e3074062",
    "status.py": "fc7dd9ed67c7a74b3173341d64ab0dc095fa2268b43ab7b3e782d87c21cd19d9",
    "supervisor.py": "cb512105b2cb0a83de02627a1cb5fe04170831fad1d00e244307cfb4dfd392b3",
}
SOURCE_LEDGER_DELTA_PATHS = (
    "src/vkgr/p_watch_positive.py", "src/vkgr/p_watch_positive_epoch.py",
    "src/vkgr/p_watch_positive_nonempty.py",
    "src/vkgr/p_watch_positive_rows.py", "src/vkgr/p_watch_positive_run.py",
    "src/vkgr/p_watch_positive_ops.py",
)
TREE_DELTA_PATHS = (
    "configs/p_watch_positive/p-watch-positive-s2024.json",
    "configs/p_watch_positive/p-watch-positive-s2025.json",
    "configs/p_watch_positive/p-watch-positive-s42.json",
    "configs/p_watch_positive/p-watch-positive-train-only.json",
    "docs/superpowers/plans/2026-10-20-p-watch-implementation.md",
    "scripts/p_watch_positive_deploy.py",
    "scripts/p_watch_positive_driver.py",
    "scripts/p_watch_positive_engineering_queue.py",
    "scripts/p_watch_positive_stats.py",
    "scripts/p_watch_positive_stats_queue.py",
    "src/vkgr/p_watch_positive.py",
    "src/vkgr/p_watch_positive_epoch.py",
    "src/vkgr/p_watch_positive_nonempty.py",
    "src/vkgr/p_watch_positive_ops.py",
    "src/vkgr/p_watch_positive_rows.py",
    "src/vkgr/p_watch_positive_run.py",
    "tests/test_p_watch_positive_core.py",
    "tests/test_p_watch_positive_deploy_layout.py",
    "tests/test_p_watch_positive_driver.py",
    "tests/test_p_watch_positive_engineering_queue.py",
    "tests/test_p_watch_positive_epoch.py",
    "tests/test_p_watch_positive_ops.py",
    "tests/test_p_watch_positive_rows.py",
    "tests/test_p_watch_positive_run.py",
    "tests/test_p_watch_positive_stats.py",
    "tests/test_p_watch_positive_stats_queue.py",
)
DELTA_PATHS = SOURCE_LEDGER_DELTA_PATHS

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def _sha_map(value, role):
    if type(value) is not dict or not value:
        raise ValueError(role + " nonempty path-to-SHA mapping required")
    for name, pin in value.items():
        path = Path(name)
        if (type(name) is not str or path.is_absolute() or ".." in path.parts
                or type(pin) is not str or len(pin) != 64
                or any(char not in "0123456789abcdef" for char in pin)):
            raise ValueError(role + " contains an invalid path/SHA")
    return value

def run_id_for_seed(seed):
    if type(seed) is not int or seed not in RUN_IDS:
        raise ValueError("unregistered P watch-positive seed")
    return RUN_IDS[seed]

def validate_control_manifests(manifests):
    if type(manifests) is not dict or set(manifests) != set(SEEDS):
        raise ValueError("all three fixed M1-mean control manifests required")
    for seed in SEEDS:
        value = manifests[seed]
        selection = value.get("selection") if type(value) is dict else None
        if type(value) is not dict or value.get("sha256") != CONTROL_MANIFEST_SHA256[seed]:
            raise ValueError("fixed M1-mean control manifest SHA mismatch")
        if type(selection) is not dict or type(selection.get("epoch")) is not int:
            raise ValueError("M1 control stop epoch metadata required")
        if selection.get("max_epochs") != 100 or selection["epoch"] != CONTROL_STOP_EPOCH[seed] or selection["epoch"] >= 100:
            raise ValueError("M1 control max_epochs boundary changed")
    return True

def read_control_manifests(references):
    if type(references) is dict and set(references) == {str(s) for s in SEEDS}:
        references = {int(seed): value for seed, value in references.items()}
    if type(references) is not dict or set(references) != set(SEEDS):
        raise ValueError("three fixed M1 control references required")
    manifests = {}
    for seed in SEEDS:
        ref = references[seed]
        if type(ref) is not dict or set(ref) != {"path", "sha256"} or ref["sha256"] != CONTROL_MANIFEST_SHA256[seed]:
            raise ValueError("fixed M1 control path/SHA reference required")
        path = Path(ref["path"])
        expected_tail = ("runs", "p-seqm1-mean-s" + str(seed), "out", "manifest.json")
        if (not path.is_absolute() or ".." in path.parts or tuple(path.parts[-4:]) != expected_tail
                or any(parent.is_symlink() for parent in (path, *path.parents))
                or not path.is_file() or sha(path) != ref["sha256"]):
            raise ValueError("M1 control manifest file/SHA mismatch")
        value = json.loads(path.read_text())
        if (value.get("schema") != "p-seqm1-train-manifest-v1" or value.get("kind") != "p_seqm1_train"
                or value.get("status") != "complete" or value.get("config", {}).get("seed") != seed
                or value.get("config", {}).get("arm") != "mean"
                or value.get("run_id") not in (None, "p-seqm1-mean-s" + str(seed))):
            raise ValueError("complete same-seed M1 control manifest required")
        binding = value.get("binding")
        source = binding.get("source") if type(binding) is dict else None
        if type(source) is not dict or type(source.get("files")) is not dict:
            raise ValueError("M1 control manifest source file ledger required")
        _sha_map(source["files"], "M1 control source ledger")
        runtime = {"python": source.get("python"), "exact_dependencies": source.get("dependencies"),
                   "lock_sha256": source.get("lock_sha")}
        manifests[seed] = {"sha256": ref["sha256"], "selection": value.get("selection"),
                           "source_files": source["files"], "runtime": runtime, "git_commit": source.get("git_commit"),
                           "git_dirty": source.get("git_dirty"), "manifest": value}
    validate_control_manifests(manifests)
    first = manifests[SEEDS[0]]
    for seed in SEEDS[1:]:
        if (manifests[seed]["source_files"] != first["source_files"] or manifests[seed]["runtime"] != first["runtime"]
                or manifests[seed]["git_commit"] != first["git_commit"] or manifests[seed]["git_dirty"] != first["git_dirty"]):
            raise ValueError("three M1 controls do not share one source ledger/runtime")
    return {"manifests": manifests, "training_files": first["source_files"], "runtime": first["runtime"],
            "base_commit": first["git_commit"]}

def validate_source_contract(contract):
    required = {"root", "training_files", "deployed_files", "pinned_delta", "tree_files", "runtime",
                "accepted_runtime", "python", "accepted_controls"}
    if type(contract) is not dict or set(contract) != required:
        raise ValueError("closed deployed-source/runtime contract required")
    root = Path(contract["root"])
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("deployed source root required")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"], text=True).strip():
        raise ValueError("deployed source tree must be clean")
    controls = contract.get("accepted_controls")
    if type(controls) is not dict or contract["training_files"] != controls.get("training_files") or contract["accepted_runtime"] != controls.get("runtime"):
        raise ValueError("source contract differs from original M1 manifest ledger/runtime")
    validate_pinned_delta(root, contract["training_files"], contract["deployed_files"], contract["pinned_delta"])
    original_dirty = validate_original_source_identity(root, controls, contract["training_files"])
    validate_deployed_git_tree(root, commit, contract["tree_files"])
    actual_runtime = subprocess.run([contract["python"], "-B", "-c", "import importlib.metadata,json,platform;print(json.dumps({'python':platform.python_version(),'exact_dependencies':{n:importlib.metadata.version(n) for n in ('numpy','torch','pyarrow','PyYAML')}},sort_keys=True))"],
                                    check=True, capture_output=True, text=True, timeout=30)
    runtime = json.loads(actual_runtime.stdout)
    lock = root / "uv.lock"
    runtime["lock_sha256"] = sha(lock)
    if runtime != contract["runtime"] or runtime != contract["accepted_runtime"]:
        raise ValueError("actual Python/dependencies/lock differ from accepted M1 controls")
    return {"commit": commit, "runtime": runtime, "source_sha256": digest(contract["deployed_files"]),
            "tree_sha256": digest(contract["tree_files"]), "old_git_dirty_recorded": original_dirty}

def _validate_original_m1_ledger(root, training_files):
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "--name-only", M1_SOURCE_COMMIT,
        "--", ".python-version", "pyproject.toml", "uv.lock", "src/vkgr"], text=True).splitlines()
    expected = {name for name in tracked if name in (".python-version", "pyproject.toml", "uv.lock")
                or (name.startswith("src/vkgr/") and "/" not in name[len("src/vkgr/"):] and name.endswith(".py"))}
    if len(expected) != 51 or set(training_files) != expected:
        raise ValueError("original M1 source ledger must equal its exact 51 tracked source paths")
    for name, pin in training_files.items():
        blob = subprocess.check_output(["git", "-C", str(root), "show", M1_SOURCE_COMMIT + ":" + name])
        if hashlib.sha256(blob).hexdigest() != pin:
            raise ValueError("historically dirty M1 source bytes differ from their original Git blobs: " + name)

def validate_original_source_identity(root, controls, training_files):
    if type(controls) is not dict or controls.get("base_commit") != M1_SOURCE_COMMIT:
        raise ValueError("fixed original M1 source commit required")
    if training_files != controls.get("training_files"):
        raise ValueError("original M1 source ledger differs from accepted manifests")
    manifests = controls.get("manifests")
    if type(manifests) is not dict or set(manifests) != set(SEEDS):
        raise ValueError("all three accepted M1 source manifests required")
    dirty_values = [manifests[seed].get("git_dirty") for seed in SEEDS]
    if type(dirty_values[0]) is not bool or any(value is not dirty_values[0] for value in dirty_values):
        raise ValueError("original M1 Git dirty metadata must be preserved consistently")
    _validate_original_m1_ledger(root, training_files)
    return dirty_values[0]

def validate_deployed_git_tree(root, current_commit, tree_files):
    if type(tree_files) is not dict or set(tree_files) != set(TREE_DELTA_PATHS):
        raise ValueError("deployed Git tree must equal the exact 26-path P project delta")
    for name, pin in tree_files.items():
        path = Path(name)
        if (path.is_absolute() or ".." in path.parts or type(pin) is not str or len(pin) != 64
                or any(char not in "0123456789abcdef" for char in pin)):
            raise ValueError("unsafe Git tree delta path/SHA")
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--name-status", "--no-renames", M1_SOURCE_COMMIT, current_commit], text=True)
    rows = [row.split("\t", 1) for row in diff.splitlines()]
    if any(len(row) != 2 or row[0] != "A" for row in rows) or {row[1] for row in rows} != set(TREE_DELTA_PATHS):
        raise ValueError("deployed Git tree differs from the exact frozen 26-path P delta")
    for name, pin in tree_files.items():
        path = Path(root) / name
        if path.is_symlink() or not path.is_file() or sha(path) != pin:
            raise ValueError("deployed Git tree file SHA mismatch: " + name)
        blob = subprocess.check_output(["git", "-C", str(root), "show", current_commit + ":" + name])
        if hashlib.sha256(blob).hexdigest() != pin:
            raise ValueError("deployed Git tree blob SHA mismatch: " + name)
    return True

def declared_vram_mib(peak_mib):
    if type(peak_mib) not in (int, float) or not math.isfinite(peak_mib) or peak_mib < 0:
        raise ValueError("finite nonnegative CUDA peak required")
    return int(math.ceil(peak_mib * 1.15 / 256.0) * 256)

def parse_verified_stdout(text, run_id, manifest_sha256):
    matches = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if (type(value) is dict and value.get("status") == "verified"
                and value.get("run_id") == run_id and value.get("manifest_sha256") == manifest_sha256):
            matches.append(value)
    return len(matches) == 1

def validate_pinned_delta(root, training_files, deployed_files, delta_files):
    if type(training_files) is not dict or type(deployed_files) is not dict or type(delta_files) is not dict:
        raise ValueError("source ledgers must be maps")
    for ledger in (training_files, deployed_files, delta_files):
        for name, pin in ledger.items():
            path = Path(name)
            if not isinstance(name, str) or path.is_absolute() or ".." in path.parts or type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
                raise ValueError("unsafe source ledger path or SHA")
    if set(delta_files) != set(DELTA_PATHS):
        raise ValueError("deployment must equal the exact six-path M1 source-ledger delta")
    if set(deployed_files) != set(training_files) | set(delta_files) or set(training_files) & set(delta_files):
        raise ValueError("deployment source ledger differs from training ledger plus pinned delta")
    expected = {**training_files, **delta_files}
    if deployed_files != expected:
        raise ValueError("deployed source SHA map differs from pinned source delta")
    for name, pin in deployed_files.items():
        path = Path(root) / name
        if path.is_symlink() or not path.is_file() or sha(path) != pin:
            raise ValueError("deployed source file SHA mismatch: " + name)
    return True

def validate_runtime(actual, expected):
    if type(actual) is not dict or actual != expected or set(actual) != {"python", "exact_dependencies", "lock_sha256"}:
        raise ValueError("Python/dependencies/lock identity differs from accepted M1 controls")
    return True

def validate_scheduler_source(scheduler_code):
    root = Path(scheduler_code)
    if root.name != "ops":
        raise ValueError("scheduler source ops directory required")
    package = root / "vkgr_ops"
    actual = {name: sha(package / name) for name in SCHEDULER_SOURCE_FILES
              if (package / name).is_file() and not (package / name).is_symlink()}
    if actual != SCHEDULER_SOURCE_FILES:
        raise ValueError("scheduler source differs from the pinned infra/04-sched-forecast package")
    return {"commit": SCHEDULER_SOURCE_COMMIT, "files": actual}

def _load_stage_request(req, required_stage):
    fields = {"stage", "stage_run_id", "deployment", "ops", "python", "inputs", "inputs_sha256",
              "configs", "stats_report", "stats_sha256", "approval", "approval_sha256", "controls",
              "source_contract", "output"}
    expected_fields = fields | ({"conservative_vram_mib"} if required_stage == "cuda-preview" else set())
    if type(req) is not dict or set(req) != expected_fields or req.get("stage") != required_stage:
        raise ValueError("closed P engineering stage request required")
    if req["stage_run_id"] != STAGE_RUN_IDS[required_stage] or os.environ.get("VKGR_RUN_ID") != req["stage_run_id"]:
        raise ValueError("P engineering stage must run as its scheduler-injected fixed run_id")
    for key in ("deployment", "ops", "python", "inputs", "configs", "stats_report", "approval", "output"):
        value = req[key]
        if key == "configs":
            continue
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or path.is_symlink():
            raise ValueError("private absolute non-symlink stage path required: " + key)
    out = Path(req["output"])
    if out.exists() or not out.is_relative_to(Path(req["ops"])):
        raise ValueError("unused preflight output path must be fresh under OPS")
    controls = read_control_manifests(req["controls"])
    contract = dict(req["source_contract"], accepted_controls=controls)
    validate_source_contract(contract)
    from .p_watch_positive_run import validate_stats_report, validate_training_approval
    if sha(req["inputs"]) != req["inputs_sha256"]:
        raise ValueError("original P input SHA changed")
    report = validate_stats_report(req["stats_report"], req["stats_sha256"])
    if report.get("engineering") is not False or report.get("synthetic") is not False:
        raise ValueError("formal real engineering=False P stats report required")
    if type(req["configs"]) is not dict or set(req["configs"]) != {str(s) for s in SEEDS}:
        raise ValueError("three fixed seed config references required")
    configs = {}
    for seed in SEEDS:
        ref = req["configs"][str(seed)]
        if type(ref) is not dict or set(ref) != {"path", "sha256"}:
            raise ValueError("seed config path/SHA reference required")
        from .retrieval_ablation_run import read_pinned
        value = read_pinned(ref["path"], ref["sha256"])
        from .p_watch_positive_run import validate_config
        validate_config(value)
        if value["seed"] != seed:
            raise ValueError("config seed differs from fixed seed identity")
        validate_training_approval(req["approval"], req["approval_sha256"], RUN_IDS[seed], seed, req["stats_sha256"])
        configs[seed] = value
    reference_config = configs[42]
    projected = {**{key: value for key, value in reference_config.items() if key not in ("schema", "task_id")},
                 "schema": "p-seq-m1-run-v1", "task_id": "P-SEQ-M1"}
    from .retrieval_seqm1_run import validate_config as validate_m1_config
    validate_m1_config(projected, False)
    return controls, report, configs, reference_config

def _real_stage_context(req, config):
    from types import SimpleNamespace
    from . import p_watch_positive_run as run
    args = SimpleNamespace(inputs=req["inputs"], inputs_sha=req["inputs_sha256"],
        config=req["configs"]["42"]["path"], config_sha=req["configs"]["42"]["sha256"],
        stats_report=req["stats_report"], stats_sha=req["stats_sha256"],
        approval=req["approval"], approval_sha=req["approval_sha256"],
        command="train", output=req["output"])
    target, check, data, inputs_binding, stats_report, auxiliary_rows, approval = run._real_context(args, config)
    return run, target, check, data, inputs_binding, stats_report, auxiliary_rows, approval

def _same_m1_context(controls, data, inputs_binding, config):
    from . import retrieval_feat02_run as feat02
    from .retrieval_b1_run import _digest
    rows, queries = feat02._view(data, config["history_len"])
    actual = {"rows_sha256": _digest(rows), "queries_sha256": _digest(queries),
        "catalog_sha256": _digest(data["catalog"]), "support_sha256": _digest(sorted(data["support"])),
        "content_sha256": hashlib.sha256(data["content"].cpu().numpy().tobytes()).hexdigest()}
    for seed in SEEDS:
        manifest = controls["manifests"][seed]["manifest"]
        binding = manifest.get("binding")
        if (type(binding) is not dict or binding.get("task") != "P-SEQ-M1"
                or manifest.get("config", {}).get("arm") != "mean"):
            raise ValueError("original same-seed M1-mean manifest binding required")
        for key, value in actual.items():
            if binding.get(key) != value:
                raise ValueError("real P context differs from original M1 " + key)
        original_inputs = binding.get("inputs")
        if type(original_inputs) is not dict or any(inputs_binding.get(key) != value for key, value in original_inputs.items()):
            raise ValueError("original M1 input bindings changed")
    if len(rows) != 86785 or len(queries) != 2520 or len(data["catalog"]) != 120415:
        raise ValueError("real P rows/query/catalog shape differs from accepted M1 controls")
    return rows, queries, actual

def _bind_filtered_auxiliary(run, data, stats_report, auxiliary_rows):
    diagnostics = data.get("watch_aux_exclusion")
    if type(diagnostics) is not dict:
        raise ValueError("formally bound empty-history exclusion diagnostics required")
    if (diagnostics.get("raw_N_aux") != stats_report.get("N_aux")
            or diagnostics.get("N_aux") != len(auxiliary_rows)
            or diagnostics.get("empty_history_excluded") != stats_report.get("empty_history_aux_count")
            or diagnostics.get("raw_bucket_counts_verified") is not True):
        raise ValueError("post-filter auxiliary pool differs from raw train-only preflight")
    binding = run.bind_auxiliary_pool(auxiliary_rows, stats_report, data["support"],
        data.get("durations_by_item", {}), diagnostics, formal=True)
    if (binding.get("raw_N_aux") != stats_report["N_aux"] or binding.get("N_aux") != len(auxiliary_rows)
            or binding.get("empty_history_excluded") != diagnostics["empty_history_excluded"]
            or binding.get("raw_pool_sha256") != diagnostics.get("raw_pool_sha256")
            or binding.get("filtered_pool_sha256") != diagnostics.get("filtered_pool_sha256")
            or binding.get("auxiliary_pool_sha256") != diagnostics.get("filtered_pool_sha256")):
        raise ValueError("formal filtered auxiliary binding does not reproduce its exclusion diagnostics")
    return binding, diagnostics

def formal_preflight(req):
    controls, report, configs, config = _load_stage_request(req, "preflight")
    run, target, check, data, inputs_binding, report, auxiliary_rows, approval = _real_stage_context(req, config)
    rows, queries, identities = _same_m1_context(controls, data, inputs_binding, config)
    pool, exclusion = _bind_filtered_auxiliary(run, data, report, auxiliary_rows)
    check()
    return {"schema": "p-watch-positive-engineering-preflight-v1", "stage_run_id": STAGE_RUN_IDS["preflight"],
        "status": "preflight_ok", "engineering": False, "synthetic": False, "validation_metrics_read": False,
        "control_manifest_sha256": {seed: controls["manifests"][seed]["sha256"] for seed in SEEDS},
        "stats_stdout_sha256": req["stats_sha256"], "approval_sha256": req["approval_sha256"],
        "source_identity": validate_source_contract(dict(req["source_contract"], accepted_controls=controls)),
        "original_m1_context": identities, "N_like": len(rows), "N_query": len(queries),
        "N_catalog": len(data["catalog"]), "N_aux": pool["N_aux"],
        "empty_history_aux_count": pool["empty_history_excluded"],
        "watch_aux_exclusion": exclusion,
        "inputs_binding_sha256": digest(inputs_binding), "auxiliary_pool_sha256": pool["auxiliary_pool_sha256"]}

def cuda_preview(req):
    controls, report, configs, config = _load_stage_request(req, "cuda-preview")
    import torch
    from . import p_watch_positive_run as run, retrieval_feat02_run as feat02, retrieval_seqm1_run as m1
    from .retrieval_b1_run import _digest
    from .retrieval_training import SamplerStats
    from .p_watch_positive_epoch import epoch_rows, epoch_tokens, frequency_table
    runtime = torch.cuda.is_available()
    if not runtime:
        raise RuntimeError("CUDA unavailable on the queue worker")
    loaded, target, check, data, inputs_binding, stats_report, auxiliary_rows, approval = _real_stage_context(req, config)
    rows, queries, identities = _same_m1_context(controls, data, inputs_binding, config)
    pool_binding, exclusion = _bind_filtered_auxiliary(run, data, stats_report, auxiliary_rows)
    from .retrieval_b1_run import execution_context
    execution_context("cuda")
    torch.cuda.init(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    memory = inspect_cuda_context(torch)
    torch.cuda.reset_peak_memory_stats()
    model = m1._model(data, config, "cuda")
    optimizer = m1.make_optimizer(model, config)
    sample_cuda_memory(torch, memory, "model-initialized")
    frequencies = frequency_table(rows, auxiliary_rows)
    tokens = epoch_tokens(len(rows), len(auxiliary_rows), 0)
    batch = epoch_rows(rows, auxiliary_rows, tokens[:min(config["batch_size"], len(tokens))])
    negative_rng = random.Random(f"p-b1-negative-v1:{config['seed']}")
    stats = SamplerStats()
    view = {**data, "rows": rows}
    train_record = feat02.train_step(model, optimizer, batch, view, stats, 0, config, negative_rng, None, frequencies)
    if train_record.get("loss") is None or train_record.get("gradient_norm") is None:
        raise ValueError("CUDA preview did not complete a real gradient update")
    sample_cuda_memory(torch, memory, "gradient-step")
    model.eval()
    device = next(model.parameters()).device
    histories, side = feat02._tensors2(batch[:1], data["idmap"], config["history_len"], config["feature"])
    with torch.no_grad():
        users = model.users(histories.to(device), side.to(device))
        catalog_indices = torch.tensor([data["idmap"][item] for item in data["catalog"]], dtype=torch.long, device=device)
        scores = users @ model.items(catalog_indices).T
        before = torch.topk(scores, min(100, scores.shape[1]), dim=1).indices
    sample_cuda_memory(torch, memory, "catalog-decode-top100")
    import io
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    state = torch.load(buffer, map_location="cpu", weights_only=True)
    del optimizer, model; torch.cuda.empty_cache()
    reloaded = m1._model(data, config, "cuda"); reloaded.load_state_dict(state); reloaded.eval()
    with torch.no_grad():
        after_users = reloaded.users(histories.to(device), side.to(device))
        after_scores = after_users @ reloaded.items(catalog_indices).T
        after = torch.topk(after_scores, min(100, after_scores.shape[1]), dim=1).indices
    if not torch.equal(before, after):
        raise ValueError("in-memory checkpoint reload changed decode top100")
    sample_cuda_memory(torch, memory, "checkpoint-reload-decode-top100")
    check()
    measured = measured_vram(memory, os.getpid())
    admission = initial_vram_declaration(req["conservative_vram_mib"])
    return {"schema": "p-watch-positive-cuda-preview-v1", "stage_run_id": STAGE_RUN_IDS["cuda-preview"],
        "status": "cuda_engineering_verified", "engineering": False, "synthetic": False,
        "validation_metrics_read": False, "gradient_step": len(batch), "decode_count": int(before.shape[1]),
        "reload_decode_equal": True, "original_m1_context": identities,
        "watch_aux_exclusion": exclusion, "pool_binding": pool_binding,
        "memory_measurement": memory, "measured_mib": measured,
        "declared_vram_mib": declared_vram_mib(measured), "not_measured": False,
        "admission_vram_mib": admission["declared_vram_mib"],
        "admission_not_measured": admission["not_measured"],
        "admission_peak_method": admission["peak_method"]}

def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("stage", choices=("preflight", "cuda-preview"))
    parser.add_argument("--request", required=True); parser.add_argument("--request-sha", required=True)
    args = parser.parse_args(argv)
    if sha(args.request) != args.request_sha:
        raise ValueError("engineering stage request SHA mismatch")
    req = json.loads(Path(args.request).read_text())
    if req.get("stage") != args.stage:
        raise ValueError("CLI/request stage mismatch")
    result = formal_preflight(req) if args.stage == "preflight" else cuda_preview(req)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0

def own_gpu_mib():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=10)
    values = [float(line.split(",")[1]) for line in result.stdout.splitlines()
              if line.split(",")[0].strip() == str(os.getpid())]
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
        raise ValueError("unique positive own-PID CUDA process measurement required")
    return values[0]

def measured_vram(memory, expected_pid=None):
    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative CUDA memory evidence required")
        return float(value)
    if type(memory) is not dict or type(memory.get("pid")) is not int or memory["pid"] <= 0:
        raise ValueError("positive CUDA worker PID required")
    if expected_pid is not None and memory["pid"] != expected_pid:
        raise ValueError("CUDA samples differ from their expected owning PID")
    if not memory.get("samples"):
        raise ValueError("CUDA gradient/decode/reload samples required")
    baseline = number(memory.get("context_process_mib"))
    reserved0 = number(memory.get("context_reserved_mib"))
    if baseline <= 0:
        raise ValueError("positive context baseline required")
    overhead = max(0.0, baseline - reserved0)
    allocator_peak = 0.0
    own_peak = baseline
    for sample in memory["samples"]:
        process = number(sample["own_process_mib"])
        reserved = number(sample["reserved_mib"])
        maximum = number(sample["max_reserved_mib"])
        allocated = number(sample["max_allocated_mib"])
        if process <= 0 or maximum < reserved or maximum < allocated:
            raise ValueError("inconsistent own-process allocator/CUDA samples")
        overhead = max(overhead, process - reserved)
        allocator_peak = max(allocator_peak, maximum, allocated)
        own_peak = max(own_peak, process)
    return max(own_peak, allocator_peak + overhead)

def inspect_cuda_context(torch):
    """Sample real own-process context and allocator peaks; caller performs actual model workload."""
    torch.cuda.synchronize()
    return {"pid": os.getpid(), "context_process_mib": own_gpu_mib(),
            "context_reserved_mib": torch.cuda.memory_reserved() / 2**20, "samples": []}

def sample_cuda_memory(torch, memory, stage):
    torch.cuda.synchronize()
    memory["samples"].append({"stage": stage, "own_process_mib": own_gpu_mib(),
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
    return memory["samples"][-1]

def initial_vram_declaration(conservative_upper_mib):
    if type(conservative_upper_mib) is not int or conservative_upper_mib <= 0:
        raise ValueError("positive conservative first-measurement upper bound required")
    return {"declared_vram_mib": int(math.ceil(conservative_upper_mib / 256) * 256),
            "not_measured": True, "peak_method": "conservative_upper_bound"}

def build_plan(configs, common, vram):
    if type(configs) is not dict or set(configs) != set(SEEDS):
        raise ValueError("exactly the frozen three seed configs required")
    entries = []
    for seed in SEEDS:
        rid = run_id_for_seed(seed)
        config = configs[seed]
        if config.get("seed") != seed or config.get("max_epochs") != 100:
            raise ValueError("frozen P config seed/max_epochs mismatch")
        entries.append({"run_id": rid, "seed": seed, "config": config, "common": common,
                        "vram": vram, "stages": ["train", "verify"], "verify_run_id": rid + "-verify"})
    return {"schema": "p-watch-positive-plan-v1", "entries": entries,
            "run_ids": [entry["run_id"] for entry in entries], "sha256": digest(entries)}

if __name__ == "__main__":
    raise SystemExit(main())
