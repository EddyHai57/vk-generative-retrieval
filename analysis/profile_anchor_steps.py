"""Short, quality-blind timing profile for the frozen G and P anchors."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import threading
import time
from types import MethodType
from types import SimpleNamespace


PROFILE_ROOT = Path(__file__).resolve().parents[1]
if str(PROFILE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROFILE_ROOT))

WARMUP_STEPS = 10
MEASURED_STEPS = 100
DIAGNOSTIC_STEPS = 10
TOTAL_STEPS = WARMUP_STEPS + MEASURED_STEPS + DIAGNOSTIC_STEPS
P_SOURCE_COMMIT = "0ca61bc35ad68bd79b86103b54e470445cbab07f"
SOURCE_PINS_FILE = PROFILE_ROOT / "configs/anchor-profile/source-pins.json"
SOURCE_PINS_FILE_SHA256 = "1227ff9a3b6dbe7cb6f6e2b05be154a12e3a6d224bf980d4df33380b05e4747b"
G_CONFIG = "configs/sid_structure/g-structure-a1-s42.json"
P_CONFIG = "configs/seqm1/p-seqm1-mean-s42.json"
LOCK_SHA256 = "8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063"
RUNTIME_PYTHON = "3.11.16"
RUNTIME_NUMPY = "1.26.4"
RUNTIME_PYARROW = "19.0.1"
RUNTIME_PYYAML = "6.0.2"
RUNTIME_TORCH = "2.11.0"


def profile_schedule(*, warmup=WARMUP_STEPS, measured=MEASURED_STEPS, diagnostic=DIAGNOSTIC_STEPS):
    counts = (warmup, measured, diagnostic)
    if any(type(value) is not int or value < 1 for value in counts):
        raise ValueError("positive integer profile step counts required")
    if sum(counts) > 500:
        raise ValueError("profile cap is 500 effective steps")
    if counts != (WARMUP_STEPS, MEASURED_STEPS, DIAGNOSTIC_STEPS):
        raise ValueError("frozen anchor profile schedule required")
    return counts


def run_steps(step, *, warmup=WARMUP_STEPS, measured=MEASURED_STEPS, diagnostic=DIAGNOSTIC_STEPS):
    counts = (warmup, measured, diagnostic)
    if any(type(value) is not int or value < 1 for value in counts) or sum(counts) > 500:
        raise ValueError("positive integer step schedule within the 500-step cap required")
    for index in range(warmup + measured + diagnostic):
        step(index)
    return {"warmup_steps": warmup, "measured_steps": measured, "diagnostic_steps": diagnostic}


@contextmanager
def stage_hooks(model, optimizer, *, tensor_builders=()):
    """Annotate original operations and collect wall spans without changing their arguments."""
    import torch

    totals = {"tensor_build": 0.0, "h2d": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    original_to = torch.Tensor.to
    original_backward = torch.Tensor.backward
    original_forward = model.forward
    original_step = optimizer.step
    originals = []
    component_forwards = []
    patched = {"tensor_to": False, "backward": False, "forward": False, "optimizer": False}

    def timed(stage, callback):
        def invoke(*args, **kwargs):
            started = time.perf_counter()
            with torch.profiler.record_function("profile_stage::" + stage):
                try:
                    return callback(*args, **kwargs)
                finally:
                    totals[stage] += time.perf_counter() - started
        return invoke

    def tensor_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if not isinstance(device, (str, torch.device)) or str(device).startswith("cpu"):
            return original_to(tensor, *args, **kwargs)
        return timed("h2d", lambda *a, **kw: original_to(tensor, *a, **kw))(*args, **kwargs)

    def model_forward(this, *args, **kwargs):
        return timed("forward", lambda *a, **kw: original_forward(*a, **kw))(*args, **kwargs)

    def backward(tensor, *args, **kwargs):
        return timed("backward", lambda *a, **kw: original_backward(tensor, *a, **kw))(*args, **kwargs)

    def optimizer_step(this, *args, **kwargs):
        return timed("optimizer", lambda *a, **kw: original_step(*a, **kw))(*args, **kwargs)

    def make_builder_wrapper(callback):
        return timed("tensor_build", callback)

    try:
        torch.Tensor.to = tensor_to
        patched["tensor_to"] = True
        torch.Tensor.backward = backward
        patched["backward"] = True
        model.forward = MethodType(model_forward, model)
        patched["forward"] = True
        optimizer.step = MethodType(optimizer_step, optimizer)
        patched["optimizer"] = True
        for name in ("users", "items"):
            callback = getattr(model, name, None)
            if callable(callback):
                component_forwards.append((name, callback))
                setattr(model, name, MethodType(lambda this, *a, __callback=callback, **kw:
                    timed("forward", lambda *x, **y: __callback(*x, **y))(*a, **kw), model))
        for module, name in tensor_builders:
            callback = getattr(module, name)
            originals.append((module, name, callback))
            setattr(module, name, make_builder_wrapper(callback))
        yield totals
    finally:
        for module, name, callback in reversed(originals):
            setattr(module, name, callback)
        for name, callback in component_forwards:
            setattr(model, name, callback)
        if patched["tensor_to"]:
            torch.Tensor.to = original_to
        if patched["backward"]:
            torch.Tensor.backward = original_backward
        if patched["forward"]:
            model.forward = original_forward
        if patched["optimizer"]:
            optimizer.step = original_step


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_fresh(path, label):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label} must be a fresh, nonexistent path")
    return path


def _source_pins():
    if _sha(SOURCE_PINS_FILE) != SOURCE_PINS_FILE_SHA256:
        raise ValueError("fixed anchor pins file mismatch")
    value = _read_json(SOURCE_PINS_FILE)
    if type(value) is not dict or value.get("schema") != "anchor-profile-pins-v1" or set(value) != {"schema", "g", "p"}:
        raise ValueError("fixed anchor pins schema mismatch")
    return value


def _p3c_profile_identity():
    pins = _source_pins()["p"]
    return {"p3c_git_blob_sha256": pins["p3c_git_sha256"],
            "p3c_data_channel_sha256": pins["p3c_data_sha256"]}


def _source_identity(root, branch):
    pins = _source_pins()[branch]
    expected_commit = pins["source_commit"]
    root = Path(root).resolve(strict=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"], text=True)
    p3c = root / "docs/p3c_validation.md"
    if branch == "p":
        git_blob = subprocess.check_output(["git", "-C", str(root), "show", actual + ":docs/p3c_validation.md"])
        git_sha = hashlib.sha256(git_blob).hexdigest()
        dirty_paths = [line[3:] for line in dirty.splitlines()]
        data_sha = _sha(p3c) if p3c.is_file() else None
        if p3c.is_symlink() or git_sha != pins["p3c_git_sha256"] or dirty_paths != ["docs/p3c_validation.md"] or data_sha != pins["p3c_data_sha256"]:
            raise ValueError("P3C data-channel override must be the sole exact registered dirty path")
    elif dirty:
        raise ValueError("source root must be clean")
    if actual != expected_commit:
        raise ValueError("source root must match the expected clean fixed commit")
    if _sha(root / "uv.lock") != LOCK_SHA256 or (root / ".python-version").read_text().strip() != "3.11":
        raise ValueError("source lock or Python minor version differs from the registered runtime")
    return actual


def _add_source(root):
    root = str(Path(root).resolve(strict=True))
    source = str(Path(root, "src"))
    sys.path.insert(0, source)
    return root


def _source_file(root, value):
    path = Path(value)
    return path.resolve(strict=True) if path.is_absolute() else (Path(root) / path).resolve(strict=True)


def _validate_source_ledger(root, branch):
    pins = _source_pins()[branch]
    files = pins["source_files"]
    if not files or type(files) is not dict:
        raise ValueError("fixed anchor source ledger required")
    payload = json.dumps(files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    json_sha = hashlib.sha256(payload).hexdigest()
    expected_json_sha = pins["source_ledger_json_sha256"]
    if json_sha != expected_json_sha:
        raise ValueError("fixed source ledger digest mismatch")
    root = Path(root)
    actual_files = {}
    for rel, expected in files.items():
        path = Path(rel)
        if type(rel) is not str or path.is_absolute() or ".." in path.parts or "." in path.parts or path.as_posix() != rel or len(expected) != 64:
            raise ValueError("source ledger path or digest invalid")
        if (root / path).is_symlink():
            raise ValueError("source ledger symlink forbidden")
        actual = _sha(root / path)
        if actual != expected:
            raise ValueError("anchor source ledger file mismatch")
        actual_files[rel] = actual
    if actual_files != files:
        raise ValueError("source ledger member set mismatch")
    if branch == "g":
        identity_sha = hashlib.sha256((payload.decode() + "\n").encode()).hexdigest()
    else:
        order = pins.get("source_order")
        if type(order) is not list or len(order) != len(actual_files) or set(order) != set(actual_files):
            raise ValueError("P original source ledger order mismatch")
        identity_sha = hashlib.sha256("".join(f"{name}\0{actual_files[name]}\n" for name in order).encode()).hexdigest()
    if identity_sha != pins["source_ledger_sha256"]:
        raise ValueError("anchor source identity digest mismatch")
    return identity_sha


def _absolute_path(value, label, *, must_exist):
    if not isinstance(value, str) or not value:
        raise ValueError(label + " path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(label + " must be absolute and normalized")
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(label + " path cannot traverse symlinks")
    if must_exist:
        try:
            path.resolve(strict=True)
        except (OSError, RuntimeError):
            raise ValueError(label + " path must exist") from None
    return path


def _make_run_receipt_identity(args):
    run_id = _run_identity(args.output, args.scratch)
    return {**_profile_identity(args.request_path), "run_id": run_id}


def _validate_ambient_environment(branch, output, run_id):
    expected = _source_pins()[branch]["runtime_environment"]
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise ValueError("numerical environment differs from the registered anchor")
    if os.environ.get("VKGR_RUN_ID") != run_id or os.environ.get("VKGR_PROGRESS_PATH") != str(Path(output).parent / "progress.json"):
        raise ValueError("launcher runtime run or progress path differs from the bound run")
    forbidden = {
        "CUDA_VISIBLE_DEVICES", "LD_PRELOAD", "NVIDIA_TF32_OVERRIDE", "PYTORCH_CUDA_ALLOC_CONF",
        "CUDNN_DETERMINISTIC", "CUDNN_BENCHMARK", "OMP_DYNAMIC", "OMP_MAX_ACTIVE_LEVELS", "OMP_SCHEDULE",
        "MKL_DYNAMIC", "PYTHONPATH", "PYTHONHOME", "PYTHONHASHSEED",
    }
    if any(os.environ.get(name) for name in forbidden):
        raise ValueError("unregistered numerical environment override")
    if any(name.startswith("VKGR_") and name not in ("VKGR_RUN_ID", "VKGR_PROGRESS_PATH") for name in os.environ):
        raise ValueError("unregistered VKGR environment override")


def _validate_request_paths(args):
    _absolute_path(args.source_root, "source root", must_exist=True)
    _absolute_path(args.output, "output root", must_exist=False)
    _absolute_path(args.scratch, "scratch root", must_exist=False)
    if args.branch == "g":
        _absolute_path(args.data_root, "G data root", must_exist=True)
        roles = ("prepared", "history_manifest", "sid_manifest", "content", "input_manifest", "vectors")
    else:
        roles = ("inputs",)
    for role in roles:
        _absolute_path(getattr(args, role), role, must_exist=True)
    _validate_output_disjoint(args)


def _read_p_input_spec(path, expected_sha):
    path = _absolute_path(str(path), "P input spec", must_exist=True)
    if _sha(path) != expected_sha:
        raise ValueError("P input spec bytes differ from the registered pin")
    value = _read_json(path)
    fields = {"attr_manifest", "attr_sha256", "ctx_manifest", "ctx_sha256", "disk_cap_bytes",
              "history_manifest", "history_sha256", "holdout", "items_metadata_sha256", "prepared",
              "raw_root", "schema", "side_manifest", "side_sha256", "test_gate"}
    if type(value) is not dict or set(value) != fields or value.get("schema") != "p-feat02-inputs-v1":
        raise ValueError("registered P input spec schema required")
    for name in ("prepared", "holdout", "raw_root"):
        _absolute_path(value[name], "P " + name, must_exist=False)
    for name in ("history_manifest", "side_manifest", "ctx_manifest", "attr_manifest"):
        if name in ("history_manifest", "side_manifest") or value[name] is not None:
            _absolute_path(value[name], "P " + name, must_exist=False)
    return value


def _input_parent(path):
    candidate = Path(path)
    if candidate.exists() and candidate.is_dir():
        return candidate.resolve(strict=False)
    return candidate.parent.resolve(strict=False)


def _protected_path_roots(args):
    roots = [Path(args.source_root).resolve(strict=False)]
    if args.branch == "g":
        roots.append(Path(args.data_root).resolve(strict=False))
        roles = ("prepared", "history_manifest", "sid_manifest", "content", "input_manifest", "vectors")
        roots.extend(_input_parent(getattr(args, role)) for role in roles)
    else:
        spec = _read_p_input_spec(args.inputs, args.inputs_sha)
        roots.append(Path(args.inputs).parent.resolve(strict=False))
        roots.append(Path(spec["raw_root"]).resolve(strict=False))
        for name in ("prepared", "holdout", "raw_root"):
            roots.append(Path(spec[name]).resolve(strict=False))
        for name in ("history_manifest", "side_manifest", "ctx_manifest", "attr_manifest"):
            if spec[name] is not None:
                roots.append(Path(spec[name]).parent.resolve(strict=False))
    return roots


def _validate_output_disjoint(args):
    outputs = [_absolute_path(args.output, "output root", must_exist=False).resolve(strict=False),
               _absolute_path(args.scratch, "scratch root", must_exist=False).resolve(strict=False)]
    if outputs[0] == outputs[1] or outputs[0] in outputs[1].parents or outputs[1] in outputs[0].parents:
        raise ValueError("output and scratch roots must be separate, non-nested paths")
    run_roots = outputs + [Path(args.output).parent.resolve(strict=False),
                          Path(args.scratch).parent.resolve(strict=False)]
    protected_roots = _protected_path_roots(args)
    for candidate in run_roots:
        for protected in protected_roots:
            if candidate == protected or candidate in protected.parents or protected in candidate.parents:
                raise ValueError("new profile run paths must be disjoint from source and approved input roots")


@contextmanager
def _pinned_g_data_root(path):
    previous = os.environ.get("VKGR_DATA_ROOT")
    if previous is not None:
        raise ValueError("ambient VKGR_DATA_ROOT override is forbidden")
    os.environ["VKGR_DATA_ROOT"] = str(path)
    try:
        yield
    finally:
        os.environ.pop("VKGR_DATA_ROOT", None)


def _run_identity(output, scratch):
    run_id = os.environ.get("VKGR_RUN_ID")
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in run_id):
        raise ValueError("launcher-injected run ID required")
    output, scratch = Path(output), Path(scratch)
    if output.name != "out" or scratch.name != "scratch" or output.parent.name != run_id or scratch.parent != output.parent \
            or output.parent.parent.name != "runs" or not output.parent.is_dir():
        raise ValueError("output and scratch must be the launcher-owned run subdirectories")
    return run_id


def _profile_identity(request_path):
    profile_commit = subprocess.check_output(["git", "-C", str(PROFILE_ROOT), "rev-parse", "HEAD"], text=True).strip()
    profile_dirty = subprocess.check_output(["git", "-C", str(PROFILE_ROOT), "status", "--porcelain=v1", "-z", "--untracked-files=all"])
    override = None
    if profile_dirty:
        if profile_dirty != b" M docs/p3c_validation.md\0":
            raise ValueError("profiler source tree has an unregistered dirty path")
        pins = _source_pins()["p"]
        git_blob = subprocess.check_output(["git", "-C", str(PROFILE_ROOT), "show", profile_commit + ":docs/p3c_validation.md"])
        p3c = PROFILE_ROOT / "docs/p3c_validation.md"
        ops_copy = Path(request_path).parent / "p3c_validation.md"
        if (hashlib.sha256(git_blob).hexdigest() != pins["p3c_git_sha256"]
                or p3c.is_symlink() or not p3c.is_file() or _sha(p3c) != pins["p3c_data_sha256"]
                or ops_copy.is_symlink() or not ops_copy.is_file() or _sha(ops_copy) != pins["p3c_data_sha256"]):
            raise ValueError("profiler P3C override differs from the fixed data-channel identity")
        override = {"path": "docs/p3c_validation.md", "base_sha256": pins["p3c_git_sha256"],
                    "sha256": pins["p3c_data_sha256"], "ops_copy_sha256": pins["p3c_data_sha256"]}
    return {"run_id": os.environ["VKGR_RUN_ID"], "request_sha256": _sha(request_path),
            "profiler_script_sha256": _sha(__file__), "profiler_source_commit": profile_commit,
            "profiler_source_override": override}


def own_gpu_mib():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=10)
    values = [float(line.split(",")[1]) for line in result.stdout.splitlines()
              if len(line.split(",")) == 2 and line.split(",")[0].strip() == str(os.getpid())]
    if len(values) != 1 or not math.isfinite(values[0]) or values[0] <= 0:
        raise ValueError("unique finite profile-process GPU memory required")
    return values[0]


def measured_vram(memory):
    def number(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative context-inclusive memory evidence required")
        return value
    if type(memory.get("pid")) is not int or memory["pid"] != os.getpid() or not memory.get("samples"):
        raise ValueError("profile PID and nonempty own-process GPU samples required")
    baseline = number(memory["context_process_mib"])
    reserved0 = number(memory["context_reserved_mib"])
    if baseline <= 0:
        raise ValueError("measured own CUDA context baseline required")
    overhead = max(0.0, baseline - reserved0)
    allocator_peak = 0.0
    own_peak = baseline
    for row in memory["samples"]:
        process = number(row["own_process_mib"])
        reserved = number(row["reserved_mib"])
        maximum = number(row["max_reserved_mib"])
        allocated = number(row["max_allocated_mib"])
        if process <= 0 or maximum < reserved or maximum < allocated:
            raise ValueError("inconsistent process/allocator memory sample")
        overhead = max(overhead, process - reserved)
        allocator_peak = max(allocator_peak, maximum, allocated)
        own_peak = max(own_peak, process)
    return max(own_peak, allocator_peak + overhead)


def declared_vram(measured_mib):
    if type(measured_mib) not in (int, float) or not math.isfinite(measured_mib) or measured_mib <= 0:
        raise ValueError("positive finite context-inclusive memory required")
    return math.ceil(measured_mib * 1.15 / 256) * 256


def _runtime_identity(torch):
    import numpy
    import pyarrow
    import yaml
    values = {"python": platform.python_version(), "numpy": numpy.__version__, "pyarrow": pyarrow.__version__,
              "pyyaml": yaml.__version__, "torch": torch.__version__.split("+", 1)[0],
              "torch_build": torch.__version__, "cuda": torch.version.cuda}
    expected = {"python": RUNTIME_PYTHON, "numpy": RUNTIME_NUMPY, "pyarrow": RUNTIME_PYARROW,
                "pyyaml": RUNTIME_PYYAML, "torch": RUNTIME_TORCH, "torch_build": "2.11.0+cu128", "cuda": "12.8"}
    if values != expected:
        raise ValueError("Python/dependency/CUDA runtime differs from the registered anchor runtime")
    return {**values, "lock_sha256": LOCK_SHA256}


def _load_g(args):
    import torch
    root = _add_source(args.source_root)
    source_commit = _source_identity(root, "g")
    source_ledger_sha = _validate_source_ledger(root, "g")
    runtime = _runtime_identity(torch)
    config_path = _source_file(root, args.config)
    try:
        config_rel = config_path.relative_to(Path(root)).as_posix()
    except ValueError:
        raise ValueError("G config must be inside the fixed source root") from None
    if config_rel != G_CONFIG or args.config_sha != _source_pins()["g"]["config_sha256"] or _sha(config_path) != args.config_sha:
        raise ValueError("G anchor config path or SHA mismatch")
    from vkgr import generative_data as gd
    from vkgr import generative_g6_model as gmodel
    from vkgr import generative_training as gt
    from vkgr import sid_depth as sd
    from vkgr import sid_structure_a1_run as runner
    config = gd._json(gd._checked_file(config_path, args.config_sha))
    scratch = _require_fresh(args.scratch, "scratch")
    scratch.parent.mkdir(parents=True, exist_ok=True)
    data_root = _absolute_path(args.data_root, "G data root", must_exist=True).resolve(strict=True)
    source_root = Path(root).resolve(strict=True)
    if data_root == source_root or data_root in source_root.parents or source_root in data_root.parents:
        raise ValueError("G code source and historical data channel must be separate trees")
    with _pinned_g_data_root(data_root):
        context, engineering, sid = runner._contexts(Path(root), Path(args.prepared), args.prepared_sha,
            Path(args.history_manifest), args.history_sha, Path(args.sid_manifest), args.sid_sha,
            Path(args.content), args.content_sha, config, scratch, input_manifest=Path(args.input_manifest),
            input_sha=args.input_sha, vectors_path=Path(args.vectors), vectors_sha=args.vectors_sha)
    if engineering is not False or context.binding.get("synthetic") is not False:
        raise ValueError("G real engineering=False input required")
    runtime = runner._runtime(False)
    if len(context.examples) < TOTAL_STEPS * 512:
        raise ValueError("G train rows do not cover the fixed profile before epoch end")
    def setup():
        runner.gr._seed(config["seed"])
        model = runner.make_model_arch_d(context.codec, config, device=runtime["device"])
        optimizer = runner.ga.make_optimizer_arch(model, config)
        tokens = sd.SIDTokensLD(context.codec, config["history_len"])
        cursor = gt.EpochCursor(config["seed"], len(context.examples))
        def step():
            indices = cursor.next_batch(512)
            started = time.perf_counter()
            rows = context.examples.rows(indices)
            batch_seconds = time.perf_counter() - started
            runner.train_batch_d(model, optimizer, rows, context.codec, tokens, config, runtime["device"])
            return batch_seconds
        return model, optimizer, step, ((gmodel, "sid_targets_l"),)
    runtime_summary = {key: runtime[key] for key in ("mode", "device", "deterministic_algorithms", "allow_tf32", "cuda_version", "gpu_name", "threads")}
    details = {"source_commit": source_commit, "source_ledger_sha256": source_ledger_sha, "runtime": runtime_summary,
               "config_sha256": args.config_sha,
               "prepared_sha256": args.prepared_sha, "history_sha256": args.history_sha, "sid_sha256": args.sid_sha,
               "content_sha256": args.content_sha, "input_manifest_sha256": args.input_sha, "vectors_sha256": args.vectors_sha,
               "data_root_path_sha256": _source_pins()["g"]["data_root_path_sha256"],
               "catalog_items": len(context.catalog), "training_examples": len(context.examples), "seed": config["seed"]}
    return runner, torch, setup, details


def _load_p(args):
    import torch
    root = _add_source(args.source_root)
    source_commit = _source_identity(root, "p")
    source_ledger_sha = _validate_source_ledger(root, "p")
    runtime = _runtime_identity(torch)
    config_path = _source_file(root, args.config)
    try:
        config_rel = config_path.relative_to(Path(root)).as_posix()
    except ValueError:
        raise ValueError("P config must be inside the fixed source root") from None
    if config_rel != P_CONFIG or args.config_sha != _source_pins()["p"]["config_sha256"] or _sha(config_path) != args.config_sha:
        raise ValueError("P anchor config path or SHA mismatch")
    from vkgr import retrieval_b1_run as b1
    from vkgr import retrieval_feat02_run as feat
    from vkgr import retrieval_seqm1_run as runner
    from vkgr.retrieval_ablation_run import read_pinned
    from vkgr.retrieval_negatives import target_frequency
    from vkgr.retrieval_training import EpochCursor, SamplerStats
    config = runner.validate_config(read_pinned(str(config_path), args.config_sha))
    spec = feat.validate_inputs(read_pinned(args.inputs, args.inputs_sha), runner.FEATURE)
    data, _, binding = b1.real_inputs(spec)
    data, extra_binding = feat.apply_feat02(data, spec, config["history_len"])
    rows, queries = feat._view(data, config["history_len"])
    data = {**data, "rows": rows, "queries": queries}
    if data.get("synthetic") is not False:
        raise ValueError("P real input required")
    if len(rows) < TOTAL_STEPS * 512:
        raise ValueError("P train rows do not cover the fixed profile before epoch end")
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no CPU fallback")
    b1.execution_context("cuda")
    if torch.get_num_threads() != 1:
        raise ValueError("P M1 execution requires the original single-thread runtime")
    runtime_summary = {"device": "cuda", "precision": "fp32", "deterministic": torch.are_deterministic_algorithms_enabled(),
                       "tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                       "threads": torch.get_num_threads()}
    _require_fresh(args.scratch, "scratch").mkdir(parents=True, exist_ok=False)
    def setup():
        model = runner._model(data, config, "cuda")
        optimizer = runner.make_optimizer(model, config)
        cursor = EpochCursor(len(rows), config["seed"])
        stats = SamplerStats()
        negative_rng = __import__("random").Random(f"p-b1-negative-v1:{config['seed']}")
        frequency = target_frequency(rows) if config["logq"] else None
        pool = sorted(data["support"]) if config["extra_uniform"] else None
        view = {**data, "rows": rows}
        def step():
            indices = cursor.next(config["batch_size"])
            started = time.perf_counter()
            batch = [rows[index] for index in indices]
            batch_seconds = time.perf_counter() - started
            feat.train_step(model, optimizer, batch, view, stats, cursor.epoch, config, negative_rng, pool, frequency)
            return batch_seconds
        return model, optimizer, step, ((feat, "_tensors2"), (b1, "_columns"), (b1, "_histories"))
    details = {"source_commit": source_commit, "source_ledger_sha256": source_ledger_sha, "runtime": runtime_summary,
               "config_sha256": args.config_sha,
               "inputs_sha256": args.inputs_sha, "prepared_sha256": binding["prepared_sha"], "query_sha256": binding["query_sha"],
               "feature_binding_sha256": b1._digest(extra_binding), "training_examples": len(rows), "seed": config["seed"],
               **_p3c_profile_identity()}
    return runner, torch, setup, details


def _write_receipt(path, receipt):
    path = Path(path)
    if path.is_symlink() or path.exists():
        raise ValueError("engineering receipt path must be fresh")
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def _fresh_roots(args):
    output = _require_fresh(args.output, "output")
    scratch = _require_fresh(args.scratch, "scratch")
    _absolute_path(str(output), "output", must_exist=False)
    _absolute_path(str(scratch), "scratch", must_exist=False)
    if output == scratch or output in scratch.parents or scratch in output.parents:
        raise ValueError("output and scratch roots must be separate, non-nested paths")


def _failure_detail(exc):
    """Return a bounded, path-free diagnostic suitable for operational stderr."""
    message = str(exc).lower()
    if "nonfinite" in message or "non-finite" in message or re.search(r"\b(nan|infinity|inf)\b", message):
        if "grad" in message:
            return "nonfinite_gradient", "nonfinite gradient detected"
        if "param" in message:
            return "nonfinite_parameters", "nonfinite parameters detected"
        return "nonfinite_loss_or_step", "nonfinite loss or step value detected"
    if any(token in message for token in ("source", "ledger", "commit", "p3c")):
        return "source_gate", "registered source identity gate failed"
    if any(token in message for token in ("input", "data root", "prepared", "manifest")):
        return "input_gate", "registered input gate failed"
    if any(token in message for token in ("runtime", "cuda", "environment", "torch")):
        return "runtime_gate", "registered runtime gate failed"
    return "transport_or_contract_gate", "profile gate rejected the request or execution"


class UtilizationSampler:
    def __init__(self, interval_seconds=30.0):
        self.interval_seconds = interval_seconds
        self.started = time.monotonic()
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        now = time.monotonic()
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds = cpu.ru_utime + cpu.ru_stime
        record = {"elapsed_seconds": now - self.started, "process_cpu_seconds": cpu_seconds}
        if self.samples:
            previous = self.samples[-1]
            delta_wall = record["elapsed_seconds"] - previous["elapsed_seconds"]
            delta_cpu = cpu_seconds - previous["process_cpu_seconds"]
            record["process_cpu_cores_used"] = delta_cpu / delta_wall if delta_wall > 0 else None
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, check=True, timeout=10)
            record["device_level_gpu_utilization_percent"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            record["device_level_gpu_utilization_percent"] = None
            record["gpu_utilization_status"] = "UNKNOWN"
        self.samples.append(record)

    def _run(self):
        self._sample()
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self):
        self.thread = threading.Thread(target=self._run, name="anchor-profile-util", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=15.0)
        if not self.samples or self.samples[-1]["elapsed_seconds"] < time.monotonic() - self.started - 1.0:
            self._sample()
        return self.samples


def _preflight(args):
    if args.branch == "g":
        runner, _, _, details = _load_g(args)
        del runner
    else:
        _, _, _, details = _load_p(args)
    return {"status": "preflight_ok", "branch": args.branch, "steps": 0, "written": "engineering_receipt_only",
            "validation_forward": False, "quality_metric": False, "checkpoint_write": False, **details}


def _load_request(path):
    value = _read_json(path)
    if type(value) is not dict or value.get("schema") != "anchor-profile-request-v1" or value.get("branch") not in ("g", "p"):
        raise ValueError("registered anchor profile request schema required")
    branch = value["branch"]
    required = {"schema", "branch", "source_root", "expected_source_commit", "source_ledger",
                "config", "scratch_root", "output_root", "inputs", "data_root"}
    if set(value) != required:
        raise ValueError("profile request fields differ from the registered branch schema")
    config = value["config"]
    if type(config) is not dict or set(config) != {"path", "sha256"}:
        raise ValueError("config path and SHA required in request")
    pins = _source_pins()[branch]
    ledger = value["source_ledger"]
    if value["expected_source_commit"] != pins["source_commit"] or type(ledger) is not dict \
            or set(ledger) != {"sha256", "files"} or ledger["sha256"] != pins["source_ledger_sha256"] \
            or ledger["files"] != pins["source_files"]:
        raise ValueError("request source identity must equal the complete registered source pin")
    expected_config = pins["config_sha256"]
    expected_config_path = G_CONFIG if branch == "g" else P_CONFIG
    if config["path"] != expected_config_path or config["sha256"] != expected_config:
        raise ValueError("request must pin the registered anchor config")
    if not all(isinstance(value[key], str) and value[key] for key in ("source_root", "scratch_root", "output_root")):
        raise ValueError("source root, scratch root, and output root required")
    inputs = value["inputs"]
    if type(inputs) is not dict:
        raise ValueError("request input pin object required")
    mapped = {"source_root": value["source_root"], "config": config["path"], "config_sha": config["sha256"],
              "scratch": value["scratch_root"], "output": value["output_root"], "branch": branch}
    if branch == "g":
        roles = {"prepared", "history_manifest", "sid_manifest", "content", "input_manifest", "vectors"}
        if set(inputs) != roles:
            raise ValueError("G request must pin each formal input role")
        for role, item in inputs.items():
            if type(item) is not dict or set(item) != {"path", "sha256"}:
                raise ValueError("G input pin must have path and SHA")
            if not isinstance(item["path"], str) or not item["path"] or item["sha256"] != pins["inputs"][role]:
                raise ValueError("G input must match its registered anchor pin")
            mapped[role] = item["path"]
            mapped[role + "_sha"] = item["sha256"]
        if not isinstance(value["data_root"], str) or hashlib.sha256(value["data_root"].encode()).hexdigest() != pins["data_root_path_sha256"]:
            raise ValueError("G historical data root must match its registered path pin")
        mapped["data_root"] = value["data_root"]
    else:
        if value["data_root"] is not None:
            raise ValueError("P data root must be null")
        if set(inputs) != {"inputs"} or type(inputs["inputs"]) is not dict or set(inputs["inputs"]) != {"path", "sha256"}:
            raise ValueError("P request must pin the formal M1 inputs file")
        if not isinstance(inputs["inputs"]["path"], str) or not inputs["inputs"]["path"] \
                or inputs["inputs"]["sha256"] != pins["inputs"]["inputs"]:
            raise ValueError("P M1 input file must match its registered anchor pin")
        mapped["inputs"] = inputs["inputs"]["path"]
        mapped["inputs_sha"] = inputs["inputs"]["sha256"]
    mapped["request_path"] = str(path)
    return SimpleNamespace(**mapped)


def _profile(args):
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA profile requires the registered GPU execution environment")
    _, _, setup_factory, details = _load_g(args) if args.branch == "g" else _load_p(args)
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    memory = {"pid": os.getpid(), "context_process_mib": own_gpu_mib(),
              "context_reserved_mib": torch.cuda.memory_reserved() / 2**20, "samples": []}
    torch.cuda.reset_peak_memory_stats()
    output = _require_fresh(args.output, "output")
    output.mkdir(parents=True, exist_ok=False)
    model, optimizer, step, tensor_builders = setup_factory()
    if not any(parameter.is_cuda for parameter in model.parameters()):
        raise ValueError("model must be on CUDA for this profile")
    import torch.profiler
    utilization = UtilizationSampler()
    utilization.start()
    batch_seconds = []
    stage_totals = {"tensor_build": 0.0, "h2d": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    try:
        for _ in range(WARMUP_STEPS):
            batch_seconds.append(step())
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(MEASURED_STEPS):
            batch_seconds.append(step())
        torch.cuda.synchronize()
        measured_wall = time.perf_counter() - begin
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
            with stage_hooks(model, optimizer, tensor_builders=tensor_builders) as stage_totals:
                for _ in range(DIAGNOSTIC_STEPS):
                    batch_seconds.append(step())
                torch.cuda.synchronize()
    finally:
        utilization_samples = utilization.stop()
    torch.cuda.synchronize()
    memory["samples"].append({"stage": "after-profile", "own_process_mib": own_gpu_mib(),
                              "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                              "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                              "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
    operator_summary = []
    for event in prof.key_averages():
        if event.key.startswith("profile_stage::"):
            operator_summary.append({"stage": event.key.removeprefix("profile_stage::"), "calls": event.count,
                                     "cpu_operator_total_us": getattr(event, "cpu_time_total", 0.0),
                                     "cuda_operator_total_us": getattr(event, "device_time_total", getattr(event, "self_cuda_time_total", 0.0))})
    operator_summary.sort(key=lambda row: row["stage"])
    measured = measured_vram(memory)
    receipt = {"status": "profile_complete", "branch": args.branch, "schedule": {"warmup": WARMUP_STEPS,
                "unprofiled_measured": MEASURED_STEPS, "profiled_diagnostic": DIAGNOSTIC_STEPS, "total": TOTAL_STEPS},
               "unprofiled_wall_seconds": measured_wall, "unprofiled_seconds_per_step": measured_wall / MEASURED_STEPS,
               "batch_materialization_seconds_all_steps": sum(batch_seconds), "profiler_stage_wall_seconds": stage_totals,
               "profiler_operator_totals": operator_summary, "python_residual": None, "python_residual_status": "UNKNOWN",
               "measured_vram_mib_context_inclusive": measured, "declared_vram_mib": int(__import__("math").ceil(measured * 1.15 / 256) * 256),
               "memory_measurement": memory, "utilization_samples": utilization_samples,
               "gpu_utilization_scope": "device-level visible GPU readout; may include other processes",
               "validation_forward": False, "quality_metric": False,
               "checkpoint_write": False, "written": "engineering_receipt_only", "limitations": [
                   "operator and hook stage totals overlap asynchronous device work and are not additive wall-clock fractions",
                   "batch materialization timer covers Python row selection/materialization, not a separately attributable storage read",
                   "unprofiled throughput includes a small batch-materialization timer overhead",
                   "device-level GPU utilization is not attributable to this PID and may include other processes",
                   "process CPU cores used is process-level, not host-wide CPU utilization"], **details}
    return receipt


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry")
    dry_group = dry.add_mutually_exclusive_group(required=True)
    dry_group.add_argument("--branch", choices=("g", "p"))
    dry_group.add_argument("--request")
    for name in ("preflight", "profile"):
        command = sub.add_parser(name)
        command.add_argument("--request", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    command = args.command
    if command == "dry":
        request = _load_request(_absolute_path(args.request, "request", must_exist=True)) if args.request else None
        branch = args.branch if args.branch else request.branch
        pins = _source_pins()[branch]
        receipt = {"status": "dry_ok", "branch": branch, "seed": 42,
                   "schedule": {"warmup": WARMUP_STEPS, "unprofiled_measured": MEASURED_STEPS,
                                "profiled_diagnostic": DIAGNOSTIC_STEPS, "total": TOTAL_STEPS},
                   "validation_forward": False, "quality_metric": False, "checkpoint_write": False}
        if request:
            receipt.update({"anchor_source_commit": pins["source_commit"],
                            "source_ledger_sha256": pins["source_ledger_sha256"],
                            "config_sha256": request.config_sha,
                            "input_sha256": pins["inputs"]})
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0
    phase = "request_gate"
    try:
        request_path = _absolute_path(args.request, "request", must_exist=True)
        request_sha = _sha(request_path)
        args = _load_request(request_path)
        args.command = command
        args.request_sha256 = request_sha
        _validate_request_paths(args)
        _fresh_roots(args)
        run_id = _run_identity(args.output, args.scratch)
        _validate_ambient_environment(args.branch, args.output, run_id)
        profile_identity = _profile_identity(request_path)
        if args.command == "preflight":
            phase = "preflight"
            receipt = _preflight(args)
        else:
            phase = "profile"
            receipt = _profile(args)
        pins = _source_pins()[args.branch]
        receipt.update({"run_id": run_id, "request_sha256": request_sha,
                        "profiler_script_sha256": profile_identity["profiler_script_sha256"],
                        "profiler_source_commit": profile_identity["profiler_source_commit"],
                        "profiler_source_override": profile_identity["profiler_source_override"],
                        "anchor_source_commit": pins["source_commit"]})
        if args.command == "preflight":
            Path(args.output).mkdir(parents=False, exist_ok=False)
            receipt_path = Path(args.output) / "preflight.json"
        else:
            receipt_path = Path(args.output) / "profile.json"
        payload = _write_receipt(receipt_path, receipt)
        print(payload)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, ImportError, subprocess.SubprocessError, AttributeError) as exc:
        reason, detail = _failure_detail(exc)
        print(f"ERROR anchor-profile phase={phase} reason_code={reason} exception={type(exc).__name__} detail={detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
