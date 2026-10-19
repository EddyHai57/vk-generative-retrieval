"""Finite queue transport for the P watch-positive read-only train preflight."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# This immutable G request supplies scheduler/root identity only. Its old ops
# directory and all experiment fields are intentionally ignored.
SOURCE_REQUEST_SHA256 = "ee313c1add3d5b224e9004d3bfc2fb922d2edae5c0b0b42d115be551532e025d"
RUN_ID = "p-watch-positive-stats-v1"
OPS_NAME = "vkgr-p-watch-positive-ops-v1"
DEPLOY_NAME = "vkgr-p-watch-positive-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def new_json(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())


def queue_state(root, run_id=RUN_ID):
    root = Path(root)
    value = {key: sorted(path.name for path in (root / "queue" / key).glob("*-" + run_id + ".json"))
             for key in ("pending", "claimed", "rejected")}
    value["reasons"] = {}
    for name in value["rejected"]:
        for path in ((root / "queue/rejected" / name).with_suffix(".reason"), root / "queue/rejected" / (name + ".reason")):
            if path.exists():
                value["reasons"][path.name] = path.read_text()[:8192]
    return value


def inventory(root, ops, job):
    run_id = job["run_id"]
    root, ops = Path(root), Path(ops)
    current = queue_state(root, run_id)
    run_dir = root / "runs" / run_id
    return {"run_id": run_id, "queue": current, "run_exists": run_dir.exists(),
            "intent_exists": (ops / ("intent-" + run_id + ".json")).exists(), "job_sha256": digest(job)}


def expected_scan_inputs(job, stats_request):
    """Closed scanner read-set derived from the pinned train-only request metadata."""
    from p_watch_positive_stats import load_p_inputs_metadata

    required_fields = {"schema", "inputs_metadata", "train_audit", "anchor_support_sha256", "quantile_method", "source_pins"}
    if (type(stats_request) is not dict or set(stats_request) != required_fields
            or stats_request.get("schema") != "p-watch-positive-stats-request-v1"
            or stats_request.get("quantile_method") != "nearest_rank"
            or stats_request.get("anchor_support_sha256") != "b080240542ae85166186fb621f7b074f9728f8b0e7d01d2c69ae0d0bff6b53a2"):
        raise ValueError("fixed P stats request schema/support/quantile required")
    expected_pin_names = {"src/vkgr/p_watch_positive.py", "scripts/p_watch_positive_stats.py",
                          "src/vkgr/retrieval_data.py", "scripts/p3_label_eda.py",
                          "configs/p_watch_positive/p-watch-positive-train-only.json"}
    if type(stats_request.get("source_pins")) is not dict or set(stats_request["source_pins"]) != expected_pin_names:
        raise ValueError("fixed P stats source pin set required")
    values = load_p_inputs_metadata(stats_request["inputs_metadata"], "a12b98003aa2fbb58df83bc70bb448e426d01c2a4cc2e6357bf1f95778f23e38")
    audit_path = Path(stats_request["train_audit"])
    if file_sha256(audit_path) != "3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe":
        raise ValueError("registered train audit identity mismatch")
    audit = json.loads(audit_path.read_text())
    catalog_path = audit_path.parent / "catalog.json"
    if file_sha256(catalog_path) != audit.get("catalog_sha256"):
        raise ValueError("train catalog binding mismatch")
    return closed_scan_inputs(job["config"], stats_request, values, audit, catalog_path)


def closed_scan_inputs(config_path, stats_request, values, audit, catalog_path):
    raw_root = Path(values["raw_root"])
    if not raw_root.is_absolute() or ".." in raw_root.parts:
        raise ValueError("accepted raw root path invalid")
    weeks = [f"subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet" for week in range(25)]
    train_records = audit.get("train_files")
    if type(train_records) is not list or len(train_records) != len(weeks) or [rec.get("path") for rec in train_records] != weeks:
        raise ValueError("closed train partition inventory mismatch")
    if any(type(rec.get("sha256")) is not str or len(rec["sha256"]) != 64 for rec in train_records):
        raise ValueError("invalid train partition digest inventory")
    files = [config_path, stats_request["inputs_metadata"], stats_request["train_audit"], str(catalog_path),
             str(raw_root / "metadata/items_metadata.parquet")]
    files.extend(str(raw_root / role) for role in weeks)
    if len(set(files)) != len(files):
        raise ValueError("duplicate scanner input path")
    return sorted(files)


def validate_job(job, source, ops, python_executable=None):
    python_executable = python_executable or sys.executable
    if type(job) is not dict or job.get("run_id") != RUN_ID or job.get("schema") != "vkgr-queue-job-v1" or job.get("kind") != "ops":
        raise ValueError("fixed P train-only stats job required")
    for name in ("root", "ops", "scheduler_code"):
        value = source.get(name)
        if type(value) is not str or not Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("scheduler source path invalid")
    cwd, config = job.get("cwd"), job.get("config")
    if type(cwd) is not str or not Path(cwd).is_absolute() or ".." in Path(cwd).parts:
        raise ValueError("absolute nontraversing job cwd required")
    if type(ops) is not str or not Path(ops).is_absolute() or ".." in Path(ops).parts:
        raise ValueError("fresh P operations path required")
    if Path(ops).name != OPS_NAME:
        raise ValueError("P stats must use its exclusive operations namespace")
    old_ops = Path(source["ops"]).resolve(); new_ops = Path(ops).resolve()
    if new_ops == old_ops or new_ops.is_relative_to(old_ops) or old_ops.is_relative_to(new_ops):
        raise ValueError("P stats ops must be disjoint from the source task ops")
    if type(config) is not str or not Path(config).is_absolute() or Path(config).parent != Path(ops):
        raise ValueError("private request must be in fresh P ops directory")
    config_path = Path(config)
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("private stats request must be a regular file")
    expected_script = str(Path(cwd) / "scripts/p_watch_positive_stats.py")
    argv = job.get("argv")
    if (type(argv) is not list or len(argv) != 5 or argv[0] != python_executable or argv[1:4] != ["-B", expected_script, "--request"] or argv[4] != config):
        raise ValueError("fixed scanner interpreter, script and request argv required")
    if (job.get("cpu_cores"), job.get("ram_mib"), job.get("disk_gib"), job.get("vram_mib"), job.get("wall_hours")) != (1, 8192, 1, 0, 1):
        raise ValueError("fixed CPU-only stats resource envelope required")
    if job.get("no_admission") is not False or job.get("seed") != 0 or type(job.get("inputs")) is not list:
        raise ValueError("stats job admission/seed/input envelope mismatch")
    if any(type(path) is not str or not Path(path).is_absolute() or ".." in Path(path).parts for path in job["inputs"]):
        raise ValueError("absolute nontraversing job inputs required")
    stats_request = json.loads(config_path.read_text())
    expected_inputs = expected_scan_inputs(job, stats_request)
    if job["inputs"] != expected_inputs:
        raise ValueError("job inputs differ from the closed train-only scanner read set")
    if job.get("pinned_files") != []:
        raise ValueError("scheduler source map must own the pinned file set")
    pinned_manifest = job.get("pinned_manifest")
    if type(pinned_manifest) is not str or not Path(pinned_manifest).is_absolute() or ".." in Path(pinned_manifest).parts or Path(pinned_manifest).parent != Path(ops):
        raise ValueError("absolute pinned source-map manifest path required")
    return job


def load_source_request(path):
    path = Path(path)
    if path.is_symlink() or file_sha256(path) != SOURCE_REQUEST_SHA256:
        raise ValueError("accepted scheduler source request hash mismatch")
    source = json.loads(path.read_text())
    if not {"root", "ops", "scheduler_code", "resource_basis"} <= set(source):
        raise ValueError("scheduler source request schema mismatch")
    return source


def verify_fresh_deployment(deployment_path, ops, job, config):
    deployment_path, ops = Path(deployment_path), Path(ops)
    if deployment_path.is_symlink() or ops.is_symlink():
        raise ValueError("deployment/ops namespace cannot be symlinks")
    if deployment_path.name != DEPLOY_NAME or deployment_path != Path(job["cwd"]):
        raise ValueError("P stats fresh deployment namespace mismatch")
    deployment_path = deployment_path.resolve()
    ops = ops.resolve()
    if deployment_path == ops or deployment_path.is_relative_to(ops) or ops.is_relative_to(deployment_path) or deployment_path.parent != ops.parent:
        raise ValueError("deployment and ops must be disjoint namespaces")
    deploy_file = deployment_path / "deployment.json"
    if deploy_file.is_symlink() or not deploy_file.is_file():
        raise ValueError("fresh deployment record missing")
    record = json.loads(deploy_file.read_text())
    pinned_path = Path(job["pinned_manifest"])
    if pinned_path.is_symlink() or not pinned_path.is_file():
        raise ValueError("pinned source manifest missing")
    pinned = json.loads(pinned_path.read_text())
    required = {"schema", "base_commit", "target_commit", "files", "overrides"}
    if type(pinned) is not dict or set(pinned) != required or pinned["schema"] != "vkgr-pinned-v1" or pinned["overrides"] != {}:
        raise ValueError("vkgr-pinned-v1 source map required")
    if type(record) is not dict or not {"commit", "files", "created_at_utc", "data_overlay"} <= set(record):
        raise ValueError("fresh deployment record schema mismatch")
    commit = record["commit"]
    if (type(commit) is not str or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)
            or pinned["base_commit"] != commit or pinned["target_commit"] != commit):
        raise ValueError("deployment commit and source map identity mismatch")
    actual = subprocess.run(["git", "-C", str(deployment_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if actual != commit:
        raise ValueError("fresh deployment commit differs from Git HEAD")
    files = pinned["files"]
    if type(files) is not dict or files != record["files"]:
        raise ValueError("deployment file hashes differ from pinned source map")
    for relative, expected in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or type(expected) is not str or len(expected) != 64:
            raise ValueError("invalid source file pin")
        target = deployment_path / path
        if target.is_symlink() or not target.is_file() or file_sha256(target) != expected:
            raise ValueError("deployed source file hash mismatch: " + relative)
    source_pins = config.get("source_pins")
    required_source_pins = {"src/vkgr/p_watch_positive.py", "scripts/p_watch_positive_stats.py",
                            "src/vkgr/retrieval_data.py", "scripts/p3_label_eda.py",
                            "configs/p_watch_positive/p-watch-positive-train-only.json"}
    if (type(source_pins) is not dict or set(source_pins) != required_source_pins
            or any(files.get(name) != digest for name, digest in source_pins.items())):
        raise ValueError("scanner source pins differ from deployment manifest")
    from p_watch_positive_stats import validate_source_pins
    validate_source_pins(source_pins, deployment_path)
    created = datetime.datetime.fromisoformat(record["created_at_utc"].replace("Z", "+00:00"))
    if created.tzinfo is None or created.utcoffset() != datetime.timedelta(0) or created > datetime.datetime.now(datetime.timezone.utc):
        raise ValueError("invalid/future UTC deployment creation time")
    if record["data_overlay"] is not False:
        raise ValueError("stats deployment cannot declare a data overlay")
    return {"commit": commit, "files_sha256": digest(files), "deployment_sha256": file_sha256(deploy_file)}


def main():
    request = json.loads(sys.stdin.read(2 * 1024 * 1024 + 1))
    required_request = {"action", "source_request_path", "source_request_sha256", "ops", "deployment_path", "job"}
    if type(request) is not dict or not required_request <= set(request) or set(request) - (required_request | {"dry_job_sha256"}):
        raise ValueError("queue request schema mismatch")
    action = request["action"]
    source = load_source_request(request["source_request_path"])
    if request.get("source_request_sha256") != SOURCE_REQUEST_SHA256:
        raise ValueError("scheduler source request identity mismatch")
    ops_value = request.get("ops")
    deployment_path = request["deployment_path"]
    job = request["job"]
    config = json.loads(Path(job["config"]).read_text())
    job = validate_job(job, source, ops_value)
    verify_fresh_deployment(deployment_path, ops_value, job, config)
    root, ops = Path(source["root"]), Path(ops_value)
    state = inventory(root, ops, job)
    if action == "inventory":
        output = state
    elif action in ("dry", "submit"):
        if state["run_exists"] or state["intent_exists"] or any(state["queue"][key] for key in ("pending", "claimed", "rejected")):
            raise ValueError("fixed stats job already exists; never resubmit")
        sys.path.insert(0, str(Path(source["scheduler_code"]) / "ops"))
        from vkgr_ops import launcher, scheduler
        prepared = launcher.prepare(job)
        size = len(json.dumps({"spec": job, "prepared": prepared}).encode())
        if size >= 48 * 1024 - 2048:
            raise ValueError("launch metadata envelope too large")
        output = {"status": "dry_ok", "run_id": RUN_ID, "job_sha256": digest(job), "launch_bytes": size}
        if action == "submit":
            if request.get("dry_job_sha256") != digest(job):
                raise ValueError("dry-run job binding mismatch")
            (ops / "publish.lock").touch(exist_ok=True)
            with (ops / "publish.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                state = inventory(root, ops, job)
                if state["run_exists"] or state["intent_exists"] or any(state["queue"][key] for key in ("pending", "claimed", "rejected")):
                    raise ValueError("fixed stats job already exists; never resubmit")
                new_json(ops / ("intent-" + RUN_ID + ".json"), output)
                output.update(status="written_unconfirmed", queue_path=str(scheduler.submit(root, job)))
    elif action == "status":
        output = inventory(root, ops, job)
        run_dir = root / "runs" / RUN_ID
        for path in (run_dir / "terminal.json", run_dir / "logs/stdout.log", run_dir / "logs/stderr.log"):
            if path.exists():
                data = path.read_bytes()
                if len(data) > 2 * 1024 * 1024:
                    raise ValueError("status receipt too large")
                try:
                    payload = json.loads(data)
                except ValueError:
                    payload = data.decode()
                output[path.name] = {"sha256": hashlib.sha256(data).hexdigest(), "payload": payload}
    else:
        raise ValueError("unknown queue action")
    output["read_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
