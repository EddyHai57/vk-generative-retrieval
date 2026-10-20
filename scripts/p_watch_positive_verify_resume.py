"""Closed verify-only continuation for the fixed P watch-positive training runs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import p_watch_positive_driver as driver
from vkgr import p_watch_positive_ops as ops

REQUEST_SCHEMA = "p-watch-positive-verify-resume-request-v1"
TERMINAL_SCHEMA = "p-watch-positive-verify-resume-terminal-v1"
OWNER_SCHEMA = "p-watch-positive-verify-resume-owner-v1"
ORIGINAL_SOURCE_COMMIT = "2c4b3ed2adff41ba298273a72d48d849fc9a028a"
ORIGINAL_DEPLOYMENT = "/workspace/vkgr-p-watch-positive-train-v6"
FROZEN_TRAIN_MANIFEST_SHA256 = {
    42: "6b1edb9b1ea4676d4e6e166bfe4672a364e7e4581f81443bdb6a25fe1204a303",
    2024: "68b28d21bafd3595b36456e8448c5223aef8baa73837f0219e8cd991191d179c",
    2025: "27cf653099f83ddd0addedf77f70325e069e70278704205c960b41730ce80617",
}
MAX_BYTES = 8 * 1024 * 1024
MAX_DEADLINE_SECONDS = 24 * 60 * 60
MIN_POLL_SECONDS = 900


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON value")


def _json_file(path, max_bytes=MAX_BYTES):
    p = Path(path)
    if p.is_symlink() or not p.is_file() or p.stat().st_size > max_bytes:
        raise ValueError("bounded nonsymlink private JSON file required")
    try:
        return json.loads(p.read_text(), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("strict UTF-8 JSON required") from exc


def _private_path(value, role, *, must_exist=False):
    if type(value) is not str:
        raise ValueError(role + " absolute private path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise ValueError(role + " absolute nonsymlink private path required")
    if must_exist and not path.exists():
        raise ValueError(role + " does not exist")
    return path


def _bundle_identity():
    root = SCRIPT_DIR.parent
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"], text=True).strip():
        raise ValueError("verify-only ops bundle must be a clean fixed Git checkout")
    return {"commit": commit, "files": {
        "scripts/p_watch_positive_driver.py": driver.sha(SCRIPT_DIR / "p_watch_positive_driver.py"),
        "scripts/p_watch_positive_verify_resume.py": driver.sha(Path(__file__)),
    }}


def _validate_request(value):
    required = {"schema", "root", "ops", "plan_path", "plan_sha256", "python", "deadline_epoch",
        "original_deadline_epoch", "original_spent_seconds", "deadline_reason", "poll_seconds"}
    if type(value) is not dict or set(value) != required or value.get("schema") != REQUEST_SCHEMA:
        raise ValueError("closed verify-only request schema required")
    root = _private_path(value["root"], "scheduler root", must_exist=True)
    ops_dir = _private_path(value["ops"], "new verify OPS")
    plan = _private_path(value["plan_path"], "original train plan", must_exist=True)
    if driver.sha(plan) != value["plan_sha256"]:
        raise ValueError("original train plan physical SHA mismatch")
    ops.validate_python_executable(value["python"])
    now = int(time.time())
    deadline = value["deadline_epoch"]
    original_deadline = value["original_deadline_epoch"]
    spent = value["original_spent_seconds"]
    poll = value["poll_seconds"]
    if type(deadline) is not int or type(original_deadline) is not int:
        raise ValueError("absolute integer deadline epochs required")
    if type(spent) is not int or spent < 0:
        raise ValueError("original spent seconds must be a nonnegative integer")
    if type(poll) is not int or poll < MIN_POLL_SECONDS:
        raise ValueError("verify-only polling interval must be at least 900 seconds")
    if deadline <= now or deadline - now < poll or deadline > now + MAX_DEADLINE_SECONDS:
        raise ValueError("verify-only absolute deadline must be future, fit one poll, and remain within 24 hours")
    if original_deadline > now:
        if value["deadline_reason"] != "within_original_deadline" or deadline > original_deadline:
            raise ValueError("verify-only deadline must remain within the original driver deadline")
    elif value["deadline_reason"] != "ddcae40_authorized_window_after_original_deadline":
        raise ValueError("expired original deadline requires the explicit ddcae40 reason")
    if root.is_symlink() or ops_dir == root or ops_dir.is_relative_to(root) or root.is_relative_to(ops_dir):
        raise ValueError("scheduler root and new verify OPS must be separate nonsymlink paths")
    return value


def _load_request(request_path, request_sha256, plan_path=None, plan_sha256=None):
    request_file = _private_path(request_path, "verify-only request", must_exist=True)
    if driver.sha(request_file) != request_sha256:
        raise ValueError("verify-only request physical SHA mismatch")
    request = _json_file(request_file)
    _validate_request(request)
    if plan_path is not None and str(_private_path(plan_path, "original train plan")) != request["plan_path"]:
        raise ValueError("CLI plan path differs from the closed request")
    if plan_sha256 is not None and plan_sha256 != request["plan_sha256"]:
        raise ValueError("CLI plan SHA differs from the closed request")
    plan = _json_file(request["plan_path"])
    if type(plan) is not dict or set(plan) != {"entries", "sha256"}:
        raise ValueError("closed original train plan schema required")
    entries = plan["entries"]
    if type(entries) is not list or len(entries) != len(ops.SEEDS) or plan["sha256"] != ops.digest(entries):
        raise ValueError("original train plan digest or three-entry scope mismatch")
    return request, plan, request_file


def _validate_plan_entries(request, plan):
    expected_runs = list(ops.RUN_IDS.values())
    expected_seeds = list(ops.SEEDS)
    expected_job_fields = {"schema", "run_id", "kind", "cwd", "config", "config_sha256", "seed", "inputs",
        "pinned_manifest", "pinned_files", "env", "cpu_cores", "ram_mib", "vram_mib", "disk_gib",
        "wall_hours", "stop_mode", "no_admission", "argv"}
    entries = plan["entries"]
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != {"run_id", "scheduler_code", "train_job"}:
            raise ValueError("closed original train plan entry required")
        run_id = expected_runs[index]
        job = entry["train_job"]
        if type(job) is not dict or set(job) != expected_job_fields:
            raise ValueError("closed original train job schema required")
        if (entry["run_id"] != run_id or job.get("run_id") != run_id or job.get("seed") != expected_seeds[index]
                or job.get("kind") != "real" or job.get("schema") != "vkgr-queue-job-v1"
                or job.get("cwd") != ORIGINAL_DEPLOYMENT):
            raise ValueError("original train identity, seed, kind, or deployment differs from the fixed plan")
        argv = job.get("argv")
        if (type(argv) is not list or argv[:5] != [job.get("argv", [None])[0], "-B", "-m",
                "vkgr.p_watch_positive_run", "train"] or argv.count("train") != 1):
            raise ValueError("original train job runner/module/action binding changed")
        p3c_pins = [pin for pin in job.get("pinned_files", [])
            if type(pin) is str and pin.startswith(ops.P3C_DATA_CHANNEL["path"] + "=")]
        expected_p3c_pin = ops.P3C_DATA_CHANNEL["path"] + "=" + ops.P3C_DATA_CHANNEL["contract_sha256"]
        if p3c_pins != [expected_p3c_pin]:
            raise ValueError("original train job p3c data-channel pin changed")
        scheduler_code = _private_path(entry["scheduler_code"], "original scheduler code", must_exist=True)
        if type(job.get("inputs")) is not list or type(job.get("env")) is not list:
            raise ValueError("original train inputs/environment shape changed")
        entry["scheduler_code"] = str(scheduler_code)
    return entries


def _assert_verify_ids_fresh(root):
    root = Path(root)
    queues = root / "queue"
    for run_id in (rid + "-verify" for rid in ops.RUN_IDS.values()):
        if (root / "runs" / run_id).exists() or (root / "runs" / run_id).is_symlink():
            raise FileExistsError("fixed verify identity already has a run directory: " + run_id)
        for state in ("pending", "claimed", "rejected"):
            folder = queues / state
            if folder.is_symlink():
                raise ValueError("scheduler queue state path cannot be a symlink")
            if folder.exists() and list(folder.glob("*-" + run_id + ".json")):
                raise FileExistsError("fixed verify identity already exists in queue: " + run_id)


def _inspect_train_receipts(request, plan):
    entries = _validate_plan_entries(request, plan)
    root = Path(request["root"])
    results = []
    for entry in entries:
        job = entry["train_job"]
        run_id = job["run_id"]
        head = subprocess.check_output(["git", "-C", job["cwd"], "rev-parse", "HEAD"], text=True).strip()
        if head != ORIGINAL_SOURCE_COMMIT:
            raise ValueError("original train science deployment commit changed")
        terminal = driver.terminal(root, run_id)
        if not driver.terminal_success(terminal, run_id):
            raise ValueError("original train terminal is not succeeded: " + run_id)
        if not driver.validate_claimed_run(root, job):
            raise ValueError("original train has no unique claimed run record: " + run_id)
        stdout = root / "runs" / run_id / "logs/stdout.log"
        manifest = root / "runs" / run_id / "out/manifest.json"
        if stdout.is_symlink() or not stdout.is_file() or stdout.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("bounded original train stdout receipt required: " + run_id)
        receipt = driver.train_receipt(stdout, run_id)
        seed = job["seed"]
        manifest_sha = FROZEN_TRAIN_MANIFEST_SHA256[seed]
        if (not manifest.is_file() or manifest.is_symlink() or driver.sha(manifest) != manifest_sha
                or receipt["manifest_sha256"] != manifest_sha):
            raise ValueError("original train manifest physical SHA differs from its frozen value: " + run_id)
        results.append({"run_id": run_id, "verify_run_id": run_id + "-verify", "seed": seed,
            "manifest_path": str(manifest), "manifest_sha256": manifest_sha})
    _assert_verify_ids_fresh(root)
    return entries, results


def _scheduler_modules(scheduler_code):
    scheduler_ops = Path(scheduler_code) / "ops"
    ops.validate_scheduler_source(scheduler_ops)
    for name in tuple(sys.modules):
        if name == "vkgr_ops" or name.startswith("vkgr_ops."):
            del sys.modules[name]
    sys.path.insert(0, str(scheduler_ops))
    from vkgr_ops import launcher, scheduler
    if Path(launcher.__file__).resolve() != (scheduler_ops / "vkgr_ops/launcher.py").resolve():
        raise ValueError("launcher import did not resolve to the pinned original scheduler source")
    return launcher, scheduler


def _build_verify_jobs(request, plan):
    entries, train_receipts = _inspect_train_receipts(request, plan)
    verify_jobs = []
    for entry, receipt in zip(entries, train_receipts):
        job = driver.verify_job(entry["train_job"], receipt["manifest_sha256"], receipt["manifest_path"])
        if job["run_id"] != receipt["verify_run_id"]:
            raise ValueError("verify job identity differs from the fixed original run")
        verify_jobs.append(job)
    return entries, train_receipts, verify_jobs


def _dry_data(request, plan, request_sha256):
    entries, train_receipts, verify_jobs = _build_verify_jobs(request, plan)
    _assert_verify_ids_fresh(request["root"])
    prepared = []
    launch_bytes = []
    for entry, job in zip(entries, verify_jobs):
        launcher, _scheduler = _scheduler_modules(entry["scheduler_code"])
        meta = launcher.prepare(job)
        size = len(ops.canonical({"spec": job, "prepared": meta}))
        if size >= 48 * 1024 - 2048:
            raise ValueError("verify-only launch envelope exceeds the scheduler size bound")
        prepared.append(meta)
        launch_bytes.append(size)
    digest = ops.digest({"request_sha256": request_sha256, "plan_sha256": request["plan_sha256"],
        "jobs": verify_jobs, "prepared": prepared})
    return {"status": "dry_ok", "dry_jobs_sha256": digest,
        "verify_run_ids": [job["run_id"] for job in verify_jobs],
        "train_manifests": [{"run_id": row["run_id"], "manifest_sha256": row["manifest_sha256"]}
            for row in train_receipts], "launch_bytes": launch_bytes}


def inventory(request_path, request_sha256, plan_path=None, plan_sha256=None):
    request, plan, _ = _load_request(request_path, request_sha256, plan_path, plan_sha256)
    ops_dir = Path(request["ops"])
    if ops_dir.exists():
        raise FileExistsError("new verify OPS directory must be absent for inventory/dry-run")
    _, receipts = _inspect_train_receipts(request, plan)
    return {"status": "inventory_ok", "train_manifests": receipts,
        "verify_run_ids": [row["verify_run_id"] for row in receipts],
        "original_spent_seconds": request["original_spent_seconds"],
        "original_deadline_epoch": request["original_deadline_epoch"],
        "deadline_epoch": request["deadline_epoch"]}


def dry_run(request_path, request_sha256, plan_path=None, plan_sha256=None):
    request, plan, _ = _load_request(request_path, request_sha256, plan_path, plan_sha256)
    if Path(request["ops"]).exists():
        raise FileExistsError("new verify OPS directory must be absent for inventory/dry-run")
    return _dry_data(request, plan, request_sha256)


def _write_new(path, value):
    target = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def start(request_path, request_sha256, plan_path, plan_sha256, dry_jobs_sha256):
    request, plan, request_file = _load_request(request_path, request_sha256, plan_path, plan_sha256)
    ops_dir = Path(request["ops"])
    if ops_dir.exists():
        raise FileExistsError("new verify OPS directory already exists; no restart or overwrite")
    dry = _dry_data(request, plan, request_sha256)
    if dry["dry_jobs_sha256"] != dry_jobs_sha256:
        raise ValueError("start requires the matching read-only dry-run digest")
    bundle = _bundle_identity()
    ops_dir.mkdir(mode=0o700)
    started = {"schema": "p-watch-positive-verify-resume-start-v1", "status": "starting",
        "request_sha256": request_sha256, "plan_sha256": request["plan_sha256"],
        "dry_jobs_sha256": dry_jobs_sha256, "bundle": bundle,
        "train_manifests": dry["train_manifests"], "verify_run_ids": dry["verify_run_ids"],
        "original_spent_seconds": request["original_spent_seconds"],
        "original_deadline_epoch": request["original_deadline_epoch"],
        "deadline_epoch": request["deadline_epoch"],
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    _write_new(ops_dir / "start.json", started)
    log = (ops_dir / "driver.log").open("xb")
    science_root = Path(plan["entries"][0]["train_job"]["cwd"])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(science_root / "src"),
        str(science_root / "scripts"), str(SCRIPT_DIR)))}
    command = [request["python"], "-B", str(Path(__file__).resolve()), "run",
        "--request", str(request_file), "--request-sha", request_sha256,
        "--plan", request["plan_path"], "--plan-sha", request["plan_sha256"],
        "--dry-jobs-sha", dry_jobs_sha256]
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            close_fds=True, env=env, cwd=str(ops_dir), start_new_session=True)
    except Exception as exc:
        log.close()
        _write_new(ops_dir / "start-failed.json", {"status": "ALERT", "detail": f"{type(exc).__name__}: {exc}"})
        raise
    log.close()
    return {"status": "started", "pid": child.pid, "session_leader": True,
        "dry_jobs_sha256": dry_jobs_sha256, "deadline_epoch": request["deadline_epoch"],
        "ops": str(ops_dir)}


def run(request_path, request_sha256, plan_path, plan_sha256, dry_jobs_sha256):
    request, plan, _ = _load_request(request_path, request_sha256, plan_path, plan_sha256)
    ops_dir = Path(request["ops"])
    if ops_dir.is_symlink() or not ops_dir.is_dir():
        raise ValueError("fresh verify OPS directory required for detached run")
    if (ops_dir / "owner.json").exists() or (ops_dir / "verify-resume-terminal.json").exists():
        raise FileExistsError("verify-only owner or terminal already exists; no restart or overwrite")
    start_receipt = _json_file(ops_dir / "start.json")
    if (type(start_receipt) is not dict or start_receipt.get("schema") != "p-watch-positive-verify-resume-start-v1"
            or start_receipt.get("request_sha256") != request_sha256
            or start_receipt.get("plan_sha256") != request["plan_sha256"]
            or start_receipt.get("dry_jobs_sha256") != dry_jobs_sha256):
        raise ValueError("detached run does not match the one-time start receipt")
    try:
        driver.wait_for_reparented_session()
        dry = _dry_data(request, plan, request_sha256)
        if dry["dry_jobs_sha256"] != dry_jobs_sha256:
            raise ValueError("detached verify plan changed after start")
        remaining = int(request["deadline_epoch"] - time.time())
        if remaining < request["poll_seconds"]:
            raise TimeoutError("absolute verify-only deadline has less than one poll remaining")
        bundle = _bundle_identity()
        owner = {"schema": OWNER_SCHEMA, "pid": os.getpid(), "ppid": os.getppid(), "sid": os.getsid(0),
            "request_sha256": request_sha256, "plan_sha256": request["plan_sha256"],
            "dry_jobs_sha256": dry_jobs_sha256, "bundle": bundle,
            "original_source_commit": ORIGINAL_SOURCE_COMMIT,
            "train_manifests": dry["train_manifests"], "verify_run_ids": dry["verify_run_ids"],
            "original_spent_seconds": request["original_spent_seconds"],
            "original_deadline_epoch": request["original_deadline_epoch"],
            "deadline_epoch": request["deadline_epoch"], "remaining_seconds_at_owner": remaining,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        _write_new(ops_dir / "owner.json", owner)
        payload = {"schema": "p-watch-positive-driver-v1", "root": request["root"], "ops": request["ops"],
            "request_sha256": request_sha256, "entries": plan["entries"],
            "plan_sha256": ops.digest(plan["entries"]), "deadline_seconds": remaining,
            "poll_seconds": request["poll_seconds"]}
        summary = driver.driver(payload)
        status = summary.get("status", "ALERT")
        terminal = {"schema": TERMINAL_SCHEMA, "status": status, "request_sha256": request_sha256,
            "plan_sha256": request["plan_sha256"], "dry_jobs_sha256": dry_jobs_sha256,
            "results": summary.get("results", []), "failed_run_id": summary.get("failed_run_id"),
            "detail": summary.get("detail"),
            "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except FileExistsError:
        raise
    except Exception as exc:
        terminal = {"schema": TERMINAL_SCHEMA, "status": "ALERT", "request_sha256": request_sha256,
            "plan_sha256": request["plan_sha256"], "dry_jobs_sha256": dry_jobs_sha256,
            "results": [], "failed_run_id": None, "detail": f"{type(exc).__name__}: {exc}",
            "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    _write_new(ops_dir / "verify-resume-terminal.json", terminal)
    return terminal


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("inventory", "dry-run", "start", "run"):
        action = sub.add_parser(name)
        action.add_argument("--request", required=True)
        action.add_argument("--request-sha", required=True)
        action.add_argument("--plan", required=True)
        action.add_argument("--plan-sha", required=True)
        if name in ("start", "run"):
            action.add_argument("--dry-jobs-sha", required=True)
    args = parser.parse_args(argv)
    if args.action == "inventory":
        result = inventory(args.request, args.request_sha, args.plan, args.plan_sha)
    elif args.action == "dry-run":
        result = dry_run(args.request, args.request_sha, args.plan, args.plan_sha)
    elif args.action == "start":
        result = start(args.request, args.request_sha, args.plan, args.plan_sha, args.dry_jobs_sha)
    else:
        result = run(args.request, args.request_sha, args.plan, args.plan_sha, args.dry_jobs_sha)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
