"""Finite detached P watch-positive train/verify controller; no model code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vkgr import p_watch_positive_ops as ops

MIN_POLL_SECONDS = 900
MAX_DEADLINE_SECONDS = 48 * 60 * 60

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file(): raise ValueError("sealed driver file missing")
    return json.loads(p.read_text())
def write_new(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False); f.flush(); os.fsync(f.fileno())
def train_receipt(path, run_id):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("bounded nonsymlink train stdout required")
    text = path.read_text()
    rows = []
    for line in text.splitlines():
        try: value = json.loads(line)
        except ValueError: continue
        if type(value) is dict and value.get("status") == "complete" and value.get("run_id") == run_id: rows.append(value)
    if len(rows) != 1 or not isinstance(rows[0].get("manifest_sha256"), str) or len(rows[0]["manifest_sha256"]) != 64:
        raise ValueError("train stdout lacks one run-bound complete manifest receipt")
    return rows[0]
def terminal(root, run_id):
    path = Path(root) / "runs" / run_id / "terminal.json"
    if not path.is_file() or path.is_symlink(): return None
    return json.loads(path.read_text())

def terminal_success(value, run_id):
    return (type(value) is dict and set(value) == {"schema", "run_id", "status", "exit_code", "signal",
            "started_at", "ended_at", "wall_seconds", "wall_cap_exceeded"}
            and value["schema"] == "vkgr-terminal-v1" and value["run_id"] == run_id
            and value["status"] == "succeeded" and value["exit_code"] == 0
            and value["signal"] is None and value["wall_cap_exceeded"] is False)

def validate_own_session(pid=None, ppid=None, sid=None):
    pid = os.getpid() if pid is None else pid
    ppid = os.getppid() if ppid is None else ppid
    sid = os.getsid(0) if sid is None else sid
    if ppid != 1 or sid != pid:
        raise ValueError("driver must be PPID 1 and its own session leader")
    return True

def wait_for_reparented_session(timeout_seconds=5):
    deadline = time.monotonic() + timeout_seconds
    while os.getppid() != 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    return validate_own_session()

def validate_driver_payload(req):
    required = {"schema", "root", "ops", "request_sha256", "plan_sha256", "entries", "deadline_seconds", "poll_seconds"}
    if type(req) is not dict or set(req) != required or req.get("schema") != "p-watch-positive-driver-v1":
        raise ValueError("sealed finite driver request required")
    deadline, poll = req["deadline_seconds"], req["poll_seconds"]
    if type(deadline) is not int or not 1 <= deadline <= MAX_DEADLINE_SECONDS:
        raise ValueError("driver deadline must be finite and <=48h")
    if type(poll) is not int or poll < MIN_POLL_SECONDS or poll > deadline:
        raise ValueError("driver poll must be >=900 seconds and fit within its deadline")
    if type(req["entries"]) is not list or [row.get("run_id") for row in req["entries"]] != list(ops.RUN_IDS.values()):
        raise ValueError("fixed three-seed plan order required")
    if req["plan_sha256"] != ops.digest(req["entries"]):
        raise ValueError("sealed plan digest mismatch")
    for row in req["entries"]:
        if type(row.get("train_job")) is not dict or row["train_job"].get("run_id") != row["run_id"]:
            raise ValueError("each plan seed must bind one train job")
    return req

def claim_state(root, run_id):
    root = Path(root); queues = root / "queue"
    result = {name: sorted((queues / name).glob("*-" + run_id + ".json"))
              for name in ("pending", "claimed", "rejected")}
    reasons = []
    for path in result["rejected"]:
        for reason in (path.with_suffix(".reason"), Path(str(path) + ".reason")):
            if reason.is_file() and not reason.is_symlink(): reasons.append(reason.read_text()[:8192])
    if result["rejected"]:
        raise ValueError("scheduler rejected fixed run: " + " | ".join(reasons))
    if len(result["claimed"]) > 1:
        raise ValueError("duplicate claimed queue identities")
    return result

def run_record(root, run_id):
    path = Path(root) / "runs" / run_id / "run.json"
    if not path.is_file() or path.is_symlink(): return None
    return json.loads(path.read_text())

def validate_claimed_run(root, job):
    run_id = job["run_id"]
    state = claim_state(root, run_id)
    if len(state["claimed"]) != 1:
        return False
    claim = json.loads(state["claimed"][0].read_text())
    if claim != {"schema": "vkgr-queue-job-v1", **job}:
        raise ValueError("claimed queue job differs from fixed train/verify request")
    record = run_record(root, run_id)
    if record is None:
        return False
    if type(record) is not dict or record.get("schema") != "vkgr-run-v1" or record.get("run_id") != run_id:
        raise ValueError("claimed run lacks exact scheduler run record")
    config_record = record.get("config")
    expected_config = str(Path(job["config"]).resolve())
    if (record.get("kind") != job["kind"] or record.get("argv") != job["argv"]
            or type(config_record) is not dict or config_record.get("path") != expected_config
            or config_record.get("sha256") != ops.sha(job["config"])):
        raise ValueError("run record source/config/argv differs from claimed job")
    git_record = record.get("git")
    head = subprocess.check_output(["git", "-C", job["cwd"], "rev-parse", "HEAD"], text=True).strip()
    if (type(git_record) is not dict or set(git_record) != {"commit", "dirty", "dirty_files"}
            or git_record.get("commit") != head or type(git_record.get("dirty")) is not bool
            or type(git_record.get("dirty_files")) is not list):
        raise ValueError("run record deployed source commit/dirty state shape mismatch")
    accepted_p3c_dirty = [{"path": ops.P3C_DATA_CHANNEL["path"],
                           "sha256": ops.P3C_DATA_CHANNEL["contract_sha256"]}]
    if ((git_record["dirty"] is False and git_record["dirty_files"] != [])
            or (git_record["dirty"] is True and git_record["dirty_files"] != accepted_p3c_dirty)):
        raise ValueError("run record has unapproved deployed source dirty paths")
    pinned_path = Path(job["pinned_manifest"])
    if pinned_path.is_symlink() or not pinned_path.is_file():
        raise ValueError("claimed run pinned source manifest missing")
    pinned_manifest = json.loads(pinned_path.read_text())
    expected_pins = dict(pinned_manifest.get("files", {}))
    for item in job.get("pinned_files", []):
        name, separator, value = item.rpartition("=")
        if not separator or name in expected_pins and expected_pins[name] != value:
            raise ValueError("claimed job contains conflicting source pin")
        expected_pins[name] = value
    if record.get("pinned_files") != expected_pins:
        raise ValueError("scheduler run source pin map differs from frozen deployment manifest")
    resources = record.get("resources") or {}
    for key in ("cpu_cores", "ram_mib", "vram_mib", "disk_gib"):
        if resources.get(key) != job.get(key):
            raise ValueError("run record resource differs from claimed scheduler job: " + key)
    if resources.get("wall_seconds") != job.get("wall_hours") * 3600.0:
        raise ValueError("run record wall cap differs from sealed job")
    recorded_env = record.get("env")
    if type(recorded_env) is not dict or recorded_env.get("VKGR_RUN_ID") != run_id:
        raise ValueError("launcher-injected VKGR_RUN_ID does not bind claimed run")
    if recorded_env.get("VKGR_PROGRESS_PATH") != str(Path(root) / "runs" / run_id / "progress.json"):
        raise ValueError("launcher-injected VKGR_PROGRESS_PATH does not bind claimed run")
    wanted_inputs = {str(Path(path).resolve()) for path in job.get("inputs", [])}
    observed_inputs = {item.get("path") for item in record.get("inputs", []) if type(item) is dict}
    if not wanted_inputs <= observed_inputs:
        raise ValueError("scheduler run record omits a pinned job input")
    recorded_by_path = {item["path"]: item.get("sha256") for item in record.get("inputs", []) if type(item) is dict}
    if any(recorded_by_path.get(path) != ops.sha(path) for path in wanted_inputs):
        raise ValueError("scheduler run input SHA differs from current source/manifest bytes")
    return True
def verify_job(train_job, manifest_sha, manifest_path):
    import copy
    job = copy.deepcopy(train_job)
    job["run_id"] = train_job["run_id"] + "-verify"
    job["seed"] = train_job["seed"]
    argv = job["argv"]
    if "train" not in argv: raise ValueError("train job must use the registered runner train command")
    argv[argv.index("train")] = "verify"
    if "--output" not in argv: raise ValueError("train job output argument required")
    output_index = argv.index("--output")
    del argv[output_index:output_index + 2]
    argv.extend(("--manifest", str(manifest_path), "--manifest-sha", manifest_sha))
    job["inputs"] = list(dict.fromkeys([*job.get("inputs", []), str(manifest_path)]))
    job["env"] = [value for value in job.get("env", [])
                   if value.partition("=")[0] not in ("VKGR_RUN_ID", "VKGR_PROGRESS_PATH")]
    job["wall_hours"] = min(float(job.get("wall_hours", 48)), 48.0)
    return job

def driver(req):
    validate_driver_payload(req)
    wait_for_reparented_session()
    root, ops_dir = Path(req["root"]), Path(req["ops"])
    end = time.monotonic() + req["deadline_seconds"]
    results = []
    for entry in req["entries"]:
        rid = entry["run_id"]
        train_job = entry.get("train_job")
        if type(train_job) is not dict or train_job.get("run_id") != rid:
            return _finish(ops_dir, "ALERT", results, rid)
        train_id = terminal(root, rid)
        while train_id is None and time.monotonic() < end:
            try:
                state = claim_state(root, rid)
                if not state["pending"] and not state["claimed"]:
                    raise ValueError("fixed train job disappeared from pending/claimed")
                validate_claimed_run(root, train_job)
            except ValueError as exc: return _finish(ops_dir, "ALERT", results, rid, str(exc))
            time.sleep(min(req["poll_seconds"], max(0, end - time.monotonic())))
            train_id = terminal(root, rid)
        if train_id is None: return _finish(ops_dir, "DEADLINE", results, rid)
        try:
            if not validate_claimed_run(root, train_job) or not terminal_success(train_id, rid):
                return _finish(ops_dir, "ALERT", results, rid, "train claim/terminal identity failed")
        except ValueError as exc: return _finish(ops_dir, "ALERT", results, rid, str(exc))
        try:
            stdout = root / "runs" / rid / "logs/stdout.log"
            receipt = train_receipt(stdout, rid)
            manifest = root / "runs" / rid / "out/manifest.json"
            actual_manifest_sha = sha(manifest)
        except Exception as exc:
            detail = f"train receipt/manifest failure: {type(exc).__name__}: {exc}"
            return _finish(ops_dir, "ALERT", results, rid, detail)
        if actual_manifest_sha != receipt["manifest_sha256"]:
            return _finish(ops_dir, "ALERT", results, rid, "train manifest SHA differs from unique raw stdout receipt")
        try:
            vjob = verify_job(train_job, receipt["manifest_sha256"], manifest)
            ops.validate_scheduler_source(Path(entry["scheduler_code"]) / "ops")
            sys.path.insert(0, str(Path(entry["scheduler_code"]) / "ops"))
            from vkgr_ops import scheduler, launcher
            verify_state = terminal(root, vjob["run_id"])
            if verify_state is not None:
                return _finish(ops_dir, "ALERT", results, vjob["run_id"], "verify identity already has a terminal")
            state = claim_state(root, vjob["run_id"])
            if state["pending"] or state["claimed"] or state["rejected"] or (Path(root) / "runs" / vjob["run_id"]).exists():
                return _finish(ops_dir, "ALERT", results, vjob["run_id"], "verify identity already exists")
            prepared = launcher.prepare(vjob)
            if len(ops.canonical({"spec": vjob, "prepared": prepared})) >= 48 * 1024 - 2048:
                return _finish(ops_dir, "ALERT", results, vjob["run_id"], "verify launch envelope exceeds fixed limit")
            intent = ops_dir / ("intent-" + vjob["run_id"] + ".json")
            write_new(intent, {"job_sha256": ops.digest(vjob), "stage": "verify", "train_manifest_sha256": receipt["manifest_sha256"]})
            scheduler.submit(root, vjob)
        except Exception as exc:
            verify_id = rid + "-verify"
            return _finish(ops_dir, "ALERT", results, verify_id,
                           f"verify queue/prepare/submit failure: {type(exc).__name__}: {exc}")
        verify_state = terminal(root, vjob["run_id"])
        while verify_state is None and time.monotonic() < end:
            try:
                state = claim_state(root, vjob["run_id"])
                if not state["pending"] and not state["claimed"]:
                    raise ValueError("fixed verify job disappeared from pending/claimed")
                validate_claimed_run(root, vjob)
            except ValueError as exc: return _finish(ops_dir, "ALERT", results, vjob["run_id"], str(exc))
            time.sleep(min(req["poll_seconds"], max(0, end - time.monotonic())))
            verify_state = terminal(root, vjob["run_id"])
        if verify_state is None: return _finish(ops_dir, "DEADLINE", results, vjob["run_id"])
        vstdout = root / "runs" / vjob["run_id"] / "logs/stdout.log"
        if vstdout.is_symlink() or not vstdout.is_file() or vstdout.stat().st_size > 2 * 1024 * 1024:
            return _finish(ops_dir, "ALERT", results, vjob["run_id"], "bounded raw verify stdout missing")
        text = vstdout.read_text()
        try: valid_verify_run = validate_claimed_run(root, vjob)
        except ValueError as exc: return _finish(ops_dir, "ALERT", results, vjob["run_id"], str(exc))
        if (not valid_verify_run or not terminal_success(verify_state, vjob["run_id"])
                or not ops.parse_verified_stdout(text, rid, receipt["manifest_sha256"])):
            return _finish(ops_dir, "ALERT", results, vjob["run_id"], "verify claim/raw stdout binding failed")
        results.append({"run_id": rid, "verify_run_id": vjob["run_id"], "manifest_sha256": receipt["manifest_sha256"], "status": "verified"})
    return _finish(ops_dir, "COMPLETE", results, None)

def _finish(ops_dir, status, results, failed, detail=None):
    summary = {"schema": "p-watch-positive-driver-terminal-v1", "status": status,
               "results": results, "failed_run_id": failed,
               "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "detail": detail}
    write_new(Path(ops_dir) / "driver-terminal.json", summary)
    return summary

def start(request_path, request_sha, plan_path, plan_sha):
    if sha(request_path) != request_sha or sha(plan_path) != plan_sha: raise ValueError("driver request/plan SHA mismatch")
    req = load(request_path); plan = load(plan_path)
    if plan.get("sha256") != ops.digest(plan.get("entries")): raise ValueError("plan digest mismatch")
    payload = {"schema": "p-watch-positive-driver-v1", "root": req["root"], "ops": req["ops"],
               "request_sha256": request_sha, "plan_sha256": ops.digest(plan["entries"]),
               "entries": plan["entries"], "deadline_seconds": req["deadline_seconds"], "poll_seconds": req["poll_seconds"]}
    if plan.get("sha256") != payload["plan_sha256"]: raise ValueError("driver plan/entry SHA mismatch")
    validate_driver_payload(payload)
    script = Path(__file__).resolve()
    log = (Path(req["ops"]) / "driver.log").open("xb")
    child = subprocess.Popen([sys.executable, "-B", str(script), "run", "--payload", json.dumps(payload)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                             env={**os.environ, "PYTHONPATH": str(script.parents[1] / "src") + os.pathsep + str(script.parents[1] / "scripts")})
    return {"status": "started", "pid": child.pid, "session_leader": True, "requested_deadline_seconds": req["deadline_seconds"], "poll_seconds": req["poll_seconds"]}

def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False); sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("start"); s.add_argument("--request", required=True); s.add_argument("--request-sha", required=True); s.add_argument("--plan", required=True); s.add_argument("--plan-sha", required=True)
    r = sub.add_parser("run"); r.add_argument("--payload", required=True)
    a = p.parse_args(argv)
    result = start(a.request, a.request_sha, a.plan, a.plan_sha) if a.action == "start" else driver(json.loads(a.payload))
    print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
