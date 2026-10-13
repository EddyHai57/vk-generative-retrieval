"""P4H-03B local operation/controller/feed primitives; remote decisions stay blocked."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import time

from . import retrieval_sasrec_experiment as science
from . import retrieval_sasrec_plan as plan

P4H01_MANIFEST_SHA256 = "2dfe420b2760327e240e15c12c4702e03ede613e4d47ccbfa87e467d1cb855c0"
OPERATIONAL_INTEGRITY_THREAT_MODEL = {
    "protects": ["accidental-operation", "wrong-job", "duplicate-writer", "stale", "rollback", "fork", "partial-write", "ordinary-tamper"],
    "out_of_scope": "OUT_OF_SCOPE_REQUIRES_EXTERNAL_TRUST_ROOT:same-UID-malicious-self-consistent-total-rewrite",
}
COMMAND_HANDLERS = {
    "prepare-controller": lambda value, *args, **kwargs: prepare_controller(value, *args, **kwargs),
    "pilot": science.pilot_capacity,
    "pilot-verify": science.verify_pilot,
    "formal-train": science.train_slot,
    "formal-resume": science.resume_slot,
    "formal-verify": science.verify_slot,
    "select": science.select_route,
    "completion-verify": science.verify_completion,
    "summarize": science.summarize,
    "infer-route": science.infer_route,
    "infer-route-verify": science.verify_infer_route,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _emit(sink, kind, value):
    if not callable(sink):
        raise ValueError("explicit operation sink required")
    sink(kind, copy.deepcopy(value))


def build_parser():
    parser = argparse.ArgumentParser(prog="retrieval_sasrec_operation")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in COMMAND_HANDLERS:
        sub = commands.add_parser(command)
        sub.add_argument("--operation-manifest", required=True)
        sub.add_argument("--operation-manifest-sha256", required=True)
    return parser


def dispatch_science(command, *args, **kwargs):
    if command not in COMMAND_HANDLERS or command == "prepare-controller":
        raise ValueError("science command required")
    function = COMMAND_HANDLERS[command]
    return getattr(science, function.__name__)(*args, **kwargs)


def _file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def _start_ticks(pid):
    path = Path("/proc") / str(pid) / "stat"
    if path.exists():
        data = path.read_bytes()[:4096]
        fields = data[data.rfind(b")") + 2 :].split()
        return fields[19].decode()
    result = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, check=True)
    value = result.stdout.strip()
    if not value:
        raise ProcessLookupError(pid)
    return value


def process_identity(pid):
    return {
        "pid": int(pid),
        "ppid": os.getppid() if pid == os.getpid() else int(subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True, check=True).stdout.strip()),
        "pgid": os.getpgid(pid),
        "sid": os.getsid(pid),
        "start_ticks": _start_ticks(pid),
    }


_CURRENT_PROCESS_START_IDENTITY = None


def current_process_identity():
    global _CURRENT_PROCESS_START_IDENTITY
    pid = os.getpid()
    if _CURRENT_PROCESS_START_IDENTITY is None or _CURRENT_PROCESS_START_IDENTITY[0] != pid:
        _CURRENT_PROCESS_START_IDENTITY = (pid, _start_ticks(pid))
    return {
        "pid": pid,
        "ppid": os.getppid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "start_ticks": _CURRENT_PROCESS_START_IDENTITY[1],
    }


def current_publisher_identity():
    identity = current_process_identity()
    executable = Path(sys.executable).resolve()
    base = {"pid": identity["pid"], "start_ticks": identity["start_ticks"], "executable_sha256": _file_sha256(executable)}
    return {**base, "start_receipt_sha256": digest(base)}


def prepare_controller(value, identity, *, sink):
    validate_static_manifest(value)
    if identity != current_process_identity():
        raise ValueError("controller birth identity")
    receipt = {
        "schema_version": 1,
        "operation_id": value["operation_id"],
        "operation_sha256": digest(value),
        "controller_pid": identity["pid"],
        "controller_pgid": identity["pgid"],
        "controller_start_ticks": identity["start_ticks"],
        "control_root_identity": digest(str(Path(value["exclusive_control_root"]).resolve())),
        "output_root_identity": digest(str(Path(value["exclusive_output_root"]).resolve())),
        "prepared_at": time.time(),
        "state": "AWAITING_ADMISSION",
        "worker_spawned": False,
    }
    _emit(sink, "controller_start", receipt)
    return receipt


def synthetic_admission(start, g_receipt_sha256, issued_at):
    return {
        "schema_version": 1,
        "operation_id": start["operation_id"],
        "operation_sha256": start["operation_sha256"],
        "controller_start_receipt_sha256": digest(start),
        "g_receipt_sha256": g_receipt_sha256,
        "issued_at": issued_at,
        "expires_at": issued_at + 10,
        "wall_cap": 5,
        "new_bytes_cap": 1048576,
        "state": "ADMITTED",
    }


def validate_feed_directory(descriptor):
    plan.validate_feed_descriptor(descriptor)
    root = Path(descriptor["feed_root"])
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents) or not root.is_dir():
        raise ValueError("feed root path")
    current = root.stat()
    if current.st_uid != descriptor["feed_owner_uid"] or stat.S_IMODE(current.st_mode) != 0o700 or {"device": current.st_dev, "inode": current.st_ino} != descriptor["feed_root_identity"]:
        raise ValueError("feed owner/mode/identity")
    return {"owner_mode_identity_verified": True}


def acquire_synthetic_publisher_lock(root):
    root = Path(root)
    fd = os.open(root / "publisher.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    current = os.fstat(fd)
    return fd, {"name": "publisher.lock", "device": current.st_dev, "inode": current.st_ino, "birth_ctime_ns": current.st_ctime_ns, "owner_uid": current.st_uid, "creation": "O_CREAT|O_EXCL"}


def validate_publisher_identity(descriptor, observed):
    expected = descriptor.get("publisher_process")
    if not isinstance(observed, dict) or set(observed) != {"pid", "start_ticks", "executable_sha256", "start_receipt_sha256"} or observed != expected:
        raise ValueError("publisher birth/executable identity")
    if observed["start_receipt_sha256"] != digest({key: observed[key] for key in ("pid", "start_ticks", "executable_sha256")}):
        raise ValueError("publisher start receipt")
    return {"publisher_birth_and_executable_verified": True}


def synthetic_writer_inventory(descriptor, writers):
    expected = descriptor["publisher_process"]
    rows = [{"pid": row["pid"], "start_ticks": row["start_ticks"], "executable_sha256": row["executable_sha256"], "fd_modes": ["w"]} for row in writers]
    value = {
        "schema_version": 1,
        "captured_at": 100.0,
        "feed_root_identity": copy.deepcopy(descriptor["feed_root_identity"]),
        "expected_publisher": {key: expected[key] for key in ("pid", "start_ticks", "executable_sha256")},
        "writable_fd_processes": rows,
        "expected_writer_count": sum(row["pid"] == expected["pid"] and row["start_ticks"] == expected["start_ticks"] and row["executable_sha256"] == expected["executable_sha256"] for row in rows),
        "unknown_writer_count": sum(not (row["pid"] == expected["pid"] and row["start_ticks"] == expected["start_ticks"] and row["executable_sha256"] == expected["executable_sha256"]) for row in rows),
    }
    value["inventory_sha256"] = digest(value)
    return value


def validate_writer_inventory(descriptor, inventory):
    expected_fields = {"schema_version", "captured_at", "feed_root_identity", "expected_publisher", "writable_fd_processes", "expected_writer_count", "unknown_writer_count", "inventory_sha256"}
    if not isinstance(inventory, dict) or set(inventory) != expected_fields:
        raise ValueError("writer inventory schema")
    unsealed = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if inventory["inventory_sha256"] != digest(unsealed) or inventory["feed_root_identity"] != descriptor["feed_root_identity"] or inventory["expected_writer_count"] != 1 or inventory["unknown_writer_count"] != 0 or len(inventory["writable_fd_processes"]) != 1:
        raise ValueError("writer inventory")
    expected = descriptor["publisher_process"]
    row = inventory["writable_fd_processes"][0]
    if {key: row[key] for key in ("pid", "start_ticks", "executable_sha256")} != {key: expected[key] for key in ("pid", "start_ticks", "executable_sha256")} or row["fd_modes"] != ["w"]:
        raise ValueError("unknown writer")
    return {"no_other_writer_inventory_verified": True}


def _receipt_value(descriptor, seq, previous, now):
    return {
        "schema_version": 1,
        "seq": seq,
        "previous_receipt_sha256": previous,
        "g_job_identity": copy.deepcopy(descriptor["g_job_identity"]),
        "publisher_process": copy.deepcopy(descriptor["publisher_process"]),
        "publisher_lock_birth": {key: descriptor["publisher_lock"][key] for key in ("device", "inode", "birth_ctime_ns", "owner_uid")},
        "issued_at": now - 0.5 + seq * 0.1,
        "expires_at": now + 10 + seq,
        "next_expected_by": now + 20 + seq,
        "status": "OPEN",
        "payload_sha256": hashlib.sha256(f"payload:{seq}".encode()).hexdigest(),
    }


def rehash_snapshot(snapshot):
    previous = None
    for index, row in enumerate(snapshot["receipts"]):
        value = row["value"]
        value["previous_receipt_sha256"] = previous
        data_sha = digest(value)
        row.update(name=f"receipt-{value['seq']:06d}-{data_sha}.json", sha256=data_sha)
        previous = data_sha
        if index == 0:
            snapshot["descriptor"]["initial_receipt_sha256"] = data_sha
    head_row = snapshot["receipts"][-1]
    snapshot["head"] = {"seq": head_row["value"]["seq"], "receipt_name": head_row["name"], "receipt_sha256": head_row["sha256"]}
    snapshot["inventory"] = synthetic_writer_inventory(snapshot["descriptor"], [snapshot["descriptor"]["publisher_process"]])
    snapshot["descriptor"]["initial_no_other_writer_inventory_sha256"] = snapshot["inventory"]["inventory_sha256"]
    return snapshot


def synthetic_feed_snapshot(now, count=1):
    if type(count) is not int or count < 1:
        raise ValueError("receipt count")
    descriptor = plan.synthetic_feed_descriptor()
    descriptor["publisher_process"] = current_publisher_identity()
    rows = []
    previous = None
    for seq in range(count):
        value = _receipt_value(descriptor, seq, previous, now)
        data_sha = digest(value)
        rows.append({"name": f"receipt-{seq:06d}-{data_sha}.json", "sha256": data_sha, "value": value})
        previous = data_sha
    snapshot = {"descriptor": descriptor, "inventory": {}, "head": {}, "receipts": rows}
    return rehash_snapshot(snapshot)


def validate_feed_snapshot(snapshot, *, now):
    if not isinstance(snapshot, dict) or set(snapshot) != {"descriptor", "inventory", "head", "receipts"}:
        raise ValueError("feed snapshot schema")
    descriptor = snapshot["descriptor"]
    plan.validate_feed_descriptor(descriptor)
    validate_publisher_identity(descriptor, descriptor["publisher_process"])
    validate_writer_inventory(descriptor, snapshot["inventory"])
    receipts = snapshot["receipts"]
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("receipt namespace")
    by_seq = {}
    for row in receipts:
        if not isinstance(row, dict) or set(row) != {"name", "sha256", "value"} or not isinstance(row["value"], dict):
            raise ValueError("receipt record")
        value = row["value"]
        fields = {"schema_version", "seq", "previous_receipt_sha256", "g_job_identity", "publisher_process", "publisher_lock_birth", "issued_at", "expires_at", "next_expected_by", "status", "payload_sha256"}
        if set(value) != fields or value["schema_version"] != 1 or type(value["seq"]) is not int or value["seq"] in by_seq or value["status"] not in descriptor["allowed_statuses"] or value["g_job_identity"] != descriptor["g_job_identity"] or value["publisher_process"] != descriptor["publisher_process"] or value["publisher_lock_birth"] != {key: descriptor["publisher_lock"][key] for key in ("device", "inode", "birth_ctime_ns", "owner_uid")}:
            raise ValueError("receipt identity")
        actual = digest(value)
        if row["sha256"] != actual or row["name"] != f"receipt-{value['seq']:06d}-{actual}.json":
            raise ValueError("receipt name/hash")
        by_seq[value["seq"]] = row
    head = snapshot["head"]
    if not isinstance(head, dict) or set(head) != {"seq", "receipt_name", "receipt_sha256"} or head["seq"] not in by_seq:
        raise ValueError("HEAD")
    head_row = by_seq[head["seq"]]
    if (head["receipt_name"], head["receipt_sha256"]) != (head_row["name"], head_row["sha256"]):
        raise ValueError("HEAD identity")
    if 0 not in by_seq or descriptor["initial_seq"] != 0 or descriptor["initial_receipt_sha256"] != by_seq[0]["sha256"]:
        raise ValueError("initial receipt")
    previous = None
    for seq in range(head["seq"] + 1):
        row = by_seq.get(seq)
        if row is None or row["value"]["previous_receipt_sha256"] != previous:
            raise ValueError("missing/gap/rollback chain")
        if seq < head["seq"] and row["value"]["status"] != "OPEN":
            raise ValueError("terminal receipt append")
        previous = row["sha256"]
    value = head_row["value"]
    if not value["issued_at"] < value["expires_at"] <= value["next_expected_by"] or now > value["expires_at"]:
        raise ValueError("stale receipt")
    return {"head_seq": head["seq"], "action": "CONTINUE" if value["status"] == "OPEN" else "STOP_REQUESTED", "receipt_sha256": head["receipt_sha256"]}


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def spawn_synthetic_worker_tree(stop_path):
    stop_path = Path(stop_path)
    if stop_path.exists() or stop_path.is_symlink():
        raise FileExistsError("exclusive synthetic stop path")
    script = """import json,pathlib,subprocess,sys,time
stop=pathlib.Path(sys.argv[1]);preemption=stop.with_name('preemption-checkpoint.json')
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
print(json.dumps({'grandchild_pid':child.pid}),flush=True)
while not stop.exists():time.sleep(.01)
with preemption.open('xb') as f:f.write(b'one')
child.terminate();child.wait(timeout=3)
"""
    process = subprocess.Popen([sys.executable, "-B", "-c", script, str(stop_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    selector = selectors.DefaultSelector();selector.register(process.stdout, selectors.EVENT_READ);events = selector.select(3.0);selector.close()
    if not events:
        process.terminate();process.wait(timeout=3);raise TimeoutError("synthetic worker handshake")
    line = process.stdout.readline();message = json.loads(line)
    worker_identity = process_identity(process.pid);grandchild_identity = process_identity(message["grandchild_pid"])
    if worker_identity["pgid"] != process.pid or worker_identity["sid"] != process.pid or grandchild_identity["pgid"] != process.pid:
        process.terminate();process.wait(timeout=3);raise ValueError("synthetic worker PGID")
    return {"process": process, "worker_identity": worker_identity, "grandchild_identity": grandchild_identity, "stop_path": str(stop_path)}


def stop_worker_tree(handle, stop_path, *, expected_identity, seal_deadline, signal_grace):
    process = handle["process"]
    try:
        current = process_identity(process.pid)
    except ProcessLookupError:
        current = None
    if current is None or current != handle["worker_identity"] or expected_identity != handle["worker_identity"] or current["pgid"] == os.getpgrp():
        return {"state": "FAILED_UNSAFE", "signals_sent": 0, "descendant_closure_verified": False, "single_preemption_checkpoint": False}
    stop_path = Path(stop_path)
    if not stop_path.exists():
        with stop_path.open("xb") as stream:
            stream.write(canonical({"state": "STOP_REQUESTED", "worker": expected_identity}))
    signals = 0
    try:
        process.wait(timeout=seal_deadline)
    except subprocess.TimeoutExpired:
        fresh = process_identity(process.pid)
        if fresh != expected_identity or fresh["pgid"] != process.pid:
            return {"state": "FAILED_UNSAFE", "signals_sent": 0, "descendant_closure_verified": False, "single_preemption_checkpoint": False}
        os.killpg(process.pid, signal.SIGTERM);signals += 1
        try:
            process.wait(timeout=signal_grace)
        except subprocess.TimeoutExpired:
            fresh = process_identity(process.pid)
            if fresh != expected_identity:
                return {"state": "FAILED_UNSAFE", "signals_sent": signals, "descendant_closure_verified": False, "single_preemption_checkpoint": False}
            os.killpg(process.pid, signal.SIGKILL);signals += 1;process.wait(timeout=2)
    process.stdout.close();process.stderr.close()
    checkpoints = list(stop_path.parent.glob("preemption-checkpoint.json"))
    closed = process.poll() is not None and not _group_alive(handle["worker_identity"]["pgid"])
    return {"state": "CLOSED" if closed and len(checkpoints) == 1 else "FAILED_UNSAFE", "signals_sent": signals, "descendant_closure_verified": closed, "single_preemption_checkpoint": len(checkpoints) == 1}


# P4H-03B executable owner. These definitions intentionally supersede the early
# pilot-only fixtures while preserving their small public helper surface.
_REMOTE_STAGES = {"pilot", "pilot-verify", "formal-train", "formal-resume", "formal-verify", "infer-route", "infer-route-verify"}
_VERIFIER_STAGES = set(plan.VERIFIER_ROLE_SOURCES)
_LOCAL_STAGES = {"select", "completion-verify", "summarize"}


def write_immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(data).hexdigest()


def load_operation_manifest(path, expected_sha256):
    if not _pin(expected_sha256):
        raise ValueError("operation manifest SHA")
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("operation manifest bytes/SHA mismatch")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("operation manifest JSON") from error
    return value


def _operation_roles(command):
    if command == "pilot":
        return plan.synthetic_pilot_manifest()[0]["roles"], None
    if command in _VERIFIER_STAGES:
        fixture = plan.synthetic_verifier_manifest(command)
        return fixture["manifest"]["roles"], fixture["sealed"]
    if command in ("formal-train", "formal-resume"):
        fixture = plan.synthetic_verifier_manifest("formal-verify")
        allowed = set(plan.COMMON_ROLES + plan.VERIFIER_ROLE_SOURCES["formal-verify"]["input_hashes"])
        return [row for row in fixture["manifest"]["roles"] if row["role"] in allowed], None
    if command == "infer-route":
        fixture = plan.synthetic_verifier_manifest("infer-route-verify")
        allowed = set(plan.COMMON_ROLES + plan.VERIFIER_ROLE_SOURCES["infer-route-verify"]["input_hashes"])
        return [row for row in fixture["manifest"]["roles"] if row["role"] in allowed], None
    return [], None


def _has_real_closed_predecessor(sealed):
    if not isinstance(sealed, dict) or sealed.get("state") != "CLOSED" or not sealed.get("artifact_hashes"):
        return False
    verifier_stage = {"pilot": "pilot-verify", "formal-train": "formal-verify", "formal-resume": "formal-verify", "infer-route": "infer-route-verify"}.get(sealed.get("predecessor_stage"))
    if verifier_stage is None:
        return False
    try:
        output = Path(sealed["artifact_hashes"][0]["path"]).parent
        wrapper_path = output / "sealed-predecessor.json"
        wrapper_role = plan.VERIFIER_ROLE_SOURCES[verifier_stage]["sealed-wrapper"][0]
        probe = {
            "stage": verifier_stage,
            "verifier_contract": sealed,
            "roles": [{"role": wrapper_role, "path": str(wrapper_path), "sha256": _file_sha256(wrapper_path)}],
        }
        _verify_predecessor(probe)
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def synthetic_operation_manifest(command, *, workspace=None, predecessor=None, feed_descriptor_path=None, feed_descriptor_sha256=None, typed_inputs=None):
    if command not in plan.COMMAND_STAGES:
        raise ValueError("unknown command")
    stage = "pilot" if command == "prepare-controller" else command
    if stage in _VERIFIER_STAGES and not _has_real_closed_predecessor(predecessor):
        raise ValueError("real closed predecessor required")
    roles, _ = _operation_roles(stage)
    if stage in _VERIFIER_STAGES:
        sealed_inputs = {row["role"]: row for row in predecessor["input_hashes"]}
        sealed_artifacts = {row["role"]: row for row in predecessor["artifact_hashes"]}
        for row in roles:
            if row["source"] == "input_hashes":
                row["sha256"] = sealed_inputs[row["role"]]["sha256"]
            elif row["source"] == "artifact_hashes":
                evidence = sealed_artifacts[row["role"]]
                row.update(path=evidence["path"], sha256=evidence["sha256"], size=evidence["size"])
            elif row["source"] == "sealed-wrapper":
                row["sha256"] = digest(predecessor)
    root = Path(workspace).resolve() if workspace is not None else Path("/synthetic/p4h")
    output = root / f"{command}-output"
    control = root / f"{command}-control"
    if stage in _VERIFIER_STAGES:
        wrapper_role = plan.VERIFIER_ROLE_SOURCES[stage]["sealed-wrapper"][0]
        wrapper = next(row for row in roles if row["role"] == wrapper_role)
        predecessor_output = Path(predecessor["artifact_hashes"][0]["path"]).parent
        wrapper["path"] = str(predecessor_output / "sealed-predecessor.json")
        wrapper["size"] = len(canonical(predecessor))
    descriptor = plan.synthetic_feed_descriptor()
    return {
        "schema_version": 1,
        "task_id": "P4H-03B",
        "stage": stage,
        "operation_id": f"synthetic-{command}-001",
        "command": command,
        "code_root": str(root / "code"),
        "python": str(Path(sys.executable).resolve()),
        "source_sha256": P4H01_MANIFEST_SHA256,
        "exclusive_output_root": str(output),
        "exclusive_control_root": str(control),
        "roles": roles,
        "verifier_contract": copy.deepcopy(predecessor),
        "typed_inputs": copy.deepcopy(typed_inputs if typed_inputs is not None else {"args": [], "kwargs": {}}),
        "remote_decisions": copy.deepcopy(plan.REMOTE_DECISIONS),
        "caps": {"wall_seconds": 5, "new_bytes": 1048576, "seal_seconds": 2, "signal_grace_seconds": 1},
        "absolute_deadline": 200.0,
        "g_feed_descriptor": descriptor,
        "g_feed_descriptor_path": None if feed_descriptor_path is None else str(Path(feed_descriptor_path).resolve()),
        "g_feed_descriptor_sha256": feed_descriptor_sha256,
        "synthetic_local": True,
    }


def _validate_exact_stage_roles(stage, rows):
    if stage in ("formal-train", "formal-resume"):
        verifier_stage = "formal-verify"
    elif stage == "infer-route":
        verifier_stage = "infer-route-verify"
    else:
        return
    expected = set(plan.COMMON_ROLES + plan.VERIFIER_ROLE_SOURCES[verifier_stage]["input_hashes"])
    if not isinstance(rows, list) or {row.get("role") for row in rows if isinstance(row, dict)} != expected or len(rows) != len(expected):
        raise ValueError("STOP_BEFORE_PATH: stage role set")
    for row in rows:
        schema_id, fields = plan._role_spec(row["role"], "common" if row["role"] in plan.COMMON_ROLES else "input_hashes")
        if row.get("schema_id") != schema_id or row.get("schema_fields") != fields:
            raise ValueError("STOP_BEFORE_PATH: stage role schema")


def _validate_typed_inputs(value):
    if not isinstance(value, dict):
        raise ValueError("typed input bindings")
    if set(value) == {"args", "kwargs"}:
        if not isinstance(value["args"], list) or not isinstance(value["kwargs"], dict):
            raise ValueError("typed input bindings")
        bindings = value["args"] + list(value["kwargs"].values())
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {"path", "sha256"} or not isinstance(binding["path"], str) or not binding["path"].startswith("/") or not _pin(binding["sha256"]):
                raise ValueError("typed input binding")
        return "legacy-local"
    if set(value) != {"audit", "args", "kwargs"} or not isinstance(value["audit"], list) or not isinstance(value["args"], list) or not isinstance(value["kwargs"], dict):
        raise ValueError("typed input bindings")
    for binding in value["audit"]:
        if not isinstance(binding, dict) or set(binding) != {"role", "path", "sha256", "schema_id"} or not isinstance(binding["role"], str) or not binding["role"] or not isinstance(binding["schema_id"], str) or not binding["schema_id"] or not isinstance(binding["path"], str) or not binding["path"].startswith("/") or not _pin(binding["sha256"]):
            raise ValueError("typed audit binding")
    references = value["args"] + list(value["kwargs"].values())
    if any(not isinstance(reference, dict) or set(reference) != {"role"} or not isinstance(reference["role"], str) for reference in references):
        raise ValueError("typed role reference")
    return "role-audit"


def validate_static_manifest(value, path_probe=None):
    expected = {
        "schema_version", "task_id", "stage", "operation_id", "command", "code_root", "python",
        "source_sha256", "exclusive_output_root", "exclusive_control_root", "roles", "verifier_contract", "typed_inputs",
        "remote_decisions", "caps", "absolute_deadline", "g_feed_descriptor", "g_feed_descriptor_path",
        "g_feed_descriptor_sha256", "synthetic_local",
    }
    if not isinstance(value, dict) or set(value) != expected or value.get("schema_version") != 1 or value.get("task_id") != "P4H-03B":
        raise ValueError("operation envelope")
    command = value.get("command")
    rule = plan.COMMAND_STAGES.get(command)
    if rule is None or value.get("stage") not in rule["stages"]:
        raise ValueError("operation command/stage")
    stage = value["stage"]
    if stage == "pilot":
        plan.validate_pilot_manifest({"stage": stage, "roles": value["roles"], "exclusive_output_root": value["exclusive_output_root"], "exclusive_control_root": value["exclusive_control_root"]})
        if value["verifier_contract"] is not None:
            raise ValueError("pilot predecessor")
    elif stage in _VERIFIER_STAGES:
        plan.validate_verifier_manifest({"stage": stage, "roles": value["roles"]}, value["verifier_contract"])
    elif stage in _REMOTE_STAGES:
        _validate_exact_stage_roles(stage, value["roles"])
        if value["verifier_contract"] is not None:
            raise ValueError("mutating predecessor")
    elif stage in _LOCAL_STAGES:
        if value["roles"] or value["verifier_contract"] is not None:
            raise ValueError("local aggregate boundary")
    else:
        raise ValueError("operation stage")
    roots = (value["code_root"], value["python"], value["exclusive_output_root"], value["exclusive_control_root"])
    if any(not isinstance(path, str) or not path.startswith("/") or ".." in path.split("/") for path in roots) or len(set(roots[2:])) != 2:
        raise ValueError("absolute distinct roots")
    if value["source_sha256"] != P4H01_MANIFEST_SHA256 or value["remote_decisions"] != plan.REMOTE_DECISIONS or value["synthetic_local"] is not True:
        raise ValueError("frozen source/decision/local scope")
    caps = value["caps"]
    if not isinstance(caps, dict) or set(caps) != {"wall_seconds", "new_bytes", "seal_seconds", "signal_grace_seconds"} or any(type(caps[name]) is not int or caps[name] <= 0 for name in caps) or type(value["absolute_deadline"]) not in (int, float):
        raise ValueError("finite caps")
    typed_mode = _validate_typed_inputs(value["typed_inputs"])
    if stage in _REMOTE_STAGES and typed_mode == "legacy-local" and (value["typed_inputs"]["args"] or value["typed_inputs"]["kwargs"]):
        raise ValueError("remote typed inputs require accepted role audit")
    plan.validate_feed_descriptor(value["g_feed_descriptor"])
    descriptor_path = value["g_feed_descriptor_path"]
    descriptor_sha = value["g_feed_descriptor_sha256"]
    if (descriptor_path is None) != (descriptor_sha is None) or descriptor_path is not None and (not descriptor_path.startswith("/") or not _pin(descriptor_sha)):
        raise ValueError("descriptor binding")
    if path_probe is not None:
        path_probe("code_root", value["code_root"]);path_probe("python", value["python"])
        for role in value["roles"]:
            path_probe(role["role"], role["path"])
    return {"stage": stage, "operation_id": value["operation_id"], "static_verified_before_path": True}


def _live_executable_sha(pid):
    if pid == os.getpid():
        return _file_sha256(Path(sys.executable).resolve())
    result = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "txt", "-Fn"], capture_output=True, text=True, check=True)
    paths = [line[1:] for line in result.stdout.splitlines() if line.startswith("n") and Path(line[1:]).is_file()]
    if not paths:
        raise ValueError("publisher executable unavailable")
    return _file_sha256(paths[0])


def _live_publisher_identity(pid):
    identity = process_identity(pid)
    executable_sha = _live_executable_sha(pid)
    base = {"pid": pid, "start_ticks": identity["start_ticks"], "executable_sha256": executable_sha}
    return {**base, "start_receipt_sha256": digest(base)}


def _lsof_writable_processes(root):
    result = subprocess.run(["lsof", "+D", str(Path(root).resolve()), "-F", "pfan"], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise ValueError("writer inventory unavailable")
    pid = None;fd = None;rows = {}
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            pid = int(line[1:]);fd = None
        elif line.startswith("f") and pid is not None:
            fd = line[1:]
        elif line.startswith("a") and pid is not None and fd is not None and line[1:] in ("w", "u"):
            rows.setdefault(pid, []).append(line[1:])
    return rows


def _live_lock_holder(lock_path, expected_pid):
    result = subprocess.run(["lsof", "-Fpa", "--", str(Path(lock_path).resolve())], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise ValueError("lock holder inventory unavailable")
    pid = None;holders = set()
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            pid = int(line[1:])
        elif line.startswith("a") and pid is not None and line[1:] in ("w", "u"):
            holders.add(pid)
    if holders != {expected_pid}:
        raise ValueError("publisher lock holder")
    return True


def fresh_writer_inventory(descriptor):
    writers = _lsof_writable_processes(descriptor["feed_root"])
    expected = descriptor["publisher_process"]
    rows = []
    for pid, modes in sorted(writers.items()):
        try:
            live = _live_publisher_identity(pid)
        except (OSError, subprocess.SubprocessError, ValueError, ProcessLookupError) as error:
            raise ValueError("writer identity unavailable") from error
        rows.append({"pid": pid, "start_ticks": live["start_ticks"], "executable_sha256": live["executable_sha256"], "fd_modes": sorted(set(modes))})
    value = {
        "schema_version": 1, "captured_at": time.time(), "feed_root_identity": copy.deepcopy(descriptor["feed_root_identity"]),
        "expected_publisher": {key: expected[key] for key in ("pid", "start_ticks", "executable_sha256")},
        "writable_fd_processes": rows,
        "expected_writer_count": sum(row["pid"] == expected["pid"] and row["start_ticks"] == expected["start_ticks"] and row["executable_sha256"] == expected["executable_sha256"] for row in rows),
        "unknown_writer_count": sum(not (row["pid"] == expected["pid"] and row["start_ticks"] == expected["start_ticks"] and row["executable_sha256"] == expected["executable_sha256"]) for row in rows),
    }
    value["inventory_sha256"] = digest(value)
    return value


class SyntheticLiveFeed:
    def __init__(self, root, now):
        self.root = Path(root).resolve();self.root.mkdir(mode=0o700, parents=True)
        self.lock_fd, lock = acquire_synthetic_publisher_lock(self.root)
        current = self.root.stat();publisher = current_publisher_identity()
        self.descriptor = plan.synthetic_feed_descriptor()
        self.descriptor.update(feed_root=str(self.root), feed_root_identity={"device": current.st_dev, "inode": current.st_ino}, feed_owner_uid=os.getuid(), publisher_lock=lock, publisher_process=publisher)
        self.descriptor["g_job_identity"] = {"job_id_hash": hashlib.sha256(b"synthetic-g-job").hexdigest(), "start_identity_hash": hashlib.sha256(canonical(publisher)).hexdigest()}
        self.receipts = []
        first = self._publish_receipt(now, previous=None, seq=0)
        inventory = fresh_writer_inventory(self.descriptor)
        self.descriptor["initial_no_other_writer_inventory_sha256"] = inventory["inventory_sha256"]
        self.descriptor["initial_receipt_sha256"] = first["sha256"]
        self.descriptor_path = self.root.parent / (self.root.name + "-descriptor.json")
        self.descriptor_sha256 = write_immutable_json(self.descriptor_path, self.descriptor)
        os.chmod(self.descriptor_path, 0o400)

    @property
    def head_path(self):
        return self.root / "HEAD.json"

    def _atomic_head(self, row):
        value = {"seq": row["value"]["seq"], "receipt_name": row["name"], "receipt_sha256": row["sha256"]}
        temporary = self.root / f".HEAD-{os.getpid()}-{time.time_ns()}"
        with temporary.open("xb") as stream:
            stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
        os.replace(temporary, self.head_path)

    def _publish_receipt(self, now, previous, seq):
        value = _receipt_value(self.descriptor, seq, previous, now)
        value["issued_at"] = now - 0.01;value["expires_at"] = now + 60;value["next_expected_by"] = now + 120
        sha = digest(value);name = f"receipt-{seq:06d}-{sha}.json";path = self.root / name
        write_immutable_json(path, value);os.chmod(path, 0o400)
        row = {"name": name, "sha256": sha, "value": value};self.receipts.append(row);self._atomic_head(row);return row

    def publish(self, now):
        return self._publish_receipt(now, self.receipts[-1]["sha256"], len(self.receipts))

    def set_head(self, seq):
        self._atomic_head(self.receipts[seq])

    def close(self):
        if self.lock_fd is not None:
            os.close(self.lock_fd);self.lock_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def synthetic_live_feed(root, *, now):
    return SyntheticLiveFeed(root, now)


class LiveFeedController:
    def __init__(self, descriptor_path, descriptor_sha256, state_path):
        self.descriptor_path = Path(descriptor_path).resolve();self.descriptor_sha256 = descriptor_sha256;self.state_path = Path(state_path).resolve()

    def _descriptor(self):
        descriptor = load_operation_manifest(self.descriptor_path, self.descriptor_sha256)
        plan.validate_feed_descriptor(descriptor)
        return descriptor

    def _snapshot(self, descriptor):
        head_data = (Path(descriptor["feed_root"]) / descriptor["head_name"]).read_bytes()
        head = json.loads(head_data)
        receipts = []
        for path in sorted(Path(descriptor["feed_root"]).glob("receipt-*.json")):
            data = path.read_bytes();value = json.loads(data);receipts.append({"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "value": value})
        return {"descriptor": descriptor, "inventory": fresh_writer_inventory(descriptor), "head": head, "receipts": receipts}

    def observe(self, *, now):
        descriptor = self._descriptor();validate_feed_directory(descriptor)
        lock_path = Path(descriptor["feed_root"]) / descriptor["publisher_lock"]["name"]
        lock = lock_path.stat();expected_lock = descriptor["publisher_lock"]
        if {"device": lock.st_dev, "inode": lock.st_ino, "birth_ctime_ns": lock.st_ctime_ns, "owner_uid": lock.st_uid} != {key: expected_lock[key] for key in ("device", "inode", "birth_ctime_ns", "owner_uid")}:
            raise ValueError("publisher lock birth drift")
        _live_lock_holder(lock_path, descriptor["publisher_process"]["pid"])
        live = _live_publisher_identity(descriptor["publisher_process"]["pid"]);validate_publisher_identity(descriptor, live)
        snapshot = self._snapshot(descriptor)
        try:
            validate_writer_inventory(descriptor, snapshot["inventory"])
        except ValueError as error:
            raise ValueError("unknown writer") from error
        result = validate_feed_snapshot(snapshot, now=now)
        if self.state_path.exists():
            previous = json.loads(self.state_path.read_text())
            if result["head_seq"] < previous["seq"]:
                raise ValueError("feed rollback")
            if result["head_seq"] == previous["seq"] and result["receipt_sha256"] != previous["sha256"]:
                raise ValueError("feed fork")
            if result["head_seq"] > previous["seq"] + 1:
                raise ValueError("feed gap")
        state = {"seq": result["head_seq"], "sha256": result["receipt_sha256"], "g_job_identity": descriptor["g_job_identity"]}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(self.state_path.name + f".{os.getpid()}.tmp")
        temporary.write_bytes(canonical(state));os.replace(temporary, self.state_path)
        return result


def admit_controller(start, admission, feed_controller, *, now, sink):
    if not isinstance(feed_controller, LiveFeedController):
        raise TypeError("live feed controller required")
    if not isinstance(start, dict) or start.get("state") != "AWAITING_ADMISSION" or start.get("worker_spawned") is not False:
        raise ValueError("controller start state")
    expected = {"schema_version", "operation_id", "operation_sha256", "controller_start_receipt_sha256", "g_receipt_sha256", "issued_at", "expires_at", "wall_cap", "new_bytes_cap", "state"}
    if not isinstance(admission, dict) or set(admission) != expected or admission["schema_version"] != 1 or admission["state"] != "ADMITTED" or admission["operation_id"] != start["operation_id"] or admission["operation_sha256"] != start["operation_sha256"] or admission["controller_start_receipt_sha256"] != digest(start) or not admission["issued_at"] <= now < admission["expires_at"] or admission["wall_cap"] <= 0 or admission["new_bytes_cap"] <= 0:
        raise ValueError("static admission")
    feed_result = feed_controller.observe(now=now)
    if feed_result["action"] != "CONTINUE" or admission["g_receipt_sha256"] != feed_result["receipt_sha256"]:
        raise ValueError("fresh G receipt")
    result = {**admission, "controller_start_receipt_sha256": digest(start), "state": "ADMITTED", "worker_spawned": False}
    _emit(sink, "admission", result);return result


def run_atomic_stage(feed_controller, stage, body, *, now):
    if not isinstance(feed_controller, LiveFeedController) or not isinstance(stage, str) or not callable(body) or not callable(now):
        raise ValueError("atomic stage contract")
    before = feed_controller.observe(now=now())
    if before["action"] != "CONTINUE":
        raise ValueError("feed stopped before stage")
    result = body()
    after = feed_controller.observe(now=now())
    if after["action"] != "CONTINUE":
        raise ValueError("feed stopped after stage")
    return result


def _artifact_set_sha(rows):
    return hashlib.sha256(plan.canonical_bytes(rows)).hexdigest()


def _write_private_record(value, name, payload):
    root = Path(value["exclusive_output_root"]) / "private"
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    sha = write_immutable_json(path, payload)
    return str(path), sha, path.stat().st_size


def _checkpoint_entry(value, name, source_sha, epoch, step):
    if source_sha is None:
        return None
    payload = {
        "schema_version": 1,
        "operation_id": value["operation_id"],
        "checkpoint_role": name,
        "source_checkpoint_sha256": source_sha,
        "epoch": epoch,
        "step": step,
    }
    path, sha, size = _write_private_record(value, f"formal-checkpoint-{name}.json", payload)
    return {"path": path, "sha256": sha, "size": size, "checkpoint_role": name, "source_checkpoint_sha256": source_sha, "epoch": epoch, "step": step, "operation_id": value["operation_id"]}


def _route_shard_entries(value, shards):
    result = []
    for ordinal, shard in enumerate(shards):
        records = copy.deepcopy(shard.get("records", []))
        start = shard.get("start", sum(item["count"] for item in result))
        end = shard.get("end", start + len(records))
        payload = {"schema_version": 1, "operation_id": value["operation_id"], "shard": ordinal, "start": start, "end": end, "records": records}
        path, sha, size = _write_private_record(value, f"route-shard-{ordinal:06d}.json", payload)
        result.append({"shard": ordinal, "range": [start, end], "path": path, "sha256": sha, "size": size, "count": len(records)})
    return result


def _artifact_payload(role, value, typed_result, *, spent, events, opened):
    operation_id = value["operation_id"]
    base = {"schema_version": 1, "operation_id": operation_id}
    if role == "pilot-capacity-measurement-index":
        arms = copy.deepcopy(typed_result["arms"])
        measurements = {arm: copy.deepcopy(row["measurements"]) for arm, row in arms.items()}
        return {**base, "arms": arms, "measurements": measurements, "units": {"seconds": "seconds", "bytes": "bytes"}}
    if role.endswith("resource-ledger"):
        return {**base, "samples": [{"wall_seconds": spent["wall_seconds"], "new_bytes": spent["new_bytes"]}], "peak": copy.deepcopy(spent), "caps": copy.deepcopy(value["caps"]), "spent": copy.deepcopy(spent)}
    if role.endswith("stage-event-ledger"):
        return {**base, "events": copy.deepcopy(events)}
    if role.endswith("opened-artifact-ledger"):
        opened_map = {row["role"]: row["sha256"] for row in opened}
        return {**base, "opened": copy.deepcopy(opened), "opened_roles": sorted(opened_map), "opened_sha256": {name: opened_map[name] for name in sorted(opened_map)}}
    if role == "formal-checkpoint-index":
        checkpoint_events = [row for row in events if row.get("kind") == "checkpoint" and isinstance(row.get("payload"), dict)]
        if checkpoint_events:
            checkpoints = checkpoint_events[-1]["payload"].get("checkpoints", {})
            return {**base, "best": copy.deepcopy(checkpoints.get("best")), "last": copy.deepcopy(checkpoints.get("last")), "preemption": copy.deepcopy(checkpoints.get("preemption"))}
        public = typed_result.get("public", typed_result) if isinstance(typed_result, dict) else {}
        checkpoints = public.get("checkpoint_roles", {})
        evaluations = public.get("evaluations", [])
        latest = evaluations[-1] if evaluations else {}
        epoch = latest.get("epoch", len(evaluations))
        step = latest.get("step", len(public.get("step_records", [])))
        return {
            **base,
            "best": _checkpoint_entry(value, "best", checkpoints.get("best"), epoch, step),
            "last": _checkpoint_entry(value, "last", checkpoints.get("last"), epoch, step),
            "preemption": _checkpoint_entry(value, "preemption", checkpoints.get("preemption"), epoch, step),
        }
    if role == "formal-epoch-metric-index":
        public = typed_result.get("public", typed_result) if isinstance(typed_result, dict) else {}
        records = copy.deepcopy(public.get("evaluations", []))
        return {**base, "records": records, "selection_replay_sha256": digest(public.get("selection", {}))}
    if role == "formal-prediction-hash-index":
        public = typed_result.get("public", typed_result) if isinstance(typed_result, dict) else {}
        evaluations = public.get("evaluations", [])
        hashes = [row.get("prediction_sha256") for row in evaluations]
        return {**base, "epoch_hashes": hashes, "fresh_best_hash": hashes[-1] if hashes else None}
    if role == "formal-draw-mask-provenance":
        public = typed_result.get("public", typed_result) if isinstance(typed_result, dict) else {}
        records = copy.deepcopy(public.get("step_records", []))
        return {**base, "step_records": records, "aggregate_sha256": digest(records)}
    if role == "route-shard-index":
        route_events = [row["payload"] for row in events if row.get("kind") == "route_shard" and isinstance(row.get("payload"), dict)]
        if route_events:
            shards = [{key: event[key] for key in ("shard", "range", "path", "sha256", "size", "count")} for event in route_events]
        else:
            shards = _route_shard_entries(value, typed_result.get("shards", []))
        return {**base, "shards": shards, "query_coverage_sha256": digest(typed_result.get("query_ids", []))}
    if role == "route-candidate-hash-index":
        shards = typed_result.get("shards", [])
        hashes = [digest(row.get("records", [])) for row in shards]
        return {**base, "per_shard_hashes": hashes, "aggregate_sha256": digest(hashes)}
    raise ValueError("unsupported artifact role")


def _write_artifact_set(value, typed_result, *, spent, events, opened):
    groups = {"pilot": "pilot-verify", "formal-train": "formal-verify", "formal-resume": "formal-verify", "infer-route": "infer-route-verify"}
    verifier = groups.get(value["stage"])
    if verifier is None:
        return []
    output = Path(value["exclusive_output_root"])
    roles = plan.VERIFIER_ROLE_SOURCES[verifier]["artifact_hashes"]
    index_role = next(role for role in roles if role.endswith("artifact-index"))
    artifacts = []
    for role in roles:
        if role == index_role:
            continue
        payload = _artifact_payload(role, value, typed_result, spent=spent, events=events, opened=opened)
        path = output / f"{role}.json"
        sha = write_immutable_json(path, payload)
        artifacts.append({"role": role, "path": str(path), "sha256": sha, "size": path.stat().st_size})
    index_payload = {
        "schema_version": 1,
        "operation_id": value["operation_id"],
        "artifacts": copy.deepcopy(artifacts),
        "complete_artifact_set_sha256": _artifact_set_sha(artifacts),
    }
    index_path = output / f"{index_role}.json"
    index_sha = write_immutable_json(index_path, index_payload)
    artifacts.append({"role": index_role, "path": str(index_path), "sha256": index_sha, "size": index_path.stat().st_size})
    return artifacts


def _exact_fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} artifact schema")


def _read_private_json(item, label):
    path = Path(item.get("path", ""))
    if not path.is_file() or type(item.get("size")) is not int or item["size"] < 0 or path.stat().st_size != item["size"] or not _pin(item.get("sha256")) or _file_sha256(path) != item["sha256"]:
        raise ValueError(f"{label} private bytes")
    try:
        return json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} private JSON") from error


def _validate_checkpoint_private(item, expected_role, operation_id):
    fields = {"path", "sha256", "size", "checkpoint_role", "source_checkpoint_sha256", "epoch", "step", "operation_id"}
    _exact_fields(item, fields, "checkpoint index")
    if item["checkpoint_role"] != expected_role or item["operation_id"] != operation_id or not _pin(item["source_checkpoint_sha256"]) or type(item["epoch"]) is not int or item["epoch"] < 0 or type(item["step"]) is not int or item["step"] < 0:
        raise ValueError("checkpoint public identity")
    payload = _read_private_json(item, "checkpoint")
    private_fields = {"schema_version", "operation_id", "checkpoint_role", "source_checkpoint_sha256", "epoch", "step"}
    _exact_fields(payload, private_fields, "checkpoint private")
    expected = {"schema_version": 1, "operation_id": operation_id, "checkpoint_role": expected_role, "source_checkpoint_sha256": item["source_checkpoint_sha256"], "epoch": item["epoch"], "step": item["step"]}
    if payload != expected:
        raise ValueError("checkpoint private identity")
    return payload


def _validate_route_private(item, operation_id, expected_ordinal, expected_query_ids, catalog):
    fields = {"shard", "range", "path", "sha256", "size", "count"}
    _exact_fields(item, fields, "route shard index")
    if item["shard"] != expected_ordinal or not isinstance(item["range"], list) or len(item["range"]) != 2 or any(type(bound) is not int or bound < 0 for bound in item["range"]) or item["range"][0] > item["range"][1] or type(item["count"]) is not int or item["count"] < 0:
        raise ValueError("route shard public identity")
    payload = _read_private_json(item, "route shard")
    _exact_fields(payload, {"schema_version", "operation_id", "shard", "start", "end", "records"}, "route shard private")
    if payload["schema_version"] != 1 or payload["operation_id"] != operation_id or payload["shard"] != expected_ordinal or [payload["start"], payload["end"]] != item["range"] or payload["end"] - payload["start"] != item["count"] or not isinstance(payload["records"], list) or len(payload["records"]) != item["count"]:
        raise ValueError("route shard private identity")
    if expected_query_ids is not None and [row.get("query_id") for row in payload["records"] if isinstance(row, dict)] != list(expected_query_ids):
        raise ValueError("route query order/coverage")
    for row in payload["records"]:
        if not isinstance(row, dict) or set(row) != {"query_id", "candidates", "fallback"} or type(row["query_id"]) is not int or not isinstance(row["candidates"], list) or len(row["candidates"]) != 80 or len(set(row["candidates"])) != 80 or any(type(candidate) is not int for candidate in row["candidates"]) or catalog is not None and any(candidate not in catalog for candidate in row["candidates"]) or type(row["fallback"]) is not bool:
            raise ValueError("route strict top80")
    return payload


def _validate_opened_role_bijection(accepted_roles, opened_rows):
    if not isinstance(accepted_roles, list) or not isinstance(opened_rows, list):
        raise ValueError("opened role bijection schema")
    expected = {}
    for row in accepted_roles:
        if not isinstance(row, dict) or not {"role", "path", "sha256", "schema_id"} <= set(row) or row["role"] in expected:
            raise ValueError("accepted role set")
        expected[row["role"]] = {key: row[key] for key in ("role", "path", "sha256", "schema_id")}
    actual = {}
    for row in opened_rows:
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "size", "schema_id"} or row["role"] in actual or type(row["size"]) is not int or row["size"] < 0:
            raise ValueError("opened role audit")
        actual[row["role"]] = {key: row[key] for key in ("role", "path", "sha256", "schema_id")}
    if actual != expected:
        raise ValueError("opened roles do not match accepted roles")
    return True


def _validate_artifact_payload(role, payload, sealed, evidence):
    identity = {"schema_version", "operation_id"}
    if payload.get("schema_version") != 1 or payload.get("operation_id") != sealed["predecessor_operation_id"]:
        raise ValueError("artifact operation identity")
    if role == "pilot-capacity-measurement-index":
        _exact_fields(payload, identity | {"arms", "measurements", "units"}, role)
        if set(payload["arms"]) != {"like", "observed"} or set(payload["measurements"]) != {"like", "observed"} or payload["units"] != {"seconds": "seconds", "bytes": "bytes"}:
            raise ValueError("artifact capacity semantics")
        for arm in ("like", "observed"):
            if payload["measurements"][arm] != payload["arms"][arm].get("measurements") or set(payload["measurements"][arm]) != set(science.PILOT_MEASUREMENTS):
                raise ValueError("artifact capacity measurements")
    elif role.endswith("resource-ledger"):
        _exact_fields(payload, identity | {"samples", "peak", "caps", "spent"}, role)
        if not isinstance(payload["samples"], list) or not payload["samples"] or payload["spent"] != sealed["spent"] or payload["spent"] != evidence["caps_spent"]["spent"] or payload["caps"] != evidence["caps_spent"]["caps"]:
            raise ValueError("artifact resource semantics")
    elif role.endswith("stage-event-ledger"):
        _exact_fields(payload, identity | {"events"}, role)
        allowed = {
            "pilot": {"controller_start", "admission", "pilot_summary", "stop", "seal"},
            "formal-train": {"controller_start", "admission", "optimizer_step", "epoch_validation", "checkpoint", "stop", "seal"},
            "formal-resume": {"controller_start", "admission", "optimizer_step", "epoch_validation", "checkpoint", "stop", "seal"},
            "infer-route": {"controller_start", "admission", "route_shard", "stop", "seal"},
        }[sealed["predecessor_stage"]]
        if payload["events"] != evidence["events"] or not payload["events"] or any(row.get("kind") not in allowed for row in payload["events"] if isinstance(row, dict)):
            raise ValueError("artifact event semantics")
    elif role.endswith("opened-artifact-ledger"):
        _exact_fields(payload, identity | {"opened", "opened_roles", "opened_sha256"}, role)
        opened_map = {row.get("role"): row.get("sha256") for row in payload["opened"] if isinstance(row, dict)} if isinstance(payload["opened"], list) else None
        if opened_map is None or len(opened_map) != len(payload["opened"]) or payload["opened_roles"] != sorted(payload["opened_sha256"]) or payload["opened_sha256"] != opened_map or payload["opened"] != evidence["opened"]:
            raise ValueError("artifact opened semantics")
    elif role.endswith("artifact-index"):
        _exact_fields(payload, identity | {"artifacts", "complete_artifact_set_sha256"}, role)
        other = [row for row in sealed["artifact_hashes"] if row["role"] != role]
        if payload["artifacts"] != other or payload["complete_artifact_set_sha256"] != _artifact_set_sha(other):
            raise ValueError("artifact index semantics")
    else:
        expected = {
            "formal-checkpoint-index": {"best", "last", "preemption"},
            "formal-epoch-metric-index": {"records", "selection_replay_sha256"},
            "formal-prediction-hash-index": {"epoch_hashes", "fresh_best_hash"},
            "formal-draw-mask-provenance": {"step_records", "aggregate_sha256"},
            "route-shard-index": {"shards", "query_coverage_sha256"},
            "route-candidate-hash-index": {"per_shard_hashes", "aggregate_sha256"},
        }.get(role)
        if expected is None:
            raise ValueError("unknown artifact role")
        _exact_fields(payload, identity | expected, role)
        if role == "formal-checkpoint-index":
            for name in ("best", "last", "preemption"):
                item = payload[name]
                if item is not None:
                    _validate_checkpoint_private(item, name, sealed["predecessor_operation_id"])
        if role == "formal-epoch-metric-index" and (not isinstance(payload["records"], list) or not _pin(payload["selection_replay_sha256"])):
            raise ValueError("artifact metric semantics")
        if role == "formal-prediction-hash-index" and (not isinstance(payload["epoch_hashes"], list) or any(not _pin(item) for item in payload["epoch_hashes"]) or payload["fresh_best_hash"] is not None and not _pin(payload["fresh_best_hash"])):
            raise ValueError("artifact prediction semantics")
        if role == "formal-draw-mask-provenance" and (not isinstance(payload["step_records"], list) or payload["aggregate_sha256"] != digest(payload["step_records"])):
            raise ValueError("artifact draw semantics")
        if role == "route-shard-index":
            if not isinstance(payload["shards"], list) or not _pin(payload["query_coverage_sha256"]):
                raise ValueError("artifact shard semantics")
            query_ids = []
            expected_start = 0
            for ordinal, item in enumerate(payload["shards"]):
                if not isinstance(item, dict) or item.get("range", [None])[0] != expected_start:
                    raise ValueError("artifact shard range continuity")
                private = _validate_route_private(item, sealed["predecessor_operation_id"], ordinal, None, evidence.get("catalog"))
                query_ids.extend(row["query_id"] for row in private["records"])
                expected_start = private["end"]
            if len(query_ids) != len(set(query_ids)) or digest(query_ids) != payload["query_coverage_sha256"] or evidence.get("query_ids") is not None and query_ids != evidence["query_ids"]:
                raise ValueError("artifact query coverage")
        if role == "route-candidate-hash-index" and (not isinstance(payload["per_shard_hashes"], list) or any(not _pin(item) for item in payload["per_shard_hashes"]) or payload["aggregate_sha256"] != digest(payload["per_shard_hashes"])):
            raise ValueError("artifact route hash semantics")


def _validate_route_artifact_closure(payloads, sealed, evidence):
    if sealed.get("predecessor_stage") != "infer-route":
        return True
    shard_index = payloads.get("route-shard-index")
    candidate_index = payloads.get("route-candidate-hash-index")
    if not isinstance(shard_index, dict) or not isinstance(candidate_index, dict):
        raise ValueError("route closure artifact set")
    _validate_artifact_payload("route-shard-index", shard_index, sealed, evidence)
    _validate_artifact_payload("route-candidate-hash-index", candidate_index, sealed, evidence)
    route_events = [row for row in evidence.get("events", []) if isinstance(row, dict) and row.get("kind") == "route_shard"]
    _validate_science_event_audit("infer-route", sealed["predecessor_operation_id"], route_events)
    shards = shard_index["shards"]
    hashes = candidate_index["per_shard_hashes"]
    if not shards or len(route_events) != len(shards) or len(hashes) != len(shards):
        raise ValueError("route closure cardinality")
    for ordinal, (item, event, candidate_hash) in enumerate(zip(shards, route_events, hashes)):
        payload = event["payload"]
        public_event = {key: payload[key] for key in ("shard", "range", "path", "sha256", "size", "count")}
        if public_event != item or payload["shard"] != ordinal or payload["record_count"] != item["count"]:
            raise ValueError("route shard event/index closure")
        private = _validate_route_private(item, sealed["predecessor_operation_id"], ordinal, None, evidence.get("catalog"))
        records_sha = digest(private["records"])
        if payload["records_sha256"] != records_sha or candidate_hash != records_sha:
            raise ValueError("route records hash closure")
    return True


def _sealed_for_stage(value, evidence_hashes, artifacts, terminal_sha, closure_sha, spent):
    stage = value["stage"]
    verifier = {"pilot": "pilot-verify", "formal-train": "formal-verify", "formal-resume": "formal-verify", "infer-route": "infer-route-verify"}.get(stage)
    if verifier is None:
        return None
    groups = plan.VERIFIER_ROLE_SOURCES[verifier]
    role_rows = {row["role"]: row for row in value["roles"]}
    inputs = [{"role": role, "sha256": role_rows[role]["sha256"]} for role in groups["input_hashes"]]
    return {
        "schema_version": 1,
        "predecessor_stage": stage,
        "predecessor_operation_id": value["operation_id"],
        "predecessor_operation_manifest_sha256": digest(value),
        "predecessor_output_root_identity": digest(str(Path(value["exclusive_output_root"]).resolve())),
        "predecessor_control_root_identity": digest(str(Path(value["exclusive_control_root"]).resolve())),
        "controller_start_receipt_sha256": evidence_hashes["controller-start.json"],
        "admission_sha256": evidence_hashes["admission.json"],
        "final_g_receipt_sha256": evidence_hashes["final-g-observation.json"],
        "worker_birth_inventory_sha256": evidence_hashes["worker-birth-inventory.json"],
        "input_hashes": inputs,
        "artifact_hashes": artifacts,
        "complete_artifact_set_sha256": _artifact_set_sha(artifacts),
        "terminal_receipt_sha256": terminal_sha,
        "closure_receipt_sha256": closure_sha,
        "spent": spent,
        "state": "CLOSED",
    }


def _read_json_bound(path, expected_sha, label):
    path = Path(path)
    if not path.is_file() or _file_sha256(path) != expected_sha:
        raise ValueError(f"sealed {label} hash")
    try:
        return json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"sealed {label} JSON") from error


_EVIDENCE_FILES = {
    "controller-start": "controller-start.json",
    "admission": "admission.json",
    "g-observations": "g-observations.json",
    "final-g-observation": "final-g-observation.json",
    "worker-birth-inventory": "worker-birth-inventory.json",
    "worker-descendant-closure": "worker-descendant-closure.json",
    "caps-spent": "caps-spent.json",
    "stage-events": "stage-events.json",
    "opened-artifacts": "opened-artifacts.json",
    "terminal-preimage": "terminal-preimage.json",
    "closure-preimage": "closure-preimage.json",
}
_COMPLETE_EVIDENCE_ROLES = set(_EVIDENCE_FILES) | {"artifacts"}


def _validate_science_event_audit(stage, operation_id, rows):
    science_rows = [row for row in rows if row.get("kind") not in {"controller_start", "admission", "stop", "seal"}]
    required = {
        "pilot": {"pilot_summary"}, "formal-train": {"optimizer_step", "epoch_validation", "checkpoint"},
        "formal-resume": {"optimizer_step", "epoch_validation", "checkpoint"}, "infer-route": {"route_shard"},
        "pilot-verify": {"pilot_verified"}, "formal-verify": {"fresh_export_exact"}, "infer-route-verify": {"route_verified"},
    }[stage]
    if not required <= {row.get("kind") for row in science_rows}:
        raise ValueError("missing science sink event")
    singleton = {"pilot_summary", "checkpoint", "pilot_verified", "fresh_export_exact", "route_verified"}
    if any(sum(row.get("kind") == kind for row in science_rows) > 1 for kind in singleton):
        raise ValueError("duplicate science sink event")
    identity_fields = {
        "pilot_summary": {"query_count", "query_identity_sha256", "artifact_identity"},
        "optimizer_step": {"slot_id", "epoch", "step", "draw_sha256", "artifact_identity"},
        "epoch_validation": {"slot_id", "epoch", "step", "prediction_sha256", "artifact_identity"},
        "checkpoint": {"slot_id", "epoch", "step", "checkpoints", "artifact_identity"},
        "fresh_export_exact": {"slot_id", "epoch", "step", "checkpoint_sha256", "prediction_sha256", "artifact_identity"},
        "route_shard": {"shard", "range", "record_count", "records_sha256", "route_identity_sha256", "artifact_identity", "path", "sha256", "size", "count"},
        "pilot_verified": {"query_identity_sha256", "artifact_identity"},
        "route_verified": {"query_count", "route_identity_sha256", "artifact_identity"},
    }
    for row in science_rows:
        payload = row.get("payload")
        fields = identity_fields.get(row.get("kind"))
        if row.get("operation_id") != operation_id or row.get("state") != "SUCCEEDED" or fields is None or not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("science sink event identity")
        for name, item in payload.items():
            if name.endswith("sha256") and not _pin(item):
                raise ValueError("science sink SHA identity")
        if payload.get("artifact_identity") == "private_path_sha256_size":
            path = Path(payload.get("path", ""))
            if not path.is_file() or path.stat().st_size != payload.get("size") or _file_sha256(path) != payload.get("sha256"):
                raise ValueError("science sink artifact identity")
    return science_rows


def _validate_disk_evidence(output, operation_id, manifest_sha, terminal, closure, artifact_rows, stage):
    terminal_preimage_fields = {"schema_version", "operation_id", "operation_manifest_sha256", "output_root_identity", "control_root_identity", "typed_result_sha256", "status", "spent", "predecessor_operation_manifest_sha256"}
    closure_preimage_fields = {"schema_version", "operation_id", "operation_manifest_sha256", "worker_descendant_closure_sha256", "worker_closed", "state"}
    _exact_fields(terminal, terminal_preimage_fields | {"evidence_sha256", "complete_evidence_sha256"}, "terminal")
    _exact_fields(closure, closure_preimage_fields | {"terminal_sha256", "complete_evidence_sha256"}, "closure")
    evidence = terminal.get("complete_evidence_sha256")
    if not isinstance(evidence, dict) or set(evidence) != _COMPLETE_EVIDENCE_ROLES or any(not _pin(pin) for pin in evidence.values()):
        raise ValueError("complete evidence map schema")
    if closure.get("complete_evidence_sha256") != evidence:
        raise ValueError("terminal/closure evidence map")
    values = {role: _read_json_bound(Path(output) / filename, evidence[role], role) for role, filename in _EVIDENCE_FILES.items()}
    if evidence["artifacts"] != _artifact_set_sha(artifact_rows):
        raise ValueError("complete evidence artifacts")
    terminal_preimage = {key: value for key, value in terminal.items() if key not in {"evidence_sha256", "complete_evidence_sha256"}}
    if values["terminal-preimage"] != terminal_preimage:
        raise ValueError("terminal preimage")
    closure_preimage = {key: value for key, value in closure.items() if key not in {"terminal_sha256", "complete_evidence_sha256"}}
    if values["closure-preimage"] != closure_preimage:
        raise ValueError("closure preimage")
    for role, payload in values.items():
        if payload.get("operation_id") != operation_id:
            raise ValueError(f"{role} operation identity")
    start = values["controller-start"]
    admission = values["admission"]
    observations = values["g-observations"]
    final_g = values["final-g-observation"]
    worker = values["worker-birth-inventory"]
    descendants = values["worker-descendant-closure"]
    caps_spent = values["caps-spent"]
    events = values["stage-events"]
    opened = values["opened-artifacts"]
    _exact_fields(start, {"schema_version", "operation_id", "operation_sha256", "controller_pid", "controller_pgid", "controller_start_ticks", "control_root_identity", "output_root_identity", "prepared_at", "state", "worker_spawned"}, "controller start")
    _exact_fields(admission, {"schema_version", "operation_id", "operation_sha256", "controller_start_receipt_sha256", "g_receipt_sha256", "issued_at", "expires_at", "wall_cap", "new_bytes_cap", "state", "worker_spawned"}, "admission")
    _exact_fields(observations, {"schema_version", "operation_id", "observations"}, "G observations")
    _exact_fields(final_g, {"schema_version", "operation_id", "admission_receipt_sha256", "head_seq", "action", "receipt_sha256"}, "final G")
    _exact_fields(worker, {"schema_version", "operation_id", "worker", "descendants"}, "worker birth")
    _exact_fields(descendants, {"schema_version", "operation_id", "state", "signals_sent", "descendant_closure_verified", "single_preemption_checkpoint"}, "worker closure")
    _exact_fields(caps_spent, {"schema_version", "operation_id", "caps", "spent"}, "caps spent")
    _exact_fields(events, {"schema_version", "operation_id", "events"}, "stage events")
    _exact_fields(opened, {"schema_version", "operation_id", "opened", "opened_roles", "opened_sha256"}, "opened artifacts")
    if start.get("operation_sha256") != manifest_sha or start.get("state") != "AWAITING_ADMISSION" or start.get("worker_spawned") is not False:
        raise ValueError("controller start semantics")
    if admission.get("operation_sha256") != manifest_sha or admission.get("controller_start_receipt_sha256") != digest(start) or admission.get("state") != "ADMITTED" or admission.get("worker_spawned") is not False or not admission.get("issued_at", 0) < admission.get("expires_at", 0):
        raise ValueError("admission semantics")
    if not isinstance(observations.get("observations"), list) or len(observations["observations"]) < 3 or final_g.get("admission_receipt_sha256") != admission.get("g_receipt_sha256"):
        raise ValueError("G observation semantics")
    if not isinstance(worker.get("worker"), dict) or not isinstance(worker.get("descendants"), list) or not worker["descendants"] or worker["worker"].get("pgid") == start.get("controller_pgid") or any(row.get("pgid") != worker["worker"].get("pgid") for row in worker["descendants"]):
        raise ValueError("worker birth semantics")
    if descendants.get("state") != "CLOSED" or descendants.get("descendant_closure_verified") is not True:
        raise ValueError("worker closure semantics")
    if caps_spent.get("spent") != terminal.get("spent") or caps_spent.get("caps", {}).get("wall_seconds", 0) <= 0 or terminal.get("spent", {}).get("wall_seconds", -1) < 0 or terminal["spent"]["wall_seconds"] > caps_spent["caps"]["wall_seconds"]:
        raise ValueError("caps/spent semantics")
    event_rows = events.get("events")
    allowed_kinds = {
        "pilot": {"controller_start", "admission", "pilot_summary", "stop", "seal"},
        "formal-train": {"controller_start", "admission", "optimizer_step", "epoch_validation", "checkpoint", "stop", "seal"},
        "formal-resume": {"controller_start", "admission", "optimizer_step", "epoch_validation", "checkpoint", "stop", "seal"},
        "infer-route": {"controller_start", "admission", "route_shard", "stop", "seal"},
        "pilot-verify": {"controller_start", "admission", "pilot_verified", "stop", "seal"},
        "formal-verify": {"controller_start", "admission", "fresh_export_exact", "stop", "seal"},
        "infer-route-verify": {"controller_start", "admission", "route_verified", "stop", "seal"},
    }[stage]
    required_science = {
        "pilot": {"pilot_summary"}, "formal-train": {"optimizer_step", "epoch_validation", "checkpoint"},
        "formal-resume": {"optimizer_step", "epoch_validation", "checkpoint"}, "infer-route": {"route_shard"},
        "pilot-verify": {"pilot_verified"}, "formal-verify": {"fresh_export_exact"}, "infer-route-verify": {"route_verified"},
    }[stage]
    if not isinstance(event_rows, list) or [row.get("sequence") for row in event_rows] != list(range(len(event_rows))) or [row.get("kind") for row in event_rows[:2]] != ["controller_start", "admission"] or [row.get("kind") for row in event_rows[-2:]] != ["stop", "seal"] or not required_science <= {row.get("kind") for row in event_rows} or any(row.get("kind") not in allowed_kinds or row.get("operation_id") != operation_id or type(row.get("observed_at")) not in (int, float) or not isinstance(row.get("state"), str) for row in event_rows) or [row["observed_at"] for row in event_rows] != sorted(row["observed_at"] for row in event_rows):
        raise ValueError("stage event semantics")
    _validate_science_event_audit(stage, operation_id, event_rows)
    if opened.get("opened_roles") != sorted(opened.get("opened_sha256", {})) or not isinstance(opened.get("opened"), list) or {row.get("role"): row.get("sha256") for row in opened["opened"] if isinstance(row, dict)} != opened.get("opened_sha256") or any(not _pin(pin) for pin in opened.get("opened_sha256", {}).values()):
        raise ValueError("opened artifact semantics")
    if closure.get("state") != "CLOSED" or closure.get("worker_closed") is not True or closure.get("worker_descendant_closure_sha256") != evidence["worker-descendant-closure"]:
        raise ValueError("closure semantics")
    return values


def _verify_predecessor(value):
    sealed = value["verifier_contract"]
    if not isinstance(sealed, dict) or sealed.get("state") != "CLOSED":
        raise ValueError("real closed predecessor required")
    wrapper_role = plan.VERIFIER_ROLE_SOURCES[value["stage"]]["sealed-wrapper"][0]
    wrapper = next(row for row in value["roles"] if row["role"] == wrapper_role)
    wrapper_path = Path(wrapper["path"])
    if not wrapper_path.is_file() or wrapper_path.read_bytes() != canonical(sealed) or wrapper["sha256"] != _file_sha256(wrapper_path):
        raise ValueError("sealed predecessor wrapper binding")
    output = wrapper_path.parent
    controller_start = _read_json_bound(output / "controller-start.json", sealed["controller_start_receipt_sha256"], "controller start")
    admission = _read_json_bound(output / "admission.json", sealed["admission_sha256"], "admission")
    final_g = _read_json_bound(output / "final-g-observation.json", sealed["final_g_receipt_sha256"], "final G")
    worker = _read_json_bound(output / "worker-birth-inventory.json", sealed["worker_birth_inventory_sha256"], "worker birth")
    terminal = _read_json_bound(output / "terminal.json", sealed["terminal_receipt_sha256"], "terminal")
    closure = _read_json_bound(output / "closure.json", sealed["closure_receipt_sha256"], "closure")
    _validate_disk_evidence(output, sealed["predecessor_operation_id"], sealed["predecessor_operation_manifest_sha256"], terminal, closure, sealed.get("artifact_hashes", []), sealed["predecessor_stage"])
    complete_map = terminal["complete_evidence_sha256"]
    if (
        sealed["controller_start_receipt_sha256"] != complete_map["controller-start"]
        or sealed["admission_sha256"] != complete_map["admission"]
        or sealed["final_g_receipt_sha256"] != complete_map["final-g-observation"]
        or sealed["worker_birth_inventory_sha256"] != complete_map["worker-birth-inventory"]
    ):
        raise ValueError("sealed/terminal evidence cross-binding")
    evidence = terminal.get("evidence_sha256")
    if not isinstance(evidence, dict):
        raise ValueError("sealed terminal evidence")
    events = _read_json_bound(output / "stage-events.json", evidence.get("stage-events.json"), "events")
    opened = _read_json_bound(output / "opened-artifacts.json", evidence.get("opened-artifacts.json"), "opened")
    caps_spent = _read_json_bound(output / "caps-spent.json", evidence.get("caps-spent.json"), "spent")
    descendants = _read_json_bound(output / "worker-descendant-closure.json", evidence.get("worker-descendant-closure.json"), "worker closure")
    if controller_start.get("operation_id") != sealed["predecessor_operation_id"] or digest(controller_start) != admission.get("controller_start_receipt_sha256"):
        raise ValueError("sealed controller/admission identity")
    if admission.get("operation_id") != sealed["predecessor_operation_id"] or admission.get("operation_sha256") != sealed["predecessor_operation_manifest_sha256"] or admission.get("g_receipt_sha256") != final_g.get("admission_receipt_sha256"):
        raise ValueError("sealed admission identity")
    if final_g.get("operation_id") != sealed["predecessor_operation_id"] or not _pin(final_g.get("receipt_sha256")):
        raise ValueError("sealed final G identity")
    if worker.get("operation_id") != sealed["predecessor_operation_id"] or not isinstance(worker.get("worker"), dict) or not isinstance(worker.get("descendants"), list):
        raise ValueError("sealed worker identity")
    if caps_spent != {"schema_version": 1, "operation_id": sealed["predecessor_operation_id"], "caps": caps_spent.get("caps"), "spent": sealed["spent"]}:
        raise ValueError("sealed spent identity")
    if terminal.get("operation_id") != sealed["predecessor_operation_id"] or terminal.get("operation_manifest_sha256") != sealed["predecessor_operation_manifest_sha256"] or terminal.get("output_root_identity") != sealed["predecessor_output_root_identity"] or terminal.get("control_root_identity") != sealed["predecessor_control_root_identity"] or terminal.get("spent") != sealed["spent"]:
        raise ValueError("sealed predecessor identity")
    if closure.get("state") != "CLOSED" or closure.get("terminal_sha256") != sealed["terminal_receipt_sha256"] or closure.get("worker_descendant_closure_sha256") != evidence.get("worker-descendant-closure.json") or descendants.get("descendant_closure_verified") is not True:
        raise ValueError("sealed closure identity")
    expected_roles = plan.VERIFIER_ROLE_SOURCES[value["stage"]]["artifact_hashes"]
    rows = sealed.get("artifact_hashes")
    if not isinstance(rows, list) or [row.get("role") for row in rows] != expected_roles:
        raise ValueError("sealed artifact roles")
    if _artifact_set_sha(rows) != sealed.get("complete_artifact_set_sha256"):
        raise ValueError("sealed artifact set")
    semantic_evidence = {"events": events["events"], "opened": opened["opened"], "caps_spent": caps_spent}
    artifact_payloads = {}
    for row in rows:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["size"] or _file_sha256(path) != row["sha256"]:
            raise ValueError("sealed artifact drift")
        payload = json.loads(path.read_bytes())
        _validate_artifact_payload(row["role"], payload, sealed, semantic_evidence)
        artifact_payloads[row["role"]] = payload
    _validate_route_artifact_closure(artifact_payloads, sealed, semantic_evidence)
    return sealed


def _execute_local_aggregate(value, *args, **kwargs):
    validate_static_manifest(value)
    output = Path(value["exclusive_output_root"])
    output.mkdir(parents=True, exist_ok=False)
    typed_result = dispatch_science(value["command"], *args, **kwargs)
    artifact = output / "result.json"
    artifact_sha = write_immutable_json(artifact, typed_result)
    receipt = {"operation_id": value["operation_id"], "operation_manifest_sha256": digest(value), "artifact_path": str(artifact), "artifact_sha256": artifact_sha, "status": "completed"}
    receipt_path = output / "receipt.json"
    receipt_sha = write_immutable_json(receipt_path, receipt)
    return {
        "state": "CLOSED",
        "terminal_status": "completed",
        "typed_result": copy.deepcopy(typed_result),
        "operation_manifest_sha256": digest(value),
        "exclusive_output_root": str(output),
        "artifact_path": str(artifact),
        "artifact_sha256": artifact_sha,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
    }


def execute_synthetic_lifecycle(value, *, operation, opened_audit=None):
    validate_static_manifest(value)
    predecessor = _verify_predecessor(value) if value["stage"] in _VERIFIER_STAGES else None
    output = Path(value["exclusive_output_root"])
    control = Path(value["exclusive_control_root"])
    output.mkdir(parents=True, exist_ok=False)
    control.mkdir(parents=True, exist_ok=False)
    started_at = time.monotonic()
    runtime_events = []

    def record_event(kind, state, payload=None):
        row = {"sequence": len(runtime_events), "kind": kind, "state": state, "observed_at": time.time(), "operation_id": value["operation_id"]}
        if payload is not None:
            row["payload"] = copy.deepcopy(payload)
        runtime_events.append(row)

    def operation_sink(kind, payload):
        if not isinstance(kind, str) or not isinstance(payload, dict):
            raise ValueError("science operation event")
        event_payload = copy.deepcopy(payload)
        if kind == "checkpoint":
            entries = {}
            for role, source_sha in event_payload.pop("checkpoint_roles").items():
                entries[role] = _checkpoint_entry(value, role, source_sha, event_payload["epoch"], event_payload["step"])
            event_payload["checkpoints"] = entries
            event_payload["artifact_identity"] = "private_path_sha256_size"
        elif kind == "route_shard":
            records = event_payload.pop("records")
            event_payload.pop("query_ids", None)
            private = {"schema_version": 1, "operation_id": value["operation_id"], "shard": event_payload["shard"], "start": event_payload["range"][0], "end": event_payload["range"][1], "records": records}
            path, sha, size = _write_private_record(value, f"route-shard-{event_payload['shard']:06d}.json", private)
            event_payload.update(path=path, sha256=sha, size=size, count=len(records), artifact_identity="private_path_sha256_size")
        record_event(kind, "SUCCEEDED", event_payload)

    sink_records = []
    sink = lambda kind, record: sink_records.append({"kind": kind, "record": record})
    start = prepare_controller(value, current_process_identity(), sink=sink)
    record_event("controller_start", "AWAITING_ADMISSION")
    feed_root = control / "live-synthetic-feed"
    handle = None
    with synthetic_live_feed(feed_root, now=time.time()) as publisher:
        controller = LiveFeedController(publisher.descriptor_path, publisher.descriptor_sha256, control / "accepted-head.json")
        observed = controller.observe(now=time.time())
        admission = synthetic_admission(start, observed["receipt_sha256"], issued_at=time.time())
        admission["wall_cap"] = value["caps"]["wall_seconds"]
        admission["new_bytes_cap"] = value["caps"]["new_bytes"]
        admitted = admit_controller(start, admission, controller, now=time.time(), sink=sink)
        record_event("admission", "ADMITTED")
        try:
            handle = spawn_synthetic_worker_tree(control / "cooperative-stop.json")
            before = controller.observe(now=time.time())
            typed_result = operation(operation_sink)
            after = controller.observe(now=time.time())
            if before["action"] != "CONTINUE" or after["action"] != "CONTINUE":
                raise ValueError("feed stopped at stage")
            final_feed = controller.observe(now=time.time())
        finally:
            if handle is not None:
                record_event("stop", "STOP_REQUESTED")
                closure_state = stop_worker_tree(handle, control / "cooperative-stop.json", expected_identity=handle["worker_identity"], seal_deadline=value["caps"]["seal_seconds"], signal_grace=value["caps"]["signal_grace_seconds"])
    if handle is None or closure_state["state"] != "CLOSED":
        raise ValueError("worker closure")
    record_event("seal", "CLOSED")
    spent = {"wall_seconds": time.monotonic() - started_at, "new_bytes": 0}
    if spent["wall_seconds"] > value["caps"]["wall_seconds"]:
        raise ValueError("operation cap exceeded")
    operation_id = value["operation_id"]
    opened = copy.deepcopy(opened_audit or [])
    opened_map = {row["role"]: row["sha256"] for row in opened}
    events_value = {"schema_version": 1, "operation_id": operation_id, "events": runtime_events}
    evidence_values = {
        "controller-start.json": start,
        "admission.json": admitted,
        "g-observations.json": {"schema_version": 1, "operation_id": operation_id, "observations": [observed, before, after, final_feed]},
        "final-g-observation.json": {"schema_version": 1, "operation_id": operation_id, "admission_receipt_sha256": admitted["g_receipt_sha256"], **final_feed},
        "worker-birth-inventory.json": {"schema_version": 1, "operation_id": operation_id, "worker": handle["worker_identity"], "descendants": [handle["grandchild_identity"]]},
        "worker-descendant-closure.json": {"schema_version": 1, "operation_id": operation_id, **closure_state},
        "caps-spent.json": {"schema_version": 1, "operation_id": operation_id, "caps": copy.deepcopy(value["caps"]), "spent": copy.deepcopy(spent)},
        "stage-events.json": events_value,
        "opened-artifacts.json": {"schema_version": 1, "operation_id": operation_id, "opened": opened, "opened_roles": sorted(opened_map), "opened_sha256": {name: opened_map[name] for name in sorted(opened_map)}},
    }
    evidence_hashes = {name: write_immutable_json(output / name, payload) for name, payload in evidence_values.items()}
    artifacts = _write_artifact_set(value, typed_result, spent=spent, events=events_value["events"], opened=opened)
    verifier = value["stage"] in _VERIFIER_STAGES
    terminal_status = "completed+verified" if verifier else "completed+unverified"
    terminal_preimage = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_manifest_sha256": digest(value),
        "output_root_identity": digest(str(output.resolve())),
        "control_root_identity": digest(str(control.resolve())),
        "typed_result_sha256": digest(typed_result),
        "status": terminal_status,
        "spent": spent,
        "predecessor_operation_manifest_sha256": None if predecessor is None else predecessor["predecessor_operation_manifest_sha256"],
    }
    terminal_preimage_sha = write_immutable_json(output / "terminal-preimage.json", terminal_preimage)
    closure_preimage = {
        "schema_version": 1,
        "operation_id": operation_id,
        "operation_manifest_sha256": digest(value),
        "worker_descendant_closure_sha256": evidence_hashes["worker-descendant-closure.json"],
        "worker_closed": True,
        "state": "CLOSED",
    }
    closure_preimage_sha = write_immutable_json(output / "closure-preimage.json", closure_preimage)
    complete_evidence = {
        "controller-start": evidence_hashes["controller-start.json"],
        "admission": evidence_hashes["admission.json"],
        "g-observations": evidence_hashes["g-observations.json"],
        "final-g-observation": evidence_hashes["final-g-observation.json"],
        "worker-birth-inventory": evidence_hashes["worker-birth-inventory.json"],
        "worker-descendant-closure": evidence_hashes["worker-descendant-closure.json"],
        "caps-spent": evidence_hashes["caps-spent.json"],
        "stage-events": evidence_hashes["stage-events.json"],
        "opened-artifacts": evidence_hashes["opened-artifacts.json"],
        "artifacts": _artifact_set_sha(artifacts),
        "terminal-preimage": terminal_preimage_sha,
        "closure-preimage": closure_preimage_sha,
    }
    terminal = {**terminal_preimage, "evidence_sha256": evidence_hashes, "complete_evidence_sha256": complete_evidence}
    terminal_sha = write_immutable_json(output / "terminal.json", terminal)
    closure = {**closure_preimage, "terminal_sha256": terminal_sha, "complete_evidence_sha256": complete_evidence}
    closure_sha = write_immutable_json(output / "closure.json", closure)
    os.link(output / "closure.json", control / "closure.json")
    result = {
        "state": "CLOSED", "terminal_status": terminal_status, "typed_result": copy.deepcopy(typed_result),
        "operation_manifest_sha256": digest(value), "exclusive_output_root": str(output), "exclusive_control_root": str(control),
        "controller_identity": {key: start[key] for key in ("controller_pid", "controller_pgid", "controller_start_ticks")},
        "admission_sha256": evidence_hashes["admission.json"], "g_receipt_sha256": evidence_hashes["final-g-observation.json"],
        "worker_pgid": handle["worker_identity"]["pgid"], "predecessor_read_only": verifier, "spent": spent,
    }
    sealed = _sealed_for_stage(value, evidence_hashes, artifacts, terminal_sha, closure_sha, spent)
    if sealed is not None:
        sealed_path = output / "sealed-predecessor.json"
        sealed_sha = write_immutable_json(sealed_path, sealed)
        result.update(sealed=sealed, sealed_path=str(sealed_path), sealed_sha256=sealed_sha)
    if predecessor is not None:
        result["predecessor_operation_manifest_sha256"] = predecessor["predecessor_operation_manifest_sha256"]
        verifier_sealed = {
            "schema_version": 1, "operation_id": operation_id, "operation_manifest_sha256": digest(value),
            "predecessor_operation_manifest_sha256": predecessor["predecessor_operation_manifest_sha256"],
            "terminal_receipt_sha256": terminal_sha, "closure_receipt_sha256": closure_sha,
            "worker_descendant_closure_sha256": evidence_hashes["worker-descendant-closure.json"],
            "complete_evidence_sha256": complete_evidence, "state": "CLOSED",
        }
        verifier_path = output / "sealed-verifier.json"
        verifier_sha = write_immutable_json(verifier_path, verifier_sealed)
        result.update(verifier_sealed_path=str(verifier_path), verifier_sealed_sha256=verifier_sha)
    return result


def execute_typed_operation(value, *args, **kwargs):
    opened_audit = kwargs.pop("_opened_audit", None)
    command = value.get("command")
    if command == "prepare-controller":
        raise ValueError("prepare-controller has no science call")
    if command in _LOCAL_STAGES:
        return _execute_local_aggregate(value, *args, **kwargs)
    kwargs.pop("sink", None)

    def invoke(operation_sink):
        science_records = []
        science_sink = lambda kind, payload: science_records.append({"kind": kind, "payload": copy.deepcopy(payload)})
        call_kwargs = {**kwargs, "operation_sink": operation_sink}
        if command != "pilot-verify":
            call_kwargs["sink"] = science_sink
        return dispatch_science(command, *args, **call_kwargs)

    return execute_synthetic_lifecycle(value, operation=invoke, opened_audit=opened_audit)


def complete_verified(verifier_manifest, verifier_receipt_path, verifier_receipt_sha256):
    if verifier_manifest.get("stage") not in _VERIFIER_STAGES or not isinstance(verifier_receipt_path, (str, Path)) or not _pin(verifier_receipt_sha256):
        raise ValueError("verifier sealed receipt required")
    sealed = _read_json_bound(verifier_receipt_path, verifier_receipt_sha256, "verifier wrapper")
    expected = {"schema_version", "operation_id", "operation_manifest_sha256", "predecessor_operation_manifest_sha256", "terminal_receipt_sha256", "closure_receipt_sha256", "worker_descendant_closure_sha256", "complete_evidence_sha256", "state"}
    if set(sealed) != expected or sealed["schema_version"] != 1 or sealed["state"] != "CLOSED" or sealed["operation_id"] != verifier_manifest["operation_id"] or sealed["operation_manifest_sha256"] != digest(verifier_manifest):
        raise ValueError("verifier sealed identity")
    output = Path(verifier_receipt_path).parent
    terminal = _read_json_bound(output / "terminal.json", sealed["terminal_receipt_sha256"], "verifier terminal")
    closure = _read_json_bound(output / "closure.json", sealed["closure_receipt_sha256"], "verifier closure")
    descendants = _read_json_bound(output / "worker-descendant-closure.json", sealed["worker_descendant_closure_sha256"], "verifier worker closure")
    if terminal.get("complete_evidence_sha256") != sealed["complete_evidence_sha256"] or closure.get("complete_evidence_sha256") != sealed["complete_evidence_sha256"]:
        raise ValueError("verifier evidence map binding")
    _validate_disk_evidence(output, sealed["operation_id"], sealed["operation_manifest_sha256"], terminal, closure, [], verifier_manifest["stage"])
    predecessor = _verify_predecessor(verifier_manifest)
    if terminal.get("status") != "completed+verified" or terminal.get("operation_manifest_sha256") != sealed["operation_manifest_sha256"] or terminal.get("predecessor_operation_manifest_sha256") != sealed["predecessor_operation_manifest_sha256"]:
        raise ValueError("verifier terminal identity")
    if closure.get("state") != "CLOSED" or closure.get("terminal_sha256") != sealed["terminal_receipt_sha256"] or closure.get("worker_descendant_closure_sha256") != sealed["worker_descendant_closure_sha256"] or descendants.get("descendant_closure_verified") is not True:
        raise ValueError("verifier closure identity")
    if predecessor["predecessor_operation_manifest_sha256"] != sealed["predecessor_operation_manifest_sha256"]:
        raise ValueError("verifier predecessor cross-binding")
    return {"status": "completed+verified", "verifier_operation_manifest_sha256": sealed["operation_manifest_sha256"], "predecessor_operation_manifest_sha256": sealed["predecessor_operation_manifest_sha256"]}


def _load_typed_inputs(value):
    bindings = value["typed_inputs"]
    if set(bindings) == {"args", "kwargs"}:
        opened = []
        def load_legacy(binding, fallback_role):
            path = Path(binding["path"]);data = path.read_bytes();actual = hashlib.sha256(data).hexdigest()
            if actual != binding["sha256"]:raise ValueError("typed input bytes/SHA mismatch")
            opened.append({"role": fallback_role, "path": str(path.resolve()), "sha256": actual, "size": len(data), "schema_id": "typed-json-v1"})
            return json.loads(data)
        args = [load_legacy(row, f"typed-arg-{index}") for index, row in enumerate(bindings["args"])]
        kwargs = {name: load_legacy(row, f"typed-kwarg-{name}") for name, row in bindings["kwargs"].items()}
        return args, kwargs, opened

    contents = {};opened = []
    for binding in bindings["audit"]:
        path = Path(binding["path"]);data = path.read_bytes();actual = hashlib.sha256(data).hexdigest()
        if actual != binding["sha256"]:raise ValueError("typed input bytes/SHA mismatch")
        if binding["role"] in contents:raise ValueError("duplicate typed input role")
        contents[binding["role"]] = json.loads(data)
        opened.append({**binding, "path": str(path.resolve()), "size": len(data)})
    _validate_opened_role_bijection(value["roles"], opened)
    references = bindings["args"] + list(bindings["kwargs"].values())
    if any(reference["role"] not in contents for reference in references):raise ValueError("typed argument role not opened")
    args = [copy.deepcopy(contents[reference["role"]]) for reference in bindings["args"]]
    kwargs = {name: copy.deepcopy(contents[reference["role"]]) for name, reference in bindings["kwargs"].items()}
    return args, kwargs, opened


def _dispatch_operation(command, value, args, kwargs):
    if value["command"] != command:
        raise ValueError("dispatch command mismatch")
    kwargs = dict(kwargs)
    if command != "pilot-verify" and "sink" not in kwargs:
        kwargs["sink"] = lambda *_: None
    return execute_typed_operation(value, *args, **kwargs)


def dispatch_prepare_controller(value, args, kwargs):
    if args or kwargs:
        raise ValueError("prepare-controller inputs")
    output = Path(value["exclusive_output_root"])
    output.mkdir(parents=True, exist_ok=False)
    receipt = prepare_controller(value, current_process_identity(), sink=lambda *_: None)
    path = output / "controller-start.json"
    sha = write_immutable_json(path, receipt)
    return {"state": "AWAITING_ADMISSION", "artifact_path": str(path), "artifact_sha256": sha}


def dispatch_pilot(value, args, kwargs): return _dispatch_operation("pilot", value, args, kwargs)
def dispatch_pilot_verify(value, args, kwargs): return _dispatch_operation("pilot-verify", value, args, kwargs)
def dispatch_formal_train(value, args, kwargs): return _dispatch_operation("formal-train", value, args, kwargs)
def dispatch_formal_resume(value, args, kwargs): return _dispatch_operation("formal-resume", value, args, kwargs)
def dispatch_formal_verify(value, args, kwargs): return _dispatch_operation("formal-verify", value, args, kwargs)
def dispatch_select(value, args, kwargs): return _dispatch_operation("select", value, args, kwargs)
def dispatch_completion_verify(value, args, kwargs): return _dispatch_operation("completion-verify", value, args, kwargs)
def dispatch_summarize(value, args, kwargs): return _dispatch_operation("summarize", value, args, kwargs)
def dispatch_infer_route(value, args, kwargs): return _dispatch_operation("infer-route", value, args, kwargs)
def dispatch_infer_route_verify(value, args, kwargs): return _dispatch_operation("infer-route-verify", value, args, kwargs)


COMMAND_DISPATCH = {
    "prepare-controller": dispatch_prepare_controller,
    "pilot": dispatch_pilot,
    "pilot-verify": dispatch_pilot_verify,
    "formal-train": dispatch_formal_train,
    "formal-resume": dispatch_formal_resume,
    "formal-verify": dispatch_formal_verify,
    "select": dispatch_select,
    "completion-verify": dispatch_completion_verify,
    "summarize": dispatch_summarize,
    "infer-route": dispatch_infer_route,
    "infer-route-verify": dispatch_infer_route_verify,
}


def main(argv=None):
    try:
        arguments = build_parser().parse_args(argv)
        value = load_operation_manifest(arguments.operation_manifest, arguments.operation_manifest_sha256)
        validate_static_manifest(value)
        if arguments.command != value["command"]:
            raise ValueError("CLI command/manifest mismatch")
        args, kwargs, opened = _load_typed_inputs(value)
        kwargs["_opened_audit"] = opened
        result = COMMAND_DISPATCH[arguments.command](value, args, kwargs)
        print(json.dumps({"command": arguments.command, "dispatch": COMMAND_DISPATCH[arguments.command].__name__, **result}, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"operation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
