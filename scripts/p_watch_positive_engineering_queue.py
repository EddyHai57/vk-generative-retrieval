"""Finite metadata-only queue bridge for P preflight, CUDA preview and three train runs."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vkgr import p_watch_positive_ops as ops

MAX_BYTES = 2 * 1024 * 1024
ACTIONS = ("inventory", "dry-run", "publish", "status")
TRAIN_DEPLOYMENT_NAME = "vkgr-p-watch-positive-train-v2"
TRAIN_OPS_NAME = "vkgr-p-watch-positive-train-ops-v2"


def load(path):
    raw = Path(path).read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("bounded private queue request required")
    return json.loads(raw)


def queue_state(root, run_id):
    root = Path(root)
    state = {name: sorted(p.name for p in (root / "queue" / name).glob("*-" + run_id + ".json"))
             for name in ("pending", "claimed", "rejected")}
    state["reasons"] = {}
    for name in state["rejected"]:
        for p in ((root / "queue/rejected" / name).with_suffix(".reason"), root / "queue/rejected" / (name + ".reason")):
            if p.is_file() and not p.is_symlink():
                state["reasons"][p.name] = p.read_text()[:8192]
    return state


def inventory(root, ops_dir, job):
    rid = job["run_id"]
    run_dir = Path(root) / "runs" / rid
    terminal = run_dir / "terminal.json"
    value = {"run_id": rid, "queue": queue_state(root, rid),
             "run_exists": run_dir.exists(),
             "intent_exists": (Path(ops_dir) / ("intent-" + rid + ".json")).exists(),
             "job_sha256": ops.digest(job)}
    if terminal.is_file() and not terminal.is_symlink():
        value["terminal"] = json.loads(terminal.read_text())
    return value

def validate_namespaces(req):
    if (type(req) is not dict or type(req.get("deployment")) is not str or type(req.get("ops")) is not str
            or Path(req["deployment"]).name != TRAIN_DEPLOYMENT_NAME
            or Path(req["ops"]).name != TRAIN_OPS_NAME):
        raise ValueError("fresh training namespace required; statistics v2 namespaces are reserved")
    return True

def _validate_deployment_pin(req, path):
    pin_path = Path(path)
    if not pin_path.is_absolute() or pin_path.is_symlink() or not pin_path.is_file():
        raise ValueError("private pinned deployment manifest required")
    pin = json.loads(pin_path.read_text())
    if (type(pin) is not dict or set(pin) != {"schema", "base_commit", "target_commit", "files", "overrides"}
            or pin.get("schema") != "vkgr-pinned-v1" or pin.get("overrides") != {}
            or pin.get("files") != req["source_contract"].get("deployed_files")):
        raise ValueError("scheduler pinned manifest differs from exact original-plus-delta source map")
    head = subprocess.check_output(["git", "-C", req["deployment"], "rev-parse", "HEAD"], text=True).strip()
    if pin.get("base_commit") != head or pin.get("target_commit") != head:
        raise ValueError("scheduler pinned manifest does not bind the fixed deployed HEAD")
    return pin


def _common(req):
    required = {"action", "mode", "root", "ops", "deployment", "scheduler_code", "python",
                "stats_report", "stats_sha256", "approval", "approval_sha256", "inputs", "inputs_sha256",
                "controls", "source_contract"}
    optional = {"jobs", "stage", "stage_request", "stage_request_sha256", "pinned_manifest",
                "conservative_vram_mib", "vram_proof", "preflight_proof", "dry_jobs_sha256"}
    if type(req) is not dict or not required <= set(req) or set(req) - required - optional:
        raise ValueError("fixed P queue request fields required")
    if req["action"] not in ACTIONS or req["mode"] not in ("stage", "training"):
        raise ValueError("metadata queue action/mode invalid")
    for key in ("root", "ops", "deployment", "scheduler_code", "stats_report", "approval", "inputs"):
        value = req[key]
        if type(value) is not str or not Path(value).is_absolute() or ".." in Path(value).parts or Path(value).is_symlink():
            raise ValueError("absolute private non-symlink path required: " + key)
    ops.validate_python_executable(req["python"])
    validate_namespaces(req)
    if Path(req["ops"]).is_symlink() or Path(req["deployment"]).is_symlink():
        raise ValueError("fresh P training namespaces cannot be symlinks")
    if ops.sha(req["stats_report"]) != req["stats_sha256"] or ops.sha(req["inputs"]) != req["inputs_sha256"]:
        raise ValueError("formal stats or original input pin changed")
    controls = ops.read_control_manifests(req["controls"])
    contract = dict(req["source_contract"], accepted_controls=controls)
    source_identity = ops.validate_source_contract(contract)
    if req["source_contract"].get("training_files") != controls["training_files"]:
        raise ValueError("P deployment source ledger differs from original M1 source ledger")
    ops.validate_scheduler_source(Path(req["scheduler_code"]) / "ops")
    from vkgr.p_watch_positive_run import validate_stats_report, validate_training_approval
    report = validate_stats_report(req["stats_report"], req["stats_sha256"])
    if report.get("engineering") is not False or report.get("synthetic") is not False:
        raise ValueError("formal real train-only stats report required")
    for seed in ops.SEEDS:
        validate_training_approval(req["approval"], req["approval_sha256"], ops.RUN_IDS[seed], seed, req["stats_sha256"])
    return controls, source_identity, report


def _scheduler(req):
    ops.validate_scheduler_source(Path(req["scheduler_code"]) / "ops")
    sys.path.insert(0, str(Path(req["scheduler_code"]) / "ops"))
    from vkgr_ops import launcher, scheduler
    return launcher, scheduler


def _stage_job(req):
    stage = req.get("stage")
    if stage not in ops.STAGE_RUN_IDS:
        raise ValueError("fixed preflight/cuda-preview stage required")
    path, pin = req.get("stage_request"), req.get("stage_request_sha256")
    if type(path) is not str or not Path(path).is_absolute() or Path(path).is_symlink() or ops.sha(path) != pin:
        raise ValueError("pinned private stage request required")
    stage_req = load(path)
    if stage_req.get("stage") != stage or stage_req.get("stage_run_id") != ops.STAGE_RUN_IDS[stage]:
        raise ValueError("stage request/job fixed identity mismatch")
    for key, req_key in (("inputs", "inputs"), ("inputs_sha256", "inputs_sha256"),
            ("stats_report", "stats_report"), ("stats_sha256", "stats_sha256"),
            ("approval", "approval"), ("approval_sha256", "approval_sha256"),
            ("controls", "controls"), ("source_contract", "source_contract")):
        if stage_req.get(key) != req[req_key]:
            raise ValueError("stage worker request differs from sealed queue metadata: " + key)
    if stage == "cuda-preview" and stage_req.get("conservative_vram_mib") != req.get("conservative_vram_mib"):
        raise ValueError("CUDA stage request differs from conservative scheduler declaration")
    if stage == "cuda-preview":
        preflight = _read_successful_stage(req, req.get("preflight_proof"), "preflight", "preflight_ok")
        if (preflight.get("schema") != "p-watch-positive-engineering-preflight-v1"
                or preflight.get("engineering") is not False or preflight.get("synthetic") is not False
                or preflight.get("validation_metrics_read") is not False):
            raise ValueError("real engineering=False preflight required before CUDA preview")
    manifest = req.get("pinned_manifest")
    if type(manifest) is not str or not Path(manifest).is_absolute() or Path(manifest).is_symlink() or not Path(manifest).is_file():
        raise ValueError("private vkgr pinned manifest required")
    _validate_deployment_pin(req, manifest)
    if stage == "preflight":
        vram = 0
    else:
        upper = req.get("conservative_vram_mib")
        declaration = ops.initial_vram_declaration(upper)
        vram = declaration["declared_vram_mib"]
        if vram != upper:
            raise ValueError("initial conservative VRAM declaration must already be 256 MiB rounded")
    runtime_env = ["PYTHONPATH=" + str(Path(req["deployment"]) / "src") + os.pathsep + str(Path(req["deployment"]) / "scripts")]
    inputs = sorted(set([path, req["inputs"], req["stats_report"], req["approval"],
                         *[str(ref["path"]) for ref in req["controls"].values()],
                         *[str(ref["path"]) for ref in stage_req["configs"].values()]]))
    return {"schema": "vkgr-queue-job-v1", "run_id": ops.STAGE_RUN_IDS[stage], "kind": "ops",
            "cwd": req["deployment"], "config": None, "seed": None, "inputs": inputs,
            "pinned_manifest": manifest, "pinned_files": [], "env": runtime_env,
            "cpu_cores": 1, "ram_mib": 8192, "vram_mib": vram, "disk_gib": 1,
            "wall_hours": 4 if stage == "preflight" else 8, "stop_mode": "signal", "no_admission": False,
            "argv": [req["python"], "-B", "-m",
                     "vkgr.p_watch_positive_run" if stage == "preflight" else "vkgr.p_watch_positive_ops", stage,
                     "--request", path, "--request-sha", pin]}

def _read_successful_stage(req, proof, stage, expected_status):
    if type(proof) is not dict or set(proof) != {"path", "sha256", "run_id"}:
        raise ValueError(stage + " stage stdout reference required")
    run_id = ops.STAGE_RUN_IDS[stage]
    expected_stdout = Path(req["root"]) / "runs" / run_id / "logs/stdout.log"
    if proof["run_id"] != run_id or Path(proof["path"]) != expected_stdout or ops.sha(proof["path"]) != proof["sha256"]:
        raise ValueError(stage + " stdout path/SHA/run identity mismatch")
    run_dir = Path(req["root"]) / "runs" / run_id
    claim_files = sorted((Path(req["root"]) / "queue" / "claimed").glob("*-" + run_id + ".json"))
    rejected = sorted((Path(req["root"]) / "queue" / "rejected").glob("*-" + run_id + ".json"))
    if rejected or len(claim_files) != 1:
        raise ValueError(stage + " queue must be uniquely claimed and not rejected")
    claim = json.loads(claim_files[0].read_text())
    terminal_path = run_dir / "terminal.json"
    record_path = run_dir / "run.json"
    if terminal_path.is_symlink() or record_path.is_symlink() or not terminal_path.is_file() or not record_path.is_file():
        raise ValueError(stage + " scheduler terminal/run receipts missing")
    terminal = json.loads(terminal_path.read_text())
    exact_terminal_keys = {"schema", "run_id", "status", "exit_code", "signal", "started_at", "ended_at", "wall_seconds", "wall_cap_exceeded"}
    if (set(terminal) != exact_terminal_keys or terminal.get("schema") != "vkgr-terminal-v1"
            or terminal.get("run_id") != run_id or terminal.get("status") != "succeeded" or terminal.get("exit_code") != 0):
        raise ValueError(stage + " terminal is not accepted scheduler succeeded")
    run_record = json.loads(record_path.read_text())
    if (run_record.get("schema") != "vkgr-run-v1" or run_record.get("run_id") != run_id
            or run_record.get("kind") != "ops" or claim.get("run_id") != run_id
            or claim.get("kind") != "ops" or claim.get("argv") != run_record.get("argv")):
        raise ValueError(stage + " accepted scheduler claim/run binding mismatch")
    if claim.get("schema") != "vkgr-queue-job-v1" or (run_record.get("env") or {}).get("VKGR_RUN_ID") != run_id:
        raise ValueError(stage + " queue claim or launcher-injected ID schema mismatch")
    expected_vram = 0 if stage == "preflight" else ops.initial_vram_declaration(worker_req_vram := json.loads(Path(claim["argv"][claim["argv"].index("--request") + 1]).read_text()).get("conservative_vram_mib"))["declared_vram_mib"]
    expected_wall = 4 if stage == "preflight" else 8
    expected_resources = {"cpu_cores": 1, "ram_mib": 8192, "vram_mib": expected_vram,
                          "disk_gib": 1}
    recorded_resources = run_record.get("resources") or {}
    if any(recorded_resources.get(key) != value or claim.get(key) != value
           for key, value in expected_resources.items()):
        raise ValueError(stage + " job/run resource binding differs from the fixed envelope")
    if claim.get("wall_hours") != expected_wall or recorded_resources.get("wall_seconds") != expected_wall * 3600.0:
        raise ValueError(stage + " wall-clock resource binding differs from the fixed envelope")
    if run_record.get("stop_mode") != "signal" or claim.get("stop_mode") != "signal" or claim.get("no_admission") is not False:
        raise ValueError(stage + " stop/admission mode differs from the fixed queue contract")
    argv = run_record.get("argv")
    if type(argv) is not list or argv.count("--request") != 1 or argv.count("--request-sha") != 1:
        raise ValueError(stage + " run record lacks the fixed request pin")
    expected_module = "vkgr.p_watch_positive_run" if stage == "preflight" else "vkgr.p_watch_positive_ops"
    if argv[:5] != [req["python"], "-B", "-m", expected_module, stage]:
        raise ValueError(stage + " stage module binding differs from the fixed runner")
    request_path = argv[argv.index("--request") + 1]
    request_sha = argv[argv.index("--request-sha") + 1]
    if ops.sha(request_path) != request_sha:
        raise ValueError(stage + " worker request bytes differ from scheduler argv pin")
    worker_req = json.loads(Path(request_path).read_text())
    if worker_req.get("stage") != stage or worker_req.get("stage_run_id") != run_id:
        raise ValueError(stage + " request stage/run identity mismatch")
    for key in ("inputs", "inputs_sha256", "stats_report", "stats_sha256", "approval", "approval_sha256", "controls", "source_contract"):
        if worker_req.get(key) != req.get(key):
            raise ValueError(stage + " request differs from sealed training request: " + key)
    input_map = {item.get("path"): item.get("sha256") for item in run_record.get("inputs", []) if type(item) is dict}
    claim_input_paths = {str(Path(value).resolve()) for value in claim.get("inputs", [])}
    if claim_input_paths != set(input_map):
        raise ValueError(stage + " scheduler claim/run input path sets differ")
    expected_inputs = {request_path, worker_req["inputs"], worker_req["stats_report"], worker_req["approval"],
        *[ref["path"] for ref in worker_req["controls"].values()],
        *[ref["path"] for ref in worker_req["configs"].values()]}
    if any(input_map.get(str(Path(path).resolve())) != ops.sha(path) for path in expected_inputs):
        raise ValueError(stage + " scheduler input map omits or misbinds a stage input")
    expected_configs = {str(job["seed"]): {"path": job["config"], "sha256": job["config_sha256"]}
                        for job in req.get("jobs", [])}
    if expected_configs and worker_req.get("configs") != expected_configs:
        raise ValueError(stage + " seed configs differ from fixed training queue")
    raw_stdout = Path(proof["path"])
    if raw_stdout.is_symlink() or not raw_stdout.is_file() or raw_stdout.stat().st_size > MAX_BYTES:
        raise ValueError(stage + " bounded nonsymlink original stdout required")
    found = []
    for line in Path(proof["path"]).read_text().splitlines():
        try: row = json.loads(line)
        except ValueError: continue
        if type(row) is dict and row.get("status") == expected_status:
            found.append(row)
    if len(found) != 1 or found[0].get("stage_run_id") != run_id:
        raise ValueError(stage + " stdout lacks one direct stage-bound success receipt")
    if stage == "cuda-preview":
        admission_vram = found[0].get("admission_vram_mib")
        resources = run_record.get("resources") or {}
        requested_upper = ops.initial_vram_declaration(worker_req.get("conservative_vram_mib"))["declared_vram_mib"]
        samples = found[0].get("memory_measurement", {}).get("samples", [])
        expected_stages = ["model-initialized", "gradient-step", "catalog-decode-top100", "checkpoint-reload-decode-top100"]
        if (found[0].get("admission_not_measured") is not True
                or found[0].get("admission_peak_method") != "conservative_upper_bound"
                or admission_vram != requested_upper
                or [sample.get("stage") for sample in samples] != expected_stages
                or found[0].get("gradient_step", 0) <= 0 or found[0].get("decode_count") != 100
                or found[0].get("reload_decode_equal") is not True
                or claim.get("vram_mib") != admission_vram or resources.get("vram_mib") != admission_vram):
            raise ValueError("first CUDA admission must bind the conservative not_measured VRAM upper bound")
    return found[0]


def _training_jobs(req):
    jobs = req.get("jobs")
    if type(jobs) is not list or len(jobs) != 3:
        raise ValueError("exactly three registered training jobs required")
    runs = {job.get("seed"): job.get("run_id") for job in jobs if type(job) is dict}
    if runs != ops.RUN_IDS:
        raise ValueError("exact three P training run identities required")
    preflight = _read_successful_stage(req, req.get("preflight_proof"), "preflight", "preflight_ok")
    if preflight.get("schema") != "p-watch-positive-engineering-preflight-v1" or preflight.get("engineering") is not False or preflight.get("synthetic") is not False or preflight.get("validation_metrics_read") is not False:
        raise ValueError("real engineering=False preflight receipt required before CUDA/training")
    preview = _read_successful_stage(req, req.get("vram_proof"), "cuda-preview", "cuda_engineering_verified")
    if (preview.get("schema") != "p-watch-positive-cuda-preview-v1" or preview.get("not_measured") is not False
            or preview.get("admission_not_measured") is not True or preview.get("validation_metrics_read") is not False):
        raise ValueError("CUDA preview is not a completed context-inclusive measurement")
    if (preflight.get("watch_aux_exclusion") != preview.get("watch_aux_exclusion")
            or preflight.get("stats_stdout_sha256") != req["stats_sha256"]
            or preview.get("pool_binding", {}).get("N_aux") != preflight.get("N_aux")
            or any(preview.get("pool_binding", {}).get(key) != value
                   for key, value in (preflight.get("watch_aux_exclusion") or {}).items())
            or preview.get("pool_binding", {}).get("auxiliary_pool_sha256")
                != (preflight.get("watch_aux_exclusion") or {}).get("filtered_pool_sha256")):
        raise ValueError("preflight/CUDA preview filtered auxiliary identity mismatch")
    memory = preview.get("memory_measurement")
    if type(memory) is not dict or type(memory.get("pid")) is not int or memory["pid"] <= 0:
        raise ValueError("CUDA preview own-PID sample proof missing")
    owner_path = Path(req["root"]) / "runs" / ops.STAGE_RUN_IDS["cuda-preview"] / "owner.json"
    if owner_path.is_symlink() or not owner_path.is_file():
        raise ValueError("CUDA preview scheduler owner record missing")
    owner = json.loads(owner_path.read_text())
    if type(owner.get("job")) is not dict or owner["job"].get("pid") != memory["pid"]:
        raise ValueError("CUDA measurement PID differs from the scheduler-owned engineering worker")
    measured = ops.measured_vram(memory, memory["pid"])
    if measured != preview.get("measured_mib"):
        raise ValueError("CUDA preview raw memory samples do not reproduce reported peak")
    expected_vram = ops.declared_vram_mib(preview.get("measured_mib"))
    if expected_vram != preview.get("declared_vram_mib"):
        raise ValueError("CUDA preview declaration does not match context-inclusive peak")
    launcher, _ = _scheduler(req)
    from vkgr.p_watch_positive_run import validate_training_approval
    for job in jobs:
        seed, rid = job["seed"], job["run_id"]
        validate_training_approval(req["approval"], req["approval_sha256"], rid, seed, req["stats_sha256"])
        if job.get("kind") != "real" or job.get("schema") != "vkgr-queue-job-v1" or job.get("no_admission") is not False:
            raise ValueError("scheduler kind=real train job and admission required")
        if job.get("cwd") != req["deployment"] or job.get("config") is None or job.get("vram_mib") != expected_vram:
            raise ValueError("train job deployment/config/measured-VRAM mismatch")
        _validate_deployment_pin(req, job.get("pinned_manifest"))
        if (job.get("cpu_cores") != 1 or job.get("ram_mib") != 8192 or job.get("disk_gib") != 1
                or type(job.get("wall_hours")) not in (int, float) or isinstance(job.get("wall_hours"), bool)
                or not math.isfinite(job["wall_hours"]) or not 0 < job["wall_hours"] <= 48):
            raise ValueError("P train scheduler resource/deadline envelope differs from fixed single-core job")
        if job.get("env") and any(x.partition("=")[0] in {"VKGR_RUN_ID", "VKGR_PROGRESS_PATH"} for x in job["env"]):
            raise ValueError("VKGR runtime identity and progress path are launcher-injected")
        if job.get("argv", [])[:4] != [req["python"], "-B", "-m", "vkgr.p_watch_positive_run"]:
            raise ValueError("fixed P runner train command required")
        if "train" not in job["argv"]:
            raise ValueError("train job must invoke runner train")
        config_path = job["config"]
        if not Path(config_path).is_file() or job.get("config_sha256") != ops.sha(config_path):
            raise ValueError("seed config SHA mismatch")
        config = json.loads(Path(config_path).read_text())
        if config.get("seed") != seed or config.get("schema") != "p-watch-positive-run-v1":
            raise ValueError("frozen per-seed config identity mismatch")
        from vkgr.p_watch_positive_run import validate_config
        validate_config(config, False)
        expected_output = str(Path(req["root"]) / "runs" / rid / "out")
        for flag, value in (("--config", config_path), ("--config-sha", job["config_sha256"]),
                            ("--output", expected_output)):
            argv = job["argv"]
            if argv.count(flag) != 1 or argv[argv.index(flag)+1] != value:
                raise ValueError("train command config/output binding mismatch: " + flag)
        required_inputs = {str(Path(path).resolve()) for path in
            (req["inputs"], req["stats_report"], req["approval"],
             *[ref["path"] for ref in req["controls"].values()])}
        actual_inputs = {str(Path(path).resolve()) for path in job.get("inputs", [])}
        if not required_inputs <= actual_inputs:
            raise ValueError("train job input list omits sealed source/stats/approval/control files")
        for path in job.get("inputs", []):
            item = Path(path)
            if not item.is_absolute() or ".." in item.parts or item.is_symlink() or not item.is_file():
                raise ValueError("train input must be a real absolute nonsymlink file")
        for flag, value in (("--inputs", req["inputs"]), ("--inputs-sha", req["inputs_sha256"]),
            ("--stats-report", req["stats_report"]), ("--stats-sha", req["stats_sha256"]),
            ("--approval", req["approval"]), ("--approval-sha", req["approval_sha256"])):
            argv = job["argv"]
            if argv.count(flag) != 1 or argv[argv.index(flag)+1] != value:
                raise ValueError("train command pin mismatch: " + flag)
        launcher._validate_static(job)
    return jobs


def validate_request(req):
    controls, source, report = _common(req)
    jobs = [_stage_job(req)] if req["mode"] == "stage" else _training_jobs(req)
    return req, jobs, controls, source, report


def _fresh(root, ops_dir, jobs):
    rows = [inventory(root, ops_dir, job) for job in jobs]
    if any(row["run_exists"] or row["intent_exists"] or any(row["queue"][s] for s in ("pending", "claimed", "rejected")) for row in rows):
        raise FileExistsError("fixed run already exists; no retry, replacement or overwrite")
    return rows


def execute(req):
    req, jobs, _, _, _ = validate_request(req)
    if req["action"] in ("inventory", "status"):
        return {"status": req["action"], "jobs": [inventory(req["root"], req["ops"], job) for job in jobs]}
    rows = _fresh(req["root"], req["ops"], jobs)
    launcher, scheduler = _scheduler(req)
    prepared = [launcher.prepare(job) for job in jobs]
    sizes = [len(ops.canonical({"spec": job, "prepared": meta})) for job, meta in zip(jobs, prepared)]
    if any(size >= 48 * 1024 - 2048 for size in sizes):
        raise ValueError("queue launch envelope exceeds fixed size limit")
    job_sha = ops.digest(jobs)
    if req["action"] == "dry-run":
        return {"status": "dry_ok", "jobs_sha256": job_sha, "launch_bytes": sizes}
    if req.get("dry_jobs_sha256") != job_sha:
        raise ValueError("publish requires matching sealed dry-run jobs digest")
    lock = Path(req["ops"]) / ".p-watch-positive-publish.lock"
    published = []
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        _fresh(req["root"], req["ops"], jobs)
        for job in jobs:
            intent = Path(req["ops"]) / ("intent-" + job["run_id"] + ".json")
            with intent.open("x") as stream:
                json.dump({"job_sha256": ops.digest(job), "mode": req["mode"],
                    "written_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, stream)
                stream.flush(); os.fsync(stream.fileno())
            published.append(str(scheduler.submit(req["root"], job)))
    return {"status": "written_unconfirmed", "queue_paths": published,
            "message": "written is not claimed; re-read queue status and rejection reasons"}


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--request", required=True); parser.add_argument("--request-sha", required=True)
    args = parser.parse_args(argv)
    if ops.sha(args.request) != args.request_sha:
        raise ValueError("private queue request SHA mismatch")
    req = load(args.request)
    if req.get("action") != args.action:
        raise ValueError("CLI/request action mismatch")
    print(json.dumps(execute(req), sort_keys=True, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
