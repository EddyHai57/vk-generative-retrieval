#!/usr/bin/env python3
"""Bounded scheduler adapter for the five authorized week 26 ops stages."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).absolute().parents[1]
OLD_HELPER = ROOT / "scripts/ur001_engineering_ops.py"
OLD_HELPER_SHA256 = "21f0ee0eb12039b1cc309971f1aa88bee8cba95442c8d4deea1034d8fcff3d40"
WORKER = "analysis/ur001_week26_bounded.py"
DEFAULT_ROOTS = {
    "a_root": "/workspace/vkgr-ur001-data-20261020-v1",
    "raw_root": "/workspace/vkgr-ur001-week26-raw-20261021-v1",
    "membership_root": "/workspace/vkgr-ur001-week26-membership-20261021-v1",
    "holdout_root": "/workspace/vkgr-ur001-week26-holdout-20261021-v1",
}
RUNS = {
    "ur001-week26-source-preflight-v1": (WORKER, "source-preflight"),
    "ur001-week26-users-v1": (WORKER, "prepare-users"),
    "ur001-week26-download-v1": (WORKER, "download"),
    "ur001-week26-filter-preflight-v1": (WORKER, "filter-preflight"),
    "ur001-week26-filter-v1": (WORKER, "filter"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_roots(config: Path, stage: str) -> dict:
    if config.is_symlink() or not config.is_file() or config.stat().st_size > 64 * 1024:
        raise ValueError("private week 26 request is absent, unsafe or oversized")
    request = json.loads(config.read_text(encoding="utf-8"))
    expected_fields = {"schema", "roots"}
    if stage in ("filter-preflight", "filter"):
        expected_fields.add("expected_users_sha256")
    if type(request) is not dict or set(request) != expected_fields:
        raise ValueError("week 26 private request has unexpected fields")
    if request.get("schema") != "ur001-week26-request-v1":
        raise ValueError("week 26 private request schema differs")
    roots = request.get("roots")
    if (type(roots) is not dict or set(roots) != set(DEFAULT_ROOTS)
            or any(type(roots.get(role)) is not str or roots[role] != value
                   for role, value in DEFAULT_ROOTS.items())):
        raise ValueError("week 26 private request must pin all four fixed roots")
    if stage in ("filter-preflight", "filter"):
        digest = request["expected_users_sha256"]
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("filter-stage private request must pin the prepare-users SHA-256")
    return request


def _expected_command(job, deployment, config, config_sha, *, root=None, ops_root=None, require_script=True):
    run_id = job.get("run_id")
    if run_id not in RUNS:
        raise ValueError("run_id is outside the five fixed week 26 ops stages")
    if job.get("kind") != "ops":
        raise ValueError("week 26 workflow accepts ops jobs only")
    rel_script, stage = RUNS[run_id]
    if stage == "filter" and (type(job.get("ram_mib")) is not int or job["ram_mib"] < 20_480
                              or type(job.get("disk_gib")) is not int or job["disk_gib"] < 37):
        raise ValueError("filter job must reserve at least 20 GiB RAM and 37 GiB disk")
    script = Path(deployment) / rel_script
    if require_script and (not script.is_file() or script.is_symlink()):
        raise ValueError("fixed week 26 worker is missing or unsafe")
    config = Path(config)
    if ops_root is None or not config.is_relative_to(Path(ops_root)):
        raise ValueError("private week 26 request must live under OPS")
    private_request = _request_roots(config, stage)
    roots = private_request["roots"]
    expected_inputs = [str(script), str(config)]
    if type(job.get("inputs")) is not list or job["inputs"] != expected_inputs:
        # This comparison happens before helper.validate opens or stats any supplied job input.
        raise ValueError("job.inputs include an unregistered input or a holdout role")
    expected = [sys.executable, "-I", "-B", str(script), stage, "--request", str(config)]
    binding = {"kind": "week26-ops", "stage": stage, "run_id": run_id,
               "roots": roots, "source_object": "interactions/test/week_26.parquet",
               "source_revision": "084a18386e149a454dd6dbe0123478e75c6b09ce",
               "source_sha256": "90c44ae75084df74b1c0251cd87daa6aa45d66377a77d1022027486c9a8b6993"}
    return rel_script, stage, expected, expected_inputs, binding


def _load_helper():
    if OLD_HELPER.is_symlink() or not OLD_HELPER.is_file() or _sha256(OLD_HELPER) != OLD_HELPER_SHA256:
        raise ValueError("ALERT pinned engineering helper SHA differs")
    spec = importlib.util.spec_from_file_location("_ur001_engineering_private_week26", OLD_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.RUNS = dict(RUNS)
    helper._expected_command = _expected_command
    return helper


def main() -> int:
    helper = _load_helper()
    raw = sys.stdin.buffer.read(helper.MAX_REQUEST_BYTES + 1)
    if len(raw) > helper.MAX_REQUEST_BYTES:
        raise ValueError("request exceeds bounded stdin size")
    req = json.loads(raw)
    if type(req) is not dict:
        raise ValueError("request must be a JSON object")
    action = req.get("action")
    if action == "deploy":
        result = helper._deployment(req)
    elif action == "inventory":
        result = helper.inventory(req)
    elif action in ("metadata", "dry", "submit", "status"):
        if req.get("job", {}).get("run_id") not in RUNS:
            raise ValueError("run_id is outside the five fixed week 26 ops stages")
        helper.validate(req)
        if action == "status":
            # The legacy helper skips argv/input equality on status; retain the same fixed
            # stage and asset-role gate while reading receipts.
            helper._fixed_request_binding(req, req["job"])
        result = {"metadata": helper.metadata, "dry": helper.dry,
                  "submit": helper.submit, "status": helper.status}[action](req)
    else:
        raise ValueError("week 26 adapter accepts deploy/inventory/metadata/dry/submit/status only")
    result["adapter_sha256"] = _sha256(Path(__file__))
    result["pinned_helper_sha256"] = OLD_HELPER_SHA256
    result["read_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
