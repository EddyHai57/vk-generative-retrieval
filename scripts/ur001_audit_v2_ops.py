#!/usr/bin/env python3
"""A-only v2 transport adapter around the exact pinned engineering helper."""
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
RUNS = {
    "ur001-metadata-counts-v2": ("analysis/ur001_audit_v2.py", "metadata-counts"),
    "ur001-audit-preflight-v2": ("analysis/ur001_audit_v2.py", "audit-preflight"),
    "ur001-audit-v2": ("analysis/ur001_audit_v2.py", "audit"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_helper():
    if OLD_HELPER.is_symlink() or _sha256(OLD_HELPER) != OLD_HELPER_SHA256:
        raise ValueError("ALERT pinned engineering helper SHA differs")
    spec = importlib.util.spec_from_file_location("_ur001_engineering_private_v2", OLD_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.RUNS = dict(RUNS)
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
            raise ValueError("run_id is outside the three fixed A v2 stages")
        helper.validate(req)
        result = {"metadata": helper.metadata, "dry": helper.dry,
                  "submit": helper.submit, "status": helper.status}[action](req)
    else:
        raise ValueError("A v2 transport accepts deploy/inventory/metadata/dry/submit/status only")
    result["adapter_sha256"] = _sha256(Path(__file__))
    result["pinned_helper_sha256"] = OLD_HELPER_SHA256
    result["read_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
