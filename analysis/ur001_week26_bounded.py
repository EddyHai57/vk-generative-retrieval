#!/usr/bin/env python3
"""Bounded download and mechanical filtering of the authorized whole week 26 file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DATASET = "deepvk/VK-LSVD"
REVISION = "084a18386e149a454dd6dbe0123478e75c6b09ce"
OBJECT = "interactions/test/week_26.parquet"
OBJECT_SIZE = 14_300_986_995
OBJECT_SHA256 = "90c44ae75084df74b1c0251cd87daa6aa45d66377a77d1022027486c9a8b6993"
EXPECTED_USERS = 99_998
EXPECTED_ROWS = 14_884_261
EXPECTED_UNION_ITEMS = 12_467_302
MAX_GAP = 5_000
MAX_BATCH_ROWS = 65_536
MAX_RSS_BYTES = 16 * 1024**3
SOURCE_INVENTORY_SHA256 = "19cd71f02aff4fce93349907dbc5fcdbf006afc679ee899d9c98965123270fa0"
OLD_WORKER_SHA256 = "86e3ea98142b762c844d0deb66b2cf3d22f239df591e31f8f5d5f7d57263f94b"
DEFAULT_ROOTS = {
    "a_root": "/workspace/vkgr-ur001-data-20261020-v1",
    "raw_root": "/workspace/vkgr-ur001-week26-raw-20261021-v1",
    "membership_root": "/workspace/vkgr-ur001-week26-membership-20261021-v1",
    "holdout_root": "/workspace/vkgr-ur001-week26-holdout-20261021-v1",
}
FEEDBACK = ("like", "dislike", "share", "bookmark", "click_on_author", "open_comments")
EXPECTED_SCHEMA = {
    "user_id": "uint32", "item_id": "uint32", "place": "uint8", "platform": "uint8",
    "agent": "uint8", "timespent": "uint8", **{name: "bool" for name in FEEDBACK},
}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _worker_sha256() -> str:
    return sha256_file(Path(__file__))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_absolute(path: Path, *, must_exist: bool = True) -> Path:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("ALERT path must be absolute and have no traversal components")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("ALERT symlink path component is forbidden")
    if must_exist and not path.exists():
        raise ValueError("ALERT path is missing")
    return path


def _secure_file(root: Path, relative: str) -> Path:
    root = _safe_absolute(Path(root))
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or not rel.parts:
        raise ValueError("ALERT unsafe relative path")
    path = root
    for part in rel.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("ALERT symlink input is forbidden")
    if not path.is_file():
        raise ValueError("ALERT authorized input file is missing")
    resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError("ALERT input escapes its authorized root")
    return resolved


def _new_root(path: Path, forbidden_roots: tuple[Path, ...]) -> Path:
    path = _safe_absolute(Path(path), must_exist=False)
    if path.exists() or path.is_symlink():
        raise FileExistsError("destination root already exists; partial roots are retained")
    resolved = path.resolve(strict=False)
    for other in forbidden_roots:
        other = _safe_absolute(Path(other), must_exist=False)
        if resolved == other or resolved in other.parents or other in resolved.parents:
            raise ValueError("destination roots overlap")
    parent = _safe_absolute(path.parent)
    if resolved.parent != parent.resolve(strict=True):
        raise ValueError("destination root parent resolution changed")
    return path


def _source_inventory(root: Path) -> tuple[dict, dict[str, dict]]:
    import importlib.util

    source_root = Path(__file__).resolve().parents[1]
    inventory_path = source_root / "configs/ur001-preparation/hf-files.json"
    old_worker = source_root / "analysis/ur001_prepare.py"
    if sha256_file(inventory_path) != SOURCE_INVENTORY_SHA256:
        raise ValueError("A inventory SHA differs from the frozen identity")
    if sha256_file(old_worker) != OLD_WORKER_SHA256:
        raise ValueError("pinned A worker SHA differs from the frozen identity")
    raw_inventory = inventory_path.read_bytes()
    inventory = json.loads(raw_inventory)
    if inventory.get("dataset") != DATASET or inventory.get("revision") != REVISION:
        raise ValueError("A inventory source identity differs")
    records = inventory.get("files")
    if type(records) is not list or len(records) != 29:
        raise ValueError("A inventory must contain the exact 29-file source list")
    expected = {record["path"]: record for record in records}
    interaction_paths = [f"subsamples/ur0.01/train/week_{week:02d}.parquet" for week in range(25)]
    interaction_paths.append("subsamples/ur0.01/validation/week_25.parquet")
    if any(path not in expected for path in interaction_paths):
        raise ValueError("A inventory lacks an authorized interaction window file")
    download_path = _secure_file(root, "download-manifest.json")
    manifest = json.loads(download_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("task_id") != "ur001-preparation"
            or manifest.get("status") != "downloaded" or manifest.get("dataset") != DATASET
            or manifest.get("revision") != REVISION
            or manifest.get("script_sha256") != OLD_WORKER_SHA256
            or manifest.get("source_manifest_sha256") != SOURCE_INVENTORY_SHA256):
        raise ValueError("A download manifest does not match the fixed old source identity")
    completed = manifest.get("files")
    if type(completed) is not list or len(completed) != 29:
        raise ValueError("A download manifest must bind all 29 fixed files")
    for record in completed:
        if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
            raise ValueError("A download manifest file record is malformed")
        source = expected.get(record["path"])
        if not source or record["size"] != source["size"] or record["sha256"] != source["sha256"]:
            raise ValueError("A download manifest file identity differs from the frozen inventory")
    if {record["path"] for record in completed} != set(expected):
        raise ValueError("A download manifest does not contain the exact 29-file set")
    if manifest.get("total_bytes") != sum(record["size"] for record in records):
        raise ValueError("A download manifest total size differs from the fixed inventory")
    return inventory, expected


def _parquet_schema(path: Path) -> dict[str, str]:
    import pyarrow.parquet as pq
    return {field.name: str(field.type) for field in pq.ParquetFile(path).schema_arrow}


def _interaction_paths(root: Path, records: dict[str, dict]) -> list[tuple[str, Path, dict]]:
    paths = [f"subsamples/ur0.01/train/week_{week:02d}.parquet" for week in range(25)]
    paths.append("subsamples/ur0.01/validation/week_25.parquet")
    return [(rel, _secure_file(root, rel), records[rel]) for rel in paths]


def _check_a_files(root: Path, records: dict[str, dict], *, read_schema: bool = True) -> list[dict]:
    checked = []
    for rel, path, record in _interaction_paths(root, records):
        if path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"A interaction file identity mismatch: {rel}")
        if read_schema and _parquet_schema(path) != EXPECTED_SCHEMA:
            raise ValueError(f"A interaction schema mismatch: {rel}")
        checked.append({"path": rel, "size": record["size"], "sha256": record["sha256"]})
    return checked


def _url_for_object() -> str:
    return (f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
            f"{quote(OBJECT, safe='/')}?download=true")


def _hf_tree_metadata(opener=urlopen) -> dict:
    base = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/interactions/test"
    url = base + "?" + urlencode({"recursive": "false", "expand": "true"})
    request = Request(url, headers={"User-Agent": "vkgr-ur001-week26-bounded/1"})
    with opener(request, timeout=30) as response:
        entries = json.loads(response.read().decode("utf-8"))
    if type(entries) is not list:
        raise ValueError("HF tree response is not a file list")
    matches = []
    for entry in entries:
        if type(entry) is not dict:
            continue
        path = entry.get("path")
        if path == "week_26.parquet":
            path = OBJECT
        if path == OBJECT:
            lfs = entry.get("lfs") or {}
            oid = lfs.get("oid") if type(lfs) is dict else None
            if type(oid) is str:
                oid = oid.removeprefix("sha256:")
            if entry.get("size") != OBJECT_SIZE or oid != OBJECT_SHA256:
                raise ValueError("HF tree size or LFS SHA differs from the frozen object")
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError("HF tree must contain exactly one fixed week 26 object")
    return {"path": OBJECT, "size": OBJECT_SIZE, "lfs_oid_sha256": OBJECT_SHA256}


def _head_object(opener=urlopen) -> dict:
    request = Request(_url_for_object(), method="HEAD",
                      headers={"User-Agent": "vkgr-ur001-week26-bounded/1"})
    with opener(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        etag = response.headers.get("ETag", "").strip('"')
    if length is not None and int(length) != OBJECT_SIZE:
        raise ValueError("HF HEAD content length differs from the frozen object")
    return {"content_length": int(length) if length is not None else None, "etag": etag or None}


def _check_disk_budget(paths: tuple[Path, ...], required_bytes: int | None = None) -> dict:
    required = 2 * OBJECT_SIZE + 8 * 1024**3 if required_bytes is None else required_bytes
    checked = {}
    for parent in paths:
        usage = shutil.disk_usage(parent)
        if usage.free / usage.total < 0.20 or usage.free - required < usage.total * 0.20:
            raise ValueError("filesystem cannot fit raw, filter and scratch budget while retaining 20% free")
        checked[str(parent)] = {"free_bytes": usage.free, "required_bytes": required,
                                "free_after_budget_bytes": usage.free - required}
    return checked


def _check_rss_limit() -> None:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    if peak_bytes > MAX_RSS_BYTES:
        raise MemoryError("worker peak RSS exceeded the fixed 16 GiB ceiling")


def source_preflight(request: dict, opener=None) -> dict:
    opener = opener or urlopen
    roots = _validate_request(request, "source-preflight")
    _new_root(Path(roots["raw_root"]), (Path(roots["a_root"]), Path(roots["membership_root"]), Path(roots["holdout_root"])))
    _new_root(Path(roots["membership_root"]), (Path(roots["a_root"]), Path(roots["raw_root"]), Path(roots["holdout_root"])))
    _new_root(Path(roots["holdout_root"]), (Path(roots["a_root"]), Path(roots["raw_root"]), Path(roots["membership_root"])))
    inventory, records = _source_inventory(Path(roots["a_root"]))
    a_files = _check_a_files(Path(roots["a_root"]), records)
    a_manifest_sha = sha256_file(Path(roots["a_root"]) / "download-manifest.json")
    tree = _hf_tree_metadata(opener)
    head = _head_object(opener)
    disk = _check_disk_budget((Path(roots["raw_root"]).parent, Path(roots["membership_root"]).parent,
                               Path(roots["holdout_root"]).parent))
    return {"schema": "ur001-week26-source-preflight-v1", "status": "ready",
            "engineering": False, "test_payload_read": False, "filtered_counts_generated": False,
            "dataset": DATASET, "revision": REVISION, "source_object": OBJECT,
            "source_size": OBJECT_SIZE, "source_sha256": OBJECT_SHA256, "a_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "roots": roots, "a_download_manifest_sha256": a_manifest_sha,
            "a_download_manifest_identity": "verified-29-files", "a_interaction_files": a_files,
            "hf_tree": tree, "hf_head": head, "disk_budget": disk}


def _validate_roots(request: dict) -> dict[str, str]:
    if type(request) is not dict or request.get("schema") != "ur001-week26-request-v1":
        raise ValueError("request schema must be ur001-week26-request-v1")
    roots = request.get("roots")
    if type(roots) is not dict or set(roots) != set(DEFAULT_ROOTS):
        raise ValueError("request roots must have exactly the four pinned roles")
    clean = {}
    for role in DEFAULT_ROOTS:
        value = roots[role]
        if type(value) is not str:
            raise ValueError(f"request {role} must be an absolute path string")
        _safe_absolute(Path(value), must_exist=(role == "a_root"))
        clean[role] = value
    resolved = {role: Path(value).resolve(strict=False) for role, value in clean.items()}
    for role, path in resolved.items():
        for other_role, other in resolved.items():
            if role >= other_role:
                continue
            if path == other or path in other.parents or other in path.parents:
                raise ValueError("request roots overlap")
    return clean


def _validate_request(request: dict, stage: str) -> dict[str, str]:
    expected = {"schema", "roots"}
    filter_stage = stage in ("filter-preflight", "filter")
    if filter_stage:
        expected.add("expected_users_sha256")
    if type(request) is not dict or set(request) != expected:
        raise ValueError("request fields do not match the fixed stage schema")
    if filter_stage:
        value = request["expected_users_sha256"]
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("filter request must pin the prepared users.json SHA-256")
    return _validate_roots(request)


def write_membership(user_ids, path: Path) -> dict:
    values = sorted({int(value) for value in user_ids})
    raw = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"users": len(values), "sha256": _sha256_bytes(raw), "bytes": len(raw)}


def _read_ids(path: Path, column: str):
    import numpy as np
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    output = []
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS, columns=[column], use_threads=False):
        output.append(np.unique(batch.column(0).to_numpy(zero_copy_only=False)))
        _check_rss_limit()
    return np.unique(np.concatenate(output)) if output else np.array([], dtype=np.uint32)


def prepare_users(request: dict) -> dict:
    roots = _validate_request(request, "prepare-users")
    a_root = Path(roots["a_root"])
    inventory, records = _source_inventory(a_root)
    checked = _check_a_files(a_root, records, read_schema=True)
    all_users: set[int] = set()
    for rel, path, _ in _interaction_paths(a_root, records):
        all_users.update(_read_ids(path, "user_id"))
    if len(all_users) != EXPECTED_USERS:
        raise ValueError("authorized A-window user membership cardinality differs from 99,998")
    membership = _new_root(Path(roots["membership_root"]),
                            (a_root, Path(roots["raw_root"]), Path(roots["holdout_root"])))
    membership.mkdir()
    users_path = membership / "users.json"
    user_identity = write_membership(all_users, users_path)
    manifest = {"schema": "ur001-week26-membership-manifest-v1", "dataset": DATASET,
                "revision": REVISION, "a_root": roots["a_root"],
                "membership_root": roots["membership_root"], "worker_sha256": _worker_sha256(),
                "a_inventory_sha256": SOURCE_INVENTORY_SHA256,
                "a_download_manifest_sha256": sha256_file(a_root / "download-manifest.json"),
                "a_interaction_files": checked, "users_sha256": user_identity["sha256"],
                "users_bytes": user_identity["bytes"], "users": user_identity["users"]}
    with (membership / "membership-manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"schema": "ur001-week26-prepare-users-v1", "status": "prepared",
            "users": EXPECTED_USERS, "users_sha256": user_identity["sha256"],
            "membership_manifest_sha256": sha256_file(membership / "membership-manifest.json")}


def _download(target: Path, opener=None) -> tuple[int, str]:
    opener = opener or urlopen
    request = Request(_url_for_object(), headers={"User-Agent": "vkgr-ur001-week26-bounded/1"})
    digest = hashlib.sha256()
    count = 0
    with opener(request, timeout=60) as response:
        with Path(target).open("xb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                count += len(chunk)
            output.flush()
            os.fsync(output.fileno())
    if count != OBJECT_SIZE:
        raise ValueError("downloaded object size differs; partial bytes retained")
    if digest.hexdigest() != OBJECT_SHA256:
        raise ValueError("downloaded object SHA differs; partial bytes retained")
    return count, digest.hexdigest()


def download(request: dict, opener=None) -> dict:
    opener = opener or urlopen
    roots = _validate_request(request, "download")
    a_root = Path(roots["a_root"])
    _, a_records = _source_inventory(a_root)
    a_files = _check_a_files(a_root, a_records)
    a_manifest_sha = sha256_file(a_root / "download-manifest.json")
    tree = _hf_tree_metadata(opener)
    head = _head_object(opener)
    raw_root = _new_root(Path(roots["raw_root"]),
                         (Path(roots["a_root"]), Path(roots["membership_root"]), Path(roots["holdout_root"])))
    _check_disk_budget((raw_root.parent, Path(roots["membership_root"]).parent,
                        Path(roots["holdout_root"]).parent))
    raw_root.mkdir()
    size, digest = _download(raw_root / "week_26.parquet", opener)
    manifest = {"schema": "ur001-week26-raw-manifest-v1", "dataset": DATASET,
                "revision": REVISION, "path": OBJECT, "size": size, "sha256": digest,
                "lfs_oid_sha256": OBJECT_SHA256, "hf_tree": tree, "hf_head": head,
                "a_root": roots["a_root"], "a_inventory_sha256": SOURCE_INVENTORY_SHA256,
                "a_download_manifest_sha256": a_manifest_sha,
                "a_interaction_files": a_files,
                "raw_root": roots["raw_root"], "worker_sha256": _worker_sha256(),
                "file": "week_26.parquet"}
    with (raw_root / "raw-manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"schema": "ur001-week26-download-v1", "status": "downloaded",
            "path": OBJECT, "size": size, "sha256": digest,
            "raw_manifest_sha256": sha256_file(raw_root / "raw-manifest.json")}


def _membership(root: Path) -> tuple[list[int], dict, str]:
    users_path = _secure_file(root, "users.json")
    raw = users_path.read_bytes()
    users = json.loads(raw.decode("utf-8"))
    if (type(users) is not list or any(type(x) is not int for x in users)
            or len(users) != EXPECTED_USERS or users != sorted(set(users))
            or json.dumps(users, separators=(",", ":")).encode("utf-8") != raw):
        raise ValueError("membership file is not the frozen canonical 99,998-user JSON")
    manifest_path = _secure_file(root, "membership-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {"schema", "dataset", "revision", "a_root", "membership_root", "worker_sha256",
                       "a_inventory_sha256", "a_download_manifest_sha256", "a_interaction_files",
                       "users_sha256", "users_bytes", "users"}
    if (set(manifest) != expected_fields
            or manifest.get("schema") != "ur001-week26-membership-manifest-v1"
            or manifest.get("dataset") != DATASET or manifest.get("revision") != REVISION
            or manifest.get("users") != EXPECTED_USERS or manifest.get("users_sha256") != _sha256_bytes(raw)
            or manifest.get("users_bytes") != len(raw)
            or manifest.get("a_inventory_sha256") != SOURCE_INVENTORY_SHA256
            or manifest.get("membership_root") != str(Path(root))
            or manifest.get("worker_sha256") != _worker_sha256()):
        raise ValueError("membership manifest identity differs")
    return users, manifest, _sha256_bytes(raw)


def _raw_source(root: Path) -> tuple[Path, dict]:
    raw = _secure_file(root, "week_26.parquet")
    manifest_path = _secure_file(root, "raw-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {"schema", "dataset", "revision", "path", "size", "sha256", "lfs_oid_sha256",
                       "hf_tree", "hf_head", "a_root", "a_inventory_sha256", "a_download_manifest_sha256",
                       "a_interaction_files", "raw_root", "worker_sha256", "file"}
    if (set(manifest) != expected_fields
            or manifest.get("schema") != "ur001-week26-raw-manifest-v1"
            or manifest.get("dataset") != DATASET or manifest.get("revision") != REVISION
            or manifest.get("path") != OBJECT or manifest.get("size") != OBJECT_SIZE
            or manifest.get("sha256") != OBJECT_SHA256 or manifest.get("lfs_oid_sha256") != OBJECT_SHA256
            or manifest.get("file") != "week_26.parquet"
            or manifest.get("raw_root") != str(Path(root))
            or manifest.get("worker_sha256") != _worker_sha256()):
        raise ValueError("raw manifest identity differs from the fixed object")
    if (manifest.get("hf_tree") != {"path": OBJECT, "size": OBJECT_SIZE,
                                     "lfs_oid_sha256": OBJECT_SHA256}
            or type(manifest.get("hf_head")) is not dict
            or set(manifest["hf_head"]) != {"content_length", "etag"}
            or manifest["hf_head"].get("content_length") not in (None, OBJECT_SIZE)
            or (manifest["hf_head"].get("etag") is not None
                and type(manifest["hf_head"].get("etag")) is not str)):
        raise ValueError("raw manifest HF tree or HEAD receipt differs from the fixed object")
    if raw.stat().st_size != OBJECT_SIZE or sha256_file(raw) != OBJECT_SHA256:
        raise ValueError("raw Parquet file size or SHA differs from the fixed object")
    if _parquet_schema(raw) != EXPECTED_SCHEMA:
        raise ValueError("raw Parquet schema differs from the fixed interaction schema")
    return raw, manifest


def filter_preflight(request: dict) -> dict:
    roots = _validate_request(request, "filter-preflight")
    a_root = Path(roots["a_root"])
    inventory, records = _source_inventory(a_root)
    a_files = _check_a_files(a_root, records)
    users, member_manifest, users_sha = _membership(Path(roots["membership_root"]))
    if (member_manifest.get("a_root") != roots["a_root"]
            or member_manifest.get("a_download_manifest_sha256") != sha256_file(a_root / "download-manifest.json")
            or member_manifest.get("a_interaction_files") != a_files):
        raise ValueError("membership manifest does not bind the current A-window input identities")
    if users_sha != request["expected_users_sha256"]:
        raise ValueError("users.json SHA differs from the prepare-users receipt pin")
    raw, raw_manifest = _raw_source(Path(roots["raw_root"]))
    if (raw_manifest.get("a_root") != roots["a_root"]
            or raw_manifest.get("a_inventory_sha256") != SOURCE_INVENTORY_SHA256
            or raw_manifest.get("a_download_manifest_sha256") != sha256_file(a_root / "download-manifest.json")
            or raw_manifest.get("a_interaction_files") != a_files):
        raise ValueError("raw manifest does not bind the current A-window input identities")
    _new_root(Path(roots["holdout_root"]),
              (a_root, Path(roots["raw_root"]), Path(roots["membership_root"])))
    _check_disk_budget((Path(roots["raw_root"]).parent, Path(roots["membership_root"]).parent,
                        Path(roots["holdout_root"]).parent), required_bytes=OBJECT_SIZE + 8 * 1024**3)
    return {"schema": "ur001-week26-filter-preflight-v1", "status": "ready",
            "engineering": False, "filtered_counts_generated": False,
            "source_object": OBJECT, "source_sha256": OBJECT_SHA256,
            "a_inventory_sha256": SOURCE_INVENTORY_SHA256,
            "a_interaction_files": a_files, "raw_sha256": OBJECT_SHA256,
            "raw_manifest_sha256": sha256_file(Path(roots["raw_root"]) / "raw-manifest.json"),
            "users": len(users), "users_sha256": users_sha,
            "membership_manifest_sha256": sha256_file(Path(roots["membership_root"]) / "membership-manifest.json")}


def filter_parquet(source: Path, target: Path, users: set[int], a_items: set[int]) -> dict:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, target = Path(source), Path(target)
    parquet = pq.ParquetFile(source)
    user_array = np.sort(np.fromiter(users, dtype=np.uint32))
    writer = pq.ParquetWriter(target, parquet.schema_arrow, write_statistics=False)
    filtered_rows = 0
    filtered_users: set[int] = set()
    item_chunks = []
    try:
        for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS, use_threads=False):
            user_values = batch.column(batch.schema.get_field_index("user_id")).to_numpy(zero_copy_only=False)
            positions = np.searchsorted(user_array, user_values)
            safe_positions = np.minimum(positions, len(user_array) - 1)
            mask = (positions < len(user_array)) & (user_array[safe_positions] == user_values)
            if not bool(mask.any()):
                continue
            filtered = batch.filter(pa.array(mask))
            _check_disk_budget((target.parent,), required_bytes=OBJECT_SIZE + 8 * 1024**3)
            writer.write_batch(filtered, row_group_size=MAX_BATCH_ROWS)
            filtered_rows += filtered.num_rows
            selected_users = filtered.column(filtered.schema.get_field_index("user_id")).to_numpy(zero_copy_only=False)
            selected_items = filtered.column(filtered.schema.get_field_index("item_id")).to_numpy(zero_copy_only=False)
            filtered_users.update(int(value) for value in np.unique(selected_users))
            item_chunks.append(np.unique(selected_items))
            _check_rss_limit()
    finally:
        writer.close()
    _check_rss_limit()
    filtered_items_array = np.unique(np.concatenate(item_chunks)) if item_chunks else np.array([], dtype=np.uint32)
    if len(filtered_items_array):
        union_items = np.union1d(np.fromiter(a_items, dtype=np.uint32), filtered_items_array)
    else:
        union_items = np.fromiter(a_items, dtype=np.uint32)
    return {"filtered_rows": filtered_rows, "filtered_users": len(filtered_users),
            "filtered_items": int(len(filtered_items_array)), "union_items": int(len(union_items)),
            "output_sha256": sha256_file(target), "output_size": target.stat().st_size}


def _unique_a_items(item_parts):
    import numpy as np
    _check_rss_limit()
    result = np.unique(np.concatenate(item_parts)) if item_parts else np.array([], dtype=np.uint32)
    _check_rss_limit()
    return result


def hypothesis_status(filtered_rows: int, union_items: int) -> tuple[int, int, str]:
    gap_rows = EXPECTED_ROWS - int(filtered_rows)
    gap_items = EXPECTED_UNION_ITEMS - int(union_items)
    status = "假设被证实" if 0 <= gap_rows <= MAX_GAP and 0 <= gap_items <= MAX_GAP else "未证实"
    return gap_rows, gap_items, status


def filter_week26(request: dict) -> dict:
    roots = _validate_request(request, "filter")
    # Repeat the read-only gate immediately before creating any holdout output.
    preflight = filter_preflight(request)
    raw_root, membership_root, a_root = (Path(roots[name]) for name in ("raw_root", "membership_root", "a_root"))
    users, _, users_sha = _membership(membership_root)
    _, records = _source_inventory(a_root)
    import numpy as np
    a_item_parts = []
    for _, path, _ in _interaction_paths(a_root, records):
        a_item_parts.append(_read_ids(path, "item_id"))
    a_items = _unique_a_items(a_item_parts)
    raw, raw_manifest = _raw_source(raw_root)
    holdout_root = _new_root(Path(roots["holdout_root"]), (a_root, raw_root, membership_root))
    holdout_root.mkdir()
    target = holdout_root / "week_26.parquet"
    counts = filter_parquet(raw, target, set(users), a_items)
    gap_rows, gap_items, status = hypothesis_status(counts["filtered_rows"], counts["union_items"])
    manifest = {"schema": "ur001-week26-holdout-manifest-v1", "dataset": DATASET,
                "revision": REVISION, "source_object": OBJECT,
                "a_root": roots["a_root"], "raw_root": roots["raw_root"],
                "membership_root": roots["membership_root"], "holdout_root": roots["holdout_root"],
                "a_interaction_files": preflight["a_interaction_files"],
                "worker_sha256": _worker_sha256(),
                "a_inventory_sha256": SOURCE_INVENTORY_SHA256,
                "source_size": OBJECT_SIZE, "source_sha256": OBJECT_SHA256,
                "raw_manifest_sha256": sha256_file(raw_root / "raw-manifest.json"),
                "a_download_manifest_sha256": sha256_file(a_root / "download-manifest.json"),
                "membership_manifest_sha256": sha256_file(membership_root / "membership-manifest.json"),
                "users_sha256": users_sha, "output": "week_26.parquet", **counts,
                "gap_rows": gap_rows, "gap_items": gap_items, "hypothesis_status": status,
                "statistics_disabled": True}
    with (holdout_root / "holdout-manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"schema": "ur001-week26-filter-v1", "status": "complete",
            "source_sha256": OBJECT_SHA256, "users_sha256": users_sha,
            "filtered_rows": counts["filtered_rows"], "filtered_users": counts["filtered_users"],
            "filtered_items": counts["filtered_items"], "union_items": counts["union_items"],
            "gap_rows": gap_rows, "gap_items": gap_items, "hypothesis_status": status,
            "output_sha256": counts["output_sha256"], "output_size": counts["output_size"],
            "holdout_manifest_sha256": sha256_file(holdout_root / "holdout-manifest.json")}


def _load_request(path: Path) -> dict:
    raw = Path(path).read_bytes()
    if len(raw) > 64 * 1024:
        raise ValueError("request exceeds 64 KiB")
    return json.loads(raw.decode("utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dry", "source-preflight", "prepare-users", "download",
                                           "filter-preflight", "filter"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "dry":
        print(json.dumps({"schema": "ur001-week26-dry-v1", "status": "ready",
                          "stages": ["source-preflight", "prepare-users", "download",
                                     "filter-preflight", "filter"]}, sort_keys=True))
        return 0
    if args.request is None:
        parser.error("--request is required outside dry")
    request = _load_request(args.request)
    result = {"source-preflight": source_preflight, "prepare-users": prepare_users,
              "download": download, "filter-preflight": filter_preflight,
              "filter": filter_week26}[args.stage](request)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
