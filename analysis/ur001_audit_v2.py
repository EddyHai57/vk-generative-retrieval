#!/usr/bin/env python3
"""Pinned v2 A structural audit; original download and science scanner stay byte-identical."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).absolute().parents[1]
OLD_WORKER = ROOT / "analysis/ur001_prepare.py"
OLD_WORKER_SHA256 = "86e3ea98142b762c844d0deb66b2cf3d22f239df591e31f8f5d5f7d57263f94b"
# Original data-card totals: Kiro adopts the whole-subset (including test week)
# interpretation as a working hypothesis, not directly verified with test.
EXPECTED_SCALE = {"rows": 407_854_360, "users": 100_000, "items": 12_467_302}
LEGACY_EXPECTED_SCALE = EXPECTED_SCALE
# Kiro second ruling (2026-10-20): observed train week00-24 + validation week25 window.
EXPECTED_SCALE_v2 = {"rows": 392_970_099, "users": 99_998, "items": 12_340_612}
DECISION = "2026-10-20 Kiro A-scale second ruling; test-week hypothesis not directly verified with test"
NAMESPACE = "audit-v2"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_old_worker():
    if OLD_WORKER.is_symlink() or _hash(OLD_WORKER) != OLD_WORKER_SHA256:
        raise ValueError("ALERT fixed original audit worker SHA differs")
    spec = importlib.util.spec_from_file_location("_ur001_original_private_v2", OLD_WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPECTED_SCALE = dict(EXPECTED_SCALE_v2)
    return module


def _roots(root: Path, old_subset_root: Path):
    old = _load_old_worker()
    root = old._secure_root(root)
    old_root = old._secure_root(old_subset_root)
    if root == old_root or root in old_root.parents or old_root in root.parents:
        raise ValueError("ALERT downloaded and old-subset roots overlap")
    return old, root, old_root


def _new_namespace_absent(old, root: Path) -> None:
    target = old._resolve_under_root(root, NAMESPACE, must_exist=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError("audit-v2 output namespace already exists")


def dry(root: Path, old_subset_root: Path) -> dict:
    old, root, old_root = _roots(root, old_subset_root)
    _new_namespace_absent(old, root)
    inventory = old.load_file_manifest()
    return {"action": "dry", "status": "ready", "revision": old.FIXED_REVISION,
            "file_count": len(inventory["files"]), "test_paths": 0,
            "data_root": str(root), "old_subset_root": str(old_root),
            "output_namespace": NAMESPACE, "output_exists": False,
            "old_worker_sha256": OLD_WORKER_SHA256, "wrapper_sha256": _hash(Path(__file__)),
            "decision": DECISION, "test_directly_verified": False}


def _count_ids(path: Path, field: str) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    dtype = parquet.schema_arrow.field(field).type
    if str(dtype) != "uint32":
        raise ValueError("ALERT metadata ID type differs from uint32")
    values = []
    rows = nulls = 0
    for batch in parquet.iter_batches(batch_size=65_536, columns=[field], use_threads=False):
        column = batch.column(0)
        rows += batch.num_rows
        nulls += column.null_count
        if column.null_count:
            column = column.drop_null()
        values.append(column.to_numpy(zero_copy_only=False))
    ids = np.concatenate(values) if values else np.empty(0, dtype=np.uint32)
    distinct = int(np.unique(ids).size)
    return {"rows": rows, "distinct_ids": distinct, "null_ids": nulls,
            "duplicate_id_rows": int(len(ids) - distinct), "column_read": field}


def metadata_counts(root: Path, old_subset_root: Path) -> dict:
    old, root, _ = _roots(root, old_subset_root)
    _new_namespace_absent(old, root)
    inventory = old.load_file_manifest()
    downloaded_path = old._resolve_under_root(root, "download-manifest.json", want_file=True)
    downloaded = json.loads(downloaded_path.read_text(encoding="utf-8"))
    required = {"schema_version", "task_id", "status", "dataset", "revision", "source_manifest_sha256",
                "script_sha256", "files", "total_bytes", "preflight", "source_metadata"}
    if (type(downloaded) is not dict or set(downloaded) != required
            or downloaded.get("schema_version") != 1 or downloaded.get("task_id") != "ur001-preparation"
            or downloaded.get("status") != "downloaded" or downloaded.get("dataset") != old.DATASET
            or downloaded.get("total_bytes") != inventory["total_bytes"]
            or downloaded.get("script_sha256") != OLD_WORKER_SHA256
            or downloaded.get("revision") != old.FIXED_REVISION
            or downloaded.get("source_manifest_sha256") != _hash(old.DEFAULT_FILE_MANIFEST)):
        raise ValueError("ALERT download manifest source identity differs")
    expected = {item["path"]: item for item in inventory["files"]}
    source_metadata = downloaded.get("source_metadata")
    if type(source_metadata) is not list or len(source_metadata) != 29 or any(
            type(item) is not dict or set(item) != {"path", "content_length", "lfs_oid_sha256"}
            for item in source_metadata):
        raise ValueError("ALERT download source metadata inventory differs")
    source_by_path = {item["path"]: item for item in source_metadata}
    if set(source_by_path) != set(expected) or any(
            source_by_path[name]["content_length"] != item["size"]
            or source_by_path[name]["lfs_oid_sha256"] != item["sha256"]
            for name, item in expected.items()):
        raise ValueError("ALERT download source metadata pin differs")
    file_rows = downloaded.get("files")
    if type(file_rows) is not list or len(file_rows) != 29 or any(
            type(item) is not dict or set(item) != {"path", "size", "sha256"} for item in file_rows):
        raise ValueError("ALERT download inventory records differ")
    recorded = {item["path"]: item for item in file_rows}
    if len(recorded) != 29 or set(recorded) != set(expected):
        raise ValueError("ALERT download inventory differs")
    results = {}
    for label, relative, field, data_card in (
        ("users", "metadata/users_metadata.parquet", "user_id", EXPECTED_SCALE_v2["users"]),
        ("items", "metadata/items_metadata.parquet", "item_id", EXPECTED_SCALE_v2["items"]),
    ):
        path = old._resolve_under_root(root, relative, want_file=True)
        record = expected[relative]
        if (recorded[relative] != record or path.stat().st_size != record["size"]
                or _hash(path) != record["sha256"]):
            raise ValueError("ALERT metadata inventory SHA/size differs: " + relative)
        result = _count_ids(path, field)
        result.update({"file": relative, "size": record["size"], "sha256": record["sha256"],
                       "window_expected": data_card,
                       "distinct_matches_window": result["distinct_ids"] == data_card,
                       "legacy_data_card_expected": LEGACY_EXPECTED_SCALE[label],
                       "distinct_matches_legacy_data_card": result["distinct_ids"] == LEGACY_EXPECTED_SCALE[label]})
        results[label] = result
    integrity_alert = any(value["null_ids"] or value["duplicate_id_rows"] for value in results.values())
    return {"action": "metadata-counts", "status": "ALERT" if integrity_alert else "counted", "metadata": results,
            "metadata_mismatch_is_descriptive": True, "test_directly_verified": False,
            "old_worker_sha256": OLD_WORKER_SHA256, "wrapper_sha256": _hash(Path(__file__)),
            "download_manifest_sha256": _hash(downloaded_path), "decision": DECISION,
            "v2_expected_scale": EXPECTED_SCALE_v2, "legacy_expected_scale": LEGACY_EXPECTED_SCALE}


def audit_preflight(root: Path, old_subset_root: Path) -> dict:
    old, root, old_root = _roots(root, old_subset_root)
    _new_namespace_absent(old, root)
    result = old.audit_preflight(root, old_root)
    return {**result, "v2_output_namespace": NAMESPACE, "v2_expected_scale": EXPECTED_SCALE_v2,
            "old_worker_sha256": OLD_WORKER_SHA256, "wrapper_sha256": _hash(Path(__file__)),
            "decision": DECISION, "test_directly_verified": False}


def _isolate_outputs(old):
    resolve = old._resolve_under_root
    create = old._create_directory_under_root
    new_file = old._new_file_under_root
    names = {"audit-work", "structure-audit.json", "structure-audit-alert.json"}

    def mapped(name):
        value = str(name)
        return NAMESPACE + "/" + value if value in names else value

    def mapped_resolve(root, relative, *args, **kwargs):
        return resolve(root, mapped(relative), *args, **kwargs)

    def mapped_create(root, relative, *args, **kwargs):
        return create(root, mapped(relative), *args, **kwargs)

    def mapped_new_file(root, relative, *args, **kwargs):
        return new_file(root, mapped(relative), *args, **kwargs)

    old._resolve_under_root = mapped_resolve
    old._create_directory_under_root = mapped_create
    old._new_file_under_root = mapped_new_file


def audit(root: Path, old_subset_root: Path) -> dict:
    old, root, old_root = _roots(root, old_subset_root)
    _new_namespace_absent(old, root)
    _isolate_outputs(old)
    try:
        report = old.audit_dataset(root, old_root)
    except (ValueError, OSError) as exc:
        reason = str(exc) if str(exc).startswith("ALERT") else "ALERT audit integrity/input failure: " + str(exc)
        report = old._write_alert(root, reason, {})
    receipt = {"schema": "ur001-audit-v2-receipt-v1", "status": report["status"],
               "original_report": "structure-audit.json" if report["status"] == "audited" else "structure-audit-alert.json",
               "old_worker_sha256": OLD_WORKER_SHA256, "wrapper_sha256": _hash(Path(__file__)),
               "download_manifest_sha256": _hash(old._resolve_under_root(root, "download-manifest.json", want_file=True)),
               "decision": DECISION, "test_directly_verified": False,
               "expected_scale_v2": EXPECTED_SCALE_v2, "legacy_expected_scale": LEGACY_EXPECTED_SCALE}
    namespace = old._resolve_under_root(root, NAMESPACE, want_dir=True)
    report_path = old._resolve_under_root(root, NAMESPACE + "/" + receipt["original_report"], want_file=True)
    receipt["original_report_sha256"] = _hash(report_path)
    path = old._new_file_under_root(root, NAMESPACE + "/v2-receipt.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return {"action": "audit", **receipt, "structure_report": report}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("dry", "metadata-counts", "audit-preflight", "audit"):
        item = sub.add_parser(command)
        item.add_argument("--root", type=Path, required=True)
        item.add_argument("--old-subset-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = {"dry": dry, "metadata-counts": metadata_counts,
                  "audit-preflight": audit_preflight, "audit": audit}[args.command](args.root, args.old_subset_root)
    except (ValueError, OSError) as exc:
        print(json.dumps({"action": args.command, "status": "ALERT", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 2 if result.get("status") == "ALERT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
