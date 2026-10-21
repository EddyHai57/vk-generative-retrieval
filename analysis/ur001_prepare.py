#!/usr/bin/env python3
"""Download and structurally audit the approved public VK-LSVD ur0.01 inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Optional
import shutil
import sys
from urllib.request import Request, urlopen


DATASET = "deepvk/VK-LSVD"
FIXED_REVISION = "084a18386e149a454dd6dbe0123478e75c6b09ce"
ROOT = Path(__file__).absolute().parents[1]
DEFAULT_FILE_MANIFEST = ROOT / "configs/ur001-preparation/hf-files.json"
FEEDBACK = ("like", "dislike", "share", "bookmark", "click_on_author", "open_comments")
EXPECTED_INTERACTION_SCHEMA = {
    "user_id": "uint32", "item_id": "uint32", "place": "uint8", "platform": "uint8",
    "agent": "uint8", "timespent": "uint8", **{name: "bool" for name in FEEDBACK},
}
EXPECTED_METADATA_SCHEMA = {"item_id": "uint32", "author_id": "uint32", "duration": "uint8",
                            "train_interactions_rank": "uint32"}
EXPECTED_USER_METADATA_SCHEMA = {"user_id": "uint32", "age": "uint8", "gender": "uint8",
                                 "geo": "uint8", "train_interactions_rank": "uint32"}
MAX_BATCH_ROWS = 65_536
PAIR_CHUNK_ROWS = 8_000_000
MAX_RSS_BYTES = 64 * 1024**3
EXPECTED_SCALE = {"rows": 407_854_360, "users": 100_000, "items": 12_467_302}


def _absolute_path_without_symlinks(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if ".." in candidate.parts:
        raise ValueError("ALERT path traversal component is forbidden")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("ALERT symlink path component is forbidden")
    return candidate


def _secure_root(root: Path, must_exist: bool = True) -> Path:
    candidate = _absolute_path_without_symlinks(root)
    if not must_exist and candidate.exists():
        raise FileExistsError("destination root already exists")
    if must_exist and not candidate.is_dir():
        raise ValueError("ALERT root must already exist as a directory")
    resolved = candidate.resolve(strict=must_exist)
    if must_exist and not resolved.is_dir():
        raise ValueError("ALERT root must be a directory")
    return resolved


def _resolve_under_root(root: Path, relative: Path | str, must_exist: bool = True,
                        want_file: bool = False, want_dir: bool = False) -> Path:
    base = _secure_root(root)
    rel = PurePosixPath(str(relative))
    if rel.is_absolute() or ".." in rel.parts or "\\" in str(relative) or not rel.parts:
        raise ValueError("ALERT unsafe relative path")
    current = base
    for index, part in enumerate(rel.parts):
        current = current / part
        if current.is_symlink():
            raise ValueError("ALERT symlink input path is forbidden")
        exists = current.exists()
        if not exists and must_exist:
            raise ValueError("ALERT authorized input path is missing")
        if exists:
            resolved = current.resolve(strict=True)
            if resolved != base and base not in resolved.parents:
                raise ValueError("ALERT resolved input escapes authorized root")
            if index < len(rel.parts) - 1 and not resolved.is_dir():
                raise ValueError("ALERT input parent is not a directory")
            current = resolved
    if must_exist:
        if want_file and not current.is_file():
            raise ValueError("ALERT authorized input is not a file")
        if want_dir and not current.is_dir():
            raise ValueError("ALERT authorized input is not a directory")
    return current


def _path_under_root(root: Path, path: Path, want_file: bool = True) -> Path:
    base = _secure_root(root)
    candidate = Path(path)
    if candidate.is_absolute():
        candidate = _absolute_path_without_symlinks(candidate)
        try:
            relative = candidate.relative_to(base)
        except ValueError:
            raise ValueError("ALERT input path is outside authorized root") from None
    else:
        relative = candidate
    return _resolve_under_root(base, relative, want_file=want_file)


def _secure_absolute_file(path: Path) -> Path:
    candidate = _absolute_path_without_symlinks(path)
    if not candidate.is_file():
        raise ValueError("ALERT source file is missing or not regular")
    return candidate.resolve(strict=True)


def _create_directory_under_root(root: Path, relative: str, exist_ok: bool = False) -> Path:
    base = _secure_root(root)
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or not rel.parts:
        raise ValueError("ALERT unsafe directory path")
    current = base
    for index, part in enumerate(rel.parts):
        next_path = current / part
        if next_path.is_symlink():
            raise ValueError("ALERT symlink output parent is forbidden")
        if next_path.exists():
            resolved = next_path.resolve(strict=True)
            if resolved != base and base not in resolved.parents:
                raise ValueError("ALERT output parent escapes authorized root")
            if not resolved.is_dir():
                raise ValueError("ALERT output parent is not a directory")
            if index == len(rel.parts) - 1 and not exist_ok:
                raise FileExistsError("output directory already exists")
            current = resolved
        else:
            next_path.mkdir()
            current = next_path.resolve(strict=True)
            if current != base and base not in current.parents:
                raise ValueError("ALERT created output directory escapes authorized root")
    return current


def _new_file_under_root(root: Path, relative: str) -> Path:
    base = _secure_root(root)
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative or not rel.parts:
        raise ValueError("ALERT unsafe output path")
    parent = _create_directory_under_root(base, "/".join(rel.parts[:-1]), exist_ok=True) if len(rel.parts) > 1 else base
    target = parent / rel.parts[-1]
    if target.is_symlink() or target.exists():
        raise FileExistsError("output file already exists or is a symlink")
    if parent.resolve(strict=True) != base and base not in parent.resolve(strict=True).parents:
        raise ValueError("ALERT output parent escapes authorized root")
    return target


def _new_file_path_under_root(root: Path, path: Path) -> Path:
    base = _secure_root(root)
    candidate = _absolute_path_without_symlinks(path)
    try:
        relative = candidate.relative_to(base).as_posix()
    except ValueError:
        raise ValueError("ALERT output path is outside authorized root") from None
    return _new_file_under_root(base, relative)


def _allowed_paths() -> list[str]:
    return ([f"subsamples/ur0.01/train/week_{week:02d}.parquet" for week in range(25)]
            + ["subsamples/ur0.01/validation/week_25.parquet", "metadata/item_embeddings.npz",
               "metadata/items_metadata.parquet", "metadata/users_metadata.parquet"])


def validate_file_manifest(value: object) -> dict:
    if type(value) is not dict or set(value) != {"dataset", "revision", "files"}:
        raise ValueError("invalid manifest fields")
    if value["dataset"] != DATASET or value["revision"] != FIXED_REVISION:
        raise ValueError("dataset revision mismatch")
    files = value["files"]
    if type(files) is not list or len(files) != 29:
        raise ValueError("inventory must contain exactly 29 authorized files")
    expected = _allowed_paths()
    seen = set()
    clean = []
    for record in files:
        if type(record) is not dict or set(record) != {"path", "size", "sha256"}:
            raise ValueError("invalid file record")
        name, size, digest = record["path"], record["size"], record["sha256"]
        if type(name) is not str:
            raise ValueError("invalid path")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or name not in expected:
            raise ValueError("path outside authorized inventory")
        if name in seen:
            raise ValueError("duplicate inventory path")
        if type(size) is not int or size <= 0:
            raise ValueError("invalid file size")
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid file sha256")
        seen.add(name)
        clean.append({"path": name, "size": size, "sha256": digest})
    if seen != set(expected):
        raise ValueError("inventory paths do not exactly match authorized inventory")
    if [x["path"] for x in clean] != expected:
        raise ValueError("inventory order differs from frozen source list")
    return {"dataset": DATASET, "revision": FIXED_REVISION, "files": clean,
            "total_bytes": sum(x["size"] for x in clean)}


def load_file_manifest(path: Path = DEFAULT_FILE_MANIFEST) -> dict:
    value = json.loads(_path_under_root(ROOT, Path(path)).read_text(encoding="utf-8"))
    return validate_file_manifest(value)


def require_new_root(root: Path, old_subset_root: Optional[Path] = None) -> Path:
    root = _absolute_path_without_symlinks(Path(root))
    if root.exists() or root.is_symlink():
        raise FileExistsError("destination root already exists")
    if old_subset_root is not None:
        old = _secure_root(Path(old_subset_root))
        candidate = root.resolve(strict=False)
        if candidate == old or old in candidate.parents or candidate in old.parents:
            raise ValueError("destination overlaps old subset root")
    _secure_root(root.parent)
    return root


def _url_for(record: dict, revision: str) -> str:
    from urllib.parse import quote
    return (f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/"
            f"{quote(record['path'], safe='/')}?download=true")


def _source_inventory_metadata(manifest: dict) -> list[dict]:
    """Read only HF tree metadata for the three authorized folders, never file contents."""
    from urllib.parse import quote, urlencode
    expected = {x["path"]: x for x in manifest["files"]}
    folders = ("subsamples/ur0.01/train", "subsamples/ur0.01/validation", "metadata")
    observed = {}
    for folder in folders:
        base = f"https://huggingface.co/api/datasets/{DATASET}/tree/{manifest['revision']}/{quote(folder, safe='/')}"
        url = base + "?" + urlencode({"recursive": "false", "expand": "true"})
        request = Request(url, headers={"User-Agent": "vkgr-ur001-preparation/1"})
        with urlopen(request, timeout=30) as response:
            entries = json.loads(response.read().decode("utf-8"))
        if type(entries) is not list:
            raise ValueError("HF tree metadata response is not a list")
        for entry in entries:
            path = entry.get("path") if type(entry) is dict else None
            if type(path) is str and path not in expected and not path.startswith(folder + "/"):
                path = f"{folder}/{path}"
            if path not in expected:
                continue
            lfs = entry.get("lfs") or {}
            oid = lfs.get("oid") if type(lfs) is dict else None
            if type(oid) is str:
                oid = oid.removeprefix("sha256:")
            size = entry.get("size")
            if type(size) is not int or oid != expected[path]["sha256"] or size != expected[path]["size"]:
                raise ValueError(f"HF tree LFS metadata mismatch: {path}")
            observed[path] = {"path": path, "content_length": size, "lfs_oid_sha256": oid}
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        raise ValueError("HF tree metadata inventory incomplete: " + ",".join(missing))
    return [observed[x["path"]] for x in manifest["files"]]


def preflight(root: Path, old_subset_root: Path, manifest: dict, check_network: bool = True) -> dict:
    root = require_new_root(root, old_subset_root)
    old_subset_root = _secure_root(Path(old_subset_root))
    parent = root.parent
    parent = _secure_root(parent)
    usage = shutil.disk_usage(parent)
    free_fraction = usage.free / usage.total if usage.total else 0.0
    if free_fraction < 0.20:
        raise ValueError("destination filesystem has less than 20% free space")
    pair_scratch_bytes = 407_854_360 * 8
    required_bytes = manifest["total_bytes"] + pair_scratch_bytes
    if usage.free - required_bytes < usage.total * 0.20:
        raise ValueError("destination filesystem cannot fit source files and audit scratch while preserving 20% headroom")
    try:
        import numpy  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ValueError("audit requires installed numpy and pyarrow") from exc
    source = _source_inventory_metadata(manifest) if check_network else []
    return {"action": "preflight", "status": "ready", "dataset": manifest["dataset"],
            "revision": manifest["revision"], "file_count": len(manifest["files"]),
            "total_bytes": manifest["total_bytes"], "destination_parent_free_bytes": usage.free,
            "destination_free_fraction": free_fraction, "old_subset_root_distinct": True,
            "pair_scratch_bytes": pair_scratch_bytes,
            "free_after_download_and_scratch_reserve": usage.free - required_bytes,
            "dependencies": {"numpy": True, "pyarrow": True},
            "source_metadata_checked": check_network, "source_metadata": source}


def _sha256(path: Path, root: Optional[Path] = None) -> str:
    path = _path_under_root(root, Path(path)) if root is not None else _secure_absolute_file(Path(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_record(root: Path, record: dict, revision: str, opener=None) -> dict:
    root = _secure_root(root)
    relative = PurePosixPath(record["path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in record["path"]:
        raise ValueError("unsafe download path")
    relative_name = relative.as_posix()
    parent_relative = "/".join(relative.parts[:-1])
    if parent_relative:
        _create_directory_under_root(root, parent_relative, exist_ok=True)
    destination = _new_file_under_root(root, relative_name)
    open_url = opener or (lambda url: urlopen(Request(url, headers={"User-Agent": "vkgr-ur001-preparation/1"}), timeout=60))
    digest = hashlib.sha256()
    count = 0
    # Write directly to a new file so a failure leaves its partial bytes as evidence.
    with open_url(_url_for(record, revision)) as response, destination.open("xb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            count += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    if count != record["size"]:
        raise ValueError(f"download size mismatch for {record['path']}: {count}")
    if digest.hexdigest() != record["sha256"]:
        raise ValueError(f"download sha256 mismatch for {record['path']}")
    return {"path": record["path"], "size": count, "sha256": digest.hexdigest()}


def _write_progress(root: Path, record: dict) -> None:
    root = _secure_root(root)
    progress_name = "download-progress.jsonl"
    candidate = root / progress_name
    if candidate.is_symlink():
        raise ValueError("ALERT download progress path is a symlink")
    progress = (_resolve_under_root(root, progress_name, want_file=True)
                if candidate.exists() else _new_file_under_root(root, progress_name))
    with progress.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _script_sha256() -> str:
    return _sha256(Path(__file__), root=ROOT)


def download_all(root: Path, old_subset_root: Path, manifest: dict, check_network: bool = True) -> dict:
    readiness = preflight(root, old_subset_root, manifest, check_network=check_network)
    root = require_new_root(root, old_subset_root)
    root.mkdir(parents=False, exist_ok=False)
    root = _secure_root(root)
    completed = []
    for record in manifest["files"]:
        result = download_record(root, record, manifest["revision"])
        completed.append(result)
        _write_progress(root, result)
    result = {"schema_version": 1, "task_id": "ur001-preparation", "status": "downloaded",
              "dataset": manifest["dataset"], "revision": manifest["revision"],
              "source_manifest_sha256": hashlib.sha256(
                  _path_under_root(ROOT, DEFAULT_FILE_MANIFEST).read_bytes()).hexdigest(),
              "script_sha256": _script_sha256(), "files": completed,
              "total_bytes": sum(x["size"] for x in completed),
              "preflight": {k: v for k, v in readiness.items() if k != "source_metadata"},
              "source_metadata": readiness["source_metadata"]}
    out = _new_file_under_root(root, "download-manifest.json")
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result


def _schema(path: Path, root: Optional[Path] = None) -> dict[str, str]:
    import pyarrow.parquet as pq
    path = _path_under_root(root, path) if root is not None else _secure_absolute_file(path)
    return {field.name: str(field.type) for field in pq.ParquetFile(path).schema_arrow}


def _check_schema(path: Path, expected: dict[str, str], label: str, root: Optional[Path] = None) -> None:
    actual = _schema(path, root=root)
    if actual != expected:
        raise ValueError(f"ALERT schema mismatch in {label}: expected frozen schema")


def _check_parquet_nulls(root: Path, relative: str, label: str) -> None:
    import pyarrow.parquet as pq
    path = _resolve_under_root(root, relative, want_file=True)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS, use_threads=False):
        if any(column.null_count for column in batch.columns):
            raise ValueError(f"ALERT null value anywhere in {label}")


def _batches(path: Path, columns=None, root: Optional[Path] = None):
    import pyarrow.parquet as pq
    path = _path_under_root(root, path) if root is not None else _secure_absolute_file(path)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS, columns=columns, use_threads=False):
        if batch.num_rows and any(column.null_count for column in batch.columns):
            raise ValueError("ALERT null values in required source table")
        yield batch


def _histogram(values: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(values.items())}


def _summary(lengths: list[int]) -> dict:
    values = sorted(lengths)
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p99": None, "max": None,
                "histogram": {}}
    def quantile(q):
        import math
        return values[max(0, math.ceil(len(values) * q) - 1)]
    return {"count": len(values), "p50": quantile(.5), "p90": quantile(.9),
            "p99": quantile(.99), "max": values[-1], "histogram": _histogram(Counter(values))}


def _candidate_summary(validation_targets: dict, like_counts: dict, exposure_counts: dict,
                       catalog: set[int]) -> tuple[dict, list[int], list[int]]:
    counts = {"eligible": 0, "in_catalog": 0, "oov": 0, "no_history": 0}
    observed_lengths, positive_lengths = [], []
    for user, target in validation_targets.items():
        positives = like_counts.get(user, 0)
        if positives < 1:
            counts["no_history"] += 1
            continue
        counts["eligible"] += 1
        counts["in_catalog" if target in catalog else "oov"] += 1
        observed_lengths.append(exposure_counts.get(user, 0))
        positive_lengths.append(positives)
    return counts, observed_lengths, positive_lengths


def audit_rows_for_test(train_partitions, validation_partition):
    """Small in-memory oracle for tests; production audit uses Arrow batches."""
    exposures, likes = Counter(), Counter()
    users, items = set(), set()
    rows = like_events = dislike_events = 0
    pair_counts = Counter()
    exact_counts = Counter()
    for week, records in train_partitions:
        for row in records:
            rows += 1
            user, item = row["user_id"], row["item_id"]
            users.add(user); items.add(item); exposures[user] += 1
            pair_counts[(user, item)] += 1
            exact_counts[(week, tuple(sorted(row.items())))] += 1
            if row["like"]:
                likes[user] += 1; like_events += 1
            if row["dislike"]:
                dislike_events += 1
    week, validation = validation_partition
    del week
    targets = {}
    validation_rows = 0
    validation_users = set()
    validation_items = set()
    for row in validation:
        validation_rows += 1
        user, item = row["user_id"], row["item_id"]
        validation_users.add(user); validation_items.add(item)
        if row["like"]:
            targets.setdefault(user, item)
    candidates, observed_lengths, positive_lengths = _candidate_summary(targets, likes, exposures, items)
    return {"train": {"rows": rows, "users": len(users), "items": len(items),
                       "like_events": like_events, "dislike_events": dislike_events,
                       "duplicate_user_item_excess_rows": sum(n - 1 for n in pair_counts.values()),
                       "exact_duplicate_rows": sum(n - 1 for n in exact_counts.values())},
            "validation": {"rows": validation_rows, "users": len(validation_users),
                           "items": len(validation_items), "candidate_counts": candidates,
                           "observed_history_length_untruncated": _summary(observed_lengths),
                           "observed_history_length_capped50": _summary([min(50, n) for n in observed_lengths]),
                           "positive_history_length": _summary(positive_lengths)}}


def _check_memory_budget() -> int:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    if peak_bytes > MAX_RSS_BYTES:
        raise ValueError("ALERT process peak RSS exceeded 64 GiB")
    return peak_bytes


def _update_counts(counter: Counter, values) -> None:
    import numpy as np
    unique, counts = np.unique(values, return_counts=True)
    counter.update({int(key): int(count) for key, count in zip(unique.tolist(), counts.tolist())})


def _membership(values, sorted_candidates):
    import numpy as np
    if not len(sorted_candidates):
        return np.zeros(len(values), dtype=bool)
    positions = np.searchsorted(sorted_candidates, values)
    in_range = positions < len(sorted_candidates)
    return in_range & (sorted_candidates[np.minimum(positions, len(sorted_candidates) - 1)] == values)


def _pair_keys(batch):
    import numpy as np
    users = batch.column(batch.schema.get_field_index("user_id")).to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    items = batch.column(batch.schema.get_field_index("item_id")).to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    return (users << np.uint64(32)) | items


def _count_sorted_duplicates(values) -> int:
    import numpy as np
    values.sort(kind="quicksort")
    count = 0
    prior = None
    for start in range(0, len(values), PAIR_CHUNK_ROWS):
        chunk = values[start:min(start + PAIR_CHUNK_ROWS, len(values))]
        if not len(chunk):
            continue
        count += int(np.count_nonzero(chunk[1:] == chunk[:-1]))
        if prior is not None and int(chunk[0]) == prior:
            count += 1
        prior = int(chunk[-1])
    values.flush()
    return count


def _count_cross_duplicates(sorted_existing, candidate_values) -> int:
    import numpy as np
    positions = np.searchsorted(sorted_existing, candidate_values)
    within = positions < len(sorted_existing)
    if not len(sorted_existing):
        return 0
    safe_positions = np.minimum(positions, len(sorted_existing) - 1)
    return int(np.count_nonzero(within & (sorted_existing[safe_positions] == candidate_values)))


def _count_cross_unique(sorted_existing, sorted_candidates) -> int:
    import numpy as np
    count = 0
    prior = None
    for start in range(0, len(sorted_candidates), PAIR_CHUNK_ROWS):
        chunk = sorted_candidates[start:min(start + PAIR_CHUNK_ROWS, len(sorted_candidates))]
        if not len(chunk):
            continue
        unique = np.ones(len(chunk), dtype=bool)
        if len(chunk) > 1:
            unique[1:] = chunk[1:] != chunk[:-1]
        if prior is not None and int(chunk[0]) == prior:
            unique[0] = False
        count += _count_cross_duplicates(sorted_existing, chunk[unique])
        prior = int(chunk[-1])
    return count


def _scan_train_files(root: Path, manifest: dict, pair_path: Path, old_subset_root: Optional[Path] = None):
    import numpy as np
    import pyarrow.parquet as pq
    root = _secure_root(root)
    if old_subset_root is not None:
        old_subset_root = _secure_root(old_subset_root)
    users, items = set(), set()
    exposures, likes, feedback = Counter(), Counter(), Counter()
    rows_count = 0
    old_users, old_items = set(), set()
    week_summaries = []
    train_records = [x for x in manifest["files"] if x["path"].startswith("subsamples/ur0.01/train/")]
    paths = [_resolve_under_root(root, record["path"], want_file=True) for record in train_records]
    for record, path in zip(train_records, paths):
        _check_schema(path, EXPECTED_INTERACTION_SCHEMA, record["path"], root=root)
    pair_capacity = sum(pq.ParquetFile(_path_under_root(root, path)).metadata.num_rows for path in paths)
    pairs = _new_pair_memmap(pair_path, pair_capacity, root=root)
    offset = 0
    for week, path in enumerate(paths):
        week_users, week_items = set(), set()
        week_rows = week_likes = week_dislikes = 0
        for batch in _batches(path, root=root):
            batch_users = batch.column(batch.schema.get_field_index("user_id")).to_numpy(zero_copy_only=False)
            batch_items = batch.column(batch.schema.get_field_index("item_id")).to_numpy(zero_copy_only=False)
            batch_likes = batch.column(batch.schema.get_field_index("like")).to_numpy(zero_copy_only=False)
            unique_users = np.unique(batch_users)
            unique_items = np.unique(batch_items)
            users.update(map(int, unique_users.tolist()))
            items.update(map(int, unique_items.tolist()))
            week_users.update(map(int, unique_users.tolist()))
            week_items.update(map(int, unique_items.tolist()))
            _update_counts(exposures, batch_users)
            _update_counts(likes, batch_users[batch_likes])
            keys = _pair_keys(batch)
            pairs[offset:offset + len(keys)] = keys
            offset += len(keys)
            rows_count += batch.num_rows
            week_rows += batch.num_rows
            for name in FEEDBACK:
                positives = int(batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False).sum())
                feedback[name + "_true"] += positives
                feedback[name + "_false"] += batch.num_rows - positives
                if name == "like":
                    week_likes += positives
                elif name == "dislike":
                    week_dislikes += positives
            _check_memory_budget()
        week_summaries.append({"week": week, "rows": week_rows, "users": len(week_users),
                               "items": len(week_items), "like_events": week_likes,
                               "dislike_events": week_dislikes})
    if offset != pair_capacity:
        raise ValueError("ALERT Parquet row count changed during train scan")
    pair_duplicates = _count_sorted_duplicates(pairs)
    del pairs
    if old_subset_root is not None and pair_duplicates == 0:
        old_names = [f"subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet" for week in range(25)]
        for name in old_names:
            path = _resolve_under_root(old_subset_root, name, want_file=True)
            _check_schema(path, EXPECTED_INTERACTION_SCHEMA, name, root=old_subset_root)
            for batch in _batches(path, columns=["user_id", "item_id"], root=old_subset_root):
                old_users.update(map(int, np.unique(batch.column(0).to_numpy(zero_copy_only=False)).tolist()))
                old_items.update(map(int, np.unique(batch.column(1).to_numpy(zero_copy_only=False)).tolist()))
                _check_memory_budget()
    summary = {"rows": rows_count, "users": len(users), "items": len(items), "like_users": len(likes),
               "like_events": sum(likes.values()), "dislike_events": feedback["dislike_true"],
               "feedback_counts": dict(feedback), "exposure_length": _summary(list(exposures.values())),
               "like_history_length": _summary(list(likes.values())),
               "weeks": week_summaries,
               "duplicate_user_item_excess_rows_train": pair_duplicates,
               "exact_duplicate_rows_train": 0 if pair_duplicates == 0 else "NOT_COMPUTED_AFTER_PAIR_ALERT",
               "old_subset_user_overlap": len(users & old_users) if old_subset_root is not None else None,
               "old_subset_user_overlap_fraction_of_new_users": len(users & old_users) / len(users) if users and old_subset_root is not None else None,
               "old_subset_item_count": len(old_items) if old_subset_root is not None else None,
               "old_subset_item_overlap": len(items & old_items) if old_subset_root is not None else None,
               "_users": users, "_items": items, "_like_counts": likes,
               "_exposure_counts": exposures, "_old_items": old_items,
               "_pair_path": pair_path, "_pair_rows": pair_capacity,
               "_pair_array": pair_path,
               "_pair_duplicates": pair_duplicates, "_memory_peak_bytes": _check_memory_budget()}
    if pair_duplicates:
        summary["status"] = "ALERT"
        summary["alert"] = "official data card states no repeated user-item pair exposures"
    else:
        summary["status"] = "ok"
    return summary


def _metadata_rank(root: Path, catalog: set[int], old_catalog: Optional[set[int]]):
    import numpy as np
    import pyarrow.parquet as pq
    root = _secure_root(root)
    path = _resolve_under_root(root, "metadata/items_metadata.parquet", want_file=True)
    parquet = pq.ParquetFile(path)
    fields = {f.name: str(f.type) for f in parquet.schema_arrow}
    if fields != EXPECTED_METADATA_SCHEMA:
        raise ValueError("ALERT items_metadata rank schema mismatch")
    catalog_ids = np.asarray(sorted(catalog), dtype=np.uint32)
    old_ids = np.asarray(sorted(old_catalog), dtype=np.uint32) if old_catalog is not None else None
    matched_ids = np.empty(len(catalog_ids), dtype=np.uint32)
    ranks = np.empty(len(catalog_ids), dtype=np.uint32)
    old_matched_ids = np.empty(len(old_ids), dtype=np.uint32) if old_ids is not None else None
    old_ranks = np.empty(len(old_ids), dtype=np.uint32) if old_ids is not None else None
    matched_count = old_count = 0
    metadata_rows = 0
    null_values = 0
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS, use_threads=False):
        metadata_rows += batch.num_rows
        ids = batch.column(batch.schema.get_field_index("item_id")).to_numpy(zero_copy_only=False)
        rank = batch.column(batch.schema.get_field_index("train_interactions_rank")).to_numpy(zero_copy_only=False)
        selected = _membership(ids, catalog_ids)
        old_selected = _membership(ids, old_ids) if old_ids is not None else np.zeros(len(ids), dtype=bool)
        for column in batch.columns:
            null_count = int(column.null_count)
            if null_count:
                raise ValueError("ALERT null value anywhere in items_metadata")
            null_values += null_count
        count = int(selected.sum())
        if matched_count + count > len(matched_ids):
            raise ValueError("ALERT duplicate item metadata ID")
        matched_ids[matched_count:matched_count + count] = ids[selected]
        ranks[matched_count:matched_count + count] = rank[selected]
        matched_count += count
        if old_ids is not None:
            old_batch_count = int(old_selected.sum())
            if old_count + old_batch_count > len(old_ranks):
                raise ValueError("ALERT duplicate old-subset item metadata ID")
            old_matched_ids[old_count:old_count + old_batch_count] = ids[old_selected]
            old_ranks[old_count:old_count + old_batch_count] = rank[old_selected]
            old_count += old_batch_count
        _check_memory_budget()
    matched_ids = matched_ids[:matched_count]
    matched_ids.sort()
    if len(matched_ids) > 1 and bool(np.any(matched_ids[1:] == matched_ids[:-1])):
        raise ValueError("ALERT duplicate item metadata ID")
    ranks = ranks[:matched_count]
    ranks.sort()
    if old_ranks is not None:
        old_matched_ids = old_matched_ids[:old_count]
        old_matched_ids.sort()
        if len(old_matched_ids) > 1 and bool(np.any(old_matched_ids[1:] == old_matched_ids[:-1])):
            raise ValueError("ALERT duplicate old-subset item metadata ID")
        old_ranks = old_ranks[:old_count]
        old_ranks.sort()
    def stats(values):
        if not len(values):
            return {"count": 0, "min": None, "p50": None, "p90": None, "max": None}
        import math
        return {"count": len(values), "min": int(values[0]),
                "p50": int(values[max(0, math.ceil(.5 * len(values)) - 1)]),
                "p90": int(values[max(0, math.ceil(.9 * len(values)) - 1)]), "max": int(values[-1])}
    return {"metadata_rows": metadata_rows, "catalog_items": len(catalog),
            "catalog_metadata_matched": matched_count, "catalog_metadata_missing": len(catalog) - matched_count,
            "metadata_null_values": null_values,
            "ur0.01_train_catalog_rank": stats(ranks),
            "old_subset_train_catalog_rank": stats(old_ranks) if old_ranks is not None else None,
            "interpretation": "descriptive rank distribution only; not evidence of the ir sampling mechanism"}


def _metadata_users(root: Path, user_ids: set[int]) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    root = _secure_root(root)
    path = _resolve_under_root(root, "metadata/users_metadata.parquet", want_file=True)
    parquet = pq.ParquetFile(path)
    fields = {f.name: str(f.type) for f in parquet.schema_arrow}
    if fields != EXPECTED_USER_METADATA_SCHEMA:
        raise ValueError("ALERT users_metadata schema mismatch")
    expected_ids = np.asarray(sorted(user_ids), dtype=np.uint32)
    matched_ids = np.empty(len(expected_ids), dtype=np.uint32)
    matched_count = 0
    rows = 0
    null_values = 0
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS, use_threads=False):
        rows += batch.num_rows
        ids = batch.column(batch.schema.get_field_index("user_id")).to_numpy(zero_copy_only=False)
        selected = _membership(ids, expected_ids)
        for column in batch.columns:
            null_count = int(column.null_count)
            if null_count:
                raise ValueError("ALERT null value anywhere in users_metadata")
            null_values += null_count
        count = int(selected.sum())
        if matched_count + count > len(matched_ids):
            raise ValueError("ALERT duplicate train user metadata ID")
        matched_ids[matched_count:matched_count + count] = ids[selected]
        matched_count += count
        _check_memory_budget()
    matched_ids = matched_ids[:matched_count]
    matched_ids.sort()
    if len(matched_ids) > 1 and bool(np.any(matched_ids[1:] == matched_ids[:-1])):
        raise ValueError("ALERT duplicate train user metadata ID")
    return {"metadata_rows": rows, "train_users": len(user_ids),
            "train_user_metadata_matched": matched_count,
            "train_user_metadata_missing": len(user_ids) - matched_count,
            "metadata_null_values": null_values}


def _inspect_embedding_archive(root: Path) -> dict:
    """Read only NPZ/NPY headers; never materialize vector arrays."""
    import zipfile
    import numpy as np
    root = _secure_root(root)
    path = _resolve_under_root(root, "metadata/item_embeddings.npz", want_file=True)
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"item_id.npy", "embedding.npy"}:
            raise ValueError("ALERT embedding archive members differ from frozen schema")
        for name in archive.namelist():
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
            shapes[name[:-4]] = {"shape": list(shape), "dtype": str(dtype), "fortran_order": bool(fortran_order)}
    ids, vectors = shapes["item_id"], shapes["embedding"]
    if len(ids["shape"]) != 1 or ids["dtype"] != "uint32" or len(vectors["shape"]) != 2 or vectors["shape"][1] != 64 or vectors["dtype"] != "float16" or ids["shape"][0] != vectors["shape"][0]:
        raise ValueError("ALERT embedding header schema mismatch")
    return {"item_count": ids["shape"][0], "embedding_dimensions": 64,
            "item_id_dtype": ids["dtype"], "embedding_dtype": vectors["dtype"],
            "values_read": False}


def _new_pair_memmap(path: Path, rows: int, root: Path):
    import numpy as np
    if rows < 1:
        raise ValueError("ALERT pair scratch path exists or dataset is empty")
    root = _secure_root(root)
    path = _new_file_path_under_root(root, Path(path))
    usage = shutil.disk_usage(root)
    required = rows * np.dtype(np.uint64).itemsize
    if usage.free - required < usage.total * 0.20:
        raise ValueError("ALERT pair scratch would consume the 20% disk reserve")
    with path.open("xb") as stream:
        stream.truncate(required)
    return np.memmap(path, dtype=np.uint64, mode="r+", shape=(rows,))


def _write_alert(root: Path, reason: str, details: dict) -> dict:
    root = _secure_root(root)
    receipt = {"schema_version": 1, "task_id": "ur001-preparation", "status": "ALERT",
               "revision": FIXED_REVISION, "reason": reason, "details": details,
               "script_sha256": _script_sha256()}
    out = _new_file_under_root(root, "structure-audit-alert.json")
    with out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return receipt


def audit_preflight(root: Path, old_subset_root: Path) -> dict:
    """Read-only engineering gate for a downloaded input root; emits no audit statistics."""
    root = _secure_root(Path(root).expanduser())
    old_subset_root = _secure_root(Path(old_subset_root).expanduser())
    new_resolved = root
    old_resolved = old_subset_root
    if new_resolved == old_resolved or new_resolved in old_resolved.parents or old_resolved in new_resolved.parents:
        raise ValueError("ALERT downloaded and old-subset roots overlap")

    file_manifest = load_file_manifest()
    manifest_path = _resolve_under_root(root, "download-manifest.json", want_file=True)
    downloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_download_fields = {"schema_version", "task_id", "status", "dataset", "revision",
                               "source_manifest_sha256", "script_sha256", "files", "total_bytes",
                               "preflight", "source_metadata"}
    if type(downloaded) is not dict or set(downloaded) != required_download_fields:
        raise ValueError("ALERT download manifest fields mismatch")
    if type(downloaded["schema_version"]) is not int or downloaded["schema_version"] != 1 or downloaded["task_id"] != "ur001-preparation":
        raise ValueError("ALERT download manifest schema/task mismatch")
    if downloaded.get("dataset") != DATASET or downloaded.get("revision") != FIXED_REVISION:
        raise ValueError("ALERT dataset or fixed revision mismatch")
    if downloaded.get("status") != "downloaded" or downloaded.get("script_sha256") != _script_sha256():
        raise ValueError("ALERT download status or script SHA mismatch")
    if downloaded.get("source_manifest_sha256") != hashlib.sha256(_path_under_root(ROOT, DEFAULT_FILE_MANIFEST).read_bytes()).hexdigest():
        raise ValueError("ALERT fixed source-manifest SHA mismatch")
    records = downloaded.get("files")
    if type(records) is not list or len(records) != 29:
        raise ValueError("ALERT download manifest must contain exactly 29 files")
    if type(downloaded.get("total_bytes")) is not int or downloaded["total_bytes"] != file_manifest["total_bytes"]:
        raise ValueError("ALERT download manifest total-bytes mismatch")
    preflight_receipt = downloaded.get("preflight")
    if type(preflight_receipt) is not dict or preflight_receipt.get("status") != "ready" or preflight_receipt.get("revision") != FIXED_REVISION or preflight_receipt.get("source_metadata_checked") is not True:
        raise ValueError("ALERT download manifest preflight receipt mismatch")
    if any(type(item) is not dict or set(item) != {"path", "size", "sha256"} for item in records):
        raise ValueError("ALERT download file record fields mismatch")
    by_path = {item.get("path"): item for item in records if type(item) is dict}
    expected_by_path = {item["path"]: item for item in file_manifest["files"]}
    if set(by_path) != set(expected_by_path):
        raise ValueError("ALERT download manifest path inventory mismatch")
    source_metadata = downloaded.get("source_metadata")
    if type(source_metadata) is not list or len(source_metadata) != 29:
        raise ValueError("ALERT download manifest source metadata missing")
    if any(type(item) is not dict or set(item) != {"path", "content_length", "lfs_oid_sha256"} for item in source_metadata):
        raise ValueError("ALERT HF tree source metadata fields mismatch")
    source_by_path = {item.get("path"): item for item in source_metadata if type(item) is dict}
    if set(source_by_path) != set(expected_by_path):
        raise ValueError("ALERT HF tree source metadata path inventory mismatch")
    for name, expected in expected_by_path.items():
        record = by_path[name]
        source = source_by_path[name]
        if record.get("size") != expected["size"] or record.get("sha256") != expected["sha256"]:
            raise ValueError(f"ALERT download manifest pin mismatch: {name}")
        if source.get("content_length") != expected["size"] or source.get("lfs_oid_sha256") != expected["sha256"]:
            raise ValueError(f"ALERT HF tree LFS pin mismatch: {name}")
        path = _resolve_under_root(root, name, want_file=True)
        if path.stat().st_size != expected["size"] or _sha256(path, root=root) != expected["sha256"]:
            raise ValueError(f"ALERT downloaded file SHA/size mismatch: {name}")

    train_names = [f"subsamples/ur0.01/train/week_{week:02d}.parquet" for week in range(25)]
    for name in train_names:
        _check_schema(root / name, EXPECTED_INTERACTION_SCHEMA, name, root=root)
        _check_parquet_nulls(root, name, "interactions")
    _check_schema(root / "subsamples/ur0.01/validation/week_25.parquet",
                  EXPECTED_INTERACTION_SCHEMA, "subsamples/ur0.01/validation/week_25.parquet", root=root)
    _check_parquet_nulls(root, "subsamples/ur0.01/validation/week_25.parquet", "interactions")
    _check_schema(root / "metadata/items_metadata.parquet", EXPECTED_METADATA_SCHEMA,
                  "metadata/items_metadata.parquet", root=root)
    _check_schema(root / "metadata/users_metadata.parquet", EXPECTED_USER_METADATA_SCHEMA,
                  "metadata/users_metadata.parquet", root=root)
    _check_parquet_nulls(root, "metadata/items_metadata.parquet", "items_metadata")
    _check_parquet_nulls(root, "metadata/users_metadata.parquet", "users_metadata")
    _inspect_embedding_archive(root)

    old_train_names = [f"subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet" for week in range(25)]
    for name in old_train_names:
        path = _resolve_under_root(old_subset_root, name, want_file=True)
        _check_schema(path, EXPECTED_INTERACTION_SCHEMA, name, root=old_subset_root)
    return {"action": "audit-preflight", "status": "preflight_ok",
            "revision": FIXED_REVISION, "engineering": False,
            "scientific_input_writes": False, "manifest_and_hashes_verified": True,
            "interaction_and_metadata_schemas_verified": True,
            "all_interaction_rows_null_checked": True,
            "all_metadata_rows_null_checked": True,
            "embedding_header_verified_without_loading_values": True,
            "old_subset_train_weeks_verified": True,
            "pinned_readme_sha256": "3b4c3fb2cb2853526cec6145e8e5f20fabb9efbd33ae024f9f96948237f3538e",
            "ordering_evidence": "pinned README week order plus physical Parquet row order; timestamps absent, chronology not independently verified",
            "statistics_generated": False, "aggregation_generated": False,
            "audit_results_written": False, "scratch_created": False,
            "test_paths_read": False}


def audit_dataset(root: Path, old_subset_root: Optional[Path] = None) -> dict:
    """Read only authorized train/validation/metadata paths and emit aggregate structure."""
    import pyarrow.parquet as pq
    root = _secure_root(Path(root))
    if old_subset_root is not None:
        old_subset_root = _secure_root(old_subset_root)
    for relative in ("audit-work", "structure-audit.json", "structure-audit-alert.json"):
        target = _resolve_under_root(root, relative, must_exist=False)
        if target.exists():
            raise FileExistsError("audit output or scratch path already exists")
    manifest_path = _resolve_under_root(root, "download-manifest.json", want_file=True)
    downloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if downloaded.get("dataset") != DATASET or downloaded.get("revision") != FIXED_REVISION or downloaded.get("status") != "downloaded":
        raise ValueError("download manifest/revision mismatch")
    if downloaded.get("script_sha256") != _script_sha256():
        raise ValueError("downloaded source script identity mismatch")
    if downloaded.get("source_manifest_sha256") != hashlib.sha256(_path_under_root(ROOT, DEFAULT_FILE_MANIFEST).read_bytes()).hexdigest():
        raise ValueError("download source inventory SHA mismatch")
    records = downloaded.get("files")
    expected = load_file_manifest()
    if len(records) != 29 or {x.get("path") for x in records} != {x["path"] for x in expected["files"]}:
        raise ValueError("download manifest inventory mismatch")
    source_metadata = downloaded.get("source_metadata")
    if type(source_metadata) is not list or len(source_metadata) != 29:
        raise ValueError("download manifest source metadata missing")
    source_by_path = {x.get("path"): x for x in source_metadata}
    for item in expected["files"]:
        evidence = source_by_path.get(item["path"])
        if evidence is None or evidence.get("lfs_oid_sha256") != item["sha256"] or evidence.get("content_length") != item["size"]:
            raise ValueError("download manifest LFS evidence mismatch")
    by_path = {x["path"]: x for x in records}
    for item in expected["files"]:
        got = by_path[item["path"]]
        path = _resolve_under_root(root, item["path"], want_file=True)
        if got.get("sha256") != item["sha256"] or got.get("size") != item["size"]:
            raise ValueError("download manifest file identity mismatch")
        if path.stat().st_size != item["size"] or _sha256(path, root=root) != item["sha256"]:
            raise ValueError("downloaded file changed since manifest")
    work = _create_directory_under_root(root, "audit-work", exist_ok=False)
    train = _scan_train_files(root, expected, work / "train-pairs.u64", old_subset_root)
    if train["status"] == "ALERT":
        return _write_alert(root, train["alert"], {k: v for k, v in train.items() if not k.startswith("_")})
    catalog = train.pop("_items"); user_ids = train.pop("_users")
    old_catalog = train.pop("_old_items")
    train_pairs = train.pop("_pair_array")
    train_pair_rows = train.pop("_pair_rows")
    train.pop("_pair_duplicates")
    train.pop("_memory_peak_bytes")
    train.pop("status")
    validation_rec = by_path["subsamples/ur0.01/validation/week_25.parquet"]
    validation_path = _resolve_under_root(root, validation_rec["path"], want_file=True)
    _check_schema(validation_path, EXPECTED_INTERACTION_SCHEMA, validation_rec["path"], root=root)
    import numpy as np
    import pyarrow.parquet as pq
    validation_pair_path = work / "validation-pairs.u64"
    validation_pair_rows = pq.ParquetFile(_path_under_root(root, validation_path)).metadata.num_rows
    validation_pairs = _new_pair_memmap(validation_pair_path, validation_pair_rows, root=root)
    validation_targets = {}
    validation_rows = 0
    validation_users = set()
    validation_items = set()
    validation_likes = validation_dislikes = 0
    all_users = set(user_ids)
    all_items = set(catalog)
    validation_pair_offset = 0
    train_pairs = _path_under_root(root, train_pairs)
    train_pair_array = np.memmap(train_pairs, dtype=np.uint64, mode="r", shape=(train_pair_rows,))
    for batch in _batches(validation_path, root=root):
        user = batch.column(batch.schema.get_field_index("user_id")).to_numpy(zero_copy_only=False)
        item = batch.column(batch.schema.get_field_index("item_id")).to_numpy(zero_copy_only=False)
        like = batch.column(batch.schema.get_field_index("like")).to_numpy(zero_copy_only=False)
        dislike = batch.column(batch.schema.get_field_index("dislike")).to_numpy(zero_copy_only=False)
        all_users.update(map(int, np.unique(user).tolist()))
        all_items.update(map(int, np.unique(item).tolist()))
        keys = _pair_keys(batch)
        validation_pairs[validation_pair_offset:validation_pair_offset + len(keys)] = keys
        validation_pair_offset += len(keys)
        validation_rows += batch.num_rows
        validation_users.update(map(int, np.unique(user).tolist()))
        validation_items.update(map(int, np.unique(item).tolist()))
        validation_likes += int(like.sum()); validation_dislikes += int(dislike.sum())
        liked_users = user[like]
        liked_items = item[like]
        unique_users, first_positions = np.unique(liked_users, return_index=True)
        for target_user, first_position in zip(unique_users.tolist(), first_positions.tolist()):
            # First encountered target across ordered batches is the P3 first-positive target.
            validation_targets.setdefault(int(target_user), int(liked_items[first_position]))
        _check_memory_budget()
    if validation_pair_offset != validation_pair_rows:
        raise ValueError("ALERT Parquet row count changed during validation scan")
    validation_pair_duplicates = _count_sorted_duplicates(validation_pairs)
    cross_pair_duplicates = _count_cross_unique(train_pair_array, validation_pairs)
    del validation_pairs
    del train_pair_array
    pair_duplicates = cross_pair_duplicates + validation_pair_duplicates
    train["duplicate_user_item_excess_rows_train"] = 0
    train["duplicate_user_item_excess_rows_validation"] = validation_pair_duplicates
    train["duplicate_user_item_cross_split_rows"] = cross_pair_duplicates
    train["exact_duplicate_rows_train"] = 0
    train["exact_duplicate_rows_validation"] = 0
    train["exact_duplicate_rows"] = 0
    if pair_duplicates:
        return _write_alert(root, "official data card states no repeated user-item pair exposures",
                            {"duplicate_user_item_excess_rows_train": 0,
                             "duplicate_user_item_excess_rows_validation": validation_pair_duplicates,
                             "duplicate_user_item_cross_split_rows": cross_pair_duplicates,
                             "exact_duplicate_rows": "NOT_COMPUTED_AFTER_PAIR_ALERT"})
    # Counts use exactly the P3 rule: first positive target per validation user, train-only positive history, min_history=1.
    candidates, validation_observed_lengths, validation_like_history_lengths = _candidate_summary(
        validation_targets, train["_like_counts"], train["_exposure_counts"], catalog)
    ranks = _metadata_rank(root, catalog, old_catalog if old_subset_root is not None else None)
    users_metadata = _metadata_users(root, user_ids)
    embedding_header = _inspect_embedding_archive(root)
    scale_check = {"observed_rows_train_plus_validation": train["rows"] + validation_rows,
                   "expected_rows": EXPECTED_SCALE["rows"],
                   "observed_users_train_plus_validation": len(all_users), "expected_users": EXPECTED_SCALE["users"],
                   "observed_items_train_plus_validation": len(all_items), "expected_items": EXPECTED_SCALE["items"]}
    scale_check["matches_data_card"] = (
        scale_check["observed_rows_train_plus_validation"] == scale_check["expected_rows"]
        and scale_check["observed_users_train_plus_validation"] == scale_check["expected_users"]
        and scale_check["observed_items_train_plus_validation"] == scale_check["expected_items"])
    if not scale_check["matches_data_card"]:
        return _write_alert(root, "observed scale conflicts with pinned data-card estimate", scale_check)
    pair_duplicate_audit = {
        "train_excess_rows": train.pop("duplicate_user_item_excess_rows_train"),
        "validation_excess_rows": train.pop("duplicate_user_item_excess_rows_validation"),
        "cross_split_excess_rows": train.pop("duplicate_user_item_cross_split_rows"),
        "exact_duplicate_rows": train.pop("exact_duplicate_rows"),
        "exact_duplicate_method": "zero inferred because no repeated user-item pair exists",
    }
    train.pop("exact_duplicate_rows_train")
    train.pop("exact_duplicate_rows_validation")
    report = {"schema_version": 1, "task_id": "ur001-preparation", "status": "audited",
              "revision": FIXED_REVISION, "script_sha256": _script_sha256(),
              "download_manifest_sha256": _sha256(manifest_path, root=root),
              "train": {k: v for k, v in train.items() if not k.startswith("_")},
              "pair_duplicate_audit": pair_duplicate_audit,
              "validation": {"rows": validation_rows, "users": len(validation_users),
                             "items": len(validation_items), "like_events": validation_likes,
                             "dislike_events": validation_dislikes, "candidate_counts": candidates,
                             "observed_history_length_untruncated": _summary(validation_observed_lengths),
                             "observed_history_length_capped50": _summary([min(50, x) for x in validation_observed_lengths]),
                             "positive_history_length_untruncated": _summary(validation_like_history_lengths),
                             "positive_history_length_capped50": _summary([min(50, x) for x in validation_like_history_lengths])},
              "metadata_rank_evidence": ranks,
              "users_metadata": users_metadata,
              "embedding_header": embedding_header,
              "scale_check": scale_check,
              "resources": {"peak_rss_bytes": _check_memory_budget(), "max_rss_bytes": MAX_RSS_BYTES,
                            "processes": 1, "pyarrow_threads": 1,
                            "pair_scratch_rows": train_pair_rows + validation_pair_rows,
                            "pair_scratch_bytes": (train_pair_rows + validation_pair_rows) * 8},
              "ordering": {"position": "(week, physical_row_ordinal)",
                           "physical_ordinal": "Parquet row-group/batch/row read order; ordinal starts at 0 per file",
                           "pinned_readme_sha256": "3b4c3fb2cb2853526cec6145e8e5f20fabb9efbd33ae024f9f96948237f3538e",
                           "source_statement": "official README states within-week timestamp order",
                           "timestamp": "absent in interaction schema; physical order is an observed proxy, not independently verified chronology"},
              "limits": ["No model metrics or predictions were read or computed.",
                         "Query counts are structural only; no validation target IDs are emitted.",
                         "No test path is in the authorized inventory or read by this entrypoint."]}
    out = _new_file_under_root(root, "structure-audit.json")
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return report


def _dry_receipt(root: Path, manifest: dict) -> dict:
    root = require_new_root(root)
    if not root.parent.is_dir():
        raise ValueError("destination parent must already exist")
    usage = shutil.disk_usage(root.parent)
    paths = [x["path"] for x in manifest["files"]]
    pair_scratch_bytes = 407_854_360 * 8
    return {"action": "dry", "status": "ready", "revision": manifest["revision"],
            "file_count": len(paths), "total_bytes": manifest["total_bytes"],
            "pair_scratch_bytes": pair_scratch_bytes,
            "destination_root_exists": root.exists(), "free_bytes": usage.free,
            "manifest_valid": True, "test_paths": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry", help="validate frozen inventory and emit a non-writing receipt")
    dry.add_argument("--root", type=Path, required=True)
    pre = sub.add_parser("preflight", help="read-only source metadata and capacity preflight")
    pre.add_argument("--root", type=Path, required=True)
    pre.add_argument("--old-subset-root", type=Path, required=True)
    download = sub.add_parser("download", help="download only the frozen 29-file ur0.01 inventory")
    download.add_argument("--root", type=Path, required=True)
    download.add_argument("--old-subset-root", type=Path, required=True)
    audit = sub.add_parser("audit", help="read-only structural audit of a completed download")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--old-subset-root", type=Path)
    audit_pre = sub.add_parser("audit-preflight", help="read-only manifest/schema gate without audit statistics")
    audit_pre.add_argument("--root", type=Path, required=True)
    audit_pre.add_argument("--old-subset-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = load_file_manifest()
        if args.command == "dry":
            receipt = _dry_receipt(args.root, manifest)
        elif args.command == "preflight":
            receipt = preflight(args.root, args.old_subset_root, manifest)
        elif args.command == "download":
            receipt = download_all(args.root, args.old_subset_root, manifest)
        elif args.command == "audit-preflight":
            receipt = audit_preflight(args.root, args.old_subset_root)
        else:
            receipt = audit_dataset(args.root, args.old_subset_root)
    except (ValueError, OSError) as exc:
        if args.command == "audit-preflight":
            print(json.dumps({"action": args.command, "status": "ALERT", "engineering": False,
                              "scientific_input_writes": False, "reason": str(exc)},
                             sort_keys=True, separators=(",", ":")))
            return 2
        if args.command == "audit":
            reason = str(exc) if str(exc).startswith("ALERT") else f"ALERT audit integrity/input failure: {exc}"
            receipt = {"action": args.command, "status": "ALERT", "reason": reason}
            try:
                safe_root = _secure_root(args.root)
                alert_path = _resolve_under_root(safe_root, "structure-audit-alert.json", must_exist=False)
                if not alert_path.exists():
                    receipt = _write_alert(safe_root, reason, {})
            except (ValueError, OSError):
                pass
            print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
            return 2
        if isinstance(exc, ValueError) and str(exc).startswith("ALERT"):
            print(json.dumps({"action": args.command, "status": "ALERT", "reason": str(exc)},
                             sort_keys=True, separators=(",", ":")))
            return 2
        raise
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 2 if receipt.get("status") == "ALERT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
