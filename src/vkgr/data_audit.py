"""P3 aggregate audits; no model or retrieval evaluation."""
from collections import Counter

FEEDBACK = ("like", "dislike", "share", "bookmark", "click_on_author", "open_comments")


def audit_rows(partitions):
    """Consume (week, ordered row iterable); publish aggregates, never user IDs."""
    users, items, positive_items = set(), set(), set()
    lengths = Counter()
    exposures, watch, cofeedback = Counter(), Counter(), Counter()
    feedback = {name: Counter() for name in FEEDBACK}
    pairs, records = set(), set()
    import hashlib
    duplicates = exact = count = 0
    weeks = []
    previous_week = -1
    for week, rows in partitions:
        if type(week) is not int or week <= previous_week:
            raise ValueError("weeks must be strictly increasing nonnegative integers")
        previous_week = week
        week_rows = week_like = week_new_items = 0
        week_users, week_positive_users = set(), set()
        for row in rows:
            user, item = row["user_id"], row["item_id"]
            if any(type(v) is not int or not 0 <= v < 2**32 for v in (user, item)):
                raise ValueError("IDs must be uint32 integers")
            if any(row.get(name) is not None and type(row[name]) is not bool for name in FEEDBACK):
                raise ValueError("feedback must be boolean or null")
            week_new_items += item not in items
            users.add(user); items.add(item); week_users.add(user)
            exposures[user] += 1
            if row.get("timespent") is not None:
                watch[row["timespent"]] += 1
            count += 1; week_rows += 1
            pair = (user, item)
            duplicates += pair in pairs
            pairs.add(pair)
            record = hashlib.sha256(canonical_bytes([week, row])).digest()
            exact += record in records
            records.add(record)
            for name in FEEDBACK:
                value = row.get(name)
                feedback[name]["null" if value is None else "true" if value is True else "false" if value is False else "invalid"] += 1
            if row.get("like") is True:
                lengths[user] += 1; positive_items.add(item)
                cofeedback.update(name for name in FEEDBACK if name != "like" and row.get(name) is True)
                week_like += 1; week_positive_users.add(user)
        weeks.append({"week": week, "rows": week_rows, "users": len(week_users), "like_events": week_like, "like_users": len(week_positive_users), "new_observed_items": week_new_items})
    histogram = Counter(lengths.values())
    histogram[0] = len(users) - len(lengths)
    return {"rows": count, "users": len(users), "items": len(items),
            "like_users": len(lengths), "like_items": len(positive_items),
            "exposure_length_histogram": {str(k): v for k, v in sorted(Counter(exposures.values()).items())},
            "timespent_histogram": {str(k): v for k, v in sorted(watch.items())},
            "like_cofeedback": dict(cofeedback),
            "feedback": {k: dict(v) for k, v in feedback.items()},
            "positive_length_histogram": {str(k): v for k, v in sorted(histogram.items()) if v},
            "users_ge_2_positive": sum(v >= 2 for v in lengths.values()),
            "duplicate_user_item": duplicates, "exact_duplicate_rows": exact, "weeks": weeks}


def _positive_events(partitions):
    previous_week = -1
    for week, rows in partitions:
        if type(week) is not int or week <= previous_week:
            raise ValueError("weeks must be strictly increasing nonnegative integers")
        previous_week = week
        for ordinal, row in enumerate(rows):
            if any(type(row[k]) is not int or not 0 <= row[k] < 2**32 for k in ("user_id", "item_id")):
                raise ValueError("IDs must be uint32 integers")
            if row.get("like") is not None and type(row["like"]) is not bool:
                raise ValueError("like must be boolean or null")
            if row.get("like") is True:
                yield row["user_id"], row["item_id"], [week, ordinal]


def _history_limits(policy):
    minimum, maximum = policy["min_history"], policy["max_history"]
    if type(minimum) is not int or type(maximum) is not int or not 1 <= minimum <= maximum:
        raise ValueError("history limits require 1 <= minimum <= maximum")
    return minimum, maximum


def build_queries(history_partitions, target_partition, policy, catalog):
    """One first-positive target per user; targets do not extend evaluation history."""
    minimum, maximum = _history_limits(policy)
    history_partitions = list(history_partitions)
    target_week, target_rows = target_partition
    if history_partitions and target_week <= history_partitions[-1][0]:
        raise ValueError("all history weeks must precede target week")
    histories = {}
    for user, item, position in _positive_events(history_partitions):
        histories.setdefault(user, []).append((item, position))
        histories[user] = histories[user][-maximum:]
    targets = {}
    for user, item, position in _positive_events([(target_week, target_rows)]):
        targets.setdefault(user, (item, position))
    queries = []
    for user, (item, position) in sorted(targets.items()):
        history = histories.get(user, [])
        if len(history) >= minimum:
            queries.append({"user_id": user, "history": [x[0] for x in history],
                            "history_positions": [x[1] for x in history],
                            "target": item, "target_position": position, "oov": item not in catalog})
    return {"queries": queries, "target_positive_users": len(targets),
            "filtered_no_history": len(targets) - len(queries)}


def training_examples(partitions, policy):
    """Every train-positive prefix only; never consumes validation/test."""
    minimum, maximum = _history_limits(policy)
    histories, examples = {}, []
    for user, item, position in _positive_events(partitions):
        history = histories.setdefault(user, [])
        if len(history) >= minimum:
            examples.append({"user_id": user, "history": [x[0] for x in history],
                             "history_positions": [x[1] for x in history],
                             "target": item, "target_position": position})
        history.append((item, position))
        histories[user] = history[-maximum:]
    return examples


def canonical_bytes(value):
    import json
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def approved_policy():
    return {"version": "p3-like-v1", "min_history": 1, "max_history": 50,
                "label": "like", "candidate_catalog": "train_observed", "seen_item_masking": False,
                "target": "first_positive_in_target_week", "validation_history": "train_only",
                "test_history": "train_only", "tie_break": "item_id_ascending", "duplicate_policy": "preserve",
                "train_weeks": list(range(25)), "validation_week": 25, "test_week": 26}


def _validate_policy_evidence(policy, evidence):
    _history_limits(policy)
    expected = approved_policy()
    if set(policy) != set(expected) or any(policy.get(k) != v or type(policy.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("policy does not satisfy complete approved P3 semantics")
    if any(type(week) is not int for week in policy["train_weeks"]):
        raise ValueError("train weeks must be exact integers, never booleans")
    if type(evidence.get("users_ge_2_positive")) is not int or evidence["users_ge_2_positive"] < 1000:
        raise ValueError("like support below preregistered engineering gate")
    import re
    for key, length in (("source_revision", 40), ("inventory_sha256", 64), ("readme_sha256", 64),
                        ("train_audit_sha256", 64), ("catalog_sha256", 64)):
        if not isinstance(evidence.get(key), str) or not re.fullmatch("[0-9a-f]{%d}" % length, evidence[key]):
            raise ValueError("missing or invalid provenance hash")
    if not evidence.get("train_files") or evidence.get("ordering") != "official_assertion_and_physical_row_ordinal":
        raise ValueError("train files and explicit ordering evidence required")
    files = evidence["train_files"]
    expected_names = [f"subsamples/ur0.01_ir0.01/train/week_{w:02}.parquet" for w in range(25)]
    if not isinstance(files, list) or [x.get("path") for x in files] != expected_names:
        raise ValueError("freeze requires exactly train weeks 00 through 24")
    for record in files:
        if type(record.get("size_bytes")) is not int or record["size_bytes"] < 0 or not isinstance(record.get("sha256"), str) or not re.fullmatch("[0-9a-f]{64}", record["sha256"]):
            raise ValueError("invalid train file evidence")


def freeze_policy(path, policy, train_evidence):
    """Write a new immutable-by-hash policy before any holdout read."""
    from datetime import datetime, timezone
    from pathlib import Path
    _validate_policy_evidence(policy, train_evidence)
    validate_producer(train_evidence.get("producer"))
    if train_evidence.get("training_policy_sha256") != sha256_bytes(canonical_bytes(policy)):
        raise ValueError("training/evaluation policy mismatch")
    manifest = {"schema_version": 1, "task_id": "P3", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "policy": policy, "policy_sha256": sha256_bytes(canonical_bytes(policy)), "train_evidence": train_evidence}
    payload = canonical_bytes(manifest)
    with Path(path).open("xb") as stream:
        stream.write(payload)
    return manifest, sha256_bytes(payload)


def validate_frozen_manifest(path, expected_sha256):
    import json
    from pathlib import Path
    payload = Path(path).read_bytes()
    if sha256_bytes(payload) != expected_sha256:
        raise ValueError("frozen manifest hash mismatch")
    manifest = json.loads(payload)
    if manifest.get("schema_version") != 1 or manifest.get("task_id") != "P3":
        raise ValueError("unsupported P3 manifest")
    _validate_policy_evidence(manifest["policy"], manifest["train_evidence"])
    if sha256_bytes(canonical_bytes(manifest["policy"])) != manifest["policy_sha256"]:
        raise ValueError("policy hash mismatch")
    return manifest


def parquet_rows(path):
    """Preserve parquet row-group/batch/row order; never invent timestamps."""
    import pyarrow.parquet as pq
    expected = {"user_id": "uint32", "item_id": "uint32", "place": "uint8", "platform": "uint8",
                "agent": "uint8", "timespent": "uint8", **{k: "bool" for k in FEEDBACK}}
    reader = pq.ParquetFile(path)
    schema = {field.name: str(field.type) for field in reader.schema_arrow}
    if schema != expected:
        raise ValueError("interaction schema differs from frozen official schema")
    for batch in reader.iter_batches(batch_size=65536, use_threads=False):
        for row in batch.to_pylist():
            if any(row[k] is None for k in expected):
                raise ValueError("null interaction requires audit stop")
            yield row


def audit_features(root, catalog):
    """Audit ID joins and finite 64D content; no fabricated missing features."""
    from pathlib import Path
    import numpy as np
    import pyarrow.parquet as pq
    root = Path(root)
    catalog_array = np.array(sorted(catalog), dtype=np.uint32)
    metadata_ids = set()
    duration_zero = duration_missing = 0
    duration_histogram = Counter()
    reader = pq.ParquetFile(root / "metadata/items_metadata.parquet")
    fields = {f.name: str(f.type) for f in reader.schema_arrow}
    if fields.get("item_id") != "uint32" or fields.get("duration") != "uint8":
        raise ValueError("item metadata schema drift")
    for batch in reader.iter_batches(batch_size=262144, columns=["item_id", "duration"], use_threads=False):
        ids = batch.column(0).to_numpy(zero_copy_only=False)
        mask = np.isin(ids, catalog_array)
        selected_ids = ids[mask]
        if len(set(map(int, selected_ids))) != len(selected_ids) or metadata_ids.intersection(map(int, selected_ids)):
            raise ValueError("duplicate catalog metadata ID")
        metadata_ids.update(map(int, selected_ids))
        duration = np.array(batch.column(1).to_pylist(), dtype=object)[mask]
        duration_missing += sum(v is None for v in duration)
        duration_zero += sum(v == 0 for v in duration if v is not None)
        duration_histogram.update(int(v) for v in duration if v is not None)
    with np.load(root / "metadata/item_embeddings.npz", allow_pickle=False) as data:
        ids = data["item_id"]
        embeddings = data["embedding"]
        if ids.dtype != np.uint32 or embeddings.dtype != np.float16 or embeddings.shape != (len(ids), 64):
            raise ValueError("embedding schema drift")
        mask = np.isin(ids, catalog_array)
        selected_ids = ids[mask]
        if len(np.unique(selected_ids)) != len(selected_ids):
            raise ValueError("duplicate catalog embedding ID")
        selected = embeddings[mask].astype(np.float32)
        nonfinite = int(np.count_nonzero(~np.isfinite(selected).all(axis=1)))
        zero = int(np.count_nonzero(np.linalg.norm(selected, axis=1) == 0))
        matched = len(selected_ids)
    return {"catalog_items": len(catalog), "metadata_matched": len(metadata_ids),
            "metadata_missing": len(catalog) - len(metadata_ids), "duration_null": duration_missing,
            "duration_zero": duration_zero, "duration_histogram": {str(k): v for k, v in sorted(duration_histogram.items())},
            "embedding_matched": matched, "embedding_missing": len(catalog) - matched,
            "embedding_nonfinite": nonfinite, "embedding_zero_norm": zero,
            "embedding_dtype": "float16", "embedding_dimensions": 64,
            "embedding_join": "item_id"}


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _check_inventory_files(root, inventory, names):
    from pathlib import Path
    records = {x["path"]: x for x in inventory["files"]}
    verified = []
    for name in names:
        record = records.get(name)
        if record is None:
            raise ValueError("required path missing from source inventory")
        path = Path(root) / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size_bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError("raw file checksum/size mismatch")
        verified.append({k: record[k] for k in ("path", "size_bytes", "sha256")})
    return verified


def _write_json(path, value):
    from pathlib import Path
    with Path(path).open("xb") as stream:
        stream.write(canonical_bytes(value))


def producer_identity():
    """Actual code/lock/Git/dependencies; CPython 3.11 patch versions may differ."""
    from importlib.metadata import version
    from pathlib import Path
    import platform
    import sys
    from .manifest import source_fingerprint, git_context
    commit, dirty = git_context()
    root = Path(__file__).resolve().parents[2]
    dependencies = {name: version(name) for name in ("PyYAML", "numpy", "pyarrow")}
    if dependencies != {"PyYAML": "6.0.2", "numpy": "1.26.4", "pyarrow": "19.0.1"} or sys.version_info[:2] != (3, 11) or platform.python_implementation() != "CPython":
        raise ValueError("unapproved producer environment")
    return {"git_commit": commit, "git_dirty": dirty, "source_fingerprint": source_fingerprint(),
            "audit_module_sha256": file_sha256(__file__), "lock_sha256": file_sha256(root / "uv.lock"),
            "dependencies": dependencies, "python_version": platform.python_version(),
            "python_major_minor": [3, 11], "python_implementation": "CPython"}


def validate_producer(producer):
    current = producer_identity()
    if not isinstance(producer, dict) or set(producer) != set(current):
        raise ValueError("missing/old producer identity")
    for key in current:
        if key not in ("python_version", "git_dirty") and (producer[key] != current[key] or type(producer[key]) is not type(current[key])):
            raise ValueError("producer source/lock/environment mismatch")
    if type(producer["git_dirty"]) is not bool or not isinstance(producer["python_version"], str) or producer["python_version"].split(".")[:2] != ["3", "11"] or not producer["python_version"].split(".")[-1].isdigit():
        raise ValueError("invalid producer Python/Git provenance")
    if any(type(v) is not int for v in producer["python_major_minor"]):
        raise ValueError("invalid Python compatibility tuple")
    return current


def run_train(raw_root, inventory_path, output):
    """Only train/feature files are opened; all output paths are new and server-only."""
    import hashlib
    import json
    import resource
    import time
    from datetime import datetime, timezone
    from pathlib import Path
    raw_root, output = Path(raw_root), Path(output)
    if output.exists():
        raise FileExistsError("output already exists")
    started = time.monotonic()
    producer = producer_identity()
    training_policy = approved_policy()
    inventory = json.loads(Path(inventory_path).read_bytes())
    names = [f"subsamples/ur0.01_ir0.01/train/week_{week:02}.parquet" for week in range(25)]
    train_files = _check_inventory_files(raw_root, inventory, names)
    feature_files = _check_inventory_files(raw_root, inventory, ["metadata/items_metadata.parquet", "metadata/item_embeddings.npz"])
    catalog, first_seen, all_users = set(), {}, set()
    row_digest = hashlib.sha256()
    def partitions(capture):
        for week, name in enumerate(names):
            def rows(week=week, name=name):
                for row in parquet_rows(raw_root / name):
                    if capture:
                        item = row["item_id"]
                        catalog.add(item); first_seen.setdefault(item, week); all_users.add(row["user_id"])
                    row_digest.update(canonical_bytes([week, row]))
                    yield row
            yield week, rows()
    audit = audit_rows(partitions(True))
    first_digest = row_digest.hexdigest()
    row_digest = hashlib.sha256()
    examples = training_examples(partitions(False), training_policy)
    if row_digest.hexdigest() != first_digest:
        raise ValueError("physical row-order reread mismatch")
    features = audit_features(raw_root, catalog)
    if features["metadata_missing"] or features["embedding_missing"] or features["embedding_nonfinite"]:
        raise ValueError("catalog feature missing/nonfinite stop gate")
    validate_producer(producer)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "catalog.json", sorted(catalog))
    _write_json(output / "first-seen.json", {str(k): v for k, v in sorted(first_seen.items())})
    _write_json(output / "train-users.json", sorted(all_users))
    _write_json(output / "training-examples.json", examples)
    result = {"task_id": "P3", "utc": datetime.now(timezone.utc).isoformat(),
              "producer": producer, "policy": training_policy, "policy_sha256": sha256_bytes(canonical_bytes(training_policy)),
              "source_revision": inventory["revision"], "inventory_sha256": file_sha256(inventory_path),
              "train_files": train_files, "feature_files": feature_files,
              "schema": {"user_id": "uint32", "item_id": "uint32", "place": "uint8", "platform": "uint8", "agent": "uint8", "timespent": "uint8", **{k: "bool" for k in FEEDBACK}},
              "audit": audit, "features": features, "training_examples": len(examples),
              "catalog_items": len(catalog), "catalog_sha256": file_sha256(output / "catalog.json"),
              "first_seen_sha256": file_sha256(output / "first-seen.json"),
              "train_users_sha256": file_sha256(output / "train-users.json"),
              "training_examples_sha256": file_sha256(output / "training-examples.json"),
              "physical_row_sequence_sha256": first_digest, "reread_sequence_match": True,
              "elapsed_seconds": time.monotonic() - started,
              "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "rss_unit": "KiB_on_Linux_bytes_on_macOS"}
    _write_json(output / "train-audit.json", result)
    return result


def run_holdout(raw_root, inventory_path, manifest_path, manifest_sha256, output):
    # This gate intentionally precedes any holdout inventory/schema/content read.
    manifest = validate_frozen_manifest(manifest_path, manifest_sha256)
    validate_producer(manifest["train_evidence"].get("producer"))
    if manifest["train_evidence"].get("training_policy_sha256") != manifest["policy_sha256"]:
        raise ValueError("training/evaluation policy mismatch")
    import json
    import resource
    import time
    from datetime import datetime, timezone
    from pathlib import Path
    raw_root, output = Path(raw_root), Path(output)
    if output.exists():
        raise FileExistsError("output already exists")
    evidence, policy = manifest["train_evidence"], manifest["policy"]
    if file_sha256(inventory_path) != evidence["inventory_sha256"]:
        raise ValueError("inventory changed since freeze")
    inventory = json.loads(Path(inventory_path).read_bytes())
    if inventory["revision"] != evidence["source_revision"]:
        raise ValueError("source revision changed since freeze")
    train_files = _check_inventory_files(raw_root, inventory, [x["path"] for x in evidence["train_files"]])
    if train_files != evidence["train_files"]:
        raise ValueError("train file evidence mismatch")
    train_output = Path(manifest_path).parent
    if file_sha256(train_output / "train-audit.json") != evidence["train_audit_sha256"] or file_sha256(train_output / "catalog.json") != evidence["catalog_sha256"]:
        raise ValueError("train artifact changed since freeze")
    train_audit = json.loads((train_output / "train-audit.json").read_bytes())
    if file_sha256(train_output / "train-users.json") != train_audit["train_users_sha256"]:
        raise ValueError("train user artifact hash mismatch")
    train_users = set(json.loads((train_output / "train-users.json").read_bytes()))
    catalog = set(json.loads((train_output / "catalog.json").read_bytes()))
    started = time.monotonic()
    results, queries_by_split, observed = {}, {}, set(catalog)
    for split, week in (("validation", policy["validation_week"]), ("test", policy["test_week"])):
        name = f"subsamples/ur0.01_ir0.01/{split}/week_{week:02}.parquet"
        verified = _check_inventory_files(raw_root, inventory, [name])
        audit = audit_rows([(week, parquet_rows(raw_root / name))])
        history_partitions = [(w, parquet_rows(raw_root / rec["path"])) for w, rec in zip(policy["train_weeks"], train_files)]
        target_rows = list(parquet_rows(raw_root / name))
        target_positive_users = {row["user_id"] for row in target_rows if row["like"] is True}
        holdout_items = {row["item_id"] for row in target_rows}
        built = build_queries(history_partitions, (week, iter(target_rows)), policy, catalog)
        queries = built.pop("queries")
        queries_by_split[split] = queries
        results[split] = {"week": week, "audit": audit, **built,
                          "eligible_queries": len(queries),
                          "eligible_oov_queries": sum(x["oov"] for x in queries),
                          "eligible_train_observed_queries": sum(not x["oov"] for x in queries),
                          "train_users_without_positive_target": len(train_users - target_positive_users),
                          "positive_target_users_unseen_in_train": len(target_positive_users - train_users),
                          "observed_items_not_in_train": len(holdout_items - catalog),
                          "first_observed_items_relative_to_previous_weeks": len(holdout_items - observed),
                          "holdout_file": verified[0]}
        observed.update(holdout_items)
    output.mkdir(parents=True, exist_ok=False)
    for split, queries in queries_by_split.items():
        _write_json(output / f"{split}-queries.json", queries)
        results[split]["query_sha256"] = file_sha256(output / f"{split}-queries.json")
    result = {"task_id": "P3", "utc": datetime.now(timezone.utc).isoformat(),
              "frozen_manifest_sha256": manifest_sha256, "policy_sha256": manifest["policy_sha256"],
              "inventory_sha256": evidence["inventory_sha256"], **results,
              "elapsed_seconds": time.monotonic() - started,
              "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "rss_unit": "KiB_on_Linux_bytes_on_macOS"}
    _write_json(output / "holdout-audit.json", result)
    return result


def main(argv=None):
    import argparse
    import json
    import sys
    from pathlib import Path
    import yaml
    parser = argparse.ArgumentParser(description="P3 train-only audit, freeze, then holdout statistics")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "holdout"):
        sub = commands.add_parser(name)
        sub.add_argument("--raw-root", required=True)
        sub.add_argument("--inventory", required=True)
        sub.add_argument("--output", required=True)
        if name == "holdout":
            sub.add_argument("--manifest", required=True)
            sub.add_argument("--manifest-sha256", required=True)
    sub = commands.add_parser("freeze")
    sub.add_argument("--train-output", required=True)
    sub.add_argument("--readme", required=True)
    sub.add_argument("--policy", required=True)
    sub.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = run_train(args.raw_root, args.inventory, args.output)
            print(canonical_bytes({"train_audit_sha256": file_sha256(Path(args.output) / "train-audit.json"), "users_ge_2_positive": result["audit"]["users_ge_2_positive"]}).decode(), end="")
        elif args.command == "freeze":
            directory = Path(args.train_output)
            audit_path = directory / "train-audit.json"
            audit = json.loads(audit_path.read_bytes())
            policy = yaml.safe_load(Path(args.policy).read_text())
            readme = Path(args.readme).read_text()
            if "rows within each file are in increasing timestamp order" not in readme or "maintain_order='left'" not in readme:
                raise ValueError("official ordering assertion missing")
            evidence = {"source_revision": audit["source_revision"], "inventory_sha256": audit["inventory_sha256"],
                        "readme_sha256": file_sha256(args.readme), "train_files": audit["train_files"],
                        "train_audit_sha256": file_sha256(audit_path), "catalog_sha256": audit["catalog_sha256"],
                        "users_ge_2_positive": audit["audit"]["users_ge_2_positive"],
                        "ordering": "official_assertion_and_physical_row_ordinal",
                        "producer": audit.get("producer"), "training_policy_sha256": audit.get("policy_sha256"),
                        "policy_config_sha256": file_sha256(args.policy)}
            _validate_policy_evidence(audit.get("policy", {}), evidence)
            if not audit["reread_sequence_match"] or file_sha256(directory / "catalog.json") != audit["catalog_sha256"]:
                raise ValueError("train sequence/catalog evidence mismatch")
            manifest, sha = freeze_policy(args.output, policy, evidence)
            print(canonical_bytes({"frozen_manifest_sha256": sha, "policy_sha256": manifest["policy_sha256"]}).decode(), end="")
        else:
            result = run_holdout(args.raw_root, args.inventory, args.manifest, args.manifest_sha256, args.output)
            print(canonical_bytes({"holdout_audit_sha256": file_sha256(Path(args.output) / "holdout-audit.json"), "frozen_manifest_sha256": result["frozen_manifest_sha256"]}).decode(), end="")
    except (ValueError, OSError, KeyError, TypeError):
        print("ERROR P3 input, evidence, or output gate failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
