"""Read-only P watch-positive train statistics; never opens validation inputs."""
import argparse
import hashlib
import json
from pathlib import Path

from vkgr.p_watch_positive import count_like_train_examples, train_statistics_from_reader, train_supported_items

TRAIN_AUDIT_SHA256 = "3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe"
ANCHOR_SUPPORT_SHA256 = "b080240542ae85166186fb621f7b074f9728f8b0e7d01d2c69ae0d0bff6b53a2"
ANCHOR_MANIFEST_SHA256 = "bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e"
P_INPUTS_SHA256 = "a12b98003aa2fbb58df83bc70bb448e426d01c2a4cc2e6357bf1f95778f23e38"
STATS_RUN_ID = "p-watch-positive-stats-v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def memory_gate():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    total, available = values["MemTotal"], values["MemAvailable"]
    for limit_path, current_path in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                    ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if Path(limit_path).exists():
            raw = Path(limit_path).read_text().strip()
            if raw != "max":
                limit = int(raw)
                if limit < total:
                    total, available = limit, limit - int(Path(current_path).read_text())
    if available / total < 0.2:
        raise ValueError("RAM headroom below 20% stop gate")


def scan(inputs_metadata_path, audit_path, anchor_support_sha256, quantile_method):
    inputs_metadata_path, audit_path = map(Path, (inputs_metadata_path, audit_path))
    if quantile_method != "nearest_rank":
        raise ValueError("formal stats require frozen nearest_rank")
    values = load_p_inputs_metadata(inputs_metadata_path, P_INPUTS_SHA256)
    raw_root = Path(values["raw_root"])
    result = scan_sources(raw_root, audit_path, anchor_support_sha256, quantile_method,
                          TRAIN_AUDIT_SHA256, values["items_metadata_sha256"], ANCHOR_SUPPORT_SHA256)
    result["inputs"]["p_inputs_metadata_sha256"] = P_INPUTS_SHA256
    result["inputs"]["anchor_manifest_sha256"] = ANCHOR_MANIFEST_SHA256
    return result


def load_p_inputs_metadata(path, expected_sha256):
    if sha256(path) != expected_sha256:
        raise ValueError("accepted P input metadata hash mismatch")
    request = json.loads(Path(path).read_text())
    values = request
    if values.get("schema") != "p-feat02-inputs-v1" or not isinstance(values.get("raw_root"), str):
        raise ValueError("accepted P input metadata schema mismatch")
    return values


def scan_sources(raw_root, audit_path, anchor_support_sha256, quantile_method,
                 expected_audit_sha256, expected_metadata_sha256, expected_support_sha256,
                 resource_check=memory_gate):
    """Path scanner behind formal pins; tests pass synthetic pins explicitly here."""
    import pyarrow.parquet as pq

    raw_root, audit_path = Path(raw_root), Path(audit_path)
    if anchor_support_sha256 != expected_support_sha256:
        raise ValueError("anchor support identity pin mismatch")
    if sha256(audit_path) != expected_audit_sha256:
        raise ValueError("registered train audit hash mismatch")
    audit = json.loads(audit_path.read_text())
    catalog_path = audit_path.parent / "catalog.json"
    if sha256(catalog_path) != audit["catalog_sha256"]:
        raise ValueError("catalog binding mismatch")
    catalog = set(json.loads(catalog_path.read_text()))
    duration_path = raw_root / "metadata" / "items_metadata.parquet"
    metadata_sha256 = sha256(duration_path)
    if metadata_sha256 != expected_metadata_sha256:
        raise ValueError("items metadata differs from expected input metadata")
    metadata_reader = pq.ParquetFile(duration_path)
    if {f.name: str(f.type) for f in metadata_reader.schema_arrow} != {"item_id": "uint32", "author_id": "uint32", "duration": "uint8", "train_interactions_rank": "uint32"}:
        raise ValueError("item metadata schema mismatch")
    durations = {}
    for batch in metadata_reader.iter_batches(batch_size=65536, columns=["item_id", "duration"], use_threads=False):
        resource_check()
        for item, duration in zip(batch.column("item_id").to_pylist(), batch.column("duration").to_pylist()):
            if item in catalog:
                if item in durations or type(duration) is not int or not 0 <= duration <= 255:
                    raise ValueError("duplicate or invalid catalog duration")
                durations[item] = duration
    if set(durations) != catalog or sha256(duration_path) != metadata_sha256:
        raise ValueError("catalog duration coverage mismatch")
    verified_train_hashes = {}
    expected_train_hashes = {}
    def partition_reader():
        for week in range(25):
            path = raw_root / "subsamples/ur0.01_ir0.01/train" / ("week_%02d.parquet" % week)
            expected_schema = audit["schema"]
            expected = next((r for r in audit["train_files"] if r["path"].endswith("week_%02d.parquet" % week)), None)
            if expected is None:
                raise ValueError("train partition missing from audit")
            expected_train_hashes[path] = expected["sha256"]
            if path not in verified_train_hashes and sha256(path) != expected["sha256"]:
                raise ValueError("train partition hash mismatch")
            verified_train_hashes[path] = expected["sha256"]
            parquet = pq.ParquetFile(path)
            if {f.name: str(f.type) for f in parquet.schema_arrow} != expected_schema:
                raise ValueError("train partition schema mismatch")
            def rows():
                for batch in parquet.iter_batches(batch_size=65536, columns=["user_id", "item_id", "timespent", "like", "dislike"], use_threads=False):
                    resource_check()
                    for row in batch.to_pylist():
                        if row["item_id"] not in catalog or type(row["timespent"]) is not int or not 0 <= row["timespent"] <= 255 or type(row["like"]) is not bool or type(row["dislike"]) is not bool:
                            raise ValueError("invalid train event")
                        yield row
            yield week, rows()

    def event_reader():
        for week, events in partition_reader():
            for ordinal, row in enumerate(events):
                yield {**row, "week": week, "ordinal": ordinal}

    supported = train_supported_items(event_reader)
    if not supported or not supported <= catalog or _digest(sorted(supported)) != anchor_support_sha256:
        raise ValueError("train-derived support differs from accepted anchor")
    n_like_examples = count_like_train_examples(partition_reader)
    result = train_statistics_from_reader(event_reader, durations, supported, quantile_method, n_like_examples)
    for path, digest in expected_train_hashes.items():
        if sha256(path) != digest:
            raise ValueError("train partition changed during preflight")
    result["inputs"] = {"train_weeks": "00-24", "train_partition_count": 25,
                        "train_partition_sha256": [next(r["sha256"] for r in audit["train_files"] if r["path"].endswith("week_%02d.parquet" % w)) for w in range(25)],
                        "train_audit_sha256": sha256(audit_path),
                        "items_metadata_sha256": metadata_sha256,
                        "supported_items_sha256": _digest(sorted(supported)),
                        "quantile_method": quantile_method}
    result["gate"] = {"week_ge_25_in_auxiliary": False, "N_aux_positive": result["N_aux"] > 0,
                      "invalid_ratio_fraction_at_or_below_1_percent": result["invalid_ratio_fraction"] <= .01,
                      "empty_history_aux_count": result["empty_history_aux_count"],
                      "stop_reasons": result["stop_reasons"]}
    return result


def scan_request(request_path):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    if request.get("schema") != "p-watch-positive-stats-request-v1" or request.get("quantile_method") != "nearest_rank":
        raise ValueError("registered stats request must use the frozen nearest_rank method")
    repo = Path(__file__).resolve().parent.parent
    pins = request.get("source_pins")
    validate_source_pins(pins, repo)
    result = scan(request["inputs_metadata"], request["train_audit"], request["anchor_support_sha256"], request["quantile_method"])
    return report_envelope(result, sha256(request_path), pins)


def report_envelope(result, request_sha256, source_pins):
    return {**result, "schema": "p-watch-positive-train-only-report-v1", "run_id": STATS_RUN_ID,
            "engineering": False, "synthetic": False,
            "inputs": {**result["inputs"], "stats_request_sha256": request_sha256, "source_pins": source_pins}}


def validate_source_pins(pins, repo):
    expected = {"src/vkgr/p_watch_positive.py", "scripts/p_watch_positive_stats.py",
                "src/vkgr/retrieval_data.py", "scripts/p3_label_eda.py",
                "configs/p_watch_positive/p-watch-positive-train-only.json"}
    if type(pins) is not dict or set(pins) != expected:
        raise ValueError("source pin set mismatch")
    for relative, digest in pins.items():
        path = Path(repo) / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("source pin mismatch: " + relative)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(scan_request(args.request), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
