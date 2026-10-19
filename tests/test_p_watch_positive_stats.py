import hashlib
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p_watch_positive_stats as scanner


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _fixture(tmp_path):
    raw = tmp_path / "raw"
    (raw / "metadata").mkdir(parents=True)
    train = raw / "subsamples/ur0.01_ir0.01/train"
    train.mkdir(parents=True)
    metadata = pa.table({
        "item_id": pa.array([1, 2, 3, 4], type=pa.uint32()),
        "author_id": pa.array([1, 1, 2, 3], type=pa.uint32()),
        "duration": pa.array([10, 10, 10, 10], type=pa.uint8()),
        "train_interactions_rank": pa.array([1, 2, 3, 4], type=pa.uint32()),
    })
    metadata_path = raw / "metadata/items_metadata.parquet"
    pq.write_table(metadata, metadata_path)
    schema = {
        "user_id": "uint32", "item_id": "uint32", "place": "uint8", "platform": "uint8", "agent": "uint8",
        "timespent": "uint8", "like": "bool", "dislike": "bool", "share": "bool", "bookmark": "bool",
        "click_on_author": "bool", "open_comments": "bool",
    }
    rows = [
        (1, 1, 10, True, False),    # first like: absent from prefix_examples training rows
        (2, 2, 10, False, False),   # candidate, supported by a later like; empty observed history
        (2, 2, 10, True, False),    # first like for user 2
        (1, 3, 10, True, False),    # second like for user 1: one original training row
        (3, 4, 10, False, True),
    ]
    files = []
    for week in range(25):
        selected = rows if week == 0 else []
        columns = {
            "user_id": pa.array([r[0] for r in selected], type=pa.uint32()),
            "item_id": pa.array([r[1] for r in selected], type=pa.uint32()),
            "place": pa.array([0] * len(selected), type=pa.uint8()),
            "platform": pa.array([0] * len(selected), type=pa.uint8()),
            "agent": pa.array([0] * len(selected), type=pa.uint8()),
            "timespent": pa.array([r[2] for r in selected], type=pa.uint8()),
            "like": pa.array([r[3] for r in selected], type=pa.bool_()),
            "dislike": pa.array([r[4] for r in selected], type=pa.bool_()),
            "share": pa.array([False] * len(selected), type=pa.bool_()),
            "bookmark": pa.array([False] * len(selected), type=pa.bool_()),
            "click_on_author": pa.array([False] * len(selected), type=pa.bool_()),
            "open_comments": pa.array([False] * len(selected), type=pa.bool_()),
        }
        path = train / ("week_%02d.parquet" % week)
        pq.write_table(pa.table(columns), path)
        files.append({"path": "subsamples/ur0.01_ir0.01/train/week_%02d.parquet" % week, "sha256": _sha(path)})
    # This malformed post-train partition proves the scanner never opens week 25.
    pq.write_table(pa.table({"bad": pa.array([1])}), train / "week_25.parquet")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("[1,2,3,4]")
    audit = {"schema": schema, "catalog_sha256": _sha(catalog_path), "train_files": files}
    audit_path = tmp_path / "train-audit.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True, separators=(",", ":")))
    support_sha = _digest([1, 2, 3])
    return raw, audit_path, metadata_path, support_sha, train


def test_synthetic_scanner_reports_ratio_ties_empty_history_and_original_prefix_count(tmp_path):
    raw, audit, metadata, support_sha, _ = _fixture(tmp_path)
    result = scanner.scan_sources(raw, audit, support_sha, "nearest_rank", _sha(audit), _sha(metadata), support_sha, resource_check=lambda: None)
    assert result["N_like_events"] == 3
    assert result["N_like_train_examples"] == 1
    assert result["buckets"]["1-10"]["n_b"] == 5
    assert result["buckets"]["1-10"]["threshold"] == 1.0
    assert result["buckets"]["1-10"]["ratio_screened_count"] == 5
    assert result["buckets"]["1-10"]["supported_aux"] == 1
    assert result["empty_history_aux_count"] == 1
    assert result["stop_reasons"] == []
    assert "requires nonempty history" in result["empty_history_compatibility_note"]
    assert result["buckets"]["11-20"]["n_b"] == 0
    assert result["buckets"]["11-20"]["threshold"] is None


@pytest.mark.parametrize("failure", ["partition", "metadata", "audit", "support"])
def test_synthetic_scanner_rejects_source_identity_mismatch(tmp_path, failure):
    raw, audit, metadata, support_sha, train = _fixture(tmp_path)
    if failure == "partition":
        with (train / "week_00.parquet").open("ab") as stream:
            stream.write(b"tamper")
    expected_audit = "0" * 64 if failure == "audit" else _sha(audit)
    expected_metadata = "0" * 64 if failure == "metadata" else _sha(metadata)
    expected_support = "0" * 64 if failure == "support" else support_sha
    with pytest.raises(ValueError):
        scanner.scan_sources(raw, audit, support_sha, "nearest_rank", expected_audit, expected_metadata, expected_support, resource_check=lambda: None)


def test_synthetic_metadata_and_source_pin_identity_checks(tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"schema": "p-feat02-inputs-v1", "raw_root": str(tmp_path / "raw"), "items_metadata_sha256": "a" * 64}))
    assert scanner.load_p_inputs_metadata(inputs, _sha(inputs))["raw_root"] == str(tmp_path / "raw")
    with pytest.raises(ValueError, match="hash"):
        scanner.load_p_inputs_metadata(inputs, "0" * 64)
    repo = tmp_path / "repo"
    files = ("src/vkgr/p_watch_positive.py", "scripts/p_watch_positive_stats.py", "src/vkgr/retrieval_data.py",
             "scripts/p3_label_eda.py", "configs/p_watch_positive/p-watch-positive-train-only.json")
    pins = {}
    for name in files:
        path = repo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        pins[name] = _sha(path)
    scanner.validate_source_pins(pins, repo)
    pins[files[0]] = "0" * 64
    with pytest.raises(ValueError, match="source pin"):
        scanner.validate_source_pins(pins, repo)


def test_formal_report_envelope_states_run_and_data_identity():
    report = scanner.report_envelope({"inputs": {"train_weeks": "00-24"}}, "a" * 64, {"source.py": "b" * 64})
    assert report["schema"] == "p-watch-positive-train-only-report-v1"
    assert report["run_id"] == "p-watch-positive-stats-v1"
    assert report["engineering"] is False
    assert report["synthetic"] is False
    assert report["inputs"]["stats_request_sha256"] == "a" * 64
