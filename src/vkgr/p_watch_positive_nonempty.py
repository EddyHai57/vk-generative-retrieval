"""Frozen-contract removal of watch-positive rows with empty history."""
from collections import Counter
import hashlib
import json

from .p_watch_positive import BUCKETS, duration_bucket


RAW_AUXILIARY_COUNT = 716194
EMPTY_HISTORY_COUNT = 15561
FILTERED_AUXILIARY_COUNT = 700633


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode("utf-8")).hexdigest()


def filter_auxiliary_pool(rows, report, supported_items, duration_by_item, item_heat_bins, *, formal):
    """Filter only rows with empty history and bind pre/post pool diagnostics.

    ``item_heat_bins`` must come from the frozen train-only S0 slices; this
    function deliberately does not estimate heat thresholds from the watch pool.
    """
    rows = list(rows)
    if type(formal) is not bool:
        raise ValueError("formal/synthetic mode must be explicit")
    if type(item_heat_bins) is not dict:
        raise ValueError("frozen S0 item heat-bin mapping required")
    if len(rows) != report.get("N_aux"):
        raise ValueError("raw auxiliary pool differs from original stats report")
    if report.get("empty_history_aux_count") != sum(not row.get("history") for row in rows):
        raise ValueError("raw empty-history count differs from original stats report")
    raw_buckets = Counter({name: 0 for name, _, _ in BUCKETS})
    excluded_buckets = Counter({name: 0 for name, _, _ in BUCKETS})
    excluded_heat = Counter({"unsupported0": 0, "low": 0, "mid": 0, "high": 0})
    raw_items = Counter()
    kept_items = Counter()
    raw_positions = set()
    filtered = []
    for row in rows:
        if row.get("target") not in supported_items:
            raise ValueError("raw auxiliary target outside train support")
        position = tuple(row.get("target_position", ()))
        if position in raw_positions:
            raise ValueError("duplicate raw auxiliary target position")
        raw_positions.add(position)
        item = row["target"]
        bucket = duration_bucket(duration_by_item.get(item))
        if bucket is None:
            raise ValueError("raw auxiliary target has no frozen duration bucket")
        raw_buckets[bucket] += 1
        raw_items[item] += 1
        if row["history"]:
            filtered.append(row)
            kept_items[item] += 1
        else:
            excluded_buckets[bucket] += 1
            if item not in item_heat_bins:
                raise ValueError("frozen S0 heat bin missing for excluded target")
            heat_bin = item_heat_bins[item]
            if heat_bin not in excluded_heat:
                raise ValueError("unknown frozen S0 heat bin")
            excluded_heat[heat_bin] += 1
    for bucket, values in report["buckets"].items():
        if values.get("supported_aux") != raw_buckets.get(bucket, 0):
            raise ValueError("raw auxiliary duration bucket differs from original stats report")
    if formal and set(report["buckets"]) != set(raw_buckets):
        raise ValueError("raw auxiliary duration bucket set differs from original stats report")
    excluded = len(rows) - len(filtered)
    if formal and (len(rows) != RAW_AUXILIARY_COUNT or excluded != EMPTY_HISTORY_COUNT
                   or len(filtered) != FILTERED_AUXILIARY_COUNT):
        raise ValueError("formal post-filter auxiliary count differs from frozen decision")
    return filtered, {
        "raw_N_aux": len(rows), "empty_history_excluded": excluded, "N_aux": len(filtered),
        "raw_pool_sha256": _digest(rows), "filtered_pool_sha256": _digest(filtered),
        "excluded_by_duration_bucket": dict(sorted(excluded_buckets.items())),
        "excluded_by_heat_bin": dict(sorted(excluded_heat.items())),
        "fully_excluded_item_count": sum(item not in kept_items for item in raw_items),
        "raw_bucket_counts_verified": True,
    }
