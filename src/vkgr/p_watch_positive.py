"""Train-only P watch-positive construction primitives.

The percentile method is deliberately mandatory: its scientific definition is
awaiting a recorded decision. No training or evaluation is performed here.
"""
from collections import Counter, defaultdict, deque
import math
import random
import struct

BUCKETS = (("1-10", 1, 10), ("11-20", 11, 20), ("21-40", 21, 40),
           ("41-60", 41, 60), ("61-120", 61, 120), ("121-255", 121, 255))


def duration_bucket(duration):
    if type(duration) is not int or not 1 <= duration <= 255:
        return None
    return next(name for name, low, high in BUCKETS if low <= duration <= high)


def watch_ratio(timespent, duration):
    """Match retrieval_features.ratio_weight's float32 division without clipping."""
    if type(duration) is not int or duration <= 0 or type(timespent) not in (int, float):
        return None
    try:
        numerator = struct.unpack("f", struct.pack("f", float(timespent)))[0]
        denominator = struct.unpack("f", struct.pack("f", float(duration)))[0]
        value = numerator / denominator
        value = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def quantile(values, fraction, method):
    """Return an explicitly selected quantile; method has no default by design."""
    if method not in ("nearest_rank", "linear"):
        raise ValueError("quantile method must be explicitly registered")
    ordered = sorted(float(v) for v in values)
    if not ordered or not 0 <= fraction <= 1 or any(not math.isfinite(v) for v in ordered):
        raise ValueError("invalid quantile input")
    if method == "nearest_rank":
        return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]
    rank = (len(ordered) - 1) * fraction
    lo = math.floor(rank); hi = math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def train_thresholds(events, duration_by_item, method):
    ratios = defaultdict(list)
    for row in events:
        if type(row.get("week")) is not int or not 0 <= row["week"] <= 24:
            raise ValueError("train event week outside 0..24")
        item = row.get("item_id")
        duration = duration_by_item.get(item)
        ratio = watch_ratio(row.get("timespent"), duration)
        bucket = duration_bucket(duration)
        if ratio is not None and bucket is not None:
            ratios[bucket].append(ratio)
    thresholds = {bucket: quantile(values, .8, method) for bucket, values in ratios.items()}
    return thresholds


def train_statistics(events, duration_by_item, supported_items, method):
    """Return preflight counts for a finite event fixture."""
    rows = list(events)
    partitions = defaultdict(list)
    for row in rows:
        partitions[row["week"]].append(row)
    n_like = count_like_train_examples(lambda: ((week, partitions[week]) for week in sorted(partitions)))
    return train_statistics_from_reader(lambda: iter(rows), duration_by_item, supported_items, method, n_like,
                                        require_nonzero_aux=False, stop_invalid_rate=False)


def count_like_train_examples(partition_reader):
    """Use the registered production prefix builder to count the original like rows."""
    from .retrieval_data import prefix_examples
    return sum(1 for _ in prefix_examples(partition_reader()))


def train_supported_items(event_reader):
    """Reconstruct the existing ID support set from train like events only."""
    return {row["item_id"] for row in event_reader() if row["like"]}


def train_statistics_from_reader(event_reader, duration_by_item, supported_items, method, n_like_train_examples,
                                 require_nonzero_aux=True, stop_invalid_rate=True):
    """Two-pass, bounded-memory preflight for a repeatable train event reader."""
    if method != "nearest_rank":
        raise ValueError("formal P watch thresholds require nearest_rank")
    ratios = defaultdict(list); total_events = invalid_ratio_events = 0
    for row in event_reader():
        total_events += 1
        duration = duration_by_item.get(row["item_id"])
        ratio = watch_ratio(row.get("timespent"), duration)
        bucket = duration_bucket(duration)
        if ratio is None or bucket is None:
            invalid_ratio_events += 1
        else:
            ratios[bucket].append(ratio)
    thresholds = {bucket: quantile(values, .8, method) for bucket, values in ratios.items()}
    linear_thresholds = {bucket: quantile(values, .8, "linear") for bucket, values in ratios.items()}
    bucket_events = Counter(); candidates = Counter(); supported_candidates = Counter()
    like_items = set(); auxiliary_items = set(); seen_users = set()
    n_like_events = unsupported = empty_history = screened = 0
    screened_by_bucket = Counter(); after_label = Counter(); bucket_events = Counter(); diff_by_bucket = Counter(); final_diff_by_bucket = Counter()
    for row in event_reader():
        week = row.get("week")
        if type(week) is not int or not 0 <= week <= 24:
            raise ValueError("train event week outside 0..24")
        item = row["item_id"]; duration = duration_by_item.get(item); bucket = duration_bucket(duration)
        user = row["user_id"]
        ratio = watch_ratio(row.get("timespent"), duration)
        if ratio is not None and bucket is not None:
            bucket_events[bucket] += 1
            if ratio >= thresholds[bucket]:
                screened += 1; screened_by_bucket[bucket] += 1
            if (ratio >= thresholds[bucket]) != (ratio >= linear_thresholds[bucket]):
                diff_by_bucket[bucket] += 1
        if row["like"]:
            n_like_events += 1
            like_items.add(item)
        if ratio is not None and bucket is not None and not row["like"] and not row["dislike"] and ratio >= thresholds[bucket]:
            candidates[bucket] += 1; after_label[bucket] += 1
            if item in supported_items:
                supported_candidates[bucket] += 1; auxiliary_items.add(item)
                if user not in seen_users:
                    empty_history += 1
            else:
                unsupported += 1
        if (ratio is not None and bucket is not None and not row["like"] and not row["dislike"] and item in supported_items
                and (ratio >= thresholds[bucket]) != (ratio >= linear_thresholds[bucket])):
            final_diff_by_bucket[bucket] += 1
        seen_users.add(user)
    n_aux = sum(supported_candidates.values())
    invalid_rate = invalid_ratio_events / total_events if total_events else 1.0
    stop_reasons = []
    if stop_invalid_rate and invalid_rate > .01:
        stop_reasons.append("invalid/nonfinite ratio events exceed 1%")
    if require_nonzero_aux and n_aux == 0:
        stop_reasons.append("N_aux=0")
    if type(n_like_train_examples) is not int or n_like_train_examples < 0:
        raise ValueError("invalid original like training-example count")
    return {"N_like": n_like_train_examples, "N_like_train_examples": n_like_train_examples,
            "N_like_events": n_like_events,
            "N_aux_candidates": sum(candidates.values()), "N_aux": n_aux,
            "unsupported_excluded": unsupported, "empty_history_aux_count": empty_history,
            "empty_history_compatibility_note": "count only; original M1 training step requires nonempty history; report count to Kiro if positive",
            "invalid_ratio_events": invalid_ratio_events, "train_observed_events": total_events,
            "invalid_ratio_fraction": invalid_rate, "ratio_screened_all_events": screened,
            "ratio_screened_all_fraction": screened / sum(bucket_events.values()) if bucket_events else None,
            "linear_final_aux_membership_difference_count": sum(final_diff_by_bucket.values()),
            "stop_reasons": stop_reasons,
            "aux_like_item_overlap": len(auxiliary_items & like_items),
            "buckets": {name: {"n_b": bucket_events[name], "threshold": thresholds.get(name),
                               "ratio_screened_count": screened_by_bucket[name],
                               "ratio_screened_fraction": (screened_by_bucket[name] / bucket_events[name]) if bucket_events[name] else None,
                               "after_like_dislike_filters": after_label[name], "unsupported_excluded": candidates[name] - supported_candidates[name],
                               "supported_aux": supported_candidates[name],
                               "linear_threshold": linear_thresholds.get(name),
                               "linear_ratio_screened_membership_difference_count": diff_by_bucket[name],
                               "linear_final_aux_membership_difference_count": final_diff_by_bucket[name]}
                        for name, _, _ in BUCKETS}}


def build_auxiliary(events, supported_items, duration_by_item, thresholds, quantile_method):
    """Build one example per eligible event, with strict-past observed L50 history."""
    if quantile_method != "nearest_rank":
        raise ValueError("formal auxiliary labels require nearest_rank")
    if not thresholds:
        raise ValueError("thresholds unavailable")
    events = list(events)
    previous = None
    for row in events:
        pos = (row.get("week"), row.get("ordinal"))
        if type(pos[0]) is not int or not 0 <= pos[0] <= 24 or type(pos[1]) is not int or pos[1] < 0:
            raise ValueError("invalid train week/ordinal")
        if previous is not None and pos <= previous:
            raise ValueError("events not in strict physical order")
        previous = pos
    histories = defaultdict(lambda: deque(maxlen=50))
    past_likes = defaultdict(set)
    out = []
    for row in events:
        user, item = row["user_id"], row["item_id"]
        duration = duration_by_item.get(item)
        bucket = duration_bucket(duration)
        ratio = watch_ratio(row.get("timespent"), duration)
        if bucket in thresholds and ratio is not None and not row["like"] and not row["dislike"] and item in supported_items:
            if ratio >= thresholds[bucket]:
                out.append({"example_id": "watch:%d:%d" % (row["week"], row["ordinal"]),
                            "user_id": user, "target": item, "target_position": [row["week"], row["ordinal"]],
                            "history": [x[0] for x in histories[user]],
                            "history_positions": [x[1] for x in histories[user]],
                            "past_likes": sorted(past_likes[user])})
        histories[user].append((item, [row["week"], row["ordinal"]]))
        if row["like"]:
            past_likes[user].add(item)
    return out


def epoch_rows(like_rows, auxiliary_rows, epoch, seed_base):
    if type(epoch) is not int or epoch < 0 or type(seed_base) is not int:
        raise ValueError("invalid epoch seed")
    count = min(len(like_rows), len(auxiliary_rows))
    rng = random.Random(seed_base + epoch)
    chosen = rng.sample(list(auxiliary_rows), count)
    mixed = list(like_rows) + chosen
    rng.shuffle(mixed)
    return mixed


def logq_frequency(like_targets, auxiliary_targets):
    """Expected epoch target frequency under equal-weight auxiliary subsampling."""
    if not like_targets or not auxiliary_targets:
        raise ValueError("like and auxiliary targets must both be nonempty")
    like = Counter(like_targets); aux = Counter(auxiliary_targets)
    r = min(1.0, len(like_targets) / len(auxiliary_targets))
    return {item: float(like[item]) + r * aux[item] for item in like.keys() | aux.keys()}


def logq_frequency_table(like_targets, auxiliary_targets):
    counts = logq_frequency(like_targets, auxiliary_targets)
    return {"name": "P_WATCH_EXPECTED_EPOCH_FREQ_V1", "counts": counts,
            "total": float(sum(counts.values()))}
