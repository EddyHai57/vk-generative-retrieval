"""Streaming strict-past M1-compatible rows for approved P watch positives."""
from collections import defaultdict, deque

from .history_side import _encode
from .p_watch_positive import duration_bucket, watch_ratio


def build_auxiliary_rows(partition_reader, supported_items, duration_by_item, thresholds):
    """Yield eligible rows in original (week, physical-row) order with L50 side evidence."""
    if not thresholds or not supported_items:
        raise ValueError("thresholds and supported items required")
    histories = defaultdict(lambda: deque(maxlen=50))
    past_likes = defaultdict(set)
    previous_week = -1
    for week, events in partition_reader():
        if type(week) is not int or not previous_week < week <= 24:
            raise ValueError("train partition order/week conflict")
        previous_week = week
        for ordinal, row in enumerate(events):
            user, item = row.get("user_id"), row.get("item_id")
            if type(user) is not int or type(item) is not int or type(row.get("like")) is not bool or type(row.get("dislike")) is not bool:
                raise ValueError("invalid observed row identity/label")
            duration = duration_by_item.get(item)
            side_value = _encode(row, duration)
            ratio = watch_ratio(row.get("timespent"), duration)
            bucket = duration_bucket(duration)
            if (ratio is not None and bucket in thresholds and not row["like"] and not row["dislike"]
                    and item in supported_items and ratio >= thresholds[bucket]):
                history = list(histories[user])
                yield {"example_id": "watch:%d:%d" % (week, ordinal), "user_id": user,
                       "target": item, "target_position": [week, ordinal],
                       "history": [entry[0] for entry in history],
                       "history_positions": [entry[1] for entry in history],
                       "past_likes": sorted(past_likes[user]), "side": [entry[2] for entry in history]}
            position = [week, ordinal]
            histories[user].append((item, position, side_value))
            if row["like"]:
                past_likes[user].add(item)
