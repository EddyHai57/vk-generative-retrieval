"""FEAT-01 HIST_V2_SIDE_V1: per-history-event side fields aligned with a pinned HIST_V2_OBS prepare.

For every event already in a history-v2 sidecar (strictly before its target), store that event's
own fields: six feedback bits, timespent, and the item's duration. Nothing is read from the target
row or any later row: the event set is fixed by the pinned ``history_positions``. Validation queries
use train weeks only (week <= 24), exactly as history-v2. The history-v2 files are not modified;
this is a new sidecar bound to the history-v2 manifest by sha256.
"""
import json
from pathlib import Path

from .history_v2 import LENGTHS, _read_pinned, _write_json, load_prepared

SIDE_VERSION = 'HIST_V2_SIDE_V1'
FEEDBACK_BITS = ('like', 'share', 'bookmark', 'open_comments', 'click_on_author', 'dislike')
FIELDS = ('feedback_mask', 'timespent', 'duration')


def _encode(row, duration):
    if any(type(row.get(name)) is not bool for name in FEEDBACK_BITS):
        raise ValueError('feedback field must be bool')
    timespent = row.get('timespent')
    if type(timespent) is not int or not 1 <= timespent <= 255:
        raise ValueError('timespent outside uint8 [1,255]')
    if type(duration) is not int or not 1 <= duration <= 255:
        raise ValueError('duration missing or outside [1,255]')
    return [sum(1 << b for b, name in enumerate(FEEDBACK_BITS) if row[name]), timespent, duration]


def build_side(partitions, history, durations):
    """``history`` is a loaded L100 history-v2 prepare (train + validation rows with positions)."""
    wanted = {}
    for split in ('train', 'validation'):
        for row in history[split]:
            for item, position in zip(row['history'], row['history_positions']):
                key = tuple(position)
                if wanted.setdefault(key, item) != item:
                    raise ValueError('history positions disagree on item')
    found = {}; previous = -1
    for week, events in partitions:
        if type(week) is not int or not previous < week <= 24:
            raise ValueError('train ordering conflict / non-train week')
        previous = week
        for ordinal, row in enumerate(events):
            item = wanted.get((week, ordinal))
            if item is None:
                continue
            if row.get('item_id') != item:
                raise ValueError('side/history item mismatch')
            found[(week, ordinal)] = _encode(row, durations.get(item))
    if len(found) != len(wanted):
        raise ValueError('history position absent from train partitions')
    return {split: [[found[tuple(p)] for p in row['history_positions']] for row in history[split]] for split in ('train', 'validation')}


def write_side(output, history_manifest, history_sha256, partitions, durations, binding):
    history = load_prepared(history_manifest, history_sha256, 100)
    side = build_side(partitions, history, durations)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    artifacts = {f'side-{split}-l100.json': _write_json(output / f'side-{split}-l100.json', side[split]) for split in ('train', 'validation')}
    manifest = {'schema_version': 1, 'kind': 'history_side_prepare', 'side_version': SIDE_VERSION, 'fields': list(FIELDS),
                'feedback_bits': list(FEEDBACK_BITS), 'history_manifest_sha256': history_sha256, 'source_length': 100,
                'definition': "each history event's own row fields (feedback bits, timespent) + item duration; event set = pinned history-v2 positions (strictly before target, train weeks only); shorter lengths are suffixes",
                'binding': binding, 'artifacts': artifacts, 'training_examples': len(side['train']), 'validation_queries': len(side['validation']),
                'test_gate': 'closed_hash_only'}
    return {'manifest': manifest, 'manifest_sha256': _write_json(output / 'manifest.json', manifest)}


def load_side(path, digest, history_sha256, length, history):
    """Load the side for one history length; every row must align 1:1 with ``history`` (same pinned prepare)."""
    if type(length) is not int or length not in LENGTHS:
        raise ValueError('registered history length required')
    path = Path(path); value = _read_pinned(path, digest)
    if value.get('kind') != 'history_side_prepare' or value.get('side_version') != SIDE_VERSION or value.get('fields') != list(FIELDS) \
            or value.get('feedback_bits') != list(FEEDBACK_BITS) or value.get('test_gate') != 'closed_hash_only':
        raise ValueError('history side manifest identity')
    if value.get('history_manifest_sha256') != history_sha256 or history.get('manifest_sha256') != history_sha256:
        raise ValueError('side not bound to this history-v2 prepare')
    result = {'manifest': value, 'manifest_sha256': digest, 'length': length}
    for split in ('train', 'validation'):
        rows = _read_pinned(path.parent / f'side-{split}-l100.json', value['artifacts'][f'side-{split}-l100.json'])
        if len(rows) != len(history[split]):
            raise ValueError('side denominator drift')
        cut = [r[-length:] for r in rows]
        if any(len(s) != len(h['history']) for s, h in zip(cut, history[split])):
            raise ValueError('side/history length mismatch')
        result[split] = cut
    return result


def load_durations(path, catalog):
    """Item duration (seconds, uint8) for every catalog item from items_metadata.parquet; missing/0 fails."""
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=['item_id', 'duration'])
    wanted = set(catalog); durations = {}
    for item, duration in zip(table.column('item_id').to_pylist(), table.column('duration').to_pylist()):
        if item in wanted:
            if item in durations:
                raise ValueError('duplicate item metadata')
            durations[item] = duration
    if set(durations) != wanted or any(type(d) is not int or not 1 <= d <= 255 for d in durations.values()):
        raise ValueError('catalog duration coverage')
    return durations


def side_digest(side):
    import hashlib
    return hashlib.sha256(json.dumps(side, separators=(',', ':')).encode()).hexdigest()
