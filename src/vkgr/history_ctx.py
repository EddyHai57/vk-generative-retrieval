"""FEAT-02 HIST_V2_CTX_V1: per-history-event scene fields [place, platform] aligned with a pinned HIST_V2_OBS prepare.

Mirrors ``history_side`` (which is not modified). For every event already in a history-v2 sidecar (strictly
before its target) store that event's own ``place`` and ``platform`` (raw uint8). ``agent`` is never read.
Nothing comes from the target row or any later row: the event set is fixed by the pinned ``history_positions``.
The sidecar is bound to the history-v2 manifest by sha256; L100 is produced, shorter lengths are suffixes.
"""
from pathlib import Path

from .history_v2 import LENGTHS, _read_pinned, _write_json, load_prepared

CTX_VERSION = 'HIST_V2_CTX_V1'
FIELDS = ('place', 'platform')


def _encode(row):
    values = [row.get(name) for name in FIELDS]
    if any(type(v) is not int or not 0 <= v <= 255 for v in values):
        raise ValueError('place/platform outside uint8 [0,255]')
    return values


def build_ctx(partitions, history):
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
                raise ValueError('ctx/history item mismatch')
            found[(week, ordinal)] = _encode(row)
    if len(found) != len(wanted):
        raise ValueError('history position absent from train partitions')
    return {split: [[found[tuple(p)] for p in row['history_positions']] for row in history[split]] for split in ('train', 'validation')}


def write_ctx(output, history_manifest, history_sha256, partitions, binding):
    history = load_prepared(history_manifest, history_sha256, 100)
    ctx = build_ctx(partitions, history)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    artifacts = {f'ctx-{split}-l100.json': _write_json(output / f'ctx-{split}-l100.json', ctx[split]) for split in ('train', 'validation')}
    manifest = {'schema_version': 1, 'kind': 'history_ctx_prepare', 'ctx_version': CTX_VERSION, 'fields': list(FIELDS),
                'history_manifest_sha256': history_sha256, 'source_length': 100,
                'definition': "each history event's own raw place and platform (agent excluded); event set = pinned history-v2 positions (strictly before target, train weeks only); shorter lengths are suffixes",
                'binding': binding, 'artifacts': artifacts, 'training_examples': len(ctx['train']), 'validation_queries': len(ctx['validation']),
                'test_gate': 'closed_hash_only'}
    return {'manifest': manifest, 'manifest_sha256': _write_json(output / 'manifest.json', manifest)}


def load_ctx(path, digest, history_sha256, length, history):
    """Load the ctx for one history length; every row must align 1:1 with ``history`` (same pinned prepare)."""
    if type(length) is not int or length not in LENGTHS:
        raise ValueError('registered history length required')
    path = Path(path); value = _read_pinned(path, digest)
    if value.get('kind') != 'history_ctx_prepare' or value.get('ctx_version') != CTX_VERSION or value.get('fields') != list(FIELDS) \
            or value.get('test_gate') != 'closed_hash_only':
        raise ValueError('history ctx manifest identity')
    if value.get('history_manifest_sha256') != history_sha256 or history.get('manifest_sha256') != history_sha256:
        raise ValueError('ctx not bound to this history-v2 prepare')
    result = {'manifest': value, 'manifest_sha256': digest, 'length': length}
    for split in ('train', 'validation'):
        rows = _read_pinned(path.parent / f'ctx-{split}-l100.json', value['artifacts'][f'ctx-{split}-l100.json'])
        if len(rows) != len(history[split]):
            raise ValueError('ctx denominator drift')
        cut = [r[-length:] for r in rows]
        if any(len(c) != len(h['history']) for c, h in zip(cut, history[split])):
            raise ValueError('ctx/history length mismatch')
        if any(len(pair) != len(FIELDS) or any(type(v) is not int or not 0 <= v <= 255 for v in pair) for c in cut for pair in c):
            raise ValueError('ctx value outside uint8 [0,255]')
        result[split] = cut
    return result

