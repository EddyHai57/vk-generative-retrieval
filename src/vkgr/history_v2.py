"""Batch1 HIST_V2_OBS_L{20,50,100}: observed/exposure history sidecars shared by P and G.

Same event definition as P4E ``retrieval_watch.build_watch`` (every train interaction row,
physical order, strict past of the target, duplicates kept), with the window length
parameterised. ``max_len=50`` is value-equal to ``build_watch``. Training examples and
validation queries keep their v1 identity (target, target_position, oov, past_likes); only
``history``/``history_positions`` are replaced. The frozen P4E module is not modified.
"""
from collections import Counter, deque
import hashlib
import json
import math
from pathlib import Path

from .data_audit import file_sha256
from .retrieval_data import catalog_map, validate_example

HISTORY_VERSION = 'HIST_V2_OBS'
LENGTHS = (100, 50, 20)
BUCKETS = (('1-20', 1, 20), ('21-50', 21, 50), ('51-100', 51, 100), ('gt100', 101, None))
L100_GATE_FRACTION = 0.10


def _bucket_counts(lengths):
    counts = {name: 0 for name, _, _ in BUCKETS}
    for n in lengths:
        for name, low, high in BUCKETS:
            if n >= low and (high is None or n <= high):
                counts[name] += 1
                break
    return counts


def _quantile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


def length_summary(lengths):
    """Untruncated observed-history length distribution (D0 of HIST-LEN)."""
    total = len(lengths)
    counts = _bucket_counts(lengths)
    return {'count': total, 'buckets': counts,
            'fractions': {k: (v / total if total else None) for k, v in counts.items()},
            'fraction_gt50': (sum(1 for n in lengths if n > 50) / total) if total else None,
            'p50': _quantile(lengths, .5), 'p90': _quantile(lengths, .9), 'p99': _quantile(lengths, .99),
            'max': max(lengths) if lengths else None}


def l100_gate(validation_summary):
    """Pre-registered D0 gate: cancel L100 when <10% validation queries exceed 50 observed events."""
    fraction = validation_summary['fraction_gt50']
    if type(fraction) not in (int, float) or not 0 <= fraction <= 1:
        raise ValueError('invalid length fraction')
    return {'threshold': L100_GATE_FRACTION, 'fraction_gt50': fraction, 'keep_l100': fraction >= L100_GATE_FRACTION}


def build_observed(partitions, examples, queries, catalog, max_len=50):
    """Row-for-row the P4E build_watch logic, with ``maxlen=max_len`` and length statistics."""
    if type(max_len) is not int or max_len not in LENGTHS:
        raise ValueError('registered history length required')
    idmap = catalog_map(catalog)
    for j, row in enumerate(examples):
        validate_example(row, idmap, j)
    targets = {tuple(e['target_position']): e for e in examples}
    if len(targets) != len(examples):
        raise ValueError('duplicate target position')
    watches = {}; likes = {}; past = {}; outputs = {}; counts = Counter(); user_counts = Counter(); previous = -1
    observed_counts = Counter(); train_full = {}
    for week, events in partitions:
        if type(week) is not int or not previous < week <= 24:
            raise ValueError('train ordering conflict')
        previous = week
        for ordinal, row in enumerate(events):
            user, item, like = row.get('user_id'), row.get('item_id'), row.get('like')
            if type(user) is not int or not 0 <= user < 2**32 or type(item) is not int or item not in idmap or type(like) is not bool:
                raise ValueError('raw interaction/catalog mismatch')
            position = [week, ordinal]
            watch = watches.setdefault(user, deque(maxlen=max_len)); hist = likes.setdefault(user, deque(maxlen=50)); seen = past.setdefault(user, set())
            example = targets.get(tuple(position))
            if example is not None:
                if not like or example['user_id'] != user or example['target'] != item or example['history'] != [x[0] for x in hist] or example['history_positions'] != [x[1] for x in hist] or example['past_likes'] != sorted(seen) or not watch:
                    raise ValueError('immutable example reconciliation mismatch')
                outputs[example['example_id']] = {'example_id': example['example_id'], 'history': [x[0] for x in watch], 'history_positions': [x[1] for x in watch]}
                train_full[example['example_id']] = observed_counts[user]
            elif like and hist:
                raise ValueError('eligible training target missing')
            watch.append((item, position)); observed_counts[user] += 1
            if like:
                hist.append((item, position)); seen.add(item); counts[item] += 1; user_counts[user] += 1
    if len(outputs) != len(examples):
        raise ValueError('target positions absent')
    validation = []; users = set(); validation_full = []
    for q in queries:
        if set(q) != {'user_id', 'history', 'history_positions', 'target', 'target_position', 'oov'} or q['user_id'] in users or q['target_position'][0] != 25 or q['oov'] != (q['target'] not in idmap):
            raise ValueError('validation identity mismatch')
        user = q['user_id']; users.add(user); hist = likes.get(user, []); watch = watches.get(user, [])
        if not watch or q['history'] != [x[0] for x in hist] or q['history_positions'] != [x[1] for x in hist]:
            raise ValueError('train-only validation like history mismatch')
        validation.append({**q, 'history': [x[0] for x in watch], 'history_positions': [x[1] for x in watch]})
        validation_full.append(observed_counts[user])
    positive = sorted(counts.values())
    if not positive:
        raise ValueError('no train positive support')
    thresholds = {name: positive[max(0, math.ceil(len(positive) * fraction) - 1)] for name, fraction in [('median', .5), ('p90', .9)]}
    slices = {'schema_version': 1, 'popularity_thresholds': thresholds, 'item_like_counts': {str(i): counts[i] for i in catalog}, 'user_like_counts': {str(u): user_counts[u] for u in sorted(user_counts)}, 'user_buckets': [1, 5, 20], 'definition': 'train_like_counts_only_no_holdout_boundaries'}
    train_summary = length_summary([train_full[j] for j in range(len(examples))])
    validation_summary = length_summary(validation_full)
    stats = {'definition': 'untruncated count of train interaction rows strictly before target (train) / all train rows (validation)',
             'train': train_summary, 'validation': validation_summary, 'l100_gate': l100_gate(validation_summary)}
    return {'train': [outputs[j] for j in range(len(examples))], 'validation': validation, 'slices': slices, 'length_stats': stats,
            'full_lengths': {'train': [train_full[j] for j in range(len(examples))], 'validation': validation_full}}


def truncate_sidecar(rows, length):
    """Shorter versions are the last ``length`` events of the longer one (L50/L20 from L100)."""
    if type(length) is not int or length not in LENGTHS:
        raise ValueError('registered history length required')
    return [{**r, 'history': r['history'][-length:], 'history_positions': r['history_positions'][-length:]} for r in rows]


def _check_history(history, positions, limit, before, catalog):
    if type(history) is not list or type(positions) is not list or not 1 <= len(history) <= limit or len(history) != len(positions):
        raise ValueError('history length outside version')
    if any(type(i) is not int or i not in catalog for i in history):
        raise ValueError('history item outside catalog')
    for p in positions:
        if type(p) is not list or len(p) != 2 or any(type(v) is not int or v < 0 for v in p) or p[0] > 24:
            raise ValueError('history position outside train weeks')
    if any(a >= b for a, b in zip(positions, positions[1:])) or not positions[-1] < list(before):
        raise ValueError('history not strictly past and increasing')


def overlay_examples(examples, sidecar, length, catalog=None):
    """Replace only history fields; identity of every v1 training example must be unchanged."""
    if len(examples) != len(sidecar):
        raise ValueError('v2 example count mismatch')
    result = []; catalog = _AnyItem() if catalog is None else catalog
    for j, (e, h) in enumerate(zip(examples, sidecar)):
        if not set(h) <= {'example_id', 'target', 'target_position', 'history', 'history_positions'} or h.get('example_id') != j or e['example_id'] != j:
            raise ValueError('v2 example identity mismatch')
        if 'target' in h and (h['target'] != e['target'] or h['target_position'] != e['target_position']):
            raise ValueError('v2 example target drift')
        _check_history(h['history'], h['history_positions'], length, e['target_position'], catalog)
        result.append({**e, 'history': list(h['history']), 'history_positions': [list(p) for p in h['history_positions']]})
    return result


class _AnyItem:
    def __contains__(self, item):
        return True


def overlay_queries(queries, sidecar, length, catalog):
    """Validation queries keep user/target/oov; only the train-only history is replaced."""
    if len(queries) != len(sidecar):
        raise ValueError('v2 query count mismatch')
    result = []
    for original, h in zip(queries, sidecar):
        if set(h) != set(original) or {k: v for k, v in h.items() if k not in ('history', 'history_positions')} != {k: v for k, v in original.items() if k not in ('history', 'history_positions')}:
            raise ValueError('v2 query identity drift')
        _check_history(h['history'], h['history_positions'], length, [25, 0], catalog)
        result.append({**original, 'history': list(h['history']), 'history_positions': [list(p) for p in h['history_positions']]})
    return result


def _write_json(path, value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    with Path(path).open('xb') as stream:
        stream.write(payload)
    return file_sha256(path)


def strict_json(text):
    """Duplicate keys and nonfinite numbers are rejected (same rule as the P4E reader)."""
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def bad(value):
        raise ValueError('nonfinite JSON')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad)


def _read_pinned(path, digest):
    path = Path(path)
    if type(digest) is not str or len(digest) != 64 or path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
        raise ValueError('external hash mismatch')
    return strict_json(path.read_text())


def write_prepared(output, partitions, examples, queries, catalog, binding):
    """One prepare producing L100/L50/L20 sidecars from a single L100 pass + D0 length stats."""
    built = build_observed(partitions, examples, queries, catalog, max_len=100)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    train100 = [{'example_id': r['example_id'], 'target': e['target'], 'target_position': e['target_position'],
                 'history': r['history'], 'history_positions': r['history_positions']} for r, e in zip(built['train'], examples)]
    artifacts = {}
    for length in LENGTHS:
        artifacts[f'train-l{length}.json'] = _write_json(output / f'train-l{length}.json', truncate_sidecar(train100, length))
        artifacts[f'validation-l{length}.json'] = _write_json(output / f'validation-l{length}.json', truncate_sidecar(built['validation'], length))
    artifacts['train-slices.json'] = _write_json(output / 'train-slices.json', built['slices'])
    artifacts['length-stats.json'] = _write_json(output / 'length-stats.json', built['length_stats'])
    digest = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest = {'schema_version': 1, 'kind': 'history_v2_prepare', 'history_version': HISTORY_VERSION, 'lengths': list(LENGTHS),
                'definition': 'all train interaction rows (any like/dislike/timespent), physical (week,row) order, strictly before target, duplicates kept; validation uses train weeks only',
                'binding': binding, 'artifacts': artifacts, 'training_examples': len(train100), 'validation_queries': len(built['validation']),
                'v1_examples_sha256': digest(examples), 'v1_queries_sha256': digest(queries), 'catalog_sha256': digest(list(catalog)),
                'l100_gate': built['length_stats']['l100_gate'], 'test_gate': 'closed_hash_only'}
    return {'manifest': manifest, 'manifest_sha256': _write_json(output / 'manifest.json', manifest)}


def load_prepared(path, digest, length):
    """Read one length version under the external manifest pin; every artifact is hash-checked."""
    if type(length) is not int or length not in LENGTHS:
        raise ValueError('registered history length required')
    path = Path(path); value = _read_pinned(path, digest)
    if value.get('kind') != 'history_v2_prepare' or value.get('history_version') != HISTORY_VERSION or value.get('lengths') != list(LENGTHS) or value.get('test_gate') != 'closed_hash_only':
        raise ValueError('history v2 manifest identity')
    artifacts = value['artifacts']
    train = _read_pinned(path.parent / f'train-l{length}.json', artifacts[f'train-l{length}.json'])
    validation = _read_pinned(path.parent / f'validation-l{length}.json', artifacts[f'validation-l{length}.json'])
    if len(train) != value['training_examples'] or len(validation) != value['validation_queries']:
        raise ValueError('history v2 denominator drift')
    for j, row in enumerate(train):
        if row.get('example_id') != j:
            raise ValueError('history v2 example order')
        _check_history(row['history'], row['history_positions'], length, row['target_position'], _AnyItem())
    for row in validation:
        _check_history(row['history'], row['history_positions'], length, [25, 0], _AnyItem())
    return {'manifest': value, 'manifest_sha256': digest, 'length': length, 'train': train, 'validation': validation,
            'slices': _read_pinned(path.parent / 'train-slices.json', artifacts['train-slices.json']),
            'length_stats': _read_pinned(path.parent / 'length-stats.json', artifacts['length-stats.json'])}
