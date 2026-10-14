"""GTOK-01 tokenizer audit for SIDMAP_V2 mappings (arXiv 2305.05065 style tokenizers and the k-means baseline).

Only the item mapping and the catalog content vectors are read; no label, query, split or history data.
Legality counters reuse the frozen ``SIDCatalog``; its ``balanced_nodes`` / ``unconverged_nodes`` are
vacuous for ``nodes=[]`` and are never interpreted here. ``balanced_children`` recomputes balance from the mapping.
Hard gates are fixed in advance (H: every tokenizer, R: rqvae only, B: bhkm only); report items carry no threshold.
"""
import hashlib
import math

import numpy as np

from .sid import SIDCatalog

CONTENT_V1_FEATURE_SHA256 = '766317a88353dcda241d7579aee6da73b5ad8ea0a627470e2ac8a68c99530c84'
KINDS = ('rqvae', 'bhkm')
BALANCE_TOLERANCE = 1  # sibling subtree sizes may differ by at most this many items
USAGE_NUM_DEN = (4, 5)  # a position passes iff used * 5 >= K * 4 (integer test, no float rounding)
REQUIRED_STEPS = 300000
HISTOGRAM_HEAD = 16
FLOAT_REL_TOL = 1e-9


def feature_digest(content):
    """sha256 of the item rows as '<f4' C-order bytes (the CONTENT_V1 pin definition)."""
    return hashlib.sha256(np.asarray(content, dtype='<f4').tobytes(order='C')).hexdigest()


def usage_ok(used, K):
    numerator, denominator = USAGE_NUM_DEN
    return used * denominator >= K * numerator


def _groups(codes):
    """Rows sorted lexicographically by all columns; returns (order, start index of every distinct row)."""
    order = np.lexsort(tuple(codes[:, j] for j in reversed(range(codes.shape[1]))))
    ordered = codes[order]
    starts = np.flatnonzero(np.r_[True, np.any(ordered[1:] != ordered[:-1], axis=1)])
    return order, starts


def _group_sizes(starts, n):
    return np.diff(np.r_[starts, n])


def _leaf_counts(sids):
    """Items per leaf; a leaf is the set of items sharing the first three codes."""
    _, starts = _groups(np.asarray(sids[:, :3], dtype=np.int64))
    return _group_sizes(starts, len(sids))


def _sse(x, codes):
    """Within-group sum of squared distances to the group mean (centred, float64)."""
    n = len(x)
    if codes.shape[1] == 0:
        order, starts = np.arange(n), np.array([0])
    else:
        order, starts = _groups(codes)
    ordered = x[order]
    sizes = _group_sizes(starts, n)
    means = np.add.reduceat(ordered, starts, axis=0) / sizes[:, None]
    delta = ordered - np.repeat(means, sizes, axis=0)
    return float((delta * delta).sum())


def _total_ss(x):
    """SST: variance of the whole catalog around the global mean."""
    return _sse(x, np.zeros((len(x), 0), dtype=np.int64))


def _explained_variance(content, sids):
    x = np.asarray(content, dtype=np.float64)
    sst = _total_ss(x)
    if not sst > 0:
        raise ValueError('degenerate content: zero total variance')
    codes = np.asarray(sids[:, :3], dtype=np.int64)
    return {str(d): float(1.0 - _sse(x, codes[:, :d]) / sst) for d in (1, 2, 3)}


def _position_report(values, base):
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return {'base': int(base), 'distinct': int(len(counts)), 'utilization': float(len(counts) / base),
            'perplexity': float(np.exp(-(p * np.log(p)).sum()))}


def balanced_children(sids, branching, tolerance=None):
    """Recompute sibling balance from the mapping alone.

    Every node at depth 0, 1, 2 with n items must have min(branching, n) non-empty children
    whose item counts differ by at most ``tolerance``.
    """
    tol = BALANCE_TOLERANCE if tolerance is None else tolerance
    codes = np.asarray(sids[:, :3], dtype=np.int64)
    n = len(codes)
    nodes = violations = spread_max = 0
    for depth in range(3):
        order, starts = _groups(codes[:, :depth + 1])
        counts = _group_sizes(starts, n)
        parents = codes[order][starts][:, :depth]
        first = np.flatnonzero(np.r_[True, np.any(parents[1:] != parents[:-1], axis=1)]) if depth else np.array([0])
        total = np.add.reduceat(counts, first)
        present = _group_sizes(first, len(counts))
        spread = np.maximum.reduceat(counts, first) - np.minimum.reduceat(counts, first)
        bad = (present != np.minimum(branching, total)) | (spread > tol)
        nodes += len(first)
        violations += int(bad.sum())
        spread_max = max(spread_max, int(spread.max()))
    return {'balanced': violations == 0, 'nodes': int(nodes), 'violations': int(violations), 'max_spread': spread_max}


def _legality(catalog):
    try:
        found = catalog.audit()
        return {key: int(found[key]) for key in
                ('full_sid_collisions', 'roundtrip_items', 'reachable_items', 'valid_prefix_deadends')}
    except ValueError:  # SIDCatalog.decode refuses a colliding SID; count what still round-trips
        ok = 0
        for item in catalog.ids:
            try:
                ok += catalog.decode(catalog.encode(int(item))) == int(item)
            except ValueError:
                pass
        # an existing prefix always has an allowed next token, so dead ends cannot occur
        return {'full_sid_collisions': int(len(catalog.ids) - len(np.unique(catalog.keys))),
                'roundtrip_items': ok, 'reachable_items': ok, 'valid_prefix_deadends': 0}


def _suffix_canonical(ids, sids):
    order = np.lexsort((ids, sids[:, 2], sids[:, 1], sids[:, 0]))
    codes = sids[order]
    starts = np.r_[True, np.any(codes[1:, :3] != codes[:-1, :3], axis=1)]
    rows = np.arange(len(codes))
    rank = rows - np.maximum.accumulate(np.where(starts, rows, 0))
    return bool(np.array_equal(codes[:, 3], rank))


def audit_mapping(ids, sids, content, *, branching, kind):
    """Every report item of the GTOK-01 tokenizer audit as a JSON-able dict (lists, str keys, no numpy)."""
    ids, sids, content = np.asarray(ids), np.asarray(sids), np.asarray(content)
    n = len(ids)
    if kind not in KINDS or type(branching) is not int or branching < 2:
        raise ValueError('invalid audit kind/branching')
    if ids.ndim != 1 or n < 1 or not np.issubdtype(ids.dtype, np.integer) or sids.shape != (n, 4) \
            or not np.issubdtype(sids.dtype, np.integer) or content.ndim != 2 or len(content) != n:
        raise ValueError('invalid audit inputs')
    ids64, sids64 = ids.astype(np.int64), sids.astype(np.int64)
    ids_ok = bool(ids64.min() >= 0 and ids64.max() < 2 ** 32 and np.all(ids64[1:] > ids64[:-1]))
    digits_ok = bool(sids64.min() >= 0 and sids64[:, :3].max() < branching and sids64.max() < 2 ** 32)
    canonical = bool(ids_ok and digits_ok and _suffix_canonical(ids64, sids64))
    finite = bool(np.isfinite(content).all())
    if ids_ok and digits_ok:
        catalog = SIDCatalog(ids64.astype(np.uint32), sids64.astype(np.uint32), branching, 3, [])
        legality = _legality(catalog)
    else:
        legality = {'full_sid_collisions': None, 'roundtrip_items': None, 'reachable_items': None,
                    'valid_prefix_deadends': None}
    suffix_vocab = int(sids64[:, 3].max()) + 1
    leaves = np.sort(_leaf_counts(sids64))
    sizes, size_counts = np.unique(leaves, return_counts=True)
    prefix_groups = {str(d): int(len(_groups(sids64[:, :d])[1])) for d in (1, 2, 3)}
    return {
        'kind': kind, 'branching': branching, 'items': int(n), 'content_sha256': feature_digest(content),
        'all_finite': finite, 'ids_ascending': ids_ok, 'digits_in_range': digits_ok, 'suffix_canonical': canonical,
        **legality,
        'positions': [_position_report(sids64[:, d], branching if d < 3 else suffix_vocab) for d in range(4)],
        'leaves': {'count': int(len(leaves)), 'max': int(leaves[-1]), 'median': float(np.median(leaves)),
                   'p95': int(np.percentile(leaves, 95, method='higher')),
                   'p99': int(np.percentile(leaves, 99, method='higher')), 'mean': float(leaves.mean()),
                   'singletons': int((leaves == 1).sum()), 'distinct_sizes': int(len(sizes)),
                   'histogram_head': [[int(s), int(c)] for s, c in zip(sizes[:HISTOGRAM_HEAD], size_counts[:HISTOGRAM_HEAD])]},
        'suffix_vocabulary': suffix_vocab,
        'semantic_collision_groups': int((leaves > 1).sum()),
        'semantic_collision_items': int(leaves[leaves > 1].sum()),
        'items_needing_suffix': int(n - len(leaves)),
        'prefix_groups': prefix_groups,
        'explained_variance': _explained_variance(content, sids64) if finite else None,
        'vocab_size': int(2 + 3 * branching + suffix_vocab),
        'balanced_children': balanced_children(sids64, branching),
    }


def _finite_numbers(value):
    if isinstance(value, dict):
        return all(_finite_numbers(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _check(check_id, ok, value, threshold):
    return {'id': check_id, 'ok': bool(ok), 'value': value, 'threshold': threshold}


def gates(audit, kind, rq=None, *, expected_content_sha256=CONTENT_V1_FEATURE_SHA256, required_steps=REQUIRED_STEPS):
    """Pre-registered hard gates. ``rq`` (rqvae only): status, steps, max_steps, elapsed, wall_cap_seconds, reconstruction."""
    if kind not in KINDS:
        raise ValueError('invalid gate kind')
    n = audit['items']
    collisions, roundtrip, reachable, deadends = (audit[k] for k in (
        'full_sid_collisions', 'roundtrip_items', 'reachable_items', 'valid_prefix_deadends'))
    structure = {k: audit[k] for k in ('ids_ascending', 'digits_in_range', 'suffix_canonical')}
    numbers_ok = audit['all_finite'] and _finite_numbers(audit) and (rq is None or _finite_numbers(rq))
    checks = [
        _check('H1', collisions == 0, collisions, 0),
        _check('H2', roundtrip == n and reachable == n, [roundtrip, reachable], n),
        _check('H3', deadends == 0, deadends, 0),
        _check('H4', all(structure.values()), structure, True),
        _check('H5', audit['content_sha256'] == expected_content_sha256, audit['content_sha256'], expected_content_sha256),
        _check('H6', numbers_ok, bool(numbers_ok), True),
    ]
    if kind == 'rqvae':
        K = audit['branching']
        used = [audit['positions'][d]['distinct'] for d in range(3)]
        checks.append(_check('R1', all(usage_ok(u, K) for u in used), used, [K, list(USAGE_NUM_DEN)]))
        steps = None if rq is None else {k: rq[k] for k in ('status', 'steps', 'max_steps', 'elapsed')}
        steps_ok = rq is not None and rq['status'] == 'complete' and rq['steps'] == required_steps \
            and rq['max_steps'] == required_steps and rq['elapsed'] <= rq['wall_cap_seconds']
        checks.append(_check('R2', steps_ok, steps, {'steps': required_steps,
                                                      'wall_cap_seconds': None if rq is None else rq['wall_cap_seconds']}))
        recon = None if rq is None else rq['reconstruction']
        r2 = None if recon is None or not recon['sst'] > 0 else 1.0 - recon['mse'] / recon['sst']
        checks.append(_check('R3', r2 is not None and r2 > 0, r2, 0.0))
        norms = None if recon is None else list(recon['residual_norms'])
        checks.append(_check('R4', norms is not None and all(a > b for a, b in zip(norms, norms[1:])), norms, 'strictly decreasing'))
    else:
        found = audit['balanced_children']
        checks.append(_check('B1', found['balanced'] is True, found['max_spread'], BALANCE_TOLERANCE))
    return {'passed': all(c['ok'] for c in checks), 'checks': checks}


def audits_equal(left, right, rel=FLOAT_REL_TOL):
    """Floats within relative ``rel``; ints, strings, booleans, None and structure must match exactly."""
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(audits_equal(left[k], right[k], rel) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) \
            and all(audits_equal(a, b, rel) for a, b in zip(left, right))
    if isinstance(left, float):
        return isinstance(right, float) and (left == right or math.isclose(left, right, rel_tol=rel, abs_tol=0.0))
    return type(left) is type(right) and left == right


def _lookup(value, path):
    for part in path.split('.'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


COMPARE_ITEMS = (
    'items', 'vocab_size', 'suffix_vocabulary',
    'positions.0.distinct', 'positions.0.utilization', 'positions.0.perplexity',
    'positions.1.distinct', 'positions.1.utilization', 'positions.1.perplexity',
    'positions.2.distinct', 'positions.2.utilization', 'positions.2.perplexity',
    'positions.3.distinct', 'positions.3.perplexity',
    'leaves.count', 'leaves.max', 'leaves.median', 'leaves.p95', 'leaves.p99', 'leaves.mean', 'leaves.singletons',
    'semantic_collision_groups', 'semantic_collision_items', 'items_needing_suffix',
    'prefix_groups.1', 'prefix_groups.2', 'prefix_groups.3',
    'explained_variance.1', 'explained_variance.2', 'explained_variance.3',
    'full_sid_collisions', 'roundtrip_items', 'valid_prefix_deadends',
    'balanced_children.balanced', 'balanced_children.max_spread',
)


def compare(audits):
    """Side-by-side table of the report items; ``audits`` maps label -> audit (label order is kept)."""
    labels = list(audits)
    return {'labels': labels, 'rows': [{'item': item, 'values': {label: _lookup(audits[label], item) for label in labels}}
                                       for item in COMPARE_ITEMS]}


def _cell(value):
    if isinstance(value, float):
        return format(value, '.6g')
    return str(value)


def render_markdown(table):
    labels = table['labels']
    lines = ['| item | ' + ' | '.join(labels) + ' |', '|---|' + '---|' * len(labels)]
    for row in table['rows']:
        lines.append('| ' + row['item'] + ' | ' + ' | '.join(_cell(row['values'][label]) for label in labels) + ' |')
    return '\n'.join(lines) + '\n'
