"""G6-PREP: depth-generic tokenizer audit and gates (new file; ``sid_tok_audit`` is imported, never edited).

``sid_tok_audit.audit_mapping`` / ``balanced_children`` / ``gates`` hard-code three semantic digits. The functions here
repeat the same measurements for ``depth`` semantic digits plus the collision suffix and reuse every depth-free helper of
the frozen module. With ``depth == 3`` the audit dict is byte-identical to the frozen one. Only the item mapping and the
catalog content vectors are read; no label, query, split or history data.

``capacity_table`` restates the capacity rule of ``sid.fit`` (a node of n items splits into min(branching, n) children of
size n // k, the first n % k children one larger). Because that rule does not depend on the content vectors, the exact
shape of a balanced k-means tree is known before any fit; ``PREREG_BHKM_B16D4`` freezes it for N = 120,415.
"""
from collections import Counter

import numpy as np

from . import sid_depth as sd
from . import sid_tok_audit as au
from .sid import SIDCatalog

# Pre-registered shape of SID_BHKM_B16D4_S42 (16-ary, 4 semantic levels, N = 120,415), derived from the capacity rule.
# Keys are audit lookups ("a.b.c" paths into the audit dict); gate B2 compares the recomputed audit to this block.
PREREG_BHKM_B16D4 = {
    'items': 120415,
    'full_sid_collisions': 0,
    'leaves.count': 65536,
    'leaves.max': 2,
    'leaves.singletons': 10657,
    'positions.0.distinct': 16,
    'positions.1.distinct': 16,
    'positions.2.distinct': 16,
    'positions.3.distinct': 16,
    'positions.4.distinct': 2,
    'prefix_groups.1': 16,
    'prefix_groups.2': 256,
    'prefix_groups.3': 4096,
    'prefix_groups.4': 65536,
    'suffix_vocabulary': 2,
    'vocab_size': 68,
    'semantic_collision_groups': 54879,
    'semantic_collision_items': 109758,
    'items_needing_suffix': 54879,
}


def capacity_table(n, branching, depth):
    """Shape of a balanced k-means tree from the capacity rule alone (no content vectors needed)."""
    depth = sd._depth(depth)
    if type(n) is not int or n < 1 or type(branching) is not int or branching < 2:
        raise ValueError('invalid capacity inputs')
    level = Counter({n: 1})
    distinct = []
    for _ in range(depth):
        following = Counter()
        widest = 0
        for size, count in level.items():
            k = min(branching, size)
            quotient, remainder = divmod(size, k)
            widest = max(widest, k)
            if remainder:
                following[quotient + 1] += count * remainder
            following[quotient] += count * (k - remainder)
        distinct.append(widest)
        level = following
    leaves = sum(level.values())
    suffix = max(level)
    return {'items': n, 'leaves': leaves, 'distinct': distinct, 'suffix_vocabulary': suffix,
            'vocab_size': 2 + depth * branching + suffix, 'leaf_sizes': dict(sorted(level.items())),
            'semantic_collision_groups': sum(c for s, c in level.items() if s > 1),
            'semantic_collision_items': sum(s * c for s, c in level.items() if s > 1),
            'singleton_leaves': level.get(1, 0), 'items_needing_suffix': n - leaves}


def _leaf_counts_d(sids, depth):
    _, starts = au._groups(np.asarray(sids[:, :depth], dtype=np.int64))
    return au._group_sizes(starts, len(sids))


def _explained_variance_d(content, sids, depth):
    x = np.asarray(content, dtype=np.float64)
    sst = au._total_ss(x)
    if not sst > 0:
        raise ValueError('degenerate content: zero total variance')
    codes = np.asarray(sids[:, :depth], dtype=np.int64)
    return {str(d): float(1.0 - au._sse(x, codes[:, :d]) / sst) for d in range(1, depth + 1)}


def balanced_children_d(sids, branching, depth, tolerance=None):
    """Sibling balance recomputed from the mapping alone, for every node at semantic levels 0 .. depth - 1."""
    depth = sd._depth(depth)
    tol = au.BALANCE_TOLERANCE if tolerance is None else tolerance
    codes = np.asarray(sids[:, :depth], dtype=np.int64)
    n = len(codes)
    nodes = violations = spread_max = 0
    for level in range(depth):
        order, starts = au._groups(codes[:, :level + 1])
        counts = au._group_sizes(starts, n)
        parents = codes[order][starts][:, :level]
        first = np.flatnonzero(np.r_[True, np.any(parents[1:] != parents[:-1], axis=1)]) if level else np.array([0])
        total = np.add.reduceat(counts, first)
        present = au._group_sizes(first, len(counts))
        spread = np.maximum.reduceat(counts, first) - np.minimum.reduceat(counts, first)
        bad = (present != np.minimum(branching, total)) | (spread > tol)
        nodes += len(first)
        violations += int(bad.sum())
        spread_max = max(spread_max, int(spread.max()))
    return {'balanced': violations == 0, 'nodes': int(nodes), 'violations': int(violations), 'max_spread': spread_max}


def _suffix_canonical_d(ids, sids, depth):
    order = np.lexsort((ids,) + tuple(sids[:, j] for j in reversed(range(depth))))
    codes = sids[order]
    starts = np.r_[True, np.any(codes[1:, :depth] != codes[:-1, :depth], axis=1)]
    rows = np.arange(len(codes))
    rank = rows - np.maximum.accumulate(np.where(starts, rows, 0))
    return bool(np.array_equal(codes[:, depth], rank))


def audit_mapping_d(ids, sids, content, *, branching, kind, depth):
    """Every report item of the tokenizer audit for ``depth`` semantic digits as a JSON-able dict."""
    depth = sd._depth(depth)
    ids, sids, content = np.asarray(ids), np.asarray(sids), np.asarray(content)
    n = len(ids)
    if kind not in au.KINDS or type(branching) is not int or branching < 2:
        raise ValueError('invalid audit kind/branching')
    if ids.ndim != 1 or n < 1 or not np.issubdtype(ids.dtype, np.integer) or sids.shape != (n, depth + 1) \
            or not np.issubdtype(sids.dtype, np.integer) or content.ndim != 2 or len(content) != n:
        raise ValueError('invalid audit inputs')
    ids64, sids64 = ids.astype(np.int64), sids.astype(np.int64)
    ids_ok = bool(ids64.min() >= 0 and ids64.max() < 2 ** 32 and np.all(ids64[1:] > ids64[:-1]))
    digits_ok = bool(sids64.min() >= 0 and sids64[:, :depth].max() < branching and sids64.max() < 2 ** 32)
    canonical = bool(ids_ok and digits_ok and _suffix_canonical_d(ids64, sids64, depth))
    finite = bool(np.isfinite(content).all())
    if ids_ok and digits_ok:
        catalog = SIDCatalog(ids64.astype(np.uint32), sids64.astype(np.uint32), branching, depth, [])
        legality = au._legality(catalog)
    else:
        legality = {'full_sid_collisions': None, 'roundtrip_items': None, 'reachable_items': None,
                    'valid_prefix_deadends': None}
    suffix_vocab = int(sids64[:, depth].max()) + 1
    leaves = np.sort(_leaf_counts_d(sids64, depth))
    sizes, size_counts = np.unique(leaves, return_counts=True)
    prefix_groups = {str(d): int(len(au._groups(sids64[:, :d])[1])) for d in range(1, depth + 1)}
    return {
        'kind': kind, 'branching': branching, 'items': int(n), 'content_sha256': au.feature_digest(content),
        'all_finite': finite, 'ids_ascending': ids_ok, 'digits_in_range': digits_ok, 'suffix_canonical': canonical,
        **legality,
        'positions': [au._position_report(sids64[:, d], branching if d < depth else suffix_vocab) for d in range(depth + 1)],
        'leaves': {'count': int(len(leaves)), 'max': int(leaves[-1]), 'median': float(np.median(leaves)),
                   'p95': int(np.percentile(leaves, 95, method='higher')),
                   'p99': int(np.percentile(leaves, 99, method='higher')), 'mean': float(leaves.mean()),
                   'singletons': int((leaves == 1).sum()), 'distinct_sizes': int(len(sizes)),
                   'histogram_head': [[int(s), int(c)] for s, c in zip(sizes[:au.HISTOGRAM_HEAD], size_counts[:au.HISTOGRAM_HEAD])]},
        'suffix_vocabulary': suffix_vocab,
        'semantic_collision_groups': int((leaves > 1).sum()),
        'semantic_collision_items': int(leaves[leaves > 1].sum()),
        'items_needing_suffix': int(n - len(leaves)),
        'prefix_groups': prefix_groups,
        'explained_variance': _explained_variance_d(content, sids64, depth) if finite else None,
        'vocab_size': int(2 + depth * branching + suffix_vocab),
        'balanced_children': balanced_children_d(sids64, branching, depth),
    }


def gates_d(audit, kind, rq=None, *, depth, expected_content_sha256=au.CONTENT_V1_FEATURE_SHA256,
            required_steps=au.REQUIRED_STEPS, prereg=None):
    """Pre-registered hard gates for ``depth`` semantic digits.

    H1-H6 every tokenizer; R1-R4 rqvae (R1 over all ``depth`` levels, R4 over ``depth + 1`` residual norms);
    B1 bhkm; B2 bhkm only and only when a pre-registered shape ``prereg`` is given.
    """
    depth = sd._depth(depth)
    if kind not in au.KINDS:
        raise ValueError('invalid gate kind')
    if prereg is not None and kind != 'bhkm':
        raise ValueError('B2 applies to balanced k-means only')
    n = audit['items']
    collisions, roundtrip, reachable, deadends = (audit[k] for k in (
        'full_sid_collisions', 'roundtrip_items', 'reachable_items', 'valid_prefix_deadends'))
    structure = {k: audit[k] for k in ('ids_ascending', 'digits_in_range', 'suffix_canonical')}
    numbers_ok = audit['all_finite'] and au._finite_numbers(audit) and (rq is None or au._finite_numbers(rq))
    checks = [
        au._check('H1', collisions == 0, collisions, 0),
        au._check('H2', roundtrip == n and reachable == n, [roundtrip, reachable], n),
        au._check('H3', deadends == 0, deadends, 0),
        au._check('H4', all(structure.values()), structure, True),
        au._check('H5', audit['content_sha256'] == expected_content_sha256, audit['content_sha256'], expected_content_sha256),
        au._check('H6', numbers_ok, bool(numbers_ok), True),
    ]
    if kind == 'rqvae':
        K = audit['branching']
        used = [audit['positions'][d]['distinct'] for d in range(depth)]
        checks.append(au._check('R1', all(au.usage_ok(u, K) for u in used), used, [K, list(au.USAGE_NUM_DEN)]))
        steps = None if rq is None else {k: rq[k] for k in ('status', 'steps', 'max_steps', 'elapsed')}
        steps_ok = rq is not None and rq['status'] == 'complete' and rq['steps'] == required_steps \
            and rq['max_steps'] == required_steps and rq['elapsed'] <= rq['wall_cap_seconds']
        checks.append(au._check('R2', steps_ok, steps, {'steps': required_steps,
                                                         'wall_cap_seconds': None if rq is None else rq['wall_cap_seconds']}))
        recon = None if rq is None else rq['reconstruction']
        r2 = None if recon is None or not recon['sst'] > 0 else 1.0 - recon['mse'] / recon['sst']
        checks.append(au._check('R3', r2 is not None and r2 > 0, r2, 0.0))
        norms = None if recon is None else list(recon['residual_norms'])
        checks.append(au._check('R4', norms is not None and len(norms) == depth + 1 and all(a > b for a, b in zip(norms, norms[1:])),
                                norms, 'strictly decreasing'))
    else:
        found = audit['balanced_children']
        checks.append(au._check('B1', found['balanced'] is True, found['max_spread'], au.BALANCE_TOLERANCE))
        if prereg is not None:
            mismatches = []
            for key, expected in prereg.items():
                actual = au._lookup(audit, key)
                if type(actual) is not type(expected) or actual != expected:
                    mismatches.append({'key': key, 'expected': expected, 'actual': actual})
            checks.append(au._check('B2', not mismatches, mismatches, dict(prereg)))
    return {'passed': all(c['ok'] for c in checks), 'checks': checks}


def prefix_agreement(ids_a, sids_a, ids_b, sids_b, k):
    """Report-only: how many items carry the same first ``k`` digits in two mappings of the same catalog."""
    ids_a, ids_b, sids_a, sids_b = (np.asarray(x) for x in (ids_a, ids_b, sids_a, sids_b))
    if ids_a.ndim != 1 or not np.array_equal(ids_a, ids_b) or sids_a.ndim != 2 or sids_b.ndim != 2 \
            or len(sids_a) != len(ids_a) or len(sids_b) != len(ids_b):
        raise ValueError('mappings must cover the same catalog in the same order')
    if type(k) is not int or not 1 <= k <= min(sids_a.shape[1], sids_b.shape[1]) - 1:
        raise ValueError('k must be between 1 and the number of semantic digits of both mappings')
    equal = int(np.all(sids_a[:, :k] == sids_b[:, :k], axis=1).sum())
    return {'items': int(len(ids_a)), 'equal_prefix_items': equal, 'rate': equal / len(ids_a), 'prefix_digits': k}
