"""G6-PREP shared synthetic fixtures (no data files, no paths). Not a test module."""
import numpy as np

from vkgr.sid import SIDCatalog
from vkgr.tiger import SIDTokens


def reference_suffix(ids, prefixes):
    """Independent reference: rank inside the group sharing all prefix digits, ascending item id, plain Python loop."""
    seen = {}
    out = np.zeros(len(ids), dtype=np.uint32)
    for row in np.argsort(ids, kind='stable'):
        key = tuple(int(x) for x in prefixes[row])
        out[row] = seen.get(key, 0)
        seen[key] = int(out[row]) + 1
    return out


def synth(n=200, branching=4, depth=3, seed=0, id_span=10 ** 6):
    """(ids uint32 ascending [N], sids uint32 [N, depth+1]) with random prefixes and canonical suffixes."""
    rng = np.random.RandomState(seed)
    ids = np.sort(rng.choice(id_span, n, replace=False)).astype(np.uint32)
    prefixes = rng.randint(0, branching, size=(n, depth))
    sids = np.zeros((n, depth + 1), dtype=np.uint32)
    sids[:, :depth] = prefixes
    sids[:, depth] = reference_suffix(ids, prefixes)
    return ids, sids


def catalog(ids, sids, branching, depth):
    return SIDCatalog(ids, sids, branching, depth, [])


def tokens_for(cat):
    """Frozen ``SIDTokens`` for depth 3 (the parity reference); the depth-generic class otherwise."""
    if cat.depth == 3:
        return SIDTokens(cat)
    from vkgr.sid_depth import SIDTokensD
    return SIDTokensD(cat)


def plain_codec(n=200, branching=4, depth=3, seed=0):
    ids, sids = synth(n, branching, depth, seed)
    return tokens_for(catalog(ids, sids, branching, depth))


def collision_leaf(sids, depth):
    """Row indices of one leaf that holds at least two items (first such leaf in row order)."""
    keys = [tuple(int(x) for x in row[:depth]) for row in sids]
    groups = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    for rows in groups.values():
        if len(rows) >= 2:
            return rows
    raise AssertionError('fixture has no collision leaf')


def balanced_codes(n, branching, depth):
    """Independent restatement of the capacity rule: split every node into min(b, size) consecutive near-equal parts."""
    codes = np.zeros((n, depth), dtype=np.int64)

    def split(lo, hi, level):
        if level == depth:
            return
        m = hi - lo
        k = min(branching, m)
        q, r = divmod(m, k)
        start = lo
        for child in range(k):
            size = q + (1 if child < r else 0)
            codes[start:start + size, level] = child
            split(start, start + size, level + 1)
            start += size
    split(0, n, 0)
    return codes


def mapping_from(codes, seed=0, id_span=10 ** 6):
    from vkgr import sid_depth
    n = len(codes)
    ids = np.sort(np.random.RandomState(seed).choice(id_span, n, replace=False)).astype(np.uint32)
    return ids, sid_depth.assign_suffix(ids, codes)
