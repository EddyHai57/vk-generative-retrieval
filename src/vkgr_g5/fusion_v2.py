"""G5 two-route fusion core (pure functions, stdlib only) following the FINAL card sections 3 and 4.

Algorithm (frozen, no search): equal-weight reciprocal rank fusion with k=60, rank from 1, exact ``Fraction(1, 60 + rank)`` sums;
total budget 200, quota ``200 // n`` per route (2 routes -> 100, 3 routes -> 66, the remaining 2 slots are never allocated);
a short G list is never padded and its empty slots are never given to another route; P and Pop must supply their full quota
(otherwise the input gate fails); duplicates inside one route are fatal; routes are merged by item id; final order is
``(-score, item_id)``; a single route is its own top 50. Nothing here reads files, models, checkpoints or the held-out split.

Statistics (frozen): H1 = C5-C1, H2 = C5-C4 and C5-C7 (intersection-union), D1 = C6-C4 descriptive; exact two-sided McNemar
``p = min(1, 2 * BinomCDF(min(b, c); b + c, 1/2))``; bootstrap 95% interval over queries with ``random.Random(20261014)``,
10000 resamples, indices drawn with ``rng.randrange(n)`` sequentially, ONE shared index set for all four comparisons, nearest-rank
quantiles (rank = ceil(p * resamples)). The executor reports mechanical condition flags only; no verdict is written here.
"""
import hashlib
import math
import random
import sys
from pathlib import Path
from fractions import Fraction

B = 200
RRF_K = 60
FINAL = 50
BOOT_N = 10000
BOOT_SEED = 20261014
MIN_HITS = 6
ALPHA = Fraction(25, 1000)
ROUTE_LETTER = {'P': 'P', 'G': 'G', 'POP': 'Pop'}
ITEM_GROUPS = ('unsupported0', 'positive_le_median', 'positive_median_to_p90', 'positive_gt_p90')
LOW_POPULARITY = ('unsupported0', 'positive_le_median')
HISTORY_BUCKETS = (('1-10', 1, 10), ('11-25', 11, 25), ('26-49', 26, 49), ('50', 50, 50))


def row_definitions(seed=42):
    """Main table for seed 42 (C1-C7); other seeds only C1, C4, C5 with same-seed P and G."""
    p, g = ('P', seed), ('G', seed)
    rows = {'C1': [p], 'C4': [p, ('POP', None)], 'C5': [p, g]}
    if seed == 42:
        rows = {'C1': [p], 'C2': [g], 'C3': [('POP', None)], 'C4': rows['C4'], 'C5': rows['C5'],
                'C6': [p, g, ('POP', None)], 'C7': [p, ('P', 2024)]}
    return rows


REPORT_NOTES = (
    'Bootstrap: G5 uses random.Random(20261014) with sequential rng.randrange(n) index draws (protocol inherited from rev3). The earlier P-line batches (stack, bidirectional '
    'confirmation, user profiles) used NumPy default_rng(0) paired bootstrap. These are NOT the same protocol: intervals must not be compared across batches directly.',
    'Threshold wording: a net gain of at least 7 hits is a necessary condition for p < 0.025 and is not sufficient; the binding constraint is that the discordant pairs are '
    'highly concentrated on one side (e.g. b/c = 7/0 passes, 6/0, 8/1 and 10/3 do not). The registered >= 6 hits bar is unchanged.',
    'Alpha: G5 uses alpha = 0.025 per family (two-family Bonferroni), stricter than the p < 0.05 used by earlier P-line cards; a G5 non-pass is not equivalent to a P-line non-promotion.',
    'Limitations: G-side candidates have no second decoding cross-check; D-1 type deviation (co-occurrence inputs use training-period events including the target itself); about 0.2% '
    'of items use a content fallback; the 623 OOV validation queries are structurally zero-hit; early-stopping right-censoring. No online A/B or causal claim.',
)


def script_sha256():
    """SHA256 of this module's bytes (it contains the bootstrap implementation); recorded in every output so results are exactly reproducible."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def quota(n):
    if type(n) is not int or n < 1:
        raise ValueError('invalid route count')
    return FINAL if n == 1 else B // n


def rrf_scores(taken):
    """Exact equal-weight RRF scores of the routes' entered parts (rank starts at 1)."""
    scores = {}
    for part in taken:
        for rank, item in enumerate(part, 1):
            scores[item] = scores.get(item, Fraction(0)) + Fraction(1, RRF_K + rank)
    return scores


def condition_met(b, c):
    """Mechanical H-condition: net gain of at least MIN_HITS hits and exact two-sided McNemar p below alpha (per family)."""
    return bool(b - c >= MIN_HITS and mcnemar_exact(b, c) < float(ALPHA))


def fuse_query(lists, names):
    """Fuse one query. ``lists``: per-route ranked item lists; ``names``: route kinds ('P'|'G'|'POP') in the same order.

    Returns final (<=50 items), pool_size, per-route filled counts and the per-route sets actually entered (first q of each route).
    """
    n = len(lists)
    q = quota(n)
    taken, filled = [], []
    for ranked, name in zip(lists, names):
        if len(set(ranked)) != len(ranked):
            raise ValueError('duplicate item inside one route')
        part = list(ranked[:q])
        if name != 'G' and len(part) < q:
            raise ValueError('input gate: route shorter than its quota')
        taken.append(part); filled.append(len(part))
    if n == 1:
        return {'final': taken[0], 'pool_size': len(taken[0]), 'filled': filled, 'entered': [set(taken[0])], 'quota': q}
    scores = rrf_scores(taken)
    order = sorted(scores, key=lambda item: (-scores[item], item))
    return {'final': order[:FINAL], 'pool_size': len(scores), 'filled': filled, 'entered': [set(p) for p in taken], 'quota': q}


def rank_of(target, final):
    try:
        return final.index(target) + 1
    except ValueError:
        return None


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))


def nearest_rank(sorted_values, p):
    rank = math.ceil(Fraction(p) * len(sorted_values))
    return sorted_values[max(rank, 1) - 1]


def bootstrap_intervals(diffs, n_resamples=BOOT_N, seed=BOOT_SEED):
    """``diffs``: ordered dict name -> per-query integer difference in {-1,0,1}; one shared index stream for all comparisons."""
    names = list(diffs)
    n = len(diffs[names[0]])
    if any(len(v) != n for v in diffs.values()) or n == 0:
        raise ValueError('equal non-empty difference vectors required')
    rng = random.Random(seed)
    sums = {name: [] for name in names}
    for _ in range(n_resamples):
        index = [rng.randrange(n) for _ in range(n)]
        for name in names:
            vector = diffs[name]
            sums[name].append(sum(vector[i] for i in index))
    return {name: [nearest_rank(sorted(values), Fraction(25, 1000)), nearest_rank(sorted(values), Fraction(975, 1000))]
            for name, values in sums.items()}


def percentile(values, p):
    return nearest_rank(sorted(values), Fraction(p)) if values else None


def summarize(values):
    values = list(values)
    if not values:
        return {'mean': None, 'median': None, 'p10': None, 'p90': None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {'mean': sum(values) / len(values), 'median': median, 'p10': nearest_rank(ordered, Fraction(1, 10)), 'p90': nearest_rank(ordered, Fraction(9, 10))}


def history_bucket(length):
    for name, low, high in HISTORY_BUCKETS:
        if low <= length <= high:
            return name
    raise ValueError('history length outside 1..50')


def slice_names(query, g_short):
    """All slice memberships of one query. Slices overlap by design (a query is in several dimensions)."""
    names = ['all', 'in_catalog' if not query['oov'] else 'oov', 'support:' + query['support'], 'popularity:' + query['popularity_group'],
             'user:' + query['user_group'], 'history:' + history_bucket(query['history_len'])]
    if g_short is not None:
        names.append('g_short:' + g_short)
    return names


def validate_queries(queries, catalog_size):
    if not queries or any(set(q) != {'target', 'oov', 'support', 'popularity_group', 'user_group', 'history_len'} for q in queries):
        raise ValueError('query metadata fields')
    for q in queries:
        if q['support'] not in ('supported', 'unsupported', 'oov') or (q['support'] == 'oov') != bool(q['oov']):
            raise ValueError('support label inconsistent with oov flag')
        if q['popularity_group'] not in ITEM_GROUPS + ('oov',) or not 1 <= q['history_len'] <= 50:
            raise ValueError('popularity group / history length')
    if type(catalog_size) is not int or catalog_size < 1:
        raise ValueError('catalog size')


def route_list(routes, key, ordinal):
    value = routes[key]
    return value if key[0] == 'POP' else value[ordinal]


def evaluate_row(row, routes, queries):
    """Fuse every query for one row; returns per-query results."""
    names = [r[0] for r in row]
    out = []
    for ordinal in range(len(queries)):
        out.append(fuse_query([route_list(routes, key, ordinal) for key in row], names))
    return out


def row_metrics(results, queries, indices=None):
    indices = range(len(queries)) if indices is None else indices
    hits, ndcg, count = 0, 0.0, 0
    for i in indices:
        count += 1
        rank = rank_of(queries[i]['target'], results[i]['final'])
        if rank is not None:
            hits += 1; ndcg += 1.0 / math.log2(rank + 1)
    return {'n': count, 'hits': hits, 'R@50': hits / count if count else None, 'N@50': ndcg / count if count else None}


def hit_vector(results, queries):
    return [1 if rank_of(q['target'], r['final']) is not None else 0 for r, q in zip(results, queries)]


def new_lost(vec, ref, indices):
    new = sum(1 for i in indices if vec[i] and not ref[i]); lost = sum(1 for i in indices if ref[i] and not vec[i])
    return {'new': new, 'lost': lost, 'net': new - lost}


def route_label(key):
    return ROUTE_LETTER[key[0]] + ('' if key[1] is None else str(key[1]))


def attribution(row, results, queries, indices):
    """Hits of the fused row split by which routes' entered sets contain the target."""
    cells = {}
    for i in indices:
        target = queries[i]['target']
        if rank_of(target, results[i]['final']) is None:
            continue
        label = '∩'.join(route_label(row[k]) for k, entered in enumerate(results[i]['entered']) if target in entered)
        cells[label] = cells.get(label, 0) + 1
    return cells


def g_independent_hits(row, results, queries, indices):
    """Final hits whose target is in the G route's entered set and not in the P route's entered set (None without a G route)."""
    names = [r[0] for r in row]
    if 'G' not in names or 'P' not in names:
        return None
    g, p = names.index('G'), names.index('P')
    return sum(1 for i in indices if rank_of(queries[i]['target'], results[i]['final']) is not None
               and queries[i]['target'] in results[i]['entered'][g] and queries[i]['target'] not in results[i]['entered'][p])


def overlap_stats(row, results):
    out = {}
    for a in range(len(row)):
        for b in range(a + 1, len(row)):
            q = results[0]['quota']
            over_q, over_b = [], []
            for r in results:
                if not r['entered'][b]:
                    raise ValueError('empty second route inside overlap statistics')
                inter = len(r['entered'][a] & r['entered'][b])
                over_q.append(inter / q); over_b.append(inter / len(r['entered'][b]))
            out[f'{route_label(row[a])}|{route_label(row[b])}'] = {'over_quota': summarize(over_q), 'over_second_route_size': summarize(over_b)}
    return out


def coverage(results, catalog_size):
    final_items, pool_items = set(), set()
    for r in results:
        final_items.update(r['final'])
        for entered in r['entered']:
            pool_items.update(entered)
    return {'final_distinct_items': len(final_items), 'final_coverage': len(final_items) / catalog_size,
            'pool_distinct_items': len(pool_items), 'pool_coverage': len(pool_items) / catalog_size}


def run(routes, queries, catalog_size, seeds=(42, 2024, 2025)):
    """Compute the whole table. ``routes``: {('P', seed): [[...]*N], ('G', seed): [[...]*N], ('POP', None): [..]}."""
    validate_queries(queries, catalog_size)
    n_queries = len(queries)
    for key, value in routes.items():
        if key[0] != 'POP' and len(value) != n_queries:
            raise ValueError(f'route {key} query count mismatch')
    g_short = {s: ['short' if len(routes[('G', s)][i]) < 100 else 'full' for i in range(n_queries)] for s in seeds if ('G', s) in routes}
    not_run, vectors, results_by = {}, {}, {}
    for seed in seeds:
        rows = row_definitions(seed)
        if any(key not in routes for row in rows.values() for key in row):
            not_run[f'seed{seed}'] = 'NOT_RUN(missing route inputs: ' + ', '.join(sorted({f"{k[0]}{'' if k[1] is None else k[1]}" for row in rows.values() for k in row if k not in routes})) + ')'
            continue
        for name, row in rows.items():
            results = evaluate_row(row, routes, queries)
            results_by[(seed, name)] = (row, results); vectors[(seed, name)] = hit_vector(results, queries)
    out = {'queries': n_queries, 'catalog_size': catalog_size, 'rows': {}}
    for (seed, name), (row, results) in results_by.items():
        gs = g_short.get(seed)
        c1, c4 = vectors.get((seed, 'C1')), vectors.get((seed, 'C4'))
        by_slice = {}
        for i, query in enumerate(queries):
            for s in slice_names(query, gs[i] if gs else None):
                by_slice.setdefault(s, []).append(i)
        entry = {'routes': [route_label(k) for k in row], 'quota': results[0]['quota'],
                 'pool_size': summarize([r['pool_size'] for r in results]), 'filled': {route_label(k): summarize([r['filled'][j] for r in results]) for j, k in enumerate(row)},
                 'g_short_queries': sum(1 for label in gs if label == 'short') if any(k[0] == 'G' for k in row) and gs else None,
                 'coverage': coverage(results, catalog_size), 'slices': {}}
        if len(row) > 1:
            entry['overlap'] = overlap_stats(row, results)
        for s, idx in by_slice.items():
            cell = row_metrics(results, queries, idx)
            if len(row) > 1:
                cell['vs_C1'] = new_lost(vectors[(seed, name)], c1, idx) if c1 else None
                cell['vs_C4'] = new_lost(vectors[(seed, name)], c4, idx) if c4 else None
                cell['attribution'] = attribution(row, results, queries, idx)
                cell['g_independent_hits'] = g_independent_hits(row, results, queries, idx)
            entry['slices'][s] = cell
        out['rows'][f'seed{seed}:{name}'] = entry
    out['bootstrap'] = {'resamples': BOOT_N, 'seed': BOOT_SEED, 'index_draw': 'sequential rng.randrange(n) per index; one shared index set for all comparisons', 'python': sys.version.split()[0],
                        'fusion_v2_script_sha256': script_sha256(), 'comparable_with_numpy_default_rng_0_batches': False}
    out['report_notes'] = list(REPORT_NOTES)
    out['not_run'] = not_run
    main_vectors = {name: vectors[(42, name)] for name in ('C1', 'C4', 'C5', 'C6', 'C7') if (42, name) in vectors}
    if all(x in main_vectors for x in ('C1', 'C4', 'C5', 'C6', 'C7')):
        out['paired'], out['mechanical_conditions'] = paired_table(main_vectors)
        out['localization'] = {name: localization(main_vectors, name, queries, g_short[42]) for name in ('C5', 'C6')}
    else:
        missing = [x for x in ('C1', 'C4', 'C5', 'C6', 'C7') if x not in main_vectors]
        out['paired'] = out['mechanical_conditions'] = out['localization'] = f'NOT_RUN(main-table rows missing: {missing})'
    for name, entry in out['rows'].items():
        oov_hits = entry['slices'].get('oov', {}).get('hits')
        entry['oov_hits'] = oov_hits
        entry['oov_nonzero_hits_anomaly'] = bool(oov_hits)
    return out


PAIRS = (('H1', 'C5', 'C1'), ('H2a', 'C5', 'C4'), ('H2b', 'C5', 'C7'), ('D1', 'C6', 'C4'))


def paired_table(main_vectors, n_resamples=BOOT_N, seed=BOOT_SEED):
    """b/c/delta/exact McNemar/bootstrap interval for H1, H2a, H2b and descriptive D1 plus the mechanical condition flags."""
    diffs = {label: [x - y for x, y in zip(main_vectors[a], main_vectors[b])] for label, a, b in PAIRS}
    intervals = bootstrap_intervals(diffs, n_resamples, seed)
    paired = {}
    for label, a, b in PAIRS:
        bb = sum(1 for x, y in zip(main_vectors[a], main_vectors[b]) if x and not y)
        cc = sum(1 for x, y in zip(main_vectors[a], main_vectors[b]) if y and not x)
        paired[label] = {'row': a, 'reference': b, 'b': bb, 'c': cc, 'delta_hits': bb - cc, 'mcnemar_p': mcnemar_exact(bb, cc),
                         'bootstrap95_delta_hits': intervals[label], 'mechanical_condition': ('descriptive_only' if label == 'D1' else condition_met(bb, cc))}
    conditions = {'H1_condition_met': paired['H1']['mechanical_condition'],
                  'H2_condition_met': bool(paired['H2a']['mechanical_condition'] and paired['H2b']['mechanical_condition']),
                  'alpha_per_family': float(ALPHA), 'min_hits': MIN_HITS,
                  'note': 'mechanical flags only; meaning and anchor decisions belong to the adjudicator'}
    return paired, conditions


def localization(vectors, name, queries, g_short_labels):
    """rev3 rule: query set counted once; localized only if net_in_set >= 6 and share >= 0.5; false if total net <= 0."""
    total = sum(vectors[name]) - sum(vectors['C1'])
    members = [i for i, q in enumerate(queries) if not q['oov'] and (q['support'] == 'unsupported' or q['popularity_group'] in LOW_POPULARITY or g_short_labels[i] == 'short')]
    in_set = sum(vectors[name][i] - vectors['C1'][i] for i in members)
    localized = bool(total > 0 and in_set >= MIN_HITS and 2 * in_set >= total)
    return {'low_support_queries': len(members), 'net_in_set': in_set, 'net_total': total, 'localized': localized}
