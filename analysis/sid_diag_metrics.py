"""Pure, read-only math for SID-FIX S0; stdlib only.

Inputs are already authorized raw lists/mappings; this module opens no files,
loads no models, fits no thresholds, and never reads test data. Item identifiers
must be hashable. Codes are equal-length sequences of nonnegative integers,
including any padded positions and collision suffix. A's ``semantic_depth``
excludes suffix AND padding (B256D2 therefore uses 2). B/D deliberately compare
stored prefix positions 1..3, including D2's third-position padding. Full SID
means the entire stored tuple; full SID matches need not be exact item hits.

Prediction rows contain item IDs in archived rank order, already resolved by the
archive adapter. Ties retain that order. Stable item deduplication precedes the
50-item cutoff. An unknown prediction is a contract error, even in an OOV query.
OOV target queries are counted separately and excluded from B denominators.
C's OOV bin has zero hits and its own query count; all other bins are in catalog.
Thresholds are supplied from the frozen train-only G4 artifact, not recomputed.

D requires each user's events already ordered according to the train contract.
Any out-of-catalog history event fails closed: silently dropping it could create
an adjacency that did not exist. Adjacent equal items are excluded without
bridging. Random pairs use independent uniform draws WITH replacement (self
pairs allowed) from catalog mapping insertion order, ``random.Random(seed)``
with two ``randrange(N)`` calls per pair. Stream order and seed are reported so
the same bundle order reproduces the result. None denotes UNKNOWN/no denominator.

All returned dicts are JSON-safe; counts are Python int, rates Python float or
None. Prefix keys are strings '1', '2', '3'. No semantic conclusion is emitted
by ``decompose_gain``: its branch identifies only the frozen arithmetic rule.
"""
from collections import Counter
from math import isfinite, log
from numbers import Integral, Real
import random


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _codes(item_to_code, min_depth):
    if not isinstance(min_depth, Integral) or isinstance(min_depth, bool) or min_depth < 1:
        raise ValueError('depth must be a positive integer')
    codes = {}
    lengths = set()
    for item, code in item_to_code.items():
        try:
            values = tuple(code)
        except TypeError as exc:
            raise ValueError('code must be an integer sequence') from exc
        if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0 for v in values):
            raise ValueError('code tokens must be nonnegative integers')
        if len(values) < min_depth:
            raise ValueError('code is shorter than requested prefix/semantic depth')
        lengths.add(len(values))
        codes[item] = tuple(int(v) for v in values)
    if len(lengths) > 1:
        raise ValueError('all stored full codes must have equal length')
    return codes


def _occupancy(counter):
    """Leaf occupancy; collision_items counts surplus IDs, pairs is unordered."""
    n = sum(counter.values())
    return {'count': len(counter), 'avg_items': _ratio(n, len(counter)),
            'max_items': max(counter.values(), default=0),
            'collision_items': n - len(counter),
            'collision_pairs': sum(v * (v - 1) // 2 for v in counter.values()),
            'collided_leaves': sum(v > 1 for v in counter.values())}


def structure_metrics(item_to_code, branching, semantic_depth):
    """A: semantic-layer stats; ``leaves`` excludes suffix/padding.

    branching is one radix or one radix per semantic layer, each >=2. Marginal
    token entropy is divided by log(radix). avg_children averages distinct
    children over occupied parent prefixes, including the root for layer 1.
    full_code_leaves separately exposes full-SID collision diagnostics.
    """
    codes = _codes(item_to_code, semantic_depth)
    if isinstance(branching, Integral):
        bases = [branching] * semantic_depth
    else:
        try:
            bases = list(branching)
        except TypeError as exc:
            raise ValueError('branching must be an integer or sequence') from exc
    if len(bases) != semantic_depth or any(not isinstance(b, Integral) or isinstance(b, bool) or b < 2 for b in bases):
        raise ValueError('one branching radix >=2 is required per semantic layer')
    for code in codes.values():
        if any(code[j] >= bases[j] for j in range(semantic_depth)):
            raise ValueError('semantic token is outside its branching radix')
    n = len(codes)
    layers = []
    for j, base in enumerate(bases):
        token_counts = Counter(code[j] for code in codes.values())
        node_counts = Counter(code[:j + 1] for code in codes.values())
        parents = {code[:j] for code in codes.values()}
        entropy = -sum((v/n) * log(v/n) for v in token_counts.values()) / log(base) if n else None
        layers.append({'k': j+1, 'branching': int(base),
                       'distinct_codes': len(token_counts),
                       'utilization': len(token_counts) / base,
                       'normalized_entropy': entropy,
                       'parents': len(parents), 'nodes': len(node_counts),
                       'avg_children': _ratio(len(node_counts), len(parents)),
                       'avg_items_per_node': _ratio(n, len(node_counts))})
    return {'n_items': n, 'semantic_depth': int(semantic_depth), 'layers': layers,
            'effective_layers': sum(x['avg_children'] is not None and x['avg_children'] > 1 for x in layers),
            'leaves': _occupancy(Counter(code[:semantic_depth] for code in codes.values())),
            'full_code_leaves': _occupancy(Counter(codes.values()))}


def _rows(targets, predictions, catalog):
    if len(targets) != len(predictions):
        raise ValueError('targets and prediction rows must have equal length')
    for target, row in zip(targets, predictions):
        seen, unique = set(), []
        for item in row:
            if item not in catalog:
                raise ValueError('prediction item outside catalog')
            if item not in seen:
                seen.add(item)
                unique.append(item)
        yield target, unique[:50]


def _hit_stats(n, one, fifty):
    return {'n': n, 'top1_hits': one, 'top50_hits': fifty,
            'top1_rate': _ratio(one, n), 'top50_rate': _ratio(fifty, n)}


def prediction_metrics(targets, predictions, item_to_code):
    """B: archived ordered item predictions -> prefix, item and full-SID hits.

    All three prefixes are required in stored codes. OOV counts are observed,
    never assumed to be 623. The runner must check 623 against its query contract.
    ``full_sid_is_item_equivalent`` is false whenever a full-code collision exists.
    """
    codes = _codes(item_to_code, 3)
    n, oov = 0, 0
    counts = {k: [0, 0] for k in ('1', '2', '3', 'exact_item', 'full_sid')}
    for target, row in _rows(targets, predictions, codes):
        if target not in codes:
            oov += 1
            continue
        n += 1
        for k in (1, 2, 3):
            same = [codes[item][:k] == codes[target][:k] for item in row]
            counts[str(k)][0] += bool(same and same[0])
            counts[str(k)][1] += any(same)
        counts['exact_item'][0] += bool(row and row[0] == target)
        counts['exact_item'][1] += target in row
        same = [codes[item] == codes[target] for item in row]
        counts['full_sid'][0] += bool(same and same[0])
        counts['full_sid'][1] += any(same)
    return {'n_total': n+oov, 'n_in_catalog': n, 'n_oov': oov,
            'prefix_by_k': {str(k): _hit_stats(n, *counts[str(k)]) for k in (1,2,3)},
            'exact_item': _hit_stats(n, *counts['exact_item']),
            'full_sid': _hit_stats(n, *counts['full_sid']),
            'full_sid_is_item_equivalent': len(set(codes.values())) == len(codes)}


def popularity_metrics(targets, predictions, train_counts, median, p90, catalog):
    """C: G4 bins with fixed supplied thresholds and exact top-50 ITEM hits.

    Precedence: outside catalog -> oov; count 0 -> unsupported;
    0<count<=median -> low; median<count<=p90 -> mid; count>p90 -> high.
    Missing train counts are 0. supplemental_longtail_unsupported sums low and
    unsupported without altering the original five bins. No train count is fit.
    catalog may be a mapping or a set of item IDs.
    """
    if any(not isinstance(v, Real) or isinstance(v, bool) or not isfinite(v) or v < 0 for v in (median,p90)) or median > p90:
        raise ValueError('thresholds must be finite 0 <= median <= p90')
    if any(not isinstance(v, Integral) or isinstance(v,bool) or v < 0 for v in train_counts.values()):
        raise ValueError('train counts must be nonnegative integers')
    bins = {key: {'n': 0, 'hits': 0} for key in ('oov','unsupported','low','mid','high')}
    for target, row in _rows(targets,predictions,catalog):
        count = train_counts.get(target,0)
        if target not in catalog:
            key = 'oov'
        elif count == 0:
            key = 'unsupported'
        elif count <= median:
            key = 'low'
        elif count <= p90:
            key = 'mid'
        else:
            key = 'high'
        bins[key]['n'] += 1
        bins[key]['hits'] += key != 'oov' and target in row
    for x in bins.values():
        x['rate'] = _ratio(x['hits'],x['n'])
    supplemental = {field: sum(bins[k][field] for k in ('low','unsupported')) for field in ('n','hits')}
    supplemental['rate'] = _ratio(supplemental['hits'],supplemental['n'])
    return {'thresholds': {'median': float(median), 'p90': float(p90)},
            'bins': bins, 'supplemental_longtail_unsupported': supplemental}


def cooccurrence_metrics(item_to_code, user_histories, seed=0):
    """D: adjacent distinct train events versus equally many uniform pairs.

    user_histories is an iterable of ordered per-user iterables. It may stream;
    only the catalog and integer counters are retained. Never joins users.
    Random endpoint draws are independent, so self-pairs are intentionally kept.
    """
    codes = _codes(item_to_code,3)
    if not isinstance(seed,Integral) or isinstance(seed,bool):
        raise ValueError('seed must be an integer')
    n_adjacent, n_same, n_pairs = 0, 0, 0
    observed = [0,0,0]
    for history in user_histories:
        previous, have_previous = None, False
        for item in history:
            if item not in codes:
                raise ValueError('history event outside catalog; adjacency cannot be silently changed')
            if have_previous:
                n_adjacent += 1
                if previous == item:
                    n_same += 1
                else:
                    n_pairs += 1
                    for j in range(3):
                        observed[j] += codes[previous][:j+1] == codes[item][:j+1]
            previous, have_previous = item, True
    population = list(codes.values())
    rng = random.Random(int(seed))
    baseline = [0,0,0]
    for _ in range(n_pairs):
        a = population[rng.randrange(len(population))]
        b = population[rng.randrange(len(population))]
        for j in range(3): baseline[j] += a[:j+1] == b[:j+1]
    return {'seed': int(seed), 'random_algorithm': 'python_random_Random_randrange_independent_with_replacement',
            'catalog_order': 'mapping_insertion_order', 'catalog_size': len(codes),
            'history_oov_policy': 'fail_closed_no_skips_no_bridging',
            'n_adjacent_events': n_adjacent, 'n_same_item_excluded': n_same,
            'n_pairs': n_pairs, 'n_random_pairs': n_pairs,
            'prefix_by_k': {str(j+1): {'n': n_pairs,
                 'cooccurrence_matches': observed[j], 'random_matches': baseline[j],
                 'cooccurrence_rate': _ratio(observed[j],n_pairs),
                 'random_rate': _ratio(baseline[j],n_pairs),
                 'lift': _ratio(observed[j],baseline[j])} for j in range(3)}}


def decompose_gain(candidate_by_seed, reference_by_seed, seeds):
    """Mechanical frozen rule, summing ONLY explicitly listed independent seeds.

    Inputs: {seed: {high, mid, low, unsupported: nonnegative exact item hit count}}.
    OOV is omitted from totals (hits are definitionally 0). Missing bins or seeds
    fail closed. No seed42 selection or A1 eligibility is inferred here; runner
    supplies frozen [2024,2025] or eligible A1 [42,2024,2025].
    """
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('explicit nonempty unique seed list required')
    keys = ('high','mid','low','unsupported')
    delta = dict.fromkeys(keys,0)
    for seed in seeds:
        if seed not in candidate_by_seed or seed not in reference_by_seed:
            raise ValueError('requested seed missing')
        for key in keys:
            for source in (candidate_by_seed,reference_by_seed):
                if key not in source[seed] or not isinstance(source[seed][key],Integral) or isinstance(source[seed][key],bool) or source[seed][key]<0:
                    raise ValueError('each seed requires all four nonnegative hit counts')
            delta[key] += int(candidate_by_seed[seed][key]) - int(reference_by_seed[seed][key])
    total = sum(delta.values())
    share = delta['high']/total if total > 0 else None
    branch = 'no_positive_gain' if total <= 0 else ('head_share_ge_0_5' if 2*delta['high'] >= total else 'head_share_lt_0_5')
    return {'seeds': seeds, 'delta_by_bin': delta, 'total_gain': total,
            'head_share': share, 'rule_branch': branch,
            'threshold': 0.5, 'threshold_has_stronger_basis': False}
