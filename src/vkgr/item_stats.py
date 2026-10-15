"""FEAT-04b ITEM_STATS_V1: per-item train statistics (exposure count, first week, like count, watch-ratio sum) for the C/D variants.

Everything here is a function of the train partitions (weeks 0-24 only), the frozen catalog and the item durations. There is no query,
validation or test input anywhere. The sidecar stores only integer aggregates in catalog order; the bucket constants live in the model module,
never in the sidecar.
"""
import hashlib
import json
from pathlib import Path

from .history_v2 import _read_pinned, _write_json

STATS_VERSION = 'ITEM_STATS_V1'
STAT_CUTOFF_WEEK = 24                                 # train weeks 0..24 only; 25 (validation) and 26 (test) are refused. Read at call time (a mutation test patches it)
TIMESPENT_MAX = 255                                   # the raw column is uint8: 0 is a legal value, 256 is not
DURATION_MIN, DURATION_MAX = 1, 255
RATIO_CAP_PM = 2000                                   # per-row watch ratio is capped at 2.0 (permille) so one row cannot dominate an item's sum
FIELDS = ('n', 'first_week', 'likes', 'ratio_pm_sum')


def ratio_pm(timespent, duration):
    """Integer watch ratio in permille, floor division, capped at RATIO_CAP_PM. timespent == 0 is legal and gives 0."""
    return min(RATIO_CAP_PM, 1000 * timespent // duration)


def _catalog(catalog):
    catalog = list(catalog)
    if len(set(catalog)) != len(catalog) or any(type(i) is not int for i in catalog):
        raise ValueError('catalog must be unique integers')
    return catalog


def aggregate_rows(partitions, catalog, durations):
    """item -> [n, first_week, likes, ratio_pm_sum] over the train partitions (the aggregation seam).

    ``partitions`` yields ``(week, events)`` with ``events`` a list of row dicts (a row's ordinal is its index in ``events``). Weeks must be
    strictly increasing ints with ``previous < week <= STAT_CUTOFF_WEEK``. A row of an item outside the catalog, a non-bool ``like``, a timespent that
    is not an int in 0..255, or a catalog item whose duration is not an int in 1..255 raises ValueError. Items without rows keep n == 0
    (``build_item_stats`` refuses them).
    """
    catalog = _catalog(catalog)
    for item in catalog:
        duration = durations.get(item)
        if type(duration) is not int or not DURATION_MIN <= duration <= DURATION_MAX:
            raise ValueError('duration missing or outside [1,255]')
    acc = {item: [0, None, 0, 0] for item in catalog}
    previous = -1
    for week, events in partitions:
        if type(week) is not int or not previous < week <= STAT_CUTOFF_WEEK:
            raise ValueError('train ordering conflict / non-train week')
        previous = week
        for event in events:
            item = event.get('item_id')
            entry = acc.get(item) if type(item) is int else None
            if entry is None:
                raise ValueError('row item outside the catalog')
            like, timespent = event.get('like'), event.get('timespent')
            if type(like) is not bool:
                raise ValueError('like must be bool')
            if type(timespent) is not int or not 0 <= timespent <= TIMESPENT_MAX:
                raise ValueError('timespent outside uint8 [0,255]')
            entry[0] += 1
            if entry[1] is None:
                entry[1] = week
            entry[2] += like
            entry[3] += ratio_pm(timespent, durations[item])
    return acc


def build_item_stats(partitions, catalog, durations):
    """The four integer columns in catalog order plus the identity fields. Any catalog item with n == 0 raises ValueError (no placeholder)."""
    catalog = _catalog(catalog)
    acc = aggregate_rows(partitions, catalog, durations)
    if any(acc[item][0] == 0 for item in catalog):
        raise ValueError('catalog item without any train row (n == 0)')
    columns = {name: [acc[item][k] for item in catalog] for k, name in enumerate(FIELDS)}
    return {'stats_version': STATS_VERSION, 'cutoff_week': STAT_CUTOFF_WEEK, 'rows_total': sum(columns['n']), **columns}


def stats_digest(stats):
    """sha256 of the canonical JSON of the four columns (content only: no binding, no identity fields)."""
    return hashlib.sha256(json.dumps({name: stats[name] for name in FIELDS}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _no_validation_binding(value):
    """L10: the producer binding may not carry any validation / query / holdout digest (the statistics never see those inputs)."""
    if isinstance(value, dict):
        for key, inner in value.items():
            if any(word in str(key).lower() for word in ('query', 'validation', 'holdout')):
                raise ValueError('binding must not carry validation/query digests')
            _no_validation_binding(inner)
    elif isinstance(value, list):
        for inner in value:
            _no_validation_binding(inner)


def write_item_stats(output, partitions, catalog, support, durations, items_metadata_sha256, history_sha256, binding):
    """Build the statistics from the train partitions and write ``stats.json`` + ``manifest.json`` (a fresh directory only)."""
    catalog = _catalog(catalog)
    if not set(support) <= set(catalog):
        raise ValueError('support outside the catalog')
    _no_validation_binding(binding)
    built = build_item_stats(partitions, catalog, durations)                      # raises before anything is created
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    artifacts = {'stats.json': _write_json(output / 'stats.json', {name: built[name] for name in FIELDS})}
    manifest = {'schema_version': 1, 'kind': 'item_stats_prepare', 'stats_version': STATS_VERSION, 'fields': list(FIELDS), 'cutoff_week': built['cutoff_week'],
                'rows_total': built['rows_total'], 'catalog_items': len(catalog), 'supported_items': len(set(support)),
                'items_metadata_sha256': items_metadata_sha256, 'history_manifest_sha256': history_sha256,
                'catalog_sha256': _digest(catalog), 'support_sha256': _digest(sorted(support)), 'stats_sha256': stats_digest(built),
                'definition': 'per catalog item (catalog order) over train weeks 0..cutoff only: n rows, first week seen, like rows, sum of min(2000, 1000*timespent//duration); integer aggregates only',
                'binding': binding, 'artifacts': artifacts, 'test_gate': 'closed_hash_only'}
    return {'manifest': manifest, 'manifest_sha256': _write_json(output / 'manifest.json', manifest)}


def _column(values, size):
    if type(values) is not list or len(values) != size or any(type(v) is not int for v in values):
        raise ValueError('stats column must be a list of ints, one per catalog item')
    return values


def load_item_stats(path, digest, items_metadata_sha256, history_sha256, catalog, support, expected_sha256=None):
    """Load and re-check the sidecar for this catalog/support; any identity, binding, range or digest difference raises ValueError.

    ``expected_sha256`` (None = not checked) pins the content digest to the audited value.
    """
    path = Path(path); value = _read_pinned(path, digest); catalog = _catalog(catalog)
    if value.get('kind') != 'item_stats_prepare' or value.get('stats_version') != STATS_VERSION or value.get('fields') != list(FIELDS) \
            or value.get('test_gate') != 'closed_hash_only':
        raise ValueError('item stats manifest identity')
    if value.get('cutoff_week') != STAT_CUTOFF_WEEK or type(value.get('cutoff_week')) is not int:
        raise ValueError('item stats cutoff week differs from the code')
    if value.get('items_metadata_sha256') != items_metadata_sha256:
        raise ValueError('stats not bound to this items metadata')
    if value.get('history_manifest_sha256') != history_sha256:
        raise ValueError('stats not bound to this history prepare')
    if value.get('catalog_sha256') != _digest(catalog) or value.get('catalog_items') != len(catalog):
        raise ValueError('stats not bound to this catalog')
    if not set(support) <= set(catalog) or value.get('support_sha256') != _digest(sorted(support)) or value.get('supported_items') != len(set(support)):
        raise ValueError('stats not bound to this support set')
    raw = _read_pinned(path.parent / 'stats.json', value['artifacts']['stats.json'])
    if type(raw) is not dict or set(raw) != set(FIELDS):
        raise ValueError('stats artifact fields')
    stats = {name: _column(raw[name], len(catalog)) for name in FIELDS}
    for n, first, likes, ratio in zip(stats['n'], stats['first_week'], stats['likes'], stats['ratio_pm_sum']):
        if n < 1 or not 0 <= first <= STAT_CUTOFF_WEEK or not 0 <= likes <= n or not 0 <= ratio <= RATIO_CAP_PM * n:
            raise ValueError('stored stats violate the per-item invariants')
    if value.get('rows_total') != sum(stats['n']) or value.get('stats_sha256') != stats_digest(stats):
        raise ValueError('stored stats do not reproduce the stored totals/digest')
    if expected_sha256 is not None and value['stats_sha256'] != expected_sha256:
        raise ValueError('stats differ from the audited sha256')
    stats = {'stats_version': STATS_VERSION, 'cutoff_week': value['cutoff_week'], 'rows_total': value['rows_total'], **stats}
    return {'manifest': value, 'manifest_sha256': digest, 'stats': stats, 'stats_sha256': value['stats_sha256']}


BUCKET_KINDS = ('expo', 'first', 'ratio', 'like')


def _check_bucket_fns(bucket_fns):
    if type(bucket_fns) is not dict or set(bucket_fns) != set(BUCKET_KINDS):
        raise ValueError('bucket_fns must hold exactly expo / first / ratio / like')
    return bucket_fns


def _buckets(bucket_fns, n, first_week, likes, ratio_sum):
    """Bucket ids of one item's statistics. ``expo(n)``, ``first(first_week)``, ``ratio(ratio_pm_sum, n)``, ``like(likes, n)``; the constants (edges,
    LIKE_P0, LIKE_K) live inside the functions, so the with / without comparison below can never move them."""
    return {'expo': bucket_fns['expo'](n), 'first': bucket_fns['first'](first_week),
            'ratio': bucket_fns['ratio'](ratio_sum, n), 'like': bucket_fns['like'](likes, n)}


def stats_self_inclusion(partitions, examples, catalog, durations, stats, bucket_fns):
    """OFFLINE diagnostic (never in the model path): how often does a training example's own target row decide its target item's bucket?

    For every example, remove exactly that one row (found through ``target_position`` in the partitions) from the target item's statistics
    (``n-1``, ``likes - like``, ``ratio_pm_sum - ratio_pm(row)``; ``first_week`` becomes the next row's week when the removed row is the item's earliest
    row) and bucket both versions with the SAME functions (LIKE_P0 and every edge stay fixed). ``n-1 == 0`` leaves ratio and first week undefined and counts
    as changed for those two; expo and like are defined by their formulas and compared as bucket ids.
    ``partitions`` must be the partitions ``stats`` was built from (the earliest week of every target item is cross-checked).
    """
    catalog = _catalog(catalog); bucket_fns = _check_bucket_fns(bucket_fns)
    index = {item: k for k, item in enumerate(catalog)}
    wanted = {}
    for example in examples:
        if example.get('target') not in index:
            raise ValueError('example target outside the catalog')
        wanted[tuple(example['target_position'])] = None
    first, second = {}, {}
    for week, events in partitions:
        for ordinal, event in enumerate(events):
            item = event.get('item_id'); position = (week, ordinal)
            if type(item) is int and item in index:
                if item not in first:
                    first[item] = position
                elif item not in second:
                    second[item] = position
            if position in wanted:
                wanted[position] = (item, event.get('like'), event.get('timespent'))
    counts = dict.fromkeys(('expo', 'ratio', 'like', 'first', 'first_is_own_row', 'n_le_3', 'n_eq_1'), 0)
    for example in examples:
        position = tuple(example['target_position']); target = example['target']; found = wanted[position]
        if found is None or found[0] != target:
            raise ValueError('target position does not hold the target item row')
        _, like, timespent = found
        if type(like) is not bool or type(timespent) is not int or not 0 <= timespent <= TIMESPENT_MAX:
            raise ValueError('target row has an invalid like/timespent')
        k = index[target]
        n, week, likes, total = (stats[name][k] for name in FIELDS)
        if first[target][0] != week:
            raise ValueError('stats do not match the partitions')
        own_first = position == first[target]
        n2, likes2, total2 = n - 1, likes - int(like), total - ratio_pm(timespent, durations[target])
        if likes2 < 0 or total2 < 0 or n2 < 0:
            raise ValueError('stats do not contain the target row')
        week2 = (second[target][0] if target in second else None) if own_first else week
        before = _buckets(bucket_fns, n, week, likes, total)
        undefined = n2 == 0
        after = {'expo': bucket_fns['expo'](n2), 'like': bucket_fns['like'](likes2, n2),
                 'ratio': None if undefined else bucket_fns['ratio'](total2, n2), 'first': None if undefined else bucket_fns['first'](week2)}
        for name in BUCKET_KINDS:
            counts[name] += after[name] is None or after[name] != before[name]
        counts['first_is_own_row'] += own_first
        counts['n_le_3'] += n <= 3
        counts['n_eq_1'] += n == 1
    size = len(examples)
    share = (lambda c: c / size) if size else (lambda c: 0.0)
    return {'examples': size, 'p_expo': share(counts['expo']), 'p_ratio': share(counts['ratio']), 'p_like': share(counts['like']),
            'p_first': share(counts['first']), 'p_first_is_own_row': share(counts['first_is_own_row']),
            'share_n_le_3': share(counts['n_le_3']), 'share_n_eq_1': share(counts['n_eq_1']), 'counts': counts}


def target_bucket_hist(examples, queries, catalog, stats, bucket_fns):
    """OFFLINE diagnostic: bucket histograms of the training targets and of the validation targets that lie in the catalog (shift check only)."""
    catalog = _catalog(catalog); bucket_fns = _check_bucket_fns(bucket_fns)
    index = {item: k for k, item in enumerate(catalog)}

    def histogram(targets):
        hist = {name: {} for name in BUCKET_KINDS}; inside = 0
        for target in targets:
            k = index.get(target) if type(target) is int else None
            if k is None:
                continue
            inside += 1
            ids = _buckets(bucket_fns, *(stats[name][k] for name in FIELDS))
            for name in BUCKET_KINDS:
                hist[name][ids[name]] = hist[name].get(ids[name], 0) + 1
        return {name: dict(sorted(hist[name].items())) for name in BUCKET_KINDS}, inside
    train_hist, train_in = histogram([e['target'] for e in examples])
    if train_in != len(examples):
        raise ValueError('a training target lies outside the catalog')
    val_hist, val_in = histogram([q['target'] for q in queries])
    return {'train': {'examples': len(examples), **train_hist},
            'validation': {'queries': len(queries), 'in_catalog': val_in, 'outside_catalog': len(queries) - val_in, **val_hist}}


def _ranks(values):
    """Average ranks (ties share the mean rank), 1-based."""
    order = sorted(range(len(values)), key=values.__getitem__); ranks = [0.0] * len(values); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rank correlation with average ranks; None when either column is constant."""
    if len(x) != len(y) or not x:
        raise ValueError('spearman needs two equal non-empty columns')
    rx, ry = _ranks(list(x)), _ranks(list(y)); n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def logq_redundancy(stats, catalog, support, counts, bucket_fns):
    """OFFLINE diagnostic: Spearman correlation, over the supported items, of the expo and like bucket ids with the logQ target-frequency column.

    ``counts`` maps item -> train target count (the ITEM_TARGET_FREQ_TRAIN_V1 counts; an item with no target example counts 0). Ranks are taken on the
    counts themselves: log q is a monotone transform of the count, so the rank correlation is the same.
    """
    catalog = _catalog(catalog); bucket_fns = _check_bucket_fns(bucket_fns)
    index = {item: k for k, item in enumerate(catalog)}
    if any(item not in index for item in support):
        raise ValueError('support outside the catalog')
    items = sorted(support)
    ids = [_buckets(bucket_fns, *(stats[name][index[item]] for name in FIELDS)) for item in items]
    frequency = [counts.get(item, 0) for item in items]
    return {'supported_items': len(items), 'expo_spearman': spearman([b['expo'] for b in ids], frequency) if items else None,
            'like_spearman': spearman([b['like'] for b in ids], frequency) if items else None}


def scan_ranges(partitions, catalog, durations):
    """Read-only range scan of the raw train rows (the ``range-check`` gate before ``prepare-stats``).

    ``partitions`` is an iterable of ``(week, events)`` with ``events`` an iterable of row dicts; ``durations`` maps item -> duration.
    Reports the timespent extremes over int values, how many rows hold timespent == 0, how many hold a non-int timespent, how many rows name an item
    outside the catalog, and the extremes of the catalog items' durations. It judges nothing and raises nothing for bad values: the caller decides.
    """
    catalog = list(catalog); known = set(catalog)
    rows = zero = non_int = outside = parts = 0
    low = high = None
    for _week, events in partitions:
        parts += 1
        for event in events:
            rows += 1
            item = event.get('item_id')
            if type(item) is not int or item not in known:
                outside += 1
            value = event.get('timespent')
            if type(value) is not int:
                non_int += 1
                continue
            zero += value == 0
            low = value if low is None else min(low, value)
            high = value if high is None else max(high, value)
    missing = bad = 0; seen = []
    for item in catalog:
        if item not in durations:
            missing += 1
        elif type(durations[item]) is not int:
            bad += 1
        else:
            seen.append(durations[item])
    return {'partitions': parts, 'rows': rows, 'timespent_min': low, 'timespent_max': high, 'timespent_zero_rows': zero,
            'timespent_zero_share': zero / rows if rows else 0.0, 'timespent_non_int_rows': non_int, 'rows_outside_catalog': outside,
            'catalog_items': len(catalog), 'duration_min': min(seen) if seen else None, 'duration_max': max(seen) if seen else None,
            'catalog_items_without_duration': missing, 'duration_non_int_items': bad}


def range_violations(report):
    """Human-readable defects of a ``scan_ranges`` report; an empty list means the gate for ``prepare-stats`` is open.

    Required: at least one row; every timespent an int in 0..255 (0 is legal); no row of an item outside the catalog; every catalog item has an int
    duration in 1..255.
    """
    found = []
    if report['rows'] == 0:
        found.append('no train rows')
    if report['timespent_non_int_rows']:
        found.append(f"{report['timespent_non_int_rows']} rows with a non-int timespent")
    if report['timespent_min'] is not None and (report['timespent_min'] < 0 or report['timespent_max'] > TIMESPENT_MAX):
        found.append(f"timespent range [{report['timespent_min']},{report['timespent_max']}] outside [0,{TIMESPENT_MAX}]")
    if report['rows_outside_catalog']:
        found.append(f"{report['rows_outside_catalog']} rows of items outside the catalog")
    if report['catalog_items_without_duration']:
        found.append(f"{report['catalog_items_without_duration']} catalog items without a duration")
    if report['duration_non_int_items']:
        found.append(f"{report['duration_non_int_items']} catalog items with a non-int duration")
    if report['duration_min'] is not None and (report['duration_min'] < DURATION_MIN or report['duration_max'] > DURATION_MAX):
        found.append(f"duration range [{report['duration_min']},{report['duration_max']}] outside [{DURATION_MIN},{DURATION_MAX}]")
    return found


def lenient_durations(path, catalog):
    """Raw ``duration`` values of the catalog items from items_metadata.parquet WITHOUT range checks (so ``range-check`` can report the extremes).

    A duplicated catalog item still raises ValueError. The strict reader for real use is ``history_side.load_durations``.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=['item_id', 'duration'])
    wanted = set(catalog); durations = {}
    for item, duration in zip(table.column('item_id').to_pylist(), table.column('duration').to_pylist()):
        if item in wanted:
            if item in durations:
                raise ValueError('duplicate item metadata')
            durations[item] = duration
    return durations
