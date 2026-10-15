"""FEAT-04b FEAT-002: ITEM_STATS_V1 (item_stats.py): range scan, builder, cutoff, ratio, timespent==0, alignment, sidecar, diagnostics.
``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); tests/test_feat04b_mutations.py re-runs them under
deliberately wrong implementations.
"""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat02_model as tm  # noqa: E402
ensure = tm.ensure


def row(item, timespent=5, like=False):
    return {'user_id': 1, 'item_id': item, 'timespent': timespent, 'like': like}


def raises(exception, call, *args):
    try:
        call(*args)
    except exception:
        return True
    return False


def m():
    from vkgr import item_stats
    return item_stats


def guard_stats_range_scan():
    """scan_ranges reports the 0 count, the extremes, non-int values, rows outside the catalog and the catalog durations (read-only)."""
    partitions = [(0, [row(1, 0), row(2, 255), row(3, 7)]), (1, [row(1, 12), row(99, 256), row(2, True), row(3, '5'), row(3, -1)])]
    durations = {1: 10, 2: 255, 3: 1}
    report = m().scan_ranges(partitions, [1, 2, 3], durations)
    ensure(report['partitions'] == 2 and report['rows'] == 8, 'row/partition counts')
    ensure(report['timespent_min'] == -1 and report['timespent_max'] == 256, 'extremes cover out-of-range values')
    ensure(report['timespent_zero_rows'] == 1 and report['timespent_zero_share'] == 1 / 8, 'zero count and share')
    ensure(report['timespent_non_int_rows'] == 2, 'bool and str are not ints')
    ensure(report['rows_outside_catalog'] == 1, 'item 99 is outside the catalog')
    ensure((report['catalog_items'], report['duration_min'], report['duration_max']) == (3, 1, 255), 'catalog durations')
    ensure(report['catalog_items_without_duration'] == 0 and report['duration_non_int_items'] == 0, 'duration coverage')
    gap = m().scan_ranges([(0, [row(1, 3)])], [1, 2], {1: 10, 3: 5})
    ensure(gap['catalog_items_without_duration'] == 1 and (gap['duration_min'], gap['duration_max']) == (10, 10), 'durations only from catalog items')
    odd = m().scan_ranges([(0, [row(1, 3)])], [1, 2], {1: 10, 2: 7.5})
    ensure(odd['duration_non_int_items'] == 1 and (odd['duration_min'], odd['duration_max']) == (10, 10), 'non-int duration is counted, not scanned')
    clean = m().scan_ranges([(0, [row(1, 0), row(2, 255)])], [1, 2], {1: 1, 2: 255})
    ensure(clean['timespent_zero_rows'] == 1 and (clean['timespent_min'], clean['timespent_max']) == (0, 255), 'boundaries 0 and 255 are legal')
    empty = m().scan_ranges([], [1], {1: 5})
    ensure(empty['rows'] == 0 and empty['timespent_min'] is None and empty['timespent_zero_share'] == 0.0, 'empty input')
    # read-only: the input rows are not modified
    before = [(w, [dict(r) for r in rows]) for w, rows in partitions]
    m().scan_ranges(partitions, [1, 2, 3], durations)
    ensure(partitions == before, 'scan_ranges must not modify its input')


CATALOG = [10, 20, 30]
DURATIONS = {10: 10, 20: 4, 30: 255}


def world():
    """Hand-computed fixture (weeks need not be consecutive). Per item: (n, first_week, likes, ratio_pm_sum).
    10: ts 5/10 -> 500, ts 0 -> 0, ts 30/10 -> capped 2000          => (3, 0, 1, 2500)
    20: ts 4/4 -> 1000, ts 8/4 -> 2000                              => (2, 0, 1, 3000)
    30: ts 255/255 -> 1000, ts 0 -> 0                               => (2, 1, 1, 1000)
    """
    return [(0, [row(10, 5, True), row(20, 4, False), row(10, 0, False)]),
            (1, [row(30, 255, True), row(10, 30, False)]),
            (3, [row(20, 8, True), row(30, 0, False)])]


EXPECTED = {'n': [3, 2, 2], 'first_week': [0, 0, 1], 'likes': [1, 1, 1], 'ratio_pm_sum': [2500, 3000, 1000]}


def build(partitions=None, catalog=CATALOG, durations=DURATIONS):
    return m().build_item_stats(world() if partitions is None else partitions, catalog, durations)


def build_ok(*args):
    """build() for inputs that must be accepted: a ValueError (e.g. from a mutant that refuses them) is a guard failure, i.e. an AssertionError."""
    try:
        return build(*args)
    except ValueError as error:
        raise AssertionError(f'valid input refused: {error}') from error


def guard_stats_formula():
    """build_item_stats equals the hand-computed table; every malformed input raises ValueError (including a validation-week partition)."""
    built = build_ok()
    for key, expected in EXPECTED.items():
        ensure(built[key] == expected, f'{key}: {built[key]} != {expected}')
    ensure(built['rows_total'] == 7 and built['cutoff_week'] == 24 and built['stats_version'] == 'ITEM_STATS_V1', 'totals and identity')
    ensure(all(type(v) is int for key in EXPECTED for v in built[key]), 'columns hold plain ints')
    # an independent per-row brute force agrees on a randomized world
    import random
    rng = random.Random(5); catalog = list(range(40)); durations = {i: rng.randint(1, 255) for i in catalog}
    parts = [(w, [row(rng.choice(catalog), rng.randint(0, 255), rng.random() < .3) for _ in range(150)]) for w in range(0, 25, 2)]
    got = build_ok(parts, catalog, durations)
    for k, item in enumerate(catalog):
        mine = [(w, r) for w, rows in parts for r in rows if r['item_id'] == item]
        if not mine:
            continue
        ensure(got['n'][k] == len(mine) and got['first_week'][k] == min(w for w, _ in mine), 'n / first_week')
        ensure(got['likes'][k] == sum(r['like'] for _, r in mine), 'likes')
        ensure(got['ratio_pm_sum'][k] == sum(min(2000, 1000 * r['timespent'] // durations[item]) for _, r in mine), 'ratio_pm_sum')
    def full(week):                                                    # every catalog item once, so only the named defect can raise
        return week, [row(10), row(20), row(30)]
    ensure(build_ok([full(0), full(1)])['n'] == [2, 2, 2], 'the padding fixture itself is valid')
    bad = [([full(0), full(25)], 'week 25 (validation) partition'), ([full(0), full(26)], 'week 26'), ([full(1), full(1)], 'repeated week'),
           ([full(2), full(1)], 'non-monotone weeks'), ([full(-1)], 'negative week'), ([(True, [row(10), row(20), row(30)])], 'bool week'),
           ([full(0), (1, [row(999)])], 'item outside the catalog'), ([(0, [row(10), row(20), row(30, 1, 1)])], 'like not bool')]
    for partitions, label in bad:
        ensure(raises(ValueError, build, partitions), label)
    ensure(raises(ValueError, build, world(), CATALOG + [40]), 'a catalog item with n == 0 is refused')
    ensure(raises(ValueError, build, world(), [10, 10, 20, 30]), 'duplicate catalog item')


def guard_stats_cutoff():
    """Week 24 is the last accepted week; 25 and 26 are refused whatever else the partitions hold."""
    ok = world() + [(24, [row(10, 3, True)])]
    ensure(build_ok(ok)['n'][0] == 4, 'week 24 is accepted')
    for week in (25, 26, 30):
        ensure(raises(ValueError, build, world() + [(week, [row(10)])]), f'week {week} must be refused')
    ensure(m().STAT_CUTOFF_WEEK == 24, 'cutoff constant')


def guard_stats_ratio_integer():
    """The ratio is an integer permille: floor division, capped at 2000 (a row with timespent >> duration cannot dominate the sum)."""
    ensure(m().ratio_pm(255, 1) == 2000 and type(m().ratio_pm(255, 1)) is int, 'capped at 2000')
    ensure(m().ratio_pm(2, 1) == 2000 and m().ratio_pm(3, 1) == 2000, 'cap boundary')
    ensure(m().ratio_pm(1, 3) == 333 and type(m().ratio_pm(1, 3)) is int, 'floor, integer')
    ensure(m().ratio_pm(2, 3) == 666 and m().ratio_pm(10, 10) == 1000, 'values')
    built = build_ok([(0, [row(10, 255, False)])], [10], {10: 1})
    ensure(built['ratio_pm_sum'] == [2000] and type(built['ratio_pm_sum'][0]) is int, 'the aggregate carries the capped integer')


def guard_stats_timespent_zero():
    """timespent == 0 is a legal row: counted in n, contributes 0 to the ratio sum; 256 and -1 (and non-ints) are refused."""
    built = build_ok([(0, [row(10, 0, False), row(10, 0, True)])], [10], {10: 7})
    ensure(built['n'] == [2] and built['likes'] == [1] and built['ratio_pm_sum'] == [0] and built['first_week'] == [0], 'zero rows are counted')
    ensure(m().ratio_pm(0, 7) == 0, 'ratio of a zero timespent')
    for bad in (256, -1, 5.0, True, '5', None):
        ensure(raises(ValueError, build, [(0, [row(10, bad)])], [10], {10: 7}), f'timespent {bad!r} must be refused')
    for bad in (0, 256, 5.0, True, None):
        ensure(raises(ValueError, build, [(0, [row(10, 3)])], [10], {10: bad}), f'duration {bad!r} must be refused')
    ensure(raises(ValueError, build, [(0, [row(10, 3)])], [10], {}), 'missing duration')
    ensure(raises(ValueError, build, [(0, [row(10, 3)])], [10, 20], {10: 7, 20: 0}), 'a bad duration of a catalog item without rows still fails')


def guard_stats_alignment():
    """Columns follow the catalog order: a permuted catalog yields the same table permuted."""
    base = build_ok()
    for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        catalog = [CATALOG[k] for k in perm]
        got = build_ok(world(), catalog, DURATIONS)
        for key in EXPECTED:
            ensure(got[key] == [base[key][k] for k in perm], f'{key} not aligned with the permuted catalog')
    guard_stats_bucket_alignment()                                          # the same alignment, one step later: catalog order -> model index space


# ---- C/D bucket functions of retrieval_feat04b. The guard keeps its OWN copy of the audited grid (= configs/feat04b/stats-bucket-constants.json): a drifted edge constant is caught here.
GRID = {'expo': (4, 8, 16, 32, 64, 128, 256, 512), 'first': (1, 3, 6, 10, 15, 20, 23), 'ratio': (200, 350, 500, 650, 800, 1000), 'like': (10, 20, 40, 80, 160)}
GRID_K, GRID_P0 = 10, 28                                  # LIKE_P0 = 28 is the audited global like rate in permille
AUDIT_JSON = Path(__file__).resolve().parents[1] / 'configs' / 'feat04b' / 'stats-bucket-constants.json'
AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'feat04b_audit_constants.py'
def ref_expo(n):
    return 1 + sum(1 for e in GRID['expo'] if n >= e)


def ref_first(week):
    return 1 + sum(1 for e in GRID['first'] if week >= e)


def ref_ratio(total, n):
    return 1 + sum(1 for e in GRID['ratio'] if total >= e * n)


def ref_like(likes, n):
    return 1 + sum(1 for e in GRID['like'] if 1000 * likes + GRID_K * GRID_P0 >= e * (n + GRID_K))


def m4():
    from vkgr import retrieval_feat04b
    return retrieval_feat04b


def guard_stats_bucket_boundaries():
    """Every edge +-1 against an independent reference (own copy of the grid): expo / first / ratio (integer form S >= e*n) / like (smoothed permille with K and P0)."""
    f = m4()
    for e in GRID['expo']:
        for n in (e - 1, e, e + 1):
            ensure(f.expo_bucket(n) == ref_expo(n) and type(f.expo_bucket(n)) is int, f'expo_bucket({n})')
    ensure(f.expo_bucket(1) == 1 and f.expo_bucket(10 ** 6) == len(GRID['expo']) + 1, 'expo range 1..len+1')
    for week in range(0, 25):
        ensure(f.first_bucket(week) == ref_first(week), f'first_bucket({week})')
    ensure(f.first_bucket(0) == 1 and f.first_bucket(24) == len(GRID['first']) + 1, 'first range')
    for e in GRID['ratio']:
        for n in (1, 3, 7, 100):
            for total in (e * n - 1, e * n, e * n + 1):
                ensure(f.ratio_bucket4(total, n) == ref_ratio(total, n), f'ratio_bucket4({total},{n})')
    ensure(f.ratio_bucket4(0, 5) == 1 and f.ratio_bucket4(2000 * 5, 5) == len(GRID['ratio']) + 1, 'ratio range')
    for n in (1, 2, 5, 9, 10, 40, 77, 300, 1000):
        for likes in sorted({0, 1, n // 7, n // 3, n // 2, n - 1, n} & set(range(n + 1))):
            ensure(f.like_bucket(likes, n) == ref_like(likes, n), f'like_bucket({likes},{n})')
        for e in GRID['like']:                                  # the exact crossing point of each edge for this n
            for likes in range(0, n + 1):
                ensure(f.like_bucket(likes, n) == ref_like(likes, n), f'like_bucket({likes},{n}) at edge {e}')
    ensure(f.like_bucket(0, 1) == ref_like(0, 1) and f.like_bucket(1, 1) == ref_like(1, 1), 'n = 1')
    ensure((f.LIKE_K, f.LIKE_P0) == (GRID_K, GRID_P0), 'smoothing constants')


def guard_stats_constants_match_audit():
    """The audited constants: stats-bucket-constants.json (the audit stdout, byte for byte canonical) == the code constants == this file's own grid copy; the merged histograms
    satisfy the 2% rule and add up to the training examples; the audit's report threshold is the one in the audit script."""
    import json
    import re
    from vkgr import item_stats
    f = m4()
    raw = AUDIT_JSON.read_text()
    doc = json.loads(raw)
    ensure(raw == json.dumps(doc, sort_keys=True, separators=(',', ':')) + '\n', 'the constants file must be the audit stdout verbatim (canonical json plus a newline)')
    for field in ('sum_likes', 'n_train_examples', 'sum_likes_minus_n_train_examples', 'likes_vs_target_freq', 'timespent', 'duration', 'stats_self_inclusion', 'p_like_report_threshold',
                  'p_like_exceeds_threshold', 'logq_redundancy', 'grids', 'like_p0', 'stats_sha256'):
        ensure(field in doc, f'AC16 field missing from the constants file: {field}')
    ensure(doc['train_only'] is True and doc['cutoff_week'] == item_stats.STAT_CUTOFF_WEEK, 'train-only audit at the code cutoff week')
    for name, attr in (('expo', 'EXPO_EDGES'), ('first', 'FIRST_EDGES'), ('ratio', 'RATIO_EDGES'), ('like', 'LIKE_EDGES')):
        edges = tuple(doc['grids'][name]['final_edges'])
        ensure(getattr(f, attr) == edges, f'{attr} differs from the audit: {getattr(f, attr)} != {edges}')
        ensure(GRID[name] == edges, f'the guard grid copy of {name} differs from the audit')
        counts = doc['grids'][name]['final_counts']
        ensure(len(counts) == len(edges) + 1 and sum(counts) == doc['n_train_examples'], f'{name}: merged histogram does not add up')
        ensure(min(counts) >= 0.02 * doc['n_train_examples'], f'{name}: a merged bucket is below the 2% rule')
    ensure((f.LIKE_K, f.LIKE_P0, f.STATS_SHA256) == (doc['like_k'], doc['like_p0'], doc['stats_sha256']), 'LIKE_K / LIKE_P0 / STATS_SHA256 differ from the audit')
    ensure((GRID_K, GRID_P0) == (doc['like_k'], doc['like_p0']), 'the guard copy of K / P0 differs from the audit')
    ensure(doc['like_p0'] == round(1000 * doc['sum_likes'] / doc['rows_total']), 'like_p0 is the rounded global like rate')
    ensure(len(f.STATS_SHA256) == 64 and set(f.STATS_SHA256) <= set('0123456789abcdef'), 'STATS_SHA256 must be a sha256 hex digest')
    match = re.search(r'^P_LIKE_REPORT_THRESHOLD\s*=\s*([0-9.]+)', AUDIT_SCRIPT.read_text(), re.M)
    ensure(match is not None and float(match.group(1)) == doc['p_like_report_threshold'], 'the audit threshold differs from the audit script constant')
    ensure(doc['p_like_exceeds_threshold'] == (doc['stats_self_inclusion']['p_like'] > doc['p_like_report_threshold']), 'p_like_exceeds_threshold is p_like > threshold')
def guard_stats_bucket_alignment():
    """stat_bucket_ids indexes by idmap (the model's index space), not by the raw item id or the catalog position; PAD row 0 stays 0; a permuted catalog gives the same table."""
    import torch
    f = m4()
    catalog = [50, 7, 33, 12]; idmap = {50: 3, 7: 1, 33: 4, 12: 2}
    stats = {'n': [1, 40, 600, 5], 'first_week': [0, 5, 24, 11], 'likes': [0, 12, 30, 4], 'ratio_pm_sum': [100, 40 * 700, 600 * 1500, 5 * 400]}
    ids = f.stat_bucket_ids(stats, catalog, idmap, 5)
    ensure(set(ids) == {'xb', 'fb', 'rb', 'lb'}, 'four tensors')
    for name, tensor in ids.items():
        ensure(tensor.dtype == torch.long and tensor.shape == (5,) and int(tensor[0]) == 0, f'{name}: long vector of the index space with PAD row 0')
    for k, item in enumerate(catalog):
        row = idmap[item]
        got = (int(ids['xb'][row]), int(ids['fb'][row]), int(ids['rb'][row]), int(ids['lb'][row]))
        want = (ref_expo(stats['n'][k]), ref_first(stats['first_week'][k]), ref_ratio(stats['ratio_pm_sum'][k], stats['n'][k]), ref_like(stats['likes'][k], stats['n'][k]))
        ensure(got == want, f'item {item}: bucket ids {got} != {want}')
    ensure(len({int(v) for v in ids['xb'][1:]}) == 4, 'fixture must use four different exposure buckets (a mis-indexed table is visible)')
    perm = [2, 0, 3, 1]
    again = f.stat_bucket_ids({k: [v[i] for i in perm] for k, v in stats.items()}, [catalog[i] for i in perm], idmap, 5)
    ensure(all(torch.equal(again[k], ids[k]) for k in ids), 'permuting catalog and stats together must not change the index-space tables')
    ensure(raises(ValueError, f.stat_bucket_ids, {**stats, 'n': stats['n'][:3]}, catalog, idmap, 5), 'a short column is refused')
    ensure(raises(ValueError, f.stat_bucket_ids, stats, catalog, {**idmap, 12: 9}, 5), 'an index outside the table is refused')
    ensure(raises(ValueError, f.stat_bucket_ids, stats, catalog, {50: 3, 7: 1, 33: 4}, 5), 'an item missing from idmap is refused')


def guard_stats_flip_increment():
    """L6: flipping the timespent of one week<=24 row moves the table by exactly the documented increment and nothing else."""
    base = build_ok()
    partitions = world(); partitions[0][1][0]['timespent'] = 20          # item 10: ts 5 -> 20  (500 -> 2000)
    after = build_ok(partitions)
    ensure(after['ratio_pm_sum'] == [base['ratio_pm_sum'][0] + 1500] + base['ratio_pm_sum'][1:], 'exact ratio increment')
    for key in ('n', 'first_week', 'likes'):
        ensure(after[key] == base[key], f'{key} must not move')


def guard_stats_signature_takes_no_query_input():
    """L10: the constructor entry points accept exactly (partitions, catalog, durations)."""
    import inspect
    for fn in (m().aggregate_rows, m().build_item_stats, m().scan_ranges):
        ensure(list(inspect.signature(fn).parameters) == ['partitions', 'catalog', 'durations'], f'{fn.__name__} signature')


# ---- offline diagnostics: hand-computed world with toy bucket functions (the real ones live in retrieval_feat04b, written later)
CATALOG2 = [10, 20, 30, 40]
DURATIONS2 = {10: 10, 20: 4, 30: 255, 40: 20}
BUCKETS = {'expo': lambda n: 1 + int(n >= 3), 'first': lambda f: 1 + int(f >= 1),
           'ratio': lambda S, n: 1 + int(S >= 500 * n) + int(S >= 1000 * n),
           'like': lambda l, n: 1 + int(1000 * l + 100 >= 250 * (n + 1))}                  # toy P0 = 100, K = 1, one edge at 250: constants fixed for with/without
EXAMPLES = [{'target': 10, 'target_position': [0, 0]}, {'target': 30, 'target_position': [1, 0]}, {'target': 20, 'target_position': [3, 0]},
            {'target': 10, 'target_position': [1, 1]}, {'target': 20, 'target_position': [0, 1]}, {'target': 40, 'target_position': [3, 2]}]


def world2():
    """world() plus a one-row item 40 (week 3, ordinal 2, ts 10 of 20 -> 500, a like). Positions: 10: (0,0) (0,2) (1,1); 20: (0,1) (3,0); 30: (1,0) (3,1); 40: (3,2).
    Stats: 10 (3,0,1,2500); 20 (2,0,1,3000); 30 (2,1,1,1000); 40 (1,3,1,500)."""
    base = world()
    return [base[0], base[1], (3, base[2][1] + [row(40, 10, True)])]


def stats2():
    return m().build_item_stats(world2(), CATALOG2, DURATIONS2)


def guard_stats_self_inclusion_formula():
    """Removing the target's own row changes only that item's n / likes / ratio_pm_sum / first_week (constants and edges fixed); hand-computed, N = 6.
    expo 10:3->2 changed, 30:2->1, 20:2->1 same, 10 again changed, 20 same, 40:1->0 same            => 2/6
    ratio (2500,3)->(2000,2) changed; (1000,2)->(0,1) changed; (3000,2)->(1000,1) same; (2500,3)->(500,2) changed; (3000,2)->(2000,1) same;
          40: n-1 == 0 is undefined => changed                                                          => 4/6
    like 10:2->1 changed (own row is a like), 30 changed, 20 changed, 10 (own row not a like) same, 20 same, 40:2->1 changed => 4/6
    first 10: own is first, the next row is also week 0 => same; 30: 1->3 both bucket 2 => same; 20 (0,1) own first -> week 3 => changed; 40 undefined => changed => 2/6
    own row is the item's earliest row: 10 (0,0) yes, 30 (1,0) yes, 20 (3,0) no, 10 (1,1) no, 20 (0,1) yes, 40 yes => 4/6; n <= 3: all 6/6; n == 1: 1/6."""
    got = m().stats_self_inclusion(world2(), EXAMPLES, CATALOG2, DURATIONS2, stats2(), BUCKETS)
    expected = {'examples': 6, 'p_expo': 2 / 6, 'p_ratio': 4 / 6, 'p_like': 4 / 6, 'p_first': 2 / 6, 'p_first_is_own_row': 4 / 6, 'share_n_le_3': 6 / 6, 'share_n_eq_1': 1 / 6}
    for key, value in expected.items():
        ensure(got[key] == value, f'{key}: {got[key]} != {value}')
    ensure(got['counts'] == {'expo': 2, 'ratio': 4, 'like': 4, 'first': 2, 'first_is_own_row': 4, 'n_le_3': 6, 'n_eq_1': 1}, f"counts {got['counts']}")
    # a row of another item at the target position is refused, as is a target outside the catalog
    wrong = [{'target': 20, 'target_position': [0, 0]}]
    ensure(raises(ValueError, m().stats_self_inclusion, world2(), wrong, CATALOG2, DURATIONS2, stats2(), BUCKETS), 'target row item mismatch')
    ensure(raises(ValueError, m().stats_self_inclusion, world2(), [{'target': 99, 'target_position': [0, 0]}], CATALOG2, DURATIONS2, stats2(), BUCKETS), 'target outside catalog')
    ensure(raises(ValueError, m().stats_self_inclusion, world2(), [{'target': 10, 'target_position': [9, 9]}], CATALOG2, DURATIONS2, stats2(), BUCKETS), 'position absent from the partitions')
    other = stats2(); other['first_week'][0] = 2
    ensure(raises(ValueError, m().stats_self_inclusion, world2(), EXAMPLES, CATALOG2, DURATIONS2, other, BUCKETS), 'stats that do not match the partitions')
    # removing a row never touches the shared constants: the toy functions are called with the same closure for the with/without sides
    seen = []
    spy = {k: (lambda *a, f=f: (seen.append(a), f(*a))[1]) for k, f in BUCKETS.items()}
    m().stats_self_inclusion(world2(), EXAMPLES[:1], CATALOG2, DURATIONS2, stats2(), spy)
    ensure((3,) in seen and (2,) in seen and (0,) in seen and (2500, 3) in seen and (2000, 2) in seen and (1, 3) in seen and (0, 2) in seen,
           'with/without stats are bucketed by the same functions')


def guard_target_bucket_hist():
    train_expected = {'expo': {1: 4, 2: 2}, 'first': {1: 4, 2: 2}, 'ratio': {2: 4, 3: 2}, 'like': {2: 6}}
    val_expected = {'expo': {1: 2, 2: 1}, 'first': {1: 2, 2: 1}, 'ratio': {2: 2, 3: 1}, 'like': {2: 3}}
    queries = [{'target': 20}, {'target': 10}, {'target': 99}, {'target': 40}]
    got = m().target_bucket_hist(EXAMPLES, queries, CATALOG2, stats2(), BUCKETS)
    ensure(got['train']['examples'] == 6 and got['validation']['queries'] == 4, 'sizes')
    ensure(got['validation']['in_catalog'] == 3 and got['validation']['outside_catalog'] == 1, 'targets outside the catalog are counted, not bucketed')
    for name in train_expected:
        ensure(got['train'][name] == train_expected[name], f"train {name}: {got['train'][name]}")
        ensure(got['validation'][name] == val_expected[name], f"validation {name}: {got['validation'][name]}")


def guard_logq_redundancy():
    import math
    counts = {10: 5, 20: 3, 30: 1, 40: 2}
    got = m().logq_redundancy(stats2(), CATALOG2, {10, 20, 30, 40}, counts, BUCKETS)
    ensure(abs(got['expo_spearman'] - 3 / math.sqrt(15)) < 1e-12, f"expo spearman {got['expo_spearman']}")
    ensure(got['like_spearman'] is None, 'a constant bucket column has no rank correlation')
    ensure(got['supported_items'] == 4, 'support size')
    only = m().logq_redundancy(stats2(), CATALOG2, {10, 20}, counts, BUCKETS)
    ensure(only['supported_items'] == 2 and abs(only['expo_spearman'] - 1.0) < 1e-12, 'two items, concordant ranks (10: bucket 2 / count 5; 20: bucket 1 / count 3)')
    zero = m().logq_redundancy(stats2(), CATALOG2, {10, 20, 30}, {10: 5, 20: 3}, BUCKETS)            # an item with no target example has count 0
    ensure(zero['supported_items'] == 3 and zero['expo_spearman'] is not None, 'missing count means 0')
    ensure(raises(ValueError, m().logq_redundancy, stats2(), CATALOG2, {10, 99}, counts, BUCKETS), 'support outside the catalog')


class BucketTests(unittest.TestCase):
    def test_bucket_boundaries(self):
        guard_stats_bucket_boundaries()

    def test_bucket_alignment(self):
        guard_stats_bucket_alignment()
    def test_constants_match_audit(self):
        guard_stats_constants_match_audit()


class DiagnosticsTests(unittest.TestCase):
    def test_self_inclusion_formula(self):
        guard_stats_self_inclusion_formula()

    def test_target_bucket_hist(self):
        guard_target_bucket_hist()

    def test_logq_redundancy(self):
        guard_logq_redundancy()

    def test_spearman_ties_and_degenerate(self):
        self.assertAlmostEqual(m().spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(m().spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(m().spearman([1, 1, 1], [1, 2, 3]))
        with self.assertRaises(ValueError):
            m().spearman([1, 2], [1])

    def test_diagnostics_do_not_modify_their_inputs(self):
        import copy
        parts, stats = world2(), stats2()
        before = (copy.deepcopy(parts), copy.deepcopy(stats), copy.deepcopy(EXAMPLES))
        m().stats_self_inclusion(parts, EXAMPLES, CATALOG2, DURATIONS2, stats, BUCKETS)
        m().target_bucket_hist(EXAMPLES, [{'target': 10}], CATALOG2, stats, BUCKETS)
        self.assertEqual((parts, stats, EXAMPLES), before)

    def test_item_stats_does_not_import_the_model_module(self):
        source = (Path(__file__).resolve().parents[1] / 'src' / 'vkgr' / 'item_stats.py').read_text()
        self.assertNotIn('retrieval_feat04b', source)


METADATA_SHA, HISTORY_SHA = 'a' * 64, 'b' * 64
SUPPORT = {10, 30}


def write_sidecar(tmp, **overrides):
    args = dict(output=Path(tmp) / 'stats', partitions=world(), catalog=CATALOG, support=SUPPORT, durations=DURATIONS,
                items_metadata_sha256=METADATA_SHA, history_sha256=HISTORY_SHA, binding={'synthetic': True, 'test_gate': 'closed_hash_only'})
    args.update(overrides)
    return m().write_item_stats(**args)


def load_sidecar(tmp, receipt, **overrides):
    args = dict(path=Path(tmp) / 'stats' / 'manifest.json', digest=receipt['manifest_sha256'], items_metadata_sha256=METADATA_SHA,
                history_sha256=HISTORY_SHA, catalog=CATALOG, support=SUPPORT)
    args.update(overrides)
    return m().load_item_stats(**args)


def rewrite_manifest(tmp, **changes):
    """Edit the manifest in place and return its new sha256 (so only the internal consistency checks can object)."""
    import json
    from vkgr.data_audit import file_sha256
    path = Path(tmp) / 'stats' / 'manifest.json'
    value = json.loads(path.read_text()); value.update(changes)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
    return file_sha256(path)


class StatsSidecarTests(unittest.TestCase):
    def test_round_trip_equals_the_build_and_carries_no_bucket_constants(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            receipt = write_sidecar(tmp)
            loaded = load_sidecar(tmp, receipt)
            for key in EXPECTED:
                self.assertEqual(loaded['stats'][key], EXPECTED[key])
            self.assertEqual(loaded['stats'], build())
            manifest = receipt['manifest']
            self.assertEqual((manifest['kind'], manifest['stats_version'], manifest['cutoff_week'], manifest['rows_total'], manifest['test_gate']),
                             ('item_stats_prepare', 'ITEM_STATS_V1', 24, 7, 'closed_hash_only'))
            self.assertEqual((manifest['catalog_items'], manifest['supported_items']), (3, 2))
            self.assertEqual(loaded['stats_sha256'], manifest['stats_sha256'])
            self.assertEqual(loaded['stats_sha256'], m().stats_digest(loaded['stats']))
            text = ''.join((Path(tmp) / 'stats' / name).read_text().lower() for name in ('manifest.json', 'stats.json'))
            for fragment in ('edge', 'bucket', 'like_p0', 'like_k', 'expo_', 'ratio_edges'):
                self.assertNotIn(fragment, text, 'the sidecar must hold no bucket constants')
            self.assertEqual(sorted(json.loads((Path(tmp) / 'stats' / 'stats.json').read_text())), sorted(EXPECTED))

    def test_stats_digest_depends_on_content_only(self):
        a = build(); b = build(); b['rows_total'] = 999
        self.assertEqual(m().stats_digest(a), m().stats_digest(b))
        c = build(); c['likes'] = [1, 1, 2]
        self.assertNotEqual(m().stats_digest(a), m().stats_digest(c))

    def test_expected_sha256_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = write_sidecar(tmp)
            digest = receipt['manifest']['stats_sha256']
            load_sidecar(tmp, receipt, expected_sha256=digest)
            with self.assertRaises(ValueError):
                load_sidecar(tmp, receipt, expected_sha256='0' * 64)

    def test_bindings_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = write_sidecar(tmp)
            for label, override in (('metadata', {'items_metadata_sha256': 'c' * 64}), ('history', {'history_sha256': 'c' * 64}),
                                    ('catalog order', {'catalog': [CATALOG[1], CATALOG[0], CATALOG[2]]}), ('catalog items', {'catalog': CATALOG + [40]}),
                                    ('support', {'support': {10}}), ('support outside catalog', {'support': {10, 99}}),
                                    ('manifest sha', {'digest': '0' * 64})):
                with self.assertRaises(ValueError, msg=label):
                    load_sidecar(tmp, receipt, **override)
            load_sidecar(tmp, receipt)

    def test_internal_consistency_is_checked(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            receipt = write_sidecar(tmp)
            for label, changes in (('cutoff', {'cutoff_week': 25}), ('rows_total', {'rows_total': 8}), ('kind', {'kind': 'item_attr_prepare'}),
                                   ('version', {'stats_version': 'ITEM_STATS_V2'}), ('gate', {'test_gate': 'open'}), ('stats sha', {'stats_sha256': '0' * 64}),
                                   ('fields', {'fields': ['n']})):
                with tempfile.TemporaryDirectory() as other:
                    write_sidecar(other)
                    digest = rewrite_manifest(other, **changes)
                    with self.assertRaises(ValueError, msg=label):
                        m().load_item_stats(Path(other) / 'stats' / 'manifest.json', digest, METADATA_SHA, HISTORY_SHA, CATALOG, SUPPORT)
            # an artifact edited after the fact fails its pinned hash
            stats = Path(tmp) / 'stats' / 'stats.json'
            value = json.loads(stats.read_text()); value['likes'][0] += 1
            stats.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
            with self.assertRaises(ValueError):
                load_sidecar(tmp, receipt)

    def test_semantically_impossible_columns_are_refused(self):
        """n == 0, first_week > cutoff, likes > n, a ratio sum above 2000 * n: refused even when every hash is consistent."""
        import json
        from vkgr.data_audit import file_sha256
        for label, edit in (('n zero', lambda v: v['n'].__setitem__(0, 0)), ('first week 25', lambda v: v['first_week'].__setitem__(0, 25)),
                            ('likes above n', lambda v: v['likes'].__setitem__(0, 99)), ('ratio above cap', lambda v: v['ratio_pm_sum'].__setitem__(0, 2000 * 3 + 1)),
                            ('short column', lambda v: v['n'].pop()), ('float', lambda v: v['likes'].__setitem__(0, 1.0))):
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                write_sidecar(tmp)
                stats = Path(tmp) / 'stats' / 'stats.json'
                value = json.loads(stats.read_text()); edit(value)
                stats.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
                manifest = json.loads((Path(tmp) / 'stats' / 'manifest.json').read_text())
                manifest['artifacts']['stats.json'] = file_sha256(stats)
                digest = rewrite_manifest(tmp, artifacts=manifest['artifacts'], stats_sha256=m().stats_digest(value), rows_total=sum(value['n']))   # hashes consistent: only the column checks can object
                with self.assertRaises(ValueError):
                    m().load_item_stats(Path(tmp) / 'stats' / 'manifest.json', digest, METADATA_SHA, HISTORY_SHA, CATALOG, SUPPORT)

    def test_write_refuses_overwrite_and_validation_bound_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp)
            with self.assertRaises(FileExistsError):
                write_sidecar(tmp)                                                # an existing output directory is never overwritten
        for binding in ({'query_sha': 'x'}, {'nested': {'validation_queries': 1}}, {'raw_records': [{'Validation': 1}]}):
            with self.subTest(binding), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    write_sidecar(tmp, binding=binding)                           # L10: no validation / query digest may enter the binding
                self.assertFalse((Path(tmp) / 'stats').exists())

    def test_write_rejects_bad_partitions_without_leaving_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_sidecar(tmp, partitions=world() + [(25, [row(10)])])
            self.assertFalse((Path(tmp) / 'stats').exists())

    def test_manifest_has_no_validation_or_query_key(self):
        def keys(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    yield k; yield from keys(v)
            elif isinstance(value, list):
                for v in value:
                    yield from keys(v)
        with tempfile.TemporaryDirectory() as tmp:
            receipt = write_sidecar(tmp)
            for key in keys(receipt['manifest']):
                self.assertNotIn('query', key.lower()); self.assertNotIn('validation', key.lower())


class StatsBuilderTests(unittest.TestCase):
    def test_formula(self):
        guard_stats_formula()

    def test_cutoff(self):
        guard_stats_cutoff()

    def test_ratio_integer(self):
        guard_stats_ratio_integer()

    def test_timespent_zero(self):
        guard_stats_timespent_zero()

    def test_alignment(self):
        guard_stats_alignment()

    def test_flip_increment(self):
        guard_stats_flip_increment()

    def test_signature(self):
        guard_stats_signature_takes_no_query_input()

    def test_partitions_may_be_a_generator(self):
        self.assertEqual(build(iter(world()))['n'], [3, 2, 2])


class RangeScanTests(unittest.TestCase):
    def test_range_scan(self):
        guard_stats_range_scan()

    def test_range_violations_name_every_defect(self):
        def violations(parts, catalog=(1, 2), durations=None):
            return m().range_violations(m().scan_ranges(parts, catalog, {1: 10, 2: 255} if durations is None else durations))
        self.assertEqual(violations([(0, [row(1, 0), row(2, 255)])]), [])                       # 0 and 255 are legal
        for label, parts, kwargs in (('timespent 256', [(0, [row(1, 256)])], {}), ('timespent -1', [(0, [row(1, -1)])], {}),
                                     ('non-int timespent', [(0, [row(1, 2.5)])], {}), ('item outside catalog', [(0, [row(7, 3)])], {}),
                                     ('duration 0', [(0, [row(1, 3)])], {'durations': {1: 0, 2: 9}}), ('duration 256', [(0, [row(1, 3)])], {'durations': {1: 10, 2: 256}}),
                                     ('duration missing', [(0, [row(1, 3)])], {'durations': {1: 10}}), ('duration not int', [(0, [row(1, 3)])], {'durations': {1: 10, 2: 7.5}}),
                                     ('no rows', [], {})):
            found = violations(parts, **kwargs)
            self.assertTrue(found, label); self.assertTrue(all(type(v) is str for v in found), label)

    def test_lenient_durations_return_raw_values_and_refuse_duplicates(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'meta.parquet'
            pq.write_table(pa.table({'item_id': pa.array([1, 2, 9], pa.uint32()), 'author_id': pa.array([5, 5, 5], pa.uint32()), 'duration': pa.array([0, 255, 7], pa.uint8())}), str(path))
            self.assertEqual(m().lenient_durations(path, [1, 2]), {1: 0, 2: 255})                    # 0 is returned, not refused; item 9 is outside the catalog
            pq.write_table(pa.table({'item_id': pa.array([1, 1], pa.uint32()), 'duration': pa.array([3, 4], pa.uint8())}), str(path))
            with self.assertRaises(ValueError):
                m().lenient_durations(path, [1])

    def test_generator_partitions_are_consumed_once(self):
        report = m().scan_ranges(((w, iter([row(1, 4)])) for w in range(3)), [1], {1: 9})
        self.assertEqual((report['partitions'], report['rows']), (3, 3))


if __name__ == '__main__':
    unittest.main()
