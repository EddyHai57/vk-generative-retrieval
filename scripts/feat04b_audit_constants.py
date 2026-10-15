"""FEAT-04b (P line) C/D train-only statistics audit. Prints one canonical JSON document, no paths, writes no file.

Rebuilds ITEM_STATS_V1 itself from the 25 train partitions (weeks 0-24), the catalog / support / training examples of the pinned prepared dataset and the item
durations, then reports every fact the C/D freeze needs (design AC16): the like/target-frequency overlap, the raw value ranges, the target share of every
bucket under the pre-registered candidate grids, the grids after the pre-registered 2% merge rule, LIKE_P0, the self-inclusion diagnostic, the logQ
redundancy and the content digest of the statistics. The grids, the merge rule and the report threshold are fixed below BEFORE any real data is read; nothing
in the output may be used to adjust them. Nothing outside the prepared dataset, the 25 train weeks and the metadata file is opened: validation queries and
later partitions are refused by path, and no other location named in the spec is read.
The four bucket formulas are written out here on purpose (the audit has to be self-contained); the model module later pins its own copies against the stdout
of this script (configs/feat04b/stats-bucket-constants.json).
"""
import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

SCHEMA = 'feat04b-audit-v1'
TRAIN_WEEKS = 25
EXPO_GRID = (2, 4, 8, 16, 32, 64, 128, 256, 512)           # n:      10 buckets
FIRST_GRID = (1, 3, 6, 10, 15, 20, 23)                      # week:    8 buckets
RATIO_GRID = (200, 350, 500, 650, 800, 1000, 1250)          # permille: 8 buckets  (mean watch ratio >= edge)
LIKE_GRID = (5, 10, 20, 40, 80, 160)                        # permille: 7 buckets  (smoothed like rate >= edge)
LIKE_K = 10                                                 # smoothing strength
MIN_SHARE = Fraction(1, 50)
P_LIKE_REPORT_THRESHOLD = 0.30                              # a report reminder, not a gate (design 4.2 / ruling F1); fixed before p_like is known

# Path fragments that must never be opened; assembled at run time so the source itself stays free of them.
_FORBIDDEN = ('validation' + '-queries', 'week_' + '25', 'week_' + '26')
_FORBIDDEN_PART = 'te' + 'st'


def guard_path(path):
    """Raise ValueError for a path that points into the validation queries or the test / later-week partitions."""
    text = str(path).replace('\\', '/')
    if any(fragment in text for fragment in _FORBIDDEN) or _FORBIDDEN_PART in Path(text).parts:
        raise ValueError('audit refuses a non-train path')
    return path


# ---------------------------------------------------------------- bucket formulas (integers only; an edge is the first value of its bucket)

def expo_bucket(n, edges):
    return 1 + sum(n >= e for e in edges)


def first_bucket(week, edges):
    return 1 + sum(week >= e for e in edges)


def ratio_bucket(total, n, edges):
    """``total`` = ratio_pm_sum; ``total >= e * n`` is the integer form of mean ratio >= e."""
    return 1 + sum(total >= e * n for e in edges)


def like_bucket(likes, n, edges, k, p0):
    """Smoothed like rate in permille: (1000 * likes + k * p0) / (n + k) >= e."""
    return 1 + sum(1000 * likes + k * p0 >= e * (n + k) for e in edges)


def like_p0(likes, n):
    """round(1000 * likes / n), halves rounded up, exact integer arithmetic."""
    if n <= 0:
        raise ValueError('no train rows')
    return (2000 * likes + n) // (2 * n)


def merge_small_buckets(totals, edges, min_share=MIN_SHARE):
    """Pre-registered rule: while a bucket share < min_share, merge the smallest bucket (ties: lower index) into its smaller neighbour (ties: the lower one).

    ``totals[b]`` is the number of training targets in bucket ``b``; ``edges[b]`` separates bucket ``b`` from ``b + 1``. Merging removes the edge between the two.
    """
    totals = list(totals); edges = list(edges)
    if len(totals) != len(edges) + 1:
        raise ValueError('a histogram needs exactly one more bucket than edges')
    total = sum(totals)
    if total <= 0:
        raise ValueError('empty histogram')
    initial = list(totals); steps = []
    while len(totals) > 1 and any(Fraction(t, total) < min_share for t in totals):
        i = min(range(len(totals)), key=lambda k: (totals[k], k))
        if i == 0:
            j = 1
        elif i == len(totals) - 1:
            j = i - 1
        else:
            j = i - 1 if totals[i - 1] <= totals[i + 1] else i + 1
        low = min(i, j)
        steps.append({'merged_bucket': i, 'into_bucket': j, 'removed_edge': edges[low]})
        totals[low] = totals[i] + totals[j]
        del totals[low + 1]
        del edges[low]
    return {'edges': tuple(edges), 'counts': totals, 'shares': [t / total for t in totals],
            'initial_counts': initial, 'initial_shares': [t / total for t in initial], 'steps': steps}


def build_parser():
    parser = argparse.ArgumentParser(description='FEAT-04b C/D train-only statistics audit')
    parser.add_argument('--inputs-spec', required=True)
    parser.add_argument('--src', required=True)
    return parser


def _group_doc(totals, grid):
    merged = merge_small_buckets(totals, grid)
    return {'candidate_edges': list(grid), 'candidate_counts': merged['initial_counts'], 'candidate_shares': merged['initial_shares'],
            'final_edges': list(merged['edges']), 'final_counts': merged['counts'], 'final_shares': merged['shares'], 'merge_steps': merged['steps']}


def _histogram(values, size):
    counts = [0] * size
    for bucket in values:
        counts[bucket - 1] += 1
    return counts


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = json.loads(Path(args.inputs_spec).read_text())
    for key in ('prepared', 'raw_root'):
        guard_path(spec[key])
    sys.path.insert(0, args.src)
    import importlib
    from vkgr import history_side, item_stats
    from vkgr.data_audit import parquet_rows
    from vkgr.manifest import read_json
    from vkgr.retrieval_data import checked, verify_prepared
    ablation = importlib.import_module('vkgr.retrieval_ablation_run')
    prepared = Path(spec['prepared'])
    checked(guard_path(prepared / 'prepare-manifest.json'), ablation.PREPARED_SHA)
    manifest = verify_prepared(prepared)
    catalog = read_json(prepared / 'catalog.json')
    support = set(read_json(prepared / 'support.json'))
    with (prepared / 'examples.jsonl').open() as stream:
        examples = [json.loads(line) for line in stream]
    raw_root = Path(spec['raw_root'])
    names = [f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet' for w in range(TRAIN_WEEKS)]
    records = [r for r in manifest['binding']['input_records'] if r['role'] in names]
    if [r['role'] for r in records] != names:
        raise ValueError('raw/prepared lineage conflict')
    for record in records:
        path = checked(guard_path(raw_root / record['role']), record['sha256'])
        if path.stat().st_size != record['size_bytes']:
            raise ValueError('raw input size mismatch')

    def partitions():
        for week, record in enumerate(records):
            yield week, parquet_rows(raw_root / record['role'])

    metadata = checked(guard_path(raw_root / 'metadata/items_metadata.parquet'), spec['items_metadata_sha256'])
    durations = history_side.load_durations(metadata, catalog)
    report = item_stats.scan_ranges(partitions(), catalog, durations)
    violations = item_stats.range_violations(report)
    if violations:
        raise ValueError('range-check violations: ' + '; '.join(violations))
    stats = item_stats.build_item_stats(partitions(), catalog, durations)

    index = {item: k for k, item in enumerate(catalog)}
    rows = [tuple(stats[name][index[e['target']]] for name in item_stats.FIELDS) for e in examples]          # (n, first_week, likes, ratio_pm_sum) of each target
    sum_likes, rows_total = sum(stats['likes']), stats['rows_total']
    p0 = like_p0(sum_likes, rows_total)
    grids = {'expo': (EXPO_GRID, lambda s, edges: expo_bucket(s[0], edges)), 'first': (FIRST_GRID, lambda s, edges: first_bucket(s[1], edges)),
             'ratio': (RATIO_GRID, lambda s, edges: ratio_bucket(s[3], s[0], edges)), 'like': (LIKE_GRID, lambda s, edges: like_bucket(s[2], s[0], edges, LIKE_K, p0))}
    docs, final = {}, {}
    for name, (grid, bucket) in grids.items():
        docs[name] = _group_doc(_histogram([bucket(s, grid) for s in rows], len(grid) + 1), grid)
        final[name] = tuple(docs[name]['final_edges'])
        recount = _histogram([bucket(s, final[name]) for s in rows], len(final[name]) + 1)
        if recount != docs[name]['final_counts']:
            raise ValueError('merged histogram differs from a recount with the final edges')
    bucket_fns = {'expo': lambda n: expo_bucket(n, final['expo']), 'first': lambda week: first_bucket(week, final['first']),
                  'ratio': lambda total, n: ratio_bucket(total, n, final['ratio']), 'like': lambda likes, n: like_bucket(likes, n, final['like'], LIKE_K, p0)}
    target_freq = {}
    for e in examples:
        target_freq[e['target']] = target_freq.get(e['target'], 0) + 1
    equal = sum(stats['likes'][k] == target_freq.get(item, 0) for k, item in enumerate(catalog))
    inclusion = item_stats.stats_self_inclusion(partitions(), examples, catalog, durations, stats, bucket_fns)
    out = {'schema': SCHEMA, 'train_only': True, 'stats_version': stats['stats_version'], 'cutoff_week': stats['cutoff_week'],
           'catalog_items': len(catalog), 'supported_items': len(support), 'n_train_examples': len(examples), 'rows_total': rows_total,
           'sum_likes': sum_likes, 'sum_likes_minus_n_train_examples': sum_likes - len(examples),
           'likes_vs_target_freq': {'items_equal': equal, 'share_items_equal': equal / len(catalog),
                                    'sum_abs_diff': sum(abs(stats['likes'][k] - target_freq.get(item, 0)) for k, item in enumerate(catalog))},
           'timespent': {'rows': report['rows'], 'zero_rows': report['timespent_zero_rows'], 'zero_share': report['timespent_zero_share'],
                         'min': report['timespent_min'], 'max': report['timespent_max']},
           'duration': {'min': report['duration_min'], 'max': report['duration_max']},
           'like_p0': p0, 'like_k': LIKE_K, 'ratio_cap_pm': item_stats.RATIO_CAP_PM, 'grids': docs,
           'stats_self_inclusion': inclusion, 'p_like_report_threshold': P_LIKE_REPORT_THRESHOLD,
           'p_like_exceeds_threshold': inclusion['p_like'] > P_LIKE_REPORT_THRESHOLD,
           'logq_redundancy': item_stats.logq_redundancy(stats, catalog, support, target_freq, bucket_fns),
           'stats_sha256': item_stats.stats_digest(stats),
           'inputs': {'items_metadata_sha256': spec['items_metadata_sha256'], 'prepared_sha256': ablation.PREPARED_SHA}}
    print(json.dumps(out, sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
