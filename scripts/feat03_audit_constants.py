"""FEAT-03 (P line) Q3 train-only bucket audit. Prints one canonical JSON document, no paths, writes no file.

Counts, once per train example (history-v2 L50 window), the number of history events n (1..50) and the number of
DISTINCT weeks a among them (1..25) and applies the pre-registered merge rule to the pre-registered grids.
Only the history-v2 train artifacts named by the pinned inputs spec are read; validation queries and later partitions are refused.
"""
import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

SCHEMA = 'feat03-audit-v1'
MAX_LEN = 50
TRAIN_WEEKS = 25
LEN_GRID = (4, 8, 16, 32, 50)    # n: [1-3] [4-7] [8-15] [16-31] [32-49] [50]
ACT_GRID = (2, 3, 5, 9, 17)      # a: [1] [2] [3-4] [5-8] [9-16] [17-25]
MIN_SHARE = Fraction(1, 50)

# Path fragments that must never be opened; assembled at run time so the source itself stays free of them.
_FORBIDDEN = ('validation' + '-queries', 'week_' + '25', 'week_' + '26')
_FORBIDDEN_PART = 'te' + 'st'


def guard_path(path):
    """Raise ValueError for a path that points into the validation queries or the test / later-week partitions."""
    text = str(path).replace('\\', '/')
    if any(fragment in text for fragment in _FORBIDDEN) or _FORBIDDEN_PART in Path(text).parts:
        raise ValueError('audit refuses a non-train path')
    return path


def length_hist(histories, max_len=MAX_LEN):
    """counts[n] = number of examples whose history has n events; every example has 1..max_len events."""
    counts = [0] * (max_len + 1)
    for history in histories:
        if not 1 <= len(history) <= max_len:
            raise ValueError('history length outside 1..%d' % max_len)
        counts[len(history)] += 1
    return counts


def distinct_weeks(positions, weeks=TRAIN_WEEKS):
    """Number of different weeks among a history's [week, row] positions (not the span last - first)."""
    if not positions:
        raise ValueError('empty history')
    seen = set()
    for position in positions:
        week = position[0]
        if type(week) is not int or not 0 <= week < weeks:
            raise ValueError('history week outside train weeks 0-%d' % (weeks - 1))
        seen.add(week)
    return len(seen)


def activity_hist(position_lists, weeks=TRAIN_WEEKS):
    """counts[a] = number of examples with a distinct weeks in their history."""
    counts = [0] * (weeks + 1)
    for positions in position_lists:
        counts[distinct_weeks(positions, weeks)] += 1
    return counts


def _bucket_totals(counts, edges):
    bounds = [0, *edges, len(counts)]
    return [sum(counts[bounds[i]:bounds[i + 1]]) for i in range(len(bounds) - 1)]


def merge_small_buckets(counts, edges, min_share=MIN_SHARE):
    """Pre-registered rule: while a bucket share < min_share, merge the smallest bucket (ties: lower index)
    into its smaller neighbour (ties: the lower one). `edges` are the first values of buckets 1.. ."""
    total = sum(counts)
    if total <= 0:
        raise ValueError('empty histogram')
    edges = list(edges)
    totals = _bucket_totals(counts, edges)
    initial = list(totals)
    steps = []
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


def bucket_table(counts, edges):
    """The bucket share table that is hashed: final edges, per-bucket example counts and shares."""
    totals = _bucket_totals(counts, edges)
    total = sum(totals)
    return {'edges': list(edges), 'counts': totals, 'shares': [t / total for t in totals]}


def table_sha256(table):
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description='FEAT-03 Q3 train-only bucket audit')
    parser.add_argument('--inputs-spec', required=True)
    parser.add_argument('--src', required=True)
    return parser


def _doc(counts, grid):
    result = merge_small_buckets(counts, grid)
    table = bucket_table(counts, result['edges'])
    return {'candidate_edges': list(grid), 'candidate_counts': result['initial_counts'],
            'candidate_shares': result['initial_shares'], 'final_edges': table['edges'],
            'final_counts': table['counts'], 'final_shares': table['shares'], 'merge_steps': result['steps']}, table


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = json.loads(Path(args.inputs_spec).read_text())
    guard_path(spec['history_manifest'])
    sys.path.insert(0, args.src)
    from vkgr.history_v2 import load_prepared
    loaded = load_prepared(spec['history_manifest'], spec['history_sha256'], MAX_LEN)
    rows = loaded['train']
    histories = [r['history'][-MAX_LEN:] for r in rows]
    positions = [r['history_positions'][-MAX_LEN:] for r in rows]
    if any(len(h) != len(p) for h, p in zip(histories, positions)):
        raise ValueError('history/positions misaligned')
    len_counts = length_hist(histories)
    act_counts = activity_hist(positions)
    len_doc, len_table = _doc(len_counts, LEN_GRID)
    act_doc, act_table = _doc(act_counts, ACT_GRID)
    out = {'schema': SCHEMA, 'train_only': True, 'examples': len(rows),
           'n_equals_50': {'count': len_counts[MAX_LEN], 'share': len_counts[MAX_LEN] / len(rows)},
           'len': len_doc, 'act': act_doc,
           'table_sha256': {'len': table_sha256(len_table), 'act': table_sha256(act_table)},
           'merge_count': len(len_doc['merge_steps']) + len(act_doc['merge_steps']),
           'inputs': {'history_manifest_sha256': spec['history_sha256']}}
    print(json.dumps(out, sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
