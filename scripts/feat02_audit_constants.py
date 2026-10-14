"""FEAT-02 (P line) train-only constants audit. Prints one canonical JSON document, no paths.

Reads ONLY train-side inputs: history-v2 L50 train rows, raw train partitions (weeks 00-24, place/platform),
the prepared catalog/support/content, and the pinned item metadata (item_id/author_id).
All paths come from the pinned inputs spec and --src; nothing else is opened.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = 'feat02-audit-v1'
MAX_LEN = 50
TRAIN_WEEKS = 25
POSITION_GRID = (1, 2, 4, 8, 16, 32)
WEEKGAP_GRID = (1, 2, 3, 5, 9, 17)
MIN_SHARE = Fraction(1, 50)


def _check_weeks(history):
    for position in history:
        if type(position[0]) is not int or not 0 <= position[0] < TRAIN_WEEKS:
            raise ValueError('history week outside train weeks 0-24')


def position_hist(histories, max_len=MAX_LEN):
    """Counts per back-index (0 = most recent event); histories are oldest->newest [week, row] lists."""
    counts = [0] * max_len
    for history in histories:
        if len(history) > max_len:
            raise ValueError('history longer than the registered length')
        _check_weeks(history)
        for index in range(len(history)):
            counts[len(history) - 1 - index] += 1
    return counts


def weekgap_hist(histories, weeks=TRAIN_WEEKS):
    """Counts per week gap = week of the LAST history event minus week of each event."""
    counts = [0] * weeks
    for history in histories:
        if not history:
            continue
        _check_weeks(history)
        last = history[-1][0]
        for position in history:
            gap = last - position[0]
            if not 0 <= gap < weeks:
                raise ValueError('week gap outside the registered range')
            counts[gap] += 1
    return counts


def derive_edges(counts, edges, min_share=MIN_SHARE):
    """Pre-registered merge rule: while a bucket share < min_share, merge the smallest bucket
    (ties: lower index) into its smaller neighbour (ties: lower neighbour)."""
    total = sum(counts)
    if total <= 0:
        raise ValueError('empty histogram')
    edges = list(edges)
    starts = [0, *edges]
    bounds = [*starts, len(counts)]
    totals = [sum(counts[bounds[i]:bounds[i + 1]]) for i in range(len(starts))]
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


def category_table(arrays):
    """Ascending unique values with row counts over an iterable of arrays/lists."""
    import numpy as np
    total = Counter()
    for array in arrays:
        values, counts = np.unique(np.asarray(array), return_counts=True)
        total.update(dict(zip(values.tolist(), counts.tolist())))
    values = sorted(total)
    return {'values': values, 'counts': [total[v] for v in values]}


def author_vocab(catalog, support, item_author):
    """Authors of supported catalog items, ascending; everything else is OOV downstream."""
    catalog = list(catalog)
    support = set(support)
    if set(item_author) != set(catalog):
        raise ValueError('catalog item without unique author metadata')
    vocab = sorted({item_author[i] for i in catalog if i in support})
    members = set(vocab)
    supported = [i for i in catalog if i in support]
    unsupported = [i for i in catalog if i not in support]
    digest = hashlib.sha256(json.dumps(vocab, separators=(',', ':')).encode()).hexdigest()
    return {'vocab': vocab, 'vocab_size': len(vocab), 'vocab_sha256': digest,
            'catalog_items': len(catalog), 'supported_items': len(supported), 'unsupported_items': len(unsupported),
            'supported_with_vocab_author_fraction': (sum(item_author[i] in members for i in supported) / len(supported)) if supported else None,
            'unsupported_with_vocab_author_fraction': (sum(item_author[i] in members for i in unsupported) / len(unsupported)) if unsupported else None}


def quantiles(values, points=(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)):
    ordered = sorted(values)
    return {str(p): ordered[int(p * (len(ordered) - 1))] for p in points}


def build_parser():
    parser = argparse.ArgumentParser(description='FEAT-02 train-only constants audit')
    parser.add_argument('--inputs-spec', required=True)
    parser.add_argument('--src', required=True)
    return parser


def _canon_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main(argv=None):
    args = build_parser().parse_args(argv)
    sys.path.insert(0, args.src)
    import numpy as np
    import pyarrow.parquet as pq
    from vkgr.history_v2 import load_prepared
    from vkgr.manifest import read_json
    from vkgr.retrieval import PREPARED_SOURCE
    from vkgr.retrieval_ablation_run import PREPARED_SHA
    from vkgr.retrieval_data import checked, verify_prepared
    spec = json.loads(Path(args.inputs_spec).read_text())
    prepared = Path(spec['prepared'])
    raw_root = Path(spec['raw_root'])
    checked(prepared / 'prepare-manifest.json', PREPARED_SHA)
    manifest = verify_prepared(prepared)
    if manifest['binding']['consumer']['source_fingerprint'] != PREPARED_SOURCE:
        raise ValueError('prepared producer mismatch')
    catalog = read_json(prepared / 'catalog.json')
    support = read_json(prepared / 'support.json')
    content = np.load(prepared / 'content.npy', allow_pickle=False)

    # history-v2 L50 train rows
    loaded = load_prepared(spec['history_manifest'], spec['history_sha256'], MAX_LEN)
    histories = [row['history_positions'] for row in loaded['train']]
    pos_counts = position_hist(histories)
    gap_counts = weekgap_hist(histories)
    lengths = [len(h) for h in histories]
    position = derive_edges(pos_counts, POSITION_GRID)
    weekgap = derive_edges(gap_counts, WEEKGAP_GRID)

    # raw train partitions, place/platform only
    names = [f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet' for w in range(TRAIN_WEEKS)]
    records = [r for r in manifest['binding']['input_records'] if r['role'] in names]
    if [r['role'] for r in records] != names:
        raise ValueError('raw/prepared lineage conflict')
    places, platforms, rows = [], [], 0
    for record in records:
        path = checked(raw_root / record['role'], record['sha256'])
        if path.stat().st_size != record['size_bytes']:
            raise ValueError('raw input size mismatch')
        table = pq.read_table(path, columns=['place', 'platform'])
        places.append(table.column('place').to_numpy())
        platforms.append(table.column('platform').to_numpy())
        rows += table.num_rows

    # author metadata (pinned), filtered by the catalog before materialising
    metadata = checked(raw_root / 'metadata/items_metadata.parquet', spec['items_metadata_sha256'])
    table = pq.read_table(metadata, columns=['item_id', 'author_id'])
    ids = table.column('item_id').to_numpy()
    keep = np.isin(ids, np.asarray(catalog, dtype=np.uint32))
    kept_ids = ids[keep].tolist()
    kept_authors = table.column('author_id').to_numpy()[keep].tolist()
    if len(set(kept_ids)) != len(kept_ids):
        raise ValueError('duplicate item metadata')
    authors = author_vocab(catalog, support, dict(zip(kept_ids, kept_authors)))
    vocab = authors.pop('vocab')
    zero_rows = ~content[1:].any(axis=1)
    support_set = set(support)
    supported_mask = np.array([i in support_set for i in catalog])
    authors['zero_content_items'] = int(zero_rows.sum())
    authors['zero_content_supported_items'] = int((zero_rows & supported_mask).sum())
    authors['vocab_contains_author_zero'] = 0 in set(vocab)

    def bucket_doc(result, grid):
        return {'candidate_edges': list(grid), 'candidate_counts': result['initial_counts'],
                'candidate_shares': result['initial_shares'], 'final_edges': list(result['edges']),
                'final_counts': result['counts'], 'final_shares': result['shares'], 'merge_steps': result['steps']}

    out = {'schema': SCHEMA, 'train_only': True,
           'history': {'examples': len(histories), 'events': sum(lengths), 'length_quantiles': quantiles(lengths)},
           'position_hist': pos_counts, 'weekgap_hist': gap_counts,
           'position_table': bucket_doc(position, POSITION_GRID), 'weekgap_table': bucket_doc(weekgap, WEEKGAP_GRID),
           'raw_train_rows': rows,
           'place': category_table(places), 'platform': category_table(platforms),
           'author': authors,
           'inputs': {'prepared_sha256': PREPARED_SHA, 'history_manifest_sha256': spec['history_sha256'],
                      'items_metadata_sha256': spec['items_metadata_sha256'],
                      'raw_train_records_sha256': _canon_sha([r['sha256'] for r in records])}}
    print(json.dumps(out, sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
