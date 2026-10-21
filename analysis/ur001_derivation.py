"""UR001 train/validation derivation primitives; no test or training backend."""
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

SCHEMA = 'UR001-HIST_V2_OBS-L50-v1'


def checked_row(row):
    for key in ('user_id', 'item_id'):
        if type(row.get(key)) is not int or not 0 <= row[key] < 2**32:
            raise ValueError('interaction ID must be uint32')
    if type(row.get('like')) is not bool:
        raise ValueError('like must be a non-null boolean')
    return row['user_id'], row['item_id'], row['like']


def week_files(folder, weeks):
    folder = Path(folder)
    expected = [folder/f'week_{week:02d}.parquet' for week in weeks]
    actual = list(folder.glob('*.parquet'))
    if set(actual) != set(expected) or any(not p.is_file() or p.is_symlink() for p in expected):
        raise ValueError('one fixed parquet file per week required')
    return expected


def parquet_rows(path, workers=1):
    """Bounded row-group prefetch, yielded in file order, never completion order."""
    import pyarrow.parquet as pq
    if type(workers) is not int or workers not in (1, 2, 4):
        raise ValueError('workers must be 1/2/4')
    path = Path(path)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('symlink input refused')
    metadata = pq.ParquetFile(path).metadata
    def load(index):
        return pq.ParquetFile(path).read_row_group(index, columns=['user_id', 'item_id', 'like'],
                                                  use_threads=False).to_pylist()
    ordinal = 0
    # A sliding window prevents Executor.map from eagerly loading an entire week.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        next_group = 0
        while next_group < min(workers, metadata.num_row_groups):
            pending.append((next_group, pool.submit(load, next_group)))
            next_group += 1
        while pending:
            index, future = pending.popleft()
            rows = future.result()
            if len(rows) != metadata.row_group(index).num_rows:
                raise ValueError('row-group count changed')
            for row in rows:
                yield row
                ordinal += 1
            if next_group < metadata.num_row_groups:
                pending.append((next_group, pool.submit(load, next_group)))
                next_group += 1
    if ordinal != metadata.num_rows:
        raise ValueError('file ordinal continuity failed')


def derive(partitions, validation, emit_train):
    """Stream original train target identities with observed H50 sidecars."""
    catalog = set()
    watches = {}
    seen_likes = {}
    item_likes, user_likes = Counter(), Counter()
    example_id = observed_rows = 0
    previous = -1
    for week, rows in partitions:
        if type(week) is not int or not previous < week <= 24:
            raise ValueError('train weeks must increase within 00..24')
        previous = week
        for ordinal, row in enumerate(rows):
            user, item, like = checked_row(row)
            position = [week, ordinal]
            catalog.add(item)
            watch = watches.setdefault(user, deque(maxlen=50))
            seen = seen_likes.setdefault(user, set())
            if like:
                if user_likes[user]:
                    if not watch or not watch[-1][1] < position:
                        raise ValueError('history is not strictly past')
                    emit_train({'example_id': example_id, 'user_id': user,
                                'history': [v[0] for v in watch],
                                'history_positions': [v[1] for v in watch],
                                'target': item, 'target_position': position,
                                'past_likes': sorted(seen)})
                    example_id += 1
                seen.add(item)
                item_likes[item] += 1
                user_likes[user] += 1
            watch.append((item, position))
            observed_rows += 1
    week, rows = validation
    if type(week) is not int or week != 25 or week <= previous:
        raise ValueError('validation must be week25 after train')
    targets = {}
    validation_rows = validation_likes = 0
    for ordinal, row in enumerate(rows):
        user, item, like = checked_row(row)
        validation_rows += 1
        if like:
            validation_likes += 1
            targets.setdefault(user, (item, [week, ordinal]))
    queries = []
    no_history = in_catalog = oov = 0
    for user, (item, position) in sorted(targets.items()):
        if user_likes[user] < 1:
            no_history += 1
            continue
        watch = watches[user]
        if not watch or any(v[1][0] > 24 for v in watch):
            raise ValueError('evaluation history must be fixed at train end')
        missing = item not in catalog
        in_catalog += not missing
        oov += missing
        queries.append({'user_id': user, 'history': [v[0] for v in watch],
                        'history_positions': [v[1] for v in watch], 'target': item,
                        'target_position': position, 'oov': missing})
    positive = sorted(item_likes.values())
    if not positive:
        raise ValueError('no train positive support')
    thresholds = {name: positive[math.ceil(len(positive)*q)-1]
                  for name, q in (('median', .5), ('p90', .9))}
    catalog = sorted(catalog)
    slices = {'schema_version': 1, 'popularity_thresholds': thresholds,
              'item_like_counts': {str(i): item_likes[i] for i in catalog},
              'user_like_counts': {str(u): user_likes[u] for u in sorted(user_likes)},
              'user_buckets': [1, 5, 20], 'definition': 'train_like_counts_only_no_holdout_boundaries'}
    return {'catalog': catalog, 'supported': sorted(i for i in catalog if item_likes[i] > 0),
            'validation': queries, 'slices': slices,
            'counts': {'train_rows': observed_rows, 'train_users': len(watches),
                       'train_like_events': sum(user_likes.values()),
                       'train_like_users': sum(n > 0 for n in user_likes.values()),
                       'train_examples': example_id, 'validation_rows': validation_rows,
                       'validation_like_events': validation_likes, 'eligible': len(queries),
                       'in_catalog': in_catalog, 'oov': oov, 'no_history': no_history}}
