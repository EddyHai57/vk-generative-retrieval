"""Build portable, columnwise memmaps from train sequences and week25 arrays."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .two_tower import Catalog, TowerConfig, user_hash
from .two_tower_runtime import configure_threads, duckdb_connection, process_metrics
from .two_tower_tensors import PAIR_STRIDE, TensorCatalog, TensorEventStore


EVENT_COLUMNS = {
    'item': ('item_ids', np.int32),
    'week': ('weeks', np.uint8),
    'place': ('places', np.uint8),
    'main': ('main_positive', np.bool_),
    'secondary': ('secondary_positive', np.bool_),
    'feedback': ('feedback_masks', np.int16),
    'ratio': ('watch_ratios', np.float32),
}
CHUNK = 1_000_000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _new_array(output, name, dtype, shape):
    path = Path(output) / (name + '.npy')
    if path.exists():
        raise FileExistsError(path)
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def _copy_array(output, name, source, dtype=None):
    dtype = source.dtype if dtype is None else dtype
    result = _new_array(output, name, dtype, source.shape)
    for start in range(0, len(source), CHUNK):
        result[start:start + CHUNK] = source[start:start + CHUNK]
    result.flush()
    del result


def build_tensor_dataset(foundation, vector_ids, vectors, output, *, users=0,
                         config=None):
    started = time.perf_counter()
    config = config or TowerConfig(batch_size=1024)
    foundation, output = Path(foundation), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    connection = duckdb_connection(output / 'duckdb-tmp')
    paths = [foundation / f'sequences/user_block={i}/users.parquet'
             for i in range(32)]
    source_paths = {f'foundation/sequences/user_block={i}/users.parquet': path
                    for i, path in enumerate(paths)}
    lengths = connection.execute(
        'SELECT user_id,len(item_ids) AS n FROM read_parquet(?) ORDER BY user_id',
        [[str(path) for path in paths]],
    ).fetchnumpy()
    ids = lengths['user_id']
    sizes = lengths['n']
    if users:
        queries = np.load(foundation / 'arrays/queries.npz', allow_pickle=False)
        chosen = sorted(queries['user_id'],
                        key=lambda u: user_hash(u, config.user_hash_seed))[:users]
        selected = np.isin(ids, chosen)
        ids, sizes = ids[selected], sizes[selected]
        queries.close()
    ids = np.asarray(ids, dtype=np.int32)
    offsets = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError('event positions exceed the compact int32 layout')
    offsets = offsets.astype(np.int32)
    total = int(offsets[-1])
    connection.register('selected_users', {'user_id': ids})
    _copy_array(output, 'user_id', ids)
    _copy_array(output, 'user_offsets', offsets)
    for short, (source, dtype) in EVENT_COLUMNS.items():
        values = _new_array(output, 'event_' + short, dtype, (total,))
        cursor = connection.execute(
            f'SELECT s.user_id,s.{source} FROM read_parquet(?) s '
            'JOIN selected_users USING(user_id) ORDER BY s.user_id',
            [[str(path) for path in paths]],
        )
        position = 0
        for ordinal in range(len(ids)):
            row = cursor.fetchone()
            if row is None or row[0] != ids[ordinal] or len(row[1]) != sizes[ordinal]:
                raise ValueError('sequence columns do not share the same user order')
            block = np.asarray(row[1], dtype=dtype)
            if short == 'week' and np.any(block > 24):
                raise ValueError('training sequences contain a week after 24')
            if short == 'feedback' and np.any((block < 0) | (block >= 512)):
                raise ValueError('feedback mask exceeds nine bits')
            values[position:position + len(block)] = block
            position += len(block)
        if position != total:
            raise ValueError('sequence column length mismatch')
        values.flush()
        del values
        print('prepared', 'event_' + short, total, flush=True)

    item_path = foundation / 'items.parquet'
    source_paths['foundation/items.parquet'] = item_path
    catalog_count = connection.execute('SELECT count(*) FROM read_parquet(?)',
                                       [str(item_path)]).fetchone()[0]
    for name, source, dtype in [
        ('item_id', 'item_id', np.int32), ('author_id', 'author_id', np.int32),
        ('duration', 'duration', np.uint8), ('train_count', 'train_count', np.int32),
        ('rank', 'train_interactions_rank', np.int32),
        ('popularity_tier', 'popularity_tier', np.uint8),
    ]:
        values = _new_array(output, 'catalog_' + name, dtype, (catalog_count,))
        cursor = connection.execute(f'SELECT {source} FROM read_parquet(?) ORDER BY item_id',
                                    [str(item_path)])
        position = 0
        while block := cursor.fetchmany(65536):
            array = np.asarray(block).reshape(-1)
            if np.issubdtype(dtype, np.integer) and len(array):
                bounds = np.iinfo(dtype)
                if array.min() < bounds.min or array.max() > bounds.max:
                    raise ValueError('catalog value does not fit ' + str(dtype))
            values[position:position + len(block)] = array
            position += len(block)
        values.flush()
        del values
    catalog_ids = np.load(output / 'catalog_item_id.npy', mmap_mode='r')
    ids_vector = np.load(vector_ids, mmap_mode='r')
    content = np.load(vectors, mmap_mode='r')
    if content.dtype != np.float16 or content.shape != (catalog_count, 64):
        raise ValueError('expected the existing float16, 64-dimensional content input')
    for start in range(0, catalog_count, CHUNK):
        if not np.array_equal(catalog_ids[start:start+CHUNK], ids_vector[start:start+CHUNK]):
            raise ValueError('content and sorted catalog item IDs differ')
    _copy_array(output, 'catalog_content', content)
    source_paths['vectors/item_id.npy'] = Path(vector_ids)
    source_paths['vectors/embedding.npy'] = Path(vectors)
    del content, ids_vector

    event = {name: np.load(output / ('event_' + name + '.npy'), mmap_mode='r')
             for name in ('item', 'main', 'secondary', 'place')}
    target_count = positive_count = 0
    for start in range(0, total, CHUNK):
        positive = event['main'][start:start+CHUNK] | event['secondary'][start:start+CHUNK]
        positive_count += int(positive.sum())
        target_count += int((positive & (event['place'][start:start+CHUNK] <= 1)).sum())
    samples = _new_array(output, 'target_event_position', np.int32, (target_count,))
    weights = _new_array(output, 'target_weight', np.float16, (target_count,))
    event_rows = _new_array(output, 'event_catalog_row', np.int32, (total,))
    event_users = _new_array(output, 'event_user_row', np.int32, (total,))
    frequency = np.zeros(catalog_count, dtype=np.int32)
    position = 0
    for start in range(0, total, CHUNK):
        stop = min(start + CHUNK, total)
        item_ids = event['item'][start:stop]
        rows = np.searchsorted(catalog_ids, item_ids)
        if np.any(rows >= catalog_count) or np.any(catalog_ids[rows] != item_ids):
            raise ValueError('training item absent from full catalog')
        event_rows[start:stop] = rows
        positive = event['main'][start:stop] | event['secondary'][start:stop]
        selected = np.flatnonzero(positive & (event['place'][start:stop] <= 1))
        samples[position:position+len(selected)] = selected + start
        weights[position:position+len(selected)] = np.where(event['main'][start:stop][selected],
                                                           1., .25)
        np.add.at(frequency, rows[selected], 1)
        position += len(selected)
    for user in range(len(ids)):
        event_users[offsets[user]:offsets[user+1]] = user
    for values in [samples, weights, event_rows, event_users]:
        values.flush()
    del samples, weights, event_rows, event_users
    _copy_array(output, 'catalog_target_count', frequency)
    probability = (frequency.astype(np.float64) / max(1, target_count)).astype(np.float32)
    _copy_array(output, 'catalog_target_probability', probability)
    frequency_rows = np.flatnonzero(frequency)
    _copy_array(output, 'frequency_item_ids', catalog_ids[frequency_rows])
    _copy_array(output, 'frequency_count', frequency[frequency_rows])
    del probability, frequency_rows
    keys = _new_array(output, 'positive_keys', np.int64, (positive_count,))
    positive_offsets = _new_array(output, 'positive_offsets', np.int32, (len(ids)+1,))
    position = 0
    positive_offsets[0] = 0
    for user in range(len(ids)):
        low, high = offsets[user:user+2]
        mask = event['main'][low:high] | event['secondary'][low:high]
        items = np.sort(event['item'][low:high][mask]).astype(np.int64)
        if len(items) > 1 and np.any(items[1:] == items[:-1]):
            raise ValueError('source violates the one-row-per-user-item contract')
        keys[position:position+len(items)] = user * PAIR_STRIDE + items
        position += len(items)
        positive_offsets[user+1] = position
    keys.flush()
    positive_offsets.flush()
    del keys, positive_offsets, event

    # The existing week25 evaluation arrays retain empty histories and zero targets.
    for namespace in ['queries', 'history', 'targets', 'items']:
        path = foundation / 'arrays' / (namespace + '.npz')
        source_paths['foundation/arrays/' + path.name] = path
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                values = archive[name]
                dtype = values.dtype
                if dtype.kind in 'ui' and values.size and values.max() <= np.iinfo(np.int32).max:
                    dtype = np.int32 if dtype.itemsize > 2 else dtype
                _copy_array(output, 'eval_' + namespace + '_' + name, values, dtype)
    history_path = foundation / 'arrays/history_items.npy'
    source_paths['foundation/arrays/history_items.npy'] = history_path
    _copy_array(output, 'eval_history_items', np.load(history_path, mmap_mode='r'))
    connection.close()
    arrays = {}
    for path in sorted(output.glob('*.npy')):
        values = np.load(path, mmap_mode='r')
        arrays[path.stem] = dict(file=path.name, dtype=values.dtype.str,
                                 shape=list(values.shape), rows=len(values),
                                 bytes=path.stat().st_size, sha256=sha256_file(path))
    sources = {name: dict(sha256=sha256_file(path), bytes=path.stat().st_size)
               for name, path in source_paths.items()}
    manifest = dict(format='lsvd-two-tower-tensors-v1', train_weeks=[0, 24],
                    validation_week=25, arrays=arrays, sources=sources,
                    counts=dict(training_events=total, training_users=len(ids),
                                training_targets=target_count,
                                all_place_positives=positive_count,
                                catalog_items=catalog_count,
                                frequency_items=int(np.count_nonzero(frequency))),
                    seconds=time.perf_counter()-started, resources=process_metrics(),
                    layout_notes=dict(position_dtype='int32',
                                      positive_key='user_row * 2**32 + raw_item_id',
                                      watch_ratio='float32 preserves reference bucket boundaries',
                                      feedback='int16 holds nine bits',
                                      epoch_shuffle='separate disk memmap, never device-resident'))
    with (output / 'manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2)
    print('prepared dataset', json.dumps(manifest['counts']), flush=True)
    return manifest


def load_tensor_dataset(root, config, *, storage_device='cpu', pinned=False):
    from .evaluation import EvaluationData

    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    arrays = {name: np.load(root / row['file'], mmap_mode='c', allow_pickle=False)
              for name, row in manifest['arrays'].items()}
    columns = {name: arrays['catalog_' + name] for name in
               ('item_id', 'author_id', 'duration', 'train_count', 'rank', 'content')}
    catalog = TensorCatalog.from_catalog(Catalog(columns, config), storage_device, pinned)
    store = TensorEventStore(arrays, catalog, storage_device, pinned)
    namespace = lambda name: {key.removeprefix('eval_' + name + '_'): value
                              for key, value in arrays.items()
                              if key.startswith('eval_' + name + '_')}
    queries, history = namespace('queries'), namespace('history')
    data = EvaluationData(queries['user_id'], history['user_id'], history['offsets'],
                          arrays['eval_history_items'], namespace('targets'), namespace('items'))
    return arrays, catalog, store, data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foundation', type=Path, required=True)
    parser.add_argument('--vector-ids', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--users', type=int, default=0)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    resources = configure_threads(args.threads)
    if args.dry_run:
        print(json.dumps(dict(action='dry_run', train_weeks=[0, 24],
                              validation_week=25, resources=resources)))
        return 0
    build_tensor_dataset(args.foundation, args.vector_ids, args.vectors, args.output,
                          users=args.users)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
