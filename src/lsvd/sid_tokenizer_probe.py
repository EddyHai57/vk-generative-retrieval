"""Six frozen CPU SID fits and per-target oracle diagnostics, never a generator."""
import argparse
import csv
import json
from pathlib import Path
import signal
import time

import faiss
import numpy as np

from .rq_timing import rss_bytes
from .sid_tokenizer import (assign_codes, build_tables, file_sha256, fit_residual,
                            filtered_ranks, item_hash, pack_codes)

GROUPS = ['本地见过', '仅本地未见', '全站新视频（rank代理）']


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(base):
    import duckdb
    source = Path('cache/round13-foundation/dataset-v3/arrays')
    with np.load(source/'items.npz') as archive:
        ids = archive['item_id']
        count = archive['train_count']
    connection = duckdb.connect()
    connection.execute("SET memory_limit='8GB'")
    connection.execute('SET threads=6')
    ranks = connection.execute(
        'SELECT item_id,train_interactions_rank FROM read_parquet(?) ORDER BY item_id',
        ['cache/ur001-raw/metadata/items_metadata.parquet']).fetchnumpy()
    np.testing.assert_array_equal(ids, ranks['item_id'])
    boundary = int(ranks['train_interactions_rank'][count > 0].max()) + 1
    group = np.where(count > 0, 0, np.where(ranks['train_interactions_rank'] < boundary, 1, 2)).astype(np.uint8)
    out = base/'inputs'
    out.mkdir()
    np.save(out/'item_ids.npy', ids)
    np.save(out/'train_count.npy', count)
    np.save(out/'groups.npy', group)
    seen_rows = np.flatnonzero(count > 0)
    rows = np.sort(np.random.default_rng(42).choice(seen_rows, 1_000_000, replace=False))
    np.save(out/'fit_rows.npy', rows)
    for vector_ids in ['cache/round9-sid/vectors/item_id.npy',
                       'cache/round19-two-tower-speedup/trial-cpu/best-cold-vectors/item_ids.npy']:
        np.testing.assert_array_equal(ids, np.load(vector_ids, mmap_mode='r'))
    write_json(out/'manifest.json', dict(catalog_items=len(ids), train_seen=len(seen_rows),
        fit_rows=len(rows), sample_seed=42, rank_boundary=boundary,
        group_sizes=np.bincount(group, minlength=3).tolist(), week26_read=False,
        vector_inputs=[dict(path=p, sha256=file_sha256(p)) for p in [
            'cache/round9-sid/vectors/embedding.npy',
            'cache/round19-two-tower-speedup/trial-cpu/best-cold-vectors/vectors.npy']],
        fit_rows_sha256=file_sha256(out/'fit_rows.npy')))


def target_statistics(base, run, bucket, bucket_counts, seen_counts, ids, train_count):
    source = Path('cache/round13-foundation/dataset-v3/arrays')
    with np.load(source/'targets.npz') as z:
        selected = z['main_positive'].astype(bool) & np.isin(z['place'], [0, 1])
        user = np.repeat(np.arange(len(z['offsets'])-1, dtype=np.uint32), np.diff(z['offsets']))[selected]
        rows = np.searchsorted(ids, z['item_id'][selected])
    with np.load(source/'queries.npz') as z:
        users = z['user_id']
    with np.load(source/'history.npz') as z:
        history_users, history_offsets = z['user_id'], z['offsets']
    histories = np.load(source/'history_items.npy', mmap_mode='r')
    groups = np.load(base/'inputs/groups.npy', mmap_mode='r')[rows]
    n_users = len(users)
    target_offsets = np.r_[0, np.cumsum(np.bincount(user, minlength=n_users))]
    target_buckets = bucket[rows]
    warm = seen_counts[target_buckets]
    outcomes = []
    names = []
    denominators = []
    target_summary = []
    for group, name in enumerate(GROUPS):
        mask = groups == group
        denominator = np.bincount(user[mask], minlength=n_users)
        # Seen target's own video contributes one train-seen item to its bucket.
        for metric, value in [('含训练见过视频（含自身）', warm > 0),
                              ('排除自身后仍含训练见过视频', warm - (train_count[rows] > 0) > 0)]:
            outcomes.append(np.bincount(user[mask], weights=value[mask], minlength=n_users))
            denominators.append(denominator)
            names.append(dict(group=name, metric=metric, order='不适用', k=0))
        sizes = bucket_counts[target_buckets[mask]]
        target_summary.append(dict(group=name, targets=int(mask.sum()),
            bucket_p50=float(np.quantile(sizes, .5)), bucket_p99=float(np.quantile(sizes, .99)),
            bucket_max=int(sizes.max())))
    hash_values = item_hash(ids)
    order_stats = []
    for order_name in ['训练曝光热度降序，固定hash破同分', '固定hash升序']:
        started = time.perf_counter()
        if order_name.startswith('训练'):
            order = np.lexsort((ids, hash_values, -train_count.astype(np.int64), bucket))
        else:
            order = np.lexsort((ids, hash_values, bucket))
        positions = np.empty(len(ids), dtype=np.int64)
        positions[order] = np.arange(len(ids))
        sorted_buckets = bucket[order]
        starts = np.r_[0, np.flatnonzero(sorted_buckets[1:] != sorted_buckets[:-1]) + 1]
        bucket_starts = starts[bucket]
        ranks = np.empty(len(rows), dtype=np.int64)
        for query_index, uid in enumerate(users):
            low, high = target_offsets[query_index:query_index+2]
            if low == high:
                continue
            hi = np.searchsorted(history_users, uid)
            if hi < len(history_users) and history_users[hi] == uid:
                history_rows = np.searchsorted(ids, histories[history_offsets[hi]:history_offsets[hi+1]])
            else:
                history_rows = np.empty(0, dtype=np.int64)
            ranks[low:high] = filtered_ranks(rows[low:high], history_rows, positions, bucket_starts)
        np.save(run/('oracle_pop_ranks.npy' if order_name.startswith('训练') else 'oracle_hash_ranks.npy'), ranks)
        for group, name in enumerate(GROUPS):
            mask = groups == group
            denominator = np.bincount(user[mask], minlength=n_users)
            for k in [50, 200, 1000]:
                hit = (ranks >= 0) & (ranks < k)
                outcomes.append(np.bincount(user[mask], weights=hit[mask], minlength=n_users))
                denominators.append(denominator)
                names.append(dict(group=name, metric='逐目标真码桶展开后item前K比例', order=order_name, k=k))
            positive = ranks[mask & (ranks >= 0)] + 1
            order_stats.append(dict(group=name, order=order_name,
                rank_p50=float(np.quantile(positive, .5)), rank_p99=float(np.quantile(positive, .99)),
                filtered_targets=int(np.count_nonzero(mask & (ranks < 0))),
                zero_train_targets=int(np.count_nonzero(mask & (train_count[rows] == 0)))))
        print({'stage': 'oracle', 'order': order_name, 'seconds': time.perf_counter()-started}, flush=True)
        del order, positions, sorted_buckets, bucket_starts
    numerators = np.asarray(outcomes).T
    denoms = np.asarray(denominators).T
    bootstrap = np.empty((1000, len(names)))
    rng = np.random.default_rng(2026112201)
    for low in range(0, 1000, 16):
        high = min(low + 16, 1000)
        weights = rng.poisson(1, (high-low, n_users)).astype(np.float64)
        bootstrap[low:high] = (weights @ numerators) / (weights @ denoms)
    for index, row in enumerate(names):
        row.update(targets=int(denoms[:, index].sum()), value=float(numerators[:, index].sum()/denoms[:, index].sum()),
                   ci_low=float(np.quantile(bootstrap[:, index], .025)),
                   ci_high=float(np.quantile(bootstrap[:, index], .975)))
    write_csv(run/'target_metrics.csv', names)
    write_csv(run/'target_buckets.csv', target_summary)
    write_csv(run/'oracle_rank_summary.csv', order_stats)
    np.savez(run/'user_statistics.npz', user_ids=users, numerators=numerators, denominators=denoms)
    return len(rows)


def run_one(args):
    started = time.time()
    base = args.output
    run = base/f'{args.source}_layers{args.layers}_size{args.size}'
    run.mkdir()
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(max(1, int(args.deadline-time.time())))
    faiss.omp_set_num_threads(4)
    ids = np.load(base/'inputs/item_ids.npy', mmap_mode='r')
    train_count = np.load(base/'inputs/train_count.npy', mmap_mode='r')
    fit_rows = np.load(base/'inputs/fit_rows.npy')
    vector_path = ('cache/round9-sid/vectors/embedding.npy' if args.source == 'official-content' else
                   'cache/round19-two-tower-speedup/trial-cpu/best-cold-vectors/vectors.npy')
    vectors = np.load(vector_path, mmap_mode='r')
    centers, fit_seconds = fit_residual(vectors, fit_rows, args.layers, args.size, run,
                                        normalize=args.source == 'official-content')
    assign_started = time.perf_counter()
    codes = np.lib.format.open_memmap(run/'codes.npy', mode='w+', dtype=np.uint16,
                                     shape=(len(ids), args.layers))
    for low in range(0, len(ids), 65536):
        high = min(low + 65536, len(ids))
        values = np.array(vectors[low:high], dtype=np.float32, order='C', copy=True)
        if args.source == 'official-content':
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        codes[low:high] = assign_codes(values, centers)
    codes.flush()
    assign_seconds = time.perf_counter()-assign_started
    print({'stage': 'assigned', 'seconds': assign_seconds}, flush=True)
    keys, buckets, counts = build_tables(ids, codes, args.size, run)
    seen_counts = np.bincount(buckets[train_count > 0], minlength=len(keys))
    np.save(run/'bucket_train_seen_counts.npy', seen_counts.astype(np.uint32))
    summary = dict(source=args.source, layers=args.layers, size=args.size,
        catalog_items=len(ids), fit_items=len(fit_rows), niter=12, nredo=1,
        occupied_codes=len(keys), collision_excess_rate=float(1-len(keys)/len(ids)),
        per_layer_used=[len(np.unique(codes[:, layer])) for layer in range(args.layers)],
        bucket_p50=float(np.quantile(counts, .5)), bucket_p99=float(np.quantile(counts, .99)),
        bucket_max=int(counts.max()),
        video_fraction_in_large_buckets={str(k):float(counts[counts > k].sum()/len(ids)) for k in [50,200,1000]},
        fit_seconds=fit_seconds, assignment_seconds=assign_seconds,
        input_sha256=file_sha256(vector_path), fit_rows_sha256=file_sha256(base/'inputs/fit_rows.npy'),
        item_ids='../inputs/item_ids.npy', map_format='codes.npy[row,:] aligns with item_ids.npy[row]',
        code_to_items='bucket_keys + bucket_offsets + bucket_items, CSR, item_id ascending',
        prefix_format='prefix_length_keys + offsets + next, CSR, per-layer token namespace',
        hash_seed=20261122, order_tie_break='SplitMix64, then item_id',
        faiss_version=faiss.__version__, week26_read=False,
        readout_scope='cold smoke vectors only; not learned generator quality')
    target_count = target_statistics(base, run, buckets, counts, seen_counts, ids, train_count)
    summary.update(targets=target_count, peak_rss_bytes=rss_bytes(), seconds=time.time()-started,
                   source_sha256={name:file_sha256(Path(__file__).parent/name) for name in [
                       'sid_tokenizer.py', 'sid_tokenizer_probe.py', 'rq_timing.py']})
    summary['version_sha256'] = file_sha256(run/'codes.npy')
    write_json(run/'manifest.json', summary)
    signal.alarm(0)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--source', choices=['official-content', 'tower-cold'])
    parser.add_argument('--layers', type=int)
    parser.add_argument('--size', type=int)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        prepare(args.output)
    else:
        run_one(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
