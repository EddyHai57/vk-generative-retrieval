"""CPU timing only for residual KMeans over the round19 smoke embeddings."""

import argparse
import csv
import json
from pathlib import Path
import resource
import signal
import sys
import time

import faiss
import numpy as np


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(max(1, int(min(7200, args.deadline - time.time()))))
    args.output.mkdir(parents=True, exist_ok=False)
    faiss.omp_set_num_threads(4)
    source = Path('cache/round19-two-tower-speedup/trial-cpu/best-cold-vectors')
    vectors = np.load(source / 'vectors.npy', mmap_mode='r', allow_pickle=False)
    with np.load('cache/round13-foundation/dataset-v3/arrays/items.npz', allow_pickle=False) as archive:
        seen = np.flatnonzero(archive['train_count'] > 0)
    rows = np.sort(np.random.default_rng(42).choice(seen, size=1_000_000, replace=False))
    np.save(args.output / 'fit_rows.npy', rows, allow_pickle=False)
    residual = np.asarray(vectors[rows], dtype=np.float32)
    centers = []
    timings = []
    for level in range(3):
        started = time.perf_counter()
        # Same FAISS settings as the round9 residual-KMeans implementation.
        km = faiss.Kmeans(64, 1024, niter=12, nredo=1, seed=42 + level,
                          max_points_per_centroid=1_000_000, min_points_per_centroid=1,
                          verbose=False)
        km.train(residual)
        center = np.asarray(km.centroids, dtype=np.float32)
        _, assigned = km.index.search(residual, 1)
        residual -= center[assigned[:, 0]]
        centers.append(center)
        np.save(args.output / f'centroids_layer{level + 1}.npy', center, allow_pickle=False)
        row = dict(stage='fit_layer', layer=level + 1, rows=len(residual),
                   seconds=time.perf_counter() - started, peak_rss_bytes=rss_bytes())
        timings.append(row)
        print(json.dumps(row), flush=True)
    del residual, seen, rows, km
    indices = []
    for center in centers:
        index = faiss.IndexFlatL2(64)
        index.add(center)
        indices.append(index)
    codes = np.lib.format.open_memmap(args.output / 'codes.npy', mode='w+', dtype=np.uint16,
                                     shape=(len(vectors), 3))
    started = time.perf_counter()
    for start in range(0, len(vectors), 65536):
        end = min(len(vectors), start + 65536)
        remaining = np.asarray(vectors[start:end], dtype=np.float32)
        for level, index in enumerate(indices):
            _, assigned = index.search(remaining, 1)
            codes[start:end, level] = assigned[:, 0]
            remaining -= centers[level][assigned[:, 0]]
        if start % (65536 * 32) == 0:
            print(json.dumps({'stage': 'assign_progress', 'rows': end,
                              'seconds': time.perf_counter() - started}), flush=True)
    codes.flush()
    timings.append(dict(stage='assign_catalog', layer=3, rows=len(vectors),
                        seconds=time.perf_counter() - started, peak_rss_bytes=rss_bytes()))
    with (args.output / 'timing.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    (args.output / 'summary.json').write_text(json.dumps({
        'status': 'complete', 'scope': 'timing only, no quality conclusion',
        'source_vectors': str(source / 'vectors.npy'), 'fit_scope': 'uniform train-seen rows',
        'fit_rows': 1_000_000, 'layers': 3, 'clusters_per_layer': 1024,
        'iterations': 12, 'restarts': 1, 'seeds': [42, 43, 44], 'threads': 4,
        'timings': timings, 'peak_rss_bytes': rss_bytes(), 'test_week26_read': False,
    }, indent=2))
    signal.alarm(0)
    print(json.dumps({'status': 'complete', 'rows': len(vectors), 'peak_rss_bytes': rss_bytes()}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
