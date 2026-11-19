"""CPU PRQ-KMeans from arXiv:2608.24207v2, §§4.2–4.5 and Algorithm 1."""

import argparse
import json
from pathlib import Path
import time

import faiss
import numpy as np

from .sid_tokenizer import file_sha256


def unit_rows(values):
    values = np.asarray(values)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, lengths, out=np.zeros_like(values), where=lengths > 1e-12)


def projection_residual(values, direction):
    """Eq. 7, followed by the normalization in Eqs. 9 and 14."""
    points = np.asarray(values, dtype=np.float64)
    axes = np.asarray(direction, dtype=np.float64)
    dot = np.sum(points * axes, axis=-1, keepdims=True)
    square = np.sum(axes * axes, axis=-1, keepdims=True)
    scale = np.divide(dot, square, out=np.zeros_like(dot), where=square > 1e-24)
    return unit_rows(points - scale * axes).astype(np.float32)


def _cosine_index(centers):
    index = faiss.IndexFlatIP(centers.shape[1])
    index.add(np.ascontiguousarray(unit_rows(centers), dtype=np.float32))
    return index


def refine_centroids(points, centers, *, neighbors, beta, chunk=16384):
    """Eqs. 10–12. Each fitting row contributes to its exact top-k centers."""
    if not 1 <= neighbors <= len(centers) or chunk < 1 or not np.isfinite(beta) or beta <= 0:
        raise ValueError('invalid soft refinement settings')
    index = _cosine_index(centers)
    totals = np.zeros(centers.shape, dtype=np.float64)
    masses = np.zeros(len(centers), dtype=np.float64)
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        similarities, selected = index.search(np.ascontiguousarray(unit_rows(block)), neighbors)
        weights = np.exp(beta * (similarities - similarities[:, :1]))
        weights /= weights.sum(axis=1, keepdims=True)
        flat = selected.ravel()
        order = np.argsort(flat, kind='stable')
        labels = flat[order]
        first = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        active = labels[first]
        w = weights.ravel()[order]
        contributions = block[order // neighbors] * w[:, None]
        totals[active] += np.add.reduceat(contributions, first, axis=0, dtype=np.float64)
        masses[active] += np.add.reduceat(w, first, dtype=np.float64)
    # Empty neighborhoods are undefined by Eq. 12. Retain their old centers.
    updated = centers.copy()
    used = masses > 0
    updated[used] = totals[used] / masses[used, None]
    return updated, int((~used).sum())


def _hard_labels(index, residual):
    _, labels = index.search(np.ascontiguousarray(residual), 1)
    labels = labels[:, 0]
    # With zero residual all cosine candidates are undefined; select code zero.
    labels[~np.any(residual, axis=1)] = 0
    return labels


class PrqModel:
    def __init__(self, mean, centers):
        self.mean = mean
        self.centers = centers
        self.indexes = [_cosine_index(center) for center in centers]

    def encode(self, values):
        residual = projection_residual(unit_rows(np.asarray(values, dtype=np.float32)), self.mean)
        codes = np.empty((len(values), len(self.centers)), dtype=np.uint16)
        for i, (centers, index) in enumerate(zip(self.centers, self.indexes)):
            labels = _hard_labels(index, residual)
            codes[:, i] = labels
            if i + 1 < len(self.centers):
                residual = projection_residual(residual, centers[labels])
        return codes


def _fit(values, fit_rows, *, layers, size, iterations, neighbors, beta, seed):
    fit = unit_rows(np.asarray(values[fit_rows], dtype=np.float32))
    mean = fit.mean(axis=0, dtype=np.float64)
    chunk = 16384
    for low in range(0, len(fit), chunk):
        fit[low:low + chunk] = projection_residual(fit[low:low + chunk], mean)
    centers, seconds, empty_counts = [], [], []
    for level in range(layers):
        started = time.perf_counter()
        rng = np.random.default_rng(seed + level)
        center = fit[rng.choice(len(fit), size, replace=False)].copy()
        empty = []
        for iteration in range(iterations):
            center, n_empty = refine_centroids(fit, center, neighbors=neighbors, beta=beta)
            empty.append(n_empty)
            print('soft-refine', level + 1, iteration + 1,
                  'seconds', round(time.perf_counter()-started, 3), flush=True)
        if level + 1 < layers:
            index = _cosine_index(center)
            for low in range(0, len(fit), chunk):
                residual = fit[low:low + chunk]
                labels = _hard_labels(index, residual)
                residual[:] = projection_residual(residual, center[labels])
        centers.append(center)
        seconds.append(time.perf_counter() - started)
        empty_counts.append(empty)
    return PrqModel(mean, centers), seconds, empty_counts


def main(argv=None):
    from .two_tower_runtime import configure_threads, process_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('vectors', 'item-ids', 'train-count', 'fit-rows', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--neighbors', type=int, default=5)
    parser.add_argument('--beta', type=float, default=15.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    configured = configure_threads(args.threads)
    values, ids, counts, rows = (np.load(path, mmap_mode='r', allow_pickle=False) for path in
                               (args.vectors, args.item_ids, args.train_count, args.fit_rows))
    if (values.ndim != 2 or values.shape != (len(ids), 64) or counts.shape != ids.shape
            or ids.ndim != 1 or ids.dtype.kind not in 'ui' or np.any(ids[1:] <= ids[:-1])):
        raise ValueError('vectors, count and item ids must share one sorted 64-dimensional catalog')
    if (rows.ndim != 1 or rows.dtype.kind not in 'ui' or not 1 <= args.size <= min(65536, len(rows))
            or args.layers != 3 or args.iterations < 1 or not 1 <= args.neighbors <= args.size
            or np.any(rows < 0) or np.any(rows >= len(ids))
            or len(np.unique(rows)) != len(rows) or np.any(counts[rows] <= 0)):
        raise ValueError('fit_rows must be distinct training-seen rows and settings must be valid')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model, fit_seconds, empty = _fit(values, rows, layers=args.layers, size=args.size,
        iterations=args.iterations, neighbors=args.neighbors, beta=args.beta, seed=args.seed)
    fit_wall_seconds = time.perf_counter() - started
    np.save(args.output / 'item_ids.npy', ids, allow_pickle=False)
    np.save(args.output / 'global_mean.npy', model.mean, allow_pickle=False)
    for i, center in enumerate(model.centers, 1):
        np.save(args.output / f'centroids_layer{i}.npy', center, allow_pickle=False)
    codes_path = args.output / 'codes.npy'
    codes = np.lib.format.open_memmap(codes_path, mode='w+', dtype=np.uint16,
                                     shape=(len(ids), args.layers))
    started = time.perf_counter()
    for low in range(0, len(values), 65536):
        block = values[low:low + 65536]
        if not np.all(np.isfinite(block)):
            raise ValueError('nonfinite vector values')
        codes[low:low + len(block)] = model.encode(block)
        if (low // 65536) % 32 == 0:
            print('assigned', low + len(block), 'of', len(ids), flush=True)
    codes.flush()
    assignment_seconds = time.perf_counter() - started
    receipt = dict(source='prq-kmeans', layers=args.layers, size=args.size, catalog_items=len(ids),
        fit_items=len(rows), iterations=args.iterations, nredo=1, beta=args.beta, neighbors=args.neighbors,
        seed=args.seed, threads=configured['configured_threads'], faiss_version=faiss.__version__,
        input_sha256=file_sha256(args.vectors), item_ids_sha256=file_sha256(args.item_ids),
        train_count_sha256=file_sha256(args.train_count), fit_rows_sha256=file_sha256(args.fit_rows),
        codes_sha256=file_sha256(codes_path), producer_sha256=file_sha256(__file__),
        fit_seconds=fit_seconds, fit_wall_seconds=fit_wall_seconds,
        assignment_seconds=assignment_seconds, empty_centroids_per_iteration=empty,
        resources=process_metrics('cpu'))
    with (args.output / 'manifest.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
