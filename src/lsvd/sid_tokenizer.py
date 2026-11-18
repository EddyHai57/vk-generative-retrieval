"""CPU residual KMeans, collision-preserving SID tables and oracle bucket ranks."""
from pathlib import Path
import hashlib
import time

import faiss
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assign_codes(values, centers):
    residual = np.array(values, dtype=np.float32, order='C', copy=True)
    codes = np.empty((len(values), len(centers)), dtype=np.uint16)
    for layer, center in enumerate(centers):
        index = faiss.IndexFlatL2(residual.shape[1])
        index.add(center)
        _, labels = index.search(residual, 1)
        codes[:, layer] = labels[:, 0]
        residual -= center[labels[:, 0]]
    return codes


def pack_codes(codes, size):
    if size ** codes.shape[1] > 2 ** 64:
        raise ValueError('packed SID exceeds uint64')
    result = np.zeros(len(codes), dtype=np.uint64)
    for layer in range(codes.shape[1]):
        result = result * np.uint64(size) + codes[:, layer].astype(np.uint64)
    return result


def build_tables(item_ids, codes, size, output):
    output = Path(output)
    packed = pack_codes(codes, size)
    order = np.argsort(packed, kind='stable')
    keys, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    offsets = np.r_[0, np.cumsum(counts, dtype=np.uint64)].astype(np.uint64)
    np.save(output / 'bucket_keys.npy', keys)
    np.save(output / 'bucket_offsets.npy', offsets)
    np.save(output / 'bucket_items.npy', item_ids[order])
    np.save(output / 'item_bucket.npy', inverse.astype(np.uint32))
    # A node's key packs only its prefix. Child tokens are sorted and unique.
    for length in range(codes.shape[1]):
        child_keys = np.unique(keys // np.uint64(size ** (codes.shape[1] - length - 1)))
        parents, child_counts = np.unique(child_keys // np.uint64(size), return_counts=True)
        np.save(output / f'prefix_{length}_keys.npy', parents)
        np.save(output / f'prefix_{length}_offsets.npy',
                np.r_[0, np.cumsum(child_counts)].astype(np.uint64))
        np.save(output / f'prefix_{length}_next.npy', (child_keys % size).astype(np.uint16))
    return keys, inverse.astype(np.uint32), counts


def item_hash(item_ids, seed=20261122):
    # SplitMix64 over public item IDs only. Overflow is defined uint64 arithmetic.
    with np.errstate(over='ignore'):
        value = np.asarray(item_ids, dtype=np.uint64) + np.uint64(seed)
        value = value + np.uint64(0x9e3779b97f4a7c15)
        value = (value ^ (value >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
        value = (value ^ (value >> 27)) * np.uint64(0x94d049bb133111eb)
        return value ^ (value >> 31)


def filtered_ranks(target_rows, history_rows, positions, bucket_starts):
    """Zero-based position within one oracle bucket after seen-item filtering."""
    history_positions = np.unique(positions[history_rows])
    target_positions = positions[target_rows]
    before = np.searchsorted(history_positions, target_positions)
    at_start = np.searchsorted(history_positions, bucket_starts[target_rows])
    ranks = target_positions - bucket_starts[target_rows] - (before - at_start)
    ranks = ranks.astype(np.int64)
    ranks[np.isin(target_positions, history_positions, assume_unique=False)] = -1
    return ranks


def fit_residual(values, rows, layers, size, output, normalize=False):
    residual = np.array(values[rows], dtype=np.float32, order='C', copy=True)
    if normalize:
        residual /= np.maximum(np.linalg.norm(residual, axis=1, keepdims=True), 1e-12)
    centers = []
    timings = []
    for layer in range(layers):
        started = time.perf_counter()
        km = faiss.Kmeans(residual.shape[1], size, niter=12, nredo=1, seed=42 + layer,
                          max_points_per_centroid=len(rows), min_points_per_centroid=1,
                          verbose=False)
        km.train(residual)
        center = np.asarray(km.centroids, dtype=np.float32)
        _, labels = km.index.search(residual, 1)
        residual -= center[labels[:, 0]]
        centers.append(center)
        np.save(Path(output) / f'centroids_layer{layer + 1}.npy', center)
        timings.append(time.perf_counter() - started)
        print({'stage': 'fit', 'layer': layer + 1, 'seconds': timings[-1]}, flush=True)
    return centers, timings
