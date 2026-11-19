"""Versioned generator inputs and bounded, strictly previous training histories."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .two_tower import Catalog, ItemTower, TowerConfig, TwoTower
from .two_tower_prepare import CHUNK, load_tensor_dataset, sha256_file


CATALOG_FIELDS = ('item_id', 'author_id', 'duration', 'train_count', 'rank', 'content')


def _json(path):
    return json.loads(Path(path).read_text())


def _publish(path, kind, names, **details):
    record = dict(format=kind, **details,
                  files={name: sha256_file(path / name) for name in sorted(names)})
    with (path / 'manifest.json').open('x') as handle:
        json.dump(record, handle, sort_keys=True, indent=2)
    return record


def _asset(path, kind, required, expected):
    manifest_path = path / 'manifest.json'
    digest = sha256_file(manifest_path)
    if expected is not None and digest != expected:
        raise ValueError('generator asset manifest sha256 mismatch')
    manifest = _json(manifest_path)
    if manifest['format'] != kind or not set(required) <= manifest['files'].keys():
        raise ValueError('invalid generator asset manifest')
    for name, expected_file_hash in manifest['files'].items():
        if Path(name).name != name or sha256_file(path / name) != expected_file_hash:
            raise ValueError(f'generator asset sha256 mismatch: {name}')
    return digest, manifest


def _npy(path, name):
    return np.load(path / f'{name}.npy', mmap_mode='c', allow_pickle=False)


def _tensor_arrays(root, manifest, names, *, verify=False):
    result = {}
    for name in names:
        row = manifest['arrays'][name]
        path = root / row['file']
        if verify and (path.stat().st_size != row['bytes'] or sha256_file(path) != row['sha256']):
            raise ValueError(f'tensor array sha256 mismatch: {name}')
        result[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    return result


def _sorted_ids(ids):
    return (ids.ndim == 1 and ids.dtype.kind in 'ui' and ids.size > 0
            and np.all(ids[1:] > ids[:-1]))


class ItemVectors:
    """Frozen item scoring and cold history tables in the same catalog order."""

    _files = ('item_ids.npy', 'full.npy', 'cold.npy', 'has_id.npy')

    def __init__(self, path, expected_sha256=None):
        path = Path(path)
        self.sha256, self.manifest = _asset(path, 'lsvd-item-vectors-v1', self._files,
                                            expected_sha256)
        self.ids, self.full, self.cold, self.has_id = (
            _npy(path, name) for name in ('item_ids', 'full', 'cold', 'has_id')
        )
        shape = (self.manifest['catalog_items'], self.manifest['dim'])
        if (not _sorted_ids(self.ids) or len(self.ids) != shape[0]
                or self.has_id.shape != self.ids.shape or self.has_id.dtype != np.bool_
                or any(array.shape != shape or array.dtype != np.float16
                       for array in (self.full, self.cold))):
            raise ValueError('item vector row identity or dtype mismatch')
        for start in range(0, len(self.ids), 65536):
            rows = slice(start, start + 65536)
            no_id = ~self.has_id[rows]
            if not np.array_equal(self.full[rows][no_id], self.cold[rows][no_id]):
                raise ValueError('item vector scoring table differs from cold on an item without ID')


def _legacy_checkpoint_binding(checkpoint, config, manifest, columns, provenance,
                               foundation, metadata):
    """Bind an old smoke checkpoint through its export receipt and original join."""
    from .two_tower_runtime import duckdb_connection

    receipt = _json(provenance)
    foundation = Path(foundation)
    checks = {
        'checkpoint_sha256': sha256_file(checkpoint),
        'foundation_sha256': sha256_file(foundation / 'foundation.json'),
        'metadata_sha256': sha256_file(metadata),
        'full_catalog_items': len(columns['item_id']),
        'item_id_mapping_sha256': manifest['sources']['vectors/item_id.npy']['sha256'],
        'input_vectors_sha256': manifest['arrays']['catalog_content']['sha256'],
    }
    if (any(receipt.get(key) != value for key, value in checks.items())
            or receipt['config'] != config
            or sha256_file(foundation / 'items.parquet')
            != manifest['sources']['foundation/items.parquet']['sha256']):
        raise ValueError('legacy checkpoint or original catalog provenance differs')
    with duckdb_connection() as connection:
        cursor = connection.execute('''
            SELECT i.item_id, i.author_id, i.duration, i.train_count,
                   m.train_interactions_rank
            FROM read_parquet(?) i JOIN read_parquet(?) m USING (item_id)
            ORDER BY i.item_id
        ''', [str(foundation / 'items.parquet'), str(metadata)])
        start = 0
        while block := cursor.fetchmany(65536):
            end = start + len(block)
            values = np.asarray(block)
            if any(not np.array_equal(columns[name][start:end], values[:, col])
                   for col, name in enumerate(CATALOG_FIELDS[:-1])):
                raise ValueError('legacy checkpoint catalog row/ID/author columns differ')
            start = end
        if start != checks['full_catalog_items']:
            raise ValueError('legacy checkpoint catalog length differs')
    return dict(kind='legacy-smoke', provenance_sha256=sha256_file(provenance), **checks)


def _checkpoint_inputs(tensor_dir, checkpoint, *, allow_smoke, legacy_provenance,
                       legacy_foundation, legacy_metadata, verify_catalog):
    tensor_dir, checkpoint = Path(tensor_dir), Path(checkpoint)
    manifest_path = tensor_dir / 'manifest.json'
    manifest = _json(manifest_path)
    tensor_sha = sha256_file(manifest_path)
    if manifest['format'] != 'lsvd-two-tower-tensors-v1':
        raise ValueError('unsupported tensor dataset format')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    config = TowerConfig(**saved['config'])
    names = ['catalog_' + name for name in CATALOG_FIELDS]
    bound_identity = saved.get('training_state', {}).get('data_identity')
    if bound_identity is not None and bound_identity != tensor_sha:
        raise ValueError('two-tower checkpoint tensor dataset identity differs')
    if not saved.get('is_best') and not allow_smoke:
        raise ValueError('item export requires a best checkpoint or an explicit smoke checkpoint')
    arrays = _tensor_arrays(tensor_dir, manifest, names,
                            verify=verify_catalog or bound_identity is None)
    columns = {name: arrays['catalog_' + name] for name in CATALOG_FIELDS}
    if not _sorted_ids(columns['item_id']):
        raise ValueError('checkpoint catalog IDs must be strictly sorted')
    if bound_identity is None:
        if not (allow_smoke and legacy_provenance and legacy_foundation and legacy_metadata):
            raise ValueError('checkpoint lacks tensor dataset identity and verified legacy provenance')
        binding = _legacy_checkpoint_binding(checkpoint, saved['config'], manifest, columns,
                                             legacy_provenance, legacy_foundation, legacy_metadata)
        if _json(legacy_provenance)['structure'] != saved['structure']:
            raise ValueError('legacy checkpoint structure differs')
    else:
        binding = dict(kind='best' if saved['is_best'] else 'smoke',
                       checkpoint_sha256=sha256_file(checkpoint))
    binding['tensor_sha256'] = tensor_sha
    binding['catalog_sha256'] = {name: manifest['arrays'][name]['sha256'] for name in names}
    return saved, config, columns, binding


def load_external_tower(tensor_dir, checkpoint, device='cpu', *, allow_smoke=False,
                        legacy_provenance=None, legacy_foundation=None, legacy_metadata=None):
    """Load the external scoring tower and its correctly indexed history reader."""
    from .two_tower_run import STRUCTURES

    saved, config, _, _ = _checkpoint_inputs(
        tensor_dir, checkpoint, allow_smoke=allow_smoke, legacy_provenance=legacy_provenance,
        legacy_foundation=legacy_foundation, legacy_metadata=legacy_metadata, verify_catalog=False,
    )
    _, catalog, store, _ = load_tensor_dataset(tensor_dir, config)
    structure, _ = STRUCTURES[saved['structure']]
    model = TwoTower(catalog, config, structure)
    model.load_state_dict(saved['model'])
    return model.eval().to(device), catalog, store


def export_item_vectors(tensor_dir, checkpoint, output, device='cpu', *, batch_size=8192,
                        allow_smoke=False, legacy_provenance=None, legacy_foundation=None,
                        legacy_metadata=None):
    """Export normalized item vectors. The scoring table falls back to cold without ID."""
    if batch_size <= 0:
        raise ValueError('export batch size must be positive')
    saved, config, columns, binding = _checkpoint_inputs(
        tensor_dir, checkpoint, allow_smoke=allow_smoke, legacy_provenance=legacy_provenance,
        legacy_foundation=legacy_foundation, legacy_metadata=legacy_metadata, verify_catalog=True,
    )
    catalog = Catalog(columns, config)
    model = ItemTower(config, catalog.id_count, catalog.author_count)
    model.load_state_dict({key.removeprefix('item.'): value for key, value in saved['model'].items()
                           if key.startswith('item.')})
    model = model.eval().to(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    n = len(catalog.ids)
    has_id = catalog.id_index != 0
    np.save(output / 'item_ids.npy', catalog.ids, allow_pickle=False)
    np.save(output / 'has_id.npy', has_id, allow_pickle=False)
    full, cold = [np.lib.format.open_memmap(output / f'{name}.npy', mode='w+',
                                           dtype=np.float16, shape=(n, config.dim))
                  for name in ('full', 'cold')]
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            rows = slice(start, start + batch_size)
            features = catalog.features(catalog.ids[rows], device)
            normal = model(features).cpu().numpy()
            fallback = model(features, cold=True).cpu().numpy()
            if not np.isfinite(normal).all() or not np.isfinite(fallback).all():
                raise ValueError('nonfinite item vector')
            cold[rows] = fallback
            full[rows] = np.where(has_id[rows, None], normal, fallback)
    full.flush()
    cold.flush()
    return _publish(output, 'lsvd-item-vectors-v1', ItemVectors._files, catalog_items=n,
                    dim=config.dim, checkpoint=binding, structure=saved['structure'],
                    id_min_count=config.id_min_count, rows_with_id=int(has_id.sum()))


def _sid_tables():
    return ['codes', 'bucket_keys', 'bucket_offsets', 'bucket_items', 'item_bucket'] + [
        f'prefix_{length}_{field}' for length in range(3)
        for field in ('keys', 'offsets', 'next')
    ]


def _csr(keys, offsets, values):
    return (all(x.ndim == 1 and x.dtype.kind in 'ui' for x in (keys, offsets, values))
            and _sorted_ids(keys) and len(offsets) == len(keys) + 1
            and offsets[0] == 0 and offsets[-1] == len(values)
            and np.all(offsets[1:] > offsets[:-1]))


def _bucket_answer_counts(item_bucket, item_counts, n_buckets):
    counts = np.zeros(n_buckets, dtype=np.int64)
    for start in range(0, len(item_counts), CHUNK):
        rows = slice(start, start + CHUNK)
        np.add.at(counts, item_bucket[rows], item_counts[rows])
    return counts


class SIDBundle:
    """Three-level raw SID codes, collision CSR and both answer frequencies."""

    _files = tuple(f'{name}.npy' for name in [
        'item_ids', *_sid_tables(), 'answer_counts', 'item_answer_counts',
    ])

    def __init__(self, path, expected_sha256=None):
        path = Path(path)
        self.sha256, self.manifest = _asset(path, 'lsvd-generator-sid-v1', self._files,
                                            expected_sha256)
        self.size = self.manifest['size']
        self.ids = _npy(path, 'item_ids')
        for name in (*_sid_tables()[:5], 'answer_counts', 'item_answer_counts'):
            setattr(self, name, _npy(path, name))
        self.prefixes = [tuple(_npy(path, f'prefix_{level}_{kind}')
                              for kind in ('keys', 'offsets', 'next')) for level in range(3)]
        self._verify_rows()

    def _verify_rows(self):
        n, buckets = len(self.ids), len(self.bucket_keys)
        if (not isinstance(self.size, int) or not 1 <= self.size <= 65536
                or self.manifest['layers'] != 3 or self.manifest['catalog_items'] != n
                or not _sorted_ids(self.ids) or self.codes.shape != (n, 3)
                or self.codes.dtype != np.uint16 or self.item_bucket.shape != (n,)
                or self.item_bucket.dtype.kind not in 'ui' or self.bucket_items.shape != (n,)
                or not _csr(self.bucket_keys, self.bucket_offsets, self.bucket_items)):
            raise ValueError('invalid SID item or bucket row identity')
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            row = slice(start, stop)
            digits = self.codes[row].astype(np.uint64)
            if np.any(digits >= self.size):
                raise ValueError('SID token out of range')
            packed = (digits[:, 0] * self.size + digits[:, 1]) * self.size + digits[:, 2]
            bucket = self.item_bucket[row]
            if np.any(bucket < 0) or np.any(bucket >= buckets) or np.any(self.bucket_keys[bucket] != packed):
                raise ValueError('SID code and bucket row identity differ')

            item = self.bucket_items[row]
            item_rows = np.searchsorted(self.ids, item)
            offsets = self.bucket_offsets[1:]
            group = np.searchsorted(offsets, np.arange(start, stop, dtype=offsets.dtype),
                                    side='right')
            prior_same = group[0] == previous_group if start else False
            if (np.any(item_rows >= n) or np.any(self.ids[item_rows] != item)
                    or np.any(self.item_bucket[item_rows] != group)
                    or np.any((group[1:] == group[:-1]) & (item[1:] <= item[:-1]))
                    or (prior_same and item[0] <= previous_item)):
                raise ValueError('SID bucket item row identity differs')
            previous_group, previous_item = group[-1], item[-1]

        for level, (keys, offsets, children) in enumerate(self.prefixes):
            if not _csr(keys, offsets, children):
                raise ValueError('invalid SID prefix CSR')
            factor = np.uint64(self.size ** (2 - level))
            cursor, previous = 0, -1
            for start in range(0, buckets, CHUNK):
                expected = np.unique(self.bucket_keys[start:start + CHUNK] // factor)
                if previous == expected[0]:
                    expected = expected[1:]
                if not len(expected):
                    continue
                stop = cursor + len(expected)
                parent = np.searchsorted(offsets[1:], np.arange(cursor, stop, dtype=offsets.dtype),
                                         side='right')
                tokens = children[cursor:stop]
                if (stop > len(children) or np.any(tokens >= self.size)
                        or np.any(tokens < 0)
                        or not np.array_equal(keys[parent] * self.size + tokens, expected)):
                    raise ValueError('SID prefixes differ from full-code buckets')
                cursor, previous = stop, expected[-1]
            if cursor != len(children):
                raise ValueError('SID prefixes differ from full-code buckets')

        if (self.answer_counts.shape != (buckets,) or self.item_answer_counts.shape != (n,)
                or any(v.dtype.kind not in 'ui' or np.any(v < 0)
                       for v in (self.answer_counts, self.item_answer_counts))
                or self.item_answer_counts.sum(dtype=np.int64) != self.manifest['main_answers']
                or not np.array_equal(self.answer_counts,
                                      _bucket_answer_counts(self.item_bucket,
                                                            self.item_answer_counts, buckets))):
            raise ValueError('SID main answer counts differ from bucket mapping')


def _main_offsets(main, place):
    for start in range(0, len(main), CHUNK):
        rows = slice(start, start + CHUNK)
        yield start, np.flatnonzero(main[rows] & (place[rows] <= 1))


def fit_sid_codes(tensor_dir, item_vectors, expected_vectors_sha256, output, *,
                  size=1024, fit_items=1_000_000):
    """Write a three-layer cold SID producer directory for prepare_sid_bundle."""
    from . import sid_tokenizer

    if not 2 <= size <= 65536 or not size <= fit_items <= 1_000_000:
        raise ValueError('require 2 <= codebook size <= fit items <= 1,000,000, with uint16 codes')
    tensor_dir, output = Path(tensor_dir), Path(output)
    manifest_path = tensor_dir / 'manifest.json'
    tensor_sha = sha256_file(manifest_path)
    vectors = ItemVectors(item_vectors, expected_vectors_sha256)
    arrays = _tensor_arrays(tensor_dir, _json(manifest_path),
                            ('catalog_item_id', 'catalog_train_count'), verify=True)
    if (vectors.manifest['checkpoint']['tensor_sha256'] != tensor_sha
            or not np.array_equal(vectors.ids, arrays['catalog_item_id'])):
        raise ValueError('cold item vectors differ from the fitted SID catalog')
    seen = np.flatnonzero(arrays['catalog_train_count'] > 0)
    sample_count = min(fit_items, len(seen))
    if sample_count < size:
        raise ValueError('too few training-seen items for the SID codebook')
    seed = 42
    fitted = np.sort(np.random.default_rng(seed).choice(seen, sample_count, replace=False))
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'item_ids.npy', vectors.ids, allow_pickle=False)
    np.save(output / 'fit_rows.npy', fitted, allow_pickle=False)
    centers, _ = sid_tokenizer.fit_residual(vectors.cold, fitted, 3, size, output)
    codes = np.lib.format.open_memmap(output / 'codes.npy', mode='w+', dtype=np.uint16,
                                     shape=(len(vectors.ids), 3))
    for start in range(0, len(codes), 65536):
        rows = slice(start, start + 65536)
        codes[rows] = sid_tokenizer.assign_codes(vectors.cold[rows], centers)
    codes.flush()
    occupied_codes = len(sid_tokenizer.build_tables(vectors.ids, codes, size, output)[0])
    return _publish(output, 'lsvd-sid-producer-v1', [path.name for path in output.glob('*.npy')],
                    source='tower-cold', layers=3, size=size, catalog_items=len(vectors.ids),
                    occupied_codes=occupied_codes, item_ids='item_ids.npy', tensor_sha256=tensor_sha,
                    input_sha256=vectors.manifest['files']['cold.npy'],
                    item_vectors_sha256=vectors.sha256, train_seen_items=len(seen),
                    fit_items=sample_count, fit_seed=seed,
                    fit_rows_sha256=sha256_file(output / 'fit_rows.npy'),
                    tokenizer_sha256=sha256_file(Path(sid_tokenizer.__file__)),
                    faiss_version=sid_tokenizer.faiss.__version__,
                    faiss_threads=sid_tokenizer.faiss.omp_get_max_threads())


def prepare_sid_bundle(existing_dir, tensor_dir, output, *, expected_main_answers=None):
    """Package a three-layer round22-format producer export with main-ruler counts."""
    existing_dir, tensor_dir, output = map(Path, (existing_dir, tensor_dir, output))
    source_manifest = existing_dir / 'manifest.json'
    source_hash = sha256_file(source_manifest)
    source = _json(source_manifest)
    if source['layers'] != 3:
        raise ValueError('generator requires three SID layers')
    ids_file = existing_dir / source.get('item_ids', 'item_ids.npy')
    if source.get('item_ids') not in (None, 'item_ids.npy', '../inputs/item_ids.npy'):
        raise ValueError('unsupported SID producer item ID path')
    manifest_path = tensor_dir / 'manifest.json'
    manifest = _json(manifest_path)
    tensor_sha = sha256_file(manifest_path)
    arrays = _tensor_arrays(tensor_dir, manifest, ('catalog_item_id', 'event_main',
                                                  'event_place', 'event_catalog_row', 'event_item'),
                            verify=True)
    ids = np.load(ids_file, mmap_mode='r', allow_pickle=False)
    if not np.array_equal(ids, arrays['catalog_item_id']) or source['catalog_items'] != len(ids):
        raise ValueError('SID producer item ID rows differ from tensor dataset')
    names = [f'{name}.npy' for name in _sid_tables()]
    centroid_files = [f'centroids_layer{level}.npy' for level in range(1, 4)]
    if any((existing_dir / name).exists() for name in centroid_files):
        names += centroid_files
    snapshots = existing_dir.parent / 'artifact_versions.json'
    producer_binding = {}
    if source.get('format') == 'lsvd-sid-producer-v1':
        _asset(existing_dir, 'lsvd-sid-producer-v1', [*names, 'item_ids.npy', 'fit_rows.npy'],
               source_hash)
        if (source['tensor_sha256'] != tensor_sha
                or source['fit_rows_sha256'] != source['files']['fit_rows.npy']):
            raise ValueError('fitted SID producer identity differs from the tensor dataset')
        producer_binding['item_vectors_sha256'] = source['item_vectors_sha256']
        names.append('fit_rows.npy')
    elif snapshots.is_file():
        pins = _json(snapshots)[existing_dir.name]
        if source_hash != pins['manifest.json'] or any(
                sha256_file(existing_dir / name) != pins[name] for name in names):
            raise ValueError('existing SID producer input sha256 differs')

    counts = np.zeros(len(ids), dtype=np.int32)
    for start, local in _main_offsets(arrays['event_main'], arrays['event_place']):
        positions = start + local
        rows = arrays['event_catalog_row'][positions]
        if (np.any(rows < 0) or np.any(rows >= len(ids))
                or not np.array_equal(ids[rows], arrays['event_item'][positions])):
            raise ValueError('main-ruler event row differs from catalog item ID')
        np.add.at(counts, rows, 1)
    total = int(counts.sum(dtype=np.int64))
    if total <= 0 or (expected_main_answers is not None and total != expected_main_answers):
        raise ValueError(f'main-ruler answer count differs: {total}')
    buckets = _npy(existing_dir, 'item_bucket')
    bucket_count = len(_npy(existing_dir, 'bucket_keys'))
    bucket_counts = _bucket_answer_counts(buckets, counts, bucket_count)
    output.mkdir(parents=True, exist_ok=False)
    for name in names:
        os.link(existing_dir / name, output / name)
    os.link(ids_file, output / 'item_ids.npy')
    np.save(output / 'item_answer_counts.npy', counts, allow_pickle=False)
    np.save(output / 'answer_counts.npy', bucket_counts, allow_pickle=False)
    return _publish(output, 'lsvd-generator-sid-v1', [*names, 'item_ids.npy',
                                                     'item_answer_counts.npy', 'answer_counts.npy'],
                    layers=3, size=source['size'], catalog_items=len(ids), main_answers=total,
                    tensor_sha256=tensor_sha, producer_sha256=source_hash,
                    source=source.get('source'), input_sha256=source.get('input_sha256'),
                    fit_rows_sha256=source.get('fit_rows_sha256'), **producer_binding)


def _indices(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'ui':
        raise ValueError('expected a one-dimensional integer index')
    return values


class GeneratorData:
    """Online gathers from the training-only columns and frozen item assets."""

    def __init__(self, tensor_dir, sid, vectors, device='cpu', *, max_history=200,
                 expected_tensor_sha256=None):
        if max_history <= 0:
            raise ValueError('max_history must be positive')
        tensor_dir = Path(tensor_dir)
        self.tensor_sha256 = sha256_file(tensor_dir / 'manifest.json')
        if expected_tensor_sha256 is not None and self.tensor_sha256 != expected_tensor_sha256:
            raise ValueError('generator tensor dataset identity differs')
        self.sid = sid if isinstance(sid, SIDBundle) else SIDBundle(sid)
        self.vectors = vectors if isinstance(vectors, ItemVectors) else ItemVectors(vectors)
        if (self.sid.manifest['tensor_sha256'] != self.tensor_sha256
                or self.vectors.manifest['checkpoint']['tensor_sha256'] != self.tensor_sha256):
            raise ValueError('generator asset tensor dataset identity differs')
        fitted_vectors = self.sid.manifest.get('item_vectors_sha256')
        if ((fitted_vectors is not None or self.sid.manifest.get('source') == 'tower-cold')
                and fitted_vectors != self.vectors.sha256):
            raise ValueError('SID was fitted on a different frozen item-vector version')
        config = TowerConfig(id_min_count=self.vectors.manifest['id_min_count'])
        self.arrays, self.catalog, self.store, self.evaluation = load_tensor_dataset(tensor_dir, config)
        self.ids = self.catalog.ids
        if (not np.array_equal(self.sid.ids, self.ids)
                or not np.array_equal(self.vectors.ids, self.ids)
                or not np.array_equal(self.vectors.has_id, self.catalog.id_index != 0)):
            raise ValueError('generator SID, vectors and catalog row identity differ')
        self.device = torch.device(device)
        self.max_history = max_history
        self._codes = torch.from_numpy(self.sid.codes)
        self._cold = torch.from_numpy(self.vectors.cold)
        expected_main = self.sid.manifest['main_answers']
        self.main_positions = np.empty(expected_main, dtype=np.int32)
        cursor = 0
        for start, local in _main_offsets(self.store.columns['main'], self.store.columns['place']):
            stop = cursor + len(local)
            if stop > expected_main:
                raise ValueError('main-ruler positions differ from SID answer count')
            self.main_positions[cursor:stop] = start + local
            cursor = stop
        if cursor != expected_main:
            raise ValueError('main-ruler positions differ from SID answer count')
        self.main_positions.setflags(write=False)

    def _to_device(self, values):
        return torch.from_numpy(values).to(self.device)

    def _history(self, ends, users, weeks):
        history = self.store.history_batch(ends, users, weeks, self.max_history, 'cpu',
                                           context_only=True)
        valid = history['valid']
        rows = history.pop('catalog_rows').to(dtype=torch.int64).reshape(-1)
        codes = torch.index_select(self._codes, 0, rows).long().reshape(*valid.shape, 3)
        cold = torch.index_select(self._cold, 0, rows).reshape(*valid.shape, self._cold.shape[1])
        history['codes'] = codes.masked_fill_(~valid[..., None], 0)
        history['cold'] = cold.masked_fill_(~valid[..., None], 0)
        return {key: value.to(self.device) for key, value in history.items()}

    def training_batch(self, positions):
        positions = _indices(positions)
        columns = self.store.columns
        if (np.any(positions < 0) or np.any(positions >= len(columns['item']))
                or not np.all(columns['main'][positions] & (columns['place'][positions] <= 1))):
            raise ValueError('positions must be main-ruler answers at place 0/1')
        users = columns['user_row'][positions]
        history = self._history(positions, users, columns['week'][positions])
        rows = columns['catalog_row'][positions]
        ids = columns['item'][positions]
        counts = self.sid.item_answer_counts[rows]
        if (not np.array_equal(self.ids[rows], ids) or np.any(counts <= 0)):
            raise ValueError('generator target catalog row or proposal frequency differs')
        return (history, self._to_device(self.sid.codes[rows].astype(np.int64)),
                self._to_device(self.vectors.full[rows]), self._to_device(ids),
                self.store.positives.batch(users, torch.from_numpy(ids)),
                self._to_device((counts / len(self.main_positions)).astype(np.float32)),
                self._to_device(np.ones(len(positions), dtype=np.float32)))

    def query_batch(self, user_ids):
        user_ids = _indices(user_ids)
        users = np.searchsorted(self.store.user_ids, user_ids)
        bounded = np.minimum(users, len(self.store.user_ids) - 1)
        found = (users < len(self.store.user_ids)) & (self.store.user_ids[bounded] == user_ids)
        starts = self.store.offsets[bounded]
        ends = np.where(found, self.store.offsets[bounded + 1], starts)
        return self._history(ends, bounded, np.full(len(users), 25, dtype=np.int64))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    vectors = subcommands.add_parser('export-vectors')
    vectors.add_argument('--checkpoint', type=Path, required=True)
    vectors.add_argument('--device', default='cpu')
    vectors.add_argument('--batch-size', type=int, default=8192)
    vectors.add_argument('--allow-smoke', action='store_true')
    vectors.add_argument('--legacy-provenance', type=Path)
    vectors.add_argument('--legacy-foundation', type=Path)
    vectors.add_argument('--legacy-metadata', type=Path)
    sid = subcommands.add_parser('prepare-sid')
    sid.add_argument('--existing-dir', type=Path, required=True)
    sid.add_argument('--expected-main-answers', type=int)
    fit = subcommands.add_parser('fit-sid', help='fit cold codes, then run prepare-sid')
    fit.add_argument('--item-vectors', type=Path, required=True)
    fit.add_argument('--item-vectors-sha256', required=True)
    fit.add_argument('--size', type=int, default=1024)
    fit.add_argument('--fit-items', type=int, default=1_000_000)
    for command in (vectors, sid, fit):
        command.add_argument('--tensor-dir', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--threads', type=int)
        command.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(action='dry_run', command=args.command, output=str(args.output))))
        return 0
    from .two_tower_runtime import configure_threads

    configure_threads(args.threads)
    action = 'prepared'
    if args.command == 'export-vectors':
        export_item_vectors(args.tensor_dir, args.checkpoint, args.output, args.device,
                            batch_size=args.batch_size, allow_smoke=args.allow_smoke,
                            legacy_provenance=args.legacy_provenance, legacy_foundation=args.legacy_foundation,
                            legacy_metadata=args.legacy_metadata)
    elif args.command == 'fit-sid':
        fit_sid_codes(args.tensor_dir, args.item_vectors, args.item_vectors_sha256, args.output,
                      size=args.size, fit_items=args.fit_items)
        action = 'fit_sid'
    else:
        prepare_sid_bundle(args.existing_dir, args.tensor_dir, args.output,
                           expected_main_answers=args.expected_main_answers)
    print(json.dumps(dict(action=action, command=args.command,
                          manifest_sha256=sha256_file(args.output / 'manifest.json'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
