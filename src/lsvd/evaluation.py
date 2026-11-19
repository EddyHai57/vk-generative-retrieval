"""Fixed-query item recall with one history filter and paired user bootstrap.

Labels and diagnostic bands are consumed verbatim from the dataset arrays.
Bootstrap draws use Poisson(1) weights in query order, shared across every
slice, K and model when the seed and query set agree.

Evaluation identity binds the arrays known to this evaluator. Model feature
inputs still require matching experiment manifests for a valid comparison.
"""
from dataclasses import dataclass, field
import csv
import hashlib
from itertools import islice
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np


PADDING_ITEM_ID = int(np.iinfo(np.uint32).max)
RULERS = {'main': 'main_positive', 'secondary': 'secondary_positive', 'like': 'like_positive'}
PLACE_SCOPES = ('0/1', '0', '1', 'all', '2')
SEEN_SCOPES = ('all', 'seen', 'unseen')
BANDS = (
    ('all', 'all', None, None),
    ('duration', '≤60秒', 'duration_band', 0),
    ('duration', '61–120秒', 'duration_band', 1),
    ('duration', '>120秒', 'duration_band', 2),
    ('duration_sensitive', '≤19秒', 'duration_band_sensitive', 0),
    ('duration_sensitive', '20–53秒', 'duration_band_sensitive', 1),
    ('duration_sensitive', '>53秒', 'duration_band_sensitive', 2),
    ('popularity', '未见', 'popularity_tier', 0),
    ('popularity', '长尾', 'popularity_tier', 1),
    ('popularity', '中部', 'popularity_tier', 2),
    ('popularity', '头部', 'popularity_tier', 3),
)


def _read_npz(path):
    # Materialize each compressed column exactly once, then close the archive.
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _hash_array(digest, name, values):
    values = np.asarray(values)
    if values.dtype.hasobject:
        raise ValueError('evaluation identity requires non-object arrays')
    header = json.dumps([name, values.dtype.str, values.shape], separators=(',', ':')).encode()
    digest.update(len(header).to_bytes(8, 'big'))
    digest.update(header)
    if not values.size:
        return
    chunk_bytes = 8 * 1024 * 1024
    if values.flags.c_contiguous:
        byte_view = memoryview(values).cast('B')
        for start in range(0, len(byte_view), chunk_bytes):
            digest.update(byte_view[start:start + chunk_bytes])
    else:
        # Buffered iteration copies only a bounded chunk of a strided view.
        for chunk in np.nditer(values, flags=['external_loop', 'buffered'],
                               op_flags=['readonly', 'contig'], order='C',
                               buffersize=max(1, chunk_bytes // values.itemsize)):
            digest.update(memoryview(chunk).cast('B'))


def _slice_specs():
    return tuple(
        dict(slice_id=index, ruler=ruler, place_scope=place, seen_scope=seen,
             breakdown=breakdown, band=band)
        for index, (ruler, place, seen, breakdown, band) in enumerate(
            (ruler, place, seen, breakdown, band)
            for ruler in RULERS for place in PLACE_SCOPES for seen in SEEN_SCOPES
            for breakdown, band, _, _ in BANDS))


@dataclass
class EvaluationData:
    user_ids: np.ndarray
    history_user_ids: np.ndarray
    history_offsets: np.ndarray
    history_items: np.ndarray
    targets: dict
    items: dict
    _counts: np.ndarray | None = field(default=None, repr=False)
    _target_users: np.ndarray | None = field(default=None, repr=False)
    _statistics_identity: str | None = field(default=None, repr=False)

    @classmethod
    def load(cls, root):
        root = Path(root)
        if (root / 'dataset' / 'arrays').is_dir():
            root = root / 'dataset' / 'arrays'
        elif (root / 'arrays').is_dir():
            root = root / 'arrays'
        queries = _read_npz(root / 'queries.npz')
        history = _read_npz(root / 'history.npz')
        targets = _read_npz(root / 'targets.npz')
        items = _read_npz(root / 'items.npz')
        result = cls(queries['user_id'], history['user_id'], history['offsets'],
                     np.load(root / 'history_items.npy', mmap_mode='r', allow_pickle=False), targets, items)
        if len(targets['offsets']) != len(result.user_ids) + 1:
            raise ValueError('target offsets must align with query users')
        for values in (result.user_ids, result.history_user_ids, items['item_id']):
            if np.any(values[1:] <= values[:-1]):
                raise ValueError('query, history and catalog IDs must be sorted and unique')
        for offsets, length in ((targets['offsets'], len(targets['item_id'])),
                                (result.history_offsets, len(result.history_items))):
            if offsets[0] != 0 or offsets[-1] != length or np.any(offsets[1:] < offsets[:-1]):
                raise ValueError('invalid ragged array offsets')
        if len(result.history_offsets) != len(result.history_user_ids) + 1:
            raise ValueError('history offsets must align with history users')
        for name in ('place', *RULERS.values(), 'train_seen', 'duration_band',
                     'duration_band_sensitive', 'popularity_tier'):
            if len(targets[name]) != len(targets['item_id']):
                raise ValueError(f'target column length mismatch: {name}')
        return result

    def evaluation_identity(self):
        """Hash current known inputs in canonical name and C-element order.

        Arrays remain mutable, so this identity is recomputed for each evaluate.
        Large contiguous arrays are hashed through bounded memoryview slices.
        """
        digest = hashlib.sha256(b'lsvd.evaluation_identity.v1\0')
        for name, values in (
                ('queries.user_id', self.user_ids),
                ('history.user_id', self.history_user_ids),
                ('history.offsets', self.history_offsets),
                ('history.items', self.history_items)):
            _hash_array(digest, name, values)
        for namespace, columns in (('targets', self.targets), ('items', self.items)):
            for name in sorted(columns):
                _hash_array(digest, f'{namespace}.{name}', columns[name])
        return digest.hexdigest()

    def seen_items(self, user_id):
        index = int(np.searchsorted(self.history_user_ids, user_id))
        if index == len(self.history_user_ids) or self.history_user_ids[index] != user_id:
            return self.history_items[:0]
        start, end = self.history_offsets[index:index + 2]
        return self.history_items[int(start):int(end)]

    def target_users(self):
        if self._target_users is None:
            self._target_users = np.repeat(np.arange(len(self.user_ids), dtype=np.int32),
                                           np.diff(self.targets['offsets']).astype(np.int64))
        return self._target_users


class CatalogMembership:
    """Reusable exact membership for a sorted catalog, packed for UInt32 IDs."""

    def __init__(self, catalog_item_ids):
        self.catalog = np.asarray(catalog_item_ids)
        self.bitmap = None
        self.max_item_id = int(self.catalog[-1]) if len(self.catalog) else -1
        if not len(self.catalog):
            self.bitmap = np.zeros(0, dtype=np.uint8)
        elif (self.catalog.dtype.kind in 'iu' and self.catalog[0] >= 0
              and self.max_item_id <= np.iinfo(np.uint32).max):
            self.bitmap = np.zeros(self.max_item_id // 8 + 1, dtype=np.uint8)
            for start in range(0, len(self.catalog), 1024 * 1024):
                items = self.catalog[start:start + 1024 * 1024]
                bits = np.left_shift(np.uint8(1), (items & 7).astype(np.uint8))
                # Repeated byte indices must accumulate every bit in the byte.
                np.bitwise_or.at(self.bitmap, items >> 3, bits)

    def contains(self, values):
        values = np.asarray(values)
        present = np.zeros(values.shape, dtype=bool)
        if not values.size:
            return present
        if values.dtype.kind not in 'iu':
            raise ValueError('ranked item IDs must be integers')
        if self.bitmap is None:
            positions = np.searchsorted(self.catalog, values)
            present = positions < len(self.catalog)
            present &= self.catalog[np.minimum(positions, len(self.catalog) - 1)] == values
            return present
        valid = (values >= 0) & (values <= self.max_item_id)
        indices = values[valid].astype(np.uint32)
        bits = np.left_shift(np.uint8(1), (indices & 7).astype(np.uint8))
        present[valid] = (self.bitmap[indices >> 3] & bits) != 0
        return present


def filter_ranked_items(ranked_items, seen_items, catalog_item_ids, limit=1000):
    """Prepare at most limit items from a complete ranking, preserving order.

    The same function is applied again by evaluate to its capped input. Catalog
    IDs must be sorted, or supplied as reusable CatalogMembership. Padding is
    ignored. Unknown IDs and nonintegers fail.
    """
    if not isinstance(limit, (int, np.integer)) or limit < 0:
        raise ValueError('limit must be a nonnegative integer')
    catalog = catalog_item_ids if isinstance(catalog_item_ids, CatalogMembership) else np.asarray(catalog_item_ids)
    seen = seen_items if isinstance(seen_items, np.ndarray) else np.asarray(list(seen_items))
    output = []
    emitted = set()
    iterator = iter(ranked_items)
    while len(output) < limit:
        batch = list(islice(iterator, 4096))
        if not batch:
            break
        values = np.asarray(batch)
        if values.dtype.kind not in 'iu':
            raise ValueError('ranked item IDs must be integers')
        values = values[values != PADDING_ITEM_ID]
        if isinstance(catalog, CatalogMembership):
            valid = catalog.contains(values)
        else:
            positions = np.searchsorted(catalog, values)
            valid = positions < len(catalog)
            if len(catalog):
                valid &= catalog[np.minimum(positions, len(catalog) - 1)] == values
        if not np.all(valid):
            raise ValueError('prediction contains an unknown catalog item ID')
        values = values[~np.isin(values, seen, assume_unique=False)]
        for item in values:
            item = int(item)
            if item not in emitted:
                emitted.add(item)
                output.append(item)
                if len(output) == limit:
                    break
    return output


def _mask(data, spec):
    targets = data.targets
    mask = targets[RULERS[spec['ruler']]].astype(bool)
    place = spec['place_scope']
    if place == '0/1':
        mask &= (targets['place'] == 0) | (targets['place'] == 1)
    elif place != 'all':
        mask &= targets['place'] == int(place)
    if spec['seen_scope'] != 'all':
        mask &= targets['train_seen'].astype(bool) == (spec['seen_scope'] == 'seen')
    for breakdown, band, column, code in BANDS:
        if column is not None and breakdown == spec['breakdown'] and band == spec['band']:
            mask &= targets[column] == code
    return mask


def _scores(counts, hits, ks):
    denominators = np.minimum(counts[:, None, :], np.asarray(ks)[None, :, None])
    return np.divide(hits, denominators, out=np.zeros(hits.shape, dtype=np.float64),
                     where=denominators > 0)


def _bootstrap_metadata(replicates, seed):
    return dict(method='user_poisson_1', replicates=int(replicates), seed=int(seed),
                rng='numpy.default_rng.PCG64', numpy_version=np.__version__,
                user_order='queries.user_id', chunk_size=16, confidence=.95,
                zero_weight_draws='excluded independently per slice',
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def _bootstrap(counts, hits, scores, replicates, seed):
    if not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError('replicates must be a positive integer')
    n_slices, n_ks, n_users = hits.shape
    active = (counts > 0).astype(np.float64)
    score_matrix = scores.reshape(n_slices * n_ks, n_users)
    hit_matrix = hits.reshape(n_slices * n_ks, n_users).astype(np.float64)
    macro = np.full((n_slices, n_ks, replicates), np.nan)
    micro = np.full_like(macro, np.nan)
    rng = np.random.default_rng(seed)
    for start in range(0, replicates, 16):
        end = min(start + 16, replicates)
        weights = rng.poisson(1, size=(end - start, n_users)).astype(np.float64).T
        macro_denominator = (active @ weights)[:, None, :]
        micro_denominator = (counts @ weights)[:, None, :]
        macro_numerator = (score_matrix @ weights).reshape(n_slices, n_ks, end - start)
        micro_numerator = (hit_matrix @ weights).reshape(n_slices, n_ks, end - start)
        np.divide(macro_numerator, macro_denominator, out=macro[:, :, start:end],
                  where=macro_denominator > 0)
        np.divide(micro_numerator, micro_denominator, out=micro[:, :, start:end],
                  where=micro_denominator > 0)
    return macro, micro


def _interval(values):
    values = values[np.isfinite(values)]
    return np.quantile(values, [.025, .975]).tolist() if len(values) else None


def _write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


@dataclass
class EvaluationResult:
    model_name: str
    user_ids: np.ndarray
    ks: tuple
    slice_specs: tuple
    counts: np.ndarray
    hits: np.ndarray
    scores: np.ndarray
    metrics: list
    metadata: dict

    def save(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(path / 'statistics.npz', user_ids=self.user_ids, ks=self.ks,
                            counts=self.counts, hits=self.hits)
        _write_json(path / 'slice_specs.json', self.slice_specs)
        _write_json(path / 'metrics.json', self.metrics)
        _write_json(path / 'manifest.json', dict(model_name=self.model_name, **self.metadata))
        _write_csv(path / 'metrics.csv', self.metrics)

    @classmethod
    def load(cls, output_dir):
        path = Path(output_dir)
        arrays = _read_npz(path / 'statistics.npz')
        metadata = json.loads((path / 'manifest.json').read_text())
        model_name = metadata.pop('model_name')
        ks = tuple(int(k) for k in arrays['ks'])
        return cls(model_name, arrays['user_ids'], ks,
                   tuple(json.loads((path / 'slice_specs.json').read_text())),
                   arrays['counts'], arrays['hits'], _scores(arrays['counts'], arrays['hits'], ks),
                   json.loads((path / 'metrics.json').read_text()), metadata)


@dataclass
class PairedResult:
    metrics: list
    metadata: dict

    def save(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=False)
        _write_json(path / 'manifest.json', self.metadata)
        _write_json(path / 'metrics.json', self.metrics)
        _write_csv(path / 'metrics.csv', self.metrics)


def evaluate(data, predictions, *, model_name, ks=(50, 200, 1000),
             replicates=1000, seed=2026111301, output_dir=None):
    """Evaluate capped input, preserving zero-target queries in stored arrays."""
    ks = tuple(ks)
    if not ks or len(set(ks)) != len(ks) or any(
            not isinstance(k, (int, np.integer)) or not 1 <= k <= 1000 for k in ks):
        raise ValueError('ks must be unique integers between 1 and 1000')
    is_mapping = isinstance(predictions, Mapping)
    if not is_mapping:
        predictions = np.asarray(predictions)
        if predictions.ndim != 2 or predictions.shape[0] != len(data.user_ids) or predictions.shape[1] > 1000:
            raise ValueError('prediction matrix must align with query users and have at most 1000 columns')
        if predictions.dtype.kind not in 'iu':
            raise ValueError('prediction matrix must contain integers')
    evaluation_identity = data.evaluation_identity()
    if evaluation_identity != data._statistics_identity:
        data._counts = None
        data._target_users = None
        data._statistics_identity = evaluation_identity
    catalog = CatalogMembership(data.items['item_id'])
    target_ranks = np.full(len(data.targets['item_id']), 1001, dtype=np.uint16)
    offsets = data.targets['offsets']
    for user_index, user_id in enumerate(data.user_ids):
        raw = predictions.get(int(user_id), ()) if is_mapping else predictions[user_index]
        raw = list(islice(iter(raw), 1001))
        if len(raw) > 1000:
            raise ValueError('each user may supply at most 1000 ranked input items')
        ranked = filter_ranked_items(raw, data.seen_items(user_id), catalog)
        start, end = (int(value) for value in offsets[user_index:user_index + 2])
        targets = data.targets['item_id'][start:end]
        if not ranked or not len(targets):
            continue
        positions = np.searchsorted(targets, ranked)
        valid = positions < len(targets)
        valid &= targets[np.minimum(positions, len(targets) - 1)] == ranked
        target_ranks[start + positions[valid]] = np.arange(1, len(ranked) + 1)[valid]
    specs = _slice_specs()
    n_users = len(data.user_ids)
    hits = np.zeros((len(specs), len(ks), n_users), dtype=np.uint32)
    build_counts = data._counts is None
    if build_counts:
        data._counts = np.zeros((len(specs), n_users), dtype=np.uint32)
    target_users = data.target_users()
    for index, spec in enumerate(specs):
        mask = _mask(data, spec)
        if build_counts:
            data._counts[index] = np.bincount(target_users[mask], minlength=n_users)
        for k_index, k in enumerate(ks):
            hits[index, k_index] = np.bincount(target_users[mask & (target_ranks <= k)], minlength=n_users)
    counts = data._counts
    scores = _scores(counts, hits, ks)
    macro_draws, micro_draws = _bootstrap(counts, hits, scores, replicates, seed)
    metrics = []
    for index, spec in enumerate(specs):
        n_active = int(np.count_nonzero(counts[index]))
        n_targets = int(counts[index].sum(dtype=np.uint64))
        for k_index, k in enumerate(ks):
            total_hits = int(hits[index, k_index].sum(dtype=np.uint64))
            metrics.append(dict(spec, k=int(k), n_users=n_active, n_queries=n_users, n_targets=n_targets,
                hits=total_hits,
                macro_capped=float(scores[index, k_index, counts[index] > 0].mean()) if n_active else None,
                micro_recall=total_hits / n_targets if n_targets else None,
                macro_ci95=_interval(macro_draws[index, k_index]),
                micro_ci95=_interval(micro_draws[index, k_index]),
                valid_draws=int(np.count_nonzero(np.isfinite(macro_draws[index, k_index])))))
    metadata = dict(evaluation_identity=evaluation_identity,
        catalog_validation='packed_membership' if catalog.bitmap is not None else 'sorted_catalog',
        bootstrap=_bootstrap_metadata(replicates, seed),
        candidate_preparation=getattr(predictions, 'candidate_preparation', 'unspecified'),
        input_limit=1000, filtering='stable unique, all training seen removed, then top K',
        score='hits / min(K, slice target count)', micro='total hits / total slice targets',
        empty_slice='UNKNOWN', popularity='diagnostic only, precomputed training bands',
        stored_statistics='per-user counts and hits; float64 scores reconstructed on load')
    result = EvaluationResult(model_name, data.user_ids, ks, specs, counts, hits, scores, metrics, metadata)
    if output_dir is not None:
        result.save(output_dir)
    return result


def paired_bootstrap(first, second, *, replicates=1000, seed=2026111301, output_dir=None):
    """Return first minus second with shared per-user Poisson weights."""
    identity = first.metadata.get('evaluation_identity')
    if not identity or identity != second.metadata.get('evaluation_identity'):
        raise ValueError('paired evaluations must share a nonempty evaluation identity')
    if ('foundation_sha256' in first.metadata and 'foundation_sha256' in second.metadata
            and first.metadata['foundation_sha256'] != second.metadata['foundation_sha256']):
        raise ValueError('paired evaluations must share the foundation manifest SHA256')
    if (first.ks != second.ks or first.slice_specs != second.slice_specs
            or not np.array_equal(first.user_ids, second.user_ids)
            or not np.array_equal(first.counts, second.counts)):
        raise ValueError('paired evaluations must share user IDs, target counts, slices and Ks')
    difference_scores = first.scores - second.scores
    difference_hits = first.hits.astype(np.int64) - second.hits.astype(np.int64)
    macro_draws, micro_draws = _bootstrap(first.counts, difference_hits, difference_scores, replicates, seed)
    metrics = []
    for index, spec in enumerate(first.slice_specs):
        n_users = int(np.count_nonzero(first.counts[index]))
        n_targets = int(first.counts[index].sum(dtype=np.uint64))
        for k_index, k in enumerate(first.ks):
            metrics.append(dict(spec, k=int(k), n_users=n_users, n_queries=len(first.user_ids), n_targets=n_targets,
                macro_difference=float(difference_scores[index, k_index, first.counts[index] > 0].mean()) if n_users else None,
                micro_difference=float(difference_hits[index, k_index].sum() / n_targets) if n_targets else None,
                macro_ci95=_interval(macro_draws[index, k_index]),
                micro_ci95=_interval(micro_draws[index, k_index]),
                valid_draws=int(np.count_nonzero(np.isfinite(macro_draws[index, k_index])))))
    metadata = dict(first_model=first.model_name, second_model=second.model_name,
        direction='first minus second', evaluation_identity=identity,
        bootstrap=_bootstrap_metadata(replicates, seed))
    if 'foundation_sha256' in first.metadata and 'foundation_sha256' in second.metadata:
        metadata['foundation_sha256'] = first.metadata['foundation_sha256']
    result = PairedResult(metrics, metadata)
    if output_dir is not None:
        result.save(output_dir)
    return result
