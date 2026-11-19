"""Tensor gathers over the frozen, compact two-tower columns."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .two_tower_measure import region


PAIR_STRIDE = 1 << 32


def disk_shuffle(path, count, rng):
    """The same int64 Fisher-Yates permutation as Generator.permutation(count)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    values = np.lib.format.open_memmap(path, mode='w+', dtype=np.int64,
                                       shape=(count,))
    for start in range(0, count, 1_000_000):
        end = min(start + 1_000_000, count)
        values[start:end] = np.arange(start, end, dtype=np.int64)
    rng.shuffle(np.asarray(values))
    values.flush()
    return values


def _tensor(values, device, *, pinned=False, dtype=None):
    result = torch.as_tensor(values, dtype=dtype)
    if str(device) != 'cpu':
        return result.to(device)
    return result.pin_memory() if pinned and torch.cuda.is_available() else result


@dataclass
class PositiveBatch:
    index: 'PositiveIndex'
    users: torch.Tensor
    cached_item_ids: torch.Tensor | None = None

    def allowed(self, item_ids):
        source = item_ids if self.cached_item_ids is None else self.cached_item_ids
        ids = source.to(self.index.keys.device, dtype=torch.int64)
        keys = self.users.to(ids.device, dtype=torch.int64)[:, None] * PAIR_STRIDE
        queries = keys + ids[None, :]
        if self.index.keys.numel():
            positions = torch.searchsorted(self.index.keys, queries)
            bounded = positions.clamp_max(len(self.index.keys) - 1)
            matched = (positions < len(self.index.keys)) & (
                self.index.keys[bounded] == queries
            )
            allowed = ~matched
        else:
            allowed = torch.ones_like(queries, dtype=torch.bool)
        allowed &= ids[:, None] != ids[None, :]
        allowed.fill_diagonal_(True)
        if self.index.pinned and allowed.device.type == 'cpu' and item_ids.device.type == 'cuda':
            allowed = allowed.pin_memory()
        return allowed.to(item_ids.device, non_blocking=self.index.pinned)


class PositiveIndex:
    def __init__(self, keys, device='cpu', pinned=False):
        self.pinned = pinned
        self.keys = _tensor(keys, device, pinned=pinned, dtype=torch.int64)

    @classmethod
    def from_sets(cls, positive_sets, device='cpu'):
        keys = np.fromiter(
            (user * PAIR_STRIDE + int(item)
             for user, items in enumerate(positive_sets) for item in sorted(items)),
            dtype=np.int64,
        )
        return cls(keys, device)

    def batch(self, users, item_ids=None):
        return PositiveBatch(self, torch.as_tensor(users, device=self.keys.device), item_ids)


class TensorCatalog:
    """Keep lookup tables on CPU/pinned CPU or on the compute device."""

    @classmethod
    def from_catalog(cls, catalog, storage_device='cpu', pinned=False):
        self = cls()
        self.ids = catalog.ids
        self.columns = catalog.columns
        self.config = catalog.config
        self.id_count = catalog.id_count
        self.author_count = catalog.author_count
        self.boundary = catalog.boundary
        self.groups = catalog.groups
        self.id_index = catalog.id_index
        self.author_index = catalog.author_index
        self.rank_index = catalog.rank_index
        self.duration_index = catalog.duration_index
        self.storage_device = str(storage_device)
        self.pinned = pinned
        self.timer = None
        self.tables = {
            name: _tensor(value, storage_device, pinned=pinned)
            for name, value in {
                'content': catalog.columns['content'],
                'id_index': np.asarray(catalog.id_index, dtype=np.int32),
                'author_index': np.asarray(catalog.author_index, dtype=np.int32),
                'duration_bucket': np.asarray(catalog.duration_index, dtype=np.uint8),
                'rank_bucket': np.asarray(catalog.rank_index, dtype=np.uint8),
            }.items()
        }
        self.item_ids = _tensor(np.asarray(catalog.ids, dtype=np.int32),
                               storage_device, pinned=pinned)
        return self

    def rows(self, ids):
        if not isinstance(ids, torch.Tensor):
            rows = np.searchsorted(self.ids, ids)
            if np.any(rows >= len(self.ids)) or np.any(self.ids[rows] != ids):
                raise ValueError('unknown catalog item')
            return rows
        ids = ids.to(self.item_ids.device, dtype=torch.int32)
        rows = torch.searchsorted(self.item_ids, ids)
        if bool((rows >= len(self.ids)).any()) or bool(
            (self.item_ids[rows] != ids).any()
        ):
            raise ValueError('unknown catalog item')
        return rows

    def features_by_rows(self, rows, device='cpu'):
        rows = torch.as_tensor(rows, device=self.item_ids.device).long()
        result = {}
        with region(self.timer, 'feature_lookup'):
            for name, values in self.tables.items():
                selected = torch.index_select(values, 0, rows.reshape(-1))
                selected = selected.reshape(*rows.shape, *values.shape[1:])
                if self.pinned and selected.device.type == 'cpu' and torch.device(device).type == 'cuda':
                    selected = selected.pin_memory()
                result[name] = selected
        with region(self.timer, 'h2d'):
            result = {name: value.to(device=device,
                                      dtype=torch.float32 if name == 'content' else torch.int64,
                                      non_blocking=self.pinned)
                      for name, value in result.items()}
        return result

    def features(self, ids, device='cpu'):
        return self.features_by_rows(self.rows(ids), device)

    def tensor_features(self, ids, valid, device):
        ids = ids.to(self.item_ids.device)
        valid = valid.to(ids.device)
        safe = torch.where(valid, ids, self.item_ids[0])
        return self.features(safe, device)


class TensorEventStore:
    def __init__(self, arrays, catalog, storage_device='cpu', pinned=False):
        self.arrays = arrays
        self.catalog = catalog
        self.samples = arrays['target_event_position']
        self.user_ids = arrays['user_id']
        self.offsets = arrays['user_offsets']
        self.storage_device = str(storage_device)
        self.pinned = pinned
        self.timer = None
        names = ['item', 'week', 'place', 'main', 'secondary', 'feedback', 'ratio',
                 'catalog_row', 'user_row']
        self.columns = {name: arrays['event_' + name] for name in names}
        self.tables = {name: _tensor(value, storage_device, pinned=pinned)
                       for name, value in self.columns.items()}
        self.target_positions = _tensor(self.samples, storage_device, pinned=pinned)
        self.user_offsets = _tensor(self.offsets, storage_device, pinned=pinned)
        self.weights = _tensor(arrays['target_weight'], storage_device, pinned=pinned)
        self.q = _tensor(arrays['catalog_target_probability'], storage_device,
                         pinned=pinned)
        self.positives = PositiveIndex(arrays['positive_keys'], storage_device,
                                       pinned=pinned)

    @classmethod
    def from_store(cls, store, catalog, storage_device='cpu', pinned=False):
        arrays = {}
        dtypes = dict(item=np.int32, week=np.uint8, place=np.uint8, main=bool,
                      secondary=bool, feedback=np.int16, ratio=np.float32)
        for name, dtype in dtypes.items():
            arrays['event_' + name] = np.asarray(store.columns[name], dtype=dtype)
        arrays['event_catalog_row'] = catalog.rows(store.columns['item']).astype(np.int32)
        arrays['event_user_row'] = np.asarray(store.user_rows, dtype=np.int32)
        arrays['user_id'] = np.asarray(store.user_ids, dtype=np.int32)
        arrays['user_offsets'] = np.asarray(store.offsets, dtype=np.int32)
        arrays['target_event_position'] = np.asarray(store.samples, dtype=np.int32)
        arrays['target_weight'] = np.asarray(store.weights, dtype=np.float16)
        probability = np.zeros(len(catalog.ids), dtype=np.float32)
        probability[catalog.rows(store.frequency_ids)] = (
            store.frequency_counts / store.frequency_counts.sum()
        ).astype(np.float32)
        arrays['catalog_target_probability'] = probability
        sets = [store.positives[int(user)] for user in store.user_ids]
        arrays['positive_keys'] = PositiveIndex.from_sets(sets).keys.numpy()
        return cls(arrays, catalog, storage_device, pinned)

    def history_batch(self, positions, users, weeks, max_history, device, *, context_only=False):
        with region(self.timer, 'history'):
            positions = torch.as_tensor(positions, device=self.storage_device).long()
            users = torch.as_tensor(users, device=self.storage_device).long()
            weeks = torch.as_tensor(weeks, device=self.storage_device).long()
            counts = (positions - self.user_offsets[users]).clamp(max=max_history)
            width = 1 if context_only and not counts.numel() else max(1, int(counts.max()))
            column = torch.arange(width, device=positions.device)[None, :]
            rows = positions[:, None] - counts[:, None] + column
            valid = column < counts[:, None]
            rows = rows.clamp(max=len(self.columns['item']) - 1)
            get = lambda name: self.tables[name][rows]
            feedback = ((get('feedback').long()[..., None]
                         >> torch.arange(9, device=positions.device)) & 1).float()
            ratio = get('ratio')
            place = get('place').long() + 1
            gap = (weeks[:, None] - get('week').long()).clamp(0, 26) + 1
            feature_rows = get('catalog_row').masked_fill(~valid, 0)
            if context_only:
                batch = dict(valid=valid, feedback=feedback.masked_fill(~valid[..., None], 0),
                             ratio=ratio.masked_fill(~valid, 0),
                             place=place.masked_fill(~valid, 0), gap=gap.masked_fill(~valid, 0),
                             position=column.expand_as(valid).masked_fill(~valid, 0),
                             catalog_rows=feature_rows)
            else:
                items = get('item').masked_fill(~valid, 0)
                bucket = 1 + (ratio >= .1).long() + (ratio >= .5).long()
                bucket = bucket + (ratio >= 1).long() + (ratio > 1).long()
                batch = dict(items=items, valid=valid, feedback=feedback, ratio=bucket,
                             place=place, gap=gap)
        with region(self.timer, 'h2d'):
            if self.pinned and torch.device(device).type == 'cuda':
                batch = {name: value.pin_memory() for name, value in batch.items()}
            batch = {name: value.to(device, non_blocking=self.pinned)
                     for name, value in batch.items()}
        if not context_only:
            batch['item_features'] = self.catalog.features_by_rows(feature_rows, device)
        return batch

    def train_batch(self, selected, config, device):
        if config.packed_window:
            raise ValueError('tensor path preserves the full-target random shuffle')
        with region(self.timer, 'take_batch'):
            selected = torch.as_tensor(selected, device=self.storage_device).long()
            positions = self.target_positions[selected].long()
            users = self.tables['user_row'][positions].long()
        batch = self.history_batch(positions, users, self.tables['week'][positions],
                                   config.max_history, device)
        rows = self.tables['catalog_row'][positions].long()
        batch['target_features'] = self.catalog.features_by_rows(rows, device)
        ids = self.tables['item'][positions]
        positives = self.positives.batch(users, ids)
        with region(self.timer, 'h2d'):
            return (batch, ids.to(device), positives, self.q[rows].to(device),
                    self.weights[selected].to(device=device, dtype=torch.float32))

    def query_batch(self, user_ids, config, device):
        users = np.searchsorted(self.user_ids, user_ids)
        bounded = np.minimum(users, len(self.user_ids) - 1)
        found = (users < len(self.user_ids)) & (self.user_ids[bounded] == user_ids)
        ends = np.where(found, self.offsets[bounded + 1], self.offsets[bounded])
        return self.history_batch(ends, bounded, np.full(len(users), 25),
                                   config.max_history, device)
