"""Read-only adapter for the delivered round13 foundation, never raw weeks."""

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from .two_tower import Catalog, user_hash
from .two_tower_batching import remaining_user_batches


class EventStore:
    def __init__(self, columns, user_ids, offsets):
        self.columns = columns
        self.user_ids = np.asarray(user_ids)
        self.offsets = np.asarray(offsets)
        self.user_rows = np.repeat(np.arange(len(user_ids)), np.diff(offsets))
        self.samples = np.flatnonzero(
            (columns["place"] <= 1) & (columns["main"] | columns["secondary"])
        )
        self.weights = np.where(columns["main"][self.samples], 1.0, 0.25).astype(
            np.float32
        )
        self.positives = {int(user): set() for user in user_ids}
        for row in np.flatnonzero(columns["main"] | columns["secondary"]):
            self.positives[int(user_ids[self.user_rows[row]])].add(
                int(columns["item"][row])
            )
        self.frequency_ids, self.frequency_counts = np.unique(
            columns["item"][self.samples], return_counts=True
        )

    def history_batch(self, positions, users, reference_weeks, max_history, device):
        positions = np.asarray(positions)
        users = np.asarray(users)
        starts = self.offsets[users]
        counts = np.minimum(positions - starts, max_history)
        width = max(1, int(counts.max(initial=0)))
        column = np.arange(width)[None, :]
        rows = positions[:, None] - counts[:, None] + column
        valid = column < counts[:, None]
        rows = np.minimum(rows, max(0, len(self.columns["item"]) - 1))
        get = lambda name: self.columns[name][rows]
        items = get("item").copy()
        items[~valid] = 0
        feedback = ((get("feedback")[..., None] >> np.arange(9)) & 1).astype(np.float32)
        ratio = get("ratio")
        bucket = (
            1
            + (ratio >= 0.1).astype(np.int64)
            + (ratio >= 0.5)
            + (ratio >= 1.0)
            + (ratio > 1.0)
        )
        place = get("place").astype(np.int64) + 1
        gap = (
            np.clip(
                np.asarray(reference_weeks)[:, None] - get("week").astype(np.int64),
                0,
                26,
            )
            + 1
        )
        values = dict(
            items=items,
            valid=valid,
            feedback=feedback,
            ratio=bucket,
            place=place,
            gap=gap,
        )
        return {
            name: torch.as_tensor(value, device=device)
            for name, value in values.items()
        }

    def train_batch(self, sample_indices, config, device):
        positions = self.samples[sample_indices]
        users = self.user_rows[positions]
        pack = self.packed_history if config.packed_window else self.history_batch
        batch = pack(
            positions,
            users,
            self.columns["week"][positions],
            config if config.packed_window else config.max_history,
            device,
        )
        targets = self.columns["item"][positions]
        probability = (
            self.frequency_counts[np.searchsorted(self.frequency_ids, targets)]
            / self.frequency_counts.sum()
        )
        return (
            batch,
            targets,
            [self.positives[int(self.user_ids[x])] for x in users],
            probability,
            self.weights[sample_indices],
        )

    def packed_batches(self, config, rng, max_samples=0):
        """Shuffle user/week chunks, visiting every natural target exactly once."""
        if config.shuffle_all_targets:
            order = rng.permutation(len(self.samples))
            if max_samples:
                order = order[:max_samples]
            for start in range(0, len(order), config.batch_size):
                yield order[start:start + config.batch_size]
            return
        selected = np.arange(len(self.samples))
        if max_samples:
            selected = np.sort(rng.permutation(selected)[:max_samples])
        positions = self.samples[selected]
        users = self.user_rows[positions]
        weeks = self.columns["week"][positions]
        boundaries = np.r_[0, np.flatnonzero((users[1:] != users[:-1]) | (weeks[1:] != weeks[:-1])) + 1, len(selected)]
        chunks = [selected[start:min(start + config.max_user_targets, end)]
                  for low, end in zip(boundaries[:-1], boundaries[1:])
                  for start in range(low, end, config.max_user_targets)]
        order = rng.permutation(len(chunks))
        if config.balance_user_batches:
            chunk_users = np.array([self.user_rows[self.samples[chunk[0]]] for chunk in chunks])
            batches = remaining_user_batches(chunks, chunk_users, order,
                                              config.batch_size, config.max_user_targets)
            if config.shuffle_balanced_batches:
                batches = list(batches)
                for index in rng.permutation(len(batches)):
                    yield batches[index]
            else:
                yield from batches
            return
        pending = deque(int(index) for index in order)
        while pending:
            batch = []
            count = 0
            used_users = set()
            deferred = []
            while pending and count < config.batch_size:
                index = pending.popleft()
                chunk = chunks[index]
                user = int(self.user_rows[self.samples[chunk[0]]])
                if user in used_users:
                    deferred.append(index)
                    continue
                take = min(len(chunk), config.batch_size - count)
                batch.append(chunk[:take])
                count += take
                used_users.add(user)
                if take < len(chunk):
                    chunks[index] = chunk[take:]
                    deferred.append(index)
            pending.extend(deferred)
            yield np.concatenate(batch)
            if config.max_user_targets == 1 and count < config.batch_size and pending:
                # Every remaining user now fits. Preserve the shuffled queue
                # order without rescanning all duplicate chunks at each step.
                tails = defaultdict(deque)
                for order, index in enumerate(pending):
                    user = int(self.user_rows[self.samples[chunks[index][0]]])
                    tails[user].append((order, index))
                while tails:
                    fronts = sorted((queue[0][0], user) for user, queue in tails.items())
                    selected = []
                    for _, user in fronts:
                        _, index = tails[user].popleft()
                        selected.append(chunks[index])
                        if not tails[user]:
                            del tails[user]
                    yield np.concatenate(selected)
                return

    def packed_history(self, positions, users, reference_weeks, config, device):
        positions, users, reference_weeks = map(np.asarray, (positions, users, reference_weeks))
        keys, groups = np.unique(np.column_stack([users, reference_weeks]), axis=0, return_inverse=True)
        sources, gaps, validity = [], [], []
        offsets = [0]
        readouts = np.zeros(len(positions), dtype=np.int64)
        blocks = []
        context = config.max_history * 2 - 1 if config.causal_readout == "window_mean" else config.max_history
        for group, (user, week) in enumerate(keys):
            indices = np.flatnonzero(groups == group)
            start = max(int(self.offsets[user]), int(positions[indices].min()) - context)
            end = int(positions[indices].max())
            rows = np.arange(start, max(start + 1, end))
            sources.append(rows)
            validity.append(rows < end)
            gaps.append(np.clip(week - self.columns["week"][rows].astype(np.int64), 0, 26) + 1)
            readouts[indices] = offsets[-1] + positions[indices] - start
            nonempty = indices[positions[indices] > self.offsets[user]]
            chunk_ids = (positions[nonempty] - start - 1) // config.sequence_chunk
            for chunk in np.unique(chunk_ids):
                targets = nonempty[chunk_ids == chunk]
                key_start = offsets[-1] + max(0, int(chunk) * config.sequence_chunk - config.max_history + 1)
                key_end = offsets[-1] + min(end - start, (int(chunk) + 1) * config.sequence_chunk)
                blocks.append((targets, key_start, key_end))
            offsets.append(offsets[-1] + len(rows))
        source = np.concatenate(sources)
        ratio = self.columns["ratio"][source]
        values = dict(items=self.columns["item"][source], valid=np.concatenate(validity),
                      feedback=((self.columns["feedback"][source, None] >> np.arange(9)) & 1).astype(np.float32),
                      ratio=1 + (ratio >= .1).astype(np.int64) + (ratio >= .5) + (ratio >= 1) + (ratio > 1),
                      place=self.columns["place"][source].astype(np.int64) + 1,
                      gap=np.concatenate(gaps), readouts=readouts,
                      target_group=groups.astype(np.int64))
        batch = {name: torch.as_tensor(value, device=device) for name, value in values.items()}
        batch.update(group_offsets=np.asarray(offsets, dtype=np.int64), source_rows=source,
                     attention_blocks=blocks)
        return batch

    def query_batch(self, user_ids, config, device):
        # Preserve query users with no training history as all-padding sequences.
        users = np.searchsorted(self.user_ids, user_ids)
        found = (users < len(self.user_ids)) & (
            self.user_ids[np.minimum(users, len(self.user_ids) - 1)] == user_ids
        )
        safe = np.minimum(users, len(self.user_ids) - 1)
        ends = np.where(found, self.offsets[safe + 1], self.offsets[safe])
        pack = self.packed_history if config.packed_window else self.history_batch
        return pack(
            ends, safe, np.full(len(users), 25), config if config.packed_window else config.max_history, device
        )


def encode_batch(model, catalog, batch, device, *, cold=False):
    features = batch.get('item_features')
    if features is None:
        features = catalog.tensor_features(batch['items'], batch['valid'], device)
    encoded = model.item(features, cold=cold)
    return model.user(encoded, batch, batch["valid"])


def subset_evaluation(data, users):
    from .evaluation import EvaluationData

    indices = np.searchsorted(data.user_ids, users)
    lengths = np.diff(data.targets["offsets"])[indices].astype(np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    rows = np.concatenate(
        [
            np.arange(data.targets["offsets"][i], data.targets["offsets"][i + 1])
            for i in indices
        ]
    )
    targets = {
        name: value[rows] for name, value in data.targets.items() if name != "offsets"
    }
    targets["offsets"] = offsets
    return EvaluationData(
        data.user_ids[indices],
        data.history_user_ids,
        data.history_offsets,
        data.history_items,
        targets,
        data.items,
    )


def select_trial_users(query_ids, train_ids, count, seed, *, all_training=False):
    if all_training:
        return np.union1d(query_ids, train_ids)
    ordered = sorted(query_ids, key=lambda user: user_hash(user, seed))
    return np.sort(np.asarray(ordered[:count], dtype=np.int64))


def load_foundation(
    foundation,
    raw_metadata,
    vector_ids,
    vectors,
    config,
    output,
    users=2000,
    all_training_users=False,
):
    import duckdb
    from .evaluation import EvaluationData

    foundation = Path(foundation)
    output = Path(output)
    connection = duckdb.connect()
    connection.execute("SET memory_limit='8GB'")
    from .two_tower_runtime import available_resources
    connection.execute(f"SET threads={min(torch.get_num_threads(), available_resources()['cpu_threads'])}")
    connection.execute("SET temp_directory=?", [str(output / "duckdb-tmp")])
    all_eval = EvaluationData.load(foundation)
    selected = select_trial_users(
        all_eval.user_ids,
        all_eval.history_user_ids,
        users,
        config.user_hash_seed,
        all_training=all_training_users,
    )
    data = all_eval if all_training_users else subset_evaluation(all_eval, selected)
    connection.register("selected", {"user_id": selected})
    print("读取训练序列，固定hash用户", len(selected), flush=True)
    # Only train sequences are queried. No raw exposure path or week26 glob.
    paths = [
        str(foundation / f"sequences/user_block={block}/users.parquet")
        for block in range(32)
    ]
    names = [
        "item_ids",
        "weeks",
        "places",
        "main_positive",
        "secondary_positive",
        "feedback_masks",
        "watch_ratios",
    ]
    short = ["item", "week", "place", "main", "secondary", "feedback", "ratio"]
    dtypes = [np.uint32, np.uint8, np.uint8, bool, bool, np.uint16, np.float32]
    pieces = {name: [] for name in short}
    ids = []
    offsets = [0]
    # DuckDB materializes only selected per-user lists, not the whole event table.
    query = (
        "SELECT s.user_id,"
        + ",".join("s." + name for name in names)
        + " FROM read_parquet(?) s JOIN selected USING(user_id) ORDER BY s.user_id"
    )
    cursor = connection.execute(query, [paths])
    while batch := cursor.fetchmany(16):
        for row in batch:
            ids.append(row[0])
            offsets.append(offsets[-1] + len(row[1]))
            for name, value, dtype in zip(short, row[1:], dtypes):
                pieces[name].append(np.asarray(value, dtype=dtype))
    columns = {name: np.concatenate(values) for name, values in pieces.items()}
    store = EventStore(columns, np.array(ids), np.array(offsets))
    print("读取全目录metadata和训练计数", flush=True)
    cols = connection.execute(
        "SELECT i.item_id,i.author_id,i.duration,i.train_count,m.train_interactions_rank AS rank FROM read_parquet(?) i JOIN read_parquet(?) m USING(item_id) ORDER BY i.item_id",
        [str(foundation / "items.parquet"), str(raw_metadata)],
    ).fetchnumpy()
    ids_vector = np.load(vector_ids, mmap_mode="r")
    content = np.load(vectors, mmap_mode="r")
    if not np.array_equal(ids_vector, cols["item_id"]):
        raise ValueError(
            "vector rows must match sorted full catalog IDs; explicit reorder needed"
        )
    cols["content"] = content
    catalog = Catalog(cols, config)
    print("缓存所选训练用户涉及的原始内容行，保持dtype和数值", flush=True)
    catalog.cache_training_content(
        np.r_[store.columns["item"], catalog.ids[:1]], output / "training-content-cache"
    )
    thresholds = []
    for threshold in (1, 2, 5, 10, 20, 50, 100):
        supported = cols["train_count"] >= threshold
        targets_supported = supported[catalog.rows(store.frequency_ids)]
        thresholds.append(
            dict(
                min_count=threshold,
                ids=int(supported.sum()),
                adamw_fp32_bytes=int(supported.sum()) * config.dim * 16,
                exposure_coverage=float(
                    cols["train_count"][supported].sum() / cols["train_count"].sum()
                ),
                sampled_target_coverage=float(
                    store.frequency_counts[targets_supported].sum()
                    / store.frequency_counts.sum()
                ),
            )
        )
    tail = cols["rank"] >= int(np.ceil(0.99 * len(cols["rank"])))
    summary = dict(
        all_training_users=all_training_users,
        users=len(data.user_ids),
        training_users=len(ids),
        events=len(columns["item"]),
        training_targets=len(store.samples),
        catalog_items=len(catalog.ids),
        rank_boundary=catalog.boundary,
        seen_in_last_one_percent=int(((cols["train_count"] > 0) & tail).sum()),
        group_items=np.bincount(catalog.groups, minlength=3).tolist(),
        id_support=thresholds,
    )
    (output / "data_summary.json").write_text(json.dumps(summary, indent=2))
    connection.close()
    return catalog, store, data, summary


def synthetic(config, user_count=64, item_count=1200):
    from .evaluation import EvaluationData

    rng = np.random.default_rng(config.seed)
    content = rng.normal(0, 0.1, (item_count, 64)).astype(np.float32)
    for item in range(item_count):
        content[item, item % 8] += 4
    cols = dict(
        item_id=np.arange(item_count, dtype=np.uint32),
        author_id=np.arange(item_count) % 8,
        duration=np.full(item_count, 30),
        train_count=np.where(np.arange(item_count) < 800, 8, 0),
        rank=np.arange(item_count),
        content=content,
    )
    catalog = Catalog(cols, config)
    records = {
        k: []
        for k in ("item", "week", "place", "main", "secondary", "feedback", "ratio")
    }
    offsets = [0]
    history = []
    ho = [0]
    targets = []
    to = [0]
    for user in range(user_count):
        group = user % 8
        items = rng.choice(np.arange(group, 800, 8), size=40, replace=False)
        for ordinal, item in enumerate(items):
            values = (
                item,
                ordinal // 2 % 25,
                0,
                ordinal % 4 != 0,
                ordinal % 4 == 0,
                3,
                1.2,
            )
            for name, value in zip(records, values):
                records[name].append(value)
        offsets.append(len(records["item"]))
        history.extend(sorted(items))
        ho.append(len(history))
        goal = np.arange(800 + group, item_count, 8)[:5]
        targets.extend(goal)
        to.append(len(targets))
    store = EventStore(
        {name: np.asarray(value) for name, value in records.items()},
        np.arange(user_count),
        np.array(offsets),
    )
    target = np.array(targets, dtype=np.uint32)
    n = len(target)
    target_columns = dict(
        offsets=np.array(to),
        item_id=target,
        place=np.zeros(n, dtype=np.uint8),
        main_positive=np.ones(n, dtype=bool),
        secondary_positive=np.ones(n, dtype=bool),
        like_positive=np.ones(n, dtype=bool),
        train_seen=np.zeros(n, dtype=bool),
        duration=np.full(n, 30),
        duration_band=np.zeros(n, dtype=np.uint8),
        duration_band_sensitive=np.ones(n, dtype=np.uint8),
        popularity_tier=np.zeros(n, dtype=np.uint8),
    )
    eval_items = {k: v for k, v in cols.items() if k not in ("rank", "content")}
    eval_items["popularity_tier"] = np.where(cols["train_count"] > 0, 1, 0)
    data = EvaluationData(
        np.arange(user_count),
        np.arange(user_count),
        np.array(ho),
        np.array(history, dtype=np.uint32),
        target_columns,
        eval_items,
    )
    return (
        catalog,
        store,
        data,
        dict(
            users=user_count,
            events=len(records["item"]),
            training_targets=len(store.samples),
            catalog_items=item_count,
            rank_boundary=catalog.boundary,
            synthetic=True,
        ),
    )
