"""Bounded-batch all-catalog export, IVF, exact audit and fixed tie handling."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from .two_tower_data import encode_batch


GROUP_NAMES = ("本地见过", "仅本地未见", "全站新视频（代理）")


def export_item_embeddings(model, catalog, device, output, batch_size=8192):
    """Persist FP32 full/available-ID vectors for the existing FAISS evaluator."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    started = time.perf_counter()
    values = np.lib.format.open_memmap(output / 'vectors.npy', mode='w+',
                                       dtype=np.float32,
                                       shape=(len(catalog.ids), model.config.dim))
    with torch.no_grad():
        for start in range(0, len(catalog.ids), batch_size):
            ids = catalog.ids[start:start+batch_size]
            vectors = model.item(catalog.features(ids, device)).float().cpu().numpy()
            if not np.isfinite(vectors).all():
                raise ValueError('nonfinite exported item vectors')
            values[start:start+len(ids)] = vectors
    values.flush()
    del values
    np.save(output / 'item_ids.npy', catalog.ids, allow_pickle=False)
    record = dict(items=len(catalog.ids), dim=model.config.dim, dtype='float32',
                   seconds=time.perf_counter()-started)
    with (output / 'manifest.json').open('x') as handle:
        json.dump(record, handle, indent=2)
    return record


def export_cold_embeddings(model, catalog, device, output, structure, batch_size=8192,
                           *, stop_requested=None):
    """Persist L2 float16 item vectors, preserving the full catalog row order."""
    import hashlib

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    started = time.perf_counter()
    vectors = np.lib.format.open_memmap(output / "vectors.npy", mode="w+", dtype=np.float16,
                                       shape=(len(catalog.ids), model.config.dim))
    max_norm_error = 0.0
    with torch.no_grad():
        for start in range(0, len(catalog.ids), batch_size):
            if stop_requested is not None and stop_requested():
                vectors.flush()
                return None
            ids = catalog.ids[start:start + batch_size]
            values = model.item(catalog.features(ids, device), cold=True).cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("nonfinite cold item vectors")
            values = values.astype(np.float16)
            max_norm_error = max(max_norm_error, float(np.abs(np.linalg.norm(values.astype(np.float32), axis=1) - 1).max()))
            vectors[start:start + len(ids)] = values
    vectors.flush()
    del vectors
    np.save(output / "item_ids.npy", catalog.ids, allow_pickle=False)
    digest = hashlib.sha256()
    with (output / "vectors.npy").open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            if stop_requested is not None and stop_requested():
                return None
            digest.update(block)
    record = dict(structure=structure, items=len(catalog.ids), dim=model.config.dim, dtype="float16",
                  candidate_path="ID zero, rank unknown, retain raw content projection, author and duration",
                  normalized_before_float16=True, max_float16_norm_error=max_norm_error,
                  vectors_sha256=digest.hexdigest(), seconds=time.perf_counter() - started)
    (output / "manifest.json").write_text(json.dumps(record, indent=2))
    print("冷启动物品向量导出完成", json.dumps(record), flush=True)
    return record


def tie_keys(catalog):
    import duckdb

    connection = duckdb.connect()
    connection.execute("SET memory_limit='8GB'")
    from .two_tower_runtime import available_resources
    connection.execute(f"SET threads={min(torch.get_num_threads(), available_resources()['cpu_threads'])}")
    connection.register("items", {"item_id": catalog.ids})
    keys = connection.execute(
        "SELECT hash(item_id,20261108::UBIGINT) AS tie FROM items"
    ).fetchnumpy()["tie"]
    connection.close()
    return keys


def top_order(rows, scores, catalog, ties, k):
    # A score cutoff retains all ties at the cutoff before hash tie-breaking.
    if len(rows) > k:
        cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
        keep = scores >= cutoff
        rows = rows[keep]
        scores = scores[keep]
    order = np.lexsort((catalog.ids[rows], ties[rows], -scores))[:k]
    return rows[order], scores[order]


def retrieve(
    model,
    catalog,
    store,
    data,
    device,
    output=None,
    *,
    cold=False,
    exact_users=8,
    export_batch=8192,
    nlist=1024,
    nprobe=64,
    save_index=False,
    vector_file=None,
):
    import faiss

    retrieval_threads = torch.get_num_threads()
    from .two_tower_runtime import available_resources
    faiss.omp_set_num_threads(min(retrieval_threads, available_resources()['cpu_threads']))
    try:
        from .evaluation import filter_ranked_items

        started = time.perf_counter()
        model.eval()
        ties = tie_keys(catalog)
        cached_vectors = None
        if vector_file is not None:
            vector_file = Path(vector_file)
            cached_vectors = np.load(vector_file, mmap_mode='r', allow_pickle=False)
            cached_ids = np.load(vector_file.parent / 'item_ids.npy', mmap_mode='r', allow_pickle=False)
            if (cached_vectors.shape != (len(catalog.ids), model.config.dim)
                    or cached_vectors.dtype != np.float32
                    or not np.array_equal(cached_ids, catalog.ids)):
                raise ValueError('exported vectors and evaluation catalog differ')
        users = []
        query_started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(data.user_ids), 64):
                ids = data.user_ids[start : start + 64]
                users.append(
                    encode_batch(
                        model,
                        catalog,
                        store.query_batch(ids, model.config, device),
                        device,
                    )
                    .cpu()
                    .numpy()
                )
        user_vectors = np.concatenate(users).astype(np.float32)
        query_encode_seconds = time.perf_counter() - query_started
        nlist = max(1, min(nlist, len(catalog.ids) // 40))
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(model.config.dim),
            model.config.dim,
            nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        rng = np.random.default_rng(model.config.seed)
        sample = np.sort(
            rng.choice(
                len(catalog.ids),
                size=min(len(catalog.ids), max(nlist * 40, 20000)),
                replace=False,
            )
        )

        def item_vectors(rows):
            if cached_vectors is not None:
                return np.asarray(cached_vectors[rows], dtype=np.float32)
            with torch.no_grad():
                return (
                    model.item(catalog.features(catalog.ids[rows], device), cold=cold)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

        train = np.concatenate(
            [
                item_vectors(sample[i : i + export_batch])
                for i in range(0, len(sample), export_batch)
            ]
        )
        fit_started = time.perf_counter()
        index.train(train)
        index_fit_seconds = time.perf_counter() - fit_started
        del train
        add_started = time.perf_counter()
        for start in range(0, len(catalog.ids), export_batch):
            index.add(
                item_vectors(
                    np.arange(start, min(start + export_batch, len(catalog.ids)))
                )
            )
        index.nprobe = min(nprobe, nlist)
        index_add_seconds = time.perf_counter() - add_started
        build_seconds = time.perf_counter() - started
        print(
            "全目录向量与IVF构建完成",
            len(catalog.ids),
            "秒",
            round(build_seconds, 2),
            "cold",
            cold,
            flush=True,
        )
        search_started = time.perf_counter()
        predictions = {}
        empty = 0
        for start in range(0, len(data.user_ids), 64):
            vectors = user_vectors[start : start + 64]
            supply = min(len(catalog.ids), 4096)
            pending = list(range(len(vectors)))
            while pending:
                scores, positions = index.search(vectors[pending], supply)
                retry = []
                for row, local in enumerate(pending):
                    user = int(data.user_ids[start + local])
                    if not np.any(vectors[local]):
                        predictions[user] = []
                        empty += 1
                        continue
                    valid = positions[row] >= 0
                    ordered, _ = top_order(
                        positions[row][valid],
                        scores[row][valid],
                        catalog,
                        ties,
                        int(valid.sum()),
                    )
                    result = filter_ranked_items(
                        catalog.ids[ordered], data.seen_items(user), catalog.ids, 1000
                    )
                    predictions[user] = result
                    if (
                        len(result) < 1000
                        and supply < len(catalog.ids)
                        and int(valid.sum()) == supply
                    ):
                        retry.append(local)
                if not retry:
                    break
                pending = retry
                supply = min(len(catalog.ids), supply * 2)
        # Exact filtered item retrieval for a deterministic hash-independent sample.
        search_seconds = time.perf_counter() - search_started
        audit_count = min(exact_users, len(data.user_ids))
        exact = [
            (np.array([], dtype=np.int64), np.array([], dtype=np.float32))
            for _ in range(audit_count)
        ]
        exact_started = time.perf_counter()
        for start in range(0, len(catalog.ids) if audit_count else 0, export_batch):
            rows = np.arange(start, min(start + export_batch, len(catalog.ids)))
            scores = user_vectors[:audit_count] @ item_vectors(rows).T
            for user in range(audit_count):
                allowed = ~np.isin(
                    catalog.ids[rows], data.seen_items(int(data.user_ids[user]))
                )
                old_rows, old_scores = exact[user]
                exact[user] = top_order(
                    np.r_[old_rows, rows[allowed]],
                    np.r_[old_scores, scores[user, allowed]],
                    catalog,
                    ties,
                    1000,
                )
        overlaps = []
        for user, (rows, _) in enumerate(exact):
            expected = set(catalog.ids[rows].tolist())
            got = set(predictions[int(data.user_ids[user])])
            overlaps.append(len(expected & got) / max(1, len(expected)))
        details = dict(
            catalog_items=len(catalog.ids),
            ivf_nlist=nlist,
            nprobe=index.nprobe,
            build_seconds=build_seconds,
            query_encode_seconds=query_encode_seconds,
            index_fit_seconds=index_fit_seconds,
            index_add_seconds=index_add_seconds,
            search_seconds=search_seconds,
            exact_audit_seconds=time.perf_counter() - exact_started,
            total_seconds=time.perf_counter() - started,
            exact_users=audit_count,
            exact_top1000_overlap=overlaps,
            mean_overlap=float(np.mean(overlaps)) if overlaps else None,
            empty_queries=empty,
            cold_candidate_path=cold,
            tie_seed=20261108,
        )
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=False)
            (output / "retrieval.json").write_text(json.dumps(details, indent=2))
            ids = np.asarray(data.user_ids)
            matrix = np.full((len(ids), 1000), np.iinfo(np.uint32).max, dtype=np.uint32)
            for row, user in enumerate(ids):
                matrix[row, : len(predictions[int(user)])] = predictions[int(user)]
            np.save(output / "user_ids.npy", ids)
            np.save(output / "predictions.npy", matrix)
            if save_index:
                faiss.write_index(index, str(output / "index.faiss"))
        return predictions, details
    finally:
        torch.set_num_threads(retrieval_threads)


def official_rank_predictions(catalog, data):
    from .evaluation import filter_ranked_items

    order = np.argsort(catalog.columns["rank"], kind="stable")
    ranked = catalog.ids[order]
    # No user's history can remove more than its own number of distinct items.
    result = {}
    for user in data.user_ids:
        seen = data.seen_items(int(user))
        result[int(user)] = filter_ranked_items(
            ranked[: min(len(ranked), 1000 + len(seen))], seen, catalog.ids, 1000
        )
    return result


def diagnostic_evaluations(
    catalog,
    data,
    predictions,
    name,
    output,
    replicates=1000,
    *,
    diagnostic_ks=(200,),
    save_diagnostics=False,
):
    from .evaluation import EvaluationData, evaluate

    output = Path(output)
    base = evaluate(
        data,
        predictions,
        model_name=name,
        ks=(50, 200, 1000),
        replicates=replicates,
        seed=2026111301,
        output_dir=output / "all",
    )
    groups = catalog.groups[catalog.rows(data.targets["item_id"])]
    diagnostics = []
    masks = [(GROUP_NAMES[group], groups == group) for group in range(3)]
    counts = catalog.columns["train_count"]
    seen = counts > 0
    hot = np.quantile(counts[seen], [0.8, 0.95], method="inverted_cdf")
    target_rows = catalog.rows(data.targets["item_id"])
    target_counts = counts[target_rows]
    for name_, mask in [
        ("本地见过-头部", target_counts >= hot[1]),
        ("本地见过-中部", (target_counts >= hot[0]) & (target_counts < hot[1])),
        ("本地见过-长尾", (target_counts > 0) & (target_counts < hot[0])),
    ]:
        masks.append((name_, mask & (groups == 0)))
    ranks = catalog.columns["rank"]
    local_unseen = ranks[catalog.groups == 1]
    cuts = (
        np.quantile(local_unseen, [1 / 3, 2 / 3], method="inverted_cdf")
        if len(local_unseen)
        else [0, 0]
    )
    target_rank = ranks[target_rows]
    for name_, mask in [
        ("仅本地未见-rank高", target_rank <= cuts[0]),
        ("仅本地未见-rank中", (target_rank > cuts[0]) & (target_rank <= cuts[1])),
        ("仅本地未见-rank低", target_rank > cuts[1]),
    ]:
        masks.append((name_, mask & (groups == 1)))
    for group in (1, 2):
        author_seen = catalog.author_index[target_rows] > 0
        for known in (True, False):
            masks.append(
                (
                    GROUP_NAMES[group]
                    + ("-本地作者见过" if known else "-本地作者未见"),
                    (groups == group) & (author_seen == known),
                )
            )
    for label, keep in masks:
        columns = dict(data.targets)
        for key in ("main_positive", "secondary_positive", "like_positive"):
            columns[key] = data.targets[key] & keep
        sliced = EvaluationData(
            data.user_ids,
            data.history_user_ids,
            data.history_offsets,
            data.history_items,
            columns,
            data.items,
        )
        # Each diagnostic uses the exact same evaluator and bootstrap seed.
        result = evaluate(
            sliced,
            predictions,
            model_name=name,
            ks=diagnostic_ks,
            replicates=replicates,
            seed=2026111301,
        )
        if save_diagnostics:
            result.save(output / "groups" / label)
        row = next(
            x
            for x in result.metrics
            if x["ruler"] == "main"
            and x["place_scope"] == "0/1"
            and x["seen_scope"] == "all"
            and x["breakdown"] == "all"
            and x["k"] == 200
        )
        diagnostics.append(dict(group=label, **row))
    (output / "group_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False)
    )
    return base, diagnostics
