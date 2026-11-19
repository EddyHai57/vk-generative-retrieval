"""Expand generator beams and evaluate item IDs with the shared LSVD SDK."""

from dataclasses import dataclass, replace
from pathlib import Path
import time

import numpy as np
import torch

from .evaluation import PADDING_ITEM_ID, evaluate
from .generator_decode import PrefixTrie, beam_search
from .two_tower_run import primary
from .two_tower_runtime import precision_mode


PROXY_GROUPS = {'locally_unseen_only': 1, 'global_new_proxy': 2}
ROUTES = ('full_sid', 'direct', 'hitchhike', 'prefix', 'content')


@dataclass
class DecodedBatch:
    start: int
    ids: np.ndarray
    usable: np.ndarray
    queries: np.ndarray
    levels: dict
    prefix: tuple | None


def _sync(cuda, device):
    if cuda:
        torch.cuda.synchronize(device)


def decode_batches(model, dataset, data, decode, device, precision='fp32', *,
                   prefix_route=False, query_batch_size=8, decoder_batch_size=128,
                   external_query=None, trie=None, timings=None):
    if not model.config.shared_head and external_query is None:
        raise ValueError('external two-tower query is required when shared head is disabled')
    if trie is None:
        trie = PrefixTrie(dataset.sid.prefixes, dataset.sid.size, device)
    timed_cuda = timings is not None and torch.device(device).type == 'cuda'
    if timings is not None:
        timings.update(encoder_seconds=0., beam_seconds=0.,
                       decode_peak_allocated_bytes=None, decode_peak_reserved_bytes=None)
    model.eval()
    with torch.no_grad(), precision_mode(device, precision):
        for start in range(0, len(data.user_ids), query_batch_size):
            t0 = time.perf_counter()
            ids = data.user_ids[start:start + query_batch_size]
            history = dataset.query_batch(ids)
            usable = history['valid'].any(dim=1).cpu().numpy()
            memory, padding = model.encode(history)
            queries = (model.user_vector(memory, padding) if model.config.shared_head
                       else external_query(ids)).float().cpu().numpy()
            _sync(timed_cuda, device)
            if timings is not None:
                timings['encoder_seconds'] += time.perf_counter() - t0
            if timed_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            score = model.make_beam_scorer(memory, padding, hypothesis_batch_size=decoder_batch_size)
            beams = beam_search(trie, score, len(ids), decode.beam_widths)
            prefix = None
            if prefix_route:
                count = min(decode.prefix_beam_width, trie.layers[1][2].numel())
                level = beams[2]
                if level.keys.shape[1] < count:
                    level = beam_search(trie, score, len(ids),
                                        (decode.beam_widths[0], decode.prefix_beam_width), depth=2)[2]
                prefix = (level.keys[:, :count].cpu().numpy(), level.logp[:, :count].cpu().numpy())
            levels = {depth: (level.keys.cpu().numpy(), level.logp.cpu().numpy())
                      for depth, level in beams.items()}
            _sync(timed_cuda, device)
            if timings is not None:
                timings['beam_seconds'] += time.perf_counter() - t0
                if timed_cuda:
                    for key, value in (
                            ('decode_peak_allocated_bytes', torch.cuda.max_memory_allocated(device)),
                            ('decode_peak_reserved_bytes', torch.cuda.max_memory_reserved(device))):
                        timings[key] = max(value, timings[key] or 0)
            del score, memory, padding, history, beams
            yield DecodedBatch(start, ids, usable, queries, levels, prefix)


def rank_inputs(batch, data):
    empty = np.empty(0)
    full_keys, full_logp = batch.levels[3]
    for row, user in enumerate(batch.ids):
        if not batch.usable[row]:
            yield None
            continue
        prefix_keys, prefix_logp = (empty, empty) if batch.prefix is None else (
            batch.prefix[0][row], batch.prefix[1][row])
        yield (full_keys[row], full_logp[row], prefix_keys, prefix_logp,
               batch.queries[row], data.seen_items(user))


def rank_batch(ranker, batch, data, routes):
    return (None if entry is None else ranker.rank(*entry, routes=routes)
            for entry in rank_inputs(batch, data))


def prepare_ranker(ranker, routes):
    from .two_tower_retrieval import tie_keys

    if ranker.ties is None:
        ranker.ties = tie_keys(ranker)
    if 'content' in routes and ranker._content_pool is None:
        ranker._build_content_index()


def predict(model, dataset, data, ranker, decode, device, precision='fp32', *,
            routes=ROUTES, query_batch_size=8, decoder_batch_size=128,
            external_query=None, trie=None, timings=None, trace_batch=None,
            ranking_directory=None):
    from .generator_rank_pool import RankPool

    if decode.ranking == 'beam_product' and tuple(routes) != ('full_sid',):
        raise ValueError('beam_product is only available for the complete full_sid list')
    predictions = {route: np.full((len(data.user_ids), decode.k), PADDING_ITEM_ID, dtype=np.uint32)
                   for route in routes}
    yield_counts = {route: dict(returned=0, candidates=0, new_candidates=0, empty=0)
                    for route in routes}
    started = time.perf_counter()
    prepare_ranker(ranker, routes)
    workers = min(decode.ranking_workers, torch.get_num_threads())
    batches = decode_batches(model, dataset, data, decode, device, precision,
                             prefix_route='prefix' in routes, query_batch_size=query_batch_size,
                             decoder_batch_size=decoder_batch_size, external_query=external_query,
                             trie=trie, timings=timings)
    with RankPool(ranker, routes, workers, temp_directory=ranking_directory) as pool:
        if timings is not None:
            timings.update(ranking_setup_seconds=time.perf_counter() - started,
                           ranking_seconds=0., ranking_workers=workers)
        for batch in batches:
            started = time.perf_counter()
            for row, outcomes in enumerate(pool.rank(rank_inputs(batch, data))):
                if outcomes is None:
                    for counts in yield_counts.values():
                        counts['empty'] += 1
                    continue
                for route, result in outcomes.items():
                    items = result.items
                    predictions[route][batch.start + row, :len(items)] = items
                    counts = yield_counts[route]
                    counts['returned'] += len(items)
                    counts['candidates'] += result.candidate_count
                    counts['new_candidates'] += result.new_candidates
                    counts['empty'] += not len(items)
            if timings is not None:
                timings['ranking_seconds'] += time.perf_counter() - started
            if trace_batch is not None:
                trace_batch(batch)
    return predictions, yield_counts


def report_ks(k):
    return tuple(sorted({value for value in (50, 200, 1000) if value <= k} | {k}))


def supply_summary(predictions, ks):
    result = {}
    for route, ranked in predictions.items():
        available = np.count_nonzero(ranked != PADDING_ITEM_ID, axis=1)
        by_k = {}
        for k in ks:
            short = int(np.count_nonzero(available < k))
            values = np.minimum(available, k)
            quantiles = np.percentile(values, [10, 50, 90]) if len(values) else [None] * 3
            by_k[k] = dict(zip(('p10', 'p50', 'p90'), quantiles),
                           short_users=short, users=len(values),
                           short_fraction=short / len(values) if len(values) else None)
        result[route] = by_k
    return result


def first_prune_depth(code_keys, retained_levels, size):
    """Ant §2.2 Eq. 4. Depth 0 means every prefix was retained."""
    code_keys = np.asarray(code_keys)
    missing_depth = np.zeros(len(code_keys), dtype=np.uint8)
    for depth in (1, 2, 3):
        missing = ~np.isin(code_keys // (size ** (3 - depth)), retained_levels[depth])
        missing_depth[(missing_depth == 0) & missing] = depth
    return missing_depth


def evaluate_predictions(data, predictions, catalog_groups, *, replicates=1000, k=1000,
                         output=None):
    """Split target labels while retaining the SDK's queries, catalog and history."""
    ids = data.items['item_id']
    target_rows = np.searchsorted(ids, data.targets['item_id'])
    if np.any(target_rows >= len(ids)) or not np.array_equal(ids[target_rows], data.targets['item_id']):
        raise ValueError('evaluation targets are outside the catalog')
    groups = np.asarray(catalog_groups)[target_rows]
    slices = {'all': data}
    for name, group in PROXY_GROUPS.items():
        keep = groups == group
        targets = dict(data.targets)
        for column in ('main_positive', 'secondary_positive', 'like_positive'):
            targets[column] = targets[column] & keep
        slices[name] = replace(data, targets=targets, _counts=None, _target_users=None,
                               _statistics_identity=None)
    results = {}
    ks = report_ks(k)
    for route, prediction in predictions.items():
        results[route] = {}
        for name, scope in slices.items():
            location = None if output is None else Path(output) / route / name
            results[route][name] = evaluate(scope, prediction, model_name=route,
                                           ks=ks, replicates=replicates, output_dir=location).metrics
    return results


def early_recall(model, dataset, data, ranker, decode, device, precision, *,
                 external_query=None, trie=None, query_batch_size=8, supply=None):
    settings = replace(decode, k=min(200, decode.k))
    # The supplied ranker must use settings; beam and target budget are unchanged.
    if ranker.config != settings:
        raise ValueError('early ranker is not using the configured early top-K')
    predictions, _ = predict(model, dataset, data, ranker, settings, device, precision,
                             routes=('full_sid',), external_query=external_query, trie=trie,
                             query_batch_size=query_batch_size)
    if supply is not None:
        supply.update(supply_summary(predictions, (200,)))
    return primary(evaluate(data, predictions['full_sid'], model_name='full_sid',
                            ks=(200,), replicates=1))['macro_capped']
