"""INFRA-02: opt-in batched constrained beam decode for G validation (new runs only).

Default stays the frozen per-query ``tiger.decode_history``. With
``VKGR_DECODE=batched`` the queries of one block are grouped by exact source
width (no padding, so every row sees the same shapes as the serial path), the
active beams of a whole group are scored by one decoder call, and the beam
expansion is vectorised with the serial float64 accumulation
(``total + float(logprob)``) and the serial ``(-score, path)`` / ``(-score, item)``
tie-breaks. Artifacts match ``generative_evaluation.evaluate_queries`` except
wall-clock observations; no tolerance is applied anywhere.
"""
import hashlib
import os
import re
import sys
import time

import numpy as np
import torch

DEFAULT_BATCH = 32
MAX_BATCH = 256
_BLOCK_FACTOR = 16  # queries decoded before publishing = whole shards covering >= 16*batch_size
_ALLOWED = {'catalog': None, 'table': {}}


def decode_mode_from_env(environ=None):
    """None = frozen serial decode; otherwise {'mode': 'batched', 'batch_size': n}."""
    environ = os.environ if environ is None else environ
    mode = environ.get('VKGR_DECODE'); size = environ.get('VKGR_DECODE_BATCH')
    if mode in (None, '', 'serial'):
        if size not in (None, ''):
            raise ValueError('decode env: VKGR_DECODE_BATCH requires VKGR_DECODE=batched')
        return None
    if mode != 'batched':
        raise ValueError('decode env: VKGR_DECODE must be serial or batched')
    if size in (None, ''):
        return {'mode': 'batched', 'batch_size': DEFAULT_BATCH}
    if not re.fullmatch(r'[1-9][0-9]{0,2}', size) or int(size) > MAX_BATCH:
        raise ValueError('decode env: VKGR_DECODE_BATCH integer 1..256')
    return {'mode': 'batched', 'batch_size': int(size)}


def check_decode(decode):
    if decode is None:
        return None
    if type(decode) is not dict or set(decode) != {'mode', 'batch_size'} or decode['mode'] != 'batched':
        raise ValueError('decode env: explicit batched decode switch')
    _check_batch(decode['batch_size'])
    return decode


def _check_batch(batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH:
        raise ValueError('decode batch size integer 1..256')


def _allowed(catalog, path):
    if _ALLOWED['catalog'] is not catalog:
        _ALLOWED['catalog'] = catalog; _ALLOWED['table'] = {}
    table = _ALLOWED['table']
    if path not in table:
        table[path] = np.asarray(catalog.allowed_next(path), dtype=np.int64)
    return table[path]


def _decode_group(model, tokens, sources, beam):
    """Same-width group; returns [(items, audit)] in input order."""
    device = next(model.parameters()).device
    catalog = tokens.catalog; offsets = tokens.offsets; n = len(sources)
    memory, padding = model.encode(torch.tensor(sources, dtype=torch.long, device=device))
    paths = [np.zeros((1, 0), dtype=np.int64) for _ in range(n)]
    totals = [np.zeros(1, dtype=np.float64) for _ in range(n)]
    audits = [{'invalid': 0, 'deadend': 0, 'collision': 0, 'max_active': 1} for _ in range(n)]
    for position in range(4):
        counts = [len(t) for t in totals]; owner = np.repeat(np.arange(n), counts)
        prefix = np.concatenate(paths); rows = len(prefix)
        decoder = np.empty((rows, position + 1), dtype=np.int64); decoder[:, 0] = 1
        for j in range(position):
            decoder[:, j + 1] = offsets[j] + prefix[:, j]
        index = torch.from_numpy(owner).to(device)
        logits = model.decode(memory.index_select(0, index), padding.index_select(0, index), torch.from_numpy(decoder).to(device))[:, -1]
        if logits.shape != (rows, tokens.vocab_size) or not torch.isfinite(logits).all():
            raise ValueError('nonfinite/invalid beam logits')
        logprob = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
        first = 0
        for q in range(n):
            parents = []; digits = []
            for i, path in enumerate(paths[q]):
                allowed = _allowed(catalog, tuple(int(x) for x in path))
                if not len(allowed):
                    audits[q]['deadend'] += 1
                parents.append(np.full(len(allowed), i, dtype=np.int64)); digits.append(allowed)
            parent = np.concatenate(parents); digit = np.concatenate(digits)
            if not len(digit):
                raise ValueError('beam deadend')
            # float32 -> float64 is exact; the add is the serial Python-float add.
            score = totals[q][parent] + logprob[first + parent, offsets[position] + digit].astype(np.float64)
            expanded = np.concatenate([paths[q][parent], digit[:, None]], axis=1)
            order = np.lexsort(tuple(expanded[:, j] for j in reversed(range(position + 1))) + (-score,))[:beam]
            paths[q] = expanded[order]; totals[q] = score[order]
            audits[q]['max_active'] = max(audits[q]['max_active'], len(order))
            first += counts[q]
    results = []
    for q in range(n):
        ranked = [(catalog.decode(tuple(int(x) for x in path)), float(total)) for path, total in zip(paths[q], totals[q])]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        audits[q]['collision'] = len(ranked) - len(set(x[0] for x in ranked))
        if audits[q]['collision']:
            raise ValueError('beam item collision')
        results.append(([x[0] for x in ranked], audits[q]))
    return results


def decode_histories_batched(model, tokens, histories, beam, batch_size):
    """Batched twin of ``tiger.decode_history`` for many histories; same (items, audit) per query."""
    if type(beam) is not int or beam < 1:
        raise ValueError('invalid beam')
    _check_batch(batch_size)
    sources = [tokens.history(h) for h in histories]
    groups = {}
    for index, source in enumerate(sources):
        groups.setdefault(len(source), []).append(index)
    results = [None] * len(sources)
    model.eval()
    with torch.no_grad():
        for width in sorted(groups):
            members = groups[width]
            for start in range(0, len(members), batch_size):
                chunk = members[start:start + batch_size]
                for index, value in zip(chunk, _decode_group(model, tokens, [sources[i] for i in chunk], beam)):
                    results[index] = value
    return results


def _gate(preds, audit, catalog):
    if not preds or len(preds) != len(set(preds)) or any(i not in catalog for i in preds) or any(audit[k] != 0 for k in ('invalid', 'deadend', 'collision')):
        raise ValueError('decoded candidate/audit gate')


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _execution(batch_size, block, seconds, device):
    return {'decode': 'batched', 'batch_size': batch_size, 'grouping': 'exact source width, no padding', 'block_queries': block,
            'decode_seconds': seconds, 'peak_vram_bytes': None if device.type == 'cpu' else torch.cuda.max_memory_allocated(device),
            'definition': 'latency_seconds = block decode wall / queries in block (amortized, not comparable to serial per-query latency)'}


def evaluate_queries_batched(model, context, *, output, checkpoint_sha256, batch_size, beam=100, shard_size=64,
                             resume_manifest=None, resume_sha256=None, stop_after_shards=None, should_stop=None, budget_check=None, on_shard=None):
    """Batched twin of ev.evaluate_queries; same shards/progress/manifest except wall-clock observations."""
    import resource
    from . import generative_evaluation as ev
    from .data_audit import canonical_bytes
    from .generative_experiment import _write_json
    from .generative_data import _no_symlink
    _check_batch(batch_size)
    binding = ev._evaluation_binding(model, context, checkpoint_sha256, beam, shard_size)
    if budget_check is not None:
        budget_check()
    if (resume_manifest is None) != (resume_sha256 is None):
        raise ValueError('paired external resume manifest pin required')
    if stop_after_shards is not None and (type(stop_after_shards) is not int or stop_after_shards < 1 or context.binding['synthetic'] is not True):
        raise ValueError('synthetic bounded interruption only')
    queries = context.queries; rows = []; shards = []
    if resume_manifest is not None:
        parent, rows = ev._read_evaluation(resume_manifest, resume_sha256, binding, queries, context.catalog); shards = parent['shards']
        prior_aggregate = ev.aggregate_items([row['items'] for row in rows], queries, context.catalog, slice_definition=context.slice_definition) if parent['complete'] else None
        if canonical_bytes(parent['aggregate']) != canonical_bytes(prior_aggregate) or parent['computation_sha256'] != ev._computation(rows, prior_aggregate):
            raise ValueError('parent evaluation aggregate/computation')
    output = _no_symlink(output); output.mkdir(mode=0o700, exist_ok=False)
    device = next(model.parameters()).device; catalog = context.catalog
    starts = list(range(len(rows), len(queries), shard_size))
    limit = len(starts) if stop_after_shards is None else min(len(starts), stop_after_shards)
    per_block = max(1, (_BLOCK_FACTOR * batch_size) // shard_size)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    cursor = 0; written = 0; stopping = False; decode_seconds = 0.
    while cursor < limit and not stopping:
        block = starts[cursor:min(limit, cursor + per_block)]; cursor += len(block)
        low = block[0]; high = min(block[-1] + shard_size, len(queries))
        if budget_check is not None:
            budget_check()
        _sync(device); begin = time.perf_counter()
        results = decode_histories_batched(model, context.codec, [queries[o]['history'] for o in range(low, high)], beam, batch_size)
        _sync(device); elapsed = time.perf_counter() - begin; decode_seconds += elapsed
        if budget_check is not None:
            budget_check()
        if len(results) != high - low:
            raise ValueError('batched result count')
        latency = elapsed / (high - low)
        for start in block:
            end = min(start + shard_size, len(queries)); new = []
            for ordinal in range(start, end):
                preds, audit = results[ordinal - low]; _gate(preds, audit, catalog)
                new.append({'ordinal': ordinal, 'query_sha256': hashlib.sha256(canonical_bytes(queries[ordinal])).hexdigest(), 'items': preds, 'audit': audit, 'latency_seconds': latency})
            shard = {'schema_version': 1, 'binding': binding, 'start': start, 'end': end, 'rows': new}
            path = output / f'shard-{start:06d}-{end:06d}.json'; pin = _write_json(path, shard)
            shards.append({'start': start, 'end': end, 'path': str(path.absolute()), 'sha256': pin}); rows.extend(new); written += 1
            progress = {'schema_version': 1, 'task_id': 'G2', 'kind': 'query_evaluation', 'binding': binding, 'query_count': len(queries), 'completed_queries': len(rows),
                        'complete': len(rows) == len(queries), 'shards': list(shards), 'computation_sha256': ev._computation(rows, None), 'aggregate': None, 'observations': None}
            if not progress['complete']:
                _write_json(output / f'progress-{len(rows):06d}.json', progress)
            if on_shard is not None:
                on_shard(len(rows), len(queries))
            if (stop_after_shards is not None and written >= stop_after_shards) or (should_stop is not None and should_stop()):
                stopping = True; break
    complete = len(rows) == len(queries)
    aggregate = ev.aggregate_items([r['items'] for r in rows], queries, catalog, slice_definition=context.slice_definition) if complete else None
    latency = [row['latency_seconds'] for row in rows]
    observations = {'latency': {'count': len(latency), 'p50_seconds': float(np.quantile(latency, .5)) if latency else None, 'p95_seconds': float(np.quantile(latency, .95)) if latency else None,
                                'definition': 'amortized per-query batched decode wall (block wall / block queries); linear quantile'},
                    'rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024), 'rss_definition': 'process lifetime observed high-water RSS; not evaluation-only peak',
                    'vram': 'not_applicable' if device.type == 'cpu' else {'allocated_bytes': torch.cuda.memory_allocated(device), 'definition': 'observed at evaluation completion; not peak'},
                    'execution': _execution(batch_size, per_block * shard_size, decode_seconds, device)}
    manifest = {'schema_version': 1, 'task_id': 'G2', 'kind': 'query_evaluation', 'binding': binding, 'query_count': len(queries), 'completed_queries': len(rows), 'complete': complete,
                'shards': shards, 'computation_sha256': ev._computation(rows, aggregate), 'aggregate': aggregate, 'observations': observations}
    path = output / 'manifest.json'; pin = _write_json(path, manifest)
    ev._read_evaluation(path, pin, binding, queries, catalog)
    return {'manifest': manifest, 'manifest_path': path, 'manifest_sha256': pin, 'aggregate': aggregate, 'observations': observations}


def evaluate_l100_batched(model, context, tokens, *, output, checkpoint_sha256, batch_size, guard=None):
    """Batched twin of generative_diagnostics.evaluate_l100 (same rows/manifest except observations)."""
    from pathlib import Path
    from . import generative_evaluation as ev
    from . import generative_diagnostics as gdiag
    from .data_audit import canonical_bytes
    from .generative_data import _no_symlink
    _check_batch(batch_size)
    binding = {'context': context.binding, 'checkpoint_sha256': checkpoint_sha256, 'model_sha256': ev.model_digest(model), 'beam': 100, 'window': tokens.max_len}
    output = _no_symlink(Path(output)); output.mkdir(mode=0o700, exist_ok=False)
    device = next(model.parameters()).device; queries = context.queries; catalog = set(context.catalog)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    if guard is not None:
        guard()
    _sync(device); begin = time.perf_counter()
    results = decode_histories_batched(model, tokens, [q['history'] for q in queries], 100, batch_size)
    _sync(device); elapsed = time.perf_counter() - begin
    if guard is not None:
        guard()
    rows = []
    for ordinal, (query, (items, audit)) in enumerate(zip(queries, results)):
        _gate(items, audit, catalog)
        rows.append({'ordinal': ordinal, 'query_sha256': hashlib.sha256(canonical_bytes(query)).hexdigest(), 'items': items, 'audit': audit, 'latency_seconds': elapsed / len(queries)})
    aggregate = gdiag._aggregate([r['items'] for r in rows], context)
    rows_sha = gdiag._write(output / 'rows.json', rows)
    manifest = {'schema_version': 1, 'task_id': 'G-B1', 'kind': 'l100_query_evaluation', 'binding': binding, 'rows_sha256': rows_sha,
                'query_count': len(rows), 'aggregate': aggregate, 'computation_sha256': ev._computation(rows, aggregate),
                'observations': {'latency': gdiag._latency(rows), 'execution': _execution(batch_size, len(queries), elapsed, device)}}
    path = output / 'manifest.json'
    return {'manifest': manifest, 'manifest_path': path, 'manifest_sha256': gdiag._write(path, manifest), 'aggregate': aggregate}
