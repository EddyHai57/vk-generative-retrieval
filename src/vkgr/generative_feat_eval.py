"""GFEAT-01: side-aware constrained beam decode and validation evaluation for the f2/f3 models.

The frozen decoders (``tiger.decode_history``, ``generative_batched``, ``generative_evaluation``) are not modified and their beam
logic is reused: ``tiger.constrained_beam`` for the per-query path and ``generative_batched._decode_group`` for the batched path.
This module only binds each history's side rows to ``encode``. Artifacts keep the frozen evaluators' layout (shards, progress
files, manifest) so ``generative_evaluation._read_evaluation`` and the Batch1 verifier read them unchanged.
"""
import hashlib
import resource
import sys
import time

import numpy as np
import torch

from . import generative_batched as gb
from . import generative_evaluation as ev
from .data_audit import canonical_bytes
from .generative_data import _no_symlink
from .generative_experiment import _write_json
from .generative_feat import side_tensor
from .tiger import constrained_beam


def decode_history_side(model, codec, history, side, beam=20):
    """``tiger.decode_history`` with the history's side rows passed to the encoder (one query)."""
    if len(side) != len(history):
        raise ValueError('side/history length mismatch')
    device = next(model.parameters()).device
    tokens = codec.history(history)  # keeps the last 50 items; the side rows keep the same window
    source = torch.tensor([tokens], dtype=torch.long, device=device)
    side_row = side_tensor([side[-(len(tokens) // 4):]], device)
    model.eval()
    with torch.no_grad():
        memory, padding = model.encode(source, side_row)

        def score(prefixes):
            rows = [[1] + [codec.offsets[j] + digit for j, digit in enumerate(path)] for path in prefixes]
            decoder = torch.tensor(rows, dtype=torch.long, device=device)
            return model.decode(memory.expand(len(rows), -1, -1), padding.expand(len(rows), -1), decoder)[:, -1]
        return constrained_beam(codec.catalog, codec, score, beam)


class _SideBound:
    """Lets the unchanged ``generative_batched._decode_group`` encode with a fixed side tensor."""

    def __init__(self, model, side):
        self.model, self.side = model, side

    def parameters(self):
        return self.model.parameters()

    def encode(self, source):
        return self.model.encode(source, self.side)

    def decode(self, memory, padding, decoder):
        return self.model.decode(memory, padding, decoder)


def check_batch(batch_size):
    """Batch size 1 only. A decoder call that mixes the beams of several queries has a different row count than the serial
    call, and float32 results then differ in the last bits (INFRA-02b: batch>1 is not bit-equal to the serial decode, and a
    near-tie can swap two neighbouring candidates). With one query per call the shapes are the serial ones."""
    if type(batch_size) is not int or batch_size != 1:
        raise ValueError('side decode is registered for batch size 1 only (batch>1 is not bit-equal to the serial decode)')


def decode_histories_batched_side(model, tokens, histories, sides, beam, batch_size=1):
    """Vectorised-beam twin (``generative_batched._decode_group``) of ``decode_history_side``; one query per decoder call
    (``batch_size`` must be 1), same ``(items, audit)`` per query as the serial decode."""
    if type(beam) is not int or beam < 1:
        raise ValueError('invalid beam')
    check_batch(batch_size)
    if len(histories) != len(sides) or any(len(s) != len(h) for h, s in zip(histories, sides)):
        raise ValueError('side/history mismatch')
    device = next(model.parameters()).device
    results = []
    model.eval()
    with torch.no_grad():
        for history, side in zip(histories, sides):
            source = tokens.history(history)
            bound = _SideBound(model, side_tensor([side[-(len(source) // 4):]], device))
            results.append(gb._decode_group(bound, tokens, [source], beam)[0])
    return results


def _publish(output, binding, queries, start, end, new, shards, rows):
    """One immutable shard file + (while incomplete) one progress receipt, exactly as the frozen evaluators write them."""
    path = output / f'shard-{start:06d}-{end:06d}.json'
    pin = _write_json(path, {'schema_version': 1, 'binding': binding, 'start': start, 'end': end, 'rows': new})
    shards.append({'start': start, 'end': end, 'path': str(path.absolute()), 'sha256': pin}); rows.extend(new)
    progress = {'schema_version': 1, 'task_id': 'G2', 'kind': 'query_evaluation', 'binding': binding, 'query_count': len(queries), 'completed_queries': len(rows),
                'complete': len(rows) == len(queries), 'shards': list(shards), 'computation_sha256': ev._computation(rows, None), 'aggregate': None, 'observations': None}
    if not progress['complete']:
        _write_json(output / f'progress-{len(rows):06d}.json', progress)


def evaluate_queries_side(model, context, validation_side, *, output, checkpoint_sha256, batch_size=None, beam=100, shard_size=64,
                          budget_check=None, on_shard=None):
    """Twin of ``ev.evaluate_queries`` (``batch_size=None``, one query at a time) and ``gb.evaluate_queries_batched`` (integer
    ``batch_size``) for side models. ``validation_side[ordinal]`` holds the side rows of ``context.queries[ordinal]['history']``.
    No resume / stop: a failed epoch evaluation is re-run whole."""
    binding = ev._evaluation_binding(model, context, checkpoint_sha256, beam, shard_size)
    queries = context.queries
    if type(validation_side) not in (list, tuple) or len(validation_side) != len(queries) \
            or any(len(side) != len(query['history']) for side, query in zip(validation_side, queries)):
        raise ValueError('validation side not aligned with the query histories')
    if batch_size is not None:
        check_batch(batch_size)
    if budget_check is not None:
        budget_check()
    output = _no_symlink(output); output.mkdir(mode=0o700, exist_ok=False)
    device = next(model.parameters()).device; catalog = set(context.catalog)
    per_block = 1 if batch_size is None else max(1, (gb._BLOCK_FACTOR * batch_size) // shard_size)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    rows = []; shards = []; decode_seconds = 0.
    starts = list(range(0, len(queries), shard_size)); cursor = 0
    while cursor < len(starts):
        block = starts[cursor:cursor + per_block]; cursor += len(block)
        low = block[0]; high = min(block[-1] + shard_size, len(queries))
        if batch_size is None:
            results = []; latency = []
            for ordinal in range(low, high):
                if budget_check is not None:
                    budget_check()
                gb._sync(device); begin = time.perf_counter()
                results.append(decode_history_side(model, context.codec, queries[ordinal]['history'], validation_side[ordinal], beam))
                gb._sync(device); latency.append(time.perf_counter() - begin)
                if budget_check is not None:
                    budget_check()
            decode_seconds += sum(latency)
        else:
            if budget_check is not None:
                budget_check()
            gb._sync(device); begin = time.perf_counter()
            results = decode_histories_batched_side(model, context.codec, [queries[o]['history'] for o in range(low, high)],
                                                    [validation_side[o] for o in range(low, high)], beam, batch_size)
            gb._sync(device); elapsed = time.perf_counter() - begin; decode_seconds += elapsed
            if budget_check is not None:
                budget_check()
            if len(results) != high - low:
                raise ValueError('batched result count')
            latency = [elapsed / (high - low)] * (high - low)
        for start in block:
            end = min(start + shard_size, len(queries)); new = []
            for ordinal in range(start, end):
                preds, audit = results[ordinal - low]; gb._gate(preds, audit, catalog)
                new.append({'ordinal': ordinal, 'query_sha256': hashlib.sha256(canonical_bytes(queries[ordinal])).hexdigest(), 'items': preds,
                            'audit': audit, 'latency_seconds': latency[ordinal - low]})
            _publish(output, binding, queries, start, end, new, shards, rows)
            if on_shard is not None:
                on_shard(len(rows), len(queries))
    aggregate = ev.aggregate_items([r['items'] for r in rows], queries, context.catalog, slice_definition=context.slice_definition)
    seconds = [row['latency_seconds'] for row in rows]
    observations = {'latency': {'count': len(seconds), 'p50_seconds': float(np.quantile(seconds, .5)) if seconds else None,
                                'p95_seconds': float(np.quantile(seconds, .95)) if seconds else None,
                                'definition': 'per-query tokenization+encoder+four-position beam; linear quantile' if batch_size is None
                                else 'amortized per-query batched decode wall (block wall / block queries); linear quantile'},
                    'rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                    'rss_definition': 'process lifetime observed high-water RSS; not evaluation-only peak',
                    'vram': 'not_applicable' if device.type == 'cpu' else {'allocated_bytes': torch.cuda.memory_allocated(device),
                                                                           'definition': 'observed at evaluation completion; not peak'}}
    if batch_size is not None:
        observations['execution'] = gb._execution(batch_size, per_block * shard_size, decode_seconds, device)
    manifest = {'schema_version': 1, 'task_id': 'G2', 'kind': 'query_evaluation', 'binding': binding, 'query_count': len(queries), 'completed_queries': len(rows),
                'complete': True, 'shards': shards, 'computation_sha256': ev._computation(rows, aggregate), 'aggregate': aggregate, 'observations': observations}
    path = output / 'manifest.json'; pin = _write_json(path, manifest)
    ev._read_evaluation(path, pin, binding, queries, context.catalog)  # fresh read-back of the published artifacts
    return {'manifest': manifest, 'manifest_path': path, 'manifest_sha256': pin, 'aggregate': aggregate, 'observations': observations}
