"""G4 tools: random-SID control codec, decode-only beam-width diagnostic, L100 serial evaluation.

None of these change the formal beam=100 evaluator (generative_evaluation is frozen).
Diagnostic manifests use ``kind='beam_diagnostic'``; the B1 selection/verify path only
accepts ``kind='g_b1_train'`` and therefore cannot consume them.
"""
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .data_audit import canonical_bytes, file_sha256

RANDOM_SID_SEED = 42
DIAGNOSTIC_BEAMS = (50, 200)


def permuted_sid_codec(codec, seed=RANDOM_SID_SEED):
    """SID_RANDPERM42_OF_BHKM_B16D3: items get the 3-digit leaf prefix of a seeded random permutation.

    Leaf occupancy multiset (hence suffix vocabulary, bases and vocab) is unchanged; inside each
    leaf the suffix is re-canonicalised by ascending item id, exactly as ``_check_codec`` requires.
    """
    from . import generative_data as gd
    from .sid import SIDCatalog
    base = codec.catalog
    ids = np.array(base.ids, dtype=np.uint32); sids = np.array(base.sids, dtype=np.uint32)
    perm = np.random.RandomState(seed).permutation(len(ids))
    prefixes = sids[perm, :base.depth]
    order = np.lexsort((ids, prefixes[:, 2], prefixes[:, 1], prefixes[:, 0]))
    codes = prefixes[order]
    starts = np.r_[True, np.any(codes[1:] != codes[:-1], axis=1)]
    rows = np.arange(len(codes))
    suffix = rows - np.maximum.accumulate(np.where(starts, rows, 0))
    new = np.zeros_like(sids); new[:, :base.depth] = prefixes; new[order, base.depth] = suffix.astype(np.uint32)
    catalog = SIDCatalog(ids, new, base.branching, base.depth, base.nodes)
    codec = gd._ValidatedCodec(catalog)
    gd._check_codec(codec)
    return codec


def decode_history_l(model, tokens, history, beam):
    """``tiger.decode_history`` with an L-window tokenizer and any beam width."""
    from .tiger import constrained_beam
    device = next(model.parameters()).device
    source = torch.tensor([tokens.history(history)], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        memory, padding = model.encode(source)
        def score(prefixes):
            rows = [[1] + [tokens.offsets[j] + digit for j, digit in enumerate(path)] for path in prefixes]
            decoder = torch.tensor(rows, dtype=torch.long, device=device)
            return model.decode(memory.expand(len(rows), -1, -1), padding.expand(len(rows), -1), decoder)[:, -1]
        return constrained_beam(tokens.catalog, tokens, score, beam)


def _aggregate(predictions, context):
    from . import generative_evaluation as ev
    # aggregate_items buckets by truncated length <=50; for L100 use the last-50 view for that slice only.
    queries = [{**q, 'history': q['history'][-50:], 'history_positions': q['history_positions'][-50:]} for q in context.queries]
    return ev.aggregate_items(predictions, queries, context.catalog, slice_definition=context.slice_definition)


def _decode_rows(model, context, tokens, beam, guard=None):
    rows = []; queries = context.queries; device = next(model.parameters()).device; catalog = set(context.catalog)
    for ordinal, query in enumerate(queries):
        if guard is not None:
            guard()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        begin = time.perf_counter(); items, audit = decode_history_l(model, tokens, query['history'], beam)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - begin
        if not items or len(items) != len(set(items)) or any(i not in catalog for i in items) or any(audit[k] != 0 for k in ('invalid', 'deadend', 'collision')):
            raise ValueError('decoded candidate/audit gate')
        rows.append({'ordinal': ordinal, 'query_sha256': hashlib.sha256(canonical_bytes(query)).hexdigest(), 'items': items, 'audit': audit, 'latency_seconds': elapsed})
    return rows


def _write(path, value):
    from .generative_experiment import _write_json
    return _write_json(Path(path), value)


def _latency(rows):
    values = [r['latency_seconds'] for r in rows]
    return {'count': len(values), 'p50_seconds': float(np.quantile(values, .5)) if values else None, 'p95_seconds': float(np.quantile(values, .95)) if values else None}


def evaluate_l100(model, context, tokens, *, output, checkpoint_sha256, guard=None):
    """Serial beam100 evaluation for L100 (sharded workers rebuild the frozen 200-position model)."""
    from . import generative_evaluation as ev
    from .generative_data import _no_symlink
    binding = {'context': context.binding, 'checkpoint_sha256': checkpoint_sha256, 'model_sha256': ev.model_digest(model), 'beam': 100, 'window': tokens.max_len}
    output = _no_symlink(Path(output)); output.mkdir(mode=0o700, exist_ok=False)
    rows = _decode_rows(model, context, tokens, 100, guard)
    aggregate = _aggregate([r['items'] for r in rows], context)
    rows_sha = _write(output / 'rows.json', rows)
    manifest = {'schema_version': 1, 'task_id': 'G-B1', 'kind': 'l100_query_evaluation', 'binding': binding, 'rows_sha256': rows_sha,
                'query_count': len(rows), 'aggregate': aggregate, 'computation_sha256': ev._computation(rows, aggregate), 'observations': {'latency': _latency(rows)}}
    path = output / 'manifest.json'
    return {'manifest': manifest, 'manifest_path': path, 'manifest_sha256': _write(path, manifest), 'aggregate': aggregate}


def read_l100_evaluation(path, pin, binding, context):
    from . import generative_evaluation as ev
    from .generative_data import _json, _checked_file
    path = Path(path); value = _json(_checked_file(path, pin))
    if value.get('kind') != 'l100_query_evaluation' or canonical_bytes(value.get('binding')) != canonical_bytes(binding):
        raise ValueError('L100 evaluation identity')
    rows = _json(_checked_file(path.parent / 'rows.json', value['rows_sha256'])); catalog = set(context.catalog)
    if len(rows) != len(context.queries):
        raise ValueError('L100 query coverage')
    for ordinal, (row, query) in enumerate(zip(rows, context.queries)):
        if row.get('ordinal') != ordinal or row.get('query_sha256') != hashlib.sha256(canonical_bytes(query)).hexdigest() or not row['items'] or any(i not in catalog for i in row['items']) or len(set(row['items'])) != len(row['items']) or any(row['audit'][k] != 0 for k in ('invalid', 'deadend', 'collision')):
            raise ValueError('L100 row gate')
    aggregate = _aggregate([r['items'] for r in rows], context)
    if canonical_bytes(aggregate) != canonical_bytes(value['aggregate']) or ev._computation(rows, aggregate) != value['computation_sha256']:
        raise ValueError('L100 aggregate/computation')
    return value, rows


def _prefix_summary(rows, context, reference_rows, top=50):
    catalog = context.codec.catalog; queries = context.queries; depth = catalog.depth
    hits = {str(k): 0 for k in range(1, depth + 2)}; in_catalog = 0; prefix3 = 0; prefix3_target = 0; rescued = 0; lost = 0
    for row, query in zip(rows, queries):
        top_items = row['items'][:top]
        if not query['oov']:
            in_catalog += 1; target = catalog.encode(int(query['target']))
            paths = [catalog.encode(int(i)) for i in top_items]
            for k in range(1, depth + 2):
                if any(p[:k] == target[:k] for p in paths):
                    hits[str(k)] += 1
            if any(p[:depth] == target[:depth] for p in paths):
                prefix3 += 1; prefix3_target += int(query['target']) in top_items
        if reference_rows is not None:
            here = query['target'] in top_items; there = query['target'] in reference_rows[row['ordinal']]['items'][:top]
            rescued += here and not there; lost += there and not here
    return {'in_catalog_queries': in_catalog,
            'prefix_hit_rate': {k: (v / in_catalog if in_catalog else None) for k, v in hits.items()},
            'suffix_hit_given_prefix3': prefix3_target / prefix3 if prefix3 else None, 'prefix3_hit_queries': prefix3,
            'rescued_vs_beam100': rescued if reference_rows is not None else None, 'lost_vs_beam100': lost if reference_rows is not None else None,
            'definition': f'top{top} candidates; prefix k = first k SID digits (k=4 is the full SID)'}


def decode_diagnostic(model, context, tokens, beam, *, output, binding, reference_rows=None, guard=None):
    """Decode-only (0 optimizer steps) beam-width diagnostic over all validation queries."""
    from .generative_data import _no_symlink
    if beam not in DIAGNOSTIC_BEAMS:
        raise ValueError('diagnostic beam must be 50 or 200; formal beam stays 100')
    output = _no_symlink(Path(output)); output.mkdir(mode=0o700, exist_ok=False)
    rows = _decode_rows(model, context, tokens, beam, guard)
    aggregate = _aggregate([r['items'] for r in rows], context)
    rows_sha = _write(output / 'rows.json', rows)
    value = {'schema_version': 1, 'task_id': 'G4', 'kind': 'beam_diagnostic', 'beam': beam, 'query_count': len(rows), 'optimizer_steps': 0,
             'binding': binding, 'rows_sha256': rows_sha, 'aggregate': aggregate,
             'summary': {**_prefix_summary(rows, context, reference_rows), 'latency': _latency(rows)},
             'limitation': 'a wider beam can still miss high-score paths; not a formal metric'}
    path = output / 'manifest.json'
    return {'manifest': value, 'manifest_path': path, 'manifest_sha256': _write(path, value)}


def read_diagnostic(path, pin):
    from .generative_data import _json, _checked_file
    value = _json(_checked_file(Path(path), pin))
    if value.get('kind') != 'beam_diagnostic' or value.get('beam') not in DIAGNOSTIC_BEAMS or value.get('optimizer_steps') != 0:
        raise ValueError('beam diagnostic identity')
    _checked_file(Path(path).parent / 'rows.json', value['rows_sha256'])
    return value
