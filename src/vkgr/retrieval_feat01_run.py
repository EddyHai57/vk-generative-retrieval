"""FEAT-01 P runner: id_mean + one history-event feature switch (F1/F2/F3) over HIST_V2_SIDE_V1.

New version only: ``retrieval_b1_run`` (Batch1) is imported and reused, never modified.
Config schema ``p-feat01-run-v1`` = Batch1 fields + ``feature`` + ``side_version``. ``feature='none'``
reproduces the Batch1 runner exactly (equivalence control; not a submitted run).
Checkpoints best+last; verify is fail-closed exactly like Batch1 (draw replay, metrics from stored
predictions, early-stop replay, best re-export, last state).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

from . import retrieval_b1_run as b1
from .data_audit import file_sha256
from .history_side import SIDE_VERSION, build_side, load_durations, load_side, write_side
from .retrieval_ablation_run import read_pinned, strict_json
from .retrieval_features import FEATURES, make_feature_model

TASK = 'FEAT-01'
SCHEMA = 'p-feat01-run-v1'


def default_config():
    return {**b1.default_config(), 'schema': SCHEMA, 'task_id': TASK, 'feature': 'f1_ratio_pool', 'side_version': SIDE_VERSION}


def validate_config(value, synthetic=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered FEAT-01 config fields')
    if value['schema'] != SCHEMA or value['task_id'] != TASK or value['side_version'] != SIDE_VERSION or value['feature'] not in FEATURES:
        raise ValueError('unregistered FEAT-01 schema/task/feature')
    if value['architecture'] != 'id_mean' or value['history_len'] != 50 or value['init'] != 'small':
        raise ValueError('FEAT-01 batch 1 is id_mean + HIST_V2_OBS_L50 + INIT_EMB_N002 only')
    b1.validate_config({**{k: v for k, v in value.items() if k not in ('feature', 'side_version')}, 'schema': 'p-b1-run-v1', 'task_id': 'P-NEG-01'}, synthetic)
    return value


# ---------------------------------------------------------------- data

def synthetic_data(seed=11):
    """Batch1 synthetic fixture + deterministic per-event side (feedback bits, timespent, item duration)."""
    data = b1.synthetic_data(seed); rng = random.Random(seed + 1)
    durations = {i: rng.choice([5, 9, 25, 60, 180]) for i in data['catalog']}
    def side(history):
        return [[sum(1 << b for b in range(6) if rng.random() < .2), rng.choice([1, 2, 5, 9, 12, 40, 70, 150, 255]), durations[i]] for i in history]
    return {**data, 'rows': [{**r, 'side': side(r['history'])} for r in data['rows']],
            'queries': [{**q, 'side': side(q['history'])} for q in data['queries']]}


def _view(data, length):
    def cut(r):
        if len(r['side']) != len(r['history']):
            raise ValueError('side/history misaligned')
        return {**r, 'history': r['history'][-length:], 'history_positions': r['history_positions'][-length:], 'side': r['side'][-length:]}
    return [cut(r) for r in data['rows']], [cut(q) for q in data['queries']]


def _tensors(batch, idmap, width):
    import torch
    histories = torch.zeros(len(batch), width, dtype=torch.long)
    for j, r in enumerate(batch):
        histories[j, :len(r['history'])] = torch.tensor([idmap[i] for i in r['history']])
    side = torch.zeros(*histories.shape, 3, dtype=torch.long)
    for j, r in enumerate(batch):
        if len(r['side']) != len(r['history']):
            raise ValueError('side/history misaligned')
        side[j, :len(r['side'])] = torch.tensor(r['side'], dtype=torch.long)
    return histories, side


def train_step(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
    import torch
    from .retrieval_negatives import masked_ce_logq
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, rng, pool)
    b1._histories(batch, data['idmap'], config['history_len'])  # same validation as Batch1
    histories, side = _tensors(batch, data['idmap'], max(len(r['history']) for r in batch))
    log_q = b1._log_q(batch, negatives, pool_size, config, frequency)
    device = next(model.parameters()).device; model.train(); optimizer.zero_grad(set_to_none=True)
    loss, skips = masked_ce_logq(model.users(histories.to(device), side.to(device)), model.items(columns.to(device)), labels.to(device), mask.to(device), config['temperature'], log_q)
    stats.record(epoch, len(batch), skips); stats.check(epoch)
    result = {'drawn': len(batch), 'skipped': skips, 'negatives': len(negatives), 'negatives_sha256': b1._digest(negatives), 'loss': None, 'gradient_norm': None}
    if loss is None:
        return result
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError('nonfinite gradient')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite parameters')
    return {**result, 'loss': float(loss.detach()), 'gradient_norm': float(norm.detach())}


def export_vectors_side(model, catalog, queries, idmap, batch_size, width, check=None):
    import numpy as np
    import torch
    from .retrieval_experiment import normalize_export
    model.eval(); device = next(model.parameters()).device; item_rows = []; user_rows = []
    with torch.no_grad():
        for start in range(0, len(catalog), batch_size):
            if check: check()
            item_rows.append(model.items(torch.tensor([idmap[i] for i in catalog[start:start + batch_size]], device=device)).cpu().numpy())
        for start in range(0, len(queries), batch_size):
            if check: check()
            histories, side = _tensors(queries[start:start + batch_size], idmap, width)
            user_rows.append(model.users(histories.to(device), side.to(device)).cpu().numpy())
    return normalize_export(np.concatenate(user_rows), check), normalize_export(np.concatenate(item_rows), check)


HISTORY_SUPPORT_BUCKETS = ('h0', 'h_lt_half', 'h_ge_half', 'h1')


def history_support_bucket(history, support):
    """Card §5 pre-registered slice; ``support`` = full-train likes (train-only). Integer comparisons only."""
    n = len(history); s = sum(1 for i in history if i in support)
    return 'h0' if s == 0 else 'h1' if s == n else 'h_lt_half' if 2 * s < n else 'h_ge_half'


def history_support_slices(predictions, queries, support, catalog):
    """R@50 / N@50 per history-supported-fraction bucket (all-eligible queries, same metric code as ``metrics``)."""
    from .baselines import metrics
    names = [history_support_bucket(q['history'], support) for q in queries]
    result = {}
    for name in HISTORY_SUPPORT_BUCKETS:
        idx = [j for j, n in enumerate(names) if n == name]
        value = metrics([predictions[j] for j in idx], [queries[j] for j in idx], catalog, (50,))['all']
        result[name] = {'count': value['count'], 'recall@50': value['recall@50'], 'ndcg@50': value['ndcg@50']}
    return result


def side_visibility(queries, support):
    """Card §0-1: how many unsupported history events F2 can make visible (needs >=1 feedback bit); F3/C0 make all visible."""
    events = [(i, s[0]) for q in queries for i, s in zip(q['history'], q['side'])]
    unsupported = [m for i, m in events if i not in support]
    with_bits = sum(1 for m in unsupported if m)
    return {'history_events': len(events), 'unsupported_events': len(unsupported),
            'unsupported_event_fraction': len(unsupported) / len(events) if events else None,
            'unsupported_with_feedback_bits': with_bits, 'unsupported_f2_visible_fraction': with_bits / len(unsupported) if unsupported else None,
            'unsupported_with_like_bit': sum(1 for m in unsupported if m & 1),
            'all_unsupported_queries_f2_visible': sum(1 for q in queries if all(i not in support for i in q['history']) and any(s[0] for s in q['side']))}


def evaluate_model(model, data, queries, config, check=None):
    from .retrieval_experiment import evaluate_vectors
    users, items = export_vectors_side(model, data['catalog'], queries, data['idmap'], config['batch_size'], 50 if config['history_len'] <= 50 else config['history_len'], check)
    result = evaluate_vectors(users, items, data['catalog'], queries, data['support'], check=check); result.pop('exact_seconds')
    predictions = result['predictions']
    diagnostics = {'zero_user_vectors': result['zero_user_vectors'], 'zero_user_vector_fraction': result['zero_user_vectors'] / len(queries),
                   'top1_is_last_history_fraction': sum(1 for p, q in zip(predictions, queries) if p and p[0] == q['history'][-1]) / len(queries),
                   'history_support_slices': history_support_slices(predictions, queries, data['support'], data['catalog'])}
    return result, diagnostics


def binding_for(data, config, rows, queries, model, device, inputs_binding):
    base = b1.binding_for(data, config, rows, queries, model.init_report, device, inputs_binding)
    return {**base, 'task': 'P-FEAT01', 'feature': config['feature'], 'side_version': SIDE_VERSION,
            'side_sha256': b1._digest([r['side'] for r in rows] + [q['side'] for q in queries])}


def _model(data, config, device):
    from .training import seed_all
    seed_all(config['seed'])
    return make_feature_model(config['feature'], data['content'], data['supported_indices'], config['history_len']).to(device)


def run_session(output, data, config, device='cpu', check=None, inputs_binding=None):
    import torch
    from .retrieval_hp import EarlyStop
    from .retrieval_training import EpochCursor, SamplerStats
    from .retrieval_negatives import target_frequency
    synthetic = bool(data.get('synthetic'))
    validate_config(config, synthetic)
    if not synthetic and (device != 'cuda' or inputs_binding is None or len(data['rows']) != 86785):
        raise ValueError('unapproved real runtime')
    b1.execution_context(device)
    rows, queries = _view(data, config['history_len'])
    seed = config['seed']; model = _model(data, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    cursor = EpochCursor(len(rows), seed); stats = SamplerStats(); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    negative_rng = random.Random(f'p-b1-negative-v1:{seed}')
    frequency = target_frequency(rows) if config['logq'] else None
    pool = sorted(data['support']) if config['extra_uniform'] else None
    binding = binding_for(data, config, rows, queries, model, device, inputs_binding)
    view = {**data, 'rows': rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    step = 0; records = []; evaluations = []; best = None; last = None; train_seconds = 0.; eval_seconds = 0.
    diagnostics = {**b1.data_diagnostics(rows, queries, data['support']), 'side_visibility': side_visibility(queries, data['support'])}
    b1._write_progress('train', 0, config['max_epochs'])
    while not stop.stopped:
        if check: check()
        begin = time.monotonic()
        indices = cursor.next(config['batch_size'])
        record = train_step(model, optimizer, [rows[i] for i in indices], view, stats, cursor.epoch, config, negative_rng, pool, frequency)
        step += 1; records.append({**record, 'step': step, 'epoch': cursor.epoch + 1, 'indices_sha256': b1._digest(indices)})
        train_seconds += time.monotonic() - begin
        if cursor.offset != cursor.size:
            continue
        begin = time.monotonic(); epoch = cursor.epoch + 1
        b1._write_progress('evaluate', epoch, config['max_epochs'])
        result, epoch_diag = evaluate_model(model, data, queries, config, check)
        primary = result['metrics']['all']; improved = stop.observe(epoch, primary['recall@50'], primary['ndcg@50'])
        predictions_name = f'predictions-epoch-{epoch:03d}.json'
        predictions_sha = b1._write_json_atomic(output / predictions_name, result['predictions'])
        evaluations.append({'epoch': epoch, 'result': {k: v for k, v in result.items() if k != 'predictions'}, 'predictions_ref': predictions_name,
                            'predictions_sha256': predictions_sha, 'diagnostics': epoch_diag})
        eval_seconds += time.monotonic() - begin
        if check: check()
        state = {'schema': 'p-feat01-checkpoint-v1', 'binding_sha256': b1._digest(binding), 'step': step, 'epoch': epoch, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'cursor': cursor.state_dict(), 'negative_rng': negative_rng.getstate(),
                 'stats': b1._stats_value(stats), 'selection': stop.state(), **b1._rng_state()}
        last = {'path': 'last.pt', 'sha256': b1._torch_save_atomic(output / 'last.pt', state), 'step': step, 'epoch': epoch}
        if improved:
            best = {'path': 'best.pt', 'sha256': b1._torch_save_atomic(output / 'best.pt', state), 'step': step, 'epoch': epoch, 'metric': list(stop.best_metric)}
        value = {'schema': 'p-feat01-train-manifest-v1', 'kind': 'p_feat01_train', 'status': 'complete' if stop.stopped else 'running', 'config': config,
                 'binding': binding, 'steps': step, 'records': records, 'evaluations': evaluations, 'selection': stop.state(),
                 'stats': b1._stats_value(stats), 'best': best, 'last': last, 'data_diagnostics': diagnostics,
                 'training_seconds': train_seconds, 'evaluation_seconds': eval_seconds,
                 'hit_max_epochs': stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience}
        sha = b1._write_json_atomic(output / 'manifest.json', value)
        b1._write_progress('complete' if stop.stopped else 'train', epoch, config['max_epochs'])
    return {**value, 'manifest_sha256': sha}


full_evaluations = b1.full_evaluations


def verify_run(path, digest, data, config, device='cpu', check=None, inputs_binding=None):
    import torch
    from .baselines import metrics
    from .retrieval_hp import EarlyStop
    from .retrieval_training import EpochCursor, SamplerStats, candidate_batch
    from .retrieval_negatives import supported_uniform_columns
    from .retrieval_ablation_run import state_equal
    path = Path(path); value = read_pinned(path, digest); run = path.parent
    if set(value) != b1.MANIFEST_FIELDS or value['schema'] != 'p-feat01-train-manifest-v1' or value['kind'] != 'p_feat01_train' or value['status'] != 'complete':
        raise ValueError('complete FEAT-01 manifest required')
    synthetic = bool(data.get('synthetic'))
    if value['config'] != config:
        raise ValueError('config mismatch')
    validate_config(config, synthetic)
    if not synthetic and device != 'cuda':
        raise ValueError('real verifier requires producer CUDA')
    b1.execution_context(device)
    rows, queries = _view(data, config['history_len'])
    model = _model(data, config, device)
    if value['binding'] != binding_for(data, config, rows, queries, model, device, inputs_binding):
        raise ValueError('run input/source/init/side binding mismatch')
    cursor = EpochCursor(len(rows), config['seed']); stats = SamplerStats(); rng = random.Random(f"p-b1-negative-v1:{config['seed']}")
    pool = sorted(data['support']) if config['extra_uniform'] else None
    epoch_ends = []
    for j, record in enumerate(value['records'], 1):
        if check and j % 50 == 1: check()
        indices = cursor.next(config['batch_size']); batch = [rows[i] for i in indices]
        if config['extra_uniform']:
            _, _, mask, negatives, _ = supported_uniform_columns(batch, pool, data['idmap'], rng, config['extra_uniform'])
        else:
            _, _, mask = candidate_batch(batch, data['idmap']); negatives = []
        skips = int((mask.sum(1) <= 1).sum())
        finite = all(record[k] is None or (type(record[k]) is float and math.isfinite(record[k])) for k in ('loss', 'gradient_norm'))
        if record.get('step') != j or record.get('epoch') != cursor.epoch + 1 or record.get('indices_sha256') != b1._digest(indices) or record.get('negatives_sha256') != b1._digest(negatives) or record.get('negatives') != len(negatives) or record.get('drawn') != len(indices) or record.get('skipped') != skips or not finite:
            raise ValueError('independent draw replay mismatch')
        stats.record(cursor.epoch, len(indices), skips); stats.check(cursor.epoch)
        if cursor.offset == cursor.size:
            epoch_ends.append((cursor.epoch + 1, j))
    if len(value['records']) != value['steps'] or b1._stats_value(stats) != value['stats'] or len(epoch_ends) != len(value['evaluations']) or not epoch_ends or epoch_ends[-1][1] != value['steps']:
        raise ValueError('draw/evaluation budget mismatch')
    stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience']); expected_best = None
    full = full_evaluations(run, value)
    for (epoch, step), evaluation, merged in zip(epoch_ends, value['evaluations'], full):
        if check: check()
        if evaluation['epoch'] != epoch:
            raise ValueError('evaluation epoch mismatch')
        recomputed = metrics(merged['result']['predictions'], queries, data['catalog'], (10, 20, 50))
        if recomputed != evaluation['result']['metrics']:
            raise ValueError('metrics not reproducible from stored predictions')
        if history_support_slices(merged['result']['predictions'], queries, data['support'], data['catalog']) != evaluation.get('diagnostics', {}).get('history_support_slices'):
            raise ValueError('history support slice not reproducible from stored predictions')
        primary = recomputed['all']
        if stop.observe(epoch, primary['recall@50'], primary['ndcg@50']):
            expected_best = (epoch, step)
    if not stop.stopped or stop.state() != value['selection']:
        raise ValueError('selection replay mismatch')
    best = value['best']
    if type(best) is not dict or (best.get('epoch'), best.get('step')) != expected_best or best.get('metric') != list(stop.best_metric) or best.get('path') != 'best.pt':
        raise ValueError('best checkpoint selection mismatch')
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True) if file_sha256(run / 'best.pt') == best['sha256'] else None
    if state is None or state.get('binding_sha256') != b1._digest(value['binding']) or state['step'] != best['step'] or state['epoch'] != best['epoch']:
        raise ValueError('best checkpoint hash/identity mismatch')
    model.load_state_dict(state['model'])
    result, _ = evaluate_model(model, data, queries, config, check)
    if result != full[best['epoch'] - 1]['result']:
        raise ValueError('best checkpoint exact re-export mismatch')
    last = value['last']
    if type(last) is not dict or last.get('path') != 'last.pt' or last.get('step') != value['steps'] or file_sha256(run / 'last.pt') != last.get('sha256'):
        raise ValueError('last checkpoint hash mismatch')
    state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
    if state.get('binding_sha256') != b1._digest(value['binding']) or state['step'] != value['steps'] or not state_equal(state['cursor'], cursor.state_dict()) or not state_equal(state['negative_rng'], rng.getstate()) or state['stats'] != value['stats'] or state['selection'] != value['selection']:
        raise ValueError('last state mismatch')
    if sorted(p.name for p in run.glob('*.pt')) != ['best.pt', 'last.pt']:
        raise ValueError('checkpoint policy best+last violated')
    return {'status': 'verified', 'replayed_epochs': [best['epoch']], 'best_epoch': best['epoch'], 'steps': value['steps']}


# ---------------------------------------------------------------- real inputs

def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_inputs(value):
    extra = ('side_manifest', 'side_sha256', 'items_metadata_sha256')
    if type(value) is not dict or value.get('schema') != 'p-feat01-inputs-v1' or any(k not in value for k in extra):
        raise ValueError('FEAT-01 inputs schema')
    b1.validate_inputs({**{k: v for k, v in value.items() if k not in extra}, 'schema': 'p-b1-inputs-v1'})
    if value['history_manifest'] is None:
        raise ValueError('pinned history v2 required')
    if (value['side_manifest'] is None) != (value['side_sha256'] is None) or (value['side_sha256'] is not None and not _sha(value['side_sha256'])):
        raise ValueError('partial/invalid side pin')
    if not _sha(value['items_metadata_sha256']):
        raise ValueError('items metadata pin required')
    return value


def apply_side(data, spec, length):
    from .history_v2 import load_prepared
    if spec['side_manifest'] is None:
        raise ValueError('pinned history side required')
    data, history_binding = b1.apply_history(data, spec, length)
    history = load_prepared(spec['history_manifest'], spec['history_sha256'], length)
    side = load_side(spec['side_manifest'], spec['side_sha256'], spec['history_sha256'], length, history)
    if [r['history'] for r in data['rows']] != [r['history'] for r in history['train']] or [q['history'] for q in data['queries']] != [q['history'] for q in history['validation']]:
        raise ValueError('side history drift')
    data = {**data, 'rows': [{**r, 'side': s} for r, s in zip(data['rows'], side['train'])],
            'queries': [{**q, 'side': s} for q, s in zip(data['queries'], side['validation'])]}
    return data, {**history_binding, 'side_manifest_sha256': spec['side_sha256']}


def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description='FEAT-01 P runner (new version; Batch1 runner untouched)')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('synthetic-train'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    for name in ('train', 'verify', 'prepare-side', 'verify-side'):
        p = sub.add_parser(name); p.add_argument('--inputs', required=True); p.add_argument('--inputs-sha', required=True)
        p.add_argument('--config'); p.add_argument('--config-sha'); p.add_argument('--output'); p.add_argument('--manifest'); p.add_argument('--manifest-sha')
    args = parser.parse_args(argv)
    try:
        if args.command == 'synthetic-train':
            value = run_session(args.output, synthetic_data(), strict_json(Path(args.config).read_text()))
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256']})); return 0
        spec = validate_inputs(read_pinned(args.inputs, args.inputs_sha))
        if args.command in ('prepare-side', 'verify-side'):
            return _side_command(args, spec)
        import torch
        if not torch.cuda.is_available():
            raise ValueError('actual CUDA environment required')
        config = read_pinned(args.config, args.config_sha); validate_config(config)
        target = Path(args.output) if args.command == 'train' else Path(args.manifest).parent
        check = b1.resource_guard(target, spec['disk_cap_bytes']); check()
        data, _, binding = b1.real_inputs(spec)
        data, side_binding = apply_side(data, spec, config['history_len'])
        inputs_binding = {**binding, **side_binding, 'inputs_sha256': args.inputs_sha}
        if args.command == 'train':
            value = run_session(target, data, config, 'cuda', check, inputs_binding)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']})); return 0
        print(json.dumps(verify_run(args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs_binding))); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR P-FEAT01 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


def _side_command(args, spec):
    from .history_v2 import load_prepared
    from .retrieval_data import checked
    data, partitions, binding = b1.real_inputs(spec, with_partitions=True)
    metadata = checked(Path(spec['raw_root']) / 'metadata/items_metadata.parquet', spec['items_metadata_sha256'])
    durations = load_durations(metadata, data['catalog'])
    binding = {**binding, 'items_metadata_sha256': spec['items_metadata_sha256'], 'consumer': b1.source_identity(), 'inputs_sha256': args.inputs_sha}
    if args.command == 'prepare-side':
        receipt = write_side(Path(args.output), spec['history_manifest'], spec['history_sha256'], partitions(), durations, binding)
        print(json.dumps({'status': 'side_prepared', 'manifest_sha256': receipt['manifest_sha256']})); return 0
    history = load_prepared(spec['history_manifest'], spec['history_sha256'], 100)
    rebuilt = build_side(partitions(), history, durations)
    loaded = load_side(args.manifest, args.manifest_sha, spec['history_sha256'], 100, history)
    if loaded['train'] != rebuilt['train'] or loaded['validation'] != rebuilt['validation']:
        raise ValueError('raw side replay mismatch')
    print(json.dumps({'status': 'side_verified'})); return 0


if __name__ == '__main__':
    raise SystemExit(main())
