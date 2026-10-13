"""Batch1 P runner: HIST-V2 / HIST-LEN / P4H-v2 / P-NEG-01 / P-REPLAY.

New version only; frozen P4D/P4E/P4H entry points are untouched. Explicit switches:
  init=legacy|small (INIT_EMB_N002), history_len=20|50|100 (HIST_V2_OBS_L*),
  max_epochs/min_epochs/patience (BUDGET_E100_MIN5_P3), logq + extra_uniform (P-NEG-01).
Checkpoints: only ``best.pt`` and ``last.pt`` (atomic replace). Verify replays the draw
sequence, recomputes metrics from stored predictions and the selection, re-exports only the
best checkpoint, and checks the last state. Every mismatch raises (fail-closed).
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

from .data_audit import file_sha256
from .retrieval_ablation_run import PREPARED_SHA, execution_context, read_pinned, source_identity, strict_json, visible_memory_fraction

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('HIST-V2', 'HIST-LEN', 'P4H-v2', 'P-NEG-01', 'P-REPLAY')
SEEDS = (42, 2024, 2025)
BUDGET = 'BUDGET_E100_MIN5_P3'
INIT_SMALL = 'INIT_EMB_N002'


def default_config():
    return {'schema': 'p-b1-run-v1', 'task_id': 'HIST-V2', 'architecture': 'id_mean', 'init': 'small',
            'history_version': 'HIST_V2_OBS', 'history_len': 50, 'learning_rate': .001, 'temperature': .15,
            'batch_size': 512, 'weight_decay': .0001, 'gradient_clip': 1., 'precision': 'fp32',
            'max_epochs': 100, 'min_epochs': 5, 'patience': 3, 'logq': False, 'logq_alpha': 1., 'extra_uniform': 0,
            'checkpoint_policy': 'best_last', 'seed': 42, 'test_gate': 'closed_hash_only'}


def validate_config(value, synthetic=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered B1 config fields')
    fixed = ('schema', 'history_version', 'temperature', 'weight_decay', 'gradient_clip', 'precision', 'patience', 'logq_alpha', 'checkpoint_policy', 'test_gate')
    if any(value[k] != base[k] or type(value[k]) is not type(base[k]) for k in fixed):
        raise ValueError('frozen B1 field changed')
    if value['task_id'] not in TASKS or value['architecture'] not in ('id_mean', 'sasrec') or value['init'] not in ('legacy', 'small'):
        raise ValueError('unregistered task/architecture/init')
    if value['history_len'] not in (20, 50, 100) or type(value['history_len']) is not int or value['architecture'] == 'sasrec' and value['history_len'] != 50:
        raise ValueError('unregistered history length')
    if value['learning_rate'] not in (.001, .0003) or type(value['seed']) is not int or value['seed'] not in SEEDS:
        raise ValueError('unregistered learning rate/seed')
    if type(value['logq']) is not bool or value['extra_uniform'] not in (0, 128) or type(value['extra_uniform']) is not int:
        raise ValueError('unregistered negative switch')
    if any(type(value[k]) is not int for k in ('batch_size', 'max_epochs', 'min_epochs')) or not (1 <= value['batch_size'] <= 512 if synthetic else value['batch_size'] == 512):
        raise ValueError('unregistered batch')
    if value['task_id'] == 'P-REPLAY':
        if (value['init'], value['max_epochs'], value['min_epochs'], value['architecture'], value['history_len'], value['logq'], value['extra_uniform']) != ('legacy', 30, 0, 'id_mean', 50, False, 0):
            raise ValueError('P-REPLAY is legacy init + old budget only')
    elif value['init'] != 'small':
        raise ValueError('new runs use INIT_EMB_N002')
    elif synthetic:
        if not 1 <= value['max_epochs'] <= 100 or not 0 <= value['min_epochs'] <= min(5, value['max_epochs']):
            raise ValueError('bounded synthetic budget')
    elif (value['max_epochs'], value['min_epochs']) != (100, 5):
        raise ValueError('new runs use BUDGET_E100_MIN5_P3')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _write_json_atomic(path, value):
    path = Path(path); tmp = path.with_name(path.name + '.partial')
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    with tmp.open('wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    return file_sha256(path)


def _torch_save_atomic(path, state):
    import torch
    path = Path(path); tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as stream:
        torch.save(state, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    return file_sha256(path)


# ---------------------------------------------------------------- data

def synthetic_data(seed=11):
    """CPU fixture: 400 items, 300 supported, histories up to 120 (exercises L100)."""
    import torch
    rng = random.Random(seed)
    catalog = list(range(400)); idmap = {i: i + 1 for i in catalog}; support = set(range(300))
    rows = []
    for j in range(64):
        length = rng.choice([3, 30, 70, 120])
        history = [rng.randrange(400) for _ in range(length)]
        rows.append({'example_id': j, 'user_id': j % 8, 'target': rng.randrange(300), 'target_position': [1, 1000 + j],
                     'history': history, 'history_positions': [[0, k] for k in range(length)], 'past_likes': sorted({history[0]})})
    queries = []
    for j in range(24):
        length = rng.choice([2, 40, 110])
        history = [rng.randrange(400) for _ in range(length)]
        target = rng.randrange(420)
        queries.append({'user_id': 100 + j, 'history': history, 'history_positions': [[0, k] for k in range(length)],
                        'target': target, 'target_position': [25, j], 'oov': target >= 400})
    queries[0]['history'] = [350, 360]; queries[0]['history_positions'] = [[0, 0], [0, 1]]  # all unsupported -> zero id_mean vector
    return {'rows': rows, 'queries': queries, 'catalog': catalog, 'idmap': idmap, 'content': torch.zeros(401, 64),
            'support': support, 'supported_indices': [idmap[i] for i in sorted(support)], 'synthetic': True}


def p4e_watch_synthetic():
    """P4E synthetic fixture with its watch histories as the v2 L50 histories (P-REPLAY test)."""
    from .retrieval_ablation_run import synthetic_data as p4e
    data = p4e()
    return {**data, 'rows': data['watch_rows'], 'queries': data['watch_queries']}


def _view(data, length):
    rows = [{**r, 'history': r['history'][-length:], 'history_positions': r['history_positions'][-length:]} for r in data['rows']]
    queries = [{**q, 'history': q['history'][-length:], 'history_positions': q['history_positions'][-length:]} for q in data['queries']]
    return rows, queries


def data_diagnostics(rows, queries, support):
    """v2-specific: id_mean sees only supported history items (risk 2)."""
    fractions = [sum(1 for i in q['history'] if i in support) / len(q['history']) for q in queries]
    ordered = sorted(fractions)
    def q(p):
        return ordered[max(0, math.ceil(len(ordered) * p) - 1)]
    lengths = [len(q['history']) for q in queries]
    return {'history_supported_fraction': {'mean': sum(fractions) / len(fractions), 'p10': q(.1), 'p50': q(.5), 'p90': q(.9)},
            'all_unsupported_history_queries': sum(1 for f in fractions if f == 0),
            'all_unsupported_history_fraction': sum(1 for f in fractions if f == 0) / len(fractions),
            'query_history_length': {'mean': sum(lengths) / len(lengths), 'max': max(lengths), 'min': min(lengths)},
            'train_history_length_mean': sum(len(r['history']) for r in rows) / len(rows)}


def binding_for(data, config, rows, queries, init_report, device, inputs_binding):
    import hashlib as h
    return {'task': 'P-B1', 'budget': BUDGET if config['task_id'] != 'P-REPLAY' else 'P4E_E30_P3', 'init_version': INIT_SMALL if config['init'] == 'small' else 'legacy_default',
            'config_sha256': _digest(config), 'history_version': f"HIST_V2_OBS_L{config['history_len']}",
            'rows_sha256': _digest(rows), 'queries_sha256': _digest(queries), 'catalog_sha256': _digest(data['catalog']),
            'support_sha256': _digest(sorted(data['support'])), 'content_sha256': h.sha256(data['content'].cpu().numpy().tobytes()).hexdigest(),
            'source': source_identity(), 'execution': {'device': device, 'precision': 'fp32', 'tf32': False, 'deterministic': True},
            'init_report': init_report, 'inputs': inputs_binding, 'synthetic': bool(data.get('synthetic')), 'test_gate': 'closed_hash_only'}


# ---------------------------------------------------------------- training

def _histories(batch, idmap, width_limit):
    import torch
    if any(not 1 <= len(r['history']) <= width_limit or any(i not in idmap for i in r['history']) for r in batch):
        raise ValueError('invalid history')
    histories = torch.zeros(len(batch), max(len(r['history']) for r in batch), dtype=torch.long)
    for j, r in enumerate(batch):
        histories[j, :len(r['history'])] = torch.tensor([idmap[i] for i in r['history']])
    return histories


def _columns(batch, data, config, rng, pool):
    from .retrieval_training import candidate_batch
    from .retrieval_negatives import supported_uniform_columns
    if config['extra_uniform']:
        columns, labels, mask, negatives, pool_size = supported_uniform_columns(batch, pool, data['idmap'], rng, config['extra_uniform'])
    else:
        (columns, labels, mask), negatives, pool_size = candidate_batch(batch, data['idmap']), [], None
    return columns, labels, mask, negatives, pool_size


def _log_q(batch, negatives, pool_size, config, frequency):
    from .retrieval_negatives import logq_batch, logq_mixed
    if not config['logq']:
        return None
    items = sorted({r['target'] for r in batch} | set(negatives))
    if config['extra_uniform']:
        return logq_mixed(items, set(negatives), frequency, batch_size=len(batch), pool_size=pool_size, extra_count=config['extra_uniform'])
    return logq_batch(items, frequency)


def train_step(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
    import torch
    from .retrieval_negatives import masked_ce_logq
    columns, labels, mask, negatives, pool_size = _columns(batch, data, config, rng, pool)
    histories = _histories(batch, data['idmap'], config['history_len'])
    log_q = _log_q(batch, negatives, pool_size, config, frequency)
    device = next(model.parameters()).device; model.train(); optimizer.zero_grad(set_to_none=True)
    loss, skips = masked_ce_logq(model.users(histories.to(device)), model.items(columns.to(device)), labels.to(device), mask.to(device), config['temperature'], log_q)
    stats.record(epoch, len(batch), skips); stats.check(epoch)
    result = {'drawn': len(batch), 'skipped': skips, 'negatives': len(negatives), 'negatives_sha256': _digest(negatives), 'loss': None, 'gradient_norm': None}
    if loss is None:
        return result
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError('nonfinite gradient')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite parameters')
    return {**result, 'loss': float(loss.detach()), 'gradient_norm': float(norm.detach())}


def export_vectors_l(model, catalog, queries, idmap, batch_size, width, check=None):
    """``width=50`` delegates to the frozen exporter (bit-identical)."""
    import numpy as np
    import torch
    from .retrieval_experiment import export_vectors, normalize_export
    if width == 50:
        return export_vectors(model, catalog, queries, idmap, batch_size, check)
    model.eval(); device = next(model.parameters()).device; item_rows = []; user_rows = []
    with torch.no_grad():
        for start in range(0, len(catalog), batch_size):
            if check: check()
            item_rows.append(model.items(torch.tensor([idmap[i] for i in catalog[start:start + batch_size]], device=device)).cpu().numpy())
        for start in range(0, len(queries), batch_size):
            if check: check()
            batch = queries[start:start + batch_size]; histories = torch.zeros((len(batch), width), dtype=torch.long, device=device)
            for j, q in enumerate(batch):
                histories[j, :len(q['history'])] = torch.tensor([idmap[i] for i in q['history']], device=device)
            user_rows.append(model.users(histories).cpu().numpy())
    return normalize_export(np.concatenate(user_rows), check), normalize_export(np.concatenate(item_rows), check)


def evaluate_model(model, data, queries, config, check=None):
    from .retrieval_experiment import evaluate_vectors
    users, items = export_vectors_l(model, data['catalog'], queries, data['idmap'], config['batch_size'], 50 if config['history_len'] <= 50 else config['history_len'], check)
    result = evaluate_vectors(users, items, data['catalog'], queries, data['support'], check=check); result.pop('exact_seconds')
    predictions = result['predictions']
    diagnostics = {'zero_user_vectors': result['zero_user_vectors'], 'zero_user_vector_fraction': result['zero_user_vectors'] / len(queries),
                   'top1_is_last_history_fraction': sum(1 for p, q in zip(predictions, queries) if p and p[0] == q['history'][-1]) / len(queries)}
    return result, diagnostics


def _rng_state():
    import numpy as np
    import torch
    r = np.random.get_state()
    return {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], 'python_rng': random.getstate(), 'numpy_rng': [r[0], r[1].tolist(), int(r[2]), int(r[3]), float(r[4])]}


def _stats_value(stats):
    return {'drawn': stats.drawn, 'skipped': stats.skipped, 'epochs': [[e, *stats.epochs[e]] for e in sorted(stats.epochs)]}


def _write_progress(stage, epoch, max_epochs, message=None):
    path = os.environ.get('VKGR_PROGRESS_PATH'); run_id = os.environ.get('VKGR_RUN_ID')
    if not path or not run_id:
        return
    record = {'schema': 'vkgr-progress-v1', 'run_id': run_id, 'line': 'P', 'stage': stage, 'phase': 'formal', 'epoch': epoch, 'max_epochs': max_epochs,
              'completed': None, 'total': None, 'message': message, 'updated_at': time.time(), 'pid': os.getpid()}
    try:
        _write_json_atomic(path, record)
    except OSError:
        pass


def run_session(output, data, config, device='cpu', check=None, inputs_binding=None):
    import torch
    from .training import seed_all
    from .retrieval_hp import EarlyStop, make_model
    from .retrieval_training import EpochCursor, SamplerStats
    from .retrieval_negatives import target_frequency
    synthetic = bool(data.get('synthetic'))
    validate_config(config, synthetic)
    if not synthetic and (device != 'cuda' or inputs_binding is None or len(data['rows']) != 86785):
        raise ValueError('unapproved real runtime')
    execution_context(device)
    rows, queries = _view(data, config['history_len'])
    seed = config['seed']
    seed_all(seed); model = make_model(config['architecture'], config['init'], data['content'], data['supported_indices'], config['history_len']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    cursor = EpochCursor(len(rows), seed); stats = SamplerStats(); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    negative_rng = random.Random(f'p-b1-negative-v1:{seed}')
    frequency = target_frequency(rows) if config['logq'] else None
    pool = sorted(data['support']) if config['extra_uniform'] else None
    binding = binding_for(data, config, rows, queries, model.init_report, device, inputs_binding)
    view = {**data, 'rows': rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    step = 0; records = []; evaluations = []; best = None; last = None; train_seconds = 0.; eval_seconds = 0.
    diagnostics = data_diagnostics(rows, queries, data['support'])
    _write_progress('train', 0, config['max_epochs'])
    while not stop.stopped:
        if check: check()
        begin = time.monotonic()
        indices = cursor.next(config['batch_size'])
        record = train_step(model, optimizer, [rows[i] for i in indices], view, stats, cursor.epoch, config, negative_rng, pool, frequency)
        step += 1; records.append({**record, 'step': step, 'epoch': cursor.epoch + 1, 'indices_sha256': _digest(indices)})
        train_seconds += time.monotonic() - begin
        if cursor.offset != cursor.size:
            continue
        begin = time.monotonic(); epoch = cursor.epoch + 1
        _write_progress('evaluate', epoch, config['max_epochs'])
        result, epoch_diag = evaluate_model(model, data, queries, config, check)
        primary = result['metrics']['all']; improved = stop.observe(epoch, primary['recall@50'], primary['ndcg@50'])
        predictions_name = f'predictions-epoch-{epoch:03d}.json'
        predictions_sha = _write_json_atomic(output / predictions_name, result['predictions'])
        evaluations.append({'epoch': epoch, 'result': {k: v for k, v in result.items() if k != 'predictions'}, 'predictions_ref': predictions_name,
                            'predictions_sha256': predictions_sha, 'diagnostics': epoch_diag})
        eval_seconds += time.monotonic() - begin
        if check: check()
        state = {'schema': 'p-b1-checkpoint-v1', 'binding_sha256': _digest(binding), 'step': step, 'epoch': epoch, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'cursor': cursor.state_dict(), 'negative_rng': negative_rng.getstate(),
                 'stats': _stats_value(stats), 'selection': stop.state(), **_rng_state()}
        last = {'path': 'last.pt', 'sha256': _torch_save_atomic(output / 'last.pt', state), 'step': step, 'epoch': epoch}
        if improved:
            best = {'path': 'best.pt', 'sha256': _torch_save_atomic(output / 'best.pt', state), 'step': step, 'epoch': epoch, 'metric': list(stop.best_metric)}
        value = {'schema': 'p-b1-train-manifest-v1', 'kind': 'p_b1_train', 'status': 'complete' if stop.stopped else 'running', 'config': config,
                 'binding': binding, 'steps': step, 'records': records, 'evaluations': evaluations, 'selection': stop.state(),
                 'stats': _stats_value(stats), 'best': best, 'last': last, 'data_diagnostics': diagnostics,
                 'training_seconds': train_seconds, 'evaluation_seconds': eval_seconds,
                 'hit_max_epochs': stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience}
        sha = _write_json_atomic(output / 'manifest.json', value)
        _write_progress('complete' if stop.stopped else 'train', epoch, config['max_epochs'])
    return {**value, 'manifest_sha256': sha}


def full_evaluations(run_dir, value):
    """Manifest evaluations with their hash-checked predictions merged back (P4E shape)."""
    result = []
    for evaluation in value['evaluations']:
        name = evaluation['predictions_ref']
        if Path(name).name != name:
            raise ValueError('predictions path escape')
        predictions = read_pinned(Path(run_dir) / name, evaluation['predictions_sha256'])
        result.append({'epoch': evaluation['epoch'], 'result': {**evaluation['result'], 'predictions': predictions}})
    return result


def compare_replay(b1_full, p4e_evaluations):
    """P-REPLAY gate: every epoch's validation result equal to P4E watch_history seed42."""
    mismatched = [a.get('epoch') for a, b in zip(b1_full, p4e_evaluations) if a.get('epoch') != b.get('epoch') or a['result'] != b['result']]
    equal = len(b1_full) == len(p4e_evaluations) and not mismatched
    return {'equal': equal, 'b1_epochs': len(b1_full), 'p4e_epochs': len(p4e_evaluations), 'mismatched_epochs': mismatched}


# ---------------------------------------------------------------- verify

MANIFEST_FIELDS = {'schema', 'kind', 'status', 'config', 'binding', 'steps', 'records', 'evaluations', 'selection', 'stats', 'best', 'last',
                   'data_diagnostics', 'training_seconds', 'evaluation_seconds', 'hit_max_epochs'}


def verify_run(path, digest, data, config, device='cpu', check=None, inputs_binding=None):
    import torch
    from .baselines import metrics
    from .training import seed_all
    from .retrieval_hp import EarlyStop, make_model
    from .retrieval_training import EpochCursor, SamplerStats, candidate_batch
    from .retrieval_negatives import supported_uniform_columns
    from .retrieval_ablation_run import state_equal
    path = Path(path); value = read_pinned(path, digest); run = path.parent
    if set(value) != MANIFEST_FIELDS or value['schema'] != 'p-b1-train-manifest-v1' or value['kind'] != 'p_b1_train' or value['status'] != 'complete':
        raise ValueError('complete B1 manifest required')
    synthetic = bool(data.get('synthetic'))
    if value['config'] != config:
        raise ValueError('config mismatch')
    validate_config(config, synthetic)
    if not synthetic and device != 'cuda':
        raise ValueError('real verifier requires producer CUDA')
    execution_context(device)
    rows, queries = _view(data, config['history_len'])
    seed_all(config['seed']); model = make_model(config['architecture'], config['init'], data['content'], data['supported_indices'], config['history_len']).to(device)
    if value['binding'] != binding_for(data, config, rows, queries, model.init_report, device, inputs_binding):
        raise ValueError('run input/source/init binding mismatch')
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
        if record.get('step') != j or record.get('epoch') != cursor.epoch + 1 or record.get('indices_sha256') != _digest(indices) or record.get('negatives_sha256') != _digest(negatives) or record.get('negatives') != len(negatives) or record.get('drawn') != len(indices) or record.get('skipped') != skips or not finite:
            raise ValueError('independent draw replay mismatch')
        stats.record(cursor.epoch, len(indices), skips); stats.check(cursor.epoch)
        if cursor.offset == cursor.size:
            epoch_ends.append((cursor.epoch + 1, j))
    if len(value['records']) != value['steps'] or _stats_value(stats) != value['stats'] or len(epoch_ends) != len(value['evaluations']) or not epoch_ends or epoch_ends[-1][1] != value['steps']:
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
        primary = recomputed['all']
        if stop.observe(epoch, primary['recall@50'], primary['ndcg@50']):
            expected_best = (epoch, step)
    if not stop.stopped or stop.state() != value['selection']:
        raise ValueError('selection replay mismatch')
    best = value['best']
    if type(best) is not dict or (best.get('epoch'), best.get('step')) != expected_best or best.get('metric') != list(stop.best_metric) or best.get('path') != 'best.pt':
        raise ValueError('best checkpoint selection mismatch')
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True) if file_sha256(run / 'best.pt') == best['sha256'] else None
    if state is None or state.get('binding_sha256') != _digest(value['binding']) or state['step'] != best['step'] or state['epoch'] != best['epoch']:
        raise ValueError('best checkpoint hash/identity mismatch')
    model.load_state_dict(state['model'])
    result, _ = evaluate_model(model, data, queries, config, check)
    if result != full[best['epoch'] - 1]['result']:
        raise ValueError('best checkpoint exact re-export mismatch')
    last = value['last']
    if type(last) is not dict or last.get('path') != 'last.pt' or last.get('step') != value['steps'] or file_sha256(run / 'last.pt') != last.get('sha256'):
        raise ValueError('last checkpoint hash mismatch')
    state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
    if state.get('binding_sha256') != _digest(value['binding']) or state['step'] != value['steps'] or not state_equal(state['cursor'], cursor.state_dict()) or not state_equal(state['negative_rng'], rng.getstate()) or state['stats'] != value['stats'] or state['selection'] != value['selection']:
        raise ValueError('last state mismatch')
    extra = sorted(p.name for p in run.glob('*.pt'))
    if extra != ['best.pt', 'last.pt']:
        raise ValueError('checkpoint policy best+last violated')
    return {'status': 'verified', 'replayed_epochs': [best['epoch']], 'best_epoch': best['epoch'], 'steps': value['steps']}


# ---------------------------------------------------------------- real inputs + guard

def validate_inputs(value):
    fields = {'schema', 'prepared', 'holdout', 'raw_root', 'history_manifest', 'history_sha256', 'disk_cap_bytes', 'test_gate'}
    if type(value) is not dict or set(value) != fields or value['schema'] != 'p-b1-inputs-v1' or value['test_gate'] != 'closed_hash_only':
        raise ValueError('B1 inputs schema')
    for key in ('prepared', 'holdout', 'raw_root'):
        if type(value[key]) is not str or not Path(value[key]).is_absolute():
            raise ValueError('absolute private input path required')
    if (value['history_manifest'] is None) != (value['history_sha256'] is None):
        raise ValueError('partial history pin')
    if type(value['disk_cap_bytes']) is not int or not 0 < value['disk_cap_bytes'] <= 64 * 1024**3:
        raise ValueError('finite disk cap required')
    return value


def real_inputs(spec, with_partitions=False):
    """P4E-equivalent pinned loading (no plan); validation only, test stays closed."""
    from .retrieval_experiment import load_training_inputs
    from .retrieval import PREPARED_SOURCE
    from .retrieval_data import validate_contract, checked
    from .data_audit import parquet_rows
    pins = validate_contract(ROOT); query_pin = next(p for p in pins['full_query_pins'] if p['role'] == 'validation-queries.json')
    data = load_training_inputs(spec['prepared'], spec['holdout'], PREPARED_SHA, PREPARED_SOURCE, query_pin['sha256'])
    data['synthetic'] = False
    binding = {'prepared_sha': PREPARED_SHA, 'query_sha': query_pin['sha256'], 'catalog_sha': pins['catalog_sha256'], 'test_gate': 'closed_hash_only'}
    if not with_partitions:
        return data, None, binding
    prepared = data['prepared']; records = prepared['binding']['input_records']
    names = [f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet' for w in range(25)]
    train_records = [r for r in records if r['role'] in names]
    if [r['role'] for r in train_records] != names:
        raise ValueError('raw/prepared lineage conflict')
    for record in train_records:
        path = checked(Path(spec['raw_root']) / record['role'], record['sha256'])
        if path.stat().st_size != record['size_bytes']:
            raise ValueError('raw input size mismatch')
    def partitions():
        for week, record in enumerate(train_records):
            yield week, parquet_rows(Path(spec['raw_root']) / record['role'])
    binding['raw_records'] = train_records
    return data, partitions, binding


def apply_history(data, spec, length):
    from .history_v2 import load_prepared, overlay_examples, overlay_queries
    if spec['history_manifest'] is None:
        raise ValueError('pinned history v2 required')
    loaded = load_prepared(spec['history_manifest'], spec['history_sha256'], length)
    data = {**data, 'rows': overlay_examples(data['rows'], loaded['train'], length, set(data['catalog'])),
            'queries': overlay_queries(data['queries'], loaded['validation'], length, set(data['catalog']))}
    return data, {'history_manifest_sha256': spec['history_sha256'], 'history_len': length}


def resource_guard(output, disk_cap_bytes, sample_seconds=30.):
    """Own-job gate (hard rule 5): RAM/VRAM/disk >=20% and run-directory disk cap; sampled."""
    import shutil
    import subprocess
    import sys
    if sys.platform != 'linux':
        raise ValueError('real resources require Linux')
    output = Path(output); last = [None]
    def check():
        now = time.monotonic()
        if last[0] is not None and now - last[0] < sample_seconds:
            return
        last[0] = now
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        membership = [line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
        mounts = [line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup2 ' in line]
        if len(membership) != 1 or len(mounts) != 1:
            raise ValueError('ambiguous cgroup2 mapping')
        mount = mounts[0]; root = Path(mount[4]); member = Path(membership[0]); mountroot = Path(mount[3])
        relative = Path('.') if member == mountroot else member.relative_to(mountroot) if mountroot in member.parents else Path(str(member).lstrip('/'))
        ram = visible_memory_fraction(root / relative, root, memory['MemAvailable'] / memory['MemTotal'])
        disk = shutil.disk_usage(output if output.exists() else output.parent)
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
        pairs = [tuple(map(float, line.split(','))) for line in gpu.stdout.strip().splitlines()]
        fractions = {'ram': ram, 'vram': min(f / t for f, t in pairs), 'disk': disk.free / disk.total}
        if any(not math.isfinite(f) or f < .2 for f in fractions.values()):
            raise ValueError('resource headroom below 20%')
        used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) if output.exists() else 0
        if used > disk_cap_bytes:
            raise ValueError('run disk cap exceeded')
    return check


# ---------------------------------------------------------------- CLI

def _load_config(path, sha):
    config = read_pinned(path, sha) if sha else strict_json(Path(path).read_text())
    return config


def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description='Batch1 P runner (new version; frozen P4D/P4E/P4H untouched)')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('synthetic-train'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    for name in ('train', 'verify', 'prepare-history', 'verify-history', 'replay-compare'):
        p = sub.add_parser(name); p.add_argument('--inputs'); p.add_argument('--inputs-sha')
        p.add_argument('--config'); p.add_argument('--config-sha'); p.add_argument('--output')
        p.add_argument('--manifest'); p.add_argument('--manifest-sha'); p.add_argument('--p4e-manifest'); p.add_argument('--p4e-manifest-sha')
    args = parser.parse_args(argv)
    try:
        if args.command == 'synthetic-train':
            value = run_session(args.output, synthetic_data(), strict_json(Path(args.config).read_text()))
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256']})); return 0
        if args.command == 'replay-compare':
            b1 = read_pinned(args.manifest, args.manifest_sha); p4e = read_pinned(args.p4e_manifest, args.p4e_manifest_sha)
            report = compare_replay(full_evaluations(Path(args.manifest).parent, b1), p4e['evaluations'])
            print(json.dumps(report, sort_keys=True)); return 0 if report['equal'] else 3
        spec = validate_inputs(read_pinned(args.inputs, args.inputs_sha))
        if args.command in ('prepare-history', 'verify-history'):
            return _history_command(args, spec)
        import torch
        if not torch.cuda.is_available():
            raise ValueError('actual CUDA environment required')
        config = read_pinned(args.config, args.config_sha); validate_config(config)
        target = Path(args.output) if args.command == 'train' else Path(args.manifest).parent
        check = resource_guard(target, spec['disk_cap_bytes']); check()
        data, _, binding = real_inputs(spec)
        data, history_binding = apply_history(data, spec, config['history_len'])
        inputs_binding = {**binding, **history_binding, 'inputs_sha256': args.inputs_sha}
        if args.command == 'train':
            value = run_session(target, data, config, 'cuda', check, inputs_binding)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']})); return 0
        value = verify_run(args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs_binding)
        print(json.dumps(value)); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR P-B1 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


def _history_command(args, spec):
    from .history_v2 import write_prepared, load_prepared, build_observed, LENGTHS, truncate_sidecar
    data, partitions, binding = real_inputs(spec, with_partitions=True)
    binding = {**binding, 'consumer': source_identity(), 'inputs_sha256': args.inputs_sha}
    if args.command == 'prepare-history':
        receipt = write_prepared(Path(args.output), partitions(), data['rows'], data['queries'], data['catalog'], binding)
        print(json.dumps({'status': 'prepared', 'manifest_sha256': receipt['manifest_sha256'], 'l100_gate': receipt['manifest']['l100_gate']})); return 0
    built = build_observed(partitions(), data['rows'], data['queries'], data['catalog'], max_len=100)
    for length in LENGTHS:
        loaded = load_prepared(args.manifest, args.manifest_sha, length)
        expected_train = [{'example_id': r['example_id'], 'target': e['target'], 'target_position': e['target_position'], 'history': r['history'], 'history_positions': r['history_positions']} for r, e in zip(truncate_sidecar(built['train'], length), data['rows'])]
        if loaded['train'] != expected_train or loaded['validation'] != truncate_sidecar(built['validation'], length):
            raise ValueError('raw history v2 replay mismatch')
    if loaded['length_stats'] != built['length_stats'] or loaded['slices'] != built['slices']:
        raise ValueError('raw history statistics replay mismatch')
    print(json.dumps({'status': 'history_verified'})); return 0


if __name__ == '__main__':
    raise SystemExit(main())
