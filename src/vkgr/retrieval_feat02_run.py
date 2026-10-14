"""FEAT-02 P runner: id_mean + the F3 anchor + one more variable (F3-masked / F4 recency / F5 scene) over HIST_V2_SIDE_V1.

New version only: ``retrieval_b1_run`` (Batch1) and ``retrieval_feat01_run`` are imported and reused, never modified.
Config schema ``p-feat02-run-v1`` has the field set of ``p-feat01-run-v1``; only schema, task_id and the ``feature``
values differ. ``feature='f3'`` reproduces the FEAT-01 F3 runner exactly (equivalence control; not a submitted run).
Checkpoints best+last; verify is fail-closed like Batch1/FEAT-01 plus: parameter report and bucket spec bound in the
binding, target-slice metrics / top-50 unsupported share recomputed from the stored predictions, diagnostics of the
best checkpoint re-exported bit for bit.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

from . import retrieval_b1_run as b1
from . import history_ctx as hc
from . import retrieval_feat01_run as f1
from .data_audit import file_sha256
from .history_side import SIDE_VERSION
from .history_v2 import load_prepared
from .retrieval_ablation_run import read_pinned, strict_json
from .retrieval_feat02 import FEATURES, MAX_WEEK, SIDE_WIDTH, bucket_spec_sha256, make_feat02_model, parameter_report

TASK = 'FEAT-02'
SCHEMA = 'p-feat02-run-v1'
INPUTS_SCHEMA = 'p-feat02-inputs-v1'
CHECKPOINT_SCHEMA = 'p-feat02-checkpoint-v1'
MANIFEST_SCHEMA = 'p-feat02-train-manifest-v1'
PIN_PAIRS = (('ctx_manifest', 'ctx_sha256'), ('attr_manifest', 'attr_sha256'))
REQUIRED_PINS = {'f3_f5_scene': ('ctx_manifest', 'ctx_sha256'), 'f3_f6_author_content': ('attr_manifest', 'attr_sha256')}


def default_config():
    return {**b1.default_config(), 'schema': SCHEMA, 'task_id': TASK, 'feature': 'f3_masked', 'side_version': SIDE_VERSION}


def validate_config(value, synthetic=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered FEAT-02 config fields')
    if value['schema'] != SCHEMA or value['task_id'] != TASK or value['side_version'] != SIDE_VERSION or value['feature'] not in FEATURES:
        raise ValueError('unregistered FEAT-02 schema/task/feature')
    if value['architecture'] != 'id_mean' or value['history_len'] != 50 or value['init'] != 'small':
        raise ValueError('FEAT-02 is id_mean + HIST_V2_OBS_L50 + INIT_EMB_N002 only')
    b1.validate_config({**{k: v for k, v in value.items() if k not in ('feature', 'side_version')}, 'schema': 'p-b1-run-v1', 'task_id': 'P-NEG-01'}, synthetic)
    return value


# ---------------------------------------------------------------- data

def synthetic_data(seed=11):
    """FEAT-01 synthetic fixture with real week spread: sorted history weeks in 0..24, validation target week 25.

    The Batch1 fixture has week 0 everywhere, which cannot exercise the week-gap feature. Within-week ordinals stay
    strictly increasing; the first training row is pinned to span weeks 0..24.
    """
    data = f1.synthetic_data(seed); rng = random.Random(seed + 2)

    def positions(history, span=False):
        weeks = sorted(rng.randrange(MAX_WEEK + 1) for _ in history)
        if span:
            weeks[0], weeks[-1] = 0, MAX_WEEK
        return [[w, k] for k, w in enumerate(weeks)]
    rows = []
    for j, r in enumerate(data['rows']):
        pos = positions(r['history'], span=j == 0)
        rows.append({**r, 'history_positions': pos, 'target_position': [rng.randrange(pos[-1][0], MAX_WEEK + 1), len(pos)]})
    queries = [{**q, 'history_positions': positions(q['history'])} for q in data['queries']]
    ctx_rng = random.Random(seed + 3)

    def scene(history):
        """[place, platform] per event; about one in ten uses a value outside the train tables (OOV path)."""
        return [[ctx_rng.randrange(24) if ctx_rng.random() > .1 else 200, ctx_rng.randrange(11) if ctx_rng.random() > .1 else 99] for _ in history]
    rows = [{**r, 'ctx': scene(r['history'])} for r in rows]
    queries = [{**q, 'ctx': scene(q['history'])} for q in queries]
    return {**data, 'rows': rows, 'queries': queries}


def _view(data, length):
    """FEAT-01 view (history, history_positions, side) plus the scene ctx cut to the last ``length`` events."""
    def cut(r):
        return {**r, 'ctx': r['ctx'][-length:]} if 'ctx' in r else r
    rows, queries = f1._view(data, length)
    return [cut(r) for r in rows], [cut(q) for q in queries]


def _tensors2(batch, idmap, width, feature):
    """Padded history ids and the 6-column side tensor [feedback_mask, timespent, duration, week, place, platform].

    Only the history events' own fields are read; the week column is filled for the recency variant only and
    the place/platform columns for the scene variant only.
    """
    import torch
    histories = torch.zeros(len(batch), width, dtype=torch.long)
    side = torch.zeros(len(batch), width, SIDE_WIDTH, dtype=torch.long)
    for j, r in enumerate(batch):
        n = len(r['history'])
        if len(r['side']) != n:
            raise ValueError('side/history misaligned')
        histories[j, :n] = torch.tensor([idmap[i] for i in r['history']])
        side[j, :n, :3] = torch.tensor(r['side'], dtype=torch.long)
        if feature == 'f3_f4_recency':
            weeks = [p[0] for p in r['history_positions']]
            if len(weeks) != n or any(type(w) is not int for w in weeks):
                raise ValueError('history weeks misaligned or not integers')
            side[j, :n, 3] = torch.tensor(weeks, dtype=torch.long)
        if feature == 'f3_f5_scene':
            ctx = r.get('ctx')
            if ctx is None or len(ctx) != n or any(len(c) != 2 or any(type(v) is not int for v in c) for c in ctx):
                raise ValueError('scene ctx missing, misaligned or not integers')
            side[j, :n, 4:6] = torch.tensor(ctx, dtype=torch.long)
    return histories, side


def train_step(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
    import torch
    from .retrieval_negatives import masked_ce_logq
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, rng, pool)
    b1._histories(batch, data['idmap'], config['history_len'])  # same validation as Batch1
    histories, side = _tensors2(batch, data['idmap'], max(len(r['history']) for r in batch), config['feature'])
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


def export_vectors_side2(model, catalog, queries, idmap, batch_size, width, feature, check=None):
    """Normalized user/item vectors plus the RAW (pre-normalization) L2 norm of every catalog item vector."""
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
            histories, side = _tensors2(queries[start:start + batch_size], idmap, width, feature)
            user_rows.append(model.users(histories.to(device), side.to(device)).cpu().numpy())
    raw_items = np.concatenate(item_rows)
    norms = np.linalg.norm(raw_items.astype(np.float64), axis=1)
    return normalize_export(np.concatenate(user_rows), check), normalize_export(raw_items, check), norms


# ---------------------------------------------------------------- diagnostics

def item_norm_stats(norms, catalog, support):
    """Raw item-vector L2 norm (before normalization): count / mean / median for supported and unsupported catalog items."""
    import numpy as np
    groups = {'supported': [], 'unsupported': []}
    for item, norm in zip(catalog, norms.tolist()):
        groups['supported' if item in support else 'unsupported'].append(norm)
    return {name: {'count': len(v), 'mean': float(np.mean(v)) if v else None, 'p50': float(np.percentile(v, 50)) if v else None} for name, v in groups.items()}


def top50_unsupported_share(predictions, support):
    """Mean over queries of the fraction of its top-50 catalog items that are unsupported (no trained id embedding)."""
    tops = [p[:50] for p in predictions]
    if not tops or any(len(top) == 0 for top in tops):
        raise ValueError('empty prediction list')
    return sum(sum(1 for i in top if i not in support) / len(top) for top in tops) / len(tops)


def support_slices(predictions, queries, support, catalog, ks=(10, 20, 50)):
    """Target slices supported / unsupported / oov, sliced exactly like ``evaluate_vectors``; verify recomputes them."""
    from .baselines import metrics
    result = {}
    for name in ('supported', 'unsupported', 'oov'):
        positions = [j for j, q in enumerate(queries) if ('oov' if q['oov'] else 'supported' if q['target'] in support else 'unsupported') == name]
        result[name] = metrics([predictions[j] for j in positions], [queries[j] for j in positions], catalog, ks)['all']
    return result


def evaluate_model(model, data, queries, config, check=None):
    from .retrieval_experiment import evaluate_vectors
    users, items, norms = export_vectors_side2(model, data['catalog'], queries, data['idmap'], config['batch_size'], 50 if config['history_len'] <= 50 else config['history_len'], config['feature'], check)
    result = evaluate_vectors(users, items, data['catalog'], queries, data['support'], check=check); result.pop('exact_seconds')
    predictions = result['predictions']
    diagnostics = {'zero_user_vectors': result['zero_user_vectors'], 'zero_user_vector_fraction': result['zero_user_vectors'] / len(queries),
                   'top1_is_last_history_fraction': sum(1 for p, q in zip(predictions, queries) if p and p[0] == q['history'][-1]) / len(queries),
                   'history_support_slices': f1.history_support_slices(predictions, queries, data['support'], data['catalog']),
                   'top50_unsupported_share': top50_unsupported_share(predictions, data['support']),
                   'item_norm_stats': item_norm_stats(norms, data['catalog'], data['support'])}
    return result, diagnostics


# ---------------------------------------------------------------- binding / run

def binding_for2(data, config, rows, queries, model, device, inputs_binding):
    base = b1.binding_for(data, config, rows, queries, model.init_report, device, inputs_binding)
    binding = {**base, 'task': 'P-FEAT02', 'feature': config['feature'], 'side_version': SIDE_VERSION,
               'side_sha256': b1._digest([r['side'] for r in rows] + [q['side'] for q in queries]),
               'bucket_spec_sha256': bucket_spec_sha256(), 'param_report': parameter_report(model)}
    if config['feature'] == 'f3_f4_recency':
        binding['weeks_sha256'] = b1._digest([[p[0] for p in r['history_positions']] for r in rows + queries])
    if config['feature'] == 'f3_f5_scene':
        binding['ctx_sha256'] = b1._digest([r['ctx'] for r in rows] + [q['ctx'] for q in queries])
    return binding


def _model(data, config, device):
    from .training import seed_all
    seed_all(config['seed'])
    return make_feat02_model(config['feature'], data['content'], data['supported_indices'], config['history_len']).to(device)


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
    binding = binding_for2(data, config, rows, queries, model, device, inputs_binding)
    view = {**data, 'rows': rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    step = 0; records = []; evaluations = []; best = None; last = None; train_seconds = 0.; eval_seconds = 0.
    diagnostics = {**b1.data_diagnostics(rows, queries, data['support']), 'side_visibility': f1.side_visibility(queries, data['support'])}
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
        state = {'schema': CHECKPOINT_SCHEMA, 'binding_sha256': b1._digest(binding), 'step': step, 'epoch': epoch, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'cursor': cursor.state_dict(), 'negative_rng': negative_rng.getstate(),
                 'stats': b1._stats_value(stats), 'selection': stop.state(), **b1._rng_state()}
        last = {'path': 'last.pt', 'sha256': b1._torch_save_atomic(output / 'last.pt', state), 'step': step, 'epoch': epoch}
        if improved:
            best = {'path': 'best.pt', 'sha256': b1._torch_save_atomic(output / 'best.pt', state), 'step': step, 'epoch': epoch, 'metric': list(stop.best_metric)}
        value = {'schema': MANIFEST_SCHEMA, 'kind': 'p_feat02_train', 'status': 'complete' if stop.stopped else 'running', 'config': config,
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
    if set(value) != b1.MANIFEST_FIELDS or value['schema'] != MANIFEST_SCHEMA or value['kind'] != 'p_feat02_train' or value['status'] != 'complete':
        raise ValueError('complete FEAT-02 manifest required')
    synthetic = bool(data.get('synthetic'))
    if value['config'] != config:
        raise ValueError('config mismatch')
    validate_config(config, synthetic)
    if not synthetic and device != 'cuda':
        raise ValueError('real verifier requires producer CUDA')
    b1.execution_context(device)
    rows, queries = _view(data, config['history_len'])
    model = _model(data, config, device)
    if value['binding'] != binding_for2(data, config, rows, queries, model, device, inputs_binding):
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
        predictions = merged['result']['predictions']; stored = evaluation.get('diagnostics', {})
        recomputed = metrics(predictions, queries, data['catalog'], (10, 20, 50))
        if recomputed != evaluation['result']['metrics']:
            raise ValueError('metrics not reproducible from stored predictions')
        if f1.history_support_slices(predictions, queries, data['support'], data['catalog']) != stored.get('history_support_slices'):
            raise ValueError('history support slice not reproducible from stored predictions')
        if support_slices(predictions, queries, data['support'], data['catalog']) != evaluation['result'].get('support_metrics'):
            raise ValueError('target support slice metrics not reproducible from stored predictions')
        if top50_unsupported_share(predictions, data['support']) != stored.get('top50_unsupported_share'):
            raise ValueError('top50 unsupported share not reproducible from stored predictions')
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
    result, diagnostics = evaluate_model(model, data, queries, config, check)
    if result != full[best['epoch'] - 1]['result']:
        raise ValueError('best checkpoint exact re-export mismatch')
    if diagnostics != value['evaluations'][best['epoch'] - 1].get('diagnostics'):
        raise ValueError('best checkpoint exact re-export diagnostics mismatch')
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

def validate_inputs(value, feature=None):
    """p-feat02-inputs-v1 = FEAT-01 inputs + optional ctx/attr pin pairs; a variant that uses a pin requires it."""
    if type(value) is not dict or value.get('schema') != INPUTS_SCHEMA or any(k not in value for pair in PIN_PAIRS for k in pair):
        raise ValueError('FEAT-02 inputs schema')
    pins = {k for pair in PIN_PAIRS for k in pair}
    f1.validate_inputs({**{k: v for k, v in value.items() if k not in pins}, 'schema': 'p-feat01-inputs-v1'})
    for manifest, digest in PIN_PAIRS:
        if (value[manifest] is None) != (value[digest] is None) or (value[digest] is not None and (type(value[manifest]) is not str or not f1._sha(value[digest]))):
            raise ValueError('partial/invalid sidecar pin')
    for key in REQUIRED_PINS.get(feature, ()):
        if value[key] is None:
            raise ValueError('pinned sidecar required by this feature')
    return value


def apply_feat02(data, spec, length):
    """Overlay pinned history-v2 and the FEAT-01 side sidecar (week comes from history_positions); with a ctx pin
    also the HIST_V2_CTX_V1 sidecar (place/platform per history event), bound by its manifest sha."""
    data, binding = f1.apply_side(data, spec, length)
    if spec['ctx_manifest'] is None:
        return data, binding
    history = load_prepared(spec['history_manifest'], spec['history_sha256'], length)
    ctx = hc.load_ctx(spec['ctx_manifest'], spec['ctx_sha256'], spec['history_sha256'], length, history)
    data = {**data, 'rows': [{**r, 'ctx': c} for r, c in zip(data['rows'], ctx['train'])],
            'queries': [{**q, 'ctx': c} for q, c in zip(data['queries'], ctx['validation'])]}
    return data, {**binding, 'ctx_manifest_sha256': spec['ctx_sha256']}


def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description='FEAT-02 P runner (new version; Batch1 and FEAT-01 runners untouched)')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('synthetic-train'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    for name in ('train', 'verify'):
        p = sub.add_parser(name); p.add_argument('--inputs', required=True); p.add_argument('--inputs-sha', required=True)
        p.add_argument('--config', required=True); p.add_argument('--config-sha', required=True); p.add_argument('--output'); p.add_argument('--manifest'); p.add_argument('--manifest-sha')
    for name in ('prepare-ctx', 'verify-ctx'):
        p = sub.add_parser(name); p.add_argument('--inputs', required=True); p.add_argument('--inputs-sha', required=True)
        p.add_argument('--output'); p.add_argument('--manifest'); p.add_argument('--manifest-sha')
    args = parser.parse_args(argv)
    try:
        if args.command == 'synthetic-train':
            value = run_session(args.output, synthetic_data(), strict_json(Path(args.config).read_text()))
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256']})); return 0
        if args.command in ('prepare-ctx', 'verify-ctx'):
            return _ctx_command(args, validate_inputs(read_pinned(args.inputs, args.inputs_sha)))
        config = read_pinned(args.config, args.config_sha); validate_config(config)
        spec = validate_inputs(read_pinned(args.inputs, args.inputs_sha), config['feature'])
        import torch
        if not torch.cuda.is_available():
            raise ValueError('actual CUDA environment required')
        target = Path(args.output) if args.command == 'train' else Path(args.manifest).parent
        check = b1.resource_guard(target, spec['disk_cap_bytes']); check()
        data, _, binding = b1.real_inputs(spec)
        data, extra_binding = apply_feat02(data, spec, config['history_len'])
        inputs_binding = {**binding, **extra_binding, 'inputs_sha256': args.inputs_sha}
        if args.command == 'train':
            value = run_session(target, data, config, 'cuda', check, inputs_binding)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']})); return 0
        print(json.dumps(verify_run(args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs_binding))); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR P-FEAT02 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


def _ctx_command(args, spec):
    """prepare-ctx writes HIST_V2_CTX_V1 from the raw train partitions; verify-ctx rebuilds it from raw and compares value by value."""
    data, partitions, binding = b1.real_inputs(spec, with_partitions=True)
    binding = {**binding, 'consumer': b1.source_identity(), 'inputs_sha256': args.inputs_sha}
    if args.command == 'prepare-ctx':
        receipt = hc.write_ctx(Path(args.output), spec['history_manifest'], spec['history_sha256'], partitions(), binding)
        print(json.dumps({'status': 'ctx_prepared', 'manifest_sha256': receipt['manifest_sha256']})); return 0
    history = load_prepared(spec['history_manifest'], spec['history_sha256'], 100)
    rebuilt = hc.build_ctx(partitions(), history)
    loaded = hc.load_ctx(args.manifest, args.manifest_sha, spec['history_sha256'], 100, history)
    if loaded['train'] != rebuilt['train'] or loaded['validation'] != rebuilt['validation']:
        raise ValueError('raw ctx replay mismatch')
    print(json.dumps({'status': 'ctx_verified'})); return 0


if __name__ == '__main__':
    raise SystemExit(main())
