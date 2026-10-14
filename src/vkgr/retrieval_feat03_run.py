"""FEAT-03 P runner: id_mean + the F3 anchor + one more variable (Q1a position only / Q1b week gap only / Q2 learnable recency pooling / Q3 user-level activity)
over HIST_V2_SIDE_V1. ``f4_control`` is the equivalence fixture of FEAT-02 F4 (synthetic runs only, never a real run).

New version only: the FEAT-02 runner is used through the module (``fr``), never modified and never imported by name (K2: the mutation
tests patch ``fr._tensors2``). ``train_step / export_vectors_side3 / evaluate_model / binding_for3 / _model3 / run_session / verify_run`` are copies of the
FEAT-02 functions with ``_tensors2 -> _tensors3`` and ``make_feat02_model -> make_feat03_model`` (K1); the f4_control run reproduces the FEAT-02 F4 run.
Config schema ``p-feat03-run-v1`` has the FEAT-02 field set; only schema, task_id and the ``feature`` values differ.
Inputs: the FEAT-02 inputs spec with all four sidecar pins (ctx/attr manifest and sha256) null.
CLI: ``synthetic-train [--device]``, ``synthetic-verify [--device]``, ``train``, ``verify``. ``--device`` defaults to cpu; the device given to
``synthetic-verify`` must equal the one used by ``synthetic-train`` (it is part of the binding, a different device exits with code 2).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from . import retrieval_b1_run as b1
from . import retrieval_feat01_run as f1
from . import retrieval_feat02_run as fr
from . import retrieval_feat03 as f3
from .data_audit import file_sha256
from .history_side import SIDE_VERSION
from .retrieval_ablation_run import read_pinned, strict_json

TASK = 'FEAT-03'
SCHEMA = 'p-feat03-run-v1'
CHECKPOINT_SCHEMA = 'p-feat03-checkpoint-v1'
MANIFEST_SCHEMA = 'p-feat03-train-manifest-v1'
CONTROL = 'f4_control'
Q3 = 'f3_user_activity'
DEFAULT_DEVICE = 'cpu'                                                  # read at call time by build_parser (a mutation tests patches it)
# Variants whose side tensor needs the history week column (f3_pos_only reads no week). Read at call time by _tensors3 (M37 patches it).
NEEDS_WEEK = frozenset({'f4_control', 'f3_gap_only', 'f4_recency_pool', 'f3_user_activity'})


def default_config():
    return {**b1.default_config(), 'schema': SCHEMA, 'task_id': TASK, 'feature': 'f3_pos_only', 'side_version': SIDE_VERSION}


def validate_config(value, synthetic=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered FEAT-03 config fields')
    if value['schema'] != SCHEMA or value['task_id'] != TASK or value['side_version'] != SIDE_VERSION or value['feature'] not in f3.FEATURES3:
        raise ValueError('unregistered FEAT-03 schema/task/feature')
    if value['feature'] == CONTROL and not synthetic:
        raise ValueError('f4_control is an equivalence fixture for tests, never a real run')
    if value['architecture'] != 'id_mean' or value['history_len'] != 50 or value['init'] != 'small':
        raise ValueError('FEAT-03 is id_mean + HIST_V2_OBS_L50 + INIT_EMB_N002 only')
    b1.validate_config({**{k: v for k, v in value.items() if k not in ('feature', 'side_version')}, 'schema': 'p-b1-run-v1', 'task_id': 'P-NEG-01'}, synthetic)
    return value


# ---------------------------------------------------------------- data

def _tensors3(batch, idmap, width, feature):
    """Padded history ids and the 6-column side tensor, built from the history events' own fields only (through ``fr._tensors2``).

    The week column is filled for the variants in NEEDS_WEEK only; place/platform stay 0.
    """
    return fr._tensors2(batch, idmap, width, 'f3_f4_recency' if feature in NEEDS_WEEK else 'f3')


def train_step(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
    from .retrieval_negatives import masked_ce_logq
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, rng, pool)
    b1._histories(batch, data['idmap'], config['history_len'])  # same validation as Batch1
    histories, side = _tensors3(batch, data['idmap'], max(len(r['history']) for r in batch), config['feature'])
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


def export_vectors_side3(model, catalog, queries, idmap, batch_size, width, feature, check=None):
    """Normalized user/item vectors plus the RAW (pre-normalization) L2 norm of every catalog item vector."""
    import numpy as np
    from .retrieval_experiment import normalize_export
    model.eval(); device = next(model.parameters()).device; item_rows = []; user_rows = []
    with torch.no_grad():
        for start in range(0, len(catalog), batch_size):
            if check: check()
            item_rows.append(model.items(torch.tensor([idmap[i] for i in catalog[start:start + batch_size]], device=device)).cpu().numpy())
        for start in range(0, len(queries), batch_size):
            if check: check()
            histories, side = _tensors3(queries[start:start + batch_size], idmap, width, feature)
            user_rows.append(model.users(histories.to(device), side.to(device)).cpu().numpy())
    raw_items = np.concatenate(item_rows)
    norms = np.linalg.norm(raw_items.astype(np.float64), axis=1)
    return normalize_export(np.concatenate(user_rows), check), normalize_export(raw_items, check), norms


# ---------------------------------------------------------------- diagnostics

def _q3_scale_median(model, queries, idmap, batch_size, width, check=None):
    """Median over the queries of ||Len[lb] + Act[ab]|| / ||pooled F3 vector||. The ratios are float64 per query; the median is taken on CPU/numpy
    (a dim-wise torch median is avoided: its determinism on CUDA is not established)."""
    import numpy as np
    model.eval(); device = next(model.parameters()).device; ratios = []
    for start in range(0, len(queries), batch_size):
        if check: check()
        histories, side = _tensors3(queries[start:start + batch_size], idmap, width, Q3)
        ratios.append(f3.q3_scale_ratio(model, histories.to(device), side.to(device)).cpu().numpy())
    return float(np.median(np.concatenate(ratios)))


def _bucket_counts(items, idmap, chunk=4096):
    """Counts per length bucket and per distinct-week bucket from the feature tensors only (no labels)."""
    length = [0] * (len(f3.LEN_EDGES) + 1); activity = [0] * (len(f3.ACT_EDGES) + 1)
    for start in range(0, len(items), chunk):
        batch = items[start:start + chunk]
        histories, side = _tensors3(batch, idmap, max(len(r['history']) for r in batch), Q3)
        valid = histories != 0
        lb = f3.length_bucket(valid.sum(1)); ab = f3.activity_bucket(f3.distinct_weeks(side[..., 3], valid))
        for i, c in enumerate(torch.bincount(lb, minlength=len(length)).tolist()):
            length[i] += c
        for i, c in enumerate(torch.bincount(ab, minlength=len(activity)).tolist()):
            activity[i] += c
    return {'length': length, 'activity': activity}


def user_activity_buckets(rows, queries, idmap):
    """Q3 data diagnostics: how many train examples / validation queries fall into each length and activity bucket (distribution shift check)."""
    return {'train': _bucket_counts(rows, idmap), 'validation': _bucket_counts(queries, idmap)}


def evaluate_model(model, data, queries, config, check=None):
    from .retrieval_experiment import evaluate_vectors
    width = 50 if config['history_len'] <= 50 else config['history_len']
    users, items, norms = export_vectors_side3(model, data['catalog'], queries, data['idmap'], config['batch_size'], width, config['feature'], check)
    result = evaluate_vectors(users, items, data['catalog'], queries, data['support'], check=check); result.pop('exact_seconds')
    predictions = result['predictions']
    diagnostics = {'zero_user_vectors': result['zero_user_vectors'], 'zero_user_vector_fraction': result['zero_user_vectors'] / len(queries),
                   'top1_is_last_history_fraction': sum(1 for p, q in zip(predictions, queries) if p and p[0] == q['history'][-1]) / len(queries),
                   'history_support_slices': f1.history_support_slices(predictions, queries, data['support'], data['catalog']),
                   'top50_unsupported_share': fr.top50_unsupported_share(predictions, data['support']),
                   'item_norm_stats': fr.item_norm_stats(norms, data['catalog'], data['support'])}
    if config['feature'] == 'f4_recency_pool':
        diagnostics['recency_logits'] = {'position': model.recency_logit_pos.detach().cpu().tolist(), 'weekgap': model.recency_logit_gap.detach().cpu().tolist()}
    if config['feature'] == Q3:
        diagnostics['q3_scale_ratio'] = _q3_scale_median(model, queries, data['idmap'], config['batch_size'], width, check)
    return result, diagnostics


# ---------------------------------------------------------------- binding / run

def binding_for3(data, config, rows, queries, model, device, inputs_binding):
    base = b1.binding_for(data, config, rows, queries, model.init_report, device, inputs_binding)
    binding = {**base, 'task': 'P-FEAT03', 'feature': config['feature'], 'side_version': SIDE_VERSION,
               'side_sha256': b1._digest([r['side'] for r in rows] + [q['side'] for q in queries]),
               'bucket_spec_sha256': f3.bucket_spec3_sha256(), 'param_report': f3.parameter_report3(model)}
    if config['feature'] in NEEDS_WEEK:
        binding['weeks_sha256'] = b1._digest([[p[0] for p in r['history_positions']] for r in rows + queries])
    return binding


def _model3(data, config, device):
    from .training import seed_all
    seed_all(config['seed'])
    return f3.make_feat03_model(config['feature'], data['content'], data['supported_indices'], config['history_len'], config['seed']).to(device)


def run_session(output, data, config, device='cpu', check=None, inputs_binding=None):
    from .retrieval_hp import EarlyStop
    from .retrieval_training import EpochCursor, SamplerStats
    from .retrieval_negatives import target_frequency
    synthetic = bool(data.get('synthetic'))
    validate_config(config, synthetic)
    if not synthetic and (device != 'cuda' or inputs_binding is None or len(data['rows']) != 86785):
        raise ValueError('unapproved real runtime')
    b1.execution_context(device)
    rows, queries = fr._view(data, config['history_len'])
    seed = config['seed']; model = _model3(data, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    cursor = EpochCursor(len(rows), seed); stats = SamplerStats(); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    negative_rng = random.Random(f'p-b1-negative-v1:{seed}')
    frequency = target_frequency(rows) if config['logq'] else None
    pool = sorted(data['support']) if config['extra_uniform'] else None
    binding = binding_for3(data, config, rows, queries, model, device, inputs_binding)
    view = {**data, 'rows': rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    step = 0; records = []; evaluations = []; best = None; last = None; train_seconds = 0.; eval_seconds = 0.
    diagnostics = {**b1.data_diagnostics(rows, queries, data['support']), 'side_visibility': f1.side_visibility(queries, data['support'])}
    if config['feature'] == Q3:
        diagnostics['user_activity_buckets'] = user_activity_buckets(rows, queries, data['idmap'])
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
        value = {'schema': MANIFEST_SCHEMA, 'kind': 'p_feat03_train', 'status': 'complete' if stop.stopped else 'running', 'config': config,
                 'binding': binding, 'steps': step, 'records': records, 'evaluations': evaluations, 'selection': stop.state(),
                 'stats': b1._stats_value(stats), 'best': best, 'last': last, 'data_diagnostics': diagnostics,
                 'training_seconds': train_seconds, 'evaluation_seconds': eval_seconds,
                 'hit_max_epochs': stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience}
        sha = b1._write_json_atomic(output / 'manifest.json', value)
        b1._write_progress('complete' if stop.stopped else 'train', epoch, config['max_epochs'])
    return {**value, 'manifest_sha256': sha}


full_evaluations = b1.full_evaluations


def verify_run(path, digest, data, config, device='cpu', check=None, inputs_binding=None):
    from .baselines import metrics
    from .retrieval_hp import EarlyStop
    from .retrieval_training import EpochCursor, SamplerStats, candidate_batch
    from .retrieval_negatives import supported_uniform_columns
    from .retrieval_ablation_run import state_equal
    path = Path(path); value = read_pinned(path, digest); run = path.parent
    if set(value) != b1.MANIFEST_FIELDS or value['schema'] != MANIFEST_SCHEMA or value['kind'] != 'p_feat03_train' or value['status'] != 'complete':
        raise ValueError('complete FEAT-03 manifest required')
    synthetic = bool(data.get('synthetic'))
    if value['config'] != config:
        raise ValueError('config mismatch')
    validate_config(config, synthetic)
    if not synthetic and device != 'cuda':
        raise ValueError('real verifier requires producer CUDA')
    b1.execution_context(device)
    rows, queries = fr._view(data, config['history_len'])
    model = _model3(data, config, device)
    if value['binding'] != binding_for3(data, config, rows, queries, model, device, inputs_binding):
        raise ValueError('run input/source/init/side binding mismatch')
    if config['feature'] == Q3 and value['data_diagnostics'].get('user_activity_buckets') != user_activity_buckets(rows, queries, data['idmap']):
        raise ValueError('user activity buckets not reproducible from the pinned data')
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
        if fr.support_slices(predictions, queries, data['support'], data['catalog']) != evaluation['result'].get('support_metrics'):
            raise ValueError('target support slice metrics not reproducible from stored predictions')
        if fr.top50_unsupported_share(predictions, data['support']) != stored.get('top50_unsupported_share'):
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
        raise ValueError('best checkpoint exact re-export diagnostics mismatch')       # includes the Q2 recency logits and the Q3 scale ratio of the best checkpoint
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
    """p-feat02-inputs-v1 with all four sidecar pins (ctx/attr manifest and sha256) null: no FEAT-03 variant uses a sidecar beyond HIST_V2_SIDE_V1."""
    fr.validate_inputs(value, feature)
    if any(value[key] is not None for pair in fr.PIN_PAIRS for key in pair):
        raise ValueError('FEAT-03 uses no ctx/attr sidecar: all four pins must be null')
    return value


def apply_feat03(data, spec, length):
    """Overlay pinned history-v2 and the FEAT-01 side sidecar (weeks come from history_positions); no ctx/attr loading."""
    return f1.apply_side(data, spec, length)


def build_parser():
    parser = argparse.ArgumentParser(description='FEAT-03 P runner (new version; FEAT-02 and older runners untouched)')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('synthetic-train'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default=DEFAULT_DEVICE)
    p = sub.add_parser('synthetic-verify'); p.add_argument('--config', required=True); p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default=DEFAULT_DEVICE)
    for name in ('train', 'verify'):
        p = sub.add_parser(name); p.add_argument('--inputs', required=True); p.add_argument('--inputs-sha', required=True)
        p.add_argument('--config', required=True); p.add_argument('--config-sha', required=True); p.add_argument('--output'); p.add_argument('--manifest'); p.add_argument('--manifest-sha')
    return parser


def main(argv=None):
    import sys
    args = build_parser().parse_args(argv)
    try:
        if args.command in ('synthetic-train', 'synthetic-verify'):
            if args.device == 'cuda' and not torch.cuda.is_available():
                raise ValueError('actual CUDA environment required')
            config = strict_json(Path(args.config).read_text())
            if args.command == 'synthetic-train':
                value = run_session(args.output, fr.synthetic_data(), config, args.device)
                print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256']})); return 0
            print(json.dumps(verify_run(args.manifest, args.manifest_sha, fr.synthetic_data(), config, args.device))); return 0
        config = read_pinned(args.config, args.config_sha); validate_config(config)
        spec = validate_inputs(read_pinned(args.inputs, args.inputs_sha), config['feature'])
        if not torch.cuda.is_available():
            raise ValueError('actual CUDA environment required')
        run_dir = Path(args.output) if args.command == 'train' else Path(args.manifest).parent
        check = b1.resource_guard(run_dir, spec['disk_cap_bytes']); check()
        data, _, binding = b1.real_inputs(spec)
        data, extra_binding = apply_feat03(data, spec, config['history_len'])
        inputs_binding = {**binding, **extra_binding, 'inputs_sha256': args.inputs_sha}
        if args.command == 'train':
            value = run_session(run_dir, data, config, 'cuda', check, inputs_binding)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']})); return 0
        print(json.dumps(verify_run(args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs_binding))); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR P-FEAT03 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
