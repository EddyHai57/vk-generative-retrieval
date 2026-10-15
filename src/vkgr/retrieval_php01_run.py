"""P-HP-01 P runner: bounded single-knob tuning on the FEAT-02 F4 anchor (id_mean + N1 logQ + F3 + F4 recency), new file only.

Config schema ``p-php01-run-v1`` = the F4 config fields + ``dim`` (int) + ``knob`` (str). Exactly one of temperature /
dim / learning_rate differs from the anchor (0.15 / 64 / 0.001) at a pre-registered level (H1-H6); H7 is the combo
written by ``retrieval_php01_select``. Every other field is checked by the frozen FEAT-02 validator (temperature and
learning rate are projected back to the anchor values first, because the frozen schema pins them).
Batch tensors, train step, evaluation and diagnostics are the FEAT-02 functions themselves; ``run_session`` and
``verify_run`` are copies of the FEAT-02 ones with only schema/kind, model, optimizer, step and binding changed
(the anchor knob reproduces the FEAT-02 F4 runner bit for bit; tests/test_php01_run.py).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

from . import retrieval_b1_run as b1
from . import retrieval_feat01_run as f1
from . import retrieval_feat02_run as fr
from .data_audit import file_sha256
from .retrieval_ablation_run import read_pinned, strict_json
from .retrieval_php01 import FEATURE, make_php01_model, parameter_report_d

TASK = 'P-HP-01'
SCHEMA = 'p-php01-run-v1'
CHECKPOINT_SCHEMA = 'p-php01-checkpoint-v1'
MANIFEST_SCHEMA = 'p-php01-train-manifest-v1'
KIND = 'p_php01_train'
COORDINATES = ('temperature', 'dim', 'learning_rate')
ANCHOR_VALUES = {'temperature': 0.15, 'dim': 64, 'learning_rate': 0.001}
KNOB_FIELD = {'H1': 'temperature', 'H2': 'temperature', 'H3': 'dim', 'H4': 'dim', 'H5': 'learning_rate', 'H6': 'learning_rate'}
KNOBS = {'anchor': dict(ANCHOR_VALUES),
         'H1': {**ANCHOR_VALUES, 'temperature': 0.1}, 'H2': {**ANCHOR_VALUES, 'temperature': 0.25},
         'H3': {**ANCHOR_VALUES, 'dim': 32}, 'H4': {**ANCHOR_VALUES, 'dim': 128},
         'H5': {**ANCHOR_VALUES, 'learning_rate': 0.002}, 'H6': {**ANCHOR_VALUES, 'learning_rate': 0.0005}}
H7_LEVELS = {c: tuple(KNOBS[k][c] for k, f in KNOB_FIELD.items() if f == c) for c in COORDINATES}
COORD_TYPES = {'temperature': float, 'dim': int, 'learning_rate': float}

_tensors2 = fr._tensors2
evaluate_model = fr.evaluate_model
full_evaluations = fr.full_evaluations


def default_config():
    return {**fr.default_config(), 'schema': SCHEMA, 'task_id': TASK, 'feature': FEATURE, 'logq': True, 'dim': 64, 'knob': 'anchor'}


def knob_values(knob):
    if knob not in KNOBS:
        raise ValueError('unregistered P-HP-01 knob')
    return dict(KNOBS[knob])


def _h7_ok(values):
    changed = [c for c in COORDINATES if values[c] != ANCHOR_VALUES[c]]
    return len(changed) >= 2 and all(values[c] in H7_LEVELS[c] for c in changed)


def validate_config(value, synthetic=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered P-HP-01 config fields')
    if value['schema'] != SCHEMA or value['task_id'] != TASK or value['feature'] != FEATURE:
        raise ValueError('unregistered P-HP-01 schema/task/feature')
    if value['logq'] is not True or value['extra_uniform'] != 0:
        raise ValueError('P-HP-01 keeps the F4 negatives (in-batch logQ alpha=1, no extra uniform)')
    if any(type(value[c]) is not COORD_TYPES[c] for c in COORDINATES):
        raise ValueError('P-HP-01 knob coordinates must be float temperature / int dim / float learning_rate')
    projected = {k: v for k, v in value.items() if k not in ('dim', 'knob')}
    fr.validate_config({**projected, 'schema': fr.SCHEMA, 'task_id': fr.TASK, **{c: ANCHOR_VALUES[c] for c in ('temperature', 'learning_rate')}}, synthetic)
    values = {c: value[c] for c in COORDINATES}
    if value['knob'] == 'H7':
        if not _h7_ok(values):
            raise ValueError('H7 must combine pre-registered levels of at least two knobs')
    elif values != knob_values(value['knob']):
        raise ValueError('P-HP-01 knob values differ from the registry')
    if value['knob'] == 'anchor' and not synthetic:
        raise ValueError('the anchor knob is an equivalence control only (the F4 seed42 run already exists)')
    return value


def make_optimizer(model, config):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


def _step(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
    """The FEAT-02 train step (temperature read from ``config``)."""
    return fr.train_step(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency)


def _model(data, config, device):
    from .training import seed_all
    seed_all(config['seed'])
    return make_php01_model(data['content'], data['supported_indices'], config['history_len'], config['dim']).to(device)


def binding_for3(data, config, rows, queries, model, device, inputs_binding):
    base = fr.binding_for2(data, config, rows, queries, model, device, inputs_binding)
    return {**base, 'task': TASK, 'knob': config['knob'], 'dim': config['dim'], 'param_report': parameter_report_d(model)}


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
    rows, queries = fr._view(data, config['history_len'])
    seed = config['seed']; model = _model(data, config, device)
    optimizer = make_optimizer(model, config)
    cursor = EpochCursor(len(rows), seed); stats = SamplerStats(); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    negative_rng = random.Random(f'p-b1-negative-v1:{seed}')
    frequency = target_frequency(rows) if config['logq'] else None
    pool = sorted(data['support']) if config['extra_uniform'] else None
    binding = binding_for3(data, config, rows, queries, model, device, inputs_binding)
    view = {**data, 'rows': rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    step = 0; records = []; evaluations = []; best = None; last = None; train_seconds = 0.; eval_seconds = 0.
    diagnostics = {**b1.data_diagnostics(rows, queries, data['support']), 'side_visibility': f1.side_visibility(queries, data['support'])}
    b1._write_progress('train', 0, config['max_epochs'])
    while not stop.stopped:
        if check: check()
        begin = time.monotonic()
        indices = cursor.next(config['batch_size'])
        record = _step(model, optimizer, [rows[i] for i in indices], view, stats, cursor.epoch, config, negative_rng, pool, frequency)
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
        value = {'schema': MANIFEST_SCHEMA, 'kind': KIND, 'status': 'complete' if stop.stopped else 'running', 'config': config,
                 'binding': binding, 'steps': step, 'records': records, 'evaluations': evaluations, 'selection': stop.state(),
                 'stats': b1._stats_value(stats), 'best': best, 'last': last, 'data_diagnostics': diagnostics,
                 'training_seconds': train_seconds, 'evaluation_seconds': eval_seconds,
                 'hit_max_epochs': stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience}
        sha = b1._write_json_atomic(output / 'manifest.json', value)
        b1._write_progress('complete' if stop.stopped else 'train', epoch, config['max_epochs'])
    return {**value, 'manifest_sha256': sha}


def verify_run(path, digest, data, config, device='cpu', check=None, inputs_binding=None):
    import torch
    from .baselines import metrics
    from .retrieval_hp import EarlyStop
    from .retrieval_training import EpochCursor, SamplerStats, candidate_batch
    from .retrieval_negatives import supported_uniform_columns
    from .retrieval_ablation_run import state_equal
    path = Path(path); value = read_pinned(path, digest); run = path.parent
    if set(value) != b1.MANIFEST_FIELDS or value['schema'] != MANIFEST_SCHEMA or value['kind'] != KIND or value['status'] != 'complete':
        raise ValueError('complete P-HP-01 manifest required')
    synthetic = bool(data.get('synthetic'))
    if value['config'] != config:
        raise ValueError('config mismatch')
    validate_config(config, synthetic)
    if not synthetic and device != 'cuda':
        raise ValueError('real verifier requires producer CUDA')
    b1.execution_context(device)
    rows, queries = fr._view(data, config['history_len'])
    model = _model(data, config, device)
    if value['binding'] != binding_for3(data, config, rows, queries, model, device, inputs_binding):
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


def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description='P-HP-01 P runner (new version; FEAT-02 runner untouched)')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('synthetic-train'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    for name in ('train', 'verify'):
        p = sub.add_parser(name); p.add_argument('--inputs', required=True); p.add_argument('--inputs-sha', required=True)
        p.add_argument('--config', required=True); p.add_argument('--config-sha', required=True); p.add_argument('--output'); p.add_argument('--manifest'); p.add_argument('--manifest-sha')
    args = parser.parse_args(argv)
    try:
        if args.command == 'synthetic-train':
            value = run_session(args.output, fr.synthetic_data(), strict_json(Path(args.config).read_text()))
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256']})); return 0
        config = read_pinned(args.config, args.config_sha); validate_config(config)
        spec = fr.validate_inputs(read_pinned(args.inputs, args.inputs_sha), FEATURE)
        import torch
        if not torch.cuda.is_available():
            raise ValueError('actual CUDA environment required')
        target = Path(args.output) if args.command == 'train' else Path(args.manifest).parent
        check = b1.resource_guard(target, spec['disk_cap_bytes']); check()
        data, _, binding = b1.real_inputs(spec)
        data, extra_binding = fr.apply_feat02(data, spec, config['history_len'])
        inputs_binding = {**binding, **extra_binding, 'inputs_sha256': args.inputs_sha}
        if args.command == 'train':
            value = run_session(target, data, config, 'cuda', check, inputs_binding)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']})); return 0
        print(json.dumps(verify_run(args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs_binding))); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR P-HP-01 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
