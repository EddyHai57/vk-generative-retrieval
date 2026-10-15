"""FEAT-04a P runner: id_mean + the F4-T010 anchor (N1 logQ, F3, F4, temperature 0.10) + one more variable, over HIST_V2_SIDE_V1.

``f4_recency_pool`` (A = Q2-T010) and ``f4_hist_author_content`` (B = F6-hist) are the real variants; ``f4_control`` is the equivalence fixture of the anchor (synthetic runs only, never a real run).
New file only (K1): the FEAT-02/FEAT-03/P-HP-01 runners stay byte-identical. ``train_step / export_vectors_side4a / evaluate_model / binding_for4a /
_model4a / run_session / verify_run`` are copies of the FEAT-03 functions with ``_tensors3 -> _tensors4a`` and ``make_feat03_model -> make_feat04a_model``;
the copy is guarded by three equivalence tests (T0.10 control == P-HP-01 H1, T0.15 A == FEAT-03 Q2, T0.15 control == FEAT-03 control).
The temperature is the only difference of the config contract to FEAT-03: the frozen B1 validator pins 0.15, so (like P-HP-01) it is projected back before
the frozen validator runs; real runs accept 0.1 only. Seams are resolved through the modules (K2): ``fr.<name>``, ``f3.<name>``, ``m4.<name>``.
Config schema ``p-feat04a-run-v1`` has the FEAT-03 field set; only schema, task_id and the ``feature`` values differ.
Inputs: the ``p-feat02-inputs-v1`` spec (validated by ``validate_inputs4a``: a pin that is not used must be null; B needs the attr sidecar pins).
CLI: ``synthetic-train [--device]``, ``synthetic-verify [--device]``, ``train``, ``verify``, ``preflight`` (read-only, CPU, never writes a run directory).
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
from . import retrieval_feat04a as m4
from . import item_attr as ia
from .data_audit import file_sha256
from .history_side import SIDE_VERSION
from .retrieval_ablation_run import read_pinned, strict_json

TASK = 'FEAT-04A'
SCHEMA = 'p-feat04a-run-v1'
CHECKPOINT_SCHEMA = 'p-feat04a-checkpoint-v1'
MANIFEST_SCHEMA = 'p-feat04a-train-manifest-v1'
KIND = 'p_feat04a_train'
DEFAULT_DEVICE = 'cpu'                                                  # read at call time by build_parser (a mutation test patches it)
ANCHOR_TEMPERATURE = 0.1
SYNTHETIC_TEMPERATURES = (0.1, 0.15)
REAL_LEARNING_RATE = 0.001


def default_config():
    return {**b1.default_config(), 'schema': SCHEMA, 'task_id': TASK, 'feature': m4.A_FEATURE, 'side_version': SIDE_VERSION}


def validate_config(value, synthetic=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered FEAT-04A config fields')
    if value['schema'] != SCHEMA or value['task_id'] != TASK or value['side_version'] != SIDE_VERSION or value['feature'] not in m4.FEATURES4A:
        raise ValueError('unregistered FEAT-04A schema/task/feature')
    if type(value['temperature']) is not float or value['temperature'] not in (SYNTHETIC_TEMPERATURES if synthetic else (ANCHOR_TEMPERATURE,)):
        raise ValueError('FEAT-04A temperature must be 0.1 (0.15 is for synthetic equivalence runs only)')
    if not synthetic:
        if value['feature'] not in m4.REAL_FEATURES4A:
            raise ValueError('f4_control is an equivalence fixture for tests, never a real run')
        if type(value['learning_rate']) is not float or value['learning_rate'] != REAL_LEARNING_RATE:
            raise ValueError('FEAT-04A real runs use learning_rate 0.001 only')
        if value['logq'] is not True or value['extra_uniform'] != 0:
            raise ValueError('FEAT-04A keeps the anchor negatives (in-batch logQ alpha=1, no extra uniform)')
    if value['architecture'] != 'id_mean' or value['history_len'] != 50 or value['init'] != 'small':
        raise ValueError('FEAT-04A is id_mean + HIST_V2_OBS_L50 + INIT_EMB_N002 only')
    projected = {k: v for k, v in value.items() if k not in ('feature', 'side_version')}
    b1.validate_config({**projected, 'schema': 'p-b1-run-v1', 'task_id': 'P-NEG-01', 'temperature': b1.default_config()['temperature']}, synthetic)
    return value


# ---------------------------------------------------------------- data

def _tensors4a(batch, idmap, width, feature):
    """Padded history ids and the 6-column side tensor, built from the history events' own fields only (through ``fr._tensors2``).

    Every FEAT-04a variant reads the F4 columns (feedback mask, timespent, duration, history week); place/platform stay 0.
    """
    if feature not in m4.FEATURES4A:
        raise ValueError('unregistered FEAT-04a variant')
    return fr._tensors2(batch, idmap, width, 'f3_f4_recency')


def train_step(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
    from .retrieval_negatives import masked_ce_logq
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, rng, pool)
    b1._histories(batch, data['idmap'], config['history_len'])  # same validation as Batch1
    histories, side = _tensors4a(batch, data['idmap'], max(len(r['history']) for r in batch), config['feature'])
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


def export_vectors_side4a(model, catalog, queries, idmap, batch_size, width, feature, check=None):
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
            histories, side = _tensors4a(queries[start:start + batch_size], idmap, width, feature)
            user_rows.append(model.users(histories.to(device), side.to(device)).cpu().numpy())
    raw_items = np.concatenate(item_rows)
    norms = np.linalg.norm(raw_items.astype(np.float64), axis=1)
    return normalize_export(np.concatenate(user_rows), check), normalize_export(raw_items, check), norms


# ---------------------------------------------------------------- diagnostics

def history_unsupported_visible_fraction(queries, support):
    """B diagnostic: mean over validation queries of (unsupported valid history events / valid history events). Uses the history item ids and the
    train-defined support set only; verify recomputes it from the pinned data."""
    return sum(sum(1 for i in q['history'] if i not in support) / len(q['history']) for q in queries) / len(queries)


def evaluate_model(model, data, queries, config, check=None):
    from .retrieval_experiment import evaluate_vectors
    width = 50 if config['history_len'] <= 50 else config['history_len']
    users, items, norms = export_vectors_side4a(model, data['catalog'], queries, data['idmap'], config['batch_size'], width, config['feature'], check)
    result = evaluate_vectors(users, items, data['catalog'], queries, data['support'], check=check); result.pop('exact_seconds')
    predictions = result['predictions']
    diagnostics = {'zero_user_vectors': result['zero_user_vectors'], 'zero_user_vector_fraction': result['zero_user_vectors'] / len(queries),
                   'top1_is_last_history_fraction': sum(1 for p, q in zip(predictions, queries) if p and p[0] == q['history'][-1]) / len(queries),
                   'history_support_slices': f1.history_support_slices(predictions, queries, data['support'], data['catalog']),
                   'top50_unsupported_share': fr.top50_unsupported_share(predictions, data['support']),
                   'item_norm_stats': fr.item_norm_stats(norms, data['catalog'], data['support'])}
    if config['feature'] == m4.A_FEATURE:
        diagnostics['recency_logits'] = {'position': model.recency_logit_pos.detach().cpu().tolist(), 'weekgap': model.recency_logit_gap.detach().cpu().tolist()}
    if config['feature'] == m4.B_FEATURE:
        diagnostics['history_unsupported_visible_fraction'] = history_unsupported_visible_fraction(queries, data['support'])
    return result, diagnostics


# ---------------------------------------------------------------- binding / run

def binding_for4a(data, config, rows, queries, model, device, inputs_binding):
    base = b1.binding_for(data, config, rows, queries, model.init_report, device, inputs_binding)
    binding = {**base, 'task': 'P-FEAT04A', 'feature': config['feature'], 'temperature': config['temperature'], 'side_version': SIDE_VERSION,
               'side_sha256': b1._digest([r['side'] for r in rows] + [q['side'] for q in queries]),
               'weeks_sha256': b1._digest([[p[0] for p in r['history_positions']] for r in rows + queries]),
               'bucket_spec_sha256': f3.bucket_spec3_sha256(), 'bucket_spec4a_sha256': m4.bucket_spec4a_sha256(config['feature']),
               'param_report': m4.parameter_report4a(model)}
    if config['feature'] == m4.B_FEATURE:
        binding['attr_sha256'] = b1._digest([data['author_index'].tolist(), data['author_vocab']])
        binding['author_vocab_sha256'] = ia.vocab_digest(data['author_vocab'])
    return binding


def _model4a(data, config, device):
    from .training import seed_all
    seed_all(config['seed'])
    if config['feature'] != m4.B_FEATURE:
        return m4.make_feat04a_model(config['feature'], data['content'], data['supported_indices'], config['history_len'], config['seed']).to(device)
    if 'author_index' not in data or 'author_vocab' not in data:
        raise ValueError('B needs the pinned ITEM_ATTR_V1 sidecar (author_index/author_vocab)')
    if not data.get('synthetic') and len(data['author_vocab']) != fr.AUTHOR_VOCAB_SIZE:
        raise ValueError('author vocabulary differs from the hard-coded audit size')
    return m4.make_feat04a_model(config['feature'], data['content'], data['supported_indices'], config['history_len'], config['seed'],
                                 author_index=data['author_index'], vocab_size=len(data['author_vocab'])).to(device)


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
    seed = config['seed']; model = _model4a(data, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    cursor = EpochCursor(len(rows), seed); stats = SamplerStats(); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    negative_rng = random.Random(f'p-b1-negative-v1:{seed}')
    frequency = target_frequency(rows) if config['logq'] else None
    pool = sorted(data['support']) if config['extra_uniform'] else None
    binding = binding_for4a(data, config, rows, queries, model, device, inputs_binding)
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
        value = {'schema': MANIFEST_SCHEMA, 'kind': KIND, 'status': 'complete' if stop.stopped else 'running', 'config': config,
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
    if set(value) != b1.MANIFEST_FIELDS or value['schema'] != MANIFEST_SCHEMA or value['kind'] != KIND or value['status'] != 'complete':
        raise ValueError('complete FEAT-04A manifest required')
    synthetic = bool(data.get('synthetic'))
    if value['config'] != config:
        raise ValueError('config mismatch')
    validate_config(config, synthetic)
    if not synthetic and device != 'cuda':
        raise ValueError('real verifier requires producer CUDA')
    b1.execution_context(device)
    rows, queries = fr._view(data, config['history_len'])
    model = _model4a(data, config, device)
    if value['binding'] != binding_for4a(data, config, rows, queries, model, device, inputs_binding):
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
        if config['feature'] == m4.B_FEATURE and stored.get('history_unsupported_visible_fraction') != history_unsupported_visible_fraction(queries, data['support']):
            raise ValueError('history unsupported visible fraction not reproducible from the pinned data')
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
        raise ValueError('best checkpoint exact re-export diagnostics mismatch')       # includes the recency logits of the best checkpoint
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

def validate_inputs4a(value, feature):
    """p-feat02-inputs-v1 (schema, sha, half-pin checks delegated) plus the rule 'a pin that is used must be set, a pin that is not used must be null'.

    A and the control use no ctx/attr sidecar: all four pins null. B uses the attr sidecar only: ctx pins null, attr pins set.
    """
    if feature not in m4.FEATURES4A:
        raise ValueError('unregistered FEAT-04a variant')
    fr.validate_inputs(value, None)
    if any(value[key] is not None for key in ('ctx_manifest', 'ctx_sha256')):
        raise ValueError('FEAT-04a uses no ctx sidecar: both ctx pins must be null')
    wants_attr = feature == m4.B_FEATURE
    if any((value[key] is not None) != wants_attr for key in ('attr_manifest', 'attr_sha256')):
        raise ValueError('the attr sidecar pins are required by B and must be null for the other variants')
    return value


def apply_feat04a(data, spec, length, feature):
    """Overlay pinned history-v2 and the FEAT-01 side sidecar (weeks come from history_positions). A variant without a sidecar refuses a non-empty pin."""
    validate_inputs4a(spec, feature)
    return fr.apply_feat02(data, spec, length)


def preflight_check(data, config, spec=None, inputs_binding=None, train_sample=512):
    """Read-only checks on REAL inputs (also callable on synthetic data). Runs on CPU, writes nothing, never looks at CUDA.

    For the first ``train_sample`` training examples and every validation query it builds the users of the variant and of the anchor control with the same seed
    and asserts: shapes and finiteness, A's users bit-equal to the control's (logits start at 0), the item tower equal to the control's with unsupported items
    exactly 0, and (F3) flipping target / target_position / oov of every validation query leaves the users bit-equal. Raises ValueError on any failure.
    """
    import copy
    feature = config['feature']
    if feature not in m4.FEATURES4A:
        raise ValueError('unregistered FEAT-04a variant')
    rows, queries = fr._view(data, config['history_len'])
    model = _model4a(data, config, 'cpu'); model.eval()
    control = m4.make_feat04a_model(m4.CONTROL, data['content'], data['supported_indices'], config['history_len'], config['seed']); control.eval()
    catalog_ids = torch.tensor([data['idmap'][i] for i in data['catalog']])
    sample = rows[:train_sample]
    with torch.no_grad():
        items, control_items = model.items(catalog_ids), control.items(catalog_ids)
        if items.shape != (len(catalog_ids), 64) or not bool(torch.isfinite(items).all()):
            raise ValueError('catalog item vectors have the wrong shape or are not finite')
        if not torch.equal(items, control_items):
            raise ValueError('item tower differs from the anchor control')
        unsupported = torch.tensor([i not in data['support'] for i in data['catalog']])
        if bool(items[unsupported].abs().sum() != 0):
            raise ValueError('unsupported catalog items must have exactly zero vectors')
        checked = {}
        for name, batch in (('train', sample), ('validation', queries)):
            users, control_users = [], []
            for start in range(0, len(batch), 256):
                part = batch[start:start + 256]; width = max(len(r['history']) for r in part)
                histories, side = _tensors4a(part, data['idmap'], width, feature)
                users.append(model.users(histories, side)); control_users.append(control.users(histories, side))
            users, control_users = torch.cat(users), torch.cat(control_users)
            if users.shape != (len(batch), 64) or not bool(torch.isfinite(users).all()):
                raise ValueError(f'{name} user vectors have the wrong shape or are not finite')
            if feature == m4.A_FEATURE and not torch.equal(users, control_users):
                raise ValueError(f'{name}: A does not start at the anchor (user vectors differ from the control)')
            if feature == m4.B_FEATURE and name == 'validation' and torch.equal(users, control_users):
                raise ValueError('B user vectors equal the anchor control for every validation query (the history side is inactive)')
            checked[name] = len(batch)
        moved = []
        for q in queries:
            other = data['catalog'][0] if q['target'] != data['catalog'][0] else data['catalog'][1]
            m = copy.deepcopy(q); m['target'] = other; m['target_position'] = [7 if q['target_position'][0] != 7 else 11, 99]; m['oov'] = not q['oov']
            moved.append(m)
        for start in range(0, len(queries), 256):
            part, flipped = queries[start:start + 256], moved[start:start + 256]; width = max(len(r['history']) for r in part)
            a = _tensors4a(part, data['idmap'], width, feature); b = _tensors4a(flipped, data['idmap'], width, feature)
            if not (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(model.users(*a), model.users(*b))):
                raise ValueError('user vectors depend on target / target_position / oov')
    binding = binding_for4a(data, config, rows, queries, model, 'cpu', inputs_binding)
    return {'status': 'preflight_ok', 'feature': feature, 'temperature': config['temperature'], 'seed': config['seed'],
            'catalog_items': len(data['catalog']), 'supported_items': len(data['support']), 'train_examples_checked': checked['train'],
            'queries_checked': checked['validation'], 'binding_sha256': b1._digest(binding), 'param_report': binding['param_report'],
            'sidecar': {k: spec[k] for k in ('side_sha256', 'history_sha256', 'attr_sha256') if spec and spec.get(k) is not None}}


def build_parser():
    parser = argparse.ArgumentParser(description='FEAT-04a P runner (new version; FEAT-02/FEAT-03/P-HP-01 runners untouched)')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('synthetic-train'); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default=DEFAULT_DEVICE)
    p = sub.add_parser('synthetic-verify'); p.add_argument('--config', required=True); p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha', required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default=DEFAULT_DEVICE)
    for name in ('train', 'verify', 'preflight'):
        p = sub.add_parser(name); p.add_argument('--inputs', required=True); p.add_argument('--inputs-sha', required=True)
        p.add_argument('--config', required=True); p.add_argument('--config-sha', required=True)
        if name != 'preflight':
            p.add_argument('--output'); p.add_argument('--manifest'); p.add_argument('--manifest-sha')
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
        spec = validate_inputs4a(read_pinned(args.inputs, args.inputs_sha), config['feature'])
        if args.command == 'preflight':                                   # read-only, CPU: no CUDA check, no resource guard, no run directory
            data, _, binding = b1.real_inputs(spec)
            data, extra_binding = apply_feat04a(data, spec, config['history_len'], config['feature'])
            print(json.dumps(preflight_check(data, config, spec, {**binding, **extra_binding, 'inputs_sha256': args.inputs_sha}))); return 0
        if not torch.cuda.is_available():
            raise ValueError('actual CUDA environment required')
        run_dir = Path(args.output) if args.command == 'train' else Path(args.manifest).parent
        check = b1.resource_guard(run_dir, spec['disk_cap_bytes']); check()
        data, _, binding = b1.real_inputs(spec)
        data, extra_binding = apply_feat04a(data, spec, config['history_len'], config['feature'])
        inputs_binding = {**binding, **extra_binding, 'inputs_sha256': args.inputs_sha}
        if args.command == 'train':
            value = run_session(run_dir, data, config, 'cuda', check, inputs_binding)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']})); return 0
        print(json.dumps(verify_run(args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs_binding))); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR P-FEAT04A gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
