"""Sequence mean + C/D/E stack runner, copied from fixed M1 with local seams only.
Mean M1 config/training/branch stay frozen; E user pooling and supported-only
candidate stats share combo's assembly. Private data contracts are separated
from clean Git source. Synthetic CLI uses only newly generated checkpoints.
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
from . import retrieval_php01_run as php
from .data_audit import file_sha256
from .retrieval_ablation_run import read_pinned, strict_json
from .retrieval_php01 import FEATURE
from .retrieval_seqm1 import branch_spec_sha256
from . import retrieval_seqm1_run as original
from . import retrieval_feat04combo_run as combo
from . import retrieval_feat04combo as cmodel
from . import retrieval_seq_combo_stack as stack
from . import retrieval_feat04b as stats
ARMS = {'mean': ('causal', 'mean', True)}

TASK = 'P-SEQ-COMBO-STACK'
SCHEMA = 'p-seq-combo-stack-run-v1'
CHECKPOINT_SCHEMA = 'p-seq-combo-stack-checkpoint-v1'
MANIFEST_SCHEMA = 'p-seq-combo-stack-train-manifest-v1'
KIND = 'p_seq_combo_stack_train'
MANIFEST_FIELDS = b1.MANIFEST_FIELDS | {'ablation_g0'}
LENGTH_BUCKETS = (('1-4', 1, 4), ('5-19', 5, 19), ('20-39', 20, 39), ('40-49', 40, 49), ('50', 50, 50))

_tensors2 = fr._tensors2
full_evaluations = fr.full_evaluations


def default_config():
    base = {k: v for k, v in php.default_config().items() if k != 'knob'}
    return {**base, 'schema': SCHEMA, 'task_id': TASK, 'temperature': .1, 'arm': 'mean'}


def validate_config(value, synthetic=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered P-SEQ-COMBO-STACK config fields')
    if value['schema'] != SCHEMA or value['task_id'] != TASK:
        raise ValueError('unregistered P-SEQ-COMBO-STACK schema/task')
    if type(value['arm']) is not str or value['arm'] not in ARMS:
        raise ValueError('unregistered P-SEQ-COMBO-STACK arm')
    projected = {**{k: v for k, v in value.items() if k != 'arm'}, 'schema': php.SCHEMA, 'task_id': php.TASK, 'knob': 'H1'}
    php.validate_config(projected, synthetic)            # freezes temperature 0.10, dim 64, lr 1e-3, logQ alpha=1, L50, budget, seeds, test_gate
    return value


def make_optimizer(model, config):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


def _model(data, config, device):
    from .training import seed_all
    seed_all(config['seed'])
    return stack.make_model(data['content'], data['supported_indices'], config['history_len'], data['stat_buckets']).to(device)


# ---------------------------------------------------------------- diagnostics

def _to_eval(model):
    model.eval()


def recall_slice(predictions, queries, catalog):
    from .baselines import metrics
    if not queries:
        return None
    return metrics(predictions, queries, catalog, (50,))['all']['recall@50']


def recall_by_history_length(predictions, queries, catalog):
    out = {}
    for name, low, high in LENGTH_BUCKETS:
        pos = [j for j, q in enumerate(queries) if low <= len(q['history']) <= high]
        out[name] = recall_slice([predictions[j] for j in pos], [queries[j] for j in pos], catalog)
    return out


def recall_by_last_event_support(predictions, queries, catalog, support):
    out = {}
    for name, flag in (('supported', True), ('unsupported', False)):
        pos = [j for j, q in enumerate(queries) if (q['history'][-1] in support) == flag]
        out[name] = recall_slice([predictions[j] for j in pos], [queries[j] for j in pos], catalog)
    return out


def _mean(values):
    return float(values.mean()) if values is not None else None


def model_diagnostics(model, data, queries, predictions, config, check=None):
    """Branch diagnostics on the validation queries (eval mode, no_grad) + recall slices recomputed from the stored predictions."""
    import torch
    _to_eval(model); device = next(model.parameters()).device
    collected = {}
    with torch.no_grad():
        for start in range(0, len(queries), config['batch_size']):
            if check: check()
            histories, side = _tensors2(queries[start:start + config['batch_size']], data['idmap'], 50, config['feature'])
            for key, value in model.diagnose(histories.to(device), side.to(device)).items():
                collected.setdefault(key, []).append(value.cpu())
    stacked = {k: torch.cat(v) for k, v in collected.items()}
    ratio = stacked.get('branch_ratio')
    return {'gate': float(model.gate.detach()) if model.anchored else None,
            'branch_ratio_median': float(torch.median(ratio)) if ratio is not None else None, 'branch_ratio_mean': _mean(ratio),
            'cos_u_m_mean': _mean(stacked['cos_u_m']), 'cos_u_last_event_mean': _mean(stacked['cos_u_last_event']),
            'attn_entropy_ratio': _mean(stacked.get('attn_entropy_ratio')), 'attn_last_weight': _mean(stacked.get('attn_last_weight')),
            'recall50_by_history_length': recall_by_history_length(predictions, queries, data['catalog']),
            'recall50_by_last_event_support': recall_by_last_event_support(predictions, queries, data['catalog'], data['support'])}


def evaluate_model(model, data, queries, config, check=None):
    result, diagnostics = combo.evaluate_model(model, data, queries, {**config, 'feature': cmodel.E_FEATURE}, check)
    return result, {**diagnostics, **model_diagnostics(model, data, queries, result['predictions'], config, check)}

def ablation_g0(model, data, queries, config, best_epoch, best_recall, check=None):
    """Best-checkpoint ablation (design review F4): R@50 with the gate forced to 0, i.e. the anchor tower with the trained tables."""
    import torch
    if not model.anchored:
        return None
    _to_eval(model)
    with torch.no_grad():
        saved = model.gate.detach().clone(); model.gate.zero_()
    try:
        result, _ = evaluate_model(model, data, queries, config, check)
    finally:
        with torch.no_grad():
            model.gate.copy_(saved)
    metric = result['metrics']['all']
    return {'epoch': best_epoch, 'recall@50': metric['recall@50'], 'ndcg@50': metric['ndcg@50'], 'best_recall@50': best_recall,
            'delta_recall@50': best_recall - metric['recall@50']}


# ---------------------------------------------------------------- binding / run

def binding_for4(data, config, rows, queries, model, device, inputs_binding):
    base = b1.binding_for(data, config, rows, queries, model.init_report, device, inputs_binding)
    return {**base, 'task': TASK, 'arm': 'mean', 'mask': model.mask_kind, 'readout': model.readout, 'anchored': model.anchored,
            'side_version': config['side_version'], 'side_sha256': b1._digest([r['side'] for r in rows + queries]),
            'weeks_sha256': b1._digest([[p[0] for p in r['history_positions']] for r in rows + queries]),
            'branch_spec_sha256': branch_spec_sha256(model), 'param_report': stack.parameter_report(model),
            'bucket_spec_sha256': cmodel.bucket_spec_sha256(cmodel.E_FEATURE),
            'stats_sha256': data['stats_sha256'], 'stat_cutoff_week': data['stat_cutoff_week'],
            'stat_bucket_ids_sha256': stats.stat_bucket_ids_sha256(model)}

def run_session(output, data, config, device='cpu', check=None, inputs_binding=None):
    import torch
    from .retrieval_hp import EarlyStop
    from .retrieval_training import EpochCursor, SamplerStats
    from .retrieval_negatives import target_frequency
    synthetic = bool(data.get('synthetic'))
    validate_config(config, synthetic)
    if not synthetic and (device != 'cuda' or inputs_binding is None or len(data['rows']) != 86785 or data.get('stats_sha256') != cmodel.STATS_SHA256 or data.get('stat_cutoff_week') != 24):
        raise ValueError('unapproved real runtime')
    b1.execution_context(device)
    rows, queries = fr._view(data, config['history_len'])
    seed = config['seed']; model = _model(data, config, device)
    optimizer = make_optimizer(model, config)
    cursor = EpochCursor(len(rows), seed); stats = SamplerStats(); stop = EarlyStop(config['max_epochs'], config['min_epochs'], config['patience'])
    negative_rng = random.Random(f'p-b1-negative-v1:{seed}')
    frequency = target_frequency(rows) if config['logq'] else None
    pool = sorted(data['support']) if config['extra_uniform'] else None
    binding = binding_for4(data, config, rows, queries, model, device, inputs_binding)
    view = {**data, 'rows': rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    step = 0; records = []; evaluations = []; best = None; last = None; train_seconds = 0.; eval_seconds = 0.
    diagnostics = {**b1.data_diagnostics(rows, queries, data['support']), 'side_visibility': f1.side_visibility(queries, data['support']), **feature_data_diagnostics(data, rows, queries)}
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
        value = {'schema': MANIFEST_SCHEMA, 'kind': KIND, 'status': 'running', 'config': config,
                 'binding': binding, 'steps': step, 'records': records, 'evaluations': evaluations, 'selection': stop.state(),
                 'stats': b1._stats_value(stats), 'best': best, 'last': last, 'data_diagnostics': diagnostics,
                 'training_seconds': train_seconds, 'evaluation_seconds': eval_seconds,
                 'hit_max_epochs': stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience, 'ablation_g0': None}
        if not stop.stopped:
            b1._write_json_atomic(output / 'manifest.json', value)
            b1._write_progress('train', epoch, config['max_epochs'])
    # finished: best-checkpoint g -> 0 ablation, then the single 'complete' manifest write
    ablation = None
    if model.anchored:
        model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=True)['model'])
        best_epoch = stop.best_epoch
        ablation = ablation_g0(model, data, queries, config, best_epoch, evaluations[best_epoch - 1]['result']['metrics']['all']['recall@50'], check)
    value = {**value, 'status': 'complete', 'ablation_g0': ablation}
    sha = b1._write_json_atomic(output / 'manifest.json', value)
    b1._write_progress('complete', epoch, config['max_epochs'])
    return {**value, 'manifest_sha256': sha}


def verify_run(path, digest, data, config, device='cpu', check=None, inputs_binding=None):
    import torch
    from .baselines import metrics
    from .retrieval_hp import EarlyStop
    from .retrieval_training import EpochCursor, SamplerStats, candidate_batch
    from .retrieval_negatives import supported_uniform_columns
    from .retrieval_ablation_run import state_equal
    path = Path(path); value = read_pinned(path, digest); run = path.parent
    if set(value) != MANIFEST_FIELDS or value['schema'] != MANIFEST_SCHEMA or value['kind'] != KIND or value['status'] != 'complete':
        raise ValueError('complete P-SEQ-COMBO-STACK manifest required')
    synthetic = bool(data.get('synthetic'))
    if value['config'] != config:
        raise ValueError('config mismatch')
    validate_config(config, synthetic)
    if not synthetic and device != 'cuda':
        raise ValueError('real verifier requires producer CUDA')
    b1.execution_context(device)
    rows, queries = fr._view(data, config['history_len'])
    model = _model(data, config, device)
    if value['binding'] != binding_for4(data, config, rows, queries, model, device, inputs_binding):
        raise ValueError('run input/source/init/side binding mismatch')
    for key, expected in feature_data_diagnostics(data, rows, queries).items():
        if value['data_diagnostics'].get(key) != expected:
            raise ValueError('stack static diagnostics mismatch: ' + key)
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
        if recall_by_history_length(predictions, queries, data['catalog']) != stored.get('recall50_by_history_length') \
                or recall_by_last_event_support(predictions, queries, data['catalog'], data['support']) != stored.get('recall50_by_last_event_support'):
            raise ValueError('recall slices by history length / last-event support not reproducible from stored predictions')
        primary = recomputed['all']
        if stop.observe(epoch, primary['recall@50'], primary['ndcg@50']):
            expected_best = (epoch, step)
    if not stop.stopped or stop.state() != value['selection']:
        raise ValueError('selection replay mismatch')
    best = value['best']
    if type(best) is not dict or (best.get('epoch'), best.get('step')) != expected_best or best.get('metric') != list(stop.best_metric) or best.get('path') != 'best.pt':
        raise ValueError('best checkpoint selection mismatch')
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True) if file_sha256(run / 'best.pt') == best['sha256'] else None
    if state is None or state.get('schema') != CHECKPOINT_SCHEMA or state.get('binding_sha256') != b1._digest(value['binding']) or state['step'] != best['step'] or state['epoch'] != best['epoch']:
        raise ValueError('best checkpoint hash/identity mismatch')
    model.load_state_dict(state['model'])
    result, diagnostics = evaluate_model(model, data, queries, config, check)
    if result != full[best['epoch'] - 1]['result']:
        raise ValueError('best checkpoint exact re-export mismatch')
    if diagnostics != value['evaluations'][best['epoch'] - 1].get('diagnostics'):
        raise ValueError('best checkpoint exact re-export diagnostics mismatch')
    if ablation_g0(model, data, queries, config, best['epoch'], recomputed_best_recall(value, best), check) != value['ablation_g0']:
        raise ValueError('best checkpoint g->0 ablation not reproducible')
    last = value['last']
    if type(last) is not dict or last.get('path') != 'last.pt' or last.get('step') != value['steps'] or file_sha256(run / 'last.pt') != last.get('sha256'):
        raise ValueError('last checkpoint hash mismatch')
    state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
    if state.get('schema') != CHECKPOINT_SCHEMA or state.get('binding_sha256') != b1._digest(value['binding']) or state['step'] != value['steps'] or not state_equal(state['cursor'], cursor.state_dict()) or not state_equal(state['negative_rng'], rng.getstate()) or state['stats'] != value['stats'] or state['selection'] != value['selection']:
        raise ValueError('last state mismatch')
    if sorted(p.name for p in run.glob('*.pt')) != ['best.pt', 'last.pt']:
        raise ValueError('checkpoint policy best+last violated')
    return {'status': 'verified', 'replayed_epochs': [best['epoch']], 'best_epoch': best['epoch'], 'steps': value['steps']}


def recomputed_best_recall(value, best):
    return value['evaluations'][best['epoch'] - 1]['result']['metrics']['all']['recall@50']


def synthetic_data(seed=11):
    return combo.synthetic_data4b(seed)


def train_step(model, optimizer, batch, data, sampler, epoch, config, rng, pool, frequency):
    return combo.train_step(model, optimizer, batch, data, sampler, epoch, {**config, 'feature': cmodel.E_FEATURE}, rng, pool, frequency)


def feature_data_diagnostics(data, rows, queries):
    return combo.feature_data_diagnostics(data, rows, queries, cmodel.E_FEATURE)


def validate_inputs(value):
    return combo.validate_inputs4b(value, cmodel.E_FEATURE)


DATA_CONTRACT_SHA = 'c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'


def real_inputs(spec, data_root=None):
    """Separate private data contract root from clean Git source; no old ROOT mutation."""
    import os
    from .retrieval_experiment import load_training_inputs
    from .retrieval import PREPARED_SOURCE
    from .retrieval_data import validate_contract
    value = data_root if data_root is not None else os.environ.get('VKGR_DATA_ROOT')
    if value is None or not Path(value).is_absolute():
        raise ValueError('explicit absolute private data root required')
    root = Path(value)
    document = root / 'docs/p3c_validation.md'
    if document.is_symlink() or file_sha256(document) != DATA_CONTRACT_SHA:
        raise ValueError('private data contract document mismatch')
    pins = validate_contract(root)
    query_pin = next(p for p in pins['full_query_pins'] if p['role'] == 'validation-queries.json')
    data = load_training_inputs(spec['prepared'], spec['holdout'], b1.PREPARED_SHA, PREPARED_SOURCE, query_pin['sha256'])
    data['synthetic'] = False
    binding = {'prepared_sha': b1.PREPARED_SHA, 'query_sha': query_pin['sha256'], 'catalog_sha': pins['catalog_sha256'],
               'test_gate': 'closed_hash_only', 'data_root': str(root), 'data_contract_sha256': DATA_CONTRACT_SHA,
               'data_contract_manifest_sha256': file_sha256(root / 'artifacts/p3d-retrieval-contract-manifest.json')}
    return data, None, binding


def private_data_parameters(path, digest):
    value = read_pinned(path, digest)
    root = value.get('data_root')
    if type(root) is not str or not Path(root).is_absolute():
        raise ValueError('private parameters require absolute data_root')
    return root


def load_real(spec, config, inputs_sha, data_root=None, parameters_sha=None):
    data, _, binding = real_inputs(spec, data_root)
    data, extra = combo.apply_feat04b(data, spec, config['history_len'], cmodel.E_FEATURE)
    return data, {**binding, **extra, 'inputs_sha256': inputs_sha, 'private_parameters_sha256': parameters_sha}


def preflight_check(data, config, inputs_binding=None, engineering=False, config_sha256=None):
    import copy
    import torch
    validate_config(config, synthetic=engineering)
    if type(engineering) is not bool or bool(data.get('synthetic')) != engineering:
        raise ValueError('engineering flag must match data identity')
    if not engineering and (inputs_binding is None or len(data['rows']) != 86785 or data['stats_sha256'] != cmodel.STATS_SHA256):
        raise ValueError('real preflight requires frozen real data and inputs binding')
    if not engineering and not f1._sha(config_sha256):
        raise ValueError('real preflight requires config file SHA')
    if data.get('stat_cutoff_week') != 24:
        raise ValueError('stat cutoff mismatch')
    rows, queries = fr._view(data, 50)
    model = _model(data, config, 'cpu').eval()
    control = original._model(data, {**config, 'schema': original.SCHEMA, 'task_id': original.TASK}, 'cpu').eval()
    for key, value in control.state_dict().items():
        if not torch.equal(value, model.state_dict()[key]):
            raise ValueError('M1 shared init mismatch: ' + key)
    checked = {}
    with torch.no_grad():
        for name, batch in (('train', rows[:512]), ('validation', queries)):
            for start in range(0, len(batch), 128):
                part = batch[start:start + 128]
                h, side = _tensors2(part, data['idmap'], 50, config['feature'])
                if not torch.equal(model.users(h, side), control.users(h, side)):
                    raise ValueError('zero-init users differ from M1 mean')
            checked[name] = len(batch)
        # Exercise both the branch and the new nonzero E weighting, not only g=beta=0.
        model.gate.fill_(.5); model.behavior_beta.copy_(torch.arange(6) * .1)
        model.recency_logit_pos.copy_(torch.arange(7) * .1); model.recency_logit_gap.copy_(torch.arange(6) * .1)
        for batch in (rows[:512], queries):
            for start in range(0, len(batch), 128):
                part = batch[start:start+128]; flipped = copy.deepcopy(part)
                for q in flipped:
                    q.update(target=data['catalog'][0], target_position=[30, 999], past_likes=999, oov=not q.get('oov', False))
                a = _tensors2(part, data['idmap'], 50, config['feature']); b = _tensors2(flipped, data['idmap'], 50, config['feature'])
                if not torch.equal(a[0], b[0]) or not torch.equal(a[1], b[1]) or not torch.equal(model.users(*a), model.users(*b)):
                    raise ValueError('target reaches stack users')
        ids = torch.tensor([data['idmap'][i] for i in data['catalog']])
        candidates = model.candidate_items(ids)
        if not torch.isfinite(candidates).all() or torch.any(candidates[~model.supported[ids]]):
            raise ValueError('unsupported or nonfinite candidates')
    # Binding is always derived from a fresh, untrained initial model.
    initial = _model(data, config, 'cpu')
    binding = binding_for4(data, config, rows, queries, initial, 'cpu', inputs_binding)
    from .seq_combo_stack_leakage import audit_runtime
    if engineering:
        leakage = {'status':'NOT_REAL_INPUT','engineering':True}
    else:
        leakage = audit_runtime(data, __import__(__name__, fromlist=['']), config)
        cuda_binding = binding_for4(data, config, rows, queries, initial, 'cuda', inputs_binding)
        leakage.update(inputs_sha256=inputs_binding['inputs_sha256'], input_sidecar_digests={k:v for k,v in inputs_binding.items() if 'sha' in k and type(v) is str},
                       source=b1.source_identity(), config_sha256=config_sha256, config_value_sha256=b1._digest(config), binding_sha256=b1._digest(cuda_binding),
                       data_hashes={k:cuda_binding[k] for k in ('rows_sha256','queries_sha256','catalog_sha256','support_sha256','content_sha256','side_sha256','weeks_sha256','stats_sha256','stat_bucket_ids_sha256')})
    return {'status': 'preflight_ok' if engineering or leakage['status']=='pass' else 'preflight_failed', 'engineering': engineering, 'seed': config['seed'],
            'train_examples_checked': checked['train'], 'queries_checked': checked['validation'],
            'binding_sha256': b1._digest(binding), 'stat_cutoff_week': 24, 'leakage_receipt':leakage}



def main(argv=None):
    import sys
    import torch
    parser = argparse.ArgumentParser(description='Sequence mean plus C/D/E stack (new entrypoint)')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('synthetic-train', 'synthetic-verify', 'train', 'verify', 'preflight'):
        p = sub.add_parser(name); p.add_argument('--config', required=True)
        p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
        if name.startswith('synthetic'):
            if name == 'synthetic-train': p.add_argument('--output', required=True)
            else: p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha', required=True)
        else:
            p.add_argument('--config-sha', required=True); p.add_argument('--inputs', required=True); p.add_argument('--inputs-sha', required=True); p.add_argument('--data-parameters', required=True); p.add_argument('--data-parameters-sha', required=True)
            if name == 'train': p.add_argument('--output', required=True)
            elif name == 'verify': p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command.startswith('synthetic'):
            config = strict_json(Path(args.config).read_text()); data = synthetic_data()
            if args.command == 'synthetic-train':
                value = run_session(args.output, data, config, args.device)
                answer = {'status': value['status'], 'manifest_sha256': value['manifest_sha256']}
            else: answer = verify_run(args.manifest, args.manifest_sha, data, config, args.device)
        else:
            config = read_pinned(args.config, args.config_sha); validate_config(config)
            spec = validate_inputs(read_pinned(args.inputs, args.inputs_sha))
            if args.command != 'preflight' and not torch.cuda.is_available():
                raise ValueError('actual CUDA required')
            data_root = private_data_parameters(args.data_parameters, args.data_parameters_sha)
            data, inputs = load_real(spec, config, args.inputs_sha, data_root, args.data_parameters_sha)
            if args.command == 'preflight': answer = preflight_check(data, config, inputs, engineering=False, config_sha256=args.config_sha)
            else:
                target = Path(args.output) if args.command == 'train' else Path(args.manifest).parent
                check = b1.resource_guard(target, spec['disk_cap_bytes']); check()
                if args.command == 'train':
                    value = run_session(target, data, config, 'cuda', check, inputs)
                    answer = {'status': value['status'], 'manifest_sha256': value['manifest_sha256']}
                else: answer = verify_run(args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs)
        print(json.dumps(answer, sort_keys=True, allow_nan=False)); return 2 if answer.get('status')=='preflight_failed' else 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR stack gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
