"""M2 fixed K4 causal runner; M1 frozen runner remains untouched. Test closed_hash_only."""
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
from .retrieval_seqm2 import branch_spec_sha256, make_seqm2_model, parameter_report_seqm2

TASK = 'P-SEQ-M2'
SCHEMA = 'p-seqm2-run-v1'
CHECKPOINT_SCHEMA = 'p-seqm2-checkpoint-v1'
MANIFEST_SCHEMA = 'p-seqm2-train-manifest-v1'
KIND = 'p_seqm2_train'
MANIFEST_FIELDS = b1.MANIFEST_FIELDS | {'ablation_g0'}
LENGTH_BUCKETS = (('1-4', 1, 4), ('5-19', 5, 19), ('20-39', 20, 39), ('40-49', 40, 49), ('50', 50, 50))

_tensors2 = fr._tensors2
full_evaluations = fr.full_evaluations


def default_config():
    base = {k: v for k, v in php.default_config().items() if k != 'knob'}
    return {**base, 'schema': SCHEMA, 'task_id': TASK, 'temperature': .1, 'arm': 'k4-causal'}


def validate_config(value, synthetic=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered P-SEQ-M2 config fields')
    if value['schema'] != SCHEMA or value['task_id'] != TASK:
        raise ValueError('unregistered P-SEQ-M2 schema/task')
    if type(value['arm']) is not str or value['arm'] != 'k4-causal':
        raise ValueError('unregistered P-SEQ-M2 arm')
    projected = {**{k: v for k, v in value.items() if k != 'arm'}, 'schema': php.SCHEMA, 'task_id': php.TASK, 'knob': 'H1'}
    php.validate_config(projected, synthetic)            # freezes temperature 0.10, dim 64, lr 1e-3, logQ alpha=1, L50, budget, seeds, test_gate
    return value


def make_optimizer(model, config):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


def _model(data, config, device):
    from .training import seed_all
    seed_all(config['seed'])
    return make_seqm2_model(data['content'], data['supported_indices'], config['history_len']).to(device)


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


def multi_interest_loss(users, items, labels, mask, temperature, log_q, optimizer_step):
    """Per-sample N1 per-head CE, target assignment detached, skip counted once."""
    import torch
    import torch.nn.functional as F
    if (users.ndim != 3 or users.shape[1] != 4 or items.ndim != 2 or users.shape[2] != items.shape[1]
            or labels.shape != (len(users),) or mask.shape != (len(users), len(items))
            or mask.dtype != torch.bool or labels.dtype != torch.long
            or type(optimizer_step) is not int or optimizer_step < 1
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('invalid multi-interest loss input')
    if not torch.isfinite(users).all() or not torch.isfinite(items).all():
        raise ValueError('nonfinite user/item vectors')
    if log_q is not None and (log_q.shape != (len(items),) or not torch.isfinite(log_q).all()):
        raise ValueError('invalid logQ')
    if not mask[torch.arange(len(users), device=users.device), labels].all():
        raise ValueError('masked target')
    valid = mask.sum(1) > 1
    skips = int((~valid).sum())
    u = F.normalize(users.float(), dim=-1); v = F.normalize(items.float(), dim=-1)
    similarities = (u * v[labels, None, :]).sum(-1)
    weights = (similarities / .10).detach().softmax(1)
    selected = similarities.detach().argmax(1)  # first index wins ties
    counts = torch.bincount(selected[valid], minlength=4).tolist()
    diag = {'optimizer_step': optimizer_step, 'phase': 'soft' if optimizer_step <= 100 else 'hard',
            'assignment_counts': counts, 'soft_weight_sum': weights[valid].sum(0).tolist(),
            'soft_assignment_entropy_sum': float((-(weights[valid] * weights[valid].clamp_min(1e-30).log()).sum()).detach())}
    if not bool(valid.any()):
        return None, skips, diag
    scores = torch.einsum('bkd,cd->bkc', u, v) / temperature
    if log_q is not None:
        scores = scores - log_q.to(scores.device)[None, None, :]
    # -inf is intentional exclusion, never included in the numerical finite guard.
    logits = scores.masked_fill(~mask[:, None, :], -torch.inf)[valid]
    target = labels[valid, None].expand(-1, 4)
    losses = F.cross_entropy(logits.reshape(-1,len(items)), target.reshape(-1), reduction='none').reshape(-1,4)
    if optimizer_step <= 100:
        loss = (weights[valid] * losses).sum(1).mean()
    else:
        loss = losses.gather(1, selected[valid,None]).mean()
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss')
    return loss, skips, diag


def train_step(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency, optimizer_step):
    import torch
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, rng, pool)
    b1._histories(batch, data['idmap'], config['history_len'])
    histories, side = _tensors2(batch, data['idmap'], max(len(r['history']) for r in batch), config['feature'])
    log_q = b1._log_q(batch, negatives, pool_size, config, frequency)
    device = next(model.parameters()).device
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss, skips, assignment = multi_interest_loss(model.users(histories.to(device),side.to(device)),model.items(columns.to(device)),
        labels.to(device),mask.to(device),config['temperature'],log_q,optimizer_step)
    stats.record(epoch,len(batch),skips); stats.check(epoch)
    record = {'drawn':len(batch),'skipped':skips,'negatives':len(negatives),'negatives_sha256':b1._digest(negatives),
              'loss':None,'gradient_norm':None,'assignment':assignment,'head_w2_gradient_norm':None}
    if loss is None:
        return record
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('nonfinite gradient')
    heads = model.attn_w.grad.norm(dim=1).tolist()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError('nonfinite gradient norm')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite parameters')
    return {**record,'loss':float(loss.detach()),'gradient_norm':float(norm.detach()),'head_w2_gradient_norm':heads}


def validate_assignment_record(record, skips, drawn, optimizer_step):
    """Validate descriptive assignment receipts without imposing head-coverage gates."""
    d=record.get('assignment',{})
    if d.get('optimizer_step') != optimizer_step or d.get('phase') != ('soft' if optimizer_step<=100 else 'hard'):
        raise ValueError('optimizer-step assignment mismatch')
    n=drawn-skips
    counts=d.get('assignment_counts')
    weights=d.get('soft_weight_sum')
    if (type(counts) is not list or len(counts)!=4 or any(type(v) is not int or v<0 for v in counts) or sum(counts)!=n
            or type(weights) is not list or len(weights)!=4 or any(not math.isfinite(v) or v<0 for v in weights)
            or abs(sum(weights)-n)>1e-4*max(1,n)):
        raise ValueError('assignment diagnostic counts mismatch')
    entropy=d.get('soft_assignment_entropy_sum')
    if not isinstance(entropy,(float,int)) or not math.isfinite(entropy) or entropy<0 or entropy>n*math.log(4)+1e-4:
        raise ValueError('assignment entropy mismatch')
    grads=record.get('head_w2_gradient_norm')
    if record['loss'] is None:
        if n or grads is not None: raise ValueError('invalid skipped optimizer step')
    elif type(grads) is not list or len(grads)!=4 or any(not math.isfinite(v) or v<0 for v in grads):
        raise ValueError('invalid head gradients')


def hard_window(records):
    selected=[r for r in records if r['loss'] is not None and 101<=r['assignment']['optimizer_step']<=200]
    counts=[sum(r['assignment']['assignment_counts'][j] for r in selected) for j in range(4)]
    total=sum(counts)
    return {'optimizer_steps_101_200_observed':len(selected),'head_counts':counts,
            'head_fractions':[v/total if total else None for v in counts],
            'effective_interests':sum(v>0 for v in counts) if total else None,
            'diagnostic_only':True}


def training_diagnostics(records, epoch):
    selected=[r for r in records if r['epoch']==epoch and r['loss'] is not None]
    n=sum(r['drawn']-r['skipped'] for r in selected)
    return {'samples':n, 'soft_weight_mean':[sum(r['assignment']['soft_weight_sum'][j] for r in selected)/n if n else None for j in range(4)],
            'assignment_entropy_mean':sum(r['assignment']['soft_assignment_entropy_sum'] for r in selected)/n if n else None,
            'w2_gradient_norm_mean':[sum(r['head_w2_gradient_norm'][j] for r in selected)/len(selected) if selected else None for j in range(4)]}


def exact_max_topk(users, items, catalog, k=50, item_block=4096, check=None):
    """Full-catalog max of four cosine routes; float64 score accumulation and item-id ties like M1."""
    import numpy as np
    users=np.asarray(users,dtype=np.float32); items=np.asarray(items,dtype=np.float32)
    ids=np.asarray(catalog)
    if (users.ndim!=3 or users.shape[1]!=4 or items.ndim!=2 or users.shape[-1]!=items.shape[-1]
            or len(items)!=len(ids) or len(set(catalog))!=len(catalog) or not np.isfinite(users).all() or not np.isfinite(items).all()
            or type(item_block) is not int or item_block<1 or type(k) is not int or k<1):
        raise ValueError('invalid exact max inputs')
    k=min(k,len(ids)); predictions=[]; winners=[]
    for start in range(0,len(users),32):
        if check: check()
        q=users[start:start+32]
        best_ids=[np.empty(0,dtype=ids.dtype) for _ in q]
        best_scores=[np.empty(0,dtype=np.float64) for _ in q]
        best_heads=[np.empty(0,dtype=np.int64) for _ in q]
        for offset in range(0,len(items),item_block):
            if check: check()
            routes=np.einsum('qhd,id->qhi',q,items[offset:offset+item_block],dtype=np.float64,optimize=False)
            scores=routes.max(1); heads=routes.argmax(1)
            for row in range(len(q)):
                ci=np.concatenate((best_ids[row],ids[offset:offset+item_block]))
                cs=np.concatenate((best_scores[row],scores[row]))
                ch=np.concatenate((best_heads[row],heads[row]))
                order=np.lexsort((ci,-cs))[:k]
                best_ids[row],best_scores[row],best_heads[row]=ci[order],cs[order],ch[order]
        predictions.extend([v.tolist() for v in best_ids])
        winners.extend(int(v[0]) for v in best_heads)
    return predictions,winners


def export_vectors(model,data,queries,config,check=None):
    import numpy as np
    import torch
    from .retrieval_experiment import normalize_export
    model.eval(); device=next(model.parameters()).device; us=[]; vs=[]
    with torch.no_grad():
        for start in range(0,len(data['catalog']),config['batch_size']):
            if check: check()
            ids=[data['idmap'][i] for i in data['catalog'][start:start+config['batch_size']]]
            vs.append(model.items(torch.tensor(ids,device=device)).cpu().numpy())
        for start in range(0,len(queries),config['batch_size']):
            if check: check()
            h,s=_tensors2(queries[start:start+config['batch_size']],data['idmap'],50,config['feature'])
            us.append(model.users(h.to(device),s.to(device)).cpu().numpy())
    raw_u=np.concatenate(us); raw_v=np.concatenate(vs)
    norms=np.linalg.norm(raw_v.astype(np.float64),axis=1)
    return normalize_export(raw_u.reshape(-1,raw_u.shape[-1]),check).reshape(raw_u.shape),normalize_export(raw_v,check),norms


def model_diagnostics(model,data,queries,predictions,config,check=None):
    import torch
    model.eval(); device=next(model.parameters()).device; collected={}
    with torch.no_grad():
        for start in range(0,len(queries),config['batch_size']):
            if check: check()
            h,s=_tensors2(queries[start:start+config['batch_size']],data['idmap'],50,config['feature'])
            for key,value in model.diagnose(h.to(device),s.to(device)).items():
                collected.setdefault(key,[]).append(value.cpu())
    out={k:torch.cat(v) for k,v in collected.items()}
    return {'gate':float(model.gate.detach()),
            'branch_ratio_median':float(out['branch_ratio'].median()),
            'head_branch_ratio_median':out['head_branch_ratio'].median(0).values.tolist(),
            'pairwise_user_cosine_mean':out['pairwise_user_cosine'].mean(0).tolist(),
            'attn_entropy_ratio':out['attn_entropy_ratio'].mean(0).tolist(),
            'attn_last_weight':out['attn_last_weight'].mean(0).tolist(),
            'cos_u_m_mean':out['cos_u_m'].mean(0).tolist(),
            'recall50_by_history_length':recall_by_history_length(predictions,queries,data['catalog']),
            'recall50_by_last_event_support':recall_by_last_event_support(predictions,queries,data['catalog'],data['support'])}


def evaluate_model(model,data,queries,config,check=None):
    import numpy as np
    from .baselines import metrics
    users,items,norms=export_vectors(model,data,queries,config,check)
    predictions,winners=exact_max_topk(users,items,data['catalog'],50,check=check)
    result={'predictions':predictions,'metrics':metrics(predictions,queries,data['catalog'],(10,20,50)),
            'support_metrics':fr.support_slices(predictions,queries,data['support'],data['catalog']),
            'zero_item_vectors':int(np.count_nonzero(np.linalg.norm(items,axis=1)==0)),
            'zero_user_vectors':int(np.count_nonzero(np.all(np.linalg.norm(users,axis=2)==0,axis=1)))}
    diag={'zero_user_vectors':result['zero_user_vectors'],'zero_user_vector_fraction':result['zero_user_vectors']/len(queries),
          'top1_is_last_history_fraction':sum(p[0]==q['history'][-1] for p,q in zip(predictions,queries))/len(queries),
          'history_support_slices':f1.history_support_slices(predictions,queries,data['support'],data['catalog']),
          'top50_unsupported_share':fr.top50_unsupported_share(predictions,data['support']),
          'item_norm_stats':fr.item_norm_stats(norms,data['catalog'],data['support']),
          'top1_winning_head_counts':[winners.count(j) for j in range(4)],
          **model_diagnostics(model,data,queries,predictions,config,check)}
    return result,diag


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
            'delta_recall@50': best_recall - metric['recall@50'], 'predictions': result['predictions']}


# ---------------------------------------------------------------- binding / run

CONTROL_DATA_KEYS = ('rows_sha256', 'queries_sha256', 'catalog_sha256', 'support_sha256', 'content_sha256')
CONTROL_INPUT_KEYS = ('prepared_sha', 'query_sha', 'catalog_sha', 'history_manifest_sha256', 'side_manifest_sha256', 'inputs_sha256')


def _control_hex(value, length=64):
    import re
    return type(value) is str and re.fullmatch(r'[0-9a-fA-F]{%d}' % length, value) is not None


def control_audit_sha256(audit):
    """Canonical seal format shared with admission: sorted default JSON, no newline."""
    import hashlib
    return hashlib.sha256(json.dumps(audit, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_control_audit(audit):
    """Validate sealed archive metadata, without reopening old predictions or stdout."""
    if (type(audit) is not dict or audit.get('schema') != 'seqm2-m1-controls-v1'
            or audit.get('seeds') != [42, 2024, 2025]):
        raise ValueError('M1 control audit schema/seeds mismatch')
    controls = audit.get('controls'); ledger = audit.get('read_artifacts_sha256')
    if (type(controls) is not dict or set(controls) != {f'p-seqm1-mean-s{s}' for s in (42,2024,2025)}
            or type(ledger) is not dict or not ledger
            or any(type(p) is not str or not p or not _control_hex(h) for p,h in ledger.items())):
        raise ValueError('M1 control audit population/ledger mismatch')

    def ledger_pin(record, path_key='path', hash_key='sha256'):
        if (type(record) is not dict or type(record.get(path_key)) is not str or not record[path_key]
                or not _control_hex(record.get(hash_key)) or ledger.get(record[path_key]) != record[hash_key]):
            raise ValueError('M1 control artifact pin/ledger mismatch')

    shared = None
    for seed in (42,2024,2025):
        rid = f'p-seqm1-mean-s{seed}'; entry = controls[rid]
        if (type(entry) is not dict or entry.get('run_id') != rid or type(entry.get('seed')) is not int
                or entry['seed'] != seed or entry.get('verified') is not True or entry.get('arm') != 'mean'
                or type(entry.get('best_epoch')) is not int or not 1 <= entry['best_epoch'] <= 100
                or type(entry.get('summary')) is not dict):
            raise ValueError('M1 control run/seed/selection identity mismatch')
        for path_key,hash_key in (('manifest_path','manifest_sha256'),('predictions_path','predictions_sha256'),
                ('history_manifest_path','history_manifest_sha256'),('query_file_path','query_file_sha256')):
            ledger_pin(entry,path_key,hash_key)
        if entry['manifest_path'] == entry['predictions_path']:
            raise ValueError('M1 manifest/predictions identity collision')
        source = entry.get('source')
        if (type(source) is not dict or not _control_hex(source.get('git_commit'),40)
                or type(source.get('files')) is not dict or not source['files']
                or any(type(p) is not str or not p or not _control_hex(h) for p,h in source['files'].items())):
            raise ValueError('M1 control producer source identity missing')
        hashes = entry.get('data_hashes')
        if type(hashes) is not dict or set(hashes) != set(CONTROL_DATA_KEYS) or any(not _control_hex(v) for v in hashes.values()):
            raise ValueError('M1 control data hashes missing')
        if shared is not None and hashes != shared:
            raise ValueError('M1 control seeds use different data')
        shared = hashes
        inputs = entry.get('inputs')
        if (type(inputs) is not dict or any(not _control_hex(inputs.get(k)) for k in CONTROL_INPUT_KEYS)
                or inputs['history_manifest_sha256'] != entry['history_manifest_sha256']):
            raise ValueError('M1 control original input pins missing')
        verifier = entry.get('verifier')
        if type(verifier) is not dict or verifier.get('verified') is not True or type(verifier.get('jobs')) is not list:
            raise ValueError('M1 control direct verifier missing')
        accepted = False
        for job in verifier['jobs']:
            if type(job) is not dict or job.get('verified') is not True:
                continue
            if job.get('run_id') != rid + '-verify':
                raise ValueError('M1 control verifier run identity mismatch')
            terminal = job.get('terminal')
            receipt = job.get('job_receipt')
            records = job.get('records'); evidence = job.get('evidence')
            if (type(terminal) is not dict or terminal.get('status') != 'succeeded'
                    or type(terminal.get('exit_code')) is not int or terminal['exit_code'] != 0
                    or type(receipt) is not dict or receipt.get('target_binding_matches') is not True
                    or type(records) is not list or not any(type(r) is dict and r.get('status') == 'verified' for r in records)
                    or type(evidence) is not list or not evidence):
                raise ValueError('M1 control verifier acceptance evidence mismatch')
            ledger_pin(receipt)
            for artifact in evidence:
                ledger_pin(artifact)
            accepted = True
        if not accepted:
            raise ValueError('M1 control has no accepted direct verifier')
    return audit


def _control_binding(inputs_binding):
    if type(inputs_binding) is not dict:
        raise ValueError('real M2 inputs require M1 control audit')
    control = inputs_binding.get('m1_control_audit')
    if type(control) is not dict or set(control) != {'sha256','audit'} or not _control_hex(control.get('sha256')):
        raise ValueError('real M2 inputs require pinned M1 control audit')
    audit = validate_control_audit(control['audit'])
    if control_audit_sha256(audit) != control['sha256']:
        raise ValueError('M1 control canonical audit pin drift')
    return control


def load_control_binding(inputs_binding):
    """Only the real CLI loads the sealed control audit from its explicit environment."""
    import os
    path = os.environ.get('VKGR_M2_CONTROL_AUDIT')
    pin = os.environ.get('VKGR_M2_CONTROL_AUDIT_SHA')
    if not path or not _control_hex(pin):
        raise ValueError('real CLI requires M1 control audit path and SHA')
    audit = validate_control_audit(read_pinned(path, pin))
    binding = {**inputs_binding, 'm1_control_audit': {'sha256': pin, 'audit': audit}}
    _control_binding(binding)
    return binding


def binding_for4(data, config, rows, queries, model, device, inputs_binding):
    control = _control_binding(inputs_binding) if not data.get('synthetic') else None
    base = fr.binding_for2(data, config, rows, queries, model, device, inputs_binding)
    if control is not None:
        for entry in control['audit']['controls'].values():
            if any(entry['data_hashes'][k] != base[k] for k in CONTROL_DATA_KEYS):
                raise ValueError('M1/M2 shared data identity mismatch')
            for key in CONTROL_INPUT_KEYS:
                if entry['inputs'][key] != inputs_binding.get(key):
                    raise ValueError('M1/M2 original input pin mismatch: ' + key)
        base = {**base, 'm1_control_audit_sha256': control['sha256'],
                'm1_control_manifest_sha256': {rid:entry['manifest_sha256'] for rid,entry in control['audit']['controls'].items()}}
    return {**base, 'task': TASK, 'arm': config['arm'], 'mask': model.mask_kind, 'readout': model.readout, 'anchored': model.anchored,
            'branch_spec_sha256': branch_spec_sha256(model), 'param_report': parameter_report_seqm2(model),
            'dependency_sha256': b1.source_identity()['files'], 'interests': 4,
            'assignment': {'soft_optimizer_steps': 100, 'temperature': .1, 'hard_tie': 'smallest-head'},
            'evaluation': 'full-catalog-exact-max-top50',
            'idmap_sha256': b1._digest(data['idmap']), 'supported_indices_sha256': b1._digest(data['supported_indices'])}


def _run_session(output, data, config, device='cpu', check=None, inputs_binding=None):
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
    binding = binding_for4(data, config, rows, queries, model, device, inputs_binding)
    view = {**data, 'rows': rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    step = 0; optimizer_steps = 0; records = []; evaluations = []; best = None; last = None; train_seconds = 0.; eval_seconds = 0.
    diagnostics = {**b1.data_diagnostics(rows, queries, data['support']), 'side_visibility': f1.side_visibility(queries, data['support'])}
    b1._write_progress('train', 0, config['max_epochs'])
    while not stop.stopped:
        if check: check()
        begin = time.monotonic()
        indices = cursor.next(config['batch_size'])
        record = train_step(model, optimizer, [rows[i] for i in indices], view, stats, cursor.epoch, config, negative_rng, pool, frequency, optimizer_steps + 1)
        optimizer_steps += int(record['loss'] is not None)
        step += 1; records.append({**record, 'step': step, 'epoch': cursor.epoch + 1, 'indices_sha256': b1._digest(indices)})
        train_seconds += time.monotonic() - begin
        if cursor.offset != cursor.size:
            continue
        begin = time.monotonic(); epoch = cursor.epoch + 1
        b1._write_progress('evaluate', epoch, config['max_epochs'])
        result, epoch_diag = evaluate_model(model, data, queries, config, check)
        epoch_diag['hard_window'] = hard_window(records)
        epoch_diag['training_assignment'] = training_diagnostics(records, epoch)
        primary = result['metrics']['all']; improved = stop.observe(epoch, primary['recall@50'], primary['ndcg@50'])
        predictions_name = f'predictions-epoch-{epoch:03d}.json'
        predictions_sha = b1._write_json_atomic(output / predictions_name, result['predictions'])
        evaluations.append({'epoch': epoch, 'result': {k: v for k, v in result.items() if k != 'predictions'}, 'predictions_ref': predictions_name,
                            'predictions_sha256': predictions_sha, 'diagnostics': epoch_diag})
        eval_seconds += time.monotonic() - begin
        if check: check()
        state = {'schema': CHECKPOINT_SCHEMA, 'binding_sha256': b1._digest(binding), 'step': step, 'optimizer_steps': optimizer_steps, 'epoch': epoch, 'model': model.state_dict(),
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
        g0_predictions = ablation.pop('predictions')
        g0_sha = b1._write_json_atomic(output / 'predictions-best-g0.json', g0_predictions)
        ablation = {**ablation, 'predictions_ref': 'predictions-best-g0.json', 'predictions_sha256': g0_sha}
    value = {**value, 'status': 'complete', 'ablation_g0': ablation}
    sha = b1._write_json_atomic(output / 'manifest.json', value)
    b1._write_progress('complete', epoch, config['max_epochs'])
    return {**value, 'manifest_sha256': sha}


def run_session(output, data, config, device='cpu', check=None, inputs_binding=None):
    """Fail closed and retain evidence for numerical failures; never retry an attempt."""
    try:
        return _run_session(output, data, config, device, check, inputs_binding)
    except (ValueError, RuntimeError) as exc:
        message = str(exc)
        numerical = any(word in message.lower() for word in ('nonfinite', 'non-finite', 'nan', 'inf'))
        target = Path(output)
        if numerical and target.is_dir():
            evidence = {'schema': 'p-seqm2-numerical-failure-v1', 'status': 'failed_no_retry',
                        'config_sha256': b1._digest(config), 'exception': type(exc).__name__, 'message': message,
                        'last_complete_epoch_manifest_present': (target / 'manifest.json').is_file()}
            b1._write_json_atomic(target / 'numerical-failure.json', evidence)
        raise


def verify_run(path, digest, data, config, device='cpu', check=None, inputs_binding=None):
    import torch
    from .baselines import metrics
    from .retrieval_hp import EarlyStop
    from .retrieval_training import EpochCursor, SamplerStats, candidate_batch
    from .retrieval_negatives import supported_uniform_columns
    from .retrieval_ablation_run import state_equal
    path = Path(path); value = read_pinned(path, digest); run = path.parent
    if set(value) != MANIFEST_FIELDS or value['schema'] != MANIFEST_SCHEMA or value['kind'] != KIND or value['status'] != 'complete':
        raise ValueError('complete P-SEQ-M2 manifest required')
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
    cursor = EpochCursor(len(rows), config['seed']); stats = SamplerStats(); rng = random.Random(f"p-b1-negative-v1:{config['seed']}")
    pool = sorted(data['support']) if config['extra_uniform'] else None
    epoch_ends = []; optimizer_steps = 0
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
        validate_assignment_record(record, skips, len(indices), optimizer_steps + 1)
        optimizer_steps += int(record['loss'] is not None)
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
        if stored.get('hard_window') != hard_window(value['records'][:step]) or stored.get('training_assignment') != training_diagnostics(value['records'], epoch):
            raise ValueError('epoch training diagnostics replay mismatch')
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
    diagnostics['hard_window'] = hard_window(value['records'][:best['step']])
    diagnostics['training_assignment'] = training_diagnostics(value['records'], best['epoch'])
    if diagnostics != value['evaluations'][best['epoch'] - 1].get('diagnostics'):
        raise ValueError('best checkpoint exact re-export diagnostics mismatch')
    ablation = ablation_g0(model, data, queries, config, best['epoch'], recomputed_best_recall(value, best), check)
    expected_g0 = value['ablation_g0']
    if expected_g0.get('predictions_ref') != 'predictions-best-g0.json':
        raise ValueError('g0 prediction path mismatch')
    if ablation.pop('predictions') != read_pinned(run / expected_g0['predictions_ref'], expected_g0['predictions_sha256']):
        raise ValueError('best checkpoint g0 exact prediction mismatch')
    if ablation != {k:v for k,v in expected_g0.items() if k not in ('predictions_ref','predictions_sha256')}:
        raise ValueError('best checkpoint g->0 ablation not reproducible')
    last = value['last']
    if type(last) is not dict or last.get('path') != 'last.pt' or last.get('step') != value['steps'] or file_sha256(run / 'last.pt') != last.get('sha256'):
        raise ValueError('last checkpoint hash mismatch')
    state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
    if state.get('binding_sha256') != b1._digest(value['binding']) or state['step'] != value['steps'] or not state_equal(state['cursor'], cursor.state_dict()) or not state_equal(state['negative_rng'], rng.getstate()) or state['stats'] != value['stats'] or state['selection'] != value['selection'] or state.get('optimizer_steps') != optimizer_steps or state.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('last state mismatch')
    if sorted(p.name for p in run.glob('*.pt')) != ['best.pt', 'last.pt']:
        raise ValueError('checkpoint policy best+last violated')
    return {'status': 'verified', 'replayed_epochs': [best['epoch']], 'best_epoch': best['epoch'], 'steps': value['steps']}


def recomputed_best_recall(value, best):
    return value['evaluations'][best['epoch'] - 1]['result']['metrics']['all']['recall@50']


def main(argv=None):
    import sys
    parser = argparse.ArgumentParser(description='SEQ-M2 P runner (new version; FEAT-02 and P-HP-01 runners untouched)')
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
        inputs_binding = load_control_binding({**binding, **extra_binding, 'inputs_sha256': args.inputs_sha})
        if args.command == 'train':
            value = run_session(target, data, config, 'cuda', check, inputs_binding)
            print(json.dumps({'status': value['status'], 'manifest_sha256': value['manifest_sha256'], 'best_epoch': value['selection']['best_epoch']})); return 0
        print(json.dumps(verify_run(args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs_binding))); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR P-SEQ-M2 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
