"""R2 fixed train-only mining and actor/reference DPO; no checkpoint I/O."""
import hashlib
import json
import time
import torch
from . import ranking_r2 as r
from .data_audit import canonical_bytes
from .generative_g6_model import full_vocab_ce_d
from .generative_training import StepResult
from .generative_hp import EarlyStop
from .sid_depth_beam import decode_histories_batched_d


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def indices_digest(indices):
    return hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest()


def mine_rows(model, tokens, rows, indices, *, engineering=False, guard=lambda: None, on_progress=None):
    if type(engineering) is not bool or not indices or len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(rows) for i in indices):
        raise ValueError('unique bounded train indices required')
    if not engineering and indices != r.sample_training_indices(len(rows)):
        raise ValueError('formal mining fixed 20000 seed0 training sample required')
    records = []
    started = time.monotonic()
    model.eval()
    for offset, index in enumerate(indices):
        guard()
        if time.monotonic()-started >= 4*3600:
            raise ValueError('new SFT best actual mining exceeded 4h; no resize/retry')
        row = rows[index]
        result = decode_histories_batched_d(model, tokens, [row['history']], 10, 1)
        items, audit = result[0]
        if any(audit.get(key, 0) for key in ('invalid', 'deadend', 'collision')):
            raise ValueError('mining beam audit failure')
        records.append({'example_index': index, 'row_sha256': digest(row), 'beam': items, 'negative': r.select_negative(items, row)})
        if on_progress is not None and (offset % 64 == 0 or offset+1 == len(indices)):
            on_progress(offset+1, len(indices))
    skipped = sum(row['negative'] is None for row in records)
    return {'schema': 'r2-mining-v1', 'sample_seed': 0, 'sample_indices_sha256': indices_digest(indices), 'sample_count': len(indices),
            'population': len(rows), 'beam': 10, 'decode_batch_size': 1, 'negative_policy': 'skip_without_resampling',
            'records': records, 'skipped': skipped, 'skip_fraction': skipped/len(indices), 'elapsed_seconds': time.monotonic()-started}


def validate_mining(value, rows, tokens, *, engineering=False):
    records = value['records']
    indices = [record['example_index'] for record in records]
    if not records or len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(rows) for i in indices):
        raise ValueError('mining training index identity')
    if not engineering and indices != r.sample_training_indices(len(rows)):
        raise ValueError('mining sample changed or resampled')
    expected = {'schema': 'r2-mining-v1', 'sample_seed': 0, 'sample_indices_sha256': indices_digest(indices), 'sample_count': len(indices),
                'population': len(rows), 'beam': 10, 'decode_batch_size': 1, 'negative_policy': 'skip_without_resampling'}
    if any(type(value.get(k)) is not type(v) or value[k] != v for k, v in expected.items()):
        raise ValueError('fixed mining protocol mismatch')
    catalog = set(map(int, tokens.catalog.ids))
    selected = []
    skipped = 0
    for record in records:
        row = rows[record['example_index']]
        beam = record['beam']
        if type(beam) is not list or not 1 <= len(beam) <= 10 or len(set(beam)) != len(beam) or any(type(i) is not int or i not in catalog for i in beam):
            raise ValueError('mining beam catalog/uniqueness mismatch')
        if record['row_sha256'] != digest(row) or record['negative'] != r.select_negative(beam, row):
            raise ValueError('mining target/full-likes/highest-eligible mismatch')
        if record['negative'] is None:
            skipped += 1
        else:
            selected.append(dict(row, negative=record['negative']))
    if value['skipped'] != skipped or value['skip_fraction'] != skipped/len(indices):
        raise ValueError('mining skip report mismatch')
    return selected


def make_optimizer(actor, config):
    if config['dpo_learning_rate'] != .00003 or any(p.dtype != torch.float32 for p in actor.parameters()):
        raise ValueError('R2 FP32 actor and LR3e-5 required')
    return torch.optim.AdamW(actor.parameters(), lr=.00003, weight_decay=config['sft']['weight_decay'])


def train_step(actor, reference, optimizer, rows, tokens, *, microbatch, device):
    if not rows or len(rows) > 512 or type(microbatch) is not int or not 1 <= microbatch <= 256:
        raise ValueError('bounded DPO pair effective/microbatch required')
    if reference.training or any(p.requires_grad or p.grad is not None for p in reference.parameters()):
        raise ValueError('reference must remain eval and frozen')
    if {id(p) for group in optimizer.param_groups for p in group['params']} & {id(p) for p in reference.parameters()}:
        raise ValueError('reference cannot enter actor optimizer')
    actor.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    sizes = []
    try:
        for start in range(0, len(rows), microbatch):
            batch = rows[start:start+microbatch]
            n = len(batch)
            histories = [row['history'] for row in batch]*2
            targets = [row['target'] for row in batch]+[row['negative'] for row in batch]
            source, decoder, labels = (x.to(device) for x in tokens.batch(histories, targets))
            logits = actor(source, decoder)
            logp = r.sequence_logprob(logits, labels)
            with torch.no_grad():
                logref = r.sequence_logprob(reference(source, decoder), labels)
            ce = full_vocab_ce_d(logits[:n], labels[:n], 4)
            loss = r.dpo_with_ce(logp[:n], logp[n:], logref[:n], logref[n:], ce)
            weighted = loss*(n/len(rows))
            if not torch.isfinite(weighted):
                raise ValueError('nonfinite DPO+CE loss')
            weighted.backward()
            total += float(loss.detach())*n/len(rows)
            sizes.append(n)
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in actor.parameters()):
            raise ValueError('nonfinite DPO actor update')
        return StepResult(total, len(rows), tuple(sizes))
    finally:
        optimizer.zero_grad(set_to_none=True)


def dpo_stop(config):
    if (config['dpo_min_epochs'], config['dpo_max_epochs'], config['dpo_patience']) != (1, 5, 2):
        raise ValueError('frozen DPO E5_MIN1_P2 required')
    # Epoch0 is reported separately and is never observed by this selector.
    return EarlyStop(5, 1, 2)
