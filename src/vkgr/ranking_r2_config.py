"""Frozen R2 engineering contract; no unresolved scientific field has a fallback."""
import copy
import json
from pathlib import Path
from . import generative_g6_run as original

SCHEMA = 'g-ranking-r2-run-v1'
TASK_ID = 'G-RANK-R2'
SEEDS = (42, 2024, 2025)
ADMISSION = {'receipt_sha256': '089148f65664ff0f7518ff517221ccb07ed0c3a9ad3b7898ec1221be347aee96',
             'within_four_hours': True, 'model_weights': 'random_initialization_engineering_only',
             'owner_ack': '2026-10-17 train-only microbenchmark ACK; new SFT best mining remains measured separately'}


def default_config(seed=42):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('fixed three seeds required')
    base = json.loads((Path(__file__).resolve().parents[2]/'configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json').read_text())
    base['seed'] = seed
    return {'schema': SCHEMA, 'task_id': TASK_ID, 'seed': seed, 'sft': base, 'negative_policy': 'skip_without_resampling',
            'mining_samples': 20000, 'mining_seed': 0, 'mining_beam': 10, 'decode_batch_size': 1,
            'dpo_beta': .1, 'positive_ce_weight': 1., 'dpo_learning_rate': .00003,
            'dpo_max_epochs': 5, 'dpo_min_epochs': 1, 'dpo_patience': 2, 'dpo_microbatch': 256,
            'effective_batch': 512, 'mining_admission': copy.deepcopy(ADMISSION), 'test_gate': 'closed'}


def validate(value, engineering=False):
    if type(value) is not dict or type(value.get('seed')) is not int or value['seed'] not in SEEDS:
        raise ValueError('explicit R2 seed/config required')
    base = default_config(value['seed'])
    if set(value) != set(base):
        raise ValueError('complete frozen R2 fields required')
    for name in base:
        if name == 'sft' and engineering:
            continue
        if type(value[name]) is not type(base[name]) or json.dumps(value[name],sort_keys=True,allow_nan=False) != json.dumps(base[name],sort_keys=True,allow_nan=False):
            raise ValueError('R2 frozen field differs or is unresolved: '+name)
    original.validate_config(value['sft'], engineering)
    if value['sft']['seed'] != value['seed']:
        raise ValueError('SFT and R2 seed mismatch')
    return value


def sft_config(value):
    from . import ranking_r2_sft as sft
    return dict(value['sft'], schema=sft.SCHEMA, task_id=sft.TASK_ID)
