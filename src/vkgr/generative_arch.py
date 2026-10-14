"""GARCH-01 explicit architecture switch for the G model: config registry, model factory and optimizer.

Nothing here is reachable from the default G path. The frozen ``generative_training._core`` accepts only heads=4, 1+1
layers, ff=4*d_model and dropout 0, so the paper-scale structure goes through this registry instead. The registry is a
whitelist: a run may only use one of two registered structures (the anchor shape is for equivalence/parity only, the
paper scale is the counted structure); everything else fails closed before any output is created.
"""
import torch
from torch.optim import AdamW

from .generative_hp import embedding_report, init_embeddings
from .tiger_arch import TigerArchModel, expected_param_count

SCHEMA = 'g-b1-arch-run-v1'
TASK_ID = 'G-ARCH-01'
SHAPE_KEYS = ('d_model', 'heads', 'head_dim', 'encoder_layers', 'decoder_layers', 'ff', 'dropout')
ANCHOR_SHAPE = (128, 4, 32, 1, 1, 512, 0.0)  # equivalence/parity only, never a counted run
PAPER_SCALE = (128, 6, 64, 4, 4, 1024, 0.1)  # TIGER 4.1.3: 4+4 layers, 6 heads x 64, MLP 1024, dropout 0.1 (d_model 128)
PRESETS = {'anchor_shape': ANCHOR_SHAPE, 'paper_scale': PAPER_SCALE}
FIXED = {'schema': SCHEMA, 'task_id': TASK_ID, 'precision': 'fp32', 'weight_decay': 1e-4, 'gradient_clip_norm': 1.0, 'effective_batch': 512,
         'init': 'small', 'history_version': 'HIST_V2_OBS', 'history_len': 50, 'patience': 3, 'sid': 'SID_BHKM_B16D3_S42', 'beam': 100, 'test_gate': 'closed'}
LEARNING_RATES = (.0003, .001)
SEEDS = (42, 2024, 2025)
INT_KEYS = ('d_model', 'heads', 'head_dim', 'encoder_layers', 'decoder_layers', 'ff', 'microbatch', 'max_epochs', 'min_epochs', 'seed')


def default_config():
    config = dict(FIXED)
    config.update(zip(SHAPE_KEYS, PAPER_SCALE))
    config.update(learning_rate=.0003, microbatch=256, seed=42, max_epochs=100, min_epochs=5)
    return config


def validate_config(value, engineering=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered G-ARCH config fields')
    if any(value[k] != v or type(value[k]) is not type(v) for k, v in FIXED.items()):
        raise ValueError('frozen G-ARCH field changed')
    if any(type(value[k]) is not int for k in INT_KEYS) or any(type(value[k]) is not float for k in ('dropout', 'learning_rate')):
        raise ValueError('typed G-ARCH config required')
    if value['learning_rate'] not in LEARNING_RATES or value['seed'] not in SEEDS:
        raise ValueError('unregistered learning rate or seed')
    shape = tuple(value[k] for k in SHAPE_KEYS)
    if engineering:
        if not (1 <= value['d_model'] <= 128 and 1 <= value['heads'] <= 8 and value['head_dim'] in (8, 16, 32, 64) and 1 <= value['encoder_layers'] <= 4
                and 1 <= value['decoder_layers'] <= 4 and 1 <= value['ff'] <= 1024 and 0.0 <= value['dropout'] <= 0.5 and 1 <= value['microbatch'] <= 512
                and 1 <= value['max_epochs'] <= 100 and 0 <= value['min_epochs'] <= value['max_epochs']):
            raise ValueError('bounded engineering G-ARCH config')
    elif shape not in PRESETS.values() or value['microbatch'] not in (512, 256, 128) or (value['max_epochs'], value['min_epochs']) != (100, 5):
        raise ValueError('formal G-ARCH = a registered structure + BUDGET_E100_MIN5_P3')
    return value


def arch_report(model):
    """Recorded in the run binding (verify recomputes it): structure, measured parameter counts, closed form."""
    spec = dict(model.spec)
    total = sum(p.numel() for p in model.parameters())
    expected = expected_param_count(spec['vocab'], spec['d_model'], spec['heads'], spec['head_dim'], spec['encoder_layers'], spec['decoder_layers'], spec['ff'], spec['positions'])
    if total != expected:
        raise ValueError('parameter count disagrees with the closed form')
    return {'spec': spec, 'params_total': total, 'params_transformer': sum(p.numel() for p in model.transformer.parameters()), 'params_expected': expected}


def make_model_arch(codec, config, *, device):
    if config['init'] not in ('legacy', 'small') or config['history_len'] != 50:
        raise ValueError('unregistered G-ARCH model switch')
    if any(type(config[k]) is not int for k in SHAPE_KEYS[:-1]) or type(config['dropout']) is not float:
        raise ValueError('typed architecture config required')
    model = TigerArchModel(codec.vocab_size, *(config[k] for k in SHAPE_KEYS)).to(device=device, dtype=torch.float32)
    report = init_embeddings(model) if config['init'] == 'small' else embedding_report(model)
    model.init_report = {'init': config['init'], 'std': 0.02 if config['init'] == 'small' else None, 'embeddings': report, 'arch': arch_report(model)}
    return model


def make_optimizer_arch(model, config):
    """Same AdamW as ``generative_training.make_optimizer``; takes the core view or the full config (needs only lr and weight decay)."""
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('FP32 parameters required')
    if config['learning_rate'] not in LEARNING_RATES or config['weight_decay'] != FIXED['weight_decay']:
        raise ValueError('frozen optimisation required')
    return AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
