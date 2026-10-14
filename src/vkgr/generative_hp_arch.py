"""G-HP-01 bounded one-factor tuning on the TIGER paper-scale anchor A1: config registry, model factory and optimizer.

Nothing here is reachable from the default G path or from the GARCH-01 entry. The model class is GARCH-01's
``TigerArchModel`` (already configurable in width / heads / head_dim / layers / ff / dropout); what blocked G1-G5 was only
the GARCH whitelist (two structures, LR 3e-4 / 1e-3). This registry is a wider whitelist: the product of
width {A1 (128, 6x64, ff 1024), G1 (256, 6x128, ff 2048)} x layers {4+4, 2+2} x dropout {0.1, 0.0} and LR {1e-4, 3e-4, 6e-4},
so a later combination run needs no code change. Which configs are actually submitted is fixed by the frozen config sha256s,
not by this whitelist. ``task_id`` is 'G-HP-01': GARCH configs are rejected here and G-HP configs are rejected by GARCH.
"""
import torch
from torch.optim import AdamW

from . import generative_arch as ga

SCHEMA = ga.SCHEMA
TASK_ID = 'G-HP-01'
SHAPE_KEYS = ga.SHAPE_KEYS
A1_SHAPE = ga.PAPER_SCALE
# (d_model, heads, head_dim, ff): G1 doubles d_model, FF and the attention width; heads stay 6 -> head_dim 64 -> 128
WIDTHS = {'a1': (128, 6, 64, 1024), 'g1': (256, 6, 128, 2048)}
LAYERS = (4, 2)  # encoder = decoder
DROPOUTS = (0.1, 0.0)
HP_SHAPES = tuple((d, h, hd, n, n, ff, p) for d, h, hd, ff in WIDTHS.values() for n in LAYERS for p in DROPOUTS)
LEARNING_RATES = (.0001, .0003, .0006)
SEEDS = ga.SEEDS
KNOBS = {'width': ('d_model', 'ff', 'head_dim'), 'lr': ('learning_rate',), 'layers': ('encoder_layers', 'decoder_layers'), 'dropout': ('dropout',)}
FIXED = {**ga.FIXED, 'task_id': TASK_ID}


def default_config():
    return {**ga.default_config(), 'task_id': TASK_ID}


def validate_config(value, engineering=False):
    base = default_config()
    if type(value) is not dict or set(value) != set(base):
        raise ValueError('unregistered G-HP config fields')
    if any(value[k] != v or type(value[k]) is not type(v) for k, v in FIXED.items()):
        raise ValueError('frozen G-HP field changed')
    if any(type(value[k]) is not int for k in ga.INT_KEYS) or any(type(value[k]) is not float for k in ('dropout', 'learning_rate')):
        raise ValueError('typed G-HP config required')
    if value['learning_rate'] not in LEARNING_RATES or value['seed'] not in SEEDS:
        raise ValueError('unregistered G-HP learning rate or seed')
    shape = tuple(value[k] for k in SHAPE_KEYS)
    if engineering:
        if not (1 <= value['d_model'] <= 256 and 1 <= value['heads'] <= 8 and value['head_dim'] in (8, 16, 32, 64, 128) and 1 <= value['encoder_layers'] <= 4
                and 1 <= value['decoder_layers'] <= 4 and 1 <= value['ff'] <= 2048 and 0.0 <= value['dropout'] <= 0.5 and 1 <= value['microbatch'] <= 512
                and 1 <= value['max_epochs'] <= 100 and 0 <= value['min_epochs'] <= value['max_epochs']):
            raise ValueError('bounded engineering G-HP config')
    elif shape not in HP_SHAPES or value['microbatch'] not in (512, 256, 128) or (value['max_epochs'], value['min_epochs']) != (100, 5):
        raise ValueError('formal G-HP = a registered structure + BUDGET_E100_MIN5_P3')
    return value


def _arch_view(config):
    """The config handed to the GARCH factory (all structure fields passed through unchanged)."""
    return {**config}


def make_model_hp(codec, config, *, device):
    """Thin delegation to ``generative_arch.make_model_arch`` (same class, same init order).

    At A1 values the model is value-for-value identical to ``ga.make_model_arch`` / a directly built ``TigerArchModel``
    under the same seed - by construction, and pinned by test H1.
    """
    if any(type(config[k]) is not int or config[k] < 1 for k in SHAPE_KEYS[:-1]) or type(config['dropout']) is not float:
        raise ValueError('typed positive G-HP structure required')
    return ga.make_model_arch(codec, _arch_view(config), device=device)


def make_optimizer_hp(model, config):
    """AdamW like ``generative_training.make_optimizer`` with the G-HP learning rates; takes the core view or the full config."""
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('FP32 parameters required')
    if config['learning_rate'] not in LEARNING_RATES or config['weight_decay'] != FIXED['weight_decay']:
        raise ValueError('frozen G-HP optimisation required')
    return AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
