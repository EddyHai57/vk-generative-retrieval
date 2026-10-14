"""G-HP-01 guards shared by the plain tests and the mutation tests (this is not a test module).

Each guard raises AssertionError when a property is violated, so test_generative_hp_mutations.py can inject a defect into
the production code and demand that the very same guard turns red. Every equality here is bitwise (torch.equal / model
digest); G-HP-01 has no tolerance tier. Run inside the frozen-base harness with OMP_NUM_THREADS=1 (CPU only).
"""
import json
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import arch_checks as ac

ROOT = Path(__file__).resolve().parents[1]
V = 80  # real G codec vocabulary
CODEC = types.SimpleNamespace(vocab_size=V)  # the factories only read vocab_size
A1_PATH = ROOT / 'configs' / 'garch01' / 'g-garch01-a1-lr3e-4-s42.json'
A2_PATH = ROOT / 'configs' / 'garch01' / 'g-garch01-a2-lr1e-3-s42.json'
CONFIG_DIR = ROOT / 'configs' / 'ghp01'
# id -> (run_id = file stem, knob, changed fields relative to A1, params_total at V=80)
GRUNS = {
    'G1': ('g-ghp01-g1-width2x-s42', 'width', {'d_model': 256, 'ff': 2048, 'head_dim': 128}, 17978448),
    'G2': ('g-ghp01-g2-lr1e-4-s42', 'lr', {'learning_rate': .0001}, 4532816),
    'G3': ('g-ghp01-g3-lr6e-4-s42', 'lr', {'learning_rate': .0006}, 4532816),
    'G4': ('g-ghp01-g4-layers2-s42', 'layers', {'encoder_layers': 2, 'decoder_layers': 2}, 2289744),
    'G5': ('g-ghp01-g5-dropout0-s42', 'dropout', {'dropout': 0.0}, 4532816),
}
DROPOUT_MODULES_4X4 = 4 * 3 + 4 * 4  # 3 per encoder layer + 4 per decoder layer


def ok(condition, message):
    if not condition:
        raise AssertionError(message)


def same(a, b, message):
    ok(torch.equal(a, b), message)


def expect_rejected(call, label):
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f'accepted but must be rejected: {label}')


def accepted(call, label):
    try:
        return call()
    except ValueError as exc:
        raise AssertionError(f'rejected but must be accepted: {label}: {exc}') from exc


def a1_config():
    return json.loads(A1_PATH.read_text())


def hp_config(**overrides):
    """A1 values re-labelled as a G-HP-01 config, plus overrides."""
    return {**a1_config(), 'task_id': 'G-HP-01', **overrides}


def grun_config(gid, **overrides):
    return hp_config(**GRUNS[gid][2], **overrides)


def _seeded(factory, *args, **kwargs):
    from vkgr import generative_real as gr
    gr._seed(42)
    return factory(*args, **kwargs)


def dropouts(model):
    return [m for m in model.modules() if isinstance(m, nn.Dropout)]


def attentions(model):
    from vkgr.tiger_arch import IndependentHeadAttention
    return [m for m in model.modules() if isinstance(m, IndependentHeadAttention)]


# ---- R: registry ----------------------------------------------------------------------------------------------
def check_registry_accepts():
    from vkgr import generative_hp_arch as hp
    ok(len(hp.HP_SHAPES) == 8 and len(set(hp.HP_SHAPES)) == 8, 'HP_SHAPES must be the 8-structure product width x layers x dropout')
    for shape in hp.HP_SHAPES:
        for lr in (.0001, .0003, .0006):
            for seed in (42, 2024, 2025):
                for microbatch in (512, 256, 128):
                    cfg = hp_config(**dict(zip(hp.SHAPE_KEYS, shape)), learning_rate=lr, seed=seed, microbatch=microbatch)
                    accepted(lambda cfg=cfg: hp.validate_config(cfg), f'{shape} lr {lr} seed {seed} mb {microbatch}')
    for gid in GRUNS:
        accepted(lambda gid=gid: hp.validate_config(grun_config(gid)), gid)


def check_registry_rejects():
    from vkgr import generative_arch as ga
    from vkgr import generative_hp_arch as hp
    bad = {'task G-ARCH-01': {'task_id': 'G-ARCH-01'}, 'lr 1e-3': {'learning_rate': .001}, 'lr 3e-3': {'learning_rate': .003},
           'd_model 192': {'d_model': 192}, 'head_dim 96': {'head_dim': 96}, '3+3 layers': {'encoder_layers': 3, 'decoder_layers': 3},
           '3+4 layers': {'encoder_layers': 3}, '4+2 layers': {'decoder_layers': 2}, 'dropout 0.05': {'dropout': 0.05},
           '6x64 on d256 (residual-only widening)': {'d_model': 256, 'ff': 2048}, '6x128 on d128': {'head_dim': 128},
           'ff 2048 on d128': {'ff': 2048}, 'core03 shape': dict(zip(ga.SHAPE_KEYS, ga.ANCHOR_SHAPE)),
           'history_len 20': {'history_len': 20}, 'history_len 100': {'history_len': 100}, 'random SID': {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'},
           'beam 200': {'beam': 200}, 'max_epochs 30': {'max_epochs': 30}, 'min_epochs 3': {'min_epochs': 3}, 'patience 4': {'patience': 4},
           'weight_decay 0': {'weight_decay': 0.0}, 'init legacy': {'init': 'legacy'}, 'schema': {'schema': 'g-b1-run-v1'},
           'test gate open': {'test_gate': 'open'}, 'bool as layer count': {'encoder_layers': True}, 'int 0 as float dropout': {'dropout': 0},
           'float d_model': {'d_model': 128.0}, 'seed 7': {'seed': 7}, 'microbatch 64': {'microbatch': 64}, 'extra field': {'x': 1}}
    for label, patch in bad.items():
        expect_rejected(lambda patch=patch: hp.validate_config(hp_config(**patch)), label)
    missing = hp_config()
    del missing['head_dim']
    expect_rejected(lambda: hp.validate_config(missing), 'missing field')
    expect_rejected(lambda: hp.validate_config([]), 'not a dict')
    small = hp_config(d_model=64, heads=3, head_dim=16, encoder_layers=2, decoder_layers=2, ff=128, dropout=0.0, microbatch=2, max_epochs=3, min_epochs=1,
                      learning_rate=.0006)
    accepted(lambda: hp.validate_config(small, engineering=True), 'engineering small shape')
    expect_rejected(lambda: hp.validate_config(small), 'engineering shape in a formal run')
    for label, patch in {'history 100': {'history_len': 100}, 'random SID': {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'}, 'd_model 512': {'d_model': 512},
                         'head_dim 24': {'head_dim': 24}, 'layers 5': {'encoder_layers': 5}, 'ff 4096': {'ff': 4096}, 'dropout 0.9': {'dropout': 0.9},
                         'lr 1e-3': {'learning_rate': .001}, 'heads 9': {'heads': 9}, 'min > max epochs': {'min_epochs': 4}}.items():
        expect_rejected(lambda patch=patch: hp.validate_config({**small, **patch}, engineering=True), f'engineering relaxation must not allow {label}')


def check_registry_isolation():
    """GARCH and G-HP registries never accept each other's configs (runs cannot cross)."""
    from vkgr import generative_arch as ga
    from vkgr import generative_hp_arch as hp
    for gid in GRUNS:
        expect_rejected(lambda gid=gid: ga.validate_config(grun_config(gid)), f'G-ARCH registry must reject {gid}')
    for path in (A1_PATH, A2_PATH):
        expect_rejected(lambda path=path: hp.validate_config(json.loads(path.read_text())), f'G-HP registry must reject {path.name}')


def check_committed_configs():
    """T-OFAT: exactly five files; each differs from the A1 file only by task_id + its knob fields, with the frozen values."""
    from vkgr import generative_hp_arch as hp
    from vkgr.tiger_arch import expected_param_count
    a1 = a1_config()
    paths = sorted(CONFIG_DIR.glob('*.json'))
    ok([p.stem for p in paths] == sorted(r[0] for r in GRUNS.values()), f'configs/ghp01 must hold exactly the five G-HP runs: {[p.name for p in paths]}')
    for gid, (run_id, knob, fields, params) in GRUNS.items():
        path = CONFIG_DIR / f'{run_id}.json'
        cfg = json.loads(path.read_text())
        ok(path.read_text() == json.dumps(cfg, indent=1, sort_keys=True) + '\n', f'{gid}: config must use the configs/ style (sorted keys, indent 1)')
        ok(set(cfg) == set(a1), f'{gid}: field set differs from A1')
        diff = {k for k in a1 if cfg[k] != a1[k] or type(cfg[k]) is not type(a1[k])}
        ok(diff == {'task_id'} | set(hp.KNOBS[knob]), f'{gid}: one-factor-at-a-time violated, changed {sorted(diff)}')
        ok(set(fields) == set(hp.KNOBS[knob]) and all(cfg[k] == v and type(cfg[k]) is type(v) for k, v in fields.items()), f'{gid}: knob values differ from the frozen table')
        ok(cfg['task_id'] == 'G-HP-01', f'{gid}: task_id')
        accepted(lambda cfg=cfg: hp.validate_config(cfg), gid)
        ok(expected_param_count(V, *(cfg[k] for k in ('d_model', 'heads', 'head_dim', 'encoder_layers', 'decoder_layers', 'ff'))) == params, f'{gid}: closed-form params')


# ---- H: factory and optimizer ---------------------------------------------------------------------------------
def _layout_equal(a, b, label):
    from vkgr import generative_evaluation as ev
    ok(list(a.state_dict()) == list(b.state_dict()), f'{label}: state_dict keys/order differ')
    ok([tuple(v.shape) for v in a.state_dict().values()] == [tuple(v.shape) for v in b.state_dict().values()], f'{label}: shapes differ')
    ok(ev.model_digest(a) == ev.model_digest(b), f'{label}: same-seed initialisation differs')


def check_a1_factory_equivalence():
    """H1: at A1 values the new factory equals ga.make_model_arch and a direct TigerArchModel, value by value (init small and legacy)."""
    from vkgr import generative_arch as ga
    from vkgr import generative_hp_arch as hp
    from vkgr.generative_hp import init_embeddings
    from vkgr.tiger_arch import TigerArchModel
    for init in ('small', 'legacy'):
        new = _seeded(hp.make_model_hp, CODEC, hp_config(init=init), device='cpu')
        old = _seeded(ga.make_model_arch, CODEC, {**a1_config(), 'init': init}, device='cpu')
        direct = _seeded(TigerArchModel, V, 128, 6, 64, 4, 4, 1024, 0.1)
        if init == 'small':
            init_embeddings(direct)
        _layout_equal(new, old, f'init {init} vs ga.make_model_arch')
        _layout_equal(new, direct, f'init {init} vs direct TigerArchModel')
        ok(new.init_report == old.init_report, f'init {init}: init_report differs')
        ok(new.init_report['arch']['spec'] == old.init_report['arch']['spec'], f'init {init}: arch spec differs')


def check_single_knob_init():
    """H2: G2/G3/G5 do not change the structure -> same-seed initialisation equals A1."""
    from vkgr import generative_evaluation as ev
    from vkgr import generative_hp_arch as hp
    reference = ev.model_digest(_seeded(hp.make_model_hp, CODEC, hp_config(), device='cpu'))
    for gid in ('G2', 'G3', 'G5'):
        ok(ev.model_digest(_seeded(hp.make_model_hp, CODEC, grun_config(gid), device='cpu')) == reference, f'{gid}: initialisation differs from A1')


def check_shapes():
    """H3 + measured T-PARAMS: the factory builds exactly the configured width / heads / layers."""
    from vkgr import generative_hp_arch as hp
    from vkgr.tiger_arch import expected_param_count
    for gid, (_, _, _, params) in GRUNS.items():
        model = _seeded(hp.make_model_hp, CODEC, grun_config(gid), device='cpu')
        cfg = grun_config(gid)
        total = sum(p.numel() for p in model.parameters())
        ok(total == params == model.init_report['arch']['params_total'], f'{gid}: measured params {total} != {params}')
        ok(total == expected_param_count(V, cfg['d_model'], cfg['heads'], cfg['head_dim'], cfg['encoder_layers'], cfg['decoder_layers'], cfg['ff']), f'{gid}: closed form')
        enc, dec = model.transformer.encoder.layers, model.transformer.decoder.layers
        ok((len(enc), len(dec)) == (cfg['encoder_layers'], cfg['decoder_layers']), f'{gid}: layer counts {len(enc)}+{len(dec)}')
        d, inner, ff = cfg['d_model'], cfg['heads'] * cfg['head_dim'], cfg['ff']
        ok(tuple(model.embedding.weight.shape) == (V, d) and tuple(model.output.weight.shape) == (V, d), f'{gid}: embedding/output width')
        for attn in attentions(model):
            ok((attn.heads, attn.head_dim) == (cfg['heads'], cfg['head_dim']), f'{gid}: heads x head_dim')
            ok(tuple(attn.in_proj_weight.shape) == (3 * inner, d) and tuple(attn.out_proj.weight.shape) == (d, inner), f'{gid}: attention projections')
        for layer in list(enc) + list(dec):
            ok(tuple(layer.linear1.weight.shape) == (ff, d) and tuple(layer.linear2.weight.shape) == (d, ff), f'{gid}: feed-forward shape')
        if gid == 'G1':
            ok(tuple(enc[0].self_attn.in_proj_weight.shape) == (2304, 256) and tuple(enc[0].self_attn.out_proj.weight.shape) == (256, 768), 'G1: 6 x 128 heads on d_model 256')
            source, decoder = ac.batch()
            model.train()
            logits = model(source, decoder)
            ok(tuple(logits.shape) == (source.shape[0], 4, V) and bool(torch.isfinite(logits).all()), 'G1: forward shape / finite')
            logits.sum().backward()
            ok(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'G1: finite gradients')


def check_optimizer():
    """H4: AdamW with the configured LR and wd 1e-4; equals the GARCH optimizer at 3e-4."""
    from vkgr import generative_arch as ga
    from vkgr import generative_hp_arch as hp
    from vkgr.tiger_arch import TigerArchModel
    model = TigerArchModel(V, 16, 2, 8, 1, 1, 32, 0.0)
    ref = ga.make_optimizer_arch(model, a1_config())
    got = hp.make_optimizer_hp(model, hp_config())
    ok(type(ref) is type(got) and len(ref.param_groups) == len(got.param_groups) == 1, 'optimizer type / groups')
    strip = lambda g: {k: v for k, v in g.items() if k != 'params'}
    ok(strip(ref.param_groups[0]) == strip(got.param_groups[0]), 'optimizer hyper-parameters differ from GARCH at lr 3e-4')
    ok(len(got.param_groups[0]['params']) == len(list(model.parameters())), 'optimizer parameter count')
    for lr in (.0001, .0003, .0006):
        group = hp.make_optimizer_hp(model, hp_config(learning_rate=lr)).param_groups[0]
        ok(group['lr'] == lr and group['weight_decay'] == 1e-4, f'optimizer lr/wd wrong at lr {lr}: {group["lr"]} / {group["weight_decay"]}')
    expect_rejected(lambda: hp.make_optimizer_hp(model, hp_config(learning_rate=.001)), 'lr 1e-3')
    expect_rejected(lambda: hp.make_optimizer_hp(model, hp_config(weight_decay=0.0)), 'weight decay 0')
    model.double()
    expect_rejected(lambda: hp.make_optimizer_hp(model, hp_config()), 'non-FP32 parameters')


# ---- D: dropout -----------------------------------------------------------------------------------------------
def check_dropout_a1(context):
    """D1: dropout 0.1 under G-HP behaves exactly like A1 (train-mode logits, loss, gradients and 3 frozen training steps)."""
    from vkgr import generative_arch as ga
    from vkgr import generative_evaluation as ev
    from vkgr import generative_hp_arch as hp
    from vkgr import generative_training as gt
    new = _seeded(hp.make_model_hp, CODEC, hp_config(), device='cpu')
    old = _seeded(ga.make_model_arch, CODEC, a1_config(), device='cpu')
    ok(len(dropouts(new)) == DROPOUT_MODULES_4X4 and all(m.p == 0.1 for m in dropouts(new)), 'A1 values: 28 dropout modules with p 0.1')
    ok(all(a.dropout_p == 0.1 for a in attentions(new)), 'A1 values: attention dropout 0.1')
    source, decoder = ac.batch()
    target = decoder.roll(-1, 1)
    new.train()
    old.train()
    torch.manual_seed(7)
    a = new(source, decoder)
    torch.manual_seed(7)
    b = old(source, decoder)
    same(a, b, 'train-mode logits differ from A1 under the same RNG')
    la, lb = F.cross_entropy(a.flatten(0, 1), target.flatten()), F.cross_entropy(b.flatten(0, 1), target.flatten())
    same(la, lb, 'loss differs from A1')
    la.backward()
    lb.backward()
    for (name, p), (_, q) in zip(new.named_parameters(), old.named_parameters()):
        same(p.grad, q.grad, f'gradient differs from A1: {name}')
    new = _seeded(hp.make_model_hp, context.codec, hp_config(), device='cpu')
    old = _seeded(ga.make_model_arch, context.codec, a1_config(), device='cpu')
    opt_new, opt_old = hp.make_optimizer_hp(new, hp_config()), ga.make_optimizer_arch(old, a1_config())
    rows = context.examples.rows(list(range(len(context.examples))))
    for step in range(1, 4):
        torch.manual_seed(100 + step)
        ra = gt.train_effective_batch(new, opt_new, rows, context.codec, microbatch=2, device='cpu')
        torch.manual_seed(100 + step)
        rb = gt.train_effective_batch(old, opt_old, rows, context.codec, microbatch=2, device='cpu')
        ok(ra.loss_mean == rb.loss_mean, f'loss differs from A1 at step {step}')
        ok(ev.model_digest(new) == ev.model_digest(old), f'weights diverge from A1 at step {step}')


def check_dropout_zero(check_rng=True):
    """D2: G5 (dropout 0) disables every dropout: train == eval, RNG-independent, RNG untouched."""
    from vkgr import generative_hp_arch as hp
    model = _seeded(hp.make_model_hp, CODEC, grun_config('G5'), device='cpu')
    ok(len(dropouts(model)) == DROPOUT_MODULES_4X4 and all(m.p == 0.0 for m in dropouts(model)), 'G5: all 28 dropout modules must have p 0.0')
    ok(all(a.dropout_p == 0.0 for a in attentions(model)), 'G5: attention dropout must be 0.0')
    source, decoder = ac.batch()
    model.train()
    torch.manual_seed(1)
    a = model(source, decoder)
    torch.manual_seed(2)
    same(a, model(source, decoder), 'G5: train passes depend on the RNG (dropout active)')
    model.eval()
    with torch.no_grad():
        same(a.detach(), model(source, decoder), 'G5: train-mode logits differ from eval-mode logits')
    if check_rng:
        model.train()
        state = torch.get_rng_state()
        model(source, decoder).sum().backward()
        same(state, torch.get_rng_state(), 'G5: a train forward/backward consumed RNG')


def check_dropout_contrast():
    """D3: the D2 assertions discriminate - at A1 (dropout 0.1) the same comparisons are unequal."""
    from vkgr import generative_hp_arch as hp
    model = _seeded(hp.make_model_hp, CODEC, hp_config(), device='cpu')
    source, decoder = ac.batch()
    model.train()
    torch.manual_seed(1)
    a = model(source, decoder)
    torch.manual_seed(2)
    ok(not torch.equal(a, model(source, decoder)), 'A1: different RNG seeds must give different train passes')
    model.eval()
    with torch.no_grad():
        ok(not torch.equal(a.detach(), model(source, decoder)), 'A1: train and eval must differ')
