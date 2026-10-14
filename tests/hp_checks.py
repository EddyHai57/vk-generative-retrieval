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


# ---- E: entry point -------------------------------------------------------------------------------------------
RUNNER_GLOBALS = ac.RUNNER_GLOBALS  # ('B1_MEMBERS', '_evaluate', 'make_model_b1', 'validate_config', 'gt')
HP_FILES = ('src/vkgr/generative_hp_arch.py', 'src/vkgr/generative_b1_hp.py')


def entry_config(**overrides):
    """Engineering-size G-HP config: 3 heads x 16 on d_model 64 (64 % 3 != 0), 2+2 layers, dropout 0, LR 6e-4."""
    return hp_config(**{'d_model': 64, 'heads': 3, 'head_dim': 16, 'encoder_layers': 2, 'decoder_layers': 2, 'ff': 128, 'dropout': 0.0,
                        'learning_rate': .0006, 'microbatch': 2, 'max_epochs': 3, 'min_epochs': 1, **overrides})


def check_binding_members(value):
    from vkgr import generative_b1_arch as gba
    files = value['binding']['source']['files']
    ok(list(files) == list(gba.MEMBERS + HP_FILES), f'G-HP binding must cover the 18 GARCH members + the 2 G-HP files, got {len(files)}')
    ok(len(files) == 20, 'G-HP binding size')
    ok(value['binding']['config']['task_id'] == 'G-HP-01', 'binding config task_id')


def check_globals_restored():
    """The G-HP runner patches generative_b1_run only for the duration of the call, also on the exception path."""
    from vkgr import generative_b1_hp as bh
    from vkgr import generative_b1_run as b1
    from vkgr import generative_hp_arch as hp
    before = {n: getattr(b1, n) for n in RUNNER_GLOBALS}

    def restored(label):
        for n in RUNNER_GLOBALS:
            ok(getattr(b1, n) is before[n], f'{label}: generative_b1_run.{n} was not restored')
    with bh._runner(None):
        ok(b1.B1_MEMBERS == bh.MEMBERS and b1.validate_config is hp.validate_config and b1.make_model_b1 is hp.make_model_hp, 'the runner did not install the G-HP hooks')
        from vkgr.tiger_arch import TigerArchModel
        probe = TigerArchModel(V, 16, 2, 8, 1, 1, 32, 0.0)
        ok(b1.gt.make_optimizer(probe, hp_config(learning_rate=.0006)).param_groups[0]['lr'] == .0006, 'training proxy must use the G-HP optimizer (LR 6e-4)')
        ok(b1.gt.train_effective_batch is before['gt'].train_effective_batch, 'training proxy must delegate the frozen training step')
    restored('normal exit')
    try:
        with bh._runner({'mode': 'batched', 'batch_size': 1}):
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    restored('exception exit')


def check_gates():
    """Workers > 1, batched decode with batch != 1 and any subcommand other than train/verify fail before any output."""
    import contextlib
    import io
    import os
    from test_generative_batched import tmpdir
    from unittest import mock
    from vkgr import generative_b1_hp as bh
    from vkgr import generative_sharded as gs
    expect_rejected(lambda: bh._runner(None, 2).__enter__(), 'sharded evaluation workers')
    for size in (2, 32):
        expect_rejected(lambda size=size: bh._runner({'mode': 'batched', 'batch_size': size}).__enter__(), f'batched decode batch {size}')

    def cli(head, env):
        clean = {k: v for k, v in os.environ.items() if not k.startswith('VKGR_')}
        clean.update(env)
        with tmpdir() as tmp, mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(gs, 'cpu_quota_cores', return_value=None):
            argv = head + ['--prepared', 'x', '--prepared-sha', '0' * 64, '--history-manifest', 'y', '--history-sha', '0' * 64,
                           '--config', 'z', '--config-sha', '0' * 64, '--output', f'{tmp}/out', '--scratch', f'{tmp}/s']
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = bh.main(argv)
            return code, err.getvalue(), os.path.exists(f'{tmp}/out')
    for label, head, env in (('workers 2', ['train'], {'VKGR_EVAL_WORKERS': '2'}), ('batched without batch size', ['train'], {'VKGR_DECODE': 'batched'}),
                             ('batched batch 2', ['train'], {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '2'}), ('beam-diagnostic', ['beam-diagnostic'], {}), ('no subcommand', [], {})):
        code, err, out = cli(head, env)
        ok(code == 1 and 'G-HP-01 gate failed' in err and not out, f'{label}: must exit 1 at the G-HP-01 gate before output (code {code}, output {out}): {err.strip()[:120]}')
    code, err, out = cli(['train'], {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '1'})
    ok(code == 1 and 'G-HP-01 gate failed' not in err and 'G-B1 gate failed' in err, 'batch size 1 must pass the G-HP gate (and fail later on the bogus inputs)')


# ---- P: scripts (gate, combo, parity) -------------------------------------------------------------------------
def load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q = 2520


def _r(hits):
    return hits / Q


def gate_inputs(hits=196, n50=0.03):
    train_sha = 'a' * 64
    manifest = {'status': 'complete', 'selection': {'best_epoch': 7, 'best_metric': [_r(hits), n50], 'epoch': 10},
                'epochs': [{} for _ in range(10)], 'wall_seconds': {'train': 100.0, 'evaluate': 50.0}, 'hit_max_epochs': False}
    return {'train_terminal': {'status': 'succeeded'}, 'train_manifest': manifest, 'train_manifest_sha256': train_sha,
            'verify_terminal': {'status': 'succeeded'}, 'verify_last_line': {'status': 'verified', 'best_epoch': 7, 'epochs': 10},
            'verify_run': {'argv': ['python', '-m', 'vkgr.generative_b1_arch', 'verify', '--manifest', 'm.json', '--manifest-sha', train_sha]}}


def check_gate(gate):
    """P2: the A1 entry gate passes only with train succeeded + verified (same best epoch, same manifest sha) + >= 196 hits."""
    ok(gate.MIN_HITS == 196 and gate.QUERIES == Q, 'gate threshold must be 196 of 2520 hits')
    value = gate.decide(**gate_inputs())
    ok(value['passed'] is True and value['hits'] == 196 and value['threshold_hits'] == 196, f'all checks satisfied must pass: {value}')
    ok(value['checks'] == {'train_succeeded': True, 'verified': True, 'r50_ok': True}, 'check flags')
    ok((value['best_epoch'], value['epochs'], value['wall_seconds'], value['hit_max_epochs']) == (7, 10, {'train': 100.0, 'evaluate': 50.0}, False), 'reported epochs / wall')
    ok(value['missing'] == [], 'missing list')
    ok(gate.decide(**gate_inputs(hits=195))['passed'] is False, '195 hits (0.077381 < 0.077397) must fail')
    ok(gate.decide(**gate_inputs(hits=195))['checks']['r50_ok'] is False, '195 hits: r50_ok')
    ok(gate.decide(**gate_inputs(hits=250))['passed'] is True, '250 hits must pass')

    def broken(label, **patch):
        value = {**gate_inputs(), **patch}
        result = gate.decide(**value)
        ok(result['passed'] is False, f'{label} must fail the gate: {result}')
        return result
    broken('train failed', train_terminal={'status': 'failed'})
    broken('manifest running', train_manifest={**gate_inputs()['train_manifest'], 'status': 'running'})
    ok(broken('verify missing', verify_terminal=None, verify_last_line=None, verify_run=None)['missing'] != [], 'missing verify must be listed')
    broken('verify job failed', verify_terminal={'status': 'failed'})
    broken('verify not verified', verify_last_line={'status': 'error', 'best_epoch': 7})
    broken('verify best epoch differs', verify_last_line={'status': 'verified', 'best_epoch': 6})
    broken('verify of another manifest', verify_run={'argv': ['--manifest-sha', 'b' * 64]})
    ok(broken('train missing', train_terminal=None, train_manifest=None)['missing'] != [], 'missing train must be listed')
    expect_rejected(lambda: gate.hits(0.0774), 'r50 that is not an integer hit count over 2520')


def combo_case(won, n50=None, status=None, verified=None, anchor_hits=200):
    """won: id -> delta hits vs the anchor; every other G run gets delta 0."""
    from vkgr import generative_hp_arch as hp
    anchor_cfg = a1_config()
    runs = []
    for gid, (run_id, _, fields, _) in GRUNS.items():
        delta = won.get(gid, 0)
        runs.append({'id': gid, 'run_id': run_id, 'status': (status or {}).get(gid, 'succeeded'), 'verified': (verified or {}).get(gid, True),
                     'r50': _r(anchor_hits + delta), 'n50': (n50 or {}).get(gid, 0.03), 'config': grun_config(gid)})
    ok(all(r['config']['task_id'] == hp.TASK_ID for r in runs), 'fixture task id')
    return {'run_id': 'g-garch01-a1-lr3e-4-s42', 'r50': _r(anchor_hits), 'n50': 0.03, 'verified': True, 'config': anchor_cfg}, runs


def check_combo(combo):
    """P3: >= 2 distinct knobs each winning by >= 6 hits trigger one combination built from A1 + the best level per knob."""
    from vkgr import generative_hp_arch as hp
    ok(combo.MIN_DELTA_HITS == 6 and combo.QUERIES == Q, 'win threshold must be 6 of 2520 hits (= dR@50 0.002)')
    ok(tuple(combo.HP_SHAPES) == hp.HP_SHAPES and tuple(combo.LEARNING_RATES) == hp.LEARNING_RATES, 'combo constants differ from generative_hp_arch')
    ok({k: tuple(v) for k, v in combo.KNOBS.items()} == hp.KNOBS, 'combo knob fields differ from generative_hp_arch')
    ok(combo.KNOB_OF == {'G1': 'width', 'G2': 'lr', 'G3': 'lr', 'G4': 'layers', 'G5': 'dropout'}, 'knob groups')
    value = combo.decide(*combo_case({}))
    ok(value['triggered'] is False and value['knobs_won'] == [] and value['combo_config'] is None, 'no winner must not trigger')
    ok(value['judged_on'] == ['G1', 'G2', 'G3', 'G4', 'G5'] and value['pending'] == ['G6'] and value['status'] == 'provisional', 'provisional judgement on G1-G5')
    value = combo.decide(*combo_case({'G2': 10, 'G3': 8}))
    ok(value['triggered'] is False and value['knobs_won'] == ['lr'], 'G2 + G3 are one knob (learning rate): must not trigger')
    ok(combo.decide(*combo_case({'G1': 5, 'G5': 9}))['triggered'] is False, '5 hits is not a win')
    value = combo.decide(*combo_case({'G1': 6, 'G5': 9}))
    ok(value['triggered'] is True and value['knobs_won'] == ['width', 'dropout'], 'G1 + G5 at >= 6 hits must trigger')
    cfg = value['combo_config']
    a1 = a1_config()
    diff = {k for k in a1 if cfg[k] != a1[k]}
    ok(diff == {'task_id', 'd_model', 'ff', 'head_dim', 'dropout'}, f'combo must change exactly width + dropout fields: {sorted(diff)}')
    ok(tuple(cfg[k] for k in hp.SHAPE_KEYS) in hp.HP_SHAPES and cfg['seed'] == 42 and cfg['task_id'] == 'G-HP-01', 'combo structure registered')
    hp.validate_config(cfg)
    import hashlib
    ok(value['combo_config_sha256'] == hashlib.sha256((json.dumps(cfg, indent=1, sort_keys=True) + '\n').encode()).hexdigest(), 'combo config sha')
    value = combo.decide(*combo_case({'G2': 8, 'G3': 12, 'G4': 7}))
    ok(value['triggered'] and value['combo_config']['learning_rate'] == .0006 and value['combo_config']['encoder_layers'] == 2, 'LR level must be the higher-R@50 winner (G3)')
    value = combo.decide(*combo_case({'G2': 8, 'G3': 8, 'G4': 7}, n50={'G2': 0.04, 'G3': 0.03}))
    ok(value['combo_config']['learning_rate'] == .0001, 'R@50 tie on the LR knob: higher N@50 (G2) wins')
    value = combo.decide(*combo_case({'G2': 8, 'G3': 8, 'G4': 7}))
    ok(value['combo_config']['learning_rate'] == .0001, 'full tie on the LR knob: enumeration order (G2) wins')
    value = combo.decide(*combo_case({'G1': 9, 'G5': 9}, verified={'G1': False}))
    ok(value['triggered'] is False and 'G1' in value['excluded'], 'unverified run must not win and must be excluded')
    value = combo.decide(*combo_case({'G1': 9, 'G5': 9}, status={'G5': 'failed'}))
    ok(value['triggered'] is False and 'G5' in value['excluded'], 'failed run must not win and must be excluded')
    anchor, runs = combo_case({})
    expect_rejected(lambda: combo.decide({**anchor, 'verified': False}, runs), 'unverified anchor')
    expect_rejected(lambda: combo.decide({**anchor, 'r50': 0.0774}, runs), 'non-integer anchor hits')
    # card 1.5 selection over anchor + verified G1-G5: R@50 -> N@50 -> enumeration order
    ok(combo.decide(*combo_case({}))['selection'] == 'anchor', 'all tied: anchor wins by enumeration order')
    ok(combo.decide(*combo_case({'G4': 3}))['selection'] == 'G4', 'highest R@50 is selected even below the win threshold')
    ok(combo.decide(*combo_case({'G2': 3, 'G4': 3}, n50={'G4': 0.05}))['selection'] == 'G4', 'R@50 tie: higher N@50')
    ok(combo.decide(*combo_case({'G2': 3, 'G4': 3}))['selection'] == 'G2', 'R@50 + N@50 tie: enumeration order')
    ok(combo.decide(*combo_case({'G4': 3}, verified={'G4': False}))['selection'] == 'anchor', 'unverified run is not a selection candidate')


PH1_KEYS = {'init_equal', 'train_steps', 'train_steps_equal', 'first_diff_step', 'max_abs_param_diff', 'eval_logits_equal', 'queries',
            'r50_reference', 'r50_reloaded', 'reload_hits_delta'}
PH2_KEYS = {'queries', 'equal', 'serial_wall_seconds', 'batched_b1_wall_seconds', 'peak_vram_bytes'}
PH3_KEYS = {'steps', 'finite', 'error', 'step_seconds_mean', 'peak_vram_bytes', 'params_total', 'params_expected'}


def parity_world(tmp):
    """Engineering world + a GARCH-entry reference run (the A1 role) + three engineering G-HP configs + a callable running the script.

    Configs: 'wide' (new structure -> PH2), 'lr' (LR only) and 'drop0' (dropout only): the last two share the reference structure.
    """
    import contextlib
    import io
    from test_generative_arch import arch_config
    from test_generative_b1 import engineering_world
    from vkgr import generative_b1_arch as ba
    from vkgr.data_audit import file_sha256
    script = load_script('ghp01_parity')
    _, receipt, manifest, sha = engineering_world(tmp)
    common = (ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha)
    reference = arch_config(learning_rate=.0003)
    value = ba.run(*common, reference, Path(tmp) / 'run', temporary=Path(tmp) / 's-ref')
    configs = {'wide': {'d_model': 128, 'head_dim': 32, 'ff': 256}, 'lr': {'learning_rate': .0006}, 'drop0': {'dropout': 0.0}}
    paths = []
    for name, patch in configs.items():
        path = Path(tmp) / f'{name}.json'
        path.write_text(json.dumps({**reference, 'task_id': 'G-HP-01', **patch}, indent=1, sort_keys=True) + '\n')
        paths.append(str(path))

    def call(scratch, steps='3', manifest_sha=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = script.main(['--prepared', str(receipt['manifest_path']), '--prepared-sha', receipt['manifest_sha256'], '--history-manifest', str(manifest),
                                '--history-sha', sha, '--manifest', str(Path(tmp) / 'run' / 'manifest.json'),
                                '--manifest-sha', manifest_sha or file_sha256(Path(tmp) / 'run' / 'manifest.json'),
                                '--scratch', str(Path(tmp) / scratch), '--train-steps', steps, '--configs', ','.join(paths)], root=ROOT)
        return code, out.getvalue(), err.getvalue()
    return script, value, call


def check_parity_report(call, scratch='ph-equal'):
    """P1: identical factories are reported equal, every gate field present, one JSON line, no per-query rows."""
    code, out, err = call(scratch)
    ok(code == 0, f'parity script failed: {err}')
    ok(len(out.strip().splitlines()) == 1 and '"items"' not in out, 'one JSON line without per-query rows')
    report = json.loads(out)
    ok(set(report) == {'ph1_a1_factory', 'ph2_decode', 'ph3_train'}, f'report sections {sorted(report)}')
    ph1, ph2, ph3 = report['ph1_a1_factory'], report['ph2_decode'], report['ph3_train']
    ok(set(ph1) == PH1_KEYS, f'PH1 keys {sorted(ph1)}')
    ok(ph1['init_equal'] is True and ph1['train_steps_equal'] is True and ph1['first_diff_step'] is None and ph1['eval_logits_equal'] is True, f'PH1 must report equal: {ph1}')
    ok((ph1['train_steps'], ph1['max_abs_param_diff'], ph1['reload_hits_delta']) == (3, 0.0, 0), f'PH1 numbers: {ph1}')
    ok(ph1['r50_reloaded'] == ph1['r50_reference'], 'reloaded best must reproduce the reference R@50 exactly on CPU')
    ok(set(ph2) == {'wide'}, f'PH2 must test only configs whose structure differs from the reference: {sorted(ph2)}')
    ok(set(ph2['wide']) == PH2_KEYS and ph2['wide']['equal'] is True and ph2['wide']['queries'] == ph1['queries'], f'PH2: {ph2}')
    ok(set(ph3) == {'per_config', 'runtime'} and set(ph3['per_config']) == {'wide', 'lr', 'drop0'}, f'PH3 sections: {sorted(ph3)}')
    for name, entry in ph3['per_config'].items():
        ok(set(entry) == PH3_KEYS and entry['finite'] is True and (entry['steps'], entry['error']) == (3, None), f'PH3 {name}: {entry}')
        ok(entry['params_total'] == entry['params_expected'], f'PH3 {name}: params')
    return report


def check_parity_perturbed(call, scratch='ph-perturbed'):
    """P1: a perturbed new factory is reported unequal, not hidden."""
    from unittest import mock
    from vkgr import generative_hp_arch as hp
    real = hp.make_model_hp

    def perturbed(codec, config, *, device):
        model = real(codec, config, device=device)
        with torch.no_grad():
            model.output.bias.add_(1e-3)
        return model
    with mock.patch.object(hp, 'make_model_hp', perturbed):
        code, out, err = call(scratch)
    ok(code == 0, f'parity script failed: {err}')
    ph1 = json.loads(out)['ph1_a1_factory']
    ok(ph1['init_equal'] is False and ph1['train_steps_equal'] is False and ph1['first_diff_step'] == 1 and ph1['max_abs_param_diff'] > 0.0, f'perturbation hidden: {ph1}')
