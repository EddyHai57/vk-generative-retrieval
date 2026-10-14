"""GARCH-01 guards shared by the plain tests and the mutation tests (this is not a test module).

Each guard raises AssertionError (or lets a production error escape) when a property is violated, so
test_generative_arch_mutations.py can inject a defect into the production code and demand that the very same
guard turns red. Numerical contract (see docs/garch01_switches.md):
  * train mode, and eval with the stock fast path disabled: bit-identical to the stock nn.Transformer model (torch.equal);
  * eval with the stock fast path enabled (what the anchor evaluation really runs): tolerance 1e-5 only, because the
    fused stock kernel computes the same maths in another fp32 order and cannot express independent head dims.
"""
import contextlib
import math

import torch
from torch import nn
from torch.nn import functional as F

V = 80  # real G codec vocabulary: 2 + 16 + 16 + 16 + 30
ANCHOR = dict(d_model=128, heads=4, head_dim=32, enc=1, dec=1, ff=512)
MULTI = dict(d_model=128, heads=4, head_dim=32, enc=2, dec=2, ff=256)
PAPER = dict(d_model=128, heads=6, head_dim=64, enc=4, dec=4, ff=1024, dropout=0.1)


def ok(condition, message):
    if not condition:
        raise AssertionError(message)


def same(a, b, message):
    ok(torch.equal(a, b), message)


@contextlib.contextmanager
def stock_fastpath(enabled):
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(enabled)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def pair(shape, seed=7, vocab=V):
    """Stock TigerModel and the independent-head model built from the same seed."""
    from vkgr.tiger import TigerModel
    from vkgr.tiger_arch import TigerArchModel
    torch.manual_seed(seed)
    stock = TigerModel(vocab, shape['d_model'], shape['heads'], shape['enc'], shape['dec'], shape['ff'])
    torch.manual_seed(seed)
    arch = TigerArchModel(vocab, shape['d_model'], shape['heads'], shape['head_dim'], shape['enc'], shape['dec'], shape['ff'], 0.0)
    return stock, arch


def batch(rows=6, width=40, seed=0):
    """Right-padded sources (never an empty row) plus a BOS-led decoder input."""
    g = torch.Generator().manual_seed(seed)
    source = torch.randint(2, V, (rows, width), generator=g)
    for row in range(1, rows):
        source[row, width - 4 * row:] = 0
    decoder = torch.cat([torch.ones(rows, 1, dtype=torch.long), torch.randint(2, V, (rows, 3), generator=g)], 1)
    return source, decoder


# ---- E: equivalence with the stock model --------------------------------------------------------------------
def check_state_layout(stock, arch):
    ok(list(stock.state_dict()) == list(arch.state_dict()), 'state_dict key order differs')
    ok([tuple(v.shape) for v in stock.state_dict().values()] == [tuple(v.shape) for v in arch.state_dict().values()], 'state_dict shapes differ')
    ok([n for n, _ in stock.named_parameters()] == [n for n, _ in arch.named_parameters()], 'parameter order differs')


def check_same_seed_init(stock, arch):
    from vkgr import generative_evaluation as ev
    ok(ev.model_digest(stock) == ev.model_digest(arch), 'same-seed initialisation differs from nn.Transformer')


def check_train_bitwise(stock, arch):
    arch.load_state_dict(stock.state_dict())
    source, decoder = batch()
    target = decoder.roll(-1, 1)
    stock.train()
    arch.train()
    a, b = stock(source, decoder), arch(source, decoder)
    same(a, b, 'train-mode logits differ')
    F.cross_entropy(a.flatten(0, 1), target.flatten()).backward()
    F.cross_entropy(b.flatten(0, 1), target.flatten()).backward()
    for (name, p), (_, q) in zip(stock.named_parameters(), arch.named_parameters()):
        same(p.grad, q.grad, f'gradient differs: {name}')


def check_eval_bitwise_without_fastpath(stock, arch):
    arch.load_state_dict(stock.state_dict())
    source, decoder = batch()
    stock.eval()
    arch.eval()
    with stock_fastpath(False), torch.no_grad():
        same(stock(source, decoder), arch(source, decoder), 'eval logits differ (stock fast path off)')
        m1, p1 = stock.encode(source)
        m2, p2 = arch.encode(source)
        same(p1, p2, 'padding masks differ')
        same(m1[~p1], m2[~p2], 'encoder memory differs on non-padding positions')


def check_eval_tolerance_with_fastpath(stock, arch, tolerance=1e-5):
    """The ONLY tolerance in the equivalence suite (max |dlogit| <= 1e-5, same argmax everywhere).

    Why not bit-equal: with the stock fast path enabled (the anchor evaluation's real path) nn.Transformer runs a fused
    kernel that evaluates the same maths in another fp32 operation order and cannot express independent head dimensions,
    so it cannot be mimicked bit for bit. The bound is orders of magnitude below the logit gaps that decide a beam
    ranking; measured 1.25e-6 (1+1 layers) and 1.64e-6 (2+2 layers) on CPU.
    """
    arch.load_state_dict(stock.state_dict())
    source, decoder = batch()
    stock.eval()
    arch.eval()
    with torch.no_grad():  # stock fast path ON: the path the anchor evaluation actually runs
        a, b = stock(source, decoder), arch(source, decoder)
    delta = float((a - b).abs().max())
    ok(delta <= tolerance, f'fast-path logits differ by {delta}')
    same(a.argmax(-1), b.argmax(-1), 'argmax differs under the fast path')
    return delta


def check_training_trajectory(shape, context, steps=5):
    """Frozen gt.train_effective_batch + AdamW + clip: model digest equal after every step."""
    from vkgr import generative_evaluation as ev
    from vkgr import generative_training as gt
    stock, arch = pair(shape, vocab=context.codec.vocab_size)
    rows = context.examples.rows(list(range(len(context.examples))))
    opt_a = torch.optim.AdamW(stock.parameters(), lr=3e-4, weight_decay=1e-4)
    opt_b = torch.optim.AdamW(arch.parameters(), lr=3e-4, weight_decay=1e-4)
    for step in range(1, steps + 1):
        ra = gt.train_effective_batch(stock, opt_a, rows, context.codec, microbatch=2, device='cpu')
        rb = gt.train_effective_batch(arch, opt_b, rows, context.codec, microbatch=2, device='cpu')
        ok(ra.loss_mean == rb.loss_mean, f'loss differs at step {step}')
        ok(ev.model_digest(stock) == ev.model_digest(arch), f'weights diverge at step {step}')


def check_decode_history(shape, context, histories):
    from vkgr.tiger import decode_history
    stock, arch = pair(shape, vocab=context.codec.vocab_size)
    arch.load_state_dict(stock.state_dict())
    with stock_fastpath(False):
        for h in histories:
            ok(decode_history(stock, context.codec, h, 100) == decode_history(arch, context.codec, h, 100), 'candidate list/audit differ (fast path off)')
    for h in histories:  # fast path on: same candidate set and top-1 (order of near ties may differ at 1e-6)
        a, _ = decode_history(stock, context.codec, h, 100)
        b, _ = decode_history(arch, context.codec, h, 100)
        ok(sorted(a) == sorted(b) and a[0] == b[0], 'candidate set/top-1 differ (fast path on)')


def check_batched_decode(model, context, histories, sizes=(1, 3)):
    from vkgr import generative_batched as gb
    from vkgr.tiger import decode_history
    expected = [decode_history(model, context.codec, h, 100) for h in histories]
    for size in sizes:
        ok(gb.decode_histories_batched(model, context.codec, histories, 100, size) == expected, f'batched decode (batch {size}) differs from serial')


# ---- C: new capability (independent head dimension) ---------------------------------------------------------
def attention_oracle(attn, query, kv, pad=None, causal=False):
    """Float64 per-head reference of one attention call (dropout off); only reads the module's weights."""
    E, H, Dh = attn.inner, attn.heads, attn.head_dim
    W, b = attn.in_proj_weight.double(), attn.in_proj_bias.double()
    Wo, bo = attn.out_proj.weight.double(), attn.out_proj.bias.double()
    out = torch.zeros(query.shape[0], query.shape[1], Wo.shape[0], dtype=torch.float64)
    for i in range(query.shape[0]):
        heads = []
        for h in range(H):
            s = slice(h * Dh, (h + 1) * Dh)
            q = query[i].double() @ W[:E][s].T + b[:E][s]
            k = kv[i].double() @ W[E:2 * E][s].T + b[E:2 * E][s]
            v = kv[i].double() @ W[2 * E:][s].T + b[2 * E:][s]
            score = q @ k.T / math.sqrt(Dh)
            if pad is not None:
                score = score.masked_fill(pad[i][None, :], float('-inf'))
            if causal:
                score = score.masked_fill(torch.ones(score.shape, dtype=torch.bool).triu(1), float('-inf'))
            heads.append(torch.softmax(score, -1) @ v)
        out[i] = torch.cat(heads, -1) @ Wo.T + bo
    return out


def check_attention_oracle(attention_cls, tolerance=1e-5):
    torch.manual_seed(3)
    attn = attention_cls(128, 6, 64, 0.0)
    for p in (attn.in_proj_bias, attn.out_proj.bias):
        nn.init.normal_(p, 0, .1)
    attn.eval()
    x, memory = torch.randn(3, 11, 128), torch.randn(3, 9, 128)
    pad_x = torch.zeros(3, 11, dtype=torch.bool)
    pad_x[1, 8:] = True
    pad_m = torch.zeros(3, 9, dtype=torch.bool)
    pad_m[2, 5:] = True
    with torch.no_grad():
        cases = {'self+padding': (attn(x, x, pad_x), attention_oracle(attn, x, x, pad_x)),
                 'cross+padding': (attn(x, memory, pad_m), attention_oracle(attn, x, memory, pad_m)),
                 'self+causal': (attn(x, x, None, True), attention_oracle(attn, x, x, causal=True))}
    for name, (got, want) in cases.items():
        delta = float((got.double() - want).abs().max())
        ok(delta <= tolerance, f'{name}: attention differs from the float64 oracle by {delta}')


def check_independent_heads_run():
    from vkgr.tiger_arch import TigerArchModel
    try:
        nn.MultiheadAttention(128, 6)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('stock MultiheadAttention accepted 6 heads at d_model 128 (the reason for the custom module is gone)')
    torch.manual_seed(1)
    model = TigerArchModel(V, 128, 6, 64, 4, 4, 1024, 0.1)
    source, decoder = batch()
    logits = model(source, decoder)
    ok(tuple(logits.shape) == (6, 4, V) and bool(torch.isfinite(logits).all()), 'paper-scale logits shape/finite')
    F.cross_entropy(logits.flatten(0, 1), decoder.roll(-1, 1).flatten()).backward()
    ok(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'paper-scale gradients missing/non-finite')


def check_shapes_and_param_count():
    from vkgr.tiger import TigerModel
    from vkgr.tiger_arch import TigerArchModel, expected_param_count
    paper = TigerArchModel(V, 128, 6, 64, 4, 4, 1024, 0.1)
    attn = paper.transformer.encoder.layers[0].self_attn
    ok(tuple(attn.in_proj_weight.shape) == (1152, 128) and tuple(attn.in_proj_bias.shape) == (1152,), 'QKV projection must map d_model -> 3 x inner_dim')
    ok(tuple(attn.out_proj.weight.shape) == (128, 384), 'output projection must map inner_dim -> d_model')
    total = sum(p.numel() for p in paper.parameters())
    ok(total == expected_param_count(V, 128, 6, 64, 4, 4, 1024, 200), f'paper-scale parameter count {total} != closed form')
    ok(total == 4_532_816, f'paper-scale parameter count is {total}, expected 4532816 for V=80')
    anchor = TigerArchModel(V, 128, 4, 32, 1, 1, 512, 0.0)
    ok(sum(p.numel() for p in anchor.parameters()) == 509_520 == expected_param_count(V, 128, 4, 32, 1, 1, 512, 200), 'anchor parameter count')
    stock = TigerModel(V, 128, 4, 2, 2, 256)  # the closed form is calibrated on the stock model (inner_dim == d_model)
    ok(sum(p.numel() for p in stock.parameters()) == expected_param_count(V, 128, 4, 32, 2, 2, 256, 200), 'closed form does not match the stock model')


def check_causal_and_padding():
    from vkgr.tiger_arch import TigerArchModel
    torch.manual_seed(8)
    model = TigerArchModel(10, 16, 3, 8, 2, 2, 32, 0.0).eval()  # 3 heads x 8 = 24 != d_model 16
    src = torch.tensor([[2, 4, 6, 8, 0, 0, 0, 0]])
    a, b = torch.tensor([[1, 2, 4, 6]]), torch.tensor([[1, 3, 5, 7]])
    with torch.no_grad():
        x, y = model(src, a), model(src, b)
        same(x[:, 0], y[:, 0], 'decoder position 0 must not see later decoder tokens')
        ok(not torch.equal(x[:, 2], y[:, 2]), 'later positions must depend on their own input')
        for position in range(3):
            c = a.clone()
            c[:, position + 1:] = 9
            ok(torch.allclose(x[:, :position + 1], model(src, c)[:, :position + 1], atol=1e-7, rtol=1e-7), f'future leak at decoder position {position}')
        ok(torch.allclose(x, model(src[:, :4], a), atol=2e-6, rtol=2e-6), 'source padding changes the output')
    try:
        model.encode(torch.zeros(1, 4, dtype=torch.long))
    except ValueError as exc:
        ok('empty source' in str(exc), 'wrong error for an all-padding source')
    else:
        raise AssertionError('an all-padding source must raise ValueError')
    attention = model.transformer.encoder.layers[0].self_attn
    try:  # a padding mask plus the causal hint would silently drop causality; refuse instead
        attention(torch.zeros(1, 4, 16), torch.zeros(1, 4, 16), torch.zeros(1, 4, dtype=torch.bool), True)
    except ValueError:
        pass
    else:
        raise AssertionError('padding mask together with is_causal must raise ValueError')


def check_dropout():
    from unittest import mock
    from vkgr import tiger_arch
    from vkgr.tiger_arch import TigerArchModel
    torch.manual_seed(5)
    model = TigerArchModel(V, 128, 6, 64, 4, 4, 1024, 0.1)
    source, decoder = batch()
    ok(sum(isinstance(m, nn.Dropout) for m in model.modules()) == 4 * 3 + 4 * 4, 'dropout modules: 3 per encoder layer + 4 per decoder layer')
    model.eval()
    with torch.no_grad():
        same(model(source, decoder), model(source, decoder), 'eval must be deterministic')
        e = model(source, decoder)
    model.train()
    torch.manual_seed(11)
    a = model(source, decoder)
    torch.manual_seed(11)
    same(a, model(source, decoder), 'train must be reproducible under the same seed')
    ok(not torch.equal(a, model(source, decoder)), 'consecutive train passes must draw fresh dropout masks')
    ok(not torch.equal(a, e), 'train and eval must differ when dropout > 0')
    attn = tiger_arch.IndependentHeadAttention(128, 6, 64, 0.5)  # attention-probability dropout alone
    x = torch.randn(2, 9, 128)
    attn.train()
    torch.manual_seed(1)
    p = attn(x, x)
    torch.manual_seed(2)
    ok(not torch.equal(p, attn(x, x)), 'attention-probability dropout inactive in train mode')
    attn.eval()
    with torch.no_grad():
        same(attn(x, x), attn(x, x), 'attention-probability dropout active in eval mode')
    seen = []
    real = tiger_arch.sdpa

    def spy(q, k, v, mask, p, causal):
        seen.append(p)
        return real(q, k, v, mask, p, causal)
    with mock.patch.object(tiger_arch, 'sdpa', spy):
        model.train()
        model(source, decoder)
        train_p = set(seen)
        seen.clear()
        model.eval()
        with torch.no_grad():
            model(source, decoder)
        eval_p = set(seen)
    ok(train_p == {0.1} and eval_p == {0.0}, f'SDPA dropout_p must be the configured 0.1 in train and 0.0 in eval, saw {train_p} / {eval_p}')


def check_layers_independent():
    from vkgr.tiger_arch import TigerArchModel
    model = TigerArchModel(V, 128, 6, 64, 4, 4, 1024, 0.1)
    enc, dec = model.transformer.encoder.layers, model.transformer.decoder.layers
    ok(len(enc) == 4 and len(dec) == 4, 'layer count follows the configuration')
    for layers in (enc, dec):
        ok(len({id(layer) for layer in layers}) == len(layers), 'layers must be distinct modules')
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                ok(not torch.equal(layers[i].linear1.weight, layers[j].linear1.weight), 'layer weights must be initialised independently')
                ok(not torch.equal(layers[i].self_attn.in_proj_weight, layers[j].self_attn.in_proj_weight), 'attention weights must be initialised independently')
    ok(not ({id(p) for p in enc.parameters()} & {id(p) for p in dec.parameters()}), 'encoder and decoder must not share parameters')


def check_positions_and_untied_output():
    from vkgr.tiger_arch import TigerArchModel
    model = TigerArchModel(V, 128, 6, 64, 1, 1, 256, 0.0)
    ok(tuple(model.position.weight.shape) == (200, 128), 'absolute position table keeps the anchor 200 rows')
    model.embed(torch.ones(1, 200, dtype=torch.long))
    for bad in (torch.ones(1, 201, dtype=torch.long), torch.ones(1, 0, dtype=torch.long), torch.ones(4, dtype=torch.long)):
        try:
            model.embed(bad)
        except ValueError:
            continue
        raise AssertionError('invalid token sequence accepted')
    ok(model.output.weight is not model.embedding.weight and model.output.weight.data_ptr() != model.embedding.weight.data_ptr(), 'output head must not be tied to the embedding')
    tokens = torch.tensor([[2, 3, 0, 5]])
    expected = model.embedding(tokens) * math.sqrt(128) + model.position(torch.arange(4))[None]
    ok(torch.allclose(model.embed(tokens), expected), 'embed must be embedding * sqrt(d_model) + absolute position')


# ---- F: config registry, model factory, optimizer proxy -----------------------------------------------------
def expect_rejected(call, label):
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f'accepted but must be rejected: {label}')


def check_registry():
    from vkgr import generative_arch as ga
    base = ga.default_config()
    ga.validate_config(base)
    for shape in (ga.ANCHOR_SHAPE, ga.PAPER_SCALE):
        for lr in (3e-4, 1e-3):
            for seed in (42, 2024, 2025):
                for microbatch in (512, 256, 128):
                    ga.validate_config({**base, **dict(zip(ga.SHAPE_KEYS, shape)), 'learning_rate': lr, 'seed': seed, 'microbatch': microbatch})
    anchor = {**base, **dict(zip(ga.SHAPE_KEYS, ga.ANCHOR_SHAPE))}
    bad = {'unregistered structure (6x64, 2+2 layers)': {'encoder_layers': 2, 'decoder_layers': 2},
           'anchor shape with dropout 0.1': {**anchor, 'dropout': 0.1}, 'bool as layer count': {**anchor, 'encoder_layers': True},
           'float d_model': {'d_model': 128.0}, 'int dropout': {**anchor, 'dropout': 0},
           'history_len 20': {'history_len': 20}, 'history_len 100': {'history_len': 100}, 'random SID': {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'},
           'beam 200': {'beam': 200}, 'init legacy': {'init': 'legacy'}, 'max_epochs 30': {'max_epochs': 30}, 'min_epochs 3': {'min_epochs': 3},
           'patience 4': {'patience': 4}, 'lr 0.01': {'learning_rate': .01}, 'weight_decay 0': {'weight_decay': 0.0}, 'clip 2.0': {'gradient_clip_norm': 2.0},
           'effective_batch 256': {'effective_batch': 256}, 'bf16': {'precision': 'bf16'}, 'test gate open': {'test_gate': 'open'},
           'history version': {'history_version': 'HIST_V1'}, 'schema': {'schema': 'g-b1-run-v1'}, 'task id': {'task_id': 'HIST-V2'},
           'seed 7': {'seed': 7}, 'seed bool': {'seed': True}, 'microbatch 64': {'microbatch': 64}, 'extra field': {'x': 1}}
    for label, patch in bad.items():
        expect_rejected(lambda patch=patch: ga.validate_config({**base, **patch}), label)
    missing = dict(base)
    del missing['head_dim']
    expect_rejected(lambda: ga.validate_config(missing), 'missing field')
    expect_rejected(lambda: ga.validate_config([]), 'not a dict')
    small = {**base, 'd_model': 64, 'heads': 3, 'head_dim': 16, 'encoder_layers': 2, 'decoder_layers': 2, 'ff': 128, 'dropout': 0.1,
             'microbatch': 2, 'max_epochs': 3, 'min_epochs': 1}
    ga.validate_config(small, engineering=True)
    expect_rejected(lambda: ga.validate_config(small), 'engineering shape in a formal run')
    for label, patch in {'history 100': {'history_len': 100}, 'random SID': {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'}, 'beam 200': {'beam': 200},
                         'd_model 256': {'d_model': 256}, 'heads 9': {'heads': 9}, 'head_dim 24': {'head_dim': 24}, 'dropout 0.9': {'dropout': 0.9},
                         'layers 5': {'encoder_layers': 5}, 'min > max epochs': {'min_epochs': 4}}.items():
        expect_rejected(lambda patch=patch: ga.validate_config({**small, **patch}, engineering=True), f'engineering relaxation must not allow {label}')


def check_factory(codec):
    import json
    from vkgr import generative_arch as ga
    from vkgr.tiger_arch import expected_param_count
    cfg = {**ga.default_config(), 'd_model': 64, 'heads': 3, 'head_dim': 16, 'encoder_layers': 2, 'decoder_layers': 2, 'ff': 128, 'dropout': 0.1}
    torch.manual_seed(42)
    legacy = ga.make_model_arch(codec, {**cfg, 'init': 'legacy'}, device='cpu')
    torch.manual_seed(42)
    small = ga.make_model_arch(codec, {**cfg, 'init': 'small'}, device='cpu')
    for name in ('embedding', 'position'):
        weight = getattr(small, name).weight.detach()
        rows = weight[1:] if name == 'embedding' else weight
        ok(abs(float(rows.std()) - 0.02) < 0.006, f'{name} std must be ~0.02 under init=small')
    ok(bool(torch.all(small.embedding.weight[0] == 0)), 'PAD row must be zero')
    for key, value in legacy.state_dict().items():
        if not key.startswith(('embedding.', 'position.')):
            same(value, small.state_dict()[key], f'init=small changed a non-embedding parameter: {key}')
    tokens = torch.tensor([[2, 3, 0]])
    ok(torch.allclose(small.embed(tokens), small.embedding(tokens) * 8.0 + small.position(torch.arange(3))[None]), 'forward must keep embedding * sqrt(d_model)')
    report = small.init_report
    ok(report['init'] == 'small' and report['std'] == 0.02 and legacy.init_report['std'] is None, 'init report fields')
    arch = report['arch']
    ok(set(arch) == {'spec', 'params_total', 'params_transformer', 'params_expected'}, 'init_report.arch fields')
    total = sum(p.numel() for p in small.parameters())
    expected = expected_param_count(codec.vocab_size, 64, 3, 16, 2, 2, 128)
    ok(arch['params_total'] == total == arch['params_expected'] == expected, 'params_total / closed form disagree')
    ok(arch['params_transformer'] == sum(p.numel() for p in small.transformer.parameters()), 'params_transformer')
    ok(arch['spec']['heads'] == 3 and arch['spec']['head_dim'] == 16 and arch['spec']['inner_dim'] == 48 and arch['spec']['vocab'] == codec.vocab_size, 'arch spec')
    json.dumps(report)  # the report enters the run binding: it must be JSON-serialisable
    expect_rejected(lambda: ga.make_model_arch(codec, {**cfg, 'init': 'big'}, device='cpu'), 'unknown init')
    expect_rejected(lambda: ga.make_model_arch(codec, {**cfg, 'history_len': 100}, device='cpu'), 'history_len 100')


def check_anchor_factory_equals_frozen(codec):
    from vkgr import generative_arch as ga
    from vkgr import generative_hp as hp
    from vkgr import generative_training as gt
    cfg = {**ga.default_config(), **dict(zip(ga.SHAPE_KEYS, ga.ANCHOR_SHAPE)), 'init': 'legacy', 'learning_rate': .0003}
    torch.manual_seed(42)
    expected = gt.make_model(codec, hp.core_config(cfg), device='cpu').state_dict()
    torch.manual_seed(42)
    actual = ga.make_model_arch(codec, cfg, device='cpu').state_dict()
    ok(list(expected) == list(actual), 'factory state_dict keys differ from the frozen factory')
    for key in expected:
        same(expected[key], actual[key], f'factory differs from gt.make_model: {key}')


def check_optimizer(codec):
    from vkgr import generative_arch as ga
    from vkgr import generative_hp as hp
    from vkgr import generative_training as gt
    for lr in (.0003, .001):
        cfg = {**ga.default_config(), **dict(zip(ga.SHAPE_KEYS, ga.ANCHOR_SHAPE)), 'learning_rate': lr}
        stock = gt.make_model(codec, hp.core_config(cfg), device='cpu')
        arch = ga.make_model_arch(codec, cfg, device='cpu')
        ref, got = gt.make_optimizer(stock, hp.core_config(cfg)), ga.make_optimizer_arch(arch, hp.core_config(cfg))
        ok(type(ref) is type(got) and len(ref.param_groups) == len(got.param_groups), 'optimizer type/groups')
        for a, b in zip(ref.param_groups, got.param_groups):
            ok({k: v for k, v in a.items() if k != 'params'} == {k: v for k, v in b.items() if k != 'params'}, f'optimizer hyper-parameters differ at lr {lr}')
            ok(len(a['params']) == len(b['params']), 'optimizer parameter count')
    arch.double()
    expect_rejected(lambda: ga.make_optimizer_arch(arch, hp.core_config(cfg)), 'non-FP32 parameters')


# ---- R: entry point ------------------------------------------------------------------------------------------
RUNNER_GLOBALS = ('B1_MEMBERS', '_evaluate', 'make_model_b1', 'validate_config', 'gt')


def check_globals_restored():
    """The arch runner patches generative_b1_run only for the duration of the call, also on the exception path."""
    from vkgr import generative_b1_arch as ba
    from vkgr import generative_b1_run as b1
    before = {n: getattr(b1, n) for n in RUNNER_GLOBALS}

    def restored(label):
        for n in RUNNER_GLOBALS:
            ok(getattr(b1, n) is before[n], f'{label}: generative_b1_run.{n} was not restored')
    with ba._runner(None):
        ok(b1.B1_MEMBERS == ba.MEMBERS and b1.validate_config is not before['validate_config'], 'the runner did not install the arch hooks')
    restored('normal exit')
    try:
        with ba._runner({'mode': 'batched', 'batch_size': 1}):
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    restored('exception exit')


def check_gates():
    """Workers > 1, batched decode with batch != 1 and any subcommand other than train/verify fail before any output."""
    import contextlib
    import io
    import os
    import tempfile
    from unittest import mock
    from vkgr import generative_b1_arch as ba
    from vkgr import generative_sharded as gs
    expect_rejected(lambda: ba._gate(None, 2), 'sharded evaluation workers')
    for size in (2, 32):
        expect_rejected(lambda size=size: ba._gate({'mode': 'batched', 'batch_size': size}, 1), f'batched decode batch {size}')
    ba._gate(None, 1)
    ba._gate({'mode': 'batched', 'batch_size': 1}, 1)

    def cli(head, env):
        clean = {k: v for k, v in os.environ.items() if not k.startswith('VKGR_')}
        clean.update(env)
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp, mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(gs, 'cpu_quota_cores', return_value=None):
            argv = head + ['--prepared', 'x', '--prepared-sha', '0' * 64, '--history-manifest', 'y', '--history-sha', '0' * 64,
                           '--config', 'z', '--config-sha', '0' * 64, '--output', f'{tmp}/out', '--scratch', f'{tmp}/s']
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = ba.main(argv)
            return code, err.getvalue(), os.path.exists(f'{tmp}/out')
    for label, head, env in (('workers 2', ['train'], {'VKGR_EVAL_WORKERS': '2'}), ('batched without batch size', ['train'], {'VKGR_DECODE': 'batched'}),
                             ('batched batch 2', ['train'], {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '2'}), ('beam-diagnostic', ['beam-diagnostic'], {}), ('no subcommand', [], {})):
        code, err, out = cli(head, env)
        ok(code == 1 and 'G-ARCH-01 gate failed' in err and not out, f'{label}: must exit 1 at the G-ARCH-01 gate before output (code {code}, output {out}): {err.strip()[:120]}')
    code, err, out = cli(['train'], {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '1'})
    ok(code == 1 and 'G-ARCH-01 gate failed' not in err and 'G-B1 gate failed' in err, 'batch size 1 must pass the arch gate (and fail later on the bogus inputs)')


# ---- R7: preflight script ------------------------------------------------------------------------------------
def parity_world(tmp):
    """Engineering world + a default-runner reference run at d_model 128 + a callable running scripts/g_arch_parity.py."""
    import contextlib
    import importlib.util
    import io
    from pathlib import Path
    from test_generative_b1 import config as b1_config, engineering_world
    from vkgr import generative_b1_run as b1
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('g_arch_parity', root / 'scripts' / 'g_arch_parity.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    _, receipt, manifest, sha = engineering_world(tmp)
    common = (root, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha)
    value = b1.run(*common, b1_config(d_model=128, ff=512, learning_rate=.0003), Path(tmp) / 'run', temporary=Path(tmp) / 's-old')

    def call(scratch, steps='3', microbatches='2'):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = script.main(['--prepared', str(receipt['manifest_path']), '--prepared-sha', receipt['manifest_sha256'], '--history-manifest', str(manifest),
                                '--history-sha', sha, '--manifest', str(Path(tmp) / 'run' / 'manifest.json'), '--manifest-sha', value['manifest_sha256'],
                                '--scratch', str(Path(tmp) / scratch), '--train-steps', steps, '--microbatches', microbatches], root=root)
        return code, out.getvalue(), err.getvalue()
    return script, value, call


def check_parity_flags(call):
    """The script must report 'equal' for identical models and 'not equal' once the new class is perturbed."""
    import json
    from unittest import mock
    from vkgr import tiger_arch
    code, out, err = call('flags-equal', steps='2')
    ok(code == 0, f'script failed: {err}')
    ok(json.loads(out)['pf1_anchor_shape']['train_steps_equal'] is True, 'identical models must be reported equal')
    original = tiger_arch.TigerArchModel.decode
    with mock.patch.object(tiger_arch.TigerArchModel, 'decode', lambda self, memory, padding, decoder: original(self, memory, padding, decoder) * 1.001):
        code, out, err = call('flags-perturbed', steps='2')
    ok(code == 0, f'script failed: {err}')
    ok(json.loads(out)['pf1_anchor_shape']['train_steps_equal'] is False, 'a perturbed new class must be reported unequal')
