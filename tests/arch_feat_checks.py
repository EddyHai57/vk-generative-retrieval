"""GFEAT-02 guards (not a test module): shared by the normal tests and the mutation tests.

Every check raises AssertionError through ``ok`` when its property is broken. Model checks use a synthetic codec
(no engineering contract, server-safe); "bitwise" means torch.equal, never allclose.
"""
import math

import torch

from test_generative_feat import card_ratio_bucket, card_timespent_bucket, model_inputs, synthetic_codec, valid_side

# engineering-size independent-head shape (3 heads x 16 on d_model 64: the stock module cannot build it), dropout on
SHAPE = dict(d_model=64, heads=3, head_dim=16, encoder_layers=2, decoder_layers=2, ff=128, dropout=0.1)
SIDE_TABLES = {'f2_feedback': ('feedback',), 'f3_watch_buckets': ('timespent_bucket', 'ratio_bucket')}


def ok(condition, message):
    if not condition:
        raise AssertionError(message)


def config(feature, **overrides):
    from vkgr import generative_arch_feat as gaf
    value = gaf.default_config(feature)
    value.update(SHAPE)
    value.update(microbatch=2, max_epochs=3, min_epochs=1, learning_rate=.001)
    value.update(overrides)
    return value


def codec():
    return synthetic_codec()


def build(codec, feature, seed=42):
    from vkgr import generative_arch_feat as gaf
    torch.manual_seed(seed)
    return gaf.make_model(codec, config(feature), device='cpu')


def plain(codec, seed=42):
    """The GARCH-01 model of the same structure, from the frozen GARCH-01 factory."""
    from vkgr import generative_arch as ga
    torch.manual_seed(seed)
    return ga.make_model_arch(codec, config('none'), device='cpu')


def batch(codec, lengths=(3, 1, 5)):
    items = [int(i) for i in codec.catalog.ids]
    histories = [[items[(7 * b + 3 * k) % len(items)] for k in range(n)] for b, n in enumerate(lengths)]
    return model_inputs(codec, histories, [valid_side(n, seed=b) for b, n in enumerate(lengths)])


def decoder_tokens(codec, rows):
    """BOS + the first code of each of the three prefix levels: valid decoder ids of this codec."""
    return torch.tensor([[1] + [codec.offsets[j] for j in range(3)]] * rows, dtype=torch.long)


def valid_items(source):
    return source.reshape(source.shape[0], -1, 4)[..., 0].ne(0)


def zero_side(model, freeze=True):
    for name in ('feedback', 'timespent_bucket', 'ratio_bucket'):
        table = getattr(model, name, None)
        if table is not None:
            with torch.no_grad():
                table.weight.zero_()
            table.weight.requires_grad_(not freeze)


# ---- (a) feature=none is the GARCH-01 model -------------------------------------------------------------------
def check_none_is_plain_arch(codec):
    from vkgr.tiger_arch import TigerArchModel
    model, ref = build(codec, 'none'), plain(codec)
    ok(type(model) is TigerArchModel, 'feature=none must build the plain TigerArchModel')
    ok(model.init_report == ref.init_report, 'feature=none init_report differs from GARCH-01')
    state, want = model.state_dict(), ref.state_dict()
    ok(list(state) == list(want), 'feature=none state_dict keys differ from GARCH-01')
    ok(all(torch.equal(state[k], want[k]) for k in want), 'feature=none same-seed initialisation differs from GARCH-01')


def check_base_params_equal_plain(codec):
    """f2/f3: every GARCH-01 parameter keeps the GARCH-01 random stream bit for bit; the only new keys are the side tables."""
    want = plain(codec).state_dict()
    for feature, tables in SIDE_TABLES.items():
        model = build(codec, feature); state = model.state_dict()
        ok(set(state) - set(want) == {f'{t}.weight' for t in tables}, f'{feature}: unexpected new parameters')
        ok(all(torch.equal(state[k], want[k]) for k in want), f'{feature}: GARCH-01 parameters differ under the same seed')
        ok(model.init_report['arch'] == plain(codec).init_report['arch'], f'{feature}: arch report differs from GARCH-01')


def check_side_tables_follow_base_init(codec):
    """Side-table values are the draws that follow the GARCH-01 initialisation (GFEAT-01 order), and building an f2/f3 model leaves
    the global random stream where feature=none leaves it (dropout masks during training are then the control's)."""
    from vkgr import generative_arch_feat as gaf
    for feature, tables in SIDE_TABLES.items():
        ref = plain(codec); gaf.TigerArchModelSide.add_side_tables(ref, feature)  # GFEAT-01's draw, in the global stream after base init
        model = build(codec, feature)
        ok(all(torch.equal(getattr(model, t).weight, getattr(ref, t).weight) for t in tables), f'{feature}: side tables are not the draws after the base init')
        plain(codec); after_none = torch.rand(16)
        build(codec, feature); after_side = torch.rand(16)
        ok(torch.equal(after_none, after_side), f'{feature}: building the side tables moved the global random stream')


# ---- injection: encoder token space, before x sqrt(d), broadcast onto the 4 SID tokens of the item ------------
def check_injection(codec, feature):
    """embed_source == (E[token] + side_vector(item of token)) * sqrt(d) + position, written out token by token."""
    model = build(codec, feature).eval(); source, side = batch(codec)
    with torch.no_grad():
        valid = valid_items(source); vector = model.side_vector(side, valid)
        tokens = model.embedding(source).clone()
        for b in range(source.shape[0]):
            for k in range(source.shape[1] // 4):
                if bool(valid[b, k]):
                    for t in range(4):
                        tokens[b, 4 * k + t] = tokens[b, 4 * k + t] + vector[b, k]
        expected = tokens * math.sqrt(model.d_model) + model.position(torch.arange(source.shape[1]))[None]
        got = model.embed_source(source, side)
        ok(float(vector[valid].abs().min()) > 0, f'{feature}: side vector unexpectedly zero')
        ok(torch.equal(got, expected), f'{feature}: side not injected as (E + side) * sqrt(d) + position on all four tokens')


def check_injection_before_scale(codec):
    """A unit side vector moves every token of its item by exactly sqrt(d) (not by 1, not by d) and nothing else."""
    model = build(codec, 'f2_feedback').eval(); source, _ = batch(codec, (2,))
    with torch.no_grad():  # E and position zero: the difference is then exact in float32 (0.5 * sqrt(64) = 4)
        model.embedding.weight.zero_(); model.position.weight.zero_()
        model.feedback.weight.zero_(); model.feedback.weight[0].fill_(0.5)
        one = torch.tensor([[[1, 5, 9], [0, 5, 9]]]); zero = torch.tensor([[[0, 5, 9], [0, 5, 9]]])
        diff = model.embed_source(source, one) - model.embed_source(source, zero)
    ok(torch.equal(diff[0, :4], torch.full((4, model.d_model), 0.5 * math.sqrt(model.d_model))), 'side not scaled by sqrt(d) exactly once')
    ok(torch.equal(diff[0, 4:], torch.zeros(4, model.d_model)), 'side leaks onto the tokens of another item')


def _exact_tables(model):
    """Table values k * 2^-6: every sum below is exact in float32, so the expectation can be compared bit for bit."""
    with torch.no_grad():
        for name in ('feedback', 'timespent_bucket', 'ratio_bucket'):
            table = getattr(model, name, None)
            if table is not None:
                table.weight.copy_(torch.arange(1, table.weight.numel() + 1, dtype=torch.float32).reshape(table.weight.shape) * 2 ** -6)
                if table.padding_idx is not None:
                    table.weight[table.padding_idx].zero_()


def check_side_vector_columns(codec, feature):
    """F2 reads column 0 (feedback mask, bit b -> row b); F3 reads columns 1, 2 (timespent bucket + watch-ratio bucket, card tables)."""
    model = build(codec, feature).eval(); source, side = batch(codec); _exact_tables(model)
    valid = valid_items(source)
    expected = torch.zeros(side.shape[0], side.shape[1], model.d_model)
    for b in range(side.shape[0]):
        for k in range(side.shape[1]):
            if not bool(valid[b, k]):
                continue
            mask, ts, dur = (int(v) for v in side[b, k])
            if feature == 'f2_feedback':
                for bit in range(6):
                    if mask >> bit & 1:
                        expected[b, k] = expected[b, k] + model.feedback.weight[bit].detach()
            else:
                expected[b, k] = model.timespent_bucket.weight[card_timespent_bucket(ts)].detach() + model.ratio_bucket.weight[card_ratio_bucket(ts, dur)].detach()
    with torch.no_grad():
        ok(torch.equal(model.side_vector(side, valid), expected), f'{feature}: side vector reads the wrong columns / definitions')


def check_bucket_definitions():
    """Bucket edges are the train-defined constants of the frozen card (written out independently), every timespent / ratio value."""
    from vkgr import generative_arch_feat as gaf
    from vkgr import retrieval_features as rf
    ok(rf.TIMESPENT_EDGES == (4, 10, 31, 61, 135, 255) and (rf.TIMESPENT_BUCKETS, rf.RATIO_BUCKETS, rf.FEEDBACK_DIM) == (7, 5, 6), 'bucket constants changed')
    ts = torch.arange(1, 256); valid = torch.ones_like(ts, dtype=torch.bool)
    ok(rf.timespent_bucket(ts, valid).tolist() == [card_timespent_bucket(v) for v in range(1, 256)], 'timespent buckets differ from the card')
    grid_ts, grid_dur = torch.meshgrid(ts, ts, indexing='ij')
    got = rf.ratio_bucket(grid_ts, grid_dur, torch.ones_like(grid_ts, dtype=torch.bool)).flatten().tolist()
    ok(got == [card_ratio_bucket(a, b) for a in range(1, 256) for b in range(1, 256)], 'ratio buckets differ from the card')
    ok(gaf.TigerArchModelSide.side_vector is gaf.gf.TigerModelSide.side_vector, 'GFEAT-02 side vector is not the GFEAT-01 one')


SHARED_FILES = ('history_side', 'retrieval_features')


def check_shared_sources(root, pinned):
    """(c) the feature definitions are the GFEAT-01 files byte for byte (sha256 against the GFEAT-01 pins)."""
    import hashlib
    for name in SHARED_FILES:
        digest = hashlib.sha256((root / 'src' / 'vkgr' / f'{name}.py').read_bytes()).hexdigest()
        ok(digest == pinned[name], f'{name}.py differs from the GFEAT-01 file')


# ---- the encoder consumes the injected input; the decoder is GARCH-01's ---------------------------------------
def check_encode_uses_injected_input(codec, feature):
    model = build(codec, feature).eval(); source, side = batch(codec)
    with torch.no_grad():
        memory, padding = model.encode(source, side)
        x = model.embed_source(source, side)
        for layer in model.transformer.encoder.layers:
            x = layer(x, padding)
        ok(torch.equal(padding, source.eq(0)), f'{feature}: encoder padding mask changed')
        ok(torch.equal(memory, model.transformer.encoder.norm(x)), f'{feature}: encoder does not run on the injected input')
        changed = side.clone(); changed[2, 0, 0] = (int(side[2, 0, 0]) + 1) % 64; changed[2, 0, 1] = 255 if int(side[2, 0, 1]) != 255 else 1
        after = model.encode(source, changed)[0]
    ok(torch.equal(after[:2], memory[:2]) and not torch.equal(after[2], memory[2]), f'{feature}: side is not per sample')


def check_decoder_untouched(codec, feature):
    """decode / forward == the plain GARCH-01 decoder with the same weights (side enters the encoder only)."""
    model = build(codec, feature).eval(); ref = plain(codec).eval()
    ref.load_state_dict({k: v for k, v in model.state_dict().items() if k in ref.state_dict()})
    source, side = batch(codec); decoder = decoder_tokens(codec, source.shape[0])
    with torch.no_grad():
        memory, padding = model.encode(source, side)
        ok(torch.equal(model.decode(memory, padding, decoder), ref.decode(memory, padding, decoder)), f'{feature}: decoder path changed')
        ok(torch.equal(model(source, decoder, side), ref.decode(memory, padding, decoder)), f'{feature}: forward is not encode(side) + GARCH-01 decode')


# ---- (b) zero side tables == feature=none, bit for bit ---------------------------------------------------------
def check_zero_side_equals_none_eval(codec, feature):
    model = build(codec, feature).eval(); ref = plain(codec).eval(); zero_side(model)
    source, side = batch(codec); decoder = decoder_tokens(codec, source.shape[0])
    with torch.no_grad():
        ok(torch.equal(model.embed_source(source, side), ref.embed(source)), f'{feature}: zero side embedding differs from none')
        ok(torch.equal(model.encode(source, side)[0], ref.encode(source)[0]), f'{feature}: zero side encoder memory differs from none')
        ok(torch.equal(model(source, decoder, side), ref(source, decoder)), f'{feature}: zero side logits differ from none')


def train_rows(codec, count=6):
    items = [int(i) for i in codec.catalog.ids]
    rows = [{'example_id': e, 'history': [items[(5 * e + 3 * k) % len(items)] for k in range(1 + e % 4)], 'target': items[(11 * e + 1) % len(items)]} for e in range(count)]
    return rows, {r['example_id']: valid_side(len(r['history']), seed=100 + r['example_id']) for r in rows}


class _Tokens:
    def __init__(self, sides):
        self.sides = sides

    def train_side(self, rows, device='cpu'):
        from vkgr.generative_feat import side_tensor
        return side_tensor([self.sides[r['example_id']] for r in rows], device)


def check_zero_side_equals_none_train(codec, feature, steps=3):
    """Zero, frozen side tables: the f2/f3 training step IS the GARCH-01 step (dropout on, same RNG): loss and weights bitwise."""
    from types import SimpleNamespace
    from vkgr import generative_arch as ga
    from vkgr import generative_feat as gf
    from vkgr import generative_training as gt
    cfg = config(feature); model = build(codec, feature); zero_side(model); ref = plain(codec)
    opt, ref_opt = ga.make_optimizer_arch(model, cfg), ga.make_optimizer_arch(ref, config('none'))
    rows, sides = train_rows(codec); context = SimpleNamespace(codec=codec)
    for step in range(steps):
        torch.manual_seed(1000 + step); got = gf.train_batch_feat(model, opt, rows, context, _Tokens(sides), cfg, 'cpu')
        torch.manual_seed(1000 + step); want = gt.train_effective_batch(ref, ref_opt, rows, codec, microbatch=cfg['microbatch'], device='cpu')
        ok(got == want, f'{feature}: step {step + 1} loss differs from none')
        state = model.state_dict()
        ok(all(torch.equal(v, state[k]) for k, v in ref.state_dict().items()), f'{feature}: step {step + 1} weights differ from none')


# ---- registry and runner --------------------------------------------------------------------------------------
def expect_rejected(call, label):
    try:
        call()
    except ValueError:
        return
    raise AssertionError(f'{label} was accepted')


def check_registry():
    from vkgr import generative_arch as ga
    from vkgr import generative_arch_feat as gaf
    for feature in ('none', 'f2_feedback', 'f3_watch_buckets'):
        value = gaf.default_config(feature)
        gaf.validate_config(value)
        ok(tuple(value[k] for k in ga.SHAPE_KEYS) == ga.PAPER_SCALE and value['learning_rate'] == .0003, 'default is not A1')
        for seed in (2024, 2025):
            gaf.validate_config({**value, 'seed': seed})
    gaf.validate_config(config('f2_feedback'), engineering=True)
    base = gaf.default_config('f2_feedback')
    bad = {'feature f1': {'feature': 'f1_ratio_pool'}, 'feature None': {'feature': None}, 'feature list': {'feature': ['none']},
           'side_version': {'side_version': 'HIST_V2_SIDE_V2'}, 'schema GARCH-01': {'schema': 'g-b1-arch-run-v1'}, 'schema GFEAT-01': {'schema': 'g-feat01-run-v1'},
           'task GARCH-01': {'task_id': 'G-ARCH-01'}, 'task GFEAT-01': {'task_id': 'GFEAT-01'}, 'anchor shape': dict(zip(ga.SHAPE_KEYS, ga.ANCHOR_SHAPE)),
           'lr 1e-3 (A2)': {'learning_rate': .001}, 'history 100': {'history_len': 100}, 'random SID': {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'},
           'init legacy': {'init': 'legacy'}, 'seed 7': {'seed': 7}, 'max_epochs 30': {'max_epochs': 30}, 'beam 200': {'beam': 200}, 'extra key': {'x': 1}}
    for label, patch in bad.items():
        expect_rejected(lambda: gaf.validate_config({**base, **patch}), label)
    for key in ('feature', 'side_version', 'head_dim'):
        expect_rejected(lambda: gaf.validate_config({k: v for k, v in base.items() if k != key}), f'missing {key}')
    expect_rejected(lambda: gaf.validate_config([]), 'non-dict')


RUNNER_GLOBALS = ('B1_MEMBERS', 'validate_config', '_contexts', 'make_model_b1', 'SIDTokensL', 'train_batch', '_evaluate', '_binding', 'gt')


def check_runner_installs_and_restores():
    from vkgr import generative_arch_feat as gaf
    from vkgr import generative_b1_arch_feat as baf
    from vkgr import generative_b1_run as b1
    before = {n: getattr(b1, n) for n in RUNNER_GLOBALS}

    def restored(label):
        for n in RUNNER_GLOBALS:
            ok(getattr(b1, n) is before[n], f'{label}: generative_b1_run.{n} was not restored')
    with baf._runner(None, None):
        ok(b1.B1_MEMBERS == baf.MEMBERS, 'runner did not install the GFEAT-02 members')
        ok(b1.validate_config is gaf.validate_config and b1.make_model_b1 is gaf.make_model, 'runner did not install the GFEAT-02 registry/factory')
        from vkgr import generative_arch as ga
        probe = torch.nn.Linear(2, 2)
        ok(type(b1.gt.make_optimizer(probe, {'learning_rate': .0003, 'weight_decay': 1e-4})) is type(ga.make_optimizer_arch(probe, {'learning_rate': .0003, 'weight_decay': 1e-4})),
           'runner did not install the GARCH-01 optimizer proxy')
        ok(all(getattr(b1, n) is not before[n] for n in ('_contexts', 'SIDTokensL', 'train_batch', '_binding', 'gt')), 'runner did not install the side hooks')
    restored('normal exit')
    try:
        with baf._runner({'mode': 'batched', 'batch_size': 1}, None):
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    restored('exception exit')


def check_gates():
    from vkgr import generative_b1_arch_feat as baf
    for label, kwargs in (('workers 2', dict(decode=None, side=None, workers=2)), ('batch 2', dict(decode={'mode': 'batched', 'batch_size': 2}, side=None)),
                          ('batch 64', dict(decode={'mode': 'batched', 'batch_size': 64}, side=None)), ('side not a pair', dict(decode=None, side='x'))):
        def enter(kwargs=kwargs):
            with baf._runner(**kwargs):
                pass
        expect_rejected(enter, label)
