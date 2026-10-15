"""SEQ-M1 (P line) model tests: SASRec-v3 = F4-T010 anchor + gated 1-layer pre-LN Transformer branch (u = m + g * r).

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); the unit tests call
them and tests/test_seqm1_mutations.py re-runs them under deliberately wrong implementations.
Design: .agents/tasks/seq-m1-design.md (G1-G16, AC1-AC7, AC17, AC18); F1/F2/N3/N4 of the design review are folded in
(per-event oracle G15d, _check on every entry point, bucket-crossing perturbations, G4 covers the bidirectional arm).
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_model import AUDIT_POSITION_EDGES, AUDIT_WEEKGAP_EDGES, CONTENT, SUPPORTED, ensure, model_inputs, ref_id  # noqa: E402

ARMS = ('mean', 'last', 'attn', 'bi', 'pure')
ANCHORED = ('mean', 'last', 'attn', 'bi')
CAUSAL = ('mean', 'last', 'attn', 'pure')
# design 3.3 / G14 literals: arm -> (mask, readout, anchored)
ARM_TABLE = {'mean': ('causal', 'mean', True), 'last': ('causal', 'last', True), 'attn': ('causal', 'attn', True),
             'bi': ('bi', 'mean', True), 'pure': ('causal', 'last', False)}
TS_EDGES = (4, 10, 31, 61, 135, 255)
# design 3.4, item embedding 120,416 x 64 = 7,706,624 (the real catalog); branch parts below
BRANCH_PARTS = {'order_table': 3264, 'layer': 49984, 'final_norm': 128}
WHOLE_MODEL_120416 = {'mean': 7761857, 'last': 7761857, 'bi': 7761857, 'attn': 7766081, 'pure': 7761856}


def build(arm, seed=42):
    from vkgr.retrieval_seqm1 import make_seqm1_model
    from vkgr.training import seed_all
    seed_all(seed)
    return make_seqm1_model(CONTENT, SUPPORTED, 50, arm)


def build_php(seed=42):
    from vkgr.retrieval_php01 import make_php01_model
    from vkgr.training import seed_all
    seed_all(seed)
    return make_php01_model(CONTENT, SUPPORTED, 50, 64)


def activate(model, seed=5, gate=.5):
    """Make the branch matter: non-zero gate, random LN_f weight, non-zero attention vector (fixture convention, design 7)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if hasattr(model, 'gate'):
            model.gate.fill_(gate)
        model.final_norm.weight.copy_(torch.rand(64, generator=g) + .5)
        if model.readout == 'attn':
            model.attn_w.copy_(torch.randn(64, generator=g))
    return model


def make_batch(lengths, width, seed=0, pool=(1, 300), last_pool=None):
    """Left-aligned histories, 6-column side with non-decreasing weeks in 0..24 (same distribution as model_inputs)."""
    g = torch.Generator().manual_seed(seed)
    batch = len(lengths)
    histories = torch.zeros(batch, width, dtype=torch.long); side = torch.zeros(batch, width, 6, dtype=torch.long)
    for j, n in enumerate(lengths):
        histories[j, :n] = torch.randint(pool[0], pool[1] + 1, (n,), generator=g)
        if last_pool is not None:
            histories[j, n - 1] = torch.randint(last_pool[0], last_pool[1] + 1, (1,), generator=g)[0]
        side[j, :n, 0] = torch.randint(0, 64, (n,), generator=g)
        side[j, :n, 1] = torch.randint(1, 256, (n,), generator=g)
        side[j, :n, 2] = torch.tensor([[5, 9, 25, 60, 180][k] for k in torch.randint(0, 5, (n,), generator=g).tolist()])
        side[j, :n, 3] = torch.sort(torch.randint(0, 25, (n,), generator=g)).values
    return histories, side


def edge_cases():
    return {'random': model_inputs(0), 'n1': make_batch([1, 1, 1], 1, 1), 'n50': make_batch([50, 50, 50], 50, 2),
            'all_unsupported': make_batch([5, 12, 1], 12, 3, pool=(200, 300)),
            'last_unsupported': make_batch([6, 9], 9, 4, pool=(1, 199), last_pool=(200, 300)),
            'variable_pad': make_batch([3, 17, 40], 50, 5)}


def pad_to(histories, side, width):
    h = torch.zeros(len(histories), width, dtype=histories.dtype); s = torch.zeros(len(histories), width, 6, dtype=side.dtype)
    w = histories.shape[1]
    h[:, :w] = histories; s[:, :w] = side
    return h, s


def controlled_row():
    """One history of 8 events with weeks [0,0,0,20,20,22,23,24] (event 2 can cross the week-gap bucket 9-24 -> 3-4)."""
    h = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18]])
    s = torch.zeros(1, 8, 6, dtype=torch.long)
    s[0, :, 0] = 1; s[0, :, 1] = torch.tensor([3, 20, 3, 40, 70, 8, 90, 120]); s[0, :, 2] = torch.tensor([30, 30, 30, 60, 80, 8, 100, 130])
    s[0, :, 3] = torch.tensor([0, 0, 0, 20, 20, 22, 23, 24])
    return h, s


def ts_bucket(v):
    return 1 + sum(1 for e in TS_EDGES if v >= e)


def ratio_id(ts, dur):
    if 10 * ts < dur:
        return 1
    if 2 * ts < dur:
        return 2
    if ts < dur:
        return 3
    return 4 if ts == dur else 5


def other_timespent(v):
    """A timespent value in a different bucket (N3: the perturbation must cross a bucket boundary)."""
    w = 255 if ts_bucket(v) != ts_bucket(255) else 1
    ensure(ts_bucket(w) != ts_bucket(v), 'fixture bug: timespent bucket did not change')
    return w


# ---------------------------------------------------------------- guards (reused by the mutation tests)

def guard_zero_step_equals_anchor():
    """G1 / AC1: with g == 0 the four anchored arms output the anchor user vector bit for bit (train and eval mode)."""
    anchor = build_php()
    for arm in ANCHORED:
        model = build(arm)
        for mode in ('train', 'eval'):
            getattr(model, mode)()
            for name, (h, s) in edge_cases().items():
                ensure(torch.equal(model.users(h, s), anchor.users(h, s)), f'{arm}/{mode}/{name}: M1.users != F4IDMeanD.users at g=0')


def guard_init_order_and_anchor_tables():
    """G2 / AC2: the five anchor tables equal make_php01_model bit for bit; O, layer, LN_f, attention drawn after them in order."""
    from vkgr import retrieval_seqm1 as m
    from vkgr.retrieval_php01 import make_php01_model
    from vkgr.training import seed_all
    calls = []
    original = m.init_embeddings
    m.init_embeddings = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    try:
        models = {arm: build(arm) for arm in ARMS}
    finally:
        m.init_embeddings = original
    ensure(len(calls) == len(ARMS), f'init_embeddings called {len(calls)} times for {len(ARMS)} constructions (must be once, before the branch)')
    for arm, model in models.items():
        ref = build_php()
        for name in ('embedding', 'timespent_bucket', 'ratio_bucket', 'position_table', 'weekgap_table'):
            ensure(torch.equal(model.get_submodule(name).weight, ref.get_submodule(name).weight), f'{arm}: {name} differs from make_php01_model')
        # replay the draws after the anchor tables
        seed_all(42); make_php01_model(CONTENT, SUPPORTED, 50, 64)         # replay: the anchor draws, then continue by hand
        order = torch.nn.Embedding(51, 64, padding_idx=0)
        with torch.no_grad():
            order.weight.normal_(0., .02); order.weight[0].zero_()
        ensure(torch.equal(model.order_table.weight, order.weight), f'{arm}: O is not the first draw after P/G (N(0,0.02), PAD 0)')
        layer = m.make_layer()
        ensure(all(torch.equal(a, b) for a, b in zip(model.layer.state_dict().values(), layer.state_dict().values())), f'{arm}: Transformer layer not drawn right after O')
        if arm == 'attn':
            proj = torch.nn.Linear(64, 64)
            ensure(torch.equal(model.attn_proj.weight, proj.weight) and torch.equal(model.attn_proj.bias, proj.bias), 'attn: W_a not drawn after the layer')
            ensure(not bool(model.attn_w.any()), 'attn: w must start at exactly 0')


def expected_report(arm, rows):
    """Independent literals: numel per group (design 3.4)."""
    value = {'item_embedding': rows * 64, 'anchor_tables': 1856, 'order_table': 3264, 'layer': 49984, 'final_norm': 128}
    if arm == 'attn':
        value['attn_pool'] = 4224
    if arm != 'pure':
        value['gate'] = 1
    value['total'] = sum(value.values())
    return value


def guard_parameter_report():
    """G3 / AC3: report == formula == design 3.4 literals, group by group (numel, not module type)."""
    from vkgr.retrieval_seqm1 import expected_param_counts_seqm1, parameter_report_seqm1
    for arm in ARMS:
        model = build(arm); got = parameter_report_seqm1(model)
        ensure(got == expected_report(arm, 301), f'{arm}: report {got} != literals')
        ensure(expected_param_counts_seqm1(arm, 301) == got, f'{arm}: formula != report')
        ensure(got['total'] == sum(p.numel() for p in model.parameters()), f'{arm}: report total != numel of parameters')
        ensure(expected_param_counts_seqm1(arm, 120416)['total'] == WHOLE_MODEL_120416[arm], f'{arm}: whole-model total on the real catalog')
    ensure(parameter_report_seqm1(build('last'))['total'] - parameter_report_seqm1(build('pure'))['total'] == 1, 'pure has exactly one parameter less (g)')


def guard_padding_invariant(atol=1e-6):
    """G4 / AC4: the same histories padded to width 20 / 35 / 50 give the same user vector (anchor path and branch path); covers arm bi (N4)."""
    h, s = make_batch([1, 4, 11, 20, 13], 20, 7)
    for arm in ARMS:
        model = activate(build(arm)).eval()
        with torch.no_grad():
            base = model.users(h, s); base_parts = model.parts(h, s)
            for width in (35, 50):
                ph, ps = pad_to(h, s, width)
                ensure(torch.allclose(model.users(ph, ps), base, atol=atol, rtol=0), f'{arm}: users depend on padding width {width}')
                got = model.parts(ph, ps)
                ensure(torch.allclose(got.r, base_parts.r, atol=atol, rtol=0), f'{arm}: branch readout depends on padding width {width}')
                if model.anchored:
                    ensure(torch.allclose(got.m, base_parts.m, atol=atol, rtol=0), f'{arm}: anchor path depends on padding width {width}')


def perturb_after(h, s, k, seed=3):
    """Change item / timespent / duration of every event at position > k of every row (week and n untouched)."""
    h2, s2 = h.clone(), s.clone(); g = torch.Generator().manual_seed(seed)
    valid = h != 0
    for j in range(len(h)):
        for p in range(k + 1, int(valid[j].sum())):
            h2[j, p] = 1 + (int(h[j, p]) + 37) % 299
            s2[j, p, 1] = other_timespent(int(s[j, p, 1])); s2[j, p, 2] = 1 + (int(s[j, p, 2]) * 7) % 255
    ensure(not torch.equal(h2, h) and not torch.equal(s2, s), 'fixture bug: nothing perturbed')
    return h2, s2


def guard_mask_semantics(k=3):
    """G5 / AC5: causal arms: z at positions <= k ignores later events; bidirectional arm does not; last event's week reaches earlier z (P/G anchor)."""
    h, s = make_batch([12, 12, 9, 12], 12, 11)
    h2, s2 = perturb_after(h, s, k)
    for arm in ARMS:
        model = activate(build(arm)).eval()
        with torch.no_grad():
            a = model.parts(h, s).z; b = model.parts(h2, s2).z
        same = torch.allclose(a[:, :k + 1], b[:, :k + 1], atol=1e-7, rtol=0)
        if arm == 'bi':
            ensure(not same, 'bi: later events must change earlier positions')
        else:
            ensure(same, f'{arm}: a later event leaked into an earlier position')
    # reverse case: the LAST event's week changes the week-gap bucket of every earlier event
    ch, cs = controlled_row()
    cs2 = cs.clone(); cs2[0, -1, 3] = 24; cs[0, -1, 3] = 22
    cs[0, 3:7, 3] = 20; cs2[0, 3:7, 3] = 20       # keep non-decreasing for both variants
    from vkgr.retrieval_feat02 import week_gap, weekgap_bucket
    valid = ch != 0
    before = weekgap_bucket(week_gap(cs[..., 3], valid), valid); after = weekgap_bucket(week_gap(cs2[..., 3], valid), valid)
    ensure(not torch.equal(before[0, :4], after[0, :4]), 'fixture bug: week-gap buckets did not change')
    for arm in CAUSAL:
        model = activate(build(arm)).eval()
        with torch.no_grad():
            a = model.parts(ch, cs).z; b = model.parts(ch, cs2).z
        ensure(not torch.allclose(a[:, :4], b[:, :4], atol=1e-7, rtol=0), f'{arm}: last-event week must reach earlier positions (known property, not a bug)')


def softmax_oracle(model, z, valid):
    """Independent attention-pooling oracle with python loops."""
    out = []
    for j in range(len(z)):
        n = int(valid[j].sum())
        scores = torch.stack([(torch.tanh(model.attn_proj(z[j, k])) * model.attn_w).sum() for k in range(n)])
        alpha = torch.softmax(scores, 0)
        out.append((alpha.unsqueeze(1) * z[j, :n]).sum(0))
    return torch.stack(out)


def guard_readouts():
    """G6 / AC6: mean / last / attn readouts against hand-written oracles; attn with w=0 equals mean; alpha is 0 on PAD and sums to 1."""
    h, s = make_batch([1, 5, 12, 12, 7], 12, 13)
    for arm in ARMS:
        model = activate(build(arm)).eval()
        with torch.no_grad():
            p = model.parts(h, s); n = p.valid.sum(1)
            if model.readout == 'mean':
                oracle = torch.stack([p.z[j, :n[j]].mean(0) for j in range(len(h))])
            elif model.readout == 'last':
                oracle = torch.stack([p.z[j, n[j] - 1] for j in range(len(h))])
            else:
                oracle = softmax_oracle(model, p.z, p.valid)
                ensure(p.alpha is not None and bool((p.alpha[~p.valid] == 0).all()), 'attn: alpha must be exactly 0 on PAD')
                ensure(torch.allclose(p.alpha.sum(1), torch.ones(len(h)), atol=1e-6), 'attn: alpha must sum to 1')
            ensure(torch.allclose(p.r, oracle, atol=1e-6, rtol=0), f'{arm}: readout differs from the oracle')
            user = model.users(h, s)
            expected = p.m + model.gate * p.r if model.anchored else oracle
            ensure(torch.allclose(user, expected, atol=1e-6, rtol=0), f'{arm}: users != combination of parts')
    model = activate(build('attn')); model.attn_w.data.zero_(); model.eval()
    with torch.no_grad():
        p = model.parts(h, s); n = p.valid.sum(1)
        mean = torch.stack([p.z[j, :n[j]].mean(0) for j in range(len(h))])
        ensure(torch.allclose(p.r, mean, atol=1e-6, rtol=0), 'attn with w=0 must equal the mean readout')


def loss_of(model, h, s, seed=1):
    c = torch.randn(len(h), 64, generator=torch.Generator().manual_seed(seed))
    return (model.users(h, s) * c).sum()


def branch_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if n.split('.')[0] in ('order_table', 'layer', 'final_norm', 'attn_proj', 'attn_w')]


def guard_gradients():
    """G7 / AC7: at g=0 only g learns (branch grads exactly 0); at g!=0 every branch tensor learns; attn pooling in two stages; pure learns from step 0."""
    h, s = make_batch([12, 12, 5, 9], 12, 17)
    for arm in ANCHORED:
        model = build(arm).eval(); model.final_norm.weight.data.copy_(torch.rand(64, generator=torch.Generator().manual_seed(2)) + .5)
        if arm == 'attn':
            model.attn_w.data.copy_(torch.randn(64, generator=torch.Generator().manual_seed(4)))   # w != 0 so r has a non-trivial pooling
        loss_of(model, h, s).backward()
        ensure(model.gate.grad is not None and float(model.gate.grad.abs()) > 0, f'{arm}: gate must get a gradient at g=0')
        for name, p in branch_parameters(model):
            ensure(p.grad is not None and int(torch.count_nonzero(p.grad)) == 0, f'{arm}: {name} must have an exactly-zero gradient at g=0')
        model = activate(build(arm)).eval()
        if arm == 'attn':
            model.attn_w.data.copy_(torch.randn(64, generator=torch.Generator().manual_seed(9)))
        loss_of(model, h, s).backward()
        for name, p in branch_parameters(model):
            ensure(p.grad is not None and float(p.grad.abs().sum()) > 0, f'{arm}: {name} has no gradient at g=0.5')
    # attn, stage (i): w == 0 -> only w learns
    model = activate(build('attn')); model.attn_w.data.zero_(); model.eval()
    loss_of(model, h, s).backward()
    ensure(float(model.attn_w.grad.abs().sum()) > 0, 'attn(i): w must get a gradient')
    ensure(int(torch.count_nonzero(model.attn_proj.weight.grad)) == 0 and int(torch.count_nonzero(model.attn_proj.bias.grad)) == 0, 'attn(i): W_a, b_a gradients are exactly 0 while w == 0')
    ensure(bool((model.attn_w == 0).all()), 'attn: w initial value must stay 0')
    # pure: no gate, everything learns at step 0
    model = build('pure').eval(); model.final_norm.weight.data.copy_(torch.rand(64, generator=torch.Generator().manual_seed(2)) + .5)
    ensure(not hasattr(model, 'gate'), 'pure has no gate')
    loss_of(model, h, s).backward()
    for name, p in branch_parameters(model):
        ensure(p.grad is not None and float(p.grad.abs().sum()) > 0, f'pure: {name} has no gradient at step 0')


def guard_left_aligned():
    """G8 / AC: a hole in the middle of a history is rejected on every arm and on both entry points."""
    h, s = make_batch([6, 6], 8, 19)
    h[0, 2] = 0; s[0, 2] = 0
    for arm in ARMS:
        model = build(arm)
        for call in (model.users, model.parts):
            try:
                call(h, s)
            except ValueError:
                continue
            raise AssertionError(f'{arm}: {call.__name__} accepted a hole in the history')


def guard_check_on_every_entry():
    """review F2: users() and parts() of all 5 arms reject week=25, decreasing weeks, PAD with values, timespent=0, width 51, empty history."""
    base_h, base_s = make_batch([5, 7], 8, 23)
    bad = {}
    s = base_s.clone(); s[0, 4, 3] = 25; bad['week25'] = (base_h, s)
    s = base_s.clone(); s[0, 1, 3] = 24; s[0, 2, 3] = 3; bad['weeks_decreasing'] = (base_h, s)
    s = base_s.clone(); s[0, 7, 1] = 5; bad['pad_with_values'] = (base_h, s)
    s = base_s.clone(); s[1, 1, 1] = 0; bad['timespent0'] = (base_h, s)
    s = base_s.clone(); s[1, 1, 2] = 256; bad['duration256'] = (base_h, s)
    bad['width51'] = (torch.ones(2, 51, dtype=torch.long), torch.zeros(2, 51, 6, dtype=torch.long))
    bad['empty'] = (torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 8, 6, dtype=torch.long))
    bad['no_side'] = (base_h, None)
    for arm in ARMS:
        model = build(arm)
        for name, (h, s) in bad.items():
            for call in (model.users, model.parts):
                try:
                    call(h, s)
                except ValueError:
                    continue
                raise AssertionError(f'{arm}: {call.__name__} accepted {name}')


def guard_finiteness():
    """G9 / D6: NaN in PAD outputs of the branch does not reach the user vector; NaN at a valid position raises."""
    class PadNaN(torch.nn.Module):
        def __init__(self, inner, on_pad):
            super().__init__(); self.inner = inner; self.on_pad = on_pad

        def forward(self, x, src_mask=None, src_key_padding_mask=None):
            out = self.inner(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            hit = src_key_padding_mask if self.on_pad else ~src_key_padding_mask
            return out.masked_fill(hit.unsqueeze(-1), float('nan'))
    h, s = make_batch([3, 8, 5], 8, 29)
    for arm in ARMS:
        model = activate(build(arm)).eval(); inner = model.layer
        model.layer = PadNaN(inner, on_pad=True)
        with torch.no_grad():
            ensure(bool(torch.isfinite(model.users(h, s)).all()), f'{arm}: NaN at PAD positions leaked into the user vector')
        model.layer = PadNaN(inner, on_pad=False)
        try:
            with torch.no_grad():
                model.users(h, s)
        except ValueError:
            continue
        raise AssertionError(f'{arm}: NaN at valid positions must raise')


def guard_eval_determinism():
    """G10: eval is deterministic; train mode has dropout noise; no_grad export path == eval+grad path within 1e-6."""
    h, s = make_batch([12, 12, 7], 12, 31)
    for arm in ARMS:
        model = activate(build(arm)).eval()
        with torch.no_grad():
            a = model.users(h, s); b = model.users(h, s)
        ensure(torch.equal(a, b), f'{arm}: eval is not deterministic')
        slow = model.users(h, s)
        ensure(torch.allclose(a, slow.detach(), atol=1e-6, rtol=0), f'{arm}: no_grad path differs from the grad path')
        model.train()
        ensure(not torch.equal(model.users(h, s), model.users(h, s)), f'{arm}: train mode has no dropout noise')


def guard_checkpoint_roundtrip():
    """G11: state_dict round trip keeps g and the branch bit for bit; a changed supported buffer is rejected."""
    for arm in ARMS:
        a = activate(build(arm, 42)); b = build(arm, 7)
        b.load_state_dict({k: v.clone() for k, v in a.state_dict().items()})
        for (n1, p1), (n2, p2) in zip(a.state_dict().items(), b.state_dict().items()):
            ensure(n1 == n2 and torch.equal(p1, p2), f'{arm}: {n1} differs after load')
        state = {k: v.clone() for k, v in a.state_dict().items()}
        state['supported'][1] = ~state['supported'][1]
        try:
            b.load_state_dict(state)
        except ValueError:
            continue
        raise AssertionError(f'{arm}: changed supported buffer accepted')


def guard_structure():
    """G14: arm table and the INSTANTIATED module attributes (not the constant table)."""
    from vkgr import retrieval_seqm1 as m
    ensure({arm: (a[0], a[1], a[2]) for arm, a in m.ARMS.items()} == ARM_TABLE, 'arm table differs from design 3.3')
    for arm in ARMS:
        model = build(arm); mask, readout, anchored = ARM_TABLE[arm]
        ensure((model.mask_kind, model.readout, model.anchored) == (mask, readout, anchored), f'{arm}: instance attributes')
        layer = model.layer
        ensure(isinstance(layer, torch.nn.TransformerEncoderLayer), f'{arm}: one TransformerEncoderLayer expected')
        ensure(layer.self_attn.num_heads == 4 and layer.self_attn.embed_dim == 64 and layer.linear1.out_features == 256, f'{arm}: 4 heads, d=64, FFN 256')
        ensure(layer.dropout.p == .1 and layer.dropout1.p == .1 and layer.dropout2.p == .1, f'{arm}: dropout 0.1')
        ensure(layer.norm_first is True, f'{arm}: pre-LN required')
        ensure(layer.activation is torch.nn.functional.relu, f'{arm}: ReLU required')
        ensure(bool((model.final_norm.weight == .1).all()) and bool((model.final_norm.bias == 0).all()), f'{arm}: LN_f init is weight 0.1, bias 0')
        ensure(model.order_table.num_embeddings == 51 and bool((model.order_table.weight[0] == 0).all()), f'{arm}: O has 51 rows with PAD row 0')
        ensure(hasattr(model, 'gate') == anchored, f'{arm}: gate exists iff anchored')
        if anchored:
            ensure(model.gate.ndim == 0 and float(model.gate) == 0. and not bool(torch.signbit(model.gate)), f'{arm}: gate must start at exactly +0')
        ensure(hasattr(model, 'attn_proj') == (readout == 'attn'), f'{arm}: attention pooling parameters iff attn')


def oracle_events(model, h, s):
    """Independent per-event anchor vector e_k = emb*sup + T + R + P[back index] + G[last week - week] (design G15d)."""
    out = torch.zeros(h.shape[0], h.shape[1], 64)
    for j in range(len(h)):
        n = int((h[j] != 0).sum()); weeks = s[j, :n, 3].tolist()
        for k in range(n):
            item = int(h[j, k]); ts, dur = int(s[j, k, 1]), int(s[j, k, 2])
            e = model.embedding.weight[item] * float(model.supported[item])
            e = e + model.timespent_bucket.weight[ts_bucket(ts)] + model.ratio_bucket.weight[ratio_id(ts, dur)]
            e = e + model.position_table.weight[ref_id(n - 1 - k, AUDIT_POSITION_EDGES)] + model.weekgap_table.weight[ref_id(weeks[-1] - weeks[k], AUDIT_WEEKGAP_EDGES)]
            out[j, k] = e
    return out


def guard_branch_input_mean_matches_anchor():
    """G15a: mean of the branch's per-event vectors equals the anchor user vector (all arms; random + unsupported + n=1 + n=50)."""
    for arm in ARMS:
        model = build(arm).eval(); anchor = build_php()
        for name, (h, s) in edge_cases().items():
            with torch.no_grad():
                p = model.parts(h, s)
                mean = p.e.sum(1) / p.valid.sum(1, keepdim=True)
                ensure(torch.allclose(mean, anchor.users(h, s), atol=1e-6, rtol=0), f'{arm}/{name}: mean of parts().e != anchor users')


def guard_branch_consumes_f3_f4():
    """G15b/c: the branch output r reacts to item / timespent / duration / week of a NON-last event (bucket-crossing edits);
    T, R, P, G receive gradient only through the branch path."""
    h, s = controlled_row()
    for arm in ARMS:
        model = activate(build(arm)).eval()
        with torch.no_grad():
            base = model.parts(h, s).r
            edits = {}
            s1 = s.clone(); s1[0, 2, 1] = other_timespent(int(s[0, 2, 1]))
            ensure(ts_bucket(int(s1[0, 2, 1])) != ts_bucket(int(s[0, 2, 1])), 'fixture: timespent bucket must change'); edits['timespent'] = (h, s1)
            s2 = s.clone(); s2[0, 3, 2] = 255
            ensure(ratio_id(int(s2[0, 3, 1]), 255) != ratio_id(int(s[0, 3, 1]), int(s[0, 3, 2])), 'fixture: ratio bucket must change'); edits['duration'] = (h, s2)
            s3 = s.clone(); s3[0, 2, 3] = 20
            from vkgr.retrieval_feat02 import week_gap, weekgap_bucket
            v = h != 0
            ensure(int(weekgap_bucket(week_gap(s3[..., 3], v), v)[0, 2]) != int(weekgap_bucket(week_gap(s[..., 3], v), v)[0, 2]), 'fixture: week-gap bucket must change'); edits['week'] = (h, s3)
            h4 = h.clone(); h4[0, 2] = 99; edits['item'] = (h4, s)
            for name, (eh, es) in edits.items():
                ensure(not torch.allclose(model.parts(eh, es).r, base, atol=1e-7, rtol=0), f'{arm}: branch output r ignores {name}')
        model = activate(build(arm)).eval()
        p = model.parts(*controlled_row())
        anchor_part = p.m.detach() if p.m is not None else torch.zeros_like(p.r)
        u = anchor_part + .5 * p.r
        u.sum().backward()
        for table in ('timespent_bucket', 'ratio_bucket', 'position_table', 'weekgap_table'):
            ensure(float(model.get_submodule(table).weight.grad.abs().sum()) > 0, f'{arm}: {table} gets no gradient through the branch')


def guard_per_event_oracle():
    """G15d (review F1): parts().e equals a hand-written per-event oracle, event by event, sample by sample; PAD rows are 0."""
    for arm in ARMS:
        model = build(arm).eval()
        for name, (h, s) in edge_cases().items():
            with torch.no_grad():
                e = model.parts(h, s).e
            ensure(torch.allclose(e, oracle_events(model, h, s), atol=1e-6, rtol=0), f'{arm}/{name}: per-event vectors differ from the oracle')
            ensure(bool((e[h == 0] == 0).all()), f'{arm}/{name}: PAD rows of e must be 0')


def guard_pure_identity():
    """G16 / AC17: pure has no gate, never calls the anchor users(), outputs z at the last valid position, shares all other parameters with `last`."""
    from vkgr.retrieval_php01 import F4IDMeanD
    h, s = make_batch([1, 6, 12], 12, 37)
    pure, last = build('pure'), build('last')
    ensure(not hasattr(pure, 'gate') and 'gate' not in dict(pure.named_parameters()), 'pure: no gate')
    for name, p in pure.named_parameters():
        ensure(torch.equal(p, dict(last.named_parameters())[name]), f'pure/last: {name} differs at the same seed')
    ensure(len(dict(last.named_parameters())) == len(dict(pure.named_parameters())) + 1, 'pure: exactly one parameter less')
    pure.eval()
    original = F4IDMeanD.users
    F4IDMeanD.users = lambda *a, **k: (_ for _ in ()).throw(AssertionError('pure called the anchor users()'))
    try:
        with torch.no_grad():
            u = pure.users(h, s); z = pure.parts(h, s).z
    finally:
        F4IDMeanD.users = original
    n = (h != 0).sum(1)
    ensure(torch.equal(u, torch.stack([z[j, n[j] - 1] for j in range(len(h))])), 'pure: users != z at the last valid position')
    ensure(pure.parts(h, s).m is None, 'pure: parts().m is None unless with_anchor=True')


# ---------------------------------------------------------------- tests

class ZeroStepTests(unittest.TestCase):
    def test_zero_step_equals_anchor(self):
        guard_zero_step_equals_anchor()

    def test_zero_step_after_dropout_noise(self):
        """r finite => m + 0*r == m even when dropout noise makes r random (train mode, repeated)."""
        anchor = build_php(); h, s = make_batch([12, 3], 12, 41)
        for arm in ANCHORED:
            model = build(arm).train()
            for _ in range(3):
                self.assertTrue(torch.equal(model.users(h, s), anchor.users(h, s)), arm)


class InitTests(unittest.TestCase):
    def test_init_order_and_anchor_tables(self):
        guard_init_order_and_anchor_tables()

    def test_parameter_report(self):
        guard_parameter_report()

    def test_structure(self):
        guard_structure()

    def test_unknown_arm_and_bad_inputs_rejected(self):
        from vkgr.retrieval_seqm1 import make_seqm1_model
        for arm in ('Mean', 'first', '', None, 3):
            with self.assertRaises(ValueError, msg=arm):
                make_seqm1_model(CONTENT, SUPPORTED, 50, arm)
        for max_len in (20, 100, 50.0):
            with self.assertRaises(ValueError, msg=max_len):
                make_seqm1_model(CONTENT, SUPPORTED, max_len, 'mean')

    def test_same_seed_is_deterministic_and_seeds_differ(self):
        for arm in ARMS:
            a, b, c = build(arm, 42).state_dict(), build(arm, 42).state_dict(), build(arm, 7).state_dict()
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a), arm)
            self.assertFalse(torch.equal(a['order_table.weight'], c['order_table.weight']), arm)


class MaskPaddingTests(unittest.TestCase):
    def test_padding_invariant(self):
        guard_padding_invariant()

    def test_padding_invariance_is_bitwise_on_cpu_or_within_atol(self):
        """Informational tightening (N1): record whether CPU is even bit for bit; never loosen the atol."""
        h, s = make_batch([4, 9], 9, 43); model = activate(build('mean')).eval()
        ph, ps = pad_to(h, s, 50)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model.users(h, s), model.users(ph, ps), atol=1e-6, rtol=0))

    def test_mask_semantics(self):
        guard_mask_semantics()

    def test_back_index_ids(self):
        from vkgr.retrieval_seqm1 import order_ids
        h, _ = make_batch([1, 3, 5], 5, 3)
        ids = order_ids(h != 0)
        self.assertEqual(ids.tolist(), [[1, 0, 0, 0, 0], [3, 2, 1, 0, 0], [5, 4, 3, 2, 1]])

    def test_appending_an_older_event_keeps_recent_order_rows(self):
        from vkgr.retrieval_seqm1 import order_ids
        h = torch.tensor([[5, 6, 7, 0]]); older = torch.tensor([[9, 5, 6, 7]])
        self.assertEqual(order_ids(h != 0)[0, :3].tolist(), order_ids(older != 0)[0, 1:].tolist())


class ReadoutTests(unittest.TestCase):
    def test_readouts(self):
        guard_readouts()


class GradientTests(unittest.TestCase):
    def test_gradients(self):
        guard_gradients()


class RobustnessTests(unittest.TestCase):
    def test_left_aligned(self):
        guard_left_aligned()

    def test_check_on_every_entry(self):
        guard_check_on_every_entry()

    def test_finiteness(self):
        guard_finiteness()

    def test_eval_determinism(self):
        guard_eval_determinism()

    def test_checkpoint_roundtrip(self):
        guard_checkpoint_roundtrip()


class BranchInputTests(unittest.TestCase):
    def test_mean_of_event_vectors_matches_anchor(self):
        guard_branch_input_mean_matches_anchor()

    def test_branch_consumes_f3_f4(self):
        guard_branch_consumes_f3_f4()

    def test_per_event_oracle(self):
        guard_per_event_oracle()


class PureTests(unittest.TestCase):
    def test_pure_identity(self):
        guard_pure_identity()


class BranchSpecTests(unittest.TestCase):
    def test_spec_hash_reads_the_instantiated_module(self):
        from vkgr.retrieval_seqm1 import branch_spec, branch_spec_sha256
        for arm in ARMS:
            model = build(arm); digest = branch_spec_sha256(model)
            self.assertEqual(len(digest), 64); self.assertEqual(digest, branch_spec_sha256(build(arm)))
            spec = branch_spec(model)
            self.assertEqual((spec['num_layers'], spec['heads'], spec['ffn'], spec['norm_first'], spec['order_rows']), (1, 4, 256, True, 51), arm)
            model.layer.norm_first = False
            self.assertNotEqual(branch_spec_sha256(model), digest, arm)
        self.assertNotEqual(branch_spec_sha256(build('mean')), branch_spec_sha256(build('last')))
        self.assertNotEqual(branch_spec_sha256(build('last')), branch_spec_sha256(build('pure')))
        for edit in (lambda m: setattr(m.layer.self_attn, 'num_heads', 2), lambda m: m.final_norm.weight.data.fill_(1.), lambda m: setattr(m.layer.dropout, 'p', .2)):
            model = build('mean'); base = branch_spec_sha256(model); edit(model)
            self.assertNotEqual(branch_spec_sha256(model), base)


if __name__ == '__main__':
    unittest.main()
