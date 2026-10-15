"""FEAT-04b model tests, scaffold + chain A stage (``f4_control`` and ``f4_recency_pool`` on the F4-T010 anchor; C/D/E come in later features).

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); the unit tests call them and
tests/test_feat04b_mutations.py re-runs them under deliberately wrong implementations.
"""
import random
import re
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat02_model as tm  # noqa: E402
import test_feat03_model as t3  # noqa: E402
import test_feat04b_stats as ts  # noqa: E402

ensure = tm.ensure
CONTROL, A, E = 'f4_control', 'f4_recency_pool', 'f4_recency_behavior'
C, D = 'f4_item_popularity', 'f4_item_quality'
CD = (C, D)
FEATURES4B_LITERAL = ('f4_control', 'f4_recency_pool', 'f4_item_popularity', 'f4_item_quality', 'f4_recency_behavior')
IMPLEMENTED_NOW = (CONTROL, A, E)
ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CALLS = re.compile(r'^\s*from\s+\.?(?:vkgr\.)?(retrieval_feat02|retrieval_feat02_run|retrieval_feat03|retrieval_feat03_run)\s+import\b', re.M)
SCANNED_SOURCES = ('retrieval_feat04b.py', 'retrieval_feat04b_run.py')          # item_stats.py joins the list when it exists (FEAT-002)


def m4b():
    from vkgr import retrieval_feat04b
    return retrieval_feat04b


def build4b(variant, seed=42):
    return m4b().make_feat04b_model(variant, tm.CONTENT, tm.SUPPORTED, 50, seed)


# ---------------------------------------------------------------- C/D fixtures: item id == model index for items 1..300 (items 200..300 are unsupported)
CD_CATALOG = list(range(1, 301))
CD_IDMAP = {i: i for i in CD_CATALOG}


def cd_stats(seed=3):
    """Seeded ITEM_STATS_V1 columns that spread the items over many buckets of every table (a mis-indexed table is visible)."""
    rng = random.Random(seed); stats = {'n': [], 'first_week': [], 'likes': [], 'ratio_pm_sum': []}
    for _ in CD_CATALOG:
        n = rng.choice([1, 2, 3, 4, 5, 9, 17, 33, 65, 129, 257, 513, 900])
        stats['n'].append(n); stats['first_week'].append(rng.choice([0, 1, 2, 4, 7, 11, 16, 21, 24]))
        stats['likes'].append(rng.randint(0, max(1, n // 2) if n > 1 else 0)); stats['ratio_pm_sum'].append(n * rng.choice([50, 250, 400, 600, 700, 900, 1100, 1500]) + rng.randint(0, n))
    return stats


def cd_buckets(stats=None):
    return m4b().stat_bucket_ids(cd_stats() if stats is None else stats, CD_CATALOG, CD_IDMAP, 301)


def build_cd(variant, seed=42, buckets=None):
    return m4b().make_feat04b_model(variant, tm.CONTENT, tm.SUPPORTED, 50, seed, stat_buckets=cd_buckets() if buckets is None else buckets)


def build_any(variant, seed=42):
    return build_cd(variant, seed) if variant in CD else build4b(variant, seed)


# ---------------------------------------------------------------- shared guards (guard_s_*)

def guard_s_default_off(model=None):
    """f4_control / chain A: history_items and candidate_items are items, bit for bit (the C/D seam is off by default)."""
    indices = torch.arange(0, 301)
    for variant in IMPLEMENTED_NOW:
        m = build4b(variant) if model is None else model
        for name in ('history_items', 'candidate_items'):
            ensure(hasattr(m, name), f'{variant}: {name} seam missing')
            ensure(torch.equal(getattr(m, name)(indices), m.items(indices)), f'{variant}: {name} differs from items')
            grid = indices[:300].view(3, 100)
            ensure(torch.equal(getattr(m, name)(grid), m.items(grid)), f'{variant}: {name} differs from items (2-d)')
        if model is not None:
            break


def guard_s_anchor_tables_bitwise():
    """Chain A draws item, T, R, P, G exactly like f4_control at the same seed; the control is the FEAT-03 control and the FEAT-02 F4 model, bit for bit."""
    from vkgr import retrieval_feat03 as f3
    control = build4b(CONTROL).state_dict()
    for other in (f3.make_feat03_model(CONTROL, tm.CONTENT, tm.SUPPORTED, 50, 42).state_dict(), tm.build('f3_f4_recency').state_dict()):
        ensure(list(control) == list(other), 'f4_control state_dict keys differ from the FEAT-03 control / F4')
        for key in control:
            ensure(torch.equal(control[key], other[key]), f'f4_control table {key} differs from the FEAT-03 control / F4')
    for variant in (A, E, C, D):
        state = build_any(variant).state_dict()
        for key in control:
            ensure(key in state and torch.equal(state[key], control[key]), f'{variant}: anchor table {key} differs from f4_control')


def guard_s_spec_hash_per_variant():
    """bucket_spec_sha256 is per variant: stable, distinct, a sha256; a constant that belongs to one variant moves only that variant's digest (C: expo/first edges,
    D: ratio/like edges, K, P0, the ratio cap; C and D: the cutoff week; E: the feedback-bit order); unregistered variants are refused."""
    import contextlib
    from unittest import mock
    from vkgr import item_stats
    m = m4b(); every = IMPLEMENTED_NOW + CD
    digests = {v: m.bucket_spec_sha256(v) for v in every}
    ensure(all(m.bucket_spec_sha256(v) == d for v, d in digests.items()), 'bucket_spec_sha256 is not stable')
    ensure(len(set(digests.values())) == len(digests), 'every variant must have its own bucket_spec_sha256')
    ensure(all(len(d) == 64 and set(d) <= set('0123456789abcdef') for d in digests.values()), 'bucket_spec_sha256 must be a sha256 hex digest')
    for bad in ('nope', 'f3_pos_only'):
        try:
            m.bucket_spec_sha256(bad)
        except ValueError:
            continue
        raise AssertionError(f'bucket_spec_sha256({bad!r}) must raise ValueError')

    def moved(module, name, value):
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(module, name, value))
            return {v for v in every if m.bucket_spec_sha256(v) != digests[v]}
    for name, value in (('LIKE_P0', 37), ('LIKE_K', 3), ('RATIO_EDGES', (200, 400)), ('LIKE_EDGES', (7, 9))):
        ensure(moved(m, name, value) == {D}, f'{name} must move only the digest of D, moved {moved(m, name, value)}')
    for name, value in (('EXPO_EDGES', (3, 6)), ('FIRST_EDGES', (2, 9))):
        ensure(moved(m, name, value) == {C}, f'{name} must move only the digest of C, moved {moved(m, name, value)}')
    ensure(moved(item_stats, 'RATIO_CAP_PM', 1500) == {D}, 'the ratio cap must move only the digest of D')
    ensure(moved(item_stats, 'STAT_CUTOFF_WEEK', 25) == {C, D}, 'the cutoff week must move the digests of C and D only')
    ensure(moved(m, 'BEHAVIOR_BITS', tuple(reversed(m.BEHAVIOR_BITS))) == {E}, 'the feedback-bit order must move only the digest of E')


def guard_s_module_attr_scan(source_text):
    """K2: no source may import a FEAT-02/03 name directly (seams are resolved through the module at call time)."""
    ensure(FORBIDDEN_CALLS.search(source_text) is None, 'from .retrieval_featXX import <name> breaks the module-attribute seams')


def guard_s_module_attribute_calls():
    for name in SCANNED_SOURCES:
        path = ROOT / 'src' / 'vkgr' / name
        if name == 'retrieval_feat04b_run.py' and not path.exists():
            continue                                                       # the runner arrives in item 4
        ensure(path.exists(), f'{name} missing')
        guard_s_module_attr_scan(path.read_text())


# ---------------------------------------------------------------- chain A guards (guard_a_*)

def guard_a_start_equals_anchor():
    """AC3: with both logit vectors at 0 the A user vector is the f4_control user vector (and the FEAT-02 F4 one) bit for bit; PAD columns, supported/unsupported mixes."""
    model, control, f4 = build4b(A), build4b(CONTROL), tm.build('f3_f4_recency')
    histories, side = tm.model_inputs(); valid = histories != 0
    ensure(bool((~valid).any()), 'fixture needs PAD columns in a mixed-length batch')
    ensure(bool((valid & ~model.supported[histories]).any()) and bool((valid & model.supported[histories]).any()), 'fixture needs supported and unsupported events')
    for h, s in ((histories, side), (tm.all_supported(histories), side), (tm.all_unsupported(histories), side), t3.pad_columns(histories, side, 7)):
        users = model.users(h, s)
        ensure(torch.equal(users, control.users(h, s)), 'A at its start differs from f4_control')
        ensure(torch.equal(users, f4.users(h, s)), 'A at its start differs from the FEAT-02 F4 model')
    ensure(torch.equal(model.items(torch.arange(0, 301)), control.items(torch.arange(0, 301))), 'A item tower differs from the anchor')


def guard_a_formula():
    """A user = sum_k w_k x_k / sum_k w_k with w_k = exp(l_k - max l), l_k = u[pb_k - 1] + v[gb_k - 1], x_k the F4 event vector (non-uniform logits)."""
    model = t3.set_logits(build4b(A)); histories, side = tm.model_inputs(); valid = histories != 0
    x, pos_ids, gap_ids = t3.reference_event_vectors(model, histories, side, valid)
    ensure(int(pos_ids.amax()) >= 3 and int((pos_ids == 1).sum()) >= 2, 'fixture must span several position buckets with n >= 2')
    ensure(int(gap_ids.amax()) == 6 and len(set(gap_ids[valid].tolist())) >= 4, 'fixture must span several week-gap buckets')
    ensure(len(set(model.recency_logit_pos.tolist()) | set(model.recency_logit_gap.tolist())) == 13, 'all logit values must be distinct')
    ell = model.recency_logit_pos[pos_ids - 1] + model.recency_logit_gap[gap_ids - 1]
    expected = torch.zeros(len(histories), 64)
    for j in range(len(histories)):
        n = int(valid[j].sum()); w = torch.exp(ell[j, :n] - ell[j, :n].max())
        expected[j] = (x[j, :n] * w.unsqueeze(-1)).sum(0) / w.sum()
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-6, rtol=0), 'A is not the normalised weighted mean')


def guard_a_pad_width_invariance():
    """Appending PAD columns changes nothing, bit for bit, also with non-zero logits."""
    model = t3.set_logits(build4b(A)); histories, side = tm.model_inputs()
    base = model.users(histories, side)
    for extra in (1, 5, 20):
        ensure(torch.equal(model.users(*t3.pad_columns(histories, side, extra)), base), f'A changed when {extra} PAD columns were appended')


def guard_a_logits_receive_gradient():
    """Both logit vectors receive finite, non-zero gradients on every used row at the zero start."""
    model = build4b(A)
    histories, side = tm.model_inputs(); valid = histories != 0
    _, pos_ids, gap_ids = t3.reference_event_vectors(model, histories, side, valid)
    projection = torch.randn(64, generator=torch.Generator().manual_seed(9))
    (model.users(histories, side) @ projection).sum().backward()
    for name, ids in (('recency_logit_pos', pos_ids), ('recency_logit_gap', gap_ids)):
        p = getattr(model, name)
        ensure(p.grad is not None, f'{name}: no gradient (logits detached or unused?)')
        ensure(bool(torch.isfinite(p.grad).all()), f'{name}: non-finite gradient')
        used = sorted(set((ids[valid] - 1).tolist()))
        ensure(len(used) >= 3, 'fixture must use several rows')
        for row in used:
            ensure(float(p.grad[row].abs()) > 0, f'{name}: used row {row} got no gradient')


def guard_a_param_count_13():
    """13 nn.Parameter scalars beyond the anchor, exactly 0 at init; the parameter report equals the formula."""
    model, control = build4b(A), build4b(CONTROL)
    extra = set(dict(model.named_parameters())) - set(dict(control.named_parameters()))
    ensure(extra == {'recency_logit_pos', 'recency_logit_gap'}, f'unexpected extra parameters {sorted(extra)}')
    ensure(sum(p.numel() for n, p in model.named_parameters() if n in extra and p.requires_grad) == 13, 'exactly 13 trainable scalars beyond the anchor')
    ensure(all(isinstance(getattr(model, n), torch.nn.Parameter) and bool((getattr(model, n) == 0).all()) for n in extra), 'logits must start exactly at 0')
    ensure(m4b().parameter_report4b(model) == m4b().expected_param_counts4b(A), 'parameter report differs from the formula')
    ensure(m4b().parameter_report4b(control) == m4b().expected_param_counts4b(CONTROL), 'control parameter report differs from the formula')


def guard_a_pooled_weights_pad_cannot_overflow():
    """pooled_weights is a double where: a 1e4 PAD logit gives finite weights, PAD weight exactly 0 and a finite backward pass (M82's guard)."""
    from vkgr import retrieval_feat03 as f3
    valid = torch.tensor([[True, True, True, False, False], [True, True, False, False, False], [True, True, True, True, True]])
    logits = torch.tensor([[0.1, -0.2, 0.3, 1e4, 1e4], [0.0, 0.2, 1e4, 1e4, 1e4], [0.5, -0.5, 0.0, 0.25, -0.25]], requires_grad=True)
    w = f3.pooled_weights(logits, valid)
    ensure(bool(torch.isfinite(w).all()), 'pooled_weights returned a non-finite weight')
    ensure(bool((w[~valid] == 0).all()), 'PAD weight must be exactly 0')
    w.sum().backward()
    ensure(bool(torch.isfinite(logits.grad).all()), 'backward through pooled_weights is not finite with a 1e4 PAD logit')


# ---------------------------------------------------------------- E guards (guard_e_*): A plus six behaviour-bit logits

def set_beta(model, seed=21, scale=0.7):
    """Distinct non-zero beta values (all 6 different)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.behavior_beta.copy_(torch.randn(6, generator=g) * scale)
    return model


def e_model(seed=42):
    """E with fixed-seed non-zero u, v and beta (a beta = 0 fixture only proves 'start equals A'; M70b adds 0 and would be invisible there)."""
    return set_beta(t3.set_logits(build4b(E, seed)))


def reference_bits(mask):
    """Independent bit extraction in plain Python: bit b of the feedback mask, b = 0..5 (history_side.FEEDBACK_BITS order)."""
    return [(int(mask) >> b) & 1 for b in range(6)]


def guard_e_start_equals_a():
    """AC3: E with beta = 0 has the A user vector bit for bit (zero logits, and non-zero shared u, v; PAD columns, supported/unsupported mixes); the item tower is the anchor's."""
    e, a, control = build4b(E), build4b(A), build4b(CONTROL)
    histories, side = tm.model_inputs(); valid = histories != 0
    ensure(bool((~valid).any()) and bool((valid & ~e.supported[histories]).any()) and bool((valid & e.supported[histories]).any()), 'fixture needs PAD columns and a supported/unsupported mix')
    ensure(bool((side[..., 0][valid] != 0).any()), 'fixture needs non-zero feedback masks')
    ensure(bool((e.behavior_beta == 0).all()), 'beta must start exactly at 0')
    for shared in (False, True):
        if shared:
            t3.set_logits(a, seed=8); t3.set_logits(e, seed=8)
        for h, s in ((histories, side), (tm.all_supported(histories), side), (tm.all_unsupported(histories), side), t3.pad_columns(histories, side, 7)):
            ensure(torch.equal(e.users(h, s), a.users(h, s)), f'E at beta = 0 differs from A (shared non-zero logits: {shared})')
            if not shared:
                ensure(torch.equal(e.users(h, s), control.users(h, s)), 'E at its start differs from f4_control (zero logits must be the uniform mean)')
    ensure(torch.equal(e.items(torch.arange(0, 301)), a.items(torch.arange(0, 301))), 'E item tower differs from the anchor')


def guard_e_formula():
    """l_k = u[pb_k - 1] + v[gb_k - 1] + sum_b bit_b(mask_k) * beta_b; user = sum_k exp(l_k - max l) x_k / sum_k exp(l_k - max l), against a plain-Python reference."""
    model = e_model(); histories, side = tm.model_inputs(); valid = histories != 0
    x, pos_ids, gap_ids = t3.reference_event_vectors(model, histories, side, valid)
    ensure(int(pos_ids.amax()) >= 3 and int(gap_ids.amax()) == 6, 'fixture must span several position / week-gap buckets')
    ensure(len(set(model.behavior_beta.tolist())) == 6 and bool((model.behavior_beta != 0).all()), 'beta must be distinct and non-zero')
    expected = torch.zeros(len(histories), 64)
    u, v, beta = model.recency_logit_pos.tolist(), model.recency_logit_gap.tolist(), model.behavior_beta.tolist()
    for j in range(len(histories)):
        n = int(valid[j].sum()); ell = []
        for k in range(n):
            bits = reference_bits(side[j, k, 0])
            ell.append(u[int(pos_ids[j, k]) - 1] + v[int(gap_ids[j, k]) - 1] + sum(b * c for b, c in zip(bits, beta)))
        ell = torch.tensor(ell); w = torch.exp(ell - ell.max())
        expected[j] = (x[j, :n] * w.unsqueeze(-1)).sum(0) / w.sum()
    users = model.users(histories, side)
    ensure(torch.allclose(users, expected, atol=1e-6, rtol=0), 'E is not the normalised weighted mean with behaviour logits')
    a = build4b(A); a.recency_logit_pos.data.copy_(model.recency_logit_pos.data); a.recency_logit_gap.data.copy_(model.recency_logit_gap.data)
    ensure(not torch.allclose(users, a.users(histories, side), atol=1e-4, rtol=0), 'beta must move the user vector (fixture too weak)')


def guard_e_bit_order():
    """Single-bit fixtures: mask = 1 << b gives exactly beta[b] at that slot and 0 at every other slot (bit b is FEEDBACK_BITS[b]); two bits give the sum;
    the bit sits on its own event (no one-slot shift) and PAD slots stay 0."""
    from vkgr.history_side import FEEDBACK_BITS
    m = m4b(); beta = torch.tensor([0.3, -0.7, 1.1, 0.5, -1.3, 2.0])
    ensure(tuple(m.BEHAVIOR_BITS) == tuple(FEEDBACK_BITS) and FEEDBACK_BITS == ('like', 'share', 'bookmark', 'open_comments', 'click_on_author', 'dislike'), 'bit order is not the registered FEEDBACK_BITS order')
    for b in range(6):
        mask = torch.zeros(1, 5, dtype=torch.long); mask[0, 2] = 1 << b
        valid = torch.tensor([[True, True, True, True, False]])
        got = m.behavior_logits(beta, mask, valid)
        ensure(got.shape == (1, 5), 'behavior_logits must return (B, W)')
        expect = torch.zeros(1, 5); expect[0, 2] = beta[b]
        ensure(torch.equal(got, expect), f'bit {b} ({FEEDBACK_BITS[b]}) does not select beta[{b}] on its own slot: {got.tolist()}')
    mask = torch.tensor([[0b100001, 0b000110, 0, 0b111111]]); valid = torch.ones(1, 4, dtype=torch.bool)
    got = m.behavior_logits(beta, mask, valid)
    ensure(torch.allclose(got, torch.tensor([[beta[0] + beta[5], beta[1] + beta[2], 0.0, float(beta.sum())]]), atol=1e-6, rtol=0), 'multi-bit masks are not the sum of their betas')
    ensure(bool((m.behavior_logits(torch.zeros(6), mask, valid) == 0).all()) and not bool(torch.isnan(got).any()), 'beta = 0 must give exact 0.0 logits')
    pad = torch.zeros(1, 3, dtype=torch.long); pad[0, 0] = 1
    ensure(bool((m.behavior_logits(beta, pad, torch.tensor([[True, False, False]]))[0, 1:] == 0).all()), 'PAD slots must stay 0')


def guard_e_logit_path_only():
    """Beta enters the pooling logits only: when every x_k is the same, beta cannot change the user vector. (M70a adds beta to x_k, M70b to x_k and the logit.)"""
    model = e_model()
    with torch.no_grad():
        model.position_table.weight.zero_(); model.weekgap_table.weight.zero_()          # the P / G terms are the only per-event difference of equal events
    n = 6; histories = torch.full((2, n), 7, dtype=torch.long); histories[1, 4:] = 0
    side = torch.zeros(2, n, 6, dtype=torch.long)
    for j in range(2):
        for k in range(n if j == 0 else 4):
            side[j, k] = torch.tensor([(k * 11 + 3) % 64, 20, 25, 5, 0, 0])               # same item, timespent, duration, week; different feedback masks
    ensure(len(set(side[0, :, 0].tolist())) == n, 'fixture: every event needs its own mask')
    on = model.users(histories, side)
    with torch.no_grad():
        model.behavior_beta.zero_()
    off = model.users(histories, side)
    ensure(bool(torch.isfinite(on).all()) and float(on.detach().abs().max()) > 0, 'fixture: non-trivial user vector needed')
    ensure(torch.allclose(on, off, atol=1e-6, rtol=0), 'beta moved the user vector although every x_k is equal: it must act on the logit path only')


def guard_e_beta_receive_gradient():
    """Beta is a leaf with a finite gradient: bits that occur in the batch get a non-zero gradient, bits that never occur get exactly 0."""
    model = e_model(); histories, side = tm.model_inputs(); valid = histories != 0
    side = side.clone(); side[..., 0] = side[..., 0] & 0b001011                          # bits 2, 4, 5 never occur
    projection = torch.randn(64, generator=torch.Generator().manual_seed(9))
    (model.users(histories, side) @ projection).sum().backward()
    grad = model.behavior_beta.grad
    ensure(grad is not None, 'behavior_beta received no gradient (detached or unused?)')
    ensure(bool(torch.isfinite(grad).all()), 'behavior_beta gradient is not finite')
    present = {b for b in range(6) if bool(((side[..., 0][valid] >> b) & 1).any())}
    ensure(present == {0, 1, 3}, 'fixture: expected bits 0, 1, 3 to occur')
    for b in range(6):
        if b in present:
            ensure(float(grad[b].abs()) > 0, f'bit {b} occurs but got no gradient')
        else:
            ensure(float(grad[b]) == 0.0, f'bit {b} never occurs but got gradient {float(grad[b])}')


def guard_e_param_count_19():
    """19 trainable scalars beyond the anchor (13 recency + 6 behaviour), beta exactly 0 at init and a leaf Parameter; report equals formula."""
    model, control = build4b(E), build4b(CONTROL)
    extra = set(dict(model.named_parameters())) - set(dict(control.named_parameters()))
    ensure(extra == {'recency_logit_pos', 'recency_logit_gap', 'behavior_beta'}, f'unexpected extra parameters {sorted(extra)}')
    ensure(sum(p.numel() for n, p in model.named_parameters() if n in extra and p.requires_grad) == 19, 'exactly 19 trainable scalars beyond the anchor')
    ensure(isinstance(model.behavior_beta, torch.nn.Parameter) and model.behavior_beta.shape == (6,) and bool((model.behavior_beta == 0).all()), 'beta must be a zero (6,) Parameter')
    ensure(m4b().parameter_report4b(model) == m4b().expected_param_counts4b(E), 'parameter report differs from the formula')
    ensure(m4b().expected_param_counts4b(E)['trainable_effective'] == m4b().expected_param_counts4b(A)['trainable_effective'] + 6, 'E must add exactly 6 trainable scalars to A')


def guard_e_recency_logits_receive_gradient():
    """E also trains the 13 recency logits (the behaviour term does not replace them): both vectors get finite gradients on every used row."""
    model = e_model(); histories, side = tm.model_inputs(); valid = histories != 0
    _, pos_ids, gap_ids = t3.reference_event_vectors(model, histories, side, valid)
    projection = torch.randn(64, generator=torch.Generator().manual_seed(9))
    (model.users(histories, side) @ projection).sum().backward()
    for name, ids in (('recency_logit_pos', pos_ids), ('recency_logit_gap', gap_ids)):
        p = getattr(model, name)
        ensure(p.grad is not None and bool(torch.isfinite(p.grad).all()), f'E {name}: missing or non-finite gradient')
        for row in sorted(set((ids[valid] - 1).tolist())):
            ensure(float(p.grad[row].abs()) > 0, f'E {name}: used row {row} got no gradient')


def guard_e_pad_cannot_overflow_with_extreme_beta():
    """Every event carries the like bit and beta[like] = -1e4, so the PAD logit (no behaviour term) is ~1e4 above every valid logit: the double where in pooled_weights
    must keep the weights finite, PAD weight exactly 0 and the backward pass finite (M82 at the E level)."""
    model = e_model()
    with torch.no_grad():
        model.behavior_beta.zero_(); model.behavior_beta[0] = -1e4
    histories, side = tm.model_inputs(); valid = histories != 0
    side = side.clone(); side[..., 0] = torch.where(valid, side[..., 0] | 1, torch.zeros_like(side[..., 0]))
    ensure(bool((~valid).any()), 'fixture needs PAD columns')
    users = model.users(histories, side)
    ensure(bool(torch.isfinite(users).all()), 'E user vectors are not finite with an extreme beta')
    users.sum().backward()
    ensure(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'E backward is not finite with an extreme beta (PAD overflow)')


def guard_e_beta_init_draws_no_random_number():
    """beta is built from zeros: building E draws exactly what building A draws (the RNG state after construction is the same)."""
    build4b(A); state_a = torch.get_rng_state(); build4b(E); state_e = torch.get_rng_state()
    ensure(torch.equal(state_a, state_e), 'building E consumed a different number of random draws than A')


# ---------------------------------------------------------------- C/D guards (guard_cd_*): candidate-side bucket tables from ITEM_STATS_V1

def cd_table_ids(variant, stats=None):
    """Independent (plain Python) bucket ids per catalog item for the two tables of ``variant``, from the raw stats and the guard's own grid."""
    st = cd_stats() if stats is None else stats
    if variant == C:
        return [(ts.ref_expo(n), ts.ref_first(f)) for n, f in zip(st['n'], st['first_week'])]
    return [(ts.ref_ratio(r, n), ts.ref_like(l, n)) for n, l, r in zip(st['n'], st['likes'], st['ratio_pm_sum'])]


def cd_tables(model, variant):
    names = [t for t, _, _ in m4b().STAT_LAYOUT[variant]]
    return [getattr(model, n) for n in names]


def guard_cd_history_path_unchanged():
    """C/D leave the user tower alone: history_items is items, and users are bit-equal to f4_control with the same seed (PAD columns, supported/unsupported mixes)."""
    control = build4b(CONTROL); histories, side = tm.model_inputs()
    ensure(bool((histories != 0).any() and (histories == 0).any()), 'fixture needs PAD columns')
    for variant in CD:
        model = build_cd(variant)
        ensure(torch.equal(model.history_items(torch.arange(0, 301)), model.items(torch.arange(0, 301))), f'{variant}: history_items differs from items')
        for h, sd in ((histories, side), (tm.all_supported(histories), side), (tm.all_unsupported(histories), side), t3.pad_columns(histories, side, 6)):
            ensure(torch.equal(model.users(h, sd), control.users(h, sd)), f'{variant}: users differ from f4_control (the bucket tables leaked into the history path)')
        ensure(not torch.equal(model.candidate_items(torch.arange(0, 301)), model.items(torch.arange(0, 301))), f'{variant}: candidate_items must differ from items on supported items')


def guard_cd_candidate_formula():
    """candidate(i) = items(i) + supported(i) * (T1[b1(i)] + T2[b2(i)]) against plain-Python bucket ids; unsupported and PAD candidates are exactly 0; the vectors differ from f4_control
    on supported items only."""
    control = build4b(CONTROL); indices = torch.arange(0, 301); supported = set(tm.SUPPORTED)
    for variant in CD:
        model = build_cd(variant); got = model.candidate_items(indices); base = model.items(indices); t1, t2 = cd_tables(model, variant)
        ids = cd_table_ids(variant)
        ensure(all(len({row[k] for row in ids}) >= 3 for k in (0, 1)), f'{variant}: fixture must spread the items over several buckets of each table')
        for i in CD_CATALOG:
            if i in supported:
                expected = base[i] + (t1.weight[ids[i - 1][0]] + t2.weight[ids[i - 1][1]])
                ensure(torch.allclose(got[i], expected, atol=1e-7, rtol=0), f'{variant}: candidate vector of supported item {i} differs from the formula')
            else:
                ensure(bool((got[i] == 0).all()), f'{variant}: unsupported item {i} must have an exactly zero candidate vector')
        ensure(bool((got[0] == 0).all()), f'{variant}: PAD candidate must be 0')
        differs = (got != control.candidate_items(indices)).any(-1)
        ensure(set(differs.nonzero().flatten().tolist()) == supported, f'{variant}: candidate vectors must differ from the anchor on supported items and nowhere else')


def guard_cd_unsupported_zero():
    """Whatever the table contents, an unsupported (or PAD) candidate stays exactly 0: the supported mask multiplies the bucket terms."""
    for variant in CD:
        model = build_cd(variant)
        with torch.no_grad():
            for table in cd_tables(model, variant):
                table.weight.copy_(torch.randn(table.weight.shape, generator=torch.Generator().manual_seed(2)))
        got = model.candidate_items(torch.arange(0, 301))
        ensure(bool((got[200:] == 0).all()) and bool((got[0] == 0).all()), f'{variant}: unsupported / PAD candidates are not exactly zero')
        ensure(bool((got[1:200].abs().sum(-1) > 0).all()), f'{variant}: supported candidates must be non-zero')


def guard_cd_tables_receive_gradient():
    """Both bucket tables of the variant get finite gradients through candidate_items on every bucket used by a supported item; the PAD row gets exactly 0."""
    for variant in CD:
        model = build_cd(variant); indices = torch.arange(1, 301)
        projection = torch.randn(64, generator=torch.Generator().manual_seed(9))
        (model.candidate_items(indices) @ projection).sum().backward()
        ids = cd_table_ids(variant)
        for k, table in enumerate(cd_tables(model, variant)):
            ensure(table.weight.grad is not None, f'{variant}: table {k} received no gradient (candidate path detached or unused?)')
            ensure(bool(torch.isfinite(table.weight.grad).all()), f'{variant}: table {k} gradient is not finite')
            ensure(bool((table.weight.grad[0] == 0).all()), f'{variant}: PAD row of table {k} must get exactly 0')
            used = sorted({ids[i - 1][k] for i in tm.SUPPORTED})
            ensure(len(used) >= 3, 'fixture must use several buckets')
            for row in used:
                ensure(float(table.weight.grad[row].abs().sum()) > 0, f'{variant}: used bucket {row} of table {k} got no gradient')


def guard_cd_init_and_draw_order():
    """Anchor tables are bit-equal to f4_control's; each bucket table is N(0, 0.02) with a zero PAD row (not all zero: the padding_idx=None trap), and the draws are
    construct -> normal_ -> clear PAD, table by table, right after the P / G draws."""
    for variant in CD:
        model = build_cd(variant); control = build4b(CONTROL)               # the control build leaves the RNG exactly where the C/D stat-table draws start
        for key, value in control.state_dict().items():
            ensure(torch.equal(model.state_dict()[key], value), f'{variant}: anchor entry {key} differs from f4_control')
        for table, _, edges in m4b().STAT_LAYOUT[variant]:
            weight = getattr(model, table).weight
            ensure(weight.shape == (len(getattr(m4b(), edges)) + 2, 64) and getattr(model, table).padding_idx == 0, f'{variant}: {table} shape / padding_idx')
            ensure(bool((weight[0] == 0).all()) and bool((weight[1:] != 0).any()), f'{variant}: {table} PAD row zero and the rest not zero')
            std = float(weight.detach()[1:].std())
            ensure(0.017 <= std <= 0.023, f'{variant}: {table} std {std} is not N(0, 0.02)')
            ref = torch.nn.Embedding(weight.shape[0], 64, padding_idx=0)
            with torch.no_grad():
                ref.weight.normal_(0.0, 0.02); ref.weight[0].zero_()
            ensure(torch.equal(weight, ref.weight), f'{variant}: {table} is not the next N(0, 0.02) draw after the P / G tables')
        ensure(all(e['rows'] > 0 for e in model.init_report['embeddings'].values()) and all(t in model.init_report['embeddings'] for t, _, _ in m4b().STAT_LAYOUT[variant]),
               f'{variant}: init_report must cover the stat tables')


def guard_cd_param_counts():
    """Parameter report == formula == independent count: exactly the two bucket tables beyond the anchor, PAD rows not trainable."""
    control = build4b(CONTROL)
    for variant in CD:
        model = build_cd(variant)
        extra = set(dict(model.named_parameters())) - set(dict(control.named_parameters()))
        names = {f'{t}.weight' for t, _, _ in m4b().STAT_LAYOUT[variant]}
        ensure(extra == names, f'{variant}: unexpected extra parameters {sorted(extra)}')
        rows = sum(len(getattr(m4b(), edges)) + 2 - 1 for _, _, edges in m4b().STAT_LAYOUT[variant])
        report, base = m4b().parameter_report4b(model), m4b().parameter_report4b(control)
        ensure(report['trainable_effective'] - base['trainable_effective'] == rows * 64, f'{variant}: trainable count differs from (rows - PAD) * 64')
        ensure(report == m4b().expected_param_counts4b(variant), f'{variant}: parameter report differs from the formula')


def guard_cd_checkpoint_immutable_bucket_ids():
    """The bucket-id buffers are part of the state dict and immutable on load: a changed id, a wrong dtype or a missing buffer is refused; only the variant's own buffers exist."""
    for variant, own, other in ((C, ('xb', 'fb'), ('rb', 'lb')), (D, ('rb', 'lb'), ('xb', 'fb'))):
        model = build_cd(variant); state = model.state_dict()
        ensure(all(k in state for k in own) and not any(k in state for k in other), f'{variant}: state dict must hold exactly its own bucket-id buffers')
        build_cd(variant).load_state_dict(state)
        for name in own:
            bad = dict(state); changed = state[name].clone(); changed[7] = changed[7] % 3 + 1; bad[name] = changed
            try:
                build_cd(variant).load_state_dict(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{variant}: a changed {name} was accepted on load')
            bad = dict(state); bad[name] = state[name].to(torch.int32)
            try:
                build_cd(variant).load_state_dict(bad)
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError(f'{variant}: a {name} with the wrong dtype was accepted on load')


# ---------------------------------------------------------------- tests

class ScaffoldTests(unittest.TestCase):
    def test_constants(self):
        m = m4b()
        self.assertEqual(m.FEATURES4B, FEATURES4B_LITERAL)
        self.assertEqual(m.IMPLEMENTED[:3], (CONTROL, A, E))

    def test_factory_rejects_bad_requests(self):
        m = m4b()
        for variant in ('nope', 'f3_pos_only', 'f3_user_activity'):
            with self.assertRaises(ValueError):                               # not a FEAT-04b variant
                m.make_feat04b_model(variant, tm.CONTENT, tm.SUPPORTED, 50, 42)
        for variant in ('f4_item_popularity', 'f4_item_quality'):
            with self.assertRaises(ValueError):                               # C/D need the stat buckets (stat_buckets=None is refused)
                m.make_feat04b_model(variant, tm.CONTENT, tm.SUPPORTED, 50, 42)
        with self.assertRaises(ValueError):
            m.make_feat04b_model(A, tm.CONTENT, tm.SUPPORTED, 33, 42)
        with self.assertRaises(ValueError):
            m.make_feat04b_model(A, tm.CONTENT, tm.SUPPORTED, 50, 42, stat_buckets=object())        # only C/D take stat buckets
        with self.assertRaises(ValueError):
            m.make_feat04b_model(E, tm.CONTENT, tm.SUPPORTED, 50, 42, stat_buckets=object())

    def test_class_bypasses_the_feat03_constructor(self):
        from vkgr import retrieval_feat03 as f3
        model = build4b(A)
        self.assertIsInstance(model, f3.Feat03IDMean)
        self.assertEqual(model.variant, A); self.assertEqual(model.feature, 'f3_f4_recency')
        with self.assertRaises(ValueError):
            m4b().Feat04bIDMean('nope', tm.CONTENT, tm.SUPPORTED, 50)

    def test_unimplemented_variant_users_raise(self):
        model = build4b(CONTROL)
        object.__setattr__(model, 'variant', 'not_a_variant')
        histories, side = tm.model_inputs()
        with self.assertRaises(ValueError):
            model.users(histories, side)

    def test_pool_mask_is_an_identity_seam(self):
        model = build4b(A); histories, side = tm.model_inputs(); valid = histories != 0
        self.assertTrue(torch.equal(model._pool_mask(histories, valid), valid))

    def test_a_users_go_through_the_pool_mask_seam(self):
        from unittest import mock
        model = t3.set_logits(build4b(A)); histories, side = tm.model_inputs(); valid = histories != 0
        base = model.users(histories, side)
        with mock.patch.object(type(model), '_pool_mask', lambda self, h, v: v & (h % 2 == 0)):
            self.assertFalse(torch.equal(model.users(histories, side), base))

    def test_event_vectors4_equals_the_feat03_event_vectors(self):
        from vkgr import retrieval_feat03 as f3
        model = build4b(CONTROL); histories, side = tm.model_inputs(); valid = histories != 0
        self.assertTrue(torch.equal(m4b().event_vectors4(model, histories, side, valid), f3.event_vectors(model, histories, side, valid)))

    def test_bucket_spec_digest(self):
        guard_s_spec_hash_per_variant()


class SharedGuardTests(unittest.TestCase):
    def test_default_off(self):
        guard_s_default_off()

    def test_anchor_tables_bitwise(self):
        guard_s_anchor_tables_bitwise()

    def test_module_attribute_calls(self):
        guard_s_module_attribute_calls()

    def test_scan_bites_on_a_direct_import(self):
        for bad in ('from .retrieval_feat03 import pooled_weights\n', '    from .retrieval_feat02 import x', 'from .retrieval_feat03_run import y',
                    'from .retrieval_feat02_run import z', 'from vkgr.retrieval_feat03 import q'):
            with self.assertRaises(AssertionError):
                guard_s_module_attr_scan(bad)
        guard_s_module_attr_scan('from . import retrieval_feat03 as f3\nf3.pooled_weights\n')


class ATests(unittest.TestCase):
    def test_start_equals_anchor(self):
        guard_a_start_equals_anchor()

    def test_formula(self):
        guard_a_formula()

    def test_pad_width_invariance(self):
        guard_a_pad_width_invariance()

    def test_logits_receive_gradient(self):
        guard_a_logits_receive_gradient()

    def test_param_count_13(self):
        guard_a_param_count_13()

    def test_pooled_weights_pad_cannot_overflow(self):
        guard_a_pooled_weights_pad_cannot_overflow()


class ETests(unittest.TestCase):
    def test_start_equals_a(self):
        guard_e_start_equals_a()

    def test_formula(self):
        guard_e_formula()

    def test_bit_order(self):
        guard_e_bit_order()

    def test_logit_path_only(self):
        guard_e_logit_path_only()

    def test_beta_receive_gradient(self):
        guard_e_beta_receive_gradient()

    def test_param_count_19(self):
        guard_e_param_count_19()

    def test_recency_logits_receive_gradient(self):
        guard_e_recency_logits_receive_gradient()

    def test_pad_cannot_overflow_with_extreme_beta(self):
        guard_e_pad_cannot_overflow_with_extreme_beta()

    def test_beta_init_draws_no_random_number(self):
        guard_e_beta_init_draws_no_random_number()

    def test_e_pad_width_invariance_and_logit_shift(self):
        model = e_model(); histories, side = tm.model_inputs()
        base = model.users(histories, side)
        for extra in (1, 6):
            self.assertTrue(torch.equal(model.users(*t3.pad_columns(histories, side, extra)), base))

    def test_e_users_go_through_the_pool_mask_seam(self):
        from unittest import mock
        model = e_model(); histories, side = tm.model_inputs(); valid = histories != 0
        base = model.users(histories, side)
        with mock.patch.object(type(model), '_pool_mask', lambda self, h, v: v & (h % 2 == 0)):
            self.assertFalse(torch.equal(model.users(histories, side), base))


class CDTests(unittest.TestCase):
    def test_history_path_unchanged(self):
        guard_cd_history_path_unchanged()

    def test_candidate_formula(self):
        guard_cd_candidate_formula()

    def test_unsupported_zero(self):
        guard_cd_unsupported_zero()

    def test_tables_receive_gradient(self):
        guard_cd_tables_receive_gradient()

    def test_init_and_draw_order(self):
        guard_cd_init_and_draw_order()

    def test_param_counts(self):
        guard_cd_param_counts()

    def test_checkpoint_immutable_bucket_ids(self):
        guard_cd_checkpoint_immutable_bucket_ids()

    def test_constructor_validates_the_bucket_ids(self):
        from vkgr import retrieval_feat04b as m
        good = cd_buckets()
        for label, bad in (('pad row non-zero', {**good, 'xb': torch.cat([torch.ones(1, dtype=torch.long), good['xb'][1:]])}),
                           ('wrong dtype', {**good, 'xb': good['xb'].to(torch.int32)}), ('wrong length', {**good, 'xb': good['xb'][:-1]}),
                           ('id past the table', {**good, 'xb': good['xb'] + 100}), ('negative id', {**good, 'fb': good['fb'] - 5}),
                           ('missing key', {k: v for k, v in good.items() if k != 'fb'})):
            with self.assertRaises(ValueError, msg=label):
                m.make_feat04b_model(C, tm.CONTENT, tm.SUPPORTED, 50, 42, stat_buckets=bad)
        with self.assertRaises(ValueError):
            m.make_feat04b_model(C, tm.CONTENT, tm.SUPPORTED, 50, 42, stat_buckets=list(good.values()))        # a dict is required
        with self.assertRaises(ValueError):
            m.make_feat04b_model(C, tm.CONTENT, tm.SUPPORTED, 50, 42)                                            # C/D need the buckets
        m.make_feat04b_model(D, tm.CONTENT, tm.SUPPORTED, 50, 42, stat_buckets={k: v for k, v in good.items() if k in ('rb', 'lb')})   # D needs only its own two

    def test_buffers_are_not_parameters_and_follow_the_model_device_dtype_rules(self):
        model = build_cd(C)
        self.assertFalse(any(n in dict(model.named_parameters()) for n in ('xb', 'fb')))
        self.assertTrue(all(model.state_dict()[k].dtype == torch.long for k in ('xb', 'fb')))

    def test_users_and_candidates_are_deterministic_for_a_seed(self):
        a, b = build_cd(D), build_cd(D)
        self.assertTrue(torch.equal(a.candidate_items(torch.arange(0, 301)), b.candidate_items(torch.arange(0, 301))))


if __name__ == '__main__':
    unittest.main()
