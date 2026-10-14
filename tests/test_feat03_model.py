"""FEAT-03 (P line) model tests, FEAT-002 part: scaffold, f4_control equivalence fixture, Q1a (position only), Q1b (week gap only).

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); the unit tests call them
and tests/test_feat03_mutations.py re-runs them under deliberately wrong implementations.
Bucket references are independent literals (tests/test_feat02_model.py transcribes them from the train-only FEAT-02 audit).
"""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat02_model as tm  # noqa: E402

ensure = tm.ensure
CONTROL, Q1A, Q1B, Q2, Q3 = 'f4_control', 'f3_pos_only', 'f3_gap_only', 'f4_recency_pool', 'f3_user_activity'
FEATURES3_LITERAL = (CONTROL, Q1A, Q1B, Q2, Q3)
IMPLEMENTED = (CONTROL, Q1A, Q1B)
# side columns a variant may read: [feedback_mask, timespent, duration, week, place, platform]
USED_COLUMNS3 = {CONTROL: {1, 2, 3}, Q1A: {1, 2}, Q1B: {1, 2, 3}, Q2: {1, 2, 3}, Q3: {1, 2, 3}}
SRC = Path(__file__).resolve().parents[1] / 'src' / 'vkgr' / 'retrieval_feat03.py'
Q3_JSON = Path(__file__).resolve().parents[1] / 'configs' / 'feat03' / 'q3-bucket-constants.json'      # audit stdout, copied verbatim; never reads .agents/
# Independent literals of the train-only Q3 audit (final edges, no merge happened; FEAT-001 receipt): n buckets [1-3] [4-7] [8-15] [16-31] [32-49] [50],
# distinct-week buckets [1] [2] [3-4] [5-8] [9-16] [17-25]. Bucket ids are 0-based (no PAD row).
AUDIT_LEN_EDGES = (4, 8, 16, 32, 50)
AUDIT_ACT_EDGES = (2, 3, 5, 9, 17)


def build3(variant, seed=42):
    from vkgr.retrieval_feat03 import make_feat03_model
    return make_feat03_model(variant, tm.CONTENT, tm.SUPPORTED, 50, seed)


def reference_ids(side, valid):
    """Independent (position id, week-gap id) per slot: PAD 0; position counted back from the last valid event, gap = last week - week."""
    pos = torch.zeros_like(valid, dtype=torch.long); gap = torch.zeros_like(valid, dtype=torch.long)
    for j in range(len(valid)):
        count = int(valid[j].sum()); weeks = side[j, :count, 3].tolist()
        for k in range(count):
            pos[j, k] = tm.ref_id(count - 1 - k, tm.AUDIT_POSITION_EDGES)
            gap[j, k] = tm.ref_id(weeks[-1] - weeks[k], tm.AUDIT_WEEKGAP_EDGES)
    return pos, gap


# ---------------------------------------------------------------- guards (reused by the mutation tests)

def guard_f4_control_equals_f4_bitwise():
    """AC2 (model level): f4_control is the FEAT-02 F4 model, bit for bit (state, items, users)."""
    control, f4 = build3(CONTROL), tm.build('f3_f4_recency')
    a, b = control.state_dict(), f4.state_dict()
    ensure(list(a) == list(b), 'f4_control state_dict keys differ from F4')
    for key in a:
        ensure(torch.equal(a[key], b[key]), f'f4_control state {key} differs from F4')
    ensure(torch.equal(control.items(torch.arange(0, 301)), f4.items(torch.arange(0, 301))), 'items differ')
    histories, side = tm.model_inputs()
    for h in (histories, tm.all_supported(histories), tm.all_unsupported(histories)):
        ensure(torch.equal(control.users(h, side), f4.users(h, side)), 'f4_control users differ from F4')


def guard_q1a_formula():
    """Q1a user vector = F3 user vector + mean_k P[position bucket of event k]."""
    model, f3 = build3(Q1A), tm.build('f3'); histories, side = tm.model_inputs(); valid = histories != 0
    pos_ids, _ = reference_ids(side, valid)
    ensure(bool((pos_ids.amax(1) >= 3).any()), 'fixture must span several position buckets')
    expected = f3.users(histories, side) + model.position_table(pos_ids).sum(1) / valid.sum(1, keepdim=True)
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-6, rtol=0), 'Q1a is not F3 + mean P[pb]')


def guard_q1b_formula():
    """Q1b user vector = F3 user vector + mean_k G[week-gap bucket of event k]."""
    model, f3 = build3(Q1B), tm.build('f3'); histories, side = tm.model_inputs(); valid = histories != 0
    _, gap_ids = reference_ids(side, valid)
    ensure(bool((gap_ids.amax(1) >= 3).any()), 'fixture must span several week-gap buckets')
    expected = f3.users(histories, side) + model.weekgap_table(gap_ids).sum(1) / valid.sum(1, keepdim=True)
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-6, rtol=0), 'Q1b is not F3 + mean G[gb]')


def guard_q1a_ignores_week_column():
    """Changing the week column (still 0..24 and non-decreasing) cannot move Q1a, although it moves the week-gap buckets."""
    model = build3(Q1A); histories, side = tm.model_inputs(); valid = histories != 0
    n = valid.sum(1); slot = torch.arange(histories.shape[1]).unsqueeze(0)
    same_week = side.clone(); same_week[..., 3] = torch.where(valid, torch.full_like(histories, 7), torch.zeros_like(histories))
    spread = side.clone(); spread[..., 3] = torch.where(valid, 24 * slot // (n - 1).clamp_min(1).unsqueeze(1), torch.zeros_like(histories))
    gaps = [reference_ids(s, valid)[1] for s in (side, same_week, spread)]
    ensure(not torch.equal(gaps[0], gaps[1]) and not torch.equal(gaps[1], gaps[2]) and not torch.equal(gaps[0], gaps[2]), 'fixture: week-gap buckets must differ')
    ensure(int(gaps[2].amax()) >= 6 and int(gaps[1].amax()) == 1, 'fixture needs a 9+ week span and an all-same-week control')
    base = model.users(histories, side)
    ensure(torch.equal(model.users(histories, same_week), base), 'Q1a changed when every event has the same week')
    ensure(torch.equal(model.users(histories, spread), base), 'Q1a changed when the weeks span 24 weeks')


def guard_zero_table_equals_f3(variant, table):
    model, f3 = build3(variant), tm.build('f3'); histories, side = tm.model_inputs()
    with torch.no_grad():
        getattr(model, table).weight.zero_()
    ensure(torch.equal(model.users(histories, side), f3.users(histories, side)), f'{variant} with {table} zeroed must equal F3 exactly')


def guard_q1a_with_zero_position_table_equals_f3():
    guard_zero_table_equals_f3(Q1A, 'position_table')


def guard_q1b_with_zero_gap_table_equals_f3():
    guard_zero_table_equals_f3(Q1B, 'weekgap_table')


def guard_q1_tables_equal_f4_tables_bitwise():
    """Same seed: Q1a's P and Q1b's G are the very tables F4 draws (the unused table is still drawn, then dropped)."""
    f4 = tm.build('f3_f4_recency')
    for variant, table, absent in ((Q1A, 'position_table', 'weekgap_table'), (Q1B, 'weekgap_table', 'position_table')):
        model = build3(variant)
        ensure(torch.equal(getattr(model, table).weight, getattr(f4, table).weight), f'{variant}: {table} differs from the F4 table at the same seed')
        ensure(not hasattr(model, absent), f'{variant}: {absent} must be dropped')
        for name in ('embedding', 'timespent_bucket', 'ratio_bucket'):
            ensure(torch.equal(getattr(model, name).weight, getattr(f4, name).weight), f'{variant}: {name} differs from F4')


def guard_new_tables_receive_gradient(variant):
    if variant == Q3:
        return guard_q3_new_tables_receive_gradient()
    ensure(variant in (Q1A, Q1B), 'unknown variant')
    model = build3(variant); histories, side = tm.model_inputs(); valid = histories != 0
    model.users(histories, side).sum().backward()
    table = model.position_table if variant == Q1A else model.weekgap_table
    p = table.weight
    ensure(p.grad is not None, f'{variant}: the new table received no gradient (output detached?)')
    pos_ids, gap_ids = reference_ids(side, valid)
    used = sorted(set((pos_ids if variant == Q1A else gap_ids)[valid].tolist()))
    ensure(len(used) >= 3, 'fixture must use several rows')
    for row in used:
        ensure(bool(p.grad[row].abs().sum() > 0), f'{variant}: used row {row} got no gradient')
    ensure(not bool(p.grad[0].any()), f'{variant}: PAD row must get no gradient')


def guard_q1a_new_table_receives_gradient():
    guard_new_tables_receive_gradient(Q1A)


def guard_q1b_new_table_receives_gradient():
    guard_new_tables_receive_gradient(Q1B)


def guard_q1b_changing_history_week_changes_feature():
    model = build3(Q1B)
    histories = torch.tensor([[11, 12, 13, 14, 15]]); side = torch.zeros(1, 5, 6, dtype=torch.long)
    side[0, :, 0] = 1; side[0, :, 1] = 20; side[0, :, 2] = 30; side[0, :, 3] = torch.tensor([0, 4, 8, 12, 16])
    base = model.users(histories, side)
    for slot, week in ((1, 8), (3, 15), (4, 24)):
        changed = side.clone(); changed[0, slot, 3] = week
        ensure(not torch.equal(model.users(histories, changed), base), f'changing the week of history event {slot} left Q1b unchanged')


# ---------------------------------------------------------------- Q2 helpers and guards

def pad_columns(histories, side, extra):
    """Append ``extra`` PAD columns (item 0, all-zero side)."""
    return torch.nn.functional.pad(histories, (0, extra)), torch.nn.functional.pad(side, (0, 0, 0, extra))


def set_logits(model, seed=3, scale=0.8):
    """Distinct non-zero logit values (all 13 different), so a mis-indexed or mis-pooled implementation is visible."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.recency_logit_pos.copy_(torch.randn(7, generator=g) * scale)
        model.recency_logit_gap.copy_(torch.randn(6, generator=g) * scale)
    return model


def reference_event_vectors(model, histories, side, valid):
    """Independent F4 per-event vector item + T + R + P[pos id] + G[gap id] built from the transcribed bucket references."""
    from vkgr.retrieval_features import ratio_bucket, timespent_bucket
    pos_ids, gap_ids = reference_ids(side, valid)
    t = model.timespent_bucket(timespent_bucket(side[..., 1], valid)); r = model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
    return model.items(histories) + t + r + model.position_table(pos_ids) + model.weekgap_table(gap_ids), pos_ids, gap_ids


def guard_q2_zero_logits_equal_f4_bitwise():
    """AC4: with both logit vectors at 0 the Q2 user vector is the F4 user vector bit for bit (PAD columns, supported/unsupported mix)."""
    model, f4 = build3(Q2), tm.build('f3_f4_recency')
    histories, side = tm.model_inputs(); valid = histories != 0
    ensure(bool((~valid).any()), 'fixture needs PAD columns in a mixed-length batch')
    ensure(bool((valid & ~model.supported[histories]).any()) and bool((valid & model.supported[histories]).any()), 'fixture needs supported and unsupported events')
    wide = pad_columns(histories, side, 7)
    for h, s in ((histories, side), (tm.all_supported(histories), side), (tm.all_unsupported(histories), side), wide):
        ensure(torch.equal(model.users(h, s), f4.users(h, s)), 'Q2 with zero logits differs from F4')


def guard_q2_formula():
    """Q2 user = sum_k w_k x_k / sum_k w_k with w_k = exp(l_k - max l), l_k = u[pb_k - 1] + v[gb_k - 1], x_k the F4 event vector."""
    model = set_logits(build3(Q2)); histories, side = tm.model_inputs(); valid = histories != 0
    x, pos_ids, gap_ids = reference_event_vectors(model, histories, side, valid)
    ensure(int(pos_ids.amax()) >= 3 and int((pos_ids == 1).sum()) >= 2, 'fixture must span several position buckets with n >= 2')
    ensure(int(gap_ids.amax()) == 6 and len(set(gap_ids[valid].tolist())) >= 4, 'fixture must span >= 9 weeks and several week-gap buckets')
    ensure(len(set(model.recency_logit_pos.tolist()) | set(model.recency_logit_gap.tolist())) == 13, 'all logit values must be distinct')
    ell = model.recency_logit_pos[pos_ids - 1] + model.recency_logit_gap[gap_ids - 1]     # PAD (id 0) slots are never read below
    expected = torch.zeros(len(histories), 64)
    for j in range(len(histories)):
        n = int(valid[j].sum()); w = torch.exp(ell[j, :n] - ell[j, :n].max())
        expected[j] = (x[j, :n] * w.unsqueeze(-1)).sum(0) / w.sum()
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-6, rtol=0), 'Q2 is not the normalised weighted mean')


def guard_q2_logit_shift_invariance():
    """Adding one constant to every logit leaves the normalised weighted mean unchanged (up to rounding)."""
    model = set_logits(build3(Q2)); histories, side = tm.model_inputs()
    base = model.users(histories, side)
    with torch.no_grad():
        model.recency_logit_pos += 3.7
    ensure(torch.allclose(model.users(histories, side), base, atol=1e-6, rtol=0), 'Q2 changed under a constant shift of the logits')


def guard_q2_pad_width_invariance():
    """Appending PAD columns changes nothing, bit for bit, also with non-zero logits."""
    model = set_logits(build3(Q2)); histories, side = tm.model_inputs()
    base = model.users(histories, side)
    for extra in (1, 5, 20):
        ensure(torch.equal(model.users(*pad_columns(histories, side, extra)), base), f'Q2 changed when {extra} PAD columns were appended')


def guard_q2_logits_receive_gradient():
    """Both logit vectors receive finite, non-zero gradients (on every used row) at the zero start, and one SGD step moves them."""
    model = build3(Q2); histories, side = tm.model_inputs(); valid = histories != 0
    _, pos_ids, gap_ids = reference_event_vectors(model, histories, side, valid)
    projection = torch.randn(64, generator=torch.Generator().manual_seed(9))
    (model.users(histories, side) @ projection).sum().backward()
    for name, ids in (('recency_logit_pos', pos_ids), ('recency_logit_gap', gap_ids)):
        p = getattr(model, name)
        ensure(p.grad is not None, f'{name}: no gradient (logits detached?)')
        ensure(bool(torch.isfinite(p.grad).all()), f'{name}: non-finite gradient')
        used = sorted(set((ids[valid] - 1).tolist()))
        ensure(len(used) >= 3, 'fixture must use several rows')
        for row in used:
            ensure(float(p.grad[row].abs()) > 0, f'{name}: used row {row} got no gradient')
        before = p.detach().clone()
        with torch.no_grad():
            p -= 0.1 * p.grad
        ensure(not torch.equal(p.detach(), before), f'{name}: an SGD step did not move the logits')


def guard_q2_init_is_exact_zero():
    """13 scalars, nn.Parameter, exactly 0 (not N(0, 0.02)); they are the only parameters beyond F4."""
    model, f4 = build3(Q2), tm.build('f3_f4_recency')
    for name, size in (('recency_logit_pos', 7), ('recency_logit_gap', 6)):
        p = getattr(model, name)
        ensure(isinstance(p, torch.nn.Parameter) and p.requires_grad and p.dtype == torch.float32 and tuple(p.shape) == (size,), f'{name}: wrong parameter')
        ensure(bool((p == 0).all()), f'{name}: init must be exactly 0')
    extra = set(dict(model.named_parameters())) - set(dict(f4.named_parameters()))
    ensure(extra == {'recency_logit_pos', 'recency_logit_gap'}, f'unexpected extra parameters {sorted(extra)}')
    ensure(sum(p.numel() for n, p in model.named_parameters() if n in extra and p.requires_grad) == 13, 'exactly 13 trainable scalars beyond F4')
    ensure(set(dict(f4.named_parameters())) <= set(dict(model.named_parameters())), 'Q2 lost an F4 parameter')
    histories, side = tm.model_inputs()
    ensure(torch.equal(model.users(histories, side), f4.users(histories, side)), 'Q2 at its init differs from F4')


def guard_q2_recent_event_gets_bucket1_logit():
    """u[0] (position bucket 1 = the most recent event) large: the user vector is the last event's vector, so the logit is wired to the recency end."""
    model = build3(Q2); histories, side = tm.model_inputs(); valid = histories != 0
    with torch.no_grad():
        model.recency_logit_pos[0] = 20.0
    x = reference_event_vectors(model, histories, side, valid)[0].detach()
    n = valid.sum(1); last = x[torch.arange(len(histories)), n - 1]
    ensure(bool((n >= 3).any()) and float((x[:, 0] - last).abs().amax()) > 1e-3, 'fixture needs rows whose first and last event differ')
    ensure(torch.allclose(model.users(histories, side), last, atol=1e-5, rtol=0), 'with u[0] large the user vector must be the most recent event vector')


def guard_q2_pooled_weights_pad_cannot_overflow():
    """pooled_weights is a double where: even when a PAD logit is 1e4 the weights are finite, PAD weight is exactly 0 and the backward pass is finite."""
    from vkgr.retrieval_feat03 import pooled_weights
    valid = torch.tensor([[True, True, True, False, False], [True, True, False, False, False], [True, True, True, True, True]])
    logits = torch.tensor([[0.1, -0.2, 0.3, 1e4, 1e4], [0.0, 0.2, 1e4, 1e4, 1e4], [0.5, -0.5, 0.0, 0.25, -0.25]], requires_grad=True)
    w = pooled_weights(logits, valid)
    ensure(bool(torch.isfinite(w).all()), 'pooled_weights returned a non-finite weight')
    ensure(bool((w[~valid] == 0).all()), 'PAD weight must be exactly 0')
    expected = torch.exp(logits.detach() - logits.detach().masked_fill(~valid, float('-inf')).amax(1, keepdim=True))
    ensure(torch.allclose(w.detach()[valid], expected[valid], atol=1e-7, rtol=0), 'valid weights must be exp(l - max valid l)')
    w.sum().backward()
    ensure(bool(torch.isfinite(logits.grad).all()), 'backward through pooled_weights is not finite with a 1e4 PAD logit')


# ---------------------------------------------------------------- Q3 helpers and guards

def q3_inputs(seed=1, batch=24, width=50):
    """Wide mixed-length batch: n spans every length bucket (lengths[0] = width = 50), distinct weeks span several activity buckets."""
    return tm.model_inputs(seed, batch=batch, width=width)


def reference_q3_ids(side, valid):
    """Independent (length bucket id, activity bucket id) per row, 0-based: n = valid slots, a = number of DISTINCT weeks among them."""
    lb = torch.zeros(len(valid), dtype=torch.long); ab = torch.zeros(len(valid), dtype=torch.long)
    for j in range(len(valid)):
        n = int(valid[j].sum()); a = len(set(side[j, :n, 3].tolist()))
        lb[j] = sum(1 for edge in AUDIT_LEN_EDGES if n >= edge); ab[j] = sum(1 for edge in AUDIT_ACT_EDGES if a >= edge)
    return lb, ab


def guard_q3_formula():
    """AC5: Q3 users = F3 users + Len[lb(n)] + Act[ab(a)] against an independent implementation (n, a counted by plain Python)."""
    model, f3 = build3(Q3), tm.build('f3'); histories, side = q3_inputs(); valid = histories != 0
    lb, ab = reference_q3_ids(side, valid)
    ensure(len(set(lb.tolist())) >= 4 and len(set(ab.tolist())) >= 4, 'fixture must span several length and activity buckets')
    expected = f3.users(histories, side) + model.len_table(lb) + model.act_table(ab)
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-6, rtol=0), 'Q3 is not F3 + Len[lb(n)] + Act[ab(a)]')


def guard_q3_pad_width_invariance():
    """Appending PAD columns changes nothing, bit for bit; the widths land in different length buckets (so n = tensor width is visible)."""
    model = build3(Q3); histories, side = tm.model_inputs(); base = model.users(histories, side)
    widths = [histories.shape[1]] + [histories.shape[1] + extra for extra in (1, 5, 20)]
    ensure(len({sum(1 for edge in AUDIT_LEN_EDGES if w >= edge) for w in widths}) >= 3, 'fixture: tensor widths must fall in different length buckets')
    for extra in (1, 5, 20):
        ensure(torch.equal(model.users(*pad_columns(histories, side, extra)), base), f'Q3 changed when {extra} PAD columns were appended')


def guard_q3_distinct_weeks_not_span():
    """The number of distinct weeks, not last - first + 1: weeks [3, 3, 9] -> 2 (a span would give 7)."""
    from vkgr import retrieval_feat03 as m
    week = torch.tensor([[3, 3, 9, 0], [7, 7, 7, 0], [1, 2, 3, 4], [0, 24, 24, 0]]); valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    out = m.distinct_weeks(week, valid)
    ensure(out.dtype == torch.long and tuple(out.shape) == (4,), 'distinct_weeks must be a (B,) long tensor')
    ensure(out.tolist() == [2, 1, 4, 2], f'distinct_weeks gave {out.tolist()}, expected [2, 1, 4, 2]')


def guard_q3_distinct_weeks_ignores_pad():
    """PAD slots (week 0) are not a week: rows whose first event week is > 0 and which carry PAD columns."""
    from vkgr import retrieval_feat03 as m
    week = torch.tensor([[5, 5, 7, 0, 0], [4, 0, 0, 0, 0], [0, 3, 3, 0, 0], [2, 3, 4, 5, 6]])
    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    out = m.distinct_weeks(week, valid)
    ensure(out.tolist() == [2, 1, 2, 5], f'distinct_weeks gave {out.tolist()}, expected [2, 1, 2, 5]')


def guard_week_shift_invariance(variant=Q3):
    """Shifting every history week by +1 (still <= 24) leaves the user vector bit-identical (no absolute calendar time channel).
    Row 0 has weeks 0..e-1 with e = ACT_EDGES[0] read at call time: its last week e-1 crosses a bucket boundary when shifted (visible to an absolute-week mutant)."""
    from vkgr import retrieval_feat03 as m
    model = build3(variant)
    if variant == Q2:
        set_logits(model)
    histories, side = q3_inputs(seed=2, batch=12, width=20); histories, side = histories.clone(), side.clone(); valid = histories != 0
    side[..., 3] = torch.where(valid, side[..., 3].clamp(max=23), side[..., 3])
    e = m.ACT_EDGES[0]
    histories[0] = 0; side[0] = 0
    histories[0, :e] = torch.arange(1, e + 1); side[0, :e, 1] = 20; side[0, :e, 2] = 30; side[0, :e, 3] = torch.arange(e)
    valid = histories != 0
    shifted = side.clone(); shifted[..., 3] = torch.where(valid, side[..., 3] + 1, side[..., 3])
    ensure(int(side[0, e - 1, 3]) == e - 1 and int(shifted[..., 3].max()) <= 24, 'fixture: row 0 must end at week e-1 and the shift must stay <= 24')
    ensure(torch.equal(model.users(histories, shifted), model.users(histories, side)), f'{variant} changed under a +1 shift of every history week')


def guard_q3_constants_match_audit():
    """AC: LEN_EDGES / ACT_EDGES equal the final edges of the audit stdout stored in the repo (configs/feat03/q3-bucket-constants.json), BOTH tuples."""
    from vkgr import retrieval_feat03 as m
    text = Q3_JSON.read_text(); data = json.loads(text)
    ensure(data['schema'] == 'feat03-audit-v1' and data['train_only'] is True, 'audit JSON is not a train-only feat03-audit-v1 result')
    ensure('/' not in text and '\\' not in text, 'audit JSON must not contain a path')
    ensure(tuple(data['len']['final_edges']) == tuple(m.LEN_EDGES), f'LEN_EDGES {tuple(m.LEN_EDGES)} differs from the audit {tuple(data["len"]["final_edges"])}')
    ensure(tuple(data['act']['final_edges']) == tuple(m.ACT_EDGES), f'ACT_EDGES {tuple(m.ACT_EDGES)} differs from the audit {tuple(data["act"]["final_edges"])}')


def guard_q3_bucket_boundaries():
    """Both bucket functions against the independent literals at every integer around every edge (all n in 1..60, all a in 1..30)."""
    from vkgr import retrieval_feat03 as m
    for function, edges, top in ((m.length_bucket, AUDIT_LEN_EDGES, 60), (m.activity_bucket, AUDIT_ACT_EDGES, 30)):
        values = torch.arange(1, top + 1)
        out = function(values)
        ensure(out.dtype == torch.long and tuple(out.shape) == tuple(values.shape), f'{function.__name__} must return a long tensor of the input shape')
        expected = torch.tensor([sum(1 for edge in edges if int(v) >= edge) for v in values])
        ensure(torch.equal(out, expected), f'{function.__name__} boundaries differ from the audit edges {edges}')


def guard_q3_new_tables_receive_gradient():
    """M33[q3]: both user-level tables receive gradient on every used row (output not detached)."""
    model = build3(Q3); histories, side = q3_inputs(); valid = histories != 0
    model.users(histories, side).sum().backward()
    lb, ab = reference_q3_ids(side, valid)
    for name, ids in (('len_table', lb), ('act_table', ab)):
        p = getattr(model, name).weight
        ensure(p.grad is not None, f'Q3 {name}: no gradient (output detached?)')
        used = sorted(set(ids.tolist()))
        ensure(len(used) >= 3, 'fixture must use several rows')
        for row in used:
            ensure(bool(p.grad[row].abs().sum() > 0), f'Q3 {name}: used row {row} got no gradient')


def guard_q3_tables_init(seed=42):
    """M34: no PAD row; Len/Act equal a reference that REPLAYS the whole draw sequence (item, T, R, P, G from make_feat02_model, then Embedding(len)+N(0,0.02),
    then Embedding(act)+N(0,0.02)); not all zero (the _small_init trap), std near 0.02, T and R bit-equal to the same-seed F3."""
    from vkgr.retrieval_feat02 import make_feat02_model
    from vkgr.training import seed_all
    model, f3 = build3(Q3, seed), tm.build('f3', seed)
    seed_all(seed)
    make_feat02_model('f3_f4_recency', tm.CONTENT, tm.SUPPORTED, 50)                              # draws item, T, R, P, G in the F4 order
    for name, rows in (('len_table', len(AUDIT_LEN_EDGES) + 1), ('act_table', len(AUDIT_ACT_EDGES) + 1)):
        reference = torch.nn.Embedding(rows, 64)
        with torch.no_grad():
            reference.weight.normal_(0.0, 0.02)
        table = getattr(model, name)
        ensure(isinstance(table, torch.nn.Embedding) and table.padding_idx is None and tuple(table.weight.shape) == (rows, 64), f'{name}: wrong shape or has a PAD row')
        ensure(torch.equal(table.weight, reference.weight), f'{name}: differs from the replayed reference generator')
        ensure(bool((table.weight != 0).any()), f'{name}: all zero')
        ensure(0.017 <= float(table.weight.std()) <= 0.023, f'{name}: std {float(table.weight.std())} not near 0.02')
    ensure(torch.equal(model.timespent_bucket.weight, f3.timespent_bucket.weight) and torch.equal(model.ratio_bucket.weight, f3.ratio_bucket.weight), 'T/R differ from the same-seed F3')


# ---------------------------------------------------------------- tests

class ScaffoldTests(unittest.TestCase):
    def test_features3_literal_and_class_attributes(self):
        from vkgr import retrieval_feat03 as m
        self.assertEqual(m.FEATURES3, FEATURES3_LITERAL)
        for variant in FEATURES3_LITERAL:
            model = build3(variant)
            self.assertEqual(model.variant, variant)
            self.assertEqual(model.feature, 'f3_f4_recency', 'the base feature string is shared by every variant (K3)')

    def test_unregistered_variant_and_length_rejected(self):
        from vkgr.retrieval_feat03 import make_feat03_model
        with self.assertRaises(ValueError):
            make_feat03_model('f4_recency_pool_typo', tm.CONTENT, tm.SUPPORTED, 50, 42)
        with self.assertRaises(ValueError):
            make_feat03_model(CONTROL, tm.CONTENT, tm.SUPPORTED, 30, 42)

    def test_f4_control_equals_f4_bitwise(self):
        guard_f4_control_equals_f4_bitwise()

    def test_same_seed_same_model_different_seed_different(self):
        a, b, c = build3(Q1A, 42), build3(Q1A, 42), build3(Q1A, 43)
        self.assertTrue(torch.equal(a.position_table.weight, b.position_table.weight))
        self.assertFalse(torch.equal(a.position_table.weight, c.position_table.weight))

    def test_users_still_validates_inputs(self):
        model = build3(CONTROL); histories, side = tm.model_inputs()
        bad = side.clone(); bad[0, 0, 3] = 25
        with self.assertRaises(ValueError):
            model.users(histories, bad)
        with self.assertRaises(ValueError):
            model.users(histories, None)

    def test_k2_no_from_import_of_feat02_names(self):
        text = SRC.read_text()
        self.assertNotIn('from .retrieval_feat02 import', text)
        self.assertIn('from . import retrieval_feat02 as f2', text)

    def test_unused_side_columns_do_not_affect_variants_and_used_ones_do(self):
        histories, side = tm.model_inputs(); valid = histories != 0
        g = torch.Generator().manual_seed(5)
        for variant in IMPLEMENTED:
            model = build3(variant); base = model.users(histories, side)
            for column in set(range(6)) - USED_COLUMNS3[variant]:
                other = side.clone(); high = 64 if column == 0 else 25 if column == 3 else 256
                values = torch.randint(0, high, valid.shape, generator=g)
                if column == 3:
                    values = torch.sort(torch.where(valid, values, torch.full_like(values, 99)), 1).values     # non-decreasing on the valid prefix
                other[..., column] = torch.where(valid, values, torch.zeros_like(values))
                self.assertTrue(torch.equal(model.users(histories, other), base), (variant, column))
            perturb = {1: lambda c: 256 - c, 2: lambda c: 256 - c, 3: lambda c: torch.zeros_like(c)}
            for column in USED_COLUMNS3[variant]:
                other = side.clone(); other[..., column] = torch.where(valid, perturb[column](side[..., column]), torch.zeros_like(side[..., column]))
                self.assertFalse(torch.equal(model.users(histories, other), base), (variant, column))

    def test_param_report_matches_formula_and_literals(self):
        from vkgr.retrieval_feat03 import expected_param_counts3, parameter_report3
        literal = {CONTROL: (896 + 960, 768 + 832), Q1A: (896 + 512, 768 + 448), Q1B: (896 + 448, 768 + 384)}
        for variant in IMPLEMENTED:
            report = parameter_report3(build3(variant))
            self.assertEqual(report, expected_param_counts3(variant), variant)
            self.assertEqual((report['total'], report['trainable_effective']), literal[variant], variant)
            self.assertEqual(sum(t['total'] for t in report['by_table'].values()), report['total'])
        self.assertEqual(sorted(parameter_report3(build3(Q1A))['by_table']), ['position_table', 'ratio_bucket', 'timespent_bucket'])
        self.assertEqual(sorted(parameter_report3(build3(Q1B))['by_table']), ['ratio_bucket', 'timespent_bucket', 'weekgap_table'])
        from vkgr import retrieval_feat02 as f2
        self.assertEqual(parameter_report3(build3(CONTROL)), f2.parameter_report(tm.build('f3_f4_recency')))

    def test_parameter_report3_does_not_touch_feat02_inert_rows(self):
        from vkgr import retrieval_feat02 as f2
        from vkgr.retrieval_feat03 import expected_param_counts3, parameter_report3
        before = copy.deepcopy(f2.INERT_ROWS)
        parameter_report3(build3(Q1A)); expected_param_counts3(Q1B)
        self.assertEqual(f2.INERT_ROWS, before)

    def test_init_report_recomputed_after_drop(self):
        from torch.nn import Embedding
        for variant in IMPLEMENTED:
            model = build3(variant)
            present = {name for name, module in model.named_modules() if isinstance(module, Embedding)}
            self.assertEqual(set(model.init_report['embeddings']), present, variant)
            self.assertEqual((model.init_report['init'], model.init_report['std']), ('small', 0.02))
        self.assertFalse(any(key.startswith('weekgap_table') for key in build3(Q1A).state_dict()))
        self.assertFalse(any(key.startswith('position_table') for key in build3(Q1B).state_dict()))


class Q1Tests(unittest.TestCase):
    def test_q1a_formula(self):
        guard_q1a_formula()

    def test_q1b_formula(self):
        guard_q1b_formula()

    def test_q1a_ignores_week_column(self):
        guard_q1a_ignores_week_column()

    def test_q1a_zero_position_table_is_f3(self):
        guard_q1a_with_zero_position_table_equals_f3()

    def test_q1b_zero_gap_table_is_f3(self):
        guard_q1b_with_zero_gap_table_equals_f3()

    def test_new_tables_equal_f4_tables_bitwise(self):
        guard_q1_tables_equal_f4_tables_bitwise()

    def test_new_tables_receive_gradient(self):
        guard_q1a_new_table_receives_gradient(); guard_q1b_new_table_receives_gradient()

    def test_q1b_changing_history_week_changes_feature(self):
        guard_q1b_changing_history_week_changes_feature()

    def test_q1b_same_week_reorder_is_documented_as_rounding_only(self):
        """Documented property, NOT a mutation guard (F4 and Q1b are equal up to rounding here): swapping two same-week events moves Q1b by <= 1e-6."""
        model = build3(Q1B); histories = torch.tensor([[11, 12, 13, 14, 15, 16]]); side = torch.zeros(1, 6, 6, dtype=torch.long)
        side[0, :, 0] = 1; side[0, :, 1] = 20; side[0, :, 2] = 30; side[0, :, 3] = torch.tensor([2, 2, 2, 5, 5, 5])
        swapped_h, swapped_s = histories.clone(), side.clone()
        swapped_h[0, [3, 4]] = histories[0, [4, 3]]; swapped_s[0, [3, 4]] = side[0, [4, 3]]
        self.assertTrue(torch.allclose(model.users(histories, side), model.users(swapped_h, swapped_s), atol=1e-6, rtol=0))

    def test_q1a_and_q1b_differ_from_f3_and_from_each_other(self):
        histories, side = tm.model_inputs(); f3 = tm.build('f3').users(histories, side)
        a, b = build3(Q1A).users(histories, side), build3(Q1B).users(histories, side)
        self.assertFalse(torch.equal(a, f3)); self.assertFalse(torch.equal(b, f3)); self.assertFalse(torch.equal(a, b))


class Q2Tests(unittest.TestCase):
    def test_zero_logits_equal_f4_bitwise(self):
        guard_q2_zero_logits_equal_f4_bitwise()

    def test_formula(self):
        guard_q2_formula()

    def test_constant_shift_of_logits_is_allclose(self):
        guard_q2_logit_shift_invariance()

    def test_appending_pad_columns_is_bit_identical(self):
        guard_q2_pad_width_invariance()

    def test_logits_receive_finite_nonzero_gradient(self):
        guard_q2_logits_receive_gradient()

    def test_init_exact_zero_and_13_extra_parameters(self):
        guard_q2_init_is_exact_zero()
        for variant in (CONTROL, Q1A, Q1B):
            self.assertFalse(hasattr(build3(variant), 'recency_logit_pos'), variant)

    def test_most_recent_event_uses_position_logit_of_bucket_1(self):
        guard_q2_recent_event_gets_bucket1_logit()

    def test_pooled_weights_double_where(self):
        guard_q2_pooled_weights_pad_cannot_overflow()

    def test_tables_equal_f4_tables_at_the_same_seed(self):
        f4, model = tm.build('f3_f4_recency'), build3(Q2)
        a, b = model.state_dict(), f4.state_dict()
        for key in b:
            self.assertTrue(torch.equal(a[key], b[key]), key)
        self.assertEqual(sorted(set(a) - set(b)), ['recency_logit_gap', 'recency_logit_pos'])
        self.assertEqual(set(model.init_report['embeddings']), {name for name, m in model.named_modules() if isinstance(m, torch.nn.Embedding)})

    def test_reorder_changes_the_user_vector_only_when_logits_are_non_zero(self):
        """Swap the content (item + timespent/duration) of the first and last event of row 0, keeping week and slot positions: F4 stays (rounding), Q2 moves."""
        histories, side = tm.model_inputs(); n = int((histories[0] != 0).sum())
        self.assertGreaterEqual(n, 6)
        swapped_h, swapped_s = histories.clone(), side.clone()
        swapped_h[0, [0, n - 1]] = histories[0, [n - 1, 0]]; swapped_s[0, [0, n - 1], :3] = side[0, [n - 1, 0], :3]
        self.assertNotEqual(int(histories[0, 0]), int(histories[0, n - 1]))
        f4 = tm.build('f3_f4_recency')
        self.assertTrue(torch.allclose(f4.users(histories, side)[0], f4.users(swapped_h, swapped_s)[0], atol=1e-6, rtol=0))
        zero = build3(Q2)
        self.assertTrue(torch.allclose(zero.users(histories, side)[0], zero.users(swapped_h, swapped_s)[0], atol=1e-6, rtol=0))
        moved = set_logits(build3(Q2))
        self.assertGreater(float((moved.users(histories, side)[0] - moved.users(swapped_h, swapped_s)[0]).abs().amax()), 1e-4)

    def test_unsupported_events_still_enter_the_weighted_mean(self):
        """Non-zero logits, every history item unsupported: the user vector is still the weighted mean of the non-zero T/R/P/G event vectors."""
        model = set_logits(build3(Q2)); histories, side = tm.model_inputs(); valid = histories != 0
        unsupported = tm.all_unsupported(histories)
        self.assertEqual(int((valid & ~model.supported[unsupported]).sum()), int(valid.sum()))
        x, pos_ids, gap_ids = reference_event_vectors(model, unsupported, side, valid)
        self.assertTrue(bool((x[valid].abs().sum(1) > 0).all()), 'unsupported events keep non-zero T/R/P/G vectors')
        ell = model.recency_logit_pos[pos_ids - 1] + model.recency_logit_gap[gap_ids - 1]
        users = model.users(unsupported, side)
        for j in range(len(histories)):
            m = int(valid[j].sum()); w = torch.exp(ell[j, :m] - ell[j, :m].max())
            expected = (x[j, :m] * w.unsqueeze(-1)).sum(0) / w.sum()
            self.assertTrue(torch.allclose(users[j], expected, atol=1e-6, rtol=0), j)

    def test_pool_mask_is_the_identity_seam(self):
        model = build3(Q2); histories, side = tm.model_inputs(); valid = histories != 0
        self.assertTrue(torch.equal(model._pool_mask(histories, valid), valid))

    def test_pooled_weights_and_weighted_mean_dtype_and_values(self):
        from vkgr.retrieval_feat03 import pooled_weights, weighted_mean
        valid = torch.tensor([[True, True, False]]); w = pooled_weights(torch.tensor([[0.0, 1.0, 5.0]]), valid)
        self.assertEqual((w.dtype, tuple(w.shape)), (torch.float32, (1, 3)))
        self.assertEqual(w[0].tolist(), [float(torch.exp(torch.tensor(-1.0))), 1.0, 0.0])      # max valid logit is 1.0; the PAD logit 5.0 is ignored
        self.assertTrue(torch.allclose(weighted_mean(torch.ones(1, 3, 64), w), torch.ones(1, 64), atol=1e-7, rtol=0))

    def test_recency_logits_index_by_id_minus_one_and_pad_does_not_wrap(self):
        from vkgr.retrieval_feat03 import recency_logits
        u, v = torch.arange(10., 17.), torch.arange(20., 26.)
        out = recency_logits(u, v, torch.tensor([[1, 7, 0]]), torch.tensor([[6, 1, 0]]), torch.tensor([[True, True, False]]))
        self.assertEqual(out.tolist(), [[10. + 25., 16. + 20., 10. + 20.]])      # PAD (id 0) reads u[0] + v[0], never u[-1]

    def test_unused_side_columns_do_not_affect_q2_and_used_ones_do(self):
        histories, side = tm.model_inputs(); valid = histories != 0
        model = set_logits(build3(Q2)); base = model.users(histories, side); g = torch.Generator().manual_seed(6)
        for column in set(range(6)) - USED_COLUMNS3[Q2]:
            other = side.clone(); values = torch.randint(0, 64 if column == 0 else 256, valid.shape, generator=g)
            other[..., column] = torch.where(valid, values, torch.zeros_like(values))
            self.assertTrue(torch.equal(model.users(histories, other), base), column)
        for column, perturb in ((1, lambda c: 256 - c), (2, lambda c: 256 - c), (3, lambda c: torch.zeros_like(c))):
            other = side.clone(); other[..., column] = torch.where(valid, perturb(side[..., column]), torch.zeros_like(side[..., column]))
            self.assertFalse(torch.equal(model.users(histories, other), base), column)

    def test_param_report_counts_the_13_scalars(self):
        from vkgr.retrieval_feat03 import expected_param_counts3, parameter_report3
        report = parameter_report3(build3(Q2))
        self.assertEqual(report, expected_param_counts3(Q2))
        self.assertEqual((report['total'], report['trainable_effective']), (896 + 960 + 13, 768 + 832 + 13))
        self.assertEqual(report['by_table']['recency_logit_pos'], {'total': 7, 'trainable': 7})
        self.assertEqual(report['by_table']['recency_logit_gap'], {'total': 6, 'trainable': 6})
        for name, counts in parameter_report3(build3(CONTROL))['by_table'].items():
            self.assertEqual(report['by_table'][name], counts, name)
        self.assertEqual(sum(t['total'] for t in report['by_table'].values()), report['total'])


class Q3Tests(unittest.TestCase):
    def test_formula(self):
        guard_q3_formula()

    def test_pooling_part_is_the_f3_operation_sequence_bitwise(self):
        """AC5: the pooling part equals the F3 user vector bit for bit, and users is exactly that pooled vector plus user_level_terms."""
        from vkgr import retrieval_feat03 as m
        model, f3 = build3(Q3), tm.build('f3'); histories, side = q3_inputs(); valid = histories != 0
        pooled = m.event_vectors(model, histories, side, valid).sum(1) / valid.sum(1, keepdim=True)
        self.assertTrue(torch.equal(pooled, f3.users(histories, side)))
        self.assertTrue(torch.equal(model.users(histories, side), pooled + m.user_level_terms(model, side, valid)))

    def test_event_vectors_are_the_two_f3_steps_only(self):
        from vkgr import retrieval_feat03 as m
        from vkgr.retrieval_features import ratio_bucket, timespent_bucket
        model = build3(Q3); histories, side = tm.model_inputs(); valid = histories != 0
        expected = model.items(histories) + model.timespent_bucket(timespent_bucket(side[..., 1], valid)) + model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
        self.assertTrue(torch.equal(m.event_vectors(model, histories, side, valid), expected))

    def test_pad_columns_are_bit_identical(self):
        guard_q3_pad_width_invariance()

    def test_distinct_weeks_is_a_count_not_a_span_and_ignores_pad(self):
        guard_q3_distinct_weeks_not_span(); guard_q3_distinct_weeks_ignores_pad()

    def test_distinct_weeks_matches_plain_python_on_the_fixture(self):
        from vkgr import retrieval_feat03 as m
        histories, side = q3_inputs(); valid = histories != 0
        expected = [len(set(side[j, :int(valid[j].sum()), 3].tolist())) for j in range(len(valid))]
        self.assertEqual(m.distinct_weeks(side[..., 3], valid).tolist(), expected)
        self.assertGreaterEqual(len(set(expected)), 6)

    def test_week_shift_leaves_users_bit_identical(self):
        guard_week_shift_invariance(Q3)

    def test_bucket_functions_values(self):
        from vkgr import retrieval_feat03 as m
        n = torch.tensor([1, 3, 4, 7, 8, 15, 16, 31, 32, 49, 50]); a = torch.tensor([1, 2, 3, 4, 5, 8, 9, 16, 17, 25])
        self.assertEqual(m.length_bucket(n).tolist(), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        self.assertEqual(m.activity_bucket(a).tolist(), [0, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        guard_q3_bucket_boundaries()

    def test_edges_are_read_at_call_time(self):
        from vkgr import retrieval_feat03 as m
        with mock.patch.object(m, 'LEN_EDGES', (10,)), mock.patch.object(m, 'ACT_EDGES', (3,)):
            self.assertEqual(m.length_bucket(torch.tensor([9, 10])).tolist(), [0, 1])
            self.assertEqual(m.activity_bucket(torch.tensor([2, 3])).tolist(), [0, 1])

    def test_constants_match_repo_json(self):
        from vkgr import retrieval_feat03 as m
        guard_q3_constants_match_audit()
        data = json.loads(Q3_JSON.read_text())
        self.assertEqual((tuple(data['len']['final_edges']), tuple(data['act']['final_edges'])), (AUDIT_LEN_EDGES, AUDIT_ACT_EDGES))
        self.assertEqual((m.LEN_EDGES, m.ACT_EDGES), (AUDIT_LEN_EDGES, AUDIT_ACT_EDGES))
        self.assertEqual((data['examples'], data['merge_count']), (86785, 0))

    def test_tables_have_no_pad_row_and_initial_values_replay_the_draw_sequence(self):
        guard_q3_tables_init()
        model = build3(Q3)
        self.assertEqual((tuple(model.len_table.weight.shape), tuple(model.act_table.weight.shape)), ((6, 64), (6, 64)))
        for variant in (CONTROL, Q1A, Q1B, Q2):
            self.assertFalse(hasattr(build3(variant), 'len_table') or hasattr(build3(variant), 'act_table'), variant)

    def test_new_tables_receive_gradient(self):
        guard_q3_new_tables_receive_gradient(); guard_new_tables_receive_gradient(Q3)

    def test_only_the_user_level_tables_are_new_and_p_g_are_dropped(self):
        f3_keys, keys = set(tm.build('f3').state_dict()), set(build3(Q3).state_dict())
        self.assertEqual(keys - f3_keys, {'len_table.weight', 'act_table.weight'})
        self.assertEqual(f3_keys - keys, set())
        model = build3(Q3)
        self.assertFalse(hasattr(model, 'position_table') or hasattr(model, 'weekgap_table') or hasattr(model, 'recency_logit_pos'))
        self.assertEqual(set(model.init_report['embeddings']), {name for name, m in model.named_modules() if isinstance(m, torch.nn.Embedding)})
        self.assertIn('len_table', model.init_report['embeddings'])
        self.assertIsNone(model.init_report['embeddings']['act_table']['pad_row_zero'])

    def test_t_and_r_equal_f3_at_the_same_seed(self):
        f3, model = tm.build('f3'), build3(Q3)
        for name in ('embedding', 'timespent_bucket', 'ratio_bucket'):
            self.assertTrue(torch.equal(getattr(model, name).weight, getattr(f3, name).weight), name)

    def test_param_report_counts_768_not_640_and_leaves_feat02_inert_rows_alone(self):
        from vkgr import retrieval_feat02 as f2
        from vkgr import retrieval_feat03 as m
        before = copy.deepcopy(f2.INERT_ROWS)
        report = m.parameter_report3(build3(Q3))
        self.assertEqual(report, m.expected_param_counts3(Q3))
        self.assertEqual((report['total'], report['trainable_effective']), (896 + 768, 768 + 768))
        self.assertEqual(sorted(report['by_table']), ['act_table', 'len_table', 'ratio_bucket', 'timespent_bucket'])
        self.assertEqual(report['by_table']['len_table'], {'total': 384, 'trainable': 384})
        self.assertEqual(report['by_table']['act_table'], {'total': 384, 'trainable': 384})
        self.assertEqual(f2.INERT_ROWS, before)
        self.assertNotIn('len_table', f2.INERT_ROWS)
        self.assertEqual((m.INERT_ROWS3['len_table'], m.INERT_ROWS3['act_table']), (0, 0))

    def test_param_count_follows_the_edge_constants(self):
        from vkgr import retrieval_feat03 as m
        with mock.patch.object(m, 'LEN_EDGES', (4, 8, 16, 32, 50, 60)), mock.patch.object(m, 'ACT_EDGES', (2, 3, 5)):
            report = m.parameter_report3(build3(Q3))
            self.assertEqual(report, m.expected_param_counts3(Q3))
            self.assertEqual(report['trainable_effective'], 768 + (7 + 4) * 64)

    def test_valid_slots_must_be_left_aligned(self):
        model = build3(Q3)
        histories = torch.tensor([[5, 0, 7]]); side = torch.zeros(1, 3, 6, dtype=torch.long)
        side[0, 0] = torch.tensor([1, 20, 30, 0, 0, 0]); side[0, 2] = torch.tensor([1, 20, 30, 3, 0, 0])
        with self.assertRaisesRegex(ValueError, 'left-aligned'):
            model.users(histories, side)

    def test_single_event_history_uses_bucket_zero_of_both_tables(self):
        model = build3(Q3); f3 = tm.build('f3')
        histories = torch.tensor([[9, 0, 0]]); side = torch.zeros(1, 3, 6, dtype=torch.long); side[0, 0] = torch.tensor([1, 20, 30, 12, 0, 0])
        expected = f3.users(histories, side) + model.len_table.weight[0] + model.act_table.weight[0]
        self.assertTrue(torch.allclose(model.users(histories, side), expected, atol=1e-6, rtol=0))

    def test_users_still_validates_inputs(self):
        model = build3(Q3); histories, side = tm.model_inputs()
        bad = side.clone(); bad[0, 0, 3] = 25
        with self.assertRaisesRegex(ValueError, 'week outside train weeks'):
            model.users(histories, bad)
        with self.assertRaisesRegex(ValueError, 'side required'):
            model.users(histories, None)

    def test_unused_side_columns_do_not_affect_q3_and_used_ones_do(self):
        histories, side = q3_inputs(); valid = histories != 0
        model = build3(Q3); base = model.users(histories, side); g = torch.Generator().manual_seed(8)
        for column in set(range(6)) - USED_COLUMNS3[Q3]:
            other = side.clone(); values = torch.randint(0, 64 if column == 0 else 256, valid.shape, generator=g)
            other[..., column] = torch.where(valid, values, torch.zeros_like(values))
            self.assertTrue(torch.equal(model.users(histories, other), base), column)
        for column, perturb in ((1, lambda c: 256 - c), (2, lambda c: 256 - c), (3, lambda c: torch.zeros_like(c))):
            other = side.clone(); other[..., column] = torch.where(valid, perturb(side[..., column]), torch.zeros_like(side[..., column]))
            self.assertFalse(torch.equal(model.users(histories, other), base), column)

    def test_q3_scale_ratio_is_a_detached_float64_per_query_ratio(self):
        from vkgr import retrieval_feat03 as m
        model, f3 = build3(Q3), tm.build('f3'); histories, side = q3_inputs(); valid = histories != 0
        lb, ab = reference_q3_ids(side, valid)
        before = model.users(histories, side)
        ratio = m.q3_scale_ratio(model, histories, side)
        self.assertEqual((ratio.dtype, tuple(ratio.shape), ratio.requires_grad), (torch.float64, (len(histories),), False))
        expected = (model.len_table(lb) + model.act_table(ab)).norm(dim=1) / f3.users(histories, side).norm(dim=1)
        self.assertTrue(torch.allclose(ratio, expected.double(), atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.equal(model.users(histories, side), before))
        with self.assertRaises(ValueError):
            m.q3_scale_ratio(build3(Q1A), histories, side)

    def test_bucket_spec3_sha256(self):
        from vkgr import retrieval_feat02 as f2
        from vkgr import retrieval_feat03 as m
        spec = {'feat02_bucket_spec_sha256': f2.bucket_spec_sha256(), 'len_edges': list(AUDIT_LEN_EDGES), 'act_edges': list(AUDIT_ACT_EDGES),
                'features3': list(FEATURES3_LITERAL), 'recency_logit_sizes': [7, 6], 'logit_init': 0}
        expected = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        digest = m.bucket_spec3_sha256()
        self.assertEqual(digest, expected)
        self.assertEqual(digest, m.bucket_spec3_sha256())
        self.assertNotEqual(digest, f2.bucket_spec_sha256())
        with mock.patch.object(m, 'LEN_EDGES', (5, 8, 16, 32, 50)):
            self.assertNotEqual(m.bucket_spec3_sha256(), digest)
        with mock.patch.object(m, 'ACT_EDGES', (2, 3, 5, 9, 18)):
            self.assertNotEqual(m.bucket_spec3_sha256(), digest)
        with mock.patch.object(m, 'FEATURES3', tuple(reversed(FEATURES3_LITERAL))):
            self.assertNotEqual(m.bucket_spec3_sha256(), digest)
        with mock.patch.object(f2, 'POSITION_EDGES', (1, 2, 4, 8, 16, 33)):
            self.assertNotEqual(m.bucket_spec3_sha256(), digest)


if __name__ == '__main__':
    unittest.main()
