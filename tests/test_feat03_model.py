"""FEAT-03 (P line) model tests, FEAT-002 part: scaffold, f4_control equivalence fixture, Q1a (position only), Q1b (week gap only).

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); the unit tests call them
and tests/test_feat03_mutations.py re-runs them under deliberately wrong implementations.
Bucket references are independent literals (tests/test_feat02_model.py transcribes them from the train-only FEAT-02 audit).
"""
import copy
import sys
import unittest
from pathlib import Path

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

    def test_q2_q3_not_implemented_yet(self):
        histories, side = tm.model_inputs()
        for variant in (Q2, Q3):
            with self.assertRaises(ValueError):
                build3(variant).users(histories, side)

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


if __name__ == '__main__':
    unittest.main()
