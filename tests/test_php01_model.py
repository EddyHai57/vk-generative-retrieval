"""P-HP-01 (P line) model tests: the F4 anchor with a configurable embedding dimension.

Primary evidence: at dim=64 ``make_php01_model`` initialises BYTE FOR BYTE like ``make_feat02_model('f3_f4_recency')`` at the
same seed (same keys, dtypes, shapes, bytes, RNG state after construction), so the dim knob cannot change the anchor.
``guard_*`` functions are plain assertion functions (survive ``python -O``); tests/test_php01_mutations.py reuses them.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_model import AUDIT_POSITION_EDGES, AUDIT_WEEKGAP_EDGES, CONTENT, SUPPORTED, ensure, model_inputs, ref_id  # noqa: E402

FEATURE = 'f3_f4_recency'
BUCKET_SPEC_SHA256 = 'cbb1eb40e01ab773da544dde640481377da0b20850fcc5b643c4c995aaaef316'   # recorded in the FEAT-02 report
TABLE_ROWS = {'embedding': 301, 'timespent_bucket': 8, 'ratio_bucket': 6, 'position_table': 8, 'weekgap_table': 7}
PARAM_COUNTS = {32: (928, 800), 64: (1856, 1600), 128: (3712, 3200)}   # (total, trainable_effective), item embedding excluded


def build(dim, seed=42):
    from vkgr.retrieval_php01 import make_php01_model
    from vkgr.training import seed_all
    seed_all(seed)
    return make_php01_model(CONTENT, SUPPORTED, 50, dim)


def build_f4(seed=42):
    from vkgr.retrieval_feat02 import make_feat02_model
    from vkgr.training import seed_all
    seed_all(seed)
    return make_feat02_model(FEATURE, CONTENT, SUPPORTED, 50)


# ---------------------------------------------------------------- guards (reused by the mutation tests)

def guard_dim64_init_bytewise_equals_f4():
    old = build_f4(); old_rng = torch.get_rng_state()
    new = build(64); new_rng = torch.get_rng_state()
    ensure(torch.equal(old_rng, new_rng), 'construction consumed a different number of random draws than F4')
    old_state, new_state = old.state_dict(), new.state_dict()
    ensure(list(new_state) == list(old_state), 'state_dict key order differs from F4')
    for key, value in old_state.items():
        other = new_state[key]
        ensure(other.dtype == value.dtype and other.shape == value.shape, f'{key}: dtype/shape differs from F4')
        ensure(other.numpy().tobytes() == value.numpy().tobytes(), f'{key}: bytes differ from F4')
    ensure([n for n, _ in new.named_parameters()] == [n for n, _ in old.named_parameters()], 'parameter order differs from F4')
    ensure(new.init_report == old.init_report, 'init_report differs from F4')
    histories, side = model_inputs()
    ensure(torch.equal(new.users(histories, side), old.users(histories, side)), 'user vectors differ from F4')
    index = torch.arange(1, 300)
    ensure(torch.equal(new.items(index), old.items(index)), 'item vectors differ from F4')
    probe = build(64); probe.load_state_dict(old.state_dict(), strict=True)
    probe = build_f4(); probe.load_state_dict(new.state_dict(), strict=True)


def guard_dim64_one_step_equals_f4():
    histories, side = model_inputs(seed=3); targets = torch.arange(1, len(histories) + 1) * 7
    params = []
    for model in (build_f4(), build(64)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
        logits = model.users(histories, side) @ model.items(targets).T / .15
        torch.nn.functional.cross_entropy(logits, torch.arange(len(targets))).backward(); optimizer.step()
        params.append([(n, p.detach().clone()) for n, p in model.named_parameters()])
    ensure([n for n, _ in params[0]] == [n for n, _ in params[1]], 'parameter names differ after one step')
    for (name, a), (_, b) in zip(*params):
        ensure(a.numpy().tobytes() == b.numpy().tobytes(), f'{name} differs from F4 after one AdamW step')


def guard_every_table_has_model_dim():
    for dim in (32, 128):
        model = build(dim)
        for name, rows in TABLE_ROWS.items():
            weight = model.get_submodule(name).weight
            ensure(tuple(weight.shape) == (rows, dim), f'dim {dim}: {name} has shape {tuple(weight.shape)}, expected {(rows, dim)}')
            ensure(bool(torch.all(weight[0] == 0)), f'dim {dim}: {name} PAD row not zero')
        histories, side = model_inputs()
        ensure(tuple(model.users(histories, side).shape) == (len(histories), dim), f'dim {dim}: user vector width')
        ensure(tuple(model.items(torch.arange(1, 10)).shape) == (9, dim), f'dim {dim}: item vector width')


def guard_init_std_is_dim_independent():
    for dim in (32, 128):
        model = build(dim); rows = model.embedding.weight.detach()[1:]
        ensure(.018 <= float(rows.std()) <= .022, f'dim {dim}: item embedding std {float(rows.std()):.5f} is not N(0, 0.02)')
        ensure(model.init_report['std'] == .02 and model.init_report['init'] == 'small', f'dim {dim}: init_report std')


def guard_f4_formula_at_dim(dim):
    from vkgr.retrieval_features import ratio_bucket, timespent_bucket
    model = build(dim); histories, side = model_inputs(); valid = histories != 0; n = valid.sum(1)
    pos_ids = torch.zeros_like(histories); gap_ids = torch.zeros_like(histories)
    for j in range(len(histories)):
        count = int(n[j]); weeks = side[j, :count, 3].tolist()
        for k in range(count):
            pos_ids[j, k] = ref_id(count - 1 - k, AUDIT_POSITION_EDGES)
            gap_ids[j, k] = ref_id(weeks[-1] - weeks[k], AUDIT_WEEKGAP_EDGES)
    t = model.timespent_bucket(timespent_bucket(side[..., 1], valid)); r = model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
    expected = (model.items(histories) + t + r + model.position_table(pos_ids) + model.weekgap_table(gap_ids)).sum(1) / n.unsqueeze(1)
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-7, rtol=0), f'dim {dim}: not mean(item + T + R + P[pos] + G[gap])')


def guard_bucket_constants_unchanged():
    from vkgr import retrieval_feat02 as f2, retrieval_features as ft, retrieval_php01 as m
    for name, source in (('POSITION_EDGES', f2), ('WEEKGAP_EDGES', f2), ('MAX_WEEK', f2), ('TIMESPENT_BUCKETS', ft), ('RATIO_BUCKETS', ft)):
        ensure(getattr(m, name) is getattr(source, name), f'{name} must be imported from the frozen module, not copied')
    ensure(m.bucket_spec_sha256() == BUCKET_SPEC_SHA256, 'bucket spec digest differs from FEAT-02')


# ---------------------------------------------------------------- tests

class DimEquivalenceTests(unittest.TestCase):
    def test_dim64_init_bytewise_equals_f4(self):
        guard_dim64_init_bytewise_equals_f4()

    def test_dim64_one_step_equals_f4(self):
        guard_dim64_one_step_equals_f4()

    def test_dim64_param_report_equals_f4(self):
        from vkgr.retrieval_feat02 import parameter_report
        from vkgr.retrieval_php01 import parameter_report_d
        self.assertEqual(parameter_report_d(build(64)), parameter_report(build_f4()))

    def test_bucket_constants_unchanged(self):
        guard_bucket_constants_unchanged()


class DimTests(unittest.TestCase):
    def test_dims_registered(self):
        from vkgr.retrieval_php01 import DIMS
        self.assertEqual(DIMS, (32, 64, 128))

    def test_every_table_has_model_dim(self):
        guard_every_table_has_model_dim()

    def test_init_std_is_dim_independent(self):
        guard_init_std_is_dim_independent()

    def test_f4_formula_at_dim32(self):
        guard_f4_formula_at_dim(32)

    def test_f4_formula_at_dim128(self):
        guard_f4_formula_at_dim(128)

    def test_same_seed_same_dim_is_deterministic(self):
        for dim in (32, 128):
            a, b = build(dim).state_dict(), build(dim).state_dict()
            self.assertEqual(list(a), list(b))
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]), (dim, key))

    def test_param_report_matches_formula_and_literals(self):
        from vkgr.retrieval_php01 import expected_param_counts_d, parameter_report_d
        for dim, literal in PARAM_COUNTS.items():
            report = parameter_report_d(build(dim))
            self.assertEqual(report, expected_param_counts_d(dim), dim)
            self.assertEqual((report['total'], report['trainable_effective']), literal, dim)

    def test_unregistered_dim_length_and_inputs_rejected(self):
        from vkgr.retrieval_php01 import make_php01_model
        for dim in (16, 256, 64.0, True, '64'):
            with self.assertRaises(ValueError, msg=dim):
                make_php01_model(CONTENT, SUPPORTED, 50, dim)
        for max_len in (20, 100, 50.0):
            with self.assertRaises(ValueError, msg=max_len):
                make_php01_model(CONTENT, SUPPORTED, max_len, 64)
        bad_pad = CONTENT.clone(); bad_pad[0, 0] = 1.
        for content in (torch.zeros(301, 32), bad_pad, torch.zeros(301, 64, dtype=torch.float64)):
            with self.assertRaises(ValueError):
                make_php01_model(content, SUPPORTED, 50, 32)
        for supported in ([0], [301], [1.0]):
            with self.assertRaises(ValueError):
                make_php01_model(CONTENT, supported, 50, 32)

    def test_load_state_dict_rejects_changed_supported_buffer(self):
        model = build(32); state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        state['supported'][1] = ~state['supported'][1]
        with self.assertRaises(ValueError):
            model.load_state_dict(state)
        with self.assertRaises(RuntimeError):
            build(32).load_state_dict(build(128).state_dict())        # a checkpoint of another dim cannot be loaded

    def test_users_fail_closed(self):
        model = build(32); histories, side = model_inputs(); valid = histories != 0
        with self.assertRaises(ValueError):
            model.users(histories, None)
        late = side.clone(); late[0, 0, 3] = 25
        with self.assertRaises(ValueError):
            model.users(histories, late)
        pad = side.clone(); slot = (~valid).nonzero()[0]; pad[slot[0], slot[1], 1] = 5
        with self.assertRaises(ValueError):
            model.users(histories, pad)
        with self.assertRaises(ValueError):
            model.users(torch.zeros_like(histories), torch.zeros_like(side))


class LeakageTests(unittest.TestCase):
    def test_user_vector_reads_only_its_own_history_at_dim32(self):
        """A row's user vector depends only on that row's history events (no other row, no target field reaches it)."""
        model = build(32); histories, side = model_inputs(); base = model.users(histories, side)
        for j in range(len(histories)):
            alone = model.users(histories[j:j + 1], side[j:j + 1])
            self.assertTrue(torch.allclose(alone[0], base[j], atol=1e-7, rtol=0), j)
        other_h, other_s = model_inputs(seed=9)
        mixed_h, mixed_s = histories.clone(), side.clone(); mixed_h[1:], mixed_s[1:] = other_h[1:], other_s[1:]
        self.assertTrue(torch.allclose(model.users(mixed_h, mixed_s)[0], base[0], atol=1e-7, rtol=0))

    def test_changing_history_week_changes_user_vector_at_dim32(self):
        model = build(32)
        histories = torch.tensor([[11, 12, 13, 14, 15]]); side = torch.zeros(1, 5, 6, dtype=torch.long)
        side[0, :, 0] = 1; side[0, :, 1] = 20; side[0, :, 2] = 30; side[0, :, 3] = torch.tensor([0, 4, 8, 12, 16])
        base = model.users(histories, side)
        for slot, week in ((1, 8), (3, 15), (4, 24)):
            changed = side.clone(); changed[0, slot, 3] = week
            self.assertFalse(torch.equal(model.users(histories, changed), base), slot)


if __name__ == '__main__':
    unittest.main()
