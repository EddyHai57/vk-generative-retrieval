"""FEAT-04b model tests, scaffold + chain A stage (``f4_control`` and ``f4_recency_pool`` on the F4-T010 anchor; C/D/E come in later features).

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); the unit tests call them and
tests/test_feat04b_mutations.py re-runs them under deliberately wrong implementations.
"""
import re
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat02_model as tm  # noqa: E402
import test_feat03_model as t3  # noqa: E402

ensure = tm.ensure
CONTROL, A = 'f4_control', 'f4_recency_pool'
FEATURES4B_LITERAL = ('f4_control', 'f4_recency_pool', 'f4_item_popularity', 'f4_item_quality', 'f4_recency_behavior')
IMPLEMENTED_NOW = (CONTROL, A)
ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CALLS = re.compile(r'^\s*from\s+\.?(?:vkgr\.)?(retrieval_feat02|retrieval_feat02_run|retrieval_feat03|retrieval_feat03_run)\s+import\b', re.M)
SCANNED_SOURCES = ('retrieval_feat04b.py', 'retrieval_feat04b_run.py')          # item_stats.py joins the list when it exists (FEAT-002)


def m4b():
    from vkgr import retrieval_feat04b
    return retrieval_feat04b


def build4b(variant, seed=42):
    return m4b().make_feat04b_model(variant, tm.CONTENT, tm.SUPPORTED, 50, seed)


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
    state = build4b(A).state_dict()
    for key in control:
        ensure(key in state and torch.equal(state[key], control[key]), f'A: anchor table {key} differs from f4_control')


def guard_s_spec_hash_per_variant():
    """bucket_spec_sha256 is per variant: stable, differs between control and A, rejects variants that are not registered / not implemented yet."""
    m = m4b()
    digests = {v: m.bucket_spec_sha256(v) for v in IMPLEMENTED_NOW}
    ensure(all(m.bucket_spec_sha256(v) == d for v, d in digests.items()), 'bucket_spec_sha256 is not stable')
    ensure(len(set(digests.values())) == len(digests), 'control and A must have different bucket_spec_sha256')
    ensure(all(len(d) == 64 and set(d) <= set('0123456789abcdef') for d in digests.values()), 'bucket_spec_sha256 must be a sha256 hex digest')
    for bad in ('nope', 'f4_item_popularity', 'f4_item_quality', 'f4_recency_behavior'):
        try:
            m.bucket_spec_sha256(bad)
        except ValueError:
            continue
        raise AssertionError(f'bucket_spec_sha256({bad!r}) must raise ValueError')


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


# ---------------------------------------------------------------- tests

class ScaffoldTests(unittest.TestCase):
    def test_constants(self):
        m = m4b()
        self.assertEqual(m.FEATURES4B, FEATURES4B_LITERAL)
        self.assertEqual(m.IMPLEMENTED[:2], (CONTROL, A))

    def test_factory_rejects_bad_requests(self):
        m = m4b()
        for variant in ('nope', 'f3_pos_only', 'f3_user_activity'):
            with self.assertRaises(ValueError):                               # not a FEAT-04b variant
                m.make_feat04b_model(variant, tm.CONTENT, tm.SUPPORTED, 50, 42)
        for variant in FEATURES4B_LITERAL[2:]:
            with self.assertRaises(ValueError):                               # registered, implemented in a later feature
                m.make_feat04b_model(variant, tm.CONTENT, tm.SUPPORTED, 50, 42)
        with self.assertRaises(ValueError):
            m.make_feat04b_model(A, tm.CONTENT, tm.SUPPORTED, 33, 42)
        with self.assertRaises(ValueError):
            m.make_feat04b_model(A, tm.CONTENT, tm.SUPPORTED, 50, 42, stat_buckets=object())        # only C/D take stat buckets

    def test_class_bypasses_the_feat03_constructor(self):
        from vkgr import retrieval_feat03 as f3
        model = build4b(A)
        self.assertIsInstance(model, f3.Feat03IDMean)
        self.assertEqual(model.variant, A); self.assertEqual(model.feature, 'f3_f4_recency')
        with self.assertRaises(ValueError):
            m4b().Feat04bIDMean('nope', tm.CONTENT, tm.SUPPORTED, 50)

    def test_unimplemented_variant_users_raise(self):
        model = build4b(CONTROL)
        object.__setattr__(model, 'variant', 'f4_item_popularity')
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


if __name__ == '__main__':
    unittest.main()
