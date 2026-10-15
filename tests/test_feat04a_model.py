"""FEAT-04a model tests (A = f4_recency_pool on the F4-T010 anchor; B is added in the second stage).

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); the unit tests call them and
tests/test_feat04a_mutations.py re-runs them under deliberately wrong implementations (named M4S/M4A/M4B/M4L in the frozen card).
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
ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CALLS = re.compile(r'^\s*from\s+\.?(?:vkgr\.)?(retrieval_feat02|retrieval_feat02_run|retrieval_feat03|retrieval_feat03_run)\s+import\b', re.M)


def m4():
    from vkgr import retrieval_feat04a
    return retrieval_feat04a


def build4(variant, seed=42):
    return m4().make_feat04a_model(variant, tm.CONTENT, tm.SUPPORTED, 50, seed)


# ---------------------------------------------------------------- shared guards (guard_s_*)

def guard_s_anchor_tables_bitwise():
    """Every non-control variant draws item, T, R, P, G exactly like f4_control at the same seed (bit for bit)."""
    control = build4(CONTROL).state_dict()
    for variant in m4().FEATURES4A:
        if variant == CONTROL:
            continue
        state = build4(variant).state_dict()
        for key in control:
            ensure(key in state and torch.equal(state[key], control[key]), f'{variant}: anchor table {key} differs from f4_control')


def scan_calls(source):
    """Fail when a source text imports a FEAT-02/03 name directly (K2: seams are resolved through the module at call time)."""
    ensure(FORBIDDEN_CALLS.search(source) is None, 'from .retrieval_featXX import <name> breaks the module-attribute seams')


def guard_s_module_attribute_calls():
    for name in ('retrieval_feat04a.py', 'retrieval_feat04a_run.py'):
        path = ROOT / 'src' / 'vkgr' / name
        ensure(path.exists(), f'{name} missing')
        scan_calls(path.read_text())


# ---------------------------------------------------------------- A guards (guard_a_*)

def guard_a_start_equals_anchor():
    """With both logit vectors at 0 the A user vector is the f4_control user vector bit for bit (PAD columns, supported/unsupported mix)."""
    model, control = build4(A), build4(CONTROL)
    histories, side = tm.model_inputs(); valid = histories != 0
    ensure(bool((~valid).any()), 'fixture needs PAD columns in a mixed-length batch')
    ensure(bool((valid & ~model.supported[histories]).any()) and bool((valid & model.supported[histories]).any()), 'fixture needs supported and unsupported events')
    for h, s in ((histories, side), (tm.all_supported(histories), side), (tm.all_unsupported(histories), side), t3.pad_columns(histories, side, 7)):
        ensure(torch.equal(model.users(h, s), control.users(h, s)), 'A at its start differs from the anchor')
    ensure(torch.equal(model.items(torch.arange(0, 301)), control.items(torch.arange(0, 301))), 'A item tower differs from the anchor')


def guard_a_formula():
    """A user = sum_k w_k x_k / sum_k w_k with w_k = exp(l_k - max l), l_k = u[pb_k - 1] + v[gb_k - 1], x_k the F4 event vector (non-uniform logits)."""
    model = t3.set_logits(build4(A)); histories, side = tm.model_inputs(); valid = histories != 0
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
    model = t3.set_logits(build4(A)); histories, side = tm.model_inputs()
    base = model.users(histories, side)
    for extra in (1, 5, 20):
        ensure(torch.equal(model.users(*t3.pad_columns(histories, side, extra)), base), f'A changed when {extra} PAD columns were appended')


def guard_a_logits_receive_gradient():
    """Both logit vectors receive finite, non-zero gradients on every used row at the zero start."""
    model = build4(A)
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
    """13 nn.Parameter scalars beyond the anchor, exactly 0 at init; parameter report equals the formula."""
    model, control = build4(A), build4(CONTROL)
    extra = set(dict(model.named_parameters())) - set(dict(control.named_parameters()))
    ensure(extra == {'recency_logit_pos', 'recency_logit_gap'}, f'unexpected extra parameters {sorted(extra)}')
    ensure(sum(p.numel() for n, p in model.named_parameters() if n in extra and p.requires_grad) == 13, 'exactly 13 trainable scalars beyond the anchor')
    ensure(all(bool((getattr(model, n) == 0).all()) and isinstance(getattr(model, n), torch.nn.Parameter) for n in extra), 'logits must start exactly at 0')
    ensure(m4().parameter_report4a(model) == m4().expected_param_counts4a(A), 'parameter report differs from the formula')
    ensure(m4().parameter_report4a(control) == m4().expected_param_counts4a(CONTROL), 'control parameter report differs from the formula')


def guard_a_pooled_weights_pad_cannot_overflow():
    """pooled_weights is a double where: a 1e4 PAD logit gives finite weights, PAD weight exactly 0 and a finite backward pass."""
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
        m = m4()
        self.assertEqual(m.CONTROL, CONTROL); self.assertEqual(m.A_FEATURE, A)
        self.assertEqual(m.FEATURES4A[:2], (CONTROL, A))
        self.assertEqual(m.REAL_FEATURES4A[0], A)
        with self.assertRaises(ValueError):
            m.make_feat04a_model('nope', tm.CONTENT, tm.SUPPORTED, 50, 42)
        with self.assertRaises(ValueError):
            m.make_feat04a_model(A, tm.CONTENT, tm.SUPPORTED, 50, 42, author_index=torch.zeros(301, dtype=torch.long))

    def test_control_is_the_feat03_control_and_the_feat02_f4_model(self):
        from vkgr import retrieval_feat03 as f3
        a = build4(CONTROL).state_dict(); b = f3.make_feat03_model(CONTROL, tm.CONTENT, tm.SUPPORTED, 50, 42).state_dict(); c = tm.build('f3_f4_recency').state_dict()
        self.assertEqual(list(a), list(b)); self.assertEqual(list(a), list(c))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]) and torch.equal(a[k], c[k]), k)

    def test_bucket_spec_digest_differs_per_variant_and_is_stable(self):
        m = m4()
        self.assertEqual(m.bucket_spec4a_sha256(A), m.bucket_spec4a_sha256(A))
        self.assertNotEqual(m.bucket_spec4a_sha256(A), m.bucket_spec4a_sha256(CONTROL))
        with self.assertRaises(ValueError):
            m.bucket_spec4a_sha256('nope')

    def test_parameter_counts(self):
        guard_a_param_count_13()


class SharedGuardTests(unittest.TestCase):
    def test_anchor_tables_bitwise(self):
        guard_s_anchor_tables_bitwise()

    def test_module_attribute_calls(self):
        guard_s_module_attribute_calls()

    def test_scan_bites_on_a_direct_import(self):
        for bad in ('from .retrieval_feat03 import pooled_weights\n', '    from .retrieval_feat02 import x', 'from .retrieval_feat03_run import y', 'from .retrieval_feat02_run import z'):
            with self.assertRaises(AssertionError):
                scan_calls(bad)
        scan_calls('from . import retrieval_feat03 as f3\nf3.pooled_weights\n')


class ATests(unittest.TestCase):
    def test_start_equals_anchor(self):
        guard_a_start_equals_anchor()

    def test_formula(self):
        guard_a_formula()

    def test_pad_width_invariance(self):
        guard_a_pad_width_invariance()

    def test_logits_receive_gradient(self):
        guard_a_logits_receive_gradient()

    def test_pooled_weights_pad_cannot_overflow(self):
        guard_a_pooled_weights_pad_cannot_overflow()


if __name__ == '__main__':
    unittest.main()
