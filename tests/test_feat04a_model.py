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
CONTROL, A, B = 'f4_control', 'f4_recency_pool', 'f4_hist_author_content'
ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_CALLS = re.compile(r'^\s*from\s+\.?(?:vkgr\.)?(retrieval_feat02|retrieval_feat02_run|retrieval_feat03|retrieval_feat03_run)\s+import\b', re.M)


def m4():
    from vkgr import retrieval_feat04a
    return retrieval_feat04a


def b_fixture():
    """(author_index, vocab_size) of the FEAT-02 F6 fixture: rows 0 PAD / 1 OOV / 2+rank, half of the unsupported items out of vocabulary."""
    index = tm.f6_author_index()
    return index, int(index.max()) - 1


def build4(variant, seed=42):
    if variant == B:
        index, vocab = b_fixture()
        return m4().make_feat04a_model(B, tm.F6_CONTENT, tm.SUPPORTED, 50, seed, author_index=index, vocab_size=vocab)
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


# ---------------------------------------------------------------- B guards (guard_b_*)

def guard_b_formula():
    """B user = mean_k [ e_k + Author[a_k]*(a_k >= 2) + W c_k + T + R + P + G ] over the valid history events (independent implementation)."""
    import torch.nn.functional as F
    model = build4(B); histories, side = tm.model_inputs(); valid = histories != 0
    x, pos_ids, gap_ids = t3.reference_event_vectors(model, histories, side, valid)
    rows = model.author_index[histories]
    ensure(len(set(rows[valid].tolist())) >= 4 and bool((rows[valid] == 1).any()) and bool((rows[valid] >= 2).any()), 'fixture must cover OOV and several vocabulary authors')
    expected = torch.zeros(len(histories), 64)
    for j in range(len(histories)):
        n = int(valid[j].sum()); acc = torch.zeros(64)
        for k in range(n):
            row = int(rows[j, k]); hv = F.linear(tm.F6_CONTENT[histories[j, k]], model.content_projection.weight)
            if row >= 2:
                hv = hv + model.author_table.weight[row]
            acc = acc + x[j, k] + hv
        expected[j] = acc / n
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-6, rtol=0), 'B is not the mean of e + Author + W c + T + R + P + G')
    ensure(not torch.allclose(model.users(histories, side), build4(CONTROL).users(histories, side), atol=1e-4, rtol=0), 'B must differ from the anchor')


def guard_b_item_tower_untouched():
    """The whole catalog item tower equals the anchor's bit for bit (B only touches the history side)."""
    model, control = build4(B), build4(CONTROL)
    ensure(torch.equal(model.items(torch.arange(0, 301)), control.items(torch.arange(0, 301))), 'B item tower differs from the anchor')


def guard_b_candidate_unsupported_zero():
    """Unsupported candidates keep exactly zero vectors (no Author / W c on the candidate side)."""
    model = build4(B); unsupported = torch.arange(200, 301)
    ensure(bool((model.items(unsupported).abs().sum(1) == 0).all()), 'an unsupported candidate got a non-zero vector')
    ensure(bool((model.items(torch.arange(1, 200)).abs().sum(1) > 0).all()), 'supported candidates must be non-zero')


def guard_b_oov_zero_no_grad():
    """Author rows 0 (PAD) and 1 (OOV) contribute exactly zero and receive no gradient; in-vocabulary rows do receive gradient."""
    model = build4(B); histories, side = tm.model_inputs(); base = model.users(histories, side)
    ensure(bool((model.author_index[histories][histories != 0] == 1).any()), 'fixture needs OOV history events')
    with torch.no_grad():
        model.author_table.weight[0] = 50.; model.author_table.weight[1] = -50.
    ensure(torch.equal(model.users(histories, side), base), 'author rows 0 and 1 changed a user vector')
    model.zero_grad(); model.users(histories, side).sum().backward(); grad = model.author_table.weight.grad
    ensure(not bool(grad[0].any()) and not bool(grad[1].any()), 'author rows 0 and 1 must receive no gradient')
    ensure(bool(grad[2:].any()), 'in-vocabulary author rows must receive gradient')


def guard_b_unsupported_history_gets_identity():
    """An unsupported history event used to be an all-zero id vector; in B it carries Author (when in vocabulary) + W c."""
    import torch.nn.functional as F
    model = build4(B); histories, _ = tm.model_inputs(); unsupported = tm.all_unsupported(histories)
    ensure(bool((model.items(unsupported[unsupported != 0]).abs().sum(1) == 0).all()), 'fixture: unsupported items have zero id vectors')
    got = m4().history_author_content(model, unsupported)
    for j, i in zip(*torch.nonzero(unsupported, as_tuple=True)):
        item = int(unsupported[j, i]); row = int(model.author_index[item])
        expected = F.linear(tm.F6_CONTENT[item], model.content_projection.weight) + (model.author_table.weight[row] if row >= 2 else 0.)
        ensure(torch.allclose(got[j, i], expected, atol=1e-6, rtol=0), f'unsupported history item {item}: wrong history term')
        ensure(float(got[j, i].abs().sum()) > 0, f'unsupported history item {item} got a zero vector')
    ensure(bool((got[unsupported == 0] == 0).all()), 'PAD slots must be exactly 0')


def guard_b_init_and_draw_order():
    """Author and W are N(0, 0.02) (not Kaiming), PAD row 0, and drawn AFTER item, T, R, P, G: replaying the stream after the control reproduces them."""
    from vkgr.training import seed_all
    model = build4(B)
    ensure(0.015 < float(model.author_table.weight[1:].detach().std()) < 0.026, 'Author init std is not about 0.02')
    ensure(0.017 < float(model.content_projection.weight.std()) < 0.023, f'W init std {float(model.content_projection.weight.std())} is not about 0.02 (Kaiming is about 0.072)')
    ensure(bool((model.author_table.weight[0] == 0).all()), 'Author PAD row must be 0')
    seed_all(42); build4(CONTROL)                       # the control seeds again and draws item, T, R, P, G; the global stream now sits right after them
    author = torch.nn.Embedding(model.vocab_size + 2, 64, padding_idx=0); projection = torch.nn.Linear(64, 64, bias=False)
    with torch.no_grad():
        author.weight.normal_(0.0, 0.02); author.weight[0].zero_(); projection.weight.normal_(0.0, 0.02)
    ensure(torch.equal(model.author_table.weight, author.weight) and torch.equal(model.content_projection.weight, projection.weight), 'draw order: Author/W must follow item, T, R, P, G')
    ensure(model.init_report['embeddings'].get('author_table', {}).get('rows') == model.vocab_size + 2, 'init_report must include the Author table')


def guard_b_pad_width_invariance():
    """Appending PAD columns changes nothing (PAD slots carry no history term)."""
    model = build4(B); histories, side = tm.model_inputs(); base = model.users(histories, side)
    for extra in (1, 5, 20):
        ensure(torch.equal(model.users(*t3.pad_columns(histories, side, extra)), base), f'B changed when {extra} PAD columns were appended')


def guard_b_new_params_receive_gradient():
    model = build4(B); histories, side = tm.model_inputs()
    (model.users(histories, side) @ torch.randn(64, generator=torch.Generator().manual_seed(9))).sum().backward()
    for name in ('author_table', 'content_projection'):
        grad = getattr(model, name).weight.grad
        ensure(grad is not None, f'{name}: no gradient (detached?)')
        ensure(bool(torch.isfinite(grad).all()) and bool(grad.abs().sum() > 0), f'{name}: gradient is zero or not finite')


def guard_b_checkpoint_immutable():
    """load_state_dict refuses a checkpoint whose author_index (or content) differs from the model's."""
    model = build4(B); state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    bad = dict(state); bad['author_index'] = state['author_index'].clone(); bad['author_index'][5] = 2 if int(bad['author_index'][5]) != 2 else 3
    ensure(_rejects(lambda: build4(B).load_state_dict(bad)), 'a different author_index must be refused')
    bad = dict(state); bad['content'] = state['content'] + 1.
    ensure(_rejects(lambda: build4(B).load_state_dict(bad)), 'a different content table must be refused')


def _rejects(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def guard_b_param_count():
    """With the audited vocabulary size (11133) B adds 716,736 total / 716,608 trainable parameters on top of the anchor."""
    from vkgr import retrieval_feat02 as f2
    g = torch.Generator().manual_seed(2)
    index = torch.randint(0, f2.AUTHOR_VOCAB_SIZE + 2, (301,), generator=g); index[0] = 0
    model = m4().make_feat04a_model(B, tm.F6_CONTENT, tm.SUPPORTED, 50, 42, author_index=index)
    report, control = m4().parameter_report4a(model), m4().parameter_report4a(build4(CONTROL))
    ensure(report == m4().expected_param_counts4a(B), 'B parameter report differs from the formula')
    ensure(report['total'] - control['total'] == 716736 and report['trainable_effective'] - control['trainable_effective'] == 716608, 'B must add 716,736 total / 716,608 trainable parameters')


def guard_b_author_content_history_only():
    """L3: changing the author row and the content row of an item that is NOT in the history leaves the user vector bit-equal;
    the reverse controls (a history item's rows) must change it, and restoring them restores it."""
    model = build4(B); histories, side = tm.model_inputs()
    used = set(histories.flatten().tolist()); target = next(i for i in range(300, 0, -1) if i not in used)
    base = model.users(histories, side)
    model.author_index[target] = 2 if int(model.author_index[target]) != 2 else 3
    model.content[target] = tm.F6_CONTENT[1] if target != 1 else tm.F6_CONTENT[2]
    ensure(torch.equal(model.users(histories, side), base), 'the user vector read the author/content of an item that is not in the history')
    item = int(histories[0, 0]); row = int(model.author_index[item])
    model.author_index[item] = 2 + (row - 1) % 6 if row >= 2 else 2
    ensure(not torch.equal(model.users(histories, side), base), 'control: changing the author row of a history item must change the user vector')
    model.author_index[item] = row
    ensure(torch.equal(model.users(histories, side), base), 'restoring the history item author row must restore the user vector')
    model.content[item] = tm.F6_CONTENT[(item % 300) + 1]
    ensure(not torch.equal(model.users(histories, side), base), 'control: changing the content row of a history item must change the user vector')


# ---------------------------------------------------------------- tests

class ScaffoldTests(unittest.TestCase):
    def test_constants(self):
        m = m4()
        self.assertEqual(m.CONTROL, CONTROL); self.assertEqual(m.A_FEATURE, A); self.assertEqual(m.B_FEATURE, B)
        self.assertEqual(m.FEATURES4A, (CONTROL, A, B))
        self.assertEqual(m.REAL_FEATURES4A, (A, B))
        with self.assertRaises(ValueError):
            m.make_feat04a_model('nope', tm.CONTENT, tm.SUPPORTED, 50, 42)
        with self.assertRaises(ValueError):
            m.make_feat04a_model(A, tm.CONTENT, tm.SUPPORTED, 50, 42, author_index=torch.zeros(301, dtype=torch.long))
        with self.assertRaises(ValueError):
            m.make_feat04a_model(B, tm.F6_CONTENT, tm.SUPPORTED, 50, 42)                          # B needs author_index
        index, vocab = b_fixture()
        for bad in (index.float(), index[:-1], torch.cat([torch.ones(1, dtype=torch.long), index[1:]])):
            with self.assertRaises(ValueError):
                m.make_feat04a_model(B, tm.F6_CONTENT, tm.SUPPORTED, 50, 42, author_index=bad, vocab_size=vocab)
        with self.assertRaises(ValueError):
            m.make_feat04a_model(B, tm.F6_CONTENT, tm.SUPPORTED, 50, 42, author_index=index, vocab_size=vocab - 3)   # index exceeds V + 1

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
        self.assertNotEqual(m.bucket_spec4a_sha256(B), m.bucket_spec4a_sha256(A))
        with self.assertRaises(ValueError):
            m.bucket_spec4a_sha256('nope')

    def test_parameter_counts(self):
        guard_a_param_count_13()
        guard_b_param_count()

    def test_b_state_keys_are_the_anchor_keys_plus_author_content(self):
        control, model = build4(CONTROL).state_dict(), build4(B).state_dict()
        self.assertEqual(set(model) - set(control), {'author_table.weight', 'content_projection.weight', 'author_index', 'content'})
        self.assertTrue(set(control) <= set(model))


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


class BTests(unittest.TestCase):
    def test_formula(self):
        guard_b_formula()

    def test_item_tower_untouched(self):
        guard_b_item_tower_untouched()

    def test_candidate_unsupported_zero(self):
        guard_b_candidate_unsupported_zero()

    def test_oov_zero_no_grad(self):
        guard_b_oov_zero_no_grad()

    def test_unsupported_history_gets_identity(self):
        guard_b_unsupported_history_gets_identity()

    def test_init_and_draw_order(self):
        guard_b_init_and_draw_order()

    def test_pad_width_invariance(self):
        guard_b_pad_width_invariance()

    def test_new_params_receive_gradient(self):
        guard_b_new_params_receive_gradient()

    def test_checkpoint_immutable(self):
        guard_b_checkpoint_immutable()

    def test_author_content_history_only(self):
        guard_b_author_content_history_only()

    def test_b_uses_uniform_pooling_not_the_recency_weights(self):
        model = build4(B); self.assertFalse(hasattr(model, 'recency_logit_pos'))
        histories, side = tm.model_inputs(); valid = histories != 0
        x = t3.reference_event_vectors(model, histories, side, valid)[0] + m4().history_author_content(model, histories)
        expected = (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        self.assertTrue(torch.allclose(model.users(histories, side), expected, atol=1e-6, rtol=0))


if __name__ == '__main__':
    unittest.main()
