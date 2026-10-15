"""SEQ-M1 anti-leak tests L1-L5, L7-L9 (L6, the holdout-open guard, lives in test_seqm1_run.py next to the CLI it patches).

Core claim (design 7): flipping every non-history field of a row (target, target week/position, past_likes, oov, the within-week
ordinal) leaves the batch tensors AND the user tower input/output bit for bit unchanged, for all five arms; causal masking
means position t cannot see anything after t. Fixture convention: anchored arms run with g=0.5 and a random LN_f so the branch
is exercised (otherwise only the m path would be covered).
"""
import copy
import inspect
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_model import ensure  # noqa: E402
from test_feat02_run import flip_non_history_fields  # noqa: E402
import test_seqm1_model as tm  # noqa: E402

FORBIDDEN_SOURCE = ('target_position', 'past_likes', "['target']", '["target"]', "['oov']", '["oov"]')


def flip_target(item, catalog, k=3):
    """flip_non_history_fields + a different, valid catalog item as the target (so a leaking mutant gets a legal id)."""
    item = flip_non_history_fields(item)
    original = item['target']
    pool = [c for c in catalog if c != original]
    item['target'] = pool[(catalog.index(original) + k) % len(pool)] if original in catalog else pool[0]
    return item


def build_for(data, arm, seed=42):
    from vkgr.retrieval_seqm1 import make_seqm1_model
    from vkgr.training import seed_all
    seed_all(seed)
    return tm.activate(make_seqm1_model(data['content'], data['supported_indices'], 50, arm))


def synthetic_view():
    from vkgr import retrieval_feat02_run as fr
    data = fr.synthetic_data(); rows, queries = fr._view(data, 50)
    return fr, data, rows, queries


def guard_target_fields_do_not_reach_the_user_tower():
    """L1: flip target week (last-history week / 24 / 26 / 30) and every other non-history field -> tensors and user vectors identical (5 arms x train rows / validation queries)."""
    fr, data, rows, queries = synthetic_view()
    models = {arm: build_for(data, arm).eval() for arm in tm.ARMS}
    picks = (lambda r: r['history_positions'][-1][0], lambda r: 24, lambda r: 26, lambda r: 30)
    for batch in (rows[:24], queries[:24]):
        width = max(len(r['history']) for r in batch)
        base = fr._tensors2(batch, data['idmap'], width, 'f3_f4_recency')
        with torch.no_grad():
            base_users = {arm: model.users(*base) for arm, model in models.items()}
        for pick in picks:
            moved = []
            for r in batch:
                x = flip_target(r, data['catalog']); x['target_position'] = [pick(r), 5]; moved.append(x)
            got = fr._tensors2(moved, data['idmap'], width, 'f3_f4_recency')
            ensure(torch.equal(got[0], base[0]) and torch.equal(got[1], base[1]), 'batch tensors changed with the target fields')
            with torch.no_grad():
                for arm, model in models.items():
                    ensure(torch.equal(model.users(*got), base_users[arm]), f'{arm}: user vector changed with the target fields')


def guard_export_path_is_target_free():
    """L2: the evaluation export (normalised user vectors) is unchanged by flipping target fields; training-style and evaluation-style tensors for the same history agree."""
    fr, data, rows, queries = synthetic_view()
    batch = queries[:20]
    flipped = [flip_target(q, data['catalog']) for q in batch]
    for arm in tm.ARMS:
        model = build_for(data, arm)
        a = fr.export_vectors_side2(model, data['catalog'], batch, data['idmap'], 16, 50, 'f3_f4_recency')[0]
        b = fr.export_vectors_side2(model, data['catalog'], flipped, data['idmap'], 16, 50, 'f3_f4_recency')[0]
        ensure((a == b).all(), f'{arm}: export_vectors_side2 changed with the target fields')
    longest = max(len(q['history']) for q in batch)
    t1 = fr._tensors2(batch, data['idmap'], longest, 'f3_f4_recency'); t2 = fr._tensors2(batch, data['idmap'], 50, 'f3_f4_recency')
    ensure(torch.equal(t2[0][:, :longest], t1[0]) and torch.equal(t2[1][:, :longest], t1[1]), 'training-style and eval-style tensors differ on the shared prefix')
    ensure(not t2[0][:, longest:].any() and not t2[1][:, longest:].any(), 'width extension is not pure PAD')


def guard_candidate_side_is_user_free():
    """L9 (signature part): items() takes indices only."""
    from vkgr import retrieval_seqm1 as m
    params = list(inspect.signature(m.SeqM1IDMean.items).parameters)
    ensure(params == ['self', 'indices'], f'items() signature is {params}')


def guard_item_tower_independent_of_users():
    """L9: item vectors are bit for bit the same before / after users() on different histories (no hidden user state in the item tower)."""
    index = torch.arange(1, 300)
    for arm in tm.ARMS:
        model = tm.activate(tm.build(arm)).eval(); before = model.items(index)
        with torch.no_grad():
            model.users(*tm.make_batch([5, 12], 12, 1)); middle = model.items(index)
            model.users(*tm.make_batch([9, 3], 12, 2)); after = model.items(index)
        ensure(torch.equal(before, middle) and torch.equal(before, after), f'{arm}: item vectors depend on previous user batches')
    guard_candidate_side_is_user_free()


def guard_changing_history_changes_user_vector():
    """L3: item / timespent (bucket-crossing) / week (bucket-crossing) edits, a prepended older event and a removed event all change the vector; restoring brings it back."""
    from vkgr.retrieval_feat02 import week_gap, weekgap_bucket
    h, s = tm.controlled_row()
    for arm in tm.ARMS:
        model = tm.activate(tm.build(arm)).eval()
        with torch.no_grad():
            base = model.users(h, s)
            ensure(torch.equal(model.users(h.clone(), s.clone()), base), f'{arm}: restore does not give the original vector')
            s1 = s.clone(); s1[0, 2, 1] = tm.other_timespent(int(s[0, 2, 1]))
            s2 = s.clone(); s2[0, 2, 3] = 20
            v = h != 0
            ensure(not torch.equal(weekgap_bucket(week_gap(s2[..., 3], v), v), weekgap_bucket(week_gap(s[..., 3], v), v)), 'fixture: week-gap bucket must change')
            h3 = h.clone(); h3[0, 2] = 99
            older_h = torch.cat([torch.tensor([[98]]), h], 1); older_s = torch.cat([torch.zeros(1, 1, 6, dtype=torch.long), s], 1)
            older_s[0, 0, :4] = torch.tensor([1, 30, 30, 0])
            cut_h, cut_s = h[:, 1:], s[:, 1:]
            for name, (eh, es) in {'timespent': (h, s1), 'week': (h, s2), 'item': (h3, s), 'older_event': (older_h, older_s), 'removed_event': (cut_h, cut_s)}.items():
                ensure(not torch.equal(model.users(eh, es), base), f'{arm}: user vector ignores {name}')


def guard_week_translation_invariance():
    """L4: adding a constant to every history week (staying inside 0..24) leaves the user vector bit for bit unchanged (no absolute calendar channel)."""
    h, s = tm.make_batch([4, 9, 12, 7], 12, 61)
    s[..., 3] = (s[..., 3] // 2) * (h != 0)                       # weeks 0..12, still non-decreasing
    shifted = s.clone(); shifted[..., 3] = (shifted[..., 3] + 9) * (h != 0)
    for arm in tm.ARMS:
        model = tm.activate(tm.build(arm)).eval()
        with torch.no_grad():
            ensure(torch.equal(model.users(h, s), model.users(h, shifted)), f'{arm}: user vector depends on the absolute week')


def guard_pad_columns_do_not_change_back_index():
    """L8: n and the O rows of the real events do not depend on how many PAD columns follow."""
    from vkgr.retrieval_seqm1 import order_ids
    h, _ = tm.make_batch([3, 5], 5, 7)
    wide = torch.zeros(2, 9, dtype=torch.long); wide[:, :5] = h
    ensure(order_ids(h != 0).tolist() == order_ids(wide != 0)[:, :5].tolist(), 'back index changed with the PAD width')


class LeakTests(unittest.TestCase):
    def test_l1_target_fields_do_not_reach_the_user_tower(self):
        guard_target_fields_do_not_reach_the_user_tower()

    def test_l2_export_path_is_target_free(self):
        guard_export_path_is_target_free()

    def test_l3_changing_history_changes_the_user_vector(self):
        guard_changing_history_changes_user_vector()

    def test_l4_week_translation_invariance(self):
        guard_week_translation_invariance()

    def test_l5_source_guard(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_seqm1 as m
        sources = {'retrieval_seqm1': inspect.getsource(m)}
        try:
            from vkgr import retrieval_seqm1_run as run
            sources['retrieval_seqm1_run'] = inspect.getsource(run)
            self.assertIs(run._tensors2, fr._tensors2)             # feature tensors only via the FEAT-02 function
        except ImportError:
            pass
        for name, text in sources.items():
            code = '\n'.join(line for line in text.splitlines() if not line.strip().startswith('#'))
            for token in FORBIDDEN_SOURCE:
                self.assertNotIn(token, code, f'{name}: {token}')

    def test_l7_week_25_rejected_on_every_arm(self):
        h, s = tm.make_batch([5, 5], 5, 3); s[0, 4, 3] = 25
        for arm in tm.ARMS:
            model = tm.build(arm)
            for call in (model.users, model.parts):
                with self.assertRaises(ValueError, msg=(arm, call.__name__)):
                    call(h, s)

    def test_l8_pad_columns(self):
        guard_pad_columns_do_not_change_back_index()
        h, s = tm.make_batch([3, 5], 5, 7); s[0, 4, 1] = 9               # PAD slot carrying a value
        for arm in tm.ARMS:
            with self.assertRaises(ValueError, msg=arm):
                tm.build(arm).users(h, s)

    def test_l9_candidate_side_independent(self):
        guard_item_tower_independent_of_users()


if __name__ == '__main__':
    unittest.main()
