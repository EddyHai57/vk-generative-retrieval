"""FEAT-02 (P line) model tests: scaffold equals FEAT-01 F3, F3-masked, F4 recency (anchored on the LAST history event).

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); the unit tests call
them and tests/test_feat02_mutations.py re-runs them under deliberately wrong implementations.
Bucket reference values below are independent literals transcribed from the train-only audit (feat02-audit-v1).
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

AUDIT_POSITION_EDGES = (1, 2, 4, 8, 16, 32)   # buckets: 0 | 1 | 2-3 | 4-7 | 8-15 | 16-31 | 32-49  (back index, 0 = last event)
AUDIT_WEEKGAP_EDGES = (1, 2, 3, 5, 9)         # buckets: 0 | 1 | 2 | 3-4 | 5-8 | 9-24             (week of last event - week)
CONTENT = torch.zeros(301, 64)
SUPPORTED = list(range(1, 200))               # items 200..300 are unsupported
# side columns a variant is allowed to read: [feedback_mask, timespent, duration, week, place, platform]
USED_COLUMNS = {'f3': {1, 2}, 'f3_masked': {1, 2}, 'f3_f4_recency': {1, 2, 3}, 'f3_f5_scene': {1, 2, 4, 5}}
AUDIT_PLACE_VALUES = tuple(range(24))         # train week 00-24 place values (audit: 24 values, 0..23)
AUDIT_PLATFORM_VALUES = tuple(range(11))      # platform values 0..10


def ensure(condition, message='guard failed'):
    if not condition:
        raise AssertionError(message)


def ref_id(value, edges):
    """Independent bucket reference: 1 + number of edges <= value (id 0 is PAD)."""
    return 1 + sum(1 for edge in edges if value >= edge)


def model_inputs(seed=0, batch=6, width=12, items=300):
    """Left-aligned histories, 6-column side with non-decreasing weeks in 0..24."""
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, width + 1, (batch,), generator=g)
    lengths[0] = width
    histories = torch.zeros(batch, width, dtype=torch.long); side = torch.zeros(batch, width, 6, dtype=torch.long)
    for j, n in enumerate(lengths.tolist()):
        histories[j, :n] = torch.randint(1, items + 1, (n,), generator=g)
        side[j, :n, 0] = torch.randint(0, 64, (n,), generator=g)
        side[j, :n, 1] = torch.randint(1, 256, (n,), generator=g)
        side[j, :n, 2] = torch.tensor([[5, 9, 25, 60, 180][k] for k in torch.randint(0, 5, (n,), generator=g).tolist()])
        side[j, :n, 3] = torch.sort(torch.randint(0, 25, (n,), generator=g)).values
    return histories, side


def scene_inputs(seed=0, **kwargs):
    """model_inputs plus place/platform columns: mostly table values, some unseen ones (24+ / 11+), the rest in range."""
    histories, side = model_inputs(seed, **kwargs); valid = histories != 0
    g = torch.Generator().manual_seed(seed + 77)
    place = torch.randint(0, 24, valid.shape, generator=g); platform = torch.randint(0, 11, valid.shape, generator=g)
    place[0, :3] = torch.tensor([24, 200, 255]); platform[0, :3] = torch.tensor([11, 77, 255])
    side[..., 4] = torch.where(valid, place, torch.zeros_like(place)); side[..., 5] = torch.where(valid, platform, torch.zeros_like(platform))
    return histories, side


def reference_category_id(value, values):
    """Independent reference: 2 + index in the table, 1 for a value that is not in it (id 0 is PAD)."""
    return 2 + list(values).index(value) if value in values else 1


def build(feature, seed=42):
    from vkgr.retrieval_feat02 import make_feat02_model
    from vkgr.training import seed_all
    seed_all(seed)
    return make_feat02_model(feature, CONTENT, SUPPORTED, 50)


def all_supported(histories):
    return torch.where(histories != 0, (histories - 1) % 199 + 1, torch.zeros_like(histories))


def all_unsupported(histories):
    return torch.where(histories != 0, 200 + histories % 101, torch.zeros_like(histories))


# ---------------------------------------------------------------- guards (reused by the mutation tests)

def guard_masked_adds_on_supported_only():
    from vkgr.retrieval_features import ratio_bucket, timespent_bucket
    model = build('f3_masked'); histories, side = model_inputs(); valid = histories != 0
    mask = (valid & model.supported[histories]).unsqueeze(-1)
    ensure(bool(mask.any()) and bool((valid.unsqueeze(-1) & ~mask).any()), 'fixture must mix supported/unsupported events')
    t = model.timespent_bucket(timespent_bucket(side[..., 1], valid)); r = model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
    m = mask.float()
    expected = (model.items(histories) + t * m + r * m).sum(1) / valid.sum(1, keepdim=True)
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-7, rtol=0), 'F3-masked is not item + (T+R) on supported events only')


def guard_masked_differs_from_f3_exactly_on_unsupported():
    from vkgr.retrieval_features import ratio_bucket, timespent_bucket
    masked, f3 = build('f3_masked'), build('f3')
    histories, side = model_inputs(); valid = histories != 0
    unsupported = (valid & ~masked.supported[histories]).unsqueeze(-1)
    ensure(bool(unsupported.any()), 'fixture needs unsupported events')
    t = f3.timespent_bucket(timespent_bucket(side[..., 1], valid)); r = f3.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
    delta = f3.users(histories, side) - masked.users(histories, side)
    expected = ((t + r) * unsupported.float()).sum(1) / valid.sum(1, keepdim=True)
    ensure(torch.allclose(delta, expected, atol=1e-6, rtol=0), 'difference to F3 is not exactly the unsupported events T+R')
    ensure(not torch.equal(f3.users(histories, side), masked.users(histories, side)), 'F3-masked must differ from F3 on unsupported events')
    sup = all_supported(histories)
    ensure(torch.equal(f3.users(sup, side), masked.users(sup, side)), 'F3-masked must equal F3 on all-supported histories')


def guard_position_bucket_boundaries():
    from vkgr.retrieval_feat02 import POSITION_EDGES, position_bucket
    ensure(POSITION_EDGES == AUDIT_POSITION_EDGES, 'POSITION_EDGES not transcribed from the audit')
    got = position_bucket(torch.ones(1, 50, dtype=torch.bool))[0].tolist()
    for slot in range(50):
        ensure(got[slot] == ref_id(49 - slot, AUDIT_POSITION_EDGES), f'back index {49 - slot} in wrong bucket')
    for edge in AUDIT_POSITION_EDGES:
        for back in (edge - 1, edge, edge + 1):
            ensure(got[49 - back] == ref_id(back, AUDIT_POSITION_EDGES), f'edge {edge} neighbour {back}')
    padded = torch.zeros(2, 50, dtype=torch.bool); padded[0, :10] = True; padded[1, :1] = True
    ids = position_bucket(padded)
    ensure(bool(ids[0, 10:].eq(0).all()) and bool(ids[1, 1:].eq(0).all()), 'PAD slots must map to 0')
    ensure(ids[0, :10].tolist() == [ref_id(9 - k, AUDIT_POSITION_EDGES) for k in range(10)], 'back index must count from the last event')
    ensure(int(ids[1, 0]) == 1, 'single event is back index 0')


def guard_weekgap_bucket_boundaries():
    from vkgr.retrieval_feat02 import WEEKGAP_EDGES, week_gap, weekgap_bucket
    ensure(WEEKGAP_EDGES == AUDIT_WEEKGAP_EDGES, 'WEEKGAP_EDGES not transcribed from the audit')
    valid = torch.ones(1, 25, dtype=torch.bool)
    gap = week_gap(torch.arange(25).unsqueeze(0), valid)
    ensure(gap[0].tolist() == [24 - k for k in range(25)], 'gap must be last week minus event week')
    got = weekgap_bucket(gap, valid)[0].tolist()
    for g in range(25):
        ensure(got[24 - g] == ref_id(g, AUDIT_WEEKGAP_EDGES), f'gap {g} in wrong bucket')
    pad = torch.zeros(1, 25, dtype=torch.bool); pad[0, :3] = True
    ensure(bool(weekgap_bucket(week_gap(torch.arange(25).unsqueeze(0), pad), pad)[0, 3:].eq(0).all()), 'PAD slots must map to 0')


def guard_week_gap_anchored_on_last_history_event():
    from vkgr.retrieval_feat02 import week_gap
    week = torch.tensor([[3, 3, 5, 9, 0, 0], [4, 7, 0, 0, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    ensure(week_gap(week, valid).tolist() == [[6, 6, 4, 0, 0, 0], [3, 0, 0, 0, 0, 0]], 'week gap not anchored on the last history event')


def guard_f4_formula():
    from vkgr.retrieval_features import ratio_bucket, timespent_bucket
    model = build('f3_f4_recency'); histories, side = model_inputs(); valid = histories != 0; n = valid.sum(1)
    pos_ids = torch.zeros_like(histories); gap_ids = torch.zeros_like(histories)
    for j in range(len(histories)):
        count = int(n[j]); weeks = side[j, :count, 3].tolist()
        for k in range(count):
            pos_ids[j, k] = ref_id(count - 1 - k, AUDIT_POSITION_EDGES)
            gap_ids[j, k] = ref_id(weeks[-1] - weeks[k], AUDIT_WEEKGAP_EDGES)
    t = model.timespent_bucket(timespent_bucket(side[..., 1], valid)); r = model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
    expected = (model.items(histories) + t + r + model.position_table(pos_ids) + model.weekgap_table(gap_ids)).sum(1) / n.unsqueeze(1)
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-7, rtol=0), 'F4 is not mean(item + T + R + P[pos] + G[gap])')


def guard_changing_history_week_changes_feature():
    model = build('f3_f4_recency')
    histories = torch.tensor([[11, 12, 13, 14, 15]]); side = torch.zeros(1, 5, 6, dtype=torch.long)
    side[0, :, 0] = 1; side[0, :, 1] = 20; side[0, :, 2] = 30; side[0, :, 3] = torch.tensor([0, 4, 8, 12, 16])   # gap buckets [6, 6, 5, 4, 1]
    base = model.users(histories, side)
    for slot, week in ((1, 8), (3, 15), (4, 24)):      # an older event moves closer / a recent one moves closer / last event later (shifts every gap); each stays monotone
        changed = side.clone(); changed[0, slot, 3] = week
        ensure(not torch.equal(model.users(histories, changed), base), f'changing the week of history event {slot} left the feature unchanged')


def guard_f5_formula():
    from vkgr.retrieval_features import ratio_bucket, timespent_bucket
    model = build('f3_f5_scene'); histories, side = scene_inputs(); valid = histories != 0
    place_ids = torch.zeros_like(histories); platform_ids = torch.zeros_like(histories)
    for j in range(len(histories)):
        for k in range(int(valid[j].sum())):
            place_ids[j, k] = reference_category_id(int(side[j, k, 4]), AUDIT_PLACE_VALUES)
            platform_ids[j, k] = reference_category_id(int(side[j, k, 5]), AUDIT_PLATFORM_VALUES)
    ensure(bool((place_ids == 1).any()) and bool((platform_ids == 1).any()) and bool((place_ids >= 2).any()), 'fixture must mix table and unseen values')
    place = model.place_table(place_ids) * (place_ids >= 2).unsqueeze(-1); platform = model.platform_table(platform_ids) * (platform_ids >= 2).unsqueeze(-1)
    t = model.timespent_bucket(timespent_bucket(side[..., 1], valid)); r = model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
    expected = (model.items(histories) + t + r + place + platform).sum(1) / valid.sum(1, keepdim=True)
    ensure(torch.allclose(model.users(histories, side), expected, atol=1e-7, rtol=0), 'F5 is not mean(item + T + R + Place[place] + Platform[platform])')


def guard_f5_unseen_value_maps_to_oov_and_contributes_zero():
    from vkgr.retrieval_feat02 import PLACE_VALUES, PLATFORM_VALUES, category_ids
    ensure(PLACE_VALUES == AUDIT_PLACE_VALUES and PLATFORM_VALUES == AUDIT_PLATFORM_VALUES, 'value tables not transcribed from the audit')
    valid = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    raw = torch.tensor([[0, 23, 24, 200, 255, 0]])
    ids = category_ids(raw, PLACE_VALUES, valid)[0].tolist()
    ensure(ids == [2, 25, 1, 1, 1, 0], f'place ids {ids}: table index + 2, unseen -> 1 (OOV), PAD -> 0')
    ids = category_ids(torch.tensor([[0, 10, 11, 77, 255, 3]]), PLATFORM_VALUES, valid)[0].tolist()
    ensure(ids == [2, 12, 1, 1, 1, 0], f'platform ids {ids}')
    model = build('f3_f5_scene'); f3 = build('f3')
    histories, side = model_inputs(); valid = histories != 0
    side[..., 4] = torch.where(valid, torch.full_like(side[..., 4], 200), torch.zeros_like(side[..., 4]))
    side[..., 5] = torch.where(valid, torch.full_like(side[..., 5], 99), torch.zeros_like(side[..., 5]))
    base = model.users(histories, side)
    ensure(torch.equal(base, f3.users(histories, side)), 'all-unseen scene must equal the F3 vector exactly')
    with torch.no_grad():
        model.place_table.weight[1] = 100.; model.platform_table.weight[1] = -100.      # the OOV rows are inert
    ensure(torch.equal(model.users(histories, side), base), 'OOV row must contribute exactly zero')
    loss = model.users(histories, side).sum(); loss.backward()
    ensure(not bool(model.place_table.weight.grad.any()) and not bool(model.platform_table.weight.grad.any()), 'unseen values must not train any table row')


class ScaffoldTests(unittest.TestCase):
    def test_constants_transcribed_from_audit(self):
        from vkgr import retrieval_feat02 as m
        self.assertEqual(m.POSITION_EDGES, AUDIT_POSITION_EDGES)
        self.assertEqual(m.WEEKGAP_EDGES, AUDIT_WEEKGAP_EDGES)
        self.assertEqual(m.MAX_WEEK, 24)
        self.assertEqual(m.SIDE_WIDTH, 6)

    def test_features_are_registered_in_order(self):
        from vkgr.retrieval_feat02 import FEATURES
        self.assertEqual(FEATURES, ('f3', 'f3_masked', 'f3_f4_recency', 'f3_f5_scene', 'f3_f6_author_content')[:len(FEATURES)])
        self.assertEqual(FEATURES[:4], ('f3', 'f3_masked', 'f3_f4_recency', 'f3_f5_scene'))

    def test_bucket_spec_sha256_is_stable_and_sensitive(self):
        from vkgr import retrieval_feat02 as m
        first = m.bucket_spec_sha256()
        self.assertEqual(first, m.bucket_spec_sha256())
        self.assertRegex(first, '^[0-9a-f]{64}$')
        with mock.patch.object(m, 'POSITION_EDGES', (1, 2, 4, 8, 16)):
            self.assertNotEqual(m.bucket_spec_sha256(), first)
        with mock.patch.object(m, 'WEEKGAP_EDGES', (1, 2, 3, 5, 9, 17)):
            self.assertNotEqual(m.bucket_spec_sha256(), first)

    def test_f3_control_equals_feat01_f3_model(self):
        from vkgr.retrieval_feat02 import parameter_report
        from vkgr.retrieval_features import make_feature_model
        from vkgr.training import seed_all
        seed_all(42); old = make_feature_model('f3_watch_buckets', CONTENT, SUPPORTED, 50)
        new = build('f3')
        self.assertEqual(list(new.state_dict()), list(old.state_dict()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(new.state_dict()[key], value), key)
        histories, side = model_inputs()
        self.assertTrue(torch.equal(new.users(histories, side), old.users(histories, side[..., :3])))
        index = torch.arange(1, 300)
        self.assertTrue(torch.equal(new.items(index), old.items(index)))
        report = parameter_report(new)
        self.assertEqual((report['total'], report['trainable_effective']), (896, 768))
        self.assertEqual(sum(p.numel() for n, p in old.named_parameters() if n != 'embedding.weight'), 896)

    def test_new_tables_do_not_disturb_anchor_draws(self):
        from vkgr.retrieval_feat02 import FEATURES
        from vkgr.retrieval_hp import make_model
        from vkgr.training import seed_all
        seed_all(42); anchor = make_model('id_mean', 'small', CONTENT, SUPPORTED, 50)
        f3 = build('f3')
        self.assertTrue(torch.equal(f3.embedding.weight, anchor.embedding.weight))
        for feature in FEATURES:
            model = build(feature)
            for name in ('embedding.weight', 'timespent_bucket.weight', 'ratio_bucket.weight'):
                self.assertTrue(torch.equal(dict(model.named_parameters())[name], dict(f3.named_parameters())[name]), (feature, name))
            self.assertEqual(model.init_report['init'], 'small')
            for table, entry in model.init_report['embeddings'].items():
                if table != 'embedding':
                    self.assertTrue(entry['pad_row_zero'], (feature, table))

    def test_unused_side_columns_do_not_affect_other_variants(self):
        from vkgr.retrieval_feat02 import FEATURES
        histories, side = model_inputs(); valid = histories != 0
        g = torch.Generator().manual_seed(5)
        for feature in FEATURES:
            model = build(feature); base = model.users(histories, side)
            for column in set(range(6)) - USED_COLUMNS[feature]:
                other = side.clone(); high = 64 if column == 0 else 25 if column == 3 else 256
                other[..., column] = torch.where(valid, torch.randint(0, high, valid.shape, generator=g), torch.zeros_like(side[..., column]))
                self.assertTrue(torch.equal(model.users(histories, other), base), (feature, column))
            perturb = {1: lambda c: 256 - c, 2: lambda c: 256 - c, 3: lambda c: torch.zeros_like(c), 4: lambda c: 255 - c, 5: lambda c: 255 - c}   # keep values in range / monotone
            for column in USED_COLUMNS[feature]:             # control: the columns a variant does use do reach the output
                other = side.clone(); other[..., column] = torch.where(valid, perturb[column](side[..., column]), torch.zeros_like(side[..., column]))
                self.assertFalse(torch.equal(model.users(histories, other), base), (feature, column))

    def test_param_report_matches_formula(self):
        from vkgr.retrieval_feat02 import FEATURES, expected_param_counts, parameter_report
        literal = {'f3': (896, 768), 'f3_masked': (896, 768), 'f3_f4_recency': (896 + 960, 768 + 832),
                   'f3_f5_scene': (896 + 26 * 64 + 13 * 64, 768 + 24 * 64 + 11 * 64)}
        for feature in FEATURES:
            report = parameter_report(build(feature))
            self.assertEqual(report, expected_param_counts(feature), feature)
            self.assertEqual((report['total'], report['trainable_effective']), literal[feature], feature)
            self.assertEqual(sum(t['total'] for t in report['by_table'].values()), report['total'])

    def test_load_state_dict_rejects_changed_supported_buffer(self):
        model = build('f3_f4_recency'); state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        state['supported'][1] = ~state['supported'][1]
        with self.assertRaises(ValueError):
            model.load_state_dict(state)

    def test_bad_side_is_rejected(self):
        from vkgr.retrieval_feat02 import FEATURES
        histories, side = model_inputs(); valid = histories != 0
        for feature in FEATURES:
            model = build(feature)
            with self.assertRaises(ValueError):
                model.users(histories, None)
            with self.assertRaises(ValueError):
                model.users(histories, side[..., :3])                       # old 3-column side
            with self.assertRaises(ValueError):
                model.users(histories, side.float())
            pad = side.clone(); slot = (~valid).nonzero()[0]; pad[slot[0], slot[1], 1] = 5
            with self.assertRaises(ValueError):
                model.users(histories, pad)                                 # PAD slot carries values
            extra = ((4, 256), (4, -1), (5, 256), (5, -1)) if feature == 'f3_f5_scene' else ()
            for column, value in ((0, 64), (1, 0), (1, 256), (2, 0), (2, 256), *extra):
                bad = side.clone(); bad[..., column] = torch.where(valid, torch.full_like(bad[..., column], value), torch.zeros_like(bad[..., column]))
                with self.assertRaises(ValueError, msg=(feature, column, value)):
                    model.users(histories, bad)
            with self.assertRaises(ValueError):
                model.users(torch.zeros_like(histories), torch.zeros_like(side))   # empty history

    def test_make_model_rejects_unregistered_feature_and_length(self):
        from vkgr.retrieval_feat02 import make_feat02_model
        with self.assertRaises(ValueError):
            make_feat02_model('f3_watch_buckets', CONTENT, SUPPORTED, 50)
        with self.assertRaises(ValueError):
            make_feat02_model('f3', CONTENT, SUPPORTED, 30)


class F3MaskedTests(unittest.TestCase):
    def test_f3_masked_adds_buckets_on_supported_events_only(self):
        guard_masked_adds_on_supported_only()

    def test_f3_masked_equals_f3_on_all_supported_history(self):
        histories, side = model_inputs(); histories = all_supported(histories)
        self.assertTrue(torch.equal(build('f3_masked').users(histories, side), build('f3').users(histories, side)))

    def test_f3_masked_equals_anchor_on_all_unsupported_history(self):
        from vkgr.retrieval_hp import make_model
        from vkgr.training import seed_all
        histories, side = model_inputs(); histories = all_unsupported(histories)
        seed_all(42); anchor = make_model('id_mean', 'small', CONTENT, SUPPORTED, 50)
        masked = build('f3_masked').users(histories, side)
        self.assertTrue(torch.equal(masked, anchor.users(histories)))
        self.assertFalse(bool(masked.any()), 'plain id_mean of an all-unsupported history is the zero vector')
        self.assertTrue(bool(build('f3').users(histories, side).any()), 'F3 itself is not zero there')

    def test_f3_masked_differs_from_f3_exactly_on_unsupported_events(self):
        guard_masked_differs_from_f3_exactly_on_unsupported()

    def test_f3_masked_param_count_equals_f3(self):
        from vkgr.retrieval_feat02 import parameter_report
        masked, f3 = parameter_report(build('f3_masked')), parameter_report(build('f3'))
        self.assertEqual(masked, f3)
        self.assertEqual((masked['total'], masked['trainable_effective']), (896, 768))


class F4RecencyTests(unittest.TestCase):
    def test_position_bucket_boundaries(self):
        guard_position_bucket_boundaries()

    def test_weekgap_bucket_boundaries(self):
        guard_weekgap_bucket_boundaries()

    def test_week_gap_anchored_on_last_history_event(self):
        guard_week_gap_anchored_on_last_history_event()

    def test_f4_formula(self):
        guard_f4_formula()

    def test_f4_changing_history_week_changes_feature(self):
        guard_changing_history_week_changes_feature()

    def test_f4_fail_closed_on_bad_weeks(self):
        histories, side = model_inputs(); valid = histories != 0
        f4 = build('f3_f4_recency'); f3 = build('f3')
        model_ok = f4.users(histories, side)
        self.assertEqual(tuple(model_ok.shape), (6, 64))
        late = side.clone(); late[0, 0, 3] = 25
        with self.assertRaises(ValueError):
            f4.users(histories, late)                                    # week 25 is validation territory
        negative = side.clone(); negative[0, 0, 3] = -1
        with self.assertRaises(ValueError):
            f4.users(histories, negative)
        decreasing = side.clone(); decreasing[0, 3, 3] = 0; decreasing[0, 2, 3] = 5
        with self.assertRaises(ValueError):
            f4.users(histories, decreasing)
        # the week column is not read by the other variants, so it is not validated there either
        f3.users(histories, late); f3.users(histories, decreasing)
        build('f3_masked').users(histories, decreasing)

    def test_f4_position_term_depends_only_on_history_length(self):
        from vkgr.retrieval_feat02 import position_bucket
        model = build('f3_f4_recency')
        histories = torch.tensor([[1, 2, 3, 4, 5, 6], [90, 91, 92, 93, 94, 95], [7, 8, 9, 0, 0, 0]]); valid = histories != 0
        term = model.position_table(position_bucket(valid)).sum(1) / valid.sum(1, keepdim=True)
        self.assertTrue(torch.equal(term[0], term[1]))
        table = model.position_table.weight
        self.assertTrue(torch.allclose(term[2], (table[1] + table[2] + table[3]) / 3, atol=1e-7, rtol=0))

    def test_f4_param_count(self):
        from vkgr.retrieval_feat02 import parameter_report
        model = build('f3_f4_recency'); report = parameter_report(model)
        self.assertEqual(tuple(model.position_table.weight.shape), (8, 64))
        self.assertEqual(tuple(model.weekgap_table.weight.shape), (7, 64))
        self.assertTrue(torch.all(model.position_table.weight[0] == 0) and torch.all(model.weekgap_table.weight[0] == 0))
        self.assertEqual({k: (v['total'], v['trainable']) for k, v in report['by_table'].items()},
                         {'timespent_bucket': (512, 448), 'ratio_bucket': (384, 320), 'position_table': (512, 448), 'weekgap_table': (448, 384)})
        self.assertEqual((report['total'], report['trainable_effective']), (1856, 1600))


class F5SceneTests(unittest.TestCase):
    def test_f5_formula(self):
        guard_f5_formula()

    def test_f5_unseen_value_maps_to_oov_and_contributes_zero(self):
        guard_f5_unseen_value_maps_to_oov_and_contributes_zero()

    def test_f5_pad_contributes_zero_and_agent_column_is_not_in_the_side_tensor(self):
        from vkgr.retrieval_feat02 import SIDE_WIDTH
        model = build('f3_f5_scene'); histories, side = scene_inputs()
        self.assertEqual(SIDE_WIDTH, 6)                                  # [mask, timespent, duration, week, place, platform]: no agent column
        self.assertTrue(bool((histories == 0).any()), 'fixture needs PAD slots')
        base = model.users(histories, side)
        with torch.no_grad():
            model.place_table.weight[0] = 5.; model.platform_table.weight[0] = 5.
        self.assertTrue(torch.equal(model.users(histories, side), base), 'PAD slots must contribute exactly zero even if the PAD row were non-zero')

    def test_f5_tables_sized_from_hardcoded_values(self):
        from vkgr import retrieval_feat02 as m
        model = build('f3_f5_scene')
        self.assertEqual(tuple(model.place_table.weight.shape), (len(AUDIT_PLACE_VALUES) + 2, 64))
        self.assertEqual(tuple(model.platform_table.weight.shape), (len(AUDIT_PLATFORM_VALUES) + 2, 64))
        self.assertEqual((m.PLACE_VALUES, m.PLATFORM_VALUES), (AUDIT_PLACE_VALUES, AUDIT_PLATFORM_VALUES))
        self.assertTrue(torch.all(model.place_table.weight[0] == 0) and torch.all(model.platform_table.weight[0] == 0))
        self.assertEqual(model.place_table.padding_idx, 0)
        first = m.bucket_spec_sha256()
        with mock.patch.object(m, 'PLACE_VALUES', tuple(range(30))):
            self.assertNotEqual(m.bucket_spec_sha256(), first)
            self.assertEqual(tuple(build('f3_f5_scene').place_table.weight.shape), (32, 64))     # sized from the constant, not a literal
        with mock.patch.object(m, 'PLATFORM_VALUES', tuple(range(12))):
            self.assertNotEqual(m.bucket_spec_sha256(), first)
            self.assertEqual(tuple(build('f3_f5_scene').platform_table.weight.shape), (14, 64))

    def test_f5_param_count(self):
        from vkgr.retrieval_feat02 import expected_param_counts, parameter_report
        report = parameter_report(build('f3_f5_scene'))
        self.assertEqual({k: (v['total'], v['trainable']) for k, v in report['by_table'].items()},
                         {'timespent_bucket': (512, 448), 'ratio_bucket': (384, 320), 'place_table': (26 * 64, 24 * 64), 'platform_table': (13 * 64, 11 * 64)})
        own = report['by_table']['place_table']['total'] + report['by_table']['platform_table']['total']
        self.assertEqual(own, (len(AUDIT_PLACE_VALUES) + 2) * 64 + (len(AUDIT_PLATFORM_VALUES) + 2) * 64)
        self.assertEqual((report['total'], report['trainable_effective']), (896 + 2496, 768 + 2240))
        self.assertEqual(report, expected_param_counts('f3_f5_scene'))

    def test_f5_leaves_f3_and_anchor_draws_untouched(self):
        f3, f5 = build('f3'), build('f3_f5_scene')
        for name in ('embedding.weight', 'timespent_bucket.weight', 'ratio_bucket.weight'):
            self.assertTrue(torch.equal(dict(f5.named_parameters())[name], dict(f3.named_parameters())[name]), name)

    def test_f5_category_ids_cover_every_uint8_value(self):
        from vkgr.retrieval_feat02 import PLACE_VALUES, PLATFORM_VALUES, category_ids
        valid = torch.ones(1, 256, dtype=torch.bool); raw = torch.arange(256).unsqueeze(0)
        for values in (PLACE_VALUES, PLATFORM_VALUES):
            self.assertEqual(category_ids(raw, values, valid)[0].tolist(), [reference_category_id(v, values) for v in range(256)])


if __name__ == '__main__':
    unittest.main()
