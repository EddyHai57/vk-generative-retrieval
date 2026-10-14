"""GFEAT-01: FEAT-01 F2/F3 side features in G's encoder (new entry only).

Run inside the INFRA-01 6.1 frozen-base harness with OMP_NUM_THREADS=1 (bare ``from test_generative_b1 import ...``
needs ``discover -s tests``). Classes marked "server-safe" do not touch the G1 engineering contract.
"""
import copy
import hashlib
import itertools
import math
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src' / 'vkgr'
P_COMMIT = 'd70a993a800aa4f37a9db55d6133383b95436e35'
BASE_COMMIT = '2c8882b4084a41d2862332a794477ba1d812fae3'
HISTORY_V2_SHA = '1c8d8718ab7527524df9d47dfbb518ee7f5b419ca34dfbb8971aca7efc31f06e'
VENDORED = {
    'history_side': '8faf06bb4dead2a83311bec1edfc84aaea1959fbbbde88871a1697de29cdea73',
    'retrieval_features': 'acc60664458a439823daa99edf1b9e44348ab6ca8783dd284a595404856878fa',
    'retrieval_hp': 'fcc5e8ac18f5fb9278556414de331a1c19ccca7432ec8282cbc657bdc0febadc',
    'retrieval_sasrec': '301b7086af2a2ada58e28c4440d18c880bdd50ddf7ccafc19fd470dcfb80f00c',
    'retrieval_models': 'a482cfde4ec0b6bf5525fa51d3bf226d8c10a4dc4aac785c1440721ee68f30f2',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob(commit, rel):
    """Bytes of ``rel`` at ``commit``; None when this checkout's object store lacks the commit."""
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{rel}'], capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return None


# Independent reference of the frozen card (FEAT-01 section 4): written from the card's tables, not from the P code.
CARD_TIMESPENT_BUCKETS = ((1, 3, 1), (4, 9, 2), (10, 30, 3), (31, 60, 4), (61, 134, 5), (135, 254, 6), (255, 255, 7))


def card_timespent_bucket(ts):
    for low, high, bucket in CARD_TIMESPENT_BUCKETS:
        if low <= ts <= high:
            return bucket
    raise AssertionError(ts)


def card_ratio_bucket(ts, dur):
    """Integer comparisons exactly as the card table: r<0.1 | [0.1,0.5) | [0.5,1) | r=1 | r>1."""
    if 10 * ts < dur:
        return 1
    if 2 * ts < dur:
        return 2
    if ts < dur:
        return 3
    if ts == dur:
        return 4
    return 5


class SharedDefinitionTests(unittest.TestCase):
    """server-safe. The feature definitions are ONE source shared with the P line (feat01/p d70a993)."""

    def test_history_v2_identical_to_p_line(self):
        self.assertEqual(sha(SRC / 'history_v2.py'), HISTORY_V2_SHA)
        for commit in (P_COMMIT, BASE_COMMIT):
            blob = git_blob(commit, 'src/vkgr/history_v2.py')
            if blob is not None:
                self.assertEqual((SRC / 'history_v2.py').read_bytes(), blob, commit)

    def test_vendored_modules_identical_to_p_d70a993(self):
        for name, expected in VENDORED.items():
            with self.subTest(module=name):
                self.assertEqual(sha(SRC / f'{name}.py'), expected)
                blob = git_blob(P_COMMIT, f'src/vkgr/{name}.py')
                if blob is not None:
                    self.assertEqual((SRC / f'{name}.py').read_bytes(), blob)

    def test_bucket_tables_match_frozen_card(self):
        from vkgr import history_side
        from vkgr import retrieval_features as rf
        self.assertEqual(rf.TIMESPENT_EDGES, (4, 10, 31, 61, 135, 255))
        self.assertEqual((rf.TIMESPENT_BUCKETS, rf.RATIO_BUCKETS, rf.FEEDBACK_DIM), (7, 5, 6))
        self.assertEqual(history_side.FEEDBACK_BITS, ('like', 'share', 'bookmark', 'open_comments', 'click_on_author', 'dislike'))
        ts = torch.arange(1, 256)
        valid = torch.ones_like(ts, dtype=torch.bool)
        self.assertEqual(rf.timespent_bucket(ts, valid).tolist(), [card_timespent_bucket(v) for v in range(1, 256)])
        self.assertEqual(rf.timespent_bucket(ts, ~valid).tolist(), [0] * 255)
        grid_ts, grid_dur = torch.meshgrid(ts, ts, indexing='ij')
        ones = torch.ones_like(grid_ts, dtype=torch.bool)
        got = rf.ratio_bucket(grid_ts, grid_dur, ones).flatten().tolist()
        want = [card_ratio_bucket(a, b) for a in range(1, 256) for b in range(1, 256)]
        self.assertEqual(got, want)
        self.assertEqual(rf.ratio_bucket(grid_ts, grid_dur, ~ones).flatten().tolist(), [0] * (255 * 255))
        self.assertEqual(sorted(set(want)), [1, 2, 3, 4, 5])


def synthetic_codec(items=200, branching=8, seed=0):
    """Small SID catalog that needs no G1 contract (same construction as test_generative_batched.large_codec)."""
    import numpy as np
    from vkgr import generative_data as gd
    from vkgr.sid import SIDCatalog
    rng = np.random.RandomState(seed); ids = np.arange(10, 10 + items, dtype=np.uint32)
    prefixes = rng.randint(0, branching, size=(items, 3)).astype(np.uint32)
    order = np.lexsort((ids, prefixes[:, 2], prefixes[:, 1], prefixes[:, 0])); codes = prefixes[order]
    starts = np.r_[True, np.any(codes[1:] != codes[:-1], axis=1)]; rows = np.arange(items)
    sids = np.zeros((items, 4), dtype=np.uint32); sids[:, :3] = prefixes
    sids[order, 3] = (rows - np.maximum.accumulate(np.where(starts, rows, 0))).astype(np.uint32)
    return gd._ValidatedCodec(SIDCatalog(ids, sids, branching, 3, None))


def feat_config(feature, **overrides):
    from test_generative_b1 import config as b1_config
    value = b1_config(); value['feature'] = feature; value.update(overrides)
    return value


def model_inputs(codec, histories, sides):
    """(source, side) exactly as the training/decode paths build them: right-padded SID tokens + (B, items, 3) side."""
    from vkgr import generative_feat as gf
    rows = [codec.history(h) for h in histories]; width = max(map(len, rows))
    source = torch.tensor([r + [0] * (width - len(r)) for r in rows], dtype=torch.long)
    return source, gf.side_tensor(sides)


def valid_side(n, seed=0):
    rng = random.Random(seed)
    return [[rng.randrange(0, 64), rng.choice(TS_CYCLE), rng.choice(DURATION_CYCLE)] for _ in range(n)]


class ModelTests(unittest.TestCase):
    """server-safe: synthetic codec, no engineering contract."""

    def setUp(self):
        self.codec = synthetic_codec()
        self.items = [int(i) for i in self.codec.catalog.ids]

    def model(self, feature, seed=42, **overrides):
        from vkgr import generative_feat as gf
        torch.manual_seed(seed)
        return gf.make_model_feat(self.codec, feat_config(feature, **overrides), device='cpu')

    def anchor(self, seed=42, **overrides):
        from vkgr.generative_hp import make_model_b1
        torch.manual_seed(seed)
        return make_model_b1(self.codec, feat_config('none', **overrides), device='cpu')

    def batch(self, lengths=(3, 1, 5)):
        histories = [[self.items[(7 * b + 3 * k) % len(self.items)] for k in range(n)] for b, n in enumerate(lengths)]
        sides = [valid_side(n, seed=b) for b, n in enumerate(lengths)]
        return model_inputs(self.codec, histories, sides)

    def test_base_parameters_equal_anchor_under_same_seed(self):
        new_keys = {'f2_feedback': {'feedback.weight'}, 'f3_watch_buckets': {'timespent_bucket.weight', 'ratio_bucket.weight'}}
        anchor = self.anchor().state_dict()
        for feature, keys in new_keys.items():
            with self.subTest(feature=feature):
                state = self.model(feature).state_dict()
                self.assertEqual(set(state) - set(anchor), keys)
                self.assertTrue(set(anchor) <= set(state))
                for name in anchor:
                    self.assertTrue(torch.equal(anchor[name], state[name]), name)

    def test_none_is_plain_tiger_model(self):
        from vkgr.tiger import TigerModel
        model = self.model('none'); anchor = self.anchor()
        self.assertIs(type(model), TigerModel)
        self.assertEqual(model.init_report, anchor.init_report)
        for name, value in anchor.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)

    def test_parameter_counts_init_and_pad_rows(self):
        for d_model, ff in ((64, 256), (128, 512)):
            anchor = self.anchor(d_model=d_model, ff=ff); base = sum(p.numel() for p in anchor.parameters())
            f2 = self.model('f2_feedback', d_model=d_model, ff=ff); f3 = self.model('f3_watch_buckets', d_model=d_model, ff=ff)
            with self.subTest(d_model=d_model):
                self.assertEqual(sum(p.numel() for p in f2.parameters()) - base, 6 * d_model)
                self.assertEqual(sum(p.numel() for p in f3.parameters()) - base, (8 + 6) * d_model)
                self.assertEqual(tuple(f2.feedback.weight.shape), (6, d_model))
                self.assertEqual(tuple(f3.timespent_bucket.weight.shape), (8, d_model))
                self.assertEqual(tuple(f3.ratio_bucket.weight.shape), (6, d_model))
                # trainable = rows that are not PAD: (7 + 5) x d (the PAD rows are fixed zeros)
                self.assertEqual(sum(t.weight.numel() - d_model for t in (f3.timespent_bucket, f3.ratio_bucket)), 12 * d_model)
                for table in (f2.feedback, f3.timespent_bucket, f3.ratio_bucket):
                    rows = table.weight.detach() if table.padding_idx is None else table.weight.detach()[1:]
                    self.assertLess(abs(float(rows.std()) - 0.02), 0.006)
                for table in (f3.timespent_bucket, f3.ratio_bucket):
                    self.assertTrue(torch.equal(table.weight[0], torch.zeros(d_model)))
                self.assertIn('feedback', f2.init_report['embeddings'])
                self.assertEqual(set(f3.init_report['embeddings']) - set(anchor.init_report['embeddings']), {'timespent_bucket', 'ratio_bucket'})
                self.assertTrue(all(f3.init_report['embeddings'][n]['pad_row_zero'] for n in ('timespent_bucket', 'ratio_bucket')))

    def _expected_embedding(self, model, source, side, vector):
        """Hand-computed encoder input: token embedding + the item's side vector on each of its 4 tokens, x sqrt(d) + position."""
        tokens = model.embedding(source).detach().clone()
        for b in range(source.shape[0]):
            for k in range(source.shape[1] // 4):
                if int(source[b, 4 * k]) != 0:
                    for t in range(4):
                        tokens[b, 4 * k + t] = tokens[b, 4 * k + t] + vector(int(side[b, k, 0]), int(side[b, k, 1]), int(side[b, k, 2]))
        return tokens * (model.d_model ** 0.5) + model.position(torch.arange(source.shape[1])).detach()[None]

    def test_f2_multi_hot_broadcast_to_all_four_tokens(self):
        from vkgr.history_side import FEEDBACK_BITS
        model = self.model('f2_feedback'); source, side = self.batch()
        table = model.feedback.weight.detach()
        def vector(mask, ts, dur):
            out = torch.zeros(model.d_model)
            for bit in range(6):
                if mask >> bit & 1:
                    out = out + table[bit]
            return out
        expected = self._expected_embedding(model, source, side, vector)
        self.assertTrue(torch.allclose(model.embed_source(source, side).detach(), expected, atol=1e-6))
        for bit, name in enumerate(FEEDBACK_BITS):  # bit order = FEEDBACK_BITS
            with self.subTest(bit=name):
                one = torch.tensor([[[1 << bit, 5, 9]]]); src = torch.tensor([self.codec.history(self.items[:1])])
                got = model.embed_source(src, one).detach() - model.embed_source(src, torch.tensor([[[0, 5, 9]]])).detach()
                self.assertTrue(torch.allclose(got, (table[bit] * model.d_model ** 0.5).expand(1, 4, -1), atol=1e-6))
        # all four tokens of one item receive the SAME vector (broadcast, not first-token-only)
        same = model.embed_source(src, torch.tensor([[[63, 5, 9]]])).detach() - model.embed_source(src, torch.tensor([[[0, 5, 9]]])).detach()
        # (not torch.equal: the two operands are different float32 sums per token, so only the rounding noise differs)
        self.assertTrue(all(torch.allclose(same[0, 0], same[0, t], atol=1e-6) for t in range(1, 4)))
        self.assertGreater(float(same.abs().min()), 1e-3)

    def test_f3_two_bucket_embeddings_broadcast(self):
        model = self.model('f3_watch_buckets'); source, side = self.batch()
        tb = model.timespent_bucket.weight.detach(); rb = model.ratio_bucket.weight.detach()
        vector = lambda mask, ts, dur: tb[card_timespent_bucket(ts)] + rb[card_ratio_bucket(ts, dur)]
        expected = self._expected_embedding(model, source, side, vector)
        self.assertTrue(torch.allclose(model.embed_source(source, side).detach(), expected, atol=1e-6))
        # the feedback mask is NOT an input of F3
        other = side.clone(); other[..., 0] = torch.where(side[..., 0] != 0, (side[..., 0] + 1) % 64, side[..., 0])
        self.assertTrue(torch.equal(model.embed_source(source, side), model.embed_source(source, other)))

    def test_side_added_before_sqrt_d_scaling(self):
        model = self.model('f2_feedback'); src = torch.tensor([self.codec.history(self.items[:2])])
        with torch.no_grad():
            model.feedback.weight.zero_(); model.feedback.weight[0].fill_(0.5)
        zero = torch.tensor([[[0, 5, 9], [0, 5, 9]]]); one = torch.tensor([[[1, 5, 9], [0, 5, 9]]])
        diff = (model.embed_source(src, one) - model.embed_source(src, zero)).detach()
        self.assertTrue(torch.allclose(diff[0, :4], torch.full((4, model.d_model), 0.5 * model.d_model ** 0.5)))
        self.assertTrue(torch.equal(diff[0, 4:], torch.zeros(4, model.d_model)))

    def test_pad_items_and_bad_side_rejected(self):
        for feature in ('f2_feedback', 'f3_watch_buckets'):
            model = self.model(feature); source, side = self.batch((3, 1, 5))
            self.assertEqual(model.encode(source, side)[0].shape[:2], source.shape)  # the good input is accepted
            def bad(name, src=source, sd=side):
                with self.subTest(feature=feature, case=name), self.assertRaises(ValueError):
                    model.encode(src, sd)
            pad_side = side.clone(); pad_side[1, 2] = torch.tensor([1, 5, 9]); bad('PAD item carries side', sd=pad_side)
            half = source.clone(); half[0, 1] = 0; bad('half-PAD item', src=half)
            bad('side=None', sd=None)
            bad('wrong item count', sd=side[:, :-1]); bad('wrong batch', sd=side[:2]); bad('float dtype', sd=side.float())
            bad('wrong last dim', sd=torch.cat([side, side[..., :1]], -1))
            for col, value, name in ((0, 64, 'mask 64'), (0, -1, 'mask -1'), (1, 0, 'timespent 0'), (1, 256, 'timespent 256'), (2, 0, 'duration 0'), (2, 256, 'duration 256')):
                broken = side.clone(); broken[0, 0, col] = value; bad(name, sd=broken)
            bad('width not multiple of 4', src=source[:, :-1], sd=side)

    def test_encode_without_side_rejected(self):
        for feature in ('f2_feedback', 'f3_watch_buckets'):
            model = self.model(feature); source, _ = self.batch(); decoder = torch.ones(source.shape[0], 2, dtype=torch.long)
            with self.subTest(feature=feature):
                with self.assertRaises(ValueError):
                    model.encode(source)
                with self.assertRaises(ValueError):
                    model(source, decoder)

    def test_zero_tables_equal_plain_model_memory_and_side_is_per_sample(self):
        from vkgr.tiger import TigerModel
        for feature in ('f2_feedback', 'f3_watch_buckets'):
            model = self.model(feature); source, side = self.batch((3, 4, 2)); anchor = self.anchor()
            plain = TigerModel(self.codec.vocab_size, 64, 4, 1, 1, 256); plain.load_state_dict(anchor.state_dict())
            with self.subTest(feature=feature):
                with torch.no_grad():
                    for table in (getattr(model, n, None) for n in ('feedback', 'timespent_bucket', 'ratio_bucket')):
                        if table is not None:
                            table.weight.zero_()
                    plain_memory, _ = plain.encode(source)
                    self.assertTrue(torch.equal(model.encode(source, side)[0], plain_memory))  # zero tables == the anchor encoder, bit for bit
                    model2 = self.model(feature)  # same base weights, non-zero side tables
                    memory = model2.encode(source, side)[0]
                    self.assertFalse(torch.equal(memory, plain_memory))
                    old = side[1, 2].tolist()
                    changed = side.clone(); changed[1, 2] = torch.tensor([63 if old[0] != 63 else 0, 255 if old[1] != 255 else 1, old[2]])
                    after = model2.encode(source, changed)[0]
                    self.assertTrue(torch.equal(after[0], memory[0]) and torch.equal(after[2], memory[2]))  # other samples untouched
                    self.assertFalse(torch.equal(after[1], memory[1]))


class FeatureRegistryTests(unittest.TestCase):
    """server-safe."""

    def test_feature_registry_reuses_the_shared_definitions(self):
        from vkgr import generative_feat as gf
        from vkgr import history_side
        from vkgr import retrieval_features as rf
        self.assertEqual(gf.FEATURES, ('none', 'f2_feedback', 'f3_watch_buckets'))
        self.assertEqual(history_side.SIDE_VERSION, 'HIST_V2_SIDE_V1')
        # imported, not copied: the single shared source
        self.assertIs(gf.SIDE_VERSION, history_side.SIDE_VERSION)
        self.assertIs(gf.timespent_bucket, rf.timespent_bucket)
        self.assertIs(gf.ratio_bucket, rf.ratio_bucket)
        self.assertEqual((gf.FEEDBACK_DIM, gf.TIMESPENT_BUCKETS, gf.RATIO_BUCKETS), (6, 7, 5))


# ---------------------------------------------------------------------------------------------------------------------
# side-data fixture (local harness only: needs the G1 engineering contract)
# ---------------------------------------------------------------------------------------------------------------------
TS_CYCLE = (1, 3, 4, 9, 10, 30, 31, 60, 61, 134, 135, 254, 255)  # hits every timespent bucket boundary of the card
DURATION_CYCLE = (5, 9, 25, 60, 180)
EXTRA_FEEDBACK = ('share', 'bookmark', 'open_comments', 'click_on_author', 'dislike')


def tmpdir():
    return tempfile.TemporaryDirectory(dir='/private/tmp')


def side_world(tmp):
    """Engineering context + history-v2 + HIST_V2_SIDE_V1 built by the shared P producer.

    Same partitions as ``test_generative_b1.engineering_world`` with every raw row enriched by the six feedback bits and
    timespent; durations are an item property (single-valued per item)."""
    from vkgr import generative_real as gr
    from vkgr import history_side, history_v2
    tmp = Path(tmp)
    context, receipt = gr.prepare_engineering(ROOT, tmp / 'prepared')
    examples = context.examples.rows(list(range(len(context.examples))))
    likes = {6: (1, 10), 7: (1, 20), 8: (1, 10), 9: (2, 50), 10: (3, 10), 11: (3, 30), 12: (4, 20), 13: (4, 40), 14: (5, 50), 15: (6, 40), 16: (7, 30)}
    week0 = [{'user_id': 1 if k % 2 else 3, 'item_id': (10, 20, 30, 40, 50, 60)[k % 6], 'like': False} for k in range(6)]
    week0 += [{'user_id': u, 'item_id': i, 'like': True} for _, (u, i) in sorted(likes.items())]
    later = [(w, [{'user_id': (1, 2, 3, 4, 5, 6, 7)[k % 7], 'item_id': (10, 20, 30, 40, 50, 60)[(k * 5 + w) % 6], 'like': False} for k in range(200)]) for w in (1, 2)]
    partitions = [(0, week0)] + later
    rng = random.Random(7); cycle = itertools.cycle(TS_CYCLE)
    for _, rows in partitions:
        for row in rows:
            for name in EXTRA_FEEDBACK:
                row[name] = rng.random() < 0.25
            row['timespent'] = next(cycle)
    durations = {int(item): DURATION_CYCLE[k % len(DURATION_CYCLE)] for k, item in enumerate(sorted(context.catalog))}
    h2 = history_v2.write_prepared(tmp / 'hist', partitions, examples, context.queries, list(context.catalog), {'synthetic': True})
    side = history_side.write_side(tmp / 'side', tmp / 'hist' / 'manifest.json', h2['manifest_sha256'], partitions, durations, {'synthetic': True})
    return SimpleNamespace(context=context, receipt=receipt, partitions=partitions, durations=durations, examples=examples,
                           hist_manifest=tmp / 'hist' / 'manifest.json', hist_sha=h2['manifest_sha256'],
                           side_manifest=tmp / 'side' / 'manifest.json', side_sha=side['manifest_sha256'])


def raw_table(world):
    return {(week, k): row for week, rows in world.partitions for k, row in enumerate(rows)}


def expected_event(raw, durations, item):
    from vkgr import history_side
    return [sum(1 << b for b, name in enumerate(history_side.FEEDBACK_BITS) if raw[name]), raw['timespent'], durations[item]]


def v2_context(world, length=50):
    from vkgr import history_v2
    from vkgr.generative_data_v2 import SID_VERSIONS, build_v2_context
    loaded = history_v2.load_prepared(world.hist_manifest, world.hist_sha, length)
    return loaded, build_v2_context(world.context, loaded, length, SID_VERSIONS[0])


class SideDataTests(unittest.TestCase):
    def test_sidecar_rows_align_with_history_positions(self):
        """history position <-> sidecar row <-> raw row, for every registered length (single shared sidecar)."""
        from vkgr import history_side, history_v2
        with tmpdir() as tmp:
            w = side_world(tmp); table = raw_table(w); seen = 0
            for length in (100, 50, 20):
                hist = history_v2.load_prepared(w.hist_manifest, w.hist_sha, length)
                loaded = history_side.load_side(w.side_manifest, w.side_sha, w.hist_sha, length, hist)
                for split in ('train', 'validation'):
                    self.assertEqual(len(loaded[split]), len(hist[split]))
                    for row, events in zip(hist[split], loaded[split]):
                        self.assertEqual(len(events), len(row['history']))
                        for item, position, event in zip(row['history'], row['history_positions'], events):
                            raw = table[tuple(position)]
                            self.assertEqual(raw['item_id'], item)
                            self.assertEqual(event, expected_event(raw, w.durations, item)); seen += 1
            self.assertGreater(seen, 100)

    def test_target_and_later_event_fields_cannot_change_side(self):
        """Anti-leakage (data layer): flip the target row and every later row of that user -> that example's side is unchanged."""
        from vkgr import history_side, history_v2
        with tmpdir() as tmp:
            w = side_world(tmp)
            hist = history_v2.load_prepared(w.hist_manifest, w.hist_sha, 100)
            base = history_side.build_side(w.partitions, hist, w.durations); checked = 0
            for e in w.examples:
                mutated = copy.deepcopy(w.partitions); week, ordinal = e['target_position']; flipped = 0
                for wk, rows in mutated:
                    for k, row in enumerate(rows):
                        if row['user_id'] == e['user_id'] and (wk, k) >= (week, ordinal):
                            row.update(timespent=255 if row['timespent'] != 255 else 1, like=not row['like'], **{n: not row[n] for n in EXTRA_FEEDBACK}); flipped += 1
                self.assertGreater(flipped, 1)
                after = history_side.build_side(mutated, hist, w.durations)
                self.assertEqual(after['train'][e['example_id']], base['train'][e['example_id']])
                checked += 1
            self.assertGreaterEqual(checked, 3)
            # reverse control: changing one PAST event of that example must change its side
            e = next(x for x in w.examples if len(hist['train'][x['example_id']]['history_positions']) >= 2)
            week, ordinal = hist['train'][e['example_id']]['history_positions'][-1]
            mutated = copy.deepcopy(w.partitions); row = mutated[week][1][ordinal]
            row['timespent'] = 254 if row['timespent'] != 254 else 253
            self.assertNotEqual(history_side.build_side(mutated, hist, w.durations)['train'][e['example_id']], base['train'][e['example_id']])

    def test_validation_side_uses_train_weeks_only(self):
        from vkgr import history_side, history_v2
        with tmpdir() as tmp:
            w = side_world(tmp)
            hist = history_v2.load_prepared(w.hist_manifest, w.hist_sha, 100)
            with self.assertRaises(ValueError):
                history_side.build_side(w.partitions + [(25, w.partitions[0][1])], hist, w.durations)

    def test_side_binding_is_fail_closed(self):
        from vkgr import history_side, history_v2
        with tmpdir() as tmp:
            w = side_world(tmp)
            hist = history_v2.load_prepared(w.hist_manifest, w.hist_sha, 50)
            with self.assertRaises(ValueError):
                history_side.load_side(w.side_manifest, '0' * 64, w.hist_sha, 50, hist)
            with self.assertRaises(ValueError):
                history_side.load_side(w.side_manifest, w.side_sha, 'f' * 64, 50, hist)
            self.assertTrue(any(len(r['history']) > 1 for r in hist['train']))
            short = {**hist, 'train': [{**r, 'history': r['history'][:-1] or r['history']} for r in hist['train']]}
            with self.assertRaises(ValueError):
                history_side.load_side(w.side_manifest, w.side_sha, w.hist_sha, 50, short)

    def test_pack_aligns_with_context_rows_and_queries(self):
        """history-position <-> sidecar-row alignment on G's side: every example_id and every query ordinal."""
        from vkgr import generative_feat as gf
        with tmpdir() as tmp:
            w = side_world(tmp); table = raw_table(w)
            loaded, ctx = v2_context(w)
            pack = gf.load_side_pack(w.side_manifest, w.side_sha, w.hist_sha, loaded, ctx, 50)
            rows = ctx.examples.rows(list(range(len(ctx.examples)))); queries = ctx.queries
            self.assertEqual((len(pack.train), len(pack.validation)), (len(rows), len(queries)))
            self.assertGreaterEqual(len(rows), 3)
            for split, history_rows in (('train', rows), ('validation', queries)):
                for ordinal, row in enumerate(history_rows):
                    side = getattr(pack, split)[ordinal]
                    if split == 'train':
                        self.assertEqual(row['example_id'], ordinal)
                    self.assertEqual(len(side), len(row['history']))
                    for item, position, event in zip(row['history'], row['history_positions'], side):
                        raw = table[tuple(position)]
                        self.assertEqual(raw['item_id'], item)
                        self.assertEqual(list(event), expected_event(raw, w.durations, item))
            self.assertEqual(max(len(q['history']) for q in queries), 50)

    def test_pack_rejects_misaligned_or_invalid_side(self):
        from vkgr import generative_feat as gf
        from vkgr import history_side
        with tmpdir() as tmp:
            w = side_world(tmp)
            loaded, ctx = v2_context(w)
            side = history_side.load_side(w.side_manifest, w.side_sha, w.hist_sha, 50, loaded)
            build = lambda s: gf.build_side_pack(s, loaded, ctx, w.hist_sha, w.side_sha)
            build(side)  # untampered side is accepted
            long_train = next(i for i, r in enumerate(side['train']) if len(r) > 1)
            def edit(split, mutate):
                bad = copy.deepcopy(side); mutate(bad[split]); return bad
            cases = {
                'row shorter than its history': edit('train', lambda rows: rows.__setitem__(long_train, rows[long_train][:-1])),
                'row longer than its history': edit('validation', lambda rows: rows[0].append(list(rows[0][0]))),
                'missing validation row': edit('validation', lambda rows: rows.pop()),
                'feedback mask 64': edit('train', lambda rows: rows[long_train][0].__setitem__(0, 64)),
                'feedback mask -1': edit('validation', lambda rows: rows[0][0].__setitem__(0, -1)),
                'timespent 0': edit('validation', lambda rows: rows[0][0].__setitem__(1, 0)),
                'timespent 256': edit('train', lambda rows: rows[long_train][0].__setitem__(1, 256)),
                'duration 0': edit('validation', lambda rows: rows[0][0].__setitem__(2, 0)),
                'duration 256': edit('train', lambda rows: rows[long_train][0].__setitem__(2, 256)),
                'bool mask': edit('validation', lambda rows: rows[0][0].__setitem__(0, True)),
            }
            for name, bad in cases.items():
                with self.subTest(case=name), self.assertRaises(ValueError):
                    build(bad)

    def test_pack_rejects_swapped_rows_via_duration_guard(self):
        """D9: duration is an item property. Swapping two same-length rows whose durations differ is caught even though
        every length / range check still passes."""
        from vkgr import generative_feat as gf
        from vkgr import history_side
        with tmpdir() as tmp:
            w = side_world(tmp)
            loaded, ctx = v2_context(w)
            side = history_side.load_side(w.side_manifest, w.side_sha, w.hist_sha, 50, loaded)
            build = lambda s: gf.build_side_pack(s, loaded, ctx, w.hist_sha, w.side_sha)
            for split, history_rows in (('validation', ctx.queries), ('train', ctx.examples.rows(list(range(len(ctx.examples)))))):
                pairs = [(a, b) for a in range(len(history_rows)) for b in range(a + 1, len(history_rows))
                         if len(history_rows[a]['history']) == len(history_rows[b]['history'])
                         and [e[2] for e in side[split][a]] != [e[2] for e in side[split][b]]]
                with self.subTest(split=split):
                    self.assertTrue(pairs, 'fixture must contain a swappable same-length pair')
                    a, b = pairs[0]
                    bad = copy.deepcopy(side); bad[split][a], bad[split][b] = bad[split][b], bad[split][a]
                    with self.assertRaises(ValueError):
                        build(bad)

    def test_pack_binding_and_attached_context(self):
        from vkgr import generative_feat as gf
        from vkgr import history_side
        from vkgr.data_audit import canonical_bytes
        with tmpdir() as tmp:
            w = side_world(tmp)
            loaded, ctx = v2_context(w)
            side = history_side.load_side(w.side_manifest, w.side_sha, w.hist_sha, 50, loaded)
            pack = gf.load_side_pack(w.side_manifest, w.side_sha, w.hist_sha, loaded, ctx, 50)
            self.assertEqual(pack.binding, {'side_version': 'HIST_V2_SIDE_V1', 'side_manifest_sha256': w.side_sha, 'history_manifest_sha256': w.hist_sha,
                                            'train_side_sha256': history_side.side_digest(side['train']),
                                            'validation_side_sha256': history_side.side_digest(side['validation'])})
            attached = gf.attach_side(ctx, pack)
            self.assertEqual(attached.binding['feat01'], pack.binding)
            self.assertEqual({k: v for k, v in attached.binding.items() if k != 'feat01'}, ctx.binding)
            self.assertNotIn('feat01', ctx.binding)
            # D6: side lives outside the queries/examples, so every query_sha256 stays the anchor's
            self.assertEqual(canonical_bytes(attached.queries), canonical_bytes(ctx.queries))
            self.assertEqual(attached.binding['queries_sha256'], ctx.binding['queries_sha256'])
            self.assertEqual(attached.examples.rows([0]), ctx.examples.rows([0]))
            self.assertIs(attached.codec, ctx.codec)


def flip_outcomes(raw):
    """Change every outcome field of a raw interaction row (what a leaking feature could read)."""
    raw.update(timespent=255 if raw['timespent'] != 255 else 1, like=not raw['like'], **{n: not raw[n] for n in EXTRA_FEEDBACK})


class TrainStepTests(unittest.TestCase):
    """Local harness only (engineering world rows). One world per class: every test below only reads it."""

    @classmethod
    def setUpClass(cls):
        from vkgr import generative_feat as gf
        cls._tmp = tmpdir(); cls.addClassCleanup(cls._tmp.cleanup)
        cls.world = side_world(cls._tmp.name)
        cls.loaded, cls.ctx = v2_context(cls.world)
        cls.pack = gf.load_side_pack(cls.world.side_manifest, cls.world.side_sha, cls.world.hist_sha, cls.loaded, cls.ctx, 50)
        cls.rows = cls.ctx.examples.rows(list(range(len(cls.ctx.examples))))

    def start(self, feature, seed=42):
        from vkgr import generative_feat as gf
        from vkgr import generative_training as gt
        from vkgr.generative_hp import core_config
        cfg = feat_config(feature); torch.manual_seed(seed)
        model = gf.make_model_feat(self.ctx.codec, cfg, device='cpu')
        return cfg, model, gt.make_optimizer(model, core_config(cfg)), gf.SIDTokensSide(self.ctx.codec, 50, self.pack)

    def pack_from(self, partitions):
        """Rebuild the sidecar from (possibly mutated) raw rows with the shared P producer and wrap it like the real loader does."""
        from vkgr import generative_feat as gf
        from vkgr import history_side, history_v2
        w = self.world
        side = history_side.build_side(partitions, history_v2.load_prepared(w.hist_manifest, w.hist_sha, 100), w.durations)
        return gf.build_side_pack({split: [r[-50:] for r in rows] for split, rows in side.items()}, self.loaded, self.ctx, w.hist_sha, w.side_sha)

    def test_side_tables_get_gradients_and_pad_rows_stay_zero(self):
        from vkgr import generative_feat as gf
        from vkgr import generative_training as gt
        from vkgr.generative_data import sid_targets
        for feature in ('f2_feedback', 'f3_watch_buckets'):
            with self.subTest(feature=feature):
                cfg, model, optimizer, tokens = self.start(feature)
                tables = [getattr(model, n) for n in ('feedback', 'timespent_bucket', 'ratio_bucket') if hasattr(model, n)]
                self.assertEqual(len(tables), 1 if feature == 'f2_feedback' else 2)
                in_optimizer = {id(p) for g in optimizer.param_groups for p in g['params']}
                self.assertTrue(all(id(t.weight) in in_optimizer for t in tables))
                source, decoder, labels = sid_targets(self.rows, self.ctx.codec)
                gt.full_vocab_ce(model(source, decoder, tokens.train_side(self.rows)), labels).backward()
                for t in tables:  # the side tables really sit on the loss path; PAD rows never receive a gradient
                    self.assertGreater(float(t.weight.grad.abs().sum()), 0.0)
                    if t.padding_idx is not None:
                        self.assertEqual(float(t.weight.grad[t.padding_idx].abs().sum()), 0.0)
                model.zero_grad(set_to_none=True)
                before = [t.weight.detach().clone() for t in tables]
                result = gf.train_batch_feat(model, optimizer, self.rows, self.ctx, tokens, cfg, 'cpu')
                self.assertEqual((result.examples, result.optimizer_steps), (len(self.rows), 1))
                self.assertEqual(result.microbatch_sizes, (2,) * (len(self.rows) // 2) + ((len(self.rows) % 2,) if len(self.rows) % 2 else ()))
                self.assertTrue(math.isfinite(result.loss_mean))
                for t, old in zip(tables, before):
                    self.assertFalse(torch.equal(t.weight, old))
                    if t.padding_idx is not None:
                        self.assertTrue(torch.equal(t.weight[t.padding_idx], torch.zeros(t.weight.shape[1])))

    def test_side_tokenizer_layout(self):
        from vkgr.generative_data import sid_targets
        _, _, _, tokens = self.start('f2_feedback')
        source, _, _ = sid_targets(self.rows, self.ctx.codec); side = tokens.train_side(self.rows)
        self.assertEqual((side.dtype, side.shape[0], side.shape[2]), (torch.long, len(self.rows), 3))
        self.assertEqual(source.shape[1], 4 * side.shape[1])
        for b, row in enumerate(self.rows):
            n = len(row['history'])
            for k, item in enumerate(row['history']):
                self.assertEqual(source[b, 4 * k:4 * k + 4].tolist(), self.ctx.codec.item(item))
                self.assertEqual(side[b, k].tolist(), list(self.pack.train[row['example_id']][k]))
            self.assertTrue(bool((source[b, 4 * n:] == 0).all()) and bool((side[b, n:] == 0).all()))  # right-padded together
        self.assertGreater(len({tuple(map(tuple, self.pack.train[r['example_id']])) for r in self.rows}), 1)  # rows really differ
        self.assertEqual(tokens.history(self.rows[0]['history']), self.ctx.codec.history(self.rows[0]['history']))  # plain SIDTokensL behaviour kept

    def test_zero_frozen_side_steps_equal_anchor_steps(self):
        """With the side tables zero and frozen, f2/f3 training IS the anchor training: same loss, same weights, bit for bit."""
        from vkgr import generative_feat as gf
        from vkgr import generative_training as gt
        from vkgr.generative_hp import core_config, make_model_b1
        for feature in ('f2_feedback', 'f3_watch_buckets'):
            with self.subTest(feature=feature):
                cfg, model, _, tokens = self.start(feature)
                for name in ('feedback', 'timespent_bucket', 'ratio_bucket'):
                    table = getattr(model, name, None)
                    if table is not None:
                        with torch.no_grad():
                            table.weight.zero_()
                        table.weight.requires_grad_(False)
                optimizer = gt.make_optimizer(model, core_config(cfg))
                none_cfg = feat_config('none'); torch.manual_seed(42)
                plain = make_model_b1(self.ctx.codec, none_cfg, device='cpu'); plain_optimizer = gt.make_optimizer(plain, core_config(none_cfg))
                for rows in (self.rows, self.rows[:-1], self.rows[1:]):
                    got = gf.train_batch_feat(model, optimizer, rows, self.ctx, tokens, cfg, 'cpu')
                    want = gt.train_effective_batch(plain, plain_optimizer, rows, self.ctx.codec, microbatch=cfg['microbatch'], device='cpu')
                    self.assertEqual(got, want)
                    state = model.state_dict()
                    for name, value in plain.state_dict().items():
                        self.assertTrue(torch.equal(value, state[name]), name)

    def test_encoder_input_independent_of_target_row_fields(self):
        """Anti-leakage (model layer): the raw target row and every later row of the user are flipped, the sidecar is REBUILT from the
        flipped rows, and that example's encoder memory and training loss stay bit-identical. Reverse control: a past event changes both."""
        import copy
        from vkgr import generative_feat as gf
        from vkgr import generative_training as gt
        from vkgr.generative_data import sid_targets
        from vkgr.generative_hp import core_config
        self.assertEqual(self.pack_from(self.world.partitions).train, self.pack.train)  # the rebuild helper reproduces the real pack
        codec = self.ctx.codec
        def measure(feature, cfg, model, pack, row):
            tokens = gf.SIDTokensSide(codec, 50, pack)
            memory = model.encode(sid_targets([row], codec)[0], tokens.train_side([row]))[0].detach()
            clone = copy.deepcopy(model)
            loss = gf.train_batch_feat(clone, gt.make_optimizer(clone, core_config(cfg)), [row], self.ctx, tokens, cfg, 'cpu').loss_mean
            return memory, loss
        for feature in ('f2_feedback', 'f3_watch_buckets'):
            cfg, model, _, _ = self.start(feature)
            for e in self.world.examples:
                row = self.rows[e['example_id']]; week, ordinal = e['target_position']
                mutated = copy.deepcopy(self.world.partitions)
                for wk, part in mutated:
                    for k, raw in enumerate(part):
                        if raw['user_id'] == e['user_id'] and (wk, k) >= (week, ordinal):
                            flip_outcomes(raw)
                (m1, l1), (m2, l2) = measure(feature, cfg, model, self.pack, row), measure(feature, cfg, model, self.pack_from(mutated), row)
                with self.subTest(feature=feature, example=e['example_id']):
                    self.assertTrue(torch.equal(m1, m2)); self.assertEqual(l1, l2)
            e = next(x for x in self.world.examples if len(self.loaded['train'][x['example_id']]['history_positions']) >= 2)
            row = self.rows[e['example_id']]; week, ordinal = self.loaded['train'][e['example_id']]['history_positions'][-1]
            mutated = copy.deepcopy(self.world.partitions); flip_outcomes(dict(mutated)[week][ordinal])
            (m1, l1), (m3, l3) = measure(feature, cfg, model, self.pack, row), measure(feature, cfg, model, self.pack_from(mutated), row)
            with self.subTest(feature=feature, control='past event'):
                self.assertFalse(torch.equal(m1, m3)); self.assertNotEqual(l1, l3)

    def test_train_rows_misaligned_side_rejected(self):
        from vkgr import generative_feat as gf
        cfg, model, optimizer, tokens = self.start('f2_feedback')
        row = next(r for r in self.rows if len(r['history']) > 1)
        cases = {'history shorter than its side row': {**row, 'history': row['history'][:-1]},
                 'missing example_id': {k: v for k, v in row.items() if k != 'example_id'},
                 'example_id past the pack': {**row, 'example_id': len(self.pack.train)},
                 'bool example_id': {**row, 'example_id': True}}
        before = {k: v.clone() for k, v in model.state_dict().items()}
        for name, bad in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                gf.train_batch_feat(model, optimizer, [bad], self.ctx, tokens, cfg, 'cpu')
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))  # a rejected batch never updates anything


if __name__ == '__main__':
    unittest.main()
