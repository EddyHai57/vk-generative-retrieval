"""G6-PREP step 2: depth-generic codec (check, tokens, validated codec, suffix assignment).

Depth 3 must agree with the frozen depth-3 code on every input (same accept/reject, same message, same values);
depth 4 is the new capability. Mutants M1..M7 live in tests/test_sid_depth_mutations.py.
"""
import unittest

import numpy as np
import torch

from tests import g6_fixtures as fx
from vkgr import generative_data as gd
from vkgr import sid_depth as sd
from vkgr import sid_rqvae as rq
from vkgr.generative_data_v2 import SIDTokensL
from vkgr.sid import SIDCatalog
from vkgr.tiger import SIDTokens


def outcome(check, *args):
    try:
        check(*args)
    except Exception as exc:  # noqa: BLE001 - the exact type and message are the contract
        return (type(exc).__name__, str(exc))
    return ('ok',)


def tampered(depth, kind, branching=4):
    """A codec that violates exactly one rule of the codec contract; ``kind`` names the rule."""
    ids, sids = fx.synth(160, branching, depth, seed=3)
    leaf = fx.collision_leaf(sids, depth)
    if kind == 'ids_unsorted':
        ids = ids.copy(); ids[[0, 1]] = ids[[1, 0]]
    elif kind == 'ids_dtype':
        ids = ids.astype(np.int64)
    elif kind == 'sids_dtype':
        sids = sids.astype(np.int64)
    elif kind == 'digit_out_of_range':
        sids = sids.copy(); sids[0, 1] = branching
    elif kind == 'last_semantic_digit_out_of_range':
        sids = sids.copy(); sids[0, depth - 1] = branching
    elif kind == 'noncanonical_suffix':
        sids = sids.copy(); sids[[leaf[0], leaf[1]], depth] = sids[[leaf[1], leaf[0]], depth]
    elif kind == 'suffix_shifted':
        sids = sids.copy(); sids[leaf, depth] += 1
    elif kind == 'duplicate_sid':
        sids = sids.copy(); sids[leaf[1], depth] = sids[leaf[0], depth]
    codec = fx.tokens_for(SIDCatalog(ids, sids, branching, depth, []))
    if kind == 'offsets':
        codec.offsets[1] += 1
    elif kind == 'vocab':
        codec.vocab_size += 1
    elif kind == 'bases':
        codec.catalog.bases[0] += 1
    elif kind == 'depth_attr':
        codec.catalog.depth = depth - 1
    elif kind == 'branching_float':
        codec.catalog.branching = float(branching)
    return codec


TAMPER_KINDS = ('ids_unsorted', 'ids_dtype', 'sids_dtype', 'digit_out_of_range', 'last_semantic_digit_out_of_range',
                'noncanonical_suffix', 'suffix_shifted', 'duplicate_sid', 'offsets', 'vocab', 'bases', 'depth_attr',
                'branching_float')


class DepthThreeMatchesFrozenCheck(unittest.TestCase):
    def test_valid_codecs_are_accepted_and_the_catalog_is_returned(self):
        for seed in range(4):
            codec = fx.plain_codec(200, 4, 3, seed)
            self.assertIs(gd._check_codec(codec), codec.catalog)
            self.assertIs(sd.check_codec_v2(codec, 3), codec.catalog)

    def test_every_tampered_codec_gets_the_same_outcome_and_message(self):
        for kind in TAMPER_KINDS:
            with self.subTest(kind=kind):
                frozen = outcome(gd._check_codec, tampered(3, kind))
                new = outcome(sd.check_codec_v2, tampered(3, kind), 3)
                self.assertEqual(frozen, new)
        self.assertEqual(outcome(gd._check_codec, tampered(3, 'noncanonical_suffix'))[0], 'ValueError')

    def test_prepared_catalog_mismatch_and_foreign_objects(self):
        codec = fx.plain_codec(200, 4, 3, 1)
        for prepared in ({'catalog': codec.catalog.ids[:-1]}, {'catalog': codec.catalog.ids}):
            self.assertEqual(outcome(gd._check_codec, codec, prepared), outcome(sd.check_codec_v2, codec, 3, prepared))
        self.assertEqual(outcome(gd._check_codec, codec.catalog), outcome(sd.check_codec_v2, codec.catalog, 3))
        self.assertEqual(outcome(gd._check_codec, codec.catalog)[0], 'ValueError')


class DepthFourCheck(unittest.TestCase):
    def test_valid_depth_four_codec_is_accepted_for_depth_four_only(self):
        codec = fx.plain_codec(300, 4, 4, 5)
        self.assertEqual(codec.catalog.depth, 4)
        self.assertIs(sd.check_codec_v2(codec, 4), codec.catalog)
        self.assertRaises(ValueError, sd.check_codec_v2, codec, 3)
        self.assertRaises(ValueError, gd._check_codec, codec)  # the frozen check really is depth 3 only

    def test_a_depth_three_codec_is_refused_at_depth_four(self):
        self.assertRaises(ValueError, sd.check_codec_v2, fx.plain_codec(200, 4, 3, 5), 4)

    def test_every_tampered_depth_four_codec_is_refused(self):
        for kind in TAMPER_KINDS:
            with self.subTest(kind=kind):
                self.assertRaises(ValueError, sd.check_codec_v2, tampered(4, kind), 4)

    def test_depth_argument_range(self):
        for depth in range(2, 9):
            with self.subTest(depth=depth):
                self.assertIsNotNone(sd.check_codec_v2(fx.plain_codec(120, 2, depth, depth), depth))
        for bad in (0, 1, 9, -3, '4', 4.0, True, None):
            with self.subTest(bad=bad):
                self.assertRaises(ValueError, sd.check_codec_v2, fx.plain_codec(200, 4, 4, 5), bad)
        for depth in (1, 9):  # a self-consistent codec of that depth is still refused: the range is part of the contract
            with self.subTest(consistent_codec_depth=depth):
                ids, sids = fx.synth(60, 2, depth, depth)
                codec = sd.SIDTokensD(SIDCatalog(ids, sids, 2, depth, []))
                self.assertRaises(ValueError, sd.check_codec_v2, codec, depth)

    def test_prefix_order_uses_every_semantic_column(self):
        # Two leaves that differ only in the last semantic digit: a canonical check that ignores that digit mixes them up.
        ids = np.array([10, 20, 30, 40], dtype=np.uint32)
        sids = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1]], dtype=np.uint32)
        codec = sd.SIDTokensD(SIDCatalog(ids, sids, 2, 4, []))
        self.assertIs(sd.check_codec_v2(codec, 4), codec.catalog)
        sids[[0, 2], 4] = sids[[2, 0], 4]  # now item 10 has suffix 1 and item 30 suffix 0: noncanonical
        self.assertRaises(ValueError, sd.check_codec_v2, sd.SIDTokensD(SIDCatalog(ids, sids, 2, 4, [])), 4)


class TokenClasses(unittest.TestCase):
    def test_depth_three_tokens_equal_the_frozen_class(self):
        ids, sids = fx.synth(200, 4, 3, 2)
        cat = SIDCatalog(ids, sids, 4, 3, [])
        old, new = SIDTokens(cat), sd.SIDTokensD(cat)
        self.assertEqual((new.offsets, new.vocab_size), (old.offsets, old.vocab_size))
        for item in ids[:25]:
            self.assertEqual(new.item(int(item)), old.item(int(item)))
        histories = [[int(x) for x in ids[i:i + 7]] for i in range(0, 40, 5)]
        targets = [int(ids[i]) for i in range(60, 68)]
        for a, b in zip(old.batch(histories, targets), new.batch(histories, targets)):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(new.history(histories[0]), old.history(histories[0]))

    def test_depth_four_tokens(self):
        ids, sids = fx.synth(300, 4, 4, 2)
        cat = SIDCatalog(ids, sids, 4, 4, [])
        tokens = sd.SIDTokensD(cat)
        suffix = int(sids[:, 4].max()) + 1
        self.assertEqual(tokens.offsets, [2, 6, 10, 14, 18])
        self.assertEqual(tokens.vocab_size, 18 + suffix)
        self.assertEqual(len(tokens.item(int(ids[0]))), 5)
        history = [int(x) for x in ids[:60]]
        self.assertEqual(len(tokens.history(history)), 250)  # last 50 items x 5 tokens
        source, decoder, labels = tokens.batch([history[:3], history[:5]], [int(ids[70]), int(ids[71])])
        self.assertEqual(tuple(source.shape), (2, 25))
        self.assertEqual(tuple(decoder.shape), (2, 5))
        self.assertEqual(tuple(labels.shape), (2, 5))
        self.assertTrue(torch.equal(decoder[:, 1:], labels[:, :-1]) and bool((decoder[:, 0] == 1).all()))
        self.assertTrue(bool((source[0, 15:] == 0).all()))  # padded to the longer history
        self.assertRaises(ValueError, tokens.history, [])

    def test_wrong_digit_count_is_refused(self):
        cat = SIDCatalog(*fx.synth(100, 4, 3, 1), 4, 3, [])
        cat.bases = cat.bases[:-1]
        self.assertRaises(ValueError, sd.SIDTokensD, cat)

    def test_windowed_tokens_depth_three_equal_the_frozen_class(self):
        ids, sids = fx.synth(200, 4, 3, 2)
        validated = sd.ValidatedCodecD(SIDCatalog(ids, sids, 4, 3, []), 3)
        old, new = SIDTokensL(validated, 50), sd.SIDTokensLD(validated, 50)
        history = [int(x) for x in ids[:80]]
        self.assertEqual(new.history(history), old.history(history))
        self.assertEqual((new.offsets, new.vocab_size, new.max_len), (old.offsets, old.vocab_size, old.max_len))
        for a, b in zip(old.batch([history[:4]], [int(ids[90])]), new.batch([history[:4]], [int(ids[90])])):
            self.assertTrue(torch.equal(a, b))

    def test_windowed_tokens_depth_four_and_registered_window_only(self):
        ids, sids = fx.synth(300, 4, 4, 2)
        validated = sd.ValidatedCodecD(SIDCatalog(ids, sids, 4, 4, []), 4)
        tokens = sd.SIDTokensLD(validated, 50)
        self.assertEqual(len(tokens.history([int(x) for x in ids[:120]])), 250)
        for bad in (20, 100, 49, 0, '50'):
            with self.subTest(window=bad):
                self.assertRaises(ValueError, sd.SIDTokensLD, validated, bad)
        self.assertRaises(ValueError, sd.SIDTokensLD, fx.plain_codec(100, 4, 4, 1), 50)  # needs a validated codec


class ValidatedCodec(unittest.TestCase):
    def test_depth_three_matches_the_frozen_validated_codec(self):
        ids, sids = fx.synth(200, 4, 3, 4)
        old = gd._ValidatedCodec(SIDCatalog(ids.copy(), sids.copy(), 4, 3, []))
        new = sd.ValidatedCodecD(SIDCatalog(ids.copy(), sids.copy(), 4, 3, []), 3)
        self.assertEqual((new.offsets, new.vocab_size), (old.offsets, old.vocab_size))
        self.assertEqual(new._namespace_bytes(), old._namespace_bytes())
        for name in ('ids', 'sids', 'keys', 'key_items'):
            self.assertFalse(getattr(new.catalog, name).flags.writeable)
            self.assertTrue(np.array_equal(getattr(new.catalog, name), getattr(old.catalog, name)))
        new.check_identity()

    def test_depth_four_identity_checks_hold_and_detect_tampering(self):
        ids, sids = fx.synth(300, 4, 4, 4)
        codec = sd.ValidatedCodecD(SIDCatalog(ids, sids, 4, 4, []), 4)
        self.assertEqual(codec.depth, 4)
        codec.check_identity()
        self.assertFalse(codec.catalog.sids.flags.writeable)
        codec.offsets[1] += 1
        self.assertRaises(ValueError, codec.check_identity)

    def test_depth_must_match_the_catalog(self):
        ids, sids = fx.synth(300, 4, 4, 4)
        self.assertRaises(ValueError, sd.ValidatedCodecD, SIDCatalog(ids, sids, 4, 4, []), 3)
        ids, sids = fx.synth(200, 4, 3, 4)
        self.assertRaises(ValueError, sd.ValidatedCodecD, SIDCatalog(ids, sids, 4, 3, []), 4)

    def test_noncanonical_suffix_is_refused_at_construction(self):
        codec = tampered(4, 'noncanonical_suffix')
        self.assertRaises(ValueError, sd.ValidatedCodecD, codec.catalog, 4)

    def test_run_context_style_identity_call_works(self):
        codec = sd.ValidatedCodecD(SIDCatalog(*fx.synth(300, 4, 4, 6), 4, 4, []), 4)
        codec.check_identity()
        self.assertEqual(tuple(int(x) for x in codec.catalog.ids), tuple(sorted(int(x) for x in codec.catalog.ids)))


class AssignSuffix(unittest.TestCase):
    def test_depth_three_equals_assign_fourth_code(self):
        for seed in range(5):
            rng = np.random.RandomState(seed)
            ids = rng.choice(10 ** 6, 250, replace=False).astype(np.uint32)  # deliberately unsorted rows
            codes = rng.randint(0, 3, size=(250, 3))
            got, want = sd.assign_suffix(ids, codes), rq.assign_fourth_code(ids, codes)
            self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
            self.assertTrue(np.array_equal(got, want))

    def test_depth_four_matches_an_independent_reference(self):
        for seed in range(5):
            rng = np.random.RandomState(seed)
            ids = rng.choice(10 ** 6, 400, replace=False).astype(np.uint32)
            codes = rng.randint(0, 3, size=(400, 4))
            got = sd.assign_suffix(ids, codes)
            self.assertEqual(got.shape, (400, 5))
            self.assertTrue(np.array_equal(got[:, :4], codes))
            self.assertTrue(np.array_equal(got[:, 4], fx.reference_suffix(ids, codes)))
            self.assertEqual(int(got[:, 4].min()), 0)

    def test_result_is_a_canonical_depth_four_codec_and_independent_of_row_order(self):
        rng = np.random.RandomState(9)
        ids = rng.choice(10 ** 6, 300, replace=False).astype(np.uint32)
        codes = rng.randint(0, 3, size=(300, 4))
        out = sd.assign_suffix(ids, codes)
        perm = rng.permutation(300)
        again = sd.assign_suffix(ids[perm], codes[perm])
        self.assertTrue(np.array_equal(again, out[perm]))
        order = np.argsort(ids)
        sd.check_codec_v2(sd.SIDTokensD(SIDCatalog(ids[order], out[order], 3, 4, [])), 4)

    def test_unique_leaves_get_suffix_zero(self):
        out = sd.assign_suffix(np.array([5, 6, 7], dtype=np.uint32), np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]))
        self.assertEqual(out[:, 4].tolist(), [0, 0, 0])

    def test_invalid_inputs(self):
        ids = np.array([1, 2, 3], dtype=np.uint32)
        good = np.zeros((3, 4), dtype=np.int64)
        for bad_ids, bad_codes in ((ids[:2], good), (np.array([1, 1, 2], dtype=np.uint32), good), (ids, good[:, :1]),
                                   (ids, np.zeros((3, 9), dtype=np.int64)), (ids, -good - 1), (ids, good.ravel()),
                                   (ids.reshape(3, 1), good)):
            with self.subTest(codes=np.shape(bad_codes)):
                self.assertRaisesRegex(ValueError, 'invalid ids/codes', sd.assign_suffix, bad_ids, bad_codes)


if __name__ == '__main__':
    unittest.main()
