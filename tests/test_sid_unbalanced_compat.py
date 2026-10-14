"""GTOK-01 FEAT-002: the frozen SID code path accepts unbalanced (Zipf-like skewed) 4-digit SIDs.

K1..K7 are the compatibility items of the plan (section 2.2). Nothing frozen is edited: the catalog, the codec
validator, the constrained beam, the batched decode and the sharded-worker pickle path are only exercised.
"""
import pickle
import random
import unittest

import numpy as np
import torch

from vkgr import generative_batched as gb
from vkgr import generative_data as gd
from vkgr import generative_training as gt
from vkgr.sid import SIDCatalog
from vkgr.sid_rqvae import assign_fourth_code
from vkgr.tiger import SIDTokens, decode_history

CONFIG = dict(d_model=64, heads=4, encoder_layers=1, decoder_layers=1, ff=256, dropout=0., precision='fp32',
              learning_rate=.001, weight_decay=.0001, gradient_clip_norm=1., effective_batch=512, seed=42)


def skewed_mapping(n, branching, seed, power=6):
    """Each of the three digits is floor(b * u**power): heavy mass on small digits, so a few leaves are huge."""
    rng = np.random.RandomState(seed)
    ids = np.sort(rng.choice(10 ** 6, n, replace=False)).astype(np.uint32)
    draws = np.minimum(np.floor(branching * rng.rand(n, 3) ** power).astype(np.int64), branching - 1)
    return ids, assign_fourth_code(ids, draws)


def make_codec(ids, sids, branching):
    return gd._ValidatedCodec(SIDCatalog(ids, sids, branching, 3, []))


def leaf_sizes(sids):
    return np.unique(sids[:, :3], axis=0, return_counts=True)[1]


class Skewed(unittest.TestCase):
    """N=20,000 catalogs with branching 256 and 16."""

    @classmethod
    def setUpClass(cls):
        cls.worlds = {}
        for branching in (256, 16):
            ids, sids = skewed_mapping(20000, branching, seed=branching)
            cls.worlds[branching] = (ids, sids, make_codec(ids, sids, branching))

    def test_fixture_is_really_skewed(self):
        for branching, (ids, sids, codec) in self.worlds.items():
            with self.subTest(branching=branching):
                self.assertGreaterEqual(int(leaf_sizes(sids).max()), 200)
                self.assertEqual(codec.catalog.suffix_vocabulary, int(leaf_sizes(sids).max()))

    def test_K1_encode_decode_round_trip_and_allowed_next_oracle(self):
        for branching, (ids, sids, codec) in self.worlds.items():
            catalog = codec.catalog
            with self.subTest(branching=branching):
                for item, row in zip(ids.tolist(), sids.tolist()):
                    path = catalog.encode(item)
                    self.assertEqual(path, tuple(row))
                    self.assertEqual(catalog.decode(path), item)
                rng = random.Random(branching)
                for _ in range(2000):
                    row = rng.randrange(len(ids))
                    length = rng.randrange(5)
                    prefix = tuple(int(x) for x in sids[row, :length])
                    mask = np.all(sids[:, :length] == np.array(prefix, dtype=np.uint32), axis=1) if length else np.ones(len(ids), bool)
                    oracle = sorted({int(x) for x in sids[mask, length]}) if length < 4 else []
                    self.assertEqual(catalog.allowed_next(prefix), oracle)

    def test_K2_check_codec_accepts_and_formulas_hold(self):
        for branching, (ids, sids, codec) in self.worlds.items():
            suffix = int(sids[:, 3].max()) + 1
            with self.subTest(branching=branching):
                self.assertIs(gd._check_codec(codec, {'catalog': ids}), codec.catalog)
                self.assertEqual(codec.catalog.bases, [branching] * 3 + [suffix])
                self.assertEqual(codec.offsets, [2, 2 + branching, 2 + 2 * branching, 2 + 3 * branching])
                self.assertEqual(codec.vocab_size, 2 + 3 * branching + suffix)
                codec.check_identity()

    def test_K2_noncanonical_suffix_and_bad_digits_are_rejected(self):
        for branching, (ids, sids, _) in self.worlds.items():
            groups, counts = np.unique(sids[:, :3], axis=0, return_counts=True)
            rows = np.flatnonzero(np.all(sids[:, :3] == groups[counts.argmax()], axis=1))
            cases = {}
            swapped = sids.copy()
            swapped[rows[0], 3], swapped[rows[1], 3] = swapped[rows[1], 3], swapped[rows[0], 3]
            cases['swapped suffixes inside one leaf'] = swapped
            gap = sids.copy()
            gap[rows[-1], 3] += 1
            cases['gap in the suffix sequence'] = gap
            duplicate = sids.copy()
            duplicate[rows[1], 3] = 0
            cases['duplicate full SID'] = duplicate
            high = sids.copy()
            high[0, 0] = branching
            cases['digit equal to branching'] = high
            for name, bad in cases.items():
                with self.subTest(branching=branching, case=name):
                    with self.assertRaises(ValueError):
                        make_codec(ids, bad, branching)
                    with self.assertRaises(ValueError):
                        gd._check_codec(SIDTokens(SIDCatalog(ids, bad, branching, 3, [])))
            with self.subTest(branching=branching, case='catalog mismatch'):
                with self.assertRaises(ValueError):
                    gd._check_codec(self.worlds[branching][2], {'catalog': ids + 1})

    def test_K3_legality_counters_are_zero_and_balance_counters_are_vacuous(self):
        for branching, (ids, sids, codec) in self.worlds.items():
            found = codec.catalog.audit()
            with self.subTest(branching=branching):
                self.assertEqual((found['full_sid_collisions'], found['roundtrip_items'], found['reachable_items'],
                                  found['valid_prefix_deadends']), (0, len(ids), len(ids), 0))
                # With nodes=[] these two are empty-set truths and carry no evidence about balance.
                self.assertTrue(found['balanced_nodes'])
                self.assertEqual(found['unconverged_nodes'], 0)
                self.assertGreater(found['max_leaf_occupancy'], 200)


class SkewedModel(unittest.TestCase):
    """N=3,000, branching 16: leaf (0,0,0) holds ~750 items, more than the beam (100)."""

    @classmethod
    def setUpClass(cls):
        cls.ids, cls.sids = skewed_mapping(3000, 16, seed=3)
        cls.codec = make_codec(cls.ids, cls.sids, 16)
        rng = random.Random(9)
        items = cls.ids.tolist()
        cls.histories = [[rng.choice(items) for _ in range(rng.choice((1, 2, 5, 20, 50)))] for _ in range(24)]

    def test_K4_serial_beam_equals_batched_decode_batch1_at_beam100(self):
        self.assertGreater(int(leaf_sizes(self.sids).max()), 100)
        for seed in (0, 1):
            torch.manual_seed(seed)
            model = gt.make_model(self.codec, CONFIG, device='cpu')
            expected = [decode_history(model, self.codec, h, 100) for h in self.histories]
            self.assertTrue(all(len(items) == 100 for items, _ in expected))
            self.assertTrue(all(audit['max_active'] == 100 for _, audit in expected))
            self.assertEqual(gb.decode_histories_batched(model, self.codec, self.histories, 100, 1), expected)

    def test_K6_codec_consumers_and_one_training_step_with_large_suffix_vocabulary(self):
        suffix = self.codec.catalog.suffix_vocabulary
        self.assertGreaterEqual(suffix, 600)
        self.assertEqual(self.codec.vocab_size, 2 + 3 * 16 + suffix)
        rows = [{'history': h, 'target': t} for h, t in zip(self.histories, self.ids.tolist()[:len(self.histories)])]
        source, decoder, labels = gd.sid_targets(rows, self.codec)
        self.assertGreater(labels.numel(), 0)
        self.assertGreaterEqual(int(labels.min()), 0)
        self.assertLess(int(labels.max()), self.codec.vocab_size)
        torch.manual_seed(0)
        model = gt.make_model(self.codec, CONFIG, device='cpu')
        loss = gt.full_vocab_ce(model(source, decoder), labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(model.output.out_features, self.codec.vocab_size)


class RealScale(unittest.TestCase):
    """N=120,415 mapping, branching 256, nodes=[]: sharded-worker pickle path and key arithmetic."""

    @classmethod
    def setUpClass(cls):
        cls.ids, cls.sids = skewed_mapping(120415, 256, seed=11)
        cls.codec = make_codec(cls.ids, cls.sids, 256)

    def test_K5_pickle_round_trip_follows_the_worker_path_and_stays_small(self):
        blob = pickle.dumps(self.codec)
        self.assertLess(len(blob), 20 * 1024 * 1024)
        self.assertEqual(self.codec.catalog.nodes, [])
        restored = gd._ValidatedCodec(pickle.loads(blob).catalog)  # exactly what generative_sharded._worker does
        gd._check_codec(restored)
        gd._check_codec(pickle.loads(blob))
        self.assertTrue(np.array_equal(restored.catalog.sids, self.sids))
        self.assertTrue(np.array_equal(restored.catalog.ids, self.ids))
        self.assertEqual(restored.vocab_size, self.codec.vocab_size)
        self.assertEqual(restored.offsets, self.codec.offsets)

    def test_K7_uint64_keys_do_not_overflow(self):
        suffix = 3000
        top = np.tile(np.array([255, 255, 255], dtype=np.uint32), (suffix, 1))
        draws = np.vstack([np.zeros((2, 3), dtype=np.uint32), top])
        ids = np.arange(1, len(draws) + 1, dtype=np.uint32)
        sids = assign_fourth_code(ids, draws.astype(np.int64))
        catalog = make_codec(ids, sids, 256).catalog
        expected = ((255 * 256 + 255) * 256 + 255) * suffix + (suffix - 1)
        self.assertEqual(catalog.suffix_vocabulary, suffix)
        self.assertEqual(int(catalog.keys.max()), expected)
        self.assertLess(256 ** 3 * 120415, 2 ** 63)  # even if one leaf held the whole catalog
        self.assertTrue(np.all(catalog.keys[1:] > catalog.keys[:-1]))
        self.assertTrue(np.array_equal(catalog.key_items, ids[np.lexsort((sids[:, 3], sids[:, 2], sids[:, 1], sids[:, 0]))]))
        keys = self.codec.catalog.keys
        self.assertEqual(len(np.unique(keys)), len(keys))
        self.assertLess(256 ** 3 * self.codec.catalog.suffix_vocabulary, 2 ** 63)


if __name__ == '__main__':
    unittest.main()
