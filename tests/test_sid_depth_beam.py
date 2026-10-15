"""G6-PREP step 3: depth-generic constrained beam. Depth 3 must equal the frozen decoders query by query."""
import unittest

import numpy as np
import torch

from tests import g6_fixtures as fx
from vkgr import generative_batched as gb
from vkgr import sid_depth as sd
from vkgr import sid_depth_beam as sdb
from vkgr import tiger
from vkgr.sid import SIDCatalog


def small_model(codec, seed=0):
    torch.manual_seed(seed)
    return tiger.TigerModel(codec.vocab_size, 16, 2, 1, 1, 32).eval()


def world(depth, n=200, branching=4, seed=1):
    ids, sids = fx.synth(n, branching, depth, seed)
    cat = SIDCatalog(ids, sids, branching, depth, [])
    codec = sd.ValidatedCodecD(cat, depth)
    return ids, codec, small_model(codec, seed)


def histories(ids, count=6, width=4):
    return [[int(x) for x in ids[i * width:(i + 1) * width]] for i in range(count)]


class FrozenDepthThree(unittest.TestCase):
    def test_serial_beam_equals_the_frozen_decoder(self):
        ids, codec, model = world(3)
        for beam in (1, 3, 20, 100):
            for history in histories(ids):
                with self.subTest(beam=beam, history=history[0]):
                    self.assertEqual(sdb.decode_history_d(model, codec, history, beam), tiger.decode_history(model, codec, history, beam))

    def test_group_decoder_equals_the_frozen_group_decoder(self):
        ids, codec, model = world(3)
        sources = [codec.history(h) for h in histories(ids, 5)]
        with torch.no_grad():
            self.assertEqual(sdb.decode_group_d(model, codec, sources, 20), gb._decode_group(model, codec, sources, 20))
            self.assertEqual(sdb.decode_group_d(model, codec, sources[:1], 100), gb._decode_group(model, codec, sources[:1], 100))

    def test_batched_histories_equal_the_frozen_batched_decoder(self):
        ids, codec, model = world(3)
        mixed = histories(ids, 6, 3) + histories(ids, 4, 5)
        for batch in (1, 3, 256):
            with self.subTest(batch=batch):
                self.assertEqual(sdb.decode_histories_batched_d(model, codec, mixed, 20, batch), gb.decode_histories_batched(model, codec, mixed, 20, batch))


class DepthFour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids, cls.codec, cls.model = world(4, n=300, branching=3, seed=2)

    def exhaustive(self, history):
        """Teacher-forced total log-probability of every catalog item; ties by item id."""
        codec, model = self.codec, self.model
        source = torch.tensor([codec.history(history)], dtype=torch.long)
        rows = []
        for item in self.ids:
            tokens = codec.item(int(item))
            rows.append([1] + tokens[:-1])
        with torch.no_grad():
            memory, padding = model.encode(source)
            decoder = torch.tensor(rows, dtype=torch.long)
            logprob = torch.log_softmax(model.decode(memory.expand(len(rows), -1, -1), padding.expand(len(rows), -1), decoder), dim=-1)
        totals = []
        for row, item in enumerate(self.ids):
            tokens = codec.item(int(item))
            totals.append((int(item), sum(float(logprob[row, position, token]) for position, token in enumerate(tokens))))
        totals.sort(key=lambda pair: (-pair[1], pair[0]))
        return [item for item, _ in totals]

    def test_wide_beam_equals_the_exhaustive_ranking(self):
        for history in histories(self.ids, 4):
            with self.subTest(history=history[0]):
                items, audit = sdb.decode_history_d(self.model, self.codec, history, 500)
                self.assertEqual(items, self.exhaustive(history))
                self.assertEqual((audit['deadend'], audit['collision'], audit['invalid']), (0, 0, 0))

    def test_serial_group_and_batched_agree_bit_for_bit(self):
        hs = histories(self.ids, 5, 3)
        serial = [sdb.decode_history_d(self.model, self.codec, h, 20) for h in hs]
        self.assertEqual(sdb.decode_histories_batched_d(self.model, self.codec, hs, 20, 1), serial)
        self.assertEqual(sdb.decode_histories_batched_d(self.model, self.codec, hs, 20, 1), [sdb.decode_histories_batched_d(self.model, self.codec, [h], 20, 1)[0] for h in hs])
        with torch.no_grad():
            self.assertEqual(sdb.decode_group_d(self.model, self.codec, [self.codec.history(h) for h in hs], 20)[2], serial[2])

    def test_every_returned_item_is_a_catalog_item_and_unique(self):
        items, audit = sdb.decode_history_d(self.model, self.codec, histories(self.ids)[0], 50)
        self.assertEqual(len(items), len(set(items)))
        self.assertTrue(set(items) <= set(int(x) for x in self.ids))
        self.assertLessEqual(audit['max_active'], 50)

    def test_five_scoring_calls_with_growing_prefixes(self):
        seen = []

        def score(prefixes):
            seen.append(sorted({len(p) for p in prefixes}))
            return torch.zeros((len(prefixes), self.codec.vocab_size))
        sdb.constrained_beam_d(self.codec.catalog, self.codec, score, 4)
        self.assertEqual(seen, [[0], [1], [2], [3], [4]])

    def test_invalid_beam_and_logits(self):
        history = histories(self.ids)[0]
        for beam in (0, -1, '3', 2.0):
            with self.subTest(beam=beam):
                self.assertRaisesRegex(ValueError, 'invalid beam', sdb.decode_history_d, self.model, self.codec, history, beam)
                self.assertRaisesRegex(ValueError, 'invalid beam', sdb.decode_histories_batched_d, self.model, self.codec, [history], beam, 1)
        nan = lambda prefixes: torch.full((len(prefixes), self.codec.vocab_size), float('nan'))
        self.assertRaisesRegex(ValueError, 'nonfinite', sdb.constrained_beam_d, self.codec.catalog, self.codec, nan, 3)
        short = lambda prefixes: torch.zeros((len(prefixes), self.codec.vocab_size - 1))
        self.assertRaisesRegex(ValueError, 'nonfinite', sdb.constrained_beam_d, self.codec.catalog, self.codec, short, 3)


class StubModel(torch.nn.Module):
    """Encoder/decoder with fixed per-position logits, so that exact score ties can be constructed."""

    def __init__(self, codec, rows):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.codec, self.rows = codec, rows

    def encode(self, source):
        return torch.zeros((len(source), 1, 1)), torch.zeros((len(source), 1), dtype=torch.bool)

    def decode(self, memory, padding, decoder):
        out = torch.full((len(decoder), decoder.shape[1], self.codec.vocab_size), -1.0e4)
        position = decoder.shape[1] - 1
        for digit, value in enumerate(self.rows[position]):
            out[:, position, self.codec.offsets[position] + digit] = value
        return out


class TieBreaks(unittest.TestCase):
    """Four items on a binary depth-2 tree whose path order is the reverse of the item order (10 -> path (1,1), ...)."""
    TIE = [[0.0, 1.0], [1.0, 0.0], [0.0]]  # paths (0,0) and (1,1) tie for second place
    FLAT = [[0.0, 0.0], [0.0, 0.0], [0.0]]

    def setUp(self):
        ids = np.array([10, 20, 30, 40], dtype=np.uint32)
        sids = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint32)
        self.codec = sd.ValidatedCodecD(SIDCatalog(ids, sids, 2, 2, []), 2)

    def stub(self, rows):
        model = StubModel(self.codec, rows)
        return lambda prefixes: model.decode(None, None, torch.ones((len(prefixes), len(prefixes[0]) + 1), dtype=torch.long))[:, -1]

    def test_equal_scores_are_ordered_by_path_not_by_generation_order(self):
        score = self.stub(self.TIE)
        # the tie must exist bit-for-bit, otherwise this test would prove nothing
        lp = [torch.log_softmax(score([(0,) * p]), dim=-1)[0] for p in range(2)]
        first = lambda position, digit: float(lp[position][self.codec.offsets[position] + digit])
        self.assertEqual(0. + first(0, 0) + first(1, 0), 0. + first(0, 1) + first(1, 1))  # paths (0,0) and (1,1)
        items, _ = sdb.constrained_beam_d(self.codec.catalog, self.codec, score, 2)
        # best path is (1,0) = item 20; the tie between (0,0) = item 40 and (1,1) = item 10 goes to the smaller path (0,0)
        self.assertEqual(items, [20, 40])

    def test_group_decoder_breaks_the_same_tie_the_same_way(self):
        model = StubModel(self.codec, self.TIE)
        self.assertEqual(sdb.decode_history_d(model, self.codec, [10], 2)[0], [20, 40])
        with torch.no_grad():
            self.assertEqual(sdb.decode_group_d(model, self.codec, [self.codec.history([10])] * 2, 2)[1][0], [20, 40])

    def test_final_ranking_breaks_ties_by_item_id_not_by_path(self):
        items, _ = sdb.constrained_beam_d(self.codec.catalog, self.codec, self.stub(self.FLAT), 4)
        self.assertEqual(items, [10, 20, 30, 40])
        model = StubModel(self.codec, self.FLAT)
        with torch.no_grad():
            self.assertEqual(sdb.decode_group_d(model, self.codec, [self.codec.history([10])], 4)[0][0], [10, 20, 30, 40])

    def test_a_flat_real_model_keeps_serial_and_group_equal(self):
        torch.manual_seed(0)
        model = tiger.TigerModel(self.codec.vocab_size, 16, 2, 1, 1, 32).eval()
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.zero_()
        serial = sdb.decode_history_d(model, self.codec, [10, 20], 3)
        with torch.no_grad():
            group = sdb.decode_group_d(model, self.codec, [self.codec.history([10, 20])], 3)
        self.assertEqual(group, [serial])
        self.assertEqual(serial[0], [20, 30, 40])  # the three smallest paths (0,0),(0,1),(1,0), then ordered by item id


if __name__ == '__main__':
    unittest.main()
