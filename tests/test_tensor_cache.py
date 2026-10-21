"""Synthetic constructor tests only; no anchor/model step replay."""
import copy
import hashlib
import random
import unittest

import numpy as np
import torch

from vkgr.tensor_cache import TensorCache, bind_step, namespace


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def pins(**changes):
    return namespace(**{**{k: digest(k) for k in ('input', 'sid', 'config', 'source', 'runtime', 'constructor')}, **changes})


class Tokens:
    max_len = 50

    def __init__(self):
        self.table = {i: [i + 2, i + 3] for i in range(20)}

    def item(self, i):
        return self.table[i]


def g_builder(rows, tokens):
    values = [[t for i in r['history'] for t in tokens.item(i)] for r in rows]
    w = max(map(len, values))
    source = torch.tensor([v + [0] * (w - len(v)) for v in values], dtype=torch.long)
    labels = torch.tensor([tokens.item(r['target']) for r in rows], dtype=torch.long)
    decoder = torch.cat([torch.ones(len(rows), 1, dtype=torch.long), labels[:, :-1]], dim=1)
    return source, decoder, labels


def p_builder(rows, idmap, width, feature):
    hist = torch.zeros(len(rows), width, dtype=torch.long)
    side = torch.zeros(len(rows), width, 6, dtype=torch.long)
    for j, row in enumerate(rows):
        n = len(row['history'])
        hist[j, :n] = torch.tensor([idmap[i] for i in row['history']])
        side[j, :n, :3] = torch.tensor(row['side'])
        if feature == 'f3_f4_recency':
            side[j, :n, 3] = torch.tensor([p[0] for p in row['history_positions']])
        if feature == 'f3_f5_scene':
            side[j, :n, 4:6] = torch.tensor(row['ctx'])
    return hist, side


ROWS = [{'history': h, 'target': i + 3, 'side': [[1, 2, 3] for _ in h],
         'history_positions': [[j, j] for j in h], 'ctx': [[1, 2] for _ in h]}
        for i, h in enumerate(([1], [1, 2, 3], [2, 4]))]


def same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and x.shape == y.shape and x.stride() == y.stride()
        assert x.contiguous().numpy().tobytes() == y.contiguous().numpy().tobytes()


class CacheTests(unittest.TestCase):
    def test_g_padding_tail_and_cursor(self):
        c, t = TensorCache(pins()), Tokens()
        for indices in ([0, 1], [2], [1, 0], [0]):
            rows = [ROWS[i] for i in indices]
            same(c.g(rows, t, g_builder), g_builder(rows, t))
        self.assertGreater(c.hits, 0)

    def test_p_padding_tail_cursor_feature(self):
        c, mapping = TensorCache(pins()), {i: i + 1 for i in range(20)}
        for feature in ('id', 'f3_f4_recency', 'f3_f5_scene'):
            for indices in ([0, 1], [2], [1, 0], [0]):
                rows = [ROWS[i] for i in indices]
                width = max(len(r['history']) for r in rows)
                same(c.p(rows, mapping, width, feature, p_builder), p_builder(rows, mapping, width, feature))

    def test_return_storage_not_shared(self):
        c, t = TensorCache(pins()), Tokens()
        value = c.g(ROWS, t, g_builder)
        value[0].fill_(999)
        same(c.g(ROWS, t, g_builder), g_builder(ROWS, t))

    def test_mapping_and_all_row_fields_keyed(self):
        c, t = TensorCache(pins()), Tokens()
        c.g(ROWS, t, g_builder)
        t.table[1] = [9, 10]
        same(c.g(ROWS, t, g_builder), g_builder(ROWS, t))
        mapping = {i: i for i in range(20)}
        c.p(ROWS, mapping, 3, 'id', p_builder)
        mapping[1] = 10
        same(c.p(ROWS, mapping, 3, 'id', p_builder), p_builder(ROWS, mapping, 3, 'id'))
        changed = copy.deepcopy(ROWS)
        changed[0]['side'][0][1] = 19
        same(c.p(changed, mapping, 3, 'id', p_builder), p_builder(changed, mapping, 3, 'id'))

    def test_namespace_all_dependencies_required_and_affect_key(self):
        base = {k: digest(k) for k in ('input', 'sid', 'config', 'source', 'runtime', 'constructor')}
        for k in base:
            altered = {**base, k: digest('different')}
            self.assertNotEqual(namespace(**base), namespace(**altered), k)
            with self.assertRaises((TypeError, ValueError)):
                namespace(**{x: v for x, v in base.items() if x != k})
        with self.assertRaises(ValueError):
            pins(input='not-sha')

    def test_bounded_lru_and_oversized(self):
        c, t = TensorCache(pins(), max_bytes=100), Tokens()
        same(c.g(ROWS, t, g_builder), g_builder(ROWS, t))
        self.assertLessEqual(c.bytes, 100)
        tiny = TensorCache(pins(), max_bytes=1)
        same(tiny.g(ROWS, t, g_builder), g_builder(ROWS, t))
        self.assertEqual(tiny.bytes, 0)

    def test_rng_consumption_order_negative_mask_logq_passthrough(self):
        # No model or optimizer is run: this synthetic step exercises the exact
        # stochastic-before/builder/stochastic-after call boundary only.
        def negative_inputs(rng):
            return rng.sample(range(100), 4), torch.tensor([False, True]), np.array([0.25, 0.75])
        def original(rows, rng):
            neg, mask, logq = _negative_inputs(rng)
            h, side = _tensors2(rows, {i: i for i in range(20)}, 3, 'id')
            return neg, h, side, mask, logq, rng.random()
        original.__globals__['_tensors2'] = p_builder
        original.__globals__['_negative_inputs'] = negative_inputs
        cache = TensorCache(pins())
        wrapped = bind_step(original, '_tensors2', lambda *a: cache.p(*a, p_builder))
        self.assertIs(wrapped.__code__, original.__code__)
        self.assertIs(original.__globals__['_tensors2'], p_builder)
        for _ in range(2):
            py = random.getstate(); np0 = np.random.get_state(); torch0 = torch.get_rng_state().clone()
            a, b = random.Random(42), random.Random(42)
            expected, actual = original(ROWS, a), wrapped(ROWS, b)
            self.assertEqual(expected[0], actual[0], 'negative input passthrough')
            same(expected[1:3], actual[1:3])
            self.assertEqual(expected[3].numpy().tobytes(), actual[3].numpy().tobytes(), 'mask input passthrough')
            self.assertEqual(expected[4].tobytes(), actual[4].tobytes(), 'logq input passthrough')
            self.assertEqual(expected[5], actual[5]); self.assertEqual(a.getstate(), b.getstate())
            self.assertEqual(py, random.getstate())
            self.assertEqual(np0[1].tobytes(), np.random.get_state()[1].tobytes())
            self.assertTrue(torch.equal(torch0, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
