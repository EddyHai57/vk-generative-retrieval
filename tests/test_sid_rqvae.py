"""GTOK-01 FEAT-001: independent RQ-VAE tokenizer core (arXiv 2305.05065 sec 3.1 / 4.1.3).

Pure synthetic data, CPU only, no engineering contract, no hard-coded temp paths.
"""
import inspect
import unittest
from unittest import mock

import numpy as np
import torch

from vkgr import generative_data as gd
from vkgr import sid_rqvae as rq
from vkgr.sid import SIDCatalog
from vkgr.tiger import SIDTokens


def unit_rows(rng, n, d=64):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def blobs(n_per=64, centres=8, d=64, seed=0, noise=0.05):
    rng = np.random.RandomState(seed)
    c = unit_rows(rng, centres, d)
    x = np.concatenate([c[i] + noise * rng.normal(size=(n_per, d)) for i in range(centres)]).astype(np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32)


def clock_stepping(delta):
    state = {'t': -delta}

    def clock():
        state['t'] += delta
        return state['t']
    return clock


def hand_codebooks():
    return [torch.tensor([[0., 0.], [1., 0.], [0., 1.]]),
            torch.tensor([[0., 0.], [.5, 0.], [0., .5]]),
            torch.tensor([[0., 0.], [.1, 0.], [0., .1]])]


HAND_Z = torch.tensor([[.9, .1], [.2, .8]])


class Architecture(unittest.TestCase):
    def test_architecture_shapes_and_param_counts(self):
        for K, expected in ((256, 427872), (16, 404832)):
            with self.subTest(K=K):
                m = rq.RQVAE(K)
                self.assertEqual(sum(p.numel() for p in m.parameters()), expected)
                self.assertEqual([tuple(c.shape) for c in m.codebooks], [(K, 32)] * 3)
                enc = [(l.in_features, l.out_features) for l in m.encoder if isinstance(l, torch.nn.Linear)]
                dec = [(l.in_features, l.out_features) for l in m.decoder if isinstance(l, torch.nn.Linear)]
                self.assertEqual(enc, [(64, 512), (512, 256), (256, 128), (128, 32)])
                self.assertEqual(dec, [(32, 128), (128, 256), (256, 512), (512, 64)])
                for seq in (m.encoder, m.decoder):
                    self.assertEqual(len(seq), 7)
                    self.assertIsInstance(seq[-1], torch.nn.Linear)  # no activation on the last layer
                    self.assertTrue(all(isinstance(seq[i], torch.nn.ReLU) for i in (1, 3, 5)))
                self.assertTrue(all(p.dtype == torch.float32 for p in m.parameters()))

    def test_prereg_constants_frozen(self):
        p = rq.PREREG
        expected = {'content_dim': 64, 'encoder_hidden': (512, 256, 128), 'latent_dim': 32, 'levels': 3,
                    'beta': 0.25, 'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1,
                    'eps': 1e-7, 'batch_size': 1024, 'max_steps': 300000, 'wall_cap_seconds': 7200,
                    'kmeans_iters': 10, 'seed': 42, 'usage_gate_num_den': (4, 5), 'log_every': 1000,
                    'eval_every': 10000, 'check_every': 100, 'probe_from': 500, 'probe_step': 2000}
        for key, value in expected.items():
            self.assertEqual(p[key], value, key)


class Quantize(unittest.TestCase):
    def test_quantize_nearest_per_level_residual_chain_hand_computed(self):
        q = rq.quantize(HAND_Z, hand_codebooks())
        self.assertEqual([c.tolist() for c in q['codes']], [[1, 2], [0, 0], [2, 1]])
        r = q['residuals']
        self.assertEqual(len(r), 4)
        expected = [[[.9, .1], [.2, .8]], [[-.1, .1], [.2, -.2]], [[-.1, .1], [.2, -.2]], [[-.1, 0.], [.1, -.2]]]
        for got, want in zip(r, expected):
            self.assertTrue(torch.allclose(got, torch.tensor(want), atol=1e-6), (got, want))
        self.assertTrue(torch.allclose(q['zhat'], torch.tensor([[1., .1], [.1, 1.]]), atol=1e-6))
        # tie -> lowest index
        tie = [torch.tensor([[1., 0.], [1., 0.], [0., 0.]])] * 3
        q2 = rq.quantize(torch.tensor([[1., 0.]]), tie)
        self.assertEqual(q2['codes'][0].tolist(), [0])

    def test_loss_terms_hand_computed_beta_025(self):
        q = rq.quantize(HAND_Z, hand_codebooks())
        x = torch.tensor([[1., 2.], [0., 0.]])
        xhat = torch.tensor([[1., 0.], [1., 1.]])
        t = rq.loss_terms(x, xhat, q)
        self.assertAlmostEqual(float(t['recon']), 3.0, places=6)  # (4 + 2) / 2, sum over features
        self.assertEqual(len(t['codebook']), 3)
        self.assertEqual(len(t['commit']), 3)
        for got, want in zip(t['codebook'], (.05, .05, .03)):
            self.assertAlmostEqual(float(got), want, places=6)
        for got, want in zip(t['commit'], (.0125, .0125, .0075)):
            self.assertAlmostEqual(float(got), want, places=6)
        self.assertAlmostEqual(float(t['total']), 3.1625, places=5)

    def test_gradient_routing(self):
        torch.manual_seed(0)
        m = rq.RQVAE(8)
        x = torch.from_numpy(unit_rows(np.random.RandomState(1), 32))
        out = m(x)
        t = rq.loss_terms(x, out['xhat'], out['q'])
        groups = {'enc': list(m.encoder.parameters()), 'dec': list(m.decoder.parameters()), 'cb': list(m.codebooks)}
        allp = groups['enc'] + groups['dec'] + groups['cb']

        def live(term):
            grads = torch.autograd.grad(term, allp, retain_graph=True, allow_unused=True)
            result, i = {}, 0
            for name, params in groups.items():
                chunk = grads[i:i + len(params)]; i += len(params)
                result[name] = any(g is not None and float(g.abs().sum()) > 0 for g in chunk)
            return result
        self.assertEqual(live(t['recon']), {'enc': True, 'dec': True, 'cb': False})
        self.assertEqual(live(sum(t['codebook'])), {'enc': False, 'dec': False, 'cb': True})
        self.assertEqual(live(sum(t['commit'])), {'enc': True, 'dec': False, 'cb': False})
        for d in range(3):
            self.assertEqual(live(t['codebook'][d])['cb'], True)
            self.assertEqual(live(t['commit'][d]), {'enc': True, 'dec': False, 'cb': False})

    def test_straight_through_value_and_gradient(self):
        torch.manual_seed(0)
        m = rq.RQVAE(8)
        x = torch.from_numpy(unit_rows(np.random.RandomState(1), 16))
        out = m(x)
        self.assertTrue(torch.allclose(out['zst'].detach(), out['q']['zhat'].detach(), atol=1e-6, rtol=0))
        (g,) = torch.autograd.grad(out['zst'].sum(), out['z'], allow_unused=True)
        self.assertIsNotNone(g)
        self.assertTrue(torch.equal(g, torch.ones_like(g)))


class KMeansInit(unittest.TestCase):
    def test_kmeans_blobs_deterministic_and_seeded(self):
        x = torch.from_numpy(blobs(n_per=50, centres=3, d=8, seed=3))
        a = rq.kmeans_lloyd(x, 3, seed=5, iters=10)
        b = rq.kmeans_lloyd(x, 3, seed=5, iters=10)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(tuple(a.shape), (3, 8))
        means = torch.stack([x[i * 50:(i + 1) * 50].mean(0) for i in range(3)])
        dist = torch.cdist(means.double(), a.double())
        self.assertTrue(bool((dist.min(1).values < 0.05).all()), dist)
        # seed controls the initial centres (0 iterations returns them untouched)
        i0 = rq.kmeans_lloyd(x, 3, seed=1, iters=0)
        i0b = rq.kmeans_lloyd(x, 3, seed=1, iters=0)
        i1 = rq.kmeans_lloyd(x, 3, seed=2, iters=0)
        self.assertTrue(torch.equal(i0, i0b))
        self.assertFalse(torch.equal(i0, i1))
        # empty clusters keep their old centre (duplicate points cannot all be separated)
        dup = torch.zeros(6, 2)
        c = rq.kmeans_lloyd(dup, 3, seed=0, iters=3)
        self.assertTrue(bool(torch.isfinite(c).all()))

    def test_codebook_init_uses_first_batch_only_and_sequential_levels(self):
        torch.manual_seed(7)
        m = rq.RQVAE(4)
        batch = torch.from_numpy(blobs(n_per=16, centres=4, d=64, seed=4))
        rq.init_codebooks(m, batch, seed=9, iters=10)
        with torch.no_grad():
            z = m.encoder(batch)
            c0 = rq.kmeans_lloyd(z, 4, seed=9, iters=10)
            self.assertTrue(torch.allclose(m.codebooks[0], c0, atol=1e-6))
            r1 = z - c0[torch.cdist(z.double(), c0.double()).argmin(1)]
            c1 = rq.kmeans_lloyd(r1, 4, seed=9, iters=10)
            self.assertTrue(torch.allclose(m.codebooks[1], c1, atol=1e-5))
            r2 = r1 - c1[torch.cdist(r1.double(), c1.double()).argmin(1)]
            c2 = rq.kmeans_lloyd(r2, 4, seed=9, iters=10)
            self.assertTrue(torch.allclose(m.codebooks[2], c2, atol=1e-5))
        # fit initialises on the first batch of the first epoch only
        X = blobs(n_per=64, centres=8, d=64, seed=5)
        res = rq.fit(X, K=4, seed=3, max_steps=1, batch_size=64, device='cpu', wall_cap_seconds=1e9)
        perm = np.random.RandomState(3).permutation(len(X))
        torch.manual_seed(3)
        ref = rq.RQVAE(4)
        rq.init_codebooks(ref, torch.from_numpy(X[perm[:64]]), seed=3, iters=rq.PREREG['kmeans_iters'])
        self.assertEqual(res['init_digest'], rq.state_digest(ref.state_dict()))


class Fit(unittest.TestCase):
    def test_fit_deterministic_cpu(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        kw = dict(K=4, max_steps=50, batch_size=64, device='cpu', wall_cap_seconds=1e9)
        a = rq.fit(X, seed=11, **kw)
        b = rq.fit(X, seed=11, **kw)
        c = rq.fit(X, seed=12, **kw)
        self.assertEqual(a['status'], 'complete')
        self.assertEqual(rq.state_digest(a['state']['state_dict']), rq.state_digest(b['state']['state_dict']))
        self.assertNotEqual(rq.state_digest(a['state']['state_dict']), rq.state_digest(c['state']['state_dict']))
        self.assertEqual(a['init_digest'], b['init_digest'])

    def test_fit_smoke_loss_decreases_no_nan(self):
        X = blobs(n_per=256, centres=8, d=64, seed=8)
        self.assertEqual(X.shape, (2048, 64))
        r = rq.fit(X, K=16, seed=42, max_steps=400, batch_size=256, device='cpu', wall_cap_seconds=1e9,
                   log_every=100, eval_every=200)
        self.assertEqual(r['status'], 'complete')
        self.assertEqual(r['step'], 400)
        losses = [v for _, v in r['loss_log']]
        self.assertEqual([s for s, _ in r['loss_log']], [100, 200, 300, 400])
        self.assertTrue(all(np.isfinite(losses)))
        self.assertLess(losses[-1], losses[0])
        self.assertEqual([s for s, _ in r['usage_log']], [200, 400])
        self.assertTrue(all(len(u) == 3 and all(1 <= n <= 16 for n in u) for _, u in r['usage_log']))
        for v in r['state']['state_dict'].values():
            self.assertTrue(bool(torch.isfinite(v).all()))
            self.assertEqual(v.dtype, torch.float32)
        rep = rq.reconstruction_report(r['state'], X)
        self.assertTrue(np.isfinite(rep['mse']))

    def test_nonfinite_detected_within_100_steps_and_no_state_returned(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        real = rq.loss_terms
        calls = {'n': 0}

        def poisoned(*a, **k):
            t = real(*a, **k)
            calls['n'] += 1
            if calls['n'] == 120:
                t = dict(t); t['total'] = t['total'] * float('nan')
            return t
        with mock.patch.object(rq, 'loss_terms', poisoned):
            r = rq.fit(X, K=4, seed=1, max_steps=1000, batch_size=64, device='cpu', wall_cap_seconds=1e9)
        self.assertEqual(r['status'], 'fail_nonfinite')
        self.assertNotIn('state', r)
        self.assertLessEqual(r['step'], 220)

    def test_wall_cap_and_projection_fail_closed_with_injected_clock(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        kw = dict(K=4, seed=1, batch_size=64, device='cpu')
        # wall: 1000 s per clock call, a check every 100 steps -> over 7200 s at step 800
        r = rq.fit(X, max_steps=5000, wall_cap_seconds=7200, clock=clock_stepping(1000.0), **kw)
        self.assertEqual(r['status'], 'fail_wall')
        self.assertEqual(r['step'], 800)
        self.assertNotIn('state', r)
        # projection: 3 s per check; rate over steps 100..300 projects far beyond the cap
        r = rq.fit(X, max_steps=300000, wall_cap_seconds=7200, clock=clock_stepping(3.0),
                   probe_from=100, probe_step=300, **kw)
        self.assertEqual(r['status'], 'fail_wall_projected')
        self.assertEqual(r['step'], 300)
        self.assertNotIn('state', r)
        # a cheap clock passes the same probe and the run completes
        r = rq.fit(X, max_steps=400, wall_cap_seconds=7200, clock=clock_stepping(0.001),
                   probe_from=100, probe_step=300, **kw)
        self.assertEqual(r['status'], 'complete')
        self.assertIn('state', r)
        self.assertIsNotNone(r['prefix_digest'])
        self.assertEqual(len(r['prefix_digest']), 64)
        # exceeding the cap at the final step is not a pass either
        r = rq.fit(X, max_steps=100, wall_cap_seconds=5, clock=clock_stepping(10.0), **kw)
        self.assertEqual(r['status'], 'fail_wall')
        self.assertNotIn('state', r)


def export_state(z, rows):
    """State whose encoder always outputs `z` (zero weights, last bias = z) with given level-0 codebook."""
    torch.manual_seed(0)
    m = rq.RQVAE(len(rows))
    with torch.no_grad():
        for layer in m.encoder:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_(); layer.bias.zero_()
        m.encoder[-1].bias.copy_(torch.as_tensor(z, dtype=torch.float32))
        m.codebooks[0].copy_(torch.as_tensor(rows, dtype=torch.float32))
    return {'K': len(rows), 'state_dict': {k: v.detach().clone() for k, v in m.state_dict().items()}}


class Export(unittest.TestCase):
    def test_export_codes_float64_matches_float32_on_separated_data_and_tie_lowest_index(self):
        X = blobs(n_per=32, centres=4, d=64, seed=2)
        torch.manual_seed(2)
        m = rq.RQVAE(4)
        rq.init_codebooks(m, torch.from_numpy(X[:64]), seed=2, iters=10)
        state = {'K': 4, 'state_dict': {k: v.detach().clone() for k, v in m.state_dict().items()}}
        codes = rq.export_codes(state, X)
        self.assertEqual(codes.dtype, np.int64)
        self.assertEqual(codes.shape, (len(X), 3))
        self.assertTrue(codes.min() >= 0 and codes.max() < 4)
        # independent float64 reference
        md = rq.RQVAE(4).double(); md.load_state_dict({k: v.double() for k, v in state['state_dict'].items()})
        with torch.no_grad():
            r = md.encoder(torch.from_numpy(X).double())
            ref = []
            for c in md.codebooks:
                idx = torch.cdist(r, c).argmin(1)
                ref.append(idx.numpy()); r = r - c[idx]
        self.assertTrue(np.array_equal(codes, np.stack(ref, 1)))
        # chunking cannot change the result
        with mock.patch.object(rq, 'EXPORT_CHUNK', 7):
            self.assertTrue(np.array_equal(rq.export_codes(state, X), codes))
        # exact ties go to the lowest index
        z = np.zeros(32, dtype=np.float32); z[0] = 1.0
        rows = np.zeros((4, 32), dtype=np.float32)
        rows[0, 0] = 5; rows[1, 0] = 1; rows[2, 0] = 1; rows[3, 0] = -5
        tie = rq.export_codes(export_state(z, rows), X[:3])
        self.assertEqual(tie[:, 0].tolist(), [1, 1, 1])
        # a 1-ulp difference that float32 cannot resolve but float64 can: the closer codeword is index 1
        delta = np.float32(1e-4)
        near = np.float32(1.0) + delta
        far = np.nextafter(near, np.float32(2.0))
        rows = np.zeros((2, 32), dtype=np.float32)
        rows[0, 0] = far; rows[1, 0] = near
        res = rq.export_codes(export_state(z, rows), X[:3])
        self.assertEqual(res[:, 0].tolist(), [1, 1, 1])

    def test_reconstruction_report(self):
        X = blobs(n_per=16, centres=4, d=64, seed=2)
        z = np.zeros(32, dtype=np.float32); z[0] = 0.5
        rows = np.zeros((4, 32), dtype=np.float32)
        state = export_state(z, rows)
        sd = state['state_dict']  # decoder outputs zero, so x_hat = 0
        for k in list(sd):
            if k.startswith('decoder.'):
                sd[k] = torch.zeros_like(sd[k])
        rep = rq.reconstruction_report(state, X)
        x64 = X.astype(np.float64)
        mse = float(np.mean(np.sum(x64 ** 2, axis=1)))
        sst = float(np.mean(np.sum((x64 - x64.mean(0)) ** 2, axis=1)))
        self.assertAlmostEqual(rep['mse'], mse, places=9)
        self.assertAlmostEqual(rep['sst'], sst, places=9)
        self.assertAlmostEqual(rep['r2'], 1 - mse / sst, places=9)
        self.assertEqual(len(rep['residual_norms']), 4)
        self.assertAlmostEqual(rep['residual_norms'][0], 0.5, places=6)


class FourthCode(unittest.TestCase):
    def test_assign_fourth_code_canonical_order_independent_and_check_codec_accepts(self):
        rng = np.random.RandomState(0)
        n = 400
        ids = np.sort(rng.choice(10 ** 6, n, replace=False)).astype(np.uint32)
        codes = rng.randint(0, 4, size=(n, 3)).astype(np.int64)  # 64 prefixes for 400 items -> many collisions
        sids = rq.assign_fourth_code(ids, codes)
        self.assertEqual(sids.dtype, np.uint32)
        self.assertEqual(sids.shape, (n, 4))
        self.assertTrue(np.array_equal(sids[:, :3], codes))
        # within a prefix group the suffix is the ascending-id rank
        for key in {tuple(c) for c in codes.tolist()}:
            members = np.where((codes == key).all(1))[0]
            self.assertEqual(sids[members[np.argsort(ids[members])], 3].tolist(), list(range(len(members))))
        catalog = SIDCatalog(ids, sids, 4, 3, [])
        self.assertGreater(catalog.suffix_vocabulary, 1)
        gd._check_codec(SIDTokens(catalog))
        # row order of the input does not change the id-aligned answer
        perm = rng.permutation(n)
        shuffled = rq.assign_fourth_code(ids[perm], codes[perm])
        back = np.empty_like(shuffled); back[perm] = shuffled
        self.assertTrue(np.array_equal(back, sids))
        # no collisions -> all zeros, still canonical
        distinct = np.stack([np.arange(30) // 9, (np.arange(30) // 3) % 3, np.arange(30) % 3], 1).astype(np.int64)
        d_ids = np.arange(30, dtype=np.uint32) * 3 + 1
        d_sids = rq.assign_fourth_code(d_ids, distinct)
        self.assertEqual(int(d_sids[:, 3].max()), 0)
        gd._check_codec(SIDTokens(SIDCatalog(d_ids, d_sids, 4, 3, [])))
        # ids that are not in ascending row order but with the earlier item_id deeper in the input
        ids2 = np.array([9, 5, 7], dtype=np.uint32)
        out = rq.assign_fourth_code(ids2, np.zeros((3, 3), dtype=np.int64))
        self.assertEqual(out[:, 3].tolist(), [2, 0, 1])


class Reports(unittest.TestCase):
    def test_codebook_usage_and_usage_gate_integer_boundary(self):
        codes = np.array([[0, 1, 2], [0, 1, 3], [1, 1, 3], [2, 1, 3]])
        self.assertEqual(rq.codebook_usage(codes, 4), [3, 1, 2])
        self.assertFalse(rq.usage_gate(204, 256))
        self.assertTrue(rq.usage_gate(205, 256))
        self.assertFalse(rq.usage_gate(12, 16))
        self.assertTrue(rq.usage_gate(13, 16))
        # exact equality used*5 == K*4 passes (a strict > would fail here)
        self.assertTrue(rq.usage_gate(8, 10))
        self.assertFalse(rq.usage_gate(7, 10))

    def test_no_label_or_target_access(self):
        source = inspect.getsource(rq).lower()
        for word in ('validation-queries', 'examples.jsonl', 'support.json', 'history', 'target'):
            self.assertNotIn(word, source, word)
        self.assertNotIn('open(', source)


if __name__ == '__main__':
    unittest.main()
