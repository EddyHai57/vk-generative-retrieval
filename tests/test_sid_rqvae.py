"""Tests for the independent RQ-VAE tokenizer core (src/vkgr/sid_rqvae.py).

Synthetic data only; CPU only; no file paths, so the suite also runs on Linux.
"""
import inspect
import unittest

import numpy as np
import torch

try:
    from vkgr import sid_rqvae as sr
except ImportError:  # RED phase: the module does not exist yet
    sr = None


def unit_clusters(n=2048, clusters=16, seed=0, noise=0.05):
    rng = np.random.RandomState(seed)
    centers = rng.randn(clusters, 64)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = rng.randint(0, clusters, size=n)
    x = centers[labels] + noise * rng.randn(n, 64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32)


class StepClock:
    """Injected clock: the i-th call returns i * dt."""

    def __init__(self, dt):
        self.dt = dt
        self.calls = 0

    def __call__(self):
        value = self.calls * self.dt
        self.calls += 1
        return value


def nonzero(parameter):
    return parameter.grad is not None and bool((parameter.grad != 0).any())


class Base(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(sr, 'sid_rqvae missing')


class ArchitectureTests(Base):
    def test_architecture_shapes_and_param_counts(self):
        model = sr.RQVAE(256)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 427872)
        self.assertEqual(sum(p.numel() for p in sr.RQVAE(16).parameters()), 404832)
        linear, relu = torch.nn.Linear, torch.nn.ReLU
        self.assertEqual([type(m) for m in model.encoder], [linear, relu, linear, relu, linear, relu, linear])
        self.assertEqual([type(m) for m in model.decoder], [linear, relu, linear, relu, linear, relu, linear])
        self.assertEqual([(m.in_features, m.out_features) for m in model.encoder if isinstance(m, linear)],
                         [(64, 512), (512, 256), (256, 128), (128, 32)])
        self.assertEqual([(m.in_features, m.out_features) for m in model.decoder if isinstance(m, linear)],
                         [(32, 128), (128, 256), (256, 512), (512, 64)])
        self.assertEqual(len(model.codebooks), 3)
        self.assertTrue(all(tuple(c.shape) == (256, 32) for c in model.codebooks))

    def test_prereg_constants_frozen(self):
        self.assertEqual(sr.PREREG, {
            'content_dim': 64, 'encoder_hidden': (512, 256, 128), 'latent_dim': 32, 'levels': 3,
            'beta': 0.25, 'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1, 'eps': 1e-7,
            'batch_size': 1024, 'max_steps': 300000, 'wall_cap_seconds': 7200, 'kmeans_iters': 10, 'seed': 42,
            'usage_gate_num_den': (4, 5), 'log_every': 1000, 'eval_every': 10000, 'check_every': 100,
            'probe_from': 500, 'probe_step': 2000})


class QuantizeAndLossTests(Base):
    def test_quantize_nearest_per_level_residual_chain_hand_computed(self):
        z = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        c0 = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 0.]])  # row 3 duplicates row 1: tie -> lowest index
        c1 = torch.tensor([[0., 0.], [-0.1, 0.1], [0.1, -0.1], [9., 9.]])
        c2 = torch.tensor([[0., 0.], [0.05, 0.05], [-0.05, -0.05], [9., 9.]])
        q = sr.quantize(z, [c0, c1, c2])
        self.assertEqual(torch.stack(q['codes'], dim=1).tolist(), [[1, 1, 0], [2, 2, 0]])
        torch.testing.assert_close(q['embeds'][0], torch.tensor([[1., 0.], [0., 1.]]))
        torch.testing.assert_close(q['embeds'][1], torch.tensor([[-0.1, 0.1], [0.1, -0.1]]))
        torch.testing.assert_close(q['embeds'][2], torch.zeros(2, 2))
        torch.testing.assert_close(q['zhat'], z, atol=1e-6, rtol=0)
        torch.testing.assert_close(q['residuals'][0], z)
        torch.testing.assert_close(q['residuals'][1], torch.tensor([[-0.1, 0.1], [0.2, -0.2]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(q['residuals'][2], torch.tensor([[0., 0.], [0.1, -0.1]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(q['residual_last'], q['residuals'][2] - q['embeds'][2], atol=1e-6, rtol=0)

    def test_loss_terms_hand_computed_beta_025(self):
        x = torch.tensor([[1., 0.], [0., 1.]])
        xhat = torch.tensor([[0.5, 0.], [0., 0.]])
        residuals = [torch.tensor([[1., 1.], [0., 0.]]), torch.tensor([[0., 2.], [1., 0.]])]
        embeds = [torch.tensor([[0., 1.], [1., 0.]]), torch.tensor([[0., 0.], [0., 2.]])]
        terms = sr.loss_terms(x, xhat, residuals, embeds, beta=0.25)
        self.assertAlmostEqual(float(terms['recon']), 0.625, places=6)
        self.assertAlmostEqual(float(terms['codebook']), 5.5, places=6)
        self.assertAlmostEqual(float(terms['commit']), 1.375, places=6)
        self.assertAlmostEqual(float(terms['total']), 7.5, places=6)

    def test_gradient_routing(self):
        torch.manual_seed(0)
        model = sr.RQVAE(16)
        x = torch.randn(8, 64)

        def routes(term):
            out = model(x)
            value = sr.loss_terms(x, out['xhat'], out['residuals'], out['embeds'], beta=0.25)[term]
            model.zero_grad(set_to_none=True)
            value.backward()
            return (any(nonzero(p) for p in model.encoder.parameters()),
                    any(nonzero(p) for p in model.decoder.parameters()),
                    [nonzero(c) for c in model.codebooks])

        self.assertEqual(routes('recon'), (True, True, [False, False, False]))
        self.assertEqual(routes('codebook'), (False, False, [True, True, True]))
        self.assertEqual(routes('commit'), (True, False, [False, False, False]))

    def test_straight_through_value_and_gradient(self):
        z = torch.tensor([[0.3, -0.2], [1.0, 0.5]], requires_grad=True)
        zhat = torch.tensor([[0.25, 0.0], [1.5, 0.5]])
        st = sr.straight_through(z, zhat)
        torch.testing.assert_close(st.detach(), zhat, atol=1e-6, rtol=0)
        weights = torch.tensor([[1., 2.], [3., 4.]])
        (st * weights).sum().backward()
        torch.testing.assert_close(z.grad, weights)


class KMeansAndInitTests(Base):
    def test_kmeans_blobs_deterministic_and_seeded(self):
        rng = np.random.RandomState(1)
        means = np.array([[0., 0.], [20., 0.], [0., 20.]])
        pts = (means[np.repeat(np.arange(3), 30)] + 0.1 * rng.randn(90, 2)).astype(np.float32)
        points = torch.from_numpy(pts)
        init = sr.kmeans_lloyd(points, 3, seed=7, iters=0)
        expected = pts[np.random.RandomState(7).choice(90, 3, replace=False)]
        torch.testing.assert_close(init, torch.from_numpy(expected))
        self.assertFalse(torch.equal(init, sr.kmeans_lloyd(points, 3, seed=8, iters=0)))
        first = sr.kmeans_lloyd(points, 3, seed=BLOB_SEED, iters=10)
        self.assertTrue(torch.equal(first, sr.kmeans_lloyd(points, 3, seed=BLOB_SEED, iters=10)))
        got = first.numpy()[np.lexsort((first.numpy()[:, 1], first.numpy()[:, 0]))]
        want = means[np.lexsort((means[:, 1], means[:, 0]))]
        np.testing.assert_allclose(got, want, atol=0.1)

    def test_kmeans_empty_cluster_keeps_centre(self):
        points = torch.ones(6, 2)
        centres = sr.kmeans_lloyd(points, 2, seed=0, iters=5)
        self.assertTrue(bool(torch.isfinite(centres).all()))
        torch.testing.assert_close(centres, torch.ones(2, 2))

    def test_codebook_init_sequential_levels_on_given_batch(self):
        torch.manual_seed(0)
        model = sr.RQVAE(16)
        batch = torch.from_numpy(unit_clusters(512, seed=3))
        sr.init_codebooks(model, batch, seed=3, iters=10)
        with torch.no_grad():
            residual = model.encode(batch)
            for level in range(3):
                centres = sr.kmeans_lloyd(residual, 16, 3, 10)
                torch.testing.assert_close(model.codebooks[level].detach(), centres, rtol=1e-6, atol=1e-6)
                residual = residual - centres[sr.nearest_code(residual, centres)]

    def test_codebook_init_uses_first_batch_only(self):
        content = unit_clusters(1024, clusters=8, seed=4)
        kw = dict(K=4, seed=5, max_steps=1, batch_size=128, wall_cap_seconds=1e9)
        inside = sr.first_batch_indices(1024, 128, 5)
        self.assertEqual(len(inside), 128)
        outside = next(i for i in range(1024) if i not in set(inside.tolist()))
        other = unit_clusters(1024, clusters=8, seed=9)
        changed_outside = content.copy(); changed_outside[outside] = other[outside]
        changed_inside = content.copy(); changed_inside[inside[0]] = other[inside[0]]
        base = sr.fit(content, **kw)['init_digest']
        self.assertEqual(base, sr.fit(changed_outside, **kw)['init_digest'])
        self.assertNotEqual(base, sr.fit(changed_inside, **kw)['init_digest'])


class FitTests(Base):
    def test_fit_deterministic_cpu(self):
        content = unit_clusters(512, clusters=8, seed=2)
        kw = dict(K=16, max_steps=50, batch_size=128, wall_cap_seconds=1e9)
        a = sr.fit(content, seed=1, **kw)
        b = sr.fit(content, seed=1, **kw)
        c = sr.fit(content, seed=2, **kw)
        self.assertEqual(a['status'], 'complete')
        self.assertEqual(sr.state_digest(a['state']), sr.state_digest(b['state']))
        self.assertNotEqual(sr.state_digest(a['state']), sr.state_digest(c['state']))

    def test_prefix_digest_matches_state_at_probe_step(self):
        content = unit_clusters(512, clusters=8, seed=2)
        kw = dict(K=8, seed=1, batch_size=128, wall_cap_seconds=1e9, check_every=10, probe_from=10)
        short = sr.fit(content, max_steps=20, probe_step=20, **kw)
        long = sr.fit(content, max_steps=30, probe_step=20, **kw)
        self.assertEqual(long['prefix_digest'], sr.state_digest(short['state']))
        self.assertIsNone(sr.fit(content, max_steps=10, probe_step=20, **kw)['prefix_digest'])

    def test_fit_smoke_loss_decreases_no_nan(self):
        content = unit_clusters(2048, clusters=16, seed=0)
        result = sr.fit(content, K=16, max_steps=400, batch_size=256, wall_cap_seconds=1e9,
                        log_every=100, eval_every=200)
        self.assertEqual(result['status'], 'complete')
        losses = [row['loss'] for row in result['loss_log']]
        self.assertEqual([row['step'] for row in result['loss_log']], [100, 200, 300, 400])
        self.assertTrue(all(np.isfinite(losses)))
        self.assertLess(losses[-1], losses[0])
        self.assertEqual([row['step'] for row in result['usage_log']], [200, 400])
        self.assertTrue(all(len(row['used']) == 3 and all(1 <= u <= 16 for u in row['used'])
                            for row in result['usage_log']))
        for tensor in result['state']['tensors'].values():
            self.assertTrue(bool(torch.isfinite(tensor).all()))

    def test_nonfinite_detected_within_100_steps_and_no_state_returned(self):
        content = unit_clusters(256, clusters=4, seed=1)
        result = sr.fit(content, K=4, lr=1e30, max_steps=300, batch_size=64, wall_cap_seconds=1e9)
        self.assertEqual(result['status'], 'fail_nonfinite')
        self.assertLessEqual(result['step'], 100)
        self.assertNotIn('state', result)
        bad = content.copy(); bad[3, 0] = np.nan
        with self.assertRaises(ValueError):
            sr.fit(bad, K=4, max_steps=1, batch_size=64, wall_cap_seconds=1e9)

    def test_wall_cap_and_projection_fail_closed_with_injected_clock(self):
        content = unit_clusters(128, clusters=4, seed=1)
        kw = dict(K=4, batch_size=32, check_every=10, probe_from=10)
        wall = sr.fit(content, max_steps=1000, probe_step=2000, wall_cap_seconds=25, clock=StepClock(10), **kw)
        self.assertEqual((wall['status'], wall['step']), ('fail_wall', 30))
        self.assertNotIn('state', wall)
        late = sr.fit(content, max_steps=30, probe_step=2000, wall_cap_seconds=15, clock=StepClock(20), **kw)
        self.assertEqual((late['status'], late['step']), ('fail_wall', 30))
        self.assertNotIn('state', late)
        projected = sr.fit(content, max_steps=1000, probe_step=20, wall_cap_seconds=50, clock=StepClock(1), **kw)
        self.assertEqual((projected['status'], projected['step']), ('fail_wall_projected', 20))
        self.assertNotIn('state', projected)
        fine = sr.fit(content, max_steps=60, probe_step=20, wall_cap_seconds=500, clock=StepClock(1), **kw)
        self.assertEqual(fine['status'], 'complete')
        self.assertIn('state', fine)


class ExportAndAssignTests(Base):
    def _constant_state(self, k, base, c0_rows):
        """All weights zero: z equals the encoder's last bias for every input."""
        model = sr.RQVAE(k)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.encoder[6].bias.copy_(torch.as_tensor(base, dtype=torch.float32))
            model.codebooks[0].copy_(torch.as_tensor(c0_rows, dtype=torch.float32))
        return sr.model_state(model)

    def test_export_codes_float64_matches_float32_on_separated_data_and_tie_lowest_index(self):
        # float32 cancels to an exact tie at level 0 (expansion form); float64 resolves it to code 1.
        base = np.full(32, 1000., dtype=np.float32)
        c0 = np.tile(base, (2, 1)); c0[0, 0] = 1000.5; c0[1, 0] = 1000.4
        state = self._constant_state(2, base, c0)
        x = np.zeros((3, 64), dtype=np.float32)
        z = torch.as_tensor(base).repeat(3, 1)
        self.assertEqual(sr.nearest_code(z, torch.as_tensor(c0, dtype=torch.float32)).tolist(), [0, 0, 0])
        codes = sr.export_codes(state, x)
        self.assertEqual((codes.dtype, codes.shape), (np.int64, (3, 3)))
        self.assertEqual(codes[:, 0].tolist(), [1, 1, 1])
        # identical code words: tie goes to the lowest index
        tie = self._constant_state(2, base, np.tile(base, (2, 1)))
        self.assertEqual(sr.export_codes(tie, x)[:, 0].tolist(), [0, 0, 0])
        # trained model on separated clusters: float64 export agrees with float32 forward
        content = unit_clusters(512, clusters=8, seed=6)
        trained = sr.fit(content, K=8, seed=1, max_steps=30, batch_size=128, wall_cap_seconds=1e9)['state']
        with torch.no_grad():
            out = sr.build_model(trained, torch.float32)(torch.from_numpy(content))
        want = torch.stack(out['codes'], dim=1).numpy()
        got = sr.export_codes(trained, content)
        self.assertGreaterEqual(float((got == want).all(axis=1).mean()), 0.99)

    def test_assign_fourth_code_canonical_order_independent_and_check_codec_accepts(self):
        from vkgr import generative_data as gd
        from vkgr.sid import SIDCatalog
        from vkgr.tiger import SIDTokens
        ids = np.array([15, 13, 10, 14, 12, 11], dtype=np.uint32)
        codes = np.array([[0, 0, 0], [1, 2, 3], [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 2, 3]], dtype=np.int64)
        sids = sr.assign_fourth_code(ids, codes)
        self.assertEqual((sids.dtype, sids.shape), (np.uint32, (6, 4)))
        self.assertEqual(sids[:, :3].tolist(), codes.tolist())
        by_id = {int(i): int(s) for i, s in zip(ids, sids[:, 3])}
        self.assertEqual(by_id, {10: 0, 11: 0, 12: 1, 13: 1, 14: 0, 15: 2})
        perm = np.array([3, 0, 5, 1, 4, 2])
        shuffled = sr.assign_fourth_code(ids[perm], codes[perm])
        self.assertEqual({int(i): int(s) for i, s in zip(ids[perm], shuffled[:, 3])}, by_id)
        order = np.argsort(ids)
        codec = SIDTokens(SIDCatalog(ids[order], sids[order], 4, 3, []))
        gd._check_codec(codec)
        distinct = sr.assign_fourth_code(np.arange(5, dtype=np.uint32), np.arange(15).reshape(5, 3) % 7)
        self.assertEqual(distinct[:, 3].tolist(), [0] * 5)
        with self.assertRaises(ValueError):
            sr.assign_fourth_code(np.array([1, 1], dtype=np.uint32), np.zeros((2, 3), dtype=np.int64))


class ReportTests(Base):
    def test_usage_gate_integer_boundary(self):
        self.assertFalse(sr.usage_gate(204, 256))
        self.assertTrue(sr.usage_gate(205, 256))
        self.assertFalse(sr.usage_gate(12, 16))
        self.assertTrue(sr.usage_gate(13, 16))
        for used, k in ((4, 5), (8, 10), (200, 250)):  # exactly 80 percent passes
            self.assertTrue(sr.usage_gate(used, k))
        self.assertFalse(sr.usage_gate(7, 10))

    def test_codebook_usage_counts_distinct_codes_per_level(self):
        codes = np.array([[0, 1, 2], [0, 1, 3], [1, 1, 3]])
        self.assertEqual(sr.codebook_usage(codes, 4), [2, 1, 2])

    def test_reconstruction_report_hand_computed(self):
        model = sr.RQVAE(2)
        z0 = np.zeros(32, dtype=np.float32); z0[0] = 1.0
        b = np.linspace(-1, 1, 64).astype(np.float32)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.encoder[6].bias.copy_(torch.from_numpy(z0))
            model.decoder[6].bias.copy_(torch.from_numpy(b))
            for level, (near, far) in enumerate(((0.5, 9.0), (0.25, 9.0), (0.125, 9.0))):
                model.codebooks[level][0, 0] = near
                model.codebooks[level][1, 0] = far
        x = np.random.RandomState(0).randn(4, 64).astype(np.float32)
        report = sr.reconstruction_report(sr.model_state(model), x)
        mse = float(np.mean(np.sum((x.astype(np.float64) - b.astype(np.float64)) ** 2, axis=1)))
        sst = float(np.mean(np.sum((x.astype(np.float64) - x.astype(np.float64).mean(axis=0)) ** 2, axis=1)))
        self.assertAlmostEqual(report['mse'], mse, places=9)
        self.assertAlmostEqual(report['sst'], sst, places=9)
        self.assertAlmostEqual(report['r2'], 1 - mse / sst, places=9)
        np.testing.assert_allclose(report['residual_norms'], [1.0, 0.5, 0.25, 0.125], atol=1e-6)


class SourceHygieneTests(Base):
    def test_no_label_or_target_access(self):
        source = inspect.getsource(sr).lower()
        for word in ('validation-queries', 'examples.jsonl', 'support.json', 'history', 'target'):
            self.assertNotIn(word, source)


BLOB_SEED = 0

if __name__ == '__main__':
    unittest.main()
