"""G6-PREP step 5: RQ-VAE with a configurable number of levels. levels=3 must equal ``sid_rqvae`` bit for bit."""
import unittest

import numpy as np
import torch

from vkgr import sid_depth as sd
from vkgr import sid_rqvae as rq
from vkgr import sid_rqvae_d as rd

FIT = dict(K=8, seed=42, max_steps=30, batch_size=64, device='cpu', wall_cap_seconds=1e9, log_every=5, eval_every=10,
           check_every=10, probe_from=10, probe_step=20)


def content(n=300, seed=1):
    rows = np.random.RandomState(seed).normal(size=(n, 64)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def same_state(a, b):
    return list(a['state_dict']) == list(b['state_dict']) and all(torch.equal(a['state_dict'][k], b['state_dict'][k]) for k in a['state_dict'])


class LevelsThreeEqualsFrozen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = content()
        cls.frozen = rq.fit(cls.x, **FIT)
        cls.new = rd.fit_d(cls.x, levels=3, **FIT)

    def test_fit_result_is_bit_identical(self):
        for key in ('status', 'step', 'K', 'seed', 'init_digest', 'loss_log', 'usage_log', 'prefix_digest'):
            with self.subTest(key=key):
                self.assertEqual(self.new[key], self.frozen[key])
        self.assertEqual(self.new['status'], 'complete')
        self.assertTrue(same_state(self.new['state'], self.frozen['state']))
        self.assertEqual(rq.state_digest(self.new['state']['state_dict']), rq.state_digest(self.frozen['state']['state_dict']))
        self.assertEqual(set(self.new['state']), {'K', 'state_dict'})

    def test_export_and_reports_are_identical(self):
        self.assertTrue(np.array_equal(rd.export_codes_d(self.new['state'], self.x), rq.export_codes(self.frozen['state'], self.x)))
        self.assertEqual(rd.reconstruction_report_d(self.new['state'], self.x), rq.reconstruction_report(self.frozen['state'], self.x))
        codes = rq.export_codes(self.frozen['state'], self.x)
        self.assertEqual(rd.codebook_usage_d(codes, 8), rq.codebook_usage(codes, 8))

    def test_initialisation_is_identical(self):
        torch.manual_seed(5)
        a = rq.RQVAE(8)
        torch.manual_seed(5)
        b = rd.RQVAED(8, 3)
        self.assertTrue(same_state({'state_dict': a.state_dict()}, {'state_dict': b.state_dict()}))
        batch = torch.from_numpy(self.x[:64])
        rq.init_codebooks(a, batch, 42)
        rq.init_codebooks(b, batch, 42)
        self.assertTrue(same_state({'state_dict': a.state_dict()}, {'state_dict': b.state_dict()}))

    def test_forward_and_loss_are_identical(self):
        torch.manual_seed(2)
        a = rq.RQVAE(8)
        torch.manual_seed(2)
        b = rd.RQVAED(8, 3)
        batch = torch.from_numpy(self.x[:32])
        oa, ob = a(batch), b(batch)
        self.assertTrue(torch.equal(oa['xhat'], ob['xhat']))
        ta, tb = rq.loss_terms(batch, oa['xhat'], oa['q']), rq.loss_terms(batch, ob['xhat'], ob['q'])
        self.assertTrue(torch.equal(ta['total'], tb['total']))

    def test_failure_statuses_match(self):
        for kwargs in (dict(wall_cap_seconds=-1.0), ):
            frozen = rq.fit(self.x, **{**FIT, **kwargs})
            new = rd.fit_d(self.x, levels=3, **{**FIT, **kwargs})
            self.assertEqual((new['status'], new['step']), (frozen['status'], frozen['step']))
            self.assertNotIn('state', new)


class LevelsFour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = content()
        cls.out = rd.fit_d(cls.x, levels=4, **FIT)

    def test_four_codebooks_and_five_digit_export(self):
        self.assertEqual(self.out['status'], 'complete')
        state = self.out['state']
        self.assertEqual([k for k in state['state_dict'] if k.startswith('codebooks')], [f'codebooks.{i}' for i in range(4)])
        codes = rd.export_codes_d(state, self.x)
        self.assertEqual(codes.shape, (300, 4))
        self.assertTrue(codes.min() >= 0 and codes.max() < 8)
        sids = sd.assign_suffix(np.arange(300, dtype=np.uint32), codes)
        self.assertEqual(sids.shape, (300, 5))
        self.assertEqual(len(rd.codebook_usage_d(codes, 8)), 4)

    def test_reconstruction_report_has_five_residual_norms(self):
        report = rd.reconstruction_report_d(self.out['state'], self.x)
        self.assertEqual(len(report['residual_norms']), 5)
        self.assertTrue(all(np.isfinite(v) for v in report['residual_norms']))
        self.assertAlmostEqual(report['r2'], 1 - report['mse'] / report['sst'], places=12)

    def test_the_first_three_levels_start_from_the_levels_three_initialisation(self):
        torch.manual_seed(9)
        three = rd.RQVAED(8, 3)
        torch.manual_seed(9)
        four = rd.RQVAED(8, 4)
        for name, value in three.state_dict().items():
            self.assertTrue(torch.equal(value, four.state_dict()[name]), name)
        batch = torch.from_numpy(self.x[:64])
        rq.init_codebooks(three, batch, 42)
        rq.init_codebooks(four, batch, 42)
        for i in range(3):
            self.assertTrue(torch.equal(three.codebooks[i], four.codebooks[i]), i)

    def test_gradient_routing_holds_for_the_fourth_level(self):
        torch.manual_seed(4)
        model = rd.RQVAED(8, 4)
        batch = torch.from_numpy(self.x[:48])
        rq.init_codebooks(model, batch, 42)
        out = model(batch)
        terms = rq.loss_terms(batch, out['xhat'], out['q'])
        self.assertEqual((len(terms['codebook']), len(terms['commit'])), (4, 4))

        def grads(term):
            model.zero_grad()
            term.backward(retain_graph=True)
            enc = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.encoder.parameters())
            dec = any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.decoder.parameters())
            books = [p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.codebooks]
            return enc, dec, books
        self.assertEqual(grads(terms['codebook'][3]), (False, False, [False, False, False, True]))
        enc, dec, books = grads(terms['commit'][3])
        self.assertTrue(enc)
        self.assertFalse(dec or any(books))
        enc, dec, books = grads(terms['recon'])
        self.assertTrue(enc and dec)
        self.assertFalse(any(books))  # recon reaches the codebooks only through the straight-through path: never directly

    def test_failure_returns_no_model(self):
        failed = rd.fit_d(self.x, levels=4, **{**FIT, 'wall_cap_seconds': -1.0})
        self.assertEqual(failed['status'], 'fail_wall')
        self.assertNotIn('state', failed)

    def test_determinism(self):
        again = rd.fit_d(self.x, levels=4, **FIT)
        self.assertEqual(again['init_digest'], self.out['init_digest'])
        self.assertTrue(same_state(again['state'], self.out['state']))


class Validation(unittest.TestCase):
    def test_levels_range_and_type(self):
        for bad in (0, 1, 9, '3', 3.0, True, None):
            with self.subTest(levels=bad):
                self.assertRaises(ValueError, rd.RQVAED, 8, bad)
                self.assertRaises(ValueError, rd.fit_d, content(), levels=bad, **FIT)

    def test_codebook_size_validation_is_kept(self):
        self.assertRaises(ValueError, rd.RQVAED, 1, 3)
        self.assertRaises(ValueError, rd.RQVAED, 8.0, 3)

    def test_export_infers_levels_from_the_state(self):
        state = rd.fit_d(content(), levels=4, **FIT)['state']
        broken = {'K': state['K'], 'state_dict': {k: v for k, v in state['state_dict'].items() if k != 'codebooks.3'}}
        self.assertEqual(rd.export_codes_d(broken, content()).shape[1], 3)  # the number of levels is read from the state
        hole = {'K': state['K'], 'state_dict': {k: v for k, v in state['state_dict'].items() if k != 'codebooks.1'}}
        self.assertRaises(ValueError, rd.export_codes_d, hole, content())


if __name__ == '__main__':
    unittest.main()
