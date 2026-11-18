"""Only the new sampling/loss contracts, with hand-computable controls."""
import importlib
import math
import unittest
import numpy as np
import torch

try:
    subject = importlib.import_module('lsvd.negative_contrasts')
except ModuleNotFoundError:
    subject = None


class NegativeContrastTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(subject, 'negative contrast implementation is missing')

    def test_corrected_formula_and_detached_weight_gradient(self):
        positive = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        negative = torch.tensor([[0.1, -0.4]], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
        mask = torch.ones_like(negative, dtype=torch.bool)
        got = subject.corrected_logq_rows(positive, negative, q, mask)
        partition = math.exp(0.1) / .25 + math.exp(-.4) / .75
        multiplier = 1 - math.exp(.7) / (math.exp(.7) + partition / 2)
        expected = multiplier * (-.7 + math.log(partition))
        self.assertAlmostEqual(float(got.detach()), expected, places=12)
        got.sum().backward()
        self.assertAlmostEqual(float(positive.grad), -multiplier, places=12)
        expected_gradient = multiplier * (math.exp(.1) / .25) / partition
        self.assertAlmostEqual(float(negative.grad[0, 0]), expected_gradient, places=12)

    def test_known_positives_duplicate_targets_and_conditional_mass(self):
        ids = np.array([10, 10, 20, 30])
        mask = subject.negative_mask(ids, [{10, 30}, {10}, {20}], ids[:3])
        np.testing.assert_array_equal(mask, [[False, False, True, False],
                                           [False, False, True, True],
                                           [True, True, False, True]])
        counts = {10: 6, 20: 3, 30: 1}
        q, mass = subject.proposal_values(ids, [{10, 30}, {10}, {20}], counts)
        np.testing.assert_allclose(q, [.6, .6, .3, .1])
        np.testing.assert_allclose(mass, [.7, .6, .3])

    def test_mixed_formula_zero_frequency_actual_tail_batch_and_weights(self):
        users = torch.eye(2, dtype=torch.float64)
        items = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
        ids = np.array([10, 20, 99])
        q = np.array([.75, .25, 0.])
        weights = torch.tensor([1., .25], dtype=torch.float64)
        mask = subject.negative_mask(ids, [{10}, {20, 99}], ids[:2])
        loss, skipped = subject.contrast_loss(users, items, q, weights, 1., mask,
                                             'mixed-uniform1024', catalog_size=100)
        scores = torch.nn.functional.normalize(users, dim=-1) @ torch.nn.functional.normalize(items, dim=-1).T
        mixed = (2 / 3) * torch.tensor(q) + (1 / 3) / 100
        logits = scores - mixed.log()[None]
        logits[0, 0] = scores[0, 0]
        logits[1, 1] = scores[1, 1]
        allowed = torch.tensor(mask)
        allowed[torch.arange(2), torch.arange(2)] = True
        manual = torch.nn.functional.cross_entropy(logits.masked_fill(~allowed, -torch.inf), torch.arange(2), reduction='none')
        expected = (manual * weights).sum() / weights.sum()
        torch.testing.assert_close(loss, expected)
        self.assertEqual(skipped, 0)
        self.assertGreater(float(mixed[-1]), 0)

    def test_corrected_all_known_mass_and_no_negatives(self):
        users = torch.eye(2, requires_grad=True)
        items = torch.eye(2, requires_grad=True)
        mask = np.zeros((2, 2), dtype=bool)
        loss, skips = subject.contrast_loss(users, items, [.5, .5], torch.ones(2), .1,
                                            mask, 'corrected-logq', excluded_mass=[1., 1.])
        self.assertEqual(skips, 2)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(users.grad).all())

    def test_corrected_conditional_normalization_and_target_weights(self):
        users = torch.eye(2)
        items = torch.eye(2)
        mask = np.array([[False, True], [True, False]])
        mass = np.array([.75, .25])
        q = np.array([.75, .25])
        weights = torch.tensor([1., .25])
        loss, skips = subject.contrast_loss(users, items, q, weights, 1., mask,
                                            'corrected-logq', excluded_mass=mass)
        # The sole allowed negative has conditional probability one in each row.
        row = (1 - math.exp(1) / (math.exp(1) + 1)) * -1
        self.assertAlmostEqual(float(loss.detach()), row, places=10)
        self.assertEqual(skips, 0)


if __name__ == '__main__':
    unittest.main()
