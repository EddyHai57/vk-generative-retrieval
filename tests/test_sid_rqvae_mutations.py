"""GTOK-01 FEAT-001: every mutant of the RQ-VAE core must be killed by tests/test_sid_rqvae.py."""
import io
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from vkgr import sid_rqvae as rq

T = 'tests.test_sid_rqvae.'
LOSS = [T + 'Quantize.test_loss_terms_hand_computed_beta_025']
ROUTING = [T + 'Quantize.test_gradient_routing']
QUANT = [T + 'Quantize.test_quantize_nearest_per_level_residual_chain_hand_computed']


def run_ids(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ids)
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


def assert_killed(case, patch_target, replacement, test_ids):
    with mock.patch(patch_target, replacement):
        result = run_ids(test_ids)
    case.assertFalse(result.wasSuccessful(), f'mutant survived: {patch_target}')


def make_loss(beta_on_codebook=False, drop_codebook=False, feature_mean=False):
    def loss_terms(x, xhat, q, beta=rq.PREREG['beta']):
        reduce = (lambda t: t.mean(1).mean()) if feature_mean else (lambda t: t.sum(1).mean())
        recon = reduce((x - xhat) ** 2)
        codebook, commit = [], []
        for residual, embed in zip(q['residuals'], q['embeds']):
            term = reduce((residual.detach() - embed) ** 2)
            if drop_codebook:
                term = term * 0
            elif beta_on_codebook:
                term = beta * term
            codebook.append(term)
            commit.append(beta * reduce((residual - embed.detach()) ** 2))
        return {'recon': recon, 'codebook': codebook, 'commit': commit,
                'total': recon + sum(codebook) + sum(commit)}
    return loss_terms


def make_quantize(subtract=True, detach=True):
    def quantize(z, codebooks):
        residual = z
        residuals, codes, embeds = [residual], [], []
        for codebook in codebooks:
            index = rq._nearest(residual, codebook)
            embed = codebook[index]
            codes.append(index)
            embeds.append(embed)
            if subtract:
                residual = residual - (embed.detach() if detach else embed)
            residuals.append(residual)
        return {'codes': codes, 'embeds': embeds, 'residuals': residuals, 'zhat': sum(embeds)}
    return quantize


def no_straight_through(self, x):
    z = self.encoder(x)
    q = rq.quantize(z, list(self.codebooks))
    return {'z': z, 'q': q, 'zst': q['zhat'], 'xhat': self.decoder(q['zhat'])}


def argmax_nearest(r, codebook):
    distance = (r * r).sum(1, keepdim=True) - 2 * (r @ codebook.T) + (codebook * codebook).sum(1)[None, :]
    return distance.argmax(1)


_real_init = rq.RQVAE.__init__


def shared_codebook_init(self, K):
    _real_init(self, K)
    shared = self.codebooks[0]
    self.codebooks = nn.ParameterList([shared, shared, shared])


def strict_usage_gate(used, K):
    numerator, denominator = rq.PREREG['usage_gate_num_den']
    return used * denominator > K * numerator


def row_order_fourth_code(ids, codes):
    ids = np.asarray(ids)
    codes = np.asarray(codes)
    rows = np.arange(len(ids))
    order = np.lexsort((rows, codes[:, 2], codes[:, 1], codes[:, 0]))
    sorted_codes = codes[order]
    starts = np.r_[True, np.any(sorted_codes[1:] != sorted_codes[:-1], axis=1)]
    pos = np.arange(len(ids))
    rank = pos - np.maximum.accumulate(np.where(starts, pos, 0))
    out = np.zeros((len(ids), 4), dtype=np.uint32)
    out[order, :3] = sorted_codes
    out[order, 3] = rank
    return out


def full_data_first_batch(X, order, batch_size):
    return X


class Mutants(unittest.TestCase):
    def test_unmutated_suite_passes(self):
        result = run_ids([T + 'Quantize', T + 'KMeansInit', T + 'Export', T + 'FourthCode', T + 'Reports'])
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)

    def test_mutants_are_killed(self):
        cases = {
            'M1 beta on codebook term': ('vkgr.sid_rqvae.loss_terms', make_loss(beta_on_codebook=True), LOSS),
            'M2 codebook term removed': ('vkgr.sid_rqvae.loss_terms', make_loss(drop_codebook=True), LOSS + ROUTING),
            'M3 no straight-through': ('vkgr.sid_rqvae.RQVAE.forward', no_straight_through,
                                       [T + 'Quantize.test_straight_through_value_and_gradient']),
            'M4 residual not subtracted': ('vkgr.sid_rqvae.quantize', make_quantize(subtract=False), QUANT),
            'M5 argmin -> argmax': ('vkgr.sid_rqvae._nearest', argmax_nearest, QUANT),
            'M6 one shared codebook': ('vkgr.sid_rqvae.RQVAE.__init__', shared_codebook_init,
                                       [T + 'Architecture.test_architecture_shapes_and_param_counts']),
            'M7 init on the full data': ('vkgr.sid_rqvae._first_batch', full_data_first_batch,
                                         [T + 'KMeansInit.test_codebook_init_uses_first_batch_only_and_sequential_levels']),
            'M8 usage gate > instead of >=': ('vkgr.sid_rqvae.usage_gate', strict_usage_gate,
                                              [T + 'Reports.test_codebook_usage_and_usage_gate_integer_boundary']),
            'M9 fourth code by row order': ('vkgr.sid_rqvae.assign_fourth_code', row_order_fourth_code,
                                            [T + 'FourthCode.test_assign_fourth_code_canonical_order_independent_and_check_codec_accepts']),
            'M10 recon mean over features': ('vkgr.sid_rqvae.loss_terms', make_loss(feature_mean=True), LOSS),
            'M11 float32 canonical export': ('vkgr.sid_rqvae.EXPORT_DTYPE', torch.float32,
                                             [T + 'Export.test_export_codes_float64_matches_float32_on_separated_data_and_tie_lowest_index']),
            'M12 residual chain without detach': ('vkgr.sid_rqvae.quantize', make_quantize(detach=False), ROUTING),
        }
        self.assertEqual(len(cases), 12)
        for name, (target, replacement, ids) in cases.items():
            with self.subTest(mutant=name):
                assert_killed(self, target, replacement, ids)


if __name__ == '__main__':
    unittest.main()
