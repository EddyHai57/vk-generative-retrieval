"""G6-PREP step 5 mutants M29..M38 of sid_rqvae_d (level count reaching the model, residual chain, export, report): all killed."""
import unittest

from tests.mutation_util import mutated, run_ids
from vkgr import sid_rqvae_d as rd

T = 'tests.test_sid_rqvae_d.'
ALL = [T + 'LevelsThreeEqualsFrozen', T + 'LevelsFour', T + 'Validation']


class Mutants(unittest.TestCase):
    def test_unmutated_selection_is_green(self):
        result = run_ids(ALL)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)
        self.assertGreaterEqual(result.testsRun, 14)

    def kill(self, old, new, ids=None, count=1):
        with mutated(rd, old, new, count):
            result = run_ids(ids or ALL)
        self.assertFalse(result.wasSuccessful(), f'mutant survived: {old!r} -> {new!r}')

    def test_M29_levels_does_not_reach_the_model(self):
        self.kill('for _ in range(levels)])', 'for _ in range(3)])', [T + 'LevelsFour'])

    def test_M30_residual_chain_misses_the_last_level(self):
        self.kill('q = rq.quantize(z, list(self.codebooks))', 'q = rq.quantize(z, list(self.codebooks)[:3])', [T + 'LevelsFour'])

    def test_M31_fit_builds_a_three_level_model(self):
        self.kill('model = RQVAED(K, levels).to(device)', 'model = RQVAED(K, 3).to(device)', [T + 'LevelsFour'])

    def test_M32_export_assumes_three_levels(self):
        self.kill('return _levels(len(indices))', 'return 3', [T + 'LevelsFour'])

    def test_M33_report_has_one_norm_too_few(self):
        self.kill('np.zeros(levels + 1)', 'np.zeros(4)', [T + 'LevelsFour'])

    def test_M34_learning_rate_changed(self):
        self.kill("lr=rq.PREREG['lr']", 'lr=0.3', [T + 'LevelsThreeEqualsFrozen'])

    def test_M35_codebooks_created_before_the_networks(self):
        self.kill("        self.encoder = rq._mlp(dims)\n        self.decoder = rq._mlp(dims[::-1])\n        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(K, rq.PREREG['latent_dim'])) for _ in range(levels)])",
                  "        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(K, rq.PREREG['latent_dim'])) for _ in range(levels)])\n        self.encoder = rq._mlp(dims)\n        self.decoder = rq._mlp(dims[::-1])",
                  [T + 'LevelsThreeEqualsFrozen', T + 'LevelsFour'])

    def test_M36_minimum_levels_off_by_one(self):
        self.kill('MIN_LEVELS = 2', 'MIN_LEVELS = 1', [T + 'Validation'])

    def test_M37_straight_through_removed(self):
        self.kill('zst = z + (q[\'zhat\'] - z).detach()', 'zst = q[\'zhat\']', [T + 'LevelsThreeEqualsFrozen', T + 'LevelsFour'])

    def test_M38_codebook_state_hole_accepted(self):
        self.kill('if indices != list(range(len(indices))):', 'if False:', [T + 'Validation'])


if __name__ == '__main__':
    unittest.main()
