"""G-HP-01 registry, configs, factory, optimizer and dropout on the TIGER paper-scale anchor A1.

Run inside the frozen-base harness with OMP_NUM_THREADS=1 (CPU only, test split never read).
"""
import unittest
from pathlib import Path

import hp_checks as hc
from test_generative_batched import synthetic, tmpdir


class RegistryTests(unittest.TestCase):
    def test_ra_registry_accepts_the_eight_structures_three_lrs(self):
        hc.check_registry_accepts()

    def test_rb_registry_rejects_off_grid_and_frozen_field_changes(self):
        hc.check_registry_rejects()

    def test_rc_garch_and_ghp_registries_reject_each_other(self):
        hc.check_registry_isolation()

    def test_ofat_committed_configs_change_only_their_knob(self):
        hc.check_committed_configs()


class FactoryTests(unittest.TestCase):
    def test_h1_a1_values_equal_existing_tiger_arch_model_same_seed(self):
        hc.check_a1_factory_equivalence()

    def test_h2_lr_and_dropout_runs_initialise_like_a1(self):
        hc.check_single_knob_init()

    def test_h3_shapes_and_measured_params(self):
        hc.check_shapes()

    def test_h4_optimizer(self):
        hc.check_optimizer()


class DropoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir()
        cls.context = synthetic(Path(cls.dir.name))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_d1_dropout_01_behaves_exactly_like_a1(self):
        hc.check_dropout_a1(self.context)

    def test_d2_dropout_0_disables_every_dropout(self):
        hc.check_dropout_zero()

    def test_d3_dropout_assertions_discriminate_at_a1(self):
        hc.check_dropout_contrast()


if __name__ == '__main__':
    unittest.main()
