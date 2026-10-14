"""GARCH-01 model: equivalence with the stock nn.Transformer TigerModel + the independent-head-dimension capability.

Run inside the frozen-base harness with OMP_NUM_THREADS=1. Bit-identical (torch.equal / digest) everywhere except E4,
the only tolerance, which compares against the stock fused fast path (see arch_checks.py docstring).
"""
import sys
import unittest
from pathlib import Path

import torch

import arch_checks as ac
from test_generative_batched import histories, synthetic, tmpdir


class StockEquivalenceTests(unittest.TestCase):
    """E0-E4 on the anchor shape (1+1 layers, 4x32) and a multi-layer shape (2+2 layers, ff 256)."""
    shapes = {'anchor': ac.ANCHOR, 'multi': ac.MULTI}

    def each(self, guard):
        for name, shape in self.shapes.items():
            with self.subTest(shape=name):
                guard(*ac.pair(shape))

    def test_e0_state_dict_and_parameter_layout_match(self):
        self.each(ac.check_state_layout)

    def test_e1_same_seed_initialisation_is_identical(self):
        self.each(ac.check_same_seed_init)

    def test_e2_train_mode_logits_and_all_gradients_are_bitwise_equal(self):
        self.each(ac.check_train_bitwise)

    def test_e3_eval_is_bitwise_equal_with_stock_fast_path_off(self):
        self.each(ac.check_eval_bitwise_without_fastpath)

    def test_e4_eval_within_1e5_with_stock_fast_path_on(self):
        for name, shape in self.shapes.items():
            with self.subTest(shape=name):
                delta = ac.check_eval_tolerance_with_fastpath(*ac.pair(shape))
                print(f'[E4] {name}: max|dlogit| vs stock fast path = {delta:.3e}', file=sys.stderr)


class ContextEquivalenceTests(unittest.TestCase):
    """E5-E7 need a synthetic G context (codec + examples)."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir()
        cls.context = synthetic(Path(cls.dir.name))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_e5_five_step_training_trajectory_is_bitwise_equal(self):
        for name, shape in (('anchor', ac.ANCHOR), ('multi', ac.MULTI)):
            with self.subTest(shape=name):
                ac.check_training_trajectory(shape, self.context)

    def test_e6_decode_history_matches_stock(self):
        hs = [q['history'] for q in self.context.queries] + histories(self.context, 12, 50, 7)
        for name, shape in (('anchor', ac.ANCHOR), ('multi', ac.MULTI)):
            with self.subTest(shape=name):
                ac.check_decode_history(shape, self.context, hs)

    def test_e7_batched_decode_batch_1_and_3_equal_serial_for_arch_models(self):
        from vkgr.tiger_arch import TigerArchModel
        vocab = self.context.codec.vocab_size
        hs = [q['history'] for q in self.context.queries] + histories(self.context, 12, 50, 9)
        for name, args in (('anchor', (128, 4, 32, 1, 1, 512, 0.0)), ('3x8-on-d32', (32, 3, 8, 2, 2, 64, 0.1)), ('6x64-on-d128', (128, 6, 64, 2, 2, 256, 0.1))):
            with self.subTest(shape=name):
                torch.manual_seed(2)
                ac.check_batched_decode(TigerArchModel(vocab, *args), self.context, hs)


class RealScaleCatalogTests(unittest.TestCase):
    """Beam 100 truncates at every position on a 3000-item catalog (the real one has ~120k items): batched decode must stay bit-equal."""

    def test_e7b_paper_scale_model_on_a_real_scale_catalog(self):
        import random
        from test_generative_batched import large_codec
        from vkgr import generative_batched as gb
        from vkgr.tiger import decode_history
        from vkgr.tiger_arch import TigerArchModel
        codec = large_codec()
        items = [int(i) for i in codec.catalog.ids]
        rng = random.Random(9)
        hs = [[rng.choice(items) for _ in range(rng.choice((1, 2, 5, 50, 50)))] for _ in range(12)]
        torch.manual_seed(0)
        model = TigerArchModel(codec.vocab_size, 128, 6, 64, 4, 4, 1024, 0.1)
        expected = [decode_history(model, codec, h, 100) for h in hs]
        self.assertTrue(all(len(found) == 100 for found, _ in expected))  # truncation really happens
        for size in (1, 4):
            with self.subTest(batch=size):
                self.assertEqual(gb.decode_histories_batched(model, codec, hs, 100, size), expected)


class CapabilityTests(unittest.TestCase):
    """C1-C8: what the stock module cannot do."""

    def test_c1_attention_matches_float64_per_head_oracle(self):
        from vkgr.tiger_arch import IndependentHeadAttention
        ac.check_attention_oracle(IndependentHeadAttention)

    def test_c2_stock_mha_rejects_6_heads_at_d128_but_the_paper_scale_model_runs(self):
        ac.check_independent_heads_run()

    def test_c3_projection_shapes_and_parameter_counts(self):
        ac.check_shapes_and_param_count()

    def test_c4_causality_padding_and_empty_source(self):
        ac.check_causal_and_padding()

    def test_c5_dropout_train_eval_and_reproducibility(self):
        ac.check_dropout()

    def test_c6_layers_are_independent(self):
        ac.check_layers_independent()

    def test_c7_c8_position_table_embed_formula_and_untied_output(self):
        ac.check_positions_and_untied_output()


if __name__ == '__main__':
    unittest.main()
