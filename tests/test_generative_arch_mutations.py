"""GARCH-01 mutation tests (M1-M20): inject a defect into the production code with unittest.mock and demand that the guards
named for it turn red. Every mutant is first run against a control (the guard is green on the real code), then under the
mutation (the guard must raise). A mutant that survives makes the test fail. Guards live in arch_checks.py.
"""
import math
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

import arch_checks as ac
from test_generative_batched import synthetic, tmpdir
from vkgr import generative_arch as ga
from vkgr import generative_b1_arch as ba
from vkgr import generative_b1_run as b1
from vkgr import tiger_arch


class MutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir()
        cls.codec = synthetic(Path(cls.dir.name)).codec
        cls.guards = {
            'E1': lambda: ac.check_same_seed_init(*ac.pair(ac.ANCHOR)), 'E1-multi': lambda: ac.check_same_seed_init(*ac.pair(ac.MULTI)),
            'E2': lambda: ac.check_train_bitwise(*ac.pair(ac.ANCHOR)), 'E3': lambda: ac.check_eval_bitwise_without_fastpath(*ac.pair(ac.ANCHOR)),
            'C1': lambda: ac.check_attention_oracle(tiger_arch.IndependentHeadAttention), 'C3': ac.check_shapes_and_param_count,
            'C4': ac.check_causal_and_padding, 'C5': ac.check_dropout, 'C6': ac.check_layers_independent, 'C7': ac.check_positions_and_untied_output,
            'F1': ac.check_registry, 'F2': lambda: ac.check_factory(cls.codec), 'F4': lambda: ac.check_optimizer(cls.codec),
            'R3': ac.check_globals_restored, 'R5': ac.check_gates}

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def kill(self, mutant, mutation, *names, error=AssertionError):
        """Control first (green on real code), then each named guard must fail through its own assertion (or the named error)."""
        for name in names:
            self.guards[name]()
        for name in names:
            with self.subTest(mutant=mutant, guard=name), mutation():
                with self.assertRaises(error, msg=f'{mutant} survived guard {name}'):
                    self.guards[name]()

    # ---- model ------------------------------------------------------------------------------------------------
    def test_m1_sdpa_scale_one(self):
        self.kill('M1', lambda: mock.patch.object(tiger_arch, 'sdpa', lambda q, k, v, mask, p, causal: F.scaled_dot_product_attention(q, k, v, mask, p, causal, scale=1.0)), 'E2', 'C1')

    def test_m2_head_dim_ignored(self):
        original = tiger_arch.IndependentHeadAttention.__init__
        self.kill('M2', lambda: mock.patch.object(tiger_arch.IndependentHeadAttention, '__init__', lambda self, d, heads, head_dim, dropout: original(self, d, heads, d // heads, dropout)), 'C3')

    def test_m3_output_projection_not_back_to_d_model(self):
        original = tiger_arch.IndependentHeadAttention.__init__

        def wrong(self, d, heads, head_dim, dropout):
            original(self, d, heads, head_dim, dropout)
            self.out_proj = nn.Linear(self.inner, self.inner)
        self.kill('M3', lambda: mock.patch.object(tiger_arch.IndependentHeadAttention, '__init__', wrong), 'C3')

    def test_m4_m5_m6_masks(self):
        forward = tiger_arch.IndependentHeadAttention.forward
        patch = lambda fn: (lambda: mock.patch.object(tiger_arch.IndependentHeadAttention, 'forward', fn))
        self.kill('M4 decoder self-attention not causal', patch(lambda self, q, kv, m=None, c=False: forward(self, q, kv, m, False)), 'C4', 'E2', 'C1')
        self.kill('M5 padding mask dropped', patch(lambda self, q, kv, m=None, c=False: forward(self, q, kv, None, c)), 'C4', 'E2', 'C1')
        self.kill('M6 padding mask inverted', patch(lambda self, q, kv, m=None, c=False: forward(self, q, kv, None if m is None else ~m, c)), 'C4', 'E2', 'C1')

    def test_m7_pre_layer_norm(self):
        def pre_ln(self, x, key_padding_mask):
            h = self.norm1(x)
            x = x + self.dropout1(self.self_attn(h, h, key_padding_mask))
            return x + self.dropout2(self.linear2(self.dropout(tiger_arch.relu(self.linear1(self.norm2(x))))))
        self.kill('M7', lambda: mock.patch.object(tiger_arch.ArchEncoderLayer, 'forward', pre_ln), 'E2', 'E3')

    def test_m8_gelu_instead_of_relu(self):
        self.kill('M8', lambda: mock.patch.object(tiger_arch, 'relu', F.gelu), 'E2')

    def test_m9_query_key_swapped(self):
        project = tiger_arch.IndependentHeadAttention._project

        def swapped(self, a, b, self_attention):
            q, k, v = project(self, a, b, self_attention)
            return (k, q, v) if self_attention else (q, k, v)
        self.kill('M9', lambda: mock.patch.object(tiger_arch.IndependentHeadAttention, '_project', swapped), 'E2', 'C1')

    def test_m10_shared_layers(self):
        self.kill('M10', lambda: mock.patch.object(tiger_arch, '_clones', lambda layer, count: nn.ModuleList([layer] * count)), 'C6', 'E1-multi', 'C3')

    def test_m11_xavier_step_skipped(self):
        self.kill('M11', lambda: mock.patch.object(tiger_arch, '_reset', lambda module: None), 'E1', 'E1-multi')

    def test_m12_dropout_active_in_eval_or_missing_in_train(self):
        real = F.scaled_dot_product_attention
        self.kill('M12a eval still drops', lambda: mock.patch.object(tiger_arch, 'sdpa', lambda q, k, v, m, p, c: real(q, k, v, m, 0.1, c)), 'C5')
        self.kill('M12b train never drops', lambda: mock.patch.object(tiger_arch, 'sdpa', lambda q, k, v, m, p, c: real(q, k, v, m, 0.0, c)), 'C5')

    @staticmethod
    def embed_variant(scale, offset=0):
        """TigerArchModel.embed with the real validation but a different scale / position offset."""
        def embed(self, tokens):
            if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= tiger_arch.POSITIONS:
                raise ValueError('invalid token sequence')
            positions = torch.arange(tokens.shape[1], device=tokens.device) + offset
            return self.embedding(tokens) * scale(self.d_model) + self.position(positions)[None]
        return embed

    def test_m13_embedding_scale(self):
        self.kill('M13a no sqrt(d)', lambda: mock.patch.object(tiger_arch.TigerArchModel, 'embed', self.embed_variant(lambda d: 1.0)), 'E2', 'C7')
        self.kill('M13b sqrt(d) twice', lambda: mock.patch.object(tiger_arch.TigerArchModel, 'embed', self.embed_variant(lambda d: float(d))), 'E2', 'C7')

    def test_m19_position_table_rows_or_offset(self):
        self.kill('M19a 100 rows', lambda: mock.patch.object(tiger_arch, 'POSITIONS', 100), 'C7')
        self.kill('M19a 100 rows: anchor checkpoints no longer load', lambda: mock.patch.object(tiger_arch, 'POSITIONS', 100), 'E2', error=RuntimeError)
        self.kill('M19b position offset', lambda: mock.patch.object(tiger_arch.TigerArchModel, 'embed', self.embed_variant(math.sqrt, offset=1)), 'E2')

    # ---- registry / factory / optimizer -----------------------------------------------------------------------
    def test_m14_optimizer_ignores_weight_decay(self):
        self.kill('M14', lambda: mock.patch.object(ga, 'AdamW', lambda params, lr, weight_decay: torch.optim.AdamW(params, lr=lr)), 'F4')

    def test_m15_registry_lets_unregistered_values_through(self):
        self.kill('M15a unregistered structure', lambda: mock.patch.object(ga, 'PRESETS', {**ga.PRESETS, 'extra': (128, 6, 64, 2, 2, 1024, 0.1)}), 'F1')
        self.kill('M15b history_len 100', lambda: mock.patch.dict(ga.FIXED, {'history_len': 100}), 'F1')
        self.kill('M15c random SID', lambda: mock.patch.dict(ga.FIXED, {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'}), 'F1')

    def test_m18_init_report_arch_segment(self):
        count = tiger_arch.expected_param_count
        self.kill('M18a arch segment missing', lambda: mock.patch.object(ga, 'arch_report', lambda model: {}), 'F2')
        self.kill('M18b wrong parameter count', lambda: mock.patch.object(ga, 'expected_param_count', lambda *a, **k: count(*a, **k) + 1), 'F2', error=ValueError)

    # ---- entry point / script ---------------------------------------------------------------------------------
    def test_m16_runner_without_restore(self):
        before = {n: getattr(b1, n) for n in ac.RUNNER_GLOBALS}
        self.guards['R3']()
        try:
            with mock.patch.object(ba, '_restore', lambda saved: None):
                with self.assertRaisesRegex(AssertionError, 'was not restored', msg='M16 survived guard R3'):
                    self.guards['R3']()
        finally:
            for name, value in before.items():  # the mutant leaves the runner patched: undo it so no test sees the residue
                setattr(b1, name, value)

    def test_m17_gate_lets_workers_and_batches_through(self):
        from vkgr import generative_batched as gb
        self.kill('M17', lambda: mock.patch.object(ba, '_gate', lambda decode, workers: gb.check_decode(decode)), 'R5')

    def test_m20_script_flag_pinned_to_a_constant(self):
        with tmpdir() as tmp:
            script, _, call = ac.parity_world(tmp)
            ac.check_parity_flags(lambda scratch, **kw: call(f'control-{scratch}', **kw))
            for constant in (True, False):
                with self.subTest(constant=constant), mock.patch.object(script, '_equal_steps', lambda first_diff, constant=constant: constant):
                    with self.assertRaisesRegex(AssertionError, 'must be reported (unequal|equal)', msg=f'M20 survived with the flag pinned to {constant}'):
                        ac.check_parity_flags(lambda scratch, constant=constant, **kw: call(f'm20-{constant}-{scratch}', **kw))


if __name__ == '__main__':
    unittest.main()
