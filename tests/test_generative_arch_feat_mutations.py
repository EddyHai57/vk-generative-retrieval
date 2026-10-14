"""GFEAT-02 mutation tests: inject a defect with unittest.mock and demand that the guards named for it turn red.

Each mutant first runs the guards on the real code (control: green), then under the mutation (each guard must raise
AssertionError through its own check). A surviving mutant fails the test. server-safe (synthetic codec, no contract).
Mutations are scoped to the GFEAT-02 class / module (``TigerArchModelSide``, ``generative_arch_feat``) so GFEAT-01's
``TigerModelSide`` is never touched.
"""
import math
import unittest
from unittest import mock

import torch

import arch_feat_checks as fc
from vkgr import generative_arch as ga
from vkgr import generative_arch_feat as gaf
from vkgr import generative_b1_arch as ba
from vkgr import generative_b1_run as b1
from vkgr import generative_batched as gb

REAL = {n: getattr(gaf.TigerArchModelSide, n) for n in ('embed_source', 'side_vector', 'encode', 'forward', '__init__')}


def embed_with(combine):
    """embed_source with the real validation and side vector, combined into the encoder input by ``combine``."""
    def embed_source(self, source, side):
        valid = self._valid(source, side)
        vector = self.side_vector(side, valid)
        pos = self.position(torch.arange(source.shape[1], device=source.device))[None]
        return combine(self, self.embedding(source), vector, pos, math.sqrt(self.d_model))
    return embed_source


def first_token_only(vector):
    out = torch.zeros(vector.shape[0], vector.shape[1] * 4, vector.shape[2], dtype=vector.dtype)
    out[:, ::4] = vector
    return out


def side_columns(order):
    def side_vector(self, side, valid):
        return REAL['side_vector'](self, side[..., list(order)], valid)
    return side_vector


def feedback_bits_reversed(self, side, valid):
    if self.feature != 'f2_feedback':
        return REAL['side_vector'](self, side, valid)
    bits = torch.stack([(side[..., 0] >> (5 - b)) & 1 for b in range(6)], -1).to(self.embedding.weight.dtype) * valid.unsqueeze(-1)
    return bits @ self.feedback.weight


def encode_without_side(self, source, side=None):
    self._valid(source, side)
    padding = source.eq(0); x = self.embed(source)
    for layer in self.transformer.encoder.layers:
        x = layer(x, padding)
    return self.transformer.encoder.norm(x), padding


def encode_side_on_memory(self, source, side=None):
    memory, padding = encode_without_side(self, source, side)
    return memory + self.side_vector(side, self._valid(source, side)).repeat_interleave(4, dim=1), padding


def forward_side_into_decoder(self, source, decoder, side=None):
    memory, padding = self.encode(source, side)
    shift = self.side_vector(side, self._valid(source, side)).mean(1, keepdim=True)
    return self.decode(memory + shift, padding, decoder)


def init_after_extra_draw(self, *args):
    REAL['__init__'](self, *args)
    torch.rand(1)  # a draw before the base initialisation: the side tables created first would do this


def none_through_side_class(codec, config, *, device):
    model = gaf.TigerArchModelSide(codec.vocab_size, *(config[k] for k in ga.SHAPE_KEYS)).to(device=device, dtype=torch.float32)
    report = gaf.init_embeddings(model)
    model.init_report = {'init': 'small', 'std': 0.02, 'embeddings': report, 'arch': ga.arch_report(model)}
    return model


class MutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = cls.codec = fc.codec()
        cls.guards = {
            'none=GARCH-01': lambda: fc.check_none_is_plain_arch(c), 'base stream': lambda: fc.check_base_params_equal_plain(c),
            'side draws': lambda: fc.check_side_tables_follow_base_init(c),
            'inject f2': lambda: fc.check_injection(c, 'f2_feedback'), 'inject f3': lambda: fc.check_injection(c, 'f3_watch_buckets'),
            'scale once': lambda: fc.check_injection_before_scale(c),
            'columns f2': lambda: fc.check_side_vector_columns(c, 'f2_feedback'), 'columns f3': lambda: fc.check_side_vector_columns(c, 'f3_watch_buckets'),
            'encoder input f2': lambda: fc.check_encode_uses_injected_input(c, 'f2_feedback'), 'encoder input f3': lambda: fc.check_encode_uses_injected_input(c, 'f3_watch_buckets'),
            'decoder f2': lambda: fc.check_decoder_untouched(c, 'f2_feedback'),
            'zero=none eval f2': lambda: fc.check_zero_side_equals_none_eval(c, 'f2_feedback'), 'zero=none eval f3': lambda: fc.check_zero_side_equals_none_eval(c, 'f3_watch_buckets'),
            'registry': fc.check_registry, 'runner': fc.check_runner_installs_and_restores, 'gates': fc.check_gates}

    def kill(self, mutant, mutation, *names):
        for name in names:
            self.guards[name]()
        for name in names:
            with self.subTest(mutant=mutant, guard=name), mutation():
                with self.assertRaises(AssertionError, msg=f'{mutant} survived guard {name}'):
                    self.guards[name]()

    def patch(self, name, value):
        return lambda: mock.patch.object(gaf.TigerArchModelSide, name, value)

    # ---- injection position / order / broadcast -------------------------------------------------------------
    def test_m1_side_added_after_sqrt_d(self):
        self.kill('M1 side after x sqrt(d)', self.patch('embed_source', embed_with(lambda m, e, v, p, s: e * s + v.repeat_interleave(4, dim=1) + p)),
                  'inject f2', 'inject f3', 'scale once')

    def test_m2_side_scaled_twice(self):
        self.kill('M2 side x sqrt(d) twice', self.patch('embed_source', embed_with(lambda m, e, v, p, s: (e + s * v.repeat_interleave(4, dim=1)) * s + p)),
                  'inject f2', 'inject f3', 'scale once')

    def test_m3_broadcast_first_token_only(self):
        self.kill('M3 first SID token only', self.patch('embed_source', embed_with(lambda m, e, v, p, s: (e + first_token_only(v)) * s + p)),
                  'inject f2', 'inject f3', 'scale once')

    def test_m4_broadcast_tiled_instead_of_interleaved(self):
        self.kill('M4 repeat (tile) not repeat_interleave', self.patch('embed_source', embed_with(lambda m, e, v, p, s: (e + v.repeat(1, 4, 1)) * s + p)),
                  'inject f2', 'inject f3', 'scale once')

    def test_m5_side_on_position_after_scale(self):
        self.kill('M5 side folded into the position term', self.patch('embed_source', embed_with(lambda m, e, v, p, s: e * s + (p + v.repeat_interleave(4, dim=1)))),
                  'inject f2', 'scale once')

    def test_m6_encoder_ignores_injected_input(self):
        self.kill('M6 encoder runs on the plain embedding', self.patch('encode', encode_without_side), 'encoder input f2', 'encoder input f3')

    def test_m7_side_added_to_encoder_memory(self):
        self.kill('M7 side added after the encoder', self.patch('encode', encode_side_on_memory), 'encoder input f2', 'encoder input f3')

    def test_m8_side_reaches_the_decoder(self):
        self.kill('M8 side shifts the decoder input', self.patch('forward', forward_side_into_decoder), 'decoder f2')

    # ---- feature columns / definitions ---------------------------------------------------------------------
    def test_m9_columns_swapped(self):
        self.kill('M9a F2 reads the timespent column', self.patch('side_vector', side_columns((1, 0, 2))), 'columns f2')
        self.kill('M9b F3 timespent <-> duration', self.patch('side_vector', side_columns((0, 2, 1))), 'columns f3')

    def test_m10_feedback_bit_order(self):
        self.kill('M10 feedback bits reversed', self.patch('side_vector', feedback_bits_reversed), 'columns f2')

    # ---- factory / random stream ---------------------------------------------------------------------------
    def test_m11_side_tables_in_global_stream(self):
        self.kill('M11 no forked RNG for the side tables', lambda: mock.patch.object(gaf, '_side_tables', lambda model, feature: model.add_side_tables(feature)), 'side draws')

    def test_m12_draw_before_base_init(self):
        self.kill('M12 random draw before the base init', self.patch('__init__', init_after_extra_draw), 'base stream', 'side draws')

    def test_m13_none_through_side_class(self):
        self.kill('M13 feature=none builds the side class', lambda: mock.patch.object(gaf.ga, 'make_model_arch', none_through_side_class), 'none=GARCH-01')

    # ---- registry / runner ---------------------------------------------------------------------------------
    def test_m14_registry_lets_a2_or_unregistered_features_through(self):
        self.kill('M14a A2 learning rate registered', lambda: mock.patch.object(gaf, 'REGISTERED', gaf.REGISTERED + ((ga.PAPER_SCALE, .001),)), 'registry')
        self.kill('M14b anchor structure registered', lambda: mock.patch.object(gaf, 'REGISTERED', gaf.REGISTERED + ((ga.ANCHOR_SHAPE, .0003),)), 'registry')
        self.kill('M14c F1 registered', lambda: mock.patch.object(gaf, 'FEATURES', gaf.FEATURES + ('f1_ratio_pool',)), 'registry')

    def test_m15_runner_without_restore(self):
        before = {n: getattr(b1, n) for n in fc.RUNNER_GLOBALS}
        self.guards['runner']()
        try:
            with mock.patch.object(ba, '_restore', lambda saved: None):
                with self.assertRaisesRegex(AssertionError, 'was not restored', msg='M15 survived'):
                    self.guards['runner']()
        finally:
            for name, value in before.items():
                setattr(b1, name, value)

    def test_m16_gate_lets_workers_through(self):
        self.kill('M16 GARCH-01 gate skipped', lambda: mock.patch.object(ba, '_gate', lambda decode, workers: gb.check_decode(decode)), 'gates')


if __name__ == '__main__':
    unittest.main()
