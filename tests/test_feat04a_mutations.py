"""FEAT-04a mutation tests (names M4S / M4A / M4B / M4L as in the frozen card): each deliberately wrong implementation must be
caught by every listed guard. ``detected`` (tests/test_feat02_mutations.py) runs the guards on the real code (green), under
``mock.patch.object`` on a module attribute (must raise AssertionError), and again after the revert (green).

Card F5: M4S-05 feeds a bad source text to ``scan_calls`` (a source edit cannot be expressed as a module-attribute patch);
M4L-03/M4L-04 are test-defined functions whose body contains the forbidden literal, caught by the L7 source guard.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat02_model as tm  # noqa: E402
import test_feat03_model as t3  # noqa: E402
import test_feat04a_model as tmod  # noqa: E402
import test_feat04a_run as trun  # noqa: E402
from test_feat02_mutations import detected  # noqa: E402

CONTROL, A, B = tmod.CONTROL, tmod.A, tmod.B


def guard_target_flip_a():
    trun.guard_target_flip_leaves_feature_unchanged(A)


def guard_target_flip_b():
    trun.guard_target_flip_leaves_feature_unchanged(B)


class SharedMutationTests(unittest.TestCase):
    def test_m4s_01_temperature_hard_coded_to_015(self):
        m = trun.r4(); original = m.train_step

        def hard(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
            return original(model, optimizer, batch, data, stats, epoch, {**config, 'temperature': 0.15}, rng, pool, frequency)
        detected(self, m, 'train_step', hard, [trun.guard_s_temperature_plumbing, trun.guard_s_equiv_h1])

    def test_m4s_02_real_path_accepts_t015_or_control(self):
        m = trun.r4(); original = m.validate_config
        detected(self, m, 'validate_config', lambda value, synthetic=False: original(value, True), [trun.guard_s_config_registry])

    def test_m4s_03_default_device_cuda(self):
        detected(self, trun.r4(), 'DEFAULT_DEVICE', 'cuda', [trun.guard_s_cli_default_device_cpu])

    def test_m4s_04_learning_rate_not_pinned(self):
        m = trun.r4(); original = m.validate_config
        detected(self, m, 'validate_config', lambda value, synthetic=False: original({**value, 'learning_rate': 0.001} if 'learning_rate' in value else value, synthetic), [trun.guard_s_config_registry])

    def test_m4s_05_direct_import_of_a_feat03_name_is_caught_by_the_scan(self):
        tmod.guard_s_module_attribute_calls()
        for bad in ('from .retrieval_feat03 import weighted_mean\n', 'x = 1\n    from .retrieval_feat02 import _small_init\n', 'from .retrieval_feat02_run import _tensors2\n'):
            with self.assertRaises(AssertionError, msg=bad):
                tmod.scan_calls(bad)

    def test_m4s_06_control_pooling_takes_the_weighted_branch(self):
        from vkgr import retrieval_feat04a as mod
        original = mod.make_feat04a_model

        def mutated(variant, content, supported, max_len, seed, author_index=None, vocab_size=None):
            if variant == CONTROL:
                return t3.set_logits(original(A, content, supported, max_len, seed))
            return original(variant, content, supported, max_len, seed, author_index, vocab_size)
        detected(self, mod, 'make_feat04a_model', mutated, [trun.guard_s_equiv_h1, trun.guard_s_equiv_f4_control_t015])


class AMutationTests(unittest.TestCase):
    def test_m4a_01_pooling_falls_back_to_the_uniform_mean(self):
        from vkgr import retrieval_feat03 as f3
        detected(self, f3, 'weighted_mean', lambda x, w: x.sum(1) / (w != 0).sum(1, keepdim=True), [tmod.guard_a_formula, tmod.guard_a_logits_receive_gradient])

    def test_m4a_02_logits_initialised_with_noise(self):
        from vkgr import retrieval_feat03 as f3

        def noisy(model):
            g = torch.Generator().manual_seed(24)
            model.recency_logit_pos = torch.nn.Parameter(torch.randn(7, generator=g) * 0.02)
            model.recency_logit_gap = torch.nn.Parameter(torch.randn(6, generator=g) * 0.02)
        detected(self, f3, 'init_recency_logits', noisy, [tmod.guard_a_start_equals_anchor])

    def test_m4a_03_logits_detached(self):
        from vkgr import retrieval_feat03 as f3
        original = f3.recency_logits
        detected(self, f3, 'recency_logits', lambda pos, gap, pos_ids, gap_ids, valid: original(pos.detach(), gap.detach(), pos_ids, gap_ids, valid), [tmod.guard_a_logits_receive_gradient])

    def test_m4a_04_unsupported_history_events_dropped_from_the_pool(self):
        from vkgr import retrieval_feat03 as f3
        detected(self, f3.Feat03IDMean, '_pool_mask', lambda self, histories, valid: valid & self.supported[histories], [tmod.guard_a_start_equals_anchor])

    def test_m4a_05_pooled_weights_single_where(self):
        from vkgr import retrieval_feat03 as f3

        def single_where(logits, valid):
            top = logits.masked_fill(~valid, float('-inf')).amax(1, keepdim=True).detach()
            return torch.where(valid, torch.exp(logits - top), torch.zeros_like(logits))
        detected(self, f3, 'pooled_weights', single_where, [tmod.guard_a_pooled_weights_pad_cannot_overflow])

    def test_m4a_06_denominator_counts_pad_columns(self):
        from vkgr import retrieval_feat03 as f3
        detected(self, f3, 'weighted_mean', lambda x, w: (x * w.unsqueeze(-1)).sum(1) / w.shape[1], [tmod.guard_a_pad_width_invariance])


class LeakMutationTests(unittest.TestCase):
    def test_m4l_01_target_item_placed_in_the_last_history_slot(self):
        from vkgr import retrieval_feat02_run as fr
        original = fr._tensors2

        def mutated(batch, idmap, width, feature):
            histories, side = original(batch, idmap, width, feature)
            for j, r in enumerate(batch):
                histories[j, len(r['history']) - 1] = idmap.get(r['target'], 1)
            return histories, side
        detected(self, fr, '_tensors2', mutated, [guard_target_flip_a, guard_target_flip_b])

    def test_m4l_02_week_anchor_replaced_by_the_target_week(self):
        from vkgr import retrieval_feat02_run as fr
        original = fr._tensors2

        def mutated(batch, idmap, width, feature):
            histories, side = original(batch, idmap, width, feature)
            for j, r in enumerate(batch):
                side[j, len(r['history']) - 1, 3] = min(r['target_position'][0], 24)
            return histories, side
        detected(self, fr, '_tensors2', mutated, [guard_target_flip_a, guard_target_flip_b])

    def test_m4l_03_feature_function_reads_the_target_literal(self):
        """A test-defined event_vectors whose body reads ['target']: only the L7 source guard can see it (it never runs)."""
        from vkgr import retrieval_feat03 as f3
        original = f3.event_vectors
        TARGETS = {}

        def leaky(model, histories, side, valid):
            return original(model, histories, side, valid) + TARGETS['target']
        detected(self, f3, 'event_vectors', leaky, [trun.guard_l7_source_has_no_target_access])

    def test_m4l_04_diagnostic_function_called_on_the_feature_path(self):
        from vkgr import retrieval_feat03 as f3
        original = f3.event_vectors

        def leaky(model, histories, side, valid):
            support_slices = None            # a feature path must not touch the evaluation helpers
            evaluate_model = support_slices
            return original(model, histories, side, valid)
        detected(self, f3, 'event_vectors', leaky, [trun.guard_l7_source_has_no_target_access])


class BMutationTests(unittest.TestCase):
    def setUp(self):
        from vkgr import retrieval_feat02 as f2
        from vkgr import retrieval_feat04a as mod
        self.f2, self.mod = f2, mod

    def test_m4b_01_author_term_omitted(self):
        f2 = self.f2
        detected(self, self.mod, 'history_author_content', lambda model, h: f2.project_content(model.content_projection, model.content[h]) * (h != 0).unsqueeze(-1), [tmod.guard_b_formula])

    def test_m4b_02_content_projection_omitted(self):
        def only_author(model, h):
            rows = model.author_index[h]
            return model.author_table(rows) * (rows >= 2).unsqueeze(-1) * (h != 0).unsqueeze(-1)
        detected(self, self.mod, 'history_author_content', only_author, [tmod.guard_b_formula])

    def test_m4b_03_oov_mask_dropped(self):
        f2 = self.f2

        def unmasked(model, h):
            return (model.author_table(model.author_index[h]) + f2.project_content(model.content_projection, model.content[h])) * (h != 0).unsqueeze(-1)
        detected(self, self.mod, 'history_author_content', unmasked, [tmod.guard_b_oov_zero_no_grad])

    def test_m4b_04_author_row_taken_from_the_wrong_index_space(self):
        f2 = self.f2

        def wrong(model, h):
            rows = h.clamp_max(model.author_table.num_embeddings - 1)
            return (model.author_table(rows) * (rows >= 2).unsqueeze(-1) + f2.project_content(model.content_projection, model.content[h])) * (h != 0).unsqueeze(-1)
        detected(self, self.mod, 'history_author_content', wrong, [tmod.guard_b_formula])

    def test_m4b_05_item_tower_also_gets_author_and_content(self):
        f2 = self.f2

        def f6_items(self, indices):
            rows = self.author_index[indices]
            return f2.Feat02IDMean.items(self, indices) + self.author_table(rows) * (rows >= 2).unsqueeze(-1) + f2.project_content(self.content_projection, self.content[indices])
        detected(self, self.mod.Feat04aIDMean, 'items', f6_items, [tmod.guard_b_item_tower_untouched, tmod.guard_b_candidate_unsupported_zero])

    def test_m4b_06_content_projection_kaiming_init(self):
        def kaiming(model):
            model.author_table = torch.nn.Embedding(model.vocab_size + 2, 64, padding_idx=0)
            model.content_projection = torch.nn.Linear(64, 64, bias=False)
            with torch.no_grad():
                model.author_table.weight.normal_(0.0, 0.02); model.author_table.weight[0].zero_()
        detected(self, self.mod, 'init_b_tables', kaiming, [tmod.guard_b_init_and_draw_order])

    def test_m4b_07_author_and_projection_drawn_before_p_and_g(self):
        from vkgr import retrieval_feat03 as f3
        from vkgr.training import seed_all
        mod, f2 = self.mod, self.f2
        original = mod.make_feat04a_model

        def reordered(variant, content, supported, max_len, seed, author_index=None, vocab_size=None):
            if variant != B:
                return original(variant, content, supported, max_len, seed, author_index, vocab_size)
            seed_all(seed)
            model = mod.Feat04aIDMean(content, supported, max_len, author_index, f2.AUTHOR_VOCAB_SIZE if vocab_size is None else vocab_size)
            f2.init_embeddings(model)
            f2._small_init(model, model.add_anchor_tables())
            mod.init_b_tables(model)
            f3.draw_variant_tables(model)
            object.__setattr__(model, 'init_report', {'init': 'small', 'std': 0.02, 'embeddings': f2.embedding_report(model)})
            return model
        detected(self, mod, 'make_feat04a_model', reordered, [tmod.guard_s_anchor_tables_bitwise, tmod.guard_b_init_and_draw_order])

    def test_m4b_08_new_parameters_detached(self):
        original = self.mod.history_author_content
        detected(self, self.mod, 'history_author_content', lambda model, h: original(model, h).detach(), [tmod.guard_b_new_params_receive_gradient])

    def test_m4b_09_load_state_dict_skips_the_immutability_check(self):
        detected(self, self.mod.Feat04aIDMean, 'load_state_dict', lambda self, state_dict, strict=True, assign=False: torch.nn.Module.load_state_dict(self, state_dict, strict, assign),
                 [tmod.guard_b_checkpoint_immutable])

    def test_m4b_10_pad_slots_get_the_projection_bias(self):
        original = self.mod.history_author_content
        detected(self, self.mod, 'history_author_content', lambda model, h: original(model, h) + model.content_projection(torch.ones(64)), [tmod.guard_b_pad_width_invariance])

    def test_m4b_11_binding_drops_attr_sha256(self):
        m = trun.r4(); original = m.binding_for4a

        def dropped(*args):
            binding = original(*args); binding.pop('attr_sha256', None)
            return binding
        detected(self, m, 'binding_for4a', dropped, [trun.guard_b_binding_has_attr_sha])

    def test_m4l_03_history_term_reads_the_target_literal(self):
        """Card F5: a test-defined history term whose body contains the forbidden literal; only the L7 source guard can see it."""
        original = self.mod.history_author_content
        TARGETS = {}

        def leaky(model, h):
            return original(model, h) + TARGETS['target']
        detected(self, self.mod, 'history_author_content', leaky, [trun.guard_l7_source_has_no_target_access])

    def test_m4l_04_b_users_calls_a_diagnostic_function(self):
        original = self.mod.Feat04aIDMean.users

        def leaky(self, histories, side=None):
            evaluate_model = None            # the feature path must not touch the evaluation helpers
            return original(self, histories, side)
        detected(self, self.mod.Feat04aIDMean, 'users', leaky, [trun.guard_l7_source_has_no_target_access])


if __name__ == '__main__':
    unittest.main()
