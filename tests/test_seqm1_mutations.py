"""SEQ-M1 mutation tests M1-M30 (design section 6 + review F1/F2): each deliberately wrong implementation must be caught by its named guard.

Same pattern as tests/test_php01_mutations.py: the guard passes on the real code, raises AssertionError under
``mock.patch`` on a module-level seam, and passes again after the revert. Runner-level mutants (M13 M14 M18 M21 and the
config/binding ones) are at the bottom of this file.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_mutations import detected  # noqa: E402
import test_seqm1_model as tm  # noqa: E402
from vkgr import retrieval_seqm1 as m  # noqa: E402


def not_detected_by(case, module, name, replacement, guard):
    """The point of review F1: a mutant that stays invisible to the weak guard (used to show G15a alone is not enough)."""
    with mock.patch.object(module, name, replacement):
        guard()


class MaskMutations(unittest.TestCase):
    def test_m1_causal_mask_reversed(self):
        detected(self, m, 'causal_mask', lambda n, device: torch.tril(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=-1), [tm.guard_mask_semantics])

    def test_m2_causal_arm_without_mask(self):
        detected(self, m, 'causal_mask', lambda n, device: None, [tm.guard_mask_semantics])

    def test_m10_key_padding_mask_dropped(self):
        """Only observable on the bidirectional arm (review N4): left-aligned PAD keys sit to the right of every causal query."""
        detected(self, m, 'key_padding', lambda valid: None, [tm.guard_padding_invariant])


class ReadoutMutations(unittest.TestCase):
    def test_m3_last_takes_the_pad_slot(self):
        detected(self, m, 'pool_last', lambda z, valid: z[:, -1], [tm.guard_readouts])

    def test_m4_mean_divides_by_width(self):
        detected(self, m, 'pool_mean', lambda z, valid: z.sum(1) / z.shape[1], [tm.guard_padding_invariant, tm.guard_readouts])

    def test_m11_multiplicative_pad_mask(self):
        detected(self, m, 'mask_pad', lambda z, valid: z * valid.unsqueeze(-1), [tm.guard_finiteness])

    def test_m12_attention_softmax_sees_pad(self):
        detected(self, m, 'attention_weights', lambda scores, valid: scores.softmax(1), [tm.guard_readouts])


class GateMutations(unittest.TestCase):
    def test_m5_gate_starts_at_0_1(self):
        with mock.patch.object(m, 'GATE_INIT', .1):
            with self.assertRaises(AssertionError):
                tm.guard_zero_step_equals_anchor()
        tm.guard_zero_step_equals_anchor()

    def test_m6_gate_detached(self):
        detected(self, m, 'combine', lambda anchor, g, r: anchor + g.detach() * r, [tm.guard_gradients])

    def test_m16_gate_on_the_anchor_path(self):
        detected(self, m, 'combine', lambda anchor, g, r: g * anchor + r, [tm.guard_zero_step_equals_anchor, tm.guard_readouts])

    def test_m19_anchor_path_not_the_anchor_code(self):
        def own_mean(self, histories, side):
            valid = histories != 0
            return self.event_vectors(histories, side, valid).flip(1).sum(1) / valid.sum(1, keepdim=True)
        detected(self, m.SeqM1IDMean, 'anchor_vector', own_mean, [tm.guard_zero_step_equals_anchor])


class BranchInputMutations(unittest.TestCase):
    def test_m7_order_table_not_in_the_input(self):
        detected(self, m, 'branch_input', lambda model, e, valid: e, [tm.guard_gradients])

    def test_m8_order_ids_from_the_oldest_event(self):
        detected(self, m, 'order_ids', lambda valid: valid.long().cumsum(1), [tm.guard_order_ids_count_from_the_latest_event])

    def test_m17_m24_week_gap_anchored_on_a_constant_week(self):
        """The gap is taken against week 24 (a target-style anchor) instead of the last history event."""
        detected(self, m, 'week_gap', lambda week, valid: (24 - week) * valid, [tm.guard_branch_input_mean_matches_anchor, tm.guard_per_event_oracle])

    def test_m23_branch_input_without_f3_f4(self):
        detected(self, m.SeqM1IDMean, 'event_vectors', lambda self, histories, side, valid: self.items(histories),
                 [tm.guard_branch_consumes_f3_f4, tm.guard_per_event_oracle])

    def test_m28_position_bucket_from_the_forward_index(self):
        """Review F1: the multiset of buckets (hence the MEAN of e) is unchanged, so G15a stays green; only the per-event oracle sees it."""
        def forward(valid):
            from vkgr.retrieval_feat02 import POSITION_EDGES
            rank = valid.long().cumsum(1) - 1
            edges = torch.tensor(POSITION_EDGES, dtype=rank.dtype)
            return (torch.bucketize(rank.clamp_min(0).contiguous(), edges, right=True) + 1) * valid
        not_detected_by(self, m, 'position_bucket', forward, tm.guard_branch_input_mean_matches_anchor)
        detected(self, m, 'position_bucket', forward, [tm.guard_per_event_oracle])

    def test_m29_timespent_column_shifted_by_one(self):
        original = m.timespent_bucket
        detected(self, m, 'timespent_bucket', lambda ts, valid: torch.roll(original(ts, valid), 1, dims=1), [tm.guard_per_event_oracle])


class StructureMutations(unittest.TestCase):
    def test_m9_post_layer_norm(self):
        detected(self, m, 'make_layer', lambda: torch.nn.TransformerEncoderLayer(64, 4, dim_feedforward=256, dropout=.1, activation='relu', batch_first=True, norm_first=False),
                 [tm.guard_structure])

    def test_m20_order_table_drawn_before_the_anchor_tables(self):
        original = m._draw_tables

        def early(model, std):
            model.order_table = m.make_table(m.ORDER_ROWS, m.DIM); m._small_init_d(model, ('order_table',), std)
            original(model, std)
        detected(self, m, '_draw_tables', early, [tm.guard_init_order_and_anchor_tables])

    def test_m25_layer_built_before_the_anchor_tables(self):
        original = m._draw_tables
        detected(self, m, '_draw_tables', lambda model, std: (m.make_layer(), original(model, std))[1], [tm.guard_init_order_and_anchor_tables])

    def test_m25_init_embeddings_called_again_after_the_build(self):
        detected(self, m, '_after_build', lambda model: m.init_embeddings(model, .02), [tm.guard_init_order_and_anchor_tables])

    def test_m22_left_alignment_check_removed(self):
        detected(self, m, 'assert_left_aligned', lambda valid: None, [tm.guard_left_aligned])


class PureMutations(unittest.TestCase):
    def test_m26_pure_gets_the_anchor_path_and_gate(self):
        with mock.patch.dict(m.ARMS, {'pure': ('causal', 'last', True)}):
            with self.assertRaises(AssertionError):
                tm.guard_pure_identity()
        tm.guard_pure_identity()

    def test_m27_pure_reads_the_mean_instead_of_the_last_position(self):
        with mock.patch.dict(m.ARMS, {'pure': ('causal', 'mean', False)}):
            with self.assertRaises(AssertionError):
                tm.guard_pure_identity()
        tm.guard_pure_identity()

    def test_m30_check_skipped(self):
        """Review F2: without _check week=25, decreasing weeks, PAD values ... pass silently (all arms, both entry points)."""
        detected(self, m.SeqM1IDMean, '_check', lambda self, histories, side: histories != 0, [tm.guard_check_on_every_entry])


class CandidateSideMutations(unittest.TestCase):
    def test_m15_item_tower_remembers_the_last_user_batch(self):
        """L9 mutant: items() returns the item vectors plus a trace of the last users() call (candidate side reads history)."""
        from test_seqm1_leak import guard_item_tower_independent_of_users
        guard_item_tower_independent_of_users()
        original_users, original_items = m.SeqM1IDMean.users, m.SeqM1IDMean.items

        def users(self, histories, side=None):
            out = original_users(self, histories, side); object.__setattr__(self, '_trace', out.detach().mean(0)); return out

        def items(self, indices):
            return original_items(self, indices) + getattr(self, '_trace', 0.)
        with mock.patch.object(m.SeqM1IDMean, 'users', users), mock.patch.object(m.SeqM1IDMean, 'items', items):
            with self.assertRaises(AssertionError):
                guard_item_tower_independent_of_users()
        guard_item_tower_independent_of_users()


class RunnerMutations(unittest.TestCase):
    """Runner-level mutants: M13 M14 M18 M21, the holdout-open guard (L6) and the branch-spec binding (G13)."""

    def test_m13_evaluation_path_puts_the_target_into_the_history(self):
        from vkgr import retrieval_feat02_run as fr
        from test_seqm1_leak import guard_export_path_is_target_free, guard_target_fields_do_not_reach_the_user_tower
        original = fr._tensors2
        def leaky(batch, idmap, width, feature):
            return original([{**r, 'history': r['history'][:-1] + [r['target'] if r['target'] in idmap else r['history'][-1]]} for r in batch], idmap, width, feature)
        detected(self, fr, '_tensors2', leaky, [guard_export_path_is_target_free, guard_target_fields_do_not_reach_the_user_tower])

    def test_m14_diagnostics_exported_in_train_mode(self):
        from vkgr import retrieval_seqm1_run as run
        import test_seqm1_run as tr
        detected(self, run, '_to_eval', lambda model: model.train(), [tr.guard_diagnostics_use_eval_mode])

    def test_m18_branch_parameters_missing_from_the_optimizer(self):
        from vkgr import retrieval_seqm1_run as run
        import test_seqm1_run as tr
        detected(self, run, 'make_optimizer', lambda model, config: torch.optim.AdamW([p for n, p in model.named_parameters() if not n.startswith(('layer', 'order_table'))], lr=config['learning_rate']),
                 [tr.guard_optimizer_covers_every_parameter])

    def test_m21_temperature_and_negatives_not_pinned(self):
        from vkgr import retrieval_seqm1_run as run
        import test_seqm1_run as tr
        detected(self, run.php, 'validate_config', lambda value, synthetic=False: value, [tr.guard_config_pins_the_anchor_conditions])

    def test_l6_a_runner_that_opens_a_holdout_path_is_caught(self):
        import tempfile
        from pathlib import Path
        from vkgr import retrieval_feat02_run as fr, retrieval_seqm1_run as run
        import test_seqm1_run as tr
        original = run.make_optimizer

        def reads_holdout(model, config):
            open(Path(tempfile.gettempdir()) / ('seqm1-' + tr.HOLDOUT_FRAGMENT + '-split.json'), 'rb')
            return original(model, config)
        with mock.patch.object(run, 'make_optimizer', reads_holdout):
            with tr.tmp_dir() as tmp, self.assertRaises(AssertionError):
                tr.guard_holdout_never_opened(lambda: run.run_session(Path(tmp) / 'run', fr.synthetic_data(), tr.small('mean', max_epochs=1, min_epochs=1)))

    def test_g13_any_branch_hyper_parameter_breaks_verify(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_seqm1_run as run
        import test_seqm1_run as tr
        data = fr.synthetic_data(); config = tr.small('last')
        variants = {
            'post_ln': lambda: mock.patch.object(m, 'make_layer', lambda: torch.nn.TransformerEncoderLayer(64, 4, dim_feedforward=256, dropout=.1, activation='relu', batch_first=True, norm_first=False)),
            'dropout': lambda: mock.patch.object(m, 'DROPOUT', .2),
            'heads': lambda: mock.patch.object(m, 'HEADS', 2),
            'ffn': lambda: mock.patch.object(m, 'FFN', 128),
            'final_norm_init': lambda: mock.patch.object(m, 'FINAL_NORM_INIT', .2),
            'order_rows': lambda: mock.patch.object(m, 'ORDER_ROWS', 52)}
        with tr.tmp_dir() as tmp:
            value = run.run_session(Path(tmp) / 'run', data, config)
            for name, patch in variants.items():
                with patch():
                    with self.assertRaises(ValueError, msg=name):
                        run.verify_run(Path(tmp) / 'run' / 'manifest.json', value['manifest_sha256'], data, config)
            self.assertEqual(run.verify_run(Path(tmp) / 'run' / 'manifest.json', value['manifest_sha256'], data, config)['status'], 'verified')


if __name__ == '__main__':
    unittest.main()
