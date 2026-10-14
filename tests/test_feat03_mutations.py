"""FEAT-03 mutation tests: FEAT-002 part (M13, M14, M16, M18, M33[q1a,q1b], M35[q1a]), FEAT-003 part (Q2: M17, M19-M24, M35[q2], M36, M38),
FEAT-004 part (Q3: M26a, M26b, M27-M30, M32a, M32b, M33[q3], M34) and FEAT-005 part (needs the runner: M15, M25, M31, M37, CLI default device;
plus the completeness check of the whole table M13-M38).

Each deliberately wrong implementation must be caught by every listed guard: the guard is green on the real code, raises
AssertionError under ``mock.patch.object`` on a module-level seam, and is green again after the revert (``detected``).
Same-week reorder is NOT a guard here: F4 and Q1b are equal up to rounding under it.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat03_model as t3  # noqa: E402
import test_feat03_run as t3r  # noqa: E402
from test_feat02_mutations import detected  # noqa: E402


def fixed_table(rows, seed):
    """A table of the right shape with a fixed small-init content (the dropped table of a Q1 variant does not exist on the model)."""
    table = torch.nn.Embedding(rows, 64, padding_idx=0)
    with torch.no_grad():
        table.weight.normal_(0.0, 0.02, generator=torch.Generator().manual_seed(seed)); table.weight[0].zero_()
    return table


class Q1MutationTests(unittest.TestCase):
    def test_m13_q1a_also_adds_the_week_gap_table(self):
        """Q1a degenerates to F4: G added next to P."""
        from vkgr import retrieval_feat02 as f2, retrieval_feat03 as m
        original, extra = m.event_vectors, fixed_table(len(f2.WEEKGAP_EDGES) + 2, 13)
        mutated = lambda model, histories, side, valid: original(model, histories, side, valid) + extra(f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid))  # noqa: E731
        detected(self, m, 'event_vectors', mutated, [t3.guard_q1a_ignores_week_column, t3.guard_q1a_with_zero_position_table_equals_f3])

    def test_m14_q1b_also_adds_the_position_table(self):
        """Q1b degenerates to F4: P added next to G."""
        from vkgr import retrieval_feat02 as f2, retrieval_feat03 as m
        original, extra = m.event_vectors, fixed_table(len(f2.POSITION_EDGES) + 2, 14)
        mutated = lambda model, histories, side, valid: original(model, histories, side, valid) + extra(f2.position_bucket(valid))  # noqa: E731
        detected(self, m, 'event_vectors', mutated, [t3.guard_q1b_formula, t3.guard_q1b_with_zero_gap_table_equals_f3])

    def test_m16_week_gap_bucket_constant_zero(self):
        from vkgr import retrieval_feat02 as f2
        detected(self, f2, 'weekgap_bucket', lambda gap, valid: torch.zeros_like(gap),
                 [t3.guard_q1b_formula, t3.guard_q1b_changing_history_week_changes_feature])

    def test_m18_only_the_selected_table_is_drawn(self):
        """The random stream shifts: the kept table no longer equals the F4 table of the same seed."""
        from vkgr import retrieval_feat02 as f2, retrieval_feat03 as m
        original = m.draw_variant_tables
        rows = {t3.Q1A: ('position_table', len(f2.POSITION_EDGES) + 2), t3.Q1B: ('weekgap_table', len(f2.WEEKGAP_EDGES) + 2)}

        def only_selected(model):
            if model.variant not in rows:
                return original(model)
            name, count = rows[model.variant]
            setattr(model, name, torch.nn.Embedding(count, 64, padding_idx=0)); f2._small_init(model, (name,))
        detected(self, m, 'draw_variant_tables', only_selected, [t3.guard_q1_tables_equal_f4_tables_bitwise])

    def _detached(self, target, guard):
        from vkgr import retrieval_feat02 as f2, retrieval_feat03 as m
        from vkgr.retrieval_features import ratio_bucket, timespent_bucket

        def mutated(model, histories, side, valid):
            x = model.items(histories)
            x = x + model.timespent_bucket(timespent_bucket(side[..., 1], valid))
            x = x + model.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
            if model.variant == t3.Q1A:
                term = model.position_table(f2.position_bucket(valid))
            else:
                term = model.weekgap_table(f2.weekgap_bucket(f2.week_gap(side[..., 3], valid), valid))
            return x + (term.detach() if model.variant == target else term)
        detected(self, m, 'event_vectors', mutated, [guard])

    def test_m33_q1a_new_table_output_detached(self):
        self._detached(t3.Q1A, t3.guard_q1a_new_table_receives_gradient)

    def test_m33_q1b_new_table_output_detached(self):
        self._detached(t3.Q1B, t3.guard_q1b_new_table_receives_gradient)

    def test_m35_q1a_position_bucket_constant_one(self):
        from vkgr import retrieval_feat02 as f2
        detected(self, f2, 'position_bucket', lambda valid: valid.long(), [t3.guard_q1a_formula])


class Q2MutationTests(unittest.TestCase):
    """FEAT-003 part: M17, M19, M20, M21, M22, M23, M24, M35[q2], M36, M38. Only guards that really raise under the patch are listed."""

    def test_m17_position_counted_from_sequence_start(self):
        from vkgr import retrieval_feat02 as f2

        def from_start(valid):
            rank = (valid.long().cumsum(1) - 1).clamp_min(0)
            edges = torch.tensor(f2.POSITION_EDGES, dtype=rank.dtype)
            return (torch.bucketize(rank.contiguous(), edges, right=True) + 1) * valid
        detected(self, f2, 'position_bucket', from_start, [t3.tm.guard_position_bucket_boundaries, t3.guard_q2_recent_event_gets_bucket1_logit])

    def test_m19_weighted_sum_divided_by_n_not_by_sum_of_weights(self):
        from vkgr import retrieval_feat03 as m
        detected(self, m, 'weighted_mean', lambda x, w: (x * w.unsqueeze(-1)).sum(1) / (w != 0).sum(1, keepdim=True), [t3.guard_q2_formula])

    def test_m20_pad_slots_get_weight(self):
        """The outer where is dropped: PAD weight is exp(0) = 1 (finite), so only the exactly-0 assertion sees it directly."""
        from vkgr import retrieval_feat03 as m

        def no_outer_where(logits, valid):
            top = logits.masked_fill(~valid, float('-inf')).amax(1, keepdim=True).detach()
            return torch.exp(torch.where(valid, logits, top) - top)
        detected(self, m, 'pooled_weights', no_outer_where,
                 [t3.guard_q2_pad_width_invariance, t3.guard_q2_zero_logits_equal_f4_bitwise, t3.guard_q2_pooled_weights_pad_cannot_overflow])

    def test_m21_pool_mask_drops_unsupported_events(self):
        from vkgr import retrieval_feat03 as m
        detected(self, m.Feat03IDMean, '_pool_mask', lambda self, histories, valid: valid & self.supported[histories], [t3.guard_q2_zero_logits_equal_f4_bitwise])

    def test_m22_logits_detached(self):
        from vkgr import retrieval_feat03 as m
        original = m.recency_logits
        detected(self, m, 'recency_logits', lambda pos, gap, pos_ids, gap_ids, valid: original(pos.detach(), gap.detach(), pos_ids, gap_ids, valid),
                 [t3.guard_q2_logits_receive_gradient])

    def test_m23_position_logit_indexed_by_week_gap_ids(self):
        """Only this direction: the reverse (gap logit indexed by position ids) would raise IndexError, not AssertionError."""
        from vkgr import retrieval_feat03 as m
        detected(self, m, 'recency_logits',
                 lambda pos, gap, pos_ids, gap_ids, valid: pos[(gap_ids - 1).clamp_min(0)] + gap[(gap_ids - 1).clamp_min(0)], [t3.guard_q2_formula])

    def test_m24_logits_initialised_with_noise(self):
        from vkgr import retrieval_feat03 as m

        def noisy(model):
            g = torch.Generator().manual_seed(24)
            model.recency_logit_pos = torch.nn.Parameter(torch.randn(7, generator=g) * 0.02)
            model.recency_logit_gap = torch.nn.Parameter(torch.randn(6, generator=g) * 0.02)
        detected(self, m, 'init_recency_logits', noisy, [t3.guard_q2_init_is_exact_zero])

    def test_m35_q2_position_bucket_constant_one(self):
        from vkgr import retrieval_feat02 as f2
        detected(self, f2, 'position_bucket', lambda valid: valid.long(), [t3.guard_q2_formula])

    def test_m36_week_gap_bucket_constant_zero(self):
        from vkgr import retrieval_feat02 as f2
        detected(self, f2, 'weekgap_bucket', lambda gap, valid: torch.zeros_like(gap), [t3.guard_q2_formula])

    def test_m38_single_where_in_pooled_weights(self):
        """Forward result is identical to the double where; only the backward pass (0 * inf = NaN) differs."""
        from vkgr import retrieval_feat03 as m

        def single_where(logits, valid):
            top = logits.masked_fill(~valid, float('-inf')).amax(1, keepdim=True).detach()
            return torch.where(valid, torch.exp(logits - top), torch.zeros_like(logits))
        detected(self, m, 'pooled_weights', single_where, [t3.guard_q2_pooled_weights_pad_cannot_overflow])
        with mock.patch.object(m, 'pooled_weights', single_where):                        # the other guards cannot see M38 (documented, not listed)
            t3.guard_q2_formula(); t3.guard_q2_zero_logits_equal_f4_bitwise()


class Q3MutationTests(unittest.TestCase):
    """FEAT-004 part: M26a, M26b, M27, M28, M29, M30, M32a, M32b, M33[q3], M34. M31 needs the runner (FEAT-005)."""

    def test_m26a_terms_scaled_by_width_over_n(self):
        """User-level term added on PAD slots before pooling: equals terms * W / n (W = tensor width)."""
        from vkgr import retrieval_feat03 as m
        original = m.user_level_terms
        detected(self, m, 'user_level_terms', lambda model, side, valid: original(model, side, valid) * side.shape[1] / valid.sum(1, keepdim=True),
                 [t3.guard_q3_pad_width_invariance, t3.guard_q3_formula])

    def test_m26b_terms_divided_by_n(self):
        from vkgr import retrieval_feat03 as m
        original = m.user_level_terms
        detected(self, m, 'user_level_terms', lambda model, side, valid: original(model, side, valid) / valid.sum(1, keepdim=True), [t3.guard_q3_formula])

    def test_m27_length_is_the_tensor_width(self):
        from vkgr import retrieval_feat03 as m

        def width_as_n(model, side, valid):
            n = torch.full_like(valid.sum(1), side.shape[1])
            return model.len_table(m.length_bucket(n)) + model.act_table(m.activity_bucket(m.distinct_weeks(side[..., 3], valid)))
        detected(self, m, 'user_level_terms', width_as_n, [t3.guard_q3_pad_width_invariance])

    def test_m28_active_weeks_counted_as_span(self):
        from vkgr import retrieval_feat03 as m
        detected(self, m, 'distinct_weeks', lambda week, valid: (week * valid).amax(1) - week[:, 0] + 1, [t3.guard_q3_distinct_weeks_not_span])

    def test_m29_distinct_weeks_counts_the_pad_zero(self):
        from vkgr import retrieval_feat03 as m
        detected(self, m, 'distinct_weeks', lambda week, valid: 1 + (week[:, 1:] != week[:, :-1]).sum(1), [t3.guard_q3_distinct_weeks_ignores_pad])

    def test_m30_absolute_last_week_used_for_the_bucket(self):
        from vkgr import retrieval_feat03 as m

        def last_week(model, side, valid):
            return model.len_table(m.length_bucket(valid.sum(1))) + model.act_table(m.activity_bucket((side[..., 3] * valid).amax(1)))
        detected(self, m, 'user_level_terms', last_week, [lambda: t3.guard_week_shift_invariance(t3.Q3)])

    def _drift(self, name, edges):
        """Every single-edge drift (+1 / -1 that stays strictly increasing) must be caught by both guards, which cover BOTH tuples."""
        from vkgr import retrieval_feat03 as m
        guards = [t3.guard_q3_constants_match_audit, t3.guard_q3_bucket_boundaries]
        for i in range(len(edges)):
            for delta in (-1, 1):
                drifted = list(edges); drifted[i] += delta
                if all(a < b for a, b in zip(drifted, drifted[1:])) and drifted[0] >= 1:
                    detected(self, m, name, tuple(drifted), guards)

    def test_m32a_length_edges_drift(self):
        self._drift('LEN_EDGES', t3.AUDIT_LEN_EDGES)

    def test_m32b_activity_edges_drift(self):
        self._drift('ACT_EDGES', t3.AUDIT_ACT_EDGES)

    def test_m33_q3_user_level_terms_detached(self):
        from vkgr import retrieval_feat03 as m
        original = m.user_level_terms
        detected(self, m, 'user_level_terms', lambda model, side, valid: original(model, side, valid).detach(), [t3.guard_q3_new_tables_receive_gradient])
        with mock.patch.object(m, 'user_level_terms', lambda model, side, valid: original(model, side, valid).detach()):
            with self.assertRaises(AssertionError):
                t3.guard_new_tables_receive_gradient(t3.Q3)                       # the generic [q1a,q1b,q3] entry point routes to the same guard

    def test_m34_tables_zeroed_by_small_init(self):
        """The trap in K6: _small_init zeroes weight[None] when padding_idx is None, i.e. the whole table."""
        from vkgr import retrieval_feat02 as f2, retrieval_feat03 as m

        def zeroed(model):
            model.len_table = torch.nn.Embedding(len(m.LEN_EDGES) + 1, 64); model.act_table = torch.nn.Embedding(len(m.ACT_EDGES) + 1, 64)
            f2._small_init(model, ('len_table', 'act_table'))
        detected(self, m, 'init_user_activity_tables', zeroed, [t3.guard_q3_tables_init])

    def test_m34_wrong_std(self):
        from vkgr import retrieval_feat03 as m

        def wide(model):
            for name, rows in (('len_table', len(m.LEN_EDGES) + 1), ('act_table', len(m.ACT_EDGES) + 1)):
                setattr(model, name, torch.nn.Embedding(rows, 64))
                with torch.no_grad():
                    getattr(model, name).weight.normal_(0.0, 0.2)
        detected(self, m, 'init_user_activity_tables', wide, [t3.guard_q3_tables_init])


def clamped_target_week_mutant(original):
    """Anchor leak: the last valid slot's week is replaced by the (clamped to <= 24) target week, as M3 of FEAT-02 does."""
    def mutated(batch, idmap, width, feature):
        histories, side = original(batch, idmap, width, feature)
        for j, r in enumerate(batch):
            side[j, len(r['history']) - 1, 3] = min(r['target_position'][0], 24)
        return histories, side
    return mutated


class RunnerMutationTests(unittest.TestCase):
    """FEAT-005 part: M15 (Q1b), M25 (Q2), M31 (Q3), M37 and the CLI default device. The patch point of M15/M25/M31 is feat02_run._tensors2,
    reached through the module attribute ``fr._tensors2`` that retrieval_feat03_run._tensors3 calls at run time (K2)."""

    def _anchor_leak(self, variant):
        from vkgr import retrieval_feat02_run as fr
        detected(self, fr, '_tensors2', clamped_target_week_mutant(fr._tensors2), [lambda: t3r.guard_target_week_change_leaves_feature_unchanged(variant)])

    def test_m15_q1b_week_gap_anchored_on_the_target_week(self):
        self._anchor_leak(t3.Q1B)

    def test_m25_q2_week_gap_anchored_on_the_target_week(self):
        self._anchor_leak(t3.Q2)

    def test_m31_q3_target_week_joins_the_distinct_week_count(self):
        """The crafted fixture (last two events share a week < 24, a = edge - 1) makes the leak change the user vector, not only the side tensor."""
        from vkgr import retrieval_feat02_run as fr, retrieval_feat03 as m
        self._anchor_leak(t3.Q3)
        data = fr.synthetic_data(); rows, _ = fr._view(data, 50)
        crafted = t3r.crafted_q3_rows(rows)[:30]; model = t3r.model3(data, t3.Q3); width = t3r.width_of(crafted)
        base = model.users(*t3r.r3()._tensors3(crafted, data['idmap'], width, t3.Q3))
        moved = [{**r, 'target_position': [24, 5]} for r in crafted]
        with mock.patch.object(fr, '_tensors2', clamped_target_week_mutant(fr._tensors2)):
            self.assertFalse(torch.equal(model.users(*t3r.r3()._tensors3(moved, data['idmap'], width, t3.Q3)), base))
            self.assertTrue({m.ACT_EDGES[0] - 1} <= {len({p[0] for p in r['history_positions']}) for r in crafted})

    def test_m37_needs_week_misses_the_user_activity_variant(self):
        m = t3r.r3()
        detected(self, m, 'NEEDS_WEEK', frozenset(m.NEEDS_WEEK - {t3.Q3}), [t3r.guard_tensors3_week_column_wiring])
        for dropped in (t3.Q1B, t3.Q2, t3.CONTROL):                                        # every member is wired, not only Q3
            detected(self, m, 'NEEDS_WEEK', frozenset(m.NEEDS_WEEK - {dropped}), [t3r.guard_tensors3_week_column_wiring])
        detected(self, m, 'NEEDS_WEEK', frozenset(m.NEEDS_WEEK | {t3.Q1A}), [t3r.guard_tensors3_week_column_wiring])

    def test_cli_default_device_cuda_turns_the_default_assertion_red(self):
        m = t3r.r3()
        detected(self, m, 'DEFAULT_DEVICE', 'cuda', [t3r.guard_cli_default_device_is_cpu])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(m, 'DEFAULT_DEVICE', 'cuda'), mock.patch('torch.cuda.is_available', return_value=False):
            Path(tmp, 'config.json').write_text(json.dumps(t3r.config3(feature=t3.Q1A)))
            code, out, _ = t3r.run_cli(['synthetic-train', '--config', str(Path(tmp) / 'config.json'), '--output', str(Path(tmp) / 'run')])
            self.assertEqual((code, out), (2, ''))                                           # the mutant cannot even run locally: the default would break every CPU use


class MutationTableTests(unittest.TestCase):
    """Completeness of the whole table M13-M38 (28 entries: M26 and M32 are split in a/b; M33 is run for q1a, q1b, q3; M35 for q1a, q2)."""

    def test_all_28_entries_exist_as_tests(self):
        import re
        names = []
        for cls in (Q1MutationTests, Q2MutationTests, Q3MutationTests, RunnerMutationTests):
            names += unittest.TestLoader().getTestCaseNames(cls)
        found = {re.match(r'test_m(\d+[ab]?)_', n).group(1) for n in names if re.match(r'test_m\d+', n)}
        expected = {str(i) for i in range(13, 39) if i not in (26, 32)} | {'26a', '26b', '32a', '32b'}
        self.assertEqual(len(expected), 28)
        self.assertEqual(found, expected)
        for part in ('test_m33_q1a_', 'test_m33_q1b_', 'test_m33_q3_', 'test_m35_q1a_', 'test_m35_q2_'):
            self.assertTrue(any(n.startswith(part) for n in names), part)


if __name__ == '__main__':
    unittest.main()
