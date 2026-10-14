"""FEAT-03 mutation tests, FEAT-002 part (M13, M14, M16, M18, M33[q1a,q1b], M35[q1a]).

Each deliberately wrong implementation must be caught by every listed guard: the guard is green on the real code, raises
AssertionError under ``mock.patch.object`` on a module-level seam, and is green again after the revert (``detected``).
Same-week reorder is NOT a guard here: F4 and Q1b are equal up to rounding under it.
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat03_model as t3  # noqa: E402
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


if __name__ == '__main__':
    unittest.main()
