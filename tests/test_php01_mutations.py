"""P-HP-01 mutation tests D1-D11: each deliberately wrong implementation must be caught by its named guard.

Same pattern as tests/test_feat02_mutations.py: guard passes on the real code, raises AssertionError under
``mock.patch.object`` on a module-level seam, passes again after the revert. Bucket constants and the F4 formula are
not mutated here (FEAT-02 M3-M5 already guard them).
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_mutations import detected  # noqa: E402
import test_php01_model as tm  # noqa: E402
import test_php01_run as tr  # noqa: E402
import test_php01_select as ts  # noqa: E402


class DimMutationTests(unittest.TestCase):
    def test_d1_every_table_ignores_dim(self):
        """All tables stay 64 wide: H3/H4 would silently run the anchor."""
        from vkgr import retrieval_php01 as m
        detected(self, m, 'make_table', lambda rows, dim: torch.nn.Embedding(rows, 64, padding_idx=0), [tm.guard_every_table_has_model_dim])

    def test_d2_side_tables_stay_64(self):
        """Only the id table follows dim; T/R/P/G stay 64 (shape is checked before the forward pass)."""
        from vkgr import retrieval_php01 as m
        detected(self, m, 'side_dim', lambda model: 64, [tm.guard_every_table_has_model_dim])

    def test_d3_variant_tables_drawn_before_anchor_tables(self):
        """Draw order P, G before T, R: no dim != 64 reference exists, so the dim=64 bytewise guard is the one that catches it."""
        from vkgr import retrieval_php01 as m

        def reversed_draws(model, std):
            anchor = model.add_anchor_tables(); variant = model.add_variant_tables()
            m._small_init_d(model, variant, std); m._small_init_d(model, anchor, std)
        detected(self, m, '_draw_tables', reversed_draws, [tm.guard_dim64_init_bytewise_equals_f4])

    def test_d4_init_std_scaled_with_dim(self):
        """std = 0.02 * sqrt(64 / dim): invisible at dim=64."""
        from vkgr import retrieval_php01 as m
        detected(self, m, 'init_std', lambda dim: 0.02 * (64 / dim) ** .5, [tm.guard_init_std_is_dim_independent])


class RunnerMutationTests(unittest.TestCase):
    def test_d5_runner_drops_config_dim(self):
        from vkgr import retrieval_php01_run as php
        original = php.make_php01_model
        detected(self, php, 'make_php01_model', lambda content, supported, max_len, dim: original(content, supported, max_len, 64),
                 [tr.guard_runner_builds_config_dim])

    def test_d6_runner_resets_temperature(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php
        detected(self, php, '_step', lambda model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency:
                 fr.train_step(model, optimizer, batch, data, stats, epoch, {**config, 'temperature': .15}, rng, pool, frequency),
                 [tr.guard_first_step_loss_uses_config_temperature])

    def test_d7_optimizer_hard_codes_lr(self):
        from vkgr import retrieval_php01_run as php
        detected(self, php, 'make_optimizer', lambda model, config: torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=config['weight_decay']),
                 [tr.guard_optimizer_uses_config_lr])


class SelectMutationTests(unittest.TestCase):
    def test_d8_two_levels_of_one_knob_count_twice(self):
        from vkgr import retrieval_php01_select as s
        detected(self, s, 'knob_group', lambda knob: knob, [ts.guard_same_knob_twice_does_not_trigger])

    def test_d9_threshold_any_positive_delta(self):
        from vkgr import retrieval_php01_select as s
        detected(self, s, 'is_winner', lambda delta_hits: delta_hits > 0, [ts.guard_five_hits_is_not_a_winner])

    def test_d10_tie_break_enumeration_reversed(self):
        from vkgr import retrieval_php01_select as s
        detected(self, s, 'selection_key', lambda e: (-e['r50'], -e['n50'], -s.ORDER.index(e['name'])), [ts.guard_tie_goes_to_earlier_enumeration])

    def test_d11_group_takes_the_worse_level(self):
        from vkgr import retrieval_php01_select as s
        detected(self, s, 'best_level', lambda entries: max(entries, key=s.selection_key), [ts.guard_h7_takes_each_knobs_best_level])


if __name__ == '__main__':
    unittest.main()
