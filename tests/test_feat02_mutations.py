"""FEAT-02 mutation tests M1-M8: each deliberately wrong implementation must be caught by its named guard.

Every guard is first run on the unmodified code (must pass), then under ``mock.patch.object`` on a module-level seam
(must raise AssertionError), then again after the patch is reverted (must pass).
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat02_model as tm  # noqa: E402
import test_feat02_run as tr  # noqa: E402
import test_feat02_data as td  # noqa: E402


def detected(case, module, name, replacement, guards):
    for guard in guards:
        guard()                                                    # sound on the real implementation
    with mock.patch.object(module, name, replacement):
        for guard in guards:
            with case.assertRaises(AssertionError, msg=f'{name}: {guard.__name__} did not catch the mutation'):
                guard()
    for guard in guards:
        guard()                                                    # back to green after the revert


class MutationTests(unittest.TestCase):
    def test_m1_masked_with_inverted_mask(self):
        """F3-masked adds the buckets on UNSUPPORTED events only."""
        from vkgr import retrieval_feat02 as m
        detected(self, m, 'supported_mask', lambda supported, histories, valid: valid & ~supported[histories], [tm.guard_masked_adds_on_supported_only])

    def test_m2_masked_without_mask(self):
        """F3-masked degenerates to F3 (mask removed)."""
        from vkgr import retrieval_feat02 as m
        detected(self, m, 'supported_mask', lambda supported, histories, valid: valid, [tm.guard_masked_differs_from_f3_exactly_on_unsupported])

    def test_m3_week_gap_anchored_on_target_week(self):
        """The last history event's week is replaced by the target week (anchor leak)."""
        from vkgr import retrieval_feat02_run as fr
        original = fr._tensors2

        def mutated(batch, idmap, width, feature):
            histories, side = original(batch, idmap, width, feature)
            for j, r in enumerate(batch):
                side[j, len(r['history']) - 1, 3] = min(r['target_position'][0], 24)
            return histories, side
        detected(self, fr, '_tensors2', mutated, [tr.guard_target_week_change_leaves_feature_unchanged])

    def test_m4_position_counted_from_sequence_start(self):
        from vkgr import retrieval_feat02 as m

        def from_start(valid):
            rank = (valid.long().cumsum(1) - 1).clamp_min(0)
            edges = torch.tensor(m.POSITION_EDGES, dtype=rank.dtype)
            return (torch.bucketize(rank.contiguous(), edges, right=True) + 1) * valid
        detected(self, m, 'position_bucket', from_start, [tm.guard_position_bucket_boundaries])
        # Mean pooling is permutation invariant: counting from the start yields the same multiset of position buckets as
        # counting from the end, so the pooled user vector (and the formula guard) cannot tell them apart (plan risk R4).
        # Only the per-slot boundary guard can; this documents why it is the named guard for M4.
        with mock.patch.object(m, 'position_bucket', from_start):
            tm.guard_f4_formula()

    def test_m5_week_gap_table_removed(self):
        from vkgr import retrieval_feat02 as m
        detected(self, m, 'weekgap_bucket', lambda gap, valid: torch.zeros_like(gap), [tm.guard_f4_formula, tm.guard_changing_history_week_changes_feature])

    def test_m6_place_and_platform_columns_swapped(self):
        """F5 reads the platform column as place and vice versa."""
        from vkgr import retrieval_feat02 as m
        original = m.scene_columns
        detected(self, m, 'scene_columns', lambda side: tuple(reversed(original(side))), [tm.guard_f5_formula])

    def test_m7_unseen_value_mapped_to_pad_instead_of_oov(self):
        """Values outside the train table get id 0 (PAD) rather than 1 (OOV)."""
        from vkgr import retrieval_feat02 as m
        original = m.category_ids
        detected(self, m, 'category_ids', lambda raw, table, valid: original(raw, table, valid) * (original(raw, table, valid) != 1), [tm.guard_f5_unseen_value_maps_to_oov_and_contributes_zero])

    def test_m8_build_ctx_reads_the_target_row(self):
        """The last ctx entry of every training example is replaced by the target row's place/platform (leak)."""
        from vkgr import history_ctx
        original = history_ctx.build_ctx

        def mutated(partitions, history):
            partitions = list(partitions); out = original(partitions, history)
            table = {(w, k): r for w, rows in partitions for k, r in enumerate(rows)}
            for j, row in enumerate(history['train']):
                target = table[tuple(row['target_position'])]
                if out['train'][j]:
                    out['train'][j][-1] = [target['place'], target['platform']]
            return out
        detected(self, history_ctx, 'build_ctx', mutated, [td.guard_ctx_target_and_future_event_fields_cannot_change_features])

    def test_m9_item_tower_still_zeroes_unsupported_items(self):
        """F6 item tower multiplies the WHOLE vector by the supported flag: unsupported items go back to zero vectors."""
        from vkgr.retrieval_feat02 import Feat02IDMean
        original = Feat02IDMean.items
        detected(self, Feat02IDMean, 'items', lambda self, indices: original(self, indices) * self.supported[indices].unsqueeze(-1),
                 [tm.guard_f6_item_vector_formula, tm.guard_f6_unsupported_items_are_nonzero_while_anchor_zero])

    def test_m10_content_projection_frozen(self):
        """The content projection output is detached: W exists and is applied but never learns."""
        from vkgr import retrieval_feat02 as m
        detected(self, m, 'project_content', lambda projection, content: projection(content).detach(), [tm.guard_f6_projection_is_learned_linear])

    def test_m11_oov_author_mapped_to_a_trainable_row(self):
        """Authors outside the vocabulary get a row of their own (>= 2) instead of the inert OOV row 1."""
        from vkgr import item_attr
        detected(self, item_attr, 'author_rows',
                 lambda authors, vocab: [0 if a is None else 2 + vocab.index(a) if a in vocab else 2 + a % max(len(vocab), 1) for a in authors],
                 [tm.guard_f6_author_pad_and_oov_rows_are_inert])

    def test_m12_vocabulary_built_from_the_whole_catalog(self):
        """The author vocabulary uses the authors of ALL catalog items instead of the supported ones."""
        from vkgr import item_attr
        detected(self, item_attr, 'author_vocab', lambda authors, support: sorted(set(authors.values())),
                 [td.guard_author_vocab_from_supported_items_only, td.guard_prepare_attr_vocab_pin_binds_the_supported_vocab])


if __name__ == '__main__':
    unittest.main()
