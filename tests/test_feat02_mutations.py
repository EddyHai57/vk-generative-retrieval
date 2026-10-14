"""FEAT-02 mutation tests M1-M5: each deliberately wrong implementation must be caught by its named guard.

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
        detected(self, m, 'position_bucket', from_start, [tm.guard_position_bucket_boundaries, tm.guard_f4_formula])

    def test_m5_week_gap_table_removed(self):
        from vkgr import retrieval_feat02 as m
        detected(self, m, 'weekgap_bucket', lambda gap, valid: torch.zeros_like(gap), [tm.guard_f4_formula, tm.guard_changing_history_week_changes_feature])


if __name__ == '__main__':
    unittest.main()
