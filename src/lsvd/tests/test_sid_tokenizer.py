import tempfile
from pathlib import Path
import unittest
import numpy as np

from lsvd.sid_tokenizer import assign_codes, pack_codes, build_tables, filtered_ranks


class SIDTokenizerTests(unittest.TestCase):
    def test_residual_assignment_matches_numpy(self):
        x = np.array([[0, 0], [2, 1], [3, 3]], dtype=np.float32)
        centers = [np.array([[0, 0], [2, 2]], dtype=np.float32),
                   np.array([[0, 0], [1, -1]], dtype=np.float32)]
        residual = x.copy()
        expected = []
        for center in centers:
            labels = ((residual[:, None] - center[None]) ** 2).sum(-1).argmin(1)
            expected.append(labels)
            residual -= center[labels]
        np.testing.assert_array_equal(assign_codes(x, centers), np.array(expected).T)

    def test_tables_preserve_items_buckets_and_legal_prefixes(self):
        ids = np.array([10, 20, 30, 40], dtype=np.uint32)
        codes = np.array([[0, 1], [0, 1], [0, 2], [2, 0]], dtype=np.uint16)
        np.testing.assert_array_equal(pack_codes(codes, 4), [1, 1, 2, 8])
        with tempfile.TemporaryDirectory() as directory:
            build_tables(ids, codes, 4, Path(directory))
            path = Path(directory)
            np.testing.assert_array_equal(np.load(path/'bucket_offsets.npy'), [0, 2, 3, 4])
            np.testing.assert_array_equal(np.load(path/'bucket_items.npy'), ids)
            np.testing.assert_array_equal(np.load(path/'prefix_0_keys.npy'), [0])
            np.testing.assert_array_equal(np.load(path/'prefix_0_next.npy'), [0, 2])
            np.testing.assert_array_equal(np.load(path/'prefix_1_keys.npy'), [0, 2])
            np.testing.assert_array_equal(np.load(path/'prefix_1_offsets.npy'), [0, 2, 3])
            np.testing.assert_array_equal(np.load(path/'prefix_1_next.npy'), [1, 2, 0])

    def test_history_filter_rank_and_equal_popularity_hash_ties(self):
        # Globally sorted positions, two contiguous code buckets.
        positions = np.array([0, 2, 1, 4, 3])
        starts = np.array([0, 0, 0, 3, 3])
        history_rows = np.array([0, 2, 3, 3])
        ranks = filtered_ranks(np.arange(5), history_rows, positions, starts)
        np.testing.assert_array_equal(ranks, [-1, 0, -1, -1, 0])


if __name__ == '__main__':
    unittest.main()
