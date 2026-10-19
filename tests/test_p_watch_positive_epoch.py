import unittest

from vkgr.p_watch_positive_epoch import epoch_tokens, epoch_rows, frequency_table


class WatchPositiveEpochTests(unittest.TestCase):
    def test_epoch_mix_keeps_every_like_and_samples_aux_without_replacement(self):
        likes = [{"target": 1}, {"target": 1}, {"target": 2}]
        auxiliary = [{"target": 2}, {"target": 3}, {"target": 3}, {"target": 4}]
        tokens = epoch_tokens(len(likes), len(auxiliary), 0)
        self.assertEqual(len(tokens), 6)
        self.assertEqual(sum(kind == "like" for kind, _ in tokens), 3)
        self.assertEqual(sum(kind == "aux" for kind, _ in tokens), 3)
        self.assertEqual(len({token for token in tokens if token[0] == "aux"}), 3)
        self.assertEqual(tokens, epoch_tokens(len(likes), len(auxiliary), 0))
        self.assertEqual(epoch_rows(likes, auxiliary, tokens), [likes[i] if kind == "like" else auxiliary[i] for kind, i in tokens])

    def test_epoch_mix_draws_all_aux_when_pool_smaller_and_logq_is_hand_calculated(self):
        self.assertEqual(len(epoch_tokens(5, 2, 4)), 7)
        self.assertEqual(frequency_table([{"target": 1}, {"target": 1}, {"target": 2}],
                                         [{"target": 2}, {"target": 2}, {"target": 2}, {"target": 3}]),
                         {"name": "P_WATCH_EXPECTED_EPOCH_FREQ_V1", "counts": {1: 2.0, 2: 3.25, 3: .75}, "total": 6.0})


if __name__ == "__main__":
    unittest.main()
