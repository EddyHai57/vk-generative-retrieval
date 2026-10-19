import unittest

from vkgr.p_watch_positive_rows import build_auxiliary_rows


def event(user, item, like=False, dislike=False, timespent=10):
    return {"user_id": user, "item_id": item, "like": like, "dislike": dislike,
            "share": False, "bookmark": False, "open_comments": False, "click_on_author": False,
            "timespent": timespent}


class WatchPositiveRowsTests(unittest.TestCase):
    def test_builder_reuses_observed_order_and_emits_exact_strict_past_side(self):
        partitions = lambda: [(0, iter([event(1, 1, like=True), event(1, 2), event(1, 3, dislike=True)]))]
        rows = list(build_auxiliary_rows(partitions, {2}, {1: 10, 2: 10, 3: 10}, {"1-10": 1.0}))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target"], 2)
        self.assertEqual(rows[0]["target_position"], [0, 1])
        self.assertEqual(rows[0]["history"], [1])
        self.assertEqual(rows[0]["history_positions"], [[0, 0]])
        self.assertEqual(rows[0]["past_likes"], [1])
        self.assertEqual(rows[0]["side"], [[1, 10, 10]])

    def test_builder_preserves_empty_history_candidates_for_explicit_runner_gate(self):
        partitions = lambda: [(0, iter([event(2, 2)]))]
        rows = list(build_auxiliary_rows(partitions, {2}, {2: 10}, {"1-10": 1.0}))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["history"], [])


if __name__ == "__main__":
    unittest.main()
