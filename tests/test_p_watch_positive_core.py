import unittest

from vkgr.p_watch_positive import (build_auxiliary, count_like_train_examples, epoch_rows, logq_frequency,
                                   quantile, train_statistics, train_statistics_from_reader, watch_ratio)


class WatchPositiveCoreTests(unittest.TestCase):
    def test_epoch_draw_is_deterministic_without_replacement_and_caps_at_like_count(self):
        a = epoch_rows(["l1", "l2", "l3", "l4"], ["a1", "a2"], 3, 20261020)
        b = epoch_rows(["l1", "l2", "l3", "l4"], ["a1", "a2"], 3, 20261020)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 6)
        self.assertEqual(sum(x.startswith("a") for x in a), 2)
        self.assertEqual(len(set(a)), 6)
        self.assertEqual(len(epoch_rows([1], [2, 3, 4], 0, 20261020)), 2)

    def test_logq_expected_frequency_hand_calculation(self):
        self.assertEqual(logq_frequency(["a", "a", "b"], ["b", "b", "b", "c"]), {"a": 2.0, "b": 3.25, "c": 0.75})

    def test_quantile_requires_explicit_method_and_train_only_stats(self):
        with self.assertRaises(ValueError):
            quantile([1, 2, 3], .8, None)
        events = [
            {"week": 0, "user_id": 8, "item_id": 1, "timespent": 10, "like": True, "dislike": False},
            {"week": 0, "user_id": 8, "item_id": 2, "timespent": 10, "like": False, "dislike": False},
            {"week": 1, "user_id": 8, "item_id": 3, "timespent": 1, "like": False, "dislike": True},
        ]
        stats = train_statistics(events, {1: 10, 2: 10, 3: 10}, {1, 2}, "nearest_rank")
        self.assertEqual(stats["N_like"], 0)
        self.assertEqual(stats["N_like_events"], 1)
        self.assertEqual(stats["N_aux_candidates"], 1)
        self.assertEqual(stats["N_aux"], 1)
        self.assertEqual(stats["aux_like_item_overlap"], 0)
        self.assertEqual(stats["buckets"]["1-10"]["threshold"], 1.0)

    def test_like_example_count_reuses_original_prefix_builder(self):
        def partitions():
            return [(0, [
                {"user_id": 9, "item_id": 1, "like": False},
                {"user_id": 9, "item_id": 2, "like": True},
                {"user_id": 9, "item_id": 3, "like": True},
            ])]
        self.assertEqual(count_like_train_examples(partitions), 1)

    def test_nearest_rank_small_bins_ties_empty_and_float32_ratio(self):
        self.assertEqual(quantile([3.0], .8, "nearest_rank"), 3.0)
        self.assertEqual(quantile([1, 2, 3], .8, "nearest_rank"), 3)
        self.assertEqual(quantile([1, 2, 3, 4, 5], .8, "nearest_rank"), 4)
        values = [1, 2, 2, 2, 3]
        threshold = quantile(values, .8, "nearest_rank")
        self.assertEqual(threshold, 2)
        self.assertEqual(sum(value >= threshold for value in values), 4)
        with self.assertRaises(ValueError):
            quantile([], .8, "nearest_rank")
        self.assertEqual(watch_ratio(1, 10), 0.10000000149011612)

    def test_invalid_ratio_over_one_percent_is_reported_as_stop(self):
        events = [
            {"user_id": 1, "item_id": 1, "week": 0, "ordinal": 0, "timespent": 1, "like": True, "dislike": False},
            {"user_id": 2, "item_id": 1, "week": 0, "ordinal": 1, "timespent": 10, "like": False, "dislike": False},
        ]
        events.extend({"user_id": 3, "item_id": 1, "week": 0, "ordinal": i, "timespent": 10, "like": False, "dislike": True}
                      for i in range(2, 98))
        events.extend({"user_id": 4, "item_id": 2, "week": 0, "ordinal": i, "timespent": 0, "like": False, "dislike": False}
                      for i in range(98, 100))
        result = train_statistics_from_reader(lambda: iter(events), {1: 10, 2: 0}, {1}, "nearest_rank", 0)
        self.assertEqual(result["invalid_ratio_events"], 2)
        self.assertEqual(result["invalid_ratio_fraction"], .02)
        self.assertIn("invalid/nonfinite ratio events exceed 1%", result["stop_reasons"])

    def test_linear_difference_is_descriptive_for_prefilter_and_final_auxiliary_set(self):
        events = [
            {"user_id": 1, "item_id": 1, "week": 0, "ordinal": 0, "timespent": 10, "like": True, "dislike": False},
            {"user_id": 1, "item_id": 1, "week": 0, "ordinal": 1, "timespent": 10, "like": True, "dislike": False},
            {"user_id": 2, "item_id": 1, "week": 0, "ordinal": 2, "timespent": 10, "like": False, "dislike": False},
            {"user_id": 2, "item_id": 1, "week": 0, "ordinal": 3, "timespent": 10, "like": False, "dislike": False},
            {"user_id": 3, "item_id": 1, "week": 0, "ordinal": 4, "timespent": 30, "like": False, "dislike": True},
        ]
        result = train_statistics_from_reader(lambda: iter(events), {1: 10}, {1}, "nearest_rank", 1)
        self.assertEqual(result["buckets"]["1-10"]["threshold"], 1.0)
        self.assertEqual(result["buckets"]["1-10"]["linear_threshold"], 1.4000000000000004)
        self.assertEqual(result["buckets"]["1-10"]["linear_ratio_screened_membership_difference_count"], 4)
        self.assertEqual(result["buckets"]["1-10"]["linear_final_aux_membership_difference_count"], 2)

    def test_builder_uses_strict_past_observed_history_and_fails_week_leak(self):
        events = [
            {"user_id": 7, "item_id": 1, "week": 0, "ordinal": 0, "like": True, "dislike": False, "timespent": 10},
            {"user_id": 7, "item_id": 2, "week": 0, "ordinal": 1, "like": False, "dislike": False, "timespent": 10},
            {"user_id": 7, "item_id": 3, "week": 0, "ordinal": 2, "like": False, "dislike": True, "timespent": 10},
        ]
        out = build_auxiliary(events, {2}, {1: 10, 2: 10, 3: 10}, {"1-10": 0.8}, quantile_method="nearest_rank")
        self.assertEqual(out[0]["target"], 2)
        self.assertEqual(out[0]["history"], [1])
        self.assertEqual(out[0]["history_positions"], [[0, 0]])
        with self.assertRaisesRegex(ValueError, "week"):
            build_auxiliary(events + [{"user_id": 9, "item_id": 4, "week": 25, "ordinal": 0, "like": False, "dislike": False, "timespent": 10}], {2}, {1: 10, 2: 10, 3: 10, 4: 10}, {"1-10": 0.8}, quantile_method="nearest_rank")


if __name__ == "__main__":
    unittest.main()
