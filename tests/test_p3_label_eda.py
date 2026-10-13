"""Literal synthetic oracles; no real IDs or data files."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("eda", Path(__file__).parents[1] / "scripts/p3_label_eda.py")
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)

class EDATests(unittest.TestCase):
    def test_threshold_boundaries_zero_duration_and_conflict(self):
        # Synthetic only: order is arbitrary; no real event examples.
        masks = eda.candidate_masks(np.array([4,5,10,20,255]), np.array([10,10,0,20,100]),
            np.array([False,False,True,False,False]), np.array([False,True,False,False,False]),
            np.array([False,False,False,True,False]))
        self.assertIn("like", masks)
        self.assertEqual(masks["like"].tolist(), [False,False,True,False,False])
        self.assertEqual(masks["like_or_share_or_bookmark"].tolist(), [False,True,True,True,False])
        self.assertEqual(masks["seconds_ge_5"].tolist(), [False,True,True,True,True])
        self.assertEqual(masks["ratio_ge_0.5"].tolist(), [False,True,False,True,True])
        self.assertEqual(masks["ratio_ge_1.0"].tolist(), [False,False,False,True,True])

    def test_hist_quantiles_include_zero_and_tail_contribution(self):
        result=eda.hist_summary({"0":2,"1":1,"4":1,"100":1})
        self.assertIn("users", result)
        self.assertEqual(result["users"],5)
        self.assertEqual(result["p50"],1)
        self.assertEqual(result["p90"],100)
        self.assertEqual(result["users_gt_50"],1)
        self.assertEqual(result["snapshot_events_beyond_last50"],50)
        self.assertEqual(result["top_1pct_users_event_share"],100/105)

    def test_numeric_summary_has_no_history_user_meaning(self):
        result = eda.numeric_summary({"5":2,"255":1}, "interaction", "seconds")
        self.assertEqual(result['sample_count'], 3)
        self.assertEqual(result['p50'], 5)
        self.assertEqual(result['max'], 255)
        self.assertNotIn('users', result)
        self.assertNotIn('snapshot_events_beyond_last50', result)

    def test_segment_sum_and_conflicts_not_filtered(self):
        watch, duration = np.array([5,20,255]), np.array([10,20,100])
        fb = {k:np.array([False,False,False]) for k in eda.FEEDBACK}
        fb['like'] = np.array([True,False,False])
        fb['dislike'] = np.array([True,True,False])
        masks = eda.candidate_masks(watch,duration,fb['like'],fb['share'],fb['bookmark'])
        result = eda.segment_summary(np.array([True,True,False]),fb,watch,duration,masks)
        self.assertEqual(result['events'],2)
        self.assertEqual(result['feedback']['dislike'],2)
        self.assertEqual(result['candidates']['like'],1)
        self.assertEqual(result['watch_sum'],25)
        self.assertEqual(result['ratio_gt_1'],0)

if __name__ == "__main__":
    unittest.main()
