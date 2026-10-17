"""Prefix supplement boundary tests; mutation targets named in test names."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

class PrefixReadTests(unittest.TestCase):
    def test_missing_entry_is_detected(self):
        self.assertIsNotNone(importlib.util.find_spec('analysis.sid_a1_prefix_read'))
    def tool(self):
        from analysis import sid_a1_prefix_read
        return sid_a1_prefix_read
    def test_unapproved_run_and_seed_subset_rejected(self):
        t=self.tool()
        for ids in [['g-coll-a1-c1-s42'],['g-coll-a1-c1-s42','g-coll-a1-c1-s2024','g-other-s2025']]:
            with self.assertRaises(ValueError):t.validate_run_ids(ids)
        t.validate_run_ids(['g-coll-a1-c1-s42','g-coll-a1-c1-s2024','g-coll-a1-c1-s2025'])
    def test_symlink_and_prohibited_ancestor_rejected_before_read(self):
        t=self.tool()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();p=root/'sample.json';p.write_text('{}');link=root/'alias.json';link.symlink_to(p)
            with self.assertRaises(ValueError):t.safe(link,[root])
            bad=root/'test'/'sample.json';bad.parent.mkdir();bad.write_text('{}')
            with self.assertRaises(ValueError):t.safe(bad,[root])
            with self.assertRaises(ValueError):t.safe(p,[root/'other'])
            self.assertEqual(t.safe(p,[root]),p)
    def test_frozen_B_hand_calculation_oov_duplicate_and_full_hit(self):
        t=self.tool();_,metrics=t.frozen(Path(os.environ['VKGR_FROZEN_DIAG_ROOT']))
        mapping={1:(0,0,0,0),2:(0,1,0,0),3:(1,0,0,0)}
        b=t.aggregate([1,2,99],[[2,2,1],[3],[1]],mapping,metrics)
        self.assertEqual((b['n_total'],b['n_in_catalog'],b['n_oov']),(3,2,1))
        self.assertEqual((b['prefix_by_k']['1']['top1_hits'],b['prefix_by_k']['1']['top50_hits']),(1,1))
        self.assertEqual((b['prefix_by_k']['2']['top1_hits'],b['prefix_by_k']['2']['top50_hits']),(0,1))
        self.assertEqual(b['exact_item']['top50_hits'],1)
        self.assertNotIn('predictions',b)
    def test_population_mismatch_and_item_hits_mismatch_rejected(self):
        t=self.tool();_,metrics=t.frozen(Path(os.environ['VKGR_FROZEN_DIAG_ROOT']))
        b=t.aggregate([1,99],[[1],[1]],{1:(0,0,0,0)},metrics)
        with self.assertRaises(ValueError):t.population(b,expected_hits=1)
        with self.assertRaises(ValueError):t.population(b,expected_hits=0,total=2,in_catalog=1,oov=1)
        t.population(b,expected_hits=1,total=2,in_catalog=1,oov=1)
    def test_tampered_frozen_source_rejected(self):
        t=self.tool()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'analysis').mkdir();(p/'analysis/sid_diag_metrics.py').write_text('raise RuntimeError()')
            with self.assertRaises(ValueError):t.frozen(p)

if __name__=='__main__':unittest.main()
