"""Literal synthetic oracles: zero support, entity denominators, joins and strata."""
import sys
import unittest
import tempfile
import hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import p3c_data_quality as quality

class QualityTests(unittest.TestCase):
    def test_counts_keep_zero_entities_and_use_full_population_for_head_share(self):
        result = quality.summarize_counts([0, 1, 1, 8])
        self.assertIn('entities', result)
        self.assertEqual(result['entities'], 4)
        self.assertEqual(result['total'], 10)
        self.assertEqual(result['zero_entities'], 1)
        self.assertEqual(result['p50'], 1)
        self.assertEqual(result['max'], 8)
        self.assertEqual(result['top_1pct_share'], .8)
        self.assertEqual(result['bins']['1']['entities'], 2)
        self.assertEqual(result['bins']['6-10']['total'], 8)

    def test_user_strata_preserve_low_activity_and_identify_flags_without_deletion(self):
        result = quality.summarize_users([1,2,10,60], [0,1,2,5], [1,0,3,0], [0,1,0,2])
        self.assertIn('like_count_bins', result)
        self.assertEqual(result['like_count_bins']['0'], {'users':1,'exposures':1,'likes':0,'prefixes':0,'ratio_gt_1_events':1,'timespent_255_events':0})
        self.assertEqual(result['like_count_bins']['2-5']['users'], 2)
        self.assertEqual(result['like_count_bins']['2-5']['prefixes'], 5)
        self.assertEqual(result['users_with_ratio_gt_1'], 2)
        self.assertEqual(result['users_with_timespent_255'], 2)
        self.assertEqual(result['prefix_total'], 5)

    def test_author_aggregation_groups_items_before_computing_zero_support(self):
        result = quality.summarize_authors([7,7,9], [2,3,5], [0,1,0])
        self.assertIn('authors', result)
        self.assertEqual(result['authors'], 2)
        self.assertEqual(result['likes']['zero_entities'], 1)
        self.assertEqual(result['exposures']['total'], 10)
        self.assertEqual(result['catalog_items_per_author']['max'], 2)

    def test_metadata_join_detects_cross_batch_duplicate_missing_and_null(self):
        result = quality.join_metadata(np.array([2,4]), [{'id':np.array([1,2]), 'x':np.array([9,8])}, {'id':np.array([4]),'x':np.array([7])}])
        self.assertIn('x', result)
        self.assertEqual(result['x'].tolist(), [8,7])
        for batches in ([{'id':np.array([2]),'x':np.array([8])}],
                        [{'id':np.array([2,4]),'x':np.array([8,7])},{'id':np.array([2]),'x':np.array([6])}],
                        [{'id':np.array([2,4]),'x':np.array([None,7],dtype=object)}]):
            with self.assertRaises(ValueError):
                quality.join_metadata(np.array([2,4]), batches)

class BindingTests(unittest.TestCase):
    def test_file_bindings_support_both_existing_size_schemas_and_stop_on_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'a').write_bytes(b'abc')
            digest = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
            records = [{'path':'a','sha256':digest,'bytes':3}, {'path':'a','sha256':digest,'size_bytes':3}]
            result = quality.verify_files(root, records)
            self.assertEqual(result, [{'path':'a','sha256':digest,'bytes':3}])
            for bad in ({'path':'a','sha256':digest,'size_bytes':4}, {'path':'a','sha256':'0'*64,'bytes':3}):
                with self.assertRaises(ValueError):
                    quality.verify_files(root, records+[bad])

class StreamingTests(unittest.TestCase):
    def test_scan_keeps_all_rows_reconciles_items_users_and_original_order(self):
        batches = [(0, {'user_id':np.array([2,2]), 'item_id':np.array([4,8]), 'like':np.array([True,True]), 'timespent':np.array([255,5])}),
                   (1, {'user_id':np.array([3,2]), 'item_id':np.array([4,9]), 'like':np.array([False,True]), 'timespent':np.array([10,4])})]
        result = quality.scan_batches(batches, np.array([2,3]), np.array([4,8,9]), np.array([5,10,4]))
        self.assertIn('rows', result)
        self.assertEqual(result['rows'], 4)
        self.assertEqual(result['item_exposures'].tolist(), [2,1,1])
        self.assertEqual(result['item_likes'].tolist(), [1,1,1])
        self.assertEqual(result['user_likes'].tolist(), [3,0])
        self.assertEqual(result['user_ratio_gt_1'].tolist(), [1,1])
        self.assertEqual(result['user_upper_code'].tolist(), [1,0])
        self.assertEqual(result['weeks'], [{'week':0,'rows':2,'like_events':2}, {'week':1,'rows':2,'like_events':1}])
        # Reversing week chronology must stop, rather than sorting rows to conceal it.
        with self.assertRaises(ValueError):
            quality.scan_batches(list(reversed(batches)), np.array([2,3]), np.array([4,8,9]), np.array([5,10,4]))
        with self.assertRaises(ValueError):
            quality.scan_batches([(0, {'user_id':np.array([7]), 'item_id':np.array([4]), 'like':np.array([True]), 'timespent':np.array([1])})], np.array([2]),np.array([4]),np.array([5]))

if __name__ == '__main__':
    unittest.main()
