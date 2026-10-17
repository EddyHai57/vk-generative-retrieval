import importlib.util
import unittest
class AuditTests(unittest.TestCase):
    def test_missing_audit_detected(self):
        self.assertIsNotNone(importlib.util.find_spec('vkgr.profile_like_audit'))
    def test_missing_is_not_false_and_invalid_is_not_true(self):
        from vkgr.profile_like_audit import summarize
        s=summarize([True,False,None,True,1,'true'])
        self.assertEqual((s['rows'],s['missing'],s['true'],s['false'],s['invalid']),(6,1,2,1,2))
        self.assertAlmostEqual(s['missing_rate'],1/6)
    def test_empty_is_unknown_not_zero_rate(self):
        from vkgr.profile_like_audit import summarize
        self.assertIsNone(summarize([])['missing_rate'])
    def test_training_only_paths_and_order(self):
        from vkgr.profile_like_audit import train_records
        rec=[{'role':'subsamples/ur0.01_ir0.01/train/week_%02d.parquet'%i,'sha256':'a'*64,'size_bytes':1} for i in range(25)]
        self.assertEqual(len(train_records({'binding':{'raw_records':rec}})),25)
        for broken in [rec[:-1],list(reversed(rec)),rec+[{'role':'subsamples/ur0.01_ir0.01/test/week_26.parquet','sha256':'a'*64,'size_bytes':1}]]:
            with self.assertRaises(ValueError):train_records({'binding':{'raw_records':broken}})
if __name__=='__main__':unittest.main()
