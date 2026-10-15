import importlib.util
from pathlib import Path
import unittest

class NormOracle(unittest.TestCase):
    def test_complete_source_ledger_detects_omitted_or_modified_dependencies(self):
        import tempfile
        import hashlib
        m=self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'src/vkgr';folder.mkdir(parents=True)
            names=['retrieval_php01.py','retrieval_php01_run.py','retrieval_features.py','retrieval_models.py','retrieval_hp.py','training.py','history_side.py']
            for name in names:(folder/name).write_text('original')
            ledger={f'src/vkgr/{name}':hashlib.sha256(b'original').hexdigest() for name in names}
            self.assertEqual(m.checked_source_ledger(root,ledger),ledger)
            (folder/'history_side.py').write_text('changed')
            with self.assertRaises(ValueError):m.checked_source_ledger(root,ledger)
            del ledger['src/vkgr/retrieval_models.py']
            with self.assertRaises(ValueError):m.checked_source_ledger(root,ledger)

    def module(self):
        p=Path(__file__).with_name('seqm1_baseline_read.py')
        self.assertTrue(p.exists(),'read-only SEQM1-000 helper missing')
        s=importlib.util.spec_from_file_location('baseline',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    def test_raw_norms_are_not_normalized_and_interval_is_inclusive(self):
        m=self.module();v=m.raw_norm_summary([[3.,4.],[0.,2.]])
        self.assertEqual(v['median'],3.5);self.assertEqual(v['p10'],2.3);self.assertEqual(v['p90'],4.7)
        self.assertTrue(v['scale_check_passed'])
        self.assertTrue(m.raw_norm_summary([[.05,0.]])['scale_check_passed'])
        self.assertTrue(m.raw_norm_summary([[5.,0.]])['scale_check_passed'])
        self.assertFalse(m.raw_norm_summary([[.049,0.]])['scale_check_passed'])
    def test_bad_or_nonfinite_norms_stop(self):
        m=self.module()
        for values in ([],[[float('nan'),0]],[[float('inf')]],[[1],[2,3]]):
            with self.assertRaises(ValueError):m.raw_norm_summary(values)

if __name__=='__main__':unittest.main()
