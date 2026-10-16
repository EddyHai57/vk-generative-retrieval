import unittest,copy
from analysis.feature_leakage_real import flip_target, audit_runtime
class LeakTests(unittest.TestCase):
 def test_flip_never_mutates_history_or_side(self):
  row={'target':7,'target_position':[25,1],'oov':False,'history':[1],'side':[[1,2,3]],'history_positions':[[24,1]],'past_likes':[7]}
  original=copy.deepcopy(row);got=flip_target(row,[8,9])
  self.assertNotEqual(got['target'],row['target']);self.assertEqual(got['history'],row['history']);self.assertEqual(row,original)
 def test_real_execution_rejects_synthetic_label(self):
  with self.assertRaises(ValueError):audit_runtime({'synthetic':True},None,None)
class ProductionFunctionFixture(unittest.TestCase):
 def test_full_runtime_matrix_on_synthetic_fixture_uses_real_functions(self):
  from vkgr import retrieval_feat04b_run as r
  from vkgr import retrieval_feat04b as m
  data=r.synthetic_data4b();data['synthetic']=False
  out=audit_runtime(data,r,m,train_count=24)
  self.assertEqual(set(out['variants']),{'A','C','D','E'})
  for v in out['variants'].values():
   for key,check in v['checks'].items():self.assertIn(check['status'],('pass','STATIC_REVIEWED','NOT_APPLICABLE'))
class ReceiptDurability(unittest.TestCase):
 def test_missing_output_parent_created_and_stdout_retained_on_failure(self):
  import tempfile,io,json
  from pathlib import Path
  from contextlib import redirect_stdout
  from unittest.mock import patch
  from analysis.feature_leakage_real import emit_receipt
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'new-run/out/receipt.json';s=io.StringIO()
   with redirect_stdout(s):emit_receipt(p,{'status':'pass'})
   self.assertEqual(json.loads(p.read_text()),{'status':'pass'})
   self.assertEqual(json.loads(s.getvalue()),{'status':'pass'})
   s=io.StringIO()
   with patch.object(Path,'mkdir',side_effect=OSError('output fault')),redirect_stdout(s):
    with self.assertRaises(OSError):emit_receipt(Path(t)/'other/out/receipt.json',{'status':'FAIL'})
   self.assertEqual(json.loads(s.getvalue()),{'status':'FAIL'})
