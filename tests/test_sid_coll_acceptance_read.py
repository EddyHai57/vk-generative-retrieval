import importlib.util,json,tempfile,unittest,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'analysis/sid_coll_acceptance_read.py'
class Read(unittest.TestCase):
 def load(self):
  s=importlib.util.spec_from_file_location('ro',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_no_weight_or_prediction_binary(self):
  m=self.load()
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'best.pt';p.write_bytes(b'never read')
   with self.assertRaises(ValueError):m.read(p)
 def test_status_alone_never_accepted(self):
  m=self.load()
  self.assertFalse(m.accept({'status':'succeeded','exit_code':0},[],[],None,None))
  receipt={'schema':'coll-stage-receipt-v1','status':'verified','run_id':'x','binding':{'commit':m.COMMIT,'spec_sha256':'a'*64}}
  expected={'run_id':'x','spec_sha256':'a'*64}
  self.assertFalse(m.accept({'status':'succeeded','exit_code':0},['c'],[],receipt,expected,raw_status='failed'))
  self.assertTrue(m.accept({'status':'succeeded','exit_code':0},['c'],[],receipt,expected,raw_status='verified'))
if __name__=='__main__':unittest.main()
