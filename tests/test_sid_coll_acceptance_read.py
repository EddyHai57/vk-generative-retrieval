import importlib.util,json,tempfile,unittest,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'analysis/sid_coll_acceptance_read.py'
class Read(unittest.TestCase):
 def load(self):
  s=importlib.util.spec_from_file_location('ro',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_epoch_metadata_only_exact_hits_and_zero_denominator(self):
  m=self.load();v={'selection':{'best_epoch':1,'epoch':2},'epochs':[{'epoch':1,'metrics':{'metrics':{'all_eligible':{'count':2,'recall@50':.5,'ndcg@50':.25}}}}]}
  a=m.epoch_rows(v);self.assertEqual(a['epochs'][0]['hits_derived'],1);self.assertEqual(a['stop_epoch'],2)
  v['epochs'][0]['metrics']['metrics']['all_eligible']['count']=0
  self.assertEqual(m.epoch_rows(v)['epochs'][0]['hits_derived'],'UNKNOWN')
 def test_producer_summary_omits_full_arrays_and_keeps_health(self):
  m=self.load();v={'dimension':64,'vectors_sha256':'a'*64,'audit':{'fallback_count':1,'fallback_mask':[True],'health':{'passed':True}}}
  a=m.producer_fields(v);self.assertEqual(a['audit']['health'],{'passed':True});self.assertNotIn('fallback_mask',a['audit'])
 def test_epoch_report_is_retained(self):
  m=self.load();value={'epoch_runs':{'s42':{'best_epoch':3}}};self.assertEqual(m.result_fields(value),value)
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
