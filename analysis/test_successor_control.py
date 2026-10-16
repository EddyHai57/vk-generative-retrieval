import importlib.util
from pathlib import Path
import unittest
class ControlChecks(unittest.TestCase):
 def module(self):
  p=Path(__file__).with_name('successor_control.py');self.assertTrue(p.exists());s=importlib.util.spec_from_file_location('s',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_latency_contract_checks_actual_call_counts(self):
  m=self.module();record={'formal':True,'device':'cpu','models':{k:{mode+'_ms_per_query':{'n':1000,'timed_calls':1000,'warmup_calls':50,'batch':b,'input_sha256':'x'} for mode,b in [('offline',512),('online',1)]} for k in ('anchor','mean')}}
  m.check_latency(record,'cpu');record['models']['mean']['offline_ms_per_query']['n']=20
  with self.assertRaises(ValueError):m.check_latency(record,'cpu')
if __name__=='__main__':unittest.main()
