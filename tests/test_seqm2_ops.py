import importlib.util
from pathlib import Path
import unittest
P=Path(__file__).resolve().parents[1]/'scripts/seqm2_ops.py'
def module():
 s=importlib.util.spec_from_file_location('seqm2_ops_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class OpsTests(unittest.TestCase):
 def test_metadata_size_blocks_unreadable_startup(self):
  m=module()
  with self.assertRaisesRegex(ValueError,'metadata'):m.metadata_guard({}, {'x':'a'*65536})
  self.assertEqual(m.metadata_guard({}, {'clean':True}),{'clean':True})
 def test_vram_measurement_not_copied(self):
  m=module();self.assertEqual(m.declared_vram(5704),6656)
  for x in [0,-1,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):m.declared_vram(x)
 def test_verify_requires_literal_and_binding(self):
  m=module();self.assertFalse(m.accepts_verify('succeeded',True,[{'status':'succeeded'}]));self.assertFalse(m.accepts_verify('succeeded',False,[{'status':'verified'}]));self.assertTrue(m.accepts_verify('succeeded',True,[{'status':'verified'}]))
 def test_failed_seed_does_not_hide_remaining(self):
  m=module();states=[{'verified':False,'train_status':'failed'},{'verified':True,'train_status':'succeeded'}];self.assertEqual(m.summary_status(states,False),'ALERT')
 def test_fresh_driver_fallback_preserves_identity(self):
  m=module();v=m.deadline_fallback([{'job':{'run_id':'p-seqm2-k4-s42'},'expected':{'config_sha256':'a'}}]);self.assertEqual(v[0]['run_id'],'p-seqm2-k4-s42');self.assertTrue(v[0]['unaccepted'])
if __name__=='__main__':unittest.main()
