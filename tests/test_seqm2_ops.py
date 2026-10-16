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
 def test_real_template_B_is_canonicalized_before_job_entry(self):
  m=module();a=['python','-B','-m',m.ENTRY,'train','--output','/new/out'];wrapped=m.wrap_argv(a)
  self.assertEqual(wrapped[3:],['-m',m.ENTRY,'train','--output','/new/out'])
  with self.assertRaises(ValueError):m.wrap_argv(['python','-I','-m',m.ENTRY,'train'])
 def test_detached_prepare_adds_ops_import_path(self):
  import sys,types
  from unittest.mock import patch
  m=module();fake=types.ModuleType('vkgr_ops');fake.launcher=types.SimpleNamespace(prepare=lambda j:{'clean':True})
  path='/workspace/vkgr-infra03-ops/ops'
  with patch.object(sys,'path',[p for p in sys.path if p!=path]),patch.dict(sys.modules,{'vkgr_ops':fake}):
   m.prepare_job({});self.assertIn(path,sys.path)
 def test_missing_or_changed_control_audit_fails_closed(self):
  m=module()
  with self.assertRaisesRegex(ValueError,'control'):m.preflight_audit_pin({})
  a={'controls':{f'p-seqm1-mean-s{s}':{'seed':s,'verified':True} for s in (42,2024,2025)}}
  p={'m1_control_audit':a,'m1_control_audit_sha256':m.audit_hash(a)};self.assertEqual(m.preflight_audit_pin(p),m.audit_hash(a))
  a['controls']['p-seqm1-mean-s42']['seed']=9
  with self.assertRaisesRegex(ValueError,'control'):m.preflight_audit_pin(p)
 def test_actual_entry_accepts_normalized_train_and_verify(self):
  import sys,types,runpy
  from unittest.mock import patch
  m=module();fake=types.ModuleType('seqm2_ops');fake.ENTRY=m.ENTRY
  fake.identity=lambda:None;fake.modules=lambda:None;fake.control_identity=lambda:None
  entry=Path(__file__).resolve().parents[1]/'scripts/seqm2_job_entry.py'
  for command in ('train','verify'):
   wrapped=m.wrap_argv(['python','-B','-m',m.ENTRY,command,'--config','/cfg'])
   with patch.dict(sys.modules,{'seqm2_ops':fake}),patch.object(sys,'argv',wrapped[2:]),patch('runpy.run_module') as call:
    runpy.run_path(str(entry),run_name='__main__')
    call.assert_called_once_with(m.ENTRY,run_name='__main__')
if __name__=='__main__':unittest.main()
