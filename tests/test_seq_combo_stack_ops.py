"""Private parameterization, approval and immutable freeze admission boundaries."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class OpsTests(unittest.TestCase):
 def module(self):
  p=ROOT/'scripts/seq_combo_stack_ops.py';self.assertTrue(p.exists(),'stack ops missing')
  spec=importlib.util.spec_from_file_location('stack_ops',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_requires_private_paths_and_review_approval(self):
  m=self.module()
  with self.assertRaises(ValueError):m.configure({})
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'review.json';p.write_text(json.dumps({'verdict':'CHANGES_REQUESTED','commit':m.COMBO_COMMIT}))
   with self.assertRaises(ValueError):m.check_combo_review(p,hashlib.sha256(p.read_bytes()).hexdigest())
   p.write_text(json.dumps({'verdict':'APPROVED','commit':m.COMBO_COMMIT}))
   self.assertTrue(m.check_combo_review(p,hashlib.sha256(p.read_bytes()).hexdigest()))
   with self.assertRaises(ValueError):m.check_combo_review(p,'0'*64)
 def test_config_batch_and_verify_derivation_identity(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as tmp:
   t=Path(tmp);paths={k:str(t/k) for k in m.PATH_KEYS};m.configure(paths)
   job={'run_id':'p-seq-combo-stack-s42','argv':['python','-m','vkgr.retrieval_seq_combo_stack_run','train','--output',str(t/'out'),'--inputs','i','--inputs-sha','a','--config','c','--config-sha','b'],'inputs':['i'],'disk_gib':8,'wall_hours':8}
   v=m.derive_verify(job,'f'*64)
   self.assertEqual(v['run_id'],job['run_id']+'-verify');self.assertIn('verify',v['argv']);self.assertNotIn('--output',v['argv']);self.assertEqual(m.flag(v['argv'],'--manifest-sha'),'f'*64)
   self.assertFalse(m.accepts_verify('succeeded',False,[{'status':'verified'}]))
   self.assertFalse(m.accepts_verify('failed',True,[{'status':'verified'}]))
 def test_engineering_requires_fullshape_cuda_and_bound_commit(self):
  m=self.module();self.assertTrue(hasattr(m,'check_engineering'),'engineering binding validator missing')
  proof={'status':'cuda_engineering_verified','commit':'a'*40,'synthetic_verify':{'status':'verified'},'vram':{'device':'cuda','catalog':120415,'batch':512,'queries':2520,'history':50,'steps':3,'variants':[{'finite_loss':True,'catalog_items_exported':120415,'users_exported':2520,'history_width':50,'declared_vram_mib':1024,'measured_mib':500.,'vram':{'method':'own_pid_nvidia_smi+torch_peaks','samples':[{'stage':stage,'pid':12,'own_process_mib':400.,'current_reserved_mib':200.,'max_allocated_mib':250.,'max_reserved_mib':300.} for stage in ('baseline','train-1','train-2','train-3','catalog-export','user-export','top50')]}}]}}
  self.assertTrue(m.check_engineering(proof,'a'*40))
  for bad in ({**proof,'commit':'b'*40},{**proof,'vram':{**proof['vram'],'device':'cpu'}},{**proof,'vram':{**proof['vram'],'catalog':100}}):
   with self.assertRaises(ValueError):m.check_engineering(bad,'a'*40)
if __name__=='__main__':unittest.main()
