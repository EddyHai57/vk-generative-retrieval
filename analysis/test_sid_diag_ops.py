import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from io import StringIO
class OpsTests(unittest.TestCase):
 def module(self):
  p=Path(__file__).with_name('sid_diag_ops.py');self.assertTrue(p.exists(),'new ops module absent')
  s=importlib.util.spec_from_file_location('sid_ops',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_new_never_overwrites(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x';m.new(p,{'x':1})
   with self.assertRaises(FileExistsError):m.new(p,{'x':2})
   self.assertEqual(json.loads(p.read_text()),{'x':1})
 def test_queue_reasons_and_duplicate_guard(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);q=root/'queue';(root/'runs').mkdir()
   for k in ('pending','claimed','rejected'):(q/k).mkdir(parents=True)
   rid='sid-fix-s0-first-v1';p=q/'rejected'/('1-'+rid+'.json');p.write_text('{}');p.with_suffix('.reason').write_text('bad pin')
   v=m.where(q,rid);self.assertEqual(v['reasons'],{'1-'+rid+'.reason':'bad pin'})
   with self.assertRaises(FileExistsError):m.publish(q,root/'runs',{'run_id':rid})
   self.assertFalse(list((q/'pending').iterdir()))
 def test_job_no_cuda_one_cpu_plan_binding(self):
  m=self.module();plan={'mode':'first_batch'}
  j=m.job(plan,'a'*64)
  self.assertEqual(j['cpu_cores'],1);self.assertEqual(j['vram_mib'],0)
  self.assertIn('CUDA_VISIBLE_DEVICES=',j['env']);self.assertIn('--plan-sha256',j['argv'])
  self.assertEqual(j['argv'][j['argv'].index('--plan-sha256')+1],'a'*64)
  self.assertNotIn('checkpoint',' '.join(j['argv']));self.assertIn('--output',j['argv'])
 def test_readonly_status_contains_real_metadata_preflight(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);dep=base/'code';dep.mkdir();ops=base/'ops';ops.mkdir()
   (ops/'first_batch-plan.json').write_text('{"mode":"first_batch"}')
   with patch.object(m,'DEP',dep),patch.object(m,'OPS',ops),patch.object(m,'R',base),patch.object(m,'identity'),patch.object(m,'inventory',return_value={}),patch.object(m,'preflight',return_value={'status':'preflight_ok'}) as pf:
    v=m.status();self.assertEqual(v['preflight']['first_batch']['status'],'preflight_ok');pf.assert_called_once()
 def test_wrapper_stdin_request(self):
  m=self.module()
  with patch.object(m.sys,'argv',['-c']),patch.object(m.sys,'stdin',StringIO('{"action":"deploy"}')):
   self.assertEqual(m.request(),{'action':'deploy'})
  with patch.object(m.sys,'argv',['-']),patch.object(m.sys,'stdin',StringIO('')):
   self.assertEqual(m.request(),{})
 def test_modes_closed(self):
  m=self.module()
  for mode in ('all','train','test'):
   with self.assertRaises(ValueError):m.job({'mode':mode},'a'*64)
 def test_metadata_limit(self):
  m=self.module();self.assertTrue(m.metadata_guard({'x':'a'},{}))
  with self.assertRaises(ValueError):m.metadata_guard({'x':'a'*50000},{})
 def test_publish_unique(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);q=root/'queue';(root/'runs').mkdir()
   for k in ('pending','claimed','rejected'):(q/k).mkdir(parents=True)
   rid='sid-fix-s0-first-v1';name=m.publish(q,root/'runs',{'run_id':rid})
   self.assertEqual(json.loads((q/'pending'/name).read_text())['run_id'],rid)
   self.assertEqual(json.loads((q/'pending'/name).read_text())['schema'],'vkgr-queue-job-v1')
   with self.assertRaises(FileExistsError):m.publish(q,root/'runs',{'run_id':rid})
if __name__=='__main__':unittest.main()
