"""Unique formal S2 stages: resource evidence, no retry and verify derivation."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module():
 p=ROOT/'scripts/sid_learn_fit_queue.py';assert p.is_file(),'missing versioned fit queue'
 spec=importlib.util.spec_from_file_location('learn_fit_queue',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class FitQueueTests(unittest.TestCase):
 def request(self,r,stage='preflight'):
  for s in ('pending','claimed','rejected'):(r/'queue'/s).mkdir(parents=True)
  (r/'ops').mkdir();(r/'runs').mkdir()
  rid={'preflight':'sid-learn-input-preflight-v1','cuda-preview':'sid-learn-fit-cuda-v1','fit':'sid-learn-fit-v1','verify':'sid-learn-fit-verify-v1'}[stage]
  args=[sys.executable,'-B','-m','vkgr.sid_learn_run',stage,'--config',str(ROOT/'configs/sid_learn/fit-s42.json'),'--config-sha','a'*64,'--inputs',str(r/'ops/inputs.json'),'--inputs-sha','b'*64,'--output',str(r/'runs'/rid/'out')]
  return {'root':str(r),'runs':str(r/'runs'),'ops':str(r/'ops'),'deployment':str(ROOT),'job':{'run_id':rid,'argv':args,'cpu_cores':1,'vram_mib':256 if stage=='preflight' else 4096,'no_admission':False,'cwd':str(ROOT),'config':str(ROOT/'configs/sid_learn/fit-s42.json'),'wall_hours':4.,'env':['CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1']},'memory_status':'not_measured'}
 def test_actual_metadata_python_c_loads_pinned_runtime_without_file_global(self):
  import subprocess,os
  req={'deployment':str(ROOT),'source_commit':'0'*40,'files':{}}
  script=(ROOT/'scripts/sid_learn_fit_queue.py').read_text()
  result=subprocess.run([sys.executable,'-B','-c',script],input=json.dumps(req),text=True,capture_output=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  self.assertNotEqual(result.returncode,0)
  self.assertIn('deployed commit differs from pinned source',result.stderr)
  self.assertNotIn('NameError',result.stderr)

 def test_initial_cuda_requires_explicit_unmeasured_reservation(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   req=self.request(Path(d),'cuda-preview');m.validate(req)
   for field,value in (('memory_status','measured'),('memory_status',None)):
    bad={**req,field:value}
    with self.assertRaises(ValueError):m.validate(bad)
 def test_formal_fit_requires_measured_context_inclusive_evidence(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   req=self.request(Path(d),'fit')
   with self.assertRaises(ValueError):m.validate(req)
   memory={'pid':1,'context_process_mib':500.,'context_reserved_mib':0.,'samples':[{'own_process_mib':1100.,'reserved_mib':600.,'max_reserved_mib':1000.,'max_allocated_mib':900.}]}
   req.update(memory_status='measured',cuda_proof={'memory_measurement':memory,'measured_mib':1500.,'declared_vram_mib':1792,'status':'cuda_engineering_verified'})
   req['job']['vram_mib']=1792;m.validate(req)
   req['job']['vram_mib']=1536
   with self.assertRaises(ValueError):m.validate(req)
 def test_wrong_entry_stage_interpreter_env_and_existing_queue_fail(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   req=self.request(Path(d));m.validate(req)
   for key,value in (('argv',[sys.executable,'-c','pass','-m','vkgr.sid_learn_run','preflight']),('run_id','other'),('cpu_cores',2),('env',[])):
    bad=copy.deepcopy(req);bad['job'][key]=value
    with self.subTest(key=key),self.assertRaises(ValueError):m.validate(bad)
   (Path(d)/'queue/rejected/001-sid-learn-input-preflight-v1.json').write_text('{}')
   with self.assertRaises(FileExistsError):m.check_fresh(req)
 def test_intent_survives_uncertain_publication_and_blocks_retry(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   req=self.request(Path(d))
   def fail(*args):raise OSError('uncertain publication')
   with self.assertRaises(OSError):m.submit(req,prepare=lambda *_:{},publisher=fail)
   self.assertTrue((Path(d)/'ops/intent-sid-learn-input-preflight-v1.json').exists())
   with self.assertRaises(FileExistsError):m.submit(req,prepare=lambda *_:{},publisher=fail)
 def test_verify_is_derived_from_same_fit_and_immutable_bundle_run_hashes(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   req=self.request(Path(d),'fit');v=m.derive_verify(req,'c'*64,'d'*64)
   self.assertEqual(v['run_id'],'sid-learn-fit-verify-v1');self.assertEqual(v['wall_hours'],4.)
   self.assertEqual(v['argv'][4],'verify');self.assertEqual(m.ops.flag(v['argv'],'--producer-run-sha'),'d'*64);self.assertEqual(m.ops.flag(v['argv'],'--sha'),'c'*64)
   self.assertEqual(v['config'],req['job']['config']);self.assertEqual(req['job']['argv'][4],'fit')
if __name__=='__main__':unittest.main()
