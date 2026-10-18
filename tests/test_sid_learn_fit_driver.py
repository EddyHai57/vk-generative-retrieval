"""Finite fit->verify continuation owns only the fixed S2 pair."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
import test_sid_learn_fit_queue as queue_tests
ROOT=Path(__file__).resolve().parents[1]
def module():
 p=ROOT/'scripts/sid_learn_fit_driver.py';assert p.is_file(),'missing finite fit driver'
 s=importlib.util.spec_from_file_location('learn_fit_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class FitDriverTests(unittest.TestCase):
 def setup_run(self,r):
  req=queue_tests.FitQueueTests().request(r,'fit');req['source_commit']='a'*40
  req.update(memory_status='measured',cuda_proof={'status':'cuda_engineering_verified','memory_measurement':{'pid':1,'context_process_mib':500.,'context_reserved_mib':0.,'samples':[{'own_process_mib':1100.,'reserved_mib':600.,'max_reserved_mib':1000.,'max_allocated_mib':900.}]},'measured_mib':1500.,'declared_vram_mib':1792});req['job']['vram_mib']=1792
  fit=r/'runs/sid-learn-fit-v1';(fit/'out/payload').mkdir(parents=True);(fit/'logs').mkdir()
  (fit/'out/payload/run_manifest.json').write_text('{}');meta={'run_id':fit.name,'git':{'commit':'a'*40,'dirty':False,'dirty_files':[]},'config':{'sha256':'a'*64},'cwd':str(ROOT),'argv':req['job']['argv']};(fit/'run.json').write_text(json.dumps(meta))
  (r/'queue/claimed/001-sid-learn-fit-v1.json').write_text(json.dumps(req['job']))
  return req,fit
 def complete(self,m,req,fit):
  (fit/'terminal.json').write_text(json.dumps({'run_id':fit.name,'status':'succeeded','exit_code':0,'wall_cap_exceeded':False}))
  (fit/'logs/stdout.log').write_text(json.dumps({'status':'complete','engineering':False,'run_id':fit.name,'run_manifest_sha256':m.ops.sha(fit/'run.json'),'manifest_sha256':m.ops.sha(fit/'out/payload/run_manifest.json'),'source_commit':'a'*40,'fit_config_file_sha256':'a'*64,'input_spec_file_sha256':'b'*64})+'\n')
 def test_launch_requires_original_fit_unique_claimed_before_detaching(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);req,fit=self.setup_run(r);req['deadline_epoch']=__import__('time').time()+3600
   m.check_launch(req)
   for mode in ('duplicate','rejected','wrong_job'):
    path=r/('queue/rejected/001-sid-learn-fit-v1.json' if mode=='rejected' else 'queue/claimed/002-sid-learn-fit-v1.json')
    if mode=='wrong_job':
     path=r/'queue/claimed/001-sid-learn-fit-v1.json';path.write_text('{}')
    else:path.write_text(json.dumps(req['job']))
    with self.subTest(mode=mode),self.assertRaises(ValueError):m.check_launch(req)
    if mode!='wrong_job':path.rename(path.with_suffix('.held'))

 def test_unfinished_fit_does_not_publish(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   req,fit=self.setup_run(Path(d));state=m.observe(req)
   self.assertFalse(state['settled']);self.assertFalse(list((Path(d)/'queue/pending').iterdir()))
 def test_failed_or_rejected_fit_settles_without_verify(self):
  m=module()
  for mode in ('failed','rejected'):
   with tempfile.TemporaryDirectory() as d:
    req,fit=self.setup_run(Path(d))
    if mode=='failed':(fit/'terminal.json').write_text('{"status":"failed"}')
    else:(Path(d)/'queue/rejected/001-sid-learn-fit-v1.json').write_text('{}')
    state=m.observe(req);self.assertTrue(state['settled']);self.assertEqual(state['status'],'ALERT');self.assertFalse(list((Path(d)/'queue/pending').iterdir()))
 def test_complete_publishes_one_bound_verify_and_never_retries_uncertainty(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   req,fit=self.setup_run(Path(d));self.complete(m,req,fit);calls=[]
   def crash(*args):calls.append(args);raise OSError('publication interrupted')
   with self.assertRaises(OSError):m.observe(req,prepare=lambda *_:{},publisher=crash)
   with self.assertRaises(FileExistsError):m.observe(req,prepare=lambda *_:{},publisher=crash)
   self.assertEqual(len(calls),1);self.assertTrue((Path(d)/'ops/intent-sid-learn-fit-verify-v1.json').exists())
 def test_raw_complete_is_required_not_only_success_terminal(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   req,fit=self.setup_run(Path(d));self.complete(m,req,fit);(fit/'logs/stdout.log').write_text('{}\n')
   with self.assertRaises(ValueError):m.observe(req,prepare=lambda *_:{})
   self.assertFalse((Path(d)/'ops/intent-sid-learn-fit-verify-v1.json').exists())
 def test_verified_requires_original_stdout_and_own_verify_manifest(self):
  m=module()
  for mode in ('valid','wrong_manifest','missing_raw'):
   with tempfile.TemporaryDirectory() as d:
    r=Path(d);req,fit=self.setup_run(r);self.complete(m,req,fit)
    verify=m.queue.derive_verify(req,m.ops.sha(fit/'out/payload/run_manifest.json'),m.ops.sha(fit/'run.json'))
    vr=r/'runs/sid-learn-fit-verify-v1';(vr/'logs').mkdir(parents=True);(vr/'out').mkdir();(vr/'run.json').write_text('{}');(vr/'terminal.json').write_text('{"status":"succeeded","exit_code":0,"wall_cap_exceeded":false}')
    (r/'queue/claimed/002-sid-learn-fit-verify-v1.json').write_text(json.dumps(verify))
    proof={'status':'verified','run_id':vr.name,'run_manifest_sha256':m.ops.sha(vr/'run.json'),'manifest_sha256':'0'*64 if mode=='wrong_manifest' else m.ops.sha(fit/'out/payload/run_manifest.json'),'source_commit':'a'*40,'producer_run_manifest_sha256':m.ops.sha(fit/'run.json')}
    (vr/'out/verification.json').write_text(json.dumps(proof));(vr/'logs/stdout.log').write_text('{}\n' if mode=='missing_raw' else json.dumps(proof)+'\n')
    if mode=='valid':self.assertEqual(m.observe(req)['status'],'verified')
    else:
     with self.assertRaises(ValueError):m.observe(req)
if __name__=='__main__':unittest.main()
