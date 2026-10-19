"""One registered metadata-only S4 job: dry, intent, original receipts."""
import copy,hashlib,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module():
 p=ROOT/'scripts/sid_scale_cost_queue.py';assert p.is_file(),'missing versioned scale queue'
 s=importlib.util.spec_from_file_location('scale_queue',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class QueueTests(unittest.TestCase):
 def request(self,r):
  from test_sid_scale_cost_v1 import module as cost_module
  cost=cost_module();ops=r/'ops';ops.mkdir();(r/'runs').mkdir()
  for name in ('pending','claimed','rejected'):(r/'queue'/name).mkdir(parents=True)
  cfg=ops/'benchmark-request.json';cfg.write_text(json.dumps(cost.default_request()));pin=hashlib.sha256(cfg.read_bytes()).hexdigest();pinned=ops/'scheduler-pinned-cost-v1.json'
  job={'schema':'vkgr-queue-job-v1','kind':'ops','run_id':'sid-scale-cost-v1','cwd':str(ROOT),'config':str(cfg),'seed':42,'cpu_cores':1,'ram_mib':18432,'vram_mib':0,'wall_hours':2.5,'disk_gib':1,'no_admission':False,'stop_mode':'signal','pinned_manifest':str(pinned),'pinned_files':[],'inputs':[str(cfg)],'env':['PYTHONPATH='+str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE=1',*[key+'=1' for key in cost.THREAD_ENV],'VKGR_PINNED_MANIFEST='+str(pinned)],'argv':[sys.executable,'-B',str(ROOT/'scripts/sid_scale_cost_v1.py'),'--request',str(cfg),'--request-sha',pin,'--output',str(r/'runs/sid-scale-cost-v1/out')]}
  return {'root':str(r),'deployment':str(ROOT),'ops':str(ops),'scheduler_ops':str(r/'scheduler_ops'),'source_commit':'a'*40,'files':{},'job':job}
 def test_only_fixed_parent_recipe_inputs_and_no_gpu_are_allowed(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   req=self.request(Path(d).resolve());m.validate(req)
   for field,value in (('run_id','other'),('cpu_cores',2),('vram_mib',256),('wall_hours',3.),('inputs',['/unapproved.npy']),('env',[]),('argv',[sys.executable,'-c','pass'])):
    bad=copy.deepcopy(req);bad['job'][field]=value
    with self.subTest(field=field),self.assertRaises(ValueError):m.validate(bad)
 def test_engineering_or_changed_points_are_not_publishable(self):
  m=module()
  for field,value in (('engineering',True),('bhkm_sizes',[250001,300000]),('rq_steps',3001),('point_seconds',1000)):
   with tempfile.TemporaryDirectory() as d:
    req=self.request(Path(d).resolve());p=Path(req['job']['config']);config=json.loads(p.read_text());config[field]=value;p.write_text(json.dumps(config));req['job']['argv'][6]=m.sha(p)
    with self.assertRaises(ValueError):m.validate(req)
 def test_any_existing_attempt_or_rejected_reason_blocks_fresh_submission(self):
  m=module()
  for mode in ('run','pending','claimed','rejected','reason','intent','hidden'):
   with tempfile.TemporaryDirectory() as d:
    r=Path(d).resolve();req=self.request(r)
    if mode=='run':(r/'runs/sid-scale-cost-v1').mkdir()
    elif mode=='intent':(r/'ops/intent-sid-scale-cost-v1.json').write_text('{}')
    else:
     stage=mode if mode in ('pending','claimed','rejected') else 'pending' if mode=='hidden' else 'rejected';name='.001-sid-scale-cost-v1.json.tmp' if mode=='hidden' else '001-sid-scale-cost-v1.reason' if mode=='reason' else '001-sid-scale-cost-v1.json';(r/'queue'/stage/name).write_text('{}')
    with self.subTest(mode=mode),self.assertRaises(FileExistsError):m.check_fresh(req)
 def test_publication_uncertainty_keeps_intent_and_forbids_retry(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   r=Path(d).resolve();req=self.request(r);calls=[]
   def crash(*args):calls.append(1);raise OSError('uncertain publication')
   with self.assertRaises(OSError):m.submit(req,prepare=lambda *_:{},publisher=crash)
   self.assertTrue((r/'ops/intent-sid-scale-cost-v1.json').is_file())
   with self.assertRaises(FileExistsError):m.submit(req,prepare=lambda *_:{},publisher=crash)
   self.assertEqual(len(calls),1)
 def test_same_dry_digest_is_required_before_publication(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   r=Path(d).resolve();req=self.request(r);m.dry(req,prepare=lambda *_:{})
   dry=r/'ops/dry-sid-scale-cost-v1.json';value=json.loads(dry.read_text());value['job_sha256']='0'*64;dry.write_text(json.dumps(value))
   with self.assertRaises(ValueError):m.submit_checked(req,prepare=lambda *_:{})
   self.assertFalse(list((r/'queue/pending').iterdir()));self.assertFalse((r/'ops/intent-sid-scale-cost-v1.json').exists())
 def test_clean_isolated_git_source_requires_exact_commit_and_all_dependencies(self):
  import subprocess,shutil
  m=module();cost=m.runtime({'deployment':str(ROOT)})
  with tempfile.TemporaryDirectory() as d:
   clone=Path(d).resolve();names=[*cost.producer()['files'],'scripts/sid_scale_cost_queue.py','scripts/sid_scale_cost_deploy.py']
   for name in names:
    path=clone/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
   subprocess.run(['git','init','-q',str(clone)],check=True);subprocess.run(['git','-C',str(clone),'add',*names],check=True)
   subprocess.run(['git','-C',str(clone),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-q','-m','isolated source'],check=True)
   commit=subprocess.check_output(['git','-C',str(clone),'rev-parse','HEAD'],text=True).strip();files={name:m.sha(clone/name) for name in names};req={'deployment':str(clone),'source_commit':commit,'files':files}
   self.assertEqual(m.identity(req)['commit'],commit)
   with self.assertRaises(ValueError):m.identity({**req,'source_commit':'0'*40})
   missing=dict(files);del missing['src/vkgr/sid_tok_audit.py']
   with self.assertRaises(ValueError):m.identity({**req,'files':missing})
   changed=dict(files);changed['scripts/sid_scale_cost_queue.py']='0'*64
   with self.assertRaises(ValueError):m.identity({**req,'files':changed})
   (clone/'src/vkgr/sid_tok_audit.py').write_text('changed bytes')
   with self.assertRaises(ValueError):m.identity(req)
   (clone/'scripts/sid_scale_cost_v1.py').write_text("raise AssertionError('unverified runtime executed')\n")
   with self.assertRaises(ValueError):m.identity(req)

 def test_actual_metadata_python_c_reaches_source_gate_without_file_global(self):
  import subprocess
  m=module();req={'deployment':str(ROOT),'source_commit':'0'*40};script=(ROOT/'scripts/sid_scale_cost_queue.py').read_text()
  result=subprocess.run([sys.executable,'-B','-c',script],input=json.dumps(req),text=True,capture_output=True)
  self.assertNotEqual(result.returncode,0);self.assertIn('fixed deployment commit differs',result.stderr);self.assertNotIn('NameError',result.stderr)
class ReceiptTests(unittest.TestCase):
 request=QueueTests.request
 def test_original_stdout_job_source_terminal_and_recomputed_projection_are_bound(self):
  m=module()
  for mode in ('valid','incomplete','raw','binding','projection','steps','source','claimed','duplicate','pending','terminal','dataflag','false_complete','datahash'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    r=Path(d).resolve();req=self.request(r);cost=m.runtime(req);base=r/'runs/sid-scale-cost-v1';(base/'out').mkdir(parents=True);(base/'logs').mkdir()
    meta={'run_id':'sid-scale-cost-v1','git':{'commit':'a'*40,'dirty':False,'dirty_files':[]},'argv':req['job']['argv'],'cwd':str(ROOT),'config':{'path':req['job']['config'],'sha256':m.sha(req['job']['config'])}};(base/'run.json').write_text(json.dumps(meta))
    terminal={'run_id':'sid-scale-cost-v1','status':'succeeded','exit_code':0,'wall_cap_exceeded':False};(base/'terminal.json').write_text(json.dumps(terminal))
    (r/'queue/claimed/001-sid-scale-cost-v1.json').write_text(json.dumps(req['job']))
    producer=cost.producer();source={'commit':'a'*40,'dirty':False,**producer};binding={'run_id':'sid-scale-cost-v1','run_manifest_sha256':m.sha(base/'run.json'),'request_file_sha256':m.sha(req['job']['config']),'source_commit':'a'*40}
    records=[{'method':'bhkm','n':n,'status':'COMPLETE','model_units':cost.bhkm_work(n)['units'],'actual_units':cost.bhkm_work(n)['units'],'fit_seconds':1.,'worker_peak_rss_bytes':10000000,'synthetic_features_sha256':'d'*64} for n in (250000,300000)]+[{'method':'rqvae','n':120415,'status':'COMPLETE','steps':3000,'fit_seconds':30.,'synthetic_features_sha256':'e'*64}]
    report={'schema':'sid-scale-cost-report-v1','status':'COMPLETE','engineering':False,'request':cost.default_request(),'source':copy.deepcopy(source),'producer':producer,'fixed_l2':cost.check_l2_source(),'binding':copy.deepcopy(binding),'synthetic_only':True,'test_read':False,'checkpoint_written':False,'mapping_exported':False,'records':records,'projection':cost.extrapolate(records)}
    if mode in ('incomplete','false_complete'):
     records[0]['status']='CENSORED_RSS';report['projection']=cost.extrapolate(records)
     if mode=='incomplete':report['status']='INCOMPLETE'
    if mode=='datahash':records[0]['synthetic_features_sha256']='not_a_sha'
    if mode=='binding':report['binding']['run_manifest_sha256']='0'*64
    if mode=='projection':report['projection']['targets'][0]['rq_fixed_steps_seconds']=1.
    if mode=='steps':records[2]['steps']=2999;report['projection']=cost.extrapolate(records)
    if mode=='source':report['source']['commit']='b'*40
    if mode=='claimed':(r/'queue/claimed/001-sid-scale-cost-v1.json').write_text('{}')
    if mode=='duplicate':(r/'queue/claimed/002-sid-scale-cost-v1.json').write_text(json.dumps(req['job']))
    if mode=='pending':(r/'queue/pending/002-sid-scale-cost-v1.json').write_text(json.dumps(req['job']))
    if mode=='terminal':terminal['status']='failed';(base/'terminal.json').write_text(json.dumps(terminal))
    if mode=='dataflag':report['mapping_exported']=True
    (base/'out/report.json').write_text(json.dumps(report));raw={'status':'benchmark_reported','report_status':report['status'],'report_sha256':m.sha(base/'out/report.json'),'synthetic_only':True,**binding};(base/'logs/stdout.log').write_text('{}\n' if mode=='raw' else json.dumps(raw)+'\n')
    if mode in ('valid','incomplete'):
     result=m.accepted_report(req,source);self.assertEqual(result['status'],'benchmark_receipt_accepted');self.assertEqual(result['calibration_status'],'INCOMPLETE' if mode=='incomplete' else 'COMPLETE')
    else:
     with self.assertRaises(ValueError):m.accepted_report(req,source)
if __name__=='__main__':unittest.main()
