"""Cost units and bounded supervision; only tiny local synthetic fixtures."""
import importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
def module():
 p=ROOT/'scripts/sid_scale_cost_v1.py';assert p.is_file(),'missing approved cost-model entry'
 s=importlib.util.spec_from_file_location('scale_cost',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class CostModelTests(unittest.TestCase):
 def test_balanced_effective_branching_units_are_hand_calculated(self):
  m=module();r=m.bhkm_work(10,k=3,depth=3,dim=2,iters=4)
  # 10*3, then 4*3+3*3+3*3, then 2*2+8*1 = 12 edges.
  self.assertEqual([x['edges'] for x in r['levels']],[30,30,12]);self.assertEqual(r['units'],576)
  self.assertEqual([x['nodes'] for x in r['levels']],[1,3,9]);self.assertEqual(r['data_bytes'],2560)
 def test_memory_is_analytic_and_never_torch_rss_slope(self):
  m=module()
  for n,expected in ((1250000,320000000),(12470000,3192320000)):
   r=m.bhkm_work(n);self.assertEqual(r['data_bytes'],expected);self.assertEqual(r['root_distance_bytes'],n*256*8)
 def test_no_complete_samples_means_unknown_constants(self):
  m=module();r=m.extrapolate([{'method':'bhkm','status':'CENSORED_WALL','n':250000,'wall_seconds':3600}],targets=(100,))
  self.assertEqual(r['bhkm_seconds_per_unit'],'UNKNOWN');self.assertEqual(r['rq_seconds_per_step'],'UNKNOWN');self.assertEqual(r['targets'][0]['bhkm_seconds'],'UNKNOWN')
 def test_fixed_rq_steps_and_fixed_per_item_visits_are_separate(self):
  m=module();units=m.bhkm_work(120415)['units']
  rows=[{'method':'bhkm','status':'COMPLETE','n':120415,'fit_seconds':units*.000001,'model_units':units},{'method':'rqvae','status':'COMPLETE','n':120415,'steps':3000,'fit_seconds':30.}]
  r=m.extrapolate(rows,targets=(120415,240830));a,b=r['targets'];self.assertAlmostEqual(a['rq_fixed_steps_seconds'],3000.);self.assertAlmostEqual(b['rq_fixed_steps_seconds'],3000.);self.assertEqual(b['rq_fixed_visits_steps'],600000);self.assertAlmostEqual(b['rq_fixed_visits_seconds'],6000.);self.assertTrue(r['assumptions'])
 def test_bhkm_memory_scenarios_use_measured_root_ratio_not_torch_rq_baseline(self):
  m=module();units=m.bhkm_work(250000)['units']
  b={'method':'bhkm','status':'COMPLETE','n':250000,'model_units':units,'fit_seconds':10.,'worker_peak_rss_bytes':2048000000}
  rq={'method':'rqvae','status':'COMPLETE','n':120415,'steps':3000,'fit_seconds':30.,'worker_peak_rss_bytes':999999999999}
  r=m.extrapolate([b,rq],targets=(1250000,))
  self.assertEqual(r['targets'][0]['bhkm_peak_rss_bytes_scenarios'],[10240000000]);self.assertEqual(r['bhkm_memory_calibration'][0]['rss_per_root_distance_byte'],4.)
  rq['worker_peak_rss_bytes']=1;again=m.extrapolate([b,rq],targets=(1250000,));self.assertEqual(r['bhkm_memory_calibration'],again['bhkm_memory_calibration'])

 def test_projection_is_not_censoring_and_platform_rejects_before_allocation(self):
  m=module()
  self.assertEqual(m.terminal_status(wall_hit=False,rss_hit=False,returncode=3),'FAILED')
  self.assertEqual(m.terminal_status(wall_hit=True,rss_hit=False,returncode=-15),'CENSORED_WALL')
  self.assertEqual(m.terminal_status(wall_hit=False,rss_hit=True,returncode=-15),'CENSORED_RSS')
  with patch.object(m.sys,'platform','darwin'),patch.dict(os.environ,VKGR_RUN_ID=m.RUN_ID),patch.object(m,'synthetic',side_effect=AssertionError('large local allocation')):
   with self.assertRaises(ValueError):m.worker('bhkm',250000,k=256,steps=3000,engineering=False)
 def test_rss_units_are_linux_kib_and_macos_bytes(self):
  m=module();self.assertEqual(m.peak_rss_bytes(123,'linux'),125952);self.assertEqual(m.peak_rss_bytes(123,'darwin'),123)
 def test_wall_and_rss_caps_keep_truthful_terminal_and_owned_subtree(self):
  m=module()
  for wall,rss,want in ((.1,1024,'CENSORED_WALL'),(5,.01,'CENSORED_RSS')):
   with self.subTest(want=want):
    r=m.supervise([sys.executable,'-B','-c','import time;time.sleep(2)'],wall_seconds=wall,rss_mib=rss)
    self.assertEqual(r['status'],want);self.assertNotEqual(r['pid'],os.getpid());self.assertTrue(r['terminated_own_group']);self.assertNotIn(os.getpid(),r['observed_pids'])
 def test_reported_peak_over_rss_cap_is_censored_even_between_samples(self):
  m=module();code='import json;print(json.dumps({"status":"COMPLETE","worker_peak_rss_bytes":2147483648}))'
  r=m.supervise([sys.executable,'-B','-c',code],wall_seconds=5,rss_mib=1024)
  self.assertEqual(r['status'],'CENSORED_RSS');self.assertEqual(r['reported_peak_rss_bytes'],2147483648)
 def test_final_wall_check_catches_completion_during_slow_resource_read(self):
  m=module();code='import json;print(json.dumps({"status":"COMPLETE","worker_peak_rss_bytes":1024}))'
  def slow(pid):time.sleep(.2);return 0,{pid}
  with patch.object(m,'own_tree_rss',side_effect=slow):r=m.supervise([sys.executable,'-B','-c',code],wall_seconds=.1,rss_mib=1024)
  self.assertEqual(r['status'],'CENSORED_WALL')

 def test_monitor_failure_stops_own_child_and_closes_pipes(self):
  import subprocess
  m=module();created=[];real=subprocess.Popen
  def record(*args,**kwargs):
   process=real(*args,**kwargs);created.append(process);return process
  try:
   with patch.object(m.subprocess,'Popen',side_effect=record),patch.object(m,'own_tree_rss',side_effect=PermissionError('monitor unavailable')):
    with self.assertRaises(PermissionError):m.supervise([sys.executable,'-B','-c','import time;time.sleep(2)'],wall_seconds=5,rss_mib=1024)
   self.assertIsNotNone(created[0].poll());self.assertTrue(created[0].stdout.closed);self.assertTrue(created[0].stderr.closed)
  finally:
   for process in created:
    if process.poll() is None:process.terminate();process.wait(timeout=3)
    for stream in (process.stdout,process.stderr):stream.close()

 def test_synthetic_used_bytes_are_hashed_without_exporting_vectors(self):
  import numpy as np,hashlib
  m=module();values=np.zeros((19,64),dtype='float32');values[:,0]=1.
  expected=hashlib.sha256((bytes.fromhex('0000803f')+bytes(63*4))*19).hexdigest()
  with patch.object(m,'synthetic',return_value=values):r=m.worker('bhkm',19,k=3,steps=3,engineering=True)
  self.assertEqual(r['synthetic_features_sha256'],expected);self.assertNotIn('values',r)

 def test_small_real_worker_reports_actual_iterations_without_export(self):
  m=module();r=m.worker('bhkm',19,k=3,steps=3,engineering=True)
  self.assertTrue(r['blas_threads']);self.assertTrue(all(x==1 for x in r['blas_threads']));self.assertEqual(r['status'],'COMPLETE');self.assertEqual(r['method'],'bhkm');self.assertEqual(r['n'],19);self.assertGreater(r['actual_units'],0);self.assertLessEqual(r['actual_units'],r['model_units']);self.assertEqual(len(r['actual_layers']),3);self.assertNotIn('sids',r);self.assertNotIn('nodes',r)
 def test_budget_cannot_grow_or_local_engineering_allocate_scale_points(self):
  m=module()
  for values in ({'overall_seconds':9001},{'point_seconds':3601},{'rss_mib':16385}):
   with self.assertRaises(ValueError):m.validate_request({**m.default_request(),**values})
  req={**m.default_request(),'engineering':True}
  with self.assertRaises(ValueError):m.validate_request(req)
class BenchmarkParentTests(unittest.TestCase):
 def request(self,m):
  return {**m.default_request(),'engineering':True,'bhkm_sizes':[19,23],'rq_n':32,'rq_steps':3,'branching':3,'point_seconds':10,'rss_mib':1024,'overall_seconds':20}
 def test_actual_tiny_parent_finishes_all_workers_and_only_returns_aggregates(self):
  m=module();r=m.run_benchmark(self.request(m))
  self.assertEqual(r['status'],'COMPLETE');self.assertEqual(len(r['records']),3)
  self.assertTrue(all(x['status']=='COMPLETE' for x in r['records']));self.assertTrue(r['engineering']);self.assertNotIn('state',json.dumps(r));self.assertEqual(r['projection']['kind'],'cost_model_projection_not_measurement')
 def test_expired_overall_budget_never_starts_a_new_child(self):
  m=module();values=iter((0.,21.,21.,21.,21.,21.))
  def forbidden(*args,**kwargs):raise AssertionError('child started after overall cap')
  r=m.run_benchmark(self.request(m),clock=lambda:next(values),supervisor=forbidden)
  self.assertEqual(r['status'],'INCOMPLETE');self.assertTrue(all(x['status']=='NOT_RUN_BUDGET' for x in r['records']));self.assertEqual(r['projection']['rq_seconds_per_step'],'UNKNOWN')
 def test_fixed_original_l2_reference_bytes_are_required(self):
  m=module()
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'src/vkgr').mkdir(parents=True);(r/'src/vkgr/sid_rqvae.py').write_text('wrong');(r/'src/vkgr/sid_rqvae_r.py').write_text('wrong')
   with self.assertRaises(ValueError):m.check_l2_source(r)
 def test_scientific_or_nonfinite_worker_failure_is_not_retried(self):
  m=module();calls=[]
  def failed(argv,**kwargs):calls.append(argv);return {'status':'FAILED','reason':'nonfinite','wall_seconds':.1}
  r=m.run_benchmark(self.request(m),supervisor=failed)
  self.assertEqual(r['status'],'FAILED');self.assertEqual(len(calls),1);self.assertEqual([x['status'] for x in r['records']],['FAILED','NOT_RUN_FAILURE','NOT_RUN_FAILURE'])
class BenchmarkCLITests(unittest.TestCase):
 def test_actual_tiny_cli_writes_exclusive_aggregate_and_marks_engineering(self):
  import subprocess,hashlib
  m=module()
  with tempfile.TemporaryDirectory() as d:
   r=Path(d).resolve();req=BenchmarkParentTests().request(m);p=r/'request.json';p.write_text(json.dumps(req));out=r/'out'
   args=[sys.executable,'-B',str(ROOT/'scripts/sid_scale_cost_v1.py'),'--request',str(p),'--request-sha',hashlib.sha256(p.read_bytes()).hexdigest(),'--output',str(out)]
   result=subprocess.run(args,capture_output=True,text=True);self.assertEqual(result.returncode,0,result.stderr);raw=json.loads(result.stdout)
   self.assertEqual(raw['status'],'engineering_benchmark_reported');self.assertEqual(set(x.name for x in out.iterdir()),{'report.json'});self.assertEqual(json.loads((out/'report.json').read_text())['status'],'COMPLETE')
   before=(out/'report.json').read_bytes();again=subprocess.run(args,capture_output=True,text=True);self.assertNotEqual(again.returncode,0);self.assertEqual((out/'report.json').read_bytes(),before)
 def metadata(self,r):
  m=module();run=r/'sid-scale-cost-v1';run.mkdir();out=run/'out';req=r/'request.json';req.write_text('{}');pin=m.hash_file(req);args=['--request',str(req),'--request-sha',pin,'--output',str(out)]
  meta={'run_id':'sid-scale-cost-v1','git':{'commit':'a'*40,'dirty':False,'dirty_files':[]},'cwd':str(ROOT),'config':{'path':str(req),'sha256':pin},'argv':[sys.executable,'-B',str(ROOT/'scripts/sid_scale_cost_v1.py'),*args]};(run/'run.json').write_text(json.dumps(meta));return m,run,out,req,pin,args,meta
 def test_formal_job_binds_actual_manifest_source_config_and_argv(self):
  import copy
  for field in ('valid','git','cwd','config','argv','run_id'):
   with tempfile.TemporaryDirectory() as d:
    m,run,out,req,pin,args,meta=self.metadata(Path(d).resolve())
    if field=='git':meta[field]['commit']='b'*40
    if field=='cwd':meta[field]='/wrong'
    if field=='config':meta[field]['sha256']='0'*64
    if field=='argv':meta[field]=[sys.executable,'-c','pass',*args]
    if field=='run_id':meta[field]='other'
    (run/'run.json').write_text(json.dumps(meta))
    with patch.dict(os.environ,VKGR_RUN_ID='sid-scale-cost-v1'):
     if field=='valid':self.assertEqual(m.bind_parent(out,req,pin,args,{'commit':'a'*40},False)['run_manifest_sha256'],m.hash_file(run/'run.json'))
     else:
      with self.assertRaises(ValueError):m.bind_parent(out,req,pin,args,{'commit':'a'*40},False)
if __name__=='__main__':unittest.main()
