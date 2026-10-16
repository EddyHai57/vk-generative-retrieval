"""Fresh RED guards for atomic owner, real leakage receipt and own-PID memory."""
import copy,importlib.util,json,os,tempfile,time,unittest
from pathlib import Path
from unittest import mock
import torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
def ops():
 spec=importlib.util.spec_from_file_location('stack_fix_ops',ROOT/'scripts/seq_combo_stack_ops.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class OwnerFixTests(unittest.TestCase):
 def fixture(self,tmp):
  m=ops();t=Path(tmp);m.configure({k:str(t/k) for k in m.PATH_KEYS});m.OPS.mkdir();m.Q.mkdir(parents=True)
  for key in ('pending','claimed','rejected'):(m.Q/key).mkdir()
  return m,{'deadline_epoch':time.time()+60,'entries':[{'job':{'run_id':rid,'argv':[]},'expected':{}} for rid in m.RIDS]}
 def test_partial_publication_keeps_owner_plan_alert_and_never_retries(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,p=self.fixture(tmp);self.assertTrue(hasattr(m,'publish_intents'),'owner publication protocol missing');original=m.publish;calls=[]
   def fail_third(q,runs,job):
    calls.append(job['run_id'])
    if len(calls)==3:raise OSError('third publication fails')
    return original(q,runs,job)
   with mock.patch.object(m,'prepare_job'),mock.patch.object(m,'publish',side_effect=fail_third):rows,failed=m.publish_intents(p)
   self.assertTrue(failed);self.assertEqual([x['status'] for x in rows.values()],['published','published','failed']);self.assertTrue((m.OPS/'publication-ALERT.json').exists());self.assertEqual(len(calls),3)
   with self.assertRaises(FileExistsError):m.publish_intents(p)
 def test_controller_start_failure_publishes_nothing_and_seals_plan(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,p=self.fixture(tmp);self.assertTrue(hasattr(m,'start_submission'),'atomic owner startup missing')
   with mock.patch.object(m.subprocess,'Popen',side_effect=OSError('cannot start detached owner')):
    with self.assertRaises(OSError):m.start_submission(p)
   self.assertEqual(m.load(m.OPS/'plan.json'),p);self.assertTrue((m.OPS/'controller-ALERT.json').exists());self.assertFalse(list((m.Q/'pending').glob('*.json')))
 def test_ready_without_release_cannot_publish(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,p=self.fixture(tmp);self.assertTrue(hasattr(m,'ready_handshake'),'ready release guard missing')
   with mock.patch.object(m.os,'getsid',return_value=os.getpid()),mock.patch.object(m.time,'time',side_effect=[100,100,131]):
    p['deadline_epoch']=200
    with self.assertRaises(TimeoutError):m.ready_handshake(p)
   self.assertTrue((m.OPS/'driver-ready.json').exists());self.assertFalse(list((m.Q/'pending').glob('*.json')))
 def test_published_train_gets_only_one_verifier_when_batch_partial(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,p=self.fixture(tmp);self.assertTrue(hasattr(m,'observation_cycle'),'publication-aware observation missing')
   for e in p['entries']:e['job']['argv']=['py','-m','vkgr.retrieval_seq_combo_stack_run','train','--output','o'];e['job']['inputs']=[]
   pubs={rid:{'status':'published' if i<2 else 'failed'} for i,rid in enumerate(m.RIDS)}
   def state(e):return {'run_id':e['job']['run_id'],'train_status':'succeeded','verify_status':None,'rejected':False,'verified':False,'manifest_sha256':'a'*64,'exceptions':[]}
   with mock.patch.object(m,'snapshot',side_effect=state),mock.patch.object(m,'prepare_job'):
    m.observation_cycle(p,pubs);m.observation_cycle(p,pubs)
   jobs=[json.loads(x.read_text()) for x in (m.Q/'pending').glob('*.json')];self.assertEqual(len(jobs),2);self.assertTrue(all(x['run_id'].endswith('-verify') for x in jobs))
 def test_intent_write_failure_after_two_publications_does_not_lose_owner(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,p=self.fixture(tmp);original=m.new
   def failing(path,value):
    if Path(path).name=='publication-intent-02.json':raise OSError('intent storage fault')
    return original(path,value)
   with mock.patch.object(m,'prepare_job'),mock.patch.object(m,'new',side_effect=failing):rows,failed=m.publish_intents(p)
   self.assertTrue(failed);self.assertEqual([x['status'] for x in rows.values()],['published','published','failed']);self.assertTrue((m.OPS/'publication-ALERT.json').exists())
 def test_queue_visible_then_exception_is_never_duplicated(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,p=self.fixture(tmp);original=m.publish
   def after_link(q,runs,job):
    original(q,runs,job);raise OSError('lost publication acknowledgement')
   with mock.patch.object(m,'prepare_job'),mock.patch.object(m,'publish',side_effect=after_link):rows,failed=m.publish_intents(p)
   self.assertTrue(failed);self.assertEqual([x['status'] for x in rows.values()],['published','not_attempted','not_attempted']);self.assertEqual(len(list((m.Q/'pending').glob('*.json'))),1)
 def test_result_receipt_write_failure_preserves_published_job_for_observation(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,p=self.fixture(tmp);original=m.new
   def failing(path,value):
    if Path(path).name=='publication-result-00.json':raise OSError('result storage fault')
    return original(path,value)
   with mock.patch.object(m,'prepare_job'),mock.patch.object(m,'new',side_effect=failing):rows,failed=m.publish_intents(p)
   self.assertTrue(failed);self.assertEqual([x['status'] for x in rows.values()],['published','not_attempted','not_attempted']);self.assertTrue((m.OPS/'publication-ALERT.json').exists())
 def test_partial_publication_stays_alert_while_observation_continues(self):
  with tempfile.TemporaryDirectory() as tmp:
   m,p=self.fixture(tmp);p['deadline_epoch']=10;m.new(m.OPS/'plan.json',p)
   pubs={rid:{'status':'published' if i<2 else 'failed'} for i,rid in enumerate(m.RIDS)}
   states=[{'run_id':rid,'verified':False,'train_status':None,'verify_status':None,'exceptions':[] if i<2 else ['unpublished']} for i,rid in enumerate(m.RIDS)]
   with mock.patch.object(m.os,'chdir'),mock.patch.object(m,'owner_source'),mock.patch.object(m,'ready_handshake'),mock.patch.object(m,'publish_intents',return_value=(pubs,True)),mock.patch.object(m,'observation_cycle',return_value=(states,False)),mock.patch.object(m.time,'time',side_effect=[1,2,11]),mock.patch.object(m.time,'sleep'),mock.patch.object(m.sys,'argv',['ops','--plan-sha',m.sha(m.OPS/'plan.json')]):m.driver()
   snapshots=list(m.OPS.glob('summary-[0-9]*.json'));self.assertEqual(len(snapshots),1);self.assertEqual(m.load(snapshots[0])['status'],'ALERT')
class LeakageFixTests(unittest.TestCase):
 def helper(self):
  p=ROOT/'src/vkgr/seq_combo_stack_leakage.py';self.assertTrue(p.exists(),'stack real-path matrix missing')
  from vkgr import seq_combo_stack_leakage as leak
  return leak
 def test_production_helper_matrix_with_synthetic_fixture_and_scope(self):
  leak=self.helper()
  from vkgr import retrieval_seq_combo_stack_run as r
  data=r.synthetic_data();data['synthetic']=False # fixture branch coverage, not real evidence
  record=leak.audit_runtime(data,r,r.default_config(),train_count=16)
  self.assertEqual(record['status'],'pass');self.assertEqual(record['train_samples'],16)
  for key in ('L1','L2','L3','L4','L5','L8','L9'):self.assertEqual(record['checks'][key]['status'],'pass',key)
  for key in ('L6','L7','L10'):self.assertEqual(record['checks'][key]['status'],'STATIC_REVIEWED',key)
  self.assertFalse(record['checkpoint_read']);self.assertFalse(record['training'])
 def test_target_leak_stops_and_keeps_failure_item(self):
  leak=self.helper()
  from vkgr import retrieval_seq_combo_stack_run as r
  data=r.synthetic_data();data['synthetic']=False
  original=r.fr._tensors2
  def bad(batch,idmap,width,feature):
   h,s=original(batch,idmap,width,feature);s[:,0,1]=torch.tensor([2 if x.get('past_likes')==[] else 100 for x in batch]);return h,s
  with mock.patch.object(r.fr,'_tensors2',side_effect=bad):record=leak.audit_runtime(data,r,r.default_config(),train_count=16)
  self.assertEqual(record['status'],'FAIL');self.assertEqual(record['checks']['L1']['status'],'FAIL');self.assertNotEqual(record['checks']['L2']['status'],'pass')
 def test_admission_requires_stack_receipt_and_exact_binding(self):
  m=ops();self.assertTrue(hasattr(m,'check_leakage_receipt'),'receipt gate missing')
  source={'git_commit':'a'*40,'source_sha':'b'*64};data={'rows_sha256':'c'*64};entry={'config_sha256':'d'*64,'expected_binding_sha256':'e'*64,'data_hashes':data}
  record={'schema':'seq-combo-stack-leakage-v1','status':'pass','model':'SeqComboStack','engineering':False,'source':source,'config_sha256':entry['config_sha256'],'binding_sha256':entry['expected_binding_sha256'],'data_hashes':data,'train_samples':512,'validation_samples':2520,'test_read':False,'checkpoint_read':False,'training':False,'checks':{k:{'status':'pass','samples':1} for k in ('L1','L2','L3','L4','L5','L8','L9')}}
  record['checks'].update({k:{'status':'STATIC_REVIEWED','source_sha256':'f'*64} for k in ('L6','L7','L10')})
  for k in ('L1','L2','L3','L8'):record['checks'][k]['samples']=3032
  for k in ('L5','L9'):record['checks'][k]['samples']=120415
  record['checks']['L2'].update(train_samples=512,validation_samples=2520);record['checks']['L9']['train_samples']=512
  record['checks']['L3_own_feedback']={'status':'pass','samples':16};record['nonzero']={'gate':.2,'recency':True,'beta':True}
  self.assertTrue(m.check_leakage_receipt(record,entry,source))
  incomplete=copy.deepcopy(record);incomplete['checks']['L1']['samples']=1
  with self.assertRaises(ValueError):m.check_leakage_receipt(incomplete,entry,source)
  incomplete=copy.deepcopy(record);incomplete['nonzero']['gate']=0.
  with self.assertRaises(ValueError):m.check_leakage_receipt(incomplete,entry,source)
  for key,bad in [('model','old_combo'),('engineering',True),('source',{}),('config_sha256','0'*64),('data_hashes',{}),('status','FAIL')]:
   with self.assertRaises(ValueError):m.check_leakage_receipt({**record,key:bad},entry,source)
class MemoryFixTests(unittest.TestCase):
 def helper(self):
  p=ROOT/'src/vkgr/seq_combo_stack_memory.py';self.assertTrue(p.exists(),'own process memory proof missing')
  from vkgr import seq_combo_stack_memory as mem
  return mem
 def test_conservative_context_plus_allocator_peak_and_tamper(self):
  mem=self.helper();samples=[{'stage':k,'pid':12,'own_process_mib':400.,'current_reserved_mib':200.,'max_allocated_mib':250.,'max_reserved_mib':300.} for k in ('baseline','train-1','train-2','train-3','catalog-export','user-export','top50')]
  self.assertEqual(mem.measured_mib(samples),500.);self.assertEqual(mem.declared_vram_mib(1100.),1280)
  for bad in ([],[{**samples[0],'own_process_mib':float('nan')}], [{**x,'pid':13 if i else 12} for i,x in enumerate(samples)]):
   with self.assertRaises(ValueError):mem.measured_mib(bad)
 def test_gate_rejects_missing_nan_and_wrong_vram_formula(self):
  m=ops();proof={'status':'cuda_engineering_verified','commit':'a'*40,'synthetic_verify':{'status':'verified'},'vram':{'device':'cuda','catalog':120415,'batch':512,'queries':2520,'history':50,'steps':3,'variants':[{'finite_loss':True,'declared_vram_mib':2048}]}}
  with self.assertRaises(ValueError):m.check_engineering(proof,'a'*40)

class MemoryTamperTests(unittest.TestCase):
 def test_own_pid_sampler_ignores_other_jobs(self):
  from vkgr import seq_combo_stack_memory as mem
  stdout=f'999999, 100000\n{os.getpid()}, 400\n'
  with mock.patch.object(mem.subprocess,'run',return_value=mock.Mock(stdout=stdout)):self.assertEqual(mem.own_gpu_mib(),400.)
  with mock.patch.object(mem.subprocess,'run',return_value=mock.Mock(stdout=f'{os.getpid()}, N/A\n')):
   with self.assertRaises(ValueError):mem.own_gpu_mib()
 def test_all_measurement_tampering_stops_admission(self):
  m=ops();samples=[{'stage':stage,'pid':12,'own_process_mib':400.,'current_reserved_mib':200.,'max_allocated_mib':250.,'max_reserved_mib':300.} for stage in ('baseline','train-1','train-2','train-3','catalog-export','user-export','top50')]
  row={'finite_loss':True,'catalog_items_exported':120415,'users_exported':2520,'history_width':50,'measured_mib':500.,'declared_vram_mib':1024,'vram':{'method':'own_pid_nvidia_smi+torch_peaks','samples':samples}}
  proof={'status':'cuda_engineering_verified','commit':'a'*40,'synthetic_verify':{'status':'verified'},'vram':{'device':'cuda','catalog':120415,'batch':512,'history':50,'queries':2520,'steps':3,'variants':[row]}}
  self.assertTrue(m.check_engineering(proof,'a'*40))
  for mutated in ({**row,'measured_mib':float('nan')},{**row,'measured_mib':-1},{**row,'declared_vram_mib':1280},{**row,'vram':{**row['vram'],'method':'global_free_delta'}},{**row,'vram':{**row['vram'],'samples':samples[:-1]}}):
   with self.assertRaises(ValueError):m.check_engineering({**proof,'vram':{**proof['vram'],'variants':[mutated]}},'a'*40)

class FullshapeBoundaryTests(unittest.TestCase):
 def test_catalog_count_excludes_pad_and_export_checks_actual_rows(self):
  spec=importlib.util.spec_from_file_location('stack_probe',ROOT/'scripts/seq_combo_stack_vram_smoke.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
  with mock.patch.object(probe,'BATCH',8),mock.patch.object(probe,'QUERIES',10):row=probe.measure('f4_combo_c_d_e','cpu',128,1)
  self.assertEqual(row.get('catalog_items_exported'),128);self.assertEqual(row.get('users_exported'),10)

class FailedReceiptExitTests(unittest.TestCase):
 def test_failed_runtime_receipt_is_nonzero_without_submission(self):
  import io
  m=ops()
  with tempfile.TemporaryDirectory() as tmp:
   request={'action':'preflight','paths':{k:str(Path(tmp)/k) for k in m.PATH_KEYS}}
   record={'status':'preflight_failed','leakage_receipt':{'status':'FAIL','checks':{'L2':{'status':'FAIL'}}}}
   with mock.patch.object(m.sys,'stdin',io.StringIO(json.dumps(request))),mock.patch.object(m,'preflight',return_value=record),mock.patch.object(m.sys,'stdout',io.StringIO()):
    with self.assertRaises(SystemExit) as result:m.main([])
   self.assertEqual(result.exception.code,2)
