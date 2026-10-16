"""Bounded orchestration: six unconditional trains, once-only verify and fail-closed proofs."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
def ops():
 import biconfirm_ops
 return biconfirm_ops
class OpsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_dedicated_ops_exists(self):self.assertTrue((ROOT/'scripts/biconfirm_ops.py').exists(),'new versioned ops entry required')
 def proofs(self):
  entries={}
  for arm in ('mean','bi'):
   for seed in (314159,271828,161803):
    rid=f'p-biconfirm-{arm}-s{seed}'
    entries[rid]={'seed':seed,'arm':arm,'config_rel':f'configs/biconfirm/{rid}.json','config_sha256':'b'*64,'expected_binding_sha256':'c'*64,'data_hashes':{'rows_sha256':'d'*64}}
  source={'git_commit':'a'*40,'git_dirty':False,'files':{'src/vkgr/retrieval_biconfirm_run.py':'e'*64}}
  pf={'status':'preflight_ok','engineering':False,'data_root':'/private-data','source':source,'inputs_sha256':'f'*64,'entries':entries,'test_read':False,'existing_checkpoint_read':False}
  cu={'status':'cuda_engineering_verified','source':source,'test_read':False,'existing_checkpoint_read':False,'real_training_rows_used':False,'preflight_sha256':ops().digest(pf),'arms':{a:{'synthetic_verify':{'status':'verified'},'full_shape':{'batch':512,'history':50,'catalog':120415,'queries':2520},'zero_start':True,'finite_gradients':True,'history_oracle':True,'measured_vram_mib':1001,'declared_vram_mib':1280} for a in ('mean','bi')}}
  request={'root':'/deployment','data_root':'/private-data','inputs':'/inputs/spec.json','runs_root':'/jobroot/runs','python':'/runtime/python','job_template':{'cpu_cores':1,'ram_gib':4,'disk_gib':4,'wall_hours':3.,'env':[],'inputs':[]},'created_epoch':1000.}
  return pf,cu,request
 def test_plan_has_six_unconditional_trains_and_unique_verifiers(self):
  m=ops();pf,cu,req=self.proofs();p=m.make_plan(pf,cu,req)
  self.assertEqual(p['deadline_epoch'],173800.)
  self.assertEqual(len(p['entries']),6)
  self.assertEqual({(e['job']['seed'],e['arm']) for e in p['entries']},{(s,a) for s in (314159,271828,161803) for a in ('mean','bi')})
  for e in p['entries']:
   j=e['job'];self.assertEqual(j['vram_mib'],1280);self.assertEqual(j['cwd'],'/deployment');self.assertIn('vkgr.retrieval_biconfirm_run',j['argv'])
   v=m.derive_verify(j,'1'*64);self.assertEqual(v['run_id'],j['run_id']+'-verify');self.assertNotIn('--output',v['argv']);self.assertIn('--manifest-sha',v['argv']);self.assertIn('verify',v['argv']);self.assertEqual(v['seed'],j['seed'])
 def test_plan_rejects_missing_seed_source_data_or_memory_drift(self):
  m=ops()
  for mutate in (lambda p,c:p['entries'].pop('p-biconfirm-bi-s161803'),lambda p,c:c['source'].update(git_commit='2'*40),lambda p,c:p.update(engineering=True),lambda p,c:c.update(preflight_sha256='0'*64),lambda p,c:c['arms']['bi'].update(declared_vram_mib=1024),lambda p,c:c['arms']['mean']['synthetic_verify'].update(status='failed'),lambda p,c:p['source'].update(git_dirty=True)):
   pf,cu,req=self.proofs();cu=copy.deepcopy(cu);mutate(pf,cu)
   with self.assertRaises(ValueError):m.make_plan(pf,cu,req)
 def test_state_machine_never_retries_and_always_verifies_success(self):
  m=ops()
  self.assertEqual(m.next_action(None,None,False,False),'wait')
  self.assertEqual(m.next_action('succeeded',None,False,False),'verify')
  self.assertEqual(m.next_action('succeeded',None,True,False),'wait')
  for train,verify,exists,rejected in [('failed',None,False,False),('succeeded','failed',True,False),('succeeded','succeeded',True,True)]:self.assertEqual(m.next_action(train,verify,exists,rejected),'ALERT')
  self.assertEqual(m.next_action('succeeded','succeeded',True,False,verified=True),'verified')
  self.assertEqual(m.next_action('succeeded','succeeded',True,False,verified=False),'ALERT')
 def test_queue_exclusive_publication_prevents_duplicate_attempts(self):
  m=ops()
  with tempfile.TemporaryDirectory(prefix='biconfirm-q-') as d:
   q=Path(d)/'queue';runs=Path(d)/'runs';runs.mkdir()
   for state in ('pending','claimed','rejected'):(q/state).mkdir(parents=True)
   j={'run_id':'p-biconfirm-bi-s314159','argv':['python']};m.publish(q,runs,j)
   with self.assertRaises(FileExistsError):m.publish(q,runs,j)
   self.assertEqual(len(list((q/'pending').glob('*.json'))),1)
 def test_cpu_engineering_replay_uses_only_new_artifacts_and_oracle_detects_wrong_masks(self):
  m=ops()
  from vkgr import retrieval_biconfirm_run as run
  with tempfile.TemporaryDirectory(prefix='biconfirm-replay-') as d:
   r=m.small_replay(run,Path(d),'cpu');self.assertEqual(set(r),{'mean','bi'})
   self.assertTrue(all(v['status']=='verified' for v in r.values()))
  for arm in ('mean','bi'):
   data=run.fr.synthetic_data();model=run._model(data,{**run.default_config(),'arm':arm},'cpu');m.history_oracle(model)
   model.mask_kind='causal' if arm=='bi' else 'bi'
   with self.assertRaises(ValueError):m.history_oracle(model)
if __name__=='__main__':unittest.main()

class ReceiptTests(unittest.TestCase):
 def test_bound_claims_allow_scheduler_metadata_but_reject_changed_verify_argv(self):
  m=ops()
  from vkgr import retrieval_biconfirm_run as run
  with tempfile.TemporaryDirectory(prefix='biconfirm-receipt-') as d:
   d=Path(d);q=d/'queue';runs=d/'runs';rid='p-biconfirm-bi-s314159'
   for state in ('pending','claimed','rejected'):(q/state).mkdir(parents=True)
   (runs/rid/'out').mkdir(parents=True);(runs/(rid+'-verify')/'logs').mkdir(parents=True)
   cfg=run.default_config();cp=d/'config.json';cp.write_text(json.dumps(cfg));binding={'task':'P-BI-CONFIRM'}
   job={'run_id':rid,'seed':314159,'config':str(cp),'inputs':[],'argv':['python','-m',m.ENTRY,'train','--output',str(runs/rid/'out')]}
   entry={'job':job,'expected':{'expected_binding_sha256':run.b1._digest(binding)}}
   mp=runs/rid/'out/manifest.json';mp.write_text(json.dumps({'schema':run.MANIFEST_SCHEMA,'kind':run.KIND,'status':'complete','config':cfg,'binding':binding}))
   (runs/rid/'terminal.json').write_text(json.dumps({'status':'succeeded'}));(runs/(rid+'-verify')/'terminal.json').write_text(json.dumps({'status':'succeeded'}))
   (q/'claimed'/('1-'+rid+'.json')).write_text(json.dumps({**job,'scheduler_note':'accepted'}))
   verifier=m.derive_verify(job,m.sha(mp));vp=q/'claimed'/('2-'+rid+'-verify.json');vp.write_text(json.dumps({**verifier,'scheduler_note':'accepted'}))
   (runs/(rid+'-verify')/'logs/stdout.log').write_text('progress\n'+json.dumps({'status':'verified'})+'\n')
   self.assertEqual(m.inspect_entry(entry,q,runs)['action'],'verified')
   verifier['argv'][-1]='0'*64;vp.write_text(json.dumps(verifier))
   with self.assertRaises(ValueError):m.inspect_entry(entry,q,runs)

class DriverFailureTests(unittest.TestCase):
 def test_source_gate_failure_persists_alert_instead_of_silent_driver_death(self):
  import time
  m=ops()
  with tempfile.TemporaryDirectory(prefix='biconfirm-driver-') as d:
   state=Path(d)/'state'
   m.driver({'deadline_epoch':time.time()+60,'source':{},'entries':[]},Path(d)/'queue',Path(d)/'launcher',state)
   value=json.loads((state/'summary-final.json').read_text())
   self.assertEqual(value['status'],'ALERT');self.assertIn('source',value['reason'])
