"""Synthetic finite pipeline tests: no server, GPU, or model imports."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
SPEC=importlib.util.spec_from_file_location('sid_coll_ops',Path(__file__).resolve().parents[1]/'scripts/sid_coll_ops.py')
class OpsTests(unittest.TestCase):
 def setUp(self):
  self.m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(self.m)
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
  for folder in ('pending','claimed','rejected'):(self.root/'queue'/folder).mkdir(parents=True)
  stages={}
  for branch in ('C1p','C2'):
   stages[branch]={'stages':{}}
   for name in self.m.STAGES:
    rid=branch.lower()+'-'+name
    stages[branch]['stages'][name]={'job':{'schema':'vkgr-queue-job-v1','run_id':rid,'argv':['python','--pin','@input.manifest_sha256'] if name not in ('input_preflight','input') else ['python'],'config':None},'proof':{'path':str(self.root/'runs'/rid/'out/proof.json'),'expected':{'status':'verified','run_id':rid,'commit':'a'*40,'config_sha256':'b'*64,'data_sha256':'c'*64}}}
  self.plan={'schema':'vkgr-sid-coll-plan-v1','root':str(self.root),'control_dir':str(self.root/'controller'),'created_epoch':1000,'deadline_epoch':1000+48*3600,'common_refs':[],'branches':stages}
  self.submitted=[]
 def tearDown(self):self.tmp.cleanup()
 def submit(self,root,job):
  self.submitted.append(job['run_id']);p=root/'queue/pending'/('1-'+job['run_id']+'.json');p.write_text(json.dumps(job));return p
 def complete(self,branch,name,hits=260,status='verified'):
  stage=self.plan['branches'][branch]['stages'][name];job=self.m.materialize(stage['job'],{s:{'manifest_sha256':'d'*64} for s in self.m.STAGES})
  for p in (self.root/'queue/pending').glob('*-'+job['run_id']+'.json'):p.unlink()
  (self.root/'queue/claimed'/('1-'+job['run_id']+'.json')).write_text(json.dumps(job))
  rd=self.root/'runs'/job['run_id'];(rd/'out').mkdir(parents=True,exist_ok=True)
  (rd/'terminal.json').write_text(json.dumps({'run_id':job['run_id'],'status':'succeeded','exit_code':0}))
  proof={**stage['proof']['expected'],'status':status,'manifest_sha256':'d'*64,'prediction_sha256':'e'*64,'hits':hits,'best_epoch':1,'stop_epoch':5,'R50':hits/2520,'N50':.1}
  (rd/'out/proof.json').write_text(json.dumps(proof))
 def tick(self,now=1001):return self.m.tick(self.plan,now=now,submitter=self.submit)
 def test_success_automatically_supplements(self):
  for name in self.m.BASE:
   self.complete('C1p',name)
  result=self.tick();self.assertIn('c1p-train2024',self.submitted);self.assertIn('c1p-train2025',self.submitted)
  self.assertEqual(result['branches']['C1p']['screening'],'SUPPLEMENT')
 def test_low_no_supplement_and_independent_branch(self):
  for name in self.m.BASE:self.complete('C1p',name,hits=259)
  result=self.tick();self.assertEqual(result['branches']['C1p']['status'],'complete_low')
  self.assertEqual(self.submitted,['c2-input_preflight'])
 def test_verified_missing_stops_one_branch(self):
  self.complete('C1p','input_preflight');self.complete('C1p','input',status='not_verified')
  result=self.tick();self.assertEqual(result['branches']['C1p']['status'],'ALERT');self.assertEqual(self.submitted,['c2-input_preflight'])
 def test_rejected_is_not_retried(self):
  rid='c1p-input_preflight';(self.root/'queue/rejected'/('1-'+rid+'.json')).write_text('{}')
  (self.root/'queue/rejected'/('1-'+rid+'.reason')).write_text('no room')
  self.tick();self.tick();self.assertNotIn(rid,self.submitted);self.assertEqual(self.submitted.count('c2-input_preflight'),1)
 def test_deadline_and_common_invalid_stop_both(self):
  result=self.tick(self.plan['deadline_epoch']);self.assertEqual(result['status'],'ALERT');self.assertEqual(self.submitted,[])
  p=self.root/'common.json';p.write_text('{}');self.plan['common_refs']=[{'path':str(p),'sha256':'0'*64}]
  result=self.tick();self.assertEqual(result['status'],'ALERT');self.assertEqual(self.submitted,[])
 def test_duplicate_publish_refused_and_lost_intent_not_retried(self):
  job=self.plan['branches']['C1p']['stages']['input_preflight']['job']
  self.m.publish(self.plan,job,self.submit)
  with self.assertRaises(FileExistsError):self.m.publish(self.plan,job,self.submit)
  for p in (self.root/'queue/pending').glob('*.json'):p.unlink()
  result=self.tick();self.assertEqual(result['branches']['C1p']['status'],'ALERT');self.assertEqual(self.submitted.count('c1p-input_preflight'),1)
 def test_schema_deadline_and_placeholder_guards(self):
  p=copy.deepcopy(self.plan);p['deadline_epoch']+=1
  with self.assertRaises(ValueError):self.m.validate_plan(p)
  with self.assertRaises(ValueError):self.m.materialize('@missing.manifest_sha256',{})
 def test_alert_is_sticky_even_if_failed_receipt_changed(self):
  self.complete('C1p','input_preflight');self.complete('C1p','input',status='not_verified');self.tick()
  self.complete('C1p','input_preflight');self.complete('C1p','input',status='verified');result=self.tick()
  self.assertEqual(result['branches']['C1p']['status'],'ALERT')
  self.assertNotIn('c1p-input_verify',self.submitted)
 def test_missing_proof_exit_zero_is_alert(self):
  self.complete('C1p','input_preflight');self.complete('C1p','input');p=Path(self.plan['branches']['C1p']['stages']['input']['proof']['path']);p.unlink()
  result=self.tick();self.assertEqual(result['branches']['C1p']['status'],'ALERT')

class WaitTests(unittest.TestCase):
 def setUp(self):
  self.m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(self.m)
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();(self.root/'runs').mkdir()
  self.plan={'root':str(self.root),'branches':{'C1p':{'stages':{'x':{'job':{'run_id':'own'}}}}}}
  self.now=0.;self.sleeps=[]
 def tearDown(self):self.tmp.cleanup()
 def clock(self):return self.now
 def sleep(self,seconds):self.sleeps.append(seconds);self.now+=seconds
 def fake(self,batches):
  owner=self
  class Backend:
   def __init__(self):self.watches=[];self.batches=list(batches)
   def watch(self,path):self.watches.append(Path(path))
   def read(self,timeout):
    if self.batches:owner.now+=1;return self.batches.pop(0)
    owner.now+=timeout;return []
   def close(self):pass
  return Backend()
 def test_only_own_terminal_completes_wait(self):
  (self.root/'runs/own').mkdir()
  b=self.fake([[(self.root/'runs/own','unrelated.json',self.m.IN_CLOSE_WRITE)],[(self.root/'runs/own','terminal.json',self.m.IN_CLOSE_WRITE)]])
  w=self.m.EventWaiter(self.plan,backend_factory=lambda:b,clock=self.clock,sleeper=self.sleep)
  self.assertEqual(w.wait(900)['wake'],'own_terminal');self.assertEqual(self.now,2)
 def test_other_events_never_end_wait_early(self):
  b=self.fake([[(self.root/'runs','external',self.m.IN_CREATE)],[(self.root/'runs/external','terminal.json',self.m.IN_CLOSE_WRITE)]])
  w=self.m.EventWaiter(self.plan,backend_factory=lambda:b,clock=self.clock,sleeper=self.sleep)
  self.assertEqual(w.wait(900)['wake'],'fallback_timeout');self.assertEqual(self.now,900)
  self.assertNotIn(self.root/'runs/external',b.watches)
  with self.assertRaises(ValueError):w.wait(899)
 def test_backend_failure_sleeps_remainder_and_reports_fallback(self):
  def broken():raise OSError('no inotify')
  w=self.m.EventWaiter(self.plan,backend_factory=broken,clock=self.clock,sleeper=self.sleep)
  result=w.wait(900)
  self.assertEqual(result['mode'],'sleep_fallback');self.assertEqual(self.sleeps,[900]);self.assertIn('no inotify',result['reason'])
 def test_created_own_directory_watch_then_terminal(self):
  b=self.fake([]);w=self.m.EventWaiter(self.plan,backend_factory=lambda:b,clock=self.clock,sleeper=self.sleep)
  (self.root/'runs/own').mkdir()
  b.batches=[[(self.root/'runs','own',self.m.IN_CREATE)],[(self.root/'runs/own','terminal.json',self.m.IN_MOVED_TO)]]
  self.assertEqual(w.wait(900)['wake'],'own_terminal');self.assertIn(self.root/'runs/own',b.watches)
 def test_runtime_exception_fallback_keeps_remaining_interval(self):
  b=self.fake([])
  def broken(timeout):self.now+=7;raise OSError('watch lost')
  b.read=broken
  w=self.m.EventWaiter(self.plan,backend_factory=lambda:b,clock=self.clock,sleeper=self.sleep)
  result=w.wait(900)
  self.assertEqual(result['mode'],'sleep_fallback');self.assertEqual(self.sleeps,[893]);self.assertEqual(self.now,900)
 def test_deadline_caps_wait_without_changing_poll_contract(self):
  b=self.fake([])
  w=self.m.EventWaiter(self.plan,backend_factory=lambda:b,clock=self.clock,sleeper=self.sleep,wallclock=lambda:1000)
  w.wait(900,deadline_epoch=1010)
  self.assertEqual(self.now,10);self.assertEqual(w.metadata()['fallback_seconds'],900)

class ReceiptHashTests(unittest.TestCase):
 setUp=OpsTests.setUp
 tearDown=OpsTests.tearDown
 complete=OpsTests.complete
 def test_receipt_hash_is_actual_and_not_written_to_receipt(self):
  self.complete('C1p','input');stage=self.plan['branches']['C1p']['stages']['input'];p=Path(stage['proof']['path']);original=p.read_bytes()
  snap,accepted=self.m.inspect_stage(self.plan,stage,{})
  import hashlib
  self.assertEqual(accepted['receipt_sha256'],hashlib.sha256(original).hexdigest())
  self.assertEqual(p.read_bytes(),original);self.assertNotIn('receipt_sha256',json.loads(original))
  self.assertEqual(self.m.materialize('@input.receipt_sha256',{'input':accepted}),snap['proof_sha256'])
