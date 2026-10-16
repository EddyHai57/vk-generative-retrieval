"""Lifecycle fault tests use real detached processes and a local queue/scheduler fixture."""
import importlib.util,json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import biconfirm_ops as ops

WORKER=r'''
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import biconfirm_ops as m
m.check_source=lambda source:None
m.prepare=lambda *a:None
base=Path(sys.argv[2]);fault=sys.argv[3];real_publish=m.publish;count=0

def simulated_scheduler(q,runs,job):
 global count
 rid=job['run_id']
 if not rid.endswith('-verify'):
  assert (base/'state/driver-ready.json').exists(),'publish before driver ready'
  assert (base/'state/publish-release.json').exists(),'publish before controller confirms ready'
  count+=1
  if count==1:
   (base/'at-first-publish').touch()
   deadline=time.time()+15
   while not (base/'allow-publish').exists():
    if time.time()>deadline:raise RuntimeError('test synchronization timed out')
    time.sleep(.01)
  if fault=='third' and count==3:raise OSError('injected third publication failure')
 name=real_publish(q,runs,job);(Path(q)/'pending'/name).rename(Path(q)/'claimed'/name)
 out=Path(runs)/rid;out.mkdir(parents=True)
 (out/'terminal.json').write_text(json.dumps({'status':'succeeded'}))
 if rid.endswith('-verify'):
  (out/'logs').mkdir();(out/'logs/stdout.log').write_text(json.dumps({'status':'verified'})+'\n')
 else:
  from vkgr import retrieval_biconfirm_run as run
  (out/'out').mkdir();cfg=json.loads(Path(job['config']).read_text())
  (out/'out/manifest.json').write_text(json.dumps({'schema':run.MANIFEST_SCHEMA,'kind':run.KIND,'status':'complete','config':cfg,'binding':{}}))
 return name
m.publish=simulated_scheduler
m.main(sys.argv[4:])
'''
CONTROLLER=r'''
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import biconfirm_ops as m
base=Path(sys.argv[2]);fault=sys.argv[3];m.dry=lambda *a:{'status':'dry_ok'}
real_popen=subprocess.Popen

def spawn(argv,**kwargs):
 return real_popen([sys.executable,'-B',str(base/'worker.py'),sys.argv[1],str(base),fault,*argv[3:]],**kwargs)
m.subprocess.Popen=spawn
m.main(['submit','--plan',str(base/'plan.json'),'--plan-sha',m.sha(base/'plan.json'),'--queue',str(base/'queue'),'--launcher-dir',str(base/'launcher'),'--state-dir',str(base/'state')])
(base/'rpc-complete').touch();time.sleep(20)
'''

def wait_file(path,timeout=12):
 until=time.monotonic()+timeout
 while time.monotonic()<until:
  if path.exists():return
  time.sleep(.02)
 raise AssertionError('missing lifecycle receipt: '+path.name)

class SubmissionTests(unittest.TestCase):
 def setup_plan(self,base):
  q=base/'queue';runs=base/'runs';runs.mkdir()
  for s in ('pending','claimed','rejected'):(q/s).mkdir(parents=True)
  entries=[]
  for rid in ops.RIDS:
   cp=base/(rid+'.json');cp.write_text('{}');j={'run_id':rid,'seed':314159,'config':str(cp),'inputs':[],'argv':[sys.executable,'-B','-m',ops.ENTRY,'train','--output',str(runs/rid/'out')]}
   entries.append({'job':j,'expected':{'expected_binding_sha256':ops.digest({})}})
  plan={'schema':'biconfirm-plan-v1','source':{},'entries':entries,'runs_root':str(runs),'deadline_epoch':time.time()+30,'poll_seconds':900};(base/'plan.json').write_text(json.dumps(plan));return plan
 def scenario(self,fault):
  with tempfile.TemporaryDirectory(prefix='biconfirm-lifecycle-') as d:
   base=Path(d);self.setup_plan(base);(base/'worker.py').write_text(WORKER);(base/'controller.py').write_text(CONTROLLER)
   env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1'}
   log=(base/'controller.log').open('w');proc=subprocess.Popen([sys.executable,'-B',str(base/'controller.py'),str(ROOT/'scripts'),str(base),fault],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   worker_pid=None
   try:
    wait_file(base/'state/driver-ready.json');ready=json.loads((base/'state/driver-ready.json').read_text());worker_pid=ready['pid'];self.assertEqual(os.getsid(worker_pid),worker_pid)
    wait_file(base/'at-first-publish');wait_file(base/'rpc-complete')
    # Kill the RPC/controller after its ready handshake; the driver owns publication.
    proc.kill();proc.wait(timeout=5);(base/'allow-publish').touch()
    wait_file(base/'state/summary-final.json');summary=json.loads((base/'state/summary-final.json').read_text())
    claimed=list((base/'queue/claimed').glob('*.json'));trains=[p for p in claimed if not p.name.endswith('-verify.json')];verifiers=[p for p in claimed if p.name.endswith('-verify.json')]
    want=2 if fault=='third' else 6
    self.assertEqual(len(trains),want);self.assertEqual(len(verifiers),want);self.assertEqual(summary['status'],'ALERT' if fault=='third' else 'verified')
    self.assertEqual(sum(s.get('verified',False) for s in summary['runs']),want)
    states=json.loads((base/'state/publication-final.json').read_text())['runs']
    self.assertEqual(sum(s['status']=='published' for s in states),want)
    if fault=='third':
     self.assertTrue((base/'state/publication-ALERT.json').exists());self.assertEqual([s['status'] for s in states],['published','published','failed','not_attempted','not_attempted','not_attempted'])
     # No hidden retry after a settled summary.
     time.sleep(.05);self.assertEqual(len(list((base/'queue/claimed').glob('*.json'))),4)
   finally:
    if proc.poll() is None:proc.kill();proc.wait(timeout=5)
    if worker_pid:
     try:os.kill(worker_pid,signal.SIGTERM)
     except ProcessLookupError:pass
    log.close()
 def test_rpc_disconnect_after_ready_keeps_six_trains_and_unique_verifiers_owned(self):self.scenario('none')
 def test_third_publish_failure_preserves_two_owned_trains_and_their_verifiers(self):self.scenario('third')
 def test_driver_start_failure_publishes_zero_jobs_and_persists_alert(self):
  from unittest import mock
  with tempfile.TemporaryDirectory(prefix='biconfirm-start-fail-') as d:
   base=Path(d);self.setup_plan(base)
   with mock.patch.object(ops,'dry',return_value={'status':'dry_ok'}),mock.patch.object(ops.subprocess,'Popen',side_effect=OSError('injected spawn failure')):
    with self.assertRaises(OSError):ops.main(['submit','--plan',str(base/'plan.json'),'--plan-sha',ops.sha(base/'plan.json'),'--queue',str(base/'queue'),'--launcher-dir',str(base/'launcher'),'--state-dir',str(base/'state')])
   self.assertEqual(list((base/'queue/pending').glob('*.json')),[])
   self.assertEqual(json.loads((base/'state/summary-final.json').read_text())['status'],'ALERT')
if __name__=='__main__':unittest.main()
