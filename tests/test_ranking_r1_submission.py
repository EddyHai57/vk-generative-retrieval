"""Real detached local-process fault tests adapted from the accepted P-bi protocol."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKER = r'''
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1]);import ranking_r1_ops as m
m.identity=lambda req:SimpleNamespace(_digest=m.digest)
m.launcher_prepare=lambda *args:None
m.READY_TIMEOUT_SECONDS=1.
base=Path(sys.argv[2]);fault=sys.argv[3];real_publish=m.publish;count=0

def publish(queue,runs,job):
 global count
 rid=job['run_id']
 if not rid.endswith('-verify'):
  assert (base/'state/driver-ready.json').exists()
  assert (base/'state/publish-release.json').exists()
  count+=1
  if count==1:
   (base/'at-first-publish').touch();until=time.time()+10
   while not (base/'allow-publish').exists():
    if time.time()>until:raise RuntimeError('fault synchronization timeout')
    time.sleep(.01)
  if count==2 and fault=='second':raise OSError('injected second publication failure')
 real_publish(queue,runs,job)
 pending=m.where(queue,rid)['pending'];assert len(pending)==1
 pending[0].rename(Path(queue)/'claimed'/pending[0].name)
 out=Path(runs)/rid;out.mkdir(parents=True)
 if rid.endswith('-verify'):
  (out/'logs').mkdir();(out/'logs/stdout.log').write_text(json.dumps({'status':'verified'})+'\n')
 else:
  (out/'out').mkdir();(out/'out/manifest.json').write_text(json.dumps({'config':{},'binding':{}}))
 (out/'terminal.json').write_text(json.dumps({'status':'succeeded'}))
m.publish=publish
m.main(sys.argv[4:])
'''
CONTROLLER = r'''
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import ranking_r1_ops as m
base=Path(sys.argv[2]);fault=sys.argv[3]
m.identity=lambda req:None
m.dry_run=lambda req:json.loads((base/'fixture-plan.json').read_text())
real_popen=subprocess.Popen

def spawn(argv,**kwargs):
 return real_popen([sys.executable,'-B',str(base/'worker.py'),sys.argv[1],str(base),fault,*argv[3:]],**kwargs)
m.subprocess.Popen=spawn
real_new=m.new

def new(path,value):
 if Path(path).name=='publish-release.json' and fault=='before-release':
  (base/'before-release').touch();time.sleep(10)
 return real_new(path,value)
m.new=new
m.main(['submit','--request',str(base/'request.json')])
(base/'rpc-complete').touch();time.sleep(10)
'''


def wait_file(path, timeout=8):
    until = time.monotonic()+timeout
    while time.monotonic()<until:
        if path.exists():
            return
        time.sleep(.02)
    raise AssertionError('missing local lifecycle receipt: '+path.name)


class Submission(unittest.TestCase):
    def scenario(self, fault):
        with tempfile.TemporaryDirectory(prefix='ranking-r1-lifecycle-', dir='/private/tmp') as temp:
            base = Path(temp)
            (base/'state').mkdir();(base/'runs').mkdir()
            for name in ('pending', 'claimed', 'rejected'):
                (base/'queue'/name).mkdir(parents=True)
            entries=[]
            for seed in (42,2024,2025):
                rid=f'g-ranking-r1-t08-s{seed}'
                config=base/(rid+'.json');config.write_text('{}')
                job={'run_id':rid,'config':str(config),'inputs':[],'argv':[sys.executable,'-B','-m','vkgr.ranking_r1_run','train','--config',str(config),'--config-sha','a'*64,'--output',str(base/'runs'/rid/'out'),'--scratch','scratch'],
                     'env':['VKGR_DECODE=batched','VKGR_DECODE_BATCH=1'],'wall_hours':24.,'disk_gib':4}
                entries.append({'job':job,'expected_binding_sha256':__import__('hashlib').sha256(b'{}').hexdigest()})
            plan={'source_commit':'a'*40,'entries':entries}
            req={'ops':str(base/'state'),'source_commit':'a'*40,'queue':str(base/'queue'),'runs':str(base/'runs'),
                 'python':sys.executable,'deadline_epoch':time.time()+20,'deployment':str(ROOT),'scheduler_ops':'synthetic'}
            for name,value in [('fixture-plan.json',plan),('request.json',req)]:
                (base/name).write_text(json.dumps(value))
            (base/'worker.py').write_text(WORKER);(base/'controller.py').write_text(CONTROLLER)
            with (base/'controller.log').open('w') as log:
                controller=subprocess.Popen([sys.executable,'-B',str(base/'controller.py'),str(ROOT/'scripts'),str(base),fault],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                worker_pid=None
                try:
                    wait_file(base/'state/driver-ready.json')
                    ready=json.loads((base/'state/driver-ready.json').read_text());worker_pid=ready['pid']
                    self.assertEqual(ready['ppid'],1)
                    self.assertEqual(os.getsid(worker_pid),worker_pid)
                    if fault=='before-release':
                        wait_file(base/'before-release')
                    else:
                        wait_file(base/'at-first-publish');wait_file(base/'rpc-complete')
                    controller.kill();controller.wait(timeout=3)
                    (base/'allow-publish').touch()
                    wait_file(base/'state/summary-final.json')
                    summary=json.loads((base/'state/summary-final.json').read_text())
                    claimed=list((base/'queue/claimed').glob('*.json'))
                    trains=[p for p in claimed if not p.name.endswith('-verify.json')]
                    verifiers=[p for p in claimed if p.name.endswith('-verify.json')]
                    wanted=0 if fault=='before-release' else 1 if fault=='second' else 3
                    self.assertEqual(len(trains),wanted)
                    self.assertEqual(len(verifiers),wanted)
                    self.assertEqual(summary['status'],'verified' if fault=='none' else 'ALERT')
                    if fault=='second':
                        states=json.loads((base/'state/publication-final.json').read_text())['runs']
                        self.assertEqual([s['status'] for s in states],['published','failed','not_attempted'])
                        time.sleep(.05)
                        self.assertEqual(len(list((base/'queue/claimed').glob('*.json'))),2)
                finally:
                    if controller.poll() is None:
                        controller.kill();controller.wait(timeout=3)
                    if worker_pid:
                        try:os.kill(worker_pid,signal.SIGTERM)
                        except ProcessLookupError:pass

    def test_rpc_kill_after_ready_keeps_three_train_and_unique_verify_owned(self):
        self.scenario('none')

    def test_second_publish_failure_still_verifies_first_without_retry(self):
        self.scenario('second')

    def test_rpc_kill_before_release_publishes_zero(self):
        self.scenario('before-release')
