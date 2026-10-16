"""One-time observer-only handoff. Preserve every scientific job and original cap.
No data deletion, run retry, plan replacement, pid-record overwrite or timer.
"""
import fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
OPS=Path('/workspace/vkgr-sid-coll-ops-v1');PLAN=OPS/'pipeline-plan.json'
SRC=Path('/workspace/vkgr-b1-src');NEW=Path('/workspace/vkgr-sid-coll-observer-v2')
PY='/workspace/vkgr-p6-20261006t2050/code-v1/.venv/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def new(p,v):
 with Path(p).open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def validate(plan,record,cmdline,pin):
 if plan['deadline_epoch']-plan['created_epoch']!=172800 or record['deadline_epoch']!=plan['deadline_epoch']:raise ValueError('original 48h deadline must remain identical')
 required=['/workspace/vkgr-sid-coll-g-v1/scripts/sid_coll_ops.py','driver','--plan',str(PLAN),'--plan-sha',pin]
 if any(value not in cmdline for value in required):raise ValueError('only original own observer can be handed off')
 return plan['deadline_epoch']
def snapshot():
 ctl=OPS/'controller';r=load(ctl/'driver-pid.json');cmd=Path('/proc')/str(r['pid'])/'cmdline'
 return {'old_record':r,'old_alive':cmd.exists(),'new_record_exists':(ctl/'driver-pid-v2.json').exists(),'new_source_exists':NEW.exists(),'plan_sha256':sha(PLAN)}
def handoff(req):
 state=snapshot();ctl=OPS/'controller';pin=req['plan_sha256'];plan=load(PLAN);r=state['old_record']
 if sha(PLAN)!=pin or state['new_record_exists'] or NEW.exists() or (OPS/'observer-handoff-v2.json').exists():raise FileExistsError('one-time observer identity already used or plan changed')
 cmd=(Path('/proc')/str(r['pid'])/'cmdline').read_bytes().decode().split('\0')
 deadline=validate(plan,r,cmd,pin)
 if time.time()>=deadline:raise ValueError('original deadline expired')
 commit=req['commit']
 subprocess.run(['git','-C',str(SRC),'fetch','-q','origin','codex/sid-coll-s1'],check=True)
 actual=subprocess.check_output(['git','-C',str(SRC),'rev-parse','origin/codex/sid-coll-s1'],text=True).strip()
 if actual!=commit:raise ValueError('pushed observer SHA differs')
 subprocess.run(['git','-C',str(SRC),'worktree','add','-q','--detach',str(NEW),commit],check=True)
 for rel,pin2 in req['files'].items():
  if Path(rel).is_absolute() or '..' in Path(rel).parts or sha(NEW/rel)!=pin2:raise ValueError('observer per-file deployment mismatch')
 new(OPS/'observer-v2-deployment.json',{'commit':commit,'files':req['files'],'plan_sha256':pin,'original_deadline_epoch':deadline})
 # Recheck identity immediately before terminating only the idle observer PID.
 validate(load(PLAN),load(ctl/'driver-pid.json'),(Path('/proc')/str(r['pid'])/'cmdline').read_bytes().decode().split('\0'),pin)
 os.kill(r['pid'],signal.SIGTERM)
 with (ctl/'driver.lock').open('a') as lock:
  end=time.monotonic()+10
  while True:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:
    if time.monotonic()>=end:raise RuntimeError('old observer lock did not release; no replacement started')
    time.sleep(.05)
  if sha(PLAN)!=pin:raise ValueError('frozen plan changed')
 # Existing intents/alerts/receipts remain in the same control_dir.
 env=os.environ.copy();env['PYTHONPATH']=str(NEW/'src')+':'+str(NEW)+':/workspace/vkgr-infra03-ops/ops'
 command=[str(NEW/'scripts/sid_coll_ops.py'),'driver','--plan',str(PLAN),'--plan-sha',pin,'--out',str(ctl/'summary-final.json'),'--pid-record','driver-pid-v2.json']
 launcher="import os,sys,time;end=time.monotonic()+10\nwhile os.getppid()!=1 and time.monotonic()<end:time.sleep(.02)\nos.execv(sys.executable,[sys.executable,'-B',*sys.argv[1:]])"
 with (ctl/'observer-v2.log').open('xb') as log:
  child=subprocess.Popen([PY,'-c',launcher,*command],env=env,cwd=NEW,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 result={'status':'observer_replacement_started_unconfirmed','old_pid':r['pid'],'new_pid':child.pid,'observer_commit':commit,'scientific_commit':'079d6bd35ae74317b29408cd43d3ec35de77c1eb','plan_sha256':pin,'original_deadline_epoch':deadline,'scientific_jobs_restarted':False,'intents_preserved':True}
 new(OPS/'observer-handoff-v2.json',result);return result
if __name__=='__main__':
 req=json.load(sys.stdin) if sys.argv[0]=='-c' else {'action':'status'}
 print(json.dumps(handoff(req) if req['action']=='handoff' else snapshot(),sort_keys=True,allow_nan=False))
