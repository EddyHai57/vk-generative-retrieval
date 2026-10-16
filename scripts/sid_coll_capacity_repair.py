"""Approval-gated metadata-only repair of two never-started CUDA capacity probes.
Default is read-only prepare. No gate/model/config/data/seed changes or resubmit.
"""
import copy,datetime,fcntl,hashlib,json,math,os,signal,stat,subprocess,sys,time
from pathlib import Path
R=Path('/workspace/vkgr-infra01-root');OPS=Path('/workspace/vkgr-sid-coll-ops-v2');DEP=Path('/workspace/vkgr-sid-coll-g-v2')
PLAN=OPS/'pipeline-plan.json';NEWPLAN=OPS/'pipeline-plan-capacity-v1.json';BACKUP=OPS/'capacity-repair-v1'
OLDPIN='f82bbcca28c2bf7a34e621a23e8007b9dc8625e23b73275ffa58750302e6e4cc'
IDS=('sid-coll-c1-cuda-v1','sid-coll-c2-cuda-v1');PY='/workspace/vkgr-p6-20261006t2050/code-v1/.venv/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(b):return hashlib.sha256(b).hexdigest()
def wire(v):return json.dumps(v,sort_keys=True,allow_nan=False).encode()
def load(p):return json.loads(Path(p).read_bytes())
def new(p,v):
 with Path(p).open('xb') as f:f.write(wire(v));f.flush();os.fsync(f.fileno())
def capacity(total,free,committed,settling):
 if any(not math.isfinite(v) or v<0 for v in (total,free,committed,settling)) or total<=0:raise ValueError('finite known capacity required')
 return max(0,math.floor(min(total*.8-committed,free-total*.2-settling)/256)*256)
def patch_bytes(raw,cap):
 value=json.loads(raw)
 if value.get('run_id') not in IDS or value.get('vram_mib')!=39312 or cap!=39168:raise ValueError('only approved two capacity fields')
 token=b'"vram_mib": 39312'
 if raw.count(token)!=1:raise ValueError('exact original serialization required')
 updated=raw.replace(token,b'"vram_mib": 39168');after=json.loads(updated)
 if {**after,'vram_mib':39312}!=value:raise ValueError('non-capacity change')
 return updated

def patched_plan(plan,cap):
 result=copy.deepcopy(plan)
 for branch in result['branches'].values():
  old=branch['stages']['cuda']['job']
  if old['run_id'] not in IDS or old['vram_mib']!=39312 or cap!=39168:raise ValueError('unexpected frozen CUDA job')
  old['vram_mib']=cap
 return result

def pending(plan):
 found={}
 for branch in plan['branches'].values():
  job=branch['stages']['cuda']['job'];rid=job['run_id'];paths=list((R/'queue/pending').glob('*-'+rid+'.json'))
  if rid not in IDS or len(paths)!=1 or (R/'runs'/rid).exists():raise ValueError('probe is not unique never-started pending')
  if any(list((R/'queue'/k).glob('*-'+rid+'.json')) for k in ('claimed','rejected')):raise ValueError('already claimed or rejected')
  p=paths[0]
  if p.is_symlink() or json.loads(p.read_bytes())!=job:raise ValueError('pending differs from frozen job')
  raw=p.read_bytes();found[rid]={'path':str(p),'old_sha256':sha(raw),'new_sha256':sha(patch_bytes(raw,39168))}
 return found

def libs():
 sys.path.insert(0,str(DEP/'scripts'));sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops')
 import sid_coll_retry_control as control
 from vkgr_ops import admission,common
 return control,admission,common

def prepare():
 control,admission,common=libs();control.identity()
 raw=PLAN.read_bytes()
 if sha(raw)!=OLDPIN:raise ValueError('original recovery plan changed')
 if NEWPLAN.exists() or BACKUP.exists() or (OPS/'controller/driver-pid-v3.json').exists():raise FileExistsError('one-time repair already used')
 plan=json.loads(raw);items=pending(plan);status=control.status()
 for b in status['summary']['branches'].values():
  if b['status']=='ALERT' or b['stages']['preflight'].get('raw_verification',{}).get('status')!='preflight_ok':raise ValueError('repaired PF not accepted')
 measured=admission.measure(R,common.Probes())
 cap=capacity(measured['totals']['vram_mib'],measured['free']['vram_mib'],measured['committed']['vram_mib'],measured['settling']['vram_mib'])
 if cap!=39168:raise ValueError('host changed; proposal must be reviewed again')
 checks={}
 for label,branch in plan['branches'].items():
  job=branch['stages']['cuda']['job'];need={k:job[k] for k in admission.RESOURCES};oldok,oldreason,_=admission.evaluate(measured,need)
  need['vram_mib']=cap;newok,newreason,detail=admission.evaluate(measured,need)
  if oldok or not newok:raise ValueError('proposal does not exclusively resolve measured admission block: '+newreason)
  checks[label]={'old_admit':oldok,'old_reason':oldreason,'new_admit':newok,'new_resources':detail}
 newplan=patched_plan(plan,cap)
 return {'status':'capacity_repair_preflight_ok','read_at_utc':now(),'pending':items,'old_vram_mib':39312,'new_vram_mib':cap,'old_plan_sha256':OLDPIN,'new_plan_path':str(NEWPLAN),'new_plan_sha256':sha(wire(newplan)),'deadline_epoch':plan['deadline_epoch'],'admission_headroom_unchanged':admission.HEADROOM,'checks':checks,'driver':status['driver'],'resubmitted':False,'checkpoint_read':False,'test_read':False}

def replace_bytes(path,raw):
 tmp=path.parent/('.capacity-'+str(time.time_ns())+'.tmp')
 with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.chmod(tmp,stat.S_IMODE(path.stat().st_mode));os.replace(tmp,path)

def apply(req):
 if req.get('approval')!='Eddy two CUDA pending capacity fields only':raise PermissionError('explicit pending-file replacement approval required')
 preview=prepare()
 if preview['pending']!=req['pending'] or preview['new_plan_sha256']!=req['new_plan_sha256']:raise ValueError('approved proposal drift')
 record=preview['driver'];pid=record['pid'];cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
 if not all(x in cmd for x in (str(DEP/'scripts/sid_coll_ops.py'),'driver',str(PLAN),OLDPIN)):raise ValueError('only current own observer may reload plan')
 plan=load(PLAN);originals={}
 with (R/'scheduler.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if pending(plan)!=req['pending']:raise ValueError('pending changed under scheduler lock')
  BACKUP.mkdir();new(BACKUP/'proposal.json',preview)
  for rid,item in preview['pending'].items():
   p=Path(item['path']);raw=p.read_bytes();originals[p]=raw
   with (BACKUP/p.name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.kill(pid,signal.SIGTERM)
  with (OPS/'controller/driver.lock').open('a') as owner:
   end=time.monotonic()+10
   while True:
    try:fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);break
    except BlockingIOError:
     if time.monotonic()>=end:raise RuntimeError('old observer did not release ownership')
     time.sleep(.05)
  try:
   new(NEWPLAN,patched_plan(plan,39168))
   for p,raw in originals.items():replace_bytes(p,patch_bytes(raw,39168))
  except BaseException:
   for p,raw in originals.items():
    if p.exists() and p.read_bytes()!=raw:replace_bytes(p,raw)
   raise
  fd=os.open(R/'queue/pending',os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
 control,_,_=libs();env=os.environ.copy();env['PYTHONPATH']=str(DEP/'src')+':'+str(DEP)+':/workspace/vkgr-infra03-ops/ops'
 command=[str(DEP/'scripts/sid_coll_ops.py'),'driver','--plan',str(NEWPLAN),'--plan-sha',preview['new_plan_sha256'],'--out',str(OPS/'controller/summary-final.json'),'--pid-record','driver-pid-v3.json']
 handoff="import os,sys,time;end=time.monotonic()+10\nwhile os.getppid()!=1 and time.monotonic()<end:time.sleep(.02)\nos.execv(sys.executable,[sys.executable,'-B',*sys.argv[1:]])"
 with (OPS/'controller/capacity-observer-v3.log').open('xb') as log:
  child=subprocess.Popen([PY,'-c',handoff,*command],cwd=DEP,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
 result={**preview,'status':'capacity_metadata_repaired_observer_unconfirmed','old_pid':pid,'new_pid':child.pid,'backup_dir':str(BACKUP),'plan_overwritten':False,'gate_changed':False}
 new(BACKUP/'applied.json',result);return result
if __name__=='__main__':
 req=json.load(sys.stdin) if sys.argv[0]=='-c' else {'action':'prepare'}
 print(json.dumps(apply(req) if req.get('action')=='apply' else prepare(),sort_keys=True,allow_nan=False))
