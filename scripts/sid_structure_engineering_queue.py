"""Finite engineering job transport: private JSON, new files only, no retries."""
import datetime,fcntl,hashlib,json,os,sys
from pathlib import Path

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def new(path,value):
 with Path(path).open('x') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
def queue(root,rid):
 root=Path(root);v={k:sorted(p.name for p in (root/'queue'/k).glob('*-'+rid+'.json')) for k in ('pending','claimed','rejected')}
 v['reasons']={}
 for name in v['rejected']:
  for p in ((root/'queue/rejected'/name).with_suffix('.reason'),root/'queue/rejected'/(name+'.reason')):
   if p.exists():v['reasons'][p.name]=p.read_text()[:8192]
 return v

def inventory(req):
 rid=req['job']['run_id'];v=queue(req['root'],rid);r=Path(req['root'])/'runs'/rid
 return {'run_id':rid,'queue':v,'run_exists':r.exists(),'intent_exists':(Path(req['ops'])/('intent-'+rid+'.json')).exists(),'job_sha256':digest(req['job'])}

def validate(req):
 job=req['job']
 for value in (req['root'],req['ops'],req['scheduler_code'],job['cwd'],job['config']):
  if type(value) is not str or not Path(value).is_absolute() or '..' in Path(value).parts:
   raise ValueError('absolute nontraversing engineering paths required')
 if job.get('run_id') not in ('g-structure-a1-preflight-v1','g-structure-a1-cuda-v1'):
  raise ValueError('fixed structural A1 engineering run IDs only')
 args=job.get('argv');stage={'g-structure-a1-preflight-v1':'preflight','g-structure-a1-cuda-v1':'cuda-preview'}[job['run_id']]
 script=str(Path(job['cwd'])/'scripts/sid_structure_ops.py')
 if type(args) is not list or len(args)!=6 or any(type(a) is not str for a in args) or not Path(args[0]).is_absolute() or Path(args[0]).resolve()!=Path(sys.executable).resolve() or args[1:5]!=['-B',script,stage,'--request'] or args[5]!=job.get('config') or Path(args[5]).parent!=Path(req['ops']):
  raise ValueError('fixed engineering interpreter/script/stage/request required')
 if job['schema']!='vkgr-queue-job-v1' or job['kind']!='ops' or job['cpu_cores']!=1 or job['no_admission'] is not False or not 0<=job['vram_mib']<=39168:
  raise ValueError('engineering resource/schema scope')
 return job

def main():
 req=json.loads(sys.stdin.read(2*1024*1024+1));action=req['action'];root=Path(req['root']);ops=Path(req['ops']);job=req['job']
 validate(req)
 inv=inventory(req)
 if action=='inventory':out=inv
 elif action=='seal':
  if inv['run_exists'] or inv['intent_exists'] or any(inv['queue'][k] for k in ('pending','claimed','rejected')):raise ValueError('existing job')
  paths={}
  for name,value in req['documents'].items():
   if '/' in name or name in ('deployment.json',):raise ValueError('fresh document basename required')
   p=ops/name;new(p,value);paths[name]=hashlib.sha256(p.read_bytes()).hexdigest()
  out={'status':'sealed','files':paths}
 elif action in ('dry','submit'):
  if inv['run_exists'] or inv['intent_exists'] or any(inv['queue'][k] for k in ('pending','claimed','rejected')):raise ValueError('never resubmit')
  sys.path.insert(0,str(Path(req['scheduler_code'])/'ops'))
  from vkgr_ops import launcher,scheduler
  prepared=launcher.prepare(job)
  size=len(json.dumps({'spec':job,'prepared':prepared}).encode())
  if size>=48*1024-2048:raise ValueError('launch metadata envelope')
  out={'status':'dry_ok','run_id':job['run_id'],'job_sha256':digest(job),'launch_bytes':size}
  if action=='submit':
   if req.get('dry_job_sha256')!=digest(job):raise ValueError('dry binding mismatch')
   with (ops/'publish.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    inv=inventory(req)
    if inv['run_exists'] or inv['intent_exists'] or any(inv['queue'][k] for k in ('pending','claimed','rejected')):raise ValueError('never resubmit')
    new(ops/('intent-'+job['run_id']+'.json'),out)
    out.update(status='written_unconfirmed',queue_path=str(scheduler.submit(root,job)))
 elif action=='status':
  out=inv;rd=root/'runs'/job['run_id']
  for p in [rd/'terminal.json',rd/'logs/stdout.log',rd/'logs/stderr.log']:
   if p.exists():
    data=p.read_bytes()
    if len(data)>2*1024*1024:raise ValueError('engineering receipt too large')
    try:payload=json.loads(data)
    except ValueError:payload=data.decode()
    out[p.name]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'payload':payload}
 else:raise ValueError('unknown engineering action')
 out['read_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
