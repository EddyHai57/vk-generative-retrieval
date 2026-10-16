"""One Eddy-approved schema-only repair of one never-launched pending record.
Default transport mode is read-only observation; no resubmit/retry/delete.
"""
import datetime,fcntl,hashlib,importlib.util,json,os,stat,sys,time
from pathlib import Path
R=Path('/workspace/vkgr-infra01-root');DEP=Path('/workspace/vkgr-sidfix-s0-code-v4');OPS=Path('/workspace/vkgr-sidfix-s0-ops-v4')
NAME='01792142148793391058-sid-fix-s0-first-v1.json';RID='sid-fix-s0-first-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def body(spec):return json.dumps({'schema':'vkgr-queue-job-v1',**spec},sort_keys=True,allow_nan=False).encode()
def guard(root,p,pin,spec):
 root,p=Path(root),Path(p)
 if p!=root/'queue/pending'/NAME or p.is_symlink():raise ValueError('only approved pending path')
 if spec.get('run_id')!=RID or (root/'runs'/RID).exists():raise ValueError('run exists or wrong identity')
 for kind in ('claimed','rejected'):
  if list((root/'queue'/kind).glob('*-'+RID+'.json')):raise ValueError('already claimed/rejected')
 if list((root/'queue/pending').glob('*-'+RID+'.json'))!=[p]:raise ValueError('unique pending required')
 raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=pin or json.loads(raw)!=spec or 'schema' in spec:raise ValueError('pending bytes/spec changed or already repaired')
 return raw

def repair(root,p,pin,spec,backup,approved=False):
 if not approved:raise ValueError('explicit Eddy approval required')
 root,p,backup=Path(root),Path(p),Path(backup)
 with (root/'scheduler.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  raw=guard(root,p,pin,spec)
  with backup.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  tmp=p.parent/('.schema-repair-'+str(time.time_ns())+'.tmp')
  with tmp.open('xb') as f:f.write(body(spec));f.flush();os.fsync(f.fileno())
  os.chmod(tmp,stat.S_IMODE(p.stat().st_mode))
  guard(root,p,pin,spec)
  os.replace(tmp,p)
  fd=os.open(p.parent,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return {'status':'schema_only_repaired','old_sha256':pin,'new_sha256':sha(p),'backup_path':str(backup),'backup_sha256':sha(backup),'run_id':RID,'resubmitted':False,'read_at_utc':now()}

def expected():
 s=importlib.util.spec_from_file_location('original_s0_ops',DEP/'analysis/sid_diag_ops.py');ops=importlib.util.module_from_spec(s);s.loader.exec_module(ops);ops.identity()
 pp=OPS/'first_batch-plan.json';spec=ops.job(ops.load(pp),ops.sha(pp))
 return spec

def main():
 req=json.load(sys.stdin) if sys.argv[0]=='-c' else {};spec=expected();p=R/'queue/pending'/NAME
 if req.get('action')=='repair':
  if req.get('approval')!='Eddy 2026-10-16 pending schema only':raise ValueError('approval identity')
  out=repair(R,p,req['old_sha256'],spec,OPS/('pending-original-'+req['old_sha256']+'.json'),approved=True)
 else:
  pin=sha(p);guard(R,p,pin,spec)
  out={'status':'repair_preflight_ok','read_at_utc':now(),'run_id':RID,'pending_path':str(p),'old_sha256':pin,'proposed_new_sha256':hashlib.sha256(body(spec)).hexdigest(),'only_added_field':{'schema':'vkgr-queue-job-v1'},'resubmitted':False}
 print(json.dumps(out,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
