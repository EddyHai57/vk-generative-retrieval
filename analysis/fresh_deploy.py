"""Fresh pinned Git worktree deployment only; paths supplied privately at runtime."""
import hashlib,json,os,re,subprocess,sys,datetime
from pathlib import Path

def validate(req):
 expected={'root','source','destination','ops','branch','commit','files','run_ids'}
 if set(req)!=expected:raise ValueError('deployment fields')
 for key in ('root','source','destination','ops'):
  p=Path(req[key])
  if not p.is_absolute() or '..' in p.parts or p.is_symlink():raise ValueError('private absolute nonsymlink paths')
 if not re.fullmatch('codex/[a-z0-9-]+',req['branch']) or not re.fullmatch('[a-f0-9]{40}',req['commit']):raise ValueError('work branch and fixed commit required')
 if not req['files'] or not req['run_ids'] or len(set(req['run_ids']))!=len(req['run_ids']):raise ValueError('empty or duplicate scope')
 for name,pin in req['files'].items():
  p=Path(name)
  if p.is_absolute() or '..' in p.parts or not re.fullmatch('[a-f0-9]{64}',pin):raise ValueError('unsafe file pin')
 for rid in req['run_ids']:
  if not re.fullmatch('[a-z0-9][a-z0-9-]{0,119}',rid):raise ValueError('invalid run identity')

def inventory(req):
 validate(req);root=Path(req['root']);jobs={}
 for rid in req['run_ids']:
  states={s:sorted(p.name for p in (root/'queue'/s).glob('*-'+rid+'.json')) for s in ('pending','claimed','rejected')}
  states['run_exists']=(root/'runs'/rid).exists();jobs[rid]=states
 return {'clear':not any(any(v[k] for k in ('pending','claimed','rejected','run_exists')) for v in jobs.values()) and not Path(req['destination']).exists() and not Path(req['ops']).exists(),'jobs':jobs,'destination_exists':Path(req['destination']).exists(),'ops_exists':Path(req['ops']).exists(),'request_sha256':hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()}

def deploy(req,pin):
 before=inventory(req)
 if not before['clear'] or before['request_sha256']!=pin:raise ValueError('fresh read-only inventory required')
 src=Path(req['source']);dest=Path(req['destination']);ops=Path(req['ops'])
 subprocess.run(['git','-C',str(src),'fetch','-q','origin',req['branch']],check=True)
 remote=subprocess.check_output(['git','-C',str(src),'rev-parse','origin/'+req['branch']],text=True).strip()
 if remote!=req['commit']:raise ValueError('published commit mismatch')
 subprocess.run(['git','-C',str(src),'worktree','add','-q','--detach',str(dest),req['commit']],check=True)
 for name,pin in req['files'].items():
  p=dest/name
  if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=pin:raise ValueError('deployment file mismatch '+name)
 dirty=subprocess.check_output(['git','-C',str(dest),'status','--porcelain'],text=True).strip()
 if dirty:raise ValueError('deployment is dirty')
 ops.mkdir()
 record={'commit':req['commit'],'files':req['files'],'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data_overlay':False}
 with (ops/'deployment.json').open('x') as f:json.dump(record,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 return {'status':'deployed','commit':req['commit'],'files_checked':len(req['files']),'data_overlay':False}

def main():
 payload=json.loads(sys.stdin.read(2*1024*1024+1));req=payload['request']
 if payload['action']=='inventory':answer=inventory(req)
 elif payload['action']=='deploy':answer=deploy(req,payload['inventory_pin'])
 else:raise ValueError('unknown action')
 answer['read_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(answer,sort_keys=True))
if __name__=='__main__':main()
