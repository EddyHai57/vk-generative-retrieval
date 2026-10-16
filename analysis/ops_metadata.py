"""Pinned metadata-only ops bridge. Real-data/CUDA work must remain queued."""
import datetime,hashlib,json,os,re,subprocess,sys
from pathlib import Path
SCRIPTS={'scripts/biconfirm_ops.py','scripts/seq_combo_stack_ops.py','scripts/ranking_r1_ops.py','scripts/ranking_r2_ops.py'}
def validate_operation(r):
 if not Path(r['deployment']).is_absolute() or not Path(r['ops']).is_absolute() or r['script'] not in SCRIPTS or not re.fullmatch('[a-f0-9]{40}',r['source_commit']) or not re.fullmatch('[a-f0-9]{64}',r['script_sha256']):raise ValueError('pinned metadata operation required')
 if not r['args'] or r['args'][0] not in ('plan','dry','dry-run','submit','status'):raise ValueError('metadata-only action required; queue all data/CUDA/fit work')
def execution_args(r):
 if Path(r['python']).resolve()!=Path(sys.executable).resolve() or r.get('env'):
  raise ValueError('only current approved runtime and fixed import environment')
 return [sys.executable,'-B','-s',str(Path(r['deployment'])/r['script']),*r['args']]
def bounded_path(path,roots):
 p=Path(path)
 if '..' in p.parts or p.is_symlink():raise ValueError('path traversal or symlink')
 actual=p.resolve(strict=True)
 if not any(actual.is_relative_to(Path(root).resolve(strict=True)) for root in roots):raise ValueError('resolved metadata path outside scope')
 return actual

def check(r):
 validate_operation(r);root=Path(r['deployment'])
 if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=r['source_commit'] or hashlib.sha256((root/r['script']).read_bytes()).hexdigest()!=r['script_sha256']:raise ValueError('deployed metadata script differs')
 return root

def main():
 r=json.loads(sys.stdin.read(2*1024*1024+1));root=check(r);ops=Path(r['ops']);action=r['action']
 if action=='seal':
  for rid in r['run_ids']:
   if (Path(r['runs'])/rid).exists() or any(list((Path(r['queue'])/s).glob('*-'+rid+'.json')) for s in ('pending','claimed','rejected')):raise ValueError('existing run identity before seal')
  out={'status':'sealed','files':{}}
  for name,value in r['documents'].items():
   if '/' in name or name=='deployment.json':raise ValueError('fresh private basename required')
   p=ops/name
   with p.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)
   out['files'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
 elif action=='read':
  out={'status':'metadata_read','files':{}}
  for name in r['read_files']:
   p=bounded_path(name,[ops,Path(r['queue']),*[Path(r['runs'])/rid for rid in r['run_ids'] if (Path(r['runs'])/rid).exists()]])
   if p.suffix!='.json' or p.is_symlink() or any(x in p.name.lower() for x in ('prediction','embedding','checkpoint')) or p.stat().st_size>2*1024*1024:raise ValueError('metadata read scope')
   data=p.read_bytes();v=json.loads(data);entry={'sha256':hashlib.sha256(data).hexdigest(),'value':v}
   if isinstance(v,dict) and type(v.get('pid')) is int:
    pid=v['pid']
    try:os.kill(pid,0);entry['alive']=True;entry['actual_ppid']=int(Path('/proc/'+str(pid)+'/stat').read_text().split(') ')[1].split()[1])
    except ProcessLookupError:entry['alive']=False
   out['files'][name]=entry
 elif action=='run':
  env={**{k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','LD_','DYLD_'))},'PYTHONNOUSERSITE':'1','PYTHONPATH':str(root/'src')+':'+str(root/'scripts'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
  cp=subprocess.run(execution_args(r),cwd=root,env=env,capture_output=True,text=True,timeout=55)
  out={'exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
  if cp.returncode:out['status']='FAILED'
  else:
   out['status']='metadata_command_ok'
   try:out['result']=json.loads(cp.stdout)
   except ValueError:pass
 else:raise ValueError('unknown metadata bridge action')
 out['read_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
