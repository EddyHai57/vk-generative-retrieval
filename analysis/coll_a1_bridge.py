"""Only fixed-SHA S5 dry-run/submit metadata; real input/CUDA work stays queued."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path

def validate(r):
 keys={'deployment','ops','source_commit','script_sha256','request_path','request_sha256','action'}
 if type(r) is not dict or set(r)!=keys or r['action'] not in ('dry-run','submit'):raise ValueError('exact bounded metadata operation required')
 for k,n in [('source_commit',40),('script_sha256',64),('request_sha256',64)]:
  if type(r[k]) is not str or not re.fullmatch('[a-f0-9]{'+str(n)+'}',r[k]):raise ValueError('fixed SHA required')
 for k in ('deployment','ops','request_path'):
  p=Path(r[k])
  if not p.is_absolute() or '..' in p.parts:raise ValueError('private absolute path without traversal required')
 if Path(r['request_path']).parent!=Path(r['ops']) or Path(r['request_path']).suffix!='.json':raise ValueError('request must be own OPS JSON')

def main():
 r=json.loads(sys.stdin.read(2*1024*1024+1));validate(r);root=Path(r['deployment']);script=root/'scripts/coll_sid_a1_ops.py';req=Path(r['request_path'])
 if script.is_symlink() or req.is_symlink() or not req.is_file() or req.stat().st_size>2*1024*1024:raise ValueError('bounded nonsymlink OPS request')
 git=lambda *a:subprocess.check_output(['git','-C',str(root),*a])
 if git('rev-parse','HEAD').decode().strip()!=r['source_commit'] or git('status','--porcelain=v1','--untracked-files=all').strip():raise ValueError('fixed clean deployment required')
 if hashlib.sha256(script.read_bytes()).hexdigest()!=r['script_sha256'] or hashlib.sha256(git('show',r['source_commit']+':scripts/coll_sid_a1_ops.py')).hexdigest()!=r['script_sha256'] or hashlib.sha256(req.read_bytes()).hexdigest()!=r['request_sha256']:raise ValueError('script/request byte binding mismatch')
 env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','LD_','DYLD_'))}
 env.update(PYTHONPATH=str(root/'src')+':'+str(root/'scripts'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 cp=subprocess.run([sys.executable,'-B','-s',str(script),r['action'],'--request',str(req)],cwd=root,env=env,capture_output=True,text=True,timeout=55)
 out={'status':'metadata_command_ok' if cp.returncode==0 else 'FAILED','exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
 try:out['result']=json.loads(cp.stdout)
 except ValueError:pass
 print(json.dumps(out,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
