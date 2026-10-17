"""Fixed-Git metadata bridge. Real data and CUDA actions must remain queued."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

def validate(r):
    if set(r)!={'deployment','ops','commit','script_sha256','args','python'}:raise ValueError('exact metadata request required')
    for k in ('deployment','ops'):
        p=Path(r[k])
        if not p.is_absolute() or '..' in p.parts:raise ValueError('absolute nontraversing scope')
    if not re.fullmatch('[a-f0-9]{40}',r['commit']) or not re.fullmatch('[a-f0-9]{64}',r['script_sha256']):raise ValueError('fixed source required')
    if type(r['args']) is not list or not r['args'] or r['args'][0] not in ('plan','dry','submit') or any(type(x) is not str for x in r['args']):raise ValueError('bounded metadata action only')

def main():
    r=json.loads(sys.stdin.read(65537));validate(r);root=Path(r['deployment']);script=root/'scripts/profile_ops.py'
    if root.is_symlink() or script.is_symlink():raise ValueError('source symlink')
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=r['commit'] or subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True).strip():raise ValueError('fixed clean source required')
    blob=subprocess.check_output(['git','-C',str(root),'show',r['commit']+':scripts/profile_ops.py'])
    if hashlib.sha256(blob).hexdigest()!=r['script_sha256'] or script.read_bytes()!=blob:raise ValueError('physical source/Git mismatch')
    if Path(r['python']).resolve()!=Path(sys.executable).resolve():raise ValueError('approved existing runtime only')
    env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','LD_','DYLD_'))};env.update(PYTHONPATH=str(root/'src')+':'+str(root/'scripts'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cp=subprocess.run([sys.executable,'-B','-s',str(script),*r['args']],cwd=root,env=env,capture_output=True,text=True,timeout=55)
    out={'status':'metadata_ok' if cp.returncode==0 else 'FAILED','exit_code':cp.returncode,'stderr':cp.stderr,'stdout_sha256':hashlib.sha256(cp.stdout.encode()).hexdigest(),'source_commit':r['commit'],'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:out['result']=json.loads(cp.stdout)
    except ValueError:out['stdout']=cp.stdout
    print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
