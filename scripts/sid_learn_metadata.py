"""Pinned metadata-only bridge for matched SID structure operations.
Real-input and CUDA commands are queue jobs, never mux computations.
"""
import datetime,hashlib,json,os,re,subprocess,sys
from pathlib import Path

def metadata_action(action):
    if action not in ('dry-run','submit'):raise ValueError('metadata-only execution; queue preflight/CUDA/fit work')
    return action

def safe_path(path):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts or any(q.is_symlink() for q in (p,*p.parents)):
        raise ValueError('absolute nonsymlink metadata path required')
    return p

def metadata_file(name,ops,runs):
    p=safe_path(name);ops=safe_path(ops);runs=safe_path(runs)
    own=p.is_relative_to(ops)
    if not own and p.is_relative_to(runs):
        relative=p.relative_to(runs)
        own=len(relative.parts)>=2 and re.fullmatch(r'(?:sid-learn|g-learn-a1)-[a-z0-9-]+',relative.parts[0]) is not None
    if not own or p.suffix!='.json' or any(w in p.name.lower() for w in ('prediction','mapping','vector','embedding','checkpoint','catalog')) or not p.is_file() or p.stat().st_size>2*1024*1024:
        raise ValueError('only bounded own-run JSON metadata allowed')
    data=p.read_bytes();value=json.loads(data)
    if type(value) is not dict:raise ValueError('metadata object required; no array export')
    row={'sha256':hashlib.sha256(data).hexdigest(),'value':value}
    if type(value.get('pid')) is int:
        pid=value['pid']
        try:
            fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
            row.update(alive=fields[0] not in ('Z','X'),actual_ppid=int(fields[1]),sid=int(fields[3]))
        except (FileNotFoundError,ProcessLookupError):row['alive']=False
    return row

def main():
    req=json.loads(sys.stdin.read(2*1024*1024+1))
    dep=safe_path(req['deployment']);ops=safe_path(req['ops']);runs=safe_path(req['runs'])
    if dep.parent!=Path('/workspace') or ops.parent!=Path('/workspace') or not dep.name.startswith('vkgr-sid-learn-') or not ops.name.startswith('vkgr-sid-learn-') or runs!=Path('/workspace/vkgr-infra01-root/runs'):
        raise ValueError('fixed project deployment/control/run scope')
    script=dep/'scripts/sid_learn_ops.py'
    if not re.fullmatch('[a-f0-9]{40}',req['source_commit']) or not re.fullmatch('[a-f0-9]{64}',req['script_sha256']):raise ValueError('fixed source pins required')
    if subprocess.check_output(['git','-C',str(dep),'rev-parse','HEAD'],text=True).strip()!=req['source_commit'] or hashlib.sha256(script.read_bytes()).hexdigest()!=req['script_sha256']:
        raise ValueError('deployed ops source differs')
    if req['action']=='read':
        if not isinstance(req['read_files'],list) or not 1<=len(req['read_files'])<=32:raise ValueError('bounded explicit metadata file list')
        out={'status':'metadata_read','files':{p:metadata_file(p,ops,runs) for p in req['read_files']}}
    else:
        action=metadata_action(req['action']);request=safe_path(req['request'])
        if not request.is_relative_to(ops) or hashlib.sha256(request.read_bytes()).hexdigest()!=req['request_sha256']:
            raise ValueError('fixed own control request required')
        env=dict(os.environ,PYTHONPATH=str(dep/'src')+':'+str(dep/'scripts'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        cp=subprocess.run([sys.executable,'-B',str(script),action,'--request',str(request),'--request-sha',req['request_sha256']],cwd=dep,env=env,capture_output=True,text=True,timeout=55)
        out={'status':'metadata_command_ok' if cp.returncode==0 else 'ALERT','exit_code':cp.returncode,'stdout':cp.stdout,'stderr':cp.stderr}
    out['read_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(out,sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
