"""Metadata-only, no-retry queue bridge for the fixed structural fit stages.
The real preflight, fit and verify always run under the scheduler, never here.
"""
import json,sys
from pathlib import Path
STAGES={'sid-structure-fit-preflight-v1':'preflight','sid-structure-fit-v1':'fit','sid-structure-fit-verify-v1':'verify'}

def runtime(req):
    sys.path.insert(0,str(Path(req['deployment'])/'scripts'))
    import sid_structure_ops as ops
    return ops

def validate(req):
    job=req['job'];args=job['argv'];rid=job['run_id']
    if rid not in STAGES or type(args) is not list or len(args)<5 or any(type(arg) is not str for arg in args) or not Path(args[0]).is_absolute() or Path(args[0]).resolve()!=Path(sys.executable).resolve() or args[1:5]!=['-B','-m','vkgr.sid_structure_run',STAGES[rid]] or args.count('-m')!=1 or '--engineering' in args:
        raise ValueError('fixed formal structural fit job only')
    if job['cpu_cores']!=1 or job['vram_mib']!=256 or job['no_admission'] is not False:
        raise ValueError('one-core no-CUDA engineering resource declaration required')
    return job

def check_fresh(req):
    job=validate(req);rid=job['run_id'];root=Path(req['root'])
    if (root/'runs'/rid).exists() or any(root.glob('queue/*/*-'+rid+'.json')) or any(root.glob('queue/*/*-'+rid+'.reason')) or any(rid in p.name for p in root.glob('queue/*/.*.tmp')) or (Path(req['ops'])/('intent-'+rid+'.json')).exists():
        raise FileExistsError('never resubmit '+rid)

def submit(req,*,prepare=None,publisher=None):
    check_fresh(req)
    ops=runtime(req) if prepare is None or publisher is None else None
    (prepare or ops.prepare_envelope)(req,req['job'])
    # Intent precedes queue publication; uncertain outcomes are never retried.
    path=Path(req['ops'])/('intent-'+req['job']['run_id']+'.json')
    with path.open('x') as stream:json.dump({'run_id':req['job']['run_id']},stream)
    (publisher or ops.publish)(Path(req['root'])/'queue',Path(req['root'])/'runs',req['job'])
    return {'status':'written_unconfirmed','run_id':req['job']['run_id']}

def main():
    req=json.loads(sys.stdin.read(2*1024*1024+1));ops=runtime(req)
    if 'scripts/sid_structure_fit_queue.py' not in req['files']:raise ValueError('queue source must be deployment-pinned')
    ops.identity(req);check_fresh(req)
    pin=ops.digest(req['job']);dry=Path(req['ops'])/('dry-'+req['job']['run_id']+'.json')
    if req['action']=='dry':
        proof={'status':'dry_ok','job_sha256':pin,'run_id':req['job']['run_id'],'envelope':ops.prepare_envelope(req,req['job'])}
        ops.new(dry,proof);result=proof
    elif req['action']=='submit':
        if ops.load(dry).get('job_sha256')!=pin:raise ValueError('original dry job binding differs')
        result=submit(req)
    else:raise ValueError('dry/submit metadata only')
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
