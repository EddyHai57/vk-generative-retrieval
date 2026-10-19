"""Fixed CPU-only read-only report transport; fresh intent, no resubmission.
No data/model calculations, SSH, retry, daemon or driver. JSON metadata only.
"""
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
sys.dont_write_bytecode=True
PF_ID='sid-learn-report-preflight-v1'
REPORT_ID='sid-learn-report-v1'
DEP='/workspace/vkgr-sid-learn-report-v1'
OPS='/workspace/vkgr-sid-learn-report-ops-v1'
SCHED='/workspace/vkgr-infra01-root'
SCRIPT='analysis/sid_learn_report_read.py'
REQUIRED=('uv.lock',SCRIPT,'scripts/sid_learn_report_queue.py','analysis/sid_diag.py','analysis/sid_diag_prime.py','analysis/sid_diag_metrics.py','analysis/phase2_validation_read.py')
LIMIT=2*1024*1024

def require(ok,message):
    if not ok:raise ValueError(message)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe(p):
    p=Path(p);require(p.is_absolute() and '..' not in p.parts and not any(x.is_symlink() for x in (p,*p.parents)),'absolute nonsymlink metadata path required');return p

def load(p):
    p=safe(p);require(p.is_file() and p.stat().st_size<=LIMIT,'bounded regular JSON metadata required');return json.loads(p.read_bytes())
def new(p,v):
    with Path(p).open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush()

def validate(req):
    fields={'action','root','ops','scheduler_code','source_commit','files','preflight_ref','job'}
    require(type(req) is dict and fields<=set(req) and not set(req)-fields-{'dry_job_sha256'},'exact transport request fields required')
    require(req['action'] in ('inventory','dry','submit','status'),'metadata-only action required')
    require(req['root']==SCHED and req['ops']==OPS and req['scheduler_code']=='/workspace/vkgr-infra03-ops','fixed project scope required')
    j=req['job'];require(type(j) is dict and j.get('run_id') in (PF_ID,REPORT_ID),'two authorized CPU report stages only')
    require(j.get('cwd')==DEP and j.get('config')==OPS+'/report-request-v1.json','fixed report deployment/config required')
    a=j.get('argv');expected=['-B',DEP+'/'+SCRIPT,'--request',j['config'],'--request-sha']
    preflight=j['run_id']==PF_ID
    require(type(a) is list and all(type(x) is str for x in a) and len(a)==(8 if preflight else 7) and a[1:6]==expected and Path(a[0]).is_absolute() and Path(a[0]).resolve()==Path(sys.executable).resolve() and re.fullmatch('[a-f0-9]{64}',a[6]) is not None,'exact interpreter/script/stage/config argv required')
    require(not preflight or a[-1]=='--preflight','preflight argv required')
    require(j.get('schema')=='vkgr-queue-job-v1' and j.get('kind')=='ops' and j.get('no_admission') is False and type(j.get('cpu_cores')) is int and j['cpu_cores']==1 and type(j.get('vram_mib')) is int and j['vram_mib']==0,'CPU nontraining resource/schema scope required')
    for key,cap in (('ram_mib',16384),('wall_hours',2),('disk_gib',2)):
        require(type(j.get(key)) in (int,float) and math.isfinite(j[key]) and 0<j[key]<=cap,'bounded report resource required')
    env=['PYTHONPATH='+DEP+'/src:'+DEP+'/analysis','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','VKGR_PINNED_MANIFEST='+OPS+'/scheduler-pinned-report-v1.json']
    require(j.get('env')==env and j.get('pinned_manifest')==OPS+'/scheduler-pinned-report-v1.json','fixed source/thread environment required')
    require(type(req['source_commit']) is str and re.fullmatch('[a-f0-9]{40}',req['source_commit']) is not None,'fixed source commit required')
    require(type(req['files']) is dict and set(REQUIRED)<=set(req['files']),'all report execution members must be pinned')
    for name,pin in req['files'].items():require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts and (name=='uv.lock' or Path(name).parts[0] in ('analysis','scripts','src','configs')) and type(pin) is str and re.fullmatch('[a-f0-9]{64}',pin) is not None,'relative fixed source file SHA required')
    require(j.get('pinned_files')==[k+'='+v for k,v in sorted(req['files'].items())],'scheduler must pin the exact report source ledger')
    require(req['preflight_ref'] is None if preflight else type(req['preflight_ref']) is dict,'own successful preflight receipt required for report')
    return j

def identity(req):
    dep=safe(req['job']['cwd']);commit=subprocess.check_output(['git','-C',str(dep),'rev-parse','HEAD'],text=True).strip()
    require(commit==req['source_commit'] and not subprocess.check_output(['git','-C',str(dep),'status','--porcelain','--untracked-files=all'],text=True),'clean fixed report source required')
    for name,pin in req['files'].items():
        p=safe(dep/name);require(sha(p)==pin,'deployed source byte differs')
        blob=subprocess.check_output(['git','-C',str(dep),'show',commit+':'+name]);require(hashlib.sha256(blob).hexdigest()==pin,'source differs from original Git bytes')
    pinned=load(req['job']['pinned_manifest'])
    require(pinned=={'schema':'vkgr-pinned-v1','base_commit':req['source_commit'],'target_commit':req['source_commit'],'overrides':{},'files':req['files']},'scheduler pinned source manifest differs')
    config=safe(req['job']['config']);require(sha(config)==req['job']['argv'][6],'report request physical SHA differs')

def inventory(root,ops,rid):
    root=Path(root);ops=Path(ops)
    return {'run_id':rid,'run_exists':(root/'runs'/rid).exists(),'intent_exists':(ops/('intent-'+rid+'.json')).exists(),'queue':{s:sorted(p.name for p in (root/'queue'/s).glob('*-'+rid+'.json')) for s in ('pending','claimed','rejected')}}

def ensure_fresh(root,ops,rid):
    v=inventory(root,ops,rid)
    if v['run_exists'] or v['intent_exists'] or any(v['queue'].values()):raise FileExistsError('original report identity already occupied; no retry')
    return v

def check_preflight(req):
    ref=req['preflight_ref'];root=Path(req['root']);base=root/'runs'/PF_ID;expected=base/'logs/stdout.log'
    require(type(ref) is dict and set(ref)=={'path','sha256'} and ref['path']==str(expected),'own original preflight stdout required')
    proof=load(expected);require(sha(expected)==ref['sha256'],'preflight stdout physical SHA differs')
    request_sha=sha(req['job']['config'])
    require(proof.get('status')=='preflight_ok' and proof.get('run_id')==PF_ID and proof.get('source_commit')==req['source_commit'] and proof.get('request_sha256')==request_sha,'preflight identity is not bound to report')
    require(proof.get('engineering') is False and proof.get('test_read') is False and proof.get('checkpoint_read') is False and proof.get('model_run') is False,'readonly preflight boundary differs')
    meta=load(base/'run.json');term=load(base/'terminal.json');jobs={s:sorted((root/'queue'/s).glob('*-'+PF_ID+'.json')) for s in ('pending','claimed','rejected')}
    require(proof.get('run_manifest_sha256')==sha(base/'run.json') and meta.get('run_id')==PF_ID and meta.get('git')=={'commit':req['source_commit'],'dirty':False,'dirty_files':[]} and meta.get('cwd')==req['job']['cwd'],'preflight original source/run differs')
    require(meta.get('config',{}).get('sha256')==request_sha and meta['config'].get('path')==req['job']['config'] and meta.get('argv')==req['job']['argv']+['--preflight'],'preflight original config/argv differs')
    require(term.get('run_id')==PF_ID and term.get('status')=='succeeded' and term.get('exit_code')==0 and term.get('wall_cap_exceeded') is False and len(jobs['claimed'])==1 and not jobs['pending'] and not jobs['rejected'],'preflight terminal/claim differs')
    job=load(jobs['claimed'][0]);require(job.get('run_id')==PF_ID and job.get('argv')==meta['argv'] and job.get('cwd')==meta['cwd'],'preflight claimed job differs')
    return {'stdout_sha256':ref['sha256'],'run_manifest_sha256':proof['run_manifest_sha256']}

def prepared_envelope(req,prepare):
    job=req['job'];prepared=prepare(job)
    require(len(json.dumps({'spec':job,'prepared':prepared},allow_nan=False).encode())<48*1024-2048,'metadata launch envelope too large')
    return prepared

def publish(req,prepare,publisher):
    root=Path(req['root']);ops=Path(req['ops']);rid=req['job']['run_id']
    ensure_fresh(root,ops,rid);prepared_envelope(req,prepare)
    require(req.get('dry_job_sha256')==digest(req['job']),'original dry job SHA differs')
    with (ops/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ensure_fresh(root,ops,rid)
        new(ops/('intent-'+rid+'.json'),{'run_id':rid,'job_sha256':digest(req['job']),'status':'attempting'})
        try:path=publisher(root,req['job'])
        except Exception as exc:
            new(ops/('publication-ALERT-'+rid+'.json'),{'status':'ALERT','reason':type(exc).__name__+': '+str(exc),'retry':False});raise
    return {'status':'written_unconfirmed','run_id':rid,'job_sha256':digest(req['job']),'queue_path':str(path)}

def status(req):
    root=Path(req['root']);rid=req['job']['run_id'];out=inventory(root,req['ops'],rid);rd=root/'runs'/rid
    for p in (rd/'terminal.json',rd/'run.json',rd/'logs/stdout.log',rd/'logs/stderr.log'):
        if p.exists():
            p=safe(p);require(p.stat().st_size<=LIMIT,'bounded aggregate receipt required');data=p.read_bytes()
            try:payload=json.loads(data)
            except ValueError:payload=data.decode()
            out[p.name]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'payload':payload}
    out['rejection_reasons']={}
    for name in out['queue']['rejected']:
        p=root/'queue/rejected'/name
        for reason in (p.with_suffix('.reason'),Path(str(p)+'.reason')):
            if reason.exists():out['rejection_reasons'][reason.name]=safe(reason).read_text()[:4096]
    return out

def main():
    data=sys.stdin.buffer.read(LIMIT+1);require(len(data)<=LIMIT,'bounded private transport request required');req=json.loads(data);job=validate(req);identity(req)
    action=req['action'];out=None
    if action=='status':out=status(req)
    elif action=='inventory':out=inventory(req['root'],req['ops'],job['run_id'])
    else:
        ensure_fresh(req['root'],req['ops'],job['run_id'])
        if job['run_id']==REPORT_ID:check_preflight(req)
        sys.path.insert(0,str(Path(req['scheduler_code'])/'ops'))
        from vkgr_ops import launcher,scheduler
        if action=='dry':
            prepared_envelope(req,launcher.prepare);out={'status':'dry_ok','run_id':job['run_id'],'job_sha256':digest(job)}
        else:out=publish(req,launcher.prepare,scheduler.submit)
    out['read_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(out,sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
