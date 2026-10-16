"""Versioned FEAT04-COMBO deployment, admission and finite server verifier.
Default stdin-source execution is read-only inventory. Writes require explicit action.
"""
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
R=Path('/workspace/vkgr-infra01-root');Q=R/'queue'
DEP=Path('/workspace/vkgr-feat04combo-p-v1');OPS=Path('/workspace/vkgr-feat04combo-ops-v1');SRC=Path('/workspace/vkgr-b1-src')
PY='/workspace/vkgr-p6-20261006t2050/code-v1/.venv/bin/python'
INPUT=Path('/workspace/vkgr-b1-inputs/p-feat04b-inputs-cd-v1.json')
P3C=Path('/workspace/vkgr-b1-p/docs/p3c_validation.md');P3C_SHA='c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
SEEDS=(42,2024,2025);RIDS=tuple(f'p-feat04combo-cde-s{s}' for s in SEEDS);SMOKE='p-feat04combo-cuda-v1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def new(p,v):
    with Path(p).open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def where(q,rid):
    out={k:sorted(p.name for p in (q/k).glob('*-'+rid+'.json')) for k in ('pending','claimed','rejected')}
    out['reasons']={}
    for n in out['rejected']:
        for p in ((q/'rejected'/n).with_suffix('.reason'),q/'rejected'/(n+'.reason')):
            if p.exists():out['reasons'][p.name]=p.read_text()[:2000]
    return out
def exists(q,runs,rid):
    v=where(q,rid);return (runs/rid).exists() or any(v[k] for k in ('pending','claimed','rejected'))
def publish(q,runs,job):
    with (q/'.feat04combo-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if exists(q,runs,job['run_id']):raise FileExistsError(job['run_id'])
        name=f"{time.time_ns():020d}-{job['run_id']}.json";tmp=q/'pending'/('.'+name+'.tmp')
        new(tmp,job);os.link(tmp,q/'pending'/name);return name

def accepts_verify(status,bound,records):return status=='succeeded' and bound and any(isinstance(v,dict) and v.get('status')=='verified' for v in records)
def one(rid):
    fs=list((Q/'claimed').glob('*-'+rid+'.json'))
    if len(fs)!=1:raise ValueError('unique claimed required: '+rid)
    return load(fs[0])
def flag(a,k):
    if a.count(k)!=1:raise ValueError('missing or duplicate '+k)
    return a[a.index(k)+1]
def replace(a,k,v):a[a.index(k)+1]=v

def modules():
    os.chdir(DEP);sys.path.insert(0,str(DEP/'src'));sys.path.insert(0,str(DEP/'scripts'))
    import torch
    torch.set_num_threads(1)
    import feat04combo_freeze as freeze
    freeze.verify(DEP,load(DEP/'configs/feat04combo/freeze-v1.json'))
    import vkgr.retrieval_feat04combo_run as run
    return run

def identity():
    m=load(OPS/'deployment.json')
    if subprocess.check_output(['git','-C',str(DEP),'rev-parse','HEAD'],text=True).strip()!=m['commit']:raise ValueError('deployed commit changed')
    for p,pin in m['files'].items():
        if sha(DEP/p)!=pin:raise ValueError('deployed file changed: '+p)
    if sha(DEP/'docs/p3c_validation.md')!=P3C_SHA:raise ValueError('data-channel contract hash changed')
    if sha(INPUT)!=m['inputs_sha256']:raise ValueError('input spec changed')
    return m

def inventory():
    return {'read_at_utc':now(),'new_submit':False,'test_read':False,'queue':{r:where(Q,r) for rid in (*RIDS,SMOKE) for r in (rid,rid+'-verify')},'existing_runs':[r for rid in (*RIDS,SMOKE) for r in (rid,rid+'-verify') if (R/'runs'/r).exists()],
            'inputs_path':str(INPUT),'inputs_sha256':sha(INPUT),'stats_manifest_sha256':load(INPUT)['stats_sha256']}

def deploy(req):
    if DEP.exists() or OPS.exists():raise FileExistsError('new deployment/ops root already exists')
    if sha(INPUT)!=req['inputs_sha256']:raise ValueError('input spec differs from read-only inventory')
    if any(exists(Q,R/'runs',r) for rid in (*RIDS,SMOKE) for r in (rid,rid+'-verify')):raise FileExistsError('batch already exists')
    subprocess.run(['git','-C',str(SRC),'fetch','-q','origin','codex/feat04-combo-pub'],check=True)
    if subprocess.check_output(['git','-C',str(SRC),'rev-parse','origin/codex/feat04-combo-pub'],text=True).strip()!=req['commit']:raise ValueError('remote target mismatch')
    subprocess.run(['git','-C',str(SRC),'worktree','add','-q','--detach',str(DEP),req['commit']],check=True)
    for p,pin in req['files'].items():
        if sha(DEP/p)!=pin:raise ValueError('per-file deployment mismatch: '+p)
    if sha(P3C)!=P3C_SHA:raise ValueError('p3c data-channel source mismatch')
    (DEP/'docs/p3c_validation.md').write_bytes(P3C.read_bytes())
    OPS.mkdir();meta={'commit':req['commit'],'tree':subprocess.check_output(['git','-C',str(DEP),'rev-parse','HEAD^{tree}'],text=True).strip(),'files':req['files'],'inputs_sha256':sha(INPUT),'created_at_utc':now()};new(OPS/'deployment.json',meta)
    return {'status':'deployed','files_checked':len(req['files']),'commit':meta['commit'],'inputs_sha256':meta['inputs_sha256'],'ops_root':str(OPS)}

def preflight():
    meta=identity();run=modules();spec=run.validate_inputs4b(load(INPUT),run.m4.E_FEATURE)
    data,_,binding=run.b1.real_inputs(spec);data,extra=run.apply_feat04b(data,spec,50,run.m4.E_FEATURE);inputs={**binding,**extra,'inputs_sha256':meta['inputs_sha256']};entries={}
    for seed,rid in zip(SEEDS,RIDS):
        cp=DEP/f'configs/feat04combo/{rid}.json';c=load(cp);run.validate_config(c);pf=run.preflight_check(data,c,spec,inputs)
        rows,queries=run.fr._view(data,50);model=run._model4b(data,c,'cpu');expected=run.binding_for4b(data,c,rows,queries,model,'cuda',inputs)
        entries[rid]={**pf,'config_sha256':sha(cp),'expected_binding_sha256':run.b1._digest(expected),'data_hashes':{k:expected[k] for k in ('rows_sha256','queries_sha256','catalog_sha256','support_sha256','content_sha256')}}
    return {'status':'preflight_ok','read_at_utc':now(),'commit':meta['commit'],'inputs_sha256':meta['inputs_sha256'],'entries':entries,'test_read':False,'new_submit':False}

def template():
    d=one('p-feat04b-c-i4-popfresh-s42')
    if d['cpu_cores']!=1 or d['ram_mib']!=9216 or d['no_admission'] is not False:raise ValueError('frozen P template changed')
    return d

def train_jobs(req):
    meta=identity();pf=req['preflight'];memory=req['engineering']
    if pf['status']!='preflight_ok' or pf['commit']!=meta['commit'] or pf['inputs_sha256']!=meta['inputs_sha256']:raise ValueError('preflight identity mismatch')
    term=load(R/'runs'/SMOKE/'terminal.json');proof=load(R/'runs'/SMOKE/'out/engineering.json')
    if term['status']!='succeeded' or memory!=proof or proof['status']!='cuda_engineering_verified' or proof['synthetic_verify']['status']!='verified':raise ValueError('CUDA engineering evidence mismatch')
    vram=proof['vram']['variants'][0]['declared_vram_mib'];parent=template();jobs=[]
    for seed,rid in zip(SEEDS,RIDS):
        cp=DEP/f'configs/feat04combo/{rid}.json';entry=pf['entries'][rid]
        if sha(cp)!=entry['config_sha256']:raise ValueError('preflight config mismatch')
        d=copy.deepcopy(parent);d.update(run_id=rid,seed=seed,cwd=str(DEP),config=str(cp),pinned_manifest=None,vram_mib=vram)
        a=d['argv'];a[a.index('-m')+1]='vkgr.retrieval_feat04combo_run';replace(a,'--inputs',str(INPUT));replace(a,'--inputs-sha',meta['inputs_sha256']);replace(a,'--config',str(cp.relative_to(DEP)));replace(a,'--config-sha',sha(cp));replace(a,'--output',str(R/'runs'/rid/'out'))
        d['inputs']=list(dict.fromkeys([*d['inputs'],str(INPUT)]));jobs.append({'job':d,'expected':entry})
    return jobs

def derive_verify(d,manifest_sha):
    v=copy.deepcopy(d);rid=d['run_id'];v.update(run_id=rid+'-verify',disk_gib=1,wall_hours=3.,inputs=[*d['inputs'],str(R/'runs'/rid/'out/manifest.json')]);a=v['argv'];a[a.index('-m')+2]='verify';i=a.index('--output');del a[i:i+2];a.extend(['--manifest',str(R/'runs'/rid/'out/manifest.json'),'--manifest-sha',manifest_sha]);return v

def snapshot(entry):
    job=entry['job'];rid=job['run_id'];t=R/'runs'/rid/'terminal.json';v=R/'runs'/(rid+'-verify')/'terminal.json';result={'run_id':rid,'train_queue':where(Q,rid),'verify_queue':where(Q,rid+'-verify'),'train_status':load(t)['status'] if t.exists() else None,'verify_status':load(v)['status'] if v.exists() else None,'verified':False,'unaccepted':True,'exceptions':[]}
    result['rejected']=bool(result['train_queue']['rejected'] or result['verify_queue']['rejected'])
    if result['train_status']=='succeeded':
        mp=R/'runs'/rid/'out/manifest.json';m=load(mp);pin=sha(mp)
        if m['config']!=load(job['config']) or m['binding']['source']['git_commit']!=load(OPS/'deployment.json')['commit']:raise ValueError('train identity changed')
        from vkgr.retrieval_b1_run import _digest
        if _digest(m['binding'])!=entry['expected']['expected_binding_sha256']:raise ValueError('preflight binding mismatch')
        result.update(manifest_path=str(mp),manifest_sha256=pin,commit=m['binding']['source']['git_commit'],config_sha256=sha(job['config']),data_hashes=entry['expected']['data_hashes'])
        if result['verify_status'] is not None:
            log=R/'runs'/(rid+'-verify')/'logs/stdout.log';payload=log.read_bytes();records=[]
            for line in payload.decode().splitlines():
                try:records.append(json.loads(line))
                except ValueError:pass
            j=one(rid+'-verify');bound=flag(j['argv'],'--manifest')==str(mp) and flag(j['argv'],'--manifest-sha')==pin
            result['verified']=accepts_verify(result['verify_status'],bound,records);result['verify_receipt']={'path':str(log),'sha256':hashlib.sha256(payload).hexdigest(),'job_bound':bound,'records':records}
        if result['verified']:
            sel=m['selection'];dg=next(x['diagnostics'] for x in m['evaluations'] if x['epoch']==sel['best_epoch']);result.update(best_epoch=sel['best_epoch'],stop_epoch=sel['epoch'],best_metric=sel['best_metric'],behavior_beta=dg['behavior_beta'],unaccepted=False)
    return result

def summary_status(states,done):
    if all(s['verified'] for s in states):return 'verified'
    if done or any(s.get('exceptions') or s.get('rejected') or s.get('train_status') not in (None,'succeeded') or (s.get('verify_status') is not None and not s['verified']) for s in states):return 'ALERT'
    return 'running'

def can_start_observation(q,terminal_present):return terminal_present or any(q[k] for k in ('pending','claimed','rejected'))
def deadline_fallback(entries):return [{'run_id':e['job']['run_id'],'verified':False,'unaccepted':True,'exceptions':['deadline expired before first poll; terminal state not read']} for e in entries]

def driver():
    os.chdir(DEP);sys.path.insert(0,str(DEP/'src'));plan=load(OPS/'plan.json');lock=(OPS/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);new(OPS/'driver-pid.json',{'pid':os.getpid(),'ppid':os.getppid(),'poll_seconds':900,'deadline_epoch':plan['deadline_epoch'],'started_at_utc':now()})
    states=deadline_fallback(plan['entries'])
    while time.time()<plan['deadline_epoch']:
        states=[];done=True
        for e in plan['entries']:
            try:
                s=snapshot(e);rid=s['run_id']
                if s['train_status']=='succeeded' and not s['rejected'] and not exists(Q,R/'runs',rid+'-verify'):
                    publish(Q,R/'runs',derive_verify(e['job'],s['manifest_sha256']));s=snapshot(e)
                settled=s['rejected'] or (s['train_status'] is not None and s['train_status']!='succeeded') or (s['train_status']=='succeeded' and s['verify_status'] is not None)
                done=done and settled
            except Exception as exc:s={'run_id':e['job']['run_id'],'verified':False,'unaccepted':True,'exceptions':[type(exc).__name__+': '+str(exc)]};done=False
            states.append(s)
        value={'utc':now(),'runs':states,'all_terminal':done,'status':summary_status(states,done)}
        new(OPS/f'summary-{time.time_ns()}.json',value)
        if done:new(OPS/'summary-final.json',value);return
        time.sleep(min(900,max(0,plan['deadline_epoch']-time.time())))
    new(OPS/'summary-deadline.json',{'utc':now(),'status':'ALERT','reason':'48h deadline','runs':states})

def read_status():
    answer=inventory();answer['read_at_utc']=now()
    for p in ('driver-pid.json','summary-final.json','summary-deadline.json'):
        if (OPS/p).exists():answer[p]=load(OPS/p)
    if (OPS/'driver-pid.json').exists():
        pid=load(OPS/'driver-pid.json')['pid']
        try:os.kill(pid,0);answer['driver_alive']=True
        except ProcessLookupError:answer['driver_alive']=False
        if answer['driver_alive']:answer['driver_current_ppid']=int(Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[1])
    if (OPS/'plan.json').exists():
        modules();answer['states']=[snapshot(e) for e in load(OPS/'plan.json')['entries']]
    if (R/'runs'/SMOKE/'terminal.json').exists():answer['smoke_terminal']=load(R/'runs'/SMOKE/'terminal.json')
    if (R/'runs'/SMOKE/'out/engineering.json').exists():answer['engineering']=load(R/'runs'/SMOKE/'out/engineering.json')
    return answer

def main():
    if '--driver' in sys.argv:driver();return
    raw=sys.stdin.read();req=json.loads(raw) if raw.strip() else {'action':'inventory'};act=req['action']
    sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops')
    if act=='inventory':answer=inventory()
    elif act=='deploy':answer=deploy(req)
    elif act=='preflight':answer=preflight()
    elif act in ('smoke-dry','smoke-submit'):
        identity();d=template();d.update(run_id=SMOKE,cwd=str(DEP),config=str(DEP/f'configs/feat04combo/{RIDS[0]}.json'),inputs=[],disk_gib=1,wall_hours=1.,vram_mib=2048,argv=[PY,'-B','scripts/feat04combo_cuda_check.py',str(R/'runs'/SMOKE/'out')])
        if exists(Q,R/'runs',SMOKE):raise FileExistsError(SMOKE)
        from vkgr_ops import launcher
        answer={'status':'smoke_dry_ok','prepared':launcher.prepare(d)}
        if act=='smoke-submit':answer['written_not_accepted']=publish(Q,R/'runs',d)
    elif act in ('dry','submit'):
        entries=train_jobs(req)
        from vkgr_ops import launcher
        for e in entries:
            for rid in (e['job']['run_id'],e['job']['run_id']+'-verify'):
                if exists(Q,R/'runs',rid):raise FileExistsError(rid)
            launcher.prepare(e['job'])
        answer={'status':'dry_ok','jobs':[e['job'] for e in entries]}
        if act=='submit':
            new(OPS/'plan.json',{'created_at_utc':now(),'deadline_epoch':time.time()+48*3600,'entries':entries});answer['written_not_accepted']=[publish(Q,R/'runs',e['job']) for e in entries]
    elif act=='start':
        identity();plan=load(OPS/'plan.json')
        if (OPS/'driver-pid.json').exists() or (OPS/'driver-launcher.log').exists():raise FileExistsError('driver already started')
        for e in plan['entries']:
            q=where(Q,e['job']['run_id'])
            if not can_start_observation(q,(R/'runs'/e['job']['run_id']/'terminal.json').exists()):raise ValueError('registered train has no queue or terminal receipt')
        log=(OPS/'driver-launcher.log').open('xb');proc=subprocess.Popen([PY,'-B',str(DEP/'scripts/feat04combo_ops.py'),'--driver'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=DEP,env={**os.environ,'PYTHONPATH':str(DEP/'src'),'OMP_NUM_THREADS':'1'});answer={'status':'started','pid':proc.pid}
    elif act=='status':answer=read_status()
    else:raise ValueError('unregistered action')
    print(json.dumps(answer,sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
