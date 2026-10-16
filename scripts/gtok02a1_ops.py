"""Finite B1024 A1 deployment/admission/verify driver; default is read-only."""
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
R=Path('/workspace/vkgr-infra01-root');Q=R/'queue'
DEP=Path('/workspace/vkgr-gtok02-a1-v1');OPS=Path('/workspace/vkgr-gtok02-a1-ops-v1')
SRC=Path('/workspace/vkgr-b1-src');OLD=Path('/workspace/vkgr-b1-g')
OLDPIN=Path('/workspace/vkgr-b1-pinned/g-6f3901dd963d.json')
BASE='0376d9a8119f3500cbf78519675cc7bf18013c1b'
PY='/workspace/vkgr-p6-20261006t2050/code-v1/.venv/bin/python'
SEEDS=(42,2024,2025);RIDS=tuple(f'g-gtok02-a1-b1024d3-s{s}' for s in SEEDS)
SMOKE='g-gtok02-a1-b1024d3-cuda-v1'
ENTRY='vkgr.generative_g6_tok2_run'
SID=R/'runs/g-tok2-b1024d3-fit/out/run_manifest.json'
SID_SHA='6a3b789859023ce1f421107ce675369c4696c8f408f43708d55fa5e9d29f4710'
REAL_SHAPE={'vocab_size':3075,'depth':3,'items':120415,'examples':86785,'queries':2520}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256((json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def new(p,v):
    with Path(p).open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
def flag(a,k):
    if a.count(k)!=1:raise ValueError('missing or duplicate '+k)
    return a[a.index(k)+1]
def replace(a,k,v):flag(a,k);a[a.index(k)+1]=str(v)
def one(rid):
    paths=list((Q/'claimed').glob('*-'+rid+'.json'))
    if len(paths)!=1:raise ValueError('unique claimed template required: '+rid)
    return load(paths[0])
def where(q,rid):
    out={k:sorted(p.name for p in (q/k).glob('*-'+rid+'.json')) for k in ('pending','claimed','rejected')}
    out['reasons']={}
    for n in out['rejected']:
        for p in ((q/'rejected'/n).with_suffix('.reason'),q/'rejected'/(n+'.reason')):
            if p.exists():out['reasons'][p.name]=p.read_text()[:2000]
    return out

def exists(q,runs,rid):
    return (runs/rid).exists() or any(where(q,rid)[k] for k in ('pending','claimed','rejected')) or any(rid in p.name for p in q.glob('*/.*.tmp'))
def publish(q,runs,job):
    with (q/'.gtok02a1-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if exists(q,runs,job['run_id']):raise FileExistsError(job['run_id'])
        name=f"{time.time_ns():020d}-{job['run_id']}.json";tmp=q/'pending'/('.'+name+'.tmp')
        new(tmp,job);os.link(tmp,q/'pending'/name)
        return name

def ids():return [r for rid in (*RIDS,SMOKE) for r in (rid,rid+'-verify')]
def inventory():
    return {'read_at_utc':now(),'test_read':False,'new_submit':False,'queue':{r:where(Q,r) for r in ids()},
        'existing_runs':[r for r in ids() if (R/'runs'/r).exists()], 'sid_sha256':sha(SID),
        'sid_fit_verify':load(R/'runs/g-tok2-b1024d3-fit-verify/out/verify.json')['status'],
        'deployment_exists':DEP.exists(),'ops_exists':OPS.exists()}

def git(*a,text=True):return subprocess.run(['git','-C',str(SRC),*a],capture_output=True,text=text,check=True,timeout=600).stdout

def deploy(req):
    if DEP.exists() or OPS.exists():raise FileExistsError('new deployment / OPS already exists')
    if any(exists(Q,R/'runs',r) for r in ids()):raise FileExistsError('new run identity already exists')
    target=req['commit'];branch=req['branch']
    if len(target)!=40 or any(c not in '0123456789abcdef' for c in target):raise ValueError('full target commit required')
    git('fetch','--quiet','origin',f'{branch}:refs/remotes/origin/{branch}')
    if git('rev-parse',f'origin/{branch}').strip()!=target:raise ValueError('pushed remote target mismatch')
    tokens=git('diff','--no-renames','--name-status','-z',BASE,target).split('\0')
    tokens=[x for x in tokens if x];pairs=list(zip(tokens[::2],tokens[1::2]))
    if len(tokens)%2 or any(s not in ('A','M') for s,_ in pairs):raise ValueError('overlay requires only A/M full-diff paths')
    names=[n for n in git('ls-tree','-r','-z','--name-only',target).split('\0') if n]
    expected={n:hashlib.sha256(git('show',f'{target}:{n}',text=False)).hexdigest() for n in names}
    if req.get('files') is not None and any(expected.get(p)!=s for p,s in req['files'].items()):raise ValueError('caller file pins disagree with target')
    oldpin=load(OLDPIN);channel={}
    for rel,pin in oldpin['files'].items():
        if rel.startswith('provenance/g0-v1/'):
            if sha(OLD/rel)!=pin:raise ValueError('old data-channel mismatch: '+rel)
            channel[rel]=pin
    p3='docs/p3c_validation.md';p3pin=oldpin['overrides'][p3]['sha256']
    if sha(OLD/p3)!=p3pin or not channel:raise ValueError('frozen data-channel source mismatch')
    git('worktree','add','--detach',str(DEP),BASE)
    for _,rel in pairs:
        dest=DEP/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(git('show',f'{target}:{rel}',text=False))
    for rel in (*channel,p3):
        dest=DEP/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((OLD/rel).read_bytes())
    actual={**expected,**channel,p3:p3pin}
    for rel,pin in actual.items():
        if sha(DEP/rel)!=pin:raise ValueError('deployed per-file mismatch: '+rel)
    files={rel:expected[rel] for _,rel in pairs if rel!=p3};files.update(channel)
    pinned={'schema':'vkgr-pinned-v1','base_commit':BASE,'target_commit':target,'files':dict(sorted(files.items())),
        'overrides':{p3:{'sha256':p3pin,'reason':'D0 frozen G1 input'}}}
    OPS.mkdir();new(OPS/'pinned.json',pinned)
    meta={'base_commit':BASE,'model_commit':target,'files':actual,'pinned_manifest':str(OPS/'pinned.json'),
        'pinned_sha256':sha(OPS/'pinned.json'),'created_at_utc':now()}
    new(OPS/'deployment.json',meta)
    identity()
    return {'status':'deployed',**meta}

def identity():
    m=load(OPS/'deployment.json');pin=OPS/'pinned.json'
    head=subprocess.check_output(['git','-C',str(DEP),'rev-parse','HEAD'],text=True).strip()
    if head!=BASE or m['base_commit']!=BASE or sha(pin)!=m['pinned_sha256']:raise ValueError('overlay deployment identity drift')
    p=load(pin)
    if p['base_commit']!=BASE or p['target_commit']!=m['model_commit']:raise ValueError('overlay target changed')
    for rel,s in m['files'].items():
        if sha(DEP/rel)!=s:raise ValueError('deployed file changed: '+rel)
    return m

def derive_job(seed,vram_mib=8192):
    if seed not in SEEDS:raise ValueError('unregistered seed')
    t=one(f'g-gtok01b-t3-b256d3-s{seed}')
    if [t[k] for k in ('cpu_cores','ram_mib','vram_mib','wall_hours','disk_gib','no_admission')]!=[1,18432,6656,24.,4,False]:raise ValueError('frozen A1 template resources changed')
    if t['argv'][t['argv'].index('-m')+1:t['argv'].index('-m')+3]!=['vkgr.generative_g6_run','train']:raise ValueError('template entry changed')
    d=copy.deepcopy(t);rid=f'g-gtok02-a1-b1024d3-s{seed}';cp=DEP/f'configs/gtok02a1/{rid}.json'
    d.update(run_id=rid,seed=seed,cwd=str(DEP),config=str(cp),pinned_manifest=str(OPS/'pinned.json'),vram_mib=vram_mib)
    a=d['argv'];old_sid=flag(a,'--sid-manifest');a[0]=PY;a[a.index('-m')+1]=ENTRY
    for k,v in {'--config':str(cp.relative_to(DEP)),'--config-sha':sha(cp),'--sid-manifest':str(SID),'--sid-sha':SID_SHA,
                '--output':str(R/'runs'/rid/'out'),'--scratch':str(R/'runs'/rid/'scratch')}.items():replace(a,k,v)
    d['inputs']=list(dict.fromkeys([str(SID) if p==old_sid else p for p in d['inputs']]))
    if str(SID) not in d['inputs']:d['inputs'].append(str(SID))
    d['env']=[e for e in d['env'] if not e.startswith('VKGR_PINNED_MANIFEST=')]+['VKGR_PINNED_MANIFEST='+str(OPS/'pinned.json')]
    return d

def configure(job):
    for k in list(os.environ):
        if k.startswith('VKGR_'):del os.environ[k]
    os.environ.update(dict(e.split('=',1) for e in job['env']))
    os.chdir(job['cwd']);sys.path.insert(0,str(DEP/'src'));sys.path.insert(0,str(DEP/'scripts'))
    import torch
    torch.set_num_threads(1)

def config_for(seed):
    job=derive_job(seed);config=load(job['config']);old=load(one(f'g-gtok01b-t3-b256d3-s{seed}')['config'])
    if config!={**old,'sid':'SID_BHKM_B1024D3_S42'} or config['seed']!=seed:raise ValueError('only SID may change from original same-seed A1 config')
    return job,config

def check_context(context,engineering,sid):
    shape={'vocab_size':int(context.codec.vocab_size),'depth':context.codec.depth,'items':len(context.catalog),'examples':len(context.examples),'queries':len(context.queries)}
    if engineering is not False or context.binding.get('synthetic') is not False or shape!=REAL_SHAPE:raise ValueError('formal real shape / synthetic gate mismatch')
    if sid.info['engineering'] is not False or sid.info['manifest_sha256']!=SID_SHA or sid.info['sid_version']!='SID_BHKM_B1024D3_S42':raise ValueError('SID identity mismatch')
    return shape

def prepare_context(seed,scratch):
    identity();job,config=config_for(seed);configure(job)
    from vkgr import generative_g6_tok2_run as g6
    g6.validate_config(config,False)
    if sha(SID)!=SID_SHA or load(R/'runs/g-tok2-b1024d3-fit-verify/out/verify.json')['status']!='verified':raise ValueError('SID fit not directly verified')
    a=job['argv']
    for path,pin in [('--prepared','--prepared-sha'),('--history-manifest','--history-sha'),('--content','--content-sha')]:
        if sha(flag(a,path))!=flag(a,pin):raise ValueError('original input pin mismatch: '+path)
    context,engineering,sid=g6._contexts(DEP,Path(flag(a,'--prepared')),flag(a,'--prepared-sha'),
        Path(flag(a,'--history-manifest')),flag(a,'--history-sha'),SID,SID_SHA,Path(flag(a,'--content')),flag(a,'--content-sha'),config,Path(scratch))
    check_context(context,engineering,sid)
    runtime=g6._runtime(False)
    if runtime['device']!='cuda:0':raise ValueError('real CUDA runtime required')
    return g6,context,engineering,sid,config,runtime

def binding_model(g6,context,config,runtime):
    # G6 initializes Embeddings after .to(device); CPU std reports differ from CUDA.
    # This is constructor-only, no gradients or validation metric computation.
    g6.gr._seed(config['seed'])
    return g6.make_model_arch_d(context.codec,config,device=runtime['device'])

def preflight():
    meta=identity();scratch=OPS/f'preflight-scratch-{time.time_ns()}'
    g6,context,engineering,sid,_,runtime=prepare_context(42,scratch)
    entries={}
    for seed,rid in zip(SEEDS,RIDS):
        job,c=config_for(seed);g6.validate_config(c,False);g6.gr._seed(seed)
        model=binding_model(g6,context,c,runtime)
        a=job['argv'];binding=g6._binding(DEP,context,c,runtime,model,flag(a,'--prepared-sha'),flag(a,'--history-sha'),sid)
        ref=binding['source']['deployment_ref']
        if binding['synthetic'] is not False or binding['test_gate']!='closed' or ref['target_commit']!=meta['model_commit'] or ref['pinned_manifest_sha256']!=meta['pinned_sha256']:raise ValueError('binding real / source gate')
        entries[rid]={'config_sha256':sha(job['config']),'expected_binding_sha256':g6._digest(binding),
            'source':binding['source'],'data_hashes':{k:binding[k] for k in ('prepared_manifest_sha256','history_manifest_sha256','context_binding_sha256')},
            'sid_manifest_sha256':SID_SHA,'real_shape':REAL_SHAPE,'seed':seed,'synthetic':False,'engineering':False,'test_gate':'closed',
            'model_parameters':sum(p.numel() for p in model.parameters())}
        del model
    return {'status':'preflight_ok','read_at_utc':now(),'model_commit':meta['model_commit'],'pinned_sha256':meta['pinned_sha256'],
        'entries':entries,'scratch':str(scratch),'test_read':False,'new_submit':False}

def probe_vram(proof):
    measured=proof.get('measured_mib')
    if not isinstance(measured,(float,int)) or isinstance(measured,bool) or not math.isfinite(measured) or measured<=0:raise ValueError('finite real memory measurement required')
    wanted=math.ceil(measured*1.15/256)*256
    if proof.get('status')!='cuda_engineering_verified' or proof.get('real_shape')!=REAL_SHAPE or proof.get('reload_decode_equal') is not True or type(proof.get('synthetic_grad_steps')) is not int or proof['synthetic_grad_steps']<1 or proof.get('declared_vram_mib')!=wanted:raise ValueError('engineering CUDA proof mismatch')
    return wanted

def train_jobs(req):
    meta=identity();pf=load(OPS/'preflight.json');proof=load(R/'runs'/SMOKE/'out/engineering.json')
    if pf['status']!='preflight_ok' or pf['model_commit']!=meta['model_commit'] or pf['pinned_sha256']!=meta['pinned_sha256']:raise ValueError('preflight identity mismatch')
    if req.get('preflight',pf)!=pf or req.get('engineering',proof)!=proof:raise ValueError('request evidence differs from server receipt')
    if load(R/'runs'/SMOKE/'terminal.json')['status']!='succeeded':raise ValueError('engineering job not succeeded')
    smoke=one(SMOKE)
    if smoke['cwd']!=str(DEP) or smoke['pinned_manifest']!=str(OPS/'pinned.json') or smoke['argv']!=[PY,'-B','scripts/gtok02a1_cuda_check.py',str(R/'runs'/SMOKE/'out')]:raise ValueError('engineering job identity mismatch')
    vram=probe_vram(proof);entries=[]
    for seed,rid in zip(SEEDS,RIDS):
        d=derive_job(seed,vram);entry=pf['entries'][rid]
        if sha(d['config'])!=entry['config_sha256']:raise ValueError('preflight config mismatch')
        entries.append({'job':d,'expected':entry})
    return entries

def derive_verify(d,manifest_sha):
    v=copy.deepcopy(d);rid=d['run_id'];vrid=rid+'-verify';mp=str(R/'runs'/rid/'out/manifest.json')
    v.update(run_id=vrid,config=None,wall_hours=2.,disk_gib=1,inputs=list(dict.fromkeys([*d['inputs'],mp])))
    a=v['argv'];a[a.index('-m')+2]='verify'
    for k in ('--config','--config-sha'):
        i=a.index(k);del a[i:i+2]
    i=a.index('--output');a[i:i]=['--manifest',mp,'--manifest-sha',manifest_sha]
    replace(a,'--output',R/'runs'/vrid/'out');replace(a,'--scratch',R/'runs'/vrid/'scratch')
    v['env']=[e for e in d['env'] if not e.startswith(('VKGR_DECODE','VKGR_EVAL_WORKERS='))]+['VKGR_EVAL_WORKERS=1']
    return v

def accepts_verify(status,bound,records):return status=='succeeded' and bound and any(isinstance(v,dict) and v.get('status')=='verified' for v in records)
def snapshot(entry):
    job=entry['job'];expected=entry['expected'];rid=job['run_id'];tp=R/'runs'/rid/'terminal.json';vp=R/'runs'/(rid+'-verify')/'terminal.json'
    result={'run_id':rid,'seed':job['seed'],'source':expected['source'],'config_sha256':expected['config_sha256'],'data_hashes':expected['data_hashes'],
        'train_queue':where(Q,rid),'verify_queue':where(Q,rid+'-verify'),'train_status':load(tp)['status'] if tp.exists() else None,
        'verify_status':load(vp)['status'] if vp.exists() else None,'verified':False,'unaccepted':True,'exceptions':[]}
    result['rejected']=bool(result['train_queue']['rejected'] or result['verify_queue']['rejected'])
    if result['train_status']=='succeeded':
        claimed=one(rid)
        if any(claimed.get(k)!=v for k,v in job.items()):raise ValueError('claimed train differs from plan')
        mp=R/'runs'/rid/'out/manifest.json';m=load(mp);pin=sha(mp)
        if m['config']!=load(job['config']) or digest(m['binding'])!=expected['expected_binding_sha256']:raise ValueError('train binding / config drift')
        result.update(manifest_path=str(mp),manifest_sha256=pin)
        if result['verify_status'] is not None:
            log=R/'runs'/(rid+'-verify')/'logs/stdout.log';payload=log.read_bytes();records=[]
            for line in payload.decode().splitlines():
                try:records.append(json.loads(line))
                except ValueError:pass
            ver=one(rid+'-verify');wanted=derive_verify(job,pin);bound=all(ver.get(k)==v for k,v in wanted.items())
            result['verified']=accepts_verify(result['verify_status'],bound,records)
            result['verify_receipt']={'path':str(log),'sha256':hashlib.sha256(payload).hexdigest(),'job_bound':bound,'records':records}
        if result['verified']:
            sel=m['selection'];epoch=next(e for e in m['epochs'] if e['epoch']==sel['best_epoch'])
            result.update(best_epoch=sel['best_epoch'],stop_epoch=sel['epoch'],metrics=epoch['metrics']['metrics']['all_eligible'],unaccepted=False)
    return result

def summary_status(states,done):
    if all(s['verified'] for s in states):return 'verified'
    if done or any(s.get('exceptions') or s.get('rejected') or s.get('train_status') not in (None,'succeeded') or (s.get('verify_status') is not None and not s['verified']) for s in states):return 'ALERT'
    return 'running'
def can_start_observation(q,terminal_present):return terminal_present or any(q[k] for k in ('pending','claimed','rejected'))
def deadline_fallback(entries):return [{'run_id':e['job']['run_id'],'source':e.get('expected',{}).get('source'),'verified':False,'unaccepted':True,'exceptions':['deadline expired before first poll; terminal state not read']} for e in entries]

def poll(entries):
    states=[];done=True
    for e in entries:
        try:
            s=snapshot(e);rid=s['run_id']
            if s['train_status']=='succeeded' and not s['rejected'] and not exists(Q,R/'runs',rid+'-verify'):
                v=derive_verify(e['job'],s['manifest_sha256']);prepare_job(v);publish(Q,R/'runs',v);s=snapshot(e)
            settled=s['rejected'] or (s['train_status'] is not None and s['train_status']!='succeeded') or (s['train_status']=='succeeded' and s['verify_status'] is not None)
            done=done and settled
        except Exception as exc:
            s={'run_id':e['job']['run_id'],'source':e['expected']['source'],'verified':False,'unaccepted':True,'exceptions':[type(exc).__name__+': '+str(exc)]};done=False
        states.append(s)
    return states,done

def driver():
    configure(load(OPS/'plan.json')['entries'][0]['job']);identity();plan=load(OPS/'plan.json')
    lock=(OPS/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    new(OPS/'driver-pid.json',{'pid':os.getpid(),'ppid':os.getppid(),'poll_seconds':900,'deadline_epoch':plan['deadline_epoch'],'started_at_utc':now()})
    states=deadline_fallback(plan['entries'])
    while time.time()<plan['deadline_epoch']:
        states,done=poll(plan['entries']);value={'utc':now(),'runs':states,'all_terminal':done,'status':summary_status(states,done)}
        new(OPS/f'summary-{time.time_ns()}.json',value)
        if done:new(OPS/'summary-final.json',value);lock.close();return
        time.sleep(min(900,max(0,plan['deadline_epoch']-time.time())))
    value={'utc':now(),'status':'ALERT','reason':'48h deadline','runs':states,'all_terminal':False}
    new(OPS/'summary-deadline.json',value);new(OPS/'summary-final.json',value);lock.close()

def prepare_job(job):
    sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops')
    from vkgr_ops import launcher
    return launcher.prepare(job)

def read_status():
    out=inventory()
    for name in ('deployment.json','preflight.json','driver-pid.json','summary-final.json','summary-deadline.json'):
        if (OPS/name).exists():out[name]=load(OPS/name)
    if (OPS/'plan.json').exists():
        out['states']=[]
        for e in load(OPS/'plan.json')['entries']:
            try:out['states'].append(snapshot(e))
            except Exception as exc:out['states'].append({'run_id':e['job']['run_id'],'verified':False,'unaccepted':True,'exceptions':[str(exc)]})
    if (OPS/'driver-pid.json').exists():
        pid=load(OPS/'driver-pid.json')['pid']
        try:os.kill(pid,0);out['driver_alive']=True
        except ProcessLookupError:out['driver_alive']=False
        if out['driver_alive']:out['driver_current_ppid']=int(Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[1])
    if (R/'runs'/SMOKE/'terminal.json').exists():out['engineering_terminal']=load(R/'runs'/SMOKE/'terminal.json')
    if (R/'runs'/SMOKE/'out/engineering.json').exists():out['engineering']=load(R/'runs'/SMOKE/'out/engineering.json')
    return out

def action(req):
    act=req.get('action','inventory')
    if act=='inventory':return inventory()
    if act=='status':return read_status()
    if act=='deploy':return deploy(req)
    if act in ('prepare','preflight'):
        result=preflight()
        if act=='prepare':new(OPS/'preflight.json',result)
        return result
    if act in ('smoke-dry','smoke-submit'):
        identity();d=derive_job(42);d.update(run_id=SMOKE,disk_gib=2,wall_hours=1.,vram_mib=8192,
            argv=[PY,'-B','scripts/gtok02a1_cuda_check.py',str(R/'runs'/SMOKE/'out')])
        if exists(Q,R/'runs',SMOKE):raise FileExistsError(SMOKE)
        result={'status':'smoke_dry_ok','prepared':prepare_job(d),'job':d}
        if act=='smoke-submit':result['written_not_accepted']=publish(Q,R/'runs',d)
        return result
    if act in ('dry','submit'):
        entries=train_jobs(req)
        for e in entries:
            for rid in (e['job']['run_id'],e['job']['run_id']+'-verify'):
                if exists(Q,R/'runs',rid):raise FileExistsError(rid)
            prepare_job(e['job'])
        result={'status':'dry_ok','jobs':[e['job'] for e in entries]}
        if act=='submit':
            new(OPS/'plan.json',{'created_at_utc':now(),'deadline_epoch':time.time()+48*3600,'entries':entries})
            result['written_not_accepted']=[publish(Q,R/'runs',e['job']) for e in entries]
        return result
    if act=='start':
        identity();plan=load(OPS/'plan.json')
        if (OPS/'driver-pid.json').exists() or (OPS/'driver-launcher.log').exists():raise FileExistsError('driver already started')
        for e in plan['entries']:
            rid=e['job']['run_id']
            if not can_start_observation(where(Q,rid),(R/'runs'/rid/'terminal.json').exists()):raise ValueError('registered train has no queue or terminal receipt')
        log=(OPS/'driver-launcher.log').open('xb')
        p=subprocess.Popen([PY,'-B',str(DEP/'scripts/gtok02a1_ops.py'),'--driver'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,cwd=DEP,env={**os.environ,'PYTHONPATH':str(DEP/'src'),'OMP_NUM_THREADS':'1','VKGR_PINNED_MANIFEST':str(OPS/'pinned.json')})
        return {'status':'started','pid':p.pid}
    raise ValueError('unregistered action')

def main():
    if '--driver' in sys.argv:driver();return
    raw=sys.stdin.read();print(json.dumps(action(json.loads(raw) if raw.strip() else {}),sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
