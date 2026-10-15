"""Frozen GTOK-02 seed supplement launcher and finite server-side verifier driver.

Approved transport: source on stdin defaults to read-only P2 dry-run; writer mode
accepts a JSON request on stdin. Driver code is persisted from the pushed source,
not edited on the server. No test data, tensors, checkpoints or embeddings read.
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

R = Path('/workspace/vkgr-infra01-root')
Q = R/'queue'
G = Path('/workspace/vkgr-gtok02-g')
PIN = Path('/workspace/vkgr-b1-pinned/g-0c87d32ce767.json')
PIN_SHA = '07fdc5148cd59aabb5d7956d81f35674c3f02a7aebdabfa39286a7b62d9068b5'
VARIANTS = ('b32d3','b64d3','b128d3','b512d3','b1024d3','b256d2')
INTERVAL = 900


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def flag(a, k):
    if a.count(k) != 1: raise ValueError('argv missing/duplicate flag '+k)
    return a[a.index(k)+1]
def put(a, k, v): a[a.index(k)+1] = v


def check_seed_config(base, new, seed):
    if seed not in (2024,2025) or base.get('seed') != 42 or new != {**base,'seed':seed}:
        raise ValueError('supplement must change seed only')


def pick_p1(rows):
    ranked=sorted(rows,key=lambda x:(-x['hits'],-x['ndcg'],x['order']))
    return [x for i,x in enumerate(ranked) if i<2 or ranked[0]['hits']-x['hits']<=10]


def derive_train(parent, seed, config_sha):
    d=copy.deepcopy(parent); old=d['run_id']; new=old.removesuffix('-s42')+'-s'+str(seed)
    rel=flag(d['argv'],'--config').replace('-s42.json',f'-s{seed}.json')
    d['run_id']=new; d['seed']=seed; d['config']=d['config'].replace('-s42.json',f'-s{seed}.json')
    put(d['argv'],'--config',rel);put(d['argv'],'--config-sha',config_sha)
    for k in ('--output','--scratch'): put(d['argv'],k,flag(d['argv'],k).replace('/'+old+'/', '/'+new+'/'))
    changed={k for k in d if d[k]!=parent[k]}
    if not changed <= {'run_id','seed','config','argv'}: raise ValueError('unapproved job change')
    return d


def where(q, rid):
    out={k:sorted(p.name for p in (q/k).glob('*-'+rid+'.json')) for k in ('pending','claimed','rejected')}
    for name in out['rejected']:
        p=q/'rejected'/name
        for reason in (p.with_suffix('.reason'),Path(str(p)+'.reason')):
            if reason.exists():out.setdefault('reasons',{})[reason.name]=reason.read_text()[:2000]
    return out


def publish(q, runs, job, name=None):
    rid=job['run_id']
    with (q/'.codex-seed-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (runs/rid).exists() or any(where(q,rid).values()): raise FileExistsError(rid)
        name=name or f'{time.time_ns():020d}-{rid}.json'
        tmp=q/'pending'/('.'+name+'.tmp')
        with tmp.open('x') as f: json.dump(job,f,sort_keys=True);f.flush();os.fsync(f.fileno())
        # Hard-link publication is atomic and refuses an existing destination.
        # Keep the private .tmp as provenance; never remove existing files.
        os.link(tmp,q/'pending'/name)
    return name


def is_verified(obj, pin): return obj.get('status')=='verified' and obj.get('manifest_sha256')==pin


def watchable_queue(state):
    return bool(state['pending'] or state['claimed']) and not state['rejected']


def accepted(rid):
    out=R/'runs'/rid/'out';mp=out/'manifest.json';m=load(mp)
    if load(R/'runs'/rid/'terminal.json')['status']!='succeeded': raise ValueError('parent train not succeeded')
    proof=verify_proof(rid)
    if not proof['verified']:raise ValueError('parent scientific verify not bound/verified: '+rid)
    return m,proof


def verify_proof(rid):
    mp=R/'runs'/rid/'out/manifest.json';pin=sha(mp);vd=R/'runs'/(rid+'-verify');vj=one(rid+'-verify')
    bound=flag(vj['argv'],'--manifest')==str(mp) and flag(vj['argv'],'--manifest-sha')==pin
    log=vd/'logs/stdout.log';records=[]
    if log.exists():
        if log.stat().st_size>1024*1024:raise ValueError('oversized verifier stdout')
        for line in log.read_text().splitlines():
            try: records.append(json.loads(line))
            except ValueError:pass
    ok=bound and load(vd/'terminal.json')['status']=='succeeded' and any(isinstance(x,dict) and x.get('status')=='verified' and x.get('manifest_sha256',pin)==pin for x in records)
    return {'verified':ok,'manifest_sha256':pin,'verify_path':str(log),'verify_sha256':sha(log),'job_manifest_binding':bound}


def one(rid):
    fs=list((Q/'claimed').glob('*-'+rid+'.json'))
    if len(fs)!=1: raise ValueError('unique claimed parent required: '+rid)
    return load(fs[0])


def pinned():
    if not PIN.is_absolute() or sha(PIN)!=PIN_SHA: raise ValueError('pinned manifest changed')
    p=load(PIN)
    for name,pin in p['files'].items():
        f=G/name
        if f.is_symlink() or sha(f)!=pin: raise ValueError('pinned source mismatch: '+name)
    for name,rec in p.get('overrides',{}).items():
        if sha(G/name)!=rec['sha256']: raise ValueError('pinned override mismatch')
    return p


def rank_new():
    rows=[]
    for order,v in enumerate(VARIANTS):
        rid=f'g-tok2-{v}-s42';m,proof=accepted(rid);bm=m['selection']['best_metric'];hits=round(bm[0]*2520)
        if abs(hits/2520-bm[0])>1e-12: raise ValueError('noninteger hits')
        enum={'b32d3':1,'b64d3':2,'b128d3':3,'b512d3':5,'b1024d3':6,'b256d2':7}
        rows.append({'variant':v,'run_id':rid,'hits':hits,'recall':bm[0],'ndcg':bm[1],'order':enum[v],'proof':proof})
    # Already accepted existing scan points; no new experiments for these rows.
    rows += [{'variant':'b16d3','hits':190,'ndcg':0.025774949,'order':0},
             {'variant':'b256d3','hits':254,'ndcg':0.03153632788646649,'order':4}]
    return rows,pick_p1(rows)


def plan(variants):
    pins=pinned();jobs=[];proofs=[]
    for v in variants:
        if v not in VARIANTS: raise ValueError('only new GTOK-02 points supported by this launcher')
        rid=f'g-tok2-{v}-s42';parent=one(rid);_,proof=accepted(rid)
        if parent['cwd']!=str(G) or parent['pinned_manifest']!=str(PIN) or parent['cpu_cores']!=1 or parent['vram_mib']!=3072:
            raise ValueError('frozen template resources/cwd/pin mismatch')
        base=load(parent['config'])
        for seed in (2024,2025):
            cfg=Path(parent['config'].replace('-s42.json',f'-s{seed}.json'));cfgsha=sha(cfg)
            rel=str(cfg.relative_to(G))
            if pins['files'].get(rel)!=cfgsha: raise ValueError('seed config absent from original pinned freeze')
            check_seed_config(base,load(cfg),seed)
            d=derive_train(parent,seed,cfgsha)
            if not all(Path(x).is_file() for x in d['inputs']): raise ValueError('input missing')
            jobs.append(d);proofs.append({'run_id':d['run_id'],'config_sha256':cfgsha,'config_seed_only':True,'seed42_proof':proof})
    return jobs,proofs,pins['target_commit']


def derive_verify(train):
    d=copy.deepcopy(train);rid=train['run_id'];v=rid+'-verify';mp=R/'runs'/rid/'out/manifest.json'
    a=d['argv'];i=a.index('-m');a[i+2]='verify'
    for f in ('--config','--config-sha'):
        i=a.index(f);del a[i:i+2]
    i=a.index('--output');a[i:i]=['--manifest',str(mp),'--manifest-sha',sha(mp)]
    for f,leaf in (('--output','out'),('--scratch','scratch')):put(a,f,str(R/'runs'/v/leaf))
    d.update(run_id=v,config=None,wall_hours=2.0,disk_gib=1,inputs=list(train['inputs'])+[str(mp)],
             env=[x for x in train['env'] if not x.startswith('VKGR_DECODE')]+['VKGR_EVAL_WORKERS=1'])
    return d


def driver(folder):
    p=load(folder/'plan.json');lock=(folder/'driver.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (folder/'driver.pid').open('x') as f:f.write(str(os.getpid()))
    deadline=time.time()+48*3600
    while time.time()<deadline:
        state={};done=0
        for job in p['jobs']:
            rid=job['run_id'];vp=R/'runs'/(rid+'-verify');t=R/'runs'/rid/'terminal.json'
            entry={'train_queue':where(Q,rid),'verify_queue':where(Q,rid+'-verify')}
            if t.exists():
                term=load(t);entry['train_terminal']=term['status']
                if term['status']!='succeeded':done+=1
                elif (vp/'terminal.json').exists():
                    vt=load(vp/'terminal.json');entry['verify_terminal']=vt['status'];done+=1
                    if vt['status']=='succeeded':
                        try:entry['verification']=verify_proof(rid)
                        except (OSError,ValueError,KeyError) as exc:entry['verification_error']=type(exc).__name__+': '+str(exc)
                    else:entry['alert']='verify terminal failure; no retry'
                elif entry['verify_queue']['rejected']:
                    done+=1;entry['alert']='verify rejected; never resubmitting'
                elif not vp.exists() and not any(entry['verify_queue'].values()):
                    try:entry['verify_written']=publish(Q,R/'runs',derive_verify(job))
                    except FileExistsError:entry['verify_existing']='another owner already published; no resubmit'
            elif entry['train_queue']['rejected']:done+=1;entry['alert']='train rejected'
            state[rid]=entry
        with (folder/'driver.jsonl').open('a') as f:f.write(json.dumps({'utc':now(),'pid':os.getpid(),'ppid':os.getppid(),'state':state,'all_terminal':done==len(p['jobs'])})+'\n')
        if done==len(p['jobs']):return
        time.sleep(INTERVAL)


def main():
    if '--driver' in sys.argv:
        driver(Path(sys.argv[sys.argv.index('--driver')+1]));return
    raw=sys.stdin.read();req=json.loads(raw) if raw.strip() else {'action':'dry','variants':['b256d2']}
    action=req['action'];variants=req.get('variants',['b256d2'])
    if action not in ('dry','submit','start-driver','inspect'):raise ValueError('action not allowed')
    jobs,proofs,target=plan(variants)
    rows,selected=rank_new()
    allowed_p1={x['variant'] for x in selected if x['variant'] in VARIANTS}-{'b256d2'}
    if len(set(variants))!=len(variants) or (set(variants)!={'b256d2'} and set(variants)!=allowed_p1):
        raise ValueError('batch must be P2 or the deduplicated preregistered P1 set')
    if action=='submit' and set(variants)!={'b256d2'}:
        for seed in (2024,2025):
            st=where(Q,f'g-tok2-b256d2-s{seed}')
            if not st['claimed'] or st['rejected']:raise ValueError('P2 must be accepted before P1')
    result={'utc':now(),'action':action,'rows':rows,'p1_selected':[x['variant'] for x in selected], 'proofs':proofs,
            'jobs':[{'run_id':j['run_id'],'seed':j['seed'],'vram_mib':j['vram_mib'],'queue':where(Q,j['run_id'])} for j in jobs],'target_commit':target,'pinned_sha256':PIN_SHA,
            'supplement_queue':{f'g-tok2-{v}-s{s}':where(Q,f'g-tok2-{v}-s{s}') for v in VARIANTS for s in (2024,2025)}}
    result['drivers']=[]
    for f in Path('/workspace/vkgr-codex-seed-drivers').glob('*/driver.pid'):
        pid=int(f.read_text());pr=Path('/proc')/str(pid);rec={'path':str(f.parent),'pid':pid,'alive':pr.exists()}
        if pr.exists():rec['ppid']=int(next(x.split()[1] for x in (pr/'status').read_text().splitlines() if x.startswith('PPid:')))
        result['drivers'].append(rec)
    if action=='submit':
        lock=(R/'codex-gtok02-seeds.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Check the entire batch before the first write; existing work is never reset.
        for j in jobs:
            if (R/'runs'/j['run_id']).exists() or any(where(Q,j['run_id']).values()):raise FileExistsError(j['run_id'])
        result['written']=[publish(Q,R/'runs',j) for j in jobs]
    elif action=='start-driver':
        folder=Path(req['driver_dir'])
        if folder.parent!=Path('/workspace/vkgr-codex-seed-drivers'):raise ValueError('fixed new driver parent required')
        folder.parent.mkdir(exist_ok=True);folder.mkdir(exist_ok=False)
        for j in jobs:
            if not watchable_queue(where(Q,j['run_id'])):raise ValueError('driver requires an existing queued own run; no rejection')
        repo=Path('/workspace/vkgr-b1-src');code=folder/'code'
        subprocess.run(['git','-C',str(repo),'fetch','origin','codex/phase2-readonly-tools'],check=True,capture_output=True)
        subprocess.run(['git','-C',str(repo),'worktree','add','--detach',str(code),req['ops_commit']],check=True,capture_output=True)
        names=subprocess.check_output(['git','-C',str(code),'ls-tree','-r','--name-only',req['ops_commit']],text=True).splitlines()
        for name in names:
            expected=hashlib.sha256(subprocess.check_output(['git','-C',str(code),'show',req['ops_commit']+':'+name])).hexdigest()
            if sha(code/name)!=expected:raise ValueError('deployed source mismatch: '+name)
        script=code/'analysis/gtok02_seed_batch.py'
        if sha(script)!=req['source_sha256']:raise ValueError('pushed driver source hash')
        with (folder/'plan.json').open('x') as f:json.dump({'jobs':jobs,'ops_commit':req['ops_commit'],'source_sha256':req['source_sha256'],'train_commit':target,'proofs':proofs,'interval_seconds':INTERVAL},f,sort_keys=True)
        log=(folder/'launcher.log').open('ab')
        proc=subprocess.Popen([sys.executable,'-B',str(script),'--driver',str(folder)],start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=log,close_fds=True)
        result['driver']={'pid':proc.pid,'path':str(folder),'interval_seconds':INTERVAL}
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
