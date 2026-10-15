"""Finite, frozen M1 batch: separate canary dispatcher and verifier/reporter.
No model calls in either driver. Real-input preflight is read-only; no test split.
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
DEP=Path('/workspace/vkgr-seqm1-p-v1');OPS=Path('/workspace/vkgr-seqm1-night-20261016')
REPO=Path('/workspace/vkgr-b1-src');PUB='0ca61bc35ad68bd79b86103b54e470445cbab07f'
BASE_INPUT=Path('/workspace/vkgr-b1-inputs/p-feat02-inputs-base-v1.json')
BASE_SHA='a12b98003aa2fbb58df83bc70bb448e426d01c2a4cc2e6357bf1f95778f23e38'
P3C='c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
ARMS=('mean','last','attn','bi','pure');SEEDS=(42,2024,2025);POLL=900
RIDS=tuple(f'p-seqm1-{a}-s{s}' for a in ARMS for s in SEEDS)
MEANS=RIDS[:3];DIAGS=RIDS[3:]

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def encoded(d):return json.dumps(d,sort_keys=True,allow_nan=False).encode()
def put_new(p,d):
    with p.open('xb') as f:f.write(encoded(d));f.flush();os.fsync(f.fileno())
def journal(folder,role,d):
    with (folder/(role+'.jsonl')).open('ab') as f:f.write(encoded({'utc':now(),**d})+b'\n')
def frozen():
    p=Path(__file__).with_name('seqm1_night_frozen.json')
    if not p.exists():p=OPS/'code/analysis/seqm1_night_frozen.json'
    d=load(p)
    if d['model_commit']!=PUB or tuple(d['arms'])!=ARMS or tuple(d['seeds'])!=SEEDS or set(d['config_sha256'])!=set(RIDS) or d['poll_seconds']!=900 or d['max_driver_hours']!=48:
        raise ValueError('unregistered frozen batch')
    return d
def check_config(base,cfg,arm,seed):
    if arm not in ARMS or seed not in SEEDS or cfg!={**base,'arm':arm,'seed':seed}:raise ValueError('only fixed arm/seed changes permitted')
def flags(a,name):
    if a.count(name)!=1:raise ValueError('missing/duplicate '+name)
    return a[a.index(name)+1]
def replace(a,k,v):a[a.index(k)+1]=v
def queue(rid):
    from gtok02_seed_batch import where
    return where(Q,rid)
def unique_job(rid):
    fs=list((Q/'claimed').glob('*-'+rid+'.json'))
    if len(fs)!=1:raise ValueError('unique accepted template required: '+rid)
    return load(fs[0])
def derive_train(parent,rid,pin):
    if rid not in RIDS:raise ValueError('unregistered M1 run')
    seed=int(rid.rsplit('s',1)[1]);d=copy.deepcopy(parent);rel=f'configs/seqm1/{rid}.json'
    d.update(run_id=rid,seed=seed,cwd=str(DEP),config=str(DEP/rel),vram_mib=1024)
    a=d['argv'];a[a.index('-m')+1]='vkgr.retrieval_seqm1_run'
    if a[a.index('-m')+2]!='train' or flags(a,'--inputs')!=str(BASE_INPUT) or flags(a,'--inputs-sha')!=BASE_SHA:raise ValueError('frozen P train template mismatch')
    replace(a,'--config',rel);replace(a,'--config-sha',pin);replace(a,'--output',str(R/'runs'/rid/'out'))
    if (d['cpu_cores'],d['ram_mib'],d['disk_gib'],d['wall_hours'],d['kind'],d['no_admission'])!=(1,9216,8,8.,'real',False):raise ValueError('P train resources/admission mismatch')
    if d['pinned_manifest'] is not None or d['pinned_files']!=['docs/p3c_validation.md='+P3C]:raise ValueError('P dirty-file pins mismatch')
    for name in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        d['env']=[x for x in d['env'] if not x.startswith(name+'=')]+[name+'=1']
    return d
def derive_verify(train,manifest_sha):
    d=copy.deepcopy(train);rid=d['run_id'];mp=R/'runs'/rid/'out/manifest.json'
    d.update(run_id=rid+'-verify',disk_gib=1,wall_hours=3.,inputs=list(d['inputs'])+[str(mp)])
    a=d['argv'];a[a.index('-m')+2]='verify';i=a.index('--output');del a[i:i+2]
    a.extend(['--manifest',str(mp),'--manifest-sha',manifest_sha]);return d
def publish_job(job):
    from gtok02_seed_batch import publish
    return publish(Q,R/'runs',queue_payload(job))

def gate(states):
    for rid in MEANS:
        s=states[rid]
        if s.get('exceptions') or s.get('rejected') or s['train_status'] not in ('running','pending','not_submitted','succeeded'):
            return 'ALERT'
        if s.get('verify_status') not in (None,'running','pending','succeeded') or (s.get('verify_status')=='succeeded' and not s['verified']):return 'ALERT'
    return 'RELEASE' if any(states[r]['train_status']=='succeeded' and states[r]['verified'] for r in MEANS) else 'WAIT'

def snapshot(job):
    rid=job['run_id'];v=rid+'-verify';tq=queue(rid);vq=queue(v)
    if sha(job['config'])!=flags(job['argv'],'--config-sha'):raise ValueError('frozen config changed after submission')
    out={'run_id':rid,'expected_commit':PUB,'config_sha256':flags(job['argv'],'--config-sha'),
         'expected_inputs_sha256':BASE_SHA,'train_queue':tq,'verify_queue':vq,'train_status':'not_submitted','verify_status':None,'verified':False,
         'rejected':bool(tq['rejected'] or vq['rejected']),'exceptions':[],
         'data_hashes':job.get('expected_data_hashes'),'best_epoch':None,'stop_epoch':None,'recall_at_50':None,'ndcg_at_50':None,'ablation_g0':None,'branch_ratio_median':None}
    for prefix,key,state in ((rid,'train_status',tq),(v,'verify_status',vq)):
        term=R/'runs'/prefix/'terminal.json'
        if term.exists():out[key]=load(term)['status']
        elif state['claimed']:out[key]='running'
        elif state['pending']:out[key]='pending'
        elif state['rejected']:out[key]='rejected'
    if out['train_status']=='succeeded':
        mp=R/'runs'/rid/'out/manifest.json';m=load(mp);ms=sha(mp);b=m['binding'];sel=m['selection']
        if m['status']!='complete' or m['config']!=load(job['config']) or b['source']['git_commit']!=PUB:raise ValueError('completed model/config identity mismatch')
        expected=job.get('expected_binding_sha256')
        binding_sha=hashlib.sha256(json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if expected and binding_sha!=expected:raise ValueError('real input/binding differs from preflight')
        best=[e for e in m['evaluations'] if e['epoch']==sel['best_epoch']]
        if len(best)!=1:raise ValueError('best epoch cardinality')
        out.update(manifest_path=str(mp),manifest_sha256=ms,commit=b['source']['git_commit'],data_hashes={k:b[k] for k in ('rows_sha256','queries_sha256','catalog_sha256','support_sha256','content_sha256')})
        from phase2_validation_read import verifier
        proof=verifier(rid,ms);out['verification_receipt']=proof;out['verified']=proof['verified']
        if out['verified']:
            out.update(best_epoch=sel['best_epoch'],stop_epoch=sel['epoch'],recall_at_50=sel['best_metric'][0],ndcg_at_50=sel['best_metric'][1],
                       ablation_g0=m['ablation_g0'],branch_ratio_median=best[0]['diagnostics']['branch_ratio_median'])
            if m['ablation_g0'] is not None:out['ablation_delta_hits']=round(m['ablation_g0']['delta_recall@50']*2520)
    out['unaccepted']=not out['verified'];return out
def mean_states(plan):
    return {j['run_id']:snapshot(j) for j in plan['jobs'] if j['run_id'] in MEANS}
def dispatch_once(folder,plan):
    terminal=folder/'dispatch-terminal.json'
    if terminal.exists():return load(terminal)
    try:states=mean_states(plan);decision=gate(states)
    except Exception as exc:
        result={'status':'ALERT','exception':type(exc).__name__+': '+str(exc),'utc':now()};put_new(terminal,result);return result
    if decision=='WAIT':return {'status':'waiting_canary'}
    result={'status':'ALERT' if decision=='ALERT' else 'released','mean_states':states,'published':[],'utc':now()}
    if decision=='RELEASE':
        for job in plan['jobs']:
            if job['run_id'] not in DIAGS:continue
            try:
                if gate(mean_states(plan))=='ALERT':raise ValueError('mean failed while dispatching')
                name=publish_job(job);result['published'].append({'run_id':job['run_id'],'queue_file':name,'accepted':False})
            except Exception as exc:
                result.update(status='ALERT',exception=type(exc).__name__+': '+str(exc));break
    put_new(terminal,result);return result
def verify_once(folder,plan):
    rows=[]
    for job in plan['jobs']:
        try:
            s=snapshot(job)
            if s['train_status']=='succeeded' and not s['rejected'] and s.get('verify_status') is None:
                try:s['verify_written_not_accepted']=publish_job(derive_verify(job,s['manifest_sha256']))
                except FileExistsError:s['existing_verify_no_resubmit']=True
        except Exception as exc:s={'run_id':job['run_id'],'exceptions':[type(exc).__name__+': '+str(exc)],'verified':False,'train_status':'UNKNOWN','verify_status':None}
        s['unaccepted']=not s['verified'];rows.append(s)
    return {'utc':now(),'runs':rows,'model_commit':PUB,'ops_commit':plan.get('ops_commit'),'approval':plan.get('approval'),'preflight_sha256':plan.get('preflight_sha256')}

def settled(s,dispatch_done):
    if s.get('rejected') or s['train_status'] in ('failed','killed','stopped','cancelled','rejected','launch_failed','wall_cap_exceeded'):return True
    if s['train_status']=='not_submitted':return dispatch_done
    return s.get('verify_status') in ('succeeded','failed','killed','stopped','cancelled','rejected','launch_failed','wall_cap_exceeded')
def run_role(role):
    if role not in ('dispatcher','verifier'):raise ValueError('fixed driver role required')
    folder=OPS;lock=(folder/(role+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    put_new(folder/(role+'-pid.json'),{'pid':os.getpid(),'started_at_utc':now(),'ppid_at_start':os.getppid()})
    plan=load(folder/'plan.json');deadline=plan['deadline_epoch']
    while time.time()<deadline:
        if role=='dispatcher':
            result=dispatch_once(folder,plan);journal(folder,role,result)
            if result['status']!='waiting_canary':return
        else:
            result=verify_once(folder,plan);dispatch_done=(folder/'dispatch-terminal.json').exists()
            dispatch=load(folder/'dispatch-terminal.json') if dispatch_done else None
            result['dispatch']=dispatch
            result['mean_gate']=gate({s['run_id']:s for s in result['runs'] if s['run_id'] in MEANS})
            if result['mean_gate']=='ALERT':result['ALERT']='mean failure/unverified terminal; dispatcher must not release further jobs'
            terminal=dispatch_done and all(settled(s,True) for s in result['runs'])
            put_new(folder/f'summary-{time.time_ns()}.json',result);journal(folder,role,{'all_terminal':terminal,'mean_gate':result['mean_gate'],'unaccepted':[s['run_id'] for s in result['runs'] if s['unaccepted']]})
            if terminal:put_new(folder/'summary-final.json',result);return
        time.sleep(POLL)
    if role=='dispatcher':
        if not (folder/'dispatch-terminal.json').exists():put_new(folder/'dispatch-terminal.json',{'status':'ALERT','reason':'48h deadline reached','utc':now()})
    else:
        result=verify_once(folder,plan);result['ALERT']='48h deadline reached; unfinished jobs preserved';put_new(folder/'summary-deadline.json',result)

def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],stderr=subprocess.STDOUT)
def check_tree(folder,commit,override=False):
    files=git('ls-tree','-r','--name-only',commit).decode().splitlines()
    for name in files:
        expected=P3C if override and name=='docs/p3c_validation.md' else hashlib.sha256(git('show',commit+':'+name)).hexdigest()
        if sha(folder/name)!=expected:raise ValueError('deployed file SHA mismatch: '+name)
    return len(files)
def deploy(req):
    if OPS.exists():raise FileExistsError('new night ops directory required')
    OPS.mkdir();git('fetch','origin','codex/phase2-readonly-tools');git('worktree','add','--detach',str(OPS/'code'),req['ops_commit'])
    count=check_tree(OPS/'code',req['ops_commit'])
    if sha(OPS/'code/analysis/seqm1_night.py')!=req['source_sha256']:raise ValueError('night source pin mismatch')
    receipt={'ops_commit':req['ops_commit'],'ops_files_checked':count,'model_files_checked':check_tree(DEP,PUB,True),'model_commit':PUB,'source_sha256':req['source_sha256']}
    put_new(OPS/'deployment.json',receipt);return receipt

def preflight():
    sys.path.insert(0,str(OPS/'code/analysis'))
    from seqm1_prepare import setup
    from seqm1_baseline_read import cpu_room
    room=cpu_room()
    if not room['one_core_fits']:return {'status':'deferred_cpu','cpu':room}
    setup()
    import gc
    import torch
    from vkgr import retrieval_b1_run as b1,retrieval_feat02_run as fr,retrieval_seqm1_run as m1
    from vkgr.retrieval_ablation_run import read_pinned
    torch.set_num_threads(1);b1.execution_context('cpu');f=frozen()
    if b1.ROOT!=DEP or sha(DEP/'docs/p3c_validation.md')!=P3C:raise ValueError('preflight model root mismatch')
    spec=fr.validate_inputs(read_pinned(BASE_INPUT,BASE_SHA),m1.FEATURE)
    data,_,base=b1.real_inputs(spec);data,extra=fr.apply_feat02(data,spec,50);inputs={**base,**extra,'inputs_sha256':BASE_SHA}
    rows,queries=fr._view(data,50);counts={'rows':len(rows),'queries':len(queries),'catalog':len(data['catalog'])}
    if counts!={'rows':86785,'queries':2520,'catalog':120415}:raise ValueError('real population mismatch')
    mean=read_pinned(DEP/'configs/seqm1/p-seqm1-mean-s42.json',f['config_sha256'][MEANS[0]]);entries={}
    for arm in ARMS:
        for seed in SEEDS:
            rid=f'p-seqm1-{arm}-s{seed}';cfg=read_pinned(DEP/f'configs/seqm1/{rid}.json',f['config_sha256'][rid]);m1.validate_config(cfg);check_config(mean,cfg,arm,seed)
            if (R/'runs'/rid).exists() or any(queue(rid).values()) or any(queue(rid+'-verify').values()):raise FileExistsError(rid)
            model=m1._model(data,cfg,'cpu');binding=m1.binding_for4(data,cfg,rows,queries,model,'cuda',inputs)
            entries[rid]={'status':'preflight_ok','config_sha256':f['config_sha256'][rid],'arm':arm,'seed':seed,'counts':counts,
                          'expected_binding_sha256':b1._digest(binding),'data_hashes':{k:binding[k] for k in ('rows_sha256','queries_sha256','catalog_sha256','support_sha256','content_sha256')},'param_report':binding['param_report']}
            del model;gc.collect()
    return {'status':'preflight_ok','entries':entries,'binding_inputs':inputs,'model_commit':PUB,'approval_sections_sha256':f['approval_sections_sha256'],'test_read':False,'training_started':False,'cpu':room}

def planned(req):
    f=frozen();pf=req['preflight']
    if pf['status']!='preflight_ok' or set(pf['entries'])!=set(RIDS) or pf['model_commit']!=PUB or pf['approval_sections_sha256']!=f['approval_sections_sha256'] or pf['test_read'] or pf['training_started']:raise ValueError('full approved preflight receipt required')
    parent=unique_job('p-php01-h1-t010-s42');jobs=[]
    for rid in RIDS:
        e=pf['entries'][rid]
        if e['status']!='preflight_ok' or e['config_sha256']!=f['config_sha256'][rid] or sha(DEP/f'configs/seqm1/{rid}.json')!=e['config_sha256']:raise ValueError('preflight/config mismatch')
        job=derive_train(parent,rid,e['config_sha256']);job['expected_binding_sha256']=e['expected_binding_sha256'];job['expected_data_hashes']=e['data_hashes']
        # Queue schema is closed: expected binding travels in plan, not job JSON.
        jobs.append(job)
    return jobs
def queue_payload(job):return {k:v for k,v in job.items() if not k.startswith('expected_')}
def main():
    if '--role' in sys.argv:
        sys.path.insert(0,str(OPS/'code/analysis'));run_role(sys.argv[sys.argv.index('--role')+1]);return
    raw=sys.stdin.read();req=json.loads(raw) if raw.strip() else {'action':'status'};action=req['action']
    sys.path.insert(0,str(OPS/'code/analysis'))
    if action=='deploy':answer=deploy(req)
    elif action=='preflight':answer=preflight()
    elif action in ('dry','submit-mean'):
        from vkgr_ops import launcher
        jobs=planned(req);prepared={}
        for job in jobs:
            rid=job['run_id']
            if (R/'runs'/rid).exists() or any(queue(rid).values()) or any(queue(rid+'-verify').values()):raise FileExistsError(rid)
            prepared[rid]=launcher.prepare(queue_payload(job))
        answer={'status':'dry_ok','jobs':jobs,'launcher_prepare':prepared,'approval':frozen(),'preflight_sha256':hashlib.sha256(encoded(req['preflight'])).hexdigest()}
        if action=='submit-mean':
            lock=(OPS/'submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (OPS/'plan.json').exists():raise FileExistsError('plan already exists; inspect, never resubmit')
            plan={'jobs':jobs,'approval':frozen(),'preflight':req['preflight'],'preflight_sha256':answer['preflight_sha256'],'ops_commit':load(OPS/'deployment.json')['ops_commit'],'created_at_utc':now(),'deadline_epoch':time.time()+48*3600}
            put_new(OPS/'plan.json',plan)
            answer['written_not_accepted']=[publish_job(queue_payload(j)) for j in jobs if j['run_id'] in MEANS]
    elif action=='start':
        plan=load(OPS/'plan.json')
        for rid in MEANS:
            s=queue(rid)
            if s['rejected'] or not (s['pending'] or s['claimed']):raise ValueError('mean must already be queued without rejection')
        if list(OPS.glob('*-pid.json')):raise FileExistsError('driver already started')
        source=OPS/'code/analysis/seqm1_night.py';py=plan['jobs'][0]['argv'][0];pids={}
        for role in ('verifier','dispatcher'):
            log=(OPS/(role+'-launcher.log')).open('xb')
            p=subprocess.Popen([py,'-B',str(source),'--role',role],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
            pids[role]=p.pid
        answer={'started':pids,'poll_seconds':900,'deadline_epoch':plan['deadline_epoch']}
    elif action=='status':
        answer={'queues':{r:{'train':queue(r),'verify':queue(r+'-verify')} for r in RIDS},'drivers':[]}
        for p in OPS.glob('*-pid.json'):
            v=load(p);proc=Path('/proc')/str(v['pid']);v.update(role=p.stem,alive=proc.exists())
            if proc.exists():v['ppid']=int(next(s.split()[1] for s in (proc/'status').read_text().splitlines() if s.startswith('PPid:')))
            answer['drivers'].append(v)
        for name in ('dispatch-terminal.json','summary-final.json','summary-deadline.json'):
            if (OPS/name).exists():answer[name]=load(OPS/name)
        snapshots=sorted(OPS.glob('summary-[0-9]*.json'))
        if snapshots:answer['latest_summary']=load(snapshots[-1])
    else:raise ValueError('unregistered action')
    answer['read_at_utc']=now();print(json.dumps(answer,sort_keys=True,allow_nan=False))

if __name__=='__main__':
    sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops')
    try:main()
    except Exception as exc:
        if '--role' in sys.argv and OPS.exists():
            role=sys.argv[sys.argv.index('--role')+1]
            failure={'status':'ALERT','exception':type(exc).__name__+': '+str(exc),'utc':now()}
            journal(OPS,role,failure)
            target=OPS/('dispatch-terminal.json' if role=='dispatcher' else 'verifier-fatal.json')
            if not target.exists():put_new(target,failure)
        raise
