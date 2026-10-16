"""Versioned P-SEQ-COMBO-STACK deployment, admission and finite server verifier.
All server paths come from private parameters JSON. Every action is explicit.
"""
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
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
COMBO_COMMIT='84b160cb0eeb3c53b877f6cf6474276ba06a6ba5'
P3C_SHA='c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
PATH_KEYS=('infra_root','deployment','ops','source','python','inputs','data_root','template','ops_library','combo_review')
SEEDS=(42,2024,2025);RIDS=tuple(f'p-seq-combo-stack-s{s}' for s in SEEDS);SMOKE='p-seq-combo-stack-cuda-v1'

def configure(paths):
    global R,Q,DEP,OPS,SRC,PY,INPUT,P3C,TEMPLATE,LIB,REVIEW,PARAMETERS
    if type(paths) is not dict or set(paths)!=set(PATH_KEYS) or any(type(v) is not str or not Path(v).is_absolute() for v in paths.values()):
        raise ValueError('explicit private absolute paths required')
    PARAMETERS=dict(paths)
    R=Path(paths['infra_root']);Q=R/'queue';DEP=Path(paths['deployment']);OPS=Path(paths['ops']);SRC=Path(paths['source'])
    PY=paths['python'];INPUT=Path(paths['inputs']);P3C=Path(paths['data_root'])/'docs/p3c_validation.md'
    TEMPLATE=Path(paths['template']);LIB=Path(paths['ops_library']);REVIEW=Path(paths['combo_review'])
    os.environ['VKGR_DATA_ROOT']=paths['data_root']
    sys.path.insert(0,str(DEP/'src'));sys.path.insert(0,str(DEP/'scripts'))

def check_combo_review(path,digest):
    if sha(path)!=digest:raise ValueError('combo review receipt hash mismatch')
    proof=load(path)
    if proof.get('verdict')!='APPROVED' or proof.get('commit')!=COMBO_COMMIT:
        raise ValueError('combo independent leakage review not APPROVED for frozen commit')
    return True

def check_leakage_receipt(record,entry,source):
    if record.get('schema')!='seq-combo-stack-leakage-v1' or record.get('model')!='SeqComboStack' or record.get('status')!='pass' or record.get('engineering') is not False:
        raise ValueError('stack own production leakage receipt required')
    if record.get('source')!=source or record.get('config_sha256')!=entry['config_sha256'] or record.get('binding_sha256')!=entry['expected_binding_sha256'] or record.get('data_hashes')!=entry['data_hashes']:
        raise ValueError('stack leakage source/data/config binding mismatch')
    if (record.get('train_samples'),record.get('validation_samples'))!=(512,2520) or any(record.get(k) is not False for k in ('test_read','checkpoint_read','training')):
        raise ValueError('stack leakage population or scope mismatch')
    checks=record.get('checks',{})
    for key in ('L1','L2','L3','L4','L5','L8','L9'):
        if checks.get(key,{}).get('status')!='pass' or type(checks[key].get('samples')) is not int or checks[key]['samples']<=0:raise ValueError('missing stack runtime check: '+key)
    for key in ('L1','L2','L3','L8'):
        if checks[key]['samples']!=3032:raise ValueError('incomplete runtime population: '+key)
    for key in ('L5','L9'):
        if checks[key]['samples']!=120415:raise ValueError('incomplete candidate catalog: '+key)
    if checks['L2'].get('train_samples')!=512 or checks['L2'].get('validation_samples')!=2520 or checks['L9'].get('train_samples')!=512 or checks['L4']['samples']>3032:
        raise ValueError('runtime subset counts mismatch')
    nonzero=record.get('nonzero',{})
    gate=nonzero.get('gate')
    if type(gate) not in (int,float) or not math.isfinite(gate) or gate<=0 or nonzero.get('recency') is not True or nonzero.get('beta') is not True:
        raise ValueError('nonzero stack pooling and open branch required')
    feedback=checks.get('L3_own_feedback',{})
    if feedback.get('status')!='pass' or type(feedback.get('samples')) is not int or not 1<=feedback['samples']<=128:
        raise ValueError('own-event feedback and branch isolation oracle required')
    for key in ('L6','L7','L10'):
        if checks.get(key,{}).get('status')!='STATIC_REVIEWED' or len(checks[key].get('source_sha256',''))!=64:raise ValueError('missing static mapping: '+key)
    return True

def check_engineering(proof,commit):
    vram=proof.get('vram',{});rows=vram.get('variants',[])
    if proof.get('status')!='cuda_engineering_verified' or proof.get('commit')!=commit or proof.get('synthetic_verify',{}).get('status')!='verified':
        raise ValueError('engineering source or replay identity mismatch')
    if (vram.get('device'),vram.get('catalog'),vram.get('batch'),vram.get('queries'),vram.get('history'))!=('cuda',120415,512,2520,50) or vram.get('steps',0)<3:
        raise ValueError('fullshape CUDA proof required')
    if len(rows)!=1 or rows[0].get('finite_loss') is not True or type(rows[0].get('declared_vram_mib')) is not int or rows[0]['declared_vram_mib']<1024:
        raise ValueError('finite measured VRAM declaration required')
    if (rows[0].get('catalog_items_exported'),rows[0].get('users_exported'),rows[0].get('history_width'))!=(120415,2520,50):raise ValueError('actual CUDA export shape mismatch')
    from vkgr.seq_combo_stack_memory import validate_measurement
    validate_measurement(rows[0])
    return True

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def new(p,v):
    p=Path(p);temporary=p.with_name('.'+p.name+'.'+str(time.time_ns())+'.tmp')
    try:
        with temporary.open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.link(temporary,p)
    finally:
        if temporary.exists():temporary.unlink()
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
    with (q/'.seq_combo_stack-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if exists(q,runs,job['run_id']):raise FileExistsError(job['run_id'])
        name=f"{time.time_ns():020d}-{job['run_id']}.json";tmp=q/'pending'/('.'+name+'.tmp')
        new(tmp,job)
        try:os.link(tmp,q/'pending'/name)
        finally:tmp.unlink()
        return name

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
    import seq_combo_stack_freeze as freeze
    freeze.verify(DEP,load(DEP/'configs/seq_combo_stack/freeze-v1.json'))
    import vkgr.retrieval_seq_combo_stack_run as run
    return run

def identity():
    m=load(OPS/'deployment.json')
    if subprocess.check_output(['git','-C',str(DEP),'rev-parse','HEAD'],text=True).strip()!=m['commit']:raise ValueError('deployed commit changed')
    for p,pin in m['files'].items():
        if sha(DEP/p)!=pin:raise ValueError('deployed file changed: '+p)
    if sha(P3C)!=P3C_SHA:raise ValueError('private data-channel contract hash changed')
    if sha(INPUT)!=m['inputs_sha256']:raise ValueError('input spec changed')
    if load(OPS/'parameters.json')!=PARAMETERS or sha(OPS/'parameters.json')!=m['parameters_sha256']:raise ValueError('private parameters changed')
    return m

def inventory():
    return {'read_at_utc':now(),'new_submit':False,'test_read':False,'queue':{r:where(Q,r) for rid in (*RIDS,SMOKE) for r in (rid,rid+'-verify')},'existing_runs':[r for rid in (*RIDS,SMOKE) for r in (rid,rid+'-verify') if (R/'runs'/r).exists()],
            'inputs_path':str(INPUT),'inputs_sha256':sha(INPUT),'stats_manifest_sha256':load(INPUT)['stats_sha256']}

def deploy(req):
    if DEP.exists() or OPS.exists():raise FileExistsError('new deployment/ops root already exists')
    if sha(INPUT)!=req['inputs_sha256']:raise ValueError('input spec differs from read-only inventory')
    if any(exists(Q,R/'runs',r) for rid in (*RIDS,SMOKE) for r in (rid,rid+'-verify')):raise FileExistsError('batch already exists')
    subprocess.run(['git','-C',str(SRC),'fetch','-q','origin',req['branch']],check=True)
    if subprocess.check_output(['git','-C',str(SRC),'rev-parse','origin/'+req['branch']],text=True).strip()!=req['commit']:raise ValueError('remote target mismatch')
    subprocess.run(['git','-C',str(SRC),'worktree','add','-q','--detach',str(DEP),req['commit']],check=True)
    for p,pin in req['files'].items():
        if sha(DEP/p)!=pin:raise ValueError('per-file deployment mismatch: '+p)
    if sha(P3C)!=P3C_SHA:raise ValueError('p3c data-channel source mismatch')
    OPS.mkdir();new(OPS/'parameters.json',PARAMETERS);meta={'commit':req['commit'],'tree':subprocess.check_output(['git','-C',str(DEP),'rev-parse','HEAD^{tree}'],text=True).strip(),'files':req['files'],'inputs_sha256':sha(INPUT),'parameters_sha256':sha(OPS/'parameters.json'),'created_at_utc':now()};new(OPS/'deployment.json',meta)
    return {'status':'deployed','files_checked':len(req['files']),'commit':meta['commit'],'inputs_sha256':meta['inputs_sha256'],'ops_root':str(OPS)}

def preflight():
    meta=identity();run=modules();spec=run.validate_inputs(load(INPUT));entries={};source=run.b1.source_identity()
    if source['git_dirty'] or source['git_commit']!=meta['commit']:raise ValueError('clean fixed source required')
    for seed,rid in zip(SEEDS,RIDS):
        cp=DEP/f'configs/seq_combo_stack/{rid}.json';c=load(cp);run.validate_config(c)
        data,inputs=run.load_real(spec,c,meta['inputs_sha256'],parameters_sha=sha(OPS/'parameters.json'))
        pf=run.preflight_check(data,c,inputs,engineering=False,config_sha256=sha(cp))
        if pf['status']!='preflight_ok':return {'status':'preflight_failed','source':source,'failed_run':rid,'leakage_receipt':pf['leakage_receipt'],'new_submit':False}
        rows,queries=run.fr._view(data,50);model=run._model(data,c,'cpu');expected=run.binding_for4(data,c,rows,queries,model,'cuda',inputs)
        entries[rid]={**pf,'config_sha256':sha(cp),'expected_binding_sha256':run.b1._digest(expected),
                      'data_hashes':{k:expected[k] for k in ('rows_sha256','queries_sha256','catalog_sha256','support_sha256','content_sha256','side_sha256','weeks_sha256','stats_sha256','stat_bucket_ids_sha256')}}
    return {'status':'preflight_ok','read_at_utc':now(),'source':source,'commit':meta['commit'],'inputs_sha256':meta['inputs_sha256'],'entries':entries,'test_read':False,'new_submit':False}

def template():
    d=load(TEMPLATE)
    if d['cpu_cores']!=1 or d['ram_mib']!=9216 or d['no_admission'] is not False:raise ValueError('frozen P template changed')
    return d

def train_jobs(req):
    check_combo_review(REVIEW,req['combo_review_sha256'])
    meta=identity();pf=req['preflight'];memory=req['engineering']
    if pf['status']!='preflight_ok' or pf['commit']!=meta['commit'] or pf['inputs_sha256']!=meta['inputs_sha256']:raise ValueError('preflight identity mismatch')
    from vkgr.retrieval_b1_run import source_identity
    if pf.get('source')!=source_identity() or pf['source'].get('git_dirty') is not False:raise ValueError('preflight source changed')
    term=load(R/'runs'/SMOKE/'terminal.json');proof=load(R/'runs'/SMOKE/'out/engineering.json')
    if term['status']!='succeeded' or memory!=proof:raise ValueError('CUDA engineering evidence mismatch')
    check_engineering(proof,meta['commit'])
    vram=proof['vram']['variants'][0]['declared_vram_mib'];parent=template();jobs=[]
    for seed,rid in zip(SEEDS,RIDS):
        cp=DEP/f'configs/seq_combo_stack/{rid}.json';entry=pf['entries'][rid]
        if entry.get('engineering') is not False:raise ValueError('real preflight required')
        check_leakage_receipt(entry.get('leakage_receipt',{}),entry,pf['source'])
        if sha(cp)!=entry['config_sha256']:raise ValueError('preflight config mismatch')
        d=copy.deepcopy(parent);d.update(run_id=rid,seed=seed,cwd=str(DEP),config=str(cp),pinned_manifest=None,vram_mib=vram)
        a=d['argv'];a[0]=PY;a[a.index('-m')+1]='vkgr.retrieval_seq_combo_stack_run';replace(a,'--inputs',str(INPUT));replace(a,'--inputs-sha',meta['inputs_sha256']);replace(a,'--config',str(cp.relative_to(DEP)));replace(a,'--config-sha',sha(cp));replace(a,'--output',str(R/'runs'/rid/'out'));a.extend(['--data-parameters',str(OPS/'parameters.json'),'--data-parameters-sha',sha(OPS/'parameters.json')])
        d['inputs']=list(dict.fromkeys([*d['inputs'],str(INPUT),str(OPS/'parameters.json')]));jobs.append({'job':d,'expected':entry})
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

def prepare_job(job):
    sys.path.insert(0,str(LIB));from vkgr_ops import launcher
    return launcher.prepare(job)

def bound_published(job):
    records=[load(Q/k/name) for k,files in where(Q,job['run_id']).items() if k in ('pending','claimed','rejected') for name in files]
    return len(records)==1 and all(records[0].get(k)==v for k,v in job.items())

def publication_alert(value):
    payload={'status':'ALERT','utc':now(),'remaining':'not attempted; no automatic retry',**value}
    try:
        if not (OPS/'publication-ALERT.json').exists():new(OPS/'publication-ALERT.json',payload)
    except Exception as exc:
        print(json.dumps({**payload,'alert_write_error':type(exc).__name__+': '+str(exc)}),file=sys.stderr,flush=True)

def publish_intents(plan):
    if (OPS/'publication-final.json').exists():raise FileExistsError('publication already attempted')
    results={};stopped=False
    for index,entry in enumerate(plan['entries']):
        job=entry['job'];result={'run_id':job['run_id'],'job_sha256':digest(job),'status':'not_attempted'}
        if not stopped:
            try:
                new(OPS/f'publication-intent-{index:02d}.json',{**result,'status':'attempting','utc':now()})
                prepare_job(job);name=publish(Q,R/'runs',job);result.update(status='published',queue_file=name)
            except Exception as exc:
                stopped=True
                try:observed=bound_published(job)
                except Exception:observed=False
                result.update(status='published' if observed else 'failed',error=type(exc).__name__+': '+str(exc),queue_visible=observed)
                publication_alert({'failed_intent':result})
        try:new(OPS/f'publication-result-{index:02d}.json',result)
        except Exception as exc:
            stopped=True;result['bookkeeping_error']=type(exc).__name__+': '+str(exc);publication_alert({'failed_receipt':result})
        results[job['run_id']]=result
    try:new(OPS/'publication-final.json',{'utc':now(),'status':'ALERT' if stopped else 'published_not_accepted','runs':list(results.values())})
    except Exception as exc:
        stopped=True;publication_alert({'final_receipt_error':type(exc).__name__+': '+str(exc)})
    return results,stopped

def ready_handshake(plan):
    if os.getsid(os.getpid())!=os.getpid():raise ValueError('detached session owner required before publication')
    new(OPS/'driver-ready.json',{'pid':os.getpid(),'ppid':os.getppid(),'plan_sha256':digest(plan),'deadline_epoch':plan['deadline_epoch'],'utc':now()})
    until=min(plan['deadline_epoch'],time.time()+30)
    while time.time()<until:
        if (OPS/'publish-release.json').exists():
            expected={'plan_sha256':digest(plan),'ready_sha256':sha(OPS/'driver-ready.json'),'driver_pid':os.getpid()}
            if load(OPS/'publish-release.json')!=expected:raise ValueError('ready/release identity mismatch')
            return
        time.sleep(.05)
    raise TimeoutError('controller never confirmed ready; zero publication attempts')

def observation_cycle(plan,publications):
    states=[];done=True
    for entry in plan['entries']:
        rid=entry['job']['run_id']
        if publications[rid]['status']!='published':
            states.append({'run_id':rid,'verified':False,'unaccepted':True,'publication_status':publications[rid]['status'],'exceptions':['unpublished intent; no retry']});continue
        try:
            state=snapshot(entry)
            if state['train_status']=='succeeded' and not state['rejected'] and not exists(Q,R/'runs',rid+'-verify'):
                intent=OPS/('verify-intent-'+rid+'.json')
                if intent.exists():raise ValueError('previous verifier intent was not published; no retry')
                job=derive_verify(entry['job'],state['manifest_sha256']);new(intent,{'job_sha256':digest(job),'utc':now()})
                try:prepare_job(job);publish(Q,R/'runs',job)
                except Exception:
                    if not bound_published(job):raise
                state=snapshot(entry)
            settled=state['rejected'] or (state['train_status'] is not None and state['train_status']!='succeeded') or (state['train_status']=='succeeded' and state['verify_status'] is not None)
            done=done and settled
        except Exception as exc:
            state={'run_id':rid,'verified':False,'unaccepted':True,'exceptions':[type(exc).__name__+': '+str(exc)]}
        states.append(state)
    return states,done

def owner_source(plan):
    meta=identity()
    from vkgr.retrieval_b1_run import source_identity
    current=source_identity()
    if current!=plan['source'] or current['git_dirty'] or meta['commit']!=current['git_commit'] or sha(OPS/'parameters.json')!=plan['parameters_sha256']:
        raise ValueError('sealed owner source or private parameters changed')

def driver():
    os.chdir(DEP);sys.path.insert(0,str(DEP/'src'));plan=load(OPS/'plan.json')
    if sha(OPS/'plan.json')!=sys.argv[sys.argv.index('--plan-sha')+1]:raise ValueError('sealed plan changed')
    with (OPS/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        new(OPS/'driver-pid.json',{'pid':os.getpid(),'ppid':os.getppid(),'poll_seconds':900,'deadline_epoch':plan['deadline_epoch'],'started_at_utc':now()})
        try:owner_source(plan);ready_handshake(plan);owner_source(plan)
        except Exception as exc:
            new(OPS/'summary-final.json',{'status':'ALERT','utc':now(),'reason':'startup: '+type(exc).__name__+': '+str(exc),'published_intents':0});return
        publications,failed=publish_intents(plan)
        states=deadline_fallback(plan['entries'])
        while time.time()<plan['deadline_epoch']:
            try:owner_source(plan)
            except Exception as exc:
                new(OPS/'summary-final.json',{'status':'ALERT','utc':now(),'reason':'source: '+type(exc).__name__+': '+str(exc)});return
            states,done=observation_cycle(plan,publications)
            status='ALERT' if failed else summary_status(states,done)
            value={'utc':now(),'runs':states,'all_terminal':done,'status':status,'publication_alert':failed}
            new(OPS/f'summary-{time.time_ns()}.json',value)
            if done:new(OPS/'summary-final.json',value);return
            time.sleep(min(900,max(0,plan['deadline_epoch']-time.time())))
        new(OPS/'summary-deadline.json',{'utc':now(),'status':'ALERT','reason':'original 48h deadline','runs':states,'publication_alert':failed})

def start_submission(plan):
    new(OPS/'plan.json',plan)
    proc=None
    try:
        with (OPS/'driver-launcher.log').open('xb') as log:
            proc=subprocess.Popen([PY,'-B',str(DEP/'scripts/seq_combo_stack_ops.py'),'--driver','--params',str(OPS/'parameters.json'),'--plan-sha',sha(OPS/'plan.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=DEP,env={**os.environ,'PYTHONPATH':str(DEP/'src'),'OMP_NUM_THREADS':'1'})
        until=min(plan['deadline_epoch'],time.time()+30)
        while time.time()<until:
            if proc.poll() is not None:raise RuntimeError('owner exited before ready')
            if (OPS/'driver-ready.json').exists():
                ready=load(OPS/'driver-ready.json')
                if ready['pid']!=proc.pid or ready['plan_sha256']!=digest(plan) or os.getsid(proc.pid)!=proc.pid:raise ValueError('detached owner ready identity mismatch')
                new(OPS/'publish-release.json',{'plan_sha256':digest(plan),'ready_sha256':sha(OPS/'driver-ready.json'),'driver_pid':proc.pid})
                return {'status':'driver_released_not_accepted','driver_launcher_pid':proc.pid,'driver_ready':ready,'plan_sha256':digest(plan),'train_intents':len(plan['entries'])}
            time.sleep(.05)
        raise TimeoutError('detached owner readiness deadline')
    except Exception as exc:
        alert={'utc':now(),'status':'ALERT','reason':'controller startup: '+type(exc).__name__+': '+str(exc),'controller_publications':0}
        new(OPS/'controller-ALERT.json',alert)
        if proc is None:new(OPS/'summary-final.json',alert)
        raise

def read_status():
    answer=inventory();answer['read_at_utc']=now()
    for p in ('driver-pid.json','driver-ready.json','publish-release.json','publication-ALERT.json','publication-final.json','controller-ALERT.json','summary-final.json','summary-deadline.json'):
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
    answer['status']='ALERT' if any((OPS/name).exists() for name in ('publication-ALERT.json','controller-ALERT.json','summary-deadline.json')) else answer.get('summary-final.json',{}).get('status','observing')
    return answer

def main():
    if '--driver' in sys.argv:
        configure(load(sys.argv[sys.argv.index('--params')+1]));driver();return
    raw=sys.stdin.read();req=json.loads(raw);configure(req['paths']);act=req['action']
    sys.path.insert(0,str(LIB))
    if act=='inventory':answer=inventory()
    elif act=='deploy':answer=deploy(req)
    elif act=='preflight':answer=preflight()
    elif act in ('smoke-dry','smoke-submit'):
        identity();d=template();d.update(run_id=SMOKE,cwd=str(DEP),config=str(DEP/f'configs/seq_combo_stack/{RIDS[0]}.json'),inputs=[],disk_gib=1,wall_hours=1.,vram_mib=2048,argv=[PY,'-B','scripts/seq_combo_stack_cuda_check.py',str(R/'runs'/SMOKE/'out')])
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
            created=time.time()
            plan={'created_at_utc':now(),'deadline_epoch':created+48*3600,'entries':entries,'source':req['preflight']['source'],
                  'preflight_sha256':digest(req['preflight']),'engineering_sha256':digest(req['engineering']),
                  'combo_review_sha256':req['combo_review_sha256'],'parameters_sha256':sha(OPS/'parameters.json')}
            answer=start_submission(plan)
    elif act=='start':
        raise ValueError('submit seals plan and starts its sole owner; separate start is forbidden')
    elif act=='status':answer=read_status()
    else:raise ValueError('unregistered action')
    print(json.dumps(answer,sort_keys=True,allow_nan=False))
    if answer.get('status')=='preflight_failed':raise SystemExit(2)
if __name__=='__main__':main()
