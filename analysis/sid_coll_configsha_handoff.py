"""New-only, pinned SID-COLL config-identity recovery; metadata and Git blobs only.

inventory is read-only; prepare seals a new plan/control directory; start detaches
one owner. The owner holds old/global/new locks before ACK and uses the unchanged
controller tick/EventWaiter. No model import, scheduler implementation, artifact
reader, score reconstruction, retry, deadline reset, or caller environment.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SCHEMA = 'vkgr-sid-coll-configsha-handoff-v1'
SMALL = 48 * 1024
LARGE = 2 * 1024 * 1024
REQUIRED = {'schema','old_plan','old_control_dir','new_plan','new_control_dir',
            'old_owner','old_pid','created_epoch','deadline_epoch','scientific_commit',
            'git_repo','model_root','source_files','python','ops_library','ops_files',
            'run_ops_source','approval','helper_root','helper_commit','helper_source_sha256'}
STAGES = ('input_preflight','input','input_verify','sid_fit','sid_verify',
          'preflight','cuda','dry','train42','verify42','train2024','verify2024',
          'train2025','verify2025')
PATCHED = tuple(g+str(s) for g in ('train','verify') for s in (42,2024,2025))


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encoded(value): return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value): return hashlib.sha256(encoded(value)).hexdigest()
def sha(raw): return hashlib.sha256(raw).hexdigest()

def hexpin(value,n=64):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{'+str(n)+'}',value):
        raise ValueError('invalid SHA pin')
    return value


def safe(path):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p,*p.parents)):
        raise ValueError('absolute nonsymlink path required')
    if re.search(r'week[_-]?26|/test/|\.(pt|pth|ckpt|npy|npz|parquet|csv)$',str(p).lower()):
        raise ValueError('checkpoint/data/test path prohibited')
    return p


def relative(value):
    p=Path(value)
    if not value or p.is_absolute() or '..' in p.parts or p.as_posix()!=value:
        raise ValueError('safe relative source path required')
    if p.suffix not in ('.py','.json','.yaml','.yml','.md','.toml','.txt','.gitignore') and p.name not in ('.gitignore','.python-version','LICENSE'):
        raise ValueError('source/config text files only')
    return p


def bounded_bytes(path,limit=SMALL):
    p=safe(path)
    if not p.is_file() or p.stat().st_size>=limit: raise ValueError('bounded metadata/source file required: '+str(p))
    raw=p.read_bytes()
    if len(raw)>=limit: raise ValueError('file grew beyond bound')
    return raw


def unique(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise ValueError('duplicate JSON key')
        result[k]=v
    return result


def read(path,pin=None,limit=SMALL):
    p=safe(path)
    if p.suffix not in ('.json','.reason'): raise ValueError('small JSON/reason required')
    raw=bounded_bytes(p,limit)
    if pin is not None and sha(raw)!=hexpin(pin): raise ValueError('metadata SHA mismatch: '+str(p))
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('finite JSON required'))),sha(raw)


def ref(value,limit=SMALL):
    if type(value) is not dict or set(value)!={'path','sha256'}: raise ValueError('exact path/SHA reference required')
    return read(value['path'],value['sha256'],limit)[0]


def once(path,raw):
    p=safe(path)
    if len(raw)>=LARGE: raise ValueError('oversized output')
    with p.open('xb') as f: f.write(raw);f.flush();os.fsync(f.fileno())


def at(value,key):
    for part in key.split('.'):
        if not isinstance(value,dict) or part not in value: raise ValueError('missing receipt identity field '+key)
        value=value[part]
    return value


def resolve(value,accepted):
    if isinstance(value,dict): return {k:resolve(v,accepted) for k,v in value.items()}
    if isinstance(value,list): return [resolve(v,accepted) for v in value]
    if isinstance(value,str) and value.startswith('@'):
        stage,sep,key=value[1:].partition('.')
        if not sep or stage not in accepted: raise ValueError('unresolved prior receipt identity')
        if key.startswith('summary.') or (key.startswith('raw_proof.') and key!='raw_proof.declared_vram_mib'):
            raise ValueError('inventory cannot read raw scores')
        return copy.deepcopy(at(accepted[stage],key))
    return value


def alive(pid):
    try: os.kill(pid,0);return True
    except ProcessLookupError: return False
    except PermissionError: return True


def acquire_existing(path):
    # rb + flock changes no old file bytes and refuses missing locks.
    handle=safe(path).open('rb')
    try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close();raise ValueError('old owner lock still held') from None
    return handle


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(safe(repo)),*args],stderr=subprocess.PIPE)


def source_check(req):
    commit=hexpin(req['scientific_commit'],40)
    if git(req['git_repo'],'rev-parse',commit+'^{commit}').decode().strip()!=commit: raise ValueError('scientific Git commit missing')
    pins=req['source_files']
    if type(pins) is not dict or not {'scripts/sid_coll_ops.py','scripts/sid_coll_job_entry.py'}<=set(pins): raise ValueError('scientific source pins incomplete')
    for name,pin in pins.items():
        rel=relative(name);hexpin(pin)
        blob=git(req['git_repo'],'show',commit+':'+str(rel))
        if len(blob)>=1024*1024 or sha(blob)!=pin or sha(bounded_bytes(safe(req['model_root'])/rel,1024*1024))!=pin:
            raise ValueError('scientific source/Git pin mismatch: '+name)
    helper=safe(req['helper_root']);helper_commit=hexpin(req['helper_commit'],40)
    if git(helper,'rev-parse','HEAD').decode().strip()!=helper_commit or git(helper,'status','--porcelain').strip():
        raise ValueError('helper Git HEAD/clean source mismatch')
    tool=helper/'analysis/sid_coll_configsha_handoff.py'
    helper_pin=hexpin(req['helper_source_sha256'])
    if sha(bounded_bytes(tool,LARGE))!=helper_pin or sha(git(helper,'show',helper_commit+':analysis/sid_coll_configsha_handoff.py'))!=helper_pin:
        raise ValueError('helper source/Git pin mismatch')
    if type(req['ops_files']) is not dict or 'vkgr_ops/__init__.py' not in req['ops_files']: raise ValueError('ops source pins required')
    actual={p.relative_to(safe(req['ops_library'])).as_posix() for p in (safe(req['ops_library'])/'vkgr_ops').glob('*.py')}
    if set(req['ops_files'])!=actual: raise ValueError('complete ops Python source pins required')
    for name,pin in req['ops_files'].items():
        if not name.startswith('vkgr_ops/') or sha(bounded_bytes(safe(req['ops_library'])/relative(name),1024*1024))!=hexpin(pin): raise ValueError('ops source pin mismatch')
    identity=req['run_ops_source']
    if type(identity) is not dict or set(identity)!={'commit','sha256'}:raise ValueError('ops source identity schema')
    hexpin(identity['commit'],40)
    aggregate=sha((json.dumps(req['ops_files'],sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode())
    if aggregate!=hexpin(identity['sha256']):raise ValueError('ops source aggregate differs from frozen run source')


def validate_request(req):
    if type(req) is not dict or set(req)!=REQUIRED or req['schema']!=SCHEMA: raise ValueError('exact recovery request schema required; no caller env')
    if req['python']!=sys.executable: raise ValueError('approved Python must equal current sys.executable')
    if type(req['old_pid']) is not int or req['old_pid']<=1: raise ValueError('old PID required')
    if any(type(req[k]) not in (int,float) or not math.isfinite(req[k]) for k in ('created_epoch','deadline_epoch')): raise ValueError('finite deadline required')
    for key in ('old_control_dir','new_control_dir','new_plan','git_repo','model_root','ops_library'): safe(req[key])
    old=safe(req['old_control_dir']);new=safe(req['new_control_dir'])
    if new==old or new.parent!=old.parent or safe(req['new_plan']).parent!=old.parent: raise ValueError('new-only sibling recovery paths required')
    owner=ref(req['old_owner'])
    if Path(req['old_owner']['path']).parent!=old or owner.get('pid')!=req['old_pid'] or owner.get('ppid')!=1 or owner.get('poll_seconds')!=900 or owner.get('deadline_epoch')!=req['deadline_epoch']:
        raise ValueError('old owner identity/deadline mismatch')
    approval=ref(req['approval'])
    if (type(approval) is not dict or set(approval)!={'source','question_item_id','answer','scope','not_yet_executed'}
        or approval['source']!='Eddy direct reply in this conversation' or approval['answer']!='批准按提案恢复'
        or type(approval['question_item_id']) is not str or not approval['question_item_id']
        or type(approval['scope']) is not list or not approval['scope'] or any(type(x) is not str or not x for x in approval['scope'])
        or approval['not_yet_executed'] is not True):
        raise ValueError('pinned human direct approval schema/answer required')
    plan=ref(req['old_plan'],LARGE)
    if plan['control_dir']!=str(old) or plan['created_epoch']!=req['created_epoch'] or plan['deadline_epoch']!=req['deadline_epoch'] or plan['deadline_epoch']-plan['created_epoch']!=172800:
        raise ValueError('original control/created/deadline must not change')
    if time.time()>=req['deadline_epoch']: raise ValueError('original deadline expired')
    if set(plan['branches'])!={'C1p','C2'} or plan.get('schema')!='vkgr-sid-coll-plan-v1': raise ValueError('original plan schema/branches')
    for branch in plan['branches'].values():
        if set(branch['stages'])!=set(STAGES): raise ValueError('original stages incomplete')
    source_check(req)
    return plan


def frozen_spec(stage):
    argv=stage['job']['argv']
    if argv.count('--spec')!=1 or argv.count('--spec-sha')!=1: raise ValueError('frozen spec flags required')
    p=argv[argv.index('--spec')+1];pin=argv[argv.index('--spec-sha')+1]
    if p!=stage['job']['config'] or pin!=stage['proof']['expected']['binding.spec_sha256']: raise ValueError('spec/job/expected pin mismatch')
    return ref({'path':p,'sha256':pin})


def patch_plan(req,plan):
    new=copy.deepcopy(plan);new['control_dir']=req['new_control_dir'];patches=[]
    for label,branch in new['branches'].items():
        for name in PATCHED:
            stage=branch['stages'][name];expected=stage['proof']['expected'];spec=frozen_spec(stage)
            generic='train' if name.startswith('train') else 'verify';seed=int(name[len(generic):])
            if set(spec['stages'])!={generic} or spec['stages'][generic]['parameters']['seed']!=seed or stage['job']['seed']!=seed: raise ValueError('frozen stage seed mismatch')
            if spec['commit']!=req['scientific_commit'] or spec['source_files']!=req['source_files'] or stage['job']['cwd']!=req['model_root'] or expected['binding.commit']!=req['scientific_commit']:
                raise ValueError('scientific source identity mismatch')
            if spec['stages'][generic]['run_id']!=stage['job']['run_id'] or expected['run_id']!=stage['job']['run_id']: raise ValueError('spec run identity mismatch')
            config=spec['refs']['config'+str(seed)]
            rel=safe(config['path']).relative_to(safe(req['model_root'])).as_posix()
            pin=sha(git(req['git_repo'],'show',req['scientific_commit']+':'+str(relative(rel))))
            if pin!=config['sha256'] or pin!=sha(bounded_bytes(config['path'])) or pin!=expected['binding.data']['config'+str(seed)]: raise ValueError('frozen config Git/spec/data pin mismatch')
            if expected.get('binding.config_sha256','missing') is not None: raise ValueError('only original null config pins may be patched')
            expected['binding.config_sha256']=pin
            patches.append({'path':['branches',label,'stages',name,'proof','expected','binding.config_sha256'],'before':None,'after':pin})
    if len(patches)!=12: raise ValueError('exactly twelve config SHA patches required')
    return new,patches


def manifest_check(path,job,req):
    run,pin=read(path);expected_commit=None
    # The original frozen job spec supplies the scientific commit, including reused v1 stages.
    spec=read(job['config'])[0];expected_commit=spec['commit']
    expected={k:job[k] for k in ('run_id','kind','cwd','argv','seed','stop_mode')}
    expected['schema']='vkgr-run-v1'
    if any(run.get(k)!=v for k,v in expected.items()): raise ValueError('run manifest frozen job identity mismatch')
    if run.get('git',{}).get('commit')!=expected_commit or run['git'].get('dirty') is not False or run['git'].get('dirty_files')!=[]: raise ValueError('run manifest clean Git identity mismatch')
    if run.get('ops_source')!=req['run_ops_source']: raise ValueError('run manifest ops source mismatch')
    if run.get('overlay') is not None or run.get('pinned_files')!={}: raise ValueError('run manifest unexpected overlay')
    env={}
    for item in job['env']:
        key,sep,value=item.partition('=')
        if not sep or key in env: raise ValueError('frozen job environment invalid')
        env[key]=value
    if any(run.get('env',{}).get(k)!=v for k,v in env.items()): raise ValueError('run manifest environment mismatch')
    resources={k:job[k] for k in ('cpu_cores','ram_mib','vram_mib','disk_gib')};resources['wall_seconds']=job['wall_hours']*3600
    if run.get('resources')!=resources: raise ValueError('run manifest resources mismatch')
    config_sha=sha(bounded_bytes(job['config'],SMALL))
    if run.get('config')!={'path':job['config'],'sha256':config_sha}: raise ValueError('run manifest config mismatch')
    expected_inputs=[]
    for p in job['inputs']:
        # Only frozen small ops documents; never data refs in the scientific spec.
        if p!=job['config'] and (Path(p).parent!=Path(job['config']).parent or Path(p).name!='deployment.json'):
            raise ValueError('manifest inputs must be frozen spec/deployment metadata only')
        raw=bounded_bytes(p,LARGE)
        if Path(p).suffix!='.json': raise ValueError('manifest inputs must be small ops JSON')
        expected_inputs.append({'path':p,'sha256':sha(raw),'bytes':len(raw)})
    if run.get('inputs')!=expected_inputs: raise ValueError('run manifest input metadata mismatch')
    return {'path':str(path),'sha256':pin}


def intent_files(plan):
    ctl=safe(plan['control_dir']);result={}
    ids={s['job']['run_id'] for b in plan['branches'].values() for s in b['stages'].values()}
    for path in sorted(ctl.glob('intent-*.json')):
        value,pin=read(path)
        if set(value)!={'run_id','job_sha256','created_utc'} or path.name!='intent-'+value['run_id']+'.json' or value['run_id'] not in ids:
            raise ValueError('unrecognized old intent protection')
        hexpin(value['job_sha256']);result[value['run_id']]={'path':str(path),'sha256':pin}
    return result


def inspect_runs(req,old,new):
    root=safe(old['root']);intents=intent_files(old);runs={};guards={}
    for label,branch in new['branches'].items():
        accepted={}
        for name in STAGES:
            stage=branch['stages'][name];rid=stage['job']['run_id'];rd=root/'runs'/rid
            if not re.fullmatch('[a-z0-9][a-z0-9_-]{0,95}',rid): raise ValueError('unsafe run ID')
            if safe(stage['proof']['path'])!=rd/'out/receipt.json': raise ValueError('own receipt path required')
            q={k:sorted((root/'queue'/k).glob('*-'+rid+'.json')) for k in ('pending','claimed','rejected')}
            snap={'status':'absent','queue':{k:[] for k in q},'intent':intents.get(rid)};runs[rid]=snap
            if q['rejected']:
                # Hash a bounded reason, but do not turn a rejection into a retry.
                for p in q['rejected']:
                    read(p)
                    for reason in (p.with_suffix('.reason'),Path(str(p)+'.reason')):
                        if reason.exists():bounded_bytes(reason)
                raise ValueError('rejected job; never resubmit: '+rid)
            if len(q['pending'])+len(q['claimed'])>1: raise ValueError('duplicate own queue identity')
            try: job=resolve(stage['job'],accepted)
            except ValueError:
                if any(q.values()) or rd.exists() or rid in intents: raise ValueError('existing attempt has unresolved frozen identity: '+rid) from None
                snap['status']='absent_deferred';continue
            frozen_spec(stage)
            for kind in ('pending','claimed'):
                for path in q[kind]:
                    queued,pin=read(path)
                    if queued!=job: raise ValueError('queue job differs from frozen identity: '+rid)
                    snap['queue'][kind].append({'path':str(path),'sha256':pin})
            if rid in intents:
                guard=ref(intents[rid])
                if guard['job_sha256']!=digest(job): raise ValueError('old intent job identity mismatch')
                guards[Path(intents[rid]['path']).name]=intents[rid]
                if not any(q.values()): raise ValueError('uncertain intent without queue: '+rid)
            terminal=rd/'terminal.json'
            if rd.exists():
                if len(q['claimed'])!=1: raise ValueError('run manifest lacks unique claimed job: '+rid)
                snap['manifest']=manifest_check(rd/'run.json',job,req)
                if terminal.exists():
                    value,pin=read(terminal)
                    if value.get('run_id')!=rid or value.get('status')!='succeeded' or value.get('exit_code')!=0: raise ValueError('failed existing job; never retry: '+rid)
                    snap['terminal']={'path':str(terminal),'sha256':pin}
                    proof,pin=read(stage['proof']['path']);expected=resolve(stage['proof']['expected'],accepted)
                    if any(at(proof,k)!=v for k,v in expected.items()): raise ValueError('receipt frozen identity mismatch: '+rid)
                    if name=='train42':
                        original=resolve(old['branches'][label]['stages'][name]['proof']['expected'],accepted)
                        different=[k for k,v in original.items() if at(proof,k)!=v]
                        if len(original)!=9 or different!=['binding.config_sha256']: raise ValueError('train42 must have only the null config receipt difference')
                    snap['receipt']={'path':stage['proof']['path'],'sha256':pin};snap['status']='accepted'
                    accepted[name]={**proof,'receipt_sha256':pin}
                else: snap['status']='running'
            elif any(q.values()):
                if q['claimed']: raise ValueError('uncertain claimed job without run manifest: '+rid)
                snap['status']='queued'
            if name in STAGES[:9] and snap['status']!='accepted': raise ValueError('existing base/train42 not accepted; no retrain: '+rid)
    if len(runs)!=28: raise ValueError('globally unique original run IDs required')
    return runs,guards


def inventory_locked(req,old):
    new,patches=patch_plan(req,old)
    for item in old.get('common_refs',[]): ref(item,LARGE)
    runs,guards=inspect_runs(req,old,new)
    stops={}
    ctl=safe(req['old_control_dir'])
    for path in sorted(ctl.glob('alert-*.json'))+([ctl/'summary-final.json'] if (ctl/'summary-final.json').exists() else []):
        _,pin=read(path,limit=LARGE);stops[path.name]={'path':str(path),'sha256':pin}
    if not {'alert-C1p.json','alert-C2.json'}<=set(stops): raise ValueError('original branch STOP records required')
    return {'schema':'vkgr-sid-coll-configsha-inventory-v1','status':'inventory_ok','new_plan':new,
            'new_plan_sha256':sha(encoded(new)),'patches':patches,'runs':runs,'guards':guards,
            'preserved_stop_refs':stops,'old_owner':req['old_owner'],'old_owner_alive':False,
            'old_lock_available':True,'created_epoch':old['created_epoch'],'deadline_epoch':old['deadline_epoch'],
            'source_plan':req['old_plan'],'approval':req['approval'],'request_sha256':digest(req)}



def inventory(req):
    old=validate_request(req)
    if alive(req['old_pid']): raise ValueError('old owner still alive')
    with acquire_existing(safe(req['old_control_dir'])/'driver.lock'):
        return inventory_locked(req,old)


def global_path(req): return safe(req['old_control_dir']).parent/'configsha-handoff.lock'

def global_lock(req,wait=0):
    p=global_path(req)
    fd=p.open('a+b')
    deadline=time.monotonic()+wait
    while True:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return fd
        except BlockingIOError:
            if time.monotonic()>=deadline:
                fd.close();raise ValueError('global handoff owner already held') from None
            time.sleep(.02)


def prepare(req):
    old=validate_request(req)
    if alive(req['old_pid']):raise ValueError('old owner still alive')
    with global_lock(req),acquire_existing(safe(req['old_control_dir'])/'driver.lock'):
        result=inventory_locked(req,old)
        newctl=safe(req['new_control_dir']);newplan=safe(req['new_plan'])
        if newctl.exists() or newplan.exists(): raise FileExistsError('new plan/control already exists; no overwrite or cleanup')
        newctl.mkdir()
        # Partial failure retains every created file for reconciliation; never retry blindly.
        for name,source in result['guards'].items(): once(newctl/name,bounded_bytes(source['path']))
        once(newplan,encoded(result['new_plan']))
        once(newctl/'handoff.json',encoded(result))
        return result


def sealed(req):
    result=read(safe(req['new_control_dir'])/'handoff.json',limit=LARGE)[0]
    if result.get('request_sha256')!=digest(req): raise ValueError('sealed request differs')
    plan=read(req['new_plan'],result['new_plan_sha256'],LARGE)[0]
    if plan!=result['new_plan']: raise ValueError('sealed plan differs')
    for name,source in result['guards'].items():
        read(source['path'],source['sha256'])
        read(safe(req['new_control_dir'])/name,source['sha256'])
    for source in result['preserved_stop_refs'].values(): ref(source,LARGE)
    if set(p.name for p in safe(req['new_control_dir']).glob('intent-*.json'))!=set(result['guards']): raise ValueError('new control guards differ before start')
    return result


def owner_env(req):
    return {'PATH':os.defpath,'PYTHONPATH':req['ops_library']+os.pathsep+req['model_root']+'/src'+os.pathsep+req['model_root'],
            'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}


def process_ppid(pid):
    # Linux /proc metadata only; no process cmdline/environment or other jobs.
    text=Path('/proc')/str(pid)/'stat'
    tail=text.read_text().rsplit(')',1)[1].split()
    return int(tail[1])


def held(path):
    with safe(path).open('rb') as fd:
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True


def lock_owned_by(text,info,pid):
    for line in text.splitlines():
        fields=line.split()
        if len(fields)<6 or fields[1:4]!=['FLOCK','ADVISORY','WRITE'] or fields[4]!=str(pid):continue
        device=fields[5].split(':')
        if len(device)==3 and (int(device[0],16),int(device[1],16),int(device[2]))==(os.major(info.st_dev),os.minor(info.st_dev),info.st_ino):return True
    return False


def ack(req):
    seal=read(safe(req['new_control_dir'])/'handoff.json',limit=LARGE)[0]
    value=read(safe(req['new_control_dir'])/'owner-ready.json')[0]
    if value.get('request_sha256')!=digest(req) or value.get('plan_sha256')!=seal['new_plan_sha256'] or value.get('deadline_epoch')!=req['deadline_epoch'] or value.get('poll_seconds')!=900:
        raise ValueError('new owner ACK identity mismatch')
    pid=value['pid']
    if not alive(pid) or process_ppid(pid)!=1 or value.get('ppid')!=1: raise ValueError('new owner not actually detached/alive')
    locks=Path('/proc/locks').read_text()
    for path in (safe(req['old_control_dir'])/'driver.lock',global_path(req),safe(req['new_control_dir'])/'driver.lock'):
        if not held(path) or not lock_owned_by(locks,path.stat(),pid): raise ValueError('new owner ACK lock not held by its actual PID')
    return {**value,'status':'ready_confirmed','observed_at_utc':utc()}


def release(req,ready):
    ctl=safe(req['new_control_dir'])
    sealed_ready,pin=read(ctl/'owner-ready.json')
    if sealed_ready!=ready or ready.get('request_sha256')!=digest(req):raise ValueError('release ready identity mismatch')
    value={'owner_pid':ready['pid'],'ready_sha256':pin,'request_sha256':digest(req),
           'plan_sha256':ready['plan_sha256'],'deadline_epoch':req['deadline_epoch']}
    once(ctl/'owner-release.json',encoded(value))
    return value


def wait_release(req,ready,timeout=30):
    ctl=safe(req['new_control_dir']);end=min(time.time()+timeout,req['deadline_epoch'])
    pin=sha(bounded_bytes(ctl/'owner-ready.json'))
    expected={'owner_pid':os.getpid(),'ready_sha256':pin,'request_sha256':digest(req),
              'plan_sha256':ready['plan_sha256'],'deadline_epoch':req['deadline_epoch']}
    while time.time()<end:
        if (ctl/'owner-release.json').exists():
            if read(ctl/'owner-release.json')[0]!=expected:raise ValueError('owner release identity mismatch')
            return
        time.sleep(.05)
    once(ctl/'launch-aborted.json',encoded({'status':'STOP','reason':'no release; zero new publications','at_utc':utc()}))
    raise ValueError('release absent/expired; no tick or submission occurred')


def owner(req):
    """Called only by detached child; locks stay held through the unchanged tick loop."""
    if os.getppid()!=1: raise ValueError('owner requires actual PPID 1')
    with global_lock(req,wait=10):
        result=sealed(req);fresh=inventory(req)
        if fresh['guards']!=result['guards'] or fresh['new_plan']!=result['new_plan']: raise ValueError('old plan/guards changed since prepare')
        with acquire_existing(safe(req['old_control_dir'])/'driver.lock'):
            ctl=safe(req['new_control_dir'])
            with (ctl/'driver.lock').open('xb') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                source_check(req)
                module_path=safe(req['model_root'])/'scripts/sid_coll_ops.py'
                module_spec=importlib.util.spec_from_file_location('sid_coll_original_configsha_owner',module_path)
                op=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(op)
                plan=op.load_plan(req['new_plan'],result['new_plan_sha256'])
                op.new(ctl/'driver-pid.json',{'pid':os.getpid(),'ppid':os.getppid(),'poll_seconds':900,'deadline_epoch':plan['deadline_epoch'],'started_utc':utc()})
                ready={'pid':os.getpid(),'ppid':os.getppid(),'poll_seconds':900,'deadline_epoch':plan['deadline_epoch'],
                       'plan_sha256':result['new_plan_sha256'],'request_sha256':digest(req),'ready_at_utc':utc(),
                       'protocol':'old_global_new_locks_ready_release_then_original_tick_eventwait'}
                op.new(ctl/'owner-ready.json',ready)
                wait_release(req,ready)
                waiter=op.EventWaiter(plan);wake={'wake':'initial',**waiter.metadata()}
                try:
                    while True:
                        current=op.load_plan(req['new_plan'],result['new_plan_sha256'])
                        snap=op.tick(current);snap['event_wait']=wake
                        op.write_summary(ctl/('summary-'+str(time.time_ns())+'.json'),snap)
                        terminal=all(s['status'] in ('complete','complete_low','ALERT') for s in snap['branches'].values())
                        if terminal or time.time()>=plan['deadline_epoch']:
                            op.write_summary(ctl/'summary-final.json',snap);return
                        wake=waiter.wait(900,deadline_epoch=plan['deadline_epoch'])
                finally:waiter.close()


def start(req,request_path=None,request_pin=None):
    if (safe(req['new_control_dir'])/'launch-intent.json').exists() or (safe(req['new_control_dir'])/'owner-ready.json').exists():
        raise FileExistsError('one owner launch only; never retry')
    # Verify all metadata before forking; repeat in the eventual owner under locks.
    result=sealed(req);fresh=inventory(req)
    if result['new_plan']!=fresh['new_plan'] or result['guards']!=fresh['guards']: raise ValueError('prepare identity changed')
    if request_path is None: raise ValueError('start requires pinned private request file')
    if ref({'path':request_path,'sha256':request_pin},LARGE)!=req: raise ValueError('request file differs')
    with global_lock(req):
        ctl=safe(req['new_control_dir'])
        if any((ctl/name).exists() for name in ('launch-intent.json','owner-ready.json','driver.lock')): raise FileExistsError('one launch attempt only; inspect uncertain owner, never retry')
        once(ctl/'launch-intent.json',encoded({'request_sha256':digest(req),'at_utc':utc(),'deadline_epoch':req['deadline_epoch']}))
        with (ctl/'launcher.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-B',str(safe(req['helper_root'])/'analysis/sid_coll_configsha_handoff.py'),'_detach','--request',request_path,'--request-sha',request_pin],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,env=owner_env(req))
            child.wait(timeout=10)
            if child.returncode!=0: raise ValueError('detach helper failed; no retry')
    end=time.monotonic()+15
    while time.monotonic()<end:
        if (ctl/'owner-ready.json').exists():
            confirmed=ack(req)
            release(req,read(ctl/'owner-ready.json')[0])
            return {**confirmed,'status':'released_independent_owner'}
        time.sleep(.05)
    raise ValueError('owner ACK UNKNOWN after launch; do not retry')


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    request_file=None
    if not argv:
        # bare python -c bootstrap has no reliable __file__; stdin is bounded.
        raw=sys.stdin.read(LARGE+1)
        if len(raw.encode())>=LARGE: raise ValueError('oversized private transport request')
        payload=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('finite JSON required')))
        if set(payload) not in ({'action','request'},{'action','request','request_file'}): raise ValueError('transport action/request fields required')
        action=payload['action'];req=payload['request'];request_file=payload.get('request_file')
        if action not in ('inventory','prepare','start','ack'): raise ValueError('transport action prohibited')
    else:
        parser=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
        parser.add_argument('action',choices=('inventory','prepare','start','ack','_detach'))
        parser.add_argument('--request',required=True);parser.add_argument('--request-sha',required=True)
        args=parser.parse_args(argv);request_file={'path':args.request,'sha256':args.request_sha}
        req=ref(request_file,LARGE);action=args.action
    if action=='_detach':
        launch=read(safe(req['new_control_dir'])/'launch-intent.json')[0]
        if launch.get('request_sha256')!=digest(req) or launch.get('deadline_epoch')!=req['deadline_epoch']:
            raise ValueError('controller launch intent identity required before detach')
        sealed(req)
        if os.fork(): os._exit(0)
        end=time.monotonic()+10
        while os.getppid()!=1 and time.monotonic()<end: time.sleep(.02)
        owner(req);return 0
    function={'inventory':inventory,'prepare':prepare,'ack':ack}.get(action)
    if action=='start':
        if request_file is None: raise ValueError('start requires deployed pinned private request_file')
        result=start(req,request_file['path'],request_file['sha256'])
    else:result=function(req)
    print(json.dumps(result,sort_keys=True,allow_nan=False));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        print(json.dumps({'status':'STOP','error':type(exc).__name__+': '+str(exc)}),file=sys.stderr);raise SystemExit(1)
