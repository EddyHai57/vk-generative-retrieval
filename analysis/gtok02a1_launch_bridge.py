"""Finite clean launcher for frozen G-TOK02 A1 science; default read-only status."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
DEP = Path('/workspace/vkgr-gtok02-a1-v1')
OPS = Path('/workspace/vkgr-gtok02-a1-ops-v1')
CTL = Path('/workspace/vkgr-gtok02-a1-launch-v2')
CODE = CTL/'code'
SOURCE_SHA = '789258e99abaa72c10d8e4aeca190fe8004a9ea05fdf5693565953d6ebc58c8d'
MODEL_SHA = 'c1c4b7413b4ae794894fc78b98828ab26029a487'
PIN_SHA = 'a48995dd9de21cd6e60fb8d9b1fe9327bde03128198a1d13c9444c12f74bf07f'
SMOKE = 'g-gtok02-a1-b1024d3-cuda-v2'

def load_ops():
    path = DEP/'scripts/gtok02a1_ops.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('frozen ops source changed')
    spec = importlib.util.spec_from_file_location('gtok02a1_ops', path)
    ops = importlib.util.module_from_spec(spec)
    sys.modules['gtok02a1_ops'] = ops
    spec.loader.exec_module(ops)
    return ops

def clean_identity(ops):
    meta = ops.load(CTL/'deployment.json')
    target = meta['commit']
    if len(target) != 40 or any(c not in '0123456789abcdef' for c in target):
        raise ValueError('full clean commit required')
    def git(*args, text=True):
        return subprocess.check_output(['git','-C',str(CODE),*args],text=text)
    if git('rev-parse','HEAD').strip() != target or git('status','--porcelain','--untracked-files=all').strip():
        raise ValueError('clean launcher identity drift')
    names = [n for n in git('ls-tree','-r','-z','--name-only',target).split('\0') if n]
    expected = {n:hashlib.sha256(git('show',f'{target}:{n}',text=False)).hexdigest() for n in names}
    if meta['files'] != expected:
        raise ValueError('clean launcher manifest drift')
    for name, pin in expected.items():
        if ops.sha(CODE/name) != pin:
            raise ValueError('clean launcher file changed: '+name)
    return meta

def guard():
    ops = load_ops()
    meta = ops.identity()  # Full original deployment per-file validation, before any science.
    if meta['model_commit'] != MODEL_SHA or meta['pinned_sha256'] != PIN_SHA:
        raise ValueError('frozen science identity drift')
    clean_identity(ops)
    return ops

def route(job):
    d = copy.deepcopy(job)
    d.update(cwd=str(CODE),pinned_manifest=None,pinned_files=[])
    d['argv'] = [d['argv'][0],'-B',str(CODE/'analysis/gtok02a1_job_entry.py'),*d['argv'][1:]]
    if '--config' in d['argv']:
        d['argv'][d['argv'].index('--config')+1] = d['config']
    d['env'] = [e for e in d['env'] if not e.startswith('PYTHONPATH=')]+[f'PYTHONPATH={DEP}/src:{DEP}/scripts']
    d['inputs'] = list(dict.fromkeys([*d['inputs'],str(OPS/'pinned.json'),str(OPS/'deployment.json'),str(CTL/'deployment.json')]))
    return d

def metadata_guard(job, prepared):
    size = len(json.dumps({'spec':job,'prepared':prepared},sort_keys=True,allow_nan=False).encode())
    if size > 48*1024:
        raise ValueError(f'launcher metadata {size} bytes exceeds 48KiB budget')
    return prepared

def probe_job(ops):
    d = ops.derive_job(42)
    d.update(run_id=SMOKE,disk_gib=2,wall_hours=1.,vram_mib=8192,
             argv=[ops.PY,'-B',str(CODE/'analysis/gtok02a1_job_entry.py'),'--probe',str(ops.R/'runs'/SMOKE/'out')])
    return d

def check_claim(actual, expected):
    if any(actual.get(k) != v for k,v in expected.items()):
        raise ValueError('engineering job identity mismatch')

def train_jobs(ops, req):
    meta=ops.identity();pf=ops.load(OPS/'preflight.json')
    proof=ops.load(ops.R/'runs'/SMOKE/'out/engineering.json')
    if pf['status']!='preflight_ok' or pf['model_commit']!=meta['model_commit'] or pf['pinned_sha256']!=meta['pinned_sha256']:
        raise ValueError('preflight identity mismatch')
    if req.get('preflight',pf)!=pf or req.get('engineering',proof)!=proof:
        raise ValueError('request evidence differs from server receipt')
    if ops.load(ops.R/'runs'/SMOKE/'terminal.json')['status']!='succeeded':
        raise ValueError('engineering job not succeeded')
    check_claim(ops.one(SMOKE),probe_job(ops))
    vram=ops.probe_vram(proof);entries=[]
    for seed,rid in zip(ops.SEEDS,ops.RIDS):
        d=ops.derive_job(seed,vram);entry=pf['entries'][rid]
        if ops.sha(d['config'])!=entry['config_sha256']:
            raise ValueError('preflight config mismatch')
        entries.append({'job':d,'expected':entry})
    return entries

def patch(ops):
    derive,prepare=ops.derive_job,ops.prepare_job
    ops.SMOKE=SMOKE
    ops.derive_job=lambda *a,**kw:route(derive(*a,**kw))
    ops.prepare_job=lambda job:metadata_guard(job,prepare(job))
    ops.train_jobs=lambda req:train_jobs(ops,req)
    return ops

def readonly_preflight(ops):
    old=ops.load(OPS/'preflight.json');result=ops.preflight()
    if result['entries']!=old['entries'] or any(result[k]!=old[k] for k in ('model_commit','pinned_sha256','status')):
        raise ValueError('readonly preflight differs from frozen receipt')
    return result

def deploy(req):
    ops=load_ops();meta=ops.identity()
    if meta['model_commit']!=MODEL_SHA or meta['pinned_sha256']!=PIN_SHA:
        raise ValueError('frozen science identity drift')
    if CTL.exists():raise FileExistsError('clean control deployment already exists')
    target=req['commit'];branch=req['branch']
    if len(target)!=40 or any(c not in '0123456789abcdef' for c in target):
        raise ValueError('full target commit required')
    ops.git('fetch','--quiet','origin',f'{branch}:refs/remotes/origin/{branch}')
    if ops.git('rev-parse',f'origin/{branch}').strip()!=target:
        raise ValueError('pushed remote target mismatch')
    names=[n for n in ops.git('ls-tree','-r','-z','--name-only',target).split('\0') if n]
    pins={n:hashlib.sha256(ops.git('show',f'{target}:{n}',text=False)).hexdigest() for n in names}
    for required in ('analysis/gtok02a1_launch_bridge.py','analysis/gtok02a1_job_entry.py'):
        if required not in pins:raise ValueError('launcher missing in target')
    if req.get('files') is not None and any(pins.get(p)!=s for p,s in req['files'].items()):
        raise ValueError('caller launcher file pins mismatch')
    CTL.mkdir();ops.git('worktree','add','--detach',str(CODE),target)
    for name,pin in pins.items():
        if ops.sha(CODE/name)!=pin:raise ValueError('deployed clean file mismatch: '+name)
    result={'commit':target,'files':pins,'created_at_utc':ops.now()}
    ops.new(CTL/'deployment.json',result);clean_identity(ops)
    return {'status':'clean_deployed',**result}

def driver_argv(ops):
    return [ops.PY,'-B',str(CODE/'analysis/gtok02a1_launch_bridge.py'),'--driver']

def start(ops):
    plan=ops.load(OPS/'plan.json')
    if (OPS/'driver-pid.json').exists() or (OPS/'driver-launcher.log').exists():
        raise FileExistsError('driver already started')
    for entry in plan['entries']:
        rid=entry['job']['run_id']
        if not ops.can_start_observation(ops.where(ops.Q,rid),(ops.R/'runs'/rid/'terminal.json').exists()):
            raise ValueError('registered train has no queue or terminal receipt')
    with (OPS/'driver-launcher.log').open('xb') as log:
        p=subprocess.Popen(driver_argv(ops),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
             start_new_session=True,cwd=CODE,env={**os.environ,'PYTHONPATH':f'{DEP}/src:{DEP}/scripts',
             'OMP_NUM_THREADS':'1','VKGR_PINNED_MANIFEST':str(OPS/'pinned.json')})
    return {'status':'started','pid':p.pid}

def action(req):
    act=req.get('action','status')
    if act=='deploy':return deploy(req)
    ops=patch(guard())
    if act=='preflight':return readonly_preflight(ops)
    if act in ('smoke-dry','smoke-submit'):
        d=probe_job(ops)
        if ops.exists(ops.Q,ops.R/'runs',SMOKE):raise FileExistsError(SMOKE)
        result={'status':'smoke_dry_ok','prepared':ops.prepare_job(d),'job':d}
        if act=='smoke-submit':result['written_not_accepted']=ops.publish(ops.Q,ops.R/'runs',d)
        return result
    if act=='start':return start(ops)
    if act in ('status','inventory','dry','submit'):return ops.action({**req,'action':act})
    raise ValueError('unregistered clean launcher action')

def main():
    if sys.argv[1:]==['--driver']:
        patch(guard()).driver();return
    raw=sys.stdin.read()
    print(json.dumps(action(json.loads(raw) if raw.strip() else {}),sort_keys=True,allow_nan=False))
if __name__=='__main__':main()
