"""G3-only local verification waiter; not a server scheduler or release."""
import datetime,json,os,selectors,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vkgr import generative_real as gr,generative_data as gd
from vkgr.generative_pilot import ExecutionPool
from vkgr.data_audit import canonical_bytes,file_sha256
LOG_CAP=32768

def canonical_path(path):
    path=Path(path)
    if '..' in path.parts:raise ValueError('canonical path required; no dotdot aliases')
    return gd._no_symlink(gd._no_symlink(path).resolve(strict=False))

def group_members(group):
    rows=subprocess.run(['ps','-axo','pid=,pgid=,stat='],capture_output=True,text=True,check=True,timeout=2).stdout.splitlines()
    return [int(f[0]) for row in rows if len(f:=row.split())>=3 and int(f[1])==group and not f[2].startswith('Z')]

def wait_verify(argv,cwd,environment,output,deadline):
    """Wait exact child, bounded logs, reject any live owned descendant."""
    output=gd._no_symlink(output);selector=selectors.DefaultSelector();child=None;streams={};start=time.time()
    try:
        for name in ('stdout','stderr'):streams[name]=(output/name).open('xb')
        child=subprocess.Popen(argv,cwd=cwd,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        for name,pipe in (('stdout',child.stdout),('stderr',child.stderr)):
            os.set_blocking(pipe.fileno(),False);selector.register(pipe,selectors.EVENT_READ,name)
        counts={'stdout':0,'stderr':0}
        while selector.get_map() or child.poll() is None:
            if time.time()>=deadline:raise ValueError('original verification deadline')
            exit_code=child.poll()
            if exit_code is not None and exit_code!=0:raise ValueError('actual verification exit nonzero')
            if exit_code is not None and group_members(child.pid):raise ValueError('owned child still alive')
            for key,_ in selector.select(.02):
                block=os.read(key.fileobj.fileno(),4096)
                if not block:selector.unregister(key.fileobj);continue
                name=key.data;counts[name]+=len(block)
                if counts[name]>LOG_CAP:raise ValueError('bounded verification log exceeded')
                streams[name].write(block)
        actual=child.wait()
        if actual!=0:raise ValueError('actual verification exit nonzero')
        if group_members(child.pid):raise ValueError('owned child still alive')
        return {'exit':actual,'pid':child.pid,'pgid':child.pid,'started_utc':start,'ended_utc':time.time()}
    except BaseException:
        if child:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=.5)
            except subprocess.TimeoutExpired:pass
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=2)
        raise
    finally:
        selector.close()
        for stream in streams.values():stream.flush();os.fsync(stream.fileno());stream.close()
        if child:
            child.stdout.close();child.stderr.close()

def invoke(manifest,pin,pool_head,verify_output,output,*,engineering=False):
    """Parent deadline also bounds sealing, serialization and durable writing."""
    manifest=canonical_path(manifest);run=gd._json(gd._checked_file(manifest,pin));deadline=datetime.datetime.fromisoformat(run['limits']['pool_spec']['deadline_utc']).timestamp()
    def timeout(signum,frame):raise ValueError('original parent verification deadline')
    old=signal.getsignal(signal.SIGALRM);timer=signal.getitimer(signal.ITIMER_REAL)
    if timer!=(0.,0.):raise ValueError('existing parent timer; cannot replace')
    remaining=deadline-time.time()
    if remaining<=0:raise ValueError('original parent verification deadline')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:return _invoke(manifest,pin,pool_head,verify_output,output,engineering=engineering)
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)

def _invoke(manifest,pin,pool_head,verify_output,output,*,engineering=False):
    manifest=gd._checked_file(canonical_path(manifest),pin);run=gd._json(manifest)
    if run['binding'].get('engineering') is not engineering or run['status']!='complete':raise ValueError('complete explicit run identity')
    output=canonical_path(output);verify_output=canonical_path(verify_output)
    if output.exists() or verify_output.exists():raise ValueError('exclusive verification outputs')
    if output==verify_output or output in verify_output.parents or verify_output in output.parents:raise ValueError('separate invocation output')
    roots={manifest.parent,canonical_path(run['binding']['pool_path']),verify_output}
    parent=run['parent']
    while parent:
        p=gd._checked_file(canonical_path(parent['manifest_path']),parent['manifest_sha256']);roots.add(p.parent);parent=gd._json(p)['parent']
        if len(roots)>8:raise ValueError('bounded resume ancestry')
    prepared_path=canonical_path(run['limits']['prepared']['manifest_path']);prepared=gd._json(gd._checked_file(prepared_path,run['limits']['prepared']['manifest_sha256']))
    if prepared['kind']=='engineering_prepared':roots.add(prepared_path.parent)
    if any(output==p or p in output.parents for p in roots):raise ValueError('invocation output inside sealed tree')
    limits=run['limits'];spec=limits['pool_spec'];_,allowance=gr.seal_bounds(spec)
    with ExecutionPool.open(Path(run['binding']['pool_path']),spec,pool_head) as pool:
        pool.reserve('replay','disk',allowance);head=pool.head
    deadline=datetime.datetime.fromisoformat(spec['deadline_utc']).timestamp()
    output.mkdir(mode=0o700)
    argv=[sys.executable,'-B','-m','vkgr.generative_real','verify','--manifest',str(manifest),'--sha256',pin,'--pool-head',head,'--replay','--output',str(verify_output)]
    if engineering:argv.append('--engineering')
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    actual=wait_verify(argv,ROOT,env,output,deadline)
    path=verify_output/'verification-receipt.json';receipt=gd._json(gd._no_symlink(path));sha=file_sha256(path)
    if receipt.get('schema_version')!=2 or receipt.get('status')!='verified' or receipt.get('replay') is not True or receipt['run_sha256']!=pin or receipt['run_path']!=str(manifest):raise ValueError('actual child verification payload')
    with ExecutionPool.open(Path(receipt['pool_path']),spec,receipt['pool_head']) as pool:
        if pool.state!=receipt['pool_state']:raise ValueError('actual sealed pool state')
    def guard():
        if time.time()>=deadline:raise ValueError('original parent sealing deadline')
    files=gr.sealed_files(receipt['tree_roots'],receipt['input_files'],spec,budget_check=guard)
    value={'schema_version':2,'kind':'verification_invocation','status':'sealed',**actual,
        'verification_path':str(path),'verification_sha256':sha,'helper_sha256':file_sha256(__file__),
        'argv':argv,'cwd':str(ROOT),'environment':{k:env[k] for k in ('PYTHONPATH','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},
        'logs':{name:{'path':str(output/name),'bytes':(output/name).stat().st_size,'sha256':file_sha256(output/name)} for name in ('stdout','stderr')},
        'pool_head':receipt['pool_head'],'pool_state':receipt['pool_state'],'files':files,'closure_sha256':gr.digest(files),'reserved_bytes':allowance}
    if time.time()>=deadline or len(canonical_bytes(value))+2*LOG_CAP>allowance:raise ValueError('original verification sealing budget')
    guard();inv=output/'invocation.json';inv_sha=gr.ge._write_json(inv,value);guard()
    return {'verification_path':str(path),'verification_sha256':sha,'invocation_path':str(inv),'invocation_sha256':inv_sha,'exit':actual['exit']}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','verify-output','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('sha256','pool-head'):p.add_argument('--'+name,required=True)
    p.add_argument('--engineering',action='store_true');a=p.parse_args()
    try:print(json.dumps(invoke(a.manifest,a.sha256,a.pool_head,a.verify_output,a.output,engineering=a.engineering),sort_keys=True))
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc:print('G3 verification waiter failed: '+str(exc),file=sys.stderr);sys.exit(1)
