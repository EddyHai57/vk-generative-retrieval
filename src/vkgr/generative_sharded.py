"""INFRA-01 A: sharded parallel G validation decode (new runs only).

Workers only decode; the parent reserves budget, checks every result, builds
rows, applies the frozen decode gate and publishes shards in ascending order,
so artifacts match ``generative_evaluation.evaluate_queries`` except for
wall-clock observations. Top level imports the standard library only: spawn
bootstrap re-imports this module while SIGINT/SIGTERM are still blocked.
"""
import hashlib,math,os,re,signal,sys,time
from pathlib import Path

MAX_WORKERS=32
_WINDOW=2  # in-flight shards per worker
_PROGRESS_DISABLED=[False]
_EXECUTION_DEFINITION='rss excludes worker processes'

def cpu_quota_cores(proc_root=Path('/proc'),sys_root=Path('/sys')):
    """floor(cgroup v2 cpu.max quota/period), min 1; None if no quota or unreadable."""
    try:
        lines=(Path(proc_root)/'self/cgroup').read_text().splitlines()
        rel=next(line[3:] for line in lines if line.startswith('0::'))
        fields=(Path(sys_root)/'fs/cgroup'/rel.lstrip('/')/'cpu.max').read_text().split()
        if len(fields)!=2 or fields[0]=='max':return None
        quota,period=int(fields[0]),int(fields[1])
        if quota<1 or period<1:return None
        return max(1,quota//period)
    except (OSError,StopIteration,ValueError):
        return None

def eval_workers_from_env(environ=None):
    environ=os.environ if environ is None else environ
    value=environ.get('VKGR_EVAL_WORKERS')
    if value is None or value=='':return 1
    quota=cpu_quota_cores()
    limit=MAX_WORKERS if quota is None else min(MAX_WORKERS,quota)
    if not re.fullmatch(r'[1-9][0-9]?',value) or int(value)>limit:
        raise ValueError('eval workers env: integer 1..32'+('' if quota is None else f' and <= cgroup cpu.max quota {quota}'))
    return int(value)

def progress_target_from_env(environ=None):
    environ=os.environ if environ is None else environ
    value=environ.get('VKGR_PROGRESS_PATH')
    if value is None or value=='':return None
    path=Path(value);run_id=environ.get('VKGR_RUN_ID')
    if not path.is_absolute() or not path.parent.is_dir() or path.is_symlink() or path.parent.is_symlink() or type(run_id) is not str or not 1<=len(run_id)<=64:raise ValueError('progress env')
    return {'path':path,'run_id':run_id}

def check_progress_outside(progress,*roots):
    if progress is None:return
    target=Path(progress['path']).resolve()
    for root in roots:
        root=Path(root).resolve()
        if target==root or root in target.parents:raise ValueError('progress path inside charged tree')

def progress_record(target,stage,phase,epoch,max_epochs,completed,total,message=None):
    return {'schema':'vkgr-progress-v1','run_id':target['run_id'],'line':'G','stage':stage,'phase':phase,
            'epoch':epoch,'max_epochs':max_epochs,'completed':completed,'total':total,'message':message,
            'updated_at':time.time(),'pid':os.getpid()}

def write_progress(target,record):
    """Atomic best-effort observation file; never affects the job."""
    import json
    if target is None or _PROGRESS_DISABLED[0]:return
    if not re.fullmatch(r'[a-z0-9-]{1,32}',record['stage']) or (record['message'] is not None and len(record['message'])>200):raise ValueError('progress record schema')
    data=json.dumps(record,sort_keys=True,separators=(',',':')).encode()
    if len(data)>4096:raise ValueError('bounded progress record')
    path=Path(target['path']);tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with open(tmp,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    except OSError as exc:
        _PROGRESS_DISABLED[0]=True
        try:tmp.unlink()
        except OSError:pass
        sys.stderr.write(f'progress write disabled: {exc.errno}\n');sys.stderr.flush()

def evaluation_projection(directory):
    """Bytes compared for sharded/serial equivalence (design 2.6)."""
    import json
    from .data_audit import canonical_bytes
    out=[]
    for path in sorted(Path(directory).iterdir(),key=lambda p:p.name):
        value=json.loads(path.read_bytes())
        if path.name.startswith('shard-'):
            value={**value,'rows':[{k:v for k,v in row.items() if k!='latency_seconds'} for row in value['rows']]}
        elif path.name.startswith('progress-') or path.name=='manifest.json':
            value={k:v for k,v in value.items() if k!='observations'}
            value['shards']=[{'start':s['start'],'end':s['end'],'name':Path(s['path']).name} for s in value['shards']]
        out.append(canonical_bytes({'file':path.name,'value':value}))
    return b''.join(out)

def _init_payload(model,context,config,runtime,beam,model_sha256):
    import io,pickle,torch
    codec=pickle.dumps(context.codec);state=io.BytesIO();torch.save(model.state_dict(),state)
    return {'codec_bytes':codec,'codec_sha256':hashlib.sha256(codec).hexdigest(),
            'histories':[q['history'] for q in context.queries],'config':config,'runtime':runtime,'beam':beam,
            'parent_pid':os.getpid(),'state_dict':state.getvalue(),'model_sha256':model_sha256}

def _send(conn,message):
    try:conn.send(message);return True
    except OSError:return False

def _worker(conn,parent_pid):
    # Order matters (design D-A4): ignore, unblock (drops pending), deathsig, orphan check.
    signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN)
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGINT,signal.SIGTERM})
    if sys.platform.startswith('linux'):
        import ctypes
        ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL)  # PR_SET_PDEATHSIG
    if os.getppid()!=parent_pid:os._exit(1)
    try:
        message=conn.recv()
    except (EOFError,OSError):return
    if type(message) is not tuple or len(message)!=2 or message[0]!='init':return
    payload=message[1]
    try:
        if payload['parent_pid']!=parent_pid:raise ValueError('worker parent identity')
        import io,pickle,torch
        from . import generative_data as gd,generative_training as gt,generative_evaluation as ev
        from .generative_real import configure_runtime
        from .tiger import decode_history
        runtime=configure_runtime(payload['runtime'])
        if hashlib.sha256(payload['codec_bytes']).hexdigest()!=payload['codec_sha256']:raise ValueError('worker codec bytes identity')
        codec=gd._ValidatedCodec(pickle.loads(payload['codec_bytes']).catalog);gd._check_codec(codec)
        device=runtime['device'];model=gt.make_model(codec,payload['config'],device=device)
        model.load_state_dict(torch.load(io.BytesIO(payload['state_dict']),map_location=device,weights_only=True))
        if ev.model_digest(model)!=payload['model_sha256']:raise ValueError('worker model digest identity')
        histories=payload['histories'];beam=payload['beam'];cuda=device.startswith('cuda')
    except Exception as exc:
        _send(conn,('error',type(exc).__name__,str(exc)));return
    if not _send(conn,('ready',)):return
    while True:
        try:message=conn.recv()
        except (EOFError,OSError):return
        if type(message) is not tuple or message[0]!='shard':return
        _,start,end=message;rows=[]
        try:
            for ordinal in range(start,end):
                if cuda:torch.cuda.synchronize()
                begin=time.perf_counter();items,audit=decode_history(model,codec,histories[ordinal],beam)
                if cuda:torch.cuda.synchronize()
                rows.append((ordinal,items,audit,time.perf_counter()-begin))
        except Exception as exc:
            _send(conn,('error',type(exc).__name__,str(exc)));return
        if not _send(conn,('ok',start,rows)):return

def _recv(conn,process):
    while True:
        try:
            if conn.poll(1.0):return conn.recv()
        except (EOFError,OSError):raise ValueError('sharded evaluation worker died') from None
        if not process.is_alive():
            try:
                if conn.poll(0):return conn.recv()
            except (EOFError,OSError):pass
            raise ValueError('sharded evaluation worker died')

def _shard_rows(message):
    """Unpack one worker reply (patch point for parent-side identity tests)."""
    if type(message) is tuple and len(message)==3 and message[0]=='error':raise ValueError(f'sharded evaluation worker: {message[1]}: {message[2]}')
    if type(message) is not tuple or len(message)!=3 or message[0]!='ok' or type(message[2]) is not list:raise ValueError('sharded result identity')
    return message[2]

def _cleanup(processes,connections):
    for conn in connections:
        try:conn.close()
        except OSError:pass
    for process in processes:
        process.join(5)
        if process.is_alive():process.terminate();process.join(1)  # workers ignore SIGTERM; kill follows
        if process.is_alive():process.kill();process.join()

def evaluate_queries_sharded(model,context,*,output,checkpoint_sha256,workers,config,runtime,
        beam=100,shard_size=64,resume_manifest=None,resume_sha256=None,
        stop_after_shards=None,should_stop=None,guard=None,before_decode=None,after_decode=None,on_shard=None):
    """Parallel twin of ev.evaluate_queries; same artifacts except wall-clock observations."""
    if type(workers) is not int or not 2<=workers<=MAX_WORKERS:raise ValueError('sharded workers 2..32')
    if type(config) is not dict or type(runtime) is not dict:raise ValueError('explicit worker config/runtime')
    if runtime.get('threads')!=1 or os.environ.get('OMP_NUM_THREADS')!='1':raise ValueError('sharded evaluation requires threads==1')
    import resource,multiprocessing as mp
    import numpy as np,torch
    from . import generative_evaluation as ev
    from .data_audit import canonical_bytes
    from .generative_experiment import _write_json
    from .generative_data import _no_symlink
    binding=ev._evaluation_binding(model,context,checkpoint_sha256,beam,shard_size)
    if guard is not None:
        begin=time.perf_counter();guard();guard_seconds=time.perf_counter()-begin
    else:guard_seconds=0.
    if (resume_manifest is None)!=(resume_sha256 is None):raise ValueError('paired external resume manifest pin required')
    if stop_after_shards is not None and (type(stop_after_shards) is not int or stop_after_shards<1 or context.binding['synthetic'] is not True):raise ValueError('synthetic bounded interruption only')
    queries=context.queries;catalog=set(context.catalog);rows=[];shards=[]
    if resume_manifest is not None:
        parent,rows=ev._read_evaluation(resume_manifest,resume_sha256,binding,queries,context.catalog);shards=parent['shards']
        prior_aggregate=ev.aggregate_items([row['items'] for row in rows],queries,context.catalog,slice_definition=context.slice_definition) if parent['complete'] else None
        if canonical_bytes(parent['aggregate'])!=canonical_bytes(prior_aggregate) or parent['computation_sha256']!=ev._computation(rows,prior_aggregate):raise ValueError('parent evaluation aggregate/computation')
    output=_no_symlink(output);output.mkdir(mode=0o700,exist_ok=False)
    starts=list(range(len(rows),len(queries),shard_size))
    limit=len(starts) if stop_after_shards is None else min(len(starts),stop_after_shards)
    count=max(1,min(workers,len(starts)))
    payload=_init_payload(model,context,config,runtime,beam,binding['model_sha256'])
    seconds={'startup':0.,'hooks':guard_seconds,'wait':0.,'write':0.};launched=time.perf_counter()
    spawn=mp.get_context('spawn');processes=[];connections=[]
    try:
        from multiprocessing import resource_tracker
        for _ in range(count):
            parent_end,child_end=spawn.Pipe()
            # Keep Process args tiny: CPython writes them into a pipe whose read
            # end the parent still holds, so a large payload makes start() block
            # until the child reads, and hang forever if it dies in bootstrap.
            process=spawn.Process(target=_worker,args=(child_end,os.getpid()),daemon=True)
            # Launching the tracker inside start() would SIG_UNBLOCK both signals
            # in this process and void the inherited mask; launch it first.
            resource_tracker.ensure_running()
            old=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
            try:process.start()
            except OSError as exc:raise ValueError(f'sharded evaluation worker: {exc}') from None
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)
            child_end.close();processes.append(process);connections.append(parent_end)
        for conn in connections:
            if not _send(conn,('init',payload)):raise ValueError('sharded evaluation worker died')
        for conn,process in zip(connections,processes):
            reply=_recv(conn,process)
            if reply!=('ready',):_shard_rows(reply);raise ValueError('sharded evaluation worker: init handshake')
        seconds['startup']=time.perf_counter()-launched
        inflight=[0]*count;dispatched=0;published=0;written=0;stopping=False
        def timed(key,hook):
            if hook is None:return
            begin=time.perf_counter();hook();seconds[key]+=time.perf_counter()-begin
        def fill():
            nonlocal dispatched,stopping
            while not stopping and dispatched<limit and inflight[dispatched%count]<_WINDOW:
                if should_stop is not None and should_stop():stopping=True;break
                start=starts[dispatched];end=min(start+shard_size,len(queries))
                for _ in range(start,end):timed('hooks',before_decode)
                w=dispatched%count
                if not _send(connections[w],('shard',start,end)):raise ValueError('sharded evaluation worker died')
                inflight[w]+=1;dispatched+=1
        fill()
        while published<dispatched:
            w=published%count;start=starts[published];end=min(start+shard_size,len(queries))
            begin=time.perf_counter();result=_shard_rows(_recv(connections[w],processes[w]));inflight[w]-=1;seconds['wait']+=time.perf_counter()-begin
            if [r[0] if type(r) is tuple and len(r)==4 else None for r in result]!=list(range(start,end)) or any(
                    type(items) is not list or any(type(i) is not int for i in items) or type(audit) is not dict or type(latency) is not float or not math.isfinite(latency)
                    for _,items,audit,latency in result):raise ValueError('sharded result identity')
            new=[]
            for ordinal,preds,audit,elapsed in result:
                timed('hooks',after_decode)
                if not preds or len(preds)!=len(set(preds)) or any(i not in catalog for i in preds) or any(audit[k]!=0 for k in ('invalid','deadend','collision')):raise ValueError('decoded candidate/audit gate')
                new.append({'ordinal':ordinal,'query_sha256':hashlib.sha256(canonical_bytes(queries[ordinal])).hexdigest(),'items':preds,'audit':audit,'latency_seconds':elapsed})
            begin=time.perf_counter();shard={'schema_version':1,'binding':binding,'start':start,'end':end,'rows':new}
            path=output/f'shard-{start:06d}-{end:06d}.json';pin=_write_json(path,shard)
            shards.append({'start':start,'end':end,'path':str(path.absolute()),'sha256':pin});rows.extend(new);written+=1;published+=1
            progress={'schema_version':1,'task_id':'G2','kind':'query_evaluation','binding':binding,'query_count':len(queries),'completed_queries':len(rows),'complete':len(rows)==len(queries),'shards':list(shards),'computation_sha256':ev._computation(rows,None),'aggregate':None,'observations':None}
            if not progress['complete']:_write_json(output/f'progress-{len(rows):06d}.json',progress)
            seconds['write']+=time.perf_counter()-begin
            if on_shard is not None:on_shard(len(rows),len(queries))
            if (stop_after_shards is not None and written>=stop_after_shards) or (should_stop is not None and should_stop()):stopping=True
            fill()
    finally:
        _cleanup(processes,connections)
    complete=len(rows)==len(queries)
    aggregate=ev.aggregate_items([r['items'] for r in rows],queries,context.catalog,slice_definition=context.slice_definition) if complete else None
    latency=[row['latency_seconds'] for row in rows];device=next(model.parameters()).device
    observations={'latency':{'count':len(latency),'p50_seconds':float(np.quantile(latency,.5)) if latency else None,'p95_seconds':float(np.quantile(latency,.95)) if latency else None,'definition':'per-query tokenization+encoder+four-position beam; linear quantile'},
                  'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'rss_definition':'process lifetime observed high-water RSS; not evaluation-only peak',
                  'vram':'not_applicable' if device.type=='cpu' else {'allocated_bytes':torch.cuda.memory_allocated(device),'definition':'observed at evaluation completion; not peak'},
                  'execution':{'workers':workers,'start_method':'spawn','assignment':'round_robin','definition':_EXECUTION_DEFINITION,
                      'parent_seconds':seconds,'worker_decode_seconds':sum(latency),'seconds_definition':'parent wall: startup=spawn+init handshake; hooks=guard/reserve hooks; wait=blocked on worker results; write=shard+progress publish. worker_decode_seconds=sum of per-query latency across workers'}}
    manifest={'schema_version':1,'task_id':'G2','kind':'query_evaluation','binding':binding,'query_count':len(queries),'completed_queries':len(rows),'complete':complete,'shards':shards,'computation_sha256':ev._computation(rows,aggregate),'aggregate':aggregate,'observations':observations}
    path=output/'manifest.json';pin=_write_json(path,manifest)
    ev._read_evaluation(path,pin,binding,queries,context.catalog)
    return {'manifest':manifest,'manifest_path':path,'manifest_sha256':pin,'aggregate':aggregate,'observations':observations}
