"""G3 bounded execution accounting and resource/cost gates."""
import copy,datetime,hashlib,math,os
from pathlib import Path
from . import generative_data as gd,generative_experiment as ge
from .data_audit import canonical_bytes,file_sha256

def _utc():return datetime.datetime.now(datetime.timezone.utc)
def _live(spec):
    try:t=datetime.datetime.fromisoformat(spec['deadline_utc'])
    except (KeyError,TypeError,ValueError) as exc:raise ValueError('aware fixed deadline required') from exc
    if t.tzinfo is None or t<=_utc():raise ValueError('execution deadline expired')

def _spec(spec):
    keys={'task_id','kind','steps_by_phase','max_steps','max_decode','max_disk_bytes','max_resumes','deadline_utc','binding_sha256'}
    if type(spec) is not dict or set(spec)!=keys or spec['task_id']!='G3' or spec['kind'] not in ('engineering_pool','pilot_pool','formal_pool'):raise ValueError('G3 pool specification')
    phases=spec['steps_by_phase']
    if type(phases) is not dict or not phases or any(type(k) is not str or not k or type(v) is not int or v<1 for k,v in phases.items()):raise ValueError('registered phase step caps')
    for k in ('max_steps','max_decode','max_disk_bytes','max_resumes'):
        if type(spec[k]) is not int or spec[k]<(0 if k=='max_resumes' else 1):raise ValueError('finite pool caps')
    if sum(phases.values())!=spec['max_steps'] or type(spec['binding_sha256']) is not str or len(spec['binding_sha256'])!=64 or any(c not in '0123456789abcdef' for c in spec['binding_sha256']):raise ValueError('pool binding/cumulative caps')
    if spec['kind']=='pilot_pool' and (spec['max_steps']>200 or spec['max_decode']>128 or spec['max_disk_bytes']>2*1024**3 or spec['max_resumes']>1):raise ValueError('pilot global caps')
    if spec['kind']=='formal_pool' and (phases!={'formal':5100,'replay':5100} or spec['max_decode']!=153720 or spec['max_resumes']>1):raise ValueError('formal registered caps')
    _live(spec);return copy.deepcopy(spec)

class ExecutionPool:
    """Single writer; reservations never refunded, including failed work."""
    @classmethod
    def create(cls,root,spec):
        s=_spec(spec);root=gd._no_symlink(root);root.mkdir(mode=0o700,exist_ok=False)
        obj=cls();obj.root=root;obj.spec=s;obj.head=None;obj.index=0
        obj._lock();obj.accounting_measurements=[];obj.state={'steps':{p:0 for p in s['steps_by_phase']},'decode':0,'disk':0,'resumes':0,'children':{},'wall_by_phase':{p:0. for p in s['steps_by_phase']}}
        try:obj._append('initial',None);return obj
        except Exception:obj.close();raise

    def _lock(self):
        import fcntl
        self._owner=gd._no_symlink(self.root/'owner.lock').open('a+b')
        try:fcntl.flock(self._owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:self._owner.close();raise ValueError('execution pool already owned') from exc

    @classmethod
    def open(cls,root,spec,expected_head):
        s=_spec(spec)
        if type(expected_head) is not str or len(expected_head)!=64:raise ValueError('caller external pool head required')
        obj=cls();obj.root=gd._no_symlink(root);obj.spec=s;obj.head=None;obj.index=0;obj._lock();obj.accounting_measurements=[]
        try:
            obj.state={'steps':{p:0 for p in s['steps_by_phase']},'decode':0,'disk':0,'resumes':0,'children':{},'wall_by_phase':{p:0. for p in s['steps_by_phase']}}
            paths=sorted(obj.root.glob('event-*.json'))
            if not paths:raise ValueError('pool genesis missing')
            for i,p in enumerate(paths):
                row=gd._json(gd._no_symlink(p))
                if p.name!=f'event-{i:06d}.json' or set(row)!={'schema_version','index','previous_head','spec','action','payload','state','observed_utc'} or type(row['schema_version']) is not int or row['schema_version']!=1 or type(row['index']) is not int or row['index']!=i or row['previous_head']!=obj.head or canonical_bytes(row['spec'])!=canonical_bytes(s):raise ValueError('pool chain identity/gap')
                stamp=datetime.datetime.fromisoformat(row['observed_utc'])
                if stamp.tzinfo is None or stamp>_utc() or i and stamp<previous_stamp:raise ValueError('pool chronological observation')
                if i==0:
                    if row['action']!='initial' or row['payload'] is not None:raise ValueError('pool genesis')
                else:obj.state=obj._transition(row['action'],row['payload'])
                obj.state['disk']+=p.stat().st_size
                if obj.state['disk']>s['max_disk_bytes']:raise ValueError('pool journal disk cap exceeded')
                if canonical_bytes(row['state'])!=canonical_bytes(obj.state):raise ValueError('pool monotonic state transition')
                previous_stamp=stamp;obj.head=file_sha256(p);obj.index=i+1
            if obj.head!=expected_head:raise ValueError('external pool head mismatch/truncation')
            return obj
        except Exception:obj.close();raise

    def _transition(self,action,payload):
        state=copy.deepcopy(self.state)
        if action=='reserve':
            if type(payload) is not dict or set(payload)!={'phase','resource','amount'}:raise ValueError('reservation schema')
            phase,res,n=payload['phase'],payload['resource'],payload['amount']
            if phase not in state['steps'] or res not in ('steps','decode','disk') or type(n) is not int or n<1:raise ValueError('registered positive reservation')
            if res=='steps':
                state['steps'][phase]+=n
                if state['steps'][phase]>self.spec['steps_by_phase'][phase] or sum(state['steps'].values())>self.spec['max_steps']:raise ValueError('cumulative optimizer cap exceeded')
            else:
                state[res]+=n
                if state[res]>self.spec['max_decode' if res=='decode' else 'max_disk_bytes']:raise ValueError('cumulative decode/disk cap exceeded')
        elif action=='resume':
            if payload is not None:raise ValueError('resume schema')
            state['resumes']+=1
            if state['resumes']>self.spec['max_resumes']:raise ValueError('pool resume cap exceeded')
        elif action=='child':
            if type(payload) is not dict or set(payload)!={'phase','ledger_path','ledger_head'} or payload['phase'] not in state['steps'] or type(payload['ledger_path']) is not str or type(payload['ledger_head']) is not str or len(payload['ledger_head'])!=64:raise ValueError('local trajectory external ledger receipt')
            state['children'][payload['phase']]={k:payload[k] for k in ('ledger_path','ledger_head')}
        elif action=='wall':
            if type(payload) is not dict or set(payload)!={'phase','seconds'} or payload['phase'] not in state['steps'] or type(payload['seconds']) not in (int,float) or not math.isfinite(payload['seconds']) or payload['seconds']<0:raise ValueError('finite actual phase wall')
            state['wall_by_phase'][payload['phase']]+=payload['seconds']
        else:raise ValueError('unknown pool operation')
        return state

    def _append(self,action,payload):
        if action!='wall':_live(self.spec)
        state=copy.deepcopy(self.state) if action=='initial' else self._transition(action,payload)
        row={'schema_version':1,'index':self.index,'previous_head':self.head,'spec':self.spec,'action':action,'payload':payload,'state':state,'observed_utc':_utc().isoformat()}
        base=state['disk'];size=0
        while True:
            state['disk']=base+size;actual=len(canonical_bytes(row))
            if actual==size:break
            size=actual
        if state['disk']>self.spec['max_disk_bytes']:raise ValueError('pool journal disk cap exceeded')
        import time
        begin=time.perf_counter();path=self.root/f'event-{self.index:06d}.json';self.head=ge._write_json(path,row);self.index+=1;self.state=state
        self.accounting_measurements.append({'kind':'pool_event','bytes':size,'seconds':time.perf_counter()-begin})
    def reserve(self,phase,resource,amount=1):self._append('reserve',{'phase':phase,'resource':resource,'amount':amount})
    def resume(self):self._append('resume',None)
    def charge_wall(self,phase,seconds):self._append('wall',{'phase':phase,'seconds':seconds})
    def bind_child(self,phase,ledger_path,ledger_head):self._append('child',{'phase':phase,'ledger_path':str(Path(ledger_path).absolute()),'ledger_head':ledger_head})
    def check(self):_live(self.spec)
    def close(self):
        if getattr(self,'_owner',None) is not None:self._owner.close();self._owner=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def accounted_ledger(pool,phase):
    """Keep the frozen local ledger protocol while reserving every event byte."""
    class AccountedLedger(ge.BudgetLedger):
        def _append(self,action,budget,pending,phase_arg=None,elapsed=0.):
            row={'schema_version':1,'index':self._index,'action':action,'binding':self.binding,'previous_head':self.head,
                 'budget':copy.deepcopy(budget),'pending':pending,'phase':phase_arg,'elapsed_seconds':elapsed}
            pool.reserve(phase,'disk',len(canonical_bytes(row)))
            import time
            begin=time.perf_counter();result=super()._append(action,budget,pending,phase_arg,elapsed)
            pool.accounting_measurements.append({'kind':'local_ledger_event','bytes':len(canonical_bytes(row)),'seconds':time.perf_counter()-begin})
            return result
    return AccountedLedger

def write_accounted_json(pool,phase,path,value,*,stamp_pool=False):
    """Reserve actual payload plus bounded receipt growth before any output."""
    allowance=len(canonical_bytes(value))+256
    pool.reserve(phase,'disk',allowance)
    if stamp_pool:
        value['pool_head']=pool.head;value['pool_state']=copy.deepcopy(pool.state)
    if len(canonical_bytes(value))>allowance:raise ValueError('serialized receipt exceeds reservation')
    return ge._write_json(path,value)

def save_accounted_checkpoint(pool,phase,path,model,optimizer,cursor,*,binding,selection,budget):
    """Serialize the frozen engine schema in RAM, then bill exact file bytes."""
    import io,time,torch
    from . import generative_training as gt
    begin=time.perf_counter()
    value={'schema_version':1,'task_id':'G2','model':dict(model.state_dict()),'optimizer':optimizer.state_dict(),
           'cursor':cursor.state_dict(),'binding':binding,'selection':selection,'budget':budget,'rng':gt._rng(model)}
    gt._checkpoint(value,model,optimizer,cursor,binding)
    buffer=io.BytesIO();torch.save(value,buffer);data=buffer.getbuffer()
    pool.reserve(phase,'disk',len(data));path=gd._no_symlink(path)
    with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    pin=file_sha256(path)
    return pin,{'kind':'checkpoint_serialize_write_fsync_hash','bytes':len(data),'seconds':time.perf_counter()-begin}

def check_headroom(observation,minimum=.2):
    if type(minimum) not in (int,float) or not math.isfinite(minimum) or not .2<=minimum<1:raise ValueError('at least 20 percent headroom')
    for total,free in (('ram_total','ram_available'),('vram_total','vram_free'),('disk_total','disk_free')):
        t=observation.get(total);f=observation.get(free)
        if type(t) not in (int,float) or type(f) not in (int,float) or not math.isfinite(t) or not math.isfinite(f) or not 0<=f<=t or t<=0 or f/t<minimum:raise ValueError('resource unknown/insufficient visible headroom')
    return dict(observation)

def observe_resources(output,device):
    """Linux visible memory ancestry plus actual GPU/target mount; no host fallback."""
    import torch
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        parts=line.split();mem[parts[0].rstrip(':')]=int(parts[1])*1024
    membership=[line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    if len(membership)!=1:raise ValueError('cgroup v2 membership required')
    choices=[]
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        before,after=line.split(' - ',1)
        if after.split()[0]=='cgroup2':
            fields=before.split();mount_root,mount_point=fields[3:5]
            group=membership[0]
            if group==mount_root or group.startswith(mount_root.rstrip('/')+'/'):
                choices.append((len(mount_root),Path(mount_point),Path(mount_point)/group[len(mount_root):].lstrip('/')))
    if not choices:raise ValueError('mapped visible cgroup ancestry required')
    _,mount,current=max(choices,key=lambda x:x[0]);limits=[];cpu=[]
    while True:
        maximum=(current/'memory.max').read_text().strip();used=int((current/'memory.current').read_text())
        quota,period=(current/'cpu.max').read_text().split()
        if quota!='max':cpu.append(int(quota)/int(period))
        if maximum!='max':
            total=int(maximum)
            if total<=0 or used<0:raise ValueError('visible memory limit schema')
            limits.append((total,max(0,total-used)))
        if current==mount:break
        parent=current.parent
        if parent==current or mount not in parent.parents and parent!=mount:raise ValueError('cgroup ancestor boundary')
        current=parent
    if not limits:raise ValueError('finite visible memory gate unavailable')
    total=min([mem['MemTotal']]+[t for t,_ in limits]);free=min([mem['MemAvailable']]+[f for _,f in limits])
    disk=os.statvfs(output);vfree,vtotal=torch.cuda.mem_get_info(device)
    import resource
    io={k:int(v) for k,v in (line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())}
    return check_headroom({'ram_total':total,'ram_available':min(free,total),'vram_total':vtotal,'vram_free':vfree,
        'disk_total':disk.f_blocks*disk.f_frsize,'disk_free':disk.f_bavail*disk.f_frsize,
        'cpu_visible_count':min([len(os.sched_getaffinity(0))]+cpu),'hidden_ancestry':'unknown',
        'gpu_peak_allocated':torch.cuda.max_memory_allocated(device),'gpu_peak_reserved':torch.cuda.max_memory_reserved(device),
        'rss_lifetime_high_water_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'io_read_bytes':io['read_bytes'],'io_write_bytes':io['write_bytes']})

def estimate_cost(*,examples,queries,epochs,step_seconds,query_seconds,fixed_seconds,checkpoint_bytes,shard_bytes,ledger_event_bytes=None,pool_event_bytes=None,fixed_artifact_bytes=None,safety_factor=2.):
    for n in (examples,queries,epochs,checkpoint_bytes,shard_bytes,ledger_event_bytes,pool_event_bytes,fixed_artifact_bytes):
        if type(n) is not int or n<1:raise ValueError('positive actual workload and serialized bytes required')
    if epochs>30:raise ValueError('registered epoch cap')
    for n in (step_seconds,query_seconds,fixed_seconds,safety_factor):
        if type(n) not in (int,float) or not math.isfinite(n) or n<=0:raise ValueError('measured finite positive cost required')
    if safety_factor<2:raise ValueError('declared conservative cost factor required')
    steps=math.ceil(examples/512);updates=2*epochs*steps;calls=(2*epochs+1)*queries
    # Both formal and full replay retain pre/post-eval snapshots and shards.
    # Bounded orchestration counts include initial/resume, every reserve/complete,
    # per-epoch wall records, checkpoint/child/disk events and one safe recovery.
    ledger_events=2*updates+2*epochs+8
    pool_events=6*updates+calls+20*epochs+64
    disk=(4*epochs+4)*checkpoint_bytes+(2*epochs+1)*shard_bytes+ledger_events*ledger_event_bytes+pool_events*pool_event_bytes+fixed_artifact_bytes
    return {'steps_per_epoch':steps,'optimizer_reservations':updates,'decode_calls':calls,
        'wall_seconds':safety_factor*(updates*step_seconds+calls*query_seconds+fixed_seconds),
        'ledger_events':ledger_events,'pool_events':pool_events,'disk_bytes':math.ceil(safety_factor*disk),'boundary':'operational estimate, not measured full validation or guaranteed bound'}

def pilot(context,output,*,engineering=False,permit_path=None,permit_sha=None,prepared=None):
    """One finite train-only workload; no validation targets or formal checkpoint reuse."""
    import time,torch
    from . import generative_real as gr,generative_training as gt
    from .tiger import decode_history
    if not engineering and (permit_path is None or permit_sha is None):raise ValueError('external pilot permit before device/input reads')
    context.check();output=gd._no_symlink(output)
    if type(prepared) is not dict or set(prepared)!={'manifest_path','manifest_sha256'}:raise ValueError('fresh-process pilot prepared external receipt required')
    if context.binding.get('engineering') is not engineering:raise ValueError('engineering/real pilot identity')
    if engineering:
        if permit_path is not None or permit_sha is not None:raise ValueError('engineering cannot accept real permit')
        runtime=gr.engineering_runtime();steps=2;batch=2;decode_count=4
        spec={'task_id':'G3','kind':'engineering_pool','steps_by_phase':{'capacity':2,'continuous':2,'resumed':2},
            'max_steps':6,'max_decode':8,'max_disk_bytes':64*1024**2,'max_resumes':1,
            'deadline_utc':(_utc()+datetime.timedelta(seconds=60)).isoformat(),'binding_sha256':gr.digest(context.binding)}
        candidates=[2]
    else:
        value=gd._json(gd._checked_file(permit_path,permit_sha))
        keys={'schema_version','task_id','kind','status','context_binding_sha256','output','runtime','pool_spec','maximum_wall_seconds','accept_hidden_ancestry_unknown'}
        if type(value) is not dict or set(value)!=keys or value['schema_version']!=1 or value['task_id']!='G3' or value['kind']!='pilot_permit' or value['status']!='approved' or value['context_binding_sha256']!=gr.digest(context.binding) or Path(value['output']).absolute()!=output.absolute() or value['accept_hidden_ancestry_unknown'] is not True:raise ValueError('approved exclusive pilot/input/resource scope')
        if type(value['maximum_wall_seconds']) is not int or not 1<=value['maximum_wall_seconds']<=1800:raise ValueError('finite <=1800 second pilot')
        spec=_spec(value['pool_spec']);remaining=(datetime.datetime.fromisoformat(spec['deadline_utc'])-_utc()).total_seconds()
        expected_phases={'capacity512':8,'capacity256':8,'capacity128':8,'continuous':8,'resumed':8,'throughput':160}
        if spec['kind']!='pilot_pool' or spec['steps_by_phase']!=expected_phases or spec['binding_sha256']!=gr.digest(context.binding) or remaining>value['maximum_wall_seconds']:raise ValueError('one global pilot budget and fixed registered phases')
        runtime=gr.configure_runtime(value['runtime'])
        if runtime['mode']!='real' or context.binding['synthetic'] is not False:raise ValueError('actual CUDA real pilot')
        steps=8;batch=512;decode_count=16;candidates=[512,256,128]
    output.mkdir(mode=0o700,exist_ok=False)
    # Stream the accepted examples, retaining only ordinals/lengths for bounded
    # workload selection; no validation query or target used to choose probes.
    lengths=[]
    for start in range(0,len(context.examples),512):
        for offset,row in enumerate(context.examples.rows(list(range(start,min(start+512,len(context.examples)))))):
            lengths.append((len(row['history']),start+offset))
    ordered=sorted(lengths,key=lambda x:(-x[0],x[1]));maximum=ordered[0][0]
    if not engineering and (maximum!=50 or len(ordered)<512 or len(ordered)<112):raise ValueError('real length50 and sufficient distinct train prefixes required')
    fixed=[i for _,i in ordered[:min(batch,len(ordered))]]
    queries=[{'ordinal':i,'history':context.examples.rows([i])[0]['history']} for i in [n for _,n in ordered[:112]]]
    observations=[];timings=[];decode_timings=[];serialization=[];attempts=[];device=runtime['device'];cfg=ge._config(1,42)
    def guard(pool):
        pool.check();context.check();gr.runtime_snapshot(runtime)
        if not engineering:observations.append(observe_resources(output,device))
    def fresh(phase,count):
        gr._seed(42);model=gt.make_model(context.codec,cfg,device=device);optimizer=gt.make_optimizer(model,cfg)
        cursor=gt.EpochCursor(42,(count+1)*batch)
        binding={**context.binding,'effective_config':cfg,'runtime':runtime,'pilot_phase':phase,'workload_ordinals_sha256':gr.digest(fixed)}
        budget={'reserved_steps':0,'completed_steps':0,'consumed_wall_by_phase':{'train':0.,'evaluate':0.,'verify':0.},'resume_attempts':0,
            'max_optimizer_steps':count,'max_resume_attempts':1,'deadline_utc':spec['deadline_utc'],'parent_ledger_head':None}
        ledger=accounted_ledger(pool,phase if phase in spec['steps_by_phase'] else 'throughput')(output/(phase+'-ledger'),budget,binding)
        return model,optimizer,cursor,binding,ledger
    def update(pool,phase,objects,micro,count):
        model,optimizer,cursor,binding,ledger=objects
        for _ in range(count):
            guard(pool);pool.reserve(phase,'steps');ledger.reserve_step();indices=cursor.next_batch(batch)
            rows=context.examples.rows([fixed[i%len(fixed)] for i in indices]);begin=time.perf_counter()
            gt.train_effective_batch(model,optimizer,rows,context.codec,microbatch=micro,device=device)
            if device.startswith('cuda'):torch.cuda.synchronize()
            elapsed=time.perf_counter()-begin;ledger.complete_step(elapsed);pool.bind_child(phase,ledger.root,ledger.head)
            pool.charge_wall(phase,elapsed)
            timings.append({'phase':phase,'d_model':cfg['d_model'],'seconds':elapsed,'examples':len(rows)});guard(pool)
    def decode(pool,model,selected,phase):
        result=[]
        for q in selected:
            guard(pool);pool.reserve(phase,'decode')
            if device.startswith('cuda'):torch.cuda.synchronize()
            begin=time.perf_counter();items,audit=decode_history(model,context.codec,q['history'],100)
            if device.startswith('cuda'):torch.cuda.synchronize()
            elapsed=time.perf_counter()-begin;pool.charge_wall(phase,elapsed);guard(pool)
            decode_timings.append({'ordinal':q['ordinal'],'phase':phase,'d_model':model.d_model,'history_length':len(q['history']),'seconds':elapsed})
            if not items or len(items)!=len(set(items)) or any(i not in context.catalog for i in items) or any(audit[k]!=0 for k in ('invalid','deadend','collision')):raise ValueError('actual pilot candidate/audit')
            result.append({'ordinal':q['ordinal'],'items':items,'audit':audit,'seconds':elapsed})
        return result
    with ExecutionPool.create(output/'pool',spec) as pool:
        micro=None
        for candidate in candidates:
            phase='capacity' if engineering else 'capacity'+str(candidate)
            objects=fresh(phase,steps)
            try:
                update(pool,phase,objects,candidate,steps);micro=candidate;attempts.append({'microbatch':candidate,'status':'passed'});break
            except torch.cuda.OutOfMemoryError:
                if engineering:raise
                attempts.append({'microbatch':candidate,'status':'OOM_reserved_not_refunded'})
                # Only one registered smaller candidate, after clearing owned
                # failed tensors and rechecking total-device headroom.
                del objects;torch.cuda.empty_cache();guard(pool)
        if micro is None:raise ValueError('all registered capacity candidates failed')
        continuous=fresh('continuous',steps);update(pool,'continuous',continuous,micro,steps)
        resumed=fresh('resumed',steps);update(pool,'resumed',resumed,micro,steps//2)
        model,optimizer,cursor,binding,ledger=resumed
        selection={'epoch':0,'best_epoch':None,'best_metric':None,'bad_epochs':0}
        checkpoint=output/'pilot-safe.pt'
        checkpoint_sha,measurement=save_accounted_checkpoint(pool,'resumed',checkpoint,model,optimizer,cursor,binding=binding,selection=selection,budget=ledger.snapshot())
        serialization.append(measurement)
        before=gr.file_sha256(checkpoint)
        import subprocess,sys,json
        worker={'task_id':'G3','kind':'pilot_restore_worker','prepared':{k:str(v) if isinstance(v,Path) else v for k,v in prepared.items()},
            'context_binding_sha256':gr.digest(context.binding),'runtime':runtime,'pool_path':str(pool.root.absolute()),'pool_spec':spec,'pool_head':pool.head,
            'checkpoint_path':str(checkpoint.absolute()),'checkpoint_sha256':checkpoint_sha,'binding':binding,
            'local_ledger_path':str(ledger.root.absolute()),'local_ledger_head':ledger.head,'fixed_ordinals':fixed,'steps_remaining':steps-steps//2,
            'batch':batch,'microbatch':micro,'cursor_examples':(steps+1)*batch,'output':str((output/'restore-worker').absolute())}
        worker_path=output/'restore-worker-input.json'
        pool.reserve('resumed','disk',len(canonical_bytes(worker)))
        pool.reserve('resumed','disk',65536)
        worker['pool_head']=pool.head
        worker_sha=ge._write_json(worker_path,worker)
        parent_measurements=list(pool.accounting_measurements)
        pool.close()  # Transfer single-writer ownership, not concurrent polling.
        command=[sys.executable,'-B','-m','vkgr.generative_pilot','--restore-worker',str(worker_path),'--sha256',worker_sha]
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        timeout=max(1.,(datetime.datetime.fromisoformat(spec['deadline_utc'])-_utc()).total_seconds())
        completed=subprocess.run(command,env=env,capture_output=True,text=True,timeout=timeout)
        with (output/'restore-worker.stderr').open('xb') as stream:stream.write(completed.stderr.encode('utf-8')[:65536])
        if completed.returncode:raise ValueError('fresh-process pilot restore failed; original stderr retained')
        child_receipt=json.loads(completed.stdout)
        reopened=ExecutionPool.open(pool.root,spec,child_receipt['pool_head']);pool.__dict__.update(reopened.__dict__);reopened._owner=None
        pool.accounting_measurements=parent_measurements+child_receipt['accounting_measurements']
        timings.extend(child_receipt['timings']);serialization.extend(child_receipt['serialization_measurements'])
        replacement=gt.make_model(context.codec,cfg,device=device);opt=gt.make_optimizer(replacement,cfg);cur=gt.EpochCursor(42,(steps+1)*batch)
        gt.load_checkpoint(Path(child_receipt['checkpoint_path']),child_receipt['checkpoint_sha256'],replacement,opt,cur,expected_binding=binding)
        a=ge._state_digest(continuous[0],continuous[1],continuous[2],selection);b=ge._state_digest(replacement,opt,cur,selection)
        if a!=b or before!=file_sha256(checkpoint):raise ValueError('actual device model/AdamW/cursor/RNG resume mismatch')
        selected=queries[:decode_count];first=decode(pool,continuous[0],selected,'continuous');second=decode(pool,replacement,selected,'resumed')
        if [{k:v for k,v in r.items() if k!='seconds'} for r in first]!=[{k:v for k,v in r.items() if k!='seconds'} for r in second]:raise ValueError('actual device restored decode mismatch')
        if not engineering:
            # Exactly 96 additional train-only invocations: both registered sizes.
            for index in (0,1):
                cfg=ge._config(index,42);throughput=fresh('throughput-'+str(index),80)
                update(pool,'throughput',throughput,micro,80)
                decode(pool,throughput[0],queries[16+index*48:16+(index+1)*48],'throughput')
        probe=output/'decode-write-probe.json';probe_value={'rows':first+second}
        pool.reserve('resumed','disk',len(canonical_bytes(probe_value)))
        begin=time.perf_counter();probe_sha=ge._write_json(probe,probe_value)
        if file_sha256(probe)!=probe_sha:raise ValueError('serialized decode probe readback')
        serialization.append({'kind':'decode_shard_write_fsync_hash_readback','bytes':probe.stat().st_size,'seconds':time.perf_counter()-begin})
        guard(pool)
        value={'schema_version':1,'task_id':'G3','kind':'pilot','status':'engineering_pilot_pass_no_GPU_claim' if engineering else 'pilot_pass_costs_pending_review',
            'context_binding_sha256':gr.digest(context.binding),'runtime':runtime,'microbatch':micro,'restore_exact':True,'restore_worker_pid':child_receipt['pid'],'measured_history_max':maximum,
            'pool_path':str(pool.root.absolute()),'pool_spec':spec,'pool_head':pool.head,'pool_state':pool.state,'attempts':attempts,
            'workload_ordinals_sha256':gr.digest(fixed),'timings':timings,'decode_timings':decode_timings,'serialization_measurements':serialization,'accounting_measurements':pool.accounting_measurements,'resource_observations':observations,
            'decode_state_sha256':gr.digest([{k:v for k,v in r.items() if k!='seconds'} for r in first]),
            'formal_training_permitted':False,'test_gate':'closed','boundary':'train-only capacity/restore; not full validation or model selection'}
        path=output/'manifest.json';pin=write_accounted_json(pool,'resumed',path,value,stamp_pool=True)
        return {**value,'manifest_path':path,'manifest_sha256':pin}

def restore_worker(path,pin):
    """Internal same-pool subprocess continuation, explicitly pinned by its owner."""
    import tempfile,time,torch
    from . import generative_real as gr,generative_training as gt
    value=gd._json(gd._checked_file(path,pin))
    keys={'task_id','kind','prepared','context_binding_sha256','runtime','pool_path','pool_spec','pool_head','checkpoint_path','checkpoint_sha256','binding','local_ledger_path','local_ledger_head','fixed_ordinals','steps_remaining','batch','microbatch','cursor_examples','output'}
    if type(value) is not dict or set(value)!=keys or value['task_id']!='G3' or value['kind']!='pilot_restore_worker':raise ValueError('internal pinned continuation spec')
    spec=_spec(value['pool_spec']);runtime=gr.configure_runtime(value['runtime'])
    if spec['kind'] not in ('engineering_pool','pilot_pool') or runtime['mode']!=('engineering' if spec['kind']=='engineering_pool' else 'real'):raise ValueError('pilot continuation mode')
    root=Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        tmp=str(Path(tmp).resolve())
        prepared=value['prepared'];context=gr.load_context(root,Path(prepared['manifest_path']),prepared['manifest_sha256'],Path(tmp))
        if gr.digest(context.binding)!=value['context_binding_sha256'] or spec['binding_sha256']!=gr.digest(context.binding) or gr.digest(value['fixed_ordinals'])!=value['binding']['workload_ordinals_sha256']:raise ValueError('actual worker context/workload identity')
        binding=value['binding'];cfg=binding['effective_config'];model=gt.make_model(context.codec,cfg,device=runtime['device']);optimizer=gt.make_optimizer(model,cfg)
        cursor=gt.EpochCursor(cfg['seed'],value['cursor_examples'])
        state=gt.load_checkpoint(Path(value['checkpoint_path']),value['checkpoint_sha256'],model,optimizer,cursor,expected_binding=binding)
        parent=ge.BudgetLedger.read_existing(Path(value['local_ledger_path']),value['local_ledger_head'],binding)
        budget=parent['budget'];budget['parent_ledger_head']=value['local_ledger_head']
        if state.budget!=budget or not 1<=value['steps_remaining']<=8 or state.budget['completed_steps']+value['steps_remaining']!=state.budget['max_optimizer_steps']:raise ValueError('exact remaining local continuation budget')
        output=gd._no_symlink(value['output']);output.mkdir(mode=0o700,exist_ok=False)
        with ExecutionPool.open(Path(value['pool_path']),spec,value['pool_head']) as pool:
            pool.resume();ledger=accounted_ledger(pool,'resumed').resume(Path(value['local_ledger_path']),value['local_ledger_head'],binding,output/'ledger');timings=[]
            for _ in range(value['steps_remaining']):
                pool.check();gr.runtime_snapshot(runtime)
                if runtime['mode']=='real':observe_resources(output,runtime['device'])
                pool.reserve('resumed','steps');ledger.reserve_step();indices=cursor.next_batch(value['batch'])
                fixed=value['fixed_ordinals'];rows=context.examples.rows([fixed[i%len(fixed)] for i in indices]);begin=time.perf_counter()
                gt.train_effective_batch(model,optimizer,rows,context.codec,microbatch=value['microbatch'],device=runtime['device'])
                if runtime['mode']=='real':torch.cuda.synchronize()
                elapsed=time.perf_counter()-begin;ledger.complete_step(elapsed);pool.charge_wall('resumed',elapsed);pool.bind_child('resumed',ledger.root,ledger.head)
                timings.append({'phase':'resumed','d_model':cfg['d_model'],'seconds':elapsed,'examples':len(rows)})
            checkpoint=output/'restored.pt'
            cksha,measurement=save_accounted_checkpoint(pool,'resumed',checkpoint,model,optimizer,cursor,binding=binding,selection=state.selection,budget=ledger.snapshot())
            return {'pool_head':pool.head,'checkpoint_path':str(checkpoint.absolute()),'checkpoint_sha256':cksha,'pid':os.getpid(),'timings':timings,'accounting_measurements':pool.accounting_measurements,'serialization_measurements':[measurement]}

if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser(description='Internal G3 pinned pilot continuation')
    parser.add_argument('--restore-worker',type=Path,required=True);parser.add_argument('--sha256',required=True)
    args=parser.parse_args();print(json.dumps(restore_worker(args.restore_worker,args.sha256),sort_keys=True))
