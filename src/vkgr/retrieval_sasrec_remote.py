"""P4H real runtime primitives. No synthetic lifecycle or implicit submission.

The adapter is deliberately separate from the frozen local operation proof.
Only a reviewed, SHA-bound real operation may consume these primitives.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

if os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')!=':4096:8':raise ValueError('frozen deterministic cuBLAS workspace required')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

from . import retrieval_sasrec_operation as operation
from . import retrieval_sasrec_plan as plan


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def read_bound(path,expected_sha256,max_bytes):
    """Hash and parse the same nofollow FD; reject moving or oversized bytes."""
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts or not operation._pin(expected_sha256) or type(max_bytes) is not int or max_bytes<1:
        raise ValueError('bound read metadata')
    for parent in (path,*path.parents):
        if stat.S_ISLNK(parent.lstat().st_mode):raise ValueError('symlink input')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size>max_bytes:raise ValueError('bounded regular input required')
        chunks=[];remaining=before.st_size+1
        while remaining:
            block=os.read(fd,min(remaining,1048576))
            if not block:break
            chunks.append(block);remaining-=len(block)
        data=b''.join(chunks);after=os.fstat(fd)
        if len(data)!=before.st_size or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) or sha(data)!=expected_sha256:
            raise ValueError('input hash/identity drift')
        return data
    finally:os.close(fd)


def write_once(path,value):
    path=Path(path);data=canonical(value)
    if not path.parent.is_dir() or path.parent.is_symlink():raise ValueError('existing owned output directory required')
    temporary=path.with_name('.'+path.name+f'.{os.getpid()}.{time.time_ns()}')
    try:
        with temporary.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if temporary.exists():temporary.unlink()
    return sha(data)


PILOT_RECOVERY_PARENT_SHA256 = 'e80524fc01ce33718dc7c3789af602443dd8a24afc9efc08202b20aeb56064f0'
PILOT_RECOVERY_FAILURE_SHA256S = ('896a363523667287a5bc28587c05f483087eb6f75f8feac16ed3ac3cb16bed32','90c096edf0163b749fba83787f002709ba97d14c7041435846d95e674c08f90c')
PILOT_RECOVERY_GRANT_SHA256 = 'f629751010724398ccb40434a0c1ac1717541010e4e519d84c47dff20658b230'


AUTONOMOUS_GRANT_SHA256 = 'f0d487472254b19087e7144bd57d191ba0c335473150bdd4a46cd65021de186a'
AUTONOMOUS_PREVIOUS_DRIVER_SHA256 = '63f98bddecea9aa011388948586dae508e96dcc6c4f3488f08e213cfdb06c150'
AUTONOMOUS_PREVIOUS_CLOSURE_SHA256 = 'bcf3bee934caf3e475dbac63ace1a9d71f63d74250864ebfa6e35bbbc446580e'


def autonomous_extension(extension):
    return isinstance(extension,dict) and extension.get('schema_id')=='p4h-autonomous-pilot-continuation-v1'


def autonomous_attempt_number(extension):
    grant=extension.get('human_grant')
    if not isinstance(grant,dict) or not isinstance(grant.get('path'),str):raise ValueError('autonomous grant path required')
    name=Path(grant['path']).parent.name
    if name not in ('attempt04','attempt05','attempt06','attempt07'):raise ValueError('reviewed autonomous generation required')
    return int(name[-2:])


def validate_autonomous_descriptor(extension):
    number=autonomous_attempt_number(extension)
    fields={'schema_id','parent_queue','human_grant','previous_driver','previous_closure'}|({'late_quiescence'} if number==7 else set())
    if set(extension)!=fields:raise ValueError('autonomous continuation fields')
    refs={k:extension[k] for k in fields-{'schema_id'}}
    if any(not isinstance(r,dict) or set(r)!={'path','sha256'} or not _absolute(r['path']) or not operation._pin(r['sha256']) for r in refs.values()):raise ValueError('autonomous continuation references')
    number=autonomous_attempt_number(extension)
    chain={4:('6db7532096823748ec9bc26bfd0c6e12a8322b2b68d26c193af55854f4ae6598',AUTONOMOUS_PREVIOUS_DRIVER_SHA256,AUTONOMOUS_PREVIOUS_CLOSURE_SHA256),5:('79639fa09a53fc373cf8ed90495c930132b63f65ba7c8cf548b27a446c669d6d','c00255718358fea03b24657b608ad5b26466b12b39c53e886bf721f5d9a050f5','a66187c6158a4712ce1ba7d771518048ffc7941a00665524d350335815c477af'),6:('8f3e09105a06e6da73f6b2488dfb4ed27c9cb4ca50a4b94045c8a8e05a228721','89ea255c325b2f913f5df271cf1b46f946037077a012d4aabdbbfc00dcfcf09a','4b8a4c51ee6dca17a5aa42eef06bb5c51ac5ec3a47a384fb0c6fb59851a31446'),7:('7a98d94d2b0fa60d9be300b823cabfaded653f6f484153a9832eb028fc76516e','6e20a00ace4b45bcceff42478e2d385bc2ad8be9465f6cd1d73df98dc6acb5d4','f49e40b3e619634801d87a94c037ead258e2ce789db3def22aec375c3f634cfd')}[number]
    expected={'parent_queue':chain[0],'human_grant':AUTONOMOUS_GRANT_SHA256,'previous_driver':chain[1],'previous_closure':chain[2]}
    if number==7:expected['late_quiescence']='18cce7cf2cda93d76e13e3e5a17ff022335346a7f94d7a6264a696102d91eeb9'
    if any(refs[k]['sha256']!=v for k,v in expected.items()):raise ValueError('autonomous direct grant and spent pins')
    root=Path(refs['parent_queue']['path']).parents[1]
    parent=root/('extension-03' if number==4 else f'attempt{number-1:02d}');previous=parent/'preworker02' if number==4 else parent
    paths={'parent_queue':parent/'queue.json','previous_driver':previous/'driver-result.json','previous_closure':previous/'pilot-operation.driver-closure.json','human_grant':root/f'attempt{number:02d}/autonomous-grant.json'}
    if number==7:paths['late_quiescence']=root/'attempt06/late-publisher-quiescence.json'
    if any(refs[k]['path']!=str(v) for k,v in paths.items()):raise ValueError('autonomous original charged ancestry')
    return root


def verify_late_publisher_quiescence(queue,failed,closed):
    ext=queue['extension'];root=validate_autonomous_descriptor(ext);prior=root/'attempt06'
    cert=_read_json(ext['late_quiescence']['path'],pin=ext['late_quiescence']['sha256'])
    pins={'driver-result.json':ext['previous_driver']['sha256'],'pilot-operation.driver-closure.json':ext['previous_closure']['sha256'],'feed/publisher-closure.json':'149d03ffd2e66b831e1fe223c4ce5769125c7c51cc5be9df461e7052a4f29b1a','feed/descriptor.json':'34d3b0d29bf03a7b699b6503f86df0ec25df93891c419dd3cc2387765a5f5eac','driver-intent.json':'7dc0af92820f57126b9d020ec7d7f110b343ee086d1c13d7accacc34702b3bff'}
    refs={name:{'path':str(prior/name),'sha256':pin} for name,pin in pins.items()}
    if cert.get('schema_id')!='p4h-observed-late-publisher-quiescence-v1' or cert.get('previous_records')!=refs or cert.get('original_status_preserved')!='failed+spent' or cert.get('original_publisher_closure_preserved')!='FAILED_UNSAFE' or cert.get('original_parent_wait')!='UNKNOWN_NOT_RECOVERABLE' or cert.get('actual_cooperative_closed') is not True or cert.get('fresh_owned_processes')!=[] or cert.get('fresh_gpu_pids')!=[] or cert.get('original_bytes_and_spent_preserved') is not True or cert.get('new_science_submit') is not False:raise ValueError('actual late quiescence; original wait must stay UNKNOWN')
    pub=failed['publisher_driver_closure']
    if pub.get('state')!='FAILED_UNSAFE' or pub.get('failure')!='ProcessLookupError: [Errno 3] No such process' or pub.get('actual_worker_exit') is not None or pub.get('signals_sent')!=0:raise ValueError('only pinned exited-child bookkeeping failure')
    descriptor=_read_json(refs['feed/descriptor.json']['path'],pin=refs['feed/descriptor.json']['sha256']);receipt=_read_json(refs['feed/publisher-closure.json']['path'],pin=refs['feed/publisher-closure.json']['sha256']);intent=_read_json(refs['driver-intent.json']['path'],pin=refs['driver-intent.json']['sha256'])
    expected=descriptor['publisher_process'];identity={'pid':expected['pid'],'birth':expected['start_ticks'],'pgid':expected['pid'],'sid':expected['pid']}
    if receipt.get('status')!='CLOSED' or receipt.get('reason')!='cooperative_stop' or receipt.get('lock_fd_closed') is not True or receipt.get('descriptor_sha256')!=refs['feed/descriptor.json']['sha256'] or receipt.get('publisher_process')!=expected or not any(_identity(row)==_identity(identity) and row['ppid']==intent['identity']['pid'] for row in pub.get('owned_identities',[])):raise ValueError('actual bound late publisher cooperative closure')
    sessions=set(closed['owned_sessions'])|set(pub['owned_sessions'])|{intent['identity']['sid']}
    if cert.get('fresh_owned_sessions_absent')!=sorted(sessions) or cert.get('host_boot_id')!=failed['queue']['host_boot_id'] or not _finite(cert.get('observed_at')) or cert['observed_at']<failed['finished_at'] or not _finite(cert.get('observed_monotonic')) or queue['started_at']<cert['observed_at'] or queue['started_monotonic']<cert['observed_monotonic'] or any(row['sid'] in sessions for row in linux_processes()):raise ValueError('fresh old session absence/clock required')
    return True


def verify_autonomous_continuation(queue):
    ext=queue['extension'];root=validate_autonomous_descriptor(ext);number=autonomous_attempt_number(ext)
    grant=_read_json(ext['human_grant']['path'],pin=ext['human_grant']['sha256'])
    envelope={'next_cumulative_attempt':4,'new_wall_seconds':14400,'total_old_plus_new_disk_bytes':20*1024**3,'resume':0,'new_resources':False,'formal_submit':False}
    if grant.get('schema_id')!='pgr01-human-autonomous-continue-v1' or grant.get('human_answer')!='不用管限制，一直做下去' or grant.get('preserve_old_spent_and_artifact_bytes') is not True or grant.get('operational_envelope')!=envelope:raise ValueError('direct autonomous human authority')
    Budget(queue['kind'],**{k:queue[k] for k in ('started_at','deadline','disk_cap','attempts','extension')})
    if queue['attempts']!=number or 'clock_extension' in queue or queue['receipt_path']!=str(root/f'attempt{number:02d}/queue.json'):raise ValueError('new immutable finite attempt4 queue')
    failed=_read_json(ext['previous_driver']['path'],pin=ext['previous_driver']['sha256'])
    closed=_read_json(ext['previous_closure']['path'],pin=ext['previous_closure']['sha256'])
    previous=failed['queue'];verify_explicit_extension(previous);verify_clock_extension(previous)
    pub=failed.get('publisher_driver_closure',{})
    publisher_ok=verify_late_publisher_quiescence(queue,failed,closed) if number==7 else failed.get('cleanup_failures')==[] and pub.get('state')=='CLOSED' and pub.get('actual_worker_exit')==0
    if failed.get('status')!='failed+spent' or failed.get('stages')!=[] or failed.get('formal_submit') is not False or closed.get('state')!='CLOSED' or type(closed.get('actual_worker_exit')) is not int or not publisher_ok or not closed.get('owned_sessions') or not pub.get('owned_sessions'):raise ValueError('previous actual failed spent owned CLOSED required')
    if previous['receipt_path']!=ext['parent_queue']['path'] or previous['receipt_sha256']!=ext['parent_queue']['sha256'] or queue['host_boot_id']!=previous['host_boot_id'] or queue['disk_cap']!=previous['disk_cap'] or queue['started_at']<failed['finished_at'] or queue['started_monotonic']<previous['started_monotonic']:raise ValueError('preserve previous queue/spent/boot/disk')
    sessions=set(closed['owned_sessions'])|set(pub['owned_sessions']);driver_pids={p['ppid'] for p in closed['owned_identities']+pub['owned_identities'] if p['ppid'] not in {r['pid'] for r in closed['owned_identities']+pub['owned_identities']}}
    if any(p['sid'] in sessions or p['pid'] in driver_pids for p in linux_processes()):raise ValueError('previous actual processes not closed')
    return {'cumulative_attempts':number,'charged_root':str(root),'old_spent_preserved':True,'human_grant_sha256':ext['human_grant']['sha256']}


def validate_extension_descriptor(extension):
    if autonomous_extension(extension):return validate_autonomous_descriptor(extension)
    fields={'schema_id','parent_queue','failure_receipts','human_grant'}
    if not isinstance(extension,dict) or set(extension)!=fields or extension['schema_id']!='p4h-explicit-pilot-extension-v1':raise ValueError('explicit versioned extension authority required')
    refs=[extension['parent_queue'],extension['human_grant']]
    if not isinstance(extension['failure_receipts'],list) or len(extension['failure_receipts'])!=2:raise ValueError('both previous spent attempts required')
    refs+=extension['failure_receipts']
    if any(not isinstance(r,dict) or set(r)!={'path','sha256'} or not _absolute(r['path']) or not operation._pin(r['sha256']) for r in refs):raise ValueError('extension external references')
    if extension['parent_queue']['sha256']!=PILOT_RECOVERY_PARENT_SHA256 or extension['human_grant']['sha256']!=PILOT_RECOVERY_GRANT_SHA256 or tuple(r['sha256'] for r in extension['failure_receipts'])!=PILOT_RECOVERY_FAILURE_SHA256S:raise ValueError('unapproved extension authority pins')
    root=Path(extension['parent_queue']['path']).parent
    if len({r['path'] for r in refs})!=4 or any(not str(Path(r['path'])).startswith(str(root)+'/') for r in refs[1:]):raise ValueError('extension ancestry or alias')
    return root


def queue_charge_root(queue):
    if 'extension' in queue:
        return validate_extension_descriptor(queue['extension'])
    return Path(queue['receipt_path']).parent


def verify_explicit_extension(queue):
    if autonomous_extension(queue.get('extension')):return verify_autonomous_continuation(queue)
    ext=queue.get('extension');root=validate_extension_descriptor(ext)
    if queue['kind']!='pilot' or queue['attempts']!=3 or not str(Path(queue['receipt_path'])).startswith(str(root)+'/extension-03/') or Path(queue['receipt_path'])==Path(ext['parent_queue']['path']):raise ValueError('single cumulative attempt3 in original charged tree')
    parent=_read_json(ext['parent_queue']['path'],pin=ext['parent_queue']['sha256'])
    Budget(parent['kind'],**{k:parent[k] for k in ('started_at','deadline','disk_cap','attempts')})
    grant=_read_json(ext['human_grant']['path'],pin=ext['human_grant']['sha256'])
    expected={'additional_attempts':1,'cumulative_attempt_cap':3,'additional_wall_seconds':14400,'cumulative_authorized_wall_cap_seconds':28800,'total_disk_bytes':20*1024**3,'preserve_old_spent':True}
    if grant.get('schema_id')!='pg-r01-human-continue-v1' or grant.get('pilot')!=expected:raise ValueError('human grant finite scope drift')
    if parent['kind']!='pilot' or parent['attempts']!=1 or queue['started_at']<parent['deadline'] or queue['started_monotonic']<parent['deadline_monotonic'] or queue['host_boot_id']!=parent['host_boot_id'] or queue['disk_cap']!=parent['disk_cap']:raise ValueError('parent clock/boot/disk or spent preservation')
    for number,ref in enumerate(ext['failure_receipts'],1):
        failure=_read_json(ref['path'],pin=ref['sha256'])
        old=failure.get('queue',{});closure=failure.get('driver_closure',{})
        if failure.get('status')!='failed+spent' or failure.get('failure')!='ValueError: real publisher bootstrap failed' or failure.get('terminal') is not None or old.get('attempts')!=number or closure.get('state')!='CLOSED' or closure.get('actual_worker_exit')!=-15 or any(old.get(k)!=v for k,v in parent.items() if k!='attempts') or not _finite(failure.get('finished_at')) or failure['finished_at']>queue['started_at']:raise ValueError('pinned bootstrap failure/spent evidence drift')
    return {'cumulative_attempts':3,'parent_queue_sha256':ext['parent_queue']['sha256'],'human_grant_sha256':ext['human_grant']['sha256'],'charged_root':str(root),'old_spent_preserved':True}


CLOCK_EXTENSION_GRANT_SHA256 = 'ee076adadab817f4af2efe1dbd8a098e05cf24fb53a354272409cfd8619e4fcd'


def clock_extension_seconds(clock):
    if clock is None:return 0
    if not isinstance(clock,dict) or set(clock)!={'schema_id','additional_seconds','original_queue','human_grant','receipt'} or clock['schema_id']!='p4h-explicit-clock-extension-v1' or clock['additional_seconds']!=7200:raise ValueError('explicit fixed 7200s clock authority required')
    for key in ('original_queue','human_grant','receipt'):
        ref=clock[key]
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not _absolute(ref['path']) or not operation._pin(ref['sha256']):raise ValueError('clock authority reference')
    if clock['human_grant']['sha256']!=CLOCK_EXTENSION_GRANT_SHA256:raise ValueError('unapproved clock grant')
    return 7200


def verify_clock_extension(queue):
    if autonomous_extension(queue.get('extension')):return verify_autonomous_continuation(queue)
    clock=queue.get('clock_extension');extra=clock_extension_seconds(clock)
    if not extra:return None
    if queue['kind']!='pilot' or queue['attempts']!=3 or 'extension' not in queue or clock['original_queue']!={'path':queue['receipt_path'],'sha256':queue['receipt_sha256']}:raise ValueError('clock only extends exact original pilot attempt3')
    root=queue_charge_root(queue)
    if any(not ref['path'].startswith(str(root)+'/') for ref in (clock['human_grant'],clock['receipt'])):raise ValueError('clock authority charged root')
    base={k:v for k,v in queue.items() if k not in ('receipt_path','receipt_sha256','clock_extension')}
    if _read_json(queue['receipt_path'],pin=queue['receipt_sha256'])!=base:raise ValueError('clock cannot mutate original queue/start/caps')
    grant=_read_json(clock['human_grant']['path'],pin=clock['human_grant']['sha256'])
    expected={'schema_id':'pgr01-human-clock-extension-v1','human_answer':'批准有限追加2h','original_queue':clock['original_queue'],'original_started_at':queue['started_at'],'original_deadline':queue['deadline'],'additional_wall_seconds':extra,'effective_deadline':queue['deadline']+extra,'attempt_cap':3,'additional_science_attempts':0,'disk_cap':queue['disk_cap'],'preserve_original_bytes_starts_spent':True,'no_science_retry_after_release':True}
    if any(grant.get(k)!=v for k,v in expected.items()):raise ValueError('explicit same-start/spent finite clock grant required')
    receipt={'schema_id':'p4h-explicit-clock-receipt-v1','original_queue':clock['original_queue'],'human_grant':clock['human_grant'],'additional_seconds':extra,'original_started_at':queue['started_at'],'original_started_monotonic':queue['started_monotonic'],'original_deadline':queue['deadline'],'original_deadline_monotonic':queue['deadline_monotonic'],'effective_deadline':queue['deadline']+extra,'effective_deadline_monotonic':queue['deadline_monotonic']+extra,'host_boot_id':queue['host_boot_id'],'attempts':3,'disk_cap':queue['disk_cap']}
    if _read_json(clock['receipt']['path'],pin=clock['receipt']['sha256'])!=receipt:raise ValueError('frozen additive clock receipt drift')
    return receipt


class Budget:
    """Continuous authorized queue clock and observed-disk STOP envelope."""
    def __init__(self,kind,*,started_at,deadline,disk_cap,attempts,extension=None,clock_extension=None):
        limits={'pilot':(14400,20*1024**3),'formal':(50400,32*1024**3)}
        autonomous=autonomous_extension(extension)
        ceiling=autonomous_attempt_number(extension) if autonomous else 3
        extended=kind=='pilot' and attempts==ceiling and extension is not None
        if extension is not None:
            if not extended:raise ValueError('only explicit cumulative pilot attempt3')
            validate_extension_descriptor(extension)
        if kind not in limits or not _finite(started_at) or not _finite(deadline) or not 0<deadline-started_at<=limits[kind][0] or type(disk_cap) is not int or not 0<disk_cap<=limits[kind][1] or type(attempts) is not int or not 1<=attempts<=(ceiling if extended else 2):
            raise ValueError('authorized finite ceiling')
        extra=clock_extension_seconds(clock_extension)
        if extra and not extended:raise ValueError('clock cannot extend other queue/attempt')
        self.kind=kind;self.started_at=started_at;self.deadline=deadline+extra;self.disk_cap=disk_cap;self.attempts=attempts

    def remaining_seconds(self,now):
        if not _finite(now) or now<self.started_at:raise ValueError('budget clock rollback')
        return max(0,self.deadline-now)

    def next_attempt(self,*,now,reason,optimizer_steps):
        if self.attempts!=1 or self.remaining_seconds(now)<=0 or type(optimizer_steps) is not int or optimizer_steps<0 or reason not in ('transport_before_step0','infrastructure_failure') or reason=='transport_before_step0' and optimizer_steps!=0:
            raise ValueError('retry not authorized')
        return Budget(self.kind,started_at=self.started_at,deadline=self.deadline,disk_cap=self.disk_cap,attempts=2)

    def require_stage(self,*,now,stage_seconds,finish_seconds,used_bytes,next_atomic_bytes,closure_bytes,free_fractions):
        durations=(stage_seconds,finish_seconds)
        if any(not _finite(v) or v<0 for v in durations) or stage_seconds+finish_seconds>self.remaining_seconds(now):raise ValueError('full completion clock reserve')
        sizes=(used_bytes,next_atomic_bytes,closure_bytes)
        if any(type(v) is not int or v<0 for v in sizes) or sum(sizes)>self.disk_cap:raise ValueError('atomic old+new disk reserve')
        if not isinstance(free_fractions,dict) or set(free_fractions)!={'ram','vram','disk'} or any(not _finite(v) or not .2<=v<=1 for v in free_fractions.values()):raise ValueError('shared headroom stop')


def measure_action(name,body,*,artifacts,sink):
    """Measurements derive only from an executed action and its actual files."""
    from .retrieval_sasrec_experiment import PILOT_MEASUREMENTS
    if name not in PILOT_MEASUREMENTS or not callable(body) or not callable(sink) or not isinstance(artifacts,list):raise ValueError('actual measured action required')
    started=time.monotonic();result=body();elapsed=time.monotonic()-started;rows=[]
    for path in artifacts:
        path=Path(path);size=path.stat().st_size
        digest=operation._file_sha256(path)
        read_bound(path,digest,max(1,size))
        rows.append({'path':str(path),'size':size,'sha256':digest})
    record={'name':name,'seconds':elapsed,'bytes':sum(r['size'] for r in rows),'artifacts':rows}
    sink(record);return result,record


def _absolute(path):
    return isinstance(path,str) and path.startswith('/') and '..' not in path.split('/') and not any(part.startswith('synthetic') or part=='server-only' for part in path.split('/'))


def validate_remote_manifest(value):
    """Pure metadata gate. No stat, resolve, hash or read is allowed here."""
    fields={'schema_id','operation_id','stage','roles','code_root','python','source_pins','python_sha256','output_root','control_root','authorization','queue','limits','g_feed','slot','predecessor'}
    if not isinstance(value,dict) or set(value)!=fields or value['schema_id']!='p4h-real-operation-v1' or not isinstance(value['operation_id'],str) or not value['operation_id'].startswith('p4h-'):
        raise ValueError('real operation envelope required')
    stage=value['stage']
    if stage not in ('pilot','pilot-verify','formal-train','formal-verify'):raise ValueError('unimplemented or unauthorized remote stage')
    if stage=='pilot':
        plan.validate_pilot_manifest({'stage':'pilot','roles':value['roles'],'exclusive_output_root':value['output_root'],'exclusive_control_root':value['control_root']})
    elif stage=='formal-train':
        operation._validate_exact_stage_roles(stage,value['roles'])
    else:
        plan.validate_verifier_manifest({'stage':stage,'roles':value['roles']},value['predecessor'])
    if stage in ('pilot','formal-train') and value['predecessor'] is not None:raise ValueError('mutating predecessor boundary')
    if stage.startswith('formal') and value['slot'] not in plan.FORMAL_SLOTS:raise ValueError('unregistered slot')
    if stage.startswith('pilot') and value['slot'] is not None:raise ValueError('pilot has no quality slot')
    roots=[value[k] for k in ('code_root','python','output_root','control_root')]
    if not all(_absolute(p) for p in roots) or len(set(roots))!=4:raise ValueError('absolute distinct real roots')
    for new in roots[2:]:
        for existing in roots:
            if new!=existing and (new.startswith(existing.rstrip('/')+'/') or existing.startswith(new.rstrip('/')+'/')):raise ValueError('overlapping root')
    for role in value['roles']:
        if not _absolute(role['path']) or any(role['path']==p or role['path'].startswith(p+'/') for p in roots[2:]):raise ValueError('role path overlaps output')
    pins=value['source_pins']
    if not isinstance(pins,dict) or not pins or len(pins)>128 or any(not isinstance(k,str) or k.startswith('/') or '..' in k.split('/') or not operation._pin(v) for k,v in pins.items()) or pins.get('src/vkgr/retrieval_sasrec.py')!='301b7086af2a2ada58e28c4440d18c880bdd50ddf7ccafc19fd470dcfb80f00c' or not operation._pin(value['python_sha256']):raise ValueError('frozen real source/runtime pins')
    queue=value['queue']
    if not isinstance(queue,dict) or set(queue) not in ({'kind','started_at','deadline','disk_cap','attempts','started_monotonic','deadline_monotonic','host_boot_id','receipt_path','receipt_sha256'},{'kind','started_at','deadline','disk_cap','attempts','started_monotonic','deadline_monotonic','host_boot_id','receipt_path','receipt_sha256','extension'},{'kind','started_at','deadline','disk_cap','attempts','started_monotonic','deadline_monotonic','host_boot_id','receipt_path','receipt_sha256','extension','clock_extension'}):raise ValueError('queue envelope')
    Budget(queue['kind'],**{k:queue[k] for k in ('started_at','deadline','disk_cap','attempts')},extension=queue.get('extension'),clock_extension=queue.get('clock_extension'))
    if not all(_finite(queue[k]) for k in ('started_monotonic','deadline_monotonic')) or queue['deadline_monotonic']-queue['started_monotonic']!=queue['deadline']-queue['started_at'] or not isinstance(queue['host_boot_id'],str) or not queue['host_boot_id'] or not _absolute(queue['receipt_path']) or not operation._pin(queue['receipt_sha256']):raise ValueError('frozen monotonic queue receipt')
    if queue['kind']!=('pilot' if stage.startswith('pilot') else 'formal'):raise ValueError('queue/stage mismatch')
    limits=value['limits'];expected={'sample_seconds','seal_seconds','signal_grace_seconds','stop_reserve_bytes','stage_seconds','finish_seconds','next_atomic_bytes'}
    if not isinstance(limits,dict) or set(limits)!=expected or any(type(v) is not int or v<=0 for v in limits.values()) or limits['sample_seconds']>5 or limits['seal_seconds']>120 or limits['signal_grace_seconds']>30:raise ValueError('finite runtime stop envelope')
    auth=value['authorization']
    if not isinstance(auth,dict) or set(auth)!={'path','sha256','issued_at','expires_at','stages'} or not _absolute(auth['path']) or not operation._pin(auth['sha256']) or not _finite(auth['issued_at']) or not _finite(auth['expires_at']) or not 0<auth['expires_at']-auth['issued_at']<=300 or not isinstance(auth['stages'],list) or stage not in auth['stages'] or any(s not in ('pilot','pilot-verify','formal-train','formal-verify') for s in auth['stages']):raise ValueError('live bounded authorization')
    feed=value['g_feed']
    if not isinstance(feed,dict) or set(feed)!={'path','sha256'} or not _absolute(feed['path']) or not operation._pin(feed['sha256']):raise ValueError('real G feed required')
    queue_root=str(queue_charge_root(queue))
    for charged in (value['code_root'],value['output_root'],value['control_root'],value['authorization']['path'],value['g_feed']['path']):
        if not charged.startswith(queue_root.rstrip('/')+'/'):raise ValueError('charged path outside original queue tree')
    minimum_finish=limits['sample_seconds']+limits['seal_seconds']+3*limits['signal_grace_seconds']+90
    if limits['finish_seconds']<minimum_finish:raise ValueError('cleanup probe/seal/grace/final reserve insufficient')
    return {'stage':stage,'operation_id':value['operation_id'],'static_verified_before_path':True}


def make_admission(value,*,manifest_sha256,start_sha256,g_receipt_sha256,now):
    validate_remote_manifest(value);auth=value['authorization'];queue=value['queue']
    if not _finite(now) or not auth['issued_at']<=now<auth['expires_at'] or not all(operation._pin(p) for p in (manifest_sha256,start_sha256,g_receipt_sha256)):raise ValueError('admission pins/window')
    return {'schema_id':'p4h-real-admission-v1','operation_id':value['operation_id'],'manifest_sha256':manifest_sha256,'start_sha256':start_sha256,'g_receipt_sha256':g_receipt_sha256,'authorization_sha256':auth['sha256'],'deadline':queue['deadline']+clock_extension_seconds(queue.get('clock_extension')),'disk_cap':queue['disk_cap'],'queue_started_at':queue['started_at'],'attempt':queue['attempts'],'limits':dict(value['limits']),'queue':dict(queue),'issued_at':now,'expires_at':auth['expires_at']}


def validate_admission(value,admission,*,manifest_sha256,start_sha256,g_receipt_sha256,now):
    if not isinstance(admission,dict) or not _finite(admission.get('issued_at')) or not admission['issued_at']<=now:raise ValueError('admission time')
    expected=make_admission(value,manifest_sha256=manifest_sha256,start_sha256=start_sha256,g_receipt_sha256=g_receipt_sha256,now=admission['issued_at'])
    if admission!=expected or not _finite(now) or now>=admission['expires_at']:raise ValueError('admission binding/window drift')
    return True


def _save_tensors_once(path,value):
    import torch
    path=Path(path);temporary=path.with_name('.'+path.name+f'.{os.getpid()}.{time.time_ns()}')
    try:
        with temporary.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if temporary.exists():temporary.unlink()
    return {'path':str(path),'sha256':operation._file_sha256(path),'size':path.stat().st_size}


def train_model(model,optimizer,batches,idmap,slot,*,evaluate,output,binding,gate):
    """Adapt frozen science to actual, hash-bound tensor files.

    This is worker science only: its result is unverified until an independent
    verifier operation has closed. The controller owns admission and closure.
    """
    from . import retrieval_sasrec_experiment as science
    if not callable(gate):raise ValueError('runtime safety gate required')
    gate();output=Path(output)
    if not output.is_dir() or output.is_symlink() or list(output.iterdir()):raise ValueError('exclusive empty worker output required')
    events=[]
    def emit(kind,value):gate();events.append({'kind':kind,'value':value})
    def evaluate_bound(current,epoch):
        gate();result=evaluate(current,epoch);gate();return result
    state=science.train_slot(model,optimizer,batches,idmap,slot,evaluate=evaluate_bound,sink=lambda *_:None,operation_sink=emit)
    if state['public']['status']!='completed+unverified':raise ValueError('incomplete science result')
    gate();tensor_record=_save_tensors_once(output/'state.pt',{'binding':binding,'slot':slot,'state':state})
    gate();write_once(output/'model-artifacts.json',{'schema_id':'p4h-real-model-artifacts-v1','binding':binding,'slot':slot,'state':tensor_record,'public':state['public'],'science_events':events,'test_gate':'closed'})
    return state


def verify_model(predecessor,model,batches,idmap,slot,*,evaluate,binding,gate,artifact_index_sha256=None):
    """Read-only science replay; no optimizer, exactly one fresh best evaluation.

    The surrounding verifier controller must independently bind and seal this
    record. This function never claims process closure or remote success.
    """
    import io
    import torch
    from . import retrieval_sasrec_experiment as science
    if not operation._pin(artifact_index_sha256):raise ValueError('externally pinned CLOSED predecessor artifact index required')
    gate();root=Path(predecessor)
    if root.is_symlink():raise ValueError('predecessor root symlink')
    index_path=root/'model-artifacts.json'
    index=json.loads(read_bound(index_path,artifact_index_sha256,64*1024**2))
    if index.get('schema_id')!='p4h-real-model-artifacts-v1' or index.get('binding')!=binding or index.get('slot')!=slot or index.get('test_gate')!='closed':raise ValueError('model binding mismatch')
    rec=index['state'];expected=root/'state.pt'
    if set(rec)!={'path','sha256','size'} or rec['path']!=str(expected) or type(rec['size']) is not int or not 0<rec['size']<=8*1024**3:raise ValueError('tensor artifact identity')
    blob=read_bound(expected,rec['sha256'],rec['size']);gate()
    loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
    if set(loaded)!={'binding','slot','state'} or loaded['binding']!=binding or loaded['slot']!=slot or loaded['state']['public']!=index['public']:raise ValueError('tensor/public binding mismatch')
    state=loaded['state'];public=state['public']
    if science._tensor_digest(state['private']['best_model_state'])!=public['checkpoint_roles']['best'] or science._tensor_digest(state['private']['model_state'])!=public['checkpoint_roles']['last']:raise ValueError('best/last tensor hash mismatch')
    best_epoch=public['selection']['best_epoch']
    def fresh(best_state,config):
        gate();model.load_state_dict(best_state);model.eval();result=evaluate(model,best_epoch);gate();return result
    return science.verify_slot(state,batches[:public['steps']],idmap,fresh_export_exact=fresh,sink=lambda *_:None,operation_sink=lambda *_:gate())


def parse_proc_stat(text):
    end=text.rfind(')');start=text.find('(')
    if start<1 or end<start:raise ValueError('proc stat schema')
    fields=text[end+2:].split()
    try:return {'pid':int(text[:start].strip()),'ppid':int(fields[1]),'pgid':int(fields[2]),'sid':int(fields[3]),'birth':int(fields[19]),'state':fields[0]}
    except (ValueError,IndexError) as error:raise ValueError('proc birth identity unavailable') from error


def _identity(row):
    return {k:row[k] for k in ('pid','pgid','sid','birth')}


def descendants(leader,processes):
    actual=next((r for r in processes if r['pid']==leader['pid']),None)
    if actual is None or _identity(actual)!=_identity(leader):raise ValueError('worker birth drift')
    owned={leader['pid']};changed=True
    while changed:
        new={r['pid'] for r in processes if r['ppid'] in owned};changed=not new<=owned;owned|=new
    return [r for r in processes if r['pid'] in owned]


def authorize_group_signal(leader,observed,registered,*,controller_pgid,protected_pgids):
    """Return P's group only after a fresh full birth/ownership inventory."""
    pgid=leader['pgid']
    if pgid!=leader['pid'] or pgid==controller_pgid or pgid in protected_pgids:raise ValueError('protected or nonexclusive PGID')
    live=[r for r in observed if r['pgid']==pgid and r['state']!='Z']
    expected={r['pid']:_identity(r) for r in registered}
    current=next((r for r in live if r['pid']==leader['pid']),None)
    if current is None or _identity(current)!=_identity(leader) or any(expected.get(r['pid'])!=_identity(r) for r in live):raise ValueError('unregistered process or birth drift; zero signal')
    return pgid


def linux_processes():
    root=Path('/proc');entries=list(root.iterdir())
    if len(entries)>8192:raise ValueError('process inventory bound')
    rows=[]
    for p in entries:
        if not p.name.isdigit():continue
        try:
            data=(p/'stat').read_bytes()
            if len(data)>4096:raise ValueError('proc stat bound')
            rows.append(parse_proc_stat(data.decode()))
        # ENOENT/ESRCH both mean the process exited after listdir; any other error stays fail-closed.
        except (FileNotFoundError,ProcessLookupError):continue
    return rows


def linux_identity(pid):
    return parse_proc_stat((Path('/proc')/str(pid)/'stat').read_text())


def validate_safety_payload(payload,runs,*,now,max_age):
    if not isinstance(payload,dict) or set(payload)!={'observed_at','runs','free_fractions','gpu_pids','allowed_gpu_pids','protected_pgids','p_writer_count'} or not _finite(now) or not _finite(max_age) or not 0<max_age<=30 or not _finite(payload['observed_at']) or not 0<=now-payload['observed_at']<=max_age:raise ValueError('stale or invalid safety observation')
    expected={r['run_id']:r for r in runs};observed=payload['runs']
    if not isinstance(observed,list) or len(observed)!=len(expected) or {r.get('run_id') for r in observed}!=set(expected):raise ValueError('G identity set drift')
    for row in observed:
        original=expected[row['run_id']]
        if row.get('start_sha256')!=original['start_sha256']:raise ValueError('G start drift')
        if row.get('status')=='RUNNING':
            if row.get('owner_live') is not True:raise ValueError('G owner lost without terminal')
        elif row.get('status')=='TERMINAL':
            if row.get('terminal_status')!='complete' or row.get('fresh_owned_groups_closed') is not True or not operation._pin(row.get('terminal_sha256')) or original.get('expected_terminal_sha256',row['terminal_sha256'])!=row['terminal_sha256']:raise ValueError('G abnormal or unsafe terminal')
        else:raise ValueError('G status UNKNOWN')
    fractions=payload['free_fractions']
    if not isinstance(fractions,dict) or set(fractions)!={'ram','vram','disk'} or any(not _finite(v) or not .2<=v<=1 for v in fractions.values()):raise ValueError('shared resource STOP')
    for field in ('gpu_pids','allowed_gpu_pids','protected_pgids'):
        if not isinstance(payload[field],list) or len(set(payload[field]))!=len(payload[field]) or any(type(pid) is not int or pid<=0 for pid in payload[field]):raise ValueError('process inventory schema')
    if not set(payload['gpu_pids'])<=set(payload['allowed_gpu_pids']) or type(payload['p_writer_count']) is not int or payload['p_writer_count']!=0:raise ValueError('unknown GPU/P writer')
    return True


def check_payload_binding(receipt,payload):
    if receipt.get('payload_sha256')!=sha(canonical(payload)):raise ValueError('safety payload hash drift')
    return True


def linux_resources():
    import subprocess
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
    lines=gpu.stdout.strip().splitlines()
    if len(lines)!=1:raise ValueError('single authorized GPU required')
    total,free=map(float,lines[0].split(','))
    if not math.isfinite(total) or not math.isfinite(free) or total<=0:raise ValueError('GPU resource unknown')
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
    pids=[int(s.strip()) for s in apps.stdout.splitlines() if s.strip()]
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,_,val=line.partition(':')
        if key in ('MemTotal','MemAvailable'):mem[key]=int(val.split()[0])*1024
    fs=os.statvfs('/workspace')
    return {'vram':free/total,'ram':current_ram_fraction(),'disk':fs.f_bavail/fs.f_blocks},pids


def _read_json(path,cap=65536,pin=None):
    path=Path(path)
    if pin is None:pin=operation._file_sha256(path)
    return json.loads(read_bound(path,pin,cap))


def observe_g(runs,*,allowed_p):
    """G owner safety observation only. Never writes/receives G terminal state."""
    processes=linux_processes();by_pid={r['pid']:r for r in processes};known=[];protected=set();rows=[]
    for fixed in runs:
        root=Path(fixed['root'])
        read_bound(root/'controller.py',fixed['controller_sha256'],262144)
        _read_json(root/'approval.json',pin=fixed['approval_sha256'])
        inp=_read_json(root/'input.json',pin=fixed['input_sha256']);start=_read_json(root/'start.json',pin=fixed['start_sha256']);owner=_read_json(root/'owner.json')
        if any(start[k]!=fixed[k] for k in ('controller_sha256','approval_sha256','input_sha256')) or owner['start_sha256']!=fixed['start_sha256'] or inp['registration']['run_id']!=fixed['run_id']:raise ValueError('G immutable identity drift')
        expected=owner['identity'];live=by_pid.get(expected['pid'])
        owner_live=bool(live and live['state']!='Z' and all(live[k]==expected[k] for k in ('pid','pgid','sid','birth')))
        row={'run_id':fixed['run_id'],'start_sha256':fixed['start_sha256'],'status':'RUNNING' if owner_live else 'UNKNOWN','owner_live':owner_live}
        terminal=root/'terminal.json'
        try:data=read_bound(terminal,operation._file_sha256(terminal),65536)
        except FileNotFoundError:data=b''
        if data.strip():
            value=json.loads(data)
            if any(value[k]!=fixed[k] for k in ('start_sha256','approval_sha256','controller_sha256','input_sha256')):raise ValueError('G terminal binding drift')
            sessions=value.get('cleanup',{}).get('owned_sessions',[])
            row.update(status='TERMINAL',terminal_status=value.get('status'),terminal_sha256=sha(data),fresh_owned_groups_closed=bool(sessions) and not any(p['sid'] in sessions for p in processes))
            protected.update(sessions)
        elif owner_live:
            group=descendants(live,processes);known.extend(p['pid'] for p in group);protected.update(p['pgid'] for p in group)
        rows.append(row)
    p_known={os.getpid()}
    for expected in allowed_p:
        current=by_pid.get(expected['pid'])
        if current and _identity(current)==_identity(expected):p_known.update(r['pid'] for r in descendants(current,processes))
        elif current:raise ValueError('P owner birth drift')
    fractions,gpu_pids=linux_resources()
    # The publisher includes only explicitly registered P ancestry in GPU scope.
    known.extend(p_known-{os.getpid()})
    p_writers=0
    for p in processes:
        if p['pid'] in p_known:continue
        try:data=(Path('/proc')/str(p['pid'])/'cmdline').read_bytes()
        except (FileNotFoundError,ProcessLookupError):continue
        if len(data)>16384:raise ValueError('command inventory bound')
        if b'vkgr-p4h' in data or b'vkgr.retrieval_sasrec_remote' in data:p_writers+=1
    payload={'observed_at':time.time(),'runs':rows,'free_fractions':fractions,'gpu_pids':sorted(set(gpu_pids)),'allowed_gpu_pids':sorted(set(known)),'protected_pgids':sorted(protected),'p_writer_count':p_writers}
    validate_safety_payload(payload,runs,now=time.time(),max_age=30)
    return payload


class FeedJournal:
    """Append-only disk protocol; runtime publisher supplies proven observations."""
    def __init__(self,root,publisher_process,g_job_identity):
        self.root=Path(root);self.root.mkdir(mode=0o700)
        if self.root.is_symlink() or stat.S_IMODE(self.root.stat().st_mode)!=0o700:raise ValueError('owned exclusive feed root')
        self.lock_fd=os.open(self.root/'publisher.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o400)
        lock=os.fstat(self.lock_fd);folder=self.root.stat()
        self.descriptor={'schema_version':1,'feed_root':str(self.root),'feed_root_identity':{'device':folder.st_dev,'inode':folder.st_ino},'feed_owner_uid':os.getuid(),'feed_mode':'0700','head_name':'HEAD.json','receipt_name_pattern':'receipt-<zero-padded-seq>-<sha256>.json','publisher_lock':{'name':'publisher.lock','device':lock.st_dev,'inode':lock.st_ino,'birth_ctime_ns':lock.st_ctime_ns,'owner_uid':lock.st_uid,'creation':'O_CREAT|O_EXCL'},'publisher_process':dict(publisher_process),'initial_no_other_writer_inventory_sha256':None,'g_job_identity':dict(g_job_identity),'receipt_schema_id':'p4h-g-feed-receipt-v2','initial_seq':0,'initial_receipt_sha256':None,'allowed_statuses':['OPEN','CLOSED','OWNER_LOST']}
        self.previous=None;self.seq=0;self.terminal=False

    def publish(self,payload,*,now,status):
        if self.terminal or status not in self.descriptor['allowed_statuses'] or not _finite(now):raise ValueError('terminal feed append or invalid status')
        payload_sha=sha(canonical(payload));path=self.root/('payload-'+payload_sha+'.json')
        if not path.exists():write_once(path,payload);os.chmod(path,0o400)
        else:read_bound(path,payload_sha,262144)
        value={'schema_version':1,'seq':self.seq,'previous_receipt_sha256':self.previous,'g_job_identity':self.descriptor['g_job_identity'],'publisher_process':self.descriptor['publisher_process'],'publisher_lock_birth':{k:self.descriptor['publisher_lock'][k] for k in ('device','inode','birth_ctime_ns','owner_uid')},'issued_at':now,'expires_at':now+30,'next_expected_by':now+45,'status':status,'payload_sha256':payload_sha}
        receipt_sha=sha(canonical(value));name=f'receipt-{self.seq:06d}-{receipt_sha}.json';write_once(self.root/name,value);os.chmod(self.root/name,0o400)
        head={'seq':self.seq,'receipt_name':name,'receipt_sha256':receipt_sha};temporary=self.root/('.HEAD-'+str(time.time_ns()))
        with temporary.open('xb') as f:f.write(canonical(head));f.flush();os.fsync(f.fileno())
        os.replace(temporary,self.root/'HEAD.json');fd=os.open(self.root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if self.seq==0:self.descriptor['initial_receipt_sha256']=receipt_sha
        self.seq+=1;self.previous=receipt_sha;self.terminal=status!='OPEN'
        return {'name':name,'sha256':receipt_sha,'value':value}

    def close(self):
        if self.lock_fd is not None:os.close(self.lock_fd);self.lock_fd=None


def linux_publisher_identity(pid):
    row=linux_identity(pid)
    if row['state']=='Z':raise ValueError('publisher is dead')
    executable=Path('/proc')/str(pid)/'exe'
    base={'pid':pid,'start_ticks':row['birth'],'executable_sha256':operation._file_sha256(executable)}
    return {**base,'start_receipt_sha256':operation.digest(base)}


def linux_writer_inventory(descriptor):
    root=Path(descriptor['feed_root']);expected=descriptor['publisher_process'];writers=[];fd_count=0
    for proc in linux_processes():
        directory=Path('/proc')/str(proc['pid'])/'fd'
        try:entries=list(directory.iterdir())
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            foreign_uid_fd_modes(proc,descriptor)
            continue
        if len(entries)>4096:raise ValueError('per-process fd bound')
        modes=set()
        for fd in entries:
            fd_count+=1
            if fd_count>32768:raise ValueError('total FD inventory bound')
            try:
                target=os.readlink(fd)
                if target!=str(root) and not target.startswith(str(root)+'/'):continue
                info=(directory.parent/'fdinfo'/fd.name).read_text()
                flags=next(int(line.split()[1],8) for line in info.splitlines() if line.startswith('flags:'))&os.O_ACCMODE
                if flags==os.O_WRONLY:modes.add('w')
                elif flags==os.O_RDWR:modes.add('u')
            except (FileNotFoundError,ProcessLookupError):continue
            except PermissionError:
                foreign_uid_fd_modes(proc,descriptor)
                break
            except (StopIteration,ValueError) as error:raise ValueError('FD access mode unavailable') from error
        if modes:
            identity=linux_publisher_identity(proc['pid']);writers.append({k:identity[k] for k in ('pid','start_ticks','executable_sha256')}|{'fd_modes':sorted(modes)})
    same=lambda r:all(r[k]==expected[k] for k in ('pid','start_ticks','executable_sha256'))
    value={'schema_version':1,'captured_at':time.time(),'feed_root_identity':descriptor['feed_root_identity'],'expected_publisher':{k:expected[k] for k in ('pid','start_ticks','executable_sha256')},'writable_fd_processes':writers,'expected_writer_count':sum(same(r) for r in writers),'unknown_writer_count':sum(not same(r) for r in writers)}
    value['inventory_sha256']=operation.digest(value);operation.validate_writer_inventory(descriptor,value)
    return value


def prove_publisher_runtime(descriptor):
    operation.validate_feed_directory(descriptor)
    lock=Path(descriptor['feed_root'])/descriptor['publisher_lock']['name'];info=lock.lstat();expected=descriptor['publisher_lock']
    if stat.S_ISLNK(info.st_mode) or {k:expected[k] for k in ('device','inode','birth_ctime_ns','owner_uid')}!={'device':info.st_dev,'inode':info.st_ino,'birth_ctime_ns':info.st_ctime_ns,'owner_uid':info.st_uid}:raise ValueError('publisher lock drift')
    if linux_publisher_identity(descriptor['publisher_process']['pid'])!=descriptor['publisher_process']:raise ValueError('publisher birth/executable drift')
    # Prove the actual writable lock FD, not only a matching PID or lock path.
    pid=descriptor['publisher_process']['pid'];held=False
    for fd in (Path('/proc')/str(pid)/'fd').iterdir():
        try:
            s=fd.stat()
            if (s.st_dev,s.st_ino)==(info.st_dev,info.st_ino):
                flags=next(int(line.split()[1],8) for line in (fd.parent.parent/'fdinfo'/fd.name).read_text().splitlines() if line.startswith('flags:'))
                if flags&os.O_ACCMODE==os.O_WRONLY:held=True
        except FileNotFoundError:continue
    if not held:raise ValueError('publisher lock FD not held')
    initial=_read_json(Path(descriptor['feed_root'])/'initial-writer-inventory.json',pin=descriptor['initial_no_other_writer_inventory_sha256']);operation.validate_writer_inventory(descriptor,initial)
    linux_writer_inventory(descriptor);return True


def read_head_snapshot(path):
    """Read one bounded mutable HEAD inode; atomic replacement may unlink it."""
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:raise ValueError('absolute HEAD path required')
    if any(stat.S_ISLNK(parent.lstat().st_mode) for parent in (path,*path.parents)):raise ValueError('symlink HEAD path')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size>4096:raise ValueError('bounded regular HEAD required')
        chunks=[];remaining=before.st_size+1
        while remaining:
            block=os.read(fd,remaining)
            if not block:break
            chunks.append(block);remaining-=len(block)
        data=b''.join(chunks);after=os.fstat(fd)
        signature=lambda info:(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
        if len(data)!=before.st_size or signature(before)!=signature(after):raise ValueError('HEAD inplace mutation')
        return json.loads(data)
    finally:os.close(fd)


class SafetyFeedReader:
    def __init__(self,descriptor_path,descriptor_sha256,runs):
        self.path=Path(descriptor_path);self.pin=descriptor_sha256;self.runs=runs;self.last_seq=-1;self.last_sha=None;self.cache={}
        self.descriptor=_read_json(self.path,pin=self.pin);plan.validate_feed_descriptor(self.descriptor)

    def observe(self,*,now):
        import re
        began=now;started=time.monotonic()
        if not _finite(began):raise ValueError('finite observation clock required')
        if _read_json(self.path,pin=self.pin)!=self.descriptor:raise ValueError('descriptor changed')
        prove_publisher_runtime(self.descriptor);root=Path(self.descriptor['feed_root']);head=read_head_snapshot(root/'HEAD.json')
        if set(head)!={'seq','receipt_name','receipt_sha256'} or type(head['seq']) is not int or head['seq']<self.last_seq or head['seq']>11999:raise ValueError('feed rollback/bound')
        names={};entries=list(root.iterdir())
        if len(entries)>24576:raise ValueError('feed namespace bound')
        for path in entries:
            match=re.fullmatch(r'receipt-([0-9]{6})-([0-9a-f]{64})\.json',path.name)
            if not match:continue
            seq=int(match[1]);pin=match[2]
            if seq in names:raise ValueError('feed fork')
            names[seq]=(path,pin)
            s=path.lstat();sig=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
            if not stat.S_ISREG(s.st_mode):raise ValueError('immutable receipt drift')
            # Only HEAD commits a receipt. The single next file may still be
            # completing fsync/chmod/temporary unlink before HEAD is replaced.
            if seq>head['seq']:continue
            if stat.S_IMODE(s.st_mode)!=0o400:raise ValueError('immutable receipt drift')
            if seq in self.cache:
                if self.cache[seq][0]!=sig or self.cache[seq][1]!=pin:raise ValueError('cached immutable receipt drift')
            else:self.cache[seq]=(sig,pin,_read_json(path,cap=16384,pin=pin))
        if any(seq>head['seq']+1 for seq in names):raise ValueError('feed future gap')
        now=began+(time.monotonic()-started)
        if not _finite(now) or now<began:raise ValueError('observation monotonic clock drift')
        previous=None;previous_value=None
        fields={'schema_version','seq','previous_receipt_sha256','g_job_identity','publisher_process','publisher_lock_birth','issued_at','expires_at','next_expected_by','status','payload_sha256'}
        for seq in range(head['seq']+1):
            if seq not in names:raise ValueError('feed missing sequence')
            path,pin=names[seq];value=self.cache[seq][2]
            if set(value)!=fields or value['schema_version']!=1 or value['seq']!=seq or value['previous_receipt_sha256']!=previous or value['g_job_identity']!=self.descriptor['g_job_identity'] or value['publisher_process']!=self.descriptor['publisher_process'] or value['publisher_lock_birth']!={k:self.descriptor['publisher_lock'][k] for k in ('device','inode','birth_ctime_ns','owner_uid')} or value['status'] not in self.descriptor['allowed_statuses'] or not operation._pin(value['payload_sha256']):raise ValueError('feed receipt identity/chain')
            if not all(_finite(value[k]) for k in ('issued_at','expires_at','next_expected_by')) or not value['issued_at']<value['expires_at']<=value['next_expected_by'] or value['issued_at']>now:raise ValueError('feed time schema')
            if previous_value and (previous_value['status']!='OPEN' or not previous_value['issued_at']<=value['issued_at']<=previous_value['next_expected_by']):raise ValueError('terminal append or owner loss gap')
            if seq==0 and pin!=self.descriptor['initial_receipt_sha256']:raise ValueError('initial receipt pin')
            previous=pin;previous_value=value
        if head['seq'] not in names or names[head['seq']][0].name!=head['receipt_name'] or previous!=head['receipt_sha256'] or head['seq']==self.last_seq and previous!=self.last_sha:raise ValueError('HEAD/hash fork')
        if not previous_value or now>=previous_value['expires_at'] or previous_value['status']!='OPEN':raise ValueError('G safety feed stale/closed')
        payload=_read_json(root/('payload-'+previous_value['payload_sha256']+'.json'),cap=262144,pin=previous_value['payload_sha256']);check_payload_binding(previous_value,payload)
        now=began+(time.monotonic()-started)
        if not _finite(now) or now<began or now>=previous_value['expires_at']:raise ValueError('G safety feed expired during observation')
        validate_safety_payload(payload,self.runs,now=now,max_age=30)
        self.last_seq=head['seq'];self.last_sha=previous
        return {'receipt_sha256':previous,'head_seq':head['seq'],'payload':payload,'observed_now':now}


def compile_training_roles(prepared,watch,catalog_payload,idmap_payload):
    """Decode strict role records while preserving original all-past-like masks.

    Real idmap.items records bind item_id/index/train-supported explicitly;
    public role field names remain items/pad/start. The input compiler must
    prove these values equal the frozen original idmap/support artifacts.
    """
    from .retrieval_data import catalog_map
    from .retrieval_watch import history_overlay
    if not isinstance(catalog_payload,dict) or set(catalog_payload)!={'item_ids'}:raise ValueError('catalog role schema')
    catalog=catalog_payload['item_ids'];mapping=catalog_map(catalog)
    if not isinstance(idmap_payload,dict) or set(idmap_payload)!={'items','pad','start'} or idmap_payload['pad']!=0 or idmap_payload['start']!=1 or not isinstance(idmap_payload['items'],list) or len(idmap_payload['items'])!=len(catalog):raise ValueError('real idmap schema')
    supported=[]
    for index,row in enumerate(idmap_payload['items']):
        if not isinstance(row,dict) or set(row)!={'item_id','index','supported'} or row['item_id']!=catalog[index] or type(row['index']) is not int or row['index']!=index+1 or type(row['supported']) is not bool:raise ValueError('idmap/support drift')
        if row['supported']:supported.append(index+1)
    if not isinstance(prepared,list) or not prepared or not isinstance(watch,list) or len(prepared)!=len(watch):raise ValueError('training identity bijection')
    histories={};seen={};events={};rows=[];previous=-1
    for index,row in enumerate(prepared):
        if not isinstance(row,dict) or set(row)!={'user_id','history','history_positions','target','target_position'} or type(row['user_id']) is not int or row['user_id']<0 or row['target'] not in mapping or mapping[row['target']] not in supported or type(row['target_position']) is not int or row['target_position']<=previous:raise ValueError('canonical train target ordering')
        previous=row['target_position'];h=row['history'];pos=row['history_positions'];user=row['user_id']
        if not isinstance(h,list) or not 1<=len(h)<=50 or not isinstance(pos,list) or len(pos)!=len(h) or any(type(p) is not int or p<0 or p>=row['target_position'] for p in pos) or any(a>=b for a,b in zip(pos,pos[1:])) or any(item not in mapping or mapping[item] not in supported for item in h):raise ValueError('like strict past identity')
        if user not in histories:
            if len(h)!=1:raise ValueError('first prefix missing earlier all-past provenance')
            histories[user]=list(zip(h,pos));seen[user]=set(h)
        if list(zip(h,pos))!=histories[user][-50:]:raise ValueError('canonical like prefix mismatch')
        rows.append({**row,'example_id':index,'past_likes':sorted(seen[user])})
        for item,position in [*zip(h,pos),(row['target'],row['target_position'])]:
            key=(user,position)
            if key in events and events[key]!=item:raise ValueError('train event identity conflict')
            events[key]=item
        histories[user].append((row['target'],row['target_position']));seen[user].add(row['target'])
    observed=history_overlay(rows,watch)
    for row in observed:
        if any(item not in mapping for item in row['history']):raise ValueError('watch catalog drift')
        for item,position in zip(row['history'],row['history_positions']):
            if type(position) is not int or position<0:raise ValueError('watch train ordering')
            key=(row['user_id'],position)
            if key in events and events[key]!=item:raise ValueError('watch/like event conflict')
            events[key]=item
    return {'rows_by_arm':{'like':rows,'observed':observed},'catalog':catalog,'idmap':mapping,'supported_indices':supported,'train_events':[{'user_id':u,'item_id':item,'position':position,'partition':'train'+str(position>>32)} for (u,position),item in sorted(events.items())]}


def finite_batches(rows,*,batch_size,seed):
    from .retrieval_training import EpochCursor
    if not isinstance(rows,list) or not rows or type(batch_size) is not int or batch_size<=0 or seed not in (42,2024,2025) or type(seed) is not int:raise ValueError('finite draw configuration')
    per_epoch=(len(rows)+batch_size-1)//batch_size
    if per_epoch*30>5100:raise ValueError('5100 registered optimizer ceiling')
    cursor=EpochCursor(len(rows),seed);batches=[]
    for _ in range(30*per_epoch):
        indices=cursor.next(batch_size)
        batches.append({'epoch':cursor.epoch+1,'epoch_end':cursor.offset==cursor.size,'rows':[rows[i] for i in indices]})
    return batches


def read_roles(value):
    validate_remote_manifest(value);contents={};opened=[];total=0
    record_roles={'prepared-train-rows','watch-train','train-prefix-target-free-sidecar','formal-history-view','validation-query-target-free','validation-target'}
    for role in value['roles']:
        size=role['size']
        if type(size) is not int or not 0<=size<=256*1024**2:raise ValueError('role byte ceiling')
        total+=size
        if total>512*1024**2:raise ValueError('operation input byte ceiling')
        data=read_bound(role['path'],role['sha256'],max(1,size))
        if len(data)!=size:raise ValueError('role size mismatch')
        payload=json.loads(data)
        if role['role'] in record_roles:
            if not isinstance(payload,list) or not payload or any(not isinstance(r,dict) or set(r)!=set(role['schema_fields']) for r in payload):raise ValueError('exact record role payload')
            if role['role']=='train-prefix-target-free-sidecar' and plan._contains_forbidden(payload):raise ValueError('target-free forbidden supervision')
        elif not isinstance(payload,dict) or set(payload)!=set(role['schema_fields']):raise ValueError('exact role payload fields')
        contents[role['role']]=payload;opened.append({'role':role['role'],'path':role['path'],'sha256':sha(data),'size':len(data),'schema_id':role['schema_id']})
    operation._validate_opened_role_bijection(value['roles'],opened)
    return contents,opened


def start_latched_child(body,stdout_path,stderr_path,*,identity_probe,python=None):
    """Create only a stdlib leaf until its birth has been independently probed.

    Probe failure closes stdin before scientific code can execute. Linux callers
    require pidfd support; a pidfd permits safe cleanup of that unreleased leaf.
    """
    import signal
    import subprocess
    import sys
    if not isinstance(body,str) or not body or len(body.encode())>131072 or not callable(identity_probe):raise ValueError('bounded verified worker body')
    stdout=Path(stdout_path).open('xb')
    try:stderr=Path(stderr_path).open('xb')
    except BaseException:stdout.close();raise
    prefix="import os,sys\nif os.read(0,1)!=b'1':sys.exit(125)\n"
    process=None;pidfd=None
    try:
        process=subprocess.Popen([python or sys.executable,'-I','-S','-B','-c',prefix+body],stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,start_new_session=True)
        if hasattr(os,'pidfd_open'):pidfd=os.pidfd_open(process.pid,0)
        identity=identity_probe(process.pid)
        if identity.get('pid')!=process.pid or identity.get('pgid')!=process.pid or identity.get('sid')!=process.pid or process.poll() is not None:raise ValueError('new leaf birth/group drift')
        return {'process':process,'identity':identity,'pidfd':pidfd,'stdout':stdout,'stderr':stderr,'released':False}
    except BaseException:
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:process.stdin.close()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if pidfd is not None and hasattr(signal,'pidfd_send_signal'):
                    signal.pidfd_send_signal(pidfd,signal.SIGKILL);process.wait(timeout=5)
                else:raise RuntimeError('FAILED_UNSAFE: unreleased child closure unavailable')
        if pidfd is not None:os.close(pidfd)
        stdout.close();stderr.close();raise


class MonotonicBudget(Budget):
    def __init__(self,queue,*,monotonic=time.monotonic,boot_id=None):
        super().__init__(queue['kind'],started_at=queue['started_at'],deadline=queue['deadline'],disk_cap=queue['disk_cap'],attempts=queue['attempts'],extension=queue.get('extension'),clock_extension=queue.get('clock_extension'))
        for key in ('started_monotonic','deadline_monotonic'):
            if not _finite(queue.get(key)):raise ValueError('original monotonic clock required')
        if queue['deadline_monotonic']-queue['started_monotonic']!=queue['deadline']-queue['started_at']:raise ValueError('monotonic budget extension')
        if not isinstance(queue.get('host_boot_id'),str) or not queue['host_boot_id']:raise ValueError('host boot identity required')
        if boot_id is None:boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if queue['host_boot_id']!=boot_id:raise ValueError('host reboot: no automatic budget reset')
        self.mono=monotonic;self.started_mono=queue['started_monotonic'];self.deadline_mono=queue['deadline_monotonic']+clock_extension_seconds(queue.get('clock_extension'))
        self.clock_authority=verify_clock_extension(queue)

    def remaining_seconds(self,now):
        mono=self.mono()
        if not _finite(mono) or mono<self.started_mono or abs((now-self.started_at)-(mono-self.started_mono))>5:raise ValueError('wall/monotonic clock drift')
        return min(super().remaining_seconds(now),max(0,self.deadline_mono-mono))


class OwnedTree:
    """Dedicated subreaper ancestry, retained births and protected neighbors."""
    def __init__(self,owner_pid,controller_pgid,*,baseline_children,protected_pgids,table=linux_processes):
        self.owner_pid=owner_pid;self.controller_pgid=controller_pgid;self.baseline_children=set(baseline_children);self.protected_pgids=set(protected_pgids);self.table=table;self.known={};self.sessions=set();self.groups=set()

    def remember(self,leader):
        if leader['ppid']!=self.owner_pid or leader['pgid']!=leader['pid'] or leader['sid']!=leader['pid'] or leader['pgid']==self.controller_pgid or leader['pgid'] in self.protected_pgids:raise ValueError('dedicated child group required')
        self.known[leader['pid']]=dict(leader);self.sessions.add(leader['sid']);self.groups.add(leader['pgid'])

    def _collect(self,rows):
        current={r['pid']:r for r in rows};selected={}
        for pid,old in self.known.items():
            row=current.get(pid)
            if row is not None:
                if _identity(row)!=_identity(old):raise ValueError('owned birth/group drift; zero signal')
                selected[pid]=row
        for row in rows:
            if row['ppid']==self.owner_pid and row['pid'] not in self.baseline_children:selected[row['pid']]=row
        for _ in range(512):
            new=[r for r in rows if r['ppid'] in selected and r['pid'] not in selected]
            if not new:break
            selected.update({r['pid']:r for r in new})
            if len(selected)>512:raise ValueError('owned descendant bound')
        if len(selected)>512:raise ValueError('owned descendant bound')
        for row in selected.values():
            self.known.setdefault(row['pid'],dict(row));self.sessions.add(row['sid']);self.groups.add(row['pgid'])
        return sorted(selected.values(),key=lambda r:r['pid'])

    def collect(self):return self._collect(self.table())

    def signal_plan(self):
        rows=self.table();owned=self._collect(rows);pids={r['pid'] for r in owned}
        if any(r['pgid']==self.controller_pgid or r['pgid'] in self.protected_pgids for r in owned):raise ValueError('protected process scope; zero signal')
        if any(r['pid'] not in pids and (r['pgid'] in self.groups or r['sid'] in self.sessions) for r in rows):raise ValueError('unknown group/session member; zero signal')
        return [r for r in owned if r['state']!='Z']

    def closed(self):
        rows=self.table();owned=self._collect(rows)
        return not owned and not any(r['sid'] in self.sessions or r['pgid'] in self.groups for r in rows)


class LinuxKernel:
    def __init__(self):
        import ctypes
        import signal
        import sys
        if sys.platform!='linux' or not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):raise ValueError('Linux pidfd runtime required before spawn')
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(36,1,0,0,0)!=0:raise ValueError('dedicated subreaper unavailable')

    def processes(self):return linux_processes()
    def identity(self,pid):
        try:return linux_identity(pid)
        except (FileNotFoundError,ProcessLookupError):return None
    def open_pidfd(self,row):
        try:fd=os.pidfd_open(row['pid'],0)
        except ProcessLookupError:
            if self.identity(row['pid']) is None:return None
            raise ValueError('pidfd birth drift')
        current=self.identity(row['pid'])
        if current is None and foreign_observer_exited(fd):os.close(fd);return None
        if current is None or _identity(current)!=_identity(row):os.close(fd);raise ValueError('pidfd birth drift')
        return fd
    def signal(self,fd,sig):
        import signal
        try:signal.pidfd_send_signal(fd,sig)
        except ProcessLookupError:pass
    def reap(self,row):
        current=self.identity(row['pid'])
        if current is None or _identity(current)!=_identity(row) or current['state']!='Z':return None
        try:pid,status=os.waitpid(row['pid'],os.WNOHANG)
        except ChildProcessError:return None
        return {'pid':pid,'birth':row['birth'],'wait_status':status} if pid else None
    def close_fd(self,fd):os.close(fd)


class KernelLease:
    """Exclusive P worker handle, pidfds, exact waits and descendant closure."""
    def __init__(self,handle,*,kernel,owner_pid,controller_pgid,baseline_children,protected_pgids):
        if handle.get('pidfd') is None:
            close_unreleased(handle)
            raise ValueError('worker pidfd required')
        self.handle=handle;self.kernel=kernel;self.tree=OwnedTree(owner_pid,controller_pgid,baseline_children=baseline_children,protected_pgids=protected_pgids,table=kernel.processes);self.tree.remember(handle['identity']);self.fds={handle['identity']['pid']:handle['pidfd']};self.adopted_waits=[]
        try:self.refresh()
        except BaseException:
            close_unreleased(handle)
            for pid,fd in self.fds.items():
                if pid!=handle['process'].pid:self.kernel.close_fd(fd)
            self.fds.clear()
            raise

    def refresh(self,reap=False):
        rows=self.tree.collect()
        for row in rows:
            if row['pid'] not in self.fds:
                fd=self.kernel.open_pidfd(row)
                if fd is not None:self.fds[row['pid']]=fd
            if reap and row['pid']!=self.handle['process'].pid:
                waited=self.kernel.reap(row)
                if waited:self.adopted_waits.append(waited)
        return rows

    def _signal(self,sig):
        rows=self.tree.signal_plan()
        for row in rows:
            if row['pid'] not in self.fds:
                fd=self.kernel.open_pidfd(row)
                if fd is not None:self.fds[row['pid']]=fd
        # Refresh every birth after the complete plan, before sending any signal.
        for row in rows:
            current=self.kernel.identity(row['pid'])
            if current is not None and _identity(current)!=_identity(row):raise ValueError('birth drift; zero signal')
        live=[row for row in rows if row['pid'] in self.fds]
        for row in live:self.kernel.signal(self.fds[row['pid']],sig)
        return len(live)

    def close(self,stop_path,*,manifest_sha256,seal_seconds,signal_grace_seconds,deadline_monotonic=None):
        import signal
        import subprocess
        if not operation._pin(manifest_sha256) or not _finite(seal_seconds) or not 0<seal_seconds<=120 or not _finite(signal_grace_seconds) or not 0<signal_grace_seconds<=30:raise ValueError('finite owned stop contract')
        started=time.monotonic();process=self.handle['process'];signals=0;actual=None;failure=None
        deadline_monotonic=started+seal_seconds+3*signal_grace_seconds if deadline_monotonic is None else deadline_monotonic
        if not _finite(deadline_monotonic):raise ValueError('original cleanup deadline required')
        try:
            self.refresh()
            if not Path(stop_path).exists():write_once(stop_path,{'schema_id':'p4h-real-stop-request-v1','operation_manifest_sha256':manifest_sha256,'worker':self.handle['identity'],'issued_at':time.time()})
            end=min(deadline_monotonic,time.monotonic()+seal_seconds)
            while time.monotonic()<end:
                self.refresh(reap=True);actual=process.poll()
                if actual is not None and self.tree.closed():break
                time.sleep(min(.02,max(0,end-time.monotonic())))
            if actual is None or not self.tree.closed():
                signals+=self._signal(signal.SIGTERM);end=min(deadline_monotonic,time.monotonic()+signal_grace_seconds)
                while time.monotonic()<end:
                    self.refresh(reap=True);actual=process.poll()
                    if actual is not None and self.tree.closed():break
                    time.sleep(min(.02,max(0,end-time.monotonic())))
            if actual is None or not self.tree.closed():signals+=self._signal(signal.SIGKILL)
            try:actual=process.wait(timeout=max(0,min(signal_grace_seconds,deadline_monotonic-time.monotonic())))
            except subprocess.TimeoutExpired:actual=None
            end=min(deadline_monotonic,time.monotonic()+signal_grace_seconds)
            while time.monotonic()<end:
                self.refresh(reap=True)
                if self.tree.closed():break
                time.sleep(.02)
            closed=actual is not None and self.tree.closed()
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
            closed=False;failure=type(error).__name__+': '+str(error)
        finally:
            for fd in self.fds.values():self.kernel.close_fd(fd)
            self.fds.clear();self.handle['stdout'].close();self.handle['stderr'].close()
        return {'state':'CLOSED' if closed else 'FAILED_UNSAFE','actual_worker_exit':actual,'elapsed_seconds':time.monotonic()-started,'signals_sent':signals,'adopted_waits':self.adopted_waits,'owned_identities':list(self.tree.known.values()),'owned_sessions':sorted(self.tree.sessions),'failure':failure}


def queue_tree_usage(root,*,max_nodes=32768):
    root=Path(root);original=root.lstat()
    if not stat.S_ISDIR(original.st_mode) or stat.S_ISLNK(original.st_mode):raise ValueError('owned queue directory required')
    pending=[root];nodes=0;total=0
    while pending:
        directory=pending.pop()
        with os.scandir(directory) as scan:
            for entry in scan:
                nodes+=1
                if nodes>max_nodes:raise ValueError('queue disk scan bound')
                s=entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(s.st_mode):raise ValueError('queue symlink or unknown writer')
                if stat.S_ISDIR(s.st_mode):pending.append(Path(entry.path))
                elif stat.S_ISREG(s.st_mode):total+=s.st_size
                else:raise ValueError('queue special file')
    after=root.lstat()
    if (original.st_dev,original.st_ino)!=(after.st_dev,after.st_ino):raise ValueError('queue root identity drift')
    return {'bytes':total,'nodes':nodes,'root_identity':{'device':original.st_dev,'inode':original.st_ino}}


class RuntimeGate:
    def __init__(self,queue_root,stop_path,budget,reader,limits,*,clock=time.time):
        if not isinstance(budget,MonotonicBudget) or not isinstance(reader,SafetyFeedReader):raise ValueError('real clock and safety feed required')
        self.root=Path(queue_root);self.stop=Path(stop_path);self.budget=budget;self.reader=reader;self.limits=dict(limits);self.clock=clock;self.samples=[];self.last_feed=None
        s=self.root.lstat();self.root_identity=(s.st_dev,s.st_ino)

    def check(self):
        if self.stop.exists() or self.stop.is_symlink():raise ValueError('cooperative stop requested')
        now=self.clock()
        if self.last_feed is not None and 0<=now-self.samples[-1]['observed_at']<self.limits.get('sample_seconds',0):
            sample=self.samples[-1]
            self.budget.require_stage(now=now,stage_seconds=self.limits['stage_seconds'],finish_seconds=self.limits['finish_seconds'],used_bytes=sample['queue_bytes'],next_atomic_bytes=self.limits['next_atomic_bytes'],closure_bytes=self.limits['stop_reserve_bytes'],free_fractions=sample['free_fractions'])
            return self.last_feed
        feed=self.reader.observe(now=now);now=feed['observed_now'];usage=queue_tree_usage(self.root)
        if self.root_identity!=(usage['root_identity']['device'],usage['root_identity']['inode']):raise ValueError('queue root drift')
        fractions=dict(feed['payload']['free_fractions']);fs=os.statvfs(self.root)
        fractions['disk']-=(self.limits['next_atomic_bytes']+self.limits['stop_reserve_bytes'])/(fs.f_blocks*fs.f_frsize)
        self.budget.require_stage(now=now,stage_seconds=self.limits['stage_seconds'],finish_seconds=self.limits['finish_seconds'],used_bytes=usage['bytes'],next_atomic_bytes=self.limits['next_atomic_bytes'],closure_bytes=self.limits['stop_reserve_bytes'],free_fractions=fractions)
        sample={'observed_at':now,'queue_bytes':usage['bytes'],'remaining_seconds':self.budget.remaining_seconds(now),'free_fractions':fractions,'g_receipt_sha256':feed['receipt_sha256']}
        self.samples.append(sample);self.last_feed=feed
        if len(self.samples)>20000:raise ValueError('finite resource sample bound')
        return feed


def encode_position(position):
    if not isinstance(position,list) or len(position)!=2 or any(type(v) is not int for v in position) or not 0<=position[0]<=24 or not 0<=position[1]<2**32:raise ValueError('proven train physical position required')
    return (position[0]<<32)|position[1]


def compile_train_prefixes(partitions,examples,catalog):
    """Replay frozen train only, never derive capacity queries from holdout."""
    from collections import deque
    from .retrieval_watch import build_watch
    likes={};observed={};support=set()
    def wrapped():
        previous=-1
        for week,events in partitions:
            if type(week) is not int or not previous<week<=24:raise ValueError('train partitions only')
            previous=week
            def rows(events=events):
                for row in events:
                    user=row['user_id'];item=row['item_id']
                    if type(row['like']) is not bool:raise ValueError('frozen like label required')
                    observed.setdefault(user,deque(maxlen=50)).append(item)
                    if row['like']:likes.setdefault(user,deque(maxlen=50)).append(item);support.add(item)
                    yield row
            yield week,rows()
    watch,validation,_=build_watch(wrapped(),examples,[],catalog)
    if validation:raise ValueError('pilot never reads validation')
    sidecar=[]
    for user in sorted(observed):
        if user in likes:sidecar.append({'query_id':user*2,'history':list(likes[user])})
        sidecar.append({'query_id':user*2+1,'history':list(observed[user])})
    return {'watch_train':watch,'sidecar':sidecar,'support':sorted(support)}


RUNTIME_SOURCE_MEMBERS = (
    'src/vkgr/__init__.py', 'src/vkgr/retrieval_sasrec_remote.py',
    'src/vkgr/retrieval_sasrec.py', 'src/vkgr/retrieval_sasrec_plan.py',
    'src/vkgr/retrieval_sasrec_operation.py', 'src/vkgr/retrieval_sasrec_experiment.py',
    'src/vkgr/retrieval_training.py', 'src/vkgr/retrieval_data.py',
    'src/vkgr/retrieval_watch.py', 'src/vkgr/retrieval_experiment.py',
    'src/vkgr/retrieval_models.py', 'src/vkgr/baselines.py',
    'src/vkgr/data_audit.py', 'src/vkgr/manifest.py', 'src/vkgr/training.py',
    'src/vkgr/retrieval_ablation_run.py', 'src/vkgr/config.py',
)


def verify_source_tree(value):
    pins=value['source_pins'];root=Path(value['code_root'])
    if not set(RUNTIME_SOURCE_MEMBERS)<=set(pins):raise ValueError('complete runtime source binding required')
    for name,pin in pins.items():
        if not isinstance(name,str) or name.startswith('/') or '..' in name.split('/') or not operation._pin(pin):raise ValueError('source member metadata')
        read_bound(root/name,pin,8*1024**2)
    return {'members':dict(pins),'source_sha256':sha(canonical(pins))}


def visible_ram_fraction(current,root,host_fraction):
    from .retrieval_ablation_run import visible_memory_fraction
    return visible_memory_fraction(current,root,host_fraction)


def current_ram_fraction():
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    membership=[line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    mounts=[line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup2 ' in line]
    if len(membership)!=1 or len(mounts)!=1:raise ValueError('ambiguous cgroup2 mapping')
    mount=mounts[0];root=Path(mount[4]);member=Path(membership[0]);mountroot=Path(mount[3])
    relative=Path('.') if member==mountroot else member.relative_to(mountroot) if mountroot in member.parents else Path(str(member).lstrip('/'))
    return visible_ram_fraction(root/relative,root,memory['MemAvailable']/memory['MemTotal'])


def materialize_training_inputs(examples,catalog,original_idmap,support,partitions,output,*,capacity_per_arm=None):
    from .retrieval_data import catalog_map
    mapping=catalog_map(catalog)
    if support!=sorted(set(support)) or any(item not in mapping for item in support):raise ValueError('original support drift')
    expected={'items':catalog,'pad':0,'start':1,'supported_indices':sorted(mapping[i] for i in support)}
    if original_idmap!=expected:raise ValueError('original mapping/support drift')
    prefix=compile_train_prefixes(partitions,examples,catalog)
    if prefix['support']!=support:raise ValueError('raw train support drift')
    if capacity_per_arm is not None:
        cohorts=select_capacity_cohort(prefix['sidecar'],per_arm=capacity_per_arm)
        prefix['sidecar']=cohorts['like']+cohorts['observed']
    prepared=[{k:row[k] for k in ('user_id','history','history_positions','target','target_position')} for row in examples]
    for row in prepared:
        row['history_positions']=[encode_position(p) for p in row['history_positions']];row['target_position']=encode_position(row['target_position'])
    watch=[{**row,'history_positions':[encode_position(p) for p in row['history_positions']]} for row in prefix['watch_train']]
    support_set=set(support)
    idmap={'items':[{'item_id':i,'index':mapping[i],'supported':i in support_set} for i in catalog],'pad':0,'start':1}
    result=compile_training_roles(prepared,watch,{'item_ids':catalog},idmap)
    if result['rows_by_arm']['like']!=[{**e,'history_positions':[encode_position(p) for p in e['history_positions']],'target_position':encode_position(e['target_position'])} for e in examples]:raise ValueError('all past prepared replay mismatch')
    output=Path(output);output.mkdir(mode=0o700)
    payloads={'prepared-train-rows':prepared,'watch-train':watch,'catalog':{'item_ids':catalog},'idmap':idmap,'train-prefix-target-free-sidecar':prefix['sidecar']}
    payloads['like-history-view']=[{'example_id':r['example_id'],'history':r['history'],'history_positions':r['history_positions']} for r in result['rows_by_arm']['like']]
    artifacts={}
    for role,payload in payloads.items():
        path=output/(role+'.json');pin=write_once(path,payload);artifacts[role]={'path':str(path),'sha256':pin,'size':path.stat().st_size}
    return {**result,'artifacts':artifacts,'sidecar':prefix['sidecar']}


def serve_publisher(root,runs,*,deadline,interval,allowed_p):
    if not _finite(deadline) or not 0<deadline-time.time()<=64800 or not _finite(interval) or not 0<interval<=5:raise ValueError('bounded publisher lifetime required')
    journal=FeedJournal(root,linux_publisher_identity(os.getpid()),{'job_id_hash':sha(canonical([r['run_id'] for r in runs])),'start_identity_hash':sha(canonical(runs))})
    status='OWNER_LOST';failure=None;last=None;closed_reason='owner_lost'
    try:
        write_once(journal.root/'runs.json',runs)
        inventory=linux_writer_inventory(journal.descriptor)
        pin=write_once(journal.root/'initial-writer-inventory.json',inventory);os.chmod(journal.root/'initial-writer-inventory.json',0o400)
        journal.descriptor['initial_no_other_writer_inventory_sha256']=pin
        last=observe_g(runs,allowed_p=allowed_p);journal.publish(last,now=time.time(),status='OPEN')
        write_once(journal.root/'descriptor.json',journal.descriptor);os.chmod(journal.root/'descriptor.json',0o400)
        while time.time()<deadline:
            if publisher_stop_requested(journal):closed_reason='cooperative_stop';break
            time.sleep(min(interval,max(0,deadline-time.time())))
            last=observe_g(runs,allowed_p=allowed_p)
            journal.publish(last,now=time.time(),status='OPEN')
            if journal.seq>=11998:raise ValueError('publisher finite sequence exhausted')
        status='CLOSED'
        if closed_reason!='cooperative_stop':closed_reason='deadline'
    except (OSError,ValueError,RuntimeError) as error:failure=type(error).__name__+': '+str(error)
    finally:
        if not journal.terminal:journal.publish(last or {'observed_at':time.time(),'failure':failure},now=time.time(),status=status)
        journal.close()
        if (journal.root/'descriptor.json').exists():
            write_once(journal.root/'publisher-closure.json',{'schema_id':'p4h-real-publisher-closure-v1','status':status,'reason':closed_reason,'descriptor_sha256':operation._file_sha256(journal.root/'descriptor.json'),'publisher_process':journal.descriptor['publisher_process'],'lock_fd_closed':journal.lock_fd is None,'last_receipt_sha256':journal.previous,'issued_at':time.time()})
    return {'status':status,'failure':failure,'receipt_count':journal.seq}



def evaluate_formal(model,epoch,catalog,queries,idmap,supported,output,*,gate):
    from .retrieval_experiment import export_vectors,evaluate_vectors
    users,items=export_vectors(model,catalog,queries,idmap,512,check=gate)
    result=evaluate_vectors(users,items,catalog,queries,supported,check=gate)
    output=Path(output);gate()
    prediction_pin=write_once(output/f'epoch-{epoch}-predictions.json',result['predictions'])
    write_once(output/f'epoch-{epoch}-metrics.json',{k:v for k,v in result.items() if k!='predictions'})
    return {'epoch':epoch,'metrics':{'all':{k:result['metrics']['all'][k] for k in ('recall@50','ndcg@50')}},'prediction_sha256':prediction_pin}


def select_capacity_cohort(sidecar,*,per_arm=2520,seed=42,lineage='train'):
    """Matched train users, biased toward longest real histories for capacity."""
    if lineage!='train' or type(seed) is not int or seed!=42 or type(per_arm) is not int or not 1<=per_arm<=2520 or not isinstance(sidecar,list):
        raise ValueError('registered train-only capacity policy required')
    by_id={}
    for row in sidecar:
        if not isinstance(row,dict) or set(row)!={'query_id','history'} or type(row['query_id']) is not int or row['query_id']<0 or not isinstance(row['history'],list) or not 1<=len(row['history'])<=50 or any(type(i)is not int or not 0<=i<2**32 for i in row['history']) or row['query_id'] in by_id:
            raise ValueError('unique target-free nonempty histories required')
        by_id[row['query_id']]=row
    users=[i//2 for i in by_id if i%2==0]
    if any(2*u+1 not in by_id for u in users) or len(users)<per_arm:
        raise ValueError('complete matched capacity cohort required')
    users.sort(key=lambda u:(-len(by_id[2*u]['history']),-len(by_id[2*u+1]['history']),sha(f'42:{u}'.encode())))
    return {arm:[{'query_id':2*u+parity,'history':list(by_id[2*u+parity]['history'])} for u in users[:per_arm]] for arm,parity in [('like',0),('observed',1)]}


def run_pilot_worker(data,output,*,binding,gate,device,steps=30,capacity_per_arm=None):
    import io
    import numpy as np
    import torch
    from .retrieval_sasrec import SASRecIDTwoTower
    from .retrieval_training import train_step,SamplerStats
    from .retrieval_experiment import export_vectors
    from .baselines import blockwise_topk
    from .training import seed_all
    from .retrieval_sasrec_experiment import _draw_digest
    if type(steps) is not int or not 1<=steps<=30 or len(data['catalog'])<80:raise ValueError('finite actual pilot configuration')
    execution=configure_real_torch()
    output=Path(output)
    if not output.is_dir() or list(output.iterdir()):raise ValueError('exclusive empty pilot output')
    queries=data['sidecar']
    if not isinstance(queries,list) or not queries or any(set(q)!={'query_id','history'} for q in queries):raise ValueError('train-only capacity queries required')
    cohorts=select_capacity_cohort(queries,per_arm=capacity_per_arm) if capacity_per_arm is not None else {'like':queries,'observed':queries}
    arms={};events=[]
    def measured(name,body,artifacts=[]):
        gate()
        def actual():
            result=body()
            if device=='cuda':torch.cuda.synchronize()
            gate();return result
        return measure_action(name,actual,artifacts=artifacts,sink=events.append)
    for arm in ('like','observed'):
        queries=cohorts[arm]
        seed_all(42);gate();model=SASRecIDTwoTower(torch.zeros((len(data['catalog'])+1,64)),data['supported_indices']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        batches=finite_batches(data['rows_by_arm'][arm],batch_size=512,seed=42)[:steps];stats=SamplerStats();measurements={}
        def train():
            for batch in batches:
                gate();train_step(model,optimizer,batch['rows'],data['idmap'],stats,batch['epoch']-1,.15)
        _,rec=measured('train_step',train);measurements['train_step']={k:rec[k] for k in ('seconds','bytes')}
        draws,rec=measured('draw_replay',lambda:[_draw_digest(b,data['idmap']) for b in batches]);measurements['draw_replay']={k:rec[k] for k in ('seconds','bytes')}
        state=output/(arm+'-state.pt')
        _,rec=measured('checkpoint_write',lambda:_save_tensors_once(state,{'binding':binding,'model':model.state_dict(),'best_model':{k:v.clone() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'draws':draws}),[state]);measurements['checkpoint_write']={k:rec[k] for k in ('seconds','bytes')}
        def export_exact():
            vectors=export_vectors(model,data['catalog'],queries,data['idmap'],512,check=gate)
            predictions=[]
            for offset in range(0,len(queries),32):
                gate();predictions.extend(blockwise_topk(vectors[0][offset:offset+32],vectors[1],data['catalog'],50).tolist())
            write_once(output/(arm+'-capacity-predictions.json'),predictions)
            write_once(output/(arm+'-capacity-format.json'),{'capacity_only':True,'query_count':len(queries),'candidate_counts':[len(p) for p in predictions],'quality_metrics_computed':False})
            return vectors,predictions
        (vectors,predictions),rec=measured('epoch_export_exact',export_exact,[output/(arm+'-capacity-predictions.json'),output/(arm+'-capacity-format.json')]);measurements['epoch_export_exact']={k:rec[k] for k in ('seconds','bytes')}
        top80,rec=measured('target_free_top80',lambda:score_capacity(vectors,data['catalog'],80,gate));measurements['target_free_top80']={k:rec[k] for k in ('seconds','bytes')}
        fresh=SASRecIDTwoTower(torch.zeros((len(data['catalog'])+1,64)),data['supported_indices']).to(device)
        def fresh_export():
            blob=read_bound(state,operation._file_sha256(state),state.stat().st_size);loaded=torch.load(io.BytesIO(blob),weights_only=True,map_location=device)
            if loaded['binding']!=binding or loaded['draws']!=draws:raise ValueError('pilot tensor binding drift')
            fresh.load_state_dict(loaded['model']);return export_vectors(fresh,data['catalog'],queries,data['idmap'],512,check=gate)
        fresh_vectors,rec=measured('final_verify_export',fresh_export);measurements['final_verify_export']={k:rec[k] for k in ('seconds','bytes')}
        def fresh_exact():
            predictions=score_capacity(fresh_vectors,data['catalog'],50,gate)
            write_once(output/(arm+'-capacity-fresh-predictions.json'),predictions)
            return predictions
        fresh_predictions,rec=measured('final_verify_exact',fresh_exact,[output/(arm+'-capacity-fresh-predictions.json')]);measurements['final_verify_exact']={k:rec[k] for k in ('seconds','bytes')}
        if fresh_predictions!=predictions or not all(np.array_equal(a,b) for a,b in zip(vectors,fresh_vectors)):raise ValueError('pilot fresh replay mismatch')
        arms[arm]={'history_source':arm,'optimizer_steps':len(batches),'measurements':measurements,'draw_sha256':sha(canonical(draws)),'top80_count':[len(r) for r in top80],'fresh_export_matches':True,'prediction_sha256':sha(canonical(predictions)),'top80_sha256':sha(canonical(top80))}
        del model,optimizer,fresh,vectors,fresh_vectors
        if device=='cuda':torch.cuda.empty_cache()
    if capacity_per_arm is not None:
        for arm,row in arms.items():
            row['capacity_query_count']=len(cohorts[arm]);row['query_identity_sha256']=sha(canonical([q['query_id'] for q in cohorts[arm]]))
    queries=data['sidecar']
    value={'status':'pilot_worker_completed+unverified','execution':execution,'binding':binding,'arms':arms,'query_count':len(queries),'query_identity_sha256':sha(canonical([q['query_id'] for q in queries])),'measurements':events}
    if capacity_per_arm is not None:value['capacity_per_arm']=capacity_per_arm
    gate();write_once(output/'pilot-worker.json',value);return value


def score_capacity(vectors,catalog,k,gate):
    from .baselines import blockwise_topk
    rows=[]
    for offset in range(0,len(vectors[0]),32):
        gate();rows.extend(blockwise_topk(vectors[0][offset:offset+32],vectors[1],catalog,k).tolist())
    return rows



def close_unreleased(handle):
    import signal
    import subprocess
    if handle.get('released'):raise ValueError('unreleased cleanup cannot own a running science tree')
    process=handle['process'];fd=handle.get('pidfd')
    try:
        if process.stdin and not process.stdin.closed:process.stdin.close()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if fd is None or not hasattr(signal,'pidfd_send_signal'):raise RuntimeError('FAILED_UNSAFE unreleased leaf')
            signal.pidfd_send_signal(fd,signal.SIGKILL);process.wait(timeout=5)
    finally:
        if fd is not None:os.close(fd);handle['pidfd']=None
        handle['stdout'].close();handle['stderr'].close()


def verify_common_bindings(value,contents):
    import importlib.metadata
    import platform
    import sys
    source=verify_source_tree(value);pins=source['members']
    for role in ('code-archive','source-fingerprint'):
        if contents[role]!=source:raise ValueError('common source/actual package binding drift')
    executable=sha(Path(sys.executable).resolve().read_bytes())
    if executable!=value['python_sha256'] or contents['python-executable-receipt']!={'python_version':platform.python_version(),'executable_sha256':executable}:raise ValueError('actual Python/runtime drift')
    controller=contents['controller-executable']
    if controller!={'path_sha256':sha(str(Path(__file__).resolve()).encode()),'source_sha256':pins['src/vkgr/retrieval_sasrec_remote.py']} or Path(__file__).resolve()!=Path(value['code_root'])/'src/vkgr/retrieval_sasrec_remote.py':raise ValueError('actual controller origin drift')
    dependencies=contents['dependency-lock']
    if 'uv.lock' not in pins or dependencies['lock_sha256']!=pins['uv.lock'] or not dependencies['dependencies']:raise ValueError('actual dependency lock binding')
    for name,version in dependencies['dependencies'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('actual dependency version drift')
    expected_files={'p4h-config':('configs/p4h-sasrec-v2.json','config_sha256'),'p4h-protocol':('docs/p4h_sasrec_protocol_v2.md','protocol_sha256')}
    for role,(member,key) in expected_files.items():
        if member not in pins or contents[role][key]!=pins[member]:raise ValueError('science configuration binding drift')
    spec=contents['p4h-plan']
    if spec['formal_slots']!=plan.FORMAL_SLOTS or spec['task_ids']!=plan.TASK_IDS or spec['plan_sha256']!=plan.EXPECTED_CONFIG['source_plan_sha256']:raise ValueError('frozen slot/plan binding drift')
    if contents['g-feed-descriptor']['descriptor_sha256']!=value['g_feed']['sha256']:raise ValueError('common Gfeed binding drift')
    if contents['cooperative-stop-control-schema']!={'request_schema':'p4h-real-stop-request-v1','receipt_schema':'p4h-real-closure-v1'}:raise ValueError('stop schema binding')
    return source


def prepare_controller(value,manifest_sha256,runs):
    validate_remote_manifest(value)
    verify_clock_extension(value['queue'])
    source=verify_source_tree(value)
    contents,opened=read_roles(value);verify_common_bindings(value,contents)
    auth=_read_json(value['authorization']['path'],pin=value['authorization']['sha256'])
    if auth.get('approved') is not True or any(auth.get(k)!=value['authorization'][k] for k in ('issued_at','expires_at','stages')):raise ValueError('approval authority binding')
    verify_queue_authority(value,auth);verify_input_origin(value,auth)
    queue=value['queue'];receipt=_read_json(queue['receipt_path'],pin=queue['receipt_sha256'])
    expected_queue={k:v for k,v in queue.items() if k not in ('receipt_path','receipt_sha256','clock_extension')};expected_queue['attempts']=queue['attempts'] if autonomous_extension(queue.get('extension')) else 3 if 'extension' in queue else 1
    if receipt!=expected_queue:raise ValueError('original queue receipt drift')
    budget=MonotonicBudget(queue);budget.remaining_seconds(time.time())
    reader=SafetyFeedReader(value['g_feed']['path'],value['g_feed']['sha256'],runs)
    feed=reader.observe(now=time.time())
    expected_g={'job_id_hash':sha(canonical([r['run_id'] for r in runs])),'start_identity_hash':sha(canonical(runs))}
    if reader.descriptor['g_job_identity']!=contents['g-feed-descriptor']['g_job_identity'] or reader.descriptor['g_job_identity']!=expected_g:raise ValueError('G fixed identities drift')
    if value['stage'].endswith('-verify'):
        predecessor_root,prior=read_closed_predecessor(value)
        for new in (value['output_root'],value['control_root']):
            if new==str(predecessor_root) or new.startswith(str(predecessor_root)+'/') or str(predecessor_root).startswith(new+'/'):raise ValueError('readonly predecessor root overlap')
    output=Path(value['output_root']);control=Path(value['control_root'])
    if output.exists() or control.exists() or output.is_symlink() or control.is_symlink():raise ValueError('exclusive operation root absence required')
    check_queue_filesystem(value,auth,reader.descriptor)
    reserve_queue_attempt(value,auth,manifest_sha256)
    kernel=LinuxKernel();owner=kernel.identity(os.getpid());baseline={r['pid'] for r in kernel.processes() if r['ppid']==os.getpid()}
    output.mkdir(mode=0o700);control.mkdir(mode=0o700)
    start={'schema_id':'p4h-real-controller-start-v1','operation_id':value['operation_id'],'manifest_sha256':manifest_sha256,'controller':owner,'output_root':str(output),'control_root':str(control),'output_identity':{'device':output.stat().st_dev,'inode':output.stat().st_ino},'control_identity':{'device':control.stat().st_dev,'inode':control.stat().st_ino},'source':source,'opened_roles':opened,'g_receipt_sha256':feed['receipt_sha256'],'issued_at':time.time()}
    write_once(control/'operation.json',value)
    start_pin=write_once(control/'start.json',start)
    gate=RuntimeGate(queue_charge_root(queue),control/'stop.json',budget,reader,value['limits'])
    gate.check()
    return {'contents':contents,'opened':opened,'kernel':kernel,'budget':budget,'reader':reader,'gate':gate,'start':start,'start_sha256':start_pin,'baseline':baseline}


# One known completed pilot, append-only serialization repair. No generic fallback.
PILOT07_LEGACY_SEAL_SHA256 = '53f83b393433cd73d8adc15bf84f6468264c0bb6fac8188799a9805d2a2bacd1'
PILOT07_CONSUMER_SHA256 = '9a59da17637249d3185fc7e5983d9aa1fd841cd47c6601f37af93e0a09e3612a'


def normalize_legacy_artifact_digest(original):
    import copy
    artifacts=original['artifact_hashes']
    if original.get('complete_artifact_set_sha256')!=sha(canonical(artifacts)):
        raise ValueError('exact legacy compact artifact digest required')
    fixed=copy.deepcopy(original)
    fixed['complete_artifact_set_sha256']=sha(plan.canonical_bytes(artifacts))
    return fixed


def digest_repair_predecessor(value,auth):
    repair=auth.get('digest_repair')
    if value['stage']!='pilot-verify' or not isinstance(repair,dict) or set(repair)!={'schema_id','original_seal'} or repair['schema_id']!='p4h-pilot07-newline-digest-repair-v1':
        raise ValueError('explicit pilot07 verifier-only repair authority required')
    ref=repair['original_seal']
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or ref['sha256']!=PILOT07_LEGACY_SEAL_SHA256 or ref['path']!='/workspace/vkgr-p4h-sol03-pilot-v1/attempt07/pilot-output/sealed.json':
        raise ValueError('fixed original pilot07 sealed bytes required')
    original=_read_json(ref['path'],pin=ref['sha256'],cap=64*1024**2)
    if value['predecessor']!=normalize_legacy_artifact_digest(original):
        raise ValueError('repair may change only complete artifact digest convention')
    control=Path('/workspace/vkgr-p4h-sol03-pilot-v1/attempt07/pilot-control')
    prior=_read_json(control/'operation.json',pin=original['predecessor_operation_manifest_sha256'],cap=1048576)
    if prior['stage']!='pilot' or prior['queue']!=value['queue'] or prior['source_pins']['src/vkgr/retrieval_sasrec_remote.py']!=PILOT07_CONSUMER_SHA256:
        raise ValueError('same actual original source and spent queue required')
    changed={'src/vkgr/retrieval_sasrec_remote.py','tests/test_retrieval_sasrec_remote.py'}
    if {k:v for k,v in prior['source_pins'].items() if k not in changed}!={k:v for k,v in value['source_pins'].items() if k not in changed}:
        raise ValueError('repair cannot change scientific source or inventory')
    verify_source_tree(prior)
    index_rec=next(r for r in original['artifact_hashes'] if r['role']=='pilot-artifact-index')
    index=_read_json(index_rec['path'],pin=index_rec['sha256'],cap=64*1024**2)
    indexed=[r for r in original['artifact_hashes'] if r['role']!='pilot-artifact-index']
    if index['artifacts']!=indexed or index['complete_artifact_set_sha256']!=sha(canonical(indexed)):
        raise ValueError('original index must prove identical legacy convention and complete artifact membership')
    return Path(ref['path']).parent,prior


def read_closed_predecessor(value):
    sealed=value['predecessor'];plan.validate_verifier_manifest({'stage':value['stage'],'roles':value['roles']},sealed)
    group=plan.VERIFIER_ROLE_SOURCES[value['stage']];wrapper=next(r for r in value['roles'] if r['role']==group['sealed-wrapper'][0])
    if _read_json(wrapper['path'],pin=wrapper['sha256'],cap=64*1024**2)!=sealed:raise ValueError('external sealed predecessor authority drift')
    auth=_read_json(value['authorization']['path'],pin=value['authorization']['sha256'])
    root=Path(wrapper['path']).parent
    if 'digest_repair' in auth:root,_=digest_repair_predecessor(value,auth)
    terminal=_read_json(root/'terminal.json',pin=sealed['terminal_receipt_sha256'])
    closure=_read_json(root/'closure.json',pin=sealed['closure_receipt_sha256'])
    if closure.get('state')!='CLOSED' or terminal.get('status')!='completed+unverified' or terminal['manifest_sha256']!=sealed['predecessor_operation_manifest_sha256'] or terminal['operation_id']!=sealed['predecessor_operation_id'] or terminal['artifact_hashes']!=sealed['artifact_hashes'] or terminal['input_hashes']!=sealed['input_hashes'] or terminal['closure_sha256']!=sealed['closure_receipt_sha256']:raise ValueError('predecessor not actually CLOSED')
    prior_control=Path(terminal['control_root'])
    start=_read_json(prior_control/'start.json',pin=sealed['controller_start_receipt_sha256'],cap=64*1024**2)
    admission=_read_json(prior_control/'admission.json',pin=sealed['admission_sha256'])
    inventory=_read_json(root/'worker-birth-inventory.json',pin=sealed['worker_birth_inventory_sha256'])
    validate_predecessor_cross_bindings(value,sealed,terminal,closure,start,admission,inventory)
    if sha(canonical({'device':prior_control.stat().st_dev,'inode':prior_control.stat().st_ino}))!=sealed['predecessor_control_root_identity']:raise ValueError('predecessor control identity drift')
    auth=_read_json(value['authorization']['path'],pin=value['authorization']['sha256'])
    if auth.get('predecessor')!={'path':wrapper['path'],'sha256':wrapper['sha256']}:raise ValueError('external authority must bind sealed predecessor')
    prior_manifest=_read_json(prior_control/'operation.json',pin=sealed['predecessor_operation_manifest_sha256'],cap=1048576)
    prior_descriptor=_read_json(prior_manifest['g_feed']['path'],pin=prior_manifest['g_feed']['sha256'])
    matches=list(Path(prior_descriptor['feed_root']).glob('receipt-*-'+sealed['final_g_receipt_sha256']+'.json'))
    if len(matches)!=1:raise ValueError('predecessor final G receipt identity')
    final=_read_json(matches[0],pin=sealed['final_g_receipt_sha256']);payload=_read_json(Path(prior_descriptor['feed_root'])/('payload-'+final['payload_sha256']+'.json'),pin=final['payload_sha256'],cap=262144);check_payload_binding(final,payload)
    processes=linux_processes()
    if any(r['sid'] in closure['owned_sessions'] for r in processes):raise ValueError('predecessor group reopened')
    if sha(canonical(terminal['output_identity']))!=sealed['predecessor_output_root_identity'] or {'device':root.stat().st_dev,'inode':root.stat().st_ino}!=terminal['output_identity']:raise ValueError('predecessor root identity drift')
    for rec in sealed['artifact_hashes']:read_bound(rec['path'],rec['sha256'],max(1,rec['size']))
    return root,terminal


# P4H-05 S02 append-only acceptance recovery. The original formal-train finished
# training and saved model-artifacts/state.pt, then its worker exited on a /proc
# ESRCH race before fresh-best export. Recovery only reads that exact model with
# the fixed runtime: no optimizer, no training, original failed terminal kept.
S02_RECOVERY_SCHEMA = 'p4h-formal-s02-acceptance-recovery-v1'
S02_ORIGINAL_CONSUMER_SHA256 = 'd7fc05ae0c20a0ae4f93776ada4035ddf5d8c9023364e1be0d4fe6255fc9bc87'
_S02_ROOT = '/workspace/vkgr-p4h-main05-formal-v1'
S02_RECOVERY_REFS = {
    'original_registration':{'path':_S02_ROOT+'/attempt-P4H-S02-1.json','sha256':'a774ba3e068c3cd48446e0e559d44f94beb4619b4d55036ea21d28109ec7f3a7'},
    'original_operation':{'path':_S02_ROOT+'/P4H-S02/formal-train-control/operation.json','sha256':'921fb442fa5873e1193f18b6b5bcd428825fffc71fbbad7ceeee1f954b5d5d1c'},
    'original_terminal':{'path':_S02_ROOT+'/P4H-S02/formal-train-output/terminal.json','sha256':'181714445fefdae1191590c49f952eea1dfaf02fd548a73ae6136d8d82b280de'},
    'original_closure':{'path':_S02_ROOT+'/P4H-S02/formal-train-output/closure.json','sha256':'c6b0e3ae0c14829229331ce99f6a2fa3bfcb9e62264ca2e8e7376ca10582feb8'},
    'original_worker_stderr':{'path':_S02_ROOT+'/P4H-S02/formal-train-control/worker.stderr','sha256':'87c55e473a3f0bfaee9c23f2098d213ce0559d6c4f05ba00406bd77571732e87'},
    'original_model_artifacts':{'path':_S02_ROOT+'/P4H-S02/formal-train-output/model/model-artifacts.json','sha256':'1e6d570b7f981f2d262afd1b424c8843d0e37eecbf9cc764352521d9a44fc1e6'},
    'original_driver_result':{'path':_S02_ROOT+'/driver-result.json','sha256':'3b7abbc8be36411d5a219df71d825bfcd324a2266b21a264ea0614e14093a45c'},
}
S02_RECOVERY_REGISTRATION = _S02_ROOT+'/attempt-P4H-S02-1-acceptance-recovery-1.json'
RECOVERY_CHANGED_SOURCES = {'src/vkgr/retrieval_sasrec_remote.py','tests/test_retrieval_sasrec_remote.py'}


def acceptance_recovery_original(value,auth):
    """Bind the exact original S02 attempt/model; any drift stays fail-closed."""
    recovery=auth.get('acceptance_recovery')
    slot=value.get('slot')
    if value.get('stage') not in ('formal-train','formal-verify') or not isinstance(slot,dict) or slot.get('slot_id')!='P4H-S02' or recovery!={'schema_id':S02_RECOVERY_SCHEMA,**S02_RECOVERY_REFS}:raise ValueError('explicit pinned S02 acceptance recovery authority required')
    if value['source_pins'].get('src/vkgr/retrieval_sasrec_remote.py')==S02_ORIGINAL_CONSUMER_SHA256:raise ValueError('recovery requires the reviewed fixed runtime')
    def load(key,cap=64*1024**2):return _read_json(recovery[key]['path'],pin=recovery[key]['sha256'],cap=cap)
    op=load('original_operation',1048576);manifest_pin=recovery['original_operation']['sha256']
    if op.get('stage')!='formal-train' or op.get('operation_id')!='p4h-main05-formal01-P4H-S02-formal-train' or op.get('slot')!=slot or op.get('queue')!=value['queue'] or op.get('predecessor') is not None or op['source_pins'].get('src/vkgr/retrieval_sasrec_remote.py')!=S02_ORIGINAL_CONSUMER_SHA256:raise ValueError('original S02 operation/slot/queue identity')
    if {k:v for k,v in op['source_pins'].items() if k not in RECOVERY_CHANGED_SOURCES}!={k:v for k,v in value['source_pins'].items() if k not in RECOVERY_CHANGED_SOURCES}:raise ValueError('recovery cannot change scientific source or inventory')
    inputs=lambda v:{r['role']:(r['path'],r['sha256']) for r in v['roles'] if r.get('source')=='input_hashes'}
    if not inputs(op) or inputs(op)!=inputs(value):raise ValueError('recovery requires the exact original input files')
    verify_source_tree(op)
    registration=load('original_registration')
    expected={'manifest_sha256':manifest_pin,'operation_id':op['operation_id'],'output_root':op['output_root'],'control_root':op['control_root'],'queue':op['queue'],'attempt':{'slot_id':'P4H-S02','number':1,'reason':'initial','previous_terminal':None}}
    if set(registration)!=set(expected)|{'issued_at'} or any(registration[k]!=v for k,v in expected.items()):raise ValueError('original S02 registration binding')
    terminal=load('original_terminal');closure=load('original_closure');driver=load('original_driver_result')
    if terminal.get('status')!='failed+spent' or terminal.get('failure')!='ValueError: actual worker failed' or terminal.get('manifest_sha256')!=manifest_pin or terminal.get('closure_sha256')!=recovery['original_closure']['sha256'] or terminal.get('worker_released') is not True or terminal.get('optimizer_steps') is not None or terminal.get('queue')!=op['queue'] or terminal.get('attempt')!=1 or terminal.get('operation_id')!=op['operation_id']:raise ValueError('original failed+spent terminal must stay exactly as written')
    if closure.get('state')!='CLOSED' or closure.get('actual_worker_exit')!=1 or closure.get('signals_sent')!=0 or closure.get('manifest_sha256')!=manifest_pin or closure.get('failure') is not None or not closure.get('owned_sessions'):raise ValueError('original actual owned CLOSED worker exit required')
    stderr=read_bound(recovery['original_worker_stderr']['path'],recovery['original_worker_stderr']['sha256'],65536)
    if not stderr.endswith(b'ProcessLookupError: [Errno 3] No such process\n') or b'in linux_processes' not in stderr or b'fresh=evaluate_formal(' not in stderr:raise ValueError('only the pinned post-training /proc exit race is recoverable')
    publisher=driver.get('publisher_driver_closure') or {}
    if driver.get('status')!='failed+spent' or driver.get('stages')!=[] or driver.get('test_read') is not False or driver.get('additional_slots_submitted') is not False or driver.get('cleanup_failures')!=[] or driver.get('queue')!=op['queue'] or publisher.get('state')!='CLOSED' or publisher.get('actual_worker_exit')!=0:raise ValueError('original driver/publisher CLOSED without other stages')
    output=Path(op['output_root']);fresh=output/'fresh-best'
    if (output/'sealed.json').exists() or (output/'worker-result.json').exists() or not fresh.is_dir() or fresh.is_symlink() or any(fresh.iterdir()):raise ValueError('original must have stopped before fresh-best export/seal')
    sessions=set(closure['owned_sessions'])|set(publisher.get('owned_sessions',[]))
    if any(p['sid'] in sessions for p in linux_processes()):raise ValueError('original owned session reopened')
    index=load('original_model_artifacts');public=index.get('public') if isinstance(index.get('public'),dict) else {}
    binding={'input_sha256':{r['role']:r['sha256'] for r in op['roles'] if r['role'] not in plan.COMMON_ROLES and r.get('source') not in ('artifact_hashes','sealed-wrapper')},'source_sha256':sha(canonical(op['source_pins'])),'test_gate':'closed'}
    model_root=output/'model';index_path=Path(recovery['original_model_artifacts']['path']);state=index.get('state')
    if index_path!=model_root/'model-artifacts.json' or index.get('schema_id')!='p4h-real-model-artifacts-v1' or index.get('binding')!=binding or index.get('slot')!=slot or index.get('test_gate')!='closed' or public.get('status')!='completed+unverified' or public.get('stop_reason') not in ('patience','max_epochs') or public.get('config')!=slot or type(public.get('steps')) is not int or public['steps']!=len(public.get('step_records',[])) or not isinstance(state,dict) or set(state)!={'path','sha256','size'} or state['path']!=str(model_root/'state.pt') or type(state['size']) is not int or not 0<state['size']<=8*1024**3:raise ValueError('original model artifact binding/completion')
    from . import retrieval_sasrec_experiment as science
    selection=science._new_selection()
    for evaluation in public.get('evaluations',[]):science._observe(selection,evaluation)
    if selection!=public.get('selection') or selection['best_epoch'] is None:raise ValueError('original selection replay')
    for evaluation in public['evaluations']:read_bound(output/f"epoch-{evaluation['epoch']}-predictions.json",evaluation['prediction_sha256'],64*1024**2)
    record={'role':'model-artifacts','path':str(index_path),'sha256':recovery['original_model_artifacts']['sha256'],'size':index_path.stat().st_size}
    return {'operation':op,'index':index,'index_record':record,'model_root':model_root,'binding':binding}


def load_recovered_state(recovery,gate):
    """Read-only tensor load of the bound original state; never builds an optimizer."""
    import io
    import torch
    from . import retrieval_sasrec_experiment as science
    index=recovery['index'];rec=index['state']
    gate();blob=read_bound(rec['path'],rec['sha256'],rec['size']);gate()
    loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
    if set(loaded)!={'binding','slot','state'} or loaded['binding']!=recovery['binding'] or loaded['slot']!=index['slot'] or loaded['state']['public']!=index['public']:raise ValueError('recovered tensor/public binding mismatch')
    state=loaded['state'];public=state['public']
    if science._tensor_digest(state['private']['best_model_state'])!=public['checkpoint_roles']['best'] or science._tensor_digest(state['private']['model_state'])!=public['checkpoint_roles']['last']:raise ValueError('recovered best/last tensor hash mismatch')
    return state


def worker_entry(value,manifest_sha256,runs):
    import torch
    from .retrieval_sasrec import SASRecIDTwoTower
    from .training import seed_all
    configure_real_torch()
    validate_remote_manifest(value);verify_clock_extension(value['queue']);verify_source_tree(value);contents,opened=read_roles(value);verify_common_bindings(value,contents)
    control=Path(value['control_root']);output=Path(value['output_root']);start=_read_json(control/'start.json');permit=_read_json(control/'admission.json')
    validate_admission(value,permit,manifest_sha256=manifest_sha256,start_sha256=sha(canonical(start)),g_receipt_sha256=permit['g_receipt_sha256'],now=time.time())
    gate=RuntimeGate(queue_charge_root(value['queue']),control/'stop.json',MonotonicBudget(value['queue']),SafetyFeedReader(value['g_feed']['path'],value['g_feed']['sha256'],runs),value['limits'])
    auth=_read_json(value['authorization']['path'],pin=value['authorization']['sha256']);verify_queue_authority(value,auth);verify_input_origin(value,auth)
    gate.check();binding={'input_sha256':{r['role']:r['sha256'] for r in value['roles'] if r['role'] not in plan.COMMON_ROLES and r.get('source') not in ('artifact_hashes','sealed-wrapper')},'source_sha256':sha(canonical(value['source_pins'])),'test_gate':'closed'}
    if 'digest_repair' in auth:
        _,original_operation=digest_repair_predecessor(value,auth)
        binding['source_sha256']=sha(canonical(original_operation['source_pins']))
    recovery=acceptance_recovery_original(value,auth) if 'acceptance_recovery' in auth else None
    if recovery is not None:
        binding['source_sha256']=recovery['binding']['source_sha256']
        if binding!=recovery['binding']:raise ValueError('recovery binding must equal the original model binding')
    stage=value['stage'];history=contents.get('watch-train',contents.get('formal-history-view'))
    data=compile_training_roles(contents['prepared-train-rows'],history,contents['catalog'],contents['idmap'])
    if stage=='pilot':
        data['sidecar']=contents['train-prefix-target-free-sidecar'];result=run_pilot_worker(data,output,binding=binding,gate=gate.check,device='cuda',capacity_per_arm=2520)
    elif stage=='pilot-verify':
        root,terminal=read_closed_predecessor(value)
        result=verify_actual_pilot(root,terminal,data,contents['train-prefix-target-free-sidecar'],gate=gate.check,binding=binding)
        write_once(output/'verification.json',result)
    else:
        slot=value['slot'];seed_all(slot['seed']);model=SASRecIDTwoTower(torch.zeros((len(data['catalog'])+1,64)),data['supported_indices']).to('cuda')
        batches=finite_batches(data['rows_by_arm'][slot['history_source']],batch_size=512,seed=slot['seed'])
        history_queries=contents['validation-query-target-free'];targets=contents['validation-target']
        if len(history_queries)!=len(targets) or not history_queries:raise ValueError('complete validation bijection')
        queries=[]
        for h,t in zip(history_queries,targets):
            if h['query_id']!=t['query_id'] or t['oov']!=(t['target'] not in data['idmap']):raise ValueError('validation identity/OOV drift')
            queries.append({**h,'target':t['target'],'oov':t['oov']})
        evaluate=lambda current,epoch:evaluate_formal(current,epoch,data['catalog'],queries,data['idmap'],[data['catalog'][i-1] for i in data['supported_indices']],output,gate=gate.check)
        if stage=='formal-train':
            if recovery is not None:
                # Acceptance recovery: read the saved original model only; no optimizer/training.
                state=load_recovered_state(recovery,gate.check);result=state['public']
            else:
                science_output=output/'model';science_output.mkdir(mode=0o700)
                optimizer=torch.optim.AdamW(model.parameters(),lr=slot['learning_rate'],weight_decay=1e-4)
                state=train_model(model,optimizer,batches,data['idmap'],slot,evaluate=evaluate,output=science_output,binding=binding,gate=gate.check);result=state['public']
            model.load_state_dict(state['private']['best_model_state']);fresh_output=output/'fresh-best';fresh_output.mkdir(mode=0o700)
            fresh=evaluate_formal(model,result['selection']['best_epoch'],data['catalog'],queries,data['idmap'],[data['catalog'][i-1] for i in data['supported_indices']],fresh_output,gate=gate.check)
            expected=next(row for row in result['evaluations'] if row['epoch']==result['selection']['best_epoch'])
            if fresh!=expected:raise ValueError('fresh best mismatch')
            write_once(fresh_output/'result.json',fresh)
        elif stage=='formal-verify':
            root,terminal=read_closed_predecessor(value)
            index=next(r for r in terminal['private_artifacts'] if r['role']=='model-artifacts');model_root=root/'model'
            if recovery is not None:
                if index!=recovery['index_record']:raise ValueError('recovered predecessor must bind the original model index')
                model_root=recovery['model_root']
            result=verify_model(model_root,model,batches,data['idmap'],slot,evaluate=evaluate,binding=binding,gate=gate.check,artifact_index_sha256=index['sha256'])
        else:raise ValueError('unsupported actual worker stage')
        write_once(output/'worker-result.json',result)
    write_once(control/'worker-resource-ledger.json',gate.samples)
    return result


def run_controller(value,manifest_path,manifest_sha256,runs):
    prepared=prepare_controller(value,manifest_sha256,runs);control=Path(value['control_root']);output=Path(value['output_root']);gate=prepared['gate'];kernel=prepared['kernel'];handle=None;lease=None;failure=None;result=None;started=time.monotonic();permit=None
    try:
        while not (control/'admission.json').exists():
            if time.time()>=value['authorization']['expires_at']:raise ValueError('approval expired before exact start')
            gate.check();time.sleep(.1)
        permit=_read_json(control/'admission.json')
        feed=gate.check();validate_admission(value,permit,manifest_sha256=manifest_sha256,start_sha256=prepared['start_sha256'],g_receipt_sha256=permit['g_receipt_sha256'],now=time.time())
        if feed['receipt_sha256']!=permit['g_receipt_sha256']:raise ValueError('exact G admission receipt superseded')
        body='import site;site.main()\nimport sys,json,hashlib\nsys.path.insert(0,'+repr(str(Path(value['code_root'])/'src'))+')\nfrom vkgr.retrieval_sasrec_remote import read_bound,worker_entry\nv=json.loads(read_bound('+repr(str(manifest_path))+','+repr(manifest_sha256)+',1048576))\nworker_entry(v,'+repr(manifest_sha256)+','+repr(runs)+')\n'
        handle=start_latched_child(body,control/'worker.stdout',control/'worker.stderr',identity_probe=kernel.identity,python=value['python'])
        lease=KernelLease(handle,kernel=kernel,owner_pid=os.getpid(),controller_pgid=os.getpgrp(),baseline_children=prepared['baseline'],protected_pgids=feed['payload']['protected_pgids'])
        write_once(control/'worker-start.json',{'manifest_sha256':manifest_sha256,'worker':handle['identity'],'controller':prepared['start']['controller'],'admission_sha256':sha(canonical(permit))})
        gate.check();handle['process'].stdin.write(b'1');handle['process'].stdin.flush();handle['process'].stdin.close();handle['released']=True
        while handle['process'].poll() is None:
            lease.refresh(reap=True);gate.check();time.sleep(value['limits']['sample_seconds'])
        if handle['process'].returncode!=0:raise ValueError('actual worker failed')
    except (OSError,ValueError,RuntimeError) as error:failure=type(error).__name__+': '+str(error)
    finally:
        if lease:
            result=lease.close(control/'stop.json',manifest_sha256=manifest_sha256,seal_seconds=value['limits']['seal_seconds'],signal_grace_seconds=value['limits']['signal_grace_seconds'],deadline_monotonic=prepared['budget'].deadline_mono)
        elif handle and not handle['released']:close_unreleased(handle)
        if result is None:result={'state':'CLOSED','actual_worker_exit':None,'owned_sessions':[],'owned_identities':[],'elapsed_seconds':0.,'signals_sent':0,'failure':failure}
        result['schema_id']='p4h-real-closure-v1';result['manifest_sha256']=manifest_sha256
        closure_pin=write_once(output/'closure.json',result)
        terminal={'operation_id':value['operation_id'],'manifest_sha256':manifest_sha256,'status':'completed+unverified' if failure is None and result['state']=='CLOSED' and result['actual_worker_exit']==0 else 'failed+spent','failure':failure,'closure_sha256':closure_pin,'start_sha256':prepared['start_sha256'],'admission_sha256':sha(canonical(permit)) if permit else None,'output_identity':prepared['start']['output_identity'],'control_identity':prepared['start']['control_identity'],'spent':{'wall_seconds':time.monotonic()-started,'new_bytes':queue_tree_usage(output)['bytes']},'resources':gate.samples,'queue':value['queue'],'attempt':value['queue']['attempts'],'control_root':value['control_root'],'worker_released':bool(handle and handle['released']),'optimizer_steps':0 if not handle or not handle['released'] else None}
        if terminal['status']=='completed+unverified':
            try:
                fresh=prepared['reader'].observe(now=time.time())
                terminal,_=seal_completed_operation(value,manifest_sha256,prepared['start'],prepared['start_sha256'],permit,result,gate.samples,time.monotonic()-started,fresh['receipt_sha256'])
            except (OSError,ValueError,RuntimeError) as error:
                terminal['status']='failed+spent';terminal['failure']=type(error).__name__+': '+str(error)
                if not (output/'terminal.json').exists():write_once(output/'terminal.json',terminal)
        else:write_once(output/'terminal.json',terminal)
    return terminal


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='Reviewed P4H real Linux runtime; no implicit submission')
    parser.add_argument('command',choices=['prepare-inputs','prepare-controller','worker','serve-publisher','admit'])
    parser.add_argument('--manifest',required=True);parser.add_argument('--sha256',required=True)
    args=parser.parse_args(argv);value=_read_json(args.manifest,pin=args.sha256,cap=1048576)
    if args.command=='prepare-inputs':return prepare_actual_inputs(value)
    if args.command=='serve-publisher':
        if set(value)!={'root','runs','deadline','interval','allowed_p'}:raise ValueError('publisher pinned envelope')
        return serve_publisher(value['root'],value['runs'],deadline=value['deadline'],interval=value['interval'],allowed_p=value['allowed_p'])
    validate_remote_manifest(value)
    descriptor=_read_json(value['g_feed']['path'],pin=value['g_feed']['sha256']);runs=_read_json(Path(descriptor['feed_root'])/'runs.json')
    if args.command=='prepare-controller':return run_controller(value,args.manifest,args.sha256,runs)
    if args.command=='worker':return worker_entry(value,args.sha256,runs)
    if args.command=='admit':
        start_path=Path(value['control_root'])/'start.json';start=_read_json(start_path);reader=SafetyFeedReader(value['g_feed']['path'],value['g_feed']['sha256'],runs);feed=reader.observe(now=time.time())
        if start['manifest_sha256']!=args.sha256:raise ValueError('start manifest drift')
        admission=make_admission(value,manifest_sha256=args.sha256,start_sha256=sha(canonical(start)),g_receipt_sha256=feed['receipt_sha256'],now=time.time());write_once(Path(value['control_root'])/'admission.json',admission);return admission






def _artifact_record(role,path):
    path=Path(path)
    return {'role':role,'path':str(path),'sha256':operation._file_sha256(path),'size':path.stat().st_size}


def complete_artifact_digest(artifacts):
    return sha(plan.canonical_bytes(artifacts))


def seal_completed_operation(value,manifest_pin,start,start_pin,permit,closure,resources,wall_seconds,final_g_pin):
    if closure.get('state')!='CLOSED' or closure.get('actual_worker_exit')!=0 or not closure.get('owned_sessions') or not resources or not operation._pin(final_g_pin):raise ValueError('actual successful owned closure required')
    if any(r['sid'] in closure['owned_sessions'] for r in linux_processes()):raise ValueError('descendant remains: no sealed success')
    output=Path(value['output_root']);stage=value['stage'];operation_id=value['operation_id']
    closure_path=output/'closure.json'
    if closure_path.exists():
        closure_pin=operation._file_sha256(closure_path)
        if _read_json(closure_path,pin=closure_pin)!=closure:raise ValueError('closure bytes mismatch')
    else:closure_pin=write_once(closure_path,closure)
    spent={'wall_seconds':wall_seconds,'new_bytes':queue_tree_usage(output)['bytes']}
    private=[]
    for path in sorted(output.rglob('*')):
        if not path.is_file() or path.name=='closure.json':continue
        role='model-artifacts' if path.name=='model-artifacts.json' else path.relative_to(output).as_posix()
        private.append(_artifact_record(role,path))
    events=[{'kind':'controller_start','sha256':start_pin},{'kind':'admission','sha256':sha(canonical(permit))}]
    payloads={};base={'schema_version':1,'operation_id':operation_id}
    if stage=='pilot':
        worker=_read_json(output/'pilot-worker.json',cap=64*1024**2);arms={}
        for arm,row in worker['arms'].items():
            measures=dict(row['measurements']);measures['safe_stop_closure']={'seconds':closure['elapsed_seconds'],'bytes':len(canonical(closure))}
            arms[arm]={'history_source':arm,'optimizer_steps':row['optimizer_steps'],'measurements':measures}
        payloads['pilot-capacity-measurement-index']={**base,'arms':arms,'measurements':{a:r['measurements'] for a,r in arms.items()},'units':{'seconds':'seconds','bytes':'bytes'}}
        events.append({'kind':'pilot_summary','sha256':operation._file_sha256(output/'pilot-worker.json')})
        prefix='pilot'
    elif stage=='formal-train':
        auth=_read_json(value['authorization']['path'],pin=value['authorization']['sha256'])
        recovery=acceptance_recovery_original(value,auth) if 'acceptance_recovery' in auth else None
        index_path=Path(recovery['index_record']['path']) if recovery else output/'model/model-artifacts.json'
        index=_read_json(index_path,cap=64*1024**2);public=index['public'];best=public['selection']['best_epoch'];last=public['selection']['epoch']
        record=_artifact_record('model-artifacts',index_path)
        if recovery is not None:
            if record!=recovery['index_record'] or index!=recovery['index']:raise ValueError('recovered original model index drift')
            private.append(record)
        def checkpoint(name,epoch):return {k:record[k] for k in ('path','sha256','size')}|{'checkpoint_role':name,'source_checkpoint_sha256':public['checkpoint_roles'][name],'epoch':epoch,'step':public['steps'],'operation_id':operation_id}
        payloads['formal-checkpoint-index']={**base,'best':checkpoint('best',best),'last':checkpoint('last',last),'preemption':None}
        payloads['formal-epoch-metric-index']={**base,'records':public['evaluations'],'selection_replay_sha256':sha(canonical(public['selection']))}
        hashes=[row['prediction_sha256'] for row in public['evaluations']];fresh=_read_json(output/'fresh-best/result.json')
        expected=next(row for row in public['evaluations'] if row['epoch']==best)
        if fresh!=expected:raise ValueError('mutating final fresh best export/exact mismatch')
        payloads['formal-prediction-hash-index']={**base,'epoch_hashes':hashes,'fresh_best_hash':fresh['prediction_sha256']}
        payloads['formal-draw-mask-provenance']={**base,'step_records':public['step_records'],'aggregate_sha256':sha(canonical(public['step_records']))}
        events.extend(index['science_events']);prefix='formal'
        if recovery is not None:events.append({'kind':'acceptance_recovery','schema_id':S02_RECOVERY_SCHEMA,'original_operation_manifest_sha256':S02_RECOVERY_REFS['original_operation']['sha256'],'original_terminal_sha256':S02_RECOVERY_REFS['original_terminal']['sha256'],'original_terminal_status_preserved':'failed+spent','original_model_artifacts_sha256':S02_RECOVERY_REFS['original_model_artifacts']['sha256'],'optimizer_steps_during_recovery':0})
    elif stage.endswith('-verify'):
        verification=_read_json(output/('verification.json' if stage=='pilot-verify' else 'worker-result.json'))
        if verification.get('status') not in ('pilot_completed+verified','completed+verified'):raise ValueError('independent verification missing')
        terminal={**base,'manifest_sha256':manifest_pin,'status':'completed+verified','verification':verification,'private_artifacts':private,'closure_sha256':closure_pin,'start_sha256':start_pin,'admission_sha256':sha(canonical(permit)),'final_g_receipt_sha256':final_g_pin,'output_identity':start['output_identity'],'control_identity':start['control_identity'],'spent':spent,'queue':value['queue'],'attempt':value['queue']['attempts'],'control_root':value['control_root']}
        write_once(output/'terminal.json',terminal);return terminal,None
    else:raise ValueError('unimplemented seal stage')
    events.extend([{'kind':'stop','signals_sent':closure['signals_sent']},{'kind':'seal','closure_sha256':closure_pin}])
    input_hashes=[{'role':r['role'],'sha256':r['sha256']} for r in value['roles'] if r['role'] not in plan.COMMON_ROLES]
    opened=start['opened_roles'];opened_map={r['role']:r['sha256'] for r in opened}
    payloads[prefix+'-resource-ledger']={**base,'samples':resources,'peak':{'new_bytes':max(r.get('queue_bytes',0) for r in resources)},'caps':value['queue'],'spent':spent}
    payloads[prefix+'-stage-event-ledger']={**base,'events':events}
    payloads[prefix+'-opened-artifact-ledger']={**base,'opened_roles':sorted(opened_map),'opened_sha256':opened_map}
    artifacts=[]
    for role,payload in payloads.items():
        write_once(output/(role+'.json'),payload);artifacts.append(_artifact_record(role,output/(role+'.json')))
    index_role=prefix+'-artifact-index';index={**base,'artifacts':artifacts.copy(),'complete_artifact_set_sha256':complete_artifact_digest(artifacts)};write_once(output/(index_role+'.json'),index);artifacts.append(_artifact_record(index_role,output/(index_role+'.json')))
    terminal={**base,'manifest_sha256':manifest_pin,'status':'completed+unverified','private_artifacts':private,'artifact_hashes':artifacts,'input_hashes':input_hashes,'closure_sha256':closure_pin,'start_sha256':start_pin,'admission_sha256':sha(canonical(permit)),'final_g_receipt_sha256':final_g_pin,'output_identity':start['output_identity'],'control_identity':start['control_identity'],'spent':spent,'queue':value['queue'],'attempt':value['queue']['attempts'],'control_root':value['control_root']}
    terminal_pin=write_once(output/'terminal.json',terminal)
    inventory_pin=write_once(output/'worker-birth-inventory.json',closure['owned_identities'])
    sealed={'schema_version':1,'predecessor_stage':stage,'predecessor_operation_id':operation_id,'predecessor_operation_manifest_sha256':manifest_pin,'predecessor_output_root_identity':sha(canonical(start['output_identity'])),'predecessor_control_root_identity':sha(canonical(start['control_identity'])),'controller_start_receipt_sha256':start_pin,'admission_sha256':sha(canonical(permit)),'final_g_receipt_sha256':final_g_pin,'worker_birth_inventory_sha256':inventory_pin,'input_hashes':input_hashes,'artifact_hashes':artifacts,'complete_artifact_set_sha256':complete_artifact_digest(artifacts),'terminal_receipt_sha256':terminal_pin,'closure_receipt_sha256':closure_pin,'spent':spent,'state':'CLOSED'}
    write_once(output/'sealed.json',sealed);return terminal,sealed


def verify_actual_pilot(root,terminal,data,queries,*,gate,device='cuda',binding=None):
    import io
    import torch
    from .retrieval_sasrec import SASRecIDTwoTower
    from .retrieval_sasrec_experiment import _draw_digest,verify_pilot
    from .retrieval_experiment import export_vectors
    configure_real_torch()
    root=Path(root);private={r['role']:r for r in terminal['private_artifacts']};gate()
    for rec in private.values():read_bound(rec['path'],rec['sha256'],max(1,rec['size']))
    rec=private['pilot-worker.json'];worker=_read_json(rec['path'],pin=rec['sha256'],cap=64*1024**2)
    if binding is None or worker['binding']!=binding:raise ValueError('current pilot input/source binding differs from predecessor')
    if worker['query_count']!=len(queries) or worker['query_identity_sha256']!=sha(canonical([q['query_id'] for q in queries])):raise ValueError('pilot query denominator drift')
    artifacts={r['role']:r for r in terminal['artifact_hashes']}
    capacity=_read_json(artifacts['pilot-capacity-measurement-index']['path'],pin=artifacts['pilot-capacity-measurement-index']['sha256'])
    cohorts=select_capacity_cohort(queries,per_arm=worker['capacity_per_arm']) if 'capacity_per_arm' in worker else {'like':queries,'observed':queries}
    for arm,row in worker['arms'].items():
        arm_queries=cohorts[arm]
        if 'capacity_per_arm' in worker and (row['capacity_query_count']!=len(arm_queries) or row['query_identity_sha256']!=sha(canonical([q['query_id'] for q in arm_queries]))):raise ValueError('arm capacity identity drift')
        gate();state=private[arm+'-state.pt'];blob=read_bound(state['path'],state['sha256'],state['size']);loaded=torch.load(io.BytesIO(blob),weights_only=True,map_location=device)
        draws=[_draw_digest(b,data['idmap']) for b in finite_batches(data['rows_by_arm'][arm],batch_size=512,seed=42)[:row['optimizer_steps']]]
        if loaded['binding']!=worker['binding'] or loaded['draws']!=draws or sha(canonical(draws))!=row['draw_sha256']:raise ValueError('pilot tensor/draw replay drift')
        model=SASRecIDTwoTower(torch.zeros((len(data['catalog'])+1,64)),data['supported_indices']).to(device);model.load_state_dict(loaded['model'])
        vectors=export_vectors(model,data['catalog'],arm_queries,data['idmap'],512,check=gate)
        if sha(canonical(score_capacity(vectors,data['catalog'],50,gate)))!=row['prediction_sha256'] or sha(canonical(score_capacity(vectors,data['catalog'],80,gate)))!=row['top80_sha256']:raise ValueError('independent pilot exact replay drift')
        if {k:v for k,v in capacity['arms'][arm]['measurements'].items() if k!='safe_stop_closure'}!=row['measurements'] or capacity['arms'][arm]['optimizer_steps']!=row['optimizer_steps']:raise ValueError('actual measurements drift')
        del model,vectors,loaded
        if device=='cuda':torch.cuda.empty_cache()
    if any(r['role'] not in set(plan.COMMON_ROLES+list(plan.PILOT_ROLE_SPECS)) for r in terminal['input_hashes']):raise ValueError('pilot unexpected holdout role')
    receipt={'status':'pilot_completed+unverified','arms':capacity['arms'],'query_identity_sha256':worker['query_identity_sha256'],'validation_target_read':False,'quality_metrics_computed':False,'event_kinds':['optimizer_step','capacity','resource','checkpoint','target_free_decode','stop','seal']}
    result=verify_pilot(receipt);result['optimizer_steps_during_verify']=0;result['query_count']=len(queries);result['fresh_tensor_replays']=2
    return result






def prepare_actual_inputs(spec):
    from .retrieval_ablation_run import PREPARED_SHA
    from .retrieval import PREPARED_SOURCE
    from .data_audit import parquet_rows
    from .retrieval_watch import build_watch
    if not isinstance(spec,dict) or spec.get('stage') not in ('pilot','formal-train') or spec.get('prepared_sha256')!=PREPARED_SHA:raise ValueError('frozen actual producer and stage required before paths')
    if set(spec)!={'stage','prepared','prepared_sha256','raw_root','output','validation'} or not all(_absolute(spec[k]) for k in ('prepared','raw_root','output')) or spec['stage']=='pilot' and spec['validation'] is not None:raise ValueError('exact train-only preparation metadata')
    prepared=Path(spec['prepared']);manifest=_read_json(prepared/'prepare-manifest.json',pin=PREPARED_SHA,cap=4*1024**2)
    if manifest['binding']['consumer']['source_fingerprint']!=PREPARED_SOURCE or manifest['status']!='success' or manifest['task_id']!='P4B':raise ValueError('frozen prepared producer drift')
    artifacts=manifest['artifacts'];payloads={}
    if set(artifacts)!={'catalog.json','content.npy','examples.jsonl','idmap.json','support.json'}:raise ValueError('original prepared members drift')
    for name,pin in artifacts.items():payloads[name]=read_bound(prepared/name,pin,256*1024**2)
    examples=[json.loads(line) for line in payloads['examples.jsonl'].splitlines()];catalog=json.loads(payloads['catalog.json']);idmap=json.loads(payloads['idmap.json']);support=json.loads(payloads['support.json'])
    if len(examples)!=manifest['examples'] or len(catalog)!=manifest['catalog_items']:raise ValueError('complete prepared count drift')
    records=manifest['binding']['input_records'];names=[f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet' for w in range(25)];raw=[r for r in records if r['role'] in names]
    if [r['role'] for r in raw]!=names:raise ValueError('full ordered train partition evidence required')
    def validate_raw():
        for rec in raw:
            data=read_bound(Path(spec['raw_root'])/rec['role'],rec['sha256'],max(1,rec['size_bytes']))
            if len(data)!=rec['size_bytes']:raise ValueError('train raw size drift')
    validate_raw()
    def partitions():
        for week,rec in enumerate(raw):yield week,parquet_rows(Path(spec['raw_root'])/rec['role'])
    queries=None;watch_queries=None
    if spec['stage']=='formal-train':
        v=spec['validation'];expected=next(r for r in manifest['binding']['query_pins'] if r['role']=='validation-queries.json')
        if not isinstance(v,dict) or set(v)!={'path','sha256','count'} or v['sha256']!=expected['sha256'] or not _absolute(v['path']) or type(v['count']) is not int or v['count']<=0:raise ValueError('full original validation pin required')
        queries=_read_json(v['path'],pin=v['sha256'],cap=64*1024**2)
        if len(queries)!=v['count']:raise ValueError('full validation denominator mismatch')
        _,watch_queries,_=build_watch(partitions(),examples,queries,catalog)
    result=materialize_training_inputs(examples,catalog,idmap,support,partitions(),spec['output'],capacity_per_arm=2520);validate_raw()
    if queries is not None:
        for arm,view in [('like',queries),('observed',watch_queries)]:
            rows=[{'query_id':q['user_id'],'user_id':q['user_id'],'history':q['history'],'history_positions':[encode_position(p) for p in q['history_positions']]} for q in view]
            path=Path(spec['output'])/('validation-query-'+arm+'.json');write_once(path,rows);result['artifacts']['validation-query-'+arm]=_artifact_record('validation-query-'+arm,path)
        targets=[{'query_id':q['user_id'],'target':q['target'],'oov':q['oov']} for q in queries];path=Path(spec['output'])/'validation-target.json';write_once(path,targets);result['artifacts']['validation-target']=_artifact_record('validation-target',path)
    receipt={'schema_id':'p4h-real-input-producer-v1','consumer_source_sha256':operation._file_sha256(Path(__file__)),'prepared_manifest_path':str(prepared/'prepare-manifest.json'),'prepared_manifest_sha256':PREPARED_SHA,'raw_root':spec['raw_root'],'raw_records':raw,'artifacts':result['artifacts'],'stage':spec['stage'],'examples':len(examples),'catalog_items':len(catalog),'validation_count':len(queries) if queries is not None else None,'test_gate':'closed'}
    write_once(Path(spec['output'])/'producer.json',receipt)
    return receipt


def verify_input_origin(value,auth):
    origin=auth.get('origin')
    if not isinstance(origin,dict) or set(origin)!={'path','sha256'}:raise ValueError('external real producer authority required')
    producer=_read_json(origin['path'],pin=origin['sha256'],cap=4*1024**2)
    from .retrieval_ablation_run import PREPARED_SHA
    origin_consumer=value['source_pins']['src/vkgr/retrieval_sasrec_remote.py']
    if 'digest_repair' in auth:
        _,original_operation=digest_repair_predecessor(value,auth)
        origin_consumer=original_operation['source_pins']['src/vkgr/retrieval_sasrec_remote.py']
    if 'acceptance_recovery' in auth:
        origin_consumer=acceptance_recovery_original(value,auth)['operation']['source_pins']['src/vkgr/retrieval_sasrec_remote.py']
    if producer.get('schema_id')!='p4h-real-input-producer-v1' or producer['prepared_manifest_sha256']!=PREPARED_SHA or producer['consumer_source_sha256']!=origin_consumer or producer['test_gate']!='closed':raise ValueError('real input consumer/origin drift')
    original=_read_json(producer['prepared_manifest_path'],pin=PREPARED_SHA,cap=4*1024**2)
    if producer['raw_records']!=[r for r in original['binding']['input_records'] if r['role'].startswith('subsamples/ur0.01_ir0.01/train/')]:raise ValueError('frozen raw producer evidence drift')
    for r in producer['raw_records']:read_bound(Path(producer['raw_root'])/r['role'],r['sha256'],max(1,r['size_bytes']))
    input_map={r['role']:r['sha256'] for r in value['roles'] if r['role'] not in plan.COMMON_ROLES and r.get('source') not in ('artifact_hashes','sealed-wrapper')}
    if auth.get('input_hashes')!=input_map or auth.get('source_sha256')!=sha(canonical(value['source_pins'])):raise ValueError('approval inputs/source binding drift')
    aliases={'formal-history-view':'like-history-view' if value['slot'] and value['slot']['history_source']=='like' else 'watch-train','validation-query-target-free':'validation-query-'+value['slot']['history_source'] if value['slot'] else None}
    for role,pin in input_map.items():
        if role in ('registered-slot-ledger','mean-control-identity'):continue
        rec=producer['artifacts'].get(aliases.get(role,role))
        if rec is None or rec['sha256']!=pin:raise ValueError('materialized input/origin mismatch')
        read_bound(rec['path'],pin,max(1,rec['size']))
    return producer






def verify_queue_authority(value,auth):
    queue=value['queue'];verify_clock_extension(queue);attempt=auth.get('attempt');slot=value['slot']['slot_id'] if value.get('slot') else 'pilot'
    if auth.get('queue')!=queue or not isinstance(attempt,dict) or set(attempt)!={'slot_id','number','reason','previous_terminal'} or attempt['slot_id']!=slot or attempt['number']!=queue['attempts']:raise ValueError('external original queue/attempt authority binding required')
    if 'extension' in queue:
        Budget(queue['kind'],**{k:queue[k] for k in ('started_at','deadline','disk_cap','attempts')},extension=queue['extension'],clock_extension=queue.get('clock_extension'))
        if attempt['reason']!='explicit_budget_extension' or attempt['previous_terminal'] is not None:raise ValueError('extension is explicit new human grant, not automatic retry')
        verify_explicit_extension(queue)
    elif queue['attempts']==1:
        if attempt['reason']!='initial' or attempt['previous_terminal'] is not None:raise ValueError('initial attempt identity drift')
    elif queue['attempts']==2:
        previous=attempt['previous_terminal']
        if not isinstance(previous,dict) or set(previous)!={'path','sha256'} or not str(previous['path']).startswith(str(Path(queue['receipt_path']).parent)+'/'):raise ValueError('previous spent terminal authority required')
        terminal=_read_json(previous['path'],pin=previous['sha256'],cap=64*1024**2)
        closure=_read_json(Path(previous['path']).parent/'closure.json',pin=terminal['closure_sha256'])
        before=terminal['queue'];base_keys=set(queue)-{'attempts'}
        if any(before[k]!=queue[k] for k in base_keys) or terminal.get('attempt')!=1 or terminal['status']!='failed+spent' or closure.get('state')!='CLOSED':raise ValueError('retry cannot reset clock/disk or retry success/unsafe terminal')
        reason=attempt['reason'];steps=terminal.get('optimizer_steps')
        if reason=='transport_before_step0':
            if steps!=0 or terminal.get('worker_released') is not False:raise ValueError('transport retry after scientific release forbidden')
        elif reason=='infrastructure_failure':
            if terminal.get('failure_kind')!='infrastructure_failure' or type(steps) is not int or steps<0:raise ValueError('infrastructure retry requires measured classified terminal')
        else:raise ValueError('quality/data/hash/resource/nonfinite/resume retry forbidden')
    else:raise ValueError('attempt finite ceiling')
    return True


PREWORKER_REPAIR_GRANT_SHA256 = '952a204bc09a088ff805f2a13e2672e457b8b7d4f8a1e9ad3111ae04a65663e3'


def preworker_registration_path(value,auth):
    """Append-only continuation of one unadmitted science attempt, never resume.

    Three bounded pre-worker revisions share the original clock/disk/attempt.
    A prior admission or worker-start prohibits this path, conservatively.
    """
    repair=auth.get('preworker_repair');root=queue_charge_root(value['queue'])
    fields={'schema_id','generation','human_grant','previous_attempt','previous_driver','previous_controller_closure'}
    if not isinstance(repair,dict) or set(repair)!=fields or repair['schema_id']!='p4h-explicit-preworker-repair-v1' or value['queue']['kind']!='pilot' or value['queue']['attempts']!=3 or 'extension' not in value['queue'] or type(repair['generation']) is not int or not 1<=repair['generation']<=3:raise ValueError('explicit bounded pre-worker authority required')
    generation=repair['generation']
    for key in fields-{'schema_id','generation'}:
        ref=repair[key]
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not str(ref['path']).startswith(str(root)+'/') or not operation._pin(ref['sha256']):raise ValueError('pre-worker authority reference')
    if repair['human_grant']['sha256']!=PREWORKER_REPAIR_GRANT_SHA256:raise ValueError('unapproved pre-worker grant')
    grant=_read_json(repair['human_grant']['path'],pin=repair['human_grant']['sha256'])
    if grant.get('human_answer')!='允许原窗口内自主修复准入编排':raise ValueError('explicit human pre-worker scope required')
    expected=root/('attempt-pilot-3.json' if generation==1 else f'attempt-pilot-3-preworker-{generation-1}.json')
    if repair['previous_attempt']['path']!=str(expected):raise ValueError('ordered original registration ancestry required')
    previous=_read_json(expected,pin=repair['previous_attempt']['sha256'])
    basequeue=lambda q:{k:v for k,v in q.items() if k!='clock_extension'}
    if basequeue(previous['queue'])!=basequeue(value['queue']) or previous['queue'].get('clock_extension') not in (None,value['queue'].get('clock_extension')) or previous['attempt']!=auth['attempt']:raise ValueError('pre-worker continuation cannot reset budget/attempt')
    control=Path(previous['control_root']);old=_read_json(control/'operation.json',pin=previous['manifest_sha256'],cap=1048576)
    if old['stage']!='pilot' or old['queue']!=previous['queue'] or old['output_root']!=previous['output_root'] or old['control_root']!=str(control):raise ValueError('prior unadmitted operation identity')
    if any((control/n).exists() or (control/n).is_symlink() for n in ('admission.json','worker-start.json','worker-resource-ledger.json')):raise ValueError('admitted or scientific worker attempt cannot continue')
    if any(value[k]==old[k] or value[k].startswith(old[k]+'/') or old[k].startswith(value[k]+'/') for k in ('output_root','control_root')):raise ValueError('preserve exclusive prior operation roots')
    changed={'src/vkgr/retrieval_sasrec_remote.py','tests/test_retrieval_sasrec_remote.py'}
    if {k:v for k,v in old['source_pins'].items() if k not in changed}!={k:v for k,v in value['source_pins'].items() if k not in changed}:raise ValueError('frozen scientific source changed')
    inputs=lambda v:{r['role']:r['sha256'] for r in v['roles'] if r['role'] in plan.VERIFIER_ROLE_SOURCES['pilot-verify']['input_hashes']}
    if inputs(old)!=inputs(value):raise ValueError('same train/catalog/cohort inputs required')
    start=_read_json(control/'start.json',cap=64*1024**2)
    closed=_read_json(repair['previous_controller_closure']['path'],pin=repair['previous_controller_closure']['sha256'],cap=64*1024**2)
    failed=_read_json(repair['previous_driver']['path'],pin=repair['previous_driver']['sha256'],cap=64*1024**2)
    publisher=failed.get('publisher_driver_closure',{})
    prior_root=control.parent
    if Path(previous['output_root']).parent!=prior_root or repair['previous_driver']['path']!=str(prior_root/'driver-result.json') or repair['previous_controller_closure']['path']!=str(prior_root/'pilot-operation.driver-closure.json') or failed.get('queue')!=previous['queue']:raise ValueError('prior driver/root/queue binding required')
    intent=_read_json(prior_root/'driver-intent.json')
    if intent.get('queue')!=previous['queue'] or intent.get('source_sha256')!=sha(canonical(old['source_pins'])) or start['controller']['ppid']!=intent['identity']['pid']:raise ValueError('actual driver/controller origin binding required')
    descriptor=_read_json(old['g_feed']['path'],pin=old['g_feed']['sha256'])
    if descriptor['feed_root']!=str(prior_root/'feed'):raise ValueError('prior publisher root binding')
    pubstop=_read_json(Path(descriptor['feed_root'])/'publisher-closure.json')
    expected_publisher=descriptor['publisher_process']
    pubidentity={'pid':expected_publisher['pid'],'pgid':expected_publisher['pid'],'sid':expected_publisher['pid'],'birth':expected_publisher['start_ticks']}
    if expected_publisher['executable_sha256']!=old['python_sha256'] or pubstop.get('status')!='CLOSED' or pubstop.get('reason')!='cooperative_stop' or pubstop.get('descriptor_sha256')!=old['g_feed']['sha256'] or pubstop.get('publisher_process')!=expected_publisher or pubstop.get('lock_fd_closed') is not True or not publisher.get('owned_sessions') or not any(_identity(p)==_identity(pubidentity) and p['ppid']==intent['identity']['pid'] for p in publisher.get('owned_identities',[])):raise ValueError('actual bound publisher CLOSED ownership required')
    if start['controller']['sid'] not in closed.get('owned_sessions',[]) or pubidentity['sid'] not in publisher['owned_sessions'] or any(p['sid'] not in closed['owned_sessions'] for p in closed.get('owned_identities',[])) or any(p['sid'] not in publisher['owned_sessions'] for p in publisher.get('owned_identities',[])):raise ValueError('actual identities must belong to claimed CLOSED sessions')
    if start['manifest_sha256']!=previous['manifest_sha256'] or closed.get('state')!='CLOSED' or type(closed.get('actual_worker_exit')) is not int or not closed.get('owned_sessions') or not any(_identity(p)==_identity(start['controller']) for p in closed.get('owned_identities',[])) or failed.get('status')!='failed+spent' or failed.get('stages') or publisher.get('state')!='CLOSED' or publisher.get('actual_worker_exit')!=0:raise ValueError('actual failed pre-admission CLOSED controller/publisher proof required')
    sessions=set(closed['owned_sessions'])|set(publisher['owned_sessions'])|{start['controller']['sid'],pubidentity['sid'],intent['identity']['sid']}
    if any(p['sid'] in sessions for p in linux_processes()):raise ValueError('prior controller/publisher session reopened')
    return root/f'attempt-pilot-3-preworker-{generation}.json'


def reserve_queue_attempt(value,auth,manifest_pin):
    verify_queue_authority(value,auth);root=queue_charge_root(value['queue']);attempt=auth['attempt'];path=root/f"attempt-{attempt['slot_id']}-{attempt['number']}.json"
    if 'preworker_repair' in auth:path=preworker_registration_path(value,auth)
    if 'acceptance_recovery' in auth:
        acceptance_recovery_original(value,auth);path=Path(S02_RECOVERY_REGISTRATION)
        if path.parent!=root:raise ValueError('recovery registration outside original charged queue')
    if value['stage'].endswith('-verify'):
        record=_read_json(path)
        if record['manifest_sha256']!=value['predecessor']['predecessor_operation_manifest_sha256'] or record['queue']!=value['queue']:raise ValueError('verifier original attempt identity drift')
        return record
    record={'manifest_sha256':manifest_pin,'operation_id':value['operation_id'],'output_root':value['output_root'],'control_root':value['control_root'],'queue':value['queue'],'attempt':attempt,'issued_at':time.time()}
    if 'preworker_repair' in auth:record['preworker_repair']=auth['preworker_repair']
    if 'acceptance_recovery' in auth:record['acceptance_recovery']=auth['acceptance_recovery']
    if path.exists():
        raise ValueError('duplicate registered attempt: preserve spent, no resubmit')
    write_once(path,record);return record


def check_queue_filesystem(value,auth,descriptor):
    root=Path(value['queue']['receipt_path']).parent;device=root.stat().st_dev
    paths=[value['code_root'],value['output_root'],value['control_root'],value['authorization']['path'],descriptor['feed_root'],auth['origin']['path']]
    for name in paths:
        path=Path(name)
        if root not in path.parents or any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('charged directory outside original queue or symlink')
        parent=path
        while not parent.exists():parent=parent.parent
        if parent.stat().st_dev!=device:raise ValueError('charged artifact on different filesystem')
    return True


def publisher_stop_requested(journal):
    stop=journal.root/'stop.json'
    if not stop.exists():return False
    request=_read_json(stop)
    expected={'schema_id':'p4h-real-publisher-stop-v1','descriptor_sha256':operation._file_sha256(journal.root/'descriptor.json'),'publisher_process':journal.descriptor['publisher_process']}
    if request!=expected:raise ValueError('publisher cooperative stop descriptor/birth drift')
    return True


def validate_predecessor_cross_bindings(value,sealed,terminal,closure,start,admission,inventory):
    pin=sealed['predecessor_operation_manifest_sha256']
    if closure.get('state')!='CLOSED' or closure.get('manifest_sha256')!=pin or terminal.get('manifest_sha256')!=pin or start.get('manifest_sha256')!=pin or admission.get('manifest_sha256')!=pin:raise ValueError('predecessor closure/start/admission manifest mismatch')
    if sha(canonical(start))!=sealed['controller_start_receipt_sha256'] or terminal.get('start_sha256')!=sealed['controller_start_receipt_sha256'] or admission.get('start_sha256')!=sealed['controller_start_receipt_sha256']:raise ValueError('predecessor controller start binding mismatch')
    if sha(canonical(admission))!=sealed['admission_sha256'] or terminal.get('admission_sha256')!=sealed['admission_sha256'] or terminal.get('final_g_receipt_sha256')!=sealed['final_g_receipt_sha256']:raise ValueError('predecessor admission/final G binding mismatch')
    if inventory!=closure.get('owned_identities') or sha(canonical(inventory))!=sealed['worker_birth_inventory_sha256']:raise ValueError('predecessor actual birth inventory mismatch')
    inputs={r['role']:r['sha256'] for r in value['roles'] if r.get('source')=='input_hashes'}
    sealed_inputs={r['role']:r['sha256'] for r in sealed['input_hashes']}
    if inputs!=sealed_inputs or terminal['input_hashes']!=sealed['input_hashes']:raise ValueError('verifier current input roles differ from sealed predecessor')
    return True






def configure_real_torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic workspace drift')
    return {'precision':'fp32','tf32':False,'deterministic':torch.are_deterministic_algorithms_enabled(),'cublas_workspace':os.environ['CUBLAS_WORKSPACE_CONFIG']}






def linux_proc_credentials(pid):
    blob=(Path('/proc')/str(pid)/'status').read_bytes()
    if len(blob)>65536:raise ValueError('proc credential metadata bound')
    rows={line.split(':',1)[0]:line.split(':',1)[1].split() for line in blob.decode().splitlines() if ':' in line}
    uid=[int(v) for v in rows['Uid']];gid=[int(v) for v in rows['Gid']]
    if len(uid)!=4 or len(gid)!=4:raise ValueError('complete kernel credentials required')
    return {'uid':uid,'gid':gid}


class ForeignUIDProbeUnavailable(ValueError):
    def __init__(self,message,credentials):
        super().__init__(message);self.credentials=credentials


class ForeignUIDIdentityDrift(ValueError):
    """An observed identity change is never resolved by an old process exit."""


def pin_foreign_observer_process(row):
    """Pin the observed birth before allowing an exit to resolve visibility."""
    fd=os.pidfd_open(row['pid'],0)
    try:
        try:current=linux_identity(row['pid'])
        except FileNotFoundError:
            if confirm_foreign_observer_exit(row,fd):return fd
            raise
        if _identity(current)!=_identity(row):raise ForeignUIDIdentityDrift('foreign observer birth drift')
    except BaseException:
        os.close(fd)
        raise
    return fd


def foreign_observer_exited(fd):
    import select
    return bool(select.select([fd],[],[],0)[0])


def confirm_foreign_observer_exit(row,fd):
    if not foreign_observer_exited(fd):return False
    try:current=linux_identity(row['pid'])
    except FileNotFoundError:return True
    if _identity(current)!=_identity(row):raise ForeignUIDIdentityDrift('foreign observer birth drift after exit')
    return True


def foreign_uid_fd_modes(row,descriptor):
    """Only kernel-confirmed termination resolves an inaccessible FD set.

    A held pidfd identifies this exact process, including a retained zombie.
    Live failures and PID/birth drift remain fail-closed. This is the same
    current-writer semantics as a disappeared /proc entry in the outer scan.
    """
    try:fd=pin_foreign_observer_process(row)
    except ProcessLookupError:return []
    try:
        if confirm_foreign_observer_exit(row,fd):return []
        try:
            try:return _foreign_uid_fd_modes_live(row,descriptor)
            except ForeignUIDProbeUnavailable as error:
                if confirm_foreign_observer_exit(row,fd):return []
                if _identity(linux_identity(row['pid']))!=_identity(row) or linux_proc_credentials(row['pid'])!=error.credentials:raise ForeignUIDIdentityDrift('foreign UID/birth drift during visibility recheck')
                time.sleep(.05)
                if confirm_foreign_observer_exit(row,fd):return []
                if _identity(linux_identity(row['pid']))!=_identity(row) or linux_proc_credentials(row['pid'])!=error.credentials:raise ForeignUIDIdentityDrift('foreign UID/birth drift before visibility recheck')
                result=_foreign_uid_fd_modes_live(row,descriptor)
                if confirm_foreign_observer_exit(row,fd):return []
                if _identity(linux_identity(row['pid']))!=_identity(row) or linux_proc_credentials(row['pid'])!=error.credentials:raise ForeignUIDIdentityDrift('foreign UID/birth drift after visibility recheck')
                return result
        except ForeignUIDIdentityDrift:raise
        except (ValueError,FileNotFoundError,PermissionError,ProcessLookupError):
            if confirm_foreign_observer_exit(row,fd):return []
            raise
    finally:os.close(fd)


def foreign_probe_target_diagnostic(row):
    """No raw argv is returned; diagnostics never grant FD visibility."""
    value={'observer_caller_pid':os.getpid()}
    try:
        value['identity']=linux_identity(row['pid'])
        value['credentials']=linux_proc_credentials(row['pid'])
        command=(Path('/proc')/str(row['pid'])/'cmdline').read_bytes()
        if len(command)>65536:raise ValueError('kernel command metadata bound')
        value['observer_helper_command']=b'p4h-readonly-uid-fd-v1' in command
    except (OSError,ValueError,KeyError):value['target_metadata_unavailable']=True
    return value


def _foreign_uid_fd_modes_live(row,descriptor):
    """Read inaccessible foreign-UID metadata by dropping child credentials.

    Never omit an inaccessible process: the helper must inspect its complete
    FD set and return the same birth/credentials. Same-UID or mixed credentials,
    any unreadable FD, any writable feed FD, or helper failure remains STOP.
    No capability, mount, permission or service change is made.
    """
    import shutil,subprocess,sys
    before=linux_proc_credentials(row['pid']);uids=before['uid'];gids=before['gid']
    if len(set(uids))!=1 or len(set(gids))!=1 or uids[0]==os.getuid() or uids[0]==descriptor['feed_owner_uid']:raise ValueError('same/mixed UID FD visibility unavailable')
    if _identity(linux_identity(row['pid']))!=_identity(row):raise ForeignUIDIdentityDrift('foreign observer birth drift')
    tool=shutil.which('setpriv')
    if tool is None:raise ValueError('readonly UID observer unavailable')
    executable=str(Path(sys.executable).resolve());python_pin=operation._file_sha256(executable)
    body="""import os,sys,json,pathlib,hashlib
spec=json.loads(sys.argv[1]);p=pathlib.Path('/proc')/str(spec['pid'])
def credentials():
 blob=(p/'status').read_bytes()
 if len(blob)>65536:raise ValueError('credential bound')
 return [int(v) for line in blob.decode().splitlines() if line.startswith('Uid:') for v in line.split()[1:]]
def birth():
 text=(p/'stat').read_text();return int(text[text.rfind(')')+2:].split()[19])
before=credentials();first=birth()
if before!=[os.getuid()]*4 or first!=spec['birth']:raise ValueError('kernel UID/birth mismatch')
entries=list((p/'fd').iterdir())
if len(entries)>4096:raise ValueError('foreign fd bound')
modes=set();unknown=0
for fd in entries:
 try:
  target=os.readlink(fd)
  if target!=spec['feed_root'] and not target.startswith(spec['feed_root']+'/'):continue
  info=(p/'fdinfo'/fd.name).read_text()
  flags=next(int(line.split()[1],8) for line in info.splitlines() if line.startswith('flags:'))&os.O_ACCMODE
  if flags==os.O_WRONLY:modes.add('w')
  elif flags==os.O_RDWR:modes.add('u')
 except FileNotFoundError:continue
 except (PermissionError,StopIteration,ValueError):unknown+=1
if credentials()!=before or birth()!=first:raise ValueError('kernel credential/birth changed')
print(json.dumps({'schema_id':'p4h-readonly-uid-fd-v1','observer_uid':os.getuid(),'pid':spec['pid'],'birth':first,'uid':before,'python_sha256':hashlib.sha256(pathlib.Path(sys.executable).resolve().read_bytes()).hexdigest(),'fd_modes':sorted(modes),'unknown_fds':unknown}))
"""
    args=[tool,'--reuid',str(uids[0]),'--regid',str(gids[0]),'--clear-groups','--inh-caps=-all','--ambient-caps=-all','--no-new-privs',executable,'-I','-S','-B','-c',body,json.dumps({'pid':row['pid'],'birth':row['birth'],'feed_root':descriptor['feed_root']})]
    completed=subprocess.run(args,capture_output=True,timeout=10,check=False)
    if completed.returncode!=0 or len(completed.stdout)>65536:
        diagnostic={'pid':row['pid'],'birth':row['birth'],'returncode':completed.returncode,'stdout_bytes':len(completed.stdout),'stderr_sha256':hashlib.sha256(completed.stderr).hexdigest(),'target':foreign_probe_target_diagnostic(row)}
        message='readonly UID observer failed; zero visibility bypass; '+json.dumps(diagnostic,sort_keys=True)
        stderr=completed.stderr
        if stderr.endswith((b'ValueError: kernel UID/birth mismatch\n',b'ValueError: kernel credential/birth changed\n')):raise ForeignUIDIdentityDrift(message)
        permission=("PermissionError: [Errno 13] Permission denied: '/proc/"+str(row['pid'])+"/fd'\n").encode()
        if completed.returncode==1 and len(stderr)<=65536 and stderr.endswith(permission):raise ForeignUIDProbeUnavailable(message,before)
        raise ValueError(message)
    value=json.loads(completed.stdout)
    if isinstance(value,dict) and ('birth' in value and value['birth']!=row['birth'] or 'pid' in value and value['pid']!=row['pid'] or 'uid' in value and value['uid']!=uids):raise ForeignUIDIdentityDrift('foreign helper UID/birth drift')
    fields={'schema_id','observer_uid','pid','birth','uid','python_sha256','fd_modes','unknown_fds'}
    if not isinstance(value,dict) or set(value)!=fields or value['schema_id']!='p4h-readonly-uid-fd-v1' or value['observer_uid']!=uids[0] or value['pid']!=row['pid'] or value['birth']!=row['birth'] or value['uid']!=uids or value['python_sha256']!=python_pin or value['unknown_fds']!=0 or value['fd_modes']!=[]:raise ValueError('foreign UID complete inventory binding/writer failure')
    if linux_proc_credentials(row['pid'])!=before or _identity(linux_identity(row['pid']))!=_identity(row):raise ForeignUIDIdentityDrift('foreign UID/birth changed after inspection')
    return []


if __name__=='__main__':main()
