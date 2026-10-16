"""Finite, queue-only SID-COLL pipeline controller; no model imports.

CLI: ACTION --plan ABS --plan-sha SHA --out NEW_JSON.
ACTION is dry/status/tick/start/driver. Linux terminal notifications advance
stages immediately; fallback state polling stays at 900 seconds. Unavailable
inotify is reported as sleep_fallback in every driver summary. dry calls launcher.prepare only; tick may
submit ready own jobs. start requires a fresh control_dir and detaches driver.

Frozen plan schema vkgr-sid-coll-plan-v1: root, control_dir, created_epoch,
deadline_epoch (=created_epoch+172800), common_refs [{path,sha256}], branches
{C1p,C2: {stages: {NAME: {job: complete vkgr-queue-job-v1 spec,
proof: {path: own run/out/receipt.json, expected: {dotted.key: literal}}}}}}.
All STAGES must exist. A whole-string @stage.dotted.key resolves only an
already accepted receipt in this branch. No expressions or shell interpolation.
Expected status must be explicit. Verify stages require status=verified.
Proof hashes/config/data identity are explicit dotted expected values chosen in
the frozen plan. Raw evidence is supplied by the job-entry receipt producer.
verify42 exposes summary.hits/R50/N50/best_epoch/stop_epoch; only then is screening
applied. The controller never reconstructs a verifier or reads checkpoints.

Each metadata read / write is bounded below 48 KiB. Submission intent is written
once before scheduler.submit, so a crash with uncertain submission stops instead
of retrying. Existing failed, rejected, or unbound jobs are never adopted.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
BASE=('input_preflight','input','input_verify','sid_fit','sid_verify','preflight','cuda','dry','train42','verify42')
STAGES=BASE+('train2024','verify2024','train2025','verify2025')
LIMIT=48*1024
POLLS=900


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()
def read(path,pin=None):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('absolute nonsymlink metadata path required')
    if p.suffix not in ('.json','.reason') or p.stat().st_size>=LIMIT:raise ValueError('metadata must be <48KiB JSON/reason')
    payload=p.read_bytes()
    if len(payload)>=LIMIT:raise ValueError('metadata grew beyond bound')
    sha=hashlib.sha256(payload).hexdigest()
    if pin is not None and sha!=pin:raise ValueError('metadata SHA mismatch: '+str(p))
    return json.loads(payload),sha

def new(path,value):
    payload=encoded(value)
    if len(payload)>=LIMIT:raise ValueError('output metadata must be <48KiB')
    p=Path(path)
    if any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('symlink output prohibited')
    with p.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())

def write_summary(path,value):
    """Batch reports are not launcher records. Keep their independent finite bound."""
    payload=encoded(value)
    if len(payload)>=2*1024*1024:raise ValueError('batch summary exceeds 2MiB')
    p=Path(path)
    if any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('symlink output prohibited')
    with p.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())

def diagnostic_summary(proof):
    result={k:{'status':'UNKNOWN','reason':'diagnostic artifact absent'} for k in ('A','B','C','D','Dprime')}
    result['sources']={}
    for name in ('mapping_diagnostics','prediction_diagnostics'):
        ref=proof.get('proofs',{}).get(name)
        if ref is None:continue
        value,_=read(ref['path'],ref['sha256'])
        result['sources'][name]=copy.deepcopy(ref)
        for key in ('A','B','C','D','Dprime','C_delta_vs_T3'):
            if key in value:result[key]=value[key]
    return result

def complete_snapshot(result,plan):
    result['unaccepted']=[]
    for label,branch in plan['branches'].items():
        state=result['branches'][label];stages=state.setdefault('stages',{})
        for name,stage in branch['stages'].items():
            if name not in stages:
                screened=state.get('status')=='complete_low' and name not in BASE
                stages[name]={'run_id':stage['job']['run_id'],'status':'SKIPPED_SCREENED_OUT' if screened else 'UNKNOWN',
                              'unaccepted':not screened,'reason':'preregistered screening' if screened else 'not yet accepted'}
            if stages[name].get('unaccepted',True):
                result['unaccepted'].append({'branch':label,'run_id':stages[name]['run_id'],'status':stages[name]['status']})
    return result

def at(value,key):
    for part in key.split('.'):
        if not isinstance(value,dict) or part not in value:raise ValueError('missing receipt field: '+key)
        value=value[part]
    return value

def materialize(value,accepted):
    if isinstance(value,dict):return {k:materialize(v,accepted) for k,v in value.items()}
    if isinstance(value,list):return [materialize(v,accepted) for v in value]
    if isinstance(value,str) and value.startswith('@'):
        stage,sep,key=value[1:].partition('.')
        if not sep or stage not in accepted:raise ValueError('unaccepted stage reference: '+value)
        return copy.deepcopy(at(accepted[stage],key))
    return value


def validate_plan(plan):
    if plan.get('schema')!='vkgr-sid-coll-plan-v1':raise ValueError('plan schema')
    if set(plan.get('branches',{}))!={'C1p','C2'}:raise ValueError('exactly C1p and C2 required')
    for key in ('created_epoch','deadline_epoch'):
        if type(plan.get(key)) not in (int,float) or not math.isfinite(plan[key]):raise ValueError('finite timestamps required')
    if plan['deadline_epoch']-plan['created_epoch']!=48*3600:raise ValueError('fixed 48h deadline required')
    for key in ('root','control_dir'):
        p=Path(plan[key])
        if not p.is_absolute() or '..' in p.parts or any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('absolute nonsymlink controller paths required')
    ids=[]
    for branch in plan['branches'].values():
        if set(branch['stages'])!=set(STAGES):raise ValueError('exact frozen stages required')
        for name,stage in branch['stages'].items():
            job=stage['job'];rid=job['run_id'];ids.append(rid)
            if job.get('schema')!='vkgr-queue-job-v1' or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,95}',rid):raise ValueError('queue schema/run ID')
            if len(encoded(job))>=LIMIT:raise ValueError('job metadata too large')
            p=Path(stage['proof']['path'])
            if p.parent!=Path(plan['root'])/'runs'/rid/'out' or p.suffix!='.json':raise ValueError('proof must be own run/out JSON')
            expected=stage['proof'].get('expected',{})
            if 'status' not in expected or (('verify' in name) and expected['status']!='verified'):raise ValueError('explicit raw status required; verify must be verified')
    if len(ids)!=len(set(ids)):raise ValueError('run IDs must be globally unique')
    return plan


def queue_state(root,rid):
    result={k:sorted((Path(root)/'queue'/k).glob('*-'+rid+'.json')) for k in ('pending','claimed','rejected')}
    return result


def default_submit(root,job):
    from vkgr_ops import scheduler
    default_prepare(job)
    return scheduler.submit(Path(root),job)


def default_prepare(job):
    from vkgr_ops import launcher
    record=launcher.prepare(job)
    if len(encoded({'job':job,'prepared':record}))>=LIMIT-2048:raise ValueError('supervisor launch record plus job exceeds safe <48KiB envelope')
    return {'launch_record_bytes':len(encoded(record)),'supervisor_prepare':'passed'}


def publish(plan,job,submitter=None):
    root=Path(plan['root']);ctl=Path(plan['control_dir']);ctl.mkdir(parents=True,exist_ok=True)
    allowed={s['job']['run_id'] for b in plan['branches'].values() for s in b['stages'].values()}
    rid=job['run_id']
    if rid not in allowed:raise ValueError('outside own frozen run IDs')
    with (ctl/'publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (root/'runs'/rid).exists() or any(queue_state(root,rid).values()) or (ctl/('intent-'+rid+'.json')).exists():raise FileExistsError('never resubmit '+rid)
        new(ctl/('intent-'+rid+'.json'),{'run_id':rid,'job_sha256':digest(job),'created_utc':utc()})
        path=(submitter or default_submit)(root,job)
        return str(path)


def inspect_stage(plan,stage,accepted):
    job=materialize(stage['job'],accepted);rid=job['run_id'];root=Path(plan['root']);q=queue_state(root,rid)
    result={'run_id':rid,'job_sha256':digest(job),'queue':{k:[p.name for p in v] for k,v in q.items()},'status':'absent','accepted':False,'unaccepted':True}
    if q['rejected']:
        result.update(status='ALERT',reason='rejected; never resubmit',reasons=[])
        for p in q['rejected']:
            for rp in (p.with_suffix('.reason'),Path(str(p)+'.reason')):
                if rp.exists():
                    if rp.stat().st_size>=LIMIT:raise ValueError('oversized rejected reason')
                    result['reasons'].append(rp.read_text()[:2000])
        return result,None
    if len(q['pending'])+len(q['claimed'])>1:raise ValueError('duplicate own queue entries')
    for p in q['pending']+q['claimed']:
        queued,pin=read(p)
        if queued!=job:raise ValueError('queue job differs from frozen plan')
        result['queue_job_sha256']=pin
    rd=root/'runs'/rid;terminal=rd/'terminal.json'
    if terminal.exists():
        value,pin=read(terminal);result.update(terminal_sha256=pin,terminal_status=value.get('status'),exit_code=value.get('exit_code'))
        if len(q['claimed'])!=1:raise ValueError('terminal lacks unique own claimed job')
        if value.get('run_id',rid)!=rid or value.get('status')!='succeeded' or value.get('exit_code')!=0:
            result.update(status='ALERT',reason='job failed');return result,None
        proof,pin=read(stage['proof']['path']);expected=materialize(stage['proof']['expected'],accepted)
        if any(at(proof,k)!=v for k,v in expected.items()):raise ValueError('raw proof / frozen identity mismatch')
        result.update(status='accepted',accepted=True,unaccepted=False,proof_path=stage['proof']['path'],proof_sha256=pin)
        summary=proof.get('summary',proof)
        result['summary']={k:summary.get(k) for k in ('commit','config_sha256','data_sha256','manifest_sha256','prediction_sha256','prediction_sha256_kind','prediction_computation_sha256','prediction_manifest_file_sha256','best_epoch','stop_epoch','R50','N50','hits','A','B','C','D','Dprime') if k in summary}
        if 'binding' in proof:result['binding']=proof['binding']
        result['artifacts']=copy.deepcopy(proof.get('proofs',{}))
        if any(n in proof.get('proofs',{}) for n in ('mapping_diagnostics','prediction_diagnostics')):
            result['diagnostics']=diagnostic_summary(proof)
        accepted_proof={**proof,'receipt_sha256':pin}
        return result,accepted_proof
    if q['pending'] or q['claimed']:result['status']='queued';return result,None
    if rd.exists():raise ValueError('run directory without queue identity')
    if (Path(plan['control_dir'])/('intent-'+rid+'.json')).exists():raise ValueError('submission intent without queue; uncertain attempt never retried')
    return result,None


def tick(plan,now=None,submitter=None,submit=True):
    validate_plan(plan);now=time.time() if now is None else now
    result={'schema':'vkgr-sid-coll-summary-v1','read_at_utc':utc(),'now_epoch':now,'deadline_epoch':plan['deadline_epoch'],'plan_digest':digest(plan),'branches':{},'status':'running','new_submissions':[]}
    common_error=None
    common_alert=Path(plan['control_dir'])/'alert-common.json'
    if common_alert.exists():common_error=read(common_alert)[0]['reason']
    if now>=plan['deadline_epoch']:common_error='48h deadline expired'
    try:
        for ref in plan.get('common_refs',[]):read(ref['path'],ref['sha256'])
    except Exception as exc:common_error='common input invalid: '+str(exc)
    if common_error:
        result.update(status='ALERT',reason=common_error)
        if submit and not common_alert.exists():
            common_alert.parent.mkdir(parents=True,exist_ok=True)
            new(common_alert,{'reason':common_error,'at_utc':utc()})
        result['branches']={b:{'status':'ALERT','reason':common_error,'stages':{name:{'run_id':stage['job']['run_id'],'status':'UNKNOWN','unaccepted':True} for name,stage in branch['stages'].items()},'unaccepted':True} for b,branch in plan['branches'].items()};return complete_snapshot(result,plan)
    for label,branch in plan['branches'].items():
        accepted={};state={'status':'running','stages':{},'screening':'PENDING','unaccepted':True};result['branches'][label]=state
        alert_path=Path(plan['control_dir'])/('alert-'+label+'.json')
        if alert_path.exists():
            prior,_=read(alert_path)
            state.update(status='ALERT',reason='persisted stop: '+prior['reason'])
            continue
        def stop(reason):
            state.update(status='ALERT',reason=reason)
            if submit:
                alert_path.parent.mkdir(parents=True,exist_ok=True)
                if not alert_path.exists():new(alert_path,{'branch':label,'reason':reason,'at_utc':utc()})
        def advance(name):
            stage=branch['stages'][name]
            try:
                snap,proof=inspect_stage(plan,stage,accepted);state['stages'][name]=snap
                if snap['status']=='ALERT':stop(snap.get('reason','rejected'));return False
                if proof is not None:accepted[name]=proof;return True
                if snap['status']=='absent' and submit:
                    job=materialize(stage['job'],accepted);path=publish(plan,job,submitter)
                    snap.update(status='submitted',submission_path=path);result['new_submissions'].append(job['run_id'])
                return False
            except Exception as exc:
                state['stages'][name]={'run_id':stage['job']['run_id'],'status':'ALERT','accepted':False,'unaccepted':True,'exception':type(exc).__name__+': '+str(exc)}
                stop('stage '+name+' failed');return False
        base_ok=True
        for name in BASE:
            if not advance(name):base_ok=False;break
        if not base_ok:continue
        summary=accepted['verify42'].get('summary',accepted['verify42']);hits=summary.get('hits')
        if type(hits) is not int or not 0<=hits<=2520:stop('verified42 integer hits missing');continue
        if hits<260:state.update(status='complete_low',screening='NO_SUPPLEMENT',unaccepted=False);continue
        state['screening']='SUPPLEMENT';completed=[]
        for seed in (2024,2025):
            ok=advance('train'+str(seed))
            if state['status']=='ALERT':break
            if ok:ok=advance('verify'+str(seed))
            completed.append(ok)
            if state['status']=='ALERT':break
        if len(completed)==2 and all(completed):state.update(status='complete',unaccepted=False)
    if any(s['status']=='ALERT' for s in result['branches'].values()):result['status']='ALERT'
    elif all(s['status'] in ('complete','complete_low') for s in result['branches'].values()):result['status']='complete'
    return complete_snapshot(result,plan)


def load_plan(path,pin):
    p=Path(path)
    if not p.is_absolute() or p.suffix!='.json' or p.stat().st_size>2*1024*1024 or any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('bounded absolute frozen JSON plan required')
    payload=p.read_bytes()
    if not re.fullmatch('[0-9a-f]{64}',pin) or hashlib.sha256(payload).hexdigest()!=pin:raise ValueError('frozen plan SHA mismatch')
    return validate_plan(json.loads(payload))


# inotify is a notification channel, not a faster state poll. Only a terminal
# event under one of the frozen own run IDs wakes tick; other runs are ignored.
IN_CLOSE_WRITE=0x00000008
IN_MOVED_TO=0x00000080
IN_CREATE=0x00000100
IN_Q_OVERFLOW=0x00004000
IN_IGNORED=0x00008000

class LinuxInotify:
    def __init__(self):
        if not sys.platform.startswith('linux'):raise OSError('Linux inotify unavailable')
        import ctypes
        self.ctypes=ctypes;self.libc=ctypes.CDLL(None,use_errno=True)
        self.libc.inotify_init1.argtypes=[ctypes.c_int];self.libc.inotify_init1.restype=ctypes.c_int
        self.libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32];self.libc.inotify_add_watch.restype=ctypes.c_int
        self.fd=self.libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);self.paths={}
        if self.fd<0:raise OSError(ctypes.get_errno(),'inotify_init1 failed')
    def watch(self,path):
        path=Path(path)
        if path in self.paths.values():return
        wd=self.libc.inotify_add_watch(self.fd,os.fsencode(path),IN_CREATE|IN_MOVED_TO|IN_CLOSE_WRITE)
        if wd<0:raise OSError(self.ctypes.get_errno(),'inotify_add_watch failed')
        self.paths[wd]=path
    def read(self,timeout):
        import select
        import struct
        if not select.select([self.fd],[],[],max(0,timeout))[0]:return []
        try:payload=os.read(self.fd,65536)
        except BlockingIOError:return []
        events=[];offset=0
        while offset<len(payload):
            if len(payload)-offset<16:raise OSError('truncated inotify header')
            wd,mask,cookie,length=struct.unpack_from('iIII',payload,offset);offset+=16
            if offset+length>len(payload):raise OSError('truncated inotify event')
            name=os.fsdecode(payload[offset:offset+length].split(b'\0',1)[0]);offset+=length
            if mask&(IN_Q_OVERFLOW|IN_IGNORED):raise OSError('inotify overflow or invalid watch')
            if wd in self.paths:events.append((self.paths[wd],name,mask))
        return events
    def close(self):
        if self.fd>=0:os.close(self.fd);self.fd=-1

class EventWaiter:
    def __init__(self,plan,backend_factory=LinuxInotify,clock=time.monotonic,sleeper=time.sleep,wallclock=time.time):
        self.runs=Path(plan['root'])/'runs'
        self.owned={stage['job']['run_id'] for branch in plan['branches'].values() for stage in branch['stages'].values()}
        self.clock=clock;self.sleeper=sleeper;self.wallclock=wallclock;self.backend=None
        self.mode='linux_inotify';self.reason=None;self.watched=set()
        try:
            self.backend=backend_factory();self.backend.watch(self.runs)
            self.refresh()
        except Exception as exc:self.fallback(exc)
    def fallback(self,exc):
        if self.backend is not None:
            try:self.backend.close()
            except OSError:pass
        self.backend=None;self.mode='sleep_fallback';self.reason=type(exc).__name__+': '+str(exc)
    def refresh(self):
        if self.backend is None:return
        for rid in sorted(self.owned):
            path=self.runs/rid
            if path.is_dir() and path not in self.watched:
                self.backend.watch(path);self.watched.add(path)
    def metadata(self):return {'mode':self.mode,'fallback_seconds':POLLS,'reason':self.reason}
    def wait(self,timeout_seconds=POLLS,deadline_epoch=None):
        if timeout_seconds<POLLS:raise ValueError('fallback interval must be >=900 seconds')
        duration=float(timeout_seconds)
        if deadline_epoch is not None:duration=min(duration,max(0,deadline_epoch-self.wallclock()))
        end=self.clock()+duration
        while True:
            remaining=max(0,end-self.clock())
            if remaining<=0:return {**self.metadata(),'wake':'fallback_timeout'}
            if self.backend is None:
                self.sleeper(remaining)
                return {**self.metadata(),'wake':'fallback_timeout'}
            try:
                events=self.backend.read(remaining)
                for parent,name,mask in events:
                    parent=Path(parent)
                    if parent==self.runs and name in self.owned and mask&(IN_CREATE|IN_MOVED_TO):
                        path=self.runs/name
                        if path.is_dir() and path not in self.watched:
                            self.backend.watch(path);self.watched.add(path)
                            # Writer may finish before directory watch registration.
                            if (path/'terminal.json').is_file():return {**self.metadata(),'wake':'own_terminal','run_id':name}
                    elif parent.parent==self.runs and parent.name in self.owned and parent in self.watched and name=='terminal.json' and mask&(IN_CREATE|IN_CLOSE_WRITE|IN_MOVED_TO):
                        return {**self.metadata(),'wake':'own_terminal','run_id':parent.name}
                # Empty batch is a blocking timeout, never a request for tick.
            except Exception as exc:self.fallback(exc)
    def close(self):
        if self.backend is not None:self.backend.close();self.backend=None


def driver(plan_path,pin,out,pid_record='driver-pid.json'):
    if not re.fullmatch(r'driver-pid(?:-v[0-9]+)?\.json',pid_record):raise ValueError('exclusive pid record basename required')
    plan=load_plan(plan_path,pin);ctl=Path(plan['control_dir'])
    if os.getppid()!=1:raise RuntimeError('detached driver requires PPID 1')
    ctl.mkdir(parents=True,exist_ok=True)
    with (ctl/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        new(ctl/pid_record,{'pid':os.getpid(),'ppid':os.getppid(),'poll_seconds':POLLS,'deadline_epoch':plan['deadline_epoch'],'started_utc':utc()})
        waiter=EventWaiter(plan);wake={'wake':'initial',**waiter.metadata()}
        while True:
            # Detect changed frozen plan before every action; deadline cannot reset.
            current=load_plan(plan_path,pin)
            result=tick(current);result['event_wait']=wake;write_summary(ctl/('summary-'+str(time.time_ns())+'.json'),result)
            terminal=all(s['status'] in ('complete','complete_low','ALERT') for s in result['branches'].values())
            if terminal or time.time()>=plan['deadline_epoch']:write_summary(out,result);waiter.close();return result
            wake=waiter.wait(POLLS,deadline_epoch=plan['deadline_epoch'])


def dry(plan):
    validate_plan(plan);prepared={}
    # Pending dynamic identities are intentionally not guessed in dry mode.
    for label,branch in plan['branches'].items():
        prepared[label]={}
        for name,stage in branch['stages'].items():
            job=stage['job'];rid=job['run_id']
            if (Path(plan['root'])/'runs'/rid).exists() or any(queue_state(plan['root'],rid).values()):raise FileExistsError('dry requires unused run ID '+rid)
            if '@' in json.dumps(job):prepared[label][name]={'status':'DEFERRED_DYNAMIC_BINDING','job_bytes':len(encoded(job))}
            else:prepared[label][name]=default_prepare(job)
    return {'status':'dry_ok','prepared':prepared,'plan_digest':digest(plan)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('dry','status','tick','start','driver'));p.add_argument('--plan',required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--out',required=True)
    p.add_argument('--pid-record',default='driver-pid.json')
    a=p.parse_args();plan=load_plan(a.plan,a.plan_sha);out=Path(a.out)
    if out.exists():raise FileExistsError('exclusive output required')
    if a.action=='driver':driver(a.plan,a.plan_sha,out,a.pid_record);return
    if a.action=='start':
        ctl=Path(plan['control_dir'])
        if ctl.exists():raise FileExistsError('new control_dir required for driver start')
        ctl.mkdir(parents=True)
        log=(ctl/'launcher.log').open('xb')
        # Tiny launcher exits immediately; child waits until reparented before driver.
        command=[sys.executable,'-B',__file__,'driver','--plan',a.plan,'--plan-sha',a.plan_sha,'--out',str(ctl/'summary-final.json')]
        # Driver handoff uses a separate process that waits for its launcher to exit.
        handoff="import os,sys,time; end=time.time()+10;\nwhile os.getppid()!=1 and time.time()<end: time.sleep(.02)\nos.execv(sys.executable,[sys.executable,*sys.argv[1:]])"
        child=subprocess.Popen([sys.executable,'-c',handoff,*command[1:]],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        result={'status':'started_waiting_for_reparent','pid':child.pid,'control_dir':str(ctl),'deadline_epoch':plan['deadline_epoch']}
    elif a.action=='dry':result=dry(plan)
    else:result=tick(plan,submit=a.action=='tick')
    write_summary(out,result);print(json.dumps({'status':result['status'],'output':str(out)}))
if __name__=='__main__':main()
