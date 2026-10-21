"""Finite frozen Popularity PF -> validation evaluate -> verify transport only.

No training, retry, model call or scheduler tick. Resource declarations are
supplied by the owner's fixed request, never inferred by this driver.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[3]
CONSUMER_SOURCE='18f98f56fc7211d549eaf296015f256d81691c5b'
ENTRY_SHA='1814f25f1bf08333e5c7a87ea301a619bfd6b6e5d80a3e676078ebef486c6a16'
OPS_SOURCE='5840e6b686ec9651314a9e28a1a2c0befcc05158'
STAGES={'preflight':'preflight','evaluate':'train','verify':'verify'}
POLL_SECONDS=900

# Acceptance remains in the independently reviewed original consumer. Context
# authenticates the actual producer DAG and binds current source/runtime.
CHECK='''import json,sys
from pathlib import Path
from analysis import p_full_popularity as p
inputs,request,pin,stage=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4]
v=p.transport(inputs,request,pin)
_,_,_,_,_,binding,guard=p.context(v,inputs)
for prior in ('preflight','evaluate','verify'):
 receipt=p.accepted(v,inputs,request,pin,prior,binding)
 if prior=='verify' and receipt.get('scientific_verified')is not True:raise ValueError('original scientific verify required')
 if prior==stage:break
guard();p.backend.close_context(v,binding)
if p.backend.encoded(p.transport(inputs,request,pin))!=p.backend.encoded(v):raise ValueError('request changed during acceptance')
print(json.dumps({'verified':True,'stage':stage,'training':False,'test_read':False}))
'''

PREPARE='''import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[2])/'ops'))
from vkgr_ops import launcher
plan=json.loads(Path(sys.argv[1]).read_bytes())
ns=launcher.build_parser().parse_args(plan['launcher_argv'])
spec=launcher.spec_from_args(ns)
metadata=launcher.prepare(spec)
run=launcher.resolve_root(ns.root)/'runs'/spec['run_id']
env=launcher._recorded_env(metadata['env'],launcher._job_env(metadata['env'],run,spec['run_id']))
actual={'schema':'vkgr-run-v1','run_id':spec['run_id'],'kind':spec['kind'],'argv':spec['argv'],'seed':spec['seed'],'stop_mode':spec['stop_mode'],'ops_source':launcher._ops_source(),'resources':{**{k:spec[k] for k in launcher.RESOURCE_ARGS},'wall_seconds':spec['wall_hours']*3600.0},**metadata,'env':env}
def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
if encoded(actual)!=encoded(plan['expected_job']):raise ValueError('complete original launcher dry job differs')
if spec.get('no_admission') or spec.get('pinned_manifest') or spec.get('pinned_files'):raise ValueError('unregistered bypass/overlay')
print(json.dumps({'verified':True,'submitted':False,'spec':spec,'metadata':metadata}))
'''


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def write_new(path,v):
    with Path(path).open('x') as f:f.write(canonical(v)+'\n')
def deadline_seconds(hours):
    if isinstance(hours,bool) or not isinstance(hours,(int,float)) or not math.isfinite(hours) or not 0<hours<=200:raise ValueError('finite deadline within 200 hours required')
    return hours*3600

def remaining(a):
    seconds=a.deadline-a.now()
    if seconds<=0:raise TimeoutError('finite Popularity chain deadline')
    return seconds

def sha(path,a=None):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            if a:remaining(a)
            h.update(block)
    return h.hexdigest()

def secure(path):
    path=Path(path)
    if not path.is_absolute() or path!=path.resolve() or any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('canonical absolute non-symlink path required')
    return path

def exact_tree(a,root,commit):
    secure(root)
    def git(*argv):
        result=subprocess.run(['git','-C',str(root),*argv],capture_output=True,check=True,timeout=remaining(a));remaining(a);return result.stdout
    if git('rev-parse','HEAD').decode().strip()!=commit:raise ValueError('fixed source commit differs')
    raw=git('archive',commit);expected={}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            remaining(a)
            if member.isdir():continue
            if not member.isfile():raise ValueError('source regular files required')
            expected[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    actual={}
    def fail(error):raise error
    for directory,dirs,files in os.walk(root,followlinks=False,onerror=fail):
        remaining(a)
        if Path(directory)==root and '.git' in dirs:dirs.remove('.git')
        for name in dirs:
            if (Path(directory)/name).is_symlink():raise ValueError('source symlink directory refused')
        for name in files:
            remaining(a);path=Path(directory)/name
            if path==root/'.git':continue
            if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('source regular file required')
            actual[path.relative_to(root).as_posix()]=sha(path,a)
    if actual!=expected:raise ValueError('exact source file set/bytes differ')
    remaining(a)

def fixed_request(a):
    exact_tree(a,ROOT,a.driver_source)
    if sha(__file__,a)!=a.driver_sha256:raise ValueError('fixed driver pin differs')
    exact_tree(a,a.consumer_code,CONSUMER_SOURCE);exact_tree(a,a.ops_code,OPS_SOURCE)
    if sha(a.consumer_code/'analysis/p_full_popularity.py',a)!=ENTRY_SHA:raise ValueError('fixed Popularity entry differs')
    if sha(a.request,a)!=a.request_sha256 or sha(a.inputs,a)!=a.inputs_sha256:raise ValueError('fixed request/input pin differs')
    v=json.loads(a.request.read_bytes())
    if v.get('source',{}).get('commit')!=CONSUMER_SOURCE or v.get('predecessor') is not None:raise ValueError('original Popularity null predecessor request required')
    ids=v.get('run_ids',{})
    if set(ids)!=set(STAGES.values()) or len(set(ids.values()))!=3 or any(not isinstance(r,str) or not r.startswith('p-full-popularity-') or '/' in r for r in ids.values()):raise ValueError('three exclusive Popularity run IDs required')
    remaining(a);return v

def evidence_path(a,v):
    root=secure(a.evidence);protected=[ROOT,a.consumer_code,a.ops_code,a.inputs,a.request]
    def visit(value):
        if isinstance(value,dict):
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,str) and value.startswith('/'):protected.append(Path(value).resolve())
    visit(v)
    if any(root==p or p in root.parents or root in p.parents for p in protected):raise ValueError('exclusive evidence outside source/input/output/OPS required')

def command(a,argv,name,decode=True):
    r=subprocess.run(argv,cwd=a.consumer_code,env=a.env,capture_output=True,timeout=remaining(a))
    for suffix,data in (('stdout',r.stdout),('stderr',r.stderr)):
        with (a.evidence/(name+'.'+suffix)).open('xb') as f:f.write(data)
    remaining(a)
    if r.returncode:raise ValueError('original command failed: '+name)
    return json.loads(r.stdout) if decode else r.stdout

def queue_state(root,rid):
    hits=[(s,p) for s in ('pending','claimed','rejected') for p in (Path(root)/'queue'/s).glob('*-'+rid+'.json')]
    if len(hits)>1:raise ValueError('duplicate original queue ID')
    if not hits:return {'state':'absent','path':None}
    state,path=hits[0]
    if state=='rejected':
        reason=path.with_suffix('.reason')
        raise ValueError('rejected; no retry: '+(reason.read_text() if reason.exists() else 'reason unavailable'))
    return {'state':state,'path':path}

def terminal(run,rid):
    path=run/'terminal.json'
    if not path.exists():return False
    v=json.loads(path.read_bytes())
    if (v.get('schema')!='vkgr-terminal-v1' or v.get('run_id')!=rid or v.get('status')!='succeeded' or type(v.get('exit_code'))is not int or v['exit_code']!=0 or v.get('signal')is not None or v.get('wall_cap_exceeded')is not False):raise ValueError('original terminal failed; no retry')
    return True

def compare_job(actual,expected):
    if type(actual)is not dict or set(actual)-set(expected)-{'created_at','admission'} or any(canonical(actual.get(k))!=canonical(v) for k,v in expected.items()):raise ValueError('complete original job differs')

def run(a,*,now=time.monotonic,sleep=time.sleep):
    if os.getppid()!=1:raise ValueError('detached PPID=1 required')
    a.now=now;a.deadline=now()+deadline_seconds(a.deadline_hours)
    for path in (a.inputs,a.request,a.consumer_code,a.ops_code,a.evidence):secure(path)
    v=fixed_request(a);evidence_path(a,v);a.evidence.mkdir(parents=True,exist_ok=False)
    submitted=[]
    try:
        ops=secure(v['ops_root']);output=secure(v['output_root'])
        for stage,key in STAGES.items():
            # Preserve the actual frozen stage environment for launcher metadata.
            a.env={**os.environ,**v['job_contract'][key]['env']}
            a.env['PYTHONPATH']=str(a.consumer_code)+':'+str(a.consumer_code/'src')+':'+str(a.ops_code/'ops')
            a.env['PYTHONDONTWRITEBYTECODE']='1'
            for k in tuple(a.env):
                if k.startswith('VKGR_') or k=='CUDA_VISIBLE_DEVICES':a.env.pop(k,None)
            # Plan/accepted use consumer import path; prepare uses pinned OPS.
            base=[sys.executable,'-B','-m','analysis.p_full_popularity']
            tail=['--inputs',str(a.inputs),'--request',str(a.request),'--request-sha256',a.request_sha256]
            fixed_request(a)
            plan=command(a,base+['plan','--stage',stage]+tail,stage+'-plan')
            if plan.get('stage')!=stage or plan.get('training') is not False or plan.get('status')!='planned_dry':raise ValueError('original Popularity plan required')
            expected=plan['expected_job'];rid=v['run_ids'][key];runroot=ops/'runs'/rid
            if expected['run_id']!=rid:raise ValueError('stage/run ID mapping differs')
            # Use exactly the launcher-declared environment (not the import path).
            a.env['PYTHONPATH']=v['job_contract'][key]['env']['PYTHONPATH']
            prepared=command(a,[sys.executable,'-B','-c',PREPARE,str(a.evidence/(stage+'-plan.stdout')),str(a.ops_code)],stage+'-dry')
            if prepared.get('verified')is not True or prepared.get('submitted')is not False:raise ValueError('original launcher dry required')
            spec=prepared['spec'];seen=False;sent=False
            while True:
                remaining(a)
                if canonical(fixed_request(a))!=canonical(v):raise ValueError('fixed original request changed')
                state=queue_state(ops,rid)
                if state['state']=='absent':
                    if seen or sent or runroot.exists():raise ValueError('original job disappeared; no retry')
                    # A new chain cannot reuse an existing phase output.
                    if (output/('popularity-'+stage+'.json')).exists():raise ValueError('orphan original receipt; no resubmit')
                    if stage=='evaluate' and any((output/n).exists() for n in ('popularity-predictions.json','popularity-metrics.json')):raise ValueError('exclusive evaluation output required')
                    fixed_request(a)
                    fresh=queue_state(ops,rid)
                    if fresh['state']!='absent' or runroot.exists():
                        seen=True
                        continue
                    # Original scheduler accepts exactly original launch argv.
                    command(a,[sys.executable,'-B',str(a.ops_code/'ops/bin/vkgr-sched'),'submit',*plan['launcher_argv']],stage+'-submit',decode=False)
                    sent=True;submitted.append(rid);state=queue_state(ops,rid)
                    if state['state'] not in ('pending','claimed'):raise ValueError('submission absent from original queue')
                seen=True
                queued=json.loads(state['path'].read_bytes());queued.pop('schema',None)
                if canonical(queued)!=canonical(spec):raise ValueError('original queue spec differs')
                if (runroot/'run.json').exists():compare_job(json.loads((runroot/'run.json').read_bytes()),expected)
                with (a.evidence/'progress.jsonl').open('a') as f:f.write(canonical({'stage':stage,'run_id':rid,'queue':state['state'],'claimed':state['state']=='claimed'})+'\n')
                if terminal(runroot,rid):
                    if state['state']!='claimed':raise ValueError('terminal lacks claimed confirmation')
                    a.env['PYTHONPATH']=str(a.consumer_code)+':'+str(a.consumer_code/'src')+':'+str(a.ops_code/'ops')
                    result=command(a,[sys.executable,'-B','-c',CHECK,str(a.inputs),str(a.request),a.request_sha256,stage],stage+'-accepted')
                    if result!={'verified':True,'stage':stage,'training':False,'test_read':False}:raise ValueError('original acceptance required')
                    fixed_request(a);remaining(a);break
                sleep(min(POLL_SECONDS,remaining(a)))
        fixed_request(a);remaining(a)
        result={'status':'verified','stages':list(STAGES),'submitted_run_ids':submitted,'training':False,'test_read':False,'scientific_interpretation':False}
    except Exception as exc:
        write_new(a.evidence/'summary.json',{'status':'ALERT','error_type':type(exc).__name__,'reason':str(exc),'retry':False});raise
    write_new(a.evidence/'summary.json',result);return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    r=sub.add_parser('run',allow_abbrev=False)
    for name in ('inputs','request','consumer-code','ops-code','evidence'):r.add_argument('--'+name,type=Path,required=True)
    for name in ('inputs-sha256','request-sha256','driver-source','driver-sha256'):r.add_argument('--'+name,required=True)
    r.add_argument('--deadline-hours',type=float,required=True);a=p.parse_args(argv)
    try:result={'status':'dry','submitted':False,'training':False,'poll_seconds':POLL_SECONDS} if a.command=='dry' else run(a)
    except Exception as exc:print(json.dumps({'status':'ALERT','reason':str(exc),'retry':False}));return 2
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
