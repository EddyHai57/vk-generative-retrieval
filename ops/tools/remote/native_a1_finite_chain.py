"""Finite native B A1 PF/train/verify/accept continuation; never submits full fit."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from ops.tools.remote.p_first_batch_chain import queue_state,sha
from ops.tools.remote.approved_input_chains import submit_argv

ROOT=Path(__file__).resolve().parents[3]
CONSUMER_SOURCE='96bfb9e6fcb4081168cf4d39563f50c58ee3fbad'
CONSUMER_ENTRY_SHA='96c1e851e84d820607ef29cd7708da9fe5bab2aeff8492d86e9aa9cd9a3deb01'
STAGES=('preflight','train','verify','accept')
POLL_SECONDS=900

# Every acceptance decision stays in the already independently reviewed consumer.
CHECK='''import json,sys
from pathlib import Path
from analysis import sgns_native_subset_a1 as m
path,pin,stage,scratch=Path(sys.argv[1]),sys.argv[2],sys.argv[3],Path(sys.argv[4])
v=m.load(path,pin);source,env=m.identity(v)
if source['commit']!=sys.argv[5] or source['files'].get('analysis/sgns_native_subset_a1.py')!=sys.argv[6]:raise ValueError('fixed original consumer differs')
if stage=='accept':
 receipt=m.output(v,'accept')/'stage-receipt.json'
 result=m.consume(receipt,m.digest(receipt),scratch)
else:
 result=m.successful(v,path,pin,stage)[1]
 if stage in ('preflight','train'):m.predecessor(v,'train' if stage=='preflight' else 'verify',pin,source,request=path)
if m.identity(v)!=(source,env):raise ValueError('consumer identity changed')
print(json.dumps(result,sort_keys=True,allow_nan=False))
'''


CHECK_PLAN='''import json,sys
from pathlib import Path
from analysis import sgns_native_subset_a1 as m
from analysis.a40_ops import load_ops
path,pin,stage,planpath=Path(sys.argv[1]),sys.argv[2],sys.argv[3],Path(sys.argv[4])
v=m.load(path,pin);source,_=m.identity(v);plan=json.loads(planpath.read_bytes());spec=plan['spec']
_,launcher,_,_=load_ops(v['ops_code']);metadata=launcher.prepare(spec)
if not m.exact(metadata,plan['metadata']):raise ValueError('formal dry metadata changed')
runroot=Path(v['ops_root'])/'runs'/m.RUN_IDS[stage]
env=launcher._recorded_env(metadata['env'],launcher._job_env(metadata['env'],runroot,spec['run_id']))
actual={'schema':'vkgr-run-v1','run_id':spec['run_id'],'kind':spec['kind'],'argv':spec['argv'],'seed':spec['seed'],'stop_mode':spec['stop_mode'],'ops_source':launcher._ops_source(),'resources':{**{k:spec[k] for k in launcher.RESOURCE_ARGS},'wall_seconds':spec['wall_hours']*3600.0},**metadata,'env':env}
expected=m.contract(v,path,pin,stage)
m.transport_job(actual,runroot=runroot,code=m.ROOT,source=source,argv=expected['argv'],config=expected['config'],inputs=expected['inputs'],resources=expected['resources'],cublas=True,data_root=v['data_root'])
if m.identity(v)[0]!=source:raise ValueError('consumer identity changed during dry')
print(json.dumps({'verified':True,'submitted':False}))
'''


def deadline_seconds(hours):
    if isinstance(hours,bool) or not isinstance(hours,(int,float)) or not math.isfinite(hours) or not 0<hours<=200:raise ValueError('finite deadline within 200 hours required')
    return hours*3600


def terminal(run,run_id):
    path=Path(run)/'terminal.json'
    if not path.exists():return False
    v=json.loads(path.read_bytes())
    if (v.get('schema')!='vkgr-terminal-v1' or v.get('run_id')!=run_id or v.get('status')!='succeeded'
        or type(v.get('exit_code'))is not int or v['exit_code']!=0 or v.get('signal')is not None or v.get('wall_cap_exceeded')is not False):raise ValueError('original stage failed; no automatic retry')
    return True


def check_plan(plan,stage):
    if (plan.get('status')!='prepared_dry' or plan.get('spec',{}).get('run_id')!='sgns-native-subset-a1-'+stage+'-v1'
        or plan.get('metadata',{}).get('git')!={'commit':CONSUMER_SOURCE,'dirty':False,'dirty_files':[]}):raise ValueError('fixed consumer formal dry differs')


def driver_identity(source,pin):
    actual=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all'],text=True).strip()
    relative=Path(__file__).resolve().relative_to(ROOT).as_posix()
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',source+':'+relative])
    if actual!=source or dirty or raw!=Path(__file__).read_bytes() or sha(__file__)!=pin:raise ValueError('clean fixed driver bytes required')


def remaining(args):
    seconds=args.deadline-args.now()
    if seconds<=0:raise TimeoutError("finite A1 chain deadline")
    return seconds


def command(args,argv,name):
    result=subprocess.run(argv,cwd=args.consumer_code,capture_output=True,env=args.env,timeout=remaining(args))
    remaining(args)
    for suffix,data in (('stdout',result.stdout),('stderr',result.stderr)):
        with (args.evidence/(name+'.'+suffix)).open('xb') as f:f.write(data)
    if result.returncode:raise ValueError('original CLI failed: '+name)
    return json.loads(result.stdout)


def fixed_request(args):
    driver_identity(args.driver_source,args.driver_sha256)
    if sha(args.request)!=args.request_sha256:raise ValueError('original request bytes differ')
    raw=subprocess.check_output(['git','-C',str(args.consumer_code),'rev-parse','HEAD'],text=True).strip()
    if raw!=CONSUMER_SOURCE or sha(args.consumer_code/'analysis/sgns_native_subset_a1.py')!=CONSUMER_ENTRY_SHA:raise ValueError('fixed consumer source differs')
    v=json.loads(args.request.read_bytes())
    if v.get('source',{}).get('commit')!=CONSUMER_SOURCE:raise ValueError('fixed request consumer differs')
    return v


def evidence_path(args,v):
    root=args.evidence.resolve()
    protected=[ROOT.resolve(),args.consumer_code.resolve()]
    def visit(value):
        if isinstance(value,dict):
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
        elif isinstance(value,str) and value.startswith("/"):protected.append(Path(value).resolve())
    visit(v)
    if not args.evidence.is_absolute() or args.evidence!=root or any(root==p or p in root.parents or root in p.parents for p in protected):raise ValueError("exclusive evidence outside original trees required")


def run(args,*,now=time.time,sleep=time.sleep):
    if os.getppid()!=1:raise ValueError('detached PPID1 required')
    seconds=deadline_seconds(args.deadline_hours)
    args.now=now;args.deadline=now()+seconds
    v=fixed_request(args);evidence_path(args,v)
    args.evidence.mkdir(parents=True,exist_ok=False)
    args.env={**os.environ,'PYTHONPATH':str(args.consumer_code)+':'+str(args.consumer_code/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    for key in ('VKGR_RUN_ID','VKGR_OPS_ROOT','VKGR_PROGRESS_PATH','CUDA_VISIBLE_DEVICES'):args.env.pop(key,None)
    deadline=args.deadline
    try:
        ops=Path(v['ops_root']);ops_code=Path(v['ops_code'])
        for index,stage in enumerate(STAGES):
            rid='sgns-native-subset-a1-'+stage+'-v1';submitted=False
            while True:
                if now()>=deadline:raise TimeoutError('finite A1 chain deadline')
                if fixed_request(args)!=v:raise ValueError('original request changed')
                state=queue_state(ops,rid);root=ops/'runs'/rid
                if terminal(root,rid):
                    if state['state']!='claimed':raise ValueError('successful original job not claimed')
                    command(args,[sys.executable,'-B','-c',CHECK,str(args.request),args.request_sha256,stage,str(args.evidence/'total-consume'),CONSUMER_SOURCE,CONSUMER_ENTRY_SHA],stage+'-accepted')
                    remaining(args);fixed_request(args);remaining(args)
                    break
                if state['state']=='absent' and root.exists():raise ValueError('original claimed stage disappeared from queue')
                if state['state']=='absent' and not root.exists():
                    if index==0 or submitted:raise ValueError('original stage disappeared or initial PF not submitted')
                    plan=command(args,[sys.executable,'-B','-m','analysis.sgns_native_subset_a1','plan','--stage',stage,'--request',str(args.request),'--request-sha256',args.request_sha256,'--ops-code',str(ops_code)],stage+'-plan')
                    check_plan(plan,stage);fixed_request(args)
                    checked=command(args,[sys.executable,'-B','-c',CHECK_PLAN,str(args.request),args.request_sha256,stage,str(args.evidence/(stage+'-plan.stdout'))],stage+'-plan-checked')
                    if checked!={'verified':True,'submitted':False}:raise ValueError('complete original launcher contract not verified')
                    fixed_request(args)
                    # submit_argv uses the registered fixed OPS install and original spec.
                    if ops!=Path('/workspace/vkgr-ops-root-a40-v1') or ops_code!=Path('/workspace/vkgr-ops-code-a40-v1'):raise ValueError('registered fixed OPS paths required')
                    result=subprocess.run(submit_argv(plan['spec']),cwd=args.consumer_code,capture_output=True,env=args.env,timeout=remaining(args))
                    remaining(args)
                    for suffix,data in (('stdout',result.stdout),('stderr',result.stderr)):
                        with (args.evidence/(stage+'-submit.'+suffix)).open('xb') as f:f.write(data)
                    if result.returncode:raise ValueError('scheduler submission failed; inspect original queue, never resubmit')
                    submitted=True;state=queue_state(ops,rid)
                    if state['state']not in ('pending','claimed'):raise ValueError('submission absent from original queue')
                with (args.evidence/'progress.jsonl').open('a') as f:f.write(json.dumps({'stage':stage,'run_id':rid,'time':now(),'queue':state['state']})+'\n')
                sleep(min(POLL_SECONDS,remaining(args)))
        remaining(args);fixed_request(args);remaining(args)
        result={'status':'original_four_gate_chain_accepted','full_fit_submitted':False,'ended_at':now()}
    except Exception as exc:
        with (args.evidence/'summary.json').open('x') as f:json.dump({'status':'ALERT','reason':str(exc),'full_fit_submitted':False,'ended_at':now()},f)
        raise
    with (args.evidence/'summary.json').open('x') as f:json.dump(result,f)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    r=sub.add_parser('run',allow_abbrev=False)
    for name in ('request','consumer-code','evidence'):r.add_argument('--'+name,type=Path,required=True)
    for name in ('request-sha256','driver-source','driver-sha256'):r.add_argument('--'+name,required=True)
    r.add_argument('--deadline-hours',type=float,default=200)
    a=p.parse_args(argv)
    try:result={'status':'dry','training':False,'full_fit_submitted':False} if a.command=='dry' else run(a)
    except Exception as exc:
        print(json.dumps({'status':'ALERT','reason':str(exc),'full_fit_submitted':False}),flush=True);return 2
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
