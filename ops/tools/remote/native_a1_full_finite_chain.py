"""Finite original native B A1 acceptance -> full PF/fit/readonly-verify transport."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tarfile
import time
from ops.tools.remote import native_a1_finite_chain as base
from ops.tools.remote.p_first_batch_chain import queue_state,sha
from ops.tools.remote.approved_input_chains import submit_argv

ROOT=Path(__file__).resolve().parents[3]
A1_SOURCE=base.CONSUMER_SOURCE
FULL_SOURCE='9fac715df389761a4869e5a7118977d802f01c1e'
FULL_ENTRY_SHA='b8799ee1c8fcbdfc3dedee9d3b590cfde25c970c70f40d9f20d151fc2873730c'
TEMPLATE_SHA='6e7963a90e457363ad9ae5591aab6017a760da4e52aa9b2eb6077f64b8339089'
FULL_CODE=Path('/workspace/vkgr-sgns-native-full-code-v1')
OPS_CODE=Path('/workspace/vkgr-ops-code-a40-v1')
OPS_ROOT=Path('/workspace/vkgr-ops-root-a40-v1')
POLL_SECONDS=900

CHECK_PLAN='''import json,sys
from pathlib import Path
from analysis import sgns_native_full as m
from analysis.a40_ops import load_ops
from analysis.sgns_subset_acceptance_a1 import transport_job
path,pin,stage,planpath,opscode=Path(sys.argv[1]),sys.argv[2],sys.argv[3],Path(sys.argv[4]),sys.argv[5]
v=m.load(path,pin);source=m.source_identity();plan=json.loads(planpath.read_bytes());spec=plan['spec']
if source['commit']!=sys.argv[6] or source['files'].get('analysis/sgns_native_full.py')!=sys.argv[7]:raise ValueError('fixed full consumer differs')
_,launcher,_,_=load_ops(opscode);metadata=launcher.prepare(spec)
if not m.exact(metadata,plan['metadata']):raise ValueError('formal dry metadata changed')
runroot=Path(v['ops_root'])/'runs'/m.rid(v,stage)
env=launcher._recorded_env(metadata['env'],launcher._job_env(metadata['env'],runroot,spec['run_id']))
actual={'schema':'vkgr-run-v1','run_id':spec['run_id'],'kind':spec['kind'],'argv':spec['argv'],'seed':spec['seed'],'stop_mode':spec['stop_mode'],'ops_source':launcher._ops_source(),'resources':{**{k:spec[k] for k in launcher.RESOURCE_ARGS},'wall_seconds':spec['wall_hours']*3600.0},**metadata,'env':env}
transport_job(actual,runroot=runroot,code=m.ROOT,source=source,argv=m.argv(stage,path,pin),config={'path':str(path),'sha256':pin},inputs=m.pins(v,path,stage),resources=m.resources(v))
if not m.exact(m.source_identity(),source) or not m.exact(m.load(path,pin),v):raise ValueError('full dry source/request changed')
print(json.dumps({'verified':True,'submitted':False}))
'''


def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)


def git(args,root,*argv):
    return subprocess.check_output(['git','-C',str(root),*argv],timeout=base.remaining(args))


def tree(args,root,source):
    if git(args,root,'rev-parse','HEAD').decode().strip()!=source or git(args,root,'status','--porcelain','--untracked-files=all').strip():raise ValueError('clean fixed deployment required')
    raw=git(args,root,'archive',source)
    expected=set()
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            base.remaining(args)
            if member.isdir():continue
            expected.add(member.name)
            path=root/member.name
            if not member.isfile() or path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) or path.read_bytes()!=archive.extractfile(member).read():raise ValueError('fixed deployment bytes changed')
    actual=set()
    def failed(error):raise error
    if root.is_symlink():raise ValueError('deployment root is a symlink')
    for parent,dirs,names in os.walk(root,followlinks=False,onerror=failed):
        base.remaining(args)
        if Path(parent)==root:
            dirs[:]=[name for name in dirs if name!='.git']
            names=[name for name in names if name!='.git']
        if any((Path(parent)/name).is_symlink() for name in dirs):raise ValueError('deployment contains symlink directory')
        for name in names:
            base.remaining(args)
            path=Path(parent)/name
            if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('deployment contains nonregular file')
            actual.add(str(path.relative_to(root)))
    if actual!=expected:raise ValueError('deployment actual file set differs from fixed Git tree')
    base.remaining(args)


def fixed_inputs(args):
    tree(args,ROOT,args.driver_source)
    if sha(__file__)!=args.driver_sha256:raise ValueError('fixed driver entry differs')
    if sha(args.template)!=TEMPLATE_SHA or sha(args.a1_request)!=args.a1_request_sha256:raise ValueError('owner template/original A1 request changed')
    template=json.loads(args.template.read_bytes());a1=json.loads(args.a1_request.read_bytes())
    if a1.get('source',{}).get('commit')!=A1_SOURCE:raise ValueError('original A1 source required')
    tree(args,Path(template['acceptance']['root']),A1_SOURCE);tree(args,FULL_CODE,FULL_SOURCE)
    if sha(Path(template['acceptance']['root'])/'analysis/sgns_native_subset_a1.py')!=base.CONSUMER_ENTRY_SHA or sha(FULL_CODE/'analysis/sgns_native_full.py')!=FULL_ENTRY_SHA:raise ValueError('fixed consumer entry differs')
    if getattr(args,'receipt_pin',None) and sha(template['acceptance']['receipt'])!=args.receipt_pin:raise ValueError('original acceptance receipt changed')
    if getattr(args,'full_pin',None) and sha(args.full_request)!=args.full_pin:raise ValueError('generated full request changed')
    base.remaining(args)
    return a1,template


def evidence_path(args,a1,template):
    args.consumer_code=FULL_CODE
    base.evidence_path(args,{'a1':a1,'template':template,'template_path':str(args.template),'a1_request':str(args.a1_request)})


def materialize(template,a1,receipt_pin):
    if (template.get('runner_source')!=FULL_SOURCE or template.get('resources',{}).get('wall_hours')!=3
        or template['acceptance']['receipt_sha256']!='0'*64 or template['ops_root']!=a1['ops_root']
        or template['acceptance']['receipt']!=str(Path(a1['output_root'])/'accept/stage-receipt.json')):raise ValueError('fixed owner template/receipt lineage differs')
    if len(receipt_pin)!=64 or set(receipt_pin)-set('0123456789abcdef') or receipt_pin=='0'*64:raise ValueError('actual accepted receipt hash required')
    value=json.loads(json.dumps(template));value['acceptance']['receipt_sha256']=receipt_pin
    return value


def execute(args,code,argv,name,*,json_output=True):
    args.consumer_code=Path(code)
    args.env={k:v for k,v in os.environ.items() if not k.startswith('VKGR_') and k not in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')}
    args.env.update(PYTHONPATH=str(code)+':'+str(Path(code)/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if Path(code)!=FULL_CODE:args.env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    result=subprocess.run(argv,cwd=code,capture_output=True,env=args.env,timeout=base.remaining(args))
    for suffix,data in (('stdout',result.stdout),('stderr',result.stderr)):
        with (args.evidence/(name+'.'+suffix)).open('xb') as f:f.write(data)
    base.remaining(args)
    if result.returncode:raise ValueError('original CLI failed: '+name)
    fixed_inputs(args);base.remaining(args)
    return json.loads(result.stdout) if json_output else result.stdout.decode().strip()


def poll(args,state,run_id,sleep):
    with (args.evidence/'progress.jsonl').open('a') as f:f.write(json.dumps({'run_id':run_id,'queue':state['state'],'time':args.now()})+'\n')
    sleep(min(POLL_SECONDS,base.remaining(args)))


def wait_a1(args,a1,template,sleep):
    rid='sgns-native-subset-a1-accept-v1';ops=Path(a1['ops_root']);run=ops/'runs'/rid;seen=False
    while True:
        base.remaining(args);fixed_inputs(args);state=queue_state(ops,rid)
        if base.terminal(run,rid):
            if state['state']!='claimed':raise ValueError('original A1 acceptance not claimed')
            result=execute(args,template['acceptance']['root'],[sys.executable,'-B','-c',base.CHECK,str(args.a1_request),args.a1_request_sha256,'accept',str(args.evidence/'a1-consume'),A1_SOURCE,base.CONSUMER_ENTRY_SHA],'a1-consumed')
            receipt=Path(template['acceptance']['receipt']);pin=sha(receipt)
            producer={name:sha(run/name) for name in ('run.json','terminal.json','logs/stdout.log')}
            if (type(result)is not dict or set(result)!={'receipt_sha256','request_sha256','producer','proof'} or result['receipt_sha256']!=pin
                or result['request_sha256']!=args.a1_request_sha256 or result['producer']!=producer):raise ValueError('original four-gate consumer binding differs')
            args.receipt_pin=pin;fixed_inputs(args)
            return pin
        if state['state']=='absent' and (run.exists() or seen):raise ValueError('original A1 acceptance disappeared')
        seen=seen or state['state'] in ('pending','claimed')
        poll(args,state,rid,sleep)


def verify_result(value,stage,result):
    rid=value['preflight_run_id' if stage=='preflight' else 'fit_run_id'];run=Path(value['ops_root'])/'runs'/rid
    producer={'run_id':rid,'receipt_sha256':sha(Path(value['output'])/stage/'result.json'),**{key:sha(run/name) for key,name in [('run_manifest_sha256','run.json'),('terminal_sha256','terminal.json'),('stdout_sha256','logs/stdout.log')]}}
    if result!={'status':'verified','readonly':True,'producer':producer}:raise ValueError('original readonly verification producer binding differs')


def full_stage(args,value,stage,sleep):
    rid=value['preflight_run_id' if stage=='preflight' else 'fit_run_id'];ops=Path(value['ops_root']);run=ops/'runs'/rid;submitted=False;seen=False
    while True:
        base.remaining(args);fixed_inputs(args);state=queue_state(ops,rid)
        if base.terminal(run,rid):
            if state['state']!='claimed':raise ValueError('successful original full stage not claimed')
            result=execute(args,FULL_CODE,[sys.executable,'-B','-m','analysis.sgns_native_full','verify','--stage',stage,'--request',str(args.full_request),'--request-sha256',args.full_pin],stage+'-verified')
            verify_result(value,stage,result);fixed_inputs(args);base.remaining(args)
            return
        if state['state']=='absent':
            if run.exists() or submitted or seen:raise ValueError('original full stage disappeared; never resubmit')
            plan=execute(args,FULL_CODE,[sys.executable,'-B','-m','analysis.sgns_native_full','plan','--stage',stage,'--request',str(args.full_request),'--request-sha256',args.full_pin,'--ops-code',str(OPS_CODE)],stage+'-plan')
            if (plan.get('status')!='prepared_dry' or plan.get('spec',{}).get('run_id')!=rid or plan.get('metadata',{}).get('git')!={'commit':FULL_SOURCE,'dirty':False,'dirty_files':[]}):raise ValueError('original full dry identity differs')
            checked=execute(args,FULL_CODE,[sys.executable,'-B','-c',CHECK_PLAN,str(args.full_request),args.full_pin,stage,str(args.evidence/(stage+'-plan.stdout')),str(OPS_CODE),FULL_SOURCE,FULL_ENTRY_SHA],stage+'-plan-checked')
            if checked!={'verified':True,'submitted':False}:raise ValueError('complete launcher contract not checked')
            fixed_inputs(args)
            if ops!=OPS_ROOT:raise ValueError('registered OPS root required')
            execute(args,FULL_CODE,submit_argv(plan['spec']),stage+'-submit',json_output=False);submitted=True
            state=queue_state(ops,rid)
            if state['state']not in ('pending','claimed'):raise ValueError('submission absent; inspect original queue, never resubmit')
        seen=seen or state['state'] in ('pending','claimed')
        poll(args,state,rid,sleep)


def run(args,*,now=time.time,sleep=time.sleep):
    if os.getppid()!=1:raise ValueError('detached PPID1 required')
    args.now=now;args.deadline=now()+base.deadline_seconds(args.deadline_hours)
    a1,template=fixed_inputs(args);evidence_path(args,a1,template);args.evidence.mkdir(parents=True,exist_ok=False)
    try:
        pin=wait_a1(args,a1,template,sleep);value=materialize(template,a1,pin)
        fixed_inputs(args);base.remaining(args)
        args.full_request=args.evidence/'full-request.json';write_new(args.full_request,value);args.full_pin=sha(args.full_request)
        fixed_inputs(args)
        for stage in ('preflight','fit'):full_stage(args,value,stage,sleep)
        fixed_inputs(args);base.remaining(args)
        result={'status':'original_native_full_verified','request_sha256':args.full_pin,'receipt_sha256':pin,'ended_at':now(),'resource_measurement':'not_measured conservative first full declaration'}
    except Exception as exc:
        write_new(args.evidence/'summary.json',{'status':'ALERT','reason':str(exc),'ended_at':now()});raise
    write_new(args.evidence/'summary.json',result)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('dry');r=sub.add_parser('run',allow_abbrev=False)
    for name in ('template','a1-request','evidence'):r.add_argument('--'+name,type=Path,required=True)
    for name in ('a1-request-sha256','driver-source','driver-sha256'):r.add_argument('--'+name,required=True)
    r.add_argument('--deadline-hours',type=float,default=200);a=p.parse_args(argv)
    try:result={'status':'dry','training':False,'submitted':False,'requires_original_four_gates':True} if a.command=='dry' else run(a)
    except Exception as exc:print(json.dumps({'status':'ALERT','reason':str(exc)}),flush=True);return 2
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
