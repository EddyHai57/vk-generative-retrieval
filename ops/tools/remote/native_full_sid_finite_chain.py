"""Finite readonly full-B verification -> original full-SID transport, no G training."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import time
from ops.tools.remote import native_a1_finite_chain as base
from ops.tools.remote.p_first_batch_chain import queue_state
from ops.tools.remote.approved_input_chains import submit_argv as original_submit_argv

ROOT=Path(__file__).resolve().parents[3]
FULL_SOURCE='9fac715df389761a4869e5a7118977d802f01c1e'
SID_SOURCE='aa8c0910e407e92d757b91eae64d3ce3137cb027'
FULL_CODE=Path('/workspace/vkgr-sgns-native-full-code-v1')
SID_CODE=Path('/workspace/vkgr-sgns-native-full-sid-code-v1')
OPS_CODE=Path('/workspace/vkgr-native-full-sid-ops-code-v1')
OPS_ROOT=Path('/workspace/vkgr-ops-root-a40-v1')
OPS_SOURCE='5840e6b686ec9651314a9e28a1a2c0befcc05158'
PROOF_HASHES=('receipt_sha256','run_manifest_sha256','terminal_sha256','stdout_sha256')
POLL_SECONDS=900

# The original entry points own all scientific, parent-DAG and runtime decisions.
CHECK_SID='''import json,sys
from pathlib import Path
from analysis import sgns_native_full_sid as m
path,pin,stage=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
v=m.load(path,pin);common,_,_,_=m.load_ops(v['ops_code'])
with m.rss_guard(common) as boundary:
 result=m.accepted(v,path,pin,stage,boundary)
print(json.dumps(result,sort_keys=True,allow_nan=False))
'''
CHECK_CONSUME='''import json,sys
from pathlib import Path
from analysis import sgns_native_full_sid as m
print(json.dumps(m.consume(Path(sys.argv[1]),sys.argv[2]),sort_keys=True,allow_nan=False))
'''
CHECK_PLAN='''import json,sys
from pathlib import Path
from analysis import sgns_native_full_sid as m
path,pin,stage,planpath=Path(sys.argv[1]),sys.argv[2],sys.argv[3],Path(sys.argv[4])
v=m.load(path,pin);ident=m.identity(v);plan=json.loads(planpath.read_bytes());spec=plan['spec']
if ident['source']['commit']!=sys.argv[5]:raise ValueError('fixed SID source differs')
_,launcher,_,_=m.load_ops(v['ops_code']);metadata=launcher.prepare(spec)
if not m.exact(metadata,plan['metadata']):raise ValueError('formal dry metadata changed')
runroot=Path(v['ops_root'])/'runs'/m.rid(v,stage)
env=launcher._recorded_env(metadata['env'],launcher._job_env(metadata['env'],runroot,spec['run_id']))
actual={'schema':'vkgr-run-v1','run_id':spec['run_id'],'kind':spec['kind'],'argv':spec['argv'],'seed':spec['seed'],'stop_mode':spec['stop_mode'],'ops_source':launcher._ops_source(),'resources':{**{k:spec[k] for k in launcher.RESOURCE_ARGS},'wall_seconds':spec['wall_hours']*3600.0},**metadata,'env':env}
m.transport_job(actual,runroot=runroot,code=m.ROOT,source=v['source'],argv=m.argv(stage,path,pin),config={'path':str(path),'sha256':pin},inputs=m.pins(v,path,stage),resources=m.resources(v))
if not m.exact(m.identity(v),ident) or not m.exact(m.load(path,pin),v):raise ValueError('SID dry identity changed')
print(json.dumps({'verified':True,'submitted':False}))
'''


def hexsha(value,n=64):
    if type(value)is not str or len(value)!=n or set(value)-set('0123456789abcdef'):raise ValueError('explicit hex pin required')
    return value


def remaining(args):return base.remaining(args)
def hash_file(args,path,*,block_size=4*1024**2):
    remaining(args);h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            remaining(args);block=f.read(block_size)
            if not block:break
            h.update(block)
    remaining(args);return h.hexdigest()


def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)


def secure(path):
    p=Path(path)
    if not p.is_absolute() or '..' in p.parts or any(q.is_symlink() for q in (p,*p.parents)):raise ValueError('absolute nonsymlink path required')
    return p


def validate_template(v):
    keys={'schema','source','lock','producer','ops_code','ops_root','output','preflight_run_id','fit_run_id','verify_run_id','resources','engineering','test_read'}
    if type(v)is not dict or set(v)!=keys or v['schema']!='UR001-native-full-SID-request-v1' or v['engineering']is not False or v['test_read']is not False:raise ValueError('closed no-test owner SID template required')
    if type(v['source'])is not dict or set(v['source'])!={'commit','files'} or v['source']['commit']!=SID_SOURCE or type(v['source']['files'])is not dict:raise ValueError('fixed SID complete source required')
    p=v['producer']
    if type(p)is not dict or set(p)!={'root','source','request','receipt','verified_producer'} or p['root']!=str(FULL_CODE) or p['source']!=FULL_SOURCE:raise ValueError('fixed full B producer required')
    proof=p['verified_producer']
    if type(proof)is not dict or set(proof)!={'run_id',*PROOF_HASHES} or type(proof['run_id'])is not str or any(proof[k]!='0'*64 for k in PROOF_HASHES):raise ValueError('pending producer proof placeholders required')
    for r in (v['lock'],p['request'],p['receipt']):
        if type(r)is not dict or set(r)!={'path','sha256'}:raise ValueError('closed owner file record required')
        secure(r['path']);hexsha(r['sha256'])
    if p['request']['sha256']!='0'*64 or p['receipt']['sha256']!='0'*64:raise ValueError('only mechanical producer pins may be pending')
    if v['ops_code']!=str(OPS_CODE) or v['ops_root']!=str(OPS_ROOT):raise ValueError('original OPS install required')
    resources=v['resources']
    if type(resources)is not dict or set(resources)!={'cpu_cores','ram_mib','vram_mib','disk_gib','wall_hours'} or any(type(x)is not int or x<0 for x in resources.values()) or any(resources[k]<=0 for k in ('cpu_cores','ram_mib','disk_gib')) or resources['vram_mib']!=0 or resources['wall_hours']!=48:raise ValueError('explicit owner CPU-only 48h resources required')
    ids=[v[k+'_run_id']for k in ('preflight','fit','verify')]
    if len(set(ids))!=3 or any(type(r)is not str or not r.startswith('ur001-native-full-sid-') or set(r)-set('abcdefghijklmnopqrstuvwxyz0123456789-')for r in ids):raise ValueError('three fixed descriptive stage IDs required')
    return v


def paths(value):
    if type(value)is dict:
        for item in value.values():yield from paths(item)
    elif type(value)is list:
        for item in value:yield from paths(item)
    elif type(value)is str and value.startswith('/'):yield secure(value)


def check_paths(args,v):
    evidence=secure(args.evidence);out=secure(v['output']);protected=[ROOT,SID_CODE,FULL_CODE,OPS_CODE,OPS_ROOT,secure(args.template),secure(args.full_request),*paths(v['lock']),*paths(v['producer'])]
    for p in protected:
        if any(root==p or root in p.parents or p in root.parents for root in (evidence,out)):raise ValueError('exclusive output/evidence outside inputs and code/OPS required')
    if evidence==out or evidence in out.parents or out in evidence.parents:raise ValueError('SID output and driver evidence must be separate')


def tree(args,root,pin):
    def git(*argv):return subprocess.check_output(['git','-C',str(root),*argv],timeout=remaining(args))
    secure(root)
    if git('rev-parse','HEAD').decode().strip()!=pin or git('status','--porcelain','--untracked-files=all').strip():raise ValueError('clean fixed deployment required')
    expected=set()
    with tarfile.open(fileobj=io.BytesIO(git('archive',pin))) as archive:
        for member in archive:
            remaining(args)
            if member.isdir():continue
            p=root/member.name;expected.add(member.name)
            if not member.isfile() or p.is_symlink() or not stat.S_ISREG(p.lstat().st_mode):raise ValueError('fixed regular files required')
            h=hashlib.sha256()
            with archive.extractfile(member) as f:
                for block in iter(lambda:f.read(4*1024**2),b''):remaining(args);h.update(block)
            if hash_file(args,p)!=h.hexdigest():raise ValueError('fixed deployment bytes changed')
    actual=set()
    def fail(error):raise error
    for parent,dirs,names in os.walk(root,followlinks=False,onerror=fail):
        remaining(args)
        if Path(parent)==root:dirs[:]=[d for d in dirs if d!='.git'];names=[n for n in names if n!='.git']
        if any((Path(parent)/d).is_symlink()for d in dirs):raise ValueError('symlink directory prohibited')
        for name in names:
            remaining(args);p=Path(parent)/name
            if not stat.S_ISREG(p.lstat().st_mode):raise ValueError('nonregular deployment file')
            actual.add(str(p.relative_to(root)))
    if actual!=expected:raise ValueError('deployment actual file set differs')
    remaining(args)


def fixed(args):
    for root,pin in ((ROOT,args.driver_source),(FULL_CODE,FULL_SOURCE),(SID_CODE,SID_SOURCE),(OPS_CODE,OPS_SOURCE)):tree(args,root,pin)
    if hash_file(args,__file__)!=args.driver_sha256 or hash_file(args,args.template)!=args.template_sha256:raise ValueError('fixed driver/template bytes changed')
    v=validate_template(json.loads(args.template.read_bytes()));check_paths(args,v)
    if Path(v['producer']['request']['path'])!=args.full_request:raise ValueError('owner full request path differs')
    if getattr(args,'full_pin',None) and hash_file(args,args.full_request)!=args.full_pin:raise ValueError('original full request changed')
    if getattr(args,'sid_pin',None) and hash_file(args,args.sid_request)!=args.sid_pin:raise ValueError('generated SID request changed')
    return v


def materialize(template,full,request_pin,receipt_pin,proof):
    validate_template(template);hexsha(request_pin);hexsha(receipt_pin);p=template['producer']
    if (full['runner_source']!=FULL_SOURCE or full['fit_run_id']!=p['verified_producer']['run_id'] or full['ops_root']!=template['ops_root'] or full['lock']!=template['lock']['path'] or p['receipt']['path']!=str(Path(full['output'])/'fit/result.json')):raise ValueError('owner/full producer lineage differs')
    if type(proof)is not dict or set(proof)!={'run_id',*PROOF_HASHES} or proof['run_id']!=full['fit_run_id']:raise ValueError('complete original verified producer required')
    for k in PROOF_HASHES:
        hexsha(proof[k])
        if proof[k]=='0'*64:raise ValueError('real producer proof required')
    if proof['receipt_sha256']!=receipt_pin:raise ValueError('producer receipt pin differs')
    result=copy.deepcopy(template);result['producer']['request']['sha256']=request_pin;result['producer']['receipt']['sha256']=receipt_pin;result['producer']['verified_producer']=proof
    return result


def execute(args,code,argv,name,*,json_output=True,before_submit=None):
    fixed(args);remaining(args)
    env={k:v for k,v in os.environ.items()if not k.startswith('VKGR_') and k not in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')}
    env.update(PYTHONPATH=str(code)+':'+str(code/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if before_submit is not None and not before_submit():
        remaining(args);return None
    result=subprocess.run(argv,cwd=code,env=env,capture_output=True,timeout=remaining(args))
    for suffix,data in (('stdout',result.stdout),('stderr',result.stderr)):
        with (args.evidence/(name+'.'+suffix)).open('xb')as f:f.write(data)
    remaining(args)
    if result.returncode:raise ValueError('original command failed: '+name)
    fixed(args);remaining(args)
    return json.loads(result.stdout)if json_output else result.stdout.decode().strip()


def poll(args,state,rid,sleep):
    with (args.evidence/'progress.jsonl').open('a')as f:f.write(json.dumps({'run_id':rid,'queue':state['state'],'time':args.now()})+'\n')
    sleep(min(POLL_SECONDS,remaining(args)))


def wait_full(args,template,sleep):
    seen=False;rid=template['producer']['verified_producer']['run_id'];run=OPS_ROOT/'runs'/rid
    while True:
        remaining(args);fixed(args);state=queue_state(OPS_ROOT,rid)
        if args.full_request.exists():
            pin=hash_file(args,args.full_request)
            if getattr(args,'full_pin',pin)!=pin:raise ValueError('original full request changed')
            args.full_pin=pin
        if base.terminal(run,rid):
            if state['state']!='claimed':raise ValueError('original full fit not claimed')
            if not getattr(args,'full_pin',None):raise ValueError('original full request absent')
            result=execute(args,FULL_CODE,[sys.executable,'-B','-m','analysis.sgns_native_full','verify','--stage','fit','--request',str(args.full_request),'--request-sha256',args.full_pin],'full-verified')
            full=json.loads(args.full_request.read_bytes());receipt=Path(template['producer']['receipt']['path'])
            proof={'run_id':rid,'receipt_sha256':hash_file(args,receipt),**{k:hash_file(args,run/n)for k,n in [('run_manifest_sha256','run.json'),('terminal_sha256','terminal.json'),('stdout_sha256','logs/stdout.log')]}}
            if result!={'status':'verified','readonly':True,'producer':proof}:raise ValueError('original full accepted closure differs')
            return materialize(template,full,args.full_pin,proof['receipt_sha256'],proof)
        if state['state']=='absent' and (run.exists() or seen):raise ValueError('full fit disappeared; no retry')
        seen=seen or state['state']in ('pending','claimed');poll(args,state,rid,sleep)


def submit_argv(spec):
    argv=original_submit_argv(spec)
    expected=['/workspace/vkgr-env-a40-v1/.venv/bin/python','-B','/workspace/vkgr-ops-code-a40-v1/ops/bin/vkgr-sched','submit','--root',str(OPS_ROOT)]
    if type(argv)is not list or argv[:6]!=expected:raise ValueError('fixed original submit prefix differs')
    return [*argv[:2],str(OPS_CODE/'ops/bin/vkgr-sched'),*argv[3:]]


def submission_state(v,stage):
    rid=v[stage+'_run_id'];state=queue_state(OPS_ROOT,rid)
    if state['state']=='absent' and ((OPS_ROOT/'runs'/rid).exists() or (Path(v['output'])/stage).exists()):raise ValueError('run/output exists without queue; never resubmit')
    return state


def sid_stage(args,v,stage,sleep):
    rid=v[stage+'_run_id'];run=OPS_ROOT/'runs'/rid;submitted=False;seen=False
    while True:
        remaining(args);fixed(args);state=queue_state(OPS_ROOT,rid)
        if base.terminal(run,rid):
            if state['state']!='claimed':raise ValueError('SID successful job not claimed')
            result=execute(args,SID_CODE,[sys.executable,'-B','-c',CHECK_SID,str(args.sid_request),args.sid_pin,stage],stage+'-accepted')
            if type(result)is not dict or result.get('producer',{}).get('run_id')!=rid:raise ValueError('original SID acceptance missing')
            return result
        if state['state']=='absent':
            if run.exists() or (Path(v['output'])/stage).exists() or seen or submitted:raise ValueError('SID stage disappeared; no retry')
            plan=execute(args,SID_CODE,[sys.executable,'-B','-m','analysis.sgns_native_full_sid','plan','--stage',stage,'--request',str(args.sid_request),'--request-sha256',args.sid_pin],stage+'-plan')
            if plan.get('status')!='prepared_dry' or plan.get('spec',{}).get('run_id')!=rid or plan.get('metadata',{}).get('git')!={'commit':SID_SOURCE,'dirty':False,'dirty_files':[]}:raise ValueError('SID formal plan differs')
            checked=execute(args,SID_CODE,[sys.executable,'-B','-c',CHECK_PLAN,str(args.sid_request),args.sid_pin,stage,str(args.evidence/(stage+'-plan.stdout')),SID_SOURCE],stage+'-plan-checked')
            if checked!={'verified':True,'submitted':False}:raise ValueError('complete launcher dry not verified')
            # Planning and hashing can take time. Observe newly queued work
            # before invoking submit, then check again after final fixed-tree
            # checks immediately before the scheduler subprocess.
            state=submission_state(v,stage)
            if state['state']!='absent':
                seen=True;poll(args,state,rid,sleep);continue
            result=execute(args,SID_CODE,submit_argv(plan['spec']),stage+'-submit',json_output=False,before_submit=lambda:submission_state(v,stage)['state']=='absent')
            submitted=result is not None
            state=submission_state(v,stage)
            if state['state']not in ('pending','claimed'):raise ValueError('submission absent; never resubmit')
        seen=seen or state['state']in ('pending','claimed');poll(args,state,rid,sleep)


def run(args,*,now=time.time,sleep=time.sleep):
    if os.getppid()!=1:raise ValueError('detached PPID1 required')
    args.now=now;args.deadline=now()+base.deadline_seconds(args.deadline_hours)
    template=fixed(args);args.evidence.mkdir(parents=True,exist_ok=False)
    try:
        v=wait_full(args,template,sleep);fixed(args);remaining(args)
        args.sid_request=args.evidence/'sid-request.json';write_new(args.sid_request,v);args.sid_pin=hash_file(args,args.sid_request)
        for stage in ('preflight','fit','verify'):sid_stage(args,v,stage,sleep)
        receipt=Path(v['output'])/'verify/stage-receipt.json';pin=hash_file(args,receipt)
        consumed=execute(args,SID_CODE,[sys.executable,'-B','-c',CHECK_CONSUME,str(receipt),pin],'sid-consumed')
        if consumed.get('producer',{}).get('run_id')!=v['verify_run_id']:raise ValueError('original SID consume binding differs')
        fixed(args);remaining(args)
        result={'status':'original_native_full_sid_consumed','request_sha256':args.sid_pin,'consume':consumed,'ended_at':now(),'resource_measurement':'not_measured owner explicit first declaration','g_training_submitted':False,'test_read':False}
    except Exception as exc:
        write_new(args.evidence/'summary.json',{'status':'ALERT','reason':str(exc),'ended_at':now()});raise
    write_new(args.evidence/'summary.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('dry');r=sub.add_parser('run',allow_abbrev=False)
    for name in ('template','full-request','evidence'):r.add_argument('--'+name,type=Path,required=True)
    for name in ('template-sha256','driver-source','driver-sha256'):r.add_argument('--'+name,required=True)
    r.add_argument('--deadline-hours',type=float,default=200);a=p.parse_args(argv)
    try:result={'status':'dry','submitted':False,'training':False,'original_full_verified_required':True}if a.command=='dry'else run(a)
    except Exception as exc:print(json.dumps({'status':'ALERT','reason':str(exc)}),flush=True);return 2
    print(json.dumps(result,sort_keys=True,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
