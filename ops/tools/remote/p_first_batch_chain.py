"""Finite single-seed approved P PF -> train -> verify transport driver.

Owner starts this only after the registered conditional-entry requirements pass.
No scientific eligibility decision, config change, retry or scheduler tick here.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
POLL_SECONDS=900


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def write_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False)


def deadline_seconds(hours):
    if isinstance(hours,bool) or not isinstance(hours,(int,float)) or not math.isfinite(hours) or not 0<hours<=200:
        raise ValueError('finite driver deadline must be within 200 hours')
    return hours*3600


def queue_state(root,run_id):
    hits=[(state,path) for state in ('pending','claimed','rejected') for path in (Path(root)/'queue'/state).glob('*-'+run_id+'.json')]
    if len(hits)>1:raise ValueError('duplicate original run ID in queue')
    if not hits:return {'state':'absent','path':None}
    state,path=hits[0]
    if state=='rejected':
        reason=path.with_suffix('.reason')
        raise ValueError('original job rejected; no resubmit: '+(reason.read_text() if reason.exists() else 'reason unavailable'))
    return {'state':state,'path':path}


def stage_request(base,predecessor,evidence,stage,state,run):
    value={**base,'predecessor':predecessor}
    if (Path(run)/'run.json').exists():
        original=json.loads((Path(run)/'run.json').read_bytes())['config']
        path=Path(original['path'])
        if sha(path)!=original['sha256']:raise ValueError('original request pin differs')
    elif state['path'] is not None:
        path=Path(json.loads(Path(state['path']).read_bytes())['config'])
    else:
        path=Path(evidence)/(stage+'-request.json')
        write_new(path,value)
    if json.loads(path.read_bytes())!=value:raise ValueError('existing stage request differs beyond mechanical predecessor')
    return path,sha(path),value


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def compare_job(actual,expected):
    if (type(actual) is not dict or set(actual)-set(expected)-{'created_at','admission'}
            or any(canonical(actual.get(k))!=canonical(v) for k,v in expected.items())):
        raise ValueError('complete original job contract differs')


def prepare_exact(plan,launcher):
    spec=launcher.spec_from_args(launcher.build_parser().parse_args(plan['launcher_argv']))
    metadata=launcher.prepare(spec)
    expected=plan['expected_job']
    env={**metadata['env'],'VKGR_RUN_ID':spec['run_id'],
         'VKGR_PROGRESS_PATH':expected['env']['VKGR_PROGRESS_PATH']}
    actual={'schema':'vkgr-run-v1','run_id':spec['run_id'],'kind':spec['kind'],'argv':spec['argv'],
            'seed':spec['seed'],'stop_mode':spec['stop_mode'],'ops_source':launcher._ops_source(),
            'resources':{**{k:spec[k] for k in launcher.RESOURCE_ARGS},'wall_seconds':spec['wall_hours']*3600.0},
            **metadata,'env':env}
    compare_job(actual,expected)
    if spec.get('no_admission') or spec.get('pinned_manifest') or spec.get('pinned_files'):
        raise ValueError('unregistered scheduler bypass/overlay refused')
    return spec,metadata


def terminal(run,run_id):
    path=Path(run)/'terminal.json'
    if not path.exists():return False
    value=json.loads(path.read_bytes())
    if (value.get('schema')!='vkgr-terminal-v1' or value.get('run_id')!=run_id or value.get('status')!='succeeded'
            or value.get('exit_code')!=0 or value.get('signal') is not None or value.get('wall_cap_exceeded') is not False):
        raise ValueError('original job terminal failed; no retry')
    return True


def complete(backend,helper,dag,value,inputs,request,digest,stage,binding=None):
    run=Path(value['ops_root'])/'runs'/value['run_ids'][stage]
    if not terminal(run,value['run_ids'][stage]):raise ValueError('required original stage is not terminal')
    path=backend.stage_receipt_path(Path(value['output_root']),stage)
    pin=helper.harvest_pin(path,run,sha(path))
    proof=dag.verify_producer(path,run,pin);receipt=proof['manifest']
    records=backend.stage_inputs(value,inputs,request,digest,stage)
    expected=backend.job_contract(value,records,request,digest,stage)
    compare_job(backend.strict_json((run/'run.json').read_bytes()),expected)
    for rec in records:
        p=backend.record_path({'path':rec['path'],'sha256':rec['sha256']})
        if p.stat().st_size!=rec['bytes']:raise ValueError('original input bytes differ')
    semantics={'schema':'p-full-backend-receipt-v1','stage':stage,'run_id':value['run_ids'][stage],
        'source':value['source'],'runtime':value['runtime'],'request_sha256':digest,
        'run_manifest_sha256':sha(run/'run.json'),'verified':True,'engineering':False,'test_read':False,
        'training':stage=='train','scientific_verified':stage=='verify','status':'complete' if stage=='train' else 'verified'}
    if any(canonical(receipt.get(k))!=canonical(v) for k,v in semantics.items()):raise ValueError('original stage receipt semantics differ')
    if binding is not None and canonical(receipt.get('binding'))!=canonical(binding):raise ValueError('original stage input binding differs')
    backend.close_context(value,receipt['binding'])
    if stage=='train' and sha(Path(value['output_root'])/'manifest.json')!=receipt.get('native_manifest_sha256'):
        raise ValueError('original native trained manifest differs')
    dag.verify_producer(path,run,pin)
    return {'manifest':str(path),'run_dir':str(run),'pin':pin},receipt['binding']


def driver_identity(expected_sha,expected=None):
    path=Path(__file__).resolve()
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all'],text=True).strip():
        raise ValueError('clean fixed driver deployment required')
    relative=path.relative_to(ROOT).as_posix()
    raw=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+relative])
    identity={'commit':commit,'path':relative,'sha256':sha(path)}
    if raw!=path.read_bytes() or identity['sha256']!=expected_sha or (expected is not None and identity!=expected):
        raise ValueError('fixed independent driver source differs')
    return identity


def wait_preflight(args,base,backend,identity,deadline,*,now,sleep):
    """Wait only for the already claimed original PF; never submit or retry it."""
    while True:
        if now()>=deadline:raise TimeoutError('finite P chain deadline waiting for original preflight')
        identity()
        value=backend.read_transport(args.inputs,args.request,args.request_sha256)
        if canonical(value)!=canonical(base):raise ValueError('original preflight request changed while waiting')
        run_id=base['run_ids']['preflight'];run=Path(base['ops_root'])/'runs'/run_id
        state=queue_state(base['ops_root'],run_id)
        if state['state']!='claimed' or not (run/'run.json').exists():
            raise ValueError('original preflight must remain claimed with its original job')
        records=backend.stage_inputs(base,args.inputs,args.request,args.request_sha256,'preflight')
        expected=backend.job_contract(base,records,args.request,args.request_sha256,'preflight')
        compare_job(backend.strict_json((run/'run.json').read_bytes()),expected)
        done=terminal(run,run_id)
        with (args.evidence/'preflight-wait.jsonl').open('a') as stream:
            stream.write(json.dumps({'run_id':run_id,'claimed':True,'terminal_succeeded':done},sort_keys=True)+'\n')
        if done:return
        sleep(POLL_SECONDS)


def run_chain(args,backend,helper,dag,launcher,scheduler,*,now=time.monotonic,sleep=time.sleep,driver_check=None):
    deadline=now()+deadline_seconds(args.deadline_hours)
    base=backend.read_transport(args.inputs,args.request,args.request_sha256)
    if base['predecessor'] is not None or base['config']['seed'] not in (42,2024,2025):
        raise ValueError('original preflight request and registered single seed required')
    native=backend.native_for(Path(base['native_root']))
    expected={**native.default_config(),'seed':base['config']['seed']}
    if base['config']!=expected:raise ValueError('frozen original single-seed config differs')
    if base['job_contract']['train']['resources']['wall_seconds']!=96*3600:
        raise ValueError('registered train wall must remain 96 hours')
    from analysis.ur001_derive_cli import source_identity
    from analysis.new_instance_prepare import runtime
    def identity():
        if driver_check:driver_check()
        dag.check_consumer(source_identity(),base['source'],runtime(backend.record_path(base['lock'])),base['runtime'])
        for contract in base['job_contract'].values():
            if launcher._ops_source()!=contract['ops_source']:raise ValueError('fixed original OPS source differs')
    identity()
    if getattr(args,'wait_preflight',False):
        wait_preflight(args,base,backend,identity,deadline,now=now,sleep=sleep)
    predecessor,binding=complete(backend,helper,dag,base,args.inputs,args.request,args.request_sha256,'preflight')
    submitted=[];observations=[]
    for stage in ('train','verify'):
        if now()>=deadline:raise TimeoutError('finite P chain deadline')
        identity()
        run_id=base['run_ids'][stage];run=Path(base['ops_root'])/'runs'/run_id
        state=queue_state(base['ops_root'],run_id)
        originally_present=state['state']!='absent' or run.exists()
        request,digest,value=stage_request(base,predecessor,args.evidence,stage,state,run)
        plan=backend.plan(stage,args.inputs,request,digest)
        write_new(args.evidence/(stage+'-plan.json'),plan)
        # The original launcher checks current source/input/config bytes.
        spec,metadata=prepare_exact(plan,launcher)
        write_new(args.evidence/(stage+'-dry.json'),{'metadata':metadata,'spec':spec})
        # Recheck original queue state immediately before the only write.
        current=queue_state(base['ops_root'],run_id)
        if current['state']=='absent' and not run.exists():
            if originally_present:raise ValueError('observed original queue/run disappeared; no resubmit')
            identity()
            path=scheduler.submit(Path(base['ops_root']),spec)
            submitted.append(run_id)
            write_new(args.evidence/(stage+'-submit.json'),{'action':'written','queue_path':str(path),'claimed':False})
        while True:
            if now()>=deadline:raise TimeoutError('finite P chain deadline')
            identity()
            state=queue_state(base['ops_root'],run_id)
            if state['path'] is not None:
                queued=json.loads(state['path'].read_bytes());queued.pop('schema',None)
                if canonical(queued)!=canonical(spec):raise ValueError('original queued spec differs')
            if (run/'run.json').exists():
                compare_job(backend.strict_json((run/'run.json').read_bytes()),plan['expected_job'])
            if state['state']=='absent' and not run.exists():raise ValueError('original queue entry disappeared; no resubmit')
            # Queue written/pending is never reported as claimed.
            record={'stage':stage,'run_id':run_id,'queue':state['state'],'claimed':state['state']=='claimed'}
            if not observations or observations[-1]!=record:
                observations.append(record)
                with (args.evidence/'progress.jsonl').open('a') as stream:stream.write(json.dumps(record,sort_keys=True)+'\n')
            if terminal(run,run_id):
                if state['state']!='claimed':raise ValueError('terminal job lacks claimed queue confirmation')
                predecessor,binding=complete(backend,helper,dag,value,args.inputs,request,digest,stage,binding)
                break
            sleep(POLL_SECONDS)
    return {'status':'verified','stages':['train','verify'],'seed':base['config']['seed'],
            'submitted_run_ids':submitted,'observations':observations,'scientific_interpretation':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    p=sub.add_parser('run')
    for name in ('inputs','request','ops-code','evidence','consumer-code'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--request-sha256',required=True);p.add_argument('--deadline-hours',type=float,required=True)
    p.add_argument('--driver-sha256',required=True)
    p.add_argument('--wait-preflight',action='store_true',help='wait for the already claimed original PF before the unchanged acceptance checks')
    args=parser.parse_args(argv)
    if args.command=='dry':
        print(json.dumps({'status':'dry','submitted':False,'poll_seconds':POLL_SECONDS}));return 0
    deadline_seconds(args.deadline_hours)
    if os.getppid()!=1:raise ValueError('detached PPID=1 required')
    consumer=args.consumer_code
    if (not consumer.is_absolute() or '..' in consumer.parts or any(p.is_symlink() for p in (consumer,*consumer.parents))
            or consumer==ROOT or ROOT in consumer.parents or consumer in ROOT.parents):
        raise ValueError('independent absolute consumer deployment required')
    driver=driver_identity(args.driver_sha256)
    os.chdir(consumer)
    sys.path[:0]=[str(consumer),str(consumer/'src'),str(args.ops_code/'ops')]
    from analysis import p_full_training_backend as backend,p_full_engineering as helper,producer_dag_binding as dag
    from vkgr_ops import launcher,scheduler
    for path in (args.inputs,args.request,args.ops_code,args.evidence,args.consumer_code):backend.checked_path(str(path))
    if backend.ROOT!=consumer:raise ValueError('imported consumer deployment differs')
    value=backend.read_transport(args.inputs,args.request,args.request_sha256)
    for protected in (ROOT,consumer,Path(value['native_root']),Path(value['stage2']),Path(value['output_root']),Path(value['ops_root'])):
        if args.evidence==protected or protected in args.evidence.parents or args.evidence in protected.parents:
            raise ValueError('exclusive evidence outside source/input/output/OPS required')
    args.evidence.mkdir(parents=True,exist_ok=False)
    try:
        result=run_chain(args,backend,helper,dag,launcher,scheduler,driver_check=lambda:driver_identity(args.driver_sha256,driver))
        result.update(driver_source=driver,consumer_source=value['source'],deadline_hours=args.deadline_hours)
    except Exception as error:
        result={'status':'ALERT','error_type':type(error).__name__,'error':str(error),'retry':False,
                'driver_source':driver,'consumer_source':value['source'],'deadline_hours':args.deadline_hours}
        write_new(args.evidence/'summary.json',result);print(json.dumps(result,sort_keys=True));return 2
    write_new(args.evidence/'summary.json',result);print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
