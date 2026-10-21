"""Closed old-source PF bridge and one exact, non-training pending RAM change."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

from analysis.sgns_subset_acceptance import absolute,record_bytes,encoded,sha,pin,source_tree

ROOT=Path(__file__).resolve().parents[1]
PINS_SHA='01ada2d0712c80346bedb423c0bfd4f8f09400fa1c6f0b51cadc82c0311c6389'
REQUEST_KEYS={'schema','source','old_code_root','preflight_run','preflight_argv','resource_plan','apply_output'}


def load_pins():
    return json.loads(record_bytes({'path':str(ROOT/'configs/full-gpu-sgns/source-bridge-pins.json'),'sha256':PINS_SHA}))


def source_delta(actual,expected,pins):
    if actual!=expected:raise ValueError('current source differs from complete request ledger')
    pin(actual['commit'],40);old=pins['old_source']['files'];files=actual['files']
    changed=set(pins['allowed_changed_paths']);added=set(pins['added_paths'])
    if (set(files)!=set(old)|added or set(old)&added or not changed<=set(old)
            or {p for p in old if files.get(p)!=old[p]}!=changed):
        raise ValueError('closed source bridge changed/additional paths differ')
    for name,digest in files.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('relative source member required')
        pin(digest)
    return {'old_commit':pins['old_source']['commit'],'old_ledger_sha256':sha(encoded(old)),
        'changed':{p:{'before':old[p],'after':files[p]} for p in sorted(changed)},
        'added':{p:files[p] for p in sorted(added)}}


def resource_override(records):
    pins=load_pins()
    if type(records) is not dict or set(records)!={'plan','before','after'}:raise ValueError('exact override records required')
    raw={}
    for name,record in records.items():
        if record.get('sha256')!=pins[name+'_sha256']:raise ValueError('only original fixed resource plan bytes allowed')
        raw[name]=record_bytes(record)
    plan,before,after=(json.loads(raw[n]) for n in ('plan','before','after'))
    if (plan.get('schema')!='vkgr-full-input-pf-resource-override-v1' or plan.get('authority')!='3c4cd42'
            or plan.get('run_id')!=pins['run_id'] or plan.get('source_commit')!=pins['old_source']['commit']
            or plan.get('before_sha256')!=sha(raw['before']) or plan.get('after_sha256')!=sha(raw['after'])
            or plan.get('before_bytes')!=len(raw['before']) or plan.get('after_bytes')!=len(raw['after'])
            or plan.get('allowed_change')!={'ram_mib':{'before':pins['old_ram_mib'],'after':pins['new_ram_mib']}}
            or before.get('schema')!='vkgr-queue-job-v1' or before.get('run_id')!=pins['run_id']
            or before.get('ram_mib')!=pins['old_ram_mib'] or after.get('ram_mib')!=pins['new_ram_mib']
            or {k:v for k,v in before.items() if k!='ram_mib'}!={k:v for k,v in after.items() if k!='ram_mib'}
            or set(before)!=set(after) or before.get('cwd')!=plan.get('code_root')):
        raise ValueError('exact sole RAM override contract differs')
    absolute(plan['queue_path']);absolute(plan['code_root'])
    return {'plan':plan,'before':before,'after':after,'raw':raw}


def read_request(path,digest,current):
    value=json.loads(record_bytes({'path':str(path),'sha256':digest}));pins=load_pins()
    if type(value) is not dict or set(value)!=REQUEST_KEYS or value.get('schema')!='gpu-sgns-full-input-source-bridge-request-v1':
        raise ValueError('complete fixed source bridge request required')
    delta=source_delta(current,value['source'],pins)
    change=resource_override(value['resource_plan']);plan=change['plan']
    queue=absolute(plan['queue_path']);ops=queue.parents[2]
    if (absolute(value['old_code_root'])!=absolute(plan['code_root'])
            or absolute(value['preflight_run'])!=ops/'runs'/pins['run_id']
            or value['preflight_argv']!=change['before']['argv']):
        raise ValueError('old PF source/root/run/argv differs from fixed plan')
    source_tree(value['old_code_root'],pins['old_source'])
    backup=absolute(value['apply_output'])
    for protected in (ROOT,absolute(value['old_code_root']),ops/'queue',ops/'runs'):
        if backup==protected or backup in protected.parents or protected in backup.parents:
            raise ValueError('exclusive apply backup separate from code/queue/runs required')
    return value,change,delta


def bridge_inputs(path,digest):
    # Launcher dry hashes request and exact plan records; no accepted PF or data
    # readiness is inferred from this transport-only metadata path.
    value=json.loads(record_bytes({'path':str(path),'sha256':digest}))
    if type(value) is not dict or set(value)!=REQUEST_KEYS or value.get('schema')!='gpu-sgns-full-input-source-bridge-request-v1':
        raise ValueError('complete fixed bridge transport request required')
    resource_override(value['resource_plan'])
    return [{'path':str(path),'sha256':digest},*value['resource_plan'].values()]


def old_contract(value,change,args):
    from analysis.sgns_gpu_full import file_record,fixed_dag
    pins=load_pins();spec=change['after'];argv=spec['argv'];names=('data','embedding-root','stage2','stage2-run','lock','output','deployment-sha')
    if (len(argv)!=19 or argv[1:5]!=['-B','-m','analysis.sgns_gpu_full','preflight']
            or argv[5:19:2]!=['--'+n for n in names]):raise ValueError('fixed original PF argv form differs')
    options=dict(zip(names,argv[6:19:2]))
    for name in names[:-1]:
        if options[name]!=str(getattr(args,name.replace('-','_'))):raise ValueError('old PF/new prepare scientific path differs')
    if options['deployment-sha']!=pins['old_source']['commit']:raise ValueError('fixed old deployment differs')
    code=absolute(value['old_code_root']);run=absolute(value['preflight_run']);dag=fixed_dag()
    inputs=[file_record(Path(options['lock']),dag['runtime']['lock_sha256']),
            file_record(Path(options['stage2'])/'UR001-manifest.json',dag['stage2_manifest_sha256'])]
    original_env={'PYTHONPATH':str(code/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    if (spec.get('cwd')!=str(code) or spec.get('config')!=str(code/'configs/full-gpu-sgns/fixed-dag.json')
            or spec.get('inputs')!=[r['path'] for r in inputs] or spec.get('env')!=[k+'='+v for k,v in original_env.items()]
            or spec.get('kind')!='ops' or type(spec.get('seed')) is not int or spec['seed']!=42
            or spec.get('cpu_cores')!=1 or spec.get('ram_mib')!=pins['new_ram_mib'] or spec.get('vram_mib')!=0
            or spec.get('disk_gib')!=16 or spec.get('wall_hours')!=24 or spec.get('stop_mode')!='signal'
            or spec.get('no_admission') is not False or spec.get('pinned_manifest') is not None or spec.get('pinned_files')!=[]):
        raise ValueError('fixed original queue spec differs')
    return {'schema':'vkgr-run-v1','run_id':pins['run_id'],'kind':'ops','cwd':str(code),'argv':argv,'seed':42,
        'git':{'commit':pins['old_source']['commit'],'dirty':False,'dirty_files':[]},
        'config':{'path':spec['config'],'sha256':pins['old_source']['files']['configs/full-gpu-sgns/fixed-dag.json']},
        'inputs':inputs,'env':{**original_env,'VKGR_RUN_ID':pins['run_id'],'VKGR_PROGRESS_PATH':str(run/'progress.json')},
        'resources':{'cpu_cores':1,'ram_mib':pins['new_ram_mib'],'vram_mib':0,'disk_gib':16,'wall_seconds':86400.0},
        'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158','sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},
        'overlay':None,'pinned_files':{},'stop_mode':'signal'}


def preflight(args,current):
    from analysis.sgns_gpu_full import producer,last_receipt,fixed_dag,fixed_records
    path=absolute(args.source_bridge_request);digest=args.source_bridge_sha256
    value,change,delta=read_request(path,digest,current);pins=load_pins()
    if str(args.preflight_run)!=value['preflight_run']:raise ValueError('original PF run differs')
    queue=absolute(change['plan']['queue_path']);claimed=queue.parent.parent/'claimed'/queue.name
    if queue.exists() or (queue.parent.parent/'rejected'/queue.name).exists() or claimed.read_bytes()!=change['raw']['after']:
        raise ValueError('exact overridden original claimed queue entry required')
    receipt=last_receipt(absolute(value['preflight_run'])/'logs/stdout.log')
    if (receipt.get('schema')!='gpu-sgns-full-input-v1' or receipt.get('status')!='verified'
            or receipt.get('verified') is not True or receipt.get('engineering') is not False
            or receipt.get('training') is not False or receipt.get('test_read') is not False
            or receipt.get('payload_written') is not False or receipt.get('source')!=pins['old_source']
            or receipt.get('runtime')!=fixed_dag()['runtime'] or receipt.get('records')!=fixed_records()):
        raise ValueError('original PF scientific identity/read-only status differs')
    contract=old_contract(value,change,args)
    job=producer(Path(value['preflight_run']),value['preflight_argv'],pins['old_source'],receipt,expected_job=contract)
    return {'request':{'path':str(path),'sha256':digest},'value':value,'change':change,'delta':delta,'receipt':receipt,
            'contract':contract,'job':job,'current':current}


def finish(context,current_receipt):
    from analysis.sgns_gpu_full import fixed_dag,producer
    pins=load_pins();expected={k:v for k,v in current_receipt.items() if k not in ('artifacts','preflight_job','prepare_argv','source_bridge')}
    expected.update(status='verified',verified=True,payload_written=False,source=pins['old_source'],
        source_delta={'base_commit':fixed_dag()['base_source']['commit'],'stage2_source':fixed_dag()['stage2_source'],
            'delta':{p:pins['old_source']['files'][p] for p in sorted(pins['old_delta_paths'])}})
    if encoded(context['receipt'])!=encoded(expected):raise ValueError('old PF/new prepare complete scientific receipt differs')
    source_tree(context['value']['old_code_root'],pins['old_source'])
    record_bytes(context['request'])
    value=context['value'];job=producer(Path(value['preflight_run']),value['preflight_argv'],pins['old_source'],context['receipt'],expected_job=context['contract'])
    if job!=context['job']:raise ValueError('original PF bytes changed during new prepare')
    change=resource_override(value['resource_plan'])
    claimed=absolute(change['plan']['queue_path']).parent.parent/'claimed'/Path(change['plan']['queue_path']).name
    queue=absolute(change['plan']['queue_path'])
    if queue.exists() or (queue.parent.parent/'rejected'/queue.name).exists() or claimed.read_bytes()!=change['raw']['after']:
        raise ValueError('claimed resource override changed')
    return {'request':context['request'],'source_delta':context['delta'],'preflight_job':job,
        'resource_plan_sha256':pins['plan_sha256'],'claimed_queue_sha256':pins['after_sha256']}


def verify_manifest(value,current):
    argv=value.get('prepare_argv',[])
    import sys
    if (len(argv)!=25 or argv[:5]!=[sys.executable,'-B','-m','analysis.sgns_gpu_full','prepare']
            or argv[5:19:2]!=['--'+n for n in ('data','embedding-root','stage2','stage2-run','lock','output','deployment-sha')]
            or argv[19:23:2]!=['--source-bridge-request','--source-bridge-sha256']
            or argv[23]!='--preflight-run'):raise ValueError('bridged prepare argv required')
    args=SimpleNamespace(**dict(zip(('data','embedding_root','stage2','stage2_run','lock','output','deployment_sha'),argv[6:19:2])),
        source_bridge_request=Path(argv[20]),source_bridge_sha256=argv[22],preflight_run=Path(argv[24]))
    if args.deployment_sha!=current['commit']:raise ValueError('bridge prepare deployment differs')
    proof=finish(preflight(args,current),value)
    if value.get('source_bridge')!=proof or value.get('preflight_job')!=proof['preflight_job']:
        raise ValueError('bundle original PF source bridge proof differs')
    return proof


def replace_pending(queue,run_id,before,after,plan,backup):
    queue=absolute(queue);backup=absolute(backup);ops=queue.parents[2]
    if not run_id or Path(run_id).name!=run_id or run_id in ('.','..'):
        raise ValueError('single run ID required')
    if queue.parent!=ops/'queue/pending':raise ValueError('original pending directory required')
    with (ops/'scheduler.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def unchanged():
            if ((ops/'runs'/run_id).exists() or list((ops/'queue/claimed').glob('*-'+run_id+'.json'))
                    or list((ops/'queue/rejected').glob('*-'+run_id+'.json')) or queue.read_bytes()!=before):
                raise ValueError('original pending SHA/state changed; STOP')
        unchanged()
        backup.mkdir(parents=True,exist_ok=False)
        for name,raw in (('before.json',before),('after.json',after),('plan.json',plan)):
            with (backup/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        for directory in (backup,backup.parent):
            dfd=os.open(directory,os.O_RDONLY)
            try:os.fsync(dfd)
            finally:os.close(dfd)
        temp=queue.with_name(queue.name+'.resource-'+sha(plan)[:16]+'.tmp')
        fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as f:f.write(after);f.flush();os.fsync(f.fileno())
        unchanged();os.replace(temp,queue)
        dfd=os.open(queue.parent,os.O_RDONLY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
        if queue.read_bytes()!=after:raise ValueError('pending replacement bytes differ; inspect retained backup')
        return {'status':'pending_resource_replaced','queue_modified':True,'before_sha256':sha(before),'after_sha256':sha(after),
            'plan_sha256':sha(plan),'run_id':run_id,'backup':str(backup),'submitted':False,'requeued':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    for name in ('inspect','apply'):
        c=sub.add_parser(name);c.add_argument('--request',type=Path,required=True);c.add_argument('--request-sha256',required=True)
    args=p.parse_args(argv)
    if args.command=='dry':result={'status':'dry','queue_modified':False,'training':False,'source_bridge_ready':False}
    else:
        from analysis.ur001_derive_cli import source_identity
        from analysis.sgns_gpu_full import check_source
        current=source_identity();value,change,delta=read_request(args.request,args.request_sha256,current);check_source(current,current['commit'])
        if args.command=='inspect':result={'status':'bridge_inputs_verified','queue_modified':False,'source_delta':delta,'resource_plan_sha256':sha(change['raw']['plan'])}
        else:result=replace_pending(Path(change['plan']['queue_path']),change['plan']['run_id'],change['raw']['before'],change['raw']['after'],change['raw']['plan'],Path(value['apply_output']))
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
