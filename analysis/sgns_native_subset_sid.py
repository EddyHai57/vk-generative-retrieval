"""Native B producer transport into unchanged subset BHKM48 scientific bytecode."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType,SimpleNamespace
from analysis import sgns_subset_sid as legacy
from analysis import sgns_gpu_native_fit as producer
from analysis.ur001_derive_cli import secure,digest,encoded,source_identity
from analysis.p_query_sprint_ops import source
from vkgr import sgns_subset_sid as science
ROOT=Path(__file__).resolve().parents[1]
MODULE='analysis.sgns_native_subset_sid'
STAGES=('preflight','fit','verify')
ADDITIONS=('analysis/sgns_native_subset_sid.py','tests/test_sgns_native_subset_sid.py','docs/superpowers/reviews/2026-10-22-sgns-native-subset-sid-implementation.md','analysis/sgns_subset_sid.py')
CALLBACKS={'load_request','input_stage','cpu_reference','own_job','successful_stage'}
FROZEN={'analysis/sgns_subset_sid.py':'bf824fceb8b95543dec46c7f717112e252fcba8806cd8fd2a569c3703b3c9dcb',
        'src/vkgr/sgns_subset_sid.py':'6ddc0169619e4d17a5e6749f06a239d5c4ff7217032f903c967e4efdf1a8da21'}
exact=producer.exact
matches=producer.matches


def validate(v):
    keys={'schema','source','lock','ops_root','ops_code','native_root','native_request','native_request_sha256','producer_delta','cpu_root','cpu_request','cpu_request_sha256','outputs','run_ids'}
    if type(v)is not dict or set(v)!=keys or v['schema']!='sgns-native-subset-sid-request-v1':raise ValueError('complete fixed native SID request required')
    for key in ('lock','ops_root','ops_code','native_root','native_request','cpu_root','cpu_request'):secure(v[key])
    for key in ('native_request_sha256','cpu_request_sha256'):producer.hexsha(v[key])
    if type(v['source'])is not dict or set(v['source'])!={'commit','files'} or type(v['source']['files'])is not dict:raise ValueError('complete source identity required')
    producer.hexsha(v['source']['commit'],40)
    for name,sha in v['source']['files'].items():
        if type(name)is not str or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('relative source member required')
        producer.hexsha(sha)
    if type(v['producer_delta'])is not dict or set(v['producer_delta'])!=set(ADDITIONS):raise ValueError('four exact pinned transport paths required')
    for sha in v['producer_delta'].values():producer.hexsha(sha)
    if type(v['outputs'])is not dict or set(v['outputs'])!=set(STAGES) or type(v['run_ids'])is not dict or set(v['run_ids'])!=set(STAGES):raise ValueError('complete distinct stage output/run roles required')
    outputs=[secure(v['outputs'][stage]) for stage in STAGES]
    roots=[ROOT,*(secure(v[k]) for k in ('native_root','cpu_root','ops_root','ops_code')),secure(v['lock']),secure(v['native_request']),secure(v['cpu_request'])]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(outputs) for b in outputs[i+1:]+roots):raise ValueError('exclusive external outputs required')
    for stage,rid in v['run_ids'].items():
        if type(rid)is not str or not rid.startswith('sgns-native-subset-sid-'+stage+'-') or set(rid)-set('abcdefghijklmnopqrstuvwxyz0123456789-'):raise ValueError('distinct descriptive native SID run identities required')
    return v


def load(path,sha):return validate(json.loads(science.read_pinned(path,producer.hexsha(sha))))


def check_delta(current,old,delta):
    if type(delta)is not dict or set(delta)!=set(ADDITIONS) or any(p in old.get('files',{}) for p in ADDITIONS):raise ValueError('closed new source increment required')
    if current.get('files')!={**old.get('files',{}),**delta} or any(current['files'].get(p)!=h for p,h in science.SCIENTIFIC_FILES.items()):raise ValueError('pre-existing bytes or pinned source delta differ')
    for sha in delta.values():producer.hexsha(sha)
    if delta['analysis/sgns_subset_sid.py']!=FROZEN['analysis/sgns_subset_sid.py']:raise ValueError('original SID transport bytes required')
    return delta


def clone(function,namespace):
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def facade(callbacks,delta_check):
    if type(callbacks)is not dict or set(callbacks)!=CALLBACKS:raise ValueError('closed transport callback set required')
    sn={**science.__dict__,'check_source_delta':delta_check}
    proxy=SimpleNamespace(**{**science.__dict__,'load_bundle':clone(science.load_bundle,sn)})
    return clone(legacy.execute,{**legacy.__dict__,**callbacks,'science':proxy})


def module_at(path,name):
    spec=importlib.util.spec_from_file_location(name,secure(path));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def native_context(v):
    root=secure(v['native_root']);entry=root/'analysis/sgns_gpu_native_fit.py'
    if digest(entry)!=digest(ROOT/'analysis/sgns_gpu_native_fit.py'):raise ValueError('native consumer code differs from producer')
    module=module_at(entry,'analysis._native_sid_producer_context')
    request=module.load(Path(v['native_request']),v['native_request_sha256'])
    old=module.ledger(request);check_delta(v['source'],old,v['producer_delta'])
    if secure(v['lock'])!=secure(request['lock']):raise ValueError('native/consumer lock differs')
    return module,request


def native_input(v,env):
    module,request=native_context(v);got=module.consume(Path(v['native_request']),v['native_request_sha256'])
    if not exact(got['sgns_receipt'].get('runtime'),json.loads(json.dumps(env))):raise ValueError('native runtime differs')
    if got['input_manifest'].get('sgns_backend')!='B':raise ValueError('explicit native B input required')
    for output in v['outputs'].values():
        out=secure(output)
        for root in (secure(request['bundle']),secure(request['output']),secure(request['ops_root'])):
            if out==root or root in out.parents or out in root.parents:raise ValueError('SID outputs overlap native evidence')
    return got


def cpu_reference(v,env):
    path=Path(v['cpu_request']);sha=v['cpu_request_sha256'];cpu=legacy.load_request(path,sha);root=secure(v['cpu_root'])
    source(root,cpu['source']['commit'])
    for out in (secure(value) for value in v['outputs'].values()):
        for evidence in (secure(cpu[role]) for role in ('bundle','sgns_output','sgns_run','fit_output','verify_output','ops_root')):
            if out==evidence or out in evidence.parents or evidence in out.parents:raise ValueError('SID output overlaps immutable CPU evidence')
    if cpu['device']!='cpu' or cpu['lock']!=v['lock']:raise ValueError('original CPU source/lock/device differs')
    module,native=native_context(v)
    if cpu['bundle']!=native['bundle'] or cpu['bundle_sha']!=native['bundle_sha256']:raise ValueError('original CPU/native input differs')
    for member,pin in FROZEN.items():
        if digest(root/member)!=pin:raise ValueError('original CPU SID transport bytes differ')
    old_science=module_at(root/'src/vkgr/sgns_subset_sid.py','vkgr._native_sid_cpu_reference')
    predecessors=((secure(cpu['sgns_run']),'sgns-gpu-subset-cpu-fit-v1'),(secure(cpu['ops_root'])/'runs'/legacy.run_id(cpu,'fit'),legacy.run_id(cpu,'fit')))
    for run,rid in predecessors:
        terminal=json.loads((run/'terminal.json').read_bytes())
        if not matches(terminal,{'schema':'vkgr-terminal-v1','run_id':rid,'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}):raise ValueError('CPU predecessor terminal/wall-cap differs')
    # Read successful original CPU verification and existing fitted bundle only.
    # No fit_bhkm_coll or same-seed mapping replay is invoked here.
    receipt,proof=legacy.successful_stage(cpu,'verify',path,sha)
    run=secure(cpu['ops_root'])/'runs'/legacy.run_id(cpu,'verify');terminal=json.loads((run/'terminal.json').read_bytes())
    if not matches(terminal,{'schema':'vkgr-terminal-v1','run_id':legacy.run_id(cpu,'verify'),'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}):raise ValueError('CPU terminal/wall-cap differs')
    if not exact(receipt.get('runtime'),json.loads(json.dumps(env))) or receipt.get('mapping_replay')is not True:raise ValueError('CPU original verified runtime/replay proof required')
    tracked=set(subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only','HEAD'],text=True).splitlines())
    if tracked!=set(cpu['source']['files']):raise ValueError('CPU full source ledger member set differs')
    for rel,pin in cpu['source']['files'].items():
        if digest(secure(root/rel))!=pin:raise ValueError('CPU original full source bytes differ')
    inputs=old_science.load_sgns_stage(Path(cpu['bundle']),cpu['bundle_sha'],Path(cpu['sgns_output']),cpu['sgns_receipt_sha'],
        Path(cpu['sgns_run']),'cpu',Path(cpu['lock']),cpu['source'],env)
    manifest=Path(cpu['fit_output'])/'run_manifest.json'
    loaded=old_science.load_bundle(manifest,receipt['manifest_sha256'],catalog_ids=inputs['ids'],vectors=inputs['vectors'],input_manifest=inputs['input_manifest'])
    if not exact(receipt.get('gates'),loaded.manifest['gates']) or loaded.manifest['gates'].get('passed')is not True:raise ValueError('CPU original gate closure differs')
    return loaded.manifest['audit'],{'request_sha256':sha,'verification_receipt_sha256':digest(Path(cpu['verify_output'])/'stage-receipt.json'),
        'manifest_sha256':receipt['manifest_sha256'],'original_verify_proof':proof,'cpu_sgns_proof':inputs['job_proof']}


def normalized(v,native):
    return {'source':v['source'],'lock':v['lock'],'ops_root':v['ops_root'],'device':'cuda',
        'sgns_receipt_sha':native['job_proof']['receipt_sha256'],**{stage+'_output':v['outputs'][stage] for stage in STAGES}}


def argv(stage,path,sha):return [sys.executable,'-B','-m',MODULE,stage,'--request',str(path),'--request-sha256',sha]
def fixed_env():return {'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def job_env(v,stage):
    rid=v['run_ids'][stage]
    return {**fixed_env(),'VKGR_RUN_ID':rid,'VKGR_PROGRESS_PATH':str(secure(v['ops_root'])/'runs'/rid/'progress.json')}


def resources(v,stage):
    ram=24576
    if stage=='verify':
        fit=json.loads((secure(v['outputs']['fit'])/'stage-receipt.json').read_bytes());measured=fit.get('rss_tree_upper_bound_mib')
        if type(measured)not in (int,float) or not math.isfinite(measured) or measured<=0:raise ValueError('measured original fit RSS required')
        ram=max(256,math.ceil(measured*1.15/256)*256)
    return {'cpu_cores':1,'ram_mib':ram,'vram_mib':0,'disk_gib':2,'wall_seconds':86400.}


def pins(v,path,stage):
    module,native=native_context(v);cpu=legacy.load_request(Path(v['cpu_request']),v['cpu_request_sha256'])
    paths=[secure(path),secure(v['lock']),secure(v['native_request']),secure(v['cpu_request']),ROOT/ADDITIONS[0],ROOT/'analysis/sgns_subset_sid.py',ROOT/'src/vkgr/sgns_subset_sid.py']
    bundle=secure(native['bundle']);paths.extend(bundle/name for name in ('manifest.json','catalog.json','content.npy','histories.jsonl'))
    output=secure(native['output'])/'fit';paths.extend(output/name for name in ('result.json','vectors.npy','audit.json'))
    run=secure(native['ops_root'])/'runs'/native['fit_run_id'];paths.extend(run/name for name in ('run.json','terminal.json','logs/stdout.log'))
    run=secure(cpu['ops_root'])/'runs'/legacy.run_id(cpu,'verify');paths.extend(run/name for name in ('run.json','terminal.json','logs/stdout.log'))
    paths.extend((secure(cpu['verify_output'])/'stage-receipt.json',secure(cpu['fit_output'])/'run_manifest.json'))
    if stage!='preflight':
        prior='preflight' if stage=='fit' else 'fit';run=secure(v['ops_root'])/'runs'/v['run_ids'][prior]
        paths.extend(run/name for name in ('run.json','terminal.json','logs/stdout.log'))
        paths.extend(secure(v['outputs'][prior])/name for name in ('stage-receipt.json','native-receipt.json'))
    return [{'path':str(secure(p)),'sha256':digest(p)} for p in paths]


def own_job(v,stage,path,sha):
    run=secure(v['ops_root'])/'runs'/v['run_ids'][stage];raw=(run/'run.json').read_bytes();record=json.loads(raw)
    expected={'schema':'vkgr-run-v1','run_id':v['run_ids'][stage],'kind':'ops','cwd':str(ROOT),'argv':argv(stage,path,sha),'seed':42,
        'git':{'commit':v['source']['commit'],'dirty':False,'dirty_files':[]},'config':{'path':str(path),'sha256':sha},'resources':resources(v,stage),'env':job_env(v,stage)}
    if not matches(record,expected) or not exact([{'path':x['path'],'sha256':x['sha256']} for x in record.get('inputs',[])],pins(v,path,stage)):raise ValueError('own native SID job binding differs')
    return {'run_id':v['run_ids'][stage],'run_manifest_sha256':hashlib.sha256(raw).hexdigest()}


def successful_stage(v,stage,path,sha):
    own=own_job(v,stage,path,sha);out=secure(v['outputs'][stage]);run=secure(v['ops_root'])/'runs'/v['run_ids'][stage]
    raw={name:(run/name).read_bytes() for name in ('run.json','terminal.json','logs/stdout.log')};terminal=json.loads(raw['terminal.json'])
    receipt_raw=(out/'stage-receipt.json').read_bytes();receipt=json.loads(receipt_raw);envelope_raw=(out/'native-receipt.json').read_bytes();envelope=json.loads(envelope_raw)
    stdout=json.loads(next((line for line in reversed(raw['logs/stdout.log'].splitlines()) if line.strip()),b'{}'))
    expected={'schema':'sgns-native-subset-sid-stage-v1','status':'verified','verified':True,'engineering':False,'backend':'B','authority':'4b0a8a1',
        'stage':stage,'source':v['source'],'request_sha256':sha,**own,'science_receipt':receipt,'science_receipt_sha256':hashlib.sha256(receipt_raw).hexdigest(),
        'subset_four_gate_acceptance':False,'test_read':False,'validation_read':False}
    module,native=native_context(v);native_run=secure(native['ops_root'])/'runs'/native['fit_run_id']
    native_proof={'run_id':native['fit_run_id'],'receipt_sha256':digest(secure(native['output'])/'fit/result.json'),
        'stdout_sha256':digest(native_run/'logs/stdout.log'),'run_manifest_sha256':digest(native_run/'run.json'),'terminal_sha256':digest(native_run/'terminal.json')}
    cp=Path(v['cpu_request']);cs=v['cpu_request_sha256'];cpu=legacy.load_request(cp,cs);cpu_receipt,cpu_proof=legacy.successful_stage(cpu,'verify',cp,cs)
    cpu_link={'request_sha256':cs,'verification_receipt_sha256':digest(secure(cpu['verify_output'])/'stage-receipt.json'),
        'manifest_sha256':cpu_receipt['manifest_sha256'],'original_verify_proof':cpu_proof,
        'cpu_sgns_proof':{name:digest(secure(cpu['sgns_run'])/name) for name in ('run.json','terminal.json','logs/stdout.log')}}
    expected.update({'native_request_sha256':v['native_request_sha256'],'native_producer':native_proof,'cpu_reference':cpu_link})
    native_receipt=json.loads((secure(native['output'])/'fit/result.json').read_bytes());audit=json.loads((secure(native['output'])/'fit/audit.json').read_bytes());ns=native_receipt['source']
    im={'schema':'coll-input-v1','status':'accepted','variant':'C1','dimension':64,'engineering':False,'catalog_sha256':audit['catalog_sha256'],
        'vectors_sha256':audit['vectors_sha256'],'content_feature_sha256':audit['content_sha256'],'train_sequence_sha256':audit['train_sequence_sha256'],
        'script_sha256':ns['files']['analysis/sgns_gpu_native_fit.py'],'commit':ns['commit'],'checkpoint_sha256':None,'source_files':ns['files'],
        'sgns_stage_receipt_sha256':native_proof['receipt_sha256'],'sgns_device':'cuda','sgns_original_job':native_proof,
        'input_manifest_file_sha256':native['bundle_sha256'],'sgns_backend':'B'}

    inner={'schema':'gpu-sgns-subset-sid-stage-v1','status':'verified','verified':True,'engineering':False,'stage':'sid-'+stage,'device':'cuda','source':v['source'],
        'request_sha256':sha,**own,'runtime':native_receipt['runtime'],'sgns_stage_receipt_sha256':native_proof['receipt_sha256'],
        'input_manifest_sha256':hashlib.sha256(science.canonical_bytes(im)).hexdigest(),'subset_four_gate_acceptance':False,'test_read':False,'validation_read':False}
    if stage=='preflight':inner.update({'payload_written':False,'mapping_replay':False,'manifest_sha256':None,'gates':None,'collision_gate':None})
    else:
        if receipt.get('mapping_replay')is not True or receipt.get('payload_written')is not True or receipt.get('gates',{}).get('passed')is not True or receipt.get('collision_gate',{}).get('passed')is not True:raise ValueError('original mapping replay and SID/collision gates required')
        if digest(secure(v['outputs']['fit'])/'run_manifest.json')!=producer.hexsha(receipt.get('manifest_sha256')):raise ValueError('original fitted SID manifest SHA differs')
    if not matches(terminal,{'schema':'vkgr-terminal-v1','run_id':v['run_ids'][stage],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}) or not exact(stdout,envelope) or not matches(envelope,expected) or not matches(receipt,inner):raise ValueError('native SID terminal/stdout/envelope closure differs')
    return receipt,{'receipt_sha256':hashlib.sha256(receipt_raw).hexdigest(),'native_receipt_sha256':hashlib.sha256(envelope_raw).hexdigest(),**{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}}


def checked_stage_bytes(raw,result):
    # The original writer serializes TorchVersion (a str subclass) as JSON str.
    # Compare its exact canonical bytes, preserving bool/int/float distinctions.
    if raw!=encoded(result):raise ValueError('original scientific stage bytes differ')
    return json.loads(raw)


def execute(stage,path,sha):
    from analysis.new_instance_prepare import runtime
    v=load(path,sha)
    if any(digest(ROOT/name)!=pin for name,pin in FROZEN.items()) or source_identity()!=v['source']:raise ValueError('original scientific/transport source bytes differ')
    env=runtime(secure(v['lock']));native=native_input(v,env);cpu=cpu_reference(v,env);own=own_job(v,stage,path,sha);before=pins(v,path,stage)
    if not exact({k:os.environ.get(k) for k in job_env(v,stage)},job_env(v,stage)):raise ValueError('fixed native SID process environment required')
    callbacks={'load_request':lambda *_:normalized(v,native),'input_stage':lambda *_:native_input(v,env),'cpu_reference':lambda *_:cpu,
        'own_job':lambda *_:own_job(v,stage,path,sha),'successful_stage':lambda _,prior,*__:successful_stage(v,prior,path,sha)}
    result=facade(callbacks,lambda current,old:check_delta(current,old,v['producer_delta']))(stage,path,sha)
    # Re-close actual producers and the request after the unchanged pipeline.
    if source_identity()!=v['source'] or not exact(load(path,sha),v) or not exact(pins(v,path,stage),before) or not exact(native_input(v,env)['job_proof'],native['job_proof']) or not exact(cpu_reference(v,env),cpu) or not exact(own_job(v,stage,path,sha),own):raise ValueError('native SID source/input/runtime/job closure changed')
    result_raw=(secure(v['outputs'][stage])/'stage-receipt.json').read_bytes()
    result=checked_stage_bytes(result_raw,result)
    envelope={'schema':'sgns-native-subset-sid-stage-v1','status':'verified','verified':True,'engineering':False,'backend':'B','authority':'4b0a8a1',
        'stage':stage,'source':v['source'],'request_sha256':sha,**own,'science_receipt':result,'science_receipt_sha256':hashlib.sha256(result_raw).hexdigest(),
        'native_producer':native['job_proof'],'native_request_sha256':v['native_request_sha256'],'cpu_reference':cpu[1],
        'subset_four_gate_acceptance':False,'test_read':False,'validation_read':False}
    science.write_new(secure(v['outputs'][stage])/'native-receipt.json',encoded(envelope));return envelope


def consume(path,sha):
    from analysis.new_instance_prepare import runtime
    v=load(path,sha)
    if source_identity()!=v['source']:raise ValueError('SID consumer requires original deployment context')
    env=runtime(secure(v['lock']));native=native_input(v,env);cpu=cpu_reference(v,env)
    verified,proof=successful_stage(v,'verify',path,sha);fit,fit_proof=successful_stage(v,'fit',path,sha)
    proxy=facade({name:(lambda *a:None) for name in CALLBACKS},lambda current,old:check_delta(current,old,v['producer_delta'])).__globals__['science']
    loaded=proxy.load_bundle(secure(v['outputs']['fit'])/'run_manifest.json',verified['manifest_sha256'],catalog_ids=native['ids'],vectors=native['vectors'],input_manifest=native['input_manifest'])
    collision=science.collision_gate(cpu[0],loaded.manifest['audit'])
    if collision.get('passed')is not True or not exact(collision,verified.get('collision_gate')) or not exact(collision,loaded.manifest.get('collision_gate')) or not exact(verified.get('gates'),loaded.manifest['gates']) or verified.get('mapping_replay')is not True or not exact(fit['manifest_sha256'],verified['manifest_sha256']):raise ValueError('native SID final original gates/collision/replay proof differs')
    envelope=json.loads((secure(v['outputs']['verify'])/'native-receipt.json').read_bytes())
    if not exact(envelope.get('native_producer'),native['job_proof']) or not exact(envelope.get('cpu_reference'),cpu[1]) or not exact(envelope.get('native_request_sha256'),v['native_request_sha256']):raise ValueError('native SID final producer linkage differs')
    if source_identity()!=v['source'] or not exact(load(path,sha),v) or not exact(successful_stage(v,'verify',path,sha)[1],proof):raise ValueError('SID consume closure changed')
    return {'loaded':loaded,'native_input':native,'verification':envelope,'job_proof':proof,'fit_job_proof':fit_proof,'cpu_reference':cpu[1]}


def prepare(stage,path,sha,ops_code):
    from analysis.a40_ops import load_ops
    v=load(path,sha)
    if source_identity()!=v['source'] or secure(ops_code)!=secure(v['ops_code']):raise ValueError('fixed native SID queue source/OPS differs')
    rid=v['run_ids'][stage];ops=secure(v['ops_root'])
    if (ops/'runs'/rid).exists() or secure(v['outputs'][stage]).exists():raise FileExistsError('existing run/output refused')
    for state in ('pending','claimed','rejected'):
        if list((ops/'queue'/state).glob('*-'+rid+'.json')):raise FileExistsError('queue collision refused')
    if stage!='preflight':successful_stage(v,'preflight' if stage=='fit' else 'fit',path,sha)
    _,launcher,_,_=load_ops(ops_code);resource=resources(v,stage)
    args=['--root',str(ops),'--run-id',rid,'--kind','ops','--cwd',str(ROOT),'--config',str(path),'--seed','42']
    for key in ('cpu_cores','ram_mib','vram_mib','disk_gib'):args+=['--'+key.replace('_','-'),str(resource[key])]
    args+=['--wall-hours','24']
    for key,val in fixed_env().items():args+=['--env',key+'='+val]
    for pin in pins(v,path,stage):args+=['--input',pin['path']]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*args,'--',*argv(stage,path,sha)]));metadata=launcher.prepare(spec)
    recorded=launcher._recorded_env(metadata['env'],launcher._job_env(metadata['env'],ops/'runs'/rid,rid))
    if not exact(recorded,job_env(v,stage)):raise ValueError('unexpected inherited launcher environment')
    return {'status':'prepared_dry','spec':spec,'metadata':metadata,'submitted':False,'training':False,'real_preflight':False}


def main(argv_in=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    for name in (*STAGES,'plan','consume'):
        p=sub.add_parser(name,allow_abbrev=False);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True)
        if name=='plan':p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--ops-code',type=Path,required=True)
    args=parser.parse_args(argv_in)
    try:
        if args.command=='dry':result={'status':'dry','training':False,'submitted':False,'config':science.CONFIG,'backend':'B','subset_four_gate_acceptance':False}
        elif args.command=='plan':result=prepare(args.stage,args.request,args.request_sha256,args.ops_code)
        elif args.command=='consume':result={'status':'verified','readonly':True,'backend':'B','producer':consume(args.request,args.request_sha256)['job_proof'],'subset_four_gate_acceptance':False}
        else:result=execute(args.command,args.request,args.request_sha256)
        print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True);return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc),'science_fit_may_have_started':args.command=='fit'},sort_keys=True),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
