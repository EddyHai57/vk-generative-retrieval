"""Explicit route-B subset producer; original health/SID scientific functions unchanged."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from analysis import sgns_gpu_native_b as b
from analysis.ur001_derive_cli import secure,digest,encoded
from analysis.p_query_sprint_ops import source
from vkgr import sgns_gpu_native as native
from vkgr.sid_coll_sgns import SGNSConfig,health_check,_array_digest,_sequence_digest,save_sgns_output,iter_positive_pairs
from vkgr.sid_tok_audit import feature_digest
from vkgr.sid_v2 import catalog_digest
ROOT=Path(__file__).resolve().parents[1]
MODULE='analysis.sgns_gpu_native_fit'
FROZEN={'src/vkgr/sgns_gpu_native.py':'dd2acadd8d38a224410ff21b3d1bc0df3d81b5a7d4ee72d31cedf46125e671c4',
        'src/vkgr/sid_coll_sgns.py':'2cc25d1997c12fe1e262e455e6644980186b5b9d0e00805014dee3148675876d',
        'analysis/sgns_gpu_native_b.py':'d9524bfbd9a6d67fedaf349b120a4005d1a90ba6a6e7998ed3e5150443329b4e'}
CONFIG=asdict(SGNSConfig())
CONTRACT={'loss_reduction':'sum','optimizer':'SGD','momentum':0,'weight_decay':0,'gradient_clipping':None,
    'frequency_power':.75,'subsampling':False,'negative_replacement':True,'negative_exclusions':['center','positive'],
    'negative_rng':'torch device Generator(seed), continuous across epochs; conditional inverse CDF',
    'positive_order':'user ascending; center ascending; context oldest to newest','distance_weighting':False,
    'watch_ratio_weighting':False,'early_stopping':False,'export_table':'center_only'}
exact=b.exact
matches=b.matches
progress=b.progress


def hexsha(value,n=64):
    if type(value)is not str or len(value)!=n or set(value)-set('0123456789abcdef'):raise ValueError('fixed lowercase SHA required')
    return value


def validate(value):
    keys={'schema','runner_source','bundle','bundle_sha256','lock','ops_root','output','preflight_run_id','fit_run_id','route_selected','benchmark_reference','compiled','resources'}
    if type(value)is not dict or set(value)!=keys or value['schema']!='sgns-gpu-native-fit-request-v1':raise ValueError('complete fixed fit request required')
    hexsha(value['runner_source'],40);hexsha(value['bundle_sha256'])
    for key in ('route_selected','compiled'):
        if type(value[key])is not bool:raise ValueError('explicit boolean route/options required')
    roots=[ROOT,*(secure(value[k]) for k in ('bundle','ops_root'))];secure(value['lock']);out=secure(value['output'])
    if any(out==root or root in out.parents or out in root.parents for root in roots):raise ValueError('exclusive output outside source/input/ops required')
    for key in ('preflight_run_id','fit_run_id'):
        rid=value[key]
        if type(rid)is not str or not rid.startswith('sgns-gpu-native-fit-') or set(rid)-set('abcdefghijklmnopqrstuvwxyz0123456789-'):raise ValueError('descriptive fixed fit run identity required')
    if value['fit_run_id']==value['preflight_run_id']:raise ValueError('distinct PF/fit identities required')
    resources=value['resources']
    if type(resources)is not dict or set(resources)!={'cpu_cores','ram_mib','vram_mib','disk_gib','wall_hours'} or any(type(v)is not int or v<=0 for v in resources.values()):raise ValueError('explicit positive integral resource declaration required')
    if resources['wall_hours']>12:raise ValueError('finite fit declaration at most 12 hours required')
    ref=value['benchmark_reference']
    if ref is not None:
        if type(ref)is not dict or set(ref)!={'root','request','request_sha256'}:raise ValueError('fixed benchmark reference required')
        root=secure(ref['root']);secure(ref['request']);hexsha(ref['request_sha256'])
        if out==root or root in out.parents or out in root.parents:raise ValueError('output must not overlap benchmark deployment')
    return value


def load(path,sha):
    raw=secure(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=hexsha(sha):raise ValueError('request bytes differ')
    return validate(json.loads(raw))


def ledger(value):
    if any(digest(ROOT/name)!=sha for name,sha in FROZEN.items()):raise ValueError('frozen original/native B source bytes differ')
    result=source(ROOT,value['runner_source'])
    paths=subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r','--name-only',value['runner_source']],text=True).splitlines()
    return {**result,'files':{name:digest(secure(ROOT/name)) for name in paths}}


def benchmark_files(value):
    ref=value['benchmark_reference']
    if ref is None:raise ValueError('verified final benchmark reference required')
    path=secure(ref['request']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['request_sha256']:raise ValueError('benchmark request SHA differs')
    old=json.loads(raw)
    # Validate using the old source root: B.validate also enforces output separation
    # against the active root, which is stronger than needed for this reference.
    b.validate(old)
    if any(not exact(old[k],value[k]) for k in ('bundle','bundle_sha256','lock','compiled')):raise ValueError('benchmark input/runtime/options differ')
    root=secure(ref['root']);source(root,old['runner_source'])
    output=secure(value['output'])
    for prior in (secure(old['output']),secure(old['ops_root']),secure(ref['request'])):
        if output==prior or prior in output.parents or output in prior.parents:raise ValueError('fit output must not overlap benchmark evidence')
    for member in ('analysis/sgns_gpu_native_b.py','src/vkgr/sgns_gpu_native.py','src/vkgr/sid_coll_sgns.py'):
        if digest(root/member)!=digest(ROOT/member):raise ValueError('benchmark/core bytes differ')
    paths=[path]
    for stage in ('preflight','benchmark'):
        run=secure(old['ops_root'])/'runs'/old[stage+'_run_id']
        terminal=json.loads((run/'terminal.json').read_bytes())
        if not matches(terminal,{'schema':'vkgr-terminal-v1','run_id':old[stage+'_run_id'],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}):raise ValueError('benchmark producer terminal/wall-cap differs')
        paths.extend(run/name for name in ('run.json','terminal.json','logs/stdout.log'))
        paths.append(secure(old['output'])/stage/'result.json')
    return old,[{'path':str(secure(p)),'sha256':digest(p)} for p in paths]


def benchmark_proof(value):
    old,before=benchmark_files(value);ref=value['benchmark_reference'];root=secure(ref['root'])
    # Invoke the already reviewed B verifier in its ORIGINAL deployment context.
    # Never rewrite B.ROOT, its argv, RNG, runtime, or producer audit.
    args=[sys.executable,'-B','-m','analysis.sgns_gpu_native_b','verify','--stage','benchmark',
          '--request',ref['request'],'--request-sha256',ref['request_sha256']]
    env={**os.environ,'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    checked=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True,check=False)
    if checked.returncode!=0:raise ValueError('original deployment benchmark verifier rejected')
    proof=json.loads(checked.stdout)
    if not matches(proof,{'status':'verified','readonly':True,'science_training':False}):raise ValueError('original benchmark verify receipt differs')
    receipt=json.loads((secure(old['output'])/'benchmark/result.json').read_bytes())
    b.check_measurement(receipt,old)
    if receipt.get('B_throughput_12h_diagnostic')is not True or receipt['full_fit_linear_projection_hours']>12:raise ValueError('route B exceeds frozen 12h diagnostic')
    expected={'run_id':old['benchmark_run_id'],'stdout_sha256':digest(secure(old['ops_root'])/'runs'/old['benchmark_run_id']/'logs/stdout.log'),
              'result_sha256':digest(secure(old['output'])/'benchmark/result.json')}
    if not exact(proof.get('producer'),expected) or not exact(benchmark_files(value)[1],before):raise ValueError('benchmark evidence closure changed')
    return {'schema':'sgns-native-fit-benchmark-proof-v1','reference':ref,'files':before,'verified':True,
            'producer':expected,'projection_hours':receipt['full_fit_linear_projection_hours']}


def fit_authority(value):
    if value.get('route_selected')is not True:raise ValueError('explicit final route_selected=true required before fit/vector export')
    return benchmark_proof(value)


@contextmanager
def epoch_progress(emit):
    original=native.NativeSGNS.epoch
    def observed(instance,**kwargs):
        result=original(instance,**kwargs)
        emit({key:result[key] for key in ('epoch','complete_epoch','positive_pairs','steps','processed_positive_pairs','summed_loss')})
        return result
    native.NativeSGNS.epoch=observed
    try:yield
    finally:native.NativeSGNS.epoch=original


def verify_audit(arrays,vectors,audit,*,device='cuda',compiled=True):
    catalog,histories,content=arrays
    if not isinstance(vectors,np.ndarray) or vectors.dtype!=np.float32 or vectors.shape!=(len(catalog),64) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors.astype('f8'),axis=1),1,rtol=0,atol=2e-6):raise ValueError('finite normalized catalog-ordered vectors required')
    pairs=0;participating=set()
    for center,positive in iter_positive_pairs(histories,5):
        pairs+=1;participating.add(center);participating.add(positive)
    per_steps=math.ceil(pairs/4096)
    expected={'schema':'sgns-gpu-native-b-audit-v1','config':CONFIG,'engineering':False,'device':device,'positive_pairs_per_epoch':pairs,
        'processed_positive_pairs':pairs*5,'steps':per_steps*5,'epochs_completed':5,'health':health_check(vectors,catalog,histories),
        'vectors_sha256':feature_digest(vectors),'catalog_sha256':catalog_digest(catalog),'train_sequence_sha256':_sequence_digest(histories),
        'content_sha256':feature_digest(content),'training_contract':CONTRACT,
        'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'compiled_step':compiled,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}}
    if not matches(audit,expected):raise ValueError('native B audit/config/count/runtime/health binding differs')
    generator=torch.Generator(device='cpu').manual_seed(42);initial=torch.empty((len(catalog),64),dtype=torch.float32)
    initial.uniform_(-.5/64,.5/64,generator=generator)
    if audit.get('input_initialization_sha256')!=_array_digest(initial.numpy()):raise ValueError('native B original initialization differs')
    hexsha(audit.get('output_table_sha256'))
    fallback=np.asarray([item not in participating for item in catalog]);count=int(fallback.sum())
    if not exact(audit.get('fallback_count'),count):raise ValueError('original content fallback count differs')
    if count:
        normalized=(content[fallback].astype('f8')/np.linalg.norm(content[fallback].astype('f8'),axis=1)[:,None]).astype('f4')
        if not np.array_equal(vectors[fallback],normalized):raise ValueError('original content fallback vectors differ')
    epochs=audit.get('epoch_audits')
    if type(epochs)is not list or len(epochs)!=5:raise ValueError('five complete native epochs required')
    for index,epoch in enumerate(epochs,1):
        wanted={'epoch':index,'complete_epoch':True,'positive_pairs':pairs,'steps':per_steps,'processed_positive_pairs':index*pairs}
        if not matches(epoch,wanted) or type(epoch.get('summed_loss'))not in (int,float) or (not np.isfinite(epoch['summed_loss']) or epoch['summed_loss']<0):raise ValueError('native epoch count/loss differs')
        # Reuse reviewed arithmetic/mass checks without inventing a statistical gate.
        for key,integer in (('negative_observed_bins',True),('negative_expected_bins',False)):
            bins=epoch.get(key)
            if type(bins)is not list or len(bins)!=64 or any(type(x)not in ((int,) if integer else (int,float)) or not np.isfinite(x) or x<0 for x in bins) or not np.isclose(sum(bins),pairs*5,rtol=1e-10,atol=1e-5):raise ValueError('native epoch negative mass differs')
        diag=epoch.get('negative_per_item_diagnostics',{})
        if not exact(diag.get('observed_total'),pairs*5) or type(diag.get('expected_total'))not in (int,float) or not np.isclose(diag['expected_total'],pairs*5,rtol=1e-10,atol=1e-5):raise ValueError('native per-item frequency totals differ')
    return {'schema':'sgns-gpu-native-fit-audit-proof-v1','verified':True,'backend':'B','health':expected['health'],
            'vectors_sha256':expected['vectors_sha256'],'epochs_completed':5,'four_gates_passed':False,
        'gradient_replay':False,'negative_trace_replay':False,'output_table_replay':False}


def rid(value,stage):return value['preflight_run_id' if stage=='preflight' else 'fit_run_id']
def argv(stage,path,sha):return [sys.executable,'-B','-m',MODULE,stage,'--request',str(path),'--request-sha256',sha,'--execute-on-new-instance']
def fixed_env():return {'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def job_env(value,stage):return {**fixed_env(),'VKGR_RUN_ID':rid(value,stage),'VKGR_PROGRESS_PATH':str(secure(value['ops_root'])/'runs'/rid(value,stage)/'progress.json')}
def resources(value):return {**{k:value['resources'][k] for k in ('cpu_cores','ram_mib','vram_mib','disk_gib')},'wall_seconds':float(value['resources']['wall_hours']*3600)}


def pins(value,path,stage):
    bundle=secure(value['bundle'])
    paths=[secure(path),secure(value['lock']),*(bundle/name for name in ('manifest.json','catalog.json','content.npy','histories.jsonl')),
        ROOT/'analysis/sgns_gpu_native_fit.py',ROOT/'analysis/sgns_gpu_native_b.py',ROOT/'src/vkgr/sgns_gpu_native.py',ROOT/'src/vkgr/sid_coll_sgns.py']
    if stage=='fit':
        paths.extend(Path(pin['path']) for pin in benchmark_files(value)[1])
        run=secure(value['ops_root'])/'runs'/value['preflight_run_id']
        paths.extend(run/name for name in ('run.json','terminal.json','logs/stdout.log'))
        paths.append(secure(value['output'])/'preflight/result.json')
    return [{'path':str(secure(p)),'sha256':digest(p)} for p in paths]


def job(value,path,sha,stage):
    run=secure(value['ops_root'])/'runs'/rid(value,stage);raw=(run/'run.json').read_bytes();record=json.loads(raw)
    expected={'schema':'vkgr-run-v1','run_id':rid(value,stage),'kind':'ops','cwd':str(ROOT),'argv':argv(stage,path,sha),'seed':42,
        'git':{'commit':value['runner_source'],'dirty':False,'dirty_files':[]},'config':{'path':str(path),'sha256':sha},'resources':resources(value),'env':job_env(value,stage)}
    if not matches(record,expected) or not exact([{'path':x['path'],'sha256':x['sha256']} for x in record.get('inputs',[])],pins(value,path,stage)):raise ValueError('own job/argv/env/source/config/input/resources binding differs')
    return hashlib.sha256(raw).hexdigest()


def accepted(value,path,sha,stage,own_source,bound):
    run=secure(value['ops_root'])/'runs'/rid(value,stage);out=secure(value['output'])/stage
    raw=(out/'result.json').read_bytes();receipt=json.loads(raw);terminal=json.loads((run/'terminal.json').read_bytes())
    lines=(run/'logs/stdout.log').read_bytes().splitlines();stdout=json.loads(next((x for x in reversed(lines) if x.strip()),b'{}'))
    expected={'schema':'gpu-sgns-subset-stage-v1','status':'verified','verified':True,'engineering':False,'authority':'4b0a8a1','stage':stage,
        'backend':'B','device':'cuda','source':own_source,'runtime':bound['runtime'],'input_binding':bound,'input_manifest_sha256':value['bundle_sha256'],
        'request_sha256':sha,'run_id':rid(value,stage),'run_manifest_sha256':job(value,path,sha,stage),'config':CONFIG,
        'subset_four_gate_acceptance':False,'validation_read':False,'test_read':False}
    if not matches(terminal,{'schema':'vkgr-terminal-v1','run_id':rid(value,stage),'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}) or not exact(stdout,receipt) or not matches(receipt,expected):raise ValueError('successful bound stage terminal/stdout required')
    if stage=='preflight':
        if not matches(receipt,{'real_readonly_preflight':True,'temporary_parameter_updates':False,'vectors_saved':False,'native_epoch_positive_pairs':34770216}):raise ValueError('real readonly native PF required')
    else:
        prior=accepted(value,path,sha,'preflight',own_source,bound)
        if not exact(receipt.get('preflight_producer'),prior) or not exact(receipt.get('benchmark_producer'),fit_authority(value)):raise ValueError('fit PF/benchmark producer differs')
        if set(receipt.get('artifacts',{}))!={'vectors.npy','audit.json'} or receipt.get('vectors_saved')is not True:raise ValueError('exact native fit artifacts required')
    return {'run_id':rid(value,stage),'receipt_sha256':hashlib.sha256(raw).hexdigest(),'stdout_sha256':digest(run/'logs/stdout.log'),
        'run_manifest_sha256':digest(run/'run.json'),'terminal_sha256':digest(run/'terminal.json')}


def consume(path,sha):
    value=load(path,sha);own_source=ledger(value);arrays,bound=b.inputs(value)
    proof=accepted(value,path,sha,'fit',own_source,bound);out=secure(value['output'])/'fit';receipt=json.loads((out/'result.json').read_bytes())
    snapshots={name:(out/name).read_bytes() for name in ('vectors.npy','audit.json')}
    if any(hashlib.sha256(raw).hexdigest()!=receipt['artifacts'][name] for name,raw in snapshots.items()):raise ValueError('persisted fit artifact bytes differ')
    vectors=np.load(io.BytesIO(snapshots['vectors.npy']),allow_pickle=False);audit=json.loads(snapshots['audit.json'])
    checked=verify_audit(arrays,vectors,audit,compiled=value['compiled'])
    if not exact(receipt.get('native_audit_verification'),checked):raise ValueError('persisted native audit verification differs')
    from vkgr.sgns_subset_sid import validate_input
    ids=np.asarray(arrays[0],dtype=np.uint32)
    im={'schema':'coll-input-v1','status':'accepted','variant':'C1','dimension':64,'engineering':False,'catalog_sha256':catalog_digest(ids),
        'vectors_sha256':feature_digest(vectors),'content_feature_sha256':feature_digest(arrays[2]),'train_sequence_sha256':_sequence_digest(arrays[1]),
        'script_sha256':own_source['files']['analysis/sgns_gpu_native_fit.py'],'commit':own_source['commit'],'checkpoint_sha256':None,
        'source_files':own_source['files'],'sgns_stage_receipt_sha256':proof['receipt_sha256'],'sgns_device':'cuda','sgns_original_job':proof,
        'input_manifest_file_sha256':value['bundle_sha256'],'sgns_backend':'B'}
    validate_input(ids,vectors,im,engineering=False)
    if not exact(ledger(value),own_source) or not exact(b.inputs(value)[1],bound) or not exact(load(path,sha),value) or not exact(accepted(value,path,sha,'fit',own_source,bound),proof):raise ValueError('consumer source/input/request/producer closure changed')
    return {'ids':ids,'vectors':vectors,'input_manifest':im,'sgns_receipt':receipt,'source_delta':{},'job_proof':proof}


def execute(stage,path,sha):
    value=load(path,sha);authority=fit_authority(value) if stage=='fit' else None
    own_source=ledger(value);own=job(value,path,sha,stage);before=pins(value,path,stage)
    if not exact({k:os.environ.get(k) for k in job_env(value,stage)},job_env(value,stage)):raise ValueError('fixed own process environment required')
    from analysis.ur001_derive_cli import guards
    guards(secure(value['output']).parent,projected_memory=8*1024**3)
    out=secure(value['output'])/stage
    if out.exists():raise FileExistsError('exclusive new stage output required')
    arrays,bound=b.inputs(value);catalog,histories,content=arrays
    prior=accepted(value,path,sha,'preflight',own_source,bound) if stage=='fit' else None
    progress('stage_begin',stage=stage)
    if stage=='preflight':
        engine=native.NativeSGNS(catalog,histories,device='cuda',compile_step=False)
        if engine.pairs_per_epoch!=34770216:raise ValueError('fixed subset positive pair count differs')
        extra={'real_readonly_preflight':True,'temporary_parameter_updates':False,'vectors_saved':False,'native_epoch_positive_pairs':engine.pairs_per_epoch}
    else:
        with epoch_progress(lambda event:progress('epoch_complete',**event)):
            vectors,audit=native.fit_sgns(catalog,histories,content,device='cuda',compile_step=value['compiled'])
        checked=verify_audit(arrays,vectors,audit,compiled=value['compiled'])
        if audit['positive_pairs_per_epoch']!=34770216:raise ValueError('fixed subset positive pair count differs')
        # Persist only after actual vectors and the native audit have passed.
        save_sgns_output(out,vectors,audit)
        snapshots={name:(out/name).read_bytes() for name in ('vectors.npy','audit.json')}
        checked_disk=verify_audit(arrays,np.load(io.BytesIO(snapshots['vectors.npy']),allow_pickle=False),json.loads(snapshots['audit.json']),compiled=value['compiled'])
        if not exact(checked_disk,checked):raise ValueError('persisted output verification differs')
        extra={'vectors_saved':True,'artifacts':{name:hashlib.sha256(raw).hexdigest() for name,raw in snapshots.items()},'native_audit_verification':checked}
    if not exact(ledger(value),own_source) or not exact(b.inputs(value)[1],bound) or not exact(pins(value,path,stage),before) or job(value,path,sha,stage)!=own or not exact(load(path,sha),value):raise ValueError('source/runtime/input/request/job closure changed')
    if stage=='fit' and (not exact(fit_authority(value),authority) or not exact(accepted(value,path,sha,'preflight',own_source,bound),prior)):raise ValueError('upstream producer closure changed')
    result={'schema':'gpu-sgns-subset-stage-v1','status':'verified','verified':True,'engineering':False,'authority':'4b0a8a1','stage':stage,
        'backend':'B','device':'cuda','source':own_source,'runtime':bound['runtime'],'input_binding':bound,'input_manifest_sha256':value['bundle_sha256'],
        'request_sha256':sha,'run_id':rid(value,stage),'run_manifest_sha256':own,'config':CONFIG,'preflight_producer':prior,'benchmark_producer':authority,
        'resources':{**value['resources'],'measurement':'not_measured conservative first declaration'},'subset_four_gate_acceptance':False,
        'validation_read':False,'test_read':False,**extra}
    if stage=='preflight':out.mkdir(parents=True,exist_ok=False)
    with (out/'result.json').open('xb') as stream:stream.write(encoded(result))
    progress('stage_complete',stage=stage)
    return result


def prepare(stage,path,sha,ops_code):
    from analysis.a40_ops import load_ops
    value=load(path,sha);ledger(value)
    if stage=='fit':fit_authority(value)
    ops=secure(value['ops_root']);run_id=rid(value,stage)
    if (ops/'runs'/run_id).exists() or (secure(value['output'])/stage).exists():raise FileExistsError('run/output collision refused')
    for state in ('pending','claimed','rejected'):
        if list((ops/'queue'/state).glob('*-'+run_id+'.json')):raise FileExistsError('queue collision refused')
    _,launcher,_,_=load_ops(ops_code)
    args=['--root',str(ops),'--run-id',run_id,'--kind','ops','--cwd',str(ROOT),'--config',str(path),'--seed','42']
    for key,val in value['resources'].items():args+=['--'+key.replace('_','-'),str(val)]
    for key,val in fixed_env().items():args+=['--env',key+'='+val]
    for pin in pins(value,path,stage):args+=['--input',pin['path']]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*args,'--',*argv(stage,path,sha)]));metadata=launcher.prepare(spec)
    recorded=launcher._recorded_env(metadata['env'],launcher._job_env(metadata['env'],ops/'runs'/run_id,run_id))
    if not exact(recorded,job_env(value,stage)):raise ValueError('unexpected inherited launcher environment')
    return {'status':'prepared_dry','spec':spec,'metadata':metadata,'submitted':False,'training':False,'real_preflight':False}


def main(argv_in=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    for name in ('plan','preflight','fit','verify'):
        p=sub.add_parser(name);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True)
        if name in ('plan','verify'):p.add_argument('--stage',choices=('preflight','fit'),required=True)
        if name=='plan':p.add_argument('--ops-code',type=Path,required=True)
        elif name!='verify':p.add_argument('--execute-on-new-instance',action='store_true',required=True)
    args=parser.parse_args(argv_in)
    try:
        if args.command=='dry':result={'status':'dry','training':False,'submitted':False,'backend':'B','config':CONFIG,'real_preflight':False,'route_selected':False}
        elif args.command=='plan':result=prepare(args.stage,args.request,args.request_sha256,args.ops_code)
        elif args.command=='verify':
            if args.stage=='fit':result={'status':'verified','readonly':True,'producer':consume(args.request,args.request_sha256)['job_proof'],'four_gates_passed':False}
            else:
                value=load(args.request,args.request_sha256);own_source=ledger(value);_,bound=b.inputs(value)
                proof=accepted(value,args.request,args.request_sha256,'preflight',own_source,bound)
                if not exact(ledger(value),own_source) or not exact(b.inputs(value)[1],bound) or not exact(load(args.request,args.request_sha256),value) or not exact(accepted(value,args.request,args.request_sha256,'preflight',own_source,bound),proof):raise ValueError('PF verify closure changed')
                result={'status':'verified','readonly':True,'producer':proof,'four_gates_passed':False}
        else:result=execute(args.command,args.request,args.request_sha256)
        print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True);return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc),'training_may_have_started':args.command=='fit'},sort_keys=True),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
