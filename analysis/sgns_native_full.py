"""Native B full transport; original four gates consumed in their fixed A1 tree."""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import numpy as np
import torch
from analysis import sgns_gpu_full as original
from analysis import sgns_gpu_native_fit as subset
from analysis.ur001_derive_cli import secure,digest,encoded,source_identity
from vkgr import sgns_gpu_native as native
from vkgr.sid_coll_sgns import save_sgns_output
ROOT=Path(__file__).resolve().parents[1]
MODULE='analysis.sgns_native_full'
INPUT_SOURCE='951c680cc77ceaf17cd9fb91e294f1da193fa447'
BENCHMARK_SOURCE='5356ac240c9aa58828ad7bdf729a302ebd031289'
ACCEPTANCE_SOURCE='96bfb9e6fcb4081168cf4d39563f50c58ee3fbad'
ACCEPTANCE_ENTRY='analysis/sgns_native_subset_a1.py'
ACCEPTANCE_ENTRY_SHA='96c1e851e84d820607ef29cd7708da9fe5bab2aeff8492d86e9aa9cd9a3deb01'
OWN_PATHS={'analysis/sgns_native_full.py','tests/test_sgns_native_full.py','docs/superpowers/reviews/2026-10-22-sgns-native-full-implementation.md'}
NATIVE_ADDITIONS={
 'analysis/p_query_blocks.py':'f200cfac3002e00c87ffb9eb0640007f37bf83ee698b43a0aff176e52aae71e5',
 'analysis/p_query_sprint_ops.py':'a0251a8ef6fc2609c208274ee8a1f31e4722b50e15913fb19f41368728e85d2d',
 'analysis/sgns_gpu_native_b.py':'d9524bfbd9a6d67fedaf349b120a4005d1a90ba6a6e7998ed3e5150443329b4e',
 'analysis/sgns_gpu_native_fit.py':'98f11a0c351307bb48b9b151890e877894252eaeabb3e57dd74c90bd16151fe4',
 'analysis/sgns_speedup_profile.py':'3d46c76cc8328546c5ab21e3a12b999c1a22f4e815ac0880b7c26ec315bb6697',
 'src/vkgr/p_query_blocks.py':'ee08c4af86fee34d21eb633ff3995aac3253273cdebbcfda0f0aa9762fb520c5',
 'src/vkgr/sgns_gpu_native.py':'dd2acadd8d38a224410ff21b3d1bc0df3d81b5a7d4ee72d31cedf46125e671c4',
 'src/vkgr/sgns_subset_sid.py':'6ddc0169619e4d17a5e6749f06a239d5c4ff7217032f903c967e4efdf1a8da21'}
exact=subset.exact
matches=subset.matches


def validate(value):
    keys={'schema','runner_source','lock','input_producer','benchmark_reference','acceptance','source_delta','ops_root','output','preflight_run_id','fit_run_id','compiled','resources'}
    if type(value)is not dict or set(value)!=keys or value['schema']!='sgns-native-full-request-v1':raise ValueError('closed native full request required')
    subset.hexsha(value['runner_source'],40);secure(value['lock'])
    if type(value['compiled'])is not bool:raise ValueError('explicit compile option required')
    p=value['input_producer']
    if type(p)is not dict or set(p)!={'root','source','bundle','manifest_sha256','prepare_run'} or p['source']!=INPUT_SOURCE:raise ValueError('fixed 951c680 full input producer required')
    for k in ('root','bundle','prepare_run'):secure(p[k])
    subset.hexsha(p['manifest_sha256'])
    ref=value['benchmark_reference']
    if type(ref)is not dict or set(ref)!={'root','request','request_sha256'}:raise ValueError('original B benchmark reference required')
    secure(ref['root']);secure(ref['request']);subset.hexsha(ref['request_sha256'])
    gate=acceptance_reference(value['acceptance'])
    if type(value['source_delta'])is not dict or set(value['source_delta'])!=OWN_PATHS:raise ValueError('exact pinned own additions required')
    for sha in value['source_delta'].values():subset.hexsha(sha)
    res=value['resources']
    if type(res)is not dict or set(res)!={'cpu_cores','ram_mib','vram_mib','disk_gib','wall_hours'} or any(type(x)is not int or x<=0 for x in res.values()):raise ValueError('positive integral first resource declaration required')
    for key in ('preflight_run_id','fit_run_id'):
        rid=value[key]
        if type(rid)is not str or not rid.startswith('sgns-native-full-') or set(rid)-set('abcdefghijklmnopqrstuvwxyz0123456789-'):raise ValueError('descriptive fixed full run ID required')
    if value['fit_run_id']==value['preflight_run_id']:raise ValueError('distinct full stage identities required')
    out=secure(value['output'])
    protected_roots=(ROOT,secure(value['ops_root']),secure(p['root']),secure(p['bundle']),secure(p['prepare_run']),secure(ref['root']),secure(ref['request']),secure(value['lock']),secure(gate['root']),secure(gate['receipt']))
    scratch=secure(gate['scratch_root'])
    for protected in (*protected_roots,out):
        if scratch==protected or scratch in protected.parents or protected in scratch.parents:raise ValueError('acceptance scratch separate from code/input/ops/output/evidence required')
    for protected in protected_roots:
        if out==protected or out in protected.parents or protected in out.parents:raise ValueError('exclusive output separate from code/input/ops/evidence required')
    return value


def load(path,sha):
    raw=secure(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=subset.hexsha(sha):raise ValueError('request bytes differ')
    return validate(json.loads(raw))


def acceptance_reference(reference):
    if type(reference)is not dict or set(reference)!={'root','receipt','receipt_sha256','scratch_root'}:
        raise ValueError('complete fixed native B four-gate reference required')
    for name in ('root','receipt','scratch_root'):secure(reference[name])
    subset.hexsha(reference['receipt_sha256'])
    return reference


def pinned_tree(root,commit):
    """Enumerate every fixed Git blob; source_tree additionally rejects shadows."""
    from analysis.sgns_subset_acceptance import source_tree
    root=secure(root)
    raw=subprocess.check_output(['git','-C',str(root),'archive',commit],stderr=subprocess.PIPE)
    files={}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive:
            if member.isdir():continue
            if not member.isfile():raise ValueError('fixed code tree requires regular Git blobs')
            files[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    source={'commit':commit,'files':files}
    source_tree(root,source)
    return source


ACCEPTANCE_VERIFY='''import json,sys
from pathlib import Path
from analysis import sgns_native_subset_a1 as module
from analysis.sgns_subset_acceptance import module_origins
expected,path,pin,scratch=json.loads(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4])
if module.source_identity()!=expected:raise ValueError('fixed A1 consumer source differs')
result=module.consume(path,pin,scratch)
if module.source_identity()!=expected:raise ValueError('A1 source changed during consume')
module_origins(sys.modules,module.ROOT,expected)
print(json.dumps(result,sort_keys=True,allow_nan=False))
'''


def acceptance_records(value):
    ref=acceptance_reference(value['acceptance']);root=secure(ref['root'])
    source=pinned_tree(root,ACCEPTANCE_SOURCE)
    if source['files'].get(ACCEPTANCE_ENTRY)!=ACCEPTANCE_ENTRY_SHA:raise ValueError('fixed A1 consumer entry differs')
    path=secure(ref['receipt']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref['receipt_sha256']:raise ValueError('native B total receipt bytes differ')
    receipt=json.loads(raw)
    expected={'schema':'sgns-native-subset-four-gate-receipt-v1','authority':'4b0a8a1','backend':'B','status':'verified','verified':True,'subset_four_gate_acceptance':True,'engineering':False,'test_read':False,'training':False,'run_id':'sgns-native-subset-a1-accept-v1'}
    if not matches(receipt,expected) or not exact(receipt.get('source'),source):raise ValueError('fixed native B total acceptance required')
    from analysis.sgns_subset_acceptance import check_proof
    check_proof(receipt['proof'])
    req=receipt['request']
    if type(req)is not dict or set(req)!={'path','sha256'}:raise ValueError('original acceptance request pin required')
    request=secure(req['path']);request_raw=request.read_bytes()
    if hashlib.sha256(request_raw).hexdigest()!=subset.hexsha(req['sha256']):raise ValueError('acceptance request bytes differ')
    registered=json.loads(request_raw);run=secure(registered['ops_root'])/'runs'/receipt['run_id']
    records=[{'path':str(path),'sha256':ref['receipt_sha256']},{'path':str(request),'sha256':req['sha256']},
             *({'path':str(root/name),'sha256':pin} for name,pin in sorted(source['files'].items())),
             *({'path':str(secure(name)),'sha256':pin} for name,pin in sorted(receipt['proof']['evidence'].items())),
             *({'path':str(run/name),'sha256':digest(run/name)} for name in ('run.json','terminal.json','logs/stdout.log'))]
    for name,want in receipt['proof']['evidence'].items():
        if digest(secure(name))!=subset.hexsha(want):raise ValueError('original four-gate evidence bytes differ')
    return source,receipt,records


def require_acceptance(value,phase='check'):
    source,receipt,before=acceptance_records(value);ref=value['acceptance'];root=secure(ref['root'])
    env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    # The original consume API requires a fresh scratch on every read-only replay.
    scratch=secure(ref['scratch_root'])/(phase+'-'+str(time.time_ns()))
    args=[sys.executable,'-B','-c',ACCEPTANCE_VERIFY,json.dumps(source,sort_keys=True),ref['receipt'],ref['receipt_sha256'],str(scratch)]
    result=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True,check=False)
    if result.returncode!=0:raise ValueError('original fixed native B four-gate consumer rejected')
    got=json.loads(result.stdout)
    if (type(got)is not dict or set(got)!={'receipt_sha256','request_sha256','producer','proof'}
            or got['receipt_sha256']!=ref['receipt_sha256'] or got['request_sha256']!=receipt['request']['sha256']
            or not exact(got['proof'],receipt['proof']) or type(got['producer'])is not dict or set(got['producer'])!={'run.json','terminal.json','logs/stdout.log'}):
        raise ValueError('original four-gate consumer result differs')
    records={r['path']:r['sha256'] for r in before}
    request=json.loads(secure(receipt['request']['path']).read_bytes());run=secure(request['ops_root'])/'runs'/receipt['run_id']
    if any(got['producer'][name]!=records[str(run/name)] for name in got['producer']):raise ValueError('original four-gate producer bytes differ')
    runtime=original.runtime(secure(value['lock']))
    if encoded(runtime)!=encoded(receipt.get('runtime')):raise ValueError('full/A1 Python dependencies or runtime lock differ')
    if not exact(acceptance_records(value),(source,receipt,before)):raise ValueError('four-gate source/evidence changed during consume')
    return {'reference':ref,'source':source,'runtime':receipt['runtime'],'consumer':got}


def source_delta(actual,old,delta):
    if old.get('commit')!=INPUT_SOURCE or type(delta)is not dict or set(delta)!=OWN_PATHS:raise ValueError('fixed original source and own delta required')
    for sha in delta.values():subset.hexsha(sha)
    additions={**NATIVE_ADDITIONS,**delta}
    if set(old['files'])&set(additions) or not exact(actual.get('files'),{**old['files'],**additions}):raise ValueError('exact old ledger union pinned additions required; old byte changes forbidden')
    return {'base_commit':INPUT_SOURCE,'old_ledger_sha256':hashlib.sha256(encoded(old['files'])).hexdigest(),'added':additions}


OLD_VERIFY='''import json,sys
from pathlib import Path
from analysis import sgns_gpu_full as f
root,bundle,manifest,run,lock,expected=sys.argv[1:]
s=f.source_identity()
if s['commit']!=expected:raise ValueError('original producer source differs')
e=f.runtime(Path(lock))
a,m=f.read_bundle(Path(bundle),manifest,s,e)
p=f.producer(Path(run),m['prepare_argv'],s,m,expected_job=f.full_job_contract(Path(run),m['prepare_argv'],s))
print(json.dumps({'source':s,'runtime':e,'manifest':m,'prepare_job':p},sort_keys=True,allow_nan=False))
'''


def input_proof(value):
    p=value['input_producer'];root=secure(p['root'])
    # Validate exact original Git and filesystem before importing any old module.
    from analysis.sgns_subset_acceptance import source_tree
    names=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only',INPUT_SOURCE],text=True,stderr=subprocess.PIPE).splitlines()
    old_source={'commit':INPUT_SOURCE,'files':{name:hashlib.sha256(subprocess.check_output(['git','-C',str(root),'show',INPUT_SOURCE+':'+name])).hexdigest() for name in names}}
    source_tree(root,old_source)
    env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    args=[sys.executable,'-B','-c',OLD_VERIFY,str(root),p['bundle'],p['manifest_sha256'],p['prepare_run'],value['lock'],INPUT_SOURCE]
    checked=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True,check=False)
    if checked.returncode!=0:raise ValueError('original full producer context verification rejected')
    proof=json.loads(checked.stdout)
    if type(proof)is not dict or set(proof)!={'source','runtime','manifest','prepare_job'} or proof['source'].get('commit')!=INPUT_SOURCE:raise ValueError('original full context proof malformed')
    if not exact(proof['source'],old_source):raise ValueError('original producer ledger changed')
    source_tree(root,old_source)
    actual=source_identity()
    source_tree(ROOT,actual)
    if actual['commit']!=value['runner_source']:raise ValueError('fixed native full source differs')
    delta=source_delta(actual,proof['source'],value['source_delta'])
    # Original code files must be byte-identical to 951c680; no ROOT or source ledger rewriting.
    current_runtime=json.loads(json.dumps(original.runtime(secure(value['lock'])),allow_nan=False))
    if not exact(current_runtime,proof['runtime']) or not exact(proof['manifest'].get('runtime'),proof['runtime']):raise ValueError('original/full runtime lock dependencies differ')
    arrays,manifest=original.read_bundle(secure(p['bundle']),p['manifest_sha256'],proof['source'],proof['runtime'])
    if not exact(manifest,proof['manifest']):raise ValueError('full input snapshot changed')
    run=secure(p['prepare_run']);pins={str(run/n):digest(run/n) for n in ('run.json','terminal.json','logs/stdout.log')}
    return arrays,{'source':actual,'source_delta':delta,'original':proof,'prepare_pins':pins}


def benchmark_arguments(value):
    ref=value['benchmark_reference'];old=json.loads(secure(ref['request']).read_bytes())
    if old.get('runner_source')!=BENCHMARK_SOURCE:raise ValueError('original verified 535 benchmark source required')
    pinned_tree(ref['root'],BENCHMARK_SOURCE)
    if old.get('lock')!=value['lock'] or not exact(old.get('compiled'),value['compiled']):raise ValueError('native full lock/options differ from benchmark')
    return {'benchmark_reference':ref,'bundle':old['bundle'],'bundle_sha256':old['bundle_sha256'],'lock':value['lock'],'compiled':value['compiled'],'output':value['output']}


def benchmark_proof(value):return subset.benchmark_proof(benchmark_arguments(value))


def registered_wall(proof):
    hours=proof.get('projection_hours')
    if type(hours)not in (float,int) or not math.isfinite(hours) or hours<=0:raise ValueError('finite original benchmark projection required')
    return math.ceil(hours*1.5)


def check_projection(seconds,processed,total,wall):
    if type(seconds)not in (float,int) or not math.isfinite(seconds) or seconds<0 or type(processed)is not int or type(total)is not int or total<=0 or not 0<=processed<=total:raise ValueError('finite measured full progress required')
    if seconds<3*3600:return None
    if processed==0:raise RuntimeError('three-hour full projection unavailable; STOP')
    projected=seconds*total/processed/3600
    if projected>wall:raise RuntimeError('three-hour full projection exceeds registered wall; STOP')
    return projected


@contextmanager
def observer(wall,emit,*,every=64):
    if type(every)is not int or every<=0:raise ValueError('positive observer interval required')
    init=native.NativeSGNS.__init__;epoch=native.NativeSGNS.epoch;started=time.monotonic()
    def wrapped_init(engine,*args,**kwargs):
        init(engine,*args,**kwargs);step=engine.step;seen=0
        def observed_step(*a,**kw):
            nonlocal seen
            result=step(*a,**kw);seen+=1
            if seen%every==0:
                if engine.device=='cuda':torch.cuda.synchronize()
                elapsed=time.monotonic()-started;processed=int(engine.processed)
                projection=check_projection(elapsed,processed,engine.total,wall)
                emit({'event':'batch_progress','epoch':engine.epochs+1,'batches':seen,'processed_positive_pairs':processed,'total_positive_pairs':engine.total,'seconds':elapsed,'full_projection_hours':projection})
            return result
        engine.step=observed_step
    def wrapped_epoch(engine,*args,**kwargs):
        result=epoch(engine,*args,**kwargs);elapsed=time.monotonic()-started
        projection=check_projection(elapsed,int(engine.processed),engine.total,wall)
        emit({'event':'epoch_complete',**result,'seconds':elapsed,'full_projection_hours':projection});return result
    native.NativeSGNS.__init__=wrapped_init;native.NativeSGNS.epoch=wrapped_epoch
    try:yield
    finally:native.NativeSGNS.__init__=init;native.NativeSGNS.epoch=epoch


def emit_progress(event):
    subset.progress(event['event'],**{k:v for k,v in event.items() if k!='event'})


def verify_audit(arrays,vectors,audit,binding,*,device='cuda',compiled=True):
    from vkgr.sgns_full_verify import check_full_binding
    check_full_binding(arrays[0],arrays[2],binding,False)
    proof=subset.verify_audit(arrays,vectors,audit,device=device,compiled=compiled)
    return {**proof,'schema':'sgns-native-full-audit-proof-v1','original_full_binding':True}


def rid(value,stage):return value['preflight_run_id' if stage=='preflight' else 'fit_run_id']
def argv(stage,path,sha):return [sys.executable,'-B','-m',MODULE,stage,'--request',str(path),'--request-sha256',sha,'--execute-on-new-instance']
def fixed_env():return {'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def job_env(value,stage):return {**fixed_env(),'VKGR_RUN_ID':rid(value,stage),'VKGR_PROGRESS_PATH':str(secure(value['ops_root'])/'runs'/rid(value,stage)/'progress.json')}
def resources(value):return {**{k:value['resources'][k] for k in ('cpu_cores','ram_mib','vram_mib','disk_gib')},'wall_seconds':float(value['resources']['wall_hours']*3600)}


def pins(value,path,stage):
    p=value['input_producer'];bundle=secure(p['bundle']);run=secure(p['prepare_run'])
    paths=[secure(path),secure(value['lock']),*(bundle/n for n in ('manifest.json','catalog.json','content.npy','histories.jsonl')),*(run/n for n in ('run.json','terminal.json','logs/stdout.log')),
        *(ROOT/n for n in sorted({*NATIVE_ADDITIONS,*OWN_PATHS}))]
    if stage=='fit':
        prior=secure(value['ops_root'])/'runs'/value['preflight_run_id']
        paths.extend(prior/n for n in ('run.json','terminal.json','logs/stdout.log'));paths.append(secure(value['output'])/'preflight/result.json')
    records=[{'path':str(secure(p)),'sha256':digest(p)} for p in paths]
    records.extend(acceptance_records(value)[2]);records.extend(subset.benchmark_files(benchmark_arguments(value))[1])
    benchmark_root=secure(value['benchmark_reference']['root'])
    records.extend({'path':str(benchmark_root/name),'sha256':want} for name,want in sorted(pinned_tree(benchmark_root,BENCHMARK_SOURCE)['files'].items()))
    seen={};result=[]
    for record in records:
        name=record['path'];want=subset.hexsha(record['sha256'])
        if name in seen and seen[name]!=want:raise ValueError('conflicting full input role pins')
        if digest(secure(name))!=want:raise ValueError('full immutable input bytes differ')
        if name not in seen:result.append(record);seen[name]=want
    return result


def job(value,path,sha,stage):
    run=secure(value['ops_root'])/'runs'/rid(value,stage);raw=(run/'run.json').read_bytes();record=json.loads(raw)
    from analysis.sgns_subset_acceptance_a1 import transport_job
    transport_job(record,runroot=run,code=ROOT,source={'commit':value['runner_source']},argv=argv(stage,path,sha),
        config={'path':str(path),'sha256':sha},inputs=pins(value,path,stage),resources=resources(value))
    return hashlib.sha256(raw).hexdigest()


def verification_guard(value,path,sha,stage):
    # Capture own evidence before the first long consumer, without opening full data.
    from analysis.sgns_subset_acceptance import source_tree
    if not exact(load(path,sha),value):raise ValueError('full CLI verification request changed')
    source=source_identity();source_tree(ROOT,source)
    if source['commit']!=value['runner_source']:raise ValueError('fixed full CLI verification source differs')
    run=secure(value['ops_root'])/'runs'/rid(value,stage);out=secure(value['output'])/stage
    paths=[run/name for name in ('run.json','terminal.json','logs/stdout.log')]+[out/'result.json']
    if stage=='fit':paths.extend(out/name for name in ('vectors.npy','audit.json'))
    request=secure(path).read_bytes();own={str(name):digest(secure(name)) if name.exists() else None for name in paths}
    if not exact(source_identity(),source) or not exact(load(path,sha),value):raise ValueError('full CLI verification source/request changed')
    source_tree(ROOT,source)
    if any((digest(secure(name)) if Path(name).exists() else None)!=pin for name,pin in own.items()):raise ValueError('full CLI own evidence changed')
    return {'source':source,'request':request,'own':own}


def verification_context(value,path,sha,stage):
    from analysis.sgns_subset_acceptance import source_tree
    if not exact(load(path,sha),value):raise ValueError('full verification request changed')
    source=source_identity();source_tree(ROOT,source)
    if source['commit']!=value['runner_source']:raise ValueError('fixed full verification source differs')
    source_delta(source,pinned_tree(value['input_producer']['root'],INPUT_SOURCE),value['source_delta'])
    runtime=encoded(original.runtime(secure(value['lock'])));inputs=pins(value,path,stage);own=job(value,path,sha,stage)
    if not exact(load(path,sha),value) or not exact(source_identity(),source) or encoded(original.runtime(secure(value['lock'])))!=runtime or not exact(pins(value,path,stage),inputs) or job(value,path,sha,stage)!=own:raise ValueError('full verification source/request/runtime/job/inputs changed')
    source_tree(ROOT,source)
    return {'source':source,'runtime':json.loads(runtime),'inputs':inputs,'job':own}


def verification_snapshot(value,path,sha,stage):
    run=secure(value['ops_root'])/'runs'/rid(value,stage);out=secure(value['output'])/stage
    paths=[run/name for name in ('run.json','terminal.json','logs/stdout.log')]+[out/'result.json']
    raw={str(name):secure(name).read_bytes() for name in paths}
    artifacts={name:digest(secure(out/name)) for name in ('vectors.npy','audit.json')} if stage=='fit' else {}
    context=verification_context(value,path,sha,stage)
    if any(secure(name).read_bytes()!=data for name,data in raw.items()) or any(digest(secure(out/name))!=pin for name,pin in artifacts.items()):raise ValueError('full own verification bytes changed')
    return {'raw':raw,'artifacts':artifacts,'context':context}


def check_terminal(value,stage,terminal):
    keys={'schema','run_id','status','exit_code','signal','wall_cap_exceeded','started_at','ended_at','wall_seconds'}
    if type(terminal)is not dict or set(terminal)!=keys or not matches(terminal,{'schema':'vkgr-terminal-v1','run_id':rid(value,stage),'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}):raise ValueError('full stage terminal binding differs')
    seconds=terminal['wall_seconds']
    if type(seconds)not in (int,float) or not math.isfinite(seconds) or seconds<0 or seconds>resources(value)['wall_seconds']:raise ValueError('full stage terminal duration exceeds registered wall')


def accepted(value,path,sha,stage,bound,arrays=None):
    before=verification_snapshot(value,path,sha,stage)
    run=secure(value['ops_root'])/'runs'/rid(value,stage);out=secure(value['output'])/stage
    raw=(out/'result.json').read_bytes();receipt=json.loads(raw);terminal=json.loads((run/'terminal.json').read_bytes())
    stdout=json.loads(next((x for x in reversed((run/'logs/stdout.log').read_bytes().splitlines()) if x.strip()),b'{}'))
    expected={'schema':'sgns-native-full-stage-v1','status':'verified','verified':True,'engineering':False,'stage':stage,'backend':'B','device':'cuda','seed':42,
        'input_binding':bound,'request_sha256':sha,'run_id':rid(value,stage),'run_manifest_sha256':job(value,path,sha,stage),'config':subset.CONFIG,'test_read':False,'validation_read':False}
    check_terminal(value,stage,terminal)
    if not exact(stdout,receipt) or not matches(receipt,expected):raise ValueError('full stage terminal/stdout/job binding differs')
    if stage=='preflight' and not matches(receipt,{'real_readonly_preflight':True,'temporary_parameter_updates':False,'vectors_saved':False}):raise ValueError('real full PF receipt required')
    gate=require_acceptance(value,'accepted-'+stage)
    benchmark=benchmark_proof(value);wall=registered_wall(benchmark)
    if value['resources']['wall_hours']!=wall or receipt.get('registered_wall_hours')!=wall or not exact(receipt.get('benchmark_producer'),benchmark):raise ValueError('full stage original benchmark/wall binding differs')
    if not exact(receipt.get('acceptance'),gate):raise ValueError('full stage four-gate binding differs')
    if stage=='fit':
        if arrays is None:raise ValueError('original full arrays required for fit verification')
        if type(receipt.get('artifacts'))is not dict or set(receipt['artifacts'])!={'vectors.npy','audit.json'}:raise ValueError('complete native full output pins required')
        for name,want in receipt['artifacts'].items():
            if digest(secure(out/name))!=subset.hexsha(want):raise ValueError('native full saved artifact bytes differ')
        vectors=np.load(secure(out/'vectors.npy'),allow_pickle=False);audit=json.loads(secure(out/'audit.json').read_bytes())
        proof=verify_audit(arrays,vectors,audit,bound['original']['manifest']['binding'],compiled=value['compiled'])
        if not exact(proof,receipt.get('native_full_audit')):raise ValueError('original native full audit recheck differs')
    if not exact(verification_snapshot(value,path,sha,stage),before):raise ValueError('full own verification closure changed')
    captured=before['raw']
    return {'run_id':rid(value,stage),'receipt_sha256':hashlib.sha256(captured[str(out/'result.json')]).hexdigest(),
        **{key:hashlib.sha256(captured[str(run/name)]).hexdigest() for key,name in [('run_manifest_sha256','run.json'),('terminal_sha256','terminal.json'),('stdout_sha256','logs/stdout.log')]}}


def execute(stage,path,sha):
    value=load(path,sha)
    # This must precede reading full inputs, allocating the engine or writing output.
    acceptance=require_acceptance(value,'execute-'+stage)
    own=job(value,path,sha,stage)
    if not exact({k:os.environ.get(k) for k in job_env(value,stage)},job_env(value,stage)):raise ValueError('fixed full process environment required')
    out=secure(value['output'])/stage
    if out.exists():raise FileExistsError('new exclusive full stage output required')
    before=pins(value,path,stage);arrays,bound=input_proof(value);benchmark=benchmark_proof(value);wall=registered_wall(benchmark)
    if value['resources']['wall_hours']!=wall:raise ValueError('wall must equal ceil verified B projection times 1.5')
    original.full_input_guards(out,projected_memory=96*1024**3)
    started=time.monotonic();prior=None;artifacts={};audit_proof=None
    if stage=='preflight':
        engine=native.NativeSGNS(arrays[0],arrays[1],device='cuda',compile_step=value['compiled'])
        summary={'real_readonly_preflight':True,'temporary_parameter_updates':False,'vectors_saved':False,'positive_pairs_per_epoch':engine.pairs_per_epoch,'total_positive_pairs':engine.total}
        del engine
    else:
        prior=accepted(value,path,sha,'preflight',bound)
        with observer(wall,emit_progress):
            vectors,audit=native.fit_sgns(*arrays,device='cuda',compile_step=value['compiled'])
        audit_proof=verify_audit(arrays,vectors,audit,bound['original']['manifest']['binding'],compiled=value['compiled'])
        save_sgns_output(out,vectors,audit)
        raw={n:secure(out/n).read_bytes() for n in ('vectors.npy','audit.json')}
        if not exact(json.loads(raw['audit.json']),audit):raise ValueError('persisted native full audit differs')
        audit_proof=verify_audit(arrays,np.load(io.BytesIO(raw['vectors.npy']),allow_pickle=False),audit,bound['original']['manifest']['binding'],compiled=value['compiled'])
        artifacts={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()};summary={'vectors_saved':True}
    _,after=input_proof(value)
    if not exact(bound,after) or not exact(pins(value,path,stage),before) or job(value,path,sha,stage)!=own or not exact(load(path,sha),value) or not exact(benchmark_proof(value),benchmark):raise ValueError('full stage request/source/runtime/input/job closure changed')
    if not exact(require_acceptance(value,'closure-'+stage),acceptance):raise ValueError('full four-gate acceptance changed')
    if stage=='fit':
        if any(digest(out/n)!=sha for n,sha in artifacts.items()):raise ValueError('persisted native full artifacts changed')
    result={'schema':'sgns-native-full-stage-v1','status':'verified','verified':True,'engineering':False,'stage':stage,'backend':'B','device':'cuda','seed':42,
        'input_binding':bound,'request_sha256':sha,'run_id':rid(value,stage),'run_manifest_sha256':own,'config':subset.CONFIG,'test_read':False,'validation_read':False,
        'seconds':time.monotonic()-started,'resource_measurement':'not_measured','registered_wall_hours':wall,'benchmark_producer':benchmark,'acceptance':acceptance,'preflight_producer':prior,'artifacts':artifacts,'native_full_audit':audit_proof,**summary}
    if stage=='preflight':out.mkdir(parents=True,exist_ok=False)
    with (out/'result.json').open('xb') as f:f.write(encoded(result))
    return result


def prepare(stage,path,sha,ops_code):
    from analysis.a40_ops import load_ops
    value=load(path,sha)
    require_acceptance(value,'prepare-'+stage)
    # Metadata path does not load full data, initialize CUDA or infer scientific readiness.
    benchmark=benchmark_proof(value)
    if value['resources']['wall_hours']!=registered_wall(benchmark):raise ValueError('wall must equal ceil verified B projection times 1.5')
    current=source_identity()
    source_delta(current,pinned_tree(value['input_producer']['root'],INPUT_SOURCE),value['source_delta'])
    if current['commit']!=value['runner_source']:raise ValueError('fixed full deployment source differs')
    for name,want in {**NATIVE_ADDITIONS,**value['source_delta']}.items():
        if current['files'].get(name)!=want:raise ValueError('fixed declared source additions differ')
    run_id=rid(value,stage);ops=secure(value['ops_root'])
    if (ops/'runs'/run_id).exists() or (secure(value['output'])/stage).exists() or any(list((ops/'queue'/state).glob('*-'+run_id+'.json')) for state in ('pending','claimed','rejected')):raise FileExistsError('full run/output/queue collision refused')
    _,launcher,_,_=load_ops(ops_code)
    args=['--root',str(ops),'--run-id',run_id,'--kind','ops','--cwd',str(ROOT),'--config',str(path),'--seed','42']
    for key,val in value['resources'].items():args+=['--'+key.replace('_','-'),str(val)]
    for key,val in fixed_env().items():args+=['--env',key+'='+val]
    for record in pins(value,path,stage):args+=['--input',record['path']]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*args,'--',*argv(stage,path,sha)]));metadata=launcher.prepare(spec)
    recorded=launcher._recorded_env(metadata['env'],launcher._job_env(metadata['env'],ops/'runs'/run_id,run_id))
    if not exact(recorded,job_env(value,stage)):raise ValueError('unexpected inherited full launcher environment')
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
        if args.command=='dry':result={'status':'dry','training':False,'submitted':False,'backend':'B','four_gates_ready':False,'four_gate_consumer_pinned':True,'full_fit_enabled':False,'config':subset.CONFIG}
        elif args.command=='plan':result=prepare(args.stage,args.request,args.request_sha256,args.ops_code)
        elif args.command=='verify':
            value=load(args.request,args.request_sha256)
            before=verification_guard(value,args.request,args.request_sha256,args.stage)
            require_acceptance(value,'verify-'+args.stage)
            arrays,bound=input_proof(value);result={'status':'verified','readonly':True,'producer':accepted(value,args.request,args.request_sha256,args.stage,bound,arrays)}
            if not exact(verification_guard(value,args.request,args.request_sha256,args.stage),before):raise ValueError('full CLI verification closure changed')
        else:result=execute(args.command,args.request,args.request_sha256)
        print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True);return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc),'training_may_have_started':args.command=='fit'},sort_keys=True),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
