"""Route B bounded subset benchmark. No full-fit/SGNS vector export CLI."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from analysis.ur001_derive_cli import secure,digest,encoded
from analysis.p_query_sprint_ops import source
from analysis.sgns_speedup_profile import NATIVE_SHA,FULL_PAIRS
from vkgr.sid_coll_sgns import SGNSConfig
from vkgr.sgns_gpu_native import NativeSGNS,positive_chunks,batches,negatives,distribution,FrequencyAudit
ROOT=Path(__file__).resolve().parents[1]
MODULE='analysis.sgns_gpu_native_b'
RESOURCES={'cpu_cores':2,'ram_mib':16384,'vram_mib':8192,'disk_gib':1,'wall_hours':2}


def exact(actual,expected):
    """JSON equality with recursive type identity; bool/int/float never alias."""
    if type(actual)is not type(expected):return False
    if type(expected)is dict:return set(actual)==set(expected) and all(exact(actual[k],v) for k,v in expected.items())
    if type(expected)is list:return len(actual)==len(expected) and all(exact(a,b) for a,b in zip(actual,expected))
    return actual==expected


def matches(actual,expected):
    # Preserve launcher/receipt extension fields, but every expected nested
    # object retains its exact key set and recursive scalar types.
    return type(actual)is dict and all(k in actual and exact(actual[k],v) for k,v in expected.items())


def progress(phase,**fields):
    print(json.dumps({'phase':phase,'monotonic':time.monotonic(),**fields},sort_keys=True),file=sys.stderr,flush=True)


def benchmark(catalog,histories,*,device='cuda',warmup=8,measured=256,compiled=True):
    if type(warmup)is not int or not 0<=warmup<=32 or type(measured)is not int or not 1<=measured<=4096:raise ValueError('bounded warmup 0..32/measured 1..4096 required')
    progress('build_begin',device=device,compiled_step=compiled)
    setup=time.perf_counter();engine=NativeSGNS(catalog,histories,device=device,compile_step=compiled)
    required=max(warmup,measured)*engine.config.batch_size
    if engine.pairs_per_epoch<required:raise ValueError('insufficient full batches; no truncated benchmark')
    if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    setup_seconds=time.perf_counter()-setup;progress('build_end',positive_pairs_per_epoch=engine.pairs_per_epoch,setup_seconds=setup_seconds)
    loss=torch.zeros((),device=device,dtype=torch.float64);frequencies=FrequencyAudit(engine.p)
    iterator=iter(batches(positive_chunks(engine.events,engine.users,5,engine.center_chunk),4096))
    started=time.perf_counter()
    # Warm the compiled update on a disposable prefix, then restore exactly the
    # original initialization/progress/RNG. Timed prefix includes its first GPU
    # pair chunk assembly; warmup must not hide that cost or change its pairs.
    progress('warmup_compile_begin',batches=warmup)
    initial=engine.center.clone()
    for _ in range(warmup):
        pair=next(iterator);noise=negatives(pair,engine.cdf,engine.p,5,engine.generator)
        engine.step(engine.center,engine.output,pair,noise,engine.processed,engine.total,.025,1e-4,engine.finite,loss)
    if device=='cuda':torch.cuda.synchronize()
    if not bool(engine.finite):raise ValueError('nonfinite warmup; STOP')
    engine.center.copy_(initial);engine.output.zero_();engine.processed.zero_();engine.finite.fill_(True);loss.zero_()
    engine.generator.manual_seed(42);del initial
    progress('warmup_compile_end',batches=warmup)
    iterator=iter(batches(positive_chunks(engine.events,engine.users,5,engine.center_chunk),4096))
    if device=='cuda':torch.cuda.synchronize()
    progress('measure_begin',batches=measured)
    timed=time.perf_counter()
    for ordinal in range(measured):
        pair=next(iterator);noise=negatives(pair,engine.cdf,engine.p,5,engine.generator)
        engine.step(engine.center,engine.output,pair,noise,engine.processed,engine.total,.025,1e-4,engine.finite,loss)
        # Same diagnostic work is included in timed throughput (conservative).
        frequencies.add(pair,noise)
        if (ordinal+1)%64==0 or ordinal+1==measured:progress('measure_progress',batches_enqueued=ordinal+1,total_batches=measured)
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-timed;total_elapsed=time.perf_counter()-started
    progress('measure_end',completed_batches=measured,measured_seconds=elapsed)
    if not bool(engine.finite) or not torch.isfinite(engine.center).all() or not torch.isfinite(engine.output).all():raise ValueError('nonfinite SGNS benchmark; STOP')
    rate=measured*4096/elapsed;projected=FULL_PAIRS/rate/3600
    # No model/acceptance output; temporary updated tables are discarded.
    return {'schema':'sgns-gpu-native-b-benchmark-v1','status':'verified','device':device,'config':asdict(engine.config),
        'compiled_step':compiled,'native_epoch_positive_pairs':engine.pairs_per_epoch,'warmup_batches':warmup,'measured_batches':measured,
        'warmup_positive_pairs':warmup*4096,'measured_positive_pairs':measured*4096,'setup_seconds':setup_seconds,
        'measured_seconds':elapsed,'warmup_and_measured_seconds':total_elapsed,'measured_pairs_per_second':rate,
        'full_fit_linear_projection_hours':projected,'B_throughput_12h_diagnostic':projected<=12 if device=='cuda' else 'UNKNOWN: CPU only',
        'full_pair_count':FULL_PAIRS,**frequencies.report(),
        'negative_bin_method':'row modulo 64; measured prefix only; conditional expectation; all 5 negatives counted',
        'warmup_policy':'disposable original prefix; restore initial tables/progress/seed before timed original prefix',
        'timing_scope':'includes first and subsequent GPU pair assembly chunks, sampling, update and frequency diagnostics',
        'summed_loss':float(loss),'vectors_saved':False,'vectors_returned':False,'four_gates_passed':False,
        'science_training':False,'temporary_parameter_updates':True,'test_read':False,'validation_read':False,
        'peak_vram_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else 'UNKNOWN',
        'peak_vram_scope':'torch_counter_excludes_context','runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
        'limits':['prefix throughput is not full fit measurement','linear fit projection excludes full setup/final health/SID/G chain',
                  'four frozen usability gates still required','RNG and floating ordering differ from original']}


def validate(value):
    keys={'schema','runner_source','bundle','bundle_sha256','lock','ops_root','output','preflight_run_id','benchmark_run_id','warmup_batches','measured_batches','compiled'}
    if type(value)is not dict or set(value)!=keys or value['schema']!='sgns-gpu-native-b-request-v1':raise ValueError('complete fixed B request required')
    for name,size in (('runner_source',40),('bundle_sha256',64)):
        text=value[name]
        if type(text)is not str or len(text)!=size or set(text)-set('0123456789abcdef'):raise ValueError('fixed source/input hash required')
    for name in ('bundle','lock','ops_root','output'):secure(value[name])
    out=secure(value['output'])
    for root in (ROOT,secure(value['bundle']),secure(value['ops_root'])):
        if out==root or root in out.parents or out in root.parents:raise ValueError('exclusive output outside immutable trees required')
    for name in ('preflight_run_id','benchmark_run_id'):
        text=value[name]
        if type(text)is not str or not text.startswith('sgns-gpu-native-b-') or set(text)-set('abcdefghijklmnopqrstuvwxyz0123456789-'):raise ValueError('descriptive B run identities required')
    if value['preflight_run_id']==value['benchmark_run_id']:raise ValueError('distinct PF and benchmark identities required')
    if type(value['compiled'])is not bool or type(value['warmup_batches'])is not int or not 0<=value['warmup_batches']<=32 or type(value['measured_batches'])is not int or not 1<=value['measured_batches']<=4096:raise ValueError('bounded fixed benchmark options required')
    return value


def load(path,sha):
    raw=secure(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('fixed request SHA differs')
    return validate(json.loads(raw))


def inputs(value):
    from analysis.new_instance_prepare import LOCK_SHA,runtime
    from analysis.sgns_gpu_subset import read_bundle
    if digest(ROOT/'src/vkgr/sid_coll_sgns.py')!=NATIVE_SHA:raise ValueError('frozen original SGNS bytes differ')
    lock=secure(value['lock']);bundle=secure(value['bundle']);raw=(bundle/'manifest.json').read_bytes()
    if digest(lock)!=LOCK_SHA or hashlib.sha256(raw).hexdigest()!=value['bundle_sha256']:raise ValueError('frozen runtime/input manifest required')
    producer=json.loads(raw)
    expected_runtime=json.loads(json.dumps(runtime(lock),allow_nan=False))
    if not exact(expected_runtime,producer.get('runtime')):raise ValueError('producer/consumer runtime differs')
    arrays,manifest=read_bundle(bundle,value['bundle_sha256'],producer['source'])
    return arrays,{'manifest_sha256':value['bundle_sha256'],'producer_source':manifest['source'],'runtime':manifest['runtime'],
                   'train_sequence_sha256':manifest['train_sequence_sha256'],'artifacts':manifest['artifacts']}


def pins(value,path,stage):
    bundle=secure(value['bundle']);paths=[secure(path),secure(value['lock']),bundle/'manifest.json',*(bundle/name for name in ('catalog.json','content.npy','histories.jsonl')),
        ROOT/'src/vkgr/sid_coll_sgns.py',ROOT/'src/vkgr/sgns_gpu_native.py',ROOT/'analysis/sgns_gpu_native_b.py']
    if stage=='benchmark':
        run=secure(value['ops_root'])/'runs'/value['preflight_run_id'];paths.extend(run/name for name in ('run.json','terminal.json','logs/stdout.log'))
        paths.append(secure(value['output'])/'preflight/result.json')
    return [{'path':str(secure(p)),'sha256':digest(p)} for p in paths]


def argv(stage,path,sha):return [sys.executable,'-B','-m',MODULE,stage,'--request',str(path),'--request-sha256',sha,'--execute-on-new-instance']
def resources():return {**{k:RESOURCES[k] for k in ('cpu_cores','ram_mib','vram_mib','disk_gib')},'wall_seconds':RESOURCES['wall_hours']*3600.}
def run_id(value,stage):return value['preflight_run_id' if stage=='preflight' else 'benchmark_run_id']


def fixed_env():return {'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def job_env(value,stage):
    rid=run_id(value,stage)
    return {**fixed_env(),'VKGR_RUN_ID':rid,'VKGR_PROGRESS_PATH':str(secure(value['ops_root'])/'runs'/rid/'progress.json')}


def job(value,path,sha,stage):
    rid=run_id(value,stage);target=secure(value['ops_root'])/'runs'/rid/'run.json';raw=target.read_bytes();record=json.loads(raw)
    expected={'schema':'vkgr-run-v1','run_id':rid,'kind':'ops','cwd':str(ROOT),'argv':argv(stage,path,sha),'seed':42,
        'git':{'commit':value['runner_source'],'dirty':False,'dirty_files':[]},'config':{'path':str(path),'sha256':sha},'resources':resources(),'env':job_env(value,stage)}
    if not matches(record,expected) or not exact([{'path':x['path'],'sha256':x['sha256']} for x in record.get('inputs',[])],pins(value,path,stage)):raise ValueError('own job/source/argv/resources/input role pins differ')
    return hashlib.sha256(raw).hexdigest()


def check_measurement(receipt,value):
    count=value['measured_batches']*4096;seconds=receipt.get('measured_seconds')
    if type(seconds)not in (int,float) or not np.isfinite(seconds) or seconds<=0:raise ValueError('positive finite measured time required')
    rate=count/seconds;hours=FULL_PAIRS/rate/3600
    for key,expected in (('measured_pairs_per_second',rate),('full_fit_linear_projection_hours',hours)):
        actual=receipt.get(key)
        if type(actual)not in (int,float) or not np.isfinite(actual) or not np.isclose(actual,expected,rtol=1e-12,atol=0):raise ValueError('measured throughput/projection arithmetic differs')
    if not exact(receipt.get('full_pair_count'),FULL_PAIRS) or receipt.get('B_throughput_12h_diagnostic')is not bool(hours<=12):raise ValueError('frozen B throughput diagnostic differs')
    for key,integer in (('negative_observed_bins',True),('negative_expected_bins',False)):
        values=receipt.get(key)
        if type(values)is not list or len(values)!=64 or any(type(x)not in ((int,) if integer else (int,float)) or not np.isfinite(x) or x<(-1e-7 if not integer else 0) for x in values):raise ValueError('complete finite negative bins required')
        if not np.isclose(sum(values),count*5,rtol=1e-10,atol=1e-5):raise ValueError('negative frequency mass differs from measured pair count')


def accepted(value,path,sha,stage,ledger,bound):
    rid=run_id(value,stage);run=secure(value['ops_root'])/'runs'/rid
    terminal=json.loads((run/'terminal.json').read_bytes());receipt=json.loads((secure(value['output'])/stage/'result.json').read_bytes())
    lines=(run/'logs/stdout.log').read_bytes().splitlines();stdout=json.loads(next((x for x in reversed(lines) if x.strip()),b'{}'))
    expected={'schema':'sgns-gpu-native-b-stage-v1','stage':stage,'status':'verified','engineering':False,'run_id':rid,'runner_source':ledger,
        'input_binding':bound,'request_sha256':sha,'run_manifest_sha256':job(value,path,sha,stage),'test_read':False,'validation_read':False,
        'science_training':False,'native_epoch_positive_pairs':34770216,'config':asdict(SGNSConfig())}
    if terminal.get('schema')!='vkgr-terminal-v1' or terminal.get('run_id')!=rid or terminal.get('status')!='succeeded' or not exact(terminal.get('exit_code'),0) or terminal.get('signal') is not None or not exact(stdout,receipt) or not matches(receipt,expected):raise ValueError('successful bound native B terminal/stdout required')
    if stage=='preflight' and (receipt.get('real_readonly_preflight')is not True or receipt.get('temporary_parameter_updates')is not False):raise ValueError('readonly real PF required')
    if stage=='benchmark':
        expected_b={'measured_batches':value['measured_batches'],'warmup_batches':value['warmup_batches'],'measured_positive_pairs':value['measured_batches']*4096,
            'warmup_positive_pairs':value['warmup_batches']*4096,'compiled_step':value['compiled'],'device':'cuda','vectors_saved':False,'vectors_returned':False,'four_gates_passed':False}
        if not matches(receipt,expected_b):raise ValueError('bounded benchmark receipt differs')
        check_measurement(receipt,value)
    return {'run_id':rid,'stdout_sha256':digest(run/'logs/stdout.log'),'result_sha256':digest(secure(value['output'])/stage/'result.json')}


def execute(stage,path,sha):
    progress('execute_begin',stage=stage)
    value=load(path,sha);ledger=source(ROOT,value['runner_source']);own=job(value,path,sha,stage);before=pins(value,path,stage)
    if not exact({k:os.environ.get(k) for k in job_env(value,stage)},job_env(value,stage)):raise ValueError('fixed own run/native thread environment required')
    from analysis.ur001_derive_cli import guards
    guards(secure(value['output']).parent,projected_memory=8*1024**3)
    out=secure(value['output'])/stage
    if out.exists():raise FileExistsError('exclusive fresh phase output required')
    progress('input_begin',stage=stage)
    arrays,bound=inputs(value);catalog,histories,_=arrays
    progress('input_end',stage=stage)
    prior=accepted(value,path,sha,'preflight',ledger,bound) if stage=='benchmark' else None
    if stage=='preflight':
        progress('build_begin',device='cuda',compiled_step=False)
        engine=NativeSGNS(catalog,histories,device='cuda',compile_step=False)
        progress('build_end',positive_pairs_per_epoch=engine.pairs_per_epoch)
        result={'device':'cuda','config':asdict(engine.config),'native_epoch_positive_pairs':engine.pairs_per_epoch,'real_readonly_preflight':True,'temporary_parameter_updates':False}
    else:result=benchmark(catalog,histories,warmup=value['warmup_batches'],measured=value['measured_batches'],compiled=value['compiled'])
    if result['native_epoch_positive_pairs']!=34770216:raise ValueError('original subset per-epoch positive pair count differs')
    progress('postcheck_begin',stage=stage)
    if not exact(inputs(value)[1],bound) or not exact(source(ROOT,value['runner_source']),ledger) or not exact(pins(value,path,stage),before) or not exact(load(path,sha),value) or job(value,path,sha,stage)!=own:raise ValueError('source/runtime/input/job/request closure changed')
    progress('postcheck_end',stage=stage)
    result.update({'schema':'sgns-gpu-native-b-stage-v1','status':'verified','stage':stage,'engineering':False,'run_id':run_id(value,stage),
        'runner_source':ledger,'input_binding':bound,'request_sha256':sha,'run_manifest_sha256':own,'preflight_producer':prior,
        'science_training':False,'test_read':False,'validation_read':False,'resources':{**RESOURCES,'measurement':'not_measured conservative first declaration'}})
    out.mkdir(parents=True,exist_ok=False)
    with (out/'result.json').open('xb') as stream:stream.write(encoded(result))
    progress('execute_end',stage=stage,status='verified')
    return result


def prepare(stage,path,sha,ops_code):
    from analysis.a40_ops import load_ops
    value=load(path,sha);source(ROOT,value['runner_source']);ops=secure(value['ops_root']);rid=run_id(value,stage)
    if (ops/'runs'/rid).exists() or (secure(value['output'])/stage).exists():raise FileExistsError('existing run/output refused')
    for state in ('pending','claimed','rejected'):
        if list((ops/'queue'/state).glob('*-'+rid+'.json')):raise FileExistsError('queue collision refused')
    _,launcher,_,_=load_ops(ops_code)
    args=['--root',str(ops),'--run-id',rid,'--kind','ops','--cwd',str(ROOT),'--config',str(path),'--seed','42']
    for key,val in RESOURCES.items():args+=['--'+key.replace('_','-'),str(val)]
    for key,val in fixed_env().items():args+=['--env',key+'='+val]
    for pin in pins(value,path,stage):args+=['--input',pin['path']]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*args,'--',*argv(stage,path,sha)]))
    metadata=launcher.prepare(spec)
    recorded=launcher._recorded_env(metadata['env'],launcher._job_env(metadata['env'],ops/'runs'/rid,rid))
    if not exact(recorded,job_env(value,stage)):raise ValueError('unexpected inherited launcher environment')
    return {'status':'prepared_dry','spec':spec,'metadata':metadata,'submitted':False,'real_preflight':False,'training':False}


def main(argv_in=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    for name in ('preflight','benchmark','prepare','verify'):
        p=sub.add_parser(name);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True)
        if name in ('prepare','verify'):p.add_argument('--stage',choices=('preflight','benchmark'),required=True)
        if name=='prepare':p.add_argument('--ops-code',type=Path,required=True)
        elif name!='verify':p.add_argument('--execute-on-new-instance',required=True,action='store_true')
    args=parser.parse_args(argv_in)
    try:
        if args.command=='dry':result={'status':'dry','training':False,'config':asdict(SGNSConfig()),'real_preflight':False,'submitted':False,'full_fit_available':False}
        elif args.command=='prepare':result=prepare(args.stage,args.request,args.request_sha256,args.ops_code)
        elif args.command=='verify':
            value=load(args.request,args.request_sha256);ledger=source(ROOT,value['runner_source']);_,bound=inputs(value)
            prior=accepted(value,args.request,args.request_sha256,'preflight',ledger,bound) if args.stage=='benchmark' else None
            own=accepted(value,args.request,args.request_sha256,args.stage,ledger,bound)
            if args.stage=='benchmark':
                receipt=json.loads((secure(value['output'])/'benchmark/result.json').read_bytes())
                if not exact(receipt.get('preflight_producer'),prior):raise ValueError('benchmark PF producer differs')
            if not exact(source(ROOT,value['runner_source']),ledger) or not exact(inputs(value)[1],bound) or not exact(load(args.request,args.request_sha256),value) or not exact(accepted(value,args.request,args.request_sha256,args.stage,ledger,bound),own):raise ValueError('verify source/runtime/input/job/receipt closure changed')
            result={'status':'verified','readonly':True,'producer':own,'science_training':False}
        else:result=execute(args.command,args.request,args.request_sha256)
        print(json.dumps(result,sort_keys=True),flush=True);return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
        progress('rejected',command=args.command)
        print(json.dumps({'status':'rejected','reason':str(exc),'training':False},sort_keys=True),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
