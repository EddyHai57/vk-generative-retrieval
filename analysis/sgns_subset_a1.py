"""Fixed ad1c428 single A1 seed42 acceptance stages; no submit or retry command."""
import argparse
import hashlib
import json
import os
import math
import resource
from pathlib import Path
import sys

from vkgr import sgns_subset_a1 as core

ROOT=Path(__file__).resolve().parents[1]
RESOURCE_SCOPE='own process sampled context-inclusive GPU; RSS self high water plus two largest direct-helper high waters, conservative bound; transient GPU peaks between samples UNKNOWN'

def canonical(value):
    from vkgr.data_audit import canonical_bytes
    return canonical_bytes(value)


def invocation(stage,request,sha):
    return [sys.executable,'-B','-m','analysis.sgns_subset_a1',stage,'--request',str(request),'--request-sha256',sha]


def read_request(path,sha,*,consume=True):
    raw=core.file_bytes({'path':str(path),'sha256':sha})
    value=json.loads(raw)
    if consume:core.accepted_request(value,sha)
    elif type(value) is not dict or value.get('schema')!=core.SCHEMA:
        raise ValueError('fixed request schema required')
    return value


def identity(value):
    from analysis.ur001_derive_cli import source_identity
    from analysis.new_instance_prepare import runtime
    source=source_identity()
    if source['commit']!=value['source_commit']:
        raise ValueError('request fixed source commit differs')
    core.check_native()
    if value['lock']['sha256']!=core.NATIVE_LOCK_SHA:raise ValueError('frozen native runtime lock required')
    env=runtime(core.absolute(value['lock']['path']))
    if core.file_bytes(value['lock']) is None:raise ValueError('fixed lock required')
    return source,env


def stage_inputs(value,request,sha):
    records=[{'path':str(request),'sha256':sha},value['lock']]
    for key in ('prepared','history','content'):records.append(value[key])
    for role in ('input_producer','cpu_sgns','gpu_sgns'):
        records.extend(value[role].values())
    for role in ('cpu_sid','gpu_sid'):
        records.extend(value[role]['producer'].values())
        records.extend(value[role][key] for key in ('manifest','input_manifest'))
    result=[];seen={}
    for record in records:
        path=str(core.absolute(record['path']));pin=core.pin(record['sha256'])
        if path in seen and seen[path]!=pin:raise ValueError('conflicting input role pins')
        if path not in seen:result.append({'path':path,'sha256':pin});seen[path]=pin
    return result


def job_binding(value,request,sha,stage):
    """Same fixed OPS identity for current and previous jobs, without live env."""
    run_id=core.RUN_IDS[stage]
    run_path=core.absolute(value['ops_root'])/'runs'/run_id/'run.json'
    raw=run_path.read_bytes();run=json.loads(raw)
    expected_inputs=stage_inputs(value,request,sha)
    actual=[{'path':r['path'],'sha256':r['sha256']} for r in run.get('inputs',[])]
    if (run.get('run_id')!=run_id or run.get('kind')!='ops' or run.get('cwd')!=str(ROOT)
            or run.get('argv')!=invocation(stage,request,sha) or run.get('seed')!=42
            or canonical(run.get('git'))!=canonical({'commit':value['source_commit'],'dirty':False,'dirty_files':[]})
            or run.get('config',{}).get('path')!=str(request)
            or run.get('config',{}).get('sha256')!=sha or actual!=expected_inputs):
        raise ValueError('fixed A1 OPS job/source/argv/input binding differs')
    for record in expected_inputs:core.file_bytes(record)
    return hashlib.sha256(raw).hexdigest()


def own_job(value,request,sha,stage):
    if os.environ.get('VKGR_RUN_ID')!=core.RUN_IDS[stage]:
        raise ValueError('own fixed A1 OPS job/source/argv/input binding differs')
    return job_binding(value,request,sha,stage)


def receipt_binding(receipt,*,prior,source,env,config,proof,request_sha,run_sha):
    expected={'schema':'gpu-sgns-subset-a1-stage-v1','authority':'ad1c428','engineering':False,
              'stage':prior,'run_id':core.RUN_IDS[prior],'run_manifest_sha256':run_sha,'request_sha256':request_sha,
              'source':source,'runtime':env,'config':config,'input_producers':proof,'test_read':False,
              'subset_four_gate_acceptance':False,'status':'ready' if prior=='preflight' else 'complete','resource_scope':RESOURCE_SCOPE}
    details={'query_count','catalog_size','native_runtime','payload_written','scratch_written'} if prior=='preflight' else {'training','scientific_verified','native_manifest_sha256'}
    if (set(receipt)!=set(expected)|details|{'context_inclusive_peak_mib','context_samples','rss_tree_upper_bound_mib'}
            or any(type(receipt.get(key)) is not type(value) or canonical(receipt.get(key))!=canonical(value) for key,value in expected.items())):
        raise ValueError('original A1 predecessor receipt semantics differ')
    measured_resources(receipt)
    return receipt


def train_output_binding(value,receipt,engine,config,native_runtime):
    """Mechanical original manifest/artifact semantics; native verify still follows."""
    path=core.absolute(value['output_root'])/'train/native/manifest.json'
    original=core.check_native()
    native=original._read_run(path,receipt.get('native_manifest_sha256'))
    binding=native['binding']
    expected={'task':original.TASK_ID,'budget':original.BUDGET,'init_version':'INIT_EMB_N002','config':config,
              'config_sha256':original._digest(config),'source':engine['_source'](ROOT),'data_root':value['data_root'],
              'runtime':native_runtime,'prepared_manifest_sha256':value['prepared']['sha256'],
              'history_manifest_sha256':value['history']['sha256'],'synthetic':False,'test_gate':'closed'}
    if (canonical(native['config'])!=canonical(config) or native.get('eval_workers')!=1 or native.get('decode') is not None
            or any(type(binding.get(key)) is not type(expected_value) or canonical(binding.get(key))!=canonical(expected_value) for key,expected_value in expected.items())
            or binding.get('sid',{}).get('manifest_sha256')!=value['gpu_sid']['manifest']['sha256']
            or binding.get('sid',{}).get('input_manifest_sha256')!=value['gpu_sid']['input_manifest']['sha256']
            or binding.get('context_b1',{}).get('history_manifest_sha256')!=value['history']['sha256']
            or binding.get('context_b1',{}).get('sid_version')!=core.SID_VERSION):
        raise ValueError('original training output semantic binding differs')
    original._check_best_hash(path.parent,native['best'])
    last=native.get('last',{})
    if (last.get('path')!='last.pt' or not native['epochs'] or last.get('epoch')!=len(native['epochs'])
            or not 1<=native['best']['epoch']<=last['epoch']):
        raise ValueError('original training output selection/artifacts differ')
    core.file_bytes({'path':str(path.parent/'last.pt'),'sha256':last.get('sha256')})
    for ordinal,row in enumerate(native['epochs'],1):
        if row.get('epoch')!=ordinal or row.get('evaluation',{}).get('path')!=f'evaluation-{ordinal:03d}':
            raise ValueError('original training epoch/evaluation output differs')
        ep=path.parent/row['evaluation']['path']/'manifest.json'
        core.file_bytes({'path':str(ep),'sha256':row['evaluation'].get('manifest_sha256')})
    return native


def predecessor(value,stage,request_sha,source,*,request=None):
    if request is None:raise ValueError('explicit current request path required for predecessor job/source/argv/input binding')
    prior='preflight' if stage=='train' else 'train'
    run_id=core.RUN_IDS[prior];root=core.absolute(value['ops_root'])/'runs'/run_id
    run_sha=job_binding(value,request,request_sha,prior)
    terminal=json.loads((root/'terminal.json').read_bytes())
    raw=(root/'logs/stdout.log').read_bytes()
    receipt=json.loads(next((s for s in reversed(raw.splitlines()) if s.strip()),b'{}'))
    artifact=core.absolute(value['output_root'])/prior/'stage-receipt.json'
    if (terminal.get('run_id')!=run_id or terminal.get('status')!='succeeded' or terminal.get('exit_code')!=0
            or terminal.get('signal') is not None or json.loads(artifact.read_bytes())!=receipt):
        raise ValueError('successful original A1 predecessor closure required')
    actual_source,env=identity(value)
    if actual_source!=source:raise ValueError('original predecessor source differs')
    engine,_,_,proof=core.engine_for(value,request_sha,source)
    config=core.registered_config(value['gpu_sid']['manifest']['sha256'])
    receipt_binding(receipt,prior=prior,source=source,env=env,config=config,proof=proof,request_sha=request_sha,run_sha=run_sha)
    native_runtime=engine['_runtime'](False)
    if prior=='preflight':
        family=json.loads(core.file_bytes(value['prepared']));parent=core.absolute(value['prepared']['path']).parent
        count=lambda name:len(json.loads(core.file_bytes({'path':str(parent/name),'sha256':family['artifacts'][name]['sha256']})))
        if (type(receipt.get('query_count')) is not int or receipt['query_count']!=count('like-validation.json')
                or type(receipt.get('catalog_size')) is not int or receipt['catalog_size']!=count('catalog.json')
                or canonical(receipt.get('native_runtime'))!=canonical(native_runtime) or receipt.get('payload_written') is not False
                or receipt.get('scratch_written') is not True
                or not (core.absolute(value['scratch_root'])/'preflight').is_dir() or set(artifact.parent.iterdir())!={artifact}):
            raise ValueError('original preflight output semantics differ')
    else:
        if receipt.get('training') is not True or receipt.get('scientific_verified') is not False:
            raise ValueError('original training receipt semantics differ')
        train_output_binding(value,receipt,engine,config,native_runtime)
    return receipt


def measured_resources(record):
    peak=record.get('context_inclusive_peak_mib');samples=record.get('context_samples');rss=record.get('rss_tree_upper_bound_mib')
    if (type(peak) not in (int,float) or not math.isfinite(peak) or peak<=0
            or type(samples) is not int or samples<=0 or type(rss) not in (int,float) or not math.isfinite(rss) or rss<=0):
        raise ValueError('positive measured context-inclusive GPU and RSS upper-bound required')
    return {'vram_mib':math.ceil(peak*1.15/256)*256,'ram_mib':math.ceil(rss*1.15/256)*256}


def prepare(stage,request,sha,ops_code):
    from analysis.a40_ops import load_ops
    value=read_request(request,sha,consume=False)
    # Transport dry consumes every declared file only as launcher hashes; it does
    # not claim producer acceptance, a real preflight or any model/data readiness.
    _,launcher,_,_=load_ops(ops_code)
    run_id=core.RUN_IDS[stage];ops=core.absolute(value['ops_root'])
    if (ops/'runs'/run_id).exists():raise FileExistsError('fixed one-shot run exists')
    for state in ('pending','claimed','rejected'):
        if list((ops/'queue'/state).glob('*-'+run_id+'.json')):raise FileExistsError('fixed one-shot job already queued')
    ram,vram=98304,32768
    scope='not_measured conservative first phase limit; context-only input preflight is not a training-peak measurement'
    if stage=='verify':
        from analysis.ur001_derive_cli import source_identity
        measured=measured_resources(predecessor(value,'verify',sha,source_identity(),request=request))
        ram,vram=measured['ram_mib'],measured['vram_mib']
        scope='original successful train peak x1.15 ceil256; RSS conservative process-tree upper bound'
    raw=['--root',str(ops),'--run-id',run_id,'--kind','ops','--cwd',str(ROOT),'--config',str(request),'--seed','42',
         '--env','PYTHONPATH='+str(ROOT/'src'),'--env','VKGR_DATA_ROOT='+str(value['data_root']),
         '--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
         '--env','CUBLAS_WORKSPACE_CONFIG=:4096:8','--cpu-cores','1','--ram-mib',str(ram),
         '--vram-mib',str(vram),'--disk-gib','4','--wall-hours','48' if stage!='preflight' else '2']
    for record in stage_inputs(value,request,sha):raw+=['--input',record['path']]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*raw,'--',*invocation(stage,request,sha)]))
    metadata=launcher.prepare(spec)
    if metadata['git']['commit']!=value['source_commit'] or metadata['git']['dirty_files']:
        raise ValueError('clean pinned source required')
    return {'status':'prepared_dry','spec':spec,'metadata':metadata,'submitted':False,
            'resource_scope':scope,
            'real_preflight':False,'subset_four_gate_acceptance':False}


def _execute(stage,request,sha,sampler):
    from analysis.ur001_derive_cli import encoded
    value=read_request(request,sha)
    source,env=identity(value)
    run_sha=own_job(value,request,sha,stage)
    if os.environ.get('VKGR_DATA_ROOT')!=value['data_root']:
        raise ValueError('fixed historical data channel required')
    engine,common,extra,proof=core.engine_for(value,sha,source)
    config=core.registered_config(value['gpu_sid']['manifest']['sha256'])
    output=core.absolute(value['output_root'])/stage
    scratch=core.absolute(value['scratch_root'])/stage
    if output.exists() or scratch.exists():raise FileExistsError('new exclusive output/scratch required')
    if stage!='preflight':predecessor(value,stage,sha,source,request=request)
    if stage=='preflight':
        context,engineering,sid=engine['_contexts'](*common,config,scratch,**extra)
        if engineering is not False or context.binding.get('synthetic') is not False:
            raise ValueError('formal real-input preflight only')
        runtime=engine['_runtime'](False)
        details={'status':'ready','query_count':len(context.queries),'catalog_size':len(context.catalog),
                 'native_runtime':runtime,'payload_written':False,'scratch_written':True}
    elif stage=='train':
        output.mkdir(parents=True,exist_ok=False)
        native=engine['run'](*common,config,output/'native',temporary=scratch,eval_workers=1,progress=core.absolute(value['scratch_root'])/'training-progress.json',decode=None,**extra)
        details={'status':native['status'],'native_manifest_sha256':native['manifest_sha256'],
                 'training':True,'scientific_verified':False,'subset_four_gate_acceptance':False}
    else:
        trained=core.absolute(value['output_root'])/'train/native/manifest.json'
        original=predecessor(value,stage,sha,source,request=request)
        output.mkdir(parents=True,exist_ok=False)
        verified=engine['verify'](*common,trained,original['native_manifest_sha256'],output/'native',temporary=scratch,eval_workers=1,decode=None,**extra)
        manifest=json.loads(trained.read_bytes());best=manifest['best']['epoch']
        aggregate=next(row for row in manifest['epochs'] if row['epoch']==best)['metrics']
        # Preserve the original integer hit numerator; do not infer it from rounded recall.
        context,engineering,_=engine['_contexts'](*common,config,core.absolute(value['scratch_root'])/'verify-hits',**extra)
        if engineering is not False:raise ValueError('formal validation rows required')
        from vkgr.sid_depth import SIDTokensLD
        best_record=next(row for row in manifest['epochs'] if row['epoch']==best)
        checked_aggregate,_,rows=engine['_read_evaluation'](best_record['evaluation']['kind'],trained.parent,best_record,context,SIDTokensLD(context.codec,50))
        if checked_aggregate!=aggregate:raise ValueError('selected validation aggregate changed')
        hits=core.validation_hits(rows,context.queries,aggregate)
        if verified.get('status')!='verified' or hits<378:
            raise ValueError('STOP: A1 validation acceptance failed')
        details={'status':'verified','verified':True,'native_verification':verified,'validation_hits':hits,'hits_gate':378,
                 'training':False,'subset_four_gate_acceptance':False}
    current_source,current_env=identity(value)
    if current_source!=source or current_env!=env:raise ValueError('source/runtime changed during A1 stage')
    core.closure(value)
    from vkgr.sgns_subset_a1_inputs import input_producer
    input_producer(value['input_producer'],value['prepared'],value['history'],source)
    sampler.sample()
    rss_self=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_child=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    factor=1 if sys.platform=='darwin' else 1024
    resources={'context_inclusive_peak_mib':sampler.peak,'context_samples':sampler.samples,
               'rss_tree_upper_bound_mib':(rss_self+2*rss_child)*factor/1024**2,
               'resource_scope':RESOURCE_SCOPE}
    measured_resources(resources)
    receipt={'schema':'gpu-sgns-subset-a1-stage-v1','authority':'ad1c428','engineering':False,
             'stage':stage,'run_id':core.RUN_IDS[stage],'run_manifest_sha256':run_sha,'request_sha256':sha,
             'source':source,'runtime':env,'config':config,'input_producers':proof,'test_read':False,
             'subset_four_gate_acceptance':False,**resources,**details}
    if stage=='preflight':output.mkdir(parents=True,exist_ok=False)
    with (output/'stage-receipt.json').open('xb') as f:f.write(encoded(receipt))
    return receipt


def execute(stage,request,sha):
    from analysis.g_decode_timing import ContextSampler
    sampler=ContextSampler()
    try:return _execute(stage,request,sha,sampler)
    finally:sampler.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    for name in ('preflight','train','verify','prepare'):
        c=sub.add_parser(name);c.add_argument('--request',type=Path,required=True);c.add_argument('--request-sha256',required=True)
        if name=='prepare':c.add_argument('--stage',choices=tuple(core.RUN_IDS),required=True);c.add_argument('--ops-code',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='dry':value={'status':'dry','training':False,'seed':42,'hits_gate':378,'test_read':False,'subset_four_gate_acceptance':False}
    elif args.command=='prepare':value=prepare(args.stage,args.request,args.request_sha256,args.ops_code)
    else:value=execute(args.command,args.request,args.request_sha256)
    print(json.dumps(value,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
