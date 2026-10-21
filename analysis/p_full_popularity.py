"""Frozen original Popularity on full UR001 validation; no training or test reader.

Uses a separate backend-shaped request and exclusive Popularity run IDs/output.
The request's train key names the evaluation job only, never model training.
"""
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import sys
import numpy as np
from analysis import p_full_training_backend as backend

MODULE='analysis.p_full_popularity'
STAGES={'preflight':'preflight','evaluate':'train','verify':'verify'}
KS=(10,20,50)


def train_events(paths,check=lambda:None,parquet=None):
    if parquet is None:
        from pyarrow.parquet import ParquetFile
        parquet=ParquetFile
    for path in paths:
        backend.stage_progress('popularity_train_week_begin',week=Path(path).stem)
        for batch in parquet(path).iter_batches(batch_size=65536,columns=['item_id','like'],use_threads=False):
            check()
            for item,like in zip(batch.column(0).to_pylist(),batch.column(1).to_pylist()):
                yield item,bool(like)
        backend.stage_progress('popularity_train_week_complete',week=Path(path).stem)


def evaluate(native,catalog,queries,supported,events):
    baselines=importlib.import_module(native.__package__+'.baselines')
    runner=importlib.import_module(native.__package__+'.baseline_run')
    counts=baselines.popularity(catalog,events)
    ids=np.asarray(catalog)
    # Original 0ca61bc baseline_run.evaluate_inputs ranking, identical tie order.
    pop=ids[np.lexsort((ids,-counts))[:min(max(KS),len(catalog))]].tolist()
    predictions=[list(pop) for _ in queries]
    result=runner.evaluate_predictions(predictions,queries,catalog,KS)
    slices={}
    for name in ('supported','unsupported','oov'):
        positions=[j for j,q in enumerate(queries) if ('oov' if q['oov'] else 'supported' if q['target'] in supported else 'unsupported')==name]
        slices[name]=baselines.metrics([predictions[j] for j in positions],[queries[j] for j in positions],catalog,KS)['all']
    return {'predictions':predictions,'metrics':result,'support_metrics':slices,
            'counts_sha256':hashlib.sha256(counts.astype('<i8',copy=False).tobytes()).hexdigest()}


def invocation(stage,inputs,request,digest):
    result=[sys.executable,'-B','-m',MODULE,stage,'--inputs',str(inputs),'--request',str(request),'--request-sha256',digest]
    if stage!='preflight':result.append('--execute-approved-science')
    return result


def transport(inputs,request,digest):
    value=backend.read_transport(inputs,request,digest)
    if value['predecessor'] is not None or any(not rid.startswith('p-full-popularity-') for rid in value['run_ids'].values()):
        raise ValueError('separate Popularity request and run IDs with null backend predecessor required')
    return value


def receipt_path(value,stage):return Path(value['output_root'])/('popularity-'+stage+'.json')
def run_path(value,stage):return Path(value['ops_root'])/'runs'/value['run_ids'][STAGES[stage]]


def input_records(value,inputs,request,digest,stage):
    records=backend.stage_inputs(value,inputs,request,digest,'preflight')
    def add(path):
        path=backend.checked_path(str(path))
        records.append({'path':str(path),'sha256':backend.file_sha(path),'bytes':path.stat().st_size})
    for prior in (('preflight',) if stage=='evaluate' else ('preflight','evaluate') if stage=='verify' else ()):
        for name in ('run.json','terminal.json','logs/stdout.log'):add(run_path(value,prior)/name)
        add(receipt_path(value,prior))
    if stage=='verify':
        add(Path(value['output_root'])/'popularity-predictions.json')
        add(Path(value['output_root'])/'popularity-metrics.json')
    if len({r['path'] for r in records})!=len(records):raise ValueError('unique Popularity role pins required')
    return records


def job_contract(value,records,request,digest,stage):
    expected=backend.job_contract(value,records,request,digest,STAGES[stage])
    expected['argv']=invocation(stage,records[1]['path'],request,digest)
    return expected


def check_job(value,records,request,digest,stage,*,actual_environment=True):
    expected=job_contract(value,records,request,digest,stage)
    raw=backend.checked_path(str(run_path(value,stage)/'run.json')).read_bytes();actual=backend.strict_json(raw)
    if (type(actual)is not dict or set(actual)-set(expected)-{'created_at','admission'}
            or any(backend.encoded(actual.get(k))!=backend.encoded(v) for k,v in expected.items())):
        raise ValueError('Popularity own complete job contract differs')
    if actual_environment and any(os.environ.get(k)!=v for k,v in expected['env'].items()):raise ValueError('Popularity actual environment differs')
    for rec in records:
        path=backend.record_path({'path':rec['path'],'sha256':rec['sha256']})
        if path.stat().st_size!=rec['bytes']:raise ValueError('Popularity role byte count differs')
    return hashlib.sha256(raw).hexdigest()


def plan(stage,inputs,request,digest):
    value=transport(inputs,request,digest);records=input_records(value,inputs,request,digest,stage)
    expected=job_contract(value,records,request,digest,stage);resources=expected['resources']
    argv=['--root',value['ops_root'],'--run-id',expected['run_id'],'--kind','ops','--cwd',str(backend.ROOT),'--config',str(request),'--seed',str(expected['seed'])]
    for k,v in value['job_contract'][STAGES[stage]]['env'].items():argv+=['--env',k+'='+v]
    for k in ('cpu_cores','ram_mib','vram_mib','disk_gib'):argv+=['--'+k.replace('_','-'),str(resources[k])]
    argv+=['--wall-hours',str(resources['wall_seconds']/3600),'--stop-mode',expected['stop_mode']]
    for rec in records:argv+=['--input',rec['path']]
    argv+=['--',*expected['argv']]
    return {'status':'planned_dry','training':False,'payload_parsed':False,'submitted':False,'stage':stage,'expected_job':expected,'launcher_argv':argv,'inputs':records}



def read_projection(root,manifest):
    values=[]
    for name in ('UR001-catalog.json','UR001-validation.jsonl','UR001-supported.json'):
        path=backend.checked_path(str(root/name));expected=manifest['artifacts'][name]
        if name.endswith('.jsonl'):
            digest=hashlib.sha256();size=0;value=[]
            with path.open('rb') as stream:
                for line in stream:
                    digest.update(line);size+=len(line);value.append(backend.strict_json(line))
            actual=digest.hexdigest()
        else:
            raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest();size=len(raw);value=backend.strict_json(raw)
        if actual!=expected['sha256'] or size!=expected['size']:raise ValueError('Popularity consumed projection bytes differ')
        values.append(value)
    return tuple(values)


def context(value,inputs):
    """Authenticate complete DAG, then parse only catalog/validation/support."""
    from analysis.ur001_derive_cli import source_identity,verify_dag
    from analysis.new_instance_prepare import runtime
    from analysis.producer_dag_binding import verify_producer,check_consumer
    source=source_identity();env=runtime(backend.record_path(value['lock']))
    check_consumer(source,value['source'],env,value['runtime'])
    root=Path(value['stage2'])
    proof=verify_producer(root/'UR001-manifest.json',Path(value['stage2_run']),value['producer_pin'])
    verification=verify_dag(root,require_profile='full')
    native=backend.native_for(Path(value['native_root']))
    config=native.validate_config(value['config'],synthetic=False)
    if backend.encoded(config)!=backend.encoded({**native.default_config(),'seed':config['seed']}):raise ValueError('original frozen mean config required')
    if type(value['disk_cap_bytes'])is not int or value['disk_cap_bytes']<=0:raise ValueError('positive frozen Popularity disk cap required')
    guard=native.b1.resource_guard(Path(value['output_root']),value['disk_cap_bytes']);guard()
    # Source ledger includes original baseline_run and baselines bytes; no model,
    # history-side, training JSONL or embedding payload is loaded here.
    manifest_raw=(root/'UR001-manifest.json').read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest()!=verification['manifest_sha256']:raise ValueError('Popularity consumed producer manifest differs')
    manifest=backend.strict_json(manifest_raw)
    if backend.encoded(manifest['counts'])!=backend.encoded(verification['counts']):raise ValueError('Popularity producer counts differ')
    catalog,queries,supported=read_projection(root,manifest)
    counts=verification['counts'];catalog_set=set(catalog)
    if (catalog!=sorted(catalog_set) or any(type(i)is not int or not 0<=i<2**32 for i in catalog)
            or len(catalog)!=counts['catalog_items'] or supported!=sorted(set(supported)) or not set(supported)<=catalog_set
            or len(queries)!=counts['eligible']):raise ValueError('Popularity full catalog/query/support count differs')
    users=set()
    for q in queries:
        if (type(q['user_id'])is not int or q['user_id'] in users or type(q['target'])is not int
                or q['oov']is not (q['target']not in catalog_set) or not 1<=len(q['history'])<=50
                or any(i not in catalog_set for i in q['history'])
                or q['target_position'][0]!=25 or q['history_positions'][-1]>=q['target_position']):raise ValueError('Popularity full validation query contract differs')
        users.add(q['user_id'])
    admitted=backend.strict_json(Path(inputs).read_bytes())
    binding={'ur001':verification,'producer':proof,'consumer_source':source,'consumer_runtime':env,'test_gate':'closed_hash_only'}
    backend.close_context(value,binding)
    return native,catalog,queries,set(supported),admitted['train'],binding,guard


def accepted(value,inputs,request,digest,stage,binding):
    records=input_records(value,inputs,request,digest,stage)
    job_sha=check_job(value,records,request,digest,stage,actual_environment=False)
    receipt=backend.strict_json(receipt_path(value,stage).read_bytes())
    run=run_path(value,stage);terminal=backend.strict_json((run/'terminal.json').read_bytes())
    lines=[x for x in (run/'logs/stdout.log').read_bytes().splitlines() if x.strip()]
    expected={'schema':'p-full-popularity-receipt-v1','status':'verified','verified':True,'stage':stage,'training':False,
              'engineering':False,'test_read':False,'run_id':value['run_ids'][STAGES[stage]],'run_manifest_sha256':job_sha,
              'request_sha256':digest,'source':value['source'],'runtime':value['runtime'],'binding':binding,
              'validation_queries':binding['ur001']['counts']['eligible'],'catalog_items':binding['ur001']['counts']['catalog_items']}
    if (terminal.get('schema')!='vkgr-terminal-v1' or terminal.get('run_id')!=expected['run_id'] or terminal.get('status')!='succeeded'
            or backend.encoded(terminal.get('exit_code'))!=b'0' or terminal.get('signal')is not None or terminal.get('wall_cap_exceeded')is not False or not lines
            or backend.encoded(backend.strict_json(lines[-1]))!=backend.encoded(receipt)
            or any(backend.encoded(receipt.get(k))!=backend.encoded(v) for k,v in expected.items())):raise ValueError('verified original Popularity producer stdout/job required')
    if stage=='preflight' and (receipt.get('scientific_verified')is not False or receipt.get('payload_written')is not False or receipt.get('metrics_computed')is not False or 'metrics'in receipt
            or backend.encoded(receipt.get('raw_train_events'))!=backend.encoded(binding['ur001']['counts']['train_rows'])
            or backend.encoded(receipt.get('raw_train_like_events'))!=backend.encoded(binding['ur001']['counts']['train_like_events'])):raise ValueError('read-only Popularity PF required')
    if stage=='evaluate':
        if receipt.get('scientific_verified')is not False or receipt.get('payload_written')is not True or set(receipt.get('artifacts',{}))!={'popularity-predictions.json','popularity-metrics.json'}:raise ValueError('Popularity evaluation artifact contract differs')
        for name,pin in receipt['artifacts'].items():backend.record_path({'path':str(Path(value['output_root'])/name),'sha256':pin})
    return receipt


def write_new(path,value):
    with path.open('xb') as stream:stream.write(backend.encoded(value)+b'\n')


def execute(stage,inputs,request,digest):
    value=transport(inputs,request,digest);records=input_records(value,inputs,request,digest,stage)
    own=check_job(value,records,request,digest,stage)
    native,catalog,queries,supported,train,binding,guard=context(value,inputs)
    output=backend.checked_path(value['output_root'])
    for root in (backend.ROOT,Path(value['native_root']),Path(value['stage2']),Path(value['ops_root'])):
        if output==root or root in output.parents or output in root.parents:raise ValueError('exclusive Popularity output outside immutable trees required')
    if receipt_path(value,stage).exists():raise FileExistsError('exclusive Popularity phase receipt required')
    if stage!='preflight':accepted(value,inputs,request,digest,'preflight',binding)
    result={'schema':'p-full-popularity-receipt-v1','status':'verified','verified':True,'stage':stage,'training':False,
            'engineering':False,'test_read':False,'run_id':value['run_ids'][STAGES[stage]],'run_manifest_sha256':own,
            'request_sha256':digest,'source':value['source'],'runtime':value['runtime'],'binding':binding,
            'scientific_verified':stage=='verify','payload_written':stage=='evaluate','validation_queries':len(queries),'catalog_items':len(catalog)}
    tally={'events':0,'likes':0}
    def counted_events():
        for item,liked in train_events(train,guard):
            tally['events']+=1;tally['likes']+=liked
            yield item,liked
    if stage=='preflight':
        # Stream the exact two-column physical path, without counts/ranks/metrics.
        events=likes=0
        for _,liked in counted_events():events+=1;likes+=liked
        result.update(raw_train_events=events,raw_train_like_events=likes,metrics_computed=False)
    else:
        prior=accepted(value,inputs,request,digest,'evaluate',binding) if stage=='verify' else None
        evaluated=evaluate(native,catalog,queries,supported,counted_events())
        predictions=evaluated.pop('predictions')
        if stage=='evaluate':
            if (output/'popularity-predictions.json').exists() or (output/'popularity-metrics.json').exists():raise FileExistsError('exclusive Popularity artifacts required')
            output.mkdir(parents=True,exist_ok=True)
            write_new(output/'popularity-predictions.json',predictions);write_new(output/'popularity-metrics.json',evaluated)
            result['artifacts']={name:backend.file_sha(output/name) for name in ('popularity-predictions.json','popularity-metrics.json')}
        else:
            if (backend.encoded(backend.strict_json((output/'popularity-predictions.json').read_bytes()))!=backend.encoded(predictions)
                    or backend.encoded(backend.strict_json((output/'popularity-metrics.json').read_bytes()))!=backend.encoded(evaluated)
                    or any(backend.encoded(prior.get(k))!=backend.encoded(v) for k,v in evaluated.items())):raise ValueError('Popularity original full replay differs')
            result['artifacts']=prior['artifacts'];result['verification']='original full train counts/ranks/validation metrics replay'
        result.update(evaluated)
    if tally!=dict(events=binding['ur001']['counts']['train_rows'],likes=binding['ur001']['counts']['train_like_events']):raise ValueError('Popularity raw train event/like conservation differs')
    guard();backend.close_context(value,binding)
    if stage!='preflight':accepted(value,inputs,request,digest,'preflight',binding)
    if stage=='verify':accepted(value,inputs,request,digest,'evaluate',binding)
    if (backend.encoded(transport(inputs,request,digest))!=backend.encoded(value)
            or backend.encoded(input_records(value,inputs,request,digest,stage))!=backend.encoded(records)
            or check_job(value,records,request,digest,stage)!=own):raise ValueError('Popularity transport/job closure changed')
    output.mkdir(parents=True,exist_ok=True);write_new(receipt_path(value,stage),result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    for name in ('plan',*STAGES):
        p=sub.add_parser(name);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True)
        if name=='plan':p.add_argument('--stage',choices=STAGES,required=True)
        elif name!='preflight':p.add_argument('--execute-approved-science',action='store_true',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='dry':result={'status':'dry','training':False,'real_preflight':False,'test_read':False,'submitted':False}
        elif args.command=='plan':result=plan(args.stage,args.inputs,args.request,args.request_sha256)
        else:result=execute(args.command,args.inputs,args.request,args.request_sha256)
    except (backend.InputRejected,ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        backend.stage_progress('popularity_rejected',error_type=type(exc).__name__,message=str(exc))
        result={'status':'HOLDOUT_INPUT_REJECTED' if isinstance(exc,backend.InputRejected) and exc.holdout else 'REJECTED','data_opened':False if isinstance(exc,backend.InputRejected) else 'UNKNOWN','training':False,'verified':False}
        print(backend.encoded(result).decode());return 2
    print(backend.encoded(result).decode());return 0


if __name__=='__main__':raise SystemExit(main())
