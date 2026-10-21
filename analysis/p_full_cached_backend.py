"""Explicit read-only sealed-cache acquisition for unchanged frozen P science.

The original PF is a cache producer, never the predecessor of this consumer's
training. A new real PF authenticates the dual source closure before training.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
from analysis import p_full_training_backend as backend
from analysis import p_full_compact_inputs as compact

ROOT=Path(__file__).resolve().parents[1]
READER_ENTRY_SHA='199941268d0e426d6e460af1fd9e5e8529de15e5bf399c5715dfe2ab9e799e02'
SCIENTIFIC_KEYS=('lock','native_root','stage2','stage2_run','producer_pin',
                 'items_metadata','embedding','config','admission_sha256','disk_cap_bytes')


def same_scientific_inputs(current,original):
    if any(k not in current or k not in original or backend.encoded(current[k])!=backend.encoded(original[k]) for k in SCIENTIFIC_KEYS):
        raise ValueError('sealed producer and new consumer scientific inputs differ')


def data_view(cached,catalog,supported,features):
    import numpy as np
    import torch
    from vkgr.retrieval_data import catalog_map
    idmap=catalog_map(catalog)
    if supported!=sorted(set(supported))or not set(supported)<=set(catalog):raise ValueError('original support differs')
    if (features.dtype!=np.float32 or features.shape!=(len(catalog)+1,64)
        or not np.isfinite(features).all()or np.any(features[0])):raise ValueError('original content differs')
    return {'rows':cached['rows'],'queries':cached['queries'],'catalog':catalog,'idmap':idmap,
            'content':torch.from_numpy(features),'support':set(supported),
            'supported_indices':[idmap[i]for i in supported],'synthetic':False}


def reader(reference,reference_sha):
    from analysis.sgns_subset_acceptance import source_tree
    record={'path':str(reference),'sha256':reference_sha}
    value=backend.strict_json(backend.record_path(record).read_bytes())
    if type(value)is not dict or set(value)!={'schema','root','source','seal'}or value['schema']!='p-full-sealed-cache-reference-v1':
        raise ValueError('closed external sealed-cache reference required')
    root=backend.checked_path(value['root']);source_tree(root,value['source'])
    entry=root/'analysis/p_pf_cache_reuse.py'
    if (backend.file_sha(entry)!=READER_ENTRY_SHA
        or value['source']['files'].get('analysis/p_pf_cache_reuse.py')!=READER_ENTRY_SHA):raise ValueError('fixed cache reader differs')
    spec=importlib.util.spec_from_file_location('analysis._p_full_sealed_cache_reader',entry)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source_tree(root,value['source']);backend.record_path(record)
    return module,value


def origin(reference):
    path=backend.record_path(reference['seal']['manifest'])
    receipt=backend.strict_json(path.read_bytes())
    old=backend.strict_json(backend.record_path(receipt['original_pf']['request']).read_bytes())
    return receipt,old


def records(reference,reference_sha):
    module,value=reader(reference,reference_sha);receipt,_=origin(value)
    result=[{'path':str(reference),'sha256':reference_sha}]
    root=Path(value['root'])
    result.extend({'path':str(root/name),'sha256':sha}for name,sha in sorted(value['source']['files'].items()))
    seal=value['seal'];result.append(seal['manifest'])
    result.extend({'path':str(Path(seal['run_dir'])/name),'sha256':sha}for name,sha in sorted(seal['pin']['job_sha256'].items()))
    job=backend.strict_json(backend.record_path({'path':str(Path(seal['run_dir'])/'run.json'),
                          'sha256':seal['pin']['job_sha256']['run.json']}).read_bytes())
    result.extend(job['inputs'])
    result.extend(receipt['inputs'].values())
    result.extend({'path':str(Path(receipt['cache_root'])/name),'sha256':item['sha256']}for name,item in receipt['semantic']['files'].items())
    # Reuse canonical records exactly; duplicate paths must agree.
    merged={}
    for item in result:
        path=backend.checked_path(item['path']);sha=item['sha256'];old=merged.get(str(path))
        record={'path':str(path),'sha256':sha,'bytes':path.stat().st_size}
        if old is not None and backend.encoded(old)!=backend.encoded(record):raise ValueError('conflicting external input roles')
        merged[str(path)]=record
    return list(merged.values())


def functions(reference,reference_sha):
    namespace=dict(vars(backend));namespace['ROOT']=ROOT
    def invocation(stage,inputs,request,pin):
        args=[sys.executable,'-B','-m','analysis.p_full_cached_backend',stage,
              '--inputs',str(inputs),'--request',str(request),'--request-sha256',pin,
              '--cache-reference',str(reference),'--cache-reference-sha256',reference_sha]
        if stage!='preflight':args.append('--execute-approved-science')
        return args
    def stage_inputs(value,inputs,request,pin,stage):
        original=backend.stage_inputs(value,inputs,request,pin,stage)
        merged={item['path']:item for item in original}
        for item in records(reference,reference_sha):
            prior=merged.get(item['path'])
            if prior is not None and backend.encoded(prior)!=backend.encoded(item):raise ValueError('cache and original input roles conflict')
            merged[item['path']]=item
        return list(merged.values())
    def read_context(inputs,request,pin):
        # Original holdout/path admission is always first, before cache access.
        value=backend.read_transport(inputs,request,pin)
        module,ref=reader(reference,reference_sha);receipt,old=origin(ref)
        same_scientific_inputs(value,old)
        native=backend.native_for(Path(value['native_root']))
        guard=compact.CombinedGuard(native.b1.resource_guard(Path(value['output_root']),value['disk_cap_bytes']),
                                    [Path(value['output_root']),Path(receipt['cache_root']),compact.cache_root(value['output_root']),
                                     Path(value['output_root']).parent/(Path(value['output_root']).name+'-parallel-eval')],value['disk_cap_bytes'])
        cached=module.consume(ref['seal'],guard)
        from analysis.ur001_derive_cli import source_identity,verify_dag
        from analysis.producer_dag_binding import verify_producer
        from analysis.new_instance_prepare import runtime
        from analysis.sgns_subset_acceptance import source_tree
        source=source_identity();env=runtime(backend.record_path(value['lock']))
        source_tree(ROOT,value['source'])
        if source!=value['source']or backend.encoded(env)!=backend.encoded(value['runtime']):raise ValueError('new consumer source/runtime differs')
        root=Path(value['stage2']);verified=verify_dag(root,require_profile='full')
        producer=verify_producer(root/'UR001-manifest.json',Path(value['stage2_run']),value['producer_pin'])
        prior=cached['scientific_binding']
        if (prior.get('ur001')!=verified or prior.get('producer')!=producer
            or prior.get('consumer_source')!=old['source']or prior.get('consumer_runtime')!=old['runtime']
            or prior.get('test_gate')!='closed_hash_only'):raise ValueError('original scientific PF binding differs')
        config=native.validate_config(value['config'],synthetic=False)
        if config!={**native.default_config(),'seed':config['seed']}or config['arm']!='mean':raise ValueError('frozen M1-mean configuration differs')
        import numpy as np
        from vkgr.retrieval_data import content_join
        catalog=json.loads((root/'UR001-catalog.json').read_bytes())
        supported=json.loads((root/'UR001-supported.json').read_bytes())
        with np.load(backend.record_path(value['embedding']),allow_pickle=False)as features:
            content=content_join(catalog,features['item_id'],features['embedding'])
        data=data_view(cached,catalog,supported,content)
        counts=verified['counts']
        if len(data['rows'])!=counts['train_examples']or len(data['queries'])!=counts['eligible']:raise ValueError('sealed row count differs')
        binding={'ur001':verified,'producer':producer,'consumer_source':source,'consumer_runtime':env,
                 'test_gate':'closed_hash_only','sealed_cache_reader':cached['reader_identity']}
        guard();backend.stage_progress('sealed_context_complete')
        facade=compact.facade(native,guard);facade.sealed_resource_guard=guard
        return value,facade,data,binding
    def close_context(value,binding):
        backend.close_context(value,binding)
        module,ref=reader(reference,reference_sha)
        cached=module.consume(ref['seal'],None)
        if cached['reader_identity']!=binding['sealed_cache_reader']:raise ValueError('sealed cache closure changed during science')
    namespace.update(invocation=invocation,stage_inputs=stage_inputs,read_context=read_context,close_context=close_context)
    for name in ('job_contract','check_job','plan'):
        f=getattr(backend,name);clone=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
        clone.__kwdefaults__=f.__kwdefaults__;namespace[name]=clone
    f=cached_execute;namespace['execute']=FunctionType(f.__code__,{**f.__globals__,**namespace},f.__name__,f.__defaults__,f.__closure__)
    return namespace


def sealed_parallel(native,output,cap,guard):
    from analysis import p_full_parallel_eval as parallel
    # Keep the original scoring function; add only the shared allocation check.
    def vectors(*args,**kwargs):
        original_check=kwargs.get('disk_check')
        def projected(additional=0):
            guard.projected(additional)
            if original_check:original_check(additional)
        kwargs['disk_check']=projected
        return parallel.evaluate_vectors(*args,**kwargs)
    original=parallel.bind
    namespace={**original.__globals__,'evaluate_vectors':vectors}
    bind=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    result=bind(native,output.parent/(output.name+'-parallel-eval'),output_root=output,disk_cap_bytes=cap)
    base_guard=result.b1.resource_guard
    def resource_guard(path,budget):
        original_guard=base_guard(path,budget)
        def check():original_guard();guard.projected(0)
        return check
    result.b1=SimpleNamespace(**{**vars(result.b1),'resource_guard':resource_guard})
    return result


def cached_execute(stage, inputs, request, request_sha):
    transport = read_transport(inputs, request, request_sha)
    records = stage_inputs(transport, inputs, request, request_sha, stage)
    run_sha = check_job(transport, records, request, request_sha, stage)
    value, native, data, binding = read_context(inputs, request, request_sha)
    native = sealed_parallel(native.compact_original, Path(value['output_root']),
                             value['disk_cap_bytes'],native.sealed_resource_guard)
    from analysis import p_full_compact_inputs as compact
    native = compact.facade(native, compact.make_guard(native, value))
    binding = {**binding, 'parallel_evaluation': native.parallel_eval_identity}
    if encoded(value) != encoded(transport):
        raise ValueError('request changed before scientific context')
    predecessor = check_predecessor(stage, value, binding)
    run_id = value['run_ids'][stage]
    native.b1.execution_context('cuda')
    output = checked_path(value['output_root'])
    for protected in (ROOT, Path(value['native_root']), Path(value['stage2'])):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('exclusive output separate from source/input required')
    guard = native.b1.resource_guard(output, value['disk_cap_bytes'])
    guard()
    if stage == 'preflight':
        receipt = {'schema': 'p-full-backend-receipt-v1', 'status': 'verified', 'verified': True,
                'stage': stage, 'scientific_verified': False, 'source': value['source'], 'runtime': value['runtime'],
                'engineering': False, 'training': False, 'payload_written': False,
                'engineering_cache_written': True,
                'test_read': False, 'data_opened': True, 'run_id': run_id,
                'request_sha256': request_sha, 'binding': binding,
                'training_rows': len(data['rows']), 'validation_queries': len(data['queries']),
                'assembly_workers': 0, 'native_scientific_functions_unchanged': True}
        close_context(value, binding)
        if check_job(value, records, request, request_sha, stage) != run_sha:
            raise ValueError('own job changed during preflight')
        receipt['run_manifest_sha256'] = run_sha
        write_stage_receipt(output, stage, receipt)
        return receipt
    config = value['config']
    if stage == 'train':
        result = run_session(native, output, data, config, 'cuda', check=guard, inputs_binding=binding,
                             expected_rows=binding['ur001']['counts']['train_examples'])
        details = {'status': result['status'], 'verified': False, 'training': True,
                   'manifest_sha256': result['manifest_sha256']}
    else:
        manifest = output / 'manifest.json'
        result = native.verify_run(manifest, predecessor['native_manifest_sha256'], data, config, 'cuda', check=guard, inputs_binding=binding)
        details = {'status': result['status'], 'verified': result['status'] == 'verified', 'training': False,
                   'native_verification': result, 'manifest_sha256': file_sha(manifest)}
    close_context(value, binding)
    check_predecessor(stage, value, binding)
    if check_job(value, records, request, request_sha, stage) != run_sha:
        raise ValueError('own job changed during science')
    guard()
    receipt = {'schema': 'p-full-backend-receipt-v1', 'engineering': False, 'test_read': False,
               'stage': stage, 'run_id': run_id, 'run_manifest_sha256': run_sha, 'request_sha256': request_sha, 'binding': binding,
               'source': value['source'], 'runtime': value['runtime'], **details,
               'native_manifest_sha256': details['manifest_sha256'],
               'scientific_verified': stage == 'verify', 'verified': True}
    write_stage_receipt(output, stage, receipt)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=p.add_subparsers(dest='stage',required=True);sub.add_parser('dry')
    for stage in ('plan','preflight','train','verify'):
        a=sub.add_parser(stage,allow_abbrev=False)
        for name in ('inputs','request','cache-reference'):a.add_argument('--'+name,type=Path,required=True)
        for name in ('request-sha256','cache-reference-sha256'):a.add_argument('--'+name,required=True)
        if stage=='plan':a.add_argument('--stage',dest='planned_stage',choices=('preflight','train','verify'),required=True)
        elif stage!='preflight':a.add_argument('--execute-approved-science',action='store_true',required=True)
    a=p.parse_args(argv)
    try:
        if a.stage=='dry':result={'status':'dry','training':False,'test_read':False,'sealed_cache':True}
        else:
            f=functions(a.cache_reference,a.cache_reference_sha256)
            result=f['plan'](a.planned_stage,a.inputs,a.request,a.request_sha256)if a.stage=='plan'else f['execute'](a.stage,a.inputs,a.request,a.request_sha256)
    except Exception as error:
        backend.stage_progress('sealed_entry_failed',error_type=type(error).__name__,message=str(error))
        print(json.dumps({'status':'REJECTED','verified':False,'test_read':False,'error_type':type(error).__name__}));return 2
    print(json.dumps(result,sort_keys=True,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
