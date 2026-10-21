"""UR001 G A1 transport adapter; original scientific function bytecode unchanged.

The dry receipt authenticates only raw input roles. It is never a science
preflight receipt. No existing subset SID or CONTENT_V1 identity is reused.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType, MappingProxyType

from analysis.p_full_compact_inputs import JsonRows
from analysis.producer_dag_binding import check_consumer, verify_producer
from analysis.ur001_science_entry import InputRejected, admit_inputs, checked_path

ROOT=Path(__file__).resolve().parents[1]
SID_VERSION='UR001-BHKM-B229D3-S42'
CONFIG_HASHES={42:'9438c9574e687496e42d3ac2453c649fd7d94adc9e51371e4a14aa96d92c69d5',2024:'7121cb5203d16dd8b909bbf0662a559c0e34a006d5dae1dfcd08b654159bceb7',2025:'0d75b877119e90a0b7fd6ba05b4b8e1850af0a3306d52bc45867f9fafd22b7a4'}


def pin(value):
    if type(value)is not str or len(value)!=64 or set(value)-set('0123456789abcdef') or value=='0'*64:
        raise ValueError('explicit nonzero SHA256 required')
    return value


def registered_config(seed,sid_sha):
    """Original registered batching/model/seed; substitute data identity only."""
    if type(seed)is not int or seed not in CONFIG_HASHES:raise ValueError('registered seed required')
    raw=(ROOT/f'configs/sid_structure/g-structure-a1-s{seed}.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CONFIG_HASHES[seed]:raise ValueError('original frozen config bytes differ')
    value=json.loads(raw)
    value.update(sid=SID_VERSION,sid_manifest_sha256=pin(sid_sha))
    return value


def validate_config(value):
    if type(value)is not dict:raise ValueError('registered config required')
    expected=registered_config(value.get('seed'),value.get('sid_manifest_sha256'))
    if set(value)!=set(expected) or any(type(value[k])is not type(v) or value[k]!=v for k,v in expected.items()):
        raise ValueError('frozen A1 scientific config differs')
    return value


def require_final_sid(record):
    # Legacy role-only entry cannot stand in for the formal request/producer.
    raise ValueError('formal request and original final full SID/trie producer required')


class LazyExamples:
    def __init__(self,view):self.view=view
    def __len__(self):return len(self.view)
    def rows(self,indices):
        import numpy as np
        indices=list(indices)
        if any((type(i)is not int and not isinstance(i,np.integer)) or isinstance(i,(bool,np.bool_)) or not 0<=i<len(self) for i in indices):
            raise ValueError('example indices out of range or noninteger')
        return [self.view[int(i)] for i in indices]


def lazy_examples(path,offsets,count,expected_sha256,guard):
    """Storage-only reuse; original row dictionaries/order reach train_batch_d."""
    path=checked_path(str(path));offsets=checked_path(str(offsets))
    return LazyExamples(JsonRows.build(path,offsets,count,pin(expected_sha256),guard))


def slice_statistics(data,*,catalog,queries,manifest_sha256,statistics_sha256,binding_sha256):
    """Adapt the unchanged UR001 S0 definition to the original exact stats type."""
    from vkgr import generative_evaluation as ev
    keys={'schema_version','popularity_thresholds','item_like_counts','user_like_counts','user_buckets','definition'}
    if (type(data)is not dict or set(data)!=keys or type(data['schema_version'])is not int or data['schema_version']!=1
            or data['user_buckets']!=[1,5,20] or any(type(x)is not int for x in data['user_buckets'])
            or data['definition']!='train_like_counts_only_no_holdout_boundaries'):
        raise ValueError('original S0 schema/definition required')
    catalog=list(map(int,catalog))
    if not catalog or catalog!=sorted(set(catalog)):raise ValueError('ordered unique original catalog required')
    items=ev._counts(data['item_like_counts']);users=ev._counts(data['user_like_counts'],positive=True)
    if set(items)!=set(catalog):raise ValueError('complete catalog counts required')
    counts=sorted(v for v in items.values() if v>0)
    if not counts or not users:raise ValueError('positive train statistics required')
    expected={'median':counts[math.ceil(len(counts)*.5)-1],'p90':counts[math.ceil(len(counts)*.9)-1]}
    thresholds=data['popularity_thresholds']
    if (type(thresholds)is not dict or set(thresholds)!=set(expected)
            or any(type(thresholds[k])is not int or thresholds[k]!=v for k,v in expected.items())):
        raise ValueError('original nearest-rank thresholds differ')
    result=ev.SliceStatistics((expected['median'],expected['p90']),MappingProxyType(items),MappingProxyType(users),
        pin(manifest_sha256),pin(statistics_sha256),pin(binding_sha256))
    ev._query_statistics(queries,result,set(catalog))
    return result


def diagnostic_context(examples,codec,queries,statistics,producer_binding):
    """In-memory parity fixture only, never a formal producer acceptance path."""
    from vkgr import generative_experiment as ge
    codec.check_identity()
    binding={'b1':producer_binding,'synthetic':True,'engineering':True,
             'queries_sha256':ge._digest(queries),'catalog_sha256':ge._digest(list(map(int,codec.catalog.ids)))}
    return ge.RunContext(ge._CONTEXT_KEY,examples=examples,codec=codec,queries=queries,binding=binding,stats=statistics).check()


def native_functions(*,context=None,sid=None,binding=None):
    """Original run/verify bytecode, with authenticated UR001 input acquisition.

    Model, optimizer, cursor, training and evaluation objects remain original.
    Without a factory-issued context the scientific input path rejects.
    """
    from vkgr.sgns_subset_a1 import check_native
    original=check_native()
    namespace=dict(original.__dict__)
    namespace['SID_VERSION']=SID_VERSION
    def contexts(*args,**kwargs):return require_final_sid(None)
    namespace['_contexts']=contexts
    if context is not None or sid is not None or binding is not None:
        if not isinstance(context,AuthenticatedContext) or sid is not context.sid or binding != context.proof:
            raise ValueError('authenticated UR001 context required')
        context.check()
        def contexts(*args,**kwargs):
            validate_config(args[9])
            if args[9]['sid_manifest_sha256']!=sid.info['manifest_sha256']:
                raise ValueError('SID/config binding differs')
            return context.context.check(),False,sid
        namespace['_contexts']=contexts
    for name in ('default_config','validate_config','_binding','run','verify'):
        value=getattr(original,name)
        clone=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
        clone.__kwdefaults__=value.__kwdefaults__
        namespace[name]=clone
    if context is not None:
        def metadata(root,ctx,config,runtime,model,prepared_sha,history_sha,loaded):
            if ctx is not context.context or loaded is not sid:
                raise ValueError('authenticated context identity differs')
            return {'task':original.TASK_ID,'budget':original.BUDGET,'init_version':'INIT_EMB_N002',
                    'config':config,'config_sha256':original._digest(config),'init_report':model.init_report,
                    'runtime':runtime,'UR001':binding,'synthetic':False,'test_gate':'closed'}
        namespace['_binding']=metadata
    return namespace



class AuthenticatedContext:
    """Issued only after original producer consumption and full UR001 validation."""
    def __init__(self,key,context,sid,proof):
        if key is not _FULL_KEY:raise ValueError('authenticated context factory required')
        self.context=context;self.sid=sid;self.proof=proof
    def check(self):
        self.context.check()
        if self.context.binding.get('b1')!=self.proof or self.sid.info['manifest_sha256']!=self.proof['sid']['manifest']['sha256']:
            raise ValueError('authenticated context identity differs')
        return self


_FULL_KEY=object()
REQUEST_SCHEMA='g-full-backend-request-v1'
SID_PRODUCER_ENTRY_SHA='3e7a1b1aa06d6e8b71269d0b702ad4e45a0ee1c5ec2f3be432f7db2bacc2876b'


def read_transport(inputs,request,request_sha):
    from analysis import p_full_training_backend as pb
    import re
    admission=admit_inputs(str(inputs))
    value=pb.strict_json(pb.record_path({'path':str(request),'sha256':pin(request_sha)}).read_bytes())
    fields={'schema','source','runtime','lock','stage2','stage2_run','producer_pin','sid_producer',
            'config','ops_root','output_root','scratch_root','run_ids','job_contract','admission_sha256','predecessor','p3c'}
    if type(value)is not dict or set(value)!=fields or value['schema']!=REQUEST_SCHEMA:
        raise ValueError('closed G backend request schema required')
    if value['admission_sha256']!=admission:raise ValueError('input admission bytes differ')
    for name in ('stage2','stage2_run','ops_root','output_root','scratch_root'):checked_path(value[name])
    for name in ('lock','p3c'):checked_path(value[name]['path']);pin(value[name]['sha256'])
    ids=value['run_ids']
    if (type(ids)is not dict or set(ids)!={'preflight','train','verify'} or len(set(ids.values()))!=3
        or any(type(v)is not str or re.fullmatch('[a-z0-9][a-z0-9._-]{0,127}',v)is None for v in ids.values())
        or type(value['job_contract'])is not dict or set(value['job_contract'])!=set(ids)):
        raise ValueError('three unique closed G job stages required')
    validate_config(value['config'])
    return value


def consume_sid(record):
    """Execute fixed producer consume in its own sealed tree; never invoke fit/replay."""
    import os,subprocess,sys
    from analysis import p_full_training_backend as pb
    from analysis.sgns_subset_acceptance import source_tree
    fields={'root','source','entry_sha256','receipt'}
    if type(record)is not dict or set(record)!=fields:raise ValueError('closed final full SID/trie producer required')
    root=checked_path(record['root'])
    if record['entry_sha256']!=SID_PRODUCER_ENTRY_SHA:raise ValueError('registered SID producer entry required')
    source_tree(root,record['source'])
    # Complete producer source ledger is frozen by the authenticated request;
    # the registered entry SHA is additionally fixed in this consumer source.
    if record['source']['files'].get('analysis/sgns_native_full_sid.py')!=pin(record['entry_sha256']):
        raise ValueError('fixed SID producer entry differs')
    receipt=pb.record_path(record['receipt'])
    if receipt.name!='stage-receipt.json' or receipt.parent.name!='verify':
        raise ValueError('independent SID verify receipt required')
    code=('import json;from analysis.sgns_native_full_sid import consume;'
          'print(json.dumps(consume('+repr(str(receipt))+','+repr(record['receipt']['sha256'])+'),sort_keys=True))')
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    env.update(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=root,env=env,capture_output=True)
    if result.returncode:raise ValueError('original fixed SID consume rejected')
    source_tree(root,record['source']);pb.record_path(record['receipt'])
    proof=pb.strict_json(result.stdout)
    if (type(proof)is not dict or set(proof)!={'manifest','producer','source','runtime','parent','codec'}
        or proof['source']!=record['source'] or proof['producer']['receipt_sha256']!=record['receipt']['sha256']):
        raise ValueError('original full SID producer proof differs')
    info=proof['codec']
    if (info.get('sid_version')!=SID_VERSION or type(info.get('branching'))is not int or info['branching']!=229
        or type(info.get('depth'))is not int or info['depth']!=3 or info.get('engineering')is not False or info.get('kind')!='bhkm'
        or info.get('manifest_sha256')!=proof['manifest']['sha256']):
        raise ValueError('frozen full K229 depth3 codec required')
    return proof


def full_context(value,guard,stage='preflight'):
    from analysis import p_full_training_backend as pb
    from analysis.ur001_derive_cli import verify_dag
    from vkgr import sid_v2d,sid_v2,generative_experiment as ge
    from types import SimpleNamespace
    import numpy as np
    root=Path(value['stage2'])
    dag=verify_dag(root,require_profile='full')
    upstream=verify_producer(root/'UR001-manifest.json',Path(value['stage2_run']),value['producer_pin'])
    manifest=pb.strict_json((root/'UR001-manifest.json').read_bytes())
    proof=consume_sid(value['sid_producer'])
    if proof['parent'].get('binding',{}).get('stage2_manifest_sha256')!=value['producer_pin']['manifest_sha256']:
        raise ValueError('SID fullB and G stage2 DAG differ')
    if proof['runtime']!=value['runtime']:raise ValueError('SID/G exact runtime differs')
    path=pb.record_path(proof['manifest']);sid_manifest=pb.strict_json(path.read_bytes())
    if (sid_manifest.get('schema')!='UR001-native-full-SID-trie-v1'
        or set(sid_manifest.get('artifacts',{}))!={'mapping.npz','trie.npz','nodes.json','audit.json','config.json'}):
        raise ValueError('closed final SID artifact contract required')
    for name,digest in sid_manifest['artifacts'].items():pb.record_path({'path':str(path.parent/name),'sha256':digest});guard()
    catalog=pb.strict_json((root/'UR001-catalog.json').read_bytes())
    with np.load(path.parent/'mapping.npz',allow_pickle=False) as data:
        if set(data.files)!={'ids','sids'}:raise ValueError('closed original SID mapping required')
        ids=data['ids'];sids=data['sids']
    codec=sid_v2d.build_codec_d(ids,sids,229,3,catalog_ids=catalog)
    if (sid_v2.catalog_digest(ids)!=proof['codec']['catalog_digest'] or proof['codec'].get('suffix_vocabulary')!=codec.catalog.suffix_vocabulary
        or proof['codec'].get('vocab_size')!=codec.vocab_size):raise ValueError('original canonical SID catalog/namespace differs')
    with np.load(path.parent/'trie.npz',allow_pickle=False) as trie:
        if set(trie.files)!={'keys','key_items','bases'}:raise ValueError('closed original trie required')
        for key in ('keys','key_items','bases'):
            if not np.array_equal(trie[key],getattr(codec.catalog,key)):raise ValueError('original trie order/bytes differ')
    cache=checked_path(value['scratch_root'])/value['run_ids'][stage]
    cache.mkdir(parents=True,exist_ok=False)
    train=manifest['artifacts']['UR001-train.jsonl']
    examples=lazy_examples(root/'UR001-train.jsonl',cache/'train-offsets.bin',manifest['counts']['train_examples'],train['sha256'],guard)
    queries=[pb.strict_json(line) for line in (root/'UR001-validation.jsonl').read_bytes().splitlines()]
    if len(queries)!=manifest['counts']['eligible']:raise ValueError('validation query count differs')
    proof_binding={'schema':'UR001-G-A1-full-context-v1','ur001':dag,'producer':upstream,'sid':proof,
                   'consumer_source':value['source'],'consumer_runtime':value['runtime'],'test_gate':'closed'}
    stats_record=manifest['artifacts']['UR001-S0.json']
    stats=slice_statistics(pb.strict_json((root/'UR001-S0.json').read_bytes()),catalog=catalog,queries=queries,
        manifest_sha256=value['producer_pin']['manifest_sha256'],statistics_sha256=stats_record['sha256'],binding_sha256=hashlib.sha256(pb.encoded(proof_binding)).hexdigest())
    # Only original UR001 dictionaries enter unchanged native batch functions.
    idset=set(catalog)
    for train_flag,rows in ((True,examples.view),(False,queries)):
        for ordinal,row in enumerate(rows):
            guard();h=row['history'];positions=row['history_positions'];target=row['target_position']
            if (not 1<=len(h)<=50 or len(h)!=len(positions) or any(i not in idset for i in h)
                or any(type(p)is not list or len(p)!=2 or any(type(v)is not int or v<0 for v in p) or p[0]>24 for p in positions)
                or any(a>=b for a,b in zip(positions,positions[1:])) or type(target)is not list or len(target)!=2
                or any(type(v)is not int or v<0 for v in target) or positions[-1]>=target
                or (train_flag and (row.get('example_id')!=ordinal or target[0]>24 or row['target']not in idset))
                or (not train_flag and (target[0]!=25 or row.get('oov')is not(row['target']not in idset)))):
                raise ValueError('UR001 original history/target identity differs')
    ctx=ge.RunContext(ge._CONTEXT_KEY,examples=examples,codec=codec,queries=queries,
        binding={'b1':proof_binding,'synthetic':False,'engineering':False,'queries_sha256':ge._digest(queries),'catalog_sha256':ge._digest(catalog)},stats=stats).check()
    sid=SimpleNamespace(info=proof['codec'],manifest=sid_manifest,codec=codec)
    guard()
    return AuthenticatedContext(_FULL_KEY,ctx,sid,proof_binding).check()


def invocation(stage,inputs,request,request_sha):
    import sys
    argv=[sys.executable,'-B','-m','analysis.g_full_training_backend',stage,'--inputs',str(inputs),'--request',str(request),'--request-sha256',request_sha]
    if stage!='preflight':argv.append('--execute-approved-science')
    return argv


def stage_inputs(value,inputs,request,request_sha,stage):
    from analysis import p_full_training_backend as pb
    result=[];seen={}
    def add(path,digest):
        path=checked_path(str(path));pin(digest)
        if str(path)in seen:
            if seen[str(path)]!=digest:raise ValueError('conflicting input role pins')
            return
        seen[str(path)]=digest
        result.append({'path':str(path),'sha256':digest,'bytes':path.stat().st_size})
    def metadata(path,digest):
        path=pb.record_path({'path':str(path),'sha256':digest});add(path,digest)
        return pb.strict_json(path.read_bytes())
    add(request,request_sha);add(inputs,value['admission_sha256']);add(value['lock']['path'],value['lock']['sha256'])
    from analysis.profile_anchor_steps import _source_pins
    if value['p3c']['sha256']!=_source_pins()['p']['p3c_data_sha256']:raise ValueError('fixed p3c data channel bytes required')
    add(value['p3c']['path'],value['p3c']['sha256'])
    root=Path(value['stage2'])
    manifest=metadata(root/'UR001-manifest.json',value['producer_pin']['manifest_sha256'])
    if set(manifest['upstream'])!={'UR001-input-manifest.json'}:raise ValueError('closed UR001 upstream required')
    physical=metadata(root/'UR001-input-manifest.json',manifest['upstream']['UR001-input-manifest.json'])
    names={'UR001-train.jsonl','UR001-validation.jsonl','UR001-catalog.json','UR001-supported.json','UR001-S0.json','UR001-counts.json'}
    if set(manifest['artifacts'])!=names:raise ValueError('closed UR001 artifact set required')
    for name in sorted(names):add(root/name,manifest['artifacts'][name]['sha256'])
    for name,digest in value['producer_pin']['job_sha256'].items():add(Path(value['stage2_run'])/name,digest)
    admitted=pb.strict_json(Path(inputs).read_bytes())
    for record,path in pb.physical_records(physical,admitted):add(path,record['sha256'])
    proof=consume_sid(value['sid_producer'])
    if value['config']['sid_manifest_sha256']!=proof['manifest']['sha256']:raise ValueError('fixed config SID bytes differ')
    metadata_record=value['sid_producer']['receipt']
    receipt=metadata(metadata_record['path'],metadata_record['sha256'])
    request_record=receipt['request'];sid_request=metadata(request_record['path'],request_record['sha256'])
    sid_root=Path(value['sid_producer']['root'])
    for name,digest in value['sid_producer']['source']['files'].items():add(sid_root/name,digest)
    for name,digest in proof['parent']['evidence'].items():add(name,digest)
    for prior in ('preflight','fit','verify'):
        run=Path(sid_request['ops_root'])/'runs'/sid_request[prior+'_run_id']
        job=pb.strict_json((run/'run.json').read_bytes())
        for name in ('run.json','terminal.json','logs/stdout.log'):add(run/name,pb.file_sha(run/name))
        for record in job['inputs']:add(record['path'],record['sha256'])
        r=Path(sid_request['output'])/prior/'stage-receipt.json';add(r,pb.file_sha(r))
    sidmanifest=metadata(proof['manifest']['path'],proof['manifest']['sha256'])
    for name,digest in sidmanifest['artifacts'].items():add(Path(proof['manifest']['path']).parent/name,digest)
    pred=value['predecessor']
    if stage=='preflight':
        if pred is not None:raise ValueError('preflight predecessor must be null')
    else:
        if type(pred)is not dict or set(pred)!={'manifest','run_dir','pin'}:raise ValueError('fixed predecessor closure required')
        prior='preflight' if stage=='train' else 'train'
        if Path(pred['manifest'])!=pb.stage_receipt_path(Path(value['output_root']),prior) or Path(pred['run_dir'])!=Path(value['ops_root'])/'runs'/value['run_ids'][prior]:
            raise ValueError('fixed predecessor paths differ')
        metadata(pred['manifest'],pred['pin']['manifest_sha256'])
        for name,digest in pred['pin']['job_sha256'].items():add(Path(pred['run_dir'])/name,digest)
        if stage=='verify':
            # Bind every saved epoch/evaluation byte that original verify reads.
            for path in sorted(Path(value['output_root']).rglob('*')):
                if path.is_file():add(path,pb.file_sha(path))
    return result


def transport_functions():
    # Reuse the existing closed launcher/environment/resource checks verbatim;
    # only the module argv and the G-specific authenticated input closure differ.
    from analysis import p_full_training_backend as pb
    namespace=dict(pb.__dict__)
    namespace.update(ROOT=ROOT,read_transport=read_transport,stage_inputs=stage_inputs,invocation=invocation)
    for name in ('job_contract','check_job','plan'):
        function=getattr(pb,name)
        namespace[name]=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    return namespace


def plan(stage,inputs,request,request_sha):
    return transport_functions()['plan'](stage,inputs,request,request_sha)


def identity(value):
    from analysis.sgns_subset_acceptance import source_tree
    from analysis.new_instance_prepare import runtime
    from analysis import p_full_training_backend as pb
    source_tree(ROOT,value['source'])
    check_consumer(value['source'],value['source'],runtime(pb.record_path(value['lock'])),value['runtime'])
    native_functions()
    return {'source':value['source'],'runtime':value['runtime']}


def predecessor(stage,value,binding):
    from analysis import p_full_training_backend as pb
    record=value['predecessor']
    if stage=='preflight':
        if record is not None:raise ValueError('preflight predecessor must be null')
        return None
    prior='preflight' if stage=='train' else 'train'
    path=pb.stage_receipt_path(Path(value['output_root']),prior)
    run=Path(value['ops_root'])/'runs'/value['run_ids'][prior]
    if type(record)is not dict or set(record)!={'manifest','run_dir','pin'} or checked_path(record['manifest'])!=path or checked_path(record['run_dir'])!=run:
        raise ValueError('fixed G predecessor closure required')
    proof=verify_producer(path,run,record['pin']);receipt=proof['manifest']
    expected={'schema':'g-full-backend-receipt-v1','stage':prior,'binding':binding,'engineering':False,'test_read':False,
              'scientific_verified':False,'training':prior=='train','status':'complete' if prior=='train' else 'verified'}
    if any(pb.encoded(receipt.get(k))!=pb.encoded(v) for k,v in expected.items()):raise ValueError('G predecessor semantics differ')
    if prior=='train' and pb.file_sha(Path(value['output_root'])/'manifest.json')!=receipt['native_manifest_sha256']:
        raise ValueError('native G trained manifest bytes differ')
    return receipt


def execute(stage,inputs,request,request_sha):
    import os
    from analysis import p_full_training_backend as pb
    value=read_transport(inputs,request,request_sha);before_identity=identity(value)
    records=stage_inputs(value,inputs,request,request_sha,stage)
    functions=transport_functions();jobsha=functions['check_job'](value,records,request,request_sha,stage)
    output=checked_path(value['output_root']);scratch=checked_path(value['scratch_root'])
    protected=[ROOT,Path(value['stage2']),Path(value['sid_producer']['root']),Path(value['sid_producer']['receipt']['path']).parent.parent]
    for target in (output,scratch):
        if any(target==p or p in target.parents or target in p.parents for p in protected):raise ValueError('exclusive output/scratch separate from inputs/source required')
    if output==scratch or output in scratch.parents or scratch in output.parents:raise ValueError('separate output and scratch required')
    native=native_functions();runtime=native['_runtime'](False)
    guard=native['_guard'](False,scratch,runtime['device']);guard()
    authenticated=full_context(value,guard,stage);binding=authenticated.proof
    if pb.encoded(value)!=pb.encoded(read_transport(inputs,request,request_sha)):raise ValueError('G request changed during loading')
    native=native_functions(context=authenticated,sid=authenticated.sid,binding=binding)
    prior=predecessor(stage,value,binding)
    sidpin=value['config']['sid_manifest_sha256'];root=Path(value['stage2'])
    # Original positional API carries byte identities only. _contexts and
    # _binding explicitly acquire/name UR001 data rather than old subset files.
    common=(ROOT,root/'UR001-manifest.json',value['producer_pin']['manifest_sha256'],root/'UR001-manifest.json',value['producer_pin']['manifest_sha256'],Path(binding['sid']['manifest']['path']),sidpin,Path(value['p3c']['path']),value['p3c']['sha256'])
    if stage=='preflight':
        details={'status':'verified','training':False,'scientific_verified':False,'data_opened':True,
                 'training_rows':len(authenticated.context.examples),'validation_queries':len(authenticated.context.queries),
                 'assembly_workers':0,'native_scientific_functions_unchanged':True}
    elif stage=='train':
        result=native['run'](*common,value['config'],output,temporary=scratch/'native-unused',eval_workers=1,progress=Path(os.environ['VKGR_PROGRESS_PATH']),decode=None)
        details={'status':result['status'],'training':True,'scientific_verified':False,'native_manifest_sha256':result['manifest_sha256']}
    else:
        result=native['verify'](*common,output/'manifest.json',prior['native_manifest_sha256'],output.parent/(output.name+'-verification-'+value['run_ids']['verify']),temporary=scratch/'native-unused',eval_workers=1,decode=None)
        details={'status':result['status'],'training':False,'scientific_verified':True,'native_verification':result,'native_manifest_sha256':prior['native_manifest_sha256']}
    # Hash-only producer consumption has no sid.fit call. Recheck all original
    # own-job, upstream stdout/terminal, source/runtime and data bytes at close.
    if identity(value)!=before_identity or consume_sid(value['sid_producer'])!=binding['sid']:raise ValueError('G source/runtime/SID closure changed')
    from analysis.ur001_derive_cli import verify_dag
    if verify_dag(root,require_profile='full')!=binding['ur001'] or verify_producer(root/'UR001-manifest.json',Path(value['stage2_run']),value['producer_pin'])!=binding['producer']:
        raise ValueError('G UR001 closure changed')
    predecessor(stage,value,binding)
    if functions['check_job'](value,records,request,request_sha,stage)!=jobsha:raise ValueError('G own job changed')
    guard()
    receipt={'schema':'g-full-backend-receipt-v1','stage':stage,'verified':True,'engineering':False,'test_read':False,
             'run_id':value['run_ids'][stage],'run_manifest_sha256':jobsha,'request_sha256':request_sha,
             'binding':binding,**before_identity,**details}
    pb.write_stage_receipt(output,stage,receipt)
    return receipt

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('dry','preflight','train','verify','plan'))
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--request')
    parser.add_argument('--request-sha256')
    parser.add_argument('--stage',choices=('preflight','train','verify'))
    parser.add_argument('--execute-approved-science',action='store_true')
    args=parser.parse_args(argv)
    if args.request is not None:
        try:
            if args.request_sha256 is None:raise ValueError('fixed request SHA required')
            if args.kind in ('train','verify') and not args.execute_approved_science:raise ValueError('explicit approved science required')
            if args.kind=='plan':
                if args.stage is None:raise ValueError('plan stage required')
                result=plan(args.stage,Path(args.inputs),Path(args.request),args.request_sha256)
            elif args.kind in ('preflight','train','verify'):
                result=execute(args.kind,Path(args.inputs),Path(args.request),args.request_sha256)
            else:raise ValueError('formal dry uses plan')
        except (InputRejected,ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
            import sys
            print(json.dumps({'stage':'entry_failed','type':type(exc).__name__,'reason':str(exc)}),file=sys.stderr,flush=True)
            print(json.dumps({'status':'HOLDOUT_INPUT_REJECTED' if isinstance(exc,InputRejected) and exc.holdout else 'REJECTED','verified':False}));return 2
        print(json.dumps(result,sort_keys=True));return 0
    result={'schema':'g-full-training-backend-preparation-v1','kind':args.kind,'engineering':False,
            'data_opened':False,'execution_allowed':False,'verified':False}
    try:
        result['inputs_sha256']=admit_inputs(args.inputs)
    except (InputRejected,OSError) as exc:
        result.update(status='HOLDOUT_INPUT_REJECTED' if isinstance(exc,InputRejected) and exc.holdout else 'INPUTS_REJECTED',reason=str(exc))
        print(json.dumps(result,sort_keys=True));return 2
    if args.kind=='dry':
        result['status']='INPUTS_ADMITTED';code=0
    else:
        try:require_final_sid(None)
        except ValueError as exc:result.update(status='FINAL_SID_CONTRACT_UNAVAILABLE',reason=str(exc))
        code=2
    print(json.dumps(result,sort_keys=True));return code


if __name__=='__main__':
    raise SystemExit(main())
