"""Explicit new-family subset input identity; native scientific payloads are reused."""
from .data_audit import canonical_bytes


def native_payloads(partitions,validation,*,ur001_train,ur001_validation,ur001_slices):
    """Original like prefixes plus observed overlay, checked against UR001 bytes.

    Only the ad1c428 subset acceptance consumes this adapter. It does not issue an
    old G1/G3/P4E manifest identity or extend the original 1a input contract.
    """
    from .retrieval_data import prefix_examples,catalog_map
    from .data_audit import approved_policy,build_queries
    from .history_v2 import build_observed,overlay_examples
    parts=list(partitions)
    if [week for week,_ in parts]!=list(range(25)):
        raise ValueError('exactly 25 train weeks required')
    if type(validation) not in (tuple,list) or len(validation)!=2 or type(validation[0]) is not int or validation[0]!=25:
        raise ValueError('validation week25 required; test closed')
    catalog=sorted({row['item_id'] for _,rows in parts for row in rows})
    catalog_map(catalog)
    originals=list(prefix_examples(parts))
    queries=build_queries(parts,validation,approved_policy(),set(catalog))['queries']
    observed=build_observed(parts,originals,queries,catalog,max_len=50)
    overlaid=overlay_examples(originals,observed['train'],50,set(catalog))
    for role,actual,expected in (('train',overlaid,ur001_train),('validation',observed['validation'],ur001_validation),('slices',observed['slices'],ur001_slices)):
        if canonical_bytes(actual)!=canonical_bytes(expected):
            raise ValueError(role+' scientific payload differs from UR001 original oracle')
    return {'catalog':catalog,'like_examples':originals,'like_queries':queries,
            'observed_examples':overlaid,'observed_queries':observed['validation'],
            'slices':observed['slices'],'history':observed}

SCHEMA='gpu-sgns-subset-a1-inputs-v1'
SUBSET_MANIFEST_SHA='a24d37c4b3d12af2dd2c9d0d9877d5cc8d7cbd8c948e956cd4885ab73fb477cf'
POLICY='same_verified_gpu_sid_as_final_codec'


def family_manifest(value,history,content,sid_sha):
    keys={'schema','authority','subset_only','engineering','synthetic','test_read','base_codec_policy',
          'gpu_sid_manifest_sha256','content','history','ur001','raw_data','raw_records','source','runtime',
          'artifacts','semantic','statistics_binding'}
    if (type(value) is not dict or set(value)!=keys or value['schema']!=SCHEMA
            or value['authority']!='ad1c428' or value['subset_only'] is not True
            or value['engineering'] is not False or value['synthetic'] is not False or value['test_read'] is not False
            or value['base_codec_policy']!=POLICY or value['gpu_sid_manifest_sha256']!=sid_sha
            or value['history']!=history or value['content']!=content
            or value['ur001'].get('sha256')!=SUBSET_MANIFEST_SHA):
        raise ValueError('explicit new-family input identity required; no legacy or test claim')
    return value


def _json(record):
    import json
    from .sgns_subset_a1 import file_bytes
    return json.loads(file_bytes(record))


def _raw_payloads(data,ur001):
    """Consume fixed real subset DAG and independently rederive native payloads."""
    import json
    from analysis.ur001_derive_cli import secure,records_for,inspect_inputs,verify_dag,progress,guards,observed_rows
    from analysis.ur001_derivation import parquet_rows
    from .sgns_subset_a1 import file_bytes
    file_bytes(ur001)
    if ur001['sha256']!=SUBSET_MANIFEST_SHA:
        raise ValueError('fixed accepted subset DAG required')
    root=secure(ur001['path']).parent
    proof=verify_dag(root,require_profile='subset')
    manifest=json.loads((root/'UR001-manifest.json').read_bytes())
    recorded=json.loads((root/'UR001-input-manifest.json').read_bytes())
    records,_,_=records_for('subset')
    actual=inspect_inputs(secure(data),records,1)
    if actual['files']!=recorded['files'] or proof['manifest_sha256']!=ur001['sha256'] or proof['oracle'].get('bitwise_equal') is not True:
        raise ValueError('fixed raw/DAG original oracle differs')
    parts=[]
    for week in range(25):
        guards(root)
        parts.append((week,list(observed_rows(parquet_rows(secure(data)/f'train/week_{week:02d}.parquet',1),lambda:guards(root)))))
        progress('a1_inputs_train_read',week=week,rows=len(parts[-1][1]))
    val=(25,list(observed_rows(parquet_rows(secure(data)/'validation/week_25.parquet',1),lambda:guards(root))))
    guards(root);progress('a1_inputs_validation_read',rows=len(val[1]))
    def rows(name):return [json.loads(line) for line in (root/name).read_bytes().splitlines()]
    result=native_payloads(parts,val,ur001_train=rows('UR001-train.jsonl'),ur001_validation=rows('UR001-validation.jsonl'),
                           ur001_slices=json.loads((root/'UR001-S0.json').read_bytes()))
    if result['catalog']!=json.loads((root/'UR001-catalog.json').read_bytes()):
        raise ValueError('raw/catalog scientific payload differs')
    from .history_v2 import build_observed
    result['history100']=build_observed(parts,result['like_examples'],result['like_queries'],result['catalog'],max_len=100)
    # Post-read identity, including the original producer DAG artifacts.
    if inspect_inputs(secure(data),records,1)!=actual or verify_dag(root,require_profile='subset')!=proof:
        raise ValueError('raw/DAG changed during read')
    return result,parts,recorded,manifest


def semantic(payload):
    import hashlib
    return {key:hashlib.sha256(canonical_bytes(payload[key])).hexdigest()
            for key in ('catalog','like_examples','like_queries','observed_examples','observed_queries','slices')}


def write_family(output,payload,parts,*,raw_data,raw_records,ur001,content,sid_sha,source,runtime):
    """New names/lineage; P4E statistics values and HIST_V2 bytes stay native."""
    import hashlib
    from .sgns_subset_a1 import absolute,pin
    from . import history_v2 as hv,generative_evaluation as ev
    from .generative_experiment import _write_json
    from .data_audit import file_sha256
    output=absolute(output);output.mkdir(mode=0o700,exist_ok=False)
    history=hv.write_prepared(output/'history',parts,payload['like_examples'],payload['like_queries'],payload['catalog'],
                              {'family':SCHEMA,'raw_records':raw_records,'ur001':ur001,'source':source})
    artifacts={}
    examples=output/'like-examples.jsonl'
    with examples.open('xb') as stream:
        for row in payload['like_examples']:stream.write(canonical_bytes(row))
    for name,value in (('like-validation.json',payload['like_queries']),('catalog.json',payload['catalog']),('train-slices.json',payload['slices'])):
        _write_json(output/name,value)
    binding={'raw_records':[{'role':r['path'],'sha256':r['sha256'],'size_bytes':r['size']} for r in raw_records if r['path'].startswith('train/')],
             'raw_producer':{'family':SCHEMA,'ur001':ur001},'prepared_sha':semantic(payload)['like_examples'],
             'catalog_sha':semantic(payload)['catalog'],'consumer':source}
    stat={'schema_version':1,'task_id':'P4E','kind':'slice_statistics','binding':binding,'statistics_ref':'train-slices.json',
          'statistics_sha':file_sha256(output/'train-slices.json'),'threshold_definition':'nearest_rank_positive_item_event_counts_median_p90',
          'thresholds':payload['slices']['popularity_thresholds'],'item_groups':list(ev.ITEM_GROUPS),'user_groups':list(ev.USER_GROUPS),'holdout_used':False}
    _write_json(output/'slice-statistics-manifest.json',stat)
    for name in ('like-examples.jsonl','like-validation.json','catalog.json','train-slices.json','slice-statistics-manifest.json'):
        artifacts[name]={'sha256':file_sha256(output/name),'size':(output/name).stat().st_size}
    value={'schema':SCHEMA,'authority':'ad1c428','subset_only':True,'engineering':False,'synthetic':False,'test_read':False,
           'base_codec_policy':POLICY,'gpu_sid_manifest_sha256':pin(sid_sha),'raw_data':str(absolute(raw_data)),'raw_records':raw_records,
           'ur001':ur001,'content':content,'source':source,'runtime':runtime,'artifacts':artifacts,'semantic':semantic(payload),
           'history':{'path':str(output/'history/manifest.json'),'sha256':history['manifest_sha256']},'statistics_binding':binding}
    sha=_write_json(output/'manifest.json',value)
    return {'path':str(output/'manifest.json'),'sha256':sha,'history':value['history'],'subset_four_gate_acceptance':False}


def load_context(root,path,pin,temporary,*,codec,sid_sha,source):
    """Recheck original raw/payload semantics, then issue original RunContext."""
    import json
    import numpy as np
    from .sgns_subset_a1 import absolute,file_bytes
    from . import generative_data as gd,generative_experiment as ge,generative_evaluation as ev,history_v2
    from .generative_training import ExampleStore
    value=_json({'path':str(path),'sha256':pin})
    family_manifest(value,value.get('history'),value.get('content'),sid_sha)
    if value['source']!=source:raise ValueError('new-family source differs')
    payload,parts,raw,producer=_raw_payloads(value['raw_data'],value['ur001'])
    if raw['files']!=value['raw_records'] or semantic(payload)!=value['semantic']:
        raise ValueError('new-family raw/semantic identity differs')
    parent=absolute(path).parent
    expected={'like-examples.jsonl','like-validation.json','catalog.json','train-slices.json','slice-statistics-manifest.json'}
    if set(value['artifacts'])!=expected:raise ValueError('new-family exact artifacts required')
    for name,record in value['artifacts'].items():
        file_bytes({'path':str(parent/name),'sha256':record['sha256']})
        if (parent/name).stat().st_size!=record['size']:raise ValueError('new-family artifact size differs')
    data=b''.join(canonical_bytes(row) for row in payload['like_examples'])
    if (parent/'like-examples.jsonl').read_bytes()!=data:
        raise ValueError('new-family like examples differ')
    for name,key in (('like-validation.json','like_queries'),('catalog.json','catalog'),('train-slices.json','slices')):
        if (parent/name).read_bytes()!=canonical_bytes(payload[key]):raise ValueError('new-family scientific payload differs')
    actual=history_v2.load_prepared(value['history']['path'],value['history']['sha256'],50)
    overlaid=history_v2.overlay_examples(payload['like_examples'],actual['train'],50,set(payload['catalog']))
    queries=history_v2.overlay_queries(payload['like_queries'],actual['validation'],50,set(payload['catalog']))
    if canonical_bytes(overlaid)!=canonical_bytes(payload['observed_examples']) or canonical_bytes(queries)!=canonical_bytes(payload['observed_queries']) or canonical_bytes(actual['slices'])!=canonical_bytes(payload['slices']):
        raise ValueError('new-family observed overlay differs')
    from io import BytesIO
    array=np.load(BytesIO(file_bytes(value['content'])),allow_pickle=False)
    if array.dtype!=np.float32 or array.shape!=(len(payload['catalog'])+1,64) or not np.isfinite(array).all() or np.any(array[0]):raise ValueError('native finite CONTENT_V1 with PAD required')
    codec.check_identity()
    if list(map(int,codec.catalog.ids))!=payload['catalog']:raise ValueError('new-family base GPU SID catalog differs')
    stat=value['artifacts']['slice-statistics-manifest.json']['sha256']
    stats=ev.read_slice_statistics(parent/'slice-statistics-manifest.json',stat,value['statistics_binding'],payload['catalog'],payload['like_queries'])
    expected_items=ev._counts(payload['slices']['item_like_counts']);expected_users=ev._counts(payload['slices']['user_like_counts'],positive=True)
    if dict(stats.item_counts)!=expected_items or dict(stats.user_counts)!=expected_users:
        raise ValueError('native P4E scientific statistics differ')
    idmap={item:i+1 for i,item in enumerate(payload['catalog'])}
    for i,row in enumerate(payload['like_examples']):gd._strict_example(row,idmap,i)
    binding={'schema_version':1,'task_id':'GPU_SGNS_SUBSET_A1_INPUTS','family':SCHEMA,'synthetic':False,'engineering':False,
             'queries_sha256':ge._digest(payload['like_queries']),'catalog_sha256':ge._digest(payload['catalog']),
             'examples_sha256':value['artifacts']['like-examples.jsonl']['sha256'],'source':source,'input_manifest_sha256':pin,
             'semantic':value['semantic'],'ur001':value['ur001'],'base_codec_policy':POLICY,'base_gpu_sid_manifest_sha256':sid_sha,
             'statistics_manifest_sha256':stat,'test_gate':'closed'}
    context=ge.RunContext(ge._CONTEXT_KEY,examples=ExampleStore(parent/'like-examples.jsonl',binding['examples_sha256'],len(payload['like_examples'])),
                          codec=codec,queries=payload['like_queries'],binding=binding,stats=stats).check()
    if _json({'path':str(path),'sha256':pin})!=value:raise ValueError('new-family manifest changed during read')
    return context


def input_producer(record,prepared,history,source):
    """Original successful input job/stdout/file bytes, linked to consumed family."""
    import json
    from .sgns_subset_a1 import file_bytes,absolute
    if type(record) is not dict or set(record)!={'run','terminal','stdout','receipt'}:raise ValueError('original input producer closure required')
    raw={key:file_bytes(value) for key,value in record.items()}
    run=json.loads(raw['run']);terminal=json.loads(raw['terminal']);receipt=json.loads(raw['receipt'])
    stdout=json.loads(next((line for line in reversed(raw['stdout'].splitlines()) if line.strip()),b'{}'))
    run_id='sgns-gpu-subset-a1-inputs-v1';argv=run.get('argv',[])
    if (run.get('run_id')!=run_id or absolute(record['run']['path']).parent.name!=run_id
            or run.get('kind')!='ops' or run.get('seed')!=42
            or run.get('git')!={'commit':source['commit'],'dirty':False,'dirty_files':[]}
            or terminal.get('run_id')!=run_id or terminal.get('status')!='succeeded' or terminal.get('exit_code')!=0 or terminal.get('signal') is not None
            or receipt!=stdout or receipt.get('schema')!='gpu-sgns-subset-a1-input-stage-v1'
            or receipt.get('authority')!='ad1c428' or receipt.get('status')!='verified' or receipt.get('verified') is not True
            or receipt.get('run_id')!=run_id or receipt.get('run_manifest_sha256')!=record['run']['sha256']
            or receipt.get('source')!=source or receipt.get('science_training') is not False or receipt.get('test_read') is not False
            or receipt.get('subset_four_gate_acceptance') is not False
            or receipt.get('output')!={**prepared,'history':history,'subset_four_gate_acceptance':False}
            or len(argv)!=9 or argv[1:6]!=['-B','-m','analysis.sgns_subset_a1_inputs','derive','--request'] or argv[7]!='--request-sha256'
            or argv[8]!=receipt.get('request_sha256') or run.get('config',{}).get('path')!=argv[6] or run.get('config',{}).get('sha256')!=argv[8]
            or {'path':argv[6],'sha256':argv[8]} not in [{'path':r['path'],'sha256':r['sha256']} for r in run.get('inputs',[])]):
        raise ValueError('original input producer binding differs')
    file_bytes({'path':argv[6],'sha256':argv[8]});file_bytes(prepared)
    return receipt
