"""Fixed ad1c428 A1 acceptance adapter; native scientific functions remain original."""
import hashlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL="b5a4d1c99ba2d8361a2d90c23bac27af5b71769d"
NATIVE_FILES={'configs/sid_structure/g-structure-a1-s2024.json': '7121cb5203d16dd8b909bbf0662a559c0e34a006d5dae1dfcd08b654159bceb7', 'configs/sid_structure/g-structure-a1-s2025.json': '0d75b877119e90a0b7fd6ba05b4b8e1850af0a3306d52bc45867f9fafd22b7a4', 'configs/sid_structure/g-structure-a1-s42.json': '9438c9574e687496e42d3ac2453c649fd7d94adc9e51371e4a14aa96d92c69d5', 'src/vkgr/baseline_run.py': '26937152a2f996d9c5f58cde1aba76c70c36a05c3fe64bf66b1ea5ef02439c45', 'src/vkgr/baselines.py': 'ec92bb0b9858e9da7e192f8cc32da96d5a274ad195413220695b093361b96a0b', 'src/vkgr/coll_sid_a1_run.py': 'b1fce2524ea36731827c51a39f4494f549d652684b16035e21bafb9109582c78', 'src/vkgr/config.py': '540a878c5cfc8945bdfcc083609770e32cae92cebf566f980f9235fe0ba24546', 'src/vkgr/data_audit.py': 'c7f5f28cdb7e19181fc46f8d69140dba9711ff66f5b8537399692e37e4ded182', 'src/vkgr/generative_arch.py': 'cd661f909ed16b770dba2194f568e44e77d53c506a091b1bb7b3d3988b48ac01', 'src/vkgr/generative_b1_batched.py': 'd9f0ffba92fe57cc3dcd137b7083bdc92459561aa03d2bcacb4f5ebdb015bbb6', 'src/vkgr/generative_b1_run.py': 'd161aec55f74339129d556991532ec49ce9e520edc6425c7179398c4b5055af5', 'src/vkgr/generative_batched.py': '3918502e374bcedcf0873d2d9e419bc690f42d7ae991f81f9db793fb688b2564', 'src/vkgr/generative_coll_run.py': '025d6869915ecaa31c27e940c28e02033c095348b60cbecc7e6af938e50f0df2', 'src/vkgr/generative_data.py': 'ea4357989c3a520e7c7dbc1dfce2bbeff192f3d89e4e5f0703cfd27bf8e0c7c7', 'src/vkgr/generative_data_v2.py': 'd90f4b4338faefb6a9c72649e965a8e7480c7566e2b60ec683b9a4282b9b8157', 'src/vkgr/generative_diagnostics.py': 'b72cd0bd974ded67f6afcae88f4db5a14ad5a6fb46368fbc9a5745b864926db4', 'src/vkgr/generative_evaluation.py': 'a70adb8ef10b118c8310cb033d0a70abc57352557c78a0e1fe5986e1372ccd20', 'src/vkgr/generative_experiment.py': 'b0a94b26151c5fab13b0bd35bf0b3a0ff8ed8a3173f6b5aa08d410a83296b846', 'src/vkgr/generative_foundation.py': '3abfd2ea84ce713296d6745256afbf37bf804f0ce2cb2069433049ee92a844b9', 'src/vkgr/generative_g6_data.py': 'cf02dd617b8625318d31258015d29b1a5f960cc1c8c688b9f888278ca3620655', 'src/vkgr/generative_g6_model.py': 'ee0be45888a5d1c142e7dd4cd56b5913ed3add20c415a0760fedeeb8285b347d', 'src/vkgr/generative_g6_run.py': '9c1c4c2c1f995b3e7f79343138f1f726321d109d882aafa7946a17a0997bddbe', 'src/vkgr/generative_hp.py': 'cecf05758a3903d4be5f66f7f5ae627ca05b30869091432a9e62e57a73e86140', 'src/vkgr/generative_pilot.py': '95b6ad8a8115ada305734f59bbca73c9eca11c9dab75890d2dc6edd715144858', 'src/vkgr/generative_real.py': '79ba4fe5770438cf3a107c7064829f04b6d64de020b7b85e0a40a727c578fa9a', 'src/vkgr/generative_sharded.py': '993ac3879c9c51c8a090656e7eb25cad5185d49ac0726208bfae4d94b02efed1', 'src/vkgr/generative_tok_data.py': '48498849f9c96fe4283e5a2e48e8ce3a1a1663ed47db2c9e981d737ebc73ee4c', 'src/vkgr/generative_training.py': '3de7c934df486dbeaa29afb3d60cd0c0b6078a365fdb3338d29fc4f4b2382bfb', 'src/vkgr/history_v2.py': '1c8d8718ab7527524df9d47dfbb518ee7f5b419ca34dfbb8971aca7efc31f06e', 'src/vkgr/manifest.py': '3b06a12b3129aec87320eebbcc17a5859814e99fb479d57d10a6a8febe415f17', 'src/vkgr/retrieval_data.py': '1fa39c85b98ba8aebfabfd0c23659c4a178a1f3f842a5985469656c9deadb867', 'src/vkgr/runtime.py': 'ede19b05543327ec790f4c59301bd05006c750f89e4c149d75f0193cb4d8f4e8', 'src/vkgr/sid.py': '9602580f907cc2e8cfa3684aec7828df2bdfc87b59e71b6fd3865a9a19de91a3', 'src/vkgr/sid_coll_export.py': '0d7541e199dcb5397a37b6cc8eec24b9c94b3925b22100725b3fe2c935ab29fc', 'src/vkgr/sid_coll_run.py': 'ffd4ce26ba4c79babba3c542080c95d45fcf9eedb490217f70b5eac4841cfef9', 'src/vkgr/sid_coll_sgns.py': '2cc25d1997c12fe1e262e455e6644980186b5b9d0e00805014dee3148675876d', 'src/vkgr/sid_coll_v1.py': '48975e424fb21b00d2e0d1fffc4541c338cf9df5c3711fee9005a0266d33191c', 'src/vkgr/sid_depth.py': 'fde27366fc70bc30eec2dd419e97be382651d9efc2905ac6720f6ed9e3a8dbdd', 'src/vkgr/sid_depth_beam.py': 'b9c27adbb5c13e6c85381f0bd4a569a334175bda3b6393f0d5be571119218291', 'src/vkgr/sid_rqvae.py': '52173bea9b80ffeb40e25943e3d7069ec53f9bd34a4509c97278194714065067', 'src/vkgr/sid_rqvae_d.py': '627f904fe75ec85d7cb9752203f96d374eefc4df85db52e14b2025f26cf4eb47', 'src/vkgr/sid_run.py': 'bea7121443d7dd7afb19a96454bceb571a4a58b184416a0e7b7af908c771bfed', 'src/vkgr/sid_structure_a1_run.py': '94c1fcc3a7a63520180326b0bf489e954283fa7dfbb6739c925ea579535ab20a', 'src/vkgr/sid_structure_run.py': '60e62069704767aad56801c8c6ba39429c32516fb1edad09aad34918b5874e81', 'src/vkgr/sid_structure_v1.py': 'faeb469e9e353c6faef449730a882cf6a58f9e19c9cc30a70ae2bf507fb9b78b', 'src/vkgr/sid_tok_audit.py': 'd4f0afb554a6f48f787c91c81ffcb0baa7360a614b91152aa2d73cfd91b3f8ba', 'src/vkgr/sid_tok_audit_d.py': 'b96c140b02649884a36c0b2db13f7e9db02316a292d0e83cc6c1c2296346958a', 'src/vkgr/sid_v2.py': 'd094606ddab0460ef470e7b7cf86793af2c4352c8237842a803d6cd1c06470ad', 'src/vkgr/sid_v2d.py': 'a2e1e24bb4ca1e7ac5257823113748841ca5b1bbfcd26bc88b8b39c843457e88', 'src/vkgr/tiger.py': '09ab37d79da8e46dabf522d8d73d6dbd88b5b77985375e459e4977896a18c352', 'src/vkgr/tiger_arch.py': '696ef8e183f6ea63e35c8f45b19bd25f79483c0c8d85c3e99a9ba817e34af41b', }
NATIVE_LOCK_SHA='8b231cef0c5056b4a6a8365f5a02070ba4e1c043b55fc9faa020188c10bf2063'
SID_VERSION='GPU_SGNS_SUBSET_B48D3_S42'
A1_ADDITIONS=('analysis/sgns_subset_a1_inputs.py','src/vkgr/sgns_subset_a1_inputs.py','tests/test_sgns_subset_a1_inputs.py','analysis/sgns_subset_a1.py', 'configs/sid_structure/g-structure-a1-s2024.json', 'configs/sid_structure/g-structure-a1-s2025.json', 'configs/sid_structure/g-structure-a1-s42.json', 'src/vkgr/coll_sid_a1_run.py', 'src/vkgr/generative_b1_batched.py', 'src/vkgr/generative_b1_run.py', 'src/vkgr/generative_coll_run.py', 'src/vkgr/generative_g6_data.py', 'src/vkgr/generative_g6_model.py', 'src/vkgr/generative_g6_run.py', 'src/vkgr/generative_pilot.py', 'src/vkgr/generative_real.py', 'src/vkgr/generative_sharded.py', 'src/vkgr/generative_tok_data.py', 'src/vkgr/sgns_subset_a1.py', 'src/vkgr/sid_coll_export.py', 'src/vkgr/sid_coll_run.py', 'src/vkgr/sid_coll_v1.py', 'src/vkgr/sid_rqvae_d.py', 'src/vkgr/sid_structure_a1_run.py', 'src/vkgr/sid_structure_run.py', 'src/vkgr/sid_structure_v1.py', 'src/vkgr/sid_tok_audit_d.py', 'src/vkgr/sid_v2d.py', 'tests/fixtures/sgns_subset_a1_launcher_chain.py', 'tests/test_sgns_subset_a1.py')
SCHEMA='gpu-sgns-subset-a1-request-v1'
RUN_IDS={'preflight':'sgns-gpu-subset-a1-preflight-v1','train':'sgns-gpu-subset-a1-train-v1','verify':'sgns-gpu-subset-a1-verify-v1'}
INPUT_PINS={
    'prepared':'a55df0a3b285ce4cf623126fe53df53378bf5d7c0645c6484f0f405641bddd82',
    'history':'8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43',
    'content':'644b2d7a656999a8962355d7c32dae07134ad46e9557b4a8418f137d62f73581'}


def pin(value):
    if type(value) is not str or len(value)!=64 or set(value)-set('0123456789abcdef') or value=='0'*64:
        raise ValueError('explicit nonzero SHA256 required')
    return value


def absolute(path):
    from analysis.ur001_derive_cli import secure
    return secure(path)


def file_bytes(record):
    if type(record) is not dict or set(record)!={'path','sha256'}:
        raise ValueError('explicit path/SHA required')
    raw=absolute(record['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin(record['sha256']):
        raise ValueError('consumed artifact bytes differ')
    return raw


def registered_config(sid_sha):
    sid_sha=pin(sid_sha)
    path='configs/sid_structure/g-structure-a1-s42.json'
    config=json.loads(file_bytes({'path':str(ROOT/path),'sha256':NATIVE_FILES[path]}))
    config['sid_manifest_sha256']=sid_sha
    config['sid']=SID_VERSION
    return config


def check_native():
    for path,sha in NATIVE_FILES.items():
        file_bytes({'path':str(ROOT/path),'sha256':sha})
    from . import sid_structure_a1_run as original
    if Path(original.__file__).resolve()!=ROOT/'src/vkgr/sid_structure_a1_run.py':
        raise ValueError('native A1 import outside deployment')
    return original


def native_functions(pincheck,source,sidloader,*,base_loader=None):
    """Reuse exact native bytecode with three explicit input/source dependencies."""
    original=check_native()
    namespace=dict(original.__dict__)
    # Original _load_sid still verifies CONTENT_V1 and normalized features.
    namespace['coll']=SimpleNamespace(**{**original.coll.__dict__,'load_bundle':sidloader})
    if base_loader is not None:
        namespace['gr']=SimpleNamespace(**{**original.gr.__dict__,'load_context':base_loader})
    namespace['check_pins']=pincheck
    namespace['_source']=source
    namespace['SID_VERSION']=SID_VERSION
    for name in ('default_config','validate_config','_load_sid','_contexts','_binding','run','verify'):
        value=getattr(original,name)
        clone=FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
        clone.__kwdefaults__=value.__kwdefaults__
        namespace[name]=clone
    return namespace


def accepted_request(value,request_sha):
    required={'schema','source_commit','lock','data_root','prepared','history','content',
              'input_producer','cpu_sgns','gpu_sgns','cpu_sid','gpu_sid','output_root','scratch_root','ops_root','producer_delta'}
    if type(value) is not dict or set(value)!=required or value['schema']!=SCHEMA:
        raise ValueError('complete accepted A1 request required')
    pin(request_sha)
    if type(value['producer_delta']) is not dict or set(value['producer_delta'])!=set(A1_ADDITIONS):
        raise ValueError('exact fixed A1 source delta required')
    for name,sha in value['producer_delta'].items():
        pin(sha)
    if type(value['source_commit']) is not str or len(value['source_commit'])!=40 or set(value['source_commit'])-set('0123456789abcdef'):
        raise ValueError('fixed deployment commit required')
    for role in ('data_root','output_root','scratch_root','ops_root'):
        absolute(value[role])
    if value['content'].get('sha256')!=INPUT_PINS['content']:
        raise ValueError('native CONTENT_V1 pin differs; STOP')
    for role in ('prepared','history','content'):file_bytes(value[role])
    from .sgns_subset_a1_inputs import family_manifest
    family_manifest(json.loads(file_bytes(value['prepared'])),value['history'],value['content'],value['gpu_sid']['manifest']['sha256'])
    file_bytes(value['lock'])
    roots=[absolute(value[x]) for x in ('data_root','output_root','scratch_root')]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('new output and scratch separate from immutable data required')
    if any(p==ROOT or ROOT in p.parents or p in ROOT.parents for p in roots[1:]):
        raise ValueError('new output/scratch outside deployment required')
    return value


def producer(record,role,device):
    """Consume pinned original OPS terminal/stdout/receipt; no status-only shortcut."""
    if type(record) is not dict or set(record)!={'run','terminal','stdout','receipt'}:
        raise ValueError('complete producer closure required')
    raw={name:file_bytes(rec) for name,rec in record.items()}
    run=json.loads(raw['run']);terminal=json.loads(raw['terminal'])
    receipt=json.loads(raw['receipt'])
    stdout=json.loads(next((line for line in reversed(raw['stdout'].splitlines()) if line.strip()),b'{}'))
    expected_run='sgns-gpu-subset-'+device+('-fit-v1' if role=='sgns' else '-sid-verify-v1')
    source=receipt.get('source',{})
    if (run.get('run_id')!=expected_run or terminal.get('run_id')!=expected_run
            or absolute(record['run']['path']).parent.name!=expected_run
            or terminal.get('status')!='succeeded' or terminal.get('exit_code')!=0
            or terminal.get('signal') is not None or run.get('kind')!='ops'
            or run.get('seed')!=42 or run.get('git',{}).get('dirty') is not False
            or run.get('git',{}).get('commit')!=source.get('commit')
            or run.get('git',{}).get('dirty_files')!=[]
            or receipt.get('run_id')!=expected_run or stdout!=receipt
            or receipt.get('status')!='verified' or receipt.get('verified') is not True
            or receipt.get('authority')!='ad1c428' or receipt.get('engineering') is not False
            or receipt.get('device')!=device or receipt.get('subset_four_gate_acceptance') is not False):
        raise ValueError('accepted original producer binding differs')
    # Fixed module/stage is checked separately from producer-specific path arguments.
    expected_module='analysis.sgns_gpu_subset' if role=='sgns' else 'analysis.sgns_subset_sid'
    expected_stage='fit' if role=='sgns' else 'verify'
    argv=run.get('argv',[])
    if len(argv)<5 or argv[1:5]!=['-B','-m',expected_module,expected_stage]:
        raise ValueError('original formal producer argv differs')
    if receipt.get('stage')!=('sgns' if role=='sgns' else 'sid-verify'):
        raise ValueError('original producer stage differs')
    if role=='sgns':
        if (len(argv)!=15 or argv[5:15:2]!=['--bundle','--bundle-sha','--device','--lock','--output']
                or argv[10]!=device or absolute(argv[14])!=absolute(record['receipt']['path']).parent
                or argv[8]!=receipt.get('input_manifest_sha256')):
            raise ValueError('fixed SGNS device/input/output argv differs')
        file_bytes({'path':str(absolute(argv[6])/'manifest.json'),'sha256':argv[8]})
        actual_inputs={r['path']:r['sha256'] for r in run.get('inputs',[])}
        if actual_inputs.get(str(absolute(argv[6])/'manifest.json'))!=argv[8]:
            raise ValueError('SGNS launcher input manifest differs')
        if receipt.get('schema')!='gpu-sgns-subset-stage-v1' or receipt.get('test_read') is not False or receipt.get('validation_read') is not False:
            raise ValueError('fixed SGNS receipt required')
        proof=receipt.get('original_audit_verification',{})
        if proof.get('status')!='verified' or proof.get('schema')!='coll-input-verify-v1':
            raise ValueError('original SGNS audit verification required')
        root=absolute(record['receipt']['path']).parent
        for name in ('vectors.npy','audit.json'):
            file_bytes({'path':str(root/name),'sha256':receipt.get('artifacts',{}).get(name)})
    else:
        if (receipt.get('schema')!='gpu-sgns-subset-sid-stage-v1'
                or receipt.get('mapping_replay') is not True
                or receipt.get('run_manifest_sha256')!=record['run']['sha256']
                or receipt.get('gates',{}).get('passed') is not True):
            raise ValueError('original SID gates and independent replay required')
    return receipt


def closure(value):
    cpu=producer(value['cpu_sgns'],'sgns','cpu')
    gpu=producer(value['gpu_sgns'],'sgns','cuda')
    if any(cpu.get(k)!=gpu.get(k) for k in ('source','runtime','input_manifest_sha256','input_producer')):
        raise ValueError('CPU/GPU reference input/source/runtime differs')
    sids={}
    for key,device,sgns in (('cpu_sid','cpu',cpu),('gpu_sid','cuda',gpu)):
        record=value[key]
        if type(record) is not dict or set(record)!={'producer','manifest','input_manifest'}:
            raise ValueError('accepted SID producer/manifest/input closure required')
        receipt=producer(record['producer'],'sid',device)
        manifest=json.loads(file_bytes(record['manifest']));file_bytes(record['input_manifest'])
        if (receipt.get('gates')!=manifest.get('gates')
                or receipt.get('collision_gate')!=manifest.get('collision_gate')):
            raise ValueError('SID verification gates differ from consumed bundle')
        if (receipt.get('manifest_sha256')!=record['manifest']['sha256']
                or receipt.get('input_manifest_sha256')!=record['input_manifest']['sha256']
                or receipt.get('sgns_stage_receipt_sha256')!=value['cpu_sgns' if device=='cpu' else 'gpu_sgns']['receipt']['sha256']):
            raise ValueError('SID to SGNS producer binding differs')
        sids[key]=receipt
    # The new SID checker owns the exact original collision counts, not rounded floats.
    if sids['gpu_sid'].get('collision_gate',{}).get('passed') is not True:
        raise ValueError('GPU/CPU collision gate has not passed')
    return {'cpu_sgns':cpu,'gpu_sgns':gpu,**sids}


def engine_for(value,request_sha,source):
    if value['lock'].get('sha256')!=NATIVE_LOCK_SHA:raise ValueError('frozen external native lock differs')
    file_bytes(value['lock'])
    proof=closure(value)
    check_a1_delta(source,proof['gpu_sid']['source'],value['producer_delta'])
    from .sgns_subset_a1_inputs import input_producer
    prepared=json.loads(file_bytes(value['prepared']))
    proof['input_producer']=input_producer(value['input_producer'],value['prepared'],value['history'],source)
    if proof['input_producer'].get('runtime')!=prepared['runtime']:raise ValueError('input producer runtime differs')
    gpu=proof['gpu_sgns'];sid=value['gpu_sid'];vectors=absolute(value['gpu_sgns']['receipt']['path']).parent/'vectors.npy'
    def check_pins(sid_sha,input_sha,vectors_sha,engineering,*,config=None):
        if (engineering is not False or sid_sha!=sid['manifest']['sha256']
                or input_sha!=sid['input_manifest']['sha256']
                or vectors_sha!=gpu['artifacts']['vectors.npy']
                or config!=registered_config(sid_sha)):
            raise ValueError('new GPU producer pins/config differ; no engineering fallback')
    frozen_lock=dict(value['lock'])
    frozen_source=json.dumps(source,sort_keys=True,separators=(',',':'))
    def bound_source(root):
        if Path(root).resolve()!=ROOT:raise ValueError('fixed source root required')
        if value['lock']!=frozen_lock:raise ValueError('fixed external native lock mapping differs')
        file_bytes(frozen_lock)
        from analysis.ur001_derive_cli import source_identity
        if json.dumps(source_identity(),sort_keys=True,separators=(',',':'))!=frozen_source or json.dumps(source,sort_keys=True,separators=(',',':'))!=frozen_source:
            raise ValueError('fixed native source identity differs')
        original=check_native()
        def mapped_sha(path):
            if Path(path)==ROOT/'uv.lock':return hashlib.sha256(file_bytes(frozen_lock)).hexdigest()
            return original.file_sha256(path)
        namespace={**original._source.__globals__,'file_sha256':mapped_sha}
        mapped_source=FunctionType(original._source.__code__,namespace,original._source.__name__,original._source.__defaults__,original._source.__closure__)
        native=mapped_source(root)
        return {**native,'ad1c428_adapter':{'request_sha256':request_sha,'source':source,
            'native_lock_mapping':{'logical':'uv.lock','physical':value['lock']['path'],'sha256':NATIVE_LOCK_SHA},
            'producer_receipts':{key:value[key]['receipt']['sha256'] for key in ('cpu_sgns','gpu_sgns')},
            'sid_producer_receipts':{key:value[key]['producer']['receipt']['sha256'] for key in ('cpu_sid','gpu_sid')}}}
    from .sgns_subset_sid import load_bundle
    def base_loader(root,path,pin,temporary):
        from .sgns_subset_a1_inputs import load_context
        if Path(root).resolve()!=absolute(value['data_root']):raise ValueError('fixed new-family data root required')
        from analysis.new_instance_prepare import runtime
        prepared=json.loads(file_bytes(value['prepared']))
        if prepared['runtime']!=runtime(absolute(value['lock']['path'])):raise ValueError('new-family runtime differs')
        catalog=json.loads(file_bytes({'path':str(absolute(value['prepared']['path']).parent/'catalog.json'),'sha256':prepared['artifacts']['catalog.json']['sha256']}))
        loaded=engine['_load_sid'](Path(sid['manifest']['path']),sid['manifest']['sha256'],Path(value['content']['path']),value['content']['sha256'],
            catalog,False,
            Path(sid['input_manifest']['path']),sid['input_manifest']['sha256'],vectors,gpu['artifacts']['vectors.npy'])
        loaded=engine['_as_depth3'](loaded)
        return load_context(root,path,pin,temporary,codec=loaded.codec,sid_sha=sid['manifest']['sha256'],source=source)
    engine=native_functions(check_pins,bound_source,load_bundle,base_loader=base_loader)
    # New-family prepared/history identities are explicit; native content stays pinned.
    common=(ROOT,Path(value['prepared']['path']),value['prepared']['sha256'],Path(value['history']['path']),value['history']['sha256'],
            Path(sid['manifest']['path']),sid['manifest']['sha256'],Path(value['content']['path']),value['content']['sha256'])
    extra={'input_manifest':Path(sid['input_manifest']['path']),'input_sha':sid['input_manifest']['sha256'],
           'vectors_path':vectors,'vectors_sha':gpu['artifacts']['vectors.npy']}
    return engine,common,extra,proof


def validation_hits(rows,queries,aggregate):
    """Count the original verified item rows; never round recall into hits."""
    if len(rows)!=len(queries) or len(rows)!=aggregate['metrics']['all_eligible']['count']:
        raise ValueError('complete original validation rows required')
    hits=0
    for ordinal,(row,query) in enumerate(zip(rows,queries)):
        if (row.get('ordinal')!=ordinal or row.get('query_sha256')!=hashlib.sha256(__import__('vkgr.data_audit',fromlist=['canonical_bytes']).canonical_bytes(query)).hexdigest()
                or type(row.get('items')) is not list or any(type(item) is not int for item in row['items'])
                or len(set(row['items']))!=len(row['items'])):
            raise ValueError('original validation row identity differs')
        hits+=int(query['target'] in row['items'][:50])
    count=len(queries)
    if aggregate['metrics']['all_eligible']['recall@50']!=(hits/count if count else None):
        raise ValueError('integer hits and original recall disagree')
    return hits


def check_a1_delta(actual,producer,delta):
    if type(delta) is not dict or set(delta)!=set(A1_ADDITIONS):
        raise ValueError('exact fixed A1 source delta required')
    old=producer.get('files',{})
    if not old or set(old)&set(delta):raise ValueError('A1 delta must be additions only')
    if actual.get('files')!={**old,**delta}:raise ValueError('actual source must equal SID producer plus frozen A1 delta')
    for path,sha in delta.items():file_bytes({'path':str(ROOT/path),'sha256':sha})
    return delta
