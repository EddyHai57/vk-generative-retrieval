"""Explicit native B A1 transport; original training and four-gate rules unchanged."""
import argparse
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from analysis import sgns_subset_a1 as legacy,sgns_subset_a1_inputs as derive_cli
from analysis import sgns_native_subset_sid as sid,sgns_subset_acceptance as acceptance
from analysis.sgns_subset_acceptance_a1 import transport_job
from analysis.ur001_derive_cli import secure,digest,encoded,source_identity
from vkgr import sgns_subset_a1 as core,sgns_subset_a1_inputs as inputs
ROOT=Path(__file__).resolve().parents[1]
MODULE='analysis.sgns_native_subset_a1'
NEW_FILES=('analysis/sgns_native_subset_a1.py','tests/test_sgns_native_subset_a1.py','docs/superpowers/reviews/2026-10-22-sgns-native-subset-a1-implementation.md')
DEPENDENCIES={'analysis/sgns_subset_a1.py': '04d1ef8ad82a5bb291b9ed4a3a81cbdde532d0a2b8dc0e63fb1fba941c814010', 'analysis/sgns_subset_a1_inputs.py': 'b2800990b7c21d8549f8baeb25d45f1a4cf479b6d0722f09b1598c57fb2fc9b0', 'analysis/sgns_subset_acceptance.py': '983e341009cdcc55f683b81d5c33a4ef590f5d2cf31a2eeff3534ff33acd2a50', 'analysis/sgns_subset_acceptance_a1.py': '750b00663b71cd209b69b8054b7e79337209b4e8304628bf5b8ee2dff0acdd26', 'configs/sid_structure/g-structure-a1-s2024.json': '7121cb5203d16dd8b909bbf0662a559c0e34a006d5dae1dfcd08b654159bceb7', 'configs/sid_structure/g-structure-a1-s2025.json': '0d75b877119e90a0b7fd6ba05b4b8e1850af0a3306d52bc45867f9fafd22b7a4', 'configs/sid_structure/g-structure-a1-s42.json': '9438c9574e687496e42d3ac2453c649fd7d94adc9e51371e4a14aa96d92c69d5', 'src/vkgr/coll_sid_a1_run.py': 'b1fce2524ea36731827c51a39f4494f549d652684b16035e21bafb9109582c78', 'src/vkgr/generative_b1_batched.py': 'd9f0ffba92fe57cc3dcd137b7083bdc92459561aa03d2bcacb4f5ebdb015bbb6', 'src/vkgr/generative_b1_run.py': 'd161aec55f74339129d556991532ec49ce9e520edc6425c7179398c4b5055af5', 'src/vkgr/generative_coll_run.py': '025d6869915ecaa31c27e940c28e02033c095348b60cbecc7e6af938e50f0df2', 'src/vkgr/generative_g6_data.py': 'cf02dd617b8625318d31258015d29b1a5f960cc1c8c688b9f888278ca3620655', 'src/vkgr/generative_g6_model.py': 'ee0be45888a5d1c142e7dd4cd56b5913ed3add20c415a0760fedeeb8285b347d', 'src/vkgr/generative_g6_run.py': '9c1c4c2c1f995b3e7f79343138f1f726321d109d882aafa7946a17a0997bddbe', 'src/vkgr/generative_pilot.py': '95b6ad8a8115ada305734f59bbca73c9eca11c9dab75890d2dc6edd715144858', 'src/vkgr/generative_real.py': '79ba4fe5770438cf3a107c7064829f04b6d64de020b7b85e0a40a727c578fa9a', 'src/vkgr/generative_sharded.py': '993ac3879c9c51c8a090656e7eb25cad5185d49ac0726208bfae4d94b02efed1', 'src/vkgr/generative_tok_data.py': '48498849f9c96fe4283e5a2e48e8ce3a1a1663ed47db2c9e981d737ebc73ee4c', 'src/vkgr/sgns_subset_a1.py': 'ec4875a625145a8465fb3e0b9ba52505583949a7d2e7975cda5284f6375ccc96', 'src/vkgr/sgns_subset_a1_inputs.py': 'ffe1343742a86872877a495d4287c6d68a3edbe896910c89645feee577598e8f', 'src/vkgr/sid_coll_export.py': '0d7541e199dcb5397a37b6cc8eec24b9c94b3925b22100725b3fe2c935ab29fc', 'src/vkgr/sid_coll_run.py': 'ffd4ce26ba4c79babba3c542080c95d45fcf9eedb490217f70b5eac4841cfef9', 'src/vkgr/sid_coll_v1.py': '48975e424fb21b00d2e0d1fffc4541c338cf9df5c3711fee9005a0266d33191c', 'src/vkgr/sid_rqvae_d.py': '627f904fe75ec85d7cb9752203f96d374eefc4df85db52e14b2025f26cf4eb47', 'src/vkgr/sid_structure_a1_run.py': '94c1fcc3a7a63520180326b0bf489e954283fa7dfbb6739c925ea579535ab20a', 'src/vkgr/sid_structure_run.py': '60e62069704767aad56801c8c6ba39429c32516fb1edad09aad34918b5874e81', 'src/vkgr/sid_structure_v1.py': 'faeb469e9e353c6faef449730a882cf6a58f9e19c9cc30a70ae2bf507fb9b78b', 'src/vkgr/sid_tok_audit_d.py': 'b96c140b02649884a36c0b2db13f7e9db02316a292d0e83cc6c1c2296346958a', 'src/vkgr/sid_v2d.py': 'a2e1e24bb4ca1e7ac5257823113748841ca5b1bbfcd26bc88b8b39c843457e88'}
RUN_IDS={name:'sgns-native-subset-a1-'+name+'-v1' for name in ('derive','preflight','train','verify','accept')}
EXEC_CALLBACKS={'read_request','identity','own_job','predecessor'}
SOURCE_SCHEMAS={'derive':'sgns-native-subset-a1-input-request-v1','a1':'sgns-native-subset-a1-request-v1'}
SCHEMA='sgns-native-subset-a1-stage-v1'
TOTAL='sgns-native-subset-four-gate-receipt-v1'
exact=sid.exact
matches=sid.matches
clone=sid.clone


def check_delta(current,old,delta):
    if type(delta)is not dict or set(delta)!=set(DEPENDENCIES)|set(NEW_FILES) or set(old.get('files',{}))&set(delta):raise ValueError('closed exact A1 source increment required')
    if current.get('files')!={**old.get('files',{}),**delta}:raise ValueError('pre-existing source bytes or increment differ')
    for path,pin in delta.items():
        core.pin(pin)
        if digest(ROOT/path)!=pin or path in DEPENDENCIES and DEPENDENCIES[path]!=pin:raise ValueError('pinned original dependency bytes differ')
    return delta


def imported(namespace,overrides):
    """Only exact relative/absolute input module names can receive a proxy."""
    allowed={'vkgr.sgns_subset_a1_inputs','vkgr.sgns_subset_sid','sgns_subset_a1_inputs','sgns_subset_sid'}
    if set(overrides)-allowed:raise ValueError('closed original import dependency set required')
    original=builtins.__import__
    def load(name,globals=None,locals=None,fromlist=(),level=0):
        if name in overrides and fromlist and (level==1 or name.startswith('vkgr.')):return overrides[name]
        return original(name,globals,locals,fromlist,level)
    return {**namespace,'__builtins__':{**builtins.__dict__,'__import__':load}}


def execution_facade(callbacks,core_proxy,input_check):
    if type(callbacks)is not dict or set(callbacks)!=EXEC_CALLBACKS:raise ValueError('closed A1 transport callback set required')
    proxy=SimpleNamespace(**{**inputs.__dict__,'input_producer':input_check})
    return clone(legacy._execute,imported({**legacy.__dict__,**callbacks,'core':core_proxy},{'vkgr.sgns_subset_a1_inputs':proxy}))


def load(path,pin,*,frozen=True):
    v=json.loads(core.file_bytes({'path':str(path),'sha256':pin}))
    common={'schema','source','lock','sid_root','sid_request','producer_delta','ops_root','ops_code'}
    derive={'raw_data','ur001','content','output_root'}
    a1={'data_root','prepared','history','content','input_request','output_root','scratch_root'}
    kind='derive' if v.get('schema')==SOURCE_SCHEMAS['derive'] else 'a1'
    if type(v)is not dict or set(v)!=common|(derive if kind=='derive' else a1) or v['schema']!=SOURCE_SCHEMAS[kind]:raise ValueError('complete native B A1 request required')
    if type(v['source'])is not dict or set(v['source'])!={'commit','files'}:raise ValueError('complete fixed source required')
    sid.producer.hexsha(v['source']['commit'],40)
    if type(v['producer_delta'])is not dict or set(v['producer_delta'])!=set(DEPENDENCIES)|set(NEW_FILES):raise ValueError('exact pinned A1 increment required')
    for name,pin in v['producer_delta'].items():
        core.pin(pin)
        if name in DEPENDENCIES and pin!=DEPENDENCIES[name]:raise ValueError('original dependency pin differs')
    for role in ('lock','sid_request','content',*(['ur001'] if kind=='derive' else ['prepared','history','input_request'])):
        record=v[role]
        if type(record)is not dict or set(record)!={'path','sha256'}:raise ValueError('exact path/pin record required')
        secure(record['path']);core.pin(record['sha256'])
    if frozen and (v['lock']['sha256']!=core.NATIVE_LOCK_SHA or v['content']['sha256']!=core.INPUT_PINS['content']):raise ValueError('original lock/content pin differs')
    roots=[secure(v['output_root']),*([secure(v['scratch_root'])] if kind=='a1' else [])]
    immutable=[ROOT,secure(v['sid_root']),secure(v['ops_root']),secure(v['ops_code']),secure(v['raw_data' if kind=='derive' else 'data_root'])]
    if any(a==b or a in b.parents or b in a.parents for i,a in enumerate(roots) for b in roots[i+1:]+immutable):raise ValueError('exclusive new output/scratch required')
    return v


def protect_outputs(v,evidence):
    outputs=[secure(v['output_root']),*([secure(v['scratch_root'])] if 'scratch_root' in v else [])]
    for out in outputs:
        for item in evidence:
            root=secure(item)
            if out==root or out in root.parents or root in out.parents:raise ValueError('A1 output/scratch overlaps immutable producer evidence')


def context(v,*,replay=None):
    root=secure(v['sid_root']);entry=root/'analysis/sgns_native_subset_sid.py'
    if digest(entry)!=digest(ROOT/'analysis/sgns_native_subset_sid.py'):raise ValueError('SID consumer differs from original producer context')
    module=sid.module_at(entry,'analysis._native_a1_sid_context');request=module.load(Path(v['sid_request']['path']),v['sid_request']['sha256'])
    acceptance.source_tree(root,request['source'])
    check_delta(v['source'],request['source'],v['producer_delta'])
    if v['lock']['path']!=request['lock']:raise ValueError('SID/A1 lock path differs')
    native=module.native_context(request)[1];cpu=module.legacy.load_request(Path(request['cpu_request']),request['cpu_request_sha256'])
    evidence=[*request['outputs'].values(),*(request[key] for key in ('ops_root','native_root','cpu_root','native_request','cpu_request')),*(native[key] for key in ('bundle','output','ops_root')),*(cpu[key] for key in ('bundle','sgns_output','sgns_run','fit_output','verify_output','ops_root')),v['sid_request']['path'],v['lock']['path'],v['content']['path']]
    if 'prepared' in v:evidence.extend((str(secure(v['prepared']['path']).parent),v['input_request']['path']))
    protect_outputs(v,evidence)
    if replay is not None:
        path,scratch=replay;scratch=secure(scratch)
        immutable=[ROOT,secure(path),*(secure(v[k]) for k in ('sid_root','ops_root','ops_code','data_root','output_root','scratch_root')),
            *(secure(record['path']) for record in v.values() if type(record)is dict and 'path' in record),*(secure(item) for item in evidence)]
        if scratch.exists() or any(scratch==root or scratch in root.parents or root in scratch.parents for root in immutable):raise ValueError('exclusive new acceptance replay scratch required')
    return module,request


def producer_closure(v):
    module,request=context(v)
    origin=clone(module.source_identity,{**module.source_identity.__globals__,'ROOT':secure(v['sid_root'])})
    consumer=clone(module.consume,{**module.__dict__,'source_identity':origin})
    got=consumer(Path(v['sid_request']['path']),v['sid_request']['sha256'])
    if got['verification'].get('backend')!='B' or got['verification'].get('schema')!='sgns-native-subset-sid-stage-v1':raise ValueError('explicit verified B SID required')
    return module,request,got


def identity(v):
    from analysis.new_instance_prepare import runtime
    if source_identity()!=v['source']:raise ValueError('clean fixed A1 source differs')
    core.check_native();core.file_bytes(v['lock']);env=json.loads(encoded(runtime(secure(v['lock']['path']))))
    context(v)
    return v['source'],env


def invocation(stage,path,pin):return [sys.executable,'-B','-m',MODULE,stage,'--request',str(path),'--request-sha256',pin]
def fixed_env(v,stage):
    value={'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    if stage!='derive':value['VKGR_DATA_ROOT']=v['data_root']
    return value


def stage_inputs(v,path,pin,stage):
    records=[{'path':str(path),'sha256':pin},v['lock'],v['sid_request'],v['content']]
    records.extend([v['ur001']] if stage=='derive' else [v['prepared'],v['history'],v['input_request']])
    module,request=context(v);records.extend(module.pins(request,Path(v['sid_request']['path']),'verify'))
    for name in ('fit','verify'):
        root=secure(request['outputs'][name]);run=secure(request['ops_root'])/'runs'/request['run_ids'][name]
        records.extend({'path':str(p),'sha256':digest(p)} for p in [root/'native-receipt.json',root/'stage-receipt.json',*(run/n for n in ('run.json','terminal.json','logs/stdout.log'))])
    if stage!='derive':
        d=json.loads(core.file_bytes(v['input_request']));root=secure(d['output_root']);run=secure(d['ops_root'])/'runs'/RUN_IDS['derive']
        records.extend({'path':str(p),'sha256':digest(p)} for p in [root/'native-receipt.json',root/'stage-receipt.json',*(run/n for n in ('run.json','terminal.json','logs/stdout.log'))])
    if stage in ('train','verify','accept'):
        prior={'train':'preflight','verify':'train','accept':'verify'}[stage];root=secure(v['output_root'])/prior;run=secure(v['ops_root'])/'runs'/RUN_IDS[prior]
        records.extend({'path':str(p),'sha256':digest(p)} for p in [root/'native-receipt.json',root/'stage-receipt.json',*(run/n for n in ('run.json','terminal.json','logs/stdout.log'))])
    seen={};result=[]
    for rec in records:
        name=str(secure(rec['path']));pin=core.pin(rec['sha256'])
        if name in seen and seen[name]!=pin:raise ValueError('conflicting input role pins')
        if name not in seen:result.append({'path':name,'sha256':pin});seen[name]=pin
    return result


def resources(v,stage):
    ram,vram=98304,0 if stage=='derive' else 32768
    if stage in ('verify','accept'):
        measured=legacy.measured_resources(json.loads((secure(v['output_root'])/'train/stage-receipt.json').read_bytes()));ram,vram=measured['ram_mib'],measured['vram_mib']
    return {'cpu_cores':1,'ram_mib':ram,'vram_mib':vram,'disk_gib':4,'wall_seconds':7200. if stage in ('derive','preflight') else 172800.}


def contract(v,path,pin,stage):
    env=fixed_env(v,stage);rid=RUN_IDS[stage];run=secure(v['ops_root'])/'runs'/rid
    return {'schema':'vkgr-run-v1','run_id':rid,'kind':'ops','cwd':str(ROOT),'argv':invocation(stage,path,pin),'seed':42,
        'git':{'commit':v['source']['commit'],'dirty':False,'dirty_files':[]},'config':{'path':str(path),'sha256':pin},
        'inputs':stage_inputs(v,path,pin,stage),'resources':resources(v,stage),'env':{**env,'VKGR_RUN_ID':rid,'VKGR_PROGRESS_PATH':str(run/'progress.json')}}


def job_binding(v,path,pin,stage):
    run=secure(v['ops_root'])/'runs'/RUN_IDS[stage];raw=(run/'run.json').read_bytes();record=json.loads(raw);expected=contract(v,path,pin,stage)
    if not matches(record,{k:val for k,val in expected.items() if k!='inputs'}) or not exact([{'path':r['path'],'sha256':r['sha256']} for r in record.get('inputs',[])],expected['inputs']):raise ValueError('native A1 job/source/argv/input/resource/env binding differs')
    transport_job(record,runroot=run,code=ROOT,source=v['source'],argv=expected['argv'],config=expected['config'],inputs=expected['inputs'],resources=expected['resources'],cublas=True,data_root=None if stage=='derive' else v['data_root'])
    return hashlib.sha256(raw).hexdigest()


def own_job(v,path,pin,stage):
    expected=contract(v,path,pin,stage)['env']
    if not exact({key:os.environ.get(key) for key in expected},expected):raise ValueError('own native A1 environment differs')
    return job_binding(v,path,pin,stage)


def output(v,stage):return secure(v['output_root']) if stage=='derive' else secure(v['output_root'])/stage

def successful(v,path,pin,stage):
    own=job_binding(v,path,pin,stage);out=output(v,stage);run=secure(v['ops_root'])/'runs'/RUN_IDS[stage]
    raw={name:(run/name).read_bytes() for name in ('run.json','terminal.json','logs/stdout.log')};inner_raw=(out/'stage-receipt.json').read_bytes();inner=json.loads(inner_raw)
    envelope=json.loads((out/'native-receipt.json').read_bytes());terminal=json.loads(raw['terminal.json'])
    expected={'schema':SCHEMA,'backend':'B','authority':'4b0a8a1','status':'verified','verified':True,'engineering':False,'stage':stage,'run_id':RUN_IDS[stage],
        'run_manifest_sha256':own,'request_sha256':pin,'source':v['source'],'science_receipt':inner,'science_receipt_sha256':hashlib.sha256(inner_raw).hexdigest(),
        'test_read':False,'subset_four_gate_acceptance':False,'sid_request':v['sid_request']}
    if not matches(envelope,expected) or not exact(acceptance.last_receipt(raw['logs/stdout.log']),envelope) or not matches(terminal,{'schema':'vkgr-terminal-v1','run_id':RUN_IDS[stage],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}):raise ValueError('native A1 terminal/stdout/science envelope differs')
    return inner,envelope,{name:hashlib.sha256(data).hexdigest() for name,data in raw.items()}


def input_check(v):
    request=v['input_request'];d=load(Path(request['path']),request['sha256'])
    receipt,envelope,proof=successful(d,Path(request['path']),request['sha256'],'derive')
    if d['source']!=v['source'] or d['lock']!=v['lock'] or d['content']!=v['content'] or d['sid_request']!=v['sid_request'] or d['sid_root']!=v['sid_root'] or secure(d['output_root'])!=secure(v['prepared']['path']).parent:raise ValueError('new B family request lineage differs')
    expected={'schema':'gpu-sgns-subset-a1-input-stage-v1','authority':'ad1c428','status':'verified','verified':True,'run_id':RUN_IDS['derive'],
        'run_manifest_sha256':proof['run.json'],'request_sha256':request['sha256'],'source':v['source'],'runtime':identity(v)[1],
        'output':{**v['prepared'],'history':v['history'],'subset_four_gate_acceptance':False},'science_training':False,'test_read':False,'subset_four_gate_acceptance':False}
    if not exact(receipt,expected):raise ValueError('new B family producer semantic binding differs')
    family=json.loads(core.file_bytes(v['prepared']));inputs.family_manifest(family,v['history'],v['content'],sid_manifest(v)['sha256'])
    if family['source']!=v['source'] or family['runtime']!=receipt['runtime'] or family['raw_data']!=d['raw_data'] or family['ur001']!=d['ur001']:raise ValueError('new B family source/runtime/data differs')
    return receipt


def sid_manifest(v):
    module,request=context(v);path=secure(request['outputs']['fit'])/'run_manifest.json'
    return {'path':str(path),'sha256':digest(path)}


def view(v):
    module,request,got=producer_closure(v);native=module.native_context(request)[1];cpu=module.legacy.load_request(Path(request['cpu_request']),request['cpu_request_sha256'])
    def rec(run,receipt):return {key:{'path':str(path),'sha256':digest(path)} for key,path in {'run':run/'run.json','terminal':run/'terminal.json','stdout':run/'logs/stdout.log','receipt':receipt}.items()}
    gpu=rec(secure(native['ops_root'])/'runs'/native['fit_run_id'],secure(native['output'])/'fit/result.json')
    gpu_sid=rec(secure(request['ops_root'])/'runs'/request['run_ids']['verify'],secure(request['outputs']['verify'])/'native-receipt.json')
    cpu_sgns=rec(secure(cpu['sgns_run']),secure(cpu['sgns_output'])/'stage-receipt.json');cpu_sid=rec(secure(cpu['ops_root'])/'runs'/module.legacy.run_id(cpu,'verify'),secure(cpu['verify_output'])/'stage-receipt.json')
    derived=json.loads(core.file_bytes(v['input_request']));irun=secure(derived['ops_root'])/'runs'/RUN_IDS['derive'];input_record=rec(irun,secure(derived['output_root'])/'native-receipt.json')
    manifest=sid_manifest(v);im=secure(request['outputs']['fit'])/'input-manifest.json'
    value={**v,'source_commit':v['source']['commit'],'gpu_sgns':gpu,'cpu_sgns':cpu_sgns,'gpu_sid':{'producer':gpu_sid,'manifest':manifest,'input_manifest':{'path':str(im),'sha256':digest(im)}},
        'cpu_sid':{'producer':cpu_sid,'manifest':{'path':str(secure(cpu['fit_output'])/'run_manifest.json'),'sha256':digest(secure(cpu['fit_output'])/'run_manifest.json')},'input_manifest':{'path':str(secure(cpu['fit_output'])/'input-manifest.json'),'sha256':digest(secure(cpu['fit_output'])/'input-manifest.json')}},'input_producer':input_record}
    proof={'cpu_sgns':json.loads(core.file_bytes(cpu_sgns['receipt'])),'gpu_sgns':got['native_input']['sgns_receipt'],'cpu_sid':json.loads(core.file_bytes(cpu_sid['receipt'])),'gpu_sid':got['verification']['science_receipt']}
    return value,proof,got


def engine_for(v,pin,source):
    value,proof,got=view(v)
    if source!=v['source']:raise ValueError('frozen source differs')
    def checked_input(*args):return input_check(v)
    def checked_delta(actual,old,delta):return check_delta(actual,old,delta)
    proxy_inputs=SimpleNamespace(**{**inputs.__dict__,'input_producer':checked_input})
    def bundle(path,sha,**kwargs):
        if secure(path)!=secure(value['gpu_sid']['manifest']['path']) or sha!=value['gpu_sid']['manifest']['sha256']:raise ValueError('native B exact SID manifest differs')
        checker=sid.facade({name:(lambda *a:None) for name in sid.CALLBACKS},lambda current,old:sid.check_delta(current,old,context(v)[1]['producer_delta'])).__globals__['science']
        return checker.load_bundle(path,sha,**kwargs)
    proxy_sid=SimpleNamespace(**{**sid.science.__dict__,'load_bundle':bundle})
    namespace=imported({**core.__dict__,'closure':lambda _:proof,'check_a1_delta':checked_delta},{'sgns_subset_a1_inputs':proxy_inputs,'sgns_subset_sid':proxy_sid})
    # The source delta belongs to the explicit SID deployment, while its inner
    # science receipt also records that identity without claiming numpy SGNS.
    return clone(core.engine_for,namespace)(value,pin,source)


def core_proxy(v,pin,source):
    return SimpleNamespace(**{**core.__dict__,'RUN_IDS':{k:RUN_IDS[k] for k in ('preflight','train','verify')},'engine_for':lambda *_:engine_for(v,pin,source),'closure':lambda _:producer_closure(v)})


def predecessor(v,stage,pin,source,*,request=None):
    if request is None:raise ValueError('explicit predecessor request required')
    prior='preflight' if stage=='train' else 'train';receipt,_,_=successful(v,request,pin,prior)
    engine,_,_,proof=engine_for(v,pin,source);config=core.registered_config(sid_manifest(v)['sha256']);env=identity(v)[1]
    bound=clone(legacy.receipt_binding,{**legacy.__dict__,'core':core_proxy(v,pin,source)})
    bound(receipt,prior=prior,source=source,env=env,config=config,proof=proof,request_sha=pin,run_sha=job_binding(v,request,pin,prior))
    value,_,_=view(v)
    if prior=='preflight':
        family=json.loads(core.file_bytes(v['prepared']));parent=secure(v['prepared']['path']).parent
        count=lambda name:len(json.loads(core.file_bytes({'path':str(parent/name),'sha256':family['artifacts'][name]['sha256']})))
        if type(receipt.get('query_count'))is not int or receipt['query_count']!=count('like-validation.json') or type(receipt.get('catalog_size'))is not int or receipt['catalog_size']!=count('catalog.json') or not exact(receipt.get('native_runtime'),json.loads(encoded(engine['_runtime'](False)))) or receipt.get('payload_written')is not False or receipt.get('scratch_written')is not True or not (secure(v['scratch_root'])/'preflight').is_dir():raise ValueError('original A1 preflight output semantics differ')
    else:
        if receipt.get('training')is not True or receipt.get('scientific_verified')is not False:raise ValueError('original training semantics differ')
        legacy.train_output_binding(value,receipt,engine,config,engine['_runtime'](False))
    return receipt


def envelope(v,path,pin,stage,receipt):
    raw=(output(v,stage)/'stage-receipt.json').read_bytes()
    if raw!=encoded(receipt):raise ValueError('original science stage bytes differ')
    value={'schema':SCHEMA,'backend':'B','authority':'4b0a8a1','status':'verified','verified':True,'engineering':False,'stage':stage,'run_id':RUN_IDS[stage],
        'run_manifest_sha256':job_binding(v,path,pin,stage),'request_sha256':pin,'source':v['source'],'science_receipt':receipt,'science_receipt_sha256':hashlib.sha256(raw).hexdigest(),
        'test_read':False,'subset_four_gate_acceptance':False,'sid_request':v['sid_request']}
    sid.science.write_new(output(v,stage)/'native-receipt.json',encoded(value));return value


def derive(path,pin):
    v=load(path,pin);source,env=identity(v);module,request,got=producer_closure(v);own_job(v,path,pin,'derive');before=stage_inputs(v,path,pin,'derive')
    from analysis.ur001_derive_cli import guards
    guards(secure(v['output_root']))
    payload,parts,raw_inputs,original=inputs._raw_payloads(v['raw_data'],v['ur001'])
    guards(secure(v['output_root']))
    result=inputs.write_family(secure(v['output_root']),payload,parts,raw_data=v['raw_data'],raw_records=raw_inputs['files'],ur001=v['ur001'],content=v['content'],sid_sha=sid_manifest(v)['sha256'],source=source,runtime=env)
    receipt={'schema':'gpu-sgns-subset-a1-input-stage-v1','authority':'ad1c428','status':'verified','verified':True,'run_id':RUN_IDS['derive'],
        'run_manifest_sha256':job_binding(v,path,pin,'derive'),'request_sha256':pin,'source':source,'runtime':env,'output':result,
        'science_training':False,'test_read':False,'subset_four_gate_acceptance':False}
    sid.science.write_new(output(v,'derive')/'stage-receipt.json',encoded(receipt))
    if identity(v)!=(source,env) or not exact(load(path,pin),v) or not exact(stage_inputs(v,path,pin,'derive'),before):raise ValueError('derive source/input/runtime changed')
    return envelope(v,path,pin,'derive',receipt)


def execute(stage,path,pin):
    from analysis.g_decode_timing import ContextSampler
    v=load(path,pin);source,env=identity(v);before=stage_inputs(v,path,pin,stage);value,_,_=view(v);input_check(v)
    callbacks={'read_request':lambda *_:value,'identity':lambda _:identity(v),'own_job':lambda _,p,h,s:own_job(v,p,h,s),'predecessor':lambda _,s,h,src,**kw:predecessor(v,s,h,src,**kw)}
    # Original _execute imports input_producer at its final closure. Both that
    # dependency and the core engine are scoped to this invocation's namespace.
    sampler=ContextSampler()
    try:receipt=execution_facade(callbacks,core_proxy(v,pin,source),lambda *a:input_check(v))(stage,path,pin,sampler)
    finally:sampler.close()
    if identity(v)!=(source,env) or not exact(load(path,pin),v) or not exact(stage_inputs(v,path,pin,stage),before):raise ValueError('A1 fixed snapshot changed')
    return envelope(v,path,pin,stage,receipt)


def replay(v,path,pin,scratch):
    """Original read-only epoch/selection/row replay; no fit/run/verify call."""
    scratch=secure(scratch)
    context(v,replay=(path,scratch))
    source,env=identity(v);module,request,got=producer_closure(v);input_check(v)
    value,_,_=view(v);engine,common,extra,proof=engine_for(v,pin,source);config=core.registered_config(sid_manifest(v)['sha256'])
    verified,_,_=successful(v,path,pin,'verify');trained=predecessor(v,'verify',pin,source,request=path);predecessor(v,'train',pin,source,request=path)
    expected={'schema':'gpu-sgns-subset-a1-stage-v1','authority':'ad1c428','engineering':False,'stage':'verify','status':'verified','verified':True,
        'run_id':RUN_IDS['verify'],'run_manifest_sha256':job_binding(v,path,pin,'verify'),'request_sha256':pin,'source':source,'runtime':env,
        'config':config,'input_producers':proof,'test_read':False,'subset_four_gate_acceptance':False,'hits_gate':378,'training':False,'resource_scope':legacy.RESOURCE_SCOPE}
    details={'native_verification','validation_hits','context_inclusive_peak_mib','context_samples','rss_tree_upper_bound_mib'}
    if set(verified)!=set(expected)|details or not matches(verified,expected):raise ValueError('original A1 verified receipt semantics differ')
    legacy.measured_resources(verified)
    native=legacy.train_output_binding(value,trained,engine,config,engine['_runtime'](False));evidence={}
    def remember(p,pin=None):
        p=secure(p);actual=digest(p)
        if pin is not None and pin!=actual:raise ValueError('replayed evidence pin differs')
        if str(p) in evidence and evidence[str(p)]!=actual:raise ValueError('replayed evidence changed')
        evidence[str(p)]=actual;return actual
    for stage in ('preflight','train','verify'):
        successful(v,path,pin,stage)
        for record in stage_inputs(v,path,pin,stage):remember(record['path'],record['sha256'])
        run=secure(v['ops_root'])/'runs'/RUN_IDS[stage]
        for name in ('run.json','terminal.json','logs/stdout.log'):remember(run/name)
    for stage in ('preflight','train','verify'):
        for parent,dirs,names in os.walk(output(v,stage),followlinks=False):
            if any((Path(parent)/name).is_symlink() for name in dirs):raise ValueError('A1 evidence directory symlink')
            for name in names:remember(Path(parent)/name)
    # The native EarlyStop and integer-hit functions are the same objects used
    # by the original isolated acceptance reader; no rounded recall shortcut.
    scratch.mkdir(parents=True,exist_ok=False)
    validation_context,engineering,_=engine['_contexts'](*common,config,scratch/'contexts',**extra)
    if engineering is not False:raise ValueError('formal original validation context required')
    from vkgr.sid_depth import SIDTokensLD
    tokens=SIDTokensLD(validation_context.codec,50);original=core.check_native();stop=original.EarlyStop(config['max_epochs'],config['min_epochs'],config['patience'])
    selected=None;selected_rows=None;selected_metrics=None;trained_path=secure(v['output_root'])/'train/native/manifest.json'
    for row in native['epochs']:
        aggregate,computation,rows=engine['_read_evaluation'](row['evaluation']['kind'],trained_path.parent,row,validation_context,tokens)
        if aggregate!=row['metrics'] or computation!=row['evaluation']['computation_sha256']:raise ValueError('original epoch validation rows differ')
        metrics=aggregate['metrics']['all_eligible']
        if stop.observe(row['epoch'],metrics['recall@50'],metrics['ndcg@50']):selected=row;selected_rows=rows;selected_metrics=aggregate
    if not stop.stopped or stop.state()!=native['selection'] or selected is None or native['best']['epoch']!=selected['epoch'] or native['best']['sha256']!=selected['checkpoint_sha256'] or native['best']['metric']!=list(stop.best_metric):raise ValueError('original selected epoch replay differs')
    hits=core.validation_hits(selected_rows,validation_context.queries,selected_metrics)
    if type(verified['validation_hits'])is not int or hits!=verified['validation_hits']:raise ValueError('original integer validation hit numerator differs')
    expected_native={'status':'verified','replayed_epochs':[selected['epoch']],'best_epoch':selected['epoch'],'epochs':len(native['epochs']),
        'run_id':trained_path.parent.parent.name,'manifest_sha256':trained['native_manifest_sha256'],'verify_run_id':RUN_IDS['verify']}
    if not exact(verified['native_verification'],expected_native):raise ValueError('original native verify binding differs')
    replay_path=secure(v['output_root'])/'verify/native'/('replay-best-%03d'%selected['epoch'])
    replay_record={**selected,'evaluation':{**selected['evaluation'],'path':replay_path.name,'manifest_sha256':remember(replay_path/'manifest.json')}}
    aggregate,computation,_=engine['_read_evaluation'](selected['evaluation']['kind'],replay_path.parent,replay_record,validation_context,tokens)
    if computation!=selected['evaluation']['computation_sha256'] or aggregate!=selected_metrics:raise ValueError('original native verify published replay differs')
    cpu_request=module.legacy.load_request(Path(request['cpu_request']),request['cpu_request_sha256']);cpu_manifest=json.loads(core.file_bytes({'path':str(secure(cpu_request['fit_output'])/'run_manifest.json'),'sha256':got['cpu_reference']['manifest_sha256']}))
    collision=sid.science.collision_gate(cpu_manifest['audit'],got['loaded'].manifest['audit'])
    if not exact(collision,got['loaded'].manifest['collision_gate']):raise ValueError('original integer collision proof differs')
    proof={'health':got['native_input']['sgns_receipt']['native_audit_verification']['health'],'sid_gates':{'cpu':cpu_manifest['gates'],'cuda':got['loaded'].manifest['gates']},
        'collision':collision,'validation_hits':hits,'hits_gate':378,'a1_native':expected_native,'evidence':evidence}
    acceptance.check_proof(proof)
    if identity(v)!=(source,env) or not exact(load(path,pin),v) or not exact(producer_closure(v)[2]['job_proof'],got['job_proof']):raise ValueError('four-gate source/input/runtime changed')
    for p,h in evidence.items():
        if digest(secure(p))!=h:raise ValueError('four-gate evidence changed')
    return proof


def accept(path,pin):
    v=load(path,pin);source,env=identity(v);own=own_job(v,path,pin,'accept');out=output(v,'accept')
    if out.exists():raise FileExistsError('exclusive four-gate output required')
    scratch=secure(v['scratch_root']);proof=replay(v,path,pin,scratch.with_name(scratch.name+'-four-gate'))
    if job_binding(v,path,pin,'accept')!=own:raise ValueError('four-gate own job changed')
    receipt={'schema':TOTAL,'authority':'4b0a8a1','backend':'B','status':'verified','verified':True,'subset_four_gate_acceptance':True,'engineering':False,'test_read':False,'training':False,
        'run_id':RUN_IDS['accept'],'run_manifest_sha256':own,'source':source,'runtime':env,'request':{'path':str(path),'sha256':pin},'proof':proof}
    out.mkdir(parents=True,exist_ok=False);sid.science.write_new(out/'stage-receipt.json',encoded(receipt));return receipt


def consume(path,pin,scratch):
    raw=core.file_bytes({'path':str(path),'sha256':pin});receipt=json.loads(raw)
    expected={'schema':TOTAL,'authority':'4b0a8a1','backend':'B','status':'verified','verified':True,'subset_four_gate_acceptance':True,'engineering':False,'test_read':False,'training':False,'run_id':RUN_IDS['accept']}
    if not matches(receipt,expected):raise ValueError('verified native B total acceptance receipt required')
    req=receipt['request'];v=load(Path(req['path']),req['sha256']);source,env=identity(v)
    if secure(path)!=output(v,'accept')/'stage-receipt.json' or not exact(receipt.get('source'),source) or not exact(receipt.get('runtime'),env):raise ValueError('total receipt source/runtime/output differs')
    own=job_binding(v,Path(req['path']),req['sha256'],'accept');run=secure(v['ops_root'])/'runs'/RUN_IDS['accept']
    values={name:(run/name).read_bytes() for name in ('run.json','terminal.json','logs/stdout.log')};terminal=json.loads(values['terminal.json'])
    if not matches(terminal,{'schema':'vkgr-terminal-v1','run_id':RUN_IDS['accept'],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}) or not exact(acceptance.last_receipt(values['logs/stdout.log']),receipt) or receipt.get('run_manifest_sha256')!=own:raise ValueError('total receipt terminal/stdout/job differs')
    acceptance.check_proof(receipt['proof'])
    if not exact(replay(v,Path(req['path']),req['sha256'],scratch),receipt['proof']):raise ValueError('total receipt differs from independently recomputed original evidence')
    if core.file_bytes({'path':str(path),'sha256':pin})!=raw or job_binding(v,Path(req['path']),req['sha256'],'accept')!=own or identity(v)!=(source,env) or not exact(load(Path(req['path']),req['sha256']),v):raise ValueError('total acceptance snapshot changed')
    if any((run/name).read_bytes()!=data for name,data in values.items()):raise ValueError('total original terminal/stdout changed')
    return {'receipt_sha256':pin,'request_sha256':req['sha256'],'producer':{name:hashlib.sha256(data).hexdigest() for name,data in values.items()},'proof':receipt['proof']}


def prepare(stage,path,pin,ops_code):
    from analysis.a40_ops import load_ops
    v=load(path,pin,frozen=False)
    if (stage=='derive')!=(v['schema']==SOURCE_SCHEMAS['derive']):raise ValueError('matching derive/A1 request required')
    if source_identity()!=v['source'] or secure(ops_code)!=secure(v['ops_code']):raise ValueError('fixed deployment/OPS required')
    # The launcher dry hashes dependencies, never claims producer acceptance.
    rid=RUN_IDS[stage];ops=secure(v['ops_root'])
    if (ops/'runs'/rid).exists() or output(v,stage).exists():raise FileExistsError('new one-shot run/output required')
    for state in ('pending','claimed','rejected'):
        if list((ops/'queue'/state).glob('*-'+rid+'.json')):raise FileExistsError('one-shot stage already queued')
    resource=resources(v,stage);_,launcher,_,_=load_ops(ops_code)
    args=['--root',str(ops),'--run-id',rid,'--kind','ops','--cwd',str(ROOT),'--config',str(path),'--seed','42']
    for key in ('cpu_cores','ram_mib','vram_mib','disk_gib'):args+=['--'+key.replace('_','-'),str(resource[key])]
    args+=['--wall-hours',str(resource['wall_seconds']/3600)]
    for key,val in fixed_env(v,stage).items():args+=['--env',key+'='+val]
    for record in stage_inputs(v,path,pin,stage):args+=['--input',record['path']]
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*args,'--',*invocation(stage,path,pin)]));metadata=launcher.prepare(spec)
    env=launcher._recorded_env(metadata['env'],launcher._job_env(metadata['env'],ops/'runs'/rid,rid))
    if not exact(env,contract(v,path,pin,stage)['env']):raise ValueError('unexpected inherited launcher environment')
    return {'status':'prepared_dry','spec':spec,'metadata':metadata,'submitted':False,'training':False,'real_preflight':False,'subset_four_gate_acceptance':False,
        'resource_scope':'not_measured conservative first phase declaration; input preflight is not a training-peak measurement' if stage in ('derive','preflight','train') else 'successful training context-inclusive GPU and RSS upper-bound x1.15 ceil256'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    for name in (*RUN_IDS,'plan','consume'):
        p=sub.add_parser(name,allow_abbrev=False);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True)
        if name=='plan':p.add_argument('--stage',choices=tuple(RUN_IDS),required=True);p.add_argument('--ops-code',type=Path,required=True)
        if name=='consume':p.add_argument('--scratch',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='dry':result={'status':'dry','backend':'B','training':False,'seed':42,'hits_gate':378,'test_read':False,'subset_four_gate_acceptance':False}
        elif args.command=='plan':result=prepare(args.stage,args.request,args.request_sha256,args.ops_code)
        elif args.command=='derive':result=derive(args.request,args.request_sha256)
        elif args.command=='accept':result=accept(args.request,args.request_sha256)
        elif args.command=='consume':result=consume(args.request,args.request_sha256,args.scratch)
        else:result=execute(args.command,args.request,args.request_sha256)
        print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True);return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc),'training_may_have_started':args.command=='train'},sort_keys=True),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
