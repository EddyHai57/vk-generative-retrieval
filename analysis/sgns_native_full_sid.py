"""UR001 original BHKM transport and explicit full-native-B producer DAG."""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import numpy as np
from analysis import sgns_native_full as full
from analysis.ur001_derive_cli import secure,digest,encoded,source_identity
from analysis.a40_ops import load_ops,PLAN_MIB
from analysis.sgns_subset_acceptance import source_tree
from analysis.sgns_subset_acceptance_a1 import transport_job
from vkgr import sid,sid_tok_audit as au,sid_v2
ROOT=Path(__file__).resolve().parents[1]
MODULE='analysis.sgns_native_full_sid'
SCHEMA='UR001-native-full-SID-trie-v1'
FULL_ENTRY_SHA='b8799ee1c8fcbdfc3dedee9d3b590cfde25c970c70f40d9f20d151fc2873730c'
ARTIFACTS=('mapping.npz','trie.npz','nodes.json','audit.json','config.json')
SCIENCE={'src/vkgr/sid.py':'9602580f907cc2e8cfa3684aec7828df2bdfc87b59e71b6fd3865a9a19de91a3','src/vkgr/sid_tok_audit.py':'d4f0afb554a6f48f787c91c81ffcb0baa7360a614b91152aa2d73cfd91b3f8ba'}
SCIENCE.update({'src/vkgr/sid_v2.py':'d094606ddab0460ef470e7b7cf86793af2c4352c8237842a803d6cd1c06470ad','src/vkgr/generative_data.py':'ea4357989c3a520e7c7dbc1dfce2bbeff192f3d89e4e5f0703cfd27bf8e0c7c7','src/vkgr/data_audit.py':'c7f5f28cdb7e19181fc46f8d69140dba9711ff66f5b8537399692e37e4ded182'})
exact=full.exact
catalog_digest=sid_v2.catalog_digest


def config(n):
    k=2
    if type(n)is not int or n<8:raise ValueError('catalog too small for frozen three-level rule')
    while (k+1)**3<=n:k+=1
    return {'schema':'UR001-BHKM-config-v1','branching':k,'depth':3,'seed':42,'max_iter':10,'external_sort':False}


def check_arrays(ids,vectors,binding,*,expected_content=None):
    if ids.dtype!=np.uint32 or ids.ndim!=1 or len(ids)<2 or np.any(ids[1:]<=ids[:-1]):raise ValueError('strict ascending uint32 catalog required')
    if vectors.dtype!=np.float32 or vectors.shape!=(len(ids),64) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors.astype('f8'),axis=1),1,atol=2e-6,rtol=0):raise ValueError('original finite float32 unit-row contract required')
    for name in ('catalog_digest','vectors_sha256','content_feature_sha256'):full.subset.hexsha(binding[name])
    if binding['catalog_digest']!=catalog_digest(ids) or binding['vectors_sha256']!=au.feature_digest(vectors):raise ValueError('catalog/H5b bytes differ')
    if expected_content is not None and binding['content_feature_sha256']!=expected_content:raise ValueError('H5a original full producer content differs')


def mapping_gates(audit,content,expected_content,vectors_sha):
    inherited=au.gates(audit,'bhkm',expected_content_sha256=vectors_sha)
    k=audit['branching'];used=[audit['positions'][i]['distinct'] for i in range(3)]
    checks=[*inherited['checks'],{'id':'H5a','ok':content==expected_content,'actual':content,'expected':expected_content},
        {'id':'H5b','ok':audit['content_sha256']==vectors_sha,'actual':audit['content_sha256'],'expected':vectors_sha},
        au._check('R1',all(au.usage_ok(u,k) for u in used),used,[k,list(au.USAGE_NUM_DEN)])]
    return {'passed':all(c['ok'] for c in checks),'checks':checks}


def fit_model(ids,vectors,boundary,*,content=None,expected_content=None):
    cfg=config(len(ids));vector_sha=au.feature_digest(vectors)
    content=vector_sha if content is None else content;expected_content=content if expected_content is None else expected_content
    check_arrays(ids,vectors,{'catalog_digest':catalog_digest(ids),'vectors_sha256':vector_sha,'content_feature_sha256':content},expected_content=expected_content)
    model=sid.fit(ids,vectors,**{k:cfg[k] for k in ('branching','depth','seed','max_iter')},on_boundary=boundary)
    boundary();audit=au.audit_mapping(model.ids,model.sids,vectors,branching=cfg['branching'],kind='bhkm')
    gates=mapping_gates(audit,content,expected_content,vector_sha)
    if gates['passed']is not True:raise ValueError('original H/B and R1/prefix gates failed')
    return model,audit,gates


def check_rss(rss,planned):
    if type(rss)is not int or rss<0 or type(planned)is not int or planned<=0 or rss*5>planned*3:raise RuntimeError('STOP process-tree RSS unavailable or exceeds planned RAM 60%')


def tree_rss(common):
    table=common.process_table(common.Probes());pid=os.getpid()
    if type(table)is not dict or pid not in table:raise RuntimeError('own process tree unavailable')
    pids=common.descendants(table,pid)
    if any(p not in table or type(table[p].get('rss_kib'))is not int or table[p]['rss_kib']<=0 for p in pids):raise RuntimeError('own process RSS unavailable')
    return sum(table[p]['rss_kib'] for p in pids)*1024


@contextmanager
def rss_guard(common):
    total,_=common.ram_info(common.Probes())
    if total!=PLAN_MIB:raise RuntimeError('original capped OPS RAM planning differs')
    stopped=threading.Event();planned=PLAN_MIB*1024**2
    def boundary():check_rss(tree_rss(common),planned)
    def watch():
        while not stopped.wait(1.):
            try:boundary()
            except Exception:
                print(json.dumps({'status':'STOP','reason':'process-tree RSS guard','planned_ram_mib':PLAN_MIB}),file=sys.stderr,flush=True)
                os.kill(os.getpid(),signal.SIGTERM);return
    boundary();thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:yield boundary
    finally:stopped.set();thread.join()


def record(path):return {'path':str(secure(path)),'sha256':digest(secure(path))}
def read(recorded):
    if type(recorded)is not dict or set(recorded)!={'path','sha256'}:raise ValueError('exact file record required')
    data=secure(recorded['path']).read_bytes()
    if hashlib.sha256(data).hexdigest()!=full.subset.hexsha(recorded['sha256']):raise ValueError('pinned bytes differ')
    return data


def validate(v):
    keys={'schema','source','lock','producer','ops_code','ops_root','output','preflight_run_id','fit_run_id','verify_run_id','resources','engineering','test_read'}
    if type(v)is not dict or set(v)!=keys or v['schema']!='UR001-native-full-SID-request-v1' or v['engineering']is not False or v['test_read']is not False:raise ValueError('closed UR001 request engineering=False and no holdout required')
    p=v['producer']
    if type(p)is not dict or set(p)!={'root','source','request','receipt','verified_producer'}:raise ValueError('complete full-native-B producer required')
    full.subset.hexsha(p['source'],40)
    proof=p['verified_producer']
    if type(proof)is not dict or set(proof)!={'run_id','receipt_sha256','run_manifest_sha256','terminal_sha256','stdout_sha256'} or type(proof['run_id'])is not str:raise ValueError('original native full verified job proof required')
    for key in ('receipt_sha256','run_manifest_sha256','terminal_sha256','stdout_sha256'):full.subset.hexsha(proof[key])
    for r in (v['lock'],p['request'],p['receipt']):
        if type(r)is not dict or set(r)!={'path','sha256'}:raise ValueError('explicit original file pin required')
        secure(r['path']);full.subset.hexsha(r['sha256'])
    for path in (p['root'],v['ops_code'],v['ops_root'],v['output']):secure(path)
    for value in (p['root'],p['request']['path'],p['receipt']['path'],v['lock']['path']):
        if any(part in ('test','holdout') or part.startswith(('week_26','week26')) for part in Path(value).parts):raise ValueError('holdout forbidden before file read')
    for key in ('preflight_run_id','fit_run_id','verify_run_id'):
        if type(v[key])is not str or not v[key].startswith('ur001-native-full-sid-') or set(v[key])-set('abcdefghijklmnopqrstuvwxyz0123456789-'):raise ValueError('fixed descriptive run ID required')
    if len({v[k] for k in ('preflight_run_id','fit_run_id','verify_run_id')})!=3:raise ValueError('separate stage run IDs required')
    if type(v['resources'])is not dict or set(v['resources'])!={'cpu_cores','ram_mib','vram_mib','disk_gib','wall_hours'} or any(type(x)is not int or x<0 for x in v['resources'].values()) or any(v['resources'][x]<=0 for x in ('cpu_cores','ram_mib','disk_gib','wall_hours')) or v['resources']['vram_mib']!=0:raise ValueError('positive CPU-only resource declaration required')
    protected=[ROOT,secure(p['root']),secure(v['ops_root']),secure(v['ops_code']),*(secure(r['path']) for r in (v['lock'],p['request'],p['receipt']))]
    out=secure(v['output'])
    if any(out==q or out in q.parents or q in out.parents for q in protected):raise ValueError('exclusive output separate from inputs/code/OPS required')
    return v


def load(path,pin):return validate(json.loads(read({'path':str(path),'sha256':pin})))
def identity(v):
    source_tree(ROOT,v['source'])
    if any(v['source']['files'].get(name)!=pin for name,pin in SCIENCE.items()):raise ValueError('original scientific files differ')
    return {'source':v['source'],'runtime':json.loads(encoded(full.original.runtime(secure(v['lock']['path']))))}


def parent_snapshot(v):
    p=v['producer'];root=secure(p['root']);source=full.pinned_tree(root,p['source'])
    if source['files'].get('analysis/sgns_native_full.py')!=FULL_ENTRY_SHA:raise ValueError('fixed full B consumer entry differs')
    request=json.loads(read(p['request']));receipt=json.loads(read(p['receipt']));out=secure(request['output'])/'fit';run=secure(request['ops_root'])/'runs'/request['fit_run_id']
    if secure(p['receipt']['path'])!=out/'result.json' or request['runner_source']!=p['source'] or receipt.get('request_sha256')!=p['request']['sha256']:raise ValueError('original full fit request/receipt path differs')
    full.validate(request)
    if request['lock']!=v['lock']['path']:raise ValueError('full B/SID lock path differs')
    job=json.loads((run/'run.json').read_bytes());records=[p['request'],p['receipt'],v['lock'],*(record(run/n) for n in ('run.json','terminal.json','logs/stdout.log'))]
    records.extend({'path':item['path'],'sha256':item['sha256']} for item in job['inputs'])
    records.extend({'path':str(root/n),'sha256':pin} for n,pin in source['files'].items())
    oldroot=secure(request['input_producer']['root']);old=receipt['input_binding']['original']['source']
    records.extend({'path':str(oldroot/n),'sha256':pin} for n,pin in old['files'].items())
    if set(receipt.get('artifacts',{}))!={'vectors.npy','audit.json'}:raise ValueError('complete native B artifacts required')
    records.extend({'path':str(out/n),'sha256':pin} for n,pin in receipt['artifacts'].items())
    ledger={}
    for r in records:
        if r['path']in ledger and ledger[r['path']]!=r['sha256']:raise ValueError('conflicting parent role pin')
        if digest(secure(r['path']))!=full.subset.hexsha(r['sha256']):raise ValueError('parent evidence bytes differ')
        ledger[r['path']]=r['sha256']
    return source,request,receipt,ledger


def parent(v):
    before=parent_snapshot(v);source,request,receipt,ledger=before;p=v['producer'];root=secure(p['root'])
    env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    args=[sys.executable,'-B','-m','analysis.sgns_native_full','verify','--stage','fit','--request',p['request']['path'],'--request-sha256',p['request']['sha256']]
    got=subprocess.run(args,cwd=root,env=env,text=True,capture_output=True)
    if got.returncode:raise ValueError('original full-native-B verify rejected')
    result=json.loads(got.stdout)
    if not full.matches(result,{'status':'verified','readonly':True}) or not exact(result.get('producer'),p['verified_producer']):raise ValueError('full-native-B original verified producer differs')
    if not exact(parent_snapshot(v),before):raise ValueError('full B producer DAG changed during verify')
    arrays_root=secure(request['input_producer']['bundle']);ids=np.asarray(json.loads((arrays_root/'catalog.json').read_bytes()),dtype=np.uint32)
    vectors=np.load(secure(Path(request['output'])/'fit/vectors.npy'),allow_pickle=False)
    audit=json.loads(secure(Path(request['output'])/'fit/audit.json').read_bytes());binding=receipt['input_binding']['original']['manifest']['binding']
    content=binding['content_feature_sha256'];bound={'catalog_digest':catalog_digest(ids),'vectors_sha256':audit['vectors_sha256'],'content_feature_sha256':content,'stage2_manifest_sha256':binding['stage2_manifest_sha256']}
    full.subset.hexsha(bound['stage2_manifest_sha256'])
    if audit.get('content_sha256')!=content or audit.get('catalog_sha256')!=bound['catalog_digest']:raise ValueError('H5a native full audit/input catalog differs')
    check_arrays(ids,vectors,bound,expected_content=binding['content_feature_sha256'])
    if len(ids)!=full.original.fixed_dag()['counts']['catalog_items']:raise ValueError('formal full catalog count differs')
    if not exact(identity(v)['runtime'],receipt['input_binding']['original']['runtime']):raise ValueError('SID/full B runtime differs')
    if not exact(parent_snapshot(v),before):raise ValueError('full B input bytes changed during loading')
    return ids,vectors,{'reference':p,'source':source,'runtime':identity(v)['runtime'],'binding':bound,'input_manifest_sha256':request['input_producer']['manifest_sha256'],'input_feature_sha256':receipt['artifacts']['vectors.npy'],'evidence':ledger}


def write_bundle(root,model,audit,gates,parent,source,runtime,request_sha):
    root=secure(root);root.mkdir(parents=True,exist_ok=False)
    sid_v2._write_npz(root/'mapping.npz',{'ids':model.ids,'sids':model.sids})
    sid_v2._write_npz(root/'trie.npz',{'keys':model.keys,'key_items':model.key_items,'bases':np.asarray(model.bases,dtype=np.uint64)})
    for name,value in [('nodes.json',model.nodes),('audit.json',{'mapping':audit,'gates':gates}),('config.json',config(len(model.ids)))]:
        with (root/name).open('xb') as f:f.write(encoded(value))
    manifest={'schema':SCHEMA,'sid_version':f'UR001-BHKM-B{model.branching}D3-S42','engineering':False,'test_read':False,'seed':42,'branching':model.branching,'depth':3,
        'catalog_items':len(model.ids),'catalog_digest':catalog_digest(model.ids),'vectors_sha256':audit['content_sha256'],'suffix_vocabulary':model.suffix_vocabulary,'vocab_size':2+sum(model.bases),
        'content_feature_sha256':parent['binding']['content_feature_sha256'],'input_manifest_sha256':parent['input_manifest_sha256'],'input_feature_sha256':parent['input_feature_sha256'],'input_dimension':64,
        'parent':parent,'source':source,'runtime':runtime,'request_sha256':request_sha,'artifacts':{n:digest(root/n) for n in ARTIFACTS}}
    with (root/'manifest.json').open('xb') as f:f.write(encoded(manifest))
    return manifest


def read_bundle(path,pin,ids,vectors,parent,boundary,*,replay=False):
    raw=read({'path':str(path),'sha256':pin});m=json.loads(raw);root=secure(path).parent
    keys={'schema','sid_version','engineering','test_read','seed','branching','depth','catalog_items','catalog_digest','vectors_sha256','suffix_vocabulary','vocab_size','content_feature_sha256','input_manifest_sha256','input_feature_sha256','input_dimension','parent','source','runtime','request_sha256','artifacts'}
    if set(m)!=keys or m['schema']!=SCHEMA or m['engineering']is not False or m['test_read']is not False or not exact(m['parent'],parent) or not exact(m['seed'],42) or not exact(m['depth'],3) or not exact(m['branching'],config(len(ids))['branching']) or m['sid_version']!=f"UR001-BHKM-B{m['branching']}D3-S42" or not exact(m['catalog_items'],len(ids)) or m['catalog_digest']!=catalog_digest(ids) or m['vectors_sha256']!=au.feature_digest(vectors) or set(m['artifacts'])!=set(ARTIFACTS):raise ValueError('closed UR001 SID manifest differs')
    if m['content_feature_sha256']!=parent['binding']['content_feature_sha256'] or m['input_manifest_sha256']!=parent['input_manifest_sha256'] or m['input_feature_sha256']!=parent['input_feature_sha256'] or not exact(m['input_dimension'],64):raise ValueError('original full input manifest/feature/H5a identity differs')
    files={n:read({'path':str(root/n),'sha256':m['artifacts'][n]}) for n in ARTIFACTS}
    with np.load(io.BytesIO(files['mapping.npz']),allow_pickle=False) as a:
        if set(a.files)!={'ids','sids'}:raise ValueError('mapping members differ')
        found_ids,codes=a['ids'],a['sids']
    if not np.array_equal(found_ids,ids) or found_ids.dtype!=np.uint32 or codes.dtype!=np.uint32:raise ValueError('mapping catalog/dtype differs')
    content=parent['binding']['content_feature_sha256'];stored_nodes=json.loads(files['nodes.json'])
    if replay:
        model,audit,gates=fit_model(ids,vectors,boundary,content=content,expected_content=content)
        if not np.array_equal(codes,model.sids) or not exact(stored_nodes,model.nodes):raise ValueError('original same-seed SID/nodes replay differs')
    else:
        boundary();audit=au.audit_mapping(found_ids,codes,vectors,branching=m['branching'],kind='bhkm')
        gates=mapping_gates(audit,content,content,au.feature_digest(vectors))
    if gates['passed']is not True or not exact(json.loads(files['audit.json']),{'mapping':audit,'gates':gates}) or not exact(json.loads(files['config.json']),config(len(ids))):raise ValueError('original H-R/config differs')
    codec=sid_v2.build_codec(found_ids,codes,m['branching'],catalog_ids=ids)
    with np.load(io.BytesIO(files['trie.npz']),allow_pickle=False) as a:
        if set(a.files)!={'keys','key_items','bases'} or a['keys'].dtype!=np.uint64 or a['key_items'].dtype!=np.uint32 or a['bases'].dtype!=np.uint64 or any(not np.array_equal(a[n],np.asarray(getattr(codec.catalog,n))) for n in a.files):raise ValueError('trie reconstruction differs')
    if not exact(m['suffix_vocabulary'],codec.catalog.suffix_vocabulary) or not exact(m['vocab_size'],codec.vocab_size):raise ValueError('codec namespace differs')
    if any((root/n).read_bytes()!=b for n,b in files.items()) or secure(path).read_bytes()!=raw:raise ValueError('SID bundle bytes changed')
    return codec,m


def rid(v,stage):return v[stage+'_run_id']
def argv(stage,path,pin):return [sys.executable,'-B','-m',MODULE,stage,'--request',str(path),'--request-sha256',pin,'--execute-on-new-instance']
def env():return {'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
def resources(v):return {**{k:v['resources'][k] for k in ('cpu_cores','ram_mib','vram_mib','disk_gib')},'wall_seconds':float(v['resources']['wall_hours']*3600)}
def pins(v,path,stage):
    records={str(secure(path)):digest(path),v['lock']['path']:v['lock']['sha256'],**parent_snapshot(v)[3]}
    if stage in ('fit','verify'):
        prior=secure(v['ops_root'])/'runs'/rid(v,'preflight')
        for p in [*(prior/n for n in ('run.json','terminal.json','logs/stdout.log')),secure(v['output'])/'preflight/stage-receipt.json']:records[str(p)]=digest(secure(p))
    if stage=='verify':
        fit=secure(v['output'])/'fit';run=secure(v['ops_root'])/'runs'/rid(v,'fit')
        for p in [*(run/n for n in ('run.json','terminal.json','logs/stdout.log')),fit/'stage-receipt.json',fit/'manifest.json',*(fit/n for n in ARTIFACTS)]:records[str(p)]=digest(secure(p))
    return [{'path':name,'sha256':pin} for name,pin in sorted(records.items())]


def job(v,path,pin,stage):
    root=secure(v['ops_root'])/'runs'/rid(v,stage);raw=secure(root/'run.json').read_bytes();recorded=json.loads(raw)
    transport_job(recorded,runroot=root,code=ROOT,source=v['source'],argv=argv(stage,path,pin),config={'path':str(path),'sha256':pin},inputs=pins(v,path,stage),resources=resources(v))
    return hashlib.sha256(raw).hexdigest()


def own_snapshot(v,path,pin,stage):
    ident=identity(v);loaded=load(path,pin)
    if not exact(loaded,v):raise ValueError('own request changed')
    run=secure(v['ops_root'])/'runs'/rid(v,stage);out=secure(v['output'])/stage
    paths=[*(run/n for n in ('run.json','terminal.json','logs/stdout.log')),out/'stage-receipt.json']
    if stage=='fit':paths.extend([out/'manifest.json',*(out/n for n in ARTIFACTS)])
    snapshots={str(p):digest(secure(p)) for p in paths}
    inputs=pins(v,path,stage);own=job(v,path,pin,stage)
    if not exact(identity(v),ident) or not exact(load(path,pin),v) or not exact(pins(v,path,stage),inputs) or job(v,path,pin,stage)!=own or any(digest(secure(n))!=sha for n,sha in snapshots.items()):raise ValueError('own verification snapshot changed')
    return {'identity':ident,'request_sha256':pin,'inputs':inputs,'job':own,'files':snapshots}


def accepted(v,path,pin,stage,boundary):
    before=own_snapshot(v,path,pin,stage);run=secure(v['ops_root'])/'runs'/rid(v,stage);out=secure(v['output'])/stage
    raw=(out/'stage-receipt.json').read_bytes();receipt=json.loads(raw)
    terminal=json.loads((run/'terminal.json').read_bytes());full.check_terminal({'preflight_run_id':rid(v,stage),'fit_run_id':rid(v,stage),'resources':v['resources']},'fit',terminal)
    last=full.original.receipt_from_bytes((run/'logs/stdout.log').read_bytes())
    expected={'schema':'UR001-native-full-SID-stage-v1','status':'verified','verified':True,'engineering':False,'test_read':False,'stage':stage,'run_id':rid(v,stage),'run_manifest_sha256':before['job'],'request':{'path':str(path),'sha256':pin},**before['identity']}
    if not full.matches(receipt,expected) or not exact(last,receipt):raise ValueError('SID original terminal/stdout/receipt binding differs')
    ids,vectors,parent_proof=parent(v)
    if not exact(receipt.get('parent'),parent_proof):raise ValueError('SID parent proof differs')
    codec=None;m=None
    if stage in ('fit','verify'):
        if receipt.get('same_seed_replay')is not True or receipt.get('fit_verify')is not True:raise ValueError('original full SID fit-verify/replay required')
        manifest=receipt['manifest']
        if manifest['path']!=str(secure(v['output'])/'fit/manifest.json'):raise ValueError('own SID manifest path differs')
        codec,m=read_bundle(Path(manifest['path']),manifest['sha256'],ids,vectors,parent_proof,boundary)
        if not exact(m['source'],v['source']) or not exact(m['runtime'],before['identity']['runtime']) or m['request_sha256']!=pin:raise ValueError('SID manifest source/runtime/request differs')
        if stage=='verify' and (receipt.get('independent_replay')is not True or not exact(receipt.get('fit_producer'),accepted(v,path,pin,'fit',boundary)['producer'])):raise ValueError('original independent verify/replay proof differs')
    elif not full.matches(receipt,{'real_readonly_preflight':True,'temporary_parameter_updates':False,'vectors_saved':False}):raise ValueError('real readonly full SID preflight required')
    if not exact(own_snapshot(v,path,pin,stage),before):raise ValueError('SID own evidence changed during verification')
    producer={'run_id':rid(v,stage),'receipt_sha256':hashlib.sha256(raw).hexdigest(),'run_manifest_sha256':before['files'][str(run/'run.json')],'terminal_sha256':before['files'][str(run/'terminal.json')],'stdout_sha256':before['files'][str(run/'logs/stdout.log')]}
    if stage=='preflight':return {'producer':producer}
    info={k:m[k] for k in ('sid_version','branching','depth','suffix_vocabulary','vocab_size','catalog_digest')}
    info.update(kind='bhkm',engineering=False,manifest_sha256=receipt['manifest']['sha256'])
    return {'manifest':receipt['manifest'],'producer':producer,**before['identity'],'parent':parent_proof,'codec':info}


def execute(v,path,pin,stage,boundary):
    ident=identity(v);ids,vectors,parent_proof=parent(v);before=pins(v,path,stage);own=job(v,path,pin,stage)
    if not exact({k:os.environ.get(k) for k in {**env(),'VKGR_RUN_ID':rid(v,stage),'VKGR_PROGRESS_PATH':str(secure(v['ops_root'])/'runs'/rid(v,stage)/'progress.json')}},{**env(),'VKGR_RUN_ID':rid(v,stage),'VKGR_PROGRESS_PATH':str(secure(v['ops_root'])/'runs'/rid(v,stage)/'progress.json')}):raise ValueError('own fixed process environment required')
    out=secure(v['output'])/stage
    if out.exists():raise FileExistsError('exclusive new stage output required')
    boundary();manifest=None;output_pins={}
    if stage=='preflight':
        cfg=config(len(ids));check_arrays(ids,vectors,parent_proof['binding'],expected_content=parent_proof['binding']['content_feature_sha256'])
        # No fit or temporary parameter update in the formal PF.
        summary={'real_readonly_preflight':True,'temporary_parameter_updates':False,'vectors_saved':False,'config':cfg}
    elif stage=='fit':
        accepted(v,path,pin,'preflight',boundary)
        content=parent_proof['binding']['content_feature_sha256'];model,audit,gates=fit_model(ids,vectors,boundary,content=content,expected_content=content)
        write_bundle(out,model,audit,gates,parent_proof,ident['source'],ident['runtime'],pin)
        manifest=record(out/'manifest.json')
        read_bundle(out/'manifest.json',manifest['sha256'],ids,vectors,parent_proof,boundary,replay=True)
        output_pins={n:digest(out/n) for n in ('manifest.json',*ARTIFACTS)}
        summary={'same_seed_replay':True,'fit_verify':True,'vectors_saved':False}
    else:
        fitted=accepted(v,path,pin,'fit',boundary);manifest=fitted['manifest'];content=parent_proof['binding']['content_feature_sha256']
        read_bundle(Path(manifest['path']),manifest['sha256'],ids,vectors,parent_proof,boundary,replay=True)
        summary={'same_seed_replay':True,'fit_verify':True,'independent_replay':True,'fit_producer':fitted['producer'],'vectors_saved':False}
    if not exact(identity(v),ident) or not exact(load(path,pin),v) or not exact(parent_snapshot(v)[3],parent_proof['evidence']) or not exact(pins(v,path,stage),before) or job(v,path,pin,stage)!=own:raise ValueError('SID stage immutable closure changed')
    if any(digest(secure(out/n))!=sha for n,sha in output_pins.items()):raise ValueError('new SID saved bundle changed during closure')
    if stage!='fit':out.mkdir(parents=True,exist_ok=False)
    result={'schema':'UR001-native-full-SID-stage-v1','status':'verified','verified':True,'engineering':False,'test_read':False,'stage':stage,'run_id':rid(v,stage),'run_manifest_sha256':own,'request':{'path':str(path),'sha256':pin},**ident,'parent':parent_proof,'manifest':manifest,'resource_measurement':'not_measured','planned_ram_mib':PLAN_MIB,**summary}
    with (out/'stage-receipt.json').open('xb') as stream:stream.write(encoded(result))
    return result


def prepare(v,path,pin,stage,launcher,boundary):
    identity(v);parent(v);boundary();run_id=rid(v,stage);ops=secure(v['ops_root'])
    if (ops/'runs'/run_id).exists() or (secure(v['output'])/stage).exists() or any(list((ops/'queue'/s).glob('*-'+run_id+'.json')) for s in ('pending','claimed','rejected')):raise FileExistsError('run/queue/output collision')
    flags=['--root',str(ops),'--run-id',run_id,'--kind','ops','--cwd',str(ROOT),'--config',str(path),'--seed','42']
    for name,n in v['resources'].items():flags.extend(['--'+name.replace('_','-'),str(n)])
    for name,value in env().items():flags.extend(['--env',name+'='+value])
    for rec in pins(v,path,stage):flags.extend(['--input',rec['path']])
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*argv(stage,path,pin)]));metadata=launcher.prepare(spec)
    return {'status':'prepared_dry','spec':spec,'metadata':metadata,'submitted':False,'training':False,'real_preflight':False}


def consume(receipt_path,sha256):
    raw=read({'path':str(receipt_path),'sha256':sha256});receipt=json.loads(raw);req=receipt['request'];v=load(req['path'],req['sha256'])
    if secure(receipt_path)!=secure(v['output'])/'verify/stage-receipt.json':raise ValueError('original independent SID verify receipt path differs')
    common,_,_,_=load_ops(v['ops_code'])
    with rss_guard(common) as boundary:result=accepted(v,Path(req['path']),req['sha256'],'verify',boundary)
    if secure(receipt_path).read_bytes()!=raw:raise ValueError('SID original receipt changed during consume')
    return result


def main(args=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    for action in ('plan','preflight','fit','verify'):
        p=sub.add_parser(action);p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True)
        if action=='plan':p.add_argument('--stage',choices=('preflight','fit','verify'),required=True)
        else:p.add_argument('--execute-on-new-instance',action='store_true',required=True)
    ns=parser.parse_args(args)
    try:
        if ns.command=='dry':result={'schema':SCHEMA,'training':False,'submitted':False,'formal_ready':False,'full_producer_required':True}
        else:
            v=load(ns.request,ns.request_sha256);common,launcher,_,_=load_ops(v['ops_code'])
            with rss_guard(common) as boundary:
                if ns.command=='plan':result=prepare(v,ns.request,ns.request_sha256,ns.stage,launcher,boundary)
                else:result=execute(v,ns.request,ns.request_sha256,ns.command,boundary)
        print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True);return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc)},sort_keys=True),flush=True);return 2


if __name__=='__main__':raise SystemExit(main())
