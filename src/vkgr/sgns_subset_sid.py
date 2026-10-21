"""Original BHKM48 and H/R gates with explicit new GPU-SGNS input bindings."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np
from . import sid, sid_tok_audit as au, sid_v2
from .data_audit import canonical_bytes
CONTENT_V1_FEATURE_SHA256=au.CONTENT_V1_FEATURE_SHA256
VARIANTS={'C1':(64,'GPU_SGNS_SUBSET_B48D3_S42')}
INPUT_REQUIRED={'schema','status','variant','dimension','engineering','catalog_sha256','vectors_sha256',
 'content_feature_sha256','train_sequence_sha256','script_sha256','commit','checkpoint_sha256','source_files'}
CONFIG={'schema':'structure-fit-v1','task_id':'SID-STRUCTURE-MATCHED','variant':'C1',
        'sid_version':'STRUCT_C1_B48D3_S42','input_dimension':64,'branching':48,'depth':3,'seed':42,'max_iter':10}
ADAPTER_MEMBERS=('src/vkgr/sgns_subset_sid.py','analysis/sgns_subset_sid.py',
                 'tests/test_sgns_subset_sid.py','tests/fixtures/sgns_subset_sid_launcher_chain.py')
SCIENTIFIC_FILES={'src/vkgr/sid.py':'9602580f907cc2e8cfa3684aec7828df2bdfc87b59e71b6fd3865a9a19de91a3',
 'src/vkgr/sid_v2.py':'d094606ddab0460ef470e7b7cf86793af2c4352c8237842a803d6cd1c06470ad',
 'src/vkgr/sid_tok_audit.py':'d4f0afb554a6f48f787c91c81ffcb0baa7360a614b91152aa2d73cfd91b3f8ba',
 'src/vkgr/generative_data.py':'ea4357989c3a520e7c7dbc1dfce2bbeff192f3d89e4e5f0703cfd27bf8e0c7c7',
 'src/vkgr/data_audit.py':'c7f5f28cdb7e19181fc46f8d69140dba9711ff66f5b8537399692e37e4ded182'}


def fit_bhkm_coll(ids,features,branching=16,depth=3,seed=42,max_iter=10,on_boundary=None):
    ids=np.asarray(ids);features=np.asarray(features)
    if ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=2**32) or len(np.unique(ids))!=len(ids):raise ValueError('invalid catalog IDs')
    if features.dtype!=np.float32 or (features.ndim!=2 or features.shape[0]!=len(ids) or features.shape[1] not in (64,128)) or not np.isfinite(features).all():raise ValueError('invalid normalized features')
    norms=np.linalg.norm(features,axis=1)
    if not np.all((norms==0)|np.isclose(norms,1,rtol=1e-5,atol=1e-6)):raise ValueError('features must match normalized P4 features')
    if any(type(v) is not int or v<1 for v in (branching,depth,max_iter)) or branching<2 or depth>8 or branching**depth*len(ids)>2**63 or type(seed) is not int or seed<0:raise ValueError('invalid tokenizer config')
    order=np.argsort(ids);ids=ids[order].astype(np.uint32);features=features[order]
    sids=np.zeros((len(ids),depth+1),dtype=np.uint32);nodes=[];rng=np.random.default_rng(seed)
    def visit(indices,prefix):
        if on_boundary is not None:on_boundary()
        level=len(prefix)
        if level==depth:
            sids[indices,-1]=np.arange(len(indices));return
        k=min(branching,len(indices));rows=features[indices];init=rng.choice(len(indices),k,replace=False)
        centers=rows[init].astype(np.float64);previous=None;objectives=[];converged=False
        for iteration in range(max_iter):
            if on_boundary is not None:on_boundary()
            distances=sid._distance(rows,centers);labels=sid._assign(distances,ids[indices])
            centers=np.stack([rows[labels==j].mean(axis=0,dtype=np.float64) for j in range(k)])
            objectives.append(float(sid._distance(rows,centers)[np.arange(len(rows)),labels].sum(dtype=np.float64)))
            if previous is not None and np.array_equal(labels,previous):converged=True;break
            previous=labels.copy()
        sizes=np.bincount(labels,minlength=k).tolist()
        nodes.append({'prefix':list(prefix),'init_item_ids':ids[indices[init]].astype(int).tolist(),'centers':centers.tolist(),'sizes':sizes,'iterations':iteration+1,'converged':converged,'objectives':objectives})
        for j in range(k):
            selected=indices[labels==j];sids[selected,level]=j;visit(selected,prefix+(j,))
    visit(np.arange(len(ids)),())
    return sid.SIDCatalog(ids,sids,branching,depth,nodes)


def _hex(value,n=64):
    if type(value) is not str or len(value)!=n or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid sha identity')
    return value


def validate_input(ids,vectors,manifest,*,engineering=False):
    """H5a pins original content; H5b independently pins the transformed producer output."""
    if type(engineering) is not bool or type(manifest) is not dict or not INPUT_REQUIRED<=set(manifest):
        raise ValueError('coll input manifest schema')
    if manifest['schema']!='coll-input-v1' or manifest['status']!='accepted' or manifest['engineering'] is not engineering:
        raise ValueError('accepted input with matching engineering flag required')
    if manifest['variant'] not in VARIANTS: raise ValueError('unregistered input variant')
    dim,_=VARIANTS[manifest['variant']]
    if type(manifest['dimension']) is not int or manifest['dimension']!=dim: raise ValueError('input dimension identity')
    if not isinstance(ids,np.ndarray) or ids.dtype!=np.uint32 or ids.ndim!=1 or len(ids)<2 or np.any(ids[1:]<=ids[:-1]):
        raise ValueError('strict ascending uint32 catalog required')
    if not engineering and len(ids)!=120415: raise ValueError('formal catalog must have 120415 items')
    if not isinstance(vectors,np.ndarray) or vectors.dtype!=np.float32 or vectors.shape!=(len(ids),dim):
        raise ValueError('float32 input dimension mismatch')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors.astype('float64'),axis=1),1,atol=2e-6,rtol=0):
        raise ValueError('input rows must be finite unit vectors')
    for k in ('catalog_sha256','vectors_sha256','content_feature_sha256','train_sequence_sha256','script_sha256'): _hex(manifest[k])
    _hex(manifest['commit'],40)
    if type(manifest['source_files']) is not dict or not manifest['source_files']: raise ValueError('input source files required')
    for name,sha in manifest['source_files'].items():
        if type(name) is not str or Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('relative source member required')
        _hex(sha)
    if manifest['script_sha256'] not in manifest['source_files'].values(): raise ValueError('script hash must be bound by source_files')
    if manifest['catalog_sha256']!=sid_v2.catalog_digest(ids): raise ValueError('catalog binding mismatch')
    if manifest['vectors_sha256']!=au.feature_digest(vectors): raise ValueError('H5b input digest mismatch')
    if not engineering and manifest['content_feature_sha256']!=CONTENT_V1_FEATURE_SHA256: raise ValueError('H5a CONTENT_V1 mismatch')
    if manifest['variant']=='C2': _hex(manifest['checkpoint_sha256'])
    elif manifest['checkpoint_sha256'] is not None: raise ValueError('C1 must not bind a checkpoint')
    return manifest


def audit_gates(ids,sids,vectors,manifest,*,engineering=False):
    validate_input(ids,vectors,manifest,engineering=engineering)
    audit=au.audit_mapping(ids,sids,vectors,branching=48,kind='bhkm')
    inherited=au.gates(audit,'bhkm',expected_content_sha256=manifest['vectors_sha256'])
    checks=[c for c in inherited['checks'] if c['id']!='H5']
    expected=manifest['content_feature_sha256'] if engineering else CONTENT_V1_FEATURE_SHA256
    checks.extend([{'id':'H5a','ok':manifest['content_feature_sha256']==expected,'actual':manifest['content_feature_sha256'],'expected':expected},
      {'id':'H5b','ok':audit['content_sha256']==manifest['vectors_sha256'],'actual':audit['content_sha256'],'expected':manifest['vectors_sha256']}])
    used=[audit['positions'][d]['distinct'] for d in range(3)]
    checks.append(au._check('R1',all(au.usage_ok(u,48) for u in used),used,[48,list(au.USAGE_NUM_DEN)]))
    return audit,{'passed':all(c['ok'] for c in checks),'checks':checks}



def read_pinned(path,sha):
    from analysis.ur001_derive_cli import secure
    raw=secure(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=_hex(sha):raise ValueError('pinned bytes differ')
    return raw


def write_new(path,raw):
    with Path(path).open('xb') as stream:stream.write(raw)


def check_source_delta(current,producer):
    expected={**producer['files'],**{p:current['files'].get(p) for p in ADAPTER_MEMBERS}}
    if (set(current['files'])!=set(producer['files'])|set(ADAPTER_MEMBERS)
            or current['files']!=expected or any(p in producer['files'] for p in ADAPTER_MEMBERS)
            or any(current['files'].get(p)!=sha for p,sha in SCIENTIFIC_FILES.items())):
        raise ValueError('exact new source delta or scientific source differs')
    return {p:current['files'][p] for p in ADAPTER_MEMBERS}


def collision_gate(cpu,gpu):
    n=cpu['items']
    if type(n) is not int or n<=0 or gpu['items']!=n:raise ValueError('same catalog required')
    a,b=cpu['semantic_collision_items'],gpu['semantic_collision_items']
    if any(type(x) is not int or not 0<=x<=n for x in (a,b)):raise ValueError('invalid collision counts')
    return {'passed':b*10<=a*11,'definition':'semantic_collision_items / items',
            'cpu_count':a,'gpu_count':b,'items':n,'comparison':'gpu_count * 10 <= cpu_count * 11',
            'cpu_rate':a/n,'gpu_rate':b/n}


def load_sgns_stage(bundle,bundle_sha,output,receipt_sha,run,device,lock,current,env):
    """Consume immutable output bytes tied to the successful original SGNS job."""
    import sys
    from analysis.sgns_gpu_subset import read_bundle,verify_original_fit
    from analysis.ur001_derive_cli import secure
    output,run,bundle,lock=map(secure,(output,run,bundle,lock))
    raw=read_pinned(output/'stage-receipt.json',receipt_sha);receipt=json.loads(raw)
    source=receipt.get('source',{});delta=check_source_delta(current,source)
    if (device not in ('cpu','cuda') or receipt.get('schema')!='gpu-sgns-subset-stage-v1'
            or receipt.get('status')!='verified' or receipt.get('verified') is not True
            or receipt.get('engineering') is not False or receipt.get('authority')!='ad1c428'
            or receipt.get('stage')!='sgns' or receipt.get('device')!=device
            or receipt.get('runtime')!=env or receipt.get('input_manifest_sha256')!=bundle_sha
            or receipt.get('subset_four_gate_acceptance') is not False
            or receipt.get('validation_read') is not False or receipt.get('test_read') is not False):
        raise ValueError('SGNS stage identity differs')
    run_bytes={name:secure(run/name).read_bytes() for name in ('run.json','terminal.json','logs/stdout.log')}
    job,terminal=json.loads(run_bytes['run.json']),json.loads(run_bytes['terminal.json'])
    rid='sgns-gpu-subset-'+device+'-fit-v1'
    expected=[sys.executable,'-B','-m','analysis.sgns_gpu_subset','fit','--bundle',str(bundle),
              '--bundle-sha',bundle_sha,'--device',device,'--lock',str(lock),'--output',str(output)]
    stdout=json.loads(next((line for line in reversed(run_bytes['logs/stdout.log'].splitlines()) if line.strip()),b'{}'))
    if (run.name!=rid or receipt.get('run_id')!=rid or job.get('run_id')!=rid
            or job.get('kind')!='ops' or job.get('git',{}).get('commit')!=source.get('commit')
            or job.get('argv')!=expected or terminal.get('run_id')!=rid
            or terminal.get('status')!='succeeded' or terminal.get('exit_code')!=0 or stdout!=receipt):
        raise ValueError('SGNS original stdout/job/terminal binding differs')
    inputs,producer=read_bundle(bundle,bundle_sha,source)
    if receipt.get('input_producer')!=producer:raise ValueError('original input producer differs')
    artifacts=receipt.get('artifacts',{})
    if set(artifacts)!={'vectors.npy','audit.json'}:raise ValueError('SGNS artifact set differs')
    snapshots={name:read_pinned(output/name,sha) for name,sha in artifacts.items()}
    vectors=np.load(io.BytesIO(snapshots['vectors.npy']),allow_pickle=False);audit=json.loads(snapshots['audit.json'])
    proof=verify_original_fit(inputs,vectors,audit)
    if proof!=receipt.get('original_audit_verification'):raise ValueError('original SGNS verification differs')
    ids=np.asarray(inputs[0],dtype=np.uint32)
    im={'schema':'coll-input-v1','status':'accepted','variant':'C1','dimension':64,'engineering':False,
        'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(vectors),
        'content_feature_sha256':audit['content_feature_sha256'],'train_sequence_sha256':audit['train_sequence_sha256'],
        'script_sha256':source['files']['analysis/sgns_gpu_subset.py'],'commit':source['commit'],
        'checkpoint_sha256':None,'source_files':source['files'],'sgns_stage_receipt_sha256':receipt_sha,
        'sgns_device':device,'sgns_original_job':{name:hashlib.sha256(b).hexdigest() for name,b in run_bytes.items()},
        'input_manifest_file_sha256':bundle_sha}
    # Formal production call preserves the original H5a/H5b/input checks.
    validate_input(ids,vectors,im,engineering=False)
    if any(secure(output/name).read_bytes()!=b for name,b in snapshots.items()):raise ValueError('SGNS artifact changed during verification')
    if any(secure(run/name).read_bytes()!=b for name,b in run_bytes.items()):raise ValueError('SGNS job changed during verification')
    return {'ids':ids,'vectors':vectors,'input_manifest':im,'sgns_receipt':receipt,
            'source_delta':delta,'job_proof':im['sgns_original_job']}


BUNDLE_ARTIFACTS=('mapping.npz','tokenizer.json','audit.json','input-manifest.json')


def load_bundle(manifest_path,manifest_sha256,*,catalog_ids,vectors,input_manifest,engineering=False):
    """Explicit new producer schema; original codec/gates, no legacy whitelist change."""
    from analysis.ur001_derive_cli import secure
    if engineering is not False:raise ValueError('formal SID bundle required')
    path=secure(manifest_path);raw=read_pinned(path,manifest_sha256);m=json.loads(raw)
    fixed={'schema':'gpu-sgns-subset-sid-v1','status':'verified','authority':'ad1c428',
           'engineering':False,'branching':48,'depth':3,'config':CONFIG,
           'input_dimension':64,'content_feature_sha256':input_manifest['content_feature_sha256'],
           'input_manifest_sha256':hashlib.sha256(canonical_bytes(input_manifest)).hexdigest(),
           'catalog_sha256':sid_v2.catalog_digest(catalog_ids),'input_feature_sha256':au.feature_digest(vectors),
           'mapping_replay':True,'subset_four_gate_acceptance':False}
    if path.name!='run_manifest.json' or any(m.get(k)!=v for k,v in fixed.items()):raise ValueError('new SID bundle identity differs')
    source=m.get('source',{})
    check_source_delta(source,{'commit':input_manifest['commit'],'files':input_manifest['source_files']})
    if any(source.get('files',{}).get(p)!=sha for p,sha in SCIENTIFIC_FILES.items()):raise ValueError('SID scientific sources differ')
    root=Path(__file__).resolve().parents[2]
    for rel,sha in source.get('files',{}).items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('source relative member required')
        if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=_hex(sha):raise ValueError('SID producer source bytes differ')
    if set(m.get('artifacts',{}))!=set(BUNDLE_ARTIFACTS):raise ValueError('SID artifact set differs')
    snapshots={name:read_pinned(path.parent/name,m['artifacts'][name]) for name in BUNDLE_ARTIFACTS}
    if json.loads(snapshots['input-manifest.json'])!=input_manifest or json.loads(snapshots['tokenizer.json'])!=CONFIG:
        raise ValueError('SID input/config artifact differs')
    with np.load(io.BytesIO(snapshots['mapping.npz']),allow_pickle=False) as arrays:
        if set(arrays.files)!={'ids','sids'}:raise ValueError('mapping member set differs')
        ids,sids=arrays['ids'],arrays['sids']
    if ids.dtype!=np.uint32 or sids.dtype!=np.uint32 or not np.array_equal(ids,catalog_ids):raise ValueError('mapping catalog differs')
    audit,gates=audit_gates(ids,sids,vectors,input_manifest,engineering=False)
    if not gates['passed'] or audit!=json.loads(snapshots['audit.json']) or audit!=m.get('audit') or gates!=m.get('gates'):
        raise ValueError('original SID gates/audit differ')
    codec=sid_v2.build_codec(ids,sids,48,catalog_ids=catalog_ids)
    info={'sid_version':'GPU_SGNS_SUBSET_B48D3_S42','manifest_sha256':manifest_sha256,'branching':48,
          'kind':'bhkm','engineering':False,'suffix_vocabulary':codec.catalog.suffix_vocabulary,'vocab_size':codec.vocab_size}
    if (m.get('suffix_vocabulary'),m.get('vocab_size'))!=(info['suffix_vocabulary'],info['vocab_size']):raise ValueError('SID vocabulary differs')
    if any(secure(path.parent/name).read_bytes()!=b for name,b in snapshots.items()) or secure(path).read_bytes()!=raw:
        raise ValueError('SID bundle changed during load')
    return sid_v2.LoadedSID(codec,m,info)
