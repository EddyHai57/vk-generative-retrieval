"""S1-only collaborative SID bundle; never extends a frozen legacy whitelist."""
from pathlib import Path
import hashlib
import os
import numpy as np
from . import generative_data as gd, sid_tok_audit as au, sid_v2
from .data_audit import canonical_bytes, file_sha256

SCHEMA='sidmap-structure-v1'
PROTOCOL='generative-like-sid-structure-v1'
TASK_ID='SID-STRUCTURE-MATCHED'
MANIFEST_NAME='run_manifest.json'
SID_VERSIONS=('STRUCT_C1_B48D3_S42',)
CONTENT_V1_FEATURE_SHA256=au.CONTENT_V1_FEATURE_SHA256
VARIANTS={'C1':(64,SID_VERSIONS[0])}
ARTIFACTS=('mapping.npz','tokenizer.json','audit.json','input-manifest.json')
INPUT_REQUIRED={'schema','status','variant','dimension','engineering','catalog_sha256','vectors_sha256',
 'content_feature_sha256','train_sequence_sha256','script_sha256','commit','checkpoint_sha256','source_files'}
MANIFEST_KEYS={'schema','protocol','task_id','status','sid_version','kind','engineering','branching','depth','input_dimension',
 'catalog_items','suffix_vocabulary','vocab_size','content_feature_sha256','input_feature_sha256','catalog_sha256',
 'input_manifest_sha256','artifacts','fit_config_sha256','producer','source_commit','source_dirty','gates','producer_runtime'}


def _hex(value,n=64):
    if type(value) is not str or len(value)!=n or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid sha identity')
    return value


FIT_CONFIG_MEMBER='configs/sid_structure/fit-b48d3-s42.json'


def producer_identity():
    from . import sid_structure_run as fit
    root=Path(__file__).resolve().parents[2]
    members=(*fit.PRODUCER_FILES,FIT_CONFIG_MEMBER)
    if len(set(members))!=len(members):raise ValueError('duplicate producer source member')
    files={name:file_sha256(root/name) for name in members}
    return {'files':files,'runtime':{**gd._environment(),'lock_sha256':files['uv.lock']}}


def check_producer_identity(manifest):
    current=producer_identity()
    if manifest.get('producer')!=current['files']:
        raise ValueError('producer fixed source ledger differs')
    if manifest.get('producer_runtime')!=current['runtime']:
        raise ValueError('producer Python/dependencies/lock runtime differs')
    if manifest.get('fit_config_sha256')!=current['files'][FIT_CONFIG_MEMBER]:
        raise ValueError('producer fit config bytes differ')
    return current


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


def input_manifest_digest(manifest): return hashlib.sha256(canonical_bytes(manifest)).hexdigest()


def write_bundle(output,*,ids,sids,vectors,input_manifest,config,config_sha,engineering,producer):
    from .sid_structure_run import validate_config
    validate_config(config)
    if config['variant']!=input_manifest['variant']: raise ValueError('fit config variant mismatch')
    identity=check_producer_identity({'producer':producer,'producer_runtime':producer_identity()['runtime'],'fit_config_sha256':config_sha})
    audit,gates=audit_gates(ids,sids,vectors,input_manifest,engineering=engineering)
    if not gates['passed']: raise ValueError('SID gates failed')
    codec=sid_v2.build_codec(ids,sids,48)
    _hex(config_sha)
    if not producer: raise ValueError('producer sources required')
    for name,sha in producer.items():
        if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('relative producer path required')
        _hex(sha)
    git=gd._git(Path(__file__).resolve().parents[2])
    if not engineering and git['git_dirty']: raise ValueError('formal SID fit requires clean committed public source')
    output=Path(output).absolute();gd._no_symlink(output)
    if output.exists() or not output.parent.is_dir(): raise ValueError('new bundle directory required')
    output.mkdir()
    sid_v2._write_npz(output/'mapping.npz',{'ids':ids,'sids':sids})
    for name,obj in [('tokenizer.json',config),('audit.json',audit),('input-manifest.json',input_manifest)]:
        (output/name).write_bytes(canonical_bytes(obj))
    manifest={'schema':SCHEMA,'protocol':PROTOCOL,'task_id':TASK_ID,'status':'complete',
      'sid_version':VARIANTS[input_manifest['variant']][1],'kind':'bhkm','engineering':engineering,'branching':48,'depth':3,
      'input_dimension':vectors.shape[1],'catalog_items':len(ids),'suffix_vocabulary':codec.catalog.suffix_vocabulary,'vocab_size':codec.vocab_size,
      'content_feature_sha256':input_manifest['content_feature_sha256'],'input_feature_sha256':input_manifest['vectors_sha256'],
      'catalog_sha256':sid_v2.catalog_digest(ids),'input_manifest_sha256':input_manifest_digest(input_manifest),
      'artifacts':{name:file_sha256(output/name) for name in ARTIFACTS},'fit_config_sha256':config_sha,'producer':dict(producer),'producer_runtime':identity['runtime'],'source_commit':git['git_base'],'source_dirty':git['git_dirty'],'gates':gates}
    (output/(MANIFEST_NAME+'.tmp')).write_bytes(canonical_bytes(manifest));os.replace(output/(MANIFEST_NAME+'.tmp'),output/MANIFEST_NAME)
    return manifest


def load_bundle(manifest_path,manifest_sha256,*,catalog_ids,vectors,input_manifest,engineering=False):
    validate_input(catalog_ids,vectors,input_manifest,engineering=engineering)
    path=gd._checked_file(manifest_path,manifest_sha256);m=gd._json(path)
    if path.name!=MANIFEST_NAME or type(m) is not dict or set(m)!=MANIFEST_KEYS: raise ValueError('coll bundle manifest schema')
    fixed={'schema':SCHEMA,'protocol':PROTOCOL,'task_id':TASK_ID,'status':'complete','kind':'bhkm','branching':48,'depth':3,
      'sid_version':VARIANTS[input_manifest['variant']][1],'input_dimension':vectors.shape[1], 'catalog_items':len(catalog_ids),
      'content_feature_sha256':input_manifest['content_feature_sha256'],'input_feature_sha256':input_manifest['vectors_sha256'],
      'catalog_sha256':sid_v2.catalog_digest(catalog_ids),'input_manifest_sha256':input_manifest_digest(input_manifest)}
    if m['engineering'] is not engineering or any(m[k]!=v for k,v in fixed.items()): raise ValueError('coll bundle identity mismatch')
    if any(type(m[k]) is not int for k in ('branching','depth','input_dimension','catalog_items','suffix_vocabulary','vocab_size')): raise ValueError('typed bundle sizes required')
    _hex(m['source_commit'],40)
    if type(m['source_dirty']) is not bool or not engineering and m['source_dirty']: raise ValueError('uncommitted formal producer source')
    _hex(m['fit_config_sha256'])
    if type(m['producer']) is not dict or not m['producer']: raise ValueError('producer identity missing')
    for name,sha in m['producer'].items():
        if type(name) is not str or Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('relative producer source required')
        _hex(sha)
    check_producer_identity(m)
    directory=path.parent
    if set(m['artifacts'])!=set(ARTIFACTS) or set(os.listdir(directory))!=set(ARTIFACTS)|{MANIFEST_NAME}: raise ValueError('bundle artifact set')
    for name in ARTIFACTS: gd._checked_file(directory/name,m['artifacts'][name])
    if canonical_bytes(gd._json(directory/'input-manifest.json'))!=canonical_bytes(input_manifest): raise ValueError('input producer identity differs')
    mapping=sid_v2._load_npz(directory/'mapping.npz',{'ids','sids'});ids,sids=mapping['ids'],mapping['sids']
    if not np.array_equal(ids,catalog_ids): raise ValueError('mapping catalog mismatch')
    codec=sid_v2.build_codec(ids,sids,48,catalog_ids=catalog_ids)
    if (codec.vocab_size,codec.catalog.suffix_vocabulary)!=(m['vocab_size'],m['suffix_vocabulary']): raise ValueError('codec vocabulary mismatch')
    audit,gates=audit_gates(ids,sids,vectors,input_manifest,engineering=engineering)
    if not gates['passed'] or not au.audits_equal(audit,gd._json(directory/'audit.json')) or not au.audits_equal(gates,m['gates']): raise ValueError('recomputed coll gates/audit mismatch')
    from .sid_structure_run import validate_config
    config=gd._json(directory/'tokenizer.json');validate_config(config)
    if config['variant']!=input_manifest['variant']: raise ValueError('tokenizer variant mismatch')
    info={'sid_version':m['sid_version'],'manifest_sha256':manifest_sha256,'branching':48,'kind':'bhkm','engineering':engineering,
      'suffix_vocabulary':codec.catalog.suffix_vocabulary,'vocab_size':codec.vocab_size}
    return sid_v2.LoadedSID(codec,m,info)
