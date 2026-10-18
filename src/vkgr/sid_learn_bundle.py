"""Isolated S2 SID bundle: exact provenance, canonical RQ export and original gates."""
import hashlib,os,math
from pathlib import Path
import numpy as np
from . import sid_learn_v1 as contract,sid_learn_pca as pca,sid_learn_fit as fit,sid_rqvae as rq,sid_v2,sid_tok_audit as au,sid_coll_sgns as sg,generative_data as gd
from .data_audit import canonical_bytes,file_sha256
CONTENT_V1_FEATURE_SHA256=contract.CONTENT_V1_FEATURE_SHA256
SID_VERSION=contract.SID_VERSION
SCHEMA=contract.SCHEMA
PROTOCOL=contract.PROTOCOL
MANIFEST='run_manifest.json'
CONFIG_MEMBER='configs/sid_learn/fit-s42.json'
PRODUCER_FILES=tuple('src/vkgr/'+n+'.py' for n in ('sid_learn_v1','sid_learn_pca','sid_learn_loss','sid_learn_fit','sid_learn_bundle','sid_rqvae','sid_rqvae_r','sid_v2','sid_tok_audit','sid','sid_coll_sgns','generative_data','data_audit','generative_b1_run','generative_pilot'))+('src/vkgr/sid_learn_run.py','src/vkgr/sid_learn_input.py','scripts/sid_coll_job_entry.py','scripts/coll_sid_a1_ops.py','analysis/sid_diag.py','analysis/sid_diag_metrics.py','src/vkgr/sid_coll_v1.py','uv.lock',CONFIG_MEMBER)
FORMAL_PHYSICAL={'content':'644b2d7a656999a8962355d7c32dae07134ad46e9557b4a8418f137d62f73581','catalog':'80201c574fad26bd29f2e2425a5ad98db8fe7c2a9be91e07d2fc6aedc48d47f3','c1_vectors':contract.C1_VECTORS_FILE_SHA256,'c1_manifest':contract.C1_INPUT_MANIFEST_SHA256}
PCA_ARRAYS=('mean','components','eigenvalues','explained_variance_ratio')
ARTIFACTS=('mapping.npz','weights.npz','pca.npz','tokenizer.json','input-manifest.json','audit.json','train-log.json')
KEYS={'schema','protocol','task_id','status','sid_version','kind','engineering','branching','depth','catalog_items','suffix_vocabulary','vocab_size','content_feature_sha256','catalog_sha256','input_family','input_family_sha256','physical_inputs','fallback_count','fallback_mask_sha256','pca','fit_config_sha256','producer','producer_runtime','source_commit','source_dirty','artifacts','gates'}

class FitGateFailure(ValueError):
    def __init__(self,audit,gates):
        super().__init__('FIT_FAILED: original S2 hard gates')
        self.audit=audit;self.gates=gates

def _hex(value):
    if type(value) is not str or len(value)!=64 or set(value)-set('0123456789abcdef'):raise ValueError('sha256 required')
    return value

def validate_physical(physical,engineering):
    if type(engineering) is not bool or type(physical) is not dict or set(physical)!=set(FORMAL_PHYSICAL):raise ValueError('exact physical input ledger required')
    for value in physical.values():_hex(value)
    if not engineering and physical!=FORMAL_PHYSICAL:raise ValueError('frozen S2 physical inputs differ')

def producer_identity():
    fit.check_frozen_l2();root=Path(__file__).resolve().parents[2]
    if len(set(PRODUCER_FILES))!=len(PRODUCER_FILES):raise ValueError('duplicate producer member')
    files={name:file_sha256(root/name) for name in PRODUCER_FILES}
    contract.validate_config(gd._json(root/CONFIG_MEMBER))
    return {'files':files,'runtime':{**gd._environment(),'lock_sha256':files['uv.lock']}}

def check_producer(manifest):
    actual=producer_identity()
    if manifest.get('producer')!=actual['files'] or manifest.get('producer_runtime')!=actual['runtime'] or manifest.get('fit_config_sha256')!=actual['files'][CONFIG_MEMBER]:raise ValueError('fixed producer/config/Python/dependencies/lock differ')
    return actual

def _family(ids,content,c1_vectors,c1_manifest,physical,engineering):
    validate_physical(physical,engineering)
    return contract.input_family_identity(ids,content,c1_vectors,c1_manifest,engineering=engineering,c1_vectors_file_sha256=physical['c1_vectors'],c1_input_manifest_sha256=physical['c1_manifest'])

def _pca_metadata(state):return {key:value for key,value in state.items() if key not in PCA_ARRAYS}

def _check_pca(vectors,state):
    pca.project_pca(vectors,state)
    actual=pca.fit_pca(vectors)
    if _pca_metadata(state)!=_pca_metadata(actual) or any(not np.array_equal(state[k],actual[k]) for k in PCA_ARRAYS):raise ValueError('PCA does not replay from fixed C1 source')
    return actual

def _fallback(mask,count,engineering):
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=(count,):raise ValueError('recomputed bool fallback set required')
    if not engineering:contract.check_fallback_gate(mask)
    return {'fallback_count':int(mask.sum()),'fallback_mask_sha256':sg._array_digest(mask)}

def _rq_info(report,state,content,config,engineering):
    if type(report) is not dict or report.get('status')!='complete' or report.get('engineering') is not engineering or report.get('K')!=state['K'] or report.get('seed')!=42 or report.get('wall_cap_seconds')!=config['wall_cap_seconds']:raise ValueError('complete bound fit report required')
    if any(type(report.get(k)) is not int for k in ('step','max_steps','K','seed','batch_size')) or report['step']!=report['max_steps'] or report['step']<1:raise ValueError('fit step schedule differs')
    if not engineering and (state['K']!=256 or report['step']!=300000 or report['batch_size']!=1024):raise ValueError('formal frozen fit schedule differs')
    if type(report.get('elapsed')) not in (int,float) or not math.isfinite(report['elapsed']) or report['elapsed']<0 or type(report.get('wall_cap_seconds')) is not int:raise ValueError('numeric fit timing types required')
    if report.get('final_state_digest')!=rq.state_digest(state['state_dict']):raise ValueError('fit state byte digest differs')
    return {'status':'complete','steps':report['step'],'max_steps':report['max_steps'],'elapsed':report['elapsed'],'wall_cap_seconds':report['wall_cap_seconds'],'required_steps':report['max_steps'] if engineering else au.REQUIRED_STEPS,'reconstruction':rq.reconstruction_report(state,content)}

def _audit(ids,sids,content,info,family):
    mapping=au.audit_mapping(ids,sids,content,branching=info['branching'],kind='rqvae')
    inherited=au.gates(mapping,'rqvae',info['rq'],expected_content_sha256=family['content_feature_sha256'],required_steps=info['rq']['required_steps'])
    checks=[c for c in inherited['checks'] if c['id']!='H5']+contract.h5_checks(family,contract.family_digest(family),engineering=info['engineering'])
    return {'mapping':mapping,'rq':info['rq']},{'passed':all(c['ok'] for c in checks),'checks':checks}

def write_bundle(output,*,ids,content,c1_vectors,c1_manifest,physical,pca_state,fallback,fit,config,engineering):
    contract.validate_config(config);family=_family(ids,content,c1_vectors,c1_manifest,physical,engineering)
    mask=_fallback(fallback,len(ids),engineering);_check_pca(c1_vectors,pca_state)
    state=fit.get('state')
    if type(state) is not dict:raise ValueError('completed own state required')
    arrays=sid_v2._weights_arrays(state,state['K']);rqinfo=_rq_info(fit,state,content,config,engineering)
    codes=rq.export_codes(state,content);sids=rq.assign_fourth_code(ids,codes);codec=sid_v2.build_codec(ids,sids,state['K'])
    audit,gates=_audit(ids,sids,content,{'branching':state['K'],'rq':rqinfo,'engineering':engineering},family)
    if not gates['passed']:raise FitGateFailure(audit,gates)
    identity=producer_identity();root=Path(__file__).resolve().parents[2];git=gd._git(root)
    if not engineering and git['git_dirty']:raise ValueError('formal fit requires clean committed source')
    output=gd._no_symlink(Path(output).absolute())
    if output.exists() or not output.parent.is_dir():raise ValueError('exclusive new bundle directory required')
    output.mkdir();sid_v2._write_npz(output/'mapping.npz',{'ids':ids,'sids':sids});sid_v2._write_npz(output/'weights.npz',arrays);sid_v2._write_npz(output/'pca.npz',{k:pca_state[k] for k in PCA_ARRAYS})
    train={key:value for key,value in fit.items() if key!='state'}
    for name,value in [('tokenizer.json',config),('input-manifest.json',c1_manifest),('audit.json',audit),('train-log.json',train)]:
        (output/name).write_bytes(canonical_bytes(value))
    version=contract.SID_VERSION if not engineering else f'ENGINEERING_LEARN_K{state["K"]}D3_S42'
    manifest={'schema':SCHEMA,'protocol':PROTOCOL,'task_id':contract.TASK_ID,'status':'complete','sid_version':version,'kind':'rqvae','engineering':engineering,'branching':state['K'],'depth':3,'catalog_items':len(ids),'suffix_vocabulary':codec.catalog.suffix_vocabulary,'vocab_size':codec.vocab_size,'content_feature_sha256':family['content_feature_sha256'],'catalog_sha256':family['catalog_sha256'],'input_family':family,'input_family_sha256':contract.family_digest(family),'physical_inputs':physical,**mask,'pca':_pca_metadata(pca_state),'fit_config_sha256':identity['files'][CONFIG_MEMBER],'producer':identity['files'],'producer_runtime':identity['runtime'],'source_commit':git['git_base'],'source_dirty':git['git_dirty'],'artifacts':{name:file_sha256(output/name) for name in ARTIFACTS},'gates':gates}
    (output/(MANIFEST+'.tmp')).write_bytes(canonical_bytes(manifest));os.replace(output/(MANIFEST+'.tmp'),output/MANIFEST)
    return manifest

def load_bundle(manifest_path,manifest_sha256,*,catalog_ids,content,c1_vectors,c1_manifest,physical,engineering=False,fallback=None):
    path=gd._checked_file(manifest_path,manifest_sha256);m=gd._json(path)
    if path.name!=MANIFEST or type(m) is not dict or set(m)!=KEYS or m['engineering'] is not engineering:raise ValueError('isolated learnable bundle schema/engineering differs')
    if any(type(m[k]) is not int for k in ('branching','depth','catalog_items','suffix_vocabulary','vocab_size','fallback_count')) or not 0<=m['fallback_count']<=m['catalog_items']:raise ValueError('plain integer bundle counts required')
    family=_family(catalog_ids,content,c1_vectors,c1_manifest,physical,engineering);check_producer(m)
    fixed={'schema':SCHEMA,'protocol':PROTOCOL,'task_id':contract.TASK_ID,'status':'complete','kind':'rqvae','depth':3,'catalog_items':len(catalog_ids),'content_feature_sha256':family['content_feature_sha256'],'catalog_sha256':family['catalog_sha256'],'input_family':family,'input_family_sha256':contract.family_digest(family),'physical_inputs':physical}
    if any(m[k]!=v for k,v in fixed.items()) or type(m['source_dirty']) is not bool or not engineering and m['source_dirty']:raise ValueError('bundle input/family/source differs')
    k=m['branching'];version=contract.SID_VERSION if not engineering else f'ENGINEERING_LEARN_K{k}D3_S42'
    if type(k) is not int or k<2 or not engineering and k!=256 or m['sid_version']!=version:raise ValueError('isolated version/branching differs')
    if type(m['source_commit']) is not str or len(m['source_commit'])!=40 or set(m['source_commit'])-set('0123456789abcdef'):raise ValueError('producer commit required separately')
    if not engineering and m['fallback_count']!=251:raise ValueError('formal fallback count differs')
    _hex(m['fallback_mask_sha256'])
    if fallback is not None and any(m[key]!=value for key,value in _fallback(fallback,len(catalog_ids),engineering).items()):raise ValueError('recomputed fallback set differs')
    directory=path.parent
    if set(m['artifacts'])!=set(ARTIFACTS) or set(os.listdir(directory))!=set(ARTIFACTS)|{MANIFEST}:raise ValueError('exact bundle members required')
    for name in ARTIFACTS:gd._checked_file(directory/name,m['artifacts'][name])
    if canonical_bytes(gd._json(directory/'input-manifest.json'))!=canonical_bytes(c1_manifest):raise ValueError('C1 input producer differs')
    config=gd._json(directory/'tokenizer.json');contract.validate_config(config)
    state=sid_v2._check_state_arrays(sid_v2._load_npz(directory/'weights.npz'),k)
    stored_pca={**m['pca'],**sid_v2._load_npz(directory/'pca.npz',PCA_ARRAYS)};_check_pca(c1_vectors,stored_pca)
    mapping=sid_v2._load_npz(directory/'mapping.npz',{'ids','sids'})
    if not np.array_equal(mapping['ids'],catalog_ids):raise ValueError('mapping catalog differs')
    sid_v2._check_weights(state,mapping['ids'],mapping['sids'],content)
    codec=sid_v2.build_codec(mapping['ids'],mapping['sids'],k,catalog_ids=catalog_ids)
    if (codec.catalog.suffix_vocabulary,codec.vocab_size)!=(m['suffix_vocabulary'],m['vocab_size']):raise ValueError('codec vocabulary differs')
    report=gd._json(directory/'train-log.json');rqinfo=_rq_info(report,state,content,config,engineering)
    audit,gates=_audit(mapping['ids'],mapping['sids'],content,{'branching':k,'rq':rqinfo,'engineering':engineering},family)
    if not gates['passed'] or not au.audits_equal(audit,gd._json(directory/'audit.json')) or not au.audits_equal(gates,m['gates']):raise ValueError('original hard audit/gates differ')
    return sid_v2.LoadedSID(codec,m,{'kind':'rqvae','sid_version':version,'manifest_sha256':manifest_sha256,'branching':k,'depth':3,'suffix_vocabulary':m['suffix_vocabulary'],'vocab_size':m['vocab_size'],'engineering':engineering})
