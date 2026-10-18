"""Pinned S2 input family and original train-only fallback reconstruction.

Only authorized CONTENT_V1/C1-prime and pinned train inputs are read.
The fixed S0 plan supplies catalog/train audit/raw root; unrelated refs are ignored.
"""
import sys
from pathlib import Path
import numpy as np
from . import sid_learn_v1 as contract,sid_learn_pca as pca,sid_learn_bundle as bundle,sid_coll_sgns as sg,generative_data as gd
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import sid_coll_job_entry as legacy
S0_SHA='3b3479c5ca62a06b9c780104bd984349f3dde18b0920f8d548330302d8b9d779'
FORMAL_PINS={**bundle.FORMAL_PHYSICAL,'s0_plan':S0_SHA}
TRAIN_NAMES=tuple(f'subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet' for week in range(25))

def checked_refs(refs,engineering):
    if type(engineering) is not bool or type(refs) is not dict or set(refs)!=set(FORMAL_PINS):raise ValueError('exact S2 source roles required')
    for role,ref in refs.items():
        if type(ref) is not dict or set(ref)!={'path','sha256'} or type(ref['path']) is not str:raise ValueError('exact input path/sha required')
        path=Path(ref['path'])
        if not path.is_absolute() or '..' in path.parts or 'test' in path.parts or path.suffix!=('.npy' if role in ('content','c1_vectors') else '.json'):raise ValueError('absolute allowed input type required; no test/checkpoint')
        bundle._hex(ref['sha256'])
    if not engineering and any(refs[k]['sha256']!=v for k,v in FORMAL_PINS.items()):raise ValueError('frozen S2 physical pins required before IO')

def load_inputs(refs,*,engineering=False):
    checked_refs(refs,engineering)
    def path(role):return gd._checked_file(refs[role]['path'],refs[role]['sha256'])
    ids_raw=gd._json(path('catalog'))
    if type(ids_raw) is not list or any(type(v) is not int or not 0<=v<2**32 for v in ids_raw):raise ValueError('uint32 catalog JSON required')
    ids=np.asarray(ids_raw,dtype=np.uint32)
    values=np.load(path('content'),allow_pickle=False)
    if values.dtype!=np.float32 or values.shape!=(len(ids)+1,64) or not np.isfinite(values).all() or np.any(values[0]):raise ValueError('finite CONTENT_V1 with zero PAD required')
    content=np.ascontiguousarray(values[1:]);c1=np.load(path('c1_vectors'),allow_pickle=False);im=gd._json(path('c1_manifest'))
    physical={role:refs[role]['sha256'] for role in bundle.FORMAL_PHYSICAL}
    family=bundle._family(ids,content,c1,im,physical,engineering)
    plan=gd._json(path('s0_plan'))
    if plan.get('catalog')!=refs['catalog']:raise ValueError('catalog differs from original C1 source plan')
    sources=im.get('input_sources');wanted=set(TRAIN_NAMES)|{'catalog','content','s0_plan','train_audit'}
    if type(sources) is not dict or set(sources)!=wanted:raise ValueError('only original 25 train sources allowed')
    for value in sources.values():bundle._hex(value)
    audit_ref=plan['train_audit']
    if audit_ref['sha256']!=sources['train_audit']:raise ValueError('train audit binding differs')
    audit=gd._json(legacy.checked_ref(audit_ref))
    histories,actual=legacy.load_histories(plan,audit,ids)
    complete={**actual,'catalog':physical['catalog'],'content':physical['content'],'s0_plan':refs['s0_plan']['sha256'],'train_audit':audit_ref['sha256']}
    if complete!=sources:raise ValueError('original train physical source ledger differs')
    sequence=sg._sequence_digest(histories)
    if sequence!=im['train_sequence_sha256']:raise ValueError('original C1 ordered history sequence differs')
    fallback=contract.recompute_fallback_mask(ids,histories,window=5)
    if not engineering:contract.check_fallback_gate(fallback)
    pca_state=pca.fit_pca(c1);target=pca.project_pca(c1,pca_state)
    return {'ids':ids,'content':content,'c1_vectors':c1,'c1_manifest':im,'physical':physical,'family':family,'fallback':fallback,'pca_state':pca_state,'target':target,'engineering':engineering,
            'history_proof':{'sequence_sha256':sequence,'physical_sources':complete,'fallback_mask_sha256':sg._array_digest(fallback)}}

def summary(data):
    return {'status':'preflight_ok','written':False,'engineering':data['engineering'],'catalog_items':len(data['ids']),'content_dimension':64,'target_dimension':32,'physical_inputs':data['physical'],'input_family_sha256':contract.family_digest(data['family']),
            'fallback_count':int(data['fallback'].sum()),'fallback_mask_sha256':data['history_proof']['fallback_mask_sha256'],'history_sequence_sha256':data['history_proof']['sequence_sha256'],
            'pca_components_sha256':data['pca_state']['components_sha256'],'pca_explained_variance_ratio':float(data['pca_state']['explained_variance_ratio'].sum()),'test_read':False}
