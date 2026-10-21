"""Frozen S1 BHKM fit and independent reload + full seed42 replay."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from . import sid, sid_coll_v1 as coll, generative_data as gd
from .data_audit import canonical_bytes,file_sha256

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


PRODUCER_FILES=('src/vkgr/sid_coll_run.py','src/vkgr/sid_coll_v1.py','src/vkgr/sid.py',
 'src/vkgr/sid_v2.py','src/vkgr/sid_tok_audit.py','src/vkgr/sid_rqvae.py','src/vkgr/generative_data.py','src/vkgr/data_audit.py')


def expected_config(variant='C1'):
    if variant not in coll.VARIANTS: raise ValueError('variant must be C1 or C2')
    dim,label=coll.VARIANTS[variant]
    return {'schema':'coll-fit-v1','task_id':coll.TASK_ID,'variant':variant,'sid_version':label,
      'input_dimension':dim,'branching':256,'depth':3,'seed':42,'max_iter':10}


def validate_config(value):
    if type(value) is not dict or value!=expected_config(value.get('variant')) or any(type(value[k]) is not int for k in ('input_dimension','branching','depth','seed','max_iter')):
        raise ValueError('frozen collaborative fit config mismatch')
    return value


def load_inputs(vectors_path,vectors_sha,catalog_path,catalog_sha,input_manifest_path,input_sha,*,engineering=False):
    vectors=np.load(gd._checked_file(vectors_path,vectors_sha),allow_pickle=False)
    raw=gd._json(gd._checked_file(catalog_path,catalog_sha))
    if type(raw) is not list or any(type(i) is not int or not 0<=i<2**32 for i in raw): raise ValueError('catalog uint32 JSON list required')
    ids=np.asarray(raw,dtype=np.uint32)
    manifest=gd._json(gd._checked_file(input_manifest_path,input_sha))
    coll.validate_input(ids,vectors,manifest,engineering=engineering)
    return ids,vectors,manifest


def fit_mapping(ids,vectors,input_manifest,config,*,engineering=False):
    validate_config(config);coll.validate_input(ids,vectors,input_manifest,engineering=engineering)
    if config['variant']!=input_manifest['variant']: raise ValueError('fit input variant mismatch')
    model=fit_bhkm_coll(ids,vectors,branching=256,depth=3,seed=42,max_iter=10)
    audit,gates=coll.audit_gates(model.ids,model.sids,vectors,input_manifest,engineering=engineering)
    if not gates['passed']: raise ValueError('FIT_FAILED: collaborative SID gates')
    return {'ids':model.ids,'sids':model.sids,'audit':audit,'gates':gates}


def verify_mapping(loaded,ids,vectors,input_manifest,config,config_sha,*,engineering=False):
    validate_config(config)
    if loaded.manifest['fit_config_sha256']!=config_sha or loaded.info['engineering'] is not engineering:
        raise ValueError('fit config or engineering binding mismatch')
    again=fit_mapping(ids,vectors,input_manifest,config,engineering=engineering)
    if not np.array_equal(loaded.codec.catalog.ids,again['ids']) or not np.array_equal(loaded.codec.catalog.sids,again['sids']):
        raise ValueError('same-seed full fit replay mismatch')
    return {'schema':'coll-fit-verify-v1','status':'verified','sid_version':loaded.info['sid_version'],
       'manifest_sha256':loaded.info['manifest_sha256'],'fit_config_sha256':config_sha,
       'input_manifest_sha256':coll.input_manifest_digest(input_manifest),'mapping_replay':True,
       'input_feature_sha256':input_manifest['vectors_sha256'],'catalog_sha256':input_manifest['catalog_sha256'],
       'checks':['H1','H2','H3','H4','H5a','H5b','H6','B1','same_seed_full_replay']}


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False);sub=p.add_subparsers(dest='command',required=True)
    for name in ('fit','verify'):
        q=sub.add_parser(name,allow_abbrev=False)
        for role in ('config','vectors','catalog','input-manifest'):
            q.add_argument('--'+role,required=True);q.add_argument('--'+role+'-sha',required=True)
        q.add_argument('--engineering',action='store_true');q.add_argument('--output',required=True)
        if name=='verify': q.add_argument('--bundle',required=True);q.add_argument('--sha',required=True)
    a=p.parse_args(argv)
    try:
        config=validate_config(gd._json(gd._checked_file(a.config,a.config_sha)))
        ids,x,im=load_inputs(a.vectors,a.vectors_sha,a.catalog,a.catalog_sha,a.input_manifest,a.input_manifest_sha,engineering=a.engineering)
        out=Path(a.output).absolute();gd._no_symlink(out)
        if out.exists() or not out.parent.is_dir(): raise ValueError('exclusive new output required')
        if a.command=='fit':
            result=fit_mapping(ids,x,im,config,engineering=a.engineering)
            root=Path(__file__).resolve().parents[2]
            producer={name:file_sha256(root/name) for name in PRODUCER_FILES}
            # Config bytes are consumed and bound alongside source members.
            producer['configs/sid_coll/'+Path(a.config).name]=a.config_sha
            coll.write_bundle(out,ids=ids,sids=result['sids'],vectors=x,input_manifest=im,config=config,
                 config_sha=a.config_sha,engineering=a.engineering,producer=producer)
            result={'status':'complete','manifest_sha256':file_sha256(out/coll.MANIFEST_NAME)}
        else:
            loaded=coll.load_bundle(a.bundle,a.sha,catalog_ids=ids,vectors=x,input_manifest=im,engineering=a.engineering)
            if gd._json(Path(a.bundle).parent/'tokenizer.json')!=config: raise ValueError('stored fit config differs')
            result=verify_mapping(loaded,ids,x,im,config,a.config_sha,engineering=a.engineering)
            out.mkdir();(out/'verification.json').write_bytes(canonical_bytes(result))
        print(json.dumps(result,sort_keys=True));return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        print('SID-COLL gate failed: '+str(exc),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
