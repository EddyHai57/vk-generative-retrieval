"""Frozen S1 BHKM fit and independent reload + full seed42 replay."""
import argparse
import os
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from . import sid, sid_structure_v1 as coll, generative_data as gd
from .data_audit import canonical_bytes,file_sha256

from . import sid_coll_run as original
from .sid_coll_run import fit_bhkm_coll

PRODUCER_FILES=original.PRODUCER_FILES+('src/vkgr/sid_structure_v1.py','src/vkgr/sid_structure_run.py','uv.lock')


def expected_config(variant='C1'):
    if variant not in coll.VARIANTS: raise ValueError('only frozen C1 input variant allowed')
    dim,label=coll.VARIANTS[variant]
    return {'schema':'structure-fit-v1','task_id':coll.TASK_ID,'variant':variant,'sid_version':label,
      'input_dimension':dim,'branching':48,'depth':3,'seed':42,'max_iter':10}


def validate_config(value):
    if type(value) is not dict or value!=expected_config(value.get('variant')) or any(type(value[k]) is not int for k in ('input_dimension','branching','depth','seed','max_iter')):
        raise ValueError('frozen collaborative fit config mismatch')
    return value


def load_inputs(vectors_path,vectors_sha,catalog_path,catalog_sha,input_manifest_path,input_sha,*,engineering=False):
    if type(engineering) is not bool:raise ValueError('explicit engineering boolean required')
    if not engineering and (vectors_sha,catalog_sha,input_sha)!=(
        'bf840eb0ecb350a4044bbe20a45e4ed82d6402f54d364aad8f17f5f554d582c7',
        '80201c574fad26bd29f2e2425a5ad98db8fe7c2a9be91e07d2fc6aedc48d47f3',
        '725fb211a342c518770623be81d490feaaec38a285c954b5b6f51f50a388e9a4'):
        raise ValueError('frozen C1 physical input pins required before loading')
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
    model=fit_bhkm_coll(ids,vectors,branching=48,depth=3,seed=42,max_iter=10)
    audit,gates=coll.audit_gates(model.ids,model.sids,vectors,input_manifest,engineering=engineering)
    if not gates['passed']: raise ValueError('FIT_FAILED: collaborative SID gates')
    return {'ids':model.ids,'sids':model.sids,'audit':audit,'gates':gates}


def verify_mapping(loaded,ids,vectors,input_manifest,config,config_sha,*,engineering=False):
    validate_config(config)
    identity=coll.check_producer_identity(loaded.manifest)
    if loaded.manifest['fit_config_sha256']!=config_sha or loaded.info['engineering'] is not engineering:
        raise ValueError('fit config or engineering binding mismatch')
    again=fit_mapping(ids,vectors,input_manifest,config,engineering=engineering)
    if not np.array_equal(loaded.codec.catalog.ids,again['ids']) or not np.array_equal(loaded.codec.catalog.sids,again['sids']):
        raise ValueError('same-seed full fit replay mismatch')
    return {'schema':'structure-fit-verify-v1','status':'verified','sid_version':loaded.info['sid_version'],
       'manifest_sha256':loaded.info['manifest_sha256'],'fit_config_sha256':config_sha,
       'input_manifest_sha256':coll.input_manifest_digest(input_manifest),'mapping_replay':True,
       'input_feature_sha256':input_manifest['vectors_sha256'],'catalog_sha256':input_manifest['catalog_sha256'],
       'checks':['H1','H2','H3','H4','H5a','H5b','H6','R1','B1','same_seed_full_replay'],
       'producer':identity['files'],'producer_runtime':identity['runtime'],
       'producer_commit':loaded.manifest['source_commit'],'verifier_commit':gd._git(Path(__file__).resolve().parents[2])['git_base']}


def verification_bindings(result,*,output,config_sha,input_sha,vectors_sha,catalog_sha,engineering):
    rid=os.environ.get('VKGR_RUN_ID')
    meta=Path(output).parent/'run.json'
    if not engineering:
        if rid!='sid-structure-fit-verify-v1' or not meta.is_file() or meta.is_symlink():
            raise ValueError('formal fit-verify own job identity required')
        job=json.loads(meta.read_text())
        if job.get('run_id')!=rid or job.get('config',{}).get('sha256')!=config_sha:
            raise ValueError('fit verify job/config identity mismatch')
    return {**result,'run_id':rid,'run_manifest_sha256':hashlib.sha256(meta.read_bytes()).hexdigest() if meta.is_file() else None,
      'fit_config_file_sha256':config_sha,'input_manifest_file_sha256':input_sha,'vectors_file_sha256':vectors_sha,'catalog_file_sha256':catalog_sha}


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False);sub=p.add_subparsers(dest='command',required=True)
    for name in ('preflight','fit','verify'):
        q=sub.add_parser(name,allow_abbrev=False)
        for role in ('config','vectors','catalog','input-manifest'):
            q.add_argument('--'+role,required=True);q.add_argument('--'+role+'-sha',required=True)
        q.add_argument('--engineering',action='store_true');q.add_argument('--output',required=True)
        if name=='verify': q.add_argument('--bundle',required=True);q.add_argument('--sha',required=True)
    a=p.parse_args(argv)
    try:
        root=Path(__file__).resolve().parents[2]
        expected_config_path=root/coll.FIT_CONFIG_MEMBER
        if not a.engineering and (Path(a.config).resolve()!=expected_config_path or a.config_sha!=file_sha256(expected_config_path)):
            raise ValueError('formal fit config must be the fixed committed file')
        config=validate_config(gd._json(gd._checked_file(a.config,a.config_sha)))
        ids,x,im=load_inputs(a.vectors,a.vectors_sha,a.catalog,a.catalog_sha,a.input_manifest,a.input_manifest_sha,engineering=a.engineering)
        out=Path(a.output).absolute();gd._no_symlink(out)
        if out.exists() or not out.parent.is_dir(): raise ValueError('exclusive new output required')
        if a.command=='preflight':
            print(json.dumps({'status':'preflight_ok','written':False,'engineering':a.engineering,'catalog_items':len(ids),'input_dimension':x.shape[1],'input_manifest_sha256':a.input_manifest_sha,'vectors_sha256':a.vectors_sha,'catalog_sha256':a.catalog_sha},sort_keys=True));return 0
        if a.command=='fit':
            result=fit_mapping(ids,x,im,config,engineering=a.engineering)
            root=Path(__file__).resolve().parents[2]
            producer={name:file_sha256(root/name) for name in PRODUCER_FILES}
            # Config bytes are consumed and bound alongside source members.
            producer['configs/sid_structure/'+Path(a.config).name]=a.config_sha
            coll.write_bundle(out,ids=ids,sids=result['sids'],vectors=x,input_manifest=im,config=config,
                 config_sha=a.config_sha,engineering=a.engineering,producer=producer)
            result={'status':'complete','manifest_sha256':file_sha256(out/coll.MANIFEST_NAME)}
        else:
            loaded=coll.load_bundle(a.bundle,a.sha,catalog_ids=ids,vectors=x,input_manifest=im,engineering=a.engineering)
            if gd._json(Path(a.bundle).parent/'tokenizer.json')!=config: raise ValueError('stored fit config differs')
            result=verify_mapping(loaded,ids,x,im,config,a.config_sha,engineering=a.engineering)
            result=verification_bindings(result,output=out,config_sha=a.config_sha,input_sha=a.input_manifest_sha,vectors_sha=a.vectors_sha,catalog_sha=a.catalog_sha,engineering=a.engineering)
            out.mkdir();(out/'verification.json').write_bytes(canonical_bytes(result))
        print(json.dumps(result,sort_keys=True));return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        print('SID-STRUCTURE gate failed: '+str(exc),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
