"""F7 train-only snapshot/quantile producer. No model training or checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
from . import profile_features as pf
from . import profile_like_audit as audit
from .data_audit import file_sha256

SCHEMA='profile-prepare-v1'

def read(path,pin):
    return audit.read(path,pin)
def write_new(path,value):
    data=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:f.write(data)
    return hashlib.sha256(data).hexdigest()

def real_inputs(spec,data_root=None):
    # Same immutable P4E/B1 inputs, using an explicit read-only contract root.
    import os
    from . import retrieval_b1_run as b1
    from .retrieval_data import validate_contract
    from .retrieval_experiment import load_training_inputs
    from .retrieval import PREPARED_SOURCE
    root=Path(data_root or os.environ.get('VKGR_DATA_ROOT',''))
    pf.require(root.is_absolute() and not root.is_symlink(),'explicit data contract root required')
    pins=validate_contract(root)
    query_pin=next(p for p in pins['full_query_pins'] if p['role']=='validation-queries.json')
    data=load_training_inputs(spec['prepared'],spec['holdout'],b1.PREPARED_SHA,PREPARED_SOURCE,query_pin['sha256']);data['synthetic']=False
    binding={'prepared_sha':b1.PREPARED_SHA,'query_sha':query_pin['sha256'],'catalog_sha':pins['catalog_sha256'],'test_gate':'closed_hash_only','data_channel':{'data_root':str(root.resolve()),'contract_sha256':file_sha256(root/'artifacts/p3d-retrieval-contract-manifest.json'),'bound_files':pins['bound_files']}}
    return data,None,binding

def original_data(inputs,inputs_sha):
    from . import retrieval_b1_run as b1,retrieval_feat02_run as fr
    spec=fr.validate_inputs(read(inputs,inputs_sha),'f3_f4_recency')
    data,_,binding=real_inputs(spec);data,extra=fr.apply_feat02(data,spec,50)
    rows,queries=fr._view(data,50)
    return {**data,'rows':rows,'queries':queries},spec,{**binding,**extra,'inputs_sha256':inputs_sha}

def raw_partitions(spec,history):
    import pyarrow.parquet as pq
    for week,record in enumerate(audit.train_records(history)):
        path=audit.safe(Path(spec['raw_root'])/record['role']);pf.require(file_sha256(path)==record['sha256'] and path.stat().st_size==record['size_bytes'],'pinned raw lineage')
        def events(p=path):
            for batch in pq.ParquetFile(p).iter_batches(batch_size=65536,columns=['user_id','item_id','like']):yield from batch.to_pylist()
        yield week,events()

def execute(inputs,inputs_sha,audit_path,audit_sha,output=None,preflight=False):
    from .retrieval_ablation_run import source_identity
    data,spec,binding=original_data(inputs,inputs_sha)
    accepted=read(audit_path,audit_sha);pf.require(accepted['status']=='like_audit_complete' and not accepted['test_read'] and not accepted['u1_read'] and accepted['audit']['invalid']==0,'accepted original like audit required')
    pf.require(accepted['input_sha256'][str(inputs)]==inputs_sha and accepted['side_manifest_sha256']==spec['side_sha256'],'audit input identity')
    history=read(spec['history_manifest'],spec['history_sha256'])
    for rec in audit.train_records(history):pf.require(accepted['input_sha256'][str(Path(spec['raw_root'])/rec['role'])]==rec['sha256'],'audit raw identity')
    base={'schema':SCHEMA,'status':'profile_prepare_preflight_ok' if preflight else 'complete','source':source_identity(),'inputs_sha256':inputs_sha,'like_audit_sha256':audit_sha,'inputs_binding':binding,'history_sha256':spec['history_sha256'],'side_sha256':spec['side_sha256'],'test_read':False,'checkpoint_read':False,'u1_read':False,'train_examples':len(data['rows']),'validation_queries':len(data['queries'])}
    pf.require(base['train_examples']==86785 and base['validation_queries']==2520,'fixed sample population')
    if preflight:return base
    profiles=pf.build_profiles(raw_partitions(spec,history),data['rows'],data['queries'])
    if accepted['audit']['missing']==0:
        for split,key in [('train','rows'),('validation','queries')]:
            for row,p in zip(data[key],profiles[split]):pf.require(p['counts'][2:]==[sum(v[0]&1 for v in row['side']),len(row['side'])],'profile raw/side like history mismatch')
    bounds=pf.fit_bounds(profiles['train'])
    out=Path(output);pf.require(out.is_absolute() and '..' not in out.parts and not any(p.is_symlink() for p in (out,*out.parents)),'new absolute nonsymlink output required');out.mkdir(parents=True,exist_ok=False)
    psha=write_new(out/'profiles.json',profiles);bsha=write_new(out/'quantiles.json',bounds)
    manifest={**base,'artifacts':{'profiles.json':psha,'quantiles.json':bsha},'fit_input_sha256':bounds['train_snapshots_sha256'],'fit_script_sha256':file_sha256(Path(__file__)),'profile_script_sha256':file_sha256(Path(pf.__file__)),'boundaries':bounds['boundaries'],'positive_samples':bounds['positive_samples'],'positive_interval_count':bounds['positive_interval_count'],'positive_occupied_bins':bounds['positive_occupied_bins']}
    msha=write_new(out/'manifest.json',manifest)
    return {'status':'profile_prepare_complete','manifest_sha256':msha,'manifest':manifest}

def load_profiles(path,pin,data,inputs_sha):
    path=Path(path);m=read(path,pin);pf.require(m['schema']==SCHEMA and m['status']=='complete' and m['inputs_sha256']==inputs_sha,'profile manifest identity')
    p=read(path.parent/'profiles.json',m['artifacts']['profiles.json']);bounds=read(path.parent/'quantiles.json',m['artifacts']['quantiles.json'])
    pf.require(pf.digest(p['train'])==m['fit_input_sha256']==bounds['train_snapshots_sha256'],'train-only fit binding')
    pf.require(m['profile_script_sha256']==file_sha256(Path(pf.__file__)) and m['fit_script_sha256']==file_sha256(Path(__file__)),'profile producer code identity')
    return {**data,'rows':pf.attach(data['rows'],p['train'],bounds,'train'),'queries':pf.attach(data['queries'],p['validation'],bounds,'validation'),'profile_manifest_sha256':pin,'profile_fit_sha256':m['artifacts']['quantiles.json']}

def _main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True);p.add_argument('--inputs-sha',required=True);p.add_argument('--audit',required=True);p.add_argument('--audit-sha',required=True);p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args();print(json.dumps(execute(a.inputs,a.inputs_sha,a.audit,a.audit_sha,a.output,a.preflight),sort_keys=True,allow_nan=False))
def main():
    from .profile_io import input_guard
    with input_guard():return _main()

if __name__=='__main__':main()
