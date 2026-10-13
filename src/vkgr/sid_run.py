"""P5 train-only SID artifacts and pinned static consumer validation."""
from pathlib import Path
import hashlib
import math
import numpy as np
from .data_audit import canonical_bytes,file_sha256
from .manifest import read_json
from .baseline_run import FROZEN_SHA,read_checked
from .sid import fit,SIDCatalog

CONFIG={'task_id':'P5','seed':42,'branching':16,'depth':3,'max_iter':10}
ARTIFACTS={'mapping.npz','constraint.npz','codebook.json','audit.json','normalized-features.npy'}


def validate_config(value):
    if type(value) is not dict or set(value)!=set(CONFIG) or any(type(value[k]) is not type(v) or value[k]!=v for k,v in CONFIG.items()):raise ValueError('unapproved SID config')
    return value


def feature_digest(features):
    return hashlib.sha256(np.asarray(features,dtype='<f4').tobytes(order='C')).hexdigest()


def check_gates(audit):
    if audit['full_sid_collisions'] or audit['roundtrip_items']!=audit['catalog_items'] or audit['reachable_items']!=audit['catalog_items'] or audit['valid_prefix_deadends'] or not audit['balanced_nodes'] or audit['max_leaf_occupancy']>math.ceil(audit['catalog_items']/16**3):raise ValueError('SID preregistered gate failed')


def write_artifacts(output,model,evidence,features):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'mapping.npz',ids=model.ids,sids=model.sids)
    np.savez(output/'constraint.npz',keys=model.keys,items=model.key_items)
    np.save(output/'normalized-features.npy',features,allow_pickle=False)
    audit=model.audit();check_gates(audit)
    for name,value in [('codebook.json',model.nodes),('audit.json',audit)]:
        with (output/name).open('xb') as stream:stream.write(canonical_bytes(value))
    value={'schema_version':1,'task_id':'P5','status':'success','effective_config':dict(CONFIG),'evidence':evidence,
           'feature_array_sha256':feature_digest(features),'catalog_items':len(model.ids),'artifacts':{name:file_sha256(output/name) for name in sorted(ARTIFACTS)}}
    # Data bundle only. A successful run manifest is published last by run().
    return value


def validate_success_manifest(value):
    import re
    required={'schema_version','task_id','status','effective_config','evidence','feature_array_sha256','catalog_items','artifacts','producer','resource_minimum','resource_samples','elapsed_seconds','rss_unit','started_at_utc'}
    if type(value) is not dict or not required.issubset(value) or type(value['schema_version']) is not int or value['schema_version']!=1 or value['task_id']!='P5' or value['status']!='success':raise ValueError('missing SID success evidence')
    producer=value['producer'];identity_keys={'git_commit','git_dirty','source_fingerprint','audit_module_sha256','lock_sha256','dependencies','python_version','python_major_minor','python_implementation'}
    if type(producer) is not dict or set(producer)!=identity_keys or producer['dependencies']!={'PyYAML':'6.0.2','numpy':'1.26.4','pyarrow':'19.0.1'} or producer['python_major_minor']!=[3,11] or any(type(x) is not int for x in producer['python_major_minor']) or producer['python_implementation']!='CPython' or type(producer['git_dirty']) is not bool or not isinstance(producer['python_version'],str) or not re.fullmatch(r'3\.11\.[0-9]+',producer['python_version']):raise ValueError('invalid static producer evidence')
    for key in ('git_commit','source_fingerprint','audit_module_sha256','lock_sha256'):
        if not isinstance(producer[key],str) or not re.fullmatch('[0-9a-f]{'+('40' if key=='git_commit' else '64')+'}',producer[key]):raise ValueError('invalid producer fingerprint')
    if type(value['resource_minimum']) is not dict or set(value['resource_minimum'])!={'ram_available_fraction','disk_available_fraction'} or type(value['resource_samples']) is not int or value['resource_samples']<1 or type(value['elapsed_seconds']) not in (int,float) or not math.isfinite(value['elapsed_seconds']) or value['elapsed_seconds']<0 or value['rss_unit']!='KiB_on_Linux_bytes_on_macOS':raise ValueError('invalid resource evidence')
    check_headroom(value['resource_minimum'])
    if any(value['resource_minimum'][key]>1 for key in value['resource_minimum']):raise ValueError('invalid resource fractions')
    from datetime import datetime
    if type(value['started_at_utc']) is not str or datetime.fromisoformat(value['started_at_utc']).utcoffset() is None:raise ValueError('invalid run start time')


def load_static(manifest_path,expected_sha256):
    """Consumer validates a caller-pinned P5 manifest; no current producer check.

    This separates P6 consumption from P5 producer replay without weakening P3.
    """
    manifest_path=Path(manifest_path);value=read_checked(manifest_path,expected_sha256)
    validate_success_manifest(value)
    validate_config(value['effective_config'])
    if value['evidence']['frozen_manifest_sha256']!=FROZEN_SHA or set(value['artifacts'])!=ARTIFACTS:raise ValueError('SID input/artifact binding mismatch')
    paths={}
    for name,digest in value['artifacts'].items():
        path=manifest_path.parent/name
        if path.is_symlink() or file_sha256(path)!=digest:raise ValueError('SID artifact hash mismatch')
        paths[name]=path
    with np.load(paths['mapping.npz'],allow_pickle=False) as data:
        if set(data.files)!={'ids','sids'}:raise ValueError('mapping schema')
        ids=data['ids'];sids=data['sids']
    if ids.dtype!=np.uint32 or ids.ndim!=1 or not len(ids) or np.any(ids[1:]<=ids[:-1]) or sids.dtype!=np.uint32 or sids.shape!=(len(ids),4) or np.any(sids[:,:3]>=16) or len(ids)!=value['catalog_items']:raise ValueError('invalid SID mapping')
    for prefix in np.unique(sids[:,:3],axis=0):
        rows=sids[np.all(sids[:,:3]==prefix,axis=1)]
        if not np.array_equal(rows[:,-1],np.arange(len(rows))):raise ValueError('noncanonical suffix')
    nodes=read_json(paths['codebook.json']);expected_prefixes={tuple(map(int,row[:length])) for row in sids for length in range(3)}
    if type(nodes) is not list or len(nodes)!=len(expected_prefixes):raise ValueError('codebook graph mismatch')
    visited=set()
    for node in nodes:
        prefix=tuple(node['prefix'])
        if prefix not in expected_prefixes or prefix in visited or any(type(x) is not int for x in prefix):raise ValueError('invalid codebook prefix')
        visited.add(prefix);rows=sids[np.all(sids[:,:len(prefix)]==prefix,axis=1)] if prefix else sids
        level=len(prefix);children,counts=np.unique(rows[:,level],return_counts=True);k=min(16,len(rows))
        if not np.array_equal(children,np.arange(k)) or node['sizes']!=counts.tolist() or max(counts)-min(counts)>1:raise ValueError('invalid child capacities')
        centers=np.asarray(node['centers'])
        if centers.shape!=(k,64) or not np.isfinite(centers).all() or type(node['iterations']) is not int or not 1<=node['iterations']<=10 or len(node['objectives'])!=node['iterations'] or any(type(x) is not float or not math.isfinite(x) or x<0 for x in node['objectives']) or type(node['converged']) is not bool:raise ValueError('invalid codebook values')
        selection=np.all(sids[:,:len(prefix)]==prefix,axis=1) if prefix else np.ones(len(ids),bool)
        init=node['init_item_ids']
        if len(init)!=k or len(set(init))!=k or any(type(x) is not int or x not in ids[selection] for x in init):raise ValueError('invalid center initialization')
    model=SIDCatalog(ids,sids,16,3,nodes)
    with np.load(paths['constraint.npz'],allow_pickle=False) as data:
        if set(data.files)!={'keys','items'} or data['keys'].dtype!=np.uint64 or data['items'].dtype!=np.uint32 or not np.array_equal(data['keys'],model.keys) or not np.array_equal(data['items'],model.key_items):raise ValueError('constraint mapping mismatch')
    features=np.load(paths['normalized-features.npy'],allow_pickle=False)
    if features.dtype!=np.float32 or features.shape!=(len(ids),64) or not np.isfinite(features).all() or feature_digest(features)!=value['feature_array_sha256']:raise ValueError('feature cache mismatch')
    audit=model.audit();check_gates(audit)
    if canonical_bytes(audit)!=canonical_bytes(read_json(paths['audit.json'])):raise ValueError('SID audit mismatch')
    return model

P4_MANIFEST_SHA='65d90cc9fb4df88824786a794349e3e2a6724ff0a8e08e4f95b573c699bdb801'
P4_SOURCE='51db4ea2320b0f8aae0a92d2036ba273bd1984966c6db553b640a5f17956176e'


def load_train_features(manifest_path,raw_root,p4_manifest_path):
    from .data_audit import validate_frozen_manifest
    from .baselines import join_normalized
    manifest_path=Path(manifest_path).resolve();raw_root=Path(raw_root).resolve();p4_manifest_path=Path(p4_manifest_path).resolve()
    frozen=validate_frozen_manifest(manifest_path,FROZEN_SHA);evidence=frozen['train_evidence']
    train=read_checked(manifest_path.parent/'train-audit.json',evidence['train_audit_sha256'])
    if train['producer']!=evidence['producer'] or train['policy_sha256']!=frozen['policy_sha256']:raise ValueError('P3 producer/policy mismatch')
    catalog=read_checked(manifest_path.parent/'catalog.json',evidence['catalog_sha256'])
    if any(type(x) is not int or not 0<=x<2**32 for x in catalog) or catalog!=sorted(set(catalog)) or len(catalog)!=train['catalog_items']:raise ValueError('invalid frozen catalog')
    p4=read_checked(p4_manifest_path,P4_MANIFEST_SHA)
    if p4['producer']['source_fingerprint']!=P4_SOURCE or p4['evidence']['frozen_manifest_sha256']!=FROZEN_SHA or p4['catalog_items']!=len(catalog):raise ValueError('P4 fixed input mismatch')
    records=[{'role':'p3_frozen','path':str(manifest_path),'sha256':FROZEN_SHA},{'role':'p3_train','path':str(manifest_path.parent/'train-audit.json'),'sha256':evidence['train_audit_sha256']},{'role':'catalog','path':str(manifest_path.parent/'catalog.json'),'sha256':evidence['catalog_sha256']},{'role':'p4_manifest','path':str(p4_manifest_path),'sha256':P4_MANIFEST_SHA}]
    content=[record for record in train['feature_files'] if record['path']=='metadata/item_embeddings.npz']
    if len(content)!=1:raise ValueError('missing official content evidence')
    record=content[0];path=raw_root/record['path']
    if path.is_symlink() or path.stat().st_size!=record['size_bytes'] or file_sha256(path)!=record['sha256']:raise ValueError('official feature hash mismatch')
    p4_feature=[x for x in p4['evidence']['inputs'] if x['path']==str(path)]
    if len(p4_feature)!=1 or p4_feature[0]['sha256']!=record['sha256']:raise ValueError('P4 feature source mismatch')
    records.append({'role':'official_features','path':str(path),'sha256':record['sha256']})
    with np.load(path,allow_pickle=False) as content:
        ids=content['item_id'];features=content['embedding']
        if ids.dtype!=np.uint32 or features.dtype!=np.float16 or features.shape!=(len(ids),64):raise ValueError('official feature schema mismatch')
        mask=np.isin(ids,np.asarray(catalog,dtype=np.uint32));normalized=join_normalized(catalog,ids[mask],features[mask])
    return {'catalog':catalog,'features':normalized,'evidence':{'frozen_manifest_sha256':FROZEN_SHA,'policy_sha256':frozen['policy_sha256'],'p3_producer':evidence['producer'],'p4_manifest_sha256':P4_MANIFEST_SHA,'p4_source_fingerprint':P4_SOURCE,'inputs':records,'feature_zero_norm':int(np.count_nonzero(np.linalg.norm(normalized,axis=1)==0))}}


def sample_headroom(root):
    import os
    path=Path('/proc/meminfo')
    if not path.exists():raise ValueError('real SID run requires Linux visible RAM measurement')
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in path.read_text().splitlines()}
    disk=os.statvfs(root)
    return {'ram_available_fraction':memory['MemAvailable']/memory['MemTotal'],'disk_available_fraction':disk.f_bavail/disk.f_blocks}


def check_headroom(sample):
    if any(type(sample[key]) not in (float,int) or not math.isfinite(sample[key]) or sample[key]<0.2 for key in ('ram_available_fraction','disk_available_fraction')):raise ValueError('resource headroom gate failed')


def run(manifest_path,raw_root,p4_manifest_path,config_path,output):
    import resource,time,threading
    from datetime import datetime,timezone
    from .data_audit import producer_identity
    output=Path(output)
    if output.exists():raise FileExistsError('exclusive SID output required')
    config_path=Path(config_path).resolve();config=validate_config(read_json(config_path));producer=producer_identity()
    paths=[str(Path(x).resolve()) for x in (manifest_path,raw_root,p4_manifest_path)]
    started=time.monotonic();utc=datetime.now(timezone.utc).isoformat();samples=[];errors=[];stop=threading.Event()
    initial=sample_headroom(output.parent);check_headroom(initial);samples.append(initial)
    def monitor():
        try:
            while not stop.wait(0.25):
                sample=sample_headroom(output.parent);samples.append(sample);check_headroom(sample)
        except Exception as error:
            errors.append(error);stop.set()
    def boundary():
        if errors:raise ValueError('resource sampling failed') from errors[0]
        for sample in samples:check_headroom(sample)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        inputs=load_train_features(*paths);boundary()
        model=fit(inputs['catalog'],inputs['features'],branching=config['branching'],depth=config['depth'],seed=config['seed'],max_iter=config['max_iter'],on_boundary=boundary)
        samples.append(sample_headroom(output.parent));boundary()
        value=write_artifacts(output,model,inputs['evidence'],inputs['features'])
        # Stop/join sampling before the final check so no pending worker error
        # can appear after a successful manifest has been published.
        stop.set();thread.join();samples.append(sample_headroom(output.parent));boundary()
        value.update({'producer':producer,'config_path':str(config_path),'config_sha256':file_sha256(config_path),'frozen_manifest_path':paths[0],'raw_root':paths[1],'p4_manifest_path':paths[2],
                      'started_at_utc':utc,'elapsed_seconds':time.monotonic()-started,'peak_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_unit':'KiB_on_Linux_bytes_on_macOS',
                      'resource_samples':len(samples),'resource_minimum':{key:min(sample[key] for sample in samples) for key in initial},
                      'resource_measurement':'250ms visible host sampling plus fit node/iteration boundaries; no instantaneous guarantee',
                      'claim_boundary':'train-only greedy balanced hierarchical kmeans; no optimality, learned model, or quality comparison'})
        validate_success_manifest(value)
        with (output/'run_manifest.json').open('xb') as stream:stream.write(canonical_bytes(value))
        return value
    finally:stop.set();thread.join()


def verify_run(manifest_path):
    from .data_audit import validate_producer
    path=Path(manifest_path);value=read_json(path);validate_producer(value['producer'])
    config=validate_config(read_checked(value['config_path'],value['config_sha256']))
    if config!=value['effective_config']:raise ValueError('SID config binding mismatch')
    model=load_static(path,file_sha256(path));inputs=load_train_features(value['frozen_manifest_path'],value['raw_root'],value['p4_manifest_path'])
    if canonical_bytes(inputs['evidence'])!=canonical_bytes(value['evidence']) or feature_digest(inputs['features'])!=value['feature_array_sha256']:raise ValueError('SID input replay mismatch')
    expected=fit(inputs['catalog'],inputs['features'],branching=16,depth=3,seed=42,max_iter=10)
    if not np.array_equal(model.ids,expected.ids) or not np.array_equal(model.sids,expected.sids) or canonical_bytes(model.nodes)!=canonical_bytes(expected.nodes):raise ValueError('SID fit/codebook replay mismatch')
    check_headroom(value['resource_minimum'])
    return value


def main(argv=None):
    import argparse,sys
    parser=argparse.ArgumentParser(description='P5 train-only deterministic SID audit')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run')
    for name in ('manifest','raw-root','p4-manifest','config','output'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('verify');p.add_argument('--manifest',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='run':
            value=run(args.manifest,args.raw_root,args.p4_manifest,args.config,args.output)
            print(canonical_bytes({'task_id':'P5','manifest_sha256':file_sha256(Path(args.output)/'run_manifest.json'),'source_fingerprint':value['producer']['source_fingerprint']}).decode(),end='')
        else:verify_run(args.manifest);print('P5 input/fit/map/codebook/constraint/audit replay verified')
    except (ValueError,OSError,KeyError,TypeError):print('ERROR P5 input, output, mapping or resource gate failed',file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
