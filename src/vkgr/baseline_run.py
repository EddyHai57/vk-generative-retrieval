"""P4 runner consuming immutable P3 artifacts, never regenerating the contract."""
import json
from pathlib import Path
from .data_audit import file_sha256, canonical_bytes
from .manifest import read_json
from .baselines import metrics

CONFIG={'task_id':'P4','seed':42,'max_queries':512,'ks':[10,20],'query_block':32,'item_block':4096}

def validate_config(config):
    if type(config) is not dict or set(config)!=set(CONFIG):raise ValueError('invalid P4 config fields')
    for key in ('task_id','seed','max_queries','ks'):
        if type(config[key]) is not type(CONFIG[key]) or config[key]!=CONFIG[key]:raise ValueError('unapproved P4 config')
    if any(type(k) is not int for k in config['ks']):raise ValueError('invalid k')
    for key,upper in (('query_block',64),('item_block',8192)):
        if type(config[key]) is not int or not 1<=config[key]<=upper:raise ValueError('unbounded block')
    return config

def read_checked(path,digest):
    path=Path(path)
    if path.is_symlink() or file_sha256(path)!=digest:raise ValueError('artifact hash mismatch')
    return read_json(path)

def evaluate_predictions(predicted,queries,catalog,ks):
    if any(len(row)!=min(max(ks),len(catalog)) for row in predicted):raise ValueError('incomplete prediction')
    return metrics(predicted,queries,catalog,ks)

def verify_metrics(expected,predicted,queries,catalog,ks):
    if canonical_bytes(expected)!=canonical_bytes(evaluate_predictions(predicted,queries,catalog,ks)):raise ValueError('metric evidence mismatch')

def evaluate_inputs(inputs,config):
    import numpy as np
    import time
    from .baselines import sample_queries,history_centroids,blockwise_topk
    validate_config(config);catalog=np.asarray(inputs['catalog']);features=inputs['features'];counts=inputs['popularity']
    if counts.shape!=(len(catalog),) or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0):raise ValueError('invalid popularity counts')
    pop=catalog[np.lexsort((catalog,-counts))[:min(max(config['ks']),len(catalog))]].tolist()
    results={};artifacts={}
    for split in ('validation','test'):
        selected=sample_queries(inputs['queries'][split],split,config['seed'],config['max_queries'])
        started=time.monotonic();pop_predictions=[list(pop) for _ in selected];pop_seconds=time.monotonic()-started
        started=time.monotonic();centroids,zeros=history_centroids([q['history'] for q in selected],catalog,features)
        content=blockwise_topk(centroids,features,catalog,max(config['ks']),config['query_block'],config['item_block']).tolist();content_seconds=time.monotonic()-started
        results[split]={'query_count':len(selected),'zero_centroids':zeros,'audit_user_ids_sha256':__import__('hashlib').sha256(canonical_bytes([q['user_id'] for q in selected])).hexdigest(),
                        'popularity':evaluate_predictions(pop_predictions,selected,catalog,config['ks']),
                        'content':evaluate_predictions(content,selected,catalog,config['ks']),
                        'popularity_seconds':pop_seconds,'content_seconds':content_seconds,
                        'content_ms_per_query':content_seconds*1000/len(selected) if selected else None}
        artifacts[split]={'queries':selected,'popularity_predictions':pop_predictions,'content_predictions':content}
    return results,artifacts

def write_outputs(output,results,artifacts):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name,value in [('metrics.json',results)]+[(f'{split}-audit.json',value) for split,value in artifacts.items()]:
        with (output/name).open('xb') as stream:stream.write(canonical_bytes(value))

FROZEN_SHA='3b042b7635972e9a255c0b2cd8de0a7b7a472a08a0d074eb3133068d527c1e0f'
HOLDOUT_SHA='f9e44a97d410488b7c132abaa27da66076b27917474bcc1adaf0875f46c6f85d'

def load_static(manifest_path,holdout_directory,raw_root):
    """Validate frozen P3 bytes, not current-P4 producer equivalence."""
    import numpy as np
    import pyarrow.parquet as pq
    from .data_audit import validate_frozen_manifest
    from .baselines import join_normalized,popularity
    manifest_path=Path(manifest_path).resolve();holdout_directory=Path(holdout_directory).resolve();raw_root=Path(raw_root).resolve()
    frozen=validate_frozen_manifest(manifest_path,FROZEN_SHA);evidence=frozen['train_evidence']
    train=read_checked(manifest_path.parent/'train-audit.json',evidence['train_audit_sha256'])
    if train['producer']!=evidence['producer'] or train['policy_sha256']!=frozen['policy_sha256']:raise ValueError('P3 producer/policy binding mismatch')
    catalog=read_checked(manifest_path.parent/'catalog.json',evidence['catalog_sha256'])
    if any(type(i) is not int or not 0<=i<2**32 for i in catalog) or sorted(set(catalog))!=catalog or len(catalog)!=train['catalog_items']:raise ValueError('invalid frozen catalog')
    holdout=read_checked(holdout_directory/'holdout-audit.json',HOLDOUT_SHA)
    if holdout['frozen_manifest_sha256']!=FROZEN_SHA or holdout['policy_sha256']!=frozen['policy_sha256']:raise ValueError('holdout contract mismatch')
    queries={split:read_checked(holdout_directory/f'{split}-queries.json',holdout[split]['query_sha256']) for split in ('validation','test')}
    candidate_set=set(catalog)
    for split,rows in queries.items():
        if len(rows)!=holdout[split]['eligible_queries']:raise ValueError('query denominator mismatch')
        for q in rows:
            if not 1<=len(q['history'])<=50 or any(i not in candidate_set for i in q['history']) or q['oov']!=(q['target'] not in candidate_set):raise ValueError('query contract mismatch')
    records=[{'path':str(manifest_path),'sha256':FROZEN_SHA},{'path':str(manifest_path.parent/'train-audit.json'),'sha256':evidence['train_audit_sha256']},
             {'path':str(manifest_path.parent/'catalog.json'),'sha256':evidence['catalog_sha256']},{'path':str(holdout_directory/'holdout-audit.json'),'sha256':HOLDOUT_SHA}]
    records += [{'path':str(holdout_directory/f'{split}-queries.json'),'sha256':holdout[split]['query_sha256']} for split in queries]
    for rec in train['train_files']+[x for x in train['feature_files'] if x['path']=='metadata/item_embeddings.npz']:
        path=raw_root/rec['path']
        if path.is_symlink() or path.stat().st_size!=rec['size_bytes'] or file_sha256(path)!=rec['sha256']:raise ValueError('raw input hash mismatch')
        records.append({'path':str(path),'sha256':rec['sha256']})
    with np.load(raw_root/'metadata/item_embeddings.npz',allow_pickle=False) as content:
        ids=content['item_id'];embedding=content['embedding']
        if ids.dtype!=np.uint32 or embedding.dtype!=np.float16 or embedding.shape!=(len(ids),64):raise ValueError('official feature schema mismatch')
        mask=np.isin(ids,np.asarray(catalog,dtype=np.uint32));features=join_normalized(catalog,ids[mask],embedding[mask])
    def events():
        for rec in train['train_files']:
            for batch in pq.ParquetFile(raw_root/rec['path']).iter_batches(batch_size=65536,columns=['item_id','like'],use_threads=False):
                for item,like in zip(batch.column(0).to_pylist(),batch.column(1).to_pylist()):yield item,like
    counts=popularity(catalog,events())
    return {'catalog':catalog,'features':features,'popularity':counts,'queries':queries,
            'evidence':{'frozen_manifest_sha256':FROZEN_SHA,'policy_sha256':frozen['policy_sha256'],'p3_producer':evidence['producer'],'inputs':records,
                        'feature_zero_norm':int(np.count_nonzero(np.linalg.norm(features,axis=1)==0))}}

def run(manifest_path,holdout_directory,raw_root,config_path,output):
    import resource,time
    from datetime import datetime,timezone
    from .data_audit import producer_identity
    output=Path(output)
    if output.exists():raise FileExistsError('exclusive output required')
    config_path=Path(config_path).resolve();config=validate_config(read_json(config_path))
    producer=producer_identity();started=time.monotonic();utc=datetime.now(timezone.utc).isoformat()
    manifest_path=Path(manifest_path).resolve();holdout_directory=Path(holdout_directory).resolve();raw_root=Path(raw_root).resolve()
    inputs=load_static(manifest_path,holdout_directory,raw_root)
    results,artifacts=evaluate_inputs(inputs,config)
    write_outputs(output,results,artifacts)
    result={'schema_version':1,'task_id':'P4','status':'success','producer':producer,
            'config_path':str(config_path),'config_sha256':file_sha256(config_path),'effective_config':config,
            'frozen_manifest_path':str(Path(manifest_path).resolve()),'holdout_directory':str(Path(holdout_directory).resolve()),'raw_root':str(Path(raw_root).resolve()),
            'evidence':inputs['evidence'],'catalog_items':len(inputs['catalog']),
            'artifacts':{name:file_sha256(output/name) for name in ('metrics.json','validation-audit.json','test-audit.json')},
            'started_at_utc':utc,'elapsed_seconds':time.monotonic()-started,
            'peak_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_unit':'KiB_on_Linux_bytes_on_macOS',
            'score_dtype':'float64_fixed_einsum_on_float32_features','max_score_block_elements':config['query_block']*config['item_block'],
            'claim_boundary':'fixed query audit only; no trained model, full test, online or causal claim'}
    with (output/'run_manifest.json').open('xb') as stream:stream.write(canonical_bytes(result))
    return result

def verify_run(manifest_path):
    from .data_audit import validate_producer
    from .baselines import sample_queries
    manifest_path=Path(manifest_path);value=read_json(manifest_path)
    if value.get('schema_version')!=1 or value.get('task_id')!='P4' or value.get('status')!='success':raise ValueError('invalid P4 manifest')
    validate_producer(value['producer'])
    config=validate_config(read_checked(value['config_path'],value['config_sha256']))
    if canonical_bytes(config)!=canonical_bytes(value['effective_config']):raise ValueError('config binding mismatch')
    if set(value['artifacts'])!={'metrics.json','validation-audit.json','test-audit.json'}:raise ValueError('invalid artifact set')
    stored={name:read_checked(manifest_path.parent/name,digest) for name,digest in value['artifacts'].items()}
    inputs=load_static(value['frozen_manifest_path'],value['holdout_directory'],value['raw_root'])
    if canonical_bytes(inputs['evidence'])!=canonical_bytes(value['evidence']) or len(inputs['catalog'])!=value['catalog_items']:raise ValueError('input binding mismatch')
    expected,artifacts=evaluate_inputs(inputs,config)
    for split in ('validation','test'):
        record=stored[f'{split}-audit.json']
        if canonical_bytes(record)!=canonical_bytes(artifacts[split]):raise ValueError('baseline prediction/query replay mismatch')
        for method in ('popularity','content'):
            verify_metrics(stored['metrics.json'][split][method],record[f'{method}_predictions'],record['queries'],inputs['catalog'],config['ks'])
        for name in ('query_count','zero_centroids','audit_user_ids_sha256'):
            if stored['metrics.json'][split][name]!=expected[split][name]:raise ValueError('audit selection/count mismatch')
    if value['score_dtype']!='float64_fixed_einsum_on_float32_features' or value['max_score_block_elements']!=config['query_block']*config['item_block']:raise ValueError('score block binding mismatch')
    return value

def main(argv=None):
    import argparse,sys
    parser=argparse.ArgumentParser(description='P4 immutable exact baseline audit')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('run')
    for name in ('manifest','holdout-directory','raw-root','config','output'):p.add_argument('--'+name,required=True)
    p=commands.add_parser('verify');p.add_argument('--manifest',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='run':
            value=run(args.manifest,args.holdout_directory,args.raw_root,args.config,args.output)
            print(canonical_bytes({'task_id':'P4','manifest_sha256':file_sha256(Path(args.output)/'run_manifest.json'),'source_fingerprint':value['producer']['source_fingerprint']}).decode(),end='')
        else:
            verify_run(args.manifest);print('P4 manifest/input/query/prediction/metric replay verified')
    except (ValueError,OSError,KeyError,TypeError):
        print('ERROR P4 input, config, output, or evaluator gate failed',file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
