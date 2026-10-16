"""Pinned, exclusive S1 queue-stage entry. No scheduling, SSH, retries or overwrite."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
import copy
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from vkgr.data_audit import canonical_bytes,file_sha256
from vkgr import sid_coll_sgns as sg, sid_coll_v1 as coll, sid_tok_audit as au, sid_v2

STAGES=('input_preflight','input','input_verify','sid_fit','sid_verify','preflight','cuda','dry','train','verify','diagnostic')
SCRIPT='scripts/sid_coll_job_entry.py'


def _hex(value,length=64):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{'+str(length)+'}',value):raise ValueError('invalid pinned SHA')
    return value


def checked_ref(ref):
    if type(ref) is not dict or set(ref)!={'path','sha256'}:raise ValueError('exact path/SHA reference required')
    path=Path(ref['path'])
    if not path.is_absolute() or '..' in path.parts:raise ValueError('absolute nontraversing reference required')
    if path.suffix.lower() in ('.pt','.pth','.ckpt') or re.search(r'week[_-]?26|/test/',str(path).lower()):
        raise ValueError('checkpoint/test reference prohibited; only guarded exporters/verifiers may read their checkpoint')
    if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('symlink reference prohibited')
    if not path.is_file() or file_sha256(path)!=_hex(ref['sha256']):raise ValueError('artifact SHA mismatch')
    return path


def _json(path):return json.loads(Path(path).read_bytes())
def _write(path,value):
    with Path(path).open('xb') as stream:stream.write(canonical_bytes(value))
def _proof(path):return {'path':str(Path(path).absolute()),'sha256':file_sha256(path)}


def verify_c1_audit(ids,histories,content,vectors,audit,*,engineering=False):
    """Independent data/algebra/progress checks; does not replay SGD gradients."""
    if not engineering and len(ids)!=120415:raise ValueError('formal catalog count')
    cfg=sg.SGNSConfig(**audit['config']);cfg.validate(engineering)
    if cfg.dim!=64 or cfg.epochs!=5:raise ValueError('S1 requires 64D and five epochs')
    if vectors.dtype!=np.float32 or vectors.shape!=(len(ids),64) or not np.isfinite(vectors).all():raise ValueError('finite float32 vectors')
    if not np.allclose(np.linalg.norm(vectors.astype('float64'),axis=1),1,atol=2e-6,rtol=0):raise ValueError('unit vectors required')
    expected_contract={'loss_reduction':'sum','optimizer':'SGD','momentum':0,'weight_decay':0,'gradient_clipping':None,
      'frequency_power':.75,'subsampling':False,'distance_weighting':False,'watch_ratio_weighting':False,
      'early_stopping':False,'export_table':'center_only','initial_input_bound':.5/cfg.dim,
      'positive_order':'user ascending; center ascending; context oldest to newest',
      'negative_rng':'numpy.default_rng(seed), continuous across epochs'}
    if audit['training_contract']!=expected_contract:raise ValueError('frozen SUM/SGD contract mismatch')
    if audit['schema']!='vkgr.sid_coll_sgns.v1' or audit['engineering'] is not engineering or audit['threads']!=1 or audit['deterministic_algorithms'] is not True:
        raise ValueError('SGNS producer runtime/schema mismatch')
    if audit['epochs_completed']!=5:raise ValueError('five complete epochs required')
    rows={int(x):i for i,x in enumerate(ids)};counts=np.zeros(len(ids),dtype=np.int64);frequency=np.zeros(len(ids),dtype=np.int64)
    for events in histories.values():
        for item in events:frequency[rows[int(item)]]+=1
    pairs=0
    for c,p in sg.iter_positive_pairs(histories,cfg.window):counts[rows[c]]+=1;counts[rows[p]]+=1;pairs+=1
    if pairs<=0:raise ValueError('no positive pairs')
    steps=math.ceil(pairs/cfg.batch_size);total=pairs*5;last_batch=pairs%cfg.batch_size or cfg.batch_size
    if audit['positive_pairs_per_epoch']!=pairs or audit['processed_positive_pairs']!=total or audit['steps']!=steps*5:
        raise ValueError('positive progress/step mismatch')
    lr=audit['learning_rates']
    expected_lrs={'initial':cfg.initial_lr,'last_batch':sg.learning_rate(total-last_batch,total,cfg.initial_lr,cfg.final_lr_ratio),
                  'final':sg.learning_rate(total,total,cfg.initial_lr,cfg.final_lr_ratio)}
    if any(type(lr.get(k)) not in (float,int) or not math.isclose(lr[k],v,rel_tol=1e-12,abs_tol=1e-15) for k,v in expected_lrs.items()):
        raise ValueError('learning rate endpoint/progress mismatch')
    epochs=audit['epoch_audits']
    if len(epochs)!=5:raise ValueError('epoch ledger length')
    for epoch,record in enumerate(epochs,1):
        if record['epoch']!=epoch or record['processed_positive_pairs']!=epoch*pairs or record['steps']!=epoch*steps or not math.isfinite(record['summed_loss']) or record['summed_loss']<0:
            raise ValueError('epoch ledger mismatch')
    fallback=counts==0
    expected={'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(vectors),
       'content_feature_sha256':au.feature_digest(content),'content_sha256':au.feature_digest(content),
       'train_sequence_sha256':sg._sequence_digest(histories),'raw_train_sequence_sha256':sg._sequence_digest(histories),
       'frequency_sha256':sg._array_digest(frequency),'positive_participation_sha256':sg._array_digest(counts),
       'fallback_mask_sha256':sg._array_digest(fallback)}
    if any(audit.get(k)!=v for k,v in expected.items()):raise ValueError('SGNS input/output/sequence digest mismatch')
    if audit['positive_participation']!=counts.tolist() or audit['fallback_mask']!=fallback.tolist() or audit['fallback_count']!=int(fallback.sum()) or audit['fallback_proportion']!=float(fallback.mean()):
        raise ValueError('fallback/participation ledger mismatch')
    norms=np.linalg.norm(content.astype('float64'),axis=1)
    if not np.isfinite(content).all() or np.any(norms[fallback]==0):raise ValueError('invalid fallback content')
    fallback_vectors=(content[fallback].astype('float64')/norms[fallback,None]).astype('float32')
    if not np.array_equal(vectors[fallback],fallback_vectors):raise ValueError('fallback algebra mismatch')
    health=sg.health_check(vectors,ids,histories)
    if audit['health']!=health:raise ValueError('health evidence mismatch')
    for key in ('negative_trace_sha256','input_initialization_sha256','output_table_sha256'):_hex(audit[key])
    if not engineering and au.feature_digest(content)!=au.CONTENT_V1_FEATURE_SHA256:raise ValueError('CONTENT_V1 source mismatch')
    return {'schema':'coll-input-verify-v1','status':'verified','gradient_replay':False,
       'negative_trace_replay':False,'negative_trace_sha256':audit['negative_trace_sha256'],
       'checks':['artifact_hashes','source_physical_hashes','sequence_recomputed','SUM_SGD_contract','five_epoch_progress',
                 'learning_rate_endpoints','fallback_algebra','unit_finite','health_recomputed','negative_trace_recorded_not_replayed'],
       'health':health}


def write_c1_producer(output,ids,histories,content,vectors,audit,spec,physical):
    output=Path(output);sg.save_sgns_output(output,vectors,audit)
    m={'schema':'coll-input-produced-v1','status':'produced','variant':'C1','dimension':64,'engineering':spec['engineering'],
       'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(vectors),
       'content_feature_sha256':au.feature_digest(content),'train_sequence_sha256':sg._sequence_digest(histories),
       'script_sha256':spec['source_files'][SCRIPT],'commit':spec['commit'],'checkpoint_sha256':None,
       'source_files':spec['source_files'],'input_sources':physical,
       'artifacts':{name:file_sha256(output/name) for name in ('vectors.npy','audit.json')}}
    path=output/'producer-manifest.json';_write(path,m);return path


def verify_c1_producer(path,pin,ids,histories,content,spec,physical):
    path=checked_ref({'path':str(Path(path).absolute()),'sha256':pin});m=_json(path)
    if any(m.get(k)!=v for k,v in {'schema':'coll-input-produced-v1','status':'produced','variant':'C1','dimension':64,
        'engineering':spec['engineering'],'commit':spec['commit'],'checkpoint_sha256':None,
        'script_sha256':spec['source_files'][SCRIPT],'source_files':spec['source_files'],'input_sources':physical}.items()):
        raise ValueError('producer source/commit/checkpoint binding mismatch')
    if set(m.get('artifacts',{}))!={'vectors.npy','audit.json'}:raise ValueError('producer artifact set')
    arrays={name:checked_ref({'path':str(path.parent/name),'sha256':sha}) for name,sha in m['artifacts'].items()}
    vectors=np.load(arrays['vectors.npy'],allow_pickle=False);audit=_json(arrays['audit.json'])
    proof=verify_c1_audit(ids,histories,content,vectors,audit,engineering=spec['engineering'])
    accepted={**m,'schema':'coll-input-v1','status':'accepted','producer_manifest_sha256':pin}
    coll.validate_input(ids,vectors,accepted,engineering=spec['engineering'])
    if m['train_sequence_sha256']!=sg._sequence_digest(histories) or m['content_feature_sha256']!=au.feature_digest(content):raise ValueError('producer sequence/content mismatch')
    return accepted,proof


def materialize_source_lock(template,parent_sha,sequence_sha,output):
    lock=copy.deepcopy(template)
    if lock.get('train_sequence_sha256')=='PENDING_TRAIN_SEQUENCE':lock['train_sequence_sha256']=sequence_sha
    elif lock.get('train_sequence_sha256')!=sequence_sha:raise ValueError('teacher lock raw sequence mismatch')
    _hex(sequence_sha);_hex(parent_sha)
    # Parent identity is recorded outside the exporter's strict original lock fields.
    path=Path(output)/'source-lock.json';_write(path,lock)
    return path


def prediction_rows(train_path,train,history_path,history_sha):
    history=_json(checked_ref({'path':str(history_path),'sha256':history_sha}))
    name='validation-l50.json'
    queries=_json(checked_ref({'path':str(Path(history_path).parent/name),'sha256':history['artifacts'][name]}))
    best=next(x for x in train['epochs'] if x['epoch']==train['best']['epoch'])
    relative=Path(best['evaluation']['path'])
    if relative.name!=str(relative):raise ValueError('evaluation directory escape')
    evaluation=Path(train_path).parent/relative/'manifest.json'
    value=_json(checked_ref({'path':str(evaluation),'sha256':best['evaluation']['manifest_sha256']}))
    if value.get('complete') is not True or value.get('query_count')!=len(queries):raise ValueError('prediction coverage')
    rows=[]
    for shard in value['shards']:
        shard_path=Path(shard['path'])
        if not shard_path.is_absolute():shard_path=evaluation.parent/shard_path
        if shard_path.parent!=evaluation.parent:raise ValueError('prediction shard path escape')
        record=_json(checked_ref({'path':str(shard_path),'sha256':shard['sha256']}))
        if shard['start']!=len(rows) or len(record['rows'])!=shard['end']-shard['start']:raise ValueError('prediction shard gap')
        rows.extend(record['rows'])
    if len(rows)!=len(queries):raise ValueError('prediction rows count')
    hits=0;ndcg=0.
    for ordinal,(row,query) in enumerate(zip(rows,queries)):
        if row['ordinal']!=ordinal or row['query_sha256']!=hashlib.sha256(canonical_bytes(query)).hexdigest():raise ValueError('prediction query identity')
        candidates=row['items'][:50]
        if query['target'] in candidates:
            hits+=1;ndcg+=1/math.log2(candidates.index(query['target'])+2)
    aggregate=best['metrics']['metrics']['all_eligible']
    if aggregate['count']!=len(queries) or not math.isclose(aggregate['recall@50'],hits/len(queries),abs_tol=1e-15) or not math.isclose(aggregate['ndcg@50'],ndcg/len(queries),abs_tol=1e-15):raise ValueError('integer hits/aggregate mismatch')
    from vkgr.generative_evaluation import _computation
    if _computation(rows,best['metrics'])!=best['evaluation']['computation_sha256'] or value['computation_sha256']!=best['evaluation']['computation_sha256']:
        raise ValueError('prediction computation digest mismatch')
    return queries,rows,hits,evaluation,best


def load_histories(plan,audit,catalog):
    """S0 exact columns/physical order, retaining user IDs for frozen user sort."""
    from analysis import sid_diag as sd
    import pyarrow.parquet as pq
    histories={};physical={};allowed=set(map(int,catalog))
    for week,rec in enumerate(sd.train_records(audit)):
        path=sd.check_ref({'path':str(Path(plan['raw_root'])/rec['path']),'sha256':rec['sha256']})
        if path.stat().st_size!=rec['size_bytes']:raise ValueError('train size mismatch')
        physical[rec['path']]=rec['sha256']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=['user_id','item_id']):
            for row in batch.to_pylist():
                user,item=row['user_id'],row['item_id']
                if type(user) is not int or not 0<=user<2**32 or type(item) is not int or item not in allowed:raise ValueError('raw user/item catalog mismatch')
                histories.setdefault(user,[]).append(item)
    return histories,physical


def resolve_refs(spec,overrides):
    resolved=copy.deepcopy(spec)
    for role,sha in overrides.items():
        if role not in spec['refs'] or spec['refs'][role].get('sha256')!='DYNAMIC':raise ValueError('static or unknown ref override prohibited')
        ref=spec['refs'][role];origin=ref.get('from')
        if type(origin) is not dict or set(origin)!={'receipt_path','stage','run_id','spec_sha256','proof'}:raise ValueError('declared dynamic producer required')
        _hex(sha);_hex(origin['spec_sha256'])
        receipt_path=Path(origin['receipt_path'])
        # The frozen producer spec/run binds the receipt; its actual SHA is kept in resolution evidence.
        prior=_json(checked_ref({'path':str(receipt_path),'sha256':file_sha256(receipt_path)}))
        expected_status={'input':'produced','sid_fit':'complete','train':'complete'}.get(origin['stage'],'verified')
        if any(prior.get(k)!=v for k,v in {'schema':'coll-stage-receipt-v1','stage':origin['stage'],'run_id':origin['run_id'],
            'variant':spec['variant'],'status':expected_status}.items()):raise ValueError('dynamic producer receipt identity mismatch')
        if prior.get('binding',{}).get('commit')!=spec['commit'] or prior['binding'].get('spec_sha256')!=origin['spec_sha256']:
            raise ValueError('dynamic producer source/spec binding mismatch')
        proof=({'path':str(receipt_path),'sha256':file_sha256(receipt_path)} if origin['proof']=='__receipt__' else prior.get('proofs',{}).get(origin['proof']))
        if proof!={'path':ref['path'],'sha256':sha}:raise ValueError('dynamic override differs from producer proof')
        checked_ref(proof)
        resolved['refs'][role]=proof
    return resolved


def validate_spec(spec):
    if type(spec) is not dict or set(spec)!={'schema','variant','engineering','commit','source_files','refs','stages'} or spec['schema']!='coll-stage-spec-v1':raise ValueError('stage spec schema')
    if spec['variant'] not in ('C1','C2') or type(spec['engineering']) is not bool:raise ValueError('stage variant/engineering')
    _hex(spec['commit'],40)
    if type(spec['source_files']) is not dict or SCRIPT not in spec['source_files']:raise ValueError('entry source pin required')
    actual=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    if actual!=spec['commit']:raise ValueError('deployed commit mismatch')
    for name,sha in spec['source_files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('source member must be repository relative')
        checked_ref({'path':str(ROOT/p),'sha256':sha})
        if not spec['engineering']:
            blob=subprocess.run(['git','-C',str(ROOT),'show',spec['commit']+':'+name],check=True,capture_output=True).stdout
            if hashlib.sha256(blob).hexdigest()!=sha:raise ValueError('source bytes differ from deployed commit')
    for ref in spec['refs'].values():
        if ref.get('sha256')=='DYNAMIC':
            if set(ref)!={'path','sha256','from'}:raise ValueError('dynamic ref declaration required')
        else:checked_ref(ref)
    if not spec['stages'] or any(k not in STAGES for k in spec['stages']):raise ValueError('unknown stage')
    for value in spec['stages'].values():
        if type(value) is not dict or set(value)!={'run_id','parameters'} or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}',value['run_id']):raise ValueError('stage run identity')
    return spec


def execute(stage,spec_path,spec_sha,output,*,overrides=None):
    spec_path=checked_ref({'path':str(spec_path),'sha256':spec_sha});frozen=validate_spec(_json(spec_path))
    spec=resolve_refs(frozen,overrides or {})
    if stage not in spec['stages']:raise ValueError('stage not frozen in spec')
    params=spec['stages'][stage]['parameters'];engineering=spec['engineering'];variant=spec['variant'];refs=spec['refs']
    output=Path(output)
    if not output.is_absolute() or any(p.is_symlink() for p in [output,*output.parents]):raise ValueError('exclusive absolute output required')
    output.mkdir(exist_ok=False)
    payload=output/'payload';proofs={};raw={};summary=None
    def path(role):
        if refs[role].get('sha256')=='DYNAMIC':raise ValueError('unresolved dynamic ref: '+role)
        return checked_ref(refs[role])
    def pin(role):return refs[role]['sha256']
    def read(role):return _json(path(role))
    def config():return path('config'+str(params['seed']))
    def data():
        ids=np.asarray(read('catalog'),dtype=np.uint32)
        if ids.ndim!=1 or np.any(ids[1:]<=ids[:-1]):raise ValueError('ascending catalog required')
        content=np.load(path('content'),allow_pickle=False)
        if content.dtype!=np.float32 or content.shape!=(len(ids)+1,64) or not np.isfinite(content).all() or np.any(content[0]):raise ValueError('CONTENT_V1 PAD shape')
        if not engineering and (len(ids)!=120415 or au.feature_digest(content[1:])!=au.CONTENT_V1_FEATURE_SHA256):raise ValueError('formal content/catalog mismatch')
        return ids,content
    def sequences(ids):
        plan=read('s0_plan')
        if not engineering and plan.get('catalog')!=refs['catalog']:raise ValueError('catalog differs from frozen S0 source plan')
        audit_ref=plan['train_audit'];audit=_json(checked_ref(audit_ref))
        histories,physical=load_histories(plan,audit,ids)
        physical['s0_plan']=pin('s0_plan');physical['train_audit']=audit_ref['sha256']
        physical['catalog']=pin('catalog');physical['content']=pin('content')
        return histories,physical
    if stage in ('input_preflight','input','input_verify'):
        ids,content=data();histories,physical=sequences(ids)
        if variant=='C2':physical['teacher_lock_template']=pin('teacher_lock')
        if stage=='input_preflight':
            from collections import Counter
            frequencies=Counter(item for events in histories.values() for item in events)
            participants=set();positive_pairs=0
            for center,positive in sg.iter_positive_pairs(histories,5):
                participants.add(center);participants.add(positive);positive_pairs+=1
            if positive_pairs<=0 or len(frequencies)<3:raise ValueError('positive/negative construction impossible')
            content_norm=np.linalg.norm(content[1:].astype('float64'),axis=1)
            fallback=np.array([int(item) not in participants for item in ids])
            if np.any(content_norm[fallback]==0):raise ValueError('fallback content zero')
            raw={'status':'verified','kind':'read_only_input_preflight','checkpoint_read':False,'training':False,
                 'catalog_items':len(ids),'users':len(histories),'events':sum(frequencies.values()),'positive_pairs_per_epoch':positive_pairs,
                 'negative_support_items':len(frequencies),'train_sequence_sha256':sg._sequence_digest(histories),
                 'source_physical_sha256':physical,'event_order':'physical week,row per actual user; SGNS users ascending'}
            if variant=='C2':
                from vkgr.sid_coll_export import metadata_preflight
                source_lock=materialize_source_lock(read('teacher_lock'),pin('teacher_lock'),sg._sequence_digest(histories),output)
                lock=_json(source_lock);support=np.load(path('support'),allow_pickle=False)
                metadata=metadata_preflight(params['teacher_root'],path('teacher_manifest'),pin('teacher_manifest'),lock,ids,content,support,engineering=engineering)
                raw['teacher_metadata']={k:v for k,v in metadata.items() if k!='checkpoint_path'}
                proofs['source_lock']=_proof(source_lock)
            _write(output/'verification.json',raw);proofs['verification']=_proof(output/'verification.json');status='verified'
        elif stage=='input':
            if variant=='C1':
                vectors,audit=sg.fit_sgns(ids,histories,content[1:],device=params.get('device','cpu'),engineering=engineering)
                producer=write_c1_producer(payload,ids,histories,content[1:],vectors,audit,spec,physical)
            else:
                from vkgr.sid_coll_export import export_c2
                lock=read('teacher_lock')
                source_lock=materialize_source_lock(lock,pin('teacher_lock'),sg._sequence_digest(histories),output)
                export_c2(params['teacher_root'],path('teacher_manifest'),pin('teacher_manifest'),source_lock,file_sha256(source_lock),
                    path('content'),pin('content'),path('catalog'),pin('catalog'),path('support'),pin('support'),payload,
                    commit=spec['commit'],engineering=engineering)
                producer=payload/'producer-manifest.json'
            _write(output/'input-sources.json',{'physical_sha256':physical,'train_sequence_sha256':sg._sequence_digest(histories)})
            proofs={'producer_manifest':_proof(producer),'vectors':_proof(payload/'vectors.npy'),'input_sources':_proof(output/'input-sources.json')}
            if variant=='C1':proofs['audit']=_proof(payload/'audit.json')
            else:proofs['source_lock']=_proof(source_lock)
            status='produced'
        else:
            producer=path('producer_manifest')
            if variant=='C1':accepted,raw=verify_c1_producer(producer,pin('producer_manifest'),ids,histories,content[1:],spec,physical)
            else:
                from vkgr.sid_coll_export import verify_c2_export
                lock=read('source_lock');template=read('teacher_lock')
                expected=copy.deepcopy(template)
                if expected.get('train_sequence_sha256')=='PENDING_TRAIN_SEQUENCE':expected['train_sequence_sha256']=sg._sequence_digest(histories)
                if lock!=expected or lock['train_sequence_sha256']!=sg._sequence_digest(histories):raise ValueError('C2 source lock template/sequence binding mismatch')
                producer_value=_json(producer)
                if producer_value.get('source_lock_sha256')!=pin('source_lock'):raise ValueError('C2 source lock SHA mismatch')
                if _json(path('input_sources'))!={'physical_sha256':physical,'train_sequence_sha256':sg._sequence_digest(histories)}:raise ValueError('C2 physical source mismatch')
                raw=verify_c2_export(producer.parent,pin('producer_manifest'),content,ids,lock,engineering=engineering)
                accepted=raw.pop('input_manifest')
                if accepted['commit']!=spec['commit'] or any(spec['source_files'].get(k)!=v for k,v in accepted['source_files'].items()):raise ValueError('C2 producer commit/source mismatch')
            _write(output/'input-manifest.json',accepted);_write(output/'verification.json',raw)
            proofs={'input_manifest':_proof(output/'input-manifest.json'),'verification':_proof(output/'verification.json'),'producer_manifest':_proof(producer)};status='verified'
    elif stage in ('sid_fit','sid_verify'):
        from vkgr import sid_coll_run as sr
        argv=['fit' if stage=='sid_fit' else 'verify']
        for role in ('fit_config','vectors','catalog','input_manifest'):
            flag={'fit_config':'config','input_manifest':'input-manifest'}.get(role,role)
            argv+=['--'+flag,str(path(role)),'--'+flag+'-sha',pin(role)]
        argv+=['--output',str(payload)]
        if engineering:argv+=['--engineering']
        if stage=='sid_verify':argv+=['--bundle',str(path('sid_manifest')),'--sha',pin('sid_manifest')]
        if sr.main(argv)!=0:raise ValueError('SID stage rejected')
        artifact=payload/(coll.MANIFEST_NAME if stage=='sid_fit' else 'verification.json')
        proofs={'sid_manifest' if stage=='sid_fit' else 'verification':_proof(artifact)}
        raw=_json(artifact);status='complete' if stage=='sid_fit' else 'verified'
        if raw.get('status') not in ('complete','verified'):raise ValueError('SID raw status mismatch')
        if stage=='sid_verify':
            from analysis.sid_coll_diagnostics import mapping_diagnostics
            ids,content=data();histories,physical=sequences(ids)
            im=read('input_manifest');vectors=np.load(path('vectors'),allow_pickle=False)
            loaded=coll.load_bundle(path('sid_manifest'),pin('sid_manifest'),catalog_ids=ids,vectors=vectors,input_manifest=im,engineering=engineering)
            mapping={int(i):tuple(map(int,code)) for i,code in zip(loaded.codec.catalog.ids,loaded.codec.catalog.sids)}
            diagnostics=mapping_diagnostics(mapping,[histories[u] for u in sorted(histories)])
            diagnostics['physical_sources']=physical
            _write(output/'mapping-diagnostics.json',diagnostics);proofs['mapping_diagnostics']=_proof(output/'mapping-diagnostics.json')
    elif stage in ('preflight','cuda','dry'):
        module=importlib.import_module('scripts.sid_coll_cuda_check')
        raw=getattr(module,{'preflight':'preflight','cuda':'rehearse','dry':'dry'}[stage])({**spec,'parameters':params},output/'detail')
        required_status={'preflight':'preflight_ok','cuda':'cuda_engineering_verified','dry':'dry_run_ok'}[stage]
        if raw.get('status')!=required_status:raise ValueError('engineering preflight failed')
        _write(output/'verification.json',raw);proofs={'verification':_proof(output/'verification.json')};status='verified'
    elif stage in ('train','verify'):
        from vkgr import generative_coll_run as gr
        root=ROOT;scratch=output/'scratch'
        if not engineering and os.environ.get('VKGR_DATA_ROOT')!=params['root']:raise ValueError('frozen external data root env mismatch')
        common=(root,path('prepared'),pin('prepared'),path('history'),pin('history'),path('sid_manifest'),pin('sid_manifest'),path('content'),pin('content'))
        kw={'temporary':scratch,'eval_workers':1,'decode':{'mode':'batched','batch_size':1},
            'input_manifest':path('input_manifest'),'input_sha':pin('input_manifest'),'vectors_path':path('vectors'),'vectors_sha':pin('vectors')}
        cfg=read('config'+str(params['seed']));gr.validate_config(cfg,engineering)
        if cfg['seed']!=params['seed']:raise ValueError('G seed differs from frozen stage')
        if stage=='train':
            raw=gr.run(*common,cfg,payload,**kw)
            if raw['status']!='complete':raise ValueError('train not complete')
            proofs={'manifest':_proof(payload/'manifest.json')};raw={'status':'complete','manifest_sha256':proofs['manifest']['sha256']};status='complete'
        else:
            train_receipt=read('train_receipt');manifest=path('train_manifest')
            if train_receipt.get('schema')!='coll-stage-receipt-v1' or train_receipt.get('stage')!='train' or train_receipt.get('status')!='complete' or train_receipt.get('variant')!=variant:
                raise ValueError('new own train receipt required before checkpoint access')
            if train_receipt.get('proofs',{}).get('manifest')!=refs['train_manifest'] or train_receipt['binding']['commit']!=spec['commit'] or train_receipt['binding']['config_sha256']!=pin('config'+str(params['seed'])):
                raise ValueError('own train manifest/source/config binding mismatch')
            train=_json(manifest)
            if train.get('schema')!=gr.MANIFEST_SCHEMA or train.get('config')!=cfg:raise ValueError('only own new S1 checkpoint permitted')
            raw=gr.verify(*common,manifest,pin('train_manifest'),payload,**kw)
            if raw.get('status')!='verified':raise ValueError('raw G verification required')
            queries,rows,hits,evaluation,best=prediction_rows(manifest,train,path('history'),pin('history'))
            aggregate=best['metrics']['metrics']['all_eligible']
            summary={'hits':hits,'R50':aggregate['recall@50'],'N50':aggregate['ndcg@50'],'best_epoch':train['best']['epoch'],
              'stop_epoch':train['selection']['epoch'],'prediction_sha256':best['evaluation']['computation_sha256'],
              'prediction_sha256_kind':'prediction_computation_sha256','prediction_computation_sha256':best['evaluation']['computation_sha256'],
              'prediction_manifest_file_sha256':file_sha256(evaluation),'manifest_sha256':pin('train_manifest')}
            _write(output/'verification.json',raw);proofs={'verification':_proof(output/'verification.json'),'train_manifest':refs['train_manifest'],'prediction_manifest':_proof(evaluation)};status='verified'
            if all(role in refs for role in ('catalog','slices','s0_result')):
                from analysis.sid_coll_diagnostics import prediction_diagnostics
                ids=np.asarray(read('catalog'),dtype=np.uint32);im=read('input_manifest');vectors=np.load(path('vectors'),allow_pickle=False)
                loaded=coll.load_bundle(path('sid_manifest'),pin('sid_manifest'),catalog_ids=ids,vectors=vectors,input_manifest=im,engineering=engineering)
                mapping={int(i):tuple(map(int,c)) for i,c in zip(loaded.codec.catalog.ids,loaded.codec.catalog.sids)}
                reference=read('s0_result')['C']['T3'][str(params['seed'])]
                diagnostics=prediction_diagnostics([q['target'] for q in queries],[r['items'][:50] for r in rows],mapping,read('slices'),reference)
            else:diagnostics={'B':{'status':'UNKNOWN'},'C':{'status':'UNKNOWN'},'C_delta_vs_T3':{'status':'UNKNOWN'},'reason':'diagnostic refs absent; acceptance gap'}
            _write(output/'prediction-diagnostics.json',diagnostics);proofs['prediction_diagnostics']=_proof(output/'prediction-diagnostics.json')

    else:
        from analysis.sid_coll_diagnostics import mapping_diagnostics, prediction_diagnostics
        ids,content=data();histories,physical=sequences(ids)
        input_value=read('input_manifest');vectors=np.load(path('vectors'),allow_pickle=False)
        loaded=coll.load_bundle(path('sid_manifest'),pin('sid_manifest'),catalog_ids=ids,vectors=vectors,input_manifest=input_value,engineering=engineering)
        mapping={int(i):tuple(map(int,c)) for i,c in zip(loaded.codec.catalog.ids,loaded.codec.catalog.sids)}
        raw=mapping_diagnostics(mapping,[histories[u] for u in sorted(histories)])
        raw.update({'B':{'status':'UNKNOWN','reason':'no verified G prediction reference'},'C':{'status':'UNKNOWN','reason':'no verified G prediction reference'}})
        if 'verify_receipt' in refs:
            verified=read('verify_receipt')
            if verified.get('stage')!='verify' or verified.get('status')!='verified' or verified.get('raw_proof',{}).get('status')!='verified' or verified.get('variant')!=variant or verified['binding']['commit']!=spec['commit']:
                raise ValueError('direct verified G proof required for B/C')
            if verified['proofs']['train_manifest']!=refs['train_manifest']:raise ValueError('B/C train manifest binding mismatch')
            train=read('train_manifest');queries,rows,hits,evaluation,best=prediction_rows(path('train_manifest'),train,path('history'),pin('history'))
            targets=[q['target'] for q in queries];predictions=[r['items'][:50] for r in rows]
            if 'slices' in refs and 's0_result' in refs:
                slices=read('slices');reference=read('s0_result')['C']['T3'][str(params['seed'])]
                raw.update(prediction_diagnostics(targets,predictions,mapping,slices,reference))
            else:raw['diagnostic_gap']='slices/s0_result references absent; B/C not accepted'
        raw.update({'status':'verified','physical_sources':physical,'test_read':False,'checkpoint_read':False})
        _write(output/'diagnostic.json',raw);proofs={'diagnostic':_proof(output/'diagnostic.json')};status='verified'
    receipt={'schema':'coll-stage-receipt-v1','status':status,'stage':stage,'variant':variant,
      'run_id':spec['stages'][stage]['run_id'],'binding':{'commit':spec['commit'],'spec_sha256':spec_sha,
      'config_sha256':pin('config'+str(params['seed'])) if stage in ('train','verify') else refs.get('fit_config',{}).get('sha256'),
      'data':{role:ref['sha256'] for role,ref in refs.items() if ref['sha256']!='DYNAMIC'},
      'resolved_from':{role:{'receipt_path':frozen['refs'][role]['from']['receipt_path'],
                     'receipt_sha256':file_sha256(frozen['refs'][role]['from']['receipt_path']),
                     'producer_spec_sha256':frozen['refs'][role]['from']['spec_sha256']} for role in (overrides or {})}},'proofs':proofs,'raw_proof':raw}
    if summary is not None:receipt['summary']=summary
    _write(output/'receipt.json',receipt);return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--stage',choices=STAGES,required=True);parser.add_argument('--spec',required=True)
    parser.add_argument('--spec-sha',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--ref',nargs=2,action='append',default=[],metavar=('ROLE','SHA'))
    args=parser.parse_args(argv)
    try:
        overrides=dict(args.ref)
        if len(overrides)!=len(args.ref):raise ValueError('duplicate dynamic ref override')
        receipt=execute(args.stage,args.spec,args.spec_sha,args.output,overrides=overrides)
        print(json.dumps({'status':receipt['status'],'receipt':_proof(Path(args.output)/'receipt.json')},sort_keys=True));return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        print('S1 stage STOP: '+str(exc),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
