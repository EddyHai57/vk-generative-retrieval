"""SID-FIX S0: read-only pinned analysis. Run only as a queued one-CPU job.

Plan: schema_version=1, root, history/catalog/train_audit pinned references,
raw_root, bundles keyed by label (path,sha256,branching,semantic_depth), and
runs keyed by label then seed. --preflight reads metadata and stats large inputs;
it does not hash/open NPZ or parquet. The execution output directory must be new.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re

try:
    from . import phase2_validation_read as archive
except ImportError:
    import phase2_validation_read as archive

READS = {}
SEEDS = (42, 2024, 2025)
BUNDLES = {'B16','random','T3','T1r','B32','B64','B128','B512','B1024','B256D2'}
RUN_LABELS = BUNDLES | {'A1B16','A1T3','Pmean'}
STEMS = {'B16':'g-hist-v2-l50','random':'g-g4-rand','T3':'g-tok-t3-bhkm-b256',
'T1r':'g-tok-t1r-rqvae-b256','A1B16':'g-garch01-a1-lr3e-4',
'A1T3':'g-gtok01b-t3-b256d3','A1B1024':'g-gtok02-a1-b1024d3','Pmean':'p-seqm1-mean',
**{label:'g-tok2-'+suffix for label,suffix in [('B1024','b1024d3'),('B512','b512d3'),
('B256D2','b256d2'),('B128','b128d3'),('B64','b64d3'),('B32','b32d3')]}}



def safe_path(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute non-traversing artifact path required')
    if path.suffix in ('.pt','.pth','.ckpt') or 'weights' in path.name.lower(): raise ValueError('model weights prohibited')
    if any(any(word in part.lower() for word in ('checkpoint','embedding')) or re.search(r'(^|[-_])test([._-]|$)',part.lower()) for part in path.parts):
        raise ValueError('prohibited artifact path')
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('symlink artifact prohibited')
    if not path.is_file(): raise ValueError('artifact missing')
    return path


def stat_ref(ref):
    if not re.fullmatch('[0-9a-f]{64}', ref.get('sha256','')): raise ValueError('SHA256 pin required')
    p = safe_path(ref['path'])
    return {'path': str(p), 'expected_sha256': ref['sha256'], 'size_bytes':p.stat().st_size}


def check_ref(ref):
    record = stat_ref(ref); p=Path(record['path']); h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    if h.hexdigest()!=ref['sha256']: raise ValueError('artifact SHA mismatch')
    READS[str(p)]=h.hexdigest()
    return p


def read_json(ref):
    p=check_ref(ref)
    if p.suffix!='.json' or p.stat().st_size>32*1024*1024: raise ValueError('bounded JSON required')
    return json.loads(p.read_bytes())


def load_mapping(ref,catalog):
    import numpy as np
    if Path(ref['path']).name!='mapping.npz': raise ValueError('only mapping.npz allowed')
    p=check_ref(ref)
    if p.name!='mapping.npz': raise ValueError('only mapping.npz allowed')
    with np.load(p,allow_pickle=False) as data:
        if set(data.files)!={'ids','sids'}: raise ValueError('strict ids/sids mapping schema')
        ids,sids=data['ids'],data['sids']
        if ids.ndim!=1 or sids.ndim!=2 or len(ids)!=len(sids) or sids.shape[1]!=4 or ids.dtype.kind not in 'iu' or sids.dtype.kind not in 'iu' or (sids<0).any():
            raise ValueError('invalid integer mapping shapes')
        keys=[int(x) for x in ids]
        if len(keys)!=len(set(keys)) or set(keys)!=set(catalog): raise ValueError('mapping catalog mismatch')
        if keys!=list(catalog): raise ValueError('mapping must retain frozen catalog order')
        return {i:tuple(map(int,row)) for i,row in zip(keys,sids)}


def validate_train_records(records):
    names=[f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet' for w in range(25)]
    if not isinstance(records,list) or [x.get('path') for x in records]!=names:
        raise ValueError('exactly ordered registered train weeks 00..24 required')
    for rec in records:
        if not re.fullmatch('[0-9a-f]{64}',rec.get('sha256','')) or type(rec.get('size_bytes')) is not int or rec['size_bytes']<0:
            raise ValueError('train file pin/size required')
    return records


def train_records(value):
    if 'train_files' in value: return validate_train_records(value['train_files'])
    records=value.get('binding',{}).get('input_records',[])
    records=[{'path':r['role'],'sha256':r['sha256'],'size_bytes':r['size_bytes']} for r in records if '/train/week_' in r.get('role','')]
    return validate_train_records(records)


def collect_histories(partitions,catalog):
    """All observed rows, preserving duplicates and physical per-user order."""
    histories={}; previous=-1; allowed=set(catalog)
    for week,events in partitions:
        if type(week) is not int or not previous<week<=24: raise ValueError('train week/order violation')
        previous=week
        for row in events:
            u,i=row['user_id'],row['item_id']
            if type(u) is not int or not 0<=u<2**32 or type(i) is not int or i not in allowed: raise ValueError('raw user/item catalog mismatch')
            histories.setdefault(u,[]).append(i)
    return list(histories.values())


def raw_histories(plan,records,catalog):
    import pyarrow.parquet as pq
    def partitions():
        for week,rec in enumerate(validate_train_records(records)):
            p=check_ref({'path':str(Path(plan['raw_root'])/rec['path']),'sha256':rec['sha256']})
            if p.stat().st_size!=rec['size_bytes']: raise ValueError('train file size mismatch')
            def events(path=p):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=65536,columns=['user_id','item_id']):
                    yield from batch.to_pylist()
            yield week, events()
    return collect_histories(partitions(),catalog)


def validate_predictions(predictions,queries,catalog,k=50):
    if len(predictions)!=len(queries): raise ValueError('ordered query coverage mismatch')
    for pred,q in zip(predictions,queries):
        if len(pred)!=k or len(set(pred))!=k or any(type(i) is not int or i not in catalog for i in pred): raise ValueError('prediction shape/identity/catalog mismatch')
        if q['oov'] != (q['target'] not in catalog): raise ValueError('query OOV identity mismatch')
    return predictions


def load_predictions(rid,queries,hist,state,catalog):
    """Reuse every immutable archive check, retaining only item predictions.

    phase2's public return contains hit vectors only. Intercept its pinned JSON
    reader during this synchronous call; no archive is reopened or bypassed.
    """
    original=archive.read; captured=[]
    def capture(path,pin=None):
        value,sha=original(path,pin)
        p=Path(path)
        if re.fullmatch(r'predictions-epoch-\d{3}\.json',p.name): captured.extend(value)
        if re.fullmatch(r'shard-\d{6}-\d{6}\.json',p.name): captured.extend(row['items'][:50] for row in value['rows'])
        return value,sha
    archive.read=capture
    try: checked=archive.load_archive(rid,queries,hist,state)
    finally: archive.read=original
    return validate_predictions(captured,queries,set(catalog)),checked['summary']


def validate_plan(plan):
    if plan.get('schema_version')!=1 or plan.get('mode','first_batch') not in ('first_batch','a1_b1024'): raise ValueError('plan schema/mode')
    labels=set(plan['runs']); allowed=RUN_LABELS|({'A1B1024'} if plan.get('mode')=='a1_b1024' else set())
    if plan.get('mode')=='a1_b1024':
        allowed={'A1B1024','A1T3'}; required_bundles={'B1024','T3'}
    else: required_bundles=BUNDLES
    if labels!=allowed or set(plan['bundles'])!=required_bundles: raise ValueError('unregistered config')
    for label,seeds in plan['runs'].items():
        expected=(42,) if label in ('B32','B64','B128') else SEEDS
        if set(map(int,seeds))!=set(expected): raise ValueError('registered seed coverage required')
        for seed,rid in seeds.items():
            if rid!=STEMS[label]+'-s'+str(seed): raise ValueError('run identity outside frozen configuration')
            if not isinstance(rid,str) or not re.fullmatch(r'[a-z0-9-]+',rid): raise ValueError('invalid run ID')
    for label,ref in plan['bundles'].items():
        if label=='random' and ref.get('derived_from')=='B16': continue
        b,d=ref['branching'],ref['semantic_depth']
        expected=(256,2) if label=='B256D2' else (16 if label in ('B16','random') else 256 if label in ('T3','T1r') else int(label[1:]),3)
        if (b,d)!=expected: raise ValueError('frozen mapping configuration mismatch')
    return plan


def _plan_inventory(plan):
    """Metadata-only discovery: no parquet or NPZ payload is opened."""
    validate_plan(plan); archive.ROOT=Path(plan['root']); archive.READS.clear()
    hist=read_json(plan['history'])
    if plan['history']['sha256']!=archive.HISTORY_PIN: raise ValueError('history must match immutable N2520 pin')
    audit=read_json(plan['train_audit']); records=train_records(audit)
    states={rid:archive.inspect_run(rid) for seeds in plan['runs'].values() for rid in seeds.values()}
    metadata={}
    for rid,state in states.items():
        if state.get('manifest_path'):
            manifest,_=archive.read(state['manifest_path'],state['manifest_sha256'])
            metadata[rid]={'binding':manifest.get('binding'), 'config':manifest.get('config')}
    slices_path=Path(plan['history']['path']).parent/'train-slices.json'
    slices=read_json({'path':str(slices_path),'sha256':hist['artifacts']['train-slices.json']})
    return {'runs':states,'run_input_bindings':metadata,'bundles':{k:({'derived_from':'B16','seed':42} if k=='random' and v.get('derived_from')=='B16' else stat_ref(v)) for k,v in plan['bundles'].items()},'train_files':[stat_ref({'path':str(Path(plan['raw_root'])/r['path']),'sha256':r['sha256']}) for r in records], 'catalog':stat_ref(plan['catalog']), 'thresholds':slices['popularity_thresholds'],'threshold_source':{'path':str(slices_path),'sha256':hist['artifacts']['train-slices.json'],'definition':slices['definition'],'bins':'head > p90; middle > median and <= p90; tail <= median; OOV separate'},'read_artifacts_sha256':{**READS,**archive.READS}}


def run(plan,outdir):
    try: from . import sid_diag_metrics as metrics
    except ImportError: import sid_diag_metrics as metrics
    validate_plan(plan)
    outdir=Path(outdir)
    if outdir.exists() or any(p.is_symlink() for p in [outdir,*outdir.parents]): raise ValueError('new exclusive output directory required')
    root=Path(plan['root']).resolve()
    if any(outdir.resolve().is_relative_to(Path(ref['path']).parent.resolve()) for ref in [plan['history'],plan['catalog'],plan['train_audit'],*[r for r in plan['bundles'].values() if 'path' in r]]):
        raise ValueError('output cannot be inside an input artifact directory')
    inventory=collect_inventory(plan)
    if not all(s['verified'] for s in inventory['runs'].values()): raise ValueError('all registered run verifiers must pass')
    hist=read_json(plan['history']); catalog=read_json(plan['catalog'])
    if len(catalog)!=120415 or len(set(catalog))!=len(catalog) or archive.plain_digest(catalog)!=hist['catalog_sha256'] or plan['catalog']['sha256']!=hist['binding']['catalog_sha']: raise ValueError('frozen catalog mismatch')
    queries=read_json({'path':str(Path(plan['history']['path']).parent/'validation-l50.json'),'sha256':hist['artifacts']['validation-l50.json']})
    if len(queries)!=archive.N or sum(q['oov'] for q in queries)!=623: raise ValueError('frozen query population mismatch')
    hist['_g_catalog_digest']=archive.digest(catalog); hist['_base_queries_canonical_digest']=hist['binding']['query_sha']
    slices=read_json({'path':inventory['threshold_source']['path'],'sha256':inventory['threshold_source']['sha256']})
    audit=read_json(plan['train_audit']); histories=raw_histories(plan,train_records(audit),catalog)
    mappings={label:load_mapping(ref,catalog) for label,ref in plan['bundles'].items() if label!='random'}
    if 'random' in plan['bundles']:
        if plan['bundles']['random'].get('derived_from')!='B16': raise ValueError('random control must derive from B16')
        mappings['random']=random_mapping(mappings['B16'])
    result={'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_metadata':{},'task_id':'SID-FIX-S0','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':archive.digest(plan),'inventory':inventory,'A':{},'B':{},'C':{},'D':{},'decomposition':{}}
    for label,mapping in mappings.items():
        ref=plan['bundles']['B16' if label=='random' else label]
        result['A'][label]=metrics.structure_metrics(mapping,ref['branching'],ref['semantic_depth'])
        result['D'][label]=metrics.cooccurrence_metrics(mapping,histories,seed=0)
    targets=[q['target'] for q in queries]; th=slices['popularity_thresholds']; counts={int(k):v for k,v in slices['item_like_counts'].items()}
    for label,seeds in plan['runs'].items():
        result['B'][label]={}; result['C'][label]={}
        for seed,rid in seeds.items():
            preds,summary=load_predictions(rid,queries,hist,inventory['runs'][rid],catalog)
            sidlabel={'A1B16':'B16','A1T3':'T3','A1B1024':'B1024'}.get(label,label)
            if sidlabel in mappings: result['B'][label][str(seed)]=metrics.prediction_metrics(targets,preds,mappings[sidlabel])
            result['C'][label][str(seed)]=metrics.popularity_metrics(targets,preds,counts,th['median'],th['p90'],catalog)
    for label,ref,seeds in [('B1024','T3',(2024,2025)),('A1B1024','A1T3',SEEDS)]:
        if label in result['C'] and ref in result['C']:
            convert=lambda rows:{int(seed):{b:r['hits'] for b,r in value['bins'].items() if b!='oov'} for seed,value in rows.items()}
            result['decomposition'][label]=metrics.decompose_gain(convert(result['C'][label]),convert(result['C'][ref]),seeds)
    result['B_seed_means']={label:mean_tree(rows) for label,rows in result['B'].items() if len(rows)==3}
    result['C_delta_vs_same_seed_T3']={}
    for label,rows in result['C'].items():
        ref='A1T3' if label.startswith('A1') else 'T3'
        if ref not in result['C']: continue
        result['C_delta_vs_same_seed_T3'][label]={seed:{b:value['bins'][b]['hits']-result['C'][ref][seed]['bins'][b]['hits'] for b in value['bins']} for seed,value in rows.items()}
    result['read_artifacts_sha256']={**inventory['read_artifacts_sha256'],**READS,**archive.READS}; result['restrictions']={'test_read':False,'checkpoint_read':False,'embedding_read':False,'training':False,'raw_history':'all observed train rows; physical week,row order; repeats retained'}
    outdir.mkdir(parents=True,exist_ok=False)
    (outdir/'result.json').write_bytes(archive.canonical(result))
    report=render_report(result)
    (outdir/'report.md').write_text(report)
    return result


def render_report(result):
    lines=['# SID-FIX S0','', '完整机器可读结果与逐输入 SHA256 收据：result.json。', '',
           '热度分档沿 G4 train-only like counts：high > p90；mid 为 median < count <= p90；low 为 0 < count <= median；unsupported 为 0；OOV 单列。',
           json.dumps(result['inventory']['threshold_source'],ensure_ascii=False,sort_keys=True),'',
           '50% 头部增益分界只作描述，没有更强依据。B1024 判读仅用 2024、2025；s42 只并列展示。', '',
           '## A SID 结构','', '|配置|视频数|语义叶子数|视频/叶子|最大叶|有效语义层数|', '|---|---:|---:|---:|---:|---:|']
    for label,value in result['A'].items():
        leaf=value['leaves']
        lines.append('|'+ '|'.join(map(str,[label,value['n_items'],leaf['count'],leaf.get('avg_items',leaf.get('mean_items')),leaf.get('max_items'),value['effective_layers']]))+'|')
    lines.extend(['','## B 每位预测','', '|配置|seed|in-catalog|OOV|top1 前1/2/3|top50 前1/2/3|item hits|', '|---|---|---:|---:|---|---|---:|'])
    for label,rows in result['B'].items():
        for seed,value in rows.items():
            prefix=value['prefix_by_k']; one=[prefix[str(k)]['top1_rate'] for k in (1,2,3)]; fifty=[prefix[str(k)]['top50_rate'] for k in (1,2,3)]
            lines.append('|'+ '|'.join(map(str,[label,seed,value['n_in_catalog'],value['n_oov'],one,fifty,value['exact_item']['top50_hits']]))+'|')
    lines.extend(['','三 seed 均值详见 result.json 的 B_seed_means。','','## C 热度命中与相对同 seed T3 增益','','|配置|seed|high|mid|low|unsupported|OOV|同 seed 差|','|---|---|---:|---:|---:|---:|---:|---|'])
    for label,rows in result['C'].items():
        for seed,value in rows.items():
            hits=[value['bins'][b]['hits'] for b in ('high','mid','low','unsupported','oov')]
            delta=result['C_delta_vs_same_seed_T3'].get(label,{}).get(seed)
            lines.append('|'+ '|'.join(map(str,[label,seed,*hits,delta]))+'|')
    lines.extend(['','## D 前缀共现 lift','','|配置|pair 数|lift1|lift2|lift3|','|---|---:|---:|---:|---:|'])
    for label,value in result['D'].items():
        lift=[value['prefix_by_k'][str(k)]['lift'] for k in (1,2,3)]
        lines.append('|'+ '|'.join(map(str,[label,value['n_pairs'],*['UNKNOWN' if x is None else x for x in lift]]))+'|')
    lines.extend(['','## 冻结判读','',json.dumps(result['decomposition'],ensure_ascii=False,sort_keys=True),''])
    return '\n'.join(lines)


def contains_pin(value,pin):
    if isinstance(value,dict): return any(contains_pin(v,pin) for v in value.values())
    if isinstance(value,list): return any(contains_pin(v,pin) for v in value)
    return value==pin


def claimed_source(root,rid):
    jobs=sorted((Path(root)/'queue/claimed').glob('*-'+rid+'.json'))
    if len(jobs)!=1: raise ValueError('exact registered training source claim required')
    job,jsha=archive.read(jobs[0])
    if job.get('run_id')!=rid: raise ValueError('claimed training run identity mismatch')
    argv=job.get('argv',[])
    if not isinstance(argv,list) or any(not isinstance(v,str) for v in argv): raise ValueError('invalid source argv')
    return job,{'path':str(jobs[0]),'sha256':jsha}


def registered_sid_source(root,rid,cache=None):
    """One resolver for discovery and admission, including G3 prepared -> spec.

    Cache only immutable (path,pin) documents, never per-run derived flags.
    Every invocation rechecks the registered run's own unique claimed receipt.
    """
    cache={} if cache is None else cache
    job,_=claimed_source(root,rid);argv=job['argv']
    def flag(name):
        if argv.count(name)>1: raise ValueError('duplicate source argv flag')
        if name not in argv: return None
        i=argv.index(name)
        if i+1==len(argv): raise ValueError('source flag has no value')
        return argv[i+1]
    def document(path,pin):
        key=(path,pin)
        if key not in cache:
            if not path or not pin: raise ValueError('pinned source metadata required')
            if safe_path(path).stat().st_size>1024*1024: raise ValueError('oversized source metadata')
            value,sha=archive.read(path,pin)
            cache[key]={'sha256':sha,'value':value}
        return cache[key]['value']
    path=flag('--sid-manifest');pin=flag('--sid-sha') or flag('--sid-manifest-sha')
    if path is not None or pin is not None:
        if not path or not pin: raise ValueError('both SID source path and SHA required')
        return {'path':path,'sha256':pin}
    prepared=document(flag('--prepared'),flag('--prepared-sha'))
    if prepared.get('kind')!='real_prepared': raise ValueError('unknown prepared source schema')
    spec=document(prepared.get('spec_path'),prepared.get('spec_sha256'))
    if not spec.get('sid_path') or not spec.get('sid_sha256'): raise ValueError('prepared spec missing SID input')
    return {'path':spec['sid_path'],'sha256':spec['sid_sha256']}


def verify_bundle_bindings(plan,inventory):
    """Prove each map is the exact SID input of its registered source jobs."""
    cache={}
    for label,ref in plan['bundles'].items():
        if label=='random': continue
        source=ref.get('source_manifest')
        if not source: raise ValueError('bundle source_manifest pin required')
        manifest=read_json(source)
        if manifest.get('artifacts',{}).get('mapping.npz')!=ref['sha256'] or Path(ref['path'])!=Path(source['path']).parent/'mapping.npz':
            raise ValueError('mapping not bound by source manifest')
        source_labels=[label]+[a for a,b in {'A1B16':'B16','A1T3':'T3','A1B1024':'B1024'}.items() if b==label]
        for rid in [r for name in source_labels for r in plan['runs'].get(name,{}).values()]:
            if registered_sid_source(plan['root'],rid,cache)!=source:
                raise ValueError('bundle differs from registered training source')


def assert_train_source_control(plan,history,inventory=None):
    """Fixed registered P anchor is metadata-only provenance in both modes."""
    rid='p-seqm1-mean-s42'
    if plan.get('train_source_control',rid)!=rid: raise ValueError('only fixed Pmean42 source control allowed')
    archive.ROOT=Path(plan['root'])
    state=(inventory or {}).get('runs',{}).get(rid)
    if state is None: state=archive.inspect_run(rid)
    if not state.get('verified') or (state.get('terminal') or {}).get('status')!='succeeded': raise ValueError('train source control not verified')
    m,_=archive.read(state['manifest_path'],state['manifest_sha256']); binding=m.get('binding',{});inputs=binding.get('inputs',{})
    if binding.get('prepared_sha')!=plan['train_audit']['sha256']: raise ValueError('train source pin differs from fixed P control prepared_sha')
    if plan['history']['sha256']!=archive.HISTORY_PIN or inputs.get('history_manifest_sha256')!=archive.HISTORY_PIN or inputs.get('query_sha')!=history['binding']['query_sha']:
        raise ValueError('train source control history/query mismatch')
    if binding.get('catalog_sha256')!=history['catalog_sha256'] or inputs.get('catalog_sha')!=history['binding']['catalog_sha']:
        raise ValueError('train source control catalog mismatch')
    return {'run_id':rid,'manifest_path':state['manifest_path'],'manifest_sha256':state['manifest_sha256'],'prepared_sha256':binding['prepared_sha'],'verified':True,'role':'metadata_only_train_source_provenance'}


def assert_metadata_bindings(plan,inventory):
    verify_bundle_bindings(plan,inventory)
    history=read_json(plan['history'])
    inventory['train_source_control']=assert_train_source_control(plan,history,inventory)
    if not all(state.get('verified') for state in inventory['runs'].values()): raise ValueError('all registered run verifiers must pass before admission')
    inventory['read_artifacts_sha256'].update({**READS,**archive.READS})


def mean_tree(rows):
    """Arithmetic mean across three seeds of numeric B metrics."""
    values=list(rows.values())
    def average(v):
        first=v[0]
        if isinstance(first,dict): return {k:average([x[k] for x in v]) for k in first}
        if type(first) in (int,float): return sum(v)/len(v) if all(x is not None for x in v) else None
        if all(x==first for x in v): return first
        return None
    return average(values)


def random_mapping(base):
    """Exact G4 seed42 prefix permutation and canonical per-leaf suffix."""
    import numpy as np
    ids=np.array(list(base),dtype=np.uint32); sids=np.array(list(base.values()),dtype=np.uint32)
    if sids.shape[1]!=4: raise ValueError('random control requires B16D3 plus suffix')
    prefixes=sids[np.random.RandomState(42).permutation(len(ids)),:3]
    order=np.lexsort((ids,prefixes[:,2],prefixes[:,1],prefixes[:,0]))
    codes=prefixes[order]; starts=np.r_[True,np.any(codes[1:]!=codes[:-1],axis=1)]
    rows=np.arange(len(codes)); suffix=rows-np.maximum.accumulate(np.where(starts,rows,0))
    new=np.zeros_like(sids);new[:,:3]=prefixes;new[order,3]=suffix.astype(np.uint32)
    return {int(i):tuple(map(int,row)) for i,row in zip(ids,new)}


def prepared_metadata_path(spec):
    return Path(spec['prepared'])/'prepare-manifest.json'


def collect_inventory(plan=None,root=None):
    """No-plan form discovers only small metadata plus mapping/train file stats.

    Return bindings and source job flags for constructing a hash-pinned plan;
    never opens mapping arrays, raw parquet, content vectors, or checkpoints.
    """
    if plan is not None:
        inventory=_plan_inventory(plan)
        assert_metadata_bindings(plan,inventory)
        return inventory
    archive.ROOT=Path(root) if root else archive.ROOT
    labels=sorted(RUN_LABELS)
    result={'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_metadata':{},'task_id':'SID-FIX-S0','mode':'first_batch','runs':{},'source_jobs':{},'bundle_manifests':{},'history':{},'catalog':{},'unresolved':[]}
    hpath=archive.ROOT/'runs/h2-prepare/out/manifest.json'
    hist,hsha=archive.read(hpath,archive.HISTORY_PIN)
    result['history']={'path':str(hpath),'sha256':hsha}
    cpath=archive.ROOT.parent/'vkgr-p4b-20261007t-pipeline-03/prepared-v3/catalog.json'
    result['catalog']=stat_ref({'path':str(cpath),'sha256':hist['binding']['catalog_sha']})
    result['history_binding']=hist['binding']
    sp=hpath.parent/'train-slices.json';slices,sha=archive.read(sp,hist['artifacts']['train-slices.json'])
    result['thresholds']={'values':slices['popularity_thresholds'],'path':str(sp),'sha256':sha,'definition':slices['definition'],'bins':'high > p90; mid median < count <= p90; low 0 < count <= median; unsupported count=0; OOV separate'}
    allowed={'--sid-manifest','--sid-sha','--sid-manifest-sha','--spec','--spec-sha','--input-spec','--input-spec-sha','--raw-root','--train-directory','--train-dir','--prepared','--prepared-sha','--prepared-manifest','--manifest','--manifest-sha','--history-manifest','--history-sha'}
    base=archive.ROOT.parent/'vkgr-b1-inputs/p-feat02-inputs-base-v1.json'
    if base.exists():
        value,bsha=archive.read(base)
        result['input_metadata'][str(base)]={'sha256':bsha,'value':value}
    result['_source_cache']={}
    for label in labels:
        result['runs'][label]={}
        for seed in ((42,) if label in ('B32','B64','B128') else SEEDS):
            rid=STEMS[label]+'-s'+str(seed);state=archive.inspect_run(rid)
            result['runs'][label][str(seed)]=state
            jobs=sorted((archive.ROOT/'queue/claimed').glob('*-'+rid+'.json'))
            if len(jobs)!=1:
                result['unresolved'].append({'run_id':rid,'reason':'exact claimed source job missing'});continue
            job,jsha=archive.read(jobs[0]);argv=job.get('argv',[])
            flags={v:argv[i+1] for i,v in enumerate(argv[:-1]) if v in allowed}
            result['source_jobs'][rid]={'path':str(jobs[0]),'sha256':jsha,'input_flags':flags,'expected_data_hashes':job.get('expected_data_hashes')}
            if state.get('manifest_path'):
                manifest,_=archive.read(state['manifest_path'],state['manifest_sha256'])
                state['input_binding']=manifest.get('binding')
            if label=='Pmean': continue
            source=registered_sid_source(archive.ROOT,rid,result['_source_cache'])
            mp=source['path'];pin=source['sha256']
            if mp and pin and mp not in result['bundle_manifests']:
                bundle,bsha=archive.read(mp,pin)
                artifacts=bundle.get('artifacts',{})
                rec=artifacts.get('mapping.npz')
                if isinstance(rec,dict): rec=rec.get('sha256')
                entry={'sha256':bsha,'manifest':bundle}
                if rec: entry['mapping']=stat_ref({'path':str(Path(mp).parent/'mapping.npz'),'sha256':rec})
                result['bundle_manifests'][mp]=entry
    for (path,pin),entry in result.pop('_source_cache').items(): result['input_metadata'][path]=entry
    if str(base) in result['input_metadata']:
        spec=result['input_metadata'][str(base)]['value']
        if 'prepared' in spec and 'raw_root' in spec:
            prepared_path=prepared_metadata_path(spec)
            prepared_value,psha=archive.read(prepared_path)
            pbindings=[state.get('input_binding',{}) for state in result['runs'].get('Pmean',{}).values()]
            if not any(contains_pin(b,psha) for b in pbindings): raise ValueError('discovered prepared manifest absent from P binding')
            records=train_records(prepared_value)
            result['train_source']={'path':str(prepared_path),'sha256':psha,'raw_root':spec['raw_root']}
            result['train_files']=[stat_ref({'path':str(Path(spec['raw_root'])/r['path']),'sha256':r['sha256']}) for r in records]
    result['read_artifacts_sha256']={**READS,**archive.READS}
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan'); p.add_argument('--plan-sha256'); p.add_argument('--preflight',action='store_true'); p.add_argument('--output')
    args=p.parse_args()
    if not args.plan:
        if not args.preflight: p.error('--plan required for execution')
        print(json.dumps(collect_inventory(),sort_keys=True,allow_nan=False));return
    plan=read_json({'path':args.plan,'sha256':args.plan_sha256})
    if args.preflight: print(json.dumps(collect_inventory(plan),sort_keys=True,allow_nan=False))
    else:
        if not args.output: p.error('--output required for execution')
        run(plan,args.output); print(json.dumps({'status':'complete','output':args.output}))

if __name__=='__main__': main()
