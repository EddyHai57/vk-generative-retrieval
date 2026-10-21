"""G2 shared train-only slice statistics and collision-corrected item evaluation."""
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from pathlib import Path
from .data_audit import canonical_bytes
from .generative_data import _checked_file,_json,_path
from .baselines import metrics


ITEM_GROUPS=('unsupported0','positive_le_median','positive_gt_median_le_p90','positive_gt_p90')
USER_GROUPS=('1','2-5','6-20','gt20')


def oracle_rank(score,codec):
    """Exhaustive four-position oracle, restricted to a tiny synthetic catalog."""
    import torch
    from .generative_data import _check_codec
    _check_codec(codec)
    ids=codec.catalog.ids
    if not 1<=len(ids)<=64:raise ValueError('bounded synthetic oracle only')
    ranked=[]
    for item in ids:
        path=tuple(codec.catalog.encode(int(item)));total=0.
        for position in range(4):
            logits=score([path[:position]])
            if logits.shape!=(1,codec.vocab_size) or not torch.isfinite(logits).all():raise ValueError('oracle logits shape/finite gate')
            total+=float(torch.log_softmax(logits,dim=-1)[0,codec.offsets[position]+path[position]])
        ranked.append((int(item),total))
    ranked.sort(key=lambda pair:(-pair[1],pair[0]))
    return [item for item,_ in ranked]


@dataclass(frozen=True)
class SliceStatistics:
    thresholds: tuple[int,int]
    item_counts: MappingProxyType
    user_counts: MappingProxyType
    manifest_sha256: str
    statistics_sha256: str
    binding_sha256: str


def _counts(value,positive=False):
    if type(value) is not dict:raise ValueError('explicit count dictionary required')
    result={}
    for key,count in value.items():
        if type(key) is not str or not key.isdecimal() or str(int(key))!=key or not 0<=int(key)<2**32 or type(count) is not int or count<(1 if positive else 0):raise ValueError('typed canonical raw ID/count required')
        result[int(key)]=count
    return result


def _query_statistics(queries,stats,catalog):
    for query in queries:
        if type(query) is not dict or type(query.get('user_id')) is not int or query['user_id'] not in stats.user_counts or type(query.get('target')) is not int or type(query.get('oov')) is not bool or query['oov']!=(query['target'] not in catalog):raise ValueError('unknown query statistics or OOV identity')


def read_slice_statistics(manifest_path,expected_sha256,expected_binding,catalog,queries):
    path=_checked_file(manifest_path,expected_sha256)
    if path.name!='slice-statistics-manifest.json':raise ValueError('fixed statistics manifest role')
    manifest=_json(path)
    keys={'schema_version','task_id','kind','binding','statistics_ref','statistics_sha','threshold_definition','thresholds','item_groups','user_groups','holdout_used'}
    if type(manifest) is not dict or set(manifest)!=keys or type(manifest['schema_version']) is not int or manifest['schema_version']!=1 or manifest['task_id']!='P4E' or manifest['kind']!='slice_statistics' or type(manifest['holdout_used']) is not bool or manifest['holdout_used']:
        raise ValueError('train-only statistics manifest schema')
    binding=manifest['binding']
    if type(binding) is not dict or set(binding)!={'raw_records','raw_producer','prepared_sha','catalog_sha','consumer'} or canonical_bytes(binding)!=canonical_bytes(expected_binding):raise ValueError('statistics producer/raw/prepared/catalog/source binding')
    if type(binding['raw_records']) is not list or not binding['raw_records'] or type(binding['raw_producer']) is not dict or type(binding['consumer']) is not dict:raise ValueError('statistics lineage schema')
    roles=set()
    for row in binding['raw_records']:
        if type(row) is not dict or set(row)!={'role','sha256','size_bytes'} or type(row['role']) is not str or not row['role'] or row['role'] in roles or type(row['size_bytes']) is not int or row['size_bytes']<1 or type(row['sha256']) is not str or len(row['sha256'])!=64 or any(x not in '0123456789abcdef' for x in row['sha256']):raise ValueError('statistics raw record identity')
        roles.add(row['role'])
    items=list(map(int,catalog))
    if not items or items!=sorted(set(items)) or binding['catalog_sha']!=hashlib.sha256(canonical_bytes(items)).hexdigest():raise ValueError('statistics actual catalog identity')
    if manifest['statistics_ref']!='train-slices.json' or manifest['threshold_definition']!='nearest_rank_positive_item_event_counts_median_p90' or canonical_bytes(manifest['item_groups'])!=canonical_bytes(list(ITEM_GROUPS)) or canonical_bytes(manifest['user_groups'])!=canonical_bytes(list(USER_GROUPS)):
        raise ValueError('fixed statistics role/threshold/group protocol')
    data=_json(_checked_file(_path(path.parent,manifest['statistics_ref']),manifest['statistics_sha']))
    if type(data) is not dict or set(data)!={'schema_version','popularity_thresholds','item_like_counts','user_like_counts','user_buckets','definition'} or type(data['schema_version']) is not int or data['schema_version']!=1 or canonical_bytes(data['user_buckets'])!=canonical_bytes([1,5,20]) or data['definition']!='train_like_counts_only_no_holdout_boundaries':raise ValueError('statistics schema/definition')
    item_counts=_counts(data['item_like_counts']);user_counts=_counts(data['user_like_counts'],positive=True)
    if set(item_counts)!=set(items):raise ValueError('complete catalog counts required; unknown is not zero')
    positive=sorted(value for value in item_counts.values() if value>0)
    if not positive or not user_counts:raise ValueError('positive train statistics required')
    thresholds={'median':positive[math.ceil(len(positive)*.5)-1],'p90':positive[math.ceil(len(positive)*.9)-1]}
    if canonical_bytes(data['popularity_thresholds'])!=canonical_bytes(thresholds) or canonical_bytes(manifest['thresholds'])!=canonical_bytes(thresholds):raise ValueError('nearest-rank train threshold mismatch')
    result=SliceStatistics((thresholds['median'],thresholds['p90']),MappingProxyType(item_counts),MappingProxyType(user_counts),expected_sha256,manifest['statistics_sha'],hashlib.sha256(canonical_bytes(binding)).hexdigest())
    _query_statistics(queries,result,set(items))
    return result


def aggregate_items(predictions,queries,catalog,*,slice_definition):
    if type(slice_definition) is not SliceStatistics:raise ValueError('loaded slice statistics required')
    stats=slice_definition;items=set(map(int,catalog))
    if set(stats.item_counts)!=items:raise ValueError('slice/catalog mismatch')
    _query_statistics(queries,stats,items)
    base=metrics(predictions,queries,catalog,(10,20,50))
    empty=metrics([],[],catalog,(10,20,50))['all']
    def grouped(assignments,names):
        result={}
        for name in names:
            indices=[i for i,label in enumerate(assignments) if label==name]
            result[name]=metrics([predictions[i] for i in indices],[queries[i] for i in indices],catalog,(10,20,50))['all'] if indices else dict(empty)
        return result
    popularity=[];users=[];support=[];history=[]
    for query in queries:
        if query['oov']:popularity.append('oov');support.append('oov')
        else:
            count=stats.item_counts[query['target']]
            popularity.append(ITEM_GROUPS[0] if count==0 else ITEM_GROUPS[1] if count<=stats.thresholds[0] else ITEM_GROUPS[2] if count<=stats.thresholds[1] else ITEM_GROUPS[3])
            support.append('supported' if count>0 else 'unsupported')
        count=stats.user_counts[query['user_id']]
        users.append(USER_GROUPS[0] if count==1 else USER_GROUPS[1] if count<=5 else USER_GROUPS[2] if count<=20 else USER_GROUPS[3])
        length=len(query['history'])
        if not 1<=length<=50:raise ValueError('bounded eligible history required')
        history.append(str(length))
    unique=set(int(item) for row in predictions for item in row)
    sizes=[len(row) for row in predictions]
    return {'metrics':{'all_eligible':base['all'],'in_catalog':base['train_observed'],'oov':base['oov']},
            'popularity':grouped(popularity,ITEM_GROUPS+('oov',)),
            'user_history':grouped(users,USER_GROUPS),'support':grouped(support,('supported','unsupported','oov')),
            'truncated_history':grouped(history,tuple(str(i) for i in range(1,51))),
            'coverage':len(unique)/len(items),'unique_candidate_count':len(unique),
            'candidate_counts':{'count':len(sizes),'sum':sum(sizes),'min':min(sizes) if sizes else None,'max':max(sizes) if sizes else None},
            'slice_statistics_manifest_sha256':stats.manifest_sha256}


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        tensor=value.detach().cpu().contiguous()
        digest.update(canonical_bytes([name,str(tensor.dtype),list(tensor.shape)]));digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _evaluation_binding(model,context,checkpoint_sha,beam,shard_size):
    from .generative_experiment import RunContext
    if type(context) is not RunContext:raise ValueError('factory-validated context required')
    context.check()
    if type(checkpoint_sha) is not str or len(checkpoint_sha)!=64 or any(c not in '0123456789abcdef' for c in checkpoint_sha):raise ValueError('external checkpoint identity')
    if type(beam) is not int or beam!=100 or type(shard_size) is not int or not 1<=shard_size<=64:raise ValueError('frozen beam/bounded shard size')
    return {'context':context.binding,'checkpoint_sha256':checkpoint_sha,'model_sha256':model_digest(model),'beam':beam,'shard_size':shard_size}


def _read_evaluation(path,pin,binding,queries,catalog):
    value=_json(_checked_file(path,pin))
    keys={'schema_version','task_id','kind','binding','query_count','completed_queries','complete','shards','computation_sha256','aggregate','observations'}
    if type(value) is not dict or set(value)!=keys or value['schema_version']!=1 or type(value['schema_version']) is not int or value['task_id']!='G2' or value['kind']!='query_evaluation' or canonical_bytes(value['binding'])!=canonical_bytes(binding) or type(value['query_count']) is not int or value['query_count']!=len(queries) or type(value['complete']) is not bool or type(value['shards']) is not list:raise ValueError('evaluation manifest schema/binding')
    rows=[];offset=0;items=set(catalog)
    for entry in value['shards']:
        if type(entry) is not dict or set(entry)!={'start','end','path','sha256'} or type(entry['start']) is not int or type(entry['end']) is not int or entry['start']!=offset or entry['end']!=min(offset+binding['shard_size'],len(queries)):raise ValueError('query shard gap/overlap/range')
        shard_path=Path(entry['path'])
        if shard_path.name!=f"shard-{entry['start']:06d}-{entry['end']:06d}.json":raise ValueError('fixed shard role')
        shard=_json(_checked_file(shard_path,entry['sha256']))
        if type(shard) is not dict or set(shard)!={'schema_version','binding','start','end','rows'} or shard['schema_version']!=1 or canonical_bytes(shard['binding'])!=canonical_bytes(binding) or shard['start']!=entry['start'] or shard['end']!=entry['end'] or type(shard['rows']) is not list or len(shard['rows'])!=entry['end']-entry['start']:raise ValueError('query shard schema/identity')
        for ordinal,row in enumerate(shard['rows'],offset):
            if type(row) is not dict or set(row)!={'ordinal','query_sha256','items','audit','latency_seconds'} or type(row['ordinal']) is not int or row['ordinal']!=ordinal or row['query_sha256']!=hashlib.sha256(canonical_bytes(queries[ordinal])).hexdigest():raise ValueError('query ordinal/identity')
            preds=row['items'];audit=row['audit'];latency=row['latency_seconds']
            if type(preds) is not list or not 1<=len(preds)<=100 or any(type(i) is not int or i not in items for i in preds) or len(set(preds))!=len(preds):raise ValueError('candidate legality/unique gate')
            if type(audit) is not dict or set(audit)!={'invalid','deadend','collision','max_active'} or any(type(v) is not int for v in audit.values()) or any(audit[k]!=0 for k in ('invalid','deadend','collision')) or not 1<=audit['max_active']<=100:raise ValueError('decode audit gate')
            if type(latency) not in (int,float) or not math.isfinite(latency) or latency<0:raise ValueError('decode latency observation')
            rows.append(row)
        offset=entry['end']
    if type(value['completed_queries']) is not int or offset!=value['completed_queries'] or value['complete']!=(offset==len(queries)):raise ValueError('full query coverage gate')
    return value,rows


def _computation(rows,aggregate):
    return hashlib.sha256(canonical_bytes({'rows':[{k:v for k,v in row.items() if k!='latency_seconds'} for row in rows],'aggregate':aggregate})).hexdigest()


def evaluate_queries(model,context,*,output,checkpoint_sha256,beam=100,shard_size=64,
                     resume_manifest=None,resume_sha256=None,stop_after_shards=None,should_stop=None,budget_check=None):
    """One query at a time; immutable complete shards are the recovery boundary."""
    import time,sys,resource
    import numpy as np
    import torch
    from .tiger import decode_history
    from .generative_experiment import _write_json
    from .generative_data import _no_symlink
    binding=_evaluation_binding(model,context,checkpoint_sha256,beam,shard_size)
    if budget_check is not None:budget_check()
    if (resume_manifest is None)!=(resume_sha256 is None):raise ValueError('paired external resume manifest pin required')
    if stop_after_shards is not None and (type(stop_after_shards) is not int or stop_after_shards<1 or context.binding['synthetic'] is not True):raise ValueError('synthetic bounded interruption only')
    queries=context.queries;rows=[];shards=[]
    if resume_manifest is not None:
        parent,rows=_read_evaluation(resume_manifest,resume_sha256,binding,queries,context.catalog);shards=parent['shards']
        prior_aggregate=aggregate_items([row['items'] for row in rows],queries,context.catalog,slice_definition=context.slice_definition) if parent['complete'] else None
        if canonical_bytes(parent['aggregate'])!=canonical_bytes(prior_aggregate) or parent['computation_sha256']!=_computation(rows,prior_aggregate):raise ValueError('parent evaluation aggregate/computation')
    output=_no_symlink(output);output.mkdir(mode=0o700,exist_ok=False)
    device=next(model.parameters()).device;written=0
    for start in range(len(rows),len(queries),shard_size):
        end=min(start+shard_size,len(queries));new=[]
        for ordinal in range(start,end):
            if budget_check is not None:budget_check()
            if device.type=='cuda':torch.cuda.synchronize(device)
            begin=time.perf_counter();preds,audit=decode_history(model,context.codec,queries[ordinal]['history'],beam)
            if device.type=='cuda':torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-begin
            if budget_check is not None:budget_check()
            if not preds or len(preds)!=len(set(preds)) or any(i not in context.catalog for i in preds) or any(audit[k]!=0 for k in ('invalid','deadend','collision')):raise ValueError('decoded candidate/audit gate')
            new.append({'ordinal':ordinal,'query_sha256':hashlib.sha256(canonical_bytes(queries[ordinal])).hexdigest(),'items':preds,'audit':audit,'latency_seconds':elapsed})
        shard={'schema_version':1,'binding':binding,'start':start,'end':end,'rows':new}
        path=output/f'shard-{start:06d}-{end:06d}.json';pin=_write_json(path,shard)
        shards.append({'start':start,'end':end,'path':str(path.absolute()),'sha256':pin});rows.extend(new);written+=1
        # Publish an immutable receipt after each complete shard, so signals or
        # process failure never make an incomplete shard authoritative.
        progress={'schema_version':1,'task_id':'G2','kind':'query_evaluation','binding':binding,'query_count':len(queries),'completed_queries':len(rows),'complete':len(rows)==len(queries),'shards':list(shards),'computation_sha256':_computation(rows,None),'aggregate':None,'observations':None}
        if not progress['complete']:_write_json(output/f'progress-{len(rows):06d}.json',progress)
        if (stop_after_shards is not None and written>=stop_after_shards) or (should_stop is not None and should_stop()):break
    complete=len(rows)==len(queries)
    aggregate=aggregate_items([r['items'] for r in rows],queries,context.catalog,slice_definition=context.slice_definition) if complete else None
    latency=[row['latency_seconds'] for row in rows]
    observations={'latency':{'count':len(latency),'p50_seconds':float(np.quantile(latency,.5)) if latency else None,'p95_seconds':float(np.quantile(latency,.95)) if latency else None,'definition':'per-query tokenization+encoder+four-position beam; linear quantile'},
                  'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'rss_definition':'process lifetime observed high-water RSS; not evaluation-only peak',
                  'vram':'not_applicable' if device.type=='cpu' else {'allocated_bytes':torch.cuda.memory_allocated(device),'definition':'observed at evaluation completion; not peak'}}
    manifest={'schema_version':1,'task_id':'G2','kind':'query_evaluation','binding':binding,'query_count':len(queries),'completed_queries':len(rows),'complete':complete,'shards':shards,'computation_sha256':_computation(rows,aggregate),'aggregate':aggregate,'observations':observations}
    path=output/'manifest.json';pin=_write_json(path,manifest)
    # Fresh read-back checks the exact published artifacts, not only exit status.
    _read_evaluation(path,pin,binding,queries,context.catalog)
    return {'manifest':manifest,'manifest_path':path,'manifest_sha256':pin,'aggregate':aggregate,'observations':observations}
