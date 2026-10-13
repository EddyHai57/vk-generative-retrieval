"""P4C fixed traditional baselines; train graph includes ALL like events."""
import math
from collections import Counter
from .retrieval_data import catalog_map

def train_graph(catalog, events):
    lookup=catalog_map(catalog);counts=[0]*len(catalog);users={}
    for user,item,liked in events:
        if type(user)is not int or not 0<=user<2**32 or type(item)is not int or item not in lookup or type(liked)is not bool:
            raise ValueError('invalid train graph event')
        if liked:
            counts[lookup[item]-1]+=1;users.setdefault(user,set()).add(item)
    users={u:sorted(items) for u,items in sorted(users.items())}
    inverted={item:[] for item in catalog}
    for user,items in users.items():
        for item in items:inverted[item].append(user)
    return {'catalog':list(catalog),'popularity':counts,'users':users,'inverted':inverted,'degrees':[len(inverted[i]) for i in catalog]}

def pop_rank(graph):
    return [item for item,_ in sorted(zip(graph['catalog'],graph['popularity']),key=lambda x:(-x[1],x[0]))]

def item_neighbors(graph, limit=200, items=None, check=None):
    """One sparse cooccurrence row at a time, never catalog-square storage."""
    if type(limit)is not int or limit<1:raise ValueError('invalid neighbor limit')
    catalog=graph['catalog'];degree=dict(zip(catalog,graph['degrees']));result={}
    for offset,item in enumerate(catalog if items is None else items):
        if item not in degree:raise ValueError('unknown graph item')
        if check is not None and offset%64==0:check()
        pairs=Counter()
        for user in graph['inverted'][item]:pairs.update(graph['users'][user])
        pairs.pop(item,None)
        row=[(other,count/math.sqrt(degree[item]*degree[other])) for other,count in pairs.items()]
        result[item]=sorted(row,key=lambda x:(-x[1],x[0]))[:limit]
    return result

def itemcf_scores(history, neighbors):
    scores={}
    if not history:raise ValueError('empty history')
    for item in history:
        if item not in neighbors:raise ValueError('unknown history item')
        for other,weight in neighbors[item]:
            if not math.isfinite(weight) or weight<=0:raise ValueError('invalid similarity')
            scores[other]=scores.get(other,0.)+float(weight)
            if not math.isfinite(scores[other]):raise ValueError('nonfinite ItemCF score')
    return scores

def itemcf_rank(history, neighbors, popularity, k):
    if type(k)is not int or k<1 or len(set(popularity))!=len(popularity):raise ValueError('invalid ranking request')
    scores=itemcf_scores(history,neighbors)
    chosen=sorted(scores,key=lambda i:(-scores[i],i))[:k]
    fill=0;seen=set(chosen)
    for item in popularity:
        if len(chosen)>=min(k,len(popularity)):break
        if item not in seen:chosen.append(item);seen.add(item);fill+=1
    return chosen,fill

def evaluate_baselines(graph, neighbors, features, queries, ks=(10,20,50), query_block=32, item_block=4096, check=None):
    import numpy as np,time
    from .baselines import history_centroids,blockwise_topk,metrics
    from .baseline_run import evaluate_predictions
    catalog=graph['catalog'];k=min(max(ks),len(catalog));pop=pop_rank(graph);pred={m:[] for m in ('popularity','itemcf','content')};latency={m:[] for m in pred};fills=[];zeros=0
    if features.dtype!=np.float32 or features.shape!=(len(catalog),64) or not np.isfinite(features).all():raise ValueError('invalid content')
    if set(neighbors)!=set(catalog):raise ValueError('incomplete graph')
    for start in range(0,len(queries),query_block):
        if check is not None:check()
        rows=queries[start:start+query_block]
        for q in rows:
            if not q['history'] or any(i not in neighbors for i in q['history']):raise ValueError('invalid query history')
            t=time.perf_counter();pred['popularity'].append(pop[:k]);latency['popularity'].append(time.perf_counter()-t)
            t=time.perf_counter();rank,fill=itemcf_rank(q['history'],neighbors,pop,k);pred['itemcf'].append(rank);fills.append(fill);latency['itemcf'].append(time.perf_counter()-t)
        t=time.perf_counter();centroids,zero=history_centroids([q['history'] for q in rows],catalog,features);zeros+=zero
        pred['content'].extend(blockwise_topk(centroids,features,catalog,k,query_block,item_block).tolist());seconds=time.perf_counter()-t
        latency['content'].extend([seconds/len(rows)]*len(rows))
    result={m:evaluate_predictions(p,queries,catalog,ks) for m,p in pred.items()}
    diagnostics={'zero_centroids':zeros,'coverage':{},'latency':{},'itemcf_fill':{}}
    for m,p in pred.items():
        diagnostics['coverage'][m]={str(cut):len({i for row in p for i in row[:cut]})/len(catalog) for cut in ks}
        values=latency[m];diagnostics['latency'][m]={'scope':{'popularity':'per-query shortlist copy only; train counting and shared Pop sort excluded','itemcf':'per-query history score sum, sort and Pop fill; graph fit and shared Pop sort excluded','content':'batch-amortized normalized history mean, fixed-pair exact scoring and ranking; input load excluded'}[m],'seconds':sum(values),'p50_ms':float(np.percentile(values,50)*1000) if values else None,'p95_ms':float(np.percentile(values,95)*1000) if values else None}
    for cut in ks:
        actual=min(cut,len(catalog));counts=[max(0,actual-(k-fill)) for fill in fills]
        diagnostics['itemcf_fill'][str(cut)]={'slots':sum(counts),'slot_rate':sum(counts)/(len(queries)*actual) if queries else None,'queries':sum(c>0 for c in counts),'query_rate':sum(c>0 for c in counts)/len(queries) if queries else None}
    result['diagnostics']=diagnostics
    return result,pred

def _digest(value):
    import hashlib
    from .data_audit import canonical_bytes
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def write_run(output,graph,neighbors,features,queries,binding,ks=(10,20,50),check=None):
    import time,resource
    from pathlib import Path
    from .data_audit import canonical_bytes,file_sha256
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result,pred=evaluate_baselines(graph,neighbors,features,queries,ks=ks,check=check)
    graphproof={'catalog':graph['catalog'],'popularity':graph['popularity'],'degrees':graph['degrees'],'neighbors':{str(i):row for i,row in neighbors.items()}}
    for name,value in (('metrics.json',result),('predictions.json',pred),('graph.json',graphproof)):
        with (output/name).open('xb') as stream:stream.write(canonical_bytes(value))
    value={'schema_version':1,'task_id':'P4C','kind':'baselines','status':'success','binding':binding,'ks':list(ks),'query_identity_sha256':_digest(queries),'catalog_identity_sha256':_digest(graph['catalog']),'queries':len(queries),'catalog_items':len(graph['catalog']),'artifacts':{name:file_sha256(output/name) for name in ('metrics.json','predictions.json','graph.json')},'elapsed_seconds':time.monotonic()-started,'peak_rss_native_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_scope':'whole process high-water mark including input audit, graph fit and evaluation; KiB Linux, bytes macOS'}
    with (output/'baseline-manifest.json').open('xb') as stream:stream.write(canonical_bytes(value))
    return value

def verify_run(output,queries,catalog,binding,graph,features,check=None):
    from pathlib import Path
    from .manifest import read_json
    from .retrieval_data import checked
    from .baseline_run import evaluate_predictions
    from .data_audit import canonical_bytes
    output=Path(output);value=read_json(output/'baseline-manifest.json')
    if value.get('schema_version')!=1 or value.get('task_id')!='P4C' or value.get('kind')!='baselines' or value.get('status')!='success' or value['binding']!=binding or value['query_identity_sha256']!=_digest(queries) or value['catalog_identity_sha256']!=_digest(catalog) or value['queries']!=len(queries) or value['catalog_items']!=len(catalog):raise ValueError('run input binding mismatch')
    if set(value['artifacts'])!={'metrics.json','predictions.json','graph.json'}:raise ValueError('run artifact schema mismatch')
    for name,digest in value['artifacts'].items():checked(output/name,digest)
    pred=read_json(output/'predictions.json');result=read_json(output/'metrics.json');proof=read_json(output/'graph.json')
    if set(pred)!={'popularity','itemcf','content'} or proof['catalog']!=catalog or len(proof['degrees'])!=len(catalog) or len(proof['popularity'])!=len(catalog):raise ValueError('graph proof mismatch')
    rebuilt=item_neighbors(graph,200,check=check)
    expected_graph={'catalog':graph['catalog'],'popularity':graph['popularity'],'degrees':graph['degrees'],'neighbors':{str(i):row for i,row in rebuilt.items()}}
    if canonical_bytes(proof)!=canonical_bytes(expected_graph):raise ValueError('graph disagrees with pinned train incidence')
    fresh_result,fresh_pred=evaluate_baselines(graph,rebuilt,features,queries,ks=tuple(value['ks']),check=check)
    if canonical_bytes(pred)!=canonical_bytes(fresh_pred):raise ValueError('predictions disagree with pinned input replay')
    for key in ('zero_centroids','coverage','itemcf_fill'):
        if canonical_bytes(result['diagnostics'][key])!=canonical_bytes(fresh_result['diagnostics'][key]):raise ValueError('deterministic diagnostics mismatch')
    neighbors={int(i):[(int(j),float(w)) for j,w in row] for i,row in proof['neighbors'].items()}
    if set(neighbors)!=set(catalog):raise ValueError('incomplete neighbors')
    for i,row in neighbors.items():
        if len(row)>200 or len({j for j,w in row})!=len(row) or row!=sorted(row,key=lambda x:(-x[1],x[0])) or any(j not in neighbors or j==i or not math.isfinite(w) or not 0<w<=1 for j,w in row):raise ValueError('invalid neighbor proof')
    pop=pop_rank(proof);k=min(max(value['ks']),len(catalog))
    for index,q in enumerate(queries):
        if pred['popularity'][index]!=pop[:k] or pred['itemcf'][index]!=itemcf_rank(q['history'],neighbors,pop,k)[0]:raise ValueError('ranking replay mismatch')
    for m,p in pred.items():
        if canonical_bytes(evaluate_predictions(p,queries,catalog,value['ks']))!=canonical_bytes(result[m]):raise ValueError('metric proof mismatch')
    return value

def graph_pilot(graph,check=None):
    import time
    lengths={user:len(items) for user,items in graph['users'].items()}
    work={item:sum(lengths[u] for u in graph['inverted'][item]) for item in graph['catalog']}
    supported=[i for i in graph['catalog'] if work[i]]
    heavy=sorted(supported,key=lambda i:(-work[i],i))[:64]
    stride=max(1,len(supported)//64);sample=supported[::stride][:64]
    selected=sorted(set(heavy+sample));start=time.monotonic();rows=item_neighbors(graph,200,selected,check)
    return {'binary_edges':sum(graph['degrees']),'cooccurrence_updates_including_self':sum(v*v for v in lengths.values()),'max_user_degree':max(lengths.values(),default=0),'max_item_degree':max(graph['degrees'],default=0),'max_row_pair_updates_including_self':max(work.values(),default=0),'pilot_items':len(selected),'pilot_pair_updates_including_self':sum(work[i] for i in selected),'pilot_seconds':time.monotonic()-start,'pilot_retained_edges':sum(map(len,rows.values())),'selection':'64 largest pair-work rows plus up to64 evenly spaced supported item IDs'}
