"""Original P4 exact retrieval: float32 features, fixed float64 pair reduction.

Both scorers sum dimensions using non-BLAS einsum (optimize=False). Scores
therefore do not depend on query/item block shape; ties use item ID ascending.
No epsilon-based score bucketing or approximate candidate filtering.
"""
import numpy as np

def _inputs(queries, items, ids, k):
    queries=np.asarray(queries,dtype=np.float32);items=np.asarray(items,dtype=np.float32);ids=np.asarray(ids)
    if queries.ndim!=2 or items.ndim!=2 or queries.shape[1]!=items.shape[1] or items.shape[0]!=len(ids) or ids.ndim!=1 or len(ids)==0:
        raise ValueError('invalid score shapes')
    if not np.isfinite(queries).all() or not np.isfinite(items).all() or len(np.unique(ids))!=len(ids) or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0):
        raise ValueError('invalid scores or IDs')
    if type(k) is not int or k<1:raise ValueError('invalid k')
    return queries,items,ids,min(k,len(ids))

def _best(scores, ids, k):
    return np.lexsort((ids,-scores))[:k]

def brute_topk(queries, items, ids, k):
    queries,items,ids,k=_inputs(queries,items,ids,k)
    # Toy-only reference deliberately materializes scores; production uses blocks.
    scores=np.einsum('qd,id->qi',queries,items,dtype=np.float64,optimize=False)
    return np.array([ids[_best(row,ids,k)] for row in scores],dtype=ids.dtype).reshape(len(queries),k)

def blockwise_topk(queries, items, ids, k, query_block=32, item_block=4096):
    queries,items,ids,k=_inputs(queries,items,ids,k)
    if any(type(x) is not int or x<1 for x in (query_block,item_block)):raise ValueError('invalid blocks')
    out=np.empty((len(queries),k),dtype=ids.dtype)
    for start in range(0,len(queries),query_block):
        q=queries[start:start+query_block];best_ids=[np.empty(0,dtype=ids.dtype) for _ in q];best_scores=[np.empty(0,dtype=np.float64) for _ in q]
        for offset in range(0,len(items),item_block):
            scores=np.einsum('qd,id->qi',q,items[offset:offset+item_block],dtype=np.float64,optimize=False);block_ids=ids[offset:offset+item_block]
            for row in range(len(q)):
                combined_ids=np.concatenate((best_ids[row],block_ids));combined_scores=np.concatenate((best_scores[row],scores[row]))
                take=_best(combined_scores,combined_ids,k);best_ids[row]=combined_ids[take];best_scores[row]=combined_scores[take]
        out[start:start+len(q)]=np.array(best_ids)
    return out

def normalize(rows):
    rows=np.asarray(rows,dtype=np.float32)
    if rows.ndim!=2 or not np.isfinite(rows).all():raise ValueError('missing/nonfinite feature')
    norms=np.linalg.norm(rows,axis=1,keepdims=True)
    return np.divide(rows,norms,out=np.zeros_like(rows),where=norms!=0)

def join_normalized(catalog, feature_ids, embeddings):
    catalog=np.asarray(catalog);feature_ids=np.asarray(feature_ids);embeddings=np.asarray(embeddings)
    if feature_ids.ndim!=1 or embeddings.ndim!=2 or len(feature_ids)!=len(embeddings) or len(np.unique(feature_ids))!=len(feature_ids) or len(np.unique(catalog))!=len(catalog):raise ValueError('invalid feature IDs')
    order=np.argsort(feature_ids);positions=np.searchsorted(feature_ids[order],catalog)
    if np.any(positions>=len(feature_ids)) or not np.array_equal(feature_ids[order][positions],catalog):raise ValueError('missing catalog feature')
    return normalize(embeddings[order[positions]])

def history_centroids(histories,catalog,features):
    lookup={int(item):i for i,item in enumerate(catalog)};queries=[]
    for history in histories:
        if not history or any(item not in lookup for item in history):raise ValueError('missing history feature')
        queries.append(np.mean(features[[lookup[item] for item in history]],axis=0,dtype=np.float32))
    if not queries:return np.empty((0,features.shape[1]),dtype=np.float32),0
    rows=np.asarray(queries);zero=int(np.count_nonzero(np.linalg.norm(rows,axis=1)==0))
    return normalize(rows),zero

def popularity(catalog,events):
    lookup={int(item):i for i,item in enumerate(catalog)};counts=np.zeros(len(catalog),dtype=np.int64)
    for item,liked in events:
        if item not in lookup or type(liked) is not bool:raise ValueError('invalid train event')
        if liked:counts[lookup[item]]+=1
    return counts

def sample_queries(queries,split,seed=42,limit=512):
    import hashlib
    if split not in ('validation','test') or type(seed) is not int or seed<0 or type(limit) is not int or limit<1:raise ValueError('invalid sample config')
    users=[q['user_id'] for q in queries]
    if any(type(x) is not int or x<0 for x in users) or len(set(users))!=len(users):raise ValueError('invalid/duplicate query users')
    return sorted(queries,key=lambda q:(hashlib.sha256(f"{seed}:{split}:{q['user_id']}".encode()).hexdigest(),q['user_id']))[:limit]

def metrics(predictions,queries,catalog,ks=(10,20)):
    import math
    catalog=set(map(int,catalog))
    if len(predictions)!=len(queries) or any(type(k) is not int or k<1 for k in ks):raise ValueError('invalid metric shapes')
    accum={s:{'count':0,**{f'{m}@{k}':0.0 for k in ks for m in ('recall','ndcg')}} for s in ('all','train_observed','oov')}
    for predicted,query in zip(predictions,queries):
        predicted=list(map(int,predicted));target=query['target'];oov=target not in catalog
        if type(query['oov']) is not bool or query['oov']!=oov or len(set(predicted))!=len(predicted) or any(x not in catalog for x in predicted):raise ValueError('invalid prediction/OOV')
        for slice_name in ('all','oov' if oov else 'train_observed'):
            result=accum[slice_name];result['count']+=1
            for k in ks:
                if target in predicted[:k]:
                    rank=predicted.index(target)+1;result[f'recall@{k}']+=1;result[f'ndcg@{k}']+=1/math.log2(rank+1)
    for result in accum.values():
        for name in result:
            if name!='count':result[name]=result[name]/result['count'] if result['count'] else None
    return accum
