"""Original capacity-constrained greedy balanced hierarchical k-means.

Canonical ID order, fixed dimension reduction, Euclidean mean centres.
Greedy capacity assignment is deterministic; it is not globally optimal.
"""
import numpy as np


def _distance(rows, centers):
    distances=np.zeros((len(rows),len(centers)),dtype=np.float64)
    for dimension in range(rows.shape[1]):
        delta=rows[:,dimension,None].astype(np.float64)-centers[None,:,dimension]
        distances+=delta*delta
    return distances


def _assign(distances, ids):
    n,k=distances.shape
    capacity=np.full(k,n//k,dtype=np.int64);capacity[:n%k]+=1
    row=np.repeat(np.arange(n),k);child=np.tile(np.arange(k),n)
    order=np.lexsort((child,ids[row],distances.ravel()))
    labels=np.full(n,-1,dtype=np.int64)
    remaining=n
    for edge in order:
        i=row[edge];j=child[edge]
        if labels[i]<0 and capacity[j]>0:
            labels[i]=j;capacity[j]-=1;remaining-=1
            if not remaining:break
    return labels


class SIDCatalog:
    def __init__(self,ids,sids,branching,depth,nodes):
        self.ids=ids;self.sids=sids;self.branching=branching;self.depth=depth;self.nodes=nodes
        self.suffix_vocabulary=int(sids[:,-1].max())+1
        self.bases=[branching]*depth+[self.suffix_vocabulary]
        keys=np.zeros(len(ids),dtype=np.uint64)
        for position,base in enumerate(self.bases):keys=keys*base+sids[:,position].astype(np.uint64)
        order=np.argsort(keys,kind='stable');self.keys=keys[order];self.key_items=ids[order]

    def _range(self,prefix):
        if not isinstance(prefix,(tuple,list)) or len(prefix)>self.depth+1:raise ValueError('invalid prefix length')
        low=0
        for position,value in enumerate(prefix):
            if type(value) is not int or not 0<=value<self.bases[position]:raise ValueError('invalid token')
            low=low*self.bases[position]+value
        width=1
        for base in self.bases[len(prefix):]:width*=base
        start=int(np.searchsorted(self.keys,low*width));end=int(np.searchsorted(self.keys,(low+1)*width))
        if start==end:raise ValueError('unreachable prefix')
        return start,end,width

    def encode(self,item):
        if type(item) is not int or item<0:raise ValueError('invalid item')
        i=int(np.searchsorted(self.ids,item))
        if i>=len(self.ids) or int(self.ids[i])!=item:raise ValueError('OOV item')
        return tuple(map(int,self.sids[i]))

    def decode(self,path):
        if not isinstance(path,(tuple,list)) or len(path)!=self.depth+1:raise ValueError('full SID required')
        start,end,_=self._range(path)
        if end-start!=1:raise ValueError('SID collision')
        return int(self.key_items[start])

    def allowed_next(self,prefix):
        start,end,width=self._range(prefix)
        if len(prefix)==self.depth+1:return []
        base=self.bases[len(prefix)];stride=width//base
        return np.unique((self.keys[start:end]//stride)%base).astype(int).tolist()

    def audit(self):
        import math
        leaves,counts=np.unique(self.sids[:,:self.depth],axis=0,return_counts=True)
        total=len(self.ids);p=counts/total;sorted_counts=np.sort(counts)
        prefixes={tuple(map(int,row[:length])) for row in self.sids for length in range(self.depth+1)}
        deadends=sum(not self.allowed_next(prefix) for prefix in prefixes)
        roundtrip=sum(self.decode(self.encode(int(item)))==int(item) for item in self.ids)
        return {'catalog_items':total,'full_sid_collisions':total-len(np.unique(self.keys)),
                'roundtrip_items':roundtrip,'reachable_items':roundtrip,'valid_prefix_deadends':deadends,
                'semantic_collision_groups':int(np.count_nonzero(counts>1)),
                'semantic_collision_items':int(counts[counts>1].sum()),'semantic_collision_excess':int((counts-1).sum()),
                'leaf_occupancy_histogram':{str(int(x)):int(np.count_nonzero(counts==x)) for x in np.unique(counts)},
                'max_leaf_occupancy':int(counts.max()),'occupied_leaves':len(leaves),
                'leaf_utilization':len(leaves)/(self.branching**self.depth),
                'leaf_entropy_bits':float(-np.sum(p*np.log2(p))),
                'leaf_gini':float((2*np.sum((np.arange(len(counts))+1)*sorted_counts)/(len(counts)*total))-(len(counts)+1)/len(counts)),
                'prefix_occupancy':{str(length):len(np.unique(self.sids[:,:length],axis=0)) for length in range(1,self.depth+1)},
                'suffix_vocabulary':self.suffix_vocabulary,'balanced_nodes':all(max(n['sizes'])-min(n['sizes'])<=1 for n in self.nodes),
                'nodes':len(self.nodes),'unconverged_nodes':sum(not n['converged'] for n in self.nodes),
                'constraint_bytes':self.keys.nbytes+self.key_items.nbytes,
                'tokens_by_position':[[f'p{position}:{value}' for value in range(base)] for position,base in enumerate(self.bases)]}


def fit(ids,features,branching=16,depth=3,seed=42,max_iter=10,on_boundary=None):
    ids=np.asarray(ids);features=np.asarray(features)
    if ids.ndim!=1 or not len(ids) or not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=2**32) or len(np.unique(ids))!=len(ids):raise ValueError('invalid catalog IDs')
    if features.dtype!=np.float32 or features.shape!=(len(ids),64) or not np.isfinite(features).all():raise ValueError('invalid normalized features')
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
            distances=_distance(rows,centers);labels=_assign(distances,ids[indices])
            centers=np.stack([rows[labels==j].mean(axis=0,dtype=np.float64) for j in range(k)])
            objectives.append(float(_distance(rows,centers)[np.arange(len(rows)),labels].sum(dtype=np.float64)))
            if previous is not None and np.array_equal(labels,previous):converged=True;break
            previous=labels.copy()
        sizes=np.bincount(labels,minlength=k).tolist()
        nodes.append({'prefix':list(prefix),'init_item_ids':ids[indices[init]].astype(int).tolist(),'centers':centers.tolist(),'sizes':sizes,'iterations':iteration+1,'converged':converged,'objectives':objectives})
        for j in range(k):
            selected=indices[labels==j];sids[selected,level]=j;visit(selected,prefix+(j,))
    visit(np.arange(len(ids)),())
    return SIDCatalog(ids,sids,branching,depth,nodes)


def collision_hits(predicted,target):
    """Explicit toy distinction; this is not an item ranking evaluator."""
    if not isinstance(predicted,(tuple,list)) or not isinstance(target,(tuple,list)) or len(predicted)!=4 or len(target)!=4 or any(type(x) is not int or x<0 for x in tuple(predicted)+tuple(target)):raise ValueError('invalid SID hit inputs')
    return {'semantic_hit':tuple(predicted[:3])==tuple(target[:3]),'item_hit':tuple(predicted)==tuple(target)}
