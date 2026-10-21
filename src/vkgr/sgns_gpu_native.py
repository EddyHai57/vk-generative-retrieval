"""Independent route B SGNS: frozen batch mathematics, device-resident data.

Conditional inverse CDF removes exactly the two forbidden probability intervals;
it has the same with-replacement distribution as rejection, a different RNG stream.
No dataset reader, model selection, checkpoint access or implicit writes.
"""
from dataclasses import asdict
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from .sid_coll_sgns import SGNSConfig,health_check,learning_rate,_array_digest,_sequence_digest
from .sid_tok_audit import feature_digest
from .sid_v2 import catalog_digest


def pack(catalog,histories,device):
    catalog=np.asarray(catalog)
    if catalog.ndim!=1 or catalog.dtype.kind not in 'iu' or len(catalog)<3 or (catalog<=0).any() or not np.all(catalog[1:]>catalog[:-1]):
        raise ValueError('unique positive ascending catalog with >=3 items required')
    for user,values in histories.items():
        if isinstance(user,(bool,np.bool_)) or not isinstance(user,(int,np.integer)):raise ValueError('integer users required')
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in values):raise ValueError('integer events required')
    lengths=np.asarray([len(histories[u]) for u in sorted(histories)],dtype=np.int64)
    events=np.concatenate([np.asarray(histories[u],dtype=np.int64) for u in sorted(histories)]) if lengths.sum() else np.empty(0,dtype=np.int64)
    rows=np.searchsorted(catalog,events)
    if np.any(rows>=len(catalog)) or not np.array_equal(catalog[rows],events):raise ValueError('PAD/OOV history forbidden')
    users=np.repeat(np.arange(len(lengths),dtype=np.int64),lengths)
    frequency=np.bincount(rows,minlength=len(catalog))
    return torch.as_tensor(rows,device=device),torch.as_tensor(users,device=device),torch.as_tensor(frequency,device=device)


def positive_chunks(events,users,window,chunk=262144):
    """Center-major and context-oldest-first, with no cross-user/same-item pair."""
    if type(window)is not int or window<1 or type(chunk)is not int or chunk<1:raise ValueError('positive window/chunk required')
    offsets=torch.cat((torch.arange(-window,0,device=events.device),torch.arange(1,window+1,device=events.device)))
    for start in range(0,len(events),chunk):
        positions=torch.arange(start,min(start+chunk,len(events)),device=events.device)
        contexts=positions[:,None]+offsets[None,:]
        safe=contexts.clamp(0,max(0,len(events)-1))
        center=events[positions,None].expand_as(contexts);positive=events[safe]
        keep=(contexts>=0)&(contexts<len(events))&(users[safe]==users[positions,None])&(center!=positive)
        # One dynamic compaction per large center chunk, never one per SGD batch.
        yield torch.stack((center,positive),-1)[keep]


def batches(chunks,batch_size):
    if type(batch_size)is not int or batch_size<1:raise ValueError('positive batch size required')
    carry=None
    for pairs in chunks:
        if carry is not None:pairs=torch.cat((carry,pairs));carry=None
        stop=len(pairs)//batch_size*batch_size
        for start in range(0,stop,batch_size):yield pairs[start:start+batch_size]
        if stop<len(pairs):carry=pairs[stop:].clone()
    if carry is not None:yield carry


def negatives(pairs,cdf,probabilities,count,generator):
    # u lives in the eligible mass. Insert the lower forbidden interval, then
    # the upper one (its threshold is measured before either insertion).
    low=torch.minimum(pairs[:,0],pairs[:,1]);high=torch.maximum(pairs[:,0],pairs[:,1])
    a=probabilities[low];b=probabilities[high]
    first=cdf[low]-a;second=cdf[high]-a-b
    u=torch.rand((len(pairs),count),dtype=torch.float64,device=pairs.device,generator=generator)*(1-a-b)[:,None]
    expanded=u+(u>=first[:,None])*a[:,None]+(u>=second[:,None])*b[:,None]
    # Clamp only the upper endpoint rounding, not the sampling probabilities.
    expanded=torch.minimum(expanded,torch.nextafter(torch.ones_like(expanded),torch.zeros_like(expanded)))
    return torch.searchsorted(cdf,expanded,right=True)


def step(center_table,output_table,pairs,negative,processed,total,initial,ratio,finite,summed_loss):
    """One SUM-loss simultaneous SGD batch; all gathers precede all writes."""
    center=center_table[pairs[:,0]];positive=output_table[pairs[:,1]];noise=output_table[negative]
    positive_dot=(center*positive).sum(-1);negative_dot=(center[:,None,:]*noise).sum(-1)
    positive_factor=torch.sigmoid(positive_dot)-1;negative_factor=torch.sigmoid(negative_dot)
    dc=positive_factor[:,None]*positive+(negative_factor[:,:,None]*noise).sum(1)
    dp=positive_factor[:,None]*center;dn=negative_factor[:,:,None]*center[:,None,:]
    loss=-F.logsigmoid(positive_dot).sum()-F.logsigmoid(-negative_dot).sum()
    finite.logical_and_(torch.isfinite(loss)&torch.isfinite(dc).all()&torch.isfinite(dp).all()&torch.isfinite(dn).all()
                        &(negative!=pairs[:,0,None]).all()&(negative!=pairs[:,1,None]).all())
    lr=initial*(1-(1-ratio)*processed.to(torch.float64)/total)
    center_table.index_add_(0,pairs[:,0],-lr*dc)
    output_table.index_add_(0,pairs[:,1],-lr*dp)
    output_table.index_add_(0,negative.flatten(),(-lr*dn).reshape(-1,center.shape[1]))
    summed_loss.add_(loss.to(torch.float64));processed.add_(len(pairs))


def distribution(pairs,negative,p,bins=64,mass=None):
    """Conditional expected/observed counts by deterministic row modulo bins."""
    if mass is None:
        mass=torch.zeros(bins,dtype=torch.float64,device=p.device)
        mass.index_add_(0,torch.arange(len(p),device=p.device)%bins,p)
    scale=negative.shape[1]/(1-p[pairs[:,0]]-p[pairs[:,1]])
    expected=mass*scale.sum()
    expected.index_add_(0,pairs[:,0]%bins,-scale*p[pairs[:,0]])
    expected.index_add_(0,pairs[:,1]%bins,-scale*p[pairs[:,1]])
    observed=torch.zeros(bins,dtype=torch.int64,device=p.device)
    observed.index_add_(0,negative.flatten()%bins,torch.ones(negative.numel(),dtype=torch.int64,device=p.device))
    return observed,expected


class FrequencyAudit:
    """Per-item counts; O(N) finalization once, O(batch) updates without sync."""
    def __init__(self,p):
        self.p=p;self.observed=torch.zeros(len(p),dtype=torch.int64,device=p.device)
        self.excluded=torch.zeros_like(p);self.scale=torch.zeros((),dtype=torch.float64,device=p.device)
    def add(self,pairs,negative):
        scale=negative.shape[1]/(1-self.p[pairs[:,0]]-self.p[pairs[:,1]])
        self.scale.add_(scale.sum())
        self.excluded.index_add_(0,pairs[:,0],scale*self.p[pairs[:,0]])
        self.excluded.index_add_(0,pairs[:,1],scale*self.p[pairs[:,1]])
        self.observed.index_add_(0,negative.flatten(),torch.ones(negative.numel(),dtype=torch.int64,device=self.p.device))
    def report(self):
        expected=self.p*self.scale-self.excluded
        if not torch.isfinite(expected).all() or bool((expected< -1e-7).any()):raise ValueError('invalid conditional expected frequency')
        expected=expected.clamp_min(0);eligible=expected>=5
        residual=self.observed.to(torch.float64)-expected
        pearson=(residual[eligible].square()/expected[eligible]).sum()
        maximum=(residual[eligible].abs()/expected[eligible].sqrt()).max() if bool(eligible.any()) else torch.zeros((),device=self.p.device)
        bins=torch.arange(len(self.p),device=self.p.device)%64
        observed_bins=torch.zeros(64,dtype=torch.int64,device=self.p.device);expected_bins=torch.zeros(64,dtype=torch.float64,device=self.p.device)
        observed_bins.index_add_(0,bins,self.observed);expected_bins.index_add_(0,bins,expected)
        return {'negative_observed_bins':observed_bins.cpu().tolist(),'negative_expected_bins':expected_bins.cpu().tolist(),
                'negative_per_item_diagnostics':{'items_with_expected_ge_5':int(eligible.sum()),'pearson_sum_expected_ge_5':float(pearson),
                    'maximum_absolute_standardized_residual':float(maximum),'observed_total':int(self.observed.sum()),
                    'expected_total':float(expected.sum()),'method':'exact per-item conditional expectation for consumed positive pairs; diagnostic, no p-value or new gate'}}


class NativeSGNS:
    def __init__(self,catalog,histories,*,device='cuda',config=None,compile_step=False,center_chunk=262144):
        self.config=config or SGNSConfig();self.config.validate(False)
        if device not in ('cpu','cuda') or device=='cuda' and not torch.cuda.is_available():raise ValueError('requested CPU/CUDA unavailable')
        self.device=device;self.catalog=np.asarray(catalog);self.histories=histories;self.center_chunk=center_chunk
        self.events,self.users,self.frequency=pack(catalog,histories,device)
        weights=self.frequency.to(torch.float64).pow(.75)
        if torch.count_nonzero(weights).item()<3:raise ValueError('negative distribution has no eligible item')
        self.p=weights/weights.sum();self.cdf=self.p.cumsum(0);self.cdf[-1]=1
        self.participation=torch.zeros(len(catalog),dtype=torch.int64,device=device);self.pairs_per_epoch=0
        for pairs in positive_chunks(self.events,self.users,self.config.window,center_chunk):
            self.pairs_per_epoch+=len(pairs)
            self.participation.index_add_(0,pairs.flatten(),torch.ones(pairs.numel(),dtype=torch.int64,device=device))
        if not self.pairs_per_epoch:raise ValueError('no positive pairs')
        generator=torch.Generator(device='cpu').manual_seed(self.config.seed)
        initial=torch.empty((len(catalog),self.config.dim),dtype=torch.float32)
        initial.uniform_(-.5/self.config.dim,.5/self.config.dim,generator=generator)
        self.initial_sha=_array_digest(initial.numpy());self.center=initial.to(device);self.output=torch.zeros_like(self.center)
        self.generator=torch.Generator(device=device).manual_seed(self.config.seed)
        self.processed=torch.zeros((),dtype=torch.int64,device=device);self.finite=torch.ones((),dtype=torch.bool,device=device)
        self.total=self.pairs_per_epoch*self.config.epochs;self.steps=0;self.epochs=0
        self.step=torch.compile(step,fullgraph=True,dynamic=False) if compile_step else step
        self.compiled=compile_step;self.partial=False
    def epoch(self,*,max_batches=None,observe=True):
        if self.partial:raise ValueError('bounded prefix cannot restart epoch or continue training')
        if self.epochs>=self.config.epochs:raise ValueError('frozen epoch count exhausted')
        if max_batches is not None and (type(max_batches)is not int or max_batches<1):raise ValueError('positive bounded prefix required')
        summed=torch.zeros((),dtype=torch.float64,device=self.device);frequencies=FrequencyAudit(self.p);count=0;steps=0
        for pair in batches(positive_chunks(self.events,self.users,self.config.window,self.center_chunk),self.config.batch_size):
            noise=negatives(pair,self.cdf,self.p,self.config.negatives,self.generator)
            self.step(self.center,self.output,pair,noise,self.processed,self.total,self.config.initial_lr,self.config.final_lr_ratio,self.finite,summed)
            if observe:
                frequencies.add(pair,noise)
            count+=len(pair);steps+=1;self.steps+=1
            if max_batches is not None and steps>=max_batches:break
        if self.device=='cuda':torch.cuda.synchronize()
        if not bool(self.finite) or not torch.isfinite(self.center).all() or not torch.isfinite(self.output).all():raise ValueError('SGNS nonfinite loss/gradient/table; STOP')
        complete=count==self.pairs_per_epoch
        if complete:self.epochs+=1
        else:self.partial=True
        return {'epoch':self.epochs if complete else self.epochs+1,'complete_epoch':complete,'positive_pairs':count,'steps':steps,
                'processed_positive_pairs':int(self.processed),'summed_loss':float(summed),
                **frequencies.report(),'frequency_check':'64 fixed row-modulo bins; conditional expected counts, no fitted parameters; diagnostic only'}


def fit_sgns(catalog,histories,content,*,device='cuda',compile_step=True):
    """Frozen five-epoch function, returning vectors/audit without any writes."""
    content=np.asarray(content,dtype=np.float32)
    if content.shape!=(len(catalog),64) or not np.isfinite(content).all():raise ValueError('finite original content required')
    native=NativeSGNS(catalog,histories,device=device,compile_step=compile_step)
    fallback=native.participation.cpu().numpy()==0;norm=np.linalg.norm(content.astype(np.float64),axis=1)
    if (norm[fallback]==0).any():raise ValueError('zero norm fallback content')
    epochs=[native.epoch() for _ in range(native.config.epochs)]
    vectors=native.center.cpu().numpy().copy();output_sha=_array_digest(native.output.cpu().numpy())
    norms=np.linalg.norm(vectors.astype(np.float64),axis=1)
    if not np.isfinite(vectors).all() or (norms[~fallback]==0).any():raise ValueError('nonfinite or participating zero vector')
    vectors[~fallback]=(vectors[~fallback].astype(np.float64)/norms[~fallback,None]).astype('f4')
    vectors[fallback]=(content[fallback].astype(np.float64)/norm[fallback,None]).astype('f4')
    health=health_check(vectors,catalog,histories)
    return vectors,{'schema':'sgns-gpu-native-b-audit-v1','config':asdict(native.config),'engineering':False,'device':device,
        'positive_pairs_per_epoch':native.pairs_per_epoch,'processed_positive_pairs':int(native.processed),'epoch_audits':epochs,
        'steps':native.steps,'epochs_completed':native.epochs,'health':health,'vectors_sha256':feature_digest(vectors),
        'catalog_sha256':catalog_digest(catalog),'train_sequence_sha256':_sequence_digest(histories),'content_sha256':feature_digest(content),
        'input_initialization_sha256':native.initial_sha,'output_table_sha256':output_sha,'fallback_count':int(fallback.sum()),
        'training_contract':{'loss_reduction':'sum','optimizer':'SGD','momentum':0,'weight_decay':0,'gradient_clipping':None,
            'frequency_power':.75,'subsampling':False,'negative_replacement':True,'negative_exclusions':['center','positive'],
            'negative_rng':'torch device Generator(seed), continuous across epochs; conditional inverse CDF',
            'positive_order':'user ascending; center ascending; context oldest to newest','distance_weighting':False,
            'watch_ratio_weighting':False,'early_stopping':False,'export_table':'center_only'},
        'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'compiled_step':compile_step,
                   'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
        'limits':['different RNG stream and scatter-add floating ordering; four gates required','no anchor or scientific acceptance claim']}
