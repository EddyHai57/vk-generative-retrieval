"""Registered F4-T010+C+D+E. Existing component code and constants unchanged.
Users are E; candidate-only C then D tables are drawn after the E init.
"""
import hashlib
import json
import torch
from . import retrieval_feat04b as old
from . import retrieval_feat02 as f2
from . import retrieval_feat03 as f3
from .retrieval_hp import init_embeddings
from .training import seed_all

CONTROL=old.CONTROL
A_FEATURE=old.A_FEATURE
E_FEATURE='f4_combo_c_d_e'
FEATURES4B=(CONTROL,E_FEATURE)
REAL_FEATURES4B=(E_FEATURE,)
STAT_VARIANTS=(E_FEATURE,)
STAT_LAYOUT={E_FEATURE:old.STAT_LAYOUT['f4_item_popularity']+old.STAT_LAYOUT['f4_item_quality']}
BEHAVIOR_BITS=old.BEHAVIOR_BITS
STATS_SHA256=old.STATS_SHA256

def stat_bucket_ids(*args,**kwargs):return old.stat_bucket_ids(*args,**kwargs)
def expo_bucket(*args):return old.expo_bucket(*args)
def first_bucket(*args):return old.first_bucket(*args)
def ratio_bucket4(*args):return old.ratio_bucket4(*args)
def like_bucket(*args):return old.like_bucket(*args)
def stat_bucket_ids_sha256(model):return old.stat_bucket_ids_sha256(model)

class ComboIDMean(old.Feat04bIDMean):
    def __init__(self,content,supported_indices,max_len,stat_buckets):
        super().__init__(old.E_FEATURE,content,supported_indices,max_len)
        if type(stat_buckets) is not dict:raise ValueError('combination requires four stat bucket vectors')
        names=[]
        for _,name,edges in STAT_LAYOUT[E_FEATURE]:
            ids=stat_buckets.get(name)
            if not isinstance(ids,torch.Tensor) or ids.dtype!=torch.long or ids.shape!=(len(content),) or int(ids[0])!=0:
                raise ValueError('invalid immutable combination stat buffer: '+name)
            if int(ids.min())<0 or int(ids.max())>len(getattr(old,edges))+1:raise ValueError('stat bucket outside frozen range')
            self.register_buffer(name,ids.detach().clone());names.append(name)
        object.__setattr__(self,'stat_buffers',tuple(names))
    def candidate_items(self,indices):
        # Candidate terms never enter inherited users/history_items.
        terms=sum(getattr(self,table)(getattr(self,buffer)[indices]) for table,buffer,_ in STAT_LAYOUT[E_FEATURE])
        return self.items(indices)+terms*self.supported[indices].unsqueeze(-1).to(terms.dtype)

def make_feat04b_model(variant,content,supported_indices,max_len,seed,stat_buckets=None):
    if variant==CONTROL:
        return old.make_feat04b_model(CONTROL,content,supported_indices,max_len,seed,stat_buckets)
    if variant!=E_FEATURE or max_len!=50:raise ValueError('only registered L50 C+D+E combination')
    seed_all(seed);model=ComboIDMean(content,supported_indices,max_len,stat_buckets)
    init_embeddings(model)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    f2._small_init(model,model.add_anchor_tables());f3.draw_variant_tables(model)
    f3.init_recency_logits(model);old.init_behavior_beta(model)
    old.init_stat_tables(model,'f4_item_popularity');old.init_stat_tables(model,'f4_item_quality')
    f3.drop_unused_tables(model,old.E_FEATURE)
    if any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n in before):raise ValueError('anchor initialization drift')
    return model

def parameter_report4b(model):return old.parameter_report4b(model)
def expected_param_counts4b(variant):
    if variant==CONTROL:return old.expected_param_counts4b(CONTROL)
    if variant!=E_FEATURE:raise ValueError('unregistered combination variant')
    entries={t:{'total':(len(getattr(old,e))+2)*64,'trainable':(len(getattr(old,e))+1)*64} for t,_,e in STAT_LAYOUT[E_FEATURE]}
    return old._with_entries(old.expected_param_counts4b(old.E_FEATURE),entries)
def bucket_spec_sha256(variant):
    if variant==CONTROL:return old.bucket_spec_sha256(CONTROL)
    if variant!=E_FEATURE:raise ValueError('unregistered combination spec')
    value={'version':'C_D_E_V1','components':[old.bucket_spec_sha256(v) for v in ('f4_item_popularity','f4_item_quality',old.E_FEATURE)],'table_draw_order':[v[0] for v in STAT_LAYOUT[E_FEATURE]],'formula':'E_users; id+supported*(C1+C2+D1+D2)'}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
