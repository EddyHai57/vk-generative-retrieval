"""Strict-prefix profile snapshots and train-only buckets; no I/O or model code."""
from bisect import bisect_left
from collections import deque
import hashlib
import json

def require(ok,msg):
    if not ok:raise ValueError(msg)
def digest(v):return hashlib.sha256((json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()
def identity(row,ordinal,split):
    # Target values do not feed either features or per-row model input checks.
    fields={k:row[k] for k in ('user_id','history','history_positions')}
    if split=='train':fields['example_id']=row['example_id']
    return digest({'split':split,'ordinal':ordinal,'fields':fields})

def build_profiles(partitions,train_rows,queries):
    targets={tuple(r['target_position']):(j,r) for j,r in enumerate(train_rows)}
    require(len(targets)==len(train_rows) and all(r['example_id']==j for j,r in enumerate(train_rows)),'training identity/order')
    totals={};recent={};outputs={};previous=-1
    def snapshot(user):
        exp,likes=totals.get(user,(0,0));hist=recent.get(user,())
        values=[e[2] for e in hist if e[2] is not None]
        return [exp,likes,sum(values),len(values)],hist
    def record(row,j,split):
        counts,hist=snapshot(row['user_id'])
        require(row['history']==[e[0] for e in hist] and row['history_positions']==[e[1] for e in hist],'profile history mismatch')
        return {'ordinal':j,'identity_sha256':identity(row,j,split),'counts':counts,'history_len':len(hist)}
    for week,events in partitions:
        require(type(week) is int and previous<week<=24,'train-only chronological weeks')
        previous=week
        for ordinal,event in enumerate(events):
            user,item,like=event['user_id'],event['item_id'],event.get('like')
            require(type(user) is int and type(item) is int and (like is None or type(like) is bool),'event domain')
            match=targets.get((week,ordinal))
            if match is not None:
                j,row=match;require(row['user_id']==user and row['target']==item,'target event identity')
                outputs[j]=record(row,j,'train') # BEFORE updating the current target event.
            exp,likes=totals.get(user,(0,0));totals[user]=(exp+1,likes+int(like is True))
            recent.setdefault(user,deque(maxlen=50)).append((item,[week,ordinal],like))
    require(len(outputs)==len(train_rows),'target positions missing from train')
    val=[]
    for j,q in enumerate(queries):
        require(q['target_position'][0]==25,'validation week25 required')
        val.append(record(q,j,'validation')) # Never ingest validation rows.
    return {'train':[outputs[j] for j in range(len(train_rows))],'validation':val}

def count_bucket(n):
    require(type(n) is int and n>=0,'nonnegative count required');return min(n.bit_length(),16)
def rate_bucket(likes,valid,boundaries):
    require(type(likes) is int and type(valid) is int and 0<=likes<=valid,'valid rate counts')
    if valid==0:return 9
    if likes==0:return 0
    return 1+bisect_left(boundaries,likes/valid)
def fit_bounds(train):
    import numpy as np
    rates=[r['counts'][2]/r['counts'][3] for r in train if r['counts'][3]>0 and r['counts'][2]>0]
    boundaries=sorted(set(float(x) for x in np.quantile(rates,[k/8 for k in range(1,8)],method='linear'))) if rates else []
    return {'schema':'profile-quantiles-v1','boundaries':boundaries,'positive_samples':len(rates),'train_snapshots_sha256':digest(train),'fit_values_sha256':digest(rates),'method':'numpy.quantile-linear','weighting':'one_training_sample_one_vote','positive_interval_count':len(boundaries)+1,'positive_occupied_bins':len({1+bisect_left(boundaries,x) for x in rates})}
def attach(rows,profiles,bounds,split):
    require(split in ('train','validation') and len(rows)==len(profiles),'profile sidecar population')
    out=[];b=bounds['boundaries'];require(len(b)<=7 and b==sorted(set(b)) and all(0<x<=1 for x in b),'fitted boundaries')
    for j,(row,profile) in enumerate(zip(rows,profiles)):
        require(profile['ordinal']==j and profile['identity_sha256']==identity(row,j,split),'profile sidecar identity/order mismatch')
        exp,likes,num,den=profile['counts'];require(likes<=exp and profile['history_len']==len(row['history']),'profile count identity')
        out.append({**row,'profile_ids':[count_bucket(exp),count_bucket(likes),rate_bucket(num,den,b)]})
    return out
