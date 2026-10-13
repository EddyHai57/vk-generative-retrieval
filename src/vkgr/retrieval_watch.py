"""P4E watch sidecars: strict physical-order train events, immutable targets."""
from collections import Counter,deque
import math
from .retrieval_data import catalog_map,validate_example


def build_watch(partitions,examples,queries,catalog):
    idmap=catalog_map(catalog)
    for j,row in enumerate(examples):validate_example(row,idmap,j)
    targets={tuple(e['target_position']):e for e in examples}
    if len(targets)!=len(examples):raise ValueError('duplicate target position')
    watches={};likes={};past={};outputs={};counts=Counter();user_counts=Counter();previous=-1
    for week,events in partitions:
        if type(week) is not int or not previous<week<=24:raise ValueError('train ordering conflict')
        previous=week
        for ordinal,row in enumerate(events):
            user,item,like=row.get('user_id'),row.get('item_id'),row.get('like')
            if type(user) is not int or not 0<=user<2**32 or type(item) is not int or item not in idmap or type(like) is not bool:
                raise ValueError('raw interaction/catalog mismatch')
            position=[week,ordinal];watch=watches.setdefault(user,deque(maxlen=50));hist=likes.setdefault(user,deque(maxlen=50));seen=past.setdefault(user,set())
            example=targets.get(tuple(position))
            if example is not None:
                if not like or example['user_id']!=user or example['target']!=item or example['history']!=[x[0] for x in hist] or example['history_positions']!=[x[1] for x in hist] or example['past_likes']!=sorted(seen) or not watch:
                    raise ValueError('immutable example reconciliation mismatch')
                outputs[example['example_id']]={'example_id':example['example_id'],'history':[x[0] for x in watch],'history_positions':[x[1] for x in watch]}
            elif like and hist:
                raise ValueError('eligible training target missing')
            watch.append((item,position))
            if like:
                hist.append((item,position));seen.add(item);counts[item]+=1;user_counts[user]+=1
    if len(outputs)!=len(examples):raise ValueError('target positions absent')
    validation=[];users=set()
    for q in queries:
        if set(q)!={'user_id','history','history_positions','target','target_position','oov'} or q['user_id'] in users or q['target_position'][0]!=25 or q['oov']!=(q['target'] not in idmap):
            raise ValueError('validation identity mismatch')
        user=q['user_id'];users.add(user);hist=likes.get(user,[]);watch=watches.get(user,[])
        if not watch or q['history']!=[x[0] for x in hist] or q['history_positions']!=[x[1] for x in hist]:
            raise ValueError('train-only validation like history mismatch')
        validation.append({**q,'history':[x[0] for x in watch],'history_positions':[x[1] for x in watch]})
    positive=sorted(counts.values())
    if not positive:raise ValueError('no train positive support')
    thresholds={name:positive[max(0,math.ceil(len(positive)*fraction)-1)] for name,fraction in [('median',.5),('p90',.9)]}
    slices={'schema_version':1,'popularity_thresholds':thresholds,'item_like_counts':{str(i):counts[i] for i in catalog},'user_like_counts':{str(u):user_counts[u] for u in sorted(user_counts)},'user_buckets':[1,5,20],'definition':'train_like_counts_only_no_holdout_boundaries'}
    return [outputs[j] for j in range(len(examples))],validation,slices


def history_overlay(examples,sidecar):
    if len(examples)!=len(sidecar):raise ValueError('watch example count mismatch')
    result=[]
    for j,(e,h) in enumerate(zip(examples,sidecar)):
        if set(h)!={'example_id','history','history_positions'} or h['example_id']!=j or not 1<=len(h['history'])<=50 or len(h['history'])!=len(h['history_positions']) or any(a>=b for a,b in zip(h['history_positions'],h['history_positions'][1:])) or h['history_positions'][-1]>=e['target_position']:
            raise ValueError('invalid strict-past watch sidecar')
        result.append({**e,'history':h['history'],'history_positions':h['history_positions']})
    return result
