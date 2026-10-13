"""P4E strategies; immutable P4D loss/model primitives remain separate."""
import math
import random
import torch
from .retrieval_training import masked_ce


def uniform_columns(rows, catalog, idmap, rng, extra_count=128):
    if not rows or catalog != sorted(set(catalog)) or any(type(i) is not int or i < 0 for i in catalog):
        raise ValueError('invalid candidate input')
    if type(extra_count) is not int or extra_count < 1 or any(idmap.get(i,0)<=0 for i in catalog):
        raise ValueError('invalid negative count/idmap')
    targets=[r['target'] for r in rows]
    if any(t not in idmap or t not in catalog or idmap[t]<=0 for t in targets):
        raise ValueError('invalid target')
    base=set(targets);pool=[i for i in catalog if i not in base]
    if len(pool)<extra_count:
        raise ValueError('insufficient distinct negative candidates')
    negatives=rng.sample(pool,extra_count)
    columns=sorted(base|set(negatives));lookup={i:j for j,i in enumerate(columns)}
    masks=[]
    for row in rows:
        past=set(row['past_likes'])
        masks.append([i==row['target'] or i not in past for i in columns])
    return (torch.tensor([idmap[i] for i in columns],dtype=torch.long),
            torch.tensor([lookup[i] for i in targets],dtype=torch.long),
            torch.tensor(masks,dtype=torch.bool),negatives)


class UserPrefixCursor:
    """Uniform eligible user, then uniform prefix, both with replacement."""
    def __init__(self,user_ids,seed,epoch_draws):
        import hashlib,json
        if not user_ids or any(type(u) is not int or u<0 for u in user_ids) or type(epoch_draws) is not int or epoch_draws!=len(user_ids):
            raise ValueError('invalid replacement population/budget')
        self.groups={u:[] for u in sorted(set(user_ids))}
        for j,user in enumerate(user_ids):self.groups[user].append(j)
        self.users=list(self.groups);self.size=epoch_draws;self.epoch=0;self.offset=0;self.rng=random.Random(seed)
        self.group_sha=hashlib.sha256(json.dumps(user_ids,separators=(',',':')).encode()).hexdigest()

    def next(self,count):
        if type(count) is not int or count<1:raise ValueError('invalid draw count')
        if self.offset==self.size:self.epoch+=1;self.offset=0
        result=[]
        for _ in range(min(count,self.size-self.offset)):
            group=self.groups[self.users[self.rng.randrange(len(self.users))]]
            result.append(group[self.rng.randrange(len(group))])
        self.offset+=len(result);return result

    def state_dict(self):
        return {'kind':'user_prefix_replacement_v1','size':self.size,'group_sha':self.group_sha,'epoch':self.epoch,'offset':self.offset,'rng':self.rng.getstate()}

    def load_state_dict(self,state):
        if set(state)!={'kind','size','group_sha','epoch','offset','rng'} or state['kind']!='user_prefix_replacement_v1' or state['size']!=self.size or state['group_sha']!=self.group_sha or type(state['epoch']) is not int or state['epoch']<0 or type(state['offset']) is not int or not 0<=state['offset']<=self.size:
            raise ValueError('replacement state mismatch')
        def tuples(v):return tuple(tuples(x) for x in v) if isinstance(v,(tuple,list)) else v
        self.rng.setstate(tuples(state['rng']));self.epoch=state['epoch'];self.offset=state['offset']


def ablation_train_step(model,optimizer,rows,idmap,stats,epoch,temperature,ablation,negative_rng):
    from .retrieval_training import candidate_batch
    if ablation not in ('uniform_negatives','user_balance','watch_history'):raise ValueError('unknown ablation')
    if ablation=='uniform_negatives':columns,labels,mask,negatives=uniform_columns(rows,sorted(idmap),idmap,negative_rng)
    else:
        columns,labels,mask=candidate_batch(rows,idmap);negatives=[]
    if any(not 1<=len(r['history'])<=50 or any(i not in idmap for i in r['history']) for r in rows):raise ValueError('invalid history')
    histories=torch.zeros(len(rows),max(len(r['history']) for r in rows),dtype=torch.long)
    for j,r in enumerate(rows):histories[j,:len(r['history'])]=torch.tensor([idmap[i] for i in r['history']])
    device=next(model.parameters()).device;model.train();optimizer.zero_grad(set_to_none=True)
    loss,skips=masked_ce(model.users(histories.to(device)),model.items(columns.to(device)),labels.to(device),mask.to(device),temperature)
    stats.record(epoch,len(rows),skips);stats.check(epoch)
    result={'drawn':len(rows),'skipped':skips,'negative_ids':negatives,'loss':None,'gradient_norm':None}
    if loss is None:return result
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite parameters')
    return {**result,'loss':float(loss.detach()),'gradient_norm':float(norm.detach())}
