"""FP32 in-batch pipeline; no formal architecture search or evaluation."""
import random
import numpy as np
import torch
class EpochCursor:
 def __init__(self,size,seed):
  if type(size)is not int or size<1:raise ValueError('invalid example count')
  self.size=size;self.rng=random.Random(seed);self.epoch=0;self.offset=0;self.order=list(range(size));self.rng.shuffle(self.order)
 def next(self,count):
  if type(count)is not int or count<1:raise ValueError('invalid batch count')
  if self.offset==self.size:
   self.epoch+=1;self.offset=0;self.order=list(range(self.size));self.rng.shuffle(self.order)
  end=min(self.offset+count,self.size);result=self.order[self.offset:end];self.offset=end;return result
 def state_dict(self):
  return {'size':self.size,'epoch':self.epoch,'offset':self.offset,'order':list(self.order),'rng':self.rng.getstate()}
 def load_state_dict(self,state):
  if state['size']!=self.size or type(state['epoch'])is not int or state['epoch']<0 or type(state['offset'])is not int or not 0<=state['offset']<=self.size or sorted(state['order'])!=list(range(self.size)):raise ValueError('invalid cursor state')
  self.rng.setstate(state['rng']);self.epoch=state['epoch'];self.offset=state['offset'];self.order=list(state['order'])

def candidate_batch(rows,idmap):
 targets=[r['target'] for r in rows]
 if not rows or any(t not in idmap or idmap[t]<=0 for t in targets):raise ValueError('invalid/PAD target')
 columns=sorted(set(targets));lookup={item:i for i,item in enumerate(columns)}
 masks=[[item==r['target'] or item not in set(r['past_likes']) for item in columns] for r in rows]
 return torch.tensor([idmap[i] for i in columns]),torch.tensor([lookup[i] for i in targets]),torch.tensor(masks,dtype=torch.bool)

def masked_ce(users,items,labels,mask,temperature):
 if not np.isfinite(temperature) or temperature<=0 or users.ndim!=2 or items.ndim!=2 or users.shape[1]!=items.shape[1] or mask.shape!=(len(users),len(items)) or labels.shape!=(len(users),):raise ValueError('invalid loss input')
 if not torch.isfinite(users).all() or not torch.isfinite(items).all():raise ValueError('nonfinite vectors')
 if not mask[torch.arange(len(users)),labels].all():raise ValueError('masked target')
 valid=mask.sum(1)>1;skips=int((~valid).sum())
 if not valid.any():return None,skips
 scores=torch.nn.functional.normalize(users.float(),dim=1)@torch.nn.functional.normalize(items.float(),dim=1).T/temperature
 loss=torch.nn.functional.cross_entropy(scores.masked_fill(~mask,-torch.inf)[valid],labels[valid])
 if not torch.isfinite(loss):raise ValueError('nonfinite loss')
 return loss,skips

class SamplerStats:
 def __init__(self):self.epochs={};self.drawn=0;self.skipped=0
 def record(self,epoch,drawn,skipped):
  if type(epoch)is not int or epoch<0 or type(drawn)is not int or type(skipped)is not int or not 0<=skipped<=drawn:raise ValueError('invalid sampler stats')
  counts=self.epochs.setdefault(epoch,[0,0]);counts[0]+=drawn;counts[1]+=skipped;self.drawn+=drawn;self.skipped+=skipped
 def check(self,epoch):
  drawn,skipped=self.epochs[epoch]
  if skipped/drawn>.01 or self.skipped/self.drawn>.01:raise ValueError('no-negative skip fraction exceeds 1%')

class IDMean(torch.nn.Module):
 def __init__(self,catalog_size,supported_indices):
  super().__init__();self.embedding=torch.nn.Embedding(catalog_size+1,64,padding_idx=0)
  supported=torch.zeros(catalog_size+1,dtype=torch.bool)
  if any(type(i)is not int or not 1<=i<=catalog_size for i in supported_indices):raise ValueError('invalid supported ID')
  supported[supported_indices]=True;self.register_buffer('supported',supported)
 def items(self,indices):
  if indices.dtype!=torch.long or torch.any(indices<0) or torch.any(indices>=len(self.supported)):raise ValueError('invalid item index')
  return self.embedding(indices)*self.supported[indices].unsqueeze(-1)
 def users(self,histories):
  if histories.ndim!=2 or torch.any((histories!=0).sum(1)==0):raise ValueError('empty history')
  return self.items(histories).sum(1)/(histories!=0).sum(1,keepdim=True)

def train_step(model,optimizer,rows,idmap,stats,epoch,temperature):
 columns,labels,mask=candidate_batch(rows,idmap)
 if any(not 1<=len(r['history'])<=50 or any(i not in idmap for i in r['history']) for r in rows):raise ValueError('invalid history')
 width=max(len(r['history']) for r in rows);histories=torch.zeros(len(rows),width,dtype=torch.long)
 for j,r in enumerate(rows):histories[j,:len(r['history'])]=torch.tensor([idmap[i] for i in r['history']])
 device=next(model.parameters()).device;model.train();optimizer.zero_grad(set_to_none=True)
 loss,skips=masked_ce(model.users(histories.to(device)),model.items(columns.to(device)),labels.to(device),mask.to(device),temperature)
 stats.record(epoch,len(rows),skips);stats.check(epoch)
 if loss is None:return {'loss':None,'skipped':skips,'drawn':len(rows)}
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
 if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
 optimizer.step()
 if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite parameters')
 return {'loss':float(loss.detach()),'skipped':skips,'drawn':len(rows),'gradient_norm':float(norm.detach())}

def save_checkpoint(path,model,optimizer,cursor,stats,step,binding,budget):
 from pathlib import Path
 from .data_audit import file_sha256
 rng=np.random.get_state()
 state={'schema_version':1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'cursor':cursor.state_dict(),'stats':{'epochs':stats.epochs,'drawn':stats.drawn,'skipped':stats.skipped},'step':step,'binding':binding,'budget':budget,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'python_rng':random.getstate(),'numpy_rng':[rng[0],rng[1].tolist(),int(rng[2]),int(rng[3]),float(rng[4])]}
 with Path(path).open('xb') as stream:torch.save(state,stream)
 return file_sha256(path)

def load_checkpoint(path,digest,model,optimizer,cursor,stats,binding):
 from .retrieval_data import checked
 state=torch.load(checked(path,digest),map_location='cpu',weights_only=True)
 if state.get('schema_version')!=1 or state['binding']!=binding or type(state['step'])is not int or state['step']<0:raise ValueError('checkpoint binding mismatch')
 cursor.load_state_dict(state['cursor']);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
 counts=state['stats'];stats.epochs=counts['epochs'];stats.drawn=counts['drawn'];stats.skipped=counts['skipped']
 if sum(v[0] for v in stats.epochs.values())!=stats.drawn or sum(v[1] for v in stats.epochs.values())!=stats.skipped or any(not 0<=v[1]<=v[0] for v in stats.epochs.values()):raise ValueError('invalid sampler checkpoint')
 torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);r=state['numpy_rng'];np.random.set_state((r[0],np.array(r[1],dtype=np.uint32),r[2],r[3],r[4]))
 if torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
 return state
