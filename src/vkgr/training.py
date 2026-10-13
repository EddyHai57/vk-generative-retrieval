"""Original bounded training with recoverable shuffled data cursor."""
from datetime import datetime,timezone
from pathlib import Path
import random
import numpy as np
import torch
from .data_audit import file_sha256


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class ShuffledCursor:
    def __init__(self,size,seed):
        if type(size) is not int or size<1: raise ValueError('empty training data')
        self.size=size;self.generator=torch.Generator().manual_seed(seed)
        self.epoch=0;self.cursor=0;self.order=torch.randperm(size,generator=self.generator)

    def next(self,count):
        if type(count) is not int or count<1: raise ValueError('invalid batch count')
        result=[]
        while len(result)<count:
            take=min(count-len(result),self.size-self.cursor)
            result.extend(self.order[self.cursor:self.cursor+take].tolist());self.cursor+=take
            if self.cursor==self.size:
                self.epoch+=1;self.cursor=0;self.order=torch.randperm(self.size,generator=self.generator)
        return result

    def state_dict(self):
        return {'size':self.size,'epoch':self.epoch,'cursor':self.cursor,'order':self.order.clone(),'generator':self.generator.get_state()}

    def load_state_dict(self,value):
        if value['size']!=self.size or not 0<=value['cursor']<self.size or sorted(value['order'].tolist())!=list(range(self.size)): raise ValueError('invalid data cursor')
        self.epoch=value['epoch'];self.cursor=value['cursor'];self.order=value['order'];self.generator.set_state(value['generator'])


def train_step(model,optimizer,codec,examples,cursor,batch_size):
    rows=[examples[i] for i in cursor.next(batch_size)]
    source,decoder,labels=codec.batch([r['history'] for r in rows],[r['target'] for r in rows],next(model.parameters()).device)
    model.train();optimizer.zero_grad(set_to_none=True)
    logits=model(source,decoder)
    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
    if not torch.isfinite(loss): raise ValueError('nonfinite loss')
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not torch.isfinite(norm): raise ValueError('nonfinite gradient')
    optimizer.step()
    return {'loss':float(loss.detach()),'gradient_norm':float(norm.detach())}


def _utc(value):
    date=datetime.fromisoformat(value)
    if date.utcoffset() is None: raise ValueError('timezone-aware UTC required')
    return date.astimezone(timezone.utc)


def stop_reason(started_at,deadline,max_seconds,now=None):
    started=_utc(started_at);end=_utc(deadline);current=_utc(now) if now is not None else datetime.now(timezone.utc)
    if current>=end: return 'deadline'
    if (current-started).total_seconds()>=max_seconds: return 'time_limit'
    return None


def save_checkpoint(path,model,optimizer,cursor,step,binding,started_at):
    path=Path(path);_utc(started_at)
    rng=np.random.get_state()
    value={'schema_version':1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'cursor':cursor.state_dict(),
           'step':step,'binding':binding,'started_at_utc':started_at,'rng_torch':torch.get_rng_state(),
           'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           'rng_python':random.getstate(),'rng_numpy':[rng[0],rng[1].tolist(),int(rng[2]),int(rng[3]),float(rng[4])]}
    with path.open('xb') as stream: torch.save(value,stream)
    return file_sha256(path)


def load_checkpoint(path,expected_sha,model,optimizer,cursor,binding):
    path=Path(path)
    if path.is_symlink() or file_sha256(path)!=expected_sha: raise ValueError('checkpoint hash mismatch')
    # Only caller-pinned checkpoints produced by this run; never arbitrary pickle.
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value.get('schema_version')!=1 or value['binding']!=binding: raise ValueError('checkpoint binding mismatch')
    if type(value['step']) is not int or value['step']<0: raise ValueError('invalid checkpoint step')
    _utc(value['started_at_utc']);model.load_state_dict(value['model']);optimizer.load_state_dict(value['optimizer']);cursor.load_state_dict(value['cursor'])
    torch.set_rng_state(value['rng_torch']);random.setstate(value['rng_python'])
    r=value['rng_numpy'];np.random.set_state((r[0],np.array(r[1],dtype=np.uint32),r[2],r[3],r[4]))
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(value['rng_cuda'])
    return value
