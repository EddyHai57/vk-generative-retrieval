"""M1-mean plus zero-initialized profile tables after all shared random draws."""
import torch
from .retrieval_seqm1 import make_seqm1_model

class ProfileMean(torch.nn.Module):
    def __init__(self,base,arm):
        super().__init__();self.base=base
        if arm not in ('u2','u3'):raise ValueError('unregistered profile arm')
        self.profile_arm=arm
        self.profile_tables=torch.nn.ModuleList([torch.nn.Embedding.from_pretrained(torch.zeros(n,64),freeze=False) for n in ([17,17] if arm=='u2' else [10])])
    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('base'),name)
    def _split(self,histories,side):
        if side is None or side.ndim!=3 or side.shape[:2]!=histories.shape or side.shape[-1]!=9 or side.dtype!=torch.long:raise ValueError('profile side tensor required')
        ids=side[:,0,6:9]
        if bool(((ids<0)|(ids>=torch.tensor([17,17,10],device=ids.device))).any()):raise ValueError('profile bucket out of range')
        valid=histories!=0
        if bool(((side[...,6:9]!=ids[:,None,:])&valid[...,None]).any()):raise ValueError('profile per-query identity mismatch')
        return side[...,:6],ids
    def profile_vector(self,ids):
        if self.profile_arm=='u2':return self.profile_tables[0](ids[:,0])+self.profile_tables[1](ids[:,1])
        return self.profile_tables[0](ids[:,2])
    def users(self,histories,side=None):
        original,ids=self._split(histories,side)
        return self.base.users(histories,original)+self.profile_vector(ids)
    def items(self,items):return self.base.items(items)
    def parts(self,histories,side,with_anchor=False):return self.base.parts(histories,side[...,:6],with_anchor)
    def diagnose(self,histories,side):
        original,ids=self._split(histories,side);out=self.base.diagnose(histories,original)
        with torch.no_grad():
            p=self.base.parts(histories,original,with_anchor=True);u=self.base.users(histories,original)+self.profile_vector(ids)
            idx=torch.arange(len(histories),device=histories.device);last=p.e[idx,p.valid.sum(1)-1]
            out.update(cos_u_m=torch.nn.functional.cosine_similarity(u,p.m,dim=1),cos_u_last_event=torch.nn.functional.cosine_similarity(u,last,dim=1),profile_norm=self.profile_vector(ids).norm(dim=1))
        return out

def make_profile_model(content,supported_indices,max_len,arm):
    # All original parameter draws finish before profile tables are constructed.
    return ProfileMean(make_seqm1_model(content,supported_indices,max_len,'mean'),arm)

def tensors(batch,idmap,width,feature):
    from .retrieval_feat02_run import _tensors2
    h,s=_tensors2(batch,idmap,width,feature)
    if any(type(r.get('profile_ids')) is not list or len(r['profile_ids'])!=3 or any(type(v) is not int for v in r['profile_ids']) for r in batch):raise ValueError('bound integer profile sidecar required')
    ids=torch.tensor([r['profile_ids'] for r in batch],dtype=torch.long)
    return h,torch.cat((s,ids[:,None,:].expand(-1,width,-1)),dim=-1)
