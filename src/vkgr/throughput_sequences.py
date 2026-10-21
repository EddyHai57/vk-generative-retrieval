"""One contiguous observed sequence per user; sample storage is user+position only."""
import hashlib
import json
from pathlib import Path
import numpy as np


def position(p):
    if len(p)!=2 or any(type(x)is not int or not 0<=x<2**32 for x in p):raise ValueError('week/ordinal must be uint32')
    return (p[0]<<32)|p[1]


class SequenceStore:
    COLUMNS=('users','offsets','items','positions','likes','side','ctx','references')

    def __init__(self,**columns):
        self.__dict__.update(columns)
        self.root=None

    @classmethod
    def from_columns(cls,users,items,positions,likes,side,ctx,references,*,byte_limit=0):
        """Production columnar path: no Python event dictionaries or histories."""
        if byte_limit and len(users)*256+len(references)*32>byte_limit:raise MemoryError('columnar coercion budget exceeded before allocation')
        users=np.asarray(users);items=np.asarray(items);positions=np.asarray(positions);likes=np.asarray(likes)
        side=np.asarray(side);ctx=np.asarray(ctx);refs=np.asarray(references).reshape(-1,2)
        for array in (side,ctx):
            if not np.issubdtype(array.dtype,np.integer)or np.any(array<np.iinfo(np.int64).min)or np.any(array>np.iinfo(np.int64).max):raise ValueError('lossless integer side/context required')
        if not np.issubdtype(refs.dtype,np.integer)or np.any(refs<0)or np.any(refs[:,0]>=2**32):raise ValueError('typed user/position references required')
        n=len(users)
        if any(len(a)!=n for a in (items,positions,likes,side,ctx))or side.shape!=(n,3)or ctx.shape!=(n,2):raise ValueError('column shape mismatch')
        if not np.issubdtype(users.dtype,np.integer)or not np.issubdtype(items.dtype,np.integer)or np.any(users<0)or np.any(users>=2**32)or np.any(items<0)or np.any(items>=2**32):raise ValueError('invalid IDs')
        if likes.dtype!=np.bool_ or not np.issubdtype(positions.dtype,np.integer)or np.any(positions<0)or np.any(positions[1:]<=positions[:-1]):raise ValueError('global physical order/typed likes required')
        # Inputs, output columns, argsort permutation + mergesort workspace,
        # unique/counts workspace and cast temporaries; conservative O(N) cap.
        estimate=2*sum(a.nbytes for a in (users,items,positions,likes,side,ctx,refs))+n*128
        if byte_limit and estimate>byte_limit:raise MemoryError('columnar allocation budget exceeded')
        refs=refs.astype(np.uint64)
        order=np.argsort(users,kind='stable');unique,counts=np.unique(users,return_counts=True)
        return cls(users=unique.astype(np.uint32),offsets=np.r_[0,np.cumsum(counts,dtype=np.int64)],
            items=items[order].astype(np.uint32),positions=positions[order].astype(np.uint64),
            likes=likes[order],side=side[order].astype(np.int64),ctx=ctx[order].astype(np.int64),references=refs)

    @classmethod
    def from_events(cls,events,references,*,byte_limit=0):
        # Fixture/convenience adapter. Full datasets must use from_columns.
        if byte_limit and len(events)*100+len(references)*16>byte_limit:raise MemoryError('sequence budget exceeded before allocation')
        groups={};previous=-1
        for e in events:
            pos=position(e['position']);u=e['user_id']
            if type(u)is not int or not 0<=u<2**32 or pos<=previous:raise ValueError('event ID/order conflict')
            if type(e['like'])is not bool:raise ValueError('typed like required')
            if type(e['item_id'])is not int or not 0<=e['item_id']<2**32:raise ValueError('invalid item ID')
            previous=pos;groups.setdefault(u,[]).append(e)
        users=sorted(groups);flat=[e for u in users for e in groups[u]]
        offsets=np.cumsum([0]+[len(groups[u])for u in users],dtype=np.int64)
        refs=np.array([(u,position(p))for u,p in references],dtype=np.uint64).reshape(-1,2)
        if any(u not in groups for u,_ in references):raise ValueError('unknown sample user')
        return cls(users=np.array(users,dtype=np.uint32),offsets=offsets,
                   items=np.array([e['item_id']for e in flat],dtype=np.uint32),
                   positions=np.array([position(e['position'])for e in flat],dtype=np.uint64),
                   likes=np.array([e['like']for e in flat],dtype=np.bool_),
                   side=np.array([e.get('side',[0,0,0])for e in flat],dtype=np.int64).reshape(-1,3),
                   ctx=np.array([e.get('ctx',[0,0])for e in flat],dtype=np.int64).reshape(-1,2),references=refs)

    def rows(self,indices,*,length=50):
        if type(length)is not int or length<1:raise ValueError('positive frozen history length required')
        result=[]
        for index in indices:
            if type(index)is not int or not 0<=index<len(self.references):raise IndexError('sample index')
            user,target=map(int,self.references[index]);slot=int(np.searchsorted(self.users,user))
            start,end=map(int,self.offsets[slot:slot+2]);j=start+int(np.searchsorted(self.positions[start:end],target))
            if j>=end or int(self.positions[j])!=target or not self.likes[j] or not self.likes[start:j].any():raise ValueError('target/ref is not an original eligible like')
            cut=max(start,j-length);past=sorted(set(map(int,self.items[start:j][self.likes[start:j]])))
            result.append({'example_id':index,'user_id':user,'target':int(self.items[j]),
                'target_position':[target>>32,target&0xffffffff],
                'history':self.items[cut:j].astype(int).tolist(),
                'history_positions':[[int(p)>>32,int(p)&0xffffffff]for p in self.positions[cut:j]],
                'past_likes':past,'side':self.side[cut:j].tolist(),'ctx':self.ctx[cut:j].tolist()})
        return result

    def rows_bound(self,indices,length):
        """Conservative serialized + Python-container bound, without building rows."""
        total=0
        for index in indices:
            user,target=map(int,self.references[index]);slot=int(np.searchsorted(self.users,user))
            start,end=map(int,self.offsets[slot:slot+2]);j=start+int(np.searchsorted(self.positions[start:end],target))
            total+=8192+min(length,j-start)*1024+int(np.count_nonzero(self.likes[start:j]))*64
        return total

    def save(self,root):
        root=Path(root);root.mkdir(parents=True,exist_ok=False);records={}
        for name in self.COLUMNS:
            p=root/(name+'.npy')
            with p.open('xb')as f:np.save(f,getattr(self,name),allow_pickle=False)
            h=hashlib.sha256()
            with p.open('rb')as f:
                for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
            records[p.name]=h.hexdigest()
        with (root/'manifest.json').open('x')as f:json.dump(records,f,sort_keys=True)

    @classmethod
    def load(cls,root,*,mmap=False):
        root=Path(root);records=json.loads((root/'manifest.json').read_text());values={}
        if any(p.is_symlink()for p in (root,*root.parents)):raise ValueError('sequence root symlink forbidden')
        if set(records)!={k+'.npy'for k in cls.COLUMNS}:raise ValueError('sequence column set differs')
        for name in cls.COLUMNS:
            p=root/(name+'.npy')
            h=hashlib.sha256()
            with p.open('rb')as f:
                for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
            if p.is_symlink()or h.hexdigest()!=records[p.name]:raise ValueError('sequence bytes changed')
            values[name]=np.load(p,mmap_mode='r'if mmap else None,allow_pickle=False)
        result=cls(**values);result.root=str(root.resolve())
        return result
