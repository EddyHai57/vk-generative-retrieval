"""G2 full-vocabulary supervision and resumable training primitives."""
from dataclasses import dataclass
import math
import torch
from .generative_data import sid_targets
from .tiger import TigerModel


def _core(config):
    required={'d_model','heads','encoder_layers','decoder_layers','ff','dropout','precision','learning_rate','weight_decay','gradient_clip_norm','effective_batch','seed'}
    if type(config) is not dict or set(config)!=required:
        raise ValueError('exact effective core config required')
    for name in ('d_model','heads','encoder_layers','decoder_layers','ff','effective_batch','seed'):
        if type(config[name]) is not int:raise ValueError('typed core config required')
    d=config['d_model']
    if d not in (64,128) or config['heads']!=4 or config['encoder_layers']!=1 or config['decoder_layers']!=1 or config['ff']!=4*d or config['effective_batch']!=512 or config['seed'] not in (42,2024,2025):
        raise ValueError('frozen core architecture/budget required')
    if config['dropout']!=0. or config['precision']!='fp32' or config['learning_rate'] not in (.001,.0003) or config['weight_decay']!=.0001 or config['gradient_clip_norm']!=1.:
        raise ValueError('frozen core optimization required')
    return config


def make_model(codec,config,*,device):
    c=_core(config)
    return TigerModel(codec.vocab_size,c['d_model'],c['heads'],1,1,c['ff']).to(device=device,dtype=torch.float32)


def make_optimizer(model,config):
    c=_core(config)
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('FP32 parameters required')
    return torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])


def full_vocab_ce(logits,labels):
    if logits.ndim!=3 or labels.ndim!=2 or logits.shape[:2]!=labels.shape or logits.shape[0]<1 or logits.shape[1]!=4 or logits.shape[2]<3 or labels.dtype!=torch.long or not logits.is_floating_point():
        raise ValueError('four-position CE tensor schema')
    if not torch.isfinite(logits).all() or (labels<0).any() or (labels>=logits.shape[2]).any():raise ValueError('nonfinite logits or invalid label index')
    return torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())


@dataclass(frozen=True)
class StepResult:
    loss_mean: float
    examples: int
    microbatch_sizes: tuple[int,...]
    optimizer_steps: int=1


def train_effective_batch(model,optimizer,rows,codec,*,microbatch,device):
    if type(microbatch) is not int or microbatch<1 or not rows or len(rows)>512:raise ValueError('nonempty bounded effective batch required')
    model.train();optimizer.zero_grad(set_to_none=True)
    n=len(rows);total=0.;sizes=[]
    try:
        for start in range(0,n,microbatch):
            batch=rows[start:start+microbatch]
            source,decoder,labels=(x.to(device) for x in sid_targets(batch,codec))
            loss=full_vocab_ce(model(source,decoder),labels)
            weighted=loss*(len(batch)/n)
            if not torch.isfinite(weighted):raise ValueError('nonfinite loss')
            weighted.backward();total+=float(loss.detach())*len(batch)/n;sizes.append(len(batch))
        try:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        except RuntimeError as exc:raise ValueError('nonfinite gradient') from exc
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite updated parameters')
        return StepResult(total,n,tuple(sizes))
    finally:
        optimizer.zero_grad(set_to_none=True)


class EpochCursor:
    """Explicit train/evaluate boundary; complete shuffled epoch without replacement."""
    def __init__(self,seed,example_count):
        if type(seed) is not int or seed<0 or type(example_count) is not int or example_count<1:raise ValueError('typed epoch seed/count required')
        self.seed=seed;self.example_count=example_count;self.epoch=1
        self._generator=torch.Generator(device='cpu').manual_seed(seed)
        self._new_order()

    def _new_order(self):
        self._before=self._generator.get_state().clone()
        self._permutation=torch.randperm(self.example_count,generator=self._generator).tolist()
        self.next_offset=0;self.phase='train'

    def next_batch(self,size):
        if type(size) is not int or size<1:raise ValueError('positive batch size required')
        if self.phase!='train':return []
        end=min(self.example_count,self.next_offset+size)
        rows=self._permutation[self.next_offset:end];self.next_offset=end
        if end==self.example_count:self.phase='evaluate'
        return rows

    def complete_evaluation(self):
        if self.phase!='evaluate':raise ValueError('complete epoch validation before advancing')
        self.phase='epoch_done'

    def advance_epoch(self):
        if self.phase!='epoch_done':raise ValueError('complete epoch validation before advancing')
        self.epoch+=1;self._new_order()

    def state_dict(self):
        return {'schema_version':1,'seed':self.seed,'example_count':self.example_count,'epoch':self.epoch,
                'permutation':list(self._permutation),'next_offset':self.next_offset,'phase':self.phase,
                'shuffle_rng_before':self._before.clone(),'shuffle_rng_after':self._generator.get_state().clone()}

    def load_state_dict(self,value):
        keys={'schema_version','seed','example_count','epoch','permutation','next_offset','phase','shuffle_rng_before','shuffle_rng_after'}
        if type(value) is not dict or set(value)!=keys:raise ValueError('epoch state schema')
        for name in ('schema_version','seed','example_count','epoch','next_offset'):
            if type(value[name]) is not int:raise ValueError('typed epoch state required')
        if value['schema_version']!=1 or value['seed']!=self.seed or value['example_count']!=self.example_count or value['epoch']<1 or not 0<=value['next_offset']<=self.example_count:raise ValueError('epoch identity/cursor mismatch')
        perm=value['permutation'];phase=value['phase'];offset=value['next_offset']
        if type(perm) is not list or any(type(x) is not int for x in perm) or sorted(perm)!=list(range(self.example_count)):raise ValueError('permutation must cover epoch once')
        if phase not in ('train','evaluate','epoch_done') or (phase=='train')!=(offset<self.example_count):raise ValueError('epoch phase/offset conflict')
        for name in ('shuffle_rng_before','shuffle_rng_after'):
            t=value[name]
            if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.dtype!=torch.uint8 or t.ndim!=1:raise ValueError('shuffle RNG schema')
        try:
            generator=torch.Generator(device='cpu');generator.set_state(value['shuffle_rng_before'])
            if torch.randperm(self.example_count,generator=generator).tolist()!=perm or not torch.equal(generator.get_state(),value['shuffle_rng_after']):raise ValueError('shuffle RNG/permutation conflict')
        except RuntimeError as exc:raise ValueError('invalid shuffle RNG') from exc
        self.epoch=value['epoch'];self.next_offset=offset;self.phase=phase;self._permutation=list(perm)
        self._before=value['shuffle_rng_before'].clone();self._generator=generator


class ExampleStore:
    """Hash-pinned JSONL random-access offsets, without materializing all histories."""
    def __init__(self,path,expected_sha256,expected_count):
        import numpy as np
        from .generative_data import _checked_file
        if type(expected_count) is not int or expected_count<1:raise ValueError('positive example count required')
        self.path=_checked_file(path,expected_sha256);initial=self._stat()
        offsets=[]
        with self.path.open('rb') as stream:
            while True:
                start=stream.tell();line=stream.readline()
                if not line:break
                row=self._parse(line)
                if type(row.get('example_id')) is not int or row['example_id']!=len(offsets):raise ValueError('ordered example id mismatch')
                offsets.append(start)
        if len(offsets)!=expected_count or initial!=self._stat():raise ValueError('example count or file changed during indexing')
        self._stamp=initial;self._offsets=np.frombuffer(np.array(offsets,dtype=np.int64).tobytes(),dtype=np.int64)
        self._offset_identity=(self._offsets,self._offsets.shape,self._offsets.strides,self._offsets.ctypes.data)

    def _stat(self):
        from .generative_data import _no_symlink
        _no_symlink(self.path);s=self.path.stat()
        return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

    @staticmethod
    def _parse(line):
        import json
        from .generative_data import _unique_pairs,_reject_constant
        try:row=json.loads(line,object_pairs_hook=_unique_pairs,parse_constant=_reject_constant)
        except (UnicodeError,json.JSONDecodeError) as exc:raise ValueError('invalid example JSON') from exc
        if type(row) is not dict:raise ValueError('example object required')
        return row

    def __len__(self):return len(self._offsets)

    def rows(self,indices):
        from collections.abc import Sequence
        if not isinstance(indices,Sequence) or isinstance(indices,(str,bytes)) or any(type(i) is not int or not 0<=i<len(self) for i in indices):raise ValueError('example indices out of range')
        array,shape,strides,pointer=self._offset_identity
        if self._offsets is not array or array.flags.writeable or array.shape!=shape or array.strides!=strides or array.ctypes.data!=pointer or self._stat()!=self._stamp:raise ValueError('example file or offsets changed')
        result=[]
        with self.path.open('rb') as stream:
            for i in indices:
                stream.seek(int(self._offsets[i]));row=self._parse(stream.readline())
                if type(row.get('example_id')) is not int or row['example_id']!=i:raise ValueError('indexed example identity mismatch')
                result.append(row)
        if self._stat()!=self._stamp:raise ValueError('example file changed during read')
        return result

@dataclass(frozen=True)
class CheckpointState:
    binding: dict
    selection: dict
    budget: dict
    cursor: dict
    restored_rng: bool


def _budget(value):
    import datetime
    keys={'reserved_steps','completed_steps','consumed_wall_by_phase','resume_attempts','max_optimizer_steps','max_resume_attempts','deadline_utc','parent_ledger_head'}
    if type(value) is not dict or set(value)!=keys:raise ValueError('budget schema')
    for name in ('reserved_steps','completed_steps','resume_attempts','max_optimizer_steps','max_resume_attempts'):
        if type(value[name]) is not int or value[name]<0:raise ValueError('typed nonnegative budget')
    if not 0<=value['completed_steps']<=value['reserved_steps']<=value['max_optimizer_steps'] or value['max_optimizer_steps']<1 or value['resume_attempts']>value['max_resume_attempts']:raise ValueError('budget exceeded')
    wall=value['consumed_wall_by_phase']
    if type(wall) is not dict or set(wall)!={'train','evaluate','verify'} or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in wall.values()):raise ValueError('finite cumulative wall required')
    try:deadline=datetime.datetime.fromisoformat(value['deadline_utc'])
    except (TypeError,ValueError) as exc:raise ValueError('absolute deadline required') from exc
    # Archiving expired work is allowed; execution/resume checks the live gate.
    if deadline.tzinfo is None:raise ValueError('absolute aware deadline required')
    head=value['parent_ledger_head']
    if head is not None and (type(head) is not str or len(head)!=64 or any(ch not in '0123456789abcdef' for ch in head)):raise ValueError('parent budget ledger hash')


def _selection(value,cursor):
    keys={'epoch','best_epoch','best_metric','bad_epochs'}
    if type(value) is not dict or set(value)!=keys or type(value['epoch']) is not int or type(value['bad_epochs']) is not int or not 0<=value['epoch']<=30 or not 0<=value['bad_epochs']<=3:raise ValueError('selection schema/budget')
    best=value['best_epoch'];metric=value['best_metric']
    if value['epoch']==0:
        if best is not None or metric is not None or value['bad_epochs']!=0:raise ValueError('empty selection conflict')
    elif type(best) is not int or not 1<=best<=value['epoch'] or not isinstance(metric,(list,tuple)) or len(metric)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in metric):raise ValueError('selection metric/epoch conflict')
    expected=cursor.epoch if cursor.phase=='epoch_done' else cursor.epoch-1
    if value['epoch']!=expected:raise ValueError('selection/cursor phase conflict')


def _rng(model):
    import random,numpy as np
    state=np.random.get_state()
    cuda=torch.cuda.get_rng_state_all() if next(model.parameters()).device.type=='cuda' else []
    return {'python':random.getstate(),'numpy':{'algorithm':state[0],'keys':state[1].tolist(),'position':state[2],'has_gauss':state[3],'cached_gaussian':state[4]},'torch_cpu':torch.get_rng_state().clone(),'cuda':[x.cpu().clone() for x in cuda]}


def _validate_rng(value,model):
    import random,numpy as np
    if type(value) is not dict or set(value)!={'python','numpy','torch_cpu','cuda'}:raise ValueError('RNG schema')
    try:random.Random().setstate(value['python'])
    except (TypeError,ValueError) as exc:raise ValueError('Python RNG state') from exc
    n=value['numpy']
    if type(n) is not dict or set(n)!={'algorithm','keys','position','has_gauss','cached_gaussian'} or n['algorithm']!='MT19937' or type(n['keys']) is not list or len(n['keys'])!=624 or any(type(v) is not int or not 0<=v<2**32 for v in n['keys']) or type(n['position']) is not int or not 0<=n['position']<=624 or type(n['has_gauss']) is not int or n['has_gauss'] not in (0,1) or type(n['cached_gaussian']) not in (int,float) or not math.isfinite(n['cached_gaussian']):raise ValueError('NumPy RNG state')
    numpy_state=(n['algorithm'],np.array(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian'])
    try:np.random.RandomState().set_state(numpy_state)
    except (TypeError,ValueError) as exc:raise ValueError('NumPy RNG state') from exc
    t=value['torch_cpu']
    if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.dtype!=torch.uint8 or t.ndim!=1:raise ValueError('Torch CPU RNG state')
    try:torch.Generator(device='cpu').set_state(t)
    except RuntimeError as exc:raise ValueError('Torch CPU RNG state') from exc
    cuda=value['cuda'];is_cuda=next(model.parameters()).device.type=='cuda'
    if type(cuda) is not list or (not is_cuda and cuda) or (is_cuda and len(cuda)!=torch.cuda.device_count()):raise ValueError('CUDA RNG device identity')
    for i,t in enumerate(cuda):
        if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.dtype!=torch.uint8 or t.ndim!=1:raise ValueError('CUDA RNG state')
        try:torch.Generator(device=f'cuda:{i}').set_state(t)
        except RuntimeError as exc:raise ValueError('CUDA RNG state') from exc
    return numpy_state


def _checkpoint(value,model,optimizer,cursor,binding):
    from .data_audit import canonical_bytes
    keys={'schema_version','task_id','model','optimizer','cursor','binding','selection','budget','rng'}
    if type(value) is not dict or set(value)!=keys or type(value['schema_version']) is not int or value['schema_version']!=1 or value['task_id']!='G2' or canonical_bytes(value['binding'])!=canonical_bytes(binding):raise ValueError('checkpoint identity/schema mismatch')
    _core(binding['effective_config']);_budget(value['budget'])
    proposed=EpochCursor(cursor.seed,cursor.example_count);proposed.load_state_dict(value['cursor']);_selection(value['selection'],proposed)
    current=model.state_dict();weights=value['model']
    if type(weights) is not dict or set(weights)!=set(current):raise ValueError('model state keys')
    for key,t in weights.items():
        if not isinstance(t,torch.Tensor) or t.dtype!=current[key].dtype or t.shape!=current[key].shape or not torch.isfinite(t).all():raise ValueError('model shape/dtype/finite')
    saved=value['optimizer'];template=optimizer.state_dict()
    if type(saved) is not dict or set(saved)!={'state','param_groups'} or type(saved['state']) is not dict or type(saved['param_groups']) is not list or len(saved['param_groups'])!=len(template['param_groups']):raise ValueError('optimizer schema')
    parameters={}
    for group,reference,live in zip(saved['param_groups'],template['param_groups'],optimizer.param_groups):
        if canonical_bytes(group)!=canonical_bytes(reference):raise ValueError('optimizer config/parameter identity')
        parameters.update(zip(group['params'],live['params']))
    expected=set(parameters) if value['budget']['completed_steps']>0 else set()
    if set(saved['state'])!=expected:raise ValueError('optimizer complete parameter moments required')
    for index,state in saved['state'].items():
        if type(state) is not dict or set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('optimizer moment schema')
        for key in ('exp_avg','exp_avg_sq'):
            t=state[key]
            if not isinstance(t,torch.Tensor) or t.shape!=parameters[index].shape or t.dtype!=parameters[index].dtype or not torch.isfinite(t).all():raise ValueError('optimizer moment shape/dtype/finite')
            if key=='exp_avg_sq' and (t<0).any():raise ValueError('optimizer variance must be nonnegative')
        t=state['step']
        if not isinstance(t,torch.Tensor) or t.numel()!=1 or not torch.isfinite(t).all() or float(t)!=value['budget']['completed_steps']:raise ValueError('optimizer exact step/budget conflict')
    numpy_state=_validate_rng(value['rng'],model)
    return proposed,numpy_state


def save_checkpoint(path,model,optimizer,cursor,*,binding,selection,budget):
    import os
    from .generative_data import _no_symlink
    from .data_audit import file_sha256
    value={'schema_version':1,'task_id':'G2','model':dict(model.state_dict()),'optimizer':optimizer.state_dict(),'cursor':cursor.state_dict(),'binding':binding,'selection':selection,'budget':budget,'rng':_rng(model)}
    _checkpoint(value,model,optimizer,cursor,binding)
    path=_no_symlink(path)
    # A killed writer can leave an unpublished candidate; only closed+hashed files
    # may enter a completed last/best receipt. No checkpoint is overwritten.
    with path.open('xb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    return file_sha256(path)


def load_checkpoint(path,expected_sha256,model,optimizer,cursor,*,expected_binding):
    import random,numpy as np,copy
    from .generative_data import _checked_file
    path=_checked_file(path,expected_sha256)
    try:value=torch.load(path,map_location='cpu',weights_only=True)
    except Exception as exc:raise ValueError('safe checkpoint decoding failed') from exc
    proposed,numpy_state=_checkpoint(value,model,optimizer,cursor,expected_binding)
    model.load_state_dict(value['model']);optimizer.load_state_dict(value['optimizer']);cursor.load_state_dict(proposed.state_dict())
    random.setstate(value['rng']['python']);np.random.set_state(numpy_state);torch.set_rng_state(value['rng']['torch_cpu'])
    if value['rng']['cuda']:torch.cuda.set_rng_state_all(value['rng']['cuda'])
    return CheckpointState(copy.deepcopy(value['binding']),copy.deepcopy(value['selection']),copy.deepcopy(value['budget']),cursor.state_dict(),True)
