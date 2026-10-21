"""CPU synthetic fixture only, running the owned frozen G/P step implementations.

No dataset, checkpoint, query evaluation or scientific experiment is read/run.
Original source trees must be locally materialized from the registered Git refs.
"""
import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys

from analysis.tensor_cache_parity import freeze, observe_p_inputs
from vkgr.tensor_cache import bind_step,namespace as constructor_namespace
from vkgr.throughput_config import Engineering
from vkgr.throughput_sequences import SequenceStore,position
from vkgr.throughput_tensors import cursor_prepared_batches
from vkgr.throughput_workers import cursor_rows


def _package(name,root):
    spec=importlib.util.spec_from_file_location(name,Path(root)/'src/vkgr/__init__.py',submodule_search_locations=[str(Path(root)/'src/vkgr')])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return lambda suffix:importlib.import_module(name+'.'+suffix)


def fixture():
    events=[{'user_id':u,'item_id':i,'position':[w,o],'like':like,'side':[1,i,3],'ctx':[2,1]}
            for u,i,w,o,like in [(2,10,0,0,True),(1,20,0,1,True),(2,11,0,2,False),
                                (2,12,1,0,True),(1,21,1,1,True),(2,13,1,2,True)]]
    refs=[(2,[1,0]),(1,[1,1]),(2,[1,2])];rows=[]
    for j,(u,p)in enumerate(refs):
        before=[e for e in events if e['user_id']==u and e['position']<p]
        target=next(e for e in events if e['user_id']==u and e['position']==p)
        rows.append({'example_id':j,'user_id':u,'target':target['item_id'],'target_position':p,
                     'history':[e['item_id']for e in before[-50:]],'history_positions':[e['position']for e in before[-50:]],
                     'past_likes':sorted({e['item_id']for e in before if e['like']}),
                     'side':[e['side']for e in before[-50:]],'ctx':[e['ctx']for e in before[-50:]]})
    return events,refs,rows


def _tokens_identity(tokens):
    """Stable identity for the immutable G tokenizer context passed to workers."""
    catalog=tokens.catalog
    digest=hashlib.sha256()
    for value in (catalog.ids,catalog.sids):
        digest.update(value.dtype.str.encode());digest.update(json.dumps(list(value.shape)).encode());digest.update(value.tobytes())
    return {'catalog_sha256':digest.hexdigest(),'offsets':list(tokens.offsets),'vocab_size':tokens.vocab_size,
            'max_len':tokens.max_len,'depth':catalog.depth}


class _ConstructorFactory:
    """Spawn-picklable access to only a frozen pure constructor and read-only inputs."""
    def __init__(self,branch,source,constructor_sha,namespace,pins,*,tokens=None,idmap=None,feature=None):
        self.branch=branch;self.source=str(Path(source).resolve());self.constructor_sha=constructor_sha
        self.namespace=namespace;self.pins=pins
        if branch=='g':
            catalog=tokens.catalog
            self.token_descriptor={'ids':catalog.ids.copy(),'sids':catalog.sids.copy(),
                'branching':catalog.branching,'depth':catalog.depth,'max_len':tokens.max_len}
        else:self.token_descriptor=None
        self.idmap=dict(idmap)if idmap is not None else None;self.feature=feature

    def __call__(self):
        load=_package('frozen_'+self.branch,self.source)
        if self.branch=='g':
            constructor=load('generative_data_v2').sid_targets_l
            catalog=load('sid').SIDCatalog(self.token_descriptor['ids'],self.token_descriptor['sids'],
                self.token_descriptor['branching'],self.token_descriptor['depth'],[])
            codec=load('sid_depth').ValidatedCodecD(catalog,self.token_descriptor['depth'])
            tokens=load('sid_depth').SIDTokensLD(codec,self.token_descriptor['max_len'])
        else:constructor=load('retrieval_feat02_run')._tensors2
        if hashlib.sha256(inspect.getsource(constructor).encode()).hexdigest()!=self.constructor_sha:
            raise ValueError('frozen constructor source identity changed')
        return _WorkerConstructor(self,constructor,tokens if self.branch=='g' else None)


class _WorkerConstructor:
    def __init__(self,factory,constructor,tokens=None):self.factory=factory;self.constructor=constructor;self.tokens=tokens
    def identity(self):
        context=_tokens_identity(self.tokens) if self.factory.branch=='g' else {'idmap':sorted(self.factory.idmap.items()),'feature':self.factory.feature}
        sid=hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return constructor_namespace(**{**self.factory.pins,'sid':sid})
    def build(self,rows):
        if self.factory.branch=='g':return self.constructor(rows,self.tokens)
        width=max(len(r['history'])for r in rows)
        return self.constructor(rows,self.factory.idmap,width,self.factory.feature)
    def validate_call(self,rows,args,kwargs):
        f=self.factory
        if kwargs:return False
        if f.branch=='g':
            return len(args)==1 and _tokens_identity(args[0])==_tokens_identity(self.tokens)
        if len(args)!=3:return False
        idmap,width,feature=args
        return idmap==f.idmap and type(width)is int and width==max(len(r['history'])for r in rows) and feature==f.feature


def _factory(branch,source,builder,*,config,input_rows,tokens=None,idmap=None,feature=None):
    constructor_sha=hashlib.sha256(inspect.getsource(builder).encode()).hexdigest()
    context=_tokens_identity(tokens) if branch=='g' else {'idmap':sorted(idmap.items()),'feature':feature}
    import numpy as np
    import torch
    def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    source_ledger={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((Path(source)/'src').rglob('*.py'))}
    pins={'input':digest(input_rows),'sid':digest(context),'config':digest(config),
          'source':digest(source_ledger),'runtime':digest([sys.version,np.__version__,torch.__version__,
          torch.version.cuda,torch.get_num_threads(),torch.get_num_interop_threads(),torch.are_deterministic_algorithms_enabled()]),
          'constructor':constructor_sha}
    namespace=constructor_namespace(**pins)
    return _ConstructorFactory(branch,source,constructor_sha,namespace,pins,tokens=tokens,idmap=idmap,feature=feature)


def run(branch,source):
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    load=_package('frozen_'+branch,source);events,refs,oldrows=fixture()
    store=SequenceStore.from_columns(np.array([e['user_id']for e in events]),np.array([e['item_id']for e in events]),
        np.array([position(e['position'])for e in events]),np.array([e['like']for e in events]),
        np.array([e['side']for e in events]),np.array([e['ctx']for e in events]),[(u,position(p))for u,p in refs])
    catalog=sorted({e['item_id']for e in events});config=json.loads((Path(source)/('configs/sid_structure/g-structure-a1-s42.json'if branch=='g'else'configs/seqm1/p-seqm1-mean-s42.json')).read_text())
    if branch=='g':
        sid=load('sid');depth=load('sid_depth');gm=load('generative_g6_model');ga=load('generative_arch')
        cat=sid.fit(np.array(catalog),np.zeros((len(catalog),64),dtype=np.float32),branching=48,depth=3,max_iter=2)
        codec=depth.ValidatedCodecD(cat,3);tokens=depth.SIDTokensLD(codec,50)
    else:
        runner=load('retrieval_seqm1_run');feat=load('retrieval_feat02_run');b1=load('retrieval_b1_run')
        train=load('retrieval_training');negative=load('retrieval_negatives')
        data={'idmap':{item:i+1 for i,item in enumerate(catalog)},'content':torch.zeros(len(catalog)+1,64),
              'supported_indices':list(range(1,len(catalog)+1)),'support':set(catalog)}
    results=[]
    for workers in (0,1,2,8):
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        trace={}
        if branch=='g':
            model=gm.make_model_arch_d(codec,config,device='cpu');opt=ga.make_optimizer_arch(model,config)
            original_step=gm.train_batch_d
            original_constructor=gm.sid_targets_l
            constructor_factory=_factory(branch,source,original_constructor,config=config,input_rows=oldrows,tokens=tokens)
            def builder(*a):
                value=gm.sid_targets_l(*a);trace.setdefault('tensors',[]).append(freeze(value));return value
            call=bind_step(gm.train_batch_d,'sid_targets_l',builder)
        else:
            model=runner._model(data,config,'cpu');opt=runner.make_optimizer(model,config)
            original_step=feat.train_step
            original_constructor=feat._tensors2
            constructor_factory=_factory(branch,source,original_constructor,config=config,input_rows=oldrows,idmap=data['idmap'],feature=config['feature'])
            def builder(*a):
                value=feat._tensors2(*a);trace.setdefault('tensors',[]).append(freeze(value));return value
            call=observe_p_inputs(bind_step(feat.train_step,'_tensors2',builder),trace)
        def before_step(o,a,kw):trace['gradients']=freeze({n:p.grad for n,p in model.named_parameters()});trace['optimizer_before']=freeze(o.state_dict())
        handle=opt.register_step_pre_hook(before_step)
        cursor=load('generative_training').EpochCursor(42,3)if branch=='g'else train.EpochCursor(3,42)
        method='next_batch'if branch=='g'else'next'
        prepared=None
        if workers==0:rows=[oldrows[i]for i in getattr(cursor,method)(3)]
        else:
            tensor_config=Engineering(assembly_workers=workers,prefetch=2)
            kwargs={'factory':constructor_factory,'namespace':constructor_factory.namespace,'role':branch,
                    'microbatch':config['microbatch'] if branch=='g' else 512,'length':50}
            prepared=next(cursor_prepared_batches(store,cursor,tensor_config,batch_size=3,method=method,**kwargs))
            rows=prepared.rows
            call=prepared.bind(original_step,'sid_targets_l' if branch=='g' else '_tensors2',namespace=constructor_factory.namespace)
            if branch=='p':call=observe_p_inputs(call,trace)
        if call.__code__ is not original_step.__code__:raise ValueError('frozen train-step code object changed')
        rng=random.Random(123)
        trace['before']=freeze({'model':model.state_dict(),'optimizer':opt.state_dict(),'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cursor':vars(cursor),'negative':rng})
        if branch=='g':result=call(model,opt,rows,codec,tokens,config,'cpu')
        else:result=call(model,opt,rows,data,train.SamplerStats(),0,config,rng,None,negative.target_frequency(oldrows))
        if prepared is not None:
            import torch
            trace['tensors']=[freeze(tuple(torch.from_numpy(a)for a in arrays))for _,arrays in prepared.chunks]
        handle.remove()
        trace['after']=freeze({'result':vars(result)if hasattr(result,'__dict__')else result,'model':model.state_dict(),'optimizer':opt.state_dict(),'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cursor':vars(cursor),'negative':rng})
        results.append(trace)
    if any(r!=results[0]for r in results[1:]):raise ValueError('tensor/parameter/gradient/optimizer/loss/RNG parity mismatch')
    return {'branch':branch,'fixture':'synthetic-only','baseline_workers':0,'workers':[1,2,8],
            'exact_step_state':True,'train_step_code_unchanged':True,'real_anchor':'NOT_RUN','quality_metric':False}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=('g','p'),required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
    result=run(a.branch,a.source)
    with Path(a.output).open('x')as f:json.dump(result,f,sort_keys=True)
    print(json.dumps(result));return 0


if __name__=='__main__':raise SystemExit(main())
