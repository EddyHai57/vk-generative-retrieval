import hashlib
import random
import unittest
import ast
import subprocess
from concurrent.futures.process import BrokenProcessPool
import sys
import types
from types import SimpleNamespace
from pathlib import Path
from vkgr.throughput_config import Engineering
from vkgr.throughput_decode import decode_queries


class Session:
    def __init__(self):self.weight=7
    def identity(self):return hashlib.sha256(str(self.weight).encode()).hexdigest()
    def token(self):return self.weight
    def decode(self,query):return ([self.weight+sum(query['history'])],{'batch':1})


def factory():return Session()


def changed_factory():
    value=Session();value.weight=8;return value


class Stochastic(Session):
    def decode(self,query):random.random();return super().decode(query)


def random_factory():return Stochastic()


class BeamSession(Session):
    depth=2
    def __init__(self):
        super().__init__()
        import torch
        torch.set_num_threads(1)
        source=(Path(__file__).parent/'fixtures/throughput_frozen_beam.py').read_text()
        tree=ast.parse(source);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='constrained_beam_d')
        ns={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),'owned_frozen_beam','exec'),ns)
        self.beam=ns['constrained_beam_d'];self.codec=SimpleNamespace(offsets=[2,4,6],vocab_size=8)
    def allowed_next(self,path):return [0,1]
    def decode_path(self,path):return 10+path[0]*4+path[1]*2+path[2]
    def decode(self,query):
        import torch
        catalog=SimpleNamespace(depth=2,allowed_next=self.allowed_next,decode=self.decode_path)
        def score(paths):return torch.arange(8,dtype=torch.float32).repeat(len(paths),1)+sum(query['history'])/100
        return self.beam(catalog,self.codec,score,beam=100)


def beam_factory():return BeamSession()


class SameIdentityOtherType(Session):pass
def wrong_type_factory():return SameIdentityOtherType()


def failing_factory():raise RuntimeError('synthetic initializer failure')


class TinyModel:
    def __new__(cls):
        import torch
        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.tensor([2.0]))
        return _Model()


def tiny_model_with_gain(gain=1.0):
    import torch
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor([2.0]));self.gain=gain
    return _Model()


class TinyCodec:
    __slots__=('catalog','_frozen')
    def __init__(self):
        import numpy as np
        object.__setattr__(self,'catalog',SimpleNamespace(ids=np.array([1,2]),sids=np.array([[3,4],[5,6]])))
        self.catalog.ids.flags.writeable=False;self.catalog.sids.flags.writeable=False
        object.__setattr__(self,'_frozen',True)
    def __setattr__(self,name,value):
        if getattr(self,'_frozen',False):raise AttributeError('immutable codec')
        object.__setattr__(self,name,value)
    def decode(self,history):return tuple(history)


class TinyCodecVariant(TinyCodec):
    def decode(self,history):return tuple(reversed(history))


def tiny_callback(model,codec,history,*,beam):return (float(model.weight.item()),codec.decode(history),beam)


def tiny_torch_factory():
    import torch
    torch.set_num_threads(1)
    from vkgr.throughput_decode import TorchSession
    return TorchSession(TinyModel(),TinyCodec(),tiny_callback,beam=100)


class _SyntheticValidatedBase:
    _arrays=('ids','sids','keys','key_items')
    def check_identity(self):
        if self.catalog is not self._model:raise ValueError('model alias changed')
        for name,ref,metadata in zip(self._arrays,self._references,self._array_metadata):
            value=getattr(self.catalog,name)
            if value is not ref or value.flags.writeable or (value.shape,value.dtype.str,value.nbytes,value.strides,value.ctypes.data)!=metadata:
                raise ValueError('array identity changed')


class ValidatedCodecD(_SyntheticValidatedBase):
    def __init__(self,*,semantic_change=False):
        import numpy as np
        from types import SimpleNamespace
        arrays={
            'ids':np.frombuffer(np.array([1,2],dtype=np.uint32).tobytes(),dtype=np.uint32),
            'sids':np.frombuffer(np.array([[0,0,0],[1,0,0]],dtype=np.uint32).tobytes(),dtype=np.uint32).reshape(2,3),
            'keys':np.frombuffer(np.array([0,4],dtype=np.uint64).tobytes(),dtype=np.uint64),
            'key_items':np.frombuffer(np.array([1,2],dtype=np.uint32).tobytes(),dtype=np.uint32),
        }
        self.catalog=SimpleNamespace(**arrays,branching=2,depth=2,suffix_vocabulary=1,bases=[2,2,1],nodes=[])
        self.offsets=[2,4,6];self.vocab_size=7;self.depth=2
        self._model=self.catalog
        self._references=[getattr(self.catalog,name)for name in self._arrays]
        self._array_metadata=[(v.shape,v.dtype.str,v.nbytes,v.strides,v.ctypes.data)for v in self._references]
        if semantic_change:
            self.catalog.branching=3;self.catalog.bases=[3,3,1];self.offsets=[2,5,8];self.vocab_size=9
        self._namespace=repr((self.catalog.branching,self.catalog.depth,self.catalog.suffix_vocabulary,self.catalog.bases,self.offsets,self.vocab_size)).encode()
    def _namespace_bytes(self):
        return repr((self.catalog.branching,self.catalog.depth,self.catalog.suffix_vocabulary,self.catalog.bases,self.offsets,self.vocab_size)).encode()
    def decode(self,history):return tuple(history)


# Exercise the production special case against a same-shape class that carries
# the exact original module/qualified-name identity in this isolated test tree.
_sid_depth_module=types.ModuleType('vkgr.sid_depth');_sid_depth_module.__file__=__file__;_sid_depth_module.ValidatedCodecD=ValidatedCodecD
_generative_data_module=types.ModuleType('vkgr.generative_data');_generative_data_module.__file__=__file__;_generative_data_module._ValidatedCodec=_SyntheticValidatedBase
sys.modules['vkgr.sid_depth']=_sid_depth_module;sys.modules['vkgr.generative_data']=_generative_data_module
import vkgr
vkgr.sid_depth=_sid_depth_module;vkgr.generative_data=_generative_data_module
ValidatedCodecD.__module__='vkgr.sid_depth'


def aliased_torch_factory():
    import torch
    torch.set_num_threads(1)
    from vkgr.throughput_decode import TorchSession
    return TorchSession(TinyModel(),ValidatedCodecD(),tiny_callback,beam=100)


class DecodeTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
    def test_model_class_attribute_changes_session_identity(self):
        a=tiny_torch_factory()
        cls=type(a.model)
        cls.runtime_gain=1
        try:
            before=a.identity()
            cls.runtime_gain=2
            self.assertNotEqual(before,a.identity())
        finally:
            delattr(cls,'runtime_gain')

    def test_owned_frozen_batch1_beam_on_synthetic_catalog(self):
        session=BeamSession();queries=[{'history':[i,i+1]}for i in range(4)]
        baseline=[session.decode(q)for q in queries]
        for n in (1,2,3):
            self.assertEqual(decode_queries(queries,session.decode,Engineering(decode_workers=n),session_factory=beam_factory,state_sha=session.identity()),baseline)
    def test_order_identity_batch1_and_rng_counts(self):
        session=Session();q=[{'history':[i,i+1]}for i in range(7)];expected=[session.decode(x)for x in q]
        before=random.getstate()
        for n in (1,2,3):
            self.assertEqual(decode_queries(q,session.decode,Engineering(decode_workers=n),session_factory=factory,state_sha=session.identity()),expected)
        self.assertEqual(before,random.getstate())

    def test_rng_and_unmeasured_gpu_parallel_rejected(self):
        session=Stochastic()
        with self.assertRaises(ValueError):decode_queries([{'history':[1]}],session.decode,Engineering(decode_workers=2),session_factory=random_factory,state_sha=session.identity())
        with self.assertRaises(ValueError):decode_queries([{'history':[1]}],session.decode,Engineering(decode_workers=2),session_factory=factory,state_sha=session.identity(),device='cuda')
        with self.assertRaises(ValueError):decode_queries([{'history':[1]}],session.decode,Engineering(decode_workers=2),session_factory=factory,state_sha='wrong')

    def test_parallel_identity_is_bound_to_original_decoder_owner(self):
        session=Session();other=Session();other.weight=8
        with self.assertRaisesRegex(ValueError,'original decoder identity'):
            decode_queries([{'history':[1]}],session.decode,Engineering(decode_workers=2),session_factory=factory,state_sha=other.identity())

    def test_result_bound_must_be_positive_integer_before_workers_start(self):
        session=Session()
        for bound in (0,-1,1.5,True):
            with self.subTest(bound=bound),self.assertRaisesRegex(ValueError,'positive integer result bound'):
                decode_queries([{'history':[1]}],session.decode,Engineering(decode_workers=2),session_factory=factory,state_sha=session.identity(),result_bound=bound)

    def test_cuda_requires_context_and_live_receipt_fields(self):
        session=Session()
        from vkgr.throughput_decode import VRAMReceipt
        bad=[VRAMReceipt(1,100,False,True),VRAMReceipt(1,100,True,False),VRAMReceipt(0,100,True,True),VRAMReceipt(-1,100,True,True),VRAMReceipt(float('nan'),100,True,True),None]
        for receipt in bad:
            with self.subTest(receipt=receipt),self.assertRaises(ValueError):
                decode_queries([{'history':[1]}],session.decode,Engineering(decode_workers=2),session_factory=factory,state_sha=session.identity(),device='cuda',vram_receipt=receipt)

    def test_torch_session_identity_covers_model_callback_codec_and_runtime(self):
        import numpy as np
        import torch

        from vkgr.throughput_decode import TorchSession,InitializerBudget
        session=TorchSession(TinyModel(),TinyCodec(),tiny_callback,beam=100)
        factory_fn=tiny_torch_factory
        self.assertEqual(session.identity(),factory_fn().identity())
        changed=TinyModel();changed.weight.data.add_(1)
        self.assertNotEqual(session.identity(),TorchSession(changed,TinyCodec(),tiny_callback,beam=100).identity())
        self.assertNotEqual(session.identity(),TorchSession(TinyModel(),TinyCodecVariant(),tiny_callback,beam=100).identity())
        extra=TinyCodec();extra.catalog.namespace_marker='changed'
        self.assertNotEqual(session.identity(),TorchSession(TinyModel(),extra,tiny_callback,beam=100).identity())
        budget=InitializerBudget.for_session(session,estimated_init_bytes=session.storage_bytes(),budget_bytes_per_worker=session.storage_bytes()*2)
        self.assertEqual(decode_queries([{'history':[7,8]}],session.decode,Engineering(decode_workers=2),session_factory=factory_fn,state_sha=session.identity(),initializer_budget=budget),[(2.0,(7,8),100)])

    def test_torch_session_identity_covers_plain_attributes_device_and_hooks(self):
        import torch
        from vkgr.throughput_decode import TorchSession
        base=TorchSession(tiny_model_with_gain(1.0),TinyCodec(),tiny_callback,beam=100)
        changed_gain=TorchSession(tiny_model_with_gain(2.0),TinyCodec(),tiny_callback,beam=100)
        self.assertNotEqual(base.identity(),changed_gain.identity())

        meta_model=tiny_model_with_gain(1.0).to(device='meta')
        changed_device=TorchSession(meta_model,TinyCodec(),tiny_callback,beam=100)
        self.assertNotEqual(base.identity(),changed_device.identity())

        hooked_model=tiny_model_with_gain(1.0)
        hooked_model.register_forward_hook(lambda module, inputs, output: output)
        changed_hooks=TorchSession(hooked_model,TinyCodec(),tiny_callback,beam=100)
        self.assertNotEqual(base.identity(),changed_hooks.identity())

        unsupported=tiny_model_with_gain(1.0)
        unsupported.runtime_marker=object()
        with self.assertRaisesRegex(ValueError,'unsupported decoder/codec namespace value'):
            TorchSession(unsupported,TinyCodec(),tiny_callback,beam=100)

    def test_validated_codec_alias_pointer_metadata_is_process_independent(self):
        import torch
        from vkgr.throughput_decode import TorchSession
        first=ValidatedCodecD();second=ValidatedCodecD()
        self.assertNotEqual(first._array_metadata[0][-1],second._array_metadata[0][-1])
        one=TorchSession(TinyModel(),first,tiny_callback,beam=100)
        two=TorchSession(TinyModel(),second,tiny_callback,beam=100)
        self.assertEqual(one.identity(),two.identity())
        self.assertEqual(one.storage_bytes(),4+8+24+16+8)
        semantic=TorchSession(TinyModel(),ValidatedCodecD(semantic_change=True),tiny_callback,beam=100)
        self.assertNotEqual(one.identity(),semantic.identity())
        result=decode_queries([{'history':[7,8]}],one.decode,Engineering(decode_workers=2),session_factory=aliased_torch_factory,state_sha=one.identity(),initializer_budget=__import__('vkgr.throughput_decode',fromlist=['InitializerBudget']).InitializerBudget.for_session(one,estimated_init_bytes=one.storage_bytes(),budget_bytes_per_worker=one.storage_bytes()*2))
        self.assertEqual(result,[(2.0,(7,8),100)])

    def test_worker_factory_type_must_match_original_owner(self):
        session=Session()
        with self.assertRaises(BrokenProcessPool):
            decode_queries([{'history':[1]}],session.decode,Engineering(decode_workers=2),session_factory=wrong_type_factory,state_sha=session.identity())

    def test_same_factory_type_wrong_weights_are_rejected(self):
        session=Session()
        with self.assertRaises(BrokenProcessPool):
            decode_queries([{'history':[1]}],session.decode,Engineering(decode_workers=2),session_factory=changed_factory,state_sha=session.identity())

    def test_init_budget_is_checked_before_factory_runs_and_factory_error_propagates(self):
        import torch
        from vkgr.throughput_decode import TorchSession,InitializerBudget
        session=TorchSession(TinyModel(),TinyCodec(),tiny_callback,beam=100);called=[]
        def too_large_factory():called.append(True);return tiny_torch_factory()
        floor=session.storage_bytes()
        unbounded=InitializerBudget(session.identity(),floor,floor-1)
        with self.assertRaisesRegex(ValueError,'initializer budget identity'):
            decode_queries([{'history':[1]}],session.decode,Engineering(decode_workers=2),session_factory=too_large_factory,state_sha=session.identity(),initializer_budget=unbounded)
        self.assertEqual(called,[])

        with self.assertRaises(BrokenProcessPool):
            decode_queries([{'history':[1]}],Session().decode,Engineering(decode_workers=2),session_factory=failing_factory,state_sha=Session().identity())


if __name__=='__main__':unittest.main()
