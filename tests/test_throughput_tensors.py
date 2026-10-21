import unittest
import hashlib
import numpy as np
import torch
from vkgr.throughput_tensors import prepared_batches,cursor_prepared_batches
from vkgr.throughput_sequences import SequenceStore
from vkgr.throughput_config import Engineering
from test_throughput_sequences import EVENTS,REFS
from test_tensor_cache import g_builder,p_builder,Tokens,same

class WideTokens(Tokens):
    def __init__(self):self.table={i:[i+2,i+3]for i in range(32)}

IDENTITY=hashlib.sha256(b'synthetic fixed input/sid/config/source/runtime/constructor').hexdigest()


class Builder:
    def __init__(self,role):self.role=role
    def identity(self):return IDENTITY
    def validate_call(self,rows,args,kwargs):
        if kwargs:return False
        if self.role=='g':return len(args)==1 and args[0].table==WideTokens().table
        return args==({i:i+1 for i in range(32)},max(len(r['history'])for r in rows),'f3_f4_recency')
    def build(self,rows):
        if self.role=='g':return g_builder(rows,WideTokens())
        return p_builder(rows,{i:i+1 for i in range(32)},max(len(r['history'])for r in rows),'f3_f4_recency')


def g_factory():return Builder('g')
def p_factory():return Builder('p')


class TensorWorkerTests(unittest.TestCase):
    def test_original_tensor_bytes_microbatch_padding_tail_and_order(self):
        store=SequenceStore.from_events(EVENTS,REFS)
        for role,factory in (('g',g_factory),('p',p_factory)):
            for workers in (1,2,8):
                prepared=list(prepared_batches(store,[[2,0],[1]],Engineering(assembly_workers=workers),
                    factory=factory,namespace=IDENTITY,role=role,microbatch=1 if role=='g' else 512))
                self.assertEqual([r['example_id']for r in prepared[0].rows],[2,0])
                for batch in prepared:
                    for rows,arrays in batch.chunks:
                        same(tuple(torch.from_numpy(a)for a in arrays),factory().build(rows))

    def test_constructor_binding_keeps_step_code_and_main_order(self):
        store=SequenceStore.from_events(EVENTS,REFS)
        batch=next(prepared_batches(store,[[0,1]],Engineering(assembly_workers=2),factory=p_factory,namespace=IDENTITY,role='p'))
        def old_step(rows):return _tensors2(rows,{i:i+1 for i in range(32)},max(len(r['history'])for r in rows),'f3_f4_recency')
        old_step.__globals__['_tensors2']=p_builder
        bound=batch.bind(old_step,'_tensors2',namespace=IDENTITY)
        self.assertIs(bound.__code__,old_step.__code__)
        same(bound(batch.rows),old_step(batch.rows))
        with self.assertRaises(ValueError):batch.bind(old_step,'_tensors2',namespace='wrong')
        with self.assertRaises(ValueError):bound(batch.rows)
        wrong=batch.bind(old_step,'_tensors2',namespace=IDENTITY)
        with self.assertRaises(ValueError):wrong(list(reversed(batch.rows)))
        changed=batch.bind(old_step,'_tensors2',namespace=IDENTITY)
        old_step.__globals__['_tensors2']=p_builder
        # Same rows but changed mapping/width must never reuse prepared tensors.
        def wrong_step(rows):return _tensors2(rows,{},99,'id')
        with self.assertRaises(ValueError):batch.bind(wrong_step,'_tensors2',namespace=IDENTITY)(batch.rows)

    def test_g_multiple_constructor_calls_and_cursor_tail(self):
        import random
        class Cursor:
            def __init__(self):self.offset=0;self.size=3;self.rng=random.Random(42);self.order=[2,0,1]
            def next(self,n):
                end=min(self.size,self.offset+n);out=self.order[self.offset:end];self.offset=end;self.rng.random();return out
        def step(rows,tokens):return [_g([row],tokens)for row in rows]
        step.__globals__['_g']=g_builder
        store=SequenceStore.from_events(EVENTS,REFS)
        for workers in (1,2,8):
            actual,expected=Cursor(),Cursor()
            for batch in cursor_prepared_batches(store,actual,Engineering(assembly_workers=workers,prefetch=2),
                    batch_size=2,factory=g_factory,namespace=IDENTITY,role='g',microbatch=1):
                indices=expected.next(2)
                self.assertEqual([r['example_id']for r in batch.rows],indices)
                self.assertEqual(actual.rng.getstate(),expected.rng.getstate())
                bound=batch.bind(step,'_g',namespace=IDENTITY)
                for left,right in zip(bound(batch.rows,WideTokens()),step(batch.rows,WideTokens())):same(left,right)

    def test_prepared_rows_cannot_silently_change_tensor_inputs(self):
        store=SequenceStore.from_events(EVENTS,REFS)
        batch=next(prepared_batches(store,[[0,1]],Engineering(),factory=p_factory,namespace=IDENTITY,role='p'))
        def step(rows):return _tensors2(rows,{i:i+1 for i in range(32)},max(len(r['history'])for r in rows),'f3_f4_recency')
        step.__globals__['_tensors2']=p_builder
        batch.rows[0]['side'][0][0]=99
        with self.assertRaises(ValueError):batch.bind(step,'_tensors2',namespace=IDENTITY)(batch.rows)

    def test_memory_budget_alone_requires_output_contract(self):
        store=SequenceStore.from_events(EVENTS,REFS)
        with self.assertRaises(ValueError):
            next(prepared_batches(store,[[0]],Engineering(memory_limit_bytes=10**12),factory=p_factory,namespace=IDENTITY,role='p'))

    def test_factory_identity_and_output_reservation(self):
        store=SequenceStore.from_events(EVENTS,REFS)
        with self.assertRaises(ValueError):
            next(prepared_batches(store,[[0]],Engineering(),factory=p_factory,namespace='0'*64,role='p'))
        with self.assertRaises(ValueError):
            next(prepared_batches(store,[[0]],Engineering(max_in_flight_bytes=100),factory=p_factory,namespace=IDENTITY,role='p'))
        with self.assertRaises(MemoryError):
            next(prepared_batches(store,[[0]],Engineering(max_in_flight_bytes=100),factory=p_factory,namespace=IDENTITY,role='p',tensor_bytes_bound=lambda n,w:100))
        from vkgr.throughput_workers import rng_state
        before=rng_state()
        list(prepared_batches(store,[[2,0],[1]],Engineering(),factory=p_factory,namespace=IDENTITY,role='p'))
        self.assertEqual(rng_state(),before)


if __name__=='__main__':unittest.main()
