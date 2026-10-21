import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
import hashlib
from types import FunctionType,SimpleNamespace
from unittest.mock import patch
import numpy as np
from vkgr.throughput_config import Engineering
from vkgr.throughput_exact import exact_topk


class ExactTests(unittest.TestCase):
    def test_scores_predictions_ties_and_tail_blocks(self):
        source=(Path(__file__).parent/'fixtures/throughput_frozen_baselines.py').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),'ec92bb0b9858e9da7e192f8cc32da96d5a274ad195413220695b093361b96a0b')
        old={};exec(compile(source,'frozen_baselines','exec'),old)
        rng=np.random.default_rng(42);q=rng.normal(size=(35,64)).astype('f4');items=rng.normal(size=(4103,64)).astype('f4');items[:4]=0;ids=np.arange(len(items),dtype='u4')[::-1].copy()
        expected_scores=[]
        def einsum(*a,**kw):
            result=np.einsum(*a,**kw);expected_scores.append((result.shape,hashlib.sha256(result.tobytes()).hexdigest()));return result
        proxy=SimpleNamespace(**{**vars(np),'einsum':einsum})
        original=old['blockwise_topk'];observe=FunctionType(original.__code__,{**original.__globals__,'np':proxy},original.__name__,original.__defaults__)
        baseline=observe(q,items,ids,20)
        scores=None
        for n in (1,2,8):
            pred,trace=exact_topk(q,items,ids,20,Engineering(eval_workers=n),trace=True)
            self.assertEqual(pred.tobytes(),baseline.tobytes())
            if scores is None:scores=trace
            self.assertEqual(trace,scores)
            self.assertEqual([x for block in trace for x in block],expected_scores)
        zeros=np.zeros((3,64),dtype='f4')
        pred,_=exact_topk(zeros,items[:4],ids[:4],3,Engineering(eval_workers=2),trace=True)
        self.assertEqual(pred[0].tolist(),sorted(ids[:4])[:3])

    def test_mmap_catalog_and_limits(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t).resolve()/'items.npy';np.save(p,np.eye(8,64,dtype='f4'))
            items=np.load(p,mmap_mode='r');q=np.ones((2,64),dtype='f4');ids=np.arange(8,dtype='u4')
            a,_=exact_topk(q,items,ids,4,Engineering(eval_workers=2,use_mmap=True))
            b,_=exact_topk(q,np.asarray(items),ids,4,Engineering())
            self.assertEqual(a.tobytes(),b.tobytes())
            with self.assertRaises(ValueError):exact_topk(q,np.asarray(items),ids,4,Engineering(eval_workers=2,use_mmap=True))

    def test_non_float32_mmap_budget_is_guarded_before_cast(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t).resolve()/'items.npy';np.save(p,np.eye(8,64,dtype='f8'))
            items=np.load(p,mmap_mode='r');q=np.ones((2,64),dtype='f4');ids=np.arange(8,dtype='u4')
            events=[];asarray=np.asarray
            def guarded(config,directory,*,projected_memory=0,projected_disk=0):
                events.append(('guard',projected_memory))
                self.assertGreaterEqual(projected_memory,items.size*4)
                raise MemoryError('pre-allocation budget')
            def observed_asarray(value,*args,**kwargs):
                if value is items:events.append(('catalog_cast',0))
                return asarray(value,*args,**kwargs)
            with patch('vkgr.throughput_exact.guard',side_effect=guarded),patch('vkgr.throughput_exact.np.asarray',side_effect=observed_asarray):
                with self.assertRaisesRegex(MemoryError,'pre-allocation budget'):
                    exact_topk(q,items,ids,4,Engineering(eval_workers=1))
            self.assertEqual([event[0]for event in events],['guard'])

    def test_parallel_mmap_rejects_non_float32_and_shifted_mapping(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t).resolve()/'items.npy';np.save(p,np.eye(8,64,dtype='f8'))
            items=np.load(p,mmap_mode='r');q=np.ones((2,64),dtype='f4');ids=np.arange(8,dtype='u4')
            with self.assertRaisesRegex(ValueError,'float32'):
                exact_topk(q,items,ids,4,Engineering(eval_workers=2,use_mmap=True))
            np.save(p,np.eye(8,64,dtype='f4'));items=np.load(p,mmap_mode='r')
            with self.assertRaisesRegex(ValueError,'full .npy mmap'):
                exact_topk(q,items[1:],ids[1:],4,Engineering(eval_workers=2,use_mmap=True))

    def test_default_single_worker_preserves_sliced_mmap_values(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t).resolve()/'items.npy';np.save(p,np.eye(8,64,dtype='f4'))
            items=np.load(p,mmap_mode='r');q=np.ones((2,64),dtype='f4');ids=np.arange(8,dtype='u4')
            pred,_=exact_topk(q,items[1:],ids[1:],4,Engineering())
            source=(Path(__file__).parent/'fixtures/throughput_frozen_baselines.py').read_bytes()
            old={};exec(compile(source,'frozen_baselines','exec'),old)
            expected=old['blockwise_topk'](q,np.asarray(items[1:]),ids[1:],4)
            self.assertEqual(pred.tobytes(),expected.tobytes())


if __name__=='__main__':unittest.main()
