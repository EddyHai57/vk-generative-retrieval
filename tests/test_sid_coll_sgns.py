"""Independent hand oracles for S1 SGNS."""
import math
import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from vkgr.sid_coll_sgns import (SGNSConfig, fit_sgns, iter_positive_pairs,
    learning_rate, sgns_loss, sample_negatives, health_check, save_sgns_output)

class SGNSTest(unittest.TestCase):
    def test_order_window_repeats_and_user_boundary(self):
        self.assertEqual(list(iter_positive_pairs({9:[3,1],2:[1,2,1]},1)),[(1,2),(2,1),(2,1),(1,2),(3,1),(1,3)])
        self.assertEqual(list(iter_positive_pairs({1:[1,1]},5)),[])

    def test_symmetric_fixed_window(self):
        self.assertEqual(list(iter_positive_pairs({1:[1,2,3]},5)),[(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)])

    def test_loss_sum_hand_oracle(self):
        c=torch.tensor([[1.,0.],[1.,0.]],requires_grad=True)
        p=torch.zeros((2,2),requires_grad=True)
        n=torch.zeros((2,5,2),requires_grad=True)
        loss=sgns_loss(c,p,n)
        self.assertAlmostEqual(loss.item(),12*math.log(2),places=5)
        loss.backward()
        np.testing.assert_allclose(p.grad.numpy(),[[-.5,0],[-.5,0]])
        np.testing.assert_allclose(n.grad.numpy()[:,:,0],.5)

    def test_lr_linear_positive_count(self):
        self.assertEqual(learning_rate(0,100),.025)
        self.assertAlmostEqual(learning_rate(50,100),(.025+.0000025)/2)
        self.assertAlmostEqual(learning_rate(100,100),.0000025)

    def test_negatives_exclude_center_positive_allow_repeats_continuous(self):
        out=sample_negatives(np.array([0,1]),np.array([1,0]),np.array([.2,.3,.5]),5,np.random.default_rng(42))
        self.assertEqual(out.tolist(),[[2]*5,[2]*5])
        rng=np.random.default_rng(42)
        whole=sample_negatives(np.array([0]*4),np.array([1]*4),np.array([.2,.3,.25,.25]),5,rng)
        rng=np.random.default_rng(42)
        split=np.concatenate([sample_negatives(np.array([0]*2),np.array([1]*2),np.array([.2,.3,.25,.25]),5,rng) for _ in range(2)])
        np.testing.assert_array_equal(whole,split)

    def test_negative_distribution_sum_is_not_per_positive(self):
        class Counted(np.ndarray):
            calls=0
            def sum(self,*args,**kwargs):
                type(self).calls+=1
                return super().sum(*args,**kwargs)
        probability=np.array([.1,.2,.3,.4]).view(Counted)
        original_asarray=np.asarray
        def track_array(value,*args,**kwargs):
            if value is probability:return probability
            return original_asarray(value,*args,**kwargs)
        with patch('vkgr.sid_coll_sgns.np.asarray',side_effect=track_array):
            sample_negatives([0]*20,[1]*20,probability,5,np.random.default_rng(42))
        self.assertLessEqual(Counted.calls,2)

    def test_weighted_inverse_cdf_hand_oracle(self):
        probabilities=np.array([.1,.2,.3,.4])
        actual=sample_negatives([0],[1],probabilities,5,np.random.default_rng(7))
        rng=np.random.default_rng(7)
        expected=[]
        while len(expected)<5:
            draw=float(rng.random())
            if draw>=.3:expected.append(2 if draw<.6 else 3)
        self.assertEqual(actual.tolist(),[expected])

    def test_health_adjacent_not_window_and_dprime(self):
        v=np.array([[1.,0.],[0.,1.],[1.,0.]])
        h={1:[1,2,3]}
        a=health_check(v,[1,2,3],h,require_pass=False)
        rng=random.Random(0)
        expected=np.mean([v[i]@v[j] for i,j in [rng.sample(range(3),2) for _ in range(2)]])
        self.assertEqual(a['adjacent_pairs'],2)
        self.assertEqual(a['cooccurrence_mean_cosine'],0)
        self.assertEqual(a['random_mean_cosine'],expected)
        with self.assertRaisesRegex(ValueError,'health'):health_check(v,[1,2,3],h)

    @staticmethod
    def fixture():
        cat=np.arange(1,7)
        h={2:[1,2]*40,1:[3,4]*40,3:[5]}
        c=np.eye(6,4,dtype=np.float32);c[4:]=[1,1,1,1]
        return cat,h,c,SGNSConfig(dim=4,window=1,batch_size=8)

    def test_fit_replay_fallback_normalization_and_exclusive_output(self):
        cat,h,c,cfg=self.fixture()
        a,x=fit_sgns(cat,h,c,engineering=True,config=cfg)
        b,y=fit_sgns(cat,dict(reversed(list(h.items()))),c,engineering=True,config=cfg)
        np.testing.assert_array_equal(a,b);self.assertEqual(x,y)
        np.testing.assert_allclose(np.linalg.norm(a,axis=1),1,atol=1e-6)
        np.testing.assert_allclose(a[4:],[[.5]*4]*2)
        self.assertEqual(x['fallback_mask'],[False]*4+[True,True])
        self.assertEqual(x['epochs_completed'],5)
        self.assertEqual(x['vectors_sha256'],hashlib.sha256(a.astype('<f4').tobytes()).hexdigest())
        self.assertEqual(x['content_feature_sha256'],hashlib.sha256(c.astype('<f4').tobytes()).hexdigest())
        self.assertEqual(x['positive_participation'],[158]*4+[0,0])
        self.assertEqual(x['processed_positive_pairs'],5*316)
        self.assertAlmostEqual(x['learning_rates']['final'],.0000025)
        self.assertTrue(x['health']['passed'])
        self.assertEqual(len(x['negative_trace_sha256']),64)
        with tempfile.TemporaryDirectory() as td:
            save_sgns_output(Path(td)/'new',a,x)
            with self.assertRaises(FileExistsError):save_sgns_output(Path(td)/'new',a,x)

    def test_zero_content_with_positive_pairs_is_not_fallback(self):
        cat,h,c,cfg=self.fixture()
        baseline,_=fit_sgns(cat,h,c,engineering=True,config=cfg)
        c[0]=0
        vectors,audit=fit_sgns(cat,h,c,engineering=True,config=cfg)
        np.testing.assert_array_equal(vectors,baseline)
        self.assertFalse(audit['fallback_mask'][0])

    def test_initialization_digest_is_before_training(self):
        cat,h,c,cfg=self.fixture()
        _,audit=fit_sgns(cat,h,c,engineering=True,config=cfg)
        g=torch.Generator(device='cpu').manual_seed(42)
        initial=torch.empty((6,4)).uniform_(-.5/4,.5/4,generator=g).numpy()
        import json
        header=json.dumps({'dtype':initial.dtype.str,'shape':[6,4]},sort_keys=True).encode()
        expected=hashlib.sha256(header+b'\n'+initial.tobytes()).hexdigest()
        self.assertEqual(audit['input_initialization_sha256'],expected)

    def test_stop_gates(self):
        for case in ['pad','oov','duplicate','content_nan','content_zero','config','no_pairs','negative_support']:
            with self.subTest(case=case):
                cat,h,c,cfg=self.fixture()
                if case=='pad':cat[0]=0
                if case=='oov':h[1]=[77]
                if case=='duplicate':cat[1]=cat[0]
                if case=='content_nan':c[0,0]=np.nan
                if case=='content_zero':c[4]=0
                if case=='no_pairs':h={1:[1],2:[2]}
                if case=='negative_support':h={1:[1,2]}
                with self.assertRaises(ValueError):fit_sgns(cat,h,c,engineering=case!='config',config=cfg)

if __name__=='__main__':unittest.main()
