"""Executable mutants prove the target, mask, readout and assignment guards fire."""
import unittest
from unittest import mock
import torch
import test_seqm2_model as tm2
import test_seqm2_leak as tl2

from vkgr import retrieval_seqm1 as m1,retrieval_seqm2 as m2,retrieval_seqm2_run as r

class MutationTests(unittest.TestCase):
    def test_readout_head_axis_mutant_is_caught(self):
        def wrong(self,z,valid):
            scores=torch.tanh(self.attn_proj(z))@self.attn_w.T
            alpha=scores.transpose(1,2).masked_fill(~valid[:,None,:],-torch.inf).softmax(1)
            return alpha@z,alpha
        with mock.patch.object(m2.SeqM2IDMean,'_readout',wrong):
            with self.assertRaises(AssertionError): tm2.ModelTests('test_readout_and_gradients').test_readout_and_gradients()

    def test_future_mask_mutant_is_caught(self):
        with mock.patch.object(m1,'causal_mask',lambda n,device:torch.zeros(n,n,dtype=torch.bool,device=device)):
            with self.assertRaises(AssertionError): tm2.ModelTests('test_padding_causal_invalid').test_padding_causal_invalid()

    def test_target_tensor_mutant_is_caught(self):
        original=r._tensors2
        def wrong(batch,idmap,width,feature):
            h,s=original(batch,idmap,width,feature)
            h[:,0]=torch.tensor([idmap[v['target']] for v in batch])
            return h,s
        with mock.patch.object(r,'_tensors2',wrong):
            with self.assertRaises(AssertionError): tl2.LeakTests('test_target_flip_and_item_independence').test_target_flip_and_item_independence()

    def test_hard_at_step100_mutant_is_caught(self):
        original=r.multi_interest_loss
        def wrong(users,items,labels,mask,temperature,log_q,step):
            return original(users,items,labels,mask,temperature,log_q,101 if step==100 else step)
        # The hand oracle is deliberately nondegenerate enough to reject early hard.
        u=torch.tensor([[[1.,0.],[.8,.6],[0.,1.],[-1.,0.]]])
        a=original(u,torch.eye(2),torch.tensor([0]),torch.ones(1,2,dtype=torch.bool),1.,None,100)[0]
        b=wrong(u,torch.eye(2),torch.tensor([0]),torch.ones(1,2,dtype=torch.bool),1.,None,100)[0]
        self.assertGreater(abs(float(a-b)),1e-4)
        self.assertEqual(original(u,torch.eye(2),torch.tensor([0]),torch.ones(1,2,dtype=torch.bool),1.,None,100)[2]['phase'],'soft')
