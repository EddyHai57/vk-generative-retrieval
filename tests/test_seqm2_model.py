"""Independent M2 invariants catch wrong draws, readout axis, gate and causal mask."""
import unittest
import torch
import test_seqm1_model as tm
def build(seed=42):
    from vkgr.retrieval_seqm2 import make_seqm2_model
    from vkgr.training import seed_all
    seed_all(seed)
    return make_seqm2_model(tm.CONTENT,tm.SUPPORTED,50)
class ModelTests(unittest.TestCase):
    def test_shared_and_zero(self):
        a=tm.build('mean'); b=build()
        for k,v in a.state_dict().items(): self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        self.assertEqual(sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters()),4416)
        attn=tm.build('attn')
        self.assertTrue(torch.equal(attn.attn_proj.weight,b.attn_proj.weight))
        self.assertTrue(torch.equal(attn.attn_proj.bias,b.attn_proj.bias))
        for mode in (False,True):
            b.train(mode)
            for h,s in tm.edge_cases().values():
                u=b.users(h,s); m=a.anchor_vector(h,s)
                self.assertEqual(tuple(u.shape),(len(h),4,64))
                self.assertTrue(torch.equal(u,m[:,None,:].expand_as(u)))
    def test_readout_and_gradients(self):
        b=build().eval(); h,s=tm.make_batch([8,5,9],12,17); p=b.parts(h,s)
        scores=torch.tanh(b.attn_proj(p.z))@b.attn_w.T
        for i in range(len(h)):
            n=int(p.valid[i].sum())
            for j in range(4):
                r=(torch.softmax(scores[i,:n,j],0)[:,None]*p.z[i,:n]).sum(0)
                self.assertTrue(torch.allclose(p.r[i,j],r,atol=1e-7,rtol=0))
        (b.users(h,s)*torch.randn(3,4,64)).sum().backward()
        self.assertGreater(float(b.gate.grad.abs()),0)
        for name,p in tm.branch_parameters(b): self.assertEqual(int(torch.count_nonzero(p.grad)),0,name)
        b.zero_grad(); b.gate.data.fill_(.5)
        from vkgr.retrieval_seqm2_run import multi_interest_loss
        loss,_,_=multi_interest_loss(b.users(h,s),torch.randn(3,64),torch.arange(3),torch.ones(3,3,dtype=torch.bool),.1,torch.zeros(3),1)
        loss.backward()
        self.assertTrue(torch.isfinite(b.attn_w.grad).all())
        self.assertTrue((b.attn_w.grad.abs().sum(1)>0).all())
    def test_padding_causal_invalid(self):
        b=build().eval(); b.gate.data.fill_(.5); h,s=tm.make_batch([4,9],9,3)
        self.assertTrue(torch.allclose(b.users(h,s),b.users(*tm.pad_to(h,s,50)),atol=1e-6,rtol=0))
        p=b.parts(h,s); h2=h.clone(); h2[1,6:]=99
        self.assertTrue(torch.equal(p.z[1,:6],b.parts(h2,s).z[1,:6]))
        for col,value in ((1,255),(2,255)):
            changed=s.clone(); changed[1,6:,col]=value
            self.assertTrue(torch.equal(p.z[1,:6],b.parts(h,changed).z[1,:6]))
        with self.assertRaises(ValueError): b.users(torch.zeros_like(h),torch.zeros_like(s))
        b.attn_w.data[0,0]=float('nan')
        with self.assertRaises(ValueError): b.users(h,s)

class PaddingFiniteTests(unittest.TestCase):
    def test_pad_nan_is_masked_valid_nan_rejected(self):
        class Inject(torch.nn.Module):
            def __init__(self,inner,pad): super().__init__(); self.inner=inner; self.pad=pad
            def forward(self,x,src_mask=None,src_key_padding_mask=None):
                out=self.inner(x,src_mask=src_mask,src_key_padding_mask=src_key_padding_mask)
                hit=src_key_padding_mask if self.pad else ~src_key_padding_mask
                return out.masked_fill(hit[...,None],float('nan'))
        h,s=tm.make_batch([3,8,5],8,29); b=build().eval(); b.gate.data.fill_(.5); layer=b.layer
        b.layer=Inject(layer,True)
        self.assertTrue(torch.isfinite(b.users(h,s)).all())
        b.layer=Inject(layer,False)
        with self.assertRaises(ValueError): b.users(h,s)
