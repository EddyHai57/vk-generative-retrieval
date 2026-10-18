"""Frozen S2 losses: hand arithmetic, real autograd, and unchanged hard export."""
import importlib,math,unittest
from pathlib import Path
import numpy as np
import torch
from vkgr import sid_rqvae as rq
ROOT=Path(__file__).resolve().parents[1]

class LearnLossTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'src/vkgr/sid_learn_loss.py').is_file(),'missing frozen loss entry')
        self.loss=importlib.import_module('vkgr.sid_learn_loss')
        torch.set_num_threads(1)

    def model(self,k=5):
        torch.manual_seed(11);m=rq.RQVAE(k).double();x=torch.randn(6,64,dtype=torch.float64)
        return m,x

    def hand(self,k=2,layers=3):
        e=torch.zeros(k,32,dtype=torch.float64);e[1:,0]=math.sqrt(32*math.log(3))
        r=torch.stack((e[0],e[1],e[0]))
        return {'residuals':[r.clone() for _ in range(layers+1)]},[e.clone() for _ in range(layers)]

    def test_hand_soft_assignment_batch_mean_and_normalized_negative_entropy(self):
        q,books=self.hand();got=self.loss.diversity_loss(q,books)
        # Each layer p is [(3/4,1/4),(1/4,3/4),(3/4,1/4)], q=(7/12,5/12).
        expected=((7/12)*math.log(7/12+1e-12)+(5/12)*math.log(5/12+1e-12))/math.log(2)
        self.assertAlmostEqual(got.item(),expected,places=12)

    def test_uniform_single_code_and_depth_code_count_normalization(self):
        values=[]
        for layers,k in ((1,48),(3,48),(3,256)):
            r=torch.zeros(2,32,dtype=torch.float64);e=torch.zeros(k,32,dtype=torch.float64)
            q={'residuals':[r for _ in range(layers+1)]}
            uniform=self.loss.diversity_loss(q,[e for _ in range(layers)])
            self.assertAlmostEqual(uniform.item(),-1.,delta=1e-9);values.append(uniform.item())
            concentrated=e.clone();concentrated[1:,0]=1000.
            self.assertAlmostEqual(self.loss.diversity_loss(q,[concentrated for _ in range(layers)]).item(),0.,delta=1e-9)
        self.assertLess(max(values)-min(values),1e-9)

    def test_diversity_reaches_encoder_and_each_codebook(self):
        m,x=self.model();out=m(x);self.loss.diversity_loss(out['q'],list(m.codebooks)).backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in m.encoder.parameters() if p.grad is not None),0.)
        for p in m.codebooks:self.assertIsNotNone(p.grad);self.assertGreater(p.grad.abs().sum().item(),0.)
        self.assertTrue(all(p.grad is None for p in m.decoder.parameters()))

    def test_uses_actual_forward_hard_residuals(self):
        m,x=self.model();q=m(x)['q'];books=list(m.codebooks)
        for r,e,codes in zip(q['residuals'],books,q['codes']):
            self.assertTrue(torch.equal(self.loss.squared_distances(r,e).argmin(1),codes))
        before=self.loss.diversity_loss(q,books)
        changed={**q,'residuals':list(q['residuals'])};changed['residuals'][1]=q['residuals'][1]+3.
        self.assertGreater(abs((self.loss.diversity_loss(changed,books)-before).item()),1e-8)

    def test_align_excludes_fallback_anchor_and_negative(self):
        z=torch.zeros(3,32,dtype=torch.float64);z[0,0]=1.;z[1,1]=1.;z[2,0]=1.;z.requires_grad_()
        target=z.detach().clone();mask=torch.tensor([False,False,True])
        value=self.loss.alignment_loss(z,target,mask)
        expected=math.log1p(math.exp(-1/.07))
        self.assertAlmostEqual(value.item(),expected,places=12)
        other=target.clone();other[2]=torch.linspace(-2,2,32,dtype=torch.float64)
        self.assertTrue(torch.equal(value,self.loss.alignment_loss(z,other,mask)))
        value.backward();self.assertEqual(z.grad[2].abs().sum().item(),0.)

    def test_gamma_zero_is_bit_equal_to_original_l2_plus_alignment(self):
        m,x=self.model();out=m(x);target=torch.randn(6,32,dtype=torch.float64);mask=torch.tensor([False,True,False,False,False,False])
        expected=rq.loss_terms(x,out['xhat'],out['q'])['total']+.1*self.loss.alignment_loss(out['zst'],target,mask)
        terms=self.loss.loss_terms(x,out,target,mask,list(m.codebooks),gamma=0.)
        self.assertTrue(torch.equal(terms['total'].view(torch.int64),expected.view(torch.int64)))
        regular=self.loss.loss_terms(x,out,target,mask,list(m.codebooks))
        self.assertTrue(torch.equal(regular['div'],self.loss.diversity_loss(out['q'],list(m.codebooks))))
        self.assertTrue(torch.equal(regular['total'],expected+.01*regular['div']))

    def test_diversity_contains_fallback_rows(self):
        m,x=self.model();out=m(x);target=torch.randn(6,32,dtype=torch.float64);mask=torch.tensor([False,True,True,False,False,False])
        value=self.loss.loss_terms(x,out,target,mask,list(m.codebooks))['div']
        self.assertTrue(torch.equal(value,self.loss.diversity_loss(out['q'],list(m.codebooks))))
        partial={**out['q'],'residuals':[r[~mask] for r in out['q']['residuals']]}
        self.assertGreater(abs((value-self.loss.diversity_loss(partial,list(m.codebooks))).item()),1e-10)

    def test_loss_parameters_do_not_change_hard_export_at_fixed_state(self):
        torch.manual_seed(17);m=rq.RQVAE(4);x=np.random.default_rng(9).normal(size=(4,64)).astype('float32');batch=torch.from_numpy(x)
        baseline=rq.export_codes({'K':4,'state_dict':m.state_dict()},x);digest=rq.state_digest(m.state_dict())
        target=torch.randn(4,32);mask=torch.zeros(4,dtype=torch.bool)
        for gamma,tau in ((0.,.25),(.01,1.),(.2,4.)):
            self.loss.loss_terms(batch,m(batch),target,mask,list(m.codebooks),gamma=gamma,tau_div=tau)
            self.assertEqual(rq.state_digest(m.state_dict()),digest)
            np.testing.assert_array_equal(rq.export_codes({'K':4,'state_dict':m.state_dict()},x),baseline)

    def test_distance_diagnostics_are_for_d_over_latent_dimension(self):
        r=torch.zeros(1,32,dtype=torch.float64);e=torch.zeros(2,32,dtype=torch.float64);e[1,0]=math.sqrt(32)
        records=self.loss.distance_diagnostics({'residuals':[r,r,r,r]},[e,e,e])
        self.assertEqual(len(records),3)
        for row in records:
            self.assertAlmostEqual(row['mean'],.5,places=12);self.assertAlmostEqual(row['p50'],.5,places=12);self.assertAlmostEqual(row['p90'],.9,places=12)

    def test_nonfinite_and_wrong_latent_dimension_stop(self):
        q,books=self.hand();q['residuals'][0][0,0]=float('nan')
        with self.assertRaises(ValueError):self.loss.diversity_loss(q,books)
        with self.assertRaises(ValueError):self.loss.squared_distances(torch.ones(2,31),torch.ones(3,31))
        with self.assertRaises(ValueError):self.loss.diversity_loss(*self.hand(),tau=0.)

if __name__=='__main__':unittest.main()
