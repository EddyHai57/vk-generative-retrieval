"""M2 assignment, exact retrieval and complete synthetic replay."""
import math
import random
import tempfile
import unittest
from pathlib import Path
import torch
from vkgr import retrieval_feat02_run as fr
class RunTests(unittest.TestCase):
    def test_assignment_oracle(self):
        from vkgr.retrieval_seqm2_run import multi_interest_loss
        u=torch.tensor([[[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]],requires_grad=True)
        items=torch.eye(2); labels=torch.tensor([0]); mask=torch.ones(1,2,dtype=torch.bool)
        losses=torch.tensor([math.log1p(math.exp(-1)),math.log1p(math.e),math.log1p(math.e),math.log1p(math.exp(-1))])
        weights=torch.softmax(torch.tensor([10.,0.,-10.,0.]),0)
        loss,skip,_=multi_interest_loss(u,items,labels,mask,1.,torch.zeros(2),100)
        self.assertAlmostEqual(float(loss.detach()),float((weights*losses).sum()),places=6)
        loss,_,diag=multi_interest_loss(u,items,labels,mask,1.,torch.zeros(2),101)
        self.assertAlmostEqual(float(loss.detach()),float(losses[0]),places=6)
        self.assertEqual(diag['assignment_counts'],[1,0,0,0]); self.assertEqual(skip,0)
        tie=u.detach()[:,0:1].repeat(1,4,1)
        self.assertEqual(multi_interest_loss(tie,items,labels,mask,1.,None,101)[2]['assignment_counts'],[1,0,0,0])
        mask[:,1]=False
        self.assertEqual(multi_interest_loss(u,items,labels,mask,1.,None,1)[:2],(None,1))
    def test_zero_loss_stats(self):
        from vkgr import retrieval_seqm2_run as r
        from vkgr.retrieval_negatives import masked_ce_logq,target_frequency
        from vkgr.retrieval_training import SamplerStats
        d=fr.synthetic_data(); c={**r.default_config(),'batch_size':16,'max_epochs':2,'min_epochs':2}
        m=r._model(d,c,'cpu'); batch=fr._view(d,50)[0][:16]
        cols,labels,mask,neg,pool=r.b1._columns(batch,d,c,random.Random(1),None)
        h,s=fr._tensors2(batch,d['idmap'],50,c['feature'])
        q=r.b1._log_q(batch,neg,pool,c,target_frequency(d['rows']))
        u=m.users(h,s); items=m.items(cols)
        a=masked_ce_logq(u[:,0],items,labels,mask,.1,q)[0]
        b=r.multi_interest_loss(u,items,labels,mask,.1,q,1)[0]
        self.assertLessEqual(abs(float((a-b).detach())),1e-6)
        b.backward()
        self.assertGreater(float(m.gate.grad.abs()),0)
        for name,param in m.named_parameters():
            if name.split('.')[0] in ('order_table','layer','final_norm','attn_proj','attn_w'):
                self.assertEqual(int(torch.count_nonzero(param.grad)),0,name)
        stats=SamplerStats()
        rec=r.train_step(m,r.make_optimizer(m,c),batch,d,stats,0,c,random.Random(1),None,target_frequency(d['rows']),1)
        self.assertEqual(stats.drawn,16); self.assertEqual(rec['drawn'],16)
    def test_exact_ties(self):
        import numpy as np
        from vkgr.retrieval_seqm2_run import exact_max_topk
        u=np.array([[[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]],dtype='float32')
        items=np.array([[1.,0.],[0.,1.],[1.,0.],[-1.,0.],[0.,-1.]],dtype='float32')
        pred,win=exact_max_topk(u,items,[9,2,8,7,3],5,item_block=2)
        self.assertEqual(pred,[[2,3,7,8,9]]); self.assertEqual(win,[1])
    def test_run_verify_and_tamper(self):
        from vkgr import retrieval_seqm2_run as r
        d=fr.synthetic_data(); c={**r.default_config(),'batch_size':16,'max_epochs':2,'min_epochs':2}
        with self.assertRaises(ValueError): r.validate_config({**r.default_config(),'arm':'mean'})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'; v=r.run_session(out,d,c)
            self.assertEqual(r.verify_run(out/'manifest.json',v['manifest_sha256'],d,c)['status'],'verified')
            self.assertIn('top1_winning_head_counts',v['evaluations'][0]['diagnostics'])
            self.assertIn('hard_window',v['evaluations'][0]['diagnostics'])
            self.assertIn('dependency_sha256',v['binding'])
            self.assertTrue((out/'predictions-best-g0.json').is_file())
            v.pop('manifest_sha256')
            v['evaluations'][v['best']['epoch']-1]['diagnostics']['branch_ratio_median']+=.1
            sha=r.b1._write_json_atomic(out/'manifest.json',v)
            with self.assertRaises(ValueError): r.verify_run(out/'manifest.json',sha,d,c)

class ContractTests(unittest.TestCase):
    def test_three_frozen_configs(self):
        import json
        from vkgr import retrieval_seqm2_run as r
        root=Path(__file__).resolve().parents[1]/'configs'/'seqm2'
        configs=[json.loads((root/f'p-seqm2-k4-s{seed}.json').read_text()) for seed in (42,2024,2025)]
        for seed,c in zip((42,2024,2025),configs):
            r.validate_config(c); self.assertEqual(c['seed'],seed)
        projected=[{k:v for k,v in c.items() if k not in ('seed','run_id')} for c in configs]
        self.assertTrue(all(c==projected[0] for c in projected))
    def test_logq_mask_and_detached_gradient(self):
        from vkgr.retrieval_seqm2_run import multi_interest_loss
        import torch.nn.functional as F
        u=torch.randn(3,4,5,generator=torch.Generator().manual_seed(7),requires_grad=True)
        items=torch.randn(4,5,generator=torch.Generator().manual_seed(8))
        labels=torch.tensor([0,1,2]); mask=torch.tensor([[1,0,1,1],[0,1,0,0],[1,1,1,0]],dtype=torch.bool)
        q=torch.tensor([-.4,-1.2,-2.3,-.7])
        loss,skips,_=multi_interest_loss(u,items,labels,mask,.1,q,1)
        got=torch.autograd.grad(loss,u)[0]
        v=F.normalize(items,dim=-1)
        oracle=[]
        for i in (0,2):
            normalized=F.normalize(u[i],dim=-1)
            weights=torch.softmax((normalized*v[labels[i]]).sum(-1)/.1,0).detach()
            per=[]
            for j in range(4):
                scores=normalized[j]@v.T/.1-q
                per.append(-scores[labels[i]]+torch.logsumexp(scores[mask[i]],0))
            oracle.append((weights*torch.stack(per)).sum())
        expected=torch.stack(oracle).mean()
        grad=torch.autograd.grad(expected,u)[0]
        self.assertEqual(skips,1); self.assertTrue(torch.allclose(got,grad,atol=1e-6,rtol=1e-5))
        self.assertTrue(torch.equal(got[1],torch.zeros_like(got[1])))
    def test_hard_window_zero_head_is_diagnostic(self):
        from vkgr import retrieval_seqm2_run as r
        records=[{'loss':1.,'assignment':{'optimizer_step':step,'assignment_counts':[2,0,0,0]}} for step in (100,101,200,201)]
        diag=r.hard_window(records)
        self.assertEqual(diag['head_counts'],[4,0,0,0]); self.assertEqual(diag['effective_interests'],1)
        self.assertTrue(diag['diagnostic_only'])
