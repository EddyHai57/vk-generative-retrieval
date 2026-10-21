import unittest
import numpy as np
import torch
from vkgr.sid_coll_sgns import SGNSConfig,iter_positive_pairs,sgns_loss,learning_rate,sample_negatives

class NativeTests(unittest.TestCase):
    def test_gpu_pair_order_batch_boundaries_and_tails(self):
        from vkgr.sgns_gpu_native import pack,positive_chunks,batches
        histories={8:[3,3,1,2,3,1],1:[4,2],5:[1],2:[]};catalog=np.arange(1,5)
        events,users,freq=pack(catalog,histories,'cpu')
        expected=[(a-1,b-1) for a,b in iter_positive_pairs(histories,5)]
        self.assertEqual(freq.tolist(),[3,2,3,1])
        for chunk in (1,2,5,100):
            actual=list(batches(positive_chunks(events,users,5,chunk),7))
            self.assertEqual(torch.cat(actual).tolist(),[list(x) for x in expected])
            self.assertTrue(all(len(x)==7 for x in actual[:-1]));self.assertEqual(len(actual[-1]),len(expected)%7 or 7)
    def test_conditional_gpu_sampler_matches_rejection_distribution_with_replacement(self):
        from vkgr.sgns_gpu_native import negatives
        p=torch.tensor([.05,.1,.2,.25,.4],dtype=torch.float64);cdf=p.cumsum(0);cdf[-1]=1
        centers=torch.zeros(50000,dtype=torch.int64);positive=torch.full_like(centers,3)
        values=negatives(torch.stack((centers,positive),1),cdf,p,5,torch.Generator().manual_seed(42))
        self.assertEqual(values.shape,(50000,5));self.assertFalse(torch.isin(values,torch.tensor([0,3])).any())
        observed=torch.bincount(values.flatten(),minlength=5).numpy()/values.numel()
        expected=np.array([0,.1,.2,0,.4])/.7
        np.testing.assert_allclose(observed,expected,atol=.003)
        original=sample_negatives(centers.numpy()[:5000],positive.numpy()[:5000],p.numpy(),5,np.random.default_rng(42))
        np.testing.assert_allclose(np.bincount(original.flatten(),minlength=5)/original.size,expected,atol=.01)
        self.assertTrue((values[:,0]==values[:,1]).any())
    def test_sum_loss_gradients_and_duplicate_scatter_match_original_autograd(self):
        from vkgr.sgns_gpu_native import step
        torch.manual_seed(42);a=torch.randn(6,4)*.01;b=torch.randn(6,4)*.01
        pairs=torch.tensor([[0,1],[0,2],[2,0],[0,1]]);neg=torch.tensor([[2,3],[3,3],[1,4],[5,2]])
        aa=a.clone().requires_grad_();bb=b.clone().requires_grad_();loss=sgns_loss(aa[pairs[:,0]],bb[pairs[:,1]],bb[neg]);loss.backward()
        expected_a=aa.detach()-.025*aa.grad;expected_b=bb.detach()-.025*bb.grad
        progress=torch.tensor(0,dtype=torch.int64);finite=torch.tensor(True);summed=torch.tensor(0.,dtype=torch.float64)
        step(a,b,pairs,neg,progress,1000,.025,1e-4,finite,summed)
        torch.testing.assert_close(a,expected_a,atol=1e-8,rtol=1e-6);torch.testing.assert_close(b,expected_b,atol=1e-8,rtol=1e-6)
        self.assertAlmostEqual(summed.item(),loss.item(),places=5);self.assertEqual(progress.item(),4);self.assertTrue(finite)
    def test_lr_uses_progress_before_batch_and_tail_then_epoch_reset(self):
        from vkgr.sgns_gpu_native import step
        pairs=torch.tensor([[0,1],[1,2]]);neg=torch.tensor([[2],[0]])
        for processed in (0,2,4,6):
            a=torch.ones(3,2)*.2;b=torch.ones(3,2)*.3
            aa=a.clone().requires_grad_();bb=b.clone().requires_grad_();loss=sgns_loss(aa[pairs[:,0]],bb[pairs[:,1]],bb[neg]);loss.backward()
            lr=learning_rate(processed,8)
            step(a,b,pairs,neg,torch.tensor(processed),8,.025,1e-4,torch.tensor(True),torch.tensor(0.,dtype=torch.float64))
            torch.testing.assert_close(a,aa.detach()-lr*aa.grad)
    def test_formal_config_is_frozen_and_rejects_invalid_histories(self):
        from vkgr.sgns_gpu_native import NativeSGNS,pack
        with self.assertRaises(ValueError):NativeSGNS(np.arange(1,4),{1:[1,2,3]},device='cpu',config=SGNSConfig(window=2))
        with self.assertRaises(ValueError):pack(np.arange(1,4),{1:[0,2]},'cpu')
        with self.assertRaises(ValueError):pack(np.arange(1,4),{True:[1,2]},'cpu')
    def test_counts_repeat_each_epoch_and_no_per_step_host_readback(self):
        from vkgr.sgns_gpu_native import NativeSGNS,step,negatives
        import inspect
        histories={1:[1,2,3,2],2:[4,3,1]};engine=NativeSGNS(np.arange(1,5),histories,device='cpu')
        expected=sum(1 for _ in iter_positive_pairs(histories,5));self.assertEqual(engine.pairs_per_epoch,expected)
        first=engine.epoch();second=engine.epoch()
        self.assertEqual(first['positive_pairs'],expected);self.assertEqual(second['positive_pairs'],expected)
        self.assertEqual(second['processed_positive_pairs'],2*expected)
        for function in (step,negatives):
            text=inspect.getsource(function)
            for marker in ('.item(','.cpu(','.numpy(','.tolist('):self.assertNotIn(marker,text)
    def test_distribution_expected_bins_match_exact_conditional_sum(self):
        from vkgr.sgns_gpu_native import distribution
        p=torch.tensor([.05,.1,.2,.25,.4],dtype=torch.float64);pairs=torch.tensor([[0,3],[1,4],[2,0]])
        n=torch.tensor([[1,2,4],[0,2,3],[1,3,4]])
        observed,expected=distribution(pairs,n,p,8)
        oracle=np.zeros(8)
        for a,b in pairs.tolist():
            x=p.numpy().copy();x[[a,b]]=0;oracle[:5]+=3*x/x.sum()
        np.testing.assert_allclose(expected.numpy(),oracle,atol=1e-12)
        from vkgr.sgns_gpu_native import FrequencyAudit
        audit=FrequencyAudit(p);audit.add(pairs,n)
        np.testing.assert_allclose((p*audit.scale-audit.excluded).numpy(),oracle[:5],atol=1e-12)
        np.testing.assert_array_equal(audit.observed.numpy(),torch.bincount(n.flatten(),minlength=5).numpy())
        report=audit.report();self.assertEqual(report['negative_per_item_diagnostics']['observed_total'],9)
        self.assertEqual(observed.sum().item(),9)
    def test_zero_frequency_gaps_and_reversed_exclusions(self):
        from vkgr.sgns_gpu_native import negatives,NativeSGNS
        p=torch.tensor([.2,0.,.3,0.,.5],dtype=torch.float64);cdf=p.cumsum(0)
        for center,positive,only in ((0,2,4),(2,0,4),(4,0,2),(0,4,2),(2,4,0)):
            pairs=torch.tensor([[center,positive]]).repeat(1000,1)
            actual=negatives(pairs,cdf,p,5,torch.Generator().manual_seed(42))
            self.assertTrue((actual==only).all())
        engine=NativeSGNS(np.arange(1,5),{1:[1,1,2,3,4,1]},device='cpu')
        expected=np.array([3,1,1,1],dtype=np.float64)**.75;expected/=expected.sum()
        np.testing.assert_allclose(engine.p.numpy(),expected,atol=1e-15)
    def test_bounded_prefix_cannot_restart_an_epoch(self):
        from vkgr.sgns_gpu_native import NativeSGNS
        engine=NativeSGNS(np.arange(1,5),{1:[1,2,3,4]*256},device='cpu')
        first=engine.epoch(max_batches=1);self.assertFalse(first['complete_epoch'])
        with self.assertRaises(ValueError):engine.epoch()
if __name__=='__main__':unittest.main()
