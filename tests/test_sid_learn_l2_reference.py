"""Exercise the unchanged L2 optimizer/reset imported from fixed original Git."""
import unittest
import numpy as np
import torch
from vkgr import sid_rqvae as rq,sid_rqvae_r as fixed,sid_learn_v1 as sl

class L2ReferenceTests(unittest.TestCase):
    def test_optimizer_and_reset_cadence_match_frozen_l2(self):
        model=rq.RQVAE(2);optimizer=fixed.make_optimizer(model.parameters(),'l2')
        self.assertIsInstance(optimizer,torch.optim.Adagrad)
        self.assertEqual(optimizer.defaults['lr'],.1);self.assertEqual(optimizer.defaults['initial_accumulator_value'],.1);self.assertEqual(optimizer.defaults['eps'],1e-7)
        cfg=fixed.PREREG_R['arms']['l2']
        for step,want in ((499,False),(500,True),(501,False),(270000,True),(270100,False)):
            self.assertEqual(fixed.reset_due(step,cfg),want)
        self.assertEqual(sl.expected_config()['reset'],{'enabled':True,**{k:cfg[k] for k in ('reset_check_every','dead_window','reset_until','rng_reset_offset')}})

    def test_reset_uses_actual_layer_residual_and_resets_adagrad_accumulator(self):
        torch.manual_seed(41);model=rq.RQVAE(2);x=torch.randn(6,64);q=model(x)['q'];optimizer=fixed.make_optimizer(model.parameters(),'l2')
        tables=fixed.new_last_used(model);rng=np.random.RandomState(42+7919);expected_rng=np.random.RandomState(42+7919)
        expected=[r.detach()[torch.from_numpy(expected_rng.choice(6,2,replace=False))].clone() for r in q['residuals'][:-1]]
        for book in model.codebooks:optimizer.state[book]['sum'].fill_(2.)
        log=fixed.reset_dead_codes(model,optimizer,q['residuals'],tables,500,rng,fixed.PREREG_R['arms']['l2'])
        self.assertEqual([v['n_reset'] for v in log],[2,2,2])
        for book,want,table in zip(model.codebooks,expected,tables):
            self.assertTrue(torch.equal(book,want));self.assertTrue(torch.equal(optimizer.state[book]['sum'],torch.full_like(book,.1)));self.assertTrue(torch.equal(table,torch.full_like(table,500)))

if __name__=='__main__':unittest.main()
