"""Independent deterministic oracles for P4E sampling and loss."""
import importlib.util
import random
import unittest

class AblationsTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_ablations'), 'P4E candidate behavior missing')
        from vkgr import retrieval_ablations
        return retrieval_ablations

    def test_uniform_excludes_existing_and_preserves_fullpast_target_override(self):
        import torch
        m = self.module()
        rows = [{'target':2,'past_likes':[1,2,4]}, {'target':4,'past_likes':[2]}]
        catalog = [0,1,2,3,4,5]
        idmap = {item:j+1 for j,item in enumerate(catalog)}
        oracle = random.Random(42).sample([0,1,3,5],3)
        columns,labels,mask,negatives = m.uniform_columns(rows,catalog,idmap,random.Random(42),3)
        items = sorted([2,4]+oracle)
        self.assertEqual(negatives,oracle)
        self.assertEqual(columns.tolist(),[idmap[i] for i in items])
        self.assertEqual(labels.tolist(),[items.index(2),items.index(4)])
        self.assertEqual(mask.tolist(),[[i==2 or i not in {1,2,4} for i in items],[i==4 or i!=2 for i in items]])
        self.assertEqual(columns.dtype,torch.int64)

    def test_uniform_pool_shortage_is_error_not_silent_fill(self):
        m = self.module()
        with self.assertRaises(ValueError):
            m.uniform_columns([{'target':1,'past_likes':[]}],[1,2],{1:1,2:2},random.Random(1),128)

    def test_loss_matches_independent_hand_calculation(self):
        import torch,math
        m=self.module()
        users=torch.tensor([[1.,0.],[0.,1.]])
        items=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
        loss,skip=m.masked_ce(users,items,torch.tensor([0,1]),torch.tensor([[True,False,True],[True,True,True]]),1.)
        want=(math.log(math.exp(1)+math.exp(-1))-1+math.log(2+math.exp(1))-1)/2
        self.assertAlmostEqual(float(loss),want,places=6)
        self.assertEqual(skip,0)

    def test_replacement_exact_two_level_draws_and_short_epoch_tail(self):
        m=self.module()
        self.assertTrue(hasattr(m,'UserPrefixCursor'),'replacement sampler behavior missing')
        users=[9,9,9,2,7,7]
        oracle_rng=random.Random(2024);groups={2:[3],7:[4,5],9:[0,1,2]};ids=[2,7,9]
        want=[]
        for _ in range(12):
            group=groups[ids[oracle_rng.randrange(3)]]
            want.append(group[oracle_rng.randrange(len(group))])
        cursor=m.UserPrefixCursor(users,2024,6)
        self.assertEqual(cursor.next(4),want[:4]);self.assertEqual(cursor.next(4),want[4:6])
        self.assertEqual(cursor.offset,6);self.assertEqual(cursor.epoch,0)
        state=cursor.state_dict();other=m.UserPrefixCursor(users,2024,6);other.load_state_dict(state)
        self.assertEqual(other.next(6),want[6:]);self.assertEqual(other.epoch,1)
        self.assertEqual(cursor.next(6),want[6:])
        with self.assertRaises(ValueError):m.UserPrefixCursor(users,1,7)
        damaged={**state,'offset':7}
        with self.assertRaises(ValueError):other.load_state_dict(damaged)

    def test_ablation_train_step_retains_repeated_queries_and_updates_finite_weights(self):
        import torch
        from vkgr.retrieval_training import IDMean,SamplerStats
        m=self.module();self.assertTrue(hasattr(m,'ablation_train_step'),'ablation train step missing')
        rows=[{'target':2,'history':[1],'past_likes':[1]}, {'target':2,'history':[1],'past_likes':[1]}, {'target':3,'history':[2],'past_likes':[2]}]
        model=IDMean(140,list(range(1,141)));opt=torch.optim.AdamW(model.parameters(),lr=.001);stats=SamplerStats()
        before=model.embedding.weight.detach().clone()
        record=m.ablation_train_step(model,opt,rows,{i:i for i in range(1,141)},stats,0,.15,'uniform_negatives',random.Random(5))
        self.assertEqual(record['drawn'],3);self.assertEqual(stats.drawn,3);self.assertEqual(len(record['negative_ids']),128)
        self.assertTrue(torch.isfinite(model.embedding.weight).all());self.assertFalse(torch.equal(before,model.embedding.weight))
