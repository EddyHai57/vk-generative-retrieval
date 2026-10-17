"""Hand-computed profile, temporal isolation and shared-init oracles."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

class Profiles(unittest.TestCase):
    def f(self):
        from vkgr import profile_features
        return profile_features
    def test_missing_feature_module(self):
        self.assertIsNotNone(importlib.util.find_spec('vkgr.profile_features'))
    def world(self):
        events=[{'user_id':1,'item_id':10+i,'like':like} for i,like in enumerate([True,False,None,True])]
        rows=[{'example_id':0,'user_id':1,'target':11,'target_position':[0,1],'history':[10],'history_positions':[[0,0]]},
              {'example_id':1,'user_id':1,'target':13,'target_position':[0,3],'history':[10,11,12],'history_positions':[[0,0],[0,1],[0,2]]}]
        qs=[{'user_id':1,'target':99,'target_position':[25,0],'oov':True,'history':[10,11,12,13],'history_positions':[[0,i] for i in range(4)]}]
        return events,rows,qs
    def test_snapshot_before_update_and_missing_hand_counts(self):
        events,rows,qs=self.world();p=self.f().build_profiles([(0,events)],rows,qs)
        self.assertEqual([r['counts'] for r in p['train']],[[1,1,1,1],[3,1,1,2]])
        self.assertEqual(p['validation'][0]['counts'],[4,2,2,3])
    def test_current_target_and_future_feedback_do_not_change_current_profile(self):
        events,rows,qs=self.world();f=self.f();p=f.build_profiles([(0,events)],rows,qs)
        changed=copy.deepcopy(events);changed[1]['like']=True;changed[3]['like']=False
        z=f.build_profiles([(0,changed)],rows,qs)
        self.assertEqual(p['train'][0]['counts'],z['train'][0]['counts'])
    def test_before_last50_changes_activity_not_like_rate(self):
        f=self.f();events=[{'user_id':1,'item_id':i,'like':i==0} for i in range(61)]
        row={'example_id':0,'user_id':1,'target':60,'target_position':[0,60],'history':list(range(10,60)),'history_positions':[[0,i] for i in range(10,60)]}
        a=f.build_profiles([(0,events)],[row],[])['train'][0]['counts'];events[0]['like']=False;b=f.build_profiles([(0,events)],[row],[])['train'][0]['counts']
        self.assertEqual(a,[60,1,0,50]);self.assertEqual(b,[60,0,0,50])
    def test_validation_fields_do_not_change_train_snapshot_or_fit(self):
        f=self.f();events,rows,qs=self.world();a=f.build_profiles([(0,events)],rows,qs);other=copy.deepcopy(qs);other[0]['target']=88
        b=f.build_profiles([(0,events)],rows,other)
        self.assertEqual([x['counts'] for x in a['train']],[x['counts'] for x in b['train']]);self.assertEqual(a['validation'][0]['counts'],b['validation'][0]['counts']);self.assertEqual(f.fit_bounds(a['train']),f.fit_bounds(b['train']))
    def test_log2_zero_cap_and_undefined_zero_separation(self):
        f=self.f();self.assertEqual([f.count_bucket(n) for n in [0,1,2,3,4,2**30]],[0,1,2,2,3,16])
        self.assertEqual(f.rate_bucket(0,0,[.5]),9);self.assertEqual(f.rate_bucket(0,4,[.5]),0)
        self.assertEqual(f.rate_bucket(1,4,[.5]),1);self.assertEqual(f.rate_bucket(3,4,[.5]),2)
    def test_positive_train_quantiles_keep_unique_boundaries(self):
        f=self.f();rows=[{'counts':[0,0,0,1]},{'counts':[0,0,1,1]},{'counts':[0,0,1,1]},{'counts':[0,0,0,0]}]
        b=f.fit_bounds(rows);self.assertEqual(b['boundaries'],[1.]);self.assertEqual(b['positive_samples'],2)
    def test_sidecar_wrong_order_and_query_identity_fail_closed(self):
        f=self.f();events,rows,qs=self.world();p=f.build_profiles([(0,events)],rows,qs);b=f.fit_bounds(p['train'])
        attached=f.attach(rows,p['train'],b,'train');self.assertEqual(len(attached),2)
        with self.assertRaises(ValueError):f.attach(rows,list(reversed(p['train'])),b,'train')
        qs2=copy.deepcopy(qs);qs2[0]['user_id']=9
        with self.assertRaises(ValueError):f.attach(qs2,p['validation'],b,'validation')
    def test_raw_week25_or26_and_unknown_like_domain_rejected(self):
        f=self.f();events,rows,qs=self.world()
        for week in [25,26]:
            with self.assertRaises(ValueError):f.build_profiles([(week,events)],rows,qs)
        events[0]['like']=1
        with self.assertRaises(ValueError):f.build_profiles([(0,events)],rows,qs)
    def test_runtime_test_artifact_rejected(self):
        from vkgr.profile_like_audit import safe
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'test'/'week_26.json';p.parent.mkdir();p.write_text('{}')
            with self.assertRaises(ValueError):safe(p)

    def test_bound_tensor_target_fields_do_not_change_input_and_bad_bucket_type_rejected(self):
        import torch
        from vkgr.retrieval_profile import tensors
        row={'user_id':1,'example_id':0,'history':[10],'history_positions':[[0,0]],'side':[[1,5,10]],'target':11,'target_position':[0,1],'profile_ids':[1,1,1]}
        h,s=tensors([row],{10:2},1,'f3_f4_recency');changed={**row,'target':999,'target_feedback':63,'future_like':True};hh,ss=tensors([changed],{10:2},1,'f3_f4_recency')
        self.assertTrue(torch.equal(h,hh) and torch.equal(s,ss))
        for ids in [[1.5,1,1],[True,1,1],[1,1]]:
            with self.assertRaises(ValueError):tensors([{**row,'profile_ids':ids}],{10:2},1,'f3_f4_recency')

class Model(unittest.TestCase):
    def test_missing_model_module(self):
        self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_profile'))
    def test_shared_init_forward_bitwise_rng_and_tables(self):
        import torch
        from vkgr.retrieval_seqm1 import make_seqm1_model
        from vkgr.retrieval_profile import make_profile_model
        torch.set_num_threads(1);content=torch.ones(8,64);content[0].zero_();h=torch.tensor([[2,3]]);side=torch.tensor([[[0,5,10,0,0,0],[1,5,10,0,0,0]]]);extra=torch.tensor([[[1,1,0],[1,1,0]]]);packed=torch.cat([side,extra],-1)
        for arm in ['u2','u3']:
            torch.manual_seed(42);old=make_seqm1_model(content,[2,3,4],50,'mean');state=torch.get_rng_state().clone()
            torch.manual_seed(42);new=make_profile_model(content,[2,3,4],50,arm);self.assertTrue(torch.equal(state,torch.get_rng_state()))
            for n,p in old.named_parameters():self.assertTrue(torch.equal(p,dict(new.base.named_parameters())[n]))
            old.eval();new.eval();self.assertTrue(torch.equal(old.users(h,side),new.users(h,packed)))
            self.assertTrue(all(torch.count_nonzero(p)==0 for n,p in new.named_parameters() if n.startswith('profile_tables.')))
    def test_profile_outside_gate_can_change_users_and_bad_ids_rejected(self):
        import torch
        from vkgr.retrieval_profile import make_profile_model
        torch.set_num_threads(1);torch.manual_seed(42);content=torch.ones(8,64);content[0].zero_();m=make_profile_model(content,[2,3,4],50,'u2').eval();h=torch.tensor([[2]]);s=torch.tensor([[[0,5,10,0,0,0,1,0,0]]]);before=m.users(h,s).detach()
        with torch.no_grad():m.profile_tables[0].weight[1].fill_(.25)
        self.assertTrue(torch.allclose(m.users(h,s),before+.25));bad=s.clone();bad[0,0,6]=17
        with self.assertRaises(ValueError):m.users(h,bad)
        with self.assertRaises(ValueError):m.users(h,s[...,:6])
if __name__=='__main__':unittest.main()
