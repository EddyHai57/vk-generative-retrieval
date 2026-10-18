"""Small engineering fits exercise real L2 updates and fail-closed output rules."""
import importlib,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from vkgr import sid_learn_v1 as contract,sid_rqvae as rq
ROOT=Path(__file__).resolve().parents[1]
class LearnFitTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'src/vkgr/sid_learn_fit.py').is_file(),'missing learnable fit entry')
        self.fit=importlib.import_module('vkgr.sid_learn_fit');torch.set_num_threads(1)
    def inputs(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(16,64)).astype('float32');target=rng.normal(size=(16,32)).astype('float32');mask=np.zeros(16,dtype=bool);mask[0]=True
        return x,target,mask
    def run_small(self,**kwargs):
        return self.fit.fit_learned(*self.inputs(),config=contract.expected_config(),engineering=True,options={'K':4,'batch_size':8,'max_steps':3,'check_every':1},**kwargs)
    def test_formal_shortcut_or_wrong_population_stops_before_model(self):
        with patch.object(rq,'RQVAE',side_effect=AssertionError('model started')):
            with self.assertRaises(ValueError):self.fit.fit_learned(*self.inputs(),config=contract.expected_config(),engineering=False,options={'max_steps':3})
            with self.assertRaises(ValueError):self.fit.fit_learned(*self.inputs(),config=contract.expected_config(),engineering=False)
    def test_small_complete_fit_logs_all_losses_usage_and_actual_distance_stats(self):
        x,target,mask=self.inputs();saved=(x.copy(),target.copy(),mask.copy())
        out=self.fit.fit_learned(x,target,mask,config=contract.expected_config(),engineering=True,options={'K':4,'batch_size':8,'max_steps':3,'check_every':1})
        self.assertEqual((out['status'],out['step'],out['max_steps'],out['K']),('complete',3,3,4));self.assertIs(out['engineering'],True)
        self.assertNotEqual(out['init_digest'],out['final_state_digest']);self.assertIn('state',out)
        self.assertEqual([v['step'] for v in out['distance_log']],[3]);self.assertEqual(len(out['distance_log'][0]['layers']),3)
        self.assertEqual([v[0] for v in out['usage_log']],[3]);self.assertEqual(len(out['usage_log'][0][1]),3)
        row=out['loss_log'][-1]
        self.assertAlmostEqual(row['total'],row['l2_total']+.1*row['align']+.01*row['div'],places=4)
        for a,b in zip((x,target,mask),saved):np.testing.assert_array_equal(a,b)
    def test_same_seed_full_small_replay_is_bit_exact(self):
        a=self.run_small();b=self.run_small();self.assertEqual(a['init_digest'],b['init_digest']);self.assertEqual(a['final_state_digest'],b['final_state_digest'])
        for k,v in a['state']['state_dict'].items():self.assertTrue(torch.equal(v,b['state']['state_dict'][k]))
    def test_nonfinite_device_bad_flag_never_returns_usable_state(self):
        real=self.fit.learn_loss.loss_terms
        def invalid(*args,**kwargs):
            result=real(*args,**kwargs);result['total']=result['total']*float('nan');return result
        with patch.object(self.fit.learn_loss,'loss_terms',side_effect=invalid):out=self.run_small()
        self.assertEqual(out['status'],'fail_nonfinite');self.assertNotIn('state',out)
    def test_transient_nonfinite_total_is_not_forgotten_before_check(self):
        real=self.fit.learn_loss.loss_terms;calls=[0]
        def invalid_once(*args,**kwargs):
            result=real(*args,**kwargs);calls[0]+=1
            if calls[0]==1:result['total']=result['total']+float('nan')
            return result
        with patch.object(self.fit.learn_loss,'loss_terms',side_effect=invalid_once):
            out=self.fit.fit_learned(*self.inputs(),config=contract.expected_config(),engineering=True,options={'K':4,'batch_size':8,'max_steps':3,'check_every':3})
        self.assertEqual(out['status'],'fail_nonfinite');self.assertNotIn('state',out)

    def test_wall_cap_does_not_return_partial_state(self):
        class Clock:
            count=0
            def __call__(self):
                self.count+=1;return 0. if self.count==1 else 14401.
        out=self.run_small(clock=Clock());self.assertEqual(out['status'],'fail_wall');self.assertNotIn('state',out)
    def test_cap_crossed_during_final_diagnostics_still_rejects_state(self):
        for last_inside in (4,5):
            class Clock:
                count=0
                def __call__(self):
                    self.count+=1;return 0. if self.count<=last_inside else 14401.
            with self.subTest(last_inside=last_inside):
                out=self.run_small(clock=Clock());self.assertEqual(out['status'],'fail_wall');self.assertNotIn('state',out)

    def test_actual_loop_keeps_step2000_final_and_l2_reset_wiring(self):
        from vkgr import sid_learn_pca as pca
        x,_,_=self.inputs();x/=np.linalg.norm(x,axis=1,keepdims=True)
        target=pca.project_pca(x,pca.fit_pca(x));mask=np.zeros(len(x),dtype=bool)
        seen=[];real=self.fit.fixed.reset_dead_codes
        def observe(model,optimizer,residuals,last_used,step,rng,cfg):
            seen.append(step)
            if step in (500,2000):
                for table in last_used:self.assertEqual(int(table.max()),step)
            return real(model,optimizer,residuals,last_used,step,rng,cfg)
        with patch.object(self.fit.fixed,'reset_dead_codes',side_effect=observe):
            out=self.fit.fit_learned(x,target,mask,config=contract.expected_config(),engineering=True,options={'K':2,'batch_size':4,'max_steps':2001,'check_every':100})
        self.assertEqual(out['status'],'complete');self.assertIn(500,seen);self.assertIn(2000,seen)
        self.assertEqual([v['step'] for v in out['distance_log']],[2000,2001])
        self.assertEqual([v[0] for v in out['usage_log']],[2000,2001])

    def test_fixed_diagnostic_events_are_step2000_and_final(self):
        for step,want in ((1,False),(1999,False),(2000,True),(2001,False),(299999,False),(300000,True)):
            self.assertEqual(self.fit.diagnostic_due(step,300000),want)
    def test_changed_l2_reference_bytes_stop_before_initialization(self):
        with patch.object(self.fit,'file_sha256',return_value='0'*64),patch.object(rq,'RQVAE',wraps=rq.RQVAE) as model:
            with self.assertRaisesRegex(ValueError,'fixed L2'):self.run_small()
            model.assert_not_called()
if __name__=='__main__':unittest.main()
