import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
class ProfileTests(unittest.TestCase):
    def test_frozen_config_and_native_hash(self):
        from analysis import sgns_speedup_profile as p
        c=p.frozen_configuration()
        self.assertEqual(c['config'],dict(dim=64,window=5,epochs=5,negatives=5,batch_size=4096,seed=42,initial_lr=.025,final_lr_ratio=.0001))
        self.assertFalse(c['high_frequency_subsampling'])
    def test_pair_reconnection_cannot_use_histogram(self):
        from analysis import sgns_speedup_profile as p
        a={1:[1,2,1,3,2,3]};b={1:[1,1,2,2,3,3]}
        self.assertNotEqual(p.pair_count(a,1),p.pair_count(b,1))
        self.assertEqual(p.keep_probability(1.,.001),( .001**.5+.001))
        with self.assertRaises(ValueError):p.keep_probability(0,.001)
    def test_scenario_arithmetic_has_unknowns(self):
        from analysis import sgns_speedup_profile as p
        s=p.scenarios()
        self.assertEqual(s['baseline']['positive_pairs'],18787015620)
        self.assertAlmostEqual(s['baseline']['fit_hours'],54.404506509735405)
        self.assertEqual(s['C1'][0]['positive_pairs'],'UNKNOWN')
        self.assertEqual(s['C2'][0]['fit_hours'],'UNKNOWN')
        self.assertEqual(s['C3_epochs'][0]['positive_pairs'],11272209372)
    def test_original_fit_bounded_no_vector_and_restores_trace(self):
        from analysis import sgns_speedup_profile as p
        import torch
        before=sys.gettrace();threads=torch.get_num_threads()
        r=p.profile_prefix(p.synthetic_inputs(),'cpu',2)
        self.assertEqual(r['completed_batches'],2)
        self.assertEqual(r['positive_pairs'],8192)
        self.assertFalse(r['vectors_returned']);self.assertFalse(r['scientific_verified'])
        self.assertEqual(r['device'],'cpu');self.assertEqual(r['cuda_kernel_us'],'UNKNOWN')
        self.assertEqual(sys.gettrace(),before);self.assertEqual(torch.get_num_threads(),threads)
        self.assertIn('negative_sampling',r['instrumented_sections_seconds'])
        self.assertEqual(r['pair_loop_intervals'],1)
        self.assertGreaterEqual(r['instrumented_pair_loop_between_batches_seconds'],0.)
    def test_subset_requires_frozen_runtime_lock(self):
        from analysis import sgns_speedup_profile as p
        with self.assertRaisesRegex(ValueError,'frozen runtime lock required'):
            p.subset_inputs(Path('/workspace/nonexistent'), 'a'*64, None)
    def test_final_batch_error_is_not_success(self):
        from analysis import sgns_speedup_profile as p
        from unittest.mock import patch
        original=p.native();loss=original.sgns_loss
        for batches in (1,8):
            calls=0
            def injected(*args):
                nonlocal calls
                calls+=1
                if calls==batches:raise RuntimeError('injected final batch failure')
                return loss(*args)
            with patch.object(original,'sgns_loss',side_effect=injected):
                with self.assertRaisesRegex(RuntimeError,'injected final batch failure'):
                    p.profile_prefix(p.synthetic_inputs(),'cpu',batches)
    def test_too_small_prefix_refused_before_fit(self):
        from analysis import sgns_speedup_profile as p
        ids,_,content=p.synthetic_inputs()
        with self.assertRaisesRegex(ValueError,'insufficient original positive pairs'):
            p.profile_prefix((ids,{1:[1,2]},content),'cpu',1)
    def test_cuda_event_api_uses_time_range_not_duration(self):
        from analysis import sgns_speedup_profile as p
        from types import SimpleNamespace as S
        events=[S(device_type='DeviceType.CUDA',name='kernel',time_range=S(elapsed_us=lambda:12.)),
                S(device_type='DeviceType.CUDA',name='Memcpy HtoD',time_range=S(elapsed_us=lambda:3.)),
                S(device_type='DeviceType.CPU',name='cpu',time_range=S(elapsed_us=lambda:100.))]
        self.assertEqual(p.cuda_times(events,'cuda'),(12.,3.))
        self.assertEqual(p.cuda_times(events,'cpu'),('UNKNOWN','UNKNOWN'))
    def test_cli_dry_illegal_and_cuda_fail_closed(self):
        env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
        for args,code in [(['dry'],0),(['illegal'],2)]:
            r=subprocess.run([sys.executable,'-B','-m','analysis.sgns_speedup_profile',*args],cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertEqual(r.returncode,code,r.stderr)
            if code==0:self.assertFalse(json.loads(r.stdout)['science_training'])
        from analysis import sgns_speedup_profile as p
        import torch
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(ValueError,'CUDA unavailable'):p.profile_prefix(p.synthetic_inputs(),'cuda',1)
if __name__=='__main__':unittest.main()
