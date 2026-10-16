import unittest
import numpy as np
from scripts.sid_scale_benchmark import synthetic_blocks, allocation_estimate, suffix_summary, streaming_fit

class ScaleTests(unittest.TestCase):
 def test_synthetic_replay_unit_norm_shape(self):
  a=np.concatenate(list(synthetic_blocks(11,block=4,seed=17)))
  b=np.concatenate(list(synthetic_blocks(11,block=4,seed=17)))
  self.assertEqual(a.shape,(11,64));self.assertEqual(a.dtype,np.float32)
  np.testing.assert_array_equal(a,b);np.testing.assert_allclose(np.linalg.norm(a,axis=1),1,atol=1e-6)
 def test_root_edge_sort_memory_is_explicit_not_measured(self):
  e=allocation_estimate(1250000,256)
  self.assertEqual(e['edge_count'],320000000)
  self.assertGreater(e['known_array_bytes'],320000000*32)
 def test_suffix_distinguishes_semantic_and_full_codes(self):
  r=suffix_summary(np.array([[0,0,0],[0,0,0],[1,0,0]]))
  self.assertEqual(r,{'occupied_leaves':2,'max_leaf':2,'suffix_vocab':2,'items_needing_suffix':2,'excess_collisions':1})
 def test_streaming_count_and_no_balanced_claim(self):
  r=streaming_fit(29,k=3,block=7,seed=2)
  self.assertEqual(r['assigned_items'],29);self.assertEqual(sum(r['cluster_counts']),29)
  self.assertFalse(r['balanced']);self.assertTrue(r['finite'])
 def test_invalid_sizes(self):
  with self.assertRaises(ValueError):list(synthetic_blocks(0))
  with self.assertRaises(ValueError):streaming_fit(3,k=4)

class WorkerTests(unittest.TestCase):
 def test_worker_is_separate_and_single_threaded(self):
  import os
  from scripts.sid_scale_benchmark import run_method
  r=run_method('streaming',29,k=3,block=7,wall_seconds=20)
  self.assertEqual(r['status'],'COMPLETE');self.assertNotEqual(r['pid'],os.getpid())
  self.assertEqual(r['torch_threads'],1);self.assertEqual(r['torch_interop_threads'],1)
  self.assertTrue(r['blas_threads']);self.assertTrue(all(x==1 for x in r['blas_threads']))
  self.assertEqual(r['assigned_items'],29);self.assertEqual(r['suffix'],'UNKNOWN')
 def test_large_baselines_skip_without_spawning(self):
  from unittest.mock import patch
  from scripts.sid_scale_benchmark import run_method
  with patch('subprocess.Popen',side_effect=AssertionError('must not spawn')):
   for method in ('bhkm','rqvae'):
    for n in ((1250000,12470000) if method=='bhkm' else (12470000,)):
     r=run_method(method,n)
     self.assertEqual(r['status'],'SKIPPED_RESOURCE')
     self.assertFalse(r['estimate']['measured']);self.assertEqual(r['suffix'],'UNKNOWN')
 def test_caps_cannot_be_relaxed(self):
  from scripts.sid_scale_benchmark import run_method
  for kwargs in ({'wall_seconds':61},{'rss_mib':4097},{'wall_seconds':0},{'rss_mib':0}):
   with self.assertRaises(ValueError):run_method('streaming',29,**kwargs)
 def test_parent_wall_censoring_not_complete(self):
  from scripts.sid_scale_benchmark import run_method
  r=run_method('streaming',29,k=3,wall_seconds=0.001)
  self.assertEqual(r['status'],'CENSORED_WALL');self.assertEqual(r['suffix'],'UNKNOWN')
 def test_parent_rss_censoring_not_complete(self):
  from scripts.sid_scale_benchmark import run_method
  r=run_method('streaming',100000,k=3,rss_mib=1,wall_seconds=20)
  self.assertEqual(r['status'],'CENSORED_RSS');self.assertEqual(r['suffix'],'UNKNOWN')
 def test_small_full_bhkm_measures_suffix(self):
  from scripts.sid_scale_benchmark import run_method
  r=run_method('bhkm',29,k=3,block=7,wall_seconds=20)
  self.assertEqual(r['status'],'COMPLETE')
  self.assertIsInstance(r['suffix'],dict)
  self.assertGreaterEqual(r['suffix']['occupied_leaves'],1)
 def test_rq_materialization_estimate_is_not_invented_tensor_copy(self):
  from scripts.sid_scale_benchmark import allocation_estimate
  e=allocation_estimate(1250000,256,'rqvae')
  self.assertEqual(e['known_array_bytes'],1250000*64*4*2)
  self.assertFalse(e['measured'])

class ControllerTests(unittest.TestCase):
 def test_controller_propagates_wall_to_all_six_workers(self):
  from unittest.mock import patch,call
  from scripts import sid_scale_benchmark as bench
  with patch('sys.argv',['benchmark','--wall-seconds','0.01']),patch.object(bench,'run_method',return_value={'status':'MOCK'}) as runner,patch('builtins.print'):
   bench.main()
  self.assertEqual(runner.call_args_list,[call(method,n,wall_seconds=0.01) for n in bench.SIZES for method in bench.METHODS])
 def test_worker_only_flags_rejected_before_any_work(self):
  from unittest.mock import patch
  from scripts import sid_scale_benchmark as bench
  for flag,value in (('--n','29'),('--k','3'),('--block','7'),('--seed','42')):
   with self.subTest(flag=flag),patch('sys.argv',['benchmark',flag,value]),patch.object(bench,'run_method',return_value={}) as runner,patch('sys.stderr'):
    with self.assertRaises(SystemExit) as caught:bench.main()
    self.assertEqual(caught.exception.code,2)
    runner.assert_not_called()
 def test_controller_rejects_invalid_wall_before_any_work(self):
  from unittest.mock import patch
  from scripts import sid_scale_benchmark as bench
  for value in ('0','61','-1','nan'):
   with self.subTest(wall=value),patch('sys.argv',['benchmark','--wall-seconds',value]),patch.object(bench,'run_method',return_value={}) as runner,patch('sys.stderr'):
    with self.assertRaises(SystemExit) as caught:bench.main()
    self.assertEqual(caught.exception.code,2)
    runner.assert_not_called()
