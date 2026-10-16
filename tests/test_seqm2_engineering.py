import importlib.util,tempfile,unittest
from pathlib import Path
import torch
class EngineeringTests(unittest.TestCase):
 def test_natural_boundary_receipt_uses_recorded_optimizer_steps(self):
  from vkgr import retrieval_seqm2_run as run
  torch.set_num_threads(1)
  path=Path(__file__).resolve().parents[1]/'scripts/seqm2_cuda_check.py';spec=importlib.util.spec_from_file_location('m2_engineering',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  with tempfile.TemporaryDirectory(prefix='m2-boundary-') as d:
   verified,steps,window=m.small_replay(run,Path(d),device='cpu')
  self.assertEqual(verified['status'],'verified');self.assertGreaterEqual(steps,201);self.assertEqual(len(window['head_counts']),4);self.assertEqual(window['optimizer_steps_101_200_observed'],100);self.assertTrue(window['diagnostic_only'])
if __name__=='__main__':unittest.main()
