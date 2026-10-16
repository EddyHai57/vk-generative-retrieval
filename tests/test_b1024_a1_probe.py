import importlib.util
from pathlib import Path
import unittest
class ProbeChecks(unittest.TestCase):
 def module(self):
  p=Path(__file__).parents[1]/'scripts/gtok02a1_cuda_check.py';self.assertTrue(p.exists());s=importlib.util.spec_from_file_location('probe',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_real_shape_rows_use_only_synthetic_catalog_ids(self):
  m=self.module();rows=m.synthetic_rows(list(range(1,100)),512,50);self.assertEqual(len(rows),512);self.assertTrue(all(len(r['history'])==50 and set(r['history'])<=set(range(1,100)) for r in rows))
 def test_declaration_uses_measured_not_old6656(self):
  m=self.module();self.assertEqual(m.declare_vram(6656),7680);self.assertEqual(m.declare_vram(5701),6656)
  with self.assertRaises(ValueError):m.declare_vram(float('nan'))
 def test_report_uses_actual_stepresult_microbatch_sizes(self):
  m=self.module()
  from vkgr.generative_training import StepResult
  result=m.step_report(StepResult(1.0,512,(256,256)),.1)
  self.assertEqual(result['microbatches'],(256,256));self.assertTrue(result['finite_loss'])
if __name__=='__main__':unittest.main()
