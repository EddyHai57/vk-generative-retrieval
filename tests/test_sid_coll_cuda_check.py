import importlib.util,math,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'scripts/sid_coll_cuda_check.py'
class CUDAContract(unittest.TestCase):
 def load(self):
  s=importlib.util.spec_from_file_location('coll_cuda',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_ceiling(self):
  m=self.load();self.assertEqual(m.declare_vram(2000),2304)
  for x in [0,-1,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):m.declare_vram(x)
 def test_fullshape(self):
  m=self.load();rows=m.synthetic_rows([1,2,3],512,50);self.assertEqual(len(rows),512);self.assertEqual(len(rows[0]['history']),50);self.assertNotEqual(rows[0]['target'],rows[0]['history'][-1])
 def test_seed_only(self):
  m=self.load();base={'seed':42,'beam':100};m.check_seed_configs({42:base,2024:{**base,'seed':2024},2025:{**base,'seed':2025}})
  with self.assertRaises(ValueError):m.check_seed_configs({42:base,2024:{**base,'seed':2024,'beam':50},2025:{**base,'seed':2025}})
if __name__=='__main__':unittest.main()
