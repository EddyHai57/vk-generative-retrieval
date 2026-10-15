import importlib.util
from pathlib import Path
import unittest

class Checks(unittest.TestCase):
    def module(self):
        p=Path(__file__).with_name('seqm1_prepare.py');self.assertTrue(p.exists(),'M1 preparation helper missing')
        s=importlib.util.spec_from_file_location('prep',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    def test_declaration_exact_115_percent_round256(self):
        m=self.module();self.assertEqual(m.declared_vram(1024),1280);self.assertEqual(m.declared_vram(2000),2304)
    def test_zero_or_nonfinite_measurement_refused(self):
        m=self.module()
        for x in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):m.declared_vram(x)

if __name__=='__main__':unittest.main()
