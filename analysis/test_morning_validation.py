import importlib.util
from pathlib import Path
import unittest

class FrozenRuleChecks(unittest.TestCase):
    def module(self):
        p=Path(__file__).with_name('morning_validation_read.py');self.assertTrue(p.exists(),'morning validation reader missing')
        s=importlib.util.spec_from_file_location('morning',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    def test_mean_row_and_m2_design_trigger_are_separate(self):
        m=self.module()
        self.assertEqual(m.mean_rules([6,6,6],.049,[.05,.05,.01])['row'],2)
        self.assertEqual(m.mean_rules([6,6,6],.05,[.05,.05,.01])['row'],3)
        x=m.mean_rules([0,-5,0],1.,[.04,.05,.04]);self.assertEqual(x['row'],4);self.assertFalse(x['m2_design_performance_and_branch'])
        self.assertEqual(m.mean_rules([-6,-6,-6],0.,[1,1,1])['row'],1)
    def test_q1_uses_only_independent_supplement_seeds(self):
        m=self.module();self.assertEqual(m.tok_state([6,6]),'UP');self.assertEqual(m.tok_state([-6,-6]),'DOWN');self.assertEqual(m.tok_state([6,5]),'FLAT')
        self.assertEqual(m.q1('FLAT','UP'),'still_rising');self.assertEqual(m.q1('DOWN','DOWN'),'peak_256');self.assertEqual(m.q1('FLAT','DOWN'),'no_further_gain')
        self.assertEqual(m.q1('T_UP','DOWN'),'undetermined')
    def test_missing_seed_never_produces_definite_conclusion(self):
        m=self.module();self.assertEqual(m.mean_rules([10,10],.01,[1,1])['row'],'UNKNOWN');self.assertEqual(m.tok_state([10]),'UNKNOWN')

if __name__=='__main__':unittest.main()
