import importlib.util
from pathlib import Path
import unittest
class RecoveryChecks(unittest.TestCase):
 def module(self):
  p=Path(__file__).with_name('transport_hold_recovery.py');self.assertTrue(p.exists())
  s=importlib.util.spec_from_file_location('recovery',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_dead_processes_and_unusable_owned_socket_are_required(self):
  m=self.module();self.assertTrue(m.recovery_allowed(False,False,True,False))
  for args in ((True,False,True,False),(False,True,True,False),(False,False,False,False),(False,False,True,True)):
   self.assertFalse(m.recovery_allowed(*args))
if __name__=='__main__':unittest.main()
