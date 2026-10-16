import importlib.util,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'scripts/sid_coll_observer_handoff.py'
class Handoff(unittest.TestCase):
 def test_identity_and_original_deadline(self):
  s=importlib.util.spec_from_file_location('h',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  p={'created_epoch':1000,'deadline_epoch':173800};r={'pid':123,'deadline_epoch':173800}
  self.assertEqual(m.validate(p,r,['/workspace/vkgr-sid-coll-g-v1/scripts/sid_coll_ops.py','driver','--plan',str(m.PLAN),'--plan-sha','a'*64],'a'*64),173800)
  with self.assertRaises(ValueError):m.validate(p,r,['unrelated'],'a'*64)
  with self.assertRaises(ValueError):m.validate(p,{**r,'deadline_epoch':200000},[],'a'*64)
if __name__=='__main__':unittest.main()
