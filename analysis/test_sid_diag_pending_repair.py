import importlib.util,json,tempfile,unittest
from pathlib import Path
class RepairTests(unittest.TestCase):
 def module(self):
  p=Path(__file__).with_name('sid_diag_pending_repair.py');self.assertTrue(p.exists(),'new repair module absent')
  s=importlib.util.spec_from_file_location('repair',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def fixture(self,base):
  q=base/'queue';(base/'runs').mkdir()
  for k in ('pending','claimed','rejected'):(q/k).mkdir(parents=True)
  spec={'run_id':'sid-fix-s0-first-v1','config':'fixed','cpu_cores':1};p=q/'pending'/'01792142148793391058-sid-fix-s0-first-v1.json';p.write_text(json.dumps(spec));return p,spec
 def test_only_schema_and_backup(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as t:
   b=Path(t);p,s=self.fixture(b);raw=p.read_bytes();pin=m.sha(p)
   x=m.repair(b,p,pin,s,b/'original.json',approved=True)
   self.assertEqual(json.loads(p.read_bytes()),{'schema':'vkgr-queue-job-v1',**s});self.assertEqual((b/'original.json').read_bytes(),raw)
   self.assertEqual(x['old_sha256'],pin);self.assertFalse((b/'runs'/s['run_id']).exists())
   with self.assertRaises(ValueError):m.repair(b,p,pin,s,b/'again.json',approved=True)
 def test_guard_noapproval_wrongpin_claimed_run(self):
  m=self.module()
  for mode in ('unapproved','pin','claimed','run','different'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
    b=Path(t);p,s=self.fixture(b);pin=m.sha(p);raw=p.read_bytes();expected=dict(s)
    if mode=='claimed':(b/'queue'/'claimed'/p.name).write_text('{}')
    if mode=='run':(b/'runs'/s['run_id']).mkdir()
    if mode=='different':expected['cpu_cores']=2
    with self.assertRaises((ValueError,FileExistsError)):
     m.repair(b,p,'0'*64 if mode=='pin' else pin,expected,b/'original.json',approved=mode!='unapproved')
    self.assertEqual(p.read_bytes(),raw);self.assertFalse((b/'original.json').exists())
if __name__=='__main__':unittest.main()
