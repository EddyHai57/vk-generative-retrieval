import importlib.util,json,tempfile,unittest
from pathlib import Path
class CompletionTests(unittest.TestCase):
 def module(self):
  p=Path(__file__).with_name('sid_diag_completion_read.py');self.assertTrue(p.exists(),'event reader absent')
  s=importlib.util.spec_from_file_location('completion',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_wait_is_driven_by_event(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);rid='sid-fix-s0-first-v1';(root/rid).mkdir();calls=[]
   def event(left):
    calls.append(left);(root/rid/'terminal.json').write_text(json.dumps({'status':'succeeded'}))
   out=m.wait_for_terminals(root,[rid],event,10,lambda:0)
   self.assertTrue(out['complete']);self.assertEqual(len(calls),1)
 def test_finished_does_not_wait(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);rid='sid-fix-s0-first-v1';(root/rid).mkdir();(root/rid/'terminal.json').write_text('{"status":"failed"}')
   out=m.wait_for_terminals(root,[rid],lambda n:self.fail('already terminal'),10,lambda:0)
   self.assertTrue(out['complete']);self.assertEqual(out['runs'][rid]['terminal']['status'],'failed')
 def test_timeout_never_changes_run(self):
  m=self.module()
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);rid='sid-fix-s0-first-v1';(root/rid).mkdir();clock=[0]
   out=m.wait_for_terminals(root,[rid],lambda n:clock.__setitem__(0,11),10,lambda:clock[0])
   self.assertFalse(out['complete']);self.assertEqual(list((root/rid).iterdir()),[])
   with self.assertRaises(ValueError):m.snapshot(root,['unregistered'])
if __name__=='__main__':unittest.main()
