import unittest,tempfile,json
from pathlib import Path
from analysis.engineering_queue import inventory,digest,new
class EngineeringTests(unittest.TestCase):
 def test_once_only_intent_and_queue_identity(self):
  with tempfile.TemporaryDirectory() as t:
   r={'root':t,'ops':t,'job':{'run_id':'x'}}
   self.assertFalse(inventory(r)['intent_exists']);new(Path(t)/'intent-x.json',{})
   self.assertTrue(inventory(r)['intent_exists'])
   with self.assertRaises(FileExistsError):new(Path(t)/'intent-x.json',{})
 def test_canonical_job_binding(self):
  self.assertEqual(digest({'a':1,'b':2}),digest({'b':2,'a':1}))
  self.assertNotEqual(digest({'a':1}),digest({'a':2}))
class SubmitGuards(unittest.TestCase):
 def test_submit_without_dry_receipt_is_rejected_before_publish(self):
  import io,types,sys
  from contextlib import redirect_stdout
  from unittest.mock import patch
  from analysis import engineering_queue as q
  with tempfile.TemporaryDirectory() as t:
   job={'run_id':'x','schema':'vkgr-queue-job-v1','kind':'ops','cpu_cores':1,'no_admission':False,'vram_mib':0}
   req={'root':t,'ops':t,'job':job,'action':'submit','scheduler_code':t}
   fake=types.ModuleType('vkgr_ops');fake.launcher=types.SimpleNamespace(prepare=lambda _:{});fake.scheduler=types.SimpleNamespace(submit=lambda *_:self.fail('must not publish without dry receipt'))
   with patch.object(sys,'stdin',io.StringIO(json.dumps(req))),patch.dict(sys.modules,{'vkgr_ops':fake}),redirect_stdout(io.StringIO()):
    with self.assertRaises(ValueError):q.main()
