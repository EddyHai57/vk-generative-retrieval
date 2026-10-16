import unittest,tempfile
from pathlib import Path
from analysis.ops_metadata import validate_operation
class OpsMetadataTests(unittest.TestCase):
 def test_only_reviewed_metadata_actions_allowed(self):
  with tempfile.TemporaryDirectory() as t:
   r={'deployment':t,'ops':t,'source_commit':'a'*40,'script':'scripts/biconfirm_ops.py','script_sha256':'b'*64,'args':['submit']}
   validate_operation(r)
   for bad in (['train'],['preflight'],['cuda-check']):
    with self.assertRaises(ValueError):validate_operation({**r,'args':bad})
 def test_no_path_escape(self):
  with self.assertRaises(ValueError):validate_operation({'deployment':'/code','ops':'/ops','source_commit':'a'*40,'script':'../model.py','script_sha256':'b'*64,'args':['submit']})
class HardenedBridge(unittest.TestCase):
 def test_interpreter_and_import_environment_cannot_be_overridden(self):
  import sys
  from analysis.ops_metadata import execution_args
  r={'deployment':'/code','script':'scripts/biconfirm_ops.py','args':['submit'],'python':sys.executable}
  self.assertEqual(execution_args(r)[0],sys.executable)
  with self.assertRaises(ValueError):execution_args({**r,'python':'/tmp/fake-python'})
  with self.assertRaises(ValueError):execution_args({**r,'env':{'PYTHONHOME':'/tmp/untrusted'}})
 def test_read_rejects_dotdot_and_symlink_escape(self):
  from analysis.ops_metadata import bounded_path
  with tempfile.TemporaryDirectory() as t:
   root=Path(t)/'ops';root.mkdir();out=Path(t)/'outside.json';out.write_text('{}');(root/'link').symlink_to(Path(t),target_is_directory=True)
   for p in (root/'../outside.json',root/'link/outside.json'):
    with self.assertRaises(ValueError):bounded_path(p,[root])
