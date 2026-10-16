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
