import unittest
from analysis import coll_a1_bridge as b
class BridgeTests(unittest.TestCase):
 def test_exact_metadata_actions(self):
  req={'deployment':'/fake/code','ops':'/fake/ops','source_commit':'a'*40,'script_sha256':'b'*64,'request_path':'/fake/ops/request.json','request_sha256':'c'*64,'action':'dry-run'}
  b.validate(req)
  for patch in ({'action':'cuda-preview'},{'request_path':'/other/request.json'},{'request_path':'/fake/ops/../escape.json'}):
   with self.assertRaises(ValueError):b.validate({**req,**patch})
 def test_unknown_field_rejected(self):
  with self.assertRaises(ValueError):b.validate({'action':'submit','extra':True})
if __name__=='__main__':unittest.main()
