import json,tempfile,unittest
from pathlib import Path
from analysis.fresh_deploy import inventory,validate
class DeployTest(unittest.TestCase):
 def req(self,t):return {'root':str(Path(t)/'opsroot'),'source':str(Path(t)/'source'),'destination':str(Path(t)/'newcode'),'ops':str(Path(t)/'newops'),'branch':'codex/new-batch','commit':'a'*40,'files':{'src/x.py':'b'*64},'run_ids':['x-train']}
 def test_inventory_detects_existing_queue_and_run(self):
  with tempfile.TemporaryDirectory() as t:
   r=self.req(t);root=Path(r['root']);(root/'queue/claimed').mkdir(parents=True);(root/'runs/x-train').mkdir(parents=True)
   (root/'queue/claimed/001-x-train.json').write_text('{}')
   got=inventory(r);self.assertFalse(got['clear']);self.assertTrue(got['jobs']['x-train']['run_exists'])
 def test_private_paths_and_safe_file_names(self):
  with tempfile.TemporaryDirectory() as t:
   r=self.req(t);validate(r)
   r['files']={'../data':'b'*64}
   with self.assertRaises(ValueError):validate(r)
