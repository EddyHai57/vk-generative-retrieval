"""Fresh deployment inventory must reject occupied run IDs and destinations."""
import importlib.util,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module():
    path=ROOT/'scripts/sid_learn_deploy.py'
    assert path.is_file(),'missing versioned deployment entry'
    spec=importlib.util.spec_from_file_location('structure_deploy',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class DeploymentTests(unittest.TestCase):
    def test_inventory_requires_every_run_and_destination_to_be_fresh(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'queue/claimed').mkdir(parents=True)
            req={'root':str(root),'source':str(root/'source'),'destination':str(root/'code'),'ops':str(root/'ops'),'branch':'codex/structure-public','commit':'a'*40,'files':{'src/a.py':'b'*64},'run_ids':['sid-learn-fit-v1']}
            inv=m.inventory(req);self.assertTrue(inv['clear'])
            (root/'queue/claimed/001-sid-learn-fit-v1.json').write_text('{}')
            self.assertFalse(m.inventory(req)['clear'])
            req['run_ids']=['sid-learn-fit-verify-v1'];(root/'code').mkdir()
            self.assertFalse(m.inventory(req)['clear'])
            with self.assertRaises(ValueError):m.validate({**req,'branch':'main'})
            with self.assertRaises(ValueError):m.validate({**req,'files':{'../escape':'b'*64}})
if __name__=='__main__':unittest.main()
