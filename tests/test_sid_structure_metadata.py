import importlib.util
from pathlib import Path
import unittest,tempfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def module():
    s=importlib.util.spec_from_file_location('structure_meta',ROOT/'scripts/sid_structure_metadata.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class MetadataTests(unittest.TestCase):
    def test_no_real_data_or_cuda_entry_can_run_in_metadata_bridge(self):
        m=module()
        for action in ('preflight','cuda-preview','train','verify','driver'):
            with self.assertRaises(ValueError):m.metadata_action(action)
        self.assertEqual(m.metadata_action('dry-run'),'dry-run')
        self.assertEqual(m.metadata_action('submit'),'submit')

    def test_only_small_own_run_json_objects_can_be_read(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();ops=root/'ops';ops.mkdir();runs=root/'runs';run=runs/'sid-structure-fit-v1';run.mkdir(parents=True)
            p=run/'audit.json';p.write_text('{"count":3}')
            self.assertEqual(m.metadata_file(p,ops,runs)['value'],{'count':3})
            for name in ('mapping.npz','predictions.json','vectors.json'):
                q=run/name;q.write_text('{}')
                with self.assertRaises(ValueError):m.metadata_file(q,ops,runs)
            p.write_text('[1,2,3]')
            with self.assertRaises(ValueError):m.metadata_file(p,ops,runs)
            foreign=runs/'old-run';foreign.mkdir();q=foreign/'manifest.json';q.write_text('{}')
            with self.assertRaises(ValueError):m.metadata_file(q,ops,runs)

if __name__=='__main__':unittest.main()
