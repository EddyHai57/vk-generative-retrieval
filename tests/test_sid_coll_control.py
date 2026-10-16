import importlib.util,json,tempfile,unittest
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'scripts/sid_coll_control.py'
class Control(unittest.TestCase):
 def load(self):
  s=importlib.util.spec_from_file_location('ctl',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 def test_safe_new_paths(self):
  m=self.load()
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'new.json';m.new(p,{'x':1})
   with self.assertRaises(FileExistsError):m.new(p,{'x':2})
   self.assertEqual(json.loads(p.read_text()),{'x':1})
 def test_no_path_traversal(self):
  m=self.load()
  for p in ['/etc/a','../a','a/../b']:
   with self.assertRaises(ValueError):m.relative(p)
  self.assertEqual(m.relative('scripts/a.py'),'scripts/a.py')
 def test_run_ids_unique(self):
  m=self.load();ids=m.ids();self.assertEqual(len(set(ids)),len(ids));self.assertTrue(all(x.startswith('sid-coll-') for x in ids))
if __name__=='__main__':unittest.main()
