import importlib.util
from pathlib import Path
import tempfile
import unittest
class InventoryChecks(unittest.TestCase):
 def test_metadata_reader_rejects_checkpoint_and_test_artifacts(self):
  p=Path(__file__).with_name('successor_inventory.py');self.assertTrue(p.exists())
  s=importlib.util.spec_from_file_location('inventory',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
  with tempfile.TemporaryDirectory() as temp:
   for name in ('test.json','checkpoint.json','weights.pt'):
    with self.assertRaises(ValueError):m.read(Path(temp)/name)
if __name__=='__main__':unittest.main()
