import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

class ControlEntry(unittest.TestCase):
    def module(self):
        p=Path(__file__).with_name('seqm1_night_control.py');self.assertTrue(p.exists(),'pinned module controller missing')
        spec=importlib.util.spec_from_file_location('control',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    def test_python_c_bridge_preserves_module_file_and_sibling_config(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'core.py';p.write_text("from pathlib import Path\ndef frozen():return Path(__file__).with_name('freeze.json').read_text()\n")
            p.with_name('freeze.json').write_text('pinned-config')
            core=m.load_core(p,hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(core.frozen(),'pinned-config')
            with self.assertRaises(ValueError):m.load_core(p,'0'*64)

if __name__=='__main__':unittest.main()
