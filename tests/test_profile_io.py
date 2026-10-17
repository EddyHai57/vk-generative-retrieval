import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
class IOTests(unittest.TestCase):
    def test_missing_runtime_guard(self):self.assertIsNotNone(importlib.util.find_spec('vkgr.profile_io'))
    def test_all_open_apis_reject_test_reads_and_old_checkpoint(self):
        from vkgr.profile_io import input_guard
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();test=root/'test'/'week_26.json';test.parent.mkdir();test.write_text('{}');old=root/'old.pt';old.write_bytes(b'old');owned=root/'new';owned.mkdir();new=owned/'best.pt';new.write_bytes(b'new')
            with input_guard(owned):
                for call in [lambda:open(test).read(),lambda:test.read_text(),lambda:os.open(test,os.O_RDONLY),lambda:old.read_bytes()]:
                    with self.assertRaises(ValueError):call()
                self.assertEqual(new.read_bytes(),b'new')
            self.assertEqual(old.read_bytes(),b'old')
    def test_default_guard_never_permits_checkpoint(self):
        from vkgr.profile_io import input_guard
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'best.pt';p.write_bytes(b'new')
            with input_guard():
                with self.assertRaises(ValueError):p.read_bytes()
    def test_symlink_alias_cannot_hide_test_file(self):
        from vkgr.profile_io import input_guard
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();p=root/'test'/'week_26.json';p.parent.mkdir();p.write_text('{}');alias=root/'validation.json';alias.symlink_to(p)
            with input_guard():
                with self.assertRaises(ValueError):alias.read_text()

    def test_installed_torch_operator_source_is_not_dataset_test(self):
        from vkgr.profile_io import input_guard
        spec=importlib.util.find_spec('torch')
        source=Path(spec.origin).resolve().parent/'_inductor'/'test_operators.py'
        self.assertTrue(source.is_file())
        with input_guard():
            self.assertIn(b'torch',source.read_bytes())
        with tempfile.TemporaryDirectory() as tmp:
            fake=Path(tmp).resolve()/'test_operators.py';fake.write_text('not dependency code')
            with input_guard():
                with self.assertRaises(ValueError):fake.read_text()

if __name__=='__main__':unittest.main()
