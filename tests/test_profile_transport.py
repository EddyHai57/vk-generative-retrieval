import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
def tool():
    s=importlib.util.spec_from_file_location('profile_transport',ROOT/'scripts/profile_transport.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class TransportTests(unittest.TestCase):
    def test_missing_transport(self):self.assertTrue((ROOT/'scripts/profile_transport.py').is_file())
    def test_real_data_commands_are_not_metadata_actions(self):
        m=tool();r={'deployment':'/fixture/code','ops':'/fixture/ops','commit':'a'*40,'script_sha256':'b'*64,'args':['dry'],'python':'/fixture/python'};m.validate(r)
        for command in ['preflight','cuda-check','driver','train']:
            with self.assertRaises(ValueError):m.validate({**r,'args':[command]})
        with self.assertRaises(ValueError):m.validate({**r,'deployment':'../elsewhere'})
if __name__=='__main__':unittest.main()
