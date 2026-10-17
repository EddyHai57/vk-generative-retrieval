"""Synthetic tests for the G5 p3c data-channel step (hashes patched to fixture bytes)."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('dc', Path(__file__).with_name('g5_p3c_data_channel_20261018.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
H = lambda b: hashlib.sha256(b).hexdigest()


class DataChannelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        self.dep, self.ops, self.src = root / 'dep', root / 'ops', root / 'src.md'
        (self.dep / 'docs').mkdir(parents=True); self.ops.mkdir(); (self.ops / 'deployment.json').write_text('{}')
        (self.dep / 'docs/p3c_validation.md').write_bytes(b'git'); self.src.write_bytes(b'pinned')
        m.PINNED, m.GIT_VERSION = H(b'pinned'), H(b'git')

    def tearDown(self):
        self.tmp.cleanup()

    def test_applies_and_keeps_original(self):
        got = m.apply(self.dep, self.ops, self.src)
        self.assertEqual((self.dep / 'docs/p3c_validation.md').read_bytes(), b'pinned'); self.assertEqual((self.ops / 'p3c-git-original.md').read_bytes(), b'git')
        self.assertEqual(got['status'], 'data_channel_applied')

    def test_refuses_wrong_source_second_run_and_missing_ops(self):
        self.src.write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'source hash'):
            m.apply(self.dep, self.ops, self.src)
        self.src.write_bytes(b'pinned'); m.apply(self.dep, self.ops, self.src)
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            m.apply(self.dep, self.ops, self.src)

    def test_refuses_without_deployment_record(self):
        (self.ops / 'deployment.json').unlink()
        with self.assertRaisesRegex(ValueError, 'ops directory'):
            m.apply(self.dep, self.ops, self.src)


if __name__ == '__main__':
    unittest.main()
