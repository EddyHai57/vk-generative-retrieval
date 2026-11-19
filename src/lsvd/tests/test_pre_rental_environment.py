import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from lsvd.check_environment import check_environment


class EnvironmentTests(unittest.TestCase):
    def test_accepts_python_312_patch_versions_only(self):
        with patch('lsvd.check_environment.platform.python_version', return_value='3.12.3'):
            self.assertTrue(check_environment(allow_cpu=True)['versions_match'])
        with patch('lsvd.check_environment.platform.python_version', return_value='3.13.0'):
            with self.assertRaisesRegex(ValueError, 'Python 3.12'):
                check_environment(allow_cpu=True)

    def test_local_versions_are_checked_and_cuda_absence_is_explicit(self):
        result = check_environment(allow_cpu=True)
        self.assertTrue(result['versions_match'])
        self.assertIn('gpu_check', result)
        if not result['cuda_available']:
            self.assertEqual(result['gpu_check'], 'skipped_cpu')

    def test_environment_cli_real_entry_and_invalid_option(self):
        process = subprocess.run([sys.executable, '-m', 'lsvd.check_environment',
                                   '--allow-cpu'], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(json.loads(process.stdout)['versions_match'])
        invalid = subprocess.run([sys.executable, '-m', 'lsvd.check_environment',
                                   '--invalid'], capture_output=True)
        self.assertNotEqual(invalid.returncode, 0)


if __name__ == '__main__':
    unittest.main()
