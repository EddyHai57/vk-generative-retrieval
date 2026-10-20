import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from analysis.new_instance_prepare import inventory, check_rebuilt, check_runtime


class MigrationTests(unittest.TestCase):
    def test_exact_inventory_and_original_receipt(self):
        value = inventory()
        self.assertEqual(len(value['files']), 30)
        self.assertEqual(value['files'][-1]['size'], 14300986995)
        ok = dict(filtered_rows=14881823, filtered_users=81790, filtered_items=1183510,
                  union_items=12467294, users_sha256='541ca824327f3a714fb078e26d2f0cdebcdafdd8fc1b19c5e0a507ffa2996fd9')
        check_rebuilt(ok)
        for k in ok:
            bad = copy.deepcopy(ok); bad[k] = 0
            with self.assertRaises(ValueError): check_rebuilt(bad)

    def test_runtime_exact_versions_no_silent_fallback(self):
        good = dict(python='3.11.16', numpy='1.26.4', pyarrow='19.0.1', pyyaml='6.0.2', torch='2.11.0+cu128', cuda='12.8')
        check_runtime(good)
        for k in good:
            bad={**good,k:'different'}
            with self.assertRaises(ValueError): check_runtime(bad)

    def test_cli_dry_and_invalid_and_execution_guard(self):
        base = [sys.executable, '-B', '-m', 'analysis.new_instance_prepare']
        r=subprocess.run(base+['dry'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertFalse(json.loads(r.stdout)['network_executed'])
        for args in (['bad'], ['download-a'], ['cuda-smoke']):
            r=subprocess.run(base+args,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)

    def test_download_a_delegates_original_manifest_loader_without_network(self):
        from analysis import new_instance_prepare as entry
        from analysis import ur001_prepare as old
        manifest=old.load_file_manifest()
        with patch.object(old, 'download_all', return_value={'status':'synthetic_not_downloaded'}) as call:
            self.assertEqual(entry.main(['download-a','--root','/synthetic-new',
                '--reference-root','/synthetic-reference','--execute-on-new-instance']),0)
        self.assertEqual(call.call_args.args[2],manifest)


if __name__ == '__main__':
    unittest.main()
