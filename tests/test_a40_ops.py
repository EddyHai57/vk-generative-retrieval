import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


class A40OpsTests(unittest.TestCase):
    def test_only_authorized_data_jobs_can_be_prepared(self):
        from analysis.a40_ops import validate_spec
        spec = dict(kind='ops', no_admission=False, run_id='a40-rebuild-prepare-users-v1',
                    argv=[sys.executable, '-B', '-m', 'analysis.new_instance_prepare',
                          'prepare-users', '--request', '/fixture/request.json', '--execute-on-new-instance'])
        validate_spec(spec)
        for change in [dict(kind='real'), dict(no_admission=True), dict(run_id='training-v1'),
                       dict(argv=[sys.executable, '-m', 'vkgr.train'])]:
            with self.assertRaises(ValueError): validate_spec({**spec, **change})

    def test_planning_cap_retains_used_memory(self):
        from analysis.a40_ops import cap_ram, PLAN_MIB
        self.assertEqual(PLAN_MIB, 489234)
        self.assertEqual(cap_ram(900000, 800000), (489234, 389234))
        self.assertEqual(cap_ram(900000, 100000), (489234, 0))
        self.assertEqual(cap_ram(100000, 80000), (100000, 80000))

    def test_cap_cannot_increase_free(self):
        from analysis.a40_ops import cap_ram
        for total, free in [(900000, 899000), (200000, 0), (900000, 500000)]:
            t, f = cap_ram(total, free)
            self.assertLessEqual(f, free)
            self.assertLessEqual(t, total)
            self.assertLessEqual(f, t)

    def test_real_ops_probe_injection(self):
        from analysis.a40_ops import install_cap
        class Common:
            @staticmethod
            def ram_info(probes): return 900000, 800000
        install_cap(Common)
        self.assertEqual(Common.ram_info(None), (489234, 389234))

    def test_cli_invalid_and_dry_receipt(self):
        command = [sys.executable, '-B', '-m', 'analysis.a40_ops']
        bad = subprocess.run(command+['bad'], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        good = subprocess.run(command+['dry'], capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(json.loads(good.stdout)['planning_ram_bytes'], 513000000000)


if __name__ == '__main__':
    unittest.main()
