import json
import tempfile
import unittest
from pathlib import Path

from ops_testlib import NOW, FixtureHost, base_spec, make_repo, make_run, read_json

from vkgr_ops import admission, common, launcher


def need(**kw):
    values = {'cpu_cores': 0, 'ram_mib': 0, 'vram_mib': 0, 'disk_gib': 0}
    values.update(kw)
    return values


class AdmissionTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.root = self.base / 'opsroot'
        (self.root / 'runs').mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def host(self, **kw):
        return FixtureHost(self.base / f'host{len(list(self.base.glob("host*")))}', **kw)

    def admit(self, host, **kw):
        ok, snapshot, reason = admission.admit(need(**kw), self.root, host.probes())
        return ok, snapshot, reason


class BoundaryTests(AdmissionTestCase):
    def test_ram_boundary(self):
        host = self.host(ram_total_mib=100000, ram_avail_mib=60000)
        self.assertTrue(self.admit(host, ram_mib=40000)[0])
        ok, _, reason = self.admit(host, ram_mib=40001)
        self.assertFalse(ok)
        self.assertIn('ram_mib', reason)

    def test_vram_boundary(self):
        host = self.host(gpu='49140, 9140')
        self.assertTrue(self.admit(host, vram_mib=30172)[0])
        ok, _, reason = self.admit(host, vram_mib=30173)
        self.assertFalse(ok)
        self.assertIn('vram_mib', reason)

    def test_disk_boundary(self):
        host = self.host(disk_total_gib=1000, disk_free_gib=500)
        self.assertTrue(self.admit(host, disk_gib=300)[0])
        self.assertFalse(self.admit(host, disk_gib=301)[0])

    def test_cpu_boundary_proc_stat_branch(self):
        host = self.host(cpu_count=16, idle_fraction=0.8)   # free 12.8 cores
        self.assertTrue(self.admit(host, cpu_cores=9)[0])
        ok, _, reason = self.admit(host, cpu_cores=10)
        self.assertFalse(ok)
        self.assertIn('cpu_cores', reason)


class CpuQuotaTests(AdmissionTestCase):
    def test_cgroup_quota_branch_uses_cpu_stat_delta(self):
        host = self.host(cpu_count=64, cpu_max='200000 100000', cgroup_usage_per_s=0.5)
        free, total = common.cpu_free(host.probes())
        self.assertAlmostEqual(total, 2.0)
        self.assertAlmostEqual(free, 1.5)

    def test_unlimited_branch_uses_proc_stat(self):
        host = self.host(cpu_count=16, idle_fraction=0.75, cpu_max='max 100000')
        free, total = common.cpu_free(host.probes())
        self.assertEqual(total, 16)
        self.assertAlmostEqual(free, 12.0)

    def test_server_quota_10_24_cores_is_total_not_cpu_count(self):
        # orchestrator fact 2026-10-14: cpu.max "1024000 100000" while cpu_count shows 64
        host = self.host(cpu_count=64, cpu_max='1024000 100000', cgroup_usage_per_s=1.0)
        free, total = common.cpu_free(host.probes())
        self.assertAlmostEqual(total, 10.24)
        self.assertAlmostEqual(free, 9.24)
        self.assertTrue(self.admit(host, cpu_cores=7)[0])
        self.assertFalse(self.admit(host, cpu_cores=8)[0])

    def test_declared_cpu_above_quota_is_rejected_explicitly(self):
        host = self.host(cpu_count=64, cpu_max='1024000 100000', cgroup_usage_per_s=0.0)
        ok, _, reason = self.admit(host, cpu_cores=11)
        self.assertFalse(ok)
        self.assertIn('quota', reason)

    def test_both_cpu_probes_failing_is_probe_failure(self):
        host = self.host()
        (host.proc / 'stat').unlink()
        (host.proc / 'self' / 'cgroup').unlink()
        with self.assertRaises(common.ProbeError):
            common.cpu_free(host.probes())
        ok, _, reason = self.admit(host)
        self.assertFalse(ok)
        self.assertIn('probe', reason)


class OccupancyTests(AdmissionTestCase):
    def occupied(self, host, run_id, *, sup=5000, job=5001, rss_kib=1024, started_at=NOW - 1000, **res):
        host.add_proc(sup, birth=111)
        host.add_proc(job, ppid=sup, birth=112, rss_kib=rss_kib)
        return make_run(self.root, run_id, created_at=started_at, **res,
                        owner={'sup': (sup, 111), 'job': (job, 112), 'started_at': started_at})

    def test_declared_greater_than_measured_rejects_by_declared(self):
        host = self.host(ram_total_mib=100000, ram_avail_mib=90000)
        self.occupied(host, 'big-declared', ram=50000, rss_kib=1000 * 1024)
        ok, snapshot, reason = self.admit(host, ram_mib=40000)
        self.assertFalse(ok)
        self.assertEqual(snapshot['resources']['ram_mib']['committed'], 50000)
        self.assertIn('ram_mib', reason)

    def test_observed_greater_than_declared_uses_observed(self):
        host = self.host(ram_total_mib=100000, ram_avail_mib=90000)
        self.occupied(host, 'big-rss', ram=1000, rss_kib=50000 * 1024)
        ok, snapshot, _ = self.admit(host, ram_mib=40000)
        self.assertFalse(ok)
        # observed = job 50000 MiB + supervisor 1 MiB (fixture default VmRSS 1024 kB)
        self.assertEqual(snapshot['resources']['ram_mib']['committed'], 50001)

    def test_observed_vram_from_compute_apps_of_descendants(self):
        host = self.host(gpu='49140, 9140', apps='5001, 20000;9999, 7000')
        host.add_proc(9999)
        self.occupied(host, 'gpu-job', vram=1000)
        _, snapshot, _ = self.admit(host)
        self.assertEqual(snapshot['resources']['vram_mib']['committed'], 20000)

    def test_settling_counts_declared_until_settle_seconds(self):
        host = self.host(ram_total_mib=100000, ram_avail_mib=60000)
        self.occupied(host, 'fresh', ram=30000, started_at=NOW - 100)
        self.assertFalse(self.admit(host, ram_mib=20000)[0])
        host2 = self.host(ram_total_mib=100000, ram_avail_mib=60000)
        run = self.root / 'runs' / 'fresh'
        owner = read_json(run / 'owner.json')
        owner['started_at'] = NOW - 700
        (run / 'owner.json').write_text(json.dumps(owner))
        host2.add_proc(5000, birth=111)
        host2.add_proc(5001, ppid=5000, birth=112)
        self.assertTrue(self.admit(host2, ram_mib=20000)[0])

    def test_settle_starts_at_owner_started_at_not_created_at(self):
        # N5: created long ago, but owner.started_at recent -> still settling
        host = self.host(ram_total_mib=100000, ram_avail_mib=60000)
        host.add_proc(5000, birth=111)
        host.add_proc(5001, ppid=5000, birth=112)
        make_run(self.root, 'slow-start', ram=30000, created_at=NOW - 5000,
                 owner={'sup': (5000, 111), 'job': (5001, 112), 'started_at': NOW - 100})
        _, snapshot, _ = self.admit(host)
        self.assertEqual(snapshot['resources']['ram_mib']['settling'], 30000)

    def test_starting_run_without_owner_settles_from_created_at(self):
        host = self.host(ram_total_mib=100000, ram_avail_mib=60000)
        make_run(self.root, 'starting', ram=30000, created_at=NOW - 10)
        _, snapshot, _ = self.admit(host)
        self.assertEqual(snapshot['resources']['ram_mib']['committed'], 30000)
        self.assertEqual(snapshot['resources']['ram_mib']['settling'], 30000)

    def test_terminal_run_is_not_counted_even_if_recent(self):
        host = self.host()
        make_run(self.root, 'done', ram=50000, created_at=NOW - 20, terminal='succeeded',
                 owner={'sup': (5000, 111), 'job': (5001, 112), 'started_at': NOW - 20})
        _, snapshot, _ = self.admit(host)
        self.assertEqual(snapshot['resources']['ram_mib']['committed'], 0)
        self.assertEqual(snapshot['resources']['ram_mib']['settling'], 0)

    def test_orphaned_run_is_counted(self):
        host = self.host()
        host.add_proc(5001, birth=112)       # job alive, supervisor 5000 dead
        make_run(self.root, 'orphan', ram=12345,
                 owner={'sup': (5000, 111), 'job': (5001, 112), 'started_at': NOW - 1000})
        _, snapshot, _ = self.admit(host)
        self.assertEqual(snapshot['resources']['ram_mib']['committed'], 12345)
        self.assertEqual(common.run_status(self.root / 'runs' / 'orphan', host.probes())[0], 'orphaned')

    def test_lost_run_is_not_counted(self):
        host = self.host()
        host.add_proc(5000, birth=999)       # pid reused: birth mismatch
        make_run(self.root, 'lost', ram=12345,
                 owner={'sup': (5000, 111), 'job': (5001, 112), 'started_at': NOW - 1000})
        _, snapshot, _ = self.admit(host)
        self.assertEqual(snapshot['resources']['ram_mib']['committed'], 0)


class GpuUnknownTests(AdmissionTestCase):
    def test_gpu_unknown_only_admits_zero_vram(self):
        host = self.host(nvidia=False)
        ok, snapshot, _ = self.admit(host, vram_mib=0)
        self.assertTrue(ok)
        self.assertFalse(snapshot['gpu_known'])
        ok, _, reason = self.admit(host, vram_mib=1)
        self.assertFalse(ok)
        self.assertIn('vram_mib', reason)


class LauncherAdmissionTests(AdmissionTestCase):
    def test_launcher_direct_call_is_rejected_by_admission(self):
        repo = self.base / 'repo'
        make_repo(repo)
        host = self.host(ram_total_mib=100000, ram_avail_mib=30000)
        spec = base_spec(repo, run_id='too-big', ram_mib=20000, no_admission=False)
        with self.assertRaises(launcher.LaunchError) as raised:
            launcher.launch(spec, self.root, probes=host.probes())
        self.assertEqual(raised.exception.code, 1)
        self.assertIn('ram_mib', str(raised.exception))
        self.assertFalse((self.root / 'runs' / 'too-big').exists())

    def test_launcher_admitted_records_snapshot(self):
        repo = self.base / 'repo'
        make_repo(repo)
        host = self.host()
        spec = base_spec(repo, run_id='fits', ram_mib=100, no_admission=False)
        run_dir = launcher.launch(spec, self.root, probes=host.probes())
        record = read_json(run_dir / 'run.json')['admission']
        self.assertTrue(record['checked'])
        self.assertIn('ram_mib', record['snapshot']['resources'])

    def test_no_admission_skips_and_records_unchecked(self):
        repo = self.base / 'repo'
        make_repo(repo)
        host = self.host(ram_total_mib=100000, ram_avail_mib=1)
        spec = base_spec(repo, run_id='skip', ram_mib=99999, no_admission=True)
        run_dir = launcher.launch(spec, self.root, probes=host.probes())
        self.assertEqual(read_json(run_dir / 'run.json')['admission'], {'checked': False})


if __name__ == '__main__':
    unittest.main()
