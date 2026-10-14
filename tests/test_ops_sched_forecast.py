"""ops/tools/server/sched_forecast.py: cost table, first-fit simulation, alert, read-only (synthetic runs/ + queue/)."""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ops_testlib import NOW, PY, ROOT, FixtureHost

TOOL = ROOT / 'ops' / 'tools' / 'server' / 'sched_forecast.py'
P_TRAIN = 'vkgr.retrieval_feat02_run train'
P_VERIFY = 'vkgr.retrieval_feat02_run verify'
G_TRAIN = 'vkgr.generative_b1_arch train'


def load():
    spec = importlib.util.spec_from_file_location('tool_sched_forecast', TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def argv_of(key, *extra):
    module, sub = key.split()
    return [PY, '-B', '-m', module, sub, '--inputs', 'x.json', *extra]


def need(cpu=1, ram=8192, vram=0, disk=1):
    return {'cpu_cores': cpu, 'ram_mib': ram, 'vram_mib': vram, 'disk_gib': disk}


def write_run(root, run_id, key=P_TRAIN, *, status=None, wall=None, created=NOW - 5000, owner=None,
              kind='real', wall_hours=6.0, argv=None, **res):
    run = Path(root) / 'runs' / run_id
    run.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({
        'schema': 'vkgr-run-v1', 'run_id': run_id, 'kind': kind, 'created_at': created,
        'argv': argv or argv_of(key), 'stop_mode': 'none',
        'resources': {**need(**res), 'wall_seconds': wall_hours * 3600.0}}))
    if owner is not None:
        sup, job, started = owner
        (run / 'owner.json').write_text(json.dumps({
            'supervisor': {'pid': sup[0], 'pgid': sup[0], 'sid': sup[0], 'birth': sup[1]},
            'job': {'pid': job[0], 'pgid': sup[0], 'sid': sup[0], 'birth': job[1]},
            'started_at': started, 'birth_source': 'proc-stat-22'}))
    if status is not None:
        (run / 'terminal.json').write_text(json.dumps({
            'schema': 'vkgr-terminal-v1', 'run_id': run_id, 'status': status, 'exit_code': 0, 'signal': None,
            'started_at': created, 'ended_at': created + (wall or 0), 'wall_seconds': wall,
            'wall_cap_exceeded': False}))
    return run


def write_pending(root, name, run_id, key=P_TRAIN, wall_hours=6.0, argv=None, **res):
    path = Path(root) / 'queue' / 'pending' / f'{name}-{run_id}.json'
    path.write_text(json.dumps({'schema': 'vkgr-queue-job-v1', 'run_id': run_id, 'kind': 'real',
                                'argv': argv or argv_of(key), 'wall_hours': wall_hours, **need(**res)}))
    return path


def tree_state(root):
    return sorted((str(p.relative_to(root)), p.stat().st_mtime_ns, p.stat().st_size)
                  for p in Path(root).rglob('*'))


class ForecastCase(unittest.TestCase):
    def setUp(self):
        self.mod = load()
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.root = self.base / 'opsroot'
        for sub in ('runs', 'queue/pending', 'queue/claimed', 'queue/rejected'):
            (self.root / sub).mkdir(parents=True)
        # Server shape: 64 visible cpus but cgroup quota 10.24 cores, A6000 49140 MiB, ~200 GB RAM, 500 GB disk.
        self.host = FixtureHost(self.base / 'host', cpu_count=64, cpu_max='1024000 100000',
                                cgroup_usage_per_s=1.0, ram_total_mib=200000, ram_avail_mib=190000,
                                disk_total_gib=500, disk_free_gib=300, gpu='49140, 0')
        self.host.t = NOW
        self.pid = 6000

    def tearDown(self):
        self._tmp.cleanup()

    def running(self, run_id, key=P_TRAIN, elapsed=60.0, **kw):
        sup, job = self.pid, self.pid + 1
        self.pid += 2
        self.host.add_proc(sup, birth=11)
        self.host.add_proc(job, ppid=sup, birth=12)
        return write_run(self.root, run_id, key, owner=((sup, 11), (job, 12), NOW - elapsed), **kw)

    def history(self, key, walls, **kw):
        for i, wall in enumerate(walls):
            write_run(self.root, f'done-{key.split()[0][5:13]}-{key.split()[1]}-{i}', key,
                      status='succeeded', wall=wall, **kw)

    def forecast(self, **kw):
        return self.mod.forecast(self.root, probes=self.host.probes(), gpu_samples=1, **kw)

    def main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = self.mod.main(['--root', str(self.root), '--gpu-samples', '1', *args], probes=self.host.probes())
        return code, out.getvalue(), err.getvalue()


class TypeKeyTests(ForecastCase):
    def test_module_and_subcommand(self):
        self.assertEqual(self.mod.type_key(argv_of(P_TRAIN)), P_TRAIN)
        self.assertEqual(self.mod.type_key(argv_of(P_VERIFY, '--manifest', 'm.json')), P_VERIFY)
        self.assertEqual(self.mod.type_key([PY, '-B', '-m', 'vkgr.generative_b1_batched', 'train']),
                         'vkgr.generative_b1_batched train')

    def test_script_fallback_and_empty(self):
        self.assertEqual(self.mod.type_key([PY, '-B', 'cache/g3-formal-invoke-v2.py', '--x', '1']),
                         'g3-formal-invoke-v2.py')
        self.assertEqual(self.mod.type_key([PY, '-I', 'ops/bin/vkgr-sched', 'serve', '--root', 'r']),
                         'vkgr-sched serve')
        self.assertIsNone(self.mod.type_key([]))
        self.assertIsNone(self.mod.type_key(None))


class CostTableTests(ForecastCase):
    def test_median_max_from_succeeded_runs_only(self):
        self.history(P_TRAIN, [600.0, 960.0, 720.0])
        self.history(P_VERIFY, [300.0])
        write_run(self.root, 'p-failed', P_TRAIN, status='failed', wall=5.0)
        write_run(self.root, 'p-capped', P_TRAIN, status='wall_cap_exceeded', wall=99999.0)
        write_run(self.root, 'sched-serve', argv=[PY, '-I', 'ops/bin/vkgr-sched', 'serve'], kind='ops',
                  status='succeeded', wall=7.0)
        table = self.mod.cost_table(self.root)
        self.assertEqual(set(table), {P_TRAIN, P_VERIFY})
        self.assertEqual(table[P_TRAIN]['n'], 3)
        self.assertEqual(table[P_TRAIN]['wall_median_s'], 720.0)
        self.assertEqual(table[P_TRAIN]['wall_max_s'], 960.0)
        self.assertEqual(table[P_VERIFY]['wall_median_s'], 300.0)

    def test_even_count_median_and_declared_resources(self):
        self.history(G_TRAIN, [3600.0, 7200.0], vram=6656, ram=20000)
        row = self.mod.cost_table(self.root)[G_TRAIN]
        self.assertEqual(row['wall_median_s'], 5400.0)
        self.assertEqual(row['vram_mib_max'], 6656)
        self.assertEqual(row['ram_mib_max'], 20000)

    def test_new_runs_update_the_table(self):
        self.history(P_TRAIN, [600.0])
        self.assertEqual(self.mod.cost_table(self.root)[P_TRAIN]['wall_max_s'], 600.0)
        write_run(self.root, 'p-new', P_TRAIN, status='succeeded', wall=1200.0)
        self.assertEqual(self.mod.cost_table(self.root)[P_TRAIN]['wall_max_s'], 1200.0)


class CapacityTests(ForecastCase):
    def test_cgroup_quota_and_headroom_caps(self):
        doc = self.forecast()
        cap = doc['capacity']
        self.assertAlmostEqual(cap['total']['cpu_cores'], 10.24)
        self.assertAlmostEqual(cap['admissible']['cpu_cores'], 10.24 * 0.8)
        self.assertEqual(cap['total']['vram_mib'], 49140)
        self.assertAlmostEqual(cap['admissible']['vram_mib'], 49140 * 0.8)
        self.assertAlmostEqual(cap['total']['disk_gib'], 500, places=3)
        self.assertEqual(cap['headroom'], 0.20)


class RemainingTests(ForecastCase):
    def test_remaining_median_then_max_then_overdue(self):
        self.history(P_TRAIN, [600.0, 720.0, 960.0])
        self.running('p-a', elapsed=300.0)
        self.running('p-b', elapsed=800.0)
        self.running('p-c', elapsed=2000.0)
        rows = {r['run_id']: r for r in self.forecast()['running']}
        self.assertAlmostEqual(rows['p-a']['remaining_s'], 420.0)
        self.assertEqual(rows['p-a']['source'], 'median')
        self.assertAlmostEqual(rows['p-b']['remaining_s'], 160.0)
        self.assertEqual(rows['p-b']['source'], 'max')
        self.assertEqual(rows['p-c']['remaining_s'], 0.0)
        self.assertEqual(rows['p-c']['source'], 'overdue')

    def test_unknown_type_uses_wall_cap_and_ops_is_permanent(self):
        self.running('mystery', key='vkgr.newthing train', elapsed=600.0, wall_hours=2.0)
        self.running('sched-serve', argv=[PY, '-I', 'ops/bin/vkgr-sched', 'serve'], kind='ops', cpu=0, ram=200)
        rows = {r['run_id']: r for r in self.forecast()['running']}
        self.assertAlmostEqual(rows['mystery']['remaining_s'], 7200.0 - 600.0)
        self.assertEqual(rows['mystery']['source'], 'wall_cap')
        self.assertIsNone(rows['sched-serve']['remaining_s'])
        self.assertEqual(rows['sched-serve']['source'], 'service')


class SimulationTests(ForecastCase):
    def test_first_fit_one_launch_per_tick_and_wait_for_free_slot(self):
        self.history(P_TRAIN, [600.0])
        for i in range(7):
            self.running(f'r{i}', elapsed=0.0 if i else 300.0)     # r0 ends at +300, r1..r6 at +600
        for i, rid in enumerate(('q1', 'q2', 'q3')):
            write_pending(self.root, f'{i:04d}', rid)
        doc = self.forecast()
        rows = {r['run_id']: r for r in doc['pending']}
        # cap 8.192 cores: 7 running + 1 fits now; q2 waits for r0 (+300); q3 for r1..r6 (+600), 1 per 30 s tick.
        self.assertEqual(rows['q1']['start_s'], 0.0)
        self.assertEqual(rows['q2']['start_s'], 300.0)
        self.assertEqual(rows['q3']['start_s'], 600.0)
        self.assertEqual(rows['q3']['end_s'], 1200.0)
        self.assertEqual([r['run_id'] for r in doc['pending']], ['q1', 'q2', 'q3'])

    def test_two_fit_at_once_start_one_tick_apart(self):
        self.history(P_TRAIN, [600.0])
        write_pending(self.root, '0001', 'q1')
        write_pending(self.root, '0002', 'q2')
        rows = {r['run_id']: r for r in self.forecast(tick_seconds=30.0)['pending']}
        self.assertEqual((rows['q1']['start_s'], rows['q2']['start_s']), (0.0, 30.0))

    def test_first_fit_skips_head_that_does_not_fit(self):
        self.history(P_TRAIN, [600.0])
        self.history(G_TRAIN, [3600.0])
        self.running('g-big', key=G_TRAIN, elapsed=0.0, vram=30000)
        write_pending(self.root, '0001', 'g-next', G_TRAIN, vram=20000)   # 50000 > 39312 admissible
        write_pending(self.root, '0002', 'p-small', vram=3072)
        rows = {r['run_id']: r for r in self.forecast()['pending']}
        self.assertEqual(rows['p-small']['start_s'], 0.0)
        self.assertEqual(rows['g-next']['start_s'], 3600.0)

    def test_unsatisfiable_job_never_starts(self):
        self.history(G_TRAIN, [3600.0])
        write_pending(self.root, '0001', 'g-huge', G_TRAIN, vram=45000)
        row = self.forecast()['pending'][0]
        self.assertIsNone(row['start_s'])
        self.assertEqual(row['state'], 'never_fits')

    def test_idle_and_cpu_below_half(self):
        self.history(P_TRAIN, [600.0])
        for i in range(6):
            self.running(f'r{i}', elapsed=0.0)          # 6 cores busy until +600
        write_pending(self.root, '0001', 'q1')          # +0 .. +600
        doc = self.forecast()
        s = doc['summary']
        self.assertEqual(s['idle_at_s'], 600.0)
        self.assertEqual(s['cpu_below_half_at_s'], 600.0)  # 7 cores >= 4.096 until everything ends
        self.assertAlmostEqual(s['backlog_job_hours'], 7 * 600 / 3600)


class AlertTests(ForecastCase):
    def test_backlog_low_on_first_line(self):
        self.history(P_TRAIN, [600.0])
        self.running('r0', elapsed=0.0)
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        self.assertTrue(out.splitlines()[0].startswith('BACKLOG_LOW'), out)

    def test_threshold_is_a_parameter(self):
        self.history(G_TRAIN, [4 * 3600.0])
        for i in range(2):
            write_pending(self.root, f'{i:04d}', f'g{i}', G_TRAIN, vram=6656)
        _, out, _ = self.main()
        self.assertFalse(out.splitlines()[0].startswith('BACKLOG_LOW'), out)
        _, out, _ = self.main('--backlog-hours', '10')
        self.assertTrue(out.splitlines()[0].startswith('BACKLOG_LOW'), out)

    def test_cap_estimates_do_not_hide_the_alert(self):
        write_pending(self.root, '0001', 'mystery', 'vkgr.newthing train', wall_hours=24.0)
        doc = self.forecast()
        self.assertTrue(doc['summary']['backlog_low'])
        self.assertAlmostEqual(doc['summary']['backlog_job_hours_wall_cap'], 24.0)


class OutputTests(ForecastCase):
    def test_json_schema_and_no_pids_or_paths(self):
        self.history(P_TRAIN, [600.0])
        self.running('r0', elapsed=10.0)
        write_pending(self.root, '0001', 'q1')
        code, out, _ = self.main('--json')
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc['schema'], 'vkgr-sched-forecast-v1')
        for key in ('capacity', 'occupancy', 'cost_table', 'running', 'pending', 'summary'):
            self.assertIn(key, doc)
        _, text, _ = self.main()
        for blob in (out, text):
            self.assertNotIn(str(self.base), blob)
            self.assertNotIn('6000', blob)       # supervisor pid
            self.assertNotIn('6001', blob)       # job pid

    def test_gpu_utilization_samples(self):
        fake = self.base / 'nvsmi.py'
        fake.write_text('import sys\na = " ".join(sys.argv[1:])\n'
                        'if "utilization.gpu" in a: print("37")\n'
                        'elif "query-gpu" in a: print("49140, 5000")\n'
                        'elif "compute-apps" in a: print("6001, 4096")\n'
                        'else: sys.exit(2)\n')
        self.host.nvidia = [PY, str(fake)]
        self.running('g0', key=G_TRAIN, vram=6656)
        doc = self.mod.forecast(self.root, probes=self.host.probes(), gpu_samples=3)
        occ = doc['occupancy']
        self.assertEqual(occ['gpu_util_mean'], 37.0)
        self.assertEqual(occ['gpu_util_max'], 37.0)
        self.assertEqual(occ['gpu_util_samples'], 3)
        self.assertEqual(occ['declared']['vram_mib'], 6656)
        self.assertEqual(occ['gpu_used_mib'], 5000)
        self.assertEqual(doc['running'][0]['observed_vram_mib'], 4096)

    def test_read_only(self):
        self.history(P_TRAIN, [600.0])
        self.running('r0', elapsed=10.0)
        write_pending(self.root, '0001', 'q1')
        before = tree_state(self.root)
        self.main()
        self.main('--json')
        self.assertEqual(tree_state(self.root), before)

    def test_missing_root_is_an_error(self):
        out, err = io.StringIO(), io.StringIO()
        env = dict(os.environ)
        env.pop('VKGR_OPS_ROOT', None)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.dict('os.environ', env, clear=True):
            code = self.mod.main([], probes=self.host.probes())
        self.assertEqual(code, 1)
        self.assertIn('ops root required', err.getvalue())


if __name__ == '__main__':
    unittest.main()
