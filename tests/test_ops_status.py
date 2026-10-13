import builtins
import fcntl
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ops_testlib import NOW, OPS, PY, FixtureHost, make_run

from vkgr_ops import scheduler, status


def progress(run_id, **kw):
    doc = {'schema': 'vkgr-progress-v1', 'run_id': run_id, 'line': 'G', 'stage': 'train', 'phase': 'formal',
           'epoch': 3, 'max_epochs': 30, 'completed': 640, 'total': 2520, 'message': None,
           'updated_at': NOW - 12.5, 'pid': 5001}
    doc.update(kw)
    return doc


class StatusCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.root = self.base / 'opsroot'
        (self.root / 'runs').mkdir(parents=True)
        (self.root / 'queue' / 'pending').mkdir(parents=True)
        self.host = FixtureHost(self.base / 'host', cpu_count=64, cpu_max='1024000 100000',
                                cgroup_usage_per_s=2.56, nr_throttled=35209, throttled_usec=17016000000,
                                gpu='49140, 9140', apps='5001, 3000;7777, 999')
        self.host.add_proc(7777)

    def tearDown(self):
        self._tmp.cleanup()

    def running(self, run_id='g-core-02', sup=5000, job=5001, **kw):
        self.host.add_proc(sup, birth=111, ticks=0, tick_rate=10)
        self.host.add_proc(job, ppid=sup, birth=112, ticks=0, tick_rate=40)
        return make_run(self.root, run_id, owner={'sup': (sup, 111), 'job': (job, 112),
                                                   'started_at': NOW - 100}, **kw)

    def collect(self):
        return status.collect(self.root, probes=self.host.probes())

    def run_row(self, doc, run_id):
        return next(r for r in doc['runs'] if r['run_id'] == run_id)


class JsonSchemaTests(StatusCase):
    def test_json_schema_and_columns(self):
        run = self.running(wall_seconds=7200.0)
        (run / 'progress.json').write_text(json.dumps(progress('g-core-02')))
        (self.root / 'queue' / 'pending' / '0001-next-1.json').write_text(
            json.dumps({'schema': 'vkgr-queue-job-v1', 'run_id': 'next-1'}))
        doc = self.collect()
        self.assertEqual(doc['schema'], 'vkgr-status-v1')
        host = doc['host']
        self.assertEqual(set(host), {'gpu_free_mib', 'gpu_total_mib', 'ram_free_mib', 'ram_total_mib',
                                     'cpu_idle_fraction', 'cpu_total', 'cpu_used_cores', 'cpu_nr_throttled',
                                     'cpu_throttled_seconds', 'disk_free_gib', 'disk_total_gib'})
        self.assertEqual((host['gpu_free_mib'], host['gpu_total_mib']), (40000, 49140))
        self.assertEqual((host['ram_free_mib'], host['ram_total_mib']), (60000, 100000))
        self.assertAlmostEqual(host['cpu_total'], 10.24)
        self.assertAlmostEqual(host['cpu_used_cores'], 2.56)
        self.assertAlmostEqual(host['cpu_idle_fraction'], 0.75)
        self.assertEqual(host['cpu_nr_throttled'], 35209)
        self.assertAlmostEqual(host['cpu_throttled_seconds'], 17016.0)
        self.assertEqual((host['disk_free_gib'], host['disk_total_gib']), (500.0, 1000.0))
        row = self.run_row(doc, 'g-core-02')
        self.assertEqual(set(row), {'run_id', 'status', 'stage', 'completed', 'total', 'progress_age_seconds',
                                    'wall_used_seconds', 'wall_cap_seconds', 'owner_pid', 'cpu_percent',
                                    'vram_mib'})
        self.assertEqual(row['status'], 'running')
        self.assertEqual((row['stage'], row['completed'], row['total']), ('train', 640, 2520))
        self.assertAlmostEqual(row['progress_age_seconds'], 12.5, delta=2.0)
        self.assertAlmostEqual(row['wall_used_seconds'], 100.0, delta=2.0)
        self.assertEqual(row['wall_cap_seconds'], 7200.0)
        self.assertEqual(row['owner_pid'], 5000)
        clk = os.sysconf('SC_CLK_TCK')
        self.assertAlmostEqual(row['cpu_percent'], 50 / clk / 0.5 * 100)
        self.assertEqual(row['vram_mib'], 3000)
        self.assertEqual(doc['pending'], [{'file': '0001-next-1.json', 'run_id': 'next-1'}])

    def test_table_output_has_quota_and_throttle(self):
        self.running()
        text = status.render_table(self.collect())
        self.assertIn('g-core-02', text)
        self.assertIn('running', text)
        self.assertIn('10.24', text)
        self.assertIn('throttled', text)
        self.assertIn('35209', text)

    def test_unavailable_values_are_null(self):
        host = FixtureHost(self.base / 'host2', nvidia=False)
        (host.proc / 'meminfo').unlink()
        doc = status.collect(self.root, probes=host.probes())
        self.assertIsNone(doc['host']['gpu_free_mib'])
        self.assertIsNone(doc['host']['ram_total_mib'])
        self.assertEqual(doc['host']['cpu_nr_throttled'], 0)


class StatusEnumTests(StatusCase):
    def test_birth_mismatch_is_lost(self):
        self.host.add_proc(6000, birth=999)
        make_run(self.root, 'reused', owner={'sup': (6000, 111), 'job': (6001, 112), 'started_at': NOW - 50})
        self.assertEqual(self.run_row(self.collect(), 'reused')['status'], 'lost')

    def test_no_owner_young_is_starting_old_is_lost(self):
        make_run(self.root, 'young', created_at=NOW - 30)
        make_run(self.root, 'old', created_at=NOW - 90)
        doc = self.collect()
        self.assertEqual(self.run_row(doc, 'young')['status'], 'starting')
        self.assertEqual(self.run_row(doc, 'old')['status'], 'lost')

    def test_orphaned_and_terminal(self):
        self.host.add_proc(6101, birth=112)
        make_run(self.root, 'orphan', owner={'sup': (6100, 111), 'job': (6101, 112), 'started_at': NOW - 50})
        make_run(self.root, 'done', terminal='succeeded',
                 owner={'sup': (6200, 1), 'job': (6201, 2), 'started_at': NOW - 50})
        doc = self.collect()
        self.assertEqual(self.run_row(doc, 'orphan')['status'], 'orphaned')
        done = self.run_row(doc, 'done')
        self.assertEqual(done['status'], 'succeeded')
        self.assertEqual(done['wall_used_seconds'], 5.0)


class CorruptFileTests(StatusCase):
    def test_corrupt_files_give_null_fields(self):
        run = self.running('bad-1')
        (run / 'progress.json').write_text('{not json')
        doc = self.collect()
        row = self.run_row(doc, 'bad-1')
        self.assertEqual(row['status'], 'running')
        self.assertIsNone(row['stage'])
        self.assertIsNone(row['completed'])
        broken = self.root / 'runs' / 'broken'
        broken.mkdir()
        (broken / 'run.json').write_text('garbage')
        row = self.run_row(self.collect(), 'broken')
        self.assertEqual(row['status'], 'lost')
        self.assertIsNone(row['wall_cap_seconds'])

    def test_progress_validation_n4(self):
        run = self.running('prog-1', sup=5100, job=5101)
        cases = {
            'wrong schema': progress('prog-1', schema='vkgr-progress-v0'),
            'wrong run_id': progress('someone-else'),
            'too large': progress('prog-1', message='x' * 5000),
        }
        for label, doc in cases.items():
            with self.subTest(label):
                (run / 'progress.json').write_text(json.dumps(doc))
                row = self.run_row(self.collect(), 'prog-1')
                self.assertIsNone(row['stage'])
                self.assertIsNone(row['progress_age_seconds'])
        (run / 'progress.json').write_text(json.dumps(progress('prog-1')))
        self.assertEqual(self.run_row(self.collect(), 'prog-1')['stage'], 'train')

    def test_oversized_run_json_is_null(self):
        run = make_run(self.root, 'huge', created_at=NOW - 10)
        (run / 'run.json').write_text(json.dumps({'pad': 'x' * 70000}))
        row = self.run_row(self.collect(), 'huge')
        self.assertIsNone(row['wall_cap_seconds'])


class ReadOnlyTests(StatusCase):
    def test_status_is_read_only(self):
        run = self.running('ro-1')
        (run / 'progress.json').write_text(json.dumps(progress('ro-1')))
        make_run(self.root, 'ro-2', terminal='failed', owner={'sup': (1, 1), 'job': (2, 2), 'started_at': NOW})
        (self.root / 'queue' / 'pending' / '0002-ro-3.json').write_text('{"run_id": "ro-3"}')
        self.host.frozen = True       # fixture must not rewrite files during the call
        before = {p: p.stat().st_mtime_ns for p in self.root.rglob('*')}
        real_open, real_popen = builtins.open, subprocess.Popen
        allowed = self.host.nvidia

        def guarded_open(file, mode='r', *a, **kw):
            if mode not in ('r', 'rb'):
                raise AssertionError(f'write open {file} {mode}')
            return real_open(file, mode, *a, **kw)

        def guarded_popen(args, *a, **kw):
            if list(args[:len(allowed)]) != allowed:
                raise AssertionError(f'subprocess {args}')
            return real_popen(args, *a, **kw)

        def forbid(name):
            def _f(*a, **kw):
                raise AssertionError(f'{name} called')
            return _f

        with patch.object(builtins, 'open', guarded_open), patch.object(io, 'open', guarded_open), \
             patch.object(subprocess, 'Popen', guarded_popen), \
             patch.object(os, 'replace', forbid('os.replace')), patch.object(os, 'kill', forbid('os.kill')), \
             patch.object(os, 'killpg', forbid('os.killpg')), patch.object(os, 'rename', forbid('os.rename')), \
             patch.object(fcntl, 'flock', forbid('flock')):
            doc = self.collect()
            text = status.render_table(doc)
        self.assertEqual(self.run_row(doc, 'ro-1')['status'], 'running')
        self.assertIn('ro-1', text)
        after = {p: p.stat().st_mtime_ns for p in self.root.rglob('*')}
        self.assertEqual(before, after)

    def test_cli_json_exit_zero(self):
        self.running('cli-1')
        out = subprocess.run([PY, '-I', str(OPS / 'bin' / 'vkgr-status'), '--root', str(self.root), '--json'],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        doc = json.loads(out.stdout)
        self.assertEqual(doc['schema'], 'vkgr-status-v1')
        self.assertIn('cli-1', [r['run_id'] for r in doc['runs']])

    def test_pending_listed_via_scheduler_submit(self):
        repo = self.base / 'repo'
        from ops_testlib import base_spec, make_repo
        make_repo(repo)
        scheduler.submit(self.root, base_spec(repo, run_id='queued-1'))
        doc = self.collect()
        self.assertEqual([p['run_id'] for p in doc['pending']], ['queued-1'])


if __name__ == '__main__':
    unittest.main()
