import fcntl
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ops_testlib import (NOW, PY, FixtureHost, base_spec, host_proc_snapshot, make_repo, make_run,
                         ppid_of, read_json, sha, wait_terminal)

from vkgr_ops import admission, common, launcher, scheduler


class SchedulerCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.root = self.base / 'opsroot'
        self.repo = self.base / 'repo'
        make_repo(self.repo)
        self.host = FixtureHost(self.base / 'host')

    def tearDown(self):
        self._tmp.cleanup()

    def submit(self, run_id, **kw):
        kw.setdefault('no_admission', False)
        return scheduler.submit(self.root, base_spec(self.repo, run_id=run_id, **kw))

    def tick(self):
        return scheduler.tick(self.root, probes=self.host.probes())

    def names(self, sub):
        return sorted(p.name for p in (self.root / 'queue' / sub).glob('*.json'))


class SubmitTests(SchedulerCase):
    def test_submit_writes_pending_file(self):
        path = self.submit('a-1')
        self.assertEqual(path.parent, self.root / 'queue' / 'pending')
        self.assertTrue(path.name.endswith('-a-1.json'))
        doc = read_json(path)
        self.assertEqual(doc['schema'], 'vkgr-queue-job-v1')
        self.assertEqual(doc['run_id'], 'a-1')

    def test_submit_rejects_invalid_spec_early(self):
        with self.assertRaises(launcher.LaunchError):
            self.submit('Bad ID')
        self.assertEqual(self.names('pending'), [])

    def test_submit_does_not_record_hashes(self):
        data = self.base / 'in.bin'
        data.write_text('v1')
        path = self.submit('a-1', inputs=[str(data)])
        self.assertNotIn(sha('v1'), path.read_text())


class TickTests(SchedulerCase):
    def test_one_launch_per_tick_first_fit_order(self):
        self.submit('too-big', ram_mib=10 ** 9)
        self.submit('b-1')
        self.submit('c-1')
        first = self.tick()
        self.assertEqual(first['launched'], 'b-1')
        self.assertTrue((self.root / 'runs' / 'b-1').exists())
        self.assertFalse((self.root / 'runs' / 'c-1').exists())
        second = self.tick()
        self.assertEqual(second['launched'], 'c-1')
        self.assertEqual(len(self.names('pending')), 1)          # too-big stays pending
        for rid in ('b-1', 'c-1'):
            wait_terminal(self.root / 'runs' / rid)

    def test_same_job_not_started_twice(self):
        self.submit('dup-1')
        self.tick()
        wait_terminal(self.root / 'runs' / 'dup-1')
        self.submit('dup-1')
        result = self.tick()
        self.assertIsNone(result['launched'])
        self.assertEqual(len(self.names('rejected')), 1)
        reason = next((self.root / 'queue' / 'rejected').glob('*.reason')).read_text()
        self.assertIn('exists', reason)

    def test_formal_admit_rejection_goes_to_rejected_not_pending(self):
        self.submit('raced-1')
        with patch.object(admission, 'admit', return_value=(False, {}, 'ram_mib: not enough (race)')):
            result = self.tick()
        self.assertIsNone(result['launched'])
        self.assertEqual(self.names('pending'), [])
        self.assertEqual(len(self.names('rejected')), 1)
        self.assertIn('ram_mib', next((self.root / 'queue' / 'rejected').glob('*.reason')).read_text())
        self.assertFalse((self.root / 'runs' / 'raced-1').exists())

    def test_launch_failed_is_not_requeued(self):
        self.submit('broken-1', argv=[str(self.base / 'missing-binary')])
        self.tick()
        self.assertEqual(read_json(self.root / 'runs' / 'broken-1' / 'terminal.json')['status'], 'launch_failed')
        self.assertEqual(len(self.names('claimed')), 1)
        self.assertEqual(self.names('pending'), [])
        self.assertIsNone(self.tick()['launched'])

    def test_launch_recomputes_hashes_after_submit(self):
        data = self.base / 'in.bin'
        data.write_text('v1')
        self.submit('rehash-1', inputs=[str(data)])
        data.write_text('v2-changed')
        self.tick()
        run = read_json(self.root / 'runs' / 'rehash-1' / 'run.json')
        self.assertEqual(run['inputs'][0]['sha256'], sha('v2-changed'))
        wait_terminal(self.root / 'runs' / 'rehash-1')

    def test_two_ticks_are_mutually_exclusive(self):
        self.submit('lock-1')
        (self.root / 'scheduler.lock').touch()
        holder = open(self.root / 'scheduler.lock', 'a+')
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with self.assertRaises(scheduler.SchedulerBusy):
                self.tick()
            self.assertEqual(scheduler.main(['tick', '--root', str(self.root)]), 3)
        finally:
            holder.close()
        self.assertEqual(len(self.names('pending')), 1)

    def test_probe_failure_launches_nothing(self):
        self.submit('probe-1')
        (self.host.proc / 'meminfo').unlink()
        result = self.tick()
        self.assertIsNone(result['launched'])
        self.assertEqual(len(self.names('pending')), 1)

    def test_serve_stops_on_stop_file(self):
        (self.root / 'queue').mkdir(parents=True)
        (self.root / 'queue' / 'STOP').touch()
        self.assertEqual(scheduler.serve(self.root, probes=self.host.probes(), interval=0.01), 0)


class SchedulerChildrenTests(SchedulerCase):
    """M3: scheduler-launched supervisors are not scheduler children (no double count, no zombies)."""

    def test_scheduler_run_excludes_jobs_and_leaves_no_zombies(self):
        me = os.getpid()
        # the scheduler itself runs as an ops run whose supervisor tree contains this process
        make_run(self.root, 'sched', created_at=NOW - 5000,
                 owner={'sup': (me, None), 'job': (me, None), 'started_at': NOW - 5000})
        job = [PY, '-c', 'import time; b = bytearray(64 * 1024 * 1024); time.sleep(3)']
        for i in range(3):
            self.submit(f'job-{i}', argv=job)
        launched = [self.tick()['launched'] for _ in range(3)]
        self.assertEqual(launched, ['job-0', 'job-1', 'job-2'])
        owners = [read_json(self.root / 'runs' / f'job-{i}' / 'owner.json') for i in range(3)]
        for owner in owners:
            self.assertNotEqual(ppid_of(owner['supervisor']['pid']), me)
        proc = host_proc_snapshot(self.base)
        table = common.process_table(common.Probes(proc_root=proc))
        mine = common.descendants(table, me)
        for owner in owners:
            self.assertNotIn(owner['supervisor']['pid'], mine)
            self.assertNotIn(owner['job']['pid'], mine)
        observed = admission.observed_usage(mine, table, None)
        job_pids = {o[k]['pid'] for o in owners for k in ('supervisor', 'job')}
        job_rss = sum(table[p]['rss_kib'] for p in job_pids if p in table) / 1024
        self.assertGreater(job_rss, 0)
        self.assertTrue(mine.isdisjoint(job_pids))
        self.assertAlmostEqual(observed['ram_mib'], sum(table[p]['rss_kib'] for p in mine) / 1024)
        with_jobs = admission.observed_usage(mine | job_pids, table, None)['ram_mib']
        self.assertAlmostEqual(with_jobs - observed['ram_mib'], job_rss)
        for i in range(3):
            wait_terminal(self.root / 'runs' / f'job-{i}')
        time.sleep(0.2)
        with self.assertRaises(ChildProcessError):
            os.waitpid(-1, os.WNOHANG)


if __name__ == '__main__':
    unittest.main()
