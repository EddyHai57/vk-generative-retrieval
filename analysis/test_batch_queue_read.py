"""Synthetic bounded metadata receipts; never opens real scheduler inputs."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analysis import batch_queue_read as m


class BatchRead(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for sub in ('queue/pending', 'queue/claimed', 'queue/rejected', 'runs/alpha/logs', 'ops'):
            (self.root/sub).mkdir(parents=True, exist_ok=True)
        self.req = {'root': str(self.root), 'run_ids': ['alpha'], 'driver_records': []}

    def put(self, rel, value):
        p = self.root/rel
        p.write_text(json.dumps(value))
        return p

    def test_claimed_terminal_identity_and_streamed_log_tail(self):
        q = self.put('queue/claimed/123-alpha.json', {'run_id': 'alpha', 'argv': ['synthetic']})
        self.put('runs/alpha/terminal.json', {'status': 'succeeded'})
        self.put('runs/alpha/run.json', {'run_id': 'alpha', 'pid': 987654})
        log = self.root/'runs/alpha/logs/stdout.log'
        raw = b'a'*25000+b'END'; log.write_bytes(raw)
        before = {str(p): p.stat().st_mtime_ns for p in self.root.rglob('*') if p.is_file()}
        result = m.collect(self.req)
        self.assertTrue(result['readonly'])
        self.assertEqual(result['runs']['alpha']['queue']['claimed'][0]['sha256'], hashlib.sha256(q.read_bytes()).hexdigest())
        entry = result['runs']['alpha']['logs']['stdout']
        self.assertEqual(entry['tail'], raw[-8192:].decode())
        self.assertEqual(entry['bytes'], len(raw))
        self.assertEqual(entry['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in self.root.rglob('*') if p.is_file()})
        self.assertIn('read_at_utc', entry)

    def test_rejected_reads_both_known_reason_names_and_no_other_runs(self):
        q = self.put('queue/rejected/123-alpha.json', {'run_id': 'alpha'})
        q.with_suffix('.reason').write_text('first')
        Path(str(q)+'.reason').write_text('second')
        self.put('queue/claimed/123-other.json', {'run_id': 'other'})
        result = m.collect(self.req)['runs']['alpha']
        self.assertEqual(len(result['queue']['rejected']), 1)
        self.assertEqual([x['tail'] for x in result['queue']['rejected'][0]['reasons']], ['first', 'second'])
        self.assertFalse(result['queue']['claimed'])
        self.assertEqual(result['terminal']['status'], 'missing')

    def test_fail_closed_path_schema_and_size(self):
        for rid in ('../alpha', '/alpha', 'alpha*', 'week26-test', 'embedding-run'):
            with self.assertRaises(ValueError):
                m.collect(dict(self.req, run_ids=[rid]))
        with self.assertRaises(ValueError):
            m.collect(dict(self.req, root='relative'))
        self.put('runs/alpha/job.json', [])
        with self.assertRaises(ValueError):
            m.collect(self.req)
        (self.root/'runs/alpha/job.json').write_bytes(b' '* (m.JSON_LIMIT+1))
        with self.assertRaises(ValueError):
            m.collect(self.req)

    def test_test_named_queue_artifact_is_never_opened(self):
        self.put('queue/claimed/test-alpha.json', {'run_id': 'alpha'})
        with self.assertRaises(ValueError):
            m.collect(self.req)

    def test_parent_symlink_rejected_before_read(self):
        outside = self.root/'outside'; outside.mkdir()
        (outside/'terminal.json').write_text('{}')
        # Existing alpha is retained; a separate requested identity is a symlink.
        (self.root/'runs/beta').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            m.collect(dict(self.req, run_ids=['beta']))

    def test_driver_actual_pid_ppid_session_from_proc_and_dead_race(self):
        path = self.put('ops/driver-pid.json', {'pid': 456, 'ppid': 999})
        req = dict(self.req, driver_records=[str(path)])
        with patch.object(m, 'process_state', return_value={'pid':456,'alive':True,'actual_ppid':1,'sid':456}):
            state = m.collect(req)['drivers'][0]['process']
        self.assertEqual(state['actual_ppid'], 1)
        stat = '456 (name with ) spaces) S 1 456 456 '+ '0 '*16+'123 0'
        with patch.object(Path, 'read_text', return_value=stat):
            self.assertEqual(m.process_state(456)['sid'], 456)
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            self.assertFalse(m.process_state(456)['alive'])
        with self.assertRaises(ValueError):
            m.process_state(-1)

    def test_stream_reads_in_chunks_and_marks_nonregular_files(self):
        stream = io.BytesIO(b'x'*20000)
        with patch.object(Path, 'open', return_value=stream):
            value = m.stream_summary(self.root/'unused')
        self.assertEqual(value['bytes'], 20000)
        self.assertEqual(len(value['tail']), 8192)


if __name__ == '__main__':
    unittest.main()
