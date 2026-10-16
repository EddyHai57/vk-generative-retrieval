"""Synthetic event backend tests; no server and no timer/daemon."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analysis import wait_terminal_event as m


class FakeEvents:
    def __init__(self, now):
        self.now = now; self.armed = []; self.actions = []; self.on_arm = None; self.closed = False
    def arm(self, path):
        self.armed.append(path)
        if self.on_arm:
            self.on_arm(path)
    def wait(self, seconds):
        if self.actions:
            self.now[0] += .01
            return self.actions.pop(0)()
        self.now[0] += seconds
        return []
    def close(self):
        self.closed = True


class EventWait(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp'); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.runs = self.root/'runs'; self.runs.mkdir()
        self.req = {'root':str(self.root),'run_ids':['alpha','beta'],'timeout_seconds':30}
        self.now = [100.]; self.backend = FakeEvents(self.now)
    def terminal(self, rid, status='succeeded'):
        folder = self.runs/rid; folder.mkdir(exist_ok=True)
        path = folder/'terminal.json'; path.write_text(json.dumps({'status':status}))
        return path
    def run_wait(self):
        return m.wait_terminal(self.req, backend_factory=lambda:self.backend, clock=lambda:self.now[0])
    def test_existing_terminal_needs_no_backend_and_has_receipt(self):
        path = self.terminal('beta')
        with patch.object(m, 'LinuxEvents', side_effect=AssertionError('must not arm')):
            result = m.wait_terminal(self.req)
        self.assertEqual(result['first']['run_id'],'beta')
        self.assertEqual(result['first']['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result['terminal'],{'alpha':False,'beta':True})
        self.assertTrue(result['readonly']); self.assertIn('read_at_utc',result['first'])
    def test_new_run_then_terminal_close_write(self):
        def directory():
            (self.runs/'alpha').mkdir()
            return [(self.runs,'alpha',m.IN_CREATE|m.IN_ISDIR)]
        def terminal():
            self.terminal('alpha')
            return [(self.runs/'alpha','terminal.json',m.IN_CLOSE_WRITE)]
        self.backend.actions=[directory,terminal]
        result=self.run_wait()
        self.assertEqual(result['status'],'TERMINAL_EVENT'); self.assertEqual(result['first']['run_id'],'alpha')
        self.assertIn(self.runs/'alpha',self.backend.armed); self.assertTrue(self.backend.closed)
    def test_incomplete_terminal_waits_for_close_write(self):
        def directory():
            path=self.terminal('alpha'); path.write_text('{')
            return [(self.runs,'alpha',m.IN_CREATE|m.IN_ISDIR)]
        def complete():
            self.terminal('alpha')
            return [(self.runs/'alpha','terminal.json',m.IN_CLOSE_WRITE)]
        self.backend.actions=[directory,complete]
        self.assertEqual(self.run_wait()['first']['value']['status'],'succeeded')
    def test_atomic_moved_to_terminal(self):
        (self.runs/'alpha').mkdir()
        self.backend.actions=[lambda:(self.terminal('alpha'),[(self.runs/'alpha','terminal.json',m.IN_MOVED_TO)])[1]]
        self.assertEqual(self.run_wait()['first']['run_id'],'alpha')
        self.assertLess(self.now[0],101.)  # Must wake on event, not only final deadline snapshot.
    def test_terminal_created_during_root_arm_closes_race(self):
        self.backend.on_arm=lambda path:self.terminal('alpha') if path==self.runs else None
        self.assertEqual(self.run_wait()['first']['run_id'],'alpha')
    def test_terminal_created_during_run_arm_closes_race(self):
        (self.runs/'alpha').mkdir()
        self.backend.on_arm=lambda path:self.terminal('alpha') if path==self.runs/'alpha' else None
        self.assertEqual(self.run_wait()['first']['run_id'],'alpha')
    def test_timeout_is_not_failure(self):
        result=self.run_wait()
        self.assertEqual(result['status'],'WAIT_TIMEOUT'); self.assertIsNone(result['first'])
        self.assertEqual(self.now[0],130.); self.assertTrue(self.backend.closed)
    def test_other_run_and_nonterminal_files_do_not_trigger(self):
        def other():
            self.terminal('other')
            return [(self.runs,'other',m.IN_CREATE|m.IN_ISDIR),(self.runs/'alpha','checkpoint.pt',m.IN_CLOSE_WRITE)]
        self.backend.actions=[other]
        self.assertEqual(self.run_wait()['status'],'WAIT_TIMEOUT')
    def test_unsafe_inputs_symlinks_and_nonobject_terminal_rejected(self):
        for field,value in [('root','relative'),('run_ids',['../alpha']),('run_ids',['test-run']),('timeout_seconds',1801),('timeout_seconds',True),('timeout_seconds',float('inf')),('run_ids',['alpha','alpha'])]:
            with self.assertRaises(ValueError):m.wait_terminal(dict(self.req,**{field:value}))
        path=self.terminal('beta'); path.write_text('[]')
        with self.assertRaises(ValueError):self.run_wait()
        path.write_bytes(b' '* (m.JSON_LIMIT+1))
        with self.assertRaises(ValueError):self.run_wait()
        path.unlink()
        (self.runs/'alpha').symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):self.run_wait()
    def test_non_linux_reports_unknown_without_fallback_polling(self):
        with patch.object(m.sys,'platform','darwin'):
            result=m.wait_terminal(self.req)
        self.assertEqual(result['status'],'UNKNOWN'); self.assertIn('inotify',result['reason'])
    def test_overflow_fails_closed(self):
        self.backend.actions=[lambda:[(None,'',m.IN_Q_OVERFLOW)]]
        with self.assertRaises(ValueError):self.run_wait()


if __name__=='__main__':unittest.main()
