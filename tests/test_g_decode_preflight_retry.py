import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


class RetryTests(unittest.TestCase):
    def test_cli_dry_and_invalid(self):
        cmd = [sys.executable, '-B', '-m', 'analysis.g_decode_preflight_retry']
        good = subprocess.run(cmd + ['dry'], capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        value = json.loads(good.stdout)
        self.assertEqual(value['run_id'], 'g-decode-timing-preflight-retry1-v1')
        self.assertEqual(value['engine_sha'], '87159f720afaf42be6252223442ece56a20e0ef9')
        self.assertEqual(value['resources'], {'cpu_cores': 1, 'ram_mib': 98304,
                                           'vram_mib': 8192, 'wall_hours': 8})
        self.assertFalse(value['science_training'])
        for argv in (['invalid'], ['run'], ['dry', '--execute-on-new-instance']):
            self.assertNotEqual(subprocess.run(cmd + argv, capture_output=True).returncode, 0)

    def test_progress_preserves_original_preflight_result_and_restores_methods(self):
        from analysis.g_decode_preflight_retry import observed_preflight
        calls = []
        marker = object()
        class Session:
            def decode(self, query):
                calls.append(query)
                return marker
        session = Session()
        class Factory:
            def __call__(self):
                return session
        def inputs(*args):
            calls.append(args)
            return [1, 2]
        def preflight(*args):
            values = engine.inputs(*args)
            current = engine.SessionFactory()()
            return [current.decode(v) for v in values]
        engine = SimpleNamespace(inputs=inputs, SessionFactory=Factory, preflight=preflight)
        original_call = Factory.__call__
        class FlushedSink(io.StringIO):
            flushes = 0
            def flush(self):
                self.flushes += 1
                super().flush()
        log = FlushedSink()
        with contextlib.redirect_stderr(log):
            actual = observed_preflight(engine, 'root', 'lock', 'ops')
        self.assertTrue(all(v is marker for v in actual))
        self.assertEqual(calls, [('root', 'lock', 'ops'), 1, 2])
        self.assertIs(engine.inputs, inputs)
        self.assertIs(Factory.__call__, original_call)
        self.assertNotIn('decode', vars(session))
        events = [json.loads(row) for row in log.getvalue().splitlines()]
        self.assertEqual([(x['stage'], x['event']) for x in events],
                         [('preflight', 'begin'), ('inputs', 'begin'), ('inputs', 'end'),
                          ('session_build', 'begin'), ('session_build', 'end'),
                          ('query', 'begin'), ('query', 'end'),
                          ('query', 'begin'), ('query', 'end'), ('preflight', 'end')])
        self.assertEqual(log.flushes, len(events))
        self.assertEqual([x['query_index'] for x in events if x['stage'] == 'query'], [1, 1, 2, 2])
        self.assertTrue(all('utc' in x and x['elapsed_seconds'] >= 0 for x in events))

    def test_failure_logs_error_and_restores_methods(self):
        from analysis.g_decode_preflight_retry import observed_preflight
        class Factory:
            def __call__(self):
                raise ValueError('fixture')
        def inputs(*args):
            return None
        engine = SimpleNamespace(inputs=inputs, SessionFactory=Factory,
                                 preflight=lambda *args: engine.SessionFactory()())
        original_call = Factory.__call__
        log = io.StringIO()
        with contextlib.redirect_stderr(log), self.assertRaises(ValueError):
            observed_preflight(engine, 'root', 'lock', 'ops')
        self.assertIs(Factory.__call__, original_call)
        self.assertIs(engine.inputs, inputs)
        self.assertEqual(json.loads(log.getvalue().splitlines()[-1])['event'], 'error')

    def test_source_identity_rejects_dirty_bytes_wrong_commit_and_symlinks(self):
        from analysis.g_decode_preflight_retry import source_ledger
        root = Path(tempfile.mkdtemp(prefix='vkgr-retry-source-')).resolve()
        def git(*argv):
            return subprocess.check_output(['git', '-C', str(root), *argv], text=True).strip()
        git('init', '-q')
        target = root / 'source.py'; target.write_text('VALUE = 1\n')
        git('add', 'source.py')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
        sha = git('rev-parse', 'HEAD')
        self.assertIn('source.py', source_ledger(root, sha)['files'])
        with self.assertRaises(ValueError):
            source_ledger(root, '0' * 40)
        target.write_text('VALUE = 2\n')
        with self.assertRaises(ValueError):
            source_ledger(root, sha)
        link = root / 'source-link'; link.symlink_to(target)
        with self.assertRaises(ValueError):
            source_ledger(link, sha)


    def test_retry_collision_refused_in_runs_and_each_queue_state(self):
        from analysis.g_decode_preflight_retry import unoccupied, RUN_ID
        for state in ('runs', 'pending', 'claimed', 'rejected'):
            root = Path(tempfile.mkdtemp(prefix='vkgr-retry-collision-')).resolve()
            if state == 'runs':
                (root / 'runs' / RUN_ID).mkdir(parents=True)
            else:
                target = root / 'queue' / state / ('000-' + RUN_ID + '.json')
                target.parent.mkdir(parents=True)
                target.write_text('{}')
            with self.subTest(state=state), self.assertRaises(FileExistsError):
                unoccupied(root)

    def test_real_launcher_prepare_and_dry_chain(self):
        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(repo), str(repo / 'src')]),
               'PYTHONDONTWRITEBYTECODE': '1'}
        result = subprocess.run([sys.executable, '-B', 'tests/fixtures/g_decode_preflight_retry_chain.py'],
                                cwd=repo, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('real_launcher_prepare_and_dry_verified', result.stdout)


if __name__ == '__main__':
    unittest.main()
