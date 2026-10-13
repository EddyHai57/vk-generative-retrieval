import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ops_testlib import (OPS, PY, FixtureHost, base_spec, env_without_ops_root, git, make_repo,
                         ppid_of, read_json, sha, wait_for, wait_terminal)

from vkgr_ops import common, launcher

LAUNCH_SHIM = OPS / 'bin' / 'vkgr-launch'
A1 = sha('A = 1\n')
A2 = sha('A = 2\n')


class LauncherCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name).resolve()
        self.root = self.base / 'opsroot'
        self.repo = self.base / 'repo'
        self.head = make_repo(self.repo, {'README.md': 'fixture\n', 'src/a.py': 'A = 1\n'})

    def tearDown(self):
        self._tmp.cleanup()

    def launch(self, **kw):
        return launcher.launch(base_spec(self.repo, **kw), self.root)

    def rejected(self, message=None, **kw):
        with self.assertRaises(launcher.LaunchError) as raised:
            self.launch(**kw)
        self.assertEqual(raised.exception.code, 1)
        if message:
            self.assertIn(message, str(raised.exception))
        self.assertFalse((self.root / 'runs' / kw.get('run_id', 'job-1')).exists())
        return raised.exception


class StaticValidationTests(LauncherCase):
    def test_bad_run_id_rejected(self):
        for bad in ('Upper', '-lead', 'a' * 65, 'a/b', ''):
            with self.subTest(bad=bad):
                self.rejected('run_id', run_id=bad)

    def test_bad_resources_rejected(self):
        self.rejected('ram_mib', ram_mib=-1)
        self.rejected('cpu_cores', cpu_cores=1.5)
        self.rejected('wall_hours', wall_hours=0)
        self.rejected('wall_hours', wall_hours=721)
        self.rejected('wall_hours', wall_hours=float('inf'))

    def test_bad_env_keys_rejected(self):
        self.rejected('env', env=['lower=1'])
        self.rejected('env', env=['VKGR_RUN_ID=x'])
        self.rejected('env', env=['VKGR_PROGRESS_PATH=/x'])

    def test_root_must_be_given(self):
        with patch.dict(os.environ, env_without_ops_root(), clear=True):
            code = launcher.main(['--run-id', 'x', '--kind', 'smoke', '--cwd', str(self.repo),
                                  '--cpu-cores', '0', '--ram-mib', '0', '--vram-mib', '0',
                                  '--disk-gib', '0', '--wall-hours', '1', '--no-admission', '--', 'true'])
        self.assertEqual(code, 1)


class RunJsonTests(LauncherCase):
    def test_run_json_fields_complete(self):
        config = self.base / 'cfg.json'
        config.write_text('{"a": 1}')
        data = self.base / 'input.bin'
        data.write_bytes(b'x' * 1000)
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '3', 'VKGR_EXTRA': 'e', 'CUDA_VISIBLE_DEVICES': ''}):
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            run_dir = self.launch(kind='real', config=str(config), seed=7, inputs=[str(data)],
                                  env=['VKGR_EVAL_WORKERS=8', 'FOO=bar'])
        run = read_json(run_dir / 'run.json')
        self.assertEqual(oct((run_dir / 'run.json').stat().st_mode & 0o777), '0o400')
        for key in ('schema', 'run_id', 'kind', 'created_at', 'cwd', 'argv', 'env', 'git', 'overlay',
                    'pinned_files', 'config', 'seed', 'inputs', 'resources', 'stop_mode', 'admission',
                    'ops_source'):
            self.assertIn(key, run)
        self.assertEqual(run['schema'], 'vkgr-run-v1')
        self.assertEqual(run['git'], {'commit': self.head, 'dirty': False, 'dirty_files': []})
        self.assertIsNone(run['overlay'])
        self.assertEqual(run['config'], {'path': str(config), 'sha256': sha('{"a": 1}')})
        self.assertEqual(run['inputs'], [{'path': str(data), 'sha256': sha(b'x' * 1000), 'bytes': 1000}])
        self.assertEqual(run['seed'], 7)
        self.assertEqual(run['resources']['wall_seconds'], 3600.0)
        # N6: env records {key: value}; --env values as passed, whitelist from current env, absent keys omitted
        self.assertEqual(run['env']['VKGR_EVAL_WORKERS'], '8')
        self.assertEqual(run['env']['FOO'], 'bar')
        self.assertEqual(run['env']['OMP_NUM_THREADS'], '3')
        self.assertEqual(run['env']['VKGR_EXTRA'], 'e')
        self.assertEqual(run['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', run['env'])
        self.assertEqual(run['env']['VKGR_RUN_ID'], 'job-1')
        self.assertEqual(run['env']['VKGR_PROGRESS_PATH'], str(run_dir / 'progress.json'))
        self.assertEqual(len(run['ops_source']['commit']), 40)
        self.assertEqual(len(run['ops_source']['sha256']), 64)
        wait_terminal(run_dir)

    def test_job_receives_env_and_progress_path(self):
        code = ('import os,json;print(json.dumps({k:os.environ.get(k) for k in '
                '("VKGR_RUN_ID","VKGR_PROGRESS_PATH","FOO")}))')
        run_dir = self.launch(argv=[PY, '-c', code], env=['FOO=bar'])
        self.assertEqual(wait_terminal(run_dir)['status'], 'succeeded')
        seen = json.loads((run_dir / 'logs' / 'stdout.log').read_text())
        self.assertEqual(seen, {'VKGR_RUN_ID': 'job-1', 'VKGR_PROGRESS_PATH': str(run_dir / 'progress.json'),
                                'FOO': 'bar'})

    def test_duplicate_run_id_rejected(self):
        run_dir = self.launch()
        wait_terminal(run_dir)
        before = (run_dir / 'run.json').read_bytes()
        with self.assertRaises(launcher.LaunchError) as raised:
            self.launch()
        self.assertIn('exists', str(raised.exception))
        self.assertEqual((run_dir / 'run.json').read_bytes(), before)


class DirtyTests(LauncherCase):
    def test_dirty_real_rejected_without_run_dir(self):
        (self.repo / 'src/a.py').write_text('A = 2\n')
        self.rejected('dirty', kind='real')
        self.rejected('dirty', kind='ops')

    def test_dirty_smoke_allowed_and_recorded(self):
        (self.repo / 'src/a.py').write_text('A = 2\n')
        (self.repo / 'new.txt').write_text('n\n')
        run_dir = self.launch(kind='smoke')
        g = read_json(run_dir / 'run.json')['git']
        self.assertTrue(g['dirty'])
        self.assertEqual(g['dirty_files'], [{'path': 'new.txt', 'sha256': sha('n\n')},
                                            {'path': 'src/a.py', 'sha256': sha('A = 2\n')}])
        wait_terminal(run_dir)

    def test_dirty_file_pinned_is_allowed(self):
        (self.repo / 'src/a.py').write_text('A = 2\n')
        run_dir = self.launch(kind='real', pinned_files=['src/a.py=' + A2])
        run = read_json(run_dir / 'run.json')
        self.assertEqual(run['pinned_files'], {'src/a.py': sha('A = 2\n')})
        wait_terminal(run_dir)

    def test_pinned_sha_wrong_rejected(self):
        (self.repo / 'src/a.py').write_text('A = 2\n')
        self.rejected('sha256', kind='real', pinned_files=[f'src/a.py={"0" * 64}'])

    def test_deleted_path_matches_only_absent(self):
        # N3: ' D' porcelain path only matches a pinned "absent"
        (self.repo / 'src/a.py').unlink()
        self.rejected(kind='real', pinned_files=['src/a.py=' + A1])
        run_dir = self.launch(kind='real', pinned_files=['src/a.py=absent'])
        self.assertEqual(read_json(run_dir / 'run.json')['git']['dirty_files'],
                         [{'path': 'src/a.py', 'sha256': 'absent'}])
        wait_terminal(run_dir)


class PinnedManifestTests(unittest.TestCase):
    """Two-commit fixture replicating the G frozen-base overlay (p3c / provenance)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base_dir = Path(self._tmp.name).resolve()
        self.root = self.base_dir / 'opsroot'
        self.repo = self.base_dir / 'repo'
        self.base = make_repo(self.repo, {'.gitignore': 'provenance/\n', 'src/a.py': 'A = 1\n',
                                          'old.txt': 'old\n'})
        (self.repo / 'src/a.py').write_text('A = 2\n')
        (self.repo / 'docs').mkdir()
        (self.repo / 'docs/p3c.md').write_text('X target\n')
        (self.repo / 'old.txt').unlink()
        git(self.repo, 'add', '-A')
        git(self.repo, 'commit', '-q', '-m', 'target')
        self.target = git(self.repo, 'rev-parse', 'HEAD')
        git(self.repo, 'checkout', '-q', '--detach', self.base)
        # overlay: target files, original p3c (Y), target-deleted file removed, provenance data
        (self.repo / 'src/a.py').write_text('A = 2\n')
        (self.repo / 'docs').mkdir(exist_ok=True)
        (self.repo / 'docs/p3c.md').write_text('Y original\n')
        (self.repo / 'old.txt').unlink()
        (self.repo / 'provenance/g0-v1').mkdir(parents=True)
        (self.repo / 'provenance/g0-v1/data.json').write_text('{"p": 1}\n')

    def tearDown(self):
        self._tmp.cleanup()

    def manifest(self, **changes):
        doc = {'schema': 'vkgr-pinned-v1', 'base_commit': self.base, 'target_commit': self.target,
               'files': {'src/a.py': sha('A = 2\n'), 'old.txt': 'absent',
                         'provenance/g0-v1/data.json': sha('{"p": 1}\n')},
               'overrides': {'docs/p3c.md': {'sha256': sha('Y original\n'), 'reason': 'D0 frozen G1 input'}}}
        doc.update(changes)
        path = self.base_dir / f'pinned-{len(list(self.base_dir.glob("pinned-*")))}.json'
        path.write_text(json.dumps(doc))
        return path

    def launch(self, manifest, **kw):
        return launcher.launch(base_spec(self.repo, kind='real', pinned_manifest=str(manifest), **kw), self.root)

    def rejected(self, manifest, message=None, **kw):
        with self.assertRaises(launcher.LaunchError) as raised:
            self.launch(manifest, **kw)
        if message:
            self.assertIn(message, str(raised.exception))
        self.assertFalse((self.root / 'runs' / 'job-1').exists())

    def test_valid_overlay_admitted_and_recorded(self):
        path = self.manifest()
        run_dir = self.launch(path)
        run = read_json(run_dir / 'run.json')
        self.assertEqual(run['overlay'], {'base_commit': self.base, 'target_commit': self.target,
                                          'pinned_manifest_sha256': common.sha256_file(path)[0],
                                          'overrides': {'docs/p3c.md': sha('Y original\n')}})
        self.assertEqual(run['pinned_files']['provenance/g0-v1/data.json'], sha('{"p": 1}\n'))
        self.assertEqual(run['pinned_files']['old.txt'], 'absent')
        self.assertTrue(run['git']['dirty'])
        wait_terminal(run_dir)

    def test_provenance_sha_is_verified(self):
        (self.repo / 'provenance/g0-v1/data.json').write_text('{"p": 2}\n')
        self.rejected(self.manifest(), 'provenance/g0-v1/data.json')

    def test_base_mismatch_rejected(self):
        self.rejected(self.manifest(base_commit=self.target), 'base_commit')

    def test_unknown_target_rejected(self):
        self.rejected(self.manifest(target_commit='1' * 40), 'target_commit')

    def test_path_escape_rejected(self):
        for bad in ('../x', '/etc/x', 'src/../../x'):
            with self.subTest(bad=bad):
                files = {'src/a.py': sha('A = 2\n'), 'old.txt': 'absent', bad: '0' * 64}
                self.rejected(self.manifest(files=files), 'path')

    def test_symlink_path_rejected(self):
        (self.repo / 'provenance/link.json').symlink_to(self.repo / 'provenance/g0-v1/data.json')
        files = {'src/a.py': sha('A = 2\n'), 'old.txt': 'absent', 'provenance/link.json': sha('{"p": 1}\n')}
        self.rejected(self.manifest(files=files), 'symlink')

    def test_pinned_file_conflict_rejected(self):
        self.rejected(self.manifest(), 'conflict', pinned_files=[f'src/a.py={"1" * 64}'])

    def test_pinned_file_duplicate_consistent_ok(self):
        run_dir = self.launch(self.manifest(), pinned_files=['src/a.py=' + A2])
        wait_terminal(run_dir)

    def test_diff_path_missing_rejected(self):
        self.rejected(self.manifest(files={'src/a.py': sha('A = 2\n')}), 'old.txt')
        self.rejected(self.manifest(overrides={}), 'docs/p3c.md')

    def test_declared_sha_not_target_blob_rejected(self):
        (self.repo / 'src/a.py').write_text('A = 3\n')
        files = {'src/a.py': sha('A = 3\n'), 'old.txt': 'absent'}
        self.rejected(self.manifest(files=files), 'target blob')

    def test_override_put_in_files_rejected(self):
        files = {'src/a.py': sha('A = 2\n'), 'old.txt': 'absent', 'docs/p3c.md': sha('Y original\n')}
        self.rejected(self.manifest(files=files, overrides={}), 'target blob')

    def test_override_sha_wrong_rejected(self):
        bad = {'docs/p3c.md': {'sha256': sha('nope'), 'reason': 'D0'}}
        self.rejected(self.manifest(overrides=bad), 'docs/p3c.md')

    def test_files_and_overrides_overlap_rejected(self):
        files = {'src/a.py': sha('A = 2\n'), 'old.txt': 'absent', 'docs/p3c.md': sha('Y original\n')}
        self.rejected(self.manifest(files=files), 'overlap')

    def test_override_reason_bounds(self):
        for reason in ('', 'r' * 201, 5):
            with self.subTest(reason=reason):
                bad = {'docs/p3c.md': {'sha256': sha('Y original\n'), 'reason': reason}}
                self.rejected(self.manifest(overrides=bad), 'reason')

    def test_override_extra_key_rejected(self):
        bad = {'docs/p3c.md': {'sha256': sha('Y original\n'), 'reason': 'x', 'extra': 1}}
        self.rejected(self.manifest(overrides=bad), 'override')

    def test_schema_strict(self):
        self.rejected(self.manifest(extra=1), 'key')
        self.rejected(self.manifest(schema='vkgr-pinned-v0'), 'schema')
        files = {'src/a.py': 'ABC', 'old.txt': 'absent'}
        self.rejected(self.manifest(files=files), 'sha256')

    def test_absent_rules(self):
        (self.repo / 'old.txt').write_text('old\n')
        self.rejected(self.manifest(), 'old.txt')
        (self.repo / 'old.txt').unlink()
        files = {'src/a.py': 'absent', 'old.txt': 'absent'}
        self.rejected(self.manifest(files=files), 'src/a.py')

    def test_overrides_optional(self):
        # without the override path in the diff the manifest is complete without overrides
        (self.repo / 'docs/p3c.md').write_text('X target\n')
        files = {'src/a.py': sha('A = 2\n'), 'old.txt': 'absent', 'docs/p3c.md': sha('X target\n'),
                 'provenance/g0-v1/data.json': sha('{"p": 1}\n')}
        doc = {'schema': 'vkgr-pinned-v1', 'base_commit': self.base, 'target_commit': self.target,
               'files': files}
        path = self.base_dir / 'pinned-nooverrides.json'
        path.write_text(json.dumps(doc))
        run_dir = self.launch(path)
        self.assertEqual(read_json(run_dir / 'run.json')['overlay']['overrides'], {})
        wait_terminal(run_dir)

    def test_unpinned_dirty_path_rejected(self):
        (self.repo / 'stray.txt').write_text('s\n')
        self.rejected(self.manifest(), 'dirty')


class OrderingTests(LauncherCase):
    def test_input_hashing_happens_before_admission_lock(self):
        data = self.base / 'big.bin'
        data.write_bytes(b'y' * 10)
        events = []
        real_sha, real_lock = common.sha256_file, launcher.acquire_admission_lock

        def record_sha(path):
            events.append(('sha', str(path)))
            return real_sha(path)

        def record_lock(*a, **kw):
            events.append(('lock', None))
            return real_lock(*a, **kw)

        host = FixtureHost(self.base / 'host')
        with patch.object(common, 'sha256_file', side_effect=record_sha), \
             patch.object(launcher, 'acquire_admission_lock', side_effect=record_lock):
            run_dir = launcher.launch(base_spec(self.repo, inputs=[str(data)], no_admission=False),
                                      self.root, probes=host.probes())
        lock_at = events.index(('lock', None))
        sha_at = [i for i, e in enumerate(events) if e == ('sha', str(data))]
        self.assertTrue(sha_at)
        self.assertTrue(all(i < lock_at for i in sha_at), events)
        wait_terminal(run_dir)

    def test_admission_lock_busy_times_out(self):
        import fcntl
        self.root.mkdir(parents=True)
        holder = open(self.root / 'admission.lock', 'a+')
        fcntl.flock(holder, fcntl.LOCK_EX)
        try:
            host = FixtureHost(self.base / 'host')
            with patch.object(launcher, 'ADMISSION_LOCK_TIMEOUT', 0.3):
                with self.assertRaises(launcher.LaunchError) as raised:
                    launcher.launch(base_spec(self.repo, no_admission=False), self.root, probes=host.probes())
            self.assertIn('admission lock busy', str(raised.exception))
            self.assertFalse((self.root / 'runs' / 'job-1').exists())
        finally:
            holder.close()


class SupervisorTests(LauncherCase):
    def test_python_isolated_supervisor_starts_with_real_argv(self):
        # M4: the real Popen argv uses -I; nothing is mocked. owner.json only appears if imports work.
        run_dir = self.launch(argv=[PY, '-c', 'print("ok")'])
        self.assertEqual(wait_terminal(run_dir)['status'], 'succeeded')
        sup_log = (run_dir / 'logs' / 'supervisor.log').read_text()
        self.assertNotIn('Error', sup_log)
        owner = read_json(run_dir / 'owner.json')
        self.assertIn(owner['birth_source'], ('proc-stat-22', 'unavailable'))

    def test_supervisor_argv_keeps_isolated_flag(self):
        argv = launcher.supervisor_argv(Path('/r/runs/x'))
        self.assertEqual(argv[1], '-I')
        self.assertTrue(argv[2].endswith('vkgr_ops/supervisor.py'))

    def test_supervisor_imports_under_isolated_mode_from_other_cwd(self):
        # M4 in isolation: python -I <ops>/vkgr_ops/supervisor.py --help must import vkgr_ops.common
        out = subprocess.run([PY, '-I', str(OPS / 'vkgr_ops' / 'supervisor.py'), '--check-imports'],
                             cwd=str(self.base), capture_output=True, text=True, env={'PATH': '/usr/bin:/bin'})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), 'imports-ok')

    def test_shims_run_under_isolated_mode(self):
        for name in ('vkgr-launch', 'vkgr-sched', 'vkgr-status'):
            with self.subTest(name=name):
                out = subprocess.run([PY, '-I', str(OPS / 'bin' / name), '--help'], cwd=str(self.base),
                                     capture_output=True, text=True)
                self.assertEqual(out.returncode, 0, out.stderr)

    def test_supervisor_is_double_forked(self):
        # M3: the supervisor must not be a child of the launching process
        run_dir = self.launch(argv=[PY, '-c', 'import time; time.sleep(2)'])
        owner = read_json(run_dir / 'owner.json')
        sup = owner['supervisor']['pid']
        self.assertNotEqual(ppid_of(sup), os.getpid())
        self.assertEqual(owner['supervisor']['sid'], sup)          # setsid done by the grandchild
        self.assertEqual(owner['job']['pgid'], owner['supervisor']['pgid'])
        wait_terminal(run_dir)
        with self.assertRaises(ChildProcessError):
            os.waitpid(-1, os.WNOHANG)                               # no zombie / unreaped child left

    def test_ssh_disconnect_does_not_stop_job(self):
        script = (f'"{PY}" -I "{LAUNCH_SHIM}" --root "{self.root}" --run-id ssh-1 --kind smoke '
                  f'--cwd "{self.repo}" --cpu-cores 0 --ram-mib 0 --vram-mib 0 --disk-gib 0 '
                  f'--wall-hours 1 --no-admission -- "{PY}" -c '
                  f'"import time; time.sleep(2); print(\'ok\')" && echo launched && sleep 30')
        middle = subprocess.Popen(['/bin/sh', '-c', script], stdout=subprocess.PIPE, text=True,
                                  start_new_session=True)
        try:
            lines = []
            while True:
                line = middle.stdout.readline()
                if not line:
                    break
                lines.append(line)
                if line.strip() == 'launched':
                    break
            self.assertIn('launched\n', lines, lines)
            os.killpg(middle.pid, signal.SIGHUP)
            time.sleep(0.2)
            try:
                os.killpg(middle.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass      # group already gone; macOS reports EPERM for a zombie-only group
        finally:
            middle.wait()
            middle.stdout.close()
        run_dir = self.root / 'runs' / 'ssh-1'
        terminal = wait_terminal(run_dir)
        self.assertEqual(terminal['status'], 'succeeded')
        self.assertEqual(terminal['exit_code'], 0)
        self.assertEqual((run_dir / 'logs' / 'stdout.log').read_text().strip(), 'ok')

    def test_job_sees_only_std_fds(self):
        code = ('import os\nd="/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"\n'
                'print(sorted(int(x) for x in os.listdir(d)))')
        run_dir = self.launch(argv=[PY, '-c', code])
        wait_terminal(run_dir)
        fds = json.loads((run_dir / 'logs' / 'stdout.log').read_text())
        self.assertEqual(fds[:3], [0, 1, 2])
        self.assertLessEqual(len(fds), 4, fds)

    def test_exec_failure_is_launch_failed(self):
        with self.assertRaises(launcher.LaunchError) as raised:
            self.launch(argv=[str(self.base / 'missing-binary')])
        self.assertEqual(raised.exception.code, 1)
        terminal = read_json(self.root / 'runs' / 'job-1' / 'terminal.json')
        self.assertEqual(terminal['status'], 'launch_failed')
        self.assertIn('missing-binary', (self.root / 'runs' / 'job-1' / 'logs' / 'supervisor.log').read_text())

    def test_failed_exit_code(self):
        run_dir = self.launch(argv=[PY, '-c', 'raise SystemExit(3)'])
        terminal = wait_terminal(run_dir)
        self.assertEqual((terminal['status'], terminal['exit_code'], terminal['signal']), ('failed', 3, None))
        self.assertEqual(oct((run_dir / 'terminal.json').stat().st_mode & 0o777), '0o400')


class StopModeTests(LauncherCase):
    WALL_1S = 1.0 / 3600

    def test_signal_mode_wall_cap(self):
        run_dir = self.launch(argv=[PY, '-c', 'import time; time.sleep(30)'], wall_hours=self.WALL_1S)
        terminal = wait_terminal(run_dir)
        self.assertEqual(terminal['status'], 'wall_cap_exceeded')
        self.assertTrue(terminal['wall_cap_exceeded'])
        self.assertEqual(terminal['signal'], signal.SIGTERM)
        self.assertLess(terminal['wall_seconds'], 20)

    def test_signal_mode_supervisor_sigterm_is_stopped(self):
        run_dir = self.launch(argv=[PY, '-c', 'import time; time.sleep(30)'])
        sup = read_json(run_dir / 'owner.json')['supervisor']['pid']
        os.kill(sup, signal.SIGTERM)
        terminal = wait_terminal(run_dir)
        self.assertEqual(terminal['status'], 'stopped')
        self.assertFalse(terminal['wall_cap_exceeded'])

    def test_signal_reason_precedence_first_wins(self):
        # N7: wall cap fires first; a later SIGTERM to the supervisor does not change the status
        code = ('import signal,time\nsignal.signal(signal.SIGTERM, lambda *a: None)\n'
                'time.sleep(3)\nraise SystemExit(0)')
        run_dir = self.launch(argv=[PY, '-c', code], wall_hours=self.WALL_1S)
        sup = read_json(run_dir / 'owner.json')['supervisor']['pid']
        time.sleep(1.6)
        os.kill(sup, signal.SIGTERM)
        terminal = wait_terminal(run_dir)
        self.assertEqual(terminal['status'], 'wall_cap_exceeded')
        self.assertEqual(terminal['exit_code'], 0)

    def test_none_mode_never_signals_job(self):
        run_dir = self.launch(argv=[PY, '-c', 'import time; time.sleep(3); print("done")'],
                              wall_hours=self.WALL_1S, stop_mode='none')
        sup = read_json(run_dir / 'owner.json')['supervisor']['pid']
        time.sleep(1.5)
        os.kill(sup, signal.SIGTERM)
        terminal = wait_terminal(run_dir)
        self.assertEqual(terminal['status'], 'succeeded')
        self.assertTrue(terminal['wall_cap_exceeded'])
        self.assertEqual(terminal['exit_code'], 0)
        self.assertEqual((run_dir / 'logs' / 'stdout.log').read_text().strip(), 'done')
        self.assertIn('SIGTERM', (run_dir / 'logs' / 'supervisor.log').read_text())


if __name__ == '__main__':
    unittest.main()
