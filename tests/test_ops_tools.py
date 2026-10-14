"""ops/tools: argument parsing, dry-run, config and spec builders (standard library only, no ssh, no vkgr)."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from ops_testlib import ROOT, make_repo

TOOLS = ROOT / 'ops' / 'tools'


def load(rel):
    path = TOOLS / rel
    spec = importlib.util.spec_from_file_location('tool_' + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_main(mod, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = mod.main(argv)
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def write_config(directory, **over):
    cfg = json.loads((TOOLS / 'config.example.json').read_text())
    cfg['state_dir'] = str(directory)
    cfg.update(over)
    path = Path(directory) / 'cfg.json'
    path.write_text(json.dumps(cfg))
    return str(path)


R3_MANIFEST = {
    'schema': 'deploy01-r3-manifest-v2',
    'step1_sign_permit': {'tool': 'launch', 'launch_args': ['--run-id', 'sign-1', '--cwd', '/g', '--', 'py']},
    'step2_train': {'tool': 'submit', 'launch_args_template': ['--run-id', 'train-1', '--cwd', '/g', '--env', 'VKGR_EVAL_WORKERS=6',
                                                               '--', 'py', '--permit-sha', '<PERMIT_SHA256>']},
    'step3_parity_decode_only': {'tool': 'submit', 'launch_args': ['--run-id', 'parity-1', '--cwd', '/g', '--', 'py']},
}


class ToolConfigTest(unittest.TestCase):
    def test_missing_config_is_error(self):
        toolcfg = load('local/toolcfg.py')
        from unittest import mock
        with mock.patch.dict('os.environ', {}, clear=True), self.assertRaisesRegex(toolcfg.ConfigError, 'config required'):
            toolcfg.load(None)
        with self.assertRaisesRegex(toolcfg.ConfigError, 'unreadable'):
            toolcfg.load('/nonexistent/x.json')

    def test_missing_server_key_is_error(self):
        toolcfg = load('local/toolcfg.py')
        with tempfile.TemporaryDirectory() as d:
            cfg = json.loads((TOOLS / 'config.example.json').read_text())
            del cfg['server']['ops_root']
            p = Path(d) / 'c.json'
            p.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(toolcfg.ConfigError, 'ops_root'):
                toolcfg.load(str(p))

    def test_every_remote_script_renders_and_compiles(self):
        toolcfg = load('local/toolcfg.py')
        with tempfile.TemporaryDirectory() as d:
            cfg = toolcfg.load(write_config(d))
            scripts = sorted((TOOLS / 'remote').glob('*.py'))
            self.assertGreaterEqual(len(scripts), 20)
            for script in scripts:
                text = toolcfg.render(script, cfg)
                compile(text, str(script), 'exec')
                self.assertTrue(text.startswith('import json as _cfg_json\nCFG = '), script.name)
                self.assertIn("CFG['server']", text, f'{script.name} must take server paths from CFG')

    def test_no_hard_coded_server_or_local_paths(self):
        for path in sorted(TOOLS.rglob('*.py')):
            text = path.read_text()
            for bad in ('/workspace/', '/Users/', '/private/tmp/vkgr'):
                self.assertNotIn(bad, text, f'{path.name} hard-codes {bad}')


class TransportTest(unittest.TestCase):
    def test_parse_rejects_readonly_with_input(self):
        t = load('local/transport.py')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            t.parse_args(['s.py', 'p', '--n', '3', '--readonly', '--input', 'x'])

    def test_parse_requires_input_or_readonly(self):
        t = load('local/transport.py')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            t.parse_args(['s.py', 'p', '--n', '3'])

    def test_dry_run_sends_nothing(self):
        t = load('local/transport.py')
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run_main(t, ['--config', write_config(d), '--n', '17', str(TOOLS / 'remote/poll_read.py'),
                                        'pfx', '--readonly', '--dry-run'])
            self.assertEqual(code, 0)
            v = json.loads(out)
            self.assertTrue(v['dry_run'])
            self.assertEqual(v['ssh_argv'][0], 'ssh')
            self.assertIn('ControlMaster=no', v['ssh_argv'])
            self.assertEqual(len(v['rendered_sha256']), 64)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ['cfg.json'])


class HoldMuxTest(unittest.TestCase):
    def test_parse_n_range(self):
        h = load('local/hold_mux.py')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            h.parse_args(['--n', '1'])

    def test_dry_run_refuses_without_previous(self):
        h = load('local/hold_mux.py')
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run_main(h, ['--config', write_config(d), '--n', '5', '--dry-run'])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(out)['guard'].startswith('refuse:'))

    def test_guard_refuses_live_previous_and_accepts_exited(self):
        h = load('local/hold_mux.py')
        import os
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'ssh-control-state-formal04.json').write_text(json.dumps(
                {'socket_path': str(d / 'gone'), 'master_pid': os.getpid(), 'owner_pid': os.getpid()}))
            (d / 'ssh-control-terminal-formal04.json').write_text(json.dumps({'actual_master_exit': 0, 'socket_exists': False}))
            with self.assertRaisesRegex(ValueError, 'still live'):
                h.guard(d, 5)
            (d / 'ssh-control-state-formal04.json').write_text(json.dumps(
                {'socket_path': str(d / 'gone'), 'master_pid': 2 ** 22 + 7, 'owner_pid': 2 ** 22 + 9}))
            self.assertEqual(h.guard(d, 5), ('formal05', 'formal04'))


class R3SubmitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest = Path(self.tmp.name) / 'm.json'
        self.manifest.write_text(json.dumps(R3_MANIFEST))
        self.mod = load('local/r3_submit.py')

    def tearDown(self):
        self.tmp.cleanup()

    def test_refuses_without_confirm_reviewed(self):
        code, _, err = run_main(self.mod, ['--manifest', str(self.manifest), '--n', '17', '--step', 'sign'])
        self.assertNotEqual(code, 0)
        self.assertIn('--confirm-reviewed', err)

    def test_dry_run_and_confirm_exclusive(self):
        code, _, _ = run_main(self.mod, ['--manifest', str(self.manifest), '--n', '17', '--step', 'sign',
                                         '--dry-run', '--confirm-reviewed'])
        self.assertNotEqual(code, 0)

    def test_dry_run_each_step(self):
        for step, run_id in (('sign', 'sign-1'), ('train', 'train-1'), ('parity', 'parity-1')):
            code, out, _ = run_main(self.mod, ['--manifest', str(self.manifest), '--n', '17', '--step', step, '--dry-run'])
            self.assertEqual(code, 0, step)
            v = json.loads(out)
            self.assertEqual(v['payload']['launch_args'][1], run_id)
        self.assertIn('<PERMIT_SHA256>', json.loads(run_main(self.mod, ['--manifest', str(self.manifest), '--n', '17',
                                                                        '--step', 'train', '--dry-run'])[1])['payload']['launch_args'])

    def test_payload_fills_permit_sha(self):
        body = self.mod.payload(R3_MANIFEST, 'train', 'a' * 64)
        self.assertEqual(body['tool'], 'submit')
        self.assertIn('a' * 64, body['launch_args'])
        self.assertNotIn('<PERMIT_SHA256>', body['launch_args'])
        with self.assertRaises(SystemExit):
            self.mod.payload(R3_MANIFEST, 'train', 'short')

    def test_wrong_schema_refused(self):
        self.manifest.write_text(json.dumps({**R3_MANIFEST, 'schema': 'deploy01-r3-manifest-v1'}))
        code, _, _ = run_main(self.mod, ['--manifest', str(self.manifest), '--n', '1', '--step', 'sign', '--dry-run'])
        self.assertNotEqual(code, 0)

    def test_sign_receipt(self):
        ok = {'sign-1': {'terminal': {'status': 'succeeded'},
                         'stdout.log': 'x\n' + json.dumps({'status': 'permit_signed_and_dry_validated', 'permit_sha256': 'b' * 64})}}
        self.assertEqual(self.mod.sign_receipt(ok, 'sign-1'), 'b' * 64)
        with self.assertRaises(SystemExit):
            self.mod.sign_receipt({'sign-1': {'terminal': {'status': 'failed'}, 'stdout.log': ''}}, 'sign-1')
        with self.assertRaises(SystemExit):
            self.mod.sign_receipt({'sign-1': {'terminal': {'status': 'succeeded'}, 'stdout.log': '{"status": "x"}'}}, 'sign-1')


class DeploySpecTest(unittest.TestCase):
    def test_tree_and_ops_update_spec(self):
        mod = load('local/deploy_spec.py')
        with tempfile.TemporaryDirectory() as d:
            head = make_repo(Path(d) / 'r', {'a.txt': 'A\n', 'sub/b.txt': 'B\n'})
            code, out, _ = run_main(mod, ['tree', '--repo', str(Path(d) / 'r'), '--commit', head,
                                          '--branch', 'infra/01-ops', '--expect-old', 'f' * 40])
            self.assertEqual(code, 0)
            v = json.loads(out)
            self.assertEqual(v['commit'], head)
            self.assertEqual(set(v['tree']), {'a.txt', 'sub/b.txt'})
            import hashlib
            self.assertEqual(v['tree']['a.txt']['sha256'], hashlib.sha256(b'A\n').hexdigest())
            self.assertEqual(v['expect_old'], 'f' * 40)

    def test_branch_requires_expect_old(self):
        mod = load('local/deploy_spec.py')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            mod.parse_args(['tree', '--repo', '.', '--commit', 'HEAD', '--branch', 'b'])

    def test_override_and_data_validation(self):
        mod = load('local/deploy_spec.py')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(['g-overlay', '--repo', '.', '--base', 'a', '--target', 'b', '--override', 'x=short'])
            with self.assertRaises(SystemExit):
                mod.parse_args(['g-overlay', '--repo', '.', '--base', 'a', '--target', 'b', '--data-root', '/', '--data', '../x'])


class ServerToolsDryRunTest(unittest.TestCase):
    def test_permit_tool_dry_run_imports_nothing(self):
        mod = load('server/g_permit_tool.py')
        code, out, _ = run_main(mod, ['sign', '--prepared', '/p/m.json', '--prepared-sha', 'c' * 64,
                                      '--train-output', '/r/train', '--permit', '/x/permit.json', '--dry-run'])
        self.assertEqual(code, 0)
        v = json.loads(out)
        self.assertEqual(v['registration'], {'run_id': 'core-03-seed42', 'config_index': 3, 'seed': 42,
                                             'output': '/r/train', 'selection': None})
        self.assertEqual(v['pool_spec']['max_decode'], 153720)
        self.assertEqual(v['pool_spec']['max_resumes'], 0)
        self.assertEqual(v['pool_spec']['max_disk_bytes'], 20 * 1024 ** 3)
        self.assertNotIn('vkgr', sys.modules)
        self.assertNotIn('torch', sys.modules)

    def test_permit_tool_rejects_bad_sha_and_config(self):
        mod = load('server/g_permit_tool.py')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                mod.parse_args(['guard', '--prepared', 'p', '--prepared-sha', 'x', '--scratch', 's'])
            with self.assertRaises(SystemExit):
                mod.parse_args(['sign', '--prepared', 'p', '--prepared-sha', 'c' * 64, '--train-output', 't',
                                '--permit', 'x', '--config-index', '7'])

    def test_parity_tool_dry_run_own_pool(self):
        mod = load('server/g_parity_tool.py')
        code, out, _ = run_main(mod, ['--train-run-dir', '/r/run', '--out-dir', '/r/out', '--workers', '6', '--dry-run'])
        self.assertEqual(code, 0)
        spec = json.loads(out)['own_pool_spec_for_2520_queries']
        self.assertEqual(spec['max_decode'], 5040)
        self.assertEqual(spec['max_steps'], 1)
        self.assertEqual(spec['steps_by_phase'], {'parity': 1})
        self.assertNotIn('vkgr', sys.modules)

    def test_parity_train_receipt(self):
        mod = load('server/g_parity_tool.py')
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / 'logs').mkdir()
            (run / 'terminal.json').write_text(json.dumps({'status': 'failed'}))
            (run / 'logs/stdout.log').write_text('{"event": "x"}\n' + json.dumps(
                {'manifest_path': '/r/train/manifest.json', 'manifest_sha256': 'd' * 64, 'pool_head': 'e' * 64, 'pool_state': {}}) + '\n')
            with self.assertRaisesRegex(SystemExit, 'not succeeded'):
                mod.train_receipt(run)
            (run / 'terminal.json').write_text(json.dumps({'status': 'succeeded'}))
            self.assertEqual(mod.train_receipt(run)['manifest_sha256'], 'd' * 64)   # real receipt has no 'status' key
            (run / 'logs/stdout.log').write_text('{"event": "x"}\n')
            with self.assertRaisesRegex(SystemExit, 'manifest_path'):
                mod.train_receipt(run)

    def test_parity_tool_workers_range(self):
        mod = load('server/g_parity_tool.py')
        with contextlib.redirect_stderr(io.StringIO()):
            for w in ('1', '33'):
                with self.assertRaises(SystemExit):
                    mod.parse_args(['--train-run-dir', 'r', '--out-dir', 'o', '--workers', w])


if __name__ == '__main__':
    unittest.main()
