import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

MODULE = Path(__file__).parents[1] / 'analysis' / 'ur001_takeover_read.py'
SPEC = importlib.util.spec_from_file_location('ur001_takeover_read', MODULE)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)

RUN_IDS = [f'p-watch-positive-s{s}{suffix}' for s in ('42', '2024', '2025') for suffix in ('', '-verify')]
RUN_IDS += [f'g-half-users-r2-s{s}{suffix}' for s in ('42', '2024', '2025') for suffix in ('', '-verify')]


def request(tmp):
    tmp = Path(tmp).resolve()
    root, deployment = tmp / 'server-root', tmp / 'deployment'
    root.mkdir()
    deployment.mkdir()
    return {
        'action': 'inventory', 'root': str(root), 'deployment': str(deployment),
        'ops_dirs': [str(tmp / 'vkgr-p-watch-positive-train-ops-v6'), str(tmp / 'vkgr-p-watch-positive-verify-owner-v1'), str(tmp / 'vkgr-g-half-users-r2-ops-v2')],
        'guide_paths': list(reader.GUIDES),
        'module_paths': ['analysis/ur001_prepare.py', 'analysis/profile_anchor_steps.py'],
        'data_root': str(root / 'data' / 'ur0.01'), 'run_ids': RUN_IDS,
    }


class TakeoverInventoryTests(unittest.TestCase):
    def test_only_exact_original_twelve_runs_are_accepted(self):
        with tempfile.TemporaryDirectory() as raw:
            req = request(raw)
            self.assertTrue(hasattr(reader, 'inspect'), 'read-only takeover inventory entrypoint must exist')
            out = reader.inspect(req)
            self.assertEqual([r['run_id'] for r in out['runs']], RUN_IDS)
            req['run_ids'] = RUN_IDS + ['unrelated-run']
            with self.assertRaises(ValueError):
                reader.inspect(req)

    def test_reads_run_and_queue_receipts_without_reading_logs(self):
        with tempfile.TemporaryDirectory() as raw:
            req = request(raw)
            run_id = RUN_IDS[0]
            run_dir = Path(req['root']) / 'runs' / run_id
            run_dir.mkdir(parents=True)
            (run_dir / 'run.json').write_text(json.dumps({
                'schema': 'vkgr-run-v1', 'run_id': run_id, 'kind': 'real', 'created_at': 1000,
                'resources': {'wall_seconds': 7200}, 'seed': 42,
                'git': {'commit': 'a' * 40, 'dirty': False}, 'inputs': [{'sha256': 'b' * 64, 'bytes': 10}],
                'config': {'sha256': 'c' * 64}, 'argv': ['private-path'], 'ops_source': {'commit': 'd' * 40},
            }))
            (run_dir / 'terminal.json').write_text(json.dumps({'status': 'succeeded', 'wall_seconds': 5}))
            manifest = run_dir / 'out' / 'manifest.json'
            manifest.parent.mkdir()
            manifest_bytes = json.dumps({'schema': 'vkgr-run-manifest-v1', 'run_id': run_id, 'commit': 'a' * 40, 'config_sha256': 'c' * 64}).encode()
            manifest.write_bytes(manifest_bytes)
            (run_dir / 'owner.json').write_text(json.dumps({'supervisor': {'pid': 99999999, 'birth': '1'}, 'job': {'pid': 99999998, 'birth': '2'}, 'started_at': 1001, 'birth_source': 'proc-stat-22'}))
            run_record = json.loads((run_dir / 'run.json').read_text())
            run_record['argv'] = ['python', 'script.py', '--manifest', str(manifest), '--manifest-sha', __import__('hashlib').sha256(manifest_bytes).hexdigest()]
            (run_dir / 'run.json').write_text(json.dumps(run_record))
            (run_dir / 'logs').mkdir()
            (run_dir / 'logs' / 'stdout.log').write_text('DO_NOT_READ_SCIENCE_OUTPUT')
            pending = Path(req['root']) / 'queue' / 'pending'
            pending.mkdir(parents=True)
            (pending / f'0001-{run_id}.json').write_text(json.dumps({'schema': 'vkgr-queue-job-v1', 'run_id': run_id}))
            self.assertTrue(hasattr(reader, 'inspect'), 'read-only takeover inventory entrypoint must exist')
            out = reader.inspect(req)
            row = next(x for x in out['runs'] if x['run_id'] == run_id)
            self.assertEqual(row['run']['git_commit'], 'a' * 40)
            self.assertEqual(row['run']['input_sha256'], ['b' * 64])
            self.assertEqual(row['terminal']['status'], 'succeeded')
            self.assertEqual(row['manifest']['summary']['config_sha256'], 'c' * 64)
            self.assertEqual(row['manifest']['sha256'], __import__('hashlib').sha256(manifest_bytes).hexdigest())
            self.assertEqual(row['owner']['supervisor']['pid'], 99999999)
            self.assertEqual(row['owner']['job']['birth'], '2')
            self.assertEqual(row['queue']['pending'][0]['job']['run_id'], run_id)
            self.assertNotIn('stdout.log', json.dumps(out))
            self.assertNotIn('DO_NOT_READ_SCIENCE_OUTPUT', json.dumps(out))

    def test_reports_guides_modules_drivers_and_data_root_as_read_only_inventory(self):
        with tempfile.TemporaryDirectory() as raw:
            req = request(raw)
            base = Path(req['deployment'])
            for rel in req['module_paths'] + ['AGENTS.md']:
                path = base / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            p_driver, verify_driver, g_driver = map(Path, req['ops_dirs'])
            p_driver.mkdir(); verify_driver.mkdir(); g_driver.mkdir()
            (p_driver / 'owner.json').write_text(json.dumps({'deadline_seconds': 172800, 'spent_seconds': 25, 'owner_pid': 99999999}))
            (p_driver / 'driver-start-receipt-v1.json').write_text(json.dumps({'pid': 99999998, 'requested_deadline_seconds': 90000}))
            (verify_driver / 'summary-final.json').write_text(json.dumps({'original_deadline_epoch': 2000000000, 'original_spent_seconds': 80}))
            (g_driver / 'verify-resume-terminal.json').write_text(json.dumps({'status': 'succeeded', 'deadline_at': 2000000001}))
            (g_driver / 'driver-ready.json').write_text(json.dumps({'driver_pid': 99999997}))
            self.assertTrue(hasattr(reader, 'inspect'), 'read-only takeover inventory entrypoint must exist')
            out = reader.inspect(req)
            self.assertTrue(out['guides']['AGENTS.md']['exists'])
            self.assertFalse(out['guides']['/etc/vast-agents-guide.md']['exists'])
            self.assertTrue(all(x['exists'] for x in out['modules'].values()))
            owner = next(x for x in out['drivers'][0]['files'] if x['name'] == 'owner.json')
            self.assertEqual(owner['deadline_seconds'], 172800)
            self.assertIsNotNone(owner['owner'], 'top-level driver owner PID must be inventoried')
            self.assertEqual(owner['owner']['pid'], 99999999)
            self.assertIsNone(owner['owner']['identity_matches'])
            start = next(x for x in out['drivers'][0]['files'] if x['name'] == 'driver-start-receipt-v1.json')
            self.assertEqual(start['requested_deadline_seconds'], 90000)
            self.assertIsNone(start['owner']['identity_matches'])
            self.assertEqual(out['drivers'][1]['files'][0]['original_spent_seconds'], 80)
            terminal = next(x for x in out['drivers'][2]['files'] if x['name'] == 'verify-resume-terminal.json')
            self.assertEqual(terminal['status'], 'succeeded')
            ready = next(x for x in out['drivers'][2]['files'] if x['name'] == 'driver-ready.json')
            self.assertIsNotNone(ready['owner'], 'driver-ready top-level PID must be inventoried')
            self.assertEqual(ready['owner']['pid'], 99999997)
            self.assertIsNone(ready['owner']['identity_matches'])
            self.assertIn('data_root', out)

    def test_cli_rejects_unknown_command_and_emits_nonempty_valid_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            req = request(raw)
            invalid = dict(req, action='submit')
            proc = subprocess.run([sys.executable, '-B', str(MODULE)], input=json.dumps(invalid), text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(proc.stdout)
            proc = subprocess.run([sys.executable, '-B', str(MODULE)], input=json.dumps(req), text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(json.loads(proc.stdout)['runs'])


if __name__ == '__main__':
    unittest.main()
