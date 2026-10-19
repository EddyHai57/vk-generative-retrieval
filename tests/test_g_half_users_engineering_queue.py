import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('g_half_users_engineering_queue',
                                               ROOT / 'scripts/g_half_users_engineering_queue.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class GHalfUsersEngineeringQueueTests(unittest.TestCase):
    def request(self, root, run_id='g-half-users-r2-preflight-v2'):
        root = Path(root).resolve()
        ops = root / 'ops'; ops.mkdir(exist_ok=True)
        for state in ('pending', 'claimed', 'rejected'):
            (root / 'queue' / state).mkdir(parents=True, exist_ok=True)
        (root / 'runs').mkdir(exist_ok=True)
        private = ops / 'half-request.json'; private.write_text('{}')
        stage = queue.STAGES[run_id]
        job = {'run_id': run_id, 'schema': 'vkgr-queue-job-v1', 'kind': 'ops',
               'cpu_cores': 1, 'vram_mib': 6656, 'no_admission': False,
               'cwd': str(ROOT), 'config': str(private),
               'env': [],
               'argv': [sys.executable, '-B', str(ROOT / 'scripts/g_half_users_ops.py'), stage,
                        '--request', str(private), '--request-sha', queue.sha(private)]}
        return {'action': 'inventory', 'root': str(root), 'ops': str(ops),
                'deployment': str(ROOT), 'scheduler_code': str(root/'scheduler'),
                'python': sys.executable, 'job': job}

    def test_only_fixed_preflight_and_cuda_ids_map_to_new_ops_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            req = self.request(Path(directory))
            queue.validate(req)
            req['job']['run_id'] = 'g-half-users-r2-cuda-v1'
            req['job']['argv'][3] = 'cuda-preview'
            req['job']['env'] = []
            queue.validate(req)
            req['job']['env'] = ['VKGR_RUN_ID=g-half-users-r2-cuda-v1']
            with self.assertRaisesRegex(ValueError, 'injected by the launcher'):
                queue.validate(req)
            req['job']['env'] = []
            for run_id, action in (('g-half-users-preflight-v1', 'preflight'),
                                   ('g-half-users-r2-preflight-v1', 'preflight'),
                                   ('g-half-users-cuda-v1', 'cuda-preview'),
                                   ('g-half-users-r2-s42', 'train'), ('other', 'preflight')):
                req['job']['run_id'] = run_id
                req['job']['argv'][3] = action
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    queue.validate(req)

    def test_pinned_scheduler_launcher_accepts_preflight_cuda_train_and_verify_envelopes(self):
        # Provenance: infra/04-sched-forecast:ops/vkgr_ops/{launcher,common,admission}.py.
        # Extract the versioned scheduler package into isolated temp storage and call its real static validator.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'vkgr_ops'; package.mkdir()
            for name in ('launcher.py', 'common.py', 'admission.py'):
                source = subprocess.run(['git', 'show', f'infra/04-sched-forecast:ops/vkgr_ops/{name}'],
                                        check=True, capture_output=True).stdout
                (package / name).write_bytes(source)
            sys.path.insert(0, str(root))
            try:
                for name in tuple(sys.modules):
                    if name == 'vkgr_ops' or name.startswith('vkgr_ops.'):
                        del sys.modules[name]
                from vkgr_ops import launcher
                user_pin = 'a' * 64
                cases = (
                    ('g-half-users-r2-preflight-v2', 'ops', []),
                    ('g-half-users-r2-cuda-v1', 'ops', []),
                    ('g-half-users-r2-s42', 'real', ['VKGR_EXPECTED_SELECTED_USERS_SHA256=' + user_pin,
                                                     'VKGR_DRIVER_DEADLINE_EPOCH=1790000000']),
                    ('g-half-users-r2-s42-verify', 'real', ['VKGR_EXPECTED_SELECTED_USERS_SHA256=' + user_pin,
                                                            'VKGR_DRIVER_DEADLINE_EPOCH=1790000000']),
                )
                for run_id, kind, env in cases:
                    with self.subTest(run_id=run_id):
                        self.assertFalse(any(item.partition('=')[0] in launcher.INJECTED_ENV for item in env))
                        spec = {'run_id': run_id, 'kind': kind, 'stop_mode': 'signal',
                                'cpu_cores': 1, 'ram_mib': 0, 'vram_mib': 6656, 'disk_gib': 1,
                                'wall_hours': 2.0, 'seed': 42 if kind == 'real' else None,
                                'env': env, 'argv': ['python', '-B', 'job.py'], 'cwd': str(root)}
                        self.assertEqual(launcher._validate_static(spec),
                                         ({'VKGR_EXPECTED_SELECTED_USERS_SHA256': user_pin,
                                           'VKGR_DRIVER_DEADLINE_EPOCH': '1790000000'}
                                          if kind == 'real' else {}))
            finally:
                sys.path.remove(str(root))
                for name in tuple(sys.modules):
                    if name == 'vkgr_ops' or name.startswith('vkgr_ops.'):
                        del sys.modules[name]

    def test_wrong_stage_python_request_sha_and_resources_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            req = self.request(Path(directory))
            queue.validate(req)
            for field, value in (('argv', [sys.executable, '-B', 'other.py', 'preflight', '--request',
                                            req['job']['config'], '--request-sha', queue.sha(req['job']['config'])]),
                                 ('vram_mib', 39169), ('cpu_cores', 2)):
                bad = json.loads(json.dumps(req))
                if field == 'argv':
                    bad['job'][field] = value
                else:
                    bad['job'][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    queue.validate(bad)

    def test_dry_is_metadata_only_and_submit_requires_matching_dry_job_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            req = self.request(root)
            fake = SimpleNamespace(launcher=SimpleNamespace(prepare=lambda job: {'metadata': 'prepared'}),
                                   scheduler=SimpleNamespace(submit=lambda root, job: '/queue/claim.json'))
            with mock.patch.dict(sys.modules, {'vkgr_ops': fake}):
                proof = queue.dry(req)
                self.assertEqual(proof['status'], 'dry_ok')
                self.assertFalse((root/'runs'/req['job']['run_id']).exists())
                req.update(action='submit', dry_job_sha256='0'*64)
                with self.assertRaisesRegex(ValueError, 'dry job hash'):
                    queue.submit(req)
                self.assertFalse((root/'ops'/('intent-'+req['job']['run_id']+'.json')).exists())
                req['dry_job_sha256'] = proof['job_sha256']
                submitted = queue.submit(req)
            self.assertEqual(submitted['status'], 'written_unconfirmed')
            self.assertEqual(queue.inventory(req)['intent_exists'], True)

    def test_status_reads_only_bounded_metadata_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            req = self.request(root)
            run = root/'runs'/req['job']['run_id']; (run/'logs').mkdir(parents=True)
            (run/'terminal.json').write_text('{"status":"succeeded"}')
            result = queue.status(req)
            self.assertEqual(result['terminal.json']['payload'], {'status': 'succeeded'})
            self.assertEqual(result['terminal.json']['sha256'], queue.sha(run/'terminal.json'))
            (run/'logs/stdout.log').write_text('x' * (queue.MAX_RECEIPT_BYTES+1))
            with self.assertRaises(ValueError):
                queue.status(req)


if __name__ == '__main__':
    unittest.main()
