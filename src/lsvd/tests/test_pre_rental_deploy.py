import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from lsvd.pre_rental_deploy import (
    code_manifest, upload_manifest, deployment_commands, stop_request,
)


class DeploymentTests(unittest.TestCase):
    def test_source_manifest_keeps_only_code_and_generator_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, value in [('src/lsvd/two_tower.py', 'x=1\n'),
                                ('src/lsvd/private.local.json', '{"private":true}'),
                                ('src/lsvd/__pycache__/a.pyc', 'binary'),
                                ('src/vkgr/tiger_arch.py', 'pass\n')]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
            result = code_manifest(root)
            names = {row['path'] for row in result['files']}
            self.assertEqual(names, {'src/lsvd/two_tower.py', 'src/vkgr/tiger_arch.py'})
            self.assertEqual(result['files'][0]['sha256'],
                             hashlib.sha256(b'x=1\n').hexdigest())

    def test_upload_list_binds_each_tensor_and_rejects_missing_hash_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'a.npy').write_bytes(b'example-array')
            row = dict(file='a.npy', bytes=13,
                       sha256=hashlib.sha256(b'example-array').hexdigest())
            (root / 'manifest.json').write_text(json.dumps(dict(arrays={'a': row},
                                                               train_weeks=[0, 24],
                                                               validation_week=25)))
            record = upload_manifest(root)
            self.assertEqual(len(record['files']), 2)
            self.assertEqual(record['total_bytes'],
                             sum(p.stat().st_size for p in root.iterdir()))
            (root / 'a.npy').write_bytes(b'different')
            with self.assertRaises(ValueError):
                upload_manifest(root)

    def test_deployment_modes_pin_commit_or_exact_rsync_file_list(self):
        connection = dict(target='synthetic-target', destination='workspace',
                           repository='https://example.invalid/project.git')
        git = deployment_commands('git', Path('.'), connection, commit='a'*40,
                                  files_from=Path('files.txt'))
        command = git[0]['argv'][-1]
        branch = 'claude/vk-generative-retrieval-takeover-f62c2a'
        self.assertIn(f'--single-branch --branch {branch}', command)
        self.assertIn('https://github.com/EddyHai57/vk-generative-retrieval.git', command)
        self.assertIn('merge-base --is-ancestor ' + 'a'*40 + f' refs/remotes/origin/{branch}', command)
        self.assertIn('checkout --detach ' + 'a'*40, command)
        self.assertNotIn(connection['repository'], command)
        rsync = deployment_commands('rsync', Path('.'), connection,
                                    files_from=Path('files.txt'))
        self.assertTrue(any('--partial' in x['argv'] for x in rsync))
        self.assertIn('--compress', rsync[-1]['argv'])
        self.assertNotIn('--append-verify', rsync[-1]['argv'])
        self.assertNotIn('--protect-args', rsync[-1]['argv'])
        self.assertTrue(any(any(s.startswith('--files-from=') for s in x['argv'])
                            for x in rsync))
        with self.assertRaises(ValueError):
            deployment_commands('git', Path('.'), connection, commit='branch-name')

    def test_builtin_rsync_and_verified_result_pull(self):
        from lsvd.pre_rental_deploy import detect_rsync, pull_results

        self.assertIn(detect_rsync()['implementation'], ('openrsync', 'rsync'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'results'
            source.mkdir()
            (source / 'best.pt').write_bytes(b'checkpoint')
            digest = hashlib.sha256(b'checkpoint').hexdigest()
            (source / 'SHA256SUMS').write_text(f'{digest}  best.pt\n')
            result = pull_results(str(source), root / 'copied', ssh=None)
            self.assertEqual(result['files'][0]['bytes'], 10)
            self.assertEqual((root / 'copied/best.pt').read_bytes(), b'checkpoint')
            (source / 'best.pt').write_bytes(b'corrupted!')
            with self.assertRaisesRegex(ValueError, 'sha256'):
                pull_results(str(source), root / 'invalid', ssh=None)
            self.assertFalse((root / 'invalid/pull-receipt.json').exists())

    def test_result_verification_rejects_missing_artifact(self):
        from lsvd.pre_rental_deploy import verify_results

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'SHA256SUMS').write_text(f'{"a"*64}  best.pt\n')
            with self.assertRaisesRegex(ValueError, 'missing'):
                verify_results(path)
            with self.assertRaisesRegex(ValueError, 'missing required artifacts'):
                verify_results(path, formal=True)

    def test_verifier_can_start_before_ml_dependencies_are_installed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'sample').write_bytes(b'checked')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(files=[dict(path='sample', bytes=7,
                sha256=hashlib.sha256(b'checked').hexdigest())])))
            result = subprocess.run([sys.executable, '-S', '-m', 'lsvd.pre_rental_deploy',
                'verify', '--root', str(root), '--manifest', str(manifest)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['files'], 1)

    def test_optional_stop_request_uses_only_instance_environment(self):
        request = stop_request(dict(CONTAINER_ID='123',
                                    CONTAINER_API_KEY='synthetic-instance-credential'))
        self.assertEqual(request.method, 'PUT')
        self.assertEqual(json.loads(request.data), {'state': 'stopped'})
        self.assertTrue(request.full_url.endswith('/instances/123/'))
        self.assertEqual(request.get_header('Authorization'),
                         'Bearer synthetic-instance-credential')
        with self.assertRaises(ValueError):
            stop_request({})


if __name__ == '__main__':
    unittest.main()
