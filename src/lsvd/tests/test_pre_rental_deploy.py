import hashlib
import json
from pathlib import Path
import tempfile
import unittest

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
        self.assertIn('a'*40, ' '.join(' '.join(x['argv']) for x in git))
        rsync = deployment_commands('rsync', Path('.'), connection,
                                    files_from=Path('files.txt'))
        self.assertTrue(any('--partial' in x['argv'] for x in rsync))
        self.assertTrue(any(any(s.startswith('--files-from=') for s in x['argv'])
                            for x in rsync))
        with self.assertRaises(ValueError):
            deployment_commands('git', Path('.'), connection, commit='branch-name')

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
