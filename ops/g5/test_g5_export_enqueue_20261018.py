"""Synthetic tests for the G5 export enqueue tool."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location('enq', Path(__file__).with_name('g5_export_enqueue_20261018.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fake_root(tmp, seeds=(42, 2024, 2025)):
    root = Path(tmp)
    for sub in ('pending', 'claimed', 'rejected'):
        (root / 'queue' / sub).mkdir(parents=True)
    for seed in seeds:
        cfg, man = m.FROZEN[seed]
        job = {'argv': ['/py', '-B', '-m', 'vkgr.retrieval_seqm1_run', 'verify', '--inputs', '/in.json', '--inputs-sha', 'a' * 64,
                        '--config', f'configs/seqm1/p-seqm1-mean-s{seed}.json', '--config-sha', cfg,
                        '--manifest', str(root / f'runs/p-seqm1-mean-s{seed}/out/manifest.json'), '--manifest-sha', man],
               'config': 'x', 'cwd': '/old', 'inputs': ['/in.json'], 'run_id': f'p-seqm1-mean-s{seed}-verify', 'vram_mib': 1024, 'wall_hours': 3.0,
               'disk_gib': 1, 'env': ['PYTHONPATH=src'], 'kind': 'real', 'schema': 'vkgr-queue-job-v1'}
        (root / 'queue/claimed' / f'0179-p-seqm1-mean-s{seed}-verify.json').write_text(json.dumps(job))
    return root


def run(req):
    out = io.StringIO()
    with mock.patch('sys.stdin', io.StringIO(json.dumps(req))), mock.patch('sys.stdout', out):
        m.main()
    return json.loads(out.getvalue())


class EnqueueTests(unittest.TestCase):
    def test_pf_job_shape_and_dry_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            got = run({'root': str(root), 'action': 'dry', 'stage': 'pf', 'seeds': [42], 'vram_mib': 1024, 'wall_hours': 1})
            job = got['jobs'][0]
            self.assertEqual(job['run_id'], 'g5-p-top100-pf-s42'); self.assertEqual(job['cwd'], m.DEPLOY)
            self.assertEqual(job['argv'][3:5], [m.MODULE, 'preflight']); self.assertIn('p-seqm1-mean-s42', job['argv'])
            self.assertNotIn('--output', job['argv']); self.assertEqual(list((root / 'queue/pending').iterdir()), [])

    def test_export_and_verify_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            e = run({'root': str(root), 'action': 'dry', 'stage': 'export', 'seeds': [2024], 'vram_mib': 1024, 'wall_hours': 1})['jobs'][0]
            self.assertEqual(e['argv'][e['argv'].index('--output') + 1], str(root / 'runs/g5-p-top100-s2024/out'))
            v = run({'root': str(root), 'action': 'dry', 'stage': 'verify', 'seeds': [2024], 'vram_mib': 1024, 'wall_hours': 1, 'export_sha': {'2024': 'b' * 64}})['jobs'][0]
            self.assertEqual(v['run_id'], 'g5-p-top100-s2024-verify'); self.assertEqual(v['inputs'][-1], str(root / 'runs/g5-p-top100-s2024/out/export-manifest.json'))
            self.assertIn('--export-manifest-sha', v['argv'])
            with self.assertRaises(ValueError):
                run({'root': str(root), 'action': 'dry', 'stage': 'verify', 'seeds': [42], 'vram_mib': 1024, 'wall_hours': 1})

    def test_submit_writes_pending_and_refuses_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            got = run({'root': str(root), 'action': 'submit', 'stage': 'pf', 'seeds': [42, 2025], 'vram_mib': 1024, 'wall_hours': 1})
            self.assertEqual(len(got['written']), 2); self.assertEqual(len(list((root / 'queue/pending').iterdir())), 2)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run({'root': str(root), 'action': 'submit', 'stage': 'pf', 'seeds': [42], 'vram_mib': 1024, 'wall_hours': 1})
            (root / 'runs/g5-p-top100-s2024').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run({'root': str(root), 'action': 'dry', 'stage': 'export', 'seeds': [2024], 'vram_mib': 1024, 'wall_hours': 1})

    def test_identity_drift_and_bad_resources_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            path = next((root / 'queue/claimed').glob('*s42-verify.json')); job = json.loads(path.read_text())
            job['argv'][job['argv'].index('--manifest-sha') + 1] = '0' * 64; path.write_text(json.dumps(job))
            with self.assertRaisesRegex(ValueError, 'frozen table'):
                run({'root': str(root), 'action': 'dry', 'stage': 'pf', 'seeds': [42], 'vram_mib': 1024, 'wall_hours': 1})
            for bad in ({'vram_mib': 100}, {'wall_hours': 9}, {'seeds': [7]}, {'stage': 'train'}):
                req = {'root': str(root), 'action': 'dry', 'stage': 'pf', 'seeds': [2024], 'vram_mib': 1024, 'wall_hours': 1, **bad}
                with self.assertRaises(ValueError):
                    run(req)


if __name__ == '__main__':
    unittest.main()
