"""Synthetic tests for the G5 fusion-stage enqueue tool."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location('fenq', Path(__file__).with_name('g5_fusion_enqueue_20261018.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
P_DEP, F_DEP = '/workspace/vkgr-g5-p-export100-v2', '/workspace/vkgr-g5-fusion-v1'


def fake_root(tmp):
    root = Path(tmp)
    for sub in ('pending', 'claimed', 'rejected'):
        (root / 'queue' / sub).mkdir(parents=True)
    job = {'argv': ['/py', '-B', '-m', 'vkgr.retrieval_seqm1_run', 'verify', '--inputs', '/in.json', '--inputs-sha', 'a' * 64, '--config', 'configs/seqm1/p-seqm1-mean-s42.json',
                    '--config-sha', m.CONFIG_SHA, '--manifest', '/m.json', '--manifest-sha', 'b' * 64], 'config': 'x', 'cwd': '/old',
           'inputs': ['/in.json', '/prep1.json', '/prep2.json', '/hist.json', '/side.json', '/ctx.json', '/m.json'], 'env': ['OMP_NUM_THREADS=1', 'PYTHONPATH=src'], 'run_id': 'p-seqm1-mean-s42-verify',
           'vram_mib': 1024, 'wall_hours': 3.0, 'disk_gib': 1, 'kind': 'real', 'schema': 'vkgr-queue-job-v1'}
    (root / 'queue/claimed' / '0179-p-seqm1-mean-s42-verify.json').write_text(json.dumps(job))
    return root


def run(req):
    out = io.StringIO(); root = req.pop('root')
    with mock.patch('sys.stdin', io.StringIO(json.dumps(req))), mock.patch('sys.stdout', out):
        m.main(root=root)
    return json.loads(out.getvalue())


class FusionEnqueueTests(unittest.TestCase):
    def base(self, root, **kw):
        return {'root': str(root), 'action': 'dry', 'stage': 'qmeta-pf', 'p_deploy': P_DEP, 'fusion_deploy': F_DEP, 'vram_mib': 256, 'wall_hours': 1, **kw}

    def test_preflight_and_run_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            pf = run(self.base(root))['job']; real = run(self.base(root, stage='qmeta'))['job']
            self.assertEqual(pf['run_id'], 'g5-fusion-qmeta-pf'); self.assertIn('--check-only', pf['argv']); self.assertNotIn('--check-only', real['argv'])
            self.assertEqual(real['argv'][3], 'vkgr_g5.query_meta_p'); self.assertEqual(real['cwd'], P_DEP)
            self.assertIn(f'PYTHONPATH=src:{F_DEP}/src', real['env']); self.assertEqual(real['argv'][real['argv'].index('--line-root') + 1], P_DEP)
            self.assertEqual(real['argv'][real['argv'].index('--output') + 1], str(root / 'runs/g5-fusion-qmeta/out/p-query-meta.json'))
            self.assertEqual(real['argv'][real['argv'].index('--split') + 1], 'validation'); self.assertNotIn('/m.json', real['inputs'])
            self.assertEqual(real['inputs'][:6], ['/in.json', '/prep1.json', '/prep2.json', '/hist.json', '/side.json', '/ctx.json'])       # every other template input is kept, in order
            manifests = [str(root / f'runs/p-seqm1-mean-s{s}/out/manifest.json') for s in (42, 2024, 2025)]
            self.assertEqual(real['inputs'][6:], manifests)
            self.assertEqual([real['argv'][i + 1] for i, a in enumerate(real['argv']) if a == '--manifest'], manifests)
            self.assertEqual([real['argv'][i + 1] for i, a in enumerate(real['argv']) if a == '--manifest-sha'], [m.P_MANIFESTS[s] for s in (42, 2024, 2025)])
            self.assertEqual(real['argv'][real['argv'].index('--expect-queries') + 1:real['argv'].index('--expect-queries') + 2], ['2520'])
            self.assertEqual(real['argv'][real['argv'].index('--expect-oov') + 1], '623')
            self.assertEqual(list((root / 'queue/pending').iterdir()), [])

    def test_submit_writes_and_duplicates_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            self.assertEqual(len(run(self.base(root, action='submit'))['written']), 1)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(self.base(root, action='submit'))
            self.assertEqual(run(self.base(root, gen='v2'))['job']['run_id'], 'g5-fusion-qmeta-pf-v2')

    def test_preflight_in_queue_does_not_block_the_real_run_but_exact_duplicates_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            self.assertEqual(len(run(self.base(root, action='submit'))['written']), 1)                         # qmeta-pf queued
            self.assertEqual(len(run(self.base(root, stage='qmeta', action='submit'))['written']), 1)          # qmeta must still be possible
            (root / 'queue/rejected' / '0179-g5-fusion-qmeta.reason').write_text('x')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(self.base(root, stage='qmeta'))
            (root / 'queue/pending' / '.0180-g5-fusion-qmeta-v2.json.tmp').write_text('x')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(self.base(root, stage='qmeta', gen='v2'))

    def test_request_cannot_redirect_the_queue_root(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = fake_root(tmp); fake_root(other)
            out = io.StringIO()
            with mock.patch('sys.stdin', io.StringIO(json.dumps({'root': other, 'action': 'submit', 'stage': 'qmeta-pf', 'p_deploy': P_DEP, 'fusion_deploy': F_DEP, 'vram_mib': 256, 'wall_hours': 1}))), mock.patch('sys.stdout', out):
                m.main(root=root)
            self.assertEqual(len(list((root / 'queue/pending').iterdir())), 1); self.assertEqual(list((Path(other) / 'queue/pending').iterdir()), [])

    def test_bad_requests_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            for bad in ({'p_deploy': '/workspace/other'}, {'fusion_deploy': '/workspace/vkgr-g5-fusion-v1/../x'}, {'vram_mib': 100}, {'wall_hours': 9}, {'stage': 'train'}, {'gen': 'a/b'}):
                with self.assertRaises(ValueError):
                    run(self.base(root, **bad))
            path = next((root / 'queue/claimed').iterdir()); job = json.loads(path.read_text()); job['argv'][job['argv'].index('--config-sha') + 1] = '0' * 64; path.write_text(json.dumps(job))
            with self.assertRaisesRegex(ValueError, 'frozen table'):
                run(self.base(root))


if __name__ == '__main__':
    unittest.main()
