"""Synthetic tests for the G5 fuse enqueue tool."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location('fqenq', Path(__file__).with_name('g5_fuse_enqueue_20261018.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
F_DEP = '/workspace/vkgr-g5-fusion-v1'
P_EXPORT_LITERALS = {42: 'be5eb969d47c453105c538f7f57a87ee3eff64b28666bf321fe196c677ce576b', 2024: '531a6c28eea6952f583fe6286648b8e0375755696ad2329802d245c719e3b989',
                     2025: '38ad413b07ed8b82342749d3f58be8438e99a2388253c06709b838f160413439'}      # independent of the module under test (experiment_log, G5 top-100 export)
G_SHAS = {'42': '1' * 64, '2024': '2' * 64, '2025': '3' * 64}


def fake_root(tmp):
    root = Path(tmp)
    for sub in ('pending', 'claimed', 'rejected'):
        (root / 'queue' / sub).mkdir(parents=True)
    job = {'argv': ['/py', '-B', '-m', 'x'], 'config': 'c', 'cwd': '/old', 'inputs': ['/i'], 'env': ['OMP_NUM_THREADS=1', 'PYTHONPATH=src', 'MKL_NUM_THREADS=1'], 'run_id': 'p-seqm1-mean-s42-verify',
           'vram_mib': 1024, 'wall_hours': 3.0, 'disk_gib': 1, 'pinned_files': ['docs/p3c_validation.md=' + 'a' * 64], 'kind': 'real', 'schema': 'vkgr-queue-job-v1'}
    (root / 'queue/claimed' / '0179-p-seqm1-mean-s42-verify.json').write_text(json.dumps(job))
    return root


def run(req, root):
    out = io.StringIO()
    with mock.patch('sys.stdin', io.StringIO(json.dumps(req))), mock.patch('sys.stdout', out):
        m.main(root=root)
    return json.loads(out.getvalue())


class FuseEnqueueTests(unittest.TestCase):
    def req(self, **kw):
        return {'action': 'dry', 'stage': 'fuse-pf', 'fusion_deploy': F_DEP, 'vram_mib': 256, 'wall_hours': 1, 'g_assembly_sha': G_SHAS, 'meta_sha': '4' * 64, **kw}

    def test_command_inputs_and_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp); job = run(self.req(), root)['job']; a = job['argv']
            self.assertEqual(a[:4], ['/py', '-B', '-m', 'vkgr_g5.fuse']); self.assertIn('--check-only', a)
            for seed in (42, 2024, 2025):
                self.assertIn(f'{seed}={root}/runs/g5-p-top100-v3-s{seed}/out/export-manifest.json@{m.P_EXPORT_MANIFEST_SHA[seed]}', a)
                self.assertIn(f'{seed}={root}/runs/g5-fusion-asm-s{seed}/out/g-assembly.json@{G_SHAS[str(seed)]}', a)
            self.assertIn(f'{root}/runs/g5-fusion-qmeta/out/p-query-meta.json@' + '4' * 64, a)
            self.assertEqual(a[a.index('--output-dir') + 1], str(root / 'runs/g5-fusion-fuse-pf/out')); self.assertEqual(a[a.index('--split') + 1], 'validation')
            self.assertEqual((job['cwd'], job['pinned_files'], job['config'], job['vram_mib']), (F_DEP, [], None, 256))
            self.assertEqual(job['env'], ['OMP_NUM_THREADS=1', f'PYTHONPATH={F_DEP}/src', 'MKL_NUM_THREADS=1'])
            self.assertEqual(len(job['inputs']), 10); self.assertTrue(all(i.startswith(str(root / 'runs')) for i in job['inputs']))
            real = run(self.req(stage='fuse'), root)['job']; self.assertNotIn('--check-only', real['argv']); self.assertEqual(real['run_id'], 'g5-fusion-fuse')

    def test_exact_names_and_forms(self):
        for sub, name in (('queue/pending', '0001-g5-fusion-fuse.json'), ('queue/claimed', '0002-g5-fusion-fuse.json'), ('queue/rejected', '0003-g5-fusion-fuse.reason'),
                          ('queue/rejected', '0003-g5-fusion-fuse.json.reason'), ('queue/pending', '.0004-g5-fusion-fuse.json.tmp'), ('runs/g5-fusion-fuse', None)):
            with tempfile.TemporaryDirectory() as tmp:
                root = fake_root(tmp); (root / sub).mkdir(parents=True, exist_ok=True)
                if name:
                    (root / sub / name).write_text('x')
                with self.assertRaisesRegex(ValueError, 'already exists', msg=f'{sub}/{name}'):
                    run(self.req(stage='fuse'), root)
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp); (root / 'queue/rejected' / '0001-g5-fusion-fuse-pf.reason').write_text('x')
            self.assertEqual(run(self.req(stage='fuse'), root)['job']['run_id'], 'g5-fusion-fuse')

    def test_bad_requests_and_root_redirect(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = fake_root(tmp); fake_root(other)
            for bad in ({'fusion_deploy': '/workspace/other'}, {'vram_mib': 100}, {'vram_mib': 6657}, {'action': 'delete'}, {'wall_hours': 9}, {'stage': 'train'}, {'gen': 'a/b'}, {'gen': 'abcdefghi'},
                        {'g_assembly_sha': {'42': '1' * 64}}, {'g_assembly_sha': {**G_SHAS, '42': 'zz'}}, {'meta_sha': 'short'}, {'meta_sha': None}):
                with self.assertRaises(ValueError, msg=str(bad)):
                    run(self.req(**bad), root)
            out = io.StringIO()
            with mock.patch('sys.stdin', io.StringIO(json.dumps({**self.req(action='submit'), 'root': other}))), mock.patch('sys.stdout', out):
                m.main(root=root)
            self.assertEqual(len(list((root / 'queue/pending').iterdir())), 1); self.assertEqual(list((Path(other) / 'queue/pending').iterdir()), [])
            self.assertEqual(run(self.req(gen='abcdefgh'), root)['job']['run_id'], 'g5-fusion-fuse-pf-abcdefgh')


    def test_unique_template_tmp_rename_sorted_sha_and_bounded_stdin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp); (root / 'queue/claimed' / '0180-p-seqm1-mean-s42-verify.json').write_text((root / 'queue/claimed' / '0179-p-seqm1-mean-s42-verify.json').read_text())
            with self.assertRaisesRegex(ValueError, 'unique claimed template'):
                run(self.req(), root)
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp); calls = []; real = m.os.replace
            def spy(src, dst):
                calls.append((Path(src).name, Path(dst).name)); return real(src, dst)
            with mock.patch.object(m.os, 'replace', spy):
                got = run(self.req(action='submit'), root)
            self.assertEqual(len(calls), 1); self.assertEqual(calls[0][0], '.' + calls[0][1] + '.tmp'); self.assertEqual([p.name for p in (root / 'queue/pending').iterdir()], [calls[0][1]])
            import hashlib
            self.assertEqual(got['job_sha256'], hashlib.sha256(json.dumps(got['job'], sort_keys=True).encode()).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run(self.req(pad='x' * 70000), fake_root(tmp))

    def test_literal_pins_boundaries_and_envelope_extras(self):
        self.assertEqual(m.P_EXPORT_MANIFEST_SHA, P_EXPORT_LITERALS)
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            self.assertEqual(run(self.req(vram_mib=256), root)['job']['vram_mib'], 256); self.assertEqual(run(self.req(vram_mib=6656), root)['job']['vram_mib'], 6656)
            self.assertEqual(run(self.req(wall_hours=4), root)['job']['wall_hours'], 4.0)
            for bad in ({'vram_mib': 255}, {'wall_hours': 4.5}, {'wall_hours': 5}, {'wall_hours': 0}):
                with self.assertRaises(ValueError, msg=str(bad)):
                    run(self.req(**bad), root)
            job = run(self.req(), root)['job']; self.assertEqual(job['disk_gib'], 1)
            path = next((root / 'queue/claimed').glob('*-verify.json')); t = json.loads(path.read_text()); t['env'] = ['OMP_NUM_THREADS=1']; path.write_text(json.dumps(t))
            with self.assertRaisesRegex(ValueError, 'PYTHONPATH'):
                run(self.req(), root)


if __name__ == '__main__':
    unittest.main()
