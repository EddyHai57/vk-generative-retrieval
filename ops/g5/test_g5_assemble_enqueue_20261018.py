"""Synthetic tests for the G5 assembly enqueue tool."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location('aenq', Path(__file__).with_name('g5_assemble_enqueue_20261018.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
G_DEP, F_DEP = '/workspace/vkgr-coll-a1-g-v1', '/workspace/vkgr-g5-fusion-v1'


def fake_root(tmp):
    root = Path(tmp)
    for sub in ('pending', 'claimed', 'rejected'):
        (root / 'queue' / sub).mkdir(parents=True)
    for seed, sha in m.MANIFEST_SHA.items():
        argv = ['/py', '-B', '-m', 'vkgr.coll_sid_a1_run', 'verify']
        for flag in m.FLAGS:
            argv += [flag, sha if flag == '--manifest-sha' else f'/x/{flag.strip("-")}']
        argv += ['--output', '/o', '--scratch', '/s']
        job = {'argv': argv, 'cwd': G_DEP, 'env': ['OMP_NUM_THREADS=1', f'PYTHONPATH={G_DEP}/src', 'VKGR_DATA_ROOT=/d'], 'inputs': ['/a', '/b'], 'run_id': f'g-coll-a1-c1-s{seed}-verify',
               'vram_mib': 6656, 'ram_mib': 18432, 'wall_hours': 2.0, 'pinned_manifest': '/pin.json', 'kind': 'real', 'schema': 'vkgr-queue-job-v1'}
        (root / 'queue/claimed' / f'0179-g-coll-a1-c1-s{seed}-verify.json').write_text(json.dumps(job))
    return root


def run(req, root):
    out = io.StringIO()
    with mock.patch('sys.stdin', io.StringIO(json.dumps(req))), mock.patch('sys.stdout', out):
        m.main(root=root)
    return json.loads(out.getvalue())


class AssembleEnqueueTests(unittest.TestCase):
    def req(self, **kw):
        return {'action': 'dry', 'stage': 'asm-pf', 'seeds': [42, 2024, 2025], 'g_deploy': G_DEP, 'fusion_deploy': F_DEP, 'vram_mib': 256, 'wall_hours': 1, **kw}

    def test_jobs_clone_the_verify_envelope_and_swap_the_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp); got = run(self.req(), root); jobs = got['jobs']
            self.assertEqual([j['run_id'] for j in jobs], ['g5-fusion-asm-pf-s42', 'g5-fusion-asm-pf-s2024', 'g5-fusion-asm-pf-s2025'])
            j = jobs[0]; a = j['argv']
            self.assertEqual(a[:5], ['/py', '-B', '-m', 'vkgr_g5.assemble_g', '--line-root']); self.assertEqual(a[5], G_DEP); self.assertIn('--check-only', a)
            for flag in m.FLAGS:
                self.assertEqual(a[a.index(flag) + 1], f'/x/{flag.strip("-")}' if flag != '--manifest-sha' else m.MANIFEST_SHA[42])
            self.assertEqual(a[a.index('--split') + 1], 'validation')
            self.assertEqual(a[a.index('--output') + 1], str(root / 'runs/g5-fusion-asm-pf-s42/out/g-assembly.json')); self.assertEqual(a[a.index('--scratch') + 1], str(root / 'runs/g5-fusion-asm-pf-s42/scratch'))
            self.assertEqual(j['env'], ['OMP_NUM_THREADS=1', f'PYTHONPATH={G_DEP}/src:{F_DEP}/src', 'VKGR_DATA_ROOT=/d'])
            self.assertEqual((j['cwd'], j['pinned_manifest'], j['ram_mib'], j['inputs'], j['vram_mib']), (G_DEP, '/pin.json', 18432, ['/a', '/b'], 256))
            self.assertEqual(list((root / 'queue/pending').iterdir()), [])

    def test_real_stage_has_no_check_only_and_exact_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            self.assertEqual(len(run(self.req(action='submit'), root)['written']), 3)
            real = run(self.req(stage='asm', seeds=[42]), root)['jobs'][0]               # asm-pf-s42 in queue must not block asm-s42
            self.assertNotIn('--check-only', real['argv']); self.assertEqual(real['run_id'], 'g5-fusion-asm-s42')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                run(self.req(action='submit'), root)

    def test_every_queue_form_blocks_and_neighbours_do_not(self):
        forms = [('queue/pending', '0001-g5-fusion-asm-s42.json'), ('queue/pending', '.0001-g5-fusion-asm-s42.json.tmp'), ('queue/claimed', '0002-g5-fusion-asm-s42.json'),
                 ('queue/rejected', '0003-g5-fusion-asm-s42.json'), ('queue/rejected', '0003-g5-fusion-asm-s42.reason'), ('queue/rejected', '0003-g5-fusion-asm-s42.json.reason'), ('runs/g5-fusion-asm-s42', None)]
        for sub, name in forms:
            with tempfile.TemporaryDirectory() as tmp:
                root = fake_root(tmp); (root / sub).mkdir(parents=True, exist_ok=True)
                if name:
                    (root / sub / name).write_text('x')
                with self.assertRaisesRegex(ValueError, 'already exists', msg=f'{sub}/{name}'):
                    run(self.req(stage='asm', seeds=[42]), root)
        for neighbour in ('0001-g5-fusion-asm-pf-s42.json', '0001-g5-fusion-asm-s420.json', '0001-g5-fusion-asm-s2024.json', '0001-xg5-fusion-asm-s42.json'):
            with tempfile.TemporaryDirectory() as tmp:
                root = fake_root(tmp); (root / 'queue/rejected' / neighbour).write_text('x')
                self.assertEqual(run(self.req(stage='asm', seeds=[42]), root)['jobs'][0]['run_id'], 'g5-fusion-asm-s42')

    def test_deployment_regex_is_enforced_independently_of_the_template_and_gen_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            for path in root.joinpath('queue/claimed').glob('*-verify.json'):                       # template and request agree on an out-of-pattern deployment
                job = json.loads(path.read_text()); job['cwd'] = '/workspace/elsewhere'; job['env'] = ['PYTHONPATH=/workspace/elsewhere/src']; path.write_text(json.dumps(job))
            with self.assertRaisesRegex(ValueError, 'bad deployment path'):
                run(self.req(g_deploy='/workspace/elsewhere'), root)
            fake_root(tmp + '/x') if False else None
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            self.assertEqual(run(self.req(seeds=[42], gen='abcdefgh'), root)['jobs'][0]['run_id'], 'g5-fusion-asm-pf-s42-abcdefgh')
            with self.assertRaisesRegex(ValueError, 'generation tag'):
                run(self.req(seeds=[42], gen='abcdefghi'), root)

    def test_identity_and_request_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_root(tmp)
            for bad in ({'g_deploy': '/workspace/other'}, {'fusion_deploy': '/workspace/vkgr-g5-fusion-v1/../x'}, {'seeds': [7]}, {'vram_mib': 100}, {'wall_hours': 9}, {'stage': 'train'}, {'gen': 'a/b'}):
                with self.assertRaises(ValueError):
                    run(self.req(**bad), root)
            path = next((root / 'queue/claimed').glob('*s42-verify.json')); job = json.loads(path.read_text()); job['argv'][job['argv'].index('--manifest-sha') + 1] = '0' * 64; path.write_text(json.dumps(job))
            with self.assertRaisesRegex(ValueError, 'frozen table'):
                run(self.req(seeds=[42]), root)
            job['argv'][job['argv'].index('--manifest-sha') + 1] = m.MANIFEST_SHA[42]; job['cwd'] = '/workspace/vkgr-coll-a1-g-v9'; path.write_text(json.dumps(job))
            with self.assertRaisesRegex(ValueError, 'frozen table'):
                run(self.req(seeds=[42]), root)
            job['cwd'] = G_DEP; job['env'] = ['PYTHONPATH=/elsewhere/src']; path.write_text(json.dumps(job))
            with self.assertRaisesRegex(ValueError, 'PYTHONPATH'):
                run(self.req(seeds=[42]), root)


if __name__ == '__main__':
    unittest.main()
