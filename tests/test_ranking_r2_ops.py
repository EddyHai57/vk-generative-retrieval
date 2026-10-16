import copy
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'scripts/ranking_r2_ops.py'
    spec = importlib.util.spec_from_file_location('r1ops', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Ops(unittest.TestCase):
    def test_verify_keeps_r2_runner_and_batched_one(self):
        m = module()
        job = {'run_id': 'g-ranking-r2-s42', 'config': 'c.json', 'inputs': ['input'], 'wall_hours': 24., 'disk_gib': 4,
               'argv': ['python', '-B', '-m', m.ENTRY, 'train', '--config', 'c.json', '--config-sha', 'a'*64, '--output', 'oldout', '--scratch', 'oldscratch'],
               'env': ['VKGR_DECODE=batched', 'VKGR_DECODE_BATCH=1', 'VKGR_EVAL_WORKERS=1']}
        prior = copy.deepcopy(job)
        verify = m.derive_verify(job, 'b'*64, Path('/new-runs'))
        self.assertEqual(job, prior)
        self.assertEqual(m.flag(verify['argv'], '-m'), m.ENTRY)
        self.assertEqual(verify['argv'][verify['argv'].index('-m')+2], 'verify')
        self.assertNotIn('--config', verify['argv'])
        self.assertEqual(m.flag(verify['argv'], '--manifest-sha'), 'b'*64)
        self.assertIn('VKGR_DECODE_BATCH=1', verify['env'])

    def test_rejected_queue_never_republished(self):
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue, runs = root/'queue', root/'runs'
            for name in ('pending', 'claimed', 'rejected'):
                (queue/name).mkdir(parents=True)
            runs.mkdir()
            (queue/'rejected/000-run.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                m.publish(queue, runs, {'run_id': 'run'})
            self.assertEqual(list((queue/'pending').iterdir()), [])

    def test_deadline_and_vram_require_finite_evidence(self):
        m = module()
        self.assertTrue(m.valid_deadline(time.time()+3600))
        self.assertFalse(m.valid_deadline(time.time()+49*3600))
        self.assertFalse(m.valid_deadline(float('nan')))
        self.assertEqual(m.declared_vram(6000), 6912)
        for value in (-1, 0, float('nan'), True):
            with self.assertRaises(ValueError):
                m.declared_vram(value)


class ProofBinding(unittest.TestCase):
    def test_cuda_proof_must_match_preflight_data_shape_and_runtime_before_dryrun(self):
        from unittest.mock import patch
        m = module()
        req = {'preflight_receipt': 'pf', 'cuda_receipt': 'cu', 'source_commit': 'a'*40}
        pf = {'status': 'preflight_ok', 'source_commit': 'a'*40, 'data_binding_sha256': 'b'*64,
              'sid_manifest_sha256': 'c'*64, 'runtime': {'allow_tf32': False}, 'model_shape': {'vocab': 100}, 'entries': []}
        proof = {'status': 'cuda_engineering_verified', 'source_commit': 'a'*40, 'synthetic_grad_steps': 2, 'reload_decode_equal': True,
                 'measured_mib': 6000, 'declared_vram_mib': 6912, 'reference_unchanged': True, 'optimizer_actor_only': True, 'synthetic_sft_steps': 2, 'actor_parameter_count': 10, 'reference_parameter_count': 10, 'memory_measurement': {'pid': 123, 'context_process_mib': 1., 'context_reserved_mib': 1., 'samples': [{'stage': 'train', 'own_process_mib': 6000., 'reserved_mib': 6000., 'max_reserved_mib': 6000., 'max_allocated_mib': 5900.}]}, 'data_binding_sha256': 'd'*64,
                 'sid_manifest_sha256': 'c'*64, 'runtime': {'allow_tf32': False}, 'model_shape': {'vocab': 100}}
        with patch.object(m, 'identity'), patch.object(m, 'load', side_effect=[pf, proof]), patch.object(m, 'derive_job') as derive:
            with self.assertRaisesRegex(ValueError, 'preflight/CUDA binding'):
                m.dry_run(req)
            derive.assert_not_called()


class Derivation(unittest.TestCase):
    def test_new_training_preserves_input_pins_and_sets_fixed_data_and_decode_env(self):
        import json
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            old = json.loads((ROOT/'configs/gtok01b/g-gtok-a1-t3-b256d3-s42.json').read_text())
            (root/'old.json').write_text(json.dumps(old))
            (root/'oldout').mkdir()
            (root/'oldout/manifest.json').write_text('{}')
            template = {'run_id': 'old', 'seed': 42, 'config': str(root/'old.json'), 'inputs': ['data'], 'cwd': 'oldcwd',
                'pinned_manifest': 'oldpin', 'cpu_cores': 1, 'ram_mib': 18432, 'vram_mib': 6656, 'wall_hours': 24., 'disk_gib': 4, 'no_admission': False,
                'argv': ['python', '-B', '-m', 'vkgr.generative_g6_run', 'train', '--config', 'old.json', '--config-sha', '0'*64,
                         '--prepared', 'original-prepared', '--prepared-sha', 'a'*64, '--sid-manifest', 'original-sid', '--sid-sha', 'b'*64,
                         '--output', str(root/'oldout'), '--scratch', 'oldscratch'],
                'env': ['PYTHONPATH=/old-source/src', 'VKGR_DATA_ROOT=old', 'VKGR_DECODE=serial', 'VKGR_EVAL_WORKERS=2']}
            (root/'template.json').write_text(json.dumps(template))
            req = {'deployment': str(ROOT), 'runs': str(root/'new-runs'), 'data_root': str(root), 'python': 'runtime-python',
                   'pinned_manifest': str(root/'newpin.json'), 'templates': {'42': str(root/'template.json')}}
            job = m.derive_job(req, 42)
            self.assertEqual(m.flag(job['argv'], '--prepared'), 'original-prepared')
            self.assertEqual(m.flag(job['argv'], '--sid-sha'), 'b'*64)
            self.assertEqual(m.flag(job['argv'], '-m'), m.ENTRY)
            self.assertEqual(job['inputs'], ['data', str(root/'oldout/manifest.json')])
            self.assertEqual(job['env'], ['PYTHONPATH='+str(ROOT/'src'), 'VKGR_PINNED_MANIFEST='+req['pinned_manifest'], 'VKGR_DECODE=batched', 'VKGR_DECODE_BATCH=1',
                                         'VKGR_EVAL_WORKERS=1', 'VKGR_DATA_ROOT='+str(root)])
            self.assertEqual(job['run_id'], 'g-ranking-r2-s42')


class DetachedDriver(unittest.TestCase):
    def test_driver_launcher_detaches_but_does_not_accept_missing_ready_ack(self):
        import json
        from unittest.mock import patch
        from types import SimpleNamespace
        m = module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            req = {'ops': temp, 'python': 'runtime-python', 'source_commit': 'a'*40}
            (root/'request.json').write_text(json.dumps(req))
            (root/'plan.json').write_text(json.dumps({'source_commit': 'a'*40, 'deadline_epoch': time.time()+3600}))
            with patch.object(m, 'identity'), patch.object(m, 'READY_TIMEOUT_SECONDS', .01), \
                 patch.object(m.subprocess, 'Popen', return_value=SimpleNamespace(pid=123, poll=lambda: 0)) as popen:
                with self.assertRaises(TimeoutError):
                    m.start_driver(req, root/'request.json')
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(popen.call_args.kwargs['stdin'], m.subprocess.DEVNULL)
            self.assertFalse((root/'publish-release.json').exists())
            self.assertEqual(json.loads((root/'controller-ALERT.json').read_text())['controller_publications'], 0)
