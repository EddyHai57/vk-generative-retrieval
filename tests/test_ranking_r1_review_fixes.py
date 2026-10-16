"""Review regression: no foreground publication; context-inclusive own-PID VRAM."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from tests.test_ranking_r1_ops import module


class ReviewFixes(unittest.TestCase):
    def test_spawn_failure_seals_plan_but_publishes_zero_and_alerts(self):
        m = module()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            root = Path(temp)
            req = {'ops': temp, 'deployment': temp, 'source_commit': 'a'*40, 'deadline_epoch': time.time()+3600,
                   'queue': str(root/'queue'), 'runs': str(root/'runs'), 'python': 'runtime-python'}
            plan = {'status': 'dry_run_ok', 'source_commit': 'a'*40, 'entries': [{'job': {'run_id': str(i)}} for i in range(3)]}
            with patch.object(m, 'dry_run', return_value=plan), patch.object(m, 'identity'), patch.object(m, 'exists', return_value=False), \
                 patch.object(m, 'publish') as publish, patch.object(m.subprocess, 'Popen', side_effect=OSError('spawn failed')):
                with self.assertRaises(OSError):
                    m.submit(req)
                publish.assert_not_called()
            self.assertTrue((root/'plan.json').exists())
            self.assertEqual(json.loads((root/'controller-ALERT.json').read_text())['controller_publications'], 0)

    def test_allocator_plus_measured_context_exceeds_allocator_alone(self):
        m = module()
        memory = {'pid': 123, 'context_process_mib': 600., 'context_reserved_mib': 100.,
                  'samples': [{'stage': 'train', 'own_process_mib': 1450., 'reserved_mib': 1000., 'max_reserved_mib': 1000., 'max_allocated_mib': 950.}]}
        self.assertEqual(m.measured_vram(memory), 1500.)
        proof = {'memory_measurement': memory, 'measured_mib': 1500., 'declared_vram_mib': 1792}
        self.assertEqual(m.validate_vram(proof), 1792)
        for key, value in [('measured_mib', 1000.), ('declared_vram_mib', 1280)]:
            with self.assertRaises(ValueError):
                m.validate_vram(dict(proof, **{key: value}))
        with self.assertRaises(ValueError):
            m.validate_vram({'measured_mib': 1000., 'declared_vram_mib': 1280})

    def test_own_gpu_measurement_ignores_other_pids_and_requires_own_pid(self):
        from types import SimpleNamespace
        m = module()
        with patch.object(m.os, 'getpid', return_value=42), patch.object(m.subprocess, 'run', return_value=SimpleNamespace(stdout='9, 9000\n42, 700\n')):
            self.assertEqual(m.own_gpu_mib(), 700.)
        with patch.object(m.os, 'getpid', return_value=42), patch.object(m.subprocess, 'run', return_value=SimpleNamespace(stdout='9, 9000\n')):
            with self.assertRaises(ValueError):
                m.own_gpu_mib()
