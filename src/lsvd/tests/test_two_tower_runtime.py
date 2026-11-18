from pathlib import Path
import tempfile
import unittest

import torch

from lsvd.two_tower_runtime import available_resources, precision_mode, process_metrics


class ResourceTests(unittest.TestCase):
    def test_nested_cgroup_quota_cpuset_and_affinity_intersect(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            child = root / 'job'
            child.mkdir()
            (root / 'cpu.max').write_text('600000 100000')
            (root / 'cpuset.cpus.effective').write_text('0-6')
            (root / 'memory.max').write_text('8589934592')
            (child / 'cpu.max').write_text('350000 100000')
            (child / 'cpuset.cpus.effective').write_text('2-5')
            (child / 'memory.max').write_text('4294967296')
            result = available_resources(cgroup_root=root, cgroup_path=child,
                                         affinity={0, 1, 2, 3, 4}, host_cpus=64)
            self.assertEqual(result['cpu_threads'], 3)
            self.assertEqual(result['cpu_quota_cores'], 3.5)
            self.assertEqual(result['cpu_allowed_ids'], [2, 3, 4])
            self.assertEqual(result['memory_limit_bytes'], 4294967296)

    def test_unlimited_quota_uses_affinity_and_never_returns_zero_threads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'cpu.max').write_text('max 100000')
            result = available_resources(cgroup_root=root, cgroup_path=root,
                                         affinity={1, 4}, host_cpus=128)
            self.assertEqual(result['cpu_threads'], 2)
            self.assertIsNone(result['cpu_quota_cores'])

    def test_cpu_fp32_context_preserves_output_and_cuda_modes_reject_cpu(self):
        x = torch.randn(4, 8)
        expected = x @ x.T
        with precision_mode('cpu', 'fp32'):
            torch.testing.assert_close(x @ x.T, expected, rtol=0, atol=0)
        for mode in ['bf16', 'tf32']:
            with self.assertRaises(ValueError):
                with precision_mode('cpu', mode):
                    pass
        metrics = process_metrics('cpu')
        self.assertGreater(metrics['rss_peak_bytes'], 0)
        self.assertIsNone(metrics['torch_peak_allocated_bytes'])


if __name__ == '__main__':
    unittest.main()
