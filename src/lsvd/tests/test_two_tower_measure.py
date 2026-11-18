from pathlib import Path
import tempfile
import unittest

import torch

from lsvd.two_tower import TowerConfig
from lsvd.two_tower_data import synthetic
from lsvd.two_tower_tensors import TensorCatalog, TensorEventStore
from lsvd.two_tower_training import TensorTrainingSession
from lsvd.two_tower_measure import StepTimer


class TimingTests(unittest.TestCase):
    def test_cuda_summary_uses_host_clocks_for_cpu_work(self):
        from lsvd.two_tower_measure import phase_clock
        for phase in ['take_batch', 'history', 'mask']:
            self.assertEqual(phase_clock(phase, 'cuda', 'cpu'), 'host_seconds')
        self.assertEqual(phase_clock('forward', 'cuda', 'cpu'), 'cuda_seconds')
        self.assertEqual(phase_clock('mask', 'cuda', 'cuda'), 'cuda_seconds')
        self.assertEqual(phase_clock('forward', 'cpu', 'cpu'), 'host_seconds')

    def test_cpu_regions_accumulate_and_do_not_emit_fake_cuda_time(self):
        timer = StepTimer('cpu')
        for _ in range(2):
            with timer.region('features'):
                torch.arange(128).square()
        result = timer.finish()
        self.assertGreater(result['host_seconds']['features'], 0)
        self.assertEqual(result['cuda_seconds'], {})

    def test_profiled_training_has_identical_loss_and_parameter_update(self):
        torch.set_num_threads(1)
        config = TowerConfig(batch_size=8)
        catalog, store, _, _ = synthetic(config, user_count=4, item_count=900)
        catalog = TensorCatalog.from_catalog(catalog)
        store = TensorEventStore.from_store(store, catalog)
        with tempfile.TemporaryDirectory() as folder:
            first = TensorTrainingSession(catalog, store, config, 'mlp_causal',
                                            'cpu', Path(folder) / 'plain')
            expected = first.step()
            second = TensorTrainingSession(catalog, store, config, 'mlp_causal',
                                             'cpu', Path(folder) / 'profiled')
            actual = second.step(profile=True)
            self.assertEqual(actual['loss'], expected['loss'])
            for a, b in zip(first.model.parameters(), second.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name in ['take_batch', 'history', 'feature_lookup', 'h2d', 'mask',
                         'forward', 'loss', 'backward', 'optimizer']:
                self.assertIn(name, actual['timings']['host_seconds'])


if __name__ == '__main__':
    unittest.main()
