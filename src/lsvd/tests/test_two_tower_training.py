from pathlib import Path
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from lsvd.two_tower import TowerConfig, TwoTower
from lsvd.two_tower_data import synthetic
from lsvd.two_tower_tensors import TensorCatalog, TensorEventStore
from lsvd.two_tower_training import TensorTrainingSession, fit_tensor


class TensorTrainingTests(unittest.TestCase):
    def test_evaluation_retains_external_checkpoint_basename(self):
        from lsvd.two_tower_run import evaluate_model
        config = TowerConfig()
        catalog, store, data, _ = synthetic(config, user_count=4, item_count=900)
        model = TwoTower(catalog, config)
        metric = dict(ruler='main', place_scope='0/1', seen_scope='all',
                       breakdown='all', k=200, macro_capped=.5)
        result = SimpleNamespace(metrics=[metric])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'new-evaluation'
            output.mkdir()
            with patch('lsvd.two_tower_run.retrieve', return_value=({}, {})), \
                 patch('lsvd.two_tower_run.diagnostic_evaluations', return_value=(result, {})):
                summary = evaluate_model(model, catalog, store, data, 'mlp_mean', 'cpu',
                    output, [dict(samples=32, training_seconds=1)], root / 'old/run.pt')
            self.assertEqual(summary['best_checkpoint'], 'run.pt')

    def test_real_training_step_resume_crosses_epoch_with_identical_losses(self):
        torch.set_num_threads(1)
        config = TowerConfig(seed=42, batch_size=32, id_min_count=5)
        catalog, store, _, _ = synthetic(config, user_count=4, item_count=900)
        catalog = TensorCatalog.from_catalog(catalog)
        store = TensorEventStore.from_store(store, catalog)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            continuous = TensorTrainingSession(catalog, store, config, 'mlp_causal',
                                                 'cpu', root / 'continuous')
            expected = [continuous.step()['loss'] for _ in range(9)]
            interrupted = TensorTrainingSession(catalog, store, config, 'mlp_causal',
                                                  'cpu', root / 'interrupted')
            actual = [interrupted.step()['loss'] for _ in range(4)]
            saved = interrupted.save()
            payload = torch.load(saved, weights_only=True)
            self.assertEqual(payload['epoch'], 1)
            self.assertEqual(payload['training_state']['epoch_index'], 0)
            resumed = TensorTrainingSession(catalog, store, config, 'mlp_causal',
                                              'cpu', root / 'resumed', resume=saved)
            actual.extend(resumed.step()['loss'] for _ in range(5))
            self.assertEqual(actual, expected)
            self.assertEqual(resumed.global_step, 9)
            self.assertEqual(resumed.epoch, 1)
            for a, b in zip(continuous.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(resumed.last_batch_users, continuous.last_batch_users)

    def test_resume_keeps_the_full_best_checkpoint_and_its_actual_weights(self):
        torch.set_num_threads(1)
        config = TowerConfig(seed=42, batch_size=32, epochs=2)
        catalog, store, _, _ = synthetic(config, user_count=4, item_count=900)
        catalog = TensorCatalog.from_catalog(catalog)
        store = TensorEventStore.from_store(store, catalog)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = TensorTrainingSession(catalog, store, config, 'mlp_mean',
                                              'cpu', root / 'first', max_samples=64)
            session.step(); session.step()
            original_best = session.register_epoch(.5, dict(epoch=1))
            expected = torch.load(original_best, weights_only=True)
            session.step()
            latest = session.save()
            resumed = TensorTrainingSession(catalog, store, config, 'mlp_mean',
                                              'cpu', root / 'resumed', max_samples=64,
                                              resume=latest)
            self.assertIsNotNone(resumed.manager.best)
            actual = torch.load(resumed.manager.best, weights_only=True)
            self.assertEqual(actual['step'], 2)
            self.assertEqual(actual['optimizer']['param_groups'],
                             expected['optimizer']['param_groups'])
            for key in expected['model']:
                torch.testing.assert_close(actual['model'][key], expected['model'][key],
                                           rtol=0, atol=0)
            continued = resumed.save()
            again = TensorTrainingSession(catalog, store, config, 'mlp_mean',
                                            'cpu', root / 'again', max_samples=64,
                                            resume=continued)
            self.assertIsNotNone(again.manager.best)
            self.assertEqual(torch.load(again.manager.best, weights_only=True)['step'], 2)

    def test_resume_at_epoch_end_finishes_pending_early_evaluation_before_training(self):
        torch.set_num_threads(1)
        config = TowerConfig(seed=42, batch_size=32, epochs=2)
        catalog, store, data, _ = synthetic(config, user_count=32, item_count=900)
        catalog = TensorCatalog.from_catalog(catalog)
        store = TensorEventStore.from_store(store, catalog)
        metric = dict(ruler='main', place_scope='0/1', seen_scope='all',
                       breakdown='all', k=200, macro_capped=.5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            interrupted = TensorTrainingSession(catalog, store, config, 'mlp_mean',
                                                  'cpu', root / 'interrupted', max_samples=64)
            interrupted.step(); interrupted.step()
            checkpoint = interrupted.save()
            with patch('lsvd.two_tower_retrieval.retrieve', return_value=({}, {})), \
                 patch('lsvd.evaluation.evaluate', return_value=SimpleNamespace(metrics=[metric])), \
                 patch('lsvd.two_tower_run.evaluate_model', return_value={'completed': True}):
                fit_tensor(catalog, store, data, config, 'mlp_mean', 'cpu', root / 'resumed',
                            64, 1, 0, resume=checkpoint, checkpoint_every=1)
            registry = json.loads((root / 'resumed/checkpoints/checkpoints.json').read_text())
            state = torch.load(root / 'resumed/checkpoints' / registry['latest'][-1],
                                weights_only=True)
            self.assertEqual(state['step'], 4)
            self.assertEqual([row['epoch'] for row in state['training_state']['progress']['epochs']],
                             [1, 2])


if __name__ == '__main__':
    unittest.main()
