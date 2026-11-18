import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lsvd.two_tower_checkpoint import CheckpointManager


def model_and_optimizer():
    model = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Dropout(.3),
                                torch.nn.Linear(7, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)
    return model, optimizer, scheduler


def update(model, optimizer, scheduler):
    values = torch.randn(8, 5) + float(np.random.rand()) + random.random()
    loss = model(values).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


class CheckpointTests(unittest.TestCase):
    def test_resume_reproduces_every_loss_optimizer_and_rng(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model, optimizer, scheduler = model_and_optimizer()
        initial = copy.deepcopy(model.state_dict())
        initial_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        expected = [update(model, optimizer, scheduler) for _ in range(7)]
        expected_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.8)
        random.setstate(initial_rng[0]); np.random.set_state(initial_rng[1])
        torch.set_rng_state(initial_rng[2])
        actual = [update(model, optimizer, scheduler) for _ in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            manager = CheckpointManager(Path(folder) / 'checkpoints')
            path = manager.save(model, optimizer, scheduler=scheduler,
                                 epoch=1, step=3, shuffle=dict(seed=42, cursor=24),
                                 training_state=dict(best=.1, bad=0))
            model, optimizer, scheduler = model_and_optimizer()
            saved = manager.restore(path, model, optimizer, scheduler=scheduler)
            self.assertEqual(saved['shuffle'], dict(seed=42, cursor=24))
            self.assertEqual(saved['training_state'], dict(best=.1, bad=0))
            actual.extend(update(model, optimizer, scheduler) for _ in range(4))
            self.assertEqual(expected, actual)
            for key, value in expected_state.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_retains_latest_two_and_best_without_removing_foreign_files(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = CheckpointManager(Path(folder) / 'checkpoints')
            foreign = manager.root / 'existing.txt'
            foreign.write_text('preserve')
            model, optimizer, scheduler = model_and_optimizer()
            best = None
            paths = []
            for step in range(5):
                paths.append(manager.save(model, optimizer, scheduler=scheduler,
                                           epoch=1, step=step, shuffle={},
                                           is_best=step == 0))
                if step == 0:
                    best = paths[-1]
            self.assertEqual(set(manager.root.glob('*.pt')), {best, paths[-2], paths[-1]})
            self.assertEqual(foreign.read_text(), 'preserve')
            reopened = CheckpointManager(manager.root, create=False)
            self.assertEqual(reopened.latest, paths[-1])
            self.assertEqual(reopened.best, best)

    def test_failed_write_keeps_previous_published_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = CheckpointManager(Path(folder) / 'checkpoints')
            model, optimizer, _ = model_and_optimizer()
            old = manager.save(model, optimizer, epoch=0, step=1, shuffle={})
            with patch('lsvd.two_tower_checkpoint.torch.save', side_effect=OSError('full disk')):
                with self.assertRaises(OSError):
                    manager.save(model, optimizer, epoch=0, step=2, shuffle={})
            reopened = CheckpointManager(manager.root, create=False)
            self.assertEqual(reopened.latest, old)
            self.assertEqual(reopened.restore(old, model, optimizer)['step'], 1)

    def test_registry_failure_can_retry_after_reopening_without_overwriting_orphan(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = CheckpointManager(Path(folder) / 'checkpoints')
            model, optimizer, _ = model_and_optimizer()
            old = manager.save(model, optimizer, epoch=0, step=1, shuffle={})
            with patch('lsvd.two_tower_checkpoint._atomic_json',
                       side_effect=OSError('registry write failed')):
                with self.assertRaises(OSError):
                    manager.save(model, optimizer, epoch=0, step=2, shuffle={})
            orphan = next(path for path in manager.root.glob('*.pt') if path != old)
            before = orphan.read_bytes()
            reopened = CheckpointManager(manager.root, create=False)
            self.assertEqual(reopened.latest, old)
            retried = reopened.save(model, optimizer, epoch=0, step=2, shuffle={})
            self.assertNotEqual(retried, orphan)
            self.assertEqual(orphan.read_bytes(), before)
            self.assertEqual(reopened.restore(retried, model, optimizer)['step'], 2)


if __name__ == '__main__':
    unittest.main()
