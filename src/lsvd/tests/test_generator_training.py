import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


class GeneratorSessionTests(unittest.TestCase):
    def test_resume_replays_same_losses_and_model(self):
        from lsvd.generator_model import GeneratorConfig
        from lsvd.generator_training import GeneratorSession
        from lsvd.two_tower_tensors import PositiveIndex

        class Fixture:
            main_positions = np.arange(12, dtype=np.int32)

            def training_batch(self, positions):
                count = len(positions)
                keys = torch.as_tensor(positions.copy()).long()
                codes = torch.stack((keys % 3, keys // 3, keys % 2), dim=1)
                history = dict(codes=codes[:, None, :], valid=torch.ones(count, 1).bool(),
                               cold=torch.ones(count, 1, 4), feedback=torch.zeros(count, 1, 9),
                               ratio=torch.zeros(count, 1), place=torch.ones(count, 1).long(),
                               gap=torch.ones(count, 1).long(), position=torch.zeros(count, 1).long())
                return (history, codes, torch.arange(count * 4).reshape(count, 4).float() + 1,
                        keys, PositiveIndex.from_sets([set() for _ in range(count)]).batch(
                            torch.arange(count)), torch.full((count,), 1 / 12), torch.ones(count))

        torch.set_num_threads(1)
        cfg = GeneratorConfig(size=4, vector_dim=4, d_model=16, heads=2, head_dim=8,
                              encoder_layers=1, decoder_layers=1, ff=32, batch_size=3)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = (Fixture(), cfg, 'cpu')
            complete = GeneratorSession(*args, root / 'continuous', identity={'fixture': 1})
            expected = [complete.step()['loss'] for _ in range(6)]
            first = GeneratorSession(*args, root / 'first', identity={'fixture': 1})
            actual = [first.step()['loss'] for _ in range(3)]
            checkpoint = first.save()
            second = GeneratorSession(*args, root / 'second', identity={'fixture': 1},
                                      resume=checkpoint)
            actual.extend(second.step()['loss'] for _ in range(3))
            self.assertEqual(expected, actual)
            self.assertEqual(second.global_step, 6)
            for before, after in zip(complete.model.parameters(), second.model.parameters()):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'identity'):
                GeneratorSession(*args, root / 'wrong', identity={'fixture': 2}, resume=checkpoint)

    def test_evaluation_entry_preserves_sdk_metric(self):
        from lsvd.evaluation import evaluate
        from lsvd.generator_eval import evaluate_predictions
        from lsvd.two_tower import TowerConfig
        from lsvd.two_tower_data import synthetic
        catalog, _, data, _ = synthetic(TowerConfig(), user_count=8, item_count=900)
        predictions = {int(user): data.targets['item_id'][start:end]
                       for user, start, end in zip(data.user_ids, data.targets['offsets'],
                                                  data.targets['offsets'][1:])}
        expected = evaluate(data, predictions, model_name='full_sid', replicates=2)
        actual = evaluate_predictions(data, {'full_sid': predictions}, catalog.groups,
                                      replicates=2)['full_sid']
        self.assertEqual(actual['all'], expected.metrics)
        reduced = evaluate_predictions(data, {'full_sid': predictions}, catalog.groups,
                                       replicates=2, k=37)['full_sid']
        self.assertEqual({row['k'] for row in reduced['all']}, {37})

    def test_mixed_manifest_retains_existing_tower_entry(self):
        from lsvd.pre_rental_queue import load_plan
        tower = dict(id='mean', structure='mlp_mean', epochs=1,
                     seed=42, id_min_count=20, patience=3, precision='fp32')
        generator = dict(id='generator', kind='generator', seed=42, epochs=1, patience=3,
                         precision='fp32', history_mode='event', shared_head=True,
                         sid_bundle='sid', sid_sha256='a'*64, item_vectors='vectors',
                         item_vectors_sha256='b'*64)
        with tempfile.TemporaryDirectory() as folder:
            plan = Path(folder) / 'plan.json'
            plan.write_text(json.dumps({'runs': [tower, generator]}))
            self.assertEqual(load_plan(plan, 'cpu'), [tower, generator])
