import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lsvd.two_tower import TowerConfig, TwoTower, corrected_loss
from lsvd.two_tower_data import synthetic, encode_batch
from lsvd.two_tower_tensors import (
    TensorCatalog,
    TensorEventStore,
    PositiveIndex,
    disk_shuffle,
)


def reference_mask(item_ids, positive_sets):
    ids = item_ids.tolist()
    result = torch.tensor([[item not in positives for item in ids]
                           for positives in positive_sets])
    result &= item_ids[None, :] != item_ids[:, None]
    result.fill_diagonal_(True)
    return result


class TensorBatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = TowerConfig(batch_size=8, dropout=0.1)
        self.catalog, self.store, self.data, _ = synthetic(
            self.config, user_count=8, item_count=900
        )

    def test_history_and_feature_fields_keep_padding_and_strict_prefix(self):
        catalog = TensorCatalog.from_catalog(self.catalog)
        store = TensorEventStore.from_store(self.store, catalog)
        selected = np.array([0, 1, 20, 41, 51, 80, 150, 250])
        before = self.store.train_batch(selected, self.config, 'cpu')
        after = store.train_batch(selected, self.config, 'cpu')
        for key, value in before[0].items():
            torch.testing.assert_close(after[0][key], value, rtol=0, atol=0,
                                       check_dtype=False)
        expected_ids = before[0]['items'].numpy().copy()
        expected_ids[~before[0]['valid'].numpy()] = self.catalog.ids[0]
        expected = self.catalog.features(expected_ids)
        for key, value in expected.items():
            torch.testing.assert_close(after[0]['item_features'][key], value,
                                       rtol=0, atol=0, check_dtype=False)
        for key, value in self.catalog.features(before[1]).items():
            torch.testing.assert_close(after[0]['target_features'][key], value,
                                       rtol=0, atol=0, check_dtype=False)
        torch.testing.assert_close(after[3], torch.as_tensor(before[3]).float(),
                                   rtol=0, atol=0)
        torch.testing.assert_close(after[4], torch.as_tensor(before[4]),
                                   rtol=0, atol=0)

    def test_sparse_item_ids_are_looked_up_as_ids_not_catalog_rows(self):
        self.catalog.ids = self.catalog.ids * 100 + 38
        self.catalog.columns['item_id'] = self.catalog.ids
        catalog = TensorCatalog.from_catalog(self.catalog)
        actual = catalog.features(torch.tensor([138, 838]))
        expected = self.catalog.features(np.array([138, 838]))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0,
                                       atol=0, check_dtype=False)

    def test_positive_search_keeps_all_place_positives_duplicates_and_diagonal(self):
        positives = [{7, 9, 13}, {7, 23}, {99}, {9, 23}]
        index = PositiveIndex.from_sets(positives)
        ids = torch.tensor([7, 9, 7, 23])
        torch.testing.assert_close(index.batch(torch.arange(4)).allowed(ids),
                                   reference_mask(ids, positives), rtol=0, atol=0)

    def test_every_positive_row_has_zero_loss_and_zero_gradient(self):
        users = torch.randn(4, 8, requires_grad=True)
        items = torch.randn(4, 8, requires_grad=True)
        ids = torch.tensor([1, 2, 1, 4])
        index = PositiveIndex.from_sets([{1, 2, 4}] * 4)
        loss, skipped = corrected_loss(users, items, ids,
                                       index.batch(torch.arange(4)),
                                       torch.full((4,), .25), torch.ones(4), .1)
        self.assertEqual(skipped, 4)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(users.grad, torch.zeros_like(users)))

    def test_precomputed_allowed_mask_has_identical_loss(self):
        torch.manual_seed(7)
        vectors = torch.randn(4, 8)
        ids = torch.tensor([7, 9, 7, 23])
        positives = PositiveIndex.from_sets([{7, 9}, {23}, {7}, {9}]).batch(torch.arange(4))
        mask = positives.allowed(ids)
        args = (torch.full((4,), .25), torch.tensor([1., .25, 1., .25]), .1)
        expected = corrected_loss(vectors, vectors, ids, positives, *args)
        actual = corrected_loss(vectors, vectors, ids, mask, *args)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        self.assertEqual(actual[1], expected[1])

    def test_mean_and_causal_one_update_keep_rng_and_parameters(self):
        catalog = TensorCatalog.from_catalog(self.catalog)
        store = TensorEventStore.from_store(self.store, catalog)
        selected = np.array([0, 1, 20, 41, 51, 80, 150, 250])
        for structure in ['mean', 'causal']:
            torch.manual_seed(42)
            old = TwoTower(self.catalog, self.config, structure)
            new = copy.deepcopy(old)
            optimizers = [torch.optim.AdamW(m.parameters(), lr=.001)
                          for m in [old, new]]
            batches = [self.store.train_batch(selected, self.config, 'cpu'),
                       store.train_batch(selected, self.config, 'cpu')]
            results = []
            for model, cat, batch_values, optimizer in zip(
                [old, new], [self.catalog, catalog], batches, optimizers
            ):
                batch, targets, positives, q, weights = batch_values
                torch.manual_seed(987)
                users = encode_batch(model, cat, batch, 'cpu')
                features = batch.get('target_features')
                vectors = model.item(cat.features(targets) if features is None
                                     else features)
                loss, _ = corrected_loss(users, vectors, torch.as_tensor(targets).long(),
                                          positives, torch.as_tensor(q).float(),
                                          torch.as_tensor(weights), .1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                results.append(float(loss))
            self.assertLessEqual(abs(results[0] - results[1]), 1e-6)
            for a, b in zip(old.parameters(), new.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=1e-6)

    def test_disk_shuffle_matches_numpy_permutation_and_resumes_cursor(self):
        with tempfile.TemporaryDirectory() as folder:
            expected_rng = np.random.default_rng(42)
            rng = np.random.default_rng(42)
            for epoch in range(2):
                expected = expected_rng.permutation(123)
                order = disk_shuffle(Path(folder) / f'epoch-{epoch}.npy', 123, rng)
                self.assertIsInstance(order, np.memmap)
                np.testing.assert_array_equal(order, expected)
                np.testing.assert_array_equal(order[37:45], expected[37:45])

    def test_disk_shuffle_passes_a_shared_base_array_to_numpy_fast_path(self):
        class ObservedGenerator:
            def __init__(self):
                self.generator = np.random.default_rng(42)
                self.input = None

            def shuffle(self, values):
                self.input = values
                self.generator.shuffle(values)

        with tempfile.TemporaryDirectory() as folder:
            observed = ObservedGenerator()
            expected_rng = np.random.default_rng(42)
            expected = expected_rng.permutation(100000)
            order = disk_shuffle(Path(folder) / 'order.npy', 100000, observed)
            np.testing.assert_array_equal(order, expected)
            self.assertTrue(np.shares_memory(order, observed.input))
            self.assertIs(type(observed.input), np.ndarray)
            self.assertEqual(observed.generator.bit_generator.state,
                             expected_rng.bit_generator.state)

    def test_exported_fp32_catalog_reuses_identical_vectors_in_retrieval(self):
        from lsvd.two_tower_retrieval import export_item_embeddings, retrieve
        torch.manual_seed(42)
        model = TwoTower(self.catalog, self.config)
        catalog = TensorCatalog.from_catalog(self.catalog)
        store = TensorEventStore.from_store(self.store, catalog)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'vectors'
            result = export_item_embeddings(model, catalog, 'cpu', output)
            vectors = np.load(output / 'vectors.npy', mmap_mode='r')
            self.assertEqual(vectors.dtype, np.dtype('float32'))
            self.assertEqual(result['items'], len(self.catalog.ids))
            old, _ = retrieve(model, catalog, store, self.data, 'cpu',
                              nlist=2, nprobe=2, exact_users=0)
            cached, details = retrieve(model, catalog, store, self.data, 'cpu',
                                        nlist=2, nprobe=2, exact_users=0,
                                        vector_file=output / 'vectors.npy')
            self.assertEqual(old, cached)
            self.assertIn('index_fit_seconds', details)
            self.assertIn('index_add_seconds', details)


if __name__ == '__main__':
    unittest.main()
