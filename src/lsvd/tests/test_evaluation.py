"""Synthetic contract tests for the shared, fixed-query evaluator."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lsvd.evaluation import (
    EvaluationData, EvaluationResult, evaluate, filter_ranked_items, paired_bootstrap,
)


def fixture(root):
    arrays = Path(root) / 'dataset' / 'arrays'
    arrays.mkdir(parents=True)
    np.savez(arrays / 'queries.npz', user_id=[10, 20, 30], history_row=[0, 1, -1],
             history_count=[205, 1, 0], last_item_id=[204, 300, 4294967295], last_author_id=[0, 0, 4294967295])
    np.savez(arrays / 'history.npz', user_id=[10, 20], offsets=[0, 205, 206])
    np.save(arrays / 'history_items.npy', np.array(list(range(205)) + [300], dtype=np.uint32))
    np.savez(arrays / 'targets.npz', offsets=[0, 4, 6, 6], item_id=[301, 302, 303, 304, 301, 305],
             place=[0, 1, 2, 0, 1, 2], main_positive=[1, 1, 1, 0, 1, 0],
             secondary_positive=[1, 0, 1, 1, 1, 0], like_positive=[0, 1, 0, 0, 0, 1],
             train_seen=[1, 0, 1, 1, 1, 0], duration=[60, 61, 121, 19, 53, 180],
             duration_band=[0, 1, 2, 0, 0, 2], duration_band_sensitive=[2, 2, 2, 0, 1, 2],
             popularity_tier=[1, 0, 3, 2, 1, 0])
    np.savez(arrays / 'items.npz', item_id=np.arange(400), author_id=np.zeros(400),
             duration=np.ones(400) * 10, train_count=np.ones(400), popularity_tier=np.ones(400))
    return EvaluationData.load(root)


def metric(result, ruler='main', place_scope='0/1', seen_scope='all', breakdown='all', band='all', k=2):
    return next(row for row in result.metrics if all(row[key] == value for key, value in
        dict(ruler=ruler, place_scope=place_scope, seen_scope=seen_scope,
             breakdown=breakdown, band=band, k=k).items()))


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = fixture(self.temp.name)

    def run_eval(self, predictions=None, **kwargs):
        if predictions is None:
            predictions = {10: [0, 301, 301, 302, 303, 304], 20: [301, 305]}
        return evaluate(self.data, predictions, model_name='fixture', ks=(1, 2, 3),
                        replicates=24, seed=71, **kwargs)

    def test_filter_full_history_before_cutoff_and_stable_duplicates(self):
        self.assertEqual(filter_ranked_items([0, 204, 301, 301, 302, 303], range(205), range(400), 2), [301, 302])
        self.assertEqual(filter_ranked_items(iter([0, 1, 301]), range(205), range(400), 1000), [301])
        self.assertEqual(filter_ranked_items([np.iinfo(np.uint32).max, 302], [], range(400)), [302])
        self.assertEqual(filter_ranked_items([], [], range(400)), [])
        with self.assertRaises(ValueError):
            filter_ranked_items([999], [], range(400))

    def test_capped_macro_micro_and_zero_target_users(self):
        result = self.run_eval()
        row = metric(result, k=1)
        self.assertEqual(row['n_users'], 2)
        self.assertEqual(row['n_queries'], 3)
        self.assertEqual(row['n_targets'], 3)
        self.assertEqual(row['hits'], 2)
        self.assertEqual(row['macro_capped'], 1.)
        self.assertAlmostEqual(row['micro_recall'], 2 / 3)
        np.testing.assert_array_equal(result.user_ids, [10, 20, 30])
        self.assertEqual(len(result.slice_specs), 495)
        self.assertEqual(result.counts.shape, (495, 3))
        self.assertEqual(result.hits.shape, (495, 3, 3))

    def test_short_lists_do_not_shrink_denominator_and_missing_users_empty(self):
        result = self.run_eval({10: [301]})
        self.assertEqual(metric(result)['macro_capped'], .25)
        self.assertEqual(metric(result)['micro_recall'], 1 / 3)
        empty = self.run_eval({})
        self.assertEqual(metric(empty)['macro_capped'], 0.)

    def test_all_stored_labels_scopes_bands_and_popularity(self):
        result = self.run_eval()
        self.assertEqual(metric(result, place_scope='all', k=3)['n_targets'], 4)
        self.assertEqual(metric(result, place_scope='2')['n_targets'], 1)
        self.assertEqual(metric(result, place_scope='0')['n_targets'], 1)
        self.assertEqual(metric(result, place_scope='1')['n_targets'], 2)
        self.assertEqual(metric(result, seen_scope='unseen')['n_targets'], 1)
        self.assertEqual(metric(result, seen_scope='seen')['n_targets'], 2)
        self.assertEqual(metric(result, ruler='secondary')['n_targets'], 3)
        self.assertEqual(metric(result, ruler='like', place_scope='all')['n_targets'], 2)
        for breakdown, band, expected in [('duration', '≤60秒', 2), ('duration', '61–120秒', 1),
                ('duration', '>120秒', 0), ('duration_sensitive', '≤19秒', 0),
                ('duration_sensitive', '20–53秒', 1), ('duration_sensitive', '>53秒', 2),
                ('popularity', '未见', 1), ('popularity', '长尾', 2),
                ('popularity', '中部', 0), ('popularity', '头部', 0)]:
            row = metric(result, breakdown=breakdown, band=band)
            self.assertEqual(row['n_targets'], expected, (breakdown, band))
            if not expected:
                self.assertIsNone(row['macro_capped'])
                self.assertIsNone(row['macro_ci95'])
                self.assertEqual(row['valid_draws'], 0)

    def test_matrix_padding_and_unknown_or_oversized_predictions(self):
        pad = np.iinfo(np.uint32).max
        matrix = np.array([[301, pad], [301, 305], [pad, pad]], dtype=np.uint32)
        self.assertEqual(metric(self.run_eval(matrix))['macro_capped'], .75)
        for predictions in ({10: [999]}, {10: [301] * 1001}, np.zeros((3, 1001), dtype=np.uint32),
                            np.zeros((2, 10), dtype=np.uint32), {10: [301.5]}):
            with self.assertRaises(ValueError):
                self.run_eval(predictions)
        with self.assertRaises(ValueError):
            self.run_eval({10: [301] * 1000 + [999]})

    def test_poisson_bootstrap_matches_manual_shared_weights(self):
        result = self.run_eval({10: [301]})
        weights = np.random.default_rng(71).poisson(1, size=(24, 3))
        scores = np.array([.5, 0., 0.])
        denominators = weights[:, :2].sum(axis=1)
        valid = denominators > 0
        draws = (weights @ scores)[valid] / denominators[valid]
        row = metric(result)
        np.testing.assert_allclose(row['macro_ci95'], np.quantile(draws, [.025, .975]))
        self.assertEqual(row['valid_draws'], int(valid.sum()))

    def test_paired_identical_zero_and_mismatches_rejected(self):
        result = self.run_eval()
        paired = paired_bootstrap(result, result, replicates=24, seed=71)
        for row in paired.metrics:
            if row['n_users']:
                self.assertEqual(row['macro_difference'], 0.)
                self.assertEqual(row['macro_ci95'], [0., 0.])
        for attribute, replacement in [('user_ids', np.array([11, 20, 30])), ('ks', (1, 2)),
                                        ('counts', result.counts + 1), ('slice_specs', ())]:
            other = self.run_eval()
            setattr(other, attribute, replacement)
            with self.assertRaises(ValueError):
                paired_bootstrap(result, other, replicates=4)

    def test_paired_difference_is_shared_user_difference(self):
        first = self.run_eval({10: [301]})
        second = self.run_eval({20: [301]})
        result = paired_bootstrap(first, second, replicates=24, seed=71)
        self.assertEqual(metric(result)['macro_difference'], -.25)
        weights = np.random.default_rng(71).poisson(1, size=(24, 3))
        denominator = weights[:, :2].sum(axis=1)
        valid = denominator > 0
        draws = (weights @ np.array([.5, -1., 0.]))[valid] / denominator[valid]
        np.testing.assert_allclose(metric(result)['macro_ci95'], np.quantile(draws, [.025, .975]))

    def test_persistence_is_reloadable_and_exclusive(self):
        output = Path(self.temp.name) / 'evaluation'
        result = self.run_eval(output_dir=output)
        loaded = EvaluationResult.load(output)
        self.assertEqual(result.metrics, loaded.metrics)
        np.testing.assert_array_equal(result.scores, loaded.scores)
        self.assertEqual(json.loads((output / 'manifest.json').read_text())['bootstrap']['seed'], 71)
        self.assertTrue((output / 'metrics.csv').is_file())
        self.assertTrue((output / 'slice_specs.json').is_file())
        with self.assertRaises(FileExistsError):
            result.save(output)


class AdditionalContractTests(unittest.TestCase):
    def test_filter_accepts_seen_iterable_and_validates_limit(self):
        self.assertEqual(filter_ranked_items([1, 2, 3], {1, 2}, np.arange(4)), [3])
        self.assertEqual(filter_ranked_items([1, 2, 3], iter([1, 2]), np.arange(4)), [3])
        self.assertEqual(filter_ranked_items(iter([1, 2]), [], np.arange(4), 0), [])
        with self.assertRaises(ValueError):
            filter_ranked_items([], [], [], -1)

    def test_float64_scores_count_cache_and_preparation_metadata(self):
        class Prepared(dict):
            candidate_preparation = 'full_ranking_filtered_before_top1000'
        with tempfile.TemporaryDirectory() as root:
            data = fixture(root)
            first = evaluate(data, Prepared({10: [301]}), model_name='a', ks=(2,), replicates=3)
            second = evaluate(data, {}, model_name='b', ks=(2,), replicates=3)
            self.assertEqual(first.scores.dtype, np.float64)
            self.assertIs(first.counts, second.counts)
            self.assertEqual(first.metadata['candidate_preparation'], Prepared.candidate_preparation)
            output = Path(root) / 'result'
            first.save(output)
            with np.load(output / 'statistics.npz') as saved:
                self.assertNotIn('scores', saved.files)
                self.assertEqual(set(saved.files), {'user_ids', 'ks', 'counts', 'hits'})
            paired = paired_bootstrap(first, second, replicates=3, output_dir=Path(root) / 'paired')
            self.assertTrue((Path(root) / 'paired' / 'metrics.csv').is_file())
            self.assertEqual(paired.metadata['direction'], 'first minus second')

    def test_no_targets_and_no_queries_are_unknown(self):
        data = EvaluationData(np.array([], dtype=np.uint32), np.array([], dtype=np.uint32),
            np.array([0]), np.array([], dtype=np.uint32),
            dict(offsets=np.array([0]), item_id=np.array([], dtype=np.uint32),
                 place=np.array([]), main_positive=np.array([]), secondary_positive=np.array([]),
                 like_positive=np.array([]), train_seen=np.array([]), duration_band=np.array([]),
                 duration_band_sensitive=np.array([]), popularity_tier=np.array([])),
            dict(item_id=np.array([], dtype=np.uint32)))
        result = evaluate(data, {}, model_name='empty', ks=(50,), replicates=3)
        self.assertEqual(len(result.metrics), 495)
        self.assertTrue(all(row['macro_capped'] is None and row['valid_draws'] == 0 for row in result.metrics))


class BootstrapSequenceTests(unittest.TestCase):
    def test_every_draw_matches_poisson_across_chunk_boundary(self):
        from lsvd.evaluation import _bootstrap
        counts = np.array([[1, 3, 0], [2, 0, 4]], dtype=np.uint32)
        hits = np.array([[[1, 1, 0]], [[1, 0, 3]]], dtype=np.uint32)
        denominators = np.minimum(counts[:, None, :], 3)
        scores = np.divide(hits, denominators, out=np.zeros(hits.shape, dtype=np.float64),
                           where=denominators > 0)
        macro, micro = _bootstrap(counts, hits, scores, 35, 431)
        weights = np.random.default_rng(431).poisson(1, size=(35, 3))
        for index in range(2):
            macro_denominator = weights @ (counts[index] > 0)
            micro_denominator = weights @ counts[index]
            expected_macro = np.divide(weights @ scores[index, 0], macro_denominator,
                out=np.full(35, np.nan), where=macro_denominator > 0)
            expected_micro = np.divide(weights @ hits[index, 0], micro_denominator,
                out=np.full(35, np.nan), where=micro_denominator > 0)
            np.testing.assert_allclose(macro[index, 0], expected_macro)
            np.testing.assert_allclose(micro[index, 0], expected_micro)


class MacroPrecisionTests(unittest.TestCase):
    def test_macro_uses_float64_mean_on_target_users_only(self):
        rng = np.random.default_rng(890)
        for _ in range(2):
            counts = rng.integers(0, 10, size=127)
            user_hits = np.array([rng.integers(0, n + 1) for n in counts])
        offsets = np.r_[0, np.cumsum(counts)]
        n_targets = int(offsets[-1])
        targets = dict(offsets=offsets, item_id=np.concatenate([np.arange(n) for n in counts]),
            place=np.zeros(n_targets), main_positive=np.ones(n_targets),
            secondary_positive=np.ones(n_targets), like_positive=np.ones(n_targets),
            train_seen=np.ones(n_targets), duration_band=np.zeros(n_targets),
            duration_band_sensitive=np.zeros(n_targets), popularity_tier=np.ones(n_targets))
        data = EvaluationData(np.arange(127), np.array([], dtype=np.uint32), np.array([0]),
                              np.array([], dtype=np.uint32), targets, dict(item_id=np.arange(10)))
        result = evaluate(data, {u: range(n) for u, n in enumerate(user_hits)},
                          model_name='precision', ks=(1000,), replicates=3)
        expected = (user_hits[counts > 0] / counts[counts > 0]).mean()
        self.assertEqual(metric(result, k=1000)['macro_capped'], expected)


class EvaluationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = fixture(self.temp.name)

    def evaluate(self):
        return evaluate(self.data, {}, model_name='identity', ks=(2,), replicates=3)

    def test_same_counts_different_target_items_cannot_be_paired(self):
        first = self.evaluate()
        self.data.targets['item_id'][-1] = 306
        second = self.evaluate()
        np.testing.assert_array_equal(first.counts, second.counts)
        with self.assertRaisesRegex(ValueError, 'identity'):
            paired_bootstrap(first, second, replicates=3)

    def test_conflicting_foundation_manifests_cannot_be_paired(self):
        first, second = self.evaluate(), self.evaluate()
        first.metadata['foundation_sha256'] = 'first-foundation'
        second.metadata['foundation_sha256'] = 'second-foundation'
        with self.assertRaisesRegex(ValueError, 'foundation'):
            paired_bootstrap(first, second, replicates=3)

    def test_missing_or_empty_identity_cannot_be_paired(self):
        first, second = self.evaluate(), self.evaluate()
        for value in (None, ''):
            with self.subTest(identity=value):
                first.metadata['evaluation_identity'] = value
                second.metadata['evaluation_identity'] = value
                with self.assertRaisesRegex(ValueError, 'identity'):
                    paired_bootstrap(first, second, replicates=3)

    def test_all_known_array_columns_participate_in_identity(self):
        original = self.data.evaluation_identity()
        self.assertEqual(len(original), 64)
        arrays = [self.data.user_ids, self.data.history_user_ids,
                  self.data.history_offsets, self.data.history_items]
        arrays += list(self.data.targets.values()) + list(self.data.items.values())
        for index, original_array in enumerate(arrays):
            if original_array is self.data.history_items:
                # The loader's history mmap is intentionally read-only.
                self.data.history_items = original_array.copy()
                array = self.data.history_items
            else:
                array = original_array
            before = array[0].copy()
            array[0] += 1
            self.assertNotEqual(self.data.evaluation_identity(), original, index)
            array[0] = before
            self.assertEqual(self.data.evaluation_identity(), original)

    def test_identity_is_canonical_for_mapping_order_and_noncontiguous_values(self):
        identity = self.data.evaluation_identity()
        self.data.targets = dict(reversed(list(self.data.targets.items())))
        self.data.items = dict(reversed(list(self.data.items.items())))
        for collection in (self.data.targets, self.data.items):
            for name, array in collection.items():
                padded = np.empty((len(array), 2), dtype=array.dtype)
                padded[:, 0] = array
                collection[name] = padded[:, 0]
        self.assertEqual(self.data.evaluation_identity(), identity)

    def test_identity_change_invalidates_cached_counts_and_target_users(self):
        first = self.evaluate()
        self.data.targets['main_positive'][0] = 0
        second = self.evaluate()
        self.assertEqual(metric(first)['n_targets'], 3)
        self.assertEqual(metric(second)['n_targets'], 2)
        self.assertNotEqual(first.metadata['evaluation_identity'], second.metadata['evaluation_identity'])

    def test_offset_change_invalidates_target_user_cache(self):
        first = self.evaluate()
        self.data.targets['offsets'][2] = 4
        second = self.evaluate()
        index = next(spec['slice_id'] for spec in second.slice_specs
                     if spec['ruler'] == 'main' and spec['place_scope'] == '0/1'
                     and spec['seen_scope'] == 'all' and spec['breakdown'] == 'all')
        np.testing.assert_array_equal(first.counts[index], [2, 1, 0])
        np.testing.assert_array_equal(second.counts[index], [2, 0, 1])

    def test_array_hash_uses_bounded_memoryviews_and_encodes_type_and_shape(self):
        import hashlib
        from lsvd.evaluation import _hash_array
        class RecordingDigest:
            def __init__(self):
                self.digest = hashlib.sha256()
                self.updates = []
            def update(self, value):
                self.updates.append((type(value), len(value)))
                self.digest.update(value)
        values = np.arange(9 * 1024 * 1024, dtype=np.uint8)
        recorded = RecordingDigest()
        _hash_array(recorded, 'large', values)
        chunks = [length for kind, length in recorded.updates if kind is memoryview]
        self.assertEqual(chunks, [8 * 1024 * 1024, 1024 * 1024])
        header = json.dumps(['large', values.dtype.str, values.shape], separators=(',', ':')).encode()
        expected = hashlib.sha256()
        expected.update(len(header).to_bytes(8, 'big'))
        expected.update(header)
        expected.update(memoryview(values).cast('B'))
        self.assertEqual(recorded.digest.hexdigest(), expected.hexdigest())
        retyped = RecordingDigest()
        _hash_array(retyped, 'large', values.view(np.int8))
        self.assertNotEqual(recorded.digest.hexdigest(), retyped.digest.hexdigest())
        reshaped = RecordingDigest()
        _hash_array(reshaped, 'large', values.reshape(-1, 2))
        self.assertNotEqual(recorded.digest.hexdigest(), reshaped.digest.hexdigest())

    def test_saved_identity_and_foundation_are_preserved_for_pairing(self):
        result = self.evaluate()
        result.metadata['foundation_sha256'] = 'same-foundation'
        path = Path(self.temp.name) / 'bound-result'
        result.save(path)
        loaded = EvaluationResult.load(path)
        self.assertEqual(loaded.metadata['evaluation_identity'], self.data.evaluation_identity())
        paired = paired_bootstrap(result, loaded, replicates=3)
        self.assertEqual(metric(paired)['macro_difference'], 0.)
        self.assertEqual(paired.metadata['evaluation_identity'], loaded.metadata['evaluation_identity'])
        self.assertEqual(paired.metadata['foundation_sha256'], 'same-foundation')


class CatalogMembershipTests(unittest.TestCase):
    def test_packed_bits_share_bytes_without_overwriting(self):
        from lsvd.evaluation import CatalogMembership
        catalog = np.array([0, 1, 2, 7, 8, 9, 15, 23], dtype=np.uint32)
        membership = CatalogMembership(catalog)
        np.testing.assert_array_equal(membership.bitmap, np.array([135, 131, 128], dtype=np.uint8))
        np.testing.assert_array_equal(membership.contains(np.arange(25)), np.isin(np.arange(25), catalog))

    def test_bounds_empty_negative_and_wider_queries(self):
        from lsvd.evaluation import CatalogMembership
        membership = CatalogMembership(np.array([0, 2, 255], dtype=np.uint32))
        np.testing.assert_array_equal(membership.contains(np.array([-1, 0, 255, 256, 2**32], dtype=np.int64)),
                                      [False, True, True, False, False])
        np.testing.assert_array_equal(membership.contains(np.array([0, 2**32, 2**64-1], dtype=np.uint64)),
                                      [True, False, False])
        self.assertEqual(membership.contains([]).shape, (0,))
        empty = CatalogMembership(np.array([], dtype=np.uint32))
        self.assertEqual(empty.bitmap.nbytes, 0)
        np.testing.assert_array_equal(empty.contains(np.array([0, 1, -1])), [False, False, False])
        with self.assertRaises(ValueError):
            membership.contains([1.5])

    def test_outside_uint32_domain_uses_exact_sorted_catalog_fallback(self):
        from lsvd.evaluation import CatalogMembership
        catalog = np.array([0, 2**32, 2**63, 2**64-1], dtype=np.uint64)
        membership = CatalogMembership(catalog)
        self.assertIsNone(membership.bitmap)
        self.assertIs(membership.catalog, catalog)
        values = np.array([0, 1, 2**32-1, 2**32, 2**63, 2**64-2, 2**64-1], dtype=np.uint64)
        np.testing.assert_array_equal(membership.contains(values), [True, False, False, True, True, False, True])
        negative_catalog = CatalogMembership(np.array([-2, 2**40], dtype=np.int64))
        self.assertIsNone(negative_catalog.bitmap)
        np.testing.assert_array_equal(negative_catalog.contains([-3, -2, 0, 2**40]), [False, True, False, True])

    def test_random_membership_matches_binary_search(self):
        from lsvd.evaluation import CatalogMembership
        rng = np.random.default_rng(581)
        for dtype in (np.uint32, np.int64, np.uint64):
            catalog = np.unique(rng.integers(0, 8192, 2000)).astype(dtype)
            values = rng.integers(0, 10000, (8, 127)).astype(dtype)
            positions = np.searchsorted(catalog, values)
            expected = positions < len(catalog)
            expected &= catalog[np.minimum(positions, len(catalog) - 1)] == values
            np.testing.assert_array_equal(CatalogMembership(catalog).contains(values), expected)

    def test_filter_matches_original_catalog_representation_exactly(self):
        from lsvd.evaluation import CatalogMembership
        rng = np.random.default_rng(586)
        catalog = np.unique(rng.integers(0, 2048, 3000)).astype(np.uint32)
        seen = rng.choice(catalog, 300, replace=False)
        ranked = rng.choice(catalog, 2000, replace=True).tolist() + [2**32 - 1]
        membership = CatalogMembership(catalog)
        for limit in (0, 1, 17, 1000):
            self.assertEqual(filter_ranked_items(iter(ranked), seen, membership, limit),
                             filter_ranked_items(iter(ranked), seen, catalog, limit))
        for invalid in ([-1], [4095], [2**32], [2**64 - 1]):
            for representation in (catalog, membership):
                with self.assertRaises(ValueError):
                    filter_ranked_items(invalid, seen, representation)

    def test_packed_contains_does_not_repeat_binary_search(self):
        from unittest.mock import patch
        from lsvd.evaluation import CatalogMembership
        membership = CatalogMembership(np.array([0, 1, 9, 15], dtype=np.uint32))
        with patch('lsvd.evaluation.np.searchsorted', side_effect=AssertionError('repeated binary search')):
            np.testing.assert_array_equal(membership.contains(np.array([0, 2, 15, 16])),
                                          [True, False, True, False])

    def test_evaluate_builds_membership_once_per_evaluation(self):
        from unittest.mock import patch
        from lsvd.evaluation import CatalogMembership
        class CountingMembership(CatalogMembership):
            builds = 0
            def __init__(self, catalog):
                type(self).builds += 1
                super().__init__(catalog)
        with tempfile.TemporaryDirectory() as root:
            data = fixture(root)
            identity = data.evaluation_identity()
            with patch('lsvd.evaluation.CatalogMembership', CountingMembership):
                result = evaluate(data, {10: [301], 20: [301]}, model_name='membership', ks=(2,), replicates=3)
            self.assertEqual(CountingMembership.builds, 1)
            self.assertEqual(result.metadata['catalog_validation'], 'packed_membership')
            self.assertEqual(result.metadata['evaluation_identity'], identity)
            self.assertEqual(metric(result)['macro_capped'], .75)


if __name__ == '__main__':
    unittest.main()
