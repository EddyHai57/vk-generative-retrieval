"""Synthetic tests for training-only retrieval baselines."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import duckdb
import numpy as np

from lsvd import baselines as b


def data():
    return b.BaselineData(
        queries=dict(user_id=np.array([10, 20, 30]), history_count=np.array([2, 1, 0]),
                     last_item_id=np.array([2, 3, -1]), last_author_id=np.array([7, 99, -1])),
        items=dict(item_id=np.arange(1, 7, dtype=np.uint32), author_id=np.array([7, 7, 7, 8, 8, 7]),
                   train_count=np.array([10, 9, 8, 7, 0, 0])),
        history_users=np.array([10, 20]), history_offsets=np.array([0, 2, 3]),
        history_items=np.array([1, 2, 3], dtype=np.uint32))


class BaselineTests(unittest.TestCase):
    def test_global_uses_complete_ranking_and_zero_count_catalog(self):
        result = b.global_popularity(data(), limit=4)
        self.assertEqual(result.dtype, np.uint32)
        self.assertEqual(result.shape, (3, 4))
        np.testing.assert_array_equal(result[0, :2], [3, 4])
        self.assertEqual(set(result[0, 2:]), {5, 6})
        np.testing.assert_array_equal(result[2], [1, 2, 3, 4])

    def test_tie_uses_frozen_duckdb_hash(self):
        d = data()
        d.items['train_count'][:] = 0
        c = duckdb.connect()
        c.register('items', d.items)
        expected = c.execute('SELECT item_id FROM items ORDER BY hash(item_id,20261108::UBIGINT),item_id').fetchnumpy()['item_id']
        np.testing.assert_array_equal(b.global_popularity(d, limit=6)[2], expected)
        c.close()

    def test_author_empty_history_and_missing_author_stay_empty(self):
        result = b.last_author_popularity(data(), limit=3)
        np.testing.assert_array_equal(result[0], [3, 6, b.PADDING])
        self.assertTrue(np.all(result[1:] == b.PADDING))

    def test_all_generators_call_shared_filter(self):
        from lsvd import evaluation
        with patch.object(evaluation, 'filter_ranked_items', wraps=evaluation.filter_ranked_items) as shared:
            b.global_popularity(data(), limit=2)
            b.last_author_popularity(data(), limit=2)
            graph = dict(anchor_item_id=np.array([2]), neighbor_item_id=np.array([4]),
                         shared_users=np.array([2]), score=np.array([.2]))
            b.covisitation_neighbors(data(), graph, limit=2)
            self.assertGreaterEqual(shared.call_count, 9)

    def test_catalog_membership_built_once_per_model_and_shared_across_users(self):
        from lsvd import evaluation
        class Index:
            ntotal = 6
            def search(self, queries, k):
                positions = np.tile(np.arange(min(k, 6)), (len(queries), 1))
                return np.zeros(positions.shape), positions
        graph = dict(anchor_item_id=np.array([2]), neighbor_item_id=np.array([4]),
                     shared_users=np.array([2]), score=np.array([.2]))
        profiles = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
        generators = (
            lambda d: b.global_popularity(d, limit=2),
            lambda d: b.last_author_popularity(d, limit=2),
            lambda d: b.covisitation_neighbors(d, graph, limit=2),
            lambda d: b.retrieve_content(d, profiles, Index(), d.items['item_id'], initial_k=6, max_k=6, limit=2),
        )
        original = evaluation.CatalogMembership
        for generate in generators:
            d = data()
            constructed = []
            class TrackedMembership(original):
                def __init__(self, ids):
                    constructed.append((self, ids))
                    super().__init__(ids)
            with patch.object(evaluation, 'CatalogMembership', TrackedMembership):
                with patch.object(evaluation, 'filter_ranked_items', wraps=evaluation.filter_ranked_items) as shared:
                    generate(d)
                    self.assertEqual(len(constructed), 1)
                    self.assertIs(constructed[0][1], d.items['item_id'])
                    catalogs = [call.args[2] for call in shared.call_args_list]
                    self.assertGreater(len(catalogs), 1)
                    self.assertTrue(all(catalog is constructed[0][0] for catalog in catalogs))

    def test_catalog_membership_is_fresh_after_catalog_changes(self):
        d = data()
        before = b.global_popularity(d, limit=6)
        self.assertIn(6, before[2])
        d.items['item_id'][-1] = 7
        after = b.global_popularity(d, limit=6)
        self.assertIn(7, after[2])
        self.assertNotIn(6, after[2])

    def test_cooccurrence_distinct_user_support_score_and_cross_week(self):
        seq = [dict(user_id=10, item_ids=[1, 2, 3], weeks=[0, 1, 1]),
               dict(user_id=20, item_ids=[2, 1], weeks=[0, 24]),
               dict(user_id=30, item_ids=[1, 4], weeks=[0, 0])]
        graph = b.build_covisitation_graph(seq, {1: 3, 2: 2, 3: 1, 4: 1})
        self.assertEqual(graph['anchor_item_id'].tolist(), [1, 2])
        self.assertEqual(graph['neighbor_item_id'].tolist(), [2, 1])
        np.testing.assert_array_equal(graph['shared_users'], [2, 2])
        np.testing.assert_allclose(graph['score'], [2 / np.sqrt(6)] * 2)

    def test_graph_ranks_score_before_tie_and_uses_last_item(self):
        graph = dict(anchor_item_id=np.array([2, 2, 3]), neighbor_item_id=np.array([4, 3, 5]),
                     shared_users=np.array([2, 2, 2]), score=np.array([.2, .8, .3]))
        result = b.covisitation_neighbors(data(), graph, limit=2)
        np.testing.assert_array_equal(result[0], [3, 4])
        np.testing.assert_array_equal(result[1], [5, b.PADDING])
        self.assertTrue(np.all(result[2] == b.PADDING))

    def test_profiles_limit_all_exposures_before_main_bit_all_places(self):
        seq = [dict(user_id=10, item_ids=[1, 2, 3, 4], main_positive=[True, True, False, True],
                    places=[0, 2, 0, 2]), dict(user_id=20, item_ids=[5], main_positive=[True])]
        ids = np.arange(1, 6)
        vectors = np.array([[1, 0], [0, 2], [5, 0], [2, 0], [0, 0]], dtype=np.float32)
        profiles, counts = b.content_profiles([10, 20, 30], seq, ids, vectors, recent=3)
        np.testing.assert_allclose(profiles[0], np.array([1, 1]) / np.sqrt(2), rtol=1e-6)
        np.testing.assert_array_equal(profiles[1:], np.zeros((2, 2)))
        np.testing.assert_array_equal(counts, [2, 1, 0])

    def test_content_retrieval_refills_shared_filter_and_zero_profiles(self):
        class Index:
            ntotal = 6
            calls = []
            def search(self, q, k):
                self.calls.append(k)
                rows = np.tile(np.arange(min(k, 6)), (len(q), 1))
                return np.zeros(rows.shape), rows
        index = Index()
        profiles = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
        result = b.retrieve_content(data(), profiles, index, np.arange(1, 7), initial_k=2, max_k=6, limit=3)
        np.testing.assert_array_equal(result[0], [3, 4, 5])
        np.testing.assert_array_equal(result[1], [1, 2, 4])
        self.assertTrue(np.all(result[2] == b.PADDING))
        self.assertIn(6, index.calls)

    def test_frozen_centroids_index_has_no_training(self):
        try:
            import faiss
        except ImportError:
            self.skipTest('FAISS unavailable in runtime')
        vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
        index = b.index_from_centroids(vectors[:2], vectors, batch_size=2, nprobe=2)
        self.assertTrue(index.is_trained)
        self.assertEqual(index.nprobe, 2)
        self.assertEqual(index.ntotal, 4)
        np.testing.assert_array_equal(index.search(vectors[:1], 1)[1], [[0]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'vectors.npy'
            np.save(path, vectors)
            restored = b.index_from_centroids(vectors[:2], path, batch_size=2, nprobe=2)
            np.testing.assert_array_equal(restored.search(vectors[:1], 1)[1], [[0]])

    def test_loader_does_not_read_targets(self):
        d = data()
        with tempfile.TemporaryDirectory() as temp:
            arrays = Path(temp) / 'dataset' / 'arrays'
            arrays.mkdir(parents=True)
            np.savez(arrays / 'queries.npz', **d.queries)
            np.savez(arrays / 'items.npz', **d.items)
            np.savez(arrays / 'history.npz', user_id=d.history_users, offsets=d.history_offsets)
            np.save(arrays / 'history_items.npy', d.history_items)
            (arrays / 'targets.npz').write_bytes(b'not readable')
            for foundation in (Path(temp), arrays.parent):
                loaded = b.BaselineData.load(foundation)
                np.testing.assert_array_equal(b.global_popularity(loaded, limit=2), b.global_popularity(d, limit=2))

class ReusedGraphAuditTests(unittest.TestCase):
    def test_reused_graph_is_recomputed_from_new_sequences_and_rejects_changed_edges(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sequences = root / 'dataset' / 'sequences' / 'user_block=0'
            sequences.mkdir(parents=True)
            graph = root / 'graph.parquet'
            connection = duckdb.connect()
            connection.execute("COPY (SELECT * FROM (VALUES (10,[1,2],[7,7],[true,false]),(20,[2,1],[7,7],[false,true]),(30,[1,3],[7,7],[true,true])) s(user_id,item_ids,authors,main_positive)) TO ? (FORMAT PARQUET)", [str(sequences / 'users.parquet')])
            score = 2 / np.sqrt(6)
            connection.execute(f"COPY (SELECT * FROM (VALUES (1,2,2,{score}),(2,1,2,{score})) g(anchor_item_id,neighbor_item_id,shared_users,score)) TO '{graph}' (FORMAT PARQUET)")
            d = b.BaselineData(dict(user_id=np.array([10]), history_count=np.array([2]), last_item_id=np.array([2]), last_author_id=np.array([7])),
                               dict(item_id=np.array([1,2,3]), train_count=np.array([3,2,1])), np.array([10]), np.array([0,2]), np.array([1,2]))
            receipt = b.audit_reused_graph(d, root, graph, sample_size=10)
            self.assertEqual(receipt['status'], 'verified')
            self.assertEqual(receipt['sampled_anchors'], 2)
            self.assertEqual(len(receipt['sources']), 2)
            direct = b.audit_reused_graph(d, root / 'dataset', graph, sample_size=10)
            self.assertEqual(receipt, direct)
            expected_sequences = [dict(user_id=10, item_ids=[1,2], main_positive=[True,False]),
                                  dict(user_id=20, item_ids=[2,1], main_positive=[False,True]),
                                  dict(user_id=30, item_ids=[1,3], main_positive=[True,True])]
            for foundation in (root, root / 'dataset'):
                self.assertEqual(list(b.sequence_rows(foundation)), expected_sequences)
            bad = root / 'bad.parquet'
            connection.execute(f"COPY (SELECT anchor_item_id,neighbor_item_id,3 AS shared_users,score*1.5 AS score FROM read_parquet('{graph}')) TO '{bad}' (FORMAT PARQUET)")
            with self.assertRaises(ValueError):
                b.audit_reused_graph(d, root, bad, sample_size=10)
            connection.close()


if __name__ == '__main__':
    unittest.main()
