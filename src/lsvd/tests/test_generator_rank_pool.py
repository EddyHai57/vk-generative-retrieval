import os
from types import SimpleNamespace

import numpy as np
import pytest

from lsvd.generator_decode import DecodeConfig, GeneratorRanker
from lsvd.generator_eval import prepare_ranker


def _ranker(vectors, ranking="item", *, all_old=False):
    ids = np.arange(10, 70, 10, dtype=np.uint32)
    bundle = SimpleNamespace(
        size=3,
        ids=ids,
        bucket_keys=np.array([0, 1, 9, 10], dtype=np.uint64),
        bucket_offsets=np.array([0, 2, 3, 5, 6]),
        bucket_items=ids.copy(),
        item_bucket=np.array([0, 0, 1, 2, 2, 3]),
        answer_counts=np.array([1, 0, 1, 0]),
    )
    train_count = np.ones(6, dtype=np.int64) if all_old else np.array([4, 0, 0, 8, 0, 0])
    return GeneratorRanker(
        bundle, vectors, train_count, np.array([3, 0, 0, 1, 0, 0]),
        DecodeConfig(candidate_budget=5, k=3, prefix_limit=2, new_item_quota=2,
                     ranking=ranking, ann_nlist=1),
    )


def _vectors():
    return np.array([[1, 0], [.75, .5], [.5, .75], [0, 1], [.75, .5], [-1, 0]],
                    dtype=np.float16)


def _entries():
    return [
        ([0, 1, 9, -1], [-1., -2., -3., -np.inf], [0, 3, -1], [-.5, -.8, -np.inf],
         [1., 0.], {10, 20}),
        None,
        (np.array([9, 10]), np.array([-4., -.1]), [3], [-1.], [0., 1.],
         np.array([50, 10, 50], dtype=np.uint32)),
        ([-1], [-np.inf], [-1], [-np.inf], [1., 1.], []),
        ([], [], [], [], [0., 1.], {10, 20, 30, 40, 50, 60}),
    ]


def _assert_equal(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        if right is None:
            assert left is None
            continue
        assert list(left) == list(right)
        for route, result in right.items():
            outcome = left[route]
            np.testing.assert_array_equal(outcome.items, result.items, strict=True)
            np.testing.assert_array_equal(outcome.scores, result.scores, strict=True)
            np.testing.assert_array_equal(outcome.sources, result.sources, strict=True)
            assert outcome.candidate_count == result.candidate_count
            assert outcome.new_candidates == result.new_candidates
            assert outcome.old_candidates == result.old_candidates


def _fail_if_ranked_in_parent(*args, **kwargs):
    pytest.fail("parallel ranking called the parent ranker")


@pytest.mark.parametrize("all_old", [False, True])
def test_parallel_routes_reuse_vector_npy_and_match_serial(tmp_path, all_old):
    from lsvd.generator_rank_pool import RankPool

    pytest.importorskip("faiss")
    vector_path = tmp_path / "vectors.npy"
    np.save(vector_path, _vectors(), allow_pickle=False)
    ranker = _ranker(np.load(vector_path, mmap_mode="c"), all_old=all_old)
    routes = ("full_sid", "direct", "hitchhike", "prefix", "content")
    entries = _entries()
    prepare_ranker(ranker, routes)
    expected = [None if entry is None else ranker.rank(*entry, routes=routes)
                for entry in entries]
    assert len(ranker._content_pool) == (0 if all_old else 4)
    assert expected[0]["full_sid"].items.tolist() == [50, 30, 40]
    assert expected[0]["content"].new_candidates == (0 if all_old else 3)
    ranker.rank = _fail_if_ranked_in_parent
    parent_threads = os.environ.get("VECLIB_MAXIMUM_THREADS")

    with RankPool(ranker, routes, workers=2, temp_directory=tmp_path) as pool:
        _assert_equal(pool.rank(iter(entries)), expected)
        with_seen_iterator = (*entries[2][:-1], iter(entries[2][-1]))
        _assert_equal(pool.rank(iter((with_seen_iterator, *entries[3:]))), expected[2:])
        assert pool.rank(iter(())) == []
        staged_shapes = [np.load(path, mmap_mode="r").shape
                         for path in tmp_path.glob("*/*.npy")]
        assert staged_shapes
        assert ranker.vectors.shape not in staged_shapes

    assert os.environ.get("VECLIB_MAXIMUM_THREADS") == parent_threads
    assert list(tmp_path.iterdir()) == [vector_path]


@pytest.mark.parametrize("workers", [1, 2])
def test_beam_product_staging_preserves_a_nontrivial_vector_view(tmp_path, workers):
    from lsvd.generator_rank_pool import RankPool

    vector_path = tmp_path / "vectors.npy"
    np.save(vector_path, _vectors()[:, ::-1], allow_pickle=False)
    ranker = _ranker(np.load(vector_path, mmap_mode="r")[:, ::-1], ranking="beam_product")
    routes = ("full_sid",)
    entries = _entries()
    expected = [None if entry is None else ranker.rank(*entry, routes=routes)
                for entry in entries]
    assert expected[0]["full_sid"].items.tolist() == [30, 50, 40]

    with RankPool(ranker, routes, workers=workers, temp_directory=tmp_path) as pool:
        _assert_equal(pool.rank(iter(entries)), expected)

    assert ranker._content_pool is None
    assert list(tmp_path.iterdir()) == [vector_path]
