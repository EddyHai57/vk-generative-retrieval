from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _bundle():
    return SimpleNamespace(
        ids=np.arange(10, 90, 10, dtype=np.uint32),
        codes=np.array([
            [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 2],
            [1, 0, 0], [1, 0, 0], [1, 0, 1], [2, 1, 0],
        ], dtype=np.uint16),
        bucket_keys=np.array([0, 1, 2, 9, 10, 21], dtype=np.uint64),
        bucket_offsets=np.array([0, 2, 3, 4, 6, 7, 8], dtype=np.uint64),
        bucket_items=np.arange(10, 90, 10, dtype=np.uint32),
        item_bucket=np.array([0, 0, 1, 2, 3, 3, 4, 5], dtype=np.uint32),
        answer_counts=np.array([4, 0, 0, 0, 0, 0], dtype=np.uint32),
        prefixes=(
            (np.array([0]), np.array([0, 3]), np.array([0, 1, 2])),
            (np.array([0, 1, 2]), np.array([0, 1, 2, 3]), np.array([0, 0, 1])),
            (np.array([0, 3, 7]), np.array([0, 3, 5, 6]), np.array([0, 1, 2, 0, 1, 0])),
        ),
        size=3,
    )


def _logits(query, prefix):
    # Illegal tokens can dominate a legal parent's normalization at the next step.
    if not prefix:
        return (2.5, 2.5, 1.0) if query == 0 else (0.0, 1.0, 2.0)
    if len(prefix) == 1:
        return ((0.0, 8.0, 8.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0))[prefix[0]]
    return ((1.0, 1.0, 1.0), (2.0, 2.0, 0.0), (4.0, 0.0, 0.0))[prefix[0]]


@pytest.mark.parametrize(("width", "sizes"), [
    (1, (1, 1, 1)), (2, (2, 2, 2)), (8, (3, 3, 6)),
    ((1, 8, 2), (1, 1, 2)), ((3, 1, 8), (3, 1, 3)), ([2, 3, 1], (2, 2, 1)),
])
def test_batched_beam_matches_independent_enumeration(width, sizes):
    from lsvd.generator_decode import PrefixTrie, beam_search

    bundle = _bundle()
    trie = PrefixTrie(bundle.prefixes, bundle.size, "cpu")
    calls = []

    def score(paths):
        calls.append(tuple(paths.shape))
        return torch.tensor([
            [_logits(query, tuple(path.tolist())) for path in row]
            for query, row in enumerate(paths)
        ], dtype=torch.float16, device=paths.device)

    output = beam_search(trie, score, batch_size=2, beam_width=width)
    assert calls == [(2, 1, 0), (2, sizes[0], 1), (2, sizes[1], 2)]
    catalog = {tuple(code) for code in bundle.codes.tolist()}
    for query in range(2):
        active = [((), 0.0)]
        for depth in range(1, 4):
            expanded = []
            for prefix, total in active:
                logits = torch.tensor(_logits(query, prefix), dtype=torch.float32)
                probs = torch.log_softmax(logits, dim=0).tolist()
                for digit in range(3):
                    path = prefix + (digit,)
                    if any(code[:depth] == path for code in catalog):
                        expanded.append((path, total + probs[digit]))
            limit = width if isinstance(width, int) else width[depth - 1]
            active = sorted(expanded, key=lambda pair: (-pair[1], pair[0]))[:limit]
            keys = [sum(digit * 3 ** (depth - i - 1) for i, digit in enumerate(path))
                    for path, _ in active]
            level = output[depth]
            assert level.logp.dtype == torch.float64
            assert tuple(level.keys.shape) == (2, sizes[depth - 1])
            assert level.keys.device == trie.device
            valid = level.keys[query] >= 0
            assert level.keys[query, valid].tolist() == keys
            torch.testing.assert_close(
                level.logp[query, valid],
                torch.tensor([score for _, score in active], dtype=torch.float64),
                rtol=0,
                atol=0,
            )
            assert torch.isneginf(level.logp[query, ~valid]).all()


def test_beam_cutoff_keeps_lexical_ties_even_when_children_arrive_out_of_order():
    from lsvd.generator_decode import PrefixTrie, beam_search

    prefixes = list(_bundle().prefixes)
    keys, offsets, _ = prefixes[-1]
    prefixes[-1] = (keys, offsets, np.array([2, 1, 0, 1, 0, 0]))
    trie = PrefixTrie(prefixes, size=3, device="cpu")

    def score(paths):
        return torch.zeros((*paths.shape[:2], 3))

    level = beam_search(trie, score, 1, (3, 3, 2))[3]
    assert level.keys.tolist() == [[0, 1]]
    assert level.logp[0, 0] == level.logp[0, 1]


def test_decode_config_requires_three_positive_widths_and_final_width_at_least_k():
    from lsvd.generator_decode import DecodeConfig

    config = DecodeConfig()
    assert config.beam_widths == (1024, 1024, 1024)
    assert config.prefix_beam_width == 64
    assert DecodeConfig(k=200, beam_widths=(8, 16, 200)).beam_widths == (8, 16, 200)
    with pytest.raises(ValueError, match="beam"):
        DecodeConfig(k=200, beam_widths=(8, 16, 199))
    for widths in ((1, 200), (0, 128, 200), (1, 2.5, 200)):
        with pytest.raises(ValueError, match="beam"):
            DecodeConfig(k=200, beam_widths=widths)
    with pytest.raises(ValueError, match="beam"):
        DecodeConfig(prefix_beam_width=0)


@pytest.mark.parametrize("widths", [(1, 2, 3), (1, 0)])
def test_two_digit_search_rejects_invalid_widths(widths):
    from lsvd.generator_decode import PrefixTrie, beam_search

    trie = PrefixTrie(_bundle().prefixes, size=3, device="cpu")
    with pytest.raises(ValueError, match="beam"):
        beam_search(trie, None, 1, widths, depth=2)


def test_two_digit_beam_survives_even_when_full_beam_prunes_its_child():
    from lsvd.generator_decode import PrefixTrie, beam_search

    prefixes = list(_bundle().prefixes)
    keys, offsets, digits = prefixes[-1]
    prefixes[-1] = (keys, np.r_[offsets[:-1], 7], np.r_[digits, 1])
    trie = PrefixTrie(prefixes, size=3, device="cpu")

    def score(paths):
        if not paths.shape[-1]:
            return torch.tensor([[[0.0, 2.0, 1.0]]])
        if paths.shape[-1] == 2:
            # Best 2-digit prefix (1, 0) has low probability for either legal child.
            return torch.where(paths[..., :1] == 1,
                               torch.tensor([0.0, 0.0, 15.0]),
                               torch.tensor([8.0, 8.0, 0.0]))
        return torch.tensor([
            [_logits(query, tuple(path.tolist())) for path in row]
            for query, row in enumerate(paths)
        ])

    joint = beam_search(trie, score, batch_size=1, beam_width=2)
    isolated = beam_search(trie, score, batch_size=1, beam_width=2, depth=2)
    assert joint[2].keys[0].tolist() == [3, 7]
    assert joint[3].keys[0].tolist() == [21, 22]
    torch.testing.assert_close(joint[2].keys, isolated[2].keys)
    torch.testing.assert_close(joint[2].logp, isolated[2].logp)


def _ranker(config):
    from lsvd.generator_decode import GeneratorRanker

    similarity = np.array([.99, .6, .8, .9, .95, .7, .5, .3], dtype=np.float32)
    vectors = np.column_stack((similarity, np.sqrt(1 - similarity ** 2))).astype(np.float16)
    return GeneratorRanker(
        _bundle(), vectors, np.array([40, 0, 0, 8000, 0, 0, 1, 0]),
        np.array([4, 0, 0, 0, 0, 0, 0, 0]), config,
    )


def test_four_sid_views_use_shared_old_items_and_preserve_full_code_collisions():
    from lsvd.generator_decode import DecodeConfig

    ranker = _ranker(DecodeConfig(candidate_budget=4, k=3, prefix_limit=2, new_item_quota=2))
    result = ranker.rank(
        [0, 1, 2, -1], [-4., -3., -2., -np.inf], [0, 3], [-1., -2.],
        [1., 0.], {10}, routes=("full_sid", "direct", "hitchhike", "prefix"),
    )
    assert result["full_sid"].items.tolist() == [40, 30, 20]
    assert result["full_sid"].sources.tolist() == ["old", "direct", "hitchhike"]
    assert result["direct"].items.tolist() == [40, 30]
    assert result["hitchhike"].items.tolist() == [40, 20]
    assert result["prefix"].items.tolist() == [50, 40, 30]
    assert result["prefix"].candidate_count == 4
    assert result["prefix"].new_candidates == 3
    assert result["prefix"].old_candidates == 1
    assert result["prefix"].sources.tolist() == ["prefix", "old", "prefix"]


def test_prefix_cap_uses_user_score_across_old_and_new_items():
    from lsvd.generator_decode import DecodeConfig

    ranker = _ranker(DecodeConfig(candidate_budget=4, k=3, prefix_limit=1))
    result = ranker.rank([2], [-1.], [0], [-.5], [1., 0.], {10}, routes=("prefix",))
    # Item 40 scores above lower-ID new items 20 and 30.
    assert result["prefix"].items.tolist() == [40]
    assert result["prefix"].new_candidates == 0


@pytest.mark.parametrize("k", [200, 1000])
def test_short_and_empty_rankings_keep_k_and_allow_a_single_group(k):
    from lsvd.generator_decode import DecodeConfig

    ranker = _ranker(DecodeConfig(k=k))
    routes = ("full_sid", "direct", "hitchhike", "prefix")
    empty = ranker.rank([], [], [], [], [1., 0.], [], routes=routes)
    assert all(len(route.items) == route.candidate_count == 0 for route in empty.values())
    results = ranker.rank(
        [0, 1, 2, 10], [0., 0., 0., 0.], [0], [-1.], [1., 0.],
        {10, 40, 70}, routes=routes,
    )
    assert results["full_sid"].items.tolist() == [30, 20]
    assert results["direct"].items.tolist() == [30]
    assert results["hitchhike"].items.tolist() == [20]
    assert all(route.old_candidates == 0 for route in results.values())
    assert ranker.content_index is None


def test_new_quota_reserves_candidates_and_backfills_old_items():
    from lsvd.generator_decode import DecodeConfig

    ranker = _ranker(DecodeConfig(candidate_budget=2, k=2, new_item_quota=1))
    result = ranker.rank([0, 1, 2, 10], [0., 0., 0., 0.], [], [], [1., 0.], [])
    # Both top candidates by unconstrained score are old.
    assert result["full_sid"].items.tolist() == [10, 30]
    assert result["full_sid"].candidate_count == 2
    result = ranker.rank([2, 10], [0., 0.], [], [], [1., 0.], [], routes=("direct",))
    assert result["direct"].items.tolist() == [40, 70]


def test_beam_product_changes_ranking_and_popularity_uses_target_count():
    from lsvd.generator_decode import DecodeConfig

    # Older high exposure item 40 has no target count and receives no popularity.
    full = [0, 1, 2]
    logp = [-4., -1., -3.]
    regular = _ranker(DecodeConfig(candidate_budget=4, k=4))
    multiplied = _ranker(DecodeConfig(candidate_budget=4, k=4, ranking="beam_product"))
    items = regular.rank(full, logp, [], [], [1., 0.], [])["full_sid"]
    assert items.items.tolist() == [10, 40, 30, 20]
    np.testing.assert_allclose(items.scores[:2], [.990234375 + .02 * np.log(5), .89990234375])
    product = multiplied.rank(full, logp, [], [], [1., 0.], [])["full_sid"]
    assert product.items.tolist() == [30, 40, 10, 20]
    np.testing.assert_allclose(product.scores[0], .7998046875 * np.exp(-1.))
    np.testing.assert_allclose(product.scores[2], .990234375 * np.exp(-4.))


@pytest.mark.parametrize("routes", [
    ("direct",), ("hitchhike",), ("prefix",), ("content",), ("full_sid", "prefix"), (),
])
def test_beam_product_requires_full_sid_only(routes):
    from lsvd.generator_decode import DecodeConfig

    ranker = _ranker(DecodeConfig(candidate_budget=4, k=4, ranking="beam_product"))
    with pytest.raises(ValueError, match="beam_product.*full_sid"):
        ranker.rank([0], [-1.], [0], [-.5], [1., 0.], [], routes=routes)


def test_equal_scores_use_the_shared_fixed_hash():
    from lsvd.generator_decode import DecodeConfig
    from lsvd.two_tower_retrieval import tie_keys

    ranker = _ranker(DecodeConfig(candidate_budget=4, k=4, popularity_weight=0))
    ranker.vectors = np.ones_like(ranker.vectors)
    result = ranker.rank([0, 1, 2], [0., 0., 0.], [], [], [1., 0.], [])["full_sid"]
    candidates = np.arange(4)
    order = np.lexsort((ranker.ids[:4], tie_keys(ranker)[candidates]))
    np.testing.assert_array_equal(result.items, ranker.ids[candidates[order]])


def test_content_ann_is_lazy_and_nominates_unseen_items_without_a_beam():
    pytest.importorskip("faiss")
    from lsvd.generator_decode import DecodeConfig

    ranker = _ranker(DecodeConfig(candidate_budget=4, k=3, new_item_quota=2, ann_nlist=1))
    result = ranker.rank([2], [-1.], [], [], [1., 0.], {})
    assert result["full_sid"].items.tolist() == [40]
    assert ranker.content_index is None
    result = ranker.rank([2], [-1.], [], [], [1., 0.], {50}, routes=("content",))
    assert result["content"].items.tolist() == [40, 30, 60]
    assert result["content"].sources.tolist() == ["old", "content", "content"]
    assert result["content"].candidate_count == 4
    assert ranker.content_index.ntotal == 5
