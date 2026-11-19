import numpy as np
import pytest

from lsvd.evaluation import PADDING_ITEM_ID


def test_supply_tracks_short_lists_and_keeps_all_users():
    from lsvd.generator_eval import supply_summary

    ranks = np.array([[1, 2, 3], [4, 5, PADDING_ITEM_ID],
                      [PADDING_ITEM_ID] * 3], dtype=np.uint32)
    receipt = supply_summary({'full_sid': ranks}, (2, 3))
    assert receipt['full_sid'][2] == {
        'p10': pytest.approx(.4), 'p50': 2, 'p90': 2,
        'short_users': 1, 'users': 3, 'short_fraction': pytest.approx(1 / 3),
    }
    assert receipt['full_sid'][3]['short_users'] == 2


def test_first_prune_is_first_absent_prefix_even_if_item_window_is_short():
    from lsvd.generator_eval import first_prune_depth

    # Packed base-4 items 000, 001, 012, 111, 211.
    codes = np.array([0, 1, 6, 21, 37])
    levels = {1: np.array([0, 1]), 2: np.array([0, 5]),
              3: np.array([0, 21])}
    np.testing.assert_array_equal(first_prune_depth(codes, levels, 4), [0, 3, 2, 0, 1])


def test_sdk_statistics_use_zip_deflate_and_load_unchanged(tmp_path):
    from zipfile import ZIP_DEFLATED, ZipFile

    from lsvd.evaluation import EvaluationResult, evaluate
    from lsvd.two_tower import TowerConfig
    from lsvd.two_tower_data import synthetic

    _, _, data, _ = synthetic(TowerConfig(), user_count=8, item_count=900)
    output = tmp_path / 'evaluation'
    result = evaluate(data, {}, model_name='fixture', replicates=1, output_dir=output)
    with ZipFile(output / 'statistics.npz') as archive:
        assert all(entry.compress_type == ZIP_DEFLATED for entry in archive.infolist())
    loaded = EvaluationResult.load(output)
    np.testing.assert_array_equal(loaded.counts, result.counts)
    np.testing.assert_array_equal(loaded.hits, result.hits)
