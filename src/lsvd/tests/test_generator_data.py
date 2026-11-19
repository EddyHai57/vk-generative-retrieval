import json
import os
import subprocess
import sys

import duckdb
import numpy as np
import pytest
import torch

from lsvd.sid_tokenizer import build_tables
from lsvd.two_tower import TowerConfig, TwoTower
from lsvd.two_tower_prepare import build_tensor_dataset, load_tensor_dataset, sha256_file
from test_two_tower_prepare import foundation_fixture


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp('generator-data')
    config = TowerConfig(id_min_count=5, use_rank=True)
    foundation = root / 'foundation'
    foundation.mkdir()
    foundation_fixture(foundation)
    items = foundation / 'items.parquet'
    updated = foundation / 'items-with-tail.parquet'
    with duckdb.connect() as conn:
        conn.read_parquet(str(items)).create_view('catalog_fixture')
        conn.execute('''COPY (SELECT * REPLACE (
            CASE WHEN item_id = 799 THEN 2 ELSE train_count END AS train_count
        ) FROM catalog_fixture) TO ? (FORMAT PARQUET)''', [str(updated)])
    updated.replace(items)
    tensors = root / 'tensors'
    build_tensor_dataset(foundation, foundation / 'vector_ids.npy',
                         foundation / 'vectors.npy', tensors, config=config)
    arrays, catalog, _, _ = load_tensor_dataset(tensors, config)
    source = root / 'sid-source'
    source.mkdir()
    ids = catalog.ids
    codes = np.stack([ids % 8, (ids // 8) % 8, (ids // 64) % 8], axis=1).astype(np.uint16)
    np.save(source / 'codes.npy', codes)
    np.save(source / 'item_ids.npy', ids)
    build_tables(ids, codes, 8, source)
    (source / 'manifest.json').write_text(json.dumps(dict(
        size=8, layers=3, catalog_items=len(ids), item_ids='item_ids.npy',
    )))
    checkpoint = root / 'best.pt'
    model = TwoTower(catalog, config)
    model.eval()
    torch.save(dict(model=model.state_dict(), config=config.__dict__,
                    structure='mlp_mean', is_best=True,
                    training_state=dict(data_identity=sha256_file(tensors / 'manifest.json'))),
               checkpoint)
    return root, tensors, source, checkpoint, model, arrays, catalog


def test_best_checkpoint_exports_aligned_fp16_full_and_cold(inputs):
    from lsvd.generator_data import ItemVectors, export_item_vectors, load_external_tower

    root, tensors, _, checkpoint, model, _, catalog = inputs
    output = root / 'vectors'
    export_item_vectors(tensors, checkpoint, output, batch_size=128)
    pack = ItemVectors(output, sha256_file(output / 'manifest.json'))
    np.testing.assert_array_equal(pack.ids, catalog.ids)
    np.testing.assert_array_equal(pack.has_id[[0, 798, 799, 800, 899]],
                                  [True, True, False, False, False])
    ids = catalog.ids[[0, 100, 799, 800]]
    features = catalog.features(ids)
    with torch.no_grad():
        full, cold = (model.item(features, cold=mode).numpy() for mode in (False, True))
    assert pack.full.dtype == pack.cold.dtype == np.float16
    np.testing.assert_allclose(pack.full[ids[:2]], full[:2], atol=3e-4, rtol=0)
    np.testing.assert_allclose(pack.cold[ids], cold, atol=3e-4, rtol=0)
    assert not np.allclose(full[2], cold[2])
    np.testing.assert_array_equal(pack.full[ids[2:]], pack.cold[ids[2:]])
    external, external_catalog, _ = load_external_tower(tensors, checkpoint)
    assert external.training is False
    np.testing.assert_array_equal(external_catalog.id_index, catalog.id_index)
    torch.testing.assert_close(external.item.ids.weight, model.item.ids.weight)
    with pytest.raises(ValueError, match='manifest sha256'):
        ItemVectors(output, '0' * 64)


def test_checkpoint_identity_and_best_are_required(inputs):
    from lsvd.generator_data import export_item_vectors

    root, tensors, _, checkpoint, _, _, _ = inputs
    saved = torch.load(checkpoint, weights_only=True)
    saved['training_state']['data_identity'] = '0' * 64
    wrong_data = root / 'wrong-data.pt'
    torch.save(saved, wrong_data)
    with pytest.raises(ValueError, match='tensor dataset identity'):
        export_item_vectors(tensors, wrong_data, root / 'wrong-data')
    saved['training_state']['data_identity'] = sha256_file(tensors / 'manifest.json')
    saved['is_best'] = False
    not_best = root / 'not-best.pt'
    torch.save(saved, not_best)
    with pytest.raises(ValueError, match='best checkpoint'):
        export_item_vectors(tensors, not_best, root / 'not-best')


def test_sid_tracks_main_answer_counts_and_refuses_asset_tampering(inputs, tmp_path):
    from lsvd.generator_data import SIDBundle, prepare_sid_bundle

    _, tensors, source, _, _, arrays, _ = inputs
    output = tmp_path / 'sid'
    prepare_sid_bundle(source, tensors, output)
    sid = SIDBundle(output)
    assert sid.size == 8
    assert int(sid.answer_counts.sum()) == 239
    assert int(sid.item_answer_counts.sum()) == 239
    for pos, accepted in [(0, False), (1, True), (5, False)]:
        row = arrays['event_catalog_row'][pos]
        assert bool(sid.item_answer_counts[row]) is accepted
    assert os.stat(source / 'codes.npy').st_ino == os.stat(output / 'codes.npy').st_ino
    with (output / 'answer_counts.npy').open('ab') as handle:
        handle.write(b'x')
    with pytest.raises(ValueError, match='sha256.*answer_counts'):
        SIDBundle(output)


def test_online_history_uses_strict_past_and_main_only_answers(inputs, tmp_path):
    from lsvd.generator_data import GeneratorData, export_item_vectors, prepare_sid_bundle

    _, tensors, source, checkpoint, _, arrays, _ = inputs
    sid, vectors = tmp_path / 'sid', tmp_path / 'vectors'
    prepare_sid_bundle(source, tensors, sid)
    export_item_vectors(tensors, checkpoint, vectors, batch_size=128)
    data = GeneratorData(tensors, sid, vectors, max_history=3)
    assert len(data.main_positions) == 239
    positions = np.array([1, 6, 41])
    history, codes, target, ids, positives, q, weights = data.training_batch(positions)
    known_rows = arrays['event_catalog_row'][[0, 3, 4, 5, 40]]
    assert history['valid'].tolist() == [[True, False, False], [True] * 3, [True, False, False]]
    assert history['position'][history['valid']].tolist() == [0, 0, 1, 2, 0]
    assert history['place'][history['valid']].tolist() == [1, 1, 1, 3, 1]
    assert history['gap'][history['valid']].tolist() == [1, 3, 2, 2, 1]
    assert history['ratio'][history['valid']].tolist() == pytest.approx([1.2] * 5)
    assert history['feedback'][1, 2].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    np.testing.assert_array_equal(history['codes'][history['valid']], data.sid.codes[known_rows])
    np.testing.assert_array_equal(history['cold'][history['valid']], data.vectors.cold[known_rows])
    np.testing.assert_array_equal(codes, data.sid.codes[arrays['event_catalog_row'][positions]])
    np.testing.assert_array_equal(target, data.vectors.full[arrays['event_catalog_row'][positions]])
    np.testing.assert_array_equal(ids, arrays['event_item'][positions])
    assert q.tolist() == pytest.approx([1 / 239] * 3)
    assert weights.tolist() == [1.] * 3
    # The all-place pos=5 remains a positive, even though it is absent from generator results.
    mask_items = torch.as_tensor(np.asarray(arrays['event_item'][[1, 5, 41]]))
    assert not positives.index.batch(torch.tensor([0, 0, 1])).allowed(mask_items)[0, 1]
    with pytest.raises(ValueError, match='main-ruler'):
        data.training_batch([0, 5])
    query = data.query_batch([0, 9999])
    assert query['valid'].tolist() == [[True] * 3, [False] * 3]
    assert query['gap'][0].tolist() == [8, 7, 7]
    np.testing.assert_array_equal(query['codes'][0], data.sid.codes[arrays['event_catalog_row'][37:40]])


def test_tensor_history_context_preserves_raw_ratio_and_catalog_alignment(inputs):
    _, tensors, _, _, model, arrays, _ = inputs
    _, _, store, _ = load_tensor_dataset(tensors, model.config)
    history = store.history_batch([6, 40], [0, 1], [3, 0], 3, 'cpu', context_only=True)
    assert history['valid'].tolist() == [[True, True, True], [False, False, False]]
    assert history['ratio'][0].tolist() == pytest.approx([1.2] * 3)
    assert history['ratio'][1].tolist() == [0.] * 3
    assert history['place'][0].tolist() == [1, 1, 3]
    assert history['gap'][0].tolist() == [3, 2, 2]
    np.testing.assert_array_equal(history['catalog_rows'][0], arrays['event_catalog_row'][3:6])
    assert 'item_features' not in history


def test_module_entry_rejects_unknown_subcommand_and_emits_dry_receipt():
    common = [sys.executable, '-m', 'lsvd.generator_data']
    invalid = subprocess.run([*common, 'unknown'], capture_output=True, text=True)
    assert invalid.returncode != 0
    assert 'invalid choice' in invalid.stderr
    dry = subprocess.run([*common, 'export-vectors', '--tensor-dir', 'dry-tensors',
                          '--checkpoint', 'dry-model', '--output', 'dry-output', '--dry-run'],
                         capture_output=True, text=True, check=True)
    assert json.loads(dry.stdout)['action'] == 'dry_run'


def test_fit_sid_entry_accepts_versioned_cold_vectors_and_feeds_prepare(inputs, tmp_path):
    from lsvd.generator_data import GeneratorData, ItemVectors, SIDBundle, export_item_vectors, prepare_sid_bundle

    _, tensors, _, checkpoint, _, arrays, _ = inputs
    vectors, raw, prepared = (tmp_path / name for name in ('vectors', 'raw', 'prepared'))
    export_item_vectors(tensors, checkpoint, vectors, batch_size=128)
    vector_manifest = vectors / 'manifest.json'
    valid_hash = sha256_file(vector_manifest)
    common = [sys.executable, '-m', 'lsvd.generator_data', 'fit-sid',
              '--tensor-dir', str(tensors), '--item-vectors', str(vectors),
              '--output', str(raw), '--size', '4', '--fit-items', '32', '--threads', '1',
              '--item-vectors-sha256']
    wrong = subprocess.run([*common, '0' * 64], capture_output=True, text=True)
    assert wrong.returncode != 0
    assert 'manifest sha256 mismatch' in wrong.stderr
    assert not raw.exists()
    run = subprocess.run([*common, valid_hash], capture_output=True, text=True, check=True)
    receipt = json.loads(run.stdout.splitlines()[-1])
    manifest = json.loads((raw / 'manifest.json').read_text())
    assert receipt['action'] == 'fit_sid'
    assert receipt['manifest_sha256'] == sha256_file(raw / 'manifest.json')
    assert manifest['source'] == 'tower-cold'
    assert manifest['item_vectors_sha256'] == valid_hash
    assert manifest['input_sha256'] == sha256_file(vectors / 'cold.npy')
    fit_rows = np.load(raw / 'fit_rows.npy', allow_pickle=False)
    assert len(np.unique(fit_rows)) == len(fit_rows) == 32
    assert np.all(arrays['catalog_train_count'][fit_rows] > 0)
    assert manifest['fit_rows_sha256'] == sha256_file(raw / 'fit_rows.npy')
    np.testing.assert_array_equal(fit_rows, np.sort(np.random.default_rng(42).choice(
        np.arange(800), size=32, replace=False)))
    prepare_sid_bundle(raw, tensors, prepared, expected_main_answers=239)
    bundle = SIDBundle(prepared)
    assert bundle.size == 4
    assert bundle.codes.shape == (900, 3)
    assert bundle.manifest['item_vectors_sha256'] == valid_hash
    assert int(bundle.answer_counts.sum()) == 239
    GeneratorData(tensors, bundle, vectors)
    incompatible = ItemVectors(vectors)
    incompatible.sha256 = 'f' * 64
    with pytest.raises(ValueError, match='different frozen item-vector version'):
        GeneratorData(tensors, bundle, incompatible)
