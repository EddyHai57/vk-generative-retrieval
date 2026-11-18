import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import duckdb
import numpy as np

from lsvd.two_tower import TowerConfig
from lsvd.two_tower_data import synthetic
from lsvd.two_tower_prepare import build_tensor_dataset, load_tensor_dataset


def foundation_fixture(root):
    catalog, store, data, _ = synthetic(TowerConfig(), user_count=8, item_count=900)
    store.columns['place'][5] = 2
    names = dict(item='item_ids', week='weeks', place='places', main='main_positive',
                 secondary='secondary_positive', feedback='feedback_masks',
                 ratio='watch_ratios')
    connection = duckdb.connect()
    connection.execute('SET threads=1')
    for block in range(32):
        folder = root / 'sequences' / f'user_block={block}'
        folder.mkdir(parents=True)
        values = {target: [store.columns[source][store.offsets[block]:store.offsets[block+1]].tolist()]
                  for source, target in names.items()} if block < 8 else {
                      target: [[]] for target in names.values()}
        where = '' if block < 8 else ' WHERE FALSE'
        columns = ','.join('?::DOUBLE[] AS ' + name if name == 'watch_ratios'
                           else '?::INTEGER[] AS ' + name for name in values)
        destination = "'" + str(folder / 'users.parquet').replace("'", "''") + "'"
        connection.execute('COPY (SELECT ?::INTEGER AS user_id,' + columns
                           + where + ') TO ' + destination + ' (FORMAT PARQUET)',
                           [block, *[row[0] for row in values.values()]])
    items = {k: v for k, v in catalog.columns.items() if k != 'content'}
    items['train_interactions_rank'] = items.pop('rank')
    items['popularity_tier'] = data.items['popularity_tier']
    connection.register('items', items)
    connection.execute('COPY items TO ? (FORMAT PARQUET)', [str(root / 'items.parquet')])
    connection.close()
    arrays = root / 'arrays'
    arrays.mkdir()
    np.savez(arrays / 'queries.npz', user_id=data.user_ids)
    np.savez(arrays / 'history.npz', user_id=data.history_user_ids,
             offsets=data.history_offsets)
    np.savez(arrays / 'targets.npz', **data.targets)
    np.savez(arrays / 'items.npz', **data.items)
    np.save(arrays / 'history_items.npy', data.history_items)
    np.save(root / 'vector_ids.npy', catalog.ids)
    np.save(root / 'vectors.npy', catalog.columns['content'].astype(np.float16))
    return catalog, store


class PreparationTests(unittest.TestCase):
    def test_columnwise_build_preserves_all_place_positives_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'source'
            root.mkdir()
            catalog, store = foundation_fixture(root)
            output = Path(folder) / 'tensors'
            manifest = build_tensor_dataset(root, root / 'vector_ids.npy',
                                            root / 'vectors.npy', output)
            self.assertEqual(manifest['counts']['training_events'], 320)
            self.assertEqual(manifest['counts']['training_targets'], 319)
            self.assertEqual(manifest['counts']['all_place_positives'], 320)
            arrays, loaded_catalog, loaded_store, data = load_tensor_dataset(
                output, TowerConfig()
            )
            self.assertEqual(arrays['event_item'].dtype, np.dtype('int32'))
            self.assertEqual(arrays['event_week'].dtype, np.dtype('uint8'))
            self.assertEqual(arrays['event_ratio'].dtype, np.dtype('float32'))
            self.assertEqual(arrays['target_weight'].dtype, np.dtype('float16'))
            self.assertEqual(arrays['positive_keys'].shape, (320,))
            self.assertTrue(np.all(np.diff(arrays['positive_keys']) > 0))
            np.testing.assert_array_equal(arrays['event_catalog_row'],
                                          catalog.rows(store.columns['item']))
            for value in manifest['arrays'].values():
                self.assertEqual(len(value['sha256']), 64)
                self.assertIn('dtype', value)
                self.assertIn('shape', value)
            self.assertGreaterEqual(len(manifest['sources']), 39)
            self.assertEqual(len(data.user_ids), 8)
            self.assertEqual(len(loaded_catalog.ids), 900)
            self.assertEqual(len(loaded_store.samples), 319)
            with self.assertRaises(FileExistsError):
                build_tensor_dataset(root, root / 'vector_ids.npy',
                                     root / 'vectors.npy', output)

    def test_tensor_entry_runs_real_dry_forward_and_finite_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'source'
            root.mkdir()
            foundation_fixture(root)
            tensors = Path(folder) / 'tensors'
            build_tensor_dataset(root, root / 'vector_ids.npy', root / 'vectors.npy', tensors)
            common = [sys.executable, '-m', 'lsvd.two_tower_run', 'tensors',
                      '--tensor-dataset', str(tensors), '--structures', 'mlp_mean',
                      '--batch-size', '8', '--threads', '1', '--max-train-samples', '64']
            dry = subprocess.run([*common, '--output', str(Path(folder) / 'dry'), '--dry-run'],
                                  capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertTrue((Path(folder) / 'dry/dry_run.json').exists())
            trained = subprocess.run([*common, '--output', str(Path(folder) / 'finite'),
                                      '--max-train-steps', '2'], capture_output=True, text=True)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            record = json.loads((Path(folder) / 'finite/tensor_summary.json').read_text())
            self.assertEqual(record['models'][0]['steps'], 2)


if __name__ == '__main__':
    unittest.main()
