import importlib
from pathlib import Path
import tempfile
import unittest

import duckdb
import numpy as np

from lsvd import data
from lsvd.tests import test_data


class VerificationTests(unittest.TestCase):
    def implementation(self):
        try:
            return importlib.import_module('lsvd.verification')
        except ModuleNotFoundError:
            self.fail('foundation verification is not implemented')

    def test_synthetic_full_verification(self):
        module = self.implementation()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, freeze = test_data.FoundationDataTests().fixture(root)
            dataset = root / 'dataset'
            data.build_foundation(raw, freeze, dataset, train_weeks=(0,),
                                  memory_limit='256MB', threads=1)
            result = module.verify_foundation(dataset, root / 'audit', memory_limit='256MB')
            self.assertEqual(result['status'], 'verified')
            self.assertEqual(result['raw_rows_checked'], 8)
            self.assertEqual(result['sequence_rows_checked'], 5)
            self.assertEqual(result['empty_history_users'], 1)

    def test_corrupt_last_query_item_is_rejected(self):
        module = self.implementation()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, freeze = test_data.FoundationDataTests().fixture(root)
            dataset = root / 'dataset'
            data.build_foundation(raw, freeze, dataset, train_weeks=(0,),
                                  memory_limit='256MB', threads=1)
            connection = duckdb.connect()
            corrupted = root / 'corrupted.parquet'
            connection.execute(f'''COPY(SELECT * REPLACE(999::BIGINT AS last_item_id)
                FROM read_parquet('{dataset}/queries.parquet'))
                TO '{corrupted}' (FORMAT PARQUET)''')
            connection.close()
            (dataset / 'queries.parquet').write_bytes(corrupted.read_bytes())
            with self.assertRaises(ValueError):
                module.verify_foundation(dataset, root / 'audit', memory_limit='256MB')

    def test_legacy_comparison_rejects_missing_row_keys(self):
        module = self.implementation()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, freeze = test_data.FoundationDataTests().fixture(root)
            dataset = root / 'dataset'
            data.build_foundation(raw, freeze, dataset, train_weeks=(0,),
                                  memory_limit='256MB', threads=1)
            legacy = root / 'legacy.duckdb'
            connection = duckdb.connect(str(legacy))
            connection.execute(f'''CREATE TABLE context_events AS SELECT week,ordinal,
                user_id,item_id,main_positive::BOOLEAN AS main_positive
                FROM read_parquet('{dataset}/master/*/*.parquet') WHERE FALSE''')
            connection.close()
            with self.assertRaises(ValueError):
                module.verify_foundation(dataset, root / 'audit', legacy_context=legacy,
                                         memory_limit='256MB')

    def test_query_and_item_array_copies_match_parquet(self):
        module = self.implementation()
        for filename, column, replacement in [('queries.npz','last_author_id',16),
                                               ('items.npz','train_count',999),
                                               ('items.npz','train_count',None),
                                               ('history.npz','user_id',999)]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                raw, freeze = test_data.FoundationDataTests().fixture(root)
                dataset = root / 'dataset'
                data.build_foundation(raw, freeze, dataset, train_weeks=(0,),
                                      memory_limit='256MB', threads=1)
                path = dataset / 'arrays' / filename
                with np.load(path) as archive:
                    values = {key:archive[key] for key in archive.files}
                if replacement is None:
                    del values[column]
                else:
                    values[column][0] = replacement
                np.savez(path, **values)
                with self.assertRaises(ValueError):
                    module.verify_foundation(dataset, root / 'audit', memory_limit='256MB')


if __name__ == '__main__':
    unittest.main()
