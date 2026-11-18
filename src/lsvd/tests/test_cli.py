import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import duckdb

from lsvd.tests import test_data


class FoundationCliTests(unittest.TestCase):
    def run_cli(self, *arguments):
        return subprocess.run(
            [sys.executable, '-m', 'lsvd', *map(str, arguments)],
            cwd=Path(__file__).resolve().parents[3],
            env=os.environ.copy(), capture_output=True, text=True,
        )

    def test_help_and_invalid_command_use_module_entrypoint(self):
        help_result = self.run_cli('--help')
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn('build', help_result.stdout)
        invalid = self.run_cli('unknown-command')
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn('invalid choice', invalid.stderr)

    def test_synthetic_build_evaluate_and_paired_chain(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw, freeze = test_data.FoundationDataTests().fixture(directory)
            foundation = directory / 'foundation'
            dataset = foundation / 'dataset'
            result = self.run_cli('build', '--raw-root', raw, '--frozen-root', freeze,
                                  '--output', dataset, '--train-weeks', '0',
                                  '--memory-limit', '256MB', '--threads', '1')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((dataset / 'foundation.json').exists())
            result = self.run_cli('verify', '--foundation', dataset, '--output', directory / 'verification',
                                  '--memory-limit', '256MB')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((directory / 'verification/verification.json').exists())
            predictions = directory / 'predictions.npy'
            np.save(predictions, np.array([[106,108],[108,106],[2**32-1,2**32-1]],dtype=np.uint32))
            scores = directory / 'scores'
            result = self.run_cli('evaluate', '--foundation', foundation,
                                  '--predictions', predictions, '--model-name', 'synthetic',
                                  '--output', scores, '--replicates', '16')
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((scores / 'metrics.json').read_text())
            main = next(row for row in metrics if row['ruler']=='main'
                        and row['place_scope']=='0/1' and row['seen_scope']=='all'
                        and row['breakdown']=='all' and row['k']==200)
            self.assertEqual(main['macro_capped'], 1)
            paired = directory / 'paired'
            result = self.run_cli('paired', '--first', scores, '--second', scores,
                                  '--output', paired, '--replicates', '16')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((paired / 'metrics.csv').exists())
            for name in ['global_popularity','last_author_popularity']:
                result = self.run_cli('baselines', '--foundation', foundation,
                                      '--model', name, '--output', directory / name)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((directory / name / 'predictions.npy').exists())
            graph = directory / 'graph.parquet'
            connection = duckdb.connect()
            connection.execute(f'''COPY(SELECT 0::UINTEGER AS anchor_item_id,
                0::UINTEGER AS neighbor_item_id,0::BIGINT AS shared_users,0::DOUBLE AS score
                WHERE FALSE) TO '{graph}' (FORMAT PARQUET)''')
            connection.close()
            result = self.run_cli('baselines', '--foundation', dataset, '--model', 'covisitation',
                                  '--graph', graph, '--output', directory / 'covisitation')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(np.all(np.load(directory / 'covisitation/predictions.npy') == 2**32-1))
            vector_ids = directory / 'ids.npy'
            vectors = directory / 'vectors.npy'
            centroids = directory / 'centroids.npy'
            np.save(vector_ids, np.arange(101,109,dtype=np.uint32))
            np.save(vectors, np.tile(np.eye(2,dtype=np.float32),(4,1)))
            np.save(centroids, np.eye(2,dtype=np.float32))
            result = self.run_cli('baselines', '--foundation', dataset, '--model', 'content_mean',
                                  '--vector-ids', vector_ids, '--vectors', vectors,
                                  '--centroids', centroids, '--output', directory / 'content')
            self.assertEqual(result.returncode, 0, result.stderr)
            content = np.load(directory / 'content/predictions.npy')
            self.assertEqual(content.shape,(3,1000))
            self.assertTrue(np.all(content[2] == 2**32-1))


if __name__=='__main__':
    unittest.main()
