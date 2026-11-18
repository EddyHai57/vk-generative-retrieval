import csv
from pathlib import Path
import tempfile
import unittest
from lsvd.batch_followup import replay_distributions
from lsvd.two_tower import TowerConfig
from lsvd.tests.test_two_tower_round19 import store_fixture


class ReplayFlagTests(unittest.TestCase):
    def test_named_replays_ignore_outer_training_shuffle_flags(self):
        config=TowerConfig(packed_window=True,batch_size=128,shuffle_all_targets=True,
                           shuffle_balanced_batches=True)
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'replay'
            replay_distributions(store_fixture(),config,output)
            with (output/'chunk-shuffle-cap8.csv').open() as f:
                first=next(csv.DictReader(f))
            self.assertEqual(int(first['targets']),16)
            self.assertEqual(int(first['distinct_users']),2)


if __name__=='__main__':
    unittest.main()
