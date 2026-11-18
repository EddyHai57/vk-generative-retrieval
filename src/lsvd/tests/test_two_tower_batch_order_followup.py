import unittest
from dataclasses import replace
import numpy as np
from lsvd.two_tower import TowerConfig
from lsvd.tests.test_two_tower_round19 import store_fixture


class BalancedBatchOrderTests(unittest.TestCase):
    def test_whole_batch_shuffle_changes_only_update_order(self):
        store=store_fixture()
        config=TowerConfig(packed_window=True,max_user_targets=8,batch_size=128,
                           balance_user_batches=True)
        ordered=list(store.packed_batches(config,np.random.default_rng(42)))
        shuffled=list(store.packed_batches(replace(config,shuffle_balanced_batches=True),
                                           np.random.default_rng(42)))
        self.assertEqual(sorted(tuple(x) for x in ordered),sorted(tuple(x) for x in shuffled))
        self.assertNotEqual([tuple(x) for x in ordered],[tuple(x) for x in shuffled])


if __name__=='__main__':
    unittest.main()
