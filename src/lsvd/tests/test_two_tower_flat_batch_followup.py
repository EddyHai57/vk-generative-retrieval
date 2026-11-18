import unittest
import numpy as np
from lsvd.two_tower import TowerConfig
from lsvd.tests.test_two_tower_round19 import store_fixture


class FlatBatchPackingTests(unittest.TestCase):
    def test_flat_shuffle_uses_exact_original_target_order(self):
        store=store_fixture()
        config=TowerConfig(packed_window=True,batch_size=128,shuffle_all_targets=True)
        batches=list(store.packed_batches(config,np.random.default_rng(42)))
        expected=np.random.default_rng(42).permutation(len(store.samples))
        np.testing.assert_array_equal(np.concatenate(batches),expected)
        self.assertEqual([len(x) for x in batches],[128,128,128,76])


if __name__=='__main__':
    unittest.main()
