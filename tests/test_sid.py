import unittest
import numpy as np
from vkgr import sid

class SIDTests(unittest.TestCase):
    def test_balanced_identical_features_and_shuffle(self):
        ids=np.arange(1,18,dtype=np.uint32);features=np.ones((17,64),np.float32)/8
        model=sid.fit(ids,features,branching=2,depth=3,seed=42,max_iter=10)
        self.assertEqual(model.ids.tolist(),ids.tolist())
        self.assertEqual(model.audit()['full_sid_collisions'],0)
        self.assertEqual(model.audit()['max_leaf_occupancy'],3)
        self.assertEqual(model.audit()['balanced_nodes'],True)
        other=sid.fit(ids[::-1],features[::-1],branching=2,depth=3,seed=42,max_iter=10)
        np.testing.assert_array_equal(model.sids,other.sids)
        for item in ids:self.assertEqual(model.decode(model.encode(int(item))),int(item))
        self.assertEqual(model.allowed_next(()),[0,1])
        with self.assertRaises(ValueError):model.encode(999)
        with self.assertRaises(ValueError):model.allowed_next((9,))
        path=model.encode(1)
        self.assertEqual(model.allowed_next(path),[])
        with self.assertRaises(ValueError):model.decode(path[:3])
        with self.assertRaises(ValueError):model.allowed_next(path+(0,))
        with self.assertRaises(ValueError):model.decode(path[:3]+(99,))

if __name__=='__main__':unittest.main()

class SIDBoundaryTests(unittest.TestCase):
    def test_small_catalog_zero_and_invalid_features(self):
        model=sid.fit([9],np.zeros((1,64),np.float32))
        self.assertEqual(model.encode(9),(0,0,0,0))
        self.assertEqual(model.allowed_next(()),[0])
        self.assertEqual(model.allowed_next((0,0,0)),[0])
        for ids,features in [([1,1],np.zeros((2,64),np.float32)),([1],np.ones((1,63),np.float32)),([1],np.full((1,64),np.nan,np.float32)),([1],np.ones((1,64),np.float32))]:
            with self.assertRaises(ValueError):sid.fit(ids,features)
        with self.assertRaises(ValueError):model.allowed_next((False,))
    def test_semantic_hit_does_not_equal_item_hit(self):
        self.assertEqual(sid.collision_hits((0,0,0,0),(0,0,0,1)),{'semantic_hit':True,'item_hit':False})
        self.assertEqual(sid.collision_hits((0,0,0,1),(0,0,0,1)),{'semantic_hit':True,'item_hit':True})
    def test_geometry_and_nonempty_balanced_nodes(self):
        features=np.zeros((6,64),np.float32);features[:3,0]=1;features[3:,0]=-1
        model=sid.fit([6,5,4,3,2,1],features,branching=2,depth=1)
        self.assertEqual(len(set(model.sids[:3,0])),1)
        self.assertEqual(len(set(model.sids[3:,0])),1)
        self.assertNotEqual(int(model.sids[0,0]),int(model.sids[-1,0]))
        self.assertEqual(model.nodes[0]['sizes'],[3,3])
