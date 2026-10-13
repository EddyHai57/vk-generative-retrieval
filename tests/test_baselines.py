import unittest
import numpy as np
from vkgr import baselines as b

class BaselinesTests(unittest.TestCase):
    def test_exact_literal_ties_and_blocks(self):
        ids=np.array([9,2,7,1,8]); items=np.array([[1,0],[1,0],[0,1],[1,0],[0,0]],dtype=np.float32)
        q=np.array([[1,0],[0,0],[0,1]],dtype=np.float32)
        expected=[[1,2,9,7,8],[1,2,7,8,9],[7,1,2,8,9]]
        for k in (1,3,10):
            self.assertEqual(b.brute_topk(q,items,ids,k).tolist(),[r[:k] for r in expected])
            for qb in (1,2):
                for ib in (1,2,3):
                    self.assertEqual(b.blockwise_topk(q,items,ids,k,qb,ib).tolist(),[r[:k] for r in expected])

    def test_features_centroid_repeat_zero_and_missing(self):
        ids=[1,2,3]; raw=np.array([[3,0],[-2,0],[0,0]],dtype=np.float16)
        features=b.join_normalized(ids,[3,1,2],raw[[2,0,1]])
        np.testing.assert_array_equal(features,[[1,0],[-1,0],[0,0]])
        q,zeros=b.history_centroids([[1,1,2],[1,2],[3]],ids,features)
        np.testing.assert_array_equal(q,[[1,0],[0,0],[0,0]]);self.assertEqual(zeros,2)
        with self.assertRaises(ValueError):b.join_normalized(ids,[1,2],raw[:2])
        with self.assertRaises(ValueError):b.history_centroids([[4]],ids,features)
        with self.assertRaises(ValueError):b.join_normalized(ids,ids,[[1,0],[float('nan'),0],[0,0]])
    def test_popularity_zero_and_metrics_oov(self):
        self.assertEqual(b.popularity([3,2,1,4],[(1,True),(2,False),(3,True),(1,True)]).tolist(),[1,0,2,0])
        queries=[{'target':1,'oov':False},{'target':99,'oov':True},{'target':3,'oov':False}]
        result=b.metrics([[1,3],[2,1],[1,3]],queries,[1,2,3],(1,2))
        self.assertEqual(result['all']['count'],3);self.assertAlmostEqual(result['all']['recall@2'],2/3)
        self.assertAlmostEqual(result['train_observed']['ndcg@2'],(1+1/np.log2(3))/2)
        self.assertEqual(result['oov']['recall@2'],0)
        empty=b.metrics([[1]],queries[:1],[1,2,3],(1,))['oov']
        self.assertEqual(empty,{'count':0,'recall@1':None,'ndcg@1':None})
    def test_sampling_target_independent(self):
        rows=[{'user_id':u,'target':u,'oov':bool(u%2)} for u in range(10)]
        # Independent literal hashlib ranking, with no target used.
        import hashlib
        expected=sorted(range(10),key=lambda u:(hashlib.sha256(f'42:test:{u}'.encode()).hexdigest(),u))[:4]
        self.assertEqual([x['user_id'] for x in b.sample_queries(rows,'test',42,4)],expected)
        altered=[dict(x,target=999,oov=False) for x in reversed(rows)]
        self.assertEqual([x['user_id'] for x in b.sample_queries(altered,'test',42,4)],expected)
        with self.assertRaises(ValueError):b.sample_queries(rows+rows[:1],'test',42,4)

    def test_identical_64d_vectors_tie_across_partial_blocks(self):
        rng=np.random.default_rng(42)
        queries=b.normalize(rng.normal(size=(2,64)))
        item=b.normalize(rng.normal(size=(1,64)))
        items=np.repeat(item,3,axis=0);ids=np.array([3,2,1])
        expected=[[1],[1]]
        self.assertEqual(b.brute_topk(queries,items,ids,1).tolist(),expected)
        for qb in (1,2,3):
            for ib in (1,2,3,4):
                self.assertEqual(b.blockwise_topk(queries,items,ids,1,qb,ib).tolist(),expected)

if __name__=='__main__':unittest.main()
