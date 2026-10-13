"""Exact IndexFlatIP reference, run explicitly in the accepted P2 FAISS env."""
import unittest
import numpy as np
from vkgr.baselines import brute_topk,blockwise_topk
try:
    import faiss
except ImportError:
    faiss=None

@unittest.skipIf(faiss is None,'FAISS reference uses isolated server P2 environment')
class FaissReferenceTests(unittest.TestCase):
    def test_indexflat_literal_oracle_with_canonical_ties(self):
        ids=np.array([9,2,7,1,8]);items=np.array([[1,0],[1,0],[0,1],[1,0],[0,0]],dtype=np.float32)
        queries=np.array([[1,0],[0,0],[0,1]],dtype=np.float32)
        expected=[[1,2,9,7,8],[1,2,7,8,9],[7,1,2,8,9]]
        index=faiss.IndexFlatIP(2);index.add(items)
        scores,positions=index.search(queries,len(ids))
        # FAISS raw ties have no item-ID contract. For the tiny oracle, retrieve
        # every toy item then apply the frozen tie rule, including boundary ties.
        actual=[ids[pos[np.lexsort((ids[pos],-score))]].tolist() for score,pos in zip(scores,positions)]
        self.assertEqual(actual,expected)
        self.assertEqual(brute_topk(queries,items,ids,3).tolist(),[x[:3] for x in expected])
        self.assertEqual(blockwise_topk(queries,items,ids,3,1,2).tolist(),[x[:3] for x in expected])

if __name__=='__main__':unittest.main()
