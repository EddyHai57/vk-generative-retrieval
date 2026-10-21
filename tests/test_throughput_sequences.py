import tempfile
from pathlib import Path
import unittest
from vkgr.throughput_config import Engineering, manifest
from vkgr.throughput_sequences import SequenceStore

EVENTS=[{'user_id':u,'item_id':i,'position':[w,o],'like':l,'side':[1,i,3],'ctx':[2,1]}
        for u,i,w,o,l in [(2,10,0,0,True),(1,20,0,1,True),(2,11,0,2,False),
                           (2,12,1,0,True),(1,21,1,1,True),(2,13,1,2,True)]]
REFS=[(2,(1,0)),(1,(1,1)),(2,(1,2))]


class SequenceTests(unittest.TestCase):
    def test_columnar_equals_fixture_adapter_without_history_copies(self):
        import numpy as np
        from vkgr.throughput_sequences import position
        columns=SequenceStore.from_columns(np.array([e['user_id']for e in EVENTS]),
            np.array([e['item_id']for e in EVENTS]),np.array([position(e['position'])for e in EVENTS]),
            np.array([e['like']for e in EVENTS]),np.array([e['side']for e in EVENTS]),np.array([e['ctx']for e in EVENTS]),
            [(u,position(p))for u,p in REFS])
        self.assertEqual(columns.rows([2,0,1]),SequenceStore.from_events(EVENTS,REFS).rows([2,0,1]))

    def test_columnar_rejects_lossy_side_context_references(self):
        import numpy as np
        columns=[np.array([1,1]),np.array([3,4]),np.array([0,1]),np.array([True,True]),
                 np.zeros((2,3),dtype=np.int64),np.zeros((2,2),dtype=np.int64),np.array([[1,1]])]
        for idx,value in [(4,np.full((2,3),1.5)),(5,np.full((2,2),2**63,dtype=np.uint64)),(6,np.array([[1.,1.5]]))]:
            args=list(columns);args[idx]=value
            with self.assertRaises(ValueError):SequenceStore.from_columns(*args)
    def test_strict_past_order_tail_and_like_masks(self):
        s=SequenceStore.from_events(EVENTS,REFS)
        rows=s.rows([2,0],length=2)
        self.assertEqual(rows[0]['history'],[11,12])
        self.assertEqual(rows[0]['past_likes'],[10,12])
        self.assertEqual(rows[1]['history'],[10,11])
        self.assertEqual(rows[1]['history_positions'],[[0,0],[0,2]])
        self.assertEqual(rows[0]['target'],13)
        self.assertEqual(len(s.rows([1],length=50)),1)

    def test_mmap_bytes_roundtrip_and_unique_output(self):
        s=SequenceStore.from_events(EVENTS,REFS)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve()/'new';s.save(root)
            loaded=SequenceStore.load(root,mmap=True)
            self.assertEqual(loaded.rows([1,0,2]),s.rows([1,0,2]))
            with self.assertRaises(FileExistsError):s.save(root)
            with (root/'items.npy').open('ab')as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):SequenceStore.load(root,mmap=True)

    def test_manifest_does_not_change_science_identity(self):
        science={'seed':42,'config_sha':'abc'}
        m=manifest(science,Engineering())
        self.assertEqual(m['science'],science)
        self.assertNotIn('engineering',science)
        self.assertNotEqual(manifest(science,Engineering(assembly_workers=2))['engineering']['sha256'],m['engineering']['sha256'])
        import hashlib
        path=Path(__file__).parents[1]/'configs/throughput/a40-20261022.yaml'
        loaded=Engineering.load(path)
        self.assertEqual(manifest(science,loaded)['engineering']['source_file_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(loaded.disk_limit_bytes,400000000000)

    def test_invalid_order_future_and_budget(self):
        with self.assertRaises(ValueError):SequenceStore.from_events(EVENTS[::-1],REFS)
        s=SequenceStore.from_events(EVENTS,REFS)
        with self.assertRaises((ValueError,IndexError)):s.rows([99])
        with self.assertRaises(ValueError):SequenceStore.from_events(EVENTS,[(2,(0,0))]).rows([0])
        with self.assertRaises(MemoryError):SequenceStore.from_events(EVENTS,REFS,byte_limit=1)


if __name__=='__main__':unittest.main()
