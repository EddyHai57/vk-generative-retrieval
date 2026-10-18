import unittest
import numpy as np
from vkgr import sid_structure_a1_run as run, coll_sid_a1_run as old, sid_v2

class StructureA1Tests(unittest.TestCase):
    def test_config_only_changes_registered_identity(self):
        a,b=old.default_config(),run.default_config()
        self.assertEqual({k for k in a if a[k]!=b[k]},{'schema','task_id','sid'})
        self.assertEqual(set(b)-set(a),{'sid_manifest_sha256'})
        with self.assertRaises(ValueError):run.validate_config(b)
        b={**b,'sid_manifest_sha256':'a'*64}
        for seed in (42,2024,2025):run.validate_config({**b,'seed':seed})
        for key,value in [('seed',314159),('sid',a['sid']),('beam',50),('microbatch',128),('learning_rate',.001)]:
            with self.assertRaises(ValueError):run.validate_config({**b,key:value})
        self.assertEqual(old.SID_VERSION,'COLL_C1_B256D3_S42')

    def test_content_and_input_pins_stay_frozen(self):
        cfg={**run.default_config(),'sid_manifest_sha256':'a'*64}
        run.check_pins('a'*64,old.INPUT_SHA,old.VECTORS_SHA,False,config=cfg)
        for inp,vec in [('0'*64,old.VECTORS_SHA),(old.INPUT_SHA,'0'*64)]:
            with self.assertRaises(ValueError):run.check_pins('a'*64,inp,vec,False,config=cfg)
        with self.assertRaises(ValueError):run.check_pins('invalid',old.INPUT_SHA,old.VECTORS_SHA,False,config=cfg)

    def test_new_codec_uses_computed_vocabulary_and_same_four_positions(self):
        ids=np.array([1,3,8,20],dtype='uint32')
        sids=np.array([[0,0,0,0],[0,0,0,1],[1,0,0,0],[47,0,0,0]],dtype='uint32')
        flat=sid_v2.build_codec(ids,sids,48)
        fake=sid_v2.LoadedSID(flat,{}, {'branching':48,'depth':3,'suffix_vocabulary':2,'vocab_size':148})
        deep=run._as_depth3(fake)
        self.assertEqual((deep.codec.depth,deep.codec.vocab_size),(3,148))
        np.testing.assert_array_equal(deep.codec.catalog.sids,sids)
        self.assertEqual(deep.info['branching'],48)

if __name__=='__main__':unittest.main()
