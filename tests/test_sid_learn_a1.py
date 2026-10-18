"""S2 A1 keeps the accepted model recipe and changes the isolated SID family."""
import importlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
from vkgr import coll_sid_a1_run as original,sid_learn_v1 as contract,sid_v2,sid_learn_bundle as bundle
from vkgr.data_audit import canonical_bytes,file_sha256
from test_sid_learn_bundle import fixture
ROOT=Path(__file__).resolve().parents[1]
class LearnA1Tests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'src/vkgr/sid_learn_a1_run.py').is_file(),'missing isolated A1 entry');self.run=importlib.import_module('vkgr.sid_learn_a1_run')
    def test_model_recipe_differs_only_sid_schema_task_and_manifest_identity(self):
        old=original.default_config();new=self.run.default_config()
        self.assertEqual({k for k in old if old[k]!=new[k]},{'schema','task_id','sid'});self.assertEqual(set(new)-set(old),{'sid_manifest_sha256'})
        self.assertEqual(new['sid'],contract.SID_VERSION)
        with self.assertRaises(ValueError):self.run.validate_config(new,False)
        for seed in (42,2024,2025):
            cfg={**new,'seed':seed,'sid_manifest_sha256':'a'*64};self.run.validate_config(cfg,False)
            self.run.check_pins('a'*64,original.INPUT_SHA,original.VECTORS_SHA,False,config=cfg)
            with self.assertRaises(ValueError):self.run.check_pins('b'*64,original.INPUT_SHA,original.VECTORS_SHA,False,config=cfg)
        with self.assertRaises(ValueError):self.run.validate_config({**new,'sid_manifest_sha256':'a'*64,'learning_rate':.001},False)
    def test_three_committed_configs_remain_unsealed_before_fit_verified(self):
        for seed in (42,2024,2025):
            cfg=json.loads((ROOT/f'configs/sid_learn/g-learn-a1-s{seed}.json').read_text())
            self.assertEqual(cfg,{**self.run.default_config(),'seed':seed})
            with self.assertRaises(ValueError):self.run.validate_config(cfg,False)
    def test_suffix_vocabulary_is_computed_without_legacy_whitelist_changes(self):
        ids=np.array([1,2,3],dtype='uint32');sids=np.array([[0,0,0,0],[0,0,0,1],[255,255,255,0]],dtype='uint32');flat=sid_v2.build_codec(ids,sids,256)
        loaded=sid_v2.LoadedSID(flat,{}, {'branching':256,'suffix_vocabulary':2,'vocab_size':772,'depth':3})
        converted=self.run._as_depth3(loaded);self.assertEqual(converted.codec.vocab_size,772);np.testing.assert_array_equal(converted.codec.catalog.sids,sids)
        self.assertNotIn(contract.SID_VERSION,sid_v2.SID_VERSIONS);self.assertEqual(original.SID_VERSION,'COLL_C1_B256D3_S42')
    def test_real_loader_calls_new_bundle_and_original_reconstruction_gates(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d).resolve();data=fixture();np.save(root/'content.npy',np.vstack((np.zeros((1,64),dtype='float32'),data['content'])),allow_pickle=False);np.save(root/'vectors.npy',data['c1_vectors'],allow_pickle=False);(root/'input.json').write_bytes(canonical_bytes(data['c1_manifest']))
            bundle.write_bundle(root/'bundle',**data);mp=root/'bundle/run_manifest.json'
            loaded=self.run._load_sid(mp,file_sha256(mp),root/'content.npy',file_sha256(root/'content.npy'),data['ids'],True,root/'input.json',file_sha256(root/'input.json'),root/'vectors.npy',file_sha256(root/'vectors.npy'))
            self.assertEqual(loaded.info['branching'],2);self.assertEqual(loaded.info.get('kind'),'rqvae');self.assertTrue(loaded.manifest['gates']['passed']);self.assertIn('pca',loaded.manifest)
if __name__=='__main__':unittest.main()
