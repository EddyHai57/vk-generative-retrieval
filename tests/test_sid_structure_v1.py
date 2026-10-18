"""Frozen matched-structure bundle/fit contracts; no real data."""
import unittest
from unittest.mock import patch
import numpy as np
from vkgr import sid_structure_v1 as sc, sid_structure_run as sr
from vkgr import sid_coll_v1 as old, sid_coll_run as original
from test_sid_coll_contract import fixture

class StructureContractTests(unittest.TestCase):
    def test_only_new_namespace_and_frozen_recipe(self):
        c=sr.expected_config()
        self.assertEqual((c['branching'],c['depth'],c['seed'],c['max_iter']),(48,3,42,10))
        self.assertEqual(c['sid_version'],'STRUCT_C1_B48D3_S42')
        self.assertNotIn(c['sid_version'],old.SID_VERSIONS)
        self.assertIs(sr.fit_bhkm_coll,original.fit_bhkm_coll)
        for key,value in [('branching',64),('depth',2),('seed',2024),('variant','C2')]:
            with self.assertRaises(ValueError):sr.validate_config({**c,key:value})

    def test_all_three_layers_must_pass_original_integer_usage_rule(self):
        ids,x,im=fixture()
        for used,want in [([48,48,48],True),([39,39,39],True),([48,48,38],False)]:
            audit={'positions':[{'distinct':n} for n in used],'content_sha256':im['vectors_sha256']}
            inherited={'passed':True,'checks':[{'id':'H5','ok':True},{'id':'B1','ok':True}]}
            with patch.object(sc.au,'audit_mapping',return_value=audit),patch.object(sc.au,'gates',return_value=inherited):
                _, gates=sc.audit_gates(ids,np.zeros((len(ids),4),dtype='uint32'),x,im,engineering=True)
            self.assertEqual(gates['passed'],want)
            self.assertEqual(next(c['ok'] for c in gates['checks'] if c['id']=='R1'),want)
        self.assertEqual(old.SID_VERSIONS,('COLL_C1_B256D3_S42','COLL_C2_B256D3_S42'))

    def test_tiny_real_algorithm_fixture_fails_utilisation_without_bypass(self):
        ids,x,im=fixture()
        with self.assertRaisesRegex(ValueError,'FIT_FAILED'):
            sr.fit_mapping(ids,x,im,sr.expected_config(),engineering=True)

    def test_wrong_input_hash_is_rejected(self):
        ids,x,im=fixture()
        with self.assertRaises(ValueError):sc.validate_input(ids,x,{**im,'vectors_sha256':'0'*64},engineering=True)

    def test_verify_stdout_is_bound_to_its_actual_job_and_physical_inputs(self):
        import tempfile,json,os
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            run=Path(d).resolve()/'sid-structure-fit-verify-v1';run.mkdir()
            (run/'run.json').write_text(json.dumps({'run_id':run.name,'config':{'sha256':'c'*64}}))
            with patch.dict(os.environ,VKGR_RUN_ID=run.name):
                proof=sr.verification_bindings({'status':'verified'},output=run/'out',config_sha='c'*64,
                    input_sha='i'*64,vectors_sha='v'*64,catalog_sha='a'*64,engineering=False)
                self.assertEqual(proof['run_id'],run.name)
                self.assertEqual(proof['vectors_file_sha256'],'v'*64)
                self.assertEqual(len(proof['run_manifest_sha256']),64)
                with self.assertRaises(ValueError):sr.verification_bindings({'status':'verified'},output=run/'out',config_sha='x'*64,input_sha='i'*64,vectors_sha='v'*64,catalog_sha='a'*64,engineering=False)

    def test_positive_bundle_roundtrip_at_smallest_whole_tree_usage_boundary(self):
        import tempfile,hashlib
        from pathlib import Path
        from vkgr import sid_v2,sid_tok_audit as au
        count=48*48*39
        ids=np.arange(1,count+1,dtype='uint32');index=np.arange(count)
        codes=np.stack((index//(48*39),(index//39)%48,index%39,np.zeros(count,dtype=int)),axis=1).astype('uint32')
        x=np.random.default_rng(9).normal(size=(count,64)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
        _,_,im=fixture();im={**im,'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(x),'content_feature_sha256':au.feature_digest(x)}
        out=Path(tempfile.mkdtemp(prefix='vkgr-structure-bundle-unit-')).resolve()/'bundle'
        sc.write_bundle(out,ids=ids,sids=codes,vectors=x,input_manifest=im,config=sr.expected_config(),config_sha=sc.producer_identity()['files'][sc.FIT_CONFIG_MEMBER],engineering=True,producer=sc.producer_identity()['files'])
        pin=hashlib.sha256((out/'run_manifest.json').read_bytes()).hexdigest()
        loaded=sc.load_bundle(out/'run_manifest.json',pin,catalog_ids=ids,vectors=x,input_manifest=im,engineering=True)
        self.assertEqual(loaded.info['sid_version'],'STRUCT_C1_B48D3_S42')
        self.assertEqual(loaded.info['branching'],48)
        np.testing.assert_array_equal(loaded.codec.catalog.sids,codes)
        self.assertTrue(loaded.manifest['gates']['passed'])

    def test_formal_physical_pins_reject_before_any_file_read(self):
        from vkgr import sid_structure_a1_run as a1
        frozen=[a1.VECTORS_SHA,'80201c574fad26bd29f2e2425a5ad98db8fe7c2a9be91e07d2fc6aedc48d47f3',a1.INPUT_SHA]
        for which in range(3):
            pins=list(frozen);pins[which]='0'*64
            with patch.object(sr.gd,'_checked_file',side_effect=AssertionError('opened unregistered input')) as io:
                with self.assertRaisesRegex(ValueError,'frozen C1 physical'):
                    sr.load_inputs('/synthetic/vectors.npy',pins[0],'/synthetic/catalog.json',pins[1],'/synthetic/input-manifest.json',pins[2],engineering=False)
                io.assert_not_called()

    def test_exact_producer_members_bytes_and_runtime_are_not_self_asserted(self):
        import copy
        identity=sc.producer_identity();base={'producer':identity['files'],'producer_runtime':identity['runtime'],
            'fit_config_sha256':identity['files'][sc.FIT_CONFIG_MEMBER]}
        sc.check_producer_identity(base)
        for mode in ('extra','missing','changed','python','dependency','lock','config'):
            bad=copy.deepcopy(base)
            if mode=='extra':bad['producer']['unregistered.py']='a'*64
            if mode=='missing':bad['producer'].pop(next(iter(bad['producer'])))
            if mode=='changed':bad['producer'][next(iter(bad['producer']))]='0'*64
            if mode=='python':bad['producer_runtime']['python']='0.0'
            if mode=='dependency':bad['producer_runtime']['exact_dependencies']['numpy']='0.0'
            if mode=='lock':bad['producer_runtime']['lock_sha256']='0'*64
            if mode=='config':bad['fit_config_sha256']='0'*64
            with self.assertRaises(ValueError):sc.check_producer_identity(bad)

if __name__=='__main__':unittest.main()
