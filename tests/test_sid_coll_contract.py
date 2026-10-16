import numpy as np
import unittest
import tempfile
from pathlib import Path
class Raise:
    def raises(self, exc): return unittest.TestCase().assertRaises(exc)
pytest=Raise()
from vkgr import sid_coll_v1 as coll
from vkgr import sid_coll_export as ex
from vkgr import sid_tok_audit as au
from vkgr import sid_v2


def fixture():
    x=np.random.default_rng(42).normal(size=(12,64)).astype('float32')
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    ids=np.arange(12,dtype='uint32')
    m={'schema':'coll-input-v1','status':'accepted','variant':'C1','dimension':64,'engineering':True,
       'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(x),
       'content_feature_sha256':au.feature_digest(x),'train_sequence_sha256':'a'*64,
       'script_sha256':'b'*64,'commit':'c'*40,'checkpoint_sha256':None,'source_files':{'src/vkgr/sid_coll_sgns.py':'b'*64}}
    return ids,x,m


def test_h5_both_and_dimension_and_unit_rows():
    ids,x,m=fixture()
    coll.validate_input(ids,x,m,engineering=True)
    for field,value in [('vectors_sha256','f'*64),('dimension',128),('train_sequence_sha256',None),('status','complete')]:
        bad={**m,field:value}
        with pytest.raises(ValueError):coll.validate_input(ids,x,bad,engineering=True)
    with pytest.raises(ValueError):coll.validate_input(ids,x*2,m,engineering=True)
    with pytest.raises(ValueError):coll.validate_input(ids,x,m,engineering=False)


def test_c2_composition_and_rejection():
    content=np.zeros((3,64),dtype='float32');content[:,1]=2
    p=np.zeros_like(content);p[0,0]=3;p[1,0]=4
    x,report=ex.compose_c2(p,content,np.array([True,False,True]))
    np.testing.assert_allclose(x[0,[0,65]],2**-.5)
    assert not x[1,:64].any() and x[1,65]==1 and x[2,65]==1
    assert report['supported_zero_items']==1
    for bad in [np.full_like(p,np.nan),np.zeros((3,128),dtype='float32')]:
        with pytest.raises(ValueError):ex.compose_c2(bad,content,np.ones(3,dtype=bool))
    with pytest.raises(ValueError):ex.compose_c2(p,content*0,np.ones(3,dtype=bool))


def test_bundle_load_recompute_replay_and_tamper(tmp_path):
    from vkgr import sid_coll_run as run
    ids,x,m=fixture();cfg=run.expected_config('C1')
    result=run.fit_mapping(ids,x,m,cfg,engineering=True)
    dest=tmp_path/'bundle'
    coll.write_bundle(dest,ids=ids,sids=result['sids'],vectors=x,input_manifest=m,
       config=cfg,config_sha='d'*64,engineering=True,producer={'src/vkgr/sid_coll_run.py':'b'*64})
    from vkgr.data_audit import file_sha256
    pin=file_sha256(dest/'run_manifest.json')
    loaded=coll.load_bundle(dest/'run_manifest.json',pin,catalog_ids=ids,vectors=x,input_manifest=m,engineering=True)
    assert loaded.info['sid_version']=='COLL_C1_B256D3_S42'
    assert run.verify_mapping(loaded,ids,x,m,cfg,'d'*64,engineering=True)['status']=='verified'
    bad=x.copy();bad[0]*=-1
    with pytest.raises(ValueError):coll.load_bundle(dest/'run_manifest.json',pin,catalog_ids=ids,vectors=bad,input_manifest=m,engineering=True)
    (dest/'mapping.npz').write_bytes(b'bad')
    with pytest.raises(ValueError):coll.load_bundle(dest/'run_manifest.json',pin,catalog_ids=ids,vectors=x,input_manifest=m,engineering=True)

class ContractTests(unittest.TestCase):
    def test_input(self): test_h5_both_and_dimension_and_unit_rows()
    def test_compose(self): test_c2_composition_and_rejection()
    def test_bundle(self):
        with tempfile.TemporaryDirectory() as tmp: test_bundle_load_recompute_replay_and_tamper(Path(tmp).resolve())

class FrozenAlgorithmTests(unittest.TestCase):
    def test_dimension_extension_is_exact_64d_original_and_128d_replay(self):
        from vkgr import sid_coll_run as run, sid
        ids=np.arange(300,dtype='uint32');rng=np.random.default_rng(11)
        x=rng.normal(size=(len(ids),64)).astype('float32');x/=np.linalg.norm(x,axis=1,keepdims=True)
        old=sid.fit(ids,x,branching=256,depth=3,seed=42,max_iter=10)
        new=run.fit_bhkm_coll(ids,x,branching=256,depth=3,seed=42,max_iter=10)
        np.testing.assert_array_equal(old.sids,new.sids);self.assertEqual(old.nodes,new.nodes)
        x128=np.concatenate((x,x),axis=1)/np.float32(2**.5)
        a=run.fit_bhkm_coll(ids,x128,branching=256,depth=3,seed=42,max_iter=10)
        b=run.fit_bhkm_coll(ids,x128,branching=256,depth=3,seed=42,max_iter=10)
        np.testing.assert_array_equal(a.sids,b.sids)
        with self.assertRaises(ValueError):run.fit_bhkm_coll(ids,x[:,:32])

    def test_negative_unchanged_B1_and_H1(self):
        from vkgr import sid_coll_run as run
        ids,x,m=fixture();result=run.fit_mapping(ids,x,m,run.expected_config('C1'),engineering=True)
        bad=result['sids'].copy();bad[:,0]=0
        audit,gates=coll.audit_gates(ids,bad,x,m,engineering=True)
        self.assertFalse(gates['passed']);self.assertFalse(next(c['ok'] for c in gates['checks'] if c['id']=='B1'))
