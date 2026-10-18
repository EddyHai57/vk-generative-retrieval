"""Hand-built engineering state tests bundle gates without scientific fitting."""
import copy,hashlib,importlib,io,itertools,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from vkgr import sid_rqvae as rq,sid_v2,sid_learn_v1 as contract,sid_learn_pca as pca,sid_coll_sgns as sg,sid_tok_audit as au
from vkgr.data_audit import canonical_bytes,file_sha256
ROOT=Path(__file__).resolve().parents[1]
def npy_sha(x):
    b=io.BytesIO();np.save(b,x,allow_pickle=False);return hashlib.sha256(b.getvalue()).hexdigest()
def fixture():
    ids=np.arange(1,9,dtype='uint32');content=np.zeros((8,64),dtype='float32');content[:,:3]=np.array(list(itertools.product((0,1),repeat=3)),dtype='float32')*np.array([4,2,1],dtype='float32')
    model=rq.RQVAE(2)
    with torch.no_grad():
        for value in model.parameters():value.zero_()
        for seq in (model.encoder,model.decoder):
            for layer in seq:
                if isinstance(layer,torch.nn.Linear):layer.weight[:32,:32]=torch.eye(32)
        for level,scale in enumerate((4,2,1)):model.codebooks[level][1,level]=scale
    state={'K':2,'state_dict':{k:v.detach().clone() for k,v in model.state_dict().items()}}
    c1=np.random.default_rng(41).normal(size=(8,64)).astype('float32');c1/=np.linalg.norm(c1,axis=1,keepdims=True);c1[-1]=content[-1]/np.linalg.norm(content[-1])
    histories={0:list(range(1,8)),1:[8]};fallback=contract.recompute_fallback_mask(ids,histories)
    im={'schema':'coll-input-v1','status':'accepted','variant':'C1','dimension':64,'engineering':True,'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(c1),'content_feature_sha256':au.feature_digest(content),'train_sequence_sha256':sg._sequence_digest(histories)}
    physical={'content':npy_sha(np.vstack((np.zeros((1,64),dtype='float32'),content))),'catalog':sid_v2.catalog_digest(ids),'c1_vectors':npy_sha(c1),'c1_manifest':hashlib.sha256(canonical_bytes(im)).hexdigest()}
    fit={'status':'complete','step':3,'max_steps':3,'elapsed':.1,'wall_cap_seconds':14400,'K':2,'seed':42,'batch_size':8,'engineering':True,'state':state,'init_digest':rq.state_digest(state['state_dict']),'final_state_digest':rq.state_digest(state['state_dict']),'loss_log':[],'usage_log':[],'distance_log':[],'reset_log':[]}
    return dict(ids=ids,content=content,c1_vectors=c1,c1_manifest=im,physical=physical,pca_state=pca.fit_pca(c1),fallback=fallback,fit=fit,config=contract.expected_config(),engineering=True)
class LearnBundleTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'src/vkgr/sid_learn_bundle.py').is_file(),'missing learn bundle entry');self.mod=importlib.import_module('vkgr.sid_learn_bundle')
    def bundle(self,root):
        inputs=fixture();path=Path(root).resolve()/'bundle';m=self.mod.write_bundle(path,**inputs)
        args={k:inputs[k] for k in ('content','c1_vectors','c1_manifest','physical','engineering')};args['catalog_ids']=inputs['ids'];args['fallback']=inputs['fallback']
        return path,m,inputs,args
    def test_positive_engineering_bundle_recomputes_all_original_gates_and_exports(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path,m,inputs,args=self.bundle(d);loaded=self.mod.load_bundle(path/'run_manifest.json',file_sha256(path/'run_manifest.json'),**args)
            self.assertTrue(m['gates']['passed']);self.assertEqual(loaded.info['branching'],2);self.assertEqual(loaded.info['vocab_size'],9)
            expected=rq.assign_fourth_code(inputs['ids'],rq.export_codes(inputs['fit']['state'],inputs['content']));np.testing.assert_array_equal(loaded.codec.catalog.sids,expected)
            self.assertNotIn('targets.npy',{p.name for p in path.iterdir()});self.assertNotIn(contract.SID_VERSION,sid_v2.SID_VERSIONS)
            with self.assertRaises(ValueError):self.mod.load_bundle(path/'run_manifest.json',file_sha256(path/'run_manifest.json'),**{**args,'engineering':False})
    def test_formal_physical_sources_are_fixed(self):
        good=dict(self.mod.FORMAL_PHYSICAL);self.mod.validate_physical(good,False)
        for key in good:
            with self.subTest(key=key),self.assertRaises(ValueError):self.mod.validate_physical({**good,key:'0'*64},False)
    def test_incomplete_fit_writes_no_directory(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            inputs=fixture();inputs['fit']['status']='fail_wall';path=Path(d).resolve()/'bundle'
            with self.assertRaises(ValueError):self.mod.write_bundle(path,**inputs)
            self.assertFalse(path.exists())
    def test_failed_original_gate_retains_structured_audit_without_bundle(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            inputs=fixture()
            for key,value in inputs['fit']['state']['state_dict'].items():
                if key.startswith('decoder.'):value.zero_()
            inputs['fit']['final_state_digest']=rq.state_digest(inputs['fit']['state']['state_dict'])
            path=Path(d).resolve()/'bundle'
            with self.assertRaises(ValueError) as caught:self.mod.write_bundle(path,**inputs)
            self.assertFalse(path.exists());self.assertTrue(hasattr(caught.exception,'gates'))
            self.assertFalse(next(row for row in caught.exception.gates['checks'] if row['id']=='R3')['ok'])

    def test_producer_members_bytes_python_dependency_and_lock_are_not_self_asserted(self):
        for mode in ('extra','missing','bytes','python','dependency','lock'):
            with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                path,m,inputs,args=self.bundle(d);bad=copy.deepcopy(m)
                if mode=='extra':bad['producer']['unregistered.py']='a'*64
                if mode=='missing':bad['producer'].pop(next(iter(bad['producer'])))
                if mode=='bytes':bad['producer'][next(iter(bad['producer']))]='0'*64
                if mode=='python':bad['producer_runtime']['python']='0.0'
                if mode=='dependency':bad['producer_runtime']['exact_dependencies']['numpy']='0.0'
                if mode=='lock':bad['producer_runtime']['lock_sha256']='0'*64
                (path/'run_manifest.json').write_bytes(canonical_bytes(bad))
                with self.subTest(mode=mode),self.assertRaises(ValueError):self.mod.load_bundle(path/'run_manifest.json',file_sha256(path/'run_manifest.json'),**args)
    def test_manifest_numeric_types_and_commit_format_are_strict(self):
        for key,value in (('suffix_vocabulary',True),('catalog_items',8.0),('vocab_size',9.0),('source_commit','z'*40)):
            with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                path,m,inputs,args=self.bundle(d);m[key]=value;(path/'run_manifest.json').write_bytes(canonical_bytes(m))
                with self.subTest(key=key),self.assertRaises(ValueError):self.mod.load_bundle(path/'run_manifest.json',file_sha256(path/'run_manifest.json'),**args)

    def test_fit_timing_values_have_numeric_types_before_publication(self):
        for key,value in (('elapsed',True),('wall_cap_seconds',14400.0)):
            with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                inputs=fixture();inputs['fit'][key]=value;path=Path(d).resolve()/'bundle'
                with self.subTest(key=key),self.assertRaises(ValueError):self.mod.write_bundle(path,**inputs)
                self.assertFalse(path.exists())

    def test_rehashed_wrong_weights_still_fail_mapping_binding(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path,m,inputs,args=self.bundle(d);arrays=sid_v2._load_npz(path/'weights.npz');arrays['codebooks.0'][1,0]=0;sid_v2._write_npz(path/'weights.npz',arrays)
            m['artifacts']['weights.npz']=file_sha256(path/'weights.npz');(path/'run_manifest.json').write_bytes(canonical_bytes(m))
            with self.assertRaises(ValueError):self.mod.load_bundle(path/'run_manifest.json',file_sha256(path/'run_manifest.json'),**args)
    def test_rehashed_pca_cannot_replace_the_fixed_projection(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path,m,inputs,args=self.bundle(d);arrays=sid_v2._load_npz(path/'pca.npz');arrays['components'][0]*=-1;sid_v2._write_npz(path/'pca.npz',arrays)
            m['artifacts']['pca.npz']=file_sha256(path/'pca.npz');m['pca']['components_sha256']=hashlib.sha256(arrays['components'].astype('<f8').tobytes()).hexdigest();(path/'run_manifest.json').write_bytes(canonical_bytes(m))
            with self.assertRaises(ValueError):self.mod.load_bundle(path/'run_manifest.json',file_sha256(path/'run_manifest.json'),**args)
    def test_changed_fallback_set_is_rejected_when_recomputed(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path,m,inputs,args=self.bundle(d);args['fallback']=np.zeros(8,dtype=bool)
            with self.assertRaises(ValueError):self.mod.load_bundle(path/'run_manifest.json',file_sha256(path/'run_manifest.json'),**args)
if __name__=='__main__':unittest.main()
