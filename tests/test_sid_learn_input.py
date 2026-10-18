"""Formal pins precede IO; only original train history construction supplies fallback."""
import copy,importlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from vkgr.data_audit import canonical_bytes,file_sha256
from vkgr import sid_coll_sgns as sg
from test_sid_learn_bundle import fixture
ROOT=Path(__file__).resolve().parents[1]
class LearnInputTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'src/vkgr/sid_learn_input.py').is_file(),'missing formal input entry');self.mod=importlib.import_module('vkgr.sid_learn_input')
    def test_all_formal_physical_pins_fail_before_io(self):
        refs={key:{'path':'/synthetic/'+key+('.npy' if key in ('content','c1_vectors') else '.json'),'sha256':value} for key,value in self.mod.FORMAL_PINS.items()}
        for key in refs:
            bad=copy.deepcopy(refs);bad[key]['sha256']='0'*64
            with self.subTest(key=key),patch.object(self.mod.gd,'_checked_file',side_effect=AssertionError('opened unregistered file')) as checked,self.assertRaises(ValueError):self.mod.load_inputs(bad,engineering=False)
            checked.assert_not_called()
    def build(self,root):
        inputs=fixture();refs={};root=Path(root).resolve()
        def save(name,value):
            p=root/(name+('.npy' if isinstance(value,np.ndarray) else '.json'))
            if isinstance(value,np.ndarray):np.save(p,value,allow_pickle=False)
            else:p.write_bytes(canonical_bytes(value))
            return {'path':str(p),'sha256':file_sha256(p)}
        refs['content']=save('content',np.vstack((np.zeros((1,64),dtype='float32'),inputs['content'])))
        refs['catalog']=save('catalog',inputs['ids'].astype(int).tolist());refs['c1_vectors']=save('c1-vectors',inputs['c1_vectors'])
        audit=save('audit',{'synthetic':True});refs['s0_plan']=save('plan',{'catalog':refs['catalog'],'raw_root':str(root/'raw'),'train_audit':audit})
        histories={0:list(range(1,8)),1:[8]};physical={f'subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet':'a'*64 for week in range(25)}
        im=inputs['c1_manifest'];im['input_sources']={**physical,'s0_plan':refs['s0_plan']['sha256'],'train_audit':audit['sha256'],'catalog':refs['catalog']['sha256'],'content':refs['content']['sha256']}
        refs['c1_manifest']=save('input-manifest',im)
        return refs,histories,physical
    def test_engineering_recomputes_only_original_train_set_and_freezes_pca(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            refs,histories,physical=self.build(d)
            with patch.object(self.mod.legacy,'load_histories',return_value=(histories,physical)) as load:
                result=self.mod.load_inputs(refs,engineering=True)
            self.assertEqual(load.call_count,1);self.assertEqual(result['target'].shape,(8,32));self.assertEqual(result['fallback'].tolist(),[False]*7+[True]);self.assertEqual(result['history_proof']['sequence_sha256'],sg._sequence_digest(histories))
            summary=self.mod.summary(result);self.assertEqual(summary['fallback_count'],1);self.assertEqual(summary['target_dimension'],32);self.assertIs(summary['written'],False)
            self.assertNotIn('target',summary);self.assertNotIn('histories',result)
    def test_changed_history_sequence_stops_before_fit(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            refs,histories,physical=self.build(d);histories[0]=list(reversed(histories[0]))
            with patch.object(self.mod.legacy,'load_histories',return_value=(histories,physical)),self.assertRaises(ValueError):self.mod.load_inputs(refs,engineering=True)
    def test_unregistered_validation_source_is_rejected(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            refs,histories,physical=self.build(d);p=Path(refs['c1_manifest']['path']);im=json.loads(p.read_text());im['input_sources']['subsamples/ur0.01_ir0.01/validation/week_25.parquet']='b'*64;p.write_bytes(canonical_bytes(im));refs['c1_manifest']['sha256']=file_sha256(p)
            with patch.object(self.mod.legacy,'load_histories',return_value=(histories,physical)),self.assertRaises(ValueError):self.mod.load_inputs(refs,engineering=True)
if __name__=='__main__':unittest.main()
