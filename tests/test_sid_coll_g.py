"""Synthetic full new G train/serial-verify without old pinned fixture artifact loaders."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch
import numpy as np
import torch
from vkgr import generative_coll_run as g, sid_coll_v1 as coll, sid_coll_run as fit
from vkgr import generative_experiment as ge, generative_data as gd, sid_tok_audit as au, sid_v2
from vkgr.generative_data_v2 import ListExamples
from vkgr.generative_evaluation import SliceStatistics
from vkgr.generative_foundation import prepare_fixture
from vkgr.data_audit import canonical_bytes,file_sha256
ROOT=Path(__file__).resolve().parents[1]

class GCollTests(unittest.TestCase):
    def test_six_configs_exact_t3_and_reject_old_label(self):
        old=json.loads((ROOT/'configs/gtok01/g-tok-t3-bhkm-b256-s42.json').read_text())
        for variant in ('c1','c2'):
            for seed in (42,2024,2025):
                cfg=json.loads((ROOT/f'configs/sid_coll/g-coll-{variant}-s{seed}.json').read_text())
                g.validate_config(cfg)
                self.assertEqual({k:v for k,v in cfg.items() if k not in ('schema','task_id','sid','seed')},
                                 {k:v for k,v in old.items() if k not in ('schema','task_id','sid','seed')})
        with self.assertRaises(ValueError):g.validate_config({**g.default_config(),'sid':old['sid']})
        with self.assertRaises(ValueError):g.validate_config({**g.default_config(),'beam':50})
        with self.assertRaises(ValueError):g.validate_config({**g.default_config(),'microbatch':256})

    def test_new_loader_c2_and_train_serial_verify(self):
        fixture=json.loads((ROOT/'tests/fixtures/g2_synthetic.json').read_text());prepared=prepare_fixture(fixture)
        ids=np.array(fixture['catalog'],dtype='uint32');n=len(ids)
        content=np.random.default_rng(42).normal(size=(n,64)).astype('float32');content/=np.linalg.norm(content,axis=1,keepdims=True)
        from vkgr.sid_coll_export import compose_c2
        vectors,_=compose_c2(content,content,np.array([True,False]*((n+1)//2))[:n])
        manifest={'schema':'coll-input-v1','status':'accepted','variant':'C2','dimension':128,'engineering':True,
          'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(vectors),'content_feature_sha256':au.feature_digest(content),
          'train_sequence_sha256':'a'*64,'script_sha256':'b'*64,'commit':'c'*40,'checkpoint_sha256':'d'*64,'source_files':{'src/vkgr/sid_coll_export.py':'b'*64}}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            tmp=Path(tmp);fcfg=fit.expected_config('C2');r=fit.fit_mapping(ids,vectors,manifest,fcfg,engineering=True)
            out=tmp/'bundle';coll.write_bundle(out,ids=ids,sids=r['sids'],vectors=vectors,input_manifest=manifest,config=fcfg,
                config_sha='e'*64,engineering=True,producer={'src/vkgr/sid_coll_run.py':'b'*64})
            np.save(tmp/'content.npy',np.vstack((np.zeros((1,64),dtype='float32'),content)))
            np.save(tmp/'vectors.npy',vectors);(tmp/'input.json').write_bytes(canonical_bytes(manifest))
            loadargs=(out/'run_manifest.json',file_sha256(out/'run_manifest.json'),tmp/'content.npy',file_sha256(tmp/'content.npy'),fixture['catalog'],True,tmp/'input.json',file_sha256(tmp/'input.json'),tmp/'vectors.npy',file_sha256(tmp/'vectors.npy'))
            loaded=g._load_sid(*loadargs)
            self.assertEqual(loaded.manifest['input_dimension'],128)
            self.assertEqual(loaded.info['sid_version'],'COLL_C2_B256D3_S42')
            with self.assertRaises(ValueError):g._load_sid(*loadargs[:5],False,*loadargs[6:])
            counts={i:0 for i in fixture['catalog']};users={}
            for part in fixture['train_partitions']:
                for row in part['rows']:
                    if row['like']:counts[row['item_id']]+=1;users[row['user_id']]=users.get(row['user_id'],0)+1
            stats=SliceStatistics((1,3),MappingProxyType(counts),MappingProxyType(users),'f'*64,'f'*64,'f'*64)
            binding={'queries_sha256':ge._digest(prepared['queries']),'catalog_sha256':ge._digest(fixture['catalog']),
                'synthetic':True,'engineering':True,'b1':{'history_version':'HIST_V2_OBS_L50'}}
            context=ge.RunContext(ge._CONTEXT_KEY,examples=ListExamples(prepared['examples']),codec=loaded.codec,
                queries=prepared['queries'],binding=binding,stats=stats).check()
            cfg={**g.default_config(),'sid':loaded.info['sid_version'],'d_model':64,'ff':256,'learning_rate':.001,
                 'microbatch':2,'max_epochs':1,'min_epochs':1}
            args=(ROOT,tmp/'unused-prepared','a'*64,tmp/'unused-hist','b'*64,*loadargs[:4])
            with patch.object(g,'_contexts',return_value=(context,True,loaded)), patch.dict(os.environ,{'VKGR_DATA_ROOT':str(tmp)}):
                value=g.run(*args,cfg,tmp/'train',temporary=tmp/'scratch')
                self.assertEqual(value['decode'],{'mode':'batched','batch_size':1})
                result=g.verify(*args,tmp/'train/manifest.json',value['manifest_sha256'],tmp/'verify',temporary=tmp/'scratch2')
                self.assertEqual(result['status'],'verified')
                self.assertEqual(value['binding']['sid']['input_dimension'],128)
                self.assertEqual(value['binding']['data_root'],str(tmp.resolve()))
                self.assertIn('src/vkgr/generative_coll_run.py',value['binding']['source']['files'])
                with patch.dict(os.environ,{'VKGR_DATA_ROOT':str(ROOT)}):
                    with self.assertRaisesRegex(ValueError,'binding mismatch'):
                        g.verify(*args,tmp/'train/manifest.json',value['manifest_sha256'],tmp/'wrongdata',temporary=tmp/'scratchwrong')
                # New own-checkpoint verification is allowed; a modified file must still fail closed.
                with (tmp/'train/best.pt').open('ab') as stream:stream.write(b'x')
                with self.assertRaises(ValueError):g.verify(*args,tmp/'train/manifest.json',value['manifest_sha256'],tmp/'badverify',temporary=tmp/'scratch3')


class DataRootRoutingTests(unittest.TestCase):
    def test_explicit_data_root_routes_only_legacy_loader(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            tmp=Path(tmp);data=tmp/'data';data.mkdir()
            class ReachedCorrectLoader(Exception):pass
            def load(root,*args):
                self.assertEqual(root,data.resolve())
                raise ReachedCorrectLoader()
            with patch.dict(os.environ,{'VKGR_DATA_ROOT':str(data)}),patch.object(g.gr,'load_context',side_effect=load):
                with self.assertRaises(ReachedCorrectLoader):
                    g._contexts(ROOT,None,None,None,None,None,None,None,None,g.default_config(),tmp/'scratch',
                       input_manifest=None,input_sha=None,vectors_path=None,vectors_sha=None)

    def test_missing_explicit_data_root_fails_before_loader(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            tmp=Path(tmp)
            with patch.dict(os.environ,{'VKGR_DATA_ROOT':str(tmp/'missing')}),patch.object(g.gr,'load_context',side_effect=AssertionError('wrong root must fail closed')):
                with self.assertRaises(ValueError):
                    g._contexts(ROOT,None,None,None,None,None,None,None,None,g.default_config(),tmp/'scratch',
                       input_manifest=None,input_sha=None,vectors_path=None,vectors_sha=None)
