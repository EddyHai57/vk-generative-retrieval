"""A1 collaborative SID registry, boundary mutations and original-procedure parity."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class Registry(unittest.TestCase):
    def module(self):
        from vkgr import coll_sid_a1_run
        return coll_sid_a1_run

    def test_entry_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('vkgr.coll_sid_a1_run'), 'new independent runner missing')

    def test_formal_configs_only_change_identity(self):
        g = self.module()
        for seed in (42, 2024, 2025):
            old=json.loads((ROOT/f'configs/gtok01b/g-gtok-a1-t3-b256d3-s{seed}.json').read_text())
            new=json.loads((ROOT/f'configs/coll_sid_a1/g-coll-a1-c1-s{seed}.json').read_text())
            self.assertEqual(new, dict(old,schema=g.SCHEMA,task_id=g.TASK_ID,sid=g.SID_VERSION))
            self.assertEqual(g.validate_config(new),new)

    def test_every_formal_field_is_frozen(self):
        g=self.module();base=g.default_config()
        for key,value in base.items():
            if key=='seed':continue
            bad=copy.deepcopy(base)
            bad[key]=value+'-changed' if isinstance(value,str) else value+1
            with self.subTest(key=key), self.assertRaises(ValueError):g.validate_config(bad)
        for seed in (0,1,43,True,2026):
            with self.subTest(seed=seed),self.assertRaises(ValueError):g.validate_config(dict(base,seed=seed))

    def test_wrong_family_manifest_fails_before_checkpoint(self):
        import tempfile
        from vkgr.data_audit import file_sha256
        g=self.module()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p=Path(d)/'manifest.json'
            value={key:None for key in ('schema','kind','status','config','binding','selection','epochs','losses','best','last','wall_seconds','hit_max_epochs','eval_workers','decode')}
            value.update(schema='g-g6-train-manifest-v1',kind=g.MANIFEST_KIND,status='complete')
            p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):g._read_run(p,file_sha256(p))

    def test_strict_bundle_pins(self):
        g=self.module()
        g.check_pins(g.SID_SHA,g.INPUT_SHA,g.VECTORS_SHA,False)
        for index in range(3):
            values=[g.SID_SHA,g.INPUT_SHA,g.VECTORS_SHA];values[index]='0'*64
            with self.subTest(index=index),self.assertRaises(ValueError):g.check_pins(*values,False)

    def test_depth3_adaptation_preserves_codec_digits(self):
        import numpy as np
        from vkgr.sid_v2 import build_codec,LoadedSID
        g=self.module()
        ids=np.array([1,2],dtype=np.uint32);sids=np.array([[0,0,0,0],[1,0,0,0]],dtype=np.uint32)
        codec=build_codec(ids,sids,256)
        old=LoadedSID(codec,{},dict(branching=256))
        new=g._as_depth3(old)
        np.testing.assert_array_equal(new.codec.catalog.sids,old.codec.catalog.sids)
        self.assertEqual(new.codec.vocab_size,old.codec.vocab_size)
        self.assertEqual(new.info['depth'],3)


    def test_synthetic_full_train_own_verify_and_tamper(self):
        import os
        import tempfile
        from types import MappingProxyType
        from unittest.mock import patch
        import numpy as np
        import torch
        from vkgr import sid_coll_v1 as coll, sid_coll_run as fit, sid_v2
        from vkgr import generative_experiment as ge, sid_tok_audit as au
        from vkgr.generative_data_v2 import ListExamples
        from vkgr.generative_evaluation import SliceStatistics
        from vkgr.generative_foundation import prepare_fixture
        from vkgr.data_audit import canonical_bytes, file_sha256
        g=self.module();torch.set_num_threads(1)
        fixture=json.loads((ROOT/'tests/fixtures/g2_synthetic.json').read_text());prepared=prepare_fixture(fixture)
        ids=np.array(fixture['catalog'],dtype='uint32');n=len(ids)
        content=np.random.default_rng(42).normal(size=(n,64)).astype('float32');content/=np.linalg.norm(content,axis=1,keepdims=True)
        manifest={'schema':'coll-input-v1','status':'accepted','variant':'C1','dimension':64,'engineering':True,
          'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(content),'content_feature_sha256':au.feature_digest(content),
          'train_sequence_sha256':'a'*64,'script_sha256':'b'*64,'commit':'c'*40,'checkpoint_sha256':None,'source_files':{'src/vkgr/sid_coll_sgns.py':'b'*64}}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            tmp=Path(tmp);fcfg=fit.expected_config('C1');result=fit.fit_mapping(ids,content,manifest,fcfg,engineering=True)
            out=tmp/'bundle';coll.write_bundle(out,ids=ids,sids=result['sids'],vectors=content,input_manifest=manifest,config=fcfg,
                config_sha='e'*64,engineering=True,producer={'src/vkgr/sid_coll_run.py':'b'*64})
            np.save(tmp/'content.npy',np.vstack((np.zeros((1,64),dtype='float32'),content)))
            np.save(tmp/'vectors.npy',content);(tmp/'input.json').write_bytes(canonical_bytes(manifest))
            loadargs=(out/'run_manifest.json',file_sha256(out/'run_manifest.json'),tmp/'content.npy',file_sha256(tmp/'content.npy'),fixture['catalog'],True,tmp/'input.json',file_sha256(tmp/'input.json'),tmp/'vectors.npy',file_sha256(tmp/'vectors.npy'))
            loaded=g._as_depth3(g.coll_base._load_sid(*loadargs))
            counts={i:0 for i in fixture['catalog']};users={}
            for part in fixture['train_partitions']:
                for row in part['rows']:
                    if row['like']:counts[row['item_id']]+=1;users[row['user_id']]=users.get(row['user_id'],0)+1
            stats=SliceStatistics((1,3),MappingProxyType(counts),MappingProxyType(users),'f'*64,'f'*64,'f'*64)
            binding={'queries_sha256':ge._digest(prepared['queries']),'catalog_sha256':ge._digest(fixture['catalog']),
                'synthetic':True,'engineering':True,'b1':{'history_version':'HIST_V2_OBS_L50'}}
            context=ge.RunContext(ge._CONTEXT_KEY,examples=ListExamples(prepared['examples']),codec=loaded.codec,
                queries=prepared['queries'],binding=binding,stats=stats).check()
            cfg=dict(g.default_config(),d_model=8,heads=1,head_dim=8,encoder_layers=1,decoder_layers=1,ff=32,dropout=0.0,
                     microbatch=2,max_epochs=1,min_epochs=1)
            g.validate_config(cfg,True)
            args=(ROOT,tmp/'unused-prepared','a'*64,tmp/'unused-hist','b'*64,*loadargs[:4])
            with patch.object(g,'_contexts',return_value=(context,True,loaded)),patch.dict(os.environ,{'VKGR_DATA_ROOT':str(tmp)}):
                value=g.run(*args,cfg,tmp/'train',temporary=tmp/'scratch',decode={'mode':'batched','batch_size':1})
                result=g.verify(*args,tmp/'train/manifest.json',value['manifest_sha256'],tmp/'verify',temporary=tmp/'scratch2',decode={'mode':'batched','batch_size':1})
                self.assertEqual(result['status'],'verified')
                self.assertEqual(value['binding']['sid']['input_dimension'],64)
                self.assertEqual(value['schema'],g.MANIFEST_SCHEMA)
                self.assertEqual(value['best']['epoch'],value['selection']['best_epoch'])
                with (tmp/'train/best.pt').open('ab') as stream:stream.write(b'x')
                with self.assertRaises(ValueError):g.verify(*args,tmp/'train/manifest.json',value['manifest_sha256'],tmp/'badverify',temporary=tmp/'bad',decode={'mode':'batched','batch_size':1})

if __name__=='__main__':unittest.main()

class ProcedureParity(unittest.TestCase):
    def test_training_and_selection_loop_is_original_a1(self):
        import ast,inspect
        from vkgr import coll_sid_a1_run as new, generative_g6_run as old
        def loop(fn):
            return ast.dump(next(node for node in ast.parse(inspect.getsource(fn)).body[0].body if isinstance(node,ast.While)),include_attributes=False)
        self.assertEqual(loop(new.run),loop(old.run))
        self.assertIs(new.make_model_arch_d,old.make_model_arch_d)
        self.assertIs(new.train_batch_d,old.train_batch_d)
    def test_verify_selection_replay_is_original_a1(self):
        import ast,inspect
        from vkgr import coll_sid_a1_run as new, generative_g6_run as old
        def loop(fn):
            return ast.dump(next(node for node in ast.parse(inspect.getsource(fn)).body[0].body if isinstance(node,ast.For)),include_attributes=False)
        self.assertEqual(loop(new.verify),loop(old.verify))
