import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from vkgr.retrieval_models import TwoTower
from vkgr.retrieval_ann import canonical, file_hash

class OfflineCLITests(unittest.TestCase):
    def run_cli(self, *args, ok=True):
        self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_infer_run'),'P4F CLI missing')
        p=subprocess.run([sys.executable,'-B','-m','vkgr.retrieval_infer_run',*map(str,args)],capture_output=True,text=True)
        if ok:self.assertEqual(p.returncode,0,p.stderr)
        else:self.assertNotEqual(p.returncode,0)
        return p

    def fixture(self, root):
        def save(name,v):
            p=root/name;p.write_bytes(canonical(v));return {'path':name,'sha256':file_hash(p)}
        catalog=[10,20,30];idmap={10:1,20:2,30:3}
        content=torch.zeros((4,64),dtype=torch.float32);model=TwoTower('id_mean',content,[1,2,3])
        with torch.no_grad():
            model.embedding.weight.zero_();model.embedding.weight[1,0]=1;model.embedding.weight[2,1]=1;model.embedding.weight[3,0]=-1
        config={'architecture':'id_mean','seed':42,'batch_size':512}
        torch.save({'schema_version':1,'model':model.state_dict(),'binding':{'original':'producer','config':config}},root/'checkpoint.pt')
        np.save(root/'content.npy',content.numpy())
        queries=[{'query_id':7,'user_id':7,'history':[10],'target':10,'oov':False},{'query_id':8,'user_id':8,'history':[20],'target':99,'oov':True},{'query_id':9,'user_id':9,'history':[],'target':30,'oov':False}]
        ranks=[[[10,20,30],[20,10,30],[30,20,10],[10,20,30]] for _ in queries]
        refs={'catalog':save('catalog.json',catalog),'idmap':save('idmap.json',{'10':1,'20':2,'30':3}),'queries':save('queries.json',queries),'rankings':save('rankings.json',ranks),'supported_indices':save('support.json',[1,2,3]),'content':{'path':'content.npy','sha256':file_hash(root/'content.npy')},'checkpoint':{'path':'checkpoint.pt','sha256':file_hash(root/'checkpoint.pt')}}
        binding={'source_sha256':'a'*64,'checkpoint_sha256':refs['checkpoint']['sha256'],'catalog_sha256':refs['catalog']['sha256'],'idmap_sha256':refs['idmap']['sha256'],'query_sha256':refs['queries']['sha256']}
        data={'kind':'p4f_input','split':'validation','synthetic':True,'binding':binding,'files':refs,'config':config,'checkpoint_binding':{'original':'producer','config':config},'popularity':[10,20,30],'producer_pins':{'original_source':'a'*64}}
        save('input.json',data);return root/'input.json'

    def common(self, input_path):return ['--input',input_path,'--input-sha256',file_hash(input_path)]

    def test_actual_fuse_infer_interruption_resume_and_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=self.fixture(root);args=self.common(inp)
            self.run_cli('fuse',*args,'--output',root/'fusion')
            selection=root/'fusion'/'manifest.json'
            fused=json.loads(selection.read_text())
            self.assertEqual(fused['coverage']['selected']['count'],3)
            self.assertEqual(fused['coverage']['route_union']['fraction'],1.)
            opt=['--selection',selection,'--selection-sha256',file_hash(selection),'--shard-size',1]
            self.run_cli('infer',*args,*opt,'--output',root/'full')
            self.run_cli('infer',*args,*opt,'--output',root/'resume','--max-shards',1)
            self.assertFalse((root/'resume'/'completion.json').exists())
            self.run_cli('infer',*args,*opt,'--output',root/'resume','--resume')
            full=json.loads((root/'full'/'completion.json').read_text());res=json.loads((root/'resume'/'completion.json').read_text())
            self.assertEqual(full['result_sha256'],res['result_sha256'])
            self.assertEqual(full['eligible_count'],2);self.assertEqual(full['fallback_count'],1)
            self.assertEqual(full['unknown_ids'],1)
            self.run_cli('verify',*args,'--output',root/'resume')
            shard=root/'resume'/'shard-000001.json';shard.write_text('{}')
            self.run_cli('verify',*args,'--output',root/'resume',ok=False)

    def test_verify_rejects_forged_predictions_even_with_rehashed_completion(self):
        from vkgr.retrieval_infer_run import _completion, inspect_shards
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=self.fixture(root);args=self.common(inp)
            self.run_cli('fuse',*args,'--output',root/'fusion');sel=root/'fusion'/'manifest.json'
            self.run_cli('infer',*args,'--selection',sel,'--selection-sha256',file_hash(sel),'--output',root/'infer','--shard-size',1)
            p=root/'infer'/'shard-000000.json';v=json.loads(p.read_text());v['records'][0]['prediction']=[30,20,10];p.write_bytes(canonical(v))
            for p in (root/'infer').glob('progress-*.json'):
                v=json.loads(p.read_text());v['shards']['shard-000000.json']=file_hash(root/'infer'/'shard-000000.json');p.write_bytes(canonical(v))
            c=root/'infer'/'completion.json';v=json.loads(c.read_text());q=json.loads((root/'queries.json').read_text());identity=v['identity']
            records,hashes=inspect_shards(root/'infer',identity,q,True)
            c.write_bytes(canonical(_completion(identity,records,hashes,q,[10,20,30])))
            self.run_cli('verify',*args,'--output',root/'infer',ok=False)

    def test_export_and_index_missing_dependency_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=self.fixture(root)
            self.run_cli('export',*self.common(inp),'--output',root/'export')
            exp=root/'export'/'manifest.json'
            users=np.load(root/'export'/'users.npy').tolist()
            audit={'query_ids':[7,8],'targets':[10,99],'vectors':users,'synthetic':True,'audit_sha256':json.loads(inp.read_text())['binding']['query_sha256']}
            (root/'audit-input.json').write_bytes(canonical(audit))
            from vkgr.baselines import blockwise_topk
            exact=blockwise_topk(np.array(users,dtype=np.float32),np.load(root/'export'/'items.npy'),np.array([10,20,30]),50).tolist()
            (root/'exact.json').write_bytes(canonical(exact))
            audit_args=['--audit',root/'audit-input.json','--audit-sha256',file_hash(root/'audit-input.json'),'--exact-predictions',root/'exact.json','--exact-predictions-sha256',file_hash(root/'exact.json')]
            if importlib.util.find_spec('faiss') is None:
                p=self.run_cli('index',*self.common(inp),'--export',exp,'--export-sha256',file_hash(exp),'--output',root/'index',*audit_args,ok=False)
                self.assertIn('existing FAISS unavailable',p.stderr)
                self.assertFalse((root/'index').exists())
            else:self.run_cli('index',*self.common(inp),'--export',exp,'--export-sha256',file_hash(exp),'--output',root/'index',*audit_args)

    def test_resume_rejects_input_drift_and_missing_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=self.fixture(root);args=self.common(inp)
            self.run_cli('fuse',*args,'--output',root/'fusion');s=root/'fusion'/'manifest.json';opt=['--selection',s,'--selection-sha256',file_hash(s),'--shard-size',1]
            self.run_cli('infer',*args,*opt,'--output',root/'partial','--max-shards',1)
            (root/'partial'/'shard-000000.json').unlink()
            self.run_cli('infer',*args,*opt,'--output',root/'partial','--resume',ok=False)
            data=json.loads(inp.read_text());data['binding']['checkpoint_sha256']='b'*64;inp.write_bytes(canonical(data))
            self.run_cli('infer',*self.common(inp),*opt,'--output',root/'partial','--resume',ok=False)

    def test_test_split_and_producer_changes_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=self.fixture(root);data=json.loads(inp.read_text());data['split']='test';inp.write_bytes(canonical(data))
            self.run_cli('fuse',*self.common(inp),'--output',root/'bad',ok=False);self.assertFalse((root/'bad').exists())
            data['split']='validation';data['files']['catalog']['sha256']='b'*64;inp.write_bytes(canonical(data))
            self.run_cli('fuse',*self.common(inp),'--output',root/'bad',ok=False)

    def real_style(self, root, inp):
        from vkgr.retrieval_infer_run import SOURCE, PLAN
        m=json.loads(inp.read_text());m['synthetic']=False;m['binding']['source_sha256']=SOURCE;m['binding']['plan_sha256']=PLAN
        def save(name,value):
            p=root/name;p.write_bytes(canonical(value));return {'path':str(p),'sha256':file_hash(p)}
        idmap={'items':[10,20,30],'pad':0,'start':1,'supported_indices':[1,2,3]}
        m['files']['idmap']=save('idmap.json',idmap);m['binding']['idmap_sha256']=m['files']['idmap']['sha256']
        m['files']['supported_indices']=save('support.json',[10,20,30])
        graph={'catalog':[10,20,30],'popularity':[3,2,1],'degrees':{'10':1,'20':1,'30':1},'neighbors':{'10':[[20,.5]],'20':[[10,.5]],'30':[]}}
        m['files']['graph']=save('graph.json',graph);m['files'].pop('rankings')
        prepared_artifacts={name:m['files'][role]['sha256'] for name,role in [('catalog.json','catalog'),('content.npy','content'),('idmap.json','idmap'),('support.json','supported_indices')]}
        prepared_artifacts['examples.jsonl']='d'*64
        pbind={'consumer':{'source_fingerprint':'e'*64},'query_pins':[{'role':'validation-queries.json','sha256':m['binding']['query_sha256']}],'legacy_producer':{'source_fingerprint':'f'*64}}
        prep={'task_id':'P4B','kind':'prepare','status':'success','binding':pbind,'artifacts':prepared_artifacts};pr=save('prepare-manifest.json',prep)
        shared={'prepared_manifest_sha256':pr['sha256'],'prepared_artifacts':prepared_artifacts,'catalog_sha256':m['binding']['catalog_sha256'],'validation_query_sha256':m['binding']['query_sha256'],'prepared_producer':pbind['consumer'],'consumer':{'source_fingerprint':'1'*64},'test_gate':'closed_hash_only'}
        m['checkpoint_binding']={**shared,'config':m['config'],'runtime_config_sha256':'2'*64,'architecture':{'name':'id_mean'},'selection_sha256':None}
        state=torch.load(root/'checkpoint.pt',weights_only=True);state['binding']=m['checkpoint_binding'];torch.save(state,root/'checkpoint.pt')
        m['files']['checkpoint']['sha256']=file_hash(root/'checkpoint.pt');m['binding']['checkpoint_sha256']=file_hash(root/'checkpoint.pt')
        best={'path':'checkpoint.pt','sha256':file_hash(root/'checkpoint.pt'),'step':1,'epoch':1,'metric':[.2,.1]}
        core={'task_id':'P4D','kind':'core_train','status':'complete','binding':m['checkpoint_binding'],'best':best};cr=save('core-manifest.json',core)
        config={**m['config'],'learning_rate':.001,'temperature':.07};m['config']=config;m['checkpoint_binding']['config']=config
        state['binding']=m['checkpoint_binding'];torch.save(state,root/'checkpoint.pt');m['files']['checkpoint']['sha256']=file_hash(root/'checkpoint.pt');m['binding']['checkpoint_sha256']=file_hash(root/'checkpoint.pt');best['sha256']=file_hash(root/'checkpoint.pt');core['binding']=m['checkpoint_binding'];core['best']=best;cr=save('core-manifest.json',core)
        chosen={'config':{k:config[k] for k in ('architecture','learning_rate','temperature','seed')},'manifest_sha256':cr['sha256'],'best':best}
        sel={'task_id':'P4D','kind':'core_selection','status':'success','input_binding':{**shared,'shared_batch_size':512},'best_architecture':'id_mean','representative_seed':42,'selected_configs':{'id_mean':chosen}}
        baseline={'task_id':'P4C','kind':'baselines','status':'success','binding':{**shared,'query_sha256':m['binding']['query_sha256']},'artifacts':{'graph.json':m['files']['graph']['sha256']}}
        m['producer_manifests']={'prepared':pr,'core_run':cr,'core_selection':save('selection.json',sel),'baselines':save('baseline-manifest.json',baseline)}
        inp.write_bytes(canonical(m));return m

    def test_real_style_producer_schema_complete_and_cross_pin_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=self.fixture(root);m=self.real_style(root,inp)
            self.run_cli('export',*self.common(inp),'--output',root/'real-export')
            export=json.loads((root/'real-export'/'manifest.json').read_text())
            self.assertIn('routes.json',export['files'])
            self.assertNotEqual(export['consumer']['source_fingerprint'],m['binding']['source_sha256'])
            m['route_export']={'path':str(root/'real-export'/'manifest.json'),'sha256':file_hash(root/'real-export'/'manifest.json')};inp.write_bytes(canonical(m))
            self.run_cli('fuse',*self.common(inp),'--output',root/'real-fuse')
            m['checkpoint_binding']['prepared_manifest_sha256']='0'*64;inp.write_bytes(canonical(m))
            self.run_cli('export',*self.common(inp),'--output',root/'bad-export',ok=False)
            self.assertFalse((root/'bad-export').exists())

    def test_atomic_publication_survives_failure_before_final_name(self):
        from unittest.mock import patch
        from vkgr.retrieval_ann import write_json
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'shard.json'
            with patch('os.link',side_effect=OSError('simulated publication interruption')):
                with self.assertRaises(OSError):write_json(p,{'records':[{'query_id':1}]})
            self.assertFalse(p.exists())
            write_json(p,{'records':[{'query_id':1}]})
            self.assertEqual(json.loads(p.read_text()),{'records':[{'query_id':1}]})
            with self.assertRaises(FileExistsError):write_json(p,{'records':[]})

    def test_infer_rejects_reduced_selection_denominator_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=self.fixture(root);args=self.common(inp)
            self.run_cli('fuse',*args,'--output',root/'fusion');p=root/'fusion'/'manifest.json';v=json.loads(p.read_text())
            v['eligible_query_ids']=[7]
            for r in v['candidate_metrics']:r['count']=1
            p.write_bytes(canonical(v))
            self.run_cli('infer',*args,'--selection',p,'--selection-sha256',file_hash(p),'--output',root/'bad',ok=False)
            self.assertFalse((root/'bad').exists())

    def test_resume_recovers_published_shard_after_progress_write_interruption(self):
        from unittest.mock import patch
        from vkgr import retrieval_infer_run as runner
        from vkgr.retrieval_ann import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inp=self.fixture(root);args=self.common(inp)
            self.run_cli('fuse',*args,'--output',root/'fusion');p=root/'fusion'/'manifest.json'
            m,paths,catalog,queries,rankings,idmap=runner.load_input(inp,file_hash(inp))
            selection=runner.selection_record(p,file_hash(p),m,file_hash(inp));selection['_path']=str(p.resolve())
            def interrupt(path,value):
                if Path(path).name.startswith('progress-'):raise OSError('interrupted after shard publication')
                write_json(path,value)
            with patch('vkgr.retrieval_infer_run.write_json',side_effect=interrupt):
                with self.assertRaises(OSError):runner.infer(m,catalog,queries,rankings,root/'partial',file_hash(inp),selection,file_hash(p),1)
            self.assertTrue((root/'partial'/'shard-000000.json').exists())
            self.assertFalse((root/'partial'/'completion.json').exists())
            self.run_cli('infer',*args,'--selection',p,'--selection-sha256',file_hash(p),'--output',root/'partial','--shard-size',1,'--resume')
            self.run_cli('verify',*args,'--output',root/'partial')

class ConsumerReleaseFlowTests(unittest.TestCase):
    fixture=OfflineCLITests.fixture
    common=OfflineCLITests.common

    def test_release_keyword_interfaces_and_cli_pair_before_input(self):
        import inspect
        from unittest.mock import patch
        from vkgr import retrieval_ann as ann, retrieval_infer_run as runner
        for function in (ann.export_vectors,ann.build_index,ann.audit_index,runner.fuse,runner.infer,runner.verify):
            self.assertIn('consumer_release',inspect.signature(function).parameters)
            self.assertEqual(inspect.signature(function).parameters['consumer_release'].kind,inspect.Parameter.KEYWORD_ONLY)
        for flag in ('--consumer-release','--consumer-release-sha256'):
            with patch.object(runner,'load_input') as load:
                with self.assertRaises((ValueError,SystemExit)):
                    runner.main(['fuse','--input','missing','--input-sha256','a'*64,'--output','missing-output',flag,'a'*64])
                load.assert_not_called()

    def test_invalid_release_before_input_or_output(self):
        from unittest.mock import patch
        from vkgr import retrieval_infer_run as runner
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out'
            with patch.object(runner,'load_input') as load:
                with self.assertRaises((ValueError,OSError,SystemExit)):
                    runner.main(['fuse','--input','missing','--input-sha256','a'*64,'--output',str(out),'--consumer-release',str(Path(tmp)/'absent'),'--consumer-release-sha256','a'*64])
                load.assert_not_called();self.assertFalse(out.exists())

    def test_explicit_release_actual_tiny_export_fusion_resume_verify_and_drift(self):
        import contextlib, copy
        from unittest.mock import patch
        from test_retrieval_ann import release_fixture, synthetic_parent_pins
        from vkgr import retrieval_ann as ann, retrieval_infer_run as runner
        self.assertIn('consumer_release',__import__('inspect').signature(runner.infer).parameters)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();code=root/'code';code.mkdir();release,ref=release_fixture(code)
            inp=self.fixture(root);args=self.common(inp)
            opts=['--consumer-release',ref['path'],'--consumer-release-sha256',ref['sha256']]
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(ann,'__file__',str(code/'src/vkgr/retrieval_ann.py')))
                stack.enter_context(patch('vkgr.manifest.ROOT',code))
                stack.enter_context(patch.object(ann,'_RELEASE_PARENTS',synthetic_parent_pins(release)))
                stack.enter_context(patch.object(ann,'_runtime_identity',return_value=release['runtime']))
                stack.enter_context(patch('vkgr.manifest.git_context',side_effect=RuntimeError('no Git')))
                runner.main(['export',*map(str,args),'--output',str(root/'export'),*opts])
                runner.main(['fuse',*map(str,args),'--output',str(root/'fusion'),*opts])
                sel=root/'fusion/manifest.json';opt=['--selection',str(sel),'--selection-sha256',file_hash(sel),'--shard-size','1']
                runner.main(['infer',*map(str,args),*opt,'--output',str(root/'partial'),'--max-shards','1',*opts])
                original=(code/'release.json').read_bytes()
                changed=copy.deepcopy(release);changed['declared_build_git']['dirty']=False
                (code/'release.json').write_bytes(canonical(changed));newopts=['--consumer-release',ref['path'],'--consumer-release-sha256',file_hash(code/'release.json')]
                with self.assertRaises(ValueError):runner.main(['infer',*map(str,args),*opt,'--output',str(root/'partial'),'--resume',*newopts])
                (code/'release.json').write_bytes(original)
                runner.main(['infer',*map(str,args),*opt,'--output',str(root/'partial'),'--resume',*opts])
                runner.main(['verify',*map(str,args),'--output',str(root/'partial'),*opts])
                with self.assertRaises(RuntimeError):runner.main(['verify',*map(str,args),'--output',str(root/'partial')])
                (code/'release.json').write_bytes(canonical(changed))
                with self.assertRaises(ValueError):runner.main(['verify',*map(str,args),'--output',str(root/'partial'),*newopts])
                (code/'release.json').write_bytes(original)
                # Historical producer validation remains mandatory in explicit mode.
                m=json.loads(inp.read_text());m['synthetic']=False;inp.write_bytes(canonical(m))
                with self.assertRaises(ValueError):runner.main(['export',*map(str,self.common(inp)),'--output',str(root/'bad-producer'),*opts])
                self.assertFalse((root/'bad-producer').exists())
