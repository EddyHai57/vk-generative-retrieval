import ast
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

class OriginalScience(unittest.TestCase):
    def subject(self):
        self.assertIsNotNone(importlib.util.find_spec('vkgr.sgns_subset_sid'), 'new bound SID adapter missing')
        from vkgr import sgns_subset_sid
        return sgns_subset_sid
    def test_original_functions_exact_and_mapping_bytes(self):
        subject=self.subject();original={}
        for file,names in [('sid_coll_run.py',['fit_bhkm_coll']),('sid_structure_v1.py',['_hex','validate_input','audit_gates'])]:
            source=subprocess.check_output(['git','show','b5a4d1c:src/vkgr/'+file],cwd=ROOT,text=True)
            for node in ast.parse(source).body:
                if isinstance(node,ast.FunctionDef) and node.name in names:
                    text='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
                    self.assertEqual(inspect.getsource(getattr(subject,node.name)),text);original[node.name]=text
        namespace={'np':np,'sid':subject.sid};exec(original['fit_bhkm_coll'],namespace)
        rng=np.random.default_rng(42);x=rng.normal(size=(128,64));x=(x/np.linalg.norm(x,axis=1)[:,None]).astype(np.float32);ids=np.arange(128,dtype=np.uint32)
        left=namespace['fit_bhkm_coll'](ids,x,48,3,42,10);right=subject.fit_bhkm_coll(ids,x,48,3,42,10)
        self.assertEqual(left.ids.tobytes(),right.ids.tobytes());self.assertEqual(left.sids.tobytes(),right.sids.tobytes());self.assertEqual(json.dumps(left.nodes,sort_keys=True),json.dumps(right.nodes,sort_keys=True))
    def test_collision_uses_involved_items_and_integer_threshold(self):
        subject=self.subject();cpu={'items':120415,'semantic_collision_items':19646,'items_needing_suffix':9823}
        self.assertTrue(subject.collision_gate(cpu,{**cpu,'semantic_collision_items':21610})['passed'])
        self.assertFalse(subject.collision_gate(cpu,{**cpu,'semantic_collision_items':21611})['passed'])
        with self.assertRaises(ValueError):subject.collision_gate(cpu,{**cpu,'items':120414})
        self.assertFalse(subject.collision_gate({**cpu,'semantic_collision_items':0},{**cpu,'semantic_collision_items':1})['passed'])
    def test_source_delta_is_exact_not_open_ended(self):
        subject=self.subject();old={'commit':'a'*40,'files':{'existing.py':'1'*64,**subject.SCIENTIFIC_FILES}};current={'commit':'b'*40,'files':{**old['files'],**{p:'2'*64 for p in subject.ADAPTER_MEMBERS}}};subject.check_source_delta(current,old)
        for bad in ({**current,'files':{**current['files'],'unexpected.py':'3'*64}},{**current,'files':{**current['files'],'existing.py':'3'*64}}):
            with self.assertRaises(ValueError):subject.check_source_delta(bad,old)
    def test_cli_form_and_exclusive_snapshot(self):
        subject=self.subject();run=subprocess.run([sys.executable,'-B','-m','analysis.sgns_subset_sid','dry'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(json.loads(run.stdout)['config'],subject.CONFIG);self.assertFalse(json.loads(run.stdout)['subset_four_gate_acceptance'])
        bad=subprocess.run([sys.executable,'-B','-m','analysis.sgns_subset_sid','bad'],cwd=ROOT,capture_output=True);self.assertNotEqual(bad.returncode,0)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'x';p.write_bytes(b'old')
            with self.assertRaises(FileExistsError):subject.write_new(p,b'new')
            self.assertEqual(p.read_bytes(),b'old')
            with self.assertRaisesRegex(ValueError,'pinned bytes differ'):subject.read_pinned(p,'0'*64)

class StageBinding(unittest.TestCase):
    def test_receipt_job_must_bind_success_argv_and_actual_output(self):
        from vkgr import sgns_subset_sid as subject
        from analysis.ur001_derive_cli import encoded
        from unittest.mock import patch
        import hashlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out=root/'fit';out.mkdir();run=root/'sgns-gpu-subset-cpu-fit-v1';(run/'logs').mkdir(parents=True)
            old={'commit':'a'*40,'files':{**subject.SCIENTIFIC_FILES,'analysis/sgns_gpu_subset.py':'1'*64}}
            current={'commit':'b'*40,'files':{**old['files'],**{p:'2'*64 for p in subject.ADAPTER_MEMBERS}}}
            vectors=np.ones((3,64),dtype=np.float32)/8;buffer=io.BytesIO();np.save(buffer,vectors,allow_pickle=False)
            audit={'catalog_sha256':subject.sid_v2.catalog_digest(np.array([1,2,3],dtype=np.uint32)),
                   'vectors_sha256':subject.au.feature_digest(vectors),'content_feature_sha256':subject.CONTENT_V1_FEATURE_SHA256,'train_sequence_sha256':'3'*64}
            artifacts={'vectors.npy':hashlib.sha256(buffer.getvalue()).hexdigest(),'audit.json':hashlib.sha256(encoded(audit)).hexdigest()}
            (out/'vectors.npy').write_bytes(buffer.getvalue());(out/'audit.json').write_bytes(encoded(audit))
            bundle=root/'input';lock=root/'lock';producer={'source':old};env={'runtime':{'python':'3.11.16'},'lock_sha256':'4'*64}
            receipt={'schema':'gpu-sgns-subset-stage-v1','status':'verified','verified':True,'engineering':False,
                     'authority':'ad1c428','stage':'sgns','device':'cpu','source':old,'runtime':env,'input_manifest_sha256':'5'*64,
                     'input_producer':producer,'run_id':run.name,'artifacts':artifacts,'original_audit_verification':{'status':'verified'},
                     'subset_four_gate_acceptance':False,'validation_read':False,'test_read':False}
            (out/'stage-receipt.json').write_bytes(encoded(receipt));pin=hashlib.sha256(encoded(receipt)).hexdigest()
            argv=[sys.executable,'-B','-m','analysis.sgns_gpu_subset','fit','--bundle',str(bundle),'--bundle-sha','5'*64,'--device','cpu','--lock',str(lock),'--output',str(out)]
            job={'run_id':run.name,'kind':'ops','argv':argv,'git':old};terminal={'run_id':run.name,'status':'succeeded','exit_code':0}
            for p,v in [(run/'run.json',job),(run/'terminal.json',terminal),(run/'logs/stdout.log',receipt)]:p.write_bytes(encoded(v))
            with patch('analysis.sgns_gpu_subset.read_bundle',return_value=(([1,2,3],{},vectors),producer)),patch('analysis.sgns_gpu_subset.verify_original_fit',return_value={'status':'verified'}),patch.object(subject,'validate_input'):
                got=subject.load_sgns_stage(bundle,'5'*64,out,pin,run,'cpu',lock,current,env)
                self.assertEqual(got['vectors'].tobytes(),vectors.tobytes())
                for field,value in [('status','running'),('run_id','wrong'),('exit_code',7)]:
                    (run/'terminal.json').write_bytes(encoded({**terminal,field:value}))
                    with self.assertRaises(ValueError):subject.load_sgns_stage(bundle,'5'*64,out,pin,run,'cpu',lock,current,env)
                    (run/'terminal.json').write_bytes(encoded(terminal))
                (out/'vectors.npy').write_bytes(buffer.getvalue()+b'tamper')
                with self.assertRaises(ValueError):subject.load_sgns_stage(bundle,'5'*64,out,pin,run,'cpu',lock,current,env)

class OriginalGateParity(unittest.TestCase):
    def test_original_h_r_gates_and_bad_mapping_match(self):
        from vkgr import sgns_subset_sid as subject
        from vkgr import sid_v2
        original=subprocess.check_output(['git','show','b5a4d1c:src/vkgr/sid_structure_v1.py'],cwd=ROOT,text=True)
        names=['_hex','validate_input','audit_gates']
        ns={key:getattr(subject,key) for key in ('np','au','sid_v2','VARIANTS','INPUT_REQUIRED','CONTENT_V1_FEATURE_SHA256')};ns['Path']=Path
        for node in ast.parse(original).body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                exec('\n'.join(original.splitlines()[node.lineno-1:node.end_lineno]),ns)
        rng=np.random.default_rng(123);x=rng.normal(size=(96,64));x=(x/np.linalg.norm(x,axis=1)[:,None]).astype(np.float32);ids=np.arange(96,dtype=np.uint32)
        model=subject.fit_bhkm_coll(ids,x,48,3,42,10)
        im={'schema':'coll-input-v1','status':'accepted','variant':'C1','dimension':64,'engineering':True,
            'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':subject.au.feature_digest(x),
            'content_feature_sha256':'1'*64,'train_sequence_sha256':'2'*64,'script_sha256':'3'*64,
            'commit':'4'*40,'checkpoint_sha256':None,'source_files':{'producer.py':'3'*64}}
        for codes in (model.sids,model.sids*0):
            actual=subject.audit_gates(ids,codes,x,im,engineering=True)
            expected=ns['audit_gates'](ids,codes,x,im,engineering=True)
            self.assertEqual(actual,expected)
        self.assertFalse(subject.audit_gates(ids,model.sids,x,im,engineering=True)[1]['passed'])
        with self.assertRaises(ValueError):subject.validate_input(ids,x,{**im,'engineering':False},engineering=False)

class RequestAndPredecessor(unittest.TestCase):
    def test_request_unknown_fields_and_predecessor_status_rejected(self):
        from analysis import sgns_subset_sid as cli
        from analysis.ur001_derive_cli import encoded
        from vkgr import sgns_subset_sid as subject
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();req={k:str(root/k) for k in cli.PATH_KEYS}
            req.update(schema='gpu-sgns-subset-sid-request-v1',device='cpu',source={'commit':'a'*40},
                       bundle_sha='1'*64,sgns_receipt_sha='2'*64,cpu_request=None,cpu_request_sha=None)
            path=root/'request.json';raw=encoded(req);path.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
            self.assertEqual(cli.load_request(path,sha),req)
            bad=encoded({**req,'beam':50});path.write_bytes(bad)
            with self.assertRaises(ValueError):cli.load_request(path,hashlib.sha256(bad).hexdigest())
            path.write_bytes(raw);rid=cli.run_id(req,'preflight');run=Path(req['ops_root'])/'runs'/rid;(run/'logs').mkdir(parents=True)
            argv=cli.worker_argv('preflight',path,sha);job={'run_id':rid,'kind':'ops','git':req['source'],'argv':argv};jobraw=encoded(job)
            receipt={'schema':'gpu-sgns-subset-sid-stage-v1','status':'verified','verified':True,'engineering':False,
                     'run_id':rid,'run_manifest_sha256':hashlib.sha256(jobraw).hexdigest(),'source':req['source'],
                     'device':'cpu','stage':'sid-preflight','subset_four_gate_acceptance':False,'request_sha256':sha}
            target=Path(req['preflight_output']);target.mkdir();(target/'stage-receipt.json').write_bytes(encoded(receipt))
            (run/'run.json').write_bytes(jobraw);(run/'logs/stdout.log').write_bytes(encoded(receipt));terminal={'run_id':rid,'status':'succeeded','exit_code':0};(run/'terminal.json').write_bytes(encoded(terminal))
            self.assertEqual(cli.successful_stage(req,'preflight',path,sha)[0],receipt)
            for field,value in [('status','running'),('run_id','wrong'),('exit_code',1)]:
                (run/'terminal.json').write_bytes(encoded({**terminal,field:value}))
                with self.assertRaises(ValueError):cli.successful_stage(req,'preflight',path,sha)
                (run/'terminal.json').write_bytes(encoded(terminal))
            changed={**receipt,'request_sha256':'3'*64}
            (target/'stage-receipt.json').write_bytes(encoded(changed));(run/'logs/stdout.log').write_bytes(encoded(changed))
            with self.assertRaises(ValueError):cli.successful_stage(req,'preflight',path,sha)

class SyntheticTransportChain(unittest.TestCase):
    def test_fit_reload_and_independent_verify_with_explicit_small_fixture_gates(self):
        import os
        if os.environ.get('SGNS_SID_TRANSPORT_CHILD')!='1':
            env={**os.environ,'SGNS_SID_TRANSPORT_CHILD':'1','PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'tests'),'PYTHONDONTWRITEBYTECODE':'1'}
            result=subprocess.run([sys.executable,'-B','-m','unittest','test_sgns_subset_sid.SyntheticTransportChain'],cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            return
        from analysis import sgns_subset_sid as cli
        from analysis.ur001_derive_cli import encoded
        from vkgr import sgns_subset_sid as subject
        from unittest.mock import patch
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();source={'commit':'a'*40,'files':{**subject.SCIENTIFIC_FILES,**{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in subject.ADAPTER_MEMBERS}}}
            rng=np.random.default_rng(42);x=rng.normal(size=(128,64));x=(x/np.linalg.norm(x,axis=1)[:,None]).astype(np.float32);ids=np.arange(128,dtype=np.uint32)
            im={'catalog_sha256':subject.sid_v2.catalog_digest(ids),'vectors_sha256':subject.au.feature_digest(x),
                'content_feature_sha256':subject.CONTENT_V1_FEATURE_SHA256,'commit':'b'*40,'source_files':subject.SCIENTIFIC_FILES}
            req={k:str(root/k) for k in cli.PATH_KEYS};req.update(schema='gpu-sgns-subset-sid-request-v1',device='cpu',source=source,bundle_sha='1'*64,sgns_receipt_sha='2'*64,cpu_request=None,cpu_request_sha=None)
            path=root/'request.json';path.write_bytes(encoded(req));sha=hashlib.sha256(encoded(req)).hexdigest();env={'runtime':{},'lock_sha256':'3'*64}
            # This fixture isolates transportation. Original scientific gate parity is tested separately.
            audit={'items':128,'semantic_collision_items':0};gates={'passed':True,'checks':[{'id':'fixture-only','ok':True}]}
            with patch.object(cli,'source_identity',return_value=source),patch('analysis.new_instance_prepare.runtime',return_value=env), \
                 patch.object(cli,'own_job',return_value={'run_id':'fixture','run_manifest_sha256':'4'*64}),patch.object(cli,'guards'), \
                 patch.object(cli,'input_stage',return_value={'ids':ids,'vectors':x,'input_manifest':im}), \
                 patch.object(cli,'successful_stage',return_value=({'runtime':env,'sgns_stage_receipt_sha256':'2'*64},{})), \
                 patch.object(subject,'audit_gates',return_value=(audit,gates)):
                result=cli.execute('fit',path,sha)
                m=json.loads((Path(req['fit_output'])/'run_manifest.json').read_bytes())
                self.assertEqual(m['input_dimension'],64)
                self.assertEqual(m['content_feature_sha256'],im['content_feature_sha256'])
                self.assertTrue(result['mapping_replay']);self.assertFalse(result['subset_four_gate_acceptance']);self.assertEqual(result['status'],'verified')
                loaded=subject.load_bundle(Path(req['fit_output'])/'run_manifest.json',result['manifest_sha256'],catalog_ids=ids,vectors=x,input_manifest=im)
                self.assertEqual(loaded.info['sid_version'],'GPU_SGNS_SUBSET_B48D3_S42')
                mp=Path(req['fit_output'])/'run_manifest.json';original=mp.read_bytes();bad={**m,'source':{'commit':'a'*40,'files':subject.SCIENTIFIC_FILES}};mp.write_bytes(encoded(bad))
                with self.assertRaises(ValueError):subject.load_bundle(mp,hashlib.sha256(mp.read_bytes()).hexdigest(),catalog_ids=ids,vectors=x,input_manifest=im)
                mp.write_bytes(original)
                with patch.object(cli,'successful_stage',return_value=({'runtime':env,'sgns_stage_receipt_sha256':'2'*64,'manifest_sha256':result['manifest_sha256']},{})):
                    verified=cli.execute('verify',path,sha)
                self.assertTrue(verified['mapping_replay']);self.assertEqual(verified['manifest_sha256'],result['manifest_sha256'])
                self.assertEqual(json.loads((Path(req['verify_output'])/'verification.json').read_bytes()),verified)
                next_req={**req,'fit_output':str(root/'second-fit')};path.write_bytes(encoded(next_req));next_sha=hashlib.sha256(path.read_bytes()).hexdigest()
                good=subject.fit_bhkm_coll(ids,x,48,3,42,10)
                from copy import deepcopy
                bad=deepcopy(good);bad.sids[0,-1]+=1
                with patch.object(subject,'fit_bhkm_coll',side_effect=[good,bad]):
                    with self.assertRaisesRegex(ValueError,'replay'):cli.execute('fit',path,next_sha)
                self.assertFalse(Path(next_req['fit_output']).exists())
                with patch.object(subject,'audit_gates',return_value=(audit,{'passed':False,'checks':[]})):
                    with self.assertRaisesRegex(ValueError,'gates'):cli.execute('fit',path,next_sha)
                self.assertFalse(Path(next_req['fit_output']).exists())
                raw=(Path(req['fit_output'])/'mapping.npz').read_bytes();(Path(req['fit_output'])/'mapping.npz').write_bytes(raw+b'tamper')
                with self.assertRaises(ValueError):subject.load_bundle(Path(req['fit_output'])/'run_manifest.json',result['manifest_sha256'],catalog_ids=ids,vectors=x,input_manifest=im)


class ManifestAuditBinding(unittest.TestCase):
    def test_loader_rejects_manifest_audit_different_from_consumed_artifact(self):
        import os
        if os.environ.get('SGNS_SID_AUDIT_CHILD')!='1':
            env={**os.environ,'SGNS_SID_AUDIT_CHILD':'1','PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT/'tests'),'PYTHONDONTWRITEBYTECODE':'1'}
            result=subprocess.run([sys.executable,'-B','-m','unittest','test_sgns_subset_sid.ManifestAuditBinding'],cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            return
        from vkgr import sgns_subset_sid as subject
        from unittest.mock import patch
        import hashlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();ids=np.arange(128,dtype=np.uint32)
            vectors=np.ones((128,64),dtype=np.float32)/8
            codes=np.column_stack((ids%48,ids//48,np.zeros(128,dtype=np.uint32),np.zeros(128,dtype=np.uint32))).astype(np.uint32)
            im={'content_feature_sha256':subject.CONTENT_V1_FEATURE_SHA256,'commit':'a'*40,'source_files':subject.SCIENTIFIC_FILES}
            source={'commit':'b'*40,'files':{**subject.SCIENTIFIC_FILES,**{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in subject.ADAPTER_MEMBERS}}}
            # Explicit transport fixture: original scientific gate parity has a separate test.
            audit={'items':128,'semantic_collision_items':0};gates={'passed':True,'checks':[]}
            codec=subject.sid_v2.build_codec(ids,codes,48,catalog_ids=ids)
            buffer=io.BytesIO();subject.sid_v2._write_npz(buffer,{'ids':ids,'sids':codes})
            artifacts={'mapping.npz':buffer.getvalue(),'tokenizer.json':subject.canonical_bytes(subject.CONFIG),
                       'audit.json':subject.canonical_bytes(audit),'input-manifest.json':subject.canonical_bytes(im)}
            for name,raw in artifacts.items():(root/name).write_bytes(raw)
            manifest={'schema':'gpu-sgns-subset-sid-v1','status':'verified','authority':'ad1c428','engineering':False,
                      'branching':48,'depth':3,'config':subject.CONFIG,'source':source,'input_dimension':64,
                      'content_feature_sha256':im['content_feature_sha256'],'input_manifest_sha256':hashlib.sha256(subject.canonical_bytes(im)).hexdigest(),
                      'catalog_sha256':subject.sid_v2.catalog_digest(ids),'input_feature_sha256':subject.au.feature_digest(vectors),
                      'mapping_replay':True,'subset_four_gate_acceptance':False,'audit':audit,'gates':gates,
                      'suffix_vocabulary':codec.catalog.suffix_vocabulary,'vocab_size':codec.vocab_size,
                      'artifacts':{name:hashlib.sha256(raw).hexdigest() for name,raw in artifacts.items()}}
            path=root/'run_manifest.json'
            def load(value):
                raw=subject.canonical_bytes(value);path.write_bytes(raw)
                return subject.load_bundle(path,hashlib.sha256(raw).hexdigest(),catalog_ids=ids,vectors=vectors,input_manifest=im)
            with patch.object(subject,'audit_gates',return_value=(audit,gates)):
                self.assertEqual(load(manifest).manifest['audit'],audit)
                for changed in ({**audit,'semantic_collision_items':127},None):
                    with self.subTest(manifest_audit=changed):
                        with self.assertRaisesRegex(ValueError,'original SID gates/audit differ'):
                            load({**manifest,'audit':changed})
            self.assertEqual((root/'audit.json').read_bytes(),artifacts['audit.json'])


class QueueEntry(unittest.TestCase):
    def test_fixed_queue_stage_parser_exists(self):
        from analysis import sgns_subset_sid as cli
        self.assertTrue(callable(getattr(cli,'queue_job',None)), 'fixed real launcher adapter missing')


class CPUReferenceRuntime(unittest.TestCase):
    def test_gpu_cannot_use_cpu_verification_from_different_runtime(self):
        from analysis import sgns_subset_sid as cli
        from unittest.mock import patch
        cpu={'device':'cpu','source':{},'lock':'/private/tmp/lock','bundle':'/private/tmp/bundle','bundle_sha':'1'*64,'ops_root':'/private/tmp/ops','ops_code':'/private/tmp/code'}
        gpu={**cpu,'device':'cuda','cpu_request':'/private/tmp/cpu-request','cpu_request_sha':'2'*64}
        with patch.object(cli,'load_request',return_value=cpu),patch.object(cli,'successful_stage',return_value=({'runtime':{'different':True}},{})),patch.object(cli,'input_stage',side_effect=AssertionError('runtime must fail before payload')):
            with self.assertRaisesRegex(ValueError,'runtime'):cli.cpu_reference(gpu,{'runtime':{}})


if __name__=='__main__':unittest.main()
