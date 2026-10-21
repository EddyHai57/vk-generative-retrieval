import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

class NativeA1Tests(unittest.TestCase):
    def setUp(self):
        from analysis import sgns_native_subset_a1 as p
        self.p=p
    def test_cli_dry_and_illegal(self):
        for command,code in [('dry',0),('illegal',2)]:
            r=subprocess.run([sys.executable,'-B','-m','analysis.sgns_native_subset_a1',command],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
            self.assertEqual(r.returncode,code,r.stderr)
            if not code:self.assertFalse(json.loads(r.stdout)['training'])
    def test_fixed_increment_rejects_changed_old_extra_missing_and_dependency_pin(self):
        p=self.p;delta={**p.DEPENDENCIES,**{name:'b'*64 for name in p.NEW_FILES}};old={'commit':'a'*40,'files':{'old.py':'a'*64}};now={'commit':'c'*40,'files':{**old['files'],**delta}}
        with patch.object(p,'digest',side_effect=lambda x:now['files'][str(Path(x).relative_to(p.ROOT))]):
            self.assertEqual(p.check_delta(now,old,delta),delta)
            missing=dict(delta);missing.pop(p.NEW_FILES[0])
            with self.assertRaises(ValueError):p.check_delta({'files':{**old['files'],**missing}},old,missing)
            for bad in ({**now,'files':{**now['files'],'old.py':'d'*64}},{**now,'files':{**now['files'],'extra.py':'d'*64}},{**now,'files':old['files']}):
                with self.assertRaises(ValueError):p.check_delta(bad,old,delta)
            bad={**delta,next(iter(p.DEPENDENCIES)):'d'*64}
            with self.assertRaises(ValueError):p.check_delta({'files':{**old['files'],**bad}},old,bad)
    def test_facade_isolated_code_and_closed_callback_set(self):
        p=self.p;before=dict(p.legacy.__dict__);cb={name:(lambda *a,**k:None) for name in p.EXEC_CALLBACKS}
        def stop(*args):raise ValueError('fixture stop')
        cb['read_request']=stop;f=p.execution_facade(cb,object(),lambda *a:None)
        self.assertIs(f.__code__,p.legacy._execute.__code__);self.assertIs(f.__defaults__,p.legacy._execute.__defaults__);self.assertIs(f.__closure__,p.legacy._execute.__closure__)
        with self.assertRaisesRegex(ValueError,'fixture stop'):f('preflight',None,None,None)
        for key,value in before.items():self.assertIs(p.legacy.__dict__[key],value)
        with self.assertRaises(ValueError):p.execution_facade({**cb,'fit':stop},object(),stop)
        with self.assertRaises(ValueError):p.imported({}, {'os':object()})
    def test_runtime_json_subclass_boundary_preserves_typed_scientific_values(self):
        import torch
        from analysis import new_instance_prepare
        p=self.p;runtime={'torch':torch.__version__,'boolean':False,'integer':0,'floating':0.0}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.fixture(base);v['output_root']=str(base/'output');out=base/'output';out.mkdir()
            with patch.object(p,'source_identity',return_value=v['source']),patch.object(p.core,'check_native'),patch.object(p.core,'file_bytes'),patch.object(new_instance_prepare,'runtime',return_value=runtime),patch.object(p,'context'):
                env=p.identity(v)[1];self.assertIs(type(env['torch']),str);self.assertIs(type(env['boolean']),bool);self.assertIs(type(env['integer']),int);self.assertIs(type(env['floating']),float)
            receipt={'runtime':runtime,'verified':True,'hits':378};(out/'stage-receipt.json').write_bytes(p.encoded(receipt))
            with patch.object(p,'job_binding',return_value='a'*64):
                self.assertEqual(p.envelope(v,base/'request.json','b'*64,'derive',receipt)['science_receipt']['hits'],378)
                for key,bad in [('verified',1),('hits',378.0)]:
                    fresh=base/key;fresh.mkdir();(fresh/'stage-receipt.json').write_bytes(p.encoded(receipt))
                    with self.subTest(key=key),self.assertRaises(ValueError):p.envelope({**v,'output_root':str(fresh)},base/'request.json','b'*64,'derive',{**receipt,key:bad})
    def test_preflight_runtime_json_subclass_and_typed_mismatch(self):
        import torch
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);scratch=base/'scratch';(scratch/'preflight').mkdir(parents=True)
            runtime={'torch':torch.__version__,'flag':False};family={};parent=base/'family';parent.mkdir()
            for name in ('like-validation.json','catalog.json'):
                f=parent/name;f.write_text('[{}]');family[name]={'sha256':p.digest(f)}
            manifest=parent/'manifest.json';manifest.write_text(json.dumps({'artifacts':family}))
            v={'prepared':{'path':str(manifest),'sha256':p.digest(manifest)},'scratch_root':str(scratch)}
            receipt={'query_count':1,'catalog_size':1,'native_runtime':json.loads(p.encoded(runtime)),'payload_written':False,'scratch_written':True}
            with patch.object(p,'successful',return_value=(receipt,{},{})),patch.object(p,'engine_for',return_value=({'_runtime':lambda *_:runtime},(),{},{})),patch.object(p,'sid_manifest',return_value={'sha256':'a'*64}),patch.object(p,'identity',return_value=({},json.loads(p.encoded(runtime)))),patch.object(p.legacy,'receipt_binding',lambda *a,**k:None),patch.object(p,'job_binding',return_value='a'*64),patch.object(p,'view',return_value=({},None,None)):
                self.assertIs(p.predecessor(v,'train','b'*64,{},request=base/'request.json'),receipt)
                receipt['native_runtime']['flag']=0
                with self.assertRaisesRegex(ValueError,'preflight output semantics'):p.predecessor(v,'train','b'*64,{},request=base/'request.json')
    def test_original_config_and_integer_four_gate_rejection(self):
        p=self.p;config=p.core.registered_config('a'*64)
        self.assertEqual(config['seed'],42)
        proof={'health':{'passed':True},'sid_gates':{'cpu':{'passed':True},'cuda':{'passed':True}},'collision':{'cpu_count':10,'gpu_count':11,'items':100,'passed':True,'comparison':'gpu_count * 10 <= cpu_count * 11'},'validation_hits':378,'hits_gate':378,'a1_native':{'status':'verified'},'evidence':{'/fixture/evidence':'a'*64}}
        self.assertEqual(p.acceptance.check_proof(proof),proof)
        for key,val in [('validation_hits',377),('validation_hits',378.),('hits_gate',377),('health',{'passed':False})]:
            with self.assertRaises(ValueError):p.acceptance.check_proof({**proof,key:val})
    def fixture(self,base):
        p=self.p
        return {'schema':p.SOURCE_SCHEMAS['derive'],'source':{'commit':'a'*40,'files':{'old.py':'b'*64}},'lock':{'path':str(base/'lock'),'sha256':p.core.NATIVE_LOCK_SHA},
            'sid_root':str(base/'sid-code'),'sid_request':{'path':str(base/'sid-request.json'),'sha256':'a'*64},'producer_delta':{**p.DEPENDENCIES,**{x:'b'*64 for x in p.NEW_FILES}},
            'ops_root':str(base/'ops'),'ops_code':str(base/'ops-code'),'raw_data':str(base/'raw'),'ur001':{'path':str(base/'ur001'),'sha256':'c'*64},
            'content':{'path':str(base/'content.npy'),'sha256':p.core.INPUT_PINS['content']},'output_root':str(base/'family')}
    def test_schema_frozen_scientific_pins_and_exclusive_paths(self):
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.fixture(base);path=base/'request.json'
            def load(value,**kwargs):path.write_text(json.dumps(value));return p.load(path,p.digest(path),**kwargs)
            self.assertEqual(load(v),v)
            for key,value in [('schema','old'),('producer_delta',{}),('extra',False),('output_root',v['raw_data']+'/child')]:
                with self.subTest(key=key),self.assertRaises(ValueError):load({**v,key:value})
            bad={**v,'lock':{**v['lock'],'sha256':'e'*64}}
            with self.assertRaises(ValueError):load(bad)
            self.assertEqual(load(bad,frozen=False),bad)
    def test_successful_stage_strict_outer_runtime_proof_terminal_and_job(self):
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.fixture(base);path=base/'request.json';path.write_text('{}');pin=p.digest(path);stage='derive';out=base/'family';out.mkdir();run=base/'ops/runs'/p.RUN_IDS[stage];(run/'logs').mkdir(parents=True)
            (run/'run.json').write_text('{}');inner={'transport':'fixture'};(out/'stage-receipt.json').write_text(json.dumps(inner));raw=(out/'stage-receipt.json').read_bytes()
            outer={'schema':p.SCHEMA,'backend':'B','authority':'4b0a8a1','status':'verified','verified':True,'engineering':False,'stage':stage,'run_id':p.RUN_IDS[stage],
                'run_manifest_sha256':'a'*64,'request_sha256':pin,'source':v['source'],'science_receipt':inner,'science_receipt_sha256':p.hashlib.sha256(raw).hexdigest(),
                'test_read':False,'subset_four_gate_acceptance':False,'sid_request':v['sid_request']}
            terminal={'schema':'vkgr-terminal-v1','run_id':p.RUN_IDS[stage],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}
            def write(value,term):
                (out/'native-receipt.json').write_text(json.dumps(value));(run/'logs/stdout.log').write_text(json.dumps(value));(run/'terminal.json').write_text(json.dumps(term))
            with patch.object(p,'job_binding',return_value='a'*64):
                write(outer,terminal);self.assertEqual(p.successful(v,path,pin,stage)[0],inner)
                for key,value in [('backend','A'),('authority','ad1c428'),('verified',1),('science_receipt_sha256','d'*64),('sid_request',{}),('source',{})]:
                    write({**outer,key:value},terminal)
                    with self.subTest(outer=key),self.assertRaises(ValueError):p.successful(v,path,pin,stage)
                for key,value in [('wall_cap_exceeded',True),('wall_cap_exceeded',0),('wall_cap_exceeded',None),('signal',9),('exit_code',False),('schema','wrong')]:
                    term={**terminal,key:value}
                    if key=='wall_cap_exceeded' and value is None:term.pop(key)
                    write(outer,term)
                    with self.subTest(terminal=key,value=value),self.assertRaises(ValueError):p.successful(v,path,pin,stage)
    def test_consume_rejects_nonacceptance_before_request_or_replay(self):
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);path=base/'receipt.json'
            for value in ({},{'schema':p.SCHEMA,'verified':True},{'schema':p.TOTAL,'backend':'A','verified':True},{'schema':p.TOTAL,'backend':'B','verified':1}):
                path.write_text(json.dumps({**value,'request':{'path':str(base/'request.json'),'sha256':'a'*64}}))
                with patch.object(p,'load',side_effect=AssertionError('receipt rejected before request')),patch.object(p,'replay',side_effect=AssertionError('no replay')):
                    with self.assertRaises(ValueError):p.consume(path,p.digest(path),base/'scratch')
    def test_real_public_trees_dependency_imports_launcher_metadata_job_and_dry(self):
        import shutil
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d)
            def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
            def clone(name,ref):
                root=base/name;subprocess.run(['git','clone','-q','--shared','--no-checkout',str(ROOT),str(root)],check=True,capture_output=True);git(root,'checkout','-q','--detach',ref);return root
            runner=clone('a1-code','f36bf15a1844e2f305c8fd2c9d71655e2465ee29');sidroot=clone('sid-code','f36bf15a1844e2f305c8fd2c9d71655e2465ee29');native=clone('native-code','305c415871bd60dc0d7d8a064a7e893830a9c20f');ops=clone('ops-code','5840e6b686ec9651314a9e28a1a2c0befcc05158')
            for name in (*p.DEPENDENCIES,*p.NEW_FILES):
                dest=runner/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
            git(runner,'add',*p.DEPENDENCIES,*p.NEW_FILES);git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-q','-m','A1 original dependency closure fixture')
            env={**os.environ,'PYTHONPATH':str(runner/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
            def identity(root):
                e={**env,'PYTHONPATH':str(root/'src')};r=subprocess.run([sys.executable,'-B','-c','from analysis.ur001_derive_cli import source_identity; import json; print(json.dumps(source_identity()))'],cwd=root,env=e,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);return json.loads(r.stdout)
            source=identity(runner);ssource=identity(sidroot)
            # Real import/default-config probing across the actual public tree,
            # with only original-byte dependencies and the three new files.
            probe=subprocess.run([sys.executable,'-B','-c','from analysis import sgns_native_subset_a1 as p; p.core.check_native(); print(p.core.registered_config("a"*64)["seed"])'],cwd=runner,env=env,capture_output=True,text=True)
            self.assertEqual(probe.returncode,0,probe.stderr);self.assertEqual(probe.stdout.strip(),'42')
            lock=base/'lock';lock.write_bytes(subprocess.check_output(['git','-C',str(ROOT),'show',p.core.ORIGINAL+':uv.lock']))
            bundle=base/'bundle';bundle.mkdir()
            for name in ('manifest.json','catalog.json','content.npy','histories.jsonl'):(bundle/name).write_text('{}')
            n={'schema':'sgns-gpu-native-fit-request-v1','runner_source':git(native,'rev-parse','HEAD'),'bundle':str(bundle),'bundle_sha256':p.digest(bundle/'manifest.json'),'lock':str(lock),
               'ops_root':str(base/'native-ops'),'output':str(base/'native-output'),'preflight_run_id':'sgns-gpu-native-fit-preflight-fixture-v1','fit_run_id':'sgns-gpu-native-fit-fit-fixture-v1',
               'route_selected':True,'benchmark_reference':None,'compiled':True,'resources':{'cpu_cores':2,'ram_mib':16384,'vram_mib':8192,'disk_gib':1,'wall_hours':3}}
            np=base/'native-request.json';np.write_text(json.dumps(n));nout=base/'native-output/fit';nout.mkdir(parents=True)
            for name in ('result.json','vectors.npy','audit.json'):(nout/name).write_text('{}')
            def files(parent,names):
                for name in names:
                    path=parent/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            files(base/'native-ops/runs'/n['fit_run_id'],('run.json','terminal.json','logs/stdout.log'))
            cpu={'schema':'gpu-sgns-subset-sid-request-v1','device':'cpu','source':{'commit':'a'*40,'files':{}},'lock':str(lock),'ops_root':str(base/'cpu-ops'),'ops_code':str(ops),
                'bundle':str(bundle),'bundle_sha':n['bundle_sha256'],'sgns_output':str(base/'cpu-sgns'),'sgns_receipt_sha':'a'*64,'sgns_run':str(base/'cpu-sgns-run'),
                'preflight_output':str(base/'cpu-sid-preflight'),'fit_output':str(base/'cpu-sid-fit'),'verify_output':str(base/'cpu-sid-verify'),'cpu_request':None,'cpu_request_sha':None}
            cp=base/'cpu-request.json';cp.write_text(json.dumps(cpu));files(base/'cpu-ops/runs/sgns-gpu-subset-cpu-sid-verify-v1',('run.json','terminal.json','logs/stdout.log'));files(base/'cpu-sid-verify',('stage-receipt.json',));files(base/'cpu-sid-fit',('run_manifest.json',))
            sr={'schema':'sgns-native-subset-sid-request-v1','source':ssource,'lock':str(lock),'ops_root':str(base/'sid-ops'),'ops_code':str(ops),'native_root':str(native),'native_request':str(np),'native_request_sha256':p.digest(np),
                'producer_delta':{name:p.digest(sidroot/name) for name in p.sid.ADDITIONS},'cpu_root':str(base/'cpu-code'),'cpu_request':str(cp),'cpu_request_sha256':p.digest(cp),
                'outputs':{name:str(base/('sid-'+name)) for name in ('preflight','fit','verify')},'run_ids':{name:'sgns-native-subset-sid-'+name+'-fixture-v1' for name in ('preflight','fit','verify')}}
            sp=base/'sid-request.json';sp.write_text(json.dumps(sr))
            for stage in ('fit','verify'):
                files(base/('sid-'+stage),('stage-receipt.json','native-receipt.json'));files(base/'sid-ops/runs'/sr['run_ids'][stage],('run.json','terminal.json','logs/stdout.log'))
            content=base/'content.npy';content.write_text('transport fixture only');ur=base/'ur001';ur.write_text('transport fixture only')
            v=self.fixture(base);v.update({'source':source,'lock':{'path':str(lock),'sha256':p.digest(lock)},'sid_root':str(sidroot),'sid_request':{'path':str(sp),'sha256':p.digest(sp)},'ops_code':str(ops),
                'producer_delta':{name:p.digest(runner/name) for name in (*p.DEPENDENCIES,*p.NEW_FILES)},'ur001':{'path':str(ur),'sha256':p.digest(ur)},'content':{'path':str(content),'sha256':p.digest(content)}})
            path=base/'request.json';path.write_text(json.dumps(v));pin=p.digest(path);cmd=[sys.executable,'-B','-m',p.MODULE]
            args=cmd+['plan','--stage','derive','--request',str(path),'--request-sha256',pin,'--ops-code',str(ops)]
            planned=subprocess.run(args,cwd=runner,env=env,capture_output=True,text=True);self.assertEqual(planned.returncode,0,(planned.stdout,planned.stderr));got=json.loads(planned.stdout)
            self.assertGreater(len(got['metadata']['inputs']),25);self.assertFalse(got['real_preflight'])
            run=base/'ops/runs'/p.RUN_IDS['derive'];run.mkdir(parents=True);metadata=got['metadata'];spec=got['spec'];record={**{key:metadata[key] for key in ('cwd','git','config','inputs','overlay','pinned_files')},
                'schema':'vkgr-run-v1','run_id':spec['run_id'],'kind':'ops','seed':42,'argv':spec['argv'],'resources':{'cpu_cores':1,'ram_mib':98304,'vram_mib':0,'disk_gib':4,'wall_seconds':7200.},
                'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158','sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},'stop_mode':'signal',
                'env':{**metadata['env'],'VKGR_RUN_ID':spec['run_id'],'VKGR_PROGRESS_PATH':str(run/'progress.json')}}
            (run/'run.json').write_text(json.dumps(record))
            code='from analysis import sgns_native_subset_a1 as p; from pathlib import Path; v=p.load(Path('+repr(str(path))+'),'+repr(pin)+',frozen=False); print(p.job_binding(v,Path('+repr(str(path))+'),'+repr(pin)+',"derive"))'
            bound=subprocess.run([sys.executable,'-B','-c',code],cwd=runner,env=env,capture_output=True,text=True);self.assertEqual(bound.returncode,0,(bound.stdout,bound.stderr));self.assertEqual(bound.stdout.strip(),p.digest(run/'run.json'))
            # The actual scientific stage never receives dry's frozen=False.
            rejected=subprocess.run(cmd+['derive','--request',str(path),'--request-sha256',pin],cwd=runner,env=env,capture_output=True,text=True);self.assertEqual(rejected.returncode,2);self.assertIn('lock/content pin differs',rejected.stdout)
            dry=subprocess.run(cmd+['dry'],cwd=runner,env=env,capture_output=True,text=True);self.assertEqual(dry.returncode,0,dry.stderr);self.assertFalse(json.loads(dry.stdout)['training']);self.assertFalse((base/'ops/queue').exists())
    def test_original_train_code_reaches_model_boundary_with_fixed_progress(self):
        from types import SimpleNamespace
        p=self.p
        from vkgr import sid_structure_a1_run as original
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);scratch=base/'scratch';scratch.mkdir();value={'data_root':str(base/'raw'),'output_root':str(base/'out'),'scratch_root':str(scratch),'gpu_sid':{'manifest':{'sha256':'a'*64}}};received={};config=p.core.registered_config('a'*64)
            def boundary(codec,actual,**kwargs):
                self.assertEqual(actual,config);self.assertEqual(received['progress'],{'path':scratch/'training-progress.json','run_id':p.RUN_IDS['train']});self.assertEqual(received['eval_workers'],1);self.assertIsNone(received['decode']);self.assertEqual(received['temporary'],scratch/'train');raise RuntimeError('original model boundary reached without model')
            namespace={**original.run.__globals__,'_contexts':lambda *a,**k:(SimpleNamespace(codec=None),False,{}),'_runtime':lambda *a:{'device':'cpu'},'_evaluator_d':lambda *a:None,'_guard':lambda *a:None,'gr':SimpleNamespace(_seed=lambda seed:None),'make_model_arch_d':boundary}
            native=p.clone(original.run,namespace)
            def run(*a,**k):received.update(k);return native(*a,**k)
            proxy=SimpleNamespace(**{**p.core.__dict__,'RUN_IDS':p.RUN_IDS,'engine_for':lambda *a:({'run':run},(base,None,None,None,None,None,None,None,None),{},{}),'registered_config':lambda _:config})
            callbacks={'read_request':lambda *a:value,'identity':lambda *_:({},{}),'own_job':lambda *a:'b'*64,'predecessor':lambda *a,**k:None};fn=p.execution_facade(callbacks,proxy,lambda *a:None)
            with patch.dict(os.environ,{'VKGR_DATA_ROOT':value['data_root']}),self.assertRaisesRegex(RuntimeError,'original model boundary reached without model'):fn('train',base/'request.json','c'*64,SimpleNamespace())
    def test_engine_reuses_native_bytecode_with_scoped_input_and_sid_dependencies(self):
        from vkgr import sid_structure_a1_run as original
        p=self.p;before=dict(p.core.__dict__)
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);lock=base/'lock';lock.write_bytes(subprocess.check_output(['git','show',p.core.ORIGINAL+':uv.lock']));family=base/'family.json';family.write_text(json.dumps({'runtime':{'fixture':True}}))
            value={'input_producer':{},'lock':{'path':str(lock),'sha256':p.digest(lock)},'prepared':{'path':str(family),'sha256':p.digest(family)},'producer_delta':{},'history':{'path':str(base/'history'),'sha256':'a'*64},'content':{'path':str(base/'content'),'sha256':'b'*64},'data_root':str(base/'data')}
            value.update({'gpu_sgns':{'receipt':{'path':str(base/'native/result.json'),'sha256':'c'*64}},'cpu_sgns':{'receipt':{'sha256':'d'*64}},'gpu_sid':{'manifest':{'path':str(base/'sid/run_manifest.json'),'sha256':'e'*64},'input_manifest':{'path':str(base/'sid/input-manifest.json'),'sha256':'f'*64},'producer':{'receipt':{'sha256':'1'*64}}},'cpu_sid':{'producer':{'receipt':{'sha256':'2'*64}}}})
            proof={'gpu_sid':{'source':{'files':{'old':'a'*64}}},'gpu_sgns':{'backend':'B','artifacts':{'vectors.npy':'3'*64}}};source={'commit':'a'*40,'files':{}}
            with patch.object(p,'view',return_value=(value,proof,{})),patch.object(p,'input_check',return_value={'runtime':{'fixture':True}}),patch.object(p,'check_delta',return_value={}):
                engine,common,extra,got=p.engine_for({'source':source},'b'*64,source)
                self.assertEqual(got['gpu_sgns']['backend'],'B');self.assertNotIn('original_audit_verification',got['gpu_sgns']);self.assertEqual(extra['vectors_sha'],'3'*64)
                for name in ('run','verify','_contexts','_binding','_load_sid'):self.assertIs(engine[name].__code__,getattr(original,name).__code__)
                self.assertEqual(common[0],p.ROOT)
            for key,obj in before.items():self.assertIs(p.core.__dict__[key],obj,key)
    def test_output_roots_do_not_overlap_any_immutable_producer(self):
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.fixture(base);v['scratch_root']=str(base/'scratch');evidence=[str(base/'frozen/evidence')]
            p.protect_outputs(v,evidence)
            for role in ('output_root','scratch_root'):
                for root in (base/'frozen/evidence',base/'frozen/evidence/child',base/'frozen'):
                    with self.subTest(role=role,root=root),self.assertRaises(ValueError):p.protect_outputs({**v,role:str(root)},evidence)
    def test_context_replay_scratch_excludes_all_evidence_and_request_records(self):
        from types import SimpleNamespace
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.fixture(base)
            v.update({k:str(base/k) for k in ('sid_root','ops_root','ops_code','data_root','scratch_root')})
            for k in ('prepared','history','content','input_request','lock','sid_request'):v[k]={'path':str(base/k/'record.json'),'sha256':'a'*64}
            request={'source':{},'lock':v['lock']['path'],'outputs':{'fit':str(base/'sid-fit'),'verify':str(base/'sid-verify')}}
            request.update({k:str(base/k) for k in ('ops_root','native_root','cpu_root','native_request','cpu_request')});request['cpu_request_sha256']='a'*64
            native={k:str(base/('native-'+k)) for k in ('bundle','output','ops_root')}
            cpu={k:str(base/('cpu-'+k)) for k in ('bundle','sgns_output','sgns_run','fit_output','verify_output','ops_root')}
            module=SimpleNamespace(load=lambda *a:request,native_context=lambda *a:(None,native),legacy=SimpleNamespace(load_request=lambda *a:cpu))
            path=base/'a1-request.json';evidence=[p.ROOT,*request['outputs'].values(),*(request[k] for k in ('ops_root','native_root','cpu_root','native_request','cpu_request')),*native.values(),*cpu.values(),path,
                *(v[k] for k in ('sid_root','ops_root','ops_code','data_root','output_root','scratch_root')),*(v[k]['path'] for k in ('prepared','history','content','input_request','lock','sid_request'))]
            with patch.object(p,'digest',return_value='a'*64),patch.object(p.sid,'module_at',return_value=module),patch.object(p.acceptance,'source_tree'),patch.object(p,'check_delta'):
                p.context(v,replay=(path,base/'safe-replay'))
                for root in evidence:
                    root=Path(root)
                    for dest in (root,root/'nested',root.parent):
                        with self.subTest(root=root,dest=dest),self.assertRaisesRegex(ValueError,'replay scratch'):p.context(v,replay=(path,dest))
                existing=base/'exists';existing.mkdir()
                with self.assertRaisesRegex(ValueError,'replay scratch'):p.context(v,replay=(path,existing))
    def test_readonly_replay_original_selection_integer_hits_and_four_gates(self):
        import copy
        from types import SimpleNamespace
        from unittest.mock import Mock
        from vkgr.generative_hp import EarlyStop
        from vkgr.data_audit import canonical_bytes
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v={'source':{'commit':'a'*40,'files':{}},'sid_root':str(base/'sid-code'),'data_root':str(base/'family'),'output_root':str(base/'out'),'scratch_root':str(base/'original-scratch'),'ops_root':str(base/'ops')};path=base/'request.json';path.write_text('{}');pin=p.digest(path);env={'fixture':True};config=p.core.registered_config('d'*64)
            queries=[{'target':i} for i in range(400)];rows=[{'ordinal':i,'query_sha256':p.hashlib.sha256(canonical_bytes(q)).hexdigest(),'items':[i] if i<378 else []} for i,q in enumerate(queries)]
            metrics={'metrics':{'all_eligible':{'count':400,'recall@50':378/400,'ndcg@50':.9}}};epochs=[];stop=EarlyStop(100,5,3)
            for i in range(1,9):
                stop.observe(i,378/400,.9);epochs.append({'epoch':i,'metrics':metrics,'checkpoint_sha256':'e'*64,'evaluation':{'kind':'synthetic-test-only','computation_sha256':'f'*64,'path':f'evaluation-{i:03d}'}})
            native={'epochs':epochs,'selection':stop.state(),'best':{'epoch':1,'sha256':'e'*64,'metric':list(stop.best_metric)}};trained={'native_manifest_sha256':'c'*64}
            expected_native={'status':'verified','replayed_epochs':[1],'best_epoch':1,'epochs':8,'run_id':'train','manifest_sha256':'c'*64,'verify_run_id':p.RUN_IDS['verify']}
            proof={'native':'fixture'};verified={'schema':'gpu-sgns-subset-a1-stage-v1','authority':'ad1c428','engineering':False,'stage':'verify','status':'verified','verified':True,'run_id':p.RUN_IDS['verify'],
                'run_manifest_sha256':'b'*64,'request_sha256':pin,'source':v['source'],'runtime':env,'config':config,'input_producers':proof,'test_read':False,'subset_four_gate_acceptance':False,
                'hits_gate':378,'training':False,'resource_scope':p.legacy.RESOURCE_SCOPE,'native_verification':expected_native,'validation_hits':378,'context_inclusive_peak_mib':1.,'context_samples':1,'rss_tree_upper_bound_mib':1.}
            for stage in ('preflight','train','verify'):
                run=base/'ops/runs'/p.RUN_IDS[stage];(run/'logs').mkdir(parents=True)
                for name in ('run.json','terminal.json','logs/stdout.log'):(run/name).write_text('{}')
                (base/'out'/stage).mkdir(parents=True)
            replay=base/'out/verify/native/replay-best-001';replay.mkdir(parents=True);(replay/'manifest.json').write_text('{}')
            cpu=base/'cpu-fit';cpu.mkdir();cpu_manifest={'audit':{'items':100,'semantic_collision_items':10},'gates':{'passed':True}};(cpu/'run_manifest.json').write_text(json.dumps(cpu_manifest))
            collision=p.sid.science.collision_gate(cpu_manifest['audit'],{'items':100,'semantic_collision_items':11});got={'job_proof':{'fixed':True},'cpu_reference':{'manifest_sha256':p.digest(cpu/'run_manifest.json')},
                'loaded':SimpleNamespace(manifest={'audit':{'items':100,'semantic_collision_items':11},'gates':{'passed':True},'collision_gate':collision}),'native_input':{'sgns_receipt':{'native_audit_verification':{'health':{'passed':True}}}}}
            module=SimpleNamespace(legacy=SimpleNamespace(load_request=lambda *a:{'fit_output':str(cpu)}));request={'cpu_request':str(base/'cpu-request.json'),'cpu_request_sha256':'a'*64}
            engine={'_runtime':lambda *_:env,'_contexts':lambda *a,**k:(SimpleNamespace(codec=object(),queries=queries),False,None),'_read_evaluation':lambda *a:(metrics,'f'*64,rows),
                'run':Mock(side_effect=AssertionError('training forbidden')),'verify':Mock(side_effect=AssertionError('model verification forbidden'))}
            count=0
            def call():
                nonlocal count
                count+=1;return p.replay(v,path,pin,base/('recompute-'+str(count)))
            with patch.object(p,'context') as replay_context,patch.object(p,'identity',return_value=(v['source'],env)),patch.object(p,'producer_closure',return_value=(module,request,got)),patch.object(p,'input_check'),patch.object(p,'view',return_value=(v,{},got)), \
                 patch.object(p,'engine_for',return_value=(engine,(),{},proof)),patch.object(p,'sid_manifest',return_value={'sha256':'d'*64}),patch.object(p,'successful',return_value=(verified,{},{})),patch.object(p,'predecessor',return_value=trained), \
                 patch.object(p,'job_binding',return_value='b'*64),patch.object(p.legacy,'train_output_binding',return_value=native),patch.object(p,'stage_inputs',return_value=[]),patch.object(p,'load',return_value=v),patch('vkgr.sid_depth.SIDTokensLD',return_value=object()):
                result=call();replay_context.assert_called_once_with(v,replay=(path,base/'recompute-1'));self.assertEqual(result['validation_hits'],378);self.assertEqual(set(result),{'health','sid_gates','collision','validation_hits','hits_gate','a1_native','evidence'})
                accept=base/'out/accept';accept.mkdir();(accept/'stage-receipt.json').write_text('{}');self.assertEqual(call(),result)
                for key,bad in [('validation_hits',377),('validation_hits',378.),('native_verification',{'status':'verified'}),('hits_gate',377)]:
                    old=verified[key];verified[key]=bad
                    with self.subTest(verified=key),self.assertRaises(ValueError):call()
                    verified[key]=old
                native['best']['epoch']=2
                with self.assertRaisesRegex(ValueError,'selected epoch'):call()
                native['best']['epoch']=1
                got['native_input']['sgns_receipt']['native_audit_verification']['health']['passed']=False
                with self.assertRaisesRegex(ValueError,'four-gate'):call()
                got['native_input']['sgns_receipt']['native_audit_verification']['health']['passed']=True
                got['loaded'].manifest['collision_gate']={}
                with self.assertRaisesRegex(ValueError,'collision'):call()
                engine['run'].assert_not_called();engine['verify'].assert_not_called()
    def test_consume_verified_total_requires_replay_equality_and_stable_original_terminal(self):
        import copy
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v={'source':{'commit':'a'*40,'files':{}},'ops_root':str(base/'ops'),'output_root':str(base/'out')};request=base/'request.json';request.write_text('{}');req={'path':str(request),'sha256':p.digest(request)};env={'fixed':True}
            out=base/'out/accept';out.mkdir(parents=True);path=out/'stage-receipt.json';run=base/'ops/runs'/p.RUN_IDS['accept'];(run/'logs').mkdir(parents=True);(run/'run.json').write_text('{}')
            proof={'health':{'passed':True},'sid_gates':{'cpu':{'passed':True},'cuda':{'passed':True}},'collision':{'cpu_count':10,'gpu_count':11,'items':100,'passed':True,'comparison':'gpu_count * 10 <= cpu_count * 11'},'validation_hits':378,'hits_gate':378,'a1_native':{'status':'verified'},'evidence':{str(base/'evidence'):'a'*64}}
            receipt={'schema':p.TOTAL,'authority':'4b0a8a1','backend':'B','status':'verified','verified':True,'subset_four_gate_acceptance':True,'engineering':False,'test_read':False,'training':False,'run_id':p.RUN_IDS['accept'],
                'run_manifest_sha256':'b'*64,'source':v['source'],'runtime':env,'request':req,'proof':proof}
            terminal={'schema':'vkgr-terminal-v1','run_id':p.RUN_IDS['accept'],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}
            def write(value,term=terminal):path.write_text(json.dumps(value));(run/'terminal.json').write_text(json.dumps(term));(run/'logs/stdout.log').write_text(json.dumps(value));return p.digest(path)
            with patch.object(p,'load',return_value=v),patch.object(p,'identity',return_value=(v['source'],env)),patch.object(p,'job_binding',return_value='b'*64),patch.object(p,'replay',return_value=proof):
                pin=write(receipt);self.assertEqual(p.consume(path,pin,base/'scratch')['proof'],proof)
                for key,bad in [('wall_cap_exceeded',True),('wall_cap_exceeded',0),('wall_cap_exceeded',None),('signal',9),('exit_code',False)]:
                    term={**terminal,key:bad}
                    if bad is None and key=='wall_cap_exceeded':term.pop(key)
                    pin=write(receipt,term)
                    with self.subTest(terminal=key,value=bad),self.assertRaises(ValueError):p.consume(path,pin,base/'scratch')
                for key,bad in [('source',{}),('runtime',{}),('run_manifest_sha256','c'*64)]:
                    pin=write({**receipt,key:bad})
                    with self.subTest(receipt=key),self.assertRaises(ValueError):p.consume(path,pin,base/'scratch')
                pin=write(receipt)
                with patch.object(p,'replay',return_value={**proof,'validation_hits':379}),self.assertRaisesRegex(ValueError,'independently recomputed'):p.consume(path,pin,base/'scratch')
                def mutate(*args):
                    (run/'terminal.json').write_text(json.dumps({**terminal,'wall_cap_exceeded':True}));return proof
                pin=write(receipt)
                with patch.object(p,'replay',side_effect=mutate),self.assertRaisesRegex(ValueError,'terminal/stdout changed'):p.consume(path,pin,base/'scratch')
    def test_own_job_full_launcher_contract_rejects_resource_env_argv_and_inputs(self):
        import copy
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.fixture(base);path=base/'request.json';path.write_text('{}');pin=p.digest(path);rid=p.RUN_IDS['derive'];run=base/'ops/runs'/rid;run.mkdir(parents=True)
            expected={'schema':'vkgr-run-v1','run_id':rid,'kind':'ops','cwd':str(p.ROOT),'argv':p.invocation('derive',path,pin),'seed':42,'git':{'commit':v['source']['commit'],'dirty':False,'dirty_files':[]},
                'config':{'path':str(path),'sha256':pin},'inputs':[{'path':str(path),'sha256':pin}],'resources':p.resources(v,'derive'),
                'env':{**p.fixed_env(v,'derive'),'VKGR_RUN_ID':rid,'VKGR_PROGRESS_PATH':str(run/'progress.json')}}
            record={**expected,'inputs':[{**expected['inputs'][0],'bytes':2}],'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158','sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},'overlay':None,'pinned_files':{},'stop_mode':'signal'}
            def write(value):(run/'run.json').write_text(json.dumps(value))
            with patch.object(p,'contract',return_value=expected):
                write(record);self.assertEqual(p.job_binding(v,path,pin,'derive'),p.digest(run/'run.json'))
                for key,bad in [('argv',[]),('cwd',str(base)),('seed',43),('resources',{**record['resources'],'vram_mib':1}),('env',{**record['env'],'VKGR_EXTRA':'1'}),('inputs',[]),('ops_source',{}),('overlay',{})]:
                    write({**record,key:bad})
                    with self.subTest(key=key),self.assertRaises(ValueError):p.job_binding(v,path,pin,'derive')
                write(record)
                with patch.dict(os.environ,expected['env']):self.assertEqual(p.own_job(v,path,pin,'derive'),p.digest(run/'run.json'))
                with patch.dict(os.environ,{**expected['env'],'VKGR_RUN_ID':'wrong'}),self.assertRaises(ValueError):p.own_job(v,path,pin,'derive')
