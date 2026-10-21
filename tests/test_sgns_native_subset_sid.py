import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

class TransportTests(unittest.TestCase):
    def setUp(self):
        from analysis import sgns_native_subset_sid as p
        self.p=p
    def test_real_cli_dry_and_illegal(self):
        for command,code in [('dry',0),('illegal',2)]:
            run=subprocess.run([sys.executable,'-B','-m','analysis.sgns_native_subset_sid',command],cwd=ROOT,
                env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
            self.assertEqual(run.returncode,code,run.stderr)
            if code==0:self.assertFalse(json.loads(run.stdout)['training'])
    def test_exact_pinned_increment_rejects_old_changes_missing_extra_and_overlap(self):
        p=self.p;producer={'commit':'a'*40,'files':{'old.py':'1'*64,**p.science.SCIENTIFIC_FILES}}
        delta={name:'2'*64 for name in p.ADDITIONS};delta['analysis/sgns_subset_sid.py']=p.FROZEN['analysis/sgns_subset_sid.py'];current={'commit':'b'*40,'files':{**producer['files'],**delta}}
        self.assertEqual(p.check_delta(current,producer,delta),delta)
        for bad in ({**current,'files':{**current['files'],'old.py':'3'*64}},
                    {**current,'files':{**current['files'],'extra.py':'3'*64}},
                    {**current,'files':{k:v for k,v in current['files'].items() if k!=p.ADDITIONS[0]}}):
            with self.assertRaises(ValueError):p.check_delta(bad,producer,delta)
        with self.assertRaises(ValueError):p.check_delta(current,producer,{**delta,'extra.py':'3'*64})
        wrong={**delta,'analysis/sgns_subset_sid.py':'3'*64}
        with self.assertRaises(ValueError):p.check_delta({**current,'files':{**producer['files'],**wrong}},producer,wrong)
        with self.assertRaises(ValueError):p.check_delta(current,{'commit':'a'*40,'files':current['files']},delta)
    def test_namespace_facade_same_code_defaults_closure_and_globals_never_mutated(self):
        p=self.p;original_cli=dict(p.legacy.__dict__);original_science=dict(p.science.__dict__)
        callbacks={name:(lambda *a,**kw:None) for name in p.CALLBACKS}
        def fail(*args):raise ValueError('synthetic transport stop')
        callbacks['load_request']=fail
        cloned=p.facade(callbacks,lambda *args:{})
        self.assertIsNot(cloned.__globals__,p.legacy.__dict__)
        self.assertEqual({key for key in cloned.__globals__ if cloned.__globals__[key] is not p.legacy.__dict__[key]},p.CALLBACKS|{'science'})
        self.assertIs(cloned.__code__,p.legacy.execute.__code__)
        self.assertIs(cloned.__defaults__,p.legacy.execute.__defaults__)
        self.assertIs(cloned.__closure__,p.legacy.execute.__closure__)
        proxy=cloned.__globals__['science']
        for name in ('fit_bhkm_coll','audit_gates','collision_gate','validate_input'):
            self.assertIs(getattr(proxy,name),getattr(p.science,name))
        self.assertIs(proxy.load_bundle.__code__,p.science.load_bundle.__code__)
        with self.assertRaisesRegex(ValueError,'synthetic transport stop'):cloned('preflight',None,None)
        for original,module in ((original_cli,p.legacy),(original_science,p.science)):
            self.assertEqual(set(original),set(module.__dict__))
            for name,obj in original.items():self.assertIs(module.__dict__[name],obj,name)
        with self.assertRaises(ValueError):p.facade({**callbacks,'fit_bhkm_coll':lambda:None},lambda *a:None)
    def request(self,base):
        return {'schema':'sgns-native-subset-sid-request-v1','source':{'commit':'a'*40,'files':{'x.py':'b'*64}},'lock':str(base/'lock'),
            'ops_root':str(base/'ops'),'ops_code':str(base/'ops-code'),'native_root':str(base/'native-code'),
            'native_request':str(base/'native-request.json'),'native_request_sha256':'b'*64,
            'producer_delta':{name:self.p.FROZEN.get(name,'c'*64) for name in self.p.ADDITIONS},'cpu_root':str(base/'cpu-code'),
            'cpu_request':str(base/'cpu-request.json'),'cpu_request_sha256':'d'*64,
            'outputs':{name:str(base/name) for name in ('preflight','fit','verify')},
            'run_ids':{name:'sgns-native-subset-sid-'+name+'-fixture-v1' for name in ('preflight','fit','verify')}}
    def test_request_schema_paths_roles_and_types_closed(self):
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            value=self.request(Path(d));self.assertEqual(p.validate(value),value)
            changes={'schema':'old','test':False,'producer_delta':{},'native_request':'/workspace/test/a.json','run_ids':{},'outputs':{}}
            for key,val in changes.items():
                with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):p.validate({**value,key:val})
            bad=copy.deepcopy(value);bad['outputs']['fit']=bad['outputs']['preflight']
            with self.assertRaises(ValueError):p.validate(bad)
            bad=copy.deepcopy(value);bad['outputs']['fit']=bad['native_root']+'/fit'
            with self.assertRaises(ValueError):p.validate(bad)
    def test_original_pf_bytecode_runs_in_facade_without_fit_or_global_change(self):
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.request(base);im={'native':'input'};native={'job_proof':{'receipt_sha256':'a'*64},'input_manifest':im}
            normalized=p.normalized(v,native);originals={name:getattr(p.science,name) for name in ('fit_bhkm_coll','audit_gates','load_bundle','check_source_delta')}
            callbacks={'load_request':lambda *_:normalized,'input_stage':lambda *_:native,'cpu_reference':lambda *_:({'items':120415},{}),
                'own_job':lambda *_:{'run_id':v['run_ids']['preflight'],'run_manifest_sha256':'b'*64},'successful_stage':lambda *_:None}
            run=p.facade(callbacks,lambda *_:{})
            run.__globals__['source_identity']=lambda:v['source'];run.__globals__['guards']=lambda *a,**kw:None
            with patch('analysis.new_instance_prepare.runtime',return_value={'fixed':'runtime'}),patch.object(p.science,'fit_bhkm_coll',side_effect=AssertionError('PF must not fit')):
                receipt=run('preflight',base/'request.json','c'*64)
                self.assertFalse(receipt['payload_written']);self.assertFalse(receipt['mapping_replay'])
                self.assertTrue((base/'preflight/stage-receipt.json').is_file())
            for name,obj in originals.items():self.assertIs(getattr(p.science,name),obj)
            self.assertNotEqual(run.__globals__['source_identity'],p.legacy.source_identity)
    def test_own_job_binds_all_transport_roles_resources_and_exact_env(self):
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.request(base);path=base/'request.json';path.write_text(json.dumps(v));sha=p.digest(path)
            run=base/'ops/runs'/v['run_ids']['preflight'];run.mkdir(parents=True)
            pins=[{'path':'role-'+str(i),'sha256':str(i%10)*64} for i in range(13)]
            record={'schema':'vkgr-run-v1','run_id':v['run_ids']['preflight'],'kind':'ops','cwd':str(p.ROOT),'argv':p.argv('preflight',path,sha),'seed':42,
                'git':{'commit':v['source']['commit'],'dirty':False,'dirty_files':[]},'config':{'path':str(path),'sha256':sha},'resources':p.resources(v,'preflight'),'env':p.job_env(v,'preflight'),'inputs':pins}
            (run/'run.json').write_text(json.dumps(record))
            with patch.object(p,'pins',return_value=pins):
                self.assertEqual(p.own_job(v,'preflight',path,sha)['run_id'],v['run_ids']['preflight'])
                for key in ('cwd','argv','seed','git','config','resources','env','inputs'):
                    bad=copy.deepcopy(record)
                    if key=='cwd':bad[key]='/unexpected'
                    elif key=='argv':bad[key][-1]='bad'
                    elif key=='seed':bad[key]=42.
                    elif key=='git':bad[key]['dirty']=0
                    elif key=='config':bad[key]['sha256']='0'*64
                    elif key=='resources':bad[key]['ram_mib']+=1
                    elif key=='env':bad[key]['VKGR_EXTRA']='1'
                    else:bad[key]=bad[key][::-1]
                    (run/'run.json').write_text(json.dumps(bad))
                    with self.subTest(key=key),self.assertRaises(ValueError):p.own_job(v,'preflight',path,sha)
    def test_verify_resources_use_original_fit_rss_rounding_not_unknown(self):
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.request(base);(base/'fit').mkdir();path=base/'fit/stage-receipt.json'
            path.write_text(json.dumps({'rss_tree_upper_bound_mib':1000.}))
            self.assertEqual(p.resources(v,'verify')['ram_mib'],1280)
            for val in (None,False,0,float('nan')):
                path.write_text(json.dumps({'rss_tree_upper_bound_mib':val}))
                with self.assertRaises(ValueError):p.resources(v,'verify')
    def test_real_launcher_deployment_metadata_job_binding_and_dry(self):
        import shutil
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d)
            def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
            def clone(name,ref):
                root=base/name;subprocess.run(['git','clone','-q','--shared','--no-checkout',str(ROOT),str(root)],check=True,capture_output=True)
                git(root,'checkout','-q','--detach',ref);return root
            runner=clone('runner','305c415871bd60dc0d7d8a064a7e893830a9c20f');native_root=clone('native-code','305c415871bd60dc0d7d8a064a7e893830a9c20f');ops=clone('ops-code','5840e6b686ec9651314a9e28a1a2c0befcc05158')
            for name in p.ADDITIONS:shutil.copyfile(ROOT/name,runner/name)
            git(runner,'add',*p.ADDITIONS);git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--allow-empty','-q','-m','native SID transport fixture')
            env={**os.environ,'PYTHONPATH':str(runner/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
            identity=subprocess.run([sys.executable,'-B','-c','from analysis.ur001_derive_cli import source_identity; import json; print(json.dumps(source_identity()))'],cwd=runner,env=env,capture_output=True,text=True)
            self.assertEqual(identity.returncode,0,identity.stderr);source=json.loads(identity.stdout)
            probe=subprocess.run([sys.executable,'-B','-c','from analysis.sgns_gpu_subset import fixed_records; from analysis.new_instance_prepare import runtime; from analysis import sgns_native_subset_sid; print(len(fixed_records()))'],cwd=runner,env=env,capture_output=True,text=True)
            self.assertEqual(probe.returncode,0,probe.stderr);self.assertEqual(probe.stdout.strip(),'26')
            self.assertFalse((runner/'analysis/ur001_prepare.py').exists());self.assertFalse((runner/'analysis/ur001_week26_bounded.py').exists())
            lock=base/'lock';lock.write_text('transport fixture only');bundle=base/'bundle';bundle.mkdir()
            for name in ('manifest.json','catalog.json','content.npy','histories.jsonl'):(bundle/name).write_text('transport fixture only')
            native={'schema':'sgns-gpu-native-fit-request-v1','runner_source':git(native_root,'rev-parse','HEAD'),'bundle':str(bundle),'bundle_sha256':p.digest(bundle/'manifest.json'),
                'lock':str(lock),'ops_root':str(base/'native-ops'),'output':str(base/'native-output'),'preflight_run_id':'sgns-gpu-native-fit-preflight-fixture-v1',
                'fit_run_id':'sgns-gpu-native-fit-fit-fixture-v1','route_selected':True,'benchmark_reference':None,'compiled':True,
                'resources':{'cpu_cores':2,'ram_mib':16384,'vram_mib':8192,'disk_gib':1,'wall_hours':3}}
            npath=base/'native-request.json';npath.write_text(json.dumps(native))
            nout=base/'native-output/fit';nout.mkdir(parents=True)
            for name in ('result.json','vectors.npy','audit.json'):(nout/name).write_text('{}')
            nrun=base/'native-ops/runs'/native['fit_run_id'];(nrun/'logs').mkdir(parents=True)
            for name in ('run.json','terminal.json','logs/stdout.log'):(nrun/name).write_text('{}')
            cpu={'schema':'gpu-sgns-subset-sid-request-v1','device':'cpu','source':{'commit':'a'*40,'files':{}},'lock':str(lock),'ops_root':str(base/'cpu-ops'),'ops_code':str(ops),
                'bundle':str(bundle),'bundle_sha':native['bundle_sha256'],'sgns_output':str(base/'cpu-sgns'),'sgns_receipt_sha':'a'*64,'sgns_run':str(base/'cpu-sgns-run'),
                'preflight_output':str(base/'cpu-sid-preflight'),'fit_output':str(base/'cpu-sid-fit'),'verify_output':str(base/'cpu-sid-verify'),'cpu_request':None,'cpu_request_sha':None}
            cpath=base/'cpu-request.json';cpath.write_text(json.dumps(cpu))
            crun=base/'cpu-ops/runs/sgns-gpu-subset-cpu-sid-verify-v1';(crun/'logs').mkdir(parents=True)
            for name in ('run.json','terminal.json','logs/stdout.log'):(crun/name).write_text('{}')
            for path in (base/'cpu-sid-verify/stage-receipt.json',base/'cpu-sid-fit/run_manifest.json'):path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            value=self.request(base);value.update({'source':source,'native_root':str(native_root),'ops_code':str(ops),'native_request_sha256':p.digest(npath),'cpu_request_sha256':p.digest(cpath),
                'producer_delta':{name:p.digest(runner/name) for name in p.ADDITIONS}})
            path=base/'request.json';path.write_text(json.dumps(value));sha=p.digest(path)
            cmd=[sys.executable,'-B','-m','analysis.sgns_native_subset_sid'];args=cmd+['plan','--stage','preflight','--request',str(path),'--request-sha256',sha,'--ops-code',str(ops)]
            planned=subprocess.run(args,cwd=runner,env=env,capture_output=True,text=True)
            self.assertEqual(planned.returncode,0,(planned.stdout,planned.stderr));prepared=json.loads(planned.stdout)
            self.assertEqual(len(prepared['metadata']['inputs']),22)
            self.assertEqual(prepared['spec']['argv'],cmd+['preflight','--request',str(path),'--request-sha256',sha])
            for extra in ({'VKGR_EXTRA':'1'},{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
                bad=subprocess.run(args,cwd=runner,env={**env,**extra},capture_output=True,text=True)
                self.assertEqual(bad.returncode,2);self.assertIn('unexpected inherited launcher environment',bad.stdout)
            from analysis.a40_ops import load_ops
            _,launcher,_,_=load_ops(ops);run=Path(value['ops_root'])/'runs'/value['run_ids']['preflight'];run.mkdir(parents=True)
            meta=prepared['metadata'];spec=prepared['spec'];record={'schema':'vkgr-run-v1','run_id':spec['run_id'],'kind':spec['kind'],'seed':spec['seed'],
                **{k:meta[k] for k in ('cwd','git','config','inputs')},'argv':spec['argv'],'resources':p.resources(value,'preflight'),
                'env':launcher._recorded_env(meta['env'],launcher._job_env(meta['env'],run,spec['run_id']))}
            (run/'run.json').write_text(json.dumps(record))
            code='from analysis import sgns_native_subset_sid as p; import pathlib,json; v=p.load(pathlib.Path('+repr(str(path))+'),'+repr(sha)+'); print(json.dumps(p.own_job(v,"preflight",pathlib.Path('+repr(str(path))+'),'+repr(sha)+')))'
            bound=subprocess.run([sys.executable,'-B','-c',code],cwd=runner,env=env,capture_output=True,text=True)
            self.assertEqual(bound.returncode,0,bound.stderr);self.assertEqual(json.loads(bound.stdout)['run_manifest_sha256'],p.digest(run/'run.json'))
            dry=subprocess.run(cmd+['dry'],cwd=runner,env=env,capture_output=True,text=True)
            self.assertEqual(dry.returncode,0,dry.stderr);self.assertFalse(json.loads(dry.stdout)['training'])
            self.assertFalse((Path(value['ops_root'])/'queue').exists())
    def test_stage_envelope_binds_runtime_native_cpu_proofs_and_wallcap(self):
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.request(base);path=base/'request.json';path.write_text(json.dumps(v));sha=p.digest(path);stage='preflight'
            n={'ops_root':str(base/'native-ops'),'fit_run_id':'native-fit','output':str(base/'native-output'),'bundle_sha256':'d'*64}
            nr=base/'native-ops/runs/native-fit';(nr/'logs').mkdir(parents=True)
            for name in ('run.json','terminal.json','logs/stdout.log'):(nr/name).write_text('{}')
            no=base/'native-output/fit';no.mkdir(parents=True);ns={'commit':'b'*40,'files':{'analysis/sgns_gpu_native_fit.py':'c'*64}}
            (no/'result.json').write_text(json.dumps({'runtime':{'fixed':True},'source':ns}));audit={key:str(i+1)*64 for i,key in enumerate(('catalog_sha256','vectors_sha256','content_sha256','train_sequence_sha256'))};(no/'audit.json').write_text(json.dumps(audit))
            npf={'run_id':'native-fit','receipt_sha256':p.digest(no/'result.json'),'stdout_sha256':p.digest(nr/'logs/stdout.log'),'run_manifest_sha256':p.digest(nr/'run.json'),'terminal_sha256':p.digest(nr/'terminal.json')}
            cpu={'verify_output':str(base/'cpu-verify'),'sgns_run':str(base/'cpu-sgns-run')};cv=base/'cpu-verify';cv.mkdir();(cv/'stage-receipt.json').write_text('{}')
            cr=base/'cpu-sgns-run';(cr/'logs').mkdir(parents=True)
            for name in ('run.json','terminal.json','logs/stdout.log'):(cr/name).write_text('{}')
            cpf={'request_sha256':v['cpu_request_sha256'],'verification_receipt_sha256':p.digest(cv/'stage-receipt.json'),'manifest_sha256':'a'*64,'original_verify_proof':{'fixed':True},'cpu_sgns_proof':{name:p.digest(cr/name) for name in ('run.json','terminal.json','logs/stdout.log')}}
            im={'schema':'coll-input-v1','status':'accepted','variant':'C1','dimension':64,'engineering':False,'catalog_sha256':audit['catalog_sha256'],'vectors_sha256':audit['vectors_sha256'],
                'content_feature_sha256':audit['content_sha256'],'train_sequence_sha256':audit['train_sequence_sha256'],'script_sha256':'c'*64,'commit':'b'*40,'checkpoint_sha256':None,'source_files':ns['files'],
                'sgns_stage_receipt_sha256':npf['receipt_sha256'],'sgns_device':'cuda','sgns_original_job':npf,'input_manifest_file_sha256':'d'*64,'sgns_backend':'B'}
            own={'run_id':v['run_ids'][stage],'run_manifest_sha256':'a'*64}
            receipt={'schema':'gpu-sgns-subset-sid-stage-v1','status':'verified','verified':True,'engineering':False,'stage':'sid-preflight','device':'cuda','source':v['source'],'request_sha256':sha,**own,
                'runtime':{'fixed':True},'sgns_stage_receipt_sha256':npf['receipt_sha256'],'input_manifest_sha256':p.hashlib.sha256(p.science.canonical_bytes(im)).hexdigest(),
                'subset_four_gate_acceptance':False,'test_read':False,'validation_read':False,'payload_written':False,'mapping_replay':False,'manifest_sha256':None,'gates':None,'collision_gate':None}
            envelope={'schema':'sgns-native-subset-sid-stage-v1','status':'verified','verified':True,'engineering':False,'backend':'B','authority':'4b0a8a1','stage':stage,'source':v['source'],'request_sha256':sha,**own,
                'science_receipt':receipt,'subset_four_gate_acceptance':False,'test_read':False,'validation_read':False,'native_request_sha256':v['native_request_sha256'],'native_producer':npf,'cpu_reference':cpf}
            out=base/stage;out.mkdir();run=base/'ops/runs'/v['run_ids'][stage];(run/'logs').mkdir(parents=True);(run/'run.json').write_text('{}')
            terminal={'schema':'vkgr-terminal-v1','run_id':v['run_ids'][stage],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}
            def write(e,inner):
                raw=json.dumps(inner).encode();(out/'stage-receipt.json').write_bytes(raw);e=copy.deepcopy(e);e['science_receipt']=inner;e['science_receipt_sha256']=p.hashlib.sha256(raw).hexdigest()
                (out/'native-receipt.json').write_text(json.dumps(e));(run/'logs/stdout.log').write_text(json.dumps(e));(run/'terminal.json').write_text(json.dumps(terminal))
            write(envelope,receipt)
            with patch.object(p,'own_job',return_value=own),patch.object(p,'native_context',return_value=(None,n)),patch.object(p.legacy,'load_request',return_value=cpu),patch.object(p.legacy,'successful_stage',return_value=({'manifest_sha256':'a'*64},{'fixed':True})):
                self.assertEqual(p.successful_stage(v,stage,path,sha)[0],receipt)
                for key,val in [('backend','A'),('native_producer',{}),('cpu_reference',{}),('native_request_sha256','e'*64),('verified',1)]:
                    write({**envelope,key:val},receipt)
                    with self.subTest(outer=key),self.assertRaises(ValueError):p.successful_stage(v,stage,path,sha)
                for key,val in [('runtime',{}),('input_manifest_sha256','e'*64),('sgns_stage_receipt_sha256','e'*64),('payload_written',True),('test_read',True)]:
                    write(envelope,{**receipt,key:val})
                    with self.subTest(inner=key),self.assertRaises(ValueError):p.successful_stage(v,stage,path,sha)
                for bad in (True,0,None):
                    if bad is None:terminal.pop('wall_cap_exceeded',None)
                    else:terminal['wall_cap_exceeded']=bad
                    write(envelope,receipt)
                    with self.subTest(wallcap=bad),self.assertRaises(ValueError):p.successful_stage(v,stage,path,sha)
    def test_cpu_reference_reads_original_verify_and_bundle_without_refit(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        p=self.p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);v=self.request(base);v['cpu_root']=str(ROOT);env={'fixed':True}
            cpu={'device':'cpu','source':{'commit':'a'*40,'files':{}},'lock':v['lock'],'bundle':str(base/'bundle'),'bundle_sha':'b'*64,
                'ops_root':str(base/'cpu-ops'),'fit_output':str(base/'cpu-fit'),'verify_output':str(base/'cpu-verify'),'sgns_output':str(base/'cpu-sgns'),
                'sgns_receipt_sha':'c'*64,'sgns_run':str(base/'cpu-sgns-run')}
            run=base/'cpu-ops/runs/sgns-gpu-subset-cpu-sid-verify-v1';run.mkdir(parents=True)
            terminal={'schema':'vkgr-terminal-v1','run_id':run.name,'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}
            (run/'terminal.json').write_text(json.dumps(terminal))
            predecessors=[(base/'cpu-sgns-run','sgns-gpu-subset-cpu-fit-v1'),(base/'cpu-ops/runs/sgns-gpu-subset-cpu-sid-fit-v1','sgns-gpu-subset-cpu-sid-fit-v1')]
            for parent,rid in predecessors:
                parent.mkdir(parents=True);(parent/'terminal.json').write_text(json.dumps({**terminal,'run_id':rid}))
            out=base/'cpu-verify';out.mkdir();(out/'stage-receipt.json').write_text('{}')
            inputs={'ids':[1,2,3],'vectors':[], 'input_manifest':{},'job_proof':{'original':'cpu'}};gates={'passed':True}
            receipt={'runtime':env,'mapping_replay':True,'manifest_sha256':'d'*64,'gates':gates}
            old=SimpleNamespace(load_sgns_stage=Mock(return_value=inputs),load_bundle=Mock(return_value=SimpleNamespace(manifest={'gates':gates,'audit':{'items':120415,'semantic_collision_items':2}})),
                fit_bhkm_coll=Mock(side_effect=AssertionError('CPU refit forbidden')))
            with patch.object(p,'source'),patch.object(p,'native_context',return_value=(None,{'bundle':cpu['bundle'],'bundle_sha256':cpu['bundle_sha']})), \
                 patch.object(p,'module_at',return_value=old),patch.object(p.legacy,'load_request',return_value=cpu),patch.object(p.legacy,'successful_stage',return_value=(receipt,{'verified':'old'})), \
                 patch.object(p.subprocess,'check_output',return_value=''):
                got=p.cpu_reference(v,env);self.assertEqual(got[0]['semantic_collision_items'],2);old.fit_bhkm_coll.assert_not_called()
                old.load_sgns_stage.assert_called_once();old.load_bundle.assert_called_once()
                for role in ('bundle','sgns_output','sgns_run','fit_output','verify_output','ops_root'):
                    original=v['outputs']['fit'];v['outputs']['fit']=cpu[role]+'/new-child'
                    with self.subTest(overlap=role),self.assertRaises(ValueError):p.cpu_reference(v,env)
                    v['outputs']['fit']=original
                for key,val in (('runtime',{}),('mapping_replay',False),('gates',{'passed':False})):
                    before=receipt[key];receipt[key]=val
                    with self.subTest(cpu_field=key),self.assertRaises(ValueError):p.cpu_reference(v,env)
                    receipt[key]=before
                for bad in (True,0,None):
                    wrong=terminal.copy()
                    if bad is None:wrong.pop('wall_cap_exceeded')
                    else:wrong['wall_cap_exceeded']=bad
                    (run/'terminal.json').write_text(json.dumps(wrong))
                    with self.subTest(wallcap=bad),self.assertRaises(ValueError):p.cpu_reference(v,env)

                (run/'terminal.json').write_text(json.dumps(terminal))
                for parent,rid in predecessors:
                    good={**terminal,'run_id':rid}
                    for key,bad in [('wall_cap_exceeded',True),('wall_cap_exceeded',0),('wall_cap_exceeded',None),('signal',9),('exit_code',False),('run_id','wrong'),('schema','wrong')]:
                        wrong=good.copy()
                        if key=='wall_cap_exceeded' and bad is None:wrong.pop(key)
                        else:wrong[key]=bad
                        (parent/'terminal.json').write_text(json.dumps(wrong))
                        with self.subTest(predecessor=rid,key=key,bad=bad),self.assertRaises(ValueError):p.cpu_reference(v,env)
                    (parent/'terminal.json').write_text(json.dumps(good))
