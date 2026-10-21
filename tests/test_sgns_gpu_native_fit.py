import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class ProducerTests(unittest.TestCase):
    def setUp(self):
        from analysis import sgns_gpu_native_fit as p
        self.p=p
    def test_formal_cli_dry_and_illegal(self):
        for command,code in [('dry',0),('illegal',2)]:
            run=subprocess.run([sys.executable,'-B','-m','analysis.sgns_gpu_native_fit',command],cwd=ROOT,
                env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
            self.assertEqual(run.returncode,code,run.stderr)
            if code==0:self.assertFalse(json.loads(run.stdout)['training'])
    def test_route_and_benchmark_gate_fail_closed(self):
        p=self.p
        with patch.object(p,'benchmark_proof') as proof:
            for selected in (False,1,None,'true'):
                with self.assertRaises(ValueError):p.fit_authority({'route_selected':selected})
            proof.assert_not_called()
        with patch.object(p,'benchmark_proof',side_effect=ValueError('missing verified benchmark')):
            with self.assertRaises(ValueError):p.fit_authority({'route_selected':True})
    def test_scoped_epoch_observer_preserves_arguments_result_and_restores_on_failure(self):
        from vkgr import sgns_gpu_native as native
        original=native.NativeSGNS.epoch
        observed=[]
        def epoch(instance,**kw):return {'epoch':1,'complete_epoch':True,'positive_pairs':7,'steps':1,'processed_positive_pairs':7,'summed_loss':3.}
        with patch.object(native.NativeSGNS,'epoch',epoch):
            with self.p.epoch_progress(observed.append):
                result=native.NativeSGNS.epoch(object(),observe=False)
                self.assertEqual(result['positive_pairs'],7)
            self.assertIs(native.NativeSGNS.epoch,epoch)
            with self.assertRaises(RuntimeError):
                with self.p.epoch_progress(observed.append):raise RuntimeError('fit failed')
            self.assertIs(native.NativeSGNS.epoch,epoch)
        self.assertIs(native.NativeSGNS.epoch,original)
        self.assertEqual(len(observed),1);self.assertEqual(observed[0]['epoch'],1)
    def test_request_closed_schema_types_paths_and_distinct_roles(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            value=self.request(Path(d));self.assertEqual(self.p.validate(value),value)
            for key,val in [('route_selected',1),('compiled',1),('fit_run_id',value['preflight_run_id']),('bundle','/workspace/test/bundle'),('output',value['bundle']+'/child'),('unknown',1),('resources',{})]:
                with self.subTest(key=key),self.assertRaises((ValueError,TypeError)):self.p.validate({**value,key:val})
    def request(self,base):
        return {'schema':'sgns-gpu-native-fit-request-v1','runner_source':'a'*40,'bundle':str(base/'bundle'),'bundle_sha256':'b'*64,
            'lock':str(base/'lock'),'ops_root':str(base/'ops'),'output':str(base/'output'),
            'preflight_run_id':'sgns-gpu-native-fit-preflight-fixture-v1','fit_run_id':'sgns-gpu-native-fit-fit-fixture-v1',
            'route_selected':False,'benchmark_reference':None,'compiled':True,
            'resources':{'cpu_cores':2,'ram_mib':16384,'vram_mib':8192,'disk_gib':1,'wall_hours':12}}
    def test_cpu_real_fit_original_health_and_audit_mutations(self):
        p=self.p
        from vkgr import sgns_gpu_native as native
        catalog=list(range(1,14));histories={i:[1+i*3+j%3 for j in range(18)] for i in range(4)};histories[4]=[13,13]
        content=np.random.default_rng(3).normal(size=(13,64)).astype('f4')
        observed=[]
        with p.epoch_progress(observed.append):vectors,audit=native.fit_sgns(catalog,histories,content,device='cpu',compile_step=False)
        proof=p.verify_audit((catalog,histories,content),vectors,audit,device='cpu',compiled=False)
        self.assertTrue(proof['verified']);self.assertEqual(len(observed),5);self.assertEqual(audit['fallback_count'],1)
        changes={'engineering':True,'device':'cuda','epochs_completed':4,'steps':0,'processed_positive_pairs':0,
                 'config':{},'health':{},'vectors_sha256':'0'*64,'content_sha256':'0'*64,'train_sequence_sha256':'0'*64,
                 'fallback_count':99,'input_initialization_sha256':'0'*64,'training_contract':{},'runtime':{}}
        for key,val in changes.items():
            bad=copy.deepcopy(audit);bad[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):p.verify_audit((catalog,histories,content),vectors,bad,device='cpu',compiled=False)
        for key,val in [('positive_pairs',0),('steps',0),('complete_epoch',False),('processed_positive_pairs',0),('negative_observed_bins',[0]*64)]:
            bad=copy.deepcopy(audit);bad['epoch_audits'][0][key]=val
            with self.subTest(epoch_key=key),self.assertRaises(ValueError):p.verify_audit((catalog,histories,content),vectors,bad,device='cpu',compiled=False)
        broken=vectors.copy();broken[0]*=2
        with self.assertRaises(ValueError):p.verify_audit((catalog,histories,content),broken,audit,device='cpu',compiled=False)

class BindingTests(unittest.TestCase):
    request=ProducerTests.request
    def setUp(self):
        from analysis import sgns_gpu_native_fit as p
        self.p=p
        self.temp=tempfile.TemporaryDirectory(dir='/private/tmp');self.base=Path(self.temp.name);self.value=self.request(self.base)
        bundle=self.base/'bundle';bundle.mkdir()
        for name in ('manifest.json','catalog.json','content.npy','histories.jsonl'):(bundle/name).write_text('fixture')
        (self.base/'lock').write_text('fixture')
        self.path=self.base/'request.json';self.path.write_text(json.dumps(self.value));self.sha=p.digest(self.path)
        self.run=self.base/'ops/runs'/self.value['preflight_run_id'];(self.run/'logs').mkdir(parents=True)
        self.source={'source':'fixed'};self.bound={'runtime':{'fixed':True}}
        self.record={'schema':'vkgr-run-v1','run_id':self.value['preflight_run_id'],'kind':'ops','cwd':str(p.ROOT),'argv':p.argv('preflight',self.path,self.sha),'seed':42,
            'git':{'commit':self.value['runner_source'],'dirty':False,'dirty_files':[]},'config':{'path':str(self.path),'sha256':self.sha},
            'resources':p.resources(self.value),'env':p.job_env(self.value,'preflight'),'inputs':p.pins(self.value,self.path,'preflight')}
        (self.run/'run.json').write_text(json.dumps(self.record))
        self.receipt={'schema':'gpu-sgns-subset-stage-v1','status':'verified','verified':True,'engineering':False,'authority':'4b0a8a1','stage':'preflight','backend':'B','device':'cuda',
            'source':self.source,'runtime':self.bound['runtime'],'input_binding':self.bound,'input_manifest_sha256':self.value['bundle_sha256'],
            'request_sha256':self.sha,'run_id':self.value['preflight_run_id'],'run_manifest_sha256':p.digest(self.run/'run.json'),'config':p.CONFIG,
            'subset_four_gate_acceptance':False,'validation_read':False,'test_read':False,'real_readonly_preflight':True,'temporary_parameter_updates':False,'vectors_saved':False,'native_epoch_positive_pairs':34770216}
        self.terminal={'schema':'vkgr-terminal-v1','run_id':self.value['preflight_run_id'],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}
        self.write(self.receipt)
    def tearDown(self):self.temp.cleanup()
    def write(self,receipt):
        out=self.base/'output/preflight';out.mkdir(parents=True,exist_ok=True)
        (out/'result.json').write_text(json.dumps(receipt));(self.run/'terminal.json').write_text(json.dumps(self.terminal));(self.run/'logs/stdout.log').write_text(json.dumps(receipt))
    def test_exact_own_job_negative_roles_and_nested_types(self):
        p=self.p;self.assertEqual(p.job(self.value,self.path,self.sha,'preflight'),p.digest(self.run/'run.json'))
        for key in ('cwd','argv','seed','git','config','resources','env','inputs'):
            bad=copy.deepcopy(self.record)
            if key=='seed':bad[key]=42.
            elif key=='git':bad[key]['dirty']=0
            elif key=='config':bad[key]['sha256']='0'*64
            elif key=='resources':bad[key]['ram_mib']+=1
            elif key=='env':bad[key]['VKGR_EXTRA']='1'
            elif key=='inputs':bad[key]=bad[key][::-1]
            elif key=='argv':bad[key][-1]='--invalid'
            else:bad[key]='/unexpected'
            (self.run/'run.json').write_text(json.dumps(bad))
            with self.subTest(key=key),self.assertRaises(ValueError):p.job(self.value,self.path,self.sha,'preflight')
        for index in range(len(self.record['inputs'])):
            bad=copy.deepcopy(self.record);bad['inputs'][index]['sha256']='0'*64
            (self.run/'run.json').write_text(json.dumps(bad))
            with self.subTest(pin=index),self.assertRaises(ValueError):p.job(self.value,self.path,self.sha,'preflight')
    def test_terminal_stdout_receipt_config_and_pf_safety(self):
        p=self.p
        self.assertEqual(p.accepted(self.value,self.path,self.sha,'preflight',self.source,self.bound)['run_id'],self.value['preflight_run_id'])
        changes={'source':{},'runtime':{},'input_binding':{},'request_sha256':'0'*64,'config':{},'engineering':True,'verified':1,'backend':'A',
            'authority':'ad1c428','real_readonly_preflight':False,'vectors_saved':True,'temporary_parameter_updates':True,'native_epoch_positive_pairs':0,
            'test_read':True,'validation_read':True,'subset_four_gate_acceptance':True,'run_manifest_sha256':'0'*64}
        for key,val in changes.items():
            self.write({**self.receipt,key:val})
            with self.subTest(key=key),self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.source,self.bound)
        self.write(self.receipt)
        for key,val in [('status','failed'),('exit_code',False),('signal',9),('run_id','another')]:
            old=self.terminal.copy();self.terminal[key]=val;self.write(self.receipt)
            with self.subTest(terminal=key),self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.source,self.bound)
            self.terminal=old
        for val in (True,0,None):
            old=self.terminal.copy()
            if val is None:self.terminal.pop('wall_cap_exceeded',None)
            else:self.terminal['wall_cap_exceeded']=val
            self.write(self.receipt)
            with self.subTest(wall_cap=val),self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.source,self.bound)
            self.terminal=old
        self.write(self.receipt);(self.run/'logs/stdout.log').write_text('{}')
        with self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.source,self.bound)
    def test_fit_denied_before_arrays_or_core_and_pf_never_fits(self):
        p=self.p
        with patch.object(p,'load',return_value=self.value),patch.object(p.b,'inputs') as arrays,patch.object(p.native,'fit_sgns') as fit:
            with self.assertRaises(ValueError):p.execute('fit',self.path,self.sha)
            arrays.assert_not_called();fit.assert_not_called()
        with patch.object(p,'load',return_value=self.value),patch.object(p,'ledger',return_value=self.source),patch.object(p,'job',return_value='a'*64),patch.object(p,'pins',return_value=[]), \
             patch.object(p.b,'inputs',return_value=(([],{},[]),self.bound)),patch.dict(os.environ,p.job_env(self.value,'preflight'),clear=True), \
             patch('analysis.ur001_derive_cli.guards'),patch.object(p.native,'fit_sgns') as fit,patch.object(p.native,'NativeSGNS') as engine:
            engine.return_value.pairs_per_epoch=34770216
            # Fresh stage path for the PF writer.
            value={**self.value,'output':str(self.base/'fresh-output')}
            with patch.object(p,'load',return_value=value):result=p.execute('preflight',self.path,self.sha)
            self.assertFalse(result['vectors_saved']);fit.assert_not_called()
    def test_benchmark_proof_projection_subprocess_and_closure(self):
        p=self.p;old={'measured_batches':256,'benchmark_run_id':'sgns-gpu-native-b-benchmark-fixture-v1','ops_root':str(self.base/'oldops'),'output':str(self.base/'oldout')}
        run=Path(old['ops_root'])/'runs'/old['benchmark_run_id'];(run/'logs').mkdir(parents=True);(run/'logs/stdout.log').write_text('fixture')
        out=Path(old['output'])/'benchmark';out.mkdir(parents=True)
        seconds=.01;rate=256*4096/seconds
        receipt={'measured_seconds':seconds,'measured_pairs_per_second':rate,'full_fit_linear_projection_hours':p.b.FULL_PAIRS/rate/3600,
            'B_throughput_12h_diagnostic':True,'full_pair_count':p.b.FULL_PAIRS,'negative_observed_bins':[81920]*64,'negative_expected_bins':[81920.]*64}
        good_receipt=copy.deepcopy(receipt)
        (out/'result.json').write_text(json.dumps(receipt))
        proof={'status':'verified','readonly':True,'science_training':False,'producer':{'run_id':old['benchmark_run_id'],'stdout_sha256':p.digest(run/'logs/stdout.log'),'result_sha256':p.digest(out/'result.json')}}
        ref={'root':str(ROOT),'request':str(self.path),'request_sha256':self.sha};value={**self.value,'benchmark_reference':ref,'route_selected':True}
        with patch.object(p,'benchmark_files',return_value=(old,[])),patch.object(p.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(proof),'')) as call:
            checked=p.fit_authority(value);self.assertTrue(checked['verified']);self.assertEqual(call.call_args.kwargs['cwd'],ROOT)
            self.assertEqual(call.call_args.args[0][3:6],['analysis.sgns_gpu_native_b','verify','--stage'])
            receipt['measured_seconds']=10000.;receipt['measured_pairs_per_second']=256*4096/10000.;receipt['full_fit_linear_projection_hours']=p.b.FULL_PAIRS/receipt['measured_pairs_per_second']/3600;receipt['B_throughput_12h_diagnostic']=False
            (out/'result.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):p.fit_authority(value)
        (out/'result.json').write_text(json.dumps(good_receipt))
        for code,stdout in [(2,json.dumps(proof)),(0,'{}'),(0,json.dumps({**proof,'readonly':1}))]:
            with patch.object(p,'benchmark_files',return_value=(old,[])),patch.object(p.subprocess,'run',return_value=subprocess.CompletedProcess([],code,stdout,'')):
                with self.assertRaises(ValueError):p.fit_authority(value)
    def test_real_launcher_layout_metadata_job_dry_chain(self):
        import shutil
        p=self.p;base=self.base/'launcher';base.mkdir()
        def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE,text=True).strip()
        def clone(name,ref):
            root=base/name;subprocess.run(['git','clone','-q','--shared','--no-checkout',str(ROOT),str(root)],check=True,capture_output=True)
            git(root,'checkout','-q','--detach',ref);return root
        runner=clone('runner',git(ROOT,'rev-parse','HEAD'));ops=clone('ops-code','5840e6b686ec9651314a9e28a1a2c0befcc05158')
        shutil.copyfile(ROOT/'analysis/sgns_gpu_native_fit.py',runner/'analysis/sgns_gpu_native_fit.py');git(runner,'add','analysis/sgns_gpu_native_fit.py')
        git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--allow-empty','-q','-m','fit transport fixture')
        value={**self.value,'runner_source':git(runner,'rev-parse','HEAD'),'ops_root':str(base/'ops'),'output':str(base/'output')}
        path=base/'request.json';path.write_text(json.dumps(value));sha=p.digest(path)
        env={**os.environ,'PYTHONPATH':str(runner/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
        cmd=[sys.executable,'-B','-m','analysis.sgns_gpu_native_fit']
        args=cmd+['plan','--stage','preflight','--request',str(path),'--request-sha256',sha,'--ops-code',str(ops)]
        result=subprocess.run(args,cwd=runner,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,(result.stdout,result.stderr));prepared=json.loads(result.stdout)
        self.assertEqual(prepared['spec']['argv'],cmd+['preflight','--request',str(path),'--request-sha256',sha,'--execute-on-new-instance'])
        self.assertEqual(len(prepared['metadata']['inputs']),10)
        for extra in ({'VKGR_EXTRA':'1'},{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
            denied=subprocess.run(args,cwd=runner,env={**env,**extra},capture_output=True,text=True)
            self.assertEqual(denied.returncode,2);self.assertIn('unexpected inherited launcher environment',denied.stdout)
        from analysis.a40_ops import load_ops
        _,launcher,_,_=load_ops(ops);run=Path(value['ops_root'])/'runs'/value['preflight_run_id'];run.mkdir(parents=True)
        meta=prepared['metadata'];spec=prepared['spec'];record={'schema':'vkgr-run-v1','run_id':spec['run_id'],'kind':spec['kind'],'seed':spec['seed'],
            **{k:meta[k] for k in ('cwd','git','config','inputs')},'argv':spec['argv'],'resources':p.resources(value),
            'env':launcher._recorded_env(meta['env'],launcher._job_env(meta['env'],run,spec['run_id']))}
        (run/'run.json').write_text(json.dumps(record))
        # Bind through the runner's actual module rather than patching its ROOT.
        code='from analysis import sgns_gpu_native_fit as p; import pathlib; v=p.load(pathlib.Path('+repr(str(path))+'),'+repr(sha)+'); print(p.job(v,pathlib.Path('+repr(str(path))+'),'+repr(sha)+',"preflight"))'
        checked=subprocess.run([sys.executable,'-B','-c',code],cwd=runner,env=env,capture_output=True,text=True)
        self.assertEqual(checked.returncode,0,checked.stderr);self.assertEqual(checked.stdout.strip(),p.digest(run/'run.json'))
        dry=subprocess.run(cmd+['dry'],cwd=runner,env=env,capture_output=True,text=True)
        self.assertEqual(dry.returncode,0,dry.stderr);self.assertEqual(json.loads(dry.stdout)['status'],'dry')
        self.assertFalse((Path(value['ops_root'])/'queue').exists())
    def test_consumer_same_shape_and_original_formal_sid_input_contract(self):
        p=self.p
        from vkgr import sgns_subset_sid as sid
        vectors=np.full((120415,64),.125,dtype='f4');catalog=list(range(1,120416));histories={1:[1,2,3]};arrays=(catalog,histories,vectors)
        source={'commit':'a'*40,'files':{'analysis/sgns_gpu_native_fit.py':'b'*64}}
        out=self.base/'output/fit';out.mkdir();np.save(out/'vectors.npy',vectors,allow_pickle=False);(out/'audit.json').write_text('{}')
        receipt={'artifacts':{name:p.digest(out/name) for name in ('vectors.npy','audit.json')},'native_audit_verification':{'verified':True}}
        (out/'result.json').write_text(json.dumps(receipt));proof={'receipt_sha256':p.digest(out/'result.json')}
        with patch.object(p,'load',return_value=self.value),patch.object(p,'ledger',return_value=source),patch.object(p.b,'inputs',return_value=(arrays,self.bound)), \
             patch.object(p,'accepted',return_value=proof),patch.object(p,'verify_audit',return_value={'verified':True}), \
             patch.object(sid,'CONTENT_V1_FEATURE_SHA256',p.feature_digest(vectors)),patch.object(sid,'validate_input',wraps=sid.validate_input) as original:
            got=p.consume(self.path,self.sha)
            self.assertEqual(set(got),{'ids','vectors','input_manifest','sgns_receipt','source_delta','job_proof'})
            self.assertEqual(got['ids'].dtype,np.uint32);self.assertEqual(got['input_manifest']['sgns_backend'],'B')
            self.assertIs(original.call_args.kwargs['engineering'],False)
            self.assertEqual(got['vectors'].shape,(120415,64))
            # Corrupt persisted bytes without changing the bound artifact SHA.
            with (out/'audit.json').open('ab') as stream:stream.write(b' ')
            with self.assertRaises(ValueError):p.consume(self.path,self.sha)

class BenchmarkTerminalTests(unittest.TestCase):
    def test_original_benchmark_terminals_require_explicit_false_wallcap(self):
        from analysis import sgns_gpu_native_fit as p
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);old={'schema':'sgns-gpu-native-b-request-v1','runner_source':'a'*40,'bundle':str(base/'bundle'),'bundle_sha256':'b'*64,
                'lock':str(base/'lock'),'ops_root':str(base/'oldops'),'output':str(base/'oldoutput'),
                'preflight_run_id':'sgns-gpu-native-b-preflight-fixture-v1','benchmark_run_id':'sgns-gpu-native-b-benchmark-fixture-v1',
                'warmup_batches':8,'measured_batches':256,'compiled':True}
            request=base/'benchmark-request.json';request.write_text(json.dumps(old));value=ProducerTests.request(self,base)
            value['benchmark_reference']={'root':str(ROOT),'request':str(request),'request_sha256':p.digest(request)}
            terminals={}
            for stage in ('preflight','benchmark'):
                run=base/'oldops/runs'/old[stage+'_run_id'];(run/'logs').mkdir(parents=True)
                terminal={'schema':'vkgr-terminal-v1','run_id':old[stage+'_run_id'],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}
                for name in ('run.json','logs/stdout.log'):(run/name).write_text('{}')
                path=run/'terminal.json';path.write_text(json.dumps(terminal));terminals[stage]=(path,terminal)
                output=base/'oldoutput'/stage;output.mkdir(parents=True);(output/'result.json').write_text('{}')
            with patch.object(p,'source'):
                self.assertEqual(len(p.benchmark_files(value)[1]),9)
                for stage,(path,terminal) in terminals.items():
                    for bad in (True,0,None):
                        wrong=terminal.copy()
                        if bad is None:wrong.pop('wall_cap_exceeded')
                        else:wrong['wall_cap_exceeded']=bad
                        path.write_text(json.dumps(wrong))
                        with self.subTest(stage=stage,wall_cap=bad),self.assertRaises(ValueError):p.benchmark_files(value)
                    path.write_text(json.dumps(terminal))
