"""Offline transport tests: synthetic only; never reads real full data."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class FullTransportTests(unittest.TestCase):
    def setUp(self):
        from analysis import sgns_native_full as p
        self.p=p
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name).resolve()
    def request(self):
        d=self.base
        return {'schema':'sgns-native-full-request-v1','runner_source':'a'*40,'lock':str(d/'lock'),
            'input_producer':{'root':str(d/'old-code'),'source':self.p.INPUT_SOURCE,'bundle':str(d/'bundle'),'manifest_sha256':'b'*64,'prepare_run':str(d/'old-ops/runs/prepare')},
            'benchmark_reference':{'root':str(d/'benchmark-code'),'request':str(d/'benchmark-request'),'request_sha256':'c'*64},
            'acceptance':{'root':str(d/'a1-code'),'receipt':str(d/'a1-output/accept/stage-receipt.json'),'receipt_sha256':'e'*64,'scratch_root':str(d/'a1-replays')},'source_delta':{k:'d'*64 for k in self.p.OWN_PATHS},'ops_root':str(d/'ops'),'output':str(d/'outputs'),
            'preflight_run_id':'sgns-native-full-preflight-fixture-v1','fit_run_id':'sgns-native-full-fit-fixture-v1','compiled':True,
            'resources':{'cpu_cores':2,'ram_mib':196608,'vram_mib':32768,'disk_gib':16,'wall_hours':3}}
    def test_closed_request_and_nonoverlap(self):
        value=self.request();self.assertEqual(self.p.validate(value),value)
        for key,val in [('compiled',1),('acceptance',{}),('resources',{**value['resources'],'wall_hours':True}),('output',value['input_producer']['bundle']+'/child'),('unexpected',1),('acceptance',{**value['acceptance'],'scratch_root':value['acceptance']['root']+'/replay'}),('acceptance',{**value['acceptance'],'scratch_root':value['output']+'/replay'})]:
            with self.subTest(key=key),self.assertRaises((ValueError,TypeError)):self.p.validate({**value,key:val})
    def test_invalid_acceptance_never_reaches_full_inputs(self):
        for gate in (None,{'passed':True},{'schema':'gpu-sgns-subset-four-gate-receipt-v1','verified':True}):
            value=self.request();value['acceptance']=gate
            with self.subTest(gate=gate),self.assertRaises((ValueError,TypeError)):self.p.require_acceptance(value)
        value=self.request();value['acceptance']=None
        path=self.base/'request.json';path.write_text(json.dumps(value));sha=hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(self.p,'input_proof',side_effect=AssertionError('must not read full inputs')):
            with self.assertRaisesRegex(ValueError,'four-gate'):self.p.execute('preflight',path,sha)
            with self.assertRaisesRegex(ValueError,'four-gate'):self.p.execute('fit',path,sha)
    def test_fixed_external_acceptance_source_and_proof_protocol(self):
        p=self.p
        self.assertEqual(p.ACCEPTANCE_SOURCE,'96bfb9e6fcb4081168cf4d39563f50c58ee3fbad')
        self.assertEqual(p.ACCEPTANCE_ENTRY_SHA,'96c1e851e84d820607ef29cd7708da9fe5bab2aeff8492d86e9aa9cd9a3deb01')
        for bad in (None,{}, {'root':'relative','receipt':'/tmp/receipt','receipt_sha256':'a'*64,'scratch_root':'/tmp/scratch'}):
            with self.assertRaises((ValueError,TypeError)):p.acceptance_reference(bad)

    def gate_fixture(self):
        p=self.p;value=self.request();ref=value['acceptance'];root=Path(ref['root']);root.mkdir()
        evidence=self.base/'gate-evidence';evidence.write_bytes(b'synthetic original evidence')
        source={'commit':p.ACCEPTANCE_SOURCE,'files':{p.ACCEPTANCE_ENTRY:p.ACCEPTANCE_ENTRY_SHA}}
        proof={'health':{'passed':True},'sid_gates':{'cpu':{'passed':True},'cuda':{'passed':True}},'collision':{'cpu_count':10,'gpu_count':11,'items':100,'passed':True,'comparison':'gpu_count * 10 <= cpu_count * 11'},'validation_hits':378,'hits_gate':378,'a1_native':{'status':'verified'},'evidence':{str(evidence):p.digest(evidence)}}
        runtime={'runtime':{'python':'synthetic-python'},'lock_sha256':'a'*64}
        request=self.base/'acceptance-request.json';request.write_text(json.dumps({'ops_root':str(self.base/'a1-ops')}))
        run=self.base/'a1-ops/runs/sgns-native-subset-a1-accept-v1';(run/'logs').mkdir(parents=True)
        for name in ('run.json','terminal.json','logs/stdout.log'):(run/name).write_bytes(b'synthetic job bytes')
        receipt={'schema':'sgns-native-subset-four-gate-receipt-v1','authority':'4b0a8a1','backend':'B','status':'verified','verified':True,'subset_four_gate_acceptance':True,'engineering':False,'test_read':False,'training':False,'run_id':'sgns-native-subset-a1-accept-v1','source':source,'runtime':runtime,'request':{'path':str(request),'sha256':p.digest(request)},'proof':proof}
        path=Path(ref['receipt']);path.parent.mkdir(parents=True);path.write_text(json.dumps(receipt));ref['receipt_sha256']=p.digest(path)
        got={'receipt_sha256':ref['receipt_sha256'],'request_sha256':receipt['request']['sha256'],'producer':{name:p.digest(run/name) for name in ('run.json','terminal.json','logs/stdout.log')},'proof':proof}
        return value,source,receipt,runtime,got

    def test_consumer_subprocess_boundary_result_and_runtime(self):
        # Trust-boundary unit fixture only: producer science is NOT fabricated
        # as a real acceptance. The actual fixed consumer rejects dummy bytes
        # in the separate real CLI/launcher test.
        p=self.p;value,source,receipt,runtime,got=self.gate_fixture();calls=[]
        def run(args,**kwargs):
            calls.append((args,kwargs))
            self.assertEqual(kwargs['cwd'],Path(value['acceptance']['root']))
            self.assertEqual(args[:3],[sys.executable,'-B','-c'])
            self.assertIn('module.consume(path,pin,scratch)',args[3])
            self.assertEqual(kwargs['env']['PYTHONPATH'],str(Path(value['acceptance']['root'])/'src'))
            return subprocess.CompletedProcess(args,0,json.dumps(got),'')
        with patch.object(p,'pinned_tree',return_value=source),patch.object(p.original,'runtime',return_value=runtime),patch.object(p.subprocess,'run',side_effect=run):
            bound=p.require_acceptance(value,'unit')
            self.assertEqual(bound['consumer'],got);self.assertEqual(bound['runtime'],runtime)
            class VersionString(str):pass
            with patch.object(p.original,'runtime',return_value={**runtime,'runtime':{'python':VersionString('synthetic-python')}}):
                self.assertEqual(p.require_acceptance(value,'version-subclass')['runtime'],runtime)
            self.assertEqual(set(bound['consumer']['proof']),{'health','sid_gates','collision','validation_hits','hits_gate','a1_native','evidence'})
            for mutate in (lambda g:g.update(receipt_sha256='0'*64),lambda g:g.update(request_sha256='f'*64),lambda g:g['producer'].update(**{'run.json':'f'*64}),lambda g:g['proof'].update(validation_hits=379),lambda g:g.update(unlisted=True)):
                old=copy.deepcopy(got);mutate(got)
                with self.assertRaises(ValueError):p.require_acceptance(value,'bad-result')
                got.clear();got.update(old)
            with patch.object(p.original,'runtime',return_value={**runtime,'unlisted':1}):
                with self.assertRaisesRegex(ValueError,'runtime lock'):p.require_acceptance(value,'bad-runtime')
            with patch.object(p.subprocess,'run',return_value=subprocess.CompletedProcess([],2,'','rejected')):
                with self.assertRaisesRegex(ValueError,'consumer rejected'):p.require_acceptance(value,'rejected')
        self.assertGreater(len(calls),1)

    def test_acceptance_records_reject_old_protocol_and_exact_original_evidence(self):
        p=self.p;value,source,receipt,runtime,got=self.gate_fixture();path=Path(value['acceptance']['receipt'])
        with patch.object(p,'pinned_tree',return_value=source):
            records=p.acceptance_records(value)[2]
            self.assertIn(receipt['request']['path'],[r['path'] for r in records])
            self.assertTrue(set(receipt['proof']['evidence'])<=set(r['path'] for r in records))
            for key,val in [('schema','gpu-sgns-subset-a1-stage-v1'),('backend','numpy'),('verified',1),('subset_four_gate_acceptance',False),('source',{'commit':source['commit'],'files':{}})]:
                bad={**receipt,key:val};path.write_text(json.dumps(bad));value['acceptance']['receipt_sha256']=p.digest(path)
                with self.subTest(key=key),self.assertRaises(ValueError):p.acceptance_records(value)
            path.write_text(json.dumps(receipt));value['acceptance']['receipt_sha256']=p.digest(path)
            Path(receipt['request']['path']).write_bytes(b'changed original request')
            with self.assertRaisesRegex(ValueError,'request bytes'):p.acceptance_records(value)

    def test_every_formal_stage_consumes_before_full_inputs_or_launch(self):
        p=self.p;value=self.request();path=self.base/'request.json';path.write_text(json.dumps(value));sha=p.digest(path)
        with patch.object(p,'verification_guard',return_value={'synthetic':True}),patch.object(p,'require_acceptance',side_effect=ValueError('four gates rejected')),patch.object(p,'input_proof',side_effect=AssertionError('full input opened')),patch('analysis.a40_ops.load_ops',side_effect=AssertionError('launcher reached')):
            for stage in ('preflight','fit'):
                with self.assertRaisesRegex(ValueError,'four gates'):p.execute(stage,path,sha)
                with self.assertRaisesRegex(ValueError,'four gates'):p.prepare(stage,path,sha,self.base/'ops-code')
                self.assertEqual(p.main(['verify','--stage',stage,'--request',str(path),'--request-sha256',sha]),2)

    def test_source_delta_closes_old_bytes_added_paths_and_pinned_hashes(self):
        old={'commit':self.p.INPUT_SOURCE,'files':{'old.py':'1'*64}};delta={k:'2'*64 for k in self.p.OWN_PATHS}
        actual={'commit':'a'*40,'files':{**old['files'],**self.p.NATIVE_ADDITIONS,**delta}}
        proof=self.p.source_delta(actual,old,delta);self.assertEqual(proof['base_commit'],self.p.INPUT_SOURCE)
        attacks=[]
        for name,sha in [('old.py','3'*64),('unlisted.py','4'*64),(next(iter(self.p.NATIVE_ADDITIONS)),'5'*64)]:
            x=copy.deepcopy(actual);x['files'][name]=sha;attacks.append(x)
        x=copy.deepcopy(actual);del x['files'][next(iter(delta))];attacks.append(x)
        for x in attacks:
            with self.assertRaises(ValueError):self.p.source_delta(x,old,delta)
    def test_wall_is_ceil_original_projection_times_one_point_five(self):
        for projection,want in [(1.01,2),(1.334,3),(2.,3),(2.01,4)]:
            self.assertEqual(self.p.registered_wall({'projection_hours':projection}),want)
        for bad in (True,0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.p.registered_wall({'projection_hours':bad})
    def test_three_hour_projection_stop_and_boundary(self):
        p=self.p
        self.assertIsNone(p.check_projection(10799,1,1000,3))
        # 10800 seconds / 50 completed of 100 = six hours; registered three hours => STOP.
        with self.assertRaisesRegex(RuntimeError,'projection'):p.check_projection(10800,50,100,3)
        self.assertEqual(p.check_projection(10800,100,100,3),3.)
        with self.assertRaises(RuntimeError):p.check_projection(10800,0,100,3)
    def test_observer_restores_native_methods_and_preserves_cpu_science(self):
        import numpy as np
        from vkgr import sgns_gpu_native as n
        catalog=list(range(1,14));histories={i:[1+i*3+j%3 for j in range(18)] for i in range(4)};histories[4]=[13,13]
        content=np.random.default_rng(3).normal(size=(13,64)).astype('f4')
        first=n.fit_sgns(catalog,histories,content,device='cpu',compile_step=False)
        before=(n.NativeSGNS.__init__,n.NativeSGNS.epoch);events=[]
        with self.p.observer(3,events.append,every=1):second=n.fit_sgns(catalog,histories,content,device='cpu',compile_step=False)
        self.assertTrue(np.array_equal(first[0],second[0]));self.assertEqual(first[1],second[1]);self.assertEqual(before,(n.NativeSGNS.__init__,n.NativeSGNS.epoch))
        self.assertTrue(any(x['event']=='batch_progress' and x['processed_positive_pairs']>0 for x in events))
        self.assertEqual(sum(x['event']=='epoch_complete' for x in events),5)
        with self.assertRaises(RuntimeError):
            with self.p.observer(3,events.append):raise RuntimeError('failed')
        self.assertEqual(before,(n.NativeSGNS.__init__,n.NativeSGNS.epoch))
    def test_native_full_audit_keeps_schema_and_formal_full_binding(self):
        import numpy as np
        from vkgr import sgns_gpu_native as n
        catalog=list(range(1,14));histories={i:[1+i*3+j%3 for j in range(18)] for i in range(4)};histories[4]=[13,13]
        content=np.random.default_rng(3).normal(size=(13,64)).astype('f4');vectors,audit=n.fit_sgns(catalog,histories,content,device='cpu',compile_step=False)
        # Synthetic cardinality cannot satisfy formal full binding: only that external frozen full DAG is substituted here.
        with patch('vkgr.sgns_full_verify.check_full_binding',return_value=None):
            proof=self.p.verify_audit((catalog,histories,content),vectors,audit,{},device='cpu',compiled=False)
            self.assertTrue(proof['verified']);self.assertFalse(proof['gradient_replay'])
            for key,val in [('schema','vkgr.sid_coll_sgns.v1'),('epochs_completed',4),('health',{}),('training_contract',{}),('fallback_count',99)]:
                with self.subTest(key=key),self.assertRaises(ValueError):self.p.verify_audit((catalog,histories,content),vectors,{**audit,key:val},{},device='cpu',compiled=False)
        with self.assertRaises(ValueError):self.p.verify_audit((catalog,histories,content),vectors,audit,{},device='cpu',compiled=False)
    def test_fit_verify_rechecks_persisted_bytes_gate_and_native_audit(self):
        import numpy as np
        p=self.p;value=self.request();path=self.base/'request.json';path.write_text(json.dumps(value));sha=p.digest(path)
        run=Path(value['ops_root'])/'runs'/value['fit_run_id'];(run/'logs').mkdir(parents=True)
        (run/'run.json').write_bytes(b'synthetic job identity')
        out=Path(value['output'])/'fit';out.mkdir(parents=True)
        np.save(out/'vectors.npy',np.ones((2,64),dtype='f4'));(out/'audit.json').write_text(json.dumps({'synthetic':True}))
        bound={'original':{'manifest':{'binding':{'synthetic_unit_fixture':True}}}};gate={'synthetic_consumer':True};proof={'synthetic_native_audit':True};benchmark={'projection_hours':2.}
        receipt={'schema':'sgns-native-full-stage-v1','status':'verified','verified':True,'engineering':False,'stage':'fit','backend':'B','device':'cuda','seed':42,'input_binding':bound,'request_sha256':sha,'run_id':value['fit_run_id'],'run_manifest_sha256':p.digest(run/'run.json'),'config':p.subset.CONFIG,'test_read':False,'validation_read':False,'acceptance':gate,'benchmark_producer':benchmark,'registered_wall_hours':3,'artifacts':{name:p.digest(out/name) for name in ('vectors.npy','audit.json')},'native_full_audit':proof}
        (out/'result.json').write_text(json.dumps(receipt));(run/'logs/stdout.log').write_text(json.dumps(receipt))
        terminal={'schema':'vkgr-terminal-v1','run_id':value['fit_run_id'],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False,'started_at':'synthetic-start','ended_at':'synthetic-end','wall_seconds':1.}
        (run/'terminal.json').write_text(json.dumps(terminal))
        with patch.object(p,'verification_context',return_value={'synthetic_full_context':True},create=True),patch.object(p,'job',return_value=p.digest(run/'run.json')),patch.object(p,'require_acceptance',return_value=gate),patch.object(p,'benchmark_proof',return_value=benchmark),patch.object(p,'verify_audit',return_value=proof) as audit:
            p.accepted(value,path,sha,'fit',bound,arrays=('synthetic',))
            self.assertEqual(audit.call_count,1)
            with patch.object(p,'verify_audit',return_value={'different':True}):
                with self.assertRaisesRegex(ValueError,'audit recheck'):p.accepted(value,path,sha,'fit',bound,arrays=('synthetic',))
            with patch.object(p,'require_acceptance',return_value={'different':True}):
                with self.assertRaisesRegex(ValueError,'four-gate binding'):p.accepted(value,path,sha,'fit',bound,arrays=('synthetic',))
            with patch.object(p,'benchmark_proof',return_value={'projection_hours':3.}):
                with self.assertRaisesRegex(ValueError,'benchmark/wall'):p.accepted(value,path,sha,'fit',bound,arrays=('synthetic',))
            with patch.object(p,'verification_context',side_effect=[{'start':True},{'end':True}]):
                with self.assertRaisesRegex(ValueError,'closure changed'):p.accepted(value,path,sha,'fit',bound,arrays=('synthetic',))
            own_paths=[run/'run.json',run/'terminal.json',run/'logs/stdout.log',out/'result.json',out/'vectors.npy',out/'audit.json']
            for changed in own_paths:
                before=changed.read_bytes()
                def drift(*args):
                    if changed==run/'terminal.json':changed.write_text(json.dumps({**terminal,'wall_cap_exceeded':True}))
                    else:changed.write_bytes(before+b' ')
                    return gate
                with self.subTest(drift=changed.name),patch.object(p,'require_acceptance',side_effect=drift):
                    with self.assertRaises((ValueError,json.JSONDecodeError)):p.accepted(value,path,sha,'fit',bound,arrays=('synthetic',))
                changed.write_bytes(before)
            (out/'vectors.npy').write_bytes(b'changed vector bytes')
            with self.assertRaisesRegex(ValueError,'artifact bytes'):p.accepted(value,path,sha,'fit',bound,arrays=('synthetic',))

    def test_terminal_requires_exact_finite_duration_within_own_registered_cap(self):
        p=self.p;value=self.request()
        terminal={'schema':'vkgr-terminal-v1','run_id':value['fit_run_id'],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False,'started_at':'synthetic-start','ended_at':'synthetic-end','wall_seconds':10800.}
        p.check_terminal(value,'fit',terminal)
        for key,val in [('wall_seconds',10800.001),('wall_seconds',float('nan')),('wall_seconds',float('inf')),('wall_seconds',True),('wall_seconds',-1),('wall_cap_exceeded',0),('exit_code',False),('unexpected',1)]:
            with self.subTest(key=key,val=val),self.assertRaises(ValueError):p.check_terminal(value,'fit',{**terminal,key:val})
        missing=dict(terminal);del missing['wall_cap_exceeded']
        with self.assertRaises(ValueError):p.check_terminal(value,'fit',missing)

    def test_acceptance_evidence_mutation_during_consume_is_rejected(self):
        p=self.p;value,source,receipt,runtime,got=self.gate_fixture()
        def changed(args,**kwargs):
            Path(next(iter(receipt['proof']['evidence']))).write_bytes(b'changed during consume')
            return subprocess.CompletedProcess(args,0,json.dumps(got),'')
        with patch.object(p,'pinned_tree',return_value=source),patch.object(p.original,'runtime',return_value=runtime),patch.object(p.subprocess,'run',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'evidence bytes'):p.require_acceptance(value,'drift')

    def test_cli_verify_closes_evidence_across_first_consume(self):
        p=self.p;value=self.request();path=self.base/'request.json';path.write_text(json.dumps(value));sha=p.digest(path)
        for stage in ('preflight','fit'):
            with patch.object(p,'verification_guard',side_effect=[{'own':'before'},{'own':'after'}]),patch.object(p,'require_acceptance',return_value={}),patch.object(p,'input_proof',return_value=(('synthetic',),{})),patch.object(p,'accepted',return_value={'synthetic':'accepted'}):
                self.assertEqual(p.main(['verify','--stage',stage,'--request',str(path),'--request-sha256',sha]),2)

    def test_exact_external_git_tree_rejects_ignored_shadow(self):
        p=self.p;root=self.base/'a1-fixed'
        subprocess.run(['git','clone','-q','--shared','--no-checkout',str(ROOT),str(root)],check=True,capture_output=True)
        subprocess.run(['git','-C',str(root),'checkout','-q','--detach',p.ACCEPTANCE_SOURCE],check=True,capture_output=True)
        source=p.pinned_tree(root,p.ACCEPTANCE_SOURCE)
        self.assertEqual(source['files'][p.ACCEPTANCE_ENTRY],p.ACCEPTANCE_ENTRY_SHA)
        shadow=root/'cache/shadow.py';shadow.parent.mkdir(exist_ok=True);shadow.write_text('ignored source shadow')
        with self.assertRaisesRegex(ValueError,'code tree'):p.pinned_tree(root,p.ACCEPTANCE_SOURCE)

    def test_formal_cli_illegal_and_nonempty_dry(self):
        for command,want in [('dry',0),('illegal',2)]:
            result=subprocess.run([sys.executable,'-B','-m','analysis.sgns_native_full',command],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
            self.assertEqual(result.returncode,want,result.stderr)
            if want==0:self.assertEqual(json.loads(result.stdout)['training'],False)

    def test_progress_emitter_uses_actual_stderr_protocol(self):
        import contextlib,io
        output=io.StringIO()
        with contextlib.redirect_stderr(output):self.p.emit_progress({'event':'batch_progress','processed_positive_pairs':4096,'seconds':1.})
        record=json.loads(output.getvalue());self.assertEqual(record['phase'],'batch_progress');self.assertEqual(record['processed_positive_pairs'],4096)
    def test_own_job_env_resource_argv_and_ordered_pins(self):
        p=self.p;value=self.request();path=self.base/'request.json';path.write_text(json.dumps(value));sha=p.digest(path)
        run=Path(value['ops_root'])/'runs'/value['preflight_run_id'];run.mkdir(parents=True)
        (self.base/'lock').write_bytes(b'synthetic lock')
        inputs=[{'path':str(path),'sha256':sha},{'path':str(self.base/'lock'),'sha256':p.digest(self.base/'lock')}]
        record={'schema':'vkgr-run-v1','run_id':value['preflight_run_id'],'kind':'ops','cwd':str(p.ROOT),'argv':p.argv('preflight',path,sha),'seed':42,
            'git':{'commit':value['runner_source'],'dirty':False,'dirty_files':[]},'config':{'path':str(path),'sha256':sha},'resources':p.resources(value),'env':p.job_env(value,'preflight'),'inputs':[{**item,'bytes':Path(item['path']).stat().st_size} for item in inputs],'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158','sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},'overlay':None,'pinned_files':{},'stop_mode':'signal'}
        with patch.object(p,'pins',return_value=inputs):
            (run/'run.json').write_text(json.dumps(record));self.assertEqual(p.job(value,path,sha,'preflight'),p.digest(run/'run.json'))
            for key,val in [('seed',True),('resources',{**record['resources'],'wall_seconds':1.}),('argv',record['argv'][:-1]),('env',{}),('inputs',list(reversed(inputs))),('ops_source',{'commit':'f'*40,'sha256':'e'*64}),('overlay',{'unregistered':'input'}),('pinned_files',{'unregistered':'f'*64}),('stop_mode','unregistered')]:
                (run/'run.json').write_text(json.dumps({**record,key:val}))
                with self.subTest(key=key),self.assertRaises(ValueError):p.job(value,path,sha,'preflight')
    def test_original_context_integrity_checked_before_loading_old_modules(self):
        value=self.request();Path(value['input_producer']['root']).mkdir()
        with self.assertRaises((ValueError,subprocess.CalledProcessError)):self.p.input_proof(value)

    def test_real_launcher_deployment_metadata_and_dry_reject_unaccepted(self):
        import shutil
        p=self.p;base=self.base/'launcher';base.mkdir()
        def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.PIPE,text=True).strip()
        def clone(name,ref):
            root=base/name;subprocess.run(['git','clone','-q','--shared','--no-checkout',str(ROOT),str(root)],check=True,capture_output=True);git(root,'checkout','-q','--detach',ref);return root
        runner=clone('runner',p.INPUT_SOURCE);ops=clone('ops-code','5840e6b686ec9651314a9e28a1a2c0befcc05158')
        a1=clone('a1-code',p.ACCEPTANCE_SOURCE)
        for name in p.NATIVE_ADDITIONS:
            target=runner/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(subprocess.check_output(['git','-C',str(ROOT),'show','305c415871bd60dc0d7d8a064a7e893830a9c20f:'+name]));git(runner,'add',name)
        for name in p.OWN_PATHS:
            target=runner/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target);git(runner,'add',name)
        git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','--allow-empty','-q','-m','full native transport fixture')
        value=self.request();value.update(runner_source=git(runner,'rev-parse','HEAD'),source_delta={n:p.digest(runner/n) for n in p.OWN_PATHS})
        value['acceptance']['root']=str(a1)
        receipt=Path(value['acceptance']['receipt']);receipt.parent.mkdir(parents=True);receipt.write_bytes(b'{"verified":true}')
        value['acceptance']['receipt_sha256']=p.digest(receipt)
        Path(value['lock']).write_bytes(b'fixture lock')
        path=base/'request.json';path.write_text(json.dumps(value));sha=p.digest(path)
        env={**os.environ,'PYTHONPATH':str(runner/'src'),'PYTHONDONTWRITEBYTECODE':'1'};cmd=[sys.executable,'-B','-m','analysis.sgns_native_full']
        # Actual pinned OPS launcher metadata and same-module dry, using inert
        # admission bytes. This is not a successful full plan or science PF.
        from analysis.a40_ops import load_ops
        _,launcher,_,_=load_ops(ops)
        argv=cmd+['preflight','--request',str(path),'--request-sha256',sha,'--execute-on-new-instance']
        flags=['--root',str(base/'ops'),'--run-id','sgns-native-full-launcher-fixture','--kind','ops','--cwd',str(runner),'--config',str(path),'--seed','42','--input',str(receipt),'--cpu-cores','2','--ram-mib','196608','--vram-mib','32768','--disk-gib','16','--wall-hours','3']
        spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*argv]));metadata=launcher.prepare(spec)
        self.assertEqual(metadata['git']['dirty_files'],[]);self.assertEqual(metadata['git']['commit'],value['runner_source']);self.assertEqual(metadata['config']['sha256'],sha)
        dry=subprocess.run(cmd+['dry'],cwd=runner,env=env,capture_output=True,text=True);self.assertEqual(dry.returncode,0,dry.stderr);self.assertFalse(json.loads(dry.stdout)['full_fit_enabled'])
        for stage in ('preflight','fit'):
            args=cmd+['plan','--stage',stage,'--request',str(path),'--request-sha256',sha,'--ops-code',str(ops)]
            denied=subprocess.run(args,cwd=runner,env=env,capture_output=True,text=True);self.assertEqual(denied.returncode,2,denied.stderr);self.assertIn('acceptance',denied.stdout)
            args=cmd+[stage,'--request',str(path),'--request-sha256',sha,'--execute-on-new-instance']
            denied=subprocess.run(args,cwd=runner,env=env,capture_output=True,text=True);self.assertEqual(denied.returncode,2,denied.stderr);self.assertIn('acceptance',denied.stdout)
            args=cmd+['verify','--stage',stage,'--request',str(path),'--request-sha256',sha]
            denied=subprocess.run(args,cwd=runner,env=env,capture_output=True,text=True);self.assertEqual(denied.returncode,2,denied.stderr);self.assertIn('acceptance',denied.stdout)
        self.assertFalse((Path(value['ops_root'])/'queue').exists());self.assertFalse(Path(value['output']).exists())

if __name__=='__main__':unittest.main()
