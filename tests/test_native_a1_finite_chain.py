import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

MODULE='ops.tools.remote.native_a1_finite_chain'

class ChainTests(unittest.TestCase):
    def module(self):return importlib.import_module(MODULE)
    def test_deadline_finite(self):
        m=self.module()
        for n in (True,0,-1,201,float('nan'),float('inf')):
            with self.subTest(n=n),self.assertRaises(ValueError):m.deadline_seconds(n)
        self.assertEqual(m.deadline_seconds(200),720000)
    def test_exact_fixed_sources(self):
        m=self.module();self.assertEqual(m.CONSUMER_SOURCE,'96bfb9e6fcb4081168cf4d39563f50c58ee3fbad')
        self.assertEqual(m.STAGES,('preflight','train','verify','accept'))
        self.assertEqual(m.POLL_SECONDS,900)
    def test_terminal_failclosed(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.assertFalse(m.terminal(p,'a'))
            good={'schema':'vkgr-terminal-v1','run_id':'a','status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}
            (p/'terminal.json').write_text(json.dumps(good));self.assertTrue(m.terminal(p,'a'))
            for k,v in [('run_id','b'),('status','failed'),('exit_code',1),('signal',15),('wall_cap_exceeded',True)]:
                (p/'terminal.json').write_text(json.dumps({**good,k:v}))
                with self.subTest(k=k),self.assertRaises(ValueError):m.terminal(p,'a')
    def test_plan_drift_refused(self):
        m=self.module();plan={'status':'prepared_dry','spec':{'run_id':'sgns-native-subset-a1-train-v1'},'metadata':{'git':{'commit':m.CONSUMER_SOURCE,'dirty':False,'dirty_files':[]}}}
        m.check_plan(plan,'train')
        for f in ('status','commit','dirty','run_id'):
            p=json.loads(json.dumps(plan))
            if f=='status':p[f]='other'
            elif f=='run_id':p['spec'][f]='wrong'
            else:p['metadata']['git'][f]='wrong'
            with self.subTest(f=f),self.assertRaises(ValueError):m.check_plan(p,'train')
    def test_formal_subprocess(self):
        for args,code in [(['dry'],0),(['unknown'],2)]:
            r=subprocess.run([sys.executable,'-B','-m',MODULE,*args],capture_output=True,text=True)
            self.assertEqual(r.returncode,code,r.stderr)
            if code==0:self.assertFalse(json.loads(r.stdout)['training'])

    def test_existing_complete_chain_delegates_each_gate_once(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();args=SimpleNamespace(evidence=p/'evidence',consumer_code=p/'code',request=p/'request',request_sha256='a'*64,driver_source='b'*40,driver_sha256='c'*64,deadline_hours=1)
            v={'ops_root':str(p/'ops'),'ops_code':str(p/'ops-code'),'output_root':str(p/'output')}
            calls=[]
            with patch.object(m.os,'getppid',return_value=1),patch.object(m,'fixed_request',return_value=v),patch.object(m,'terminal',return_value=True),patch.object(m,'queue_state',return_value={'state':'claimed','path':None}),patch.object(m,'command',side_effect=lambda a,b,n:calls.append(n)):
                result=m.run(args,now=lambda:0,sleep=lambda _:self.fail('completed chain slept'))
            self.assertEqual(calls,[s+'-accepted' for s in m.STAGES]);self.assertFalse(result['full_fit_submitted'])
    def test_absent_initial_pf_alert_never_submits(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();args=SimpleNamespace(evidence=p/'evidence',consumer_code=p/'code',request=p/'request',request_sha256='a'*64,driver_source='b'*40,driver_sha256='c'*64,deadline_hours=1)
            v={'ops_root':str(p/'ops'),'ops_code':str(p/'ops-code'),'output_root':str(p/'output')}
            with patch.object(m.os,'getppid',return_value=1),patch.object(m,'fixed_request',return_value=v),patch.object(m,'queue_state',return_value={'state':'absent','path':None}),patch.object(m,'command') as command,patch.object(m.subprocess,'run') as submit:
                with self.assertRaisesRegex(ValueError,'initial PF'):m.run(args,now=lambda:0,sleep=lambda _:None)
            command.assert_not_called();submit.assert_not_called();self.assertEqual(json.loads((args.evidence/'summary.json').read_bytes())['status'],'ALERT')
    def test_evidence_rejects_original_roots_before_creation(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();args=SimpleNamespace(evidence=p/'original'/'new',consumer_code=p/'code')
            with self.assertRaises(ValueError):m.evidence_path(args,{'data_root':str(p/'original')})
            self.assertFalse(args.evidence.exists())
    def test_no_training_permit_via_unknown_stage(self):
        m=self.module()
        with self.assertRaises(ValueError):m.check_plan({'status':'prepared_dry'},'fullfit')

    def test_accept_consumer_gets_receipt_path_not_request(self):
        m=self.module();import analysis
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();request=p/'request.json';request.write_text('{}');receipt=p/'output'/'accept'/'stage-receipt.json';receipt.parent.mkdir(parents=True);receipt.write_text('{"verified":true}')
            source={'commit':m.CONSUMER_SOURCE,'files':{'analysis/sgns_native_subset_a1.py':m.CONSUMER_ENTRY_SHA}}
            def consume(path,pin,scratch):
                self.assertEqual(path,receipt);self.assertEqual(pin,m.sha(receipt));return {'verified':True}
            fake=SimpleNamespace(load=lambda *_:{'output_root':str(p/'output')},identity=lambda _:(source,{}),consume=consume,digest=m.sha,output=lambda v,s:Path(v['output_root'])/s)
            with patch.object(analysis,'sgns_native_subset_a1',fake,create=True),patch.dict(sys.modules,{'analysis.sgns_native_subset_a1':fake}),patch.object(sys,'argv',['check',str(request),m.sha(request),'accept',str(p/'scratch'),m.CONSUMER_SOURCE,m.CONSUMER_ENTRY_SHA]):exec(compile(m.CHECK,'check','exec'),{})

    def test_final_identity_drift_cannot_write_success(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();args=SimpleNamespace(evidence=p/'evidence',consumer_code=p/'code',request=p/'request',request_sha256='a'*64,driver_source='b'*40,driver_sha256='c'*64,deadline_hours=1)
            v={'ops_root':str(p/'ops'),'ops_code':str(p/'ops-code'),'output_root':str(p/'output')};changed=[False]
            def fixed(_):
                if changed[0]:raise ValueError('driver changed')
                return v
            def command(a,b,n):
                if n=='accept-accepted':changed[0]=True
            with patch.object(m.os,'getppid',return_value=1),patch.object(m,'fixed_request',side_effect=fixed),patch.object(m,'terminal',return_value=True),patch.object(m,'queue_state',return_value={'state':'claimed','path':None}),patch.object(m,'command',side_effect=command):
                with self.assertRaisesRegex(ValueError,'changed'):m.run(args,now=lambda:0)
            self.assertEqual(json.loads((args.evidence/'summary.json').read_bytes())['status'],'ALERT')

    def test_missing_queue_with_residual_run_alerts_immediately(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();(p/'ops'/'runs'/'sgns-native-subset-a1-preflight-v1').mkdir(parents=True)
            args=SimpleNamespace(evidence=p/'evidence',consumer_code=p/'code',request=p/'request',request_sha256='a'*64,driver_source='b'*40,driver_sha256='c'*64,deadline_hours=1)
            v={'ops_root':str(p/'ops'),'ops_code':str(p/'ops-code'),'output_root':str(p/'output')}
            with patch.object(m.os,'getppid',return_value=1),patch.object(m,'fixed_request',return_value=v),patch.object(m,'queue_state',return_value={'state':'absent','path':None}):
                with self.assertRaisesRegex(ValueError,'disappeared'):m.run(args,now=lambda:0,sleep=lambda _:self.fail('lost job polled'))

    def test_accept_over_deadline_cannot_write_success(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();args=SimpleNamespace(evidence=p/'evidence',consumer_code=p/'code',request=p/'request',request_sha256='a'*64,driver_source='b'*40,driver_sha256='c'*64,deadline_hours=1)
            v={'ops_root':str(p/'ops'),'ops_code':str(p/'ops-code'),'output_root':str(p/'output')};clock=[0]
            def command(a,b,n):
                if n=='accept-accepted':clock[0]=3601
            with patch.object(m.os,'getppid',return_value=1),patch.object(m,'fixed_request',return_value=v),patch.object(m,'terminal',return_value=True),patch.object(m,'queue_state',return_value={'state':'claimed','path':None}),patch.object(m,'command',side_effect=command):
                with self.assertRaises(TimeoutError):m.run(args,now=lambda:clock[0])
            self.assertEqual(json.loads((args.evidence/'summary.json').read_bytes())['status'],'ALERT')

    def test_subprocess_is_bounded_by_remaining_deadline(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();args=SimpleNamespace(evidence=p,consumer_code=p,env={},deadline=20,now=lambda:10)
            def execute(*argv,**kwargs):
                self.assertGreater(kwargs['timeout'],0);self.assertLessEqual(kwargs['timeout'],10)
                return SimpleNamespace(returncode=0,stdout=b'{}',stderr=b'')
            with patch.object(m.subprocess,'run',side_effect=execute):m.command(args,['dry'],'bounded')
