import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

MODULE='ops.tools.remote.native_a1_full_finite_chain'

class FullChainTests(unittest.TestCase):
    def module(self):return importlib.import_module(MODULE)
    def fixture(self,p):
        m=self.module();a1={'source':{'commit':m.A1_SOURCE},'ops_root':str(p/'ops'),'ops_code':str(p/'ops-code'),'output_root':str(p/'a1-output')}
        template={'schema':'sgns-native-full-request-v1','runner_source':m.FULL_SOURCE,'resources':{'wall_hours':3},'ops_root':a1['ops_root'],'acceptance':{'root':str(p/'a1-code'),'receipt':str(p/'a1-output/accept/stage-receipt.json'),'receipt_sha256':'0'*64},'preflight_run_id':'sgns-native-full-preflight-v1','fit_run_id':'sgns-native-full-fit-v1','output':str(p/'full-output')}
        args=SimpleNamespace(evidence=p/'evidence',a1_request=p/'a1.json',a1_request_sha256='a'*64,template=p/'template.json',driver_source='b'*40,driver_sha256='c'*64,deadline_hours=1)
        return args,a1,template
    def test_fixed_sources_wall_and_pending_hash_only(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);_,a1,t=self.fixture(p);old=json.loads(json.dumps(t));receipt=Path(t['acceptance']['receipt']);receipt.parent.mkdir(parents=True);receipt.write_text('{}')
            got=m.materialize(t,a1,m.sha(receipt));self.assertEqual(t,old);self.assertEqual(got['acceptance']['receipt_sha256'],m.sha(receipt));got['acceptance']['receipt_sha256']='0'*64;self.assertEqual(got,t)
            for field,value in [('runner_source','wrong'),('resources',{'wall_hours':4})]:
                bad={**t,field:value}
                with self.assertRaises(ValueError):m.materialize(bad,a1,m.sha(receipt))
            t['acceptance']['receipt_sha256']='f'*64
            with self.assertRaises(ValueError):m.materialize(t,a1,m.sha(receipt))
    def test_evidence_overlap_before_write(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();args,a1,t=self.fixture(p);args.evidence=p/'full-output/new'
            with self.assertRaises(ValueError):m.evidence_path(args,a1,t)
            self.assertFalse(args.evidence.exists())
    def test_receipt_producer_hashes_exact(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);_,a1,t=self.fixture(p);stage='fit';rid=t['fit_run_id'];run=p/'ops/runs'/rid;(run/'logs').mkdir(parents=True);out=p/'full-output/fit';out.mkdir(parents=True)
            for f in (run/'run.json',run/'terminal.json',run/'logs/stdout.log',out/'result.json'):f.write_text('{}')
            good={'status':'verified','readonly':True,'producer':{'run_id':rid,'receipt_sha256':m.sha(out/'result.json'),'run_manifest_sha256':m.sha(run/'run.json'),'terminal_sha256':m.sha(run/'terminal.json'),'stdout_sha256':m.sha(run/'logs/stdout.log')}}
            m.verify_result(t,stage,good);good['producer']['stdout_sha256']='0'*64
            with self.assertRaises(ValueError):m.verify_result(t,stage,good)
    def test_formal_module_dry_and_bad_command(self):
        for argv,code in [(['dry'],0),(['unknown'],2)]:
            r=subprocess.run([sys.executable,'-B','-m',MODULE,*argv],capture_output=True,text=True);self.assertEqual(r.returncode,code,r.stderr)
            if code==0:self.assertFalse(json.loads(r.stdout)['training'])
    def test_a1_science_rejection_prevents_full_request_and_submission(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();args,a1,t=self.fixture(p)
            with patch.object(m.os,'getppid',return_value=1),patch.object(m,'fixed_inputs',return_value=(a1,t)),patch.object(m,'wait_a1',side_effect=ValueError('four gates failed')),patch.object(m,'execute') as execute,patch.object(m.subprocess,'run') as submit:
                with self.assertRaisesRegex(ValueError,'four gates failed'):m.run(args,now=lambda:0)
            execute.assert_not_called();submit.assert_not_called();self.assertFalse((args.evidence/'full-request.json').exists());self.assertEqual(json.loads((args.evidence/'summary.json').read_text())['status'],'ALERT')

    def test_scheduler_plain_receipt_is_not_json(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args=SimpleNamespace(evidence=p,deadline=10,now=lambda:0)
            r=SimpleNamespace(returncode=0,stdout=b'00001-sgns-native-full-fit-v1.json\n',stderr=b'')
            with patch.object(m,'fixed_inputs'),patch.object(m.subprocess,'run',return_value=r) as command:
                got=m.execute(args,m.FULL_CODE,['scheduler'],'submit',json_output=False)
            self.assertEqual(got,'00001-sgns-native-full-fit-v1.json');self.assertEqual(command.call_args.kwargs['timeout'],10);self.assertEqual((p/'submit.stdout').read_bytes(),r.stdout)

    def test_no_duplicate_submission_after_absent_receipt(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args,_,v=self.fixture(p);args.evidence.mkdir();args.now=lambda:0;args.deadline=10;args.full_request=p/'request';args.full_pin='d'*64;v['ops_root']=str(m.OPS_ROOT)
            plan={'status':'prepared_dry','spec':{'run_id':v['preflight_run_id']},'metadata':{'git':{'commit':m.FULL_SOURCE,'dirty':False,'dirty_files':[]}}};calls=[]
            def execute(a,c,argv,name,**kwargs):
                calls.append(name)
                if name.endswith('-plan'):return plan
                if name.endswith('-checked'):return {'verified':True,'submitted':False}
                return 'queue-receipt.json'
            with patch.object(m,'fixed_inputs'),patch.object(m,'queue_state',side_effect=[{'state':'absent','path':None}]*2),patch.object(m.base,'terminal',return_value=False),patch.object(m,'execute',side_effect=execute),patch.object(m,'submit_argv',return_value=['submit']):
                with self.assertRaisesRegex(ValueError,'submission absent'):m.full_stage(args,v,'preflight',lambda _:self.fail('must stop'))
            self.assertEqual(calls,['preflight-plan','preflight-plan-checked','preflight-submit'])

    def test_acceptance_receipt_and_request_lineage_must_match(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args,a1,t=self.fixture(p);args.evidence.mkdir();args.now=lambda:0;args.deadline=10;receipt=Path(t['acceptance']['receipt']);receipt.parent.mkdir(parents=True);receipt.write_text('{}');run=Path(a1['ops_root'])/'runs/sgns-native-subset-a1-accept-v1';(run/'logs').mkdir(parents=True)
            for f in ('run.json','terminal.json','logs/stdout.log'):(run/f).write_text('{}')
            good={'receipt_sha256':m.sha(receipt),'request_sha256':args.a1_request_sha256,'producer':{f:m.sha(run/f) for f in ('run.json','terminal.json','logs/stdout.log')},'proof':{}}
            with patch.object(m,'fixed_inputs'),patch.object(m.base,'terminal',return_value=True),patch.object(m,'queue_state',return_value={'state':'claimed','path':None}),patch.object(m,'execute',return_value=good):self.assertEqual(m.wait_a1(args,a1,t,lambda _:None),m.sha(receipt))
            good['request_sha256']='f'*64
            with patch.object(m,'fixed_inputs'),patch.object(m.base,'terminal',return_value=True),patch.object(m,'queue_state',return_value={'state':'claimed','path':None}),patch.object(m,'execute',return_value=good):
                with self.assertRaises(ValueError):m.wait_a1(args,a1,t,lambda _:None)

    def test_pending_a1_last_wake_never_requeries(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args,a1,t=self.fixture(p);args.evidence.mkdir();clock=[0];args.now=lambda:clock[0];args.deadline=1;queries=[];sleeps=[]
            def query(*_):queries.append(clock[0]);return {'state':'absent','path':None}
            def sleep(s):sleeps.append(s);clock[0]+=s
            with patch.object(m,'fixed_inputs'),patch.object(m,'queue_state',side_effect=query):
                with self.assertRaises(TimeoutError):m.wait_a1(args,a1,t,sleep)
            self.assertEqual(queries,[0]);self.assertEqual(sleeps,[1])

    def test_final_identity_and_deadline_prevent_accepted_summary(self):
        m=self.module()
        for mode in ('identity','deadline'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                p=Path(d).resolve();args,a1,t=self.fixture(p);clock=[0];changed=[False]
                def fixed(a):
                    if changed[0] and mode=='identity':raise ValueError('driver changed')
                    return a1,t
                def stage(a,v,s,sl):
                    if s=='fit':changed[0]=True;clock[0]=3601 if mode=='deadline' else 0
                with patch.object(m.os,'getppid',return_value=1),patch.object(m,'fixed_inputs',side_effect=fixed),patch.object(m,'wait_a1',return_value='e'*64),patch.object(m,'full_stage',side_effect=stage):
                    with self.assertRaises((ValueError,TimeoutError)):m.run(args,now=lambda:clock[0])
                self.assertEqual(json.loads((args.evidence/'summary.json').read_text())['status'],'ALERT')

    def test_existing_pending_job_disappearing_never_resubmits(self):
        m=self.module()
        for kind in ('a1','full'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                p=Path(d);args,a1,v=self.fixture(p);args.evidence.mkdir();args.now=lambda:0;args.deadline=10;args.full_request=p/'request';args.full_pin='e'*64;calls=[]
                def execute(*a,**kw):calls.append(a);raise AssertionError('duplicate submission/plan')
                with patch.object(m,'fixed_inputs'),patch.object(m.base,'terminal',return_value=False),patch.object(m,'queue_state',side_effect=[{'state':'pending','path':None},{'state':'absent','path':None}]),patch.object(m,'execute',side_effect=execute):
                    with self.assertRaisesRegex(ValueError,'disappeared'):
                        if kind=='a1':m.wait_a1(args,a1,v,lambda _:None)
                        else:m.full_stage(args,v,'preflight',lambda _:None)
                self.assertEqual(calls,[])

    def test_owner_template_request_and_generated_receipt_drift_rejected(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args,a1,t=self.fixture(p);args.now=lambda:0;args.deadline=10;args.template.write_text(json.dumps(t));args.a1_request.write_text(json.dumps(a1));args.a1_request_sha256=m.sha(args.a1_request);args.driver_sha256=m.sha(m.__file__)
            expected=m.sha(args.template);original_sha=m.sha
            def hashed(path):
                if str(path).endswith('analysis/sgns_native_subset_a1.py'):return m.base.CONSUMER_ENTRY_SHA
                if str(path).endswith('analysis/sgns_native_full.py'):return m.FULL_ENTRY_SHA
                return original_sha(path)
            with patch.object(m,'tree'),patch.object(m,'TEMPLATE_SHA',expected),patch.object(m,'sha',side_effect=hashed):
                self.assertEqual(m.fixed_inputs(args),(a1,t))
                args.template.write_text(json.dumps({**t,'compiled':False}))
                with self.assertRaisesRegex(ValueError,'template/original A1'):m.fixed_inputs(args)
                args.template.write_text(json.dumps(t));args.a1_request.write_text('{}')
                with self.assertRaisesRegex(ValueError,'template/original A1'):m.fixed_inputs(args)
                args.a1_request.write_text(json.dumps(a1));receipt=Path(t['acceptance']['receipt']);receipt.parent.mkdir(parents=True);receipt.write_text('{}');args.receipt_pin='f'*64
                with self.assertRaisesRegex(ValueError,'acceptance receipt changed'):m.fixed_inputs(args)
                args.receipt_pin=m.sha(receipt);args.full_request=p/'full.json';args.full_request.write_text('{}');args.full_pin='f'*64
                with self.assertRaisesRegex(ValueError,'generated full request changed'):m.fixed_inputs(args)

    def test_execute_original_failure_cannot_continue(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);args=SimpleNamespace(evidence=p,deadline=10,now=lambda:0)
            failed=SimpleNamespace(returncode=2,stdout=b'{"status":"rejected"}',stderr=b'original failure')
            with patch.object(m.subprocess,'run',return_value=failed),patch.object(m,'fixed_inputs') as check:
                with self.assertRaisesRegex(ValueError,'original CLI failed'):m.execute(args,m.FULL_CODE,['verify'],'failed')
            check.assert_not_called();self.assertEqual((p/'failed.stderr').read_bytes(),b'original failure')

    def test_execute_long_subcommand_deadline(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);clock=[0];args=SimpleNamespace(evidence=p,deadline=10,now=lambda:clock[0])
            def execute(*argv,**kwargs):
                self.assertEqual(kwargs['timeout'],10);clock[0]=11;return SimpleNamespace(returncode=0,stdout=b'{}',stderr=b'')
            with patch.object(m.subprocess,'run',side_effect=execute),patch.object(m,'fixed_inputs'):
                with self.assertRaises(TimeoutError):m.execute(args,m.FULL_CODE,['dry'],'expired')
            self.assertTrue((p/'expired.stdout').exists())

    def test_failed_full_preflight_verification_never_reaches_fit(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();args,a1,t=self.fixture(p);stages=[]
            def stage(a,v,s,sl):stages.append(s);raise ValueError('original preflight verification rejected')
            with patch.object(m.os,'getppid',return_value=1),patch.object(m,'fixed_inputs',return_value=(a1,t)),patch.object(m,'wait_a1',return_value='e'*64),patch.object(m,'full_stage',side_effect=stage):
                with self.assertRaisesRegex(ValueError,'preflight verification rejected'):m.run(args,now=lambda:0)
            self.assertEqual(stages,['preflight']);self.assertEqual(json.loads((args.evidence/'summary.json').read_text())['status'],'ALERT')

    def test_tree_rejects_ignored_extras_and_symlink_directory(self):
        m=self.module()
        for kind in ('ignored-extra','ignored-bytecode','symlink-dir'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d)/'code';root.mkdir()
                (root/'.gitignore').write_text('ignored-*\n__pycache__/\n')
                (root/'entry.py').write_text('value = 1\n')
                def git(*argv):return subprocess.check_output(['git','-C',str(root),*argv],stderr=subprocess.PIPE).decode().strip()
                git('init','-q');git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','commit','-qm','Synthetic fixed tree')
                source=git('rev-parse','HEAD');args=SimpleNamespace(now=lambda:0,deadline=30)
                m.tree(args,root,source)
                if kind=='ignored-extra':(root/'ignored-shadow.py').write_text('shadow = True\n')
                elif kind=='ignored-bytecode':
                    (root/'__pycache__').mkdir();(root/'__pycache__/entry.pyc').write_bytes(b'synthetic ignored bytecode')
                else:
                    target=Path(d)/'outside';target.mkdir();(root/'ignored-link').symlink_to(target,target_is_directory=True)
                self.assertEqual(git('status','--porcelain','--untracked-files=all'),'')
                with self.assertRaises(ValueError):m.tree(args,root,source)
