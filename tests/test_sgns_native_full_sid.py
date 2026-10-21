"""Synthetic original-sid.fit parity and UR001 transport boundaries only."""
import copy
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

class FullSIDTests(unittest.TestCase):
    def setUp(self):
        from analysis import sgns_native_full_sid as m
        self.m=m;self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
    def toy(self):
        ids=np.arange(40,dtype=np.uint32);x=np.random.default_rng(7).normal(size=(40,64));x=(x/np.linalg.norm(x,axis=1)[:,None]).astype('f4')
        return ids,x
    def test_frozen_config_integer_cube_root(self):
        self.assertEqual(self.m.config(12012895),{'schema':'UR001-BHKM-config-v1','branching':229,'depth':3,'seed':42,'max_iter':10,'external_sort':False})
        for n,k in [(8,2),(26,2),(27,3),(12008989,229)]:self.assertEqual(self.m.config(n)['branching'],k)
    def test_original_fit_exact_tensors_centers_trie_and_replay(self):
        from vkgr import sid
        ids,x=self.toy();model,audit,gates=self.m.fit_model(ids,x,lambda:None)
        old=sid.fit(ids,x,branching=3,depth=3,seed=42,max_iter=10)
        self.assertTrue(np.array_equal(model.ids,old.ids));self.assertTrue(np.array_equal(model.sids,old.sids));self.assertEqual(model.nodes,old.nodes)
        self.assertTrue(np.array_equal(model.keys,old.keys));self.assertTrue(np.array_equal(model.key_items,old.key_items));self.assertEqual(model.bases,old.bases)
        self.assertTrue(gates['passed']);self.assertEqual(self.m.fit_model(ids,x,lambda:None)[1:],(audit,gates))
    def test_input_unit_dtype_order_and_h5_binding(self):
        ids,x=self.toy();binding={'catalog_digest':self.m.catalog_digest(ids),'vectors_sha256':self.m.au.feature_digest(x),'content_feature_sha256':'a'*64}
        self.m.check_arrays(ids,x,binding)
        for a,b in [(ids[::-1],x),(ids.astype('i8'),x),(ids,x.astype('f8')),(ids,np.zeros_like(x)),(ids,x*2)]:
            with self.assertRaises(ValueError):self.m.check_arrays(a,b,binding)
        for key in binding:
            with self.assertRaises(ValueError):self.m.check_arrays(ids,x,{**binding,key:'f'*64},expected_content='a'*64)
    def test_original_h_gates_and_integer_r1_prefix_reports(self):
        from vkgr import sid_tok_audit as au
        ids,x=self.toy();model,audit,gates=self.m.fit_model(ids,x,lambda:None)
        original=au.gates(audit,'bhkm',expected_content_sha256=au.feature_digest(x))
        self.assertEqual(gates['checks'][:len(original['checks'])],original['checks'])
        self.assertEqual([c['id'] for c in gates['checks']],['H1','H2','H3','H4','H5','H6','B1','H5a','H5b','R1'])
        self.assertFalse(self.m.mapping_gates(audit,'a'*64,'b'*64,self.m.au.feature_digest(x))['passed'])
        self.assertFalse(self.m.mapping_gates(audit,'a'*64,'a'*64,'f'*64)['passed'])
        for change in ('positions','prefix_groups'):
            bad=copy.deepcopy(audit)
            if change=='positions':bad['positions'][1]['distinct']=1
            else:bad['prefix_groups']['3']=1
            self.assertEqual(self.m.mapping_gates(bad,'a'*64,'a'*64,self.m.au.feature_digest(x))['passed'],change=='prefix_groups')
    def test_rss_guard_process_tree_boundary_and_integer_cap(self):
        m=self.m;common=type('C',(),{})();common.Probes=lambda:None
        common.process_table=lambda p:{1:{'ppid':0,'rss_kib':1},2:{'ppid':1,'rss_kib':2},3:{'ppid':0,'rss_kib':999999}}
        common.descendants=lambda table,pid:{1,2}
        with patch.object(m.os,'getpid',return_value=1):self.assertEqual(m.tree_rss(common),3072)
        self.assertIsNone(m.check_rss(60,100))
        with self.assertRaises(RuntimeError):m.check_rss(61,100)
        with self.assertRaises(RuntimeError):m.check_rss(None,100)
    def test_bundle_pins_replay_and_trie_tamper(self):
        ids,x=self.toy();parent={'synthetic_unit_fixture':True,'binding':{'content_feature_sha256':'a'*64},'input_manifest_sha256':'b'*64,'input_feature_sha256':'c'*64};model,audit,gates=self.m.fit_model(ids,x,lambda:None,content='a'*64,expected_content='a'*64)
        root=self.root/'bundle';manifest=self.m.write_bundle(root,model,audit,gates,parent,{'synthetic':True},{'synthetic':True},'a'*64)
        pin=self.m.digest(root/'manifest.json')
        codec,got=self.m.read_bundle(root/'manifest.json',pin,ids,x,parent,lambda:None)
        self.assertEqual(got,manifest);self.assertEqual(codec.catalog.nodes,[])
        original=(root/'audit.json').read_bytes();(root/'audit.json').write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'pinned bytes'):self.m.read_bundle(root/'manifest.json',pin,ids,x,parent,lambda:None)
        (root/'audit.json').write_bytes(original)
        (root/'trie.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.m.read_bundle(root/'manifest.json',pin,ids,x,parent,lambda:None)
    def request(self):
        m=self.m;d=self.root
        return {'schema':'UR001-native-full-SID-request-v1','source':{'commit':'a'*40,'files':dict(m.SCIENCE)},'lock':{'path':str(d/'lock'),'sha256':'b'*64},'producer':{'root':str(d/'full-code'),'source':'c'*40,'request':{'path':str(d/'full-request.json'),'sha256':'d'*64},'receipt':{'path':str(d/'full-output/fit/result.json'),'sha256':'e'*64},'verified_producer':{'run_id':'sgns-native-full-fit-v1',**{k:'f'*64 for k in ('receipt_sha256','run_manifest_sha256','terminal_sha256','stdout_sha256')}}},'ops_code':str(d/'ops-code'),'ops_root':str(d/'ops'),'output':str(d/'outputs'),'preflight_run_id':'ur001-native-full-sid-preflight-v1','fit_run_id':'ur001-native-full-sid-fit-v1','verify_run_id':'ur001-native-full-sid-verify-v1','resources':{'cpu_cores':1,'ram_mib':196608,'vram_mib':0,'disk_gib':16,'wall_hours':48},'engineering':False,'test_read':False}
    def test_closed_request_no_holdout_and_exclusive_output(self):
        m=self.m;v=self.request();self.assertEqual(m.validate(v),v)
        for key,value in [('engineering',True),('test_read',True),('holdout','forbidden'),('output',v['producer']['root']+'/nested'),('verify_run_id',v['fit_run_id']),('resources',{**v['resources'],'vram_mib':1})]:
            with self.subTest(key=key),self.assertRaises(ValueError):m.validate({**v,key:value})
        bad=copy.deepcopy(v);bad['producer']['receipt']['path']=str(self.root/'holdout/result.json')
        with self.assertRaisesRegex(ValueError,'holdout'):m.validate(bad)
        bad['producer']['receipt']['path']=str(self.root/'week_26.parquet')
        with self.assertRaisesRegex(ValueError,'holdout'):m.validate(bad)
    def test_three_original_fit_calls_and_zero_consumer_replay(self):
        m=self.m;v=self.request();ids,x=self.toy();path=self.root/'request.json';path.write_bytes(m.encoded(v));pin=m.digest(path)
        parent={'synthetic_unit_fixture':True,'binding':{'catalog_digest':m.catalog_digest(ids),'vectors_sha256':m.au.feature_digest(x),'content_feature_sha256':'a'*64},'input_manifest_sha256':'b'*64,'input_feature_sha256':'c'*64,'evidence':{}}
        ident={'source':v['source'],'runtime':{'synthetic':True}}
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name,val in [('identity',ident),('parent',(ids,x,parent)),('pins',[]),('job','a'*64),('load',v),('parent_snapshot',(None,None,None,{}))]:stack.enter_context(patch.object(m,name,return_value=val))
            stack.enter_context(patch.dict(os.environ,{**m.env(),'VKGR_RUN_ID':v['fit_run_id'],'VKGR_PROGRESS_PATH':str(Path(v['ops_root'])/'runs'/v['fit_run_id']/'progress.json')}))
            with patch.object(m,'accepted',return_value={'producer':{'synthetic':True}}),patch.object(m.sid,'fit',wraps=m.sid.fit) as spy:
                receipt=m.execute(v,path,pin,'fit',lambda:None);self.assertEqual(spy.call_count,2)
            fitted={'manifest':receipt['manifest'],'producer':{'synthetic_fit':True}}
            with patch.dict(os.environ,{'VKGR_RUN_ID':v['verify_run_id'],'VKGR_PROGRESS_PATH':str(Path(v['ops_root'])/'runs'/v['verify_run_id']/'progress.json')}),patch.object(m,'accepted',return_value=fitted),patch.object(m.sid,'fit',wraps=m.sid.fit) as spy:
                verification=m.execute(v,path,pin,'verify',lambda:None);self.assertEqual(spy.call_count,1);self.assertTrue(verification['independent_replay'])
            with patch.object(m.sid,'fit',side_effect=AssertionError('consumer must never fit')):
                m.read_bundle(Path(receipt['manifest']['path']),receipt['manifest']['sha256'],ids,x,parent,lambda:None)
    def test_boundary_propagation_and_rss_missing(self):
        ids,x=self.toy()
        with self.assertRaisesRegex(RuntimeError,'RSS STOP'):self.m.fit_model(ids,x,lambda:(_ for _ in ()).throw(RuntimeError('RSS STOP')))
        common=type('C',(),{'Probes':lambda self:None,'process_table':lambda self,p:None})()
        with self.assertRaises(RuntimeError):self.m.tree_rss(common)
    def test_rss_thread_and_boundary_enforce_capped_ops_planning(self):
        from contextlib import contextmanager
        m=self.m;common=type('C',(),{})();common.Probes=lambda:None;common.ram_info=lambda p:(m.PLAN_MIB+1,0)
        with self.assertRaisesRegex(RuntimeError,'planning'): 
            with m.rss_guard(common):pass
        common.ram_info=lambda p:(m.PLAN_MIB,0)
        with patch.object(m,'tree_rss',return_value=1):
            with m.rss_guard(common) as boundary:boundary()
        with patch.object(m,'tree_rss',return_value=m.PLAN_MIB*1024**2):
            with self.assertRaises(RuntimeError):
                with m.rss_guard(common):pass
    def test_real_launcher_metadata_dry_and_illegal_with_exact_deployment(self):
        import shutil
        m=self.m;base=self.root/'layout';base.mkdir()
        def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
        def clone(name,sha):
            root=base/name;subprocess.run(['git','clone','-q','--shared','--no-checkout',str(ROOT),str(root)],check=True,capture_output=True);git(root,'checkout','-q','--detach',sha);return root
        runner=clone('runner','b647105b324f8576e78187bee39ab7c3a0d0c982');ops=clone('ops','5840e6b686ec9651314a9e28a1a2c0befcc05158')
        names=['analysis/sgns_native_full_sid.py','tests/test_sgns_native_full_sid.py','docs/superpowers/reviews/2026-10-22-sgns-native-full-sid-implementation.md']
        for name in names:
            target=runner/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target);git(runner,'add',name)
        git(runner,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-q','-m','SID deployment fixture')
        v=self.request();path=base/'request.json';path.write_bytes(m.encoded(v));pin=m.digest(path)
        from analysis.a40_ops import load_ops
        _,launcher,_,_=load_ops(ops)
        cmd=[sys.executable,'-B','-m','analysis.sgns_native_full_sid'];flags=['--root',str(base/'runroot'),'--run-id','ur001-native-full-sid-preflight-fixture-v1','--kind','ops','--cwd',str(runner),'--config',str(path),'--seed','42','--input',str(path),'--cpu-cores','1','--ram-mib','196608','--vram-mib','0','--disk-gib','16','--wall-hours','48']
        for stage in ('preflight','fit','verify'):
            spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*cmd,stage,'--request',str(path),'--request-sha256',pin,'--execute-on-new-instance']));metadata=launcher.prepare(spec)
            self.assertEqual(metadata['git']['commit'],git(runner,'rev-parse','HEAD'));self.assertEqual(metadata['config']['sha256'],pin)
        env={**os.environ,'PYTHONPATH':str(runner/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
        dry=subprocess.run(cmd+['dry'],cwd=runner,env=env,text=True,capture_output=True);self.assertEqual(dry.returncode,0,dry.stderr);self.assertFalse(json.loads(dry.stdout)['formal_ready'])
        self.assertFalse((base/'runroot/queue').exists());self.assertFalse(Path(v['output']).exists())

    def test_original_adapter_h_fields_and_thresholds_are_preserved(self):
        from vkgr import sgns_subset_sid as old
        m=self.m;ids,x=self.toy();model,audit,gates=m.fit_model(ids,x,lambda:None,content='a'*64,expected_content='a'*64)
        manifest={'schema':'coll-input-v1','status':'accepted','variant':'C1','dimension':64,'engineering':True,'catalog_sha256':m.catalog_digest(ids),'vectors_sha256':m.au.feature_digest(x),'content_feature_sha256':'a'*64,'train_sequence_sha256':'b'*64,'script_sha256':'c'*64,'commit':'d'*40,'checkpoint_sha256':None,'source_files':{'science.py':'c'*64}}
        _,original=old.audit_gates(ids,model.sids,x,manifest,engineering=True)
        old_checks={c['id']:c for c in original['checks']};checks={c['id']:c for c in gates['checks']}
        for name in ('H1','H2','H3','H4','H6','H5a','H5b'):self.assertEqual(checks[name],old_checks[name])
        self.assertEqual(checks['R1']['threshold'],[3,[4,5]])
    def accepted_fixture(self):
        m=self.m;v=self.request();ids,x=self.toy();path=self.root/'request.json';path.write_bytes(m.encoded(v));pin=m.digest(path)
        parent={'binding':{'content_feature_sha256':'a'*64},'input_manifest_sha256':'b'*64,'input_feature_sha256':'c'*64,'evidence':{}}
        ident={'source':v['source'],'runtime':{'synthetic':True}};out=Path(v['output'])/'fit'
        model,audit,gates=m.fit_model(ids,x,lambda:None,content='a'*64,expected_content='a'*64)
        m.write_bundle(out,model,audit,gates,parent,ident['source'],ident['runtime'],pin)
        run=Path(v['ops_root'])/'runs'/v['fit_run_id'];(run/'logs').mkdir(parents=True);(run/'run.json').write_bytes(b'synthetic own job')
        terminal={'schema':'vkgr-terminal-v1','run_id':v['fit_run_id'],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False,'started_at':'synthetic-start','ended_at':'synthetic-end','wall_seconds':1.}
        (run/'terminal.json').write_bytes(m.encoded(terminal))
        receipt={'schema':'UR001-native-full-SID-stage-v1','status':'verified','verified':True,'engineering':False,'test_read':False,'stage':'fit','run_id':v['fit_run_id'],'run_manifest_sha256':m.digest(run/'run.json'),'request':{'path':str(path),'sha256':pin},**ident,'parent':parent,'manifest':m.record(out/'manifest.json'),'same_seed_replay':True,'fit_verify':True}
        (out/'stage-receipt.json').write_bytes(m.encoded(receipt));(run/'logs/stdout.log').write_bytes(m.encoded(receipt))
        return v,path,pin,parent,ident,ids,x,run,out
    def test_accepted_closes_own_bytes_and_never_fits(self):
        m=self.m;v,path,pin,parent,ident,ids,x,run,out=self.accepted_fixture()
        with patch.object(m,'identity',return_value=ident),patch.object(m,'pins',return_value=[]),patch.object(m,'job',return_value=m.digest(run/'run.json')),patch.object(m,'parent',return_value=(ids,x,parent)),patch.object(m.sid,'fit',side_effect=AssertionError('consume cannot fit')):
            got=m.accepted(v,path,pin,'fit',lambda:None);self.assertEqual(got['manifest'],m.record(out/'manifest.json'))
            for changed in [run/'run.json',run/'terminal.json',run/'logs/stdout.log',out/'stage-receipt.json',out/'mapping.npz',out/'trie.npz',out/'nodes.json']:
                before=changed.read_bytes()
                def drift(value):changed.write_bytes(before+b' ');return ids,x,parent
                with self.subTest(path=changed.name),patch.object(m,'parent',side_effect=drift):
                    with self.assertRaises((ValueError,json.JSONDecodeError)):m.accepted(v,path,pin,'fit',lambda:None)
                changed.write_bytes(before)
    def test_complete_original_ops_metadata_contract(self):
        m=self.m;v=self.request();path=self.root/'request.json';path.write_bytes(m.encoded(v));pin=m.digest(path);run=Path(v['ops_root'])/'runs'/v['fit_run_id'];run.mkdir(parents=True)
        inputs=[m.record(path)]
        job={'schema':'vkgr-run-v1','run_id':v['fit_run_id'],'kind':'ops','cwd':str(m.ROOT),'argv':m.argv('fit',path,pin),'seed':42,'git':{'commit':v['source']['commit'],'dirty':False,'dirty_files':[]},'config':{'path':str(path),'sha256':pin},'inputs':[{**inputs[0],'bytes':path.stat().st_size}],'env':{**m.env(),'VKGR_RUN_ID':v['fit_run_id'],'VKGR_PROGRESS_PATH':str(run/'progress.json')},'resources':m.resources(v),'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158','sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},'overlay':None,'pinned_files':{},'stop_mode':'signal'}
        with patch.object(m,'pins',return_value=inputs):
            (run/'run.json').write_bytes(m.encoded(job));m.job(v,path,pin,'fit')
            for key,val in [('seed',True),('ops_source',{}),('overlay',{}),('pinned_files',{'extra':'f'*64}),('stop_mode','other'),('unregistered',1)]:
                (run/'run.json').write_bytes(m.encoded({**job,key:val}))
                with self.subTest(key=key),self.assertRaises(ValueError):m.job(v,path,pin,'fit')

    def test_full_B_parent_context_receipt_runtime_and_evidence_closure(self):
        # Synthetic trust-boundary fixture: original full CLI scientific result is
        # mocked; this does not assert an actual full SGNS producer exists.
        from tests.test_sgns_native_full import FullTransportTests
        old=FullTransportTests();old.setUp();self.addCleanup(old.doCleanups)
        m=self.m;v=self.request();ids,x=self.toy();request=old.request();request['runner_source']=v['producer']['source'];request['lock']=v['lock']['path']
        Path(request['input_producer']['bundle']).mkdir();catalog=Path(request['input_producer']['bundle'])/'catalog.json';catalog.write_text(json.dumps(ids.tolist()))
        source={'commit':v['producer']['source'],'files':{'analysis/sgns_native_full.py':m.FULL_ENTRY_SHA}}
        code=Path(v['producer']['root']);(code/'analysis').mkdir(parents=True);(code/'analysis/sgns_native_full.py').write_bytes((ROOT/'analysis/sgns_native_full.py').read_bytes())
        oldroot=Path(request['input_producer']['root']);oldroot.mkdir();(oldroot/'old.py').write_bytes(b'old source')
        old_source={'commit':m.full.INPUT_SOURCE,'files':{'old.py':m.digest(oldroot/'old.py')}}
        lock=Path(v['lock']['path']);lock.write_bytes(b'synthetic lock');v['lock']['sha256']=m.digest(lock)
        req=Path(v['producer']['request']['path']);req.write_bytes(m.encoded(request));v['producer']['request']['sha256']=m.digest(req)
        out=Path(request['output'])/'fit';out.mkdir(parents=True);np.save(out/'vectors.npy',x)
        audit={'vectors_sha256':m.au.feature_digest(x),'catalog_sha256':m.catalog_digest(ids),'content_sha256':'a'*64};(out/'audit.json').write_bytes(m.encoded(audit))
        run=Path(request['ops_root'])/'runs'/request['fit_run_id'];(run/'logs').mkdir(parents=True)
        for name in ('terminal.json','logs/stdout.log'):(run/name).write_bytes(b'synthetic original job evidence')
        (run/'run.json').write_bytes(m.encoded({'inputs':[m.record(catalog)]}))
        receipt={'request_sha256':v['producer']['request']['sha256'],'input_binding':{'original':{'source':old_source,'manifest':{'binding':{'content_feature_sha256':'a'*64,'stage2_manifest_sha256':'d'*64}},'runtime':{'synthetic':True}}},'artifacts':{n:m.digest(out/n) for n in ('vectors.npy','audit.json')}}
        rec=out/'result.json';rec.write_bytes(m.encoded(receipt));v['producer']['receipt']=m.record(rec)
        response={'status':'verified','readonly':True,'producer':v['producer']['verified_producer']};calls=[]
        def invoked(args,**kw):
            calls.append(kw);self.assertEqual(kw['cwd'],code);self.assertIn('analysis.sgns_native_full',args);return subprocess.CompletedProcess(args,0,json.dumps(response),'')
        with patch.object(m.full,'pinned_tree',return_value=source),patch.object(m,'identity',return_value={'runtime':{'synthetic':True}}),patch.object(m.full.original,'fixed_dag',return_value={'counts':{'catalog_items':len(ids)}}),patch.object(m.subprocess,'run',side_effect=invoked):
            got_ids,got_x,parent=m.parent(v);self.assertTrue(np.array_equal(got_ids,ids));self.assertTrue(np.array_equal(got_x,x));self.assertEqual(parent['binding']['content_feature_sha256'],'a'*64)
            def drift(args,**kw):(run/'terminal.json').write_bytes(b'changed terminal during consumer');return subprocess.CompletedProcess(args,0,json.dumps(response),'')
            with patch.object(m.subprocess,'run',side_effect=drift):
                with self.assertRaisesRegex(ValueError,'changed during verify'):m.parent(v)
            with patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],2,'','rejected')):
                with self.assertRaisesRegex(ValueError,'verify rejected'):m.parent(v)
        self.assertTrue(calls)

    def test_cli_dry_and_invalid(self):
        for action,status in [('dry',0),('unknown',2)]:
            p=subprocess.run([sys.executable,'-B','-m','analysis.sgns_native_full_sid',action],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
            self.assertEqual(p.returncode,status,p.stderr)
            if not status:self.assertFalse(json.loads(p.stdout)['training'])

if __name__=='__main__':unittest.main()
