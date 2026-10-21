"""Negative-only acceptance fixtures: none represents four-gate success."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

class AcceptanceClosed(unittest.TestCase):
    def test_unfrozen_real_code_pins_refuse_before_reading_request(self):
        from analysis import sgns_subset_acceptance as s
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();p=root/'configs/full-gpu-sgns/acceptance-pins.json';p.parent.mkdir(parents=True)
            raw=json.dumps({'a1_source':None,'a1_api':None}).encode();p.write_bytes(raw)
            with patch.object(s,'ROOT',root),patch.object(s,'PINS_SHA',s.sha(raw)):
                with self.assertRaisesRegex(ValueError,'four-gate receipt code pins'):
                    s.read_request(Path('/nonexistent/request.json'),'a'*64)

    def test_frozen_retry1_a1_source_is_complete_and_rejects_other_trees(self):
        import hashlib
        from analysis import sgns_subset_acceptance as s
        pins,_=s.code_pins();source=pins['a1_source']
        self.assertEqual(source['commit'],'b7254990e8f8babbd77ed9b61c6339599616eaea')
        repo=Path(__file__).resolve().parents[1]
        def git(*args):return subprocess.check_output(['git',*args],stderr=subprocess.PIPE,text=True).strip()
        names=git('ls-tree','-r','--name-only',source['commit']).splitlines()
        expected={n:hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+n],cwd=repo)).hexdigest() for n in names}
        self.assertEqual(len(expected),158)
        self.assertEqual(source['files'],expected)
        with tempfile.TemporaryDirectory(prefix='a1-retry1-frozen-source-') as temp:
            code=Path(temp).resolve()/'CODE'
            git('clone','-q','--shared','--no-checkout',str(repo),str(code))
            git('-C',str(code),'checkout','-q','--detach',source['commit'])
            self.assertEqual(s.source_tree(code,source),source)
            with self.assertRaisesRegex(ValueError,'clean fixed'):
                s.source_tree(code,{**source,'commit':'f'*40})
            changed={**source,'files':{**source['files'],'README.md':'0'*64}}
            with self.assertRaisesRegex(ValueError,'frozen file ledger'):
                s.source_tree(code,changed)
            extra=code/'cache/unknown.py';extra.parent.mkdir(exist_ok=True);extra.write_bytes(b'unknown')
            with self.assertRaisesRegex(ValueError,'frozen file ledger'):
                s.source_tree(code,source)
            git('-C',str(code),'checkout','-q','--detach','1ca14e4971df4d40534e28b6a89d2a2d151a1c8f')
            with self.assertRaisesRegex(ValueError,'clean fixed'):
                s.source_tree(code,source)

    def test_handwritten_success_is_not_authorization(self):
        from analysis import sgns_gpu_full as full
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp).resolve()/'fake.json'
            p.write_text(json.dumps({'verified':True,'subset_four_gate_acceptance':True}))
            with self.assertRaises(ValueError):
                full.require_acceptance(p)

    def test_full_consumer_forwards_pins_and_never_accepts_adapter_rejection(self):
        from analysis import sgns_gpu_full as full
        from analysis import sgns_subset_acceptance as s
        with patch.object(s,'code_pins'), patch.object(s,'consume',side_effect=ValueError('original closure rejected')) as consume:
            with self.assertRaisesRegex(ValueError,'original closure rejected'):
                full.require_acceptance(Path('/fake/receipt'),sha256='a'*64,scratch=Path('/new/scratch'))
            consume.assert_called_once_with(Path('/fake/receipt'),'a'*64,Path('/new/scratch'))

    def test_exact_record_and_consumed_hash(self):
        from analysis import sgns_subset_acceptance as s
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp).resolve()/'record';p.write_bytes(b'not a receipt')
            for record in ({'path':str(p),'sha256':'a'*64},
                           {'path':str(p),'sha256':'a'*64,'verified':True}):
                with self.assertRaises(ValueError):s.record_bytes(record)

    def test_module_origin_rejects_foreign_shadow(self):
        from analysis import sgns_subset_acceptance as s
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            with self.assertRaisesRegex(ValueError,'origin'):
                s.module_origins({'vkgr.shadow':SimpleNamespace(__file__='/foreign/shadow.py')},root,{'files':{}})

    def test_code_tree_refuses_unknown_ignored_file(self):
        from analysis import sgns_subset_acceptance as s
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();(root/'ignored.py').write_text('raise RuntimeError("shadow")')
            with self.assertRaisesRegex(ValueError,'tree'):
                s.filesystem_tree(root,{'files':{}})

    def test_proof_rejects_bool_hits_false_health_and_wrong_collision(self):
        from analysis import sgns_subset_acceptance as s
        for proof in ({}, {'health':{'passed':True},'validation_hits':True},
                      {'health':{'passed':False},'validation_hits':378},
                      {'cpu_collision_items':1,'gpu_collision_items':2,'validation_hits':378}):
            with self.assertRaises(ValueError):s.check_proof(proof)

    def test_original_terminal_and_stdout_negatives(self):
        from analysis import sgns_subset_acceptance as s
        # These are transport-only failed fixtures, never a four-gate receipt.
        run={'schema':'vkgr-run-v1','run_id':'fixture','kind':'ops','seed':42,
             'git':{'dirty':False,'dirty_files':[]}}
        terminal={'schema':'vkgr-terminal-v1','run_id':'fixture','status':'succeeded','exit_code':0,
                  'signal':None,'wall_cap_exceeded':False,'started_at':'x','ended_at':'y','wall_seconds':1}
        receipt={'run_id':'fixture','verified':False,'schema':'transport-only-failed-fixture'}
        cases=[('signal',15),('wall_cap_exceeded',True),('exit_code',False),('status','failed'),
               ('wall_seconds',float('nan')),('run_id','wrong')]
        for key,val in cases:
            changed={**terminal,key:val}
            raw={'run':json.dumps(run).encode(),'terminal':json.dumps(changed).encode(),
                 'stdout':json.dumps(receipt).encode(),'receipt':json.dumps(receipt).encode()}
            with self.subTest(key=key),self.assertRaises(ValueError):
                s.producer_bytes(raw,run_path=Path('/runs/fixture/run.json'))
        raw={'run':json.dumps(run).encode(),'terminal':json.dumps(terminal).encode(),
             'stdout':b'{}','receipt':json.dumps(receipt).encode()}
        with self.assertRaises(ValueError):s.producer_bytes(raw,run_path=Path('/runs/fixture/run.json'))
        with self.assertRaises(ValueError):s.producer_bytes(raw,run_path=Path('/runs/wrong/run.json'))

    def test_isolated_adapter_refuses_incomplete_request_before_scratch(self):
        from analysis import sgns_subset_acceptance as s
        with tempfile.TemporaryDirectory() as temp:
            scratch=Path(temp).resolve()/'new-scratch';request=Path(temp).resolve()/'fake-request.json';request.write_bytes(b'{}')
            run=subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve().parents[1]/'analysis/sgns_subset_acceptance_a1.py'),
                '--request',str(request),'--request-sha256',s.sha(b'{}'),'--scratch',str(scratch)],capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('complete fixed four-gate request',run.stderr)
            self.assertFalse(scratch.exists())

    def test_retry1_input_producer_complete_transport_closure(self):
        self._input_producer_transport_closure('sgns-gpu-subset-a1-inputs-retry1-v1',True)

    def test_original_input_id_is_not_a_retry1_producer(self):
        self._input_producer_transport_closure('sgns-gpu-subset-a1-inputs-v1',False)

    def test_unregistered_input_id_is_not_a_retry1_producer(self):
        self._input_producer_transport_closure('sgns-gpu-subset-a1-inputs-retry2-v1',False)

    def _input_producer_transport_closure(self,rid,accepted_id):
        # Transport-only synthetic source/artifacts; never a four-gate receipt.
        import importlib.util
        from analysis import sgns_subset_acceptance as s
        spec=importlib.util.spec_from_file_location('input_adapter_test',Path(__file__).resolve().parents[1]/'analysis/sgns_subset_acceptance_a1.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();code=root/'CODE';code.mkdir();output=root/'family';output.mkdir()
            ops=root/'OPS';job=ops/'runs'/rid;(job/'logs').mkdir(parents=True)
            def write(name,value):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value if type(value) is bytes else s.encoded(value))
                return {'path':str(p),'sha256':s.sha(p.read_bytes())}
            source={'commit':'b'*40,'files':{}};env={'transport_only':True}
            lock=write('lock',b'lock');content=write('content',b'content');ur001=write('ur001',b'{}');history=write('family/history',b'{}')
            sid_sha='d'*64
            family={'source':source,'runtime':env,'history':history,'content':content,'ur001':ur001,
                    'raw_data':str(root/'RAW'),'gpu_sid_manifest_sha256':sid_sha}
            prepared=write('family/manifest.json',family)
            request=write('request.json',{'schema':'gpu-sgns-subset-a1-input-request-v1','source_commit':source['commit'],
                'lock':lock,'raw_data':family['raw_data'],'ur001':ur001,'content':content,'gpu_sid_manifest_sha256':sid_sha,
                'output_root':str(output),'ops_root':str(ops)})
            argv=[sys.executable,'-B','-m','analysis.sgns_subset_a1_inputs','derive','--request',request['path'],'--request-sha256',request['sha256']]
            declared=[request,lock,ur001,content]
            run={'schema':'vkgr-run-v1','run_id':rid,'kind':'ops','cwd':str(code),'argv':argv,'seed':42,
                'git':{'commit':source['commit'],'dirty':False,'dirty_files':[]},'config':request,
                'inputs':[{**r,'bytes':len(Path(r['path']).read_bytes())} for r in declared],
                'env':{'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(code/'src'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                    'VKGR_RUN_ID':rid,'VKGR_PROGRESS_PATH':str(job/'progress.json')},
                'resources':{'cpu_cores':1,'ram_mib':98304,'vram_mib':0,'disk_gib':4,'wall_seconds':7200.0},
                'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158','sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},
                'overlay':None,'pinned_files':{},'stop_mode':'signal'}
            terminal={'schema':'vkgr-terminal-v1','run_id':rid,'status':'succeeded','exit_code':0,'signal':None,
                'wall_cap_exceeded':False,'started_at':1,'ended_at':2,'wall_seconds':1}
            def fixture(r,t):
                runrec=write('OPS/runs/'+rid+'/run.json',r)
                receipt={'schema':'gpu-sgns-subset-a1-input-stage-v1','authority':'ad1c428','status':'verified','verified':True,
                    'run_id':rid,'run_manifest_sha256':runrec['sha256'],'request_sha256':request['sha256'],'source':source,'runtime':env,
                    'output':{**prepared,'history':history,'subset_four_gate_acceptance':False},'science_training':False,'test_read':False,
                    'subset_four_gate_acceptance':False}
                records={'run':runrec,'terminal':write('OPS/runs/'+rid+'/terminal.json',t),
                    'stdout':write('OPS/runs/'+rid+'/logs/stdout.log',receipt),'receipt':write('family/stage-receipt.json',receipt)}
                accepted={'source_commit':source['commit'],'data_root':str(output),'output_root':str(root/'A1_OUTPUT'),
                    'scratch_root':str(root/'A1_SCRATCH'),'input_producer':records,'prepared':prepared,'history':history,'content':content,'lock':lock,
                    'gpu_sid':{'manifest':{'sha256':sid_sha}}}
                return accepted,receipt
            accepted,receipt=fixture(run,terminal)
            if not accepted_id:
                with self.assertRaisesRegex(ValueError,'original input producer derive argv differs'):
                    adapter.check_input_producer(accepted,source,env,code)
                return
            checked=adapter.check_input_producer(accepted,source,env,code)
            self.assertEqual(checked['receipt'],receipt)
            good_records={k:write('GOOD/'+rid+'/'+k,Path(rec['path']).read_bytes()) for k,rec in accepted['input_producer'].items()}
            # Generic checker binds a run record's parent to its run_id.
            # Keep the inert original verify fixture in that exact layout.
            good_records['run']=write('GOOD/'+rid+'/run.json',Path(accepted['input_producer']['run']['path']).read_bytes())
            attacks=[('wall_cap',run,{**terminal,'wall_cap_exceeded':True}),
                ('bool_exit',run,{**terminal,'exit_code':False}),('foreign_cwd',{**run,'cwd':str(root/'foreign')},terminal),
                ('missing_fields',{k:v for k,v in run.items() if k not in ('cwd','env','resources','ops_source')},terminal),
                ('empty_inputs',{**run,'inputs':[]},terminal),('empty_env',{**run,'env':{}},terminal),
                ('wrong_resource',{**run,'resources':{**run['resources'],'ram_mib':1}},terminal),
                ('wrong_ops',{**run,'ops_source':{}},terminal)]
            for name,r,t in attacks:
                accepted,_=fixture(r,t)
                with self.subTest(attack=name),self.assertRaises(ValueError):
                    adapter.check_input_producer(accepted,source,env,code)

            # Exercise the actual adapter call path before any scientific
            # context/rows, so dropping the new hook is observable too.
            from types import SimpleNamespace
            accepted,_=fixture(run,{**terminal,'wall_cap_exceeded':True})
            value={'a1_code_root':str(code),'a1_request':request,'a1_verify':good_records,'lock':lock}
            fake_a1=SimpleNamespace(read_request=lambda *a:accepted,identity=lambda *a:(source,env))
            with patch.object(adapter.s,'code_pins',return_value=({'a1_api':adapter.API,'a1_source':source},True)), \
                 patch.object(adapter.s,'read_request',return_value=value), \
                 patch.object(adapter,'bootstrap',return_value=(fake_a1,None,None,None,None)), \
                 patch.object(adapter.Path,'cwd',return_value=code):
                with self.assertRaisesRegex(ValueError,'original producer terminal/stdout'):
                    adapter.main(['--request',request['path'],'--request-sha256',request['sha256'],
                        '--scratch',str(root/'NEW_SCRATCH')])
            self.assertFalse((root/'NEW_SCRATCH').exists())

    def test_subprocess_dry_and_invalid(self):
        run=subprocess.run([sys.executable,'-B','-m','analysis.sgns_subset_acceptance','dry'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertTrue(json.loads(run.stdout)['code_pins_ready'])
        bad=subprocess.run([sys.executable,'-B','-m','analysis.sgns_subset_acceptance','invalid'],capture_output=True)
        self.assertNotEqual(bad.returncode,0)

if __name__=='__main__':unittest.main()
