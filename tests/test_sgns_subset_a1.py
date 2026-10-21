import ast
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL='b5a4d1c99ba2d8361a2d90c23bac27af5b71769d'

def old(path):
    return subprocess.check_output(['git','-C',str(ROOT),'show',ORIGINAL+':'+path])

class A1IdentityTests(unittest.TestCase):
    def test_original_native_source_and_config(self):
        from vkgr import sgns_subset_a1 as worker
        self.assertGreater(len(worker.NATIVE_FILES),40)
        for path,pin in worker.NATIVE_FILES.items():
            self.assertEqual(hashlib.sha256(old(path)).hexdigest(),pin)
            self.assertEqual((ROOT/path).read_bytes(),old(path),path)
        self.assertEqual(worker.registered_config('1'*64)['seed'],42)
        original=json.loads(old('configs/sid_structure/g-structure-a1-s42.json'))
        actual=worker.registered_config('1'*64)
        self.assertEqual({k:v for k,v in actual.items() if k not in ('sid','sid_manifest_sha256')},
                         {k:v for k,v in original.items() if k not in ('sid','sid_manifest_sha256')})
    def test_config_sid_only(self):
        from vkgr import sgns_subset_a1 as worker
        for value in ('',None,'0'*64,'G'*64):
            with self.assertRaises(ValueError):worker.registered_config(value)
    def test_fail_closed_without_native_inputs(self):
        from vkgr import sgns_subset_a1 as worker
        with self.assertRaisesRegex(ValueError,'complete accepted'):
            worker.accepted_request({},'1'*64)
    def test_same_original_training_and_verify_code(self):
        from vkgr import sgns_subset_a1 as worker
        from vkgr import sid_structure_a1_run as original
        engine=worker.native_functions(lambda *a,**k:None,lambda *a,**k:None,lambda *a,**k:None)
        for name in ('run','verify','_contexts','_binding'):
            self.assertIs(engine[name].__code__,getattr(original,name).__code__)
        self.assertIs(engine['run'].__globals__['_contexts'],engine['_contexts'])
        self.assertIs(engine['_contexts'].__globals__['check_pins'],engine['check_pins'])
    def test_cli_dry_and_illegal(self):
        import os,sys
        env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
        p=subprocess.run([sys.executable,'-B','-m','analysis.sgns_subset_a1','dry'],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr); d=json.loads(p.stdout)
        self.assertFalse(d['training']);self.assertEqual(d['seed'],42);self.assertEqual(d['hits_gate'],378)
        p=subprocess.run([sys.executable,'-B','-m','analysis.sgns_subset_a1','illegal'],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)

class A1BindingTests(unittest.TestCase):
    def fixture(self,base,device='cpu'):
        from vkgr import sgns_subset_a1 as core
        def write(path,value):
            raw=value if isinstance(value,bytes) else json.dumps(value,sort_keys=True).encode()
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
        run_id='sgns-gpu-subset-'+device+'-fit-v1';run_root=base/'OPS/runs'/run_id;out=base/device
        vec=write(out/'vectors.npy',b'vectors');audit=write(out/'audit.json',b'{}')
        im=write(base/'input/manifest.json',b'{}');lock=write(base/'lock',b'lock')
        source={'commit':'a'*40,'files':{}}
        run={'run_id':run_id,'kind':'ops','seed':42,'git':{'commit':'a'*40,'dirty':False,'dirty_files':[]},
             'argv':['python','-B','-m','analysis.sgns_gpu_subset','fit','--bundle',str(base/'input'),'--bundle-sha',im['sha256'],
                     '--device',device,'--lock',lock['path'],'--output',str(out)],'inputs':[im,lock]}
        receipt={'schema':'gpu-sgns-subset-stage-v1','stage':'sgns','run_id':run_id,'status':'verified','verified':True,
                 'authority':'ad1c428','engineering':False,'device':device,'subset_four_gate_acceptance':False,'source':source,
                 'validation_read':False,'test_read':False,'artifacts':{'vectors.npy':vec['sha256'],'audit.json':audit['sha256']},
                 'input_manifest_sha256':im['sha256'],'original_audit_verification':{'schema':'coll-input-verify-v1','status':'verified'}}
        record={'run':write(run_root/'run.json',run),'terminal':write(run_root/'terminal.json',{'run_id':run_id,'status':'succeeded','exit_code':0,'signal':None}),
                'stdout':write(run_root/'logs/stdout.log',receipt),'receipt':write(out/'stage-receipt.json',receipt)}
        return record,run,receipt,write
    def test_successful_original_sgns_closure_and_refusals(self):
        import tempfile,copy
        from vkgr import sgns_subset_a1 as core
        with tempfile.TemporaryDirectory() as td:
            record,run,receipt,write=self.fixture(Path(td).resolve())
            self.assertEqual(core.producer(record,'sgns','cpu'),receipt)
            corrupt=copy.deepcopy(run);corrupt['argv'][10]='cuda'
            record['run']=write(Path(record['run']['path']),corrupt)
            with self.assertRaisesRegex(ValueError,'device|argv'):
                core.producer(record,'sgns','cpu')
    def test_validation_integer_hits_and_row_identity(self):
        from vkgr import sgns_subset_a1 as core
        from vkgr.data_audit import canonical_bytes
        q=[{'target':7},{'target':8}]
        rows=[{'ordinal':j,'query_sha256':hashlib.sha256(canonical_bytes(query)).hexdigest(),'items':[7,9]} for j,query in enumerate(q)]
        aggregate={'metrics':{'all_eligible':{'count':2,'recall@50':.5}}}
        self.assertEqual(core.validation_hits(rows,q,aggregate),1)
        rows[1]['ordinal']=0
        with self.assertRaisesRegex(ValueError,'identity'):core.validation_hits(rows,q,aggregate)
    def test_terminal_and_receipt_output_binding(self):
        import tempfile,copy
        from vkgr import sgns_subset_a1 as core
        with tempfile.TemporaryDirectory() as td:
            record,run,receipt,write=self.fixture(Path(td).resolve())
            terminal={'run_id':run['run_id'],'status':'succeeded','exit_code':0,'signal':9}
            record['terminal']=write(Path(record['terminal']['path']),terminal)
            with self.assertRaisesRegex(ValueError,'binding'):core.producer(record,'sgns','cpu')
            terminal['signal']=None;record['terminal']=write(Path(record['terminal']['path']),terminal)
            write(Path(record['receipt']['path']).parent/'vectors.npy',b'changed')
            with self.assertRaisesRegex(ValueError,'bytes'):core.producer(record,'sgns','cpu')
    def test_delta_does_not_accept_extra_or_existing_paths(self):
        from vkgr import sgns_subset_a1 as core
        delta={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in core.A1_ADDITIONS if (ROOT/p).exists()}
        # Complete fixture path is added before this test runs in final checkout.
        self.assertGreater(len(delta),20)
        with self.assertRaisesRegex(ValueError,'delta'):core.check_a1_delta({}, {},{})
        delta['unexpected.py']='1'*64
        with self.assertRaisesRegex(ValueError,'delta'):core.check_a1_delta({}, {},delta)
    def test_exact_source_delta_and_actual_new_bytes(self):
        from vkgr import sgns_subset_a1 as core
        import copy
        delta={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in core.A1_ADDITIONS}
        old={'files':{'existing.py':'a'*64}}
        actual={'files':{**old['files'],**delta}}
        self.assertEqual(core.check_a1_delta(actual,old,delta),delta)
        for kind in ('extra','missing','old-byte-change','new-byte-change'):
            broken=copy.deepcopy(actual)
            if kind=='extra':broken['files']['unregistered.py']='b'*64
            elif kind=='missing':broken['files'].pop(core.A1_ADDITIONS[0])
            elif kind=='old-byte-change':broken['files']['existing.py']='c'*64
            else:broken['files'][core.A1_ADDITIONS[0]]='d'*64
            with self.assertRaisesRegex(ValueError,'source'):core.check_a1_delta(broken,old,delta)
    def test_mapped_original_lock_is_explicit(self):
        from vkgr import sgns_subset_a1 as core
        self.assertEqual(hashlib.sha256(old('uv.lock')).hexdigest(),core.NATIVE_LOCK_SHA)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path=Path(td).resolve()/'original-anchor.lock';path.write_bytes(old('uv.lock'))
            self.assertEqual(core.file_bytes({'path':str(path),'sha256':core.NATIVE_LOCK_SHA}),old('uv.lock'))
        self.assertNotIn('uv.lock',core.NATIVE_FILES)
    def test_fixed_delta_set_cannot_shrink_with_matching_actual(self):
        from vkgr import sgns_subset_a1 as core
        delta={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in core.A1_ADDITIONS}
        delta.pop(core.A1_ADDITIONS[0])
        old={'files':{'existing.py':'a'*64}}
        with self.assertRaisesRegex(ValueError,'exact fixed A1 source delta'):
            core.check_a1_delta({'files':{**old['files'],**delta}},old,delta)
    def test_measured_followup_resource_declaration(self):
        from analysis import sgns_subset_a1 as cli
        self.assertEqual(cli.measured_resources({'context_inclusive_peak_mib':1000,'context_samples':2,'rss_tree_upper_bound_mib':1000}),
                         {'vram_mib':1280,'ram_mib':1280})
        for record in ({},{'context_inclusive_peak_mib':0,'context_samples':2,'rss_tree_upper_bound_mib':1000},
                       {'context_inclusive_peak_mib':1000,'context_samples':0,'rss_tree_upper_bound_mib':1000}):
            with self.assertRaises(ValueError):cli.measured_resources(record)
    def test_sid_verified_gates_equal_consumed_manifest(self):
        import tempfile,copy
        from unittest.mock import patch
        from vkgr import sgns_subset_a1 as core
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve();value={'cpu_sgns':{'receipt':{'sha256':'1'*64}},'gpu_sgns':{'receipt':{'sha256':'2'*64}}}
            sgns={'source':{},'runtime':{},'input_manifest_sha256':'3'*64,'input_producer':{}}
            receipts=[]
            for role,sha in (('cpu_sid','1'*64),('gpu_sid','2'*64)):
                manifest={'gates':{'passed':True},'collision_gate':None if role=='cpu_sid' else {'passed':True}}
                path=base/(role+'.json');raw=json.dumps(manifest).encode();path.write_bytes(raw)
                input_path=base/(role+'-input.json');input_path.write_bytes(b'{}')
                value[role]={'producer':{},'manifest':{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()},
                             'input_manifest':{'path':str(input_path),'sha256':hashlib.sha256(b'{}').hexdigest()}}
                receipts.append({'gates':manifest['gates'],'collision_gate':manifest['collision_gate'],
                                 'manifest_sha256':value[role]['manifest']['sha256'],'input_manifest_sha256':value[role]['input_manifest']['sha256'],
                                 'sgns_stage_receipt_sha256':sha})
            with patch.object(core,'producer',side_effect=[sgns,sgns,*receipts]):self.assertIn('gpu_sid',core.closure(value))
            bad=copy.deepcopy(receipts);bad[1]['collision_gate']['extra']=True
            with patch.object(core,'producer',side_effect=[sgns,sgns,*bad]):
                with self.assertRaisesRegex(ValueError,'gates differ'):core.closure(value)

class A1PredecessorAttackTests(unittest.TestCase):
    def fixture(self,base):
        from analysis import sgns_subset_a1 as cli
        def put(path,value):
            raw=json.dumps(value,sort_keys=True).encode();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
        fixed=put(base/'fixed.json',{'fixture':'metadata only'})
        producer={key:fixed for key in ('run','terminal','stdout','receipt')}
        source={'commit':'a'*40,'files':{'fixed.py':'b'*64}}
        value={'source_commit':'a'*40,'ops_root':str(base/'ops'),'output_root':str(base/'out'),'prepared':fixed,'history':fixed,'content':fixed,'lock':fixed,
               'input_producer':producer,'cpu_sgns':producer,'gpu_sgns':producer,
               'cpu_sid':{'producer':producer,'manifest':fixed,'input_manifest':fixed},'gpu_sid':{'producer':producer,'manifest':fixed,'input_manifest':fixed}}
        request=put(base/'request.json',value);rid='sgns-gpu-subset-a1-preflight-retry1-v1'
        run={'run_id':rid,'kind':'ops','cwd':str(ROOT),'argv':cli.invocation('preflight',Path(request['path']),request['sha256']),
             'seed':42,'git':{'commit':'a'*40,'dirty':False,'dirty_files':[]},'config':request,
             'inputs':cli.stage_inputs(value,Path(request['path']),request['sha256'])}
        def save(run):
            r=put(base/'ops/runs'/rid/'run.json',run)
            receipt={'source':source,'request_sha256':request['sha256'],'stage':'preflight','status':'ready','run_id':rid,'run_manifest_sha256':r['sha256']}
            put(base/'ops/runs'/rid/'terminal.json',{'run_id':rid,'status':'succeeded','exit_code':0,'signal':None})
            put(base/'ops/runs'/rid/'logs/stdout.log',receipt);put(base/'out/preflight/stage-receipt.json',receipt)
        return value,request,source,run,save
    def test_synchronized_run_hash_and_stdout_do_not_allow_mutated_predecessor_job(self):
        import tempfile,copy,inspect
        from analysis import sgns_subset_a1 as cli
        with tempfile.TemporaryDirectory(prefix='a1-independent-predecessor-') as td:
            value,request,source,run,save=self.fixture(Path(td).resolve())
            bads={'kind':'train','seed':2025,'argv':['python','wrong-module'],'cwd':'/tmp/wrong',
                  'git':{'commit':'a'*40,'dirty':True,'dirty_files':['changed.py']},'config':{},'inputs':[]}
            for field,bad in bads.items():
                with self.subTest(field=field):
                    changed=copy.deepcopy(run);changed[field]=bad;save(changed)
                    options={'request':Path(request['path'])} if 'request' in inspect.signature(cli.predecessor).parameters else {}
                    with self.assertRaisesRegex(ValueError,'job/source/argv/input binding'):
                        cli.predecessor(value,'train',request['sha256'],source,**options)

class A1ReceiptSemanticsTests(unittest.TestCase):
    def test_receipt_schema_runtime_config_proof_and_false_flags_are_exact(self):
        import copy
        from analysis import sgns_subset_a1 as cli
        expected={'prior':'preflight','source':{'commit':'a'*40},'env':{'lock':'b'*64,'flags':False},'config':{'seed':42,'dropout':0.0},
                  'proof':{'producer':{'verified':True}},'request_sha':'c'*64,'run_sha':'d'*64}
        receipt={'schema':'gpu-sgns-subset-a1-stage-v1','authority':'ad1c428','engineering':False,'stage':'preflight',
                 'run_id':'sgns-gpu-subset-a1-preflight-retry1-v1','run_manifest_sha256':'d'*64,'request_sha256':'c'*64,
                 'source':expected['source'],'runtime':expected['env'],'config':expected['config'],'input_producers':expected['proof'],
                 'test_read':False,'subset_four_gate_acceptance':False,'status':'ready',
                 'context_inclusive_peak_mib':512,'context_samples':2,'rss_tree_upper_bound_mib':1024,'resource_scope':cli.RESOURCE_SCOPE,
                 'query_count':1,'catalog_size':3,'native_runtime':{},'payload_written':False,'scratch_written':True}
        self.assertEqual(cli.receipt_binding(receipt,**expected),receipt)
        for field,bad in [('schema','other'),('authority','other'),('engineering',True),('runtime',{'lock':'b'*64,'flags':0}),
                          ('config',{'seed':42,'dropout':False}),('input_producers',{'producer':{'verified':1}}),('status','complete'),('test_read',True),('resource_scope','torch_counter_excludes_context')]:
            with self.subTest(field=field):
                wrong=copy.deepcopy(receipt);wrong[field]=bad
                with self.assertRaisesRegex(ValueError,'receipt semantics'):
                    cli.receipt_binding(wrong,**expected)
    def test_current_env_run_id_is_not_used_for_prior_job(self):
        import tempfile,os
        from unittest.mock import patch
        from analysis import sgns_subset_a1 as cli
        with tempfile.TemporaryDirectory() as td:
            value,request,source,run,save=A1PredecessorAttackTests().fixture(Path(td).resolve());save(run)
            with patch.dict(os.environ,{'VKGR_RUN_ID':'sgns-gpu-subset-a1-train-v1'}):
                self.assertEqual(len(cli.job_binding(value,Path(request['path']),request['sha256'],'preflight')),64)
                with self.assertRaisesRegex(ValueError,'own fixed'):
                    cli.own_job(value,Path(request['path']),request['sha256'],'preflight')

class A1NativeOutputMetadataTests(unittest.TestCase):
    def test_original_native_manifest_and_consumed_artifact_identity(self):
        import tempfile,copy
        from analysis import sgns_subset_a1 as cli
        from vkgr import sgns_subset_a1 as core
        original=core.check_native();config=core.registered_config('1'*64);source={'fixture':'metadata-not-science'};runtime={'device':'fixture'}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();out=root/'train/native';out.mkdir(parents=True)
            def blob(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return hashlib.sha256(data).hexdigest()
            best=blob(out/'best.pt',b'fixture checkpoint, never loaded');last=blob(out/'last.pt',b'fixture checkpoint, never loaded')
            ev=blob(out/'evaluation-001/manifest.json',b'{"fixture":"metadata"}')
            value={'output_root':str(root),'data_root':str(root/'raw'),'prepared':{'sha256':'2'*64},'history':{'sha256':'3'*64},
                   'gpu_sid':{'manifest':{'sha256':'1'*64},'input_manifest':{'sha256':'4'*64}}}
            binding={'task':original.TASK_ID,'budget':original.BUDGET,'init_version':'INIT_EMB_N002','config':config,'config_sha256':original._digest(config),
                     'source':source,'data_root':value['data_root'],'runtime':runtime,'prepared_manifest_sha256':'2'*64,'history_manifest_sha256':'3'*64,
                     'synthetic':False,'test_gate':'closed','sid':{'manifest_sha256':'1'*64,'input_manifest_sha256':'4'*64},
                     'context_b1':{'history_manifest_sha256':'3'*64,'sid_version':core.SID_VERSION}}
            native={'schema':original.MANIFEST_SCHEMA,'kind':original.MANIFEST_KIND,'status':'complete','config':config,'binding':binding,'selection':{},
                    'epochs':[{'epoch':1,'evaluation':{'path':'evaluation-001','manifest_sha256':ev}}],'losses':[],'best':{'path':'best.pt','sha256':best,'epoch':1},
                    'last':{'path':'last.pt','sha256':last,'epoch':1},'wall_seconds':{},'hit_max_epochs':False,'eval_workers':1,'decode':None}
            def put(n):return {'native_manifest_sha256':blob(out/'manifest.json',json.dumps(n,sort_keys=True).encode())}
            engine={'_source':lambda r:source}
            self.assertEqual(cli.train_output_binding(value,put(native),engine,config,runtime),native)
            for key,bad in [('eval_workers',2),('decode','approx'),('config',{}),('last',{'path':'else.pt','sha256':last,'epoch':1})]:
                changed=copy.deepcopy(native);changed[key]=bad
                with self.assertRaises(ValueError):cli.train_output_binding(value,put(changed),engine,config,runtime)
            changed=copy.deepcopy(native);changed['binding']['history_manifest_sha256']='5'*64
            with self.assertRaisesRegex(ValueError,'semantic binding'):cli.train_output_binding(value,put(changed),engine,config,runtime)
            receipt=put(native);blob(out/'evaluation-001/manifest.json',b'changed')
            with self.assertRaisesRegex(ValueError,'consumed artifact'):cli.train_output_binding(value,receipt,engine,config,runtime)

class A1IntegratedRealFunctionsTests(unittest.TestCase):
    @unittest.skipUnless(__import__('os').environ.get('VKGR_A1_INTEGRATED')=='1','requires final c9a60c5 + SID4 + exact A1 30-path tree')
    def test_real_engine_and_native_context_reject_synthetic_before_formal_content(self):
        """No function stubs. Metadata fixture is never a scientific acceptance."""
        import tempfile,copy
        import numpy as np
        from vkgr import sgns_subset_a1 as core,sgns_subset_a1_inputs as inputs,sgns_subset_sid as sid_module
        from analysis.ur001_derive_cli import source_identity
        from analysis.new_instance_prepare import runtime
        from analysis.ur001_derivation import derive
        actual=source_identity();delta={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in core.A1_ADDITIONS}
        self.assertEqual(len(delta),30)
        sidfiles={p:s for p,s in actual['files'].items() if p not in delta}
        sidpaths={'src/vkgr/sgns_subset_sid.py','analysis/sgns_subset_sid.py','tests/test_sgns_subset_sid.py','tests/fixtures/sgns_subset_sid_launcher_chain.py'}
        self.assertTrue(sidpaths<=set(sidfiles))
        sgns_source={'commit':'b'*40,'files':{p:s for p,s in sidfiles.items() if p not in sidpaths}}
        sid_source={'commit':'c'*40,'files':sidfiles}
        with tempfile.TemporaryDirectory(prefix='a1-no-stub-integrated-') as td:
            base=Path(td).resolve();(base/'raw').mkdir()
            frozen_lock=base/'original-anchor.lock';frozen_lock.write_bytes(old('uv.lock'))
            cutoff='native H5a CONTENT_V1 mismatch'
            try:env=runtime(frozen_lock)
            except ValueError as error:
                self.assertIn('original runtime identity differs',str(error))
                env={'fixture':'synthetic metadata only; original runtime mismatch must reject'}
                cutoff='original strict runtime identity mismatch'
            def put(path,value):
                raw=value if isinstance(value,bytes) else json.dumps(value,sort_keys=True).encode()
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
                return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
            producers={}
            for device in ('cpu','cuda'):
                record,run,receipt,_=A1BindingTests().fixture(base,device)
                run['git']['commit']=sgns_source['commit'];receipt['source']=sgns_source;receipt['runtime']=env
                receipt['fixture']='synthetic metadata only; not actual successful scientific producer'
                record['run']=put(Path(record['run']['path']),run)
                record['receipt']=put(Path(record['receipt']['path']),receipt);record['stdout']=put(Path(record['stdout']['path']),receipt)
                producers[device]=record
            sids={}
            for device in ('cpu','cuda'):
                im=put(base/device/'sid/input-manifest.json',{})
                manifest=put(base/device/'sid/run_manifest.json',{'gates':{'passed':True},'collision_gate':{'passed':True},'fixture':'synthetic metadata only'})
                rid='sgns-gpu-subset-'+device+'-sid-verify-v1';rroot=base/'OPS/runs'/rid
                run={'run_id':rid,'kind':'ops','seed':42,'git':{'commit':sid_source['commit'],'dirty':False,'dirty_files':[]},
                     'argv':['python','-B','-m','analysis.sgns_subset_sid','verify']}
                r=put(rroot/'run.json',run)
                receipt={'schema':'gpu-sgns-subset-sid-stage-v1','authority':'ad1c428','status':'verified','verified':True,
                         'run_id':rid,'run_manifest_sha256':r['sha256'],'stage':'sid-verify','engineering':False,'device':device,
                         'source':sid_source,'runtime':env,'mapping_replay':True,'gates':{'passed':True},'collision_gate':{'passed':True},
                         'manifest_sha256':manifest['sha256'],'input_manifest_sha256':im['sha256'],
                         'sgns_stage_receipt_sha256':producers[device]['receipt']['sha256'],'subset_four_gate_acceptance':False,
                         'fixture':'synthetic metadata only; not actual SID scientific producer'}
                sids[device]={'manifest':manifest,'input_manifest':im,'producer':{'run':r,
                     'terminal':put(rroot/'terminal.json',{'run_id':rid,'status':'succeeded','exit_code':0,'signal':None}),
                     'stdout':put(rroot/'logs/stdout.log',receipt),'receipt':put(base/device/'sid/stage-receipt.json',receipt)}}
            parts=[(i,[{'user_id':1,'item_id':j,'like':True} for j in (3,4,5)] if i==0 else []) for i in range(25)]
            val=(25,[{'user_id':1,'item_id':3,'like':True}]);train=[];ur=derive(parts,val,train.append)
            payload=inputs.native_payloads(parts,val,ur001_train=train,ur001_validation=ur['validation'],ur001_slices=ur['slices'])
            cpath=base/'content.npy';np.save(cpath,np.zeros((4,64),dtype=np.float32))
            content={'path':str(cpath),'sha256':hashlib.sha256(cpath.read_bytes()).hexdigest()}
            prepared=inputs.write_family(base/'family',payload,parts,raw_data=str(base/'raw'),raw_records=[{'path':'train/week_00.parquet','size':1,'sha256':'e'*64}],
                 ur001={'path':str(base/'UR001-manifest.json'),'sha256':inputs.SUBSET_MANIFEST_SHA},content=content,
                 sid_sha=sids['cuda']['manifest']['sha256'],source=actual,runtime=env)
            rid='sgns-gpu-subset-a1-inputs-retry1-v1';request=put(base/'input-request.json',{'fixture':'synthetic only'})
            run={'run_id':rid,'kind':'ops','seed':42,'git':{'commit':actual['commit'],'dirty':False,'dirty_files':[]},'config':request,'inputs':[request],
                 'argv':['python','-B','-m','analysis.sgns_subset_a1_inputs','derive','--request',request['path'],'--request-sha256',request['sha256']]}
            r=put(base/'OPS/runs'/rid/'run.json',run)
            receipt={'schema':'gpu-sgns-subset-a1-input-stage-v1','authority':'ad1c428','status':'verified','verified':True,'run_id':rid,
                     'run_manifest_sha256':r['sha256'],'request_sha256':request['sha256'],'source':actual,'runtime':env,'output':prepared,
                     'science_training':False,'test_read':False,'subset_four_gate_acceptance':False,'fixture':'synthetic metadata only'}
            ip={'run':r,'terminal':put(base/'OPS/runs'/rid/'terminal.json',{'run_id':rid,'status':'succeeded','exit_code':0,'signal':None}),
                'stdout':put(base/'OPS/runs'/rid/'logs/stdout.log',receipt),'receipt':put(base/'family/stage-receipt.json',receipt)}
            value={'source_commit':actual['commit'],'lock':{'path':str(frozen_lock),'sha256':core.NATIVE_LOCK_SHA},
                   'prepared':{k:prepared[k] for k in ('path','sha256')},'history':prepared['history'],'content':content,'data_root':str(base/'raw'),
                   'input_producer':ip,'cpu_sgns':producers['cpu'],'gpu_sgns':producers['cuda'],'cpu_sid':sids['cpu'],'gpu_sid':sids['cuda'],'producer_delta':delta}
            engine,common,extra,proof=core.engine_for(value,'f'*64,actual)
            # Real original source/binding/output metadata paths, no function stubs.
            native_source=engine['_source'](ROOT)
            original=core.check_native()
            self.assertEqual(set(native_source['files']),set(original.G6_MEMBERS))
            self.assertEqual(native_source['files']['uv.lock'],core.NATIVE_LOCK_SHA)
            self.assertEqual(native_source['source_sha256'],original._digest(native_source['files']))
            self.assertFalse((ROOT/'uv.lock').exists())
            from types import SimpleNamespace
            cfg=core.registered_config(sids['cuda']['manifest']['sha256'])
            b1={'history_manifest_sha256':value['history']['sha256'],'sid_version':core.SID_VERSION}
            context=SimpleNamespace(binding={'b1':b1,'synthetic':False})
            info={'sid_version':core.SID_VERSION,'manifest_sha256':sids['cuda']['manifest']['sha256'],
                  'branching':48,'depth':3,'suffix_vocabulary':0,'vocab_size':146,'kind':'fixture','engineering':False}
            sm={'content_feature_sha256':'a'*64,'input_manifest_sha256':sids['cuda']['input_manifest']['sha256'],
                'input_feature_sha256':'b'*64,'input_dimension':64}
            model=SimpleNamespace(init_report={'fixture':'metadata only; no model training'})
            import os
            from unittest.mock import patch
            with patch.dict(os.environ,{'VKGR_DATA_ROOT':str(base/'raw')}):
                binding=engine['_binding'](ROOT,context,cfg,env,model,value['prepared']['sha256'],value['history']['sha256'],SimpleNamespace(info=info,manifest=sm))
            self.assertEqual(binding['source'],native_source)
            out=base/'acceptance/train/native';out.mkdir(parents=True)
            best=put(out/'best.pt',b'fixture checkpoint never loaded')['sha256'];last=put(out/'last.pt',b'fixture checkpoint never loaded')['sha256']
            ev=put(out/'evaluation-001/manifest.json',{'fixture':'metadata only'})['sha256']
            manifest={'schema':original.MANIFEST_SCHEMA,'kind':original.MANIFEST_KIND,'status':'complete','config':cfg,'binding':binding,'selection':{},
                      'epochs':[{'epoch':1,'evaluation':{'path':'evaluation-001','manifest_sha256':ev}}],'losses':[],
                      'best':{'path':'best.pt','sha256':best,'epoch':1},'last':{'path':'last.pt','sha256':last,'epoch':1},
                      'wall_seconds':{},'hit_max_epochs':False,'eval_workers':1,'decode':None}
            from analysis import sgns_subset_a1 as cli
            ov={**value,'output_root':str(base/'acceptance')};mr=put(out/'manifest.json',manifest)
            self.assertEqual(cli.train_output_binding(ov,{'native_manifest_sha256':mr['sha256']},engine,cfg,env),manifest)
            lock_bytes=frozen_lock.read_bytes()
            frozen_lock.write_bytes(b'wrong bytes')
            with self.assertRaisesRegex(ValueError,'consumed artifact'):engine['_source'](ROOT)
            frozen_lock.write_bytes(lock_bytes)
            old_record=dict(value['lock']);value['lock']={**old_record,'sha256':'1'*64}
            with self.assertRaisesRegex(ValueError,'lock mapping'):engine['_source'](ROOT)
            value['lock']=old_record
            alternate=base/'alternate.lock';alternate.write_bytes(lock_bytes);value['lock']={**old_record,'path':str(alternate)}
            with self.assertRaisesRegex(ValueError,'lock mapping'):engine['_source'](ROOT)
            value['lock']=old_record
            original_entry=actual['files']['analysis/sgns_subset_a1.py'];actual['files']['analysis/sgns_subset_a1.py']='1'*64
            with self.assertRaisesRegex(ValueError,'source identity'):engine['_source'](ROOT)
            actual['files']['analysis/sgns_subset_a1.py']=original_entry
            self.assertIs(engine['_contexts'].__code__,core.check_native()._contexts.__code__)
            self.assertIs(engine['_load_sid'].__globals__['coll'].load_bundle,sid_module.load_bundle)
            # An actual source/producer/API closure is built; real native H5a rejects synthetic CONTENT_V1.
            import os
            from unittest.mock import patch
            with patch.dict(os.environ,{'VKGR_DATA_ROOT':str(base/'raw')}):
                with self.assertRaisesRegex(ValueError,'H5a CONTENT_V1 mismatch|STOP: original runtime identity differs'):
                    engine['_contexts'](*common,core.registered_config(sids['cuda']['manifest']['sha256']),base/'contexts',**extra)
            print(json.dumps({'integration':'actual engine_for and original _contexts, no function stubs','a1_paths':len(delta),
                              'sid_paths':4,'cutoff':cutoff,'formal_preflight':False,'scientific_passed':False},sort_keys=True))

class A1PreflightContextTests(unittest.TestCase):
    def test_formal_preflight_creates_context_before_final_sample_without_rng(self):
        # Missing allocation, wrong GPU, early release or missing sync must fail.
        import contextlib, os, tempfile, threading, weakref
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from analysis import sgns_subset_a1 as cli, g_decode_timing as timing
        from vkgr import sgns_subset_a1_inputs as inputs
        state={};rng=torch.get_rng_state().clone();empty=torch.empty
        def allocate(size,*,device):
            self.assertEqual(size,1);self.assertEqual(device,'cuda')
            tensor=empty(size);state['held']=weakref.ref(tensor)
            return tensor
        def sync():state['synced']=True
        def nvidia(*args,**kwargs):
            if state.get('held',lambda:None)() is not None and state.get('synced'):
                return f'{os.getpid()}, 512\n'
            return '999999999, 512\n'
        sampler=timing.ContextSampler.__new__(timing.ContextSampler)
        sampler.peak=0;sampler.samples=0;sampler.lock=threading.Lock()
        with tempfile.TemporaryDirectory(prefix='a1-context-contract-') as td,contextlib.ExitStack() as stack:
            base=Path(td).resolve()
            value={'data_root':str(base/'raw'),'output_root':str(base/'out'),'scratch_root':str(base/'scratch'),
                   'input_producer':{},'prepared':{},'history':{},'gpu_sid':{'manifest':{'sha256':'1'*64}}}
            def contexts(*a,**k):
                (base/'scratch/preflight').mkdir(parents=True)
                return SimpleNamespace(binding={'synthetic':False},queries=[{}],catalog=[1,2]),False,None
            engine={'_contexts':contexts,'_runtime':lambda x:{'device':'cuda'}}
            for name,value_ in [('read_request',value),('identity',({'commit':'a'*40},{})),('own_job','b'*64)]:
                stack.enter_context(patch.object(cli,name,return_value=value_))
            stack.enter_context(patch.object(cli.core,'engine_for',return_value=(engine,(),{},{})))
            stack.enter_context(patch.object(cli.core,'closure',return_value={}))
            stack.enter_context(patch.object(inputs,'input_producer',return_value={}))
            stack.enter_context(patch.dict(os.environ,{'VKGR_DATA_ROOT':value['data_root']}))
            stack.enter_context(patch.object(torch,'empty',side_effect=allocate))
            stack.enter_context(patch.object(torch.cuda,'synchronize',side_effect=sync))
            stack.enter_context(patch.object(timing.subprocess,'check_output',side_effect=nvidia))
            receipt=cli._execute('preflight',base/'request.json','c'*64,sampler)
            self.assertEqual(receipt['context_inclusive_peak_mib'],512)
            self.assertEqual(receipt['context_samples'],1)
            self.assertEqual(receipt['status'],'ready')
            self.assertFalse(receipt['payload_written'])
            self.assertEqual(torch.get_rng_state().tolist(),rng.tolist())
            self.assertEqual(json.loads((base/'out/preflight/stage-receipt.json').read_bytes()),receipt)
