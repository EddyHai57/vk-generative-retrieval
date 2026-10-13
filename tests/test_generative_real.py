"""G3 original engineering fixtures; no real data or CUDA claim."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from vkgr.data_audit import canonical_bytes
try:
    from vkgr import generative_real as gr
except ImportError:
    gr=None

ROOT=Path(__file__).resolve().parents[1]
def write(path,value):
    path.write_bytes(canonical_bytes(value));return hashlib.sha256(path.read_bytes()).hexdigest()

def invoke_verified(run,base):
    import importlib.util
    spec=importlib.util.spec_from_file_location('g3_invoke',ROOT/'cache/g3-formal-invoke-v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.invoke(run['manifest_path'],run['manifest_sha256'],run['pool_head'],base/'verify',base/'invocation',engineering=True)

class AdapterTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(gr,'G3 real adapter missing')
    def test_missing_pin_and_test_role_reject_before_contract_read(self):
        with patch('vkgr.generative_data.load_contract',side_effect=AssertionError('must not open')):
            for spec in ({}, {'schema_version':1,'task_id':'G3','kind':'real_input_spec','status':'inputs_accepted','validation_path':'test-queries.json'}):
                with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                    p=Path(d)/'input.json';pin=write(p,spec)
                    with self.assertRaises(ValueError):gr.prepare_real(ROOT,p,pin,Path(d)/'out')
                    self.assertFalse((Path(d)/'out').exists())
    def test_wrong_external_input_spec_pin_rejects_before_parse(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p=Path(d)/'input.json';write(p,{'test':'do not parse'})
            with patch('vkgr.generative_data._json',side_effect=AssertionError('must not parse')):
                with self.assertRaises(ValueError):gr.prepare_real(ROOT,p,'0'*64,Path(d)/'out')

class RuntimeTests(unittest.TestCase):
    def test_engineering_factory_stays_synthetic_and_preserves_old_source(self):
        self.assertTrue(hasattr(gr,'prepare_engineering'),'G3 engineering factory missing')
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            context,receipt=gr.prepare_engineering(ROOT,Path(d)/'prepared')
            self.assertEqual(context.binding['task_id'],'G3');self.assertTrue(context.binding['synthetic'])
            self.assertTrue(context.binding['engineering']);self.assertEqual(context.binding['g2_source_sha256'],gr.G2_SHA)
            self.assertEqual(len(context.binding['consumer']['source_members']),46)
            self.assertEqual(len(context.examples),4);context.check()
            with self.assertRaises((AttributeError,ValueError)):context._binding=b'{}'
    def test_invalid_real_runtime_stops_before_cuda_initialization(self):
        self.assertTrue(hasattr(gr,'runtime_snapshot'),'G3 runtime gate missing')
        with patch('torch.cuda.is_available',side_effect=AssertionError('no CUDA reads')):
            with self.assertRaises(ValueError):gr.runtime_snapshot({'mode':'real','device':'cpu'})
    def test_formal_registration_config_seed_output_and_selection_are_pinned(self):
        self.assertTrue(hasattr(gr,'check_registration'),'registered formal run gate missing')
        from test_generative_training import config
        registration={'run_id':'core-00-seed42','config_index':0,'seed':42,'output':'/private/tmp/only-this-run','selection':None}
        gr.check_registration(config(),registration,Path('/private/tmp/only-this-run'))
        for mode in ('seed','output','selected_seed'):
            cfg=config();entry=dict(registration)
            if mode=='seed':entry['seed']=2024
            elif mode=='output':entry['output']='/private/tmp/different'
            else:cfg['seed']=2024;entry['seed']=2024;entry['run_id']='selected-seed2024'
            with self.assertRaises(ValueError):gr.check_registration(cfg,entry,Path('/private/tmp/only-this-run'))

class RunnerTests(unittest.TestCase):
    def test_actual_complete_and_step_resume_equal_local_state_with_global_cost(self):
        self.assertTrue(hasattr(gr,'run'),'G3 orchestration missing')
        from test_generative_training import config
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            limits=gr.engineering_limits(context,prepared,max_epochs=2)
            full=gr.run(context,config(),limits,base/'full')
            split_limits=gr.engineering_limits(context,prepared,max_epochs=2)
            first=gr.run(context,config(),split_limits,base/'first',stop_after_step=1)
            self.assertEqual(first['manifest']['status'],'interrupted')
            child=gr.resume(context,first['manifest_path'],first['manifest_sha256'],first['pool_head'],base/'child')
            self.assertEqual(child['manifest']['status'],'complete')
            for key in ('state_sha256','selection','losses','epoch_metrics'):
                self.assertEqual(full['manifest'][key],child['manifest'][key])
            self.assertEqual(child['manifest']['budget']['completed_steps'],2)
            self.assertEqual(child['pool_state']['steps']['formal'],2)
            self.assertEqual(child['pool_state']['resumes'],1)
            self.assertEqual(child['manifest']['task_id'],'G3')
            self.assertTrue(child['manifest']['binding']['synthetic'])
    def test_actual_full_replay_and_rehashed_oov_candidate_rejection(self):
        self.assertTrue(hasattr(gr,'verify'),'G3 actual verifier missing')
        from test_generative_training import config
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            result=gr.run(context,config(),gr.engineering_limits(context,prepared,max_epochs=2),base/'run')
            receipt=gr.verify(context,result['manifest_path'],result['manifest_sha256'],result['pool_head'],base/'verified',replay=True)
            self.assertTrue(receipt['replay']);self.assertEqual(receipt['pool_state']['steps'],{'formal':2,'replay':2})
            self.assertEqual(receipt['pool_state']['decode'],28)
            result=gr.run(context,config(),gr.engineering_limits(context,prepared,max_epochs=2),base/'fresh-tamper-run')
            value=json.loads(result['manifest_path'].read_bytes());evaluation=value['evaluations'][0]
            epath=Path(evaluation['manifest_path']);ev=json.loads(epath.read_bytes());shard=ev['shards'][0];spath=Path(shard['path']);s=json.loads(spath.read_bytes())
            oov=next(row for row in s['rows'] if context.queries[row['ordinal']]['oov']);oov['items'].reverse()
            shard['sha256']=write(spath,s)
            from vkgr import generative_evaluation as engine
            allrows=[]
            for r in ev['shards']:allrows.extend(json.loads(Path(r['path']).read_bytes())['rows'])
            ev['computation_sha256']=engine._computation(allrows,ev['aggregate']);evaluation['manifest_sha256']=write(epath,ev)
            newpin=write(result['manifest_path'],value)
            # Another fresh pool is not allowed to conceal prior costs; structural
            # per-query checks still run before consuming a new replay reservation.
            with self.assertRaisesRegex(ValueError,'actual per-query'):gr.verify(context,result['manifest_path'],newpin,result['pool_head'],base/'tampered',replay=True)
    def test_evaluate_full_best_queries_consumes_remaining_decode_budget(self):
        self.assertTrue(hasattr(gr,'evaluate'),'G3 selected full evaluation missing')
        from test_generative_training import config
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            result=gr.run(context,config(),gr.engineering_limits(context,prepared),base/'run')
            checked=gr.verify(context,result['manifest_path'],result['manifest_sha256'],result['pool_head'],base/'verify',replay=True)
            evaluated=gr.evaluate(context,result['manifest_path'],result['manifest_sha256'],checked['pool_head'],base/'evaluate')
            self.assertEqual(evaluated['pool_state']['decode'],35)
            self.assertEqual(evaluated['aggregate']['metrics']['all_eligible']['count'],7)
            with self.assertRaises(ValueError):gr.evaluate(context,result['manifest_path'],result['manifest_sha256'],evaluated['pool_head'],base/'again')

class SelectionTests(unittest.TestCase):
    def test_missing_registered_config_cannot_freeze_real_selection(self):
        self.assertTrue(hasattr(gr,'select'),'G3 registered selection missing')
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            with self.assertRaises(ValueError):gr.select([],[],Path(d)/'selection',engineering=False)
            self.assertFalse((Path(d)/'selection').exists())
    def test_actual_cpu_cli_chain_and_real_gate_without_permit(self):
        self.assertTrue(hasattr(gr,'main'),'G3 CLI missing')
        import subprocess,sys,os
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1'}
            def cli(*args):
                r=subprocess.run([sys.executable,'-B','-m','vkgr.generative_real',*map(str,args)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
                self.assertEqual(r.returncode,0,r.stderr);return json.loads(r.stdout)
            p=cli('prepare','--engineering','--output',base/'prepared')
            pilot=cli('pilot','--engineering','--prepared',p['manifest_path'],'--prepared-sha',p['manifest_sha256'],'--output',base/'pilot')
            self.assertEqual(pilot['status'],'engineering_pilot_pass_no_GPU_claim')
            a=cli('train','--engineering','--prepared',p['manifest_path'],'--prepared-sha',p['manifest_sha256'],'--config-index','0','--seed','42','--stop-after-step','1','--output',base/'first')
            b=cli('resume','--engineering','--parent',a['manifest_path'],'--parent-sha',a['manifest_sha256'],'--pool-head',a['pool_head'],'--output',base/'resumed')
            e=cli('evaluate','--engineering','--manifest',b['manifest_path'],'--sha256',b['manifest_sha256'],'--pool-head',b['pool_head'],'--output',base/'evaluate')
            result=subprocess.run([sys.executable,'-B',str(ROOT/'cache/g3-formal-invoke-v2.py'),'--engineering','--manifest',b['manifest_path'],'--sha256',b['manifest_sha256'],'--pool-head',e['pool_head'],'--verify-output',str(base/'verify'),'--output',str(base/'invocation')],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr);v=json.loads(result.stdout)
            s=cli('select','--engineering','--verification',v['verification_path'],'--verification-sha',v['verification_sha256'],'--verification-invocation',v['invocation_path'],'--verification-invocation-sha',v['invocation_sha256'],'--output',base/'selection')
            self.assertEqual(s['status'],'engineering_selection_no_real_winner')
            failed=subprocess.run([sys.executable,'-B','-m','vkgr.generative_real','train','--prepared','/missing','--prepared-sha','0'*64,'--output',str(base/'forbidden')],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0);self.assertFalse((base/'forbidden').exists())

class EngineeringCapTests(unittest.TestCase):
    def test_epoch_progress_is_bounded_and_does_not_pollute_stdout(self):
        from test_generative_training import config
        import io,contextlib
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            stdout=io.StringIO();stderr=io.StringIO()
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                gr.run(context,config(),gr.engineering_limits(context,prepared),base/'run')
            rows=[json.loads(line) for line in stderr.getvalue().splitlines()]
            self.assertEqual([r['epoch'] for r in rows if r['event']=='epoch_complete'],[1,2])
            self.assertLessEqual(len(stderr.getvalue().encode()),8192);self.assertEqual(stdout.getvalue(),'')

    def test_frozen_closure_and_identity_mutations_are_rejected(self):
        from test_generative_training import config
        import copy
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            run=gr.run(context,config(),gr.engineering_limits(context,prepared),base/'run');v=invoke_verified(run,base)
            p=Path(v['verification_path']);original=json.loads(p.read_text());raw=p.read_bytes()
            for mode in ('schema','config','source','input','evaluator','engineering','coverage'):
                altered=copy.deepcopy(original)
                if mode=='schema':altered['schema_version']=1
                elif mode=='config':altered['binding']['effective_config']['seed']=2024
                elif mode=='source':altered['binding']['consumer']['source_sha256']='0'*64
                elif mode=='input':altered['binding']['examples_sha256']='0'*64
                elif mode=='evaluator':altered['binding']['consumer']['source_members']['src/vkgr/generative_evaluation.py']='0'*64
                elif mode=='engineering':altered['binding']['engineering']=False
                else:altered['query_count']=0
                pin=write(p,altered)
                with self.subTest(mode=mode):
                    with self.assertRaises(ValueError):gr.read_verified_v2(p,pin,v['invocation_path'],v['invocation_sha256'],True)
                p.write_bytes(raw)
            checkpoint=Path(run['manifest']['last']['path']);data=checkpoint.read_bytes();checkpoint.write_bytes(data+b'bad')
            with self.assertRaises(ValueError):gr.read_verified_v2(p,v['verification_sha256'],v['invocation_path'],v['invocation_sha256'],True)
            checkpoint.write_bytes(data)
            ledger=next((base/'run-pool').glob('event-*.json'));data=ledger.read_bytes();ledger.unlink()
            with self.assertRaises(ValueError):gr.read_verified_v2(p,v['verification_sha256'],v['invocation_path'],v['invocation_sha256'],True)
            ledger.write_bytes(data)
            self.assertEqual(gr.read_verified_v2(p,v['verification_sha256'],v['invocation_path'],v['invocation_sha256'],True)['selection'],run['manifest']['selection'])

    def test_invocation_schema_and_logs_are_authenticated(self):
        from test_generative_training import config
        import copy
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            run=gr.run(context,config(),gr.engineering_limits(context,prepared),base/'run');v=invoke_verified(run,base)
            p=Path(v['invocation_path']);original=json.loads(p.read_text())
            for mode in ('missing_exit','failed_exit','bool_exit','unknown_field','argv','environment','log'):
                altered=copy.deepcopy(original);log=base/'invocation/stdout';log_bytes=log.read_bytes()
                if mode=='missing_exit':del altered['exit']
                elif mode=='failed_exit':altered['exit']=1
                elif mode=='bool_exit':altered['exit']=False
                elif mode=='unknown_field':altered['unverified_extension']=True
                elif mode=='argv':altered['argv'][3]='another.module'
                elif mode=='environment':altered['environment']['PYTHONPATH']='/untrusted'
                else:log.write_bytes(log_bytes+b'tampered')
                pin=write(p,altered)
                with self.subTest(mode=mode):
                    with self.assertRaises(ValueError):gr.read_verified_v2(v['verification_path'],v['verification_sha256'],p,pin,True)
                log.write_bytes(log_bytes)
            write(p,original)

    def test_selection_does_not_deserialize_or_reaggregate_verified_trajectory(self):
        from test_generative_training import config
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            run=gr.run(context,config(),gr.engineering_limits(context,prepared),base/'run')
            v=invoke_verified(run,base)
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for directory in (base/'run-pool',base/'verify',base/'invocation') for p in directory.rglob('*') if p.is_file()}
            from contextlib import ExitStack
            with ExitStack() as stack:
                for obj,name in ((gr,'_read_run'),(gr,'_train_trajectory'),(gr,'configure_runtime'),(gr.gt,'make_model'),(gr.gt,'load_checkpoint'),(gr.ev,'aggregate_items'),(gr.ev,'evaluate_queries')):
                    stack.enter_context(patch.object(obj,name,side_effect=AssertionError('select must only hash frozen artifacts')))
                for index in range(2):
                    result=gr.select([v['verification_path']],[v['verification_sha256']],base/('selection'+str(index)),engineering=True,invocation_paths=[v['invocation_path']],invocation_pins=[v['invocation_sha256']])
                    self.assertEqual(result['status'],'engineering_selection_no_real_winner')
            self.assertEqual(before,{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in before})
            selected=json.loads(result['manifest_path'].read_text())
            self.assertEqual(selected['engineering_oracle']['metric'],run['manifest']['selection']['best_metric'])
            receipt=json.loads(Path(v['verification_path']).read_text());self.assertEqual(receipt['pool_state']['steps'],{'formal':2,'replay':2})

    def test_engineering_profile_cannot_extend_decode_or_absolute_window(self):
        from test_generative_training import config
        import copy,datetime
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            limits=gr.engineering_limits(context,prepared)
            for mode in ('decode','deadline'):
                altered=copy.deepcopy(limits)
                if mode=='decode':altered['pool_spec']['max_decode']+=1
                else:altered['pool_spec']['deadline_utc']=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat()
                with self.subTest(mode=mode):
                    with self.assertRaises(ValueError):gr.run(context,config(),altered,base/mode)
                    self.assertFalse((base/mode).exists())

if __name__=='__main__':unittest.main()
