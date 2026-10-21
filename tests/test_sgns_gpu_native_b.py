import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class EntryTests(unittest.TestCase):
    def test_cli_dry_and_illegal(self):
        env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
        for name,code in [('dry',0),('invalid',2)]:
            r=subprocess.run([sys.executable,'-B','-m','analysis.sgns_gpu_native_b',name],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,code,r.stderr)
            if code==0:self.assertFalse(json.loads(r.stdout)['training'])
    def test_prefix_bound_and_projection_diagnostic_not_acceptance(self):
        from analysis.sgns_gpu_native_b import benchmark
        from analysis.sgns_speedup_profile import synthetic_inputs
        catalog,histories,_=synthetic_inputs()
        r=benchmark(catalog,histories,device='cpu',warmup=1,measured=3,compiled=False)
        self.assertEqual(r['warmup_positive_pairs'],4096);self.assertEqual(r['measured_positive_pairs'],12288)
        self.assertFalse(r['vectors_saved']);self.assertFalse(r['four_gates_passed']);self.assertEqual(r['native_epoch_positive_pairs'],119040)
        self.assertEqual(len(r['negative_observed_bins']),64)
        with self.assertRaises(ValueError):benchmark(catalog,histories,device='cpu',warmup=0,measured=4097,compiled=False)
    def test_projection_and_distribution_receipt_are_mechanically_verified(self):
        from analysis.sgns_gpu_native_b import check_measurement,FULL_PAIRS
        value={'measured_batches':3};rate=12288/.1
        receipt={'measured_seconds':.1,'measured_pairs_per_second':rate,'full_fit_linear_projection_hours':FULL_PAIRS/rate/3600,
                 'B_throughput_12h_diagnostic':False,'full_pair_count':FULL_PAIRS,
                 'negative_observed_bins':[960]*64,'negative_expected_bins':[960.]*64}
        check_measurement(receipt,value)
        for key,val in {'measured_seconds':float('nan'),'measured_pairs_per_second':1.,'full_fit_linear_projection_hours':0.,
                        'B_throughput_12h_diagnostic':True,'full_pair_count':float(FULL_PAIRS),'negative_observed_bins':[0]*64,'negative_expected_bins':[0.]*64}.items():
            with self.subTest(key=key),self.assertRaises(ValueError):check_measurement({**receipt,key:val},value)
    def test_request_sha_holdout_and_frozen_identity(self):
        from analysis.sgns_gpu_native_b import validate,load
        with self.assertRaises(ValueError):validate({})
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path=Path(d)/'request.json';path.write_text('{}')
            with self.assertRaises(ValueError):load(path,'a'*64)


class BindingTests(unittest.TestCase):
    def setUp(self):
        from analysis import sgns_gpu_native_b as p
        self.p=p;self.temp=tempfile.TemporaryDirectory(dir='/private/tmp');base=Path(self.temp.name)
        bundle=base/'bundle';bundle.mkdir()
        for name in ('manifest.json','catalog.json','content.npy','histories.jsonl'):(bundle/name).write_text('fixture')
        lock=base/'lock';lock.write_text('fixture')
        self.value={'schema':'sgns-gpu-native-b-request-v1','runner_source':'a'*40,'bundle':str(bundle),'bundle_sha256':p.digest(bundle/'manifest.json'),
            'lock':str(lock),'ops_root':str(base/'ops'),'output':str(base/'output'),'preflight_run_id':'sgns-gpu-native-b-preflight-fixture-v1',
            'benchmark_run_id':'sgns-gpu-native-b-benchmark-fixture-v1','warmup_batches':1,'measured_batches':3,'compiled':True}
        self.path=base/'request.json';self.path.write_text(json.dumps(self.value));self.sha=p.digest(self.path)
        self.ledger={'source':'fixed'};self.bound={'input':'fixed'}
        self.run=base/'ops/runs'/self.value['preflight_run_id'];(self.run/'logs').mkdir(parents=True)
        from dataclasses import asdict
        self.record={'schema':'vkgr-run-v1','run_id':self.value['preflight_run_id'],'kind':'ops','cwd':str(p.ROOT),'argv':p.argv('preflight',self.path,self.sha),'seed':42,
            'git':{'commit':'a'*40,'dirty':False,'dirty_files':[]},'config':{'path':str(self.path),'sha256':self.sha},'resources':p.resources(),'env':{'PYTHONPATH':str(p.ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VKGR_RUN_ID':self.value['preflight_run_id'],'VKGR_PROGRESS_PATH':str(self.run/'progress.json')},'inputs':p.pins(self.value,self.path,'preflight')}
        (self.run/'run.json').write_text(json.dumps(self.record))
        self.receipt={'schema':'sgns-gpu-native-b-stage-v1','stage':'preflight','status':'verified','engineering':False,'run_id':self.value['preflight_run_id'],
            'runner_source':self.ledger,'input_binding':self.bound,'request_sha256':self.sha,'run_manifest_sha256':p.digest(self.run/'run.json'),
            'test_read':False,'validation_read':False,'science_training':False,'native_epoch_positive_pairs':34770216,'config':asdict(p.SGNSConfig()),
            'real_readonly_preflight':True,'temporary_parameter_updates':False}
        self.terminal={'schema':'vkgr-terminal-v1','run_id':self.value['preflight_run_id'],'status':'succeeded','exit_code':0,'signal':None}
        self.write(self.receipt)
    def tearDown(self):self.temp.cleanup()
    def write(self,value):
        (self.run/'terminal.json').write_text(json.dumps(self.terminal));(self.run/'logs/stdout.log').write_text(json.dumps(value))
        out=Path(self.value['output'])/'preflight';out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(value))
    def test_pf_source_job_full_counts_and_science_flags_closed(self):
        p=self.p;self.assertEqual(p.accepted(self.value,self.path,self.sha,'preflight',self.ledger,self.bound)['run_id'],self.value['preflight_run_id'])
        for key,val in {'status':'ready','native_epoch_positive_pairs':1,'config':{},'test_read':True,'validation_read':True,'science_training':True,
                        'temporary_parameter_updates':True,'real_readonly_preflight':False,'runner_source':{},'input_binding':{}}.items():
            self.write({**self.receipt,key:val})
            with self.subTest(key=key),self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.ledger,self.bound)
        self.write(self.receipt);self.terminal['status']='failed';self.write(self.receipt)
        with self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.ledger,self.bound)
    def test_request_scope_holdout_and_source_changes_refused(self):
        p=self.p
        self.assertEqual(p.load(self.path,self.sha),self.value)
        for value in ({**self.value,'bundle':'/workspace/holdout/bundle'},{**self.value,'output':self.value['bundle']+'/new'},
                      {**self.value,'preflight_run_id':self.value['benchmark_run_id']},{**self.value,'measured_batches':4097}):
            with self.assertRaises(ValueError):p.validate(value)
        record={**self.record,'seed':43};(self.run/'run.json').write_text(json.dumps(record))
        with self.assertRaises(ValueError):p.job(self.value,self.path,self.sha,'preflight')
    def test_benchmark_requires_original_pf_before_any_gpu_updates(self):
        from unittest.mock import patch
        p=self.p
        with patch.object(p,'benchmark') as work,patch.object(p,'load',return_value=self.value),patch.object(p,'source',return_value=self.ledger),patch.object(p,'job',return_value='a'*64), \
             patch.object(p,'pins',return_value=[]),patch.object(p,'inputs',return_value=(([],{},[]),self.bound)), \
             patch.dict(os.environ,{**self.record['env'],'VKGR_RUN_ID':self.value['benchmark_run_id'],'VKGR_PROGRESS_PATH':str(self.run.parent/self.value['benchmark_run_id']/'progress.json')},clear=True), \
             patch('analysis.ur001_derive_cli.guards'):
            self.terminal['status']='failed';self.write(self.receipt)
            with self.assertRaises(ValueError):p.execute('benchmark',self.path,self.sha)
            work.assert_not_called()

    def test_reviewer_nested_type_confusion_attacks_refused(self):
        import copy
        p=self.p
        for key in ('seed','epochs'):
            wrong=copy.deepcopy(self.receipt);wrong['config'][key]=float(wrong['config'][key]);self.write(wrong)
            with self.subTest(key=key),self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.ledger,self.bound)
        wrong=copy.deepcopy(self.receipt);wrong['config']['extra']=1;self.write(wrong)
        with self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.ledger,self.bound)
        self.write(self.receipt)
        wrong=copy.deepcopy(self.record);wrong['seed']=42.;wrong['git']['dirty']=0
        (self.run/'run.json').write_text(json.dumps(wrong))
        with self.assertRaises(ValueError):p.job(self.value,self.path,self.sha,'preflight')
    def test_own_job_environment_is_exact_and_required(self):
        import copy
        p=self.p
        for key in ('absent','PYTHONPATH','extra'):
            wrong=copy.deepcopy(self.record)
            if key=='absent':del wrong['env']
            elif key=='PYTHONPATH':wrong['env'][key]='/unexpected/src'
            else:wrong['env']['VKGR_UNREGISTERED']='1'
            (self.run/'run.json').write_text(json.dumps(wrong))
            with self.subTest(key=key),self.assertRaises(ValueError):p.job(self.value,self.path,self.sha,'preflight')
    def test_execute_checks_pythonpath_and_injected_progress_before_inputs(self):
        from unittest.mock import patch
        p=self.p
        fixed={**self.record['env'],'VKGR_RUN_ID':self.value['benchmark_run_id'],'VKGR_PROGRESS_PATH':str(self.run.parent/self.value['benchmark_run_id']/'progress.json')}
        for key in ('PYTHONPATH','VKGR_PROGRESS_PATH'):
            with patch.object(p,'load',return_value=self.value),patch.object(p,'source',return_value=self.ledger),patch.object(p,'job',return_value='a'*64),patch.object(p,'pins',return_value=[]),patch.object(p,'inputs') as work,patch.dict(os.environ,{**fixed,key:'/unexpected'},clear=True):
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'environment'):p.execute('benchmark',self.path,self.sha)
                work.assert_not_called()
    def test_request_hash_and_parse_consume_same_single_read(self):
        from unittest.mock import patch
        p=self.p;swapped=json.dumps({**self.value,'compiled':False}).encode()
        with patch.object(Path,'read_bytes',return_value=swapped),self.assertRaises(ValueError):p.load(self.path,self.sha)
        raw=self.path.read_bytes()
        with patch.object(Path,'read_bytes',return_value=raw) as read,patch.object(p,'digest',side_effect=AssertionError('second digest read forbidden')):
            self.assertEqual(p.load(self.path,self.sha),self.value);self.assertEqual(read.call_count,1)
    def test_runtime_comparison_uses_json_scalar_types_for_torch_version(self):
        from unittest.mock import patch
        from torch.torch_version import TorchVersion
        from analysis.new_instance_prepare import LOCK_SHA
        p=self.p
        runtime={'torch':TorchVersion('2.11.0+cu128'),'cuda_smoke_executed':False}
        producer={'runtime':json.loads(json.dumps(runtime)),'source':{},'train_sequence_sha256':'a'*64,'artifacts':{}}
        path=Path(self.value['bundle'])/'manifest.json';path.write_text(json.dumps(producer));self.value['bundle_sha256']=p.digest(path)
        digest=p.digest
        with patch.object(p,'digest',side_effect=lambda path:LOCK_SHA if str(path)==self.value['lock'] else digest(path)), \
             patch('analysis.new_instance_prepare.runtime',return_value=runtime),patch('analysis.sgns_gpu_subset.read_bundle',return_value=(([],{},[]),producer)):
            self.assertEqual(p.inputs(self.value)[1]['runtime'],producer['runtime'])
            producer['runtime']['cuda_smoke_executed']=0;path.write_text(json.dumps(producer));self.value['bundle_sha256']=digest(path)
            with self.assertRaises(ValueError):p.inputs(self.value)
    def test_nested_runtime_input_and_terminal_types_refused(self):
        import copy
        p=self.p;self.ledger['files_count']=7;self.bound['runtime']={'cuda_smoke_executed':False}
        self.receipt['runner_source']=self.ledger;self.receipt['input_binding']=self.bound
        for role,key,value in (('runner_source','files_count',7.),('input_binding','runtime',{'cuda_smoke_executed':0})):
            wrong=copy.deepcopy(self.receipt);wrong[role][key]=value;self.write(wrong)
            with self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.ledger,self.bound)
        self.write(self.receipt);self.terminal['exit_code']=False;self.write(self.receipt)
        with self.assertRaises(ValueError):p.accepted(self.value,self.path,self.sha,'preflight',self.ledger,self.bound)

class ProgressTests(unittest.TestCase):
    def test_benchmark_stages_are_observable_only_on_stderr(self):
        from contextlib import redirect_stdout,redirect_stderr
        import io
        from analysis.sgns_gpu_native_b import benchmark
        from analysis.sgns_speedup_profile import synthetic_inputs
        catalog,histories,_=synthetic_inputs();out=io.StringIO();err=io.StringIO()
        with redirect_stdout(out),redirect_stderr(err):benchmark(catalog,histories,device='cpu',warmup=1,measured=3,compiled=False)
        events=[json.loads(line) for line in err.getvalue().splitlines()]
        self.assertEqual(out.getvalue(),'')
        self.assertEqual([x['phase'] for x in events],['build_begin','build_end','warmup_compile_begin','warmup_compile_end','measure_begin','measure_progress','measure_end'])
        self.assertEqual(events[-2]['batches_enqueued'],3)
if __name__=='__main__':unittest.main()
