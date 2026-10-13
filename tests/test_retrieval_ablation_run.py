"""P4E external binding gates, independent state replay and CLI integration."""
import importlib.util
import unittest
import tempfile
from pathlib import Path

class RunTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_ablation_run'),'P4E run binding behavior missing')
        from vkgr import retrieval_ablation_run
        return retrieval_ablation_run

    def test_strict_json_rejects_duplicates_nonfinite_and_config_extra_fields(self):
        m=self.module()
        for text in ('{"seed":42,"seed":2024}','{"x":NaN}','{"x":Infinity}'):
            with self.assertRaises(ValueError):m.strict_json(text)
        config=m.default_config();m.validate_config(config)
        with self.assertRaises(ValueError):m.validate_config({**config,'extra':1})
        with self.assertRaises(ValueError):m.validate_config({**config,'batch_size':256})

    def test_external_plan_pin_rejects_rehashed_config_and_changed_core(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json';plan=m.build_plan(m.default_config(),{'prepared_sha':m.PREPARED_SHA,'core_sha':m.CORE_SHA,'test_gate':'closed_hash_only'})
            digest=m.write_json(p,plan)
            self.assertEqual(len(m.verify_plan(p,digest,m.CORE_SHA)['slots']),9)
            m.write_json(Path(d)/'changed.json',{**plan,'slots':plan['slots'][:-1]})
            with self.assertRaises(ValueError):m.verify_plan(Path(d)/'changed.json',digest,m.CORE_SHA)
            with self.assertRaises(ValueError):m.verify_plan(p,digest,'0'*64)

    def test_full_state_resume_three_strategies_matches_continuous_and_external_tamper_fails(self):
        import torch
        m=self.module();self.assertTrue(hasattr(m,'run_session'),'P4E checkpoint/run behavior missing')
        for ablation in m.ABLATIONS:
            with self.subTest(ablation=ablation),tempfile.TemporaryDirectory() as d:
                data=m.synthetic_data();root=Path(d)
                continuous=m.run_session(root/'continuous',data,ablation,42,batch_size=8,until_step=20)
                first=m.run_session(root/'split',data,ablation,42,batch_size=8,until_step=7)
                resumed=m.run_session(root/'split',data,ablation,42,batch_size=8,until_step=20,resume=(root/'split'/first['manifest_name'],first['manifest_sha']))
                self.assertEqual(continuous['records'],resumed['records'])
                left=torch.load(root/'continuous'/continuous['last']['path'],weights_only=True)
                right=torch.load(root/'split'/resumed['last']['path'],weights_only=True)
                for key in ('model','optimizer','cursor','negative_rng','python_rng','numpy_rng','torch_rng','stats','selection'):
                    self.assertTrue(m.state_equal(left[key],right[key]),key)
                manifest=root/'split'/resumed['manifest_name']
                m.verify_run(manifest,resumed['manifest_sha'],data,core_sha=m.CORE_SHA)
                changed={**resumed,'records':resumed['records'][:-1]};changed.pop('manifest_sha',None)
                m.write_json(root/'split'/'tampered.json',changed)
                with self.assertRaises(ValueError):m.verify_run(root/'split'/'tampered.json',resumed['manifest_sha'],data,core_sha=m.CORE_SHA)

    def test_cli_synthetic_independent_process_resume_and_external_verify(self):
        import os,sys,subprocess,json,hashlib
        m=self.module();self.assertTrue(hasattr(m,'main'),'P4E CLI missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);env={**os.environ,'PYTHONPATH':str(m.ROOT/'src')}
            def cli(*args):
                r=subprocess.run([sys.executable,'-B','-m','vkgr.retrieval_ablation_run',*args],capture_output=True,text=True,env=env)
                self.assertEqual(r.returncode,0,r.stderr);return r.stdout
            for a in m.ABLATIONS:
                continuous=root/(a+'-continuous');split=root/(a+'-split')
                cli('synthetic-train','--ablation',a,'--seed','42','--until-step','20','--output',str(continuous))
                cli('synthetic-train','--ablation',a,'--seed','42','--until-step','7','--output',str(split))
                first=split/'manifest-step-7.json';digest=hashlib.sha256(first.read_bytes()).hexdigest()
                cli('synthetic-train','--ablation',a,'--seed','42','--until-step','20','--output',str(split),'--resume',str(first),'--resume-sha',digest)
                latest=max(split.glob('manifest-step-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
                final=json.loads(latest.read_text());other=json.loads((continuous/latest.name).read_text())
                self.assertEqual(final['records'],other['records']);self.assertEqual(final['evaluations'],other['evaluations'])
                cli('verify','--synthetic','--manifest',str(latest),'--manifest-sha',hashlib.sha256(latest.read_bytes()).hexdigest(),'--core-sha',m.CORE_SHA)
            self.assertIn('synthetic-train',cli('--help'))

    def test_rehashed_wrong_best_checkpoint_is_rejected(self):
        m=self.module();self.assertTrue(hasattr(m,'run_session'),'P4E run missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=m.synthetic_data();result=m.run_session(root/'run',data,'uniform_negatives',42,batch_size=8,until_step=7)
            value={k:v for k,v in result.items() if k!='manifest_sha'};value['best']={**value['last'],'metric':[0.,0.]}
            p=root/'run'/'wrong-best.json';sha=m.write_json(p,value)
            with self.assertRaises(ValueError):m.verify_run(p,sha,data,m.CORE_SHA)

    def test_watch_disk_artifacts_external_source_pin_and_raw_replay(self):
        m=self.module();self.assertTrue(hasattr(m,'write_watch'),'watch disk provenance missing')
        from test_retrieval_watch import WatchTests
        events,e,q=WatchTests().fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'watch';binding={'raw_records':[{'path':'week_00.parquet','sha256':'1'*64,'size_bytes':5}],'prepared_sha':m.PREPARED_SHA,'query_sha':'2'*64,'catalog_sha':'3'*64,'core_sha':m.CORE_SHA,'consumer':m.source_identity(),'test_gate':'closed_hash_only'}
            value=m.write_watch(root,[(0,events)],[e],[q],[0,2,3,4],binding)
            checked=m.verify_watch(root/'watch-manifest.json',value['manifest_sha'],[(0,events)],[e],[q],[0,2,3,4],binding)
            self.assertEqual(checked['binding'],binding)
            with self.assertRaises(ValueError):m.verify_watch(root/'watch-manifest.json','0'*64,[(0,events)],[e],[q],[0,2,3,4],binding)

    def test_slice_summary_requires_nine_unique_seeds_and_reports_paired_sample_std(self):
        m=self.module();self.assertTrue(hasattr(m,'summarize_metrics'),'paired summary behavior missing')
        records=[];cores={}
        for seed,metric in zip((42,2024,2025),(.1,.2,.3)):
            cores[str(seed)]={'all':{'recall@50':metric}}
        for a in m.ABLATIONS:
            for seed,metric in zip((42,2024,2025),(.2,.3,.4)):
                records.append({'ablation':a,'seed':seed,'metrics':{'all':{'recall@50':metric}}})
        result=m.summarize_metrics(records,cores)
        self.assertAlmostEqual(result['uniform_negatives']['all']['recall@50']['mean'],.3)
        self.assertAlmostEqual(result['uniform_negatives']['all']['recall@50']['sample_std'],.1)
        self.assertAlmostEqual(result['uniform_negatives']['all']['recall@50']['paired_delta_mean'],.1)
        with self.assertRaises(ValueError):m.summarize_metrics(records[:-1]+[records[0]],cores)

    def test_runtime_plan_rejects_unbounded_budget_unknown_fields_or_test_scope(self):
        m=self.module();self.assertTrue(hasattr(m,'validate_runtime_plan'),'bounded real plan schema missing')
        runtime={'prepared':'/workspace/prepared','holdout':'/workspace/holdout','raw_root':'/workspace/raw','core_selection':'/workspace/core.json','watch_manifest':None,'watch_sha':None,'wall_seconds':600,'disk_bytes':1048576,'allowed_operations':['prepare-watch'],'pilot_steps':30,'resume_steps':[7,20,30]}
        self.assertEqual(m.validate_runtime_plan(runtime),runtime)
        for changes in ({'wall_seconds':0},{'disk_bytes':float('inf')},{'extra':1},{'allowed_operations':['test']},{'pilot_steps':31}):
            with self.assertRaises(ValueError):m.validate_runtime_plan({**runtime,**changes})

    def test_cgroup_ancestor_memory_headroom_uses_tightest_visible_limit(self):
        m=self.module();self.assertTrue(hasattr(m,'visible_memory_fraction'),'visible resource hierarchy missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);child=root/'job';child.mkdir()
            (root/'memory.max').write_text('1000');(root/'memory.current').write_text('900')
            (child/'memory.max').write_text('500');(child/'memory.current').write_text('100')
            self.assertAlmostEqual(m.visible_memory_fraction(child,root,.9),.1)
            (root/'memory.max').write_text('max');(child/'memory.max').write_text('max')
            with self.assertRaises(ValueError):m.visible_memory_fraction(child,root,.9)

    def test_real_loader_rejects_raw_hash_drift_and_preserves_complete_input_bindings(self):
        import unittest.mock as mock
        m=self.module();self.assertTrue(hasattr(m,'real_data'),'real provenance loader missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);core=root/'core.json';core_sha=m.write_json(core,{'selected':'fixed'})
            records=[]
            for week in range(25):
                name=f'subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet';p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
                records.append({'role':name,'sha256':m.file_sha256(p),'size_bytes':7})
            runtime={'prepared':str(root/'prepared'),'holdout':str(root/'holdout'),'raw_root':str(root),'core_selection':str(core),'watch_manifest':None,'watch_sha':None,'wall_seconds':600,'disk_bytes':1048576,'allowed_operations':['prepare-watch'],'pilot_steps':30,'resume_steps':[7,20,30]}
            pins={'full_query_pins':[{'role':'validation-queries.json','sha256':'a'*64}],'catalog_sha256':'b'*64,'legacy_p3_producer':{'source':'frozen'}}
            data=m.synthetic_data();data['prepared']={'artifacts':{'catalog.json':'b'*64},'binding':{'input_records':records,'query_pins':pins['full_query_pins'],'legacy_producer':pins['legacy_p3_producer']}}
            # Patch only unavailable heavyweight prepared/contract readers; real raw hashing is exercised.
            with mock.patch.object(m,'CORE_SHA',core_sha),mock.patch('vkgr.retrieval_experiment.load_training_inputs',return_value=data),mock.patch('vkgr.retrieval_data.validate_contract',return_value=pins):
                actual,parts,binding=m.real_data({'inputs':{'runtime':runtime}})
                self.assertFalse(actual['synthetic']);self.assertEqual(binding['raw_records'],records)
                (root/records[3]['role']).write_bytes(b'drift!!')
                with self.assertRaises(ValueError):m.real_data({'inputs':{'runtime':runtime}})

    def test_shared_slice_manifest_is_train_only_bound_and_rejects_rehashed_statistics(self):
        m=self.module()
        from test_retrieval_watch import WatchTests
        events,e,q=WatchTests().fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'watch';binding={'raw_records':[{'path':'week_00.parquet','sha256':'1'*64,'size_bytes':5}],'raw_producer':{'source':'frozen'},'prepared_sha':m.PREPARED_SHA,'query_sha':'2'*64,'catalog_sha':'3'*64,'core_sha':m.CORE_SHA,'consumer':m.source_identity(),'test_gate':'closed_hash_only'}
            result=m.write_watch(root,[(0,events)],[e],[q],[0,2,3,4],binding)
            p=root/'slice-statistics-manifest.json';self.assertTrue(p.exists(),'independent shared slice manifest missing')
            value=m.read_pinned(p,result['artifacts'][p.name]);self.assertNotIn('query_sha',value['binding']);self.assertEqual(value['binding']['raw_producer'],binding['raw_producer'])
            self.assertEqual(value['threshold_definition'],'nearest_rank_positive_item_event_counts_median_p90')

    def test_cli_prepare_watch_uses_approved_plan_and_writes_replayable_artifacts(self):
        import unittest.mock as mock
        m=self.module();self.assertTrue(hasattr(m,'resource_guard'),'real resource callback missing')
        from test_retrieval_watch import WatchTests
        events,e,q=WatchTests().fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime={'operation_root':str(root),'prepared':str(root/'prepared'),'holdout':str(root/'holdout'),'raw_root':str(root),'core_selection':str(root/'core.json'),'watch_manifest':None,'watch_sha':None,'wall_seconds':600,'disk_bytes':1048576,'allowed_operations':['prepare-watch'],'pilot_steps':30,'resume_steps':[7,20,30]}
            plan={'inputs':{'runtime':runtime}}
            binding={'raw_records':[{'path':'week_00.parquet','sha256':'1'*64,'size_bytes':5}],'raw_producer':{'source':'frozen'},'prepared_sha':m.PREPARED_SHA,'query_sha':'2'*64,'catalog_sha':'3'*64,'core_sha':m.CORE_SHA,'consumer':m.source_identity(),'test_gate':'closed_hash_only'}
            data={'rows':[e],'queries':[q],'catalog':[0,2,3,4]}
            with mock.patch.object(m,'verify_plan',return_value=plan),mock.patch.object(m,'real_data',return_value=(data,lambda:iter([(0,events)]),binding)),mock.patch.object(m,'resource_guard',return_value=lambda:None):
                code=m.main(['prepare-watch','--plan','private-plan.json','--plan-sha','f'*64,'--core-sha',m.CORE_SHA,'--output',str(root/'watch')])
                self.assertEqual(code,0);self.assertTrue((root/'watch'/'watch-manifest.json').exists())

    def test_bounded_pilot_runs_three_strategies_and_reuses_segments_without_extra_draws(self):
        m=self.module();self.assertTrue(hasattr(m,'pilot_session'),'bounded pilot behavior missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'pilot';value=m.pilot_session(root,m.synthetic_data(),lambda:None,device='cpu')
            self.assertEqual([r['ablation'] for r in value['runs']],list(m.ABLATIONS))
            self.assertTrue(all(r['steps']<=30 for r in value['runs']))
            self.assertTrue(all(r['export_verified'] for r in value['runs']))
            self.assertTrue(all('worst_case' in r for r in value['runs']),'pilot worst-case allocation evidence missing')
            self.assertTrue(all(r['worst_case']['history_width']==50 for r in value['runs']))
            self.assertTrue((root/'pilot-manifest.json').exists())

    def test_operation_budget_deadline_survives_new_invocation_and_output_escape_fails(self):
        m=self.module();self.assertTrue(hasattr(m,'operation_guard'),'persistent operation budget missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime={'operation_root':str(root),'wall_seconds':60,'disk_bytes':10000}
            first=m.operation_guard(root/'slot1',runtime,'a'*64,now=lambda:100.)
            first();second=m.operation_guard(root/'slot2',runtime,'a'*64,now=lambda:170.)
            with self.assertRaises(ValueError):second()
            with self.assertRaises(ValueError):m.operation_guard(root.parent/'outside',runtime,'a'*64,now=lambda:100.)

    def test_worst_case_allocation_has_last50_and_uniform640_columns_without_optimizer_step(self):
        m=self.module();self.assertTrue(hasattr(m,'worst_case_context'),'worst-case resource geometry missing')
        data=m.synthetic_data();value=m.worst_case_context(data,'uniform_negatives',lambda:None,device='cpu',batch_size=8)
        self.assertEqual(value['history_width'],50);self.assertEqual(value['candidate_columns'],136);self.assertEqual(value['optimizer_steps'],0)

    def test_real_train_gate_requires_external_pilot_before_loading_data(self):
        import unittest.mock as mock
        m=self.module();self.assertTrue(hasattr(m,'dispatch_real_run'),'real train/pilot dispatcher missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            runtime={'operation_root':str(root),'prepared':str(root/'prepared'),'holdout':str(root/'holdout'),'raw_root':str(root),'core_selection':str(root/'core.json'),'watch_manifest':None,'watch_sha':None,'wall_seconds':600,'disk_bytes':1000000,'allowed_operations':['train'],'pilot_steps':30,'resume_steps':[7,20,30]}
            plan={'inputs':{'runtime':runtime},'slots':[{'slot':1,'ablation':'uniform_negatives','seed':42}]}
            args=m.argparse.Namespace(command='train',output=str(root/'slot1'),slot=1,pilot_plan=None,pilot_plan_sha=None,plan_sha='a'*64,watch_sha=None,until_step=None,resume=None,resume_sha=None)
            with self.assertRaises(ValueError):m.dispatch_real_run(args,plan)

    def test_summary_requires_actual_nine_complete_runs_and_zero_stage_exits(self):
        m=self.module();self.assertTrue(hasattr(m,'verify_completion'),'completion gate missing')
        self.assertTrue(hasattr(m,'summary_session'),'verified summary delivery missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=m.synthetic_data();rows=[]
            for j,(ablation,seed) in enumerate(((a,s) for a in m.ABLATIONS for s in (42,2024,2025)),1):
                directory=root/f'slot-{j:02d}'
                run=m.run_session(directory,data,ablation,seed,8)
                for name in ('train.exit','verify.exit'):(directory/name).write_text('0\n')
                rows.append({'slot':j,'manifest_ref':f'slot-{j:02d}/{run["manifest_name"]}','manifest_sha':run['manifest_sha'],'train_exit_sha':m.file_sha256(directory/'train.exit'),'verify_exit_sha':m.file_sha256(directory/'verify.exit')})
            p=root/'completion.json';sha=m.write_json(p,{'schema_version':1,'task_id':'P4E','test_gate':'closed_hash_only','plan_sha':'a'*64,'runs':rows})
            result=m.verify_completion(p,sha,data,'a'*64,None)
            self.assertEqual(len(result),9)
            slices={'popularity_thresholds':{'median':1,'p90':2},'item_like_counts':{str(i):1 for i in data['catalog']},'user_like_counts':{str(i):1 for i in range(4)}}
            baseline=m.slice_metrics([[q['target']] for q in data['queries']],data['queries'],data['catalog'],slices)
            summary=m.summary_session(p,sha,data,slices,{str(seed):baseline for seed in (42,2024,2025)},'a'*64,None,root/'summary.json')
            self.assertEqual(len(summary['runs']),9);self.assertEqual(set(summary['aggregate']),set(m.ABLATIONS))
            import unittest.mock as mock
            data['slices']=slices
            runtime={'operation_root':str(root),'prepared':str(root/'prepared'),'holdout':str(root/'holdout'),'raw_root':str(root),'core_selection':str(root/'core.json'),'watch_manifest':None,'watch_sha':None,'wall_seconds':600,'disk_bytes':1000000,'allowed_operations':['summarize'],'pilot_steps':30,'resume_steps':[7,20,30]}
            args=m.argparse.Namespace(command='summarize',output=str(root/'summary-cli.json'),manifest=str(p),manifest_sha=sha,plan_sha='a'*64,watch_sha=None,core_runs=str(root))
            def check():return {}
            check.samples=[]
            with mock.patch('torch.cuda.is_available',return_value=True),mock.patch.object(m,'operation_guard',return_value=check),mock.patch.object(m,'resource_guard',return_value=check),mock.patch.object(m,'real_data',return_value=(data,None,{})),mock.patch.object(m,'load_watch_artifacts',return_value=data),mock.patch.object(m,'load_core_metrics',return_value={str(seed):baseline for seed in (42,2024,2025)}):
                try:exit_code=m.dispatch_real_run(args,{'inputs':{'runtime':runtime}})
                except ValueError as error:self.fail(f'approved summary dispatcher missing: {error}')
                self.assertEqual(exit_code,0)
            self.assertTrue((root/'summary-cli.json').is_file())
            (root/'slot-09'/'verify.exit').write_text('2\n')
            with self.assertRaises(ValueError):m.verify_completion(p,sha,data,'a'*64,None)

    def test_config_rejects_boolean_numeric_and_synthetic_history_is_strictly_prior(self):
        m=self.module()
        with self.assertRaises(ValueError):m.validate_config({**m.default_config(),'gradient_clip':True})
        for row in m.synthetic_data()['watch_rows']:
            self.assertTrue(all(tuple(p)<tuple(row['target_position']) for p in row['history_positions']))

    def test_core_pairing_uses_pinned_predictions_and_same_train_slices(self):
        import unittest.mock as mock
        m=self.module();self.assertTrue(hasattr(m,'load_core_metrics'),'accepted core prediction reader missing')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=m.synthetic_data();data['query_sha256']='b'*64
            data['prepared']={'artifacts':{'catalog.json':'c'*64}}
            slices={'popularity_thresholds':{'median':1,'p90':2},'item_like_counts':{str(i):1 for i in data['catalog']},'user_like_counts':{str(i):1 for i in range(4)}}
            records=[]
            for seed in (42,2024,2025):
                directory=root/str(seed);directory.mkdir();epoch=directory/'epoch-1';epoch.mkdir()
                predictions=[[q['target']] for q in data['queries']]
                pred_sha=m.write_json(epoch/'predictions.json',predictions)
                config={'architecture':'id_mean','learning_rate':.001,'temperature':.15,'batch_size':512,'seed':seed}
                terminal={'status':'complete','binding':{'config':config,'validation_query_sha256':'b'*64,'catalog_sha256':'c'*64,'test_gate':'closed_hash_only'},'selection':{'best_epoch':1},'evaluations':[{'epoch':1,'result':{'artifacts':{'predictions.json':pred_sha}}}]}
                sha=m.write_json(directory/'terminal.json',terminal)
                records.append({'config':config,'manifest_logical_ref':f'{seed}/terminal.json','manifest_sha256':sha,'best_artifacts':{'predictions.json':pred_sha}})
            original=m.read_pinned
            def read(path,sha):
                if Path(path)==m.ROOT/'artifacts/p4d-core-manifest.json':
                    self.assertEqual(sha,'6ddccee458c3c66bb185d51527c0dc32981ae83949efe420eef97ed45881ac6e')
                    return {'runs':records}
                return original(path,sha)
            with mock.patch.object(m,'read_pinned',side_effect=read):
                result=m.load_core_metrics(root,data,slices)
                self.assertEqual(set(result),{'42','2024','2025'})
                self.assertEqual(result['42']['all']['recall@50'],1.)
                (root/'2025/epoch-1/predictions.json').write_text('[]')
                with self.assertRaises(ValueError):m.load_core_metrics(root,data,slices)

    def test_paired_summary_rejects_different_query_denominators(self):
        m=self.module()
        rows=[{'ablation':a,'seed':s,'metrics':{'all':{'count':4,'recall@50':.5}}} for a in m.ABLATIONS for s in (42,2024,2025)]
        cores={str(s):{'all':{'count':3,'recall@50':.5}} for s in (42,2024,2025)}
        with self.assertRaises(ValueError):m.summarize_metrics(rows,cores)

    def test_verifier_checks_budget_before_any_artifact_read(self):
        m=self.module()
        def expired():raise RuntimeError('approved budget expired')
        try:
            with self.assertRaisesRegex(RuntimeError,'approved budget expired'):
                m.verify_run('/nonexistent','a'*64,{},m.CORE_SHA,check=expired)
        except TypeError as error:self.fail(f'verifier budget callback missing: {error}')

    def test_watch_cli_rejects_shared_deadline_disk_and_path_before_input_loading(self):
        import unittest.mock as mock
        from test_retrieval_watch import WatchTests
        m=self.module();events,e,q=WatchTests().fixture()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);approved=root/'operation';approved.mkdir()
            runtime={'operation_root':str(approved),'prepared':str(root/'prepared'),'holdout':str(root/'holdout'),'raw_root':str(root),'core_selection':str(root/'core.json'),'watch_manifest':None,'watch_sha':None,'wall_seconds':600,'disk_bytes':1048576,'allowed_operations':['prepare-watch','verify-watch'],'pilot_steps':30,'resume_steps':[7,20,30]}
            plan={'inputs':{'runtime':runtime}}
            binding={'raw_records':[],'raw_producer':{'source':'frozen'},'prepared_sha':m.PREPARED_SHA,'query_sha':'2'*64,'catalog_sha':'3'*64,'core_sha':m.CORE_SHA,'consumer':m.source_identity(),'test_gate':'closed_hash_only'}
            data={'rows':[e],'queries':[q],'catalog':[0,2,3,4]}
            common=['--plan','private-plan.json','--plan-sha','f'*64,'--core-sha',m.CORE_SHA]
            with mock.patch.object(m,'verify_plan',return_value=plan),mock.patch.object(m,'real_data',return_value=(data,lambda:iter([(0,events)]),binding)) as loader,mock.patch.object(m,'resource_guard',return_value=lambda:None),mock.patch('time.time',return_value=100):
                with self.subTest(gate='outside'):
                    self.assertEqual(m.main(['prepare-watch',*common,'--output',str(root/'outside')]),2)
                    self.assertEqual(loader.call_count,0);self.assertFalse((root/'outside').exists())
                self.assertEqual(m.main(['prepare-watch',*common,'--output',str(approved/'watch')]),0)
                manifest=approved/'watch/watch-manifest.json';sha=m.file_sha256(manifest)
                loader.reset_mock()
                with mock.patch('time.time',return_value=701),self.subTest(gate='shared_deadline'):
                    self.assertEqual(m.main(['verify-watch',*common,'--manifest',str(manifest),'--manifest-sha',sha]),2)
                    self.assertEqual(loader.call_count,0)
                (approved/'other-stage.bin').write_bytes(b'x'*1048576)
                loader.reset_mock()
                with self.subTest(gate='cumulative_disk'):
                    self.assertEqual(m.main(['verify-watch',*common,'--manifest',str(manifest),'--manifest-sha',sha]),2)
                    self.assertEqual(loader.call_count,0)

    def test_producer_device_precision_is_bound_and_verifier_refuses_cpu_fallback(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=m.synthetic_data()
            run=m.run_session(root/'run',data,'user_balance',42,8,1)
            self.assertEqual(run['binding'].get('execution'),{'device':'cpu','precision':'fp32','tf32':False,'deterministic':True})
            p=root/'run'/run['manifest_name']
            value=m.read_pinned(p,run['manifest_sha']);value['binding']['execution']['device']='cuda'
            changed=root/'run'/'cuda-bound.json';sha=m.write_json(changed,value)
            import unittest.mock as mock
            with mock.patch('torch.cuda.is_available',return_value=False):
                with self.assertRaises(ValueError):m.verify_run(changed,sha,data,m.CORE_SHA)
