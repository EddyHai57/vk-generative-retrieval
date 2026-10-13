"""Independent G2 budget-chain and end-to-end replay oracles."""
import copy,datetime,hashlib,json,tempfile,unittest
from pathlib import Path
try:
    from vkgr import generative_experiment as ge
except ImportError:ge=None

def initial(max_steps=3,max_resumes=1):
    return dict(reserved_steps=0,completed_steps=0,consumed_wall_by_phase={'train':0.,'evaluate':0.,'verify':0.},resume_attempts=0,max_optimizer_steps=max_steps,max_resume_attempts=max_resumes,deadline_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=60)).isoformat(),parent_ledger_head=None)
class BudgetTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(ge,'G2 experiment/budget ledger missing')
    def test_write_ahead_reservation_and_completion_and_explicit_wall(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            ledger=ge.BudgetLedger(Path(d)/'ledger',initial(),{'source':'a'*64})
            ledger.reserve_step();self.assertEqual((ledger.snapshot()['reserved_steps'],ledger.snapshot()['completed_steps']),(1,0))
            with self.assertRaises(ValueError):ledger.reserve_step()
            ledger.complete_step(.25);ledger.charge_wall('evaluate',.5);ledger.charge_wall('verify',.125)
            state=ge.BudgetLedger.read_existing(ledger.root,ledger.head,{'source':'a'*64})
            self.assertEqual(state['budget']['consumed_wall_by_phase'],{'train':.25,'evaluate':.5,'verify':.125});self.assertEqual(state['budget']['completed_steps'],1)
    def test_resume_preserves_deadline_counters_and_charges_unfinished_reservation(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            binding={'source':'a'*64};parent=ge.BudgetLedger(Path(d)/'parent',initial(),binding)
            parent.reserve_step();parent.complete_step(.2);parent.reserve_step();before={p.name:p.read_bytes() for p in parent.root.iterdir()}
            child=ge.BudgetLedger.resume(parent.root,parent.head,binding,Path(d)/'child')
            state=child.snapshot();self.assertEqual((state['reserved_steps'],state['completed_steps'],state['resume_attempts']),(2,1,1));self.assertEqual(state['deadline_utc'],parent.snapshot()['deadline_utc'])
            self.assertEqual({p.name:p.read_bytes() for p in parent.root.iterdir()},before)
            child.reserve_step();child.complete_step(.3)
            with self.assertRaises(ValueError):child.reserve_step()
            with self.assertRaises(ValueError):ge.BudgetLedger.resume(child.root,child.head,binding,Path(d)/'again')
            self.assertFalse((Path(d)/'again').exists())
    def test_budget_caps_expiry_tamper_and_parent_head_reject(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d);binding={'source':'a'*64};ledger=ge.BudgetLedger(root/'ledger',initial(1),binding);ledger.reserve_step();ledger.complete_step(.01)
            with self.assertRaises(ValueError):ledger.reserve_step()
            with self.assertRaises(ValueError):ge.BudgetLedger.read_existing(ledger.root,'0'*64,binding)
            with self.assertRaises(ValueError):ge.BudgetLedger.read_existing(ledger.root,ledger.head,{'source':'b'*64})
            row=ledger.root/'ledger-000001.json';value=json.loads(row.read_bytes());value['budget']['reserved_steps']=0;row.write_text(json.dumps(value))
            with self.assertRaises(ValueError):ge.BudgetLedger.read_existing(ledger.root,ledger.head,binding)
            old=initial();old['deadline_utc']=(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=1)).isoformat()
            with self.assertRaises(ValueError):ge.BudgetLedger(root/'expired',old,binding)
            self.assertFalse((root/'expired').exists())
    def test_expired_checkpoint_can_be_preserved_but_not_resumed(self):
        from test_generative_training import codec,config,selection
        from vkgr import generative_training as gt
        import torch
        b=initial();b['deadline_utc']=(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=1)).isoformat()
        m=gt.make_model(codec(),config(),device='cpu');o=gt.make_optimizer(m,config());cursor=gt.EpochCursor(42,4);binding={'effective_config':config(),'fixture_sha256':'a'*64,'device':'cpu'}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path=Path(d)/'expired.pt';pin=gt.save_checkpoint(path,m,o,cursor,binding=binding,selection=selection(),budget=b)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),pin)
            with self.assertRaises(ValueError):ge.BudgetLedger(Path(d)/'new-run',b,binding)
class ContextTests(unittest.TestCase):
    def test_synthetic_full_event_statistics_and_sealed_context(self):
        self.assertTrue(hasattr(ge,'prepare_synthetic'),'G2 prepared context missing')
        root=Path(__file__).resolve().parents[1];fixture=root/'tests/fixtures/g2_synthetic.json'
        pin=hashlib.sha256(fixture.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            context=ge.prepare_synthetic(root,root/'artifacts/generative-contract-manifest.json','40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d',fixture,pin,Path(d)/'prepared')
            self.assertEqual(len(context.examples),4);self.assertEqual(len(context.queries),7)
            self.assertEqual(context.slice_definition.item_counts[10],3)
            self.assertEqual(context.slice_definition.item_counts[60],0)
            self.assertEqual(context.slice_definition.user_counts[2],1)
            self.assertEqual(context.slice_definition.thresholds,(2,3))
            self.assertEqual(sum(q['oov'] for q in context.queries),1)
            q=context.queries;q[0]['history'].append(999);self.assertNotIn(999,context.queries[0]['history'])
            context.check()
            altered=context.queries;altered[0]['history']=[60]
            from vkgr.data_audit import canonical_bytes
            with self.assertRaises((AttributeError,ValueError)):
                context._queries=canonical_bytes(altered);context.check()
            with self.assertRaises((AttributeError,ValueError)):
                context._binding=canonical_bytes({});context.check()
            with self.assertRaises(ValueError):ge.RunContext()
            context.codec.offsets[0]+=1
            with self.assertRaises(ValueError):context.check()
    def test_test_and_real_gates_run_before_any_file_read(self):
        self.assertTrue(hasattr(ge,'prepare_context'),'G2 role gate missing')
        from unittest.mock import patch
        with patch('vkgr.generative_data.load_contract',side_effect=AssertionError('must not read')):
            for kwargs in ({'synthetic':False},{'synthetic':True,'validation_path':Path('/test-queries.json')}):
                with self.assertRaises(ValueError):ge.prepare_context(Path('/missing'),Path('/missing'),'0'*64,**kwargs)

class RunnerTests(unittest.TestCase):
    def test_two_epochs_continuous_and_step_boundary_resume_exact(self):
        self.assertTrue(hasattr(ge,'run_train'),'G2 runner missing')
        from test_generative_training import config
        root=Path(__file__).resolve().parents[1];fixture=root/'tests/fixtures/g2_synthetic.json';pin=hashlib.sha256(fixture.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            out=Path(d);context=ge.prepare_synthetic(root,root/'artifacts/generative-contract-manifest.json','40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d',fixture,pin,out/'prepared')
            limits={'max_epochs':2,'max_wall_seconds':60,'max_resume_attempts':1,'microbatch':2,'max_optimizer_steps':2}
            first=ge.run_train(context,config(),output=out/'first',limits=limits,stop_after_step=1)
            self.assertEqual(first['manifest']['status'],'interrupted');self.assertIsNone(first['manifest']['best'])
            with self.assertRaises(ValueError):ge.verify_run(root,first['manifest_path'],first['manifest_sha256'],replay=False)
            resumed=ge.run_train(context,config(),output=out/'resumed',limits=limits,resume_path=Path(first['manifest']['last']['path']),resume_sha256=first['manifest']['last']['sha256'],parent_manifest=first['manifest_path'],parent_sha256=first['manifest_sha256'])
            full=ge.run_train(context,config(),output=out/'full',limits=limits)
            self.assertEqual(resumed['manifest']['status'],'complete')
            self.assertEqual(resumed['manifest']['state_sha256'],full['manifest']['state_sha256'])
            self.assertEqual(resumed['manifest']['selection'],full['manifest']['selection'])
            self.assertEqual(resumed['manifest']['epoch_metrics'],full['manifest']['epoch_metrics'])
            self.assertEqual(resumed['manifest']['budget']['resume_attempts'],1)
            self.assertEqual(resumed['manifest']['budget']['completed_steps'],2)
            self.assertEqual(resumed['manifest']['budget']['deadline_utc'],first['manifest']['budget']['deadline_utc'])
            bad=config();bad['learning_rate']=.0003
            with self.assertRaises(ValueError):ge.run_train(context,bad,output=out/'wrong-config',limits=limits,resume_path=Path(first['manifest']['last']['path']),resume_sha256=first['manifest']['last']['sha256'],parent_manifest=first['manifest_path'],parent_sha256=first['manifest_sha256'])
            self.assertFalse((out/'wrong-config').exists())
    def test_synthetic_budget_cannot_expand_or_drop_parent_pins(self):
        self.assertTrue(hasattr(ge,'run_train'),'G2 runner missing')
        with self.assertRaises(ValueError):ge.run_train({}, {},output=Path('/must-not-create'),limits={'max_epochs':30})

class CLITests(unittest.TestCase):
    def test_actual_subprocess_prepare_train_resume_evaluate_replay_select(self):
        self.assertTrue(hasattr(ge,'main'),'G2 CLI missing')
        import subprocess,sys,os
        root=Path(__file__).resolve().parents[1];fixture=root/'tests/fixtures/g2_synthetic.json';pin=hashlib.sha256(fixture.read_bytes()).hexdigest()
        commands=[]
        def invoke(*args):
            result=subprocess.run([sys.executable,'-B','-m','vkgr.generative_experiment',*map(str,args)],cwd=root,env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'1'},capture_output=True,text=True,timeout=30)
            commands.append((args,result.returncode));self.assertEqual(result.returncode,0,result.stderr)
            return json.loads(result.stdout)
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            out=Path(d)
            prepared=invoke('prepare','--synthetic','--fixture',fixture,'--fixture-sha',pin,'--contract',root/'artifacts/generative-contract-manifest.json','--contract-sha','40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d','--output',out/'prepared')
            first=invoke('train','--synthetic','--prepared',prepared['manifest_path'],'--prepared-sha',prepared['manifest_sha256'],'--config-index','0','--seed','42','--microbatch','2','--max-epochs','2','--max-wall-seconds','60','--stop-after-step','1','--output',out/'first')
            parent=json.loads(Path(first['manifest_path']).read_bytes());last=parent['last']
            final=invoke('resume','--synthetic','--checkpoint',last['path'],'--checkpoint-sha',last['sha256'],'--parent-manifest',first['manifest_path'],'--parent-sha',first['manifest_sha256'],'--output',out/'resumed')
            evaluation=invoke('evaluate','--synthetic','--run-manifest',final['manifest_path'],'--run-sha',final['manifest_sha256'],'--shard-size','3','--output',out/'evaluation')
            replay=invoke('verify','--manifest',final['manifest_path'],'--sha256',final['manifest_sha256'],'--operation-head',evaluation['operation_head'],'--replay')
            selected=invoke('select','--synthetic','--run-manifest',final['manifest_path'],'--run-sha',final['manifest_sha256'],'--operation-head',replay['operation_head'],'--output',out/'selection')
            self.assertTrue(replay['replay']);self.assertEqual(replay['status'],'verified')
            result=json.loads(Path(evaluation['manifest_path']).read_bytes());self.assertEqual(result['aggregate']['metrics']['all_eligible']['count'],7)
            chosen=json.loads(Path(selected['manifest_path']).read_bytes());self.assertEqual(chosen['boundary'],'synthetic selection oracle; no core winner')
            self.assertEqual(len(commands),6)
            # Keep every aggregate unchanged, but tamper one OOV ranking.
            from vkgr.data_audit import canonical_bytes
            from vkgr import generative_evaluation as ev
            original=json.loads(Path(final['manifest_path']).read_bytes())
            receipt=original['evaluations'][0];evaluation_path=Path(receipt['manifest_path']);evaluation=json.loads(evaluation_path.read_bytes())
            entry=evaluation['shards'][0];shard_path=Path(entry['path']);shard=json.loads(shard_path.read_bytes())
            shard['rows'][1]['items'].reverse();shard_path.write_bytes(canonical_bytes(shard));entry['sha256']=hashlib.sha256(shard_path.read_bytes()).hexdigest()
            evaluation['computation_sha256']=ev._computation(shard['rows'],evaluation['aggregate'])
            evaluation_path.write_bytes(canonical_bytes(evaluation));receipt['manifest_sha256']=hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
            forged=out/'prediction-forged.json';forged.write_bytes(canonical_bytes(original));forged_sha=hashlib.sha256(forged.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):ge.verify_run(root,forged,forged_sha,replay=True)
            # A forged best receipt cannot pass structural verification merely
            # because its containing run manifest has a newly supplied hash.
            bad=json.loads(Path(final['manifest_path']).read_bytes());bad['best']['epoch']=30
            path=out/'forged.json';path.write_text(json.dumps(bad));bad_sha=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):ge.verify_run(root,path,bad_sha,replay=False)

class EvaluationRecoveryBudgetTests(unittest.TestCase):
    def test_truncated_journal_cannot_be_reopened_as_first_operation(self):
        from test_generative_training import config
        root=Path(__file__).resolve().parents[1];fixture=root/'tests/fixtures/g2_synthetic.json';pin=hashlib.sha256(fixture.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            out=Path(d);context=ge.prepare_synthetic(root,root/'artifacts/generative-contract-manifest.json','40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d',fixture,pin,out/'prepared')
            run=ge.run_train(context,config(),output=out/'run',limits={'max_epochs':2,'max_wall_seconds':60,'max_resume_attempts':1,'microbatch':2,'max_optimizer_steps':2})
            ge.verify_run(root,run['manifest_path'],run['manifest_sha256'],replay=False)
            for record in (out/'run'/'operations').glob('operation-*.json'):record.unlink()
            with self.assertRaises(ValueError):ge.verify_run(root,run['manifest_path'],run['manifest_sha256'],replay=False)

    def test_actual_sigterm_during_decode_publishes_pending_training_receipt(self):
        import subprocess,sys,os
        root=Path(__file__).resolve().parents[1];fixture=root/'tests/fixtures/g2_synthetic.json';pin=hashlib.sha256(fixture.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            out=Path(d);ge.prepare_synthetic(root,root/'artifacts/generative-contract-manifest.json','40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d',fixture,pin,out/'prepared')
            prepared=out/'prepared'/'manifest.json';prepared_sha=hashlib.sha256(prepared.read_bytes()).hexdigest()
            args=['train','--synthetic','--prepared',str(prepared),'--prepared-sha',prepared_sha,'--shard-size','3','--microbatch','2','--output',str(out/'signal-run')]
            script="""import os,signal,sys,json
from vkgr import tiger,generative_experiment as ge
original=tiger.decode_history
once=[False]
def decode(*args,**kwargs):
    result=original(*args,**kwargs)
    if not once[0]:
        once[0]=True
        os.kill(os.getpid(),signal.SIGTERM)
    return result
tiger.decode_history=decode
raise SystemExit(ge.main(json.loads(sys.argv[1])))
"""
            result=subprocess.run([sys.executable,'-B','-c',script,json.dumps(args)],cwd=root,env={**os.environ,'OMP_NUM_THREADS':'1','PYTHONPATH':'src'},capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            receipt=json.loads(result.stdout);run=json.loads(Path(receipt['manifest_path']).read_bytes())
            self.assertEqual(run['status'],'interrupted');self.assertIsNotNone(run['pending_evaluation'])
            self.assertEqual(json.loads(Path(run['pending_evaluation']['manifest_path']).read_bytes())['completed_queries'],3)
            last=run['last']
            resumed=subprocess.run([sys.executable,'-B','-m','vkgr.generative_experiment','resume','--synthetic','--checkpoint',last['path'],'--checkpoint-sha',last['sha256'],'--parent-manifest',receipt['manifest_path'],'--parent-sha',receipt['manifest_sha256'],'--output',str(out/'resumed')],cwd=root,env={**os.environ,'OMP_NUM_THREADS':'1','PYTHONPATH':'src'},capture_output=True,text=True,timeout=30)
            self.assertEqual(resumed.returncode,0,resumed.stderr);final=json.loads(Path(json.loads(resumed.stdout)['manifest_path']).read_bytes());self.assertEqual(final['status'],'complete')

    def test_training_validation_partial_receipt_resumes_missing_shards(self):
        from test_generative_training import config
        root=Path(__file__).resolve().parents[1];fixture=root/'tests/fixtures/g2_synthetic.json';pin=hashlib.sha256(fixture.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            out=Path(d);context=ge.prepare_synthetic(root,root/'artifacts/generative-contract-manifest.json','40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d',fixture,pin,out/'prepared')
            limits={'max_epochs':2,'max_wall_seconds':60,'max_resume_attempts':1,'microbatch':2,'max_optimizer_steps':2}
            first=ge.run_train(context,config(),output=out/'first',limits=limits,evaluation_shard_size=3,stop_after_evaluation_shards=1)
            self.assertEqual(first['manifest']['status'],'interrupted');pending=first['manifest']['pending_evaluation']
            self.assertEqual(json.loads(Path(pending['manifest_path']).read_bytes())['completed_queries'],3)
            prefix=Path(pending['manifest_path']).parent/'shard-000000-000003.json';prefix_bytes=prefix.read_bytes()
            resumed=ge.run_train(context,config(),output=out/'resumed',limits=limits,evaluation_shard_size=3,resume_path=Path(first['manifest']['last']['path']),resume_sha256=first['manifest']['last']['sha256'],parent_manifest=first['manifest_path'],parent_sha256=first['manifest_sha256'])
            full=ge.run_train(context,config(),output=out/'full',limits=limits,evaluation_shard_size=3)
            self.assertEqual(resumed['manifest']['state_sha256'],full['manifest']['state_sha256']);self.assertEqual(resumed['manifest']['epoch_metrics'],full['manifest']['epoch_metrics'])
            self.assertEqual(prefix.read_bytes(),prefix_bytes)
            evaluation=json.loads(Path(resumed['manifest']['evaluations'][0]['manifest_path']).read_bytes())
            self.assertEqual([r['start'] for r in evaluation['shards']],[0,3,6]);self.assertEqual(evaluation['shards'][0]['path'],str(prefix))
            from unittest.mock import patch
            from vkgr import generative_evaluation as ev
            original=ev.evaluate_queries
            def interruption(*args,**kwargs):
                kwargs['stop_after_shards']=1;original(*args,**kwargs)
                raise InterruptedError('after immutable first shard')
            with patch('vkgr.generative_evaluation.evaluate_queries',side_effect=interruption):
                interrupted=ge.run_train(context,config(),output=out/'exception-stop',limits=limits,evaluation_shard_size=3)
            self.assertEqual(interrupted['manifest']['status'],'interrupted')
            self.assertIsNotNone(interrupted['manifest']['pending_evaluation'])
            recovered=ge.run_train(context,config(),output=out/'exception-resume',limits=limits,evaluation_shard_size=3,resume_path=Path(interrupted['manifest']['last']['path']),resume_sha256=interrupted['manifest']['last']['sha256'],parent_manifest=interrupted['manifest_path'],parent_sha256=interrupted['manifest_sha256'])
            self.assertEqual(recovered['manifest']['state_sha256'],full['manifest']['state_sha256'])
    def test_independent_evaluate_and_verify_share_persisted_deadline_wall(self):
        from test_generative_training import config
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1];fixture=root/'tests/fixtures/g2_synthetic.json';pin=hashlib.sha256(fixture.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            out=Path(d);context=ge.prepare_synthetic(root,root/'artifacts/generative-contract-manifest.json','40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d',fixture,pin,out/'prepared')
            limits={'max_epochs':2,'max_wall_seconds':60,'max_resume_attempts':1,'microbatch':2,'max_optimizer_steps':2};run=ge.run_train(context,config(),output=out/'run',limits=limits)
            first=ge.verify_run(root,run['manifest_path'],run['manifest_sha256'],replay=False)
            journal=out/'run'/'operations';self.assertTrue(journal.exists(),'verification wall must be persisted')
            initial_files={p.name:p.read_bytes() for p in journal.iterdir()}
            second=ge.verify_run(root,run['manifest_path'],run['manifest_sha256'],replay=False,expected_operation_head=first['operation_head'])
            self.assertGreater(second['budget']['consumed_wall_by_phase']['verify'],first['budget']['consumed_wall_by_phase']['verify'])
            self.assertGreater(len(list(journal.iterdir())),len(initial_files))
            self.assertTrue(all((journal/name).read_bytes()==data for name,data in initial_files.items()))
            # Only temporary journal records are removed; the caller keeps the
            # externally received head, and the first-operation marker remains.
            for record in journal.glob('operation-*.json'):record.unlink()
            with self.assertRaises(ValueError):ge.verify_run(root,run['manifest_path'],run['manifest_sha256'],replay=False,expected_operation_head=second['operation_head'])
            with self.assertRaises(ValueError):ge.verify_run(root,run['manifest_path'],run['manifest_sha256'],replay=False)
            real_datetime=datetime.datetime
            class Future(real_datetime):
                @classmethod
                def now(cls,tz=None):return real_datetime.now(tz)+datetime.timedelta(seconds=61)
            with patch('vkgr.generative_experiment.datetime.datetime',Future):
                result=ge.main(['evaluate','--synthetic','--run-manifest',str(run['manifest_path']),'--run-sha',run['manifest_sha256'],'--output',str(out/'expired-evaluation')])
                self.assertEqual(result,1);self.assertFalse((out/'expired-evaluation').exists())
                with self.assertRaises(ValueError):ge.verify_run(root,run['manifest_path'],run['manifest_sha256'],replay=False)

if __name__=='__main__':unittest.main()
