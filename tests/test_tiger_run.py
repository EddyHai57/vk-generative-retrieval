"""Runner gates use literal synthetic input evidence."""
import unittest,tempfile
from pathlib import Path
from vkgr import tiger_run
from vkgr.data_audit import canonical_bytes,file_sha256


class RunTests(unittest.TestCase):
    def test_config_freeze_strict_and_input_hash_drift(self):
        self.assertEqual(tiger_run.validate_config(dict(tiger_run.CONFIG)),tiger_run.CONFIG)
        c=dict(tiger_run.CONFIG);c['batch_size']=32
        with self.assertRaises(ValueError): tiger_run.validate_config(c)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json';p.write_bytes(canonical_bytes({'toy':1}));records=[{'role':'toy','path':str(p),'sha256':file_sha256(p)}]
            tiger_run.verify_inputs(records);p.write_bytes(canonical_bytes({'toy':2}))
            with self.assertRaisesRegex(ValueError,'hash'):tiger_run.verify_inputs(records)

    def test_train_temporal_oov_and_audit_contract(self):
        rows=[{'user_id':1,'history':[10,20],'history_positions':[[0,1],[0,2]],'target':30,'target_position':[0,3]}]
        tiger_run.validate_examples(rows,[10,20,30])
        rows[0]['history_positions'][1]=[0,3]
        with self.assertRaisesRegex(ValueError,'temporal'): tiger_run.validate_examples(rows,[10,20,30])
        rows[0]['history_positions'][1]=[0,2];rows[0]['target']=999
        with self.assertRaisesRegex(ValueError,'catalog'): tiger_run.validate_examples(rows,[10,20,30])

    def test_failure_cannot_publish_success_and_partial_status_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run_manifest.json'
            for stage in ({'step':99,'resume_from_step':0,'status':'time_limit'},{'step':1000,'resume_from_step':100,'status':'step_limit'}):
                with self.assertRaises(ValueError): tiger_run.publish_success(p,stage,{'validation':{'queries':511},'test':{'queries':512}})
                self.assertFalse(p.exists())
            with self.assertRaises(ValueError):tiger_run.publish_success(p,{'step':1000,'resume_from_step':0,'status':'step_limit'},{'validation':{'queries':512},'test':{'queries':512}})
            self.assertFalse(p.exists())

    def test_real_train_segment_deadline_safe_save_and_resume(self):
        from test_tiger import catalog
        from vkgr.tiger import SIDTokens
        from vkgr.manifest import read_json
        import torch
        c=dict(tiger_run.CONFIG);c.update(d_model=16,encoder_layers=1,decoder_layers=1,ff=32,batch_size=2)
        c['train_deadline_utc']='2099-01-01T00:00:00+00:00'
        rows=[{'history':[10],'target':20},{'history':[30],'target':40}]
        inputs={'toy':'c'*64}
        with tempfile.TemporaryDirectory() as d:
            first=tiger_run.train_session(Path(d),SIDTokens(catalog()),rows,c,inputs,end_step=2,device='cpu')
            self.assertEqual(first['step'],2);self.assertEqual(first['status'],'interrupted')
            second=tiger_run.train_session(Path(d),SIDTokens(catalog()),rows,c,inputs,end_step=5,device='cpu',resume=first)
            self.assertEqual(second['step'],5);self.assertEqual(second['resume_from_step'],2)
            self.assertEqual(first['started_at_utc'],second['started_at_utc'])
            c['train_deadline_utc']='2000-01-01T00:00:00+00:00'
            deadline_dir=Path(d)/'deadline';deadline_dir.mkdir()
            third=tiger_run.train_session(deadline_dir,SIDTokens(catalog()),rows,c,inputs,end_step=8,device='cpu')
            self.assertEqual(third['status'],'deadline');self.assertEqual(third['step'],0)
            self.assertTrue((deadline_dir/third['checkpoint_path']).is_file())
            self.assertFalse((Path(d)/'run_manifest.json').exists())

    def test_decode_item_metrics_oov_and_deadline_failure(self):
        from test_tiger import catalog
        from vkgr.tiger import SIDTokens,TigerModel
        import torch
        torch.manual_seed(4);codec=SIDTokens(catalog());m=TigerModel(10,16,4,1,1,32)
        queries={'validation':[{'user_id':1,'history':[10],'target':999,'oov':True}], 'test':[{'user_id':2,'history':[20],'target':30,'oov':False}]}
        c=dict(tiger_run.CONFIG);c['global_deadline_utc']='2099-01-01T00:00:00+00:00'
        results,artifacts=tiger_run.evaluate_model(m,codec,queries,c)
        self.assertEqual(results['validation']['metrics']['oov']['recall@20'],0.)
        self.assertEqual(results['test']['metrics']['all']['recall@20'],1.)
        for split in queries:
            self.assertEqual(set(artifacts[split]['predictions'][0]),{10,20,30,40})
            self.assertEqual(results[split]['invalid'],0);self.assertEqual(results[split]['deadend'],0);self.assertEqual(results[split]['collision'],0)
        c['global_deadline_utc']='2000-01-01T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'deadline'): tiger_run.evaluate_model(m,codec,queries,c)

    def test_runner_failure_keeps_evidence_without_success(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=root/'config.json';config.write_bytes(canonical_bytes(tiger_run.CONFIG));output=root/'run'
            with self.assertRaises(Exception):tiger_run.run('train',root/'missing-p3',root/'missing-p4',root/'missing-p5',config,output)
            self.assertFalse((output/'run_manifest.json').exists())
            self.assertTrue((output/'failure.json').is_file())

    def test_monitor_error_propagates_before_publish(self):
        def broken():raise OSError('resource sampling failed')
        monitor=tiger_run.ResourceMonitor(broken)
        with self.assertRaisesRegex(OSError,'resource'):monitor.check()

    def test_verify_rejects_fake_success_missing_bound_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run_manifest.json';p.write_bytes(canonical_bytes({'schema_version':1,'task_id':'P6','status':'success'}))
            with self.assertRaisesRegex(ValueError,'P6'):tiger_run.verify_run(p)

    def test_fresh_subprocess_verify_initializes_determinism_before_env_compare(self):
        import subprocess,sys,os
        from vkgr.training import seed_all
        seed_all(42)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run_manifest.json'
            value={'schema_version':1,'task_id':'P6','status':'success','training':{},'splits':{},'effective_config':{},
                   'config_path':str(Path(d)/'missing-config'),'config_sha256':'a'*64,'source_fingerprint':tiger_run.source_fingerprint(),
                   'environment':tiger_run.environment(),'inputs':[],'resource':{},'artifacts':{},'training_completion':'partial_time_bound'}
            p.write_bytes(canonical_bytes(value))
            code="from vkgr.tiger_run import verify_run; import sys;\ntry: verify_run(sys.argv[1])\nexcept FileNotFoundError: print('ENV_INITIALIZATION_PASSED')\n"
            result=subprocess.run([sys.executable,'-c',code,str(p)],text=True,capture_output=True,env=os.environ.copy())
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('ENV_INITIALIZATION_PASSED',result.stdout)
