import importlib.util
import unittest
class EntryTests(unittest.TestCase):
    def test_missing_prepare_entry(self):self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_profile_prepare'))
    def test_missing_training_entry(self):self.assertIsNotNone(importlib.util.find_spec('vkgr.retrieval_profile_run'))
    def test_new_config_projects_exact_mean_conditions(self):
        from vkgr import retrieval_profile_run as run
        from vkgr import retrieval_seqm1_run as old
        for arm in ['u2','u3']:
            c=run.default_config(arm);run.validate_config(c);projected={**c,'schema':old.SCHEMA,'task_id':old.TASK,'arm':'mean'};self.assertEqual(projected,old.default_config())
            for key,val in [('temperature',.2),('arm','mean'),('seed',7),('batch_size',16)]:
                with self.assertRaises(ValueError):run.validate_config({**c,key:val})
    def test_synthetic_train_verify_uses_profile_features(self):
        import torch,tempfile
        from pathlib import Path
        from vkgr import retrieval_profile_run as run
        torch.set_num_threads(1)
        for arm in ['u2','u3']:
            data=run.synthetic_data();cfg={**run.default_config(arm),'batch_size':16,'max_epochs':6,'min_epochs':2}
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/'run';m=run.run_session(out,data,cfg)
                proof=run.verify_run(out/'manifest.json',m['manifest_sha256'],data,cfg)
                self.assertEqual(proof['status'],'verified');self.assertIn('profile_ids_sha256',m['binding']);self.assertEqual(m['binding']['profile_arm'],arm)
    def test_profile_prepare_roundtrip_rejects_fit_binding_and_order_tampering(self):
        import copy,tempfile
        from pathlib import Path
        from test_retrieval_profile import Profiles
        from vkgr import profile_features as f,retrieval_profile_prepare as prep
        events,rows,qs=Profiles().world();profiles=f.build_profiles([(0,events)],rows,qs);bounds=f.fit_bounds(profiles['train'])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp).resolve();ps=prep.write_new(out/'profiles.json',profiles);bs=prep.write_new(out/'quantiles.json',bounds)
            m={'schema':prep.SCHEMA,'status':'complete','inputs_sha256':'a'*64,'artifacts':{'profiles.json':ps,'quantiles.json':bs},'fit_input_sha256':bounds['train_snapshots_sha256'],'profile_script_sha256':prep.file_sha256(Path(f.__file__)),'fit_script_sha256':prep.file_sha256(Path(prep.__file__))}
            ms=prep.write_new(out/'manifest.json',m);loaded=prep.load_profiles(out/'manifest.json',ms,{'rows':rows,'queries':qs},'a'*64)
            self.assertIn('profile_ids',loaded['rows'][0])
            with self.assertRaises(ValueError):prep.load_profiles(out/'manifest.json',ms,{'rows':list(reversed(rows)),'queries':qs},'a'*64)
            with self.assertRaises(ValueError):prep.load_profiles(out/'manifest.json',ms,{'rows':rows,'queries':qs},'b'*64)
            bad={**m,'fit_input_sha256':'0'*64};badsha=prep.write_new(out/'bad.json',bad)
            with self.assertRaises(ValueError):prep.load_profiles(out/'bad.json',badsha,{'rows':rows,'queries':qs},'a'*64)

    def test_full_synthetic_cli_under_runtime_guard(self):
        import json,os,subprocess,sys,tempfile
        from pathlib import Path
        from vkgr import retrieval_profile_run as run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();cfg=root/'config.json';cfg.write_text(json.dumps({**run.default_config('u2'),'batch_size':16,'max_epochs':6,'min_epochs':2}))
            cp=subprocess.run([sys.executable,'-B','-m','vkgr.retrieval_profile_run','synthetic-train','--config',str(cfg),'--output',str(root/'run')],capture_output=True,text=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},timeout=30)
            self.assertEqual(cp.returncode,0,cp.stderr);self.assertEqual(json.loads(cp.stdout)['status'],'complete')

if __name__=='__main__':unittest.main()
