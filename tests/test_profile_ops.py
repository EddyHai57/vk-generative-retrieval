import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
def ops():
    spec=importlib.util.spec_from_file_location('profile_ops',ROOT/'scripts/profile_ops.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class OpsTests(unittest.TestCase):
    def test_missing_ops_entry(self):self.assertTrue((ROOT/'scripts/profile_ops.py').is_file())
    def test_six_configs_fixed_modes_seeds(self):
        m=ops();configs=m.registered_configs();self.assertEqual(len(configs),6);self.assertEqual({c['arm'] for p,c in configs.values()},{'u2','u3'});self.assertEqual({c['seed'] for p,c in configs.values()},{42,2024,2025})
    def test_exact_vram_formula_invalid_values_rejected(self):
        m=ops();self.assertEqual(m.declared_vram(1000),1280)
        for v in [0,-1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):m.declared_vram(v)
    def test_verify_derivation_preserves_profile_and_manifest_binding(self):
        m=ops();job={'run_id':'p-profile-u2-s42','argv':['python','-B','-m',m.ENTRY,'train','--profile-manifest','/fixture/profile.json','--profile-sha','a'*64,'--output','/fixture/runs/p-profile-u2-s42/out'],'inputs':['/fixture/profile.json'],'disk_gib':4}
        v=m.derive_verify(job,'b'*64);self.assertEqual(v['run_id'],'p-profile-u2-s42-verify');self.assertEqual(v['argv'][v['argv'].index(m.ENTRY)+1],'verify');self.assertIn('a'*64,v['argv']);self.assertIn('b'*64,v['argv']);self.assertNotIn('--output',v['argv'])
    def test_failure_stop_and_raw_verified_gate(self):
        m=ops();self.assertEqual(m.next_action('failed',None,False,False),'ALERT');self.assertEqual(m.next_action('succeeded','succeeded',True,False,False),'ALERT');self.assertEqual(m.next_action('succeeded',None,False,False),'verify');self.assertEqual(m.next_action('succeeded','succeeded',True,False,True),'verified')
    def test_plan_requires_real_preflight_cuda_and_binds_six_jobs(self):
        import sys,time,copy
        m=ops();src={'git_commit':'a'*40,'git_dirty':False};shape={'batch':512,'history':50,'catalog':120415,'content_rows':120417,'supported':21496,'queries':2520}
        pf={'status':'preflight_ok','engineering':False,'test_read':False,'existing_checkpoint_read':False,'source':src,'inputs_sha256':'b'*64,'profile_manifest':'/fixture/profile/manifest.json','profile_sha256':'c'*64,'data_root':'/fixture/data','shape':shape,'entries':{}}
        for rid,(path,cfg) in m.registered_configs().items():pf['entries'][rid]={'arm':cfg['arm'],'seed':cfg['seed'],'config_rel':path.relative_to(m.ROOT).as_posix(),'config_sha256':m.sha(path),'expected_binding_sha256':'d'*64,'data_hashes':{'catalog':'e'*64}}
        proof={'full_shape':shape,'synthetic_verify':{'status':'verified'},'zero_start':True,'finite_gradients':True,'history_oracle':True,'measured_vram_mib':1000,'declared_vram_mib':1280}
        cu={'status':'cuda_engineering_verified','test_read':False,'existing_checkpoint_read':False,'real_training_rows_used':False,'source':src,'preflight_sha256':m.digest(pf),'arms':{arm:copy.deepcopy(proof) for arm in m.ARMS}}
        req={'root':str(m.ROOT),'runs_root':'/fixture/runs','data_root':'/fixture/data','inputs':'/fixture/inputs.json','python':sys.executable,'job_template':{'inputs':[],'env':[]},'pinned_manifest':'/fixture/pinned.json','created_epoch':time.time()}
        plan=m.make_plan(pf,cu,req);self.assertEqual(len(plan['entries']),6);self.assertEqual(plan['poll_seconds'],900);self.assertEqual(plan['deadline_epoch']-plan['created_epoch'],48*3600)
        for e in plan['entries']:self.assertIn(pf['profile_manifest'],e['job']['argv']);self.assertIn(pf['profile_manifest'],e['job']['inputs']);self.assertEqual(e['job']['vram_mib'],1280)
        for variant in [dict(pf,engineering=True),dict(pf,status='synthetic_ok')]:
            with self.assertRaises(ValueError):m.make_plan(variant,cu,req)
        broken=copy.deepcopy(cu);broken['arms']['u2']['zero_start']=False
        with self.assertRaises(ValueError):m.make_plan(pf,broken,req)

    def test_real_detached_ready_is_ppid1_and_no_publication_before_release(self):
        import json,os,subprocess,sys,tempfile,time,signal
        m=ops()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();q=root/'queue';runs=root/'runs';state=root/'state';state.mkdir();runs.mkdir()
            for name in ['pending','claimed','rejected']:(q/name).mkdir(parents=True)
            plan={'source':{},'entries':[{'job':{'run_id':rid}} for rid in m.RIDS],'runs_root':str(runs),'deadline_epoch':time.time()+60}
            req=root/'request.json';req.write_text(json.dumps({'plan':plan,'queue':str(q),'state':str(state)}))
            harness="import importlib.util,json,sys; s=importlib.util.spec_from_file_location('ops',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); r=json.load(open(sys.argv[2])); m.check_source=lambda _:None; m.prepare=lambda *a:{}; m.inspect_entry=lambda e,*a:{'run_id':e['job']['run_id'],'action':'verified','verified':True}; m.driver(r['plan'],r['queue'],'.',r['state'])"
            logpath=root/'harness.log';log=logpath.open('w');parent=subprocess.Popen([sys.executable,'-B','-c',harness,str(ROOT/'scripts/profile_ops.py'),str(req)],stdout=log,stderr=log);child=None
            try:
                parent.wait(timeout=5);until=time.monotonic()+5;ready=state/'driver-ready.json'
                while not ready.exists() and time.monotonic()<until:time.sleep(.02)
                self.assertTrue(ready.exists());value=json.loads(ready.read_text());child=value['pid'];self.assertEqual(value['ppid'],1)
                self.assertEqual(int(subprocess.check_output(['ps','-p',str(child),'-o','ppid='],text=True).strip()),1)
                self.assertEqual(list((q/'pending').glob('*.json')),[])
                m.new(state/'publish-release.json',{'driver_pid':child,'plan_sha256':m.digest(plan),'ready_sha256':m.sha(ready)})
                final=state/'summary-final.json';until=time.monotonic()+20
                while not final.exists() and time.monotonic()<until:time.sleep(.02)
                self.assertTrue(final.exists(),logpath.read_text());self.assertEqual(json.loads(final.read_text())['status'],'verified');self.assertEqual(len(list((q/'pending').glob('*.json'))),6)
            finally:
                if child:
                    try:os.kill(child,signal.SIGTERM)
                    except ProcessLookupError:pass
                log.close()

if __name__=='__main__':unittest.main()
