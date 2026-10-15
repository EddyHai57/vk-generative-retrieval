import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

def module():
    p=Path(__file__).with_name('seqm1_night.py')
    if not p.exists():raise AssertionError('night batch implementation missing')
    s=importlib.util.spec_from_file_location('night',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class NightChecks(unittest.TestCase):
    def setUp(self):self.m=module()
    def state(self,train='running',verify=None,verified=False,rejected=False):
        return {'train_status':train,'verify_status':verify,'verified':verified,'rejected':rejected}
    def test_canary_never_uses_recall_and_requires_actual_verified(self):
        m=self.m;s={r:self.state() for r in m.MEANS}
        self.assertEqual(m.gate(s),'WAIT')
        s[m.MEANS[0]]=self.state('succeeded','succeeded',False)
        self.assertEqual(m.gate(s),'ALERT')
        s[m.MEANS[0]]=self.state('succeeded','succeeded',True)
        self.assertEqual(m.gate(s),'RELEASE')
        s[m.MEANS[1]]=self.state('failed')
        self.assertEqual(m.gate(s),'ALERT')
    def test_pending_verify_does_not_mean_science_failure(self):
        m=self.m;s={r:self.state('succeeded','pending') for r in m.MEANS}
        self.assertEqual(m.gate(s),'WAIT')
        s[m.MEANS[0]]=self.state('succeeded','failed')
        self.assertEqual(m.gate(s),'ALERT')
    def test_rejected_mean_never_releases(self):
        m=self.m;s={r:self.state('succeeded','succeeded',True) for r in m.MEANS}
        s[m.MEANS[2]]['rejected']=True;self.assertEqual(m.gate(s),'ALERT')
    def test_launcher_and_wall_limit_failures_are_terminal(self):
        for terminal in ('launch_failed','wall_cap_exceeded'):
            self.assertTrue(self.m.settled(self.state(terminal),True))
            self.assertTrue(self.m.settled(self.state('succeeded',terminal),True))
    def test_exact_fixed_fifteen_and_seed_arm_only_config_changes(self):
        m=self.m;self.assertEqual(len(m.RIDS),15);self.assertEqual(len(m.DIAGS),12)
        a={'seed':42,'arm':'mean','temperature':.1};b={**a,'seed':2024,'arm':'pure'}
        m.check_config(a,b,'pure',2024)
        with self.assertRaises(ValueError):m.check_config(a,{**b,'temperature':.2},'pure',2024)
    def test_derive_train_and_verify_keep_frozen_p_environment(self):
        m=self.m
        parent={'run_id':'p-old','seed':42,'cwd':'/old','config':'/old/config','inputs':['spec','history','side'],
                'argv':['python','-B','-m','old','train','--inputs',str(m.BASE_INPUT),'--inputs-sha',m.BASE_SHA,'--config','old','--config-sha','old','--output','old'],
                'cpu_cores':1,'ram_mib':9216,'vram_mib':3072,'disk_gib':8,'wall_hours':8.,'kind':'real','no_admission':False,'env':['OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=src'],
                'pinned_manifest':None,'pinned_files':['docs/p3c_validation.md='+m.P3C],'schema':'vkgr-queue-job-v1','stop_mode':'signal'}
        rid='p-seqm1-mean-s2024';job=m.derive_train(parent,rid,'a'*64)
        self.assertEqual((job['seed'],job['cpu_cores'],job['vram_mib']),(2024,1,1024));self.assertFalse(job['no_admission'])
        v=m.derive_verify(job,'b'*64)
        self.assertEqual(v['run_id'],rid+'-verify');self.assertEqual(v['argv'][4],'verify');self.assertNotIn('--output',v['argv'])
        self.assertIn('--config-sha',v['argv']);self.assertEqual(v['inputs'][-1],str(m.R/'runs'/rid/'out/manifest.json'))
    def test_dispatch_publishes_all_twelve_once_after_canary(self):
        m=self.m
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);jobs=[{'run_id':r} for r in m.RIDS];plan={'jobs':jobs};published=[]
            states={r:self.state('succeeded','succeeded',True) for r in m.MEANS}
            with patch.object(m,'mean_states',return_value=states),patch.object(m,'publish_job',side_effect=lambda j:published.append(j['run_id'])):
                result=m.dispatch_once(folder,plan)
            self.assertEqual(result['status'],'released');self.assertEqual(published,list(m.DIAGS))
            with patch.object(m,'publish_job',side_effect=AssertionError('duplicate')):
                self.assertEqual(m.dispatch_once(folder,plan)['status'],'released')
    def test_dispatch_failure_writes_alert_and_never_publishes(self):
        m=self.m
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);s={r:self.state() for r in m.MEANS};s[m.MEANS[0]]=self.state('failed')
            with patch.object(m,'mean_states',return_value=s),patch.object(m,'publish_job',side_effect=AssertionError('unsafe publish')):
                result=m.dispatch_once(folder,{'jobs':[{'run_id':r} for r in m.RIDS]})
            self.assertEqual(result['status'],'ALERT');self.assertTrue((folder/'dispatch-terminal.json').exists())
    def test_rejection_during_dispatch_is_not_retried(self):
        m=self.m
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);s={r:self.state('succeeded','succeeded',True) for r in m.MEANS};called=[]
            def publish(j):called.append(j['run_id']);raise FileExistsError('rejected or duplicate')
            with patch.object(m,'mean_states',return_value=s),patch.object(m,'publish_job',side_effect=publish):
                result=m.dispatch_once(folder,{'jobs':[{'run_id':r} for r in m.RIDS]})
            self.assertEqual(result['status'],'ALERT');self.assertEqual(len(called),1)
    def test_verifier_failed_train_is_not_restarted_or_verified(self):
        m=self.m
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with patch.object(m,'snapshot',return_value={'run_id':'p-fixture','train_status':'failed','verify_status':None,'verified':False,'rejected':False}),patch.object(m,'publish_job',side_effect=AssertionError('retry')):
                result=m.verify_once(folder,{'jobs':[{'run_id':'p-fixture'}]})
            self.assertEqual(result['runs'][0]['train_status'],'failed');self.assertTrue(result['runs'][0]['unaccepted'])

    def fixture(self,root,rid,job,accepted=True):
        m=self.m;d=root/'runs'/rid/'out';d.mkdir(parents=True)
        binding={'source':{'git_commit':m.PUB},**{k:'d'*64 for k in ('rows_sha256','queries_sha256','catalog_sha256','support_sha256','content_sha256')}}
        value={'status':'complete','config':json.loads(Path(job['config']).read_text()),'binding':binding,
               'selection':{'best_epoch':1,'epoch':4,'best_metric':[1/2520,.001]},
               'evaluations':[{'epoch':1,'diagnostics':{'branch_ratio_median':.08}}],
               'ablation_g0':{'delta_recall@50':1/2520}}
        (d/'manifest.json').write_text(json.dumps(value));(d.parent/'terminal.json').write_text(json.dumps({'status':'succeeded','exit_code':0}))
        v=m.derive_verify(job,m.sha(d/'manifest.json'));vdir=root/'runs'/v['run_id'];(vdir/'logs').mkdir(parents=True)
        (vdir/'terminal.json').write_text(json.dumps({'status':'succeeded','exit_code':0}))
        (vdir/'logs/stdout.log').write_text(json.dumps({'status':'verified' if accepted else 'complete'})+'\n')
        (root/'queue/claimed'/('0000-'+v['run_id']+'.json')).write_text(json.dumps(v))

    def test_filesystem_canary_to_twelve_jobs_and_real_verifier_binding(self):
        m=self.m
        import sys
        with tempfile.TemporaryDirectory() as tmp,patch.object(sys,'path',[str(Path(__file__).parent),*sys.path]):
            import phase2_validation_read as reader
            root=Path(tmp);folder=root/'ops';folder.mkdir();(root/'runs').mkdir()
            for part in ('pending','claimed','rejected'):(root/'queue'/part).mkdir(parents=True)
            jobs=[]
            for rid in m.RIDS:
                cfg=root/(rid+'.json');cfg.write_text(json.dumps({'arm':rid.split('-')[2]}))
                jobs.append({'run_id':rid,'config':str(cfg),'argv':['python','-B','-m','runner','train','--config-sha',m.sha(cfg),'--output','out'],
                             'inputs':[],'expected_binding_sha256':None,'expected_data_hashes':{}})
            with patch.object(m,'R',root),patch.object(m,'Q',root/'queue'),patch.object(reader,'ROOT',root):
                self.fixture(root,m.MEANS[0],jobs[0]);result=m.dispatch_once(folder,{'jobs':jobs})
                self.assertEqual(result['status'],'released')
                queued=list((root/'queue/pending').glob('*.json'));self.assertEqual(len(queued),12)
                self.assertTrue(all(not any(k.startswith('expected_') for k in json.loads(p.read_text())) for p in queued))
                row=m.snapshot(jobs[0]);self.assertTrue(row['verified']);self.assertEqual(row['ablation_delta_hits'],1)
                row2=m.verify_once(folder,{'jobs':jobs});self.assertEqual(len(row2['runs']),15)
                self.assertTrue(row2['runs'][0]['verified'])

    def test_filesystem_supervisor_exit_zero_without_verified_blocks_dispatch(self):
        m=self.m
        import sys
        with tempfile.TemporaryDirectory() as tmp,patch.object(sys,'path',[str(Path(__file__).parent),*sys.path]):
            import phase2_validation_read as reader
            root=Path(tmp);folder=root/'ops';folder.mkdir();(root/'runs').mkdir()
            for part in ('pending','claimed','rejected'):(root/'queue'/part).mkdir(parents=True)
            jobs=[]
            for rid in m.RIDS:
                cfg=root/(rid+'.json');cfg.write_text('{}');jobs.append({'run_id':rid,'config':str(cfg),'argv':['python','-B','-m','runner','train','--config-sha',m.sha(cfg),'--output','out'],'inputs':[]})
            with patch.object(m,'R',root),patch.object(m,'Q',root/'queue'),patch.object(reader,'ROOT',root):
                self.fixture(root,m.MEANS[0],jobs[0],accepted=False)
                result=m.dispatch_once(folder,{'jobs':jobs});self.assertEqual(result['status'],'ALERT')
                self.assertEqual(list((root/'queue/pending').glob('*.json')),[])

if __name__=='__main__':unittest.main()
