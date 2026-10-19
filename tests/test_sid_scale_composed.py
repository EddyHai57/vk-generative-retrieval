"""Frozen two-component wiring tests, no formal calibration or scientific inputs."""
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
def tool():
    p=ROOT/'scripts/sid_scale_composed.py'
    assert p.is_file(),'missing formal component wiring'
    s=importlib.util.spec_from_file_location('cost_composed',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class ComposedTests(unittest.TestCase):
    def rows(self):
        return [{'method':'core','status':'COMPLETE','core_seconds_per_step':.01,'steps':3000,'usage_calls':0},
                *[{'method':'usage','status':'COMPLETE','n':n,'seconds':n/1000000,'usage_calls':1} for n in (120415,250000,300000)]]
    def histories(self):
        return [{'run_id':'t1','n':120415,'steps':300000,'k':256,'d':32,'levels':3,'usage_interval':2000,'wall_seconds':4329.242808818817},
                {'run_id':'t2','n':120415,'steps':300000,'k':16,'d':32,'levels':3,'usage_interval':2000,'wall_seconds':4282.5187928676605}]
    def test_both_step_scopes_and_actual_historical_interval_use_separate_components(self):
        t=tool();p=t.projection(self.rows(),self.histories(),{'usage_interval':10000})
        self.assertTrue(p['rq_selfcheck_passed'])
        self.assertAlmostEqual(p['usage_regression']['r_squared'],1.)
        self.assertEqual(p['historical_selfchecks'][0]['composition']['n_usage'],150)
        full=p['rq_targets'][-1]
        self.assertEqual(full['fixed_steps']['n_usage'],30)
        self.assertEqual(full['fixed_visits']['n_usage'],3106)
        self.assertEqual(full['fixed_steps']['kind'],'composed_from_components')
        self.assertGreater(full['fixed_steps']['total_fit_seconds'],full['fixed_steps']['core_seconds'])
        self.assertNotEqual(full['fixed_visits']['usage_to_core_wall_ratio'],full['fixed_visits']['usage_item_forwards']/(full['fixed_visits']['steps']*1024))
    def test_any_historical_failure_marks_every_rq_extrapolation_unknown(self):
        t=tool();h=self.histories();h[1]['wall_seconds']=100
        p=t.projection(self.rows(),h,{'usage_interval':10000})
        self.assertFalse(p['rq_selfcheck_passed']);self.assertEqual(p['rq_targets'],'UNKNOWN')
        self.assertEqual(p['alert'],'historical_factor_exceeds_two')
    def test_missing_usage_contaminated_core_and_duplicate_pass_cannot_extrapolate(self):
        t=tool();rows=self.rows()
        for change in ('missing','contaminated','duplicate'):
            import copy
            r=copy.deepcopy(rows)
            if change=='missing':r[-1]['status']='CENSORED_WALL'
            elif change=='contaminated':r[0]['usage_calls']=1
            else:r[1]['usage_calls']=2
            self.assertEqual(t.projection(r,self.histories(),{'usage_interval':10000})['rq_targets'],'UNKNOWN')
    def test_default_request_preserves_frozen_bhkm_core_and_usage_scope(self):
        t=tool();r=t.default_request();self.assertEqual(r['bhkm_sizes'],[250000,300000]);self.assertEqual(r['usage_sizes'],[120415,250000,300000]);self.assertEqual(r['rq_steps'],3000)
        t.validate_request(r)
        import copy
        for k,v in [('bhkm_sizes',[100000,250000]),('usage_sizes',[120415]),('rq_steps',10000),('overall_seconds',9001)]:
            b=copy.deepcopy(r);b[k]=v
            with self.assertRaises(ValueError):t.validate_request(b)
if __name__=='__main__':unittest.main()

class FormalWiringTests(unittest.TestCase):
    def test_history_parameters_use_actual_fit_not_effective_default(self):
        t=tool();self.assertTrue(callable(getattr(t,'history_entry',None)))
        m={'status':'complete','engineering':False,'branching':256,'catalog_items':120415,'producer':dict(t.components.PINS)}
        tok={'config':{'arm':'l2','fit':{'eval_every':2000,'max_steps':300000},'effective_prereg':{'eval_every':10000,'latent_dim':32,'levels':3}}}
        meta={'run_id':'g-tok-t1r-fit','argv':['python','--device','cuda'],'git':{'dirty':True}}
        term={'run_id':'g-tok-t1r-fit','status':'succeeded','exit_code':0,'wall_cap_exceeded':False,'wall_seconds':4329.242808818817}
        h=t.history_entry('g-tok-t1r-fit',m,tok,meta,term)
        self.assertEqual(h['usage_interval'],2000);self.assertEqual(h['device'],'cuda')
        self.assertEqual(h['wall_seconds'],4329.242808818817)
        import copy
        for key in ('producer','catalog_items'):
            bad=copy.deepcopy(m)
            if key=='producer':bad[key]['src/vkgr/sid_rqvae.py']='0'*64
            else:bad[key]=1
            with self.assertRaises(ValueError):t.history_entry('g-tok-t1r-fit',bad,tok,meta,term)
    def test_core_purity_gate_is_not_optional(self):
        t=tool();self.assertTrue(callable(getattr(t,'core_result',None)))
        fitted={'step':3000,'status':'complete','elapsed':30.,'usage_log':[]}
        r=t.core_result(fitted,{'usage_interval':10000})
        self.assertEqual(r['usage_calls'],0);self.assertEqual(r['core_seconds_per_step'],.01)
        fitted['usage_log']=[(2000,[1,1,1])]
        with self.assertRaises(ValueError):t.core_result(fitted,{'usage_interval':10000})

class TinyWorkerTests(unittest.TestCase):
    def test_tiny_actual_core_and_usage_are_distinct_bounded_cpu_workers(self):
        import subprocess,json,os,sys
        for method in ('core','usage'):
            cp=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/sid_scale_composed.py'),'--worker',method,'--n','16','--k','4','--steps','3','--engineering'],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=45)
            self.assertEqual(cp.returncode,0,cp.stderr)
            r=json.loads(cp.stdout);self.assertEqual(r['method'],method);self.assertEqual(r['status'],'COMPLETE');self.assertTrue(r['engineering']);self.assertEqual(r['device'],'cpu')
            self.assertEqual(r['usage_calls'],0 if method=='core' else 1)
            if method=='core':self.assertTrue(r['core_interval_uncontaminated']);self.assertEqual(r['steps'],3)
            else:self.assertEqual(r['function'],'vkgr.sid_rqvae._usage');self.assertFalse(r['checkpoint_written'])

class StageOrchestrationTests(unittest.TestCase):
    def test_one_core_three_usage_stages_preserve_shared_overall_budget(self):
        t=tool();r=t.default_request();r.update(engineering=True,bhkm_sizes=[8,16],rq_n=16,rq_steps=3,branching=4,usage_sizes=[16,32,64],overall_seconds=4,point_seconds=.5)
        calls=[];now=[0.]
        def clock():now[0]+=.01;return now[0]
        def supervisor(argv,*,wall_seconds,rss_mib):
            method=argv[argv.index('--worker')+1];n=int(argv[argv.index('--n')+1]);calls.append((method,n,wall_seconds,rss_mib))
            p={'method':method,'n':n,'engineering':True,'producer':t.producer()}
            if method=='bhkm':p.update(model_units=t.base.bhkm_work(n)['units'],fit_seconds=.01)
            elif method=='core':p.update(core_seconds_per_step=.01,steps=3,usage_calls=0)
            else:p.update(seconds=.01,usage_calls=1)
            return {'status':'COMPLETE','result':p}
        result=t.run_benchmark(r,[],{'usage_interval':10000},clock=clock,supervisor=supervisor)
        self.assertEqual([(m,n) for m,n,_,_ in calls],[('bhkm',8),('bhkm',16),('core',16),('usage',16),('usage',32),('usage',64)])
        self.assertTrue(all(0<wall<4 and rss==16384 for _,_,wall,rss in calls))
        self.assertEqual(result['projection']['rq_targets'],'UNKNOWN')

class ReviewIntegerContractTests(unittest.TestCase):
    def test_formal_usage_points_reject_float_equivalents_before_any_worker(self):
        t=tool();r=t.default_request();r['usage_sizes']=[120415.,250000.,300000.]
        with self.assertRaises(ValueError):t.validate_request(r)

class ReviewBHKMIntegerContractTests(unittest.TestCase):
    def test_formal_bhkm_points_reject_float_equivalents_before_any_worker(self):
        t=tool();r=t.default_request();r['bhkm_sizes']=[250000.,300000.]
        with self.assertRaises(ValueError):t.validate_request(r)
