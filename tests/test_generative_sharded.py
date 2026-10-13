"""INFRA-01 A: sharded parallel G validation; synthetic CPU fixtures only."""
import hashlib,json,math,os,signal,subprocess,sys,tempfile,time,unittest
import multiprocessing as mp
from pathlib import Path
from unittest.mock import patch
import torch
from vkgr import generative_evaluation as ev,generative_experiment as ge,generative_training as gt
from vkgr.data_audit import file_sha256
try:
    from vkgr import generative_sharded as gs
except ImportError:
    gs=None
from vkgr import generative_real as gr
from test_generative_training import config

ROOT=Path(__file__).resolve().parents[1]
PIN='40863ebbe96f9e2e1da8f53d916e68ead1fa098b41ab80237481e53a22e2506d'
CK='a'*64

def synthetic(base):
    fixture=ROOT/'tests/fixtures/g2_synthetic.json'
    return ge.prepare_synthetic(ROOT,ROOT/'artifacts/generative-contract-manifest.json',PIN,fixture,file_sha256(fixture),base/'prepared')

def tiny(context,cfg=None):
    torch.manual_seed(42);return gt.make_model(context.codec,cfg or config(),device='cpu')

def sharded(model,context,output,workers,**kw):
    return gs.evaluate_queries_sharded(model,context,output=output,checkpoint_sha256=CK,workers=workers,
        config=config(),runtime=gr.engineering_runtime(),**kw)

def serial(model,context,output,**kw):
    return ev.evaluate_queries(model,context,output=output,checkpoint_sha256=CK,**kw)

def leftovers():
    deadline=time.time()+5
    while mp.active_children() and time.time()<deadline:time.sleep(.05)
    return mp.active_children()

class Hooks:
    def __init__(self):self.events=[]
    def guard(self):self.events.append('guard')
    def before(self):self.events.append('before')
    def after(self):self.events.append('after')
    def kw(self):return {'guard':self.guard,'before_decode':self.before,'after_decode':self.after}
    def count(self,name):return self.events.count(name)

class EnvTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(gs,'generative_sharded missing')
    def test_eval_workers_env_parsing(self):
        with patch.object(gs,'cpu_quota_cores',return_value=None):
            self.assertEqual(gs.eval_workers_from_env({}),1)
            self.assertEqual(gs.eval_workers_from_env({'VKGR_EVAL_WORKERS':''}),1)
            self.assertEqual(gs.eval_workers_from_env({'VKGR_EVAL_WORKERS':'4'}),4)
            self.assertEqual(gs.eval_workers_from_env({'VKGR_EVAL_WORKERS':'32'}),32)
            for bad in ('0','04',' 4','x','33','4 ','-1','1.5'):
                with self.subTest(bad=bad):
                    with self.assertRaisesRegex(ValueError,'eval workers env: integer 1..32'):gs.eval_workers_from_env({'VKGR_EVAL_WORKERS':bad})
    def test_eval_workers_capped_by_cgroup_cpu_quota(self):
        with patch.object(gs,'cpu_quota_cores',return_value=10):
            self.assertEqual(gs.eval_workers_from_env({'VKGR_EVAL_WORKERS':'10'}),10)
            self.assertEqual(gs.eval_workers_from_env({}),1)
            with self.assertRaisesRegex(ValueError,'eval workers env: integer 1..32'):gs.eval_workers_from_env({'VKGR_EVAL_WORKERS':'11'})
    def test_cpu_quota_reads_cgroup_v2_cpu_max(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            proc=Path(d)/'proc';sysfs=Path(d)/'sys';(proc/'self').mkdir(parents=True)
            cg=sysfs/'fs/cgroup/job';cg.mkdir(parents=True)
            (proc/'self/cgroup').write_text('0::/job\n')
            (cg/'cpu.max').write_text('1024000 100000\n')
            self.assertEqual(gs.cpu_quota_cores(proc,sysfs),10)
            (cg/'cpu.max').write_text('max 100000\n')
            self.assertIsNone(gs.cpu_quota_cores(proc,sysfs))
            (cg/'cpu.max').write_text('50000 100000\n')
            self.assertEqual(gs.cpu_quota_cores(proc,sysfs),1)
            (cg/'cpu.max').unlink()
            self.assertIsNone(gs.cpu_quota_cores(proc,sysfs))
            self.assertIsNone(gs.cpu_quota_cores(Path(d)/'missing',sysfs))
    def test_progress_env_parsing(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            d=Path(d).resolve()
            self.assertIsNone(gs.progress_target_from_env({}))
            self.assertEqual(gs.progress_target_from_env({'VKGR_PROGRESS_PATH':str(d/'p.json'),'VKGR_RUN_ID':'g-1'}),{'path':d/'p.json','run_id':'g-1'})
            (d/'real').write_text('x');(d/'link').symlink_to(d/'real')
            for env in ({'VKGR_PROGRESS_PATH':'rel/p.json','VKGR_RUN_ID':'g'},{'VKGR_PROGRESS_PATH':str(d/'missing/p.json'),'VKGR_RUN_ID':'g'},
                        {'VKGR_PROGRESS_PATH':str(d/'link'),'VKGR_RUN_ID':'g'},{'VKGR_PROGRESS_PATH':str(d/'p.json')},{'VKGR_PROGRESS_PATH':str(d/'p.json'),'VKGR_RUN_ID':''}):
                with self.subTest(env=env):
                    with self.assertRaisesRegex(ValueError,'progress env'):gs.progress_target_from_env(env)
    def test_cli_rejects_invalid_env_before_any_output(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);_,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            argv=['train','--engineering','--prepared',str(prepared['manifest_path']),'--prepared-sha',prepared['manifest_sha256'],'--output',str(base/'X')]
            import io,contextlib
            for env in ({'VKGR_EVAL_WORKERS':'0'},{'VKGR_PROGRESS_PATH':'relative.json','VKGR_RUN_ID':'g'},
                        {'VKGR_PROGRESS_PATH':str(base/'X/p.json'),'VKGR_RUN_ID':'g'}):
                with self.subTest(env=env):
                    err=io.StringIO()
                    with patch.dict(os.environ,env),contextlib.redirect_stderr(err):code=gr.main(argv)
                    self.assertEqual(code,1);self.assertIn('G3 gate failed: ',err.getvalue())
                    self.assertFalse((base/'X').exists());self.assertFalse((base/'X-pool').exists())
            (base/'X').mkdir()
            err=io.StringIO()
            with patch.dict(os.environ,{'VKGR_PROGRESS_PATH':str(base/'X/p.json'),'VKGR_RUN_ID':'g'}),contextlib.redirect_stderr(err):code=gr.main(argv)
            self.assertEqual(code,1);self.assertIn('progress path inside charged tree',err.getvalue());self.assertFalse((base/'X-pool').exists())

class EquivalenceTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(gs,'generative_sharded missing')
    def test_projection_equal_to_serial_across_shard_sizes_and_workers(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context=synthetic(base);model=tiny(context)
            for size in (1,3):
                reference=serial(model,context,base/f'serial-{size}',shard_size=size)
                expected=gs.evaluation_projection(base/f'serial-{size}')
                for workers in (2,4):
                    with self.subTest(shard_size=size,workers=workers):
                        hooks=Hooks();out=base/f'sharded-{size}-{workers}'
                        result=sharded(model,context,out,workers,shard_size=size,**hooks.kw())
                        self.assertEqual(gs.evaluation_projection(out),expected)
                        self.assertEqual(result['manifest']['computation_sha256'],reference['manifest']['computation_sha256'])
                        self.assertEqual(result['aggregate'],reference['aggregate'])
                        self.assertEqual(sorted(p.name for p in out.iterdir()),sorted(p.name for p in (base/f'serial-{size}').iterdir()))
                        self.assertEqual(hooks.events[0],'guard');self.assertEqual(hooks.count('guard'),1)
                        self.assertEqual(hooks.count('before'),7);self.assertEqual(hooks.count('after'),7)
                        seen_before=0;seen_after=0
                        for e in hooks.events[1:]:
                            seen_before+=e=='before';seen_after+=e=='after';self.assertLessEqual(seen_after,seen_before)
                        execution=result['observations']['execution']
                        self.assertEqual({k:execution[k] for k in ('workers','start_method','assignment','definition')},{'workers':workers,'start_method':'spawn','assignment':'round_robin','definition':'rss excludes worker processes'})
                        self.assertEqual(set(execution['parent_seconds']),{'startup','hooks','wait','write'})
                        self.assertEqual(leftovers(),[])
    def test_projection_ignores_only_whitelisted_fields(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context=synthetic(base);model=tiny(context)
            serial(model,context,base/'a',shard_size=3);before=gs.evaluation_projection(base/'a')
            shard=sorted((base/'a').glob('shard-*.json'))[0];value=json.loads(shard.read_bytes())
            value['rows'][0]['latency_seconds']+=1.;shard.write_bytes(json.dumps(value).encode())
            self.assertEqual(gs.evaluation_projection(base/'a'),before)
            value['rows'][0]['items'].reverse();shard.write_bytes(json.dumps(value).encode())
            self.assertNotEqual(gs.evaluation_projection(base/'a'),before)
    def test_cross_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context=synthetic(base);model=tiny(context)
            full=serial(model,context,base/'full',shard_size=3)
            a=serial(model,context,base/'a',shard_size=3,stop_after_shards=1)
            self.assertFalse(a['manifest']['complete'])
            hooks=Hooks()
            b=sharded(model,context,base/'b',2,shard_size=3,resume_manifest=a['manifest_path'],resume_sha256=a['manifest_sha256'],**hooks.kw())
            self.assertEqual(b['manifest']['computation_sha256'],full['manifest']['computation_sha256']);self.assertEqual(hooks.count('before'),4)
            c=sharded(model,context,base/'c',2,shard_size=3,stop_after_shards=1)
            self.assertFalse(c['manifest']['complete']);self.assertEqual(c['manifest']['completed_queries'],3)
            e=serial(model,context,base/'e',shard_size=3,resume_manifest=c['manifest_path'],resume_sha256=c['manifest_sha256'])
            self.assertEqual(e['manifest']['computation_sha256'],full['manifest']['computation_sha256'])
            self.assertEqual(leftovers(),[])
    def test_stop_drains_inflight_and_publishes_contiguous_prefix(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context=synthetic(base);model=tiny(context);hooks=Hooks()
            result=sharded(model,context,base/'s',2,shard_size=1,should_stop=lambda:hooks.count('after')>=1,**hooks.kw())
            manifest=result['manifest'];self.assertFalse(manifest['complete'])
            self.assertEqual([s['start'] for s in manifest['shards']],list(range(manifest['completed_queries'])))
            self.assertEqual(hooks.count('before'),manifest['completed_queries']);self.assertEqual(hooks.count('after'),manifest['completed_queries'])
            self.assertLess(manifest['completed_queries'],7)
            hooks=Hooks();k=sharded(model,context,base/'k',2,shard_size=1,stop_after_shards=2,**hooks.kw())
            self.assertEqual(k['manifest']['completed_queries'],2);self.assertEqual(hooks.count('before'),2)
            self.assertEqual(leftovers(),[])

class FailureTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(gs,'generative_sharded missing')
    def test_worker_death_is_fatal_without_manifest(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context=synthetic(base);model=tiny(context);calls=[0]
            def before():
                calls[0]+=1
                if calls[0]==2:os.kill(mp.active_children()[0].pid,signal.SIGKILL)
            with self.assertRaisesRegex(ValueError,'sharded evaluation worker died'):sharded(model,context,base/'x',2,shard_size=1,before_decode=before)
            self.assertFalse((base/'x/manifest.json').exists());self.assertEqual(leftovers(),[])
    def test_worker_init_failure_is_fatal(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context=synthetic(base);model=tiny(context)
            with self.assertRaisesRegex(ValueError,'sharded evaluation worker: '):
                gs.evaluate_queries_sharded(model,context,output=base/'x',checkpoint_sha256=CK,workers=2,config={**config(),'d_model':128},runtime=gr.engineering_runtime(),shard_size=3)
            self.assertFalse((base/'x/manifest.json').exists());self.assertEqual(leftovers(),[])
    def test_threads_and_workers_preconditions_before_output(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context=synthetic(base);model=tiny(context)
            with patch.dict(os.environ,{'OMP_NUM_THREADS':'2'}):
                with self.assertRaisesRegex(ValueError,'sharded evaluation requires threads==1'):sharded(model,context,base/'x',2)
            with self.assertRaisesRegex(ValueError,'sharded evaluation requires threads==1'):
                gs.evaluate_queries_sharded(model,context,output=base/'x',checkpoint_sha256=CK,workers=2,config=config(),runtime={**gr.engineering_runtime(),'threads':2})
            for workers in (1,33,True,'4'):
                with self.assertRaises(ValueError):sharded(model,context,base/'x',workers)
            self.assertFalse((base/'x').exists())
    def test_parent_checks_result_identity_and_decode_gate(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context=synthetic(base);model=tiny(context);original=gs._shard_rows
            bad_item=max(context.catalog)+1
            def tamper(mode):
                def wrapped(message):
                    rows=[list(r) for r in original(message)]
                    if mode=='missing':rows=rows[:-1]
                    elif mode=='order':rows.reverse()
                    elif mode=='nan':rows[0][3]=float('nan')
                    else:rows[0][1]=[bad_item]+rows[0][1][1:]
                    return [tuple(r) for r in rows]
                return wrapped
            for mode,message in (('missing','sharded result identity'),('order','sharded result identity'),('nan','sharded result identity'),('catalog','decoded candidate/audit gate')):
                with self.subTest(mode=mode),patch.object(gs,'_shard_rows',side_effect=tamper(mode)):
                    with self.assertRaisesRegex(ValueError,message):sharded(model,context,base/mode,2,shard_size=3)
                    self.assertFalse((base/mode/'manifest.json').exists());self.assertEqual(leftovers(),[])
            payload=gs._init_payload(model,context,config(),gr.engineering_runtime(),100,ev.model_digest(model))
            self.assertNotIn('catalog',payload);self.assertNotIn('queries',payload)
            self.assertEqual(payload['histories'],[q['history'] for q in context.queries])
            self.assertNotIn('target',json.dumps(payload['histories']))
            self.assertEqual(set(payload),{'codec_bytes','codec_sha256','histories','config','runtime','beam','parent_pid','state_dict','model_sha256'})

RUNNER='''
import json,os,signal,sys,tempfile
from pathlib import Path
os.setsid()
import multiprocessing as mp
import test_generative_sharded as t
stop=[False]
signal.signal(signal.SIGTERM,lambda s,f:stop.__setitem__(0,True))
# Children inherit this flag; the test sitecustomize then holds each spawned
# interpreter in bootstrap for 1s, so the signal deterministically lands
# before the worker target's SIG_IGN (pending across exec, or during the sleep).
os.environ['VKGR_TEST_SLOW_BOOT']='1'
import threading
original=mp.process.BaseProcess.start;fired=[]
def start(self):
    if not fired:
        # start() blocks until the child reads its (large) spawn payload, i.e.
        # after the 1s sitecustomize hold. Fire 0.3s in: child has exec'd and is
        # held in bootstrap, parent is inside its blocked-mask start window.
        fired.append(1);threading.Timer(.3,lambda:os.killpg(os.getpgid(0),signal.SIGTERM)).start()
    original(self)
mp.process.BaseProcess.start=start
with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
    base=Path(d);context=t.synthetic(base);model=t.tiny(context)
    try:
        r=t.sharded(model,context,base/'out',2,shard_size=1,should_stop=lambda:stop[0])
        out={'complete':r['manifest']['complete'],'completed':r['manifest']['completed_queries'],'error':None}
    except Exception as exc:out={'error':repr(exc)}
    out.update(stop=stop[0],fired=len(fired),children=len(t.leftovers()))
print(json.dumps(out))
'''

class SignalWindowTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(gs,'generative_sharded missing')
    def test_sigterm_during_worker_bootstrap_becomes_clean_stop(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            Path(d,'sitecustomize.py').write_text("import os,time\nif os.environ.get('VKGR_TEST_SLOW_BOOT')=='1':time.sleep(1.0)\n")
            env={**os.environ,'PYTHONPATH':os.pathsep.join([d,str(ROOT/'src'),str(ROOT/'tests')]),'OMP_NUM_THREADS':'1'}
            env.pop('VKGR_TEST_SLOW_BOOT',None)
            r=subprocess.run([sys.executable,'-B','-c',RUNNER],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
        self.assertEqual(r.returncode,0,r.stderr)
        out=json.loads(r.stdout.strip().splitlines()[-1])
        self.assertIsNone(out['error'],out);self.assertEqual(out['fired'],1);self.assertTrue(out['stop'])
        self.assertFalse(out['complete']);self.assertEqual(out['children'],0)

def projection_after_reproduce(test,context,trajectory,base,tag):
    """Re-decode every epoch checkpoint with the other mode; projections must be byte-equal."""
    binding=trajectory['binding'];cfg=binding['effective_config']
    for index,receipt in enumerate(trajectory['evaluations']):
        model=gt.make_model(context.codec,cfg,device='cpu');opt=gt.make_optimizer(model,cfg);cursor=gt.EpochCursor(cfg['seed'],len(context.examples))
        ck=receipt['checkpoint'];gt.load_checkpoint(Path(ck['path']),ck['sha256'],model,opt,cursor,expected_binding=binding)
        original=Path(receipt['manifest_path']).parent
        workers=json.loads(Path(receipt['manifest_path']).read_bytes())['observations'].get('execution',{}).get('workers',1)
        out=base/f'reproduce-{tag}-{index}'
        if workers==1:gs.evaluate_queries_sharded(model,context,output=out,checkpoint_sha256=ck['sha256'],workers=4,config=cfg,runtime=binding['runtime'],shard_size=64)
        else:ev.evaluate_queries(model,context,output=out,checkpoint_sha256=ck['sha256'],shard_size=64)
        test.assertEqual(gs.evaluation_projection(out),gs.evaluation_projection(original),f'{tag} epoch {index+1}')
    return len(trajectory['evaluations'])

class EndToEndTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(gs,'generative_sharded missing')
    def test_run_verify_evaluate_workers1_vs_workers4(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            env={k:v for k,v in os.environ.items() if k!='VKGR_EVAL_WORKERS'}
            with patch.dict(os.environ,env,clear=True):
                begin=time.perf_counter();r1=gr.run(context,config(),gr.engineering_limits(context,prepared),base/'w1');t1=time.perf_counter()-begin
                v1=gr.verify(context,r1['manifest_path'],r1['manifest_sha256'],r1['pool_head'],base/'w1-verify',replay=True,eval_workers=4)
                e1=gr.evaluate(context,r1['manifest_path'],r1['manifest_sha256'],v1['pool_head'],base/'w1-eval',eval_workers=4)
                begin=time.perf_counter();r4=gr.run(context,config(),gr.engineering_limits(context,prepared),base/'w4',eval_workers=4);t4=time.perf_counter()-begin
                v4=gr.verify(context,r4['manifest_path'],r4['manifest_sha256'],r4['pool_head'],base/'w4-verify',replay=True,eval_workers=1)
                e4=gr.evaluate(context,r4['manifest_path'],r4['manifest_sha256'],v4['pool_head'],base/'w4-eval',eval_workers=1)
            m1,m4=r1['manifest'],r4['manifest']
            for key in ('state_sha256','selection','losses','epoch_metrics','status'):self.assertEqual(m1[key],m4[key],key)
            self.assertEqual(m1['budget']['completed_steps'],m4['budget']['completed_steps'])
            for key in ('steps','decode','resumes'):self.assertEqual(r1['pool_state'][key],r4['pool_state'][key],key)
            for a,b in zip(m1['evaluations'],m4['evaluations']):
                self.assertEqual(json.loads(Path(a['manifest_path']).read_bytes())['computation_sha256'],json.loads(Path(b['manifest_path']).read_bytes())['computation_sha256'])
            self.assertEqual(json.loads(Path(m4['evaluations'][0]['manifest_path']).read_bytes())['observations']['execution']['workers'],4)
            self.assertNotIn('execution',json.loads(Path(m1['evaluations'][0]['manifest_path']).read_bytes())['observations'])
            replay1=json.loads((base/'w1-verify/manifest.json').read_bytes());replay4=json.loads((base/'w4-verify/manifest.json').read_bytes())
            self.assertEqual(json.loads(Path(replay1['evaluations'][0]['manifest_path']).read_bytes())['observations']['execution']['workers'],4)
            self.assertEqual(v1['status'],'verified');self.assertEqual(v4['status'],'verified')
            self.assertEqual(v1['pool_state']['decode'],28);self.assertEqual(v4['pool_state']['decode'],28)
            self.assertEqual(e1['aggregate'],e4['aggregate']);self.assertEqual(e1['pool_state']['decode'],35);self.assertEqual(e4['pool_state']['decode'],35)
            self.assertEqual(e1['manifest']['computation_sha256'],e4['manifest']['computation_sha256'])
            checked=0
            for tag,trajectory in (('w1-formal',m1),('w4-formal',m4),('w1-replay',replay1),('w4-replay',replay4)):checked+=projection_after_reproduce(self,context,trajectory,base,tag)
            self.assertEqual(checked,8);self.assertEqual(leftovers(),[])
            sys.stderr.write(f'INFRA01-SPEED eval workers=1 {t1:.2f}s workers=4 {t4:.2f}s\n')
    def test_sharded_identity_guard_once_per_shard_budget_guard_per_query(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context=synthetic(base);model=tiny(context);shards=[]
            sharded(model,context,base/'x',2,shard_size=3,before_shard=lambda:shards.append(1))
            self.assertEqual(len(shards),3)
            context,prepared=gr.prepare_engineering(ROOT,base/'prepared-e')
            run=gr.run(context,config(),gr.engineering_limits(context,prepared),base/'run')
            with patch.object(gr,'_guard',wraps=gr._guard) as full,patch.object(gr,'_query_guard',wraps=gr._query_guard) as cheap:
                result=gr.evaluate(context,run['manifest_path'],run['manifest_sha256'],run['pool_head'],base/'eval',eval_workers=4)
            self.assertEqual(full.call_count,2);self.assertEqual(cheap.call_count,14)
            self.assertEqual(result['pool_state']['decode'],21)
            with patch.object(gr,'_guard',wraps=gr._guard) as full,patch.object(gr,'_query_guard',wraps=gr._query_guard) as cheap:
                gr.evaluate(context,run['manifest_path'],run['manifest_sha256'],result['pool_head'],base/'eval-serial')
            self.assertEqual(full.call_count,15);self.assertEqual(cheap.call_count,0)
            self.assertEqual(json.loads((base/'eval/manifest.json').read_bytes())['computation_sha256'],json.loads((base/'eval-serial/manifest.json').read_bytes())['computation_sha256'])
            self.assertEqual(leftovers(),[])
    def test_progress_file_written_and_charged_tree_rejected(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d).resolve();context,prepared=gr.prepare_engineering(ROOT,base/'prepared')
            target={'path':base/'p.json','run_id':'t'};records=[]
            original=gs.write_progress
            def spy(t,record):records.append(record);return original(t,record)
            with patch.object(gs,'write_progress',side_effect=spy):
                gr.run(context,config(),gr.engineering_limits(context,prepared),base/'run',eval_workers=4,progress=target)
            data=(base/'p.json').read_bytes();self.assertLessEqual(len(data),4096);value=json.loads(data)
            self.assertEqual(set(value),{'schema','run_id','line','stage','phase','epoch','max_epochs','completed','total','message','updated_at','pid'})
            self.assertEqual((value['schema'],value['run_id'],value['line'],value['stage'],value['pid']),('vkgr-progress-v1','t','G','complete',os.getpid()))
            stages=[r['stage'] for r in records]
            self.assertIn('train',stages);self.assertIn('evaluate',stages)
            shard=[r for r in records if r['stage']=='evaluate']
            self.assertEqual([r['completed'] for r in shard],[7,7]);self.assertTrue(all(r['total']==7 for r in shard))
            self.assertFalse(list(base.glob('.p.json*')))
            with self.assertRaisesRegex(ValueError,'progress path inside charged tree'):
                gr.run(context,config(),gr.engineering_limits(context,prepared),base/'run2',eval_workers=4,progress={'path':base/'run2/p.json','run_id':'t'})
            self.assertFalse((base/'run2').exists());self.assertFalse((base/'run2-pool').exists())
    def test_progress_write_failure_disables_without_failing(self):
        import io,contextlib
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            target={'path':Path(d)/'gone'/'p.json','run_id':'t'};err=io.StringIO()
            record=gs.progress_record(target,'train','formal',1,2,1,2)
            with patch.object(gs,'_PROGRESS_DISABLED',[False]),contextlib.redirect_stderr(err):
                gs.write_progress(target,record);gs.write_progress(target,record)
            self.assertEqual(err.getvalue().count('progress write disabled: '),1)

if __name__=='__main__':unittest.main()
