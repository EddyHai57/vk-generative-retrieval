"""Independent G3 cross-trajectory budget and headroom oracles."""
import datetime,tempfile,unittest
from pathlib import Path
try:
    from vkgr import generative_pilot as gp
except ImportError:gp=None

def spec(steps=2,decode=3):
    return {'task_id':'G3','kind':'engineering_pool','steps_by_phase':{'formal':steps,'replay':steps},
            'max_steps':steps*2,'max_decode':decode,'max_disk_bytes':1000000,'max_resumes':1,
            'deadline_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=60)).isoformat(),
            'binding_sha256':'a'*64}

class PoolTests(unittest.TestCase):
    def test_genesis_and_local_ledger_bytes_are_charged(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            tiny=spec();tiny['max_disk_bytes']=1
            with self.assertRaises(ValueError):gp.ExecutionPool.create(Path(d)/'tiny',tiny)
            with gp.ExecutionPool.create(Path(d)/'pool',spec()) as pool:
                self.assertEqual(pool.state['disk'],sum(p.stat().st_size for p in pool.root.glob('event-*.json')))
                ledger_type=gp.accounted_ledger(pool,'formal')
                budget={'reserved_steps':0,'completed_steps':0,'consumed_wall_by_phase':{'train':0.,'evaluate':0.,'verify':0.},'resume_attempts':0,'max_optimizer_steps':2,'max_resume_attempts':1,'deadline_utc':pool.spec['deadline_utc'],'parent_ledger_head':None}
                ledger=ledger_type(Path(d)/'ledger',budget,{'task_id':'G3'})
                ledger.reserve_step();ledger.complete_step(.1)
                actual=sum(p.stat().st_size for p in Path(d).rglob('*.json'))
                self.assertEqual(pool.state['disk'],actual)

    def test_production_tempdirs_use_platform_default(self):
        from vkgr import generative_real as gr
        for module in (gr,gp):
            self.assertNotIn("TemporaryDirectory(dir='/private/tmp')",Path(module.__file__).read_text())

    def test_accounted_writer_reserves_large_manifest_before_write(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            with gp.ExecutionPool.create(Path(d)/'pool',spec()) as pool:
                path=Path(d)/'large.json';value={'losses':[{'epoch':1,'loss_mean':.1,'examples':512} for _ in range(5100)]}
                gp.write_accounted_json(pool,'formal',path,value)
                self.assertGreater(path.stat().st_size,65536)
                self.assertGreaterEqual(pool.state['disk'],sum(p.stat().st_size for p in Path(d).rglob('*.json')))
                with self.assertRaises(ValueError):gp.write_accounted_json(pool,'formal',Path(d)/'too-large.json',{'payload':'x'*1000000})
                self.assertFalse((Path(d)/'too-large.json').exists())

    def test_checkpoint_wrapper_matches_frozen_engine_and_rejects_before_write(self):
        from vkgr import generative_real as gr,generative_training as gt,generative_experiment as ge
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,_=gr.prepare_engineering(root,base/'prepared');cfg=ge._config(0,42)
            gr._seed(42);model=gt.make_model(context.codec,cfg,device='cpu');optimizer=gt.make_optimizer(model,cfg);cursor=gt.EpochCursor(42,len(context.examples))
            binding=ge.build_binding(root,context,cfg)
            budget={'reserved_steps':0,'completed_steps':0,'consumed_wall_by_phase':{'train':0.,'evaluate':0.,'verify':0.},'resume_attempts':0,'max_optimizer_steps':2,'max_resume_attempts':1,'deadline_utc':spec()['deadline_utc'],'parent_ledger_head':None}
            selection={'epoch':0,'best_epoch':None,'best_metric':None,'bad_epochs':0}
            old=gt.save_checkpoint(base/'old.pt',model,optimizer,cursor,binding=binding,selection=selection,budget=budget)
            with gp.ExecutionPool.create(base/'pool',spec()) as pool:
                new,measurement=gp.save_accounted_checkpoint(pool,'formal',base/'new.pt',model,optimizer,cursor,binding=binding,selection=selection,budget=budget)
                self.assertEqual(new,old);self.assertEqual(measurement['bytes'],(base/'new.pt').stat().st_size)
                gt.load_checkpoint(base/'new.pt',new,model,optimizer,cursor,expected_binding=binding)
            tiny=spec();tiny['max_disk_bytes']=1024
            with gp.ExecutionPool.create(base/'tiny',tiny) as pool:
                with self.assertRaises(ValueError):gp.save_accounted_checkpoint(pool,'formal',base/'forbidden.pt',model,optimizer,cursor,binding=binding,selection=selection,budget=budget)
                self.assertFalse((base/'forbidden.pt').exists())
    def setUp(self):self.assertIsNotNone(gp,'G3 cumulative execution pool missing')
    def test_cross_trajectory_reservations_failure_no_refund_and_external_head(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d)/'pool';s=spec();pool=gp.ExecutionPool.create(root,s)
            pool.reserve('formal','steps');pool.reserve('formal','steps')
            with self.assertRaises(ValueError):pool.reserve('formal','steps')
            pool.reserve('replay','steps');pool.reserve('replay','steps')
            head=pool.head;self.assertEqual(sum(pool.state['steps'].values()),4);pool.close()
            with self.assertRaises(ValueError):gp.ExecutionPool.open(root,s,None)
            with self.assertRaises(ValueError):gp.ExecutionPool.open(root,s,'0'*64)
            with gp.ExecutionPool.open(root,s,head) as recovered:
                self.assertEqual(sum(recovered.state['steps'].values()),4)
                with self.assertRaises(ValueError):recovered.reserve('replay','steps')
    def test_decode_resume_disk_caps_and_deleted_tail_reject(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d)/'pool';s=spec();pool=gp.ExecutionPool.create(root,s)
            pool.reserve('formal','decode',3)
            with self.assertRaises(ValueError):pool.reserve('replay','decode')
            pool.resume()
            with self.assertRaises(ValueError):pool.resume()
            pool.reserve('formal','disk',s['max_disk_bytes']//2)
            with self.assertRaises(ValueError):pool.reserve('formal','disk',s['max_disk_bytes'])
            head=pool.head;pool.close()
            sorted(root.glob('event-*.json'))[-1].unlink()
            with self.assertRaises(ValueError):gp.ExecutionPool.open(root,s,head)
    def test_expiry_and_resource_unknown_are_stop_not_zero(self):
        s=spec();s['deadline_utc']='2000-01-01T00:00:00+00:00'
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            with self.assertRaises(ValueError):gp.ExecutionPool.create(Path(d)/'pool',s)
        good={'ram_total':100,'ram_available':30,'vram_total':100,'vram_free':30,'disk_total':100,'disk_free':30}
        gp.check_headroom(good,.2)
        for mode in ('missing','low','nonfinite'):
            x=dict(good)
            if mode=='missing':del x['ram_total']
            elif mode=='low':x['ram_available']=19
            else:x['vram_free']=float('nan')
            with self.assertRaises(ValueError):gp.check_headroom(x,.2)
    def test_cost_counts_full_validation_replay_and_tail(self):
        cost=gp.estimate_cost(examples=513,queries=7,epochs=3,step_seconds=2.,query_seconds=.5,
            fixed_seconds=10.,checkpoint_bytes=100,shard_bytes=200,ledger_event_bytes=50,pool_event_bytes=60,fixed_artifact_bytes=70,safety_factor=2.)
        self.assertEqual(cost['steps_per_epoch'],2)
        self.assertEqual(cost['optimizer_reservations'],12)
        self.assertEqual(cost['decode_calls'],49)
        self.assertEqual(cost['wall_seconds'],2*(12*2+49*.5+10))
        self.assertEqual(cost['disk_bytes'],2*((4*3+4)*100+(2*3+1)*200+cost['ledger_events']*50+cost['pool_events']*60+70))
        with self.assertRaises(ValueError):gp.estimate_cost(examples=513,queries=7,epochs=3,step_seconds=None,query_seconds=.5,fixed_seconds=10.,checkpoint_bytes=100,shard_bytes=200,safety_factor=2.)
    def test_actual_phase_wall_persists_and_single_writer(self):
        self.assertTrue(hasattr(gp.ExecutionPool,'charge_wall'),'phase wall accounting missing')
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d)/'pool';s=spec()
            pool=gp.ExecutionPool.create(root,s);pool.charge_wall('formal',.25);head=pool.head
            with self.assertRaises(ValueError):gp.ExecutionPool.open(root,s,head)
            pool.close()
            with gp.ExecutionPool.open(root,s,head) as child:
                child.charge_wall('replay',.5)
                self.assertEqual(child.state['wall_by_phase'],{'formal':.25,'replay':.5})

class PilotTests(unittest.TestCase):
    def test_engineering_pilot_real_model_restore_and_pool_counts(self):
        self.assertTrue(hasattr(gp,'pilot'),'actual bounded pilot missing')
        from vkgr import generative_real as gr
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            base=Path(d);context,prepared=gr.prepare_engineering(root,base/'prepared')
            result=gp.pilot(context,base/'pilot',engineering=True,prepared=prepared)
            self.assertEqual(result['status'],'engineering_pilot_pass_no_GPU_claim')
            self.assertTrue(result['restore_exact'])
            import os
            self.assertNotEqual(result['restore_worker_pid'],os.getpid())
            self.assertEqual(sum(result['pool_state']['steps'].values()),6)
            self.assertEqual(result['pool_state']['decode'],8)
            self.assertEqual(result['pool_state']['resumes'],1)
            self.assertEqual(result['measured_history_max'],2)
            self.assertEqual(len(result['decode_timings']),8)
            self.assertEqual(len(result['timings']),6)
            self.assertTrue(result['serialization_measurements'])
            self.assertTrue(all(x['bytes']>0 and x['seconds']>0 for x in result['serialization_measurements']))
            self.assertTrue(result['accounting_measurements'])
    def test_real_pilot_without_permit_rejects_before_resources(self):
        self.assertTrue(hasattr(gp,'pilot'),'actual bounded pilot missing')
        with patch_resource():
            with self.assertRaises(ValueError):gp.pilot(None,Path('/forbidden'),engineering=False)

def patch_resource():
    from unittest.mock import patch
    return patch('vkgr.generative_pilot.observe_resources',side_effect=AssertionError('no resource/device reads'))

if __name__=='__main__':unittest.main()
