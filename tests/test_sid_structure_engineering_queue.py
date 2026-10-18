"""The scientific preflight/CUDA commands must be queued, never mux computations."""
import importlib.util,json,tempfile,unittest,io,contextlib,sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
def module():
    path=ROOT/'scripts/sid_structure_engineering_queue.py'
    assert path.is_file(),'missing versioned engineering queue bridge'
    spec=importlib.util.spec_from_file_location('structure_engineering_queue',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class EngineeringQueue(unittest.TestCase):
    def req(self,r):
        for s in ('pending','claimed','rejected'):(r/'queue'/s).mkdir(parents=True)
        (r/'ops').mkdir()
        config=r/'ops/request.json';config.write_text('{}');dep=r/'code'
        return {'action':'dry','root':str(r),'ops':str(r/'ops'),'scheduler_code':str(r/'scheduler'),'job':{'schema':'vkgr-queue-job-v1','kind':'ops','cpu_cores':1,'vram_mib':39168,'no_admission':False,'run_id':'g-structure-a1-preflight-v1','cwd':str(dep),'config':str(config),'argv':[sys.executable,'-B',str(dep/'scripts/sid_structure_ops.py'),'preflight','--request',str(config)]}}
    def test_only_the_two_engineering_run_ids_are_allowed(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            req=self.req(Path(d));m.validate(req)
            req['job']['run_id']='g-structure-a1-cuda-v1';req['job']['argv'][3]='cuda-preview';m.validate(req)
            for rid in ('g-structure-a1-s42','other-experiment'):
                req['job']['run_id']=rid
                with self.assertRaises(ValueError):m.validate(req)
    def test_fixed_id_rejects_wrong_program_stage_and_request(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            req=self.req(Path(d));job=req['job'];good=list(job['argv'])
            for args in ([sys.executable,'-c',"print('wrong')"],['/bin/echo',*good[1:]],good[:3]+['cuda-preview']+good[4:],good[:-1]+['/outside.json']):
                job['argv']=args
                with self.subTest(args=args),self.assertRaises(ValueError):m.validate(req)

    def test_relative_request_or_roots_are_rejected_before_launch(self):
        import copy
        m=module()
        with tempfile.TemporaryDirectory() as d:
            req=self.req(Path(d))
            bad=copy.deepcopy(req);bad['ops']='ops';bad['job']['config']='ops/request.json';bad['job']['argv'][-1]='ops/request.json'
            with self.assertRaises(ValueError):m.validate(bad)
            for key in ('root','ops','scheduler_code'):
                bad=copy.deepcopy(req);bad[key]='relative'
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate(bad)
            bad=copy.deepcopy(req);bad['job']['cwd']='code';bad['job']['argv'][2]='code/scripts/sid_structure_ops.py'
            with self.assertRaises(ValueError):m.validate(bad)

    def test_resource_scope_rejects_training_or_over_reservation(self):
        import copy
        m=module()
        with tempfile.TemporaryDirectory() as d:
            req=self.req(Path(d))
            for key,value in (('schema','other'),('kind','real'),('cpu_cores',2),('no_admission',True),('vram_mib',39169),('vram_mib',-1)):
                bad=copy.deepcopy(req);bad['job'][key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):m.validate(bad)

    def test_rejected_is_visible_and_wrong_dry_binding_never_publishes(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);req=self.req(r);rid=req['job']['run_id'];reject=r/'queue/rejected'/('001-'+rid+'.json');reject.write_text('{}');reject.with_suffix('.reason').write_text('blocked')
            snap=m.inventory(req);self.assertEqual(len(snap['queue']['rejected']),1);self.assertEqual(list(snap['queue']['reasons'].values()),['blocked'])
            req['action']='submit';req['dry_job_sha256']='0'*64
            with patch.object(sys,'stdin',io.StringIO(json.dumps(req))),self.assertRaises(ValueError):m.main()
            self.assertFalse((r/'ops'/('intent-'+rid+'.json')).exists())
    def test_dry_digest_mismatch_after_fresh_inventory_blocks_publication(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);req=self.req(r);req.update(action='submit',dry_job_sha256='0'*64)
            calls=[]
            fake=SimpleNamespace(launcher=SimpleNamespace(prepare=lambda job:{}),scheduler=SimpleNamespace(submit=lambda *args:calls.append(args)))
            with patch.dict(sys.modules,{'vkgr_ops':fake}),patch.object(sys,'stdin',io.StringIO(json.dumps(req))),self.assertRaisesRegex(ValueError,'dry binding'):m.main()
            self.assertEqual(calls,[]);self.assertFalse(list((r/'ops').glob('intent-*')))
if __name__=='__main__':unittest.main()
