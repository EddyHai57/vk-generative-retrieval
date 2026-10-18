"""Metadata publication only: no real data, jobs, network or training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]

def tool():
    p=ROOT/'scripts/sid_structure_report_queue.py'
    assert p.is_file(),'missing bounded read-only report transport'
    spec=importlib.util.spec_from_file_location('report_queue',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class ReportQueueTests(unittest.TestCase):
    def request(self,t):
        dep='/workspace/vkgr-sid-structure-report-v1';ops='/workspace/vkgr-sid-structure-report-ops-v1'
        return {'action':'dry','root':'/workspace/vkgr-infra01-root','ops':ops,'scheduler_code':'/workspace/vkgr-infra03-ops',
                'source_commit':'a'*40,'files':{name:'b'*64 for name in t.REQUIRED},'preflight_ref':None,
                'job':{'schema':'vkgr-queue-job-v1','run_id':t.PF_ID,'kind':'ops','cwd':dep,'config':ops+'/report-request-v1.json',
                       'argv':[sys.executable,'-B',dep+'/analysis/sid_structure_report_read.py','--request',ops+'/report-request-v1.json','--request-sha','c'*64,'--preflight'],
                       'env':['PYTHONPATH='+dep+'/src:'+dep+'/analysis','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','VKGR_PINNED_MANIFEST='+ops+'/scheduler-pinned-report-v1.json'],'pinned_manifest':ops+'/scheduler-pinned-report-v1.json','cpu_cores':1,'ram_mib':16384,'vram_mib':0,'wall_hours':2.,'disk_gib':1,'no_admission':False}}
    def test_exact_stage_argv_scope_and_resources(self):
        import copy
        t=tool();req=self.request(t);req['job']['pinned_files']=[k+'='+v for k,v in sorted(req['files'].items())];t.validate(req)
        bad=copy.deepcopy(req);bad['job']['pinned_files']=[]
        with self.assertRaises(ValueError):t.validate(bad)
        for role,value in (('run_id','g-learn-a1-s42'),('vram_mib',1),('cpu_cores',2),('kind','real'),('no_admission',True)):
            bad=copy.deepcopy(req);bad['job'][role]=value
            with self.assertRaises(ValueError):t.validate(bad)
        bad=copy.deepcopy(req);bad['job']['argv'][-1]='--other'
        with self.assertRaises(ValueError):t.validate(bad)
        bad=copy.deepcopy(req);bad['job']['cwd']='/workspace/other'
        with self.assertRaises(ValueError):t.validate(bad)
        bad=copy.deepcopy(req);bad['job']['argv'][2]='/workspace/other.py'
        with self.assertRaises(ValueError):t.validate(bad)
        bad=copy.deepcopy(req);bad['job']['argv'][0]='/usr/bin/false'
        with self.assertRaises(ValueError):t.validate(bad)

    def test_existing_run_rejection_or_intent_always_stops_publication(self):
        t=tool()
        for occupied in ('run','pending','claimed','rejected','intent'):
            with self.subTest(occupied=occupied),tempfile.TemporaryDirectory() as d:
                root=Path(d).resolve();ops=root/'ops';ops.mkdir()
                for stage in ('pending','claimed','rejected'):(root/'queue'/stage).mkdir(parents=True)
                (root/'runs').mkdir()
                if occupied=='run':(root/'runs'/t.PF_ID).mkdir()
                elif occupied=='intent':(ops/('intent-'+t.PF_ID+'.json')).write_text('{}')
                else:(root/'queue'/occupied/('001-'+t.PF_ID+'.json')).write_text('{}')
                with self.assertRaises(FileExistsError):t.ensure_fresh(root,ops,t.PF_ID)

    def test_published_intent_blocks_second_write_even_if_publisher_raises(self):
        t=tool()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();ops=root/'ops';ops.mkdir();(root/'runs').mkdir()
            for stage in ('pending','claimed','rejected'):(root/'queue'/stage).mkdir(parents=True)
            req={'root':str(root),'ops':str(ops),'job':{'run_id':t.PF_ID},'dry_job_sha256':t.digest({'run_id':t.PF_ID})}
            calls=[]
            def broken(_root,job):calls.append(job);raise OSError('transport failed after intent')
            with self.assertRaises(OSError):t.publish(req,lambda job:{},broken)
            with self.assertRaises(FileExistsError):t.publish(req,lambda job:{},broken)
            self.assertEqual(len(calls),1)

    def test_report_cannot_use_unbound_preflight_stdout(self):
        t=tool()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();run=root/'runs'/t.PF_ID;(run/'logs').mkdir(parents=True)
            log=run/'logs/stdout.log';log.write_text(json.dumps({'status':'preflight_ok'}))
            req={'root':str(root),'source_commit':'a'*40,'job':{'config':str(root/'request.json')},'preflight_ref':{'path':str(log),'sha256':t.sha(log)}}
            (root/'request.json').write_text('{}')
            with self.assertRaises(ValueError):t.check_preflight(req)
            req['preflight_ref']['path']=str(root/'other.json')
            with self.assertRaises(ValueError):t.check_preflight(req)

class PreflightBindingTests(unittest.TestCase):
    def test_complete_original_preflight_chain_and_adversarial_proofs(self):
        import copy
        t=tool()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();run=root/'runs'/t.PF_ID;(run/'logs').mkdir(parents=True)
            for stage in ('pending','claimed','rejected'):(root/'queue'/stage).mkdir(parents=True)
            request=root/'request.json';request.write_text('{}');pin=t.sha(request)
            argv=[sys.executable,'-B',str(ROOT/'analysis/sid_structure_report_read.py'),'--request',str(request),'--request-sha',pin]
            meta={'run_id':t.PF_ID,'git':{'commit':'a'*40,'dirty':False,'dirty_files':[]},'cwd':str(ROOT),'config':{'path':str(request),'sha256':pin},'argv':argv+['--preflight']}
            mp=run/'run.json';mp.write_text(json.dumps(meta));(run/'terminal.json').write_text(json.dumps({'run_id':t.PF_ID,'status':'succeeded','exit_code':0,'wall_cap_exceeded':False}))
            (root/'queue/claimed'/('001-'+t.PF_ID+'.json')).write_text(json.dumps({'run_id':t.PF_ID,'argv':meta['argv'],'cwd':str(ROOT)}))
            proof={'status':'preflight_ok','run_id':t.PF_ID,'source_commit':'a'*40,'request_sha256':pin,'run_manifest_sha256':t.sha(mp),'test_read':False,'checkpoint_read':False,'model_run':False,'engineering':False}
            log=run/'logs/stdout.log';req={'root':str(root),'source_commit':'a'*40,'job':{'cwd':str(ROOT),'config':str(request),'argv':argv}}
            def save(value):
                log.write_text(json.dumps(value));req['preflight_ref']={'path':str(log),'sha256':t.sha(log)}
            save(proof);self.assertEqual(t.check_preflight(req)['run_manifest_sha256'],t.sha(mp))
            for key,value in (('status','verified'),('run_id','other'),('source_commit','b'*40),('request_sha256','c'*64),('run_manifest_sha256','d'*64),('test_read',True),('engineering',True)):
                bad=copy.deepcopy(proof);bad[key]=value;save(bad)
                with self.assertRaises(ValueError):t.check_preflight(req)

class SourceIdentityTests(unittest.TestCase):
    def test_source_git_bytes_scheduler_ledger_and_request_all_match(self):
        import subprocess
        t=tool()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();dep=root/'dep';ops=root/'ops';dep.mkdir();ops.mkdir();files={}
            for name in t.REQUIRED:
                p=dep/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixed '+name);files[name]=t.sha(p)
            for args in (['init','-q'],['add',*t.REQUIRED],['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture']):
                subprocess.run(['git','-C',str(dep),*args],check=True,capture_output=True)
            commit=subprocess.check_output(['git','-C',str(dep),'rev-parse','HEAD'],text=True).strip()
            config=ops/'report-request.json';config.write_text('{}');pin=t.sha(config);pinned=ops/'pinned.json'
            record={'schema':'vkgr-pinned-v1','base_commit':commit,'target_commit':commit,'overrides':{},'files':files};pinned.write_text(json.dumps(record))
            req={'source_commit':commit,'files':files,'job':{'cwd':str(dep),'config':str(config),'pinned_manifest':str(pinned),'argv':[sys.executable,'-B',str(dep/t.SCRIPT),'--request',str(config),'--request-sha',pin]}}
            t.identity(req)
            pinned.write_text(json.dumps({**record,'files':{}}))
            with self.assertRaises(ValueError):t.identity(req)
            pinned.write_text(json.dumps(record));(dep/t.SCRIPT).write_text('changed source')
            with self.assertRaises(ValueError):t.identity(req)
            (dep/t.SCRIPT).write_text('fixed '+t.SCRIPT);(dep/'unexpected.py').write_text('untracked extra source')
            with self.assertRaises(ValueError):t.identity(req)

if __name__=='__main__':unittest.main()
