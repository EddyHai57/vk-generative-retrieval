"""Finite CPU cost publication guards; synthetic metadata, no jobs or network."""
import importlib.util
import copy
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def tool():
    p=ROOT/'scripts/sid_scale_composed_queue.py';assert p.is_file(),'missing composed queue bridge'
    s=importlib.util.spec_from_file_location('cost_queue',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class QueueTests(unittest.TestCase):
    def request(self,t):
        j={'schema':'vkgr-queue-job-v1','run_id':t.PF_ID,'kind':'ops','cwd':t.DEP,'config':t.OPS+'/report-request-v1.json',
           'argv':[sys.executable,'-B',t.DEP+'/'+t.SCRIPT,'--request',t.OPS+'/report-request-v1.json','--request-sha','c'*64,'--output',t.SCHED+'/runs/'+t.PF_ID+'/out','--preflight'],
           'env':['PYTHONPATH='+t.DEP+'/src:'+t.DEP,'PYTHONDONTWRITEBYTECODE=1',*[k+'=1' for k in t.THREAD_ENV],'VKGR_PINNED_MANIFEST='+t.OPS+'/scheduler-pinned-report-v1.json'],
           'pinned_manifest':t.OPS+'/scheduler-pinned-report-v1.json','cpu_cores':1,'ram_mib':18432,'vram_mib':0,'wall_hours':2.5,'disk_gib':1,'no_admission':False,'stop_mode':'signal'}
        files={k:'b'*64 for k in t.REQUIRED};j['pinned_files']=[k+'='+v for k,v in sorted(files.items())]
        return {'action':'dry','root':t.SCHED,'ops':t.OPS,'scheduler_code':'/workspace/vkgr-infra03-ops','source_commit':'a'*40,'files':files,'preflight_ref':None,'job':j}
    def test_exact_parent_and_preflight_paths_and_cpu_resources(self):
        t=tool();r=self.request(t);t.validate(r)
        for field,val in [('run_id','new-arm'),('vram_mib',1),('cpu_cores',2),('wall_hours',3),('kind','real')]:
            bad=copy.deepcopy(r);bad['job'][field]=val
            with self.assertRaises(ValueError):t.validate(bad)
        bad=copy.deepcopy(r);bad['job']['argv'][-2]='/wrong/out'
        with self.assertRaises(ValueError):t.validate(bad)
        r['job']['run_id']=t.REPORT_ID;r['job']['argv']=r['job']['argv'][:-1];r['job']['argv'][-1]=t.SCHED+'/runs/'+t.REPORT_ID+'/out';r['preflight_ref']={'path':'fake','sha256':'a'*64};t.validate(r)
    def test_original_run_queue_or_intent_never_republishes(self):
        t=tool()
        for kind in ('run','pending','claimed','rejected','intent'):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d).resolve();ops=root/'ops';ops.mkdir();(root/'runs').mkdir()
                for stage in ('pending','claimed','rejected'):(root/'queue'/stage).mkdir(parents=True)
                if kind=='run':(root/'runs'/t.REPORT_ID).mkdir()
                elif kind=='intent':(ops/('intent-'+t.REPORT_ID+'.json')).write_text('{}')
                else:(root/'queue'/kind/('001-'+t.REPORT_ID+'.json')).write_text('{}')
                with self.assertRaises(FileExistsError):t.ensure_fresh(root,ops,t.REPORT_ID)

class ReviewResourceContractTests(unittest.TestCase):
    def test_declared_resources_cannot_be_lower_than_frozen_parent_recipe(self):
        t=tool()
        for key,value in [('ram_mib',1),('wall_hours',.001),('disk_gib',.1)]:
            r=QueueTests().request(t);r['job'][key]=value
            with self.assertRaises(ValueError):t.validate(r)
