"""Transport admission evidence, independent of catalog allocation."""
import io
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from analysis import p_query_sprint_ops as sprint
from analysis.p_query_blocks import parity
from vkgr.throughput_config import Engineering
from vkgr.throughput_workers import ordered_map

class ObservedPool(ProcessPoolExecutor):
    """Observe real submitted futures until ordered delivery consumes them."""
    instances=[]
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.pending=0;self.peak=0;self.submitted=[];self.delivered=[]
        self.instances.append(self)
    def submit(self,function,*args,**kwargs):
        future=super().submit(function,*args,**kwargs)
        self.pending+=1;self.peak=max(self.peak,self.pending);self.submitted.append(args[1])
        original=future.result
        def delivered(*a,**kw):
            value=original(*a,**kw);self.pending-=1;self.delivered.append(value[0]);return value
        future.result=delivered
        return future

class CapacityTests(unittest.TestCase):
    def probe(self,workers,cap):
        with patch('vkgr.throughput_workers.ProcessPoolExecutor',ObservedPool):
            result=list(ordered_map(abs,range(-8,0),workers=workers,prefetch=1,max_in_flight_bytes=cap,
                        result_size_hint=lambda _:32*50*4+8192*(1+12000000//4096)))
        pool=ObservedPool.instances[-1]
        self.assertEqual(result,list(range(8,0,-1)))
        self.assertEqual(pool.delivered,list(range(8)))
        return pool.peak
    def test_existing_cap_mechanically_serializes_full_catalog_hint(self):
        for workers in (2,4,8):
            with self.subTest(workers=workers):self.assertEqual(self.probe(workers,128*sprint.MIB),1)
    def test_execute_cap_admits_each_worker_and_retains_ordinal_delivery(self):
        for workers in (2,4,8):
            with self.subTest(workers=workers),tempfile.TemporaryDirectory() as raw:
                root=Path(raw).resolve();data=root/'data';data.mkdir();(data/'inputs.json').write_text('{"files":[]}')
                req={'workers':workers,'stage':'diagnostic','root':str(data),'output':str(root/'out'),
                     'run_id':'fixture','query_count':256};request=root/'request';request.write_text('{}')
                binding={'inputs_manifest_sha256':sprint.digest(data/'inputs.json')}
                declaration={'ram_mib':98304,'measurement':'not_measured'}
                def transport(req,arrays,config):
                    self.assertEqual(config.prefetch,1)
                    self.assertEqual(self.probe(workers,config.max_in_flight_bytes),workers)
                    self.assertLessEqual(config.max_in_flight_bytes,workers*128*sprint.MIB)
                    path=Path(req['output'])/'queries-q256.npy';np.save(path,np.zeros((256,64),dtype='f4'))
                    return {'status':'verified','query_copy':{'file':path.name,'sha256':sprint.digest(path)}}
                with redirect_stderr(io.StringIO()),patch.object(sprint,'plan',return_value=(req,{},binding,None,declaration,None)),patch.object(sprint,'diagnostic',side_effect=transport):
                    sprint.execute(SimpleNamespace(request=request))
    def test_multiple_query_blocks_raw_float64_score_and_prediction_bytes(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw).resolve();rng=np.random.default_rng(47);arrays=[]
            for name,value in [('queries',rng.normal(size=(256,64)).astype('f4')),
                               ('items',rng.normal(size=(4103,64)).astype('f4')),
                               ('ids',np.arange(4103,dtype='u4')[::-1].copy())]:
                path=root/(name+'.npy');np.save(path,value);arrays.append(np.load(path,mmap_mode='r'))
            for workers in (2,4,8):
                with self.subTest(workers=workers),patch('vkgr.p_query_blocks.cpu_capacity',return_value=9),patch('vkgr.throughput_workers.ProcessPoolExecutor',ObservedPool):
                    result=parity(*arrays,Engineering(eval_workers=workers,use_mmap=True,prefetch=1,
                                  max_in_flight_bytes=workers*128*sprint.MIB),k=50)
                    self.assertTrue(result['score_bytes_equal_to_original'])
                    self.assertTrue(result['prediction_bytes_equal_to_original'])
                    self.assertEqual(result['score_blocks_checked'],16)
                    self.assertEqual(ObservedPool.instances[-1].peak,workers)
                    self.assertEqual(ObservedPool.instances[-1].delivered,list(range(8)))


@unittest.skipUnless(os.environ.get('VKGR_P3C_DATA_SOURCE') or subprocess.run(
    ['git','cat-file','-e','7041c90:docs/p3c_validation.md'],capture_output=True).returncode==0,
    'private fixed P3C data-channel source unavailable')
class SourceDeltaTests(unittest.TestCase):
    def test_real_launcher_cross_source_chain(self):
        import subprocess
        import sys
        result=subprocess.run([sys.executable,'-B','tests/fixtures/p_query_sprint_capacity_chain.py'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'cross_source_real_launcher_prepare_and_dry_verified')

    def test_fixed_old_producer_full_ledger_and_closed_delta(self):
        import copy
        import shutil
        import subprocess
        oldsha='43eb5bec6d32a5d37746a033452824fd399f2dc4'
        with tempfile.TemporaryDirectory() as raw:
            base=Path(raw).resolve();oldroot=base/'old-code';newroot=base/'new-code'
            def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
            for root in (oldroot,newroot):
                subprocess.run(['git','clone','-q','--shared','--no-checkout',str(sprint.ROOT),str(root)],check=True,capture_output=True)
                git(root,'checkout','-q','--detach',oldsha)
            shutil.copyfile(sprint.ROOT/'analysis/p_query_sprint_ops.py',newroot/'analysis/p_query_sprint_ops.py')
            # Force a genuine new fixed runner tree even during the RED phase.
            (newroot/'tests/test_p_query_sprint_capacity.py').write_text('# synthetic transport test fixture\n')
            git(newroot,'add','analysis/p_query_sprint_ops.py','tests/test_p_query_sprint_capacity.py')
            git(newroot,'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-q','-m','capacity fixture')
            newsha=git(newroot,'rev-parse','HEAD')
            private=(Path(os.environ['VKGR_P3C_DATA_SOURCE']).read_bytes() if os.environ.get('VKGR_P3C_DATA_SOURCE')
                     else subprocess.check_output(['git','-C',str(sprint.ROOT),'show','7041c90:'+sprint.P3C]))
            (newroot/sprint.P3C).parent.mkdir(parents=True,exist_ok=True);(newroot/sprint.P3C).write_bytes(private)
            self.assertFalse((oldroot/sprint.P3C).exists())
            self.assertEqual(git(oldroot,'ls-tree','--name-only',oldsha,'--',sprint.P3C),'')
            data=base/'data';data.mkdir();lock=base/'lock';lock.write_text('fixture')
            engine=base/'engine';(engine/'analysis').mkdir(parents=True);(engine/'analysis/a40_cost.py').write_text('fixture')
            original_ops=base/'original-ops'
            for stage in ('generate','preflight'):
                target=original_ops/'runs'/('a40-exact-'+stage+'-v1');(target/'logs').mkdir(parents=True)
                for name in ('run.json','terminal.json','logs/stdout.log'):(target/name).write_text('{}')
            for name in ('inputs.json','preflight.json'):(data/name).write_text('{}')
            old={'schema':'p-query-sprint-v1','stage':'diagnostic','run_id':'p-query-diagnostic-q32-w4-v1',
                 'runner_source':oldsha,'original_source':'b'*40,'workers':4,'query_count':32,'engine':str(engine),
                 'root':str(data),'lock':str(lock),'original_ops_root':str(original_ops),'ops_root':str(base/'ops'),
                 'output':str(base/'old-output')}
            oldpath=base/'old-request.json';oldpath.write_text(json.dumps(old))
            run=Path(old['ops_root'])/'runs'/old['run_id'];(run/'logs').mkdir(parents=True)
            binding={'fixed':'input'};resources=sprint.resources(old)
            payload={'schema':'p-query-sprint-result-v1','status':'verified','stage':'diagnostic','workers':4,'queries':32,
                     'input_binding':binding,'run_id':old['run_id'],'request_sha256':sprint.digest(oldpath),
                     'argv':sprint.invocation(oldpath),'runner_source':sprint.source(oldroot,oldsha,overlay=True),
                     'resources':{**resources,'process_tree_peak_rss_upper_bound_bytes':1024**3}}
            Path(old['output']).mkdir();(Path(old['output'])/'result.json').write_text(json.dumps(payload))
            with patch.object(sprint,'ROOT',oldroot):pins=[{'path':str(p),'sha256':sprint.digest(p)} for p in sprint.input_files(old,oldpath)]
            record={'schema':'vkgr-run-v1','run_id':old['run_id'],'kind':'ops','git':{'commit':oldsha,'dirty':False,'dirty_files':[]},
                    'cwd':str(oldroot),'argv':sprint.invocation(oldpath),'config':{'path':str(oldpath),'sha256':sprint.digest(oldpath)},
                    'inputs':pins,'resources':sprint.ops_resources(resources)}
            # input_files historically uses __file__, so pin its producer path explicitly.
            for pin in record['inputs']:
                if pin['path']==str(Path(sprint.__file__).resolve()):pin.update(path=str(oldroot/'analysis/p_query_sprint_ops.py'),sha256=sprint.digest(oldroot/'analysis/p_query_sprint_ops.py'))
            (run/'run.json').write_text(json.dumps(record));(run/'terminal.json').write_text(json.dumps(
                {'schema':'vkgr-terminal-v1','run_id':old['run_id'],'status':'succeeded','exit_code':0}))
            (run/'logs/stdout.log').write_text(json.dumps(payload))
            names=git(newroot,'diff','--name-only',oldsha,newsha).splitlines()
            delta={'schema':'p-query-source-delta-v1','old_source':oldsha,'new_source':newsha,'files':{
                name:{'old_sha256':sprint.digest(oldroot/name) if (oldroot/name).exists() else None,
                      'new_sha256':sprint.digest(newroot/name)} for name in names}}
            delta_path=base/'delta.json';delta_path.write_text(json.dumps(delta))
            req={**old,'run_id':'p-query-diagnostic-q64-w4-v1','query_count':64,'runner_source':newsha,
                 'output':str(base/'new-output'),'measurement_request':str(oldpath),
                 'source_delta':str(delta_path),'source_delta_sha256':sprint.digest(delta_path)}
            with patch.object(sprint,'ROOT',newroot):
                sprint.validate_request(req)
                self.assertEqual(sprint.measurement(req,binding)[1]['run_id'],old['run_id'])
                for case in ('bad_hash','wrong_old_source','missing_delta_path','extra_delta_path','wrong_new_hash','old_source_bytes','new_source_bytes','producer_cwd','producer_ledger','producer_ledger_presence','producer_git_dirty','old_input_pin','old_p3c_wrong','old_p3c_unrecorded','old_p3c_symlink','new_p3c_missing','new_p3c_wrong'):
                    changed=copy.deepcopy(req);doc=copy.deepcopy(delta);rec=copy.deepcopy(record);pay=copy.deepcopy(payload)
                    if case=='bad_hash':changed['source_delta_sha256']='0'*64
                    elif case=='wrong_old_source':doc['old_source']='a'*40
                    elif case=='missing_delta_path':doc['files'].pop(names[0])
                    elif case=='extra_delta_path':doc['files']['src/vkgr/throughput_exact.py']={'old_sha256':'a'*64,'new_sha256':'a'*64}
                    elif case=='wrong_new_hash':doc['files'][names[0]]['new_sha256']='0'*64
                    elif case=='old_source_bytes':(oldroot/'src/vkgr/throughput_exact.py').write_text('tampered')
                    elif case=='new_source_bytes':(newroot/'src/vkgr/throughput_exact.py').write_text('tampered')
                    elif case=='producer_cwd':rec['cwd']=str(newroot)
                    elif case=='producer_ledger':pay['runner_source']={'commit':oldsha}
                    elif case=='producer_ledger_presence':pay['runner_source']={**payload['runner_source'],'p3c_data_channel':True}
                    elif case=='producer_git_dirty':rec['git']={'commit':oldsha,'dirty':True,'dirty_files':[{'path':sprint.P3C,'sha256':sprint.P3C_SHA}]}
                    elif case=='old_input_pin':rec['inputs']=[]
                    elif case=='old_p3c_wrong':(oldroot/sprint.P3C).write_bytes(b'wrong P3C')
                    elif case=='old_p3c_unrecorded':(oldroot/sprint.P3C).write_bytes(private)
                    elif case=='old_p3c_symlink':(oldroot/sprint.P3C).symlink_to(base/'absent-link-target')
                    elif case=='new_p3c_missing':(newroot/sprint.P3C).rename(base/'new-p3c-preserved')
                    elif case=='new_p3c_wrong':(newroot/sprint.P3C).write_bytes(b'wrong P3C')
                    delta_path.write_text(json.dumps(doc))
                    if case!='bad_hash':changed['source_delta_sha256']=sprint.digest(delta_path)
                    (run/'run.json').write_text(json.dumps(rec));(run/'logs/stdout.log').write_text(json.dumps(pay))
                    (Path(old['output'])/'result.json').write_text(json.dumps(pay))
                    with self.subTest(case=case),self.assertRaises((ValueError,FileNotFoundError)):sprint.measurement(changed,binding)
                    if case.startswith('old_p3c_'):(oldroot/sprint.P3C).rename(base/('preserved-'+case))
                    if case=='new_p3c_missing':(base/'new-p3c-preserved').rename(newroot/sprint.P3C)
                    if case=='new_p3c_wrong':(newroot/sprint.P3C).write_bytes(private)
                    if case in ('old_source_bytes','new_source_bytes'):
                        # This fixture owns its new temporary clone; restore the fixture bytes.
                        subprocess.run(['git','-C',str(oldroot if case=='old_source_bytes' else newroot),'checkout','--','src/vkgr/throughput_exact.py'],check=True)
                # The existing fixed-overlay mode stays valid with matching historical metadata.
                delta_path.write_text(json.dumps(delta))
                (oldroot/sprint.P3C).write_bytes(private)
                overlay_payload={**payload,'runner_source':sprint.source(oldroot,oldsha,overlay=True)}
                overlay_record={**record,'git':{'commit':oldsha,'dirty':True,
                                'dirty_files':[{'path':sprint.P3C,'sha256':sprint.P3C_SHA}]}}
                (run/'run.json').write_text(json.dumps(overlay_record))
                (run/'logs/stdout.log').write_text(json.dumps(overlay_payload))
                (Path(old['output'])/'result.json').write_text(json.dumps(overlay_payload))
                self.assertEqual(sprint.measurement(req,binding)[1]['run_id'],old['run_id'])

if __name__=='__main__':unittest.main()
