import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from analysis import p_query_sprint_ops as sprint
from vkgr.throughput_config import Engineering

class SprintTests(unittest.TestCase):
    def request(self,root,**changes):
        value={'schema':'p-query-sprint-v1','stage':'diagnostic','run_id':'p-query-diagnostic-q32-w2-v1',
               'runner_source':'a'*40,'original_source':'b'*40,'workers':2,'query_count':32,
               **{key:str(root/key) for key in ('engine','root','lock','original_ops_root','ops_root','output')}}
        value.update(changes);return value
    def test_fixed_query_worker_stage_and_output_contract(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw).resolve();good=self.request(root)
            sprint.validate_request(good)
            for change in ({'query_count':33},{'workers':3},{'stage':'train'},{'runner_source':'main'},
                           {'output':good['root']},{'output':'relative'},{'query_count':26827}):
                with self.subTest(change=change),self.assertRaises(ValueError):sprint.validate_request({**good,**change})
    def test_diagnostic_copies_only_prefix_and_preserves_original_mmaps(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw).resolve();rng=np.random.default_rng(42)
            values=[rng.normal(size=(4103,64)).astype('f4'),rng.normal(size=(65,64)).astype('f4'),np.arange(4103,dtype='u4')]
            arrays=[]
            for name,value in zip(('items','queries','ids'),values):
                path=root/(name+'.npy');np.save(path,value);arrays.append(np.load(path,mmap_mode='r'))
            output=root/'new';output.mkdir()
            request=self.request(root,output=str(output))
            original=[sprint.digest(a.filename) for a in arrays]
            result=sprint.diagnostic(request,arrays,Engineering(eval_workers=2,use_mmap=True))
            self.assertTrue(result['score_bytes_equal_to_original'])
            self.assertFalse(result['full_query_set'])
            self.assertEqual(result['queries'],32)
            self.assertEqual([p.name for p in output.iterdir()],['queries-q32.npy'])
            copied=np.load(output/'queries-q32.npy',mmap_mode='r')
            self.assertEqual(copied.tobytes(),arrays[1][:32].tobytes())
            self.assertEqual(original,[sprint.digest(a.filename) for a in arrays])
            with self.assertRaises(FileExistsError):sprint.diagnostic(request,arrays,Engineering(eval_workers=2,use_mmap=True))
    def test_run_queue_collision_each_state(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);rid='p-query-diagnostic-q32-w2-v1'
            for state in ('runs','pending','claimed','rejected'):
                target=root/state
                if state=='runs':(target/'runs'/rid).mkdir(parents=True)
                else:
                    path=target/'queue'/state/('000-'+rid+'.json');path.parent.mkdir(parents=True);path.write_text('{}')
                with self.subTest(state=state),self.assertRaises(FileExistsError):sprint.unoccupied(target,rid)
    def test_first_resources_are_conservative_and_cpu_is_measured(self):
        req={'workers':8,'stage':'diagnostic','query_count':32}
        with patch.object(sprint,'cpu_capacity',return_value=8):
            with self.assertRaisesRegex(ValueError,'CPU'):sprint.resources(req)
        with patch.object(sprint,'cpu_capacity',return_value=61.2):
            result=sprint.resources(req)
            self.assertEqual(result['cpu_cores'],9)
            self.assertEqual(result['ram_mib'],98304)
            self.assertEqual(result['measurement'],'not_measured')
    def test_measured_resources_require_verified_same_worker_and_input_binding(self):
        request={'workers':2,'query_count':64}
        prior={'status':'verified','stage':'diagnostic','workers':2,'queries':32,
               'input_binding':{'input':'same'},'resources':{'process_tree_peak_rss_upper_bound_bytes':8*1024**3}}
        result=sprint.measured_resources(request,prior,{'input':'same'})
        self.assertEqual(result['measurement'],'measured_upper_bound')
        for change in ({'workers':4},{'status':'ready'},{'input_binding':{'input':'other'}}):
            with self.assertRaises(ValueError):sprint.measured_resources(request,{**prior,**change},{'input':'same'})
    def test_failed_original_cost_is_rejected_before_parallel_work(self):
        req={'engine':'/fixture','original_source':'a'*40,'root':'/fixture','lock':'/fixture','original_ops_root':'/fixture'}
        with patch.object(sprint,'bind_original_cost',side_effect=ValueError('original terminal')),patch.object(sprint,'parallel_topk') as work:
            with self.assertRaisesRegex(ValueError,'terminal'):sprint.formal_cost(req,Engineering())
            work.assert_not_called()
    def test_successful_diagnostic_producer_and_binding_tampering(self):
        with tempfile.TemporaryDirectory() as raw:
            base=Path(raw).resolve();req=self.request(base);path=base/'request.json';path.write_text(json.dumps(req))
            pin=base/'pinned';pin.write_text('fixed')
            run=Path(req['ops_root'])/'runs'/req['run_id'];(run/'logs').mkdir(parents=True)
            dirty=([{'path':sprint.P3C,'sha256':sprint.P3C_SHA}] if (sprint.ROOT/sprint.P3C).exists() and sprint.digest(sprint.ROOT/sprint.P3C)==sprint.P3C_SHA else [])
            payload={'schema':'p-query-sprint-result-v1','status':'verified','run_id':req['run_id'],
                     'request_sha256':sprint.digest(path),'argv':sprint.invocation(path),'runner_source':{'commit':req['runner_source']},
                     'resources':sprint.resources(req)}
            record={'schema':'vkgr-run-v1','run_id':req['run_id'],'kind':'ops',
                    'git':{'commit':req['runner_source'],'dirty':bool(dirty),'dirty_files':dirty},
                    'cwd':str(sprint.ROOT),'argv':sprint.invocation(path),'config':{'path':str(path),'sha256':sprint.digest(path)},
                    'inputs':[{'path':str(pin),'sha256':sprint.digest(pin)}],
                    **{key:payload['resources'][key] for key in ('cpu_cores','ram_mib','vram_mib','disk_gib','wall_hours')}}
            terminal={'schema':'vkgr-terminal-v1','run_id':req['run_id'],'status':'succeeded','exit_code':0}
            def write(r=record,t=terminal,p=payload):
                (run/'run.json').write_text(json.dumps(r));(run/'terminal.json').write_text(json.dumps(t))
                (run/'logs/stdout.log').write_text(json.dumps(p))
            write()
            with patch.object(sprint,'input_files',return_value=[pin]):
                self.assertEqual(sprint.accepted(path,payload)['run_id'],req['run_id'])
                variants=[({'ram_mib':1},{}),({'cpu_cores':1},{}),({'argv':['wrong']},{}),({'config':{'path':str(path),'sha256':'wrong'}},{}),
                          ({'inputs':[]},{}),({'cwd':'wrong'},{}),({'git':{'commit':'b'*40}},{}),({}, {'status':'failed'})]
                for change,termchange in variants:
                    write({**record,**change},{**terminal,**termchange})
                    with self.subTest(change=change,terminal=termchange),self.assertRaisesRegex(ValueError,'producer'):
                        sprint.accepted(path,payload)
                write(p={**payload,'request_sha256':'wrong'})
                with self.assertRaisesRegex(ValueError,'producer'):sprint.accepted(path,payload)

    def test_real_diagnostic_execute_and_no_final_receipt_on_binding_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            base=Path(raw).resolve();req=self.request(base);data=Path(req['root']);data.mkdir()
            rng=np.random.default_rng(42);arrays=[];files=[]
            for name,value in [('items',rng.normal(size=(4103,64)).astype('f4')),
                               ('queries',rng.normal(size=(65,64)).astype('f4')),
                               ('ids',np.arange(4103,dtype='u4'))]:
                path=data/(name+'.npy');np.save(path,value);arrays.append(np.load(path,mmap_mode='r'))
                files.append({'path':path.name,'sha256':sprint.digest(path)})
            (data/'inputs.json').write_text(json.dumps({'files':files}))
            request=base/'request.json';request.write_text(json.dumps(req))
            binding={'inputs_manifest_sha256':sprint.digest(data/'inputs.json')}
            declaration={'ram_mib':98304,'measurement':'not_measured'}
            with patch.object(sprint,'plan',return_value=(req,{'commit':req['runner_source']},binding,arrays,declaration,None)):
                result=sprint.execute(SimpleNamespace(request=request))
            self.assertFalse(result['engineering']);self.assertFalse(result['full_query_set'])
            self.assertEqual(json.loads((Path(req['output'])/'result.json').read_text()),result)
            self.assertGreater(result['resources']['process_tree_peak_rss_upper_bound_bytes'],0)
            bad={**req,'output':str(base/'bad-output')}
            with patch.object(sprint,'plan',return_value=(bad,{'commit':req['runner_source']},
                              {'inputs_manifest_sha256':'wrong'},arrays,declaration,None)):
                with self.assertRaisesRegex(ValueError,'manifest changed'):sprint.execute(SimpleNamespace(request=request))
            self.assertFalse((Path(bad['output'])/'result.json').exists())

    def test_cli_dry_illegal_and_missing_execute(self):
        cmd=[sys.executable,'-B','-m','analysis.p_query_sprint_ops']
        result=subprocess.run(cmd+['dry'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'dry')
        for args in (['illegal'],['run','--request','missing']):
            self.assertNotEqual(subprocess.run(cmd+args,capture_output=True,text=True).returncode,0)
    @unittest.skipUnless(os.environ.get('VKGR_P3C_DATA_SOURCE') or subprocess.run(
        ['git','cat-file','-e','7041c90:docs/p3c_validation.md'],capture_output=True).returncode==0,
        'private fixed P3C data-channel source unavailable')
    def test_real_launcher_prepare_and_dry_chain(self):
        result=subprocess.run([sys.executable,'-B','tests/fixtures/p_query_sprint_chain.py'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'real_launcher_prepare_and_dry_verified')
if __name__=='__main__':unittest.main()
