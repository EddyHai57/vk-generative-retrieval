import importlib
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
import os
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import torch
from test_p_full_backend import native


class CachedBackendTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('analysis.p_full_cached_backend')
        except ModuleNotFoundError:self.fail('explicit sealed-cache consumer is missing')

    def test_scientific_input_identity_cannot_be_changed(self):
        m=self.module()
        old={k:{'value':k} for k in m.SCIENTIFIC_KEYS}
        new={**old,'source':{'new':True},'output_root':'new'}
        m.same_scientific_inputs(new,old)
        for key in m.SCIENTIFIC_KEYS:
            with self.subTest(key=key),self.assertRaises(ValueError):
                m.same_scientific_inputs({**new,key:None},old)

    def test_data_uses_original_readonly_row_objects_and_index_order(self):
        m=self.module();rows=[{'history':[2],'target':7}];queries=[{'history':[7],'target':2}]
        features=np.zeros((3,64),dtype=np.float32);features[1,0]=1
        got=m.data_view({'rows':rows,'queries':queries},[2,7],[7],features)
        self.assertIs(got['rows'],rows);self.assertIs(got['queries'],queries)
        self.assertEqual(got['idmap'],{2:1,7:2})
        self.assertEqual(got['support'],{7});self.assertEqual(got['supported_indices'],[2])
        self.assertTrue(torch.equal(got['content'],torch.from_numpy(features)))
        self.assertFalse(got['synthetic'])
        with self.assertRaises(ValueError):m.data_view({'rows':rows,'queries':queries},[2,7],[7,7],features)

    def test_invocation_binds_external_reference_and_explicit_science(self):
        m=self.module();f=m.functions('/reference.json','a'*64)
        pf=f['invocation']('preflight','/inputs','/request','b'*64)
        train=f['invocation']('train','/inputs','/request','b'*64)
        self.assertEqual(pf[2:5],['-m','analysis.p_full_cached_backend','preflight'])
        self.assertIn('/reference.json',pf);self.assertIn('a'*64,pf)
        self.assertNotIn('--execute-approved-science',pf)
        self.assertIn('--execute-approved-science',train)
        self.assertIs(f['run_session'],m.backend.run_session)
        self.assertIs(f['native_for'],m.backend.native_for)

    def test_holdout_admission_precedes_cache_access(self):
        m=self.module();f=m.functions('/reference.json','a'*64)
        with patch.object(m.backend,'read_transport',side_effect=ValueError('HOLDOUT')):
            with patch.object(m,'reader',side_effect=AssertionError('cache opened early')):
                with self.assertRaisesRegex(ValueError,'HOLDOUT'):
                    f['read_context']('/inputs','/request','b'*64)

    def test_external_reader_does_not_accept_arbitrary_pinned_code(self):
        m=self.module()
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp).resolve();(root/'analysis').mkdir()
            code=root/'analysis/p_pf_cache_reuse.py';code.write_text("raise RuntimeError('substituted reader executed')")
            ref=root/'reference.json';ref.write_text(json.dumps({'schema':'p-full-sealed-cache-reference-v1','root':str(root),
                'source':{'files':{'analysis/p_pf_cache_reuse.py':hashlib.sha256(code.read_bytes()).hexdigest()}},'seal':{}}))
            with patch('analysis.sgns_subset_acceptance.source_tree'):
                with self.assertRaisesRegex(ValueError,'fixed cache reader'):
                    m.reader(ref,hashlib.sha256(ref.read_bytes()).hexdigest())

    def test_cache_inventory_mapping_and_original_job_inputs_are_pinned(self):
        m=self.module()
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp).resolve();source=root/'source';source.mkdir();cache=root/'cache';cache.mkdir()
            raw=root/'rows.jsonl';raw.write_bytes(b'{}\n');offset=cache/'train-offsets';offset.write_bytes(b'12345678')
            request=root/'seal-request.json';request.write_bytes(b'{}')
            run=root/'run';run.mkdir();job=run/'run.json'
            pin=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            job.write_text(json.dumps({'inputs':[{'path':str(request),'sha256':pin(request),'bytes':2}]}))
            receiptfile=root/'sealed.json';receiptfile.write_bytes(b'{}')
            ref={'root':str(source),'source':{'files':{}},'seal':{'manifest':{'path':str(receiptfile),'sha256':pin(receiptfile)},
                'run_dir':str(run),'pin':{'job_sha256':{'run.json':pin(job)}}}}
            receipt={'inputs':{'train':{'path':str(raw),'sha256':pin(raw),'count':1}},'cache_root':str(cache),
                     'semantic':{'files':{'train-offsets':{'sha256':pin(offset),'bytes':8}}}}
            reference=root/'reference.json';reference.write_bytes(b'{}')
            with patch.object(m,'reader',return_value=(None,ref)),patch.object(m,'origin',return_value=(receipt,{})):
                found={r['path']:r for r in m.records(reference,pin(reference))}
            self.assertIn(str(offset),found);self.assertIn(str(request),found)
            self.assertEqual(found[str(offset)]['sha256'],pin(offset))


if __name__=='__main__':unittest.main()


def test_original_three_steps_and_evaluation_with_readonly_views(tmp_path,native):
    from analysis import p_full_cached_backend as m,p_full_compact_inputs as compact
    from analysis.p_full_engineering import capture
    original=native.fr.synthetic_data();cached={}
    for key in ('rows','queries'):
        path=tmp_path/(key+'.jsonl')
        path.write_bytes(b''.join(json.dumps(row,sort_keys=True).encode()+b'\n'for row in original[key]))
        cached[key]=compact.JsonRows.build(path,tmp_path/(key+'-offsets'),len(original[key]),
                                          hashlib.sha256(path.read_bytes()).hexdigest(),lambda:None)
    data=m.data_view(cached,original['catalog'],sorted(original['support']),original['content'].numpy())
    # Only this CPU synthetic oracle retains the original fixture's synthetic flag.
    data['synthetic']=original['synthetic']
    config={**native.default_config(),'batch_size':8}
    assert capture(native,original,config,'cpu',lambda:None)==capture(compact.facade(native,lambda:None),data,config,'cpu',lambda:None)


def test_formal_module_cli_dry_and_invalid_subcommand():
    from analysis import p_full_cached_backend as m
    env={**os.environ,'PYTHONPATH':str(m.ROOT/'src')+os.pathsep+str(m.ROOT),'PYTHONDONTWRITEBYTECODE':'1'}
    for stage,status in [('dry',0),('illegal',2)]:
        result=subprocess.run([sys.executable,'-B','-m','analysis.p_full_cached_backend',stage],cwd=m.ROOT,env=env,capture_output=True,text=True)
        assert result.returncode==status
        if not status:assert json.loads(result.stdout)['training']is False


def test_cached_consumer_real_launcher_deployment_metadata_dry(tmp_path):
    import io,tarfile
    from analysis import p_full_cached_backend as m
    deployment=tmp_path/'deployment';deployment.mkdir()
    ops=tmp_path/'ops-source';ops.mkdir()
    for target,args in ((deployment,['18f98f56','analysis','src']),(ops,['5840e6b','ops'])):
        with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',*args])))as archive:
            archive.extractall(target,filter='data')
    (deployment/'analysis/p_full_cached_backend.py').write_bytes(Path(m.__file__).read_bytes())
    native=tmp_path/'native';(native/'docs').mkdir(parents=True)
    (native/'docs/p3c_validation.md').write_bytes(b'synthetic data channel fixture\n')
    subprocess.run(['git','init','-q',str(deployment)],check=True)
    subprocess.run(['git','-C',str(deployment),'add','.'],check=True)
    subprocess.run(['git','-C',str(deployment),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','synthetic fixture'],check=True)
    request=tmp_path/'request.json';request.write_text('{}')
    worker=[sys.executable,'-B','-m','analysis.p_full_cached_backend','dry']
    script="""import json,sys
from vkgr_ops import launcher
root,cwd,request,worker=json.loads(sys.argv[1])
a=launcher.build_parser().parse_args(['--root',root,'--run-id','p-cached-backend-fixture','--kind','ops','--cwd',cwd,'--config',request,'--input',request,'--cpu-cores','1','--ram-mib','1024','--vram-mib','0','--disk-gib','0','--wall-hours','1','--',*worker])
metadata=launcher.prepare(launcher.spec_from_args(a))
assert metadata['git']['dirty']is False and len(metadata['inputs'])==1
print(json.dumps({'status':'prepared_dry'}))
"""
    result=subprocess.run([sys.executable,'-B','-c',script,json.dumps([str(tmp_path/'ops-root'),str(deployment),str(request),worker])],env={**os.environ,'PYTHONPATH':str(ops/'ops')},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='prepared_dry'
    result=subprocess.run(worker,cwd=deployment,env={**os.environ,'PYTHONPATH':str(deployment)+os.pathsep+str(deployment/'src')},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['training']is False
    assert not(tmp_path/'ops-root/runs').exists()


def test_read_context_guard_counts_old_cache_and_parallel(tmp_path):
    from types import SimpleNamespace
    from analysis import p_full_cached_backend as m
    p=tmp_path.resolve();out=p/'output';cache=p/'sealed-cache';parallel=p/'output-parallel-eval'
    for root in (out,cache,parallel):root.mkdir()
    (out/'payload').write_bytes(b'x');(cache/'side').write_bytes(b'x'*6);(parallel/'vectors').write_bytes(b'x'*6)
    value={'native_root':str(p/'native'),'output_root':str(out),'disk_cap_bytes':10};captured=[]
    def consume(ref,guard):captured.append(guard);raise RuntimeError('synthetic acquisition stop')
    native=SimpleNamespace(b1=SimpleNamespace(resource_guard=lambda *a:lambda:None))
    with patch.object(m.backend,'read_transport',return_value=value),patch.object(m,'reader',return_value=(SimpleNamespace(consume=consume),{'seal':{}})),patch.object(m,'origin',return_value=({'cache_root':str(cache)},value)),patch.object(m,'same_scientific_inputs'),patch.object(m.backend,'native_for',return_value=native):
        with __import__('pytest').raises(RuntimeError,match='synthetic acquisition stop'):
            m.functions('/reference','a'*64)['read_context']('/inputs','/request','b'*64)
    with __import__('pytest').raises(OSError):captured[0].projected(0)


def test_parallel_projected_allocations_share_sealed_cache_cap(tmp_path):
    from types import SimpleNamespace
    from analysis import p_full_cached_backend as m,p_full_parallel_eval as parallel
    def bind_fixture(native,temp_root,*,output_root,disk_cap_bytes):
        return SimpleNamespace(score=evaluate_vectors,b1=SimpleNamespace(resource_guard=lambda *a:lambda:None))
    output=tmp_path.resolve()/'out';cache=tmp_path.resolve()/'cache';cache.mkdir();(cache/'payload').write_bytes(b'x'*6)
    guard=m.compact.CombinedGuard(lambda:None,[output,cache],10)
    with patch.object(parallel,'bind',bind_fixture),patch.object(parallel,'evaluate_vectors',return_value='original-scoring')as score:
        adapted=m.sealed_parallel(SimpleNamespace(),output,10,guard)
        assert adapted.score(disk_check=lambda size:None)=='original-scoring'
        with __import__('pytest').raises(OSError):score.call_args.kwargs['disk_check'](5)
        score.call_args.kwargs['disk_check'](4)
        (output).mkdir();(output/'checkpoint').write_bytes(b'x'*5)
        with __import__('pytest').raises(OSError):adapted.b1.resource_guard(output,10)()
