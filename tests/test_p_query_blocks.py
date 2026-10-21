import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from vkgr.throughput_config import Engineering
from vkgr.p_query_blocks import parallel_topk
from analysis.p_query_blocks import parity, bind_original_cost

class QueryBlockTests(unittest.TestCase):
    def arrays(self,root,q=35,n=4103):
        rng=np.random.default_rng(42)
        values=[rng.normal(size=(q,64)).astype('f4'),rng.normal(size=(n,64)).astype('f4'),np.arange(n,dtype='u4')[::-1].copy()]
        values[0][:2]=0
        out=[]
        for name,value in zip(('queries','items','ids'),values):
            path=root/(name+'.npy');np.save(path,value);out.append(np.load(path,mmap_mode='r'))
        return out
    def test_small_boundary_tail_ties_exact_score_and_item_bytes(self):
        for q,n in ((0,1),(1,1),(32,4096),(35,4103)):
            with self.subTest(q=q,n=n),tempfile.TemporaryDirectory() as raw:
                arrays=self.arrays(Path(raw),q,n)
                result=parity(*arrays,Engineering(eval_workers=2,use_mmap=True),k=50)
                self.assertTrue(result['score_bytes_equal_to_original'])
                self.assertTrue(result['prediction_bytes_equal_to_original'])
                self.assertFalse(result['original_cost_terminal_compared'])
                self.assertEqual(result['queries'],q)
    def test_parallel_rejects_catalog_copy_and_oversubscribed_cpu_before_kernel(self):
        with tempfile.TemporaryDirectory() as raw:
            arrays=self.arrays(Path(raw),2,3)
            with self.assertRaisesRegex(ValueError,'mmap'):
                parallel_topk(arrays[0],np.asarray(arrays[1]),arrays[2],2,Engineering(eval_workers=2))
            with patch('vkgr.p_query_blocks.cpu_capacity',return_value=2),patch('vkgr.p_query_blocks.exact_topk') as kernel:
                with self.assertRaisesRegex(ValueError,'CPU'):
                    parallel_topk(*arrays,2,Engineering(eval_workers=2,use_mmap=True))
                kernel.assert_not_called()
    def test_guard_and_source_pin_fail_before_kernel(self):
        with tempfile.TemporaryDirectory() as raw:
            arrays=self.arrays(Path(raw),1,3)
            with patch('vkgr.p_query_blocks.guard',side_effect=MemoryError('guard')),patch('vkgr.p_query_blocks.exact_topk') as kernel:
                with self.assertRaises(MemoryError):parallel_topk(*arrays,2,Engineering())
                kernel.assert_not_called()
            with patch('analysis.p_query_blocks.REFERENCE_SHA','0'*64):
                with self.assertRaisesRegex(ValueError,'baseline'):parity(*arrays,Engineering(),k=2)
    def test_missing_original_terminal_stops_before_arrays(self):
        with tempfile.TemporaryDirectory() as raw:
            with patch('numpy.load') as load,self.assertRaises((ValueError,FileNotFoundError,subprocess.CalledProcessError)):
                bind_original_cost(Path(raw),'0'*40,Path(raw),Path(raw)/'lock',Path(raw)/'ops')
            load.assert_not_called()
    def test_cli_dry_and_illegal_subcommand(self):
        cmd=[sys.executable,'-B','-m','analysis.p_query_blocks']
        good=subprocess.run(cmd+['dry'],capture_output=True,text=True)
        self.assertEqual(good.returncode,0,good.stderr)
        self.assertEqual(json.loads(good.stdout)['status'],'dry')
        self.assertNotEqual(subprocess.run(cmd+['illegal'],capture_output=True,text=True).returncode,0)
    def test_cli_parity_receipt_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);self.arrays(root,35,4103)
            argv=[sys.executable,'-B','-m','analysis.p_query_blocks','parity','--workers','2',
                  '--memory-limit-bytes',str(2*1024**3),'--output',str(root/'parity.json')]
            for name in ('queries','items','ids'):argv+=['--'+name,str(root/(name+'.npy'))]
            first=subprocess.run(argv,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            value=json.loads(first.stdout)
            self.assertTrue(value['score_bytes_equal_to_original'])
            self.assertEqual(set(value['input_sha256']),{'queries','items','ids'})
            self.assertEqual(value,json.loads((root/'parity.json').read_text()))
            again=subprocess.run(argv,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)
            self.assertIn('exclusive output',again.stderr)

    def test_full_query_count_with_small_catalog(self):
        with tempfile.TemporaryDirectory() as raw:
            arrays=self.arrays(Path(raw),26827,1)
            value=parity(*arrays,Engineering(eval_workers=2,use_mmap=True),k=50)
            self.assertEqual(value['queries'],26827)
            self.assertEqual(value['score_blocks_checked'],839)
            self.assertTrue(value['score_bytes_equal_to_original'])

    def test_original_cost_failed_terminal_and_payload_tampering_are_rejected(self):
        import shutil
        from analysis import a40_cost as cost
        with tempfile.TemporaryDirectory() as raw:
            base=Path(raw).resolve();engine=base/'engine';(engine/'analysis').mkdir(parents=True)
            shutil.copyfile(cost.__file__,engine/'analysis/a40_cost.py')
            root=base/'arrays';root.mkdir();lock=base/'uv.lock';lock.write_text('fixture')
            for name in ('cost','inputs','preflight'):(root/(name+'.json')).write_text('{}')
            ops=base/'ops';run=ops/'runs/a40-exact-run-v1';(run/'logs').mkdir(parents=True)
            source='a'*40
            oldroot=cost.ROOT;cost.ROOT=engine
            try:argv=cost.worker_argv('run',root,lock,ops)
            finally:cost.ROOT=oldroot
            pins=[{'path':str(path.resolve()),'sha256':cost.sha_file(path)} for path in (lock,root/'inputs.json',root/'preflight.json')]
            record={'schema':'vkgr-run-v1','run_id':run.name,'kind':'ops','git':{'commit':source,'dirty':False},
                    'cwd':str(engine),'argv':argv,'config':{'path':str(engine/'configs/throughput/a40-m153999.yaml'),'sha256':cost.YAML_SHA},'inputs':pins}
            (run/'run.json').write_text(json.dumps(record));(run/'logs/stdout.log').write_text('{}')
            for status,stdout in (('running','{}'),('succeeded','{"forged":true}')):
                (run/'terminal.json').write_text(json.dumps({'schema':'vkgr-terminal-v1','run_id':run.name,'status':status,'exit_code':0}))
                (run/'logs/stdout.log').write_text(stdout)
                with patch('analysis.p_query_blocks.git',side_effect=[source,'']),patch('numpy.load') as load:
                    with self.assertRaisesRegex(ValueError,'binding|stdout'):
                        bind_original_cost(engine,source,root,lock,ops)
                    load.assert_not_called()

    @unittest.skipUnless(os.environ.get('VKGR_P3C_DATA_SOURCE') or subprocess.run(
        ['git','cat-file','-e','7041c90:docs/p3c_validation.md'],capture_output=True).returncode==0,
        'private fixed P3C data-channel source is unavailable')
    def test_real_launcher_prepare_to_dry(self):
        result=subprocess.run([sys.executable,'-B','tests/fixtures/p_query_blocks_chain.py'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'],'real_launcher_prepare_and_dry_verified')
if __name__=='__main__':unittest.main()
