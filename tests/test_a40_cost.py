import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class CostTests(unittest.TestCase):
    def test_untracked_manifest_is_rejected_before_array_open(self):
        from analysis import a40_cost as cost
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw).resolve();lock=root/'uv.lock';lock.write_text('fixture')
            doc={'schema':'a40-p-exact-synthetic-v1','data_kind':'synthetic','items':12_000_000,
                 'queries':26_827,'dim':64,'top_k':50,'synthetic_seed':42,
                 'generator_sha256':cost.sha_file(cost.__file__),'selected_yaml_sha256':cost.YAML_SHA,
                 'baseline_sha256':cost.REFERENCE_SHA,
                 'files':[{'path':p,'size':1,'sha256':'0'*64}for p in ['items.npy','queries.npy','ids.npy']]}
            (root/'inputs.json').write_text(json.dumps(doc))
            with patch.object(cost,'runtime',return_value={}),patch.object(cost,'selected'), \
                 patch.object(cost,'source_sha',return_value='source'), \
                 patch('numpy.load') as load,self.assertRaisesRegex(ValueError,'producer receipt'):
                cost.inputs(root,lock,root/'ops')
            load.assert_not_called()

    def test_source_and_yaml_bytes_are_frozen(self):
        from analysis import a40_cost as cost
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw)
            for name,function in [('REFERENCE',cost.reference),('YAML',cost.selected)]:
                path=root/name;path.write_bytes(getattr(cost,name).read_bytes()+b'\n')
                with patch.object(cost,name,path),self.assertRaises(ValueError):function()

    def test_original_producer_stdout_and_argv_binding(self):
        from analysis.a40_cost import accepted_stage,ROOT
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw).resolve();run=root/'runs'/'a40-exact-generate-v1';(run/'logs').mkdir(parents=True)
            doc={'status':'generated','files':[]};argv=['python','generator']
            (run/'run.json').write_text(json.dumps({'schema':'vkgr-run-v1','run_id':run.name,'kind':'ops',
                'git':{'commit':'source','dirty':False},'cwd':str(ROOT),'argv':argv,
                'config':{'path':'yaml','sha256':'cfg'},'inputs':[{'path':'lock','sha256':'locked'}]}))
            (run/'terminal.json').write_text(json.dumps({'schema':'vkgr-terminal-v1','run_id':run.name,'status':'succeeded','exit_code':0}))
            (run/'logs/stdout.log').write_text(json.dumps(doc)+'\n')
            accepted_stage(root,'generate','source',argv,{'lock':'locked'},{'path':'yaml','sha256':'cfg'},doc)
            with self.assertRaises(ValueError):accepted_stage(root,'generate','other',argv,{'lock':'locked'},{'path':'yaml','sha256':'cfg'},doc)
            with self.assertRaises(ValueError):accepted_stage(root,'generate','source',['wrong'],{'lock':'locked'},{'path':'yaml','sha256':'cfg'},doc)
            with self.assertRaises(ValueError):accepted_stage(root,'generate','source',argv,{'lock':'locked'},{'path':'yaml','sha256':'cfg'},{'status':'forged'})

    def test_run_requires_bound_formal_preflight(self):
        from analysis.a40_cost import require_preflight
        from analysis.a40_cost import REFERENCE_SHA
        good={'status':'ready','engineering':False,'inputs_manifest_sha256':'inputs',
              'script_sha256':'script','selected_yaml_sha256':'yaml','worker_parity':[1,2,8],
              'score_bytes_equal_to_original':True,'synthetic_input':True,'science_data_read':False,
              'baseline_sha256':REFERENCE_SHA,'generation_stdout_sha256':'generation',
              'items':12_000_000,'queries':26_827,'dim':64,'top_k':50,
              'parity_scope':'35x4103 synthetic; full fixed block and tails'}
        require_preflight(good,'inputs','script','yaml','generation')
        for change in ({'status':'failed'},{'inputs_manifest_sha256':'other'},
                       {'script_sha256':'other'},{'selected_yaml_sha256':'other'},
                       {'worker_parity':[1]},{'score_bytes_equal_to_original':False}):
            with self.assertRaises(ValueError):require_preflight({**good,**change},'inputs','script','yaml','generation')

    def test_original_reference_and_ties(self):
        from analysis.a40_cost import reference
        baseline = reference()
        q = np.zeros((2,64),dtype='f4'); items = np.zeros((5,64),dtype='f4')
        ids = np.array([4,2,3,1,0],dtype='u4')
        self.assertEqual(baseline(q,items,ids,3).tolist(),[[0,1,2],[0,1,2]])

    def test_synthetic_arrays_are_reproducible_and_exclusive(self):
        from analysis.a40_cost import build_arrays
        with tempfile.TemporaryDirectory() as raw:
            a=Path(raw)/'a'; b=Path(raw)/'b'; a.mkdir(); b.mkdir()
            first=build_arrays(a,71,35); second=build_arrays(b,71,35)
            self.assertEqual(first,second)
            with self.assertRaises(FileExistsError): build_arrays(a,71,35)

    def test_exact_memory_disk_and_retention_bounds(self):
        from analysis.a40_cost import array_bytes, bhkm_bound
        self.assertEqual(array_bytes(12_000_000,26_827),3_126_867_712)
        # The fit bound reserves every level and maximum iteration; replay is additive.
        value=bhkm_bound(12_000_000,229,3,10,8192)
        self.assertGreater(value['fit_bytes'],10**12)
        self.assertEqual(value['fit_plus_replay_bytes'],value['fit_bytes']*2)

    def test_cli_dry_and_illegal_and_missing_execute(self):
        cmd=[sys.executable,'-B','-m','analysis.a40_cost']
        good=subprocess.run(cmd+['dry'],capture_output=True,text=True)
        self.assertEqual(good.returncode,0,good.stderr)
        self.assertEqual(json.loads(good.stdout)['top_k'],50)
        for args in (['unknown'],['generate'],['run']):
            self.assertNotEqual(subprocess.run(cmd+args,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
