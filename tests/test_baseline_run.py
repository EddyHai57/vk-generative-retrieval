import unittest
import tempfile
import json
from pathlib import Path
from vkgr import baseline_run as r

class RunnerTests(unittest.TestCase):
    def test_config_strict_and_fixed(self):
        good={'task_id':'P4','seed':42,'max_queries':512,'ks':[10,20],'query_block':32,'item_block':4096}
        self.assertEqual(r.validate_config(good),good)
        for key,value in [('seed',True),('seed',43),('max_queries',513),('ks',[20,10]),('query_block',0),('item_block',200000)]:
            with self.assertRaises(ValueError):r.validate_config(dict(good,**{key:value}))
        with self.assertRaises(ValueError):r.validate_config(dict(good,extra=1))
    def test_hash_and_prediction_metric_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.json';path.write_text('{"a":1}')
            digest=r.file_sha256(path)
            self.assertEqual(r.read_checked(path,digest),{'a':1})
            path.write_text('{"a":2}')
            with self.assertRaises(ValueError):r.read_checked(path,digest)
        queries=[{'user_id':1,'history':[1],'target':1,'oov':False}]
        predicted=[[1,2]];catalog=[1,2]
        result=r.evaluate_predictions(predicted,queries,catalog,[10,20])
        self.assertEqual(result['all']['recall@10'],1)
        with self.assertRaises(ValueError):r.verify_metrics(dict(result,all=dict(result['all'],**{'recall@10':0})),predicted,queries,catalog,[10,20])
        r.verify_metrics(result,predicted,queries,catalog,[10,20])

    def test_evaluation_and_bound_output(self):
        import numpy as np
        queries={split:[{'user_id':1,'history':[1,1,2],'target':1,'oov':False},{'user_id':2,'history':[3],'target':99,'oov':True}] for split in ('validation','test')}
        inputs={'catalog':[1,2,3],'features':np.array([[1,0],[-1,0],[0,0]],dtype=np.float32),'popularity':np.array([1,5,0]),'queries':queries}
        result,artifacts=r.evaluate_inputs(inputs,dict(r.CONFIG,query_block=1,item_block=1))
        self.assertEqual(artifacts['test']['content_predictions'],[[1,3,2],[1,2,3]])
        self.assertEqual(artifacts['test']['popularity_predictions'],[[2,1,3],[2,1,3]])
        self.assertEqual(result['test']['zero_centroids'],1)
        self.assertEqual(result['test']['content']['all']['recall@10'],0.5)
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'new'
            r.write_outputs(output,result,artifacts)
            self.assertTrue((output/'metrics.json').is_file())
            before=(output/'metrics.json').read_bytes()
            with self.assertRaises(FileExistsError):r.write_outputs(output,result,artifacts)
            self.assertEqual((output/'metrics.json').read_bytes(),before)

    def test_static_contract_refuses_tamper_before_data_read(self):
        root=Path(__file__).resolve().parents[1]
        original=root/'artifacts/p3-data-contract-manifest.json'
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/'frozen-contract-manifest.json';copy.write_bytes(original.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'hash'):
                r.load_static(copy,Path(d)/'absent',Path(d)/'raw')

    def test_manifest_replay_rejects_rehashed_predictions(self):
        import numpy as np
        from unittest.mock import patch
        queries={split:[{'user_id':1,'history':[1],'target':1,'oov':False}] for split in ('validation','test')}
        inputs={'catalog':[1,2],'features':np.array([[1,0],[0,1]],dtype=np.float32),'popularity':np.array([1,0]),'queries':queries,'evidence':{'inputs':[]}}
        with tempfile.TemporaryDirectory() as d, patch.object(r,'load_static',return_value=inputs):
            config=Path(d)/'config.json';config.write_text(json.dumps(r.CONFIG));out=Path(d)/'run'
            r.run('freeze','holdout','raw',config,out)
            r.verify_run(out/'run_manifest.json')
            artifact=out/'test-audit.json';data=json.loads(artifact.read_text());data['content_predictions']=[[2,1]];artifact.write_bytes(r.canonical_bytes(data))
            manifest=out/'run_manifest.json';m=json.loads(manifest.read_text());m['artifacts']['test-audit.json']=r.file_sha256(artifact);manifest.write_bytes(r.canonical_bytes(m))
            with self.assertRaisesRegex(ValueError,'prediction'):
                r.verify_run(manifest)
            with self.assertRaises(FileExistsError):r.run('freeze','holdout','raw',config,out)

    def test_relative_input_roots_run_then_verify(self):
        import os,numpy as np
        from unittest.mock import patch
        queries={split:[{'user_id':1,'history':[1],'target':1,'oov':False}] for split in ('validation','test')}
        def loader(manifest,holdout,raw):
            return {'catalog':[1,2],'features':np.array([[1,0],[0,1]],dtype=np.float32),'popularity':np.array([1,0]),'queries':queries,
                    'evidence':{'inputs':[{'path':str(Path(manifest))},{'path':str(Path(holdout))},{'path':str(Path(raw))}]}}
        prior=Path.cwd()
        with tempfile.TemporaryDirectory() as d, patch.object(r,'load_static',side_effect=loader):
            try:
                os.chdir(d);Path('config.json').write_text(json.dumps(r.CONFIG))
                r.run('freeze','holdout','raw','config.json','run')
                r.verify_run(Path(d)/'run/run_manifest.json')
            finally:os.chdir(prior)

if __name__=='__main__':unittest.main()
