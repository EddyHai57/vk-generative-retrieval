"""G0 CPU synthetic integration; expectations independent of the adapter."""
import copy
import json
import math
from pathlib import Path
import unittest
import tempfile
import hashlib
import subprocess
import sys
import os
from vkgr import generative_foundation as g0

FIXTURE = Path(__file__).parent / "fixtures/g0_synthetic.json"

def fixture():
    return json.loads(FIXTURE.read_text())

class FoundationTests(unittest.TestCase):
    def test_strict_prefix_sid_targets_and_first_oov_query(self):
        prepared = g0.prepare_fixture(fixture())
        self.assertEqual([r['history'] for r in prepared['examples']], [[10], [10,20], [30]])
        self.assertEqual([r['target'] for r in prepared['examples']], [20,10,40])
        self.assertEqual([q['history'] for q in prepared['queries']], [[10,20,10],[30,40]])
        self.assertEqual([q['target'] for q in prepared['queries']], [20,999])
        self.assertEqual(prepared['codec'].item(20), [2,5,8,12])
        self.assertEqual(prepared['codec'].catalog.decode([0,0,0,1]),20)
        self.assertEqual(prepared['features'].shape,(12,64))

    def test_rejects_future_catalog_sid_and_holdout_boundary_drift(self):
        cases = []
        def bad(edit):
            data=fixture();edit(data);cases.append(data)
        bad(lambda d:d.update(synthetic=False))
        bad(lambda d:d.update(test={"week":26,"rows":[]}))
        bad(lambda d:d['train_partitions'][0].update(week=25))
        bad(lambda d:d['validation'].update(week=26))
        bad(lambda d:d['catalog'].__setitem__(0,True))
        bad(lambda d:d['catalog'].__setitem__(1,10))
        bad(lambda d:d['train_partitions'][0]['rows'][0].update(item_id=999))
        bad(lambda d:d['train_partitions'][0]['rows'][0].update(like=None))
        bad(lambda d:d['validation']['rows'][0].update(like=None))
        bad(lambda d:d['sids'].__setitem__(1,[0,0,0,0]))
        bad(lambda d:d['sids'][0].__setitem__(0,-1))
        bad(lambda d:d['sids'][0].__setitem__(0,True))
        bad(lambda d:d['sids'][0].__setitem__(0,3))
        bad(lambda d:d['sids'].__setitem__(0,[0,0,0]))
        bad(lambda d:d['features'][0].__setitem__(0,float('nan')))
        bad(lambda d:d['features'][0].pop())
        for i,data in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                g0.prepare_fixture(data)

    def test_preserves_repeats_and_last_50_without_future(self):
        data=fixture()
        data['train_partitions'][0]['rows'].extend([{'user_id':1,'item_id':10,'like':True} for _ in range(51)])
        prepared=g0.prepare_fixture(data)
        self.assertEqual(prepared['queries'][0]['history'],[10]*50)
        self.assertEqual(len(prepared['examples']),54)
        for row in prepared['examples']:
            self.assertLess(row['history_positions'][-1],row['target_position'])
            self.assertLessEqual(len(row['history']),50)

    def test_cpu_training_decode_reproducible_and_item_oracle(self):
        first=g0.run_fixture(fixture()); second=g0.run_fixture(fixture())
        self.assertEqual(first,second)
        self.assertNotEqual(first['initial_parameters_sha256'],first['final_parameters_sha256'])
        self.assertEqual(first['drawn_example_ids'],[0,2,1])
        self.assertEqual(first['batch_sizes'],[2,1])
        self.assertEqual(first['device'],'cpu')
        self.assertEqual(len(first['losses']),2)
        self.assertTrue(all(math.isfinite(x) for x in first['losses']))
        targets=[20,999]; members=set(fixture()['catalog'])
        for predictions in first['predictions']:
            self.assertEqual(len(predictions),12)
            self.assertEqual(len(set(predictions)),12)
            self.assertEqual(set(predictions),members)
        for k in (10,20,50):
            ranks=[p.index(t)+1 if t in p[:k] else None for p,t in zip(first['predictions'],targets)]
            self.assertEqual(first['metrics']['all'][f'recall@{k}'],sum(r is not None for r in ranks)/2)
            self.assertEqual(first['metrics']['all'][f'ndcg@{k}'],sum(1/math.log2(r+1) if r else 0 for r in ranks)/2)
            self.assertEqual(first['metrics']['oov'][f'recall@{k}'],0)
            self.assertEqual(first['metrics']['train_observed'][f'ndcg@{k}'],2*first['metrics']['all'][f'ndcg@{k}'])
        self.assertEqual(first['decode_audits'],[{'invalid':0,'deadend':0,'collision':0,'max_active':12}]*2)

    def test_semantic_collision_does_not_count_as_item_hit(self):
        q=[{'target':20,'oov':False},{'target':999,'oov':True}]
        result=g0.evaluate_items([[10],[10]],q,[10,20])
        self.assertEqual(result['all']['count'],2)
        for k in (10,20,50):
            self.assertEqual(result['all'][f'recall@{k}'],0)
            self.assertEqual(result['all'][f'ndcg@{k}'],0)

    def test_manifest_replay_hash_tamper_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            manifest=g0.write_manifest(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(g0.verify_manifest(path,digest),manifest)
            self.assertEqual(manifest['test_gate'],'closed')
            self.assertEqual(manifest['summary']['query_count'],2)
            self.assertNotIn('predictions',manifest['summary'])
            self.assertEqual(len(manifest['input_hashes']['queries']),64)
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):g0.write_manifest(path)
            self.assertEqual(path.read_bytes(),original)
            with self.assertRaises(ValueError):g0.verify_manifest(path,'0'*64)
            modified=json.loads(original);modified['summary']['query_count']=512
            path.write_text(json.dumps(modified))
            with self.assertRaises(ValueError):g0.verify_manifest(path,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_cli_help_and_test_split_rejection(self):
        env={**os.environ,'PYTHONPATH':str(FIXTURE.parents[2]/'src'),'OMP_NUM_THREADS':'1'}
        help_run=subprocess.run([sys.executable,'-B','-m','vkgr.generative_foundation','--help'],env=env,capture_output=True,text=True)
        self.assertEqual(help_run.returncode,0,help_run.stderr)
        self.assertIn('synthetic',help_run.stdout)
        refused=subprocess.run([sys.executable,'-B','-m','vkgr.generative_foundation','run','--split','test'],env=env,capture_output=True,text=True)
        self.assertNotEqual(refused.returncode,0)

    def test_rehashed_manifest_type_drift_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            original=g0.write_manifest(path)
            for edit in (lambda d:d.update(schema_version=True),lambda d:d['config'].update(seed=42.0)):
                value=copy.deepcopy(original);edit(value);path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):g0.verify_manifest(path,hashlib.sha256(path.read_bytes()).hexdigest())
