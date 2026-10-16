"""Synthetic-only tests for fixed M2 metadata readout."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('m2meta',Path(__file__).with_name('multi_interest_metadata_read_20261017.py'))
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class MetadataTests(unittest.TestCase):
    def fixture(self):
        rid='p-seqm2-k4-s42'; config={'seed':42}
        binding={'source':{'git_commit':m.COMMIT},'synthetic':False,'test_gate':'closed_hash_only','queries_sha256':'query'}
        manifest={'config':config,'binding':binding,'selection':{'best_epoch':1,'epoch':4,'best_metric':[.2,.1]},'best':{'epoch':1},'evaluations':[{'epoch':1,'predictions_ref':'predictions-epoch-001.json','predictions_sha256':'a'*64,'diagnostics':{'head':[1,2,3,4]}}],'ablation_g0':{'best_epoch':1,'recall@50':.1,'predictions_ref':'predictions-best-g0.json','predictions_sha256':'b'*64}}
        job={'run_id':rid,'seed':42,'config':'/config.json','argv':['python','train','--output','/output'],'inputs':['/input']}
        entry={'job':job,'expected':{'config_sha256':m.digest(config),'expected_binding_sha256':m.digest(binding),'data_hashes':{'queries_sha256':'query'}}}
        return manifest,entry,config
    def test_identity_and_metadata(self):
        a,e,c=self.fixture(); got=m.metadata(a,e,c,{'commit':m.COMMIT})
        self.assertEqual(got['best_epoch'],1);self.assertEqual(got['stop_epoch'],4)
        self.assertEqual(got['best_predictions']['ref'],'predictions-epoch-001.json')
        self.assertEqual(got['ablation_g0']['recall@50'],.1)
    def test_identity_mutations_fail(self):
        a,e,c=self.fixture()
        for which in ('source','seed','data','best','ref','nonfinite'):
            bad=copy.deepcopy(a)
            if which=='source':bad['binding']['source']['git_commit']='other'
            if which=='seed':bad['config']['seed']=2024
            if which=='data':bad['binding']['queries_sha256']='other'
            if which=='best':bad['best']['epoch']=2
            if which=='ref':bad['evaluations'][0]['predictions_ref']='../checkpoint.json'
            if which=='nonfinite':bad['ablation_g0']['recall@50']=float('nan')
            with self.assertRaises(ValueError):m.metadata(bad,e,c,{'commit':m.COMMIT})
    def test_verify_derivation_is_full_job_comparison(self):
        _,e,_=self.fixture();v=m.derive_verify(e['job'],'a'*64,Path('/synthetic'))
        self.assertEqual(v['run_id'],'p-seqm2-k4-s42-verify')
        self.assertNotIn('--output',v['argv']);self.assertIn('--manifest-sha',v['argv'])
        self.assertEqual(v['disk_gib'],1);self.assertEqual(v['wall_hours'],3.)
    def test_small_reader_rejects_model_names_symlink_size_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'latency.json';p.write_text('{"total_ms":2}')
            self.assertEqual(m.small_json(p,{},hashlib.sha256(p.read_bytes()).hexdigest())['total_ms'],2)
            with self.assertRaises(ValueError):m.small_json(p,{},'0'*64)
            q=Path(tmp)/'checkpoint.json';q.write_text('{}')
            with self.assertRaises(ValueError):m.small_json(q,{})
            q=Path(tmp)/'alias.json';q.symlink_to(p)
            with self.assertRaises(ValueError):m.small_json(q,{})
            q=Path(tmp)/'huge.json';q.write_bytes(b' '*65537)
            with self.assertRaises(ValueError):m.small_json(q,{})
    def test_latency_scan_nonrecursive_and_never_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'latency.json').write_text('{"user_ms":1,"total_ms":4}')
            (p/'checkpoint-latency.json').write_text('not json')
            (p/'predictions-latency.json').write_text('not json')
            (p/'subdir').mkdir();(p/'subdir/latency.json').write_text('not json')
            got=m.latency_inventory([p],{})
            self.assertEqual(len(got['receipts']),1)
            self.assertEqual(got['receipts'][0]['payload']['total_ms'],4)
            self.assertFalse(got['recursive'])
    def test_summary_nested_timing_is_inventoried(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'summary-final.json').write_text('{"states":[],"latency":{"total_ms":2}}')
            got=m.latency_inventory([p],{})
            self.assertEqual(len(got['receipts']),1)
            self.assertEqual(got['receipts'][0]['payload']['latency']['total_ms'],2)

    def test_pin_before_exec(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.py';p.write_text('VALUE=3\n')
            self.assertEqual(m.pinned_module(p,hashlib.sha256(p.read_bytes()).hexdigest(),{}).VALUE,3)
            with self.assertRaises(ValueError):m.pinned_module(p,'0'*64,{})
    def test_no_mutation_or_model_imports(self):
        import ast
        tree=ast.parse(Path(m.__file__).read_text())
        forbidden={'write_text','write_bytes','unlink','mkdir','Popen','run','system','load_archive','rglob','torch','export_vectors'}
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(calls & forbidden)

if __name__=='__main__':unittest.main()

class PrivatePathsTest(unittest.TestCase):
 def test_relative_private_location_rejected(self):
  from analysis import multi_interest_metadata_read_20261017 as r
  with self.assertRaises(ValueError):r.configure_paths({'phase2':'relative','ops':'/workspace/fixture','deployment':'/workspace/fixture'})
