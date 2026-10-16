"""Synthetic-only regression for the read-only combination analysis."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("combo", Path(__file__).with_name("feat04combo_paired_read.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

class ComboReadTests(unittest.TestCase):
    def test_feature_rules_and_significance_are_separate(self):
        pairs = [{"delta_hits": d, "mcnemar_exact_two_sided": p} for d, p in [(6,.01),(6,.04),(0,.8)]]
        got = r.mechanical(pairs)
        self.assertEqual(got["classification"], "inconclusive")
        self.assertFalse(got["promotion_statistical_condition"])
        pairs[2]["delta_hits"] = 6
        self.assertTrue(r.mechanical(pairs)["promotion_statistical_condition"])
        self.assertNotIn("anchor", r.mechanical(pairs))
    def test_incomplete_rules_fail_closed(self):
        self.assertEqual(r.mechanical([])["classification"], "UNKNOWN")
    def test_pin_before_execution_and_no_file_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"source.py";p.write_text("VALUE=3\n")
            reads={}
            m=r.pinned_module(p, hashlib.sha256(p.read_bytes()).hexdigest(), reads)
            self.assertEqual(m.VALUE,3)
            self.assertEqual(len(reads),1)
            with self.assertRaisesRegex(ValueError,"source SHA"):
                r.pinned_module(p,"0"*64,{})
    def test_combo_binding_rejects_config_source_and_preflight_mutations(self):
        manifest={"config":{"seed":42},"binding":{"source":{"git_commit":"abc"},"catalog_sha256":"cat","queries_sha256":"q","support_sha256":"s","rows_sha256":"r","content_sha256":"c"}}
        entry={"job":{"run_id":"p-feat04combo-cde-s42","seed":42},"expected":{"expected_binding_sha256":r.plain_digest(manifest["binding"]),"data_hashes":{k:v for k,v in manifest["binding"].items() if k!="source"}}}
        r.check_combo_identity(manifest,entry,manifest["config"],{"commit":"abc"})
        for key in ("config","binding"):
            bad=copy.deepcopy(manifest)
            if key=="config":bad[key]["seed"]=2024
            else:bad[key]["source"]["git_commit"]="changed"
            with self.assertRaises(ValueError):r.check_combo_identity(bad,entry,manifest["config"],{"commit":"abc"})
        bad=copy.deepcopy(entry);bad["expected"]["data_hashes"]["support_sha256"]="changed"
        with self.assertRaises(ValueError):r.check_combo_identity(manifest,bad,manifest["config"],{"commit":"abc"})
    def test_candidate_catalog_and_support_mismatch_fail(self):
        r.candidate_check([[1,2]],{1,2},"s","s")
        with self.assertRaises(ValueError):r.candidate_check([[1,3]],{1,2},"s","s")
        with self.assertRaises(ValueError):r.candidate_check([[1,2]],{1,2},"s","other")
    def test_accepted_archive_checks_real_schema_and_stored_slices(self):
        from types import SimpleNamespace
        rid="p-feat04combo-cde-s42"
        manifest={"config":{"seed":42},"binding":{"synthetic":False,"test_gate":"closed_hash_only","support_sha256":"s"},"selection":{"best_epoch":1,"best_metric":[0.,0.]},"evaluations":[{"epoch":1,"predictions_ref":"predictions-epoch-001.json","predictions_sha256":"p","result":{"metrics":{k:{"count":n,"recall@50":0.,"ndcg@50":0.} for k,n in (("all",2520),("oov",623),("train_observed",1897))}}}]}
        def read(path,pin):return (manifest if Path(path).name=="manifest.json" else [[1,2]],pin)
        api=SimpleNamespace(ROOT=Path("/synthetic"),read=read,load_archive=lambda *a:{"hits":[0],"ndcg":[0.]})
        state={"manifest_sha256":"m"}
        r.accepted_archive(api,rid,state,[],{}, {1,2},"s")
        manifest["evaluations"][0]["result"]["metrics"]["oov"]["recall@50"]=.1
        with self.assertRaisesRegex(ValueError,"slice metrics"):
            r.accepted_archive(api,rid,state,[],{}, {1,2},"s")
        manifest["binding"]["synthetic"]=True
        with self.assertRaisesRegex(ValueError,"closed test gate"):
            r.accepted_archive(api,rid,state,[],{}, {1,2},"s")
    def test_pinned_bootstrap_uses_original_algorithm(self):
        phase_path=Path(__file__).with_name("phase2_validation_read.py")
        phase=r.pinned_module(phase_path,r.PHASE2_SHA,{})
        got=phase.paired({"hits":[1,1,1],"ndcg":[1.,1.,1.]},{"hits":[0,0,0],"ndcg":[0.,0.,0.]})
        self.assertEqual(got["mcnemar_exact_two_sided"],.25)
        self.assertEqual(got["recall_ci95"],[1.,1.])
        self.assertEqual(got["ndcg_ci95"],[1.,1.])
        self.assertEqual(got["bootstrap"],{"replicates":10000,"seed":0,"unit":"query"})
    def test_nonfinite_metrics_fail_closed(self):
        from types import SimpleNamespace
        manifest={"config":{"seed":42},"binding":{"synthetic":False,"test_gate":"closed_hash_only","support_sha256":"s"},"selection":{"best_epoch":1,"best_metric":[float("nan"),0.]},"evaluations":[{"epoch":1,"predictions_ref":"predictions-epoch-001.json","predictions_sha256":"p","result":{"metrics":{k:{"count":n,"recall@50":0.,"ndcg@50":0.} for k,n in (("all",2520),("oov",623),("train_observed",1897))}}}]}
        def read(path,pin):return (manifest if Path(path).name=="manifest.json" else [[1,2]],pin)
        api=SimpleNamespace(ROOT=Path("/synthetic"),read=read,load_archive=lambda *a:{"hits":[0],"ndcg":[0.]})
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            r.accepted_archive(api,"p-feat04combo-cde-s42",{"manifest_sha256":"m"},[],{}, {1,2},"s")
    def test_read_only_ast(self):
        import ast
        tree=ast.parse(Path(r.__file__).read_text())
        forbidden={"write_text","write_bytes","unlink","mkdir","Popen","run","system","modules","read_status","snapshot"}
        self.assertFalse({n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)} & forbidden)

if __name__=="__main__":unittest.main()
