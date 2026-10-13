import importlib.util
import unittest
import tempfile
from pathlib import Path
import json
from unittest.mock import patch


class DataAuditTests(unittest.TestCase):
    def test_train_audit_counts_feedback_duplicates_and_positive_lengths(self):
        # Removing duplicate accounting or counting exposures as positives must fail.
        self.assertIsNotNone(importlib.util.find_spec("vkgr.data_audit"), "P3 audit behavior missing")
        from vkgr.data_audit import audit_rows
        rows = [
            {"user_id": 1, "item_id": 10, "like": True},
            {"user_id": 1, "item_id": 11, "like": False},
            {"user_id": 1, "item_id": 12, "like": True},
            {"user_id": 2, "item_id": 10, "like": True},
            {"user_id": 2, "item_id": 10, "like": True},
        ]
        result = audit_rows([(0, rows)])
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["users"], 2)
        self.assertEqual(result["items"], 3)
        self.assertEqual(result["feedback"]["like"]["true"], 4)
        self.assertEqual(result["duplicate_user_item"], 1)
        self.assertEqual(result["exact_duplicate_rows"], 1)
        self.assertEqual(result["positive_length_histogram"], {"2": 2})
        self.assertEqual(result["users_ge_2_positive"], 2)

    def test_rejects_invalid_ids_labels_and_out_of_order_weeks(self):
        from vkgr.data_audit import audit_rows
        for partitions in [
            [(1, []), (0, [])], [(0, []), (0, [])],
            [(0, [{"user_id": None, "item_id": 1, "like": True}])],
            [(0, [{"user_id": 1, "item_id": 1, "like": 1}])],
            [(0, [{"user_id": 1, "item_id": -1, "like": True}])],
        ]:
            with self.subTest(partitions=partitions), self.assertRaises(ValueError):
                audit_rows(partitions)

    def test_missing_feedback_remains_missing_not_negative(self):
        from vkgr.data_audit import audit_rows
        result = audit_rows([(0, [{"user_id": 1, "item_id": 1, "like": None}])])
        self.assertEqual(result["feedback"]["like"], {"null": 1})
        self.assertEqual(result["positive_length_histogram"], {"0": 1})

    def test_queries_use_prefix_first_positive_truncate_and_keep_oov(self):
        from vkgr import data_audit
        self.assertTrue(callable(getattr(data_audit, "build_queries", None)), "P3 prefix query behavior missing")
        history = [(0, [{"user_id": 1, "item_id": 10, "like": True},
                        {"user_id": 1, "item_id": 11, "like": True},
                        {"user_id": 1, "item_id": 12, "like": True},
                        {"user_id": 2, "item_id": 10, "like": False}])]
        target = (1, [{"user_id": 1, "item_id": 99, "like": False},
                      {"user_id": 1, "item_id": 20, "like": True},
                      {"user_id": 1, "item_id": 21, "like": True},
                      {"user_id": 2, "item_id": 22, "like": True}])
        result = data_audit.build_queries(history, target, {"min_history": 1, "max_history": 2}, {10, 11, 12})
        self.assertEqual(result["queries"], [{"user_id": 1, "history": [11, 12], "history_positions": [[0, 1], [0, 2]], "target": 20, "target_position": [1, 1], "oov": True}])
        self.assertEqual(result["filtered_no_history"], 1)
        self.assertEqual(result["target_positive_users"], 2)
        with self.assertRaises(ValueError):
            data_audit.build_queries(history, (0, []), {"min_history": 1, "max_history": 2}, {10})
        with self.assertRaises(ValueError):
            data_audit.build_queries(history, target, {"min_history": 0, "max_history": 2}, {10})

    def test_training_examples_never_use_future_positives(self):
        from vkgr import data_audit
        self.assertTrue(callable(getattr(data_audit, "training_examples", None)), "P3 training prefix behavior missing")
        rows = [(0, [{"user_id": 1, "item_id": 8, "like": True},
                     {"user_id": 2, "item_id": 9, "like": True},
                     {"user_id": 1, "item_id": 7, "like": True}]),
                (1, [{"user_id": 1, "item_id": 6, "like": True}])]
        self.assertEqual(data_audit.training_examples(rows, {"min_history": 1, "max_history": 1}), [
            {"user_id": 1, "history": [8], "history_positions": [[0, 0]], "target": 7, "target_position": [0, 2]},
            {"user_id": 1, "history": [7], "history_positions": [[0, 2]], "target": 6, "target_position": [1, 0]}])

    def test_freeze_is_exclusive_and_detects_tamper_before_holdout(self):
        from vkgr import data_audit
        self.assertTrue(callable(getattr(data_audit, "freeze_policy", None)), "P3 freeze behavior missing")
        policy = {"version": "p3-like-v1", "label": "like", "min_history": 1, "max_history": 50,
                  "candidate_catalog": "train_observed", "seen_item_masking": False,
                  "target": "first_positive_in_target_week", "validation_history": "train_only",
                  "test_history": "train_only", "tie_break": "item_id_ascending",
                  "duplicate_policy": "preserve", "train_weeks": list(range(25)),
                  "validation_week": 25, "test_week": 26}
        evidence = {"source_revision": "a" * 40, "inventory_sha256": "b" * 64,
                    "readme_sha256": "c" * 64, "train_files": [{"path": f"subsamples/ur0.01_ir0.01/train/week_{w:02}.parquet", "size_bytes": 1, "sha256": "d" * 64} for w in range(25)],
                    "train_audit_sha256": "e" * 64, "catalog_sha256": "f" * 64,
                    "users_ge_2_positive": 1000,
                    "ordering": "official_assertion_and_physical_row_ordinal"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.json"
            evidence.update({"producer": data_audit.producer_identity(), "training_policy_sha256": data_audit.sha256_bytes(data_audit.canonical_bytes(policy))})
            manifest, sha = data_audit.freeze_policy(path, policy, evidence)
            self.assertEqual(data_audit.validate_frozen_manifest(path, sha)["policy"], policy)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                data_audit.freeze_policy(path, policy, evidence)
            self.assertEqual(path.read_bytes(), before)
            path.write_text(path.read_text().replace('"max_history":50', '"max_history":51'))
            with self.assertRaises(ValueError):
                data_audit.validate_frozen_manifest(path, sha)
            for bad in ([], evidence["train_files"][:-1], [{"path":"subsamples/ur0.01_ir0.01/test/week_26.parquet","size_bytes":1,"sha256":"d"*64}]*25):
                invalid={**evidence,"train_files":bad}
                with self.assertRaises(ValueError):
                    data_audit.freeze_policy(Path(tmp)/"invalid.json",policy,invalid)
            with self.assertRaises(ValueError):
                data_audit.freeze_policy(Path(tmp)/"invalid-policy.json",{**policy,"seen_item_masking":0},evidence)
            evidence["users_ge_2_positive"] = 999
            absent = Path(tmp) / "low_support.json"
            with self.assertRaises(ValueError):
                data_audit.freeze_policy(absent, policy, evidence)
            self.assertFalse(absent.exists())

    def test_parquet_reader_preserves_order_and_rejects_schema_drift(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from vkgr import data_audit
        self.assertTrue(callable(getattr(data_audit, "parquet_rows", None)), "P3 parquet behavior missing")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.parquet"
            fields = {"user_id": pa.array([2, 1], pa.uint32()), "item_id": pa.array([9, 8], pa.uint32())}
            for name in ("place", "platform", "agent", "timespent"):
                fields[name] = pa.array([0, 1], pa.uint8())
            for name in data_audit.FEEDBACK:
                fields[name] = pa.array([False, True], pa.bool_())
            pq.write_table(pa.table(fields), path, row_group_size=1)
            self.assertEqual([r["item_id"] for r in data_audit.parquet_rows(path)], [9, 8])
            fields["like"] = pa.array([0, 1], pa.uint8())
            pq.write_table(pa.table(fields), path)
            with self.assertRaises(ValueError):
                list(data_audit.parquet_rows(path))

    def test_features_match_ids_not_position_and_report_missing(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from vkgr import data_audit
        self.assertTrue(callable(getattr(data_audit, "audit_features", None)), "P3 feature matching behavior missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata").mkdir()
            pq.write_table(pa.table({"item_id": pa.array([2, 1], pa.uint32()), "duration": pa.array([0, 10], pa.uint8())}), root / "metadata/items_metadata.parquet")
            np.savez(root / "metadata/item_embeddings.npz", item_id=np.array([1, 2], dtype=np.uint32), embedding=np.array([[1]*64, [0]*64], dtype=np.float16))
            result = data_audit.audit_features(root, {1, 2, 3})
            self.assertEqual(result["metadata_missing"], 1)
            self.assertEqual(result["embedding_missing"], 1)
            self.assertEqual(result["embedding_zero_norm"], 1)
            self.assertEqual(result["duration_zero"], 1)
            self.assertEqual(result["embedding_nonfinite"], 0)
            np.savez(root / "metadata/item_embeddings.npz", item_id=np.array([1, 1], dtype=np.uint32), embedding=np.ones((2, 64), dtype=np.float16))
            with self.assertRaises(ValueError):
                data_audit.audit_features(root, {1, 2})

    def test_train_never_reads_holdout_and_holdout_requires_valid_freeze(self):
        import hashlib
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        from vkgr import data_audit
        self.assertTrue(callable(getattr(data_audit, "run_train", None)), "P3 train orchestration missing")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); raw = base / "raw"
            sub = raw / "subsamples/ur0.01_ir0.01"
            (sub / "train").mkdir(parents=True); (sub / "validation").mkdir(); (sub / "test").mkdir()
            (raw / "metadata").mkdir()
            files = []
            for week in range(25):
                n = 1000 if week < 2 else 0
                fields = {"user_id": pa.array(list(range(n)), pa.uint32()), "item_id": pa.array([week+1]*n, pa.uint32())}
                for name in ("place", "platform", "agent", "timespent"):
                    fields[name] = pa.array([0]*n, pa.uint8())
                for name in data_audit.FEEDBACK:
                    fields[name] = pa.array([name == "like"]*n, pa.bool_())
                path = sub / f"train/week_{week:02}.parquet"; pq.write_table(pa.table(fields), path)
                files.append({"path": str(path.relative_to(raw)), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            # Invalid holdout bytes would error if train touched them.
            (sub / "validation/week_25.parquet").write_bytes(b"not parquet")
            (sub / "test/week_26.parquet").write_bytes(b"not parquet")
            pq.write_table(pa.table({"item_id": pa.array([1, 2], pa.uint32()), "duration": pa.array([10, 11], pa.uint8())}), raw / "metadata/items_metadata.parquet")
            np.savez(raw / "metadata/item_embeddings.npz", item_id=np.array([1, 2], np.uint32), embedding=np.ones((2,64), np.float16))
            for name in ("metadata/items_metadata.parquet", "metadata/item_embeddings.npz"):
                path = raw / name; files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            inventory = base / "inventory.json"
            inventory.write_text(json.dumps({"revision": "a"*40, "files": files}))
            out = base / "train_out"
            result = data_audit.run_train(raw, inventory, out)
            self.assertEqual(result["audit"]["users_ge_2_positive"], 1000)
            self.assertEqual(result["training_examples"], 1000)
            self.assertEqual(result["catalog_items"], 2)
            self.assertEqual(result["policy"], data_audit.approved_policy())
            self.assertEqual(result["producer"]["dependencies"], {"PyYAML":"6.0.2","numpy":"1.26.4","pyarrow":"19.0.1"})
            readme=base/"README.md"
            readme.write_text("rows within each file are in increasing timestamp order\nmaintain_order='left'\n")
            policy_path=Path(__file__).resolve().parents[1]/"configs/p3.yaml"
            audit_path=out/"train-audit.json"
            valid_audit=audit_path.read_bytes()
            old={k:v for k,v in result.items() if k!="producer"}
            audit_path.write_bytes(data_audit.canonical_bytes(old))
            self.assertEqual(data_audit.main(["freeze","--train-output",str(out),"--readme",str(readme),"--policy",str(policy_path),"--output",str(out/"old-rejected.json")]),2)
            self.assertFalse((out/"old-rejected.json").exists())
            audit_path.write_bytes(valid_audit)
            before = (out / "train-audit.json").read_bytes()
            with self.assertRaises(FileExistsError):
                data_audit.run_train(raw, inventory, out)
            self.assertEqual((out / "train-audit.json").read_bytes(), before)
            with self.assertRaises(ValueError):
                data_audit.run_holdout(raw, inventory, out / "train-audit.json", "0"*64, base / "forbidden")
            self.assertFalse((base / "forbidden").exists())

    def test_valid_frozen_holdout_keeps_first_oov_and_train_only_history(self):
        import hashlib
        import pyarrow as pa
        import pyarrow.parquet as pq
        import yaml
        from vkgr import data_audit
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); raw = base / "raw"; train = base / "train"
            train.mkdir(); sub = raw / "subsamples/ur0.01_ir0.01"
            files = []
            def write(week, split, items):
                path = sub / split / f"week_{week:02}.parquet"; path.parent.mkdir(parents=True, exist_ok=True)
                n=len(items); fields={"user_id": pa.array([1]*n, pa.uint32()), "item_id": pa.array(items, pa.uint32())}
                for name in ("place", "platform", "agent", "timespent"):
                    fields[name]=pa.array([0]*n, pa.uint8())
                for name in data_audit.FEEDBACK:
                    fields[name]=pa.array([name=="like"]*n, pa.bool_())
                pq.write_table(pa.table(fields), path)
                files.append({"path": str(path.relative_to(raw)), "size_bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            for week in range(25):
                write(week, "train", [10, 11] if week==0 else [])
            write(25, "validation", [99, 10]); write(26, "test", [12, 11])
            inventory=base/"inventory.json";inventory.write_text(json.dumps({"revision":"a"*40,"files":files}))
            (train/"catalog.json").write_bytes(data_audit.canonical_bytes([10,11,12]))
            (train/"train-users.json").write_bytes(data_audit.canonical_bytes([1,2]))
            producer=data_audit.producer_identity()
            (train/"train-audit.json").write_bytes(data_audit.canonical_bytes({"train_users_sha256":data_audit.file_sha256(train/"train-users.json"),"producer":producer,"policy_sha256":data_audit.sha256_bytes(data_audit.canonical_bytes(data_audit.approved_policy()))}))
            policy=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/p3.yaml").read_text())
            evidence={"source_revision":"a"*40,"inventory_sha256":data_audit.file_sha256(inventory),
                      "readme_sha256":"c"*64,"train_audit_sha256":data_audit.file_sha256(train/"train-audit.json"),
                      "catalog_sha256":data_audit.file_sha256(train/"catalog.json"),"train_files":files[:25],
                      "users_ge_2_positive":1000,"ordering":"official_assertion_and_physical_row_ordinal"}
            evidence.update({"producer": producer, "training_policy_sha256":data_audit.sha256_bytes(data_audit.canonical_bytes(policy))})
            path=train/"frozen-manifest.json";manifest,sha=data_audit.freeze_policy(path,policy,evidence)
            try:
                result=data_audit.run_holdout(raw,inventory,path,sha,base/"holdout")
            except ValueError:
                result=None
            self.assertIsInstance(result, dict, "valid frozen input must produce holdout audit")
            valid=json.loads((base/"holdout/validation-queries.json").read_bytes())
            test=json.loads((base/"holdout/test-queries.json").read_bytes())
            self.assertEqual(valid[0]["target"],99)
            self.assertTrue(valid[0]["oov"])
            self.assertEqual(test[0]["history"],[10,11])
            self.assertEqual(result["validation"]["eligible_oov_queries"],1)
            self.assertEqual(result["test"]["eligible_oov_queries"],0)
            self.assertEqual(result["validation"]["train_users_without_positive_target"],1)

    def test_cli_errors_are_safe_and_preserve_output(self):
        from vkgr import data_audit
        self.assertTrue(callable(getattr(data_audit, "main", None)), "P3 CLI behavior missing")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/"untouched"
            self.assertEqual(data_audit.main(["train", "--raw-root", str(root/"absent"), "--inventory", str(root/"none"), "--output", str(output)]), 2)
            self.assertFalse(output.exists())

    def test_audit_reports_exposure_lengths_watch_ranges_and_cofeedback(self):
        from vkgr.data_audit import audit_rows
        rows=[{"user_id":1,"item_id":10,"like":True,"dislike":True,"timespent":0},
              {"user_id":1,"item_id":11,"like":False,"dislike":False,"timespent":255},
              {"user_id":2,"item_id":12,"like":True,"dislike":False,"timespent":2}]
        result=audit_rows([(0,rows)])
        self.assertIn("exposure_length_histogram", result, "P3 exposure/watch/cofeedback audit missing")
        self.assertEqual(result["exposure_length_histogram"], {"1":1,"2":1})
        self.assertEqual(result["timespent_histogram"], {"0":1,"2":1,"255":1})
        self.assertEqual(result["like_cofeedback"]["dislike"],1)
        self.assertEqual(result["weeks"][0]["new_observed_items"],3)

    def test_review_policy_limits_version_and_nested_week_types_are_fixed(self):
        import yaml
        from vkgr import data_audit
        policy=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/p3.yaml").read_text())
        evidence={"source_revision":"a"*40,"inventory_sha256":"b"*64,"readme_sha256":"c"*64,
                  "train_audit_sha256":"d"*64,"catalog_sha256":"e"*64,
                  "users_ge_2_positive":1000,"ordering":"official_assertion_and_physical_row_ordinal",
                  "train_files":[{"path":f"subsamples/ur0.01_ir0.01/train/week_{w:02}.parquet","size_bytes":1,"sha256":"f"*64} for w in range(25)]}
        for changed in ({"min_history":2},{"max_history":51},{"version":"p3-other-v1"},
                        {"train_weeks":[False,True]+list(range(2,25))}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                data_audit._validate_policy_evidence({**policy,**changed},evidence)

    def test_review_train_records_real_producer_and_rejects_old_producer(self):
        from vkgr import data_audit
        self.assertTrue(callable(getattr(data_audit,"producer_identity",None)), "producer identity binding missing")
        producer=data_audit.producer_identity()
        self.assertEqual(producer["dependencies"], {"PyYAML":"6.0.2","numpy":"1.26.4","pyarrow":"19.0.1"})
        self.assertEqual(producer["python_major_minor"],[3,11])
        data_audit.validate_producer({**producer,"python_version":"3.11.16"})
        self.assertRegex(producer["source_fingerprint"],r"^[a-f0-9]{64}$")
        self.assertRegex(producer["git_commit"],r"^[a-f0-9]{40}$")
        for changed in ({"source_fingerprint":"0"*64},{"lock_sha256":"0"*64},
                        {"dependencies":{**producer["dependencies"],"numpy":"1.0"}},
                        {"python_major_minor":[3,12]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                data_audit.validate_producer({**producer,**changed})
        with self.assertRaises(ValueError):
            data_audit.validate_producer({})

    def test_review_holdout_rejects_source_dependency_and_bool_before_any_reader(self):
        from vkgr import data_audit
        policy=data_audit.approved_policy();producer=data_audit.producer_identity()
        evidence={"source_revision":"a"*40,"inventory_sha256":"b"*64,"readme_sha256":"c"*64,
                  "train_audit_sha256":"d"*64,"catalog_sha256":"e"*64,
                  "users_ge_2_positive":1000,"ordering":"official_assertion_and_physical_row_ordinal",
                  "train_files":[{"path":f"subsamples/ur0.01_ir0.01/train/week_{w:02}.parquet","size_bytes":1,"sha256":"f"*64} for w in range(25)],
                  "producer":producer,"training_policy_sha256":data_audit.sha256_bytes(data_audit.canonical_bytes(policy))}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for index,changed in enumerate(({"producer":{}}, {"producer":{**producer,"source_fingerprint":"0"*64}},
                    {"producer":{**producer,"lock_sha256":"0"*64}},
                    {"producer":{**producer,"dependencies":{**producer["dependencies"],"numpy":"1.0"}}},
                    {"bad_policy":{**policy,"train_weeks":[False,True]+list(range(2,25))}})):
                selected_policy=changed.get("bad_policy",policy)
                selected_evidence={**evidence,**{k:v for k,v in changed.items() if k!="bad_policy"}}
                manifest={"schema_version":1,"task_id":"P3","policy":selected_policy,
                          "policy_sha256":data_audit.sha256_bytes(data_audit.canonical_bytes(selected_policy)),"train_evidence":selected_evidence}
                path=root/f"bad-{index}.json";path.write_bytes(data_audit.canonical_bytes(manifest))
                with self.subTest(changed=changed), patch("vkgr.data_audit.parquet_rows",side_effect=AssertionError("reader called before provenance gate")), self.assertRaises(ValueError):
                    data_audit.run_holdout(root/"missing-raw",root/"missing-inventory",path,data_audit.file_sha256(path),root/"forbidden")
                self.assertFalse((root/"forbidden").exists())


if __name__ == "__main__":
    unittest.main()
