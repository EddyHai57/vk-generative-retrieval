import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from vkgr.p_watch_positive_run import (default_config, validate_config, validate_training_approval,
                                       validate_stats_report, validate_stats_data_binding,
                                       bind_auxiliary_pool)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class WatchPositiveRunGateTests(unittest.TestCase):
    def test_module_entry_exposes_train_verify_and_readonly_preflight_help_and_rejects_bad_command(self):
        import subprocess
        repo = Path(__file__).resolve().parents[1]
        env = {**__import__("os").environ, "PYTHONPATH": str(repo / "src") + __import__("os").pathsep + str(repo / "scripts"),
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "KMP_USE_SHM": "0", "KMP_INIT_AT_FORK": "FALSE", "MKL_THREADING_LAYER": "SEQUENTIAL"}
        for command in ("train", "verify", "preflight"):
            result = subprocess.run([sys.executable, "-B", "-m", "vkgr.p_watch_positive_run", command, "--help"],
                                    cwd=repo, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.strip())
        bad = subprocess.run([sys.executable, "-B", "-m", "vkgr.p_watch_positive_run", "bogus"],
                             cwd=repo, env=env, text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)

    def test_module_preflight_cli_reaches_formal_preflight_gate(self):
        import subprocess
        repo = Path(__file__).resolve().parents[1]
        env = {**__import__("os").environ, "PYTHONPATH": str(repo / "src") + __import__("os").pathsep + str(repo / "scripts"),
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "KMP_USE_SHM": "0", "KMP_INIT_AT_FORK": "FALSE", "MKL_THREADING_LAYER": "SEQUENTIAL"}
        with tempfile.TemporaryDirectory() as temp:
            request = Path(temp) / "fake-preflight-request.json"
            request.write_text(json.dumps({"stage": "preflight"}))
            request_sha = sha(request.read_bytes())
            result = subprocess.run([sys.executable, "-B", "-m", "vkgr.p_watch_positive_run", "preflight",
                                     "--request", str(request), "--request-sha", request_sha],
                                    cwd=repo, env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("closed P engineering stage request required", result.stderr)
        self.assertNotIn("unrecognized arguments", result.stderr)

    def test_config_preserves_m1_mean_contract_for_only_registered_seeds(self):
        base = default_config()
        self.assertEqual((base["arm"], base["temperature"], base["learning_rate"], base["batch_size"],
                          base["logq_alpha"], base["max_epochs"], base["min_epochs"], base["patience"]),
                         ("mean", .1, .001, 512, 1., 100, 5, 3))
        for seed in (42, 2024, 2025):
            config = {**base, "seed": seed}
            self.assertIs(validate_config(config, synthetic=True), config)
        with self.assertRaises(ValueError):
            validate_config({**base, "arm": "last"}, synthetic=True)
        repo = Path(__file__).resolve().parents[1]
        for seed in (42, 2024, 2025):
            path = repo / ("configs/p_watch_positive/p-watch-positive-s%d.json" % seed)
            value = json.loads(path.read_text())
            self.assertEqual(value["seed"], seed)
            validate_config(value, synthetic=True)

    def approval(self, evidence_path, evidence_sha, stats_sha):
        return {
            "schema": "p-watch-positive-training-approval-v1",
            "status": "approved",
            "approved_by": "Kiro",
            "stats_run_id": "p-watch-positive-stats-v1",
            "stats_stdout_sha256": stats_sha,
            "authorized_run_ids": ["p-watch-positive-s42", "p-watch-positive-s2024", "p-watch-positive-s2025"],
            "seeds": [42, 2024, 2025],
            "decision_evidence": {"path": str(evidence_path), "sha256": evidence_sha},
            "card_sha256": "c" * 64,
        }

    def test_training_approval_binds_stats_run_seed_and_decision_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "decision-receipt.json"; evidence_bytes = b"approved decision receipt"
            evidence.write_bytes(evidence_bytes)
            stats_sha = "a" * 64
            approval = self.approval(evidence, sha(evidence_bytes), stats_sha)
            path = Path(temp) / "approval.json"; raw = json.dumps(approval, sort_keys=True, separators=(",", ":")).encode(); path.write_bytes(raw)
            result = validate_training_approval(path, sha(raw), "p-watch-positive-s2024", 2024, stats_sha)
            self.assertEqual(result["approval_sha256"], sha(raw))
            self.assertEqual(result["decision_evidence_sha256"], sha(evidence_bytes))
            with self.assertRaises(ValueError):
                validate_training_approval(path, sha(raw), "p-watch-positive-s42", 2024, stats_sha)
            with self.assertRaises(ValueError):
                validate_training_approval(path, sha(raw), "p-watch-positive-s2024", 2024, "b" * 64)

    def test_stats_report_requires_verified_formal_preflight_with_no_stop_reasons(self):
        report = {
            "schema": "p-watch-positive-train-only-report-v1", "run_id": "p-watch-positive-stats-v1",
            "engineering": False, "synthetic": False, "N_like_train_examples": 5, "N_aux": 2,
            "empty_history_aux_count": 0, "empty_history_compatibility_note": "count only", "stop_reasons": [],
            "gate": {"week_ge_25_in_auxiliary": False, "N_aux_positive": True,
                     "invalid_ratio_fraction_at_or_below_1_percent": True, "empty_history_aux_count": 0},
            "N_like": 5, "N_aux_candidates": 2,
            "inputs": {"stats_request_sha256": "d" * 64, "train_weeks": "00-24", "quantile_method": "nearest_rank",
                       "supported_items_sha256": "b080240542ae85166186fb621f7b074f9728f8b0e7d01d2c69ae0d0bff6b53a2"},
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stats.stdout"
            raw = json.dumps(report, sort_keys=True, separators=(",", ":")).encode(); path.write_bytes(raw)
            validate_stats_report(path, sha(raw))
            bad_synthetic = Path(temp) / "synthetic.stdout"
            bad = json.dumps({**report, "synthetic": True}, sort_keys=True, separators=(",", ":")).encode(); bad_synthetic.write_bytes(bad)
            with self.assertRaises(ValueError):
                validate_stats_report(bad_synthetic, sha(bad))
            bad_stop = Path(temp) / "stop.stdout"
            bad = json.dumps({**report, "stop_reasons": ["N_aux=0"]}, sort_keys=True, separators=(",", ":")).encode(); bad_stop.write_bytes(bad)
            with self.assertRaises(ValueError):
                validate_stats_report(bad_stop, sha(bad))
            bad_method = Path(temp) / "linear.stdout"
            bad = json.dumps({**report, "inputs": {**report["inputs"], "quantile_method": "linear"}}, sort_keys=True, separators=(",", ":")).encode(); bad_method.write_bytes(bad)
            with self.assertRaises(ValueError):
                validate_stats_report(bad_method, sha(bad))

    def test_stats_binding_matches_the_exact_loaded_train_support_and_sources(self):
        from vkgr.retrieval_b1_run import _digest
        from vkgr.p_watch_positive_run import Path as RunnerPath
        support = {2, 7}
        names = ("src/vkgr/p_watch_positive.py", "scripts/p_watch_positive_stats.py", "src/vkgr/retrieval_data.py",
                 "scripts/p3_label_eda.py", "configs/p_watch_positive/p-watch-positive-train-only.json")
        repo = RunnerPath(__file__).resolve().parents[1]
        source_pins = {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in names}
        report = {"inputs": {"p_inputs_metadata_sha256": "a" * 64,
                             "train_audit_sha256": "3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe",
                             "items_metadata_sha256": "b" * 64,
                             "supported_items_sha256": _digest(sorted(support)),
                             "anchor_manifest_sha256": "bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e",
                             "source_pins": source_pins,
                             "train_partition_sha256": ["d" * 64, "e" * 64]}}
        base_binding = {"raw_records": [{"sha256": "d" * 64}, {"sha256": "e" * 64}]}
        result = validate_stats_data_binding(report, "f" * 64, "a" * 64, "b" * 64, support, base_binding)
        self.assertEqual(result["stats_stdout_sha256"], "f" * 64)
        self.assertEqual(result["supported_items_sha256"], _digest(sorted(support)))
        changed = {**report, "inputs": {**report["inputs"], "train_partition_sha256": ["0" * 64, "e" * 64]}}
        with self.assertRaises(ValueError):
            validate_stats_data_binding(changed, "f" * 64, "a" * 64, "b" * 64, support, base_binding)
        altered_pins = {**source_pins, names[0]: "0" * 64}
        changed = {**report, "inputs": {**report["inputs"], "source_pins": altered_pins}}
        with self.assertRaises(ValueError):
            validate_stats_data_binding(changed, "f" * 64, "a" * 64, "b" * 64, support, base_binding)

    def test_auxiliary_pool_binding_never_filters_empty_history_rows(self):
        row = {"example_id": "watch:0:1", "user_id": 2, "target": 9, "target_position": [0, 1],
               "history": [8], "history_positions": [[0, 0]], "past_likes": [8], "side": [[1, 10, 10]]}
        report = {"N_aux": 1, "empty_history_aux_count": 0,
                  "buckets": {"1-10": {"supported_aux": 1}}}
        bound = bind_auxiliary_pool([row], report, {9}, {9: 10})
        self.assertEqual(bound["N_aux"], 1)
        self.assertEqual(len(bound["auxiliary_pool_sha256"]), 64)
        empty = {**row, "history": [], "history_positions": [], "side": []}
        report_empty = {"N_aux": 1, "empty_history_aux_count": 1,
                        "buckets": {"1-10": {"supported_aux": 1}}}
        with self.assertRaisesRegex(ValueError, "filtered auxiliary pool contains empty history"):
            bind_auxiliary_pool([empty], report_empty, {9}, {9: 10})

    def test_formal_auxiliary_filter_is_exact_and_binds_exclusion_diagnostics(self):
        from vkgr.p_watch_positive_nonempty import filter_auxiliary_pool
        rows = [
            {"example_id": "watch:0:0", "user_id": 2, "target": 9, "target_position": [0, 0],
             "history": [], "history_positions": [], "past_likes": [], "side": []},
            {"example_id": "watch:0:1", "user_id": 2, "target": 9, "target_position": [0, 1],
             "history": [8], "history_positions": [[0, 0]], "past_likes": [8], "side": [[1, 10, 10]]},
            {"example_id": "watch:0:2", "user_id": 3, "target": 10, "target_position": [0, 2],
             "history": [], "history_positions": [], "past_likes": [], "side": []},
        ]
        report = {"N_aux": 3, "empty_history_aux_count": 2,
                  "buckets": {"1-10": {"supported_aux": 3}}}
        filtered, diag = filter_auxiliary_pool(rows, report, {9, 10}, {9: 10, 10: 5},
                                                {9: "low", 10: "high"}, formal=False)
        self.assertEqual([row["example_id"] for row in filtered], ["watch:0:1"])
        self.assertEqual(diag["raw_N_aux"], 3)
        self.assertEqual(diag["empty_history_excluded"], 2)
        self.assertEqual(diag["N_aux"], 1)
        self.assertEqual(diag["excluded_by_duration_bucket"], {"1-10": 2, "11-20": 0, "21-40": 0,
                                                                "41-60": 0, "61-120": 0, "121-255": 0})
        self.assertEqual(diag["excluded_by_heat_bin"], {"unsupported0": 0, "high": 1, "mid": 0, "low": 1})
        self.assertEqual(diag["fully_excluded_item_count"], 1)
        self.assertEqual(diag["raw_bucket_counts_verified"], True)
        self.assertEqual(len(diag["raw_pool_sha256"]), 64)
        self.assertEqual(len(diag["filtered_pool_sha256"]), 64)

    def test_formal_auxiliary_filter_requires_exact_frozen_counts(self):
        from vkgr.p_watch_positive_nonempty import (filter_auxiliary_pool, RAW_AUXILIARY_COUNT,
                                                    EMPTY_HISTORY_COUNT, FILTERED_AUXILIARY_COUNT)
        self.assertEqual((RAW_AUXILIARY_COUNT, EMPTY_HISTORY_COUNT, FILTERED_AUXILIARY_COUNT),
                         (716194, 15561, 700633))
        report = {"N_aux": 2, "empty_history_aux_count": 1,
                  "buckets": {"1-10": {"supported_aux": 2}, "11-20": {"supported_aux": 0},
                              "21-40": {"supported_aux": 0}, "41-60": {"supported_aux": 0},
                              "61-120": {"supported_aux": 0}, "121-255": {"supported_aux": 0}}}
        rows = [
            {"example_id": "watch:0:0", "user_id": 2, "target": 9, "target_position": [0, 0],
             "history": [], "history_positions": [], "past_likes": [], "side": []},
            {"example_id": "watch:0:1", "user_id": 2, "target": 9, "target_position": [0, 1],
             "history": [8], "history_positions": [[0, 0]], "past_likes": [8], "side": [[1, 10, 10]]},
        ]
        with self.assertRaisesRegex(ValueError, "formal post-filter auxiliary count"):
            filter_auxiliary_pool(rows, report, {9}, {9: 10}, {9: "low"}, formal=True)

    def test_real_context_with_positive_empty_history_reaches_filter(self):
        from vkgr import p_watch_positive_run as runner
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); raw_root = root / "raw"; raw_root.mkdir()
            metadata = raw_root / "metadata" / "items_metadata.parquet"
            metadata.parent.mkdir(); metadata.write_bytes(b"synthetic metadata")
            history_dir = root / "history"; history_dir.mkdir()
            slices_path = history_dir / "train-slices.json"
            slices_path.write_text(json.dumps({"popularity_thresholds": {"median": 1, "p90": 11},
                                               "item_like_counts": {"9": 1}}))
            report = {"empty_history_aux_count": 15561, "buckets": {"1-10": {"threshold": 1.0}}}
            spec = {"raw_root": str(raw_root), "disk_cap_bytes": 1000,
                    "items_metadata_sha256": sha(metadata.read_bytes()),
                    "history_manifest": str(history_dir / "manifest.json"), "history_sha256": "a" * 64}
            data = {"support": {9}, "catalog": [9]}
            args = SimpleNamespace(stats_report="stats", stats_sha="b" * 64,
                                    approval="approval", approval_sha="c" * 64,
                                    inputs="inputs", inputs_sha="d" * 64, command="train", output=str(root / "out"))
            reached = RuntimeError("filter_reached")
            with (mock.patch.object(runner, "validate_stats_report", return_value=report),
                  mock.patch.object(runner, "validate_training_approval", return_value={"approval_sha256": "e" * 64}),
                  mock.patch.object(runner.feat02, "validate_inputs", return_value=spec),
                  mock.patch("torch.cuda.is_available", return_value=True),
                  mock.patch.object(runner.b1, "resource_guard", return_value=lambda: None),
                  mock.patch.object(runner.b1, "real_inputs", return_value=(data, lambda: iter(()), {})),
                  mock.patch.object(runner.feat02, "apply_feat02", return_value=(data, {})),
                  mock.patch.object(runner, "validate_stats_data_binding", return_value={}),
                  mock.patch("vkgr.history_side.load_durations", return_value={9: 10}),
                  mock.patch("vkgr.p_watch_positive_rows.build_auxiliary_rows", return_value=[]),
                  mock.patch.object(runner, "read_pinned", return_value={"artifacts": {"train-slices.json": "6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26"}}),
                  mock.patch.object(runner, "file_sha256", side_effect=lambda path: ("6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26"
                                                                                       if Path(path) == slices_path
                                                                                       else sha(Path(path).read_bytes()))),
                  mock.patch("vkgr.p_watch_positive_nonempty.filter_auxiliary_pool", side_effect=reached) as filter_mock):
                with self.assertRaisesRegex(RuntimeError, "filter_reached"):
                    runner._real_context(args, {"seed": 42, "history_len": 50})
            filter_mock.assert_called_once()

    def test_synthetic_run_records_fixed_epoch_mix_and_logq_binding(self):
        from vkgr.retrieval_feat02_run import synthetic_data
        from vkgr.p_watch_positive_run import run_session, verify_run
        with tempfile.TemporaryDirectory() as temp:
            data = synthetic_data()
            data["durations_by_item"] = {item: 10 for item in data["catalog"]}
            data["rows"] = [{**data["rows"][index % len(data["rows"])], "example_id": index, "past_likes": []}
                            for index in range(1023)]
            target = min(data["support"])
            base = data["rows"][0]
            aux_row = {**base, "example_id": "watch:24:0", "target": target,
                       "user_id": 99999, "past_likes": [],
                       "target_position": [base["history_positions"][-1][0], base["history_positions"][-1][1] + 1],
                       "history": base["history"][-50:], "history_positions": base["history_positions"][-50:],
                       "side": base["side"][-50:]}
            aux_row.pop("ctx", None)
            auxiliary = [aux_row]
            empty_aux_row = {**aux_row, "example_id": "watch:24:1", "target": target + 1,
                             "target_position": [aux_row["target_position"][0], aux_row["target_position"][1] + 1],
                             "history": [], "history_positions": [], "side": []}
            auxiliary.append(empty_aux_row)
            report = {"N_like": len(data["rows"]), "N_like_train_examples": len(data["rows"]), "N_aux": 2,
                      "empty_history_aux_count": 1, "buckets": {"1-10": {"supported_aux": 2}}}
            from vkgr.p_watch_positive_nonempty import filter_auxiliary_pool
            auxiliary, data["watch_aux_exclusion"] = filter_auxiliary_pool(
                auxiliary, report, data["support"], data["durations_by_item"],
                {target: "low", target + 1: "low"}, formal=False)
            config = {**default_config(), "batch_size": 64, "max_epochs": 2, "min_epochs": 0}
            result = run_session(Path(temp) / "run", data, config, auxiliary, report, "a" * 64)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["run_id"], "p-watch-positive-s42")
            self.assertEqual([row["auxiliary_drawn"] for row in result["watch_positive"]["epoch_draws"]], [1, 1])
            self.assertEqual(len(result["watch_positive"]["epoch_frequency_sha256"]), 64)
            self.assertEqual(result["watch_positive"]["watch_aux_exclusion"]["N_aux"], 1)
            self.assertEqual(result["watch_positive"]["watch_aux_exclusion"]["empty_history_excluded"], 1)
            verified = verify_run(Path(temp) / "run/manifest.json", result["manifest_sha256"], data, config,
                                  auxiliary, report, "a" * 64)
            self.assertEqual(verified["status"], "verified")
            self.assertEqual(verified["run_id"], "p-watch-positive-s42")
            manifest_path = Path(temp) / "run/manifest.json"
            value = json.loads(manifest_path.read_text())
            value["watch_positive"]["epoch_draws"][0]["tokens_sha256"] = "0" * 64
            tampered = Path(temp) / "run/tampered.json"
            tampered_bytes = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            tampered.write_bytes(tampered_bytes)
            with self.assertRaisesRegex(ValueError, "epoch mix replay"):
                verify_run(tampered, sha(tampered_bytes), data, config, auxiliary, report, "a" * 64)


if __name__ == "__main__":
    unittest.main()
