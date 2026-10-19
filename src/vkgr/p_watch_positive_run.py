"""P watch-positive runner gates and the M1-mean training/verify integration."""
import hashlib
import json
from pathlib import Path

from . import retrieval_seqm1_run as m1
from . import retrieval_feat02_run as feat02
from . import retrieval_b1_run as b1
from .data_audit import file_sha256
from .retrieval_ablation_run import read_pinned, strict_json
from .retrieval_ablation_run import state_equal
from .retrieval_training import SamplerStats
from .p_watch_positive_epoch import epoch_rows, epoch_tokens, frequency_table

TASK = "P-WATCH-POSITIVE"
SCHEMA = "p-watch-positive-run-v1"
CHECKPOINT_SCHEMA = "p-watch-positive-checkpoint-v1"
MANIFEST_SCHEMA = "p-watch-positive-train-manifest-v1"
RUN_IDS = ("p-watch-positive-s42", "p-watch-positive-s2024", "p-watch-positive-s2025")
SEEDS = (42, 2024, 2025)
APPROVAL_SCHEMA = "p-watch-positive-training-approval-v1"
TRAIN_AUDIT_SHA256 = "3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe"
ANCHOR_MANIFEST_SHA256 = "bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e"
_BASE_CONFIG = m1.default_config()
MANIFEST_FIELDS = m1.MANIFEST_FIELDS | {"watch_positive", "run_id"}


def default_config():
    return {**{key: value for key, value in _BASE_CONFIG.items() if key not in ("schema", "task_id")},
            "schema": SCHEMA, "task_id": TASK}


def run_id_for_seed(seed):
    mapping = dict(zip(SEEDS, RUN_IDS))
    if type(seed) is not int or seed not in mapping:
        raise ValueError("unregistered P watch-positive seed")
    return mapping[seed]


def validate_config(value, synthetic=False):
    if type(value) is not dict or set(value) != set(default_config()):
        raise ValueError("unregistered P watch-positive config fields")
    if value["schema"] != SCHEMA or value["task_id"] != TASK or value["arm"] != "mean":
        raise ValueError("P watch-positive is the frozen M1-mean arm")
    projected = {**{key: field for key, field in value.items() if key not in ("schema", "task_id")},
                 "schema": m1.SCHEMA, "task_id": m1.TASK}
    m1.validate_config(projected, synthetic)
    return value


def validate_training_approval(path, expected_sha256, run_id, seed, stats_stdout_sha256):
    path = Path(path)
    if (type(expected_sha256) is not str or len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256)
            or path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha256):
        raise ValueError("training approval file hash mismatch")
    if type(stats_stdout_sha256) is not str or len(stats_stdout_sha256) != 64 or any(c not in "0123456789abcdef" for c in stats_stdout_sha256):
        raise ValueError("stats stdout SHA256 required")
    approval = strict_json(path.read_text())
    expected_fields = {"schema", "status", "approved_by", "stats_run_id", "stats_stdout_sha256",
                       "authorized_run_ids", "seeds", "decision_evidence", "card_sha256"}
    if type(approval) is not dict or set(approval) != expected_fields:
        raise ValueError("training approval schema mismatch")
    if approval["schema"] != APPROVAL_SCHEMA or approval["status"] != "approved" or approval["approved_by"] != "Kiro":
        raise ValueError("Kiro training approval required")
    if approval["stats_run_id"] != "p-watch-positive-stats-v1" or approval["stats_stdout_sha256"] != stats_stdout_sha256:
        raise ValueError("approval is not bound to this stats report")
    if approval["authorized_run_ids"] != list(RUN_IDS) or approval["seeds"] != list(SEEDS):
        raise ValueError("approval run/seed scope differs from the frozen three-seed set")
    expected_run = dict(zip(SEEDS, RUN_IDS))
    if type(seed) is not int or expected_run.get(seed) != run_id:
        raise ValueError("run id and seed binding mismatch")
    evidence = approval["decision_evidence"]
    if type(evidence) is not dict or set(evidence) != {"path", "sha256"}:
        raise ValueError("decision evidence pin required")
    evidence_path = Path(evidence["path"])
    if (type(evidence["path"]) is not str or not evidence_path.is_absolute() or ".." in evidence_path.parts
            or evidence_path.is_symlink() or not evidence_path.is_file() or file_sha256(evidence_path) != evidence["sha256"]):
        raise ValueError("decision evidence file hash mismatch")
    if type(approval["card_sha256"]) is not str or len(approval["card_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in approval["card_sha256"]):
        raise ValueError("approved card digest required")
    return {"approval_sha256": expected_sha256, "decision_evidence_sha256": evidence["sha256"],
            "stats_stdout_sha256": stats_stdout_sha256, "run_id": run_id, "seed": seed,
            "card_sha256": approval["card_sha256"]}


def validate_stats_report(path, expected_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("preflight stats stdout hash mismatch")
    value = strict_json(path.read_text())
    if (value.get("schema") != "p-watch-positive-train-only-report-v1" or value.get("run_id") != "p-watch-positive-stats-v1"
            or value.get("engineering") is not False or value.get("synthetic") is not False):
        raise ValueError("formal train-only stats report required")
    gate = value.get("gate")
    if (type(gate) is not dict or gate.get("week_ge_25_in_auxiliary") is not False
            or gate.get("N_aux_positive") is not True or gate.get("invalid_ratio_fraction_at_or_below_1_percent") is not True
            or value.get("stop_reasons") != []):
        raise ValueError("stats report contains a frozen STOP condition")
    if type(value.get("N_like_train_examples")) is not int or value["N_like_train_examples"] < 1 or type(value.get("N_aux")) is not int or value["N_aux"] < 1:
        raise ValueError("stats report positive pool counts invalid")
    if type(value.get("empty_history_aux_count")) is not int or value["empty_history_aux_count"] < 0 or type(value.get("empty_history_compatibility_note")) is not str:
        raise ValueError("stats report empty-history compatibility count missing")
    if gate.get("empty_history_aux_count") != value["empty_history_aux_count"]:
        raise ValueError("stats report empty-history count mismatch")
    inputs = value.get("inputs")
    if (type(inputs) is not dict or inputs.get("train_weeks") != "00-24"
            or inputs.get("quantile_method") != "nearest_rank"
            or inputs.get("supported_items_sha256") != "b080240542ae85166186fb621f7b074f9728f8b0e7d01d2c69ae0d0bff6b53a2"):
        raise ValueError("stats report source scope/nearest-rank/support identity mismatch")
    if value.get("N_like") != value.get("N_like_train_examples") or value["N_aux"] > value.get("N_aux_candidates", -1):
        raise ValueError("stats report pool denominator mismatch")
    return value


def validate_stats_data_binding(report, stats_stdout_sha256, inputs_sha256, items_metadata_sha256,
                                support, raw_inputs_binding):
    inputs = report.get("inputs")
    raw_records = raw_inputs_binding.get("raw_records")
    if type(inputs) is not dict or type(raw_records) is not list or not raw_records:
        raise ValueError("stats/train source bindings unavailable")
    from .retrieval_b1_run import _digest
    expected_partitions = [record["sha256"] for record in raw_records]
    source_pins = inputs.get("source_pins")
    expected_source_names = {"src/vkgr/p_watch_positive.py", "scripts/p_watch_positive_stats.py",
                             "src/vkgr/retrieval_data.py", "scripts/p3_label_eda.py",
                             "configs/p_watch_positive/p-watch-positive-train-only.json"}
    if type(source_pins) is not dict or set(source_pins) != expected_source_names:
        raise ValueError("stats report source pin set mismatch")
    repo = Path(__file__).resolve().parents[2]
    current_source_pins = {name: file_sha256(repo / name) for name in sorted(expected_source_names)}
    if (inputs.get("p_inputs_metadata_sha256") != inputs_sha256
            or inputs.get("train_audit_sha256") != TRAIN_AUDIT_SHA256
            or inputs.get("items_metadata_sha256") != items_metadata_sha256
            or inputs.get("anchor_manifest_sha256") != ANCHOR_MANIFEST_SHA256
            or inputs.get("supported_items_sha256") != _digest(sorted(support))
            or source_pins != current_source_pins
            or inputs.get("train_partition_sha256") != expected_partitions):
        raise ValueError("stats report differs from loaded P source/support identity")
    return {"stats_stdout_sha256": stats_stdout_sha256, "p_inputs_metadata_sha256": inputs_sha256,
            "train_audit_sha256": TRAIN_AUDIT_SHA256, "items_metadata_sha256": items_metadata_sha256,
            "anchor_manifest_sha256": ANCHOR_MANIFEST_SHA256,
            "supported_items_sha256": inputs["supported_items_sha256"],
            "train_partition_sha256": expected_partitions}


def bind_auxiliary_pool(rows, report, supported_items, duration_by_item, exclusion_diagnostics=None, *, formal=False):
    rows = list(rows)
    if any(not row.get("history") for row in rows):
        raise ValueError("filtered auxiliary pool contains empty history")
    if type(exclusion_diagnostics) is not dict:
        if formal:
            raise ValueError("formal watch auxiliary exclusion diagnostics required")
        exclusion_diagnostics = {"raw_N_aux": len(rows), "empty_history_excluded": 0,
                                 "N_aux": len(rows), "raw_pool_sha256": b1._digest(rows),
                                 "filtered_pool_sha256": b1._digest(rows),
                                 "excluded_by_duration_bucket": {}, "excluded_by_heat_bin": {},
                                 "fully_excluded_item_count": 0, "raw_bucket_counts_verified": False}
    if len(rows) != exclusion_diagnostics.get("N_aux"):
        raise ValueError("filtered auxiliary pool differs from exclusion diagnostics")
    if (exclusion_diagnostics.get("raw_N_aux") != report.get("N_aux")
            or exclusion_diagnostics.get("empty_history_excluded") != report.get("empty_history_aux_count")
            or exclusion_diagnostics.get("N_aux") != exclusion_diagnostics.get("raw_N_aux") - exclusion_diagnostics.get("empty_history_excluded")):
        raise ValueError("watch auxiliary raw/post count binding mismatch")
    for key in ("raw_pool_sha256", "filtered_pool_sha256"):
        digest = exclusion_diagnostics.get(key)
        if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("watch auxiliary pool digest missing")
    if exclusion_diagnostics.get("filtered_pool_sha256") != b1._digest(rows):
        raise ValueError("filtered auxiliary pool digest mismatch")
    if (sum(exclusion_diagnostics.get("excluded_by_duration_bucket", {}).values())
            != exclusion_diagnostics.get("empty_history_excluded")
            or sum(exclusion_diagnostics.get("excluded_by_heat_bin", {}).values())
            != exclusion_diagnostics.get("empty_history_excluded")):
        raise ValueError("watch auxiliary exclusion distribution mismatch")
    if formal and (exclusion_diagnostics["raw_N_aux"] != 716194
                   or exclusion_diagnostics["empty_history_excluded"] != 15561
                   or exclusion_diagnostics["N_aux"] != 700633
                   or exclusion_diagnostics.get("raw_bucket_counts_verified") is not True
                   or exclusion_diagnostics.get("s0_train_slices_sha256") != "6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26"):
        raise ValueError("formal post-filter auxiliary count differs from frozen decision")
    bucket_counts = {}; positions = set()
    from .p_watch_positive import duration_bucket
    for row in rows:
        if set(row) != {"example_id", "user_id", "target", "target_position", "history", "history_positions", "past_likes", "side"}:
            raise ValueError("auxiliary training row schema mismatch")
        position = tuple(row["target_position"])
        if position in positions or not 0 <= position[0] <= 24:
            raise ValueError("duplicate/out-of-train auxiliary target position")
        positions.add(position)
        if row["target"] not in supported_items or len(row["history"]) != len(row["history_positions"]) or len(row["history"]) != len(row["side"]):
            raise ValueError("auxiliary target/support/history/side mismatch")
        if any(tuple(p) >= position for p in row["history_positions"]):
            raise ValueError("auxiliary history is not strict-past")
        bucket = duration_bucket(duration_by_item.get(row["target"]))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    excluded_buckets = exclusion_diagnostics.get("excluded_by_duration_bucket", {})
    for bucket, counts in report["buckets"].items():
        if counts["supported_aux"] != bucket_counts.get(bucket, 0) + excluded_buckets.get(bucket, 0):
            raise ValueError("raw auxiliary duration-bucket count differs from preflight")
    return {**exclusion_diagnostics, "auxiliary_pool_sha256": b1._digest(rows), "bucket_counts": bucket_counts}


def binding_for_watch(data, config, rows, queries, model, device, inputs_binding, watch_binding):
    base = m1.binding_for4(data, config, rows, queries, model, device, inputs_binding)
    return {**base, "task": TASK, "arm": "mean", "watch_positive": watch_binding}


def run_session(output, data, config, auxiliary_rows, stats_report, stats_stdout_sha256,
                approval_binding=None, device="cpu", check=None, inputs_binding=None):
    import random
    import time
    import torch
    from .retrieval_hp import EarlyStop

    synthetic = bool(data.get("synthetic"))
    validate_config(config, synthetic)
    if not synthetic and (device != "cuda" or inputs_binding is None or approval_binding is None):
        raise ValueError("real P watch training requires CUDA, source binding and explicit Kiro approval")
    b1.execution_context(device)
    rows, queries = feat02._view(data, config["history_len"])
    auxiliary_rows = list(auxiliary_rows)
    pool_binding = bind_auxiliary_pool(auxiliary_rows, stats_report, data["support"], data.get("durations_by_item", {}),
                                       data.get("watch_aux_exclusion"), formal=not synthetic)
    if len(rows) != stats_report.get("N_like_train_examples"):
        raise ValueError("loaded M1-mean training rows differ from frozen N_like")
    if type(stats_stdout_sha256) is not str or len(stats_stdout_sha256) != 64:
        raise ValueError("stats stdout SHA256 required")
    if approval_binding is not None and approval_binding.get("stats_stdout_sha256") != stats_stdout_sha256:
        raise ValueError("approval/stats binding mismatch")
    frequencies = frequency_table(rows, auxiliary_rows)
    watch = {**pool_binding, "watch_aux_exclusion": data.get("watch_aux_exclusion"),
             "N_like": len(rows), "stats_stdout_sha256": stats_stdout_sha256,
             "approval_sha256": (approval_binding or {}).get("approval_sha256"),
             "decision_evidence_sha256": (approval_binding or {}).get("decision_evidence_sha256"),
             "epoch_frequency_sha256": b1._digest(frequencies),
             "epoch_seed_rule": "Random(20261020 + one_based_epoch)",
             "epoch_draw_count_rule": "min(N_aux,N_like), without replacement"}
    model = m1._model(data, config, device)
    optimizer = m1.make_optimizer(model, config)
    stop = EarlyStop(config["max_epochs"], config["min_epochs"], config["patience"])
    stats = SamplerStats(); negative_rng = random.Random(f"p-b1-negative-v1:{config['seed']}")
    binding = binding_for_watch(data, config, rows, queries, model, device, inputs_binding, watch)
    view = {**data, "rows": rows}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    step = 0; records = []; evaluations = []; epoch_draws = []; best = None; last = None
    train_seconds = 0.; eval_seconds = 0.
    diagnostics = {**b1.data_diagnostics(rows, queries, data["support"]),
                   "side_visibility": m1.f1.side_visibility(queries, data["support"])}
    b1._write_progress("train", 0, config["max_epochs"])
    while not stop.stopped:
        if check: check()
        epoch_index = stop.epoch
        tokens = epoch_tokens(len(rows), len(auxiliary_rows), epoch_index)
        epoch_draws.append({"epoch": epoch_index + 1, "auxiliary_drawn": min(len(rows), len(auxiliary_rows)),
                            "tokens_sha256": b1._digest(tokens)})
        for offset in range(0, len(tokens), config["batch_size"]):
            if check: check()
            batch_tokens = tokens[offset:offset + config["batch_size"]]
            batch = epoch_rows(rows, auxiliary_rows, batch_tokens)
            begin = time.monotonic()
            record = feat02.train_step(model, optimizer, batch, view, stats, epoch_index, config,
                                       negative_rng, None, frequencies)
            step += 1
            records.append({**record, "step": step, "epoch": epoch_index + 1,
                            "indices_sha256": b1._digest(batch_tokens)})
            train_seconds += time.monotonic() - begin
        begin = time.monotonic()
        epoch = epoch_index + 1
        b1._write_progress("evaluate", epoch, config["max_epochs"])
        result, epoch_diag = m1.evaluate_model(model, data, queries, config, check)
        primary = result["metrics"]["all"]
        improved = stop.observe(epoch, primary["recall@50"], primary["ndcg@50"])
        predictions_name = f"predictions-epoch-{epoch:03d}.json"
        predictions_sha = b1._write_json_atomic(output / predictions_name, result["predictions"])
        evaluations.append({"epoch": epoch, "result": {k: v for k, v in result.items() if k != "predictions"},
                            "predictions_ref": predictions_name, "predictions_sha256": predictions_sha,
                            "diagnostics": epoch_diag})
        eval_seconds += time.monotonic() - begin
        if check: check()
        state = {"schema": CHECKPOINT_SCHEMA, "binding_sha256": b1._digest(binding), "step": step,
                 "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "negative_rng": negative_rng.getstate(), "stats": b1._stats_value(stats),
                 "selection": stop.state(), **b1._rng_state()}
        last = {"path": "last.pt", "sha256": b1._torch_save_atomic(output / "last.pt", state), "step": step, "epoch": epoch}
        if improved:
            best = {"path": "best.pt", "sha256": b1._torch_save_atomic(output / "best.pt", state),
                    "step": step, "epoch": epoch, "metric": list(stop.best_metric)}
        value = {"schema": MANIFEST_SCHEMA, "kind": "p_watch_positive_train", "status": "running",
                 "run_id": run_id_for_seed(config["seed"]),
                 "config": config, "binding": binding, "steps": step, "records": records,
                 "evaluations": evaluations, "selection": stop.state(), "stats": b1._stats_value(stats),
                 "best": best, "last": last, "data_diagnostics": diagnostics,
                 "training_seconds": train_seconds, "evaluation_seconds": eval_seconds,
                 "hit_max_epochs": stop.epoch >= stop.max_epochs and stop.bad_epochs < stop.patience,
                 "ablation_g0": None, "watch_positive": {**watch, "epoch_draws": epoch_draws}}
        if not stop.stopped:
            b1._write_json_atomic(output / "manifest.json", value)
            b1._write_progress("train", epoch, config["max_epochs"])
    if model.anchored:
        model.load_state_dict(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["model"])
        best_epoch = stop.best_epoch
        ablation = m1.ablation_g0(model, data, queries, config, best_epoch,
                                  evaluations[best_epoch - 1]["result"]["metrics"]["all"]["recall@50"], check)
    else:
        ablation = None
    value = {**value, "status": "complete", "ablation_g0": ablation,
             "run_id": run_id_for_seed(config["seed"]),
             "watch_positive": {**watch, "epoch_draws": epoch_draws}}
    manifest_sha256 = b1._write_json_atomic(output / "manifest.json", value)
    b1._write_progress("complete", epoch, config["max_epochs"])
    return {**value, "manifest_sha256": manifest_sha256, "run_id": run_id_for_seed(config["seed"])}


def verify_run(path, manifest_sha256, data, config, auxiliary_rows, stats_report, stats_stdout_sha256,
               approval_binding=None, device="cpu", check=None, inputs_binding=None):
    import math
    import random
    import torch
    from .baselines import metrics
    from .retrieval_hp import EarlyStop
    from .retrieval_training import SamplerStats, candidate_batch

    path = Path(path); value = read_pinned(path, manifest_sha256); run = path.parent
    if set(value) != MANIFEST_FIELDS or value["schema"] != MANIFEST_SCHEMA or value["kind"] != "p_watch_positive_train" or value["status"] != "complete":
        raise ValueError("complete P watch-positive manifest required")
    synthetic = bool(data.get("synthetic"))
    if value["config"] != config:
        raise ValueError("config mismatch")
    validate_config(config, synthetic)
    if value.get("run_id") != run_id_for_seed(config["seed"]):
        raise ValueError("manifest run_id/seed mismatch")
    if not synthetic and (device != "cuda" or approval_binding is None):
        raise ValueError("real verifier requires CUDA and explicit Kiro approval")
    b1.execution_context(device)
    rows, queries = feat02._view(data, config["history_len"])
    auxiliary_rows = list(auxiliary_rows)
    pool_binding = bind_auxiliary_pool(auxiliary_rows, stats_report, data["support"], data.get("durations_by_item", {}),
                                       data.get("watch_aux_exclusion"), formal=not synthetic)
    if len(rows) != stats_report.get("N_like_train_examples"):
        raise ValueError("loaded original train examples differ from preflight")
    if approval_binding is not None and approval_binding.get("stats_stdout_sha256") != stats_stdout_sha256:
        raise ValueError("approval/stats binding mismatch")
    frequencies = frequency_table(rows, auxiliary_rows)
    watch = {**pool_binding, "watch_aux_exclusion": data.get("watch_aux_exclusion"),
             "N_like": len(rows), "stats_stdout_sha256": stats_stdout_sha256,
             "approval_sha256": (approval_binding or {}).get("approval_sha256"),
             "decision_evidence_sha256": (approval_binding or {}).get("decision_evidence_sha256"),
             "epoch_frequency_sha256": b1._digest(frequencies),
             "epoch_seed_rule": "Random(20261020 + one_based_epoch)",
             "epoch_draw_count_rule": "min(N_aux,N_like), without replacement"}
    if value["binding"] != binding_for_watch(data, config, rows, queries, m1._model(data, config, device), device, inputs_binding, watch):
        raise ValueError("M1/watch/source binding mismatch")
    if value["watch_positive"] != {**watch, "epoch_draws": value["watch_positive"].get("epoch_draws")}:
        raise ValueError("watch-positive manifest summary mismatch")
    stats = SamplerStats(); negative_rng = random.Random(f"p-b1-negative-v1:{config['seed']}")
    epoch_ends = []; record_index = 0
    for epoch_index, evaluation in enumerate(value["evaluations"]):
        if evaluation.get("epoch") != epoch_index + 1:
            raise ValueError("evaluation epoch order mismatch")
        tokens = epoch_tokens(len(rows), len(auxiliary_rows), epoch_index)
        expected_mix = value["watch_positive"]["epoch_draws"][epoch_index]
        if expected_mix != {"epoch": epoch_index + 1, "auxiliary_drawn": min(len(rows), len(auxiliary_rows)),
                            "tokens_sha256": b1._digest(tokens)}:
            raise ValueError("deterministic epoch mix replay mismatch")
        for offset in range(0, len(tokens), config["batch_size"]):
            if check and record_index % 50 == 0: check()
            batch_tokens = tokens[offset:offset + config["batch_size"]]
            batch = epoch_rows(rows, auxiliary_rows, batch_tokens)
            (columns, labels, mask), negatives = candidate_batch(batch, data["idmap"]), []
            record = value["records"][record_index]; step = record_index + 1
            skips = int((mask.sum(1) <= 1).sum())
            finite = all(record.get(key) is None or (type(record.get(key)) is float and math.isfinite(record[key]))
                         for key in ("loss", "gradient_norm"))
            if (record.get("step") != step or record.get("epoch") != epoch_index + 1
                    or record.get("indices_sha256") != b1._digest(batch_tokens)
                    or record.get("negatives_sha256") != b1._digest(negatives)
                    or record.get("negatives") != 0 or record.get("drawn") != len(batch_tokens)
                    or record.get("skipped") != skips or not finite):
                raise ValueError("independent mixed draw replay mismatch")
            stats.record(epoch_index, len(batch_tokens), skips); stats.check(epoch_index)
            record_index += 1
        epoch_ends.append((epoch_index + 1, record_index))
    if (record_index != len(value["records"]) or record_index != value["steps"]
            or b1._stats_value(stats) != value["stats"] or len(epoch_ends) != len(value["evaluations"])
            or not epoch_ends or epoch_ends[-1][1] != value["steps"]):
        raise ValueError("mixed draw/evaluation budget mismatch")
    stop = EarlyStop(config["max_epochs"], config["min_epochs"], config["patience"]); expected_best = None
    full = m1.full_evaluations(run, value)
    for (epoch, step), evaluation, merged in zip(epoch_ends, value["evaluations"], full):
        if check: check()
        predictions = merged["result"]["predictions"]; stored = evaluation.get("diagnostics", {})
        recomputed = metrics(predictions, queries, data["catalog"], (10, 20, 50))
        if recomputed != evaluation["result"]["metrics"]:
            raise ValueError("metrics not reproducible from stored predictions")
        if m1.f1.history_support_slices(predictions, queries, data["support"], data["catalog"]) != stored.get("history_support_slices"):
            raise ValueError("history support slices not reproducible")
        if feat02.support_slices(predictions, queries, data["support"], data["catalog"]) != evaluation["result"].get("support_metrics"):
            raise ValueError("support metrics not reproducible")
        if feat02.top50_unsupported_share(predictions, data["support"]) != stored.get("top50_unsupported_share"):
            raise ValueError("unsupported-share diagnostic not reproducible")
        if (m1.recall_by_history_length(predictions, queries, data["catalog"]) != stored.get("recall50_by_history_length")
                or m1.recall_by_last_event_support(predictions, queries, data["catalog"], data["support"]) != stored.get("recall50_by_last_event_support")):
            raise ValueError("M1 slice diagnostics not reproducible")
        primary = recomputed["all"]
        if stop.observe(epoch, primary["recall@50"], primary["ndcg@50"]):
            expected_best = (epoch, step)
    if not stop.stopped or stop.state() != value["selection"]:
        raise ValueError("selection replay mismatch")
    best = value["best"]
    if type(best) is not dict or (best.get("epoch"), best.get("step")) != expected_best or best.get("metric") != list(stop.best_metric) or best.get("path") != "best.pt":
        raise ValueError("best checkpoint selection mismatch")
    model = m1._model(data, config, device)
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=True) if file_sha256(run / "best.pt") == best["sha256"] else None
    if state is None or state.get("binding_sha256") != b1._digest(value["binding"]) or state["step"] != best["step"] or state["epoch"] != best["epoch"]:
        raise ValueError("best checkpoint hash/identity mismatch")
    model.load_state_dict(state["model"])
    result, diagnostics = m1.evaluate_model(model, data, queries, config, check)
    if result != full[best["epoch"] - 1]["result"] or diagnostics != value["evaluations"][best["epoch"] - 1].get("diagnostics"):
        raise ValueError("best checkpoint exact re-export mismatch")
    if m1.ablation_g0(model, data, queries, config, best["epoch"], m1.recomputed_best_recall(value, best), check) != value["ablation_g0"]:
        raise ValueError("best checkpoint g->0 ablation mismatch")
    last = value["last"]
    if type(last) is not dict or last.get("path") != "last.pt" or last.get("step") != value["steps"] or file_sha256(run / "last.pt") != last.get("sha256"):
        raise ValueError("last checkpoint hash/identity mismatch")
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    if (state.get("binding_sha256") != b1._digest(value["binding"]) or state["step"] != value["steps"]
            or state["stats"] != value["stats"] or state["selection"] != value["selection"]
            or not state_equal(state["negative_rng"], negative_rng.getstate())):
        raise ValueError("last checkpoint state mismatch")
    if sorted(candidate.name for candidate in run.glob("*.pt")) != ["best.pt", "last.pt"]:
        raise ValueError("checkpoint policy best+last violated")
    return {"status": "verified", "run_id": run_id_for_seed(config["seed"]),
            "best_epoch": best["epoch"], "steps": value["steps"], "manifest_sha256": manifest_sha256}


def _real_context(args, config):
    import torch
    stats_report = validate_stats_report(args.stats_report, args.stats_sha)
    approval_binding = validate_training_approval(args.approval, args.approval_sha, run_id_for_seed(config["seed"]),
                                                   config["seed"], args.stats_sha)
    spec = feat02.validate_inputs(read_pinned(args.inputs, args.inputs_sha), m1.FEATURE)
    if not torch.cuda.is_available():
        raise ValueError("actual CUDA environment required")
    target = Path(args.output) if args.command == "train" else Path(args.manifest).parent
    check = b1.resource_guard(target, spec["disk_cap_bytes"]); check()
    data, partitions, base_binding = b1.real_inputs(spec, with_partitions=True)
    data, extra_binding = feat02.apply_feat02(data, spec, config["history_len"])
    inputs_binding = {**base_binding, **extra_binding, "inputs_sha256": args.inputs_sha}
    stats_input_binding = validate_stats_data_binding(stats_report, args.stats_sha, args.inputs_sha,
                                                       spec["items_metadata_sha256"], data["support"], base_binding)
    metadata_path = Path(spec["raw_root"]) / "metadata/items_metadata.parquet"
    from .history_side import load_durations
    durations = load_durations(metadata_path, data["catalog"])
    if file_sha256(metadata_path) != spec["items_metadata_sha256"]:
        raise ValueError("items metadata changed after approved train-only preflight")
    thresholds = {name: bucket["threshold"] for name, bucket in stats_report["buckets"].items()
                  if bucket["threshold"] is not None}
    from .p_watch_positive_rows import build_auxiliary_rows
    raw_auxiliary_rows = list(build_auxiliary_rows(partitions, data["support"], durations, thresholds))
    data = {**data, "durations_by_item": durations}
    history_manifest = read_pinned(spec["history_manifest"], spec["history_sha256"])
    slice_sha = history_manifest.get("artifacts", {}).get("train-slices.json")
    slice_path = Path(spec["history_manifest"]).parent / "train-slices.json"
    if (slice_sha != "6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26"
            or slice_path.is_symlink() or not slice_path.is_file() or file_sha256(slice_path) != slice_sha):
        raise ValueError("frozen S0 train-slices identity mismatch")
    slices = strict_json(slice_path.read_text())
    if slices.get("popularity_thresholds") != {"median": 1, "p90": 11}:
        raise ValueError("frozen S0 heat thresholds mismatch")
    item_counts = slices.get("item_like_counts")
    if type(item_counts) is not dict:
        raise ValueError("frozen S0 item heat counts unavailable")
    item_heat_bins = {}
    for item in data["support"]:
        count = item_counts.get(str(item))
        if type(count) is not int or count < 1:
            raise ValueError("frozen S0 supported-item heat count missing")
        item_heat_bins[item] = "low" if count <= 1 else ("mid" if count <= 11 else "high")
    from .p_watch_positive_nonempty import filter_auxiliary_pool
    auxiliary_rows, exclusion = filter_auxiliary_pool(raw_auxiliary_rows, stats_report, data["support"], durations,
                                                        item_heat_bins, formal=True)
    exclusion["s0_train_slices_sha256"] = slice_sha
    data["watch_aux_exclusion"] = exclusion
    pool_binding = bind_auxiliary_pool(auxiliary_rows, stats_report, data["support"], durations, exclusion, formal=True)
    if pool_binding["N_aux"] != 700633:
        raise ValueError("constructed auxiliary row count differs from frozen post-filter count")
    inputs_binding = {**inputs_binding, "watch_stats": stats_input_binding,
                      "training_approval_sha256": approval_binding["approval_sha256"],
                      "s0_train_slices_sha256": slice_sha}
    return target, check, data, inputs_binding, stats_report, auxiliary_rows, approval_binding


def main(argv=None):
    import argparse
    import sys
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] == "preflight":
        from vkgr import p_watch_positive_ops
        return p_watch_positive_ops.main(["preflight", *args_in[1:]])
    parser = argparse.ArgumentParser(description="P watch-positive M1-mean trainer and independent verifier")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--inputs", required=True); p.add_argument("--inputs-sha", required=True)
        p.add_argument("--config", required=True); p.add_argument("--config-sha", required=True)
        p.add_argument("--stats-report", required=True); p.add_argument("--stats-sha", required=True)
        p.add_argument("--approval", required=True); p.add_argument("--approval-sha", required=True)
        if name == "train":
            p.add_argument("--output", required=True)
        else:
            p.add_argument("--manifest", required=True); p.add_argument("--manifest-sha", required=True)
    args = parser.parse_args(args_in)
    try:
        config = read_pinned(args.config, args.config_sha); validate_config(config)
        target, check, data, inputs_binding, stats_report, auxiliary_rows, approval_binding = _real_context(args, config)
        if args.command == "train":
            result = run_session(target, data, config, auxiliary_rows, stats_report, args.stats_sha,
                                 approval_binding, "cuda", check, inputs_binding)
            print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                              "manifest_sha256": result["manifest_sha256"],
                              "best_epoch": result["selection"]["best_epoch"]}, sort_keys=True))
            return 0
        result = verify_run(args.manifest, args.manifest_sha, data, config, auxiliary_rows,
                            stats_report, args.stats_sha, approval_binding, "cuda", check, inputs_binding)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f"ERROR P-WATCH-POSITIVE gate failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
