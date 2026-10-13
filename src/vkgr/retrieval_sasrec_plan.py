"""Frozen P4H-01 plan schemas; no model, process, path I/O, or remote action."""
import copy
import hashlib
import json

TASK_IDS = [
    "P4H-01", "P4H-02", "P4H-03A", "P4H-03B", "P4H-04",
    "P4H-05", "P4H-06", "P4H-07", "P4H-08",
]
REMOTE_DECISIONS = {
    "D-P4H-REMOTE-01": "BLOCKED",
    "D-P4H-REMOTE-02": "BLOCKED",
    "D-P4H-G5-01": "BLOCKED_DEFAULT_ORIGINAL_P4D_TT",
}
ACCEPTED_INPUTS = {
    "p4d_core_manifest": "6ddccee458c3c66bb185d51527c0dc32981ae83949efe420eef97ed45881ac6e",
    "p4d_runtime_config": "d96ef65a7d3946edf0d1b6010e6e4b91bf2bf1d9d0843973df7d2aaf0acd272c",
    "p4d_representative_checkpoint": "5a8549b8f1fe6799028f3f2b71fb4226a28c7516486044ffde11c0b9088725db",
    "p4e_config": "dd094dd7cb378feefe48a89b26934c45899f9c13e547d15d427bd9504ce614cd",
    "p4e_manifest": "b25336929fe6b94d24a5e847343f63cfe4c54b24ae0f9cd9b6f05deab58c58cc",
    "p4e_completion": "7ffcf57e1c4aedf15b73e12c1ebfc07e1f9d72da29a131d3a3d38f1ebcb7da65",
    "p4e_summary": "64af9526000212fcb6ffa3159a60b2b894b1619b7dba0507761f359ca8600623",
    "p4f_target_review": "c227d3ab6c75bf632b5ebde550b51c3397fc75510494f4ce5c3c95e77885d437",
    "p4f_fresh_review": "92459e7429291d50fb7bb87472c1ea6380fac604b5e255a62cdbbd0e771dd9f3",
    "p4f_fusion_source": "5f67b553f8661ab81488ec15a32336c139789721901ff9d94511a66352ddbdeb",
    "p4f_ann_source": "00732bc23f1130d9fcc83d0d592ec0bc8fab36e197d841ef8d203ad9539181b0",
    "p4f_infer_source": "c43d6d50cdc70c87f02e817658b350a50db469880e55b3e05f7d337fe098ae97",
    "p4f_export_manifest": "1cdf012b8ebf31c1b1dc67cf66503b51f0cf08d95b3b659ca9d5ee4ec14e0fe8",
    "p4f_fusion_manifest": "3f4301f1b3152984ce6972c302ab760fcbcf8264458290a30a43a283421c6da7",
    "p4f_infer_completion": "60f28a36045840ebafeaf97c907ee9bebaa143537e9ce42adcb263592e0c402d",
    "p4g_freeze": "c0bf8ddc9cdbee1a5902f364c3a84717f2c4e17662bcb3995eec4146743c3f87",
    "p4g_freeze_review": "c8fd0ffa76f2a884bddb8e8b7607c41f3fd77731dbeff8f501c51786c4700e8d",
    "p4g_selection": "2fe523023eb38e09308a7e26e17fee793146698c1caa2f596c8ba18a4f9534b9",
}


def _slot(slot_id, history_source, seed, learning_rate, temperature, role):
    return {
        "slot_id": slot_id,
        "history_source": history_source,
        "encoder": "id_only_sasrec_style",
        "seed": seed,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "role": role,
    }


FORMAL_SLOTS = [
    _slot("P4H-S01", "like", 42, 0.001, 0.07, "secondary_screen"),
    _slot("P4H-S02", "like", 42, 0.001, 0.15, "screen_and_primary_matched"),
    _slot("P4H-S03", "like", 42, 0.0003, 0.07, "secondary_screen"),
    _slot("P4H-S04", "like", 42, 0.0003, 0.15, "secondary_screen"),
    _slot("P4H-S05", "observed", 42, 0.001, 0.07, "secondary_screen"),
    _slot("P4H-S06", "observed", 42, 0.001, 0.15, "screen_and_primary_matched"),
    _slot("P4H-S07", "observed", 42, 0.0003, 0.07, "secondary_screen"),
    _slot("P4H-S08", "observed", 42, 0.0003, 0.15, "secondary_screen"),
    _slot("P4H-C01", "like", 2024, 0.001, 0.15, "independent_confirmation_primary"),
    _slot("P4H-C02", "like", 2025, 0.001, 0.15, "independent_confirmation_primary"),
    _slot("P4H-C03", "observed", 2024, 0.001, 0.15, "independent_confirmation_primary"),
    _slot("P4H-C04", "observed", 2025, 0.001, 0.15, "independent_confirmation_primary"),
]

COMMON_ROLES = [
    "code-archive", "source-fingerprint", "dependency-lock", "python-executable-receipt",
    "controller-executable", "p4h-plan", "p4h-config", "p4h-protocol",
    "g-feed-descriptor", "cooperative-stop-control-schema",
]
PILOT_ROLE_SPECS = {
    "code-archive": ("p4h-code-archive-v1", ["members", "source_sha256"]),
    "source-fingerprint": ("p4h-source-fingerprint-v1", ["source_sha256", "members"]),
    "dependency-lock": ("p4h-dependency-lock-v1", ["lock_sha256", "dependencies"]),
    "python-executable-receipt": ("p4h-python-executable-v1", ["python_version", "executable_sha256"]),
    "controller-executable": ("p4h-controller-executable-v1", ["path_sha256", "source_sha256"]),
    "prepared-train-rows": ("p4h-prepared-train-rows-v1", ["user_id", "history", "history_positions", "target", "target_position"]),
    "watch-train": ("p4h-watch-train-v1", ["example_id", "history", "history_positions"]),
    "catalog": ("p4h-catalog-v1", ["item_ids"]),
    "idmap": ("p4h-idmap-v1", ["items", "pad", "start"]),
    "train-prefix-target-free-sidecar": ("p4h-train-prefix-target-free-v1", ["query_id", "history"]),
    "p4h-plan": ("p4h-plan-v5", ["plan_sha256", "task_ids", "formal_slots"]),
    "p4h-config": ("p4h-config-v2", ["config_sha256", "remote_decisions"]),
    "p4h-protocol": ("p4h-protocol-v2", ["protocol_sha256", "schema_version"]),
    "g-feed-descriptor": ("p4h-g-feed-descriptor-v2", ["descriptor_sha256", "g_job_identity"]),
    "cooperative-stop-control-schema": ("p4h-stop-control-v1", ["request_schema", "receipt_schema"]),
}

VERIFIER_ROLE_SOURCES = {
    "pilot-verify": {
        "sealed-wrapper": ["sealed-pilot-operation-receipt"],
        "input_hashes": ["prepared-train-rows", "watch-train", "catalog", "idmap", "train-prefix-target-free-sidecar"],
        "artifact_hashes": ["pilot-capacity-measurement-index", "pilot-resource-ledger", "pilot-stage-event-ledger", "pilot-opened-artifact-ledger", "pilot-artifact-index"],
    },
    "formal-verify": {
        "sealed-wrapper": ["sealed-formal-run-receipt"],
        "input_hashes": ["prepared-train-rows", "formal-history-view", "validation-query-target-free", "validation-target", "catalog", "idmap", "mean-control-identity", "registered-slot-ledger"],
        "artifact_hashes": ["formal-checkpoint-index", "formal-epoch-metric-index", "formal-prediction-hash-index", "formal-draw-mask-provenance", "formal-resource-ledger", "formal-stage-event-ledger", "formal-opened-artifact-ledger", "formal-artifact-index"],
    },
    "infer-route-verify": {
        "sealed-wrapper": ["sealed-route-operation-receipt"],
        "input_hashes": ["chosen-checkpoint", "chosen-history-view", "inference-query-target-free", "catalog", "idmap", "selection-receipt", "completion-receipt", "route-contract"],
        "artifact_hashes": ["route-shard-index", "route-candidate-hash-index", "route-resource-ledger", "route-stage-event-ledger", "route-opened-artifact-ledger", "route-artifact-index"],
    },
}

COMMAND_STAGES = {
    "prepare-controller": {"stages": ["pilot", "pilot-verify", "formal-train", "formal-resume", "formal-verify", "infer-route", "infer-route-verify"], "runtime": "remote", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "pilot": {"stages": ["pilot"], "runtime": "remote", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "pilot-verify": {"stages": ["pilot-verify"], "runtime": "remote", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "formal-train": {"stages": ["formal-train"], "runtime": "remote", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "formal-resume": {"stages": ["formal-resume"], "runtime": "remote", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "formal-verify": {"stages": ["formal-verify"], "runtime": "remote", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "select": {"stages": ["select"], "runtime": "local", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "completion-verify": {"stages": ["completion-verify"], "runtime": "local", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "summarize": {"stages": ["summarize"], "runtime": "local", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "infer-route": {"stages": ["infer-route"], "runtime": "remote", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
    "infer-route-verify": {"stages": ["infer-route-verify"], "runtime": "remote", "business_args": ["--operation-manifest", "--operation-manifest-sha256"]},
}

FEED_CLAIM_BOUNDARY = "accidental_duplicate_stale_wrong_job_only_not_malicious_same_uid"
FEED_FIELDS = {
    "schema_version", "feed_root", "feed_root_identity", "feed_owner_uid", "feed_mode",
    "head_name", "receipt_name_pattern", "publisher_lock", "publisher_process",
    "initial_no_other_writer_inventory_sha256", "g_job_identity", "receipt_schema_id",
    "initial_seq", "initial_receipt_sha256", "allowed_statuses",
}
SEALED_FIELDS = {
    "schema_version", "predecessor_stage", "predecessor_operation_id",
    "predecessor_operation_manifest_sha256", "predecessor_output_root_identity",
    "predecessor_control_root_identity", "controller_start_receipt_sha256",
    "admission_sha256", "final_g_receipt_sha256", "worker_birth_inventory_sha256",
    "input_hashes", "artifact_hashes", "complete_artifact_set_sha256",
    "terminal_receipt_sha256", "closure_receipt_sha256", "spent", "state",
}

EXPECTED_CONFIG = {
    "schema_version": 2,
    "task_id": "P4H-01",
    "plan_version": "p-sasrec-parallel-v5",
    "source_plan_sha256": "ea3f7566750c4fad13dfba80fd03a5f8fdf7e81c4c68aa8912b09b3e0e732121",
    "source_review_sha256": "3de5ad402b6dd9e033d068567d046071ba4998e888914f5ac49c155c035287a1",
    "task_ids": TASK_IDS,
    "formal_slots": FORMAL_SLOTS,
    "mean_controls": [
        {"history_source": "like", "source": "P4D-id_mean", "mode": "read_only"},
        {"history_source": "observed", "source": "P4E-watch_history-id_mean", "mode": "read_only"},
    ],
    "remote_decisions": REMOTE_DECISIONS,
    "proposed_thresholds": {
        "throughput_floor": {"value": 0.8, "status": "PROPOSED_BLOCKING_DECISION"},
        "allowed_g_slowdown": {"value": 1.25, "status": "PROPOSED_BLOCKING_DECISION"},
        "approval_ttl": {"value": None, "status": "PROPOSED_BLOCKING_DECISION"},
        "wall_safety_multiplier": {"value": None, "status": "PROPOSED_BLOCKING_DECISION"},
    },
    "common_test": "CLOSED",
    "best_seed_selection": False,
    "g4_cutoff": {"trigger": "G4_FREEZE", "hard_inputs": ["G4-freeze-receipt"], "optional_inputs": ["P4H-route-receipt"], "default": "ORIGINAL_P4D_TT", "wait_for_p4h": False},
    "route": {"top_k": 80, "tie_break": "item_id_ascending", "seen_masking": False, "g5_route_quota": 80, "g5_total_candidates": 400, "p4h_replaces": "best_core_TT", "p4h_adds_sixth_route": False},
    "selection": {"primary": ["validation_all_eligible_recall@50", "validation_all_eligible_ndcg@50"], "epoch_tie": "earlier", "config_tie": "fixed_enumeration", "route_tie": "like", "confirmation_seeds": [2024, 2025], "eligibility_recall_delta": "each_seed_strictly_positive"},
    "primary_slices": ["train_like_count_1", "train_like_count_2_5", "train_like_count_6_20", "train_like_count_gt20"],
    "accepted_inputs": ACCEPTED_INPUTS,
    "contract_registry": {
        "pilot_roles": list(PILOT_ROLE_SPECS),
        "verifier_role_counts": {"pilot-verify": 21, "formal-verify": 27, "infer-route-verify": 25},
        "commands": list(COMMAND_STAGES),
        "sealed_predecessor_schema": "p4h-sealed-predecessor-operation-v1",
        "feed_descriptor_schema": "p4h-g-feed-descriptor-v2",
    },
    "operation_lifecycle": {"mutating_terminal": "completed+unverified", "verified_terminal": "completed+verified", "failure_terminal": "failed+spent", "verifier_requires_predecessor": "CLOSED", "distinct_roots_controller_admission_g_receipt_worker_pgid": True},
    "g_feed": {"receipt_schema_id": "p4h-g-feed-receipt-v2", "claim_boundary": FEED_CLAIM_BOUNDARY, "publisher_lock_creation": "O_CREAT|O_EXCL", "feed_mode": "0700", "unknown_writer_policy": "BLOCKED_OR_COOPERATIVE_STOP", "new_dependency": False},
    "public_boundary": {"server_paths_allowed": False, "per_query_predictions_allowed": False, "checkpoint_or_index_allowed": False, "identity_hash_examples_only": True},
    "local_scope": {"real_data": False, "validation_or_test_read": False, "remote_action": False, "gpu": False, "commit_push_publish": False},
}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_config(value):
    if not isinstance(value, dict) or canonical_bytes(value) != canonical_bytes(EXPECTED_CONFIG):
        raise ValueError("frozen P4H-01 config mismatch")
    if len({row["slot_id"] for row in value["formal_slots"]}) != 12:
        raise ValueError("slot identity mismatch")
    if any(not _pin(digest) for digest in value["accepted_inputs"].values()):
        raise ValueError("accepted input pin")
    return value


def _stop(reason):
    raise ValueError("STOP_BEFORE_PATH: " + reason)


def _lineage(role):
    return {
        "split": "train" if role in ("prepared-train-rows", "watch-train", "train-prefix-target-free-sidecar") else "static",
        "source_partitions": ["train00", "train01"] if role in ("prepared-train-rows", "watch-train", "train-prefix-target-free-sidecar") else [],
        "derived_from": "frozen_train_prefixes" if role == "train-prefix-target-free-sidecar" else "accepted_static_or_train_source",
        "semantic_tags": ["query_identity", "history_item_ids"] if role == "train-prefix-target-free-sidecar" else [],
    }


def synthetic_pilot_manifest():
    roles = []
    for index, (role, (schema_id, fields)) in enumerate(PILOT_ROLE_SPECS.items()):
        roles.append({
            "role": role,
            "schema_id": schema_id,
            "schema_fields": list(fields),
            "lineage": _lineage(role),
            "path": "/server-only/p4h/" + role + ".json",
            "sha256": hashlib.sha256(role.encode()).hexdigest(),
            "size": index + 1,
        })
    train_events = [
        {"user_id": 7, "item_id": 10, "position": 0, "partition": "train00"},
        {"user_id": 7, "item_id": 11, "position": 1, "partition": "train00"},
        {"user_id": 7, "item_id": 12, "position": 2, "partition": "train00"},
    ]
    payloads = {
        "train_events": train_events,
        "prepared-train-rows": [{"user_id": 7, "history": [10, 11], "history_positions": [0, 1], "target": 12, "target_position": 2}],
        "watch-train": [{"example_id": 0, "history": [10, 11], "history_positions": [0, 1]}],
        "train-prefix-target-free-sidecar": [{"query_id": 0, "history": [10, 11]}],
        "event_ledger": ["optimizer_step", "capacity", "resource", "checkpoint", "target_free_decode", "stop", "seal"],
        "opened_roles": list(PILOT_ROLE_SPECS),
    }
    return {"stage": "pilot", "roles": roles, "exclusive_output_root": "/server-only/p4h/pilot-output", "exclusive_control_root": "/server-only/p4h/pilot-control"}, payloads


def validate_pilot_manifest(manifest, before_path=None):
    if not isinstance(manifest, dict) or set(manifest) != {"stage", "roles", "exclusive_output_root", "exclusive_control_root"} or manifest["stage"] != "pilot":
        _stop("pilot manifest schema")
    roles = manifest["roles"]
    if not isinstance(roles, list) or len(roles) != len(PILOT_ROLE_SPECS):
        _stop("pilot role count")
    names = [row.get("role") for row in roles if isinstance(row, dict)]
    if len(names) != len(roles) or len(set(names)) != len(names) or set(names) != set(PILOT_ROLE_SPECS):
        _stop("pilot exact role set")
    for row in roles:
        if set(row) != {"role", "schema_id", "schema_fields", "lineage", "path", "sha256", "size"}:
            _stop("role fields")
        role = row["role"]
        schema_id, fields = PILOT_ROLE_SPECS[role]
        if row["schema_id"] != schema_id or row["schema_fields"] != fields:
            _stop("role schema")
        lineage = row["lineage"]
        if not isinstance(lineage, dict) or set(lineage) != {"split", "source_partitions", "derived_from", "semantic_tags"}:
            _stop("lineage schema")
        text = json.dumps(lineage, sort_keys=True).lower()
        if "validation" in text or "holdout" in text or "test" in text:
            _stop("non-train lineage")
        if role in ("prepared-train-rows", "watch-train", "train-prefix-target-free-sidecar"):
            if lineage["split"] != "train" or not lineage["source_partitions"] or any(not value.startswith("train") for value in lineage["source_partitions"]):
                _stop("train lineage")
        if role == "train-prefix-target-free-sidecar":
            semantic_text = " ".join([lineage["derived_from"], *lineage["semantic_tags"]]).lower()
            if any(token in semantic_text for token in ("target", "oov", "metric", "quality")):
                _stop("sidecar forbidden semantics")
        if not isinstance(row["path"], str) or not row["path"].startswith("/") or not _pin(row["sha256"]) or not isinstance(row["size"], int) or row["size"] < 0:
            _stop("declared path metadata")
    if before_path is not None:
        for row in roles:
            before_path(row["role"])
    return {"role_count": len(roles), "validation_target_read": False, "quality_metrics_computed": False}


def _record(record, fields):
    return isinstance(record, dict) and set(record) == set(fields)


def _contains_forbidden(value):
    if isinstance(value, dict):
        return any(any(token in str(key).lower() for token in ("target", "oov", "metric", "quality")) or _contains_forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden(item) for item in value)
    if isinstance(value, str):
        return any(token in value.lower() for token in ("target", "oov", "metric", "quality"))
    return False


def validate_pilot_payloads(payloads):
    if not isinstance(payloads, dict) or set(payloads) != {"train_events", "prepared-train-rows", "watch-train", "train-prefix-target-free-sidecar", "event_ledger", "opened_roles"}:
        raise ValueError("pilot payload set")
    events = payloads["train_events"]
    event_index = {(row["user_id"], row["position"]): row for row in events if _record(row, ("user_id", "item_id", "position", "partition")) and row["partition"].startswith("train")}
    prepared = payloads["prepared-train-rows"]
    watch = payloads["watch-train"]
    if len(prepared) != len(watch) or not prepared:
        raise ValueError("prepared/watch bijection")
    for index, (row, observed) in enumerate(zip(prepared, watch)):
        if not _record(row, PILOT_ROLE_SPECS["prepared-train-rows"][1]) or not _record(observed, PILOT_ROLE_SPECS["watch-train"][1]) or observed["example_id"] != index:
            raise ValueError("prepared/watch identity")
        if len(row["history"]) != len(row["history_positions"]) or any(left >= right for left, right in zip(row["history_positions"], row["history_positions"][1:])) or any(position >= row["target_position"] for position in row["history_positions"]):
            raise ValueError("prepared strict past")
        target_event = event_index.get((row["user_id"], row["target_position"]))
        if target_event is None or target_event["item_id"] != row["target"]:
            raise ValueError("prepared target train event")
        if len(observed["history"]) != len(observed["history_positions"]) or any(position >= row["target_position"] for position in observed["history_positions"]) or any(event_index.get((row["user_id"], position), {}).get("item_id") != item for item, position in zip(observed["history"], observed["history_positions"])):
            raise ValueError("watch strict past identity")
    sidecar = payloads["train-prefix-target-free-sidecar"]
    if not sidecar or any(not _record(row, PILOT_ROLE_SPECS["train-prefix-target-free-sidecar"][1]) or _contains_forbidden(row) for row in sidecar):
        raise ValueError("target-free sidecar")
    allowed_events = {"optimizer_step", "capacity", "resource", "checkpoint", "target_free_decode", "stop", "seal"}
    if set(payloads["event_ledger"]) - allowed_events or payloads["opened_roles"] != list(PILOT_ROLE_SPECS):
        raise ValueError("pilot ledgers")
    return {"prepared_watch_bijection": True, "sidecar_target_free": True, "validation_target_read": False, "quality_metrics_computed": False}


SEALED_FIELD_ORDER = [
    "schema_version", "predecessor_stage", "predecessor_operation_id",
    "predecessor_operation_manifest_sha256", "predecessor_output_root_identity",
    "predecessor_control_root_identity", "controller_start_receipt_sha256",
    "admission_sha256", "final_g_receipt_sha256", "worker_birth_inventory_sha256",
    "input_hashes", "artifact_hashes", "complete_artifact_set_sha256",
    "terminal_receipt_sha256", "closure_receipt_sha256", "spent", "state",
]
ROLE_SCHEMA_SPECS = {role: spec for role, spec in PILOT_ROLE_SPECS.items()}
ROLE_SCHEMA_SPECS.update({
    "formal-history-view": ("p4h-formal-history-view-v1", ["example_id", "history", "history_positions"]),
    "validation-query-target-free": ("p4h-validation-query-target-free-v1", ["query_id", "user_id", "history", "history_positions"]),
    "validation-target": ("p4h-validation-target-v1", ["query_id", "target", "oov"]),
    "mean-control-identity": ("p4h-mean-control-identity-v1", ["schema_version", "history_source", "checkpoint_sha256", "prediction_sha256", "metric_record_sha256"]),
    "registered-slot-ledger": ("p4h-registered-slot-ledger-v1", ["schema_version", "slot_id", "arm", "lr", "temperature", "seed", "split", "spent_before"]),
    "pilot-capacity-measurement-index": ("p4h-pilot-capacity-measurements-v1", ["schema_version", "operation_id", "arms", "measurements", "units"]),
    "pilot-resource-ledger": ("p4h-resource-ledger-v1", ["schema_version", "operation_id", "samples", "peak", "caps", "spent"]),
    "pilot-stage-event-ledger": ("p4h-stage-event-ledger-v1", ["schema_version", "operation_id", "events"]),
    "pilot-opened-artifact-ledger": ("p4h-opened-artifact-ledger-v1", ["schema_version", "operation_id", "opened_roles", "opened_sha256"]),
    "pilot-artifact-index": ("p4h-artifact-index-v1", ["schema_version", "operation_id", "artifacts", "complete_artifact_set_sha256"]),
    "formal-checkpoint-index": ("p4h-formal-checkpoint-index-v1", ["schema_version", "operation_id", "best", "last", "preemption"]),
    "formal-epoch-metric-index": ("p4h-formal-epoch-metric-index-v1", ["schema_version", "operation_id", "records", "selection_replay_sha256"]),
    "formal-prediction-hash-index": ("p4h-formal-prediction-hash-index-v1", ["schema_version", "operation_id", "epoch_hashes", "fresh_best_hash"]),
    "formal-draw-mask-provenance": ("p4h-formal-draw-mask-provenance-v1", ["schema_version", "operation_id", "step_records", "aggregate_sha256"]),
    "formal-resource-ledger": ("p4h-resource-ledger-v1", ["schema_version", "operation_id", "samples", "peak", "caps", "spent"]),
    "formal-stage-event-ledger": ("p4h-stage-event-ledger-v1", ["schema_version", "operation_id", "events"]),
    "formal-opened-artifact-ledger": ("p4h-opened-artifact-ledger-v1", ["schema_version", "operation_id", "opened_roles", "opened_sha256"]),
    "formal-artifact-index": ("p4h-artifact-index-v1", ["schema_version", "operation_id", "artifacts", "complete_artifact_set_sha256"]),
    "chosen-checkpoint": ("p4h-chosen-checkpoint-v1", ["schema_version", "slot_id", "history_source", "checkpoint_sha256", "selection_receipt_sha256"]),
    "chosen-history-view": ("p4h-route-history-view-v1", ["query_id", "user_id", "history", "history_positions"]),
    "inference-query-target-free": ("p4h-inference-query-target-free-v1", ["query_id", "user_id", "history", "history_positions"]),
    "selection-receipt": ("p4h-selection-receipt-v1", ["schema_version", "chosen_slot_id", "chosen_checkpoint_sha256", "history_source", "eligibility_inputs_sha256"]),
    "completion-receipt": ("p4h-completion-receipt-v1", ["schema_version", "registered_slot_ledger_sha256", "terminal_identities", "eligibility"]),
    "route-contract": ("p4h-route-contract-v1", ["schema_version", "split", "top_k", "tie_break", "seen_masking", "catalog_sha256", "query_sha256"]),
    "route-shard-index": ("p4h-route-shard-index-v1", ["schema_version", "operation_id", "shards", "query_coverage_sha256"]),
    "route-candidate-hash-index": ("p4h-route-candidate-hash-index-v1", ["schema_version", "operation_id", "per_shard_hashes", "aggregate_sha256"]),
    "route-resource-ledger": ("p4h-resource-ledger-v1", ["schema_version", "operation_id", "samples", "peak", "caps", "spent"]),
    "route-stage-event-ledger": ("p4h-stage-event-ledger-v1", ["schema_version", "operation_id", "events"]),
    "route-opened-artifact-ledger": ("p4h-opened-artifact-ledger-v1", ["schema_version", "operation_id", "opened_roles", "opened_sha256"]),
    "route-artifact-index": ("p4h-artifact-index-v1", ["schema_version", "operation_id", "artifacts", "complete_artifact_set_sha256"]),
})


def _role_spec(role, source):
    if source == "sealed-wrapper":
        return "p4h-sealed-predecessor-operation-v1", SEALED_FIELD_ORDER
    if role not in ROLE_SCHEMA_SPECS:
        raise ValueError("unknown role schema")
    return ROLE_SCHEMA_SPECS[role]


def synthetic_verifier_manifest(stage):
    if stage not in VERIFIER_ROLE_SOURCES:
        raise ValueError("verifier stage")
    groups = VERIFIER_ROLE_SOURCES[stage]
    roles = []
    for role in COMMON_ROLES:
        schema_id, schema_fields = _role_spec(role, "common")
        roles.append({"role": role, "source": "common", "schema_id": schema_id, "schema_fields": list(schema_fields), "path": "/server-only/" + role, "sha256": hashlib.sha256(role.encode()).hexdigest(), "size": 1})
    for source in ("sealed-wrapper", "input_hashes", "artifact_hashes"):
        for role in groups[source]:
            schema_id, schema_fields = _role_spec(role, source)
            roles.append({"role": role, "source": source, "schema_id": schema_id, "schema_fields": list(schema_fields), "path": "/server-only/" + role, "sha256": hashlib.sha256(role.encode()).hexdigest(), "size": 1})
    predecessor = {"pilot-verify": "pilot", "formal-verify": "formal-train", "infer-route-verify": "infer-route"}[stage]
    input_hashes = [{"role": role, "sha256": hashlib.sha256(("input:" + role).encode()).hexdigest()} for role in groups["input_hashes"]]
    artifact_hashes = [{"role": role, "path": "/sealed/" + role, "sha256": hashlib.sha256(("artifact:" + role).encode()).hexdigest(), "size": 1} for role in groups["artifact_hashes"]]
    sealed = {
        "schema_version": 1,
        "predecessor_stage": predecessor,
        "predecessor_operation_id": "predecessor-001",
        "predecessor_operation_manifest_sha256": "1" * 64,
        "predecessor_output_root_identity": "2" * 64,
        "predecessor_control_root_identity": "3" * 64,
        "controller_start_receipt_sha256": "4" * 64,
        "admission_sha256": "5" * 64,
        "final_g_receipt_sha256": "6" * 64,
        "worker_birth_inventory_sha256": "7" * 64,
        "input_hashes": input_hashes,
        "artifact_hashes": artifact_hashes,
        "complete_artifact_set_sha256": hashlib.sha256(canonical_bytes(artifact_hashes)).hexdigest(),
        "terminal_receipt_sha256": "8" * 64,
        "closure_receipt_sha256": "9" * 64,
        "spent": {"wall_seconds": 1, "new_bytes": 1},
        "state": "CLOSED",
    }
    return {"manifest": {"stage": stage, "roles": roles}, "sealed": sealed}


def validate_verifier_manifest(manifest, sealed, before_nested_path=None):
    if not isinstance(manifest, dict) or set(manifest) != {"stage", "roles"} or manifest["stage"] not in VERIFIER_ROLE_SOURCES:
        _stop("verifier manifest")
    stage = manifest["stage"]
    groups = VERIFIER_ROLE_SOURCES[stage]
    expected_sources = {role: "common" for role in COMMON_ROLES}
    for source, roles in groups.items():
        expected_sources.update({role: source for role in roles})
    rows = manifest["roles"]
    role_names = [row.get("role") for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not isinstance(rows, list) or len(rows) != len(expected_sources) or len(role_names) != len(rows) or len(set(role_names)) != len(rows):
        _stop("verifier role count")
    if {row.get("role"): row.get("source") for row in rows} != expected_sources:
        _stop("verifier role source")
    for row in rows:
        if set(row) != {"role", "source", "schema_id", "schema_fields", "path", "sha256", "size"}:
            _stop("verifier role fields")
        schema_id, schema_fields = _role_spec(row["role"], row["source"])
        if row["schema_id"] != schema_id or row["schema_fields"] != schema_fields:
            _stop("verifier role schema")
        if not isinstance(row["path"], str) or not row["path"].startswith("/") or not _pin(row["sha256"]) or not isinstance(row["size"], int) or row["size"] < 0:
            _stop("verifier role path metadata")
    if not isinstance(sealed, dict) or set(sealed) != SEALED_FIELDS or sealed["state"] != "CLOSED":
        _stop("sealed schema/state")
    expected_predecessors = {
        "pilot-verify": {"pilot"},
        "formal-verify": {"formal-train", "formal-resume"},
        "infer-route-verify": {"infer-route"},
    }
    if sealed["predecessor_stage"] not in expected_predecessors[stage]:
        _stop("sealed predecessor stage")
    wrapper = groups["sealed-wrapper"]
    if len(wrapper) != 1:
        _stop("sealed wrapper")
    input_roles = [row.get("role") for row in sealed["input_hashes"] if isinstance(row, dict)]
    artifact_roles = [row.get("role") for row in sealed["artifact_hashes"] if isinstance(row, dict)]
    if set(input_roles) != set(groups["input_hashes"]) or len(input_roles) != len(set(input_roles)):
        _stop("sealed input role set")
    if set(artifact_roles) != set(groups["artifact_hashes"]) or len(artifact_roles) != len(set(artifact_roles)):
        _stop("sealed artifact role set")
    if set(wrapper) & (set(input_roles) | set(artifact_roles)) or set(input_roles) & set(artifact_roles):
        _stop("verifier source overlap")
    for row in sealed["input_hashes"]:
        if set(row) != {"role", "sha256"} or not _pin(row["sha256"]):
            _stop("sealed input record")
    for row in sealed["artifact_hashes"]:
        if set(row) != {"role", "path", "sha256", "size"} or not _pin(row["sha256"]) or not isinstance(row["size"], int) or row["size"] < 0:
            _stop("sealed artifact record")
    if sealed["complete_artifact_set_sha256"] != hashlib.sha256(canonical_bytes(sealed["artifact_hashes"])).hexdigest():
        _stop("complete artifact digest")
    if before_nested_path is not None:
        for role in input_roles + artifact_roles:
            before_nested_path(role)
    return {
        "manifest_role_count": len(rows),
        "groups": {"common": list(COMMON_ROLES), "sealed_wrapper": list(wrapper), "input_hashes": input_roles, "artifact_hashes": artifact_roles},
        "complete_artifact_set_roles": artifact_roles,
    }


def synthetic_operation(stage, suffix):
    return {
        "stage": stage,
        "operation_id": stage + "-" + suffix,
        "exclusive_output_root": "/server-only/p4h/" + stage + "-" + suffix + "-output",
        "exclusive_control_root": "/server-only/p4h/" + stage + "-" + suffix + "-control",
        "controller_identity": hashlib.sha256(("controller:" + stage + suffix).encode()).hexdigest(),
        "admission_sha256": hashlib.sha256(("admission:" + stage + suffix).encode()).hexdigest(),
        "g_receipt_sha256": hashlib.sha256(("g:" + stage + suffix).encode()).hexdigest(),
        "worker_pgid": 1000 + sum(stage.encode()) + len(suffix),
        "state": "CLOSED" if stage in ("pilot", "formal-train", "formal-resume", "infer-route") else "PREPARED",
    }


def validate_operation_pair(mutating, verifier):
    pairs = {"pilot": "pilot-verify", "formal-train": "formal-verify", "formal-resume": "formal-verify", "infer-route": "infer-route-verify"}
    if mutating.get("stage") not in pairs or verifier.get("stage") != pairs[mutating["stage"]] or mutating.get("state") != "CLOSED":
        raise ValueError("operation order/state")
    fields = ("operation_id", "exclusive_output_root", "exclusive_control_root", "controller_identity", "admission_sha256", "g_receipt_sha256", "worker_pgid")
    if any(mutating.get(field) == verifier.get(field) for field in fields):
        raise ValueError("mutating/verifier identity reuse")
    for role in (mutating, verifier):
        if not role["exclusive_output_root"].startswith("/") or not role["exclusive_control_root"].startswith("/"):
            raise ValueError("absolute roots")
    return {"independent_operations": True, "predecessor_read_only": True}


def synthetic_feed_descriptor():
    return {
        "schema_version": 1,
        "feed_root": "/server-only/p4h/g-feed",
        "feed_root_identity": {"device": 1, "inode": 2},
        "feed_owner_uid": 1000,
        "feed_mode": "0700",
        "head_name": "HEAD.json",
        "receipt_name_pattern": "receipt-<zero-padded-seq>-<sha256>.json",
        "publisher_lock": {"name": "publisher.lock", "device": 1, "inode": 3, "birth_ctime_ns": 4, "owner_uid": 1000, "creation": "O_CREAT|O_EXCL"},
        "publisher_process": {"pid": 10, "start_ticks": 20, "executable_sha256": "a" * 64, "start_receipt_sha256": "b" * 64},
        "initial_no_other_writer_inventory_sha256": "c" * 64,
        "g_job_identity": {"job_id_hash": "d" * 64, "start_identity_hash": "e" * 64},
        "receipt_schema_id": "p4h-g-feed-receipt-v2",
        "initial_seq": 0,
        "initial_receipt_sha256": "f" * 64,
        "allowed_statuses": ["OPEN", "CLOSED", "OWNER_LOST"],
    }


def validate_feed_descriptor(value):
    if not isinstance(value, dict) or set(value) != FEED_FIELDS or value["schema_version"] != 1 or not value["feed_root"].startswith("/") or value["feed_mode"] != "0700" or value["head_name"] != "HEAD.json" or value["receipt_name_pattern"] != "receipt-<zero-padded-seq>-<sha256>.json" or value["receipt_schema_id"] != "p4h-g-feed-receipt-v2" or value["initial_seq"] != 0 or value["allowed_statuses"] != ["OPEN", "CLOSED", "OWNER_LOST"]:
        raise ValueError("feed descriptor schema")
    if set(value["feed_root_identity"]) != {"device", "inode"} or set(value["publisher_lock"]) != {"name", "device", "inode", "birth_ctime_ns", "owner_uid", "creation"} or value["publisher_lock"]["creation"] != "O_CREAT|O_EXCL" or set(value["publisher_process"]) != {"pid", "start_ticks", "executable_sha256", "start_receipt_sha256"} or set(value["g_job_identity"]) != {"job_id_hash", "start_identity_hash"}:
        raise ValueError("feed nested schema")
    pins = [value["publisher_process"]["executable_sha256"], value["publisher_process"]["start_receipt_sha256"], value["initial_no_other_writer_inventory_sha256"], value["g_job_identity"]["job_id_hash"], value["g_job_identity"]["start_identity_hash"], value["initial_receipt_sha256"]]
    if not all(_pin(item) for item in pins):
        raise ValueError("feed pins")
    return {"schema_only": True, "runtime_admission": "BLOCKED"}


def frozen_config():
    return copy.deepcopy(EXPECTED_CONFIG)
