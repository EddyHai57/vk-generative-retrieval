"""P4H-03A finite typed science API; operation and I/O ownership live elsewhere."""
import copy
import hashlib
import json
import math
import statistics

import torch

from . import retrieval_sasrec_plan as plan
from .retrieval_training import SamplerStats, candidate_batch, train_step

SCIENCE_API = (
    "pilot_capacity", "verify_pilot", "train_slot", "resume_slot", "verify_slot",
    "select_route", "verify_completion", "summarize", "infer_route", "verify_infer_route",
)
PILOT_MEASUREMENTS = (
    "train_step", "draw_replay", "epoch_export_exact", "target_free_top80",
    "final_verify_export", "final_verify_exact", "checkpoint_write", "safe_stop_closure",
)
METRICS = ("recall@10", "recall@20", "recall@50", "ndcg@10", "ndcg@20", "ndcg@50")
PRIMARY_SLICES = ("train_like_count_1", "train_like_count_2_5", "train_like_count_6_20", "train_like_count_gt20")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pin(value):
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _emit(sink, kind, value):
    if not callable(sink):
        raise ValueError("explicit output sink required")
    sink(kind, copy.deepcopy(value))


def _emit_operation(sink, kind, value):
    if sink is None:
        return
    if not callable(sink):
        raise ValueError("operation sink must be callable")
    sink(kind, copy.deepcopy(value))


def _finite_metric(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _metric_record(value):
    if not isinstance(value, dict) or set(value) != set(METRICS) or not all(_finite_metric(item) for item in value.values()):
        raise ValueError("invalid metric record")
    return value


def _forbidden(value):
    tokens = ("target", "oov", "metric", "quality")
    if isinstance(value, dict):
        return any(any(token in str(key).lower() for token in tokens) or _forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_forbidden(item) for item in value)
    if isinstance(value, str):
        return any(token in value.lower() for token in tokens)
    return False


def pilot_capacity(measurements, target_free_queries, *, sink, operation_sink=None):
    if not isinstance(measurements, dict) or set(measurements) != {"like", "observed"}:
        raise ValueError("two pilot arms required")
    checked = {}
    for arm in ("like", "observed"):
        row = measurements[arm]
        if not isinstance(row, dict) or set(row) != {"history_source", "optimizer_steps", "measurements"} or row["history_source"] != arm or type(row["optimizer_steps"]) is not int or not 0 <= row["optimizer_steps"] <= 30 or set(row["measurements"]) != set(PILOT_MEASUREMENTS):
            raise ValueError("pilot arm contract")
        for name, measurement in row["measurements"].items():
            if not isinstance(measurement, dict) or set(measurement) != {"seconds", "bytes"} or type(measurement["seconds"]) not in (int, float) or not math.isfinite(measurement["seconds"]) or measurement["seconds"] < 0 or type(measurement["bytes"]) is not int or measurement["bytes"] < 0:
                raise ValueError("pilot measurement")
        checked[arm] = copy.deepcopy(row)
    if not isinstance(target_free_queries, list) or not target_free_queries:
        raise ValueError("target-free train queries required")
    query_ids = []
    for query in target_free_queries:
        if not isinstance(query, dict) or set(query) != {"query_id", "history"} or type(query["query_id"]) is not int or query["query_id"] < 0 or query["query_id"] in query_ids or not isinstance(query["history"], list) or not 1 <= len(query["history"]) <= 50 or any(type(item) is not int or item < 0 for item in query["history"]) or _forbidden(query):
            raise ValueError("target-free query contract")
        query_ids.append(query["query_id"])
    receipt = {
        "status": "pilot_completed+unverified",
        "arms": checked,
        "query_count": len(target_free_queries),
        "query_identity_sha256": hashlib.sha256(_canonical(query_ids)).hexdigest(),
        "validation_target_read": False,
        "quality_metrics_computed": False,
        "event_kinds": ["optimizer_step", "capacity", "resource", "checkpoint", "target_free_decode", "stop", "seal"],
    }
    _emit_operation(operation_sink, "pilot_summary", {"query_count": len(target_free_queries), "query_identity_sha256": receipt["query_identity_sha256"], "artifact_identity": "in_memory_sha256_only"})
    _emit(sink, "pilot", receipt)
    return receipt


def verify_pilot(receipt, *, operation_sink=None):
    if not isinstance(receipt, dict) or receipt.get("status") != "pilot_completed+unverified" or receipt.get("validation_target_read") is not False or receipt.get("quality_metrics_computed") is not False or set(receipt.get("arms", {})) != {"like", "observed"} or not _pin(receipt.get("query_identity_sha256")):
        raise ValueError("pilot receipt")
    allowed = {"optimizer_step", "capacity", "resource", "checkpoint", "target_free_decode", "stop", "seal"}
    if set(receipt.get("event_kinds", [])) - allowed:
        raise ValueError("pilot quality event")
    result = {"status": "pilot_completed+verified", "validation_target_read": False, "quality_metrics_computed": False}
    _emit_operation(operation_sink, "pilot_verified", {"query_identity_sha256": receipt["query_identity_sha256"], "artifact_identity": "in_memory_sha256_only"})
    return result


def _slot(slot):
    if not isinstance(slot, dict):
        raise ValueError("slot record")
    matches = [registered for registered in plan.FORMAL_SLOTS if registered == slot]
    if len(matches) != 1:
        raise ValueError("unregistered slot")
    return copy.deepcopy(slot)


def _clone_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensors(item) for item in value)
    return copy.deepcopy(value)


def _tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("model state tensor required")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _draw_digest(batch, idmap):
    if not isinstance(batch, dict) or set(batch) != {"epoch", "epoch_end", "rows"} or type(batch["epoch"]) is not int or not 1 <= batch["epoch"] <= 30 or type(batch["epoch_end"]) is not bool or not isinstance(batch["rows"], list) or not batch["rows"]:
        raise ValueError("batch contract")
    columns, labels, mask = candidate_batch(batch["rows"], idmap)
    value = {"epoch": batch["epoch"], "rows": batch["rows"], "columns": columns.tolist(), "labels": labels.tolist(), "mask": mask.tolist()}
    return hashlib.sha256(_canonical(value)).hexdigest()


def _new_selection():
    return {"epoch": 0, "best_epoch": None, "best_metric": None, "bad_epochs": 0}


def _observe(selection, evaluation):
    if not isinstance(evaluation, dict) or set(evaluation) != {"epoch", "metrics", "prediction_sha256"} or evaluation["epoch"] != selection["epoch"] + 1 or not _pin(evaluation["prediction_sha256"]):
        raise ValueError("evaluation identity")
    metrics = evaluation["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {"all"}:
        raise ValueError("evaluation metric scope")
    row = metrics["all"]
    if not isinstance(row, dict) or set(row) != {"recall@50", "ndcg@50"} or not all(_finite_metric(value) for value in row.values()):
        raise ValueError("selection metric")
    metric = (row["recall@50"], row["ndcg@50"])
    improved = selection["best_metric"] is None or metric > tuple(selection["best_metric"])
    selection["epoch"] = evaluation["epoch"]
    if improved:
        selection["best_epoch"] = evaluation["epoch"]
        selection["best_metric"] = list(metric)
        selection["bad_epochs"] = 0
    else:
        selection["bad_epochs"] += 1
    return improved


def _public_state(slot, steps, records, evaluations, selection, checkpoint_roles, complete, stop_reason):
    return {
        "slot_id": slot["slot_id"],
        "config": copy.deepcopy(slot),
        "status": "completed+unverified" if complete else "segment_complete+unverified",
        "stop_reason": stop_reason,
        "steps": steps,
        "completed_epochs": selection["epoch"],
        "step_records": copy.deepcopy(records),
        "evaluations": copy.deepcopy(evaluations),
        "selection": copy.deepcopy(selection),
        "checkpoint_roles": dict(checkpoint_roles),
        "spent": {"optimizer_steps": steps, "failed_refund": False},
        "test_gate": "closed",
    }


def _run(model, optimizer, batches, idmap, slot, *, evaluate, sink, stop_after, prior=None, operation_sink=None):
    slot = _slot(slot)
    if not isinstance(batches, list) or not batches or len(batches) > 5100 or not callable(evaluate):
        raise ValueError("finite batches/evaluator required")
    prior_epoch = 0;closed_epochs=set()
    for index, batch in enumerate(batches):
        _draw_digest(batch, idmap)
        epoch = batch["epoch"]
        if epoch < prior_epoch or epoch > prior_epoch + 1 or epoch in closed_epochs or prior_epoch and epoch > prior_epoch and prior_epoch not in closed_epochs:
            raise ValueError("batch epoch order/boundary")
        if batch["epoch_end"]:
            closed_epochs.add(epoch)
        prior_epoch = epoch
    if batches[-1]["epoch_end"] is not True:
        raise ValueError("planned batches must end at epoch boundary")
    if not isinstance(idmap, dict) or not idmap:
        raise ValueError("idmap")
    if prior is None:
        start = 0;records=[];evaluations=[];selection=_new_selection();best_state=None
    else:
        if not isinstance(prior, dict) or set(prior) != {"public", "private"} or prior["public"]["config"] != slot or prior["public"]["status"] != "segment_complete+unverified":
            raise ValueError("resume state")
        private = prior["private"]
        model.load_state_dict(_clone_tensors(private["model_state"]));optimizer.load_state_dict(_clone_tensors(private["optimizer_state"]));torch.set_rng_state(private["torch_rng"].clone())
        start=private["next_batch"];records=copy.deepcopy(prior["public"]["step_records"]);evaluations=copy.deepcopy(prior["public"]["evaluations"]);selection=copy.deepcopy(prior["public"]["selection"]);best_state=_clone_tensors(private["best_model_state"])
    limit = len(batches) if stop_after is None else stop_after
    if type(limit) is not int or not start < limit <= len(batches):
        raise ValueError("segment boundary")
    stats = SamplerStats()
    for record in records:
        stats.record(record["epoch"] - 1, record["drawn"], record["skipped"])
    processed = start;terminal_reason=None
    for index in range(start, limit):
        batch=batches[index];draw_sha=_draw_digest(batch,idmap);result=train_step(model,optimizer,batch["rows"],idmap,stats,batch["epoch"]-1,slot["temperature"])
        records.append({"step":index+1,"epoch":batch["epoch"],"draw_sha256":draw_sha,"loss":result["loss"],"drawn":result["drawn"],"skipped":result["skipped"]});processed=index+1
        _emit_operation(operation_sink,"optimizer_step",{"slot_id":slot["slot_id"],"epoch":batch["epoch"],"step":processed,"draw_sha256":draw_sha,"artifact_identity":"in_memory_record"})
        if batch["epoch_end"]:
            evaluation=evaluate(model,batch["epoch"]);improved=_observe(selection,evaluation);evaluations.append(copy.deepcopy(evaluation))
            _emit_operation(operation_sink,"epoch_validation",{"slot_id":slot["slot_id"],"epoch":evaluation["epoch"],"step":processed,"prediction_sha256":evaluation["prediction_sha256"],"artifact_identity":"in_memory_sha256_only"})
            if improved:best_state=_clone_tensors(model.state_dict())
            if selection["bad_epochs"]>=3:
                terminal_reason="patience";break
            if selection["epoch"]>=30:
                terminal_reason="max_epochs";break
    complete=terminal_reason is not None
    stop_reason=terminal_reason or "segment_limit"
    model_state=_clone_tensors(model.state_dict());checkpoint_roles={"best":_tensor_digest(best_state),"last":_tensor_digest(model_state)}
    if not complete:checkpoint_roles["preemption"]=checkpoint_roles["last"]
    public=_public_state(slot,processed,records,evaluations,selection,checkpoint_roles,complete,stop_reason)
    private={"model_state":model_state,"optimizer_state":_clone_tensors(optimizer.state_dict()),"torch_rng":torch.get_rng_state().clone(),"next_batch":processed,"best_model_state":best_state}
    state={"public":public,"private":private}
    _emit_operation(operation_sink,"checkpoint",{"slot_id":slot["slot_id"],"epoch":selection["epoch"],"step":processed,"checkpoint_roles":dict(checkpoint_roles),"artifact_identity":"in_memory_sha256_only"})
    _emit(sink,"train" if prior is None else "resume",public);return state


def train_slot(model, optimizer, batches, idmap, slot, *, evaluate, sink, stop_after=None, operation_sink=None):
    return _run(model,optimizer,batches,idmap,slot,evaluate=evaluate,sink=sink,stop_after=stop_after,operation_sink=operation_sink)


def resume_slot(prior, model, optimizer, batches, idmap, slot, *, evaluate, sink, stop_after=None, operation_sink=None):
    return _run(model,optimizer,batches,idmap,slot,evaluate=evaluate,sink=sink,stop_after=stop_after,prior=prior,operation_sink=operation_sink)


def verify_slot(state, replay_batches, idmap, *, fresh_export_exact, sink, operation_sink=None):
    if not isinstance(state, dict) or set(state) != {"public", "private"} or state["public"].get("status") != "completed+unverified" or not callable(fresh_export_exact):
        raise ValueError("completed unverified slot required")
    public=state["public"]
    if len(replay_batches)!=public["steps"] or len(public["step_records"])!=public["steps"]:
        raise ValueError("draw replay count")
    for batch,record in zip(replay_batches,public["step_records"]):
        if record["draw_sha256"]!=_draw_digest(batch,idmap) or record["epoch"]!=batch["epoch"]:
            raise ValueError("draw replay")
    selection=_new_selection()
    for evaluation in public["evaluations"]:_observe(selection,evaluation)
    if selection!=public["selection"] or selection["best_epoch"] is None or set(public["checkpoint_roles"])!={"best","last"}:
        raise ValueError("selection/checkpoint replay")
    fresh=fresh_export_exact(_clone_tensors(state["private"]["best_model_state"]),copy.deepcopy(public["config"]));expected=next(row for row in public["evaluations"] if row["epoch"]==selection["best_epoch"])
    if fresh!=expected:
        raise ValueError("fresh best export/exact mismatch")
    _emit_operation(operation_sink,"fresh_export_exact",{"slot_id":public["slot_id"],"epoch":selection["best_epoch"],"step":public["steps"],"checkpoint_sha256":public["checkpoint_roles"]["best"],"prediction_sha256":fresh["prediction_sha256"],"artifact_identity":"in_memory_sha256_only"})
    result={"status":"completed+verified","slot_id":public["slot_id"],"fresh_best_export_exact_calls":1,"optimizer_steps_during_verify":0,"best_checkpoint_sha256":public["checkpoint_roles"]["best"],"test_gate":"closed"};_emit(sink,"verify",result);return result


def _terminal_map(records):
    if not isinstance(records,list) or len(records)!=12:raise ValueError("12 terminal identities required")
    names=[record.get("slot_id") for record in records if isinstance(record,dict)]
    if len(names)!=12 or len(set(names))!=12 or set(names)!={slot["slot_id"] for slot in plan.FORMAL_SLOTS}:raise ValueError("missing/duplicate registered slot")
    result={}
    for record in records:
        registered=next(slot for slot in plan.FORMAL_SLOTS if slot["slot_id"]==record["slot_id"])
        if record.get("config")!=registered or record.get("status") not in ("completed+verified","failed+spent") or not isinstance(record.get("spent"),dict):raise ValueError("terminal record")
        if record["status"]=="failed+spent" and (not isinstance(record.get("failure_reason"),str) or not record["failure_reason"]):raise ValueError("failed terminal reason")
        if record["status"]=="completed+verified":
            metrics = record.get("metrics")
            if not _pin(record.get("checkpoint_sha256")) or not isinstance(metrics,dict) or "all" not in metrics:
                raise ValueError("completed terminal evidence")
            for scope, metric in metrics.items():
                if not isinstance(scope, str) or not scope:
                    raise ValueError("metric scope")
                _metric_record(metric)
        result[record["slot_id"]]=record
    return result


def verify_completion(records, *, sink):
    terminal=_terminal_map(records);failed=sorted(name for name,row in terminal.items() if row["status"]=="failed+spent")
    terminal_by_slot={name:{"status":row["status"],"spent":row["status"]=="failed+spent","failure_reason":row.get("failure_reason")} for name,row in sorted(terminal.items())}
    result={"status":"incomplete/negative" if failed else "complete","registered_identities":12,"failed_slots":failed,"terminal_by_slot":terminal_by_slot,"eligibility":not failed,"route_status":"NOT_ELIGIBLE" if failed else "PENDING_SELECTION","replacement_slots":0,"refund_allowed":False};_emit(sink,"completion",result);return result


def select_route(records, mean_controls, *, sink):
    completion=verify_completion(records,sink=lambda *_:None)
    if not completion["eligibility"]:
        result={"eligible":False,"route_status":"NOT_ELIGIBLE","failed_slots":completion["failed_slots"]};_emit(sink,"selection",result);return result
    terminal=_terminal_map(records);like=terminal["P4H-S02"];observed=terminal["P4H-S06"]
    like_key=(like["metrics"]["all"]["recall@50"],like["metrics"]["all"]["ndcg@50"]);observed_key=(observed["metrics"]["all"]["recall@50"],observed["metrics"]["all"]["ndcg@50"])
    chosen=observed if observed_key>like_key else like;history=chosen["config"]["history_source"]
    confirmation_ids={"like":{2024:"P4H-C01",2025:"P4H-C02"},"observed":{2024:"P4H-C03",2025:"P4H-C04"}}[history]
    if not isinstance(mean_controls,dict) or set(mean_controls)!={"like","observed"}:raise ValueError("mean controls")
    confirmation={}
    for seed,slot_id in confirmation_ids.items():
        control=mean_controls.get(history,{}).get(seed);_metric_record(control)
        confirmation[seed]=terminal[slot_id]["metrics"]["all"]["recall@50"]-control["recall@50"]>0
    result={"slot_id":chosen["slot_id"],"history_source":history,"checkpoint_sha256":chosen["checkpoint_sha256"],"selection_scope":"all_eligible","selection_conditioned":True,"confirmation":confirmation,"eligible":all(confirmation.values()),"route_status":"PENDING_VALIDATION_ROUTE" if all(confirmation.values()) else "NOT_ELIGIBLE"};_emit(sink,"selection",result);return result


def _matched(terminal,history,seed):
    mapping={("like",42):"P4H-S02",("observed",42):"P4H-S06",("like",2024):"P4H-C01",("like",2025):"P4H-C02",("observed",2024):"P4H-C03",("observed",2025):"P4H-C04"}
    return terminal[mapping[(history,seed)]]["metrics"]["all"]


def summarize(records, mean_controls, slices, *, sink):
    completion=verify_completion(records,sink=lambda *_:None);terminal=_terminal_map(records)
    if set(slices)!={*PRIMARY_SLICES}:raise ValueError("primary slice set")
    for name,row in slices.items():
        if not isinstance(row,dict) or set(row)!={"members"} or set(row["members"])!={"like_mean","observed_mean","like_sasrec","observed_sasrec"} or len(set(row["members"].values()))!=1 or not all(_pin(value) for value in row["members"].values()):raise ValueError("primary slice member identity")
    if not completion["eligibility"]:
        failed=completion["failed_slots"];reason="undefined_due_to_failed_spent:"+",".join(failed)
        missing={seed:{metric:{name:reason for name in ("like_encoder_effect","observed_encoder_effect","mean_history_source_effect","sasrec_history_source_effect","interaction")} for metric in METRICS} for seed in (42,2024,2025)}
        result={"status":"incomplete/negative","route_status":"NOT_ELIGIBLE","infer_allowed":False,"failed_slots":failed,"negative_ledger":completion["terminal_by_slot"],"missing_estimand_cells":missing,"estimands":"undefined_due_to_failed_spent","selection_conditioned":True,"eligibility":False};_emit(sink,"summary",result);return result
    if set(mean_controls)!={"like","observed"}:raise ValueError("mean controls")
    estimands={}
    for seed in (42,2024,2025):
        like_mean=mean_controls["like"].get(seed);observed_mean=mean_controls["observed"].get(seed);_metric_record(like_mean);_metric_record(observed_mean)
        like_sasrec=_matched(terminal,"like",seed);observed_sasrec=_matched(terminal,"observed",seed);rows={}
        for metric in METRICS:
            like_effect=like_sasrec[metric]-like_mean[metric];observed_effect=observed_sasrec[metric]-observed_mean[metric]
            rows[metric]={"like_encoder_effect":like_effect,"observed_encoder_effect":observed_effect,"mean_history_source_effect":observed_mean[metric]-like_mean[metric],"sasrec_history_source_effect":observed_sasrec[metric]-like_sasrec[metric],"interaction":observed_effect-like_effect}
        estimands[seed]=rows
    aggregate={}
    for metric in METRICS:
        aggregate[metric]={}
        for name in ("like_encoder_effect","observed_encoder_effect","mean_history_source_effect","sasrec_history_source_effect","interaction"):
            values=[estimands[seed][metric][name] for seed in (42,2024,2025)];aggregate[metric][name]={"values":values,"mean":statistics.mean(values),"sample_std":statistics.stdev(values)}
    result={"status":"complete","estimands":estimands,"aggregate":aggregate,"selection_conditioned":True,"significance_claim":False,"primary_slice_member_sha256":{name:next(iter(row["members"].values())) for name,row in slices.items()},"eligibility":True};_emit(sink,"summary",result);return result


def _query_contract(queries,catalog,idmap,popularity):
    if not isinstance(catalog,list) or len(catalog)<80 or catalog!=sorted(set(catalog)) or any(type(item)is not int or item<0 for item in catalog):raise ValueError("frozen catalog must contain at least 80 unique items")
    if idmap!={item:index+1 for index,item in enumerate(catalog)} or not isinstance(popularity,list) or len(popularity)!=len(catalog) or len(set(popularity))!=len(catalog) or set(popularity)!=set(catalog):raise ValueError("idmap/popularity identity")
    if not isinstance(queries,list) or not queries:raise ValueError("queries")
    ids=[]
    for query in queries:
        if not isinstance(query,dict) or set(query)!={"query_id","user_id","history","history_positions"} or type(query["query_id"])is not int or query["query_id"] in ids or type(query["user_id"])is not int or not isinstance(query["history"],list) or not isinstance(query["history_positions"],list) or len(query["history"])!=len(query["history_positions"]) or len(query["history"])>50 or any(item not in idmap for item in query["history"]) or any(type(position)is not int for position in query["history_positions"]) or any(left>=right for left,right in zip(query["history_positions"],query["history_positions"][1:])):raise ValueError("target-free query identity")
        ids.append(query["query_id"])
    return ids


def _route_identity(model, catalog, idmap, queries, popularity, shard_size):
    return hashlib.sha256(_canonical({"model":_tensor_digest(model.state_dict()),"catalog":catalog,"idmap":sorted(idmap.items()),"queries":queries,"popularity":popularity,"shard_size":shard_size,"top_k":80,"tie":"item_id_ascending"})).hexdigest()


def _check_shards(shards,identity,query_ids,catalog,allow_partial):
    if not isinstance(shards,list):raise ValueError("shards")
    records=[]
    for index,shard in enumerate(shards):
        if not isinstance(shard,dict) or set(shard)!={"index","identity_sha256","start","end","records"} or shard["index"]!=index or shard["identity_sha256"]!=identity or shard["start"]!=len(records) or shard["end"]!=len(records)+len(shard["records"]):raise ValueError("shard identity/order")
        for record in shard["records"]:
            expected=query_ids[len(records)] if len(records)<len(query_ids) else None
            if set(record)!={"query_id","candidates","fallback"} or record["query_id"]!=expected or len(record["candidates"])!=80 or len(set(record["candidates"]))!=80 or any(item not in catalog for item in record["candidates"]) or type(record["fallback"])is not bool:raise ValueError("strict top80 record")
            records.append(record)
    if len(records)>len(query_ids) or not allow_partial and len(records)!=len(query_ids):raise ValueError("query coverage")
    return records


def infer_route(model,catalog,idmap,queries,popularity,*,sink,shard_size=128,resume=None,max_shards=None,operation_sink=None):
    if type(shard_size)is not int or shard_size<1 or max_shards is not None and (type(max_shards)is not int or max_shards<1):raise ValueError("shard budget")
    query_ids=_query_contract(queries,catalog,idmap,popularity);identity=_route_identity(model,catalog,idmap,queries,popularity,shard_size)
    if resume is not None and not isinstance(resume, dict):
        raise ValueError("resume record")
    shards=[] if resume is None else copy.deepcopy(resume.get("shards"))
    records=_check_shards(shards,identity,query_ids,catalog,True)
    if (resume is not None and resume.get("identity_sha256")!=identity) or (len(records)%shard_size and len(records)!=len(query_ids)):
        raise ValueError("resume identity/partial shard")
    was_training=model.training;model.eval();device=next(model.parameters()).device
    with torch.no_grad():
        item_indices=torch.tensor([idmap[item] for item in catalog],device=device);item_vectors=torch.nn.functional.normalize(model.items(item_indices),dim=1)
        new_shards=0
        for start in range(len(records),len(queries),shard_size):
            rows=[]
            for query in queries[start:start+shard_size]:
                if not query["history"]:candidates=popularity[:80];fallback=True
                else:
                    history=torch.tensor([[idmap[item] for item in query["history"]]],device=device);user=torch.nn.functional.normalize(model.users(history),dim=1)[0];scores=(item_vectors@user).detach().cpu().tolist();candidates=sorted(catalog,key=lambda item:(-scores[idmap[item]-1],item))[:80];fallback=False
                if len(candidates)!=80 or len(set(candidates))!=80:raise ValueError("strict top80 generation")
                rows.append({"query_id":query["query_id"],"candidates":candidates,"fallback":fallback})
            shard={"index":len(shards),"identity_sha256":identity,"start":start,"end":start+len(rows),"records":rows};shards.append(shard);records.extend(rows);new_shards+=1
            _emit_operation(operation_sink,"route_shard",{"shard":shard["index"],"range":[shard["start"],shard["end"]],"query_ids":[row["query_id"] for row in rows],"record_count":len(rows),"records_sha256":hashlib.sha256(_canonical(rows)).hexdigest(),"route_identity_sha256":identity,"artifact_identity":"in_memory_sha256_only","records":copy.deepcopy(rows)})
            if max_shards is not None and new_shards>=max_shards:break
    model.train(was_training)
    complete=len(records)==len(queries);result={"status":"complete" if complete else "partial","identity_sha256":identity,"top_k":80,"tie_break":"item_id_ascending","query_ids":query_ids,"shards":shards,"quality_metrics_computed":False,"test_gate":"closed"};_emit(sink,"infer_route",result);return result


def verify_infer_route(result,queries,catalog,*,sink,operation_sink=None):
    if not isinstance(result,dict) or result.get("status")!="complete" or result.get("top_k")!=80 or result.get("tie_break")!="item_id_ascending" or result.get("quality_metrics_computed")is not False or result.get("test_gate")!="closed":raise ValueError("complete route required")
    query_ids=[query.get("query_id") for query in queries];records=_check_shards(result.get("shards"),result.get("identity_sha256"),query_ids,catalog,False)
    for query,record in zip(queries,records):
        if record["fallback"]!=(not query["history"]):raise ValueError("fallback identity")
    verified={"status":"P4H_ROUTE_VALIDATION_READY","query_count":len(records),"top_k":80,"identity_sha256":result["identity_sha256"],"quality_metrics_computed":False,"test_gate":"closed"}
    _emit_operation(operation_sink,"route_verified",{"query_count":len(records),"route_identity_sha256":result["identity_sha256"],"artifact_identity":"in_memory_sha256_only"})
    _emit(sink,"verify_infer_route",verified);return verified
