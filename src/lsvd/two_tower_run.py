"""Executable round18 CPU/MPS trial. Every output directory must be new."""

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from .two_tower import TowerConfig, TwoTower, corrected_loss, user_hash
from .two_tower_data import load_foundation, synthetic, encode_batch, subset_evaluation
from .two_tower_retrieval import (
    retrieve,
    official_rank_predictions,
    diagnostic_evaluations,
    tie_keys,
    export_cold_embeddings,
)

STRUCTURES = {
    "mlp_mean": ("mean", 0),
    "mlp_causal": ("causal", 0),
    "dcn_causal": ("causal", 1),
}


def peak_rss_bytes():
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum if sys.platform == "darwin" else maximum * 1024


def primary(result):
    return next(
        row
        for row in result.metrics
        if row["ruler"] == "main"
        and row["place_scope"] == "0/1"
        and row["seen_scope"] == "all"
        and row["breakdown"] == "all"
        and row["k"] == 200
    )


def fit(
    catalog,
    store,
    data,
    config,
    structure,
    device,
    output,
    max_samples,
    replicates,
    exact_users,
    max_train_seconds=7200,
):
    from .evaluation import evaluate

    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    user_structure, cross_layers = STRUCTURES[structure]
    effective = replace(config, cross_layers=cross_layers)
    model = TwoTower(catalog, effective, user_structure).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    early_ids = np.array(
        [u for u in data.user_ids if user_hash(u, config.user_hash_seed) % 10 == 0]
    )
    if not len(early_ids):
        raise ValueError("fixed 10% hash sample is empty")
    early_data = subset_evaluation(data, early_ids)
    best = -float("inf")
    bad = 0
    best_path = None
    epochs = []
    samples_processed = 0
    training_seconds = 0.0
    for epoch in range(config.epochs):
        model.train()
        if config.packed_window:
            batches = store.packed_batches(config, rng, max_samples)
        else:
            order = rng.permutation(len(store.samples))
            if max_samples:
                order = order[:max_samples]
            batches = (order[start:start + config.batch_size] for start in range(0, len(order), config.batch_size))
        losses = []
        skips = 0
        processed = 0
        batch_users = []
        batch_targets = []
        timed_out = False
        started = time.perf_counter()
        for step, selected in enumerate(batches):
            if training_seconds + time.perf_counter() - started >= max_train_seconds:
                timed_out = True
                break
            batch_users.append(int(np.unique(store.user_rows[store.samples[selected]]).size))
            batch_targets.append(len(selected))
            loss, skipped, _ = training_step(model, catalog, store, config, device, optimizer, selected)
            losses.append(loss)
            skips += skipped
            processed += len(selected)
            if step % 100 == 0:
                print(
                    structure,
                    "epoch",
                    epoch + 1,
                    "样本",
                    processed,
                    "loss",
                    round(losses[-1], 4),
                    "elapsed_seconds",
                    round(time.perf_counter() - started, 2),
                    flush=True,
                )
        with (output / f"epoch-{epoch+1}-batches.csv").open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["optimizer_step", "targets", "distinct_users"])
            writer.writerows((index + 1, size, users)
                             for index, (size, users) in enumerate(zip(batch_targets, batch_users)))
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        training_seconds += elapsed
        samples_processed += processed
        optimizer.zero_grad(set_to_none=True)
        if device == "mps":
            torch.mps.empty_cache()
        # Real full-catalog early stopping. No validation rows enter training.
        predictions, search = retrieve(
            model, catalog, store, early_data, device, exact_users=0
        )
        score = primary(
            evaluate(
                early_data, predictions, model_name=structure, ks=(200,), replicates=1
            )
        )["macro_capped"]
        checkpoint = output / f"epoch-{epoch+1}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(effective),
                "structure": structure,
                "epoch": epoch + 1,
            },
            checkpoint,
        )
        row = dict(
            epoch=epoch + 1,
            mean_loss=float(np.mean(losses)),
            samples=processed,
            skipped_rows=skips,
            training_seconds=elapsed,
            samples_per_second=processed / elapsed,
            effective_targets_per_second=(processed - skips) / elapsed,
            timed_out=timed_out,
            optimizer_steps=len(losses),
            batch_distinct_users_mean=float(np.mean(batch_users)),
            batch_targets_mean=float(np.mean(batch_targets)),
            batch_distinct_users_min=int(min(batch_users)),
            batch_distinct_users_max=int(max(batch_users)),
            early_recall200=score,
            early_users=len(early_ids),
            retrieval=search,
        )
        epochs.append(row)
        (output / f"epoch-{epoch+1}.json").write_text(json.dumps(row, indent=2))
        print(structure, "epoch完成", json.dumps(row), flush=True)
        if score is not None and score > best:
            best = score
            bad = 0
            best_path = checkpoint
        else:
            bad += 1
        if timed_out or bad >= config.patience:
            break
    if best_path is None:
        raise ValueError("early-stop users have no main targets")
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)["model"]
    )
    del optimizer
    return evaluate_model(
        model,
        catalog,
        store,
        data,
        structure,
        device,
        output,
        epochs,
        best_path,
        replicates,
        exact_users,
    )


def training_step(model, catalog, store, config, device, optimizer, selected, *, timer=None):
    from .two_tower_measure import region
    from .two_tower_tensors import PositiveBatch

    batch, targets, positives, probability, weights = store.train_batch(selected, config, device)
    with region(timer, 'forward'):
        users = encode_batch(model, catalog, batch, device)
        target_features = batch.get('target_features')
        items = model.item(catalog.features(targets, device) if target_features is None
                           else target_features)
    with region(timer, 'mask'):
        if timer is not None and isinstance(positives, PositiveBatch):
            positives = positives.allowed(targets)
    with region(timer, 'loss'):
        loss, skipped = corrected_loss(
            users, items, torch.as_tensor(targets, dtype=torch.int64, device=device), positives,
            torch.as_tensor(probability, dtype=torch.float32, device=device),
            torch.as_tensor(weights, device=device), config.temperature,
        )
    with region(timer, 'optimizer'):
        optimizer.zero_grad(set_to_none=True)
    with region(timer, 'backward'):
        loss.backward()
    with region(timer, 'optimizer'):
        optimizer.step()
    return float(loss.detach()), skipped, int(batch["valid"].sum())


def benchmark(catalog, store, config, structures, device, output, steps):
    records = []
    for structure in structures:
        torch.manual_seed(config.seed)
        user_structure, cross_layers = STRUCTURES[structure]
        effective = replace(config, cross_layers=cross_layers)
        model = TwoTower(catalog, effective, user_structure).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        batches = iter(store.packed_batches(config, np.random.default_rng(config.seed)))
        training_step(model, catalog, store, config, device, optimizer, next(batches))
        if device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        targets = skips = events = completed = 0
        for selected in batches:
            _, skipped, n_events = training_step(model, catalog, store, config, device, optimizer, selected)
            targets += len(selected)
            skips += skipped
            events += n_events
            completed += 1
            if completed == steps:
                break
        if device == "mps":
            torch.mps.synchronize()
        seconds = time.perf_counter() - started
        row = dict(structure=structure, device=device, steps=completed, warmup_steps=1,
                   targets=targets, skipped_rows=skips, event_tokens=events, seconds=seconds,
                   effective_targets_per_second=(targets - skips) / seconds,
                   peak_process_rss_bytes=peak_rss_bytes())
        records.append(row)
        print(json.dumps(row), flush=True)
        del model, optimizer
        if device == "mps":
            torch.mps.empty_cache()
    (output / "benchmark.json").write_text(json.dumps(records, indent=2))
    return records


def evaluate_model(
    model,
    catalog,
    store,
    data,
    structure,
    device,
    output,
    epochs,
    best_path,
    replicates=1000,
    exact_users=8,
):
    samples_processed = sum(epoch["samples"] for epoch in epochs)
    training_seconds = sum(epoch["training_seconds"] for epoch in epochs)
    predictions, search = retrieve(
        model,
        catalog,
        store,
        data,
        device,
        output / "retrieval",
        exact_users=exact_users,
    )
    result, groups = diagnostic_evaluations(
        catalog, data, predictions, structure, output / "evaluation", replicates
    )
    cold, cold_search = retrieve(
        model,
        catalog,
        store,
        data,
        device,
        output / "cold-retrieval",
        cold=True,
        exact_users=exact_users,
    )
    cold_result, cold_groups = diagnostic_evaluations(
        catalog, data, cold, structure + "_cold", output / "cold-evaluation", replicates
    )
    summary = dict(
        structure=structure,
        config=asdict(model.config),
        epochs=epochs,
        trained_epochs=len(epochs),
        best_checkpoint=(str(best_path.relative_to(output))
                         if best_path.is_relative_to(output) else best_path.name),
        parameter_count=sum(p.numel() for p in model.parameters()),
        training_samples_per_second=samples_processed / training_seconds,
        training_seconds=training_seconds,
        samples_processed=samples_processed,
        main=primary(result),
        groups=groups,
        cold_main=primary(cold_result),
        cold_groups=cold_groups,
        retrieval=search,
        cold_retrieval=cold_search,
        peak_process_rss_bytes=peak_rss_bytes(),
    )
    (output / "model_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary


def evaluate_checkpoint(
    checkpoint, catalog, store, data, device, output, replicates=1000, exact_users=8
):
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    config = TowerConfig(**saved["config"])
    structure = saved["structure"]
    model = TwoTower(catalog, config, STRUCTURES[structure][0]).to(device)
    model.load_state_dict(saved["model"])
    epoch = json.loads(checkpoint.with_suffix(".json").read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = evaluate_model(
        model,
        catalog,
        store,
        data,
        structure,
        device,
        output,
        [epoch],
        checkpoint,
        replicates,
        exact_users,
    )
    result["checkpoint_source"] = str(checkpoint.resolve())
    (output / "checkpoint_evaluation_source.json").write_text(
        json.dumps(
            {"checkpoint": str(checkpoint.resolve()), "epoch": saved["epoch"]}, indent=2
        )
    )
    return result


def forward_only_preflight(catalog, store, data, config, structures, device, output):
    rng = np.random.default_rng(config.seed)
    selected = rng.choice(
        len(store.samples), size=min(32, len(store.samples)), replace=False
    )
    records = []
    for structure in structures:
        torch.manual_seed(config.seed)
        user_structure, cross_layers = STRUCTURES[structure]
        effective = replace(config, cross_layers=cross_layers)
        model = TwoTower(catalog, effective, user_structure).to(device)
        batch, targets, positives, probability, weights = store.train_batch(
            selected, effective, device
        )
        with torch.no_grad():
            users = encode_batch(model, catalog, batch, device)
            target_features = batch.get('target_features')
            items = model.item(catalog.features(targets, device) if target_features is None
                               else target_features)
            loss, skips = corrected_loss(
                users,
                items,
                torch.as_tensor(targets, dtype=torch.int64, device=device),
                positives,
                torch.as_tensor(probability, dtype=torch.float32, device=device),
                torch.as_tensor(weights, device=device),
                effective.temperature,
            )
            model.eval()
            fixed = encode_batch(model, catalog, batch, device)
            repeated = encode_batch(model, catalog, batch, device)
            forward_difference = float((fixed - repeated).abs().max())
            new_items = catalog.ids[catalog.columns["train_count"] == 0][:8]
            cold_items = model.item(catalog.features(new_items, device))
            finite = bool(
                torch.isfinite(loss)
                & torch.isfinite(users).all()
                & torch.isfinite(cold_items).all()
            )
        if not finite:
            raise ValueError("nonfinite preflight forward output")
        records.append(
            dict(
                structure=structure,
                probe_batch=len(selected),
                loss_finite=finite,
                skipped_rows=skips,
                cold_candidates=len(new_items),
                users_shape=list(users.shape),
                item_shape=list(items.shape),
                fixed_seed_forward_max_difference=forward_difference,
            )
        )
        del model, batch, users, items, cold_items, loss
        if device == "mps":
            torch.mps.empty_cache()
    record = dict(
        kind="forward_only_preflight",
        optimizer_steps=0,
        catalog_items=len(catalog.ids),
        train_targets=len(store.samples),
        queries=len(data.user_ids),
        config=asdict(config),
        models=records,
    )
    (output / "dry_run.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("synthetic", "trial", "tensors"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--users", type=int, default=2000 if name == "trial" else 64)
        p.add_argument(
            "--all-training-users",
            action="store_true",
            help="include every train user, including those absent from validation",
        )
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument(
            "--batch-size", type=int, default=1024 if name in ("trial", "tensors") else 128
        )
        p.add_argument(
            "--max-train-samples",
            type=int,
            default=0,
            help="engineering only: uniformly draw up to this many natural target samples per epoch; 0 means all",
        )
        p.add_argument(
            "--id-min-count", type=int, default=100 if name == "trial" else 5
        )
        p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
        p.add_argument("--replicates", type=int, default=1000)
        p.add_argument("--exact-users", type=int, default=8)
        p.add_argument("--threads", type=int)
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--use-rank", action="store_true")
        p.add_argument("--id-dropout", type=float, default=0.3)
        p.add_argument("--temperature", type=float, default=0.10)
        p.add_argument("--patience", type=int, default=3)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--packed-window", action="store_true")
        p.add_argument("--sequence-chunk", type=int, default=128)
        p.add_argument("--max-user-targets", type=int, default=64)
        p.add_argument("--causal-readout", choices=("last", "window_mean"), default="last")
        p.add_argument("--benchmark-steps", type=int, default=0)
        p.add_argument("--max-train-seconds", type=float, default=7200)
        p.add_argument("--export-best-cold", action="store_true")
        p.add_argument(
            "--structures",
            nargs="+",
            choices=list(STRUCTURES),
            default=list(STRUCTURES),
        )
        p.add_argument("--foundation", type=Path)
        p.add_argument("--metadata", type=Path)
        p.add_argument("--vector-ids", type=Path)
        p.add_argument("--vectors", type=Path)
        p.add_argument('--tensor-dataset', type=Path)
        p.add_argument('--data-placement', choices=['cpu', 'cuda'], default='cpu')
        p.add_argument('--precision', choices=['fp32', 'tf32', 'bf16'], default='fp32')
        p.add_argument('--resume', type=Path)
        p.add_argument('--checkpoint-every', type=int, default=1000)
        p.add_argument('--max-train-steps', type=int, default=0)
    args = parser.parse_args(argv)
    if args.users <= 0 or args.epochs <= 0 or args.batch_size < 2:
        parser.error("positive users/epochs and batch >=2 required")
    if args.command == "trial":
        for key in ("foundation", "metadata", "vector_ids", "vectors"):
            if getattr(args, key) is None:
                parser.error("--" + key.replace("_", "-") + " required for trial")
        report = args.foundation.parents[2] / "docs/research/foundation_2026-11.md"
        if not report.is_file():
            parser.error("round13 foundation report not delivered")
    if args.command == 'tensors' and args.tensor_dataset is None:
        parser.error('--tensor-dataset is required')
    args.output.mkdir(parents=True, exist_ok=False)
    from .two_tower_runtime import configure_threads
    configure_threads(args.threads)
    config = TowerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        id_min_count=args.id_min_count,
        use_rank=args.use_rank,
        id_dropout=args.id_dropout,
        temperature=args.temperature,
        patience=args.patience,
        seed=args.seed,
        packed_window=args.packed_window,
        sequence_chunk=args.sequence_chunk,
        max_user_targets=args.max_user_targets,
        causal_readout=args.causal_readout,
    )
    if args.command == 'tensors':
        from .two_tower_training import run_tensor_command
        return run_tensor_command(args, config)
    started = time.perf_counter()
    if args.command == "synthetic":
        catalog, store, data, data_summary = synthetic(config, user_count=args.users)
    else:
        catalog, store, data, data_summary = load_foundation(
            args.foundation,
            args.metadata,
            args.vector_ids,
            args.vectors,
            config,
            args.output,
            args.users,
            all_training_users=args.all_training_users,
        )
    (args.output / "effective_config.json").write_text(
        json.dumps(
            dict(
                config=asdict(config),
                arguments={
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                data=data_summary,
            ),
            indent=2,
        )
    )
    if args.dry_run:
        forward_only_preflight(
            catalog, store, data, config, args.structures, args.device, args.output
        )
        return 0
    if args.benchmark_steps:
        benchmark(catalog, store, config, args.structures, args.device, args.output, args.benchmark_steps)
        return 0
    from .evaluation import evaluate

    rank_predictions = official_rank_predictions(catalog, data)
    rank_result, rank_groups = diagnostic_evaluations(
        catalog,
        data,
        rank_predictions,
        "official_rank",
        args.output / "official-rank",
        args.replicates,
    )
    # Local-count global popularity, same fixed-query users and all-catalog filtering.
    from .evaluation import filter_ranked_items

    order = np.lexsort(
        (
            catalog.ids,
            tie_keys(catalog),
            -catalog.columns["train_count"].astype(np.int64),
        )
    )
    ranked = catalog.ids[order]
    global_predictions = {
        int(u): filter_ranked_items(
            ranked[: 1000 + len(data.seen_items(int(u)))],
            data.seen_items(int(u)),
            catalog.ids,
            1000,
        )
        for u in data.user_ids
    }
    global_result = evaluate(
        data,
        global_predictions,
        model_name="local_global_popularity",
        replicates=args.replicates,
        output_dir=args.output / "global-popularity",
    )
    models = []
    for structure in args.structures:
        models.append(
            fit(
                catalog,
                store,
                data,
                config,
                structure,
                args.device,
                args.output / structure,
                args.max_train_samples,
                args.replicates,
                args.exact_users,
                args.max_train_seconds,
            )
        )
    if args.export_best_cold:
        selected = max(models, key=lambda result: result["main"]["macro_capped"])
        saved = torch.load(args.output / selected["structure"] / selected["best_checkpoint"], map_location=args.device, weights_only=True)
        export_model = TwoTower(catalog, TowerConfig(**saved["config"]), STRUCTURES[selected["structure"]][0]).to(args.device)
        export_model.load_state_dict(saved["model"])
        export_cold_embeddings(export_model, catalog, args.device, args.output / "best-cold-vectors", selected["structure"])
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("two_tower*.py")
    }
    summary = dict(
        mode=args.command,
        data=data_summary,
        config=asdict(config),
        models=models,
        global_popularity=primary(global_result),
        official_rank=primary(rank_result),
        official_rank_groups=rank_groups,
        seconds=time.perf_counter() - started,
        peak_process_rss_bytes=peak_rss_bytes(),
        code_sha256=sources,
        test_week26_read=False,
        scientific_conclusion=False,
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(
        json.dumps(
            dict(status="complete", seconds=summary["seconds"], models=len(models)),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
