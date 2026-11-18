"""Two approved CPU-only negative-sampling contrasts, one epoch each."""
import argparse
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .data import file_sha256
from .negative_contrasts import negative_mask, proposal_values, contrast_loss
from .two_tower import TowerConfig, TwoTower
from .two_tower_data import load_foundation, encode_batch
from .two_tower_run import peak_rss_bytes, primary
from .two_tower_retrieval import retrieve, diagnostic_evaluations


def save_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def run_step(model, catalog, store, config, selected, mode, q_full, mass,
             uniform_rng, sampled, effective):
    batch, targets, positives, _, weights = store.train_batch(selected, config, 'cpu')
    users = encode_batch(model, catalog, batch, 'cpu')
    items = model.item(catalog.features(targets, 'cpu'))
    candidate_ids = targets
    if mode == 'mixed-uniform1024':
        uniform_rows = uniform_rng.integers(0, len(catalog.ids), size=1024)
        sampled[uniform_rows] = True
        uniform_ids = catalog.ids[uniform_rows]
        uniform_items = model.item(catalog.features(uniform_ids, 'cpu'))
        candidate_ids = np.r_[targets, uniform_ids]
        items = torch.cat([items, uniform_items])
    mask = negative_mask(candidate_ids, positives, targets)
    if mode == 'mixed-uniform1024':
        effective[uniform_rows[mask[:, len(targets):].any(0)]] = True
    user_rows = store.user_rows[store.samples[selected]]
    loss, skips = contrast_loss(users, items, q_full[catalog.rows(candidate_ids)],
                                torch.as_tensor(weights), config.temperature, mask, mode,
                                excluded_mass=mass[user_rows], catalog_size=len(catalog.ids))
    return loss, skips, len(np.unique(user_rows))


def evaluate_paths(model, catalog, store, data, output):
    records = {}
    model.eval()
    for label, cold in [('all', False), ('cold', True)]:
        predictions, search = retrieve(model, catalog, store, data, 'cpu',
                                       output / (label + '-retrieval'), cold=cold, exact_users=8)
        result, groups = diagnostic_evaluations(catalog, data, predictions, label,
                                                output / (label + '-evaluation'), 1000,
                                                save_diagnostics=True)
        records[label] = dict(main=primary(result), groups=groups, search=search)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['corrected-logq', 'mixed-uniform1024'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(4)
    reference = Path('cache/round20-batch-diag/round18-mean-epoch1-eval/mlp_mean/model_summary.json')
    config = replace(TowerConfig(**json.loads(reference.read_text())['config']), epochs=1,
                     packed_window=False, batch_size=1024, seed=42)
    old = json.loads(Path('cache/round19-two-tower-speedup/trial-cpu/effective_config.json').read_text())['arguments']
    catalog, store, data, summary = load_foundation(Path(old['foundation']), Path(old['metadata']),
        Path(old['vector_ids']), Path(old['vectors']), config, output, users=2000)
    np.testing.assert_array_equal(data.user_ids, np.load('cache/round23-batch-negatives/cohort_user_ids.npy'))
    assert len(store.samples) == 4211452 and len(catalog.ids) == 19627601
    assert store.columns['week'].max() <= 24
    frequency = dict(zip(map(int, store.frequency_ids), map(int, store.frequency_counts)))
    q_full, mass = proposal_values(catalog.ids, [store.positives[int(u)] for u in store.user_ids], frequency)
    del frequency
    gc.collect()
    snapshot = output / 'source'
    snapshot.mkdir()
    hashes = {}
    for source in Path(__file__).parent.glob('*.py'):
        shutil.copy2(source, snapshot / source.name)
        hashes[source.name] = file_sha256(source)
    save_json(output / 'frozen.json', dict(mode=args.mode, config=asdict(config), data=summary,
        source_sha256=hashes, baseline_reference_sha256=file_sha256(reference),
        design_sha256=file_sha256('cache/round24-negatives/design_before_implementation.md'),
        evaluation_sdk_sha256=file_sha256('src/lsvd/evaluation.py'),
        uniform_per_batch=1024 if args.mode == 'mixed-uniform1024' else 0,
        input_paths={key: old[key] for key in ('foundation', 'metadata', 'vector_ids', 'vectors')},
        preflight_only=args.preflight_only, week26_read=False))
    torch.manual_seed(config.seed)
    model = TwoTower(catalog, config, 'mean')
    order = np.random.default_rng(config.seed).permutation(len(store.samples))
    uniform_rng = np.random.default_rng(config.seed)
    sampled = np.zeros(len(catalog.ids), dtype=bool)
    effective = np.zeros(len(catalog.ids), dtype=bool)
    if args.preflight_only:
        model.eval()
        with torch.no_grad():
            first, skips, _ = run_step(model, catalog, store, config, order[:32], args.mode,
                                      q_full, mass, uniform_rng, sampled, effective)
            second, _, _ = run_step(model, catalog, store, config, order[:32], args.mode,
                                      q_full, mass, np.random.default_rng(config.seed), sampled, effective)
        assert torch.isfinite(first) and torch.equal(first, second)
        save_json(output / 'preflight.json', dict(loss=float(first), skipped=skips,
            repeated_seed_difference=float(abs(first - second)), optimizer_steps=0,
            actual_training_targets=len(store.samples), catalog_items=len(catalog.ids),
            seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss_bytes()))
        print('preflight verified', args.mode, flush=True)
        return 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    model.train()
    train_started = time.perf_counter()
    skips_total = 0
    losses = []
    uniform_draws = 0
    with (output / 'batches.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['optimizer_step', 'targets', 'distinct_users', 'skipped_rows', 'loss', 'elapsed_seconds'])
        for step, low in enumerate(range(0, len(order), config.batch_size), 1):
            selected = order[low:low + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, skips, users = run_step(model, catalog, store, config, selected, args.mode,
                                          q_full, mass, uniform_rng, sampled, effective)
            if not torch.isfinite(loss):
                raise ValueError(f'nonfinite loss at step {step}')
            loss.backward()
            optimizer.step()
            skips_total += skips
            losses.append(float(loss.detach()))
            uniform_draws += 1024 if args.mode == 'mixed-uniform1024' else 0
            elapsed = time.perf_counter() - train_started
            writer.writerow([step, len(selected), users, skips, losses[-1], elapsed])
            if step % 100 == 0:
                handle.flush()
                print(args.mode, 'step', step, 'targets', low + len(selected), 'seconds', round(elapsed, 1),
                      'loss', round(losses[-1], 5), flush=True)
    training_seconds = time.perf_counter() - train_started
    training_peak = peak_rss_bytes()
    torch.save(dict(model=model.state_dict(), config=asdict(config), structure='mlp_mean', epoch=1),
               output / 'epoch-1.pt')
    new = catalog.groups == 2
    contact = dict(uniform_draws=uniform_draws, sampled_unique=int(sampled.sum()),
        effective_negative_unique=int(effective.sum()),
        sampled_global_new_proxy_unique=int((sampled & new).sum()),
        effective_negative_global_new_proxy_unique=int((effective & new).sum()),
        note='sampled versus unmasked negative columns; rank proxy only, validation not consulted')
    np.save(output / 'sampled_catalog_rows.npy', np.flatnonzero(sampled))
    np.save(output / 'effective_negative_catalog_rows.npy', np.flatnonzero(effective))
    save_json(output / 'negative_contact.json', contact)
    epoch = dict(targets=len(order), effective_targets=len(order) - skips_total,
        optimizer_steps=len(losses), skipped_rows=skips_total, training_seconds=training_seconds,
        effective_targets_per_second=(len(order) - skips_total) / training_seconds,
        mean_loss=float(np.mean(losses)), training_peak_rss_bytes=training_peak)
    save_json(output / 'epoch-1.json', epoch)
    del optimizer, q_full, mass, sampled, effective, loss
    gc.collect()
    results = evaluate_paths(model, catalog, store, data, output)
    for name, expected in hashes.items():
        assert file_sha256(Path(__file__).parent / name) == expected
    save_json(output / 'summary.json', dict(mode=args.mode, config=asdict(config), data=summary,
        epoch=epoch, negative_contact=contact, results=results, seconds=time.perf_counter() - started,
        peak_process_rss_bytes=peak_rss_bytes(), week26_read=False))
    print('completed', args.mode, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
