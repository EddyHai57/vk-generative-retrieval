"""Compare real batches against the preserved pre-infrastructure Python path."""

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .two_tower import TowerConfig, TwoTower, corrected_loss
from .two_tower_data import encode_batch, load_foundation
from .two_tower_prepare import load_tensor_dataset, sha256_file
from .two_tower_runtime import configure_threads, process_metrics


def _load_reference(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def maximum_difference(first, second):
    if first.shape != second.shape:
        raise AssertionError('tensor shapes differ')
    if not first.dtype.is_floating_point and not second.dtype.is_floating_point:
        if not torch.equal(first.long(), second.long()):
            raise AssertionError('integer or mask mismatch')
        return 0.
    difference = float((first.float() - second.float()).abs().max()) if first.numel() else 0.
    if not np.isfinite(difference) or difference > 1e-6:
        raise AssertionError(f'float mismatch {difference}')
    return difference


def run_equivalence(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    configure_threads(args.threads)
    config = TowerConfig(batch_size=1024, id_min_count=100, seed=42)
    reference = _load_reference(args.reference / 'src/lsvd/two_tower.py',
                                'lsvd._round30_model_reference')
    reference_data = _load_reference(args.reference / 'src/lsvd/two_tower_data.py',
                                     'lsvd._round30_data_reference')
    catalog, store, _, scope = load_foundation(
        args.foundation, args.metadata, args.vector_ids, args.vectors, config,
        output, users=args.users,
    )
    arrays, tensor_catalog, tensor_store, _ = load_tensor_dataset(args.tensors, config)
    # Both paths use the same full-training proposal, while batches use the
    # round29 hash-selected users whose full histories fit this Mac.
    store.frequency_ids = arrays['frequency_item_ids']
    store.frequency_counts = arrays['frequency_count']
    full_reference = reference_data.EventStore.__new__(reference_data.EventStore)
    full_reference.columns = tensor_store.columns
    full_reference.offsets = tensor_store.offsets
    order = np.random.default_rng(config.seed).permutation(len(store.samples))
    rows = []
    for structure in ['mean', 'causal']:
        torch.manual_seed(config.seed)
        old_model = reference.TwoTower(catalog, config, structure)
        new_model = TwoTower(tensor_catalog, config, structure)
        new_model.load_state_dict(old_model.state_dict())
        optimizers = [torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                        weight_decay=config.weight_decay)
                      for model in [old_model, new_model]]
        for step in range(args.batches):
            take_started = time.perf_counter()
            selected = order[step*config.batch_size:(step+1)*config.batch_size]
            take_seconds = time.perf_counter() - take_started
            if len(selected) != config.batch_size:
                raise ValueError('not enough selected real targets for the required batches')
            old_positions = store.samples[selected]
            old_users = store.user_rows[old_positions]
            ids = store.user_ids[old_users]
            users = np.searchsorted(tensor_store.user_ids, ids)
            positions = tensor_store.offsets[users] + old_positions - store.offsets[old_users]
            new_selected = np.searchsorted(tensor_store.samples, positions)
            np.testing.assert_array_equal(tensor_store.samples[new_selected], positions)
            old_values = store.train_batch(selected, config, 'cpu')
            feature_seconds = [0.]
            original_features = tensor_catalog.features_by_rows

            def timed_features(*values, **keywords):
                started = time.perf_counter()
                result = original_features(*values, **keywords)
                feature_seconds[0] += time.perf_counter() - started
                return result

            tensor_catalog.features_by_rows = timed_features
            started = time.perf_counter()
            new_values = tensor_store.train_batch(new_selected, config, 'cpu')
            assembly_seconds = time.perf_counter() - started
            tensor_catalog.features_by_rows = original_features
            old_batch, old_targets, old_positives, old_q, old_weights = old_values
            new_batch, new_targets, new_positives, new_q, new_weights = new_values
            # First verify every visible history field against the original
            # parquet-backed selected-user store. Its right-padding can read
            # a different next user after the other users have been omitted.
            for name, value in old_batch.items():
                valid = old_batch['valid']
                if value.ndim == 3:
                    valid = valid[..., None].expand_as(value)
                source = value if value.dtype.is_floating_point else value.long()
                candidate = new_batch[name]
                candidate = candidate if candidate.dtype.is_floating_point else candidate.long()
                maximum_difference(source[valid], candidate[valid])
            maximum_difference(old_batch['valid'], new_batch['valid'])
            old_batch = full_reference.history_batch(
                positions, users, arrays['event_week'][positions],
                config.max_history, 'cpu',
            )
            field_difference = max(maximum_difference(value, new_batch[name])
                                   for name, value in old_batch.items())
            maximum_difference(torch.as_tensor(old_targets), new_targets)
            padded = old_batch['items'].numpy().copy()
            padded[~old_batch['valid'].numpy()] = catalog.ids[0]
            expected_features = catalog.features(padded)
            expected_targets = catalog.features(old_targets)
            feature_difference = max(
                maximum_difference(expected_features[name], value)
                for name, value in new_batch['item_features'].items()
            )
            target_difference = max(
                maximum_difference(expected_targets[name], value)
                for name, value in new_batch['target_features'].items()
            )
            maximum_difference(torch.as_tensor(old_targets), new_targets)
            maximum_difference(torch.as_tensor(old_q).float(), new_q)
            maximum_difference(torch.as_tensor(old_weights), new_weights)
            columns = [int(item) for item in old_targets]
            expected_mask = torch.tensor([[item not in positive for item in columns]
                                          for positive in old_positives])
            old_target_tensor = torch.as_tensor(old_targets).long()
            expected_mask &= old_target_tensor[:, None] != old_target_tensor[None, :]
            expected_mask.fill_diagonal_(True)
            started = time.perf_counter()
            actual_mask = new_positives.allowed(new_targets)
            mask_seconds = time.perf_counter() - started
            maximum_difference(expected_mask, actual_mask)
            rng = torch.get_rng_state()
            old_users_vector = reference_data.encode_batch(old_model, catalog, old_batch, 'cpu')
            old_item_vector = old_model.item(expected_targets)
            old_loss, old_skipped = reference.corrected_loss(
                old_users_vector, old_item_vector, old_target_tensor,
                old_positives, torch.as_tensor(old_q).float(),
                torch.as_tensor(old_weights), config.temperature,
            )
            optimizers[0].zero_grad(set_to_none=True)
            old_loss.backward()
            optimizers[0].step()
            old_rng_after = torch.get_rng_state()
            torch.set_rng_state(rng)
            started = time.perf_counter()
            new_users_vector = encode_batch(new_model, tensor_catalog, new_batch, 'cpu')
            new_item_vector = new_model.item(new_batch['target_features'])
            new_loss, new_skipped = corrected_loss(
                new_users_vector, new_item_vector, new_targets.long(), actual_mask,
                new_q, new_weights, config.temperature,
            )
            forward_seconds = time.perf_counter() - started
            optimizers[1].zero_grad(set_to_none=True)
            started = time.perf_counter()
            new_loss.backward()
            backward_seconds = time.perf_counter() - started
            started = time.perf_counter()
            optimizers[1].step()
            optimizer_seconds = time.perf_counter() - started
            maximum_difference(old_rng_after, torch.get_rng_state())
            loss_difference = abs(float(old_loss.detach()) - float(new_loss.detach()))
            if loss_difference > 1e-6 or old_skipped != new_skipped:
                raise AssertionError('loss or skipped row mismatch')
            parameter_difference = max(maximum_difference(old, new) for old, new in
                                       zip(old_model.parameters(), new_model.parameters()))
            row = dict(structure=structure, batch=step, targets=len(selected),
                       field_max_difference=field_difference,
                       history_feature_max_difference=feature_difference,
                       target_feature_max_difference=target_difference,
                       mask_exact=True, loss_difference=loss_difference,
                       parameter_max_difference=parameter_difference,
                       take_batch_seconds=take_seconds,
                       history_seconds=assembly_seconds-feature_seconds[0],
                       feature_seconds=feature_seconds[0], mask_seconds=mask_seconds,
                       assembly_and_mask_seconds=assembly_seconds+mask_seconds,
                       forward_seconds=forward_seconds,
                       backward_seconds=backward_seconds,
                       optimizer_seconds=optimizer_seconds)
            rows.append(row)
            with (output / 'progress.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print('equivalence', structure, step, 'assembly_mask_ms',
                  round(row['assembly_and_mask_seconds']*1000, 3), flush=True)
        del old_model, new_model, optimizers
    with (output / 'batches.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(status='pass', batches_per_structure=args.batches,
                   targets_per_batch=1024, threads=args.threads,
                   reference_users=args.users, full_training_proposal=True,
                   visible_fields_checked_against_original_parquet=True,
                   padding_reference='original history function over the same full physical event rows',
                   counts=json.loads((args.tensors / 'manifest.json').read_text())['counts'],
                   rows=rows, resources=process_metrics(),
                   producer_sha256=sha256_file(Path(__file__)),
                   reference_model_sha256=sha256_file(args.reference / 'src/lsvd/two_tower.py'),
                   reference_data_sha256=sha256_file(args.reference / 'src/lsvd/two_tower_data.py'))
    (output / 'summary.json').open('x').write(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['reference', 'tensors', 'foundation', 'metadata', 'vector-ids',
                  'vectors', 'output']:
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--users', type=int, default=2000)
    parser.add_argument('--batches', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.batches < 20:
        parser.error('real equivalence requires at least 20 batches per structure')
    run_equivalence(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
