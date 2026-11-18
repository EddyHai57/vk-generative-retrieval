"""Approved mean-tower batching follow-up with an independent time window."""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .data import file_sha256
from .two_tower import TowerConfig
from .two_tower_data import load_foundation
from .two_tower_run import fit


def replay_distributions(store, config, output):
    output.mkdir(exist_ok=False)
    for rule, cap in [('target-shuffle', None), ('chunk-shuffle-cap64', 64),
                      ('chunk-shuffle-cap8', 8), ('chunk-shuffle-cap1', 1),
                      ('remaining-targets-cap8', 8)]:
        rng = np.random.default_rng(42)
        if cap is None:
            order = rng.permutation(len(store.samples))
            batches = (order[low:low + config.batch_size]
                       for low in range(0, len(order), config.batch_size))
        else:
            case = replace(config, max_user_targets=cap,
                           balance_user_batches=rule.startswith('remaining'),
                           shuffle_all_targets=False, shuffle_balanced_batches=False)
            batches = store.packed_batches(case, rng)
        with (output / f'{rule}.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['optimizer_step', 'targets', 'distinct_users', 'target_weight_sum'])
            for step, indices in enumerate(batches, 1):
                users = store.user_rows[store.samples[indices]]
                writer.writerow([step, len(indices), len(np.unique(users)),
                                 float(store.weights[indices].sum())])
        print('replayed', rule, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cap', type=int, choices=[8, 64], default=8)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--shuffle-balanced-batches', action='store_true')
    parser.add_argument('--target-shuffle', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args(argv)
    started = time.time()
    allowance = min(7200, args.deadline - started)
    if allowance <= 600:
        raise SystemExit('insufficient time remaining')
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(int(allowance))
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {p.name:file_sha256(p) for p in Path(__file__).parent.glob('two_tower*.py')}
    source_hashes[Path(__file__).name] = file_sha256(__file__)
    reference = Path('cache/round19-two-tower-speedup/trial-cpu/effective_config.json')
    previous = json.loads(reference.read_text())
    config = replace(TowerConfig(**previous['config']),
                     max_user_targets=1024 if args.target_shuffle else args.cap,
                     epochs=1, balance_user_batches=args.balanced,
                     shuffle_balanced_batches=args.shuffle_balanced_batches,
                     shuffle_all_targets=args.target_shuffle)
    old = previous['arguments']
    torch.set_num_threads(4)
    catalog, store, data, summary = load_foundation(
        Path(old['foundation']), Path(old['metadata']), Path(old['vector_ids']),
        Path(old['vectors']), config, args.output, users=2000)
    expected = np.load('cache/round19-two-tower-speedup/trial-cpu/mlp_mean/retrieval/user_ids.npy')
    np.testing.assert_array_equal(data.user_ids, expected)
    if args.replay:
        replay_distributions(store, config, args.output/'replay')
    (args.output/'effective_config.json').write_text(json.dumps({
        'config':asdict(config),'data':summary,'source_sha256':source_hashes,
        'deadline_unix':args.deadline,'single_run_allowance_seconds':allowance,
        'reference_sha256':file_sha256(reference)},ensure_ascii=False,indent=2))
    budget = allowance - (time.time() - started) - 600
    result = fit(catalog, store, data, config, 'mlp_mean', 'cpu',
                 args.output/'mlp_mean', 0, 1000, 8, budget)
    for name, expected_sha in source_hashes.items():
        assert file_sha256(Path(__file__).parent/name) == expected_sha
    (args.output/'summary.json').write_text(json.dumps({
        'models':[result],'data':summary,'config':asdict(config),
        'seconds':time.time()-started,'code_sha256':source_hashes,
        'test_week26_read':False},ensure_ascii=False,indent=2))
    signal.alarm(0)
    print(json.dumps({'status':'complete','seconds':time.time()-started}),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
