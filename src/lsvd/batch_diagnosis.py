"""One bounded CPU smoke run using the round19 model, data and evaluator."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .two_tower import TowerConfig
from .two_tower_data import load_foundation
from .two_tower_run import evaluate_checkpoint, fit
from .data import file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cap', type=int, choices=(64, 8, 1), required=True)
    parser.add_argument('--structure', choices=('mlp_mean', 'mlp_causal'), default='mlp_mean')
    parser.add_argument('--readout', choices=('last', 'window_mean'), default='last')
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--control-diversity', action='store_true')
    parser.add_argument('--evaluate-checkpoint', type=Path)
    args = parser.parse_args(argv)
    started = time.time()
    allowance = min(7200, args.deadline - started)
    if allowance <= 600:
        raise SystemExit('insufficient remaining time for a run plus evaluation')
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(int(allowance))
    args.output.mkdir(parents=True, exist_ok=False)
    reference = Path('cache/round19-two-tower-speedup/trial-cpu/effective_config.json')
    previous = json.loads(reference.read_text())
    config = replace(TowerConfig(**previous['config']), max_user_targets=args.cap, epochs=1,
                     causal_readout=args.readout)
    torch.set_num_threads(4)
    old = previous['arguments']
    catalog, store, data, summary = load_foundation(
        Path(old['foundation']), Path(old['metadata']), Path(old['vector_ids']),
        Path(old['vectors']), config, args.output, users=2000,
    )
    expected = np.load('cache/round19-two-tower-speedup/trial-cpu/mlp_mean/retrieval/user_ids.npy')
    if not np.array_equal(data.user_ids, expected):
        raise ValueError('round19 query cohort changed')
    if args.control_diversity:
        counts, sizes = [], []
        for indices in store.packed_batches(replace(config, max_user_targets=64), np.random.default_rng(42)):
            counts.append(int(np.unique(store.user_rows[store.samples[indices]]).size))
            sizes.append(len(indices))
        (args.output / 'control64_diversity.json').write_text(json.dumps({
            'source': 'round19 cap64 seed42 sampler replay without training',
            'optimizer_steps': len(sizes), 'targets': sum(sizes),
            'batch_distinct_users_mean': float(np.mean(counts)),
            'batch_targets_mean': float(np.mean(sizes)),
        }, indent=2))
    (args.output / 'effective_config.json').write_text(json.dumps({
        'config': asdict(config), 'data': summary, 'reference_sha256': file_sha256(reference),
        'deadline_unix': args.deadline, 'run_allowance_seconds': allowance,
        'readout_history_support': 399 if args.readout == 'window_mean' else 200,
        'window_mean_status': '最多399条曝光，只作诊断，不进主线',
    }, indent=2))
    training_budget = allowance - (time.time() - started) - 600
    if args.evaluate_checkpoint is not None:
        counts, sizes = [], []
        order = np.random.default_rng(42).permutation(len(store.samples))
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start + config.batch_size]
            counts.append(int(np.unique(store.user_rows[store.samples[indices]]).size))
            sizes.append(len(indices))
        (args.output / 'round18_random_diversity.json').write_text(json.dumps({
            'source': 'round18 epoch1 flat target shuffle seed42 replay without training',
            'optimizer_steps': len(sizes), 'targets': sum(sizes),
            'batch_distinct_users_mean': float(np.mean(counts)),
            'batch_targets_mean': float(np.mean(sizes)),
        }, indent=2))
        result = evaluate_checkpoint(args.evaluate_checkpoint, catalog, store, data, 'cpu',
                                     args.output / args.structure, replicates=1000, exact_users=8)
    else:
        result = fit(catalog, store, data, config, args.structure, 'cpu',
                     args.output / args.structure, 0, 1000, 8, training_budget)
    sources = {p.name: file_sha256(p) for p in Path(__file__).parent.glob('two_tower*.py')}
    (args.output / 'summary.json').write_text(json.dumps({
        'models': [result], 'data': summary, 'config': asdict(config),
        'seconds': time.time() - started, 'code_sha256': sources,
        'test_week26_read': False,
    }, ensure_ascii=False, indent=2))
    signal.alarm(0)
    print(json.dumps({'status': 'complete', 'cap': args.cap,
                      'structure': args.structure, 'seconds': time.time() - started}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
