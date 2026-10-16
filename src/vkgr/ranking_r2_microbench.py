"""Read-only train-input CLI; random A1/T3 weights, beam10 b1, no checkpoint I/O.

Run only after owner review, pinned deployment, resource admission and preflight.
Stdout is the only output; no predictions, weights or training rows are persisted.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from . import generative_data as gd
from . import history_v2 as hv
from . import ranking_r2 as r

A1_T3_CONFIG_SHA256 = '9a938cf3ac722e0ad11fd8136d5a037b0f44516cf53e4df85a8188770dca737e'
SCIENTIFIC_BASE_COMMIT = 'fdc988443672dd729a81dde6c0e7ca42a448cd13'


def _read(path, sha):
    return gd._json(gd._checked_file(Path(path), sha))


def load_training(prepared_path, prepared_sha, history_path, history_sha):
    """Read the pinned P4B train examples/catalog and only train-l50 history.

    Unlike existing context builders this function never opens validation, statistics,
    raw holdout partitions or checkpoints. Complete past_likes is checked BEFORE the
    exposure-history overlay, using the frozen sequential-prefix rule.
    """
    prepared_path, history_path = Path(prepared_path), Path(history_path)
    prepared = _read(prepared_path, prepared_sha)
    if prepared_path.name != 'prepare-manifest.json' or any(prepared.get(k) != v for k, v in
        {'schema_version': 1, 'task_id': 'P4B', 'kind': 'prepare', 'status': 'success'}.items()):
        raise ValueError('pinned P4B prepared manifest required')
    artifacts = prepared['artifacts']
    catalog = _read(prepared_path.parent / 'catalog.json', artifacts['catalog.json'])
    from .retrieval_data import catalog_map
    idmap = catalog_map(catalog)
    examples_path = gd._checked_file(prepared_path.parent / 'examples.jsonl', artifacts['examples.jsonl'])
    previous = {}
    previous_target = None
    rows = []
    with examples_path.open('rb') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line, object_pairs_hook=gd._unique_pairs, parse_constant=gd._reject_constant)
            gd._strict_example(row, idmap, index)
            if previous_target is not None and row['target_position'] <= previous_target:
                raise ValueError('global training prefix ordering conflict')
            prior = previous.get(row['user_id'])
            if prior is None:
                valid = len(row['history']) == 1 and row['past_likes'] == sorted(set(row['history']))
            else:
                valid = (row['history'] == (prior['history'] + [prior['target']])[-50:]
                         and row['history_positions'] == (prior['history_positions'] + [prior['target_position']])[-50:]
                         and row['past_likes'] == sorted(set(prior['past_likes'] + [prior['target']])))
            if not valid:
                raise ValueError('full target-prefix like evidence mismatch')
            rows.append(row)
            previous[row['user_id']] = row
            previous_target = row['target_position']
    if not rows or len(rows) != prepared['examples'] or len(catalog) != prepared['catalog_items']:
        raise ValueError('training example/catalog denominator mismatch')
    history = _read(history_path, history_sha)
    if any(history.get(k) != v for k, v in {'kind': 'history_v2_prepare', 'history_version': 'HIST_V2_OBS',
        'lengths': list(hv.LENGTHS), 'test_gate': 'closed_hash_only', 'training_examples': len(rows)}.items()):
        raise ValueError('pinned training history manifest required')
    sidecar = _read(history_path.parent / 'train-l50.json', history['artifacts']['train-l50.json'])
    return hv.overlay_examples(rows, sidecar, 50, set(catalog)), catalog


def execution_commit():
    """Bind the receipt to the checkout containing this module, independent of cwd."""
    return subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'history', 'sid', 'content'):
        parser.add_argument('--' + name, required=True, type=Path)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--config', required=True, type=Path, help='unchanged seed42 A1/T3 config; bytes checked against scientific base')
    parser.add_argument('--queries', type=int, default=128, help='timed prefix of the fixed 20000 train sample; 1..20000')
    args = parser.parse_args(argv)
    if not 1 <= args.queries <= 20000:
        parser.error('--queries must be 1..20000')
    config = _read(args.config, A1_T3_CONFIG_SHA256)
    from . import generative_g6_run as g6
    from .generative_g6_model import make_model_arch_d
    from .sid_depth import SIDTokensLD
    import torch
    g6.validate_config(config, engineering=False)
    # CUDA is mandatory for any real-input timing claim; unit tests exercise CPU separately.
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for real-shape engineering microbenchmark')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows, catalog = load_training(args.prepared, args.prepared_sha256, args.history, args.history_sha256)
    sampled = r.sample_training_indices(len(rows))
    sid = g6._load_sid_v2(args.sid, args.sid_sha256, args.content, args.content_sha256, catalog, False)
    if sid.info['sid_version'] != config['sid'] or sid.codec.depth != 3 or sid.codec.catalog.branching != 256:
        raise ValueError('formal T3 B256D3 bundle required')
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    model = make_model_arch_d(sid.codec, config, device='cuda')
    result = r.benchmark(model, SIDTokensLD(sid.codec, 50), [rows[i] for i in sampled[:args.queries]])
    result.update({'execution_commit': execution_commit(), 'scientific_base_commit': SCIENTIFIC_BASE_COMMIT, 'config_sha256': A1_T3_CONFIG_SHA256,
                   'input_sha256': {k: getattr(args, k + '_sha256') for k in ('prepared', 'history', 'sid', 'content')},
                   'sample_seed': 0, 'sample_count': 20000, 'sample_algorithm': 'python-random-sample-without-replacement',
                   'sample_indices_sha256': hashlib.sha256(json.dumps(sampled, separators=(',', ':')).encode()).hexdigest(),
                   'model_seed': config['seed'], 'model_shape': model.spec, 'torch_version': torch.__version__,
                   'cuda_version': torch.version.cuda, 'gpu_model': torch.cuda.get_device_name(0),
                   'cpu_threads': 1, 'timing_includes': 'decode and negative filtering; excludes input loading, random model init and one-query warmup'})
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
