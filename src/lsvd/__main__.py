"""Command-line entry points for shared VK-LSVD preparation and evaluation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


def _json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _dataset(root):
    root = Path(root)
    return root / 'dataset' if (root / 'dataset').exists() else root


def _build(args):
    from .data import build_foundation
    weeks = tuple(int(value) for value in args.train_weeks.split(','))
    result = build_foundation(args.raw_root, args.frozen_root, args.output,
                              train_weeks=weeks, memory_limit=args.memory_limit,
                              threads=args.threads, reuse_master=args.reuse_master)
    print(json.dumps({'status': result['status'], 'train_rows': result['train_rows'],
                      'validation_rows': result['validation_rows']}), flush=True)


def _baselines(args):
    from . import baselines
    from .data import file_sha256
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    data = baselines.BaselineData.load(args.foundation)
    sources = []
    details = {}
    if args.model == 'global_popularity':
        predictions = baselines.global_popularity(data)
    elif args.model == 'last_author_popularity':
        predictions = baselines.last_author_popularity(data)
    elif args.model == 'covisitation':
        if args.graph is None:
            raise ValueError('covisitation requires --graph')
        audit = baselines.audit_reused_graph(data, args.foundation, args.graph)
        _json(args.output / 'graph_audit.json', audit)
        import duckdb
        with duckdb.connect(config={'memory_limit': '2GB', 'threads': 2}) as connection:
            graph = connection.execute('SELECT * FROM read_parquet(?)', [str(args.graph)]).fetchnumpy()
        predictions = baselines.covisitation_neighbors(data, graph)
        sources.append(baselines.source_binding(args.graph))
    else:
        import faiss
        faiss.omp_set_num_threads(args.threads)
        vector_ids = np.load(args.vector_ids, mmap_mode='r', allow_pickle=False)
        if not np.array_equal(vector_ids, data.items['item_id']):
            raise ValueError('vector catalog does not match complete metadata catalog')
        vectors = np.load(args.vectors, mmap_mode='r', allow_pickle=False)
        profiles, positive_counts = baselines.content_profiles(
            data.queries['user_id'], baselines.sequence_rows(args.foundation),
            vector_ids, vectors, recent=200)
        vectors._mmap.close()
        details = {'profile_sha256': hashlib.sha256(profiles.tobytes()).hexdigest(),
                   'positive_counts_sha256': hashlib.sha256(positive_counts.tobytes()).hexdigest(),
                   'zero_profiles': int((positive_counts == 0).sum()),
                   'history': 'latest200_all_place_exposures_then_stored_main_positive',
                   'nprobe': 64, 'initial_candidates': 4096, 'max_candidates': 131072}
        print(json.dumps({'stage': 'content_profiles', 'seconds': time.monotonic() - start,
                          'zero_profiles': details['zero_profiles']}), flush=True)
        centroids = np.load(args.centroids, allow_pickle=False)
        index = baselines.index_from_centroids(centroids, args.vectors, nprobe=64)
        print(json.dumps({'stage': 'full_catalog_index', 'items': index.ntotal,
                          'seconds': time.monotonic() - start}), flush=True)
        predictions = baselines.retrieve_content(data, profiles, index, vector_ids)
        del index, profiles
        vector_ids._mmap.close()
        gc.collect()
        sources.extend(baselines.source_binding(path) for path in
                       [args.vector_ids, args.vectors, args.centroids])
    np.save(args.output / 'predictions.npy', predictions, allow_pickle=False)
    np.save(args.output / 'user_ids.npy', data.queries['user_id'], allow_pickle=False)
    lengths = (predictions != baselines.PADDING).sum(axis=1)
    manifest = {'status': 'complete', 'model_name': args.model,
                'candidate_preparation': baselines.CANDIDATE_PREPARATION,
                'queries': len(predictions), 'catalog_items': len(data.items['item_id']),
                'empty_users': int((lengths == 0).sum()), 'mean_returned': float(lengths.mean()),
                'foundation_sha256': file_sha256(_dataset(args.foundation) / 'foundation.json'),
                'prediction_sha256': file_sha256(args.output / 'predictions.npy'),
                'code_sha256': file_sha256(Path(baselines.__file__)),
                'sources': sources, 'details': details,
                'seconds': time.monotonic() - start,
                'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'test_week26_read': False, 'trained_retrieval_model': False}
    _json(args.output / 'manifest.json', manifest)
    print(json.dumps({'status': 'complete', 'model': args.model,
                      'seconds': manifest['seconds'], 'empty_users': manifest['empty_users']}), flush=True)


def _evaluate(args):
    from .data import file_sha256
    from .evaluation import EvaluationData, evaluate
    data = EvaluationData.load(args.foundation)
    path = args.predictions
    preparation = 'unspecified'
    if path.is_dir():
        source = json.loads((path / 'manifest.json').read_text())
        if source['foundation_sha256'] != file_sha256(_dataset(args.foundation) / 'foundation.json'):
            raise ValueError('predictions belong to another foundation')
        users = np.load(path / 'user_ids.npy', allow_pickle=False)
        if not np.array_equal(users, data.user_ids):
            raise ValueError('prediction query users do not align')
        preparation = source['candidate_preparation']
        path = path / 'predictions.npy'
    predictions = np.load(path, mmap_mode='r', allow_pickle=False)
    result = evaluate(data, predictions, model_name=args.model_name,
                      replicates=args.replicates, seed=args.seed)
    result.metadata['candidate_preparation'] = preparation
    result.metadata['foundation_sha256'] = file_sha256(_dataset(args.foundation) / 'foundation.json')
    result.metadata['prediction_sha256'] = file_sha256(path)
    result.save(args.output)
    print(json.dumps({'status': 'complete', 'model': args.model_name,
                      'metric_rows': len(result.metrics), 'query_users': len(data.user_ids)}), flush=True)


def _paired(args):
    from .evaluation import EvaluationResult, paired_bootstrap
    first = EvaluationResult.load(args.first)
    second = EvaluationResult.load(args.second)
    result = paired_bootstrap(first, second, replicates=args.replicates, seed=args.seed)
    result.save(args.output)
    print(json.dumps({'status': 'complete', 'paired_metric_rows': len(result.metrics)}), flush=True)


def _verify(args):
    from .verification import verify_foundation
    result = verify_foundation(_dataset(args.foundation), args.output,
                               legacy_context=args.legacy_context,
                               memory_limit=args.memory_limit)
    print(json.dumps({'status': result['status'], 'raw_rows_checked': result['raw_rows_checked'],
                      'legacy_main_differences': result['legacy_main_differences']}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m lsvd')
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--raw-root', type=Path, required=True)
    build.add_argument('--frozen-root', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--train-weeks', default=','.join(map(str, range(25))))
    build.add_argument('--memory-limit', default='8GB')
    build.add_argument('--threads', type=int, default=4)
    build.add_argument('--reuse-master', type=Path)
    build.set_defaults(function=_build)
    baseline = commands.add_parser('baselines')
    baseline.add_argument('--foundation', type=Path, required=True)
    baseline.add_argument('--model', choices=['global_popularity', 'last_author_popularity', 'covisitation', 'content_mean'], required=True)
    baseline.add_argument('--output', type=Path, required=True)
    baseline.add_argument('--graph', type=Path)
    baseline.add_argument('--vector-ids', type=Path, default=Path('cache/round9-sid/vectors/item_id.npy'))
    baseline.add_argument('--vectors', type=Path, default=Path('cache/round9-sid/vectors/embedding.npy'))
    baseline.add_argument('--centroids', type=Path, default=Path('cache/round9-sid/quality/ann_exact_audit/centroids.npy'))
    baseline.add_argument('--threads', type=int, default=3)
    baseline.set_defaults(function=_baselines)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--foundation', type=Path, required=True)
    evaluation.add_argument('--predictions', type=Path, required=True)
    evaluation.add_argument('--model-name', required=True)
    evaluation.add_argument('--output', type=Path, required=True)
    evaluation.add_argument('--replicates', type=int, default=1000)
    evaluation.add_argument('--seed', type=int, default=2026111301)
    evaluation.set_defaults(function=_evaluate)
    paired = commands.add_parser('paired')
    paired.add_argument('--first', type=Path, required=True)
    paired.add_argument('--second', type=Path, required=True)
    paired.add_argument('--output', type=Path, required=True)
    paired.add_argument('--replicates', type=int, default=1000)
    paired.add_argument('--seed', type=int, default=2026111301)
    paired.set_defaults(function=_paired)
    verification = commands.add_parser('verify')
    verification.add_argument('--foundation', type=Path, required=True)
    verification.add_argument('--output', type=Path, required=True)
    verification.add_argument('--legacy-context', type=Path)
    verification.add_argument('--memory-limit', default='8GB')
    verification.set_defaults(function=_verify)
    args = parser.parse_args(argv)
    args.function(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
