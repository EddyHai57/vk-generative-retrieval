"""G5 G-side assembly (read-only, CPU): the accepted G anchor run's best-epoch candidates, query metadata, slice labels and the Pop list.

Runs with the G deployment's ``vkgr`` first on PYTHONPATH (the deployment the run was trained in, unchanged). It re-reads the run's best
epoch ``evaluation-NNN`` archive through the project's own ``_read_evaluation`` (sha chain, binding, per-query legality, aggregate and
computation replay), then emits, per validation query ordinal: user_id, target, oov, history, the archived candidate items, and the
support / popularity / user-history labels computed with exactly the rule of ``generative_evaluation.aggregate_items``. A consistency gate
regroups the archived rows with those labels and requires the per-slice metrics to equal the archive's own aggregate byte for byte.
It never opens a checkpoint, never decodes, never touches a model, and has no held-out split path. Output: one NEW file.
"""
import argparse
import hashlib
import json
import resource
import subprocess
import sys
from pathlib import Path

SCHEMA = 'g5-g-assembly-v1'
POP_K = 100


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def max_rss_mib():
    """Process high-water RSS in MiB (ru_maxrss is KiB on Linux, bytes on macOS)."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / (1024 * 1024 if sys.platform == 'darwin' else 1024), 1)


def labels(stats, queries, item_groups, user_groups):
    """The classification used by ``aggregate_items`` (copied rule, vocabulary taken from the G run's own constants)."""
    popularity, users, support, history = [], [], [], []
    for query in queries:
        if query['oov']:
            popularity.append('oov'); support.append('oov')
        else:
            count = stats.item_counts[query['target']]
            popularity.append(item_groups[0] if count == 0 else item_groups[1] if count <= stats.thresholds[0] else item_groups[2] if count <= stats.thresholds[1] else item_groups[3])
            support.append('supported' if count > 0 else 'unsupported')
        count = stats.user_counts[query['user_id']]
        users.append(user_groups[0] if count == 1 else user_groups[1] if count <= 5 else user_groups[2] if count <= 20 else user_groups[3])
        length = len(query['history'])
        if not 1 <= length <= 50:
            raise ValueError('bounded eligible history required')
        history.append(str(length))
    return {'popularity': popularity, 'user_history': users, 'support': support, 'truncated_history': history}


def regroup_metrics(items, queries, catalog, assigned, names, metrics):
    empty = metrics([], [], catalog, (10, 20, 50))['all']
    result = {}
    for name in names:
        index = [i for i, label in enumerate(assigned) if label == name]
        result[name] = metrics([items[i] for i in index], [queries[i] for i in index], catalog, (10, 20, 50))['all'] if index else dict(empty)
    return result


def consistency_gate(items, queries, catalog, assigned, aggregate, item_groups, user_groups, metrics, canonical_bytes):
    spec = (('popularity', item_groups + ('oov',)), ('user_history', user_groups), ('support', ('supported', 'unsupported', 'oov')),
            ('truncated_history', tuple(str(i) for i in range(1, 51))))
    for section, names in spec:
        mine = regroup_metrics(items, queries, catalog, assigned[section], names, metrics)
        if canonical_bytes(mine) != canonical_bytes(aggregate[section]):
            raise ValueError(f'slice consistency gate failed: {section}')
    return True


def pop_ranking(stats, catalog, k=POP_K):
    """Pop = train-only item like counts, ranked by (-count, item_id); one list for every query (no validation fitting)."""
    ranking = sorted((int(i) for i in catalog), key=lambda i: (-stats.item_counts[i], i))[:k]
    if len(ranking) < k:
        raise ValueError('catalog smaller than the Pop depth')
    return ranking


def assemble_core(value, run_dir, context, tokens, read_evaluation, ev, metrics, canonical_bytes, pop_k=POP_K, digest=None):
    """Everything after the context exists. ``read_evaluation`` is the project's ``_read_evaluation(kind, run, record, context, tokens)``."""
    best = value['best']; epoch = best['epoch']
    if type(epoch) is not int or not 1 <= epoch <= len(value['epochs']):
        raise ValueError('best epoch outside the run')
    record = value['epochs'][epoch - 1]
    if record.get('epoch') != epoch or record['evaluation']['kind'] != 'query_evaluation':
        raise ValueError('best epoch must carry a query_evaluation archive')
    if best.get('sha256') != record['checkpoint_sha256']:
        raise ValueError('best checkpoint hash differs from the best epoch record')
    aggregate, computation, rows = read_evaluation(record['evaluation']['kind'], Path(run_dir), record, context, tokens)
    if canonical_bytes(aggregate) != canonical_bytes(record['metrics']) or computation != record['evaluation']['computation_sha256']:
        raise ValueError('archive replay differs from the recorded epoch metrics')
    main = aggregate['metrics']['all_eligible']
    if [main['recall@50'], main['ndcg@50']] != list(best['metric']):
        raise ValueError('best metric does not match the replayed archive')
    queries = context.queries; stats = context.slice_definition
    if len(rows) != len(queries) or [r['ordinal'] for r in rows] != list(range(len(queries))):
        raise ValueError('archive rows do not cover the validation queries in ordinal order')
    items = [list(r['items']) for r in rows]
    assigned = labels(stats, queries, ev.ITEM_GROUPS, ev.USER_GROUPS)
    consistency_gate(items, queries, context.catalog, assigned, aggregate, ev.ITEM_GROUPS, ev.USER_GROUPS, metrics, canonical_bytes)
    out_rows = []
    for ordinal, (query, row) in enumerate(zip(queries, rows)):
        out_rows.append({'ordinal': ordinal, 'user_id': query['user_id'], 'target': query['target'], 'oov': bool(query['oov']), 'history': [int(i) for i in query['history']],
                         'items': items[ordinal], 'support': assigned['support'][ordinal], 'popularity_group': assigned['popularity'][ordinal],
                         'user_group': assigned['user_history'][ordinal]})
    pop = pop_ranking(stats, context.catalog, pop_k)
    pop_hits = sum(1 for q in queries if q['target'] in set(pop[:50]))
    short = sum(1 for r in out_rows if len(r['items']) < 100)
    return {'schema': SCHEMA, 'split': 'validation', 'best_epoch': epoch, 'query_count': len(out_rows), 'catalog_size': len(context.catalog),
            'evaluation_manifest_sha256': record['evaluation']['manifest_sha256'], 'computation_sha256': computation,
            'checkpoint_sha256_recorded': record['checkpoint_sha256'], 'best_metric': list(best['metric']),
            'slice_statistics_manifest_sha256': stats.manifest_sha256, 'thresholds': list(stats.thresholds),
            'item_groups': list(ev.ITEM_GROUPS), 'user_groups': list(ev.USER_GROUPS), 'g_short_queries': short,
            'catalog_digest': digest(list(context.catalog)) if digest else None,
            'checks': {'archive_replay_equal': True, 'slice_consistency_equal': True, 'metric_equals_best': True},
            'pop': {'ranking': pop, 'counts_depth': pop_k, 'hits_at_50': pop_hits, 'tie_rule': '(-count, item_id)'}, 'rows': out_rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description='G5 G-side assembly (read-only)')
    parser.add_argument('--line-root', required=True)
    for flag in ('prepared', 'prepared-sha', 'history-manifest', 'history-sha', 'sid-manifest', 'sid-sha', 'content', 'content-sha',
                 'input-manifest', 'input-sha', 'vectors', 'vectors-sha', 'manifest', 'manifest-sha'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--scratch', required=True)
    parser.add_argument('--split', required=True, choices=('validation',)); parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        import vkgr
        root = Path(args.line_root).resolve()
        if Path(vkgr.__file__).resolve().parent != root / 'src' / 'vkgr':
            raise ValueError('vkgr is not imported from --line-root')
        out = Path(args.output).resolve(); scratch = Path(args.scratch).resolve()
        if out == scratch or out in scratch.parents or scratch in out.parents:
            raise ValueError('output and scratch must be distinct, non-nested paths')
        for path in (out, scratch):
            if root in path.parents or path.exists() or path.with_name(path.name + '.partial').exists():
                raise ValueError('output and scratch must be new paths outside the deployment')
        from vkgr import coll_sid_a1_run as cr
        from vkgr import generative_b1_run as b1
        from vkgr import generative_experiment as ge
        from vkgr import generative_evaluation as ev
        from vkgr import sid_depth as sd
        from vkgr.baselines import metrics
        from vkgr.data_audit import canonical_bytes
        manifest_path = Path(args.manifest)
        value = cr._read_run(manifest_path, args.manifest_sha)
        config = value['config']
        context, engineering, sid = cr._contexts(root, Path(args.prepared), args.prepared_sha, Path(args.history_manifest), args.history_sha, Path(args.sid_manifest),
                                                 args.sid_sha, Path(args.content), args.content_sha, config, scratch, input_manifest=Path(args.input_manifest),
                                                 input_sha=args.input_sha, vectors_path=Path(args.vectors), vectors_sha=args.vectors_sha)
        if engineering:
            raise ValueError('formal (non-engineering) context required')
        recorded = value['binding']
        if cr._digest(context.binding) != recorded['context_binding_sha256']:
            raise ValueError('context binding differs from the run binding')
        fresh = cr._source(root)
        if fresh['files'] != recorded['source']['files'] or fresh['source_sha256'] != recorded['source']['source_sha256']:
            raise ValueError('deployment source differs from the run source ledger')
        tokens = sd.SIDTokensLD(context.codec, config['history_len'])
        record = assemble_core(value, manifest_path.parent, context, tokens, b1._read_evaluation, ev, metrics, canonical_bytes, digest=cr._digest)
        record.update(run_manifest_sha256=args.manifest_sha, seed=config['seed'], run_id=manifest_path.parent.parent.name, script_sha256=script_sha256())
        import torch
        summary = {'run_id': record['run_id'], 'seed': record['seed'], 'best_epoch': record['best_epoch'], 'query_count': record['query_count'],
                   'evaluation_manifest_sha256': record['evaluation_manifest_sha256'], 'computation_sha256': record['computation_sha256'],
                   'slice_statistics_manifest_sha256': record['slice_statistics_manifest_sha256'], 'g_short_queries': record['g_short_queries'],
                   'pop_hits_at_50': record['pop']['hits_at_50'], 'cuda_initialized': bool(torch.cuda.is_initialized()), 'manifest_sha256': args.manifest_sha,
                   'script_sha256': record['script_sha256'], 'catalog_digest': record['catalog_digest'], 'max_rss_mib': max_rss_mib(), 'vkgr_file': str(Path(vkgr.__file__).resolve())}
        if args.check_only:
            print(json.dumps({'status': 'preflight_ok', 'written': False, **summary})); return 0
        out.parent.mkdir(parents=True, exist_ok=True)
        sha = ge._write_json(out, record)             # exclusive-create ('xb'), canonical bytes
        print(json.dumps({'status': 'complete', 'output_sha256': sha, **summary})); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'ERROR G5-ASSEMBLE-G gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
