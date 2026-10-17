"""G5 P-side validation query metadata (read-only, CPU). New file; runs with the P deployment's ``vkgr`` on PYTHONPATH.

Emits, per validation query ordinal (the order of the accepted P runs' archives): user_id, target, oov flag, the L50 history item list,
and the P-side support label of the target. Used to align the P candidates with the G candidates query by query and to cross-check
support labels. Loads exactly the data the accepted P runs used (same pinned config/inputs, ``real_inputs`` + ``apply_feat02``); no
model, no checkpoint, no embeddings, no held-out split. Output goes to a NEW file outside the deployment directory.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCHEMA = 'g5-p-query-meta-v1'


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def nearest_existing(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def check_against_manifests(record, support_sha256, manifests):
    """Strict equality of the viewed queries / catalog / support with every accepted run manifest binding (the only proof of faithfulness)."""
    if not manifests:
        raise ValueError('at least one accepted run manifest is required')
    for value in manifests:
        binding = value['binding']
        if binding['queries_sha256'] != record['queries_digest'] or binding['catalog_sha256'] != record['catalog_digest'] or binding['support_sha256'] != support_sha256:
            raise ValueError('queries/catalog/support differ from an accepted run binding')
    return len(manifests)


def extract(data, history_len):
    """Per-ordinal metadata from loaded P data (works for synthetic and real data)."""
    from vkgr import retrieval_feat02_run as fr
    from vkgr import retrieval_b1_run as b1
    _, queries = fr._view(data, history_len)
    support = set(data['support'])
    rows = []
    for ordinal, q in enumerate(queries):
        rows.append({'ordinal': ordinal, 'user_id': q['user_id'], 'target': q['target'], 'oov': bool(q['oov']), 'history': list(q['history']),
                     'support_p': 'oov' if q['oov'] else ('supported' if q['target'] in support else 'unsupported')})
    if len({r['user_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate user_id among validation queries')
    if any(not 1 <= len(r['history']) <= history_len for r in rows):
        raise ValueError('history length outside the L-view')
    return {'schema': SCHEMA, 'split': 'validation', 'history_len': history_len, 'query_count': len(rows), 'catalog_size': len(data['catalog']),
            'queries_digest': b1._digest(queries), 'catalog_digest': b1._digest(list(data['catalog'])), 'support_sha256': b1._digest(sorted(data['support'])), 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description='G5 P-side validation query metadata')
    parser.add_argument('--line-root', required=True); parser.add_argument('--inputs', required=True); parser.add_argument('--inputs-sha', required=True)
    parser.add_argument('--config', required=True); parser.add_argument('--config-sha', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--split', required=True, choices=('validation',))
    parser.add_argument('--check-only', action='store_true', help='real-path preflight: load, extract and verify everything but write nothing')
    parser.add_argument('--manifest', action='append', required=True); parser.add_argument('--manifest-sha', action='append', required=True)
    parser.add_argument('--expect-queries', type=int, required=True); parser.add_argument('--expect-oov', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        import vkgr
        root = Path(args.line_root).resolve()
        if Path(vkgr.__file__).resolve().parent != root / 'src' / 'vkgr':
            raise ValueError('vkgr is not imported from --line-root')
        out = Path(args.output).resolve()
        if root in out.parents or out.exists() or out.with_name(out.name + '.partial').exists():
            raise ValueError('output must be a new file outside the deployment')
        from vkgr import retrieval_b1_run as b1
        from vkgr import retrieval_feat02_run as fr
        from vkgr import retrieval_seqm1_run as sm
        from vkgr.retrieval_ablation_run import read_pinned
        from vkgr.retrieval_php01 import FEATURE
        config = read_pinned(args.config, args.config_sha); sm.validate_config(config)
        spec = fr.validate_inputs(read_pinned(args.inputs, args.inputs_sha), FEATURE)
        if len(args.manifest) != len(args.manifest_sha):
            raise ValueError('every --manifest needs its --manifest-sha')
        check = b1.resource_guard(nearest_existing(out.parent), spec['disk_cap_bytes']); check()
        data, _, _ = b1.real_inputs(spec)
        data, _ = fr.apply_feat02(data, spec, config['history_len'])
        record = extract(data, config['history_len'])
        if record['query_count'] != args.expect_queries or sum(1 for r in record['rows'] if r['oov']) != args.expect_oov:
            raise ValueError('query / OOV counts differ from the registered validation set')
        checked = check_against_manifests(record, record['support_sha256'], [read_pinned(m, d) for m, d in zip(args.manifest, args.manifest_sha)])
        record['script_sha256'] = script_sha256(); record['manifests_checked'] = checked
        size = len(json.dumps(record, sort_keys=True, allow_nan=False))
        if args.check_only:
            print(json.dumps({'status': 'preflight_ok', 'written': False, 'serialized_bytes': size, 'manifests_checked': checked, 'support_sha256': record['support_sha256'], 'script_sha256': record['script_sha256'], 'query_count': record['query_count'], 'queries_digest': record['queries_digest'],
                              'catalog_digest': record['catalog_digest'], 'catalog_size': record['catalog_size'],
                              'oov_queries': sum(1 for r in record['rows'] if r['oov']), 'support_p': {k: sum(1 for r in record['rows'] if r['support_p'] == k) for k in ('supported', 'unsupported', 'oov')}}))
            return 0
        out.parent.mkdir(parents=True, exist_ok=False)          # the run's own fresh out/ directory; an existing directory is refused
        sha = b1._write_json_atomic(out, record)
        print(json.dumps({'status': 'complete', 'output_sha256': sha, 'manifests_checked': checked, 'support_sha256': record['support_sha256'], 'script_sha256': record['script_sha256'], 'query_count': record['query_count'], 'queries_digest': record['queries_digest'],
                          'catalog_digest': record['catalog_digest']}))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'ERROR G5-QUERY-META-P gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
