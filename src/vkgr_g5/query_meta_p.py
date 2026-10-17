"""G5 P-side validation query metadata (read-only, CPU). New file; runs with the P deployment's ``vkgr`` on PYTHONPATH.

Emits, per validation query ordinal (the order of the accepted P runs' archives): user_id, target, oov flag, the L50 history item list,
and the P-side support label of the target. Used to align the P candidates with the G candidates query by query and to cross-check
support labels. Loads exactly the data the accepted P runs used (same pinned config/inputs, ``real_inputs`` + ``apply_feat02``); no
model, no checkpoint, no embeddings, no held-out split. Output goes to a NEW file outside the deployment directory.
"""
import argparse
import json
import sys
from pathlib import Path

SCHEMA = 'g5-p-query-meta-v1'


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
            'queries_digest': b1._digest(queries), 'catalog_digest': b1._digest(list(data['catalog'])), 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description='G5 P-side validation query metadata')
    parser.add_argument('--line-root', required=True); parser.add_argument('--inputs', required=True); parser.add_argument('--inputs-sha', required=True)
    parser.add_argument('--config', required=True); parser.add_argument('--config-sha', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--split', required=True, choices=('validation',))
    args = parser.parse_args(argv)
    try:
        import vkgr
        root = Path(args.line_root).resolve()
        if root not in Path(vkgr.__file__).resolve().parents:
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
        check = b1.resource_guard(out.parent, spec['disk_cap_bytes']); check()
        data, _, _ = b1.real_inputs(spec)
        data, _ = fr.apply_feat02(data, spec, config['history_len'])
        record = extract(data, config['history_len'])
        sha = b1._write_json_atomic(out, record)
        print(json.dumps({'status': 'complete', 'output_sha256': sha, 'query_count': record['query_count'], 'queries_digest': record['queries_digest'],
                          'catalog_digest': record['catalog_digest']}))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR G5-QUERY-META-P gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
