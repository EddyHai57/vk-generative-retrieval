"""G5 fuse command (CPU, stdlib only): loads the pinned candidate files, checks identity and alignment, runs the frozen fusion core, writes one new
result directory. It reads exactly: the three P top-100 exports (+ their export manifests), the three G assemblies, the P query metadata.
It never opens a model, checkpoint, embedding, held-out split or any existing run directory beyond those pinned input files.

Hard checks before any fusion number is produced (fail closed, exit 2): every input sha256; P export manifests are complete, depth 100, first-50 gate
true and bound to the accepted P train manifest; G assemblies are bound to the accepted G train manifests; all files share the validation query
count; per-ordinal user_id / target / oov / history agree between the P metadata and every G assembly; the single-route hit counts at 50 equal the
accepted runs' registered values (P 466/481/471, G 388/376/384); no path component names the held-out split.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import fusion_v2 as fv

SCHEMA = 'g5-fusion-results-v1'
P_MANIFEST_SHA = {42: 'bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e', 2024: '8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8',
                  2025: '783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb'}
G_MANIFEST_SHA = {42: 'a431a5b1734930671cf7ca092e469a6da16e3c130ffb30836b9aa57aaca8b36b', 2024: '09a0b790c6b027613ef15bb6c9855d7b1adee79d9bf6ea093c16d09a1a5b8d36',
                  2025: '86767e4d30ababc8087898f6f960197021ce54fd09cc6d775da4c373b242c712'}
# Accepted single-route hits at 50 (experiment_log: M1 acceptance table; 2026-10-17 19:00 S5 table): a cross-check of the loaded candidates.
EXPECT_HITS = {('P', 42): 466, ('P', 2024): 481, ('P', 2025): 471, ('G', 42): 388, ('G', 2024): 376, ('G', 2025): 384}
SEEDS = (42, 2024, 2025)
QUERIES = 2520
HELD_OUT_COMPONENT = re.compile(r'(?i)(^|[-_.])(test|week[-_]?26)([-_.]|$)')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refuse_held_out_paths(paths):
    for path in paths:
        for part in Path(path).parts:
            if HELD_OUT_COMPONENT.search(part):
                raise ValueError('a path component names the held-out split')


def read_pinned_json(path, digest):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or type(digest) is not str or len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f'input hash mismatch: {path.name}')
    return json.loads(path.read_text())


def load_p(seed, export_manifest, export_sha, expected_queries):
    record = read_pinned_json(export_manifest, export_sha)
    gate = record.get('gate') or {}
    if record.get('status') != 'complete' or record.get('schema') != 'p-seqm1-export100-v1' or record.get('depth') != 100 or record.get('queries') != expected_queries \
            or gate.get('first50_equal') is not True or gate.get('mismatched_queries') != 0 or record.get('seed') != seed:
        raise ValueError(f'P export manifest not acceptable (seed {seed})')
    if record.get('train_manifest_sha256') != P_MANIFEST_SHA[seed]:
        raise ValueError(f'P export is not bound to the accepted P train manifest (seed {seed})')
    lists = read_pinned_json(Path(export_manifest).with_name('top100.json'), record['top100_sha256'])
    if len(lists) != expected_queries or any(len(row) != 100 for row in lists):
        raise ValueError(f'P candidate lists have the wrong shape (seed {seed})')
    return lists, record


def load_g(seed, path, digest, expected_queries):
    value = read_pinned_json(path, digest)
    if value.get('schema') != 'g5-g-assembly-v1' or value.get('split') != 'validation' or value.get('run_manifest_sha256') != G_MANIFEST_SHA[seed] \
            or value.get('seed') != seed or value.get('query_count') != expected_queries or value['checks'] != {'archive_replay_equal': True, 'slice_consistency_equal': True, 'metric_equals_best': True}:
        raise ValueError(f'G assembly not acceptable or not bound to the accepted G run (seed {seed})')
    return value


def align(meta, assemblies):
    """Per ordinal user_id / target / oov / history must agree between the P metadata and every G assembly (fatal otherwise)."""
    rows = meta['rows']
    for seed, assembly in assemblies.items():
        grows = assembly['rows']
        if len(grows) != len(rows):
            raise ValueError(f'query count differs between P metadata and G assembly (seed {seed})')
        for ordinal, (p, g) in enumerate(zip(rows, grows)):
            if p['ordinal'] != ordinal or g['ordinal'] != ordinal or (p['user_id'], p['target'], p['oov'], p['history']) != (g['user_id'], g['target'], g['oov'], g['history']):
                raise ValueError(f'query alignment failed at ordinal {ordinal} (seed {seed})')
    return True


def single_route_hits(lists, targets):
    return sum(1 for ranked, target in zip(lists, targets) if target in ranked[:50])


def build_inputs(p_inputs, g_inputs, meta_input, expected_queries=None, expect_hits=None):
    expected_queries = QUERIES if expected_queries is None else expected_queries
    expect_hits = EXPECT_HITS if expect_hits is None else expect_hits
    meta = read_pinned_json(*meta_input)
    if meta.get('schema') != 'g5-p-query-meta-v1' or meta.get('split') != 'validation' or meta.get('query_count') != expected_queries:
        raise ValueError('P query metadata not acceptable')
    p_lists, p_records = {}, {}
    for seed in SEEDS:
        p_lists[seed], p_records[seed] = load_p(seed, *p_inputs[seed], expected_queries)
    assemblies = {seed: load_g(seed, *g_inputs[seed], expected_queries) for seed in SEEDS}
    align(meta, assemblies)
    base = assemblies[42]
    if any(a['catalog_size'] != meta['catalog_size'] for a in assemblies.values()):
        raise ValueError('catalog size differs between P metadata and a G assembly')
    targets = [r['target'] for r in meta['rows']]
    got = {('P', s): single_route_hits(p_lists[s], targets) for s in SEEDS}
    got.update({('G', s): single_route_hits([r['items'] for r in assemblies[s]['rows']], targets) for s in SEEDS})
    if got != expect_hits:
        raise ValueError(f'single-route hits differ from the accepted runs: {sorted(got.items())} != {sorted(expect_hits.items())}')
    support_mismatch = sum(1 for p, g in zip(meta['rows'], base['rows']) if p['support_p'] != g['support'])
    queries = [{'target': r['target'], 'oov': r['oov'], 'support': r['support'], 'popularity_group': r['popularity_group'], 'user_group': r['user_group'], 'history_len': len(r['history'])}
               for r in base['rows']]
    routes = {('P', s): p_lists[s] for s in SEEDS}
    routes.update({('G', s): [r['items'] for r in assemblies[s]['rows']] for s in SEEDS})
    routes[('POP', None)] = base['pop']['ranking']
    return routes, queries, meta['catalog_size'], {'single_route_hits': {f'{k[0]}{k[1]}': v for k, v in sorted(got.items())}, 'support_label_mismatch_p_vs_g': support_mismatch,
                                                  'pop_hits_at_50_recorded': base['pop']['hits_at_50'], 'slice_statistics_manifest_sha256': base['slice_statistics_manifest_sha256']}


def render_markdown(result):
    lines = ['# G5 fusion results (mechanical report; no verdict)', '', f"Generated {result['generated_at_utc']}. Script SHA256 fuse `{result['provenance']['fuse_script_sha256']}`, fusion core `{result['provenance']['fusion_v2_script_sha256']}`.", '',
             '| row | routes | hits | R@50 | N@50 | pool mean |', '|---|---|---:|---:|---:|---:|']
    for name, row in result['fusion']['rows'].items():
        cell = row['slices']['all']
        lines.append(f"| {name} | {'+'.join(row['routes'])} | {cell['hits']} | {cell['R@50']:.6f} | {cell['N@50']:.6f} | {row['pool_size']['mean']:.1f} |")
    paired = result['fusion']['paired']
    if isinstance(paired, dict):
        lines += ['', '| comparison | b | c | delta | exact McNemar p | bootstrap95 delta | condition flag |', '|---|---:|---:|---:|---:|---|---|']
        for label, p in paired.items():
            lines.append(f"| {label}: {p['row']} - {p['reference']} | {p['b']} | {p['c']} | {p['delta_hits']} | {p['mcnemar_p']:.6g} | {p['bootstrap95_delta_hits']} | {p['mechanical_condition']} |")
    else:
        lines += ['', str(paired)]
    lines += ['', '## Notes'] + [f'- {note}' for note in result['fusion']['report_notes']]
    return '\n'.join(lines) + '\n'


def pair(text):
    path, _, digest = text.rpartition('@')
    if not path or len(digest) != 64:
        raise ValueError('expected PATH@SHA256')
    return path, digest


def seeded(texts):
    out = {}
    for text in texts:
        seed, _, rest = text.partition('=')
        if not seed.isdigit() or int(seed) not in SEEDS or int(seed) in out:
            raise ValueError('expected SEED=PATH@SHA256 once per seed')
        out[int(seed)] = pair(rest)
    if set(out) != set(SEEDS):
        raise ValueError('all three seeds are required')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='G5 fuse (read-only inputs, one new output directory)')
    parser.add_argument('--p-export', action='append', required=True, metavar='SEED=EXPORT_MANIFEST@SHA')
    parser.add_argument('--g-assembly', action='append', required=True, metavar='SEED=PATH@SHA')
    parser.add_argument('--p-query-meta', required=True, metavar='PATH@SHA')
    parser.add_argument('--output-dir', required=True); parser.add_argument('--split', required=True, choices=('validation',))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        p_inputs, g_inputs, meta_input = seeded(args.p_export), seeded(args.g_assembly), pair(args.p_query_meta)
        out = Path(args.output_dir).resolve()
        refuse_held_out_paths([out, meta_input[0], *[v[0] for v in p_inputs.values()], *[v[0] for v in g_inputs.values()]])
        if out.exists():
            raise ValueError('output directory must not exist')
        routes, queries, catalog_size, checks = build_inputs(p_inputs, g_inputs, meta_input)
        if args.check_only:
            print(json.dumps({'status': 'preflight_ok', 'written': False, 'checks': checks, 'queries': len(queries)}, sort_keys=True)); return 0
        result = fv.run(routes, queries, catalog_size)
        generated = datetime.now(timezone.utc).isoformat()
        result = {'schema': SCHEMA, 'generated_at_utc': generated, 'checks': checks, 'fusion': result,
                  'inputs': {'p_export': {str(s): v[1] for s, v in p_inputs.items()}, 'g_assembly': {str(s): v[1] for s, v in g_inputs.items()}, 'p_query_meta': meta_input[1]},
                  'provenance': {'fuse_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'fusion_v2_script_sha256': fv.script_sha256(),
                                 'python': sys.version.split()[0]}}
        out.mkdir(parents=True, exist_ok=False)
        payload = json.dumps(result, sort_keys=True, allow_nan=False).encode() + b'\n'
        with (out / 'fusion-results.json').open('xb') as stream:
            stream.write(payload)
        with (out / 'fusion-results.md').open('x', encoding='utf-8') as stream:
            stream.write(render_markdown(result))
        print(json.dumps({'status': 'complete', 'results_sha256': hashlib.sha256(payload).hexdigest(), 'generated_at_utc': generated, 'checks': checks}, sort_keys=True)); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f'ERROR G5-FUSE gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
