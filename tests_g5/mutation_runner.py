"""Generic isolated mutation runner. Each mutant is applied to a fresh copy of the repository subset and run with a unique bytecode cache.

Usage: python tests_g5/mutation_runner.py <suite>   (suite = qmeta | enqueue)
Output: markdown ledger on stdout. Exit 1 on any unexpected outcome.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = {
    'qmeta': {'target': 'src/vkgr_g5/query_meta_p.py', 'copy': ['src', 'tests_g5'], 'cmd': ['-m', 'unittest', 'tests_g5.test_query_meta_p'], 'mutants': [
        ('dup-user-gate-off', "    if len({r['user_id'] for r in rows}) != len(rows):", "    if False:", 'kill'),
        ('history-gate-off', "    if any(not 1 <= len(r['history']) <= history_len for r in rows):", "    if False:", 'kill'),
        ('support-flip', "'supported' if q['target'] in support else 'unsupported'", "'unsupported' if q['target'] in support else 'supported'", 'kill'),
        ('oov-label', "'oov' if q['oov'] else (", "'oov' if not q['oov'] else (", 'kill'),
        ('query-digest-over-rows', "'queries_digest': b1._digest(queries)", "'queries_digest': b1._digest(rows)", 'kill'),
        ('catalog-digest-wrong', "'catalog_digest': b1._digest(list(data['catalog']))", "'catalog_digest': b1._digest(sorted(data['catalog'])[::-1])", 'kill'),
        ('support-digest-unsorted', "b1._digest(sorted(data['support']))", "b1._digest(list(data['support']))", 'kill'),
        ('manifest-queries-skipped', "binding['queries_sha256'] != record['queries_digest'] or ", "False or ", 'kill'),
        ('manifest-catalog-skipped', "or binding['catalog_sha256'] != record['catalog_digest'] or ", "or False or ", 'kill'),
        ('manifest-support-skipped', "or binding['support_sha256'] != support_sha256:", "or False:", 'kill'),
        ('manifest-required-off', "    if not manifests:", "    if False:", 'kill'),
        ('manifest-pair-length-off', "        if len(args.manifest) != len(args.manifest_sha):", "        if False:", 'kill'),
        ('only-first-manifest', "for value in manifests:", "for value in manifests[:1]:", 'kill'),
        ('query-count-lt', "if record['query_count'] != args.expect_queries or", "if record['query_count'] < args.expect_queries or", 'kill'),
        ('oov-count-lt', "!= args.expect_oov:", "< args.expect_oov:", 'kill'),
        ('manifest-arg-optional', "parser.add_argument('--manifest', action='append', required=True)", "parser.add_argument('--manifest', action='append', required=False)", 'equivalent'),
        ('count-check-off', "        if record['query_count'] != args.expect_queries or sum(1 for r in record['rows'] if r['oov']) != args.expect_oov:", "        if False:", 'kill'),
        ('oov-count-ignored', "or sum(1 for r in record['rows'] if r['oov']) != args.expect_oov:", ":", 'syntax'),
        ('mkdir-in-check-only', "        if args.check_only:\n", "        out.parent.mkdir(parents=True, exist_ok=True)\n        if args.check_only:\n", 'kill'),
        ('mkdir-exist-ok', "out.parent.mkdir(parents=True, exist_ok=False)", "out.parent.mkdir(parents=True, exist_ok=True)", 'kill'),
        ('check-only-writes', "        if args.check_only:\n", "        if False:\n", 'kill'),
        ('line-root-ancestor-ok', "        if Path(vkgr.__file__).resolve().parent != root / 'src' / 'vkgr':", "        if root not in Path(vkgr.__file__).resolve().parents:", 'kill'),
        ('output-in-deployment-ok', "        if root in out.parents or out.exists() or out.with_name(out.name + '.partial').exists():", "        if out.exists() or out.with_name(out.name + '.partial').exists():", 'kill'),
        ('existing-output-ok', "        if root in out.parents or out.exists() or out.with_name(out.name + '.partial').exists():", "        if root in out.parents or out.with_name(out.name + '.partial').exists():", 'kill'),
        ('partial-check-off', "        if root in out.parents or out.exists() or out.with_name(out.name + '.partial').exists():", "        if root in out.parents or out.exists():", 'kill'),
        ('resource-guard-on-output-parent', "b1.resource_guard(nearest_existing(out.parent), spec['disk_cap_bytes'])", "b1.resource_guard(out.parent, spec['disk_cap_bytes'])", 'kill'),
        ('subprocess-error-uncaught', "RuntimeError, subprocess.SubprocessError)", "RuntimeError)", 'kill'),
    ]},
    'enqueue': {'target': 'ops/g5/g5_fusion_enqueue_20261018.py', 'copy': ['ops'], 'cmd': ['-m', 'unittest', 'discover', '-s', 'ops/g5', '-p', 'test_g5_fusion_enqueue*'], 'mutants': [
        ('claimed-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'rejected')", 'kill'),
        ('rejected-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'claimed')", 'kill'),
        ('runs-dir-not-checked', "    if (root / 'runs' / rid).exists():\n        return True\n", "", 'kill'),
        ('reason-form-dropped', "(json|reason|json\\.reason)", "(json)", 'kill'),
        ('json-reason-form-dropped', "(json|reason|json\\.reason)", "(json|reason)", 'kill'),
        ('tmp-form-dropped', "(\\.tmp)?'", "'", 'kill'),
        ('config-from-template', "config=str(Path(p_deploy) / get('--config')),", "config=t['config'],", 'kill'),
        ('gen-limit-80', "if gen and (len(gen) > 8 or not gen.isalnum()):", "if gen and (len(gen) > 80 or not gen.isalnum()):", 'kill'),
        ('exact-match-off', "    return any(pattern.fullmatch(p.name) for sub", "    return any(rid in p.name for sub", 'kill'),
        ('inputs-positional', "inputs=[i for i in t['inputs'] if i != get('--manifest')] + manifests,", "inputs=[get('--inputs')] + [i for i in t['inputs'][1:3]] + manifests,", 'kill'),
        ('manifest-kept-in-inputs', "inputs=[i for i in t['inputs'] if i != get('--manifest')] + manifests,", "inputs=list(t['inputs']) + manifests,", 'kill'),
        ('manifests-dropped-from-inputs', "inputs=[i for i in t['inputs'] if i != get('--manifest')] + manifests,", "inputs=[i for i in t['inputs'] if i != get('--manifest')],", 'kill'),
        ('manifest-sha-wrong', "'--manifest-sha', P_MANIFESTS[seed]]", "'--manifest-sha', P_MANIFESTS[42]]", 'kill'),
        ('expect-queries-wrong', "EXPECT_QUERIES, EXPECT_OOV = 2520, 623", "EXPECT_QUERIES, EXPECT_OOV = 2519, 623", 'kill'),
        ('expect-oov-wrong', "EXPECT_QUERIES, EXPECT_OOV = 2520, 623", "EXPECT_QUERIES, EXPECT_OOV = 2520, 622", 'kill'),
        ('root-from-request', "root = Path(root)      # the queue root", "root = Path(req.get('root', root))      # the queue root", 'kill'),
        ('pythonpath-not-rewritten', "f'PYTHONPATH=src:{fusion_deploy}/src' if e.startswith('PYTHONPATH=') else e", "e", 'kill'),
        ('fusion-first', "f'PYTHONPATH=src:{fusion_deploy}/src'", "f'PYTHONPATH={fusion_deploy}/src:src'", 'kill'),
        ('config-sha-check-off', "    if get('--config-sha') != CONFIG_SHA:", "    if False:", 'kill'),
        ('check-only-missing', "    if STAGES[stage][1]:\n        cmd.append('--check-only')", "    if False:\n        cmd.append('--check-only')", 'kill'),
        ('vram-floor-off', "or type(vram_mib) is not int or vram_mib < 256", "or type(vram_mib) is not int", 'kill'),
        ('wall-cap-off', "or not 0 < wall_hours <= 4:", "or not 0 < wall_hours:", 'kill'),
        ('p-deploy-regex-open', "P_DEPLOY_PATTERN = r'/workspace/vkgr-g5-p-export100-v[0-9]+'", "P_DEPLOY_PATTERN = r'/workspace/.*'", 'kill'),
        ('fusion-regex-open', "FUSION_DEPLOY_PATTERN = r'/workspace/vkgr-g5-fusion-v[0-9]+'", "FUSION_DEPLOY_PATTERN = r'/workspace/.*'", 'kill'),
        ('duplicate-check-off', "    if exists_anywhere(root, job['run_id']):", "    if False:", 'kill'),
    ]},
    'asm_enqueue': {'target': 'ops/g5/g5_assemble_enqueue_20261018.py', 'copy': ['ops'], 'cmd': ['-m', 'unittest', 'discover', '-s', 'ops/g5', '-p', 'test_g5_assemble_enqueue*'], 'mutants': [
        ('prefix-match', "    return any(pattern.fullmatch(p.name) for sub", "    return any(rid in p.name for sub", 'kill'),
        ('claimed-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'rejected')", 'kill'),
        ('rejected-not-scanned', "for sub in ('pending', 'claimed', 'rejected')", "for sub in ('pending', 'claimed')", 'kill'),
        ('runs-dir-not-checked', "    if (root / 'runs' / rid).exists():\n        return True\n", "", 'kill'),
        ('manifest-sha-check-off', "get('--manifest-sha') != MANIFEST_SHA[seed] or ", "False or ", 'kill'),
        ('cwd-check-off', "or t['cwd'] != g_deploy or ", "or ", 'kill'),
        ('entry-check-off', "or argv[argv.index('-m') + 1:argv.index('-m') + 3] != ['vkgr.coll_sid_a1_run', 'verify']:", ":", 'syntax'),
        ('check-only-missing', "    if STAGES[stage][1]:\n        cmd.append('--check-only')", "    if False:\n        cmd.append('--check-only')", 'kill'),
        ('fusion-first', "f'PYTHONPATH={g_deploy}/src:{fusion_deploy}/src'", "f'PYTHONPATH={fusion_deploy}/src:{g_deploy}/src'", 'kill'),
        ('pythonpath-check-off', "    if env == t['env'] or f'PYTHONPATH={g_deploy}/src' not in t['env']:", "    if False:", 'kill'),
        ('flag-dropped', "    for flag in FLAGS:\n        cmd += [flag, get(flag)]", "    for flag in FLAGS[:-2]:\n        cmd += [flag, get(flag)]", 'kill'),
        ('split-wrong', "'--split', 'validation']", "'--split', 'train']", 'kill'),
        ('vram-floor-off', "or type(vram_mib) is not int or vram_mib < 256", "or type(vram_mib) is not int", 'kill'),
        ('gen-limit-80', "if gen and (len(gen) > 8 or not gen.isalnum()):", "if gen and (len(gen) > 80 or not gen.isalnum()):", 'kill'),
        ('duplicate-check-off', "        if exists_anywhere(root, job['run_id']):", "        if False:", 'kill'),
        ('g-deploy-regex-open', "G_DEPLOY_PATTERN = r'/workspace/vkgr-coll-a1-g-v[0-9]+'", "G_DEPLOY_PATTERN = r'/workspace/.*'", 'kill'),
    ]},
    'fuse': {'target': 'src/vkgr_g5/fuse.py', 'copy': ['src', 'tests_g5'], 'cmd': ['-m', 'unittest', 'tests_g5.test_fuse'], 'mutants': [
        ('p-gate-flag-off', "gate.get('first50_equal') is not True or ", "False or ", 'kill'),
        ('p-mismatch-count-off', "or gate.get('mismatched_queries') != 0", "", 'kill'),
        ('p-depth-off', "record.get('depth') != 100 or ", "False or ", 'kill'),
        ('p-binding-off', "    if record.get('train_manifest_sha256') != P_MANIFEST_SHA[seed]:", "    if False:", 'kill'),
        ('p-shape-off', "    if len(lists) != expected_queries or any(len(row) != 100 for row in lists):", "    if False:", 'kill'),
        ('p-short-row-ok', "any(len(row) != 100 for row in lists)", "any(len(row) < 99 for row in lists)", 'kill'),
        ('g-binding-off', "value.get('run_manifest_sha256') != G_MANIFEST_SHA[seed]", "False", 'kill'),
        ('g-checks-off', " or value['checks'] != {'archive_replay_equal': True, 'slice_consistency_equal': True, 'metric_equals_best': True}", "", 'kill'),
        ('g-seed-off', "value.get('seed') != seed or ", "False or ", 'kill'),
        ('align-history-off', "(p['user_id'], p['target'], p['oov'], p['history']) != (g['user_id'], g['target'], g['oov'], g['history'])", "(p['user_id'], p['target'], p['oov']) != (g['user_id'], g['target'], g['oov'])", 'kill'),
        ('align-target-off', "(p['user_id'], p['target'], p['oov'], p['history']) != (g['user_id'], g['target'], g['oov'], g['history'])", "(p['user_id'], p['oov'], p['history']) != (g['user_id'], g['oov'], g['history'])", 'kill'),
        ('align-only-seed-42', "for seed, assembly in assemblies.items():", "for seed, assembly in list(assemblies.items())[:1]:", 'kill'),
        ('hits-check-off', "    if got != expect_hits:", "    if False:", 'kill'),
        ('hits-g-skipped', "got.update({('G', s): single_route_hits([r['items'] for r in assemblies[s]['rows']], targets) for s in SEEDS})", "got.update({('G', s): expect_hits[('G', s)] for s in SEEDS})", 'kill'),
        ('hits-uses-top30', "return sum(1 for ranked, target in zip(lists, targets) if target in ranked[:50])", "return sum(1 for ranked, target in zip(lists, targets) if target in ranked[:30])", 'kill'),
        ('sha-pin-off', "sha256_file(path) != digest", "False", 'kill'),
        ('held-out-check-off', "            if HELD_OUT_COMPONENT.search(part):", "            if False:", 'kill'),
        ('held-out-substring', "HELD_OUT_COMPONENT = re.compile(r'(?i)(^|[-_.])(test|week[-_]?26)([-_.]|$)')", "HELD_OUT_COMPONENT = re.compile(r'(?i)test')", 'kill'),
        ('output-exists-ok', "        if out.exists():\n            raise ValueError('output directory must not exist')", "", 'kill'),
        ('check-only-writes', "        if args.check_only:\n            print(json.dumps({'status': 'preflight_ok'", "        if False:\n            print(json.dumps({'status': 'preflight_ok'", 'kill'),
        ('catalog-size-check-off', "    if any(a['catalog_size'] != meta['catalog_size'] for a in assemblies.values()):", "    if False:", 'kill'),
        ('pop-from-wrong-seed', "routes[('POP', None)] = base['pop']['ranking']", "routes[('POP', None)] = assemblies[2025]['pop']['ranking']", 'kill'),
        ('queries-from-wrong-labels', "'popularity_group': r['popularity_group'], 'user_group': r['user_group'], 'history_len': len(r['history'])}", "'popularity_group': r['popularity_group'], 'user_group': r['user_group'], 'history_len': 1}", 'kill'),
        ('support-mismatch-not-counted', "support_mismatch = sum(1 for p, g in zip(meta['rows'], base['rows']) if p['support_p'] != g['support'])", "support_mismatch = 0", 'kill'),
        ('query-count-meta-off', " or meta.get('query_count') != expected_queries:", ":", 'syntax'),
        ('seeds-required-off', "    if set(out) != set(SEEDS):", "    if False:", 'kill'),
        ('duplicate-seed-ok', "or int(seed) in out:", ":", 'syntax'),
        ('fusion-hash-not-recorded', "'fusion_v2_script_sha256': fv.script_sha256(),", "'fusion_v2_script_sha256': '0' * 64,", 'kill'),
    ]},
    'assemble': {'target': 'src/vkgr_g5/assemble_g.py', 'copy': ['src', 'tests_g5'], 'cmds': [['-m', 'unittest', 'tests_g5.test_assemble_g.LabelTests', 'tests_g5.test_assemble_g.GateTests', 'tests_g5.test_assemble_g.CliTests'], ['-m', 'unittest', 'tests_g5.test_assemble_g.IntegrationTests']], 'env': {'VKGR_G5_G_ROOT': '/Users/eddy/.codex/worktrees/sid-coll-s1/vk-generative-retrieval'}, 'mutants': [
        ('popularity-boundary', "count <= stats.thresholds[0] else item_groups[2] if count <= stats.thresholds[1]", "count < stats.thresholds[0] else item_groups[2] if count <= stats.thresholds[1]", 'kill'),
        ('user-boundary', "users.append(", "users.append", 'syntax'),
        ('support-flip', "'supported' if count > 0 else 'unsupported'", "'unsupported' if count > 0 else 'supported'", 'kill'),
        ('oov-popularity', "popularity.append('oov'); support.append('oov')", "popularity.append('oov'); support.append('unsupported')", 'kill'),
        ('history-gate-off', "        if not 1 <= length <= 50:", "        if False:", 'kill'),
        ('gate-section-skipped', "    for section, names in spec:", "    for section, names in spec[:2]:", 'kill'),
        ('pop-tie-rule', "key=lambda i: (-stats.item_counts[i], i))", "key=lambda i: (-stats.item_counts[i], -i))", 'kill'),
        ('pop-depth-gate-off', "    if len(ranking) < k:", "    if False:", 'kill'),
        ('ordinal-order-gate-off', " or [r['ordinal'] for r in rows] != list(range(len(queries))):", ":", 'syntax'),
        ('metric-gate-off', "    if [main['recall@50'], main['ndcg@50']] != list(best['metric']):", "    if False:", 'kill'),
        ('archive-replay-gate-off', "    if canonical_bytes(aggregate) != canonical_bytes(record['metrics']) or computation != record['evaluation']['computation_sha256']:", "    if False:", 'kill'),
        ('kind-gate-off', "    if record.get('epoch') != epoch or record['evaluation']['kind'] != 'query_evaluation':", "    if False:", 'kill'),
        ('consistency-gate-off', "    consistency_gate(items, queries, context.catalog, assigned, aggregate, ev.ITEM_GROUPS, ev.USER_GROUPS, metrics, canonical_bytes)", "    pass", 'kill'),
        ('items-copied-from-wrong-row', "items = [list(r['items']) for r in rows]", "items = [list(r['items']) for r in rows[::-1]]", 'kill'),
        ('line-root-ancestor-ok', "        if root not in Path(vkgr.__file__).resolve().parents:", "        if False:", 'kill'),
        ('engineering-ok', "        if engineering:\n            raise ValueError('formal (non-engineering) context required')", "", 'equivalent'),
    ]},
}


def suite_run(work, cmd):
    if cmd and isinstance(cmd[0], list):          # several isolated interpreters (the P and G code bases both provide a package named vkgr)
        results = [suite_run(work, c) for c in cmd]
        return max(r[0] for r in results), sorted({t for r in results for t in r[1]})
    cache = tempfile.mkdtemp(prefix='mut-pyc-')
    env = {**os.environ, 'PYTHONPYCACHEPREFIX': cache, 'PYTHONDONTWRITEBYTECODE': '1', **EXTRA_ENV}
    proc = subprocess.run([sys.executable, '-B', *cmd], cwd=work, env=env, capture_output=True, text=True)
    return proc.returncode, sorted(set(re.findall(r'^(?:FAIL|ERROR): (\w+) ', proc.stderr, re.M)))


EXTRA_ENV = {}


def main(name):
    spec = SUITES[name]
    EXTRA_ENV.update(spec.get('env', {}))
    print(f'# Mutation ledger: {name} ({spec["target"]})\n')
    print('Isolation: every run uses a fresh copy of the sources, `python -B`, `PYTHONDONTWRITEBYTECODE=1` and a unique `PYTHONPYCACHEPREFIX`.\n')
    def fresh():
        work = Path(tempfile.mkdtemp(prefix='mut-'))
        for part in spec['copy']:
            shutil.copytree(ROOT / part, work / part, ignore=shutil.ignore_patterns('__pycache__'))
        return work
    base_code, failed = suite_run(fresh(), spec.get('cmds') or spec['cmd'])
    print(f'Baseline: returncode {base_code}, failing tests {failed}\n')
    print('| id | change | expected | result | killed by |\n|---|---|---|---|---|')
    bad = 0
    for mid, old, new, expected in spec['mutants']:
        if expected == 'syntax':
            continue
        work = fresh(); target = work / spec['target']; source = target.read_text()
        if source.count(old) < 1:
            print(f'| {mid} | {old[:50]!r} | {expected} | PATTERN_NOT_FOUND | |'); bad += 1; continue
        target.write_text(source.replace(old, new, 1))
        code, failed = suite_run(work, spec.get('cmds') or spec['cmd'])
        result = 'killed' if code != 0 else 'SURVIVED'
        if (expected == 'kill') != (result == 'killed'):
            bad += 1
        print(f'| {mid} | {new[:60]!r} | {expected} | {result} | {", ".join(failed[:3]) or "-"} |')
    print(f'\nMutants: {len([m for m in spec["mutants"] if m[3] != "syntax"])}; unexpected outcomes: {bad}.')
    return 1 if bad or base_code else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
