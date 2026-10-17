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
}


def suite_run(work, cmd):
    cache = tempfile.mkdtemp(prefix='mut-pyc-')
    env = {**os.environ, 'PYTHONPYCACHEPREFIX': cache, 'PYTHONDONTWRITEBYTECODE': '1'}
    proc = subprocess.run([sys.executable, '-B', *cmd], cwd=work, env=env, capture_output=True, text=True)
    return proc.returncode, sorted(set(re.findall(r'^(?:FAIL|ERROR): (\w+) ', proc.stderr, re.M)))


def main(name):
    spec = SUITES[name]
    print(f'# Mutation ledger: {name} ({spec["target"]})\n')
    print('Isolation: every run uses a fresh copy of the sources, `python -B`, `PYTHONDONTWRITEBYTECODE=1` and a unique `PYTHONPYCACHEPREFIX`.\n')
    def fresh():
        work = Path(tempfile.mkdtemp(prefix='mut-'))
        for part in spec['copy']:
            shutil.copytree(ROOT / part, work / part, ignore=shutil.ignore_patterns('__pycache__'))
        return work
    base_code, failed = suite_run(fresh(), spec['cmd'])
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
        code, failed = suite_run(work, spec['cmd'])
        result = 'killed' if code != 0 else 'SURVIVED'
        if (expected == 'kill') != (result == 'killed'):
            bad += 1
        print(f'| {mid} | {new[:60]!r} | {expected} | {result} | {", ".join(failed[:3]) or "-"} |')
    print(f'\nMutants: {len([m for m in spec["mutants"] if m[3] != "syntax"])}; unexpected outcomes: {bad}.')
    return 1 if bad or base_code else 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
