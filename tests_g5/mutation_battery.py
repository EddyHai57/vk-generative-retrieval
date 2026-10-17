"""Mutation battery for src/vkgr_g5/fusion_v2.py (a tool, not a unit test).

Every mutant is applied to a throw-away COPY of src/vkgr_g5 and tests_g5 (the working tree is never touched) and the test suite is run
with `python -B` and a fresh PYTHONPYCACHEPREFIX per run, so a stale bytecode cache cannot hide a mutation (the isolation is asserted
in the report header). Usage: python tests_g5/mutation_battery.py > ledger.md
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = 'src/vkgr_g5/fusion_v2.py'
# (id, review group or '-', description, old, new, expected) ; expected = 'kill' or 'equivalent'
MUTANTS = [
    ('h2-or', '01 H2 and->or', 'H2 requires either family member', "paired['H2a']['mechanical_condition'] and paired['H2b']['mechanical_condition']", "paired['H2a']['mechanical_condition'] or paired['H2b']['mechanical_condition']", 'kill'),
    ('h2b-ref', '02 H2b reference', 'C5-C7 compared against C1', "('H2b', 'C5', 'C7')", "('H2b', 'C5', 'C1')", 'kill'),
    ('loc-support-off', '03 localization support', 'unsupported alone no longer places a query in the set', "(q['support'] == 'unsupported' or", "(False or", 'kill'),
    ('loc-popularity-off', '04 localization popularity', 'low popularity alone no longer places a query in the set', "or q['popularity_group'] in LOW_POPULARITY or", "or False or", 'kill'),
    ('loc-gshort-off', '-', 'g_short alone no longer places a query in the set', "or g_short_labels[i] == 'short')", "or False)", 'kill'),
    ('bootstrap-extra-draw', '05 bootstrap stream', 'one extra RNG draw per resample', "        index = [rng.randrange(n) for _ in range(n)]", "        index = [rng.randrange(n) for _ in range(n)]; rng.random()", 'kill'),
    ('bootstrap-per-comparison-stream', '-', 'independent stream per comparison', "        index = [rng.randrange(n) for _ in range(n)]", "        index = [random.Random(seed + _).randrange(n) for _ in range(n)]", 'kill'),
    ('bootstrap-default-n', '06 bootstrap defaults', 'BOOT_N changed', "BOOT_N = 10000", "BOOT_N = 9999", 'kill'),
    ('bootstrap-default-seed', '06 bootstrap defaults', 'BOOT_SEED changed', "BOOT_SEED = 20261014", "BOOT_SEED = 20261015", 'kill'),
    ('bootstrap-call-n', '06 bootstrap defaults', 'run() passes a different resample count', "paired_table(main_vectors)", "paired_table(main_vectors, 5000)", 'kill'),
    ('bootstrap-call-seed', '06 bootstrap defaults', 'run() passes a different seed', "paired_table(main_vectors)", "paired_table(main_vectors, 10000, 1)", 'kill'),
    ('attribution-flip', '07 attribution', 'cells use routes NOT containing the target', "if target in entered)", "if target not in entered)", 'kill'),
    ('g-indep-flip', '08 G-independent hits', 'P-only instead of G-only', "and queries[i]['target'] in results[i]['entered'][g] and queries[i]['target'] not in results[i]['entered'][p])", "and queries[i]['target'] not in results[i]['entered'][g] and queries[i]['target'] in results[i]['entered'][p])", 'kill'),
    ('overlap-denominator', '09 overlap', 'second denominator = quota', "over_b.append(inter / len(r['entered'][b]))", "over_b.append(inter / q)", 'kill'),
    ('coverage-set', '10 coverage', 'pool coverage built from final lists', "            pool_items.update(entered)", "            pool_items.update(r['final'])", 'kill'),
    ('median', '11 summarize median', 'even-length median takes the upper middle', "    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2", "    median = ordered[mid]", 'kill'),
    ('g-short-threshold', '12 g_short', 'a list of exactly 100 counts as short', "'short' if len(routes[('G', s)][i]) < 100 else 'full'", "'short' if len(routes[('G', s)][i]) <= 100 else 'full'", 'kill'),
    ('g-short-seed', '12 g_short', 'rows of every seed use seed 42 G lengths', "gs = g_short.get(seed)", "gs = g_short.get(42)", 'kill'),
    ('extra-rows', '13 seed!=42 rows', 'reproduction seeds also produce C2/C3/C6/C7', "    if seed == 42:", "    if True:", 'kill'),
    ('vs-c4-uses-c1', '14 vs_C4 reference', 'vs_C4 computed against C1', "cell['vs_C4'] = new_lost(vectors[(seed, name)], c4, idx) if c4 else None", "cell['vs_C4'] = new_lost(vectors[(seed, name)], c1, idx) if c1 else None", 'kill'),
    ('not-run-dropped', '15 NOT_RUN seeds', 'missing-seed reason is not recorded', "            not_run[f'seed{seed}'] =", "            _unused = ", 'kill'),
    ('not-run-paired-empty', '15 NOT_RUN paired', 'paired/localization written as empty dicts when main rows are missing', "        out['paired'] = out['mechanical_conditions'] = out['localization'] = f'NOT_RUN(main-table rows missing: {missing})'", "        out['paired'] = out['mechanical_conditions'] = out['localization'] = {}", 'kill'),
    ('not-run-localization-only', '15 NOT_RUN localization', 'localization silently missing when main rows are missing', "        out['paired'] = out['mechanical_conditions'] = out['localization'] = f'NOT_RUN(main-table rows missing: {missing})'", "        out['paired'] = out['mechanical_conditions'] = f'NOT_RUN(main-table rows missing: {missing})'; out['localization'] = None", 'kill'),
    ('oov-flag', '-', 'OOV non-zero hits flag disabled', "entry['oov_nonzero_hits_anomaly'] = bool(oov_hits)", "entry['oov_nonzero_hits_anomaly'] = False", 'kill'),
    ('script-sha-wrong', '-', 'manifest records a constant instead of the script hash', "'fusion_v2_script_sha256': script_sha256()", "'fusion_v2_script_sha256': '0' * 64", 'kill'),
    ('min-hits-bar-off', '-', 'drop the >=6 hits bar (p < alpha already implies net >= 7)', "    return bool(b - c >= MIN_HITS and mcnemar_exact(b, c) < float(ALPHA))", "    return bool(mcnemar_exact(b, c) < float(ALPHA))", 'equivalent'),
    ('rrf-k', '-', 'k = 61', "RRF_K = 60", "RRF_K = 61", 'kill'),
    ('quota-ceil', '-', 'ceil quota', "return FINAL if n == 1 else B // n", "return FINAL if n == 1 else (B + n - 1) // n", 'kill'),
    ('tie-order', '-', 'ties go to the larger id', "key=lambda item: (-scores[item], item))", "key=lambda item: (-scores[item], -item))", 'kill'),
    ('input-gate-off', '-', 'P/Pop shortfall no longer fails', "        if name != 'G' and len(part) < q:", "        if False:", 'kill'),
    ('dup-only-in-quota', '-', 'duplicate check only inside the quota', "        if len(set(ranked)) != len(ranked):", "        if len(set(part := list(ranked[:q]))) != len(part):", 'kill'),
    ('float-scores', '-', 'float RRF scores', "scores.get(item, Fraction(0)) + Fraction(1, RRF_K + rank)", "scores.get(item, 0.0) + 1.0 / (RRF_K + rank)", 'kill'),
    ('rank-from-zero', '-', 'rank starts at 0', "enumerate(part, 1):", "enumerate(part, 0):", 'kill'),
    ('g-padded', '-', 'short G list padded', "        taken.append(part); filled.append(len(part))", "        taken.append(part + [-1] * (q - len(part))); filled.append(len(part))", 'kill'),
    ('rank-floor', '-', 'floor instead of ceil nearest rank', "    rank = math.ceil(Fraction(p) * len(sorted_values))", "    rank = math.floor(Fraction(p) * len(sorted_values))", 'kill'),
    ('mcnemar-tail', '-', 'tail loses one term', "tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))", "tail = sum(math.comb(n, i) for i in range(min(b, c)))", 'kill'),
    ('p-condition-off', '-', 'significance condition dropped', "    return bool(b - c >= MIN_HITS and mcnemar_exact(b, c) < float(ALPHA))", "    return bool(b - c >= MIN_HITS)", 'kill'),
    ('localization-share-strict', '-', 'share bar strict', "2 * in_set >= total)", "2 * in_set > total)", 'kill'),
    ('localization-total-off', '-', 'total<=0 allowed', "bool(total > 0 and in_set >= MIN_HITS", "bool(in_set >= MIN_HITS", 'kill'),
    ('localization-oov-in-set', '-', 'OOV queries can enter the set', "if not q['oov'] and (q['support']", "if (q['support']", 'kill'),
    ('ndcg-rank', '-', 'N@50 rank offset', "ndcg += 1.0 / math.log2(rank + 1)", "ndcg += 1.0 / math.log2(rank)", 'kill'),
]


def run_suite(workdir):
    cache = tempfile.mkdtemp(prefix='mut-pyc-')
    env = {**os.environ, 'PYTHONPYCACHEPREFIX': cache, 'PYTHONDONTWRITEBYTECODE': '1'}
    proc = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'tests_g5.test_fusion_v2'], cwd=workdir, env=env, capture_output=True, text=True)
    failed = sorted(set(re.findall(r'^(?:FAIL|ERROR): (\w+) ', proc.stderr, re.M)))
    return proc.returncode, failed


def main():
    print('# Mutation ledger for src/vkgr_g5/fusion_v2.py\n')
    print('Isolation: every run uses a fresh copy of the sources, `python -B`, `PYTHONDONTWRITEBYTECODE=1` and a unique `PYTHONPYCACHEPREFIX`; no bytecode cache is shared between runs.\n')
    base_dir = Path(tempfile.mkdtemp(prefix='mut-base-'))
    shutil.copytree(ROOT / 'src/vkgr_g5', base_dir / 'src/vkgr_g5'); shutil.copytree(ROOT / 'tests_g5', base_dir / 'tests_g5')
    base_code, failed = run_suite(base_dir)
    code = base_code
    print(f'Baseline (unmutated copy): returncode {base_code}, failing tests {failed}\n')
    print('| id | review group | change | expected | result | killed by |\n|---|---|---|---|---|---|')
    bad = 0
    for mid, group, text, old, new, expected in MUTANTS:
        work = Path(tempfile.mkdtemp(prefix='mut-'))
        shutil.copytree(ROOT / 'src/vkgr_g5', work / 'src/vkgr_g5'); shutil.copytree(ROOT / 'tests_g5', work / 'tests_g5')
        target = work / TARGET; source = target.read_text()
        if source.count(old) < 1:
            print(f'| {mid} | {group} | {text} | {expected} | PATTERN_NOT_FOUND | |'); bad += 1; continue
        target.write_text(source.replace(old, new, 1))
        code, failed = run_suite(work)
        result = 'killed' if code != 0 else 'SURVIVED'
        if (expected == 'kill') != (result == 'killed'):
            bad += 1
        print(f'| {mid} | {group} | {text} | {expected} | {result} | {", ".join(failed[:3]) or "-"} |')
    print(f'\nMutants: {len(MUTANTS)}; unexpected outcomes: {bad}.')
    return 1 if bad or base_code else 0


if __name__ == '__main__':
    raise SystemExit(main())
