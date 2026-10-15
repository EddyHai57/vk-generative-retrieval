#!/usr/bin/env python3
"""FEAT-04a mechanical freeze tool, one gate per variant (A and B are frozen independently; a later B freeze never touches A's record).

--write --variant A|B --worktree W --plan PLAN --card CARD --record RECORD --review-verdict V --baseline-json JSON
    needs a clean tree and a card/record that are git-ignored; writes RECORD (json) with the commit, the sha256 of the untouched-guard test,
    the sha256 of the variant's three configs, the plan sha256, the pre-registered decision bullets and the review verdict.
--check  (same arguments, no verdict/baseline needed)  exit 0 only if RECORD equals git and the code at HEAD.
Checked on both: clean tree; base reachable and only 'A' lines in `git diff --name-status --no-renames BASE..HEAD`; the variant's configs
validate with ``validate_config(synthetic=False)``, differ from each other in `seed` only, and (seed 42) equal the P-HP-01 H1 config except
schema / task_id / feature with dim / knob dropped; the plan file hashes to PLAN_SHA256 and the three decision bullets of the card are
byte-equal to the first three bullets of plan section 1.2; card and record are git-ignored.
Shas, the commit and the time come from git and from the imported code, never from typing. Nothing here contains a local path.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BASE = '3463846c8774e795eecef57c686a8793402af9da'
PLAN_SHA256 = '9121286e463f1a27ec15185f026e289d8c01790d0eb6177d537a38e134a384e0'
UNTOUCHED_TEST = 'tests/test_feat04a_untouched.py'
H1_CONFIG = 'configs/php01/p-php01-h1-t010-s42.json'
SEEDS = (42, 2024, 2025)
VARIANTS = {'A': ('f4_recency_pool', 'p-feat04a-a-q2t010-s%d.json'), 'B': ('f4_hist_author_content', 'p-feat04a-b-f6hist-s%d.json')}
BULLET_PREFIXES = ('- 有效：', '- 转正（成为新 anchor 的一部分）：', '- inconclusive：')
VERDICTS = ('APPROVED', 'APPROVED_WITH_NITS')


def fail(message):
    print('FREEZE-ERROR: ' + message, file=sys.stderr)
    raise SystemExit(1)


def git(worktree, *args, check=True):
    done = subprocess.run(['git', '-C', str(worktree), *args], capture_output=True, text=True)
    if check and done.returncode != 0:
        fail('git %s failed: %s' % (' '.join(args), done.stderr.strip()))
    return done


def git_blob(worktree, rel):
    raw = subprocess.run(['git', '-C', str(worktree), 'show', 'HEAD:' + rel], capture_output=True)
    if raw.returncode != 0:
        fail('file not committed at HEAD: ' + rel)
    return raw.stdout


def decision_bullets(text):
    """The three pre-registered bullets (exact lines) of a markdown text: the first line starting with each prefix."""
    lines = text.splitlines(); found = []
    for prefix in BULLET_PREFIXES:
        hits = [line for line in lines if line.startswith(prefix)]
        if not hits:
            fail('decision bullet missing: ' + prefix)
        found.append(hits[0])
    return found


def check_ignored(path):
    path = Path(path)
    if git(path.parent, 'check-ignore', '-q', str(path.name), check=False).returncode != 0:
        fail('not git-ignored (it holds local and server paths): ' + path.name)


def only_additions(worktree):
    if git(worktree, 'cat-file', '-e', BASE + '^{commit}', check=False).returncode != 0:
        fail('base commit not reachable')
    bad = [line for line in git(worktree, 'diff', '--name-status', '--no-renames', BASE + '..HEAD').stdout.splitlines() if line.strip() and not line.startswith('A\t')]
    if bad:
        fail('non-additive changes since base: %s' % bad)


def config_facts(worktree, variant):
    feature, pattern = VARIANTS[variant]
    sys.path.insert(0, str(Path(worktree) / 'src'))
    from vkgr import retrieval_feat04a_run as run
    h1 = json.loads(git_blob(worktree, H1_CONFIG))
    values, facts = {}, {}
    for seed in SEEDS:
        rel = 'configs/feat04a/' + pattern % seed
        raw = git_blob(worktree, rel); value = json.loads(raw)
        run.validate_config(value, synthetic=False)
        if value['feature'] != feature or value['seed'] != seed:
            fail('config feature/seed mismatch: ' + rel)
        values[seed] = value; facts[rel] = hashlib.sha256(raw).hexdigest()
    if any({k: v for k, v in values[s].items() if k != 'seed'} != {k: v for k, v in values[42].items() if k != 'seed'} for s in SEEDS):
        fail('configs differ in more than the seed')
    expected = {k: v for k, v in h1.items() if k not in ('dim', 'knob')}
    expected.update(schema=run.SCHEMA, task_id=run.TASK, feature=feature, seed=42)
    if values[42] != expected:
        fail('seed-42 config is not the H1 config with only schema/task_id/feature changed and dim/knob dropped')
    return facts


def code_facts(worktree, variant):
    sys.path.insert(0, str(Path(worktree) / 'src'))
    import torch
    from vkgr import retrieval_feat04a as m4
    feature = VARIANTS[variant][0]
    content = torch.zeros(9, 64)
    kwargs = {}
    if feature == 'f4_hist_author_content':
        kwargs = {'author_index': torch.tensor([0, 1, 2, 3, 2, 3, 1, 2, 3], dtype=torch.long), 'vocab_size': 2}
        content = torch.nn.functional.normalize(torch.randn(9, 64, generator=torch.Generator().manual_seed(1)), dim=1); content[0] = 0
    model = m4.make_feat04a_model(feature, content, [1, 2, 3, 4], 50, 42, **kwargs)
    return {'feature': feature, 'bucket_spec4a_sha256': m4.bucket_spec4a_sha256(feature), 'params_small_fixture': m4.parameter_report4a(model)}


def gather(args):
    worktree = args.worktree
    if git(worktree, 'status', '--porcelain').stdout.strip():
        fail('worktree is not clean (commit or stash first)')
    only_additions(worktree)
    for path in (args.card, args.record):
        check_ignored(path)
    plan = Path(args.plan).read_bytes()
    if hashlib.sha256(plan).hexdigest() != PLAN_SHA256:
        fail('plan file sha256 differs from the frozen reference')
    section = plan.decode('utf-8').split('### 1.2', 1)
    if len(section) != 2:
        fail('plan section 1.2 not found')
    plan_bullets = decision_bullets(section[1].split('### 1.3', 1)[0])
    if decision_bullets(Path(args.card).read_text(encoding='utf-8')) != plan_bullets:
        fail('card decision bullets differ from plan section 1.2')
    return {'schema': 'feat04a-freeze-v1', 'variant': args.variant, 'status': 'FROZEN', 'commit': git(worktree, 'rev-parse', 'HEAD').stdout.strip(), 'base_commit': BASE,
            'untouched_test_sha256': hashlib.sha256(git_blob(worktree, UNTOUCHED_TEST)).hexdigest(), 'plan_sha256': PLAN_SHA256, 'decision_bullets': plan_bullets,
            'configs': config_facts(worktree, args.variant), 'code': code_facts(worktree, args.variant)}


def write(args):
    if args.review_verdict not in VERDICTS:
        fail('review verdict must be one of %s' % (VERDICTS,))
    try:
        baseline = json.loads(args.baseline_json or '')
    except ValueError:
        fail('--baseline-json must be json')
    facts = gather(args)
    facts.update(review_verdict=args.review_verdict, baseline=baseline, freeze_time_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
    Path(args.record).write_text(json.dumps(facts, indent=1, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'written', 'variant': args.variant, 'commit': facts['commit'], 'configs': facts['configs']}, sort_keys=True))
    return 0


def check(args):
    record = json.loads(Path(args.record).read_text(encoding='utf-8'))
    facts = gather(args)
    errors = [k for k, v in facts.items() if record.get(k) != v]
    if record.get('review_verdict') not in VERDICTS or not record.get('baseline') or not record.get('freeze_time_utc'):
        errors.append('review_verdict/baseline/freeze_time_utc')
    if errors:
        for item in errors:
            print('CHECK-FAIL: ' + item, file=sys.stderr)
        return 1
    print(json.dumps({'status': 'frozen_ok', 'variant': args.variant, 'commit': facts['commit'], 'config_sha256': facts['configs']}, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true'); mode.add_argument('--check', action='store_true')
    parser.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    for name in ('worktree', 'plan', 'card', 'record'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--review-verdict'); parser.add_argument('--baseline-json')
    args = parser.parse_args(argv)
    return write(args) if args.write else check(args)


if __name__ == '__main__':
    raise SystemExit(main())
