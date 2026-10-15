#!/usr/bin/env python3
"""FEAT-04b two-stage mechanical freeze tool (shape of scripts/feat03_freeze.py; records are append-only, every fact comes from git or from the imported code).

--write --stage N ...   needs a clean tree; gathers the facts at HEAD, writes the stage record (exclusive create, never rewritten) and appends the stage
                        block to the card (stage 1 -> section 10, stage 2 -> section 11) between the feat04b-freeze markers.
--check --stage N ...   exit 0 only if the record, the card block and the facts gathered at HEAD agree (HEAD must be the stage commit).
Every failure prints one line ``FREEZE-ERROR: ...`` to stderr and exits with code 2.

Batches (the only legal stage-1 values): cd = C + D (6 configs), e = E (3 configs). Stage 2 takes the other batch. The tool never reads the test split and
never looks at a result: it only hashes files, runs synthetic CPU runs for the behaviour pins and compares texts.

Arguments (paths outside the repository are passed in, nothing is guessed):
  --worktree W --card CARD --record RECORD --plan PLAN                  (all stages)
  --batch {cd,e}                                                        (--write; --check reads it from the record)
  --review REVIEW_JSON                                                  (needs "verdict": "APPROVED" and "commit"; written into the record)
  --decision-entry '<YYYY-MM-DD HH:Mx 第N条>' --decision-log LOG        (stage 1; the N-th item of a matching section must speak of C9 as a report item)
  --stage1-record RECORD1                                               (stage 2)
  --audit-receipt RECEIPT --stats-commit S                              (batch cd; S = commit that was deployed for prepare-stats)
  --k10 K10_JSON                                                        (batch e; {"feat04a_freeze_sha", "merge_method": "merge"|"cherry-pick"})
  --baseline TEXT                                                       (optional; stored verbatim, e.g. the baseline test counts)
The card is outside the repository; the record and the card may contain paths, the repository files never do.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = 'feat04b-freeze-stage-v1'
BASE = '3463846c8774e795eecef57c686a8793402af9da'
BATCHES = {'cd': ('f4_item_popularity', 'f4_item_quality'), 'e': ('f4_recency_behavior',)}
COMPLEMENT = {'cd': 'e', 'e': 'cd'}
IMPLEMENTED = ('f4_control', 'f4_recency_pool', 'f4_recency_behavior', 'f4_item_popularity', 'f4_item_quality')
CONFIG_TAGS = {'f4_item_popularity': 'c-i4-popfresh', 'f4_item_quality': 'd-i5-quality', 'f4_recency_behavior': 'e-q2-behavior'}
SEEDS = (42, 2024, 2025)
CONFIG_DIR = 'configs/feat04b'
PINNED_FILE = 'configs/feat04b/pinned-files.json'
CONSTANTS_FILE = 'configs/feat04b/stats-bucket-constants.json'
ANCHOR_CONFIG = 'configs/php01/p-php01-h1-t010-s42.json'
AUDIT_SCRIPT = 'scripts/feat04b_audit_constants.py'
ITEM_STATS = 'src/vkgr/item_stats.py'
PIN_OVERRIDES = {'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}            # the behaviour pin is a short synthetic CPU run of the variant
EDGE_NAMES = {'expo': 'EXPO_EDGES', 'first': 'FIRST_EDGES', 'ratio': 'RATIO_EDGES', 'like': 'LIKE_EDGES'}
AC16_FIELDS = ('sum_likes', 'n_train_examples', 'sum_likes_minus_n_train_examples', 'likes_vs_target_freq', 'timespent', 'duration', 'stats_self_inclusion',
               'p_like_report_threshold', 'p_like_exceeds_threshold', 'logq_redundancy', 'grids', 'like_p0', 'stats_sha256')
THRESHOLD_RE = re.compile(r'P_LIKE_REPORT_THRESHOLD\s*=\s*(0?\.\d+|\d+\.\d+)')
ENTRY_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\w{2})\s*第\s*(\d+)\s*条$')
HEX40 = re.compile(r'^[0-9a-f]{40}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')


def fail(message):
    print('FREEZE-ERROR: ' + message, file=sys.stderr)
    raise SystemExit(2)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


# ------------------------------------------------------------------ git
def git(worktree, *args, check=True):
    done = subprocess.run(['git', '-C', str(worktree), *args], capture_output=True, text=True)
    if check and done.returncode != 0:
        fail('git %s failed: %s' % (' '.join(args), done.stderr.strip()))
    return done


def head_of(worktree):
    return git(worktree, 'rev-parse', 'HEAD').stdout.strip()


def clean_tree(worktree):
    if git(worktree, 'status', '--porcelain').stdout.strip():
        fail('worktree is not clean (commit or stash first)')


def git_blob(worktree, rel, rev='HEAD'):
    done = subprocess.run(['git', '-C', str(worktree), 'show', '%s:%s' % (rev, rel)], capture_output=True)
    if done.returncode != 0:
        fail('file not committed at %s: %s' % (rev, rel))
    return done.stdout


def is_ancestor(worktree, older, newer='HEAD'):
    return git(worktree, 'merge-base', '--is-ancestor', older, newer, check=False).returncode == 0


def only_added(worktree):
    """Nothing but additions between the base commit and HEAD (the old entry points stay bit-for-bit)."""
    if git(worktree, 'cat-file', '-e', BASE + '^{commit}', check=False).returncode != 0:
        fail('base commit %s is not reachable in this worktree' % BASE)
    bad = [line for line in git(worktree, 'diff', '--name-status', '--no-renames', '%s..HEAD' % BASE).stdout.splitlines() if line.strip() and not line.startswith('A\t')]
    if bad:
        fail('non-additive changes since the base commit: %s' % bad[:5])


def pinned_facts(worktree):
    doc = json.loads(git_blob(worktree, PINNED_FILE))
    if doc.get('schema') != 'p-feat04b-pinned-files-v1' or doc.get('base') != BASE or not isinstance(doc.get('files'), dict) or not doc['files']:
        fail('pinned-files.json is malformed')
    for rel, digest in sorted(doc['files'].items()):
        if sha256_bytes(git_blob(worktree, rel)) != digest:
            fail('pinned file changed: ' + rel)
    return {'file': PINNED_FILE, 'sha256': sha256_bytes(git_blob(worktree, PINNED_FILE)), 'files': len(doc['files'])}


# ------------------------------------------------------------------ configs
def config_path(variant, seed):
    return '%s/p-feat04b-%s-s%d.json' % (CONFIG_DIR, CONFIG_TAGS[variant], seed)


def config_facts(worktree, variants):
    """Each real config at HEAD: registered fields, float temperature 0.1 / learning_rate 0.001, and different from the anchor H1 config only in schema, task_id, feature, seed."""
    anchor = json.loads(git_blob(worktree, ANCHOR_CONFIG))
    anchor = {k: v for k, v in anchor.items() if k not in ('dim', 'knob')}
    out = {}
    for variant in variants:
        out[variant] = {}
        for seed in SEEDS:
            rel = config_path(variant, seed)
            raw = git_blob(worktree, rel)
            value = json.loads(raw)
            if value.get('schema') != 'p-feat04b-run-v1' or value.get('task_id') != 'FEAT-04B' or value.get('feature') != variant or value.get('seed') != seed:
                fail('config schema/task_id/feature/seed mismatch: ' + rel)
            for key, expected in (('temperature', 0.1), ('learning_rate', 0.001)):
                if type(value.get(key)) is not float or value[key] != expected:
                    fail('%s: %s must be the float %s' % (rel, key, expected))
            if set(value) != set(anchor):
                fail('%s: field set differs from the anchor config' % rel)
            for key in anchor:
                if key not in ('schema', 'task_id', 'feature', 'seed') and (anchor[key] != value[key] or type(anchor[key]) is not type(value[key])):
                    fail('%s: %s differs from the anchor config' % (rel, key))
            out[variant][str(seed)] = {'path': rel, 'sha256': sha256_bytes(raw)}
    return out


# ------------------------------------------------------------------ card, plan, decision log
def section7_of(card_text):
    """The text from the '## 7.' heading to the next level-2 heading, and its 7.1 bullets (the '> - ' lines of '### 7.1')."""
    lines = card_text.split('\n')
    start = next((i for i, line in enumerate(lines) if re.match(r'^## 7[.\s]', line)), None)
    if start is None:
        fail('card has no section 7')
    stop = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('## ')), len(lines))
    body = lines[start:stop]
    first = next((i for i, line in enumerate(body) if re.match(r'^### 7\.1', line)), None)
    if first is None:
        fail('card section 7 has no 7.1')
    last = next((i for i in range(first + 1, len(body)) if body[i].startswith('### ')), len(body))
    bullets = [line[2:] for line in body[first:last] if line.startswith('> - ')]
    return '\n'.join(body).strip('\n'), bullets


def plan_bullets(plan_text):
    lines = plan_text.split('\n')
    start = next((i for i, line in enumerate(lines) if line.startswith('### 1.2')), None)
    if start is None:
        fail('plan has no section 1.2')
    stop = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('### ')), len(lines))
    return [line for line in lines[start + 1:stop] if line.startswith('- ')]


def plan_text_facts(plan_path, section7_bullets):
    """guard_freeze_plan_text: the four section 7.1 bullets of the card are the plan section 1.2 bullets, character for character."""
    raw = Path(plan_path).read_bytes()
    plan = plan_bullets(raw.decode('utf-8'))
    if len(plan) != 4 or len(section7_bullets) != 4 or plan != section7_bullets:
        fail('card section 7.1 differs from plan section 1.2 (four bullets, verbatim)')
    return {'plan_sha256': sha256_bytes(raw), 'bullets': 4, 'verbatim_equal': True}


def decision_entry_facts(log_path, entry):
    """Stage 1 needs the orchestrator's correction entry (C9 is a report item, not a gate) to exist in decision_log.md: a '## ' heading that starts with the entry's
    date and time, whose N-th numbered item speaks of C9 and of a report item."""
    match = ENTRY_RE.match(entry or '')
    if not match:
        fail("--decision-entry must look like '2026-10-15 16:3x 第2条'")
    stamp, number = match.group(1), match.group(2)
    try:
        text = Path(log_path).read_text(encoding='utf-8')
    except OSError:
        fail('decision log unreadable')
    sections = re.split(r'(?m)^(?=## )', text)
    for section in sections:
        heading = section.split('\n', 1)[0]
        if not heading.startswith('## ' + stamp):
            continue
        for line in section.split('\n')[1:]:
            if re.match(r'^%s\.\s' % number, line) and 'C9' in line and '报告项' in line:
                return {'entry': entry, 'heading': heading[3:], 'item_sha256': sha256_bytes(line.encode('utf-8'))}
    fail('decision entry not found in decision_log.md (needs a heading starting "%s" whose item %s names C9 as a report item): %s' % (stamp, number, entry))


def review_facts(worktree, review_path):
    path = Path(review_path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError):
        fail('review file unreadable')
    if value.get('verdict') != 'APPROVED':
        fail('review verdict is %r, not APPROVED' % value.get('verdict'))
    commit = value.get('commit')
    if not isinstance(commit, str) or not HEX40.match(commit) or not is_ancestor(worktree, commit):
        fail('review commit missing or not an ancestor of HEAD')
    return {'path': path.name, 'sha256': sha256_bytes(raw), 'verdict': 'APPROVED', 'commit': commit}


# ------------------------------------------------------------------ code facts (separate process: the worktree's own src on PYTHONPATH)
def subprocess_env(worktree):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', PYTHONPATH=str(Path(worktree) / 'src'), PYTHONDONTWRITEBYTECODE='1')
    return env


CONSTANTS_PROGRAM = r"""
import json, sys
from vkgr import retrieval_feat04b as m, item_stats
variants = json.loads(sys.argv[1])
out = {'constants': {name: (list(getattr(m, name)) if isinstance(getattr(m, name), tuple) else getattr(m, name))
                     for name in ('EXPO_EDGES', 'FIRST_EDGES', 'RATIO_EDGES', 'LIKE_EDGES', 'LIKE_K', 'LIKE_P0', 'STATS_SHA256')},
       'bucket_spec_sha256': {v: m.bucket_spec_sha256(v) for v in variants},
       'stat_cutoff_week': item_stats.STAT_CUTOFF_WEEK, 'ratio_cap_pm': item_stats.RATIO_CAP_PM}
print(json.dumps(out))
"""


def code_facts(worktree, variants):
    done = subprocess.run([sys.executable, '-B', '-c', CONSTANTS_PROGRAM, json.dumps(list(variants))], capture_output=True, text=True, env=subprocess_env(worktree), cwd=str(worktree))
    if done.returncode != 0:
        fail('could not import the constants from the worktree: ' + (done.stderr.strip().splitlines()[-1] if done.stderr.strip() else 'import failed'))
    return json.loads(done.stdout.strip().splitlines()[-1])


def behavior_pin(worktree, variant):
    """sha256 of the canonical JSON of {records, evaluations} of a short synthetic CPU run of the variant (config = its real seed-42 config with the PIN_OVERRIDES)."""
    config = json.loads(git_blob(worktree, config_path(variant, 42))) if variant in CONFIG_TAGS else None
    if config is None:
        fail('no real config for ' + variant)
    config.update(PIN_OVERRIDES)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / 'config.json'
        cfg.write_text(json.dumps(config))
        done = subprocess.run([sys.executable, '-B', '-m', 'vkgr.retrieval_feat04b_run', 'synthetic-train', '--config', str(cfg), '--output', str(Path(tmp) / 'out')],
                              capture_output=True, text=True, env=subprocess_env(worktree), cwd=str(worktree))
        if done.returncode != 0:
            fail('synthetic run of %s failed: %s' % (variant, done.stderr.strip()[-200:]))
        manifest = json.loads((Path(tmp) / 'out' / 'manifest.json').read_text())
    return sha256_bytes(canonical({'records': manifest['records'], 'evaluations': manifest['evaluations']}).encode('utf-8'))


# ------------------------------------------------------------------ batch specific facts
def audit_facts(worktree, args, section7_text, stage1_section7_sha):
    """Batch cd. Constants JSON <-> audit receipt <-> code constants (the fourth place, the card, is compared by --check); AC16 fields present; the report threshold
    constant of the audit script equals the one printed in the card section 7 and the commit that introduced it is an ancestor of (or equal to) HEAD; item_stats.py is
    byte-identical since the commit that was deployed for prepare-stats."""
    raw = git_blob(worktree, CONSTANTS_FILE)
    constants_doc = json.loads(raw)
    try:
        receipt = json.loads(Path(args.audit_receipt).read_text())
    except (OSError, ValueError, TypeError):
        fail('audit receipt unreadable (--audit-receipt)')
    audit = receipt.get('audit')
    if (receipt.get('terminal') or {}).get('status') != 'succeeded' or not isinstance(audit, dict):
        fail('audit receipt: the audit job did not succeed or carries no audit document')
    missing = [f for f in AC16_FIELDS if f not in audit]
    if missing:
        fail('audit receipt lacks AC16 fields: %s' % missing)
    if audit != constants_doc:
        fail('stats-bucket-constants.json differs from the audit receipt')
    code = code_facts(worktree, ())['constants']
    for name, edges in EDGE_NAMES.items():
        if list(constants_doc['grids'][name]['final_edges']) != code[edges]:
            fail('code %s differs from the audit (%s)' % (edges, name))
    for key, value in (('like_p0', code['LIKE_P0']), ('stats_sha256', code['STATS_SHA256'])):
        if constants_doc[key] != value or value is None:
            fail('code constant differs from the audit: ' + key)
    if constants_doc.get('like_k') != code['LIKE_K']:
        fail('code LIKE_K differs from the audit')
    script = git_blob(worktree, AUDIT_SCRIPT).decode('utf-8')
    code_threshold = THRESHOLD_RE.search(script)
    card_threshold = THRESHOLD_RE.search(section7_text)
    if not code_threshold or not card_threshold or float(code_threshold.group(1)) != float(card_threshold.group(1)):
        fail('P_LIKE_REPORT_THRESHOLD of the audit script differs from the card section 7')
    if float(constants_doc['p_like_report_threshold']) != float(card_threshold.group(1)):
        fail('the audit receipt threshold differs from the card section 7')
    log = git(worktree, 'log', '--reverse', '--format=%H', '-S', 'P_LIKE_REPORT_THRESHOLD', '--', AUDIT_SCRIPT).stdout.split()
    if not log or not is_ancestor(worktree, log[0]):
        fail('the commit that introduced P_LIKE_REPORT_THRESHOLD is not an ancestor of HEAD')
    if not (args.stats_commit and HEX40.match(args.stats_commit) and is_ancestor(worktree, args.stats_commit)):
        fail('--stats-commit must be a commit that is an ancestor of HEAD')
    if git(worktree, 'diff', '--quiet', args.stats_commit, 'HEAD', '--', ITEM_STATS, check=False).returncode != 0:
        fail('src/vkgr/item_stats.py changed since the deployed stats commit')
    if stage1_section7_sha is not None and stage1_section7_sha != sha256_bytes(section7_text.encode('utf-8')):
        fail('card section 7 changed since stage 1 (the threshold must precede the audit)')
    return {'constants_file': CONSTANTS_FILE, 'constants_sha256': sha256_bytes(raw), 'receipt_equals_constants': True, 'ac16_fields_present': True,
            'code_equals_audit': True, 'report_threshold': float(card_threshold.group(1)), 'threshold_commit': log[0],
            'audit_script_sha256': sha256_bytes(script.encode('utf-8')), 'stats_commit': args.stats_commit, 'item_stats_unchanged_since_stats_commit': True,
            'section7_sha256_before_audit': stage1_section7_sha}


def run_e4(worktree):
    """The E4 bridge test (04b chain A against the merged 04a runner) must really run: OK and not skipped."""
    done = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'tests.test_feat04b_run.BridgeE4Tests'], capture_output=True, text=True, env=subprocess_env(worktree), cwd=str(worktree))
    tail = done.stderr.strip().splitlines()
    ran = re.search(r'Ran (\d+) test', done.stderr)
    return {'ok': done.returncode == 0 and bool(tail) and tail[-1].startswith('OK'), 'skipped': 'skipped' in (tail[-1] if tail else ''), 'ran': int(ran.group(1)) if ran else 0}


def e_facts(worktree, args):
    """Batch e: the 04a freeze SHA is part of the history (merged, or cherry-picked with -x) and E4 really ran green on this tree."""
    try:
        raw = Path(args.k10).read_bytes()
        record = json.loads(raw)
    except (OSError, ValueError, TypeError):
        fail('K10 record unreadable (--k10)')
    sha, method = record.get('feat04a_freeze_sha'), record.get('merge_method')
    if not isinstance(sha, str) or not HEX40.match(sha) or method not in ('merge', 'cherry-pick'):
        fail('K10 record needs feat04a_freeze_sha and merge_method (merge | cherry-pick)')
    if method == 'merge' and not is_ancestor(worktree, sha):
        fail('the 04a freeze SHA is not an ancestor of HEAD')
    if method == 'cherry-pick' and ('cherry picked from commit ' + sha) not in git(worktree, 'log', '--format=%B', '%s..HEAD' % BASE).stdout:
        fail('no commit message records a cherry-pick -x of the 04a freeze SHA')
    e4 = run_e4(worktree)
    if not e4['ok'] or e4['skipped'] or e4['ran'] == 0:
        fail('E4 did not run green (ok=%s skipped=%s ran=%s)' % (e4['ok'], e4['skipped'], e4['ran']))
    return {'k10': {'sha256': sha256_bytes(raw), 'feat04a_freeze_sha': sha, 'merge_method': method}, 'e4': e4}


# ------------------------------------------------------------------ gather / render
def load_record(path):
    try:
        raw = Path(path).read_bytes()
        return json.loads(raw), sha256_bytes(raw)
    except (OSError, ValueError, TypeError):
        fail('record unreadable: %s' % path)


def card_block(text, stage):
    begin, end = '<!-- feat04b-freeze:begin stage=%d -->' % stage, '<!-- feat04b-freeze:end stage=%d -->' % stage
    if begin not in text and end not in text:
        return None
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        fail('card markers of stage %d are not exactly one begin followed by one end' % stage)
    head, rest = text.split(begin)
    inner, tail = rest.split(end)
    return head, inner, tail


def machine_block(inner, stage):
    tag = '```json feat04b-freeze-stage%d\n' % stage
    start, stop = inner.find(tag), inner.rfind('\n```')
    if start < 0 or stop < start or inner.count(tag) != 1:
        fail('card block must contain exactly one %s block' % tag.strip())
    return json.loads(inner[start + len(tag):stop])


def gather(args, stage, batch, stage1=None):
    """All facts of the stage at HEAD (no clock, no random numbers): the same input gives the same document. Cheap checks first, the synthetic runs last."""
    worktree = args.worktree
    clean_tree(worktree)
    only_added(worktree)
    variants = BATCHES[batch]
    card_text = Path(args.card).read_text(encoding='utf-8')
    section7, bullets = section7_of(card_text)
    facts = {'schema': SCHEMA, 'stage': stage, 'batch': batch, 'variants': list(variants), 'commit': head_of(worktree), 'base_commit': BASE,
             'configs': config_facts(worktree, variants), 'section7_sha256': sha256_bytes(section7.encode('utf-8')),
             'plan': plan_text_facts(args.plan, bullets), 'pinned': pinned_facts(worktree), 'baseline': args.baseline,
             'review': review_facts(worktree, args.review)}
    if stage == 1:
        facts['decision_entry'] = decision_entry_facts(args.decision_log, args.decision_entry)
    if batch == 'cd':
        facts['audit'] = audit_facts(worktree, args, section7, stage1['section7_sha256'] if stage1 else None)
    else:
        facts.update(e_facts(worktree, args))
    if stage == 2:
        x1 = stage1['commit']
        if not is_ancestor(worktree, x1):
            fail('stage 1 commit is not an ancestor of HEAD')
        if stage1['section7_sha256'] != facts['section7_sha256']:
            fail('card section 7 changed since stage 1')
        diff = git(worktree, 'diff', '--name-status', '--no-renames', '%s..HEAD' % x1).stdout.splitlines()
        frozen = {entry['path'] for per in stage1['configs'].values() for entry in per.values()}
        touched = [line for line in diff if line.split('\t')[0] != 'A' and line.split('\t')[-1] in frozen]
        if touched:
            fail('stage 1 configs changed since stage 1: %s' % touched)
        if config_facts(worktree, stage1['variants']) != stage1['configs']:
            fail('a stage 1 config sha256 changed')
    code = code_facts(worktree, IMPLEMENTED)
    facts['bucket_spec_sha256'] = {v: code['bucket_spec_sha256'][v] for v in variants}
    facts['behavior_pins'] = {v: behavior_pin(worktree, v) for v in variants}
    if stage == 2:
        if {v: code['bucket_spec_sha256'][v] for v in stage1['variants']} != stage1['bucket_spec_sha256']:
            fail('bucket_spec_sha256 of a stage 1 variant changed')
        again = {v: behavior_pin(worktree, v) for v in stage1['variants']}
        if again != stage1['behavior_pins']:
            fail('behaviour pin of a stage 1 variant changed (%s)' % sorted(v for v in again if again[v] != stage1['behavior_pins'][v]))
        facts['stage1'] = {'record_sha256': args.stage1_sha, 'commit': x1, 'behavior_pins_recomputed_equal': True, 'name_status_x1_to_head': diff,
                           'shared_source_modified': sorted(line.split('\t')[-1] for line in diff if line[0] == 'M')}
    return facts


def render(facts, when, record_sha):
    stage = facts['stage']
    title = {1: '## 10. FEAT-04b 阶段 1 冻结记录', 2: '## 11. FEAT-04b 阶段 2 冻结记录'}[stage]
    names = {'cd': 'C + D（6 个配置）', 'e': 'E（3 个配置）'}
    lines = ['<!-- feat04b-freeze:begin stage=%d -->' % stage, title, '',
             '状态：**FROZEN**。冻结时间（UTC）：%s。代码 commit：`%s`。基线：`%s`。批次：%s。' % (when, facts['commit'], BASE, names[facts['batch']]),
             '本节由冻结脚本从 git 与代码机械生成（sha256、commit、常量、行为钉都不是手写）；本节之外的卡内容一字未动，只在末尾追加。', '',
             '- 记录文件 sha256：`%s`。' % record_sha, '- §7 全文 sha256：`%s`（阶段 1 整体冻结，阶段 2 不得改）。' % facts['section7_sha256'],
             '- plan §1.2 逐字比对：四条 bullet 与 §7.1 相等；plan sha256 `%s`。' % facts['plan']['plan_sha256'],
             '- 钉死文件：%d 个，清单 `%s` sha256 `%s`。' % (facts['pinned']['files'], facts['pinned']['file'], facts['pinned']['sha256'])]
    for variant, per in sorted(facts['configs'].items()):
        for seed, entry in sorted(per.items(), key=lambda item: int(item[0])):
            lines.append('- config `%s`（%s，seed %s）sha256 `%s`。' % (entry['path'], variant, seed, entry['sha256']))
    for variant in facts['variants']:
        lines.append('- `%s`：bucket_spec_sha256 `%s`；行为钉 `%s`。' % (variant, facts['bucket_spec_sha256'][variant], facts['behavior_pins'][variant]))
    lines.append('- review：verdict %s，review commit `%s`，文件 sha256 `%s`。' % (facts['review']['verdict'], facts['review']['commit'], facts['review']['sha256']))
    if 'decision_entry' in facts:
        lines.append('- decision_log 条目（C9 更正为报告项）：%s（标题：%s）。' % (facts['decision_entry']['entry'], facts['decision_entry']['heading']))
    if 'audit' in facts:
        a = facts['audit']
        lines += ['- 审计常量：`%s` sha256 `%s`；与审计回执、代码常量一致；AC16 字段齐全。' % (a['constants_file'], a['constants_sha256']),
                  '- P_LIKE_REPORT_THRESHOLD = %s（审计脚本与卡 §7 相等）；引入它的 commit `%s` 是本阶段 commit 的祖先或本身；审计脚本 sha256 `%s`。' % (a['report_threshold'], a['threshold_commit'], a['audit_script_sha256']),
                  '- `src/vkgr/item_stats.py` 自 prepare-stats 部署 commit `%s` 起逐字节不变。' % a['stats_commit']]
    if 'k10' in facts:
        lines += ['- K10：04a 冻结 SHA `%s`（%s），记录文件 sha256 `%s`。' % (facts['k10']['feat04a_freeze_sha'], facts['k10']['merge_method'], facts['k10']['sha256']),
                  '- E4 桥接：真实运行 %d 个测试，通过，无 skip。' % facts['e4']['ran']]
    if 'stage1' in facts:
        s = facts['stage1']
        lines += ['- 阶段 1 记录 sha256 `%s`（与卡内登记一致），阶段 1 commit `%s` 是本阶段 commit 的祖先；阶段 1 变体的行为钉、配置 sha256、bucket_spec_sha256 在本 commit 上逐位复核相等。' % (s['record_sha256'], s['commit'])]
    lines += ['', '```json feat04b-freeze-stage%d' % stage, json.dumps(facts, indent=1, sort_keys=True, ensure_ascii=False), '```', '<!-- feat04b-freeze:end stage=%d -->' % stage]
    return '\n'.join(lines)


# ------------------------------------------------------------------ commands
def stage1_context(args, batch):
    """Stage 2 context: the stage 1 record must equal what the card registered, and must be of the complementary batch."""
    record, sha = load_record(args.stage1_record)
    args.stage1_sha = sha
    card_text = Path(args.card).read_text(encoding='utf-8')
    parts = card_block(card_text, 1)
    if parts is None:
        fail('card has no stage 1 block')
    if ('`%s`' % sha) not in parts[1] or machine_block(parts[1], 1) != record:
        fail('stage 1 record sha256 differs from the one registered in the card')
    if record.get('stage') != 1 or record.get('schema') != SCHEMA:
        fail('not a stage 1 record')
    if batch != COMPLEMENT[record['batch']]:
        fail('stage 2 must take the other batch (%s after %s)' % (COMPLEMENT[record['batch']], record['batch']))
    if not is_ancestor(args.worktree, record['commit']):
        fail('stage 1 commit is not an ancestor of HEAD')
    return record


def write_stage(args):
    if args.batch not in BATCHES:
        fail('--batch must be cd or e')
    stage1 = stage1_context(args, args.batch) if args.stage == 2 else None
    if args.stage == 1 and not args.decision_entry:
        fail('--write --stage 1 needs --decision-entry')
    facts = gather(args, args.stage, args.batch, stage1)
    when = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    facts['freeze_time_utc'] = when
    card = Path(args.card)
    text = card.read_text(encoding='utf-8')
    if card_block(text, args.stage) is not None:
        fail('card already has a stage %d block (records are append-only)' % args.stage)
    if args.stage == 2 and card_block(text, 1) is None:
        fail('card has no stage 1 block')
    record = Path(args.record)
    body = (json.dumps(facts, indent=1, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')
    try:
        with open(record, 'xb') as handle:                                              # exclusive: a stage record is never rewritten
            handle.write(body)
    except FileExistsError:
        fail('stage record already exists (append-only): %s' % record)
    before = card.stat().st_mtime_ns
    new = text + ('' if text.endswith('\n') else '\n') + '\n' + render(facts, when, sha256_bytes(body)) + '\n'
    if card.stat().st_mtime_ns != before:
        fail('card changed while writing (another writer?); re-run')
    tmp = card.with_name(card.name + '.tmp')
    tmp.write_text(new, encoding='utf-8')
    tmp.replace(card)
    print(json.dumps({'status': 'written', 'stage': args.stage, 'batch': args.batch, 'commit': facts['commit'], 'record_sha256': sha256_bytes(body)}))
    return 0


def check_stage(args):
    record, record_sha = load_record(args.record)
    if record.get('schema') != SCHEMA or record.get('stage') != args.stage:
        fail('record is not a stage %d record' % args.stage)
    text = Path(args.card).read_text(encoding='utf-8')
    parts = card_block(text, args.stage)
    if parts is None:
        fail('card has no stage %d block' % args.stage)
    if ('`%s`' % record_sha) not in parts[1] or machine_block(parts[1], args.stage) != record:
        fail('card block differs from the stage record (or registers another record sha256)')
    begin, end = '<!-- feat04b-freeze:begin stage=%d -->' % args.stage, '<!-- feat04b-freeze:end stage=%d -->' % args.stage
    if begin + parts[1] + end != render(record, record['freeze_time_utc'], record_sha):
        fail('the text of the card block is not what the record renders to (edited by hand?)')
    stage1 = stage1_context(args, record['batch']) if args.stage == 2 else None
    if head_of(args.worktree) != record['commit']:
        fail('HEAD is not the frozen commit %s' % record['commit'])
    args.decision_entry = (record.get('decision_entry') or {}).get('entry') if args.stage == 1 else None
    args.baseline = record.get('baseline')
    facts = gather(args, args.stage, record['batch'], stage1)
    facts['freeze_time_utc'] = record['freeze_time_utc']
    if facts != record:
        differing = sorted(k for k in set(facts) | set(record) if facts.get(k) != record.get(k))
        fail('facts at HEAD differ from the record: %s' % differing)
    print(json.dumps({'status': 'frozen_ok', 'stage': args.stage, 'batch': record['batch'], 'commit': record['commit'], 'record_sha256': record_sha}, sort_keys=True))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--stage', type=int, choices=(1, 2), required=True)
    parser.add_argument('--worktree', required=True)
    parser.add_argument('--card', required=True)
    parser.add_argument('--record', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--review', required=True)
    parser.add_argument('--batch')
    parser.add_argument('--decision-entry')
    parser.add_argument('--decision-log')
    parser.add_argument('--stage1-record')
    parser.add_argument('--audit-receipt')
    parser.add_argument('--stats-commit')
    parser.add_argument('--k10')
    parser.add_argument('--baseline')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.stage == 2 and not args.stage1_record:
        fail('--stage 2 needs --stage1-record')
    if args.stage == 1 and not args.decision_log:
        fail('--stage 1 needs --decision-log')
    return write_stage(args) if args.write else check_stage(args)


if __name__ == '__main__':
    raise SystemExit(main())
