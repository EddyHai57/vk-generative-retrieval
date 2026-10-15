#!/usr/bin/env python3
"""P-HP-02 mechanical freeze tool (same shape as the FEAT-03 one, no code imports: only git and the card text).
--write --worktree W --card CARD   needs a clean tree; reads HEAD and the six config blobs (git show HEAD:<path>, never the working
                                   tree) and fills the block between the php02-freeze markers. Nothing outside the markers is touched.
--check --worktree W --card CARD   exit 0 only if the card block equals git at HEAD, the thresholds in the block equal THRESHOLDS below,
                                   and the card's section 5 (the pre-registered rule text) states the same thresholds and control hits.
Shas and the commit are never typed by hand. The card is public: nothing written here contains local paths, IPs or company names.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

BEGIN = '<!-- php02-freeze:begin -->'
END = '<!-- php02-freeze:end -->'
FENCE = '```json php02-freeze\n'
CONFIG_DIR = 'configs/php02'
SEEDS = (42, 2024, 2025)
VARIANTS = (('T005', 0.05, 't005'), ('T007', 0.07, 't007'))
# Pre-registered (decision_log 2026-10-15 15:2x item 2; card section 5). Editing after a result exists is a post-hoc change.
THRESHOLDS = {'effective_mean_dhits': 6, 'effective_min_seeds': 2, 'effective_seed_dhits': 6, 'harmful_mean_dhits': -6}
CONTROL_HITS = {'42': 439, '2024': 434, '2025': 444}      # F4-T010 (= P-HP-01 H1) validation hits per seed


def fail(message):
    print('FREEZE-ERROR: ' + message, file=sys.stderr)
    raise SystemExit(1)


def git(worktree, *args):
    done = subprocess.run(['git', '-C', str(worktree), *args], capture_output=True, text=True)
    if done.returncode != 0:
        fail('git %s failed: %s' % (' '.join(args), done.stderr.strip()))
    return done.stdout


def clean_tree(worktree):
    if git(worktree, 'status', '--porcelain').strip():
        fail('worktree is not clean (commit or stash first)')


def config_facts(worktree):
    configs = {}
    for knob, temperature, tag in VARIANTS:
        for seed in SEEDS:
            name = 'p-php02-%s-s%d' % (tag, seed)
            rel = '%s/%s.json' % (CONFIG_DIR, name)
            raw = subprocess.run(['git', '-C', str(worktree), 'show', 'HEAD:' + rel], capture_output=True)
            if raw.returncode != 0:
                fail('config not committed at HEAD: ' + rel)
            value = json.loads(raw.stdout)
            if (value.get('knob'), value.get('temperature'), value.get('seed'), value.get('schema'), value.get('task_id')) != (knob, temperature, seed, 'p-php02-run-v1', 'P-HP-02'):
                fail('config content does not match its name: ' + rel)
            configs[name] = {'path': rel, 'knob': knob, 'temperature': temperature, 'seed': seed, 'sha256': hashlib.sha256(raw.stdout).hexdigest()}
    return configs


def gather(worktree):
    return {'schema': 'php02-freeze-v1', 'status': 'FROZEN', 'commit': git(worktree, 'rev-parse', 'HEAD').strip(),
            'configs': config_facts(worktree), 'thresholds': THRESHOLDS, 'control_hits': CONTROL_HITS}


def render(facts):
    lines = [BEGIN, '状态：**FROZEN**。代码 commit（`php02/p` 的 HEAD）：`%s`。本节由 `scripts/php02_freeze.py --write` 从 git 机械生成。' % facts['commit'], '',
             '- 6 个训练配置（run_id = 文件名，sha256 取自 `git show HEAD:<path>`）：']
    for name, entry in facts['configs'].items():
        lines.append('  - `%s`：温度 %s，seed %d，sha256 `%s`' % (entry['path'], entry['temperature'], entry['seed'], entry['sha256']))
    t = facts['thresholds']
    lines += ['- 阈值（Δhits）：均值 ≥ %d 且至少 %d 个 seed ≥ %d 为有效；均值 ≤ %d 为有害。' % (t['effective_mean_dhits'], t['effective_min_seeds'], t['effective_seed_dhits'], t['harmful_mean_dhits']),
              '- 对照 F4-T010 hits：seed 42 = %d，seed 2024 = %d，seed 2025 = %d。' % tuple(facts['control_hits'][str(s)] for s in SEEDS), '',
              FENCE + json.dumps(facts, indent=1, sort_keys=True, ensure_ascii=False), '```', END]
    return '\n'.join(lines)


def split_card(text):
    """(before, inner, after) around the single marker pair. Only a marker that is a whole line counts (prose may mention them)."""
    begins = [m for m in re.finditer(r'^' + re.escape(BEGIN) + r'[ \t]*$', text, re.M)]
    ends = [m for m in re.finditer(r'^' + re.escape(END) + r'[ \t]*$', text, re.M)]
    if len(begins) != 1 or len(ends) != 1 or begins[0].start() > ends[0].start():
        fail('card markers are not exactly one begin line followed by one end line')
    return text[:begins[0].start()], text[begins[0].end():ends[0].start()], text[ends[0].end():]


def machine_block(inner):
    start = inner.find(FENCE)
    if start < 0 or inner.count(FENCE) != 1:
        fail('card block has no php02-freeze json block (run --write first)')
    stop = inner.rfind('\n```')
    try:
        return json.loads(inner[start + len(FENCE):stop])
    except ValueError:
        fail('card block json is not parseable')


def section5(text):
    start = text.find('\n## 5.')
    if start < 0:
        fail('card has no section 5')
    stop = text.find('\n## ', start + 1)
    return text[start:stop if stop > 0 else len(text)]


def card_rules(text):
    """Thresholds and control hits as the card's section 5 states them."""
    sec = section5(text)
    eff = re.search(r'三 seed 均值 Δhits ≥ (\d+) 且至少 (\d+) 个 seed Δhits ≥ (\d+)', sec)
    harm = re.search(r'均值 Δhits ≤ [−-](\d+)', sec)
    hits = re.search(r'F4-T010[^\n]*?hits\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)', sec)
    if not (eff and harm and hits):
        fail('card section 5 does not state the thresholds / control hits in the expected wording')
    thresholds = {'effective_mean_dhits': int(eff.group(1)), 'effective_min_seeds': int(eff.group(2)), 'effective_seed_dhits': int(eff.group(3)),
                  'harmful_mean_dhits': -int(harm.group(1))}
    return thresholds, dict(zip(map(str, SEEDS), map(int, hits.groups())))


def write(args):
    clean_tree(args.worktree)
    facts = gather(args.worktree)
    card = Path(args.card)
    text = card.read_text(encoding='utf-8')
    before, _, after = split_card(text)
    before_mtime = card.stat().st_mtime_ns
    new = before + render(facts) + after
    tmp = card.with_name(card.name + '.tmp')
    if card.stat().st_mtime_ns != before_mtime:
        fail('card changed while writing (another writer?); re-run')
    tmp.write_text(new, encoding='utf-8')
    tmp.replace(card)
    print(json.dumps({'status': 'written', 'commit': facts['commit']}))
    return 0


def check(args):
    clean_tree(args.worktree)
    head = git(args.worktree, 'rev-parse', 'HEAD').strip()
    text = Path(args.card).read_text(encoding='utf-8')
    _, inner, _ = split_card(text)
    block = machine_block(inner)
    errors = []
    if block.get('commit') != head or head not in inner:
        errors.append('card commit %s != HEAD %s' % (block.get('commit'), head))
    if block.get('status') != 'FROZEN' or '**FROZEN**' not in inner:
        errors.append('FROZEN missing')
    configs = config_facts(args.worktree)
    if block.get('configs') != configs:
        errors.append('configs differ between card block and git HEAD')
    for name, entry in configs.items():
        if entry['sha256'] not in inner or entry['path'] not in inner:
            errors.append('config %s sha256/path missing from the block text' % name)
    if block.get('thresholds') != THRESHOLDS:
        errors.append('thresholds in the card block differ from the freeze script')
    if block.get('control_hits') != CONTROL_HITS:
        errors.append('control hits in the card block differ from the freeze script')
    thresholds, hits = card_rules(text)
    if thresholds != THRESHOLDS:
        errors.append('card section 5 thresholds %s differ from the freeze script %s' % (thresholds, THRESHOLDS))
    if hits != CONTROL_HITS:
        errors.append('card section 5 control hits %s differ from the freeze script %s' % (hits, CONTROL_HITS))
    if errors:
        for item in errors:
            print('CHECK-FAIL: ' + item, file=sys.stderr)
        return 1
    print(json.dumps({'status': 'frozen_ok', 'commit': head, 'config_sha256': {k: v['sha256'] for k, v in configs.items()}}, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--worktree', required=True)
    parser.add_argument('--card', required=True)
    args = parser.parse_args(argv)
    return write(args) if args.write else check(args)


if __name__ == '__main__':
    raise SystemExit(main())
