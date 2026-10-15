"""FEAT-04b FEAT-005: the VRAM smoke script (declaration rule, CPU run on small shapes) and the two-stage freeze script (decision-entry check, append-only stage
records, behaviour pins, plan-text check, audit-threshold rule for C/D in either stage, item_stats unchanged since the deployed commit, K10 + E4 for E, review verdict).
The freeze tests run the script against a throw-away clone of this repository (committed state plus the working-tree files) with fake outside files."""
import contextlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
VRAM_SCRIPT = ROOT / 'scripts' / 'feat04b_vram_smoke.py'
REAL_VARIANTS = ('f4_item_popularity', 'f4_item_quality', 'f4_recency_behavior')


def load_vram():
    spec = importlib.util.spec_from_file_location('feat04b_vram_smoke', VRAM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VramSmokeTests(unittest.TestCase):
    def test_declaration_rule_boundaries(self):
        """declared = max(1024, ceil(measured * 1.15 / 256) * 256): the 1024 floor, the last value that still fits 1024 (890) and the first that does not (891)."""
        module = load_vram()
        for measured, expected in ((0, 1024), (1, 1024), (456, 1024), (890, 1024), (890.4, 1024), (890.5, 1280), (891, 1280), (1113, 1280), (1114, 1536), (1500, 1792), (6000, 6912)):
            self.assertEqual(module.declared_vram_mib(measured), expected, msg=str(measured))
        for measured in range(0, 4000, 37):                                       # never below the floor, always a multiple of 256, always >= measured * 1.15
            declared = module.declared_vram_mib(measured)
            self.assertGreaterEqual(declared, 1024); self.assertEqual(declared % 256, 0); self.assertGreaterEqual(declared, measured * 1.15)

    def test_constants_are_the_full_size_shapes(self):
        module = load_vram()
        self.assertEqual((module.CATALOG, module.SUPPORTED, module.BATCH, module.WIDTH, module.QUERIES, module.TEMPERATURE),
                         (120415, 21496, 512, 50, 2520, 0.10))
        self.assertEqual(module.MIN_VRAM_MIB, 1024)

    def test_no_path_arguments_and_default_variants(self):
        module = load_vram()
        actions = {a.dest: a for a in module.build_parser()._actions} 
        self.assertEqual(set(actions) - {'help'}, {'device', 'variants', 'catalog', 'steps'})
        args = module.build_parser().parse_args([])
        self.assertEqual((args.device, list(args.variants)), ('cuda', list(REAL_VARIANTS)))

    def test_cpu_small_run_reports_every_variant(self):
        module = load_vram(); out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(module.main(['--device', 'cpu', '--catalog', '3000', '--steps', '1']), 0)
        lines = out.getvalue().strip().splitlines(); self.assertEqual(len(lines), 1)
        value = json.loads(lines[0])
        self.assertEqual((value['task'], value['device'], value['catalog'], value['steps'], value['temperature']), ('P-FEAT04B', 'cpu', 3000, 1, 0.10))
        self.assertEqual([r['variant'] for r in value['variants']], list(REAL_VARIANTS))
        for row in value['variants']:
            self.assertTrue(row['finite_loss']); self.assertEqual(row['vram'], 'UNKNOWN'); self.assertGreater(row['ru_maxrss'], 0)
            self.assertNotIn('declared_vram_mib', row)
            self.assertEqual(row['optimizer_state_bytes'], 2 * row['param_bytes'])
            self.assertGreater(row['param_bytes'], 3000 * 64 * 4)                    # at least the item table
        # C adds two bucket tables (len(EXPO_EDGES) + 2 and len(FIRST_EDGES) + 2 rows of 64); E adds 19 scalars (13 recency logits + 6 behaviour weights)
        from vkgr import retrieval_feat04b as m4
        rows = len(m4.EXPO_EDGES) + 2 + len(m4.FIRST_EDGES) + 2
        self.assertEqual(value['variants'][0]['param_bytes'] - value['variants'][2]['param_bytes'], (rows * 64 - 19) * 4)

    def test_declaration_is_taken_from_the_larger_of_the_two_measurements(self):
        module = load_vram()
        self.assertEqual(module.measured_mib(used_mib=900.0, reserved_mib=700.0), 900.0)
        self.assertEqual(module.measured_mib(used_mib=500.0, reserved_mib=850.0), 850.0)



# ================================================================== freeze script
FREEZE_SCRIPT = ROOT / 'scripts' / 'feat04b_freeze.py'
GIT = ['git', '-c', 'user.name=guard', '-c', 'user.email=guard@example.invalid', '-c', 'commit.gpgsign=false']
ENTRY = '2026-10-15 16:3x 第2条'
BULLETS = ['- 有效：三 seed 均值 Δhits ≥ 6（ΔR@50 ≥ 0.002），且至少 2 个 seed 的 Δhits ≥ 6。',
           '- 转正（成为新 anchor 的一部分）：有效，且至少 2 个 seed 逐 seed exact McNemar p < 0.05，且改动了用户塔池化或新增数据的变体（A、B、E、C、D 都算）通过一次独立泄漏 review。',
           '- inconclusive：−6 < 均值 Δhits < 6；有害：均值 Δhits ≤ −6。',
           '- **条件组合（1 个名额）**：C/D/E 中有 ≥2 个“有效”时，跑一次组合（三 seed）；否则不跑。']
PLAN = '# plan\n\n### 1.1 something\n- not a rule\n### 1.2 预注册判定（设计步骤抄进冻结卡，看结果前写死）\n' + '\n'.join(BULLETS) + '\n### 1.3 成本与顺序\n- one\n'
CARD = ('# card\n\n## 1. Intro\n\ntext\n\n## 7. 预注册判定（看结果前写死）\n\n### 7.1 plan §1.2 原文\n\n' + '\n'.join('> ' + b for b in BULLETS) +
        '\n\n### 7.2 澄清\n\n- C9　预设报告提醒；阈值常量 `P_LIKE_REPORT_THRESHOLD = 0.30`。\n\n## 8. 冻结清单\n\nlater\n')
DECISION_LOG = ('# decisions\n\n## 2026-10-15 16:0x：old\n1. one\n3. **C9**（D 的转正附加条件）：旧口径。\n\n## 2026-10-15 16:3x：更正 16:0x 第 3 条——C9 改为报告项（Kiro）\n'
                '1. **更正**：越权。\n2. **现行口径**：C9 只是报告项，D 的转正按字面判定。\n3. 同理，C2 也是报告项。\n')
E4_GREEN = {'ok': True, 'skipped': False, 'ran': 3}
_FREEZE = []


def load_freeze():
    if not _FREEZE:
        spec = importlib.util.spec_from_file_location('feat04b_freeze', FREEZE_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FREEZE.append(module)
    return _FREEZE[0]


def sh(*args, cwd=None):
    done = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError('%s failed: %s' % (args, done.stderr))
    return done.stdout.strip()


def commit_all(repo, message):
    sh(*GIT, '-C', str(repo), 'add', '-A')
    if sh('git', '-C', str(repo), 'status', '--porcelain'):
        sh(*GIT, '-C', str(repo), 'commit', '-q', '-m', message)
    return sh('git', '-C', str(repo), 'rev-parse', 'HEAD')


class Outside:
    """The files that live outside the repository: card, plan, decision log, review, audit receipt, K10 record and the two stage records."""
    def __init__(self, directory):
        self.dir = Path(directory)


class World:
    """A clone of this repository (committed state) with the working-tree files overlaid, the C/D audit constants prepared inside it, and one commit on top."""
    def __init__(self, base_repo=None):
        self.tmp = tempfile.TemporaryDirectory()
        _TEMPS.append(self.tmp)
        self.repo = Path(self.tmp.name) / 'repo'
        self.freeze = load_freeze()
        if base_repo is None:
            sh('git', 'clone', '-q', '--no-hardlinks', str(ROOT), str(self.repo))
            for rel in sh('git', '-C', str(ROOT), 'ls-files', '-co', '--exclude-standard').splitlines():
                if (ROOT / rel).is_file():
                    (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / rel, self.repo / rel)
            self.overlay_commit = commit_all(self.repo, 'overlay of the working tree')
            self.prepare_cd()
            self.head = commit_all(self.repo, 'audit constants prepared')
        else:
            sh('git', 'clone', '-q', '--no-hardlinks', str(base_repo.repo), str(self.repo))
            self.overlay_commit, self.head, self.doc = base_repo.overlay_commit, base_repo.head, base_repo.doc
        self.stats_commit = self.head
        self.cache = {}

    def prepare_cd(self):
        """Make the code constants, the constants JSON and (below) the receipt agree: the real values may not exist yet, so the test world builds a consistent set."""
        code = self.freeze.code_facts(self.repo, ())['constants']
        digest = code['STATS_SHA256'] or '5' * 64
        source = self.repo / 'src/vkgr/retrieval_feat04b.py'
        text = source.read_text()
        if 'STATS_SHA256 = None' in text:
            source.write_text(text.replace('STATS_SHA256 = None', "STATS_SHA256 = '%s'" % digest, 1))
        code = self.freeze.code_facts(self.repo, ())['constants']
        self.doc = {'sum_likes': 100, 'n_train_examples': 90, 'sum_likes_minus_n_train_examples': 10, 'likes_vs_target_freq': {}, 'timespent': {}, 'duration': {},
                    'stats_self_inclusion': {'p_like': 0.1}, 'p_like_report_threshold': 0.3, 'p_like_exceeds_threshold': False, 'logq_redundancy': {},
                    'grids': {name: {'final_edges': code[edges]} for name, edges in self.freeze.EDGE_NAMES.items()},
                    'like_p0': code['LIKE_P0'], 'like_k': code['LIKE_K'], 'stats_sha256': code['STATS_SHA256']}
        (self.repo / 'configs/feat04b/stats-bucket-constants.json').write_text(json.dumps(self.doc, sort_keys=True, separators=(',', ':')))

    def fork(self):
        return World(self)

    def outside(self, **override):
        """Fresh outside files (card, plan, ...). ``override`` replaces the text of the named file (card, plan, log) or the review / receipt / k10 content."""
        directory = Path(tempfile.mkdtemp(dir=self.tmp.name))
        out = Outside(directory)
        out.card, out.plan, out.log = directory / 'card.md', directory / 'plan.md', directory / 'decision_log.md'
        out.card.write_text(override.get('card', CARD)); out.plan.write_text(override.get('plan', PLAN)); out.log.write_text(override.get('log', DECISION_LOG))
        out.review, out.receipt, out.k10 = directory / 'review.json', directory / 'receipt.json', directory / 'k10.json'
        out.review.write_text(json.dumps(override.get('review', {'verdict': 'APPROVED', 'commit': self.head})))
        out.receipt.write_text(json.dumps(override.get('receipt', {'run_id': 'p-feat04b-stats-audit', 'terminal': {'status': 'succeeded'}, 'audit': self.doc})))
        out.k10.write_text(json.dumps(override.get('k10', {'feat04a_freeze_sha': self.overlay_commit, 'merge_method': 'merge'})))
        out.rec1, out.rec2 = directory / 'stage1.json', directory / 'stage2.json'
        return out

    def copy_outside(self, source):
        out = Outside(Path(tempfile.mkdtemp(dir=self.tmp.name)))
        for name in ('card', 'plan', 'log', 'review', 'receipt', 'k10', 'rec1', 'rec2'):
            target = out.dir / getattr(source, name).name
            if getattr(source, name).exists():
                shutil.copyfile(getattr(source, name), target)
            setattr(out, name, target)
        return out

    def argv(self, out, mode, stage, batch=None, **override):
        flags = {'--stage': str(stage), '--worktree': str(self.repo), '--card': str(out.card), '--record': str(out.rec1 if stage == 1 else out.rec2),
                 '--plan': str(out.plan), '--review': str(out.review)}
        if stage == 1:
            flags.update({'--decision-log': str(out.log)})
            if mode == '--write':
                flags['--decision-entry'] = ENTRY
        else:
            flags['--stage1-record'] = str(out.rec1)
        if mode == '--write':
            flags['--batch'] = batch
        if batch == 'cd':
            flags.update({'--audit-receipt': str(out.receipt), '--stats-commit': self.stats_commit})
        else:
            flags['--k10'] = str(out.k10)
        flags.update(override)
        args = [mode]
        for key, value in flags.items():
            if value is not None:
                args += [key, value]
        return args

    def run(self, argv, e4=E4_GREEN):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), mock.patch.object(self.freeze, 'run_e4', lambda worktree: dict(e4)):
            try:
                code = self.freeze.main(argv)
            except SystemExit as stop:
                code = stop.code if isinstance(stop.code, int) else 1
        return code, stdout.getvalue(), stderr.getvalue()

    def write(self, out, stage, batch, **override):
        return self.run(self.argv(out, '--write', stage, batch, **override))

    def check(self, out, stage, batch, **override):
        return self.run(self.argv(out, '--check', stage, batch, **override))

    def stage1_done(self, batch):
        """A copy of a finished stage 1 (memoised: the synthetic pins cost a few seconds)."""
        if batch not in self.cache:
            out = self.outside()
            code, _, err = self.write(out, 1, batch)
            if code != 0:
                raise AssertionError('stage 1 (%s) could not be written: %s' % (batch, err))
            self.cache[batch] = out
        return self.copy_outside(self.cache[batch])


_WORLD = []
_TEMPS = []


def world():
    if not _WORLD:
        _WORLD.append(World())
    return _WORLD[0]


def tearDownModule():
    for temp in _TEMPS:
        temp.cleanup()
    _TEMPS.clear()
    _WORLD.clear()


def error_of(stderr):
    lines = [line for line in stderr.splitlines() if line.startswith('FREEZE-ERROR')]
    return lines[-1] if lines else stderr.strip()[-200:]


# ---------------------------------------------------------------- guards (reused by the M86 mutation tests)

def guard_freeze_stage2_append_only(w):
    """Stage 2 writes its own record and block and nothing else: the stage 1 record file and the stage 1 card block stay byte for byte; a changed stage 1
    record, or a stage 1 commit that is not an ancestor of HEAD, is refused with exit code 2."""
    out = w.stage1_done('cd')
    before_record, before_card = out.rec1.read_bytes(), out.card.read_text()
    code, _, err = w.write(out, 2, 'e')
    if code != 0:
        raise AssertionError('stage 2 write failed: ' + err)
    if out.rec1.read_bytes() != before_record:
        raise AssertionError('stage 2 rewrote the stage 1 record')
    if not out.card.read_text().startswith(before_card):
        raise AssertionError('stage 2 changed the card text before its own block')
    if w.check(out, 2, 'e')[0] != 0:
        raise AssertionError('--check --stage 2 failed on the frozen tree')
    tampered = w.copy_outside(w.stage1_done('cd'))
    tampered.rec1.write_bytes(tampered.rec1.read_bytes() + b' ')
    if w.write(tampered, 2, 'e')[0] != 2:
        raise AssertionError('a changed stage 1 record was accepted')
    other = w.copy_outside(w.stage1_done('cd'))                                    # the record's commit is not an ancestor of HEAD
    record = json.loads(other.rec1.read_text())
    record['commit'] = '0' * 40
    body = (json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + '\n').encode()
    old_sha = hashlib_sha(other.rec1.read_bytes())
    other.rec1.write_bytes(body)
    card = other.card.read_text()
    block = card[card.index('```json feat04b-freeze-stage1'):card.index('<!-- feat04b-freeze:end stage=1 -->')]
    card = card.replace(block, '```json feat04b-freeze-stage1\n' + json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + '\n```\n').replace(old_sha, hashlib_sha(body))
    other.card.write_text(card)
    code, _, err = w.write(other, 2, 'e')
    if code != 2 or 'not an ancestor' not in err:
        raise AssertionError('stage 1 commit that is not an ancestor was accepted: ' + error_of(err))


def hashlib_sha(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def guard_freeze_behavior_pin():
    """A change in the behaviour of a stage 1 variant after stage 1 (E after an E-first stage 1) must stop the stage 2 write with exit code 2."""
    base = world()
    out = base.stage1_done('e')
    fork = base.fork()
    path = fork.repo / 'src/vkgr/retrieval_feat04b.py'
    text = path.read_text()
    changed = text.replace('    return bits @ beta\n', '    return 2.0 * (bits @ beta)\n', 1)
    if changed == text:
        raise AssertionError('test fixture: behavior_logits return line not found')
    path.write_text(changed)
    fork.head = commit_all(fork.repo, 'E behaviour changed after stage 1')
    out.review.write_text(json.dumps({'verdict': 'APPROVED', 'commit': fork.head}))
    code, _, err = fork.write(out, 2, 'cd')
    if code != 2 or 'behaviour pin' not in err:
        raise AssertionError('a changed stage 1 behaviour was accepted: ' + error_of(err))


class HelperTests(unittest.TestCase):
    def test_section7_and_plan_bullets(self):
        freeze = load_freeze()
        section7, bullets = freeze.section7_of(CARD)
        self.assertTrue(section7.startswith('## 7.') and 'later' not in section7)
        self.assertEqual(bullets, BULLETS)
        self.assertEqual(freeze.plan_bullets(PLAN), BULLETS)

    def test_decision_entry_rules(self):
        freeze = load_freeze()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'log.md'; log.write_text(DECISION_LOG)
            facts = freeze.decision_entry_facts(log, ENTRY)
            self.assertEqual(facts['entry'], ENTRY); self.assertIn('16:3x', facts['heading'])
            for entry in (None, '', '16:3x 第2条', '2026-10-15 16:3x 第 x 条', '2026-10-15 16:4x 第2条',             # malformed / no such heading
                          '2026-10-15 16:3x 第1条',                                                               # item 1 does not name C9 as a report item
                          '2026-10-15 16:3x 第9条', '2026-10-15 16:0x 第3条'):                                     # no such item / the old heading never says report item
                with self.assertRaises(SystemExit) as stop, contextlib.redirect_stderr(io.StringIO()):
                    freeze.decision_entry_facts(log, entry)
                self.assertEqual(stop.exception.code, 2, msg=str(entry))
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    freeze.decision_entry_facts(Path(tmp) / 'absent.md', ENTRY)

    def test_stage_blocks_need_exactly_one_marker_pair(self):
        freeze = load_freeze()
        self.assertIsNone(freeze.card_block('no block', 1))
        text = 'a\n<!-- feat04b-freeze:begin stage=1 -->\nx\n<!-- feat04b-freeze:end stage=1 -->\nb'
        self.assertEqual(freeze.card_block(text, 1)[1], '\nx\n')
        for bad in (text + text, text.replace('end stage=1', 'end stage=2')):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                freeze.card_block(bad, 1)


class StageOneTests(unittest.TestCase):
    def test_cd_write_then_check(self):
        w = world(); out = w.outside()
        code, stdout, err = w.write(out, 1, 'cd')
        self.assertEqual(code, 0, msg=error_of(err))
        record = json.loads(out.rec1.read_text())
        self.assertEqual((record['stage'], record['batch'], record['commit']), (1, 'cd', w.head))
        self.assertEqual(sorted(record['configs']), ['f4_item_popularity', 'f4_item_quality'])
        self.assertTrue(all(len(per) == 3 for per in record['configs'].values()))
        self.assertEqual(sorted(record['behavior_pins']), ['f4_item_popularity', 'f4_item_quality'])
        self.assertEqual(record['audit']['report_threshold'], 0.3)
        self.assertEqual(record['plan']['verbatim_equal'], True)
        self.assertEqual(w.check(out, 1, 'cd')[0], 0)
        text = out.card.read_text()
        self.assertEqual(text.count('<!-- feat04b-freeze:begin stage=1 -->'), 1)
        self.assertIn(hashlib_sha(out.rec1.read_bytes()), text)
        tampered = w.copy_outside(out)
        tampered.card.write_text(text.replace(record['configs']['f4_item_quality']['42']['sha256'], '0' * 64, 1))
        self.assertEqual(w.check(tampered, 1, 'cd')[0], 2)                                      # the card block differs from the record

    def test_e_write_then_check(self):
        w = world(); out = w.outside()
        code, _, err = w.write(out, 1, 'e')
        self.assertEqual(code, 0, msg=error_of(err))
        record = json.loads(out.rec1.read_text())
        self.assertEqual((record['batch'], record['variants'], record['e4']['ran']), ('e', ['f4_recency_behavior'], 3))
        self.assertNotIn('audit', record)
        self.assertEqual(w.check(out, 1, 'e')[0], 0)

    def test_stage_one_needs_a_valid_decision_entry(self):
        w = world()
        for entry in (None, '2026-10-15 16:3x 第1条', '2026-10-15 17:9x 第2条', 'nonsense'):
            out = w.outside(); before = out.card.read_text()
            code, _, err = w.run(w.argv(out, '--write', 1, 'cd', **{'--decision-entry': entry}))
            self.assertEqual(code, 2, msg=str(entry))
            self.assertFalse(out.rec1.exists()); self.assertEqual(out.card.read_text(), before)

    def test_records_are_append_only(self):
        w = world(); out = w.outside()
        self.assertEqual(w.write(out, 1, 'e')[0], 0)
        record, card = out.rec1.read_bytes(), out.card.read_text()
        code, _, err = w.write(out, 1, 'e')                                                      # same record path
        self.assertEqual(code, 2)
        other = w.copy_outside(out); other.rec1 = other.dir / 'other.json'                        # another record path, the card already has the block
        code, _, err = w.run(w.argv(other, '--write', 1, 'e', **{'--record': str(other.rec1)}))
        self.assertEqual(code, 2); self.assertIn('append-only', err)
        self.assertEqual((out.rec1.read_bytes(), out.card.read_text()), (record, card))

    def test_guard_freeze_plan_text(self):
        w = world()
        for plan in (PLAN.replace('≥ 6（ΔR@50', '≥ 7（ΔR@50'), PLAN.replace('\n' + BULLETS[3], ''), PLAN.replace('否则不跑', '否则也跑')):
            out = w.outside(plan=plan)
            code, _, err = w.write(out, 1, 'e')
            self.assertEqual(code, 2); self.assertIn('differs from plan', err); self.assertFalse(out.rec1.exists())
        out = w.outside(card=CARD.replace('> ' + BULLETS[0], '> ' + BULLETS[0].replace('三 seed', '两 seed')))
        self.assertEqual(w.write(out, 1, 'e')[0], 2)

    def test_review_must_be_approved_for_a_commit_in_the_history(self):
        w = world()
        for review in ({'verdict': 'CHANGES_REQUESTED', 'commit': w.head}, {'verdict': 'APPROVED', 'commit': '1' * 40}, {'verdict': 'APPROVED'}, {'verdict': 'DEFERRED', 'commit': w.head}):
            out = w.outside(review=review)
            self.assertEqual(w.write(out, 1, 'e')[0], 2, msg=str(review)); self.assertFalse(out.rec1.exists())

    def test_audit_threshold_rule_applies_to_the_cd_batch(self):
        w = world()
        out = w.outside(card=CARD.replace('P_LIKE_REPORT_THRESHOLD = 0.30', 'P_LIKE_REPORT_THRESHOLD = 0.35'))
        code, _, err = w.write(out, 1, 'cd')
        self.assertEqual(code, 2); self.assertIn('P_LIKE_REPORT_THRESHOLD', err)
        out = w.outside(card=CARD.replace('P_LIKE_REPORT_THRESHOLD = 0.30', 'P_LIKE_REPORT_THRESHOLD = 0.35'))                  # E first: no audit involved at stage 1
        self.assertEqual(w.write(out, 1, 'e')[0], 0)

    def test_audit_receipt_and_constants_must_agree(self):
        w = world()
        doc = dict(w.doc)
        cases = {'receipt differs from the constants json': {'receipt': {'terminal': {'status': 'succeeded'}, 'audit': {**doc, 'sum_likes': 101}}},
                 'ac16 field missing': {'receipt': {'terminal': {'status': 'succeeded'}, 'audit': {k: v for k, v in doc.items() if k != 'logq_redundancy'}}},
                 'audit job failed': {'receipt': {'terminal': {'status': 'failed'}, 'audit': doc}},
                 'no audit document': {'receipt': {'terminal': {'status': 'succeeded'}}}}
        for label, override in cases.items():
            out = w.outside(**override)
            code, _, err = w.write(out, 1, 'cd')
            self.assertEqual(code, 2, msg=label); self.assertFalse(out.rec1.exists())
        out = w.outside()
        self.assertEqual(w.run(w.argv(out, '--write', 1, 'cd', **{'--audit-receipt': str(out.dir / 'absent.json')}))[0], 2)

    def test_code_constants_must_equal_the_audit(self):
        w = world().fork()
        path = w.repo / 'src/vkgr/retrieval_feat04b.py'
        text = path.read_text()
        for old, new in (("EXPO_EDGES = (4, 8,", "EXPO_EDGES = (5, 8,"), ('LIKE_P0 = 28', 'LIKE_P0 = 29'), ('LIKE_K = 10', 'LIKE_K = 11')):
            self.assertIn(old, text)
            path.write_text(text.replace(old, new, 1)); w.head = commit_all(w.repo, 'drift ' + new)
            out = w.outside()
            code, _, err = w.write(out, 1, 'cd')
            self.assertEqual(code, 2, msg=new); self.assertFalse(out.rec1.exists())
            path.write_text(text); w.head = commit_all(w.repo, 'revert')

    def test_item_stats_must_be_unchanged_since_the_deployed_commit(self):
        w = world().fork()
        with open(w.repo / 'src/vkgr/item_stats.py', 'a') as stream:
            stream.write('\n# changed after the statistics were built\n')
        w.head = commit_all(w.repo, 'item_stats changed')
        out = w.outside()
        code, _, err = w.write(out, 1, 'cd')                       # stats_commit is still the commit before the change
        self.assertEqual(code, 2); self.assertIn('item_stats.py changed', err)
        w.stats_commit = w.head
        out = w.outside()
        self.assertEqual(w.write(out, 1, 'cd')[0], 0)

    def test_e_batch_needs_the_k10_record_and_a_green_e4(self):
        w = world()
        for override in ({'k10': {'feat04a_freeze_sha': '2' * 40, 'merge_method': 'merge'}}, {'k10': {'feat04a_freeze_sha': w.head, 'merge_method': 'rebase'}},
                         {'k10': {'feat04a_freeze_sha': w.head, 'merge_method': 'cherry-pick'}}):
            out = w.outside(**override)
            self.assertEqual(w.write(out, 1, 'e')[0], 2, msg=str(override))
        for e4 in ({'ok': False, 'skipped': False, 'ran': 3}, {'ok': True, 'skipped': True, 'ran': 3}, {'ok': True, 'skipped': False, 'ran': 0}):
            out = w.outside()
            self.assertEqual(w.run(w.argv(out, '--write', 1, 'e'), e4=e4)[0], 2, msg=str(e4))

    def test_a_modified_pinned_file_stops_the_freeze(self):
        w = world().fork()
        with open(w.repo / 'src/vkgr/retrieval_feat03.py', 'a') as stream:
            stream.write('\n')
        w.head = commit_all(w.repo, 'pinned file touched')
        out = w.outside(review={'verdict': 'APPROVED', 'commit': w.head})
        self.assertEqual(w.write(out, 1, 'e')[0], 2)

    def test_a_dirty_tree_stops_the_freeze(self):
        w = world().fork()
        (w.repo / 'untracked.txt').write_text('x')
        out = w.outside()
        code, _, err = w.write(out, 1, 'e')
        self.assertEqual(code, 2); self.assertIn('not clean', err)

    def test_real_config_checks(self):
        w = world().fork()
        path = w.repo / 'configs/feat04b/p-feat04b-e-q2-behavior-s2024.json'
        for key, value in (('temperature', 0.15), ('learning_rate', 0.0003), ('batch_size', 256)):
            doc = json.loads(path.read_text()); original = dict(doc); doc[key] = value
            path.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n'); w.head = commit_all(w.repo, 'bad config ' + key)
            out = w.outside(review={'verdict': 'APPROVED', 'commit': w.head})
            self.assertEqual(w.write(out, 1, 'e')[0], 2, msg=key)
            path.write_text(json.dumps(original, indent=1, sort_keys=True) + '\n'); w.head = commit_all(w.repo, 'restore')


class StageTwoTests(unittest.TestCase):
    def test_cd_first_then_e(self):
        guard_freeze_stage2_append_only(world())

    def test_e_first_then_cd_runs_the_audit_checks_at_stage_two(self):
        w = world(); out = w.stage1_done('e')
        code, _, err = w.write(out, 2, 'cd')
        self.assertEqual(code, 0, msg=error_of(err))
        record = json.loads(out.rec2.read_text())
        self.assertEqual((record['stage'], record['batch']), (2, 'cd'))
        self.assertEqual(record['audit']['section7_sha256_before_audit'], json.loads(out.rec1.read_text())['section7_sha256'])
        self.assertTrue(record['stage1']['behavior_pins_recomputed_equal'])
        self.assertEqual(w.check(out, 2, 'cd')[0], 0)
        self.assertEqual(w.check(out, 1, 'e')[0], 0)                                               # HEAD is still the stage 1 commit here (nothing was added)

    def test_the_other_batch_only(self):
        w = world(); out = w.stage1_done('cd')
        self.assertEqual(w.write(out, 2, 'cd')[0], 2)
        self.assertFalse(out.rec2.exists())

    def test_section_7_may_not_change_between_the_stages(self):
        w = world(); out = w.stage1_done('e')
        out.card.write_text(out.card.read_text().replace('澄清', '澄清（改）'))
        code, _, err = w.write(out, 2, 'cd')
        self.assertEqual(code, 2); self.assertIn('section 7 changed', err)

    def test_stage_two_needs_the_card_to_register_the_stage_one_record(self):
        w = world(); out = w.stage1_done('e')
        out.card.write_text(out.card.read_text().replace(hashlib_sha(out.rec1.read_bytes()), '0' * 64))
        self.assertEqual(w.write(out, 2, 'cd')[0], 2)
        out = w.stage1_done('e')
        self.assertEqual(w.run(w.argv(out, '--write', 2, 'cd', **{'--stage1-record': None}))[0], 2)

    def test_guard_freeze_behavior_pin(self):
        guard_freeze_behavior_pin()

    def test_a_stage_one_config_changed_after_stage_one(self):
        base = world(); out = base.stage1_done('e')
        fork = base.fork()
        path = fork.repo / 'configs/feat04b/p-feat04b-e-q2-behavior-s2025.json'
        path.write_text(json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True) + '\n')
        fork.head = commit_all(fork.repo, 'reformatted a frozen config')
        out.review.write_text(json.dumps({'verdict': 'APPROVED', 'commit': fork.head}))
        code, _, err = fork.write(out, 2, 'cd')
        self.assertEqual(code, 2); self.assertIn('stage 1 configs changed', err)

    def test_stage_two_may_add_files_after_stage_one(self):
        base = world(); out = base.stage1_done('e')
        fork = base.fork()
        (fork.repo / 'docs').mkdir(exist_ok=True)
        (fork.repo / 'docs/new_note.md').write_text('added after stage 1\n')
        fork.head = commit_all(fork.repo, 'new file for stage 2')
        out.review.write_text(json.dumps({'verdict': 'APPROVED', 'commit': fork.head}))
        code, _, err = fork.write(out, 2, 'cd')
        self.assertEqual(code, 0, msg=error_of(err))
        self.assertEqual(fork.check(out, 2, 'cd')[0], 0)
        self.assertEqual(fork.check(out, 1, 'e')[0], 2)                                            # stage 1 is checked at its own commit only


class FreezeMutationTests(unittest.TestCase):
    """M86: a stage 2 that rewrites the stage 1 record, or does not recheck the stage 1 behaviour pins, must be caught by the two guards."""
    def test_m86_stage_two_rewrites_the_stage_one_record(self):
        from test_feat02_mutations import detected
        freeze = load_freeze(); w = world(); real = freeze.write_stage

        def rewriting(args):
            result = real(args)
            if args.stage == 2:
                Path(args.stage1_record).write_bytes(Path(args.stage1_record).read_bytes() + b' ')
            return result
        detected(self, freeze, 'write_stage', rewriting, [lambda: guard_freeze_stage2_append_only(w)])

    def test_m86_behaviour_pins_not_rechecked(self):
        from test_feat02_mutations import detected
        freeze = load_freeze(); real = freeze.behavior_pin
        recorded = json.loads(world().stage1_done('e').rec1.read_text())['behavior_pins']

        def blind(worktree, variant):
            return recorded[variant] if variant in recorded else real(worktree, variant)   # the stage 1 pins are 'recomputed' from the record: a changed behaviour is never noticed
        detected(self, freeze, 'behavior_pin', blind, [guard_freeze_behavior_pin])


if __name__ == '__main__':
    unittest.main()
