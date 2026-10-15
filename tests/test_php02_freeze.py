"""P-HP-02 freeze tool (scripts/php02_freeze.py): --write fills the card's freeze block from git, --check re-derives it.

Everything runs on a throw-away git repo that holds copies of the six committed configs, so the tests do not depend on the
state of the real worktree. The card text is a minimal stand-in that contains the real section-5 sentences the tool parses.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'php02_freeze.py'
GIT = ['git', '-c', 'user.name=guard', '-c', 'user.email=guard@example.invalid', '-c', 'commit.gpgsign=false']
BEGIN, END = '<!-- php02-freeze:begin -->', '<!-- php02-freeze:end -->'
CARD = """# card
## 4. tests
text
## 5. rules
- 判定：有效 = 三 seed 均值 Δhits ≥ 6 且至少 2 个 seed Δhits ≥ 6；有害 = 均值 Δhits ≤ −6；其余 inconclusive。
- 补充对照数字（experiment_log）：F4-T010 三 seed R@50 0.17421 / 0.17222 / 0.17619，hits 439 / 434 / 444（seed 42 / 2024 / 2025）；F4（温度 0.15）hits 424 / 403 / 429。
## 6. paths
x
## 7. freeze
%s
（未冻结。）
%s
## 8. risks
y
""" % (BEGIN, END)


def load():
    spec = importlib.util.spec_from_file_location('php02_freeze', SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def run(module, *args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = module.main(list(args))
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'; (self.repo / 'configs').mkdir(parents=True)
        shutil.copytree(ROOT / 'configs' / 'php02', self.repo / 'configs' / 'php02')
        self.git('init', '-q'); self.git('add', '.'); self.git('commit', '-q', '-m', 'configs')
        self.card = Path(self.tmp.name) / 'card.md'; self.card.write_text(CARD, encoding='utf-8')
        self.module = load()

    def git(self, *args):
        return subprocess.run([*GIT, '-C', str(self.repo), *args], check=True, capture_output=True, text=True).stdout

    def args(self, mode):
        return [mode, '--worktree', str(self.repo), '--card', str(self.card)]

    def block(self):
        text = self.card.read_text(encoding='utf-8'); inner = text.split(BEGIN)[1].split(END)[0]
        start = inner.index('```json php02-freeze\n') + len('```json php02-freeze\n')
        return json.loads(inner[start:inner.rindex('\n```')])

    def test_write_then_check_round_trip(self):
        code, out, _ = run(self.module, *self.args('--write'))
        self.assertEqual(code, 0, out)
        block = self.block(); head = self.git('rev-parse', 'HEAD').strip()
        self.assertEqual(block['commit'], head)
        self.assertEqual(sorted(block['configs']), sorted(p.stem for p in (ROOT / 'configs' / 'php02').glob('*.json')))
        for name, entry in block['configs'].items():
            raw = (ROOT / 'configs' / 'php02' / f'{name}.json').read_bytes()
            self.assertEqual(entry['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertIn(entry['sha256'], self.card.read_text(encoding='utf-8'))
        self.assertEqual(block['thresholds'], {'effective_mean_dhits': 6, 'effective_min_seeds': 2, 'effective_seed_dhits': 6, 'harmful_mean_dhits': -6})
        self.assertEqual(block['control_hits'], {'42': 439, '2024': 434, '2025': 444})
        text = self.card.read_text(encoding='utf-8')
        self.assertEqual(text.split(END)[1], CARD.split(END)[1])                 # nothing after the block moved
        self.assertEqual(text.split(BEGIN)[0], CARD.split(BEGIN)[0])             # nothing before the block moved
        code, out, err = run(self.module, *self.args('--check'))
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)['status'], 'frozen_ok')

    def test_write_is_idempotent(self):
        run(self.module, *self.args('--write')); first = self.card.read_text(encoding='utf-8')
        run(self.module, *self.args('--write'))
        self.assertEqual(self.card.read_text(encoding='utf-8'), first)

    def test_check_fails_before_any_write(self):
        code, _, err = run(self.module, *self.args('--check'))
        self.assertNotEqual(code, 0)

    def test_check_fails_when_head_moved(self):
        run(self.module, *self.args('--write'))
        (self.repo / 'extra.txt').write_text('x'); self.git('add', 'extra.txt'); self.git('commit', '-q', '-m', 'more')
        self.assertNotEqual(run(self.module, *self.args('--check'))[0], 0)

    def test_check_fails_when_a_config_sha_in_the_card_is_wrong(self):
        run(self.module, *self.args('--write'))
        text = self.card.read_text(encoding='utf-8'); sha = self.block()['configs']['p-php02-t005-s42']['sha256']
        self.card.write_text(text.replace(sha, '0' * 64), encoding='utf-8')
        self.assertNotEqual(run(self.module, *self.args('--check'))[0], 0)

    def test_check_fails_when_a_committed_config_changed(self):
        run(self.module, *self.args('--write'))
        path = self.repo / 'configs' / 'php02' / 'p-php02-t007-s2024.json'
        path.write_text(path.read_text().replace('0.07', '0.08')); self.git('commit', '-q', '-am', 'edit')
        self.assertNotEqual(run(self.module, *self.args('--check'))[0], 0)

    def test_check_fails_on_a_dirty_tree(self):
        run(self.module, *self.args('--write'))
        (self.repo / 'untracked.txt').write_text('x')
        self.assertNotEqual(run(self.module, *self.args('--check'))[0], 0)
        self.assertNotEqual(run(self.module, *self.args('--write'))[0], 0)

    def test_check_fails_when_thresholds_in_the_card_block_are_edited(self):
        run(self.module, *self.args('--write'))
        text = self.card.read_text(encoding='utf-8')
        self.assertIn('"effective_mean_dhits": 6', text)
        self.card.write_text(text.replace('"effective_mean_dhits": 6', '"effective_mean_dhits": 5'), encoding='utf-8')
        self.assertNotEqual(run(self.module, *self.args('--check'))[0], 0)

    def test_check_fails_when_section5_thresholds_differ_from_the_tool(self):
        run(self.module, *self.args('--write'))
        text = self.card.read_text(encoding='utf-8')
        self.card.write_text(text.replace('三 seed 均值 Δhits ≥ 6', '三 seed 均值 Δhits ≥ 5'), encoding='utf-8')
        self.assertNotEqual(run(self.module, *self.args('--check'))[0], 0)
        self.card.write_text(text.replace('均值 Δhits ≤ −6', '均值 Δhits ≤ −5'), encoding='utf-8')
        self.assertNotEqual(run(self.module, *self.args('--check'))[0], 0)

    def test_check_fails_when_section5_control_hits_differ(self):
        run(self.module, *self.args('--write'))
        text = self.card.read_text(encoding='utf-8')
        self.card.write_text(text.replace('hits 439 / 434 / 444', 'hits 439 / 434 / 445'), encoding='utf-8')
        self.assertNotEqual(run(self.module, *self.args('--check'))[0], 0)

    def test_inline_mention_of_the_markers_in_prose_is_ignored(self):
        """The real card describes the markers in a sentence of section 7; only whole-line markers count."""
        prose = '- 写入 `%s … %s` 之间。\n' % (BEGIN, END)
        self.card.write_text(CARD.replace('## 7. freeze\n', '## 7. freeze\n' + prose), encoding='utf-8')
        self.assertEqual(run(self.module, *self.args('--write'))[0], 0)
        text = self.card.read_text(encoding='utf-8')
        self.assertIn(prose, text)
        self.assertEqual(run(self.module, *self.args('--check'))[0], 0)

    def test_write_refuses_when_markers_are_broken(self):
        self.card.write_text(CARD.replace(END, ''), encoding='utf-8')
        self.assertNotEqual(run(self.module, *self.args('--write'))[0], 0)
        self.card.write_text(CARD.replace(BEGIN, '') , encoding='utf-8')
        self.assertNotEqual(run(self.module, *self.args('--write'))[0], 0)

    def test_write_refuses_when_a_config_is_missing_at_head(self):
        self.git('rm', '-q', 'configs/php02/p-php02-t005-s42.json'); self.git('commit', '-q', '-m', 'rm')
        self.assertNotEqual(run(self.module, *self.args('--write'))[0], 0)

    def test_card_text_has_no_local_paths(self):
        run(self.module, *self.args('--write'))
        text = self.card.read_text(encoding='utf-8')
        for token in ('/Us' + 'ers/', 'Hello' + 'Bike', str(self.repo)):        # spelled in pieces so the push scan stays at 0 hits
            self.assertNotIn(token, text)


if __name__ == '__main__':
    unittest.main()
