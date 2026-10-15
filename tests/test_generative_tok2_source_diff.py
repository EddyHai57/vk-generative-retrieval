"""GTOK-02: generative_tok2_run is generative_tok_run with a fixed list of literal replacements and nothing else.

Both files are read as text, the module docstrings are dropped, REPLACEMENTS (applied in order; each must hit at least once) are
applied to the GTOK-01 source, and the result must equal the GTOK-02 source character for character. Any other line in the new
runner, or any drift of the old one, makes this test fail. It needs no harness: it only reads two source files.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'src' / 'vkgr' / 'generative_tok_run.py'
NEW = ROOT / 'src' / 'vkgr' / 'generative_tok2_run.py'

TOK_MEMBERS_BLOCK = """TOK_MEMBERS = b1.B1_MEMBERS + ('src/vkgr/generative_batched.py', 'src/vkgr/generative_b1_batched.py', 'src/vkgr/sid_v2.py',
                               'src/vkgr/sid_rqvae.py', 'src/vkgr/sid_tok_audit.py', 'src/vkgr/generative_tok_data.py',
                               'src/vkgr/generative_tok_run.py')"""

REPLACEMENTS = (
    ("SCHEMA = 'g-tok-run-v1'", "SCHEMA = 'g-tok2-run-v1'"),
    ("TASK_ID = 'G-TOK-01'", "TASK_ID = 'G-TOK-02'"),
    ("MANIFEST_SCHEMA = 'g-tok-train-manifest-v1'", "MANIFEST_SCHEMA = 'g-tok2-train-manifest-v1'"),
    ("MANIFEST_KIND = 'g_tok_train'", "MANIFEST_KIND = 'g_tok2_train'"),
    ("CHECKPOINT_SCHEMA = 'g-tok-checkpoint-v1'", "CHECKPOINT_SCHEMA = 'g-tok2-checkpoint-v1'"),
    ("from . import sid_v2\n", "from . import sid_v2s\n"),
    ("from .generative_tok_data import HISTORY_LENGTH, build_tok_context\n",
     "from .generative_tok_data import HISTORY_LENGTH, build_tok_context\nfrom .generative_tok_run import TOK_MEMBERS\n"),
    (TOK_MEMBERS_BLOCK, "TOK2_MEMBERS = TOK_MEMBERS + ('src/vkgr/sid_v2s.py', 'src/vkgr/generative_tok2_run.py')"),
    ("for name in TOK_MEMBERS}", "for name in TOK2_MEMBERS}"),
    ("sid_v2.", "sid_v2s."),
    ("'manifest_sha256': info['manifest_sha256'],\n", "'manifest_sha256': info['manifest_sha256'], 'depth': info['depth'],\n"),
    ("SIDMAP_V2 label", "SIDMAP_V2S label"),
    ("G-TOK ", "G-TOK2 "),
    ("GTOK-01 G runner for SIDMAP_V2 tokenizers", "GTOK-02 G runner for SIDMAP_V2S tokenizers"),
)


def without_docstring(text):
    stripped = re.sub(r'\A""".*?"""\n', '', text, count=1, flags=re.S)
    if stripped == text:
        raise AssertionError('module docstring expected')
    return stripped


def transformed(old_text):
    text = without_docstring(old_text)
    for old, new in REPLACEMENTS:
        if old not in text:
            raise AssertionError(f'replacement matches nothing: {old[:60]!r}')
        text = text.replace(old, new)
    return text


class SourceDiffTests(unittest.TestCase):
    def test_new_runner_is_the_old_runner_with_the_literal_replacements_only(self):
        expected = transformed(OLD.read_text())
        actual = without_docstring(NEW.read_text())
        self.assertEqual(actual, expected)

    def test_every_replacement_hits_the_old_source(self):
        text = without_docstring(OLD.read_text())
        for old, _ in REPLACEMENTS:
            with self.subTest(old[:50]):
                self.assertGreaterEqual(text.count(old), 1)

    def test_the_replacements_are_the_whole_difference(self):
        old_lines = set(without_docstring(OLD.read_text()).splitlines())
        new_lines = [line for line in without_docstring(NEW.read_text()).splitlines() if line not in old_lines]
        self.assertTrue(new_lines)
        for line in new_lines:
            with self.subTest(line.strip()[:70]):
                self.assertTrue(any(new.split('\n')[0].strip() in line or new.split('\n')[-1].strip() in line for _, new in REPLACEMENTS),
                                'unexplained new line')

    def test_old_runner_still_registers_only_the_gtok01_contract(self):
        text = OLD.read_text()
        self.assertIn("TASK_ID = 'G-TOK-01'", text)
        self.assertNotIn('sid_v2s', text)

    def test_new_runner_has_no_leftover_gtok01_identity(self):
        text = NEW.read_text()
        for stale in ("'G-TOK-01'", "'g-tok-run-v1'", "'g_tok_train'", "'g-tok-train-manifest-v1'", "'g-tok-checkpoint-v1'", 'sid_v2.'):
            with self.subTest(stale):
                self.assertNotIn(stale, text)


if __name__ == '__main__':
    unittest.main()
