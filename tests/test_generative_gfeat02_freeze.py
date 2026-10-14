"""GFEAT-02 freeze script (gfeat02_freeze.py): append-only card section, mechanical sha256 / parameter counts, --check fails closed.

server-safe (no G1 contract). Works on a temporary copy of a card, never on the real one. ``--no-git`` (test only): the harness is
not the GFEAT-02 checkout, so the commit cross-check is skipped and the block records commit null, which the real --check rejects.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = '# card\n\n## 9. earlier section\ntext that must stay byte-identical\n'


def load():
    spec = importlib.util.spec_from_file_location('gfeat02_freeze', ROOT / 'gfeat02_freeze.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.fz = load()
        self._tmp = tempfile.TemporaryDirectory(dir='/private/tmp'); self.addCleanup(self._tmp.cleanup)
        self.card = Path(self._tmp.name) / 'card.md'; self.card.write_text(PRIOR)

    def test_write_appends_only_and_check_passes(self):
        self.assertEqual(self.fz.main(['--card', str(self.card), '--write', '--no-git']), 0)
        text = self.card.read_text()
        self.assertTrue(text.startswith(PRIOR))
        self.assertEqual(text.count(self.fz.BEGIN), 1); self.assertTrue(text.rstrip().endswith(self.fz.END))
        self.assertIn('## 11. GFEAT-02', text)
        block = self.fz.read_block(text)
        self.assertEqual(set(block['configs']), {'none_ctrl', 'f2_feedback', 'f3_watch_buckets'})
        self.assertEqual({k: v['counted'] for k, v in block['configs'].items()}, {'none_ctrl': False, 'f2_feedback': True, 'f3_watch_buckets': True})
        self.assertEqual(block['params']['none_ctrl']['params_total'], 4532816)  # GARCH-01 PF3 on the real codec (V=80)
        self.assertEqual(block['params']['f2_feedback']['params_total'] - 4532816, 6 * 128)
        self.assertEqual(block['params']['f3_watch_buckets']['params_total'] - 4532816, 14 * 128)
        self.assertEqual(block['params']['f3_watch_buckets']['params_side_trainable'], 12 * 128)
        self.assertEqual((block['rule']['delta_recall_at_50'], block['rule']['hits'], block['rule']['queries']), (0.002, 6, 2520))
        self.assertEqual(self.fz.main(['--card', str(self.card), '--check', '--no-git']), 0)
        self.assertIsNone(block['commit'])
        self.assertEqual(self.fz.main(['--card', str(self.card), '--check']), 1)  # a block without a commit never passes the real check

    def test_check_fails_closed(self):
        self.fz.main(['--card', str(self.card), '--write', '--no-git'])
        good = self.card.read_text()
        cases = {'prior text edited': good.replace('must stay', 'may change'),
                 'config sha forged': good.replace(self.fz.read_block(good)['configs']['f2_feedback']['sha256'], '0' * 64),
                 'block missing': PRIOR,
                 'block twice': good + good[len(PRIOR):]}
        for name, text in cases.items():
            with self.subTest(case=name):
                self.card.write_text(text)
                self.assertEqual(self.fz.main(['--card', str(self.card), '--check', '--no-git']), 1)

    def test_write_refuses_a_card_that_already_has_the_block(self):
        self.fz.main(['--card', str(self.card), '--write', '--no-git'])
        before = self.card.read_text()
        self.assertEqual(self.fz.main(['--card', str(self.card), '--write', '--no-git']), 1)
        self.assertEqual(self.card.read_text(), before)


if __name__ == '__main__':
    unittest.main()
