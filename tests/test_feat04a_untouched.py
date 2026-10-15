"""FEAT-04a FEAT4A-000: the 17 src files and 14 test files FEAT-04a builds on stay byte-identical (31 sha256 pins),
and, when the base commit is reachable, nothing but additions happens after it ('A' lines only).

The pins were measured in the FEAT-04a worktree at the base commit. The working-tree cleanliness check lives in the freeze script, not here.
"""
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BASE = '3463846c8774e795eecef57c686a8793402af9da'
PINS = {
    'src/vkgr/retrieval_feat02.py': '9fb1bba3c22018c7dbf353b6dbfc3f75c0ca424fbcc2c0f89cce48616bef22f0',
    'src/vkgr/retrieval_feat02_run.py': '1e66de6586a183565bc9817ec971247ae140f762646c24411b56950e108984fd',
    'src/vkgr/retrieval_features.py': 'acc60664458a439823daa99edf1b9e44348ab6ca8783dd284a595404856878fa',
    'src/vkgr/retrieval_feat01_run.py': 'afa986f88eddd00a7e700c578c792f48fb039c4b311f71c636e288b5c3ddcfb9',
    'src/vkgr/retrieval_feat03.py': 'f0ff468901ec74ff6421823be8b92aea976087d5e2e12a3568eb92cd95c3581c',
    'src/vkgr/retrieval_feat03_run.py': '76eddcfead3e9a6def5bd37ddd0acd2cbad0f64b2b35dc2728000646f0abc43d',
    'src/vkgr/retrieval_php01.py': '94cd292ba49af0008c763515d49ca791fd5beb9e8b7aeaedf191485d98b2c801',
    'src/vkgr/retrieval_php01_run.py': '86e87014fdb266d06faebe9e57e3b8805c8e464edcee19897e3c542648fcdb54',
    'src/vkgr/retrieval_b1_run.py': '5b8ba48afcd86e86b222455d2105764454f1dbab151a4641b0cdf649c30243d6',
    'src/vkgr/retrieval_negatives.py': '242e5798e38f0b76c9d45037518122e2571004fb7d3f9f466e72c9dcf6f0abaa',
    'src/vkgr/retrieval_training.py': 'b947d301087fd8ab74d46ba93a976e01ed876f8dbeb2f615cb75087bffd8e9a6',
    'src/vkgr/history_side.py': '8faf06bb4dead2a83311bec1edfc84aaea1959fbbbde88871a1697de29cdea73',
    'src/vkgr/history_v2.py': '1c8d8718ab7527524df9d47dfbb518ee7f5b419ca34dfbb8971aca7efc31f06e',
    'src/vkgr/item_attr.py': '4da2d3fde75aaec8b19c2ab63c82bcba63c5fe2435eb78c5a5c6a812c711042c',
    'src/vkgr/retrieval_models.py': 'a482cfde4ec0b6bf5525fa51d3bf226d8c10a4dc4aac785c1440721ee68f30f2',
    'src/vkgr/retrieval_hp.py': 'fcc5e8ac18f5fb9278556414de331a1c19ccca7432ec8282cbc657bdc0febadc',
    'src/vkgr/retrieval_experiment.py': '4b29d02710582a933c2aed8fc908f0113ee70a7111f09da112c6f5a935f7caa7',
    'tests/test_feat02_audit.py': 'c391173b8018f11062cfdf08526ae4e0bce5cc0e1e2ac3172dca4cb9d32aa28b',
    'tests/test_feat02_data.py': '905b168c2d97fe1bb49855f4907fb4de1ecae93e2ba7ff952e53635188e32225',
    'tests/test_feat02_model.py': 'c4048995f5f9d87cf083c7912696124958cab03207af0558b5f5f8af1c6e3c2b',
    'tests/test_feat02_mutations.py': 'ca8ef770b55c4a5f8631ba2cd697073b6674800430ad7974113286c1e6124894',
    'tests/test_feat02_run.py': 'e489037b57b3c619d1a5b5c29c30ab9711c5a1d348bca33aa7121fee27c56703',
    'tests/test_feat03_audit.py': 'f48d04605ef404c653f586586e539d37eb68aa7cde8d4bea77863a2038ac2613',
    'tests/test_feat03_model.py': '8a42719ec2c211b39d3162b01db901a463174117255af751f1523ab8e4d2fa41',
    'tests/test_feat03_mutations.py': '7507a9be7f512bf4c4980d11057d7f14dda3213b3909d1ea62a2f976923e41df',
    'tests/test_feat03_run.py': 'cbd10d51ee03ed51acb2d5d2c7a9a4c80eea18029009476849090cfe64fcb8b8',
    'tests/test_feat03_untouched.py': '14e51a61887253052d37ff9a493fdcd8b6953035d145431c885e812800bf97a9',
    'tests/test_php01_model.py': '14c5467e9b7e2163c7644d16664ea78b1abbf7d25f234d63cffd1028ed962b50',
    'tests/test_php01_mutations.py': '5a39b0649288dba4928b9c60da1ac3935ee4f6e8423ada1a3980743f700b1d3e',
    'tests/test_php01_run.py': '5a24f02b89021206992fe9d2727f16a3c3ac2c1dc44719ae12f87b31d6ce8cb4',
    'tests/test_php01_select.py': 'cd86d25148befac39d87c462f694d355fdfc39052aadb5c26ebda253035b8ad8',
}
GIT_QUIET = ['-c', 'user.name=guard', '-c', 'user.email=guard@example.invalid', '-c', 'commit.gpgsign=false']


def check_pinned(root, pins=PINS):
    """Every pinned file must exist and hash to its pinned sha256 (no git needed)."""
    for rel, digest in pins.items():
        try:
            actual = hashlib.sha256((Path(root) / rel).read_bytes()).hexdigest()
        except OSError as error:
            raise AssertionError(f'pinned file unreadable: {rel}: {error}') from error
        if actual != digest:
            raise AssertionError(f'pinned file changed: {rel}')


def _git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)


def check_only_added(root, base=BASE):
    """`git diff --name-status --no-renames base..HEAD` may only show 'A' lines.

    No usable git or no base commit object -> SkipTest (not an error). Working-tree cleanliness is not checked here.
    """
    try:
        probe = _git(root, 'cat-file', '-e', f'{base}^{{commit}}')
    except OSError:
        raise unittest.SkipTest('base commit absent (git unavailable)')
    if probe.returncode != 0:
        raise unittest.SkipTest('base commit absent')
    try:
        diff = _git(root, 'diff', '--name-status', '--no-renames', f'{base}..HEAD')
    except OSError:
        raise unittest.SkipTest('base commit absent (git unavailable)')
    if diff.returncode != 0:
        raise AssertionError(f'git diff failed: {diff.stderr.strip()}')
    bad = [line for line in diff.stdout.splitlines() if line.strip() and not line.startswith('A\t')]
    if bad:
        raise AssertionError(f'non-additive changes since base: {bad}')


class UntouchedTests(unittest.TestCase):
    def test_pin_table_has_17_src_and_14_test_files(self):
        self.assertEqual(sum(1 for p in PINS if p.startswith('src/vkgr/')), 17)
        self.assertEqual(sum(1 for p in PINS if p.startswith('tests/')), 14)
        self.assertEqual(sum(1 for p in PINS if p.startswith('tests/test_feat02_')), 5)
        self.assertEqual(sum(1 for p in PINS if p.startswith('tests/test_feat03_')), 5)
        self.assertEqual(sum(1 for p in PINS if p.startswith('tests/test_php01_')), 4)

    def test_pinned_files_are_byte_identical(self):
        check_pinned(ROOT)

    def test_nothing_but_additions_since_base(self):
        check_only_added(ROOT)

    def test_guard_bites_on_a_modified_pinned_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rel in PINS:
                target = Path(tmp) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, target)
            check_pinned(Path(tmp))                               # the copy is sound
            victim = Path(tmp) / 'src/vkgr/retrieval_feat03.py'
            victim.write_bytes(victim.read_bytes() + b'\n')
            with self.assertRaises(AssertionError):
                check_pinned(Path(tmp))
            victim.unlink()
            with self.assertRaises(AssertionError):               # a missing pinned file is a failure, not a skip
                check_pinned(Path(tmp))

    def test_diff_guard_bites_on_modification_and_deletion(self):
        def git(repo, *args):
            subprocess.run(['git', *GIT_QUIET, '-C', str(repo), *args], check=True, capture_output=True)

        def head(repo):
            return subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, 'init', '-q')
            (repo / 'old.txt').write_text('one\n')
            git(repo, 'add', 'old.txt')
            git(repo, 'commit', '-q', '-m', 'base')
            base = head(repo)
            (repo / 'new.txt').write_text('two\n')
            git(repo, 'add', 'new.txt')
            git(repo, 'commit', '-q', '-m', 'add only')
            check_only_added(repo, base)                          # additions only: sound
            (repo / 'old.txt').write_text('changed\n')
            git(repo, 'commit', '-q', '-am', 'modify')
            with self.assertRaises(AssertionError):
                check_only_added(repo, base)
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, 'init', '-q')
            (repo / 'old.txt').write_text('one\n')
            git(repo, 'add', 'old.txt')
            git(repo, 'commit', '-q', '-m', 'base')
            base = head(repo)
            git(repo, 'rm', '-q', 'old.txt')
            git(repo, 'commit', '-q', '-m', 'delete')
            with self.assertRaises(AssertionError):
                check_only_added(repo, base)

    def test_unreachable_base_skips_instead_of_erroring(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(['git', '-C', tmp, 'init', '-q'], check=True, capture_output=True)
            with self.assertRaises(unittest.SkipTest):            # git exists, base commit does not
                check_only_added(Path(tmp), BASE)
        with mock.patch('subprocess.run', side_effect=FileNotFoundError):
            with self.assertRaises(unittest.SkipTest):            # no git executable at all
                check_only_added(ROOT)
        with mock.patch('subprocess.run', side_effect=PermissionError):
            with self.assertRaises(unittest.SkipTest):            # any OSError counts as unavailable
                check_only_added(ROOT)
        with mock.patch('subprocess.run', side_effect=FileNotFoundError):
            check_pinned(ROOT)                                    # the sha256 half does not depend on git


if __name__ == '__main__':
    unittest.main()
