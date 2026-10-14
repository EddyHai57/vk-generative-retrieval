"""FEAT-03 FEAT-001: the feat02/feat01 files FEAT-03 builds on stay byte-identical (sha256 pins),
and, when the base commit is reachable, nothing but additions happens after it."""
import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BASE = '94eb0bcbf66b73397a7a5d1332ef9c55a8fa39f2'
PINS = {
    'src/vkgr/retrieval_feat02.py': '9fb1bba3c22018c7dbf353b6dbfc3f75c0ca424fbcc2c0f89cce48616bef22f0',
    'src/vkgr/retrieval_feat02_run.py': '1e66de6586a183565bc9817ec971247ae140f762646c24411b56950e108984fd',
    'src/vkgr/retrieval_features.py': 'acc60664458a439823daa99edf1b9e44348ab6ca8783dd284a595404856878fa',
    'src/vkgr/retrieval_feat01_run.py': 'afa986f88eddd00a7e700c578c792f48fb039c4b311f71c636e288b5c3ddcfb9',
    'tests/test_feat02_model.py': 'c4048995f5f9d87cf083c7912696124958cab03207af0558b5f5f8af1c6e3c2b',
    'tests/test_feat02_run.py': 'e489037b57b3c619d1a5b5c29c30ab9711c5a1d348bca33aa7121fee27c56703',
    'tests/test_feat02_mutations.py': 'ca8ef770b55c4a5f8631ba2cd697073b6674800430ad7974113286c1e6124894',
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
            victim = Path(tmp) / 'src/vkgr/retrieval_feat02.py'
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
