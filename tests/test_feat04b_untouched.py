"""FEAT-04b FEAT-001: the files FEAT-04b builds on stay byte-identical (sha256 pins from
configs/feat04b/pinned-files.json), and, when the base commit is reachable, nothing but additions
happens after it."""
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PINNED_JSON = ROOT / 'configs' / 'feat04b' / 'pinned-files.json'
BASE = '3463846c8774e795eecef57c686a8793402af9da'
GIT_QUIET = ['-c', 'user.name=guard', '-c', 'user.email=guard@example.invalid', '-c', 'commit.gpgsign=false']


def load_pins(path=PINNED_JSON):
    """The pinned set is whatever the JSON lists (design ruling F5: no hard-coded count)."""
    doc = json.loads(Path(path).read_text())
    if doc.get('schema') != 'p-feat04b-pinned-files-v1':
        raise AssertionError('pinned-files.json schema mismatch')
    if doc.get('base') != BASE:
        raise AssertionError('pinned-files.json base mismatch')
    pins = doc.get('files')
    if not isinstance(pins, dict) or not pins:
        raise AssertionError('pinned-files.json lists no files')
    return pins


def check_pinned(root, pins):
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
    def test_pin_list_covers_src_and_old_tests(self):
        pins = load_pins()
        src = [rel for rel in pins if rel.startswith('src/vkgr/')]
        self.assertEqual(len(src), 17)
        for prefix in ('tests/test_feat02_', 'tests/test_feat03_', 'tests/test_php01_'):
            self.assertTrue([rel for rel in pins if rel.startswith(prefix)], prefix)

    def test_pinned_files_are_byte_identical(self):
        check_pinned(ROOT, load_pins())

    def test_nothing_but_additions_since_base(self):
        check_only_added(ROOT)

    def test_guard_bites_on_a_modified_pinned_file(self):
        pins = load_pins()
        with tempfile.TemporaryDirectory() as tmp:
            for rel in pins:
                target = Path(tmp) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, target)
            check_pinned(Path(tmp), pins)                         # the copy is sound
            victim = Path(tmp) / 'src/vkgr/retrieval_feat03.py'
            victim.write_bytes(victim.read_bytes() + b'\n')
            with self.assertRaises(AssertionError):
                check_pinned(Path(tmp), pins)
            victim.unlink()
            with self.assertRaises(AssertionError):               # a missing pinned file is a failure, not a skip
                check_pinned(Path(tmp), pins)

    def test_pin_list_loader_rejects_a_bad_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'pins.json'
            for doc in ({'schema': 'x', 'base': BASE, 'files': {'a': 'b'}},
                        {'schema': 'p-feat04b-pinned-files-v1', 'base': '0' * 40, 'files': {'a': 'b'}},
                        {'schema': 'p-feat04b-pinned-files-v1', 'base': BASE, 'files': {}}):
                bad.write_text(json.dumps(doc))
                with self.assertRaises(AssertionError):
                    load_pins(bad)

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
            check_pinned(ROOT, load_pins())                       # the sha256 half does not depend on git


if __name__ == '__main__':
    unittest.main()
