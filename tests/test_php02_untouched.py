"""P-HP-02: the old entry point and everything it imports stay byte-identical (sha256 pins taken mechanically from BASE with
``git show BASE:<path> | sha256``), and since BASE only additions happen (``git diff --name-status --no-renames BASE..HEAD`` shows only 'A').

Server rule: run this module on the server worktree and require ``skipped=0``; a SkipTest here means "base commit not reachable" = not verified.
"""
import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat03_untouched import check_only_added, check_pinned  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = '3463846c8774e795eecef57c686a8793402af9da'
PINS = {
    'src/vkgr/retrieval_php01.py': '94cd292ba49af0008c763515d49ca791fd5beb9e8b7aeaedf191485d98b2c801',
    'src/vkgr/retrieval_php01_run.py': '86e87014fdb266d06faebe9e57e3b8805c8e464edcee19897e3c542648fcdb54',
    'src/vkgr/retrieval_php01_select.py': 'e3498968fc5499548a2ebeb8dffe7df66eb7a5a3e01f381333fad7031012e338',
    'src/vkgr/retrieval_feat02_run.py': '1e66de6586a183565bc9817ec971247ae140f762646c24411b56950e108984fd',
    'src/vkgr/retrieval_feat02.py': '9fb1bba3c22018c7dbf353b6dbfc3f75c0ca424fbcc2c0f89cce48616bef22f0',
    'src/vkgr/retrieval_b1_run.py': '5b8ba48afcd86e86b222455d2105764454f1dbab151a4641b0cdf649c30243d6',
    'src/vkgr/retrieval_feat01_run.py': 'afa986f88eddd00a7e700c578c792f48fb039c4b311f71c636e288b5c3ddcfb9',
    'src/vkgr/retrieval_negatives.py': '242e5798e38f0b76c9d45037518122e2571004fb7d3f9f466e72c9dcf6f0abaa',
    'src/vkgr/retrieval_training.py': 'b947d301087fd8ab74d46ba93a976e01ed876f8dbeb2f615cb75087bffd8e9a6',
    'src/vkgr/retrieval_ablation_run.py': '2f9feda7d0bd542bbc7ba9ab22720c84f226542ca73e37406e131ac79203f543',
    'src/vkgr/retrieval_hp.py': 'fcc5e8ac18f5fb9278556414de331a1c19ccca7432ec8282cbc657bdc0febadc',
    'tests/test_php01_model.py': '14c5467e9b7e2163c7644d16664ea78b1abbf7d25f234d63cffd1028ed962b50',
    'tests/test_php01_mutations.py': '5a39b0649288dba4928b9c60da1ac3935ee4f6e8423ada1a3980743f700b1d3e',
    'tests/test_php01_run.py': '5a24f02b89021206992fe9d2727f16a3c3ac2c1dc44719ae12f87b31d6ce8cb4',
    'tests/test_php01_select.py': 'cd86d25148befac39d87c462f694d355fdfc39052aadb5c26ebda253035b8ad8',
    'scripts/php01_vram_smoke.py': '8c890475ba9327d776f37dede283959fef9a21c0d88a23cfad2df8cd49c7dc93',
    'configs/php01/p-php01-h1-t010-s42.json': '299df44ba16e982a400e67bf384756d4f3bc12f8422aaf6069100fb38cb2b913',
    'configs/php01/p-php01-h2-t025-s42.json': '19bff9818b0d96aaec4da84a1300ffc85c2d7ccac1ebbd2e41337e7461396e55',
    'configs/php01/p-php01-h3-d32-s42.json': '3c1d03772136c72a06d2255cfb51f0b2f9dc22ffdf46ab09749c5309a4fbc20a',
    'configs/php01/p-php01-h4-d128-s42.json': '63e3d10ec3108dde0dd96a4a4d7364f93129999247ba45c91a84dee9d4f8f9a2',
    'configs/php01/p-php01-h5-lr002-s42.json': '579038f3494c0fbaa02eec7d6ab4f1e3599ba2a7c8b478a5d66318a5e2cfe4aa',
    'configs/php01/p-php01-h6-lr0005-s42.json': 'ac9df853de7d1e99fb7ae8da9809f4646f4f8008cfcbe0f45844f9b6902a5fb9',
    'configs/php01/p-php01-h7-combo-rule.json': 'e4ef4c162f9c13a6f0726780b16666fa6508b40ec07196c099c724b15aad6c4a',
}
# step-0 additions (not in BASE): the H1 control configs for seed 2024 / 2025, taken byte for byte from the php01c branch
STEP0_PINS = {
    'configs/php01/p-php01-h1-t010-s2024.json': '2d710a75bb415eeaa13b0da475df6c31204159c73070a2898a8033431def897b',
    'configs/php01/p-php01-h1-t010-s2025.json': '94aa8b2660632b13df441eee55fb82e740a766d779939225453d35a097bbb5c0',
}


class UntouchedTests(unittest.TestCase):
    def test_base_is_the_full_40_hex_commit(self):
        self.assertEqual(BASE, '3463846c8774e795eecef57c686a8793402af9da')
        self.assertEqual(len(BASE), 40)
        int(BASE, 16)

    def test_pinned_files_are_byte_identical(self):
        check_pinned(ROOT, PINS)
        check_pinned(ROOT, STEP0_PINS)

    def test_nothing_but_additions_since_base(self):
        check_only_added(ROOT, BASE)

    def test_pin_table_covers_the_modules_php02_imports(self):
        for rel in ('src/vkgr/retrieval_php01.py', 'src/vkgr/retrieval_php01_run.py', 'src/vkgr/retrieval_php01_select.py',
                    'src/vkgr/retrieval_feat02_run.py', 'src/vkgr/retrieval_feat02.py', 'src/vkgr/retrieval_b1_run.py',
                    'src/vkgr/retrieval_feat01_run.py', 'src/vkgr/retrieval_negatives.py', 'src/vkgr/retrieval_training.py',
                    'src/vkgr/retrieval_ablation_run.py', 'src/vkgr/retrieval_hp.py'):
            self.assertIn(rel, PINS)
        for path in ROOT.joinpath('tests').glob('test_php01_*.py'):
            self.assertIn(f'tests/{path.name}', PINS)
        for path in ROOT.joinpath('configs', 'php01').glob('*.json'):
            self.assertIn(f'configs/php01/{path.name}', {**PINS, **STEP0_PINS})

    def test_guard_bites_on_a_modified_or_missing_pinned_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rel in PINS:
                target = Path(tmp) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, target)
            check_pinned(Path(tmp), PINS)
            victim = Path(tmp) / 'src/vkgr/retrieval_php01_run.py'
            victim.write_bytes(victim.read_bytes() + b'\n')
            with self.assertRaises(AssertionError):
                check_pinned(Path(tmp), PINS)
            victim.unlink()
            with self.assertRaises(AssertionError):
                check_pinned(Path(tmp), PINS)

    def test_pins_equal_git_show_of_base_when_reachable(self):
        import subprocess
        probe = subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', f'{BASE}^{{commit}}'], capture_output=True)
        if probe.returncode != 0:
            self.skipTest('base commit absent')
        for rel, digest in PINS.items():
            blob = subprocess.run(['git', '-C', str(ROOT), 'show', f'{BASE}:{rel}'], capture_output=True, check=True).stdout
            self.assertEqual(hashlib.sha256(blob).hexdigest(), digest, rel)


if __name__ == '__main__':
    unittest.main()
