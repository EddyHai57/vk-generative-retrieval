"""FEAT-04a scripts: freeze tool helpers, VRAM smoke (CPU, small catalog), CUDA parity script on CPU.
No path of this machine appears here: the plan file is only read when FEAT04A_PLAN points at it (skipped otherwise)."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


PLAN_TEXT = """### 1.2 预注册判定

- 有效：三 seed 均值 Δhits ≥ 6，且至少 2 个 seed 的 Δhits ≥ 6。
- 转正（成为新 anchor 的一部分）：有效，且至少 2 个 seed 逐 seed exact McNemar p < 0.05。
- inconclusive：−6 < 均值 Δhits < 6；有害：均值 Δhits ≤ −6。
- **条件组合（1 个名额）**：不在本卡。

### 1.3 成本
- 有效：这一行不属于 1.2
"""


class FreezeHelperTests(unittest.TestCase):
    def test_decision_bullets_take_the_first_matching_line_of_the_section(self):
        freeze = load('feat04a_freeze')
        section = PLAN_TEXT.split('### 1.2', 1)[1].split('### 1.3', 1)[0]
        bullets = freeze.decision_bullets(section)
        self.assertEqual(len(bullets), 3)
        self.assertTrue(bullets[0].startswith('- 有效：') and bullets[2].startswith('- inconclusive：'))
        with self.assertRaises(SystemExit):
            freeze.decision_bullets('- 有效：only one\n')

    def test_a_and_b_variant_tables(self):
        freeze = load('feat04a_freeze')
        self.assertEqual(set(freeze.VARIANTS), {'A', 'B'})
        self.assertEqual(freeze.SEEDS, (42, 2024, 2025))
        self.assertEqual(freeze.BASE, '3463846c8774e795eecef57c686a8793402af9da')

    def test_a_configs_pass_the_freeze_checks_at_head(self):
        """config_facts reads the committed blobs: validate_config(real), seeds differ only in `seed`, seed 42 equals H1 minus dim/knob."""
        if subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', 'HEAD:configs/feat04a/p-feat04a-a-q2t010-s42.json'], capture_output=True).returncode != 0:
            self.skipTest('configs not committed yet')
        freeze = load('feat04a_freeze')
        facts = freeze.config_facts(ROOT, 'A')
        self.assertEqual(len(facts), 3)
        self.assertTrue(all(len(v) == 64 for v in facts.values()))

    def test_check_ignored_refuses_a_tracked_or_unignored_file(self):
        freeze = load('feat04a_freeze')
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(['git', '-C', tmp, 'init', '-q'], check=True, capture_output=True)
            (Path(tmp) / '.gitignore').write_text('secret.md\n')
            (Path(tmp) / 'secret.md').write_text('x'); (Path(tmp) / 'open.md').write_text('x')
            freeze.check_ignored(Path(tmp) / 'secret.md')
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                freeze.check_ignored(Path(tmp) / 'open.md')

    @unittest.skipUnless(os.environ.get('FEAT04A_PLAN'), 'set FEAT04A_PLAN to the plan file to check its reference sha256')
    def test_plan_reference_sha256(self):
        import hashlib
        freeze = load('feat04a_freeze')
        self.assertEqual(hashlib.sha256(Path(os.environ['FEAT04A_PLAN']).read_bytes()).hexdigest(), freeze.PLAN_SHA256)


class SmokeScriptTests(unittest.TestCase):
    def test_vram_smoke_declared_rule_and_cpu_run(self):
        smoke = load('feat04a_vram_smoke')
        self.assertEqual([smoke.declared_vram_mib(m) for m in (100, 640, 890, 891, 1200)], [1024, 1024, 1024, 1280, 1536])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(smoke.main(['--device', 'cpu', '--catalog', '3000', '--steps', '1']), 0)
        row = json.loads(out.getvalue())
        self.assertEqual((row['task'], row['device']), ('FEAT-04A', 'cpu'))
        self.assertTrue(row['features'][0]['finite_loss']); self.assertEqual(row['features'][0]['vram'], 'UNKNOWN')

    def test_cuda_flag_without_cuda_is_refused(self):
        import torch
        smoke = load('feat04a_vram_smoke')
        if torch.cuda.is_available():
            self.skipTest('CUDA present')
        with self.assertRaises(SystemExit):
            smoke.main(['--device', 'cuda'])
        parity = load('feat04a_cuda_parity')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(parity.main(['--device', 'cuda']), 2)

    def test_cuda_parity_script_on_cpu(self):
        parity = load('feat04a_cuda_parity')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(parity.main(['--device', 'cpu', '--epochs', '2']), 0)
        self.assertEqual(json.loads(out.getvalue())['status'], 'parity_ok')

    def test_cuda_parity_script_detects_a_difference(self):
        from unittest import mock
        from vkgr import retrieval_feat04a_run as run4
        parity = load('feat04a_cuda_parity')
        original = run4.train_step

        def hard(model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency):
            return original(model, optimizer, batch, data, stats, epoch, {**config, 'temperature': 0.15}, rng, pool, frequency)
        out = io.StringIO()
        with mock.patch.object(run4, 'train_step', hard), contextlib.redirect_stdout(out):
            self.assertEqual(parity.main(['--device', 'cpu', '--epochs', '2']), 2)
        self.assertEqual(json.loads(out.getvalue())['status'], 'parity_differs')


if __name__ == '__main__':
    unittest.main()
