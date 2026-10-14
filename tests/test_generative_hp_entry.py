"""G-HP-01 entry point (python -m vkgr.generative_b1_hp): run lifecycle, binding, isolation, guards, CLI.

Run inside the frozen-base harness with OMP_NUM_THREADS=1 (engineering fixtures, CPU only, test split never read).
"""
import contextlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import hp_checks as hc
from test_generative_arch import arch_config
from test_generative_b1 import config as b1_config, engineering_world
from test_generative_batched import tmpdir

ROOT = Path(__file__).resolve().parents[1]


class EntryTests(unittest.TestCase):
    def _common(self, tmp):
        _, receipt, manifest, sha = engineering_world(tmp)
        return (ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha)

    def test_e1_train_verify_roundtrip_binding_and_tamper_rejection(self):
        from vkgr import generative_b1_hp as bh
        from vkgr.tiger_arch import expected_param_count
        with tmpdir() as tmp:
            common = self._common(tmp)
            run = Path(tmp) / 'run'
            value = bh.run(*common, hc.entry_config(), run, temporary=Path(tmp) / 's1')
            self.assertEqual(value['status'], 'complete')
            self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
            hc.check_binding_members(value)
            arch = value['binding']['init_report']['arch']
            self.assertEqual(arch['params_total'], expected_param_count(arch['spec']['vocab'], 64, 3, 16, 2, 2, 128))
            self.assertEqual(arch['spec']['dropout'], 0.0)
            report = bh.verify(*common, run / 'manifest.json', value['manifest_sha256'], Path(tmp) / 'v1', temporary=Path(tmp) / 's2')
            self.assertEqual(report['replayed_epochs'], [value['best']['epoch']])
            (run / 'best.pt').write_bytes((run / 'best.pt').read_bytes() + b'0')
            with self.assertRaises(ValueError):
                bh.verify(*common, run / 'manifest.json', value['manifest_sha256'], Path(tmp) / 'v2', temporary=Path(tmp) / 's3')

    def test_e2_batched_b1_training_equals_serial_and_serial_verify_passes(self):
        from vkgr import generative_b1_hp as bh
        with tmpdir() as tmp:
            common = self._common(tmp)
            serial = bh.run(*common, hc.entry_config(), Path(tmp) / 'serial', temporary=Path(tmp) / 's1')
            batched = bh.run(*common, hc.entry_config(), Path(tmp) / 'batched', temporary=Path(tmp) / 's2', decode={'mode': 'batched', 'batch_size': 1})
            self.assertEqual([e['model_sha256'] for e in serial['epochs']], [e['model_sha256'] for e in batched['epochs']])
            self.assertEqual([e['evaluation']['computation_sha256'] for e in serial['epochs']], [e['evaluation']['computation_sha256'] for e in batched['epochs']])
            self.assertEqual(serial['selection'], batched['selection'])
            self.assertTrue(all(e['observations']['execution']['decode'] == 'batched' for e in batched['epochs']))
            report = bh.verify(*common, Path(tmp) / 'batched' / 'manifest.json', batched['manifest_sha256'], Path(tmp) / 'v', temporary=Path(tmp) / 's3')
            self.assertEqual(report['status'], 'verified')

    def test_e3_runner_restores_generative_b1_run_globals(self):
        from vkgr import generative_b1_hp as bh
        from vkgr import generative_b1_run as b1
        hc.check_globals_restored()
        before = {n: getattr(b1, n) for n in hc.RUNNER_GLOBALS}
        with tmpdir() as tmp:
            common = self._common(tmp)
            with self.assertRaises(ValueError):
                bh.run(*common, hc.entry_config(history_len=100), Path(tmp) / 'bad', temporary=Path(tmp) / 's1')
            self.assertFalse((Path(tmp) / 'bad').exists())
        for n in hc.RUNNER_GLOBALS:
            self.assertIs(getattr(b1, n), before[n], n)

    def test_e4_ghp_runs_do_not_cross_verify_with_other_entries(self):
        from vkgr import generative_b1_arch as ba
        from vkgr import generative_b1_batched as bb
        from vkgr import generative_b1_hp as bh
        from vkgr import generative_b1_run as b1
        with tmpdir() as tmp:
            common = self._common(tmp)
            mine = bh.run(*common, hc.entry_config(), Path(tmp) / 'hp', temporary=Path(tmp) / 's1')
            arch = ba.run(*common, arch_config(), Path(tmp) / 'arch', temporary=Path(tmp) / 's2')
            plain = b1.run(*common, b1_config(), Path(tmp) / 'plain', temporary=Path(tmp) / 's3')
            cases = (('default entry on G-HP run', b1, 'hp', mine), ('batched entry on G-HP run', bb, 'hp', mine), ('GARCH entry on G-HP run', ba, 'hp', mine),
                     ('G-HP entry on GARCH run', bh, 'arch', arch), ('G-HP entry on default run', bh, 'plain', plain))
            for index, (label, entry, run_dir, manifest) in enumerate(cases):
                with self.subTest(label), self.assertRaises(ValueError):
                    entry.verify(*common, Path(tmp) / run_dir / 'manifest.json', manifest['manifest_sha256'], Path(tmp) / f'cross-v{index}', temporary=Path(tmp) / f'cross-s{index}')

    def test_e5_guards_workers_batch_size_subcommands_before_output(self):
        from vkgr import generative_b1_hp as bh
        hc.check_gates()
        with tmpdir() as tmp:
            common = self._common(tmp)
            for label, kwargs in (('workers 2', {'eval_workers': 2}), ('batch 2', {'decode': {'mode': 'batched', 'batch_size': 2}})):
                with self.subTest(label), self.assertRaises(ValueError):
                    bh.run(*common, hc.entry_config(), Path(tmp) / f'out-{label[:1]}', temporary=Path(tmp) / f's-{label[:1]}', **kwargs)
                self.assertFalse((Path(tmp) / f'out-{label[:1]}').exists())

    def test_e6_cli_train_with_batched_b1_then_serial_verify(self):
        from vkgr import generative_b1_hp as bh
        from vkgr.data_audit import file_sha256
        with tmpdir() as tmp:
            _, receipt, manifest, sha = engineering_world(tmp)
            config_path = Path(tmp) / 'cfg.json'
            config_path.write_text(json.dumps(hc.entry_config(), indent=1, sort_keys=True) + '\n')
            common = ['--prepared', str(receipt['manifest_path']), '--prepared-sha', receipt['manifest_sha256'], '--history-manifest', str(manifest), '--history-sha', sha]

            def cli(args, env):
                clean = {k: v for k, v in os.environ.items() if not k.startswith('VKGR_')}
                clean.update(env)
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.dict(os.environ, clean, clear=True), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = bh.main(args)
                return code, out.getvalue(), err.getvalue()
            code, out, err = cli(['train', *common, '--config', str(config_path), '--config-sha', file_sha256(config_path), '--output', f'{tmp}/run', '--scratch', f'{tmp}/s1'],
                                 {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '1'})
            self.assertEqual(code, 0, err)
            trained = json.loads(out.strip().splitlines()[-1])
            self.assertEqual(trained['status'], 'complete')
            code, out, err = cli(['verify', *common, '--manifest', f'{tmp}/run/manifest.json', '--manifest-sha', trained['manifest_sha256'], '--output', f'{tmp}/v', '--scratch', f'{tmp}/s2'], {})
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out.strip().splitlines()[-1])['status'], 'verified')


if __name__ == '__main__':
    unittest.main()
