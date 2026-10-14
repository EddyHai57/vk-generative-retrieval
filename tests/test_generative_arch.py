"""GARCH-01 explicit switch: config registry, factory, optimizer proxy, entry point and run lifecycle.

Run inside the frozen-base harness with OMP_NUM_THREADS=1 (engineering fixtures, CPU only, test split never read).
"""
import json
import subprocess
import unittest
from unittest import mock
from pathlib import Path

import arch_checks as ac
from test_generative_b1 import config as b1_config, engineering_world
from test_generative_batched import synthetic, tmpdir

ROOT = Path(__file__).resolve().parents[1]
BASE = '2c8882b'  # INFRA-02b tip: the default path GARCH-01 must leave byte-identical
B1_NAMES = ('history_v2', 'generative_data_v2', 'generative_hp', 'generative_diagnostics', 'generative_b1_run',
            'generative_sharded', 'generative_real', 'generative_training', 'generative_evaluation', 'generative_experiment', 'generative_data', 'tiger', 'sid')
DEFAULT_PATH_FILES = tuple(f'src/vkgr/{n}.py' for n in B1_NAMES) + ('src/vkgr/generative_batched.py', 'src/vkgr/generative_b1_batched.py')


def arch_config(**overrides):
    """Engineering-size independent-head config: 3 heads x 16 on d_model 64 (64 % 3 != 0, impossible for the stock module)."""
    from vkgr import generative_arch as ga
    value = ga.default_config()
    value.update({'d_model': 64, 'heads': 3, 'head_dim': 16, 'encoder_layers': 2, 'decoder_layers': 2, 'ff': 128, 'dropout': 0.1,
                  'microbatch': 2, 'max_epochs': 3, 'min_epochs': 1, 'learning_rate': .001})
    value.update(overrides)
    return value


class RegistryFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir()
        cls.context = synthetic(Path(cls.dir.name))

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_f1_registry_accepts_only_registered_values(self):
        ac.check_registry()

    def test_f2_factory_small_init_and_arch_report(self):
        ac.check_factory(self.context.codec)

    def test_f3_anchor_shape_legacy_factory_equals_frozen_factory(self):
        ac.check_anchor_factory_equals_frozen(self.context.codec)

    def test_f4_optimizer_proxy_matches_frozen_optimizer(self):
        ac.check_optimizer(self.context.codec)


class DefaultPathUntouchedTests(unittest.TestCase):
    def test_f5_default_path_files_are_byte_identical_to_2c8882b(self):
        """Branch-local guard: GARCH-01 only adds files. Drop or re-pin this test when integrating with tasks that edit these files."""
        from vkgr import generative_b1_run as b1
        self.assertEqual(b1.B1_MEMBERS, tuple(f'src/vkgr/{n}.py' for n in B1_NAMES))
        try:
            subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', f'{BASE}^{{commit}}'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, OSError):
            self.skipTest(f'{BASE} not in this checkout')
        for path in DEFAULT_PATH_FILES:
            with self.subTest(path=path):
                blob = subprocess.run(['git', '-C', str(ROOT), 'show', f'{BASE}:{path}'], capture_output=True, check=True).stdout
                self.assertEqual((ROOT / path).read_bytes(), blob)


class EntryTests(unittest.TestCase):
    def _common(self, tmp):
        _, receipt, manifest, sha = engineering_world(tmp)
        return (ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha)

    def test_r1_train_verify_roundtrip_binding_and_tamper_rejection(self):
        from vkgr import generative_b1_arch as ba
        from vkgr.data_audit import file_sha256
        from vkgr.tiger_arch import expected_param_count
        with tmpdir() as tmp:
            common = self._common(tmp)
            run = Path(tmp) / 'run'
            value = ba.run(*common, arch_config(), run, temporary=Path(tmp) / 's1')
            self.assertEqual(value['status'], 'complete')
            self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
            files = value['binding']['source']['files']
            for name in ba.ARCH_MEMBERS + ('src/vkgr/generative_batched.py', 'src/vkgr/generative_b1_batched.py'):
                self.assertIn(name, files)
            self.assertEqual(value['binding']['config']['schema'], 'g-b1-arch-run-v1')
            arch = value['binding']['init_report']['arch']
            self.assertEqual(arch['params_total'], expected_param_count(arch['spec']['vocab'], 64, 3, 16, 2, 2, 128))
            report = ba.verify(*common, run / 'manifest.json', value['manifest_sha256'], Path(tmp) / 'v1', temporary=Path(tmp) / 's2')
            self.assertEqual(report['replayed_epochs'], [value['best']['epoch']])
            with self.assertRaises(ValueError):
                ba.verify(*common, run / 'manifest.json', '0' * 64, Path(tmp) / 'v2', temporary=Path(tmp) / 's3')
            forged_value = json.loads((run / 'manifest.json').read_text())
            forged_value['epochs'][0]['metrics']['metrics']['all_eligible']['recall@50'] = 0.999
            forged = run / 'forged.json'
            forged.write_text(json.dumps(forged_value))
            with self.assertRaises(ValueError):
                ba.verify(*common, forged, file_sha256(forged), Path(tmp) / 'v3', temporary=Path(tmp) / 's4')
            (run / 'best.pt').write_bytes((run / 'best.pt').read_bytes() + b'0')
            with self.assertRaises(ValueError):
                ba.verify(*common, run / 'manifest.json', value['manifest_sha256'], Path(tmp) / 'v4', temporary=Path(tmp) / 's5')

    def test_r2_batched_b1_training_equals_serial_and_serial_verify_passes(self):
        from vkgr import generative_b1_arch as ba
        with tmpdir() as tmp:
            common = self._common(tmp)
            serial = ba.run(*common, arch_config(), Path(tmp) / 'serial', temporary=Path(tmp) / 's1')
            batched = ba.run(*common, arch_config(), Path(tmp) / 'batched', temporary=Path(tmp) / 's2', decode={'mode': 'batched', 'batch_size': 1})
            self.assertEqual([e['model_sha256'] for e in serial['epochs']], [e['model_sha256'] for e in batched['epochs']])
            self.assertEqual([e['evaluation']['computation_sha256'] for e in serial['epochs']], [e['evaluation']['computation_sha256'] for e in batched['epochs']])
            self.assertEqual(serial['selection'], batched['selection'])
            self.assertTrue(all(e['observations']['execution']['decode'] == 'batched' for e in batched['epochs']))
            self.assertTrue(all('execution' not in e['observations'] for e in serial['epochs']))
            report = ba.verify(*common, Path(tmp) / 'batched' / 'manifest.json', batched['manifest_sha256'], Path(tmp) / 'v', temporary=Path(tmp) / 's3')
            self.assertEqual(report['status'], 'verified')

    def test_r3_runner_restores_generative_b1_run_globals(self):
        from vkgr import generative_b1_arch as ba
        from vkgr import generative_b1_run as b1
        ac.check_globals_restored()
        before = {n: getattr(b1, n) for n in ac.RUNNER_GLOBALS}
        with tmpdir() as tmp:
            common = self._common(tmp)
            value = ba.run(*common, arch_config(), Path(tmp) / 'run', temporary=Path(tmp) / 's1')
            ba.verify(*common, Path(tmp) / 'run' / 'manifest.json', value['manifest_sha256'], Path(tmp) / 'v', temporary=Path(tmp) / 's2')
            with self.assertRaises(ValueError):
                ba.run(*common, arch_config(history_len=100), Path(tmp) / 'bad', temporary=Path(tmp) / 's3')
            self.assertFalse((Path(tmp) / 'bad').exists())
        for n in ac.RUNNER_GLOBALS:
            self.assertIs(getattr(b1, n), before[n], n)

    def test_r4_default_batched_and_arch_entries_do_not_cross_verify(self):
        from vkgr import generative_b1_arch as ba
        from vkgr import generative_b1_batched as bb
        from vkgr import generative_b1_run as b1
        with tmpdir() as tmp:
            common = self._common(tmp)
            mine = ba.run(*common, arch_config(), Path(tmp) / 'arch', temporary=Path(tmp) / 's1')
            plain = b1.run(*common, b1_config(), Path(tmp) / 'plain', temporary=Path(tmp) / 's2')
            cases = (('default entry on arch run', b1, 'arch', mine), ('batched entry on arch run', bb, 'arch', mine), ('arch entry on default run', ba, 'plain', plain))
            for index, (label, entry, run_dir, manifest) in enumerate(cases):
                with self.subTest(label), self.assertRaises(ValueError):
                    entry.verify(*common, Path(tmp) / run_dir / 'manifest.json', manifest['manifest_sha256'], Path(tmp) / f'cross-v{index}', temporary=Path(tmp) / f'cross-s{index}')

    def test_r5_guards_workers_batch_size_subcommands_and_registry_at_run_level(self):
        from vkgr import generative_b1_arch as ba
        ac.check_gates()
        with tmpdir() as tmp:
            common = self._common(tmp)
            for label, kwargs in (('workers 2', {'eval_workers': 2}), ('batch 2', {'decode': {'mode': 'batched', 'batch_size': 2}})):
                with self.subTest(label), self.assertRaises(ValueError):
                    ba.run(*common, arch_config(), Path(tmp) / f'out-{label[:1]}', temporary=Path(tmp) / f's-{label[:1]}', **kwargs)
                self.assertFalse((Path(tmp) / f'out-{label[:1]}').exists())
            for label, overrides in (('random SID', {'sid': 'SID_RANDPERM42_OF_BHKM_B16D3'}), ('history 100', {'history_len': 100}), ('beam 200', {'beam': 200})):
                with self.subTest(label), self.assertRaises(ValueError):
                    ba.run(*common, arch_config(**overrides), Path(tmp) / f'out-{label[:2]}', temporary=Path(tmp) / f's-{label[:2]}')
                self.assertFalse((Path(tmp) / f'out-{label[:2]}').exists())

    def test_r6_committed_configs_are_the_two_registered_runs(self):
        from vkgr import generative_arch as ga
        from vkgr import generative_data as gd
        paths = sorted((ROOT / 'configs' / 'garch01').glob('g-*.json'))
        self.assertEqual([p.name for p in paths], ['g-garch01-a1-lr3e-4-s42.json', 'g-garch01-a2-lr1e-3-s42.json'])
        for path, lr in zip(paths, (.0003, .001)):
            with self.subTest(path.name):
                cfg = gd._json(path)
                ga.validate_config(cfg)
                self.assertEqual((cfg['learning_rate'], cfg['microbatch'], cfg['seed']), (lr, 256, 42))
                self.assertEqual(tuple(cfg[k] for k in ga.SHAPE_KEYS), ga.PAPER_SCALE)
                self.assertEqual(path.read_text(), json.dumps(cfg, indent=1, sort_keys=True) + '\n')


class ParityScriptTests(unittest.TestCase):
    """R7: scripts/g_arch_parity.py on the CPU engineering world (the server runs the same code on real data + GPU)."""
    PF1_KEYS = {'init_equal', 'train_steps', 'train_steps_equal', 'first_diff_step', 'max_abs_param_diff', 'eval_logits_max_abs_diff_fastpath_on',
                'eval_logits_equal_fastpath_off', 'decode_lists_equal', 'queries', 'r50_old', 'r50_new'}
    PF2_KEYS = {'queries', 'equal', 'serial_wall_seconds', 'batched_b1_wall_seconds', 'peak_vram_bytes'}
    PF3_KEYS = {'params_total', 'params_expected', 'per_microbatch', 'runtime'}
    MB_KEYS = {'steps', 'finite', 'step_seconds_mean', 'peak_vram_bytes', 'error'}

    def _world(self, tmp):
        return ac.parity_world(tmp)

    def test_r7_anchor_shape_pf1_pf2_pf3_report_one_json_line(self):
        with tmpdir() as tmp:
            _, _, call = self._world(tmp)
            code, out, err = call('scratch', microbatches='2,1')
            self.assertEqual(code, 0, err)
            self.assertEqual(len(out.strip().splitlines()), 1)
            self.assertNotIn('"items"', out)  # no per-query rows
            report = json.loads(out)
            self.assertEqual(set(report), {'pf1_anchor_shape', 'pf2_paper_scale_decode', 'pf3_paper_scale_train'})
            pf1, pf2, pf3 = report['pf1_anchor_shape'], report['pf2_paper_scale_decode'], report['pf3_paper_scale_train']
            self.assertEqual((set(pf1), set(pf2), set(pf3)), (self.PF1_KEYS, self.PF2_KEYS, self.PF3_KEYS))
            self.assertIs(pf1['init_equal'], True)
            self.assertIs(pf1['train_steps_equal'], True)
            self.assertIsNone(pf1['first_diff_step'])
            self.assertEqual((pf1['train_steps'], pf1['max_abs_param_diff']), (3, 0.0))
            self.assertIs(pf1['eval_logits_equal_fastpath_off'], True)
            self.assertLessEqual(pf1['eval_logits_max_abs_diff_fastpath_on'], 1e-5)
            self.assertEqual(pf1['decode_lists_equal'], pf1['queries'])
            self.assertEqual(pf1['queries'], pf2['queries'])
            self.assertAlmostEqual(pf1['r50_old'], pf1['r50_new'], places=12)
            self.assertIs(pf2['equal'], True)
            self.assertIsNone(pf2['peak_vram_bytes'])  # CPU
            self.assertEqual(set(pf3['per_microbatch']), {'2', '1'})
            for entry in pf3['per_microbatch'].values():
                self.assertEqual(set(entry), self.MB_KEYS)
                self.assertIs(entry['finite'], True)
                self.assertEqual((entry['steps'], entry['error']), (3, None))
            self.assertEqual(pf3['params_total'], pf3['params_expected'])

    def test_r7_perturbed_new_class_is_reported_not_hidden(self):
        from vkgr import tiger_arch
        original = tiger_arch.TigerArchModel.decode
        with tmpdir() as tmp:
            _, _, call = self._world(tmp)
            with mock.patch.object(tiger_arch.TigerArchModel, 'decode', lambda self, memory, padding, decoder: original(self, memory, padding, decoder) * 1.001):
                code, out, err = call('scratch')
            self.assertEqual(code, 0, err)
            pf1 = json.loads(out)['pf1_anchor_shape']
            self.assertIs(pf1['train_steps_equal'], False)
            self.assertEqual(pf1['first_diff_step'], 1)
            self.assertGreater(pf1['max_abs_param_diff'], 0.0)
            self.assertIs(pf1['eval_logits_equal_fastpath_off'], False)

    def test_r7_tampered_reference_checkpoint_or_bad_arguments_exit_1_without_output(self):
        with tmpdir() as tmp:
            _, value, call = self._world(tmp)
            (Path(tmp) / 'run' / 'best.pt').write_bytes((Path(tmp) / 'run' / 'best.pt').read_bytes() + b'0')
            code, out, err = call('scratch-tampered')
            self.assertEqual((code, out), (1, ''))
            self.assertIn('arch parity gate failed', err)
        with tmpdir() as tmp:
            _, _, call = self._world(tmp)
            for label, kwargs in (('zero steps', {'steps': '0'}), ('bad microbatch', {'microbatches': '0'}), ('non-integer microbatch', {'microbatches': 'x'})):
                with self.subTest(label):
                    code, out, err = call(f'scratch-{label[:3]}', **kwargs)
                    self.assertEqual((code, out), (1, ''))


if __name__ == '__main__':
    unittest.main()
