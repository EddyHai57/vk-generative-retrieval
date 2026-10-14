"""G-HP-01 scripts: server preflight PH1-PH3 (scripts/ghp01_parity.py), A1 entry gate (ghp01_gate.py), combination trigger + card 1.5
selection (ghp01_combo.py). Run inside the frozen-base harness with OMP_NUM_THREADS=1 (CPU engineering world, test split never read).
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import hp_checks as hc
from test_generative_batched import tmpdir

ROOT = Path(__file__).resolve().parents[1]


class GateTests(unittest.TestCase):
    def test_p2_gate_requires_train_verify_and_196_hits(self):
        hc.check_gate(hc.load_script('ghp01_gate'))

    def test_p2_gate_cli_reads_files_and_prints_one_json_line(self):
        from vkgr.data_audit import file_sha256
        inputs = hc.gate_inputs()
        with tmpdir() as tmp:
            tmp = Path(tmp)
            (tmp / 'train_terminal.json').write_text(json.dumps(inputs['train_terminal']))
            (tmp / 'manifest.json').write_text(json.dumps(inputs['train_manifest']))
            sha = file_sha256(tmp / 'manifest.json')
            (tmp / 'verify_terminal.json').write_text(json.dumps(inputs['verify_terminal']))
            (tmp / 'stdout.log').write_text('noise\n' + json.dumps(inputs['verify_last_line']) + '\n')
            (tmp / 'run.json').write_text(json.dumps({'argv': ['--manifest-sha', sha]}))
            args = [sys.executable, '-B', str(ROOT / 'scripts' / 'ghp01_gate.py'), '--train-terminal', str(tmp / 'train_terminal.json'),
                    '--train-manifest', str(tmp / 'manifest.json'), '--verify-terminal', str(tmp / 'verify_terminal.json'),
                    '--verify-stdout', str(tmp / 'stdout.log'), '--verify-run', str(tmp / 'run.json')]
            done = subprocess.run(args, capture_output=True, text=True, check=True)
            self.assertEqual(len(done.stdout.strip().splitlines()), 1)
            self.assertIs(json.loads(done.stdout)['passed'], True)
            (tmp / 'stdout.log').unlink()
            done = subprocess.run(args, capture_output=True, text=True, check=True)
            value = json.loads(done.stdout)
            self.assertIs(value['passed'], False)
            self.assertIn('verify_stdout', value['missing'])


class ComboTests(unittest.TestCase):
    def test_p3_combo_trigger_levels_exclusions_and_selection(self):
        hc.check_combo(hc.load_script('ghp01_combo'))

    def test_p3_combo_cli_reads_manifests_and_prints_one_json_line(self):
        anchor, runs = hc.combo_case({'G1': 7, 'G4': 6})
        with tmpdir() as tmp:
            tmp = Path(tmp)

            def manifest(name, r50, n50, config):
                path = tmp / f'{name}-manifest.json'
                path.write_text(json.dumps({'config': config, 'selection': {'best_metric': [r50, n50]}}))
                return path

            def verify(name, status='verified'):
                path = tmp / f'{name}-stdout.log'
                path.write_text(json.dumps({'status': status}) + '\n')
                return path

            def terminal(name, status='succeeded'):
                path = tmp / f'{name}-terminal.json'
                path.write_text(json.dumps({'status': status}))
                return path
            args = [sys.executable, '-B', str(ROOT / 'scripts' / 'ghp01_combo.py'), '--anchor-manifest', str(manifest('a', anchor['r50'], anchor['n50'], anchor['config'])),
                    '--anchor-verify-stdout', str(verify('a'))]
            for r in runs:
                paths = (manifest(r['id'], r['r50'], r['n50'], r['config']), verify(r['id']), terminal(r['id']))
                args += ['--run', f"{r['id']}={','.join(map(str, paths))}"]
            done = subprocess.run(args, capture_output=True, text=True, check=True)
            self.assertEqual(len(done.stdout.strip().splitlines()), 1)
            value = json.loads(done.stdout)
            self.assertIs(value['triggered'], True)
            self.assertEqual(value['knobs_won'], ['width', 'layers'])
            self.assertEqual(value['selection'], 'G1')
            terminal('G4', 'failed')
            value = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
            self.assertIs(value['triggered'], False)
            self.assertIn('G4', value['excluded'])


class ParityScriptTests(unittest.TestCase):
    def test_p1_identical_factories_reported_equal_one_json_line(self):
        with tmpdir() as tmp:
            _, _, call = hc.parity_world(tmp)
            hc.check_parity_report(call)

    def test_p1_perturbed_new_factory_reported_unequal(self):
        with tmpdir() as tmp:
            _, _, call = hc.parity_world(tmp)
            hc.check_parity_perturbed(call)

    def test_p1_tampered_reference_or_bad_arguments_exit_1_without_output(self):
        with tmpdir() as tmp:
            _, value, call = hc.parity_world(tmp)
            code, out, err = call('ph-sha', manifest_sha='0' * 64)
            self.assertEqual((code, out), (1, ''))
            code, out, err = call('ph-steps', steps='0')
            self.assertEqual((code, out), (1, ''))
            best = Path(tmp) / 'run' / 'best.pt'
            best.write_bytes(best.read_bytes() + b'0')
            code, out, err = call('ph-tampered')
            self.assertEqual((code, out), (1, ''))
            self.assertIn('G-HP-01 parity gate failed', err)


if __name__ == '__main__':
    unittest.main()
