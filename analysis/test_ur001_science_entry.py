"""Formal full-scale front-door tests; no dataset or model execution."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ENTRY = Path(__file__).with_name('ur001_science_entry.py')
A = '/workspace/vkgr-ur001-data-20261020-v1'
H = '/workspace/vkgr-ur001-week26-holdout-20261021-v1'

def spec():
    return {'schema': 'ur001-science-inputs-v1',
            'train': [f'{A}/subsamples/ur0.01/train/week_{w:02}.parquet' for w in range(25)],
            'validation': f'{A}/subsamples/ur0.01/validation/week_25.parquet',
            'metadata': [f'{A}/metadata/{n}' for n in ('users_metadata.parquet', 'items_metadata.parquet', 'item_embeddings.npz')]}

class ScienceEntryTests(unittest.TestCase):
    def call(self, kind, path):
        self.assertTrue(ENTRY.is_file(), 'new formal front door is not implemented')
        return subprocess.run([sys.executable, '-I', '-B', str(ENTRY), kind, '--inputs', str(path)],
                              capture_output=True, text=True)

    def test_real_cli_admits_only_dry_and_science_stays_closed(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p = Path(d) / 'inputs.json'; p.write_text(json.dumps(spec()))
            run = self.call('dry', p)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            body = json.loads(run.stdout)
            self.assertEqual(body['status'], 'INPUTS_ADMITTED')
            self.assertFalse(body['execution_allowed'])
            self.assertFalse(body['data_opened'])
            for kind in ('train', 'eval', 'verify'):
                with self.subTest(kind=kind):
                    run = self.call(kind, p)
                    self.assertNotEqual(run.returncode, 0)
                    self.assertEqual(json.loads(run.stdout)['status'], 'SCIENCE_EXECUTION_CLOSED')

    def test_all_formal_cli_kinds_reject_holdout_before_data_open(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p = Path(d) / 'inputs.json'
            for kind in ('train', 'eval', 'verify'):
                for role in ('train', 'validation', 'metadata'):
                    with self.subTest(kind=kind, role=role):
                        value = spec()
                        if role == 'validation': value[role] = H + '/week_26.parquet'
                        else: value[role][0] = H + '/week_26.parquet'
                        p.write_text(json.dumps(value))
                        run = self.call(kind, p)
                        self.assertNotEqual(run.returncode, 0)
                        body = json.loads(run.stdout)
                        self.assertEqual(body['status'], 'HOLDOUT_INPUT_REJECTED')
                        self.assertFalse(body['data_opened'])

    def test_config_itself_in_holdout_is_rejected_before_open(self):
        for kind in ('train', 'eval', 'verify'):
            with self.subTest(kind=kind):
                run = self.call(kind, H + '/inputs.json')
                self.assertEqual(json.loads(run.stdout)['status'], 'HOLDOUT_INPUT_REJECTED')

    def test_actual_main_never_opens_data_paths(self):
        self.assertTrue(ENTRY.is_file(), 'new formal front door is not implemented')
        from analysis import ur001_science_entry as entry
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p = Path(d) / 'inputs.json'; value = spec(); value['validation'] = H + '/week_26.parquet'
            p.write_text(json.dumps(value)); original = io.open
            def guarded(file, *args, **kwargs):
                if str(file).startswith('/workspace/'):
                    raise AssertionError('data path was opened before rejection')
                return original(file, *args, **kwargs)
            with patch('io.open', side_effect=guarded), patch('sys.stdout', new=io.StringIO()) as output:
                code = entry.main(['train', '--inputs', str(p)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())['status'], 'HOLDOUT_INPUT_REJECTED')

    def test_extra_roles_unknown_paths_and_symlinks_fail_closed(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            p = Path(d) / 'inputs.json'
            mutations = []
            value = spec(); value['holdout'] = H; mutations.append(value)
            value = spec(); value['train'][0] = A + '/subsamples/ur0.01/train/../test/week_26.parquet'; mutations.append(value)
            value = spec(); value['metadata'][0] = '/workspace/other/users_metadata.parquet'; mutations.append(value)
            value = spec(); value['train'][0] = value['train'][1]; mutations.append(value)
            for value in mutations:
                p.write_text(json.dumps(value)); run = self.call('dry', p)
                self.assertNotEqual(run.returncode, 0)
                self.assertIn(json.loads(run.stdout)['status'], ('INPUTS_REJECTED', 'HOLDOUT_INPUT_REJECTED'))
            p.write_text(json.dumps(spec())); link = Path(d) / 'link.json'; link.symlink_to(p)
            run = self.call('dry', link)
            self.assertEqual(json.loads(run.stdout)['status'], 'INPUTS_REJECTED')

    def test_published_policy_and_asset_roots_are_bound_to_the_real_front_door(self):
        from analysis import ur001_science_entry as entry
        from scripts import ur001_week26_ops as ops
        self.assertEqual(entry.A_ROOT, Path(ops.DEFAULT_ROOTS['a_root']))
        self.assertEqual(set(entry.FORBIDDEN_ROOTS), {Path(ops.DEFAULT_ROOTS[k]) for k in ('raw_root', 'membership_root', 'holdout_root')})
        policy = Path(__file__).parents[1] / 'configs/ur001-preparation/science-inputs-closed.json'
        self.assertEqual(json.loads(policy.read_text()), spec())
        run = self.call('dry', policy)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
        self.assertFalse(json.loads(run.stdout)['execution_allowed'])

    def test_config_path_is_denied_by_actual_main_before_any_data_open(self):
        from analysis import ur001_science_entry as entry
        original = io.open
        def guarded(file, *args, **kwargs):
            if str(file).startswith('/workspace/'):
                raise AssertionError('restricted configuration path was opened')
            return original(file, *args, **kwargs)
        with patch('io.open', side_effect=guarded), patch('sys.stdout', new=io.StringIO()) as output:
            code = entry.main(['verify', '--inputs', H + '/inputs.json'])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(output.getvalue())['status'], 'HOLDOUT_INPUT_REJECTED')

    def test_unknown_cli_command_fails(self):
        run = self.call('not-a-command', '/unused')
        self.assertNotEqual(run.returncode, 0)

if __name__ == '__main__':
    unittest.main()
