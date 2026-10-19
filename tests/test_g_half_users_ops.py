import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SPEC = importlib.util.spec_from_file_location('g_half_users_ops', ROOT / 'scripts/g_half_users_ops.py')
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)


def ops_flag(argv, name):
    if argv.count(name) != 1:
        raise AssertionError('expected single argument ' + name)
    return argv[argv.index(name)+1]


class GHalfUsersOpsTests(unittest.TestCase):
    def test_pinned_delta_is_the_exact_new_file_inventory(self):
        from vkgr import g_half_users_a1_run as half
        expected = {
            'src/vkgr/g_half_users_a1_run.py',
            'scripts/g_half_users_ops.py',
            'scripts/g_half_users_engineering_queue.py',
            'tests/test_g_half_users.py',
            'tests/test_g_half_users_ops.py',
            'tests/test_g_half_users_engineering_queue.py',
            'configs/g_half_users/g-half-users-s42.json',
            'configs/g_half_users/g-half-users-s2024.json',
            'configs/g_half_users/g-half-users-s2025.json',
            'docs/superpowers/plans/2026-10-20-g-half-implementation.md',
        }
        self.assertEqual(set(ops.DELTA_PATHS), expected)
        self.assertEqual(set(half.SOURCE_MEMBERS) | {path for path in expected if path.startswith('configs/')}, expected)
        self.assertTrue(all((ROOT / path).is_file() for path in expected))

    def test_deployment_inventory_is_exact_anchor_union_frozen_delta(self):
        anchor = {f'old/source-{index}.py': str(index % 10) * 64 for index in range(45)}
        delta = {name: str(index % 10) * 64 for index, name in enumerate(ops.DELTA_PATHS, 1)}
        files = {**anchor, **delta}
        self.assertEqual(ops.check_member_sets(files, anchor, delta, accepted_training_files=anchor),
                         {'training_file_count': 45, 'pinned_delta_file_count': len(delta)})
        with self.assertRaises(ValueError):
            ops.check_member_sets({**files, 'extra.py': 'f' * 64}, anchor, delta, accepted_training_files=anchor)
        with self.assertRaises(ValueError):
            ops.check_member_sets(files, {**anchor, 'src/changed.py': 'b' * 64}, delta,
                                  accepted_training_files=anchor)
        rejected_44 = dict(list(anchor.items())[:-1])
        with self.assertRaises(ValueError):
            ops.check_member_sets({**rejected_44, **delta}, rejected_44, delta, accepted_training_files=anchor)
        rejected_sha = {**anchor, 'old/source-00.py': 'f' * 64}
        with self.assertRaises(ValueError):
            ops.check_member_sets({**rejected_sha, **delta}, rejected_sha, delta, accepted_training_files=anchor)

    def test_original_ops_request_pin_is_frozen_and_byte_verified(self):
        raw = {'source_commit': ops.ANCHOR_SOURCE_COMMIT,
               'files': {f'old/source-{index}.py': str(index % 10) * 64 for index in range(45)}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'a1-request.json'
            path.write_text(json.dumps(raw))
            pin = ops.sha(path)
            with mock.patch.object(ops, 'ANCHOR_SOURCE_REQUEST_SHA256', pin):
                result = ops.read_accepted_source_request({'path': str(path), 'sha256': pin})
            self.assertEqual(result['source_commit'], ops.ANCHOR_SOURCE_COMMIT)
            self.assertEqual(len(result['files']), 45)
            with mock.patch.object(ops, 'ANCHOR_SOURCE_REQUEST_SHA256', 'a'*64), self.assertRaises(ValueError):
                ops.read_accepted_source_request({'path': str(path), 'sha256': pin})

    def test_science_41_plus_delta_10_and_ops_45_plus_delta_10_are_separate(self):
        ops_files = {f'old/ops-{index}.py': str(index % 10) * 64 for index in range(45)}
        science_files = dict(list(ops_files.items())[:41])
        delta = {name: str(index % 10) * 64 for index, name in enumerate(ops.DELTA_PATHS, 1)}
        ops_ledger = {**ops_files, **delta}
        science_ledger = {**science_files, **delta}
        self.assertEqual(ops.check_member_sets(ops_ledger, ops_files, delta, accepted_training_files=ops_files),
                         {'training_file_count': 45, 'pinned_delta_file_count': 10})
        self.assertEqual(ops.check_member_sets(science_ledger, science_files, delta,
                                               accepted_training_files=science_files),
                         {'training_file_count': 41, 'pinned_delta_file_count': 10})
        with self.assertRaises(ValueError):
            ops.check_member_sets({**ops_files, **delta}, {**ops_files, 'old/ops-00.py': 'f'*64}, delta,
                                  accepted_training_files=ops_files)

    def test_accepted_anchor_manifest_binds_41_file_source_and_runtime(self):
        source_files = {f'src/vkgr/source-{index}.py': str(index % 10) * 64 for index in range(40)}
        source_files['uv.lock'] = 'f' * 64
        environment = {'python': '3.11.16', 'exact_dependencies':
                       {'numpy': '1.26.4', 'torch': '2.11.0+cu128', 'pyarrow': '19.0.1', 'PyYAML': '6.0.2'}}
        value = {'schema': 'g-sid-structure-a1-train-manifest-v1', 'kind': 'g_sid_structure_a1_train',
                 'status': 'complete', 'config': {'seed': 42, 'sid': 'STRUCT_C1_B48D3_S42'},
                 'binding': {'task': 'SID-STRUCTURE-MATCHED', 'context_binding_sha256': 'a'*64,
                             'source': {'files': source_files, 'git': {'git_base': 'b'*40, 'git_dirty': False}},
                             'runtime': {'environment': environment}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'anchor-manifest.json'; path.write_text(json.dumps(value))
            sha = ops.sha(path)
            with mock.patch.object(ops, 'ANCHOR_MANIFEST_SHA256', sha):
                anchor = ops.read_accepted_anchor_manifest({'path': str(path), 'sha256': sha})
            self.assertEqual(len(anchor['source_files']), 41)
            self.assertEqual(anchor['runtime'], {**environment, 'lock_sha256': 'f'*64})
            with mock.patch.object(ops, 'ANCHOR_MANIFEST_SHA256', 'c'*64), self.assertRaises(ValueError):
                ops.read_accepted_anchor_manifest({'path': str(path), 'sha256': sha})

    def test_full_source_contract_binds_accepted_45_plus_10_and_science_41_plus_10(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            root = tmp/'deployment'; root.mkdir()
            control = tmp/'control'; control.mkdir()
            contents = {}
            def pin(name, raw):
                contents[name] = raw
                return hashlib.sha256(raw).hexdigest()
            science = {f'src/vkgr/source-{index:02d}.py': pin(f'src/vkgr/source-{index:02d}.py',
                                                               f'old science {index}\n'.encode()) for index in range(40)}
            source_lock = ROOT/'uv.lock'
            contents['uv.lock'] = source_lock.read_bytes()
            science['uv.lock'] = hashlib.sha256(contents['uv.lock']).hexdigest()
            extra_ops = {f'scripts/old-ops-{index}.py': pin(f'scripts/old-ops-{index}.py',
                                                           f'old ops {index}\n'.encode()) for index in range(4)}
            accepted_ops_files = {**science, **extra_ops}
            delta = {name: pin(name, ('pinned delta ' + name + '\n').encode()) for name in ops.DELTA_PATHS}
            for name, pin_value in accepted_ops_files.items():
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[name])
            subprocess.run(['git','init','-q',str(root)],check=True)
            subprocess.run(['git','-C',str(root),'config','user.name','Review Fixture'],check=True)
            subprocess.run(['git','-C',str(root),'config','user.email','review-fixture@example.invalid'],check=True)
            subprocess.run(['git','-C',str(root),'add','-A'],check=True)
            subprocess.run(['git','-C',str(root),'commit','-qm','accepted source fixture'],check=True)
            original_source_commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
            for name, pin_value in delta.items():
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents[name])
            subprocess.run(['git','-C',str(root),'add','-A'],check=True)
            subprocess.run(['git','-C',str(root),'commit','-qm','pinned delta fixture'],check=True)
            source_commit = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
            files = {**accepted_ops_files, **delta}

            old_request = control/'a1-request-v1.json'
            old_request.write_text(json.dumps({'source_commit':original_source_commit,'files':accepted_ops_files}))
            old_request_sha = ops.sha(old_request)
            runtime = ops.actual_runtime(sys.executable, root/'uv.lock')
            anchor = {'schema':'g-sid-structure-a1-train-manifest-v1','kind':'g_sid_structure_a1_train',
                      'status':'complete','config':{'seed':42,'sid':'STRUCT_C1_B48D3_S42'},
                      'binding':{'task':'SID-STRUCTURE-MATCHED','context_binding_sha256':'a'*64,
                                 'source':{'files':science,'git':{'git_base':original_source_commit,'git_dirty':False}},
                                 'runtime':{'environment':{k:runtime[k] for k in ('python','exact_dependencies')}}}}
            anchor_path = control/'anchor-manifest.json'; anchor_path.write_text(json.dumps(anchor))
            anchor_sha = ops.sha(anchor_path)
            req = {'deployment':str(root),'source_commit':source_commit,'files':files,
                   'training_files':accepted_ops_files,'pinned_delta':delta,'training_runtime':runtime,
                   'python':sys.executable,
                   'anchor_source_request':{'path':str(old_request),'sha256':old_request_sha},
                   'anchor_manifest':{'path':str(anchor_path),'sha256':anchor_sha}}
            with mock.patch.object(ops,'ANCHOR_SOURCE_REQUEST_SHA256',old_request_sha), \
                    mock.patch.object(ops,'ANCHOR_SOURCE_COMMIT',original_source_commit), \
                    mock.patch.object(ops,'ANCHOR_MANIFEST_SHA256',anchor_sha):
                identity = ops.validate_source_contract(req)
                self.assertEqual(identity['ops_source_file_count'],55)
                self.assertEqual(identity['science_source_file_count'],51)
                self.assertEqual(identity['anchor_ops_file_count'],45)
                self.assertEqual(identity['anchor_science_file_count'],41)
                self.assertEqual(identity['pinned_delta_file_count'],10)

                missing_old = copy.deepcopy(req)
                missing = next(iter(accepted_ops_files))
                missing_old['training_files'].pop(missing); missing_old['files'].pop(missing)
                with self.assertRaises(ValueError): ops.validate_source_contract(missing_old)

                changed_old_sha = copy.deepcopy(req)
                changed_old_sha['training_files'][missing] = 'e'*64
                changed_old_sha['files'][missing] = 'e'*64
                with self.assertRaises(ValueError): ops.validate_source_contract(changed_old_sha)

                unpinned_delta = copy.deepcopy(req)
                unpinned_delta['pinned_delta']['src/vkgr/unregistered.py'] = 'd'*64
                unpinned_delta['files']['src/vkgr/unregistered.py'] = 'd'*64
                with self.assertRaises(ValueError): ops.validate_source_contract(unpinned_delta)

                wrong_request_sha = copy.deepcopy(req)
                wrong_request_sha['anchor_source_request']['sha256'] = 'f'*64
                with self.assertRaises(ValueError): ops.validate_source_contract(wrong_request_sha)
                raw_request_bytes = old_request.read_bytes()
                old_request.write_bytes(raw_request_bytes + b'\n')
                with self.assertRaises(ValueError): ops.validate_source_contract(req)
                old_request.write_bytes(raw_request_bytes)

                for label, relative, content in (
                    ('old_tracked_change', next(name for name in science if name != 'uv.lock'), b'changed old bytes\n'),
                    ('unlisted_tree_addition', 'src/vkgr/unlisted-tree-path.py', b'unlisted\n'),
                ):
                    clone = tmp/label
                    subprocess.run(['git','clone','-q',str(root),str(clone)],check=True)
                    subprocess.run(['git','-C',str(clone),'config','user.name','Review Fixture'],check=True)
                    subprocess.run(['git','-C',str(clone),'config','user.email','review-fixture@example.invalid'],check=True)
                    changed_path = clone/relative; changed_path.parent.mkdir(parents=True,exist_ok=True)
                    changed_path.write_bytes(content)
                    subprocess.run(['git','-C',str(clone),'add','-A'],check=True)
                    subprocess.run(['git','-C',str(clone),'commit','-qm',label],check=True)
                    changed_req = copy.deepcopy(req)
                    changed_req['deployment'] = str(clone)
                    changed_req['source_commit'] = subprocess.check_output(
                        ['git','-C',str(clone),'rev-parse','HEAD'],text=True).strip()
                    with self.subTest(tree_mutation=label), self.assertRaises(ValueError):
                        ops.validate_source_contract(changed_req)

    def test_verify_stdout_requires_one_bound_verified_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'stdout.log'
            row = {'status': 'verified', 'run_id': 'g-half-users-s42',
                   'manifest_sha256': 'a' * 64, 'verify_run_id': 'g-half-users-s42-verify'}
            path.write_text('progress\n' + json.dumps(row) + '\n')
            result = ops.verify_stdout(path, ops.sha(path), run_id=row['run_id'],
                                       manifest_sha256=row['manifest_sha256'], verify_run_id=row['verify_run_id'])
            self.assertEqual(result, row)
            path.write_text('progress\n' + json.dumps({**row, 'run_id': 'other'}) + '\n')
            with self.assertRaises(ValueError):
                ops.verify_stdout(path, ops.sha(path), run_id=row['run_id'],
                                  manifest_sha256=row['manifest_sha256'], verify_run_id=row['verify_run_id'])

    def test_derived_job_keeps_anchor_recipe_and_binds_new_runner_and_sample_config(self):
        from vkgr import g_half_users_a1_run as half
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old_config = json.loads((ROOT / 'configs/sid_structure/g-structure-a1-s42.json').read_text())
            old_config_path = root / 'configs/sid_structure/g-structure-a1-s42.json'
            old_config_path.parent.mkdir(parents=True)
            old_config_path.write_text(json.dumps(old_config))
            half_config = {**old_config, 'schema': half.SCHEMA, 'task_id': half.TASK_ID,
                           'user_sample_fraction': 0.5, 'user_sample_seed': 20261020,
                           'user_sample_count_rule': half.SAMPLE_COUNT_RULE,
                           'user_sample_method': half.SAMPLE_METHOD}
            half_config_path = root / 'configs/g_half_users/g-half-users-s42.json'
            half_config_path.parent.mkdir(parents=True)
            half_config_path.write_text(json.dumps(half_config))

            def anchor_job(req, seed, vram):
                return {'run_id': 'g-structure-a1-s42', 'seed': seed, 'cwd': str(root),
                        'config': str(old_config_path), 'vram_mib': vram, 'inputs': ['prepared'],
                        'env': ['VKGR_RUN_ID=old'], 'argv': ['python', '-B', '-m', 'vkgr.coll_sid_a1_run', 'train',
                            '--config', 'old.json', '--config-sha', 'a'*64, '--output', 'old/out', '--scratch', 'old/scratch']}

            fake_old_ops = SimpleNamespace(derive_job=anchor_job,
                load=lambda path: json.loads(Path(path).read_text()),
                replace=lambda argv, name, value: argv.__setitem__(argv.index(name)+1, str(value)))
            req = {'deployment': str(root), 'runs': str(root/'runs'), 'deadline_epoch': 1790000000,
                   'files': {'src/vkgr/g_half_users_a1_run.py': 'a'*64},
                   'coll_refs': {'sid_manifest': {'sha256': old_config['sid_manifest_sha256']}},
                   'expected_selected_users_sha256': 'b' * 64}
            with mock.patch.object(ops, '_old_ops', return_value=fake_old_ops), \
                    mock.patch.object(ops, '_half_module', return_value=half):
                job = ops.derive_job(req, 42, 4096)
            self.assertEqual(job['run_id'], 'g-half-users-r2-s42')
            self.assertEqual(job['argv'][job['argv'].index('-m')+1], 'vkgr.g_half_users_a1_run')
            self.assertEqual(ops_flag(job['argv'], '--config-sha'), ops.sha(half_config_path))
            self.assertEqual(ops_flag(job['argv'], '--config'), 'configs/g_half_users/g-half-users-s42.json')
            self.assertEqual(job['vram_mib'], 4096)
            self.assertNotIn('VKGR_RUN_ID=g-half-users-r2-s42', job['env'])
            self.assertFalse(any(value.startswith('VKGR_PROGRESS_PATH=') for value in job['env']))
            self.assertIn('VKGR_EXPECTED_SELECTED_USERS_SHA256=' + 'b'*64, job['env'])
            self.assertIn('VKGR_DRIVER_DEADLINE_EPOCH=1790000000', job['env'])
            verify = ops.derive_verify(job, 'c' * 64, root/'runs')
            self.assertEqual(verify['run_id'], 'g-half-users-r2-s42-verify')
            self.assertNotIn('VKGR_RUN_ID=g-half-users-r2-s42-verify', verify['env'])
            self.assertFalse(any(value.startswith('VKGR_PROGRESS_PATH=') for value in verify['env']))
            self.assertIn('VKGR_EXPECTED_SELECTED_USERS_SHA256=' + 'b'*64, verify['env'])
            self.assertEqual(ops_flag(verify['argv'], '--manifest-sha'), 'c'*64)

    def test_preflight_seal_checks_all_three_jobs_share_the_selected_users_sha(self):
        selected_sha = 'b' * 64
        req = {'expected_selected_users_sha256': selected_sha}
        preflight = {'status': 'preflight_ok', 'expected_selected_users_sha256': selected_sha,
                     'entries': [{'seed': seed,
                                  'training_subset': {'selected_users_sha256': selected_sha},
                                  'job': {'run_id': f'g-half-users-r2-s{seed}',
                                          'env': ['VKGR_EXPECTED_SELECTED_USERS_SHA256=' + selected_sha]}}
                                 for seed in ops.SEEDS]}
        self.assertEqual(ops.check_request_selection_against_preflight(req, preflight), selected_sha)
        preflight['entries'][1]['training_subset']['selected_users_sha256'] = 'c'*64
        with self.assertRaises(ValueError):
            ops.check_request_selection_against_preflight(req, preflight)

    def test_module_preflight_uses_full_training_argv_and_binds_raw_cli_receipt(self):
        selected = 'a' * 64
        source_sha = 'b' * 64
        req = {'deployment': str(ROOT), 'ops': tempfile.gettempdir(),
               'python': sys.executable, 'expected_selected_users_sha256': selected,
               'anchor_manifest': {'path': '/private/anchor/manifest.json', 'sha256': 'c' * 64}}
        argv = [sys.executable, '-B', '-m', 'vkgr.g_half_users_a1_run', 'train',
                '--prepared', 'prepared', '--prepared-sha', '1'*64,
                '--history-manifest', 'history', '--history-sha', '2'*64,
                '--sid-manifest', 'sid', '--sid-sha', '3'*64,
                '--content', 'content', '--content-sha', '4'*64,
                '--input-manifest', 'input', '--input-sha', '5'*64,
                '--vectors', 'vectors', '--vectors-sha', '6'*64,
                '--output', 'old-output', '--scratch', 'old-scratch',
                '--config', 'config', '--config-sha', '7'*64]
        expected = {'status': 'preflight_ok', 'engineering': False, 'written': False,
                    'training_subset': {'selected_users_sha256': selected},
                    'source_sha256': source_sha}
        def child(*args, **kwargs):
            command = args[0]
            self.assertEqual(command[:5], [sys.executable, '-B', '-m',
                                           'vkgr.g_half_users_a1_run', 'preflight'])
            self.assertEqual(ops_flag(command, '--prepared-sha'), '1'*64)
            self.assertNotEqual(ops_flag(command, '--output'), 'old-output')
            self.assertNotEqual(ops_flag(command, '--scratch'), 'old-scratch')
            self.assertEqual(ops_flag(command, '--anchor-manifest'), req['anchor_manifest']['path'])
            self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '1')
            return SimpleNamespace(returncode=0, stdout=json.dumps(expected) + '\n', stderr='')
        with mock.patch.object(ops.subprocess, 'run', side_effect=child):
            result = ops.run_module_preflight(req, {'argv': argv, 'env': []}, source_sha)
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['receipt'], expected)
        self.assertEqual(result['stdout_sha256'], hashlib.sha256((json.dumps(expected)+'\n').encode()).hexdigest())
        call_dir = Path(result['call_dir'])
        self.assertTrue(call_dir.is_dir())
        self.assertEqual(json.loads((call_dir/'argv.json').read_text()), result['argv'])
        self.assertEqual((call_dir/'stdout.log').read_text(), json.dumps(expected)+'\n')
        self.assertEqual((call_dir/'stderr.log').read_text(), '')
        self.assertEqual((call_dir/'returncode.txt').read_text(), '0\n')
        stored = json.loads((call_dir/'evidence.json').read_text())
        self.assertEqual(stored['argv_sha256'], hashlib.sha256(ops.canonical(result['argv'])).hexdigest())
        self.assertEqual(stored['stdout_sha256'], result['stdout_sha256'])
        for bad in (SimpleNamespace(returncode=1, stdout=json.dumps(expected)),
                    SimpleNamespace(returncode=0, stdout=''),
                    SimpleNamespace(returncode=0, stdout='not-json'),
                    SimpleNamespace(returncode=0, stdout=json.dumps({**expected,
                        'training_subset': {'selected_users_sha256': 'd'*64}}))):
            bad.stderr = 'captured error'
            with mock.patch.object(ops.subprocess, 'run', return_value=bad), self.assertRaises(ValueError) as caught:
                ops.run_module_preflight(req, {'argv': argv, 'env': []}, source_sha)
            evidence = Path(str(caught.exception).rsplit(' ', 1)[-1])
            self.assertTrue((evidence/'argv.json').is_file())
            self.assertTrue((evidence/'stdout.log').is_file())
            self.assertTrue((evidence/'stderr.log').is_file())
            self.assertTrue((evidence/'returncode.txt').is_file())
            self.assertTrue((evidence/'evidence.json').is_file())

    def test_formal_preflight_keeps_module_identity_and_uses_mapping_before_m_subprocess(self):
        source = types.ModuleType('sid_structure_a1_run')
        source_identity = {'science_source_sha256': 'b'*64}
        req = {'deployment': str(ROOT), 'ops': tempfile.gettempdir(), 'python': sys.executable,
               'expected_selected_users_sha256': 'a'*64,
               'anchor_manifest': {'path': '/private/anchor/manifest.json', 'sha256': 'c'*64}}
        job = {'argv': [sys.executable, '-B', '-m', 'vkgr.g_half_users_a1_run', 'train',
                        '--prepared', 'p', '--prepared-sha', '1'*64,
                        '--history-manifest', 'h', '--history-sha', '2'*64,
                        '--sid-manifest', 's', '--sid-sha', '3'*64,
                        '--content', 'c', '--content-sha', '4'*64,
                        '--input-manifest', 'i', '--input-sha', '5'*64,
                        '--vectors', 'v', '--vectors-sha', '6'*64,
                        '--output', 'out', '--scratch', 'scratch', '--config', 'cfg', '--config-sha', '7'*64],
               'env': []}
        fake_ops = SimpleNamespace(identity=lambda value: source, prepare=lambda value: None)
        mocked_child = SimpleNamespace(returncode=0, stdout=json.dumps({'status': 'preflight_ok',
            'engineering': False, 'written': False, 'training_subset': {'selected_users_sha256': 'a'*64},
            'source_sha256': 'b'*64})+'\n', stderr='')
        with mock.patch.object(ops, 'validate_source_contract', return_value=source_identity), \
             mock.patch.object(ops, '_old_ops', return_value=fake_ops), \
             mock.patch.object(ops, 'derive_job', return_value=job), \
             mock.patch.object(ops.subprocess, 'run', return_value=mocked_child) as run_child, \
             mock.patch.object(ops, 'run_module_preflight', wraps=ops.run_module_preflight), \
             mock.patch.object(fake_ops, 'prepare', side_effect=RuntimeError('stop after CLI')):
            with self.assertRaisesRegex(RuntimeError, 'stop after CLI'):
                ops.formal_preflight(req)
        run_child.assert_called_once()

    def test_formal_preflight_compares_each_cli_validation_pin_before_science_loop(self):
        pins = {'context_binding_sha256': 'a'*64,
                'queries_sha256': 'b'*64, 'validation_file_sha256': 'c'*64}
        context = object()
        anchor = {'config': {'seed': 42, 'sid': 'STRUCT_C1_B48D3_S42'},
                  'binding': {'task': 'task', 'context_binding_sha256': 'd'*64,
                              'source': {'git': {'git_base': 'e'*40, 'git_dirty': False}}}}
        run = SimpleNamespace(TASK_ID='task', _read_run=lambda *args: anchor)
        fake_ops = SimpleNamespace(identity=lambda req: types.ModuleType('sid_structure_a1_run'),
                                   prepare=lambda req: (run, context, object(), {}, 'scratch'),
                                   derive_job=lambda req, seed: {'argv': ['python', '-B', '-m', 'x', 'train']})
        req = {'deployment': str(ROOT), 'ops': tempfile.gettempdir(), 'python': sys.executable,
               'expected_selected_users_sha256': 'f'*64,
               'anchor_manifest': {'path': '/private/anchor/manifest.json', 'sha256': '0'*64}}
        half = SimpleNamespace(validate_anchor_context=lambda *args, **kwargs: pins)
        for key in pins:
            for wrong in ('missing', 'wrong'):
                receipt = dict(pins)
                if wrong == 'missing':
                    receipt.pop(key)
                else:
                    receipt[key] = '9'*64
                cli = {'receipt': {'status': 'preflight_ok', 'engineering': False,
                                   'written': False, 'training_subset': {'selected_users_sha256': 'f'*64},
                                   'source_sha256': '1'*64, **receipt},
                       'argv': [], 'returncode': 0, 'stdout_sha256': '2'*64,
                       'stderr_sha256': '3'*64, 'call_dir': '/tmp/call'}
                with self.subTest(pin=key, failure=wrong), \
                     mock.patch.object(ops, 'identity', return_value=fake_ops.identity(req)), \
                     mock.patch.object(ops, 'validate_source_contract', return_value={'science_source_sha256':'1'*64}), \
                     mock.patch.object(ops, '_old_ops', return_value=fake_ops), \
                     mock.patch.object(ops, 'derive_job', return_value={'argv':['python','train']}), \
                     mock.patch.object(ops, 'run_module_preflight', return_value=cli), \
                     mock.patch.object(ops, '_half_module', return_value=half), \
                     self.assertRaisesRegex(ValueError, 'validation pin mismatch'):
                    ops.formal_preflight(req)

    def test_anchor_context_adapter_has_no_formal_cli_local_dependency(self):
        pins = {'context_binding_sha256': 'a'*64, 'queries_sha256': 'b'*64,
                'validation_file_sha256': 'c'*64}
        anchor = {'config': {'seed': 42, 'sid': 'STRUCT_C1_B48D3_S42'},
                  'binding': {'task': 'task', 'context_binding_sha256': 'd'*64,
                              'source': {'git': {'git_base': 'e'*40, 'git_dirty': False}}}}
        run = SimpleNamespace(TASK_ID='task', _read_run=lambda *args: anchor)
        half = SimpleNamespace(validate_anchor_context=lambda *args, **kwargs: pins)
        with mock.patch.object(ops, '_half_module', return_value=half):
            result = ops._anchor_context({'deployment': str(ROOT),
                                         'anchor_manifest': {'path': '/private/anchor', 'sha256': 'f'*64}},
                                         run, object())
        self.assertEqual(result, (half, anchor, pins))

    def test_verified_observe_recomputes_delta_from_g_half_anchor_controls(self):
        raw_by_seed = {42: 408, 2024: 392, 2025: 384}
        calls = []
        def old_observe(req, entry):
            seed = entry['job']['seed']; hits = raw_by_seed[seed]
            calls.append(seed)
            return {'run_id': entry['job']['run_id'], 'seed': seed, 'verified': True, 'status': 'verified',
                    'hits': hits, 'delta_hits': hits-{42:388, 2024:376, 2025:384}[seed]}
        fake_ops = SimpleNamespace(observe=old_observe)
        with mock.patch.object(ops, '_old_ops', return_value=fake_ops):
            results = [ops.observe({'deployment':'/tmp'}, {'job':{'run_id':f'g-half-users-s{seed}','seed':seed}})
                       for seed in ops.SEEDS]
        self.assertEqual(calls, list(ops.SEEDS))
        self.assertEqual([result['control_hits'] for result in results], [398, 384, 384])
        self.assertEqual([result['delta_hits'] for result in results], [10, 8, 0])
        self.assertEqual([result['hits'] for result in results], [408, 392, 384])
        # The old structure controls create +10/+8/0 inflation for the same raw hits.
        self.assertEqual([raw_by_seed[seed]-{42:388, 2024:376, 2025:384}[seed] for seed in ops.SEEDS],
                         [20, 16, 0])

    def test_unverified_observe_preserves_original_result(self):
        result = {'run_id': 'g-half-users-s42', 'seed': 42, 'verified': False,
                  'status': 'running', 'hits': None, 'delta_hits': None}
        fake_ops = SimpleNamespace(observe=lambda req, entry: dict(result))
        with mock.patch.object(ops, '_old_ops', return_value=fake_ops):
            observed = ops.observe({'deployment': '/tmp'}, {'job': {'seed': 42}})
        self.assertEqual(observed, result)

    def test_adapter_prepare_uses_original_full_input_gate_before_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            full = object(); sid = object(); run = object(); runtime = {'device': 'cuda:0'}; scratch = root/'scratch'
            fake = SimpleNamespace(__file__='original_ops.py')
            raw_derive = lambda req, seed, vram=None: {'seed': seed, 'vram': vram}
            observations = []

            def raw_prepare(req):
                observations.append(fake.derive_job is raw_derive)
                return run, full, sid, runtime, scratch

            fake.identity = lambda req: object()
            fake.derive_job = raw_derive
            fake.derive_verify = lambda *args: None
            fake.prepare = raw_prepare
            fake.start_driver = lambda *args: Path(fake.__file__).name
            fake.cuda_preview = lambda req: fake.prepare(req)[1]
            fake.dry_run = lambda req: None
            fake.submit = lambda req: fake.start_driver(req, root/'request.json')
            fake.driver = lambda req, sha: None
            half = SimpleNamespace(sample_context=lambda context, **kwargs: ('sampled', kwargs),
                                   check_selected_users_pin=lambda *args, **kwargs: None)
            req = {'deployment': str(root), 'expected_selected_users_sha256': 'a'*64}
            with mock.patch.object(ops, '_old_ops', return_value=fake), \
                    mock.patch.object(ops, '_half_module', return_value=half), \
                    mock.patch.object(ops, '_anchor_context', return_value=(half, {}, {})):
                with ops.adapted_old_ops(req) as adapted:
                    prepared = adapted.prepare(req)
                    self.assertEqual(prepared, (run, ('sampled', {'expected_selected_users_sha256': 'a'*64}),
                                                sid, runtime, scratch))
                self.assertEqual(observations, [True])
                self.assertIs(fake.derive_job, raw_derive)
                self.assertEqual(ops.submit(req), 'g_half_users_ops.py')
                self.assertEqual(fake.__file__, 'original_ops.py')


if __name__ == '__main__':
    unittest.main()
