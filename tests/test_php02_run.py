"""P-HP-02 (temperature sweep 0.05 / 0.07 on the F4 anchor) runner tests.

Equivalence: php02 T010 == php01 H1 and php02 T015 == FEAT-02 F4, epoch for epoch (CPU synthetic).
Rejection: only (T005, 0.05) and (T007, 0.07) are real; every other temperature is refused loudly.
``guard_*`` functions are plain assertion functions; tests/test_php02_mutations.py re-runs them under wrong implementations.
"""
import contextlib
import io
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_model import ensure  # noqa: E402
from test_feat02_run import flip_non_history_fields  # noqa: E402
from test_php01_run import f4_small, _first_batch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'configs' / 'php02'
H1_DIR = ROOT / 'configs' / 'php01'
# independent literals (design card section 2 / 3)
REAL = {'T005': 0.05, 'T007': 0.07}
SEEDS = (42, 2024, 2025)
H1_SHA = {42: '299df44ba16e982a400e67bf384756d4f3bc12f8422aaf6069100fb38cb2b913',
          2024: '2d710a75bb415eeaa13b0da475df6c31204159c73070a2898a8033431def897b',
          2025: '94aa8b2660632b13df441eee55fb82e740a766d779939225453d35a097bbb5c0'}
RUN_IDS = {(k, s): f'p-php02-t{"005" if k == "T005" else "007"}-s{s}' for k in REAL for s in SEEDS}


def config02(knob, temperature=None, synthetic_shape=False, **overrides):
    from vkgr.retrieval_php02_run import default_config
    value = {**default_config(), 'knob': knob,
             'temperature': {**REAL, 'T010': 0.1, 'T015': 0.15}[knob] if temperature is None else temperature}
    if synthetic_shape:
        value.update(batch_size=16, max_epochs=8, min_epochs=2)
    value.update(overrides)
    return value


def small02(knob, **overrides):
    return config02(knob, synthetic_shape=True, **overrides)


def h1_small():
    from vkgr.retrieval_php01_run import default_config
    return {**default_config(), 'knob': 'H1', 'temperature': 0.1, 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}


# ---------------------------------------------------------------- guards (reused by the mutation tests)

def guard_rejects_near_miss_temperature():
    from vkgr.retrieval_php02_run import validate_config
    for knob, near in (('T005', 0.0500001), ('T005', 0.05000000001), ('T007', 0.0700001), ('T007', 0.06999999999)):
        for synthetic in (False, True):
            try:
                validate_config(config02(knob, near), synthetic)
            except ValueError:
                continue
            raise AssertionError(f'{knob} accepted near-miss temperature {near} (synthetic={synthetic})')


def guard_equivalence_knobs_are_synthetic_only():
    from vkgr.retrieval_php02_run import validate_config
    for knob in ('T010', 'T015'):
        validate_config(small02(knob), synthetic=True)
        try:
            validate_config(config02(knob), synthetic=False)
        except ValueError:
            continue
        raise AssertionError(f'{knob} accepted as a real run')


def guard_first_step_loss_uses_config_temperature_p02():
    from vkgr import retrieval_php02_run as p2
    from vkgr.retrieval_training import SamplerStats
    from test_php01_run import _independent_loss
    for knob, temperature in (('T005', 0.05), ('T007', 0.07)):
        config = small02(knob)
        data, rows, batch, frequency = _first_batch(config)
        model = p2._model(data, config, 'cpu'); optimizer = p2.make_optimizer(model, config)
        record = p2._step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)
        expected, anchor = _independent_loss(config, temperature), _independent_loss(config, 0.15)
        ensure(abs(expected - anchor) > 1e-4, 'fixture cannot tell the temperatures apart')
        ensure(abs(record['loss'] - expected) < 1e-6, f'{knob}: first-step loss {record["loss"]} is not the T={temperature} loss {expected}')


def guard_manifest_binding_is_php02():
    from vkgr import retrieval_feat02_run as fr, retrieval_php02_run as p2
    with tempfile.TemporaryDirectory() as tmp:
        value = p2.run_session(Path(tmp) / 'run', fr.synthetic_data(), small02('T005', max_epochs=2))
    ensure(value['binding']['task'] == 'P-HP-02' and value['binding']['knob'] == 'T005', f'binding {value["binding"]["task"]}/{value["binding"]["knob"]}')
    ensure((value['schema'], value['kind']) == ('p-php02-train-manifest-v1', 'p_php02_train'), 'manifest schema/kind')


def guard_cross_schema_rejected():
    from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php, retrieval_php02_run as p2
    good = small02('T005')
    p2.validate_config(good, synthetic=True)
    for override in ({'schema': php.SCHEMA}, {'task_id': php.TASK}, {'schema': fr.SCHEMA}, {'task_id': fr.TASK}, {'schema': php.SCHEMA, 'task_id': php.TASK}):
        try:
            p2.validate_config({**good, **override}, synthetic=True)
        except ValueError:
            continue
        raise AssertionError(f'php02 accepted a foreign schema/task {override}')


def guard_registry_is_exactly_two_real_temperatures():
    from vkgr.retrieval_php02_run import validate_config
    for knob, temperature in (('T006', 0.06), ('T003', 0.03), ('T008', 0.08), ('T020', 0.2)):
        try:
            validate_config(config02(knob, temperature), synthetic=True)
        except ValueError:
            continue
        raise AssertionError(f'unregistered knob {knob} accepted')
    for temperature in (0.06, 0.03, 0.08, 0.2):
        for knob in REAL:
            try:
                validate_config(config02(knob, temperature), synthetic=True)
            except ValueError:
                continue
            raise AssertionError(f'{knob} accepted temperature {temperature}')


# ---------------------------------------------------------------- tests

class EquivalenceTests(unittest.TestCase):
    def _compare(self, tmp, a, b, drop):
        """a, b: manifests (a = reference, b = php02). Everything the card lists must be equal."""
        from vkgr import retrieval_feat02_run as fr
        self.assertGreater(len(a['records']), 8)
        for key in ('loss', 'gradient_norm', 'drawn', 'skipped', 'indices_sha256', 'negatives_sha256'):
            self.assertEqual([r[key] for r in a['records']], [r[key] for r in b['records']], key)
        self.assertEqual(a['selection'], b['selection'])
        self.assertEqual([e['result'] for e in fr.full_evaluations(tmp / 'ref', a)], [e['result'] for e in fr.full_evaluations(tmp / 'p2', b)])
        self.assertEqual([e['diagnostics'] for e in a['evaluations']], [e['diagnostics'] for e in b['evaluations']])
        self.assertEqual([e['predictions_sha256'] for e in a['evaluations']], [e['predictions_sha256'] for e in b['evaluations']])
        self.assertEqual(a['data_diagnostics'], b['data_diagnostics'])
        for name in ('best.pt', 'last.pt'):
            sa = torch.load(tmp / 'ref' / name, map_location='cpu', weights_only=True)['model']
            sb = torch.load(tmp / 'p2' / name, map_location='cpu', weights_only=True)['model']
            self.assertEqual(list(sa), list(sb))
            for key in sa:
                self.assertEqual(sa[key].numpy().tobytes(), sb[key].numpy().tobytes(), (name, key))
        self.assertEqual({k: v for k, v in a['binding'].items() if k not in drop}, {k: v for k, v in b['binding'].items() if k not in drop})

    def test_e1_t010_equals_php01_h1_epoch_for_epoch(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php, retrieval_php02_run as p2
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); data = fr.synthetic_data()
            a = php.run_session(tmp / 'ref', data, h1_small()); b = p2.run_session(tmp / 'p2', data, small02('T010'))
            self._compare(tmp, a, b, ('task', 'knob', 'config_sha256', 'source'))
            self.assertEqual((a['binding']['task'], a['binding']['knob']), ('P-HP-01', 'H1'))
            self.assertEqual((b['binding']['task'], b['binding']['knob'], b['binding']['dim']), ('P-HP-02', 'T010', 64))
            self.assertEqual(set(a['binding']), set(b['binding']))

    def test_e2_t015_equals_feat02_f4_epoch_for_epoch(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php02_run as p2
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); data = fr.synthetic_data()
            a = fr.run_session(tmp / 'ref', data, f4_small()); b = p2.run_session(tmp / 'p2', data, small02('T015'))
            self._compare(tmp, a, b, ('task', 'knob', 'dim', 'config_sha256', 'source'))
            self.assertEqual(set(b['binding']) - set(a['binding']), {'knob', 'dim'})
            self.assertEqual((b['binding']['task'], b['binding']['knob'], b['binding']['dim']), ('P-HP-02', 'T015', 64))

    def test_reuses_the_feat02_batch_tensors_and_evaluation(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php, retrieval_php02_run as p2
        self.assertIs(p2._tensors2, fr._tensors2)
        self.assertIs(p2.evaluate_model, fr.evaluate_model)
        self.assertIs(p2.full_evaluations, fr.full_evaluations)
        self.assertIs(p2._model, php._model)
        self.assertIs(p2.make_optimizer, php.make_optimizer)
        with mock.patch.object(fr, 'train_step', return_value='sentinel') as step:
            self.assertEqual(p2._step(*range(10)), 'sentinel')
        step.assert_called_once_with(*range(10))

    def test_e3_committed_configs_differ_from_h1_in_four_fields(self):
        from vkgr.retrieval_php02_run import validate_config
        for seed in SEEDS:
            h1_path = H1_DIR / f'p-php01-h1-t010-s{seed}.json'
            import hashlib
            self.assertEqual(hashlib.sha256(h1_path.read_bytes()).hexdigest(), H1_SHA[seed], seed)
            h1 = json.loads(h1_path.read_text())
            for knob, temperature in REAL.items():
                path = CONFIG_DIR / f'{RUN_IDS[(knob, seed)]}.json'
                value = json.loads(path.read_text())
                validate_config(value)                                             # real (non-synthetic) validation
                self.assertEqual(path.read_text(), json.dumps(value, indent=1, sort_keys=True) + '\n', path.name)
                self.assertEqual(set(value), set(h1))
                changed = {k for k in h1 if h1[k] != value[k]}
                self.assertEqual(changed, {'schema', 'task_id', 'knob', 'temperature'}, path.name)
                self.assertEqual((value['schema'], value['task_id'], value['knob'], value['temperature'], value['seed']),
                                 ('p-php02-run-v1', 'P-HP-02', knob, temperature, seed))
                self.assertEqual((value['dim'], value['learning_rate']), (64, 0.001))
        files = sorted(p.name for p in CONFIG_DIR.glob('*.json'))
        self.assertEqual(files, sorted(f'{r}.json' for r in RUN_IDS.values()))
        self.assertEqual(len(files), 6)

    def test_e3_t010_synthetic_config_projects_onto_the_h1_file(self):
        h1 = json.loads((H1_DIR / 'p-php01-h1-t010-s42.json').read_text())
        value = config02('T010')
        self.assertEqual({k: v for k, v in value.items() if k not in ('schema', 'task_id', 'knob')},
                         {k: v for k, v in h1.items() if k not in ('schema', 'task_id', 'knob')})

    def test_e4_first_step_loss_uses_config_temperature(self):
        guard_first_step_loss_uses_config_temperature_p02()

    def test_e5_t005_and_t007_train_and_verify(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php02_run as p2
        data = fr.synthetic_data()
        for knob in REAL:
            with self.subTest(knob=knob), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp) / 'run'; config = small02(knob, max_epochs=6)
                value = p2.run_session(run, data, config)
                self.assertEqual((value['schema'], value['kind'], value['status']), ('p-php02-train-manifest-v1', 'p_php02_train', 'complete'))
                self.assertEqual(value['config']['temperature'], REAL[knob])
                self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
                self.assertEqual(torch.load(run / 'best.pt', map_location='cpu', weights_only=True)['schema'], 'p-php02-checkpoint-v1')
                self.assertTrue(all(type(r['loss']) is float and math.isfinite(r['loss']) for r in value['records']))
                self.assertEqual(p2.verify_run(run / 'manifest.json', value['manifest_sha256'], data, config)['status'], 'verified')
                other = 'T007' if knob == 'T005' else 'T005'
                with self.assertRaises(ValueError):
                    p2.verify_run(run / 'manifest.json', value['manifest_sha256'], data, small02(other, max_epochs=6))
                side = {**data, 'rows': [{**r, 'side': [[m ^ 1, t, d] for m, t, d in r['side']]} for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    p2.verify_run(run / 'manifest.json', value['manifest_sha256'], side, config)

    def test_binding_and_schema_are_php02(self):
        guard_manifest_binding_is_php02()


class RejectionTests(unittest.TestCase):
    def test_registry_values(self):
        from vkgr import retrieval_php02_run as p2
        self.assertEqual((p2.SCHEMA, p2.TASK, p2.KIND), ('p-php02-run-v1', 'P-HP-02', 'p_php02_train'))
        self.assertEqual((p2.CHECKPOINT_SCHEMA, p2.MANIFEST_SCHEMA), ('p-php02-checkpoint-v1', 'p-php02-train-manifest-v1'))
        self.assertEqual(p2.TEMPERATURE_KNOBS, {'T005': 0.05, 'T007': 0.07})
        self.assertEqual(p2.EQUIVALENCE_ONLY, {'T010': 0.1, 'T015': 0.15})

    def test_real_knobs_validate(self):
        from vkgr.retrieval_php02_run import validate_config
        for knob in REAL:
            validate_config(config02(knob)); validate_config(small02(knob), synthetic=True)
        for knob in ('T010', 'T015'):
            validate_config(small02(knob), synthetic=True)

    def test_other_temperatures_are_rejected_not_rounded(self):
        from vkgr.retrieval_php02_run import validate_config
        bad = [0.06, 0.03, 0.08, 0.2, 0.0, -0.05, float('nan'), float('inf'), -float('inf'), 0.0500001, 0.05000000001, 0.15, 0.1, 0.25]
        for knob in REAL:
            for temperature in bad:
                if temperature == REAL[knob]:
                    continue
                for synthetic in (False, True):
                    with self.assertRaises(ValueError, msg=(knob, temperature, synthetic)):
                        validate_config(config02(knob, temperature), synthetic)
        for knob in ('T010', 'T015'):                                # equivalence knobs keep their own exact value too
            for temperature in (0.06, 0.05, 0.07, 0.1 if knob == 'T015' else 0.15):
                with self.assertRaises(ValueError, msg=(knob, temperature)):
                    validate_config(config02(knob, temperature), True)

    def test_temperature_type_is_float(self):
        from vkgr.retrieval_php02_run import validate_config
        for knob in REAL:
            for temperature in ('0.05', '0.07', 0, 1, True, False, None, [0.05]):
                if type(temperature) is float and temperature == REAL[knob]:
                    continue
                with self.assertRaises(ValueError, msg=(knob, temperature)):
                    validate_config({**config02(knob), 'temperature': temperature})

    def test_knob_temperature_mismatch_and_unknown_knobs(self):
        from vkgr.retrieval_php02_run import validate_config
        with self.assertRaises(ValueError):
            validate_config(config02('T005', 0.07))
        with self.assertRaises(ValueError):
            validate_config(config02('T007', 0.05))
        for knob in ('H1', 'H7', 'T006', 'anchor', 'T0.05', '', 'T005 ', None, 5, ['T005']):
            with self.assertRaises(ValueError, msg=knob):
                validate_config({**config02('T005'), 'knob': knob}, synthetic=True)

    def test_equivalence_knobs_are_not_real_runs(self):
        guard_equivalence_knobs_are_synthetic_only()

    def test_only_the_temperature_may_move(self):
        from vkgr.retrieval_php02_run import validate_config
        for knob in REAL:
            for override in ({'dim': 32}, {'dim': 128}, {'dim': 64.0}, {'dim': True}, {'learning_rate': 0.002}, {'learning_rate': 0.0005},
                             {'learning_rate': 1}, {'seed': 7}, {'logq': False}, {'extra_uniform': 128}, {'history_len': 20}, {'test_gate': 'open'},
                             {'feature': 'f3'}, {'feature': 'f3_pos_only'}, {'init': 'legacy'}, {'max_epochs': 30}, {'batch_size': 256}, {'weight_decay': 0.001},
                             {'side_version': 'v0'}, {'architecture': 'sasrec'}, {'precision': 'fp16'}, {'gradient_clip': 2.0}, {'logq_alpha': 0.5}):
                with self.assertRaises(ValueError, msg=(knob, override)):
                    validate_config({**config02(knob), **override})

    def test_extra_or_missing_fields(self):
        from vkgr.retrieval_php02_run import validate_config
        base = config02('T005')
        with self.assertRaises(ValueError):
            validate_config({**base, 'unknown': 1})
        for key in ('dim', 'knob', 'temperature', 'seed', 'schema'):
            with self.assertRaises(ValueError, msg=key):
                validate_config({k: v for k, v in base.items() if k != key})
        with self.assertRaises(ValueError):
            validate_config([base])

    def test_cross_schema_rejected_both_ways(self):
        from vkgr import retrieval_php01_run as php, retrieval_php02_run as p2
        guard_cross_schema_rejected()
        h1 = json.loads((H1_DIR / 'p-php01-h1-t010-s42.json').read_text())
        with self.assertRaises(ValueError):
            p2.validate_config(h1)                                    # php01 H1 into php02
        with self.assertRaises(ValueError):
            p2.validate_config(h1, synthetic=True)
        for knob in REAL:                                             # php02 config into php01: old registry not widened
            for synthetic in (False, True):
                with self.assertRaises(ValueError, msg=(knob, synthetic)):
                    php.validate_config(config02(knob), synthetic)
                with self.assertRaises(ValueError, msg=(knob, synthetic)):
                    php.validate_config({**config02(knob), 'schema': php.SCHEMA, 'task_id': php.TASK}, synthetic)

    def test_registry_is_exactly_two_real_temperatures(self):
        guard_registry_is_exactly_two_real_temperatures()

    def test_old_php01_registry_object_is_unchanged(self):
        from vkgr import retrieval_php01_run as php
        self.assertEqual(sorted(php.KNOBS), ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'anchor'])
        self.assertEqual(php.KNOBS['H1']['temperature'], 0.1)


class LeakageTests(unittest.TestCase):
    def test_batch_tensors_ignore_target_fields_on_the_php02_path(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php02_run as p2
        data = fr.synthetic_data(); rows, queries = fr._view(data, 50)
        model = p2._model(data, small02('T005'), 'cpu'); model.eval()
        for batch in (rows[:24], queries[:24]):
            width = max(len(r['history']) for r in batch)
            base = p2._tensors2(batch, data['idmap'], width, 'f3_f4_recency')
            flipped = p2._tensors2([flip_non_history_fields(r) for r in batch], data['idmap'], width, 'f3_f4_recency')
            self.assertTrue(torch.equal(base[0], flipped[0]) and torch.equal(base[1], flipped[1]))
            self.assertTrue(torch.equal(model.users(*base), model.users(*flipped)))


class CLITests(unittest.TestCase):
    SPEC = {'schema': 'p-feat02-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
            'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64, 'items_metadata_sha256': 'c' * 64,
            'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}

    def test_cli_synthetic_train_and_real_path_wiring(self):
        from vkgr import retrieval_php02_run as p2
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'config.json').write_text(json.dumps(small02('T007')))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(p2.main(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run')]), 0)
            self.assertEqual(json.loads(out.getvalue())['status'], 'complete')
            (tmp / 'inputs.json').write_text(json.dumps(self.SPEC)); (tmp / 'real.json').write_text(json.dumps(config02('T005')))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            with mock.patch('torch.cuda.is_available', return_value=False), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(p2.main(['train', *common, '--output', str(tmp / 'x')]), 2)
            self.assertIn('P-HP-02 gate failed', err.getvalue()); self.assertFalse((tmp / 'x').exists())
            fake_data = {'rows': [], 'queries': [], 'catalog': []}; seen = {}
            with mock.patch('torch.cuda.is_available', return_value=True), \
                    mock.patch.object(p2.b1, 'resource_guard', lambda *a, **k: (lambda: None)), \
                    mock.patch.object(p2.b1, 'real_inputs', lambda s: (fake_data, None, {'prepared_sha': 'x'})), \
                    mock.patch.object(p2.fr, 'apply_feat02', lambda d, s, n: (d, {'history_manifest_sha256': 'h'})), \
                    mock.patch.object(p2, 'run_session', lambda *a, **k: seen.update(train=a) or {'status': 'complete', 'manifest_sha256': 'm', 'selection': {'best_epoch': 3}}), \
                    mock.patch.object(p2, 'verify_run', lambda *a, **k: seen.update(verify=a) or {'status': 'verified'}):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(p2.main(['train', *common, '--output', str(tmp / 'real-run')]), 0)
                self.assertEqual(json.loads(out.getvalue())['best_epoch'], 3)
                self.assertEqual((seen['train'][2]['knob'], seen['train'][3], seen['train'][5]['inputs_sha256']), ('T005', 'cuda', file_sha256(tmp / 'inputs.json')))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(p2.main(['verify', *common, '--manifest', str(tmp / 'real-run' / 'manifest.json'), '--manifest-sha', 'a' * 64]), 0)
                self.assertEqual(seen['verify'][4], 'cuda')
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(p2.main(['train', *common[:2], '--inputs-sha', '0' * 64, *common[4:], '--output', str(tmp / 'x')]), 2)
            self.assertIn('external hash mismatch', err.getvalue())

    def test_cli_refuses_an_unregistered_temperature_on_the_real_path(self):
        from vkgr import retrieval_php02_run as p2
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'inputs.json').write_text(json.dumps(CLITests.SPEC)); (tmp / 'real.json').write_text(json.dumps(config02('T005', 0.06)))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(p2.main(['train', *common, '--output', str(tmp / 'x')]), 2)
            self.assertIn('P-HP-02 gate failed', err.getvalue()); self.assertFalse((tmp / 'x').exists())

    def test_cli_has_only_the_three_subcommands(self):
        from vkgr import retrieval_php02_run as p2
        for command in ('prepare-ctx', 'prepare-attr', 'verify-ctx'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                p2.main([command, '--inputs', 'x', '--inputs-sha', 'y'])


if __name__ == '__main__':
    unittest.main()
