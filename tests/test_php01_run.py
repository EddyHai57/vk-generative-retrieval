"""P-HP-01 (P line) runner tests: anchor knob == FEAT-02 F4 runner, each knob trains + verifies, the knob values reach training.

``guard_*`` functions are plain assertion functions; tests/test_php01_mutations.py re-runs them under wrong implementations.
"""
import contextlib
import io
import itertools
import json
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

ROOT = Path(__file__).resolve().parent.parent
F4_CONFIG = ROOT / 'configs' / 'feat02' / 'p-feat02-f4-recency-s42.json'
# independent literals of the pre-registered knob values (card 3A): (temperature, dim, learning_rate)
ANCHOR = (0.15, 64, 0.001)
KNOB_VALUES = {'H1': (0.1, 64, 0.001), 'H2': (0.25, 64, 0.001), 'H3': (0.15, 32, 0.001),
               'H4': (0.15, 128, 0.001), 'H5': (0.15, 64, 0.002), 'H6': (0.15, 64, 0.0005)}
KNOB_FIELD = {'H1': 'temperature', 'H2': 'temperature', 'H3': 'dim', 'H4': 'dim', 'H5': 'learning_rate', 'H6': 'learning_rate'}
CONFIG_STEMS = {'p-php01-h1-t010-s42': 'H1', 'p-php01-h2-t025-s42': 'H2', 'p-php01-h3-d32-s42': 'H3',
                'p-php01-h4-d128-s42': 'H4', 'p-php01-h5-lr002-s42': 'H5', 'p-php01-h6-lr0005-s42': 'H6'}
H7_CONFIG_STEM = 'p-php01-h7-combo-s42'          # written by the select script, committed to Git before it is enqueued


def knob_config(knob, **overrides):
    from vkgr.retrieval_php01_run import default_config
    t, d, lr = ANCHOR if knob == 'anchor' else KNOB_VALUES[knob]
    value = {**default_config(), 'knob': knob, 'temperature': t, 'dim': d, 'learning_rate': lr}
    value.update(overrides)
    return value


def small(knob, **overrides):
    return knob_config(knob, batch_size=16, max_epochs=8, min_epochs=2, **overrides)


def f4_small():
    from vkgr.retrieval_feat02_run import default_config
    return {**default_config(), 'feature': 'f3_f4_recency', 'logq': True, 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}


# ---------------------------------------------------------------- guards (reused by the mutation tests)

def guard_runner_builds_config_dim():
    from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php
    from vkgr.retrieval_php01 import expected_param_counts_d
    data = fr.synthetic_data(); rows, queries = fr._view(data, 50)
    for knob, dim in (('H3', 32), ('H4', 128)):
        config = small(knob); model = php._model(data, config, 'cpu')
        for name in ('embedding', 'timespent_bucket', 'ratio_bucket', 'position_table', 'weekgap_table'):
            ensure(model.get_submodule(name).weight.shape[1] == dim, f'{knob}: {name} is not {dim} wide')
        binding = php.binding_for3(data, config, rows, queries, model, 'cpu', None)
        ensure(binding['dim'] == dim and binding['param_report'] == expected_param_counts_d(dim), f'{knob}: binding param report')


def _first_batch(config):
    from vkgr import retrieval_feat02_run as fr
    from vkgr.retrieval_negatives import target_frequency
    from vkgr.retrieval_training import EpochCursor
    data = fr.synthetic_data(); rows, _ = fr._view(data, 50)
    indices = EpochCursor(len(rows), config['seed']).next(config['batch_size'])
    return data, rows, [rows[i] for i in indices], target_frequency(rows)


def _independent_loss(config, temperature):
    from vkgr import retrieval_b1_run as b1, retrieval_feat02_run as fr
    from vkgr.retrieval_negatives import masked_ce_logq
    from vkgr.retrieval_php01 import make_php01_model
    from vkgr.training import seed_all
    data, rows, batch, frequency = _first_batch(config)
    seed_all(config['seed']); model = make_php01_model(data['content'], data['supported_indices'], 50, config['dim']); model.train()
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, random.Random(0), None)
    histories, side = fr._tensors2(batch, data['idmap'], max(len(r['history']) for r in batch), 'f3_f4_recency')
    log_q = b1._log_q(batch, negatives, pool_size, config, frequency)
    loss, _ = masked_ce_logq(model.users(histories, side), model.items(columns), labels, mask, temperature, log_q)
    return float(loss)


def guard_first_step_loss_uses_config_temperature():
    from vkgr import retrieval_php01_run as php
    from vkgr.retrieval_training import SamplerStats
    for knob, temperature in (('H1', 0.1), ('H2', 0.25)):
        config = small(knob); data, rows, batch, frequency = _first_batch(config)
        model = php._model(data, config, 'cpu'); optimizer = php.make_optimizer(model, config)
        record = php._step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)
        expected, anchor = _independent_loss(config, temperature), _independent_loss(config, 0.15)
        ensure(abs(expected - anchor) > 1e-4, 'fixture cannot tell the temperatures apart')
        ensure(abs(record['loss'] - expected) < 1e-6, f'{knob}: first-step loss {record["loss"]} is not the T={temperature} loss {expected}')


def guard_optimizer_uses_config_lr():
    from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php
    for knob, lr in (('H5', 0.002), ('H6', 0.0005)):
        with tempfile.TemporaryDirectory() as tmp:
            php.run_session(Path(tmp) / 'run', fr.synthetic_data(), knob_config(knob, batch_size=16, max_epochs=1, min_epochs=1))
            group = torch.load(Path(tmp) / 'run' / 'last.pt', map_location='cpu', weights_only=True)['optimizer']['param_groups'][0]
            ensure(group['lr'] == lr and group['weight_decay'] == 1e-4, f'{knob}: optimizer lr {group["lr"]} / wd {group["weight_decay"]}')


# ---------------------------------------------------------------- tests

class RunnerEquivalenceTests(unittest.TestCase):
    def test_anchor_knob_runner_equals_feat02_f4_runner(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); data = fr.synthetic_data()
            a = fr.run_session(tmp / 'f4', data, f4_small()); b = php.run_session(tmp / 'php', data, small('anchor'))
            self.assertGreater(len(a['records']), 8)
            for key in ('loss', 'gradient_norm', 'drawn', 'skipped', 'indices_sha256', 'negatives_sha256'):
                self.assertEqual([r[key] for r in a['records']], [r[key] for r in b['records']], key)
            self.assertEqual(a['selection'], b['selection'])
            self.assertEqual([e['result'] for e in fr.full_evaluations(tmp / 'f4', a)], [e['result'] for e in php.full_evaluations(tmp / 'php', b)])
            self.assertEqual([e['diagnostics'] for e in a['evaluations']], [e['diagnostics'] for e in b['evaluations']])
            self.assertEqual(a['data_diagnostics'], b['data_diagnostics'])
            for name in ('best.pt', 'last.pt'):
                sa = torch.load(tmp / 'f4' / name, map_location='cpu', weights_only=True)['model']
                sb = torch.load(tmp / 'php' / name, map_location='cpu', weights_only=True)['model']
                self.assertEqual(list(sa), list(sb))
                for key in sa:
                    self.assertEqual(sa[key].numpy().tobytes(), sb[key].numpy().tobytes(), (name, key))
            drop = ('task', 'knob', 'dim', 'config_sha256', 'source')
            self.assertEqual({k: v for k, v in a['binding'].items() if k not in drop}, {k: v for k, v in b['binding'].items() if k not in drop})
            self.assertEqual(set(b['binding']) - set(a['binding']), {'knob', 'dim'})
            self.assertEqual((b['binding']['task'], b['binding']['knob'], b['binding']['dim']), ('P-HP-01', 'anchor', 64))

    def test_runner_reuses_the_feat02_batch_tensors_and_train_step(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php
        self.assertIs(php._tensors2, fr._tensors2)
        self.assertIs(php.evaluate_model, fr.evaluate_model)
        self.assertIs(php.full_evaluations, fr.full_evaluations)
        with mock.patch.object(fr, 'train_step', return_value='sentinel') as step:
            self.assertEqual(php._step(*range(10)), 'sentinel')
        step.assert_called_once_with(*range(10))


class KnobRunTests(unittest.TestCase):
    def test_each_knob_trains_and_verifies(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php
        from vkgr.retrieval_feat02 import bucket_spec_sha256
        from vkgr.retrieval_php01 import expected_param_counts_d
        data = fr.synthetic_data()
        for knob in KNOB_VALUES:
            with self.subTest(knob=knob), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp) / 'run'; config = knob_config(knob, batch_size=16, max_epochs=6, min_epochs=2)
                value = php.run_session(run, data, config)
                self.assertEqual((value['schema'], value['kind'], value['status']), ('p-php01-train-manifest-v1', 'p_php01_train', 'complete'))
                self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
                binding = value['binding']; dim = KNOB_VALUES[knob][1]
                self.assertEqual((binding['task'], binding['knob'], binding['dim'], binding['feature']), ('P-HP-01', knob, dim, 'f3_f4_recency'))
                self.assertEqual(binding['bucket_spec_sha256'], bucket_spec_sha256())
                self.assertIn('weeks_sha256', binding)
                self.assertEqual(binding['param_report'], expected_param_counts_d(dim))
                state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
                self.assertEqual(state['schema'], 'p-php01-checkpoint-v1')
                self.assertEqual(state['model']['embedding.weight'].shape[1], dim)
                self.assertEqual(php.verify_run(run / 'manifest.json', value['manifest_sha256'], data, config)['status'], 'verified')
                other = 'H1' if knob != 'H1' else 'H2'
                with self.assertRaises(ValueError):
                    php.verify_run(run / 'manifest.json', value['manifest_sha256'], data, knob_config(other, batch_size=16, max_epochs=6, min_epochs=2))
                side = {**data, 'rows': [{**r, 'side': [[m ^ 1, t, d] for m, t, d in r['side']]} for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    php.verify_run(run / 'manifest.json', value['manifest_sha256'], side, config)
                weeks = {**data, 'rows': [{**r, 'history_positions': [[(w + 1) % 25 if k == len(r['history_positions']) - 1 else w, o]
                                                                      for k, (w, o) in enumerate(r['history_positions'])]} for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    php.verify_run(run / 'manifest.json', value['manifest_sha256'], weeks, config)

    def test_runner_builds_config_dim(self):
        guard_runner_builds_config_dim()

    def test_first_step_loss_uses_config_temperature(self):
        guard_first_step_loss_uses_config_temperature()

    def test_optimizer_uses_config_lr(self):
        guard_optimizer_uses_config_lr()


class ConfigRegistryTests(unittest.TestCase):
    def test_registry_values(self):
        from vkgr import retrieval_php01_run as php
        self.assertEqual((php.SCHEMA, php.TASK), ('p-php01-run-v1', 'P-HP-01'))
        self.assertEqual(tuple(php.ANCHOR_VALUES[k] for k in ('temperature', 'dim', 'learning_rate')), ANCHOR)
        for knob, values in KNOB_VALUES.items():
            self.assertEqual(tuple(php.knob_values(knob)[k] for k in ('temperature', 'dim', 'learning_rate')), values, knob)
            self.assertEqual(php.KNOB_FIELD[knob], KNOB_FIELD[knob])
        self.assertEqual(php.H7_LEVELS, {'temperature': (0.1, 0.25), 'dim': (32, 128), 'learning_rate': (0.002, 0.0005)})

    def test_every_knob_validates(self):
        from vkgr.retrieval_php01_run import validate_config
        for knob in KNOB_VALUES:
            validate_config(knob_config(knob)); validate_config(small(knob), synthetic=True)
        validate_config(small('anchor'), synthetic=True)
        with self.assertRaisesRegex(ValueError, 'anchor'):
            validate_config(knob_config('anchor'))           # the anchor run exists (FEAT-02 F4 s42): never a real P-HP-01 run

    def test_invalid_configs_rejected(self):
        from vkgr.retrieval_php01_run import validate_config
        base = knob_config('H1')
        bad = [{'temperature': 0.25}, {'temperature': 0.15}, {'knob': 'H8'}, {'knob': 'H5'}, {'feature': 'f3'}, {'history_len': 20},
               {'init': 'legacy'}, {'max_epochs': 30}, {'test_gate': 'open'}, {'seed': 7}, {'schema': 'p-feat02-run-v1'}, {'task_id': 'FEAT-02'},
               {'logq': False}, {'extra_uniform': 128}, {'batch_size': 256}, {'weight_decay': 0.001}, {'side_version': 'v0'}, {'architecture': 'sasrec'}]
        for override in bad:
            with self.assertRaises(ValueError, msg=override):
                validate_config({**base, **override})
        for knob, override in (('H3', {'dim': 64}), ('H3', {'dim': 32.0}), ('H4', {'dim': True}), ('H1', {'dim': 32}),
                               ('H5', {'learning_rate': 0.001}), ('H6', {'learning_rate': 0.002}), ('H5', {'temperature': 0.1})):
            with self.assertRaises(ValueError, msg=(knob, override)):
                validate_config({**knob_config(knob), **override})
        with self.assertRaises(ValueError):
            validate_config({**base, 'unknown': 1})
        with self.assertRaises(ValueError):
            validate_config({k: v for k, v in base.items() if k != 'dim'})

    def test_h7_combinations(self):
        from vkgr.retrieval_php01_run import validate_config
        levels = {'temperature': (0.15, 0.1, 0.25), 'dim': (64, 32, 128), 'learning_rate': (0.001, 0.002, 0.0005)}
        accepted = 0
        for t, d, lr in itertools.product(*levels.values()):
            config = knob_config('anchor', knob='H7', temperature=t, dim=d, learning_rate=lr)
            changed = sum(1 for value, anchor in zip((t, d, lr), ANCHOR) if value != anchor)
            if changed >= 2:
                validate_config(config); accepted += 1
            else:
                with self.assertRaises(ValueError, msg=(t, d, lr)):
                    validate_config(config)
        self.assertEqual(accepted, 20)
        for bad in ({'temperature': 0.2, 'dim': 32}, {'dim': 96, 'learning_rate': 0.002}):
            with self.assertRaises(ValueError):
                validate_config(knob_config('anchor', knob='H7', **bad))

    def test_anchor_knob_projects_to_the_f4_config_file(self):
        from vkgr.retrieval_feat02_run import SCHEMA, TASK
        f4 = json.loads(F4_CONFIG.read_text())
        projected = {k: v for k, v in knob_config('anchor').items() if k not in ('dim', 'knob')}
        self.assertEqual({**projected, 'schema': SCHEMA, 'task_id': TASK}, f4)

    def test_committed_knob_configs(self):
        from vkgr.retrieval_php01_run import validate_config
        f4 = json.loads(F4_CONFIG.read_text())
        files = {p.stem: p for p in (ROOT / 'configs' / 'php01').glob('p-php01-h*-s42.json')}
        self.assertEqual(set(files) - {H7_CONFIG_STEM}, set(CONFIG_STEMS))
        for stem, knob in CONFIG_STEMS.items():
            path = files[stem]; value = json.loads(path.read_text())
            validate_config(value)
            self.assertEqual(path.read_text(), json.dumps(value, indent=1, sort_keys=True) + '\n', stem)
            self.assertEqual(value['knob'], knob, stem)
            self.assertEqual((value['temperature'], value['dim'], value['learning_rate']), KNOB_VALUES[knob], stem)
            self.assertEqual(set(value), set(f4) | {'dim', 'knob'}, stem)
            changed = {k for k in f4 if f4[k] != value[k]}
            self.assertEqual(changed, {'schema', 'task_id'} | ({KNOB_FIELD[knob]} if KNOB_FIELD[knob] != 'dim' else set()), stem)
            self.assertEqual(value['dim'] != 64, KNOB_FIELD[knob] == 'dim', stem)
        if H7_CONFIG_STEM in files:                          # present only after the combo was triggered and committed
            value = json.loads(files[H7_CONFIG_STEM].read_text())
            validate_config(value); self.assertEqual(value['knob'], 'H7')


class LeakageTests(unittest.TestCase):
    def test_batch_tensors_ignore_target_fields_on_a_dim32_model(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php
        data = fr.synthetic_data(); rows, queries = fr._view(data, 50)
        model = php._model(data, small('H3'), 'cpu'); model.eval()
        for batch in (rows[:24], queries[:24]):
            width = max(len(r['history']) for r in batch)
            base = php._tensors2(batch, data['idmap'], width, 'f3_f4_recency')
            flipped = php._tensors2([flip_non_history_fields(r) for r in batch], data['idmap'], width, 'f3_f4_recency')
            self.assertTrue(torch.equal(base[0], flipped[0]) and torch.equal(base[1], flipped[1]))
            self.assertTrue(torch.equal(model.users(*base), model.users(*flipped)))


class CLITests(unittest.TestCase):
    SPEC = {'schema': 'p-feat02-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
            'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64, 'items_metadata_sha256': 'c' * 64,
            'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}

    def test_cli_synthetic_train_and_real_path_wiring(self):
        from vkgr import retrieval_php01_run as php
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'config.json').write_text(json.dumps(small('H3')))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(php.main(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run')]), 0)
            self.assertEqual(json.loads(out.getvalue())['status'], 'complete')
            (tmp / 'inputs.json').write_text(json.dumps(self.SPEC)); (tmp / 'real.json').write_text(json.dumps(knob_config('H4')))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            with mock.patch('torch.cuda.is_available', return_value=False), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(php.main(['train', *common, '--output', str(tmp / 'x')]), 2)
            self.assertIn('P-HP-01 gate failed', err.getvalue()); self.assertFalse((tmp / 'x').exists())
            fake_data = {'rows': [], 'queries': [], 'catalog': []}; seen = {}
            with mock.patch('torch.cuda.is_available', return_value=True), \
                    mock.patch.object(php.b1, 'resource_guard', lambda *a, **k: (lambda: None)), \
                    mock.patch.object(php.b1, 'real_inputs', lambda s: (fake_data, None, {'prepared_sha': 'x'})), \
                    mock.patch.object(php.fr, 'apply_feat02', lambda d, s, n: (d, {'history_manifest_sha256': 'h'})), \
                    mock.patch.object(php, 'run_session', lambda *a, **k: seen.update(train=a) or {'status': 'complete', 'manifest_sha256': 'm', 'selection': {'best_epoch': 3}}), \
                    mock.patch.object(php, 'verify_run', lambda *a, **k: seen.update(verify=a) or {'status': 'verified'}):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(php.main(['train', *common, '--output', str(tmp / 'real-run')]), 0)
                self.assertEqual(json.loads(out.getvalue())['best_epoch'], 3)
                self.assertEqual((seen['train'][2]['knob'], seen['train'][3], seen['train'][5]['inputs_sha256']), ('H4', 'cuda', file_sha256(tmp / 'inputs.json')))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(php.main(['verify', *common, '--manifest', str(tmp / 'real-run' / 'manifest.json'), '--manifest-sha', 'a' * 64]), 0)
                self.assertEqual(seen['verify'][4], 'cuda')
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(php.main(['train', *common[:2], '--inputs-sha', '0' * 64, *common[4:], '--output', str(tmp / 'x')]), 2)
            self.assertIn('external hash mismatch', err.getvalue())

    def test_cli_has_only_train_verify_subcommands(self):
        from vkgr import retrieval_php01_run as php
        for command in ('prepare-ctx', 'prepare-attr', 'verify-ctx'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                php.main([command, '--inputs', 'x', '--inputs-sha', 'y'])


if __name__ == '__main__':
    unittest.main()
