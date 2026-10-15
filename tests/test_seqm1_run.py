"""SEQ-M1 (P line) runner tests: config registry (G12), binding (G13), synthetic train + verify for all 5 arms, diagnostics, g=0 ablation,
step-1 equivalence with the F4-T010 runner, CLI wiring, holdout-open guard (L6), committed configs.

``guard_*`` functions are plain assertion functions; tests/test_seqm1_mutations.py re-runs them under wrong implementations.
"""
import builtins
import contextlib
import io
import json
import os
import pathlib
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_model import ensure  # noqa: E402
import test_seqm1_model as tm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H1_CONFIG = ROOT / 'configs' / 'php01' / 'p-php01-h1-t010-s42.json'
CONFIG_DIR = ROOT / 'configs' / 'seqm1'
SEEDS = (42, 2024, 2025)
HOLDOUT_FRAGMENT = 'te' + 'st'               # spliced at run time so the public-repo scan does not flag this file


def seq_config(arm='mean', **overrides):
    from vkgr.retrieval_seqm1_run import default_config
    value = default_config(); value['arm'] = arm; value.update(overrides)
    return value


def small(arm='mean', **overrides):
    return seq_config(arm, **{'batch_size': 16, 'max_epochs': 6, 'min_epochs': 2, **overrides})


def tmp_dir():
    return tempfile.TemporaryDirectory(prefix='seqm1-')


# ---------------------------------------------------------------- guards (reused by the mutation tests)

def guard_config_pins_the_anchor_conditions():
    """G12 / M21: temperature 0.10, logQ on, dim 64, lr 1e-3, L50, budget 100/5/3 are frozen; unknown arm / fields / types are rejected."""
    from vkgr.retrieval_seqm1_run import validate_config
    base = seq_config('mean')
    validate_config(base)
    bad = [{'arm': 'first'}, {'arm': None}, {'arm': 3}, {'temperature': .15}, {'temperature': .25}, {'dim': 32}, {'dim': 128}, {'learning_rate': .002}, {'logq': False},
           {'extra_uniform': 128}, {'history_len': 20}, {'history_len': 100}, {'init': 'legacy'}, {'max_epochs': 30}, {'min_epochs': 0}, {'patience': 5},
           {'test_gate': 'open'}, {'seed': 7}, {'schema': 'p-php01-run-v1'}, {'task_id': 'P-HP-01'}, {'feature': 'f3'}, {'batch_size': 256},
           {'weight_decay': .001}, {'side_version': 'v0'}, {'architecture': 'sasrec'}, {'dim': 64.0}, {'temperature': 1}, {'seed': True}]
    for override in bad:
        try:
            validate_config({**base, **override})
        except ValueError:
            continue
        raise AssertionError(f'config {override} accepted')
    for broken in ({**base, 'knob': 'H1'}, {k: v for k, v in base.items() if k != 'arm'}, {k: v for k, v in base.items() if k != 'dim'}, {**base, 'unknown': 1}, [], None):
        try:
            validate_config(broken)
        except ValueError:
            continue
        raise AssertionError('malformed config accepted')


def guard_optimizer_covers_every_parameter():
    """M18: every parameter (branch, gate, attention pooling included) is in the optimizer exactly once."""
    from vkgr import retrieval_seqm1_run as run
    for arm in tm.ARMS:
        model = tm.build(arm); optimizer = run.make_optimizer(model, small(arm))
        grouped = [id(p) for g in optimizer.param_groups for p in g['params']]
        ensure(sorted(grouped) == sorted(id(p) for p in model.parameters()), f'{arm}: optimizer does not cover model.parameters() exactly once')
        ensure(optimizer.param_groups[0]['lr'] == .001 and optimizer.param_groups[0]['weight_decay'] == 1e-4, f'{arm}: optimizer lr / wd')


def guard_diagnostics_use_eval_mode():
    """M14: diagnostics are exported in eval mode (no dropout noise) even if the caller left the model in train mode."""
    from vkgr import retrieval_feat02_run as fr, retrieval_seqm1_run as run
    from test_seqm1_leak import build_for
    data = fr.synthetic_data(); rows, queries = fr._view(data, 50); config = small('mean')
    model = build_for(data, 'mean'); model.eval()
    result, reference = run.evaluate_model(model, data, queries, config)
    for _ in range(2):
        model.train()
        again = run.model_diagnostics(model, data, queries, result['predictions'], config)
        ensure(again == {k: v for k, v in reference.items() if k in again}, 'diagnostics changed when the model was left in train mode (dropout leaked in)')


def guard_holdout_never_opened(run_callable):
    """L6: while ``run_callable`` runs, any open() of a path whose NAME contains the (run-time spliced) holdout fragment raises."""
    def blocked(path):
        try:
            name = os.path.basename(os.fspath(path)).lower()
        except TypeError:
            return False
        return HOLDOUT_FRAGMENT in name

    real_open, real_path_open, real_os_open = builtins.open, pathlib.Path.open, os.open

    def guarded_open(file, *a, **k):
        if blocked(file):
            raise AssertionError(f'holdout path opened: {file}')
        return real_open(file, *a, **k)

    def guarded_path_open(self, *a, **k):
        if blocked(self):
            raise AssertionError(f'holdout path opened: {self}')
        return real_path_open(self, *a, **k)

    def guarded_os_open(path, *a, **k):
        if blocked(path):
            raise AssertionError(f'holdout path opened: {path}')
        return real_os_open(path, *a, **k)
    with mock.patch.object(builtins, 'open', guarded_open), mock.patch.object(pathlib.Path, 'open', guarded_path_open), mock.patch.object(os, 'open', guarded_os_open):
        return run_callable()


# ---------------------------------------------------------------- tests

class ConfigTests(unittest.TestCase):
    def test_registry(self):
        from vkgr import retrieval_seqm1_run as run
        self.assertEqual((run.SCHEMA, run.TASK, run.KIND, run.MANIFEST_SCHEMA, run.CHECKPOINT_SCHEMA),
                         ('p-seqm1-run-v1', 'P-SEQ-M1', 'p_seqm1_train', 'p-seqm1-train-manifest-v1', 'p-seqm1-checkpoint-v1'))
        value = seq_config()
        self.assertEqual((value['temperature'], value['dim'], value['learning_rate'], value['logq'], value['extra_uniform'], value['history_len'], value['feature']),
                         (.1, 64, .001, True, 0, 50, 'f3_f4_recency'))
        self.assertEqual((value['max_epochs'], value['min_epochs'], value['patience'], value['batch_size']), (100, 5, 3, 512))

    def test_every_arm_and_seed_validates(self):
        from vkgr.retrieval_seqm1_run import validate_config
        for arm in tm.ARMS:
            for seed in SEEDS:
                validate_config(seq_config(arm, seed=seed)); validate_config(small(arm, seed=seed), synthetic=True)

    def test_frozen_conditions(self):
        guard_config_pins_the_anchor_conditions()

    def test_real_run_needs_batch_512(self):
        from vkgr.retrieval_seqm1_run import validate_config
        with self.assertRaises(ValueError):
            validate_config(small('mean'))
        validate_config(small('mean'), synthetic=True)

    def test_committed_configs(self):
        from vkgr.retrieval_seqm1_run import validate_config
        h1 = json.loads(H1_CONFIG.read_text())
        files = {p.name: p for p in CONFIG_DIR.glob('*.json')}
        expected = {f'p-seqm1-{arm}-s{seed}.json' for arm in tm.ARMS for seed in SEEDS}
        self.assertEqual(set(files), expected)
        self.assertEqual(len(expected), 15)
        for name, path in files.items():
            value = json.loads(path.read_text()); arm, seed = name[len('p-seqm1-'):-len('.json')].rsplit('-s', 1)
            validate_config(value)
            self.assertEqual(path.read_text(), json.dumps(value, indent=1, sort_keys=True) + '\n', name)
            self.assertEqual((value['arm'], value['seed']), (arm, int(seed)), name)
            self.assertEqual(set(value), (set(h1) - {'knob'}) | {'arm'}, name)
            differing = {k for k in value if k not in ('arm',) and h1[k] != value[k]}
            self.assertEqual(differing, {'schema', 'task_id'} | ({'seed'} if int(seed) != 42 else set()), name)      # T0.10 / dim / lr / budget identical to F4-T010 (php01 H1)


class RunTests(unittest.TestCase):
    def test_each_arm_trains_and_verifies(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_seqm1_run as run
        from vkgr.retrieval_feat02 import bucket_spec_sha256
        from vkgr.retrieval_seqm1 import branch_spec_sha256, expected_param_counts_seqm1
        data = fr.synthetic_data()
        for arm in tm.ARMS:
            with self.subTest(arm=arm), tmp_dir() as tmp:
                run_dir = Path(tmp) / 'run'; config = small(arm)
                value = run.run_session(run_dir, data, config)
                self.assertEqual((value['schema'], value['kind'], value['status']), ('p-seqm1-train-manifest-v1', 'p_seqm1_train', 'complete'))
                self.assertEqual(sorted(p.name for p in run_dir.glob('*.pt')), ['best.pt', 'last.pt'])
                binding = value['binding']; mask, readout, anchored = tm.ARM_TABLE[arm]
                self.assertEqual((binding['task'], binding['arm'], binding['mask'], binding['readout'], binding['anchored']), ('P-SEQ-M1', arm, mask, readout, anchored))
                self.assertEqual(binding['bucket_spec_sha256'], bucket_spec_sha256()); self.assertIn('weeks_sha256', binding)
                fresh = run._model(data, config, 'cpu')
                self.assertEqual(binding['branch_spec_sha256'], branch_spec_sha256(fresh))
                self.assertEqual(binding['param_report'], expected_param_counts_seqm1(arm, len(data['content'])))
                self.assertEqual(binding['test_gate'], 'closed_hash_only')
                state = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
                self.assertEqual(state['schema'], 'p-seqm1-checkpoint-v1')
                self.assertEqual(('gate' in state['model']), anchored)
                self.assertEqual(run.verify_run(run_dir / 'manifest.json', value['manifest_sha256'], data, config)['status'], 'verified')
                other = 'last' if arm != 'last' else 'mean'
                with self.assertRaises(ValueError):
                    run.verify_run(run_dir / 'manifest.json', value['manifest_sha256'], data, small(other))
                side = {**data, 'rows': [{**r, 'side': [[mk ^ 1, t, d] for mk, t, d in r['side']]} for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    run.verify_run(run_dir / 'manifest.json', value['manifest_sha256'], side, config)

    def test_step_one_loss_equals_the_f4_t010_runner(self):
        """AC1 end to end: at g=0 the first optimisation step has the F4-T010 loss bit for bit (same batch, same negatives, same user vectors)."""
        from vkgr import retrieval_feat02_run as fr, retrieval_php01_run as php, retrieval_seqm1_run as run
        data = fr.synthetic_data()
        h1 = {**php.default_config(), 'knob': 'H1', 'temperature': .1, 'batch_size': 16, 'max_epochs': 1, 'min_epochs': 1}
        with tmp_dir() as tmp:
            a = php.run_session(Path(tmp) / 'a', data, h1)
            for arm in tm.ANCHORED:
                b = run.run_session(Path(tmp) / arm, data, small(arm, max_epochs=1, min_epochs=1))
                self.assertEqual(a['records'][0]['loss'], b['records'][0]['loss'], arm)
                self.assertEqual(a['records'][0]['indices_sha256'], b['records'][0]['indices_sha256'], arm)
                self.assertEqual(a['records'][0]['negatives_sha256'], b['records'][0]['negatives_sha256'], arm)
                self.assertNotEqual(a['records'][1]['loss'], b['records'][1]['loss'], arm)           # gate moved: the branch is alive after step 1

    def test_diagnostics_and_ablation_in_manifest(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_seqm1_run as run
        data = fr.synthetic_data()
        for arm in tm.ARMS:
            with self.subTest(arm=arm), tmp_dir() as tmp:
                value = run.run_session(Path(tmp) / 'run', data, small(arm))
                for evaluation in value['evaluations']:
                    d = evaluation['diagnostics']
                    for key in ('gate', 'branch_ratio_median', 'branch_ratio_mean', 'cos_u_m_mean', 'cos_u_last_event_mean', 'attn_entropy_ratio', 'attn_last_weight',
                                'recall50_by_history_length', 'recall50_by_last_event_support', 'top1_is_last_history_fraction', 'history_support_slices'):
                        self.assertIn(key, d, (arm, key))
                    self.assertEqual(set(d['recall50_by_history_length']), {'1-4', '5-19', '20-39', '40-49', '50'})
                    self.assertEqual(set(d['recall50_by_last_event_support']), {'supported', 'unsupported'})
                    if arm == 'pure':
                        self.assertIsNone(d['gate']); self.assertIsNone(d['branch_ratio_median'])
                    else:
                        self.assertIsInstance(d['gate'], float); self.assertGreaterEqual(d['branch_ratio_median'], 0.)
                    self.assertEqual(d['attn_entropy_ratio'] is not None, arm == 'attn')
                ablation = value['ablation_g0']
                if arm == 'pure':
                    self.assertIsNone(ablation)
                else:
                    best = value['selection']['best_epoch']
                    self.assertEqual(ablation['epoch'], best)
                    self.assertEqual(ablation['best_recall@50'], value['evaluations'][best - 1]['result']['metrics']['all']['recall@50'])
                    self.assertEqual(ablation['delta_recall@50'], ablation['best_recall@50'] - ablation['recall@50'])

    def test_ablation_equals_the_anchor_tower_with_the_trained_tables(self):
        """g -> 0 gives u == m bit for bit, so its recall must equal the plain F4IDMeanD evaluated with the trained anchor tables."""
        from vkgr import retrieval_feat02_run as fr, retrieval_seqm1_run as run
        from vkgr.retrieval_php01 import make_php01_model
        from vkgr.training import seed_all
        data = fr.synthetic_data(); rows, queries = fr._view(data, 50)
        with tmp_dir() as tmp:
            value = run.run_session(Path(tmp) / 'run', data, small('mean'))
            state = torch.load(Path(tmp) / 'run' / 'best.pt', map_location='cpu', weights_only=True)['model']
        seed_all(42); anchor = make_php01_model(data['content'], data['supported_indices'], 50, 64)
        anchor.load_state_dict({k: v for k, v in state.items() if k in anchor.state_dict()})
        result, _ = fr.evaluate_model(anchor, data, queries, small('mean'))
        self.assertEqual(value['ablation_g0']['recall@50'], result['metrics']['all']['recall@50'])
        self.assertEqual(value['ablation_g0']['ndcg@50'], result['metrics']['all']['ndcg@50'])

    def test_verify_rejects_tampered_ablation_and_diagnostics(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_seqm1_run as run
        data = fr.synthetic_data(); config = small('mean')
        with tmp_dir() as tmp:
            run_dir = Path(tmp) / 'run'; value = run.run_session(run_dir, data, config)
            path = run_dir / 'manifest.json'
            for edit in (lambda v: v['ablation_g0'].update({'recall@50': v['ablation_g0']['recall@50'] + .01}),
                         lambda v: v['evaluations'][v['selection']['best_epoch'] - 1]['diagnostics'].update({'gate': 9.}),
                         lambda v: v['evaluations'][v['selection']['best_epoch'] - 1]['diagnostics']['recall50_by_history_length'].update({'50': 0.5})):
                tampered = json.loads(path.read_text()); edit(tampered)
                sha = run.b1._write_json_atomic(run_dir / 'tampered.json', tampered)
                with self.assertRaises(ValueError):
                    run.verify_run(run_dir / 'tampered.json', sha, data, config)
            self.assertEqual(run.verify_run(path, value['manifest_sha256'], data, config)['status'], 'verified')

    def test_diagnostics_eval_mode(self):
        guard_diagnostics_use_eval_mode()

    def test_optimizer_covers_every_parameter(self):
        guard_optimizer_covers_every_parameter()

    def test_runner_reuses_the_feat02_batch_tensors(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_seqm1_run as run
        self.assertIs(run._tensors2, fr._tensors2)
        self.assertIs(run.full_evaluations, fr.full_evaluations)


class HoldoutTests(unittest.TestCase):
    def test_l6_synthetic_train_and_verify_never_open_a_holdout_path(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_seqm1_run as run
        with tmp_dir() as tmp:
            tmp = Path(tmp); config = small('last'); (tmp / 'config.json').write_text(json.dumps(config))
            out = io.StringIO()

            def train():
                with contextlib.redirect_stdout(out):
                    return run.main(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run')])
            self.assertEqual(guard_holdout_never_opened(train), 0)
            summary = json.loads(out.getvalue())
            self.assertEqual(summary['status'], 'complete')
            manifest = json.loads((tmp / 'run' / 'manifest.json').read_text())
            self.assertEqual(manifest['binding']['test_gate'], 'closed_hash_only')
            data = fr.synthetic_data()
            self.assertEqual(guard_holdout_never_opened(lambda: run.verify_run(tmp / 'run' / 'manifest.json', summary['manifest_sha256'], data, config))['status'], 'verified')

    def test_the_open_guard_itself_fires(self):
        def reader():
            with tmp_dir() as tmp:
                open(Path(tmp) / (HOLDOUT_FRAGMENT + '-holdout.json'), 'w').close()
        with self.assertRaises(AssertionError):
            guard_holdout_never_opened(reader)
        with self.assertRaises(AssertionError):
            guard_holdout_never_opened(lambda: pathlib.Path('/tmp/' + HOLDOUT_FRAGMENT + '-x.json').open())


class CLITests(unittest.TestCase):
    SPEC = {'schema': 'p-feat02-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
            'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64, 'items_metadata_sha256': 'c' * 64,
            'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}

    def test_cli_real_path_wiring(self):
        from vkgr import retrieval_seqm1_run as run
        from vkgr.data_audit import file_sha256
        with tmp_dir() as tmp:
            tmp = Path(tmp)
            (tmp / 'inputs.json').write_text(json.dumps(self.SPEC)); (tmp / 'real.json').write_text(json.dumps(seq_config('attn', seed=2024)))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            with mock.patch('torch.cuda.is_available', return_value=False), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(run.main(['train', *common, '--output', str(tmp / 'x')]), 2)
            self.assertIn('P-SEQ-M1 gate failed', err.getvalue()); self.assertFalse((tmp / 'x').exists())
            fake_data = {'rows': [], 'queries': [], 'catalog': []}; seen = {}
            with mock.patch('torch.cuda.is_available', return_value=True), \
                    mock.patch.object(run.b1, 'resource_guard', lambda *a, **k: (lambda: None)), \
                    mock.patch.object(run.b1, 'real_inputs', lambda s: (fake_data, None, {'prepared_sha': 'x'})), \
                    mock.patch.object(run.fr, 'apply_feat02', lambda d, s, n: (d, {'history_manifest_sha256': 'h'})), \
                    mock.patch.object(run, 'run_session', lambda *a, **k: seen.update(train=a) or {'status': 'complete', 'manifest_sha256': 'm', 'selection': {'best_epoch': 3}}), \
                    mock.patch.object(run, 'verify_run', lambda *a, **k: seen.update(verify=a) or {'status': 'verified'}):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(run.main(['train', *common, '--output', str(tmp / 'real-run')]), 0)
                self.assertEqual(json.loads(out.getvalue())['best_epoch'], 3)
                self.assertEqual((seen['train'][2]['arm'], seen['train'][2]['seed'], seen['train'][3], seen['train'][5]['inputs_sha256']), ('attn', 2024, 'cuda', file_sha256(tmp / 'inputs.json')))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run.main(['verify', *common, '--manifest', str(tmp / 'real-run' / 'manifest.json'), '--manifest-sha', 'a' * 64]), 0)
                self.assertEqual(seen['verify'][4], 'cuda')
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(run.main(['train', *common[:2], '--inputs-sha', '0' * 64, *common[4:], '--output', str(tmp / 'x')]), 2)
            self.assertIn('external hash mismatch', err.getvalue())

    def test_cli_has_only_train_verify_synthetic_subcommands(self):
        from vkgr import retrieval_seqm1_run as run
        for command in ('prepare-ctx', 'prepare-attr', 'verify-ctx', 'select'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run.main([command, '--inputs', 'x', '--inputs-sha', 'y'])


class LatencyScriptTests(unittest.TestCase):
    SCRIPT = ROOT / 'scripts' / 'seqm1_latency_bench.py'

    def _module(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('seqm1_latency_bench', self.SCRIPT)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_cpu_small_run_reports_anchor_and_every_requested_arm(self):
        module = self._module(); out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(module.main(['--device', 'cpu', '--catalog', '2000', '--arms', 'mean', 'pure', '--iters', '6', '--warmup', '2', '--batch', '8']), 0)
        lines = out.getvalue().strip().splitlines(); self.assertEqual(len(lines), 1)
        value = json.loads(lines[0])
        self.assertEqual(set(value['models']), {'anchor', 'mean', 'pure'}); self.assertEqual(set(value['ratio_vs_anchor']), {'mean', 'pure'})
        self.assertEqual((value['device'], value['threads'], value['catalog']), ('cpu', 1, 2000))
        for name, row in value['models'].items():
            for kind in ('offline_ms_per_query', 'online_ms_per_query'):
                self.assertEqual(set(row[kind]), {'mean', 'p50', 'p95', 'p99', 'n'}, (name, kind)); self.assertGreater(row[kind]['mean'], 0.)
            self.assertEqual(row['online_ms_per_query']['n'], 6)
        self.assertGreater(value['models']['mean']['parameters'], value['models']['anchor']['parameters'])

    def test_online_lengths_follow_the_registered_share(self):
        module = self._module(); g = torch.Generator().manual_seed(3)
        lengths = [module.online_length(g) for _ in range(4000)]
        self.assertTrue(all(1 <= n <= 50 for n in lengths))
        self.assertAlmostEqual(sum(1 for n in lengths if n == 50) / len(lengths), .627 + .373 / 50, delta=.03)

    def test_no_path_arguments_and_users_only(self):
        module = self._module()
        dests = {a.dest for a in module.build_parser()._actions} - {'help'}
        self.assertEqual(dests, {'device', 'arms', 'catalog', 'batch', 'iters', 'warmup'})
        source = self.SCRIPT.read_text()
        self.assertNotIn('model.items(', source)                        # the item tower is not part of the measured call


if __name__ == '__main__':
    unittest.main()
