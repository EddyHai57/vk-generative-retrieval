"""Batch1 P switches: init, min_epochs early stop, logQ/mixed negatives, best+last run + fail-closed verify."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'batch1_early_stop_cases.json'


def b1_config(**overrides):
    from vkgr.retrieval_b1_run import default_config
    value = default_config()
    value.update(overrides)
    return value


class EarlyStopTests(unittest.TestCase):
    def test_shared_case_table(self):
        from vkgr.retrieval_hp import EarlyStop
        cases = json.loads(FIXTURE.read_text())['cases']
        self.assertGreaterEqual(len(cases), 6)
        for case in cases:
            stop = EarlyStop(case['max_epochs'], case['min_epochs'], case['patience'])
            for epoch, (recall, ndcg) in enumerate(case['metrics'], 1):
                self.assertFalse(stop.stopped, case['name'])
                stop.observe(epoch, recall, ndcg)
            self.assertTrue(stop.stopped, case['name'])
            self.assertEqual(stop.epoch, case['stop_epoch'], case['name'])
            self.assertEqual(stop.best_epoch, case['best_epoch'], case['name'])

    def test_min0_max30_matches_legacy_validation_selection(self):
        import random
        from vkgr.retrieval_hp import EarlyStop
        from vkgr.retrieval_experiment import ValidationSelection
        rng = random.Random(3)
        for _ in range(50):
            legacy = ValidationSelection(); new = EarlyStop(30, 0, 3)
            epoch = 0
            while not legacy.stopped:
                epoch += 1; metric = (rng.choice([0.1, 0.2, 0.3]), rng.choice([0.0, 0.5]))
                self.assertEqual(legacy.observe(epoch, *metric), new.observe(epoch, *metric))
                self.assertEqual(legacy.stopped, new.stopped)
            self.assertEqual((legacy.best_epoch, legacy.best_metric, legacy.epoch), (new.best_epoch, new.best_metric, new.epoch))

    def test_rejects_invalid_budget(self):
        from vkgr.retrieval_hp import EarlyStop
        for args in ((0, 0, 3), (100, -1, 3), (100, 5, 0), (5, 6, 3)):
            with self.assertRaises(ValueError):
                EarlyStop(*args)


class InitTests(unittest.TestCase):
    def setUp(self):
        self.content = torch.zeros(301, 64)
        self.supported = list(range(1, 200))

    def test_legacy_is_value_equal_to_frozen_models(self):
        from vkgr.retrieval_hp import make_model
        from vkgr.retrieval_models import TwoTower
        from vkgr.retrieval_sasrec import SASRecIDTwoTower
        from vkgr.training import seed_all
        for arch, frozen in (('id_mean', lambda: TwoTower('id_mean', self.content, self.supported)),
                             ('sasrec', lambda: SASRecIDTwoTower(self.content, self.supported))):
            seed_all(42); expected = frozen().state_dict()
            seed_all(42); actual = make_model(arch, 'legacy', self.content, self.supported, 50).state_dict()
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]), (arch, key))

    def test_small_init_only_touches_embeddings(self):
        from vkgr.retrieval_hp import make_model
        from vkgr.training import seed_all
        for arch in ('id_mean', 'sasrec'):
            seed_all(42); legacy = make_model(arch, 'legacy', self.content, self.supported, 50)
            seed_all(42); small = make_model(arch, 'small', self.content, self.supported, 50)
            embeddings = [name for name, m in small.named_modules() if isinstance(m, torch.nn.Embedding)]
            self.assertTrue(embeddings)
            for name, module in small.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    weight = module.weight.detach()
                    rows = weight[1:] if module.padding_idx == 0 else weight
                    self.assertAlmostEqual(float(rows.std()), 0.02, delta=0.002, msg=(arch, name))
                    if module.padding_idx is not None:
                        self.assertTrue(torch.equal(weight[module.padding_idx], torch.zeros(weight.shape[1])))
            legacy_state = legacy.state_dict(); small_state = small.state_dict()
            for key in legacy_state:
                if not any(key.startswith(e + '.') for e in embeddings):
                    self.assertTrue(torch.equal(legacy_state[key], small_state[key]), (arch, key))
            report = small.init_report
            self.assertEqual(report['init'], 'small')
            for name in embeddings:
                self.assertAlmostEqual(report['embeddings'][name]['std'], 0.02, delta=0.002)
                self.assertIn(report['embeddings'][name]['pad_row_zero'], (True, None))

    def test_id_mean_l100_accepts_long_history_and_l50_matches(self):
        from vkgr.retrieval_hp import make_model
        from vkgr.retrieval_models import TwoTower
        from vkgr.training import seed_all
        seed_all(1); frozen = TwoTower('id_mean', self.content, self.supported)
        seed_all(1); l100 = make_model('id_mean', 'legacy', self.content, self.supported, 100)
        histories = torch.randint(1, 300, (4, 50))
        self.assertTrue(torch.equal(frozen.users(histories), l100.users(histories)))
        self.assertEqual(l100.users(torch.randint(1, 300, (2, 100))).shape, (2, 64))
        with self.assertRaises(ValueError):
            l100.users(torch.randint(1, 300, (2, 101)))
        with self.assertRaises(ValueError):
            make_model('sasrec', 'small', self.content, self.supported, 100)


class NegativeTests(unittest.TestCase):
    def test_masked_ce_logq_none_is_masked_ce(self):
        from vkgr.retrieval_negatives import masked_ce_logq
        from vkgr.retrieval_training import masked_ce
        torch.manual_seed(0)
        users = torch.randn(6, 8); items = torch.randn(9, 8)
        labels = torch.tensor([0, 1, 2, 3, 4, 5]); mask = torch.ones(6, 9, dtype=torch.bool); mask[0, 7] = False
        a, sa = masked_ce(users, items, labels, mask, .15)
        b, sb = masked_ce_logq(users, items, labels, mask, .15, None)
        self.assertEqual(sa, sb)
        self.assertTrue(torch.equal(a, b))

    def test_logq_subtracts_alpha_log_q_from_every_column(self):
        from vkgr.retrieval_negatives import masked_ce_logq
        torch.manual_seed(1)
        users = torch.randn(3, 4); items = torch.randn(5, 4)
        labels = torch.tensor([0, 1, 2]); mask = torch.ones(3, 5, dtype=torch.bool)
        log_q = torch.log(torch.tensor([.5, .2, .1, .1, .1]))
        loss, _ = masked_ce_logq(users, items, labels, mask, .15, log_q, alpha=1.0)
        scores = torch.nn.functional.normalize(users, dim=1) @ torch.nn.functional.normalize(items, dim=1).T / .15 - log_q
        expected = torch.nn.functional.cross_entropy(scores, labels)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        with self.assertRaises(ValueError):
            masked_ce_logq(users, items, labels, mask, .15, torch.tensor([0., 0., float('inf'), 0., 0.]))

    def test_target_frequency_and_logq_tables(self):
        from vkgr.retrieval_negatives import target_frequency, logq_batch, logq_mixed
        rows = [{'target': t} for t in (5, 5, 6, 7, 5)]
        freq = target_frequency(rows)
        self.assertEqual(freq['counts'], {5: 3, 6: 1, 7: 1})
        self.assertEqual(freq['total'], 5)
        batch = logq_batch([5, 6], freq)
        self.assertTrue(torch.allclose(batch, torch.log(torch.tensor([3 / 5, 1 / 5]))))
        mixed = logq_mixed([5, 6, 9], {9}, freq, batch_size=4, pool_size=10, extra_count=2)
        expected = torch.log(torch.tensor([4 * 3 / 5, 4 * 1 / 5, 2 / 10]))
        self.assertTrue(torch.allclose(mixed, expected))
        with self.assertRaises(ValueError):
            logq_batch([8], freq)

    def test_supported_uniform_columns(self):
        import random
        from vkgr.retrieval_negatives import supported_uniform_columns
        catalog = list(range(400)); idmap = {i: i + 1 for i in catalog}; pool = list(range(100, 300))
        rows = [{'target': 100 + j, 'past_likes': [100 + j, 150]} for j in range(8)]
        columns, labels, mask, negatives, pool_size = supported_uniform_columns(rows, pool, idmap, random.Random(4), 128)
        self.assertEqual(len(negatives), 128)
        self.assertEqual(len(set(negatives)), 128)
        self.assertTrue(set(negatives) <= set(pool) - {r['target'] for r in rows})
        self.assertEqual(pool_size, 200 - 8)
        self.assertEqual(columns.tolist(), [idmap[i] for i in sorted({r['target'] for r in rows} | set(negatives))])
        self.assertTrue(mask[torch.arange(8), labels].all())
        lookup = sorted({r['target'] for r in rows} | set(negatives))
        if 150 in lookup:
            self.assertFalse(mask[1, lookup.index(150)])
        with self.assertRaises(ValueError):
            supported_uniform_columns(rows, list(range(100, 130)), idmap, random.Random(4), 128)


class RunTests(unittest.TestCase):
    def test_replay_config_reproduces_p4e_watch_history_epoch_by_epoch(self):
        from vkgr import retrieval_ablation_run as p4e
        from vkgr import retrieval_b1_run as b1
        with tempfile.TemporaryDirectory() as tmp:
            legacy = p4e.run_session(Path(tmp) / 'p4e', p4e.synthetic_data(), 'watch_history', 42, 8)
            data = b1.p4e_watch_synthetic()
            config = b1_config(task_id='P-REPLAY', init='legacy', max_epochs=30, min_epochs=0, batch_size=8)
            new = b1.run_session(Path(tmp) / 'b1', data, config)
            full = b1.full_evaluations(Path(tmp) / 'b1', new)
            self.assertEqual(len(full), len(legacy['evaluations']))
            for a, b in zip(legacy['evaluations'], full):
                self.assertEqual(a['epoch'], b['epoch'])
                self.assertEqual(a['result'], b['result'])
            self.assertEqual(new['selection']['best_epoch'], legacy['selection']['best_epoch'])
            self.assertEqual(b1.compare_replay(full, legacy['evaluations'])['equal'], True)
            changed = json.loads(json.dumps(legacy['evaluations'])); changed[0]['result']['metrics']['all']['recall@50'] = 0.123
            self.assertEqual(b1.compare_replay(full, changed)['equal'], False)
            b1.verify_run(Path(tmp) / 'b1' / 'manifest.json', new['manifest_sha256'], data, config)
            self._assert_same_trajectory(Path(tmp) / 'p4e', legacy, Path(tmp) / 'b1', new)

    def _assert_same_trajectory(self, p4e_dir, legacy, b1_dir, new):
        self.assertEqual([r['loss'] for r in legacy['records']], [r['loss'] for r in new['records']])
        self.assertEqual([r['gradient_norm'] for r in legacy['records']], [r['gradient_norm'] for r in new['records']])
        a = torch.load(p4e_dir / legacy['last']['path'], weights_only=True)['model']
        b = torch.load(b1_dir / 'last.pt', weights_only=True)['model']
        self.assertEqual(set(a), set(b))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_replay_matches_p4e_on_nondegenerate_fixture(self):
        from vkgr import retrieval_ablation_run as p4e
        from vkgr import retrieval_b1_run as b1
        data = b1.synthetic_data()
        rows, queries = b1._view(data, 50)
        data50 = {**data, 'rows': rows, 'queries': queries}
        legacy_data = {**data50, 'watch_rows': rows, 'watch_queries': queries}
        with tempfile.TemporaryDirectory() as tmp:
            legacy = p4e.run_session(Path(tmp) / 'p4e', legacy_data, 'watch_history', 2024, 16)
            config = b1_config(task_id='P-REPLAY', init='legacy', max_epochs=30, min_epochs=0, batch_size=16, seed=2024)
            new = b1.run_session(Path(tmp) / 'b1', data50, config)
            full = b1.full_evaluations(Path(tmp) / 'b1', new)
            self.assertTrue(b1.compare_replay(full, legacy['evaluations'])['equal'])
            self.assertGreater(len({json.dumps(e['result']['predictions'][:3]) for e in legacy['evaluations']}), 1, 'fixture must move predictions')
            self._assert_same_trajectory(Path(tmp) / 'p4e', legacy, Path(tmp) / 'b1', new)

    def _run(self, tmp, **overrides):
        from vkgr import retrieval_b1_run as b1
        data = b1.synthetic_data()
        config = b1_config(batch_size=16, max_epochs=8, min_epochs=2, **overrides)
        value = b1.run_session(Path(tmp) / 'run', data, config)
        return b1, data, config, value

    def test_best_last_only_and_fail_closed_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            b1, data, config, value = self._run(tmp)
            run = Path(tmp) / 'run'
            self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
            self.assertEqual(value['status'], 'complete')
            self.assertIn('zero_user_vector_fraction', value['evaluations'][0]['diagnostics'])
            self.assertIn('history_supported_fraction', value['data_diagnostics'])
            sha = value['manifest_sha256']
            verified = b1.verify_run(run / 'manifest.json', sha, data, config)
            self.assertEqual(verified['replayed_epochs'], [value['best']['epoch']])
            with self.assertRaises(ValueError):
                b1.verify_run(run / 'manifest.json', '0' * 64, data, config)
            with self.assertRaises(ValueError):
                b1.verify_run(run / 'manifest.json', sha, data, {**config, 'learning_rate': .0003})
            original = (run / 'best.pt').read_bytes()
            (run / 'best.pt').write_bytes(original + b'x')
            with self.assertRaises(ValueError):
                b1.verify_run(run / 'manifest.json', sha, data, config)
            (run / 'best.pt').write_bytes(original)
            b1.verify_run(run / 'manifest.json', sha, data, config)
            state = torch.load(run / 'best.pt', weights_only=True)
            state['model']['embedding.weight'][1] += 1.0
            (run / 'best.pt').unlink(); torch.save(state, run / 'best.pt')
            with self.assertRaises(ValueError):
                b1.verify_run(run / 'manifest.json', sha, data, config)

    def test_verify_rejects_forged_manifest_even_with_rehash(self):
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            b1, data, config, value = self._run(tmp)
            run = Path(tmp) / 'run'
            for mutate in (lambda m: m['evaluations'][0]['result']['metrics']['all'].__setitem__('recall@50', 0.99),
                           lambda m: m['records'][0].__setitem__('indices_sha256', '0' * 64),
                           lambda m: m['selection'].__setitem__('best_epoch', 1 if m['selection']['best_epoch'] != 1 else 2),
                           lambda m: m.__setitem__('status', 'running')):
                manifest = json.loads((run / 'manifest.json').read_text()); mutate(manifest)
                forged = Path(tmp) / 'run' / 'forged.json'
                if forged.exists():
                    forged.unlink()
                forged.write_text(json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n')
                with self.assertRaises(ValueError):
                    b1.verify_run(forged, file_sha256(forged), data, config)

    def test_logq_and_mixed_negative_runs_verify(self):
        for overrides in ({'logq': True}, {'extra_uniform': 128}, {'logq': True, 'extra_uniform': 128}):
            with self.subTest(**overrides), tempfile.TemporaryDirectory() as tmp:
                b1, data, config, value = self._run(tmp, task_id='P-NEG-01', **overrides)
                self.assertTrue(all(math.isfinite(r['loss']) for r in value['records'] if r['loss'] is not None))
                if overrides.get('extra_uniform'):
                    self.assertTrue(all(r['negatives'] == 128 for r in value['records']))
                b1.verify_run(Path(tmp) / 'run' / 'manifest.json', value['manifest_sha256'], data, config)

    def test_sasrec_small_init_run_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            b1, data, config, value = self._run(tmp, task_id='P4H-v2', architecture='sasrec')
            self.assertEqual(value['binding']['init_report']['init'], 'small')
            b1.verify_run(Path(tmp) / 'run' / 'manifest.json', value['manifest_sha256'], data, config)

    def test_l100_id_mean_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            b1, data, config, value = self._run(tmp, task_id='HIST-LEN', history_len=100)
            self.assertGreater(max(len(r['history']) for r in data['rows']), 50)
            b1.verify_run(Path(tmp) / 'run' / 'manifest.json', value['manifest_sha256'], data, config)

    def test_committed_batch1_configs_are_registered(self):
        from vkgr.retrieval_b1_run import validate_config
        from vkgr.retrieval_ablation_run import strict_json
        paths = sorted((Path(__file__).resolve().parents[1] / 'configs' / 'batch1').glob('p-*.json'))
        self.assertEqual(len(paths), 11)
        for path in paths:
            validate_config(strict_json(path.read_text()))

    def test_cli_synthetic_train(self):
        import contextlib
        import io
        from vkgr import retrieval_b1_run as b1
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'c.json'; config.write_text(json.dumps(b1_config(batch_size=16, max_epochs=2, min_epochs=0)))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(b1.main(['synthetic-train', '--config', str(config), '--output', str(Path(tmp) / 'run')]), 0)
            self.assertEqual(json.loads(out.getvalue())['status'], 'complete')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(b1.main(['train', '--inputs', str(config), '--inputs-sha', '0' * 64]), 2)

    def test_config_registry_rejects_unregistered_values(self):
        from vkgr.retrieval_b1_run import validate_config
        validate_config(b1_config())
        for bad in ({'learning_rate': .01}, {'init': 'xavier'}, {'history_len': 30}, {'architecture': 'sasrec', 'history_len': 100},
                    {'max_epochs': 30}, {'logq_alpha': .5}, {'extra_uniform': 64}, {'checkpoint_policy': 'every_epoch'},
                    {'init': 'legacy'}, {'unknown': 1}, {'patience': 4}):
            with self.assertRaises(ValueError, msg=bad):
                validate_config(b1_config(**bad))
        validate_config(b1_config(task_id='P-REPLAY', init='legacy', max_epochs=30, min_epochs=0))


if __name__ == '__main__':
    unittest.main()
