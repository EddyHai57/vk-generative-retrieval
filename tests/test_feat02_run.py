"""FEAT-02 (P line) runner tests: equivalence with the FEAT-01 runner, anti-leak batch tensors, train + verify round trip."""
import contextlib
import copy
import inspect
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_model import ensure  # noqa: E402
from test_feat01 import prepared_world  # noqa: E402
from test_feat02_data import prepared_full_world  # noqa: E402


def f2_config(**overrides):
    from vkgr.retrieval_feat02_run import default_config
    value = default_config(); value.update(overrides)
    return value


def small_config(**overrides):
    return f2_config(batch_size=16, max_epochs=8, min_epochs=2, **overrides)


def flip_non_history_fields(item):
    """Everything about the target (and the within-week ordinal of the history positions) changed; history weeks kept."""
    item = copy.deepcopy(item)
    item['target'] = (item['target'] + 1) % 300
    week, ordinal = item['target_position']
    item['target_position'] = [(week + 7) % 31, ordinal + 99]
    if 'past_likes' in item:
        item['past_likes'] = [0, 1, 2]
    if 'oov' in item:
        item['oov'] = not item['oov']
    item['history_positions'] = [[w, 7] for w, _ in item['history_positions']]
    return item


def guard_target_week_change_leaves_feature_unchanged():
    from vkgr import retrieval_feat02_run as fr
    data = fr.synthetic_data(); rows, queries = fr._view(data, 50)
    model = fr._model(data, small_config(feature='f3_f4_recency'), 'cpu'); model.eval()
    picks = (lambda r: r['history_positions'][-1][0], lambda r: 24, lambda r: 26, lambda r: 30)
    for batch in (rows[:24], queries[:24]):
        width = max(len(r['history']) for r in batch)
        base = fr._tensors2(batch, data['idmap'], width, 'f3_f4_recency'); base_users = model.users(*base)
        for pick in picks:
            moved = []
            for r in batch:
                m = flip_non_history_fields(r); m['target_position'] = [pick(r), 5]
                moved.append(m)
            got = fr._tensors2(moved, data['idmap'], width, 'f3_f4_recency')
            ensure(torch.equal(got[0], base[0]) and torch.equal(got[1], base[1]), 'batch tensors changed with the target week/position')
            ensure(torch.equal(model.users(*got), base_users), 'user vectors changed with the target week/position')


class Feat02RunTests(unittest.TestCase):
    def _run(self, tmp, **overrides):
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data()
        config = small_config(**overrides)
        value = fr.run_session(Path(tmp) / 'run', data, config)
        return fr, data, config, value

    def test_synthetic_data_has_real_week_spread(self):
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data()
        weeks = [p[0] for r in data['rows'] + data['queries'] for p in r['history_positions']]
        self.assertEqual((min(weeks), max(weeks)), (0, 24))
        for r in data['rows'] + data['queries']:
            positions = [tuple(p) for p in r['history_positions']]
            self.assertEqual(positions, sorted(positions)); self.assertEqual(len(set(positions)), len(positions))
            self.assertEqual(len(r['side']), len(r['history']))
        for r in data['rows']:
            self.assertTrue(tuple(r['history_positions'][-1]) < tuple(r['target_position']) and r['target_position'][0] <= 24)
        self.assertTrue(all(q['target_position'][0] == 25 for q in data['queries']))
        self.assertTrue(len({r['target_position'][0] for r in data['rows']}) > 3)

    def test_config_registry_feat02(self):
        from vkgr.retrieval_feat02 import FEATURES
        from vkgr.retrieval_feat02_run import default_config, validate_config
        self.assertEqual(FEATURES[:3], ('f3', 'f3_masked', 'f3_f4_recency'))
        self.assertIn(default_config()['feature'], FEATURES)
        self.assertEqual((default_config()['schema'], default_config()['task_id']), ('p-feat02-run-v1', 'FEAT-02'))
        for feature in FEATURES:
            validate_config(f2_config(feature=feature))
        for bad in ({'feature': 'f3_watch_buckets'}, {'feature': 'none'}, {'feature': 'f4_recency'}, {'init': 'legacy'}, {'architecture': 'sasrec'},
                    {'history_len': 20}, {'max_epochs': 30}, {'side_version': 'v0'}, {'schema': 'p-feat01-run-v1'}, {'task_id': 'FEAT-01'}, {'unknown': 1}):
            with self.assertRaises(ValueError, msg=bad):
                validate_config(f2_config(**bad))
        with self.assertRaises(ValueError):
            validate_config({k: v for k, v in f2_config().items() if k != 'feature'})

    def test_f3_control_runner_equals_feat01_runner(self):
        from vkgr import retrieval_feat01_run as old
        from vkgr import retrieval_feat02_run as new
        for extra in ({}, {'logq': True}):
            with self.subTest(**extra), tempfile.TemporaryDirectory() as tmp:
                data = new.synthetic_data()
                a = old.run_session(Path(tmp) / 'old', data, {**old.default_config(), 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2, 'feature': 'f3_watch_buckets', **extra})
                b = new.run_session(Path(tmp) / 'new', data, small_config(feature='f3', **extra))
                self.assertGreater(len(a['records']), 8)
                for key in ('loss', 'gradient_norm', 'drawn', 'skipped', 'indices_sha256', 'negatives_sha256'):
                    self.assertEqual([r[key] for r in a['records']], [r[key] for r in b['records']], key)
                self.assertEqual(a['selection'], b['selection'])
                self.assertEqual([e['result'] for e in old.full_evaluations(Path(tmp) / 'old', a)], [e['result'] for e in new.full_evaluations(Path(tmp) / 'new', b)])
                for name in ('best.pt', 'last.pt'):
                    sa = torch.load(Path(tmp) / 'old' / name, map_location='cpu', weights_only=True)['model']
                    sb = torch.load(Path(tmp) / 'new' / name, map_location='cpu', weights_only=True)['model']
                    self.assertEqual(list(sa), list(sb))
                    for key in sa:
                        self.assertTrue(torch.equal(sa[key], sb[key]), (name, key))

    def test_batch_tensors_ignore_target_fields(self):
        from vkgr import retrieval_feat02_run as fr
        from vkgr.retrieval_feat02 import FEATURES
        source = inspect.getsource(fr._tensors2)
        for name in ('target', 'past_likes', 'oov'):
            self.assertNotIn(name, source)
        data = fr.synthetic_data(); rows, queries = fr._view(data, 50)
        batch = rows[:16] + queries[:16]; width = max(len(r['history']) for r in batch)
        flipped = [flip_non_history_fields(r) for r in batch]
        self.assertNotEqual([r['target'] for r in batch], [r['target'] for r in flipped])
        for feature in FEATURES:
            a = fr._tensors2(batch, data['idmap'], width, feature); b = fr._tensors2(flipped, data['idmap'], width, feature)
            self.assertTrue(torch.equal(a[0], b[0]), feature); self.assertTrue(torch.equal(a[1], b[1]), feature)
            self.assertEqual(tuple(a[1].shape), (len(batch), width, 6))
        # control: the history weeks (and only for F4) do reach the side tensor
        moved = copy.deepcopy(batch); moved[0]['history_positions'][0] = [moved[0]['history_positions'][-1][0], 0]
        if moved[0]['history_positions'][0][0] != batch[0]['history_positions'][0][0]:
            self.assertFalse(torch.equal(fr._tensors2(moved, data['idmap'], width, 'f3_f4_recency')[1], fr._tensors2(batch, data['idmap'], width, 'f3_f4_recency')[1]))
            self.assertTrue(torch.equal(fr._tensors2(moved, data['idmap'], width, 'f3')[1], fr._tensors2(batch, data['idmap'], width, 'f3')[1]))

    def test_f4_target_week_change_leaves_feature_unchanged(self):
        guard_target_week_change_leaves_feature_unchanged()

    def test_tensors_fail_closed_on_misaligned_or_non_int_weeks(self):
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data(); rows, _ = fr._view(data, 50)
        row = copy.deepcopy(rows[0]); width = len(row['history'])
        short = {**row, 'side': row['side'][:-1]}
        with self.assertRaises(ValueError):
            fr._tensors2([short], data['idmap'], width, 'f3')
        bad = {**row, 'history_positions': row['history_positions'][:-1]}
        with self.assertRaises(ValueError):
            fr._tensors2([bad], data['idmap'], width, 'f3_f4_recency')
        fractional = {**row, 'history_positions': [[w + 0.5, o] for w, o in row['history_positions']]}
        with self.assertRaises(ValueError):
            fr._tensors2([fractional], data['idmap'], width, 'f3_f4_recency')

    def test_each_variant_trains_and_verifies(self):
        from vkgr.retrieval_feat02 import FEATURES, bucket_spec_sha256, parameter_report
        for feature in FEATURES:
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                fr, data, config, value = self._run(tmp, feature=feature)
                run = Path(tmp) / 'run'
                self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
                self.assertEqual((value['schema'], value['kind'], value['status']), ('p-feat02-train-manifest-v1', 'p_feat02_train', 'complete'))
                binding = value['binding']
                self.assertEqual((binding['task'], binding['feature'], binding['side_version']), ('P-FEAT02', feature, 'HIST_V2_SIDE_V1'))
                self.assertEqual(binding['bucket_spec_sha256'], bucket_spec_sha256())
                self.assertEqual(binding['param_report'], parameter_report(fr._model(data, config, 'cpu')))
                self.assertEqual(('weeks_sha256' in binding), feature == 'f3_f4_recency')
                self.assertEqual(('ctx_sha256' in binding), feature == 'f3_f5_scene')
                self.assertEqual(('attr_sha256' in binding), feature == 'f3_f6_author_content')
                if feature == 'f3_f6_author_content':
                    self.assertIn('item_attr_coverage', value['data_diagnostics'])
                    moved = data['author_index'].clone(); k = int((moved >= 2).nonzero()[0])
                    moved[k] = 2 if int(moved[k]) != 2 else 3
                    with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                        fr.verify_run(run / 'manifest.json', value['manifest_sha256'], {**data, 'author_index': moved}, config)
                    with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                        fr.verify_run(run / 'manifest.json', value['manifest_sha256'], {**data, 'author_vocab': data['author_vocab'] + [10 ** 6]}, config)
                    with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                        fr.verify_run(run / 'manifest.json', value['manifest_sha256'], {**data, 'content': data['content'].flip(1)}, config)
                    with self.assertRaises(ValueError):
                        fr.verify_run(run / 'manifest.json', value['manifest_sha256'], {k: v for k, v in data.items() if k != 'author_index'}, config)
                if feature == 'f3_f5_scene':
                    ctx = {**data, 'rows': [{**r, 'ctx': [[(place + 1) % 256, platform] for place, platform in r['ctx']]} for r in data['rows']]}
                    with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                        fr.verify_run(run / 'manifest.json', value['manifest_sha256'], ctx, config)
                    queries_ctx = {**data, 'queries': [{**q, 'ctx': [[place, (platform + 1) % 256] for place, platform in q['ctx']]} for q in data['queries']]}
                    with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                        fr.verify_run(run / 'manifest.json', value['manifest_sha256'], queries_ctx, config)
                self.assertEqual(fr.verify_run(run / 'manifest.json', value['manifest_sha256'], data, config)['status'], 'verified')
                with self.assertRaises(ValueError):
                    fr.verify_run(run / 'manifest.json', value['manifest_sha256'], data, {**config, 'feature': 'f3' if feature != 'f3' else 'f3_masked'})
                side = {**data, 'rows': [{**r, 'side': [[m ^ 1, t, d] for m, t, d in r['side']]} for r in data['rows']]}
                with self.assertRaises(ValueError):
                    fr.verify_run(run / 'manifest.json', value['manifest_sha256'], side, config)
                weeks = {**data, 'rows': [{**r, 'history_positions': [[(w + 1) % 25 if k == len(r['history_positions']) - 1 else w, o] for k, (w, o) in enumerate(r['history_positions'])]}
                                          for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    fr.verify_run(run / 'manifest.json', value['manifest_sha256'], weeks, config)

    def test_f4_binding_pins_weeks_independently_of_rows(self):
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data(); rows, queries = fr._view(data, 50)
        a = fr._model(data, small_config(feature='f3_f4_recency'), 'cpu')
        base = fr.binding_for2(data, small_config(feature='f3_f4_recency'), rows, queries, a, 'cpu', None)
        moved = copy.deepcopy(queries); moved[0]['history_positions'][0] = [moved[0]['history_positions'][0][0], 99]
        again = fr.binding_for2(data, small_config(feature='f3_f4_recency'), rows, moved, a, 'cpu', None)
        self.assertEqual(base['weeks_sha256'], again['weeks_sha256'])      # ordinal inside the week is not a model input
        moved[0]['history_positions'][0] = [(moved[0]['history_positions'][0][0] + 1) % 25, 99]
        self.assertNotEqual(base['weeks_sha256'], fr.binding_for2(data, small_config(feature='f3_f4_recency'), rows, moved, a, 'cpu', None)['weeks_sha256'])

    def test_diagnostics_are_recorded_and_recomputed_by_verify(self):
        from vkgr.data_audit import file_sha256
        from vkgr.retrieval_ablation_run import strict_json
        with tempfile.TemporaryDirectory() as tmp:
            fr, data, config, value = self._run(tmp, feature='f3_masked')
            run = Path(tmp) / 'run'; _, queries = fr._view(data, 50)
            self.assertIn('side_visibility', value['data_diagnostics'])
            for evaluation, merged in zip(value['evaluations'], fr.full_evaluations(run, value)):
                diag = evaluation['diagnostics']; predictions = merged['result']['predictions']
                for key in ('zero_user_vectors', 'zero_user_vector_fraction', 'top1_is_last_history_fraction', 'history_support_slices', 'top50_unsupported_share', 'item_norm_stats'):
                    self.assertIn(key, diag)
                self.assertEqual(diag['top50_unsupported_share'], fr.top50_unsupported_share(predictions, data['support']))
                self.assertEqual(evaluation['result']['support_metrics'], fr.support_slices(predictions, queries, data['support'], data['catalog']))
                self.assertEqual(set(diag['item_norm_stats']), {'supported', 'unsupported'})
                self.assertEqual(diag['item_norm_stats']['supported']['count'] + diag['item_norm_stats']['unsupported']['count'], len(data['catalog']))
            fr.verify_run(run / 'manifest.json', value['manifest_sha256'], data, config)
            best = value['best']['epoch'] - 1
            for path, pattern in ((lambda m: m['evaluations'][0]['result']['support_metrics']['supported'].__setitem__('count', 999), 'support'),
                                  (lambda m: m['evaluations'][0]['diagnostics'].__setitem__('top50_unsupported_share', 0.123456), 'top50'),
                                  (lambda m: m['evaluations'][best]['diagnostics']['item_norm_stats']['supported'].__setitem__('mean', 9.0), 'diagnostics')):
                forged = strict_json((run / 'manifest.json').read_text()); path(forged)
                (run / 'forged.json').write_text(json.dumps(forged))
                with self.assertRaisesRegex(ValueError, pattern):
                    fr.verify_run(run / 'forged.json', file_sha256(run / 'forged.json'), data, config)

    def test_diagnostic_helpers(self):
        import numpy as np
        from vkgr import retrieval_feat02_run as fr
        support = {1, 2}
        self.assertEqual(fr.top50_unsupported_share([[1, 3], [3, 4, 2, 1]], support), (0.5 + 0.5) / 2)
        self.assertEqual(fr.top50_unsupported_share([[1, 2]], support), 0.0)
        self.assertEqual(fr.top50_unsupported_share([[1, 2] * 25 + [9] * 10], support), 0.0)    # only the first 50 count
        with self.assertRaises(ValueError):
            fr.top50_unsupported_share([[]], support)
        stats = fr.item_norm_stats(np.array([1., 3., 2., 8.]), [1, 2, 3, 4], support)
        self.assertEqual(stats, {'supported': {'count': 2, 'mean': 2.0, 'p50': 2.0}, 'unsupported': {'count': 2, 'mean': 5.0, 'p50': 5.0}})
        self.assertEqual(fr.item_norm_stats(np.array([1.]), [1], support)['unsupported'], {'count': 0, 'mean': None, 'p50': None})

    def test_export_returns_raw_item_norms_before_normalization(self):
        import numpy as np
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data(); _, queries = fr._view(data, 50)
        model = fr._model(data, small_config(feature='f3_f4_recency'), 'cpu')
        users, items, norms = fr.export_vectors_side2(model, data['catalog'], queries, data['idmap'], 16, 50, 'f3_f4_recency')
        self.assertEqual(norms.shape, (len(data['catalog']),))
        raw = model.items(torch.tensor([data['idmap'][i] for i in data['catalog']])).detach().numpy()
        self.assertTrue(np.allclose(norms, np.linalg.norm(raw.astype(np.float64), axis=1)))
        self.assertTrue(np.allclose(np.linalg.norm(items[norms > 0], axis=1), 1.0, atol=1e-5))

    def test_synthetic_data_carries_aligned_ctx_with_unseen_values(self):
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data()
        values = set()
        for r in data['rows'] + data['queries']:
            self.assertEqual(len(r['ctx']), len(r['history']))
            values.update(v for pair in r['ctx'] for v in pair)
        self.assertTrue(any(v >= 24 for v in values), 'synthetic ctx must contain values outside the train tables')
        self.assertTrue(all(0 <= v <= 255 for v in values))

    def test_tensors2_fills_scene_columns_for_f5_only_and_fails_closed(self):
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data(); rows, _ = fr._view(data, 50)
        batch = rows[:20]; width = max(len(r['history']) for r in batch)
        histories, tensor = fr._tensors2(batch, data['idmap'], width, 'f3_f5_scene')
        for j, r in enumerate(batch):
            n = len(r['history'])
            self.assertEqual(tensor[j, :n, 4].tolist(), [c[0] for c in r['ctx']]); self.assertEqual(tensor[j, :n, 5].tolist(), [c[1] for c in r['ctx']])
            self.assertEqual(tensor[j, :n, 3].tolist(), [0] * n)                            # week column is the recency variant's
            self.assertTrue(bool(tensor[j, n:].eq(0).all()))
        for feature in ('f3', 'f3_masked', 'f3_f4_recency'):
            other = fr._tensors2(batch, data['idmap'], width, feature)[1]
            self.assertTrue(bool(other[..., 4:].eq(0).all()), feature)
        no_ctx = [{k: v for k, v in r.items() if k != 'ctx'} for r in batch]
        fr._tensors2(no_ctx, data['idmap'], width, 'f3')                                    # other variants never need ctx
        with self.assertRaises(ValueError):
            fr._tensors2(no_ctx, data['idmap'], width, 'f3_f5_scene')                      # missing ctx pin / data
        with self.assertRaises(ValueError):
            fr._tensors2([{**batch[0], 'ctx': batch[0]['ctx'][:-1]}], data['idmap'], width, 'f3_f5_scene')
        with self.assertRaises(ValueError):
            fr._tensors2([{**batch[0], 'ctx': [[1.5, 2]] * len(batch[0]['ctx'])}], data['idmap'], width, 'f3_f5_scene')
        with self.assertRaises(ValueError):
            fr._tensors2([{**batch[0], 'ctx': [[1, 2, 3]] * len(batch[0]['ctx'])}], data['idmap'], width, 'f3_f5_scene')

    def test_view_cuts_ctx_to_the_history_length(self):
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data()
        longest = max(data['rows'], key=lambda r: len(r['history']))
        rows, _ = fr._view({**data, 'rows': [longest], 'queries': []}, 3)
        self.assertEqual(rows[0]['ctx'], longest['ctx'][-3:])
        self.assertEqual(len(rows[0]['ctx']), len(rows[0]['history']))

    def test_real_ctx_path_prepare_verify_apply(self):
        """Real CLI path (prepare-ctx / verify-ctx / apply_feat02) on pinned synthetic files with mocked real_inputs."""
        import argparse
        from vkgr import retrieval_feat02_run as fr
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            partitions, examples, queries, catalog, durations, h2, side, _ = prepared_full_world(tmp / 'w', seed=9)
            import shutil
            shutil.rmtree(tmp / 'w' / 'ctx')                                                     # prepare-ctx must create it from raw
            spec = {'schema': 'p-feat02-inputs-v1', 'prepared': str(tmp / 'p'), 'holdout': str(tmp / 'h'), 'raw_root': str(tmp / 'raw'),
                    'history_manifest': str(tmp / 'w' / 'h2' / 'manifest.json'), 'history_sha256': h2['manifest_sha256'],
                    'side_manifest': str(tmp / 'w' / 'side' / 'manifest.json'), 'side_sha256': side['manifest_sha256'], 'items_metadata_sha256': 'c' * 64,
                    'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}
            fr.validate_inputs(spec)
            with self.assertRaises(ValueError):
                fr.validate_inputs(spec, 'f3_f5_scene')                                            # F5 needs the ctx pin
            data = {'rows': examples, 'queries': queries, 'catalog': catalog, 'synthetic': False}
            state = {'partitions': partitions}
            fake = lambda s, with_partitions=False: (data, (lambda: iter(state['partitions'])) if with_partitions else None, {'prepared_sha': 'x'})
            plain, plain_binding = fr.apply_feat02(data, spec, 50)
            self.assertNotIn('ctx', plain['rows'][0]); self.assertNotIn('ctx_manifest_sha256', plain_binding)     # no pin -> no ctx loaded
            with mock.patch.object(fr.b1, 'real_inputs', fake):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(fr.main(['prepare-ctx', '--inputs', str(self._write(tmp / 'inputs.json', spec)), '--inputs-sha', file_sha256(tmp / 'inputs.json'),
                                              '--output', str(tmp / 'ctx')]), 0)
                self.assertEqual(json.loads(out.getvalue())['status'], 'ctx_prepared')
                ctx_sha = file_sha256(tmp / 'ctx' / 'manifest.json')
                manifest = json.loads((tmp / 'ctx' / 'manifest.json').read_text())
                self.assertEqual((manifest['ctx_version'], manifest['fields'], manifest['history_manifest_sha256']), ('HIST_V2_CTX_V1', ['place', 'platform'], h2['manifest_sha256']))
                self.assertEqual(manifest['binding']['inputs_sha256'], file_sha256(tmp / 'inputs.json'))
                args = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--manifest', str(tmp / 'ctx' / 'manifest.json'), '--manifest-sha', ctx_sha]
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(fr.main(['verify-ctx', *args]), 0)
                self.assertEqual(json.loads(out.getvalue())['status'], 'ctx_verified')
                spec_c = {**spec, 'ctx_manifest': str(tmp / 'ctx' / 'manifest.json'), 'ctx_sha256': ctx_sha}
                fr.validate_inputs(spec_c, 'f3_f5_scene')
                table = {(w, k): r for w, rows in partitions for k, r in enumerate(rows)}
                for length in (50, 20):
                    applied, binding = fr.apply_feat02(data, spec_c, length)
                    self.assertEqual(binding['ctx_manifest_sha256'], ctx_sha)
                    base, base_binding = fr.apply_feat02(data, spec, length)
                    self.assertEqual({k: v for k, v in binding.items() if k != 'ctx_manifest_sha256'}, base_binding)
                    for r, b in zip(applied['rows'] + applied['queries'], base['rows'] + base['queries']):
                        self.assertEqual({k: v for k, v in r.items() if k != 'ctx'}, b)
                        self.assertEqual(len(r['ctx']), len(r['history']))
                        for pos, pair in zip(r['history_positions'], r['ctx']):
                            self.assertEqual(pair, [table[tuple(pos)]['place'], table[tuple(pos)]['platform']])
                # raw drift after prepare -> verify-ctx fails closed
                mutated = copy.deepcopy(partitions)
                pos = applied['rows'][-1]['history_positions'][-1]
                mutated[pos[0]][1][pos[1]]['place'] = (mutated[pos[0]][1][pos[1]]['place'] + 1) % 256
                state['partitions'] = mutated
                with self.assertRaisesRegex(ValueError, 'raw ctx replay mismatch'):
                    fr._ctx_command(argparse.Namespace(command='verify-ctx', inputs_sha='a' * 64, output=None, manifest=str(tmp / 'ctx' / 'manifest.json'), manifest_sha=ctx_sha), spec)
                state['partitions'] = partitions
                # wrong pins fail closed
                with self.assertRaises(ValueError):
                    fr._ctx_command(argparse.Namespace(command='verify-ctx', inputs_sha='a' * 64, output=None, manifest=str(tmp / 'ctx' / 'manifest.json'), manifest_sha='0' * 64), spec)
                with self.assertRaises(ValueError):
                    fr.apply_feat02(data, {**spec_c, 'ctx_sha256': '0' * 64}, 50)
                with self.assertRaises(ValueError):
                    fr.apply_feat02(data, {**spec_c, 'history_sha256': 'f' * 64}, 50)
                with self.assertRaises(ValueError):
                    fr._ctx_command(argparse.Namespace(command='verify-ctx', inputs_sha='a' * 64, output=None, manifest=str(tmp / 'ctx' / 'manifest.json'), manifest_sha=ctx_sha),
                                    {**spec, 'history_sha256': 'f' * 64})
                # an existing output directory is never overwritten
                with self.assertRaises(FileExistsError):
                    fr._ctx_command(argparse.Namespace(command='prepare-ctx', inputs_sha='a' * 64, output=str(tmp / 'ctx'), manifest=None, manifest_sha=None), spec)

    @staticmethod
    def _write(path, value):
        Path(path).write_text(json.dumps(value)); return path

    def test_f5_user_vectors_react_to_ctx_and_ignore_target_fields(self):
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data(); rows, _ = fr._view(data, 50)
        model = fr._model(data, small_config(feature='f3_f5_scene'), 'cpu'); model.eval()
        batch = rows[:24]; width = max(len(r['history']) for r in batch)
        base = model.users(*fr._tensors2(batch, data['idmap'], width, 'f3_f5_scene'))
        same = model.users(*fr._tensors2([flip_non_history_fields(r) for r in batch], data['idmap'], width, 'f3_f5_scene'))
        self.assertTrue(torch.equal(base, same))
        moved = [{**r, 'ctx': [[(p + 1) % 24, q] for p, q in r['ctx']]} for r in batch]
        self.assertFalse(torch.equal(base, model.users(*fr._tensors2(moved, data['idmap'], width, 'f3_f5_scene'))))

    def test_inputs_schema_requires_ctx_attr_pairs_and_required_if_used(self):
        from vkgr.retrieval_feat02_run import validate_inputs
        good = {'schema': 'p-feat02-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
                'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64, 'items_metadata_sha256': 'c' * 64,
                'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 8 * 1024**3, 'test_gate': 'closed_hash_only'}
        validate_inputs(good); validate_inputs(good, 'f3_f4_recency')
        validate_inputs({**good, 'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'd' * 64, 'attr_manifest': '/g/manifest.json', 'attr_sha256': 'e' * 64}, 'f3_f5_scene')
        for bad in ({'side_sha256': None}, {'items_metadata_sha256': 'x'}, {'schema': 'p-feat01-inputs-v1'}, {'history_manifest': None},
                    {'ctx_manifest': '/f/manifest.json'}, {'ctx_sha256': 'd' * 64}, {'attr_manifest': '/g/manifest.json'}, {'attr_sha256': 'x' * 64},
                    {'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'nothex'}):
            with self.assertRaises(ValueError, msg=bad):
                validate_inputs({**good, **bad})
        with self.assertRaises(ValueError):
            validate_inputs({k: v for k, v in good.items() if k != 'attr_sha256'})
        with self.assertRaises(ValueError):
            validate_inputs(good, 'f3_f5_scene')              # scene variant needs the ctx pin
        with self.assertRaises(ValueError):
            validate_inputs(good, 'f3_f6_author_content')     # author variant needs the attr pin

    def test_apply_feat02_reuses_apply_side_and_weeks_come_from_history_positions(self):
        from vkgr import retrieval_feat01_run as f1
        from vkgr import retrieval_feat02_run as fr
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            partitions, examples, queries, catalog, durations, h2, side = prepared_world(tmp)
            spec = {'schema': 'p-feat02-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': str(tmp / 'h2' / 'manifest.json'),
                    'history_sha256': h2['manifest_sha256'], 'side_manifest': str(tmp / 'side' / 'manifest.json'), 'side_sha256': side['manifest_sha256'],
                    'items_metadata_sha256': 'c' * 64, 'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None,
                    'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}
            fr.validate_inputs(spec)
            data = {'rows': examples, 'queries': queries, 'catalog': catalog, 'synthetic': False}
            for length in (50, 20):
                a, binding_a = f1.apply_side(data, {k: spec[k] for k in ('history_manifest', 'history_sha256', 'side_manifest', 'side_sha256')}, length)
                b, binding_b = fr.apply_feat02(data, spec, length)
                self.assertEqual((a, binding_a), (b, binding_b))
                idmap = {i: i + 1 for i in catalog}; batch = (b['rows'] + b['queries'])[:40]; width = max(len(r['history']) for r in batch)
                histories, tensor = fr._tensors2(batch, idmap, width, 'f3_f4_recency')
                for j, r in enumerate(batch):
                    self.assertEqual(tensor[j, :len(r['history']), 3].tolist(), [p[0] for p in r['history_positions']])
                    self.assertEqual(tensor[j, :len(r['history']), 1].tolist(), [s[1] for s in r['side']])
                    self.assertTrue(bool(tensor[j, len(r['history']):].eq(0).all()))
                self.assertTrue(max(int(tensor[..., 3].max()), 0) <= 24)

    def test_synthetic_data_carries_the_f6_attr_fixture(self):
        from vkgr import item_attr
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data()
        content, index, vocab = data['content'], data['author_index'], data['author_vocab']
        self.assertEqual(tuple(content.shape), (401, 64)); self.assertEqual(tuple(index.shape), (401,))
        self.assertEqual(content.dtype, torch.float32); self.assertEqual(index.dtype, torch.long)
        self.assertTrue(torch.all(content[0] == 0))
        self.assertTrue(torch.allclose(content[1:].norm(dim=1), torch.ones(400), atol=1e-5), 'non-zero L2-normalised content rows')
        self.assertEqual(int(index[0]), 0)
        self.assertEqual(vocab, sorted(set(vocab)))
        supported = [data['idmap'][i] for i in sorted(data['support'])]; unsupported = [data['idmap'][i] for i in data['catalog'] if i not in data['support']]
        self.assertTrue(all(int(index[k]) >= 2 for k in supported), 'supported items always have in-vocabulary authors')
        self.assertTrue(any(int(index[k]) == 1 for k in unsupported) and any(int(index[k]) >= 2 for k in unsupported), 'unsupported items mix OOV and in-vocabulary authors')
        self.assertEqual(vocab, item_attr.author_vocab(data['authors'], data['support']))
        self.assertEqual(index.tolist(), item_attr.author_rows([None] + [data['authors'][i] for i in data['catalog']], vocab))
        self.assertEqual(int(index.max()), len(vocab) + 1)

    def test_f6_target_slices_recomputed_by_verify(self):
        from vkgr.data_audit import file_sha256
        from vkgr.retrieval_ablation_run import strict_json
        with tempfile.TemporaryDirectory() as tmp:
            fr, data, config, value = self._run(tmp, feature='f3_f6_author_content')
            run = Path(tmp) / 'run'; _, queries = fr._view(data, 50)
            best = value['best']['epoch'] - 1
            slices = value['evaluations'][0]['result']['support_metrics']
            self.assertEqual(set(slices), {'supported', 'unsupported', 'oov'})
            self.assertEqual(sum(slices[k]['count'] for k in slices), len(queries))
            if slices['oov']['count']:
                self.assertEqual(slices['oov']['recall@50'], 0.0, 'OOV targets are not in the frozen catalog: zero by construction')
            coverage = value['data_diagnostics']['item_attr_coverage']
            self.assertEqual((coverage['catalog_items'], coverage['supported_items'], coverage['unsupported_items'], coverage['zero_content_items']), (400, 300, 100, 0))
            self.assertTrue(0 < coverage['unsupported_with_vocab_author_fraction'] < 1)
            for evaluation, merged in zip(value['evaluations'], fr.full_evaluations(run, value)):
                predictions = merged['result']['predictions']
                self.assertEqual(evaluation['result']['support_metrics'], fr.support_slices(predictions, queries, data['support'], data['catalog']))
                self.assertEqual(evaluation['diagnostics']['top50_unsupported_share'], fr.top50_unsupported_share(predictions, data['support']))
            fr.verify_run(run / 'manifest.json', value['manifest_sha256'], data, config)
            for path, pattern in ((lambda m: m['evaluations'][0]['result']['support_metrics']['unsupported'].__setitem__('count', 999), 'support'),
                                  (lambda m: m['evaluations'][0]['result']['support_metrics']['supported'].__setitem__('recall@50', 0.987654), 'support'),
                                  (lambda m: m['evaluations'][0]['diagnostics'].__setitem__('top50_unsupported_share', 0.123456), 'top50'),
                                  (lambda m: m['evaluations'][best]['diagnostics']['item_norm_stats']['unsupported'].__setitem__('mean', 9.0), 'diagnostics'),
                                  (lambda m: m['data_diagnostics']['item_attr_coverage'].__setitem__('zero_content_items', 7), 'coverage')):
                forged = strict_json((run / 'manifest.json').read_text()); path(forged)
                (run / 'forged.json').write_text(json.dumps(forged))
                with self.assertRaisesRegex(ValueError, pattern):
                    fr.verify_run(run / 'forged.json', file_sha256(run / 'forged.json'), data, config)

    def test_f6_catalog_has_no_zero_vectors_while_f3_has_one_per_unsupported_item(self):
        """Measured on the synthetic catalog (400 items, 100 unsupported), exported through the runner exactly as evaluation does."""
        import numpy as np
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data(); _, queries = fr._view(data, 50); counts = {}
        for feature in ('f3', 'f3_f6_author_content'):
            model = fr._model(data, small_config(feature=feature), 'cpu')
            _, _, norms = fr.export_vectors_side2(model, data['catalog'], queries, data['idmap'], 16, 50, feature)
            counts[feature] = int(np.sum(norms == 0))
        self.assertEqual(counts, {'f3': 100, 'f3_f6_author_content': 0})

    def test_f6_requires_author_index_and_content_comes_only_from_the_prepared_catalog(self):
        import inspect
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data()
        for key in ('author_index', 'author_vocab'):
            with self.assertRaises(ValueError, msg=key):
                fr._model({k: v for k, v in data.items() if k != key}, small_config(feature='f3_f6_author_content'), 'cpu')
        model = fr._model(data, small_config(feature='f3_f6_author_content'), 'cpu')
        self.assertTrue(torch.equal(model.content, data['content']), 'the content buffer is the prepared catalog embedding, nothing else')
        self.assertTrue(torch.equal(model.author_index, data['author_index']))
        self.assertEqual(model.author_table.weight.shape[0], len(data['author_vocab']) + 2)
        for source in (inspect.getsource(fr._tensors2), inspect.getsource(fr.export_vectors_side2)):
            for banned in ('content', 'author'):
                self.assertNotIn(banned, source)                 # per-sample tensors never carry attributes: only the model buffers do
        self.assertTrue(torch.equal(fr._model(data, small_config(feature='f3'), 'cpu').embedding.weight, model.embedding.weight))

    def test_f6_user_vectors_react_to_history_items_and_ignore_target_fields(self):
        from vkgr import retrieval_feat02_run as fr
        data = fr.synthetic_data(); rows, _ = fr._view(data, 50)
        model = fr._model(data, small_config(feature='f3_f6_author_content'), 'cpu'); model.eval()
        batch = rows[:24]; width = max(len(r['history']) for r in batch)
        base = model.users(*fr._tensors2(batch, data['idmap'], width, 'f3_f6_author_content'))
        same = model.users(*fr._tensors2([flip_non_history_fields(r) for r in batch], data['idmap'], width, 'f3_f6_author_content'))
        self.assertTrue(torch.equal(base, same))
        moved = [{**r, 'history': [(i + 1) % 400 for i in r['history']]} for r in batch]
        self.assertFalse(torch.equal(base, model.users(*fr._tensors2(moved, data['idmap'], width, 'f3_f6_author_content'))))

    def test_real_attr_path_prepare_verify_apply(self):
        """Real CLI path (prepare-attr / verify-attr / apply_feat02) on pinned synthetic files with mocked real_inputs."""
        import argparse
        from vkgr import item_attr
        from vkgr import retrieval_feat02_run as fr
        from vkgr.data_audit import file_sha256
        from test_feat02_data import write_metadata
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            partitions, examples, queries, catalog, durations, h2, side, _ = prepared_full_world(tmp / 'w', seed=9)
            authors = {i: 1000 + k % 7 if (k % 2 == 0 or k % 3 == 0) else 3000 + k for k, i in enumerate(catalog)}
            support = set(catalog[::2])
            (tmp / 'raw' / 'metadata').mkdir(parents=True)
            metadata = write_metadata(tmp / 'raw' / 'metadata' / 'items_metadata.parquet', authors)
            vocab = item_attr.author_vocab(authors, support); size, sha = len(vocab), item_attr.vocab_digest(vocab)
            self.assertEqual(size, 7); self.assertTrue(any(authors[i] not in vocab for i in catalog))
            spec = {'schema': 'p-feat02-inputs-v1', 'prepared': str(tmp / 'p'), 'holdout': str(tmp / 'h'), 'raw_root': str(tmp / 'raw'),
                    'history_manifest': str(tmp / 'w' / 'h2' / 'manifest.json'), 'history_sha256': h2['manifest_sha256'],
                    'side_manifest': str(tmp / 'w' / 'side' / 'manifest.json'), 'side_sha256': side['manifest_sha256'], 'items_metadata_sha256': file_sha256(metadata),
                    'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}
            fr.validate_inputs(spec)
            with self.assertRaises(ValueError):
                fr.validate_inputs(spec, 'f3_f6_author_content')                                 # F6 needs the attr pin
            data = {'rows': examples, 'queries': queries, 'catalog': catalog, 'synthetic': False, 'support': support,
                    'idmap': {i: k + 1 for k, i in enumerate(catalog)}, 'content': torch.zeros(len(catalog) + 1, 64)}
            fake = lambda s, with_partitions=False: (data, None, {'prepared_sha': 'x'})
            plain, plain_binding = fr.apply_feat02(data, spec, 50)
            self.assertNotIn('author_index', plain); self.assertNotIn('attr_manifest_sha256', plain_binding)      # no pin -> nothing loaded
            ns = lambda command, **kw: argparse.Namespace(command=command, inputs_sha='a' * 64, output=kw.get('output'), manifest=kw.get('manifest'), manifest_sha=kw.get('manifest_sha'))
            with mock.patch.object(fr.b1, 'real_inputs', fake), mock.patch.object(fr, 'AUTHOR_VOCAB_SIZE', size), mock.patch.object(fr, 'AUTHOR_VOCAB_SHA256', sha):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(fr.main(['prepare-attr', '--inputs', str(self._write(tmp / 'inputs.json', spec)), '--inputs-sha', file_sha256(tmp / 'inputs.json'),
                                              '--output', str(tmp / 'attr')]), 0)
                self.assertEqual(json.loads(out.getvalue())['status'], 'attr_prepared')
                attr_sha = file_sha256(tmp / 'attr' / 'manifest.json')
                manifest = json.loads((tmp / 'attr' / 'manifest.json').read_text())
                self.assertEqual((manifest['attr_version'], manifest['items_metadata_sha256'], manifest['vocab_size'], manifest['vocab_sha256']), ('ITEM_ATTR_V1', file_sha256(metadata), size, sha))
                self.assertEqual(manifest['binding']['inputs_sha256'], file_sha256(tmp / 'inputs.json'))
                args = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--manifest', str(tmp / 'attr' / 'manifest.json'), '--manifest-sha', attr_sha]
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(fr.main(['verify-attr', *args]), 0)
                self.assertEqual(json.loads(out.getvalue())['status'], 'attr_verified')
                spec_a = {**spec, 'attr_manifest': str(tmp / 'attr' / 'manifest.json'), 'attr_sha256': attr_sha}
                fr.validate_inputs(spec_a, 'f3_f6_author_content')
                for length in (50, 20):
                    applied, binding = fr.apply_feat02(data, spec_a, length)
                    base, base_binding = fr.apply_feat02(data, spec, length)
                    self.assertEqual(binding['attr_manifest_sha256'], attr_sha)
                    self.assertEqual({k: v for k, v in binding.items() if k != 'attr_manifest_sha256'}, base_binding)
                    self.assertEqual(applied['author_vocab'], vocab)
                    self.assertEqual(applied['author_index'].tolist(), item_attr.author_rows([None] + [authors[i] for i in catalog], vocab))
                    self.assertEqual({k: v for k, v in applied.items() if k not in ('author_index', 'author_vocab')}, base)
                # a changed constant (size or sha) makes the sidecar unusable and prepare-attr refuse to write
                for name, wrong in (('AUTHOR_VOCAB_SIZE', size + 1), ('AUTHOR_VOCAB_SHA256', '0' * 64)):
                    with mock.patch.object(fr, name, wrong):
                        with self.assertRaises(ValueError, msg=name):
                            fr.apply_feat02(data, spec_a, 50)
                        with self.assertRaises(ValueError, msg=name):
                            fr._attr_command(ns('prepare-attr', output=str(tmp / f'attr-{name}')), spec)
                        self.assertFalse((tmp / f'attr-{name}').exists())
                        with self.assertRaises(ValueError, msg=name):
                            fr._attr_command(ns('verify-attr', manifest=str(tmp / 'attr' / 'manifest.json'), manifest_sha=attr_sha), spec)
                # metadata pin wrong
                wrong_pin = {**spec, 'items_metadata_sha256': 'f' * 64}
                with self.assertRaises(ValueError):
                    fr._attr_command(ns('prepare-attr', output=str(tmp / 'attr-pin')), wrong_pin)
                with self.assertRaises(ValueError):
                    fr._attr_command(ns('verify-attr', manifest=str(tmp / 'attr' / 'manifest.json'), manifest_sha=attr_sha), wrong_pin)
                with self.assertRaises(ValueError):
                    fr.apply_feat02(data, {**spec_a, 'items_metadata_sha256': 'f' * 64}, 50)         # sidecar was prepared from another metadata file
                # wrong attr pins
                with self.assertRaises(ValueError):
                    fr.apply_feat02(data, {**spec_a, 'attr_sha256': '0' * 64}, 50)
                with self.assertRaises(ValueError):
                    fr._attr_command(ns('verify-attr', manifest=str(tmp / 'attr' / 'manifest.json'), manifest_sha='0' * 64), spec)
                # raw drift: the metadata file changes (new pin) -> the sidecar no longer binds it; with the old pin the file hash fails
                drift = dict(authors); drift[catalog[0]] = 7777
                write_metadata(tmp / 'raw' / 'metadata' / 'items_metadata.parquet.new', drift)
                (tmp / 'raw' / 'metadata' / 'items_metadata.parquet').write_bytes((tmp / 'raw' / 'metadata' / 'items_metadata.parquet.new').read_bytes())
                with self.assertRaises(ValueError):
                    fr._attr_command(ns('verify-attr', manifest=str(tmp / 'attr' / 'manifest.json'), manifest_sha=attr_sha), spec)                  # stale metadata pin
                drifted = {**spec, 'items_metadata_sha256': file_sha256(tmp / 'raw' / 'metadata' / 'items_metadata.parquet')}
                with self.assertRaises(ValueError):
                    fr._attr_command(ns('verify-attr', manifest=str(tmp / 'attr' / 'manifest.json'), manifest_sha=attr_sha), drifted)                # sidecar bound to the old file
                write_metadata(tmp / 'raw' / 'metadata' / 'items_metadata.parquet.new', authors)
                (tmp / 'raw' / 'metadata' / 'items_metadata.parquet').write_bytes((tmp / 'raw' / 'metadata' / 'items_metadata.parquet.new').read_bytes())
                self.assertEqual(file_sha256(tmp / 'raw' / 'metadata' / 'items_metadata.parquet'), spec['items_metadata_sha256'])
                # replay mismatch with the pins intact: the raw author table read back differs from the stored sidecar
                real = item_attr.load_authors
                with mock.patch.object(item_attr, 'load_authors', lambda path, cat: {**real(path, cat), catalog[1]: 9999}):
                    with self.assertRaisesRegex(ValueError, 'raw attr replay mismatch'):
                        fr._attr_command(ns('verify-attr', manifest=str(tmp / 'attr' / 'manifest.json'), manifest_sha=attr_sha), spec)
                fr._attr_command(ns('verify-attr', manifest=str(tmp / 'attr' / 'manifest.json'), manifest_sha=attr_sha), spec)
                with self.assertRaises(FileExistsError):
                    fr._attr_command(ns('prepare-attr', output=str(tmp / 'attr')), spec)             # an existing output directory is never overwritten

    def test_train_cli_requires_the_attr_pin_for_f6(self):
        from vkgr import retrieval_feat02_run as fr
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec = {'schema': 'p-feat02-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
                    'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64, 'items_metadata_sha256': 'c' * 64,
                    'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}
            (tmp / 'inputs.json').write_text(json.dumps(spec)); (tmp / 'real.json').write_text(json.dumps(f2_config(feature='f3_f6_author_content')))
            argv = ['train', '--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'),
                    '--config-sha', file_sha256(tmp / 'real.json'), '--output', str(tmp / 'x')]
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(fr.main(argv), 2)
            self.assertIn('pinned sidecar required', err.getvalue())
            self.assertFalse((tmp / 'x').exists())

    def test_cli_synthetic_train_and_real_path_wiring(self):
        from vkgr import retrieval_feat02_run as fr
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'config.json').write_text(json.dumps(small_config(feature='f3_f4_recency')))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(fr.main(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run')]), 0)
            self.assertEqual(json.loads(out.getvalue())['status'], 'complete')
            self.assertTrue((tmp / 'run' / 'manifest.json').exists())
            spec = {'schema': 'p-feat02-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
                    'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64, 'items_metadata_sha256': 'c' * 64,
                    'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}
            (tmp / 'inputs.json').write_text(json.dumps(spec)); (tmp / 'real.json').write_text(json.dumps(f2_config(feature='f3_masked')))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            fake_data = {'rows': [], 'queries': [], 'catalog': []}
            with mock.patch('torch.cuda.is_available', return_value=True), \
                    mock.patch.object(fr.b1, 'resource_guard', lambda *a, **k: (lambda: None)), \
                    mock.patch.object(fr.b1, 'real_inputs', lambda s: (fake_data, None, {'prepared_sha': 'x'})), \
                    mock.patch.object(fr, 'apply_feat02', lambda d, s, n: (d, {'history_manifest_sha256': 'h'})), \
                    mock.patch.object(fr, 'run_session', lambda *a, **k: {'status': 'complete', 'manifest_sha256': 'm', 'selection': {'best_epoch': 3}}), \
                    mock.patch.object(fr, 'verify_run', lambda *a, **k: {'status': 'verified'}):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(fr.main(['train', *common, '--output', str(tmp / 'real-run')]), 0)
                self.assertEqual(json.loads(out.getvalue())['best_epoch'], 3)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(fr.main(['verify', *common, '--manifest', str(tmp / 'real-run' / 'manifest.json'), '--manifest-sha', 'a' * 64]), 0)
                self.assertEqual(json.loads(out.getvalue())['status'], 'verified')
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(fr.main(['train', *common[:2], '--inputs-sha', '0' * 64, '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json'), '--output', str(tmp / 'x')]), 2)
            self.assertIn('P-FEAT02', err.getvalue())

    def test_prepare_attr_output_does_not_depend_on_validation_queries(self):
        """prepare-attr loads the queries through real_inputs but must never use them: swapping them changes no output byte."""
        import argparse
        from vkgr import item_attr
        from vkgr import retrieval_feat02_run as fr
        from vkgr.data_audit import file_sha256
        from test_feat02_data import write_metadata
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            partitions, examples, queries, catalog, durations, h2, side, _ = prepared_full_world(tmp / 'w', seed=9)
            authors = {i: 1000 + k % 7 if (k % 2 == 0 or k % 3 == 0) else 3000 + k for k, i in enumerate(catalog)}
            support = set(catalog[::2])
            (tmp / 'raw' / 'metadata').mkdir(parents=True)
            metadata = write_metadata(tmp / 'raw' / 'metadata' / 'items_metadata.parquet', authors)
            vocab = item_attr.author_vocab(authors, support)
            spec = {'schema': 'p-feat02-inputs-v1', 'prepared': 'p', 'holdout': 'h', 'raw_root': str(tmp / 'raw'),
                    'history_manifest': 'm', 'history_sha256': 'a' * 64, 'side_manifest': 's', 'side_sha256': 'b' * 64,
                    'items_metadata_sha256': file_sha256(metadata), 'ctx_manifest': None, 'ctx_sha256': None,
                    'attr_manifest': None, 'attr_sha256': None, 'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}
            base = {'rows': examples, 'catalog': catalog, 'synthetic': False, 'support': support}
            variants = {'real': queries, 'empty': [], 'scrambled': [{**q, 'target': catalog[-1], 'history': catalog[:3]} for q in queries]}
            outputs = {}
            with mock.patch.object(fr, 'AUTHOR_VOCAB_SIZE', len(vocab)), mock.patch.object(fr, 'AUTHOR_VOCAB_SHA256', item_attr.vocab_digest(vocab)):
                for name, qs in variants.items():
                    with mock.patch.object(fr.b1, 'real_inputs', lambda s, with_partitions=False, qs=qs: ({**base, 'queries': qs}, None, {'prepared_sha': 'x'})):
                        ns = argparse.Namespace(command='prepare-attr', inputs_sha='a' * 64, output=str(tmp / name), manifest=None, manifest_sha=None)
                        with contextlib.redirect_stdout(io.StringIO()):
                            fr._attr_command(ns, spec)
                    outputs[name] = {f: (tmp / name / f).read_bytes() for f in ('authors.json', 'manifest.json')}
            self.assertEqual(outputs['real'], outputs['empty'])
            self.assertEqual(outputs['real'], outputs['scrambled'])

    CONFIG_STEMS = {'p-feat02-f3-masked-s42': 'f3_masked', 'p-feat02-f4-recency-s42': 'f3_f4_recency',
                    'p-feat02-f5-scene-s42': 'f3_f5_scene', 'p-feat02-f6-author-content-s42': 'f3_f6_author_content'}

    def test_committed_feat02_configs_are_registered(self):
        from vkgr.retrieval_feat02_run import validate_config
        root = Path(__file__).resolve().parent.parent / 'configs' / 'feat02'
        files = sorted(root.glob('p-*.json'))
        self.assertEqual({f.stem for f in files}, set(self.CONFIG_STEMS))
        for path in files:
            value = json.loads(path.read_text())
            validate_config(value)
            self.assertEqual(value['feature'], self.CONFIG_STEMS[path.stem], path.stem)
            self.assertEqual((value['seed'], value['history_len'], value['logq'], value['extra_uniform']), (42, 50, True, 0), path.stem)
            self.assertEqual(path.read_text(), json.dumps(value, indent=1, sort_keys=True) + '\n', path.stem)

    def test_feat02_config_files_differ_from_f3_only_in_schema_task_feature(self):
        root = Path(__file__).resolve().parent.parent / 'configs'
        anchor = json.loads((root / 'feat01' / 'p-feat01-f3-watch-buckets-s42.json').read_text())
        files = sorted((root / 'feat02').glob('p-*.json'))
        self.assertEqual(len(files), 4)
        for path in files:
            value = json.loads(path.read_text())
            self.assertEqual(set(value), set(anchor), path.stem)
            self.assertEqual({k for k in anchor if anchor[k] != value[k]}, {'schema', 'task_id', 'feature'}, path.stem)
            self.assertEqual((value['schema'], value['task_id']), ('p-feat02-run-v1', 'FEAT-02'))


if __name__ == '__main__':
    unittest.main()
