"""FEAT-01 (P line): history side sidecar (HIST_V2_SIDE_V1) + F1/F2/F3 user-tower switches."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_history_v2 import synthetic_world  # noqa: E402


def side_world(seed=7, rows_per_week=200):
    """synthetic_world rows enriched with the six feedback bits and timespent; per-item durations."""
    import random
    partitions, examples, queries, catalog = synthetic_world(seed, rows_per_week=rows_per_week)
    rng = random.Random(seed + 100)
    for _, rows in partitions:
        for row in rows:
            for name in ('share', 'bookmark', 'open_comments', 'click_on_author', 'dislike'):
                row[name] = rng.random() < 0.2
            row['timespent'] = rng.choice([1, 2, 5, 9, 12, 40, 70, 150, 255])
    durations = {i: rng.choice([5, 9, 25, 60, 180]) for i in catalog}
    return partitions, examples, queries, catalog, durations


def prepared_world(tmp, seed=7):
    from vkgr import history_v2, history_side
    partitions, examples, queries, catalog, durations = side_world(seed)
    h2 = history_v2.write_prepared(Path(tmp) / 'h2', partitions, examples, queries, catalog, binding={'synthetic': True})
    side = history_side.write_side(Path(tmp) / 'side', Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], partitions, durations, binding={'synthetic': True})
    return partitions, examples, queries, catalog, durations, h2, side


class SideSidecarTests(unittest.TestCase):
    def test_side_is_aligned_with_history_events(self):
        from vkgr import history_side, history_v2
        with tempfile.TemporaryDirectory() as tmp:
            partitions, examples, queries, catalog, durations, h2, side = prepared_world(tmp)
            table = {(w, k): r for w, rows in partitions for k, r in enumerate(rows)}
            for length in (100, 50, 20):
                hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], length)
                loaded = history_side.load_side(Path(tmp) / 'side' / 'manifest.json', side['manifest_sha256'], h2['manifest_sha256'], length, hist)
                for split in ('train', 'validation'):
                    self.assertEqual(len(loaded[split]), len(hist[split]))
                    for row, s in zip(hist[split], loaded[split]):
                        self.assertEqual(len(s), len(row['history']))
                        for item, pos, (mask, ts, dur) in zip(row['history'], row['history_positions'], s):
                            raw = table[tuple(pos)]
                            self.assertEqual(raw['item_id'], item)
                            expected = sum(1 << b for b, name in enumerate(history_side.FEEDBACK_BITS) if raw[name])
                            self.assertEqual((mask, ts, dur), (expected, raw['timespent'], durations[item]))

    def test_target_and_future_event_fields_cannot_change_features(self):
        """Leak test: mutating the target row (and every later row of that user) leaves that example's side unchanged."""
        from vkgr import history_side, history_v2
        partitions, examples, queries, catalog, durations = side_world(3)
        with tempfile.TemporaryDirectory() as tmp:
            h2 = history_v2.write_prepared(Path(tmp) / 'h2', partitions, examples, queries, catalog, binding={})
            hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 100)
        base = history_side.build_side(partitions, hist, durations)
        checked = 0
        for e in examples[::max(1, len(examples) // 12)]:
            mutated = copy.deepcopy(partitions)
            week, ordinal = e['target_position']
            for w, rows in mutated:
                for k, row in enumerate(rows):
                    if row['user_id'] == e['user_id'] and (w, k) >= (week, ordinal):
                        row.update(timespent=255 if row['timespent'] != 255 else 1, share=not row['share'], bookmark=not row['bookmark'],
                                   open_comments=not row['open_comments'], click_on_author=not row['click_on_author'], dislike=not row['dislike'])
            after = history_side.build_side(mutated, hist, durations)
            self.assertEqual(after['train'][e['example_id']], base['train'][e['example_id']])
            checked += 1
        self.assertGreater(checked, 5)
        # control: mutating a past event of the same user does change that example's side
        e = next(x for x in examples if len(x['history']) >= 2)
        week, ordinal = hist['train'][e['example_id']]['history_positions'][-1]
        mutated = copy.deepcopy(partitions); mutated[week][1][ordinal]['timespent'] = 254 if mutated[week][1][ordinal]['timespent'] != 254 else 253
        self.assertNotEqual(history_side.build_side(mutated, hist, durations)['train'][e['example_id']], base['train'][e['example_id']])

    def test_side_binding_is_fail_closed(self):
        from vkgr import history_side, history_v2
        with tempfile.TemporaryDirectory() as tmp:
            partitions, examples, queries, catalog, durations, h2, side = prepared_world(tmp)
            hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 50)
            manifest = Path(tmp) / 'side' / 'manifest.json'
            with self.assertRaises(ValueError):
                history_side.load_side(manifest, '0' * 64, h2['manifest_sha256'], 50, hist)
            with self.assertRaises(ValueError):
                history_side.load_side(manifest, side['manifest_sha256'], 'f' * 64, 50, hist)
            short = {**hist, 'train': [{**r, 'history': r['history'][:-1] or r['history']} for r in hist['train']]}
            if any(len(r['history']) > 1 for r in hist['train']):
                with self.assertRaises(ValueError):
                    history_side.load_side(manifest, side['manifest_sha256'], h2['manifest_sha256'], 50, short)
            bad = copy.deepcopy(partitions); bad[0][1][0]['item_id'] = (bad[0][1][0]['item_id'] + 1) % len(catalog)
            hist100 = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 100)
            if any([0, 0] in r['history_positions'] for r in hist100['train'] + hist100['validation']):
                with self.assertRaises(ValueError):
                    history_side.build_side(bad, hist100, durations)
            for broken in ({**durations, catalog[0]: 0}, {k: v for k, v in durations.items() if k != catalog[0]}):
                used = any(catalog[0] in r['history'] for r in hist100['train'])
                if used:
                    with self.assertRaises(ValueError):
                        history_side.build_side(partitions, hist100, broken)

    def test_validation_side_uses_train_weeks_only(self):
        from vkgr import history_side, history_v2
        with tempfile.TemporaryDirectory() as tmp:
            partitions, examples, queries, catalog, durations, h2, side = prepared_world(tmp)
            hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 100)
            with self.assertRaises(ValueError):
                history_side.build_side(partitions + [(25, partitions[0][1])], hist, durations)


def _model_inputs(seed=0, batch=6, width=12, items=300):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, width + 1, (batch,), generator=g)
    histories = torch.zeros(batch, width, dtype=torch.long); side = torch.zeros(batch, width, 3, dtype=torch.long)
    for j, n in enumerate(lengths.tolist()):
        histories[j, :n] = torch.randint(1, items + 1, (n,), generator=g)
        side[j, :n, 0] = torch.randint(0, 64, (n,), generator=g)
        side[j, :n, 2] = torch.tensor([[5, 9, 25, 60, 180][k] for k in torch.randint(0, 5, (n,), generator=g).tolist()])
        side[j, :n, 1] = torch.randint(1, 256, (n,), generator=g)
    return histories, side


class FeatureModelTests(unittest.TestCase):
    def setUp(self):
        self.content = torch.zeros(301, 64); self.supported = list(range(1, 200))

    def _model(self, feature, seed=42):
        from vkgr.training import seed_all
        from vkgr.retrieval_features import make_feature_model
        seed_all(seed)
        return make_feature_model(feature, self.content, self.supported, 50)

    def test_none_equals_frozen_small_init_id_mean(self):
        from vkgr.training import seed_all
        from vkgr.retrieval_hp import make_model
        seed_all(42); frozen = make_model('id_mean', 'small', self.content, self.supported, 50)
        histories, side = _model_inputs()
        for feature in ('none', 'f1_ratio_pool', 'f2_feedback', 'f3_watch_buckets'):
            model = self._model(feature)
            self.assertTrue(torch.equal(model.embedding.weight, frozen.embedding.weight), feature)
        none = self._model('none')
        self.assertEqual(sum(p.numel() for p in none.parameters()), sum(p.numel() for p in frozen.parameters()))
        self.assertTrue(torch.equal(none.users(histories, side), frozen.users(histories)))
        idx = torch.arange(1, 50)
        self.assertTrue(torch.equal(none.items(idx), frozen.items(idx)))

    def test_f1_weight_formula_and_no_new_parameters(self):
        from vkgr.retrieval_features import ratio_weight, RATIO_WEIGHT_MIN, RATIO_WEIGHT_MAX
        self.assertEqual((RATIO_WEIGHT_MIN, RATIO_WEIGHT_MAX), (0.1, 2.0))
        ts = torch.tensor([[1, 9, 30, 60, 255, 0]]); dur = torch.tensor([[180, 18, 30, 30, 9, 0]])
        expected = torch.tensor([[0.1, 0.5, 1.0, 2.0, 2.0, 0.0]])
        self.assertTrue(torch.allclose(ratio_weight(ts, dur, ts != 0), expected))
        f1 = self._model('f1_ratio_pool'); none = self._model('none')
        self.assertEqual(sum(p.numel() for p in f1.parameters()), sum(p.numel() for p in none.parameters()))
        histories, side = _model_inputs()
        valid = histories != 0
        enc = none.items(histories); w = ratio_weight(side[..., 1], side[..., 2], valid)
        self.assertTrue(torch.allclose(f1.users(histories, side), (enc * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True)))
        equal = side.clone(); equal[..., 1] = equal[..., 2] * valid  # ratio 1 everywhere -> plain mean
        self.assertTrue(torch.allclose(f1.users(histories, equal), none.users(histories, equal), atol=1e-7))

    def test_f2_multi_hot_feedback_added_to_item_embedding(self):
        from vkgr.retrieval_features import FEEDBACK_DIM
        f2 = self._model('f2_feedback'); none = self._model('none')
        self.assertEqual(FEEDBACK_DIM, 6)
        self.assertEqual(tuple(f2.feedback.weight.shape), (6, 64))
        self.assertLess(abs(float(f2.feedback.weight.detach().std()) - 0.02), 0.01)
        histories, side = _model_inputs()
        valid = histories != 0
        bits = torch.stack([(side[..., 0] >> b) & 1 for b in range(6)], -1).float() * valid.unsqueeze(-1)
        expected = (none.items(histories) + bits @ f2.feedback.weight).sum(1) / valid.sum(1, keepdim=True)
        self.assertTrue(torch.allclose(f2.users(histories, side), expected, atol=1e-7))
        zero = side.clone(); zero[..., 0] = 0
        self.assertTrue(torch.equal(f2.users(histories, zero), none.users(histories, zero)))
        one = torch.zeros(1, 2, dtype=torch.long); one[0, 0] = 250  # unsupported item, like+dislike bits
        s = torch.zeros(1, 2, 3, dtype=torch.long); s[0, 0] = torch.tensor([1 | 32, 5, 9])
        self.assertTrue(torch.allclose(f2.users(one, s)[0], f2.feedback.weight[0] + f2.feedback.weight[5]))

    def test_f3_bucket_boundaries(self):
        from vkgr.retrieval_features import timespent_bucket, ratio_bucket, TIMESPENT_EDGES
        self.assertEqual(TIMESPENT_EDGES, (4, 10, 31, 61, 135, 255))
        ts = torch.tensor([1, 3, 4, 9, 10, 30, 31, 60, 61, 134, 135, 254, 255])
        self.assertEqual(timespent_bucket(ts, ts != 0).tolist(), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7])
        self.assertEqual(timespent_bucket(torch.tensor([0]), torch.tensor([False])).tolist(), [0])
        ts = torch.tensor([1, 2, 3, 10, 9, 19, 20, 40, 41, 255, 0]); dur = torch.tensor([11, 20, 10, 20, 10, 20, 20, 40, 40, 9, 0])
        self.assertEqual(ratio_bucket(ts, dur, ts != 0).tolist(), [1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 0])

    def test_f3_adds_two_bucket_embeddings(self):
        from vkgr.retrieval_features import timespent_bucket, ratio_bucket
        f3 = self._model('f3_watch_buckets'); none = self._model('none')
        self.assertEqual((f3.timespent_bucket.num_embeddings, f3.ratio_bucket.num_embeddings), (8, 6))
        self.assertTrue(torch.all(f3.timespent_bucket.weight[0] == 0) and torch.all(f3.ratio_bucket.weight[0] == 0))
        histories, side = _model_inputs()
        valid = histories != 0
        add = f3.timespent_bucket(timespent_bucket(side[..., 1], valid)) + f3.ratio_bucket(ratio_bucket(side[..., 1], side[..., 2], valid))
        expected = (none.items(histories) + add).sum(1) / valid.sum(1, keepdim=True)
        self.assertTrue(torch.allclose(f3.users(histories, side), expected, atol=1e-7))

    def test_pad_side_and_bad_side_rejected(self):
        histories, side = _model_inputs()
        for feature in ('f1_ratio_pool', 'f2_feedback', 'f3_watch_buckets'):
            model = self._model(feature)
            bad = side.clone(); bad[0, -1] = torch.tensor([1, 5, 9]) if histories[0, -1] == 0 else bad[0, -1]
            if histories[0, -1] == 0:
                with self.assertRaises(ValueError):
                    model.users(histories, bad)
            with self.assertRaises(ValueError):
                model.users(histories, None)
            big = side.clone(); big[..., 0] = torch.where(histories != 0, torch.full_like(big[..., 0], 64), 0)
            with self.assertRaises(ValueError):
                model.users(histories, big)
            zero_dur = side.clone(); zero_dur[..., 2] = 0
            with self.assertRaises(ValueError):
                model.users(histories, zero_dur)


def f_config(**overrides):
    from vkgr.retrieval_feat01_run import default_config
    value = default_config(); value.update(overrides)
    return value


class Feat01RunTests(unittest.TestCase):
    def _run(self, tmp, **overrides):
        from vkgr import retrieval_feat01_run as fr
        data = fr.synthetic_data()
        config = f_config(batch_size=16, max_epochs=8, min_epochs=2, **overrides)
        value = fr.run_session(Path(tmp) / 'run', data, config)
        return fr, data, config, value

    def test_feature_none_reproduces_batch1_runner_exactly(self):
        from vkgr import retrieval_b1_run as b1
        from vkgr import retrieval_feat01_run as fr
        for extra in ({}, {'logq': True, 'extra_uniform': 128}):
            with self.subTest(**extra), tempfile.TemporaryDirectory() as tmp:
                data = fr.synthetic_data()
                old_config = {**b1.default_config(), 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2, **extra}
                if extra:
                    old_config['task_id'] = 'P-NEG-01'
                old = b1.run_session(Path(tmp) / 'b1', data, old_config)
                new = fr.run_session(Path(tmp) / 'f', data, f_config(batch_size=16, max_epochs=8, min_epochs=2, feature='none', **extra))
                self.assertEqual([r['loss'] for r in old['records']], [r['loss'] for r in new['records']])
                self.assertEqual(old['selection'], new['selection'])
                a = b1.full_evaluations(Path(tmp) / 'b1', old); b = fr.full_evaluations(Path(tmp) / 'f', new)
                self.assertEqual([e['result'] for e in a], [e['result'] for e in b])

    def test_each_feature_trains_and_verifies(self):
        for feature in ('f1_ratio_pool', 'f2_feedback', 'f3_watch_buckets'):
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                fr, data, config, value = self._run(tmp, feature=feature)
                run = Path(tmp) / 'run'
                self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
                self.assertEqual(value['binding']['feature'], feature)
                self.assertEqual(value['binding']['side_version'], 'HIST_V2_SIDE_V1')
                fr.verify_run(run / 'manifest.json', value['manifest_sha256'], data, config)
                with self.assertRaises(ValueError):
                    fr.verify_run(run / 'manifest.json', value['manifest_sha256'], data, {**config, 'feature': 'none'})
                changed = {**data, 'rows': [{**r, 'side': [[m ^ 1, t, d] for m, t, d in r['side']]} for r in data['rows']]}
                with self.assertRaises(ValueError):
                    fr.verify_run(run / 'manifest.json', value['manifest_sha256'], changed, config)

    def test_train_step_feeds_side_to_user_tower(self):
        import copy as _copy
        import random as _random
        from vkgr import retrieval_feat01_run as fr
        from vkgr.retrieval_negatives import masked_ce_logq, target_frequency
        from vkgr.retrieval_training import SamplerStats
        from vkgr import retrieval_b1_run as b1
        data = fr.synthetic_data(); rows, _ = fr._view(data, 50)
        for feature in ('f1_ratio_pool', 'f2_feedback', 'f3_watch_buckets'):
            config = f_config(batch_size=16, max_epochs=8, min_epochs=2, feature=feature)
            model = fr._model(data, config, 'cpu'); before = _copy.deepcopy(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            batch = rows[:16]; frequency = target_frequency(rows)
            record = fr.train_step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, _random.Random(1), None, frequency)
            columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, _random.Random(1), None)
            histories, side = fr._tensors(batch, data['idmap'], max(len(r['history']) for r in batch))
            expected, _ = masked_ce_logq(before.users(histories, side), before.items(columns), labels, mask, .15, b1._log_q(batch, negatives, pool_size, config, frequency))
            self.assertEqual(record['loss'], float(expected.detach()), feature)

    def test_feature_bit_order_is_frozen(self):
        from vkgr.history_side import FEEDBACK_BITS, _encode
        self.assertEqual(FEEDBACK_BITS, ('like', 'share', 'bookmark', 'open_comments', 'click_on_author', 'dislike'))
        row = {'like': True, 'share': False, 'bookmark': True, 'open_comments': False, 'click_on_author': False, 'dislike': True, 'timespent': 7}
        self.assertEqual(_encode(row, 30), [1 + 4 + 32, 7, 30])

    def test_features_change_the_model_output(self):
        """Mutation guard: each switch must actually reach the user vector."""
        import numpy as np
        from vkgr import retrieval_feat01_run as fr
        from vkgr.training import seed_all
        data = fr.synthetic_data(); rows, queries = fr._view(data, 50)
        outputs = {}
        for feature in ('none', 'f1_ratio_pool', 'f2_feedback', 'f3_watch_buckets'):
            seed_all(42); model = fr.make_feature_model(feature, data['content'], data['supported_indices'], 50)
            users, _ = fr.export_vectors_side(model, data['catalog'], queries, data['idmap'], 16, 50)
            outputs[feature] = users
        for feature in ('f1_ratio_pool', 'f2_feedback', 'f3_watch_buckets'):
            self.assertFalse(np.array_equal(outputs[feature], outputs['none']), feature)

    def test_config_registry(self):
        from vkgr.retrieval_feat01_run import validate_config
        validate_config(f_config())
        for bad in ({'feature': 'f4_recency'}, {'init': 'legacy'}, {'architecture': 'sasrec'}, {'history_len': 20}, {'max_epochs': 30},
                    {'side_version': 'v0'}, {'schema': 'p-b1-run-v1'}, {'task_id': 'HIST-V2'}, {'unknown': 1}):
            with self.assertRaises(ValueError, msg=bad):
                validate_config(f_config(**bad))

    def test_committed_feat01_configs_are_registered(self):
        from vkgr.retrieval_feat01_run import validate_config
        from vkgr.retrieval_ablation_run import strict_json
        paths = sorted((Path(__file__).resolve().parents[1] / 'configs' / 'feat01').glob('p-*.json'))
        self.assertEqual([p.stem for p in paths], ['p-feat01-f1-ratio-pool-s42', 'p-feat01-f2-feedback-s42', 'p-feat01-f3-watch-buckets-s42'])
        for path in paths:
            value = validate_config(strict_json(path.read_text()))
            # anchor = P-NEG N1 (in-batch logQ alpha=1, no extra uniform), all other fields equal to configs/batch1/p-neg-n1-logq.json
            self.assertEqual((value['seed'], value['history_len'], value['logq'], value['extra_uniform']), (42, 50, True, 0))
            anchor = strict_json((path.parents[1] / 'batch1' / 'p-neg-n1-logq.json').read_text())
            self.assertEqual({k: v for k, v in value.items() if k not in ('schema', 'task_id', 'feature', 'side_version')},
                             {k: v for k, v in anchor.items() if k not in ('schema', 'task_id')})

    def test_inputs_schema_requires_side_and_metadata_pins(self):
        from vkgr.retrieval_feat01_run import validate_inputs
        good = {'schema': 'p-feat01-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
                'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64,
                'items_metadata_sha256': 'c' * 64, 'disk_cap_bytes': 8 * 1024**3, 'test_gate': 'closed_hash_only'}
        validate_inputs(good)
        for bad in ({'side_sha256': None}, {'items_metadata_sha256': 'x'}, {'schema': 'p-b1-inputs-v1'}, {'history_manifest': None}):
            with self.assertRaises(ValueError, msg=bad):
                validate_inputs({**good, **bad})


if __name__ == '__main__':
    unittest.main()
