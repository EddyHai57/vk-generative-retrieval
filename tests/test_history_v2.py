"""Batch1 HIST_V2_OBS: observed (all train interaction) history with configurable length."""
import json
import random
import tempfile
import unittest
from pathlib import Path


def synthetic_world(seed=7, users=6, weeks=3, rows_per_week=160, items=40):
    """Physical-order train events plus like-prefix examples and validation queries (v1 shape)."""
    from vkgr.retrieval_data import prefix_examples
    rng = random.Random(seed)
    catalog = list(range(items))
    partitions = []
    for week in range(weeks):
        rows = [{'user_id': rng.randrange(users), 'item_id': rng.randrange(items), 'like': rng.random() < 0.3}
                for _ in range(rows_per_week)]
        partitions.append((week, rows))
    examples = list(prefix_examples(partitions))
    likes = {}
    for week, rows in partitions:
        for ordinal, row in enumerate(rows):
            if row['like']:
                likes.setdefault(row['user_id'], []).append((row['item_id'], [week, ordinal]))
    queries = []
    for j, user in enumerate(sorted(likes)):
        hist = likes[user][-50:]
        target = rng.randrange(items + 3)
        queries.append({'user_id': user, 'history': [x[0] for x in hist], 'history_positions': [x[1] for x in hist],
                        'target': target, 'target_position': [25, j], 'oov': target >= items})
    return partitions, examples, queries, catalog


class HistoryV2Tests(unittest.TestCase):
    def test_l50_is_value_equal_to_p4e_build_watch(self):
        from vkgr.history_v2 import build_observed
        from vkgr.retrieval_watch import build_watch
        for seed in (1, 7, 11):
            partitions, examples, queries, catalog = synthetic_world(seed)
            expected = build_watch(partitions, examples, queries, catalog)
            actual = build_observed(partitions, examples, queries, catalog, max_len=50)
            self.assertEqual(actual['train'], expected[0])
            self.assertEqual(actual['validation'], expected[1])
            self.assertEqual(actual['slices'], expected[2])
            self.assertTrue(any(len(r['history']) == 50 for r in actual['train']), 'fixture must hit the 50 cap')

    def test_shorter_lengths_are_suffixes_of_l100(self):
        from vkgr.history_v2 import build_observed, truncate_sidecar
        partitions, examples, queries, catalog = synthetic_world(3, rows_per_week=300)
        full = build_observed(partitions, examples, queries, catalog, max_len=100)
        self.assertTrue(any(len(r['history']) > 50 for r in full['train']))
        for length in (50, 20):
            built = build_observed(partitions, examples, queries, catalog, max_len=length)
            self.assertEqual(truncate_sidecar(full['train'], length), built['train'])
            self.assertEqual(truncate_sidecar(full['validation'], length), built['validation'])

    def test_length_statistics_count_untruncated_observed_history(self):
        from vkgr.history_v2 import build_observed
        events = [{'user_id': 1, 'item_id': 0, 'like': True}] + [{'user_id': 1, 'item_id': 2, 'like': False}] * 120 + [{'user_id': 1, 'item_id': 3, 'like': True}]
        e = {'example_id': 0, 'user_id': 1, 'history': [0], 'history_positions': [[0, 0]], 'target': 3, 'target_position': [0, 121], 'past_likes': [0]}
        q = {'user_id': 1, 'history': [0, 3], 'history_positions': [[0, 0], [0, 121]], 'target': 3, 'target_position': [25, 0], 'oov': False}
        out = build_observed([(0, events)], [e], [q], [0, 2, 3], max_len=100)
        stats = out['length_stats']
        self.assertEqual(stats['train']['buckets'], {'1-20': 0, '21-50': 0, '51-100': 0, 'gt100': 1})
        self.assertEqual(stats['validation']['buckets'], {'1-20': 0, '21-50': 0, '51-100': 0, 'gt100': 1})
        self.assertEqual(stats['train']['max'], 121)
        self.assertEqual(stats['validation']['max'], 122)
        self.assertEqual(stats['validation']['fraction_gt50'], 1.0)
        self.assertTrue(stats['l100_gate']['keep_l100'])
        self.assertEqual(len(out['train'][0]['history']), 100)

    def test_l100_gate_cancels_when_validation_rarely_exceeds_50(self):
        from vkgr.history_v2 import l100_gate
        self.assertFalse(l100_gate({'fraction_gt50': 0.0999})['keep_l100'])
        self.assertTrue(l100_gate({'fraction_gt50': 0.10})['keep_l100'])

    def test_sidecar_validation_is_fail_closed(self):
        from vkgr.history_v2 import build_observed, overlay_examples, overlay_queries
        partitions, examples, queries, catalog = synthetic_world(5)
        out = build_observed(partitions, examples, queries, catalog, max_len=20)
        rows = overlay_examples(examples, out['train'], 20)
        self.assertEqual([r['target'] for r in rows], [e['target'] for e in examples])
        self.assertEqual([r['past_likes'] for r in rows], [e['past_likes'] for e in examples])
        self.assertTrue(all(1 <= len(r['history']) <= 20 for r in rows))
        overlay_queries(queries, out['validation'], 20, set(catalog))
        bad = [dict(r) for r in out['train']]
        bad[0] = {**bad[0], 'history': bad[0]['history'] + [bad[0]['history'][0]] * 30, 'history_positions': bad[0]['history_positions'] + [[0, 900 + k] for k in range(30)]}
        with self.assertRaises(ValueError):
            overlay_examples(examples, bad, 20)
        mutated = [dict(r) for r in out['train']]
        mutated[1] = {**mutated[1], 'history_positions': list(reversed(mutated[1]['history_positions']))}
        if len(mutated[1]['history']) > 1:
            with self.assertRaises(ValueError):
                overlay_examples(examples, mutated, 20)
        late = [dict(r) for r in out['train']]
        late[0] = {**late[0], 'history_positions': late[0]['history_positions'][:-1] + [list(examples[0]['target_position'])]}
        with self.assertRaises(ValueError):
            overlay_examples(examples, late, 20)
        week = [dict(q) for q in out['validation']]
        week[0] = {**week[0], 'history_positions': week[0]['history_positions'][:-1] + [[25, 0]]}
        with self.assertRaises(ValueError):
            overlay_queries(queries, week, 20, set(catalog))
        drift = [dict(q) for q in out['validation']]
        drift[0] = {**drift[0], 'target': drift[0]['target'] + 1}
        with self.assertRaises(ValueError):
            overlay_queries(queries, drift, 20, set(catalog))

    def test_prepare_writes_three_lengths_and_verifies(self):
        from vkgr import history_v2
        partitions, examples, queries, catalog = synthetic_world(9, rows_per_week=300)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'hist'
            receipt = history_v2.write_prepared(out, partitions, examples, queries, catalog, binding={'synthetic': True})
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['lengths'], [100, 50, 20])
            self.assertEqual(manifest['history_version'], 'HIST_V2_OBS')
            for length in (100, 50, 20):
                data = history_v2.load_prepared(out / 'manifest.json', receipt['manifest_sha256'], length)
                self.assertEqual(len(data['train']), len(examples))
                self.assertEqual(len(data['validation']), len(queries))
                self.assertTrue(all(len(r['history']) <= length for r in data['train']))
            l100 = history_v2.load_prepared(out / 'manifest.json', receipt['manifest_sha256'], 100)
            l50 = history_v2.load_prepared(out / 'manifest.json', receipt['manifest_sha256'], 50)
            self.assertEqual(history_v2.truncate_sidecar(l100['train'], 50), l50['train'])
            with self.assertRaises(ValueError):
                history_v2.load_prepared(out / 'manifest.json', '0' * 64, 50)
            with self.assertRaises(ValueError):
                history_v2.load_prepared(out / 'manifest.json', receipt['manifest_sha256'], 30)


if __name__ == '__main__':
    unittest.main()
