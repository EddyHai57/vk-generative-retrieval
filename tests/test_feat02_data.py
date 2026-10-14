"""FEAT-02 (P line) data layer: HIST_V2_CTX_V1 sidecar (per-history-event [place, platform]) for the F5 scene variant.

``guard_*`` functions are plain assertion functions reused by tests/test_feat02_mutations.py (M8).
"""
import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_model import ensure  # noqa: E402
from test_history_v2 import synthetic_world  # noqa: E402

OOV_PLACE, OOV_PLATFORM = 200, 99          # values that are in no train-defined table


def ctx_world(seed=7, weeks=3, rows_per_week=160):
    """synthetic_world rows enriched with place / platform / agent, feedback bits and timespent.

    Every 53rd raw row carries an out-of-table place and platform so the OOV path is exercised downstream.
    """
    partitions, examples, queries, catalog = synthetic_world(seed, weeks=weeks, rows_per_week=rows_per_week)
    rng = random.Random(seed + 200); count = 0
    for _, rows in partitions:
        for row in rows:
            for name in ('share', 'bookmark', 'open_comments', 'click_on_author', 'dislike'):
                row[name] = rng.random() < 0.2
            row['timespent'] = rng.choice([1, 2, 5, 9, 12, 40, 70, 150, 255])
            row['place'], row['platform'], row['agent'] = rng.randrange(24), rng.randrange(11), rng.randrange(4)
            count += 1
            if count % 53 == 0:
                row['place'], row['platform'] = OOV_PLACE, OOV_PLATFORM
    return partitions, examples, queries, catalog


def prepared_ctx_world(tmp, seed=7, weeks=3):
    from vkgr import history_ctx, history_v2
    partitions, examples, queries, catalog = ctx_world(seed, weeks=weeks)
    h2 = history_v2.write_prepared(Path(tmp) / 'h2', partitions, examples, queries, catalog, binding={'synthetic': True})
    ctx = history_ctx.write_ctx(Path(tmp) / 'ctx', Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], partitions, binding={'synthetic': True})
    return partitions, examples, queries, catalog, h2, ctx


def prepared_full_world(tmp, seed=7, weeks=3):
    """history-v2 + HIST_V2_SIDE_V1 + HIST_V2_CTX_V1 on the same raw rows (what the real FEAT-02 F5 run pins)."""
    from vkgr import history_ctx, history_side, history_v2
    partitions, examples, queries, catalog = ctx_world(seed, weeks=weeks)
    durations = {i: random.Random(seed + i).choice([5, 9, 25, 60, 180]) for i in catalog}
    h2 = history_v2.write_prepared(Path(tmp) / 'h2', partitions, examples, queries, catalog, binding={'synthetic': True})
    manifest = Path(tmp) / 'h2' / 'manifest.json'
    side = history_side.write_side(Path(tmp) / 'side', manifest, h2['manifest_sha256'], partitions, durations, binding={'synthetic': True})
    ctx = history_ctx.write_ctx(Path(tmp) / 'ctx', manifest, h2['manifest_sha256'], partitions, binding={'synthetic': True})
    return partitions, examples, queries, catalog, durations, h2, side, ctx


def flip_row(row):
    """Everything a raw row carries, changed (including place, platform, agent)."""
    row.update(place=(row['place'] + 5) % 256, platform=(row['platform'] + 3) % 256, agent=(row['agent'] + 1) % 256,
               timespent=255 if row['timespent'] != 255 else 1, like=not row['like'], share=not row['share'], bookmark=not row['bookmark'],
               open_comments=not row['open_comments'], click_on_author=not row['click_on_author'], dislike=not row['dislike'])


def guard_ctx_target_and_future_event_fields_cannot_change_features():
    """Mutating the target row and every later row of that user leaves the example's ctx unchanged; a past event does change it."""
    from vkgr import history_ctx, history_v2
    partitions, examples, queries, catalog = ctx_world(3)
    with tempfile.TemporaryDirectory() as tmp:
        h2 = history_v2.write_prepared(Path(tmp) / 'h2', partitions, examples, queries, catalog, binding={})
        hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 100)
    base = history_ctx.build_ctx(partitions, hist)
    checked = 0
    for e in examples[::max(1, len(examples) // 12)]:
        mutated = copy.deepcopy(partitions)
        week, ordinal = e['target_position']
        for w, rows in mutated:
            for k, row in enumerate(rows):
                if row['user_id'] == e['user_id'] and (w, k) >= (week, ordinal):
                    flip_row(row)
        after = history_ctx.build_ctx(mutated, hist)
        ensure(after['train'][e['example_id']] == base['train'][e['example_id']], 'ctx changed when the target/later rows changed')
        checked += 1
    ensure(checked > 5, 'too few examples checked')
    e = next(x for x in examples if len(x['history']) >= 2)
    week, ordinal = hist['train'][e['example_id']]['history_positions'][-1]
    mutated = copy.deepcopy(partitions); mutated[week][1][ordinal]['place'] = (mutated[week][1][ordinal]['place'] + 1) % 256
    ensure(history_ctx.build_ctx(mutated, hist)['train'][e['example_id']] != base['train'][e['example_id']], 'control: a past event place must change ctx')


class CtxSidecarTests(unittest.TestCase):
    def test_ctx_is_aligned_with_history_events(self):
        from vkgr import history_ctx, history_v2
        with tempfile.TemporaryDirectory() as tmp:
            partitions, examples, queries, catalog, h2, ctx = prepared_ctx_world(tmp)
            table = {(w, k): r for w, rows in partitions for k, r in enumerate(rows)}
            seen_oov = False
            for length in (100, 50, 20):
                hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], length)
                loaded = history_ctx.load_ctx(Path(tmp) / 'ctx' / 'manifest.json', ctx['manifest_sha256'], h2['manifest_sha256'], length, hist)
                for split in ('train', 'validation'):
                    self.assertEqual(len(loaded[split]), len(hist[split]))
                    for row, c in zip(hist[split], loaded[split]):
                        self.assertEqual(len(c), len(row['history']))
                        for item, pos, (place, platform) in zip(row['history'], row['history_positions'], c):
                            raw = table[tuple(pos)]
                            self.assertEqual(raw['item_id'], item)
                            self.assertEqual((place, platform), (raw['place'], raw['platform']))
                            seen_oov = seen_oov or place == OOV_PLACE
            self.assertTrue(seen_oov, 'fixture must put an out-of-table value into some history')
            self.assertEqual(history_ctx.FIELDS, ('place', 'platform'))
            self.assertEqual(history_ctx.CTX_VERSION, 'HIST_V2_CTX_V1')

    def test_ctx_target_and_future_event_fields_cannot_change_features(self):
        guard_ctx_target_and_future_event_fields_cannot_change_features()

    def test_ctx_agent_never_enters(self):
        from vkgr import history_ctx, history_v2
        partitions, examples, queries, catalog = ctx_world(4)
        with tempfile.TemporaryDirectory() as tmp:
            h2 = history_v2.write_prepared(Path(tmp) / 'h2', partitions, examples, queries, catalog, binding={})
            hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 100)
        base = history_ctx.build_ctx(partitions, hist)
        changed = copy.deepcopy(partitions)
        for _, rows in changed:
            for row in rows:
                row['agent'] = (row['agent'] + 17) % 256
        self.assertEqual(history_ctx.build_ctx(changed, hist), base)
        self.assertEqual(history_ctx.FIELDS, ('place', 'platform'))
        for row in changed[0][1]:
            del row['agent']
        self.assertEqual(history_ctx.build_ctx(changed, hist), base)      # the column is not even read
        for split in ('train', 'validation'):
            self.assertTrue(all(len(pair) == 2 for rows in base[split] for pair in rows))

    def test_ctx_binding_is_fail_closed(self):
        from vkgr import history_ctx, history_v2
        with tempfile.TemporaryDirectory() as tmp:
            partitions, examples, queries, catalog, h2, ctx = prepared_ctx_world(tmp)
            manifest = Path(tmp) / 'ctx' / 'manifest.json'
            hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 50)
            history_ctx.load_ctx(manifest, ctx['manifest_sha256'], h2['manifest_sha256'], 50, hist)    # sound baseline
            with self.assertRaises(ValueError):
                history_ctx.load_ctx(manifest, '0' * 64, h2['manifest_sha256'], 50, hist)             # wrong ctx sha
            with self.assertRaises(ValueError):
                history_ctx.load_ctx(manifest, ctx['manifest_sha256'], 'f' * 64, 50, hist)            # wrong history sha
            with self.assertRaises(ValueError):
                history_ctx.load_ctx(manifest, ctx['manifest_sha256'], h2['manifest_sha256'], 30, hist)   # unregistered length
            self.assertTrue(any(len(r['history']) > 1 for r in hist['train']))
            short = {**hist, 'train': [{**r, 'history': r['history'][:-1] or r['history']} for r in hist['train']]}
            with self.assertRaises(ValueError):
                history_ctx.load_ctx(manifest, ctx['manifest_sha256'], h2['manifest_sha256'], 50, short)  # misaligned lengths
            with self.assertRaises(ValueError):
                history_ctx.load_ctx(manifest, ctx['manifest_sha256'], h2['manifest_sha256'], 50, {**hist, 'train': hist['train'][:-1]})   # denominator drift
            artifact = Path(tmp) / 'ctx' / 'ctx-train-l100.json'; text = artifact.read_text()
            artifact.write_text(text.replace('[', '[ ', 1))
            with self.assertRaises(ValueError):
                history_ctx.load_ctx(manifest, ctx['manifest_sha256'], h2['manifest_sha256'], 50, hist)   # artifact tampered after prepare
            artifact.write_text(text)
            hist100 = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 100)
            for bad in (1.0, True, '3', None, 256, -1):
                broken = copy.deepcopy(partitions)
                pos = hist100['train'][-1]['history_positions'][-1]
                broken[pos[0]][1][pos[1]]['place'] = bad
                with self.assertRaises(ValueError, msg=bad):
                    history_ctx.build_ctx(broken, hist100)
                broken = copy.deepcopy(partitions); broken[pos[0]][1][pos[1]]['platform'] = bad
                with self.assertRaises(ValueError, msg=bad):
                    history_ctx.build_ctx(broken, hist100)
            for field in ('place', 'platform'):
                broken = copy.deepcopy(partitions); del broken[pos[0]][1][pos[1]][field]
                with self.assertRaises(ValueError, msg=field):
                    history_ctx.build_ctx(broken, hist100)
            broken = copy.deepcopy(partitions); broken[pos[0]][1][pos[1]]['item_id'] = (broken[pos[0]][1][pos[1]]['item_id'] + 1) % len(catalog)
            with self.assertRaises(ValueError):
                history_ctx.build_ctx(broken, hist100)                   # item mismatch
            with self.assertRaises(ValueError):
                history_ctx.build_ctx([(w, r) for w, r in partitions if w != pos[0]], hist100)   # history event absent from raw

    def test_ctx_validation_uses_train_weeks_only(self):
        from vkgr import history_ctx, history_v2
        with tempfile.TemporaryDirectory() as tmp:
            partitions, examples, queries, catalog, h2, ctx = prepared_ctx_world(tmp)
            hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 100)
            with self.assertRaises(ValueError):
                history_ctx.build_ctx(partitions + [(25, partitions[0][1])], hist)
            with self.assertRaises(ValueError):
                history_ctx.build_ctx(partitions + [(2, partitions[0][1])], hist)         # weeks must strictly increase

    def test_ctx_week_spread_up_to_24(self):
        """Real week range: 25 train weeks are accepted and the sidecar covers events from the last week too."""
        from vkgr import history_ctx, history_v2
        partitions, examples, queries, catalog = ctx_world(5, weeks=25, rows_per_week=40)
        with tempfile.TemporaryDirectory() as tmp:
            h2 = history_v2.write_prepared(Path(tmp) / 'h2', partitions, examples, queries, catalog, binding={})
            hist = history_v2.load_prepared(Path(tmp) / 'h2' / 'manifest.json', h2['manifest_sha256'], 100)
        built = history_ctx.build_ctx(partitions, hist)
        weeks = {p[0] for r in hist['validation'] for p in r['history_positions']}
        self.assertTrue(max(weeks) >= 20)
        self.assertEqual(len(built['validation']), len(hist['validation']))

    def test_build_ctx_has_no_target_or_holdout_parameters(self):
        import inspect
        from vkgr import history_ctx
        self.assertEqual(list(inspect.signature(history_ctx.build_ctx).parameters), ['partitions', 'history'])
        self.assertEqual(list(inspect.signature(history_ctx.write_ctx).parameters), ['output', 'history_manifest', 'history_sha256', 'partitions', 'binding'])


if __name__ == '__main__':
    unittest.main()
