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


# ---------------------------------------------------------------- ITEM_ATTR_V1 (F6 author vocabulary and item -> author table)

def attr_world(seed=3, items=40, supported=15):
    """(catalog, support, authors): supported items use authors 1000..1005; half of the unsupported items reuse them,
    the others have an author that no supported item has (those authors are out of the train-defined vocabulary)."""
    rng = random.Random(seed)
    catalog = list(range(100, 100 + items)); support = set(catalog[:supported]); authors = {}
    for k, item in enumerate(catalog):
        if item in support:
            authors[item] = 1000 + k % 6
        else:
            authors[item] = 1000 + rng.randrange(6) if k % 2 == 0 else 2000 + k
    return catalog, support, authors


def write_metadata(path, authors, extra=(), item_type=None, author_type=None, drop=()):
    """items_metadata-like parquet: item_id / author_id (uint32 by default) / duration; ``extra`` rows are appended as given."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [(i, a) for i, a in authors.items() if i not in drop] + list(extra)
    pq.write_table(pa.table({'item_id': pa.array([r[0] for r in rows], item_type or pa.uint32()),
                             'author_id': pa.array([r[1] for r in rows], author_type or pa.uint32()),
                             'duration': pa.array([7] * len(rows), pa.uint8())}), str(path))
    return path


def guard_author_vocab_from_supported_items_only():
    """The vocabulary is the sorted authors of SUPPORTED catalog items: a new author on an unsupported item changes nothing."""
    from vkgr import item_attr
    catalog, support, authors = attr_world()
    expected = sorted({authors[i] for i in catalog if i in support})
    ensure(item_attr.author_vocab(authors, support) == expected, 'vocabulary must be the sorted authors of supported items')
    ensure(item_attr.build_item_attr(authors, catalog, support)['vocab'] == expected, 'build_item_attr vocabulary')
    unsupported = next(i for i in catalog if i not in support)
    changed = {**authors, unsupported: 987654}
    ensure(item_attr.author_vocab(changed, support) == expected, 'an unsupported item author must not enter the vocabulary')
    ensure(item_attr.build_item_attr(changed, catalog, support)['vocab'] == expected, 'build_item_attr: unsupported author entered the vocabulary')
    supported = next(iter(sorted(support)))
    changed = {**authors, supported: 987654}
    ensure(987654 in item_attr.author_vocab(changed, support), 'a supported item author must enter the vocabulary')
    ensure(len(expected) == 6 and any(authors[i] not in expected for i in catalog), 'fixture needs out-of-vocabulary authors')


def guard_prepare_attr_vocab_pin_binds_the_supported_vocab():
    """The hard-coded size/sha256 pins are checked against the vocabulary computed from the supported items."""
    import hashlib
    import json
    import tempfile
    from vkgr import item_attr
    catalog, support, authors = attr_world()
    vocab = sorted({authors[i] for i in catalog if i in support})
    size, sha = len(vocab), hashlib.sha256(json.dumps(vocab, separators=(',', ':')).encode()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            item_attr.write_item_attr(Path(tmp) / 'attr', 'a' * 64, catalog, support, authors, {'synthetic': True}, size, sha)
        except ValueError as exc:
            raise AssertionError(f'vocabulary pin rejected the genuine supported-item vocabulary: {exc}')


class ItemAttrTests(unittest.TestCase):
    def test_author_vocab_from_supported_items_only(self):
        guard_author_vocab_from_supported_items_only()

    def test_author_rows_pad0_oov1_and_rank(self):
        from vkgr import item_attr
        self.assertEqual(item_attr.author_rows([None, 1001, 1003, 77, 1005, 1001], [1001, 1003, 1005]), [0, 2, 3, 1, 4, 2])
        self.assertEqual(item_attr.author_rows([], [1]), [])
        self.assertEqual(item_attr.author_rows([5, 6], []), [1, 1])                 # empty vocabulary: everything is out of vocabulary
        for bad in ([1.0], [True], ['7'], [-1], [1 << 32]):
            with self.assertRaises(ValueError, msg=bad):
                item_attr.author_rows(bad, [1])
        with self.assertRaises(ValueError):
            item_attr.author_rows([1], [3, 2])                                     # vocabulary must be strictly ascending
        with self.assertRaises(ValueError):
            item_attr.author_rows([1], [2, 2])
        self.assertEqual(item_attr.ATTR_VERSION, 'ITEM_ATTR_V1')

    def test_vocab_digest_is_sha256_of_compact_ascending_json(self):
        import hashlib
        from vkgr import item_attr
        self.assertEqual(item_attr.vocab_digest([3, 5, 9]), hashlib.sha256(b'[3,5,9]').hexdigest())

    def test_item_attr_aligned_with_catalog_and_fail_closed(self):
        from vkgr import item_attr
        catalog, support, authors = attr_world()
        vocab = item_attr.author_vocab(authors, support)
        size, sha = len(vocab), item_attr.vocab_digest(vocab)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'attr'
            receipt = item_attr.write_item_attr(out, 'a' * 64, catalog, support, authors, {'synthetic': True}, size, sha)
            manifest = out / 'manifest.json'
            loaded = item_attr.load_item_attr(manifest, receipt['manifest_sha256'], 'a' * 64, catalog, support, size, sha)
            self.assertEqual(loaded['authors'], [authors[i] for i in catalog])                  # catalog order
            self.assertEqual((loaded['vocab'], loaded['vocab_size'], loaded['vocab_sha256']), (vocab, size, sha))
            value = receipt['manifest']
            self.assertEqual((value['kind'], value['attr_version'], value['items_metadata_sha256'], value['vocab_size'], value['vocab_sha256']),
                             ('item_attr_prepare', 'ITEM_ATTR_V1', 'a' * 64, size, sha))
            self.assertEqual(value['test_gate'], 'closed_hash_only')
            for key in ('catalog_sha256', 'support_sha256'):
                self.assertRegex(value[key], '^[0-9a-f]{64}$')
            args = (manifest, receipt['manifest_sha256'], 'a' * 64, catalog, support, size, sha)

            def load(**override):
                names = ('path', 'digest', 'items_metadata_sha256', 'catalog', 'support', 'expected_size', 'expected_sha256')
                return item_attr.load_item_attr(*[override.get(n, a) for n, a in zip(names, args)])
            for label, override in (('wrong manifest sha', {'digest': '0' * 64}), ('wrong metadata sha', {'items_metadata_sha256': 'b' * 64}),
                                    ('wrong catalog (reordered)', {'catalog': list(reversed(catalog))}), ('wrong catalog (shorter)', {'catalog': catalog[:-1]}),
                                    ('wrong catalog (longer)', {'catalog': catalog + [999]}), ('wrong support', {'support': support - {min(support)}}),
                                    ('wrong support (extra)', {'support': support | {max(catalog)}}), ('wrong vocab size pin', {'expected_size': size + 1}),
                                    ('wrong vocab sha pin', {'expected_sha256': '0' * 64})):
                with self.assertRaises(ValueError, msg=label):
                    load(**override)
            artifact = out / 'authors.json'; text = artifact.read_text()
            artifact.write_text(text.replace('[', '[ ', 1))
            with self.assertRaises(ValueError):
                load()                                                                           # artifact tampered after prepare
            artifact.write_text(text)
            load()
            with self.assertRaises(FileExistsError):
                item_attr.write_item_attr(out, 'a' * 64, catalog, support, authors, {}, size, sha)   # never overwritten
            # inconsistent inputs are rejected at build time
            with self.assertRaises(ValueError):
                item_attr.build_item_attr({k: v for k, v in authors.items() if k != catalog[0]}, catalog, support)   # missing catalog item
            with self.assertRaises(ValueError):
                item_attr.build_item_attr(authors, catalog, support | {12345})                    # support outside the catalog
            with self.assertRaises(ValueError):
                item_attr.build_item_attr({**authors, catalog[0]: 1.5}, catalog, support)
            with self.assertRaises(ValueError):
                item_attr.build_item_attr(authors, catalog + [catalog[0]], support)               # duplicate catalog item

    def test_load_authors_from_parquet(self):
        import pyarrow as pa
        from vkgr import item_attr
        catalog, support, authors = attr_world()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_metadata(Path(tmp) / 'ok.parquet', authors, extra=[(7, 9), (7, 10), (8, 4_000_000_000)])      # rows outside the catalog are ignored (even duplicates)
            self.assertEqual(item_attr.load_authors(path, catalog), authors)
            self.assertEqual(item_attr.load_authors(path, catalog[:5]), {i: authors[i] for i in catalog[:5]})
            big = write_metadata(Path(tmp) / 'big.parquet', {**authors, catalog[0]: 4_294_967_295})
            self.assertEqual(item_attr.load_authors(big, catalog)[catalog[0]], 4_294_967_295)
            for label, kwargs in (('float author', {'author_type': pa.float64()}), ('float item', {'item_type': pa.float64()}), ('string author', {'author_type': pa.string()})):
                bad = Path(tmp) / f'{label.replace(" ", "_")}.parquet'
                if label == 'string author':
                    write_metadata(bad, {i: str(a) for i, a in authors.items()}, **kwargs)
                else:
                    write_metadata(bad, authors, **kwargs)
                with self.assertRaises(ValueError, msg=label):
                    item_attr.load_authors(bad, catalog)
            with self.assertRaises(ValueError):
                item_attr.load_authors(write_metadata(Path(tmp) / 'dup.parquet', authors, extra=[(catalog[3], authors[catalog[3]])]), catalog)   # duplicate catalog item
            with self.assertRaises(ValueError):
                item_attr.load_authors(write_metadata(Path(tmp) / 'miss.parquet', authors, drop={catalog[4]}), catalog)                       # catalog item missing
            with self.assertRaises(ValueError):
                item_attr.load_authors(write_metadata(Path(tmp) / 'null.parquet', {**authors, catalog[2]: None}), catalog)                    # null author
            with self.assertRaises((ValueError, OSError)):
                item_attr.load_authors(Path(tmp) / 'absent.parquet', catalog)

    def test_prepare_attr_checks_hardcoded_vocab(self):
        from vkgr import item_attr
        catalog, support, authors = attr_world()
        vocab = item_attr.author_vocab(authors, support); size, sha = len(vocab), item_attr.vocab_digest(vocab)
        guard_prepare_attr_vocab_pin_binds_the_supported_vocab()
        with tempfile.TemporaryDirectory() as tmp:
            for label, pins in (('size', (size + 1, sha)), ('sha', (size, '0' * 64)), ('both', (11133, 'f' * 64))):
                with self.assertRaises(ValueError, msg=label):
                    item_attr.write_item_attr(Path(tmp) / f'attr-{label}', 'a' * 64, catalog, support, authors, {}, *pins)
                self.assertFalse((Path(tmp) / f'attr-{label}').exists(), 'nothing may be written when the vocabulary pin fails')
        # the production pins are the audit constants of the train-only audit (real data only; the synthetic world cannot match them)
        from vkgr import retrieval_feat02 as m
        self.assertEqual(m.AUTHOR_VOCAB_SIZE, 11133)
        self.assertEqual(m.AUTHOR_VOCAB_SHA256, '327ee976f2c8fff4e40267546bc2dd16818ad4d59e97da34106bb515dd1626bc')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                item_attr.write_item_attr(Path(tmp) / 'attr', 'a' * 64, catalog, support, authors, {}, m.AUTHOR_VOCAB_SIZE, m.AUTHOR_VOCAB_SHA256)

    def test_build_item_attr_has_no_query_or_holdout_parameters(self):
        import inspect
        from vkgr import item_attr
        self.assertEqual(list(inspect.signature(item_attr.build_item_attr).parameters), ['authors', 'catalog', 'support'])
        self.assertEqual(list(inspect.signature(item_attr.author_vocab).parameters), ['authors', 'support'])
        self.assertEqual(list(inspect.signature(item_attr.load_authors).parameters), ['path', 'catalog'])
        self.assertEqual(list(inspect.signature(item_attr.write_item_attr).parameters),
                         ['output', 'items_metadata_sha256', 'catalog', 'support', 'authors', 'binding', 'expected_size', 'expected_sha256'])
        self.assertEqual(list(inspect.signature(item_attr.load_item_attr).parameters),
                         ['path', 'digest', 'items_metadata_sha256', 'catalog', 'support', 'expected_size', 'expected_sha256'])
        for function in (item_attr.build_item_attr, item_attr.author_vocab, item_attr.load_authors, item_attr.write_item_attr, item_attr.load_item_attr, item_attr.author_rows):
            for name in inspect.signature(function).parameters:
                for banned in ('quer', 'holdout', 'valid', 'target', 'test'):
                    self.assertNotIn(banned, name, (function.__name__, name))

    def test_author_ids_are_never_derived_from_the_ranking_column(self):
        """items_metadata.parquet also has train_interactions_rank: it must not be read."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from vkgr import item_attr
        catalog, support, authors = attr_world()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.parquet'
            pq.write_table(pa.table({'item_id': pa.array(list(authors), pa.uint32()), 'author_id': pa.array(list(authors.values()), pa.uint32()),
                                     'train_interactions_rank': pa.array(['x'] * len(authors))}), str(path))      # a column that would not parse as an int
            self.assertEqual(item_attr.load_authors(path, catalog), authors)


if __name__ == '__main__':
    unittest.main()
