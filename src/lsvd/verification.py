"""Read-only row and view checks for a completed foundation."""
from pathlib import Path
import csv
import json
import resource
import time

import duckdb
import numpy as np

from .data import SEQUENCE_FIELDS, _literal, file_sha256
from .evaluation import EvaluationData


def verify_foundation(dataset, output, *, legacy_context=None, memory_limit='8GB'):
    dataset, output = Path(dataset), Path(output)
    manifest = json.loads((dataset / 'foundation.json').read_text())
    if manifest['status'] != 'complete' or any(row['week'] > 25 for row in manifest['source_rows']):
        raise ValueError('verification requires a complete train/validation foundation')
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    connection = duckdb.connect(config={'threads': 2, 'memory_limit': memory_limit,
                                      'temp_directory': str(output / 'tmp')})
    sql_log = (output / 'executed.sql').open('x')
    checks = {}
    weeks = []
    blocks = []

    def query(sql):
        sql_log.write(sql + ';\n\n')
        sql_log.flush()
        return connection.execute(sql)

    def require(name, value):
        checks[name] = bool(value)
        if not value:
            raise ValueError(f'foundation verification failed: {name}')

    try:
        master = Path(manifest['master_source'])
        query(f"CREATE VIEW master AS SELECT * FROM read_parquet({_literal(master / '*' / '*.parquet')},hive_partitioning=true)")
        query(f"CREATE VIEW sequences AS SELECT * FROM read_parquet({_literal(dataset / 'sequences' / '*' / '*.parquet')})")
        for name in ['queries', 'targets', 'items']:
            query(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({_literal(dataset / (name + ".parquet"))})')
        if legacy_context is not None:
            query(f'ATTACH {_literal(Path(legacy_context))} AS legacy(READ_ONLY)')
        raw_files = {int(Path(row['path']).stem.split('_')[1]): Path(row['path'])
                     for row in manifest['inputs'] if Path(row['path']).name.startswith('week_')
                     and Path(row['path']).suffix == '.parquet'}
        raw_fields = ['user_id', 'item_id', 'place', 'platform', 'agent', 'timespent',
                      'like', 'dislike', 'share', 'bookmark', 'click_on_author', 'open_comments']
        raw_mismatch = ' OR '.join(f'e."{name}" IS DISTINCT FROM r."{name}"' for name in raw_fields)
        mask = '+'.join(f'{name}::USMALLINT*{1 << bit}' for bit, name in enumerate(manifest['feedback_mask_bits']))
        raw_total = 0
        for source in manifest['source_rows']:
            week = int(source['week'])
            query(f'''CREATE OR REPLACE TEMP VIEW current_master AS SELECT * FROM master WHERE week={week}''')
            original = raw_files[week]
            rows, matched, differences = query(f'''SELECT count(*),count(e.ordinal),count(*) FILTER(WHERE {raw_mismatch})
FROM read_parquet({_literal(original)},file_row_number=true) r
LEFT JOIN current_master e ON r.file_row_number=e.ordinal''').fetchone()
            distinct_rows, total = query('SELECT count(DISTINCT ordinal),count(*) FROM current_master').fetchone()
            require(f'raw_week{week:02}_all_fields', rows == matched == distinct_rows == total == source['rows'] and differences == 0)
            invalid = query(f'''SELECT count(*) FROM current_master WHERE
main_positive NOT IN(0,1) OR secondary_positive NOT IN(0,1) OR like_positive<>"like"::UTINYINT
OR main_positive IS NULL OR secondary_positive IS NULL
OR (dislike AND(main_positive=1 OR secondary_positive=1))
OR quick_skip<>(timespent<3)::UTINYINT OR watch_ratio<>timespent::DOUBLE/duration
OR over_duration<>(timespent>duration)::UTINYINT OR at_cap<>(timespent=255)::UTINYINT
OR target_main<>((place IN(0,1)) AND main_positive=1)::UTINYINT
OR train_weight<>CASE WHEN week<=24 AND place IN(0,1) THEN CASE WHEN main_positive=1 THEN 1
WHEN secondary_positive=1 THEN 0.25 ELSE 0 END ELSE 0 END::FLOAT
OR feedback_mask<>({mask})::USMALLINT OR user_block<>user_id%32''').fetchone()[0]
            require(f'derived_week{week:02}_no_rehash', invalid == 0)
            legacy_difference = None
            if legacy_context is not None:
                legacy_count, legacy_difference = query(f'''SELECT count(r.user_id),
count(*) FILTER(WHERE e.main_positive IS DISTINCT FROM r.main_positive::UTINYINT)
FROM current_master e LEFT JOIN legacy.context_events r
ON r.week={week} AND e.ordinal=r.ordinal AND e.user_id=r.user_id AND e.item_id=r.item_id''').fetchone()
                require(f'legacy_week{week:02}_keys', legacy_count == total)
            raw_total += total
            weeks.append({'week': week, 'rows': total, 'raw_field_differences': differences,
                          'derived_differences': invalid, 'legacy_main_differences': legacy_difference})
            print(json.dumps({'stage': 'verify_week', **weeks[-1]}), flush=True)

        sequence_total = 0
        for block in range(32):
            path = dataset / 'sequences' / f'user_block={block}' / 'users.parquet'
            query(f'CREATE OR REPLACE TEMP VIEW current_sequences AS SELECT * FROM read_parquet({_literal(path)})')
            different_lengths = ' OR '.join(f'len({name})<>len(item_ids)' for name in SEQUENCE_FIELDS)
            bad = query(f'''SELECT count(*) FROM current_sequences WHERE user_id%32<>{block}
OR {different_lengths} OR list_zip(weeks,ordinals)<>list_sort(list_zip(weeks,ordinals))''').fetchone()[0]
            require(f'sequence_block{block:02}_shape_and_order', bad == 0)
            expanded = ','.join(f'unnest({name}) AS "{column}"' for name, column in SEQUENCE_FIELDS.items())
            compare = ' OR '.join(f's."{column}" IS DISTINCT FROM e."{column}"' for column in SEQUENCE_FIELDS.values())
            rows, differences = query(f'''WITH expanded AS(SELECT user_id,{expanded} FROM current_sequences)
SELECT count(*),count(*) FILTER(WHERE {compare}) FROM expanded s
LEFT JOIN master e ON e.user_block={block} AND e.week<=24
AND s.user_id=e.user_id AND s.week=e.week AND s.ordinal=e.ordinal''').fetchone()
            source_count = query(f'SELECT count(*) FROM master WHERE week<=24 AND user_block={block}').fetchone()[0]
            require(f'sequence_block{block:02}_full_fields', rows == source_count and differences == 0)
            sequence_total += rows
            blocks.append({'user_block': block, 'rows': rows, 'differences': differences})
            print(json.dumps({'stage': 'verify_sequence', **blocks[-1]}), flush=True)
        require('sequence_total_training_rows', sequence_total == manifest['train_rows'])

        query_difference = query('''WITH history AS(SELECT user_id,row_number() OVER(ORDER BY user_id)-1 AS row,
len(item_ids) AS n,item_ids[-1] AS item,authors[-1] AS author FROM sequences)
SELECT count(*) FROM queries q LEFT JOIN history h USING(user_id)
WHERE q.history_count IS DISTINCT FROM coalesce(h.n,0)
OR q.history_row IS DISTINCT FROM coalesce(h.row,-1)
OR q.last_item_id IS DISTINCT FROM coalesce(h.item,-1)
OR q.last_author_id IS DISTINCT FROM coalesce(h.author,-1)
OR q.history_block<>q.user_id%32''').fetchone()[0]
        require('query_history_references_and_anchors', query_difference == 0)
        q_difference = query('''SELECT count(*) FROM(
(SELECT user_id FROM queries EXCEPT SELECT DISTINCT user_id FROM master WHERE week=25)
UNION ALL(SELECT DISTINCT user_id FROM master WHERE week=25 EXCEPT SELECT user_id FROM queries))''').fetchone()[0]
        require('all_validation_users_retained', q_difference == 0)
        query('''CREATE TEMP VIEW expected_targets AS SELECT e.user_id,e.item_id,e.place,e.duration,
e.main_positive,e.secondary_positive,e.like_positive,(i.train_count>0) AS train_seen,
CASE WHEN e.duration<=60 THEN 0 WHEN e.duration<=120 THEN 1 ELSE 2 END::UTINYINT AS duration_band,
CASE WHEN e.duration<=19 THEN 0 WHEN e.duration<=53 THEN 1 ELSE 2 END::UTINYINT AS duration_band_sensitive,
i.popularity_tier FROM master e JOIN items i USING(item_id)
WHERE week=25 AND(main_positive=1 OR secondary_positive=1 OR like_positive=1)''')
        target_difference = query('''SELECT count(*) FROM(
(SELECT * FROM targets EXCEPT SELECT * FROM expected_targets)
UNION ALL(SELECT * FROM expected_targets EXCEPT SELECT * FROM targets))''').fetchone()[0]
        require('target_union_and_all_context_columns', target_difference == 0)
        require('target_unique_user_item', query('SELECT count(*)=count(DISTINCT(user_id,item_id)) FROM targets').fetchone()[0])
        p80, p95 = query('SELECT quantile_disc(train_count,.8),quantile_disc(train_count,.95) FROM items WHERE train_count>0').fetchone()
        require('popularity_train_seen_quantiles', (p80, p95) == (manifest['popularity']['p80'], manifest['popularity']['p95']))
        invalid_tiers = query(f'''SELECT count(*) FROM items WHERE popularity_tier<>
CASE WHEN train_count=0 THEN 0 WHEN train_count>={p95} THEN 3 WHEN train_count>={p80} THEN 2 ELSE 1 END''').fetchone()[0]
        require('popularity_equal_boundaries_hotter', invalid_tiers == 0)

        arrays = EvaluationData.load(dataset)
        queries = query('SELECT * FROM queries ORDER BY user_id').fetchnumpy()
        require('query_array_users', np.array_equal(queries['user_id'], arrays.user_ids))
        with np.load(dataset / 'arrays' / 'queries.npz', allow_pickle=False) as archive:
            query_arrays = {name: archive[name] for name in archive.files}
        require('query_array_all_columns', queries.keys() == query_arrays.keys()
                and all(np.array_equal(values, query_arrays[name]) for name, values in queries.items()))
        history = query('SELECT user_id,len(item_ids) AS n FROM sequences ORDER BY user_id').fetchnumpy()
        require('history_array_users', np.array_equal(history['user_id'], arrays.history_user_ids))
        require('history_array_offsets', np.array_equal(np.r_[0, np.cumsum(history['n'])], arrays.history_offsets))
        expected_item_columns = {'item_id', 'author_id', 'duration', 'train_count', 'popularity_tier'}
        require('item_array_column_schema', set(arrays.items) == expected_item_columns)
        item_columns = ','.join(f'"{name}"' for name in sorted(expected_item_columns))
        item_arrays = query(f'SELECT {item_columns} FROM items ORDER BY item_id').fetchnumpy()
        require('item_array_all_columns', all(np.array_equal(values, arrays.items[name])
                for name, values in item_arrays.items()))
        del item_arrays
        target_arrays = query('SELECT * FROM targets ORDER BY user_id,item_id').fetchnumpy()
        target_users = target_arrays.pop('user_id')
        positions = np.searchsorted(arrays.user_ids, target_users)
        counts = np.bincount(positions, minlength=len(arrays.user_ids))
        require('target_array_offsets', np.array_equal(arrays.targets['offsets'], np.r_[0, np.cumsum(counts)]))
        require('target_array_all_columns', all(np.array_equal(values, arrays.targets[name]) for name, values in target_arrays.items()))
        del target_arrays, target_users, positions, counts
        for block in range(32):
            path = dataset / 'sequences' / f'user_block={block}' / 'users.parquet'
            rows = query(f'SELECT user_id,item_ids FROM read_parquet({_literal(path)})').fetchnumpy()
            require(f'seen_array_block{block:02}', all(np.array_equal(np.sort(items), arrays.seen_items(user_id))
                    for user_id, items in zip(rows['user_id'], rows['item_ids'])))
        empty = query('SELECT count(*) FROM queries WHERE history_count=0').fetchone()[0]
        zero_main = query('''SELECT count(*) FROM queries q ANTI JOIN
(SELECT DISTINCT user_id FROM targets WHERE main_positive=1 AND place IN(0,1)) t USING(user_id)''').fetchone()[0]
        for name, rows in [('weekly.csv', weeks), ('sequence_blocks.csv', blocks)]:
            with (output / name).open('x', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        result = {'status': 'verified', 'checks': checks, 'raw_rows_checked': raw_total,
                  'sequence_rows_checked': sequence_total, 'empty_history_users': empty,
                  'zero_main_target_users': zero_main,
                  'legacy_main_differences': sum(row['legacy_main_differences'] or 0 for row in weeks),
                  'foundation_sha256': file_sha256(dataset / 'foundation.json'),
                  'code_sha256': file_sha256(Path(__file__)), 'seconds': time.monotonic() - start,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'test_week26_read': False, 'label_hash_recomputed': False}
        with (output / 'verification.json').open('x') as handle:
            json.dump(result, handle, indent=2)
        return result
    finally:
        sql_log.close()
        connection.close()
