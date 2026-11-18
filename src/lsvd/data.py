"""Build shared, immutable VK-LSVD tables from explicit weekly inputs."""
from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import resource
import time

import duckdb
import numpy as np

BLOCKS = 32
DISK_LIMIT = 30_000_000_000
FEEDBACK_FIELDS = (
    'main_positive', 'secondary_positive', 'quick_skip', 'like_positive',
    'share', 'bookmark', 'open_comments', 'click_on_author', 'dislike',
)
SEQUENCE_FIELDS = {
    'item_ids': 'item_id', 'authors': 'author_id', 'weeks': 'week',
    'ordinals': 'ordinal', 'places': 'place', 'platforms': 'platform',
    'agents': 'agent', 'timespents': 'timespent', 'durations': 'duration',
    'main_positive': 'main_positive', 'secondary_positive': 'secondary_positive',
    'quick_skip': 'quick_skip', 'like_positive': 'like_positive',
    'share': 'share', 'bookmark': 'bookmark', 'open_comments': 'open_comments',
    'click_on_author': 'click_on_author', 'dislike': 'dislike',
    'watch_ratios': 'watch_ratio', 'over_duration': 'over_duration',
    'at_cap': 'at_cap', 'feedback_masks': 'feedback_mask',
}


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _literal(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _write_json(path: Path, value: dict) -> None:
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _disk_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())


def _frozen_integer(value: str) -> int:
    number = Decimal(value)
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f'frozen threshold value is not an exact integer: {value}')
    return int(number)


def frozen_expressions(path: Path) -> dict[str, str]:
    """Extract the balanced expressions immediately before frozen aliases."""
    sql = Path(path).read_text()
    expressions = {}
    for alias, name in [('flat25', 'main_positive'), ('flat50', 'secondary_positive')]:
        marker = ' ' + alias
        position = sql.find(marker)
        if position < 0:
            raise ValueError(f'missing frozen label alias {alias}')
        end = position
        while end and sql[end - 1].isspace():
            end -= 1
        depth = 0
        start = None
        for index in range(end - 1, -1, -1):
            if sql[index] == ')':
                depth += 1
            elif sql[index] == '(':
                depth -= 1
                if depth == 0:
                    start = index
                    break
        if start is None or not sql[start:end].startswith('(NOT e.dislike'):
            raise ValueError(f'invalid frozen expression for {alias}')
        expressions[name] = sql[start:end]
    return expressions


def popularity_tiers(counts: np.ndarray, p80: int, p95: int) -> np.ndarray:
    counts = np.asarray(counts)
    return np.select(
        [counts == 0, counts >= p95, counts >= p80],
        [0, 3, 2], default=1,
    ).astype(np.uint8)


def sequence_select_sql(relation: str, condition: str) -> str:
    """Sort each history once, sharing the ordered struct across all columns."""
    packed = ','.join(f'{name}:={source}' for name, source in SEQUENCE_FIELDS.items())
    lists = ','.join(f'list_transform(ordered_events,event->event.{name}) AS {name}'
                     for name in SEQUENCE_FIELDS)
    return f'''WITH grouped AS(
SELECT user_id,list(struct_pack({packed}) ORDER BY week,ordinal) AS ordered_events
FROM {relation} WHERE {condition} GROUP BY user_id)
SELECT user_id,{lists} FROM grouped ORDER BY user_id'''


def build_foundation(
    raw_root: Path,
    frozen_root: Path,
    output: Path,
    *,
    train_weeks: tuple[int, ...] = tuple(range(25)),
    memory_limit: str = '8GB',
    threads: int = 4,
    disk_limit: int = DISK_LIMIT,
    reuse_master: Path | None = None,
) -> dict:
    raw_root, frozen_root, output = map(Path, (raw_root, frozen_root, output))
    if not train_weeks or len(set(train_weeks)) != len(train_weeks) or any(week < 0 or week > 24 for week in train_weeks):
        raise ValueError('training weeks must be distinct values from 00 through 24')
    files = [(week, raw_root / 'train' / f'week_{week:02}.parquet') for week in sorted(train_weeks)]
    files.append((25, raw_root / 'validation' / 'week_25.parquet'))
    metadata_path = raw_root / 'metadata' / 'items_metadata.parquet'
    csv_path = frozen_root / 'training_thresholds_and_hash_cutoffs.csv'
    sql_path = frozen_root / 'label_select.sql'
    expressions = frozen_expressions(sql_path)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    master = Path(reuse_master) if reuse_master is not None else output / 'master'
    sequences = output / 'sequences'
    arrays = output / 'arrays'
    if reuse_master is None:
        master.mkdir()
    sequences.mkdir()
    arrays.mkdir()
    log = (output / 'executed.sql').open('x')
    connection = duckdb.connect(config={
        'memory_limit': memory_limit, 'threads': threads,
        'temp_directory': str(output / 'tmp'),
    })
    stages = []

    def query(sql):
        log.write(sql + ';\n\n')
        log.flush()
        return connection.execute(sql)

    def checkpoint(stage, **values):
        used = _disk_bytes(output.parent)
        row = {'stage': stage, 'seconds': time.monotonic() - start,
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'new_bytes': used, **values}
        stages.append(row)
        print(json.dumps(row), flush=True)
        if used > disk_limit:
            raise RuntimeError(f'new output bytes exceed disk limit {disk_limit}')

    try:
        query('SET preserve_insertion_order=false')
        query(f'CREATE TABLE metadata AS SELECT * FROM read_parquet({_literal(metadata_path)})')
        n_items, unique_items = query('SELECT count(*),count(DISTINCT item_id) FROM metadata').fetchone()
        if n_items != unique_items:
            raise ValueError('metadata contains duplicate item identifiers')
        with csv_path.open() as handle:
            thresholds = list(csv.DictReader(handle))
        columns = ['min_d', 'q25', 'need25', 'cut25_20261105', 'q50', 'need50', 'cut50_20261105']
        values = []
        for row in thresholds:
            fields = [str(_frozen_integer(row[column])) + ('::UBIGINT' if column.startswith('cut') else '') for column in columns]
            values.append('(' + ','.join(fields) + ')')
        threshold_statement = 'CREATE TABLE thresholds AS SELECT * FROM(VALUES ' + ','.join(values) + ') t(' + ','.join(columns) + ')'
        if reuse_master is None:
            query(threshold_statement)
            if query('SELECT count(*)<>count(DISTINCT min_d) FROM thresholds').fetchone()[0]:
                raise ValueError('frozen thresholds contain duplicate durations')
        else:
            source_sql = (master.parent / 'executed.sql').read_text()
            if threshold_statement not in source_sql or any(expression not in source_sql for expression in expressions.values()):
                raise ValueError('reused master has different frozen threshold or label SQL bindings')
            if any(_literal(path) not in source_sql for _, path in files):
                raise ValueError('reused master has different raw weekly input bindings')
        source_rows = []
        for week, path in files:
            if reuse_master is not None:
                count = query(f'SELECT num_rows FROM parquet_file_metadata({_literal(path)})').fetchone()[0]
                source_rows.append({'week': week, 'rows': int(count)})
                continue
            if _disk_bytes(output.parent) + path.stat().st_size * 4 > disk_limit:
                raise RuntimeError('insufficient output disk budget for the next weekly block')
            query(f'''CREATE OR REPLACE TEMP VIEW raw_week AS
SELECT e.* EXCLUDE(file_row_number),{week}::UTINYINT AS week,
e.file_row_number::UINTEGER AS ordinal,i.author_id,i.duration
FROM read_parquet({_literal(path)},file_row_number=true) e
LEFT JOIN metadata i USING(item_id)''')
            count, invalid = query('''SELECT count(*),count(*) FILTER(WHERE
user_id IS NULL OR item_id IS NULL OR author_id IS NULL OR duration IS NULL
OR duration NOT BETWEEN 5 AND 180 OR timespent IS NULL OR "like" IS NULL
OR dislike IS NULL OR share IS NULL OR bookmark IS NULL OR open_comments IS NULL
OR click_on_author IS NULL OR place IS NULL OR platform IS NULL OR agent IS NULL)
FROM raw_week''').fetchone()
            if invalid:
                raise ValueError(f'week {week} has {invalid} invalid rows; original input retained')
            missing = query('SELECT count(*) FROM raw_week e ANTI JOIN thresholds f ON e.duration=f.min_d').fetchone()[0]
            if missing:
                raise ValueError(f'week {week} has {missing} rows without frozen thresholds')
            query(f'''CREATE OR REPLACE TEMP VIEW labeled_week AS SELECT e.*,
({expressions['main_positive']})::UTINYINT AS main_positive,
({expressions['secondary_positive']})::UTINYINT AS secondary_positive,
e."like"::UTINYINT AS like_positive,(e.timespent<3)::UTINYINT AS quick_skip,
e.timespent::DOUBLE/e.duration AS watch_ratio,
(e.timespent>e.duration)::UTINYINT AS over_duration,(e.timespent=255)::UTINYINT AS at_cap
FROM raw_week e JOIN thresholds f ON e.duration=f.min_d''')
            mask = '+'.join(f'{field}::USMALLINT*{1 << index}' for index, field in enumerate(FEEDBACK_FIELDS))
            query(f'''COPY (SELECT *,((place IN(0,1)) AND main_positive=1)::UTINYINT AS target_main,
CASE WHEN week<=24 AND place IN(0,1) THEN CASE WHEN main_positive=1 THEN 1
WHEN secondary_positive=1 THEN 0.25 ELSE 0 END ELSE 0 END::FLOAT AS train_weight,
({mask})::USMALLINT AS feedback_mask,(user_id%32)::UTINYINT AS user_block
FROM labeled_week) TO {_literal(master)} (FORMAT PARQUET,PARTITION_BY(user_block),
APPEND,FILENAME_PATTERN 'week_{week:02}_{{uuid}}',COMPRESSION ZSTD,COMPRESSION_LEVEL 3)''')
            source_rows.append({'week': week, 'rows': int(count)})
            checkpoint('master_week', week=week, rows=count)

        master_glob = master / '*' / '*.parquet'
        query(f'CREATE VIEW events AS SELECT * FROM read_parquet({_literal(master_glob)},hive_partitioning=true)')
        actual_rows = query('SELECT week,count(*) FROM events GROUP BY week ORDER BY week').fetchall()
        if actual_rows != [(row['week'], row['rows']) for row in source_rows]:
            raise ValueError('master row counts differ from original inputs')
        query('CREATE TABLE train_items AS SELECT item_id,count(*)::UINTEGER AS train_count FROM events WHERE week<=24 GROUP BY item_id')
        p80, p95 = query('SELECT quantile_disc(train_count,0.8),quantile_disc(train_count,0.95) FROM train_items').fetchone()
        query(f'''CREATE TABLE items AS SELECT m.*,coalesce(t.train_count,0)::UINTEGER AS train_count,
CASE WHEN t.train_count IS NULL THEN 0 WHEN t.train_count>={int(p95)} THEN 3
WHEN t.train_count>={int(p80)} THEN 2 ELSE 1 END::UTINYINT AS popularity_tier
FROM metadata m LEFT JOIN train_items t USING(item_id)''')
        query(f'COPY items TO {_literal(output / "items.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        query('CREATE TABLE validation_users AS SELECT DISTINCT user_id FROM events WHERE week=25')
        for block in range(BLOCKS):
            directory = sequences / f'user_block={block}'
            directory.mkdir()
            select = sequence_select_sql('events', f'week<=24 AND user_block={block}')
            query(f'''COPY ({select}) TO {_literal(directory / 'users.parquet')}
(FORMAT PARQUET,COMPRESSION ZSTD)''')
            checkpoint('sequences_block', user_block=block)

        sequence_glob = sequences / '*' / '*.parquet'
        query(f'CREATE VIEW sequences AS SELECT * FROM read_parquet({_literal(sequence_glob)})')
        query('''CREATE TABLE history_index AS SELECT user_id,
row_number() OVER(ORDER BY user_id)-1 AS history_row,len(item_ids)::UINTEGER AS history_count,
list_extract(item_ids,-1)::BIGINT AS last_item_id,list_extract(authors,-1)::BIGINT AS last_author_id
FROM sequences''')
        query('''CREATE TABLE queries AS SELECT v.user_id,(v.user_id%32)::UTINYINT AS history_block,
coalesce(h.history_row,-1)::BIGINT AS history_row,coalesce(h.history_count,0)::UINTEGER AS history_count,
coalesce(h.last_item_id,-1)::BIGINT AS last_item_id,coalesce(h.last_author_id,-1)::BIGINT AS last_author_id
FROM validation_users v LEFT JOIN history_index h USING(user_id) ORDER BY v.user_id''')
        query(f'COPY queries TO {_literal(output / "queries.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        query('''CREATE VIEW targets AS SELECT e.user_id,e.item_id,e.place,e.duration,
e.main_positive,e.secondary_positive,e.like_positive,(i.train_count>0) AS train_seen,
CASE WHEN e.duration<=60 THEN 0 WHEN e.duration<=120 THEN 1 ELSE 2 END::UTINYINT AS duration_band,
CASE WHEN e.duration<=19 THEN 0 WHEN e.duration<=53 THEN 1 ELSE 2 END::UTINYINT AS duration_band_sensitive,
i.popularity_tier FROM events e JOIN items i USING(item_id)
WHERE week=25 AND(main_positive=1 OR secondary_positive=1 OR like_positive=1)''')
        query(f'COPY (SELECT * FROM targets ORDER BY user_id,item_id) TO {_literal(output / "targets.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        duplicates = query('SELECT count(*)-count(DISTINCT(user_id,item_id)) FROM targets').fetchone()[0]
        if duplicates:
            raise ValueError('validation target user-item pairs are not unique')
        checkpoint('validation_views')

        queries = query('SELECT * FROM queries ORDER BY user_id').fetchnumpy()
        np.savez(arrays / 'queries.npz', **queries)
        target_arrays = query('SELECT * FROM targets ORDER BY user_id,item_id').fetchnumpy()
        positions = np.searchsorted(queries['user_id'], target_arrays.pop('user_id'))
        target_counts = np.bincount(positions, minlength=len(queries['user_id']))
        np.savez(arrays / 'targets.npz', offsets=np.concatenate(([0], np.cumsum(target_counts))), **target_arrays)
        item_arrays = query('SELECT item_id,author_id,duration,train_count,popularity_tier FROM items ORDER BY item_id').fetchnumpy()
        np.savez(arrays / 'items.npz', **item_arrays)
        del target_arrays, item_arrays
        history = query('SELECT user_id,history_count FROM history_index ORDER BY user_id').fetchnumpy()
        offsets = np.concatenate(([0], np.cumsum(history['history_count'], dtype=np.int64)))
        np.savez(arrays / 'history.npz', user_id=history['user_id'], offsets=offsets)
        flat_history = np.lib.format.open_memmap(arrays / 'history_items.npy', mode='w+', dtype=np.uint32, shape=(int(offsets[-1]),))
        for block in range(BLOCKS):
            sequence_path = sequences / f'user_block={block}' / 'users.parquet'
            rows = query(f'SELECT user_id,item_ids FROM read_parquet({_literal(sequence_path)}) ORDER BY user_id').fetchnumpy()
            global_rows = np.searchsorted(history['user_id'], rows['user_id'])
            for index, global_row in enumerate(global_rows):
                values = np.sort(np.asarray(rows['item_ids'][index], dtype=np.uint32))
                if len(values) > 1 and np.any(values[1:] == values[:-1]):
                    raise ValueError(f'duplicate training user-item for user {rows["user_id"][index]}')
                flat_history[offsets[global_row]:offsets[global_row + 1]] = values
        flat_history.flush()
        del flat_history
        checkpoint('shared_arrays')

        tier_rows = query('''SELECT popularity_tier,count(*) AS videos,sum(train_count) AS train_exposures
FROM items GROUP BY popularity_tier ORDER BY popularity_tier''').fetchall()
        target_tier_rows = query('''SELECT popularity_tier,count(*) FILTER(WHERE main_positive=1) AS all_place_main_targets,
count(*) FILTER(WHERE main_positive=1 AND place IN(0,1)) AS place01_main_targets
FROM targets GROUP BY popularity_tier ORDER BY popularity_tier''').fetchall()
        target_lookup = {int(row[0]): row[1:] for row in target_tier_rows}
        train_rows = sum(row['rows'] for row in source_rows if row['week'] <= 24)
        all_main = query('SELECT count(*) FROM targets WHERE main_positive=1').fetchone()[0]
        place_main = query('SELECT count(*) FROM targets WHERE main_positive=1 AND place IN(0,1)').fetchone()[0]
        diagnostics = []
        for tier, videos, exposures in tier_rows:
            all_targets, main_targets = target_lookup.get(int(tier), (0, 0))
            diagnostics.append({'tier': int(tier), 'videos': int(videos), 'train_exposures': int(exposures),
                                'train_exposure_share': exposures / train_rows,
                                'all_place_main_targets': int(all_targets), 'all_place_target_share': all_targets / all_main if all_main else None,
                                'place01_main_targets': int(main_targets), 'place01_target_share': main_targets / place_main if place_main else None})
        with (output / 'popularity_diagnostics.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
            writer.writeheader()
            writer.writerows(diagnostics)
        quality = query('''SELECT count(*) AS n,sum(main_positive) AS main_positive,
sum(secondary_positive) AS secondary_positive,sum(like_positive) AS like_positive,
sum(target_main) AS target_main,count(*) FILTER(WHERE week=25 AND train_weight<>0) AS validation_nonzero_weight,
count(*) FILTER(WHERE user_block<>user_id%32) AS wrong_block FROM events''').fetchone()
        if quality[-1] or quality[-2]:
            raise ValueError('invalid validation weight or user-block mapping')
        inputs = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_sha256(path)} for path in [metadata_path, csv_path, sql_path] + [path for _, path in files]]
        manifest = {'status': 'complete', 'train_rows': int(train_rows), 'validation_rows': int(source_rows[-1]['rows']),
                    'train_users': len(history['user_id']), 'validation_users': len(queries['user_id']), 'metadata_items': int(n_items),
                    'main_targets_all_places': int(all_main), 'main_targets_place01': int(place_main),
                    'source_rows': source_rows, 'duckdb_version': duckdb.__version__, 'numpy_version': np.__version__,
                    'memory_limit': memory_limit, 'threads': threads, 'blocks': BLOCKS,
                    'frozen_expressions': expressions, 'inputs': inputs,
                    'master_source': str(master),
                    'master_source_sql_sha256': file_sha256(master.parent / 'executed.sql'),
                    'master_files': [{'path': str(path), 'bytes': path.stat().st_size,
                                      'sha256': file_sha256(path)} for path in sorted(master.glob('*/*.parquet'))],
                    'code_sha256': file_sha256(Path(__file__)),
                    'feedback_mask_bits': list(FEEDBACK_FIELDS),
                    'popularity': {'status': '初始诊断口径', 'fit_items': 'train_seen_only', 'quantile_method': 'quantile_disc',
                                   'p80': int(p80), 'p95': int(p95), 'equality': 'hotter_layer',
                                   'tier_names': ['未见', '长尾', '中部', '头部']},
                    'history_reference': {'parquet_key': ['history_block', 'user_id'], 'history_row': 'arrays/history.npz row, -1 for empty'},
                    'validation_train_weight': 0, 'test_week26_read': False,
                    'seconds': time.monotonic() - start,
                    'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    'stages': stages}
        checkpoint('complete')
        manifest_bytes = len(json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
        if _disk_bytes(output.parent) + manifest_bytes > disk_limit:
            raise RuntimeError('insufficient output disk budget to publish foundation manifest')
        _write_json(output / 'foundation.json', manifest)
        return manifest
    except BaseException as error:
        _write_json(output / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'stages': stages})
        raise
    finally:
        log.close()
        connection.close()
