"""Original bounded train-only data-quality audit. Aggregate output only.

Does not train, change labels, delete records, read holdout values, or download.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from p3_label_eda import memory_gate, positions, sha

COUNT_BINS = ((0, 0, '0'), (1, 1, '1'), (2, 5, '2-5'), (6, 10, '6-10'),
              (11, 50, '11-50'), (51, 100, '51-100'), (101, None, '101+'))
AUDIT_SHA = '3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe'
EDA_SHA = '3ec3a384e3fa3fb8f0444a8ecb792225b2c9b60d51457e0773e7d4c9c8282239'


def summarize_counts(counts):
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1 or not len(values) or (values < 0).any():
        raise ValueError('counts must be nonempty nonnegative vector')
    ordered = np.sort(values)
    total = int(values.sum())
    result = {'entities': len(values), 'total': total, 'zero_entities': int((values == 0).sum()),
              'min': int(ordered[0]), 'max': int(ordered[-1]), 'bins': {}}
    for percentile in (50, 90, 95, 99):
        result[f'p{percentile}'] = int(ordered[math.ceil(len(values)*percentile/100)-1])
    for frac, name in ((.01, 'top_1pct_share'), (.1, 'top_10pct_share')):
        take = math.ceil(len(values)*frac)
        result[name] = int(ordered[-take:].sum()) / total if total else None
    for lo, hi, name in COUNT_BINS:
        mask = (values >= lo) & ((values <= hi) if hi is not None else True)
        result['bins'][name] = {'entities': int(mask.sum()), 'total': int(values[mask].sum())}
    return result


def summarize_users(exposure, likes, ratio_gt_one, upper_code):
    exposure, likes, ratio_gt_one, upper_code = [np.asarray(v, dtype=np.int64) for v in (exposure, likes, ratio_gt_one, upper_code)]
    if not all(v.shape == exposure.shape for v in (likes, ratio_gt_one, upper_code)):
        raise ValueError('user vector shape mismatch')
    prefixes = np.maximum(likes-1, 0)
    result = {'like_count_bins': {}, 'users_with_ratio_gt_1': int((ratio_gt_one > 0).sum()),
              'users_with_timespent_255': int((upper_code > 0).sum()),
              'prefix_total': int(prefixes.sum()), 'prefix_contribution': summarize_counts(prefixes),
              'exposures': summarize_counts(exposure), 'likes': summarize_counts(likes)}
    for lo, hi, name in COUNT_BINS:
        mask = (likes >= lo) & ((likes <= hi) if hi is not None else True)
        result['like_count_bins'][name] = {'users': int(mask.sum()), 'exposures': int(exposure[mask].sum()),
            'likes': int(likes[mask].sum()), 'prefixes': int(prefixes[mask].sum()),
            'ratio_gt_1_events': int(ratio_gt_one[mask].sum()), 'timespent_255_events': int(upper_code[mask].sum())}
    return result


def summarize_authors(author_ids, exposures, likes):
    authors, index = np.unique(np.asarray(author_ids), return_inverse=True)
    size = len(authors)
    # Exact integer accumulation: no weighted bincount floating-point conversion.
    counts = np.zeros(size, dtype=np.int64)
    events = np.zeros(size, dtype=np.int64)
    positives = np.zeros(size, dtype=np.int64)
    np.add.at(counts, index, 1)
    np.add.at(events, index, exposures)
    np.add.at(positives, index, likes)
    return {'authors': size, 'catalog_items_per_author': summarize_counts(counts),
            'exposures': summarize_counts(events), 'likes': summarize_counts(positives)}


def join_metadata(ids, batches):
    ids = np.asarray(ids)
    if not len(ids) or (ids[1:] <= ids[:-1]).any():
        raise ValueError('join IDs must be unique sorted')
    seen = np.zeros(len(ids), dtype=bool)
    result = {}
    columns = None
    for rows in batches:
        if columns is None:
            columns = set(rows)-{'id'}
        if set(rows)-{'id'} != columns:
            raise ValueError('metadata columns changed')
        idx, ok = positions(ids, rows['id'])
        idx = idx[ok]
        if len(np.unique(idx)) != len(idx) or seen[idx].any():
            raise ValueError('duplicate selected metadata ID')
        seen[idx] = True
        for name in columns:
            values = np.asarray(rows[name])
            selected = values[ok]
            if any(value is None for value in selected):
                raise ValueError('required selected metadata null')
            if name not in result:
                result[name] = np.zeros(len(ids), dtype=values.dtype)
            result[name][idx] = selected
    if not seen.all():
        raise ValueError('missing selected metadata ID')
    return result

def scan_batches(batches, users, catalog, durations, resource_check=None):
    result = {'rows': 0, 'weeks': []}
    for name, length in (('item_exposures', len(catalog)), ('item_likes', len(catalog)),
                         ('user_exposures', len(users)), ('user_likes', len(users)),
                         ('user_ratio_gt_1', len(users)), ('user_upper_code', len(users))):
        result[name] = np.zeros(length, dtype=np.int64)
    previous = -1
    for week, rows in batches:
        if type(week) is not int or week < 0 or week < previous:
            raise ValueError('week order must be nondecreasing')
        if resource_check is not None:
            resource_check()
        count = len(rows['user_id'])
        if any(np.asarray(rows[name]).shape != (count,) for name in ('user_id','item_id','like','timespent')):
            raise ValueError('interaction column shape mismatch')
        if np.asarray(rows['like']).dtype != np.bool_:
            raise ValueError('like must be bool')
        for name, upper in (('user_id',2**32), ('item_id',2**32), ('timespent',256)):
            values = np.asarray(rows[name])
            if values.dtype.kind not in 'ui' or (values < 0).any() or (values >= upper).any():
                raise ValueError('interaction invalid integer or ID')
        ui, uok = positions(users, rows['user_id'])
        ii, iok = positions(catalog, rows['item_id'])
        if not uok.all() or not iok.all():
            raise ValueError('train user/catalog join failed')
        like = rows['like']
        ratio = rows['timespent'].astype(np.int16) > durations[ii].astype(np.int16)
        for name, index in (('item_exposures',ii), ('user_exposures',ui), ('item_likes',ii[like]),
                            ('user_likes',ui[like]), ('user_ratio_gt_1',ui[ratio]),
                            ('user_upper_code',ui[rows['timespent'] == 255])):
            np.add.at(result[name], index, 1)
        if week != previous:
            result['weeks'].append({'week':week, 'rows':0, 'like_events':0})
        result['weeks'][-1]['rows'] += count
        result['weeks'][-1]['like_events'] += int(like.sum())
        result['rows'] += count
        previous = week
    return result


def verify_files(raw_root, records, resource_check=None):
    checked = {}
    for entry in records:
        name = entry['path']
        if name not in checked:
            path = Path(raw_root)/name
            checked[name] = {'path':name,'sha256':sha(path),'bytes':path.stat().st_size}
            if resource_check is not None:
                resource_check()
        expected_size = entry['bytes'] if 'bytes' in entry else entry['size_bytes']
        if checked[name]['sha256'] != entry['sha256'] or checked[name]['bytes'] != expected_size:
            raise ValueError('raw file hash/size drift')
    return [checked[name] for name in sorted(checked)]


def run(raw_root, audit_path, eda_path, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    started = time.monotonic()
    raw_root, audit_path, eda_path, output = map(Path, (raw_root,audit_path,eda_path,output))
    if output.exists():
        raise FileExistsError('exclusive output already exists')
    if sha(audit_path) != AUDIT_SHA or sha(eda_path) != EDA_SHA:
        raise ValueError('pinned prior aggregate hash differs')
    audit, eda = [json.loads(p.read_text()) for p in (audit_path,eda_path)]
    if audit['source_revision'] != '084a18386e149a454dd6dbe0123478e75c6b09ce' or eda['existing_train_audit_sha256'] != AUDIT_SHA:
        raise ValueError('source/aggregate lineage differs')
    resource_samples = []
    def check_resources():
        mem = memory_gate()
        disk = shutil.disk_usage(output.parent)
        disk_fraction = disk.free / disk.total
        if disk_fraction < .2:
            raise ValueError('disk headroom below 20%')
        resource_samples.append((mem['effective_total_bytes'],mem['headroom_fraction'],disk_fraction))
    check_resources()
    # Rehash existing content bytes; do not load or copy full embedding arrays.
    input_files = verify_files(raw_root, eda['input_files'] + audit['feature_files'], check_resources)
    selected = {}
    for name, key in (('catalog.json','catalog_sha256'),('train-users.json','train_users_sha256')):
        path = audit_path.parent/name
        if sha(path) != audit[key]:
            raise ValueError('existing entity IDs hash differs')
        values = np.asarray(json.loads(path.read_text()),dtype=np.uint32)
        if not len(values) or (values[1:] <= values[:-1]).any():
            raise ValueError('existing IDs not unique sorted')
        selected[name] = values
    catalog, users = selected['catalog.json'], selected['train-users.json']
    metadata = {}
    joined = {}
    for name, ids, id_field in (('items_metadata.parquet',catalog,'item_id'), ('users_metadata.parquet',users,'user_id')):
        path = raw_root/'metadata'/name
        reader = pq.ParquetFile(path)
        schema = {f.name:str(f.type) for f in reader.schema_arrow}
        if schema != eda['metadata'][name]['schema']:
            raise ValueError('metadata schema drift')
        nulls = Counter()
        def batches():
            for batch in reader.iter_batches(batch_size=65536,use_threads=False):
                check_resources()
                _, mask = positions(ids,batch.column(id_field).to_numpy(zero_copy_only=False))
                for field in schema:
                    nulls[field] += batch.column(field).filter(pa.array(mask)).null_count
                if any(nulls.values()):
                    raise ValueError('required train metadata null')
                yield {('id' if field == id_field else field):batch.column(field).to_numpy(zero_copy_only=False) for field in schema}
        joined[name] = join_metadata(ids,batches())
        metadata[name] = {'schema':schema,'source_rows':reader.metadata.num_rows,'selected_entities':len(ids),
                          'matched':len(ids),'selected_nulls':dict(nulls)}
    nulls = Counter()
    schema_rows = []
    def train_batches():
        for week in range(25):
            name = f'subsamples/ur0.01_ir0.01/train/week_{week:02}.parquet'
            reader = pq.ParquetFile(raw_root/name)
            schema = {f.name:str(f.type) for f in reader.schema_arrow}
            if schema != audit['schema']:
                raise ValueError('train schema drift')
            if reader.metadata.num_rows != audit['audit']['weeks'][week]['rows']:
                raise ValueError('train partition count drift')
            schema_rows.append({'week':week,'rows':reader.metadata.num_rows,'schema_matches':True})
            for batch in reader.iter_batches(batch_size=65536,use_threads=False):
                for field in schema:
                    nulls[field] += batch.column(field).null_count
                if any(nulls.values()):
                    raise ValueError('required interaction null')
                yield week, {field:batch.column(field).to_numpy(zero_copy_only=False) for field in ('user_id','item_id','like','timespent')}
    stats = scan_batches(train_batches(),users,catalog,joined['items_metadata.parquet']['duration'],check_resources)
    ue, ul = stats['user_exposures'], stats['user_likes']
    ie, il = stats['item_exposures'], stats['item_likes']
    reconciliation = {
        'rows':stats['rows'] == audit['audit']['rows'],
        'users':len(users) == audit['audit']['users'] and bool((ue > 0).all()),
        'catalog':len(catalog) == audit['audit']['items'] and bool((ie > 0).all()),
        'likes':int(ul.sum()) == audit['audit']['feedback']['like']['true'] == int(il.sum()),
        'positive_user_hist':{str(k):v for k,v in Counter(map(int,ul)).items()} == audit['audit']['positive_length_histogram'],
        'exposure_user_hist':{str(k):v for k,v in Counter(map(int,ue)).items()} == audit['audit']['exposure_length_histogram'],
        'like_items':int((il > 0).sum()) == audit['audit']['like_items'],
        'prefixes':int(np.maximum(ul-1,0).sum()) == audit['training_examples'],
        'ratio_gt_1':int(stats['user_ratio_gt_1'].sum()) == eda['diagnostics']['ratio_gt_1'],
        'timespent_255':int(stats['user_upper_code'].sum()) == eda['diagnostics']['timespent_255'],
        'weeks':stats['weeks'] == [{k:r[k] for k in ('week','rows','like_events')} for r in audit['audit']['weeks']],
        'duration_hist':{str(k):v for k,v in Counter(map(int,joined['items_metadata.parquet']['duration'])).items()} == audit['features']['duration_histogram'],
    }
    if not all(reconciliation.values()):
        raise ValueError('aggregate reconciliation failed: '+','.join(k for k,v in reconciliation.items() if not v))
    user_metadata = {field:{'min':int(values.min()),'max':int(values.max()),'distinct_codes':len(np.unique(values)),
                           'zero_count':int((values == 0).sum()),'encoding_semantics':'unknown; not interpreted'}
                     for field,values in joined['users_metadata.parquet'].items() if field in ('age','gender','geo')}
    cardinality = {'interactions':stats['rows'],'train_users':len(users),'catalog_items':len(catalog),
                   'like_events':int(ul.sum()),'like_users':int((ul > 0).sum()),'liked_items':int((il > 0).sum()),
                   'users_ge_2_likes':int((ul >= 2).sum()),'like_prefixes':int(np.maximum(ul-1,0).sum())}
    result = {'task_id':'P3-C','scope':'train-only; no deletion/label change/training/holdout read',
              'utc':datetime.now(timezone.utc).isoformat(),'source_revision':audit['source_revision'],
              'script_sha256':sha(__file__),'helper_script_sha256':sha(Path(__file__).with_name('p3_label_eda.py')),
              'prior_train_audit_sha256':AUDIT_SHA,'prior_eda_sha256':EDA_SHA,
              'catalog_sha256':audit['catalog_sha256'],'train_users_sha256':audit['train_users_sha256'],
              'input_files':input_files,'train_schema':audit['schema'],'schema_partition_rows':schema_rows,
              'interaction_nulls':dict(nulls),'metadata':metadata,'user_metadata_profile':user_metadata,
              'item_support':{'exposures':summarize_counts(ie),'likes':summarize_counts(il)},
              'author_support':summarize_authors(joined['items_metadata.parquet']['author_id'],ie,il),
              'user_support':summarize_users(ue,ul,stats['user_ratio_gt_1'],stats['user_upper_code']),
              'embedding_evidence':{'mode':'rehash file; reuse prior complete ID join/finite/norm audit',**audit['features']},
              'duplicate_evidence':{'mode':'reuse pinned complete train audit',
                  'duplicate_user_item':audit['audit']['duplicate_user_item'],'exact_duplicate_week_rows':audit['audit']['exact_duplicate_rows'],
                  'semantics':'user-item pairs global across train; exact records include week; no event-ID/timestamp semantic uniqueness guarantee'},
              'ordering_evidence':{'mode':'rehash same parquet bytes; reuse prior two-pass physical sequence digest',
                  'physical_row_sequence_sha256':audit['physical_row_sequence_sha256'],'reread_sequence_match':audit['reread_sequence_match'],
                  'timestamp_available':False,'session_available':False,'assumption':'official week/physical ordinal order'},
              'cleaning':{'raw_immutable':True,'deleted_interactions':0,'deleted_users':0,'deleted_items':0,
                  'before':cardinality,'after':dict(cardinality),'decision':'preserve all; unknown anomalies flagged; no 5-core, clipping, conflict removal'},
              'reconciliation':reconciliation,'resource':{'checks':len(resource_samples),
                  'effective_ram_bytes_min':min(r[0] for r in resource_samples),
                  'minimum_ram_headroom_fraction':min(r[1] for r in resource_samples),
                  'minimum_disk_headroom_fraction':min(r[2] for r in resource_samples),
                  'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'elapsed_seconds':time.monotonic()-started,'mode':'CPU; visible host/cgroup, not hidden quota guarantee'}}
    check_resources()
    # No success file exists until all input/count/resource gates have passed.
    with output.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',required=True)
    parser.add_argument('--train-audit',required=True)
    parser.add_argument('--eda',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    run(args.raw_root,args.train_audit,args.eda,args.output)
    print('P3-C aggregate audit complete; all reconciliation gates passed')
