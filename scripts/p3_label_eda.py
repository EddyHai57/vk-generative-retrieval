"""Original bounded train-only EDA. Outputs aggregates, never IDs or raw rows.

Exploration grid is fixed before reading data; it does not replace p3-like-v1.
No model, training, dependency installation, or holdout reader is provided.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

FEEDBACK = ('like', 'dislike', 'share', 'bookmark', 'click_on_author', 'open_comments')
SECONDS = (5, 10, 20)
RATIOS = (0.5, 0.8, 1.0)
DURATION_BINS = ('1-10', '11-20', '21-40', '41-60', '61-120', '121-255')


def candidate_masks(watch, duration, like, share, bookmark):
    """Do not remove dislike conflicts or clip ratio; duration zero is excluded."""
    masks = {'like': like, 'like_or_share_or_bookmark': like | share | bookmark}
    masks.update({f'seconds_ge_{s}': watch >= s for s in SECONDS})
    masks.update({f'ratio_ge_{r:.1f}': (duration > 0) & (watch >= r * duration) for r in RATIOS})
    return masks


def hist_summary(hist):
    """Nearest-rank quantiles and exact event share of highest-count users."""
    pairs = sorted((int(k), int(v)) for k, v in hist.items())
    total = sum(v for _, v in pairs)
    events = sum(k * v for k, v in pairs)
    result = {'users': total, 'events': events, 'zero_users': dict(pairs).get(0, 0),
              'users_gt_50': sum(v for k, v in pairs if k > 50),
              'snapshot_events_beyond_last50': sum((k - 50) * v for k, v in pairs if k > 50)}
    for percentile in (50, 90, 95, 99):
        rank = math.ceil(total * percentile / 100)
        cumulative = 0
        for k, v in pairs:
            cumulative += v
            if cumulative >= rank:
                result[f'p{percentile}'] = k
                break
    for frac, key in ((0.01, 'top_1pct_users_event_share'), (0.1, 'top_10pct_users_event_share')):
        remaining = math.ceil(total * frac)
        contribution = 0
        for k, v in reversed(pairs):
            take = min(remaining, v)
            contribution += take * k
            remaining -= take
            if not remaining:
                break
        result[key] = contribution / events if events else None
    return result


def numeric_summary(hist, sample_unit, value_unit):
    """Numeric sample quantiles; deliberately omit user/history-only meanings."""
    summary = hist_summary(hist)
    values = [int(k) for k, count in hist.items() if count]
    return {'sample_count': summary['users'], 'sample_unit': sample_unit, 'value_unit': value_unit,
            'min': min(values), 'max': max(values),
            **{f'p{p}': summary[f'p{p}'] for p in (50,90,95,99)}}


def repair_summaries(result, existing):
    """Reproduce a summary-only repair from prior aggregate; no raw reader."""
    result['watch_summary'] = numeric_summary(existing['audit']['timespent_histogram'], 'interaction', 'seconds')
    result['duration_item_summary'] = numeric_summary(existing['features']['duration_histogram'], 'catalog_item', 'seconds')
    for name in ('exposure_summary', 'like_summary'):
        old = result[name]
        if 'events_dropped_by_last50' in old:
            old['snapshot_events_beyond_last50'] = old.pop('events_dropped_by_last50')
    for value in result['candidates'].values():
        old = value['history_summary']
        if 'events_dropped_by_last50' in old:
            old['snapshot_events_beyond_last50'] = old.pop('events_dropped_by_last50')
    prefix_hist = Counter()
    for count, users in existing['audit']['positive_length_histogram'].items():
        prefix_hist[max(int(count)-1, 0)] += users
    result['like_training_prefix_contribution'] = hist_summary(prefix_hist)
    return result


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def memory_gate():
    """Use the stricter visible host/cgroup headroom; do not assume host RAM is quota."""
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    total, available = values['MemTotal'], values['MemAvailable']
    for limit_path, current_path in (('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                                    ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        if Path(limit_path).exists():
            raw = Path(limit_path).read_text().strip()
            if raw != 'max':
                limit = int(raw)
                if limit < total:
                    total, available = limit, limit - int(Path(current_path).read_text())
    if available / total < 0.2:
        raise ValueError('RAM headroom below 20% stop gate')
    return {'effective_total_bytes': total, 'available_bytes': available, 'headroom_fraction': available / total}


def positions(sorted_ids, selected):
    index = np.searchsorted(sorted_ids, selected)
    safe = np.minimum(index, len(sorted_ids)-1)
    match = (index < len(sorted_ids)) & (sorted_ids[safe] == selected)
    return safe, match


def segment_summary(mask, feedback, watch, duration, masks):
    n = int(mask.sum())
    return {'events': n, 'feedback': {k: int((mask & v).sum()) for k, v in feedback.items()},
            'watch_sum': int(watch[mask].sum()), 'duration_sum': int(duration[mask].sum()),
            'timespent_255': int((mask & (watch == 255)).sum()),
            'ratio_gt_1': int((mask & (duration > 0) & (watch > duration)).sum()),
            'candidates': {k: int((mask & v).sum()) for k, v in masks.items()}}


def merge_stats(target, update):
    for k, v in update.items():
        if isinstance(v, dict):
            merge_stats(target.setdefault(k, {}), v)
        else:
            target[k] = target.get(k, 0) + v


def run(raw_root, audit_path, output):
    import pyarrow.parquet as pq
    started = time.monotonic()
    raw_root, audit_path, output = Path(raw_root), Path(audit_path), Path(output)
    if output.exists():
        raise FileExistsError('exclusive output already exists')
    existing = json.loads(audit_path.read_text())
    if sha(audit_path) != '3cc2d3ffb9812d7e83709f53d85463876f9be13daf4173829b748dbe1ee1c3fe':
        raise ValueError('existing train audit hash differs')
    catalog_path, users_path = audit_path.parent/'catalog.json', audit_path.parent/'train-users.json'
    if sha(catalog_path) != existing['catalog_sha256'] or sha(users_path) != existing['train_users_sha256']:
        raise ValueError('existing catalog/user binding differs')
    catalog = np.array(json.loads(catalog_path.read_text()), dtype=np.uint32)
    users = np.array(json.loads(users_path.read_text()), dtype=np.uint32)
    if np.any(catalog[1:] <= catalog[:-1]) or np.any(users[1:] <= users[:-1]):
        raise ValueError('catalog/users not unique sorted')
    resources = [memory_gate()]
    duration = np.zeros(len(catalog), dtype=np.int16)
    matched = np.zeros(len(catalog), dtype=bool)
    metadata = {}
    input_files = []
    for name, ids, id_col in (('items_metadata.parquet', catalog, 'item_id'), ('users_metadata.parquet', users, 'user_id')):
        path = raw_root/'metadata'/name
        reader = pq.ParquetFile(path)
        schema = {f.name: str(f.type) for f in reader.schema_arrow}
        expected_schema = ({'item_id':'uint32','author_id':'uint32','duration':'uint8','train_interactions_rank':'uint32'} if id_col == 'item_id' else {'user_id':'uint32','age':'uint8','gender':'uint8','geo':'uint8','train_interactions_rank':'uint32'})
        if schema != expected_schema:
            raise ValueError('metadata schema drift')
        columns = list(schema) if id_col == 'user_id' else ['item_id', 'duration']
        seen = np.zeros(len(ids), dtype=bool)
        nulls = Counter()
        for batch in reader.iter_batches(batch_size=65536, columns=columns, use_threads=False):
            resources.append(memory_gate())
            index, mask = positions(ids, batch.column(id_col).to_numpy())
            selected_index = index[mask]
            if len(np.unique(selected_index)) != len(selected_index) or seen[selected_index].any():
                raise ValueError('duplicate selected metadata ID')
            seen[selected_index] = True
            for col in columns:
                nulls[col] += batch.column(col).filter(__import__('pyarrow').array(mask)).null_count
            if id_col == 'item_id':
                arr = batch.column('duration')
                if arr.null_count:
                    raise ValueError('item duration null stop gate')
                duration[selected_index] = arr.to_numpy()[mask]
        metadata[name] = {'schema': schema, 'total_rows': reader.metadata.num_rows,
                          'selected_entities': len(ids), 'matched': int(seen.sum()), 'selected_nulls': dict(nulls)}
        if not seen.all() or any(nulls.values()):
            raise ValueError('metadata coverage/null stop gate')
        if id_col == 'item_id':
            matched = seen
        input_files.append({'path': 'metadata/'+name, 'sha256': sha(path), 'bytes': path.stat().st_size})
    if dict(Counter(map(int, duration))) != {int(k):v for k,v in existing['features']['duration_histogram'].items()}:
        raise ValueError('existing item duration histogram mismatch')
    templates = candidate_masks(np.array([0]), np.array([1]), np.array([False]), np.array([False]), np.array([False]))
    counters = {name: {'user_counts': np.zeros(len(users), dtype=np.int32), 'items': np.zeros(len(catalog), dtype=bool),
                       'feedback': Counter(), 'events': 0} for name in templates}
    exposures = np.zeros(len(users), dtype=np.int32)
    observed_items = np.zeros(len(catalog), dtype=bool)
    feedback_total, cofeedback, watch_hist = Counter(), Counter(), Counter()
    nulls = Counter()
    segments = {'week': {}, 'place': {}, 'platform': {}, 'agent': {}, 'duration_bin': {}}
    diagnostics = Counter()
    for week in range(25):
        relative = f'subsamples/ur0.01_ir0.01/train/week_{week:02}.parquet'
        path = raw_root/relative
        expected = next(r for r in existing['train_files'] if r['path'] == relative)
        actual_sha = sha(path)
        if actual_sha != expected['sha256']:
            raise ValueError('train input hash drift')
        reader = pq.ParquetFile(path)
        if {f.name: str(f.type) for f in reader.schema_arrow} != existing['schema']:
            raise ValueError('interaction schema drift')
        input_files.append({'path': relative, 'sha256': actual_sha, 'bytes': path.stat().st_size})
        for batch in reader.iter_batches(batch_size=65536, use_threads=False):
            resources.append(memory_gate())
            for name in batch.schema.names:
                nulls[name] += batch.column(name).null_count
            if any(nulls.values()):
                raise ValueError('interaction null stop gate')
            rows = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in batch.schema.names}
            uidx, uok = positions(users, rows['user_id'])
            iidx, iok = positions(catalog, rows['item_id'])
            if not uok.all() or not iok.all():
                raise ValueError('train catalog/user mismatch')
            exposures += np.bincount(uidx, minlength=len(users)).astype(np.int32)
            observed_items[iidx] = True
            watch = rows['timespent'].astype(np.int16)
            dur = duration[iidx]
            fb = {k: rows[k] for k in FEEDBACK}
            masks = candidate_masks(watch, dur, fb['like'], fb['share'], fb['bookmark'])
            diagnostics['events'] += len(watch)
            diagnostics['timespent_255'] += int((watch == 255).sum())
            diagnostics['duration_255_events'] += int((dur == 255).sum())
            diagnostics['duration_zero_events'] += int((dur == 0).sum())
            diagnostics['ratio_gt_1'] += int(((dur > 0) & (watch > dur)).sum())
            diagnostics['ratio_eq_1'] += int(((dur > 0) & (watch == dur)).sum())
            watch_hist.update(dict(zip(*[x.tolist() for x in np.unique(watch, return_counts=True)])))
            feedback_total.update({k: int(v.sum()) for k, v in fb.items()})
            for j, a in enumerate(FEEDBACK):
                for b in FEEDBACK[j+1:]:
                    cofeedback[f'{a}&{b}'] += int((fb[a] & fb[b]).sum())
            for name, mask in masks.items():
                c = counters[name]
                c['events'] += int(mask.sum())
                c['user_counts'] += np.bincount(uidx[mask], minlength=len(users)).astype(np.int32)
                c['items'][iidx[mask]] = True
                c['feedback'].update({k: int((mask & v).sum()) for k, v in fb.items()})
            all_mask = np.ones(len(watch), dtype=bool)
            merge_stats(segments['week'].setdefault(str(week), {}), segment_summary(all_mask, fb, watch, dur, masks))
            for field in ('place', 'platform', 'agent'):
                for value in np.unique(rows[field]):
                    mask = rows[field] == value
                    merge_stats(segments[field].setdefault(str(int(value)), {}), segment_summary(mask, fb, watch, dur, masks))
            labels = np.digitize(dur, (10,20,40,60,120), right=True)
            for b, label in enumerate(DURATION_BINS):
                mask = (labels == b) & (dur > 0)
                merge_stats(segments['duration_bin'].setdefault(label, {}), segment_summary(mask, fb, watch, dur, masks))
    candidates = {}
    for name, c in counters.items():
        counts = c['user_counts']
        candidates[name] = {'events': c['events'], 'users': int((counts > 0).sum()), 'items': int(c['items'].sum()),
                            'users_ge_2': int((counts >= 2).sum()), 'feedback': dict(c['feedback']),
                            'history_summary': hist_summary(dict(zip(*[x.tolist() for x in np.unique(counts, return_counts=True)])))}
    aggregate = existing['audit']
    checks = {'rows': diagnostics['events'] == aggregate['rows'] == 3720328,
              'users': int((exposures > 0).sum()) == aggregate['users'] == 89371,
              'items': int(observed_items.sum()) == aggregate['items'] == 120415,
              'likes': candidates['like']['events'] == aggregate['feedback']['like']['true'] == 105700,
              'like_users_ge_2': candidates['like']['users_ge_2'] == aggregate['users_ge_2_positive'] == 10863,
              'exposure_histogram': dict(zip(*[x.tolist() for x in np.unique(exposures, return_counts=True)])) == {int(k):v for k,v in aggregate['exposure_length_histogram'].items()},
              'watch_histogram': dict(watch_hist) == {int(k):v for k,v in aggregate['timespent_histogram'].items()},
              'all_feedback': all(feedback_total[k] == aggregate['feedback'][k]['true'] for k in FEEDBACK)}
    if not all(checks.values()):
        raise ValueError('existing train count reconciliation failed')
    result = {'task_id': 'P3', 'scope': 'supplemental train-only EDA; frozen p3-like-v1 unchanged',
              'utc': datetime.now(timezone.utc).isoformat(), 'source_revision': existing['source_revision'],
              'script_sha256': sha(__file__), 'existing_train_audit_sha256': sha(audit_path),
              'exploration_grid': {'seconds_ge': SECONDS, 'ratio_ge': RATIOS, 'ratio_zero_duration': 'excluded',
                                   'ratio_gt1': 'retained; no clipping', 'dislike_conflicts': 'retained and reported'},
              'schema': existing['schema'], 'null_counts': dict(nulls), 'metadata': metadata, 'input_files': input_files,
              'diagnostics': dict(diagnostics), 'feedback': dict(feedback_total), 'pairwise_cofeedback': dict(cofeedback),
              'exposure_summary': hist_summary(aggregate['exposure_length_histogram']),
              'like_summary': hist_summary(aggregate['positive_length_histogram']),
              'watch_summary': numeric_summary(aggregate['timespent_histogram'], 'interaction', 'seconds'),
              'duration_item_summary': numeric_summary(existing['features']['duration_histogram'], 'catalog_item', 'seconds'),
              'candidates': candidates, 'segments': segments, 'reconciliation': checks,
              'elapsed_seconds': time.monotonic()-started, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'minimum_ram_headroom_fraction': min(x['headroom_fraction'] for x in resources),
              'effective_ram_bytes': resources[0]['effective_total_bytes']}
    repair_summaries(result, existing)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({'output_sha256': sha(output), 'reconciliation': checks,
                      'elapsed_seconds': result['elapsed_seconds'], 'peak_rss_kib': result['peak_rss_kib']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--train-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.raw_root, args.train_audit, args.output)
