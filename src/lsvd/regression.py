"""Per-user historical regression checks, independent of reported rounding."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def compare_users(new_users, new_counts, new_hits, old_users, old_counts,
                  old_hits, old_scores, k):
    """Compare fixed-query targets and scores after exact user ID alignment."""
    new_users, old_users = np.asarray(new_users), np.asarray(old_users)
    if (len(np.unique(new_users)) != len(new_users)
            or len(np.unique(old_users)) != len(old_users)
            or not np.array_equal(np.sort(new_users), np.sort(old_users))):
        raise ValueError('historical and current query user sets differ')
    new_order, old_order = np.argsort(new_users), np.argsort(old_users)
    new_counts = np.asarray(new_counts)[new_order]
    new_hits = np.asarray(new_hits)[new_order]
    old_counts = np.asarray(old_counts)[old_order]
    old_scores = np.asarray(old_scores)[old_order]
    denominator = np.minimum(k, new_counts)
    scores = np.divide(new_hits, denominator, out=np.zeros(len(new_users), dtype=np.float64),
                       where=denominator > 0)
    active = new_counts > 0
    old_active = old_counts > 0
    changed_targets = int(np.count_nonzero(new_counts != old_counts))
    changed_scores = int(np.count_nonzero(scores != old_scores))
    changed_hits = None if old_hits is None else int(np.count_nonzero(
        new_hits != np.asarray(old_hits)[old_order]))
    same = changed_targets == 0 and changed_scores == 0
    if old_hits is None and same:
        status = '目标数和逐用户分数一致，旧命中数未保存'
    elif same and changed_hits == 0:
        status = '一致'
    else:
        status = '不一致'
    return dict(status=status, k=int(k), queries=len(new_users),
                target_users=int(active.sum()), targets=int(new_counts.sum()),
                new_macro=float(scores[active].mean()) if active.any() else None,
                old_macro=float(old_scores[old_active].mean()) if old_active.any() else None,
                changed_targets=changed_targets, changed_hits=changed_hits,
                changed_scores=changed_scores,
                max_score_difference=float(np.max(np.abs(scores - old_scores), initial=0)))


def compare_content_evaluation(current, legacy, legacy_users, output):
    """Join round-nine user/group/K scores to current slice/K/user statistics.

    The legacy axes come from analyze_vector_quality.py evaluate(), whose
    assignment is X[user, group, ki]. Group order is all, seen, unseen.
    """
    current, legacy = Path(current), Path(legacy)
    specs = json.loads((current / 'slice_specs.json').read_text())
    old_scores = np.load(legacy / 'metrics_nprobe64.npy', mmap_mode='r', allow_pickle=False)
    old_counts = np.load(legacy / 'target_counts_nprobe64.npy', mmap_mode='r', allow_pickle=False)
    old_users = np.load(legacy_users, mmap_mode='r', allow_pickle=False)
    legacy_ks = (50, 200, 1000)
    rows = []
    with np.load(current / 'statistics.npz', allow_pickle=False) as archive:
        users, counts, hits, ks = (archive[name] for name in ('user_ids', 'counts', 'hits', 'ks'))
        for group_index, seen_scope in enumerate(('all', 'seen', 'unseen')):
            slice_id = next(spec['slice_id'] for spec in specs
                            if spec['ruler'] == 'main' and spec['place_scope'] == 'all'
                            and spec['seen_scope'] == seen_scope and spec['breakdown'] == 'all')
            for k_index, k in enumerate(ks):
                legacy_k_index = legacy_ks.index(int(k))
                row = compare_users(users, counts[slice_id], hits[slice_id, k_index],
                                    old_users, old_counts[:, group_index], None,
                                    old_scores[:, group_index, legacy_k_index], int(k))
                rows.append(dict(model='content_mean', place_scope='all', seen_scope=seen_scope,
                                 **row, source=str(legacy), legacy_axes='user,target_group,K'))
    with Path(output).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description='核对第九轮内容均值逐用户结果')
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--legacy-users', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    rows = compare_content_evaluation(args.current, args.legacy, args.legacy_users, args.output)
    same = all(row['changed_targets'] == row['changed_scores'] == 0 for row in rows)
    print(json.dumps({'status': '一致' if same else '不一致', 'rows': len(rows)}, ensure_ascii=False))
    return 0 if same else 1


if __name__ == '__main__':
    raise SystemExit(main())
