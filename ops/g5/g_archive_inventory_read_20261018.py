"""Read-only metadata inventory of the G anchor runs' archives (names, sizes, manifest key names and best-epoch evaluation pointers).

Never opens shard contents (per-query items), checkpoints, vectors or test. stdout only. Reads each run's manifest.json only if it is
below 16 MiB and emits key names, the best epoch and the evaluation pointer fields (kind/path/manifest_sha256/computation_sha256)
plus the evaluation directory listing (file names and sizes).
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
RUN_IDS = ('g-coll-a1-c1-s42', 'g-coll-a1-c1-s2024', 'g-coll-a1-c1-s2025')
MAX = 16 * 1024 * 1024


def listing(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        return None
    names = sorted(directory.iterdir())[:200]
    return [(p.name, None if p.is_dir() else p.stat().st_size) for p in names]


def inventory(root=ROOT):
    out = {}
    for rid in RUN_IDS:
        run = root / 'runs' / rid / 'out'
        item = {'out_listing': listing(run)}
        manifest = run / 'manifest.json'
        if manifest.is_file() and not manifest.is_symlink() and manifest.stat().st_size <= MAX:
            payload = manifest.read_bytes(); value = json.loads(payload)
            item['manifest_sha256'] = hashlib.sha256(payload).hexdigest(); item['manifest_size'] = len(payload)
            item['top_level_keys'] = sorted(value)[:80]
            best = (value.get('best') or {}); item['best_keys'] = sorted(best) if isinstance(best, dict) else None
            item['best_epoch'] = best.get('epoch') if isinstance(best, dict) else None
            epochs = value.get('epochs') if isinstance(value.get('epochs'), list) else []
            item['epoch_count'] = len(epochs)
            if isinstance(item['best_epoch'], int) and 1 <= item['best_epoch'] <= len(epochs):
                record = epochs[item['best_epoch'] - 1]
                item['best_record_keys'] = sorted(record) if isinstance(record, dict) else None
                evaluation = record.get('evaluation') if isinstance(record, dict) else None
                if isinstance(evaluation, dict):
                    item['evaluation_pointer'] = {k: evaluation.get(k) for k in ('kind', 'path', 'manifest_sha256', 'computation_sha256')}
                    path = evaluation.get('path')
                    if isinstance(path, str):
                        resolved = Path(path)
                        item['evaluation_dir_listing'] = listing(resolved if resolved.is_absolute() else run / resolved)
            binding = value.get('binding') if isinstance(value.get('binding'), dict) else {}
            item['binding_keys'] = sorted(binding)[:60]
        out[rid] = item
    return {'runs': out, 'boundaries': {'shard_content_read': False, 'checkpoint_read': False, 'test_read': False}}


if __name__ == '__main__':
    print(json.dumps(inventory(), sort_keys=True))
