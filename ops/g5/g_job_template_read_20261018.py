"""Read-only: print the claimed queue job specs of the accepted G anchor verify jobs (job metadata only; stdout).

Reads only the claimed receipt JSON files named by exact run id (small files), never logs, checkpoints, archives, predictions or test.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
RUN_IDS = ('g-coll-a1-c1-s42-verify', 'g-coll-a1-c1-s2024-verify', 'g-coll-a1-c1-s2025-verify')
MAX = 65536


def main():
    out = {}
    for rid in RUN_IDS:
        found = []
        for path in sorted((ROOT / 'queue/claimed').glob('*-' + rid + '.json')):
            if path.is_symlink() or path.stat().st_size > MAX:
                found.append({'path': str(path), 'read': False}); continue
            payload = path.read_bytes(); found.append({'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(), 'value': json.loads(payload)})
        out[rid] = found
    print(json.dumps({'jobs': out, 'boundaries': {'archive_read': False, 'checkpoint_read': False, 'test_read': False}}, sort_keys=True))


if __name__ == '__main__':
    main()
