"""Read-only status of every G5 export-stage run (names starting with g5-p-top100): queue state, terminal.json, small stdout log.

Reads only terminal.json, owner/progress-free fields and logs/stdout.log (<= 64 KiB, parsed to its last JSON line). Never opens
predictions, checkpoints or test. stdout only.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
PREFIX = 'g5-p-top100'
MAX = 65536


def small(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    if path.stat().st_size > MAX:
        return {'oversize': path.stat().st_size}
    payload = path.read_bytes()
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'text': payload.decode('utf-8', 'replace')}


def last_json(text):
    for line in reversed([x for x in text.splitlines() if x.strip()]):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def status(root=ROOT):
    out = {}
    names = sorted(p.name for p in (root / 'runs').glob(PREFIX + '*'))
    queue = {sub: sorted(p.name for p in (root / 'queue' / sub).glob('*' + PREFIX + '*')) for sub in ('pending', 'claimed', 'rejected')}
    for name in names:
        run = root / 'runs' / name
        terminal = small(run / 'terminal.json'); log = small(run / 'logs/stdout.log'); err = small(run / 'logs/stderr.log')
        out[name] = {'terminal': json.loads(terminal['text']) if terminal and 'text' in terminal else terminal,
                     'stdout_sha256': log.get('sha256') if log else None, 'stdout_last_json': last_json(log['text']) if log and 'text' in log else None,
                     'stderr_tail': err['text'][-600:] if err and 'text' in err else None,
                     'out_listing': sorted((p.name, p.stat().st_size) for p in (run / 'out').iterdir()) if (run / 'out').is_dir() else None}
    return {'runs': out, 'queue': queue, 'boundaries': {'prediction_content_read': False, 'checkpoint_read': False, 'test_read': False}}


if __name__ == '__main__':
    print(json.dumps(status(), sort_keys=True))
