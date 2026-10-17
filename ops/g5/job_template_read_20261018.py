"""Read-only: print the claimed queue job specs and run.json of the accepted P anchor runs (job metadata only).

Used to clone the exact job envelope (argv, env, cwd, resources, inputs) for the G5 top-100 export job. Reads only small JSON
files named by exact run id; never opens predictions, checkpoints, embeddings, logs or test. stdout only.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/workspace/vkgr-infra01-root')
RUN_IDS = ('p-seqm1-mean-s42', 'p-seqm1-mean-s42-verify')
MAX = 65536


def small(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX:
        return {'read': False, 'reason': 'absent, symlink or oversize'}
    payload = path.read_bytes()
    return {'read': True, 'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(), 'value': json.loads(payload)}


def main():
    out = {}
    for rid in RUN_IDS:
        claimed = sorted((ROOT / 'queue/claimed').glob('*-' + rid + '.json'))
        out[rid] = {'claimed': [small(p) for p in claimed], 'run_json': small(ROOT / 'runs' / rid / 'run.json'),
                    'terminal': small(ROOT / 'runs' / rid / 'terminal.json')}
    print(json.dumps({'jobs': out, 'boundaries': {'prediction_content_read': False, 'checkpoint_read': False, 'test_read': False}}, sort_keys=True))


if __name__ == '__main__':
    main()
