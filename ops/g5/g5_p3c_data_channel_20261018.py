"""Server-side, one-shot: restore the frozen data-channel version of docs/p3c_validation.md in a fresh G5 deployment.

The public Git copy of that document is the sanitised variant (sha 5449378c...), while the registered data contract pins the original
bytes (sha c89b6466...). Same procedure as the accepted M1 / M2 deployments: the Git version is moved to the OPS directory (kept),
and the pinned original bytes are copied from the already accepted source file. Refuses to run unless every hash matches; never
overwrites; touches nothing else. Reads a JSON request on stdin with absolute paths: {"deployment", "ops", "source"}.
"""
import hashlib
import json
import sys
from pathlib import Path

PINNED = 'c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
GIT_VERSION = '5449378c3e2c00d5af4102da73e93a28839898107c139df110f29d50c321c466'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply(deployment, ops, source):
    dep, ops, src = Path(deployment), Path(ops), Path(source)
    target = dep / 'docs/p3c_validation.md'; kept = ops / 'p3c-git-original.md'
    if any(p.is_symlink() for p in (dep, ops, src, target)):
        raise ValueError('symlinks refused')
    if sha(src) != PINNED:
        raise ValueError('data-channel source hash mismatch')
    if sha(target) != GIT_VERSION:
        raise ValueError('deployed git version hash unexpected (already overlaid?)')
    if kept.exists() or not (ops / 'deployment.json').is_file():
        raise ValueError('ops directory state unexpected')
    target.rename(kept)
    with target.open('xb') as stream:
        stream.write(src.read_bytes())
    if sha(target) != PINNED or sha(kept) != GIT_VERSION:
        raise ValueError('post-write hash mismatch')
    return {'status': 'data_channel_applied', 'p3c_sha256': sha(target), 'git_original_kept_sha256': sha(kept)}


def main():
    req = json.loads(sys.stdin.read(65537))
    if set(req) != {'deployment', 'ops', 'source'}:
        raise ValueError('request fields')
    print(json.dumps(apply(req['deployment'], req['ops'], req['source']), sort_keys=True))


if __name__ == '__main__':
    main()
