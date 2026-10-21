#!/usr/bin/env python3
"""Full-scale science front door: input admission is active; science execution is closed.

The future approved backend must attach after admit_inputs in this same main.
No dataset, checkpoint, history or model loader is imported or called here.
JSON request reads are configuration reads, not scientific data reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

A_ROOT = Path('/workspace/vkgr-ur001-data-20261020-v1')
FORBIDDEN_ROOTS = tuple(Path('/workspace') / name for name in (
    'vkgr-ur001-week26-raw-20261021-v1',
    'vkgr-ur001-week26-membership-20261021-v1',
    'vkgr-ur001-week26-holdout-20261021-v1',
))
MAX_REQUEST_BYTES = 65_536
SCHEMA = 'ur001-science-inputs-v1'

class InputRejected(ValueError):
    def __init__(self, reason: str, holdout: bool = False):
        super().__init__(reason)
        self.holdout = holdout


def _beneath(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def checked_path(value: str, *, config: bool = False) -> Path:
    if type(value) is not str or not value or '\\' in value:
        raise InputRejected('invalid input path')
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise InputRejected('input path must be absolute without traversal')
    if any(_beneath(path, root) for root in FORBIDDEN_ROOTS):
        raise InputRejected('bounded week26 assets are excluded from science inputs', holdout=True)
    if config and _beneath(path, A_ROOT):
        raise InputRejected('configuration cannot be a dataset member')
    for member in (path, *path.parents):
        if member.is_symlink():
            raise InputRejected('input path may not traverse symlinks')
    return path


def admit_inputs(path: str) -> str:
    config = checked_path(path, config=True)
    with config.open('rb') as stream:
        raw = stream.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise InputRejected('input configuration exceeds bounded size')
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        raise InputRejected('invalid input configuration JSON') from None
    if (type(value) is not dict or set(value) != {'schema', 'train', 'validation', 'metadata'}
            or value['schema'] != SCHEMA):
        raise InputRejected('input configuration schema or role set differs')
    train = value['train']; metadata = value['metadata']
    if type(train) is not list or type(metadata) is not list:
        raise InputRejected('train and metadata roles must be lists')
    # Check every declared role before comparing to the exact approved whitelist.
    for item in [*train, value['validation'], *metadata]:
        checked_path(item)
    expected_train = [str(A_ROOT / f'subsamples/ur0.01/train/week_{w:02}.parquet') for w in range(25)]
    expected_val = str(A_ROOT / 'subsamples/ur0.01/validation/week_25.parquet')
    expected_metadata = [str(A_ROOT / 'metadata' / name) for name in (
        'users_metadata.parquet', 'items_metadata.parquet', 'item_embeddings.npz')]
    if train != expected_train or value['validation'] != expected_val or metadata != expected_metadata:
        raise InputRejected('input roles differ from the exact approved train/validation whitelist')
    return hashlib.sha256(raw).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('dry', 'train', 'eval', 'verify'))
    parser.add_argument('--inputs', required=True)
    args = parser.parse_args(argv)
    common = {'kind': args.kind, 'stage': 'input_admission',
              'execution_allowed': False, 'data_opened': False}
    try:
        sha = admit_inputs(args.inputs)
    except (InputRejected, OSError) as exc:
        is_holdout = isinstance(exc, InputRejected) and exc.holdout
        result = {**common, 'status': 'HOLDOUT_INPUT_REJECTED' if is_holdout else 'INPUTS_REJECTED',
                  'reason': str(exc)}
        print(json.dumps(result, sort_keys=True, separators=(',', ':')))
        return 2
    if args.kind == 'dry':
        result = {**common, 'status': 'INPUTS_ADMITTED', 'config_sha256': sha}
        code = 0
    else:
        # No scientific backend is authorized or connected in this preparation phase.
        result = {**common, 'status': 'SCIENCE_EXECUTION_CLOSED', 'config_sha256': sha,
                  'reason': 'full-scale scientific execution requires its own frozen card and authorization'}
        code = 2
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
