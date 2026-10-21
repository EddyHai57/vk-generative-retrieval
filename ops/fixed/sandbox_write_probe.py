"""Run ONLY in the default executor sandbox, after Eddy installs the reviewed copy.

Try a write-only open without O_TRUNC or write(). File contents are never modified,
even if the boundary is unexpectedly writable. Never escalate this probe.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def probe(root):
    spec = importlib.util.spec_from_file_location('fixed_install_check',Path(__file__).absolute().with_name('verify_install.py'))
    checker = importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    checker.inspect_install(root)
    path = root/'fixed/server.py'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    denied = None
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    except PermissionError as error:
        denied = error.errno
    else:
        os.close(fd)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'write_open_denied': denied is not None, 'errno': denied, 'bytes_unchanged': before == after,
            'before_sha256': before, 'after_sha256': after, 'uid': os.getuid(),
            'probe': 'O_WRONLY|O_NOFOLLOW; no O_TRUNC; no write; no escalation',
            'denial_source': 'filesystem_or_sandbox_not_distinguishable_from_errno'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-root', required=True, type=Path)
    args = p.parse_args(argv)
    try:
        result = probe(args.install_root)
    except (OSError, ValueError) as error:
        print('PENDING/FAILED installed-copy probe: '+type(error).__name__,file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result['write_open_denied'] and result['bytes_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
