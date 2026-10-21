"""Build a new private staging bundle; never install files or Codex rules."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import stat
import sys

ROOT = Path(__file__).absolute().parents[2]
VERSION = 'vkgr-fixed-ops-v2'
_spec = importlib.util.spec_from_file_location('package_local', Path(__file__).with_name('local.py'))
_local = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local)
RUNTIME = {'bin/vkgr-fixed': 'ops/bin/vkgr-fixed', 'fixed/local.py': 'ops/fixed/local.py',
           'fixed/server.py': 'ops/fixed/server.py'}


def private_receipts():
    return Path(pwd.getpwuid(os.getuid()).pw_dir)/'Library/Logs/VKGRFixedOps-v2'


def install_destination(path):
    if (not path.is_absolute() or '..' in path.parts
            or not re.fullmatch(r'/[A-Za-z0-9_./-]+', str(path))):
        raise ValueError('use a canonical absolute install path without spaces/metacharacters')
    if path.exists() or path.is_symlink():
        raise FileExistsError('installation version must be new')
    for parent in path.parents:
        if parent.is_symlink():
            raise ValueError('installation ancestor symlink refused')
        if parent.exists():
            st = parent.stat()
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != 0 or st.st_mode & 0o022 or os.access(parent, os.W_OK):
                raise ValueError('installation ancestors must be root-controlled')
            _local.check_acl(parent)


def render_rules(entry):
    text = (ROOT/'docs/ops/codex-allow.rules.draft').read_text()
    if not str(entry).startswith('/') or '@INSTALLED_ENTRY@' not in text:
        raise ValueError('absolute installed entry and placeholder draft required')
    return text.replace('@INSTALLED_ENTRY@', str(entry))


def build_package(output, install_root):
    output, install_root = Path(output), Path(install_root)
    install_destination(install_root)
    output.mkdir(mode=0o700, exist_ok=False)
    for rel, source in RUNTIME.items():
        dst = output/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open('xb') as stream:
            stream.write((ROOT/source).read_bytes())
        if rel == 'bin/vkgr-fixed':
            dst.chmod(0o755)
    config = {'schema': 'vkgr-fixed-install-config-v2', 'version': VERSION,
              'install_root': str(install_root), 'private_receipts': str(private_receipts())}
    (output/'install.json').write_text(json.dumps(config, sort_keys=True, indent=2)+'\n')
    files = {rel: hashlib.sha256((output/rel).read_bytes()).hexdigest()
             for rel in (*RUNTIME, 'install.json')}
    manifest = {'schema': 'vkgr-fixed-install-manifest-v2', 'version': VERSION, 'files': files}
    (output/'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    (output/'rendered.rules.draft').write_text(render_rules(install_root/'bin/vkgr-fixed'))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args(argv)
    try:
        result = build_package(args.output, args.install_root)
    except (OSError, ValueError) as error:
        print('package refused: '+type(error).__name__, file=sys.stderr); return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
