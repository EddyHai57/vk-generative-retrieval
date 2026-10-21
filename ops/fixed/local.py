"""Client for a finite set of versioned engineering actions."""
import json
import os
from pathlib import Path
import hashlib
import pwd
import re
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).absolute().parents[1]
VERSION = 'vkgr-fixed-ops-v2'
ACTIONS = ('status', 'resources', 'queue', 'receipts', 'deploy')
PAYLOAD_FILES = ('bin/vkgr-fixed', 'fixed/local.py', 'fixed/server.py', 'install.json')


def plan(action=None):
    return {'schema': 'vkgr-fixed-ops-plan-v1', 'version': VERSION, 'action': action,
            'commands': [*ACTIONS, 'plan'], 'deploy_commit': '661a444870dbb07b4fbcd69cebb57f25c691bb26',
            'writes': 'new deployment and pin directories only; git fetch/worktree metadata' if action == 'deploy' else 'none',
            'source': 'fixed ops 5840e6b; no training, push or submit action'}


def environment():
    # Do not inherit PYTHONPATH, LD_PRELOAD, GIT_SSH_COMMAND or caller options.
    return {'PATH': '/usr/bin:/bin', 'HOME': pwd.getpwuid(os.getuid()).pw_dir, 'LANG': 'C.UTF-8'}


def private_receipts():
    # Fixed per-OS-user directory; HOME/cwd/install location cannot retarget it.
    return Path(pwd.getpwuid(os.getuid()).pw_dir)/'Library/Logs/VKGRFixedOps-v2'


def check_acl_text(text):
    lines = text.splitlines()
    if not lines or not lines[0]:
        raise ValueError('missing native ACL metadata')
    for line in lines[1:]:
        ace = re.fullmatch(r'\s*\d+: \S+ (?:inherited )?(allow|deny) [a-z_,]+', line)
        if ace is None or ace.group(1) != 'deny':
            raise ValueError('allow or unrecognized ACL refused')


def check_acl(path):
    # macOS W_OK does not cover delete_child/writesecurity or inherited read ACEs.
    result = subprocess.run(['/bin/ls', '-lde', str(path)], capture_output=True,
                            text=True, env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'}, timeout=10)
    if result.returncode:
        raise ValueError('native ACL inspection failed')
    check_acl_text(result.stdout)


def root_chain(root):
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('installation root must be canonical absolute')
    for path in (root, *root.parents):
        st = path.lstat()
        if (not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode)
                or st.st_uid != 0 or st.st_mode & 0o022 or os.access(path, os.W_OK)):
            raise ValueError('installation ancestors must be root-controlled')
        check_acl(path)


def payload(root, expected_owner=0, require_readonly=True):
    # Separate payload validator permits fixture tests; production never overrides defaults.
    for rel in ('', 'bin', 'fixed', 'manifest.json', *PAYLOAD_FILES):
        path = root/rel
        st = path.lstat()
        directory = rel in ('', 'bin', 'fixed')
        if (st.st_uid != expected_owner or stat.S_ISLNK(st.st_mode)
                or (directory and not stat.S_ISDIR(st.st_mode))
                or (not directory and not stat.S_ISREG(st.st_mode))
                or (require_readonly and (st.st_mode & 0o222 or os.access(path, os.W_OK)))):
            raise ValueError('installation payload ownership/mode differs')
        if require_readonly:
            check_acl(path)
    if (root/'manifest.json').stat().st_size > 65536:
        raise ValueError('installation manifest too large')
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('version') != VERSION or manifest.get('schema') != 'vkgr-fixed-install-manifest-v2'
            or set(manifest.get('files', {})) != set(PAYLOAD_FILES)):
        raise ValueError('installation manifest identity differs')
    for rel in PAYLOAD_FILES:
        if hashlib.sha256((root/rel).read_bytes()).hexdigest() != manifest['files'][rel]:
            raise ValueError('installation file hash differs')
    return manifest


def check_private_dir():
    path = private_receipts()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('private transport directory symlink refused')
    st = path.stat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise ValueError('private transport directory must be owned by this user and mode0700')
    check_acl(path)
    return path


def installation():
    root_chain(ROOT)
    manifest = payload(ROOT)
    config = json.loads((ROOT/'install.json').read_text())
    if (set(config) != {'schema', 'version', 'install_root', 'private_receipts'}
            or config['schema'] != 'vkgr-fixed-install-config-v2' or config['version'] != VERSION
            or config['install_root'] != str(ROOT)
            or config['private_receipts'] != str(private_receipts())):
        raise ValueError('installation configuration differs')
    check_private_dir()
    return manifest


def installed_source():
    installation()
    return (ROOT/'fixed/server.py').read_text()


def remote_receipt(text, action):
    lines = text.rstrip().splitlines()
    if not lines:
        raise ValueError('missing remote receipt')
    doc = json.loads(lines[-1])
    if (not isinstance(doc, dict) or doc.get('schema') != 'vkgr-fixed-ops-receipt-v1'
            or doc.get('action') != action or 'result' not in doc):
        raise ValueError('remote receipt identity mismatch')
    return doc


def save_transport(run):
    folder = check_private_dir()
    fd, filename = tempfile.mkstemp(prefix='transport-', suffix='.json', dir=folder)
    with os.fdopen(fd, 'w') as stream:
        st = os.fstat(stream.fileno())
        if st.st_uid != os.getuid() or st.st_mode & 0o777 != 0o600:
            raise ValueError('private transport receipt owner/mode differs')
        check_acl(Path(filename))
        # Check before writing any private content; retain empty failure evidence.
        json.dump({'exit': run.returncode, 'stdout_tail': run.stdout[-65536:],
                   'stderr_tail': run.stderr[-65536:]}, stream)
    return filename


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args == ['plan']:
        print(json.dumps(plan(), sort_keys=True)); return 0
    if len(args) != 2 or args[0] not in ACTIONS or args[1] not in ('execute', 'plan'):
        print('refuse: ACTION execute|plan; no extra arguments', file=sys.stderr)
        return 2
    action, mode = args
    if mode == 'plan':
        print(json.dumps(plan(action), sort_keys=True)); return 0
    try:
        source = installed_source()
        run = subprocess.run(['/usr/bin/ssh', '-T', 'vast-a40',
                              '/workspace/vkgr-env-a40-v1/.venv/bin/python -I - ' + action],
                             input=source, text=True, capture_output=True, env=environment(), timeout=360)
        save_transport(run)
        if run.returncode:
            print('fixed remote action failed; inspect private transport receipt', file=sys.stderr)
            return 1
        # SSH login banners must not become JSON or executable instructions.
        doc = remote_receipt(run.stdout, action)
        print(json.dumps(doc, sort_keys=True))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('fixed ops refused: installation/transport ' + type(error).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
