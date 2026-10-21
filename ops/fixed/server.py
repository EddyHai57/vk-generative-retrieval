"""Fixed server dispatch; no caller supplied paths, source, job requests or shell."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OPS = Path('/workspace/vkgr-ops-code-a40-v1')
REPO = Path('/workspace/vkgr-repo-a40-v1')
ROOT = Path('/workspace/vkgr-ops-root-a40-v1')
PYTHON = '/workspace/vkgr-env-a40-v1/.venv/bin/python'
OPS_COMMIT = '5840e6b686ec9651314a9e28a1a2c0befcc05158'
DEPLOY_COMMIT = '661a444870dbb07b4fbcd69cebb57f25c691bb26'
DEPLOY_BRANCH = 'codex/sgns-throughput-public-20261021-v1'
TARGET = Path('/workspace/vkgr-fixed-entry-audit-code-v1')
PINS = Path('/workspace/vkgr-fixed-entry-audit-pins-v1')
RUNS = ('a40-exact-run-v1', 'a40-rebuild-scheduler-v1',
        'g-decode-timing-preflight-v1', 'sgns-throughput-2000-v1')
ACTIONS = ('status', 'resources', 'queue', 'receipts', 'deploy')
RECEIPT_LIMIT = 65536


def clean_env():
    return {'PATH': '/usr/bin:/bin', 'HOME': '/root', 'LANG': 'C.UTF-8'}


def call(argv, **kwargs):
    return subprocess.run([str(v) for v in argv], check=True, capture_output=True,
                          env=clean_env(), timeout=300, **kwargs).stdout


def small_json(path):
    # O_NOFOLLOW also rejects a last-component symlink raced after an lstat.
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            raise ValueError('receipt symlink refused')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        import stat
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('receipt must be regular')
        data = stream.read(RECEIPT_LIMIT + 1)
    if len(data) > RECEIPT_LIMIT:
        raise ValueError('receipt too large')
    return json.loads(data)


def receipt_summary(doc):
    return {key: doc[key] for key in ('schema', 'run_id', 'kind', 'resources', 'status',
                                     'reason', 'exit_code', 'wall_seconds') if key in doc}


def new_targets(target, pins):
    for path in (target, pins):
        if path.exists() or path.is_symlink():
            raise ValueError('refuse: target exists')
        for parent in path.parents:
            if parent.is_symlink():
                raise ValueError('refuse: target parent symlink')


def check_ops():
    head = call(['git', '-C', OPS, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = call(['git', '-C', OPS, 'status', '--porcelain=v1', '--untracked-files=all'], text=True)
    if head != OPS_COMMIT or dirty:
        raise ValueError('fixed ops source identity mismatch')


def deploy_tree(repo, target, pins, commit, tools=OPS):
    new_targets(target, pins)
    # Reuse the original, byte-pinned deploy_spec and g_deploy, without overlays.
    spec = json.loads(call([sys.executable, '-I', tools/'ops/tools/local/deploy_spec.py',
                           'tree', '--repo', repo, '--commit', commit], text=True))
    payload = {'base': commit, 'target': commit, 'target_tree': spec['tree'],
               'data': {}, 'overrides': {}}
    cfg = {'server': {'ops_repo': str(repo), 'g_deploy': str(target), 'pinned_dir': str(pins)}}
    code = 'CFG = ' + repr(cfg) + '\n' + (tools/'ops/tools/remote/g_deploy.py').read_text()
    # Exclusive reservation closes the old tool's exist_ok pin-directory race.
    pins.mkdir(mode=0o755, exist_ok=False)
    result = json.loads(call([sys.executable, '-I', '-c', code],
                             input=json.dumps(payload), text=True))
    if (result['head'] != commit or result['target_tree_bad'] or result['overlay_sha_bad']
            or result['data_sha_bad'] or result['porcelain_lines']):
        raise ValueError('deployment verification failed; preserve new paths for diagnosis')
    return result


def dispatch(action):
    check_ops()
    if action == 'deploy':
        new_targets(TARGET, PINS)
        # Fetch only the registered public branch, require its fixed published SHA.
        call(['git', '-C', REPO, 'fetch', 'origin', 'refs/heads/' + DEPLOY_BRANCH])
        fetched = call(['git', '-C', REPO, 'rev-parse', 'FETCH_HEAD'], text=True).strip()
        if fetched != DEPLOY_COMMIT:
            raise ValueError('published deployment identity mismatch')
        return deploy_tree(REPO, TARGET, PINS, DEPLOY_COMMIT)
    if action in ('status', 'resources'):
        doc = json.loads(call([PYTHON, '-I', OPS/'ops/bin/vkgr-status', '--root', ROOT, '--json'], text=True))
        if action == 'resources':
            return {'host': doc.get('host')}
        return {'runs': [r for r in doc.get('runs', []) if r.get('run_id') in RUNS]}
    if action == 'queue':
        out = {}
        for state in ('pending', 'claimed', 'rejected'):
            folder = ROOT/'queue'/state
            if folder.is_symlink():
                raise ValueError('queue symlink refused')
            out[state] = sorted(p.name for p in folder.glob('*.json')
                                if any(p.name.endswith('-' + rid + '.json') for rid in RUNS))
        return out
    # No raw stdout, cost arrays, dataset paths or arbitrary receipt names.
    return {rid: {name: receipt_summary(small_json(ROOT/'runs'/rid/name))
                  for name in ('run.json', 'terminal.json')
                  if (ROOT/'runs'/rid/name).exists()} for rid in RUNS}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in ACTIONS:
        print('refuse: expected one fixed server action', file=sys.stderr)
        return 2
    try:
        print(json.dumps({'schema': 'vkgr-fixed-ops-receipt-v1', 'action': args[0],
                          'result': dispatch(args[0])}, sort_keys=True))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('fixed ops refused: ' + type(error).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
