"""vkgr-launch: validate, admit and start one detached job (design §4.2).

Steps 1-5 (static, git, pinned manifest, dirty rule, input/config hashes) always run
in full on every launch() and always before admission.lock is taken (D12).
"""
import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from vkgr_ops import admission, common

OPS_DIR = Path(__file__).resolve().parents[1]
SUPERVISOR = Path(__file__).resolve().with_name('supervisor.py')
ENV_KEY_RE = re.compile(r'^[A-Z_][A-Z0-9_]*$')
HEX40 = re.compile(r'^[0-9a-f]{40}$')
HEX64 = re.compile(r'^[0-9a-f]{64}$')
ABSENT = 'absent'
INJECTED_ENV = ('VKGR_RUN_ID', 'VKGR_PROGRESS_PATH')
ENV_WHITELIST = ('OMP_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH')
RESOURCE_ARGS = ('cpu_cores', 'ram_mib', 'vram_mib', 'disk_gib')
KINDS = ('real', 'smoke', 'ops')
STOP_MODES = ('signal', 'none')
ADMISSION_LOCK_TIMEOUT = 30.0
FIRST_FORK_TIMEOUT = 5.0
OWNER_TIMEOUT = 10.0
MAX_WALL_HOURS = 720


class LaunchError(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- CLI

def build_parser(prog='vkgr-launch', with_root=True):
    p = argparse.ArgumentParser(prog=prog, description='Start one detached, admitted job.')
    if with_root:
        p.add_argument('--root', default=None, help='ops root (or VKGR_OPS_ROOT)')
    p.add_argument('--run-id', required=True)
    p.add_argument('--kind', required=True, choices=KINDS)
    p.add_argument('--cwd', required=True)
    p.add_argument('--config', default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--input', dest='inputs', action='append', default=[])
    p.add_argument('--pinned-manifest', default=None)
    p.add_argument('--pinned-file', dest='pinned_files', action='append', default=[])
    p.add_argument('--env', action='append', default=[])
    for name in RESOURCE_ARGS:
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=int, required=True)
    p.add_argument('--wall-hours', type=float, required=True)
    p.add_argument('--stop-mode', default='signal', choices=STOP_MODES)
    p.add_argument('--no-admission', action='store_true')
    p.add_argument('argv', nargs=argparse.REMAINDER)
    return p


def spec_from_args(ns):
    argv = list(ns.argv)
    if argv and argv[0] == '--':
        argv = argv[1:]
    keys = ('run_id', 'kind', 'cwd', 'config', 'seed', 'inputs', 'pinned_manifest', 'pinned_files', 'env',
            *RESOURCE_ARGS, 'wall_hours', 'stop_mode', 'no_admission')
    spec = {k: getattr(ns, k) for k in keys}
    spec['argv'] = argv
    return spec


def resolve_root(value):
    value = value or os.environ.get('VKGR_OPS_ROOT')
    if not value:
        raise LaunchError('ops root required: pass --root or set VKGR_OPS_ROOT')
    return Path(value).resolve()


# ---------------------------------------------------------------- steps 1-5

def _git(repo, *args, binary=False):
    try:
        out = subprocess.run(['git', '-C', str(repo), *args], stdin=subprocess.DEVNULL,
                             capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, 'stderr', b'') or b''
        raise LaunchError(f'git {" ".join(args[:2])} failed: {detail.decode(errors="replace").strip()}') from error
    return out.stdout if binary else out.stdout.decode()


def _git_ok(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], stdin=subprocess.DEVNULL,
                          capture_output=True).returncode == 0


def _validate_static(spec):
    if not common.valid_run_id(spec.get('run_id')):
        raise LaunchError(f'invalid run_id {spec.get("run_id")!r}: must match {common.RUN_ID_RE.pattern}')
    if spec.get('kind') not in KINDS:
        raise LaunchError(f'invalid kind {spec.get("kind")!r}')
    if spec.get('stop_mode') not in STOP_MODES:
        raise LaunchError(f'invalid stop_mode {spec.get("stop_mode")!r}')
    for name in RESOURCE_ARGS:
        value = spec.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise LaunchError(f'{name} must be a non-negative int, got {value!r}')
    wall = spec.get('wall_hours')
    if (not isinstance(wall, (int, float)) or isinstance(wall, bool) or not math.isfinite(wall)
            or wall <= 0 or wall > MAX_WALL_HOURS):
        raise LaunchError(f'wall_hours must be a finite float in (0, {MAX_WALL_HOURS}], got {wall!r}')
    if spec.get('seed') is not None and (not isinstance(spec['seed'], int) or isinstance(spec['seed'], bool)):
        raise LaunchError('seed must be an int')
    env = {}
    for item in spec.get('env') or []:
        key, sep, value = item.partition('=')
        if not sep or not ENV_KEY_RE.match(key):
            raise LaunchError(f'invalid --env {item!r}: key must match {ENV_KEY_RE.pattern}')
        if key in INJECTED_ENV:
            raise LaunchError(f'invalid --env {key}: injected by the launcher')
        env[key] = value
    if not spec.get('argv'):
        raise LaunchError('job argv required after --')
    cwd = Path(spec['cwd'])
    if not cwd.is_dir():
        raise LaunchError(f'cwd is not a directory: {cwd}')
    return env


def _pin_value_ok(value):
    return value == ABSENT or (isinstance(value, str) and HEX64.match(value) is not None)


def _check_rel_path(repo, rel):
    if not isinstance(rel, str) or not rel or rel.startswith('/') or '\\' in rel:
        raise LaunchError(f'pinned path {rel!r} invalid: must be relative')
    parts = rel.split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise LaunchError(f'pinned path {rel!r} invalid: no empty, "." or ".." segments')
    path = repo / rel
    if path.is_symlink():
        raise LaunchError(f'pinned path {rel!r} is a symlink')
    try:
        path.resolve().relative_to(repo)
    except ValueError:
        raise LaunchError(f'pinned path {rel!r} resolves outside the repository') from None
    return path


def _actual(repo, rel):
    path = _check_rel_path(repo, rel)
    if not path.exists():
        return ABSENT
    if not path.is_file():
        raise LaunchError(f'pinned path {rel!r} is not a regular file')
    return common.sha256_file(path)[0]


def _parse_pinned_files(items):
    pins = {}
    for item in items or []:
        rel, sep, value = item.rpartition('=')
        if not sep or not _pin_value_ok(value):
            raise LaunchError(f'invalid --pinned-file {item!r}: expected REL=<sha256 64hex>|absent')
        if rel in pins and pins[rel] != value:
            raise LaunchError(f'--pinned-file {rel} declared twice with different sha256')
        pins[rel] = value
    return pins


def _load_pinned_manifest(path, repo, head):
    raw = common.read_bytes_limited(path, 16 * 1024 * 1024)
    if raw is None:
        raise LaunchError(f'pinned manifest unreadable: {path}')
    try:
        doc = json.loads(raw)
    except ValueError:
        raise LaunchError('pinned manifest is not JSON') from None
    if not isinstance(doc, dict):
        raise LaunchError('pinned manifest must be an object')
    allowed = {'schema', 'base_commit', 'target_commit', 'files', 'overrides'}
    keys = set(doc)
    if not keys <= allowed or not allowed - {'overrides'} <= keys:
        raise LaunchError(f'pinned manifest key set must be {sorted(allowed)} (overrides optional), got {sorted(keys)}')
    if doc['schema'] != 'vkgr-pinned-v1':
        raise LaunchError(f'pinned manifest schema must be vkgr-pinned-v1, got {doc["schema"]!r}')
    base, target = doc['base_commit'], doc['target_commit']
    if not isinstance(base, str) or not HEX40.match(base):
        raise LaunchError('pinned manifest base_commit must be 40 hex')
    if not isinstance(target, str) or not HEX40.match(target):
        raise LaunchError('pinned manifest target_commit must be 40 hex')
    if base != head:
        raise LaunchError(f'pinned manifest base_commit {base} != HEAD {head}')
    if not _git_ok(repo, 'cat-file', '-e', f'{target}^{{commit}}'):
        raise LaunchError(f'pinned manifest target_commit {target} not in object store (git fetch first)')
    files = doc['files']
    overrides = doc.get('overrides', {})
    if not isinstance(files, dict) or not isinstance(overrides, dict):
        raise LaunchError('pinned manifest files/overrides must be objects')
    for rel, value in files.items():
        _check_rel_path(repo, rel)
        if not _pin_value_ok(value):
            raise LaunchError(f'pinned manifest files[{rel!r}] must be sha256 64hex or "absent"')
    for rel, value in overrides.items():
        _check_rel_path(repo, rel)
        if not isinstance(value, dict) or set(value) != {'sha256', 'reason'}:
            raise LaunchError(f'pinned manifest override {rel!r} key set must be exactly sha256, reason')
        if not isinstance(value['sha256'], str) or not HEX64.match(value['sha256']):
            raise LaunchError(f'pinned manifest override {rel!r} sha256 must be 64hex')
        reason = value['reason']
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 200:
            raise LaunchError(f'pinned manifest override {rel!r} reason must be 1..200 chars')
    overlap = sorted(set(files) & set(overrides))
    if overlap:
        raise LaunchError(f'pinned manifest files and overrides overlap: {overlap}')
    diff = [p for p in _git(repo, 'diff', '--name-only', '-z', base, target).split('\0') if p]
    missing = sorted(p for p in diff if p not in files and p not in overrides)
    if missing:
        raise LaunchError(f'pinned manifest incomplete: diff paths not pinned: {missing}')
    diff_set = set(diff)
    for rel in sorted(files):
        declared = files[rel]
        in_target = _git_ok(repo, 'cat-file', '-e', f'{target}:{rel}')
        if in_target:
            blob = common.sha256_bytes(_git(repo, 'cat-file', 'blob', f'{target}:{rel}', binary=True))
            if declared != blob:
                raise LaunchError(f'pinned {rel}: declared {declared} != target blob {blob}')
        elif rel in diff_set and declared != ABSENT:
            raise LaunchError(f'pinned {rel}: deleted in target, must be declared "absent"')
        actual = _actual(repo, rel)
        if actual != declared:
            raise LaunchError(f'pinned {rel}: actual {actual} != declared {declared}')
    for rel in sorted(overrides):
        actual = _actual(repo, rel)
        if actual != overrides[rel]['sha256']:
            raise LaunchError(f'pinned override {rel}: actual {actual} != declared {overrides[rel]["sha256"]}')
    return doc, files, {rel: v['sha256'] for rel, v in overrides.items()}


def _dirty_paths(repo):
    raw = _git(repo, 'status', '--porcelain=v1', '-z', '--untracked-files=all', binary=True).decode()
    tokens = raw.split('\0')
    paths, i = set(), 0
    while i < len(tokens):
        entry = tokens[i]
        i += 1
        if not entry:
            continue
        code, path = entry[:2], entry[3:]
        paths.add(path)
        if code[0] in 'RC':          # rename/copy: next token is the original path
            if i < len(tokens) and tokens[i]:
                paths.add(tokens[i])
            i += 1
    return sorted(paths)


def _ops_source():
    commit = _git(OPS_DIR, 'rev-parse', 'HEAD').strip()
    files = {f'vkgr_ops/{p.name}': common.sha256_file(p)[0]
             for p in sorted((OPS_DIR / 'vkgr_ops').glob('*.py'))}
    return {'commit': commit, 'sha256': common.sha256_bytes(common.canonical_json(files))}


def prepare(spec):
    """Steps 1-5. Returns the launch-time record fields; raises LaunchError (no side effects)."""
    env = _validate_static(spec)
    repo = Path(spec['cwd']).resolve()
    head = _git(repo, 'rev-parse', 'HEAD').strip()
    pins = _parse_pinned_files(spec.get('pinned_files'))
    overlay, overrides = None, {}
    if spec.get('pinned_manifest'):
        manifest_path = Path(spec['pinned_manifest']).resolve()
        doc, files, overrides = _load_pinned_manifest(manifest_path, repo, head)
        for rel, value in files.items():
            if rel in pins and pins[rel] != value:
                raise LaunchError(f'--pinned-file {rel} conflicts with pinned manifest sha256')
        for rel in pins:
            if rel in overrides:
                raise LaunchError(f'--pinned-file {rel} conflicts with pinned manifest override')
        pins = {**files, **pins}
        overlay = {'base_commit': doc['base_commit'], 'target_commit': doc['target_commit'],
                   'pinned_manifest_sha256': common.sha256_file(manifest_path)[0], 'overrides': overrides}
    actual_pins = {rel: _actual(repo, rel) for rel in sorted(pins)}
    for rel, value in pins.items():
        if actual_pins[rel] != value:
            raise LaunchError(f'pinned {rel}: actual sha256 {actual_pins[rel]} != declared {value}')
    dirty_files = []
    for rel in _dirty_paths(repo):
        actual = _actual_unchecked(repo, rel)
        dirty_files.append({'path': rel, 'sha256': actual})
        pinned = pins.get(rel, overrides.get(rel))
        if spec['kind'] in ('real', 'ops') and pinned != actual:
            raise LaunchError(f'dirty path not pinned for kind={spec["kind"]}: {rel}')
    inputs = []
    for item in spec.get('inputs') or []:
        path = Path(item).resolve()
        try:
            digest, size = common.sha256_file(path)
        except OSError as error:
            raise LaunchError(f'input unreadable: {item}: {error}') from None
        inputs.append({'path': str(path), 'sha256': digest, 'bytes': size})
    config = None
    if spec.get('config'):
        path = Path(spec['config']).resolve()
        try:
            config = {'path': str(path), 'sha256': common.sha256_file(path)[0]}
        except OSError as error:
            raise LaunchError(f'config unreadable: {spec["config"]}: {error}') from None
    return {'env': env, 'cwd': str(repo),
            'git': {'commit': head, 'dirty': bool(dirty_files), 'dirty_files': dirty_files},
            'overlay': overlay, 'pinned_files': pins, 'config': config, 'inputs': inputs}


def _actual_unchecked(repo, rel):
    path = repo / rel
    if path.is_symlink():
        return common.sha256_bytes(os.readlink(path).encode())
    if not path.exists():
        return ABSENT
    return common.sha256_file(path)[0] if path.is_file() else ABSENT


# ---------------------------------------------------------------- locking / start

def acquire_admission_lock(root, timeout=None):
    timeout = ADMISSION_LOCK_TIMEOUT if timeout is None else timeout
    handle = open(Path(root) / 'admission.lock', 'a+')
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise LaunchError('admission lock busy') from None
            time.sleep(0.05)


def supervisor_argv(run_dir):
    """-I isolates PYTHON* env; supervisor.py puts <ops>/ on sys.path itself (M4)."""
    return [sys.executable, '-I', str(SUPERVISOR), str(run_dir)]


def _job_env(env, run_dir, run_id):
    full = dict(os.environ)
    full.update(env)
    full['VKGR_RUN_ID'] = run_id
    full['VKGR_PROGRESS_PATH'] = str(run_dir / 'progress.json')
    return full


def _recorded_env(env, job_env):
    """N6: {key: value}; --env keys with passed values, whitelist/VKGR_* from current env."""
    recorded = {k: v for k, v in job_env.items() if k in ENV_WHITELIST or k.startswith('VKGR_')}
    recorded.update(env)
    return dict(sorted(recorded.items()))


def _launch_failed(run_dir, run_id, message):
    log = run_dir / 'logs' / 'supervisor.log'
    with open(log, 'a') as handle:
        handle.write(f'launch_failed: {message}\n')
    now = time.time()
    try:
        common.write_json_once(run_dir / 'terminal.json', {
            'schema': 'vkgr-terminal-v1', 'run_id': run_id, 'status': 'launch_failed', 'exit_code': None,
            'signal': None, 'started_at': None, 'ended_at': now, 'wall_seconds': None,
            'wall_cap_exceeded': False})
    except FileExistsError:
        pass


def _start_supervisor(run_dir, run_id, job_env):
    log = open(run_dir / 'logs' / 'supervisor.log', 'ab')
    try:
        first = subprocess.Popen(supervisor_argv(run_dir), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 close_fds=True, env=job_env, cwd=str(run_dir))
    except OSError as error:
        log.close()
        _launch_failed(run_dir, run_id, f'cannot start supervisor: {error}')
        raise LaunchError(f'cannot start supervisor: {error}') from None
    log.close()
    try:
        # M3: supervisor.py forks; this first process exits at once and is reaped here.
        code = first.wait(timeout=FIRST_FORK_TIMEOUT)
    except subprocess.TimeoutExpired:
        first.kill()
        first.wait()
        _launch_failed(run_dir, run_id, 'supervisor did not detach')
        raise LaunchError('supervisor did not detach') from None
    if code != 0:
        _launch_failed(run_dir, run_id, f'supervisor first process exited {code}')
        raise LaunchError(f'supervisor first process exited {code}')
    deadline = time.monotonic() + OWNER_TIMEOUT
    while time.monotonic() < deadline:
        if (run_dir / 'owner.json').exists():
            return
        terminal = common.read_json_limited(run_dir / 'terminal.json')
        if terminal is not None:
            raise LaunchError(f'job {terminal.get("status")}: see {run_dir / "logs" / "supervisor.log"}')
        time.sleep(0.05)
    raise LaunchError(f'owner.json not written within {OWNER_TIMEOUT:.0f}s; job may still be starting', code=2)


def launch(spec, root, probes=None):
    """Full launch (§4.2); returns the run directory. Same function for CLI and scheduler."""
    root = Path(root).resolve()
    prepared = prepare(spec)                                    # steps 1-5, outside the lock
    ops_source = _ops_source()
    run_id = spec['run_id']
    run_dir = root / 'runs' / run_id
    if run_dir.exists():
        raise LaunchError(f'run directory exists: {run_dir}')
    (root / 'runs').mkdir(parents=True, exist_ok=True)
    lock = None
    try:
        record = {'checked': False}
        if not spec.get('no_admission'):
            lock = acquire_admission_lock(root)
            need = {r: spec[r] for r in RESOURCE_ARGS}
            ok, snapshot, reason = admission.admit(need, root, probes)
            if not ok:
                raise LaunchError(f'admission rejected: {reason}')
            record = {'checked': True, 'snapshot': snapshot}
        try:
            run_dir.mkdir()
        except FileExistsError:
            raise LaunchError(f'run directory exists: {run_dir}') from None
        (run_dir / 'logs').mkdir()
        job_env = _job_env(prepared['env'], run_dir, run_id)
        run = {
            'schema': 'vkgr-run-v1', 'run_id': run_id, 'kind': spec['kind'], 'created_at': time.time(),
            'cwd': prepared['cwd'], 'argv': list(spec['argv']), 'env': _recorded_env(prepared['env'], job_env),
            'git': prepared['git'], 'overlay': prepared['overlay'], 'pinned_files': prepared['pinned_files'],
            'config': prepared['config'], 'seed': spec.get('seed'), 'inputs': prepared['inputs'],
            'resources': {**{r: spec[r] for r in RESOURCE_ARGS}, 'wall_seconds': spec['wall_hours'] * 3600.0},
            'stop_mode': spec['stop_mode'], 'admission': record, 'ops_source': ops_source,
        }
        common.write_json_once(run_dir / 'run.json', run)
        _start_supervisor(run_dir, run_id, job_env)
        return run_dir
    finally:
        if lock is not None:
            lock.close()


def main(argv=None, probes=None):
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        root = resolve_root(ns.root)
        run_dir = launch(spec_from_args(ns), root, probes)
    except LaunchError as error:
        print(f'vkgr-launch: {error}', file=sys.stderr)
        return error.code
    print(run_dir.name)
    return 0


if __name__ == '__main__':
    sys.exit(main())
