"""Read-only, bounded takeover inventory for the twelve completed P/G runs."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

MAX_REQUEST_BYTES = 256 * 1024
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_GUIDE_BYTES = 256 * 1024
RUN_IDS = tuple(
    [f'p-watch-positive-s{s}{suffix}' for s in ('42', '2024', '2025') for suffix in ('', '-verify')]
    + [f'g-half-users-r2-s{s}{suffix}' for s in ('42', '2024', '2025') for suffix in ('', '-verify')]
)
DRIVER_DIRS = ('vkgr-p-watch-positive-train-ops-v6', 'vkgr-p-watch-positive-verify-owner-v1',
               'vkgr-g-half-users-r2-ops-v2')
GUIDES = ('/etc/vast-agents-guide.md', 'AGENTS.md')
MODULES = ('analysis/ur001_prepare.py', 'analysis/profile_anchor_steps.py')
PYTHON_DEPS = ('numpy', 'torch', 'pandas', 'pyarrow', 'faiss')
JOB_FIELDS = ('schema', 'run_id', 'kind', 'seed', 'resources', 'stop_mode', 'no_admission')
DRIVER_FIELDS = ('deadline_seconds', 'requested_deadline_seconds', 'spent_seconds', 'poll_seconds', 'owner_pid', 'owner_start_ticks',
                 'pid', 'driver_pid', 'birth', 'start_ticks',
                 'started_at', 'deadline_at', 'deadline_epoch', 'original_deadline_epoch',
                 'original_deadline_conservative_lower_bound_epoch', 'original_spent_seconds',
                 'earliest_train_queue_publication_epoch', 'old_duration_seconds', 'run_ids', 'status', 'terminal')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def safe_path(raw, role, exists=False):
    if type(raw) is not str or not Path(raw).is_absolute() or '..' in Path(raw).parts:
        raise ValueError(f'{role} must be an absolute, non-traversing path')
    path = Path(raw)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'{role} may not traverse symlinks')
    if exists and not path.exists():
        raise ValueError(f'{role} does not exist')
    return path


def _read(path, max_bytes=MAX_FILE_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    size = path.stat().st_size
    if size > max_bytes:
        return {'oversize': True, 'bytes': size, 'sha256': _hash_file(path)}
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        payload = None
    return {'bytes': len(raw), 'sha256': sha(raw), 'payload': payload}


def _hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _queue(root, run_id):
    state = {}
    for category in ('pending', 'claimed', 'rejected'):
        directory = Path(root) / 'queue' / category
        rows = []
        if directory.is_dir() and not directory.is_symlink():
            for path in sorted(directory.glob('*-' + run_id + '.json')):
                item = _read(path)
                if item is None:
                    continue
                payload = item.get('payload') if not item.get('oversize') else None
                job = {}
                if type(payload) is dict:
                    job = {key: payload[key] for key in JOB_FIELDS if key in payload}
                    argv = payload.get('argv')
                    if isinstance(argv, list):
                        job['argv_sha256'] = sha(canonical(argv))
                    config = payload.get('config')
                    if isinstance(config, str):
                        job['config_basename'] = Path(config).name
                row = {'entry': path.name, 'sha256': item['sha256'], 'bytes': item['bytes'], 'job': job}
                if category == 'rejected':
                    reason = path.with_suffix('.reason')
                    if reason.is_file() and not reason.is_symlink() and reason.stat().st_size <= 8192:
                        row['reason_sha256'] = _hash_file(reason)
                        row['reason'] = reason.read_text(errors='replace')[:8192]
                rows.append(row)
        state[category] = rows
    return state


def _process_identity(pid, expected_birth=None):
    unknown_identity = expected_birth is None
    if type(pid) is not int or pid <= 0:
        return {'pid_alive': False, 'identity_matches': None if unknown_identity else False}
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return {'pid_alive': False, 'identity_matches': None if unknown_identity else False}
    result = {'pid_alive': True, 'identity_matches': None}
    stat_path = Path('/proc') / str(pid) / 'stat'
    if stat_path.is_file() and not stat_path.is_symlink():
        raw = stat_path.read_text()
        tail = raw[raw.rfind(')') + 2:].split()
        if len(tail) > 19:
            actual_birth = tail[19]
            result['actual_birth'] = actual_birth
            if expected_birth is not None:
                result['identity_matches'] = str(expected_birth) == actual_birth
    return result


def _owner_summary(payload):
    result = {key: payload.get(key) for key in ('started_at', 'birth_source') if key in payload}
    found_nested = False
    for role in ('supervisor', 'job'):
        ident = payload.get(role)
        if isinstance(ident, dict):
            found_nested = True
            details = {key: ident.get(key) for key in ('pid', 'pgid', 'sid', 'birth') if key in ident}
            details.update(_process_identity(ident.get('pid'), ident.get('birth')))
            result[role] = details
    process = payload.get('process')
    if isinstance(process, dict):
        pid = process.get('pid')
        birth = process.get('birth', process.get('start_ticks'))
        result.update({'pid': pid, 'birth': birth})
        result.update(_process_identity(pid, birth))
        found_nested = True
    if not found_nested:
        pid = payload.get('pid', payload.get('driver_pid', payload.get('owner_pid')))
        birth = payload.get('birth', payload.get('start_ticks', payload.get('owner_start_ticks')))
        if pid is not None:
            result.update({'pid': pid, 'birth': birth})
            result.update(_process_identity(pid, birth))
    return result if result else None


def _manifest_inventory(root, run_id, argv):
    train_id = run_id[:-7] if run_id.endswith('-verify') else run_id
    expected = Path(root) / 'runs' / train_id / 'out' / 'manifest.json'
    requested = []
    pinned_sha = None
    if isinstance(argv, list):
        for index, item in enumerate(argv[:-1]):
            if item == '--manifest':
                requested.append(argv[index + 1])
            elif item == '--manifest-sha':
                pinned_sha = argv[index + 1]
    if len(requested) > 1:
        raise ValueError(f'multiple manifest paths in original argv: {run_id}')
    if requested and Path(requested[0]) != expected:
        raise ValueError(f'original argv manifest path differs from fixed train manifest: {run_id}')
    path = expected
    result = {'path': str(path), 'expected_path': str(expected), 'exists': False,
              'argv_sha256': pinned_sha}
    if path.is_symlink():
        raise ValueError(f'science manifest path is a symlink: {run_id}')
    if not path.is_file():
        return result
    item = _read(path)
    if item is None or item.get('oversize'):
        result['oversize_or_unreadable'] = True
        if item:
            result.update({k: item[k] for k in ('bytes', 'sha256') if k in item})
        return result
    if pinned_sha and item['sha256'] != pinned_sha:
        raise ValueError(f'original --manifest-sha does not match source manifest: {run_id}')
    payload = item.get('payload')
    result.update({'exists': True, 'bytes': item['bytes'], 'sha256': item['sha256']})
    if isinstance(payload, dict):
        result['summary'] = {key: payload[key] for key in ('schema', 'run_id', 'commit', 'git_commit',
                                                           'config_sha256', 'seed', 'source_fingerprint') if key in payload}
    return result


def _run_inventory(root, run_id, now):
    run_dir = Path(root) / 'runs' / run_id
    row = {'run_id': run_id, 'exists': run_dir.is_dir(), 'queue': _queue(root, run_id)}
    if not run_dir.is_dir() or run_dir.is_symlink():
        return row
    terminal_raw = _read(run_dir / 'terminal.json')
    run_payload = None
    for name in ('run.json', 'terminal.json', 'owner.json', 'source.json'):
        item = _read(run_dir / name)
        if item is None:
            continue
        if item.get('oversize'):
            row[name] = {k: item[k] for k in ('oversize', 'bytes', 'sha256')}
            continue
        payload = item.get('payload')
        if not isinstance(payload, dict):
            row[name] = {'bytes': item['bytes'], 'sha256': item['sha256'], 'parseable_json': False}
            continue
        record = {'bytes': item['bytes'], 'sha256': item['sha256']}
        if name == 'run.json':
            run_payload = payload
            git = payload.get('git') or {}
            config = payload.get('config') or {}
            ops_source = payload.get('ops_source') or {}
            inputs = payload.get('inputs') or []
            resources = payload.get('resources') or {}
            record.update({'schema': payload.get('schema'), 'run_id': payload.get('run_id'),
                           'kind': payload.get('kind'), 'seed': payload.get('seed'),
                           'git_commit': git.get('commit'), 'git_dirty': git.get('dirty'),
                           'ops_source': ops_source,
                           'config_sha256': config.get('sha256') if isinstance(config, dict) else None,
                           'input_sha256': [x.get('sha256') for x in inputs if isinstance(x, dict)],
                           'input_bytes': [x.get('bytes') for x in inputs if isinstance(x, dict)],
                           'resources': resources, 'created_at': payload.get('created_at'),
                           'argv_sha256': sha(canonical(payload.get('argv', []))),
                           'pinned_files': payload.get('pinned_files'), 'overlay': payload.get('overlay')})
            created = payload.get('created_at')
            wall = resources.get('wall_seconds') if isinstance(resources, dict) else None
            terminal_payload = terminal_raw.get('payload') if isinstance(terminal_raw, dict) else None
            actual_spent = terminal_payload.get('wall_seconds') if isinstance(terminal_payload, dict) else None
            if isinstance(actual_spent, (int, float)) and actual_spent >= 0:
                record['spent_seconds'] = actual_spent
                record['spent_basis'] = 'terminal.wall_seconds'
            elif isinstance(created, (int, float)):
                record['spent_seconds'] = max(0, now - created)
                record['spent_basis'] = 'elapsed_since_run_record_created'
            else:
                record['spent_seconds'] = None
                record['spent_basis'] = 'unknown'
            record['wall_seconds'] = wall
            record['remaining_wall_seconds'] = max(0, wall - record['spent_seconds']) if isinstance(wall, (int, float)) and record['spent_seconds'] is not None else None
            config_path = config.get('path') if isinstance(config, dict) else None
            if isinstance(config_path, str):
                record['config_basename'] = Path(config_path).name
        elif name == 'terminal.json':
            record.update({key: payload.get(key) for key in ('schema', 'run_id', 'status', 'exit_code', 'wall_seconds', 'wall_cap_exceeded')})
        elif name == 'owner.json':
            record.update(_owner_summary(payload))
        else:
            record.update({key: payload.get(key) for key in ('schema', 'run_id', 'commit', 'git_commit', 'config_sha256', 'seed', 'source_fingerprint') if key in payload})
        row[name[:-5]] = record
    if isinstance(run_payload, dict):
        row['manifest'] = _manifest_inventory(root, run_id, run_payload.get('argv'))
    return row


def _file_check(base, paths):
    out = {}
    for rel in paths:
        path = Path(rel) if rel.startswith('/') else base / rel
        if path.is_symlink() or not path.is_file():
            out[rel] = {'exists': False}
            continue
        size = path.stat().st_size
        if rel in GUIDES:
            if size > MAX_GUIDE_BYTES:
                out[rel] = {'exists': True, 'oversize': True, 'bytes': size, 'sha256': _hash_file(path)}
                continue
            raw = path.read_bytes()
            out[rel] = {'exists': True, 'bytes': len(raw), 'sha256': sha(raw),
                        'private_receipt_text': raw.decode('utf-8', errors='replace')}
        else:
            out[rel] = {'exists': True, 'bytes': size, 'sha256': _hash_file(path)}
    return out


def _drivers(paths):
    rows = []
    names = {'owner.json', 'summary-final.json', 'verify-resume-terminal.json', 'terminal.json'}
    for path in paths:
        record = {'directory': path.name, 'exists': path.is_dir(), 'files': []}
        if path.is_dir() and not path.is_symlink():
            candidates = set()
            for pattern in ('driver*.json', '*driver-request*.json', '*request*.json'):
                candidates.update(path.glob(pattern))
            for exact in names:
                if (path / exact).is_file():
                    candidates.add(path / exact)
            for item in sorted(candidates):
                result = _read(item)
                if result is None:
                    continue
                entry = {'name': item.name, 'sha256': result['sha256'], 'bytes': result['bytes']}
                payload = result.get('payload')
                if isinstance(payload, dict):
                    entry.update({key: payload[key] for key in DRIVER_FIELDS if key in payload})
                    entry['owner'] = _owner_summary(payload)
                record['files'].append(entry)
        rows.append(record)
    return rows


def inspect(req):
    required = {'action', 'root', 'deployment', 'ops_dirs', 'guide_paths', 'module_paths', 'data_root', 'run_ids'}
    if type(req) is not dict or set(req) != required or req.get('action') != 'inventory':
        raise ValueError('inventory accepts only its fixed read-only fields')
    root = safe_path(req['root'], 'server root', True)
    deployment = safe_path(req['deployment'], 'deployment', True)
    data_root = safe_path(req['data_root'], 'data_root')
    if tuple(req['run_ids']) != RUN_IDS:
        raise ValueError('inventory run_ids must be exactly the original twelve P/G train and verify IDs in fixed order')
    if type(req['ops_dirs']) is not list or len(req['ops_dirs']) != len(DRIVER_DIRS):
        raise ValueError('exactly the three original driver directories are required')
    drivers = [safe_path(p, 'driver directory') for p in req['ops_dirs']]
    if tuple(p.name for p in drivers) != DRIVER_DIRS:
        raise ValueError('driver directories must be the original P-train, P-verify and G-half directories')
    if tuple(req['guide_paths']) != GUIDES:
        raise ValueError('guide_paths must equal the complete fixed guide list in order')
    if tuple(req['module_paths']) != MODULES:
        raise ValueError('module_paths must be the fixed A/B entrypoint list')
    now = time.time()
    root_stat = None
    if data_root.exists():
        st = data_root.stat()
        usage = shutil.disk_usage(data_root)
        root_stat = {'exists': True, 'is_directory': data_root.is_dir(), 'device': st.st_dev,
                     'free_bytes': usage.free, 'total_bytes': usage.total}
    else:
        root_stat = {'exists': False}
    return {
        'schema': 'ur001-takeover-inventory-v1',
        'captured_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'server_root_basename': root.name,
        'data_root': {'path': str(data_root), **root_stat},
        'runs': [_run_inventory(root, run_id, now) for run_id in RUN_IDS],
        'drivers': _drivers(drivers),
        'guides': _file_check(deployment, req['guide_paths']),
        'modules': _file_check(deployment, req['module_paths']),
        'python_dependencies': {name: importlib.util.find_spec(name) is not None for name in PYTHON_DEPS},
        'read_scope': {'run_ids': list(RUN_IDS), 'read_logs': False, 'read_data': False,
                       'read_checkpoints': False, 'read_test': False, 'mutations': False},
    }


def main():
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError('request exceeds bounded stdin size')
    req = json.loads(raw)
    result = inspect(req)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
