"""One-shot, bounded, read-only receipts for explicitly named scheduler runs.

stdin: {root: absolute scheduler root, run_ids: [...], driver_records: [...]}.
Only queue/*/*-<run_id>.json, two known rejection reason spellings, run
terminal.json/run.json/job.json and logs/{stdout,stderr}.log are opened.
Explicit driver JSON records are metadata, never an invitation to scan OPS.
No global run discovery: unknown scheduler PID schemas must not imply idle.
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
sys.dont_write_bytecode = True
JSON_LIMIT = 256 * 1024
TAIL_LIMIT = 8192
CHUNK = 64 * 1024
FORBIDDEN = re.compile(r'test|checkpoint|embedding|prediction|tensor', re.I)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def safe_path(path, root):
    path, root = Path(path), Path(root)
    if '..' in path.parts:
        raise ValueError('no traversal')
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        raise ValueError('outside explicit metadata scope')
    probe = root
    for part in parts:
        probe = probe/part
        if probe.is_symlink():
            raise ValueError('metadata symlink prohibited')
    return path


def stream_summary(path):
    digest, count, tail = hashlib.sha256(), 0, b''
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk); count += len(chunk)
            tail = (tail+chunk)[-TAIL_LIMIT:]
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': count,
            'tail': tail.decode('utf-8', errors='replace'), 'read_at_utc': utc()}


def receipt(path, root, *, text=False):
    path = safe_path(path, root)
    try:
        info = path.stat()
    except FileNotFoundError:
        return {'path': str(path), 'status': 'missing', 'read_at_utc': utc()}
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('regular metadata file required')
    if text:
        return dict(stream_summary(path), status='present')
    if info.st_size > JSON_LIMIT:
        raise ValueError('bounded metadata JSON required')
    with path.open('rb') as stream:
        data = stream.read(JSON_LIMIT+1)
    if len(data) > JSON_LIMIT:
        raise ValueError('metadata grew beyond JSON bound')
    value = json.loads(data)
    if type(value) is not dict:
        raise ValueError('metadata object schema required')
    return {'path': str(path), 'status': 'present', 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'value': value, 'read_at_utc': utc()}


def process_state(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('positive PID required')
    result = {'pid': pid, 'read_at_utc': utc()}
    try:
        # Linux /proc stat fields: state, ppid, pgrp, session. The comm may contain ')'.
        fields = Path('/proc')/str(pid)/'stat'
        values = fields.read_text().rsplit(') ', 1)[1].split()
        result.update(alive=values[0] not in ('Z', 'X'), actual_ppid=int(values[1]), sid=int(values[3]), process_state=values[0])
    except (FileNotFoundError, ProcessLookupError):
        result['alive'] = False
    except PermissionError:
        result.update(alive=None, error='proc_permission_denied')
    except (IndexError, ValueError):
        result.update(alive=None, error='proc_schema_unknown')
    return result


def collect(request):
    if type(request) is not dict or set(request) != {'root', 'run_ids', 'driver_records'}:
        raise ValueError('explicit root/run_ids/driver_records required')
    root = Path(request['root'])
    ids, drivers = request['run_ids'], request['driver_records']
    if not root.is_absolute() or '..' in root.parts or not root.is_dir():
        raise ValueError('existing absolute scheduler root required')
    root = root.resolve(strict=True)
    if type(ids) is not list or not 1 <= len(ids) <= 64 or len(set(ids)) != len(ids):
        raise ValueError('one to 64 unique explicit run IDs required')
    if any(type(rid) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', rid) or FORBIDDEN.search(rid) for rid in ids):
        raise ValueError('safe non-test run identity required')
    if type(drivers) is not list or len(drivers) > 32 or any(type(p) is not str for p in drivers):
        raise ValueError('bounded explicit driver record list required')
    result = {'readonly': True, 'root': str(root), 'read_at_utc': utc(), 'runs': {}, 'drivers': [],
              'global_busy': 'UNKNOWN; only explicitly requested identities inspected'}
    for rid in ids:
        entry = {'queue': {}, 'logs': {}}
        for stage in ('pending', 'claimed', 'rejected'):
            directory = safe_path(root/'queue'/stage, root)
            records = []
            for path in sorted(directory.glob('*-'+rid+'.json')):
                if FORBIDDEN.search(path.name):
                    raise ValueError('test or sensitive named queue artifact prohibited')
                record = receipt(path, root)
                if record['value'].get('run_id') != rid:
                    raise ValueError('queue job/run identity mismatch')
                if stage == 'rejected':
                    record['reasons'] = [receipt(p, root, text=True) for p in (path.with_suffix('.reason'), Path(str(path)+'.reason')) if p.exists()]
                records.append(record)
            entry['queue'][stage] = records
        for name in ('terminal', 'run', 'job'):
            record = receipt(root/'runs'/rid/(name+'.json'), root)
            if record.get('status') == 'present' and name in ('run', 'job'):
                value = record['value']
                if value.get('run_id', rid) != rid:
                    raise ValueError('run metadata identity mismatch')
            entry[name] = record
        for channel in ('stdout', 'stderr'):
            entry['logs'][channel] = receipt(root/'runs'/rid/'logs'/(channel+'.log'), root, text=True)
        result['runs'][rid] = entry
    for name in drivers:
        path = Path(name)
        if not path.is_absolute() or path.suffix != '.json' or FORBIDDEN.search(path.name):
            raise ValueError('explicit absolute driver metadata JSON required')
        # Exact requested files only; directory and basename symlinks are rejected.
        record = receipt(path, Path(path.anchor))
        if record['status'] == 'present' and type(record['value'].get('pid')) is int:
            record['process'] = process_state(record['value']['pid'])
        result['drivers'].append(record)
    return result


def main():
    data = sys.stdin.buffer.read(JSON_LIMIT+1)
    if len(data) > JSON_LIMIT:
        raise ValueError('bounded private request required')
    print(json.dumps(collect(json.loads(data)), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
