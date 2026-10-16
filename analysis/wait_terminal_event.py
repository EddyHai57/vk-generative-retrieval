"""One finite passive RPC wait for explicitly requested terminal.json metadata.

stdin: {root: absolute scheduler root, run_ids: [...], timeout_seconds: (0,1800]}.
Linux inotify/select only: no polling, sleep, subprocess, daemon, submission or
filesystem writes. This observer does not acquire or replace scientific ownership.
Other platforms return UNKNOWN if no terminal already exists.
"""
import ctypes
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import stat
import struct
import sys
import time
sys.dont_write_bytecode = True
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
IN_ISDIR = 0x40000000
JSON_LIMIT = 64 * 1024
MASK = IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE | IN_DELETE_SELF | IN_MOVE_SELF


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def safe_path(path, root):
    path, root = Path(path), Path(root)
    current = root
    for part in path.relative_to(root).parts:
        current = current/part
        if current.is_symlink():
            raise ValueError('terminal metadata symlink prohibited')
    return path


def terminal(root, rid):
    path = safe_path(root/'runs'/rid/'terminal.json', root)
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > JSON_LIMIT:
            raise ValueError('bounded regular terminal JSON required')
        data = stream.read(JSON_LIMIT+1)
    if len(data) > JSON_LIMIT:
        raise ValueError('terminal JSON grew beyond bound')
    try:
        value = json.loads(data)
    except json.JSONDecodeError:
        # A create event may race the writer; CLOSE_WRITE will wake us again.
        return None
    if type(value) is not dict or type(value.get('status')) is not str or not value['status']:
        raise ValueError('terminal status object required')
    return {'run_id': rid, 'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'value': value, 'read_at_utc': utc()}


class LinuxEvents:
    """Kernel event backend; no path discovery beyond armed directories."""
    def __init__(self):
        if sys.platform != 'linux':
            raise NotImplementedError('Linux inotify unavailable on this platform')
        self.lib = ctypes.CDLL(None, use_errno=True)
        self.lib.inotify_init1.argtypes = [ctypes.c_int]
        self.lib.inotify_init1.restype = ctypes.c_int
        self.lib.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self.lib.inotify_add_watch.restype = ctypes.c_int
        self.fd = self.lib.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
        if self.fd < 0:
            error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
        self.paths = {}
    def arm(self, path):
        # ONLYDIR and DONT_FOLLOW close the last-component watch symlink race.
        wd = self.lib.inotify_add_watch(self.fd, os.fsencode(path), MASK | 0x01000000 | 0x02000000)
        if wd < 0:
            error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
        self.paths[wd] = Path(path)
    def wait(self, seconds):
        try:
            ready, _, _ = select.select([self.fd], [], [], seconds)
        except InterruptedError:
            return []
        if not ready:
            return []
        try:
            data = os.read(self.fd, 64*1024)
        except BlockingIOError:
            return []
        events = []; offset = 0
        while offset < len(data):
            wd, mask, _, length = struct.unpack_from('iIII', data, offset)
            offset += 16
            name = os.fsdecode(data[offset:offset+length].split(b'\0', 1)[0]); offset += length
            events.append((self.paths.get(wd), name, mask))
        return events
    def close(self):
        os.close(self.fd)


def wait_terminal(request, *, backend_factory=None, clock=time.monotonic):
    if type(request) is not dict or set(request) != {'root', 'run_ids', 'timeout_seconds'}:
        raise ValueError('explicit root/run_ids/timeout_seconds required')
    ids, duration = request['run_ids'], request['timeout_seconds']
    if (type(ids) is not list or not 1 <= len(ids) <= 64 or
        any(type(rid) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', rid) or
            re.search(r'test|checkpoint|embedding|prediction|tensor', rid, re.I) for rid in ids) or len(set(ids)) != len(ids)):
        raise ValueError('safe unique explicitly requested run identities required')
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 1800:
        raise ValueError('finite timeout in (0,1800] required')
    root = Path(request['root'])
    if not root.is_absolute() or '..' in root.parts or not root.is_dir():
        raise ValueError('existing absolute scheduler root required')
    root = root.resolve(strict=True)
    runs = safe_path(root/'runs', root)
    if not runs.is_dir():
        raise ValueError('existing scheduler runs directory required')
    deadline = clock()+duration
    def snapshot(status='TERMINAL_EVENT', reason=None):
        values = {rid: terminal(root, rid) for rid in ids}
        first = next((value for value in values.values() if value is not None), None)
        out = {'readonly': True, 'status': 'TERMINAL_EVENT' if first else status,
               'first': first, 'terminal': {rid: value is not None for rid, value in values.items()},
               'read_at_utc': utc(), 'selection': 'first observed; simultaneous files use request order'}
        if reason:
            out['reason'] = reason
        return out
    initial = snapshot()
    if initial['first']:
        return initial
    try:
        backend = (backend_factory or LinuxEvents)()
    except NotImplementedError as exc:
        return snapshot('UNKNOWN', str(exc))
    armed = set()
    def arm_runs():
        for rid in ids:
            directory = safe_path(runs/rid, root)
            if directory not in armed and directory.exists():
                if not directory.is_dir():
                    raise ValueError('run identity must be a directory')
                try:
                    backend.arm(directory)
                except FileNotFoundError:
                    continue  # Parent watch retains the next CREATE/MOVED_TO event.
                armed.add(directory)
    try:
        backend.arm(runs)
        arm_runs()
        # Recheck after every watch has been armed, closing initial discovery races.
        current = snapshot()
        if current['first']:
            return current
        while True:
            remaining = deadline-clock()
            if remaining <= 0:
                return snapshot('WAIT_TIMEOUT')
            events = backend.wait(remaining)
            relevant = False
            for path, name, mask in events:
                if mask & (IN_Q_OVERFLOW | IN_DELETE_SELF | IN_MOVE_SELF | IN_IGNORED):
                    raise ValueError('inotify observation lost; no polling fallback')
                if path == runs and name in ids and mask & (IN_CREATE | IN_MOVED_TO):
                    relevant = True
                elif path in armed and name == 'terminal.json' and mask & (IN_CLOSE_WRITE | IN_MOVED_TO):
                    relevant = True
            if relevant:
                arm_runs()
                current = snapshot()
                if current['first']:
                    return current
    finally:
        backend.close()


def main():
    data = sys.stdin.buffer.read(JSON_LIMIT+1)
    if len(data) > JSON_LIMIT:
        raise ValueError('bounded private request required')
    print(json.dumps(wait_terminal(json.loads(data)), sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
