"""Shared helpers: atomic/write-once JSON, sha256, run ids, process identity, host probes.

Every host read goes through a ``Probes`` instance so tests can inject fixture
``/proc`` and ``/sys`` trees, a fake ``nvidia-smi``, a clock and a sleep.
"""
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, List

RUN_ID_RE = re.compile(r'^[a-z0-9][a-z0-9._-]{0,63}$')
READ_LIMIT = 64 * 1024
STAT_LIMIT = 4096
STARTING_GRACE_SECONDS = 60
NVIDIA_TIMEOUT_SECONDS = 5
TERMINAL_STATUSES = ('succeeded', 'failed', 'wall_cap_exceeded', 'stopped', 'launch_failed')
OCCUPYING_STATUSES = ('starting', 'running', 'orphaned')


class ProbeError(RuntimeError):
    """A host probe could not produce a value (recoverable: this round admits nothing)."""


@dataclasses.dataclass
class Probes:
    proc_root: Path = Path('/proc')
    sys_root: Path = Path('/sys')
    nvidia_smi: List[str] = dataclasses.field(default_factory=lambda: ['nvidia-smi'])
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    cpu_count: Callable[[], int] = os.cpu_count
    statvfs: Callable = os.statvfs


# ---------------------------------------------------------------- files / hashes

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    """Streaming sha256; returns (hexdigest, byte count)."""
    digest = hashlib.sha256()
    size = 0
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n').encode()


def _write_tmp(path, value):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp')
    data = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return tmp


def write_json_atomic(path, value):
    os.replace(_write_tmp(path, value), path)


def write_json_once(path, value, mode=0o400):
    """Atomically create ``path``; raises FileExistsError if it already exists."""
    tmp = _write_tmp(path, value)
    try:
        os.chmod(tmp, mode)
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def read_bytes_limited(path, limit=READ_LIMIT):
    """Read-only; None when missing, unreadable or larger than ``limit``."""
    try:
        with open(path, 'rb') as handle:
            data = handle.read(limit + 1)
    except OSError:
        return None
    return None if len(data) > limit else data


def read_json_limited(path, limit=READ_LIMIT):
    data = read_bytes_limited(path, limit)
    if data is None:
        return None
    try:
        value = json.loads(data)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def valid_run_id(value):
    return isinstance(value, str) and RUN_ID_RE.match(value) is not None


# ---------------------------------------------------------------- processes

def parse_stat(text):
    """Parse /proc/<pid>/stat; None if malformed."""
    try:
        rest = text[text.rindex(')') + 2:].split()
        return {'ppid': int(rest[1]), 'pgrp': int(rest[2]), 'session': int(rest[3]),
                'ticks': int(rest[11]) + int(rest[12]), 'birth': int(rest[19])}
    except (ValueError, IndexError):
        return None


def proc_stat(probes, pid):
    data = read_bytes_limited(Path(probes.proc_root) / str(pid) / 'stat', STAT_LIMIT)
    return None if data is None else parse_stat(data.decode('utf-8', 'replace'))


def has_proc(probes):
    return (Path(probes.proc_root) / 'self').is_dir() or (Path(probes.proc_root) / '1').is_dir()


def identity(pid, probes=None):
    """{pid, pgid, sid, birth}; birth = /proc stat field 22 on Linux, else None."""
    probes = probes or Probes()
    stat = proc_stat(probes, pid)
    return {'pid': pid, 'pgid': os.getpgid(pid), 'sid': os.getsid(pid),
            'birth': None if stat is None else stat['birth']}


def alive(probes, pid, birth):
    """Same process still running: pid exists and (when recorded) birth matches."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if has_proc(probes):
        stat = proc_stat(probes, pid)
        return stat is not None and (birth is None or stat['birth'] == birth)
    if birth is not None:
        return False
    try:                       # non-Linux dev hosts only: signal 0 delivers nothing
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_table(probes):
    """{pid: {ppid, ticks, rss_kib}} from /proc; unparseable rows skipped; None without /proc."""
    root = Path(probes.proc_root)
    if not has_proc(probes):
        return None
    table = {}
    try:
        names = os.listdir(root)
    except OSError:
        return None
    for name in names:
        if not name.isdigit():
            continue
        stat = proc_stat(probes, int(name))
        if stat is None:
            continue
        rss = 0
        status = read_bytes_limited(root / name / 'status', READ_LIMIT)
        if status is not None:
            for line in status.decode('utf-8', 'replace').splitlines():
                if line.startswith('VmRSS:'):
                    try:
                        rss = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
        table[int(name)] = {'ppid': stat['ppid'], 'ticks': stat['ticks'], 'rss_kib': rss}
    return table


def descendants(table, root_pid):
    """root_pid plus every process reachable through ppid links."""
    children = {}
    for pid, row in table.items():
        children.setdefault(row['ppid'], []).append(pid)
    found, stack = set(), [root_pid]
    while stack:
        pid = stack.pop()
        if pid in found:
            continue
        found.add(pid)
        stack.extend(children.get(pid, ()))
    return found


# ---------------------------------------------------------------- GPU

def _nvidia(probes, args):
    try:
        out = subprocess.run([*probes.nvidia_smi, *args], stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, timeout=NVIDIA_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _csv_ints(text, width):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != width:
            raise ValueError(line)
        rows.append([int(float(p)) for p in parts])
    return rows


def gpu_memory(probes):
    """(total_mib, used_mib) summed over GPUs, or None when nvidia-smi is unavailable."""
    text = _nvidia(probes, ['--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'])
    try:
        rows = _csv_ints(text, 2) if text is not None else None
    except ValueError:
        return None
    if not rows:
        return None
    return sum(r[0] for r in rows), sum(r[1] for r in rows)


def compute_apps(probes):
    """{pid: used_mib} or None."""
    text = _nvidia(probes, ['--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'])
    if text is None:
        return None
    try:
        rows = _csv_ints(text, 2)
    except ValueError:
        return None
    apps = {}
    for pid, used in rows:
        apps[pid] = apps.get(pid, 0) + used
    return apps


def vram_of(pids, apps, table):
    """VRAM used by ``pids``; None when apps are unknown or the pid namespace does not match."""
    if apps is None or table is None:
        return None
    if apps and not any(pid in table for pid in apps):
        return None
    return sum(used for pid, used in apps.items() if pid in pids)


# ---------------------------------------------------------------- CPU / RAM / disk

def _read_text(path):
    data = read_bytes_limited(path, READ_LIMIT)
    return None if data is None else data.decode('utf-8', 'replace')


def cgroup_dir(probes):
    text = _read_text(Path(probes.proc_root) / 'self' / 'cgroup')
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith('0::'):
            path = Path(probes.sys_root) / 'fs' / 'cgroup' / line[3:].strip().lstrip('/')
            return path if path.is_dir() else None
    return None


def cpu_quota(probes):
    """Cores allowed by cgroup v2 cpu.max, or None when unlimited / not cgroup v2."""
    directory = cgroup_dir(probes)
    text = None if directory is None else _read_text(directory / 'cpu.max')
    if text is None:
        return None
    parts = text.split()
    if len(parts) != 2 or parts[0] == 'max':
        return None
    try:
        quota, period = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return quota / period if quota > 0 and period > 0 else None


def cgroup_cpu_stat(probes):
    """cgroup cpu.stat as {key: int}, or None."""
    directory = cgroup_dir(probes)
    text = None if directory is None else _read_text(directory / 'cpu.stat')
    if text is None:
        return None
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip('-').isdigit():
            values[parts[0]] = int(parts[1])
    return values


def _proc_cpu_times(probes):
    text = _read_text(Path(probes.proc_root) / 'stat')
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith('cpu '):
            try:
                fields = [int(x) for x in line.split()[1:9]]
            except ValueError:
                return None
            if len(fields) < 5:
                return None
            return sum(fields), fields[3] + fields[4]
    return None


def cpu_free(probes):
    """(free_cores, total_cores). cgroup quota -> cpu.stat usage delta; else /proc/stat idle delta."""
    count = probes.cpu_count() or 1
    quota = cpu_quota(probes)
    if quota is not None:
        total = min(count, quota)
        first = cgroup_cpu_stat(probes)
        if first is not None and 'usage_usec' in first:
            t0 = probes.clock()
            probes.sleep(1.0)
            second = cgroup_cpu_stat(probes)
            dt = probes.clock() - t0
            if second is not None and 'usage_usec' in second and dt > 0:
                used = (second['usage_usec'] - first['usage_usec']) / (dt * 1e6)
                return min(max(total - used, 0.0), total), total
    first = _proc_cpu_times(probes)
    if first is None:
        raise ProbeError('cpu probe failed: no cgroup cpu.stat and no /proc/stat')
    probes.sleep(1.0)
    second = _proc_cpu_times(probes)
    if second is None or second[0] <= first[0]:
        raise ProbeError('cpu probe failed: /proc/stat did not advance')
    idle_fraction = (second[1] - first[1]) / (second[0] - first[0])
    return min(max(idle_fraction, 0.0), 1.0) * count, count


def ram_info(probes):
    """(total_mib, free_mib): min(MemTotal, memory.max); MemAvailable capped by memory.max-current."""
    text = _read_text(Path(probes.proc_root) / 'meminfo')
    if text is None:
        raise ProbeError('ram probe failed: no meminfo')
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            values[parts[0].rstrip(':')] = int(parts[1])
    if 'MemTotal' not in values or 'MemAvailable' not in values:
        raise ProbeError('ram probe failed: meminfo incomplete')
    total = values['MemTotal'] / 1024
    free = values['MemAvailable'] / 1024
    directory = cgroup_dir(probes)
    if directory is not None:
        limit = _read_text(directory / 'memory.max')
        current = _read_text(directory / 'memory.current')
        if limit is not None and limit.strip().isdigit():
            limit_mib = int(limit.strip()) / (1024 * 1024)
            total = min(total, limit_mib)
            if current is not None and current.strip().isdigit():
                free = min(free, limit_mib - int(current.strip()) / (1024 * 1024))
    return total, max(free, 0.0)


def disk_info(probes, path):
    """(total_gib, free_gib) of the filesystem holding ``path``."""
    try:
        st = probes.statvfs(str(path))
    except OSError as error:
        raise ProbeError(f'disk probe failed: {error}') from error
    gib = 1024 ** 3
    return st.f_blocks * st.f_frsize / gib, st.f_bavail * st.f_frsize / gib


# ---------------------------------------------------------------- run status

def run_status(run_dir, probes, now=None):
    """(status, info). info = {run, owner, terminal} as read (None when missing/corrupt)."""
    run_dir = Path(run_dir)
    now = probes.clock() if now is None else now
    run = read_json_limited(run_dir / 'run.json')
    owner = read_json_limited(run_dir / 'owner.json')
    terminal = read_json_limited(run_dir / 'terminal.json')
    info = {'run': run, 'owner': owner, 'terminal': terminal}
    if terminal is not None and terminal.get('status') in TERMINAL_STATUSES:
        return terminal['status'], info
    if owner is None:
        created = run.get('created_at') if run else None
        if isinstance(created, (int, float)) and now - created < STARTING_GRACE_SECONDS:
            return 'starting', info
        return 'lost', info
    sup = owner.get('supervisor') or {}
    job = owner.get('job') or {}
    if alive(probes, sup.get('pid'), sup.get('birth')):
        return 'running', info
    if alive(probes, job.get('pid'), job.get('birth')):
        return 'orphaned', info
    return 'lost', info


def started_at(info):
    """N5: settle/wall clock start = owner.started_at, else run.created_at."""
    owner, run = info.get('owner') or {}, info.get('run') or {}
    value = owner.get('started_at', run.get('created_at'))
    return value if isinstance(value, (int, float)) else None


def tracked_root_pid(status, info):
    """Process whose descendants belong to the run: supervisor, or the job when orphaned."""
    owner = info.get('owner') or {}
    key = 'job' if status == 'orphaned' else 'supervisor'
    pid = (owner.get(key) or {}).get('pid')
    return pid if isinstance(pid, int) else None
