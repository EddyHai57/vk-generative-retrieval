"""Shared fixtures for tests/test_ops_*.py (standard library only)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / 'ops'
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from vkgr_ops import common  # noqa: E402

PY = sys.executable
FAKE_NVSMI = ROOT / 'tests' / 'fixtures' / 'ops' / 'fake-nvidia-smi'
NOW = 1_760_000_000.0
GIT_ID = ['-c', 'user.name=ops-test', '-c', 'user.email=ops-test@example.invalid',
          '-c', 'commit.gpgsign=false', '-c', 'init.defaultBranch=main']


def git(repo, *args):
    return subprocess.run(['git', *GIT_ID, '-C', str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def make_repo(path, files=None):
    """Clean fixture repo with one commit; returns its HEAD sha."""
    path.mkdir(parents=True)
    subprocess.run(['git', *GIT_ID, 'init', '-q', str(path)], check=True)
    for rel, text in (files or {'README.md': 'fixture\n'}).items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    git(path, 'add', '-A')
    git(path, 'commit', '-q', '-m', 'base')
    return git(path, 'rev-parse', 'HEAD')


def sha(text):
    return common.sha256_bytes(text.encode() if isinstance(text, str) else text)


class FakeStat:
    def __init__(self, total_gib, free_gib):
        self.f_frsize = 1024 * 1024
        self.f_blocks = int(total_gib * 1024)
        self.f_bavail = int(free_gib * 1024)


class FixtureHost:
    """A fake /proc + /sys tree whose counters advance linearly on each sleep()."""

    def __init__(self, base, *, cpu_count=16, idle_fraction=0.8, ram_total_mib=100000,
                 ram_avail_mib=60000, cpu_max='max 100000', cgroup_usage_per_s=None,
                 nr_throttled=0, throttled_usec=0, memory_max='max', memory_current=0,
                 disk_total_gib=1000, disk_free_gib=500, gpu='49140, 9140', apps='',
                 nvidia=True):
        self.proc = base / 'proc'
        self.sys = base / 'sys'
        self.cgroup = self.sys / 'fs' / 'cgroup' / 'job'
        self.cgroup.mkdir(parents=True)
        (self.proc / 'self').mkdir(parents=True)
        (self.proc / 'self' / 'cgroup').write_text('0::/job\n')
        (self.proc / 'meminfo').write_text(
            f'MemTotal: {ram_total_mib * 1024} kB\nMemFree: 1 kB\nMemAvailable: {ram_avail_mib * 1024} kB\n')
        (self.cgroup / 'cpu.max').write_text(cpu_max + '\n')
        (self.cgroup / 'memory.max').write_text(f'{memory_max}\n')
        (self.cgroup / 'memory.current').write_text(f'{memory_current}\n')
        self.cpu_count = cpu_count
        self.idle_fraction = idle_fraction
        self.usage_per_s = cgroup_usage_per_s
        self.nr_throttled = nr_throttled
        self.throttled_usec = throttled_usec
        self.disk = FakeStat(disk_total_gib, disk_free_gib)
        self.nvidia = ([PY, str(FAKE_NVSMI), '--fake-gpu', gpu, '--fake-apps', apps]
                       if nvidia else [str(base / 'no-such-nvidia-smi')])
        self.pids = {}
        self.t = NOW
        self.phase = 0
        self.frozen = False
        self.render()

    def add_proc(self, pid, *, ppid=1, birth=1000, rss_kib=1024, ticks=0, tick_rate=0):
        self.pids[pid] = dict(ppid=ppid, birth=birth, rss_kib=rss_kib, ticks=ticks, rate=tick_rate)
        (self.proc / str(pid)).mkdir(exist_ok=True)
        self.render()

    def render(self):
        busy = 1000 * self.phase
        idle = int(busy * self.idle_fraction / (1 - self.idle_fraction)) if self.idle_fraction < 1 else busy
        (self.proc / 'stat').write_text(f'cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\ncpu0 1 0 0 1 0 0 0 0 0 0\n')
        usage = int((self.usage_per_s or 0) * 1e6 * (self.t - NOW))
        (self.cgroup / 'cpu.stat').write_text(
            f'usage_usec {usage}\nuser_usec 0\nsystem_usec 0\nnr_periods 10\n'
            f'nr_throttled {self.nr_throttled}\nthrottled_usec {self.throttled_usec}\n')
        for pid, p in self.pids.items():
            ticks = p['ticks'] + p['rate'] * self.phase
            fields = ['S', p['ppid'], pid, pid] + [0] * 7 + [ticks, 0] + [0] * 6 + [p['birth']] + [0] * 10
            (self.proc / str(pid) / 'stat').write_text(f'{pid} (fake job) ' + ' '.join(map(str, fields)) + '\n')
            (self.proc / str(pid) / 'status').write_text(f'Name:\tfake\nVmRSS:\t{p["rss_kib"]} kB\n')

    def sleep(self, seconds):
        self.t += seconds
        if not self.frozen:
            self.phase += 1
            self.render()

    def probes(self):
        return common.Probes(proc_root=self.proc, sys_root=self.sys, nvidia_smi=self.nvidia,
                             clock=lambda: self.t, sleep=self.sleep,
                             cpu_count=lambda: self.cpu_count, statvfs=lambda _p: self.disk)


def make_run(root, run_id, *, cpu=0, ram=0, vram=0, disk=0, created_at=NOW - 1000,
             owner=None, terminal=None, wall_seconds=3600.0):
    """Write a fake run directory. owner = dict(sup=(pid,birth), job=(pid,birth), started_at)."""
    run = Path(root) / 'runs' / run_id
    run.mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({
        'schema': 'vkgr-run-v1', 'run_id': run_id, 'kind': 'smoke', 'created_at': created_at,
        'argv': ['true'], 'stop_mode': 'signal',
        'resources': {'cpu_cores': cpu, 'ram_mib': ram, 'vram_mib': vram, 'disk_gib': disk,
                      'wall_seconds': wall_seconds}}))
    if owner is not None:
        sup, job = owner['sup'], owner['job']
        (run / 'owner.json').write_text(json.dumps({
            'supervisor': {'pid': sup[0], 'pgid': sup[0], 'sid': sup[0], 'birth': sup[1]},
            'job': {'pid': job[0], 'pgid': sup[0], 'sid': sup[0], 'birth': job[1]},
            'started_at': owner.get('started_at', created_at), 'birth_source': 'proc-stat-22'}))
    if terminal is not None:
        (run / 'terminal.json').write_text(json.dumps({
            'schema': 'vkgr-terminal-v1', 'run_id': run_id, 'status': terminal,
            'exit_code': 0, 'signal': None, 'started_at': created_at,
            'ended_at': NOW - 10, 'wall_seconds': 5.0, 'wall_cap_exceeded': False}))
    return run


def base_spec(cwd, run_id='job-1', argv=None, **overrides):
    spec = {
        'run_id': run_id, 'kind': 'smoke', 'cwd': str(cwd), 'config': None, 'seed': None,
        'inputs': [], 'pinned_manifest': None, 'pinned_files': [], 'env': [],
        'cpu_cores': 1, 'ram_mib': 100, 'vram_mib': 0, 'disk_gib': 1, 'wall_hours': 1.0,
        'stop_mode': 'signal', 'no_admission': True,
        'argv': argv or [PY, '-c', 'print("ok")'],
    }
    spec.update(overrides)
    return spec


def wait_for(path, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if Path(path).exists():
            return True
        time.sleep(0.05)
    return False


def read_json(path):
    return json.loads(Path(path).read_text())


def wait_terminal(run_dir, timeout=20.0):
    if not wait_for(Path(run_dir) / 'terminal.json', timeout):
        raise AssertionError(f'no terminal.json in {run_dir}')
    time.sleep(0.05)
    return read_json(Path(run_dir) / 'terminal.json')


def ppid_of(pid):
    out = subprocess.run(['ps', '-o', 'ppid=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) if out else None


def host_proc_snapshot(base):
    """Real process table as a fixture /proc tree (Linux: use /proc directly; macOS: from ps)."""
    if Path('/proc/self/stat').exists():
        return Path('/proc')
    proc = base / 'realproc'
    proc.mkdir()
    out = subprocess.run(['ps', '-axo', 'pid=,ppid=,rss='], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        pid, ppid, rss = line.split()
        d = proc / pid
        d.mkdir()
        fields = ['S', ppid, pid, pid] + ['0'] * 15 + ['1'] + ['0'] * 10
        (d / 'stat').write_text(f'{pid} (p) ' + ' '.join(fields) + '\n')
        (d / 'status').write_text(f'VmRSS:\t{rss} kB\n')
    return proc


def env_without_ops_root():
    env = dict(os.environ)
    env.pop('VKGR_OPS_ROOT', None)
    return env
