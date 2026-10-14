"""Hold one finite, foreground, authenticated SSH control master (formalNN, replaces formal{N-1}).

usage: hold_mux.py --config C --n N [--dry-run]

Guards (from DEPLOY-01 ``hold_mux_n.py``): the previous master must have actually exited
(terminal receipt, socket gone, owner/master pids dead); no duplicate state for N.
Writes ``ssh-control-state-formalNN.json`` / ``ssh-control-terminal-formalNN.json`` in ``state_dir``.
The socket dir is a new 0700 dir under ``socket_prefix``. Exits after ``mux_max_seconds``.
``--dry-run`` prints the ssh argv and the guard verdict without starting anything.
"""
import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toolcfg  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--config')
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if not 2 <= a.n <= 99:
        p.error('--n must be 2..99 (formal{N-1} must exist)')
    return a


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def guard(directory, n):
    """Raise ValueError unless formal{n-1} actually exited and formal{n} is fresh."""
    cur, prev = f'formal{n:02d}', f'formal{n - 1:02d}'
    old = json.loads((directory / f'ssh-control-state-{prev}.json').read_text())
    done = json.loads((directory / f'ssh-control-terminal-{prev}.json').read_text())
    if Path(old['socket_path']).exists() or done.get('socket_exists') is not False or type(done.get('actual_master_exit')) is not int:
        raise ValueError(prev + ' master must have actually exited; no duplicate master')
    if any(_alive(old[k]) for k in ('master_pid', 'owner_pid')):
        raise ValueError(prev + ' owner/master still live')
    if (directory / f'ssh-control-state-{cur}.json').exists():
        raise ValueError(cur + ' state exists; no duplicate master')
    return cur, prev


def master_argv(host, socket, log):
    return ['ssh', '-E', str(log), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ControlMaster=yes',
            '-o', 'ControlPersist=no', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
            '-S', str(socket), '-N', host]


def main(argv=None):
    a = parse_args(argv)
    cfg = toolcfg.load(a.config)
    directory = toolcfg.state_dir(cfg)
    max_seconds = int(cfg['mux_max_seconds'])
    if a.dry_run:
        try:
            verdict = list(guard(directory, a.n))
        except (OSError, ValueError) as error:
            verdict = f'refuse: {error}'
        print(json.dumps({'dry_run': True, 'guard': verdict, 'max_seconds': max_seconds,
                          'ssh_argv': master_argv(cfg['ssh_host'], '<socket>', '<log>')}, indent=1))
        return 0
    cur, prev = guard(directory, a.n)
    parent, prefix = os.path.split(cfg['socket_prefix'])
    sock_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    os.chmod(sock_dir, 0o700)
    socket, log = sock_dir / 'ctl', sock_dir / 'ssh-diagnostic.log'
    started = time.monotonic()
    deadline = started + max_seconds
    with log.open('xb') as diagnostic:
        os.chmod(log, 0o600)
        process = subprocess.Popen(master_argv(cfg['ssh_host'], socket, log), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=diagnostic)
        try:
            bootstrap = time.monotonic() + 30
            while not socket.exists():
                if process.poll() is not None or time.monotonic() >= bootstrap:
                    raise RuntimeError('foreground authenticated master bootstrap failed')
                time.sleep(.1)
            check = subprocess.run(['ssh', '-S', str(socket), '-O', 'check', cfg['ssh_host']],
                                   capture_output=True, timeout=5)
            if check.returncode or not stat.S_ISSOCK(socket.stat().st_mode):
                raise RuntimeError('actual owned master check failed')
            state = {'socket_path': str(socket), 'created_at': time.time(), 'started_monotonic': started,
                     'deadline_monotonic': deadline, 'max_seconds': max_seconds, 'diagnostic_log_path': str(log),
                     'master_pid': process.pid, 'owner_pid': os.getpid(), 'foreground_actual_wait': True,
                     'new_server_resource': False, 'server_config_change': False, 'replaces': prev}
            with (directory / f'ssh-control-state-{cur}.json').open('x') as f:
                json.dump(state, f)
            print(json.dumps({'status': 'authenticated_foreground_transport_ready', 'master_pid': process.pid,
                              'finite_seconds': max_seconds}), flush=True)
            try:
                exit_code = process.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    exit_code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait(timeout=10)
            with (directory / f'ssh-control-terminal-{cur}.json').open('x') as f:
                json.dump({'actual_master_exit': exit_code, 'elapsed_seconds': time.monotonic() - started,
                           'socket_exists': socket.exists(), 'remote_job_state': 'UNKNOWN_REQUIRES_REMOTE_RECEIPT'}, f)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
