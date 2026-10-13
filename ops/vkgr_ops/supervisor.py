"""Detached job supervisor (design §4.2), started as ``python -I supervisor.py <run_dir>``.

M4: ``-I`` implies ``-P`` on Python 3.11, so the script directory is not on sys.path.
Only the standard library is imported before <ops>/ is inserted explicitly.
M3: the supervisor double-forks (first process exits at once and is reaped by the
launcher; the grandchild calls setsid), so it is never a child of the scheduler.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # <ops>/

from vkgr_ops import common  # noqa: E402

KILL_GRACE_SECONDS = 300
POLL_SECONDS = 0.1


def log(message):
    print(f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} {message}', flush=True)


def detach():
    """Double fork: the parent returns exit 0 immediately; the child becomes a session leader."""
    if os.fork() > 0:
        os._exit(0)
    os.setsid()


def terminal_record(run_id, started_at, returncode, reason, wall_cap_hit):
    """N7: a signal reason sent by the supervisor (first one wins) beats the exit code."""
    ended = time.time()
    exit_code = returncode if returncode is not None and returncode >= 0 else None
    signal_number = -returncode if returncode is not None and returncode < 0 else None
    if reason is not None:
        status = reason
    else:
        status = 'succeeded' if exit_code == 0 else 'failed'
    return {'schema': 'vkgr-terminal-v1', 'run_id': run_id, 'status': status, 'exit_code': exit_code,
            'signal': signal_number, 'started_at': started_at, 'ended_at': ended,
            'wall_seconds': None if started_at is None else ended - started_at,
            'wall_cap_exceeded': bool(wall_cap_hit)}


def supervise(run_dir):
    run_dir = Path(run_dir).resolve()
    run = common.read_json_limited(run_dir / 'run.json')
    if run is None:
        log('run.json unreadable; nothing to supervise')
        return 1
    run_id = run['run_id']
    stop_mode = run['stop_mode']
    wall_seconds = float(run['resources']['wall_seconds'])
    signal.signal(signal.SIGHUP, signal.SIG_IGN)            # inherited across exec (nohup)
    requests = []
    signal.signal(signal.SIGTERM, lambda *_: requests.append(time.monotonic()))
    logs = run_dir / 'logs'
    try:
        with open(logs / 'stdout.log', 'ab') as out, open(logs / 'stderr.log', 'ab') as err:
            job = subprocess.Popen(run['argv'], cwd=run['cwd'], stdin=subprocess.DEVNULL, stdout=out,
                                   stderr=err, close_fds=True, env=dict(os.environ))
    except OSError as error:
        log(f'launch_failed: cannot exec {run["argv"][0]}: {error}')
        common.write_json_once(run_dir / 'terminal.json',
                               terminal_record(run_id, None, None, 'launch_failed', False))
        return 1
    started_at = time.time()
    probes = common.Probes()
    try:
        job_identity = common.identity(job.pid, probes)
    except OSError:          # job already exited (zombie): it shares our session and group
        job_identity = {'pid': job.pid, 'pgid': os.getpgid(0), 'sid': os.getsid(0), 'birth': None}
    common.write_json_once(run_dir / 'owner.json', {
        'supervisor': common.identity(os.getpid(), probes), 'job': job_identity,
        'started_at': started_at,
        'birth_source': 'proc-stat-22' if common.has_proc(probes) else 'unavailable'})
    log(f'started job pid={job.pid} stop_mode={stop_mode} wall_seconds={wall_seconds}')
    start = time.monotonic()
    reason, wall_cap_hit, kill_at, handled = None, False, None, 0
    while True:
        try:
            returncode = job.wait(timeout=POLL_SECONDS)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        if not wall_cap_hit and now - start >= wall_seconds:
            wall_cap_hit = True
            if stop_mode == 'signal':
                log('wall cap reached: SIGTERM job')
                reason = reason or 'wall_cap_exceeded'
                os.kill(job.pid, signal.SIGTERM)                  # job pid only, never killpg
                kill_at = kill_at or now + KILL_GRACE_SECONDS
            else:
                log('wall cap reached: stop-mode none, job not signalled')
        while handled < len(requests):
            handled += 1
            if stop_mode == 'signal':
                log('SIGTERM received: forwarding SIGTERM to job')
                reason = reason or 'stopped'
                os.kill(job.pid, signal.SIGTERM)
                kill_at = kill_at or now + KILL_GRACE_SECONDS
            else:
                log('SIGTERM received: stop-mode none, ignored (use the job stop file)')
        if kill_at is not None and now >= kill_at:
            log('grace expired: SIGKILL job')
            os.kill(job.pid, signal.SIGKILL)
            kill_at = float('inf')
    record = terminal_record(run_id, started_at, returncode, reason, wall_cap_hit)
    common.write_json_once(run_dir / 'terminal.json', record)
    log(f'terminal status={record["status"]} exit_code={record["exit_code"]} signal={record["signal"]}')
    return 0


def main(argv):
    if argv == ['--check-imports']:
        print('imports-ok')
        return 0
    if len(argv) != 1:
        print('usage: supervisor.py <run_dir>', file=sys.stderr)
        return 2
    detach()
    try:
        return supervise(argv[0])
    except Exception as error:  # last resort: leave a reason in supervisor.log
        log(f'supervisor error: {type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
