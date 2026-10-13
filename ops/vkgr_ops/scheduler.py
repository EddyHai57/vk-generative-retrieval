"""vkgr-sched submit|tick|serve (design §4.3): first-fit, at most one launch per tick, no auto-retry."""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from vkgr_ops import admission, common, launcher

QUEUE_SCHEMA = 'vkgr-queue-job-v1'
SERVE_INTERVAL = 30.0


class SchedulerBusy(Exception):
    """Another tick holds scheduler.lock (CLI exit 3)."""


def _queue(root, name):
    path = Path(root) / 'queue' / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def submit(root, spec):
    """Early-reject with steps 1-5 (results discarded), then O_EXCL-write the pending entry."""
    launcher.prepare(spec)
    pending = _queue(root, 'pending')
    path = pending / f'{time.time_ns():020d}-{spec["run_id"]}.json'
    data = (json.dumps({'schema': QUEUE_SCHEMA, **spec}, indent=2, sort_keys=True) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def _spec_of(doc):
    spec = dict(doc)
    spec.pop('schema', None)
    return spec


def _reject(claimed_file, reason):
    rejected = _queue(claimed_file.parents[2], 'rejected')     # <root>/queue/claimed/<f>

    target = rejected / claimed_file.name
    os.rename(claimed_file, target)
    target.with_suffix('.reason').write_text(reason.strip() + '\n')


def tick(root, probes=None):
    """One round. Returns {'launched': run_id|None, 'rejected': [...], 'note': str}."""
    root = Path(root).resolve()
    probes = probes or common.Probes()
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / 'scheduler.lock', 'a+')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SchedulerBusy('scheduler.lock held by another tick') from None
        return _tick_locked(root, probes)
    finally:
        lock.close()


def _tick_locked(root, probes):
    result = {'launched': None, 'rejected': [], 'note': ''}
    pending = _queue(root, 'pending')
    claimed = _queue(root, 'claimed')
    entries = sorted(p for p in pending.glob('*.json'))
    if not entries:
        result['note'] = 'queue empty'
        return result
    try:
        measured = admission.measure(root, probes)          # lock-free pre-judgement, declared need only
    except common.ProbeError as error:
        print(f'vkgr-sched: probe failed: {error}', file=sys.stderr)
        result['note'] = f'probe failed: {error}'
        return result
    chosen = None
    for entry in entries:
        doc = common.read_json_limited(entry, 1024 * 1024)
        if doc is None or doc.get('schema') != QUEUE_SCHEMA:
            continue
        need = {r: doc.get(r, 0) for r in launcher.RESOURCE_ARGS}
        if doc.get('no_admission') or admission.evaluate(measured, need)[0]:
            chosen = (entry, doc)
            break
    if chosen is None:
        result['note'] = 'no pending job fits'
        return result
    entry, doc = chosen
    target = claimed / entry.name
    os.rename(entry, target)                                     # atomic claim
    run_id = doc.get('run_id')
    others = [p for p in claimed.glob('*.json') if p != target and p.name.split('-', 1)[-1] == f'{run_id}.json']
    if (root / 'runs' / str(run_id)).exists() or others:
        _reject(target, f'run {run_id} already exists or is already claimed')
        result['rejected'].append(run_id)
        return result
    try:
        launcher.launch(_spec_of(doc), root, probes)
    except launcher.LaunchError as error:
        if (root / 'runs' / str(run_id)).exists():
            result['note'] = f'launch_failed (left in claimed): {error}'   # run dir exists: no requeue
        else:
            _reject(target, str(error))
            result['rejected'].append(run_id)
        return result
    result['launched'] = run_id
    return result


def serve(root, probes=None, interval=SERVE_INTERVAL):
    stop = Path(root) / 'queue' / 'STOP'
    while not stop.exists():
        try:
            outcome = tick(root, probes)
            if outcome['launched'] or outcome['rejected']:
                print(json.dumps(outcome), flush=True)
        except SchedulerBusy:
            print('vkgr-sched: scheduler.lock busy, skipping round', file=sys.stderr, flush=True)
        (probes.sleep if probes else time.sleep)(interval)
    return 0


def main(argv=None, probes=None):
    parser = argparse.ArgumentParser(prog='vkgr-sched', description='Queue and admit jobs.')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('submit', add_help=False, help='queue a job (same arguments as vkgr-launch)')
    for name in ('tick', 'serve'):
        p = sub.add_parser(name)
        p.add_argument('--root', default=None)
    args, rest = parser.parse_known_args(argv)
    try:
        if args.command == 'submit':
            ns = launcher.build_parser(prog='vkgr-sched submit').parse_args(rest)
            path = submit(launcher.resolve_root(ns.root), launcher.spec_from_args(ns))
            print(path.name)
            return 0
        if rest:
            parser.error(f'unrecognized arguments: {" ".join(rest)}')
        root = launcher.resolve_root(args.root)
        if args.command == 'tick':
            print(json.dumps(tick(root, probes)))
            return 0
        return serve(root, probes)
    except SchedulerBusy as error:
        print(f'vkgr-sched: {error}', file=sys.stderr)
        return 3
    except launcher.LaunchError as error:
        print(f'vkgr-sched: {error}', file=sys.stderr)
        return error.code


if __name__ == '__main__':
    sys.exit(main())
