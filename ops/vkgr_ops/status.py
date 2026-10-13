"""vkgr-status (design §4.5): read-only view of host headroom, runs and the pending queue.

Read-only: no file writes, no locks, no signals, no git. The only subprocess is
nvidia-smi. Missing/corrupt/oversized files give null fields; exit code is 0.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from vkgr_ops import common

SCHEMA = 'vkgr-status-v1'
PROGRESS_SCHEMA = 'vkgr-progress-v1'
PROGRESS_LIMIT = 4096
CPU_SAMPLE_SECONDS = 0.5


def _progress(run_dir):
    """N4: wrong schema, run_id != directory name or >4096 bytes -> corrupt (None)."""
    doc = common.read_json_limited(run_dir / 'progress.json', PROGRESS_LIMIT)
    if doc is None or doc.get('schema') != PROGRESS_SCHEMA or doc.get('run_id') != run_dir.name:
        return None
    return doc


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _host(root, probes, gpu):
    host = dict.fromkeys(('gpu_free_mib', 'gpu_total_mib', 'ram_free_mib', 'ram_total_mib', 'cpu_idle_fraction',
                          'cpu_total', 'cpu_used_cores', 'cpu_nr_throttled', 'cpu_throttled_seconds',
                          'disk_free_gib', 'disk_total_gib'))
    if gpu is not None:
        host['gpu_total_mib'], host['gpu_free_mib'] = gpu[0], gpu[0] - gpu[1]
    try:
        host['ram_total_mib'], host['ram_free_mib'] = common.ram_info(probes)
    except common.ProbeError:
        pass
    try:
        free, total = common.cpu_free(probes)
        host.update(cpu_total=total, cpu_used_cores=total - free,
                    cpu_idle_fraction=free / total if total else None)
    except common.ProbeError:
        pass
    stat = common.cgroup_cpu_stat(probes)
    if stat is not None:
        host['cpu_nr_throttled'] = stat.get('nr_throttled')
        if 'throttled_usec' in stat:
            host['cpu_throttled_seconds'] = stat['throttled_usec'] / 1e6
    try:
        host['disk_total_gib'], host['disk_free_gib'] = common.disk_info(probes, root)
    except common.ProbeError:
        pass
    return host


def _cpu_percent(probes):
    """Two process-table samples CPU_SAMPLE_SECONDS apart; None without /proc."""
    first = common.process_table(probes)
    if first is None:
        return None, None, None
    t0 = probes.clock()
    probes.sleep(CPU_SAMPLE_SECONDS)
    second = common.process_table(probes)
    dt = probes.clock() - t0
    return first, second, dt


def _row(run_dir, probes, now, samples, apps):
    status, info = common.run_status(run_dir, probes, now)
    run, owner, terminal = info['run'] or {}, info['owner'] or {}, info['terminal'] or {}
    progress = _progress(run_dir)
    row = {'run_id': run_dir.name, 'status': status, 'stage': None, 'completed': None, 'total': None,
           'progress_age_seconds': None, 'wall_used_seconds': None,
           'wall_cap_seconds': _num((run.get('resources') or {}).get('wall_seconds')),
           'owner_pid': _num((owner.get('supervisor') or {}).get('pid')), 'cpu_percent': None, 'vram_mib': None}
    if progress is not None:
        row['stage'] = progress.get('stage') if isinstance(progress.get('stage'), str) else None
        row['completed'], row['total'] = _num(progress.get('completed')), _num(progress.get('total'))
        updated = _num(progress.get('updated_at'))
        row['progress_age_seconds'] = None if updated is None else now - updated
    if terminal:
        row['wall_used_seconds'] = _num(terminal.get('wall_seconds'))
    else:
        start = common.started_at(info)
        row['wall_used_seconds'] = None if start is None or owner == {} else now - start
    if status in ('running', 'orphaned'):
        first, second, dt = samples
        root_pid = common.tracked_root_pid(status, info)
        if first is not None and second is not None and root_pid and dt and dt > 0:
            pids = common.descendants(second, root_pid)
            ticks = sum(second[p]['ticks'] - first[p]['ticks'] for p in pids if p in first)
            row['cpu_percent'] = ticks / os.sysconf('SC_CLK_TCK') / dt * 100
        if second is not None and root_pid:
            row['vram_mib'] = common.vram_of(common.descendants(second, root_pid), apps, second)
    return row


def _list_dirs(path):
    try:
        return sorted(p for p in Path(path).iterdir() if p.is_dir())
    except OSError:
        return []


def _pending(root):
    out = []
    try:
        files = sorted(p for p in (Path(root) / 'queue' / 'pending').iterdir() if p.name.endswith('.json'))
    except OSError:
        return out
    for path in files:
        doc = common.read_json_limited(path, 1024 * 1024)
        run_id = doc.get('run_id') if doc else None
        out.append({'file': path.name, 'run_id': run_id if isinstance(run_id, str) else None})
    return out


def collect(root, probes=None):
    probes = probes or common.Probes()
    root = Path(root)
    gpu = common.gpu_memory(probes)
    apps = common.compute_apps(probes) if gpu is not None else None
    host = _host(root, probes, gpu)
    samples = _cpu_percent(probes)
    now = probes.clock()
    runs = [_row(d, probes, now, samples, apps) for d in _list_dirs(root / 'runs')]
    return {'schema': SCHEMA, 'host': host, 'runs': runs, 'pending': _pending(root)}


def _fmt(value, spec='{:.0f}'):
    return '-' if value is None else spec.format(value)


def _duration(seconds):
    if seconds is None:
        return '-'
    seconds = int(seconds)
    return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'


def render_table(doc):
    h = doc['host']
    lines = [
        f'host  GPU {_fmt(h["gpu_free_mib"])}/{_fmt(h["gpu_total_mib"])} MiB free  '
        f'RAM {_fmt(h["ram_free_mib"])}/{_fmt(h["ram_total_mib"])} MiB free  '
        f'CPU quota {_fmt(h["cpu_total"], "{:.2f}")} cores, used {_fmt(h["cpu_used_cores"], "{:.2f}")}, '
        f'idle {_fmt(h["cpu_idle_fraction"], "{:.0%}")}, throttled {_fmt(h["cpu_nr_throttled"])} '
        f'({_fmt(h["cpu_throttled_seconds"])}s)  '
        f'disk {_fmt(h["disk_free_gib"], "{:.1f}")}/{_fmt(h["disk_total_gib"], "{:.1f}")} GiB free',
        f'{"RUN_ID":<24} {"STATUS":<18} {"STAGE":<14} {"PROGRESS":<12} {"WALL(used/cap,left)":<22} '
        f'{"OWNER":<8} {"CPU%":>6} {"VRAM_MiB":>9}',
    ]
    for r in doc['runs']:
        progress = '-' if r['completed'] is None else f'{r["completed"]}/{_fmt(r["total"])}'
        left = None if r['wall_used_seconds'] is None or r['wall_cap_seconds'] is None \
            else r['wall_cap_seconds'] - r['wall_used_seconds']
        wall = f'{_duration(r["wall_used_seconds"])}/{_duration(r["wall_cap_seconds"])},{_duration(left)}'
        lines.append(f'{r["run_id"]:<24} {r["status"]:<18} {r["stage"] or "-":<14} {progress:<12} {wall:<22} '
                     f'{_fmt(r["owner_pid"]):<8} {_fmt(r["cpu_percent"]):>6} {_fmt(r["vram_mib"]):>9}')
    lines.append(f'pending ({len(doc["pending"])}):')
    lines.extend(f'  {p["file"]}  {p["run_id"] or "-"}' for p in doc['pending'])
    return '\n'.join(lines)


def main(argv=None, probes=None):
    parser = argparse.ArgumentParser(prog='vkgr-status', description='Read-only status of the ops root.')
    parser.add_argument('--root', default=None, help='ops root (or VKGR_OPS_ROOT)')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    root = args.root or os.environ.get('VKGR_OPS_ROOT')
    if not root:
        print('vkgr-status: ops root required: pass --root or set VKGR_OPS_ROOT', file=sys.stderr)
        return 1
    doc = collect(root, probes)
    print(json.dumps(doc, indent=2) if args.json else render_table(doc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
