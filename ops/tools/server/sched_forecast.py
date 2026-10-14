"""Read-only scheduling forecast for an ops root (SCHED-01; see ops/SCHEDULING.md).

    python -I <ops>/tools/server/sched_forecast.py --root <ops_root> [--json]
        [--backlog-hours 4] [--gpu-samples 5] [--tick-seconds 30] [--tz-hours 8]

Run directly on the server (not as a vkgr-launch job). Read-only: writes no file, takes no
lock, sends no signal, never touches queue/. Subprocesses: nvidia-smi only.
Output never contains paths, pids, argv, env, IPs or ports.

- capacity: totals (cgroup cpu.max, nvidia-smi, /proc/meminfo, statvfs) and the admissible cap
  total * (1 - HEADROOM), the same 20 % rule as vkgr_ops.admission.
- occupancy: declared sums over occupying runs, observed RAM/VRAM, GPU utilisation samples.
- cost table: per job type (entry module + subcommand, e.g. 'vkgr.retrieval_feat02_run verify'),
  median/max wall_seconds over succeeded runs' terminal.json. Nothing hard-coded.
- forecast: running remaining = cost - elapsed; pending simulated with the scheduler's first-fit
  rule (one launch per tick, declared resources, entries that do not fit stay pending).
- alert: first line BACKLOG_LOW when cost-based backlog job-hours < --backlog-hours.
"""
import argparse
import datetime
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

OPS = Path(__file__).resolve().parents[2]
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from vkgr_ops import common  # noqa: E402
from vkgr_ops.admission import HEADROOM, RESOURCES  # noqa: E402

SCHEMA = 'vkgr-sched-forecast-v1'
QUEUE_SCHEMA = 'vkgr-queue-job-v1'
SERVICE_TYPES = ('vkgr-sched serve',)
WORD_RE = re.compile(r'^[a-z][a-z0-9_-]{0,31}$')
INF = math.inf


# ---------------------------------------------------------------- job type

def _sub(argv, i):
    return argv[i] if i < len(argv) and isinstance(argv[i], str) and WORD_RE.match(argv[i]) else None


def type_key(argv):
    """'module sub' for python -m, else 'script-basename [sub]'; None when unknown. Never a path."""
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        return None
    i = 1 if os.path.basename(argv[0]).startswith('python') else 0
    while i and i < len(argv) and argv[i].startswith('-'):
        if argv[i] == '-m':
            if i + 1 >= len(argv):
                return None
            sub = _sub(argv, i + 2)
            return argv[i + 1] if sub is None else f'{argv[i + 1]} {sub}'
        if argv[i] == '-c':
            return 'python -c'
        i += 2 if argv[i] in ('-W', '-X') else 1
    if i >= len(argv):
        return None
    sub = _sub(argv, i + 1)
    name = os.path.basename(argv[i])
    return name if sub is None else f'{name} {sub}'


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _declared(doc):
    return {r: _num(doc.get(r)) or 0 for r in RESOURCES}


def _run_dirs(root):
    try:
        return sorted(p for p in (Path(root) / 'runs').iterdir() if p.is_dir())
    except OSError:
        return []


# ---------------------------------------------------------------- cost table

def cost_table(root):
    """{type: {n, wall_median_s, wall_max_s, <resource>_max}} over succeeded, non-service runs."""
    walls, peaks = {}, {}
    for run_dir in _run_dirs(root):
        terminal = common.read_json_limited(run_dir / 'terminal.json')
        run = common.read_json_limited(run_dir / 'run.json')
        if not terminal or not run or terminal.get('status') != 'succeeded':
            continue
        wall, key = _num(terminal.get('wall_seconds')), type_key(run.get('argv'))
        if wall is None or key is None or key in SERVICE_TYPES:
            continue
        walls.setdefault(key, []).append(float(wall))
        declared = _declared(run.get('resources') or {})
        row = peaks.setdefault(key, dict.fromkeys(RESOURCES, 0))
        for r in RESOURCES:
            row[r] = max(row[r], declared[r])
    return {key: {'n': len(values), 'wall_median_s': statistics.median(values), 'wall_max_s': max(values),
                  **{f'{r}_max': peaks[key][r] for r in RESOURCES}}
            for key, values in sorted(walls.items())}


# ---------------------------------------------------------------- host

def _safe(fn, *args):
    try:
        return fn(*args)
    except common.ProbeError:
        return None


def _gpu_util(probes, samples, interval=1.0):
    values = []
    for k in range(max(samples, 0)):
        if k:
            probes.sleep(interval)
        text = common._nvidia(probes, ['--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        try:
            rows = common._csv_ints(text, 1) if text is not None else []
        except ValueError:
            rows = []
        if rows:
            values.append(sum(r[0] for r in rows) / len(rows))
    return values


def capacity(root, probes, gpu):
    quota = common.cpu_quota(probes)
    count = probes.cpu_count() or 1
    ram = _safe(common.ram_info, probes)
    disk = _safe(common.disk_info, probes, root)
    total = {'cpu_cores': min(count, quota) if quota else count, 'ram_mib': ram[0] if ram else None,
             'vram_mib': gpu[0] if gpu else None, 'disk_gib': disk[0] if disk else None}
    free = {'ram_mib': ram[1] if ram else None, 'vram_mib': gpu[0] - gpu[1] if gpu else None,
            'disk_gib': disk[1] if disk else None}
    return {'headroom': HEADROOM, 'cpu_quota_cores': quota, 'total': total, 'free_measured': free,
            'admissible': {r: None if v is None else v * (1 - HEADROOM) for r, v in total.items()}}


# ---------------------------------------------------------------- forecast

def _fits(committed, need, totals):
    """The admission reserved rule: (total - committed - need) / total >= HEADROOM for every resource."""
    for r in RESOURCES:
        total = totals[r]
        if total is None:
            continue
        if total <= 0 or (total - committed[r] - need[r]) / total < HEADROOM:
            return False
    return True


def _estimate(key, elapsed, costs, wall_cap):
    if key in SERVICE_TYPES:
        return None, 'service'
    cost = costs.get(key)
    if cost is None:
        return (None, 'unknown') if wall_cap is None else (max(wall_cap - elapsed, 0.0), 'wall_cap')
    if elapsed < cost['wall_median_s']:
        return cost['wall_median_s'] - elapsed, 'median'
    if elapsed < cost['wall_max_s']:
        return cost['wall_max_s'] - elapsed, 'max'
    return 0.0, 'overdue'


def _running(root, probes, now, costs, table, apps):
    rows = []
    for run_dir in _run_dirs(root):
        status, info = common.run_status(run_dir, probes, now)
        if status not in common.OCCUPYING_STATUSES:
            continue
        run = info.get('run') or {}
        res = run.get('resources') or {}
        start = common.started_at(info)
        elapsed = 0.0 if start is None else max(now - start, 0.0)
        key = type_key(run.get('argv'))
        remaining, source = _estimate(key, elapsed, costs, _num(res.get('wall_seconds')))
        root_pid = common.tracked_root_pid(status, info)
        pids = common.descendants(table, root_pid) if table is not None and root_pid else set()
        seen = {'ram_mib': None, 'vram_mib': None}
        if pids:
            seen = {'ram_mib': sum(table[p]['rss_kib'] for p in pids if p in table) / 1024,
                    'vram_mib': common.vram_of(pids, apps, table)}
        declared = _declared(res)
        reserved = {r: max(declared[r], seen.get(r) or 0) for r in RESOURCES}
        rows.append({'run_id': run_dir.name, 'type': key, 'status': status, 'elapsed_s': elapsed,
                     'remaining_s': remaining, 'source': source, 'declared': declared, 'reserved': reserved,
                     'observed_ram_mib': seen['ram_mib'], 'observed_vram_mib': seen['vram_mib']})
    return rows


def _pending(root, costs):
    rows, ignored = [], 0
    try:
        files = sorted(p for p in (Path(root) / 'queue' / 'pending').iterdir() if p.name.endswith('.json'))
    except OSError:
        files = []
    for path in files:
        doc = common.read_json_limited(path, 1024 * 1024)
        if doc is None or doc.get('schema') != QUEUE_SCHEMA:
            ignored += 1
            continue
        key = type_key(doc.get('argv'))
        cap = _num(doc.get('wall_hours'))
        cost = costs.get(key)
        if cost is not None:
            duration, source = cost['wall_median_s'], 'median'
        elif cap is not None:
            duration, source = cap * 3600.0, 'wall_cap'
        else:
            duration, source = None, 'unknown'
        run_id = doc.get('run_id')
        rows.append({'run_id': run_id if isinstance(run_id, str) else None, 'type': key,
                     'need': _declared(doc), 'no_admission': bool(doc.get('no_admission')),
                     'duration_s': duration, 'source': source, 'state': None, 'start_s': None, 'end_s': None})
    return rows, ignored


def simulate(running, pending, totals, tick):
    """First-fit over pending (queue order), at most one start per tick; mutates pending rows."""
    active = [(INF if r['remaining_s'] is None else r['remaining_s'], r['reserved']) for r in running]
    waiting = []
    for p in pending:
        unsat = any(totals[r] is not None and p['need'][r] > 0 and (totals[r] - p['need'][r]) / totals[r] < HEADROOM
                    for r in RESOURCES)
        if unsat and not p['no_admission']:
            p['state'] = 'never_fits'
        elif p['duration_s'] is None:
            p['state'] = 'unknown_cost'
        else:
            waiting.append(p)
    t = 0.0
    while waiting:
        active = [a for a in active if a[0] > t]
        committed = {r: sum(a[1][r] for a in active) for r in RESOURCES}
        pick = next((p for p in waiting if p['no_admission'] or _fits(committed, p['need'], totals)), None)
        if pick is not None:
            pick.update(state='scheduled', start_s=t, end_s=t + pick['duration_s'])
            active.append((pick['end_s'], pick['need']))
            waiting.remove(pick)
            t += tick
            continue
        future = [a[0] for a in active if t < a[0] < INF]
        if not future:
            for p in waiting:
                p['state'] = 'blocked'
            break
        t = math.ceil(min(future) / tick - 1e-9) * tick
    return pending


def _summary(running, pending, cap, backlog_hours):
    measured_s = capped_s = 0.0
    spans = []                                         # (start, end, cpu) of finite work
    services_cpu = 0.0
    for r in running:
        if r['remaining_s'] is None:
            services_cpu += r['reserved']['cpu_cores']
            continue
        if r['source'] == 'wall_cap':
            capped_s += r['remaining_s']
        else:
            measured_s += r['remaining_s']
        spans.append((0.0, r['remaining_s'], r['reserved']['cpu_cores']))
    for p in pending:
        if p['state'] != 'scheduled':
            continue
        if p['source'] == 'wall_cap':
            capped_s += p['duration_s']
        else:
            measured_s += p['duration_s']
        spans.append((p['start_s'], p['end_s'], p['need']['cpu_cores']))
    idle = max((e for _, e, _ in spans), default=0.0)
    half = None if cap['admissible']['cpu_cores'] is None else cap['admissible']['cpu_cores'] * 0.5
    below = None
    if half is not None:
        for t in sorted({0.0, *(s for s, _, _ in spans), *(e for _, e, _ in spans)}):
            used = services_cpu + sum(c for s, e, c in spans if s <= t < e)
            if used < half:
                below = t
                break
    hours = measured_s / 3600.0
    return {'backlog_job_hours': hours, 'backlog_job_hours_wall_cap': capped_s / 3600.0,
            'backlog_threshold_hours': backlog_hours, 'backlog_low': hours < backlog_hours,
            'cpu_half_cores': half, 'cpu_below_half_at_s': below, 'idle_at_s': idle,
            'pending_states': {s: sum(1 for p in pending if p['state'] == s)
                               for s in ('scheduled', 'never_fits', 'blocked', 'unknown_cost')}}


def forecast(root, probes=None, gpu_samples=5, tick_seconds=30.0, backlog_hours=4.0):
    probes = probes or common.Probes()
    root = Path(root)
    now = probes.clock()
    gpu = common.gpu_memory(probes)
    apps = common.compute_apps(probes) if gpu is not None else None
    table = common.process_table(probes)
    cap = capacity(root, probes, gpu)
    costs = cost_table(root)
    running = _running(root, probes, now, costs, table, apps)
    cpu = _safe(common.cpu_free, probes)
    util = _gpu_util(probes, gpu_samples) if gpu is not None else []
    pending, ignored = _pending(root, costs)
    simulate(running, pending, cap['total'], tick_seconds)
    observed = [r['observed_vram_mib'] for r in running]
    occupancy = {
        'runs': len(running),
        'declared': {r: sum(x['declared'][r] for x in running) for r in RESOURCES},
        'observed_ram_mib': sum(x['observed_ram_mib'] or 0 for x in running) if table is not None else None,
        'observed_vram_mib': None if not running or any(v is None for v in observed) else sum(observed),
        'gpu_used_mib': gpu[1] if gpu else None,
        'cpu_used_cores': None if cpu is None else cpu[1] - cpu[0],
        'gpu_util_mean': statistics.mean(util) if util else None,
        'gpu_util_max': max(util) if util else None, 'gpu_util_samples': len(util)}
    return {'schema': SCHEMA,
            'measured_at_utc': datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat(timespec='seconds'),
            'measured_at': now, 'tick_seconds': tick_seconds, 'capacity': cap, 'occupancy': occupancy,
            'cost_table': costs, 'running': running, 'pending': pending, 'pending_ignored': ignored,
            'summary': _summary(running, pending, cap, backlog_hours)}


# ---------------------------------------------------------------- output

def _f(value, spec='{:.0f}'):
    return '-' if value is None else spec.format(value)


def _dur(seconds):
    if seconds is None:
        return '-'
    if seconds == INF:
        return 'inf'
    seconds = int(round(seconds))
    return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'


def _at(doc, seconds, tz_hours):
    if seconds is None:
        return '-'
    tz = datetime.timezone(datetime.timedelta(hours=tz_hours))
    stamp = datetime.datetime.fromtimestamp(doc['measured_at'] + seconds, tz).strftime('%m-%d %H:%M')
    return f'+{_dur(seconds)} ({stamp})'


def render(doc, tz_hours=8.0):
    s, cap, occ = doc['summary'], doc['capacity'], doc['occupancy']
    flag = 'BACKLOG_LOW' if s['backlog_low'] else 'BACKLOG_OK'
    tz = f'UTC{tz_hours:+g}'
    lines = [f'{flag} backlog {s["backlog_job_hours"]:.2f} job-h (cost-based) vs threshold '
             f'{s["backlog_threshold_hours"]:g} h; +{s["backlog_job_hours_wall_cap"]:.2f} job-h estimated from '
             f'wall caps; idle at {_at(doc, s["idle_at_s"], tz_hours)}; CPU < 50% of cap at '
             f'{_at(doc, s["cpu_below_half_at_s"], tz_hours)}  [times {tz}]']
    t, a, fr = cap['total'], cap['admissible'], cap['free_measured']
    lines.append(f'capacity (admissible = total x {1 - cap["headroom"]:.2f})  CPU {_f(a["cpu_cores"], "{:.2f}")}/'
                 f'{_f(t["cpu_cores"], "{:.2f}")} cores  RAM {_f(a["ram_mib"])}/{_f(t["ram_mib"])} MiB '
                 f'(free {_f(fr["ram_mib"])})  VRAM {_f(a["vram_mib"])}/{_f(t["vram_mib"])} MiB '
                 f'(free {_f(fr["vram_mib"])})  disk {_f(a["disk_gib"], "{:.0f}")}/{_f(t["disk_gib"], "{:.0f}")} GiB '
                 f'(free {_f(fr["disk_gib"], "{:.0f}")})')
    d = occ['declared']
    lines.append(f'occupancy {occ["runs"]} runs  declared CPU {d["cpu_cores"]:g} RAM {d["ram_mib"]:g} '
                 f'VRAM {d["vram_mib"]:g} disk {d["disk_gib"]:g}  observed RAM {_f(occ["observed_ram_mib"])} '
                 f'VRAM {_f(occ["observed_vram_mib"])} (GPU used {_f(occ["gpu_used_mib"])})  CPU used '
                 f'{_f(occ["cpu_used_cores"], "{:.2f}")} cores  GPU util mean {_f(occ["gpu_util_mean"])}% '
                 f'max {_f(occ["gpu_util_max"])}% ({occ["gpu_util_samples"]} samples)')
    lines.append(f'cost table (succeeded runs)  {"TYPE":<40} {"N":>3} {"MEDIAN":>7} {"MAX":>7} '
                 f'{"CPU":>4} {"RAM":>7} {"VRAM":>6}')
    for key, c in doc['cost_table'].items():
        lines.append(f'  {key:<66} {c["n"]:>3} {_dur(c["wall_median_s"]):>7} {_dur(c["wall_max_s"]):>7} '
                     f'{c["cpu_cores_max"]:>4g} {c["ram_mib_max"]:>7g} {c["vram_mib_max"]:>6g}')
    lines.append(f'running ({len(doc["running"])})  {"RUN_ID":<34} {"TYPE":<36} {"ELAPSED":>8} {"LEFT":>8} '
                 f'{"FROM":<8} {"VRAM decl/obs":>14}')
    for r in doc['running']:
        lines.append(f'  {r["run_id"]:<44} {r["type"] or "-":<36} {_dur(r["elapsed_s"]):>8} '
                     f'{_dur(r["remaining_s"]):>8} {r["source"]:<8} '
                     f'{r["declared"]["vram_mib"]:>7g}/{_f(r["observed_vram_mib"]):<6}')
    lines.append(f'pending ({len(doc["pending"])}, ignored {doc["pending_ignored"]})  {"RUN_ID":<34} {"TYPE":<36} '
                 f'{"START":>22} {"END":>22} {"FROM":<8} STATE')
    for p in doc['pending']:
        lines.append(f'  {p["run_id"] or "-":<44} {p["type"] or "-":<36} {_at(doc, p["start_s"], tz_hours):>22} '
                     f'{_at(doc, p["end_s"], tz_hours):>22} {p["source"]:<8} {p["state"]}')
    return '\n'.join(lines)


def main(argv=None, probes=None):
    parser = argparse.ArgumentParser(prog='sched_forecast', description='Read-only scheduling forecast.')
    parser.add_argument('--root', default=None, help='ops root (or VKGR_OPS_ROOT)')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--backlog-hours', type=float, default=4.0)
    parser.add_argument('--gpu-samples', type=int, default=5, help='GPU utilisation samples, 1 s apart')
    parser.add_argument('--tick-seconds', type=float, default=30.0, help='scheduler serve interval')
    parser.add_argument('--tz-hours', type=float, default=8.0, help='UTC offset for printed clock times')
    args = parser.parse_args(argv)
    root = args.root or os.environ.get('VKGR_OPS_ROOT')
    if not root:
        print('sched_forecast: ops root required: pass --root or set VKGR_OPS_ROOT', file=sys.stderr)
        return 1
    doc = forecast(root, probes, gpu_samples=args.gpu_samples, tick_seconds=args.tick_seconds,
                   backlog_hours=args.backlog_hours)
    print(json.dumps(doc, indent=2, default=str) if args.json else render(doc, args.tz_hours))
    return 0


if __name__ == '__main__':
    sys.exit(main())
