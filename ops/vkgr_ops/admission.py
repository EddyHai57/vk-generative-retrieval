"""Admission (design §4.3, H1): shared by launcher and scheduler.

    reserved_X(run) = max(declared_X, observed_X)
    committed_X     = sum reserved_X over occupying runs
    settling_X      = sum declared_X over occupying runs started < SETTLE_SECONDS ago
    admit iff (total - committed - need)/total >= 0.20
          and (free_measured - settling - need)/total >= 0.20

CPU is counted in cores; total = min(cpu_count, cgroup quota/period). A job may
never declare more cores than that total (2026-10-14 orchestrator requirement).
"""
import os
from pathlib import Path

from vkgr_ops import common

RESOURCES = ('vram_mib', 'ram_mib', 'disk_gib', 'cpu_cores')
HEADROOM = 0.20
SETTLE_SECONDS = 600


def observed_usage(pids, table, apps):
    """Observed RAM (VmRSS sum) and VRAM (compute-apps intersection) of ``pids``; None = unknown."""
    ram = None if table is None else sum(table[p]['rss_kib'] for p in pids if p in table) / 1024
    return {'ram_mib': ram, 'vram_mib': common.vram_of(pids, apps, table)}


def _runs(root):
    runs = Path(root) / 'runs'
    try:
        return sorted(p for p in runs.iterdir() if p.is_dir())
    except OSError:
        return []


def occupancy(root, probes, now, table, apps):
    """committed/settling per resource over occupying runs (§4.3, N4/N5)."""
    committed = {r: 0.0 for r in RESOURCES}
    settling = {r: 0.0 for r in RESOURCES}
    counted = []
    for run_dir in _runs(root):
        status, info = common.run_status(run_dir, probes, now)
        if status not in common.OCCUPYING_STATUSES:
            continue
        declared = ((info.get('run') or {}).get('resources') or {})
        declared = {r: declared.get(r) if isinstance(declared.get(r), (int, float)) else 0 for r in RESOURCES}
        root_pid = common.tracked_root_pid(status, info)
        pids = common.descendants(table, root_pid) if (table is not None and root_pid) else set()
        seen = observed_usage(pids, table, apps) if pids else {'ram_mib': None, 'vram_mib': None}
        for r in RESOURCES:
            observed = seen.get(r)
            committed[r] += max(declared[r], observed) if observed is not None else declared[r]
        start = common.started_at(info)
        if start is None or now - start < SETTLE_SECONDS:
            for r in RESOURCES:
                settling[r] += declared[r]
        counted.append({'run_id': run_dir.name, 'status': status})
    return committed, settling, counted


def measure(root, probes):
    """Host totals/free + occupancy. Raises common.ProbeError on any non-GPU probe failure."""
    now = probes.clock()
    gpu = common.gpu_memory(probes)
    apps = common.compute_apps(probes) if gpu is not None else None
    table = common.process_table(probes)
    ram_total, ram_free = common.ram_info(probes)
    disk_total, disk_free = common.disk_info(probes, root)
    cpu_free, cpu_total = common.cpu_free(probes)
    totals = {'ram_mib': ram_total, 'disk_gib': disk_total, 'cpu_cores': cpu_total,
              'vram_mib': gpu[0] if gpu else None}
    free = {'ram_mib': ram_free, 'disk_gib': disk_free, 'cpu_cores': cpu_free,
            'vram_mib': gpu[0] - gpu[1] if gpu else None}
    committed, settling, counted = occupancy(root, probes, probes.clock(), table, apps)
    return {'measured_at': now, 'gpu_known': gpu is not None, 'cpu_quota_cores': common.cpu_quota(probes),
            'totals': totals, 'free': free, 'committed': committed, 'settling': settling, 'runs': counted}


def evaluate(measured, need):
    """(ok, reason, per-resource detail) for one ``need`` dict against a measurement."""
    detail, problems = {}, []
    for r in RESOURCES:
        n = need.get(r, 0) or 0
        total, free = measured['totals'][r], measured['free'][r]
        row = {'need': n, 'total': total, 'free': free,
               'committed': measured['committed'][r], 'settling': measured['settling'][r]}
        if r == 'vram_mib' and not measured['gpu_known']:
            row['ok'] = n == 0
            if n:
                problems.append(f'vram_mib: GPU unknown (nvidia-smi unavailable), only vram_mib=0 admitted, need={n}')
            detail[r] = row
            continue
        if r == 'cpu_cores' and n > total:
            row['ok'] = False
            problems.append(f'cpu_cores: need={n} exceeds cpu quota total={total:.2f} cores')
            detail[r] = row
            continue
        a = (total - row['committed'] - n) / total if total > 0 else -1.0
        b = (free - row['settling'] - n) / total if total > 0 else -1.0
        row.update(reserved_headroom=a, measured_headroom=b, ok=a >= HEADROOM and b >= HEADROOM)
        if not row['ok']:
            problems.append(f'{r}: need={n} total={total:.2f} committed={row["committed"]:.2f} '
                            f'free={free:.2f} settling={row["settling"]:.2f} '
                            f'headroom={min(a, b):.3f} < {HEADROOM}')
        detail[r] = row
    return not problems, '; '.join(problems), detail


def unsatisfiable(snapshot):
    """Why a request can never be admitted, from an ``admit()`` snapshot; '' when it may fit later.

    A need is unsatisfiable when it fails the headroom rule on an idle host,
    i.e. (total - need) / total < HEADROOM. Load, settling runs, an unknown GPU and probe
    failures (empty snapshot) never count: those are only "not now".
    """
    problems = []
    rows = (snapshot or {}).get('resources') or {}
    for r in RESOURCES:
        row = rows.get(r) or {}
        n, total = row.get('need') or 0, row.get('total')
        if n > 0 and total and (total - n) / total < HEADROOM:
            problems.append(f'{r}: need={n} exceeds the admissible cap {total * (1 - HEADROOM):.2f} '
                            f'of total={total:.2f}')
    return '; '.join(problems)


def admit(need, root, probes=None):
    """(ok, snapshot, reason). Probe failure -> (False, {}, 'probe failed: ...')."""
    probes = probes or common.Probes()
    try:
        measured = measure(root, probes)
    except common.ProbeError as error:
        return False, {}, f'probe failed: {error}'
    ok, reason, detail = evaluate(measured, need)
    snapshot = {'measured_at': measured['measured_at'], 'gpu_known': measured['gpu_known'],
                'cpu_quota_cores': measured['cpu_quota_cores'], 'headroom': HEADROOM,
                'settle_seconds': SETTLE_SECONDS, 'resources': detail, 'occupying_runs': measured['runs'],
                'pid': os.getpid()}
    return ok, snapshot, reason
