"""G serial-vs-sharded parity tool: decode-only job with its OWN pool and OWN decode budget (DEPLOY-01b).
Runs ON SERVER via vkgr-launch, cwd=<G deploy>, PYTHONPATH=<G deploy>/src, OMP_NUM_THREADS=1. Design §2.6 / R2.

Run only after the R3 train run has terminal status succeeded. For the SAME best checkpoint of that run it evaluates
twice, in-process, mirroring generative_real.evaluate (same guards, same per-query decode reservation):
  1. serial  (ev.evaluate_queries,          workers=1)
  2. sharded (gs.evaluate_queries_sharded,  workers=W)
and compares (booleans only, no metric values):
  - gs.evaluation_projection(w1) == evaluation_projection(wW)  (byte equality)
  - manifest computation_sha256 equal; aggregate canonical bytes equal; both complete

Why its own pool: the train run's formal pool has max_decode = (2*30+1)*2520 = 153720, i.e. 30 epochs + 30 replay
epochs + ONE best evaluate. Two extra evaluates (2*2520) do not fit if the run uses its full epoch budget, and the
formal cap is pinned by generative_pilot._spec, so it is not raised. This tool creates a NEW ExecutionPool under
--out-dir with caps max_decode = 2*len(queries), max_steps = 1 (no training), its own deadline. generative_pilot._spec
only knows three kinds; 'engineering_pool' is used as the kind label because it is the only kind without pinned caps —
the caps themselves are this tool's and are printed in the receipt. The train run's pool is never opened or written.
The train run's permit deadline still applies (generative_real._limits checks it), so run before it expires.

Also records wall seconds of each evaluate and peak VRAM (nvidia-smi compute-apps) of this process tree.
One attempt, no retry. If the own pool refuses, the tool stops and reports.
usage: python -B g_parity_tool.py --train-run-dir R/runs/<rid> --out-dir <NEW dir> --workers W
       [--deadline-seconds 10800] [--max-disk-gib 2] [--dry-run]
"""
import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SHARD = 64


def last_json(text):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)
    return None


def train_receipt(run):
    """Train receipt from a vkgr-launch run dir. generative_real `train` prints {manifest_path, manifest_sha256,
    pool_head, pool_state} (no 'status' key — DEPLOY-01 v1 tool wrongly required status=complete); the trajectory
    status=='complete' is checked later on the pinned manifest itself by generative_real._read_run."""
    term = json.loads((run / 'terminal.json').read_text())
    if term.get('status') != 'succeeded':
        raise SystemExit('train run not succeeded: ' + str(term.get('status')))
    rec = last_json((run / 'logs/stdout.log').read_text())
    if not rec or not isinstance(rec.get('manifest_path'), str) or len(str(rec.get('manifest_sha256'))) != 64:
        raise SystemExit('train receipt needs manifest_path + manifest_sha256')
    return rec


def own_pool_spec(binding_sha, queries, deadline_seconds, max_disk_gib, now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return {'task_id': 'G3', 'kind': 'engineering_pool', 'steps_by_phase': {'parity': 1}, 'max_steps': 1,
            'max_decode': 2 * queries, 'max_disk_bytes': max_disk_gib * 1024 ** 3, 'max_resumes': 0,
            'deadline_utc': (now + datetime.timedelta(seconds=deadline_seconds)).isoformat(),
            'binding_sha256': binding_sha}


def _tree(root_pid):
    kids = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                st = (p / 'stat').read_text()
                kids.setdefault(int(st[st.rindex(')') + 2:].split()[1]), []).append(int(p.name))
            except OSError:
                pass
    out, stack = {root_pid}, [root_pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


class VramPeak:
    """Samples nvidia-smi once per second for this process tree."""

    def __init__(self):
        self.peak = {'total_mib': 0, 'per_pid_max_mib': 0, 'samples': 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rows = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=5).stdout
                mine = _tree(os.getpid())
                vals = [int(m) for p, m in (l.split(', ') for l in rows.strip().splitlines() if l) if int(p) in mine]
                self.peak['samples'] += 1
                self.peak['total_mib'] = max(self.peak['total_mib'], sum(vals))
                self.peak['per_pid_max_mib'] = max([self.peak['per_pid_max_mib'], *vals])
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(5)


def run_parity(a, rec):
    from vkgr import generative_real as gr, generative_data as gd, generative_training as gt
    from vkgr import generative_evaluation as ev, generative_sharded as gs
    from vkgr.generative_pilot import ExecutionPool
    from vkgr.data_audit import canonical_bytes
    root = Path.cwd().resolve()
    manifest = Path(rec['manifest_path'])
    value = gd._json(gd._checked_file(manifest, rec['manifest_sha256']))
    prepared = value['limits']['prepared']
    out = Path(a.out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        context = gr.load_context(root, Path(prepared['manifest_path']), prepared['manifest_sha256'], Path(tmp).resolve())
        gr.configure_runtime(value['limits']['runtime'])
        value = gr._read_run(context, manifest, rec['manifest_sha256'])
        if value['status'] != 'complete':
            raise SystemExit('selected best evaluation complete run only')
        limits = gr._limits(context, value['limits'])
        cfg = value['binding']['effective_config']
        q = len(context.queries)
        spec = own_pool_spec(gr.digest(context.binding), q, a.deadline_seconds, a.max_disk_gib)
        res = {'train_receipt': {k: rec.get(k) for k in ('manifest_path', 'manifest_sha256', 'pool_head')},
               'own_pool': {'path': str(out / 'pool'), 'max_decode': spec['max_decode'], 'max_steps': spec['max_steps'],
                            'max_disk_bytes': spec['max_disk_bytes'], 'deadline_utc': spec['deadline_utc']},
               'queries': q, 'workers': a.workers, 'best_epoch': value['best']['epoch']}
        best = value['best']
        with ExecutionPool.create(out / 'pool', spec) as pool:
            def one(workers, target):
                model = gt.make_model(context.codec, cfg, device=limits['runtime']['device'])
                opt = gt.make_optimizer(model, cfg)
                cursor = gt.EpochCursor(cfg['seed'], len(context.examples))
                gt.load_checkpoint(Path(best['path']), best['sha256'], model, opt, cursor, expected_binding=value['binding'])
                pool.reserve('parity', 'disk', q * 4096 + 65536 * (1 + math.ceil(q / SHARD)))
                hook_calls = 0

                def guard():
                    nonlocal hook_calls
                    gr._guard(context, limits, out, pool)
                    if hook_calls > 0 and hook_calls % 2 == 1:
                        pool.reserve('parity', 'decode')
                    hook_calls += 1

                def before_decode():
                    gr._query_guard(context, limits, out, pool)
                    pool.reserve('parity', 'decode')
                t0 = time.monotonic()
                with VramPeak() as vram:
                    if workers == 1:
                        r = ev.evaluate_queries(model, context, output=target, checkpoint_sha256=best['sha256'],
                                                shard_size=SHARD, budget_check=guard)
                    else:
                        r = gs.evaluate_queries_sharded(
                            model, context, output=target, checkpoint_sha256=best['sha256'], workers=workers, config=cfg,
                            runtime=limits['runtime'], shard_size=SHARD, guard=lambda: gr._guard(context, limits, out, pool),
                            before_shard=lambda: gr._guard(context, limits, out, pool), before_decode=before_decode,
                            after_decode=lambda: gr._query_guard(context, limits, out, pool))
                wall = time.monotonic() - t0
                pool.charge_wall('parity', wall)
                return r, {'workers': workers, 'wall_seconds': wall, 'vram': vram.peak,
                           'complete': bool(r['manifest']['complete']), 'decode_after': pool.state['decode']}
            r1, res['w1'] = one(1, out / 'w1')
            rN, res['wN'] = one(a.workers, out / f'w{a.workers}')
            m1, mN = r1['manifest'], rN['manifest']
            res['compare'] = {
                'projection_bytes_equal': gs.evaluation_projection(out / 'w1') == gs.evaluation_projection(out / f'w{a.workers}'),
                'computation_sha256_equal': m1['computation_sha256'] == mN['computation_sha256'],
                'aggregate_equal': canonical_bytes(m1['aggregate']) == canonical_bytes(mN['aggregate']),
                'complete_both': bool(m1['complete'] and mN['complete'])}
            res['verdict'] = 'EQUAL' if all(res['compare'].values()) else 'NOT_EQUAL'
            res['own_pool']['head'] = pool.head
            res['own_pool']['decode_used'] = pool.state['decode']
    return res


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='G serial vs sharded parity (decode-only, own pool)')
    p.add_argument('--train-run-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--deadline-seconds', type=int, default=10800)
    p.add_argument('--max-disk-gib', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if not 2 <= a.workers <= 32:
        p.error('--workers must be 2..32 (workers=1 is always the reference)')
    if a.deadline_seconds < 1 or a.max_disk_gib < 1:
        p.error('--deadline-seconds and --max-disk-gib must be positive')
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.dry_run:
        print(json.dumps({'dry_run': True, 'args': {k: v for k, v in vars(a).items() if k != 'dry_run'},
                          'own_pool_spec_for_2520_queries': own_pool_spec('0' * 64, 2520, a.deadline_seconds, a.max_disk_gib),
                          'steps': ['evaluate workers=1', f'evaluate workers={a.workers}', 'compare']}, sort_keys=True))
        return 0
    rec = train_receipt(Path(a.train_run_dir))
    out = Path(a.out_dir)
    if out.exists():
        raise SystemExit('out dir must be new')
    out.mkdir(mode=0o700, parents=False)
    print(json.dumps(run_parity(a, rec), sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
