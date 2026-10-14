"""G permit tool (runs ON SERVER via vkgr-launch, cwd=<G deploy>, PYTHONPATH=<G deploy>/src, OMP_NUM_THREADS=1).
No training, no model, no checkpoint, no test split, no quality metric.
  sign  --prepared P --prepared-sha S --train-output DIR --permit OUT [--config-index 3 --seed 42 --run-id core-03-seed42]
        [--deadline-seconds 108000] [--max-disk-gib 20]
        Build the G3 formal permit (max_epochs 30, microbatch 512, pool max_resumes 0) from the ACTUAL runtime
        (CUDA identity, env, threads), write it exclusively, then dry-validate with the real loaders:
        load_context + generative_real._limits + configure_runtime + check_registration.
  guard --prepared P --prepared-sha S --scratch DIR [--n 30]
        Speed only: time a single generative_real._guard (and its parts) at real scale with a scratch pool.
  --dry-run (either command): print the parsed plan as JSON; imports nothing from vkgr/torch, writes nothing.
Derived from DEPLOY-01 g_permit_tool_v1.py (logic unchanged; registration made a parameter, main(argv) + --dry-run added).
"""
import argparse
import datetime
import json
import sys
import tempfile
import time
from pathlib import Path

FORMAL_MAX_DECODE = 153720  # (2*30+1)*2520, pinned by generative_pilot._spec for formal_pool
FORMAL_STEPS = {'formal': 5100, 'replay': 5100}


def _runtime():
    import torch
    from vkgr import generative_data as gd
    prop = torch.cuda.get_device_properties(0)
    return {'mode': 'real', 'device': 'cuda:0', 'deterministic_algorithms': True, 'allow_tf32': False,
            'environment': gd._environment(), 'cuda_version': torch.version.cuda, 'gpu_name': prop.name,
            'gpu_uuid': str(prop.uuid), 'threads': torch.get_num_threads()}


def _context(root, prepared, sha, tmp):
    from vkgr import generative_real as gr
    t = time.perf_counter()
    c = gr.load_context(root, Path(prepared), sha, Path(tmp))
    return c, time.perf_counter() - t


def pool_spec(binding_sha, deadline_seconds, max_disk_gib, now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    deadline = now + datetime.timedelta(seconds=deadline_seconds)
    return {'task_id': 'G3', 'kind': 'formal_pool', 'steps_by_phase': dict(FORMAL_STEPS),
            'max_steps': sum(FORMAL_STEPS.values()), 'max_decode': FORMAL_MAX_DECODE,
            'max_disk_bytes': max_disk_gib * 1024 ** 3, 'max_resumes': 0, 'deadline_utc': deadline.isoformat(),
            'binding_sha256': binding_sha}


def registration(a):
    return {'run_id': a.run_id, 'config_index': a.config_index, 'seed': a.seed,
            'output': str(Path(a.train_output).absolute()), 'selection': None}


def sign(a):
    from vkgr import generative_real as gr
    import hashlib
    import torch
    from vkgr.data_audit import canonical_bytes
    root = Path.cwd().resolve()
    out = Path(a.permit)
    if out.exists() or not out.parent.is_dir():
        raise SystemExit('permit path must be new; parent must exist')
    with tempfile.TemporaryDirectory() as tmp:
        context, load_s = _context(root, a.prepared, a.prepared_sha, str(Path(tmp).resolve()))
        b = gr.digest(context.binding)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        limits = {'kind': 'formal_limits', 'runtime': _runtime(), 'max_epochs': 30, 'microbatch': 512, 'max_resumes': 1,
                  'prepared': {'manifest_path': str(Path(a.prepared).absolute()), 'manifest_sha256': a.prepared_sha},
                  'pool_spec': pool_spec(b, a.deadline_seconds, a.max_disk_gib),
                  'registration': registration(a)}
        permit = {'task_id': 'G3', 'kind': 'formal_permit', 'status': 'approved', 'context_binding_sha256': b, 'limits': limits}
        with out.open('xb') as f:
            f.write(canonical_bytes(permit))
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
        full = {**limits, 'permit': {'path': str(out.absolute()), 'sha256': sha}}
        checked = gr._limits(context, full)
        gr.configure_runtime(checked['runtime'])
        from vkgr import generative_experiment as ge
        gr.check_registration(ge._config(a.config_index, a.seed), checked['registration'], a.train_output)
        cfg = ge._config(a.config_index, a.seed)
        print(json.dumps({'status': 'permit_signed_and_dry_validated', 'permit_path': str(out.absolute()), 'permit_sha256': sha,
                          'context_binding_sha256': b, 'deadline_utc': limits['pool_spec']['deadline_utc'], 'runtime': limits['runtime'],
                          'config': {k: cfg.get(k) for k in ('learning_rate', 'd_model', 'seed')}, 'registration': limits['registration'],
                          'context_load_seconds': load_s, 'queries': len(context.queries), 'catalog': len(context.catalog)}, sort_keys=True))


def guard(a):
    from vkgr import generative_real as gr, generative_data as gd
    from vkgr.generative_pilot import ExecutionPool, observe_resources
    import torch
    root = Path.cwd().resolve()
    scratch = Path(a.scratch)
    if scratch.exists():
        raise SystemExit('scratch must be new')
    scratch.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory() as tmp:
        context, load_s = _context(root, a.prepared, a.prepared_sha, str(Path(tmp).resolve()))
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        limits = {'runtime': _runtime()}
        pool = ExecutionPool.create(scratch / 'pool', pool_spec(gr.digest(context.binding), 3600, 1))
        out = scratch / 'out'
        out.mkdir()
        parts = {'full_guard': lambda: gr._guard(context, limits, out, pool), 'pool_check': pool.check,
                 'runtime_snapshot': lambda: gr.runtime_snapshot(limits['runtime']), 'environment': gd._environment,
                 'context_check': context.check, 'observe_resources': lambda: observe_resources(out, limits['runtime']['device']),
                 'query_guard': lambda: gr._query_guard(context, limits, out, pool)}
        res = {}
        for name, fn in parts.items():
            fn()
            ts = []
            for _ in range(a.n):
                t = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t)
            ts.sort()
            res[name] = {'n': a.n, 'median_ms': 1e3 * ts[len(ts) // 2], 'min_ms': 1e3 * ts[0], 'max_ms': 1e3 * ts[-1]}
        q = len(context.queries)
        print(json.dumps({'status': 'guard_timing_only', 'queries': q, 'catalog': len(context.catalog), 'context_load_seconds': load_s,
                          'timings': res, 'serial_eval_guard_calls_per_epoch': 2 * q,
                          'serial_guard_seconds_per_epoch_est': 2 * q * res['full_guard']['median_ms'] / 1e3}, sort_keys=True))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='G permit sign / guard timing')
    s = p.add_subparsers(dest='cmd', required=True)
    x = s.add_parser('sign')
    x.add_argument('--prepared', required=True)
    x.add_argument('--prepared-sha', required=True)
    x.add_argument('--train-output', required=True)
    x.add_argument('--permit', required=True)
    x.add_argument('--config-index', type=int, choices=range(4), default=3)
    x.add_argument('--seed', type=int, choices=(42, 2024, 2025), default=42)
    x.add_argument('--run-id', default='core-03-seed42')
    x.add_argument('--deadline-seconds', type=int, default=108000)
    x.add_argument('--max-disk-gib', type=int, default=20)
    y = s.add_parser('guard')
    y.add_argument('--prepared', required=True)
    y.add_argument('--prepared-sha', required=True)
    y.add_argument('--scratch', required=True)
    y.add_argument('--n', type=int, default=30)
    for sp in (x, y):
        sp.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if len(a.prepared_sha) != 64:
        p.error('--prepared-sha must be 64 hex')
    if a.cmd == 'sign' and (a.deadline_seconds < 1 or a.max_disk_gib < 1):
        p.error('--deadline-seconds and --max-disk-gib must be positive')
    return a


def main(argv=None):
    a = parse_args(argv)
    if a.dry_run:
        plan = {'dry_run': True, 'cmd': a.cmd, 'args': {k: v for k, v in vars(a).items() if k != 'dry_run'}}
        if a.cmd == 'sign':
            plan['registration'] = registration(a)
            plan['pool_spec'] = pool_spec('0' * 64, a.deadline_seconds, a.max_disk_gib)
        print(json.dumps(plan, sort_keys=True))
        return 0
    (sign if a.cmd == 'sign' else guard)(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
