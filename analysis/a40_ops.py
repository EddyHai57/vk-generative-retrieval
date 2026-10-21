"""New-instance OPS adapter: tighten RAM planning without changing frozen OPS.

All launch/submit/serve/status calls use the same capped probe. This adapter
does not grant scientific permission and does not change data or model code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

OPS_SHA = '5840e6b686ec9651314a9e28a1a2c0befcc05158'
PLAN_BYTES = 513000000000
PLAN_MIB = PLAN_BYTES // (1024 * 1024)
DATA_STAGES = ('download-a', 'source-preflight', 'prepare-users', 'download-week26', 'filter-preflight', 'filter')


def validate_spec(spec):
    """Keep this task adapter restricted to data rebuild and its scheduler."""
    if spec['kind'] != 'ops' or spec.get('no_admission'):
        raise ValueError('only admitted ops jobs are authorized')
    argv = spec['argv']
    if len(argv) < 5 or argv[:3] != [sys.executable, '-B', '-m']:
        raise ValueError('fixed Python module invocation required')
    if argv[3] == 'analysis.new_instance_prepare':
        stage = argv[4]
        prefix = 'a40-rebuild-'+stage+'-v'
        if stage not in DATA_STAGES or not spec['run_id'].startswith(prefix) or not spec['run_id'][len(prefix):].isdigit():
            raise ValueError('data stage/run identity outside task')
        tail = argv[5:]
        flags = ('--root', '--reference-root') if stage == 'download-a' else ('--request',)
        if len(tail) != len(flags)*2+1 or tail[-1] != '--execute-on-new-instance':
            raise ValueError('fixed data argv required')
        for i, flag in enumerate(flags):
            path = Path(tail[i*2+1])
            if tail[i*2] != flag or not path.is_absolute() or '..' in path.parts:
                raise ValueError('fixed flag and absolute task path required')
        return
    if (argv[3:5] == ['analysis.a40_ops', 'serve'] and
            spec['run_id'] == 'a40-rebuild-scheduler-v1' and
            len(argv) == 9 and argv[5] == '--ops-code' and argv[7] == '--root'):
        return
    raise ValueError('worker outside authorized data rebuild')


def cap_ram(total, free):
    effective = min(total, PLAN_MIB)
    return effective, max(0, effective - (total - free))


def install_cap(common):
    original = common.ram_info
    common.ram_info = lambda probes: cap_ram(*original(probes))


def load_ops(path):
    root = Path(path).resolve(strict=True)
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != OPS_SHA:
        raise ValueError('OPS fixed SHA differs')
    # Check every tracked OPS file; no open-ended source exception.
    names = subprocess.check_output(['git', '-C', str(root), 'ls-tree', '-r', '--name-only', 'HEAD', 'ops'], text=True).splitlines()
    for name in names:
        target = root / name
        expected = subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:'+name])
        if target.is_symlink() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(expected).digest():
            raise ValueError('OPS source bytes differ: '+name)
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all', '--', 'ops'], text=True).strip():
        raise ValueError('OPS tree dirty')
    sys.path.insert(0, str(root / 'ops'))
    from vkgr_ops import common, launcher, scheduler, status
    install_cap(common)
    return common, launcher, scheduler, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['dry', 'status', 'launch', 'submit', 'serve', 'tick', 'prepare', 'cuda-matmul'])
    parser.add_argument('--ops-code')
    args, rest = parser.parse_known_args(argv)
    if args.command == 'dry':
        if rest: parser.error('unexpected dry arguments')
        print(json.dumps({'planning_ram_bytes': PLAN_BYTES, 'planning_ram_mib': PLAN_MIB, 'ops_sha': OPS_SHA}))
        return 0
    if args.command == 'cuda-matmul':
        import os
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('deterministic cuBLAS environment required')
        import torch
        torch.use_deterministic_algorithms(True)
        if not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
        a = torch.arange(16, dtype=torch.float64, device='cuda').reshape(4, 4)
        actual = (a @ a.T).cpu()
        if not torch.equal(actual, a.cpu() @ a.cpu().T): raise ValueError('matrix product differs')
        print(json.dumps({'status': 'cuda_matmul_verified', 'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']}))
        return 0
    if not args.ops_code: parser.error('--ops-code required')
    common, launcher, scheduler, status = load_ops(args.ops_code)
    if args.command == 'status': return status.main(rest)
    if args.command in ('serve', 'tick'): return scheduler.main([args.command]+rest)
    ns = launcher.build_parser().parse_args(rest)
    spec = launcher.spec_from_args(ns)
    validate_spec(spec)
    if Path(spec['cwd']).resolve() != Path(__file__).resolve().parents[1]:
        raise ValueError('worker cwd must equal the pinned adapter deployment')
    if args.command == 'launch': return launcher.main(rest)
    if args.command == 'submit': return scheduler.main(['submit']+rest)
    result = launcher.prepare(spec)
    print(json.dumps({'status': 'prepared_dry', 'ops_sha': OPS_SHA, 'planning_ram_bytes': PLAN_BYTES,
                      'spec': spec, 'metadata': result}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
