"""Runtime CPU/cgroup limits, precision contexts and process measurements."""

from contextlib import contextmanager
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import torch


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ''


def _cpus(text):
    result = set()
    for part in text.split(','):
        if not part.strip():
            continue
        bounds = [int(value) for value in part.split('-')]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def _physical_memory():
    if sys.platform == 'darwin':
        try:
            return int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'],
                                              stderr=subprocess.DEVNULL))
        except (OSError, subprocess.SubprocessError):
            return None
    for line in _read('/proc/meminfo').splitlines():
        if line.startswith('MemTotal:'):
            return int(line.split()[1]) * 1024
    return None


def available_resources(*, cgroup_root=Path('/sys/fs/cgroup'), cgroup_path=None,
                        affinity=None, host_cpus=None):
    root = Path(cgroup_root)
    if cgroup_path is None:
        relative = next((line.split(':', 2)[2] for line in
                         _read('/proc/self/cgroup').splitlines()
                         if line.startswith('0::')), '/')
        cgroup_path = root / relative.lstrip('/')
        if not Path(cgroup_path).is_dir():
            cgroup_path = root
    path = Path(cgroup_path)
    if affinity is None:
        affinity = (os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity')
                    else set(range(os.cpu_count() or 1)))
    allowed = set(affinity)
    host_cpus = host_cpus or os.cpu_count() or 1
    quotas = []
    memory_limits = []
    while path == root or root in path.parents:
        quota = _read(path / 'cpu.max').split()
        if len(quota) == 2 and quota[0] != 'max':
            quotas.append(int(quota[0]) / int(quota[1]))
        cpuset = _cpus(_read(path / 'cpuset.cpus.effective'))
        if cpuset:
            allowed &= cpuset
        memory = _read(path / 'memory.max')
        if memory and memory != 'max':
            memory_limits.append(int(memory))
        if path == root:
            break
        path = path.parent
    physical = _physical_memory()
    if physical is not None:
        memory_limits.append(physical)
    quota_cores = min(quotas) if quotas else None
    threads = min(len(allowed), host_cpus)
    if quota_cores is not None:
        threads = min(threads, math.floor(quota_cores))
    return dict(cpu_threads=max(1, threads), cpu_host_logical=host_cpus,
                cpu_quota_cores=quota_cores, cpu_affinity_count=len(affinity),
                cpu_allowed_ids=sorted(allowed),
                memory_limit_bytes=min(memory_limits) if memory_limits else None,
                cgroup_path=str(cgroup_path))


_blas_controller = None


def configure_threads(requested=None):
    global _blas_controller
    limits = available_resources()
    threads = limits['cpu_threads']
    if requested is not None:
        if requested < 1:
            raise ValueError('thread count must be positive')
        threads = min(threads, requested)
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass  # A long-lived caller can have initialized this pool already.
    from threadpoolctl import threadpool_limits
    _blas_controller = threadpool_limits(limits=threads)
    import faiss
    faiss.omp_set_num_threads(threads)
    limits['configured_threads'] = threads
    limits['torch_threads'] = torch.get_num_threads()
    limits['torch_interop_threads'] = torch.get_num_interop_threads()
    limits['faiss_threads'] = faiss.omp_get_max_threads()
    return limits


def duckdb_connection(temp_directory=None, memory_limit='8GB'):
    import duckdb

    connection = duckdb.connect()
    connection.execute(f"SET threads={min(torch.get_num_threads(), available_resources()['cpu_threads'])}")
    connection.execute('SET memory_limit=?', [memory_limit])
    if temp_directory is not None:
        connection.execute('SET temp_directory=?', [str(temp_directory)])
    return connection


@contextmanager
def precision_mode(device, precision='fp32'):
    device = torch.device(device)
    if precision not in ('fp32', 'tf32', 'bf16'):
        raise ValueError('unknown precision')
    if precision != 'fp32' and device.type != 'cuda':
        raise ValueError('TF32 and bf16 measurement require CUDA')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('bf16 is unavailable on this CUDA device')
    if precision == 'tf32' and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError('TF32 is unavailable on this CUDA device')
    previous = (torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = precision == 'tf32'
        torch.backends.cudnn.allow_tf32 = precision == 'tf32'
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == 'bf16'):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous[0]
        torch.backends.cudnn.allow_tf32 = previous[1]


def process_metrics(device='cpu'):
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = maximum if sys.platform == 'darwin' else maximum * 1024
    result = dict(rss_peak_bytes=rss, process_cpu_seconds=time.process_time(),
                  torch_peak_allocated_bytes=None, torch_peak_reserved_bytes=None,
                  gpu_context_inclusive_peak_bytes=None, shm_bytes=None)
    if Path('/dev/shm').exists():
        stat = os.statvfs('/dev/shm')
        result['shm_bytes'] = stat.f_blocks * stat.f_frsize
    if torch.device(device).type == 'cuda' and torch.cuda.is_available():
        result['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        result['torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                 '--format=csv,noheader,nounits'], text=True, timeout=5
            )
            used = [int(line.split(',')[1]) * 1024**2
                    for line in output.splitlines()
                    if int(line.split(',')[0]) == os.getpid()]
            result['gpu_process_snapshot_bytes'] = max(used) if used else None
            result['gpu_peak_scope'] = 'torch_counter_excludes_context'
            result['context_measurement'] = 'nvidia-smi process snapshot'
        except (OSError, ValueError, subprocess.SubprocessError):
            result['context_measurement'] = 'UNKNOWN; torch counters exclude context'
    return result
