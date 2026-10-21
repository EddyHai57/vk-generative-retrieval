"""Strict mmap transport for the existing fixed-arithmetic tool 5 kernel."""
import os
from pathlib import Path
import numpy as np
from .throughput_cgroups import effective_limits
from .throughput_config import guard
from .throughput_exact import exact_topk, _descriptor


def cpu_capacity():
    cores=float(os.cpu_count() or 1)
    if hasattr(os,'sched_getaffinity'):cores=min(cores,len(os.sched_getaffinity(0)))
    quota=effective_limits()['cpu_cores']
    return min(cores,quota) if quota is not None else cores


def parallel_topk(queries,items,ids,k,config,*,trace=False):
    workers=config.eval_workers
    # Parent handles result delivery/guards; reserve one core beside the pool.
    if (workers+1 if workers>1 else 1)>cpu_capacity():raise ValueError('CPU quota cannot admit workers plus parent')
    if workers>1:
        if not config.use_mmap:raise ValueError('parallel inputs require mmap transport')
        for value in (queries,items,ids):
            if not isinstance(value,np.memmap):raise ValueError('parallel inputs require full .npy mmap')
            _descriptor(value)
        if queries.dtype!=np.dtype('f4') or items.dtype!=np.dtype('f4'):
            raise ValueError('parallel feature mmap must be float32')
    guard(config,Path.cwd())
    return exact_topk(queries,items,ids,k,config,trace=trace)
