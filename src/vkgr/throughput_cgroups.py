"""Effective limits of this process: root plus nested cgroup ancestors, v1/v2."""
from pathlib import Path


def effective_limits(root='/sys/fs/cgroup',membership='/proc/self/cgroup'):
    root=Path(root);memory=[];cpu=[];paths={root};v1_memory={root/'memory'};v1_cpu={root/'cpu',root/'cpu,cpuacct'}
    try:lines=Path(membership).read_text().splitlines()
    except OSError:lines=[]
    for line in lines:
        fields=line.split(':',2)
        if len(fields)!=3:continue
        _,controllers,relative=fields;parts=Path(relative.lstrip('/')).parts
        if '..'in parts:continue
        for length in range(len(parts)+1):
            rel=Path(*parts[:length])
            if controllers=='':paths.add(root/rel)
            if 'memory'in controllers.split(','):v1_memory.add(root/'memory'/rel)
            if 'cpu'in controllers.split(','):
                v1_cpu.add(root/'cpu'/rel);v1_cpu.add(root/'cpu,cpuacct'/rel)
    def integer(path):
        try:return int(path.read_text().strip())
        except (OSError,ValueError):return None
    for p in paths:
        limit=integer(p/'memory.max')
        if limit is not None and 0<limit<2**60:memory.append(limit)
        try:
            q,period=(p/'cpu.max').read_text().split()
            if q!='max'and int(q)>0 and int(period)>0:cpu.append(int(q)/int(period))
        except (OSError,ValueError):pass
    for p in v1_memory:
        limit=integer(p/'memory.limit_in_bytes')
        if limit is not None and 0<limit<2**60:memory.append(limit)
    for p in v1_cpu:
        q,period=integer(p/'cpu.cfs_quota_us'),integer(p/'cpu.cfs_period_us')
        if q is not None and period is not None and q>0 and period>0:cpu.append(q/period)
    return {'memory_bytes':min(memory)if memory else None,'cpu_cores':min(cpu)if cpu else None}
