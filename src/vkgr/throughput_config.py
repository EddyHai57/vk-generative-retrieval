"""Engineering controls only; scientific identities and frozen source stay intact."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import os
import subprocess


@dataclass(frozen=True)
class Engineering:
    sequence_enabled: bool = False
    use_mmap: bool = False
    assembly_workers: int = 1
    prefetch: int = 1
    bhkm_external: bool = False
    bhkm_workers: int = 1
    bhkm_rows: int = 8192
    temp_dir: str | None = None
    week_workers: int = 1
    eval_workers: int = 1
    decode_workers: int = 1
    memory_limit_bytes: int = 0
    disk_limit_bytes: int = 0
    max_in_flight_bytes: int = 0
    planning_memory_bytes: int = 0

    def __post_init__(self):
        for k,v in asdict(self).items():
            if k in ('sequence_enabled','use_mmap','bhkm_external'):
                if type(v)is not bool:raise ValueError(k+' must be bool')
            elif k=='temp_dir':
                if v is not None and (not isinstance(v,str)or not Path(v).is_absolute()or'..'in Path(v).parts):raise ValueError('absolute temp_dir required')
            elif type(v)is not int or v < (0 if k.endswith('_bytes')else 1):raise ValueError(k+' invalid integer')

    @classmethod
    def load(cls,path):
        import yaml
        raw=Path(path).read_bytes();value=yaml.safe_load(raw.decode())
        if type(value)is not dict:raise ValueError('mapping config required')
        result=cls(**value)
        object.__setattr__(result,'_source_sha256',hashlib.sha256(raw).hexdigest())
        return result

    def identity(self):
        raw=json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()
        return hashlib.sha256(raw).hexdigest()


def manifest(science,config):
    return {'science':dict(science),'engineering':{'configuration':asdict(config),'sha256':config.identity(),
            'source_file_sha256':getattr(config,'_source_sha256',None)}}


def rss_bytes(*,include_children=False):
    if include_children:
        records={}
        if Path('/proc/self/stat').exists():
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    tail=(p/'stat').read_text().rsplit(')',1)[1].split()
                    records[int(p.name)]=(int(tail[1]),int(tail[21])*os.sysconf('SC_PAGE_SIZE'))
                except (OSError,ValueError,IndexError):continue
        else:
            result=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True)
            for line in result.stdout.splitlines():
                pid,ppid,rss=map(int,line.split());records[pid]=(ppid,rss*1024)
        selected={os.getpid()}
        while True:
            new={pid for pid,(parent,_)in records.items()if parent in selected}
            if new<=selected:break
            selected|=new
        if os.getpid()in records:return sum(records[p][1]for p in selected if p in records)
    p=Path('/proc/self/statm')
    if p.exists():
        return int(p.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def guard(config, directory, *, projected_memory=0, projected_disk=0):
    """Configured caps only tighten the registered 60% RSS / 80% disk lines."""
    mem=Path('/proc/meminfo')
    if mem.exists():
        total=int(next(l.split()[1]for l in mem.read_text().splitlines()if l.startswith('MemTotal:')))*1024
        from .throughput_cgroups import effective_limits
        cap=effective_limits()['memory_bytes']
        if cap is not None:total=min(total,cap)
        if config.planning_memory_bytes:total=min(total,config.planning_memory_bytes)
        if rss_bytes(include_children=True)+projected_memory>total*.6:raise MemoryError('registered RSS limit exceeded')
    if config.memory_limit_bytes and rss_bytes(include_children=True)+projected_memory>config.memory_limit_bytes:raise MemoryError('engineering memory cap exceeded')
    disk=shutil.disk_usage(directory)
    if disk.used+projected_disk>disk.total*.8:raise OSError('registered disk headroom exceeded')
    if config.disk_limit_bytes and disk.used+projected_disk>config.disk_limit_bytes:raise OSError('engineering disk cap exceeded')
