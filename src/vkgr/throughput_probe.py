"""Read-only LOCAL resource inventory; recommendations never change science."""
from dataclasses import asdict
from pathlib import Path
import os
import shutil
import subprocess
from .throughput_config import Engineering,manifest
from .throughput_cgroups import effective_limits


def recommend(record):
    cores=record.get('cpu_cores')or 1;ram=record.get('memory_planning_bytes')or record.get('memory_total_bytes')or 0
    vram=record.get('gpu_total_mib')or 0;disk=record.get('disk_total_bytes')or 0
    if cores<12 or ram<64*10**9 or vram<23000 or (record.get('disk_free_bytes')or 0)<400*10**9:
        return Engineering()
    n=min(8,max(1,int(cores)//4))if ram>=96*10**9 else 2
    return Engineering(sequence_enabled=True,use_mmap=True,assembly_workers=n,prefetch=2,
        bhkm_external=True,bhkm_workers=n,bhkm_rows=8192 if ram>=96*10**9 else 2048,
        temp_dir='/workspace/throughput-new-temp',week_workers=n,eval_workers=n,
        decode_workers=2 if vram>=45000 else 1,memory_limit_bytes=int(ram*.5),
        disk_limit_bytes=int(disk*.8),max_in_flight_bytes=1024**3 if ram>=96*10**9 else 256*1024**2)


def observe(directory):
    cores=float(os.cpu_count()or 1)
    if hasattr(os,'sched_getaffinity'):cores=min(cores,len(os.sched_getaffinity(0)))
    limits=effective_limits()
    if limits['cpu_cores']is not None:cores=min(cores,limits['cpu_cores'])
    memory=None;p=Path('/proc/meminfo')
    if p.exists():memory=int(next(x.split()[1]for x in p.read_text().splitlines()if x.startswith('MemTotal:')))*1024
    else:
        try:memory=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))
        except (OSError,ValueError,subprocess.SubprocessError):pass
    if limits['memory_bytes']is not None:memory=min(memory or limits['memory_bytes'],limits['memory_bytes'])
    disk=shutil.disk_usage(directory);gpus=[]
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
        for line in result.stdout.splitlines():
            total,free,driver=[x.strip()for x in line.split(',')];gpus.append({'total_mib':int(total),'free_mib':int(free),'driver':driver})
    except (OSError,ValueError,subprocess.SubprocessError):pass
    return {'cpu_cores':cores,'memory_total_bytes':memory,'disk_total_bytes':disk.total,
            'disk_free_bytes':disk.free,'gpu_total_mib':gpus[0]['total_mib']if gpus else None,'gpus':gpus,
            'gpu_status':'observed'if gpus else'UNKNOWN','measured_context_inclusive_peak':'UNKNOWN'}


def resource_manifest(directory,selected=None):
    values=observe(directory)
    if selected is not None and selected.planning_memory_bytes:
        values['memory_planning_bytes']=min(values.get('memory_total_bytes')or selected.planning_memory_bytes,selected.planning_memory_bytes)
    config=recommend(values)
    return {'schema':'throughput-resource-manifest-v1','observed':values,
            'recommendation':manifest({},config)['engineering'],'selected_engineering':manifest({},selected)['engineering']if selected else None,'recommendation_applied':False,
            'science_config_changed':False,'gpu_parallel_requires_new_peak_measurement':True}
