"""Own-PID GPU samples and conservative allocator plus context accounting."""
import math
import os
import subprocess
REQUIRED_STAGES={'baseline','train-1','train-2','train-3','catalog-export','user-export','top50'}

def own_gpu_mib():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    values=[float(line.split(',')[1]) for line in result.stdout.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
    if len(values)!=1 or not math.isfinite(values[0]) or values[0]<=0:raise ValueError('unique finite positive own GPU PID sample required')
    return values[0]

def declared_vram_mib(measured):
    if type(measured) not in (int,float) or not math.isfinite(measured) or measured<=0:raise ValueError('finite positive measured VRAM required')
    return max(1024,math.ceil(measured*1.15/256)*256)

def measured_mib(samples):
    if not isinstance(samples,list) or not samples or any(type(s) is not dict for s in samples) or not REQUIRED_STAGES<={s.get('stage') for s in samples}:raise ValueError('baseline/train/export own-PID samples required')
    pids={s.get('pid') for s in samples}
    if len(pids)!=1 or type(next(iter(pids))) is not int or next(iter(pids))<=0:raise ValueError('one positive measurement PID required')
    for s in samples:
        for key in ('own_process_mib','current_reserved_mib','max_allocated_mib','max_reserved_mib'):
            value=s.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (key=='own_process_mib' and value==0):raise ValueError('invalid sample: '+key)
        if s['max_reserved_mib']<s['max_allocated_mib'] or s['max_reserved_mib']<s['current_reserved_mib']:raise ValueError('allocator sample contradiction')
    nonallocator=max(max(0,s['own_process_mib']-s['current_reserved_mib']) for s in samples)
    return max(max(s['own_process_mib'] for s in samples),max(s['max_reserved_mib'] for s in samples)+nonallocator)

def validate_measurement(row):
    proof=row.get('vram')
    if not isinstance(proof,dict) or proof.get('method')!='own_pid_nvidia_smi+torch_peaks':raise ValueError('own-PID GPU method required')
    measured=measured_mib(proof.get('samples'))
    if row.get('measured_mib')!=measured or row.get('declared_vram_mib')!=declared_vram_mib(measured):raise ValueError('measured VRAM or declaration formula mismatch')
    return True
