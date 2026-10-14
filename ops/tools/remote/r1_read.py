"""Readonly: live vkgr/python processes (pid/ppid/sid/cmd, secrets redacted), GPU apps, disk, existing vkgr-* dirs,
cgroup cpu.max. No writes. (DEPLOY-01 deploy01_r1_read.py; redaction added in DEPLOY-01b)"""
import json,os,re,shutil,subprocess
from pathlib import Path
SECRET=re.compile(r'((?:token|apikey|api-key|api_key|password|secret)[=:])[^\s&]+',re.I)
procs=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
        st=(p/'stat').read_text();rest=st[st.rindex(')')+2:].split()
        if not cmd:continue
        procs.append({'pid':int(p.name),'ppid':int(rest[1]),'sid':int(rest[3]),'cmd':SECRET.sub(r'\1<REDACTED>',cmd)[:300]})
    except Exception:pass
interesting=[x for x in procs if 'vkgr' in x['cmd'] or 'python' in x['cmd']]
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout
B=Path(CFG['server']['scope_prefix']).parent
du=shutil.disk_usage(B)
dirs=sorted(str(p) for p in B.glob('vkgr-*'))
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None
print(json.dumps({'interesting':interesting,'gpu_apps':gpu,'disk_free_gb':du.free/1e9,'workspace_dirs':dirs,'git':shutil.which('git'),'cpu_max':cpu,'nproc':os.cpu_count()},indent=1))
