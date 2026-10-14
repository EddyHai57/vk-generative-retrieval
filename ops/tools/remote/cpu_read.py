"""Readonly: CPU budget input for choosing VKGR_EVAL_WORKERS. Prints cgroup cpu.max, cgroup usage over 5s, and every
occupying run under CFG ops_root (no terminal.json) with declared resources and age (run.json mtime). Process lines are
limited to vkgr processes with secrets redacted. No writes. (DEPLOY-01b deploy01b_cpu_read.py)"""
import json,re,subprocess,time
from pathlib import Path
SECRET=re.compile(r'((?:token|apikey|api-key|api_key|password|secret)[=:])[^\s&]+',re.I)
R=Path(CFG['server']['ops_root'])/'runs';now=time.time();out={'running':{}}
for d in sorted(R.iterdir()) if R.exists() else []:
    if (d/'terminal.json').exists() or not (d/'run.json').exists():continue
    v=json.loads((d/'run.json').read_text())
    out['running'][d.name]={'kind':v.get('kind'),'resources':v.get('resources'),'age_seconds':now-(d/'run.json').stat().st_mtime,
        'env_workers':(v.get('env') or {}).get('VKGR_EVAL_WORKERS')}
cg=Path('/sys/fs/cgroup');out['cpu.max']=(cg/'cpu.max').read_text().strip()
q,p=out['cpu.max'].split();out['quota_cores']=None if q=='max' else int(q)/int(p)
def usage():return int([l for l in (cg/'cpu.stat').read_text().splitlines() if l.startswith('usage_usec')][0].split()[1])
u0=usage();t0=time.monotonic();time.sleep(5);u1=usage();out['used_cores_5s']=(u1-u0)/((time.monotonic()-t0)*1e6)
ps=subprocess.run(['ps','-eo','pid,ppid,pcpu,rss,etimes,args','--sort=-pcpu'],capture_output=True,text=True).stdout.splitlines()
out['vkgr_procs']=[SECRET.sub(r'\1<REDACTED>',l)[:220] for l in ps[1:] if 'vkgr' in l][:20]
print(json.dumps(out,indent=1))
