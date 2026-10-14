"""Readonly: scalar fields of terminal.json (+ worker-receipt wait summary) for each legacy G seed dir in
CFG server.legacy.g_seed_dirs. No writes. (DEPLOY-01 deploy01_seed_terminal_read.py)"""
import json
from pathlib import Path
dirs=CFG['server'].get('legacy',{}).get('g_seed_dirs')
if not dirs:raise SystemExit('config server.legacy.g_seed_dirs required')
out={}
for s,root in sorted(dirs.items()):
    R=Path(root);r={}
    t=R/'terminal.json'
    if t.exists():
        b=t.read_bytes().rstrip(b'\0 \n')
        try:v=json.loads(b);r['terminal']={k:v[k] for k in v if not isinstance(v[k],(dict,list))};r['terminal_keys']=sorted(v)
        except Exception as e:r['terminal_raw_head']=b[:1500].decode(errors='replace');r['err']=repr(e)
        r['terminal_mtime']=t.stat().st_mtime
    w=R/'worker-receipt.json'
    if w.exists():
        try:
            v=json.loads(w.read_bytes().rstrip(b'\0 \n'));r['worker']=[{k:x.get(k) for k in ('stage','actual_worker_exit','stop_reason','elapsed_seconds')} for x in v.get('waits',[])];r['worker_top']={k:v[k] for k in v if not isinstance(v[k],(dict,list))}
        except Exception as e:r['worker_err']=repr(e)
    out[s]=r
print(json.dumps(out,indent=1))
