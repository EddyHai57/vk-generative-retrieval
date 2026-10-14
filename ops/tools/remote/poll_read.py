"""Readonly poller: for each dir under CFG work_dir, print exit.json and tails of logs. No writes. (DEPLOY-01 deploy01_poll_read.py)"""
import json
from pathlib import Path
W=Path(CFG['server']['work_dir']);out={}
for d in sorted(W.iterdir()) if W.exists() else []:
    r={}
    p=d/'exit.json';r['exit.json']=json.loads(p.read_text()) if p.exists() else None
    for name in ('stdout.log','stderr.log','wrapper.log'):
        p=d/name
        if p.exists():
            b=p.read_bytes();r[name]={'bytes':len(b),'tail':b[-1500:].decode(errors='replace')}
    out[d.name]=r
print(json.dumps(out,indent=1))
