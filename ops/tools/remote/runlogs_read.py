"""Readonly: for every run under CFG ops_root/runs print tails of logs and terminal.json; for prepare runs also the
prepared manifest sha256 + consumer git/env + source member count. No writes. (DEPLOY-01 deploy01_runlogs_read.py)"""
import hashlib,json
from pathlib import Path
R=Path(CFG['server']['ops_root'])/'runs';out={}
for d in sorted(R.iterdir()):
    r={n:(d/'logs'/n).read_bytes()[-2500:].decode(errors='replace') for n in ('supervisor.log','stdout.log','stderr.log') if (d/'logs'/n).exists()}
    r['terminal']=json.loads((d/'terminal.json').read_text()) if (d/'terminal.json').exists() else None
    m=d/'prepared/manifest.json'
    if m.exists():
        v=json.loads(m.read_text());c=v['binding']['consumer']
        r['prepared']={'path':str(m),'sha256':hashlib.sha256(m.read_bytes()).hexdigest(),'git_base':c.get('git_base'),'git_dirty':c.get('git_dirty'),'python':c.get('python'),'n_source_members':len(c.get('source_members',{})),'source_sha256':c.get('source_sha256')}
    out[d.name]=r
print(json.dumps(out,indent=1))
