"""Readonly: list small json control files of legacy G dirs CFG server.legacy.g_inventory_dirs (depth<=4) and print
permit/prepared/spec/registration JSON (<=64KiB; control metadata only, no data/test files). No writes.
(DEPLOY-01 deploy01_oldg_inventory_read.py)"""
import json
from pathlib import Path
dirs=CFG['server'].get('legacy',{}).get('g_inventory_dirs')
if not dirs:raise SystemExit('config server.legacy.g_inventory_dirs required')
out={}
for root in dirs:
    R=Path(root);files=[]
    for p in R.rglob('*'):
        rel=p.relative_to(R)
        if len(rel.parts)>4 or not p.is_file() or p.is_symlink():continue
        files.append([str(rel),p.stat().st_size])
    out[root]={'n':len(files),'files':sorted(files)[:400]}
    for rel,size in files:
        if size<=65536 and rel.endswith('.json') and any(k in rel for k in ('permit','prepared/manifest','spec','registration','argv','invoke','launch')):
            try:out[root+'::'+rel]=json.loads((R/rel).read_text())
            except Exception as e:out[root+'::'+rel]=repr(e)
print(json.dumps(out,indent=1)[:200000])
