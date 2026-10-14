"""Readonly: state of one legacy G seed dir (CFG server.legacy.g_seed_dirs[CFG server.legacy.state_seed]): depth<=2 files with
size/mtime, tails of small non-checkpoint control files. No writes, no signals. (DEPLOY-01 deploy01_seed2024_state_read.py)"""
import json,time
from pathlib import Path
L=CFG['server'].get('legacy',{});seed=L.get('state_seed')
if not seed or seed not in L.get('g_seed_dirs',{}):raise SystemExit('config server.legacy.state_seed / g_seed_dirs required')
R=Path(L['g_seed_dirs'][seed]);out={'seed':seed,'now':time.time(),'files':[]}
for p in sorted(R.rglob('*')):
    rel=p.relative_to(R)
    if len(rel.parts)>2 or not p.is_file():continue
    st=p.stat();out['files'].append([str(rel),st.st_size,st.st_mtime])
show={}
for rel,size,mt in out['files']:
    name=Path(rel).name
    if size<=20000 and not name.endswith('.pt') and not name.endswith('.py') and ('evaluation-' not in rel) and (name.endswith('.json') or name in ('stderr','stdout') or name.endswith('.log') or name.endswith('.txt')):
        show[rel]=(R/rel).read_bytes()[-3000:].decode(errors='replace')
out['show']=show
print(json.dumps(out,indent=1))
