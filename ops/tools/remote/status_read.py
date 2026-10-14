"""Readonly: vkgr-status (table + --json) for CFG ops_root, plus run dir log sizes and owner pid/sid/ppid. No writes.
(DEPLOY-01 deploy01_status_read.py)"""
import json,os,subprocess,sys
from pathlib import Path
S=CFG['server'];OPS=S['ops_repo'];ROOT=Path(S['ops_root']);PY=S['python']
env={k:v for k,v in os.environ.items() if not k.startswith('VKGR_')}
t=subprocess.run([PY,'-I',OPS+'/ops/bin/vkgr-status','--root',str(ROOT)],capture_output=True,text=True,env=env,timeout=60)
j=subprocess.run([PY,'-I',OPS+'/ops/bin/vkgr-status','--root',str(ROOT),'--json'],capture_output=True,text=True,env=env,timeout=60)
runs={}
for d in sorted((ROOT/'runs').iterdir()) if (ROOT/'runs').exists() else []:
    r={p.name:p.stat().st_size for p in (d/'logs').iterdir()} if (d/'logs').exists() else {}
    o=json.loads((d/'owner.json').read_text()) if (d/'owner.json').exists() else None
    ids={}
    if o:
        for role in ('supervisor','job'):
            pid=o[role]['pid']
            try:
                st=Path(f'/proc/{pid}/stat').read_text();rest=st[st.rindex(')')+2:].split();ids[role]={'pid':pid,'state':rest[0],'ppid':int(rest[1]),'pgid':int(rest[2]),'sid':int(rest[3])}
            except FileNotFoundError:ids[role]={'pid':pid,'alive':False}
    term=json.loads((d/'terminal.json').read_text()) if (d/'terminal.json').exists() else None
    prog=json.loads((d/'progress.json').read_text()) if (d/'progress.json').exists() else None
    runs[d.name]={'logs':r,'proc':ids,'terminal':term,'progress':prog}
print(json.dumps({'table':t.stdout,'table_err':t.stderr[-500:],'json':j.stdout,'json_err':j.stderr[-500:],'runs':runs},indent=1))
