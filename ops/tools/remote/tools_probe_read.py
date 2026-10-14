"""Readonly: sha256 of files in CFG tools_dir, ops checkout HEAD/porcelain/worktrees/remotes, run status under
CFG ops_root. No writes. (DEPLOY-01b deploy01b_probe_read.py)"""
import hashlib,json,subprocess
from pathlib import Path
S=CFG['server'];T=Path(S['tools_dir']);O=S['ops_repo']
g=lambda *a:subprocess.run(['git','-C',O,*a],capture_output=True,text=True,timeout=60).stdout
out={'tools':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(T.iterdir())} if T.exists() else None,
 'ops_head':g('rev-parse','HEAD').strip(),'ops_porcelain':g('status','--porcelain=v1','--untracked-files=all'),
 'worktrees':g('worktree','list','--porcelain'),'remotes':g('remote','-v'),'ops_tools_exists':Path(O,'ops/tools').exists()}
R=Path(S['ops_root'])/'runs'
out['runs']={d.name:(json.loads((d/'terminal.json').read_text()).get('status') if (d/'terminal.json').exists() else 'no-terminal') for d in sorted(R.iterdir())} if R.exists() else None
print(json.dumps(out,indent=1))
