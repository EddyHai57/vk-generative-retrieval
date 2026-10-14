"""Writer (stdin = JSON {'commit','tree'}): git worktree add --detach NEW CFG p_deploy <commit>
from the CFG ops_repo clone (refuse if exists); verify HEAD, clean porcelain, per-file sha256. Writes only that dir.
(DEPLOY-01 deploy01_p_deploy_remote.py)"""
import hashlib,json,subprocess,sys
from pathlib import Path
spec=json.loads(sys.stdin.buffer.read())
R=Path(CFG['server']['ops_repo']);D=Path(CFG['server']['p_deploy'])
if D.exists():raise SystemExit('refuse: target exists')
def git(*a,cwd=R):return subprocess.run(['git','-C',str(cwd),*a],capture_output=True,text=True,check=True,timeout=300).stdout
git('worktree','add','--detach',str(D),spec['commit'])
head=git('rev-parse','HEAD',cwd=D).strip();porc=git('status','--porcelain=v1','--untracked-files=all',cwd=D)
tracked=set(git('ls-files','-z',cwd=D).split('\0'))-{''}
bad=[p for p,v in spec['tree'].items() if hashlib.sha256((D/p).read_bytes()).hexdigest()!=v['sha256']]
print(json.dumps({'dir':str(D),'head':head,'head_ok':head==spec['commit'],'porcelain_clean':porc=='',
 'files_expected':len(spec['tree']),'files_tracked':len(tracked),'tracked_set_equal':tracked==set(spec['tree']),'sha_mismatch':bad}))
