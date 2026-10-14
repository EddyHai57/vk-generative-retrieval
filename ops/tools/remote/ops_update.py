"""Writer (stdin = JSON {'branch','commit','expect_old','tree':{path:{sha256}}}). New in DEPLOY-01b.
Move the EXISTING CFG ops_repo checkout to a pushed commit:
- refuse unless HEAD == expect_old and porcelain is clean (no local edits on the server)
- git fetch origin <branch>; refuse unless origin/<branch> == commit and commit descends from expect_old
- git checkout --detach commit; verify HEAD, clean porcelain, tracked set == tree, per-file sha256
Other worktrees sharing this clone (G/P deploys) are not touched (their HEADs are reported)."""
import hashlib,json,subprocess,sys
from pathlib import Path
spec=json.loads(sys.stdin.buffer.read())
D=Path(CFG['server']['ops_repo'])
def g(*a,check=True):return subprocess.run(['git','-C',str(D),*a],capture_output=True,text=True,check=check,timeout=300).stdout
head0=g('rev-parse','HEAD').strip()
if head0!=spec['expect_old']:raise SystemExit('refuse: HEAD %s != expect_old'%head0)
if g('status','--porcelain=v1','--untracked-files=all'):raise SystemExit('refuse: ops checkout dirty')
wt_before=g('worktree','list','--porcelain')
b=spec['branch'];g('fetch','-q','origin',f'{b}:refs/remotes/origin/{b}')
remote=g('rev-parse',f'refs/remotes/origin/{b}').strip()
if remote!=spec['commit']:raise SystemExit('refuse: origin/%s=%s != commit'%(b,remote))
if subprocess.run(['git','-C',str(D),'merge-base','--is-ancestor',head0,spec['commit']]).returncode:raise SystemExit('refuse: not a fast-forward of expect_old')
g('checkout','-q','--detach',spec['commit'])
head=g('rev-parse','HEAD').strip();porc=g('status','--porcelain=v1','--untracked-files=all')
tracked=set(g('ls-files','-z').split('\0'))-{''}
bad=[p for p,v in spec['tree'].items() if hashlib.sha256((D/p).read_bytes()).hexdigest()!=v['sha256']]
print(json.dumps({'dir':str(D),'head_before':head0,'head':head,'head_ok':head==spec['commit'],'origin_branch':remote,'porcelain_clean':porc=='',
 'files_expected':len(spec['tree']),'files_tracked':len(tracked),'tracked_set_equal':tracked==set(spec['tree']),'sha_mismatch':bad,
 'worktrees_before':wt_before,'worktrees_after':g('worktree','list','--porcelain')}))
