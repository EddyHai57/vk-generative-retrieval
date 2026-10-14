"""Writer (stdin = JSON {'commit':..., 'tree':{path:{sha256,mode}}}).
Clone CFG repo_url into NEW CFG ops_repo (refuse if exists), checkout --detach commit,
verify HEAD, clean porcelain, and per-file sha256 against the locally computed tree. Writes only that dir.
(DEPLOY-01 deploy01_ops_deploy_remote.py)"""
import hashlib,json,subprocess,sys
from pathlib import Path
spec=json.loads(sys.stdin.buffer.read())
D=Path(CFG['server']['ops_repo']);URL=CFG['server']['repo_url']
if D.exists():raise SystemExit('refuse: target exists')
def g(*a,**k):return subprocess.run(['git','-C',str(D),*a],capture_output=True,text=True,check=True,timeout=300,**k).stdout
subprocess.run(['git','clone','-q','--no-checkout',URL,str(D)],check=True,timeout=300)
g('checkout','-q','--detach',spec['commit'])
head=g('rev-parse','HEAD').strip();porc=g('status','--porcelain=v1','--untracked-files=all')
tracked=set(g('ls-files','-z').split('\0'))-{''}
bad=[p for p,v in spec['tree'].items() if hashlib.sha256((D/p).read_bytes()).hexdigest()!=v['sha256']]
print(json.dumps({'dir':str(D),'head':head,'head_ok':head==spec['commit'],'porcelain_clean':porc=='','files_expected':len(spec['tree']),'files_tracked':len(tracked),'tracked_set_equal':tracked==set(spec['tree']),'sha_mismatch':bad}))
