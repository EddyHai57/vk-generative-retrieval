"""Writer (stdin = JSON {'base','target','target_tree','data','overrides'}). Frozen-base overlay for G (R0/D0):
- git worktree add --detach NEW CFG g_deploy <base> (from the CFG ops_repo clone; refuse if exists)
- overlay every path of `git diff --name-only base target` with `git show target:path` (A/M only; D refused)
- write data-channel files (base64 in stdin) and verify each sha256
- verify every overlaid file == locally computed target tree sha256
- write pinned manifest to NEW CFG pinned_dir/g-<target12>.json (outside worktree, N8)
Writes only those two new paths. (DEPLOY-01 deploy01_g_deploy_remote.py)"""
import base64,hashlib,json,subprocess,sys
from pathlib import Path
spec=json.loads(sys.stdin.buffer.read())
S=CFG['server'];R=Path(S['ops_repo']);G=Path(S['g_deploy']);P=Path(S['pinned_dir'])
base,target=spec['base'],spec['target']
if G.exists():raise SystemExit('refuse: G exists')
def git(*a,cwd=R,text=True):return subprocess.run(['git','-C',str(cwd),*a],capture_output=True,text=text,check=True,timeout=300).stdout
git('cat-file','-e',target+'^{commit}');git('worktree','add','--detach',str(G),base)
# -z name-status alternates status/path tokens
toks=[t for t in git('diff','--name-status','-z',base,target).split('\0') if t];pairs=list(zip(toks[0::2],toks[1::2]))
if any(s not in ('A','M') for s,_ in pairs):raise SystemExit('unexpected diff status')
sha=lambda b:hashlib.sha256(b).hexdigest()
bad=[]
for s,p in pairs:
    data=git('show',f'{target}:{p}',text=False);f=G/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data)
    if spec['target_tree'][p]['sha256']!=sha(f.read_bytes()):bad.append(p)
data_bad=[]
for p,v in spec['data'].items():
    b=base64.b64decode(v['b64']);f=G/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
    if sha(f.read_bytes())!=v['sha256']:data_bad.append(p)
# full-tree check: every target tracked file present with target sha (except override path)
tree_bad=[p for p,v in spec['target_tree'].items() if p not in spec['overrides'] and sha((G/p).read_bytes())!=v['sha256']]
files={p:spec['target_tree'][p]['sha256'] for _,p in pairs if p not in spec['overrides']}
for p,v in spec['data'].items():
    if p not in spec['overrides']:files[p]=v['sha256']
pinned={'schema':'vkgr-pinned-v1','base_commit':base,'target_commit':target,'files':dict(sorted(files.items())),'overrides':{p:{'sha256':s,'reason':'D0 frozen G1 input'} for p,s in spec['overrides'].items()}}
P.mkdir(mode=0o755,exist_ok=True);pf=P/f'g-{target[:12]}.json'
with pf.open('x') as f:json.dump(pinned,f,indent=1,sort_keys=True)
porc=git('status','--porcelain=v1','--untracked-files=all',cwd=G)
print(json.dumps({'g_dir':str(G),'head':git('rev-parse','HEAD',cwd=G).strip(),'diff_paths':len(pairs),'overlay_sha_bad':bad,'data_files':len(spec['data']),'data_sha_bad':data_bad,'target_tree_bad':tree_bad,
 'pinned_path':str(pf),'pinned_sha256':sha(pf.read_bytes()),'pinned_files':len(files),'overrides':pinned['overrides'],'porcelain_lines':len([l for l in porc.splitlines() if l])}))
