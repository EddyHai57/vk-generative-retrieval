"""Writer (stdin = JSON {'dir': <path under CFG scope_prefix>, 'files': {name: {b64, sha256}}}):
write each file exclusively (refuse overwrite) into that dir, verify sha256. (DEPLOY-01 deploy01_upload_remote.py)"""
import base64,hashlib,json,sys
from pathlib import Path
spec=json.loads(sys.stdin.buffer.read());D=Path(spec['dir'])
if not str(D).startswith(CFG['server']['scope_prefix']):raise SystemExit('dir scope')
D.mkdir(mode=0o755,parents=True,exist_ok=True);res={}
for name,v in spec['files'].items():
    if '/' in name:raise SystemExit('flat names only')
    f=D/name
    with f.open('xb') as o:o.write(base64.b64decode(v['b64']))
    res[name]=hashlib.sha256(f.read_bytes()).hexdigest()==v['sha256']
print(json.dumps({'dir':str(D),'sha_ok':res}))
