"""Readonly: legacy G prepared manifest CFG server.legacy.g_prepared_manifest (minus binding) and its spec JSON + sha256.
Control metadata only. No writes. (DEPLOY-01 deploy01_oldprep_read.py)"""
import hashlib,json
from pathlib import Path
path=CFG['server'].get('legacy',{}).get('g_prepared_manifest')
if not path:raise SystemExit('config server.legacy.g_prepared_manifest required')
m=Path(path);v=json.loads(m.read_text())
spec=Path(v['spec_path']);sb=spec.read_bytes()
print(json.dumps({'manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest(),'manifest_keys':sorted(v),'spec_path':str(spec),'spec_sha_declared':v['spec_sha256'],'spec_sha_actual':hashlib.sha256(sb).hexdigest(),
 'spec':json.loads(sb),'binding_consumer_keys':sorted(v['binding']['consumer']),'consumer_git':{k:v['binding']['consumer'].get(k) for k in ('git_base','git_dirty','python')},
 'n_source_members':len(v['binding']['consumer'].get('source_members',{}))},indent=1))
