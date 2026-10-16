"""Read-only entry to the fixed deployed combination tool; no queue writes."""
import hashlib
import importlib.util
import json
from pathlib import Path
p=Path('/workspace/vkgr-feat04combo-p-v1/scripts/feat04combo_ops.py')
meta=json.loads(Path('/workspace/vkgr-feat04combo-ops-v1/deployment.json').read_text())
if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['files']['scripts/feat04combo_ops.py']:raise ValueError('deployed tool SHA changed')
s=importlib.util.spec_from_file_location('combo_ops',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps(m.preflight(),sort_keys=True,allow_nan=False))
